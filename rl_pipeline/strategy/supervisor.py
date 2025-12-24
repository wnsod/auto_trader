"""
메타 인지 감독관 (Strategy Supervisor)
전략들의 실시간 성과를 감시하여 시장의 '진짜 상태'를 파악하고,
통합 분석 시 전략 가중치를 동적으로 보정하는 메타 학습 모듈.
"""

import logging
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.core.env import config

logger = logging.getLogger(__name__)

class StrategySupervisor:
    """전략 메타 인지 및 감독 시스템"""
    
    def __init__(self, coin: str):
        self.coin = coin
        self.db_path = config.get_strategy_db_path(coin)
        
        # 전략 타입 그룹 정의
        self.strategy_groups = {
            'trend': ['trend', 'aggressive', 'breakout'],
            'mean_reversion': ['mean_reversion', 'range', 'balanced'],
            'defensive': ['conservative', 'defensive'],
            'short_term': ['short_term', 'scalping']
        }
        
    def get_market_meta_state(self, interval: str) -> Dict[str, Any]:
        """
        시장 메타 상태 분석
        지표가 아닌 '전략들의 성과'를 기반으로 시장 상태를 역추적
        """
        # 최근 성과 조회
        recent_performance = self._get_recent_performance_by_group(interval)
        
        meta_state = {
            'dominant_group': None,      # 현재 가장 잘 통하는 그룹
            'struggling_group': None,    # 현재 가장 힘못쓰는 그룹
            'market_phase': 'uncertain', # bull_trap, bear_trap, real_trend, choppy
            'confidence': 0.0,
            'group_scores': recent_performance
        }
        
        if not recent_performance:
            return meta_state
            
        # 1. 지배적 그룹 찾기
        sorted_groups = sorted(recent_performance.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        best_group, best_stats = sorted_groups[0]
        worst_group, worst_stats = sorted_groups[-1]
        
        meta_state['dominant_group'] = best_group
        meta_state['struggling_group'] = worst_group
        
        # 2. 메타 상태 추론
        # 예: 추세 전략이 죽고 역추세가 산다 -> 횡보장 (Choppy)
        # 예: 다 죽고 방어형만 산다 -> 하락장 또는 불확실성 (Bear/Uncertain)
        trend_wr = recent_performance.get('trend', {}).get('win_rate', 0.0)
        reversion_wr = recent_performance.get('mean_reversion', {}).get('win_rate', 0.0)
        
        if trend_wr > 0.6 and reversion_wr < 0.4:
            meta_state['market_phase'] = 'strong_trend'
            meta_state['confidence'] = trend_wr
        elif trend_wr < 0.4 and reversion_wr > 0.6:
            meta_state['market_phase'] = 'range_bound'
            meta_state['confidence'] = reversion_wr
        elif trend_wr < 0.3 and reversion_wr < 0.3:
            meta_state['market_phase'] = 'chaos' # 모두가 죽는 장 (False Breakout 다발)
            meta_state['confidence'] = 0.8
        else:
            meta_state['market_phase'] = 'mixed'
            meta_state['confidence'] = 0.5
            
        logger.debug(f"🕵️ Supervisor 메타 진단 ({self.coin}-{interval}): {meta_state['market_phase']} "
                     f"(Trend WR: {trend_wr:.2f}, Rev WR: {reversion_wr:.2f})")
                     
        return meta_state

    def get_correction_factor(self, strategy_type: str, meta_state: Dict[str, Any]) -> float:
        """
        전략 타입별 가중치 보정 계수 반환
        메타 상태에 따라 특정 전략을 강화하거나 억제
        """
        if not meta_state or not meta_state.get('group_scores'):
            return 1.0
            
        # 해당 전략이 속한 그룹 찾기
        my_group = 'unknown'
        for group, types in self.strategy_groups.items():
            if any(t in str(strategy_type).lower() for t in types):
                my_group = group
                break
        
        # 1. 성과 기반 직접 보정 (잘하는 놈 밀어주기)
        group_stats = meta_state['group_scores'].get(my_group)
        if group_stats:
            win_rate = group_stats['win_rate']
            # 승률 50%를 기준으로 가감 (0.0 ~ 2.0 범위)
            # 70% 승률 -> 1.4배, 30% 승률 -> 0.6배
            performance_factor = max(0.2, min(2.0, win_rate * 2.0))
        else:
            performance_factor = 1.0
            
        # 2. 메타 상태 기반 전략적 보정 (감독의 개입)
        strategic_factor = 1.0
        phase = meta_state.get('market_phase')
        
        if phase == 'chaos':
            # 혼란장에서는 방어형 전략 우대, 나머지는 페널티
            if my_group == 'defensive':
                strategic_factor = 1.5
            else:
                strategic_factor = 0.5
        elif phase == 'strong_trend':
            if my_group == 'trend':
                strategic_factor = 1.2
            elif my_group == 'mean_reversion':
                strategic_factor = 0.8
        
        # 최종 보정 계수
        return performance_factor * strategic_factor

    def _get_recent_performance_by_group(self, interval: str, lookback: int = 50) -> Dict[str, Dict]:
        """최근 거래 성과를 그룹별로 집계"""
        try:
            performance = {}
            
            with get_optimized_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 최근 N개 에피소드(가상 거래) 조회
                # rl_episode_summary 테이블과 strategies 테이블 조인
                query = """
                    SELECT 
                        s.strategy_type,
                        AVG(CASE WHEN es.acc_flag = 1 THEN 1.0 ELSE 0.0 END) as win_rate,
                        AVG(es.realized_ret_signed) as avg_return,
                        COUNT(*) as count
                    FROM rl_episode_summary es
                    JOIN rl_episodes e ON es.episode_id = e.episode_id
                    JOIN strategies s ON e.strategy_id = s.id
                    WHERE e.symbol = ? AND e.interval = ?
                    GROUP BY s.strategy_type
                    ORDER BY e.ts_entry DESC
                    LIMIT ?
                """
                # LIMIT는 그룹핑 전 전체 개수 제어가 안되므로, 서브쿼리나 시간 기준으로 해야 정확하나
                # 여기서는 단순화를 위해 전체 집계 후 Python에서 그룹핑
                
                # 시간 기준 조회 (최근 2일)
                query_time = """
                    SELECT 
                        s.strategy_type,
                        es.acc_flag,
                        es.realized_ret_signed
                    FROM rl_episode_summary es
                    JOIN rl_episodes e ON es.episode_id = e.episode_id
                    JOIN strategies s ON e.strategy_id = s.id
                    WHERE e.symbol = ? AND e.interval = ?
                      AND e.ts_entry > datetime('now', '-2 days')
                """
                
                cursor.execute(query_time, (self.coin, interval))
                rows = cursor.fetchall()
                
                # 그룹별 집계
                group_data = {g: {'wins': 0, 'total': 0, 'returns': []} for g in self.strategy_groups}
                group_data['unknown'] = {'wins': 0, 'total': 0, 'returns': []}
                
                for stype, acc, ret in rows:
                    stype = str(stype).lower()
                    matched = False
                    for group, types in self.strategy_groups.items():
                        if any(t in stype for t in types):
                            group_data[group]['wins'] += 1 if acc else 0
                            group_data[group]['total'] += 1
                            group_data[group]['returns'].append(ret or 0.0)
                            matched = True
                            break
                    if not matched:
                        group_data['unknown']['wins'] += 1 if acc else 0
                        group_data['unknown']['total'] += 1
                        group_data['unknown']['returns'].append(ret or 0.0)
                
                # 최종 통계 계산
                for group, data in group_data.items():
                    if data['total'] > 0:
                        performance[group] = {
                            'win_rate': data['wins'] / data['total'],
                            'avg_return': sum(data['returns']) / data['total'],
                            'count': data['total']
                        }
                        
            return performance
            
        except Exception as e:
            logger.warning(f"⚠️ 최근 성과 집계 실패 ({self.coin}-{interval}): {e}")
            return {}

