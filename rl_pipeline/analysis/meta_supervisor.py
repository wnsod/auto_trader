"""
Meta-Cognitive Supervisor (메타 인지 감독관)
전략 그룹별 실전 성과를 모니터링하고 동적 가중치를 조정
시장 상황(Regime)과 실제 성과(Performance) 간의 괴리를 보정
"""

import logging
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MetaCognitiveSupervisor:
    """
    메타 인지 감독관
    
    기능:
    1. 전략 그룹(Trend, Reversion 등)별 최근 성과 모니터링
    2. 지표(Regime)와 성과(Performance) 괴리 탐지 (예: 상승장인데 Trend 전략 연패)
    3. 통합 분석 가중치 동적 보정 (Correction Factor)
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        # 그룹별 성과 추적 (최근 N회)
        self.performance_window = 20
        
    def analyze_performance_discrepancy(self, coin: str, interval: str, current_regime: str) -> Dict[str, float]:
        """
        시장 상황(Regime)과 실제 성과 간의 괴리 분석
        Returns: 그룹별 보정 가중치 (1.0 = 정상, < 1.0 = 페널티, > 1.0 = 부스트)
        """
        correction_factors = {
            'trend': 1.0,
            'mean_reversion': 1.0,
            'scalping': 1.0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 최근 매매 결과 조회 (전략 타입별)
                # rl_episode_summary와 strategies 테이블 조인
                query = """
                    SELECT 
                        s.strategy_type,
                        e.realized_ret_signed,
                        e.acc_flag
                    FROM rl_episode_summary e
                    JOIN strategies s ON e.strategy_id = s.id
                    WHERE s.symbol = ? AND s.interval = ?
                    ORDER BY e.ts_exit DESC
                    LIMIT 100
                """
                cursor.execute(query, (coin, interval))
                rows = cursor.fetchall()
                
                if not rows:
                    return correction_factors

                # 그룹별 성과 집계
                group_stats = {'trend': [], 'mean_reversion': [], 'scalping': []}
                
                for r in rows:
                    stype = r[0].lower() if r[0] else ''
                    profit = r[1] or 0.0
                    win = 1 if r[2] else 0
                    
                    # 전략 타입 매핑
                    group = 'trend' # 기본값
                    if 'reversion' in stype or 'range' in stype:
                        group = 'mean_reversion'
                    elif 'short' in stype or 'scalp' in stype:
                        group = 'scalping'
                    elif 'trend' in stype or 'momentum' in stype:
                        group = 'trend'
                        
                    group_stats[group].append({'profit': profit, 'win': win})
                
                # 괴리 분석 및 보정 계수 산출
                for group, stats in group_stats.items():
                    if not stats:
                        continue
                        
                    recent_stats = stats[:self.performance_window]
                    avg_profit = np.mean([s['profit'] for s in recent_stats])
                    win_rate = np.mean([s['win'] for s in recent_stats])
                    
                    # 로직 1: 상승장(Trend 유리)인데 Trend 전략이 손실 중이면 -> 가짜 상승장(Bull Trap) 의심
                    if current_regime in ['bullish', 'extreme_bullish'] and group == 'trend':
                        if avg_profit < 0 or win_rate < 0.4:
                            logger.info(f"🧠 [메타인지] {coin}-{interval}: 상승장이지만 Trend 전략 부진 (승률 {win_rate:.2f}). Bull Trap 의심 -> Trend 비중 축소")
                            correction_factors['trend'] *= 0.5  # 비중 반토막
                            correction_factors['mean_reversion'] *= 1.5  # 역추세 비중 확대
                            
                    # 로직 2: 횡보장(Range 유리)인데 Reversion 전략이 손실 중이면 -> 추세 이탈(Breakout) 의심
                    elif current_regime in ['sideways', 'neutral'] and group == 'mean_reversion':
                        if avg_profit < 0 or win_rate < 0.4:
                            logger.info(f"🧠 [메타인지] {coin}-{interval}: 횡보장이지만 Reversion 전략 부진. Breakout 의심 -> Reversion 비중 축소")
                            correction_factors['mean_reversion'] *= 0.5
                            correction_factors['trend'] *= 1.3
                            
                return correction_factors
                
        except Exception as e:
            logger.warning(f"⚠️ 메타 인지 분석 실패 ({coin}-{interval}): {e}")
            return correction_factors

