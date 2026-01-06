#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전이학습 시스템
모든 코인의 동일 패턴 데이터를 통합 분석하여 학습 결과를 공유 및 전이
"""

import sqlite3
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple

# 🆕 중앙 DB 연결 유틸리티 임포트
try:
    from trade.core.database import get_db_connection
except ImportError:
    def get_db_connection(path, read_only=False):
        return sqlite3.connect(path, timeout=30.0)

class TransferLearner:
    """
    전이학습 시스템 - 모든 코인의 동일 패턴 데이터를 통합하여 학습
    
    목표: 최고 수익률만 추구하지 않고, 다양한 매수/매도를 통한 점진적 수익률
    """
    
    def __init__(self, strategy_db_path: str, trading_db_path: str, thompson_sampler):
        self.strategy_db_path = strategy_db_path
        self.trading_db_path = trading_db_path
        self.thompson_sampler = thompson_sampler
        self.min_trades_for_transfer = 10
        self.min_coins_for_transfer = 2
        
        self.last_transfer_time = 0
        self.transfer_interval = 6 * 3600
        
        print("🔄 전이학습 시스템 초기화 완료")
    
    def collect_pattern_data(self, signal_pattern: str, 
                           volatility_regime: str = None,
                           volume_regime: str = None,
                           market_regime: str = None) -> Dict:
        """패턴 및 시장 조건별 데이터 수집 (읽기 전용 안정성 강화)"""
        try:
            # 🚀 읽기 전용 모드로 조회 (잠금 방지)
            with get_db_connection(self.trading_db_path, read_only=True) as conn:
                conditions = ["signal_pattern = ?"]
                params = [signal_pattern]
                
                if volatility_regime:
                    conditions.append("volatility_regime = ?")
                    params.append(volatility_regime)
                if volume_regime:
                    conditions.append("volume_regime = ?")
                    params.append(volume_regime)
                if market_regime:
                    conditions.append("market_regime = ?")
                    params.append(market_regime)
                
                where_clause = " AND ".join(conditions)
                
                query = f"""
                    SELECT coin, profit_loss_pct, holding_duration, entry_timestamp, exit_timestamp
                    FROM virtual_trade_history
                    WHERE {where_clause}
                    ORDER BY exit_timestamp DESC
                """
                df = pd.read_sql(query, conn, params=tuple(params))
            
            if df.empty or len(df) < self.min_trades_for_transfer:
                return None
            
            total_trades = len(df)
            avg_profit = df['profit_loss_pct'].mean()
            success_rate = len(df[df['profit_loss_pct'] > 0]) / total_trades
            
            return {
                'signal_pattern': signal_pattern,
                'total_trades': total_trades,
                'avg_profit': avg_profit,
                'success_rate': success_rate,
                'all_profits': df['profit_loss_pct'].tolist()
            }
        except Exception as e:
            print(f"⚠️ 패턴 데이터 수집 오류: {e}")
            return None

    def calculate_transfer_score(self, pattern_data: Dict) -> float:
        """전이 점수 계산"""
        if not pattern_data: return 0.0
        return pattern_data['success_rate'] * 0.6 + min(pattern_data['avg_profit'] / 5.0, 1.0) * 0.4

    def transfer_learning(self, signal_pattern: str) -> bool:
        """개별 패턴에 대한 전이 학습 실행"""
        try:
            # 1. 해당 패턴의 모든 코인 통합 데이터 수집
            pattern_data = self.collect_pattern_data(signal_pattern)
            if not pattern_data:
                return False
                
            # 2. 전이 점수(성능) 계산
            transfer_score = self.calculate_transfer_score(pattern_data)
            
            # 3. Thompson Sampling 학습기에 전이된 지식 반영
            # 성공률과 평균 수익률을 통합 데이터 기반으로 업데이트
            if self.thompson_sampler:
                self.thompson_sampler.update_distribution(
                    pattern=signal_pattern,
                    success=(pattern_data['success_rate'] > 0.5),
                    profit_pct=pattern_data['avg_profit'],
                    weight=0.3  # 전이된 지식은 30%의 가중치만 부여 (개별 코인 특성 존중)
                )
                
            return True
        except Exception as e:
            print(f"⚠️ 개별 패턴 전이 실패 ({signal_pattern}): {e}")
            return False

    def execute_transfer_learning(self):
        """전체 패턴에 대해 전이 학습 실행 (읽기 전용 안정성 강화)"""
        try:
            current_time = time.time()
            if current_time - self.last_transfer_time < self.transfer_interval:
                return False
                
            print("🔄 [전이학습] 모든 코인의 통합 패턴 분석 중...")
            
            # 1. 존재하는 모든 패턴 조회 (읽기 전용 모드)
            with get_db_connection(self.trading_db_path, read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT signal_pattern FROM virtual_trade_history")
                patterns = [row[0] for row in cursor.fetchall()]
            
            # 2. 각 패턴별 데이터 수집 및 전이
            for pattern in patterns:
                self.transfer_learning(pattern)
                
            self.last_transfer_time = current_time
            print(f"✅ [전이학습] {len(patterns)}개 패턴에 대한 통합 학습 완료")
            return True
            
        except Exception as e:
            print(f"🚨 전이 학습 실행 오류: {e}")
            return False

