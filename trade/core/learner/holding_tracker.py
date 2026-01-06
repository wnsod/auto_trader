#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
보유 포지션 추적 학습기 - 매수 이후부터 매도까지의 중간 상태 학습
"""

import time
import pandas as pd
from trade.core.database import get_db_connection, TRADING_SYSTEM_DB_PATH

# 헬퍼 함수
def safe_float(value, default: float = 0.0) -> float:
    """안전한 float 변환"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

class HoldingPositionTracker:
    """보유 포지션 중간 추적 학습기"""
    
    def __init__(self, learner):
        self.learner = learner
        self.batch_size = 500

    def learn_holding_positions(self, conn) -> int:
        """보유 중인 포지션의 중간 추적 내용 학습"""
        holding_count = 0
        try:
            # virtual_trade_history에서 보유 중인 포지션 조회
            query = """
                SELECT coin, entry_price, entry_timestamp, entry_signal_score, 
                       profit_loss_pct, holding_duration, action, signal_pattern
                FROM virtual_trade_history
                WHERE (exit_timestamp = 0 OR exit_timestamp IS NULL)
                  AND entry_timestamp > ?
                ORDER BY entry_timestamp DESC
                LIMIT ?
            """
            current_time = int(time.time())
            lookback_timestamp = current_time - (24 * 3600)  # 최근 24시간
            
            df = pd.read_sql(query, conn, params=(lookback_timestamp, self.batch_size))
            
            if df.empty:
                return 0
            
            print(f"📊 보유 중인 포지션 {len(df)}건 발견, 중간 추적 학습 시작...")
            
            for _, row in df.iterrows():
                coin = row['coin']
                entry_timestamp = row['entry_timestamp']
                profit_loss_pct = safe_float(row['profit_loss_pct'], 0.0)
                holding_duration = safe_float(row['holding_duration'], 0.0)
                signal_pattern = row.get('signal_pattern', 'unknown')
                
                # 이미 학습한 포지션 제외 (메인 학습기의 processed_trade_ids 활용)
                holding_id = f"{coin}_{entry_timestamp}_holding"
                if holding_id in self.learner.processed_trade_ids:
                    continue
                
                # 수익/손실 상태 학습
                if profit_loss_pct > 0:
                    holding_pattern = f"{signal_pattern}_holding_profit"
                    self.learner.thompson_sampler.update_distribution(
                        pattern=holding_pattern, success=True, profit_pct=profit_loss_pct, weight=0.5
                    )
                elif profit_loss_pct < 0:
                    holding_pattern = f"{signal_pattern}_holding_loss"
                    self.learner.thompson_sampler.update_distribution(
                        pattern=holding_pattern, success=False, profit_pct=profit_loss_pct, weight=0.5
                    )
                
                # 보유 효율성 학습
                if holding_duration > 0:
                    holding_hours = holding_duration / 3600.0
                    if holding_hours >= 24 and profit_loss_pct < 5.0:
                        efficiency_pattern = f"{signal_pattern}_low_efficiency"
                        self.learner.thompson_sampler.update_distribution(
                            pattern=efficiency_pattern, success=False, profit_pct=profit_loss_pct, weight=0.3
                        )
                
                self.learner.processed_trade_ids.add(holding_id)
                holding_count += 1
            
            if holding_count > 0:
                print(f"✅ 보유 중인 포지션 {holding_count}건의 중간 추적 학습 완료")
            
            return holding_count
        except Exception as e:
            print(f"⚠️ 보유 포지션 학습 오류: {e}")
            return 0

