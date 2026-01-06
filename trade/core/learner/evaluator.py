#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매매 사후 평가 시스템 (Advanced)
MFE(최대 수익폭)/MAE(최대 손실폭) 및 매도 품질 정밀 진단
"""

import time
import sqlite3
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from trade.core.models import SignalInfo, VirtualPosition

class PostTradeEvaluator:
    """매매 사후 평가기 - 매도 후 흐름을 추적하여 '최적 타이밍'이었는지 학습 데이터 생성"""
    
    def __init__(self, strategy_db_path: str):
        self.strategy_db_path = strategy_db_path
        self.tracked_trades = {}  # trade_id: {data}
        self.tracking_duration = 24 * 3600  # 24시간 추적
        self.pending_feedback = [] # 학습기에 전달할 피드백 큐

    def add_trade(self, trade_data: dict):
        """매도/손절 발생 시 추적 시작"""
        try:
            trade_id = f"{trade_data['coin']}_{trade_data['entry_timestamp']}"
            self.tracked_trades[trade_id] = {
                'coin': trade_data['coin'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data['exit_price'],
                'exit_timestamp': trade_data['exit_timestamp'],
                'profit_loss_pct': trade_data['profit_loss_pct'],
                'max_profit_pct': trade_data.get('max_profit_pct', 0.0),
                'signal_pattern': trade_data.get('signal_pattern', 'unknown'),
                'trend_type': trade_data.get('trend_type', 'unknown'),
                'highest_after': trade_data['exit_price'],
                'lowest_after': trade_data['exit_price'],
                'mfe': 0.0,
                'mae': 0.0
            }
        except Exception as e:
            print(f"⚠️ 평가 추적 추가 오류: {e}")

    def check_evaluations(self, current_prices: dict):
        """실시간 가격을 받아 추적 중인 거래들 업데이트"""
        current_time = int(time.time())
        completed = []

        for tid, data in list(self.tracked_trades.items()):
            # 1. 추적 기간 만료 체크
            if current_time - data['exit_timestamp'] > self.tracking_duration:
                self._finalize_evaluation(tid)
                completed.append(tid)
                continue

            # 2. 고점/저점 갱신
            cp = current_prices.get(data['coin'])
            if cp:
                data['highest_after'] = max(data['highest_after'], cp)
                data['lowest_after'] = min(data['lowest_after'], cp)
                
                # MFE/MAE 업데이트 (%)
                data['mfe'] = ((data['highest_after'] - data['exit_price']) / data['exit_price']) * 100
                data['mae'] = ((data['lowest_after'] - data['exit_price']) / data['exit_price']) * 100

        return completed

    def _finalize_evaluation(self, trade_id: str):
        """추적 종료 후 매도 품질 최종 평가"""
        data = self.tracked_trades.get(trade_id)
        if not data: return

        mfe = data['mfe']
        mae = data['mae']
        
        feedback = {
            'coin': data['coin'],
            'signal_pattern': data['signal_pattern'],
            'profit_loss_pct': data['profit_loss_pct'],
            'mfe': mfe,
            'mae': mae,
            'is_panic_sell': False,
            'is_perfect_exit': False,
            'adjustment_weight': 0.0
        }

        # 1. 패닉 셀 감지 (팔고 나서 폭등)
        if mfe > 5.0 and data['profit_loss_pct'] < 0:
            print(f"📉 [정밀매도평가] {data['coin']}: 패닉 셀 확정! 매도 후 +{mfe:.1f}% 폭등. 성격 교정 필요.")
            feedback['is_panic_sell'] = True
            feedback['adjustment_weight'] = -0.2 # 매도 기준을 더 높이도록 유도

        # 2. 신의 매도 감지 (팔자마자 폭락)
        elif mae < -5.0 and mfe < 1.0:
            print(f"🎯 [정밀매도평가] {data['coin']}: 완벽한 고점 매도! 매도 후 {mae:.1f}% 급락. 이 패턴 신뢰도 상승.")
            feedback['is_perfect_exit'] = True
            feedback['adjustment_weight'] = 0.2

        self.pending_feedback.append(feedback)
        del self.tracked_trades[trade_id]

    def get_pending_feedback(self) -> List[dict]:
        feedback_copy = self.pending_feedback[:]
        self.pending_feedback = []
        return feedback_copy
