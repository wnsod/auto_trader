"""
실시간 학습기 클래스
"""
import os
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
# signal_selector imports
try:
    from signal_selector.core.types import SignalInfo, SignalAction
except ImportError:
    import sys
    _current = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_current)
    sys.path.insert(0, _parent)
    from core.types import SignalInfo, SignalAction


class RealTimeLearner:
    """실시간 학습기 - 즉시 학습 및 적응"""
    def __init__(self):
        self.learning_rate = 0.01
        self.recent_trades = []
        self.pattern_performance = {}
        
    def learn_from_trade(self, signal_pattern: str, trade_result: dict):
        """거래 결과로부터 즉시 학습"""
        try:
            profit = trade_result.get('profit_loss_pct', 0.0)
            success = profit > 0
            
            # 패턴 성과 업데이트
            if signal_pattern not in self.pattern_performance:
                self.pattern_performance[signal_pattern] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'total_profit': 0.0,
                    'success_rate': 0.0
                }
            
            perf = self.pattern_performance[signal_pattern]
            perf['total_trades'] += 1
            perf['total_profit'] += profit
            
            if success:
                perf['successful_trades'] += 1
            
            perf['success_rate'] = perf['successful_trades'] / perf['total_trades']
            
            print(f"🧠 실시간 학습: {signal_pattern} 패턴 성과 업데이트 (성공률: {perf['success_rate']:.2f})")
            
        except Exception as e:
            print(f"⚠️ 실시간 학습 오류: {e}")



