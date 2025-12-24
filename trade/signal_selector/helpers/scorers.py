"""
액션별 점수 계산 클래스
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


class ActionSpecificScorer:
    """액션별 스코어 계산기"""
    def __init__(self):
        self.action_scores = {
            'buy': {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0},
            'sell': {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0},
            'hold': {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        }
    
    def update_action_score(self, action: str, success: bool, profit: float):
        """액션별 성과 업데이트"""
        if action in self.action_scores:
            self.action_scores[action]['total_trades'] += 1
            if success:
                self.action_scores[action]['success_rate'] += 1
            self.action_scores[action]['avg_profit'] += profit
    
    def get_action_score(self, action: str) -> float:
        """액션별 점수 반환"""
        if action not in self.action_scores:
            return 0.0
        
        score_data = self.action_scores[action]
        if score_data['total_trades'] == 0:
            return 0.0
        
        success_rate = score_data['success_rate'] / score_data['total_trades']
        avg_profit = score_data['avg_profit'] / score_data['total_trades']
        
        return success_rate * avg_profit



