"""
가중치 계산 클래스
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


class ExponentialDecayWeight:
    """최근성 가중치 계산기"""
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
    
    def calculate_weight(self, time_diff_hours: float) -> float:
        """시간 차이에 따른 가중치 계산"""
        return math.exp(-self.decay_rate * time_diff_hours)



class BayesianSmoothing:
    """베이지안 스무딩 시스템"""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, kappa: float = 1.0):
        self.alpha = alpha  # Beta 분포 파라미터
        self.beta = beta    # Beta 분포 파라미터
        self.kappa = kappa  # 정규 분포 파라미터
    
    def smooth_success_rate(self, wins: int, total_trades: int) -> float:
        """승률 베이지안 스무딩"""
        return (wins + self.alpha) / (total_trades + self.alpha + self.beta)
    
    def smooth_avg_profit(self, profits: List[float], global_avg: float) -> float:
        """평균 수익률 베이지안 스무딩"""
        if not profits:
            return global_avg
        
        weighted_sum = sum(profits) + self.kappa * global_avg
        total_weight = len(profits) + self.kappa
        
        return weighted_sum / total_weight



