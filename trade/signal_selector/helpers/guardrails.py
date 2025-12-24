"""
이상치 가드레일 클래스
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


class OutlierGuardrail:
    """이상치 컷 시스템"""
    def __init__(self, percentile_cut: float = 0.05):
        self.percentile_cut = percentile_cut
    
    def winsorize_profits(self, profits: List[float]) -> List[float]:
        """수익률 Winsorizing"""
        if len(profits) < 10:  # 데이터가 적으면 그대로 반환
            return profits
        
        sorted_profits = sorted(profits)
        n = len(sorted_profits)
        
        # 상하위 5% 절단
        lower_cut = int(n * self.percentile_cut)
        upper_cut = int(n * (1 - self.percentile_cut))
        
        # 절단된 값으로 대체
        winsorized = []
        for profit in profits:
            if profit < sorted_profits[lower_cut]:
                winsorized.append(sorted_profits[lower_cut])
            elif profit > sorted_profits[upper_cut]:
                winsorized.append(sorted_profits[upper_cut])
            else:
                winsorized.append(profit)
        
        return winsorized
    
    def calculate_robust_avg_profit(self, profits: List[float]) -> float:
        """견고한 평균 수익률 계산"""
        winsorized_profits = self.winsorize_profits(profits)
        return sum(winsorized_profits) / len(winsorized_profits)

# 🆕 진화형 AI 시스템 클래스들


