"""
컨텍스트 메모리 클래스
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


class ContextMemory:
    """맥락 메모리 - 시장 상황과 패턴 기억"""
    def __init__(self):
        self.market_contexts = {}
        self.pattern_memories = {}
        self.success_patterns = {}
        self.failure_patterns = {}
        
    def remember_market_context(self, coin: str, interval: str, context: dict):
        """시장 상황 기억"""
        key = f"{coin}_{interval}"
        self.market_contexts[key] = context
        
    def remember_pattern_result(self, pattern: str, success: bool, profit: float):
        """패턴 결과 기억"""
        if success:
            if pattern not in self.success_patterns:
                self.success_patterns[pattern] = []
            self.success_patterns[pattern].append(profit)
        else:
            if pattern not in self.failure_patterns:
                self.failure_patterns[pattern] = []
            self.failure_patterns[pattern].append(profit)



