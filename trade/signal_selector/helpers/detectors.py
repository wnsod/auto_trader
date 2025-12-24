"""
시장 레짐 감지 클래스
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


class RegimeChangeDetector:
    """레짐 전환 감지기"""
    def __init__(self):
        self.regime_history = []
        self.current_regime = 'unknown'
        self.regime_threshold = 0.3
        
    def detect_regime_change(self, market_indicators: Dict[str, float]) -> str:
        """레짐 전환 감지"""
        try:
            # 현재 레짐 결정
            new_regime = self._determine_regime(market_indicators)
            
            # 레짐 변화 감지
            if new_regime != self.current_regime:
                self.regime_history.append({
                    'timestamp': time.time(),
                    'old_regime': self.current_regime,
                    'new_regime': new_regime,
                    'indicators': market_indicators
                })
                self.current_regime = new_regime
                return 'changed'
            
            return 'stable'
            
        except Exception as e:
            print(f"⚠️ 레짐 전환 감지 오류: {e}")
            return 'unknown'
    
    def _determine_regime(self, indicators: Dict[str, float]) -> str:
        """레짐 결정"""
        try:
            adx = indicators.get('adx', 25.0)
            atr = indicators.get('atr', 0.0)
            ma_slope = indicators.get('ma_slope', 0.0)
            
            # 추세 강도 기반 레짐 분류
            if adx > 30 and abs(ma_slope) > 0.01:
                return 'trending'
            elif adx < 20 and atr < 0.02:
                return 'sideways_low_vol'
            elif adx < 20 and atr > 0.05:
                return 'sideways_high_vol'
            else:
                return 'transitional'
                
        except Exception as e:
            print(f"⚠️ 레짐 결정 오류: {e}")
            return 'unknown'

# 🆕 성능 업그레이드 시스템 클래스들


