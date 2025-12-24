"""
컨텍스트 특징 추출 클래스
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


class ContextFeatureExtractor:
    """컨텍스트 특징 추출기"""
    def __init__(self):
        self.context_bins = {
            'volatility': ['low', 'medium', 'high'],
            'volume_ratio': ['low', 'medium', 'high'],
            'market_trend': ['bullish', 'bearish', 'sideways'],
            'btc_dominance': ['low', 'medium', 'high']
        }
    
    def extract_context_features(self, candle: pd.Series, market_data: dict) -> dict:
        """컨텍스트 특징 추출"""
        context = {}
        
        # 변동성 구간화
        volatility = candle.get('volatility', 0.0)
        if volatility < 0.02:
            context['volatility'] = 'low'
        elif volatility < 0.05:
            context['volatility'] = 'medium'
        else:
            context['volatility'] = 'high'
        
        # 거래량 비율 구간화
        volume_ratio = candle.get('volume_ratio', 1.0)
        if volume_ratio < 0.8:
            context['volume_ratio'] = 'low'
        elif volume_ratio < 1.2:
            context['volume_ratio'] = 'medium'
        else:
            context['volume_ratio'] = 'high'
        
        # 시장 트렌드 구간화
        market_trend = market_data.get('trend', 'sideways')
        context['market_trend'] = market_trend
        
        return context
    
    def get_context_key(self, context: dict) -> str:
        """컨텍스트 키 생성"""
        return f"{context['volatility']}_{context['volume_ratio']}_{context['market_trend']}"



