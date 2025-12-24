"""
진화 엔진 클래스
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


class EvolutionEngine:
    """진화형 AI 엔진 - 시그널 진화 및 적응"""
    def __init__(self):
        self.pattern_weights = {}
        self.market_adaptations = {}
        self.evolution_history = []
        
    def evolve_signal(self, base_signal: SignalInfo, coin: str, interval: str) -> SignalInfo:
        """시그널을 진화시켜 더 정확한 시그널 생성"""
        try:
            # 패턴 기반 가중치 적용
            pattern_weight = self._get_pattern_weight(base_signal, coin, interval)
            
            # 시장 적응 가중치 적용
            market_weight = self._get_market_adaptation_weight(coin, interval)
            
            # 진화된 시그널 점수 계산
            evolved_score = base_signal.signal_score * pattern_weight * market_weight
            
            # 진화된 시그널 생성
            evolved_signal = SignalInfo(
                coin=base_signal.coin,
                interval=base_signal.interval,
                action=base_signal.action,
                signal_score=evolved_score,
                confidence=base_signal.confidence * pattern_weight,
                reason=f"{base_signal.reason} + 진화적적응",
                timestamp=base_signal.timestamp,
                price=base_signal.price,
                volume=base_signal.volume,
                rsi=base_signal.rsi,
                macd=base_signal.macd,
                wave_phase=base_signal.wave_phase,
                pattern_type=base_signal.pattern_type,
                risk_level=base_signal.risk_level,
                volatility=base_signal.volatility,
                volume_ratio=base_signal.volume_ratio,
                wave_progress=base_signal.wave_progress,
                structure_score=base_signal.structure_score,
                pattern_confidence=base_signal.pattern_confidence,
                integrated_direction=base_signal.integrated_direction,
                integrated_strength=base_signal.integrated_strength
            )
            
            return evolved_signal
            
        except Exception as e:
            print(f"⚠️ 시그널 진화 오류: {e}")
            # 🆕 진화형 AI 시그널 진화 (candle 변수 없이 진행)
            evolved_signal = base_signal  # 기본 시그널 그대로 사용
            
            # 🆕 시그널 패턴 추출 및 저장
            signal_pattern = self._extract_signal_pattern(evolved_signal)
            market_context = self._get_market_context(coin, interval)
            
            # 🆕 학습 데이터 저장
            self._save_signal_for_learning(evolved_signal, signal_pattern, market_context)
            
            print(f"🧬 진화형 시그널 생성: {coin}-{interval} (패턴: {signal_pattern})")
            
            return evolved_signal
    
    def _get_pattern_weight(self, signal: SignalInfo, coin: str, interval: str) -> float:
        """패턴 기반 가중치 계산"""
        try:
            pattern_key = f"{coin}_{interval}_{signal.pattern_type}"
            if pattern_key in self.pattern_weights:
                return self.pattern_weights[pattern_key]
            return 1.0  # 기본값
        except:
            return 1.0
    
    def _get_market_adaptation_weight(self, coin: str, interval: str) -> float:
        """시장 적응 가중치 계산"""
        try:
            market_key = f"{coin}_{interval}"
            if market_key in self.market_adaptations:
                return self.market_adaptations[market_key]
            return 1.0  # 기본값
        except:
            return 1.0



