"""
기술지표 처리 모듈
"""

from typing import Dict, Any
import pandas as pd
from .helpers import safe_float, safe_str


# 기술지표 설정 (모든 지표의 기본값과 처리 로직 통합)
TECHNICAL_INDICATORS_CONFIG = {
    # 기본 지표들
    'rsi': {'default': 50.0, 'type': 'float'},
    'macd': {'default': 0.0, 'type': 'float'},
    'volume_ratio': {'default': 1.0, 'type': 'float'},
    'wave_progress': {'default': 0.5, 'type': 'float'},
    'structure_score': {'default': 0.5, 'type': 'float'},
    'pattern_confidence': {'default': 0.0, 'type': 'float'},
    
    # 고급 지표들
    'mfi': {'default': 50.0, 'type': 'float'},
    'atr': {'default': 0.0, 'type': 'float'},
    'adx': {'default': 25.0, 'type': 'float'},
    'ma20': {'default': 0.0, 'type': 'float'},
    'ma20_pct_diff': {'default': 0.5, 'type': 'float'},
    'rsi_ema': {'default': 50.0, 'type': 'float'},
    'rsi_smoothed': {'default': 50.0, 'type': 'float'},
    'macd_signal': {'default': 0.0, 'type': 'float'},
    'macd_diff': {'default': 0.0, 'type': 'float'},
    'macd_smoothed': {'default': 0.0, 'type': 'float'},
    'wave_momentum': {'default': 0.0, 'type': 'float'},
    'confidence': {'default': 0.5, 'type': 'float'},
    'volatility': {'default': 0.0, 'type': 'float'},
    'risk_score': {'default': 0.5, 'type': 'float'},
    'integrated_strength': {'default': 0.5, 'type': 'float'},
    'pattern_quality': {'default': 0.0, 'type': 'float'},
    
    # 볼린저 밴드 관련
    'bb_upper': {'default': 0.0, 'type': 'float'},
    'bb_lower': {'default': 0.0, 'type': 'float'},
    'bb_middle': {'default': 0.0, 'type': 'float'},
    'bb_bandwidth': {'default': 0.0, 'type': 'float'},
    
    # 텍스트 지표들
    'pattern_type': {'default': 'unknown', 'type': 'str'},
    'pattern_class': {'default': 'unknown', 'type': 'str'},
    'flow_level_meta': {'default': 'unknown', 'type': 'str'},
    'volatility_level': {'default': 'unknown', 'type': 'str'},
    'wave_phase': {'default': 'unknown', 'type': 'str'},
    'pattern_direction': {'default': 'unknown', 'type': 'str'},
    'pattern_volume_ratio': {'default': 'unknown', 'type': 'str'},
    'pattern_pivot_strength': {'default': 'unknown', 'type': 'str'},
    'volume_avg': {'default': 'unknown', 'type': 'str'},
    'volume_normalized': {'default': 'unknown', 'type': 'str'},
    'zigzag': {'default': 'unknown', 'type': 'str'},
    'zigzag_direction': {'default': 'unknown', 'type': 'str'},
    'pivot_point': {'default': 'unknown', 'type': 'str'},
    'wave_number': {'default': 'unknown', 'type': 'str'},
    'wave_step': {'default': 'unknown', 'type': 'str'},
    'integrated_wave_phase': {'default': 'unknown', 'type': 'str'},
    'integrated_direction': {'default': 'unknown', 'type': 'str'},
    'three_wave_pattern': {'default': 'unknown', 'type': 'str'},
    'sideways_pattern': {'default': 'unknown', 'type': 'str'},
}

# 상태 이산화 설정
STATE_DISCRETIZATION_CONFIG = {
    'rsi': {'low': 30, 'high': 70, 'states': ['low', 'mid', 'high']},
    'macd': {'threshold': 0, 'states': ['neg', 'pos']},
    'volume_ratio': {'low': 0.8, 'high': 1.5, 'states': ['low', 'normal', 'high']},
    'wave_progress': {'low': 0.3, 'high': 0.7, 'states': ['early', 'mid', 'late']},
    'structure_score': {'threshold': 0.6, 'states': ['weak', 'strong']},
    'pattern_confidence': {'threshold': 0.5, 'states': ['uncertain', 'confident']},
    'mfi': {'low': 20, 'high': 80, 'states': ['low', 'mid', 'high']},
    'adx': {'threshold': 25, 'states': ['weak', 'strong']},
    'wave_momentum': {'threshold': 0.1, 'states': ['low', 'high']},
    'confidence': {'low': 0.3, 'high': 0.7, 'states': ['low', 'mid', 'high']},
    'volatility': {'low': 0.02, 'high': 0.05, 'states': ['low', 'mid', 'high']},
    'bb_width': {'low': 0.05, 'high': 0.1, 'states': ['narrow', 'normal', 'wide']},
    'bb_squeeze': {'threshold': 0.8, 'states': ['normal', 'squeezed']},
    'trend_strength': {'low': 0.3, 'high': 0.7, 'states': ['weak', 'moderate', 'strong']},
    'pattern_quality': {'low': 0.3, 'high': 0.7, 'states': ['low', 'mid', 'high']},
    'risk_score': {'low': 0.3, 'high': 0.7, 'states': ['low', 'mid', 'high']},
    'integrated_strength': {'low': 0.3, 'high': 0.7, 'states': ['weak', 'moderate', 'strong']},
}


def discretize_value(value: float, config: Dict[str, Any]) -> str:
    """
    값을 이산화하여 상태로 변환
    
    Args:
        value: 이산화할 값
        config: 이산화 설정 딕셔너리
    
    Returns:
        이산화된 상태 문자열
    """
    if 'threshold' in config:
        return config['states'][1] if value > config['threshold'] else config['states'][0]
    elif 'low' in config and 'high' in config:
        if value < config['low']:
            return config['states'][0]
        elif value > config['high']:
            return config['states'][2]
        else:
            return config['states'][1]
    return config['states'][0]


def process_technical_indicators(candle: pd.Series) -> Dict[str, Any]:
    """
    모든 기술지표를 설정 기반으로 처리
    
    Args:
        candle: 캔들 데이터 (pandas Series)
    
    Returns:
        처리된 기술지표 딕셔너리
    """
    indicators = {}
    
    # 설정 기반으로 모든 지표 처리
    for name, config in TECHNICAL_INDICATORS_CONFIG.items():
        value = candle.get(name)
        if config['type'] == 'float':
            indicators[name] = safe_float(value, config['default'])
        else:
            indicators[name] = safe_str(value, config['default'])
    
    # 특별 처리 로직들
    # 볼린저 밴드 위치 계산
    close = safe_float(candle.get('close'), 0.0)
    bb_middle = indicators['bb_middle']
    if bb_middle > 0 and close > 0:
        if close > bb_middle:
            indicators['bb_position'] = 'upper'
        elif close < bb_middle:
            indicators['bb_position'] = 'lower'
        else:
            indicators['bb_position'] = 'middle'
    else:
        indicators['bb_position'] = 'unknown'
    
    # 볼린저 밴드 스퀴즈 계산
    bb_upper = indicators['bb_upper']
    bb_lower = indicators['bb_lower']
    if bb_upper > 0 and bb_lower > 0:
        indicators['bb_squeeze'] = bb_upper - bb_lower
    else:
        indicators['bb_squeeze'] = 0.0
    
    # 볼린저 밴드 너비
    indicators['bb_width'] = indicators['bb_bandwidth']
    
    # 추세 강도
    indicators['trend_strength'] = indicators['ma20_pct_diff']
    
    # 새로 추가된 고급 지표들 (기본값으로 설정)
    indicators['rsi_divergence'] = 'none'
    indicators['macd_divergence'] = 'none'
    indicators['volume_divergence'] = 'none'
    indicators['price_momentum'] = 0.0
    indicators['volume_momentum'] = 0.0
    indicators['support_resistance'] = 'unknown'
    indicators['fibonacci_levels'] = 'unknown'
    indicators['elliott_wave'] = 'unknown'
    indicators['harmonic_patterns'] = 'none'
    indicators['candlestick_patterns'] = 'none'
    indicators['market_structure'] = 'unknown'
    indicators['risk_level'] = 'unknown'
    
    # 패턴 품질 특별 처리
    if indicators['pattern_quality'] == 0.0:
        # 패턴 품질을 다른 지표들을 기반으로 계산하는 로직은 나중에 처리
        pass
    
    return indicators

