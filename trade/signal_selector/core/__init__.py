"""
Core 모듈 - 핵심 시그널 선택기 및 타입 정의
"""

from .types import SignalInfo, SignalAction

# SignalSelector와 StrategyScoreCalculator는 아직 원본 파일에 있으므로 선택적 import
try:
    from .signal_selector import SignalSelector
except ImportError:
    SignalSelector = None

try:
    from .strategy_calculator import StrategyScoreCalculator
except ImportError:
    StrategyScoreCalculator = None

__all__ = ['SignalInfo', 'SignalAction']
if SignalSelector is not None:
    __all__.append('SignalSelector')
if StrategyScoreCalculator is not None:
    __all__.append('StrategyScoreCalculator')

