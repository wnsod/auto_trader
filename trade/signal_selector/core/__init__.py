"""
Core 모듈 - 핵심 시그널 선택기 및 타입 정의
"""

from .types import SignalInfo, SignalAction

# SignalSelector와 StrategyScoreCalculator는 상위 레벨에서 임포트하거나 지연 임포트 사용
# (패키지 초기화 시의 순환 참조 방지)
SignalSelector = None
StrategyScoreCalculator = None

__all__ = ['SignalInfo', 'SignalAction']
if SignalSelector is not None:
    __all__.append('SignalSelector')
if StrategyScoreCalculator is not None:
    __all__.append('StrategyScoreCalculator')

