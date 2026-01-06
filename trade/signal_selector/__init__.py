"""
Signal Selector 모듈 - 실시간 시그널 생성 시스템

분할된 구조:
- core/: 핵심 클래스 및 타입
- db/: 데이터베이스 로드/저장
- analysis/: 기술적/시장 분석
- scoring/: 점수 계산
- generator/: 시그널 생성
- cache/: 캐시 관리
- strategy/: 전략 관리
- evaluators/: 평가자 클래스
- utils/: 유틸리티 함수
- config.py: 설정
"""

# 타입 정의는 즉시 임포트
from .core.types import SignalInfo, SignalAction

# SignalSelector는 필요한 경우에만 임포트할 수 있도록 보조 함수 제공
def get_signal_selector():
    """순환 참조 방지를 위해 지연 임포트를 수행하는 팩토리 함수"""
    try:
        from .core.selector import SignalSelector
        return SignalSelector()
    except Exception as e:
        print(f"⚠️ SignalSelector 로드 실패: {e}")
        return None

# 하위 호환성을 위해 시도하지만 실패해도 패키지 로드는 차단하지 않음
try:
    from .core.selector import SignalSelector
    __all__ = ['SignalSelector', 'SignalInfo', 'SignalAction', 'get_signal_selector']
except Exception:
    SignalSelector = None
    __all__ = ['SignalInfo', 'SignalAction', 'get_signal_selector']

