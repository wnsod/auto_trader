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

from .core.types import SignalInfo, SignalAction

# SignalSelector는 통합 클래스에서 import
try:
    from .core.selector import SignalSelector
    __all__ = ['SignalSelector', 'SignalInfo', 'SignalAction']
except ImportError as e:
    # 개별 Mixin만 사용하는 경우
    print(f"⚠️ SignalSelector 통합 클래스 로드 실패: {e}")
    __all__ = ['SignalInfo', 'SignalAction']

