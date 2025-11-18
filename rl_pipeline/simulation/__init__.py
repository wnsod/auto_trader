"""
시뮬레이션 모듈
새로운 파이프라인에 맞춘 Self-play 시뮬레이션 인터페이스
"""

# Self-play 시뮬레이션만 유지
try:
    from .selfplay import (
        MarketDataGenerator, StrategyAgent, SelfPlaySimulator,
        run_self_play_test, run_self_play_evolution
    )
    SELFPLAY_AVAILABLE = True
except ImportError as e:
    # 🔥 logger로 변경 (print 제거) - 필수 모듈이므로 경고 유지
    import logging
    logging.getLogger(__name__).warning(f"⚠️ Self-play 모듈 import 실패: {e}")
    SELFPLAY_AVAILABLE = False

# 예측 실현 Self-play (선택적, 현재 미구현)
try:
    from .predictive_selfplay import (
        PredictiveSelfPlayTrainer,
        run_predictive_self_play_test
    )
    PREDICTIVE_SELFPLAY_AVAILABLE = True
except ImportError:
    # 🔥 조용하게 처리 (선택적 모듈이므로 경고 불필요)
    PREDICTIVE_SELFPLAY_AVAILABLE = False
    PredictiveSelfPlayTrainer = None
    run_predictive_self_play_test = None

__all__ = [
    # Self-play 시뮬레이션
    "MarketDataGenerator", "StrategyAgent", "SelfPlaySimulator",
    "run_self_play_test", "run_self_play_evolution",
    # 예측 실현 Self-play (선택적)
    "PREDICTIVE_SELFPLAY_AVAILABLE"
]

# 🔥 조건부 export (모듈이 있을 때만)
if PREDICTIVE_SELFPLAY_AVAILABLE:
    __all__.extend(["PredictiveSelfPlayTrainer", "run_predictive_self_play_test"])