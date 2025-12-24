"""
예측형 강화학습 시스템 엔진 모듈
"""

from rl_pipeline.engine.interval_profile import INTERVAL_PROFILE, get_interval_profile
from rl_pipeline.engine.reward_engine import RewardEngine
from rl_pipeline.engine.weight_engine import calc_weight, calc_weights_batch
from rl_pipeline.engine.adaptive_rollup import (
    calculate_adaptive_rollup_days,
    get_coin_rollup_profile,
    create_adaptive_rollup_view
)
from rl_pipeline.engine.rollup_batch import (
    run_rollup_batch,
    run_full_rollup_and_grades
)

# 🔥 수정: 미구현 모듈은 선택적 import
try:
    from rl_pipeline.engine.prediction_generator import (
        Prediction,
        PredictionGenerator,
        generate_prediction
    )
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False
    Prediction = None
    PredictionGenerator = None
    generate_prediction = None

__all__ = [
    'INTERVAL_PROFILE',
    'get_interval_profile',
    'RewardEngine',
    'calc_weight',
    'calc_weights_batch',
    'calculate_adaptive_rollup_days',
    'get_coin_rollup_profile',
    'create_adaptive_rollup_view',
    'run_rollup_batch',
    'run_full_rollup_and_grades',
    'Prediction',
    'PredictionGenerator',
    'generate_prediction',
    'PREDICTION_AVAILABLE',
]

