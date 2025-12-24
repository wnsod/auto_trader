"""
🧬 Phase 3 Evolution Module
Self-Play + RL 통합 진화 시스템

Auto-Evolution System:
- Phase 1: 통계 기반 (MFE/MAE EntryScore)
- Phase 2: MFE/MAE 예측 모델 (XGBoost/LightGBM)
- Phase 3: 타이밍 최적화 (RL Agent)

종목별로 정확도에 따라 자동으로 Phase를 승격/강등합니다.
"""

from .label_reward_system import LabelRewardSystem, StrategyReward, RewardWeights
from .phase_manager import (
    PhaseManager, 
    Phase, 
    PhaseState, 
    PhaseThresholds,
    get_phase_manager,
    reset_phase_manager
)
from .accuracy_tracker import (
    AccuracyTracker,
    PredictionRecord,
    get_accuracy_tracker
)
from .auto_evolution import (
    AutoEvolutionSystem,
    SignalResult,
    get_auto_evolution,
    run_evolution_check
)

__all__ = [
    # Label Reward System
    'LabelRewardSystem', 
    'StrategyReward', 
    'RewardWeights',
    
    # Phase Manager
    'PhaseManager',
    'Phase',
    'PhaseState',
    'PhaseThresholds',
    'get_phase_manager',
    'reset_phase_manager',
    
    # Accuracy Tracker
    'AccuracyTracker',
    'PredictionRecord',
    'get_accuracy_tracker',
    
    # Auto Evolution
    'AutoEvolutionSystem',
    'SignalResult',
    'get_auto_evolution',
    'run_evolution_check'
]
