"""
전략 모듈
새로운 파이프라인에 맞춘 전략 관리 인터페이스
"""

import logging

logger = logging.getLogger(__name__)

# 글로벌 전략 Synthesizer 추가
try:
    from .global_synthesizer import GlobalStrategySynthesizer, create_global_synthesizer
    GLOBAL_SYNTHESIZER_AVAILABLE = True
except ImportError:
    # 🔥 선택적 모듈이므로 조용하게 처리
    GLOBAL_SYNTHESIZER_AVAILABLE = False

# 전략 관리자
try:
    from .manager import (
        StrategyManager, get_strategy_manager,
        generate_strategies, generate_strategies_with_indicators,
        save_strategies_to_db, generate_and_save_strategies,
        get_strategy_statistics, create_run_record, update_run_record,
        create_missing_tables_if_needed,
    )
    STRATEGY_MANAGER_AVAILABLE = True
except ImportError as e:
    # 🔥 필수 모듈이므로 경고 유지 (logger 사용)
    logger.warning(f"전략 관리자 import 실패: {e}")
    STRATEGY_MANAGER_AVAILABLE = False

# 전략 생성
try:
    from .creator import (
        create_coin_strategies, create_intelligent_strategies,
        create_coin_strategies_dynamic,
    )
except ImportError:
    # 🔥 선택적 모듈이므로 조용하게 처리
    pass

# 전략 라우팅
try:
    from .router import (
        run_coin_dynamic_routing,
        run_dynamic_routing_by_market_condition,
        create_dna_fractal_based_routing_strategies,
        create_enhanced_dynamic_routing_strategies,
        save_dynamic_routing_strategies_to_db,
    )
except ImportError:
    # 🔥 선택적 모듈이므로 조용하게 처리
    pass

# 전략 검증
try:
    from .validator import (
        revalidate_coin_strategies,
        revalidate_coin_strategies_dynamic,
        perform_enhanced_strategy_validation,
        update_strategy_grade,
        load_high_grade_strategies,
    )
except ImportError:
    # 🔥 선택적 모듈이므로 조용하게 처리
    pass

# 전략 분석
try:
    from .analyzer import (
        extract_optimal_conditions_from_analysis,
        extract_routing_patterns_from_analysis,
    )
except ImportError:
    # 🔥 선택적 모듈이므로 조용하게 처리
    pass

# AI 데이터 수집
try:
    from .ai_collector import (
        collect_strategy_performance_for_ai,
        collect_strategy_comparison_for_ai,
        collect_learning_episode_for_ai,
        collect_learning_state_for_ai,
        collect_learning_action_for_ai,
        collect_learning_reward_for_ai,
        collect_model_training_data_for_ai,
    )
except ImportError:
    # 🔥 선택적 모듈이므로 조용하게 처리
    pass

__all__ = [
    # Global Synthesizer
    "GlobalStrategySynthesizer", "create_global_synthesizer",
    # Manager
    "StrategyManager", "get_strategy_manager",
    "generate_strategies", "generate_strategies_with_indicators",
    "save_strategies_to_db", "generate_and_save_strategies",
    "get_strategy_statistics", "create_run_record", "update_run_record",
    "create_coin_strategies", "create_intelligent_strategies", "revalidate_coin_strategies",
    "run_coin_dynamic_routing", "extract_optimal_conditions_from_analysis",
    "perform_enhanced_strategy_validation", "update_strategy_grade",
    "extract_routing_patterns_from_analysis", "load_high_grade_strategies",
    "create_dna_fractal_based_routing_strategies", "create_enhanced_dynamic_routing_strategies",
    "save_dynamic_routing_strategies_to_db", "create_missing_tables_if_needed",
    # 🆕 동적 분할 및 시장 상황별 함수들
    "create_coin_strategies_dynamic", "revalidate_coin_strategies_dynamic",
    "run_dynamic_routing_by_market_condition",
    # 🤖 AI 학습용 데이터 수집 함수들
    "collect_strategy_performance_for_ai", "collect_strategy_comparison_for_ai",
    "collect_learning_episode_for_ai", "collect_learning_state_for_ai",
    "collect_learning_action_for_ai", "collect_learning_reward_for_ai",
    "collect_model_training_data_for_ai"
]