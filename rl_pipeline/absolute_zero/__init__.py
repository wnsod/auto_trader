"""
Absolute Zero 시스템 - 통합 오케스트레이터 패키지
모든 파이프라인 기능을 통합한 모듈화된 시스템
"""

from .az_config import (
    configure_logging,
    ensure_storage_ready,
    ensure_file_exists,
    AZ_DEBUG,
    AZ_LOG_FILE,
    AZ_SIMULATION_VERBOSE,
    AZ_INTERVALS,
    AZ_CANDLE_DAYS,
    AZ_ALLOW_FALLBACK,
    AZ_FALLBACK_PAIRS,
    AZ_SELFPLAY_EPISODES,
    AZ_SELFPLAY_AGENTS_PER_EPISODE,
    AZ_STRATEGY_POOL_SIZE,
    PREDICTIVE_SELFPLAY_RATIO,
    STRATEGIES_DB_PATH,
    CANDLES_DB_PATH,
    LEARNING_RESULTS_DB_PATH,
    ENABLE_STRATEGY_FILTERING,
    MIN_CANDLES_PER_INTERVAL
)

from .az_utils import (
    sort_intervals,
    execute_wal_checkpoint,
    format_time_duration,
    check_data_sufficiency,
    create_run_metadata,
    log_system_info,
    validate_environment
)

from .az_analysis import (
    calculate_global_analysis_data,
    calculate_fractal_score,
    calculate_multi_timeframe_coherence,
    calculate_indicator_cross_validation,
    validate_strategy_quality,
    analyze_strategy_distribution
)

from .az_global_strategies import generate_global_strategies_only

from .az_main import (
    run_absolute_zero,
    main
)

__all__ = [
    # Config
    'configure_logging',
    'ensure_storage_ready',
    'ensure_file_exists',
    'AZ_DEBUG',
    'AZ_LOG_FILE',
    'AZ_SIMULATION_VERBOSE',
    'AZ_INTERVALS',
    'AZ_CANDLE_DAYS',
    'AZ_ALLOW_FALLBACK',
    'AZ_FALLBACK_PAIRS',
    'AZ_SELFPLAY_EPISODES',
    'AZ_SELFPLAY_AGENTS_PER_EPISODE',
    'AZ_STRATEGY_POOL_SIZE',
    'PREDICTIVE_SELFPLAY_RATIO',
    'STRATEGIES_DB_PATH',
    'CANDLES_DB_PATH',
    'LEARNING_RESULTS_DB_PATH',
    'ENABLE_STRATEGY_FILTERING',
    'MIN_CANDLES_PER_INTERVAL',

    # Utils
    'sort_intervals',
    'execute_wal_checkpoint',
    'format_time_duration',
    'check_data_sufficiency',
    'create_run_metadata',
    'log_system_info',
    'validate_environment',

    # Analysis
    'calculate_global_analysis_data',
    'calculate_fractal_score',
    'calculate_multi_timeframe_coherence',
    'calculate_indicator_cross_validation',
    'validate_strategy_quality',
    'analyze_strategy_distribution',

    # Global Strategies
    'generate_global_strategies_only',

    # Main
    'run_absolute_zero',
    'main'
]

__version__ = '2.0.0'