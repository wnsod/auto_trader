"""
데이터베이스 모듈
연결 풀, 읽기/쓰기, 스키마 관리
"""

from .connection_pool import (
    DatabaseConnectionPool, BatchLoadingConnectionPool,
    get_candle_db_pool, get_strategy_db_pool, get_batch_loading_pool,
    close_all_pools, get_optimized_db_connection,
    validate_simulation_results, validate_dna_results,
    validate_fractal_results, validate_pipeline_results
)

from .reads import (
    fetch_df, fetch_one, fetch_many, fetch_all,
    get_candle_data, get_strategy_data, get_top_strategies,
    get_strategy_by_id, get_dna_data, get_fractal_data,
    get_synergy_data, get_performance_data, get_run_history,
    check_table_exists, get_table_info, get_database_status
)

from .writes import (
    write_batch, upsert, update_strategy_performance,
    save_strategy_dna, save_fractal_analysis, save_synergy_analysis,
    save_run_metadata, delete_strategies, cleanup_old_data,
    transaction
)

from .schema import (
    ensure_indexes, create_candles_table, create_coin_strategies_table,
    create_strategy_dna_table, create_fractal_analysis_table,
    create_synergy_analysis_table, create_runs_table,
    setup_database_tables,
    migrate, check_database_integrity, repair_database
)

__all__ = [
    # Connection Pool
    "DatabaseConnectionPool", "BatchLoadingConnectionPool",
    "get_candle_db_pool", "get_strategy_db_pool", "get_batch_loading_pool",
    "close_all_pools", "get_optimized_db_connection",
    "validate_simulation_results", "validate_dna_results",
    "validate_fractal_results", "validate_pipeline_results",
    
    # Reads
    "fetch_df", "fetch_one", "fetch_many", "fetch_all",
    "get_candle_data", "get_strategy_data", "get_top_strategies",
    "get_strategy_by_id", "get_dna_data", "get_fractal_data",
    "get_synergy_data", "get_performance_data", "get_run_history",
    "check_table_exists", "get_table_info", "get_database_status",
    
    # Writes
    "write_batch", "upsert", "update_strategy_performance",
    "save_strategy_dna", "save_fractal_analysis", "save_synergy_analysis",
    "save_run_metadata", "delete_strategies", "cleanup_old_data",
    "transaction",
    
    # Schema
    "ensure_indexes", "create_candles_table", "create_coin_strategies_table",
    "create_strategy_dna_table", "create_fractal_analysis_table",
    "create_synergy_analysis_table", "create_runs_table",
    "setup_database_tables",
    "migrate", "check_database_integrity", "repair_database"
]
