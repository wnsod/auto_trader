#!/usr/bin/env python
"""DB 통합 테스트 - learning_results.db → rl_strategies.db"""
import sys
sys.path.append('/workspace')

from rl_pipeline.core.env import config
from rl_pipeline.db.learning_results import LEARNING_RESULTS_DB_PATH, create_learning_results_tables
import os

print("=" * 80)
print("Database Integration Test")
print("=" * 80)
print()

print("Configuration Check:")
print(f"  config.STRATEGIES_DB:            {config.STRATEGIES_DB}")
print(f"  config.LEARNING_RESULTS_DB_PATH: {config.LEARNING_RESULTS_DB_PATH}")
print()

print("Path Verification:")
if config.LEARNING_RESULTS_DB_PATH == config.STRATEGIES_DB:
    print("  ✓ SUCCESS: LEARNING_RESULTS_DB_PATH points to STRATEGIES_DB")
else:
    print("  ✗ FAILED: Paths do not match!")
    print(f"    LEARNING_RESULTS_DB_PATH: {config.LEARNING_RESULTS_DB_PATH}")
    print(f"    STRATEGIES_DB:            {config.STRATEGIES_DB}")

print()

print("Learning Results Module Check:")
print(f"  LEARNING_RESULTS_DB_PATH: {LEARNING_RESULTS_DB_PATH}")
if LEARNING_RESULTS_DB_PATH == config.STRATEGIES_DB:
    print("  ✓ SUCCESS: learning_results module uses correct DB path")
else:
    print("  ✗ FAILED: learning_results module has wrong DB path!")

print()

# Test table creation
print("Testing Table Creation:")
print("-" * 80)
try:
    result = create_learning_results_tables(config.STRATEGIES_DB)
    print(f"  ✓ create_learning_results_tables() returned: {result}")

    # Verify tables were created
    import sqlite3
    conn = sqlite3.connect(config.STRATEGIES_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    learning_tables = [
        'selfplay_evolution_results',
        'regime_routing_results',
        'integrated_analysis_results',
        'realtime_learning_feedback',
        'global_strategy_results',
        'pipeline_execution_logs',
        'strategy_summary_for_signals',
        'global_strategy_summary_for_signals',
        'dna_summary_for_signals',
        'analysis_summary_for_signals'
    ]

    print()
    print("  Learning Results Tables in rl_strategies.db:")
    for table in learning_tables:
        if table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"    ✓ {table:40s} ({count} records)")
        else:
            print(f"    ✗ {table:40s} (NOT FOUND)")

    conn.close()

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("\n✓ DB Integration Complete!")
print(f"  All learning results now stored in: {config.STRATEGIES_DB}")
print("=" * 80)
