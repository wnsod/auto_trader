"""테이블 스키마 확인"""
import sqlite3

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def check_schemas():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = [
        'strategy_grades',
        'integrated_analysis_results',
        'coin_strategies',
        'rl_strategy_rollup',
        'rl_episode_summary'
    ]

    for table in tables:
        print(f"\n{'='*80}")
        print(f"📋 {table}")
        print('='*80)

        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()

        for col in columns:
            col_id, name, col_type, not_null, default_val, pk = col
            print(f"  {name:30s} {col_type:15s} {'NOT NULL' if not_null else ''}")

    conn.close()

if __name__ == "__main__":
    check_schemas()
