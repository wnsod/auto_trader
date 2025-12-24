#!/usr/bin/env python3
"""
DB 경로 문제 해결 스크립트
"""

import os
import sqlite3
from pathlib import Path

def ensure_db_directories():
    """필요한 디렉토리 생성"""
    directories = [
        '/workspace/data_storage',
        '/workspace/rl_pipeline'
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 확인/생성: {dir_path}")

def check_db_files():
    """DB 파일 존재 여부 확인"""
    db_files = [
        '/workspace/data_storage/rl_candles.db',
        '/workspace/data_storage/learning_strategies.db',
        '/workspace/rl_pipeline/learning_strategies.db'
    ]

    print("\nDB 파일 상태:")
    for db_path in db_files:
        if os.path.exists(db_path):
            size = os.path.getsize(db_path) / 1024  # KB
            print(f"  ✅ {db_path} ({size:.1f} KB)")
        else:
            print(f"  ❌ {db_path} (없음)")

def create_learning_results_table():
    """learning_results 테이블 생성"""
    db_path = '/workspace/data_storage/learning_strategies.db'

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # learning_results 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            coin TEXT,
            interval TEXT,
            episode_id TEXT,
            episode_num INTEGER,
            timestamp INTEGER,
            total_reward REAL,
            win_rate REAL,
            avg_profit REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            num_trades INTEGER,
            best_strategy_id TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # paper_trading_results 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trading_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            coin TEXT,
            interval TEXT,
            strategy_id TEXT,
            timestamp INTEGER,
            signal_type TEXT,
            entry_price REAL,
            exit_price REAL,
            profit_pct REAL,
            position_size REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        conn.close()
        print("\n✅ learning_results 테이블 생성 완료")
        print("✅ paper_trading_results 테이블 생성 완료")

    except Exception as e:
        print(f"\n❌ 테이블 생성 실패: {e}")

def fix_permissions():
    """파일 권한 수정"""
    db_files = [
        '/workspace/data_storage/rl_candles.db',
        '/workspace/data_storage/learning_strategies.db',
        '/workspace/rl_pipeline/learning_strategies.db'
    ]

    print("\n권한 수정:")
    for db_path in db_files:
        if os.path.exists(db_path):
            try:
                os.chmod(db_path, 0o666)  # 읽기/쓰기 권한
                print(f"  ✅ {db_path}")
            except Exception as e:
                print(f"  ⚠️ {db_path}: {e}")

def main():
    print("="*60)
    print("DB 경로 문제 해결 시작")
    print("="*60)

    # 1. 디렉토리 생성
    ensure_db_directories()

    # 2. DB 파일 확인
    check_db_files()

    # 3. 누락된 테이블 생성
    create_learning_results_table()

    # 4. 권한 수정
    fix_permissions()

    print("\n" + "="*60)
    print("완료! 이제 absolute_zero_system.py를 다시 실행해보세요.")
    print("="*60)

if __name__ == "__main__":
    main()