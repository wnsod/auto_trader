#!/usr/bin/env python3
"""마이그레이션 결과 확인 스크립트"""

import sqlite3
from pathlib import Path

db_path = "/workspace/data_storage/rl_strategies.db"
if not Path(db_path).exists():
    db_path = "./data_storage/rl_strategies.db"

print(f"🔍 DB 경로: {db_path}")

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # 전체 데이터 개수
    cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
    total = cursor.fetchone()[0]
    print(f"\n📊 전체 데이터 개수: {total}개")

    # 샘플 데이터 5개
    cursor.execute("""
        SELECT coin, interval, regime, final_signal_score, signal_action, created_at
        FROM integrated_analysis_results
        ORDER BY created_at DESC
        LIMIT 5
    """)

    print("\n📝 최신 데이터 샘플 (5개):")
    for row in cursor.fetchall():
        print(f"  - {row[0]}-{row[1]}: regime={row[2]}, score={row[3]:.3f}, action={row[4]}, time={row[5]}")

    # regime 분포
    cursor.execute("""
        SELECT regime, COUNT(*) as cnt
        FROM integrated_analysis_results
        GROUP BY regime
        ORDER BY cnt DESC
    """)

    print("\n📊 Regime 분포:")
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]}개")

print("\n✅ 마이그레이션 결과 확인 완료!")
