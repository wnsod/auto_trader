#!/usr/bin/env python3
"""Check recent strategy quality_grades in database"""
import sqlite3
import sys

sys.path.append('/workspace')

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cur = conn.cursor()

# First, list all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print("📋 Available tables:")
for table in tables:
    print(f"  - {table[0]}")

# Check coin_strategies table
if ('coin_strategies',) in tables:
    print("\n📊 coin_strategies 스키마:")
    cur.execute("PRAGMA table_info(coin_strategies)")
    for col in cur.fetchall():
        print(f"  {col[1]} ({col[2]})")

    print("\n📊 최근 20개 전략의 quality_grade (coin_strategies):")
    print("-" * 100)
    # Check if quality_grade column exists
    cur.execute("PRAGMA table_info(coin_strategies)")
    columns = [col[1] for col in cur.fetchall()]

    if 'quality_grade' in columns:
        cur.execute("""
            SELECT id, quality_grade, coin, interval, created_at
            FROM coin_strategies
            ORDER BY created_at DESC
            LIMIT 20
        """)
        for row in cur.fetchall():
            strategy_id, grade, coin, interval, timestamp = row
            if strategy_id and grade:
                print(f"{strategy_id[:20]}... | {grade:12s} | {coin:5s} | {interval:5s} | {timestamp}")
    else:
        print("  ⚠️ quality_grade 컬럼 없음")

# Check strategy_grades table
if ('strategy_grades',) in tables:
    print("\n📊 strategy_grades 테이블:")
    cur.execute("SELECT COUNT(*) FROM strategy_grades")
    count = cur.fetchone()[0]
    print(f"  총 레코드: {count}")

    cur.execute("SELECT * FROM strategy_grades LIMIT 5")
    for row in cur.fetchall():
        print(f"  {row}")

conn.close()
