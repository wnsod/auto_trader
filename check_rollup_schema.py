#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# Get table schema
cursor.execute("PRAGMA table_info(rl_strategy_rollup)")
columns = cursor.fetchall()

print("rl_strategy_rollup 테이블 스키마:")
print("=" * 70)
for col in columns:
    print(f"{col[1]:25s} {col[2]:15s} {'NOT NULL' if col[3] else ''}")

print()

# Sample data
cursor.execute("SELECT * FROM rl_strategy_rollup LIMIT 1")
row = cursor.fetchone()

if row:
    print("\n샘플 데이터:")
    print("=" * 70)
    for col, val in zip([c[1] for c in columns], row):
        print(f"{col:25s} = {val}")

conn.close()
