#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("coin_strategies 테이블 스키마:")
cursor.execute("PRAGMA table_info(coin_strategies)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]:30s} {col[2]:15s}")

print("\nLINK-15m 전략 샘플 (1개):")
cursor.execute("SELECT * FROM coin_strategies WHERE coin='LINK' AND interval='15m' LIMIT 1")
row = cursor.fetchone()
if row:
    cursor.execute("PRAGMA table_info(coin_strategies)")
    col_names = [c[1] for c in cursor.fetchall()]
    for name, value in zip(col_names, row):
        if value is not None and len(str(value)) < 200:
            print(f"  {name}: {value}")

conn.close()
