#!/usr/bin/env python
"""learning_results.db 스키마 및 데이터 확인"""
import sqlite3
import os

learning_db = '/workspace/data_storage/learning_results.db'
strategies_db = '/workspace/data_storage/rl_strategies.db'

print("=" * 80)
print("Learning Results DB Analysis")
print("=" * 80)
print()

# learning_results.db 확인
print(f"Database: {learning_db}")
print(f"Size: {os.path.getsize(learning_db)} bytes")
print()

conn = sqlite3.connect(learning_db)
cursor = conn.cursor()

# 테이블 목록 가져오기
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cursor.fetchall()]

print(f"Tables ({len(tables)}):")
for table in tables:
    # 각 테이블의 레코드 수 확인
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  - {table}: {count} records")

print()

# 각 테이블의 스키마 확인
print("Table Schemas:")
print("-" * 80)
for table in tables:
    print(f"\nTable: {table}")
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()

    print("  Columns:")
    for col in columns:
        col_id, name, type_, notnull, default, pk = col
        print(f"    {name:20s} {type_:15s} {'NOT NULL' if notnull else ''} {'PRIMARY KEY' if pk else ''}")

conn.close()

print()
print("=" * 80)
print("\nrl_strategies.db 확인:")
print("-" * 80)

if os.path.exists(strategies_db):
    print(f"Size: {os.path.getsize(strategies_db)} bytes")

    conn2 = sqlite3.connect(strategies_db)
    cursor2 = conn2.cursor()

    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    strat_tables = [row[0] for row in cursor2.fetchall()]

    print(f"\nTables ({len(strat_tables)}):")
    for table in strat_tables:
        cursor2.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor2.fetchone()[0]
        print(f"  - {table}: {count} records")

    conn2.close()
else:
    print("rl_strategies.db does not exist yet")

print()
print("=" * 80)
