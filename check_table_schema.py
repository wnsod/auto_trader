#!/usr/bin/env python
import sys
sys.path.append('/workspace')
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 테이블 스키마 확인
cursor.execute("PRAGMA table_info(rl_strategy_rollup)")
print("rl_strategy_rollup 테이블 스키마:")
print("=" * 70)
for row in cursor.fetchall():
    col_id, name, dtype, notnull, default, pk = row
    print(f"{col_id}: {name:30} {dtype:15} {'NOT NULL' if notnull else ''} {'PK' if pk else ''}")

print()

# 샘플 데이터 확인
cursor.execute("SELECT * FROM rl_strategy_rollup WHERE coin='LINK' LIMIT 3")
print("샘플 데이터:")
print("=" * 70)
rows = cursor.fetchall()

# 컬럼 이름
cursor.execute("PRAGMA table_info(rl_strategy_rollup)")
columns = [col[1] for col in cursor.fetchall()]
print(f"컬럼: {', '.join(columns)}")
print()

for row in rows:
    print(row[:5])  # 처음 5개만 출력

conn.close()
