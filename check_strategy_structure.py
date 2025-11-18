#!/usr/bin/env python
"""전략 구조 확인"""
import sys
sys.path.append('/workspace')

import sqlite3
import json

DB_PATH = '/workspace/data_storage/rl_strategies.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. coin_strategies 테이블 스키마
print("=" * 70)
print("coin_strategies 테이블 컬럼:")
print("=" * 70)
cursor.execute("PRAGMA table_info(coin_strategies)")
cols = cursor.fetchall()
for c in cols:
    print(f"  {c[1]:<30} {c[2]}")
print()

# 2. 샘플 전략 조회
cursor.execute("""
    SELECT id, coin, interval, strategy_type,
           rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
           params
    FROM coin_strategies
    LIMIT 1
""")
row = cursor.fetchone()

if row:
    print("=" * 70)
    print("샘플 전략:")
    print("=" * 70)
    cols_names = [d[0] for d in cursor.description]
    data = dict(zip(cols_names, row))

    for k, v in data.items():
        if k == 'params' and isinstance(v, str):
            try:
                v = json.loads(v)
            except:
                pass
        print(f"  {k}: {v}")
    print()

# 3. params 필드가 JSON인지 확인
cursor.execute("SELECT params FROM coin_strategies LIMIT 3")
params_samples = cursor.fetchall()

print("=" * 70)
print("params 샘플 3개:")
print("=" * 70)
for i, (p,) in enumerate(params_samples, 1):
    print(f"{i}. Type: {type(p)}")
    if isinstance(p, str):
        try:
            parsed = json.loads(p)
            print(f"   Parsed keys: {list(parsed.keys())[:10]}")
        except:
            print(f"   Content: {p[:100]}")
    else:
        print(f"   Value: {p}")
    print()

conn.close()
