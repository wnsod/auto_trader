#!/usr/bin/env python
"""global_strategies 테이블 스키마 확인"""
import sys
sys.path.append('/workspace')

import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("global_strategies 테이블 스키마")
print("=" * 70)
print()

cursor.execute("PRAGMA table_info(global_strategies)")
cols = cursor.fetchall()

for c in cols:
    print(f"  {c[1]:<30} {c[2]}")

conn.close()
print()
