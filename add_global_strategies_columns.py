#!/usr/bin/env python
"""global_strategies 테이블에 파라미터 컬럼 추가"""
import sys
sys.path.append('/workspace')

import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("global_strategies 테이블에 파라미터 컬럼 추가")
print("=" * 70)
print()

# 추가할 컬럼 목록
columns_to_add = [
    ('rsi_min', 'REAL'),
    ('rsi_max', 'REAL'),
    ('volume_ratio_min', 'REAL'),
    ('volume_ratio_max', 'REAL'),
    ('macd_buy_threshold', 'REAL'),
    ('macd_sell_threshold', 'REAL'),
    ('mfi_min', 'REAL'),
    ('mfi_max', 'REAL'),
    ('atr_min', 'REAL'),
    ('atr_max', 'REAL'),
    ('adx_min', 'REAL'),
    ('stop_loss_pct', 'REAL'),
    ('take_profit_pct', 'REAL'),
    ('ma_period', 'INTEGER'),
    ('bb_period', 'INTEGER'),
    ('bb_std', 'REAL'),
]

for col_name, col_type in columns_to_add:
    try:
        cursor.execute(f"ALTER TABLE global_strategies ADD COLUMN {col_name} {col_type}")
        print(f"✅ {col_name} {col_type} 추가 완료")
    except sqlite3.OperationalError as e:
        if 'duplicate column name' in str(e):
            print(f"⚠️  {col_name} 이미 존재 (건너뜀)")
        else:
            print(f"❌ {col_name} 추가 실패: {e}")

conn.commit()
conn.close()

print()
print("=" * 70)
print("✅ 컬럼 추가 완료!")
print("=" * 70)
