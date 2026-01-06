import sqlite3
import os

# BTC 전략 DB 스키마 확인
db_path = '/workspace/market/coin_market/data_storage/learning_strategies/BTC_strategies.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# strategies 테이블 스키마
print('=== strategies 테이블 스키마 ===')
cursor.execute("PRAGMA table_info(strategies)")
for col in cursor.fetchall():
    print(f'  {col[1]} ({col[2]})')

# analysis_ratios 테이블 스키마
print('\n=== analysis_ratios 테이블 스키마 ===')
cursor.execute("PRAGMA table_info(analysis_ratios)")
cols = cursor.fetchall()
if cols:
    for col in cols:
        print(f'  {col[1]} ({col[2]})')
else:
    print('  테이블 없음 또는 비어있음')

# 샘플 데이터 확인
print('\n=== strategies 샘플 데이터 (컬럼명) ===')
cursor.execute("SELECT * FROM strategies LIMIT 1")
if cursor.description:
    col_names = [desc[0] for desc in cursor.description]
    print(f'  컬럼: {col_names}')

conn.close()
