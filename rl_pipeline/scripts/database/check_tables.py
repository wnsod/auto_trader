import sqlite3

# 캔들 DB 테이블 확인
conn = sqlite3.connect('/workspace/data_storage/rl_candles.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("rl_candles.db 테이블 목록:")
for table in tables:
    print(f"  - {table[0]}")

# 첫 번째 테이블의 샘플 데이터 확인
if tables:
    first_table = tables[0][0]
    cursor.execute(f"SELECT * FROM {first_table} LIMIT 1")
    print(f"\n{first_table} 테이블 샘플:")
    print(cursor.fetchone())

    # 컬럼 정보
    cursor.execute(f"PRAGMA table_info({first_table})")
    columns = cursor.fetchall()
    print(f"\n{first_table} 컬럼:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

conn.close()