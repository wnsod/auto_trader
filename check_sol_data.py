import sqlite3

db_path = '/workspace/data_storage/learning_results.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 70)
print("SOL 코인 데이터 상세 확인")
print("=" * 70)

# SOL의 모든 데이터 조회
cursor.execute("""
    SELECT interval, signal_action, final_signal_score, created_at
    FROM integrated_analysis_results
    WHERE coin = 'SOL'
    ORDER BY created_at DESC
""")

rows = cursor.fetchall()
print(f"\nSOL 총 레코드 수: {len(rows)}개\n")

if rows:
    print("SOL 데이터 목록 (최신순):")
    print("-" * 70)
    for i, row in enumerate(rows, 1):
        interval, signal, score, created = row
        print(f"{i}. interval={interval}, signal={signal}, score={score:.4f}")
        print(f"   created_at={created}\n")
else:
    print("❌ SOL 데이터가 없습니다.")

# 최신 all_intervals 조회
print("\n" + "=" * 70)
print("SOL all_intervals 조회")
print("=" * 70)
cursor.execute("""
    SELECT interval, signal_action, final_signal_score, created_at
    FROM integrated_analysis_results
    WHERE coin = 'SOL' AND interval = 'all_intervals'
    ORDER BY created_at DESC
    LIMIT 1
""")
row = cursor.fetchone()
if row:
    print(f"✅ all_intervals 존재: {row[1]} (점수: {row[2]:.4f})")
    print(f"   생성시간: {row[3]}")
else:
    print("❌ all_intervals 데이터 없음")

# 개별 인터벌 조회
print("\n" + "=" * 70)
print("SOL 개별 인터벌 조회")
print("=" * 70)
for test_interval in ['15m', '30m', '240m', '1d']:
    cursor.execute("""
        SELECT signal_action, final_signal_score, created_at
        FROM integrated_analysis_results
        WHERE coin = 'SOL' AND interval = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (test_interval,))
    row = cursor.fetchone()
    if row:
        print(f"✅ {test_interval}: {row[0]} (점수: {row[1]:.4f}), 시간: {row[2]}")
    else:
        print(f"❌ {test_interval}: 데이터 없음")

conn.close()
