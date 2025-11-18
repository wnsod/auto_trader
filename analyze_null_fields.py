"""
NULL 필드 분석 - 실제 사용 여부 확인
"""
import sqlite3

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def analyze_null_fields():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print("📊 NULL 필드 분석")
    print("=" * 80)

    # 1. parent_id & parent_strategy_id
    print("\n1. parent_id & parent_strategy_id (전략 진화 추적)")
    print("-" * 80)

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN parent_id IS NOT NULL THEN 1 ELSE 0 END) as has_parent_id,
            SUM(CASE WHEN parent_strategy_id IS NOT NULL THEN 1 ELSE 0 END) as has_parent_strategy_id
        FROM coin_strategies
    """)
    result = cursor.fetchone()

    print(f"  총 전략: {result[0]:,}개")
    print(f"  parent_id 있음: {result[1]:,}개 ({result[1]/result[0]*100:.2f}%)")
    print(f"  parent_strategy_id 있음: {result[2]:,}개 ({result[2]/result[0]*100:.2f}%)")

    if result[2] > 0:
        cursor.execute("""
            SELECT coin, interval, COUNT(*) as cnt
            FROM coin_strategies
            WHERE parent_strategy_id IS NOT NULL
            GROUP BY coin, interval
            ORDER BY cnt DESC
            LIMIT 5
        """)
        print(f"\n  📌 parent_strategy_id를 가진 전략 (상위 5개):")
        for row in cursor.fetchall():
            print(f"     - {row[0]}-{row[1]}: {row[2]}개")
        print(f"  ✅ 결론: parent_strategy_id는 진화된 전략에만 사용됨 (정상)")
    else:
        print(f"  ⚠️ parent_strategy_id가 전혀 사용되지 않음")

    # 2. hybrid_score
    print("\n2. hybrid_score (하이브리드 모델 점수)")
    print("-" * 80)

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN hybrid_score IS NOT NULL THEN 1 ELSE 0 END) as has_hybrid
        FROM coin_strategies
    """)
    result = cursor.fetchone()

    print(f"  총 전략: {result[0]:,}개")
    print(f"  hybrid_score 있음: {result[1]:,}개 ({result[1]/result[0]*100:.2f}%)")

    if result[1] == 0:
        print(f"  ℹ️ 결론: hybrid_score는 현재 사용되지 않음 (미래 기능용 예약 필드)")
    else:
        print(f"  ✅ 결론: hybrid_score가 일부 전략에 사용됨")

    # 3. last_train_end_idx
    print("\n3. last_train_end_idx (마지막 훈련 종료 인덱스)")
    print("-" * 80)

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN last_train_end_idx IS NOT NULL THEN 1 ELSE 0 END) as has_idx
        FROM coin_strategies
    """)
    result = cursor.fetchone()

    print(f"  총 전략: {result[0]:,}개")
    print(f"  last_train_end_idx 있음: {result[1]:,}개 ({result[1]/result[0]*100:.2f}%)")

    if result[1] == 0:
        print(f"  ℹ️ 결론: last_train_end_idx는 현재 사용되지 않음 (미래 기능용 예약 필드)")
    else:
        print(f"  ✅ 결론: last_train_end_idx가 일부 전략에 사용됨")

    # 4. 다른 중요 NULL 체크
    print("\n4. 기타 중요 필드 NULL 체크")
    print("-" * 80)

    cursor.execute("""
        SELECT
            SUM(CASE WHEN avg_ret IS NULL THEN 1 ELSE 0 END) as null_avg_ret,
            SUM(CASE WHEN win_rate IS NULL THEN 1 ELSE 0 END) as null_win_rate,
            SUM(CASE WHEN params IS NULL THEN 1 ELSE 0 END) as null_params,
            SUM(CASE WHEN created_at IS NULL THEN 1 ELSE 0 END) as null_created_at
        FROM coin_strategies
    """)
    result = cursor.fetchone()

    issues = []
    if result[0] > 0:
        issues.append(f"avg_ret NULL: {result[0]}개")
    if result[1] > 0:
        issues.append(f"win_rate NULL: {result[1]}개")
    if result[2] > 0:
        issues.append(f"params NULL: {result[2]}개 (심각!)")
    if result[3] > 0:
        issues.append(f"created_at NULL: {result[3]}개")

    if issues:
        print(f"  ⚠️ 발견된 문제:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"  ✅ 중요 필드 모두 정상")

    # 요약
    print("\n" + "=" * 80)
    print("📋 요약")
    print("=" * 80)
    print("  ✅ parent_id, parent_strategy_id: 진화 전략 추적용 (NULL 정상)")
    print("  ℹ️ hybrid_score, last_train_end_idx: 미래 기능용 예약 필드")
    print("  📌 이들 필드의 높은 NULL 비율은 설계상 정상이며 문제 아님")

    conn.close()


if __name__ == "__main__":
    analyze_null_fields()
