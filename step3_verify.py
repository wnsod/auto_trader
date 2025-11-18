#!/usr/bin/env python
"""
3단계 검증: 롤업 및 등급 측정 결과 확인
"""
import sqlite3

COIN = 'LINK'
INTERVAL = '15m'

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("3단계 검증: 롤업 및 등급 측정")
print("=" * 70)
print()

# 1. 롤업 데이터 확인
print("📊 롤업 데이터 확인")
print("-" * 70)

cursor.execute("""
    SELECT COUNT(*) FROM rl_strategy_rollup
    WHERE coin=? AND interval=?
""", (COIN, INTERVAL))
total_rollup = cursor.fetchone()[0]
print(f"총 롤업 레코드: {total_rollup}개")

if total_rollup > 0:
    # avg_ret이 NULL이 아닌 레코드 수
    cursor.execute("""
        SELECT COUNT(*) FROM rl_strategy_rollup
        WHERE coin=? AND interval=? AND avg_ret IS NOT NULL AND avg_ret != 0
    """, (COIN, INTERVAL))
    non_zero_count = cursor.fetchone()[0]
    print(f"avg_ret > 0인 레코드: {non_zero_count}개")

    # 평균 통계
    cursor.execute("""
        SELECT
            AVG(avg_ret) as mean_ret,
            AVG(win_rate) as mean_wr,
            AVG(predictive_accuracy) as mean_acc,
            AVG(episodes_trained) as mean_episodes
        FROM rl_strategy_rollup
        WHERE coin=? AND interval=?
    """, (COIN, INTERVAL))
    mean_ret, mean_wr, mean_acc, mean_episodes = cursor.fetchone()

    print(f"\n평균 통계:")
    if mean_ret is not None:
        print(f"  평균 수익률: {mean_ret:.4f} ({mean_ret*100:.2f}%)")
    if mean_wr is not None:
        print(f"  평균 승률: {mean_wr:.2f}")
    if mean_acc is not None:
        print(f"  평균 예측 정확도: {mean_acc:.2f}")
    if mean_episodes is not None:
        print(f"  평균 에피소드 수: {mean_episodes:.1f}개")

    # 상위 5개 전략
    cursor.execute("""
        SELECT
            strategy_id,
            episodes_trained,
            avg_ret,
            win_rate,
            predictive_accuracy
        FROM rl_strategy_rollup
        WHERE coin=? AND interval=?
        ORDER BY avg_ret DESC
        LIMIT 5
    """, (COIN, INTERVAL))
    top_strategies = cursor.fetchall()

    print(f"\n상위 5개 전략 (avg_ret 기준):")
    for sid, episodes, avg_ret, win_rate, pred_acc in top_strategies:
        sid_short = sid[:50] + "..." if len(sid) > 50 else sid
        print(f"  {sid_short}")
        ret_pct = avg_ret * 100 if avg_ret else 0
        print(f"    에피소드: {episodes}, avg_ret: {avg_ret:.4f} ({ret_pct:.2f}%), win_rate: {win_rate:.2f}, acc: {pred_acc:.2f}")

else:
    print("❌ 롤업 데이터가 없습니다.")

print()

# 2. 등급 데이터 확인
print("🎓 등급 데이터 확인")
print("-" * 70)

cursor.execute("""
    SELECT COUNT(*) FROM strategy_grades
    WHERE coin=? AND interval=?
""", (COIN, INTERVAL))
total_grades = cursor.fetchone()[0]
print(f"총 등급 레코드: {total_grades}개")

if total_grades > 0:
    # 등급 분포
    cursor.execute("""
        SELECT grade, COUNT(*)
        FROM strategy_grades
        WHERE coin=? AND interval=?
        GROUP BY grade
        ORDER BY
            CASE grade
                WHEN 'S' THEN 1
                WHEN 'A' THEN 2
                WHEN 'B' THEN 3
                WHEN 'C' THEN 4
                WHEN 'D' THEN 5
                WHEN 'F' THEN 6
                ELSE 7
            END
    """, (COIN, INTERVAL))
    grade_dist = cursor.fetchall()

    print(f"\n등급 분포:")
    for grade, count in grade_dist:
        pct = count / total_grades * 100
        print(f"  {grade}: {count:3d}개 ({pct:5.1f}%)")

    # S등급 전략
    cursor.execute("""
        SELECT strategy_id, predictive_accuracy
        FROM strategy_grades
        WHERE coin=? AND interval=? AND grade='S'
        LIMIT 5
    """, (COIN, INTERVAL))
    s_grades = cursor.fetchall()

    if s_grades:
        print(f"\nS등급 전략 (최대 5개):")
        for sid, pred_acc in s_grades:
            sid_short = sid[:50] + "..." if len(sid) > 50 else sid
            print(f"  {sid_short}")
            print(f"    예측 정확도: {pred_acc:.2f}")

else:
    print("❌ 등급 데이터가 없습니다.")

print()
print("=" * 70)

if total_rollup > 0 and total_grades > 0:
    print("✅ 3단계 검증 완료: 롤업 및 등급 측정 성공")
elif total_rollup > 0:
    print("⚠️ 롤업은 성공했지만 등급 데이터가 없습니다.")
elif total_grades > 0:
    print("⚠️ 등급 데이터는 있지만 롤업 데이터가 없습니다.")
else:
    print("❌ 롤업 및 등급 데이터가 모두 없습니다.")

print("=" * 70)

conn.close()
