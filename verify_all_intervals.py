#!/usr/bin/env python
"""
LINK 전체 인터벌 처리 결과 검증
"""
import sqlite3

COIN = 'LINK'
INTERVALS = ['15m', '30m', '240m', '1d']

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print(f"{COIN} 전체 인터벌 처리 결과")
print("=" * 70)
print()

# 인터벌별 결과
for interval in INTERVALS:
    print(f"📊 {interval} 결과:")
    print("-" * 70)

    # 전략 수
    cursor.execute("""
        SELECT COUNT(*) FROM coin_strategies
        WHERE coin=? AND interval=?
    """, (COIN, interval))
    strategy_count = cursor.fetchone()[0]

    # 에피소드 수
    cursor.execute("""
        SELECT COUNT(*) FROM rl_episode_summary
        WHERE episode_id LIKE ?
    """, (f"pred_{COIN}_{interval}_%",))
    episode_count = cursor.fetchone()[0]

    # 롤업 수
    cursor.execute("""
        SELECT COUNT(*), AVG(avg_ret), AVG(win_rate)
        FROM rl_strategy_rollup
        WHERE coin=? AND interval=?
    """, (COIN, interval))
    rollup_count, avg_ret, avg_win_rate = cursor.fetchone()

    # 등급 수
    cursor.execute("""
        SELECT COUNT(*) FROM strategy_grades
        WHERE coin=? AND interval=?
    """, (COIN, interval))
    grade_count = cursor.fetchone()[0]

    print(f"  전략: {strategy_count:3d}개")
    print(f"  에피소드: {episode_count:5d}개")
    print(f"  롤업: {rollup_count:3d}개")
    if avg_ret is not None:
        print(f"    평균 수익률: {avg_ret:.4f} ({avg_ret*100:.2f}%)")
        print(f"    평균 승률: {avg_win_rate:.2f}")
    print(f"  등급: {grade_count:3d}개")

    # 등급 분포
    if grade_count > 0:
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
        """, (COIN, interval))
        grades = cursor.fetchall()
        grade_str = ', '.join([f"{g}:{c}" for g, c in grades])
        print(f"    분포: {grade_str}")

    print()

# 전체 요약
print("=" * 70)
print("전체 요약")
print("=" * 70)

cursor.execute("""
    SELECT COUNT(*) FROM coin_strategies
    WHERE coin=?
""", (COIN,))
total_strategies = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) FROM rl_episode_summary
    WHERE episode_id LIKE ?
""", (f"pred_{COIN}_%",))
total_episodes = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) FROM rl_strategy_rollup
    WHERE coin=?
""", (COIN,))
total_rollups = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) FROM strategy_grades
    WHERE coin=?
""", (COIN,))
total_grades = cursor.fetchone()[0]

print(f"전체 전략: {total_strategies}개")
print(f"전체 에피소드: {total_episodes}개")
print(f"전체 롤업: {total_rollups}개")
print(f"전체 등급: {total_grades}개")

conn.close()
