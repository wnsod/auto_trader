#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("롤업 상태 확인")
print("=" * 70)
print()

# 1. 전체 예측 에피소드 통계
cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN first_event = 'TP' THEN 1 ELSE 0 END) as tp,
        SUM(CASE WHEN first_event = 'SL' THEN 1 ELSE 0 END) as sl,
        SUM(CASE WHEN first_event = 'expiry' THEN 1 ELSE 0 END) as expiry,
        AVG(realized_ret_signed) as avg_ret
    FROM rl_episode_summary
    WHERE episode_id LIKE 'pred_%'
""")

total, tp, sl, expiry, avg_ret = cursor.fetchone()

print("📊 전체 예측 에피소드:")
print(f"  전체: {total:,}개")
if total > 0:
    print(f"  TP: {tp:,}개 ({tp/total*100:.2f}%)")
    print(f"  SL: {sl:,}개 ({sl/total*100:.2f}%)")
    print(f"  만료: {expiry:,}개 ({expiry/total*100:.2f}%)")
    print(f"  평균 수익률: {avg_ret:.6f}")
print()

# 2. 롤업 데이터 확인
print("📈 롤업 데이터:")
cursor.execute("""
    SELECT COUNT(*) FROM rl_strategy_rollup
""")
rollup_count = cursor.fetchone()[0]
print(f"  롤업 레코드: {rollup_count:,}개")

if rollup_count > 0:
    cursor.execute("""
        SELECT
            strategy_id,
            episodes_trained,
            avg_ret,
            win_rate,
            predictive_accuracy
        FROM rl_strategy_rollup
        WHERE avg_ret IS NOT NULL AND avg_ret != 0
        LIMIT 5
    """)

    non_zero_rollups = cursor.fetchall()
    if non_zero_rollups:
        print(f"\n  avg_ret이 0이 아닌 전략 (샘플 {len(non_zero_rollups)}개):")
        for strategy_id, episodes, avg_ret, win_rate, pred_acc in non_zero_rollups:
            print(f"    {strategy_id[:50]}...")
            print(f"      에피소드: {episodes}, avg_ret: {avg_ret:.4f}, win_rate: {win_rate:.2f}")
    else:
        print("\n  ⚠️ 모든 롤업 레코드의 avg_ret이 0 또는 NULL입니다.")

        # 0인 롤업 샘플 확인
        cursor.execute("""
            SELECT
                strategy_id,
                episodes_trained,
                avg_ret,
                win_rate
            FROM rl_strategy_rollup
            LIMIT 5
        """)
        zero_rollups = cursor.fetchall()
        print("\n  샘플 롤업 레코드:")
        for strategy_id, episodes, avg_ret, win_rate in zero_rollups:
            print(f"    {strategy_id[:50]}...")
            print(f"      에피소드: {episodes}, avg_ret: {avg_ret}, win_rate: {win_rate}")

print()

# 3. 롤업 로직 실행 필요 여부 판단
if total > 0 and rollup_count == 0:
    print("❌ 에피소드는 있지만 롤업 데이터가 없습니다 → 롤업 실행 필요")
elif total > 0 and avg_ret != 0 and rollup_count > 0:
    # 롤업된 에피소드 수 확인
    cursor.execute("""
        SELECT SUM(episodes_trained) FROM rl_strategy_rollup
    """)
    rollup_total_episodes = cursor.fetchone()[0] or 0

    if rollup_total_episodes < total * 0.5:
        print(f"⚠️ 롤업된 에피소드 수({rollup_total_episodes:,})가 전체 에피소드({total:,})의 50% 미만")
        print("   → 롤업 업데이트 필요")
    else:
        print(f"✅ 롤업 정상: {rollup_total_episodes:,}개 에피소드가 롤업됨")
else:
    print("⚠️ 에피소드의 avg_ret이 0입니다 → 시뮬레이션 문제 또는 롤업 업데이트 필요")

conn.close()

print()
print("=" * 70)
print("확인 완료")
print("=" * 70)
