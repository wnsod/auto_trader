#!/usr/bin/env python
import sqlite3
from datetime import datetime

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("최근 예측 self-play 에피소드 검증")
print("=" * 70)
print()

# rl_episodes에서 ts_entry 확인
cursor.execute("""
    SELECT episode_id, ts_entry, entry_price, coin, interval
    FROM rl_episodes
    WHERE episode_id LIKE 'pred_%'
    ORDER BY ts_entry DESC
    LIMIT 10
""")

episodes = cursor.fetchall()
print(f"📊 총 {len(episodes)}개 에피소드 발견")
print()

for ep_id, ts_entry, entry_price, coin, interval in episodes:
    # 타임스탬프 검증
    if ts_entry and ts_entry > 1000:
        entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M:%S')
        days_ago = (datetime.now().timestamp() - ts_entry) / 86400

        status = "✅"
        if ts_entry < 1000:
            status = "❌ (너무 작음)"
        elif days_ago > 365:
            status = "❌ (너무 오래됨)"
        elif days_ago < 0:
            status = "❌ (미래)"
        elif days_ago > 7:
            status = "⚠️ (조금 오래됨)"

        print(f"{status} {coin}-{interval}")
        print(f"  Episode: {ep_id[:60]}...")
        print(f"  진입 시간: {entry_time} ({days_ago:.1f}일 전)")
        print(f"  진입 가격: {entry_price:,.0f}")
        print(f"  ts_entry: {ts_entry}")
    else:
        print(f"❌ {coin}-{interval}")
        print(f"  Episode: {ep_id[:60]}...")
        print(f"  ts_entry: {ts_entry} (오류!)")
    print()

# rl_episode_summary에서 결과 확인
print("=" * 70)
print("에피소드 결과 요약")
print("=" * 70)
print()

cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN first_event = 'TP' THEN 1 ELSE 0 END) as tp_count,
        SUM(CASE WHEN first_event = 'SL' THEN 1 ELSE 0 END) as sl_count,
        SUM(CASE WHEN first_event = 'expiry' THEN 1 ELSE 0 END) as expiry_count,
        AVG(realized_ret_signed) as avg_ret
    FROM rl_episode_summary
    WHERE episode_id LIKE 'pred_%'
""")

total, tp, sl, expiry, avg_ret = cursor.fetchone()

if total and total > 0:
    print(f"전체 에피소드: {total}개")
    print(f"  TP: {tp}개 ({tp/total*100:.1f}%)")
    print(f"  SL: {sl}개 ({sl/total*100:.1f}%)")
    print(f"  만료: {expiry}개 ({expiry/total*100:.1f}%)")
    print(f"  평균 수익률: {avg_ret:.4f}" if avg_ret else "  평균 수익률: N/A")
else:
    print("에피소드 없음")

conn.close()
