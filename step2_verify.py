#!/usr/bin/env python
"""
2단계 검증: 예측 Self-play 결과 확인
"""
import sqlite3
from datetime import datetime

COIN = 'LINK'
INTERVAL = '15m'

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("2단계 검증: 예측 Self-play")
print("=" * 70)
print()

# rl_episode_summary 테이블 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_episode_summary'")
has_summary = cursor.fetchone() is not None

if has_summary:
    cursor.execute("""
        SELECT COUNT(*) FROM rl_episode_summary
        WHERE episode_id LIKE ?
    """, (f"pred_{COIN}_{INTERVAL}_%",))
    total_episodes = cursor.fetchone()[0]
    print(f"📊 총 에피소드 (summary): {total_episodes}개")

    if total_episodes > 0:
        # 결과 분포
        cursor.execute("""
            SELECT
                first_event,
                COUNT(*) as count
            FROM rl_episode_summary
            WHERE episode_id LIKE ?
            GROUP BY first_event
            ORDER BY COUNT(*) DESC
        """, (f"pred_{COIN}_{INTERVAL}_%",))
        events = cursor.fetchall()
        print("\n결과 분포:")
        for event, count in events:
            print(f"  {event:10s}: {count:5d}개 ({count/total_episodes*100:.1f}%)")

        # 수익률 통계
        cursor.execute("""
            SELECT
                AVG(realized_ret_signed) as avg_ret,
                MIN(realized_ret_signed) as min_ret,
                MAX(realized_ret_signed) as max_ret
            FROM rl_episode_summary
            WHERE episode_id LIKE ?
        """, (f"pred_{COIN}_{INTERVAL}_%",))
        avg_ret, min_ret, max_ret = cursor.fetchone()

        print(f"\n수익률 통계:")
        if avg_ret is not None:
            print(f"  평균: {avg_ret:.4f} ({avg_ret*100:.2f}%)")
            print(f"  최소: {min_ret:.4f} ({min_ret*100:.2f}%)")
            print(f"  최대: {max_ret:.4f} ({max_ret*100:.2f}%)")
        else:
            print(f"  ⚠️ 수익률 데이터 없음 (NULL)")

        # horizon 통계
        cursor.execute("""
            SELECT AVG(t_hit), MIN(t_hit), MAX(t_hit)
            FROM rl_episode_summary
            WHERE episode_id LIKE ? AND t_hit IS NOT NULL
        """, (f"pred_{COIN}_{INTERVAL}_%",))
        avg_t, min_t, max_t = cursor.fetchone()

        if avg_t:
            print(f"\nHorizon (캔들 수) 통계:")
            print(f"  평균: {avg_t:.1f}캔들")
            print(f"  최소: {min_t}캔들, 최대: {max_t}캔들")

        # 샘플 에피소드 5개
        cursor.execute("""
            SELECT
                episode_id, ts_exit, first_event, t_hit, realized_ret_signed
            FROM rl_episode_summary
            WHERE episode_id LIKE ?
            LIMIT 5
        """, (f"pred_{COIN}_{INTERVAL}_%",))
        samples = cursor.fetchall()

        if samples:
            print(f"\n샘플 에피소드 5개:")
            for ep_id, ts_exit, first_event, t_hit, ret in samples:
                ep_name = ep_id[:50] + "..." if len(ep_id) > 50 else ep_id
                if ts_exit:
                    exit_time = datetime.fromtimestamp(ts_exit).strftime('%Y-%m-%d %H:%M')
                    print(f"  {ep_name}")
                    print(f"    종료: {exit_time}, 이벤트: {first_event}, t={t_hit}, 수익률: {ret:.4f}")
                else:
                    print(f"  {ep_name}: ts_exit NULL")

        print()
        print("✅ 2단계 검증 완료: 에피소드 생성 성공")

    else:
        print("❌ 에피소드가 생성되지 않았습니다 (summary 테이블)")

else:
    print("⚠️ rl_episode_summary 테이블이 없습니다.")

# rl_episodes 테이블도 확인
print()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_episodes'")
has_episodes = cursor.fetchone() is not None

if has_episodes:
    cursor.execute("SELECT COUNT(*) FROM rl_episodes WHERE coin=? AND interval=?", (COIN, INTERVAL))
    total_pred = cursor.fetchone()[0]
    print(f"📊 총 에피소드 (episodes): {total_pred}개")

    if total_pred > 0:
        # ts_entry 검증
        cursor.execute("""
            SELECT ts_entry, entry_price
            FROM rl_episodes
            WHERE coin=? AND interval=?
            ORDER BY ts_entry DESC
            LIMIT 5
        """, (COIN, INTERVAL))
        entries = cursor.fetchall()

        print("\n진입 시점 샘플:")
        now = int(datetime.now().timestamp())
        for ts_entry, entry_price in entries:
            entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M')
            days_ago = (now - ts_entry) / 86400
            print(f"  {entry_time} (ts={ts_entry}) @ {entry_price:,.0f}원")
            print(f"    {days_ago:.1f}일 전")

        # ts_entry 유효성 검증
        cursor.execute("""
            SELECT
                MIN(ts_entry) as min_ts,
                MAX(ts_entry) as max_ts,
                AVG(ts_entry) as avg_ts
            FROM rl_episodes
            WHERE coin=? AND interval=?
        """, (COIN, INTERVAL))
        min_ts, max_ts, avg_ts = cursor.fetchone()

        min_days_ago = (now - min_ts) / 86400
        max_days_ago = (now - max_ts) / 86400

        print(f"\nts_entry 범위:")
        print(f"  최소: {datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d')} ({min_days_ago:.1f}일 전)")
        print(f"  최대: {datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d')} ({max_days_ago:.1f}일 전)")

        if min_ts < 1000:
            print("  ❌ ts_entry가 너무 작음 (타임스탬프 오류)")
        elif max_days_ago < 0:
            print("  ❌ 미래 시간 오류")
        elif min_days_ago > 365:
            print("  ⚠️ 1년 이상 오래된 데이터")
        else:
            print("  ✅ ts_entry 유효성 정상")
else:
    print("⚠️ rl_episodes 테이블이 없습니다.")

conn.close()
