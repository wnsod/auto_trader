#!/usr/bin/env python
import sqlite3
from datetime import datetime

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("에피소드 상세 분석")
print("=" * 70)
print()

# 최근 에피소드 1개 선택
cursor.execute("""
    SELECT
        e.episode_id, e.ts_entry, e.entry_price, e.target_move_pct, e.horizon_k,
        e.predicted_dir, e.predicted_conf,
        s.ts_exit, s.first_event, s.t_hit, s.realized_ret_signed
    FROM rl_episodes e
    LEFT JOIN rl_episode_summary s ON e.episode_id = s.episode_id
    WHERE e.episode_id LIKE 'pred_%'
    ORDER BY e.ts_entry DESC
    LIMIT 5
""")

episodes = cursor.fetchall()

for ep in episodes:
    (ep_id, ts_entry, entry_price, target_move_pct, horizon_k,
     predicted_dir, predicted_conf,
     ts_exit, first_event, t_hit, realized_ret) = ep

    print(f"Episode: {ep_id[:70]}")
    print(f"  진입 시간: {datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  진입 가격: {entry_price:,.2f}")
    print(f"  예측 방향: {predicted_dir} (신뢰도: {predicted_conf:.2f})")
    print(f"  목표 변동: {target_move_pct:.4f}% (horizon: {horizon_k} 캔들)")

    if target_move_pct > 0:
        # TP/SL 가격 계산
        if predicted_dir == 1:  # Buy
            tp_price = entry_price * (1 + target_move_pct / 100)
            sl_price = entry_price * (1 - target_move_pct / 100)
        else:  # Sell or Hold
            tp_price = entry_price * (1 - target_move_pct / 100)
            sl_price = entry_price * (1 + target_move_pct / 100)

        print(f"  TP 가격: {tp_price:,.2f}")
        print(f"  SL 가격: {sl_price:,.2f}")

    if ts_exit:
        print(f"  종료 시간: {datetime.fromtimestamp(ts_exit).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  첫 이벤트: {first_event} (t={t_hit})")
        print(f"  실현 수익률: {realized_ret:.4f}")
    else:
        print(f"  ⚠️ 종료 데이터 없음")

    # horizon_k 검증
    duration_seconds = ts_exit - ts_entry if ts_exit else 0
    duration_candles = duration_seconds / 900  # 15분 = 900초

    if first_event == 'expiry':
        print(f"  ⚠️ 만료: {duration_candles:.1f} 캔들 경과 (목표: {horizon_k})")
        if target_move_pct < 0.01:
            print(f"  ⚠️ target_move_pct가 너무 작음: {target_move_pct:.6f}%")

    print()

# 통계 확인
cursor.execute("""
    SELECT
        AVG(target_move_pct) as avg_target,
        MIN(target_move_pct) as min_target,
        MAX(target_move_pct) as max_target,
        AVG(horizon_k) as avg_horizon
    FROM rl_episodes
    WHERE episode_id LIKE 'pred_%'
""")

avg_target, min_target, max_target, avg_horizon = cursor.fetchone()

print("=" * 70)
print("전체 에피소드 통계:")
print("=" * 70)
print(f"평균 목표 변동: {avg_target:.4f}%")
print(f"최소 목표 변동: {min_target:.4f}%")
print(f"최대 목표 변동: {max_target:.4f}%")
print(f"평균 horizon: {avg_horizon:.1f} 캔들")

conn.close()
