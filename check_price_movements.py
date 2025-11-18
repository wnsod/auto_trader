#!/usr/bin/env python
import sys
sys.path.append('/workspace')

import sqlite3
import pandas as pd
from rl_pipeline.data.candle_loader import load_candle_data_for_coin

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 최근 에피소드 1개 선택
cursor.execute("""
    SELECT
        e.episode_id, e.ts_entry, e.entry_price, e.target_move_pct, e.horizon_k,
        e.predicted_dir, e.coin, e.interval,
        s.first_event, s.t_hit, s.realized_ret_signed
    FROM rl_episodes e
    LEFT JOIN rl_episode_summary s ON e.episode_id = s.episode_id
    WHERE e.episode_id LIKE 'pred_%' AND e.interval = '15m'
    ORDER BY e.ts_entry DESC
    LIMIT 1
""")

ep = cursor.fetchone()
if not ep:
    print("에피소드를 찾을 수 없습니다.")
    sys.exit(1)

(ep_id, ts_entry, entry_price, target_move_pct, horizon_k,
 predicted_dir, coin, interval,
 first_event, t_hit, realized_ret) = ep

print("=" * 70)
print(f"에피소드: {ep_id}")
print("=" * 70)
print(f"코인: {coin}-{interval}")
print(f"진입 시간: {ts_entry}")
print(f"진입 가격: {entry_price:,.2f}원")
print(f"예측 방향: {predicted_dir} (1=Buy, -1=Sell, 0=Hold)")
print(f"목표 변동: {target_move_pct} (decimal)")
print(f"horizon_k: {horizon_k}")
print(f"결과: {first_event} (t={t_hit})")
print()

# TP/SL 계산
tp_pct = target_move_pct  # 2% = 0.02
sl_pct = -target_move_pct * 0.5  # -1% = -0.01

print("TP/SL 설정:")
print(f"  tp_pct: {tp_pct} ({tp_pct*100:.2f}%)")
print(f"  sl_pct: {sl_pct} ({sl_pct*100:.2f}%)")

if predicted_dir == 1:
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 + sl_pct)
elif predicted_dir == -1:
    tp_price = entry_price * (1 - tp_pct)
    sl_price = entry_price * (1 - sl_pct)
else:
    tp_price = entry_price
    sl_price = entry_price

print(f"  TP 가격: {tp_price:,.2f}원 ({tp_price - entry_price:+.2f}원)")
print(f"  SL 가격: {sl_price:,.2f}원 ({sl_price - entry_price:+.2f}원)")
print()

# 캔들 데이터 로드
print("캔들 데이터 분석 중...")
candle_data_dict = load_candle_data_for_coin(coin, [interval])
if (coin, interval) not in candle_data_dict:
    print(f"❌ 캔들 데이터를 찾을 수 없습니다: {coin}-{interval}")
    sys.exit(1)

candle_data = candle_data_dict[(coin, interval)]
print(f"캔들 데이터: {len(candle_data)}개")

# 진입 시점 찾기
entry_candle_matches = candle_data[candle_data['timestamp'] == pd.Timestamp(ts_entry, unit='s')]
if len(entry_candle_matches) == 0:
    print(f"⚠️ 정확한 진입 시점을 찾을 수 없습니다 (ts={ts_entry})")
    # 가장 가까운 시점 찾기
    candle_data_sorted = candle_data.sort_values('timestamp').reset_index(drop=True)
    candle_data_sorted['ts_unix'] = candle_data_sorted['timestamp'].apply(lambda x: int(x.timestamp()))
    candle_data_sorted['ts_diff'] = abs(candle_data_sorted['ts_unix'] - ts_entry)
    entry_idx = candle_data_sorted['ts_diff'].idxmin()
    print(f"가장 가까운 캔들 사용: idx={entry_idx}, ts={candle_data_sorted.iloc[entry_idx]['ts_unix']}")
else:
    candle_data_sorted = candle_data.sort_values('timestamp').reset_index(drop=True)
    entry_idx = candle_data_sorted[candle_data_sorted['timestamp'] == pd.Timestamp(ts_entry, unit='s')].index[0]
    print(f"✅ 진입 캔들 발견: idx={entry_idx}")

print()

# horizon_k 범위 내 가격 움직임 분석
print(f"향후 {horizon_k}개 캔들의 가격 움직임:")
print(f"{'k':>3} | {'가격':>10} | {'변동률':>8} | {'TP도달':>6} | {'SL도달':>6}")
print("-" * 50)

hit_tp = False
hit_sl = False
hit_k = None

for k in range(1, min(horizon_k + 5, len(candle_data_sorted) - entry_idx)):
    if entry_idx + k >= len(candle_data_sorted):
        print(f"⚠️ 캔들 데이터 부족 (k={k}에서 종료)")
        break

    current_candle = candle_data_sorted.iloc[entry_idx + k]
    current_price = float(current_candle['close'])

    # 수익률 계산
    if predicted_dir == 1:  # Buy
        move_pct = (current_price - entry_price) / entry_price
    elif predicted_dir == -1:  # Sell
        move_pct = (entry_price - current_price) / entry_price
    else:  # Hold
        move_pct = abs(current_price - entry_price) / entry_price

    # TP/SL 체크
    tp_hit = "✅" if move_pct >= tp_pct else ""
    sl_hit = "❌" if move_pct <= sl_pct else ""

    if not hit_tp and move_pct >= tp_pct:
        hit_tp = True
        hit_k = k
        print(f"{k:3d} | {current_price:10,.2f} | {move_pct*100:7.3f}% | {tp_hit:^6} | {sl_hit:^6} ← TP HIT!")
    elif not hit_sl and move_pct <= sl_pct:
        hit_sl = True
        hit_k = k
        print(f"{k:3d} | {current_price:10,.2f} | {move_pct*100:7.3f}% | {tp_hit:^6} | {sl_hit:^6} ← SL HIT!")
    elif k <= 10 or k == horizon_k:  # 처음 10개와 horizon_k만 표시
        print(f"{k:3d} | {current_price:10,.2f} | {move_pct*100:7.3f}% | {tp_hit:^6} | {sl_hit:^6}")

print()
if hit_tp:
    print(f"✅ TP 도달: k={hit_k}")
elif hit_sl:
    print(f"❌ SL 도달: k={hit_k}")
else:
    print(f"⏱️ 만료: TP/SL 도달 없음")

print()
print(f"데이터베이스 결과: {first_event} (t={t_hit})")

conn.close()
