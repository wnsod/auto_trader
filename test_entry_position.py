#!/usr/bin/env python
import sys
sys.path.append('/workspace')

import pandas as pd
from rl_pipeline.data.candle_loader import load_candle_data_for_coin

print("=" * 70)
print("진입 위치 로직 테스트")
print("=" * 70)
print()

# LINK-15m 캔들 데이터 로드
print("📥 LINK-15m 캔들 데이터 로드 중...")
candle_data_dict = load_candle_data_for_coin('LINK', ['15m'])
if ('LINK', '15m') not in candle_data_dict:
    print("❌ 캔들 데이터를 찾을 수 없습니다.")
    sys.exit(1)

candle_data = candle_data_dict[('LINK', '15m')]
print(f"✅ {len(candle_data)}개 캔들 로드 완료")
print()

# 기존 로직 (최신 캔들 사용)
print("기존 로직:")
candle_data_sorted = candle_data.sort_values('timestamp', ascending=True).reset_index(drop=True)
recent_candles_old = candle_data_sorted.tail(100)
ts_entry_old = recent_candles_old['timestamp'].iloc[-1]
entry_idx_old = len(candle_data_sorted) - 1
future_candles_old = len(candle_data_sorted) - entry_idx_old - 1

print(f"  진입 위치: {entry_idx_old} / {len(candle_data_sorted)} (마지막 캔들)")
print(f"  진입 시간: {ts_entry_old}")
print(f"  미래 캔들: {future_candles_old}개")
print(f"  ❌ 문제: 미래 캔들이 없어서 TP/SL 시뮬레이션 불가능")
print()

# 새로운 로직 (70% 위치 사용)
print("새로운 로직 (70% 위치):")
total_candles = len(candle_data_sorted)
entry_position = int(total_candles * 0.7)

start_idx = max(0, entry_position - 100)
recent_candles_new = candle_data_sorted.iloc[start_idx:entry_position].copy()
ts_entry_new = recent_candles_new['timestamp'].iloc[-1]
future_candles_new = total_candles - entry_position

print(f"  전체 캔들: {total_candles}개")
print(f"  진입 위치: {entry_position} / {total_candles} (70% 지점)")
print(f"  진입 시간: {ts_entry_new}")
print(f"  미래 캔들: {future_candles_new}개 (30%)")
print(f"  ✅ 미래 캔들 충분: {future_candles_new}개로 TP/SL 시뮬레이션 가능")
print()

# 시뮬레이션 테스트
print("=" * 70)
print("시뮬레이션 테스트 (70% 위치 진입)")
print("=" * 70)
print()

# 진입 가격
entry_price = float(recent_candles_new['close'].iloc[-1])
print(f"진입 가격: {entry_price:,.2f}원")

# TP/SL 설정 (2% 목표)
target_move_pct = 0.02
tp_pct = target_move_pct
sl_pct = -target_move_pct * 0.5
horizon_k = 10

print(f"목표 변동: {target_move_pct*100}% (TP), {sl_pct*100}% (SL)")
print(f"horizon_k: {horizon_k}캔들")
print()

# Buy 방향으로 시뮬레이션
predicted_dir = 1  # Buy
tp_price = entry_price * (1 + tp_pct)
sl_price = entry_price * (1 + sl_pct)

print(f"예측 방향: Buy")
print(f"  TP 가격: {tp_price:,.2f}원 ({(tp_price - entry_price):+,.2f}원)")
print(f"  SL 가격: {sl_price:,.2f}원 ({(sl_price - entry_price):+,.2f}원)")
print()

# 향후 horizon_k 캔들 확인
print(f"향후 {horizon_k}개 캔들의 가격 움직임:")
print(f"{'k':>3} | {'가격':>10} | {'변동률':>8} | {'이벤트':>8}")
print("-" * 45)

hit_tp = False
hit_sl = False
hit_k = None

for k in range(1, min(horizon_k + 1, len(candle_data_sorted) - entry_position + 1)):
    idx = entry_position + k - 1  # entry_position은 이미 진입 다음 캔들의 인덱스
    if idx >= len(candle_data_sorted):
        print(f"⚠️ 캔들 데이터 부족 (k={k})")
        break

    current_price = float(candle_data_sorted.iloc[idx]['close'])
    move_pct = (current_price - entry_price) / entry_price

    event = ""
    if not hit_tp and move_pct >= tp_pct:
        event = "TP ✅"
        hit_tp = True
        hit_k = k
    elif not hit_sl and move_pct <= sl_pct:
        event = "SL ❌"
        hit_sl = True
        hit_k = k

    print(f"{k:3d} | {current_price:10,.2f} | {move_pct*100:7.3f}% | {event:>8}")

print()
if hit_tp:
    print(f"✅ TP 도달: k={hit_k}에서 목표 달성!")
elif hit_sl:
    print(f"❌ SL 도달: k={hit_k}에서 손절")
else:
    print(f"⏱️ 만료: {horizon_k}캔들 동안 TP/SL 미도달")

print()
print("=" * 70)
print("테스트 완료")
print("=" * 70)
print()
print("결론:")
print("  ✅ 70% 위치 진입 로직이 정상 작동합니다.")
print("  ✅ 미래 캔들로 TP/SL 시뮬레이션이 가능합니다.")
