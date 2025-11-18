#!/usr/bin/env python
"""개선된 전략 방향 분류 테스트"""
import sys
sys.path.append('/workspace')

import sqlite3

print("=" * 80)
print("개선된 전략 방향 분류 테스트")
print("=" * 80)
print()

db_path = '/workspace/data_storage/rl_strategies.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# ADA 전략 조회
cursor.execute("""
    SELECT id, coin, interval, rsi_min, rsi_max, macd_buy_threshold, macd_sell_threshold,
           pattern_source, volume_ratio_min, mfi_min, mfi_max, adx_min, stop_loss_pct, take_profit_pct
    FROM coin_strategies
    WHERE coin = 'ADA'
    ORDER BY interval, id
    LIMIT 200
""")

strategies = cursor.fetchall()

print(f"📊 총 {len(strategies)}개 ADA 전략 로드")
print()

# 개선된 전략 방향 분류 로직
def classify_strategy_direction_improved(strategy_row):
    """전략 방향 분류 (개선 버전)"""
    (id, coin, interval, rsi_min, rsi_max, macd_buy, macd_sell, pattern_source,
     volume_ratio_min, mfi_min, mfi_max, adx_min, stop_loss_pct, take_profit_pct) = strategy_row

    buy_score = 0.0
    sell_score = 0.0

    # 1. 전략 ID/이름 기반
    if 'oversold' in id.lower():
        buy_score += 0.8
    elif 'overbought' in id.lower():
        sell_score += 0.8
    elif 'buy' in id.lower():
        buy_score += 0.5
    elif 'sell' in id.lower():
        sell_score += 0.5

    # 2. RSI 중앙값 기반
    rsi_midpoint = (rsi_min + rsi_max) / 2.0 if rsi_min and rsi_max else 50.0
    rsi_range = (rsi_max - rsi_min) if rsi_min and rsi_max else 40.0

    if rsi_midpoint < 50:
        buy_score += (50 - rsi_midpoint) / 50.0
    elif rsi_midpoint > 50:
        sell_score += (rsi_midpoint - 50) / 50.0

    # RSI 범위 특화
    if rsi_range < 30:
        specialization_bonus = (30 - rsi_range) / 30.0 * 0.3
        if rsi_midpoint < 50:
            buy_score += specialization_bonus
        else:
            sell_score += specialization_bonus

    # 극단적 RSI
    if rsi_min and rsi_min < 30:
        buy_score += (30 - rsi_min) / 30.0 * 0.5
    if rsi_max and rsi_max > 70:
        sell_score += (rsi_max - 70) / 30.0 * 0.5

    # 3. MACD
    if macd_buy and macd_buy > 0:
        buy_score += min(macd_buy * 10, 0.5)
    if macd_sell and macd_sell < 0:
        sell_score += min(abs(macd_sell) * 10, 0.5)

    if macd_buy and macd_sell:
        macd_diff = macd_buy - macd_sell
        if macd_diff > 0.02:
            buy_score += 0.2
        elif macd_diff < -0.02:
            sell_score += 0.2

    # 4. MFI
    if mfi_min and mfi_max:
        mfi_midpoint = (mfi_min + mfi_max) / 2.0
        if mfi_midpoint < 50:
            buy_score += (50 - mfi_midpoint) / 100.0
        elif mfi_midpoint > 50:
            sell_score += (mfi_midpoint - 50) / 100.0

    # 5. 최종 분류 (임계값 0.05)
    score_diff = abs(buy_score - sell_score)

    if buy_score > sell_score and score_diff > 0.05:
        return 'buy', buy_score, sell_score
    elif sell_score > buy_score and score_diff > 0.05:
        return 'sell', buy_score, sell_score
    else:
        # RSI 중앙값으로 최종 결정
        if rsi_midpoint < 48:
            return 'buy', buy_score, sell_score
        elif rsi_midpoint > 52:
            return 'sell', buy_score, sell_score
        else:
            return 'neutral', buy_score, sell_score

# 인터벌별 분류
interval_classification = {}

for strategy in strategies:
    interval = strategy[2]
    classified, buy_score, sell_score = classify_strategy_direction_improved(strategy)

    if interval not in interval_classification:
        interval_classification[interval] = {'buy': 0, 'sell': 0, 'neutral': 0, 'total': 0}

    interval_classification[interval][classified] += 1
    interval_classification[interval]['total'] += 1

print("인터벌별 전략 방향 분류 (개선 버전):")
print("-" * 80)

for interval in sorted(interval_classification.keys()):
    stats = interval_classification[interval]
    total = stats['total']

    print(f"\n{interval}:")
    print(f"  - 총 전략: {total}개")
    print(f"  - BUY 전략: {stats['buy']}개 ({stats['buy']/total*100:.1f}%)")
    print(f"  - SELL 전략: {stats['sell']}개 ({stats['sell']/total*100:.1f}%)")
    print(f"  - NEUTRAL 전략: {stats['neutral']}개 ({stats['neutral']/total*100:.1f}%)")

# 샘플 상세 분석
print("\n" + "=" * 80)
print("샘플 전략 상세 (15m 인터벌, 처음 10개):")
print("-" * 80)

cursor.execute("""
    SELECT id, rsi_min, rsi_max, macd_buy_threshold, macd_sell_threshold,
           pattern_source, volume_ratio_min, mfi_min, mfi_max
    FROM coin_strategies
    WHERE coin = 'ADA' AND interval = '15m'
    LIMIT 10
""")

samples = cursor.fetchall()

for i, sample in enumerate(samples, 1):
    id, rsi_min, rsi_max, macd_buy, macd_sell, pattern_source, vol_min, mfi_min, mfi_max = sample

    # Full row 구성
    full_row = (id, 'ADA', '15m', rsi_min, rsi_max, macd_buy, macd_sell, pattern_source,
                vol_min, mfi_min, mfi_max, 15.0, 0.02, 0.04)  # 기본값 사용

    classified, buy_score, sell_score = classify_strategy_direction_improved(full_row)

    rsi_mid = (rsi_min + rsi_max) / 2.0 if rsi_min and rsi_max else 50.0

    print(f"\n전략 #{i} (ID: {id}):")
    print(f"  - RSI 범위: {rsi_min:.1f} ~ {rsi_max:.1f} (중앙: {rsi_mid:.1f})")
    print(f"  - MACD 임계값: buy={macd_buy:.3f}, sell={macd_sell:.3f}")
    print(f"  - Pattern Source: {pattern_source}")
    print(f"  - 점수: BUY={buy_score:.3f}, SELL={sell_score:.3f}")
    print(f"  - 분류 결과: {classified.upper()}")

conn.close()

print("\n" + "=" * 80)
print("✅ 분석 완료!")
print("=" * 80)
