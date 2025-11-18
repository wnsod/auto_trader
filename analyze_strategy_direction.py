#!/usr/bin/env python
"""전략 방향 분류 분석"""
import sys
sys.path.append('/workspace')

import sqlite3
import json

print("=" * 80)
print("전략 방향 분류 분석")
print("=" * 80)
print()

db_path = '/workspace/data_storage/rl_strategies.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# ADA 전략 조회
cursor.execute("""
    SELECT id, coin, interval, rsi_min, rsi_max, macd_buy_threshold, macd_sell_threshold,
           pattern_source
    FROM coin_strategies
    WHERE coin = 'ADA'
    ORDER BY interval, id
    LIMIT 200
""")

strategies = cursor.fetchall()

print(f"📊 총 {len(strategies)}개 ADA 전략 로드")
print()

# 전략 방향 분류 로직 (코드에서 복사)
def classify_strategy_direction(strategy_row):
    """전략 방향 분류"""
    id, coin, interval, rsi_min, rsi_max, macd_buy, macd_sell, pattern_source = strategy_row

    # 1. 명시적 방향성 특화 전략 (현재는 direction 컬럼 없음)
    # pattern_source만 확인

    # 2. 파라미터 기반 분류
    buy_score = 0.0
    sell_score = 0.0

    # RSI 기준
    if rsi_min and rsi_min < 35:
        buy_score = 1.0 - (rsi_min / 35.0)

    if rsi_max and rsi_max > 65:
        sell_score = (rsi_max - 65.0) / 25.0

    # MACD 기준
    if macd_buy and macd_buy > 0:
        buy_score += 0.3
    if macd_sell and macd_sell < 0:
        sell_score += 0.3

    # 최종 분류
    if buy_score > sell_score + 0.2:
        return 'buy'
    elif sell_score > buy_score + 0.2:
        return 'sell'
    else:
        return 'neutral'

# 인터벌별 분류
interval_classification = {}

for strategy in strategies:
    interval = strategy[2]
    classified = classify_strategy_direction(strategy)

    if interval not in interval_classification:
        interval_classification[interval] = {'buy': 0, 'sell': 0, 'neutral': 0, 'total': 0}

    interval_classification[interval][classified] += 1
    interval_classification[interval]['total'] += 1

print("인터벌별 전략 방향 분류:")
print("-" * 80)

for interval in sorted(interval_classification.keys()):
    stats = interval_classification[interval]
    total = stats['total']

    print(f"\n{interval}:")
    print(f"  - 총 전략: {total}개")
    print(f"  - BUY 전략: {stats['buy']}개 ({stats['buy']/total*100:.1f}%)")
    print(f"  - SELL 전략: {stats['sell']}개 ({stats['sell']/total*100:.1f}%)")
    print(f"  - NEUTRAL 전략: {stats['neutral']}개 ({stats['neutral']/total*100:.1f}%)")

# 샘플 전략 상세 분석
print("\n" + "=" * 80)
print("샘플 전략 상세 (15m 인터벌, 처음 10개):")
print("-" * 80)

cursor.execute("""
    SELECT id, rsi_min, rsi_max, macd_buy_threshold, macd_sell_threshold,
           pattern_source
    FROM coin_strategies
    WHERE coin = 'ADA' AND interval = '15m'
    LIMIT 10
""")

samples = cursor.fetchall()

for i, sample in enumerate(samples, 1):
    id, rsi_min, rsi_max, macd_buy, macd_sell, pattern_source = sample
    full_row = (id, 'ADA', '15m', rsi_min, rsi_max, macd_buy, macd_sell, pattern_source)
    classified = classify_strategy_direction(full_row)

    print(f"\n전략 #{i} (ID: {id}):")
    print(f"  - RSI 범위: {rsi_min:.1f} ~ {rsi_max:.1f}")
    print(f"  - MACD 임계값: buy={macd_buy:.3f}, sell={macd_sell:.3f}")
    print(f"  - Pattern Source: {pattern_source}")
    print(f"  - 분류 결과: {classified.upper()}")

conn.close()

print("\n" + "=" * 80)
print("✅ 분석 완료!")
print("=" * 80)
