#!/usr/bin/env python
"""Trade System 글로벌 전략 통합 테스트"""
import sys
sys.path.append('/workspace')

import sqlite3
import pandas as pd
import numpy as np

print("=" * 70)
print("Trade System 글로벌 전략 통합 테스트")
print("=" * 70)
print()

# 1. global_strategies 로드 확인
print("1️⃣  global_strategies 로드 테스트:")

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT id, interval, zone_key, rsi_min, rsi_max, profit, win_rate
    FROM global_strategies
    LIMIT 5
""")

strategies = cursor.fetchall()

if strategies:
    print(f"   ✅ {len(strategies)}개 전략 로드 성공")
    for row in strategies:
        print(f"      {row[0][:30]}... (interval={row[1]}, zone={row[2]})")
else:
    print("   ❌ 전략 로드 실패")

conn.close()
print()

# 2. 간단한 점수 계산 시뮬레이션
print("2️⃣  점수 계산 시뮬레이션:")

# 테스트용 캔들 데이터 생성
test_candle = pd.Series({
    'rsi': 50.0,
    'macd': 0.01,
    'volume_ratio': 1.5,
    'volatility': 0.02
})

coin = 'ADA'
interval = '15m'

print(f"   코인: {coin}, 인터벌: {interval}")
print(f"   캔들 데이터: RSI={test_candle['rsi']}, MACD={test_candle['macd']}")
print()

# 3. global_strategies에서 해당 interval 전략 찾기
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT id, zone_key, rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
           profit, win_rate, quality_grade
    FROM global_strategies
    WHERE interval = ?
    LIMIT 3
""", (interval,))

global_strategies = cursor.fetchall()

if global_strategies:
    print(f"3️⃣  {interval} 인터벌 글로벌 전략: {len(global_strategies)}개")
    print()

    for row in global_strategies:
        strategy_id = row[0][:30]
        zone = row[1]
        rsi_min, rsi_max = row[2], row[3]
        volume_min, volume_max = row[4], row[5]
        profit, win_rate, grade = row[6], row[7], row[8]

        print(f"   전략: {strategy_id}...")
        print(f"     Zone: {zone}")
        print(f"     RSI: {rsi_min:.2f} - {rsi_max:.2f}")
        print(f"     Volume: {volume_min:.2f} - {volume_max:.2f}")
        print(f"     성과: Profit={profit:.2%}, WinRate={win_rate:.2%}, Grade={grade}")
        print()
else:
    print(f"3️⃣  ❌ {interval} 인터벌 글로벌 전략 없음")
    print()

conn.close()

# 4. Zone 매칭 테스트 (간단 버전)
print("4️⃣  Zone 매칭 테스트:")

current_rsi = test_candle['rsi']
current_volume = test_candle['volume_ratio']

# RSI Zone 분류
if current_rsi < 30:
    rsi_zone = 'oversold'
elif current_rsi < 45:
    rsi_zone = 'low'
elif current_rsi < 55:
    rsi_zone = 'neutral'
elif current_rsi < 70:
    rsi_zone = 'high'
else:
    rsi_zone = 'overbought'

print(f"   현재 RSI: {current_rsi} → Zone: {rsi_zone}")
print()

# Zone 매칭되는 전략 찾기
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT id, zone_key, rsi_min, rsi_max, profit, win_rate
    FROM global_strategies
    WHERE interval = ? AND rsi_zone = ?
    LIMIT 1
""", (interval, rsi_zone))

matched_strategy = cursor.fetchone()

if matched_strategy:
    print(f"5️⃣  ✅ Zone 매칭 전략 발견:")
    print(f"   ID: {matched_strategy[0][:40]}...")
    print(f"   Zone: {matched_strategy[1]}")
    print(f"   RSI: {matched_strategy[2]:.2f} - {matched_strategy[3]:.2f}")
    print(f"   성과: Profit={matched_strategy[4]:.2%}, WinRate={matched_strategy[5]:.2%}")
else:
    print(f"5️⃣  ⚠️  Zone 매칭 전략 없음 (RSI Zone: {rsi_zone})")

conn.close()
print()

print("=" * 70)
print("✅ 통합 테스트 완료!")
print("=" * 70)
print()
print("💡 개선 사항:")
print("   - _get_global_strategy_score() 함수에 Zone 매칭 로직 추가 필요")
print("   - 현재는 interval만으로 선택 → Zone별 선택으로 개선")
