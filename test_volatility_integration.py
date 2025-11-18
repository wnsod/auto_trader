#!/usr/bin/env python3
"""변동성 시스템 통합 테스트"""
import sys
sys.path.insert(0, '/workspace/trade')
from realtime_signal_selector import SignalSelector
import pandas as pd

print('=== 변동성 시스템 통합 테스트 ===\n')

# 1. Signal Selector 초기화
print('1. Signal Selector 초기화 중...')
selector = SignalSelector()
print('✅ 초기화 완료\n')

# 2. 각 코인의 변동성 그룹 확인
print('2. 코인별 변동성 그룹 확인:')
coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE']
for coin in coins:
    vol_group = selector.get_coin_volatility_group(coin)
    print(f'   {coin}: {vol_group}')
print()

# 3. 변동성 기반 가중치 테스트
print('3. 변동성 기반 동적 가중치 테스트:')
test_market_conditions = ['bull_market', 'bear_market', 'sideways_market']
for coin in ['BTC', 'ADA', 'DOGE']:
    for market_condition in test_market_conditions:
        weights = selector.get_volatility_based_weights(coin, market_condition, False)
        vol_group = selector.get_coin_volatility_group(coin)
        print(f'   {coin} ({vol_group}) / {market_condition}:')
        total = sum(weights.values())
        print(f'      base={weights["base"]:.3f}, dna={weights["dna"]:.3f}, rl={weights["rl"]:.3f}, integrated={weights["integrated"]:.3f}')
        print(f'      합계: {total:.3f}')
print()

# 4. 변동성 기반 임계값 테스트
print('4. 변동성 기반 동적 임계값 테스트:')
for coin in ['BTC', 'ADA', 'DOGE']:
    thresholds = selector.get_volatility_based_thresholds(coin)
    vol_group = selector.get_coin_volatility_group(coin)
    print(f'   {coin} ({vol_group}):')
    print(f'      BUY: >{thresholds["weak_buy"]:.2f} (약) / >{thresholds["strong_buy"]:.2f} (강)')
    print(f'      SELL: <{thresholds["weak_sell"]:.2f} (약) / <{thresholds["strong_sell"]:.2f} (강)')
print()

print('✅ 모든 테스트 완료!')
