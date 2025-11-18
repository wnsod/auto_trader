#!/usr/bin/env python3
"""Combined 시그널 변동성 임계값 적용 검증"""
import sys
sys.path.insert(0, '/workspace/trade')
from realtime_signal_selector import SignalSelector, SignalData, SignalAction
from datetime import datetime

print('=' * 80)
print('Combined 시그널 변동성 임계값 적용 검증')
print('=' * 80)

# 1. Signal Selector 초기화
print('\n1. Signal Selector 초기화...')
selector = SignalSelector()
print('✅ 초기화 완료')

# 2. 테스트 시나리오 준비
print('\n2. 테스트 시나리오 준비')
print('-' * 80)

test_scenarios = [
    # (코인, 변동성그룹, signal_score, 예상_action)
    ('BTC', 'LOW', 0.0013, 'HOLD'),     # BTC threshold: ±0.3
    ('BTC', 'LOW', 0.35, 'BUY'),        # > 0.3
    ('BTC', 'LOW', -0.35, 'SELL'),      # < -0.3

    ('ADA', 'HIGH', 0.0013, 'HOLD'),    # ADA threshold: ±0.15
    ('ADA', 'HIGH', 0.18, 'BUY'),       # > 0.15
    ('ADA', 'HIGH', -0.18, 'SELL'),     # < -0.15

    ('DOGE', 'VERY_HIGH', 0.0013, 'HOLD'),  # DOGE threshold: ±0.1
    ('DOGE', 'VERY_HIGH', 0.12, 'BUY'),     # > 0.1
    ('DOGE', 'VERY_HIGH', -0.12, 'SELL'),   # < -0.1
]

# 3. 더미 시그널 생성 및 테스트
print('\n3. Combined 시그널 생성 및 액션 검증')
print('-' * 80)

for coin, expected_vol_group, test_score, expected_action in test_scenarios:
    # 변동성 그룹 확인
    vol_group = selector.get_coin_volatility_group(coin)
    thresholds = selector.get_volatility_based_thresholds(coin)

    # 더미 interval 시그널 생성 (15m, 30m)
    dummy_signal_15m = SignalData(
        coin=coin,
        interval='15m',
        action=SignalAction.HOLD,
        confidence=0.7,
        signal_score=test_score,
        price=100.0,
        timestamp=datetime.now(),
        indicators={'rsi': 50, 'macd': 0, 'volatility': 0.01},
        base_score=0.5,
        dna_score=0.5,
        rl_score=0.5,
        integrated_analysis_score=0.5
    )

    dummy_signal_30m = SignalData(
        coin=coin,
        interval='30m',
        action=SignalAction.HOLD,
        confidence=0.7,
        signal_score=test_score,
        price=100.0,
        timestamp=datetime.now(),
        indicators={'rsi': 50, 'macd': 0, 'volatility': 0.01},
        base_score=0.5,
        dna_score=0.5,
        rl_score=0.5,
        integrated_analysis_score=0.5
    )

    # Combined 시그널 생성
    interval_signals = {
        '15m': dummy_signal_15m,
        '30m': dummy_signal_30m
    }

    combined_signal = selector.combine_interval_signals(coin, interval_signals)

    # 결과 검증
    actual_action = combined_signal.action.name
    status = '✅' if actual_action == expected_action else '❌'

    print(f'{status} {coin} ({vol_group}):')
    print(f'   Signal Score: {test_score:+.4f}')
    print(f'   Thresholds: BUY>{thresholds["weak_buy"]:.2f}, SELL<{thresholds["weak_sell"]:.2f}')
    print(f'   Expected: {expected_action}, Actual: {actual_action}')

    if actual_action != expected_action:
        print(f'   ⚠️ 예상과 다름!')
    print()

print('=' * 80)
print('검증 완료!')
print('=' * 80)
