#!/usr/bin/env python
"""240m 인터벌 0% 정확도 문제 분석"""
import sqlite3
import json

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 240m 전략들의 strategy_type 분포 확인
print('=' * 80)
print('240m 전략 타입 분포:')
print('=' * 80)

cursor.execute('''
    SELECT strategy_type, COUNT(*) as count
    FROM coin_strategies
    WHERE coin = 'ADA' AND interval = '240m'
    GROUP BY strategy_type
    ORDER BY count DESC
''')

for row in cursor.fetchall():
    strategy_type, count = row
    print(f"{strategy_type}: {count}개")

print()

# 240m 샘플 전략 10개 확인
print('=' * 80)
print('240m 샘플 전략 (처음 10개):')
print('=' * 80)

cursor.execute('''
    SELECT id, strategy_type, rsi_min, rsi_max, macd_buy_threshold, macd_sell_threshold
    FROM coin_strategies
    WHERE coin = 'ADA' AND interval = '240m'
    LIMIT 10
''')

for row in cursor.fetchall():
    id, strategy_type, rsi_min, rsi_max, macd_buy, macd_sell = row
    rsi_mid = (rsi_min + rsi_max) / 2.0 if rsi_min and rsi_max else 50.0
    print(f"{id}:")
    print(f"  - strategy_type: {strategy_type}")
    print(f"  - RSI: {rsi_min:.1f} ~ {rsi_max:.1f} (중앙: {rsi_mid:.1f})")
    print(f"  - MACD: buy={macd_buy:.3f}, sell={macd_sell:.3f}")
    print()

# 240m 캔들 데이터 개수 확인
print('=' * 80)
print('240m 캔들 데이터 개수:')
print('=' * 80)

conn_candles = sqlite3.connect('/workspace/data_storage/rl_candles.db')
cursor_candles = conn_candles.cursor()

cursor_candles.execute('''
    SELECT COUNT(*) as count
    FROM candles
    WHERE coin = 'ADA' AND interval = '240m'
''')

candle_count = cursor_candles.fetchone()[0]
print(f"ADA 240m 캔들: {candle_count}개")
print()

conn_candles.close()
conn.close()

print('=' * 80)
print('✅ 분석 완료')
print('=' * 80)
