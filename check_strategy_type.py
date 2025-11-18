#!/usr/bin/env python
"""전략 타입 확인"""
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 테이블 목록 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print('📋 rl_strategies.db 테이블 목록:')
for table in tables:
    print(f'  - {table[0]}')
print()

# coin_strategies 테이블이 있으면 strategy_type 확인
if any('coin_strategies' in str(t) for t in tables):
    print('=' * 80)
    print('ADA 전략 타입별 분포:')
    print('=' * 80)

    cursor.execute('''
        SELECT interval, strategy_type, COUNT(*) as count
        FROM coin_strategies
        WHERE coin = 'ADA'
        GROUP BY interval, strategy_type
        ORDER BY interval, count DESC
    ''')

    current_interval = None
    for row in cursor.fetchall():
        interval, strategy_type, count = row

        if interval != current_interval:
            print(f'\n{interval}:')
            current_interval = interval

        print(f'  - {strategy_type}: {count}개')

    print()
    print('=' * 80)
    print('샘플 전략 strategy_type 확인 (15m, 처음 10개):')
    print('=' * 80)

    cursor.execute('''
        SELECT id, strategy_type, rsi_min, rsi_max
        FROM coin_strategies
        WHERE coin = 'ADA' AND interval = '15m'
        LIMIT 10
    ''')

    for row in cursor.fetchall():
        id, strategy_type, rsi_min, rsi_max = row
        rsi_mid = (rsi_min + rsi_max) / 2.0 if rsi_min and rsi_max else 50.0
        print(f'{id}:')
        print(f'  - strategy_type: {strategy_type}')
        print(f'  - RSI 범위: {rsi_min:.1f} ~ {rsi_max:.1f} (중앙: {rsi_mid:.1f})')
        print()

conn.close()
