#!/usr/bin/env python
"""DB에 있는 코인들 확인"""
import sys
sys.path.append('/workspace')

import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 코인별 전략 개수 확인
cursor.execute("""
    SELECT coin, COUNT(*) as strategy_count,
           COUNT(DISTINCT interval) as interval_count
    FROM rl_strategy_rollup
    GROUP BY coin
    ORDER BY coin
""")

print("=" * 70)
print("DB에 저장된 코인 데이터")
print("=" * 70)
print()

results = cursor.fetchall()

if results:
    print(f"{'코인':<10} {'전략 개수':>12} {'인터벌 개수':>12}")
    print("-" * 70)
    for coin, strategy_count, interval_count in results:
        print(f"{coin:<10} {strategy_count:>12} {interval_count:>12}")
    print()
    print(f"총 코인 개수: {len(results)}")
else:
    print("❌ 데이터가 없습니다!")

print()

# 인터벌별 확인 (LINK 예시)
print("=" * 70)
print("LINK 인터벌별 전략 개수")
print("=" * 70)
print()

cursor.execute("""
    SELECT interval, COUNT(*) as count
    FROM rl_strategy_rollup
    WHERE coin = 'LINK'
    GROUP BY interval
    ORDER BY interval
""")

for interval, count in cursor.fetchall():
    print(f"{interval:>6}: {count:>4}개")

conn.close()
