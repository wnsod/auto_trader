#!/usr/bin/env python
"""Check source coin_strategies performance data"""
import sys
sys.path.append('/workspace')

import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("Source coin_strategies 성과 데이터 확인")
print("=" * 70)
print()

# 성과가 있는 전략 확인
cursor.execute("""
    SELECT id, profit, win_rate, trades_count, quality_grade
    FROM coin_strategies
    WHERE profit > 0
    LIMIT 10
""")

strategies = cursor.fetchall()

if strategies:
    print(f"✅ 성과 데이터가 있는 전략: {len(strategies)}개")
    print()
    for row in strategies[:5]:
        print(f"  ID: {row[0][:50]}...")
        print(f"    Profit: {row[1]:.4f} ({row[1]*100:.2f}%)")
        print(f"    Win Rate: {row[2]:.4f} ({row[2]*100:.2f}%)")
        print(f"    Trades: {row[3]}")
        print(f"    Grade: {row[4]}")
        print()
else:
    print("❌ 성과 데이터가 있는 전략 없음")
    print()

# 전체 전략 수 확인
cursor.execute("SELECT COUNT(*) FROM coin_strategies")
total = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM coin_strategies WHERE profit > 0")
with_profit = cursor.fetchone()[0]

print(f"전체 전략: {total}개")
print(f"성과 있는 전략: {with_profit}개 ({with_profit/total*100:.1f}%)")

conn.close()
