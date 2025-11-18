#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# LINK 전략들의 sharpe_ratio 확인
cursor.execute("""
    SELECT interval, COUNT(*) as total,
           SUM(CASE WHEN avg_sharpe_ratio IS NOT NULL AND avg_sharpe_ratio != 0 THEN 1 ELSE 0 END) as non_zero,
           AVG(avg_sharpe_ratio) as avg_sharpe,
           MAX(avg_sharpe_ratio) as max_sharpe,
           MIN(avg_sharpe_ratio) as min_sharpe
    FROM rl_strategy_rollup
    WHERE coin = 'LINK'
    GROUP BY interval
""")

print("LINK Sharpe Ratio 데이터:")
print("=" * 70)
for row in cursor.fetchall():
    interval, total, non_zero, avg_sharpe, max_sharpe, min_sharpe = row
    print(f"{interval}:")
    print(f"  전체: {total}개")
    print(f"  Non-zero: {non_zero}개")
    print(f"  평균: {avg_sharpe}")
    print(f"  최대: {max_sharpe}")
    print(f"  최소: {min_sharpe}")
    print()

# 샘플 데이터 확인
print("=" * 70)
print("샘플 데이터 (15m, 처음 5개):")
print("=" * 70)
cursor.execute("""
    SELECT strategy_id, avg_ret, win_rate, avg_sharpe_ratio
    FROM rl_strategy_rollup
    WHERE coin = 'LINK' AND interval = '15m'
    LIMIT 5
""")

for row in cursor.fetchall():
    sid, avg_ret, win_rate, sharpe = row
    print(f"전략: {sid[:50]}...")
    print(f"  avg_ret: {avg_ret}")
    print(f"  win_rate: {win_rate}")
    print(f"  sharpe: {sharpe}")
    print()

conn.close()
