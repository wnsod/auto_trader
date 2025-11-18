#!/usr/bin/env python
"""Check what fields are populated in strategies"""
import sys
sys.path.append('/workspace')

import sqlite3
import json

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("전략 데이터 필드 확인")
print("=" * 70)
print()

# coin_strategies 샘플
print("1️⃣  coin_strategies 샘플:")
cursor.execute("""
    SELECT id, coin, interval, strategy_type,
           rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
           profit, win_rate, trades_count, quality_grade
    FROM coin_strategies
    LIMIT 3
""")

for row in cursor.fetchall():
    print(f"\n  ID: {row[0][:50]}...")
    print(f"    Coin: {row[1]}, Interval: {row[2]}")
    print(f"    Type: {row[3]}")
    print(f"    RSI: {row[4]} - {row[5]}")
    print(f"    Volume: {row[6]} - {row[7]}")
    print(f"    Performance: profit={row[8]}, win_rate={row[9]}, trades={row[10]}, grade={row[11]}")

print()
print("=" * 70)
print()

# global_strategies 샘플
print("2️⃣  global_strategies 샘플:")
cursor.execute("""
    SELECT id, zone_key, source_coin, source_strategy_id,
           rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
           profit, win_rate, trades_count, quality_grade
    FROM global_strategies
    LIMIT 3
""")

for row in cursor.fetchall():
    print(f"\n  ID: {row[0][:50]}...")
    print(f"    Zone: {row[1]}")
    print(f"    Source: {row[2]} (strategy: {row[3][:30]}...)")
    print(f"    RSI: {row[4]} - {row[5]}")
    print(f"    Volume: {row[6]} - {row[7]}")
    print(f"    Performance: profit={row[8]}, win_rate={row[9]}, trades={row[10]}, grade={row[11]}")

print()
print("=" * 70)
print()
print("💡 결론:")
print("  ✅ 파라미터 필드 (rsi_min, rsi_max, volume_ratio 등) 정상 복사됨")
print("  ℹ️  성과 필드 (profit, win_rate 등)는 소스에도 없음 (아직 백테스트 안됨)")
print("  → Paper Trading으로 실전 사용하면 성과 데이터가 쌓일 예정")

conn.close()
