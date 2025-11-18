#!/usr/bin/env python
"""글로벌 전략 데이터 검증"""
import sys
sys.path.append('/workspace')

import sqlite3
import json

DB_PATH = '/workspace/data_storage/rl_strategies.db'

print("=" * 70)
print("글로벌 전략 데이터 검증")
print("=" * 70)
print()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. 전략 개수 확인
cursor.execute("SELECT COUNT(*) FROM global_strategies")
total_count = cursor.fetchone()[0]

print(f"1️⃣  전체 전략 개수: {total_count}개")
print()

# 2. Zone별 분포 확인
cursor.execute("""
    SELECT regime, COUNT(*)
    FROM global_strategies
    GROUP BY regime
""")
regime_dist = cursor.fetchall()

print("2️⃣  Regime 분포:")
for regime, count in regime_dist:
    print(f"   {regime}: {count}개")
print()

# 3. RSI Zone 분포
cursor.execute("""
    SELECT rsi_zone, COUNT(*)
    FROM global_strategies
    GROUP BY rsi_zone
""")
rsi_dist = cursor.fetchall()

print("3️⃣  RSI Zone 분포:")
for rsi, count in rsi_dist:
    print(f"   {rsi}: {count}개")
print()

# 4. 출처 코인 확인
cursor.execute("""
    SELECT source_coin, COUNT(*)
    FROM global_strategies
    GROUP BY source_coin
""")
source_dist = cursor.fetchall()

print("4️⃣  출처 코인 분포:")
for coin, count in source_dist:
    print(f"   {coin}: {count}개")
print()

# 5. 파라미터 검증
cursor.execute("""
    SELECT id, zone_key, rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
           mfi_min, mfi_max, atr_min, atr_max
    FROM global_strategies
    WHERE rsi_min IS NULL OR rsi_max IS NULL
""")

null_params = cursor.fetchall()

if null_params:
    print("5️⃣  ⚠️  파라미터 누락 전략:")
    for row in null_params[:5]:
        print(f"   {row[0][:40]}... - Zone: {row[1]}")
    print(f"   총 {len(null_params)}개 전략에 파라미터 누락")
    print()
else:
    print("5️⃣  ✅ 모든 전략에 파라미터 존재")
    print()

# 6. 전략 유사도 검증 (params 필드)
print("6️⃣  파라미터 다양성 검증:")

cursor.execute("""
    SELECT rsi_min, rsi_max, volume_ratio_min, volume_ratio_max
    FROM global_strategies
    LIMIT 10
""")

params = cursor.fetchall()
unique_params = set(params)

print(f"   샘플 10개 중 고유한 파라미터 조합: {len(unique_params)}개")
if len(unique_params) > 1:
    print("   ✅ 파라미터가 다양함 (중복 제거 성공)")
else:
    print("   ⚠️  모든 전략이 동일한 파라미터 (중복 제거 실패 가능)")
print()

# 7. 상세 전략 검증 (샘플 3개)
print("7️⃣  상세 전략 샘플:")
print("-" * 70)

cursor.execute("""
    SELECT id, zone_key, source_coin, source_strategy_id,
           rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
           profit, win_rate, trades_count, quality_grade,
           params
    FROM global_strategies
    LIMIT 3
""")

for row in cursor.fetchall():
    print(f"ID: {row[0]}")
    print(f"  Zone: {row[1]}")
    print(f"  출처: {row[2]} (전략 ID: {row[3][:30]}...)")
    print(f"  파라미터:")
    print(f"    RSI: {row[4]:.2f} - {row[5]:.2f}")
    print(f"    Volume Ratio: {row[6]:.2f} - {row[7]:.2f}")
    print(f"  성과:")
    print(f"    Profit: {row[8]:.2%}")
    print(f"    Win Rate: {row[9]:.2%}")
    print(f"    Trades: {row[10]}")
    print(f"    Grade: {row[11]}")

    # params JSON 검증
    if row[12]:
        try:
            params_json = json.loads(row[12])
            print(f"  params JSON 필드: {len(params_json)}개 키")
        except:
            print(f"  params JSON 파싱 실패")
    print()

# 8. Zone 커버리지 확인
cursor.execute("""
    SELECT COUNT(DISTINCT zone_key) FROM global_strategies
""")
unique_zones = cursor.fetchone()[0]

print("8️⃣  Zone 커버리지:")
print(f"   고유한 Zone 개수: {unique_zones}개")
print(f"   이론적 최대 Zone: 180개 (3×5×3×4)")
print(f"   커버리지: {unique_zones/180*100:.1f}%")
print()

conn.close()

print("=" * 70)
print("✅ 검증 완료!")
print("=" * 70)
