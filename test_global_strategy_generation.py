#!/usr/bin/env python
"""글로벌 전략 생성 테스트"""
import sys
sys.path.append('/workspace')

import sqlite3
from rl_pipeline.strategy.zone_based_global_creator import create_zone_based_global_strategies
from rl_pipeline.db.connection_pool import get_optimized_db_connection

print("=" * 70)
print("글로벌 전략 생성 테스트")
print("=" * 70)
print()

# 1. coin_strategies에서 모든 전략 로드
print("1️⃣  coin_strategies 로드 중...")
all_coin_strategies = {}

with get_optimized_db_connection("strategies") as conn:
    cursor = conn.cursor()

    # 코인 목록 가져오기
    cursor.execute("SELECT DISTINCT coin FROM coin_strategies")
    coins = [row[0] for row in cursor.fetchall()]

    print(f"   코인 목록: {coins}")
    print()

    for coin in coins:
        coin_strategies = {}

        # 인터벌 목록 가져오기
        cursor.execute("SELECT DISTINCT interval FROM coin_strategies WHERE coin = ?", (coin,))
        intervals = [row[0] for row in cursor.fetchall()]

        for interval in intervals:
            # 전략 로드
            cursor.execute("SELECT * FROM coin_strategies WHERE coin = ? AND interval = ?", (coin, interval))
            results = cursor.fetchall()

            if results:
                columns_query = "PRAGMA table_info(coin_strategies)"
                columns_info = cursor.execute(columns_query).fetchall()
                columns = [col[1] for col in columns_info]

                strategies = []
                for row in results:
                    strategy_dict = dict(zip(columns, row))
                    # 출처 정보 추가
                    strategy_dict['_source_coin'] = coin
                    strategy_dict['_source_interval'] = interval
                    strategies.append(strategy_dict)

                coin_strategies[interval] = strategies
                print(f"   {coin} {interval}: {len(strategies)}개 전략")

        if coin_strategies:
            all_coin_strategies[coin] = coin_strategies

print()
print(f"✅ 총 {len(all_coin_strategies)}개 코인, {sum(len(s) for c in all_coin_strategies.values() for s in c.values())}개 전략 로드 완료")
print()

# 2. 글로벌 전략 생성 (유사도 검사 비활성화)
print("2️⃣  글로벌 전략 생성 중 (유사도 검사 비활성화)...")
print()

global_strategies = create_zone_based_global_strategies(
    all_coin_strategies,
    enable_similarity_check=False  # 유사도 검사 비활성화
)

print()
print(f"✅ 글로벌 전략 생성 완료: {len(global_strategies)}개")
print()

# 3. 생성된 전략 샘플 출력
if global_strategies:
    print("3️⃣  생성된 전략 샘플 (최대 5개):")
    print("-" * 70)

    for i, strategy in enumerate(global_strategies[:5], 1):
        print(f"{i}. ID: {strategy['id']}")
        print(f"   Zone: {strategy['zone_key']}")
        print(f"   출처: {strategy['source_coin']}")
        print(f"   성과: profit={strategy['profit']:.2%}, win_rate={strategy['win_rate']:.2%}")
        print(f"   파라미터: rsi_min={strategy.get('rsi_min')}, rsi_max={strategy.get('rsi_max')}")
        print(f"   volume_ratio: min={strategy.get('volume_ratio_min')}, max={strategy.get('volume_ratio_max')}")
        print()
else:
    print("⚠️  생성된 전략이 없습니다!")
    print()

# 4. global_strategies 테이블에 저장
if global_strategies:
    print("4️⃣  global_strategies 테이블에 저장 중...")

    from rl_pipeline.strategy.zone_based_global_creator import save_global_strategies_to_db

    saved_count = save_global_strategies_to_db(global_strategies)
    print(f"✅ {saved_count}개 전략 저장 완료")
    print()
else:
    print("⚠️  저장할 전략이 없습니다!")
    print()

# 5. 저장 확인
print("5️⃣  저장 확인:")
with get_optimized_db_connection("strategies") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM global_strategies")
    count = cursor.fetchone()[0]

    print(f"   global_strategies 테이블: {count}개 전략")

    if count > 0:
        cursor.execute("""
            SELECT id, zone_key, source_coin, profit, win_rate, rsi_min, rsi_max
            FROM global_strategies
            ORDER BY created_at DESC
            LIMIT 5
        """)

        print()
        print("   최근 저장된 전략 5개:")
        print("   " + "-" * 66)

        for row in cursor.fetchall():
            strategy_id = row[0][:30]
            zone_key = row[1]
            source_coin = row[2]
            profit = row[3] or 0
            win_rate = row[4] or 0
            rsi_min = row[5]
            rsi_max = row[6]

            print(f"   {strategy_id}...")
            print(f"     Zone: {zone_key}, 출처: {source_coin}")
            print(f"     성과: {profit:.2%} / {win_rate:.2%}, RSI: {rsi_min}-{rsi_max}")

print()
print("=" * 70)
print("✅ 테스트 완료!")
print("=" * 70)
