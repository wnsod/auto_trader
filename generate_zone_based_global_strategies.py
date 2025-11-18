#!/usr/bin/env python
"""
실전 구역 기반 글로벌 전략 생성 및 검증

실제 모든 코인의 전략을 로드하여 글로벌 전략 생성:
- 모든 코인 대상 (제한 없음)
- 전체 전략 로드 (limit 없음)
- 구역별 커버리지 검증
- 변동성 그룹별 분포 확인
"""

import sys
import logging
from datetime import datetime
from collections import Counter, defaultdict

# Windows 인코딩
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print(f"실전 구역 기반 글로벌 전략 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    # 1. DB 초기화
    print("\n1️⃣ 데이터베이스 초기화")
    print("-" * 80)

    from rl_pipeline.db.schema import setup_database_tables

    if setup_database_tables():
        print("✅ 데이터베이스 테이블 준비 완료")
    else:
        print("❌ 데이터베이스 초기화 실패")
        sys.exit(1)

    # 2. 모든 코인 목록 조회
    print("\n2️⃣ 전략이 있는 코인 조회")
    print("-" * 80)

    from rl_pipeline.db.connection_pool import get_optimized_db_connection

    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()

        # 전략이 있는 모든 코인 조회
        cursor.execute("""
            SELECT DISTINCT coin
            FROM coin_strategies
            WHERE coin IS NOT NULL
            ORDER BY coin
        """)

        all_coins = [row[0] for row in cursor.fetchall()]

        print(f"전략이 있는 코인: {len(all_coins)}개")
        print(f"코인 목록: {', '.join(all_coins)}")

    # 3. 모든 코인 전략 로드
    print("\n3️⃣ 모든 코인의 전략 로드 (실전)")
    print("-" * 80)

    from rl_pipeline.db.reads import load_strategies_pool

    # 주요 인터벌
    intervals = ['15m', '60m', '240m']

    all_coin_strategies = {}
    total_loaded = 0

    for coin in all_coins:
        coin_strategies = {}

        for interval in intervals:
            # 실전: limit=0으로 모든 전략 로드
            strategies = load_strategies_pool(
                coin=coin,
                interval=interval,
                limit=0,  # 0이면 제한 없음
                order_by="created_at DESC"
            )

            if strategies:
                coin_strategies[interval] = strategies
                total_loaded += len(strategies)
                print(f"  {coin} {interval}: {len(strategies)}개 전략 로드")

        if coin_strategies:
            all_coin_strategies[coin] = coin_strategies

    print(f"\n✅ 전체 로드: {len(all_coin_strategies)}개 코인, {total_loaded}개 전략")

    if total_loaded == 0:
        print("⚠️ 전략 없음, 생성 중단")
        sys.exit(0)

    # 4. 구역 기반 글로벌 전략 생성
    print("\n4️⃣ 구역 기반 글로벌 전략 생성 (실전)")
    print("-" * 80)

    from rl_pipeline.strategy.zone_based_global_creator import create_zone_based_global_strategies

    global_strategies = create_zone_based_global_strategies(all_coin_strategies)

    print(f"\n✅ 글로벌 전략 생성 완료: {len(global_strategies)}개")

    if not global_strategies:
        print("⚠️ 생성된 글로벌 전략 없음")
        sys.exit(0)

    # 5. DB 저장
    print("\n5️⃣ DB 저장")
    print("-" * 80)

    from rl_pipeline.strategy.zone_based_global_creator import save_global_strategies_to_db

    saved_count = save_global_strategies_to_db(global_strategies)
    print(f"✅ DB 저장 완료: {saved_count}개")

    # 6. 데이터 검증
    print("\n6️⃣ 데이터 검증")
    print("-" * 80)

    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()

        # 전체 글로벌 전략 수
        cursor.execute("SELECT COUNT(*) FROM global_strategies")
        total_global = cursor.fetchone()[0]

        # 구역 기반 전략 수
        cursor.execute("SELECT COUNT(*) FROM global_strategies WHERE zone_key IS NOT NULL")
        zone_based = cursor.fetchone()[0]

        print(f"전체 글로벌 전략: {total_global}개")
        print(f"구역 기반 전략: {zone_based}개")

        # 구역 기반 전략 분석
        cursor.execute("""
            SELECT regime, rsi_zone, volatility_level, zone_key, source_coin, profit, win_rate
            FROM global_strategies
            WHERE zone_key IS NOT NULL
            ORDER BY created_at DESC
        """)

        zone_strategies = cursor.fetchall()

        if zone_strategies:
            # 통계 수집
            regime_dist = Counter()
            rsi_dist = Counter()
            volatility_dist = Counter()
            coin_dist = Counter()

            for row in zone_strategies:
                regime, rsi_zone, volatility, zone_key, source_coin, profit, win_rate = row

                if regime:
                    regime_dist[regime] += 1
                if rsi_zone:
                    rsi_dist[rsi_zone] += 1
                if volatility:
                    volatility_dist[volatility] += 1
                if source_coin:
                    coin_dist[source_coin] += 1

            print(f"\n📊 구역별 분포:")
            print(f"\n레짐별:")
            for regime, count in sorted(regime_dist.items()):
                print(f"  {regime}: {count}개")

            print(f"\nRSI 구역별:")
            for rsi, count in sorted(rsi_dist.items()):
                print(f"  {rsi}: {count}개")

            print(f"\n변동성 그룹별:")
            for vol, count in sorted(volatility_dist.items()):
                print(f"  {vol}: {count}개")

            print(f"\n출처 코인별:")
            for coin, count in sorted(coin_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {coin}: {count}개 전략 기여")

    # 7. 커버리지 분석
    print("\n7️⃣ 구역 커버리지 분석")
    print("-" * 80)

    # 이론적 최대 구역 수
    total_possible_zones = 3 * 5 * 3 * 4  # regime × RSI × market × volatility
    print(f"이론적 최대 구역 수: {total_possible_zones}개")
    print(f"  - 레짐: 3개 (ranging, trending, volatile)")
    print(f"  - RSI 구역: 5개 (oversold, low, neutral, high, overbought)")
    print(f"  - 시장 상황: 3개 (bearish, neutral, bullish)")
    print(f"  - 변동성 그룹: 4개 (LOW, MEDIUM, HIGH, VERY_HIGH)")

    coverage_pct = (len(global_strategies) / total_possible_zones) * 100
    print(f"\n실제 생성된 구역: {len(global_strategies)}개")
    print(f"커버리지: {coverage_pct:.1f}%")

    # 8. 샘플 데이터 확인
    print("\n8️⃣ 저장된 데이터 샘플 (최근 10개)")
    print("-" * 80)

    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, zone_key, source_coin, regime, rsi_zone, volatility_level, profit, win_rate, sharpe_ratio
            FROM global_strategies
            WHERE zone_key IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 10
        """)

        samples = cursor.fetchall()

        for i, sample in enumerate(samples, 1):
            strategy_id, zone_key, source_coin, regime, rsi_zone, vol_level, profit, win_rate, sharpe = sample

            print(f"\n{i}. 구역: {zone_key}")
            print(f"   ID: {strategy_id[:50]}...")
            print(f"   출처: {source_coin}")
            print(f"   레짐: {regime}, RSI: {rsi_zone}, 변동성: {vol_level}")
            print(f"   성과: profit={profit:.2%}, win_rate={win_rate:.2%}, sharpe={sharpe:.3f}")

    # 9. 변동성 그룹별 상세 분석
    print("\n9️⃣ 변동성 그룹별 상세 커버리지")
    print("-" * 80)

    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()

        for vol_group in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
            cursor.execute("""
                SELECT COUNT(DISTINCT zone_key)
                FROM global_strategies
                WHERE volatility_level = ?
            """, (vol_group,))

            count = cursor.fetchone()[0]

            # 각 변동성 그룹당 가능한 구역 수: 3 × 5 × 3 = 45개
            group_max_zones = 3 * 5 * 3
            group_coverage = (count / group_max_zones) * 100 if group_max_zones > 0 else 0

            print(f"{vol_group:12} : {count:3}개 / {group_max_zones}개 ({group_coverage:.1f}%)")

    # 10. 결과 요약
    print("\n" + "=" * 80)
    print("실전 데이터 검증 결과 요약")
    print("=" * 80)

    print(f"\n✅ 처리 완료:")
    print(f"  - 코인: {len(all_coin_strategies)}개")
    print(f"  - 전체 전략: {total_loaded}개")
    print(f"  - 생성된 글로벌 전략: {len(global_strategies)}개")
    print(f"  - DB 저장: {saved_count}개")
    print(f"  - 전체 커버리지: {coverage_pct:.1f}% ({len(global_strategies)}/{total_possible_zones})")

    if coverage_pct >= 50:
        print(f"\n🎉 우수! 50% 이상 커버리지 달성!")
    elif coverage_pct >= 30:
        print(f"\n✅ 양호! 30% 이상 커버리지 달성")
    else:
        print(f"\n⚠️ 더 많은 전략 학습이 필요합니다 (목표: 50% 이상)")

except Exception as e:
    print(f"\n❌ 실행 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print(f"실행 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
