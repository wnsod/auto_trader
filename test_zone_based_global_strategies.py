#!/usr/bin/env python
"""
구역 기반 글로벌 전략 시스템 테스트

테스트 단계:
1. DB 스키마 확인 (증분 학습 컬럼)
2. 개별 코인 전략 로드
3. 구역 기반 글로벌 전략 생성
4. DB 저장 및 검증
5. 구역 커버리지 분석
"""

import sys
import logging
from datetime import datetime

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
print(f"구역 기반 글로벌 전략 시스템 테스트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    # 1. DB 스키마 확인
    print("\n1️⃣ DB 스키마 확인")
    print("-" * 80)

    from rl_pipeline.db.schema import setup_database_tables

    if setup_database_tables():
        print("✅ 데이터베이스 테이블 준비 완료")
    else:
        print("❌ 데이터베이스 초기화 실패")
        sys.exit(1)

    # global_strategies 테이블 컬럼 확인
    from rl_pipeline.db.connection_pool import get_optimized_db_connection

    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(global_strategies)")
        columns = cursor.fetchall()

        print("\nglobal_strategies 테이블 컬럼:")
        similarity_cols = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            if 'similarity' in col_name or 'zone' in col_name or 'source' in col_name:
                similarity_cols.append(f"  {col_name} ({col_type})")

        if similarity_cols:
            print("증분 학습 및 구역 관련 컬럼:")
            for col in similarity_cols:
                print(col)
        else:
            print("⚠️ 증분 학습 컬럼 없음!")

    # 2. 개별 코인 전략 로드
    print("\n2️⃣ 개별 코인 전략 로드")
    print("-" * 80)

    from rl_pipeline.db.reads import load_strategies_pool

    # 테스트용 코인 목록
    test_coins = ['BTC', 'ETH', 'ADA', 'SOL']
    test_intervals = ['15m', '60m', '240m']

    all_coin_strategies = {}

    for coin in test_coins:
        coin_strategies = {}

        for interval in test_intervals:
            strategies = load_strategies_pool(
                coin=coin,
                interval=interval,
                limit=50,  # 테스트용 50개만
                order_by="created_at DESC"
            )

            if strategies:
                coin_strategies[interval] = strategies
                print(f"  {coin} {interval}: {len(strategies)}개 전략 로드")

        if coin_strategies:
            all_coin_strategies[coin] = coin_strategies

    total_strategies = sum(
        len(strategies)
        for coin_data in all_coin_strategies.values()
        for strategies in coin_data.values()
    )

    print(f"\n✅ 전체 로드: {len(all_coin_strategies)}개 코인, {total_strategies}개 전략")

    if total_strategies == 0:
        print("⚠️ 전략 없음, 테스트 중단")
        sys.exit(0)

    # 3. 구역 기반 글로벌 전략 생성
    print("\n3️⃣ 구역 기반 글로벌 전략 생성")
    print("-" * 80)

    from rl_pipeline.strategy.zone_based_global_creator import create_zone_based_global_strategies

    global_strategies = create_zone_based_global_strategies(all_coin_strategies)

    print(f"\n✅ 글로벌 전략 생성 완료: {len(global_strategies)}개")

    if global_strategies:
        # 첫 3개 예시 출력
        print("\n글로벌 전략 예시 (처음 3개):")
        for i, strategy in enumerate(global_strategies[:3], 1):
            print(f"\n  {i}. {strategy.get('name')}")
            print(f"     ID: {strategy.get('id')}")
            print(f"     구역: {strategy.get('zone_key')}")
            print(f"     출처: {strategy.get('source_coin')}")
            print(f"     성과: profit={strategy.get('profit', 0):.2%}, win_rate={strategy.get('win_rate', 0):.2%}")

    # 4. DB 저장 및 검증
    print("\n4️⃣ DB 저장 및 검증")
    print("-" * 80)

    from rl_pipeline.strategy.zone_based_global_creator import save_global_strategies_to_db

    saved_count = save_global_strategies_to_db(global_strategies)
    print(f"✅ DB 저장 완료: {saved_count}개")

    # 저장된 데이터 검증
    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()

        # 전체 개수
        cursor.execute("SELECT COUNT(*) FROM global_strategies")
        total_count = cursor.fetchone()[0]
        print(f"\nDB 전체 글로벌 전략 수: {total_count}개")

        # 구역 기반 전략 개수
        cursor.execute("SELECT COUNT(*) FROM global_strategies WHERE zone_key IS NOT NULL")
        zone_based_count = cursor.fetchone()[0]
        print(f"구역 기반 전략 수: {zone_based_count}개")

        # 최근 5개 확인
        cursor.execute("""
            SELECT id, zone_key, source_coin, profit, win_rate
            FROM global_strategies
            WHERE zone_key IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """)
        rows = cursor.fetchall()

        if rows:
            print("\n최근 저장된 5개 전략:")
            for row in rows:
                strategy_id, zone_key, source_coin, profit, win_rate = row
                print(f"  ID: {strategy_id[:40]}...")
                print(f"    구역: {zone_key}")
                print(f"    출처: {source_coin}")
                print(f"    성과: profit={profit:.2%}, win_rate={win_rate:.2%}")
                print()

    # 5. 구역 커버리지 분석
    print("5️⃣ 구역 커버리지 분석")
    print("-" * 80)

    # 이론적 구역 수
    regimes = ['ranging', 'trending', 'volatile']
    rsi_zones = ['oversold', 'low', 'neutral', 'high', 'overbought']
    market_conditions = ['bearish', 'neutral', 'bullish']
    volatility_levels = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']  # 기존 coin_volatility.py 시스템

    total_possible_zones = len(regimes) * len(rsi_zones) * len(market_conditions) * len(volatility_levels)

    print(f"이론적 최대 구역 수: {total_possible_zones}개")
    print(f"  - 레짐: {len(regimes)}개")
    print(f"  - RSI 구역: {len(rsi_zones)}개")
    print(f"  - 시장 상황: {len(market_conditions)}개")
    print(f"  - 변동성 그룹: {len(volatility_levels)}개 (LOW/MEDIUM/HIGH/VERY_HIGH)")

    print(f"\n실제 생성된 구역 수: {len(global_strategies)}개")
    coverage = (len(global_strategies) / total_possible_zones) * 100 if total_possible_zones > 0 else 0
    print(f"커버리지: {coverage:.1f}%")

    # 레짐별 분포
    from collections import Counter

    regimes_in_data = [s.get('regime') for s in global_strategies if s.get('regime')]
    regime_dist = Counter(regimes_in_data)

    print(f"\n레짐별 분포:")
    for regime, count in sorted(regime_dist.items()):
        print(f"  {regime}: {count}개")

    # RSI 구역별 분포
    rsi_zones_in_data = [s.get('rsi_zone') for s in global_strategies if s.get('rsi_zone')]
    rsi_dist = Counter(rsi_zones_in_data)

    print(f"\nRSI 구역별 분포:")
    for rsi_zone, count in sorted(rsi_dist.items()):
        print(f"  {rsi_zone}: {count}개")

    # 변동성 그룹별 분포
    volatility_in_data = [s.get('volatility_level') for s in global_strategies if s.get('volatility_level')]
    volatility_dist = Counter(volatility_in_data)

    print(f"\n변동성 그룹별 분포:")
    for vol_group, count in sorted(volatility_dist.items()):
        print(f"  {vol_group}: {count}개")

    # 6. 종합 결과
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)

    success_checks = [
        ("DB 스키마 (증분 학습 컬럼)", len(similarity_cols) >= 3),
        ("개별 코인 전략 로드", total_strategies > 0),
        ("글로벌 전략 생성", len(global_strategies) > 0),
        ("DB 저장", saved_count > 0),
        ("구역 커버리지", coverage >= 2)  # 최소 2% 커버리지 (180개 구역 기준)
    ]

    passed = sum(1 for _, result in success_checks if result)
    total = len(success_checks)

    print(f"\n통과한 테스트: {passed}/{total}")
    for check_name, result in success_checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")

    if passed == total:
        print("\n🎉 모든 테스트 통과! 구역 기반 글로벌 전략 시스템 정상 작동")
    elif passed >= total * 0.8:
        print(f"\n⚠️ 대부분 통과 ({passed}/{total}), 일부 개선 필요")
    else:
        print(f"\n❌ 테스트 실패 ({passed}/{total})")

except Exception as e:
    print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print(f"테스트 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
