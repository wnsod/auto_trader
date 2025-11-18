#!/usr/bin/env python
"""
구역 기반 글로벌 전략 데이터 상세 검증
"""

import sys
import json
from collections import Counter

# Windows 인코딩
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("=" * 80)
print("구역 기반 글로벌 전략 데이터 상세 검증")
print("=" * 80)

try:
    from rl_pipeline.db.connection_pool import get_optimized_db_connection

    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()

        # 1. 전체 구역 기반 전략 조회
        print("\n1️⃣ 구역 기반 전략 전체 조회")
        print("-" * 80)

        cursor.execute("""
            SELECT zone_key, regime, rsi_zone, volatility_level, source_coin,
                   profit, win_rate, sharpe_ratio, created_at
            FROM global_strategies
            WHERE zone_key IS NOT NULL
            ORDER BY created_at DESC
        """)

        all_zone_strategies = cursor.fetchall()
        print(f"전체 구역 기반 전략: {len(all_zone_strategies)}개")

        # 2. 변동성 그룹별 상세 분석
        print("\n2️⃣ 변동성 그룹별 상세 분석")
        print("-" * 80)

        volatility_dist = Counter()
        for row in all_zone_strategies:
            zone_key, regime, rsi_zone, vol_level, source_coin, profit, win_rate, sharpe, created_at = row
            volatility_dist[vol_level] += 1

        print("변동성 그룹별 개수:")
        for vol, count in sorted(volatility_dist.items()):
            print(f"  {vol}: {count}개")

        # 3. 이전 시스템(low/medium/high) 전략 확인
        print("\n3️⃣ 이전 3단계 시스템 전략 (삭제 필요)")
        print("-" * 80)

        cursor.execute("""
            SELECT zone_key, volatility_level, source_coin, created_at
            FROM global_strategies
            WHERE volatility_level IN ('low', 'medium', 'high')
            ORDER BY created_at DESC
        """)

        old_strategies = cursor.fetchall()
        if old_strategies:
            print(f"이전 시스템 전략: {len(old_strategies)}개 (삭제 권장)")
            for i, row in enumerate(old_strategies[:5], 1):
                zone_key, vol_level, source_coin, created_at = row
                print(f"  {i}. {zone_key} ({vol_level}) - {source_coin} - {created_at}")
        else:
            print("✅ 이전 시스템 전략 없음")

        # 4. 신규 4그룹 시스템 전략 확인
        print("\n4️⃣ 신규 4그룹 시스템 전략")
        print("-" * 80)

        cursor.execute("""
            SELECT zone_key, volatility_level, source_coin, created_at
            FROM global_strategies
            WHERE volatility_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
            ORDER BY created_at DESC
        """)

        new_strategies = cursor.fetchall()
        print(f"신규 4그룹 시스템 전략: {len(new_strategies)}개")

        # 5. 레짐 × 변동성 교차 분석
        print("\n5️⃣ 레짐 × 변동성 교차 분석")
        print("-" * 80)

        cursor.execute("""
            SELECT regime, volatility_level, COUNT(*) as cnt
            FROM global_strategies
            WHERE zone_key IS NOT NULL
            GROUP BY regime, volatility_level
            ORDER BY regime, volatility_level
        """)

        cross_analysis = cursor.fetchall()
        print("레짐 × 변동성 조합:")
        for regime, vol, count in cross_analysis:
            print(f"  {regime:10} × {vol:12} : {count}개")

        # 6. 출처 코인별 기여도 (변동성 그룹별)
        print("\n6️⃣ 출처 코인별 기여도 (변동성 그룹별)")
        print("-" * 80)

        for vol_group in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
            cursor.execute("""
                SELECT source_coin, COUNT(*) as cnt
                FROM global_strategies
                WHERE volatility_level = ?
                GROUP BY source_coin
                ORDER BY cnt DESC
            """, (vol_group,))

            coin_contributions = cursor.fetchall()
            if coin_contributions:
                print(f"\n{vol_group}:")
                for coin, count in coin_contributions:
                    print(f"  {coin}: {count}개")

        # 7. 최신 생성 전략 샘플 (params 포함)
        print("\n7️⃣ 최신 생성 전략 샘플 (params 포함)")
        print("-" * 80)

        cursor.execute("""
            SELECT zone_key, volatility_level, source_coin, params,
                   profit, win_rate, sharpe_ratio
            FROM global_strategies
            WHERE zone_key IS NOT NULL
              AND volatility_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
            ORDER BY created_at DESC
            LIMIT 3
        """)

        samples = cursor.fetchall()
        for i, row in enumerate(samples, 1):
            zone_key, vol_level, source_coin, params_str, profit, win_rate, sharpe = row

            print(f"\n{i}. 구역: {zone_key}")
            print(f"   변동성: {vol_level} | 출처: {source_coin}")
            print(f"   성과: profit={profit:.2%}, win_rate={win_rate:.2%}, sharpe={sharpe:.3f}")

            # params 파싱
            try:
                if params_str:
                    params = json.loads(params_str) if isinstance(params_str, str) else params_str
                    atr_min = params.get('atr_min', 'N/A')
                    atr_max = params.get('atr_max', 'N/A')
                    rsi_min = params.get('rsi_min', 'N/A')
                    rsi_max = params.get('rsi_max', 'N/A')
                    print(f"   파라미터: ATR={atr_min}~{atr_max}, RSI={rsi_min}~{rsi_max}")
            except:
                print(f"   파라미터: (파싱 실패)")

        # 8. 구역별 최고 성능 전략
        print("\n8️⃣ 변동성 그룹별 최고 성능 전략")
        print("-" * 80)

        for vol_group in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
            cursor.execute("""
                SELECT zone_key, source_coin, profit, win_rate, sharpe_ratio
                FROM global_strategies
                WHERE volatility_level = ?
                ORDER BY (profit * win_rate) DESC
                LIMIT 1
            """, (vol_group,))

            best = cursor.fetchone()
            if best:
                zone_key, source_coin, profit, win_rate, sharpe = best
                score = profit * win_rate
                print(f"{vol_group:12} : {zone_key} (출처: {source_coin})")
                print(f"               profit={profit:.2%}, win_rate={win_rate:.2%}, score={score:.4f}")

        # 9. 커버리지 요약
        print("\n9️⃣ 커버리지 요약")
        print("-" * 80)

        total_zones = 180  # 3 × 5 × 3 × 4

        cursor.execute("""
            SELECT COUNT(DISTINCT zone_key)
            FROM global_strategies
            WHERE volatility_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
        """)

        covered_zones = cursor.fetchone()[0]
        coverage = (covered_zones / total_zones) * 100

        print(f"전체 커버리지: {covered_zones}/{total_zones} ({coverage:.1f}%)")

        # 변동성 그룹별
        for vol_group in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
            cursor.execute("""
                SELECT COUNT(DISTINCT zone_key)
                FROM global_strategies
                WHERE volatility_level = ?
            """, (vol_group,))

            group_covered = cursor.fetchone()[0]
            group_total = 45  # 3 × 5 × 3
            group_coverage = (group_covered / group_total) * 100

            print(f"{vol_group:12} : {group_covered:2}/{group_total} ({group_coverage:5.1f}%)")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("검증 완료")
print("=" * 80)
