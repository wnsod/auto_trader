#!/usr/bin/env python
"""
빠른 다중 코인 통합 분석

기존 데이터를 활용하여 여러 코인에 대해 통합 분석 v1 실행
(RL 학습 없이 기존 전략 데이터만 사용)
"""

import sys
sys.path.append('/workspace')

import sqlite3
from datetime import datetime
import json
from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1

# 분석할 코인 목록
COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'LINK', 'DOT', 'MATIC', 'AVAX']


def check_coin_data(coin: str, db_path: str = '/workspace/data_storage/rl_strategies.db'):
    """
    코인의 전략 데이터 존재 여부 확인

    Returns:
        (exists, interval_count, strategy_count)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT interval), COUNT(*)
        FROM rl_strategy_rollup
        WHERE coin = ?
    """, (coin,))

    result = cursor.fetchone()
    conn.close()

    if result and result[1] > 0:
        return (True, result[0], result[1])
    else:
        return (False, 0, 0)


def analyze_coin_with_check(coin: str):
    """
    코인 데이터 확인 후 통합 분석 실행

    Returns:
        (success, result, data_info)
    """
    # 데이터 확인
    exists, interval_count, strategy_count = check_coin_data(coin)

    if not exists:
        return (False, None, {'intervals': 0, 'strategies': 0, 'reason': 'no_data'})

    if interval_count < 4:
        return (False, None, {
            'intervals': interval_count,
            'strategies': strategy_count,
            'reason': 'insufficient_intervals'
        })

    # 통합 분석 실행
    try:
        analyzer = IntegratedAnalyzerV1()
        result = analyzer.analyze(coin)
        return (True, result, {'intervals': interval_count, 'strategies': strategy_count})
    except Exception as e:
        return (False, None, {
            'intervals': interval_count,
            'strategies': strategy_count,
            'reason': f'analysis_error: {str(e)}'
        })


def main():
    print("=" * 70)
    print("🔍 빠른 다중 코인 통합 분석")
    print("=" * 70)
    print()
    print("📊 기존 전략 데이터를 활용한 통합 분석 v1 실행")
    print("   (RL 학습 없음 - 빠른 평가)")
    print()
    print(f"대상 코인: {', '.join(COINS)}")
    print()

    # 결과 저장
    results = {}
    analyzed_coins = []
    failed_coins = []

    # 1단계: 데이터 확인 및 분석
    print("=" * 70)
    print("📊 코인별 데이터 확인 및 분석")
    print("=" * 70)
    print()

    for coin in COINS:
        print(f"🔍 {coin} 확인 중...")

        success, result, data_info = analyze_coin_with_check(coin)

        if success:
            print(f"   ✅ 분석 완료")
            print(f"      인터벌: {data_info['intervals']}개, 전략: {data_info['strategies']}개")
            print(f"      방향: {result['direction']}, 타이밍: {result['timing']}, "
                  f"크기: {result['size']:.1%}, 확신도: {result['confidence']:.1%}")
            analyzed_coins.append(coin)
            results[coin] = {
                'success': True,
                'analysis': result,
                'data': data_info
            }
        else:
            reason = data_info.get('reason', 'unknown')
            print(f"   ❌ 분석 불가: {reason}")
            if data_info['intervals'] > 0:
                print(f"      인터벌: {data_info['intervals']}개 (4개 필요), "
                      f"전략: {data_info['strategies']}개")
            failed_coins.append(coin)
            results[coin] = {
                'success': False,
                'data': data_info
            }

        print()

    # 2단계: 결과 요약
    print("=" * 70)
    print("📋 전체 결과 요약")
    print("=" * 70)
    print()

    print(f"분석 성공: {len(analyzed_coins)}/{len(COINS)} "
          f"({len(analyzed_coins)/len(COINS)*100:.0f}%)")
    print()

    if analyzed_coins:
        print("분석 성공 코인:")
        for coin in analyzed_coins:
            print(f"  ✅ {coin}")
        print()
    else:
        print("⚠️  분석 성공한 코인 없음!")
        print()

    if failed_coins:
        print("분석 실패 코인:")
        for coin in failed_coins:
            data_info = results[coin]['data']
            reason = data_info.get('reason', 'unknown')
            if reason == 'no_data':
                print(f"  ❌ {coin}: 데이터 없음")
            elif reason == 'insufficient_intervals':
                print(f"  ❌ {coin}: 인터벌 부족 ({data_info['intervals']}/4)")
            else:
                print(f"  ❌ {coin}: {reason}")
        print()

    # 3단계: 상세 분석 결과
    if analyzed_coins:
        print("=" * 70)
        print("📊 상세 분석 결과")
        print("=" * 70)
        print()

        print(f"{'코인':<8} {'방향':<8} {'타이밍':<8} {'크기':>8} {'확신도':>8} "
              f"{'기간':<8} {'인터벌':>8} {'전략':>8}")
        print("-" * 70)

        for coin in analyzed_coins:
            r = results[coin]['analysis']
            d = results[coin]['data']
            print(f"{coin:<8} {r['direction']:<8} {r['timing']:<8} "
                  f"{r['size']:>7.1%} {r['confidence']:>7.1%} {r['horizon']:<8} "
                  f"{d['intervals']:>8} {d['strategies']:>8}")

        print()

        # 통계
        long_count = sum(1 for coin in analyzed_coins
                         if results[coin]['analysis']['direction'] == 'LONG')
        short_count = sum(1 for coin in analyzed_coins
                          if results[coin]['analysis']['direction'] == 'SHORT')
        hold_count = sum(1 for coin in analyzed_coins
                         if results[coin]['analysis']['direction'] == 'HOLD')

        now_count = sum(1 for coin in analyzed_coins
                        if results[coin]['analysis']['timing'] == 'NOW')
        wait_count = sum(1 for coin in analyzed_coins
                         if results[coin]['analysis']['timing'] == 'WAIT')

        avg_size = sum(results[coin]['analysis']['size'] for coin in analyzed_coins) / len(analyzed_coins)
        avg_confidence = sum(results[coin]['analysis']['confidence'] for coin in analyzed_coins) / len(analyzed_coins)

        print("=" * 70)
        print("📈 통계")
        print("=" * 70)
        print()

        print("방향 분포:")
        print(f"  LONG:  {long_count:2}개 ({long_count/len(analyzed_coins)*100:4.0f}%)")
        print(f"  SHORT: {short_count:2}개 ({short_count/len(analyzed_coins)*100:4.0f}%)")
        print(f"  HOLD:  {hold_count:2}개 ({hold_count/len(analyzed_coins)*100:4.0f}%)")
        print()

        print("타이밍 분포:")
        print(f"  NOW:   {now_count:2}개 ({now_count/len(analyzed_coins)*100:4.0f}%)")
        print(f"  WAIT:  {wait_count:2}개 ({wait_count/len(analyzed_coins)*100:4.0f}%)")
        print()

        print(f"평균 포지션 크기: {avg_size:.1%}")
        print(f"평균 확신도: {avg_confidence:.1%}")
        print()

        # 거래 추천
        print("=" * 70)
        print("💡 거래 추천 (방향=LONG/SHORT, 타이밍=NOW)")
        print("=" * 70)
        print()

        tradeable = [coin for coin in analyzed_coins
                     if results[coin]['analysis']['direction'] in ['LONG', 'SHORT']
                     and results[coin]['analysis']['timing'] == 'NOW']

        if tradeable:
            print(f"{'코인':<8} {'방향':<8} {'크기':>8} {'확신도':>8} {'기간':<8}")
            print("-" * 70)

            # 확신도 순으로 정렬
            tradeable_sorted = sorted(
                tradeable,
                key=lambda c: results[c]['analysis']['confidence'],
                reverse=True
            )

            for coin in tradeable_sorted:
                r = results[coin]['analysis']
                print(f"{coin:<8} {r['direction']:<8} {r['size']:>7.1%} "
                      f"{r['confidence']:>7.1%} {r['horizon']:<8}")

            print()
            print(f"총 {len(tradeable)}개 코인 거래 추천")
        else:
            print("현재 거래 추천 코인 없음 (모두 HOLD 또는 WAIT)")

        print()

    # 4단계: 데이터 수집 필요한 코인
    print("=" * 70)
    print("📊 데이터 수집 필요 코인")
    print("=" * 70)
    print()

    no_data = [coin for coin in failed_coins
               if results[coin]['data'].get('reason') == 'no_data']

    insufficient = [coin for coin in failed_coins
                    if results[coin]['data'].get('reason') == 'insufficient_intervals']

    if no_data:
        print("데이터 없음 (학습 필요):")
        for coin in no_data:
            print(f"  • {coin}: RL 파이프라인 실행 필요")
        print()

    if insufficient:
        print("인터벌 부족 (추가 학습 필요):")
        for coin in insufficient:
            d = results[coin]['data']
            missing = 4 - d['intervals']
            print(f"  • {coin}: {d['intervals']}/4 인터벌, {missing}개 더 필요")
        print()

    if no_data or insufficient:
        print("💡 RL 파이프라인 실행 명령:")
        print("   python run_multi_coin_analysis.py")
        print("   (예상 소요 시간: 1-2시간/코인)")
    else:
        print("✅ 모든 코인 데이터 충분")

    print()

    # 결과 저장
    output_file = f'/workspace/quick_multi_coin_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_data = {
        'coins_checked': COINS,
        'analyzed_coins': analyzed_coins,
        'failed_coins': failed_coins,
        'results': {
            coin: {
                'success': data['success'],
                'analysis': {
                    'direction': data['analysis']['direction'],
                    'timing': data['analysis']['timing'],
                    'size': float(data['analysis']['size']),
                    'confidence': float(data['analysis']['confidence']),
                    'horizon': data['analysis']['horizon']
                } if data['success'] else None,
                'data': data['data']
            }
            for coin, data in results.items()
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("=" * 70)
    print(f"결과 저장: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
