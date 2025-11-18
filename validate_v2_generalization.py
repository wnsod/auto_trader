#!/usr/bin/env python
"""
v2 일반화 검증 - 다른 코인에서 성능 확인

LINK로 최적화된 v2 파라미터가 BTC, ETH, SOL 등 다른 코인에서도
v1보다 좋은 성능을 보이는지 검증
"""

import sys
sys.path.append('/workspace')

import json
from typing import Dict
from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1
from rl_pipeline.analysis.integrated_analysis_v2 import IntegratedAnalyzerV2, V2Parameters
from rl_pipeline.analysis.v2_backtest import simple_backtest

# 테스트할 코인들
TEST_COINS = ['BTC', 'ETH', 'SOL', 'LINK']  # LINK는 베이스라인 확인용

def load_optimized_v2_params() -> V2Parameters:
    """최적화된 v2 파라미터 로드"""
    # 최신 파라미터 파일 찾기
    import glob
    param_files = glob.glob('/workspace/v2_params_optimized_*.json')

    if not param_files:
        raise FileNotFoundError("최적화된 파라미터 파일이 없습니다.")

    # 가장 최신 파일 사용
    latest_file = sorted(param_files)[-1]

    with open(latest_file, 'r') as f:
        data = json.load(f)

    raw_params = data['raw_params']

    print(f"✅ 최적화 파라미터 로드: {latest_file}")
    print(f"   최적화 코인: {data['coin']}")
    print(f"   최적화 점수: {data['score']:.3f}")
    print(f"   최적화 시간: {data['timestamp']}")
    print()

    return V2Parameters(raw_params)


def evaluate_coin(coin: str, params_v1: V2Parameters, params_v2: V2Parameters) -> Dict:
    """코인 하나에 대해 v1과 v2 비교"""

    print(f"📊 {coin} 평가 중...")

    # v1 평가
    analyzer_v1 = IntegratedAnalyzerV1()
    signal_v1 = analyzer_v1.analyze(coin)
    score_v1 = simple_backtest(params_v1.to_raw(), coin)

    # v2 평가
    analyzer_v2 = IntegratedAnalyzerV2(params_v2)
    signal_v2 = analyzer_v2.analyze(coin)
    score_v2 = simple_backtest(params_v2.to_raw(), coin)

    # 개선율 계산
    if score_v1 != 0:
        improvement = ((score_v2 / score_v1) - 1) * 100
    else:
        improvement = 0.0

    result = {
        'coin': coin,
        'v1': {
            'score': score_v1,
            'direction': signal_v1['direction'],
            'timing': signal_v1['timing'],
            'size': signal_v1['size'],
            'confidence': signal_v1['confidence']
        },
        'v2': {
            'score': score_v2,
            'direction': signal_v2['direction'],
            'timing': signal_v2['timing'],
            'size': signal_v2['size'],
            'confidence': signal_v2['confidence']
        },
        'improvement': improvement,
        'improved': score_v2 > score_v1
    }

    print(f"   v1 점수: {score_v1:.3f}")
    print(f"   v2 점수: {score_v2:.3f}")
    print(f"   개선:    {improvement:+.1f}%")
    print()

    return result


def main():
    print("=" * 70)
    print("v2 일반화 검증")
    print("=" * 70)
    print()

    # 파라미터 로드
    print("=" * 70)
    print("1️⃣  파라미터 로드")
    print("=" * 70)
    print()

    params_v1 = V2Parameters()  # v1 기본 파라미터
    params_v2 = load_optimized_v2_params()  # v2 최적 파라미터

    print("v1 파라미터:")
    v1_transformed = params_v1.transform()
    print(f"  방향 가중치: 1d={v1_transformed['DIRECTION_WEIGHTS']['1d']:.3f}, "
          f"240m={v1_transformed['DIRECTION_WEIGHTS']['240m']:.3f}")
    print(f"  타이밍 가중치: 30m={v1_transformed['TIMING_WEIGHTS']['30m']:.3f}, "
          f"15m={v1_transformed['TIMING_WEIGHTS']['15m']:.3f}")
    print()

    print("v2 파라미터:")
    v2_transformed = params_v2.transform()
    print(f"  방향 가중치: 1d={v2_transformed['DIRECTION_WEIGHTS']['1d']:.3f}, "
          f"240m={v2_transformed['DIRECTION_WEIGHTS']['240m']:.3f}")
    print(f"  타이밍 가중치: 30m={v2_transformed['TIMING_WEIGHTS']['30m']:.3f}, "
          f"15m={v2_transformed['TIMING_WEIGHTS']['15m']:.3f}")
    print()

    # 각 코인별 평가
    print("=" * 70)
    print("2️⃣  코인별 평가")
    print("=" * 70)
    print()

    results = []

    for coin in TEST_COINS:
        try:
            result = evaluate_coin(coin, params_v1, params_v2)
            results.append(result)
        except Exception as e:
            print(f"❌ {coin} 평가 실패: {e}")
            print()

    # 결과 요약
    print("=" * 70)
    print("3️⃣  결과 요약")
    print("=" * 70)
    print()

    # 테이블 출력
    print(f"{'코인':<8} {'v1 점수':>10} {'v2 점수':>10} {'개선율':>10} {'결과':>8}")
    print("-" * 70)

    for r in results:
        improved_mark = "✅" if r['improved'] else "❌"
        print(f"{r['coin']:<8} {r['v1']['score']:>10.3f} {r['v2']['score']:>10.3f} "
              f"{r['improvement']:>9.1f}% {improved_mark:>8}")

    print()

    # 통계
    improved_count = sum(1 for r in results if r['improved'])
    total_count = len(results)
    avg_improvement = sum(r['improvement'] for r in results) / total_count if total_count > 0 else 0

    print("=" * 70)
    print("4️⃣  통계")
    print("=" * 70)
    print(f"개선된 코인:     {improved_count}/{total_count} ({improved_count/total_count*100:.0f}%)")
    print(f"평균 개선율:     {avg_improvement:+.1f}%")
    print()

    # 일반화 판정
    print("=" * 70)
    print("5️⃣  일반화 판정")
    print("=" * 70)
    print()

    if improved_count == total_count:
        print("✅ 완벽한 일반화!")
        print("   → v2 파라미터가 모든 코인에서 v1보다 우수합니다.")
        print("   → Orchestrator에 v2 통합을 권장합니다.")
    elif improved_count >= total_count * 0.75:
        print("⚠️  부분 일반화")
        print(f"   → v2 파라미터가 {improved_count}/{total_count} 코인에서 개선되었습니다.")
        print("   → 코인별 파라미터 학습을 고려하세요.")
    elif improved_count >= total_count * 0.5:
        print("⚠️  제한적 일반화")
        print(f"   → v2 파라미터가 절반 정도 코인에서만 개선되었습니다.")
        print("   → 코인별 파라미터 필수, 또는 v1 유지 권장")
    else:
        print("❌ 과적합 의심")
        print(f"   → v2 파라미터가 대부분 코인에서 v1보다 나쁩니다.")
        print("   → LINK에 과적합되었을 가능성이 높습니다.")
        print("   → v1 유지 권장")

    print()

    # 상세 비교 (LINK vs 다른 코인)
    print("=" * 70)
    print("6️⃣  LINK vs 다른 코인 비교")
    print("=" * 70)
    print()

    link_result = next((r for r in results if r['coin'] == 'LINK'), None)
    other_results = [r for r in results if r['coin'] != 'LINK']

    if link_result and other_results:
        link_improvement = link_result['improvement']
        other_improvements = [r['improvement'] for r in other_results]
        avg_other = sum(other_improvements) / len(other_improvements)

        print(f"LINK 개선율:           {link_improvement:+.1f}%")
        print(f"다른 코인 평균 개선율: {avg_other:+.1f}%")
        print(f"차이:                  {link_improvement - avg_other:+.1f}%p")
        print()

        if abs(link_improvement - avg_other) < 1.0:
            print("✅ LINK와 다른 코인들의 개선율이 유사합니다.")
            print("   → 과적합 가능성 낮음")
        elif link_improvement > avg_other + 2.0:
            print("⚠️  LINK의 개선율이 다른 코인보다 훨씬 높습니다.")
            print("   → LINK에 과적합되었을 가능성이 있습니다.")
        else:
            print("✅ LINK보다 다른 코인들의 개선율이 높습니다.")
            print("   → 일반화가 잘 되었습니다!")

    print()

    # 결과 저장
    output_file = '/workspace/v2_generalization_test.json'
    output_data = {
        'results': results,
        'statistics': {
            'improved_count': improved_count,
            'total_count': total_count,
            'success_rate': improved_count / total_count if total_count > 0 else 0,
            'avg_improvement': avg_improvement
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("=" * 70)
    print(f"결과 저장: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
