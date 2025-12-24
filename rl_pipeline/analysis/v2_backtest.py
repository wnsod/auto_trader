#!/usr/bin/env python
"""
v2 백테스트 함수 (⚠️ v2와 함께 폐기됨)

기존 전략 성능 데이터를 활용한 빠른 백테스트

⚠️⚠️⚠️ v2 파라미터 학습이 실패하여 이 파일도 폐기됨 ⚠️⚠️⚠️

============================================================================
이 파일의 역할과 한계
============================================================================

**역할:**
- v2 파라미터 최적화를 위한 목적 함수(objective function) 제공
- 기존 전략 성능 데이터(rl_strategy_rollup)로 빠르게 평가
- Bayesian Optimization의 평가 함수로 사용

**방법:**
1. V2Parameters로 파라미터 생성
2. IntegratedAnalyzerV2로 전략 가중치 계산
3. 가중 평균 조정 수익률 (avg_ret × win_rate × weight) 반환
4. 점수를 10배 스케일링하여 최적화에 적합하게 변환

**한계와 문제:**
1. ⚠️ 기존 데이터만 활용 → 새로운 시장 환경에 대응 못함
2. ⚠️ Sharpe Ratio가 모두 0이라 avg_ret로 대체 → 리스크 미고려
3. ⚠️ 단순 가중 평균 → 실제 거래 시뮬레이션 부족
4. ⚠️ 과적합 방지 메커니즘 없음

**v2 실패 원인과의 관계:**
- 이 백테스트 함수 자체는 잘 작동했음
- 하지만 평가 방식이 너무 단순해서 실전 성능 예측 못함
- Train 데이터에 과적합된 파라미터를 찾아냄
- → Walk-Forward Analysis로 검증했을 때 Test 성능이 나쁨

**교훈:**
✅ 백테스트 함수가 Train 데이터에서 좋은 점수를 찾아도
   Test 데이터에서 나쁠 수 있음 (과적합)
✅ 더 정교한 평가 함수 필요 (Sharpe, 리스크 고려 등)
✅ 단순 가중 평균 대신 실제 시뮬레이션이 더 정확

**관련 문서:**
- INTEGRATED_ANALYSIS_V2_FINAL_REPORT.md

⚠️ 이 코드는 참고용으로만 보관. 실제 사용하지 말 것!
============================================================================
"""

import sys
import os
sys.path.append('/workspace')

import numpy as np
from typing import List, Dict
from rl_pipeline.analysis.integrated_analysis_v2 import V2Parameters, IntegratedAnalyzerV2
from rl_pipeline.core.env import config


def simple_backtest(raw_params: List[float],
                   coin: str = 'LINK',
                   db_path: str = None) -> float:
    """
    간단한 백테스트 - 현재 데이터 기준으로 성능 평가

    전략:
    1. raw_params로 V2Parameters 생성
    2. v2 analyzer로 코인 분석
    3. 선택된 전략들의 가중 평균 Sharpe Ratio 반환

    Args:
        raw_params: 14개 raw 파라미터
        coin: 코인 심볼
        db_path: DB 경로

    Returns:
        Sharpe Ratio (음수면 나쁨, 양수면 좋음)
    """
    try:
        if db_path is None:
            db_path = config.STRATEGIES_DB

        # 파라미터 생성
        params = V2Parameters(raw_params)

        # v2 analyzer 생성
        analyzer = IntegratedAnalyzerV2(params, db_path)

        # 인터벌 데이터 로드
        interval_data = analyzer._load_interval_data(coin)

        if not interval_data:
            return -10.0  # 페널티

        # 각 인터벌별 avg_ret 수집 (sharpe_ratio가 0이므로 avg_ret 사용)
        returns = []
        weights = []

        for interval in ['15m', '30m', '240m', '1d']:
            if interval_data.get(interval) and interval_data[interval]:
                data = interval_data[interval]
                strategies = data['strategies']

                if not strategies:
                    continue

                # 전략들의 avg_ret 가중 평균
                for s in strategies:
                    avg_ret = s.get('avg_ret', 0.0)
                    win_rate = s.get('win_rate', 0.0)
                    weight = s.get('total_weight', 0.0)

                    if weight > 0 and avg_ret is not None:
                        # 조정된 수익률 = avg_ret * win_rate (리스크 고려)
                        adjusted_return = avg_ret * win_rate
                        returns.append(adjusted_return)
                        weights.append(weight)

        if not returns:
            return -10.0  # 페널티

        # 가중 평균 조정 수익률
        returns = np.array(returns)
        weights = np.array(weights)

        weighted_return = np.sum(returns * weights) / np.sum(weights)

        # 수익률을 0~1 범위로 스케일링 (최적화용)
        # avg_ret은 보통 -0.05 ~ 0.20 범위
        # 10배 스케일링하여 -0.5 ~ 2.0 범위로
        scaled_score = weighted_return * 10

        return scaled_score

    except Exception as e:
        print(f"백테스트 오류: {e}")
        return -10.0  # 페널티


def objective_function(raw_params: List[float]) -> float:
    """
    Bayesian Optimization용 목적 함수

    목표: 조정 수익률 최대화
    → Minimize -score (음수로 변환)

    Args:
        raw_params: 14개 raw 파라미터

    Returns:
        -score (최소화 대상)
    """
    score = simple_backtest(raw_params)
    return -score  # Minimize -score = Maximize score


def evaluate_params(params: V2Parameters, coin: str = 'LINK') -> Dict:
    """
    파라미터 평가 (상세)

    Args:
        params: V2Parameters
        coin: 코인 심볼

    Returns:
        평가 결과 딕셔너리
    """
    raw_params = params.to_raw()
    score = simple_backtest(raw_params, coin)

    # v2로 시그널 생성
    analyzer = IntegratedAnalyzerV2(params)
    signal = analyzer.analyze(coin)

    return {
        'score': score,
        'direction': signal['direction'],
        'timing': signal['timing'],
        'size': signal['size'],
        'confidence': signal['confidence'],
        'horizon': signal['horizon']
    }


if __name__ == '__main__':
    # 테스트: v1 기본 파라미터 평가
    print("=" * 70)
    print("v2 백테스트 테스트")
    print("=" * 70)
    print()

    # v1 기본 파라미터
    params_v1 = V2Parameters()

    print("📊 v1 기본 파라미터 평가:")
    print(params_v1)
    print()

    result = evaluate_params(params_v1, 'LINK')

    print("결과:")
    print(f"  점수:         {result['score']:.3f}")
    print(f"  방향:         {result['direction']}")
    print(f"  타이밍:       {result['timing']}")
    print(f"  크기:         {result['size']:.3f}")
    print(f"  확신도:       {result['confidence']:.3f}")
    print(f"  기간:         {result['horizon']}")
    print()

    # 랜덤 파라미터 테스트
    print("=" * 70)
    print("랜덤 파라미터 테스트")
    print("=" * 70)
    print()

    np.random.seed(42)
    random_raw = np.random.randn(14).tolist()

    params_random = V2Parameters(random_raw)
    print("📊 랜덤 파라미터:")
    print(params_random)
    print()

    result_random = evaluate_params(params_random, 'LINK')

    print("결과:")
    print(f"  점수:         {result_random['score']:.3f}")
    print(f"  방향:         {result_random['direction']}")
    print(f"  타이밍:       {result_random['timing']}")
    print(f"  크기:         {result_random['size']:.3f}")
    print(f"  확신도:       {result_random['confidence']:.3f}")
    print(f"  기간:         {result_random['horizon']}")
    print()

    # 비교
    print("=" * 70)
    print("비교")
    print("=" * 70)
    print(f"v1 점수:       {result['score']:.3f}")
    print(f"랜덤 점수:     {result_random['score']:.3f}")

    if result['score'] > result_random['score']:
        print("→ v1이 더 좋습니다")
    else:
        print("→ 랜덤이 더 좋습니다 (운이 좋았네요!)")
