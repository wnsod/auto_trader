#!/usr/bin/env python
"""
Walk-Forward Analysis for v2 Parameters (✅ 성공적인 검증 도구)

LINK 데이터를 시간별로 나눠서 과적합 여부 검증
Train 데이터로 최적화, Test 데이터로 평가

============================================================================
이 스크립트의 의의: v2의 과적합을 발견한 핵심 도구 ✅
============================================================================

**역할:**
- Train/Test split으로 파라미터 학습의 과적합 여부 검증
- 시간 순서를 유지한 검증 (Walk-Forward)
- v2가 실패했음을 입증한 결정적 증거 제공

**결과:**
┌──────────────┬──────────┬──────────┬────────────┐
│ 데이터셋     │ v1 점수  │ v2 점수  │ Test/Train │
├──────────────┼──────────┼──────────┼────────────┤
│ Train (70%)  │ 0.560    │ 0.580    │ -          │
│ Test (30%)   │ 0.505    │ 0.503    │ -          │
│ 안정성       │ 90.2%    │ 86.7%    │ v2 과적합! │
└──────────────┴──────────┴──────────┴────────────┘

**핵심 발견:**
✅ v2가 Train에서는 좋지만 (+3.6%), Test에서는 나쁨 (-0.4%)
✅ v2의 Test/Train 비율 86.7% < 90% (과적합 기준)
✅ v1이 더 안정적 (90.2%)

**교훈:**
1. ✅ Train 성능만 보면 안 됨 - Test 검증 필수
2. ✅ Walk-Forward Analysis는 과적합 감지에 매우 유용
3. ✅ Test/Train 비율이 중요한 지표
4. ✅ 단순 모델(v1)이 복잡 모델(v2)보다 나을 수 있음

**향후 사용:**
→ 새로운 파라미터 학습 시도 시 이 스크립트로 검증 필수
→ 다른 ML/최적화 접근법 검증에도 사용 가능
→ 코인별 파라미터 학습 시에도 적용 가능

**관련 문서:**
- INTEGRATED_ANALYSIS_V2_FINAL_REPORT.md (전체 결과)
- walk_forward_analysis_result.json (검증 데이터)

⚠️ 이 코드는 유용한 검증 도구이므로 보관 및 재사용 권장!
============================================================================
"""

import sys
sys.path.append('/workspace')

import numpy as np
import sqlite3
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real
import json

from rl_pipeline.analysis.integrated_analysis_v2 import V2Parameters, IntegratedAnalyzerV2

# 설정
COIN = 'LINK'
DB_PATH = '/workspace/data_storage/rl_strategies.db'
TRAIN_RATIO = 0.7  # 70% Train, 30% Test
N_CALLS = 30
N_RANDOM_STARTS = 10
RANDOM_STATE = 42

# 파라미터 공간 (optimize_v2_params.py와 동일)
space = [
    Real(-3, 3, name='direction_logit_0'),
    Real(-3, 3, name='direction_logit_1'),
    Real(-3, 3, name='timing_logit_0'),
    Real(-3, 3, name='timing_logit_1'),
    Real(-3, 3, name='grade_logit_0'),
    Real(-3, 3, name='grade_logit_1'),
    Real(-3, 3, name='grade_logit_2'),
    Real(np.log(3), np.log(30), name='log_half_life'),
    Real(np.log(0.005), np.log(0.05), name='log_direction_threshold'),
    Real(np.log(0.001), np.log(0.02), name='log_timing_threshold'),
    Real(-3, 3, name='logit_min_position'),
    Real(-3, 3, name='confidence_logit_0'),
    Real(-3, 3, name='confidence_logit_1'),
    Real(-3, 3, name='confidence_logit_2'),
]


def split_data_by_time(coin: str, train_ratio: float = 0.7):
    """
    전략 데이터를 시간 기준으로 Train/Test 분할

    Returns:
        train_strategy_ids: Train 전략 ID 리스트
        test_strategy_ids: Test 전략 ID 리스트
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # LINK 전략들을 last_updated 기준 정렬
    cursor.execute("""
        SELECT strategy_id, updated_at
        FROM rl_strategy_rollup
        WHERE coin = ?
        ORDER BY updated_at
    """, (coin,))

    results = cursor.fetchall()
    conn.close()

    if not results:
        raise ValueError(f"No data found for coin {coin}")

    strategy_ids = [r[0] for r in results]
    timestamps = [r[1] for r in results]

    # Train/Test split
    split_idx = int(len(strategy_ids) * train_ratio)
    train_ids = strategy_ids[:split_idx]
    test_ids = strategy_ids[split_idx:]

    # 시간 범위 출력
    train_start = datetime.fromtimestamp(timestamps[0])
    train_end = datetime.fromtimestamp(timestamps[split_idx - 1])
    test_start = datetime.fromtimestamp(timestamps[split_idx])
    test_end = datetime.fromtimestamp(timestamps[-1])

    print(f"Train 전략 개수: {len(train_ids)}")
    print(f"Train 기간: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
    print(f"Test 전략 개수: {len(test_ids)}")
    print(f"Test 기간: {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")
    print()

    return train_ids, test_ids


def backtest_with_subset(raw_params, coin: str, strategy_ids):
    """
    특정 전략 subset으로 백테스트

    Args:
        raw_params: 14개 raw 파라미터
        coin: 코인 심볼
        strategy_ids: 사용할 전략 ID 리스트

    Returns:
        score: 조정 수익률
    """
    try:
        params = V2Parameters(raw_params)
        analyzer = IntegratedAnalyzerV2(params, DB_PATH)

        # 인터벌 데이터 로드
        interval_data = analyzer._load_interval_data(coin)

        if not interval_data:
            return -10.0

        # strategy_ids 세트로 변환 (빠른 검색)
        valid_ids = set(strategy_ids)

        returns = []
        weights = []

        for interval in ['15m', '30m', '240m', '1d']:
            if interval_data.get(interval) and interval_data[interval]:
                data = interval_data[interval]
                strategies = data['strategies']

                if not strategies:
                    continue

                # valid_ids에 있는 전략만 사용
                for s in strategies:
                    if s['strategy_id'] not in valid_ids:
                        continue

                    avg_ret = s.get('avg_ret', 0.0)
                    win_rate = s.get('win_rate', 0.0)
                    weight = s.get('total_weight', 0.0)

                    if weight > 0 and avg_ret is not None:
                        adjusted_return = avg_ret * win_rate
                        returns.append(adjusted_return)
                        weights.append(weight)

        if not returns:
            return -10.0

        returns = np.array(returns)
        weights = np.array(weights)

        weighted_return = np.sum(returns * weights) / np.sum(weights)
        scaled_score = weighted_return * 10

        return scaled_score

    except Exception as e:
        print(f"백테스트 오류: {e}")
        return -10.0


def objective_train(raw_params):
    """Train 데이터로 최적화하는 목적 함수"""
    global train_strategy_ids
    score = backtest_with_subset(raw_params, COIN, train_strategy_ids)
    return -score


def main():
    global train_strategy_ids

    print("=" * 70)
    print("Walk-Forward Analysis")
    print("=" * 70)
    print()

    # 1. 데이터 분할
    print("=" * 70)
    print("1️⃣  데이터 분할")
    print("=" * 70)
    print()

    train_strategy_ids, test_strategy_ids = split_data_by_time(COIN, TRAIN_RATIO)

    # 2. v1 기본 파라미터 평가 (베이스라인)
    print("=" * 70)
    print("2️⃣  v1 베이스라인 평가")
    print("=" * 70)
    print()

    params_v1 = V2Parameters()
    v1_raw = params_v1.to_raw()

    v1_train_score = backtest_with_subset(v1_raw, COIN, train_strategy_ids)
    v1_test_score = backtest_with_subset(v1_raw, COIN, test_strategy_ids)

    print(f"v1 Train 점수: {v1_train_score:.3f}")
    print(f"v1 Test 점수:  {v1_test_score:.3f}")

    if v1_train_score > 0:
        v1_ratio = v1_test_score / v1_train_score
        print(f"v1 Test/Train: {v1_ratio:.3f} ({v1_ratio * 100:.1f}%)")
    print()

    # 3. Train 데이터로 v2 최적화
    print("=" * 70)
    print("3️⃣  Train 데이터로 v2 최적화")
    print("=" * 70)
    print()

    print(f"평가 횟수: {N_CALLS}")
    print(f"랜덤 초기화: {N_RANDOM_STARTS}")
    print()

    start_time = datetime.now()

    result = gp_minimize(
        objective_train,
        space,
        n_calls=N_CALLS,
        n_random_starts=N_RANDOM_STARTS,
        random_state=RANDOM_STATE,
        verbose=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print()
    print(f"최적화 완료 (소요 시간: {elapsed:.1f}초)")
    print()

    # 4. 최적 파라미터로 Train/Test 평가
    print("=" * 70)
    print("4️⃣  최적 파라미터 평가")
    print("=" * 70)
    print()

    best_raw_params = result.x
    params_v2 = V2Parameters(best_raw_params)

    v2_train_score = backtest_with_subset(best_raw_params, COIN, train_strategy_ids)
    v2_test_score = backtest_with_subset(best_raw_params, COIN, test_strategy_ids)

    print("v2 최적 파라미터:")
    print(params_v2)
    print()

    print(f"v2 Train 점수: {v2_train_score:.3f}")
    print(f"v2 Test 점수:  {v2_test_score:.3f}")

    if v2_train_score > 0:
        v2_ratio = v2_test_score / v2_train_score
        print(f"v2 Test/Train: {v2_ratio:.3f} ({v2_ratio * 100:.1f}%)")
    print()

    # 5. 결과 비교
    print("=" * 70)
    print("5️⃣  결과 비교")
    print("=" * 70)
    print()

    # 테이블 출력
    print(f"{'데이터셋':<15} {'v1 점수':>10} {'v2 점수':>10} {'개선율':>10}")
    print("-" * 70)

    if v1_train_score > 0:
        train_improvement = ((v2_train_score / v1_train_score) - 1) * 100
    else:
        train_improvement = 0.0

    if v1_test_score > 0:
        test_improvement = ((v2_test_score / v1_test_score) - 1) * 100
    else:
        test_improvement = 0.0

    print(f"{'Train (70%)':<15} {v1_train_score:>10.3f} {v2_train_score:>10.3f} {train_improvement:>9.1f}%")
    print(f"{'Test (30%)':<15} {v1_test_score:>10.3f} {v2_test_score:>10.3f} {test_improvement:>9.1f}%")
    print()

    # 6. 과적합 판정
    print("=" * 70)
    print("6️⃣  과적합 판정")
    print("=" * 70)
    print()

    if v2_train_score <= 0:
        print("❌ Train 데이터에서 유효한 점수를 얻지 못했습니다.")
        print("   → 데이터 또는 최적화 문제")
    elif v2_test_score <= 0:
        print("❌ Test 데이터에서 유효한 점수를 얻지 못했습니다.")
        print("   → 심각한 과적합 또는 데이터 문제")
    else:
        if v2_ratio >= 0.95:
            print(f"✅ 과적합 없음! (Test/Train = {v2_ratio:.3f})")
            print("   → v2 파라미터가 시간적으로 안정적입니다.")
            print("   → Orchestrator에 v2 통합을 권장합니다.")
        elif v2_ratio >= 0.90:
            print(f"⚠️  약간 과적합 (Test/Train = {v2_ratio:.3f})")
            print("   → Test 성능이 Train보다 약간 낮습니다.")
            print("   → 주의하며 사용 가능, 실전 검증 필요")
        else:
            print(f"❌ 심각한 과적합! (Test/Train = {v2_ratio:.3f})")
            print("   → Test 성능이 Train보다 훨씬 낮습니다.")
            print("   → v1 유지 권장")

    print()

    # v1과 비교
    print("=" * 70)
    print("7️⃣  v1 vs v2 종합 비교")
    print("=" * 70)
    print()

    if v1_train_score > 0 and v2_train_score > 0:
        print("Test/Train 비율:")
        print(f"  v1: {v1_ratio:.3f}")
        print(f"  v2: {v2_ratio:.3f}")
        print()

        if v2_ratio >= v1_ratio and v2_test_score > v1_test_score:
            print("✅ v2가 v1보다 우수합니다!")
            print("   → 과적합도 낮고, Test 성능도 높음")
            print("   → v2 사용 강력 권장")
        elif v2_test_score > v1_test_score:
            print("⚠️  v2가 Test 성능은 좋지만, 과적합도가 높습니다.")
            print("   → 실전 검증 후 사용 결정")
        else:
            print("❌ v1이 더 안정적입니다.")
            print("   → v1 유지 권장")

    print()

    # 결과 저장
    output_file = '/workspace/walk_forward_analysis_result.json'
    output_data = {
        'train_ratio': TRAIN_RATIO,
        'train_count': len(train_strategy_ids),
        'test_count': len(test_strategy_ids),
        'v1': {
            'train_score': float(v1_train_score),
            'test_score': float(v1_test_score),
            'test_train_ratio': float(v1_ratio) if v1_train_score > 0 else 0.0
        },
        'v2': {
            'train_score': float(v2_train_score),
            'test_score': float(v2_test_score),
            'test_train_ratio': float(v2_ratio) if v2_train_score > 0 else 0.0,
            'raw_params': [float(x) for x in best_raw_params]
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
