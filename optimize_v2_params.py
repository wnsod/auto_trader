#!/usr/bin/env python
"""
v2 파라미터 최적화 - Bayesian Optimization (⚠️ 실험 실패)

⚠️⚠️⚠️ 이 스크립트는 실험 실패로 폐기되었습니다 ⚠️⚠️⚠️

============================================================================
실험 결과: v2 파라미터 학습이 과적합됨 - v1 사용 권장
============================================================================

**이 스크립트의 결과:**
- LINK 전체 데이터로 최적화
- v2 점수: 0.562 (v1: 0.546, +2.9% 개선)
- 하지만 이것은 Train 데이터에 과적합된 결과였음!

**Walk-Forward Analysis 검증 결과:**
- Train 데이터: v2 0.580 > v1 0.560 (+3.6%)
- Test 데이터: v1 0.505 > v2 0.503 (-0.4%)  ← v1이 더 좋음!
- Test/Train: v2 86.7% < v1 90.2%  ← v2 과적합!

**문제점:**
1. Train 성능만 보고 좋다고 판단했지만, Test에서는 나쁨
2. 파라미터가 매우 불안정 (데이터에 따라 극단적으로 변함)
3. 전체 데이터로 최적화한 결과를 신뢰할 수 없음

**교훈:**
⚠️ 전체 데이터로 최적화하면 과적합 위험 높음
✅ 반드시 Train/Test split 후 검증 필요
✅ Test 성능이 Train보다 현저히 낮으면 과적합

**대신 사용할 것:**
→ walk_forward_analysis.py (Train/Test split으로 검증)
→ 하지만 그 결과도 v1이 더 우수함을 보임

**관련 문서:**
- INTEGRATED_ANALYSIS_V2_FINAL_REPORT.md

원래 목표: 조정 수익률 최대화
원래 방법: scikit-optimize (gp_minimize)

⚠️ 이 코드는 참고용으로만 보관. 실제 사용하지 말 것!
============================================================================
"""

import sys
sys.path.append('/workspace')

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import json
from datetime import datetime

from rl_pipeline.analysis.v2_backtest import objective_function, evaluate_params
from rl_pipeline.analysis.integrated_analysis_v2 import V2Parameters

# 최적화 설정
COIN = 'LINK'
N_CALLS = 30  # 30회 평가 (테스트용, 실제는 100+)
N_RANDOM_STARTS = 10  # 처음 10회는 랜덤
RANDOM_STATE = 42

# 파라미터 공간 정의 (14개)
space = [
    # direction_logits (2개)
    Real(-3, 3, name='direction_logit_0'),
    Real(-3, 3, name='direction_logit_1'),

    # timing_logits (2개)
    Real(-3, 3, name='timing_logit_0'),
    Real(-3, 3, name='timing_logit_1'),

    # grade_logits (3개)
    Real(-3, 3, name='grade_logit_0'),
    Real(-3, 3, name='grade_logit_1'),
    Real(-3, 3, name='grade_logit_2'),

    # log_half_life (1개)
    Real(np.log(3), np.log(30), name='log_half_life'),  # 3~30일

    # log_direction_threshold (1개)
    Real(np.log(0.005), np.log(0.05), name='log_direction_threshold'),  # 0.5%~5%

    # log_timing_threshold (1개)
    Real(np.log(0.001), np.log(0.02), name='log_timing_threshold'),  # 0.1%~2%

    # logit_min_position (1개)
    Real(-3, 3, name='logit_min_position'),  # sigmoid → 0.05~0.95

    # confidence_logits (3개)
    Real(-3, 3, name='confidence_logit_0'),
    Real(-3, 3, name='confidence_logit_1'),
    Real(-3, 3, name='confidence_logit_2'),
]

print("=" * 70)
print("v2 파라미터 최적화")
print("=" * 70)
print(f"코인:            {COIN}")
print(f"평가 횟수:       {N_CALLS}")
print(f"랜덤 초기화:     {N_RANDOM_STARTS}")
print(f"파라미터 공간:   14차원")
print()

# v1 기본 파라미터 평가
print("=" * 70)
print("v1 기본 파라미터 (베이스라인)")
print("=" * 70)

params_v1 = V2Parameters()
result_v1 = evaluate_params(params_v1, COIN)

print(f"점수:            {result_v1['score']:.3f}")
print(f"방향:            {result_v1['direction']}")
print(f"타이밍:          {result_v1['timing']}")
print(f"크기:            {result_v1['size']:.3f}")
print()

v1_score = result_v1['score']

# 최적화 실행
print("=" * 70)
print("Bayesian Optimization 시작")
print("=" * 70)
print()

start_time = datetime.now()

result = gp_minimize(
    objective_function,
    space,
    n_calls=N_CALLS,
    n_random_starts=N_RANDOM_STARTS,
    random_state=RANDOM_STATE,
    verbose=True
)

end_time = datetime.now()
elapsed = (end_time - start_time).total_seconds()

print()
print("=" * 70)
print("최적화 완료")
print("=" * 70)
print(f"소요 시간:       {elapsed:.1f}초")
print(f"평가 횟수:       {len(result.func_vals)}")
print()

# 최적 파라미터
best_raw_params = result.x
best_score = -result.fun  # objective는 -score를 반환

params_best = V2Parameters(best_raw_params)
result_best = evaluate_params(params_best, COIN)

print("=" * 70)
print("최적 파라미터")
print("=" * 70)
print(params_best)
print()

print("=" * 70)
print("결과")
print("=" * 70)
print(f"점수:            {result_best['score']:.3f}")
print(f"방향:            {result_best['direction']}")
print(f"타이밍:          {result_best['timing']}")
print(f"크기:            {result_best['size']:.3f}")
print(f"확신도:          {result_best['confidence']:.3f}")
print(f"기간:            {result_best['horizon']}")
print()

# 비교
print("=" * 70)
print("v1 vs v2 비교")
print("=" * 70)
print(f"v1 점수:         {v1_score:.3f}")
print(f"v2 점수:         {result_best['score']:.3f}")
print(f"개선:            {result_best['score'] - v1_score:.3f} ({(result_best['score'] / v1_score - 1) * 100:.1f}%)")
print()

if result_best['score'] > v1_score:
    print("✅ v2가 v1보다 {:.1f}% 개선되었습니다!".format((result_best['score'] / v1_score - 1) * 100))
else:
    print("❌ v2가 v1보다 나빠졌습니다. v1 파라미터를 유지하세요.")

print()

# 최적화 이력
print("=" * 70)
print("최적화 이력")
print("=" * 70)
scores = [-f for f in result.func_vals]
print(f"최고 점수: {max(scores):.3f}")
print(f"최저 점수: {min(scores):.3f}")
print(f"평균 점수: {np.mean(scores):.3f}")
print()

# Top 5
top5_indices = np.argsort(scores)[-5:][::-1]
print("Top 5 평가:")
for i, idx in enumerate(top5_indices, 1):
    print(f"{i}. 점수: {scores[idx]:.3f}")

print()

# 파라미터 저장
output_file = f"/workspace/v2_params_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

params_dict = params_best.transform()
params_dict['raw_params'] = best_raw_params
params_dict['score'] = result_best['score']
params_dict['coin'] = COIN
params_dict['n_calls'] = N_CALLS
params_dict['timestamp'] = datetime.now().isoformat()

with open(output_file, 'w') as f:
    json.dump(params_dict, f, indent=2, default=str)

print("=" * 70)
print(f"최적 파라미터 저장: {output_file}")
print("=" * 70)
