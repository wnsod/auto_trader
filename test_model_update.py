#!/usr/bin/env python
"""모델 구조 업데이트 테스트"""
import sys
sys.path.append('/workspace')

import numpy as np

print("=" * 80)
print("모델 구조 업데이트 테스트")
print("=" * 80)
print()

# 1. 모델 초기화 테스트
print("1. 모델 초기화 테스트...")
try:
    from rl_pipeline.hybrid.neural_policy_jax import init_model, apply
    import jax

    rng_key = jax.random.PRNGKey(0)
    params = init_model(rng_key, obs_dim=25, action_dim=3, hidden_dim=128)

    print("  ✓ 모델 초기화 성공")
    print(f"    - obs_dim: {params['obs_dim']}")
    print(f"    - action_dim: {params['action_dim']}")
    print(f"    - hidden_dim: {params['hidden_dim']}")
except Exception as e:
    print(f"  ✗ 모델 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 2. 순전파 테스트
print("2. 순전파 테스트 (4개 출력 확인)...")
try:
    # 더미 상태 벡터 생성
    state_vec = np.random.randn(25).astype(np.float32)

    rng_key, subkey = jax.random.split(rng_key)
    result = apply(params, state_vec, subkey, deterministic=True)

    print("  ✓ 순전파 성공")
    print()
    print("  출력:")
    print(f"    - action: {result['action']} ({result['action_name']})")
    print(f"    - confidence: {result['confidence']:.3f}")
    print(f"    - value: {result['value']:.3f}")
    print(f"    - price_change_pct: {result['price_change_pct']:.4f} ({result['price_change_pct']*100:.2f}%)")
    print(f"    - horizon_k: {result['horizon_k']} 캔들")

    # 범위 확인
    assert -0.1 <= result['price_change_pct'] <= 0.1, f"price_change_pct 범위 오류: {result['price_change_pct']}"
    assert 1 <= result['horizon_k'] <= 20, f"horizon_k 범위 오류: {result['horizon_k']}"

    print()
    print("  ✓ 출력 범위 검증 통과")
    print(f"    - price_change_pct: -10% ~ +10% 범위 내")
    print(f"    - horizon_k: 1 ~ 20 캔들 범위 내")

except Exception as e:
    print(f"  ✗ 순전파 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 3. 여러 번 예측 테스트 (다양성 확인)
print("3. 다양한 예측값 생성 테스트 (10회)...")
try:
    price_changes = []
    horizons = []

    for i in range(10):
        state_vec = np.random.randn(25).astype(np.float32)
        rng_key, subkey = jax.random.split(rng_key)
        result = apply(params, state_vec, subkey, deterministic=False)

        price_changes.append(result['price_change_pct'])
        horizons.append(result['horizon_k'])

    print("  ✓ 10회 예측 완료")
    print()
    print("  변동률 예측 분포:")
    print(f"    평균: {np.mean(price_changes)*100:.2f}%")
    print(f"    최소: {np.min(price_changes)*100:.2f}%")
    print(f"    최대: {np.max(price_changes)*100:.2f}%")
    print(f"    표준편차: {np.std(price_changes)*100:.2f}%")
    print()
    print("  타이밍 예측 분포:")
    print(f"    평균: {np.mean(horizons):.1f} 캔들")
    print(f"    최소: {np.min(horizons)} 캔들")
    print(f"    최대: {np.max(horizons)} 캔들")
    print(f"    표준편차: {np.std(horizons):.1f} 캔들")

except Exception as e:
    print(f"  ✗ 다양성 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✅ 모든 테스트 통과!")
print()
print("결과:")
print("  - 모델이 방향(UP/DOWN/NEUTRAL) 예측")
print("  - 모델이 변동률(-10% ~ +10%) 예측")
print("  - 모델이 타이밍(1~20 캔들) 예측")
print("=" * 80)
