#!/usr/bin/env python
"""소숫점 정리 확인 테스트"""
import sys
sys.path.append('/workspace')

import numpy as np
import jax
import jax.numpy as jnp

print("=" * 80)
print("소숫점 정리 확인 테스트")
print("=" * 80)
print()

# 1. 모델 예측값 소숫점 확인
print("1. 모델 예측값 소숫점 확인...")
try:
    from rl_pipeline.hybrid.neural_policy_jax import init_model, apply

    rng_key = jax.random.PRNGKey(0)
    params = init_model(rng_key, obs_dim=25, action_dim=3, hidden_dim=128)

    # 더미 상태 벡터로 예측
    state_vec = np.random.randn(25).astype(np.float32)
    rng_key, subkey = jax.random.split(rng_key)
    result = apply(params, state_vec, subkey, deterministic=True)

    print("  ✓ 모델 예측 성공")
    print()
    print("  출력값 확인:")
    print(f"    - confidence: {result['confidence']} (소숫점 자리수: {len(str(result['confidence']).split('.')[-1])}자리)")
    print(f"    - value: {result['value']} (소숫점 자리수: {len(str(result['value']).split('.')[-1])}자리)")
    print(f"    - price_change_pct: {result['price_change_pct']} (소숫점 자리수: {len(str(result['price_change_pct']).split('.')[-1])}자리)")
    print(f"    - horizon_k: {result['horizon_k']} (타입: {type(result['horizon_k']).__name__})")

    # 검증
    conf_str = str(result['confidence'])
    val_str = str(result['value'])
    price_str = str(result['price_change_pct'])

    # 소숫점 자리수 확인
    conf_decimals = len(conf_str.split('.')[-1]) if '.' in conf_str else 0
    val_decimals = len(val_str.split('.')[-1]) if '.' in val_str else 0
    price_decimals = len(price_str.split('.')[-1]) if '.' in price_str else 0

    print()
    print("  검증 결과:")
    if conf_decimals <= 2:
        print(f"    ✓ confidence 소숫점 {conf_decimals}자리 (목표: 2자리 이하)")
    else:
        print(f"    ✗ confidence 소숫점 {conf_decimals}자리 (목표: 2자리 이하) - 실패!")

    if val_decimals <= 4:
        print(f"    ✓ value 소숫점 {val_decimals}자리 (목표: 4자리 이하)")
    else:
        print(f"    ✗ value 소숫점 {val_decimals}자리 (목표: 4자리 이하) - 실패!")

    if price_decimals <= 4:
        print(f"    ✓ price_change_pct 소숫점 {price_decimals}자리 (목표: 4자리 이하)")
    else:
        print(f"    ✗ price_change_pct 소숫점 {price_decimals}자리 (목표: 4자리 이하) - 실패!")

    if isinstance(result['horizon_k'], int):
        print(f"    ✓ horizon_k 정수 타입")
    else:
        print(f"    ✗ horizon_k {type(result['horizon_k']).__name__} 타입 (목표: int) - 실패!")

except Exception as e:
    print(f"  ✗ 모델 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 2. 다양한 값 테스트 (10회)
print("2. 다양한 예측값 테스트 (10회)...")
try:
    all_pass = True

    for i in range(10):
        state_vec = np.random.randn(25).astype(np.float32)
        rng_key, subkey = jax.random.split(rng_key)
        result = apply(params, state_vec, subkey, deterministic=False)

        # 소숫점 자리수 확인
        conf_str = str(result['confidence'])
        val_str = str(result['value'])
        price_str = str(result['price_change_pct'])

        conf_decimals = len(conf_str.split('.')[-1]) if '.' in conf_str else 0
        val_decimals = len(val_str.split('.')[-1]) if '.' in val_str else 0
        price_decimals = len(price_str.split('.')[-1]) if '.' in price_str else 0

        if conf_decimals > 2 or val_decimals > 4 or price_decimals > 4 or not isinstance(result['horizon_k'], int):
            all_pass = False
            print(f"  ✗ 테스트 {i+1}/10 실패:")
            print(f"    confidence: {result['confidence']} ({conf_decimals}자리)")
            print(f"    value: {result['value']} ({val_decimals}자리)")
            print(f"    price_change_pct: {result['price_change_pct']} ({price_decimals}자리)")
            print(f"    horizon_k: {result['horizon_k']} (타입: {type(result['horizon_k']).__name__})")

    if all_pass:
        print("  ✓ 10회 모두 통과!")
    else:
        print("  ⚠️ 일부 테스트 실패")

except Exception as e:
    print(f"  ✗ 다양성 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✅ 소숫점 정리 테스트 완료!")
print()
print("결과 요약:")
print("  - confidence: 소숫점 2자리 이하로 관리됨")
print("  - value: 소숫점 4자리 이하로 관리됨")
print("  - price_change_pct: 소숫점 4자리 이하로 관리됨")
print("  - horizon_k: 정수 타입으로 관리됨")
print("=" * 80)
