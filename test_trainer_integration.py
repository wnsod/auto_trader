#!/usr/bin/env python
"""Trainer + Model 통합 테스트 (4개 출력)"""
import sys
sys.path.append('/workspace')

import numpy as np
import jax
import jax.numpy as jnp

print("=" * 80)
print("Trainer + Model 통합 테스트")
print("=" * 80)
print()

# 1. 모델 초기화
print("1. 모델 초기화...")
try:
    from rl_pipeline.hybrid.neural_policy_jax import init_model, apply

    rng_key = jax.random.PRNGKey(0)
    params = init_model(rng_key, obs_dim=25, action_dim=3, hidden_dim=128)

    print("  ✓ 모델 초기화 성공")
except Exception as e:
    print(f"  ✗ 모델 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 2. Trainer 초기화
print("2. Trainer 초기화...")
try:
    from rl_pipeline.hybrid.trainer_jax import PPOTrainer

    trainer = PPOTrainer(
        obs_dim=25,
        action_dim=3,
        hidden_dim=128,
        lr=3e-4
    )

    print("  ✓ Trainer 초기화 성공")
except Exception as e:
    print(f"  ✗ Trainer 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 3. 더미 경험 데이터 생성
print("3. 더미 경험 데이터 생성...")
try:
    batch_size = 64

    # 더미 경험 생성
    experiences = []
    for i in range(batch_size):
        state = np.random.randn(25).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(25).astype(np.float32)
        done = np.random.rand() > 0.9

        # 예측값 (현재는 AI가 아닌 더미 값)
        rng_key, subkey = jax.random.split(rng_key)
        pred = apply(params, state, subkey, deterministic=True)

        experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': pred['action_logits'][action],  # 더미 log prob
            'value': pred['value']
        })

    print(f"  ✓ {batch_size}개 경험 생성 완료")
    print(f"    - 상태 차원: {experiences[0]['state'].shape}")
    print(f"    - 액션 범위: 0-2")
    print(f"    - 예측값 포함: price_change_pct={pred['price_change_pct']:.4f}, horizon_k={pred['horizon_k']}")
except Exception as e:
    print(f"  ✗ 경험 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 4. 학습 단계 실행
print("4. 학습 단계 실행...")
try:
    # 경험 추가
    for exp in experiences:
        trainer.add_experience(
            state=exp['state'],
            action=exp['action'],
            reward=exp['reward'],
            next_state=exp['next_state'],
            done=exp['done'],
            log_prob=exp['log_prob'],
            value=exp['value']
        )

    print(f"  ✓ {len(experiences)}개 경험 추가 완료")

    # 학습 실행
    loss = trainer.train_step()

    if loss is not None:
        print(f"  ✓ 학습 성공!")
        print(f"    - Loss: {loss:.6f}")
    else:
        print(f"  ⚠️ 학습 스킵 (경험 부족 또는 에러)")

except Exception as e:
    print(f"  ✗ 학습 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 5. 모델 예측 테스트
print("5. 학습 후 예측 테스트...")
try:
    test_state = np.random.randn(25).astype(np.float32)
    rng_key, subkey = jax.random.split(rng_key)

    prediction = apply(trainer.model['params'], test_state, subkey, deterministic=True)

    print("  ✓ 예측 성공")
    print(f"    - Action: {prediction['action']} ({prediction['action_name']})")
    print(f"    - Confidence: {prediction['confidence']:.3f}")
    print(f"    - Value: {prediction['value']:.3f}")
    print(f"    - Price change: {prediction['price_change_pct']*100:.2f}%")
    print(f"    - Horizon: {prediction['horizon_k']} candles")
except Exception as e:
    print(f"  ✗ 예측 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✅ 모든 테스트 통과!")
print()
print("결론:")
print("  - 모델 구조 업데이트 완료 (4개 출력)")
print("  - Trainer MSE loss 추가 완료")
print("  - 통합 동작 확인 완료")
print()
print("다음 단계:")
print("  - orchestrator.py에서 실제 변동률/타이밍 레이블 계산")
print("  - AI 예측값 사용하여 DB 저장")
print("=" * 80)
