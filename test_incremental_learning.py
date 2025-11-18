#!/usr/bin/env python
"""
증분 학습 테스트 스크립트

Phase 1-3 구현 검증:
1. 유사도 계산 (기본, 스마트)
2. 전략 분류 (duplicate, copy, finetune, novel)
3. 동적 에피소드 조정
"""

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("=" * 80)
print("증분 학습 (Incremental Learning) 테스트")
print("=" * 80)

# 1. 유사도 계산 함수 테스트
print("\n1️⃣ Phase 1: 기본 유사도 계산 테스트")
print("-" * 80)

from rl_pipeline.strategy.similarity import (
    vectorize_strategy_params,
    calculate_basic_similarity,
    calculate_smart_similarity,
    calculate_finetuning_episodes,
    classify_strategy_by_similarity
)

# 테스트 전략들
strategy1 = {
    'id': 'test_1',
    'rsi_min': 30.0,
    'rsi_max': 70.0,
    'volume_ratio_min': 1.0,
    'volume_ratio_max': 2.0,
    'macd_buy_threshold': 0.01,
    'macd_sell_threshold': -0.01,
    'mfi_min': 20.0,
    'mfi_max': 80.0,
    'atr_min': 0.01,
    'atr_max': 0.05,
    'adx_min': 15.0,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'regime': 'ranging',
    'strategy_type': 'hybrid'
}

# 거의 동일한 전략 (duplicate)
strategy2 = {
    'id': 'test_2',
    'rsi_min': 30.0,
    'rsi_max': 70.0,
    'volume_ratio_min': 1.0,
    'volume_ratio_max': 2.0,
    'macd_buy_threshold': 0.01,
    'macd_sell_threshold': -0.01,
    'mfi_min': 20.0,
    'mfi_max': 80.0,
    'atr_min': 0.01,
    'atr_max': 0.05,
    'adx_min': 15.0,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'regime': 'ranging',
    'strategy_type': 'hybrid'
}

# 매우 유사한 전략 (copy)
strategy3 = {
    'id': 'test_3',
    'rsi_min': 31.0,  # 약간 다름
    'rsi_max': 69.0,
    'volume_ratio_min': 1.05,
    'volume_ratio_max': 2.05,
    'macd_buy_threshold': 0.012,
    'macd_sell_threshold': -0.012,
    'mfi_min': 21.0,
    'mfi_max': 79.0,
    'atr_min': 0.011,
    'atr_max': 0.049,
    'adx_min': 14.5,
    'stop_loss_pct': 0.021,
    'take_profit_pct': 0.041,
    'regime': 'ranging',
    'strategy_type': 'hybrid'
}

# 어느 정도 유사한 전략 (finetune)
strategy4 = {
    'id': 'test_4',
    'rsi_min': 25.0,  # 중간 정도 다름
    'rsi_max': 75.0,
    'volume_ratio_min': 1.2,
    'volume_ratio_max': 2.5,
    'macd_buy_threshold': 0.02,
    'macd_sell_threshold': -0.02,
    'mfi_min': 15.0,
    'mfi_max': 85.0,
    'atr_min': 0.015,
    'atr_max': 0.06,
    'adx_min': 18.0,
    'stop_loss_pct': 0.025,
    'take_profit_pct': 0.05,
    'regime': 'ranging',
    'strategy_type': 'hybrid'
}

# 완전히 다른 전략 (novel)
strategy5 = {
    'id': 'test_5',
    'rsi_min': 20.0,  # 매우 다름
    'rsi_max': 80.0,
    'volume_ratio_min': 2.0,
    'volume_ratio_max': 5.0,
    'macd_buy_threshold': 0.05,
    'macd_sell_threshold': -0.05,
    'mfi_min': 10.0,
    'mfi_max': 90.0,
    'atr_min': 0.02,
    'atr_max': 0.1,
    'adx_min': 25.0,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.1,
    'regime': 'trending',
    'strategy_type': 'momentum'
}

# 유사도 계산
print("\nPhase 1: 기본 유사도 (파라미터만)")
sim_basic_duplicate = calculate_basic_similarity(strategy1, strategy2)
sim_basic_copy = calculate_basic_similarity(strategy1, strategy3)
sim_basic_finetune = calculate_basic_similarity(strategy1, strategy4)
sim_basic_novel = calculate_basic_similarity(strategy1, strategy5)

print(f"  test_1 vs test_2 (거의 동일): {sim_basic_duplicate:.4f}")
print(f"  test_1 vs test_3 (매우 유사): {sim_basic_copy:.4f}")
print(f"  test_1 vs test_4 (어느정도 유사): {sim_basic_finetune:.4f}")
print(f"  test_1 vs test_5 (완전히 다름): {sim_basic_novel:.4f}")

print("\nPhase 2: 스마트 유사도 (파라미터 + 레짐 + 타입)")
sim_smart_duplicate = calculate_smart_similarity(strategy1, strategy2)
sim_smart_copy = calculate_smart_similarity(strategy1, strategy3)
sim_smart_finetune = calculate_smart_similarity(strategy1, strategy4)
sim_smart_novel = calculate_smart_similarity(strategy1, strategy5)

print(f"  test_1 vs test_2 (거의 동일): {sim_smart_duplicate:.4f}")
print(f"  test_1 vs test_3 (매우 유사): {sim_smart_copy:.4f}")
print(f"  test_1 vs test_4 (어느정도 유사): {sim_smart_finetune:.4f}")
print(f"  test_1 vs test_5 (완전히 다름): {sim_smart_novel:.4f}")

# 2. 전략 분류 테스트
print("\n2️⃣ 전략 분류 테스트")
print("-" * 80)

existing_strategies = [strategy1]  # test_1을 기존 전략으로 가정

classification2, similarity2, parent2 = classify_strategy_by_similarity(strategy2, existing_strategies, use_smart=True)
classification3, similarity3, parent3 = classify_strategy_by_similarity(strategy3, existing_strategies, use_smart=True)
classification4, similarity4, parent4 = classify_strategy_by_similarity(strategy4, existing_strategies, use_smart=True)
classification5, similarity5, parent5 = classify_strategy_by_similarity(strategy5, existing_strategies, use_smart=True)

print(f"  test_2: {classification2} (유사도: {similarity2:.4f}, 부모: {parent2})")
print(f"  test_3: {classification3} (유사도: {similarity3:.4f}, 부모: {parent3})")
print(f"  test_4: {classification4} (유사도: {similarity4:.4f}, 부모: {parent4})")
print(f"  test_5: {classification5} (유사도: {similarity5:.4f}, 부모: {parent5})")

# 3. Phase 3: 동적 에피소드 조정
print("\n3️⃣ Phase 3: 동적 에피소드 조정")
print("-" * 80)

episodes2 = calculate_finetuning_episodes(similarity2)
episodes3 = calculate_finetuning_episodes(similarity3)
episodes4 = calculate_finetuning_episodes(similarity4)
episodes5 = calculate_finetuning_episodes(similarity5)

print(f"  test_2 ({classification2}, sim={similarity2:.3f}): {episodes2} 에피소드")
print(f"  test_3 ({classification3}, sim={similarity3:.3f}): {episodes3} 에피소드")
print(f"  test_4 ({classification4}, sim={similarity4:.3f}): {episodes4} 에피소드")
print(f"  test_5 ({classification5}, sim={similarity5:.3f}): {episodes5} 에피소드")

# 4. 종합 평가
print("\n" + "=" * 80)
print("종합 평가")
print("=" * 80)

success_count = 0
total_tests = 4

# Test 1: duplicate 분류 확인
if classification2 == 'duplicate':
    print("✅ Test 1: 거의 동일한 전략을 'duplicate'로 분류")
    success_count += 1
else:
    print(f"❌ Test 1: 거의 동일한 전략 분류 실패 (expected: duplicate, got: {classification2})")

# Test 2: copy 분류 확인
if classification3 == 'copy':
    print("✅ Test 2: 매우 유사한 전략을 'copy'로 분류")
    success_count += 1
else:
    print(f"❌ Test 2: 매우 유사한 전략 분류 실패 (expected: copy, got: {classification3})")

# Test 3: finetune 분류 확인
if classification4 == 'finetune':
    print("✅ Test 3: 어느 정도 유사한 전략을 'finetune'로 분류")
    success_count += 1
else:
    print(f"❌ Test 3: 어느 정도 유사한 전략 분류 실패 (expected: finetune, got: {classification4})")

# Test 4: novel 분류 확인
if classification5 == 'novel':
    print("✅ Test 4: 완전히 다른 전략을 'novel'로 분류")
    success_count += 1
else:
    print(f"❌ Test 4: 완전히 다른 전략 분류 실패 (expected: novel, got: {classification5})")

print(f"\n전체 테스트: {success_count}/{total_tests} 통과 ({success_count/total_tests*100:.0f}%)")

if success_count == total_tests:
    print("\n🎉 모든 테스트 통과! Phase 1-3 구현 성공!")
elif success_count >= total_tests * 0.75:
    print("\n⚠️ 대부분의 테스트 통과 (75% 이상)")
else:
    print("\n❌ 테스트 실패 (75% 미만)")

print("=" * 80)
