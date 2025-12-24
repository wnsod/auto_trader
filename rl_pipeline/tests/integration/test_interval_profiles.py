#!/usr/bin/env python3
"""
인터벌 프로필 테스트 스크립트
"""

import sys
import pandas as pd
import numpy as np
from core.interval_profiles import (
    INTERVAL_PROFILES,
    generate_labels,
    calculate_reward,
    get_integration_weights,
    get_interval_role
)

def test_interval_profiles():
    """인터벌 프로필 기능 테스트"""

    print("=" * 60)
    print("인터벌 프로필 테스트 시작")
    print("=" * 60)

    # 1. 프로필 확인
    print("\n1. 정의된 인터벌 프로필:")
    for interval, profile in INTERVAL_PROFILES.items():
        print(f"\n  {interval}:")
        print(f"    - 역할: {profile['role']}")
        print(f"    - 설명: {profile['description']}")
        print(f"    - 라벨 타입: {profile['labeling']['label_type']}")
        print(f"    - 예측 기간: {profile['labeling']['target_horizon']} 캔들")
        print(f"    - 통합 가중치: {profile['integration_weight']:.2f}")

    # 2. 가중치 확인
    print("\n2. 통합 분석 가중치:")
    weights = get_integration_weights()
    total_weight = sum(weights.values())
    for interval, weight in weights.items():
        role = get_interval_role(interval)
        print(f"  {interval}: {weight:.2f} ({weight/total_weight*100:.0f}%)")
        print(f"    -> {role}")

    # 3. 라벨 생성 테스트
    print("\n3. 라벨 생성 테스트:")

    # 가짜 데이터 생성
    n_candles = 100
    dates = pd.date_range(start='2024-01-01', periods=n_candles, freq='15min')

    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(n_candles).cumsum() + 100,
        'high': np.random.randn(n_candles).cumsum() + 101,
        'low': np.random.randn(n_candles).cumsum() + 99,
        'close': np.random.randn(n_candles).cumsum() + 100,
        'volume': np.random.rand(n_candles) * 1000000
    })

    for interval in ['15m', '30m', '240m', '1d']:
        try:
            labeled_df = generate_labels(test_df.copy(), interval)
            label_counts = labeled_df['label'].value_counts()
            print(f"\n  {interval} 라벨 생성 결과:")
            print(f"    - 라벨 타입: {labeled_df['label_type'].iloc[0]}")
            print(f"    - 타겟 기간: {labeled_df['target_horizon'].iloc[0]} 캔들")
            print(f"    - 라벨 분포:")
            for label, count in label_counts.items():
                print(f"      {label}: {count}개 ({count/len(labeled_df)*100:.1f}%)")
        except Exception as e:
            print(f"  {interval} 라벨 생성 실패: {e}")

    # 4. 보상 계산 테스트
    print("\n4. 보상 계산 테스트:")

    # 테스트 예측과 실제 결과
    test_prediction = {
        'direction': 1,
        'return': 0.02,
        'regime': 'bull',
        'swing': 'up',
        'trend': 'continuation',
        'entry_quality': 'good',
        'r_multiple': 2.0
    }

    test_actual = {
        'direction': 1,
        'return': 0.025,
        'regime': 'bull',
        'swing': 'strong_up',
        'trend': 'continuation',
        'entry_quality': 'excellent',
        'r_multiple': 2.5,
        'stop_hit': False
    }

    for interval in ['15m', '30m', '240m', '1d']:
        try:
            reward = calculate_reward(interval, test_prediction, test_actual)
            print(f"  {interval} 보상: {reward:.3f}")
        except Exception as e:
            print(f"  {interval} 보상 계산 실패: {e}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_interval_profiles()