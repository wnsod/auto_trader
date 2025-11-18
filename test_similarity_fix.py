"""
전략 벡터화 수정 테스트
"""

import sys
sys.path.append('/workspace')

from rl_pipeline.strategy.similarity import vectorize_strategy_params, calculate_smart_similarity
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

# Strategy 클래스 모의 (실제 Strategy 클래스와 유사)
@dataclass
class Strategy:
    id: str = "test_001"
    params: Dict[str, Any] = None

    # 속성으로도 접근 가능
    rsi_min: float = 30.0
    rsi_max: float = 70.0
    volume_ratio_min: float = 1.0
    volume_ratio_max: float = 2.0
    macd_buy_threshold: float = 0.01
    macd_sell_threshold: float = -0.01
    mfi_min: float = 20.0
    mfi_max: float = 80.0
    atr_min: float = 0.01
    atr_max: float = 0.05
    adx_min: float = 15.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    regime: str = "ranging"
    strategy_type: str = "hybrid"

    def __post_init__(self):
        if self.params is None:
            self.params = {}

# 테스트 1: 딕셔너리 객체
print("=" * 60)
print("테스트 1: 딕셔너리 전략")
print("=" * 60)
dict_strategy = {
    'id': 'test_001',
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

try:
    vec = vectorize_strategy_params(dict_strategy)
    print(f"✅ 딕셔너리 벡터화 성공!")
    print(f"   벡터 크기: {vec.shape}")
    print(f"   벡터: {vec}")
except Exception as e:
    print(f"❌ 딕셔너리 벡터화 실패: {e}")

# 테스트 2: Strategy 객체
print("\n" + "=" * 60)
print("테스트 2: Strategy 객체")
print("=" * 60)
strategy_obj = Strategy()

try:
    vec = vectorize_strategy_params(strategy_obj)
    print(f"✅ Strategy 객체 벡터화 성공!")
    print(f"   벡터 크기: {vec.shape}")
    print(f"   벡터: {vec}")
except Exception as e:
    print(f"❌ Strategy 객체 벡터화 실패: {e}")

# 테스트 3: params에 값이 있는 Strategy 객체
print("\n" + "=" * 60)
print("테스트 3: params 딕셔너리가 있는 Strategy 객체")
print("=" * 60)
strategy_with_params = Strategy()
strategy_with_params.params = {
    'rsi_min': 25.0,
    'rsi_max': 75.0
}

try:
    vec = vectorize_strategy_params(strategy_with_params)
    print(f"✅ params 있는 Strategy 객체 벡터화 성공!")
    print(f"   벡터 크기: {vec.shape}")
    print(f"   벡터: {vec}")
    print(f"   (params의 rsi_min=25.0이 우선 적용되어야 함)")
except Exception as e:
    print(f"❌ params 있는 Strategy 객체 벡터화 실패: {e}")

# 테스트 4: 스마트 유사도 계산
print("\n" + "=" * 60)
print("테스트 4: 스마트 유사도 계산")
print("=" * 60)

strategy1 = Strategy()
strategy2 = Strategy()
strategy2.rsi_min = 35.0

try:
    sim = calculate_smart_similarity(strategy1, strategy2)
    print(f"✅ 스마트 유사도 계산 성공!")
    print(f"   유사도: {sim:.4f}")
except Exception as e:
    print(f"❌ 스마트 유사도 계산 실패: {e}")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
