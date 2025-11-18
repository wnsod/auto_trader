"""
메타 학습 구현 테스트
- State 벡터에 전략 파라미터가 제대로 포함되는지 검증
- 차원이 올바른지 확인 (30차원 or 35차원)
"""

import sys
import os

# JAX CPU 모드 강제 설정 (GPU 초기화 방지)
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from rl_pipeline.hybrid.features import (
    build_state_vector,
    build_state_vector_with_strategy,
    build_state_vector_with_analysis,
    build_state_vector_with_analysis_and_strategy,
    get_feature_names,
    get_feature_names_with_strategy,
    get_feature_names_with_analysis,
    get_feature_names_with_analysis_and_strategy,
    FEATURE_DIM,
    FEATURE_DIM_WITH_ANALYSIS,
    FEATURE_DIM_WITH_STRATEGY,
    FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY
)

def test_feature_dimensions():
    """피처 차원 상수 확인"""
    print("=" * 80)
    print("🧪 테스트 1: 피처 차원 상수 확인")
    print("=" * 80)

    print(f"✅ FEATURE_DIM (기본): {FEATURE_DIM}")
    print(f"✅ FEATURE_DIM_WITH_ANALYSIS: {FEATURE_DIM_WITH_ANALYSIS}")
    print(f"🚀 FEATURE_DIM_WITH_STRATEGY: {FEATURE_DIM_WITH_STRATEGY}")
    print(f"🚀 FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY: {FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY}")

    assert FEATURE_DIM_WITH_STRATEGY == 30, f"Expected 30, got {FEATURE_DIM_WITH_STRATEGY}"
    assert FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY == 35, f"Expected 35, got {FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY}"
    print("\n✅ 차원 상수 검증 통과!\n")


def test_basic_state_vector():
    """기본 State 벡터 생성 테스트"""
    print("=" * 80)
    print("🧪 테스트 2: 기본 State 벡터 생성 (20차원)")
    print("=" * 80)

    market_state = {
        'rsi': 65.0,
        'macd': 0.5,
        'macd_signal': 0.3,
        'mfi': 60.0,
        'adx': 30.0,
        'atr': 0.03,
        'bb_upper': 52000.0,
        'bb_middle': 50000.0,
        'bb_lower': 48000.0,
        'close': 50500.0,
        'volume': 1500000.0,
        'volume_ratio': 1.2,
        'volatility': 0.025,
        'regime_stage': 4,
        'regime_confidence': 0.7,
        'wave_progress': 0.6,
        'pattern_confidence': 0.75,
        'structure_score': 0.8,
        'sentiment': 0.2,
        'regime_transition_prob': 0.1
    }

    state_vec = build_state_vector(market_state)
    print(f"생성된 State 벡터 차원: {state_vec.shape}")
    print(f"State 벡터 샘플 (처음 10개): {state_vec[:10]}")

    assert state_vec.shape == (20,), f"Expected shape (20,), got {state_vec.shape}"
    assert not np.any(np.isnan(state_vec)), "State 벡터에 NaN 발견!"
    assert not np.any(np.isinf(state_vec)), "State 벡터에 Inf 발견!"
    print("\n✅ 기본 State 벡터 생성 성공!\n")

    return market_state, state_vec


def test_state_vector_with_strategy():
    """전략 파라미터 포함 State 벡터 테스트 (30차원)"""
    print("=" * 80)
    print("🧪 테스트 3: 전략 파라미터 포함 State 벡터 (30차원)")
    print("=" * 80)

    market_state = {
        'rsi': 65.0,
        'macd': 0.5,
        'close': 50000.0,
        'volume_ratio': 1.2,
    }

    strategy_params = {
        'rsi_min': 30.0,
        'rsi_max': 70.0,
        'macd_buy_threshold': 0.01,
        'volume_ratio_min': 1.5,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.05,
        'position_size': 0.5,
        'trend_strength_min': 0.3,
        'confirmation_threshold': 0.6,
        'signal_threshold': 0.7,
    }

    state_vec = build_state_vector_with_strategy(market_state, strategy_params)
    print(f"생성된 State 벡터 차원: {state_vec.shape}")
    print(f"State 벡터 샘플 (처음 10개): {state_vec[:10]}")
    print(f"전략 파라미터 부분 (마지막 10개): {state_vec[-10:]}")

    assert state_vec.shape == (30,), f"Expected shape (30,), got {state_vec.shape}"
    assert not np.any(np.isnan(state_vec)), "State 벡터에 NaN 발견!"
    assert not np.any(np.isinf(state_vec)), "State 벡터에 Inf 발견!"

    # 전략 파라미터가 0-1 범위로 정규화되었는지 확인
    strategy_part = state_vec[-10:]
    assert np.all(strategy_part >= 0.0) and np.all(strategy_part <= 1.0), \
        f"전략 파라미터가 [0,1] 범위를 벗어남: min={strategy_part.min()}, max={strategy_part.max()}"

    print("\n✅ 전략 파라미터 포함 State 벡터 생성 성공!")
    print(f"   전략 파라미터 정규화 범위: [{strategy_part.min():.3f}, {strategy_part.max():.3f}]\n")

    return state_vec


def test_state_vector_with_analysis_and_strategy():
    """분석 점수 + 전략 파라미터 포함 State 벡터 테스트 (35차원)"""
    print("=" * 80)
    print("🧪 테스트 4: 분석 + 전략 파라미터 포함 State 벡터 (35차원)")
    print("=" * 80)

    market_state = {
        'rsi': 55.0,
        'macd': 0.3,
        'close': 48000.0,
        'volume_ratio': 1.8,
    }

    strategy_params = {
        'rsi_min': 25.0,
        'rsi_max': 75.0,
        'macd_buy_threshold': -0.05,
        'volume_ratio_min': 2.0,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.08,
        'position_size': 0.8,
        'trend_strength_min': 0.4,
        'confirmation_threshold': 0.5,
        'signal_threshold': 0.6,
    }

    # 분석 점수
    fractal_score = 0.7
    multi_timeframe_score = 0.6
    indicator_cross_score = 0.8
    ensemble_score = 0.65
    ensemble_confidence = 0.75

    state_vec = build_state_vector_with_analysis_and_strategy(
        market_state,
        strategy_params,
        fractal_score=fractal_score,
        multi_timeframe_score=multi_timeframe_score,
        indicator_cross_score=indicator_cross_score,
        ensemble_score=ensemble_score,
        ensemble_confidence=ensemble_confidence
    )

    print(f"생성된 State 벡터 차원: {state_vec.shape}")
    print(f"State 벡터 샘플 (처음 10개): {state_vec[:10]}")
    print(f"분석 점수 부분 (20-24): {state_vec[20:25]}")
    print(f"전략 파라미터 부분 (25-34): {state_vec[25:35]}")

    assert state_vec.shape == (35,), f"Expected shape (35,), got {state_vec.shape}"
    assert not np.any(np.isnan(state_vec)), "State 벡터에 NaN 발견!"
    assert not np.any(np.isinf(state_vec)), "State 벡터에 Inf 발견!"

    # 분석 점수 확인 (20-24번 인덱스)
    analysis_part = state_vec[20:25]
    print(f"\n분석 점수 검증:")
    print(f"  프랙탈: {analysis_part[0]:.3f} (기대값: {fractal_score:.3f})")
    print(f"  멀티TF: {analysis_part[1]:.3f} (기대값: {multi_timeframe_score:.3f})")
    print(f"  지표교차: {analysis_part[2]:.3f} (기대값: {indicator_cross_score:.3f})")
    print(f"  앙상블: {analysis_part[3]:.3f} (기대값: {ensemble_score:.3f})")
    print(f"  앙상블 신뢰도: {analysis_part[4]:.3f} (기대값: {ensemble_confidence:.3f})")

    # 전략 파라미터 확인 (25-34번 인덱스)
    strategy_part = state_vec[25:35]
    assert np.all(strategy_part >= 0.0) and np.all(strategy_part <= 1.0), \
        f"전략 파라미터가 [0,1] 범위를 벗어남: min={strategy_part.min()}, max={strategy_part.max()}"

    print(f"\n✅ 분석 + 전략 파라미터 포함 State 벡터 생성 성공!")
    print(f"   전략 파라미터 정규화 범위: [{strategy_part.min():.3f}, {strategy_part.max():.3f}]\n")

    return state_vec


def test_feature_names():
    """피처 이름 확인 테스트"""
    print("=" * 80)
    print("🧪 테스트 5: 피처 이름 확인")
    print("=" * 80)

    # 기본 피처 이름 (20개)
    base_names = get_feature_names()
    print(f"\n기본 피처 이름 (20개):")
    for i, name in enumerate(base_names):
        print(f"  [{i:2d}] {name}")
    assert len(base_names) == 20, f"Expected 20 feature names, got {len(base_names)}"

    # 분석 포함 피처 이름 (25개)
    analysis_names = get_feature_names_with_analysis()
    print(f"\n분석 포함 피처 이름 (25개):")
    for i in range(20, 25):
        print(f"  [{i:2d}] {analysis_names[i]}")
    assert len(analysis_names) == 25, f"Expected 25 feature names, got {len(analysis_names)}"

    # 전략 포함 피처 이름 (30개)
    strategy_names = get_feature_names_with_strategy()
    print(f"\n🚀 전략 파라미터 포함 피처 이름 (30개):")
    for i in range(20, 30):
        print(f"  [{i:2d}] {strategy_names[i]}")
    assert len(strategy_names) == 30, f"Expected 30 feature names, got {len(strategy_names)}"

    # 분석+전략 포함 피처 이름 (35개)
    full_names = get_feature_names_with_analysis_and_strategy()
    print(f"\n🚀 분석 + 전략 파라미터 포함 피처 이름 (35개):")
    print(f"  분석 점수 (20-24):")
    for i in range(20, 25):
        print(f"    [{i:2d}] {full_names[i]}")
    print(f"  전략 파라미터 (25-34):")
    for i in range(25, 35):
        print(f"    [{i:2d}] {full_names[i]}")
    assert len(full_names) == 35, f"Expected 35 feature names, got {len(full_names)}"

    print("\n✅ 모든 피처 이름 검증 통과!\n")


def test_meta_learning_concept():
    """메타 학습 개념 테스트: 같은 전략 파라미터를 다른 시장 상황에서 사용"""
    print("=" * 80)
    print("🧪 테스트 6: 메타 학습 개념 검증")
    print("=" * 80)

    # 동일한 전략 파라미터
    strategy_params = {
        'rsi_min': 30.0,
        'rsi_max': 70.0,
        'macd_buy_threshold': 0.01,
        'volume_ratio_min': 1.5,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.05,
        'position_size': 0.5,
        'trend_strength_min': 0.3,
        'confirmation_threshold': 0.6,
        'signal_threshold': 0.7,
    }

    # 시장 상황 1: 강세장 (과매수)
    market_state_bullish = {
        'rsi': 75.0,  # 과매수
        'macd': 1.2,  # 강한 상승
        'close': 52000.0,
        'volume_ratio': 2.5,  # 높은 거래량
        'adx': 35.0,  # 강한 추세
    }

    # 시장 상황 2: 약세장 (과매도)
    market_state_bearish = {
        'rsi': 25.0,  # 과매도
        'macd': -0.8,  # 강한 하락
        'close': 48000.0,
        'volume_ratio': 2.2,  # 높은 거래량
        'adx': 32.0,  # 강한 추세
    }

    # 같은 전략 파라미터로 다른 State 벡터 생성
    state_vec_bullish = build_state_vector_with_strategy(market_state_bullish, strategy_params)
    state_vec_bearish = build_state_vector_with_strategy(market_state_bearish, strategy_params)

    print(f"\n🚀 메타 학습 핵심 개념:")
    print(f"   - 동일한 전략 파라미터를 다른 시장 상황에 적용")
    print(f"   - RL 에이전트가 '언제' 이 전략을 BUY/SELL/HOLD로 사용할지 학습\n")

    print(f"전략 파라미터 (동일):")
    print(f"  RSI 범위: [{strategy_params['rsi_min']}, {strategy_params['rsi_max']}]")
    print(f"  MACD 임계값: {strategy_params['macd_buy_threshold']}")
    print(f"  손절/익절: {strategy_params['stop_loss_pct']:.1%} / {strategy_params['take_profit_pct']:.1%}\n")

    print(f"시장 상황 1 (강세장):")
    print(f"  RSI: {market_state_bullish['rsi']:.1f} (과매수)")
    print(f"  MACD: {market_state_bullish['macd']:.2f} (강한 상승)")
    print(f"  State 벡터 차원: {state_vec_bullish.shape}")
    print(f"  전략 파라미터 부분: {state_vec_bullish[-10:]}\n")

    print(f"시장 상황 2 (약세장):")
    print(f"  RSI: {market_state_bearish['rsi']:.1f} (과매도)")
    print(f"  MACD: {market_state_bearish['macd']:.2f} (강한 하락)")
    print(f"  State 벡터 차원: {state_vec_bearish.shape}")
    print(f"  전략 파라미터 부분: {state_vec_bearish[-10:]}\n")

    # State 벡터 차이 확인
    market_part_diff = np.abs(state_vec_bullish[:20] - state_vec_bearish[:20]).mean()
    strategy_part_diff = np.abs(state_vec_bullish[-10:] - state_vec_bearish[-10:]).mean()

    print(f"State 벡터 비교:")
    print(f"  시장 상태 부분 (0-19) 평균 차이: {market_part_diff:.4f} (다름 ✓)")
    print(f"  전략 파라미터 부분 (20-29) 평균 차이: {strategy_part_diff:.4f} (동일 ✓)")

    assert market_part_diff > 0.1, "시장 상태가 충분히 다르지 않음"
    assert strategy_part_diff < 0.001, "전략 파라미터가 동일하지 않음"

    print(f"\n✅ 메타 학습 개념 검증 성공!")
    print(f"   → RL 에이전트는 동일 전략을 강세장에서는 SELL, 약세장에서는 BUY로 사용 가능!\n")


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 80)
    print("🚀 메타 학습 구현 테스트 시작")
    print("=" * 80 + "\n")

    try:
        # 1. 피처 차원 확인
        test_feature_dimensions()

        # 2. 기본 State 벡터 생성
        market_state, base_state_vec = test_basic_state_vector()

        # 3. 전략 파라미터 포함 State 벡터
        strategy_state_vec = test_state_vector_with_strategy()

        # 4. 분석 + 전략 파라미터 포함 State 벡터
        full_state_vec = test_state_vector_with_analysis_and_strategy()

        # 5. 피처 이름 확인
        test_feature_names()

        # 6. 메타 학습 개념 검증
        test_meta_learning_concept()

        # 최종 결과
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        print(f"\n메타 학습 구현 요약:")
        print(f"  📊 기본 State 벡터: 20차원 (시장 지표)")
        print(f"  🚀 전략 파라미터 추가: +10차원 → 30차원")
        print(f"  📈 분석 점수 추가: +5차원 → 35차원")
        print(f"\n🎯 메타 학습 목표:")
        print(f"  - 동일 전략 파라미터를 상황에 따라 BUY/SELL/HOLD로 적응적 활용")
        print(f"  - RL 에이전트가 '언제' 어떤 전략을 사용할지 자동 학습")
        print("\n✅ 구현 완료! 이제 실제 학습을 진행할 수 있습니다.\n")

        return True

    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
