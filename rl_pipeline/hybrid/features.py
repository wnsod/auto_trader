"""
Features 변환 모듈
MarketState → 정규화된 state_vector 변환
"""

import numpy as np
import logging
from typing import Optional
from rl_pipeline.simulation.market_models import MarketState

logger = logging.getLogger(__name__)

# 피처 버전 및 차원 고정
FEATURES_VERSION = "FEATURES_V4"  # 🚀 메타 학습: 전략 파라미터 포함 버전
FEATURE_DIM = 15  # 기본 차원 (분석 점수 없을 때) - 실제로는 build_state_vector가 20 반환
FEATURE_DIM_WITH_ANALYSIS = 25  # 분석 점수 + 확장 지표 포함 시 차원 (20 → 25)
FEATURE_DIM_WITH_STRATEGY = 30  # 🚀 전략 파라미터 포함 (20 base + 10 strategy params)
FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY = 35  # 🚀 분석 + 전략 파라미터 포함 (25 + 10 strategy params)


def build_state_vector(market_state) -> np.ndarray:
    """
    MarketState 또는 Dict → 정규화된 상태 벡터 변환
    
    Args:
        market_state: 시장 상태 객체 (MarketState) 또는 딕셔너리
    
    Returns:
        np.ndarray: shape (20,) 정규화된 상태 벡터 (확장 지표 포함)
    """
    try:
        # 딕셔너리인 경우 키로 접근, 객체인 경우 속성으로 접근
        def get_value(key: str, default: float = 0.0) -> float:
            if isinstance(market_state, dict):
                return float(market_state.get(key, default))
            else:
                return float(getattr(market_state, key, default))
        
        # 가격 추출 (dict의 경우 price 또는 close 사용)
        price = get_value('price', get_value('close', 50000.0))
        
        # 정규화된 피처 추출
        features = np.array([
            # 0: RSI [0~1]
            get_value('rsi', 50.0) / 100.0,
            
            # 1: MACD (원본, 정규화 필요 시 후처리)
            get_value('macd', 0.0),
            
            # 2: MACD Signal
            get_value('macd_signal', 0.0),
            
            # 3: MACD Histogram
            (get_value('macd', 0.0) - get_value('macd_signal', 0.0)),
            
            # 4: MFI [0~1]
            get_value('mfi', 50.0) / 100.0,
            
            # 5: ADX [0~1]
            get_value('adx', 25.0) / 100.0,
            
            # 6: ATR (원본, 후처리에서 정규화 가능)
            get_value('atr', 0.02),
            
            # 7: BB Position [0~1] - 가격이 볼린저 밴드 내 어디에 있는지
            _calculate_bb_position_dict(market_state, price) if isinstance(market_state, dict) else _calculate_bb_position(market_state),
            
            # 8: BB Width (표준화) - 밴드 폭
            _calculate_bb_width_dict(market_state) if isinstance(market_state, dict) else _calculate_bb_width(market_state),
            
            # 9: Volume Ratio (로그 스케일)
            np.log1p(max(0, get_value('volume_ratio', 1.0))),
            
            # 10: Regime Stage [0~1] - 1-7 단계를 0-1로 정규화
            get_value('regime_stage', 3) / 7.0,
            
            # 11: Regime Confidence [0~1]
            get_value('regime_confidence', 0.5),
            
            # 12: Volatility
            get_value('volatility', 0.02),
            
            # 13: Price Position in BB [0~1]
            _calculate_price_position_dict(market_state, price) if isinstance(market_state, dict) else _calculate_price_position(market_state),
            
            # 14: Volume (정규화)
            np.log1p(max(0, get_value('volume', 1e6) / 1e6)),
            
            # 🚀 확장 지표 (5개) - 1단계 확장
            # 15: Wave Progress [0~1] - 파동 진행률
            np.clip(get_value('wave_progress', 0.5), 0.0, 1.0),
            
            # 16: Pattern Confidence [0~1] - 패턴 신뢰도
            np.clip(get_value('pattern_confidence', 0.5), 0.0, 1.0),
            
            # 17: Structure Score [0~1] - 구조 점수
            np.clip(get_value('structure_score', 0.5), 0.0, 1.0),
            
            # 18: Sentiment [-1~1] → [0~1] - 심리도 점수 정규화
            np.clip((get_value('sentiment', 0.0) + 1.0) / 2.0, 0.0, 1.0),
            
            # 19: Regime Transition Prob [0~0.4] → [0~1] - 레짐 전환 확률 정규화
            np.clip(get_value('regime_transition_prob', 0.05) / 0.4, 0.0, 1.0),
        ], dtype=np.float32)
        
        # NaN/Inf 체크 및 처리
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 값 범위 클리핑 (안전장치)
        features = np.clip(features, -10.0, 10.0)
        
        return features
        
    except Exception as e:
        logger.error(f"❌ 상태 벡터 변환 실패: {e}")
        # 에러 시 기본값 반환 (중립 상태) - 20차원 (확장 지표 포함)
        return np.zeros(20, dtype=np.float32)


def _calculate_bb_position_dict(market_state: dict, price: float) -> float:
    """딕셔너리 버전 BB Position 계산"""
    try:
        bb_upper = float(market_state.get('bb_upper', price * 1.02))
        bb_lower = float(market_state.get('bb_lower', price * 0.98))
        
        if bb_upper == bb_lower:
            return 0.5
        
        if bb_upper <= bb_lower:
            return 0.5
        
        position = (price - bb_lower) / (bb_upper - bb_lower)
        return np.clip(position, 0.0, 1.0)
    except:
        return 0.5


def _calculate_bb_width_dict(market_state: dict) -> float:
    """딕셔너리 버전 BB Width 계산"""
    try:
        bb_upper = float(market_state.get('bb_upper', 1.02))
        bb_lower = float(market_state.get('bb_lower', 0.98))
        bb_middle = float(market_state.get('bb_middle', 1.0))
        
        if bb_middle == 0:
            return 0.0
        
        width_pct = (bb_upper - bb_lower) / bb_middle
        normalized = np.clip(width_pct / 0.2, 0.0, 1.0)
        return float(normalized)
    except:
        return 0.5


def _calculate_price_position_dict(market_state: dict, price: float) -> float:
    """딕셔너리 버전 Price Position 계산"""
    try:
        bb_upper = float(market_state.get('bb_upper', price * 1.02))
        bb_lower = float(market_state.get('bb_lower', price * 0.98))
        
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return 0.5
        
        position = (price - bb_lower) / bb_range
        return np.clip(position, 0.0, 1.0)
    except:
        return 0.5


def _calculate_bb_position(market_state: MarketState) -> float:
    """
    볼린저 밴드 내 가격 위치 [0~1]
    
    0: bb_lower (하단)
    0.5: bb_middle (중간)
    1: bb_upper (상단)
    """
    try:
        if market_state.bb_upper == market_state.bb_lower:
            return 0.5
        
        if market_state.bb_upper <= market_state.bb_lower:
            return 0.5
        
        position = (market_state.price - market_state.bb_lower) / (market_state.bb_upper - market_state.bb_lower)
        return np.clip(position, 0.0, 1.0)
    except:
        return 0.5


def _calculate_bb_width(market_state: MarketState) -> float:
    """
    볼린저 밴드 폭 (표준화) [0~1]
    
    밴드 폭이 클수록 변동성이 큼
    """
    try:
        if market_state.bb_middle == 0:
            return 0.0
        
        width_pct = (market_state.bb_upper - market_state.bb_lower) / market_state.bb_middle
        # 일반적으로 0~0.2 범위, 이를 0~1로 정규화
        normalized = np.clip(width_pct / 0.2, 0.0, 1.0)
        return float(normalized)
    except:
        return 0.5


def _calculate_price_position(market_state: MarketState) -> float:
    """
    가격 위치 계산 [0~1]
    
    BB 상단/하단 기준 가격 위치
    """
    try:
        bb_range = market_state.bb_upper - market_state.bb_lower
        if bb_range <= 0:
            return 0.5
        
        position = (market_state.price - market_state.bb_lower) / bb_range
        return np.clip(position, 0.0, 1.0)
    except:
        return 0.5


def build_state_vector_with_analysis(
    market_state,
    fractal_score: float = 0.5,
    multi_timeframe_score: float = 0.5,
    indicator_cross_score: float = 0.5,
    ensemble_score: float = 0.5,
    ensemble_confidence: float = 0.5
) -> np.ndarray:
    """
    MarketState + 분석 점수 → 정규화된 상태 벡터 변환 (25차원)
    
    Args:
        market_state: 시장 상태 객체 (MarketState) 또는 딕셔너리
        fractal_score: 프랙탈 분석 점수 [0~1]
        multi_timeframe_score: 다중 타임프레임 분석 점수 [0~1]
        indicator_cross_score: 지표 교차 분석 점수 [0~1]
        ensemble_score: 앙상블 점수 [0~1]
        ensemble_confidence: 앙상블 신뢰도 [0~1]
    
    Returns:
        np.ndarray: shape (25,) 정규화된 상태 벡터
    """
    try:
        # 기존 20차원 피처 (15개 기본 + 5개 확장 지표)
        base_features = build_state_vector(market_state)
        
        # 추가 분석 피처 (5차원)
        analysis_features = np.array([
            float(fractal_score),           # 20: 프랙탈 점수 [0~1]
            float(multi_timeframe_score),   # 21: 다중 타임프레임 점수 [0~1]
            float(indicator_cross_score),   # 22: 지표 교차 점수 [0~1]
            float(ensemble_score),          # 23: 앙상블 점수 [0~1]
            float(ensemble_confidence),    # 24: 앙상블 신뢰도 [0~1]
        ], dtype=np.float32)
        
        # 결합
        enhanced_features = np.concatenate([base_features, analysis_features])
        
        # NaN/Inf 체크 및 클리핑
        enhanced_features = np.nan_to_num(enhanced_features, nan=0.5, posinf=1.0, neginf=0.0)
        enhanced_features = np.clip(enhanced_features, -10.0, 10.0)
        
        return enhanced_features
        
    except Exception as e:
        logger.error(f"❌ 분석 점수 포함 상태 벡터 변환 실패: {e}")
        # 에러 시 기본값 반환 (25차원)
        base_features = build_state_vector(market_state)
        analysis_features = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        return np.concatenate([base_features, analysis_features])


def get_feature_names() -> list:
    """피처 이름 리스트 반환 (디버깅/로깅용) - 20차원 (15개 기본 + 5개 확장)"""
    return [
        'rsi_norm',              # 0
        'macd',                  # 1
        'macd_signal',           # 2
        'macd_hist',             # 3
        'mfi_norm',              # 4
        'adx_norm',              # 5
        'atr',                   # 6
        'bb_position',           # 7
        'bb_width',              # 8
        'volume_ratio_log',      # 9
        'regime_stage_norm',     # 10
        'regime_confidence',    # 11
        'volatility',            # 12
        'price_position',        # 13
        'volume_log',            # 14
        'wave_progress',         # 15 🚀 확장 지표
        'pattern_confidence',    # 16 🚀 확장 지표
        'structure_score',       # 17 🚀 확장 지표
        'sentiment_norm',        # 18 🚀 확장 지표
        'regime_transition_prob', # 19 🚀 확장 지표
    ]


def get_feature_names_with_analysis() -> list:
    """분석 점수 포함 피처 이름 리스트 반환 (25차원)"""
    base_names = get_feature_names()
    analysis_names = [
        'fractal_score',          # 20
        'multi_timeframe_score',  # 21
        'indicator_cross_score',  # 22
        'ensemble_score',         # 23
        'ensemble_confidence',    # 24
    ]
    return base_names + analysis_names


def _normalize_strategy_params(strategy_params: dict) -> np.ndarray:
    """
    전략 파라미터를 정규화하여 10차원 벡터로 변환

    Args:
        strategy_params: 전략 파라미터 딕셔너리

    Returns:
        np.ndarray: shape (10,) 정규화된 전략 파라미터 벡터
    """
    try:
        # 기본값 설정 (전략 파라미터가 없을 경우)
        default_params = {
            'rsi_min': 30.0,
            'rsi_max': 70.0,
            'macd_buy_threshold': 0.0,
            'volume_ratio_min': 1.5,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.05,
            'position_size': 0.5,
            'trend_strength_min': 0.3,
            'confirmation_threshold': 0.5,
            'signal_threshold': 0.5,
        }

        # 실제 값 또는 기본값 사용
        params = {**default_params, **strategy_params}

        # 정규화된 파라미터 배열 생성
        normalized = np.array([
            # 0: RSI Min [20~80] → [0~1]
            (float(params['rsi_min']) - 20.0) / 60.0,

            # 1: RSI Max [20~80] → [0~1]
            (float(params['rsi_max']) - 20.0) / 60.0,

            # 2: MACD Buy Threshold [-100~100] → [0~1]
            (float(params['macd_buy_threshold']) + 100.0) / 200.0,

            # 3: Volume Ratio Min [1.0~5.0] → [0~1]
            (float(params['volume_ratio_min']) - 1.0) / 4.0,

            # 4: Stop Loss % [0.01~0.10] → [0~1]
            (float(params['stop_loss_pct']) - 0.01) / 0.09,

            # 5: Take Profit % [0.01~0.20] → [0~1]
            (float(params['take_profit_pct']) - 0.01) / 0.19,

            # 6: Position Size [0.1~1.0] → [0~1]
            (float(params['position_size']) - 0.1) / 0.9,

            # 7: Trend Strength Min [0.0~1.0] - 이미 정규화됨
            float(params['trend_strength_min']),

            # 8: Confirmation Threshold [0.0~1.0] - 이미 정규화됨
            float(params['confirmation_threshold']),

            # 9: Signal Threshold [0.0~1.0] - 이미 정규화됨
            float(params['signal_threshold']),
        ], dtype=np.float32)

        # NaN/Inf 체크 및 클리핑
        normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    except Exception as e:
        logger.error(f"❌ 전략 파라미터 정규화 실패: {e}")
        # 에러 시 중립 값 반환
        return np.full(10, 0.5, dtype=np.float32)


def build_state_vector_with_strategy(
    market_state,
    strategy_params: dict
) -> np.ndarray:
    """
    🚀 메타 학습: MarketState + 전략 파라미터 → 정규화된 상태 벡터 변환 (30차원)

    동일한 전략 파라미터라도 시장 상황에 따라 다르게 활용할 수 있도록
    전략 파라미터를 State 벡터에 포함합니다.

    Args:
        market_state: 시장 상태 객체 (MarketState) 또는 딕셔너리
        strategy_params: 전략 파라미터 딕셔너리

    Returns:
        np.ndarray: shape (30,) 정규화된 상태 벡터 (20 base + 10 strategy params)
    """
    try:
        # 기존 20차원 피처 (15개 기본 + 5개 확장 지표)
        base_features = build_state_vector(market_state)

        # 전략 파라미터 10차원
        strategy_features = _normalize_strategy_params(strategy_params)

        # 결합
        enhanced_features = np.concatenate([base_features, strategy_features])

        # NaN/Inf 체크 및 클리핑
        enhanced_features = np.nan_to_num(enhanced_features, nan=0.5, posinf=1.0, neginf=0.0)
        enhanced_features = np.clip(enhanced_features, -10.0, 10.0)

        return enhanced_features

    except Exception as e:
        logger.error(f"❌ 전략 파라미터 포함 상태 벡터 변환 실패: {e}")
        # 에러 시 기본값 반환 (30차원)
        base_features = build_state_vector(market_state)
        strategy_features = np.full(10, 0.5, dtype=np.float32)
        return np.concatenate([base_features, strategy_features])


def build_state_vector_with_analysis_and_strategy(
    market_state,
    strategy_params: dict,
    fractal_score: float = 0.5,
    multi_timeframe_score: float = 0.5,
    indicator_cross_score: float = 0.5,
    ensemble_score: float = 0.5,
    ensemble_confidence: float = 0.5
) -> np.ndarray:
    """
    🚀 메타 학습: MarketState + 분석 점수 + 전략 파라미터 → 정규화된 상태 벡터 변환 (35차원)

    Args:
        market_state: 시장 상태 객체 (MarketState) 또는 딕셔너리
        strategy_params: 전략 파라미터 딕셔너리
        fractal_score: 프랙탈 분석 점수 [0~1]
        multi_timeframe_score: 다중 타임프레임 분석 점수 [0~1]
        indicator_cross_score: 지표 교차 분석 점수 [0~1]
        ensemble_score: 앙상블 점수 [0~1]
        ensemble_confidence: 앙상블 신뢰도 [0~1]

    Returns:
        np.ndarray: shape (35,) 정규화된 상태 벡터 (25 with analysis + 10 strategy params)
    """
    try:
        # 기존 25차원 피처 (20개 기본 + 5개 분석)
        base_features = build_state_vector_with_analysis(
            market_state,
            fractal_score,
            multi_timeframe_score,
            indicator_cross_score,
            ensemble_score,
            ensemble_confidence
        )

        # 전략 파라미터 10차원
        strategy_features = _normalize_strategy_params(strategy_params)

        # 결합
        enhanced_features = np.concatenate([base_features, strategy_features])

        # NaN/Inf 체크 및 클리핑
        enhanced_features = np.nan_to_num(enhanced_features, nan=0.5, posinf=1.0, neginf=0.0)
        enhanced_features = np.clip(enhanced_features, -10.0, 10.0)

        return enhanced_features

    except Exception as e:
        logger.error(f"❌ 분석+전략 파라미터 포함 상태 벡터 변환 실패: {e}")
        # 에러 시 기본값 반환 (35차원)
        base_features = build_state_vector_with_analysis(
            market_state, fractal_score, multi_timeframe_score,
            indicator_cross_score, ensemble_score, ensemble_confidence
        )
        strategy_features = np.full(10, 0.5, dtype=np.float32)
        return np.concatenate([base_features, strategy_features])


def get_feature_names_with_strategy() -> list:
    """🚀 전략 파라미터 포함 피처 이름 리스트 반환 (30차원)"""
    base_names = get_feature_names()
    strategy_names = [
        'strategy_rsi_min',              # 20
        'strategy_rsi_max',              # 21
        'strategy_macd_threshold',       # 22
        'strategy_volume_ratio_min',    # 23
        'strategy_stop_loss_pct',        # 24
        'strategy_take_profit_pct',      # 25
        'strategy_position_size',        # 26
        'strategy_trend_strength_min',  # 27
        'strategy_confirmation_threshold', # 28
        'strategy_signal_threshold',     # 29
    ]
    return base_names + strategy_names


def get_feature_names_with_analysis_and_strategy() -> list:
    """🚀 분석 + 전략 파라미터 포함 피처 이름 리스트 반환 (35차원)"""
    base_names = get_feature_names_with_analysis()  # 25차원
    strategy_names = [
        'strategy_rsi_min',              # 25
        'strategy_rsi_max',              # 26
        'strategy_macd_threshold',       # 27
        'strategy_volume_ratio_min',    # 28
        'strategy_stop_loss_pct',        # 29
        'strategy_take_profit_pct',      # 30
        'strategy_position_size',        # 31
        'strategy_trend_strength_min',  # 32
        'strategy_confirmation_threshold', # 33
        'strategy_signal_threshold',     # 34
    ]
    return base_names + strategy_names

