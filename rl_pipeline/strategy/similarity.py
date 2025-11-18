"""
전략 유사도 계산 및 증분 학습 관련 유틸리티

Phase 1: 기본 파라미터 기반 유사도
Phase 2: 레짐/타입 포함 정교한 유사도
Phase 3: 동적 에피소드 조정
"""

import numpy as np
import hashlib
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: 기본 파라미터 기반 유사도
# =============================================================================

def _get_value(obj, key: str, default: Any):
    """객체나 딕셔너리에서 값을 가져오는 헬퍼 함수"""
    if isinstance(obj, dict):
        # 먼저 최상위 레벨 확인
        if key in obj:
            return obj[key]
        # 최상위에 없으면 params 서브 딕셔너리 확인
        if 'params' in obj and isinstance(obj['params'], dict):
            if key in obj['params']:
                return obj['params'][key]
        return default
    else:
        # Strategy 객체인 경우
        # 먼저 params 딕셔너리 확인
        if hasattr(obj, 'params') and isinstance(obj.params, dict):
            if key in obj.params:
                return obj.params[key]
        # 그 다음 객체 속성 확인
        return getattr(obj, key, default)


def vectorize_strategy_params(strategy: Dict[str, Any]) -> np.ndarray:
    """전략 파라미터를 벡터로 변환"""
    try:
        # 주요 파라미터 추출 및 정규화
        params = []

        # RSI (0-100)
        rsi_min = float(_get_value(strategy, 'rsi_min', 30.0)) / 100.0
        rsi_max = float(_get_value(strategy, 'rsi_max', 70.0)) / 100.0
        params.extend([rsi_min, rsi_max])

        # Volume ratio (0-10)
        vol_min = float(_get_value(strategy, 'volume_ratio_min', 1.0)) / 10.0
        vol_max = float(_get_value(strategy, 'volume_ratio_max', 2.0)) / 10.0
        params.extend([vol_min, vol_max])

        # MACD thresholds (-1 to 1)
        macd_buy = (float(_get_value(strategy, 'macd_buy_threshold', 0.01)) + 1.0) / 2.0
        macd_sell = (float(_get_value(strategy, 'macd_sell_threshold', -0.01)) + 1.0) / 2.0
        params.extend([macd_buy, macd_sell])

        # MFI (0-100)
        mfi_min = float(_get_value(strategy, 'mfi_min', 20.0)) / 100.0
        mfi_max = float(_get_value(strategy, 'mfi_max', 80.0)) / 100.0
        params.extend([mfi_min, mfi_max])

        # ATR (0-1)
        atr_min = float(_get_value(strategy, 'atr_min', 0.01))
        atr_max = float(_get_value(strategy, 'atr_max', 0.05))
        params.extend([atr_min, atr_max])

        # ADX (0-100)
        adx_min = float(_get_value(strategy, 'adx_min', 15.0)) / 100.0
        params.append(adx_min)

        # Stop loss & Take profit (0-1)
        stop_loss = float(_get_value(strategy, 'stop_loss_pct', 0.02))
        take_profit = float(_get_value(strategy, 'take_profit_pct', 0.04))
        params.extend([stop_loss, take_profit])

        return np.array(params)

    except Exception as e:
        logger.warning(f"⚠️ 전략 벡터화 실패: {e}, 기본값 사용")
        return np.zeros(13)


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    try:
        # 영벡터 체크
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 코사인 유사도
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # 0~1 범위로 클리핑
        return float(np.clip(similarity, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"⚠️ 코사인 유사도 계산 실패: {e}")
        return 0.0


def calculate_basic_similarity(strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> float:
    """Phase 1: 기본 파라미터 기반 유사도 (0~1)"""
    vec1 = vectorize_strategy_params(strategy1)
    vec2 = vectorize_strategy_params(strategy2)
    return calculate_cosine_similarity(vec1, vec2)


# =============================================================================
# Phase 2: 정교한 유사도 (레짐/타입 포함)
# =============================================================================

def calculate_smart_similarity(
    strategy1: Dict[str, Any],
    strategy2: Dict[str, Any],
    param_weight: float = 0.6,
    regime_weight: float = 0.2,
    type_weight: float = 0.2
) -> float:
    """
    Phase 2: 레짐과 타입을 고려한 정교한 유사도

    Args:
        strategy1, strategy2: 비교할 전략
        param_weight: 파라미터 유사도 가중치 (기본 60%)
        regime_weight: 레짐 일치도 가중치 (기본 20%)
        type_weight: 전략 타입 일치도 가중치 (기본 20%)

    Returns:
        종합 유사도 (0~1)
    """
    try:
        # 1. 파라미터 유사도 (60%)
        param_sim = calculate_basic_similarity(strategy1, strategy2) * param_weight

        # 2. 레짐 일치도 (20%)
        regime1 = _get_value(strategy1, 'regime', 'ranging')
        regime2 = _get_value(strategy2, 'regime', 'ranging')

        if regime1 == regime2:
            regime_match = 1.0
        elif regime1 in ['trending', 'bullish', 'bearish'] and regime2 in ['trending', 'bullish', 'bearish']:
            regime_match = 0.5  # 둘 다 트렌딩 계열
        elif regime1 in ['ranging', 'sideways', 'neutral'] and regime2 in ['ranging', 'sideways', 'neutral']:
            regime_match = 0.5  # 둘 다 레인징 계열
        else:
            regime_match = 0.2  # 완전 다름

        regime_sim = regime_match * regime_weight

        # 3. 전략 타입 일치도 (20%)
        type1 = _get_value(strategy1, 'strategy_type', 'hybrid')
        type2 = _get_value(strategy2, 'strategy_type', 'hybrid')

        if type1 == type2:
            type_match = 1.0
        elif type1 == 'hybrid' or type2 == 'hybrid':
            type_match = 0.6  # 하이브리드는 중간
        else:
            type_match = 0.3  # 다른 타입

        type_sim = type_match * type_weight

        # 종합 유사도
        total_similarity = param_sim + regime_sim + type_sim

        return float(np.clip(total_similarity, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"⚠️ 스마트 유사도 계산 실패: {e}, 기본 유사도 사용")
        return calculate_basic_similarity(strategy1, strategy2)


# =============================================================================
# Phase 3: 동적 에피소드 조정
# =============================================================================

def calculate_finetuning_episodes(
    similarity: float,
    min_episodes: int = 3,
    max_episodes: int = 12
) -> int:
    """
    Phase 3: 유사도에 따라 Fine-tuning 에피소드 수 동적 조정

    Args:
        similarity: 유사도 (0~1)
        min_episodes: 최소 에피소드 (매우 유사한 경우)
        max_episodes: 최대 에피소드 (어느정도 유사한 경우)

    Returns:
        Fine-tuning 에피소드 수
    """
    try:
        if similarity > 0.97:
            return min_episodes  # 3 에피소드
        elif similarity > 0.93:
            return (min_episodes + max_episodes) // 2  # 7 에피소드
        else:
            return max_episodes  # 12 에피소드

    except Exception as e:
        logger.warning(f"⚠️ 에피소드 계산 실패: {e}, 기본값 사용")
        return (min_episodes + max_episodes) // 2


# =============================================================================
# 유사도 기반 전략 분류
# =============================================================================

def find_most_similar_strategy(
    new_strategy: Dict[str, Any],
    existing_strategies: List[Dict[str, Any]],
    use_smart: bool = True
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    기존 전략 중 가장 유사한 전략 찾기

    Args:
        new_strategy: 새로운 전략
        existing_strategies: 기존 전략 리스트
        use_smart: True면 Phase 2 스마트 유사도, False면 Phase 1 기본 유사도

    Returns:
        (최대 유사도, 가장 유사한 전략) 튜플
    """
    if not existing_strategies:
        return 0.0, None

    max_similarity = 0.0
    most_similar = None

    similarity_func = calculate_smart_similarity if use_smart else calculate_basic_similarity

    for existing in existing_strategies:
        sim = similarity_func(new_strategy, existing)
        if sim > max_similarity:
            max_similarity = sim
            most_similar = existing

    return max_similarity, most_similar


def classify_strategy_by_similarity(
    new_strategy: Dict[str, Any],
    existing_strategies: List[Dict[str, Any]],
    duplicate_threshold: float = 0.9995,  # 🔥 조정: 0.99 → 0.9995 (더 엄격)
    copy_threshold: float = 0.995,  # 🔥 조정: 0.97 → 0.995
    finetune_threshold: float = 0.95,  # 🔥 조정: 0.90 → 0.95
    use_smart: bool = True
) -> Tuple[str, float, Optional[str]]:
    """
    유사도 기반 전략 분류

    Args:
        new_strategy: 새로운 전략
        existing_strategies: 기존 전략 리스트
        duplicate_threshold: 중복 판정 임계값 (기본 0.99)
        copy_threshold: 정책 복사 임계값 (기본 0.97)
        finetune_threshold: Fine-tuning 임계값 (기본 0.90)
        use_smart: Phase 2 스마트 유사도 사용 여부

    Returns:
        (분류, 유사도, 부모 전략 ID) 튜플
        분류: 'duplicate', 'copy', 'finetune', 'novel'
    """
    max_sim, parent = find_most_similar_strategy(new_strategy, existing_strategies, use_smart)

    parent_id = _get_value(parent, 'id', None) if parent else None

    if max_sim >= duplicate_threshold:
        return 'duplicate', max_sim, parent_id
    elif max_sim >= copy_threshold:
        return 'copy', max_sim, parent_id
    elif max_sim >= finetune_threshold:
        return 'finetune', max_sim, parent_id
    else:
        return 'novel', max_sim, parent_id


# =============================================================================
# 배치 처리 유틸리티
# =============================================================================

def vectorize_strategies_batch(strategies: List[Dict[str, Any]]) -> List[np.ndarray]:
    """전략 리스트를 벡터 리스트로 일괄 변환 (성능 최적화)"""
    return [vectorize_strategy_params(s) for s in strategies]


def classify_new_strategies_batch(
    new_strategies: List[Dict[str, Any]],
    existing_strategies: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    새 전략 리스트를 일괄 분류

    Returns:
        {
            'duplicate': [...],
            'copy': [...],
            'finetune': [...],
            'novel': [...]
        }
    """
    result = {
        'duplicate': [],
        'copy': [],
        'finetune': [],
        'novel': []
    }

    for new_strat in new_strategies:
        classification, similarity, parent_id = classify_strategy_by_similarity(
            new_strat, existing_strategies, **kwargs
        )

        # 메타데이터 추가 (dict와 Strategy 객체 모두 지원)
        if isinstance(new_strat, dict):
            new_strat['similarity_classification'] = classification
            new_strat['similarity_score'] = similarity
            new_strat['parent_strategy_id'] = parent_id
        else:
            # Strategy 객체인 경우
            # params에 저장
            if hasattr(new_strat, 'params') and isinstance(new_strat.params, dict):
                new_strat.params['similarity_classification'] = classification
                new_strat.params['similarity_score'] = similarity
                new_strat.params['parent_strategy_id'] = parent_id
            # 객체 속성으로도 저장
            new_strat.similarity_classification = classification
            new_strat.similarity_score = similarity
            new_strat.parent_strategy_id = parent_id

        result[classification].append(new_strat)

    return result
