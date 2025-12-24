"""
가중치 엔진 (Weight Engine)
공통화된 가중치 계산 시스템
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from rl_pipeline.db.connection_pool import get_optimized_db_connection

logger = logging.getLogger(__name__)

# 기본 최소 임계값
DEFAULT_TAU = 0.15

# 기본 퍼센타일 클립
DEFAULT_PERCENTILE = 85


def calc_weight(
    grade_score: float,
    acc_adaptive: float,      # 적응형 예측 정확도
    conf: float,              # 예측 확신도
    regime_fit: float,        # 레짐 적합도
    clip: bool = True,
    tau: float = DEFAULT_TAU,
    percentile: float = DEFAULT_PERCENTILE
) -> float:
    """
    가중치 계산
    
    공식: weight = grade_score × predictive_accuracy_adaptive × predicted_conf × regime_fit
    
    Args:
        grade_score: 등급 점수 (0.0 ~ 1.0)
        acc_adaptive: 적응형 예측 정확도 (0.0 ~ 1.0)
        conf: 예측 확신도 (0.0 ~ 1.0)
        regime_fit: 레짐 적합도 (0.0 ~ 1.0)
        clip: 퍼센타일 클립 적용 여부
        tau: 최소 임계값 (이하일 경우 0)
        percentile: 퍼센타일 클립 기준 (상위 N%)
    
    Returns:
        계산된 가중치 (0.0 이상)
    """
    try:
        # 기본 계산
        weight = grade_score * acc_adaptive * conf * regime_fit
        
        # NaN/0 처리
        if not weight or weight <= 0 or np.isnan(weight) or np.isinf(weight):
            return 0.0
        
        # 최소 임계값 체크
        if weight < tau:
            return 0.0
        
        # 퍼센타일 클립 (전체 전략 풀에서 상위 N%만 선택)
        if clip:
            # 동적 퍼센타일 클립은 별도 함수로 처리
            # 여기서는 단순히 값을 반환하고, 실제 클립은 calc_weights_batch에서 수행
            pass
        
        return float(weight)
        
    except Exception as e:
        logger.error(f"❌ 가중치 계산 실패: {e}")
        return 0.0


def calc_weights_batch(
    strategies: List[Dict[str, Any]],
    clip: bool = True,
    tau: float = DEFAULT_TAU,
    percentile: float = DEFAULT_PERCENTILE
) -> List[float]:
    """
    전략 리스트의 가중치 일괄 계산
    
    Args:
        strategies: 전략 딕셔너리 리스트
        clip: 퍼센타일 클립 적용 여부
        tau: 최소 임계값
        percentile: 퍼센타일 클립 기준
    
    Returns:
        가중치 리스트
    """
    try:
        weights = []
        
        for strategy in strategies:
            grade_score = strategy.get('grade_score', 0.5)
            acc_adaptive = strategy.get('predictive_accuracy', 0.5)
            conf = strategy.get('predicted_conf', 0.5)
            regime_fit = strategy.get('regime_fit', 1.0)
            
            weight = calc_weight(
                grade_score=grade_score,
                acc_adaptive=acc_adaptive,
                conf=conf,
                regime_fit=regime_fit,
                clip=False,  # 일괄 계산에서는 나중에 클립
                tau=tau
            )
            
            weights.append(weight)
        
        # 퍼센타일 클립 적용
        if clip and weights:
            weights = apply_percentile_clip(weights, percentile)
        
        return weights
        
    except Exception as e:
        logger.error(f"❌ 일괄 가중치 계산 실패: {e}")
        return [0.0] * len(strategies)


def apply_percentile_clip(weights: List[float], percentile: float = DEFAULT_PERCENTILE) -> List[float]:
    """
    퍼센타일 클립 적용
    
    상위 N%만 선택하고 나머지는 0으로 설정
    
    Args:
        weights: 가중치 리스트
        percentile: 퍼센타일 기준 (예: 85 = 상위 15%)
    
    Returns:
        클립된 가중치 리스트
    """
    try:
        if not weights:
            return []
        
        # 퍼센타일 임계값 계산
        threshold = np.percentile(weights, percentile)
        
        # 임계값 이하는 0으로 설정
        clipped_weights = [w if w >= threshold else 0.0 for w in weights]
        
        logger.debug(f"✅ 퍼센타일 클립 적용: {len([w for w in clipped_weights if w > 0])}/{len(weights)}개 선택")
        
        return clipped_weights
        
    except Exception as e:
        logger.error(f"❌ 퍼센타일 클립 실패: {e}")
        return weights  # 실패 시 원본 반환


def normalize_weights(weights: List[float]) -> List[float]:
    """
    가중치 정규화 (합이 1이 되도록)
    
    Args:
        weights: 가중치 리스트
    
    Returns:
        정규화된 가중치 리스트
    """
    try:
        if not weights:
            return []
        
        total = sum(weights)
        
        if total == 0:
            # 모두 0이면 균등 분배
            n = len(weights)
            return [1.0 / n] * n if n > 0 else []
        
        return [w / total for w in weights]
        
    except Exception as e:
        logger.error(f"❌ 가중치 정규화 실패: {e}")
        return weights


def get_strategy_weight(
    strategy_id: str,
    coin: str,
    interval: str,
    current_regime: str = "neutral",
    db_connection=None
) -> float:
    """
    DB에서 전략 정보를 조회하여 가중치 계산
    
    Args:
        strategy_id: 전략 ID
        coin: 코인
        interval: 인터벌
        current_regime: 현재 레짐
        db_connection: DB 연결 (None이면 새로 생성)
    
    Returns:
        계산된 가중치
    """
    try:
        # DB 연결
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _get_strategy_weight_from_db(
                    strategy_id, coin, interval, current_regime, conn
                )
        else:
            return _get_strategy_weight_from_db(
                strategy_id, coin, interval, current_regime, db_connection
            )
            
    except Exception as e:
        logger.error(f"❌ 전략 가중치 조회 실패: {e}")
        return 0.0


def _get_strategy_weight_from_db(
    strategy_id: str,
    coin: str,
    interval: str,
    current_regime: str,
    conn
) -> float:
    """DB에서 전략 정보 조회 후 가중치 계산"""
    try:
        cursor = conn.cursor()
        
        # strategy_grades에서 예측 정확도 조회
        cursor.execute("""
            SELECT predictive_accuracy, grade_score
            FROM strategy_grades
            WHERE strategy_id = ? AND coin = ? AND interval = ?
        """, (strategy_id, coin, interval))
        
        grade_result = cursor.fetchone()
        
        if grade_result:
            acc_adaptive = grade_result[0] if grade_result[0] is not None else 0.5
            grade_score = grade_result[1] if grade_result[1] is not None else 0.5
        else:
            # strategy_grades에 없으면 strategies에서 조회
            cursor.execute("""
                SELECT quality_grade, score
                FROM strategies
                WHERE id = ? AND coin = ? AND interval = ?
            """, (strategy_id, coin, interval))
            
            strategy_result = cursor.fetchone()
            if strategy_result:
                # quality_grade를 grade_score로 변환
                grade_map = {'S': 0.95, 'A': 0.85, 'B': 0.70, 'C': 0.50, 'D': 0.30, 'F': 0.10}
                quality_grade = strategy_result[0] if strategy_result[0] else 'C'
                grade_score = grade_map.get(quality_grade, 0.5)
                acc_adaptive = 0.5  # 기본값 (예측 정확도 없음)
            else:
                # 전략 없음
                return 0.0
        
        # 예측 확신도 조회 (최근 realtime_predictions에서)
        cursor.execute("""
            SELECT predicted_conf
            FROM realtime_predictions
            WHERE symbol = ? AND interval = ?
            ORDER BY ts DESC
            LIMIT 1
        """, (coin, interval))
        
        conf_result = cursor.fetchone()
        conf = conf_result[0] if conf_result and conf_result[0] is not None else 0.5
        
        # 레짐 적합도 (간단한 구현, 나중에 regime_router와 통합 가능)
        regime_fit = _calculate_regime_fit(current_regime, coin, interval, conn)
        
        # 가중치 계산
        weight = calc_weight(
            grade_score=grade_score,
            acc_adaptive=acc_adaptive,
            conf=conf,
            regime_fit=regime_fit,
            clip=False  # 단일 전략이므로 클립 불필요
        )
        
        return weight
        
    except Exception as e:
        logger.error(f"❌ DB 조회 중 가중치 계산 실패: {e}")
        return 0.0


def _calculate_regime_fit(
    current_regime: str,
    coin: str,
    interval: str,
    conn
) -> float:
    """
    레짐 적합도 계산 (간단한 구현)
    
    나중에 regime_router와 통합하여 정교하게 구현 가능
    """
    try:
        # 간단한 구현: 레짐별 기본 적합도
        regime_fit_map = {
            'extreme_bearish': 0.9,
            'bearish': 0.95,
            'sideways_bearish': 0.95,
            'neutral': 1.0,
            'sideways_bullish': 1.0,
            'bullish': 1.0,
            'extreme_bullish': 1.0
        }
        
        # 기본값
        base_fit = regime_fit_map.get(current_regime, 1.0)
        
        # TODO: 실제 레짐 라우터와 통합하여 정교한 계산
        # from rl_pipeline.routing.regime_router import RegimeRouter
        # router = RegimeRouter()
        # regime_fit = router.calculate_regime_fitness(...)
        
        return base_fit
        
    except Exception as e:
        logger.debug(f"⚠️ 레짐 적합도 계산 실패, 기본값 사용: {e}")
        return 1.0


if __name__ == "__main__":
    # 테스트
    print("가중치 엔진 테스트:")
    
    # 테스트 1: 기본 계산
    weight1 = calc_weight(
        grade_score=0.8,
        acc_adaptive=0.7,
        conf=0.9,
        regime_fit=1.0
    )
    print(f"테스트 1: weight={weight1:.3f}")
    
    # 테스트 2: 일괄 계산
    strategies = [
        {'grade_score': 0.8, 'predictive_accuracy': 0.7, 'predicted_conf': 0.9, 'regime_fit': 1.0},
        {'grade_score': 0.6, 'predictive_accuracy': 0.5, 'predicted_conf': 0.8, 'regime_fit': 0.9},
        {'grade_score': 0.9, 'predictive_accuracy': 0.8, 'predicted_conf': 0.95, 'regime_fit': 1.0},
    ]
    weights = calc_weights_batch(strategies, clip=True)
    print(f"테스트 2: weights={weights}")
    
    # 테스트 3: 정규화
    normalized = normalize_weights(weights)
    print(f"테스트 3: normalized={normalized}, sum={sum(normalized):.3f}")

