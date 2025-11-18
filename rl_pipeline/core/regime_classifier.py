"""
레짐 분류 모듈 (7개 → 3개 단순화)

기존 7개 레짐을 3개로 단순화:
- ranging: 횡보 (RSI 40-60)
- trending: 추세 (RSI < 30 or > 70)
- volatile: 변동성 (ATR/Price > 5%)
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 레짐 정의
# ============================================================================

# 기존 7개 레짐 (rl_candles_integrated.py에서 사용)
OLD_REGIME_STAGES = {
    1: "extreme_bearish",    # RSI < 20
    2: "bearish",           # RSI 20-40
    3: "sideways_bearish",  # RSI 40-50
    4: "neutral",           # RSI 45-55
    5: "sideways_bullish",  # RSI 50-60
    6: "bullish",           # RSI 60-80
    7: "extreme_bullish"    # RSI > 80
}

# 새로운 3개 레짐
SIMPLIFIED_REGIMES = ["ranging", "trending", "volatile"]

# 7개 → 3개 매핑
REGIME_MAPPING = {
    "extreme_bearish": "trending",
    "bearish": "trending",
    "sideways_bearish": "ranging",
    "neutral": "ranging",
    "sideways_bullish": "ranging",
    "bullish": "trending",
    "extreme_bullish": "trending"
}


# ============================================================================
# 레짐 분류 함수
# ============================================================================

def classify_regime_from_old(old_regime: str) -> str:
    """
    기존 7개 레짐을 3개로 변환

    Args:
        old_regime: 기존 레짐 이름 (extreme_bearish, bearish, ...)

    Returns:
        새 레짐 이름 (ranging, trending, volatile)
    """
    return REGIME_MAPPING.get(old_regime, "ranging")


def classify_regime_from_rsi_stage(rsi_stage: int) -> str:
    """
    RSI 스테이지 번호 (1-7)를 3개 레짐으로 변환

    Args:
        rsi_stage: RSI 스테이지 (1-7)

    Returns:
        레짐 이름 (ranging, trending, volatile)
    """
    old_regime = OLD_REGIME_STAGES.get(rsi_stage, "neutral")
    return classify_regime_from_old(old_regime)


def calculate_regime_from_indicators(
    rsi: float,
    atr: Optional[float] = None,
    price: Optional[float] = None,
    volatility: Optional[float] = None
) -> str:
    """
    지표 기반 레짐 계산 (직접 계산)

    우선순위:
    1. 변동성 체크 (ATR/Price > 5% 또는 volatility > 0.05)
    2. 추세 체크 (RSI < 30 또는 RSI > 70)
    3. 기본값: ranging

    Args:
        rsi: RSI 값 (0-100)
        atr: ATR 값 (절대값, 옵션)
        price: 현재 가격 (옵션)
        volatility: 변동성 비율 (ATR/Price, 옵션)

    Returns:
        레짐 이름 (ranging, trending, volatile)
    """
    # RSI 기본값
    if rsi is None or np.isnan(rsi):
        rsi = 50.0

    # 변동성 계산
    if volatility is None and atr is not None and price is not None and price > 0:
        volatility = atr / price

    # 1순위: 변동성 체크 (> 5%)
    if volatility is not None and volatility > 0.05:
        return "volatile"

    # 2순위: 추세 체크
    if rsi < 30 or rsi > 70:
        return "trending"

    # 기본: 횡보
    return "ranging"


def calculate_regime_from_candle_data(candle_data: Dict) -> str:
    """
    캔들 데이터에서 레짐 계산

    Args:
        candle_data: 캔들 데이터 딕셔너리
            - rsi: RSI 값
            - atr: ATR 값 (옵션)
            - close: 종가 (옵션)
            - volatility: 변동성 (옵션)

    Returns:
        레짐 이름 (ranging, trending, volatile)
    """
    rsi = candle_data.get('rsi', 50.0)
    atr = candle_data.get('atr')
    price = candle_data.get('close')
    volatility = candle_data.get('volatility')

    return calculate_regime_from_indicators(rsi, atr, price, volatility)


# ============================================================================
# 배치 처리
# ============================================================================

def classify_regimes_batch(candles: List[Dict]) -> List[str]:
    """
    여러 캔들 데이터의 레짐을 일괄 분류

    Args:
        candles: 캔들 데이터 리스트

    Returns:
        레짐 리스트
    """
    return [calculate_regime_from_candle_data(candle) for candle in candles]


def get_regime_distribution(regimes: List[str]) -> Dict[str, int]:
    """
    레짐 분포 계산

    Args:
        regimes: 레짐 리스트

    Returns:
        레짐별 개수 딕셔너리
    """
    distribution = {regime: 0 for regime in SIMPLIFIED_REGIMES}
    for regime in regimes:
        if regime in distribution:
            distribution[regime] += 1

    return distribution


# ============================================================================
# 레짐 전환 감지
# ============================================================================

def detect_regime_transition(
    current_regime: str,
    prev_regime: str,
    min_stay_periods: int = 2
) -> Dict:
    """
    레짐 전환 감지

    Args:
        current_regime: 현재 레짐
        prev_regime: 이전 레짐
        min_stay_periods: 최소 체류 기간 (기본 2)

    Returns:
        전환 정보 딕셔너리
    """
    is_transition = current_regime != prev_regime

    return {
        'is_transition': is_transition,
        'from_regime': prev_regime,
        'to_regime': current_regime,
        'transition_type': f"{prev_regime}_to_{current_regime}" if is_transition else "stable"
    }


# ============================================================================
# 레짐별 통계
# ============================================================================

def get_regime_stats(candles: List[Dict]) -> Dict:
    """
    레짐별 통계 계산

    Args:
        candles: 캔들 데이터 리스트

    Returns:
        통계 딕셔너리
    """
    if not candles:
        return {}

    regimes = classify_regimes_batch(candles)
    distribution = get_regime_distribution(regimes)

    total = len(regimes)
    percentages = {
        regime: count / total * 100 if total > 0 else 0
        for regime, count in distribution.items()
    }

    return {
        'total_candles': total,
        'distribution': distribution,
        'percentages': percentages,
        'dominant_regime': max(distribution, key=distribution.get) if distribution else None
    }


# ============================================================================
# 유틸리티 함수
# ============================================================================

def is_trending_market(rsi: float, threshold: float = 30.0) -> bool:
    """
    추세 시장 여부 판단

    Args:
        rsi: RSI 값
        threshold: 임계값 (기본 30, RSI < 30 or > 70)

    Returns:
        추세 시장 여부
    """
    return rsi < threshold or rsi > (100 - threshold)


def is_volatile_market(volatility: float, threshold: float = 0.05) -> bool:
    """
    변동성 높은 시장 여부 판단

    Args:
        volatility: 변동성 (ATR/Price)
        threshold: 임계값 (기본 5%)

    Returns:
        고변동성 시장 여부
    """
    return volatility > threshold


def is_ranging_market(rsi: float, volatility: float) -> bool:
    """
    횡보 시장 여부 판단

    Args:
        rsi: RSI 값
        volatility: 변동성

    Returns:
        횡보 시장 여부
    """
    return not is_trending_market(rsi) and not is_volatile_market(volatility)


# ============================================================================
# 테스트 함수
# ============================================================================

def test_regime_classification():
    """레짐 분류 테스트"""
    logger.info("\n" + "="*80)
    logger.info("레짐 분류 테스트")
    logger.info("="*80)

    # 테스트 케이스
    test_cases = [
        {"rsi": 25.0, "atr": 0.02, "close": 100.0, "expected": "trending"},
        {"rsi": 75.0, "atr": 0.02, "close": 100.0, "expected": "trending"},
        {"rsi": 50.0, "atr": 0.02, "close": 100.0, "expected": "ranging"},
        {"rsi": 50.0, "atr": 0.06, "close": 100.0, "expected": "volatile"},
        {"rsi": 45.0, "atr": 0.03, "close": 100.0, "expected": "ranging"},
    ]

    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, 1):
        result = calculate_regime_from_candle_data(case)
        expected = case['expected']
        status = "✅" if result == expected else "❌"

        logger.info(
            f"Test {i}: RSI={case['rsi']:5.1f}, "
            f"ATR/Price={case['atr']/case['close']:.2%} → "
            f"{result:10s} (expected: {expected:10s}) {status}"
        )

        if result == expected:
            passed += 1
        else:
            failed += 1

    logger.info("\n" + "="*80)
    logger.info(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    logger.info("="*80)

    return failed == 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_regime_classification()
