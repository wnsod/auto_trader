"""
레짐 기반 전략 관리 모듈

기능:
1. 레짐별 전략 수 제한 (최소 100, 최대 300)
2. 레짐별 전략 커버리지 보장
3. 전략 생성 시 레짐 타겟팅
4. 전략 정리 (하위 전략 제거)
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.core.regime_classifier import (
    calculate_regime_from_indicators,
    SIMPLIFIED_REGIMES
)

logger = logging.getLogger(__name__)


# ============================================================================
# 설정 상수
# ============================================================================

MIN_STRATEGIES_PER_REGIME = 100
MAX_STRATEGIES_PER_REGIME = 300
DEFAULT_REGIME = "neutral"  # 🔥 7개 레짐 체계에 맞춤


# ============================================================================
# 전략 수 집계
# ============================================================================

def count_strategies_by_regime(coin: str, interval: str) -> Dict[str, int]:
    """
    코인-인터벌별 레짐별 전략 수 집계

    Args:
        coin: 코인 심볼
        interval: 인터벌

    Returns:
        레짐별 전략 수 딕셔너리
    """
    try:
        # 🔥 코인별 DB 경로 사용
        from rl_pipeline.core.env import config
        coin_db_path = config.get_strategy_db_path(coin)
        
        with get_optimized_db_connection(coin_db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT regime, COUNT(*) as count
                FROM strategies
                WHERE symbol = ? AND interval = ?
                GROUP BY regime
            """, (coin, interval))

            regime_counts = {}
            for row in cursor.fetchall():
                regime = row[0] or DEFAULT_REGIME
                regime_counts[regime] = row[1]

            # 누락된 레짐은 0으로 설정
            for regime in SIMPLIFIED_REGIMES:
                if regime not in regime_counts:
                    regime_counts[regime] = 0

            return regime_counts

    except Exception as e:
        logger.error(f"❌ 레짐별 전략 수 집계 실패: {e}")
        return {regime: 0 for regime in SIMPLIFIED_REGIMES}


def get_total_strategy_count(coin: str, interval: str) -> int:
    """
    코인-인터벌의 총 전략 수 조회

    Args:
        coin: 코인 심볼
        interval: 인터벌

    Returns:
        총 전략 수
    """
    try:
        # 🔥 코인별 DB 경로 사용
        from rl_pipeline.core.env import config
        coin_db_path = config.get_strategy_db_path(coin)
        
        with get_optimized_db_connection(coin_db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM strategies
                WHERE symbol = ? AND interval = ?
            """, (coin, interval))

            return cursor.fetchone()[0]

    except Exception as e:
        logger.error(f"❌ 전략 수 조회 실패: {e}")
        return 0


# ============================================================================
# 레짐 커버리지 분석
# ============================================================================

def check_regime_coverage(coin: str, interval: str) -> Dict:
    """
    레짐 커버리지 체크

    Args:
        coin: 코인 심볼
        interval: 인터벌

    Returns:
        커버리지 정보 딕셔너리
    """
    regime_counts = count_strategies_by_regime(coin, interval)

    covered = []
    under_min = []
    over_max = []

    for regime, count in regime_counts.items():
        if count >= MIN_STRATEGIES_PER_REGIME:
            covered.append(regime)
            if count > MAX_STRATEGIES_PER_REGIME:
                over_max.append((regime, count))
        else:
            under_min.append((regime, count))

    return {
        'total_strategies': sum(regime_counts.values()),
        'regime_counts': regime_counts,
        'covered': covered,
        'under_min': under_min,
        'over_max': over_max,
        'coverage_rate': len(covered) / len(SIMPLIFIED_REGIMES) if SIMPLIFIED_REGIMES else 0
    }


def get_missing_regimes(coin: str, interval: str) -> List[str]:
    """
    최소 기준 미달 레짐 조회

    Args:
        coin: 코인 심볼
        interval: 인터벌

    Returns:
        부족한 레짐 리스트
    """
    regime_counts = count_strategies_by_regime(coin, interval)
    return [
        regime for regime, count in regime_counts.items()
        if count < MIN_STRATEGIES_PER_REGIME
    ]


# ============================================================================
# 전략 제한 및 정리
# ============================================================================

def limit_strategies_per_regime(
    coin: str,
    interval: str,
    regime: str,
    max_count: int = MAX_STRATEGIES_PER_REGIME,
    use_grade: bool = True
) -> int:
    """
    레짐별 전략 수 제한 (하위 성과 전략 제거)

    Args:
        coin: 코인 심볼
        interval: 인터벌
        regime: 레짐 타입
        max_count: 최대 전략 수
        use_grade: True이면 등급 우선, False이면 profit 우선

    Returns:
        삭제된 전략 수
    """
    try:
        # 🔥 코인별 DB 경로 사용
        from rl_pipeline.core.env import config
        coin_db_path = config.get_strategy_db_path(coin)
        
        with get_optimized_db_connection(coin_db_path) as conn:
            cursor = conn.cursor()

            # 현재 전략 수 확인
            cursor.execute("""
                SELECT COUNT(*) FROM strategies
                WHERE symbol = ? AND interval = ? AND regime = ?
            """, (coin, interval, regime))

            current_count = cursor.fetchone()[0]

            if current_count <= max_count:
                return 0

            # 삭제할 전략 수
            delete_count = current_count - max_count

            logger.info(
                f"🗑️ [{coin}-{interval}-{regime}] "
                f"전략 제한: {current_count}개 → {max_count}개 "
                f"(삭제: {delete_count}개, 기준: {'등급 우선' if use_grade else '수익 우선'})"
            )

            if use_grade:
                # 등급 기반 삭제: F > D > C > UNKNOWN > B > A > S
                # 같은 등급 내에서는 profit이 낮은 순서로 삭제
                cursor.execute("""
                    DELETE FROM strategies
                    WHERE id IN (
                        SELECT id FROM strategies
                        WHERE symbol = ? AND interval = ? AND regime = ?
                        ORDER BY
                            CASE quality_grade
                                WHEN 'F' THEN 1
                                WHEN 'D' THEN 2
                                WHEN 'C' THEN 3
                                WHEN 'UNKNOWN' THEN 4
                                WHEN 'B' THEN 5
                                WHEN 'A' THEN 6
                                WHEN 'S' THEN 7
                                ELSE 4  -- NULL은 UNKNOWN과 동일하게 처리
                            END ASC,
                            profit ASC
                        LIMIT ?
                    )
                """, (coin, interval, regime, delete_count))
            else:
                # 수익 기반 삭제 (기존 방식)
                cursor.execute("""
                    DELETE FROM strategies
                    WHERE id IN (
                        SELECT id FROM strategies
                        WHERE symbol = ? AND interval = ? AND regime = ?
                        ORDER BY profit ASC
                        LIMIT ?
                    )
                """, (coin, interval, regime, delete_count))

            conn.commit()

            logger.info(
                f"✅ [{coin}-{interval}-{regime}] "
                f"{delete_count}개 전략 삭제 완료"
            )

            return delete_count

    except Exception as e:
        logger.error(f"❌ 전략 제한 실패: {e}")
        return 0


def cleanup_all_regimes(coin: str, interval: str, use_grade: bool = True) -> Dict[str, int]:
    """
    모든 레짐의 전략 정리

    Args:
        coin: 코인 심볼
        interval: 인터벌
        use_grade: True이면 등급 기반, False이면 profit 기반

    Returns:
        레짐별 삭제 수 딕셔너리
    """
    deleted_counts = {}

    for regime in SIMPLIFIED_REGIMES:
        deleted = limit_strategies_per_regime(coin, interval, regime, use_grade=use_grade)
        if deleted > 0:
            deleted_counts[regime] = deleted

    return deleted_counts


# ============================================================================
# 전략 생성 타겟팅
# ============================================================================

def get_target_regime_for_generation(coin: str, interval: str) -> str:
    """
    신규 전략 생성 시 타겟 레짐 결정

    우선순위:
    1. 최소 기준 미달 레짐 (< 100개)
    2. 적게 분포된 레짐
    3. 기본값: ranging

    Args:
        coin: 코인 심볼
        interval: 인터벌

    Returns:
        타겟 레짐 이름
    """
    regime_counts = count_strategies_by_regime(coin, interval)

    # 1순위: 최소 기준 미달 레짐
    under_min = [
        regime for regime, count in regime_counts.items()
        if count < MIN_STRATEGIES_PER_REGIME
    ]

    if under_min:
        # 가장 적은 레짐 선택
        target = min(under_min, key=lambda r: regime_counts[r])
        logger.debug(
            f"📍 [{coin}-{interval}] 타겟 레짐: {target} "
            f"(현재: {regime_counts[target]}개, 목표: {MIN_STRATEGIES_PER_REGIME}개)"
        )
        return target

    # 2순위: 적게 분포된 레짐 (최대치 미만)
    available = [
        regime for regime, count in regime_counts.items()
        if count < MAX_STRATEGIES_PER_REGIME
    ]

    if available:
        target = min(available, key=lambda r: regime_counts[r])
        logger.debug(
            f"📍 [{coin}-{interval}] 타겟 레짐: {target} "
            f"(현재: {regime_counts[target]}개)"
        )
        return target

    # 기본값
    logger.debug(f"📍 [{coin}-{interval}] 기본 타겟 레짐: {DEFAULT_REGIME}")
    return DEFAULT_REGIME


def distribute_generation_targets(
    coin: str,
    interval: str,
    total_to_generate: int
) -> Dict[str, int]:
    """
    생성할 전략을 레짐별로 분배

    Args:
        coin: 코인 심볼
        interval: 인터벌
        total_to_generate: 총 생성할 전략 수

    Returns:
        레짐별 생성 수 딕셔너리
    """
    regime_counts = count_strategies_by_regime(coin, interval)

    # 각 레짐별 부족분 계산
    needs = {}
    for regime in SIMPLIFIED_REGIMES:
        current = regime_counts[regime]
        if current < MIN_STRATEGIES_PER_REGIME:
            needs[regime] = MIN_STRATEGIES_PER_REGIME - current
        elif current < MAX_STRATEGIES_PER_REGIME:
            needs[regime] = MAX_STRATEGIES_PER_REGIME - current
        else:
            needs[regime] = 0

    total_need = sum(needs.values())

    # 분배 계산
    distribution = {}

    if total_need == 0:
        # 모든 레짐이 최대치 → 균등 분배
        per_regime = total_to_generate // len(SIMPLIFIED_REGIMES)
        for regime in SIMPLIFIED_REGIMES:
            distribution[regime] = per_regime
    elif total_need <= total_to_generate:
        # 부족분을 모두 채울 수 있음
        for regime, need in needs.items():
            distribution[regime] = need

        # 남은 수량은 균등 분배
        remaining = total_to_generate - total_need
        per_regime = remaining // len(SIMPLIFIED_REGIMES)
        for regime in SIMPLIFIED_REGIMES:
            distribution[regime] = distribution.get(regime, 0) + per_regime
    else:
        # 부족분을 비율대로 분배
        for regime, need in needs.items():
            distribution[regime] = int(total_to_generate * need / total_need)

    logger.info(
        f"📊 [{coin}-{interval}] 전략 생성 분배: "
        f"총 {total_to_generate}개 → {distribution}"
    )

    return distribution


# ============================================================================
# 전략 관리 메인 함수
# ============================================================================

def manage_regime_strategies(coin: str, interval: str, use_grade: bool = True) -> Dict:
    """
    레짐 기반 전략 관리 (정리 + 커버리지 체크)

    Args:
        coin: 코인 심볼
        interval: 인터벌
        use_grade: True이면 등급 기반 정리, False이면 profit 기반 정리

    Returns:
        관리 결과 딕셔너리
    """
    try:
        logger.info(f"\n🔧 [{coin}-{interval}] 레짐 기반 전략 관리 시작 (기준: {'등급' if use_grade else '수익'})...")

        # 1. 현재 상태 체크
        coverage_before = check_regime_coverage(coin, interval)
        logger.info(
            f"📊 현재 상태: 총 {coverage_before['total_strategies']}개, "
            f"커버리지: {coverage_before['coverage_rate']:.1%}"
        )

        # 2. 초과 전략 정리
        deleted = cleanup_all_regimes(coin, interval, use_grade=use_grade)
        if deleted:
            logger.info(f"🗑️ 삭제된 전략: {sum(deleted.values())}개 ({deleted})")

        # 3. 최종 상태 체크
        coverage_after = check_regime_coverage(coin, interval)
        logger.info(
            f"✅ 최종 상태: 총 {coverage_after['total_strategies']}개, "
            f"커버리지: {coverage_after['coverage_rate']:.1%}"
        )

        # 4. 부족한 레짐 확인
        if coverage_after['under_min']:
            logger.warning(
                f"⚠️ 최소 기준 미달 레짐: "
                f"{[(r, c) for r, c in coverage_after['under_min']]}"
            )

        return {
            'before': coverage_before,
            'after': coverage_after,
            'deleted': deleted,
            'total_deleted': sum(deleted.values()) if deleted else 0
        }

    except Exception as e:
        logger.error(f"❌ 레짐 기반 전략 관리 실패: {e}", exc_info=True)
        return {}


# ============================================================================
# 배치 관리
# ============================================================================

def manage_all_strategies(coins: List[str], intervals: List[str]) -> Dict:
    """
    모든 코인-인터벌의 전략 관리

    Args:
        coins: 코인 리스트
        intervals: 인터벌 리스트

    Returns:
        전체 관리 결과 딕셔너리
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 전체 레짐 기반 전략 관리 시작")
    logger.info("="*80)

    total_deleted = 0
    results = {}

    for coin in coins:
        for interval in intervals:
            key = f"{coin}-{interval}"
            result = manage_regime_strategies(coin, interval)
            results[key] = result
            total_deleted += result.get('total_deleted', 0)

    logger.info("\n" + "="*80)
    logger.info(f"✅ 전체 관리 완료: 총 {total_deleted}개 전략 정리")
    logger.info("="*80)

    return {
        'total_deleted': total_deleted,
        'details': results
    }


# ============================================================================
# 유틸리티
# ============================================================================

def print_regime_summary(coin: str, interval: str):
    """레짐별 전략 현황 출력"""
    regime_counts = count_strategies_by_regime(coin, interval)
    coverage = check_regime_coverage(coin, interval)

    logger.info(f"\n📊 [{coin}-{interval}] 레짐별 전략 현황:")
    for regime in SIMPLIFIED_REGIMES:
        count = regime_counts[regime]
        status = "✅" if count >= MIN_STRATEGIES_PER_REGIME else "❌"
        if count > MAX_STRATEGIES_PER_REGIME:
            status = "⚠️"

        logger.info(
            f"   {regime:10s}: {count:4d}개 "
            f"(최소: {MIN_STRATEGIES_PER_REGIME}, 최대: {MAX_STRATEGIES_PER_REGIME}) "
            f"{status}"
        )

    logger.info(f"   커버리지: {coverage['coverage_rate']:.1%}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 테스트
    test_coin = "BTC"
    test_interval = "15m"

    print_regime_summary(test_coin, test_interval)
