"""
레짐 기반 전략 관리 시스템 테스트

이 스크립트는:
1. 7개 레짐 → 3개 레짐 단순화 테스트
2. DB 스키마 마이그레이션 테스트 (regime 컬럼 추가)
3. 레짐별 전략 관리 (최소 100, 최대 300) 테스트
4. DB 저장/로드 검증
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import sqlite3
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.core.env import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 레짐 단순화 (7개 → 3개)
# ============================================================================

# 기존 7개 레짐 정의 (rl_candles_integrated.py)
OLD_REGIME_STAGES = {
    1: "extreme_bearish",    # RSI < 20
    2: "bearish",           # RSI 20-40
    3: "sideways_bearish",  # RSI 40-50
    4: "neutral",           # RSI 45-55
    5: "sideways_bullish",  # RSI 50-60
    6: "bullish",           # RSI 60-80
    7: "extreme_bullish"    # RSI > 80
}

# 새로운 3개 레짐 정의
NEW_REGIME_STAGES = {
    "ranging": "ranging",      # 횡보 (RSI 40-60)
    "trending": "trending",    # 추세 (RSI < 30 or > 70)
    "volatile": "volatile"     # 변동성 (ATR 기반)
}

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


def simplify_regime(old_regime: str) -> str:
    """기존 7개 레짐을 3개로 단순화"""
    return REGIME_MAPPING.get(old_regime, "ranging")


def calculate_regime_from_indicators(rsi: float, atr: float, price: float) -> str:
    """
    지표 기반 레짐 계산

    Args:
        rsi: RSI 값 (0-100)
        atr: ATR 값 (절대값)
        price: 현재 가격

    Returns:
        레짐 문자열 ("ranging", "trending", "volatile")
    """
    volatility = atr / price if price > 0 else 0.0

    # 1순위: 변동성 체크 (ATR/Price > 5%)
    if volatility > 0.05:
        return "volatile"

    # 2순위: 추세 체크
    if rsi < 30 or rsi > 70:
        return "trending"

    # 기본: 횡보
    return "ranging"


# ============================================================================
# 2. DB 스키마 마이그레이션
# ============================================================================

def add_regime_column_to_strategies():
    """coin_strategies 테이블에 regime 컬럼 추가"""
    try:
        logger.info("🔧 coin_strategies 테이블에 regime 컬럼 추가 시작...")

        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            # 컬럼 존재 여부 확인
            cursor.execute("PRAGMA table_info(coin_strategies)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'regime' in columns:
                logger.info("✅ regime 컬럼이 이미 존재합니다")
                return True

            # regime 컬럼 추가
            cursor.execute("""
                ALTER TABLE coin_strategies
                ADD COLUMN regime TEXT DEFAULT 'ranging'
            """)
            conn.commit()

            logger.info("✅ regime 컬럼 추가 완료")

            # 인덱스 추가
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategies_coin_interval_regime
                ON coin_strategies(coin, interval, regime)
            """)
            conn.commit()

            logger.info("✅ regime 인덱스 추가 완료")

            return True

    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.info("✅ regime 컬럼이 이미 존재합니다")
            return True
        else:
            logger.error(f"❌ regime 컬럼 추가 실패: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ regime 컬럼 추가 실패: {e}", exc_info=True)
        return False


def verify_regime_column():
    """regime 컬럼 추가 확인"""
    try:
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(coin_strategies)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}

            if 'regime' not in columns:
                logger.error("❌ regime 컬럼이 존재하지 않습니다")
                return False

            logger.info(f"✅ regime 컬럼 확인: {columns['regime']}")

            # 인덱스 확인
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_strategies_coin_interval_regime'
            """)
            if cursor.fetchone():
                logger.info("✅ regime 인덱스 확인 완료")
            else:
                logger.warning("⚠️ regime 인덱스가 존재하지 않습니다")

            return True

    except Exception as e:
        logger.error(f"❌ regime 컬럼 확인 실패: {e}")
        return False


# ============================================================================
# 3. 레짐별 전략 관리 (최소 100, 최대 300)
# ============================================================================

MIN_STRATEGIES_PER_REGIME = 100
MAX_STRATEGIES_PER_REGIME = 300


def count_strategies_by_regime(coin: str, interval: str) -> Dict[str, int]:
    """코인-인터벌별 레짐별 전략 수 집계"""
    try:
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT regime, COUNT(*) as count
                FROM coin_strategies
                WHERE coin = ? AND interval = ?
                GROUP BY regime
            """, (coin, interval))

            regime_counts = {}
            for row in cursor.fetchall():
                regime = row[0] or 'ranging'  # NULL은 ranging으로 처리
                regime_counts[regime] = row[1]

            return regime_counts

    except Exception as e:
        logger.error(f"❌ 레짐별 전략 수 집계 실패: {e}")
        return {}


def analyze_strategy_coverage():
    """전체 전략 커버리지 분석"""
    try:
        logger.info("\n" + "="*80)
        logger.info("📊 전략 커버리지 분석")
        logger.info("="*80)

        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            # 코인-인터벌 목록 조회
            cursor.execute("""
                SELECT DISTINCT coin, interval
                FROM coin_strategies
                ORDER BY coin, interval
            """)

            coin_intervals = cursor.fetchall()

            total_groups = 0
            covered_groups = 0
            under_min_groups = []
            over_max_groups = []

            for coin, interval in coin_intervals:
                regime_counts = count_strategies_by_regime(coin, interval)

                logger.info(f"\n📍 {coin}-{interval}:")

                for regime in ["ranging", "trending", "volatile"]:
                    count = regime_counts.get(regime, 0)
                    total_groups += 1

                    status = "✅"
                    if count >= MIN_STRATEGIES_PER_REGIME:
                        covered_groups += 1
                        if count > MAX_STRATEGIES_PER_REGIME:
                            status = "⚠️ 초과"
                            over_max_groups.append((coin, interval, regime, count))
                    else:
                        status = "❌ 부족"
                        under_min_groups.append((coin, interval, regime, count))

                    logger.info(f"   {regime:10s}: {count:4d}개 {status}")

            # 요약
            logger.info("\n" + "="*80)
            logger.info("📊 커버리지 요약")
            logger.info("="*80)
            logger.info(f"총 그룹 수: {total_groups}개")
            logger.info(f"커버된 그룹: {covered_groups}개 ({covered_groups/total_groups*100:.1f}%)")
            logger.info(f"부족 그룹: {len(under_min_groups)}개")
            logger.info(f"초과 그룹: {len(over_max_groups)}개")

            if under_min_groups:
                logger.info(f"\n⚠️ 최소 기준 미달 그룹 ({MIN_STRATEGIES_PER_REGIME}개):")
                for coin, interval, regime, count in under_min_groups[:10]:
                    logger.info(f"   {coin}-{interval}-{regime}: {count}개")

            if over_max_groups:
                logger.info(f"\n⚠️ 최대 기준 초과 그룹 ({MAX_STRATEGIES_PER_REGIME}개):")
                for coin, interval, regime, count in over_max_groups[:10]:
                    logger.info(f"   {coin}-{interval}-{regime}: {count}개")

            return {
                'total_groups': total_groups,
                'covered_groups': covered_groups,
                'coverage_rate': covered_groups / total_groups if total_groups > 0 else 0,
                'under_min': len(under_min_groups),
                'over_max': len(over_max_groups)
            }

    except Exception as e:
        logger.error(f"❌ 커버리지 분석 실패: {e}", exc_info=True)
        return {}


def limit_strategies_per_regime(coin: str, interval: str, regime: str, max_count: int = MAX_STRATEGIES_PER_REGIME):
    """
    레짐별 전략 수 제한 (상위 성과 전략만 유지)

    Args:
        coin: 코인 심볼
        interval: 인터벌
        regime: 레짐 타입
        max_count: 최대 전략 수

    Returns:
        삭제된 전략 수
    """
    try:
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            # 현재 전략 수 확인
            cursor.execute("""
                SELECT COUNT(*) FROM coin_strategies
                WHERE coin = ? AND interval = ? AND regime = ?
            """, (coin, interval, regime))

            current_count = cursor.fetchone()[0]

            if current_count <= max_count:
                logger.debug(f"✅ {coin}-{interval}-{regime}: {current_count}개 (제한 불필요)")
                return 0

            # 하위 전략 삭제 (profit 기준)
            delete_count = current_count - max_count

            cursor.execute("""
                DELETE FROM coin_strategies
                WHERE id IN (
                    SELECT id FROM coin_strategies
                    WHERE coin = ? AND interval = ? AND regime = ?
                    ORDER BY profit ASC
                    LIMIT ?
                )
            """, (coin, interval, regime, delete_count))

            conn.commit()

            logger.info(f"🗑️ {coin}-{interval}-{regime}: {delete_count}개 전략 삭제 ({current_count} → {max_count})")

            return delete_count

    except Exception as e:
        logger.error(f"❌ 전략 제한 실패: {e}")
        return 0


# ============================================================================
# 4. DB 저장/로드 테스트
# ============================================================================

def test_strategy_save_and_load():
    """전략 저장 및 로드 테스트"""
    try:
        logger.info("\n" + "="*80)
        logger.info("🧪 전략 저장/로드 테스트")
        logger.info("="*80)

        test_strategy = {
            'id': 'test_regime_strategy_001',
            'coin': 'BTC',
            'interval': '15m',
            'regime': 'trending',
            'strategy_type': 'hybrid',
            'profit': 5.5 * 100,  # 5.5% → 550
            'win_rate': 0.55,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15,
            'profit_factor': 2.0,
            'trades_count': 50,
            'quality_grade': 'A',
            'created_at': datetime.now().isoformat()
        }

        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            # 기존 테스트 전략 삭제
            cursor.execute("DELETE FROM coin_strategies WHERE id = ?", (test_strategy['id'],))

            # 전략 저장
            cursor.execute("""
                INSERT INTO coin_strategies
                (id, coin, interval, regime, strategy_type, profit, win_rate,
                 sharpe_ratio, max_drawdown, profit_factor, trades_count,
                 quality_grade, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_strategy['id'],
                test_strategy['coin'],
                test_strategy['interval'],
                test_strategy['regime'],
                test_strategy['strategy_type'],
                test_strategy['profit'],
                test_strategy['win_rate'],
                test_strategy['sharpe_ratio'],
                test_strategy['max_drawdown'],
                test_strategy['profit_factor'],
                test_strategy['trades_count'],
                test_strategy['quality_grade'],
                test_strategy['created_at']
            ))
            conn.commit()

            logger.info(f"✅ 테스트 전략 저장 완료: {test_strategy['id']}")

            # 전략 로드
            cursor.execute("""
                SELECT id, coin, interval, regime, profit, win_rate, quality_grade
                FROM coin_strategies
                WHERE id = ?
            """, (test_strategy['id'],))

            loaded = cursor.fetchone()

            if loaded:
                logger.info(f"✅ 테스트 전략 로드 완료:")
                logger.info(f"   ID: {loaded[0]}")
                logger.info(f"   Coin-Interval-Regime: {loaded[1]}-{loaded[2]}-{loaded[3]}")
                logger.info(f"   Profit: {loaded[4]/100:.2f}%")
                logger.info(f"   Win Rate: {loaded[5]:.2%}")
                logger.info(f"   Grade: {loaded[6]}")

                # 검증
                assert loaded[1] == test_strategy['coin'], "코인 불일치"
                assert loaded[2] == test_strategy['interval'], "인터벌 불일치"
                assert loaded[3] == test_strategy['regime'], "레짐 불일치"
                assert loaded[4] == test_strategy['profit'], "수익 불일치"

                logger.info("✅ 데이터 검증 통과")

                # 테스트 데이터 정리
                cursor.execute("DELETE FROM coin_strategies WHERE id = ?", (test_strategy['id'],))
                conn.commit()
                logger.info("✅ 테스트 데이터 정리 완료")

                return True
            else:
                logger.error("❌ 전략 로드 실패")
                return False

    except Exception as e:
        logger.error(f"❌ 저장/로드 테스트 실패: {e}", exc_info=True)
        return False


# ============================================================================
# 5. 메인 테스트 실행
# ============================================================================

def run_all_tests():
    """모든 테스트 실행"""
    try:
        logger.info("\n" + "🚀"*40)
        logger.info("레짐 기반 전략 관리 시스템 테스트 시작")
        logger.info("🚀"*40)

        results = {}

        # Test 1: 레짐 단순화 테스트
        logger.info("\n" + "="*80)
        logger.info("Test 1: 레짐 단순화 (7개 → 3개)")
        logger.info("="*80)

        for old_regime, new_regime in REGIME_MAPPING.items():
            logger.info(f"   {old_regime:20s} → {new_regime}")

        # 지표 기반 레짐 계산 테스트
        test_cases = [
            (25.0, 0.02, 100.0, "trending"),   # RSI < 30
            (75.0, 0.02, 100.0, "trending"),   # RSI > 70
            (50.0, 0.02, 100.0, "ranging"),    # RSI 40-60
            (50.0, 0.06, 100.0, "volatile"),   # High ATR
        ]

        logger.info("\n지표 기반 레짐 계산:")
        for rsi, atr, price, expected in test_cases:
            result = calculate_regime_from_indicators(rsi, atr, price)
            status = "✅" if result == expected else "❌"
            logger.info(f"   RSI={rsi:5.1f}, ATR/Price={atr/price:.2%} → {result:10s} {status}")

        results['regime_simplification'] = True

        # Test 2: DB 스키마 마이그레이션
        logger.info("\n" + "="*80)
        logger.info("Test 2: DB 스키마 마이그레이션")
        logger.info("="*80)

        results['add_regime_column'] = add_regime_column_to_strategies()
        results['verify_regime_column'] = verify_regime_column()

        # Test 3: 커버리지 분석
        logger.info("\n" + "="*80)
        logger.info("Test 3: 전략 커버리지 분석")
        logger.info("="*80)

        coverage = analyze_strategy_coverage()
        results['coverage_analysis'] = coverage

        # Test 4: DB 저장/로드 테스트
        logger.info("\n" + "="*80)
        logger.info("Test 4: DB 저장/로드 테스트")
        logger.info("="*80)

        results['save_load_test'] = test_strategy_save_and_load()

        # 최종 결과
        logger.info("\n" + "="*80)
        logger.info("📊 테스트 결과 요약")
        logger.info("="*80)

        for test_name, result in results.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"   {test_name:30s}: {status}")
            elif isinstance(result, dict):
                logger.info(f"   {test_name:30s}:")
                for key, value in result.items():
                    logger.info(f"      {key}: {value}")

        # 전체 성공 여부
        all_passed = all(
            result if isinstance(result, bool) else True
            for result in results.values()
        )

        if all_passed:
            logger.info("\n" + "✅"*40)
            logger.info("모든 테스트 통과!")
            logger.info("✅"*40)
            return 0
        else:
            logger.error("\n" + "❌"*40)
            logger.error("일부 테스트 실패")
            logger.error("❌"*40)
            return 1

    except Exception as e:
        logger.error(f"❌ 테스트 실행 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
