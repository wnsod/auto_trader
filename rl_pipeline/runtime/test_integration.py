"""
Phase 4 통합 검증 테스트
필터링, 다양한 설정, 엣지 케이스 검증
"""
import sys
import os
import logging

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.runtime import SignalRuntimeAdapter, AdapterConfig
from rl_pipeline.db.connection_pool import get_strategy_db_pool

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_filtering():
    """필터링 테스트"""
    logger.info("🧪 TEST 1: 필터링 검증")
    logger.info("=" * 80)

    # 낮은 등급 필터링
    config_strict = AdapterConfig(
        min_grade='A',  # A 이상만
        min_n_signals=100,  # 표본 100개 이상
        min_pf=1.5  # PF 1.5 이상
    )

    adapter_strict = SignalRuntimeAdapter(config_strict)

    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        # 모든 등급의 전략 가져오기
        cursor.execute("""
            SELECT DISTINCT s.coin, s.interval, s.regime_tag, s.strategy_id,
                   g.grade, g.grade_score, s.n_signals, s.pf
            FROM strategy_label_stats s
            JOIN strategy_grades g
                ON s.strategy_id = g.strategy_id
                AND s.interval = g.interval
                AND s.regime_tag = g.regime_tag
            ORDER BY g.grade DESC, s.n_signals DESC
            LIMIT 30
        """)

        test_cases = cursor.fetchall()

    passed_count = 0
    filtered_count = 0

    for coin, interval, regime_tag, sid, grade, score, n_signals, pf in test_cases:
        result = adapter_strict.get_signal_parameters(coin, interval, regime_tag, sid)

        expected_pass = (
            grade in ['S', 'A'] and
            n_signals >= 100 and
            pf >= 1.5
        )

        if result is not None:
            passed_count += 1
            if not expected_pass:
                logger.warning(f"  ⚠️ 예상 필터링 실패: {grade} n={n_signals} pf={pf:.2f}")
        else:
            filtered_count += 1

    logger.info(f"\n  ✅ 통과: {passed_count}개")
    logger.info(f"  🚫 필터링: {filtered_count}개")
    logger.info(f"  Total: {len(test_cases)}개\n")

def test_different_configs():
    """다양한 설정 테스트"""
    logger.info("🧪 TEST 2: 설정 변화 검증")
    logger.info("=" * 80)

    # 3가지 설정
    configs = {
        'conservative': AdapterConfig(
            tp_conservatism=0.5,  # TP 더 보수적 (50%)
            sl_conservatism=1.5,  # SL 더 넓게 (150%)
            min_grade='B'
        ),
        'balanced': AdapterConfig(
            tp_conservatism=0.7,  # 기본
            sl_conservatism=1.3,
            min_grade='C'
        ),
        'aggressive': AdapterConfig(
            tp_conservatism=0.9,  # TP 공격적 (90%)
            sl_conservatism=1.0,  # SL 타이트 (100%)
            min_grade='D'
        )
    }

    # S등급 전략 하나로 테스트
    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT s.coin, s.interval, s.regime_tag, s.strategy_id
            FROM strategy_label_stats s
            JOIN strategy_grades g
                ON s.strategy_id = g.strategy_id
                AND s.interval = g.interval
                AND s.regime_tag = g.regime_tag
            WHERE g.grade = 'S'
            LIMIT 1
        """)

        test_strategy = cursor.fetchone()

    if not test_strategy:
        logger.error("  ❌ S등급 전략 없음")
        return

    coin, interval, regime_tag, sid = test_strategy

    logger.info(f"  전략: {coin} {interval} {regime_tag}\n")

    for config_name, config in configs.items():
        adapter = SignalRuntimeAdapter(config)
        params = adapter.get_signal_parameters(coin, interval, regime_tag, sid)

        if params:
            logger.info(f"  [{config_name.upper()}]")
            logger.info(f"    TP: {params.tp*100:.2f}% | SL: {params.sl*100:.2f}%")
            logger.info(f"    Size: {params.size:.2f}x | Hold: {params.target_hold} candles")
        else:
            logger.info(f"  [{config_name.upper()}] - 필터링됨")

    logger.info("")

def test_edge_cases():
    """엣지 케이스 테스트"""
    logger.info("🧪 TEST 3: 엣지 케이스 검증")
    logger.info("=" * 80)

    adapter = SignalRuntimeAdapter()

    test_cases = [
        ("BTC", "15m", "ranging", "nonexistent_strategy_id_12345"),
        ("INVALID_COIN", "15m", "ranging", "some_strategy"),
        ("BTC", "INVALID_INTERVAL", "ranging", "some_strategy"),
    ]

    for coin, interval, regime_tag, sid in test_cases:
        result = adapter.get_signal_parameters(coin, interval, regime_tag, sid)

        status = "✅ 정상 None 반환" if result is None else f"⚠️ 예상외 결과: {result}"
        logger.info(f"  {coin} {interval} {regime_tag}: {status}")

    logger.info("")

def test_grade_distribution():
    """등급별 파라미터 분포 검증"""
    logger.info("🧪 TEST 4: 등급별 파라미터 분포")
    logger.info("=" * 80)

    adapter = SignalRuntimeAdapter()

    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        # 각 등급별로 샘플 추출
        for grade in ['S', 'A', 'B', 'C']:
            cursor.execute("""
                SELECT DISTINCT s.coin, s.interval, s.regime_tag, s.strategy_id
                FROM strategy_label_stats s
                JOIN strategy_grades g
                    ON s.strategy_id = g.strategy_id
                    AND s.interval = g.interval
                    AND s.regime_tag = g.regime_tag
                WHERE g.grade = ?
                LIMIT 5
            """, (grade,))

            strategies = cursor.fetchall()

            if not strategies:
                continue

            tp_list = []
            sl_list = []
            size_list = []

            for coin, interval, regime_tag, sid in strategies:
                params = adapter.get_signal_parameters(coin, interval, regime_tag, sid)
                if params:
                    tp_list.append(params.tp * 100)
                    sl_list.append(abs(params.sl) * 100)
                    size_list.append(params.size)

            if tp_list:
                logger.info(f"\n  [{grade}등급]")
                logger.info(f"    TP: {min(tp_list):.1f}~{max(tp_list):.1f}% (평균 {sum(tp_list)/len(tp_list):.1f}%)")
                logger.info(f"    SL: {min(sl_list):.1f}~{max(sl_list):.1f}% (평균 {sum(sl_list)/len(sl_list):.1f}%)")
                logger.info(f"    Size: {min(size_list):.2f}~{max(size_list):.2f}x (평균 {sum(size_list)/len(size_list):.2f}x)")

    logger.info("\n")

def main():
    """통합 검증 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("🚀 Phase 4 통합 검증 시작")
    logger.info("=" * 80 + "\n")

    try:
        test_filtering()
        test_different_configs()
        test_edge_cases()
        test_grade_distribution()

        logger.info("=" * 80)
        logger.info("✅ Phase 4 통합 검증 완료!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 검증 실패: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
