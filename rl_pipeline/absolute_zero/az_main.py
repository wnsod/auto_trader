"""
Absolute Zero 시스템 - 메인 실행 모듈
핵심 실행 함수와 파이프라인 조정
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional

from .az_config import (
    configure_logging,
    ensure_storage_ready,
    AZ_DEBUG,
    AZ_INTERVALS,
    AZ_CANDLE_DAYS,
    AZ_ALLOW_FALLBACK,
    ENABLE_STRATEGY_FILTERING,
    STRATEGIES_DB_PATH
)
from .az_utils import (
    sort_intervals,
    execute_wal_checkpoint,
    format_time_duration,
    check_data_sufficiency,
    create_run_metadata,
    log_system_info,
    validate_environment
)
from .az_analysis import calculate_global_analysis_data
from .az_global_strategies import generate_global_strategies_only

logger = logging.getLogger(__name__)

# 새로운 파이프라인 구조 import
try:
    import rl_pipeline.core.env as core_env
    import rl_pipeline.core.errors as core_errors
    import rl_pipeline.strategy.manager as strategy_manager
    import rl_pipeline.simulation.selfplay as selfplay
    import rl_pipeline.routing.regime_router as regime_router
    import rl_pipeline.analysis.integrated_analyzer as integrated_analyzer
    import rl_pipeline.db.schema as db_schema
    import rl_pipeline.db.connection_pool as db_pool
    from rl_pipeline.monitoring import SessionManager

    config = core_env.config
    AZError = core_errors.AZError
    create_run_record = strategy_manager.create_run_record
    update_run_record = strategy_manager.update_run_record
    create_coin_strategies = strategy_manager.create_coin_strategies
    create_global_strategies = strategy_manager.create_global_strategies
    run_self_play_test = selfplay.run_self_play_test
    RegimeRouter = regime_router.RegimeRouter
    create_regime_routing_strategies = regime_router.create_regime_routing_strategies
    IntegratedAnalyzer = integrated_analyzer.IntegratedAnalyzer
    analyze_coin_strategies = integrated_analyzer.analyze_coin_strategies
    analyze_global_strategies = integrated_analyzer.analyze_global_strategies
    ensure_indexes = db_schema.ensure_indexes
    setup_database_tables = db_schema.setup_database_tables
    create_coin_strategies_table = db_schema.create_coin_strategies_table
    get_optimized_db_connection = db_pool.get_optimized_db_connection

    NEW_PIPELINE_AVAILABLE = True

except ImportError as e:
    logger.error(f"새로운 파이프라인 모듈 import 실패: {e}")
    config = None
    AZError = Exception
    NEW_PIPELINE_AVAILABLE = False

# 분리된 모듈 imports
from rl_pipeline.pipelines.orchestrator import (
    PipelineResult,
    IntegratedPipelineOrchestrator,
)
from rl_pipeline.db.learning_results import (
    create_learning_results_tables,
    save_pipeline_execution_log,
    save_regime_routing_results,
    get_pipeline_performance_summary,
)
from rl_pipeline.data.candle_loader import (
    get_available_coins_and_intervals,
    load_candle_data_for_coin,
)

def run_absolute_zero(
    coin: Optional[str] = None,
    interval: str = "15m",
    n_strategies: int = 300,
    intervals: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Absolute Zero 시스템 실행 - 새로운 파이프라인 구조 사용

    Args:
        coin: 코인 심볼 (None이면 자동 선택)
        interval: 기본 인터벌
        n_strategies: 생성할 전략 수
        intervals: 사용할 인터벌 리스트

    Returns:
        실행 결과 딕셔너리
    """
    try:
        start_time = datetime.now()

        # 환경 검증
        if not validate_environment():
            return {"error": "환경 검증 실패"}

        # 디버그 세션 생성
        session_manager = SessionManager()
        session_id = None

        try:
            # 인터벌 리스트 준비
            if intervals and len(intervals) > 0:
                intervals_for_session = intervals
            elif AZ_INTERVALS:
                intervals_for_session = [i.strip() for i in AZ_INTERVALS.split(',')]
            else:
                intervals_for_session = [interval]

            # 코인 결정
            coin_for_session = coin
            if coin_for_session is None:
                try:
                    available = get_available_coins_and_intervals()
                    coins = sorted(list({c for c, _ in available}))
                    if coins:
                        coin_for_session = coins[0]
                except:
                    coin_for_session = "UNKNOWN"

            session_id = session_manager.create_session(
                coins=[coin_for_session] if coin_for_session else ["UNKNOWN"],
                intervals=intervals_for_session,
                config={
                    "n_strategies": n_strategies,
                    "candle_days": AZ_CANDLE_DAYS
                }
            )
            logger.info(f"✅ 디버그 세션 생성: {session_id}")
        except Exception as session_err:
            logger.warning(f"⚠️ 디버그 세션 생성 실패 (계속 진행): {session_err}")
            session_id = None

        # 인터벌 처리
        if intervals and len(intervals) > 0:
            intervals_raw = intervals
        elif AZ_INTERVALS:
            intervals_raw = [i.strip() for i in AZ_INTERVALS.split(',')]
        else:
            intervals_raw = [interval]

        intervals_to_use = sort_intervals(intervals_raw)

        # 코인 기본값 설정
        if coin is None:
            try:
                available = get_available_coins_and_intervals()
                coins = sorted(list({c for c, _ in available}))
                if not coins:
                    raise ValueError("❌ DB에 사용 가능한 코인이 없습니다.")
                coin = coins[0]
            except Exception as e:
                logger.error(f"❌ 코인 목록 조회 실패: {e}")
                raise ValueError("❌ 코인을 지정하거나 DB에 캔들 데이터가 필요합니다.") from e

        logger.info(f"🚀 Absolute Zero 시스템 시작: {coin} {intervals_to_use}")
        logger.info(f"🗓️ 캔들 히스토리 일수: {AZ_CANDLE_DAYS}일")

        # 시스템 정보 로깅 (디버그 모드에서만)
        if AZ_DEBUG:
            log_system_info()

        # 새로운 파이프라인 사용 가능 여부 확인
        if not NEW_PIPELINE_AVAILABLE:
            logger.error("❌ 새로운 파이프라인 모듈을 사용할 수 없습니다")
            return {"error": "새로운 파이프라인 모듈 사용 불가"}

        # 실행 메타데이터 생성
        metadata = create_run_metadata(coin, intervals_to_use)

        # 실행 기록 생성
        try:
            create_run_record(
                metadata['run_id'],
                "Absolute Zero System 실행",
                coin=metadata['coin'],
                interval=metadata['interval_str']
            )
            logger.info(f"✅ 실행 기록 생성 완료: {metadata['run_id']}")
        except Exception as e:
            logger.warning(f"⚠️ 실행 기록 생성 실패: {e}")

        # 전략 필터링 (환경변수로 제어)
        if ENABLE_STRATEGY_FILTERING:
            try:
                logger.info("🔧 전략 필터링 시작...")
                from rl_pipeline.core.strategy_filter import remove_low_grade_strategies
                removed = remove_low_grade_strategies()
                if removed > 0:
                    logger.info(f"✅ {removed}개 F 등급 전략 제거")
            except Exception as e:
                logger.warning(f"⚠️ 전략 필터링 실패 (계속 진행): {e}")

        # 캔들 데이터 로드
        logger.info(f"📊 {coin} 캔들 데이터 로드 시작 (목표: {AZ_CANDLE_DAYS}일)...")
        all_candle_data = load_candle_data_for_coin(coin, intervals_to_use)

        if not all_candle_data:
            logger.error(f"❌ {coin} 캔들 데이터 로드 실패")
            return {"error": f"{coin} 캔들 데이터 로드 실패"}

        # 데이터 충분성 체크
        data_sufficient, insufficient_intervals = check_data_sufficiency(all_candle_data, coin)
        if not data_sufficient:
            return {"error": f"{coin}: 캔들 데이터 없음"}

        # 통합된 파이프라인 실행
        logger.info(f"🔄 {coin} 통합 파이프라인 실행 시작...")

        # 파이프라인 오케스트레이터 초기화
        orchestrator = IntegratedPipelineOrchestrator(session_id=session_id)

        # 각 인터벌별로 통합 파이프라인 실행
        pipeline_results = []
        for idx, interval in enumerate(intervals_to_use):
            try:
                logger.info(f"📊 {coin}-{interval} 통합 파이프라인 실행...")

                candle_data = all_candle_data.get((coin, interval))
                if candle_data is None or candle_data.empty:
                    logger.warning(f"⚠️ {coin}-{interval} 캔들 데이터 없음, 건너뜀")
                    continue

                # 1-2단계만 실행: 전략생성 → Self-play → 통합분석
                result = orchestrator.run_partial_pipeline(coin, interval, candle_data)
                pipeline_results.append(result)

                logger.info(f"✅ {coin}-{interval} 개별 인터벌 처리 완료")

                # WAL 체크포인트 (다음 인터벌 준비)
                if idx < len(intervals_to_use) - 1:
                    execute_wal_checkpoint(STRATEGIES_DB_PATH)

            except Exception as e:
                logger.error(f"❌ {coin}-{interval} 파이프라인 실패: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        if not pipeline_results:
            logger.error(f"❌ {coin}: 모든 인터벌 파이프라인 실패")
            return {"error": "모든 인터벌 파이프라인 실패"}

        # 글로벌 분석 수행
        logger.info("🌍 글로벌 분석 시작...")
        all_coin_strategies = {}

        for result in pipeline_results:
            if result and result.coin_strategies:
                key = f"{result.coin}_{result.interval}"
                all_coin_strategies[key] = {
                    'strategies': result.coin_strategies,
                    'analysis': result.coin_analysis
                }

        global_analysis = calculate_global_analysis_data(all_coin_strategies)

        # 실행 시간 계산
        end_time = datetime.now()
        execution_time = format_time_duration(start_time, end_time)

        # 최종 결과 정리
        final_result = {
            "success": True,
            "coin": coin,
            "intervals": intervals_to_use,
            "pipeline_results": len(pipeline_results),
            "global_analysis": global_analysis,
            "execution_time": execution_time,
            "insufficient_intervals": insufficient_intervals if insufficient_intervals else None
        }

        # 세션 종료
        if session_id:
            try:
                session_manager.end_session(session_id, summary=final_result)
                logger.info(f"✅ 디버그 세션 종료: {session_id}")
            except Exception as e:
                logger.warning(f"⚠️ 세션 종료 실패: {e}")

        logger.info(f"🎯 Absolute Zero 시스템 완료 - 실행 시간: {execution_time}")

        return final_result

    except Exception as e:
        logger.error(f"❌ Absolute Zero 시스템 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    """메인 실행 함수"""
    try:
        configure_logging()
        ensure_storage_ready()

        logger.info("=" * 60)
        logger.info("🚀 Absolute Zero System 시작")
        logger.info("=" * 60)

        # 데이터베이스 테이블 초기화
        setup_database_tables()
        create_coin_strategies_table()
        create_learning_results_tables()

        # 인덱스 생성
        ensure_indexes()

        # 사용 가능한 코인과 인터벌 조회
        available = get_available_coins_and_intervals()
        if not available:
            logger.error("❌ 사용 가능한 캔들 데이터가 없습니다")
            return {"error": "캔들 데이터 없음"}

        # 코인별로 고유한 것만 추출
        coins = sorted(list({coin for coin, _ in available}))
        intervals = sorted(list({interval for _, interval in available}))

        logger.info(f"📊 사용 가능한 코인: {coins[:10]}... (총 {len(coins)}개)")
        logger.info(f"📊 사용 가능한 인터벌: {intervals}")

        # 첫 번째 코인으로 시스템 실행
        coin = coins[0]
        logger.info(f"🎯 선택된 코인: {coin}")

        # Absolute Zero 시스템 실행 (모든 인터벌 사용)
        result = run_absolute_zero(
            coin=coin,
            intervals=intervals,
            n_strategies=300
        )

        # 결과 출력
        if result.get("success"):
            logger.info("✅ Absolute Zero System 성공적으로 완료")
            if result.get("global_analysis"):
                ga = result["global_analysis"]
                logger.info(f"📊 글로벌 분석:")
                logger.info(f"  - 프랙탈 점수: {ga.get('fractal_score', 0):.2f}")
                logger.info(f"  - 다중 시간대 일관성: {ga.get('multi_timeframe_coherence', 0):.2f}")
                logger.info(f"  - 지표 교차 검증: {ga.get('indicator_cross_validation', 0):.2f}")
                logger.info(f"  - 총 전략 수: {ga.get('total_strategies', 0)}")
        else:
            logger.error(f"❌ 실행 실패: {result.get('error', 'Unknown error')}")

        return result

    except Exception as e:
        logger.error(f"❌ 메인 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"메인 실행 실패: {e}"}

if __name__ == "__main__":
    main()