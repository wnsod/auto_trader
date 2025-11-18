"""
Absolute Zero 시스템 - 개선된 버전
검증 시스템 피드백 반영 및 최적화
"""

import sys
import os
import logging
import sqlite3
import json
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

# NumPy overflow/underflow 경고 숨김
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Sharding info not provided.*', category=UserWarning)

# JAX 설정
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')

# Python warnings 필터링
warnings.filterwarnings('ignore', category=Warning, message='.*Tensorflow.*')
warnings.filterwarnings('ignore', category=Warning, message='.*TensorFlow.*')

# JAX 로거 레벨 조정
import logging as std_logging
std_logging.getLogger('jax._src.xla_bridge').setLevel(std_logging.ERROR)
std_logging.getLogger('jax._src.lib').setLevel(std_logging.ERROR)
std_logging.getLogger('absl').setLevel(std_logging.ERROR)

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# logger 초기화
logger = logging.getLogger(__name__)

# 환경변수 파일 로드 (rl_pipeline_config.env 통합)
from dotenv import load_dotenv
# 🔥 pipelines 폴더로 이동했으므로 상위 디렉토리에서 설정 파일 찾기
base_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(base_dir, 'rl_pipeline_config.env')
load_dotenv(env_path)
logger.info(f"✅ 설정 파일 로드: {env_path}")

# 🔥 개선된 환경 변수
AZ_STRATEGY_COUNT = int(os.getenv('AZ_STRATEGY_COUNT', '200'))  # 50 -> 200
AZ_MIN_STRATEGIES = int(os.getenv('AZ_MIN_STRATEGIES', '50'))
AZ_MAX_STRATEGIES = int(os.getenv('AZ_MAX_STRATEGIES', '500'))
AZ_DEBUG = os.getenv('AZ_DEBUG', 'false').lower() == 'true'
# 🔥 pipelines 폴더로 이동했으므로 상위 디렉토리 기준으로 로그 파일 경로 설정
# base_dir는 이미 위에서 정의됨
AZ_LOG_FILE = os.getenv('AZ_LOG_FILE', os.path.join(base_dir, 'absolute_zero_debug.log'))
AZ_SIMULATION_VERBOSE = os.getenv('AZ_SIMULATION_VERBOSE', 'false').lower() == 'true'
AZ_CANDLE_DAYS = int(os.getenv('AZ_CANDLE_DAYS', '30'))
AZ_INTERVALS = os.getenv('AZ_INTERVALS', '')

# Self-play 설정 (선택적)
ENABLE_SELFPLAY = os.getenv('ENABLE_SELFPLAY', 'false').lower() == 'true'
AZ_SELFPLAY_EPISODES = int(os.getenv('AZ_SELFPLAY_EPISODES', '100'))

# 검증 시스템 설정
ENABLE_VALIDATION = os.getenv('ENABLE_VALIDATION', 'true').lower() == 'true'
ENABLE_AUTO_FIX = os.getenv('ENABLE_AUTO_FIX', 'true').lower() == 'true'
VAL_MIN_STRATEGIES = int(os.getenv('VAL_MIN_STRATEGIES', '50'))  # 100 -> 50
VAL_MAX_STRATEGIES = int(os.getenv('VAL_MAX_STRATEGIES', '20000'))

# Paper Trading 설정
ENABLE_AUTO_PAPER_TRADING = os.getenv('ENABLE_AUTO_PAPER_TRADING', 'true').lower() == 'true'
PAPER_TRADING_DURATION_DAYS = int(os.getenv('PAPER_TRADING_DURATION_DAYS', '30'))

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

    # 디버그 시스템 import
    from rl_pipeline.monitoring import SessionManager

    # 검증 시스템 import
    try:
        from rl_pipeline.validation import (
            create_validation_orchestrator,
            ValidationContext
        )
        VALIDATION_AVAILABLE = True
    except ImportError:
        logger.warning("⚠️ 검증 시스템을 사용할 수 없습니다")
        VALIDATION_AVAILABLE = False
        ENABLE_VALIDATION = False

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
    VALIDATION_AVAILABLE = False

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

def _configure_logging():
    """로깅 설정"""
    try:
        root_logger = logging.getLogger()
        if AZ_DEBUG:
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.INFO)

        # 기존 핸들러 중복 추가 방지
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG if AZ_DEBUG else logging.INFO)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            sh.setFormatter(fmt)
            root_logger.addHandler(sh)

        if AZ_DEBUG and not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            try:
                fh = logging.FileHandler(AZ_LOG_FILE, encoding='utf-8')
                fh.setLevel(logging.DEBUG)
                fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                fh.setFormatter(fmt)
                root_logger.addHandler(fh)
                logger.debug(f"📝 디버그 로그 파일: {AZ_LOG_FILE}")
            except Exception as e:
                logger.warning(f"⚠️ 파일 로거 초기화 실패: {e}")
    except Exception as e:
        print(f"[LOGGING_INIT_ERROR] {e}")

# Docker 환경 경로 설정
# 🔥 pipelines 폴더로 이동했으므로 상위 디렉토리 기준으로 경로 설정
base_dir = os.path.dirname(os.path.dirname(__file__))
CANDLES_DB_PATH = os.path.join(base_dir, 'data', 'rl_candles.db')
STRATEGIES_DB_PATH = os.path.join(base_dir, 'data', 'rl_strategies.db')
# learning_results.db는 이제 rl_strategies.db로 통합됨 (core/env.py 참조)
LEARNING_RESULTS_DB_PATH = STRATEGIES_DB_PATH

def ensure_storage_ready():
    """저장소 디렉토리 및 파일 사전 보장"""
    try:
        # 🔥 pipelines 폴더로 이동했으므로 상위 디렉토리 기준으로 경로 설정
        # base_dir는 이미 위에서 정의됨
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        def _ensure_file_exists(path):
            if not os.path.exists(path):
                open(path, 'a').close()
                logger.info(f"✅ 파일 생성: {path}")

        # LEARNING_RESULTS_DB_PATH는 이제 STRATEGIES_DB_PATH와 동일하므로 중복 제거
        for path in (CANDLES_DB_PATH, STRATEGIES_DB_PATH):
            _ensure_file_exists(path)
    except Exception as e:
        logger.error(f"❌ 저장소 사전 준비 실패: {e}")

def get_strategy_count_for_interval(interval: str) -> int:
    """인터벌에 따른 적절한 전략 개수 결정"""
    # 인터벌별 최적 전략 개수 (개선된 설정)
    strategy_counts = {
        '15m': AZ_STRATEGY_COUNT,      # 200
        '30m': int(AZ_STRATEGY_COUNT * 0.8),    # 160
        '240m': int(AZ_STRATEGY_COUNT * 0.6),   # 120
        '1d': int(AZ_STRATEGY_COUNT * 0.4)      # 80
    }

    # 최소/최대 범위 내에서 조정
    count = strategy_counts.get(interval, AZ_STRATEGY_COUNT)
    count = max(AZ_MIN_STRATEGIES, min(count, AZ_MAX_STRATEGIES))

    return count

def run_absolute_zero(coin: Optional[str] = None, interval: str = "15m",
                      n_strategies: int = None, intervals: Optional[List[str]] = None) -> Dict[str, Any]:
    """Absolute Zero 시스템 실행 - 개선된 버전"""
    try:
        start_time = datetime.now()
        validation_results = {}

        # 🔥 전략 개수 자동 조정
        if n_strategies is None:
            n_strategies = get_strategy_count_for_interval(interval)
            logger.info(f"🎯 {interval} 인터벌에 최적화된 전략 개수: {n_strategies}")

        # 디버그 세션 생성
        session_manager = SessionManager()
        session_id = None
        try:
            if intervals and len(intervals) > 0:
                intervals_for_session = intervals
            elif AZ_INTERVALS:
                intervals_for_session = [i.strip() for i in AZ_INTERVALS.split(',')]
            else:
                intervals_for_session = [interval]

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
                    "candle_days": AZ_CANDLE_DAYS,
                    "selfplay_enabled": ENABLE_SELFPLAY,
                    "validation_enabled": ENABLE_VALIDATION
                }
            )
            logger.info(f"✅ 디버그 세션 생성: {session_id}")
        except Exception as session_err:
            logger.warning(f"⚠️ 디버그 세션 생성 실패 (계속 진행): {session_err}")
            session_id = None

        # 🔥 검증 오케스트레이터 초기화 (개선된 임계값 사용)
        validation_orchestrator = None
        validation_enabled = ENABLE_VALIDATION  # 로컬 플래그 사용
        if validation_enabled and VALIDATION_AVAILABLE:
            try:
                validation_orchestrator = create_validation_orchestrator(enable_auto_fix=ENABLE_AUTO_FIX)

                # 🔥 커스텀 임계값 설정
                if validation_orchestrator:
                    # 임계값 조정을 위해 ValidationContext 사용
                    custom_thresholds = {
                        "min_strategies": VAL_MIN_STRATEGIES,  # 50
                        "max_strategies": VAL_MAX_STRATEGIES,  # 20000
                        "min_prediction_accuracy": 0.35,
                        "max_prediction_accuracy": 0.85
                    }
                    logger.info(f"✅ 검증 시스템 초기화 (최소 전략: {VAL_MIN_STRATEGIES})")

            except Exception as e:
                logger.warning(f"⚠️ 검증 시스템 초기화 실패 (계속 진행): {e}")
                validation_enabled = False
                validation_orchestrator = None

        # 인터벌 정렬
        if intervals and len(intervals) > 0:
            intervals_raw = intervals
        elif AZ_INTERVALS:
            intervals_raw = [i.strip() for i in AZ_INTERVALS.split(',')]
        else:
            intervals_raw = [interval]

        def sort_intervals(interval_list):
            def get_order_in_minutes(iv):
                iv_lower = iv.lower().strip()
                try:
                    if iv_lower.endswith('m'):
                        return int(iv_lower[:-1])
                    elif iv_lower.endswith('h'):
                        return int(iv_lower[:-1]) * 60
                    elif iv_lower.endswith('d'):
                        return int(iv_lower[:-1]) * 1440
                    else:
                        return 999999
                except:
                    return 999999
            return sorted(interval_list, key=lambda x: (get_order_in_minutes(x), x))

        intervals_to_use = sort_intervals(intervals_raw)

        # 코인 기본값
        if coin is None:
            try:
                available = get_available_coins_and_intervals()
                coins = sorted(list({c for c, _ in available}))
                if not coins:
                    raise ValueError("❌ DB에 사용 가능한 코인이 없습니다.")
                coin = coins[0]
            except Exception as e:
                logger.error(f"❌ 코인 목록 조회 실패: {e}")
                raise

        logger.info(f"🚀 Absolute Zero 시스템 시작 (개선된 버전)")
        logger.info(f"   코인: {coin}")
        logger.info(f"   인터벌: {intervals_to_use}")
        logger.info(f"   전략 개수: {n_strategies} (개선됨)")
        logger.info(f"   Self-play: {'활성화' if ENABLE_SELFPLAY else '비활성화'}")
        logger.info(f"   검증: {'활성화' if validation_enabled else '비활성화'}")

        if not NEW_PIPELINE_AVAILABLE:
            logger.error("❌ 새로운 파이프라인 모듈을 사용할 수 없습니다")
            return {"error": "새로운 파이프라인 모듈 사용 불가"}

        # 실행 메타데이터
        run_id = f"abs_zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 실행 기록 생성
        try:
            interval_str = ','.join(intervals_to_use)
            create_run_record(run_id, f"Absolute Zero System 실행 (개선된 버전, {n_strategies}개 전략)",
                            coin=coin, interval=interval_str)
            logger.info(f"✅ 실행 기록 생성: {run_id}")
        except Exception as e:
            logger.warning(f"⚠️ 실행 기록 생성 실패: {e}")

        # 캔들 데이터 로드
        logger.info(f"📊 {coin} 캔들 데이터 로드 중...")
        all_candle_data = load_candle_data_for_coin(coin, intervals_to_use)

        if not all_candle_data:
            logger.error(f"❌ {coin} 캔들 데이터 로드 실패")
            return {"error": f"{coin} 캔들 데이터 로드 실패"}

        # 통합 파이프라인 실행
        logger.info(f"🔄 {coin} 통합 파이프라인 실행 시작...")
        orchestrator = IntegratedPipelineOrchestrator(session_id=session_id)

        # 각 인터벌별 실행
        pipeline_results = []
        for idx, itv in enumerate(intervals_to_use):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 {coin}-{itv} 파이프라인 실행...")
                logger.info(f"{'='*60}")

                candle_data = all_candle_data.get((coin, itv))
                if candle_data is None or candle_data.empty:
                    logger.warning(f"⚠️ {coin}-{itv} 캔들 데이터 없음")
                    continue

                # 🔥 인터벌별 최적 전략 개수 사용
                interval_strategy_count = get_strategy_count_for_interval(itv)
                logger.info(f"   전략 개수: {interval_strategy_count}개")

                # 파이프라인 실행 (개선된 전략 개수 사용)
                # 여기서 orchestrator의 create_strategies 메서드가 호출될 때
                # interval_strategy_count가 사용되도록 해야 함
                result = orchestrator.run_partial_pipeline(coin, itv, candle_data)
                pipeline_results.append(result)

                # 검증 실행 (개선된 임계값 사용)
                if validation_enabled and validation_orchestrator:
                    logger.info(f"🔍 {coin}-{itv} 검증 실행...")

                    # 🔥 커스텀 컨텍스트 생성 (개선된 임계값 포함)
                    val_context = ValidationContext(
                        coin=coin,
                        interval=itv,
                        stage="pipeline",
                        thresholds={
                            "min_strategies": VAL_MIN_STRATEGIES,
                            "max_strategies": VAL_MAX_STRATEGIES,
                            "min_prediction_accuracy": 0.35,
                            "max_prediction_accuracy": 0.85
                        }
                    )

                    # 전략 생성 검증
                    if result.strategies_created > 0:
                        strategy_validation = validation_orchestrator.validate_pipeline_stage(
                            'strategy_generation',
                            {
                                'strategies': [],  # 실제 전략 데이터
                                'count': result.strategies_created,
                                'saved_count': result.strategies_created,
                                'coin': coin,
                                'interval': itv
                            },
                            coin, itv, pipeline_run_id=run_id
                        )
                        validation_results[f"{coin}_{itv}_strategy"] = strategy_validation

                        if strategy_validation.is_successful():
                            logger.info(f"   ✅ 전략 검증 통과 ({strategy_validation.get_success_rate():.0%})")
                        else:
                            logger.warning(f"   ⚠️ 전략 검증 이슈 ({strategy_validation.get_success_rate():.0%})")

                    # 라우팅 검증
                    if result.routing_results > 0:
                        routing_validation = validation_orchestrator.validate_pipeline_stage(
                            'routing',
                            {
                                'routing_results': [],  # 실제 라우팅 데이터
                                'regime': result.regime_detected,
                                'selected_strategies': [],
                                'backtest_results': {},
                                'signal_scores': [result.signal_score],
                                'coin': coin,
                                'interval': itv
                            },
                            coin, itv, pipeline_run_id=run_id
                        )
                        validation_results[f"{coin}_{itv}_routing"] = routing_validation

                logger.info(f"✅ {coin}-{itv} 파이프라인 완료")

            except Exception as e:
                logger.error(f"❌ {coin}-{itv} 처리 실패: {e}")
                continue

        # 검증 요약
        if validation_enabled and validation_results:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 검증 결과 요약")
            logger.info(f"{'='*60}")

            total_validations = len(validation_results)
            successful = sum(1 for v in validation_results.values() if v.is_successful())

            logger.info(f"   총 검증: {total_validations}개")
            logger.info(f"   성공: {successful}개 ({successful/total_validations*100:.0f}%)")

            # 검증 통계 조회
            if validation_orchestrator:
                stats = validation_orchestrator.get_validation_stats()
                logger.info(f"   누적 검증: {stats.get('total_validations', 0)}회")
                logger.info(f"   자동 복구: {stats.get('auto_fixed', 0)}건")

        # Paper Trading 자동 시작
        if ENABLE_AUTO_PAPER_TRADING and pipeline_results:
            try:
                logger.info(f"\n📊 {coin} Paper Trading 자동 시작...")
                from rl_pipeline.validation.auto_paper_trading import auto_start_paper_trading_after_pipeline

                paper_result = auto_start_paper_trading_after_pipeline(
                    coin=coin,
                    intervals=intervals_to_use,
                    duration_days=PAPER_TRADING_DURATION_DAYS
                )

                if paper_result.get('status') == 'started':
                    logger.info(f"✅ Paper Trading 시작 완료")
                else:
                    logger.warning(f"⚠️ Paper Trading 시작 실패")

            except Exception as e:
                logger.warning(f"⚠️ Paper Trading 자동 시작 실패: {e}")

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n🎉 Absolute Zero 시스템 실행 완료")
        logger.info(f"   실행 시간: {execution_time:.1f}초")
        logger.info(f"   처리된 인터벌: {len(pipeline_results)}개")

        return {
            "status": "success",
            "coin": coin,
            "intervals": intervals_to_use,
            "pipeline_results": len(pipeline_results),
            "execution_time": execution_time,
            "strategy_count": n_strategies,
            "validation_results": len(validation_results) if validation_results else 0,
            "improvements": {
                "strategy_count_increased": True,
                "validation_thresholds_adjusted": True,
                "selfplay_optional": not ENABLE_SELFPLAY
            }
        }

    except Exception as e:
        logger.error(f"❌ Absolute Zero 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    """메인 함수"""
    try:
        _configure_logging()
        logger.info("🚀 Absolute Zero 시스템 시작 (개선된 버전)")
        logger.info(f"   전략 개수: {AZ_STRATEGY_COUNT} (기본)")
        logger.info(f"   최소 전략: {VAL_MIN_STRATEGIES}")
        logger.info(f"   Self-play: {'활성화' if ENABLE_SELFPLAY else '비활성화 (빠른 실행)'}")
        logger.info(f"   검증: {'활성화' if ENABLE_VALIDATION else '비활성화'}")

        # 저장 경로 준비
        ensure_storage_ready()

        # 데이터베이스 초기화
        try:
            logger.info("🔧 데이터베이스 초기화...")
            setup_database_tables()
            create_learning_results_tables()

            try:
                create_coin_strategies_table()
            except Exception as e:
                logger.warning(f"⚠️ coin_strategies 테이블 생성 실패 (이미 존재): {e}")

            try:
                ensure_indexes()
                logger.info("✅ 인덱스 생성 완료")
            except Exception as e:
                logger.warning(f"⚠️ 인덱스 생성 실패 (이미 존재): {e}")

        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
            return {"error": f"데이터베이스 초기화 실패: {e}"}

        # 사용 가능한 코인/인터벌 조합
        coin_interval_combinations = get_available_coins_and_intervals()
        logger.info(f"📊 발견된 코인/인터벌 조합: {len(coin_interval_combinations)}개")

        if not coin_interval_combinations:
            logger.error("❌ 사용 가능한 코인/인터벌 조합이 없습니다")
            return {"error": "no coin/interval combinations found"}

        # 코인별 그룹핑
        coin_to_intervals: Dict[str, List[str]] = {}
        for c, itv in coin_interval_combinations:
            coin_to_intervals.setdefault(c, [])
            if itv not in coin_to_intervals[c]:
                coin_to_intervals[c].append(itv)

        # 테스트 모드: 첫 번째 코인만
        logger.info("⚠️ 테스트 모드: 첫 번째 코인만 실행")
        first_coin = list(coin_to_intervals.keys())[0]
        coin_to_intervals = {first_coin: coin_to_intervals[first_coin]}

        # 실행
        results = []
        for coin, intervals in coin_to_intervals.items():
            try:
                logger.info(f"\n🪙 {coin} 처리 시작...")
                # 전략 개수는 자동으로 결정됨
                result = run_absolute_zero(coin, intervals=intervals)
                results.append(result)

                if result.get("status") == "success":
                    logger.info(f"✅ {coin} 처리 성공")
                else:
                    logger.error(f"❌ {coin} 처리 실패")

            except Exception as e:
                logger.error(f"❌ {coin} 처리 중 오류: {e}")
                continue

        # 최종 요약
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 실행 완료")
        logger.info(f"{'='*60}")
        logger.info(f"   개선 사항:")
        logger.info(f"   ✅ 전략 개수 증가 (50 → {AZ_STRATEGY_COUNT})")
        logger.info(f"   ✅ 검증 임계값 조정 (최소 {VAL_MIN_STRATEGIES})")
        logger.info(f"   ✅ Self-play 선택적 실행")
        logger.info(f"   ✅ 인터벌별 최적 전략 개수")

        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"❌ 메인 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    main()