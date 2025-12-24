"""
Absolute Zero 시스템 - 통합 오케스트레이터 (검증 시스템 통합 버전)
모든 파이프라인 기능을 통합한 단일 시스템 + 데이터 검증
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

# TensorFlow Protobuf 버전 경고 숨김 (JAX 로드 시 발생하는 경고, 기능 영향 없음)
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Sharding info not provided.*', category=UserWarning)

# JAX TPU/ROCm 백엔드 방지 및 CUDA 강제 사용
import os
# TensorFlow 경고 완전 억제 (JAX가 TensorFlow 없이도 작동 가능)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# CUDA만 사용하도록 명시 (ROCm 제외)
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')  # CUDA만 사용, ROCm 제외

# Python warnings 필터링 (TensorFlow 관련)
warnings.filterwarnings('ignore', category=Warning, message='.*Tensorflow.*')
warnings.filterwarnings('ignore', category=Warning, message='.*TensorFlow.*')

# JAX 로거 레벨 조정 (TensorFlow 경고 억제)
import logging as std_logging
std_logging.getLogger('jax._src.xla_bridge').setLevel(std_logging.ERROR)
std_logging.getLogger('jax._src.lib').setLevel(std_logging.ERROR)
std_logging.getLogger('absl').setLevel(std_logging.ERROR)

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 🔥 logger 초기화 (import 전에 설정)
logger = logging.getLogger(__name__)

# 새로운 파이프라인 구조 import
try:
    # 직접 import 시도
    import rl_pipeline.core.env as core_env
    import rl_pipeline.core.errors as core_errors
    import rl_pipeline.strategy.manager as strategy_manager
    import rl_pipeline.simulation.selfplay as selfplay
    import rl_pipeline.routing.regime_router as regime_router
    import rl_pipeline.analysis.integrated_analyzer as integrated_analyzer
    import rl_pipeline.db.schema as db_schema
    import rl_pipeline.db.connection_pool as db_pool

    # 🔥 디버그 시스템 import
    from rl_pipeline.monitoring import SessionManager

    # 🆕 검증 시스템 import
    from rl_pipeline.validation import (
        create_validation_orchestrator,
        validate_absolute_zero_stage
    )

    config = core_env.config
    AZError = core_errors.AZError
    create_run_record = strategy_manager.create_run_record
    update_run_record = strategy_manager.update_run_record
    create_strategies = strategy_manager.create_strategies
    create_global_strategies = strategy_manager.create_global_strategies
    run_self_play_test = selfplay.run_self_play_test
    RegimeRouter = regime_router.RegimeRouter
    create_regime_routing_strategies = regime_router.create_regime_routing_strategies
    IntegratedAnalyzer = integrated_analyzer.IntegratedAnalyzer
    analyze_strategies = integrated_analyzer.analyze_strategies
    analyze_global_strategies = integrated_analyzer.analyze_global_strategies
    ensure_indexes = db_schema.ensure_indexes
    setup_database_tables = db_schema.setup_database_tables
    create_strategies_table = db_schema.create_strategies_table
    get_optimized_db_connection = db_pool.get_optimized_db_connection

    NEW_PIPELINE_AVAILABLE = True

except ImportError as e:
    logger.error(f"새로운 파이프라인 모듈 import 실패: {e}")
    config = None
    AZError = Exception
    NEW_PIPELINE_AVAILABLE = False

# 환경변수 파일 로드
from dotenv import load_dotenv
# 🔥 pipelines 폴더로 이동했으므로 상위 디렉토리에서 설정 파일 찾기
base_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(base_dir, 'rl_pipeline_config.env')
load_dotenv(env_path)

# 환경 변수
AZ_DEBUG = os.getenv('AZ_DEBUG', 'false').lower() == 'true'
# 🔥 pipelines 폴더로 이동했으므로 상위 디렉토리 기준으로 로그 파일 경로 설정
# base_dir는 이미 위에서 정의됨
AZ_LOG_FILE = os.getenv('AZ_LOG_FILE', os.path.join(base_dir, 'absolute_zero_debug.log'))
AZ_SIMULATION_VERBOSE = os.getenv('AZ_SIMULATION_VERBOSE', 'false').lower() == 'true'
AZ_CANDLE_DAYS = int(os.getenv('AZ_CANDLE_DAYS', '30'))
AZ_INTERVALS = os.getenv('AZ_INTERVALS', '')

# 🆕 검증 시스템 환경변수
ENABLE_VALIDATION = os.getenv('ENABLE_VALIDATION', 'true').lower() == 'true'
ENABLE_AUTO_FIX = os.getenv('ENABLE_AUTO_FIX', 'true').lower() == 'true'

# 🆕 전역 검증 오케스트레이터 생성
validation_orchestrator = None
if ENABLE_VALIDATION:
    try:
        validation_orchestrator = create_validation_orchestrator(enable_auto_fix=ENABLE_AUTO_FIX)
        logger.info("✅ 검증 시스템 초기화 완료")
    except Exception as e:
        logger.warning(f"⚠️ 검증 시스템 초기화 실패 (계속 진행): {e}")
        ENABLE_VALIDATION = False

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
# base_dir는 이미 위에서 정의됨
CANDLES_DB_PATH = os.path.join(base_dir, 'data', 'rl_candles.db')
STRATEGIES_DB_PATH = os.path.join(base_dir, 'data', 'learning_strategies.db')
# learning_results.db는 이제 learning_strategies.db로 통합됨 (core/env.py 참조)
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

def run_absolute_zero(coin: Optional[str] = None, interval: str = "15m", n_strategies: int = 300, intervals: Optional[List[str]] = None) -> Dict[str, Any]:
    """Absolute Zero 시스템 실행 - 검증 시스템 통합"""
    try:
        start_time = datetime.now()
        validation_results = {}  # 🆕 검증 결과 저장

        # 🔥 디버그 세션 생성
        session_manager = SessionManager()
        session_id = None
        try:
            # 인터벌 리스트 미리 준비 (세션 생성용)
            if intervals and len(intervals) > 0:
                intervals_for_session = intervals
            elif AZ_INTERVALS:
                intervals_for_session = [i.strip() for i in AZ_INTERVALS.split(',')]
            else:
                intervals_for_session = [interval]

            # 코인 결정 (세션 생성용)
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

        # 다중 인터벌 지원 우선순위: 함수 인자 intervals > AZ_INTERVALS 환경변수 > 단일 interval
        if intervals and len(intervals) > 0:
            intervals_raw = intervals
        elif AZ_INTERVALS:
            intervals_raw = [i.strip() for i in AZ_INTERVALS.split(',')]
        else:
            intervals_raw = [interval]

        # 인터벌 순서 정렬
        def sort_intervals(interval_list):
            """인터벌을 시간 순서로 정렬"""
            def get_order_in_minutes(iv):
                iv_lower = iv.lower().strip()
                try:
                    if iv_lower.endswith('m'):
                        minutes = int(iv_lower[:-1])
                        return minutes
                    elif iv_lower.endswith('h'):
                        hours = int(iv_lower[:-1])
                        return hours * 60
                    elif iv_lower.endswith('d'):
                        days = int(iv_lower[:-1])
                        return days * 1440
                    else:
                        return 999999
                except (ValueError, AttributeError):
                    return 999999

            return sorted(interval_list, key=lambda x: (get_order_in_minutes(x), x))

        intervals_to_use = sort_intervals(intervals_raw)

        # 코인 기본값: DB에서 사용 가능한 코인 목록 우선 사용
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
        logger.info(f"🔍 데이터 검증 시스템: {'활성화' if ENABLE_VALIDATION else '비활성화'}")

        if not NEW_PIPELINE_AVAILABLE:
            logger.error("❌ 새로운 파이프라인 모듈을 사용할 수 없습니다")
            return {"error": "새로운 파이프라인 모듈 사용 불가"}

        # 실행 메타데이터 생성
        run_id = f"abs_zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_span = f"{datetime.now().strftime('%Y-%m-%d')}"

        # 실행 기록 생성
        try:
            interval_str = intervals_to_use[0] if intervals_to_use else interval
            if len(intervals_to_use) > 1:
                interval_str = ','.join(intervals_to_use)
            create_run_record(run_id, "Absolute Zero System 실행", coin=coin, interval=interval_str)
            logger.info(f"✅ 실행 기록 생성 완료: {run_id} (coin={coin}, intervals={interval_str})")
        except Exception as e:
            logger.warning(f"⚠️ 실행 기록 생성 실패: {e}")

        # 캔들 데이터 로드
        logger.info(f"📊 {coin} 캔들 데이터 로드 시작 (목표: {AZ_CANDLE_DAYS}일)...")
        all_candle_data = load_candle_data_for_coin(coin, intervals_to_use)

        if not all_candle_data:
            logger.error(f"❌ {coin} 캔들 데이터 로드 실패")
            return {"error": f"{coin} 캔들 데이터 로드 실패"}

        # 데이터 충분성 체크
        total_candles = sum(len(df) for df in all_candle_data.values())
        if total_candles == 0:
            logger.error(f"❌ {coin}: 사용 가능한 캔들 데이터가 없습니다")
            return {"error": f"{coin}: 캔들 데이터 없음"}

        # 통합 파이프라인 실행
        logger.info(f"🔄 {coin} 통합 파이프라인 실행 시작...")

        # 파이프라인 오케스트레이터 초기화
        orchestrator = IntegratedPipelineOrchestrator(session_id=session_id)

        # 각 인터벌별로 통합 파이프라인 실행
        pipeline_results = []
        for idx, interval in enumerate(intervals_to_use):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 {coin}-{interval} 통합 파이프라인 실행...")
                logger.info(f"{'='*60}")

                candle_data = all_candle_data.get((coin, interval))
                if candle_data is None or candle_data.empty:
                    logger.warning(f"⚠️ {coin}-{interval} 캔들 데이터 없음, 건너뜀")
                    continue

                # 1-3단계 실행: 전략생성 → Self-play → 레짐라우팅
                result = orchestrator.run_partial_pipeline(coin, interval, candle_data)
                pipeline_results.append(result)

                # 🆕 검증 시스템 실행
                if ENABLE_VALIDATION and validation_orchestrator:
                    logger.info(f"\n🔍 {coin}-{interval} 파이프라인 결과 검증 시작...")

                    # 1. 전략 생성 검증
                    if result.strategies_created > 0:
                        strategy_validation = validation_orchestrator.validate_pipeline_stage(
                            'strategy_generation',
                            {
                                'strategies': getattr(result, 'strategies', []),
                                'count': result.strategies_created,
                                'saved_count': result.strategies_created,
                                'coin': coin,
                                'interval': interval
                            },
                            coin, interval, pipeline_run_id=run_id
                        )
                        validation_results[f"{coin}_{interval}_strategy"] = strategy_validation

                        if not strategy_validation.is_successful():
                            logger.warning(f"⚠️ 전략 생성 검증 이슈: {strategy_validation.get_success_rate():.1%} 성공률")
                            if strategy_validation.has_critical_issues():
                                logger.error(f"❌ Critical 이슈 발견!")

                    # 2. Self-play 검증
                    if result.selfplay_result and result.selfplay_episodes > 0:
                        selfplay_validation = validation_orchestrator.validate_pipeline_stage(
                            'selfplay',
                            {
                                'episodes': result.selfplay_result.get('episodes', []),
                                'total_episodes': result.selfplay_episodes,
                                'evolved_strategies': result.selfplay_result.get('evolved_strategies', []),
                                'prediction_accuracy': result.selfplay_result.get('prediction_accuracy', 0),
                                'average_return': result.selfplay_result.get('average_return', 0),
                                'win_rate': result.selfplay_result.get('win_rate', 0),
                                'coin': coin,
                                'interval': interval
                            },
                            coin, interval, pipeline_run_id=run_id
                        )
                        validation_results[f"{coin}_{interval}_selfplay"] = selfplay_validation

                        if not selfplay_validation.is_successful():
                            logger.warning(f"⚠️ Self-play 검증 이슈: {selfplay_validation.get_success_rate():.1%} 성공률")

                    # 3. 라우팅 검증
                    if result.routing_results > 0:
                        routing_validation = validation_orchestrator.validate_pipeline_stage(
                            'routing',
                            {
                                'routing_results': getattr(result, 'routing_data', []),
                                'regime': result.regime_detected,
                                'selected_strategies': getattr(result, 'selected_strategies', []),
                                'backtest_results': getattr(result, 'backtest_results', {}),
                                'signal_scores': [result.signal_score],
                                'coin': coin,
                                'interval': interval
                            },
                            coin, interval, pipeline_run_id=run_id
                        )
                        validation_results[f"{coin}_{interval}_routing"] = routing_validation

                        if not routing_validation.is_successful():
                            logger.warning(f"⚠️ 라우팅 검증 이슈: {routing_validation.get_success_rate():.1%} 성공률")

                logger.info(f"✅ {coin}-{interval} 파이프라인 완료: 레짐라우팅까지 완료")

            except Exception as e:
                logger.error(f"❌ {coin}-{interval} 처리 중 오류: {e}")
                continue

        # 🆕 전체 검증 요약
        if ENABLE_VALIDATION and validation_orchestrator and validation_results:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 검증 결과 요약 - {coin}")
            logger.info(f"{'='*60}")

            total_checks = 0
            total_passed = 0
            critical_issues = []

            for key, val_result in validation_results.items():
                total_checks += val_result.total_checks
                total_passed += val_result.passed_checks

                logger.info(f"  {key}: {val_result.overall_status.value} "
                          f"({val_result.get_success_rate():.1%} 성공률)")

                if val_result.has_critical_issues():
                    critical_issues.append(key)

            overall_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0
            logger.info(f"\n  전체 성공률: {overall_rate:.1f}%")
            logger.info(f"  총 검증: {total_checks}개 (✅ {total_passed}개)")

            if critical_issues:
                logger.warning(f"  ⚠️ Critical 이슈 발견: {', '.join(critical_issues)}")

            # 검증 통계 조회
            val_stats = validation_orchestrator.get_validation_stats()
            logger.info(f"\n  누적 검증 통계:")
            logger.info(f"    - 총 검증: {val_stats['total_validations']}회")
            logger.info(f"    - 성공: {val_stats['successful_validations']}회")
            logger.info(f"    - 자동 복구: {val_stats['auto_fixed']}회")

        # 글로벌 전략 생성 (필요시)
        # ... (기존 코드)

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n🎉 Absolute Zero 시스템 실행 완료")
        logger.info(f"⏱️ 실행 시간: {execution_time:.1f}초")

        return {
            "status": "success",
            "coin": coin,
            "intervals": intervals_to_use,
            "pipeline_results": len(pipeline_results),
            "execution_time": execution_time,
            "validation_enabled": ENABLE_VALIDATION,
            "validation_summary": {
                "total_validations": len(validation_results),
                "overall_success_rate": overall_rate if validation_results else 100.0,
                "critical_issues": len(critical_issues) if validation_results else 0
            } if ENABLE_VALIDATION else None
        }

    except Exception as e:
        logger.error(f"❌ Absolute Zero 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    """메인 함수 - 모든 코인/인터벌 조합에 대해 실행"""
    try:
        _configure_logging()
        logger.info("🚀 Absolute Zero 시스템 메인 실행 시작 (검증 시스템 활성화)")

        # 저장 경로 및 DB 파일 사전 보장
        ensure_storage_ready()

        # 🆕 시스템 시작 시 한 번만 데이터베이스 초기화
        try:
            logger.info("🔧 시스템 데이터베이스 초기화 시작...")
            setup_database_tables()
            create_learning_results_tables()

            # 필수 테이블 보강 생성 (방어적)
            try:
                create_strategies_table()
            except Exception as se:
                logger.warning(f"⚠️ strategies 보강 생성 실패(무시 가능): {se}")

            # 인덱스 생성 (한 번만)
            try:
                logger.info("🔧 인덱스 생성 시작...")
                ensure_indexes()
                logger.info("✅ 인덱스 생성 완료")
            except Exception as ie:
                logger.warning(f"⚠️ 인덱스 생성 실패(무시 가능): {ie}")

            logger.info("✅ 시스템 데이터베이스 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 시스템 데이터베이스 초기화 실패: {e}")
            return {"error": f"데이터베이스 초기화 실패: {e}"}

        # 🆕 사용 가능한 코인/인터벌 조합 가져오기
        coin_interval_combinations = get_available_coins_and_intervals()
        logger.info(f"📊 발견된 코인/인터벌 조합: {len(coin_interval_combinations)}개")

        # 코인별 전체 인터벌로 그룹핑
        coin_to_intervals: Dict[str, List[str]] = {}
        for c, itv in coin_interval_combinations:
            coin_to_intervals.setdefault(c, [])
            if itv not in coin_to_intervals[c]:
                coin_to_intervals[c].append(itv)

        # 인터벌 정렬
        def sort_intervals_for_main(interval_list):
            """인터벌을 시간 순서로 정렬"""
            def get_order_in_minutes(iv):
                iv_lower = iv.lower().strip()
                try:
                    if iv_lower.endswith('m'):
                        minutes = int(iv_lower[:-1])
                        return minutes
                    elif iv_lower.endswith('h'):
                        hours = int(iv_lower[:-1])
                        return hours * 60
                    elif iv_lower.endswith('d'):
                        days = int(iv_lower[:-1])
                        return days * 1440
                    else:
                        return 999999
                except (ValueError, AttributeError):
                    return 999999

            return sorted(interval_list, key=lambda x: (get_order_in_minutes(x), x))

        for c in coin_to_intervals:
            try:
                coin_to_intervals[c] = sort_intervals_for_main(coin_to_intervals[c])
            except Exception:
                pass

        if not coin_interval_combinations:
            logger.error("❌ 사용 가능한 코인/인터벌 조합이 없습니다.")
            logger.error("❌ 캔들 데이터를 먼저 수집하세요: python candles_collector.py")
            return {"error": "no coin/interval combinations found", "message": "캔들 데이터를 먼저 수집하세요"}

        # 테스트를 위해 첫 번째 코인만 실행 (전체 실행 원하면 이 부분 제거)
        logger.info("⚠️ 테스트 모드: 첫 번째 코인만 실행합니다.")
        first_coin = list(coin_to_intervals.keys())[0] if coin_to_intervals else None
        if first_coin:
            coin_to_intervals = {first_coin: coin_to_intervals[first_coin]}

        # 각 조합에 대해 실행
        results = []
        failed_runs = []

        for coin, intervals in coin_to_intervals.items():
            try:
                logger.info(f"\n🪙 {coin} {', '.join(intervals)} 처리 시작")
                result = run_absolute_zero(coin, interval=intervals[0], n_strategies=200, intervals=intervals)
                results.append(result)

                if result.get("status") == "success":
                    logger.info(f"✅ {coin} 처리 성공")
                else:
                    logger.error(f"❌ {coin} 처리 실패: {result.get('message', 'Unknown error')}")
                    failed_runs.append(f"{coin}_{','.join(intervals)}")

            except Exception as e:
                logger.error(f"❌ {coin} 처리 중 오류: {e}")
                failed_runs.append(f"{coin}_{','.join(intervals)}")
                continue

        # 결과 요약
        successful_runs = len([r for r in results if r.get("status") == "success"])
        total_runs = len(coin_to_intervals)

        logger.info(f"\n🎉 Absolute Zero 시스템 실행 완료")
        logger.info(f"📊 총 실행: {total_runs}개, 성공: {successful_runs}개, 실패: {len(failed_runs)}개")

        if failed_runs:
            logger.warning(f"⚠️ 실패한 조합: {failed_runs}")

        # 🆕 검증 시스템 최종 리포트
        if ENABLE_VALIDATION and validation_orchestrator:
            logger.info("\n📊 검증 시스템 최종 리포트:")
            final_report = validation_orchestrator.generate_report()

            if 'total_validations' in final_report:
                logger.info(f"  - 오늘 총 검증: {final_report['total_validations']}회")
                logger.info(f"  - 평균 성공률: {final_report.get('average_success_rate', 0):.1%}")
                logger.info(f"  - Critical 이슈: {final_report.get('critical_issues', 0)}건")
                logger.info(f"  - 자동 복구: {final_report.get('auto_fixed_issues', 0)}건")

            # 신뢰도 현황
            trust_stats = validation_orchestrator.trust_manager.get_global_stats()
            logger.info(f"\n  시스템 건강도: {trust_stats.get('system_health', 'Unknown')}")

            if 'problematic_components' in trust_stats and trust_stats['problematic_components']:
                logger.warning(f"  ⚠️ 문제 컴포넌트: {trust_stats['problematic_components']}")

        return {
            "status": "success",
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": len(failed_runs)
        }

    except Exception as e:
        logger.error(f"❌ 메인 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    main()