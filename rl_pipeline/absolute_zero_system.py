"""
Absolute Zero 시스템 - 통합 오케스트레이터
모든 파이프라인 기능을 통합한 단일 시스템
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

    config = core_env.config
    AZError = core_errors.AZError
    create_run_record = strategy_manager.create_run_record
    update_run_record = strategy_manager.update_run_record
    create_global_strategies = strategy_manager.create_global_strategies
    run_self_play_test = selfplay.run_self_play_test
    RegimeRouter = regime_router.RegimeRouter
    create_regime_routing_strategies = regime_router.create_regime_routing_strategies
    IntegratedAnalyzer = integrated_analyzer.IntegratedAnalyzer
    analyze_global_strategies = integrated_analyzer.analyze_global_strategies
    ensure_indexes = db_schema.ensure_indexes
    setup_database_tables = db_schema.setup_database_tables
    create_coin_strategies_table = db_schema.create_strategies_table
    get_optimized_db_connection = db_pool.get_optimized_db_connection
    
    NEW_PIPELINE_AVAILABLE = True
    # 🔥 import 성공 시 로그 제거 (정상 동작이므로 불필요)
    
except ImportError as e:
    # 🔥 import 실패 시에만 로그 출력 (logger 사용)
    logger.error(f"새로운 파이프라인 모듈 import 실패: {e}")
    # 기본값으로 설정
    config = None
    AZError = Exception
    NEW_PIPELINE_AVAILABLE = False
AZ_DEBUG = os.getenv('AZ_DEBUG', 'false').lower() == 'true'
AZ_LOG_FILE = os.getenv('AZ_LOG_FILE', os.path.join(os.path.dirname(__file__), 'absolute_zero_debug.log'))
AZ_SIMULATION_VERBOSE = os.getenv('AZ_SIMULATION_VERBOSE', 'false').lower() == 'true'

# 환경변수 설명:
# AZ_DEBUG=true: 모든 DEBUG 로그 출력 (매우 상세)
# AZ_SIMULATION_VERBOSE=true: 시뮬레이션 상세 로그 출력 (전략별 RSI/MACD/Volume 로그)

# ============================================================================
# 통합된 파이프라인 구조
# ============================================================================

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

# ============================================================================
# 🔧 공통 유틸리티 함수 (중복 제거)
# ============================================================================

def get_interval_minutes(interval: str) -> int:
    """인터벌을 분 단위로 변환 (정렬/비교용)
    
    Args:
        interval: 인터벌 문자열 (예: '15m', '1h', '1d')
        
    Returns:
        분 단위 값 (파싱 실패 시 999999)
    """
    iv_lower = interval.lower().strip()
    try:
        if iv_lower.endswith('m'):
            return int(iv_lower[:-1])
        elif iv_lower.endswith('h'):
            return int(iv_lower[:-1]) * 60
        elif iv_lower.endswith('d'):
            return int(iv_lower[:-1]) * 1440
        elif iv_lower.endswith('w'):
            return int(iv_lower[:-1]) * 10080
        else:
            return 999999
    except (ValueError, AttributeError):
        return 999999


def sort_intervals(interval_list: List[str]) -> List[str]:
    """인터벌을 시간 순서로 정렬 (단기 → 장기)
    
    Args:
        interval_list: 인터벌 리스트
        
    Returns:
        정렬된 인터벌 리스트
    """
    return sorted(interval_list, key=lambda x: (get_interval_minutes(x), x))

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

# 환경변수 파일 로드
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), 'config/rl_pipeline_config.env')
load_dotenv(env_path)

# 🔥 동적 경로 설정 (하드코딩 제거 - 엔진화)
# 현재 파일 기준으로 경로 추론
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))  # rl_pipeline/
_AUTO_TRADER_ROOT_INFERRED = os.path.dirname(_CURRENT_FILE_DIR)  # auto_trader/

# 환경변수 우선, 없으면 동적 추론
WORKSPACE_ROOT = os.getenv('WORKSPACE_ROOT', _AUTO_TRADER_ROOT_INFERRED)
AUTO_TRADER_ROOT = os.getenv('AUTO_TRADER_ROOT', _AUTO_TRADER_ROOT_INFERRED)
RL_PIPELINE_ROOT = os.getenv('RL_PIPELINE_ROOT', _CURRENT_FILE_DIR)

# DATA_STORAGE_PATH 동적 추론 (컨텍스트 인식)
# 🔥 run_learning.py / run_trading.py에서 설정한 환경변수 우선 사용
# 1. 명시적 환경변수 (최우선)
# 2. 전략 DB 경로의 상위 디렉토리
# 3. 기본값은 사용하지 않음 (환경변수가 없으면 에러)
_strategy_db_env = os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
if _strategy_db_env:
    _inferred_storage = os.path.dirname(_strategy_db_env)
else:
    _inferred_storage = None

# 환경변수가 없으면 에러 (run_learning.py / run_trading.py에서 설정해야 함)
DATA_STORAGE_PATH = os.getenv('DATA_STORAGE_PATH')
if not DATA_STORAGE_PATH:
    if _inferred_storage:
        DATA_STORAGE_PATH = _inferred_storage
    else:
        # 환경변수가 전혀 없으면 현재 작업 디렉토리 기준으로 추론 시도
        import warnings
        _cwd_storage = os.path.join(os.getcwd(), 'data_storage')
        warnings.warn(
            f"⚠️ DATA_STORAGE_PATH 환경변수가 설정되지 않았습니다. "
            f"run_learning.py 또는 run_trading.py에서 설정해야 합니다. "
            f"임시로 {_cwd_storage} 사용합니다.",
            UserWarning
        )
        DATA_STORAGE_PATH = _cwd_storage

# 실행 규모/범위 설정 (환경변수로 제어)
AZ_INTERVALS = os.getenv('AZ_INTERVALS')  # 예: "15m,30m,240m,1d"
AZ_CANDLE_DAYS = int(os.getenv('AZ_CANDLE_DAYS', '60'))  # 기본 60일 (신생 코인은 가용 데이터만큼 사용)
AZ_ALLOW_FALLBACK = os.getenv('AZ_ALLOW_FALLBACK', 'false').lower() == 'true'
AZ_FALLBACK_PAIRS = os.getenv('AZ_FALLBACK_PAIRS', '')  # 예: "BTC:15m;ETH:15m" (가능하면 DB에서 코인/인터벌 자동 탐색)

# 🔥 자동 재학습 강제 비활성화 (속도 개선) - 환경변수로 제어

# Self-play 및 전략 풀 설정 (환경변수로 제어)
AZ_SELFPLAY_EPISODES = int(os.getenv('AZ_SELFPLAY_EPISODES', '200'))  # Self-play 에피소드 수
AZ_SELFPLAY_AGENTS_PER_EPISODE = int(os.getenv('AZ_SELFPLAY_AGENTS_PER_EPISODE', '4'))  # 에피소드당 에이전트 수
AZ_STRATEGY_POOL_SIZE = int(os.getenv('AZ_STRATEGY_POOL_SIZE', '15000'))  # DB에서 로드할 최대 전략 수

# 🆕 점진적 통합: 예측 실현 Self-play 비율 (0.0-1.0)
PREDICTIVE_SELFPLAY_RATIO = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))  # 기본값: 20%

# 데이터베이스 경로 설정
# 환경 변수 RL_DB_PATH, STRATEGY_DB_PATH가 설정되어 있으면 최우선 사용 (run_learning.py 등에서 설정함)
try:
    DEFAULT_RL_DB = config.RL_DB
    DEFAULT_STRATEGIES_DB = config.STRATEGIES_DB
except (ImportError, AttributeError):
    # 환경변수가 없을 경우를 대비한 기본값은 제거하고, 환경변수 설정을 강제함
    DEFAULT_RL_DB = None
    DEFAULT_STRATEGIES_DB = None

CANDLES_DB_PATH = os.getenv('RL_DB_PATH', DEFAULT_RL_DB)
STRATEGIES_DB_PATH = os.getenv('STRATEGY_DB_PATH', DEFAULT_STRATEGIES_DB)

# 🔥 강제 보정: rl_strategies.db가 경로에 포함되어 있으면 learning_strategies.db로 교체 (레거시 호환성)
if STRATEGIES_DB_PATH and 'rl_strategies.db' in STRATEGIES_DB_PATH:
    STRATEGIES_DB_PATH = STRATEGIES_DB_PATH.replace('rl_strategies.db', 'learning_strategies.db')

if not CANDLES_DB_PATH or not STRATEGIES_DB_PATH:
    # 필수 환경변수 미설정 시 에러 발생 (하드코딩 방지)
    error_msg = "❌ RL_DB_PATH 또는 STRATEGY_DB_PATH 환경변수가 설정되지 않았습니다. run_learning.py 등에서 설정해주세요."
    logger.error(error_msg)
    raise ValueError(error_msg)
# LEARNING_RESULTS_DB_PATH는 config에서 가져옴 (동적 처리: 파일 or 디렉토리/common.db)
try:
    LEARNING_RESULTS_DB_PATH = config.LEARNING_RESULTS_DB_PATH
except:
    LEARNING_RESULTS_DB_PATH = STRATEGIES_DB_PATH

logger.info(f"📂 캔들 DB 경로: {CANDLES_DB_PATH}")
logger.info(f"📂 전략 DB 경로: {STRATEGIES_DB_PATH}")
logger.info(f"📂 학습 결과 DB 경로: {LEARNING_RESULTS_DB_PATH}")

def _ensure_file_exists(db_path: str) -> None:
    """DB 파일이 없으면 생성 (원천 데이터 DB는 제외)
    
    Note: db_path가 디렉토리인 경우, 해당 디렉토리가 존재하는지만 확인하고 종료
    """
    try:
        # 디렉토리인 경우 (확장자 검사 또는 isdir 검사)
        is_directory = not db_path.endswith('.db')
        if is_directory:
            if not os.path.exists(db_path):
                os.makedirs(db_path, exist_ok=True)
                logger.info(f"📂 전략 DB 디렉토리 생성: {db_path}")
            return

        parent = os.path.dirname(db_path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
                logger.debug(f"📂 DB 디렉토리 생성: {parent}")
            except Exception as dir_err:
                # 디렉토리 생성 실패는 경고로 처리 (이미 ensure_storage_ready에서 생성했을 수 있음)
                logger.debug(f"⚠️ DB 디렉토리 생성 시도 (무시 가능): {parent} - {dir_err}")
        
        # rl_candles.db는 원천 데이터 - 절대 생성하거나 수정하면 안됨
        if 'candles' in db_path and not os.path.exists(db_path):
            logger.debug(f"⚠️ 원천 데이터 DB가 없습니다: {db_path} (생성하지 않음)")
            return
            
        if not os.path.exists(db_path):
            # 빈 SQLite 파일 생성 (rl_candles.db 제외)
            try:
                conn = sqlite3.connect(db_path)
                conn.close()
                logger.info(f"🗃️ DB 파일 생성: {db_path}")
            except Exception as create_err:
                # DB 파일 생성 실패는 경고로 처리 (connection_pool에서 자동 생성할 수 있음)
                logger.debug(f"⚠️ DB 파일 생성 시도 실패 (무시 가능, 연결 풀에서 재시도): {db_path} - {create_err}")
    except Exception as e:
        # 예상치 못한 오류만 에러로 처리
        logger.warning(f"⚠️ DB 파일 준비 중 오류 (무시 가능, 연결 풀에서 재시도): {db_path} - {e}")

def ensure_storage_ready() -> None:
    """데이터 저장소 디렉터리와 DB 파일들을 사전 보장"""
    try:
        logger.debug(f"📁 DATA_STORAGE_PATH={DATA_STORAGE_PATH}")
        if not os.path.exists(DATA_STORAGE_PATH):
            os.makedirs(DATA_STORAGE_PATH, exist_ok=True)
            logger.info(f"📂 데이터 저장 디렉터리 생성: {DATA_STORAGE_PATH}")
        # 권한/쓰기 가능 여부 점검
        try:
            test_path = os.path.join(DATA_STORAGE_PATH, '.write_test')
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write('ok')
            os.remove(test_path)
            logger.debug("✅ 데이터 디렉터리 쓰기 테스트 통과")
        except Exception as e:
            logger.error(f"❌ 데이터 디렉터리 쓰기 불가: {DATA_STORAGE_PATH} -> {e}")
            raise
        
        # DB 경로 준비 (CANDLES, STRATEGIES, LEARNING_RESULTS)
        for path in (CANDLES_DB_PATH, STRATEGIES_DB_PATH, LEARNING_RESULTS_DB_PATH):
            if path:
                _ensure_file_exists(path)
                
    except Exception as e:
        logger.error(f"❌ 저장소 사전 준비 실패: {e}")

def run_absolute_zero(coin: Optional[str] = None, interval: str = "15m", n_strategies: int = 300, intervals: Optional[List[str]] = None) -> Dict[str, Any]:
    """Absolute Zero 시스템 실행 - 새로운 파이프라인 구조 사용"""
    try:
        start_time = datetime.now()

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
        
        # 공통 함수 사용하여 인터벌 정렬
        intervals_to_use = sort_intervals(intervals_raw)

        # 코인 기본값: DB에서 사용 가능한 코인 목록 우선 사용
        if coin is None:
            try:
                available = get_available_coins_and_intervals()
                coins = sorted(list({c for c, _ in available}))
                if not coins:
                    raise ValueError("❌ DB에 사용 가능한 코인이 없습니다. 캔들 데이터를 먼저 수집하세요.")
                coin = coins[0]
            except Exception as e:
                logger.error(f"❌ 코인 목록 조회 실패: {e}")
                raise ValueError("❌ 코인을 지정하거나 DB에 캔들 데이터가 필요합니다.") from e
        logger.info(f"🚀 Absolute Zero 시스템 시작: {coin} {intervals_to_use}")
        logger.info(f"🗓️ 캔들 히스토리 일수: {AZ_CANDLE_DAYS}일")
        
        # 새로운 파이프라인 사용 가능 여부 확인
        if not NEW_PIPELINE_AVAILABLE:
            logger.error("❌ 새로운 파이프라인 모듈을 사용할 수 없습니다")
            return {"error": "새로운 파이프라인 모듈 사용 불가"}
        
        # 데이터베이스는 이미 시스템 시작 시 초기화됨

        # 🆕 실행 메타데이터 생성
        run_id = f"abs_zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_span = f"{datetime.now().strftime('%Y-%m-%d')}"
        regime = "mixed"  # 실제로는 시장 분석 결과에 따라 결정

        # 🆕 실행 기록 생성 (테이블 생성 후) - coin, interval 포함
        try:
            # 여러 interval 사용 시 첫 번째 interval 사용 (또는 ','로 구분된 문자열)
            interval_str = intervals_to_use[0] if intervals_to_use else interval
            if len(intervals_to_use) > 1:
                interval_str = ','.join(intervals_to_use)  # 여러 interval을 ','로 구분
            create_run_record(run_id, "Absolute Zero System 실행", coin=coin, interval=interval_str)
            logger.info(f"✅ 실행 기록 생성 완료: {run_id} (coin={coin}, intervals={interval_str})")
        except Exception as e:
            logger.warning(f"⚠️ 실행 기록 생성 실패: {e}")

        # 인덱스는 이미 시스템 시작 시 생성됨

        # 🆕 개선 사항 자동 실행 (환경변수로 제어)
        # 🔥 시작 시점 검증/비교 제거 -> 실행 완료 후 수행 (신규 전략 포함 필터링)
        ENABLE_STRATEGY_FILTERING = os.getenv('ENABLE_STRATEGY_FILTERING', 'false').lower() == 'true'
        
        # 🔥 코인별 DB 초기화 (매우 중요: 코인별로 별도 DB 파일 생성 및 테이블 초기화)
        from rl_pipeline.db.schema import setup_database_tables
        
        try:
            coin_strategy_db_path = config.get_strategy_db_path(coin)
            logger.info(f"🔧 {coin} 전략 DB 초기화: {coin_strategy_db_path}")
            
            # 해당 코인 DB에 테이블 생성 (없으면 생성됨)
            if setup_database_tables(coin_strategy_db_path):
                logger.info(f"✅ {coin} 전략 DB 테이블 초기화 완료")
            else:
                logger.warning(f"⚠️ {coin} 전략 DB 테이블 초기화 실패 (이미 존재할 수 있음)")
                
        except Exception as db_init_err:
            logger.error(f"❌ {coin} 전략 DB 초기화 중 오류: {db_init_err}")
            # 초기화 실패 시에도 일단 진행 (연결 풀에서 생성 시도할 수 있음)

        # 🆕 캔들 데이터 로드
        logger.info(f"📊 {coin} 캔들 데이터 로드 시작 (목표: {AZ_CANDLE_DAYS}일)...")
        all_candle_data = load_candle_data_for_coin(coin, intervals_to_use)

        if not all_candle_data:
            logger.error(f"❌ {coin} 캔들 데이터 로드 실패")
            return {"error": f"{coin} 캔들 데이터 로드 실패"}

        # 신생 코인 체크
        total_candles = sum(len(df) for df in all_candle_data.values())
        if total_candles == 0:
            logger.error(f"❌ {coin}: 사용 가능한 캔들 데이터가 없습니다")
            return {"error": f"{coin}: 캔들 데이터 없음"}

        # 데이터 충분성 체크
        min_candles_per_interval = {
            '15m': 672,  # 7일 최소 데이터
            '30m': 336,
            '240m': 42,
            '1d': 7
        }

        insufficient_intervals = []
        for (c, interval), df in all_candle_data.items():
            min_required = min_candles_per_interval.get(interval, 100)
            if len(df) < min_required:
                insufficient_intervals.append(f"{interval}({len(df)}개)")

        if insufficient_intervals:
            logger.warning(f"⚠️ {coin}: 신생 코인 감지 - 일부 인터벌 데이터 부족: {', '.join(insufficient_intervals)}")
            logger.info(f"📊 {coin}: 가용 데이터로 진행합니다")

        # 🆕 통합된 파이프라인 실행
        logger.info(f"🔄 {coin} 통합 파이프라인 실행 시작...")

        # 파이프라인 오케스트레이터 초기화 (session_id 전달)
        orchestrator = IntegratedPipelineOrchestrator(session_id=session_id)
        
        # 각 인터벌별로 통합 파이프라인 실행
        pipeline_results = []
        for idx, interval in enumerate(intervals_to_use):
            try:
                logger.info(f"📊 {coin}-{interval} 통합 파이프라인 실행...")
                
                candle_data = all_candle_data.get((coin, interval))
                if candle_data is None or candle_data.empty:
                    logger.warning(f"⚠️ {coin}-{interval} 캔들 데이터 없음, 건너뜜")
                    continue
                
                # 1-2단계만 실행: 전략생성 → Self-play → 통합분석 (레짐 라우팅 제거)
                result = orchestrator.run_partial_pipeline(coin, interval, candle_data)
                pipeline_results.append(result)
                
                # 실행 결과 로깅
                logger.info(f"✅ {coin}-{interval} 개별 인터벌 처리 완료: 전략 생성 → 예측 self-play → 롤업/등급 평가 완료")
                logger.info(f"   💡 전체 통합 분석 및 학습은 모든 인터벌 완료 후 실행됩니다")
                
                # 🔧 WAL 체크포인트 (간소화 - 마지막에 한 번만 수행하므로 중간 인터벌에서는 생략)
                # 참고: 최종 WAL 정리는 cleanup_all_database_files()에서 수행됨
                
            except Exception as e:
                logger.error(f"❌ {coin}-{interval} 파이프라인 실행 실패: {e}")
                continue
        
        # 🔥 MFE/MAE 라벨링 및 통계 갱신 (전략 생성 완료 후 실행)
        if pipeline_results:
            try:
                from rl_pipeline.labeling.chart_future_scanner import ChartFutureScanner
                from rl_pipeline.labeling.stats_generator import StatsGenerator
                
                logger.info(f"🔄 {coin} MFE/MAE 라벨링 시작 (신규 전략 대상)...")
                
                # 🔥 [Fix] 코인별 DB 경로 명시 (개별 코인 DB에서 전략 로드)
                coin_strategy_db_path = config.get_strategy_db_path(coin)
                
                # 1. 라벨링 실행 (생성된 전략들의 신호를 과거 차트에 대입)
                scanner = ChartFutureScanner(strategy_db_path=coin_strategy_db_path)
                labeling_results = scanner.run_full_labeling(coins=[coin], intervals=intervals_to_use)
                total_labels = sum(labeling_results.values())
                
                if total_labels > 0:
                    logger.info(f"✅ {coin} 라벨링 완료: {total_labels}개 신호 라벨 생성")
                    
                    # 2. 통계 생성 (MFE/MAE 분포 계산) - 동일 DB 경로 사용
                    generator = StatsGenerator(strategy_db_path=coin_strategy_db_path)
                    stats_count = generator.generate_all_stats()
                    logger.info(f"✅ {coin} MFE/MAE 통계 생성 완료: {stats_count}개 전략 통계")
                    
                    # 3. 🔥 리그 승강제 업데이트 (EntryScore 기반)
                    from rl_pipeline.core.strategy_filter import update_league_rankings
                    league_result = update_league_rankings(
                        db_path=coin_strategy_db_path,
                        top_n_per_group=100,  # 코인×인터벌별 상위 100개 major
                        min_entry_score=0.0   # 최소 손익분기점 이상만 major
                    )
                    logger.info(f"🏆 {coin} 리그 업데이트: major {league_result['total_major']}개 (↑{league_result['promoted']} ↓{league_result['demoted']})")
                    
                    # 4. 🧬 자동 진화 시스템 체크 (Phase 승격/강등)
                    try:
                        from rl_pipeline.evolution import run_evolution_check
                        evolution_summary = run_evolution_check(coins=[coin], intervals=intervals_to_use)
                        
                        if evolution_summary.get('total_symbols', 0) > 0:
                            dist = evolution_summary.get('distribution', {})
                            logger.info(f"🧬 {coin} 진화 현황: Phase1={dist.get('STATISTICAL', 0)}, Phase2={dist.get('PREDICTIVE', 0)}, Phase3={dist.get('TIMING_OPTIMIZED', 0)}")
                    except Exception as evo_err:
                        logger.debug(f"⚠️ 진화 체크 실패 (무시 가능): {evo_err}")
                else:
                    logger.info(f"📊 {coin} 라벨링: 신호 없음 (전략 조건에 맞는 과거 구간이 적음)")
                
            except Exception as labeling_err:
                logger.warning(f"⚠️ {coin} 라벨링/통계 갱신 실패 (무시 가능): {labeling_err}")
        
        # 전체 인터벌 통합분석 실행
        if pipeline_results:
            # 🔥 전략 필터링: 파이프라인 실행 완료 후 수행 (신규 생성된 전략까지 포함하여 검증)
            if ENABLE_STRATEGY_FILTERING:
                try:
                    logger.info(f"🔧 {coin} 전략 필터링 시작 (파이프라인 완료 후)...")
                    from rl_pipeline.core.strategy_filter import remove_low_grade_strategies, apply_physics_laws_filter, perform_stress_test, keep_top_strategies
                    
                    # DB 경로 설정 (코인별 DB 사용 시 동적 처리 필요)
                    strategy_db_path = config.get_strategy_db_path(coin)
                    logger.debug(f"🔧 필터링 대상 DB: {strategy_db_path}")
                    
                    # 1. 물리 법칙 필터링 (생존 조건) - 파산 확률 0% 도전
                    # 🔥 사용자 요청: 필터링 대폭 완화 (연구용 X, 실전 데이터 확보 O) - 환경변수 기반 제어
                    removed_physics = apply_physics_laws_filter(
                        db_path=strategy_db_path, 
                        max_mdd_pct=float(os.getenv('FILTER_MAX_MDD_PCT', '0.99')),  # 기본값 0.99 (사실상 해제)
                        min_trades=int(os.getenv('FILTER_MIN_TRADES', '0')),         # 기본값 0 (해제)
                        strict_mode=os.getenv('FILTER_STRICT_MODE', 'false').lower() == 'true' # 기본값 False
                    )
                    if removed_physics > 0:
                        logger.info(f"⚖️ {removed_physics}개 전략 물리 법칙 위반으로 즉시 제거됨")

                    # 2. 정원 관리 (Capacity Management) - 인터벌/레짐별 최적화 🔥
                    # 설정된 정원(STRATEGIES_PER_COMBINATION)을 초과하는 경우 꼴등 제거
                    kept_count, removed_capacity = keep_top_strategies(db_path=strategy_db_path)
                    if removed_capacity > 0:
                        logger.info(f"🧹 정원 관리: {removed_capacity}개 하위 전략 제거 (용량 최적화)")

                    # 3. 스트레스 테스트 (Monte Carlo Simulation) - 최악의 시나리오 검증
                    # 물리 법칙을 통과한 정예 전략들만 대상으로 수행
                    stress_results = perform_stress_test(db_path=strategy_db_path)
                    if stress_results.get("failed_count", 0) > 0:
                        logger.info(f"📉 {stress_results['failed_count']}개 전략 스트레스 테스트(VaR 95%) 탈락")

                    # 4. 등급 기반 필터링 (나머지 정리)
                    # 🔥 MFE/MAE 필터링 추가
                    from rl_pipeline.core.strategy_filter import apply_mfe_filter
                    removed_mfe = apply_mfe_filter(db_path=strategy_db_path, min_entry_score=-0.005)
                    if removed_mfe > 0:
                         logger.info(f"📉 {removed_mfe}개 전략 MFE/MAE Gate 필터링 (EntryScore < -0.5%)")

                    removed = remove_low_grade_strategies(db_path=strategy_db_path)
                    if removed > 0:
                        logger.info(f"✅ {removed}개 F 등급 전략 제거")
                except Exception as e:
                    logger.warning(f"⚠️ 전략 필터링 실패 (계속 진행): {e}")

            try:
                logger.info(f"🔍 {coin} 전체 인터벌 통합분석 시작...")
                final_result = orchestrator.run_integrated_analysis_all_intervals(coin, pipeline_results, all_candle_data)
                logger.info(f"✅ {coin} 전체 통합분석 완료: {final_result.signal_action} (점수: {final_result.signal_score:.3f})")
                
                # 최종 결과를 pipeline_results에 추가
                pipeline_results.append(final_result)
                
                # 학습 결과 DB에 저장
                try:
                    save_pipeline_execution_log(
                        coin=coin,
                        interval="all_intervals",
                        strategies_created=sum(len(result.strategies) for result in pipeline_results if hasattr(result, 'strategies')),
                        selfplay_episodes=sum(result.selfplay_episodes for result in pipeline_results if hasattr(result, 'selfplay_episodes')),
                        regime_detected="multi_interval",
                        routing_results=sum(result.routing_results for result in pipeline_results if hasattr(result, 'routing_results')),
                        signal_score=final_result.signal_score,
                        signal_action=final_result.signal_action,
                        execution_time=final_result.execution_time,
                        status="success"
                    )
                    logger.info(f"✅ 전체 통합분석 로그 저장 완료: {coin}")
                except Exception as log_error:
                    logger.warning(f"⚠️ 로그 저장 실패: {log_error}")
                
            except Exception as e:
                logger.error(f"❌ {coin} 전체 통합분석 실패: {e}")
                # 기본 결과 생성
                final_result = PipelineResult(
                    coin=coin,
                    interval="all",
                    signal_action="HOLD",
                    signal_score=0.5,
                    execution_time=0.0,
                    strategies_created=0,
                    selfplay_episodes=0,
                    regime_detected="unknown",
                    routing_results=0,
                    status="failed",
                    created_at=datetime.now().isoformat()
                )
                pipeline_results.append(final_result)
            
            successful_results = [r for r in pipeline_results if r.status == "success"]
            logger.info(f"✅ {coin} 통합 파이프라인 완료: {len(successful_results)}/{len(pipeline_results)} 성공")
            
            # 🆕 실행 기록 업데이트 (통계 정보 포함)
            try:
                # 통계 정보 계산
                total_strategies = sum(r.strategies_created for r in pipeline_results if r.strategies_created)
                successful_results_count = len(successful_results)
                total_errors = len([r for r in pipeline_results if r.status == "failed"])
                
                update_run_record(
                    run_id, 
                    "completed", 
                    f"{coin} 통합 파이프라인 성공: {successful_results_count}/{len(pipeline_results)} 성공",
                    strategies_count=total_strategies,
                    successful_strategies=successful_results_count,
                    error_count=total_errors
                )
            except Exception as e:
                logger.warning(f"⚠️ 실행 기록 업데이트 실패: {e}")
            
            total_ms = (datetime.now() - start_time).total_seconds() * 1000.0

            # 🔥 디버그 세션 종료
            if session_id:
                try:
                    session_manager.end_session(
                        session_id=session_id,
                        summary={
                            "status": "success",
                            "pipeline_results": len(pipeline_results),
                            "successful_results": len(successful_results),
                            "elapsed_ms": round(total_ms, 2)
                        }
                    )
                    logger.info(f"✅ 디버그 세션 종료: {session_id}")
                except Exception as end_err:
                    logger.warning(f"⚠️ 디버그 세션 종료 실패: {end_err}")

            # Self-play 결과 데이터 추출 (글로벌 학습용)
            selfplay_data = {}
            for r in pipeline_results:
                if r.status in ["success", "partial_complete"] and r.selfplay_result:
                    selfplay_data[r.interval] = r.selfplay_result

            return {
                "run_id": run_id,
                "coin": coin,
                "interval": ",".join(intervals_to_use),
                "status": "success",
                "message": f"{coin} 통합 파이프라인 성공",
                "pipeline_results": len(pipeline_results),
                "successful_results": len(successful_results),
                "selfplay_data": selfplay_data,  # 🆕 Self-play 결과 추가
                "elapsed_ms": round(total_ms, 2),
                "session_id": session_id
            }
        else:
            logger.error(f"❌ {coin} 통합 파이프라인 실패")
            total_ms = (datetime.now() - start_time).total_seconds() * 1000.0

            # 🔥 디버그 세션 종료 (실패)
            if session_id:
                try:
                    session_manager.end_session(
                        session_id=session_id,
                        summary={
                            "status": "failed",
                            "message": f"{coin} 통합 파이프라인 실패",
                            "elapsed_ms": round(total_ms, 2)
                        }
                    )
                except Exception as end_err:
                    logger.warning(f"⚠️ 디버그 세션 종료 실패: {end_err}")

            return {
                "run_id": run_id,
                "coin": coin,
                "interval": ",".join(intervals_to_use),
                "status": "failed",
                "message": f"{coin} 통합 파이프라인 실패",
                "elapsed_ms": round(total_ms, 2),
                "session_id": session_id
            }

    except Exception as e:
        logger.error(f"❌ Absolute Zero 시스템 실행 실패: {e}")

        # 🔥 디버그 세션 종료 (예외)
        if 'session_id' in locals() and session_id:
            try:
                if 'session_manager' in locals():
                    session_manager.end_session(
                        session_id=session_id,
                        summary={
                            "status": "error",
                            "error": str(e)
                        }
                    )
            except Exception as end_err:
                logger.warning(f"⚠️ 디버그 세션 종료 실패: {end_err}")

        return {"error": f"시스템 실행 실패: {e}"}

def _calculate_global_analysis_data(all_coin_strategies: Dict[str, Any]) -> Dict[str, float]:
    """실제 전략 데이터 기반으로 글로벌 분석 데이터 계산"""
    try:
        from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
        
        # 모든 코인의 전략 데이터 수집
        all_strategies = []
        for coin, intervals in all_coin_strategies.items():
            for interval, strategies in intervals.items():
                all_strategies.extend(strategies)
        
        if not all_strategies:
            return {
                'fractal_score': 0.5,
                'multi_timeframe_score': 0.5,
                'indicator_cross_score': 0.5,
                'ensemble_score': 0.5,
                'ensemble_confidence': 0.5
            }
        
        # IntegratedAnalyzer 인스턴스 생성 (session_id는 선택적)
        # session_id가 없으면 None으로 전달
        analyzer = IntegratedAnalyzer(session_id=None)
        
        # 1. Fractal 점수 계산 (전략 파라미터 분포 기반)
        fractal_score = _calculate_fractal_score(all_strategies)
        
        # 2. Multi-timeframe 점수 계산 (여러 타임프레임 간 일관성)
        multi_timeframe_score = _calculate_multi_timeframe_coherence(all_coin_strategies)
        
        # 3. Indicator cross-validation 점수 계산 (지표 간 교차 검증)
        indicator_cross_score = _calculate_indicator_cross_validation(all_strategies)
        
        # 4. Ensemble 점수 계산
        ensemble_score = (fractal_score + multi_timeframe_score + indicator_cross_score) / 3.0
        ensemble_confidence = min(1.0, max(0.0, len(all_strategies) / 1000.0))
        
        return {
            'fractal_score': round(fractal_score, 3),
            'multi_timeframe_score': round(multi_timeframe_score, 3),
            'indicator_cross_score': round(indicator_cross_score, 3),
            'ensemble_score': round(ensemble_score, 3),
            'ensemble_confidence': round(ensemble_confidence, 3)
        }
        
    except Exception as e:
        logger.warning(f"⚠️ 글로벌 분석 데이터 계산 실패, 기본값 사용: {e}")
        return {
            'fractal_score': 0.5,
            'multi_timeframe_score': 0.5,
            'indicator_cross_score': 0.5,
            'ensemble_score': 0.5,
            'ensemble_confidence': 0.5
        }

def _get_value(obj, key: str, default: Any):
    """객체나 딕셔너리에서 값을 가져오는 헬퍼 함수"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        # Strategy 객체인 경우
        # 먼저 params 딕셔너리 확인
        if hasattr(obj, 'params') and isinstance(obj.params, dict):
            if key in obj.params:
                return obj.params[key]
        # 그 다음 객체 속성 확인
        return getattr(obj, key, default)

def _format_price(price: float) -> str:
    """가격 포맷팅: 1원 미만은 소수점 4자리, 100원 미만은 소수점 2자리, 100원 이상은 천단위 콤마"""
    try:
        if price is None:
            return "0"
        if price == 0:
            return "0"
        
        # 1원 미만인 경우 소수점 4자리까지 정확히 표시
        if price < 1.0:
            return f"{price:.4f}"
        
        # 1원 이상 100원 미만인 경우 소수점 2자리까지 표시
        if price < 100.0:
            return f"{price:.2f}"
        
        # 100원 이상인 경우 천단위 콤마 추가
        return f"{int(price):,}"
            
    except Exception:
        return f"{price}"

def _calculate_fractal_score(strategies: List[Dict]) -> float:
    """Fractal 점수 계산 - 전략 파라미터 분포 기반 프랙탈 패턴 분석"""
    if not strategies:
        return 0.5
    
    try:
        # RSI 파라미터 분포 분석
        rsi_mins = [_get_value(s, 'rsi_min', 30.0) for s in strategies if hasattr(s, 'rsi_min') or (isinstance(s, dict) and 'rsi_min' in s)]
        rsi_maxs = [_get_value(s, 'rsi_max', 70.0) for s in strategies if hasattr(s, 'rsi_max') or (isinstance(s, dict) and 'rsi_max' in s)]
        
        if rsi_mins and rsi_maxs:
            rsi_min_std = np.std(rsi_mins) if len(rsi_mins) > 1 else 0.0
            rsi_max_std = np.std(rsi_maxs) if len(rsi_maxs) > 1 else 0.0
            
            # 분산이 적절하면 높은 점수 (일관된 패턴)
            # 너무 낮으면 단조롭고, 너무 높으면 무작위적
            avg_std = (rsi_min_std + rsi_max_std) / 2.0
            # 이상적인 표준편차: 5-15 사이
            if 5.0 <= avg_std <= 15.0:
                fractal_score = 0.8
            elif avg_std < 5.0:
                fractal_score = 0.5  # 너무 단조로움
            else:
                fractal_score = 0.6  # 너무 다양함
        else:
            fractal_score = 0.5
        
        return fractal_score
        
    except Exception as e:
        logger.debug(f"⚠️ Fractal 점수 계산 실패: {e}")
        return 0.5

def _calculate_multi_timeframe_coherence(all_coin_strategies: Dict[str, Dict]) -> float:
    """Multi-timeframe 일관성 점수 계산 - 여러 타임프레임 간 전략 일관성 분석"""
    try:
        coherence_scores = []
        
        for coin, intervals in all_coin_strategies.items():
            if len(intervals) < 2:
                continue  # 타임프레임이 2개 미만이면 건너뜀
            
            # 각 타임프레임의 평균 RSI 범위 계산
            interval_rsi_ranges = {}
            for interval, strategies in intervals.items():
                if not strategies:
                    continue

                rsi_mins = [_get_value(s, 'rsi_min', 30.0) for s in strategies if hasattr(s, 'rsi_min') or (isinstance(s, dict) and 'rsi_min' in s)]
                rsi_maxs = [_get_value(s, 'rsi_max', 70.0) for s in strategies if hasattr(s, 'rsi_max') or (isinstance(s, dict) and 'rsi_max' in s)]
                
                if rsi_mins and rsi_maxs:
                    avg_min = np.mean(rsi_mins)
                    avg_max = np.mean(rsi_maxs)
                    interval_rsi_ranges[interval] = (avg_min, avg_max)
            
            # 타임프레임 간 RSI 범위 차이 계산
            if len(interval_rsi_ranges) >= 2:
                ranges = list(interval_rsi_ranges.values())
                min_diffs = [abs(ranges[i][0] - ranges[j][0]) for i in range(len(ranges)) for j in range(i+1, len(ranges))]
                max_diffs = [abs(ranges[i][1] - ranges[j][1]) for i in range(len(ranges)) for j in range(i+1, len(ranges))]
                
                avg_min_diff = np.mean(min_diffs) if min_diffs else 0.0
                avg_max_diff = np.mean(max_diffs) if max_diffs else 0.0
                
                # 차이가 작을수록 일관성 높음 (높은 점수)
                # 이상적인 차이: 5-10 사이 (적절한 다양성과 일관성)
                avg_diff = (avg_min_diff + avg_max_diff) / 2.0
                if avg_diff <= 10.0:
                    coherence = 0.8 - (avg_diff / 25.0)  # 차이가 작을수록 높은 점수
                else:
                    coherence = 0.5
                
                coherence_scores.append(coherence)
        
        if coherence_scores:
            return np.mean(coherence_scores)
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"⚠️ Multi-timeframe 일관성 계산 실패: {e}")
        return 0.5

def _calculate_indicator_cross_validation(strategies: List[Dict]) -> float:
    """Indicator 교차 검증 점수 계산 - 지표 간 일관성"""
    if not strategies:
        return 0.5
    
    try:
        # RSI와 MACD 임계값 간 상관관계 분석
        rsi_scores = []
        macd_scores = []
        
        for strategy in strategies:
            # RSI 점수 (rsi_min, rsi_max의 적절성)
            rsi_min = _get_value(strategy, 'rsi_min', 30.0)
            rsi_max = _get_value(strategy, 'rsi_max', 70.0)
            if 20.0 <= rsi_min <= 40.0 and 60.0 <= rsi_max <= 80.0:
                rsi_scores.append(1.0)
            else:
                rsi_scores.append(0.5)

            # MACD 점수 (macd_buy_threshold, macd_sell_threshold의 적절성)
            macd_buy = _get_value(strategy, 'macd_buy_threshold', 0.0)
            macd_sell = _get_value(strategy, 'macd_sell_threshold', 0.0)
            if macd_buy > 0 and macd_sell < 0:
                macd_scores.append(1.0)
            else:
                macd_scores.append(0.5)
        
        # 두 지표 점수의 평균 및 일관성
        avg_rsi = np.mean(rsi_scores) if rsi_scores else 0.5
        avg_macd = np.mean(macd_scores) if macd_scores else 0.5
        
        # 두 지표가 모두 높으면 높은 점수
        cross_score = (avg_rsi + avg_macd) / 2.0
        
        return cross_score
        
    except Exception as e:
        logger.debug(f"⚠️ Indicator 교차 검증 계산 실패: {e}")
        return 0.5

def report_strategy_performance(coin: str):
    """전략 방향성 및 예측 정확도 리포트 출력"""
    try:
        logger.info(f"\n📊 {coin} 전략 성과 리포트 (방향성 및 정확도)")
        logger.info("=" * 80)
        logger.info(f"{'Interval':<10} | {'Total':<6} | {'Buy':<5} | {'Sell':<5} | {'Win Rate':<10} | {'Avg Profit':<10} | {'Top Grade':<10}")
        logger.info("-" * 80)
        
        with get_optimized_db_connection(config.get_strategy_db_path(coin)) as conn:
            cursor = conn.cursor()
            
            # 인터벌별 전략 통계 조회
            # symbol 컬럼 우선 사용, 없으면 coin 컬럼 사용
            try:
                cursor.execute("PRAGMA table_info(strategies)")
                columns = [row[1] for row in cursor.fetchall()]
                coin_col = 'symbol' if 'symbol' in columns else 'coin'
            except:
                coin_col = 'coin'

            cursor.execute(f"""
                SELECT interval, 
                       COUNT(*) as total_count,
                       AVG(win_rate) as avg_win_rate,
                       AVG(profit) as avg_profit,
                       SUM(CASE WHEN strategy_type LIKE '%_buy' OR strategy_type LIKE '%buy%' THEN 1 ELSE 0 END) as buy_count,
                       SUM(CASE WHEN strategy_type LIKE '%_sell' OR strategy_type LIKE '%sell%' THEN 1 ELSE 0 END) as sell_count,
                       MAX(quality_grade) as top_grade,
                       GROUP_CONCAT(DISTINCT regime) as regimes
                FROM strategies
                WHERE {coin_col} = ?
                GROUP BY interval
                ORDER BY interval
            """, (coin,))
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("  ⚠️ 전략 데이터가 없습니다.")
                logger.info("=" * 80)
                return

            # 인터벌 정렬을 위한 헬퍼
            def get_minutes(iv):
                iv = iv.lower()
                if iv.endswith('m'): return int(iv[:-1])
                if iv.endswith('h'): return int(iv[:-1]) * 60
                if iv.endswith('d'): return int(iv[:-1]) * 1440
                return 99999

            # rows 정렬
            rows.sort(key=lambda x: get_minutes(x[0]))
            
            for row in rows:
                interval, total, win_rate, profit, buy, sell, top_grade, regimes = row
                # None 처리
                win_rate = win_rate if win_rate else 0.0
                profit = profit if profit else 0.0
                buy = buy if buy else 0
                sell = sell if sell else 0
                top_grade = top_grade if top_grade else '-'
                regimes_str = regimes if regimes else 'none'
                
                logger.info(f"{interval:<10} | {total:<6} | {buy:<5} | {sell:<5} | {win_rate*100:>9.1f}% | {profit:>10.2f} | {top_grade:<10}")
                logger.info(f"   └─ 커버 레짐: {regimes_str}")

        logger.info("=" * 80)
            
    except Exception as e:
        logger.warning(f"⚠️ 리포트 생성 실패: {e}")

# ============================================================================
# 🔥 체크포인트 기능 - 학습 중단 시 이어서 진행 가능
# ============================================================================

def _get_checkpoint_path() -> str:
    """체크포인트 파일 경로 반환"""
    return os.path.join(DATA_STORAGE_PATH, "learning_checkpoint.json")

def load_checkpoint() -> Dict[str, Any]:
    """체크포인트 로드 - 완료된 코인 목록 반환"""
    checkpoint_path = _get_checkpoint_path()
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                completed = checkpoint.get('completed_coins', [])
                last_updated = checkpoint.get('last_updated', 'unknown')
                logger.info(f"📂 체크포인트 로드: {len(completed)}개 코인 완료됨 (마지막 업데이트: {last_updated})")
                return checkpoint
    except Exception as e:
        logger.warning(f"⚠️ 체크포인트 로드 실패 (처음부터 시작): {e}")
    return {'completed_coins': [], 'last_updated': None}

def save_checkpoint(completed_coins: List[str]) -> bool:
    """체크포인트 저장 - 완료된 코인 목록 저장"""
    checkpoint_path = _get_checkpoint_path()
    try:
        checkpoint = {
            'completed_coins': completed_coins,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"💾 체크포인트 저장: {len(completed_coins)}개 코인 완료")
        return True
    except Exception as e:
        logger.warning(f"⚠️ 체크포인트 저장 실패: {e}")
        return False

def clear_checkpoint() -> bool:
    """체크포인트 삭제 - 전체 학습 완료 시 호출"""
    checkpoint_path = _get_checkpoint_path()
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("🗑️ 체크포인트 삭제 완료 (전체 학습 완료)")
        return True
    except Exception as e:
        logger.warning(f"⚠️ 체크포인트 삭제 실패: {e}")
        return False

def cleanup_all_database_files():
    """모든 데이터베이스 임시 파일 정리 및 연결 종료"""
    logger.info("🧹 모든 데이터베이스 임시 파일 정리 시작...")
    
    try:
        # 1. 모든 연결 종료
        if db_pool:
            db_pool.close_all_connections(verbose=True)
            
            # 2. 각 풀별 WAL 파일 정리
            pools_to_clean = [
                getattr(db_pool, '_strategy_pool', None),
                getattr(db_pool, '_candle_pool', None),
                getattr(db_pool, '_learning_results_pool', None),
                getattr(db_pool, '_batch_pool', None)
            ]
            
            for pool in pools_to_clean:
                if pool:
                    try:
                        pool.cleanup_wal_files()
                    except Exception as wal_err:
                        pass
                        
            # 코인별 전략 풀도 정리
            if hasattr(db_pool, '_strategy_pools') and db_pool._strategy_pools:
                for pool in list(db_pool._strategy_pools.values()):
                    try:
                        pool.close_all_connections()
                        pool.cleanup_wal_files()
                    except:
                        pass
                
        logger.info("✅ 데이터베이스 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 정리 실패: {e}")

def main():
    """메인 함수 - 모든 코인/인터벌 조합에 대해 실행"""
    try:
        _configure_logging()
        logger.info("🚀 Absolute Zero 시스템 메인 실행 시작")
        
        # 저장 경로 및 DB 파일 사전 보장
        ensure_storage_ready()
        
        # 🆕 시스템 시작 시 한 번만 데이터베이스 초기화
        try:
            logger.info("🔧 시스템 데이터베이스 초기화 시작...")
            setup_database_tables()
            create_learning_results_tables()  # 새로운 학습 결과 테이블 생성
            
            # 필수 테이블 보강 생성 (방어적)
            try:
                create_coin_strategies_table()
            except Exception as se:
                logger.warning(f"⚠️ coin_strategies 보강 생성 실패(무시 가능): {se}")
            
            # 🆕 데이터베이스 마이그레이션 실행 (누락된 컬럼 추가)
            try:
                from rl_pipeline.db.schema import migrate_strategies_table
                migrate_strategies_table()
                # migrate_rl_episode_summary_table()  # 존재하지 않으므로 주석 처리
                logger.info("✅ 데이터베이스 마이그레이션 완료")
            except Exception as me:
                logger.warning(f"⚠️ 데이터베이스 마이그레이션 실패(무시 가능): {me}")
            
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

        # 사용 가능한 코인/인터벌 조합 가져오기
        coin_interval_combinations = get_available_coins_and_intervals()
        logger.info(f"📊 발견된 코인/인터벌 조합: {len(coin_interval_combinations)}개")
        # 코인별 전체 인터벌로 그룹핑
        coin_to_intervals: Dict[str, List[str]] = {}
        for c, itv in coin_interval_combinations:
            coin_to_intervals.setdefault(c, [])
            if itv not in coin_to_intervals[c]:
                coin_to_intervals[c].append(itv)
        # 공통 함수 사용하여 인터벌 정렬
        for c in coin_to_intervals:
            coin_to_intervals[c] = sort_intervals(coin_to_intervals[c])
        
        if not coin_interval_combinations:
            logger.error("❌ 사용 가능한 코인/인터벌 조합이 없습니다.")
            logger.error("❌ 캔들 데이터를 먼저 수집하세요: python candles_collector.py")
            return {"error": "no coin/interval combinations found", "message": "캔들 데이터를 먼저 수집하세요"}
        
        # 🔥 체크포인트 로드 - 이전에 완료된 코인 확인
        checkpoint = load_checkpoint()
        completed_coins = set(checkpoint.get('completed_coins', []))
        
        # 각 조합에 대해 실행
        results = []
        failed_runs = []
        skipped_coins = []
        
        # 모든 코인의 self-play 결과 수집
        all_coin_strategies = {}
        all_coin_selfplay = {}  # 🆕 글로벌 학습용 self-play 데이터
        total_strategies = 0
        
        # 🔥 코인 목록 정렬 (일관된 순서 보장)
        sorted_coins = sorted(coin_to_intervals.keys())
        total_coins = len(sorted_coins)

        for idx, coin in enumerate(sorted_coins):
            intervals = coin_to_intervals[coin]
            
            # 🔥 이미 완료된 코인은 건너뛰기
            if coin in completed_coins:
                logger.info(f"⏭️ [{idx+1}/{total_coins}] {coin} 건너뛰기 (이미 완료됨)")
                skipped_coins.append(coin)
                continue
            
            try:
                logger.info(f"\n🪙 [{idx+1}/{total_coins}] {coin} {', '.join(intervals)} 처리 시작")
                result = run_absolute_zero(coin, interval=intervals[0], n_strategies=200, intervals=intervals)
                results.append(result)
                
                if result.get("status") == "success":
                    logger.info(f"✅ {coin} 처리 성공")
                    
                    # 🆕 Self-play 데이터 수집
                    if "selfplay_data" in result:
                        all_coin_selfplay[coin] = result["selfplay_data"]

                    # 🔥 전략 성과 리포트 출력
                    report_strategy_performance(coin)
                    
                    # 🔥 완료된 코인 체크포인트 저장
                    completed_coins.add(coin)
                    save_checkpoint(list(completed_coins))
                else:
                    logger.error(f"❌ {coin} 처리 실패: {result.get('message', 'Unknown error')}")
                    failed_runs.append(f"{coin}_{','.join(intervals)}")
                    
            except Exception as e:
                logger.error(f"❌ {coin} 처리 중 오류: {e}")
                failed_runs.append(f"{coin}_{','.join(intervals)}")
                continue
        
        # 결과 요약
        successful_runs = len([r for r in results if r.get("status") == "success"])
        total_coins_count = len(coin_to_intervals)
        actually_run = total_coins_count - len(skipped_coins)
        
        logger.info(f"\n🎉 Absolute Zero 시스템 실행 완료")
        logger.info(f"📊 전체 코인: {total_coins_count}개")
        if skipped_coins:
            logger.info(f"⏭️ 건너뛴 코인 (이전 완료): {len(skipped_coins)}개")
        logger.info(f"📊 이번 실행: {actually_run}개, 성공: {successful_runs}개, 실패: {len(failed_runs)}개")
        logger.info(f"📊 누적 완료: {len(completed_coins)}개 / {total_coins_count}개")
        
        if failed_runs:
            logger.warning(f"⚠️ 실패한 조합: {failed_runs}")
        
        # 🔥 전체 완료 시 체크포인트 삭제 (다음 실행은 처음부터)
        if len(completed_coins) >= total_coins_count and len(failed_runs) == 0:
            clear_checkpoint()
            logger.info("🎊 모든 코인 학습 완료! 다음 실행은 처음부터 시작합니다.")
        
        # 🌍 글로벌 전략 생성 (모든 코인의 모든 시간대 완료 후)
        if successful_runs > 0:
            try:
                # 🔥 글로벌 전략 생성 전에도 라벨링/통계 갱신 수행 (안전을 위해)
                # (이미 개별 코인 처리 시 수행되었으므로 생략 가능하나, 글로벌 전용 로직이 필요할 수 있음)
                
                logger.info("\n🌍 글로벌 전략 생성 시작 (모든 코인의 모든 시간대 완료 후)...")
                
                logger.info("📊 코인별 전략 로드 상세 정보:")
                for coin, intervals in coin_to_intervals.items():
                    coin_strategies = {}
                    coin_total = 0
                    
                    for interval in intervals:
                        try:
                            # self-play로 진화된 전략 로드 (모든 등급 포함)
                            from rl_pipeline.db.reads import fetch_all
                            
                            # 🔥 코인별 DB 경로 사용
                            coin_db_path = config.get_strategy_db_path(coin)
                            
                            strategies = []
                            with get_optimized_db_connection(coin_db_path) as conn:
                                cursor = conn.cursor()
                                
                                # 🔥 컬럼명 동적 확인 (coin vs symbol)
                                cursor.execute("PRAGMA table_info(strategies)")
                                columns_info = cursor.fetchall()
                                columns = [col[1] for col in columns_info]
                                coin_col = 'symbol' if 'symbol' in columns else 'coin'
                                
                                # 모든 등급의 상위 전략 로드 (등급 우선순위: S > A > B > C > D > F, 제한 없음)
                                query = f"""
                                    SELECT * FROM strategies 
                                    WHERE {coin_col} = ? AND interval = ?
                                    ORDER BY 
                                        CASE COALESCE(quality_grade, 'Z')
                                            WHEN 'S' THEN 0
                                            WHEN 'A' THEN 1
                                            WHEN 'B' THEN 2
                                            WHEN 'C' THEN 3
                                            WHEN 'D' THEN 4
                                            WHEN 'F' THEN 5
                                            ELSE 6
                                        END ASC,
                                        win_rate DESC,
                                        profit DESC
                                """
                                cursor.execute(query, (coin, interval))
                                results_sql = cursor.fetchall()
                                
                                if results_sql:
                                    # 🔥 해당 DB에서 테이블 정보 가져오기
                                    columns_query = "PRAGMA table_info(strategies)"
                                    columns_info = cursor.execute(columns_query).fetchall()
                                    columns = [col[1] for col in columns_info]
                                    
                                    for row in results_sql:
                                        strategy_dict = dict(zip(columns, row))
                                        strategies.append(strategy_dict)
                            
                            if strategies:
                                coin_strategies[interval] = strategies
                                coin_total += len(strategies)
                                total_strategies += len(strategies)
                                
                                # 전략 품질 정보 표시
                                if strategies:
                                    # profit을 달러에서 퍼센트로 변환
                                    avg_profit_pnl = sum(_get_value(s, 'profit', 0) or 0 for s in strategies) / len(strategies)
                                    avg_profit_pct = (avg_profit_pnl / 10000.0) * 100  # 퍼센트로 변환
                                    
                                    # PnL 평균값 포맷팅 (예: 1,000,000)
                                    avg_profit_str = _format_price(avg_profit_pnl)

                                    avg_win_rate = sum(_get_value(s, 'win_rate', 0) or 0 for s in strategies) / len(strategies)

                                    # 등급 분포 계산
                                    grade_dist = {}
                                    for s in strategies:
                                        grade = _get_value(s, 'quality_grade', None) or 'UNKNOWN'
                                        grade_dist[grade] = grade_dist.get(grade, 0) + 1

                                    grade_str = ', '.join([f"{k}({v})" for k, v in sorted(grade_dist.items())])

                                    # 데이터 상태 확인
                                    has_performance_data = any(_get_value(s, 'profit', 0) != 0 or _get_value(s, 'win_rate', 0) != 0 for s in strategies)
                                    has_grades = any(_get_value(s, 'quality_grade', None) and _get_value(s, 'quality_grade', 'UNKNOWN') != 'UNKNOWN' for s in strategies)
                                    
                                    if not has_performance_data and not has_grades:
                                        logger.info(f"  ✅ {coin} {interval}: {len(strategies)}개 전략 (평균 PnL: {avg_profit_str}, 평균 수익: {avg_profit_pct:+.2f}%, 평균 승률: {avg_win_rate:.3f}, 등급: {grade_str}) [💡 성과 데이터 없음]")
                                    elif not has_performance_data:
                                        logger.info(f"  ✅ {coin} {interval}: {len(strategies)}개 전략 (평균 PnL: {avg_profit_str}, 평균 수익: {avg_profit_pct:+.2f}%, 평균 승률: {avg_win_rate:.3f}, 등급: {grade_str}) [💡 수익/승률 데이터 없음]")
                                    elif not has_grades:
                                        logger.info(f"  ✅ {coin} {interval}: {len(strategies)}개 전략 (평균 PnL: {avg_profit_str}, 평균 수익: {avg_profit_pct:+.2f}%, 평균 승률: {avg_win_rate:.3f}, 등급: {grade_str}) [💡 등급 데이터 없음]")
                                    else:
                                        logger.info(f"  ✅ {coin} {interval}: {len(strategies)}개 전략 (평균 PnL: {avg_profit_str}, 평균 수익: {avg_profit_pct:+.2f}%, 평균 승률: {avg_win_rate:.3f}, 등급: {grade_str})")
                            else:
                                logger.warning(f"  ⚠️ {coin} {interval}: 전략 없음")
                        except Exception as e:
                            logger.warning(f"  ❌ {coin} {interval} 전략 로드 실패: {e}")
                    
                    if coin_strategies:
                        all_coin_strategies[coin] = coin_strategies
                        logger.info(f"📊 {coin}: 총 {coin_total}개 전략 로드 완료")
                
                logger.info(f"📊 전체 통계: {len(all_coin_strategies)}개 코인, {total_strategies}개 전략")
                
                if all_coin_strategies:
                    # 🌍 글로벌 전략 생성 (🆕 세밀한 구간화 기반 + 기존 방식 병행)
                    logger.info("\n🌍 글로벌 전략 생성 시작")
                    
                    available_combinations = get_available_coins_and_intervals()
                    intervals = sorted(list({itv for _, itv in available_combinations}))
                    if not intervals:
                        intervals = config.UNIFIED_INTERVALS
                    
                    global_strategies_count = 0
                    binned_predictions_count = 0
                    
                    # ===== 1. 새로운 방식: 세밀한 구간화 기반 글로벌 예측값 =====
                    logger.info("\n📊 [방식 1] 세밀한 구간화 기반 글로벌 예측값 생성")
                    try:
                        from rl_pipeline.strategy.binned_global_synthesizer import create_binned_global_synthesizer
                        
                        binned_synthesizer = create_binned_global_synthesizer(
                            source_db_path=config.STRATEGIES_DB,
                            output_db_path=config.STRATEGIES_DB,
                            intervals=intervals,
                            seed=123
                        )
                        
                        # 세밀한 구간화 파이프라인 실행
                        result = binned_synthesizer.run_synthesis(
                            min_trades=5,    # 최소 거래 5회
                            max_dd=0.8,      # 최대 DD 80%
                            min_samples=2    # 최소 샘플 2개 (중간값 의미있게)
                        )
                        
                        if result['success']:
                            binned_predictions_count = result['output_predictions']
                            logger.info(f"✅ 구간화 기반 글로벌 예측값 생성 완료: {binned_predictions_count}개")
                            for interval, count in result['interval_stats'].items():
                                logger.info(f"    ● {interval}: {count}개")
                        else:
                            logger.warning(f"⚠️ 구간화 기반 글로벌 예측값 생성 실패: {result.get('error')}")
                    
                    except Exception as be:
                        logger.warning(f"⚠️ 구간화 기반 글로벌 예측값 생성 실패: {be}")
                        import traceback
                        logger.warning(traceback.format_exc())
                    
                    # ===== 2. 기존 방식: 레짐별 대표 전략 (폴백용) =====
                    logger.info("\n📊 [방식 2] 레짐별 대표 전략 생성 (폴백용)")
                    try:
                        from rl_pipeline.strategy.global_synthesizer import create_global_synthesizer

                        synthesizer = create_global_synthesizer(config.STRATEGIES_DB, intervals, seed=123)
                        
                        # 7단계 Synthesizer 파이프라인 실행
                        logger.info("  📊 1단계: 개별 전략 수집...")
                        pool = synthesizer.load_pool(coins=list(all_coin_strategies.keys()), min_trades=0, max_dd=1.0)
                        
                        logger.info("  📊 2단계: 전략 표준화...")
                        std_pool = synthesizer.standardize(pool)
                        
                        logger.info("  📊 3단계: 공통 패턴 추출...")
                        patterns = synthesizer.extract_common_patterns(std_pool)
                        
                        logger.info("  📊 4단계: 글로벌 전략 조립...")
                        assembled = synthesizer.assemble_global_strategies(patterns)
                        
                        logger.info("  📊 5단계: 샌티백테스트...")
                        tested = synthesizer.quick_sanity_backtest(assembled)
                        
                        logger.info("  📊 6단계: 폴백 적용...")
                        final = synthesizer.apply_fallbacks(tested)
                        
                        logger.info("  📊 7단계: 저장...")
                        synthesizer.save(final)
                        
                        global_strategies_count = sum(len(s) for s in final.values())
                        logger.info(f"✅ 레짐별 대표 전략 생성 완료: {global_strategies_count}개")
                    
                    except Exception as ge:
                        logger.error(f"❌ 레짐별 대표 전략 생성 실패: {ge}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    logger.info(f"\n✨ 글로벌 전략 생성 총계:")
                    logger.info(f"   📊 구간화 기반 예측값: {binned_predictions_count}개 (global_strategy_predictions)")
                    logger.info(f"   📊 레짐별 대표 전략: {global_strategies_count}개 (global_strategies)")

                    # 🔥 코인 vs 글로벌 전략 동적 가중치 계산 및 저장
                    try:
                        from rl_pipeline.db.writes import save_coin_global_weights
                        from rl_pipeline.db.reads import fetch_all

                        logger.info("⚖️ 코인 vs 글로벌 전략 동적 가중치 계산 시작...")

                        # 🔥 글로벌 전략 성능 계산 (매수/매도 그룹 분리)
                        # 매수 그룹
                        global_buy_strats = fetch_all(
                            """SELECT profit, win_rate, trades_count
                               FROM global_strategies
                               WHERE profit IS NOT NULL AND trades_count > 0
                                 AND (strategy_type LIKE '%_buy' OR strategy_type LIKE '%buy%')""",
                            db_path=config.STRATEGIES_DB
                        )

                        # 매도 그룹
                        global_sell_strats = fetch_all(
                            """SELECT profit, win_rate, trades_count
                               FROM global_strategies
                               WHERE profit IS NOT NULL AND trades_count > 0
                                 AND (strategy_type LIKE '%_sell' OR strategy_type LIKE '%sell%')""",
                            db_path=config.STRATEGIES_DB
                        )
                        
                        # 전체 (방향성 없는 전략 포함)
                        global_all_strats = fetch_all(
                            """SELECT profit, win_rate, trades_count
                               FROM global_strategies
                               WHERE profit IS NOT NULL AND trades_count > 0""",
                            db_path=config.STRATEGIES_DB
                        )

                        # 매수 그룹 성능
                        global_buy_avg_profit = sum(s[0] for s in global_buy_strats) / len(global_buy_strats) if global_buy_strats else 0.0
                        global_buy_avg_win_rate = sum(s[1] for s in global_buy_strats) / len(global_buy_strats) if global_buy_strats else 0.0
                        global_buy_score = (global_buy_avg_win_rate * 0.8 + global_buy_avg_profit * 0.2) if global_buy_strats else 0.0
                        
                        # 매도 그룹 성능
                        global_sell_avg_profit = sum(s[0] for s in global_sell_strats) / len(global_sell_strats) if global_sell_strats else 0.0
                        global_sell_avg_win_rate = sum(s[1] for s in global_sell_strats) / len(global_sell_strats) if global_sell_strats else 0.0
                        global_sell_score = (global_sell_avg_win_rate * 0.8 + global_sell_avg_profit * 0.2) if global_sell_strats else 0.0
                        
                        # 전체 성능 (하위 호환성)
                        global_avg_profit = sum(s[0] for s in global_all_strats) / len(global_all_strats) if global_all_strats else 0.0
                        global_avg_win_rate = sum(s[1] for s in global_all_strats) / len(global_all_strats) if global_all_strats else 0.0
                        global_strategy_count = len(global_all_strats)
                        global_score = (global_avg_win_rate * 0.8 + global_avg_profit * 0.2) if global_all_strats else 0.0

                        logger.info(f"  📊 글로벌 전략 성능 (전체): profit={global_avg_profit:.4f}, win_rate={global_avg_win_rate:.4f}, count={global_strategy_count}")
                        if global_buy_strats:
                            logger.info(f"  📊 글로벌 매수 전략: profit={global_buy_avg_profit:.4f}, win_rate={global_buy_avg_win_rate:.4f}, count={len(global_buy_strats)}")
                        if global_sell_strats:
                            logger.info(f"  📊 글로벌 매도 전략: profit={global_sell_avg_profit:.4f}, win_rate={global_sell_avg_win_rate:.4f}, count={len(global_sell_strats)}")

                        # 각 코인별 가중치 계산
                        for coin in all_coin_strategies.keys():
                            try:
                                # 코인 전략 성능 계산
                                # 🔥 컬럼명 동적 확인 (coin vs symbol)
                                coin_db_path = config.get_strategy_db_path(coin)
                                with get_optimized_db_connection(coin_db_path) as conn:
                                    cursor = conn.cursor()
                                    cursor.execute("PRAGMA table_info(strategies)")
                                    cols = [c[1] for c in cursor.fetchall()]
                                    coin_col = 'symbol' if 'symbol' in cols else 'coin'
                                
                                coin_strats = fetch_all(
                                    f"""SELECT profit, win_rate, trades_count
                                       FROM strategies
                                       WHERE {coin_col} = ? AND profit IS NOT NULL AND trades_count > 0""",
                                    (coin,),
                                    db_path=coin_db_path
                                )

                                if not coin_strats:
                                    logger.debug(f"  ⚠️ {coin}: 유효한 전략 없음, 기본 가중치 사용")
                                    continue

                                coin_avg_profit = sum(s[0] for s in coin_strats) / len(coin_strats)
                                coin_avg_win_rate = sum(s[1] for s in coin_strats) / len(coin_strats)
                                coin_strategy_count = len(coin_strats)
                                coin_score = (coin_avg_win_rate * 0.8 + coin_avg_profit * 0.2)

                                # 데이터 품질 점수 계산 (전략 개수 기반)
                                # 많은 전략 = 높은 품질, 적은 전략 = 낮은 품질
                                min_required_strategies = 10
                                data_quality_score = min(1.0, coin_strategy_count / min_required_strategies)

                                # 동적 가중치 계산
                                # 1. 데이터가 부족하면 글로벌 비중 증가
                                # 2. 코인 성능이 좋으면 코인 비중 증가
                                # 3. 글로벌 성능이 좋으면 글로벌 비중 증가

                                base_coin_weight = 0.7
                                base_global_weight = 0.3

                                # 데이터 품질에 따른 조정 (-0.3 ~ +0.2)
                                quality_adjustment = (data_quality_score - 0.5) * 0.5  # -0.25 ~ +0.25

                                # 성능 차이에 따른 조정 (-0.2 ~ +0.2)
                                if global_score > 0:
                                    performance_ratio = coin_score / global_score if global_score > 0 else 1.0
                                    performance_adjustment = (performance_ratio - 1.0) * 0.2  # 코인이 더 좋으면 +, 글로벌이 더 좋으면 -
                                else:
                                    performance_adjustment = 0.0

                                # 최종 가중치 계산 (0.1 ~ 0.9 범위)
                                coin_weight = base_coin_weight + quality_adjustment + performance_adjustment
                                coin_weight = max(0.1, min(0.9, coin_weight))  # 최소 10%, 최대 90%
                                global_weight = 1.0 - coin_weight

                                # DB에 저장
                                weights_data = {
                                    'coin_weight': coin_weight,
                                    'global_weight': global_weight,
                                    'coin_score': coin_score,
                                    'global_score': global_score,
                                    'data_quality_score': data_quality_score,
                                    'coin_strategy_count': coin_strategy_count,
                                    'global_strategy_count': global_strategy_count,
                                    'coin_avg_profit': coin_avg_profit,
                                    'global_avg_profit': global_avg_profit,
                                    'coin_win_rate': coin_avg_win_rate,
                                    'global_win_rate': global_avg_win_rate
                                }

                                save_coin_global_weights(coin, weights_data)
                                logger.info(f"  ✅ {coin}: 가중치 저장 (coin={coin_weight:.2f}, global={global_weight:.2f}, quality={data_quality_score:.2f})")

                            except Exception as coin_err:
                                logger.warning(f"  ⚠️ {coin} 가중치 계산 실패: {coin_err}")
                                continue

                        logger.info("✅ 코인 vs 글로벌 전략 동적 가중치 계산 완료")
                    except Exception as weight_err:
                        logger.warning(f"⚠️ 동적 가중치 계산 실패 (계속 진행): {weight_err}")

                    # 🔥 글로벌 전략 레짐 라우팅 제거됨 (개별 코인과 동일하게 제거)

                else:
                    logger.warning("⚠️ 글로벌 전략 생성을 위한 전략 데이터가 없습니다")
                    
            except Exception as e:
                logger.error(f"❌ 글로벌 전략 생성 실패: {e}")
        
        # 🌍 통합된 파이프라인 결과 요약
        if successful_runs > 0:
            logger.info("\n🌍 통합된 파이프라인 결과 요약...")
            
            try:
                # 파이프라인 성능 요약 생성
                performance_summary = get_pipeline_performance_summary(days=1)
                
                if performance_summary:
                    logger.info("✅ 파이프라인 성능 요약 완료!")
                    logger.info(f"📊 총 실행: {performance_summary.get('total_runs', 0)}개")
                    logger.info(f"📊 성공률: {performance_summary.get('success_rate', 0):.1f}%")
                    logger.info(f"📊 평균 실행 시간: {performance_summary.get('avg_execution_time', 0):.2f}초")
                    
                    # 🆕 글로벌 전략 결과 저장
                    try:
                        from rl_pipeline.db.learning_results import save_global_strategy_results
                        
                        # 상위 성과자 추출 (성공한 결과에서)
                        top_performers = []
                        for r in results:
                            # 🔥 타입 확인: 딕셔너리가 아니면 건너뛰기
                            if not isinstance(r, dict):
                                continue
                            
                            if r.get("status") == "success":
                                top_performers.append({
                                    'coin': r.get('coin', ''),
                                    'interval': r.get('interval', ''),
                                    'score': r.get('elapsed_ms', 0) / 1000.0  # 실행 시간 기반 점수
                                })
                        
                        # 전체 점수 계산
                        # 🔥 performance_summary 타입 확인
                        if not isinstance(performance_summary, dict):
                            logger.warning("⚠️ performance_summary가 딕셔너리가 아닙니다. 기본값 사용")
                            overall_score = successful_runs / max(total_coins_count, 1)
                            overall_confidence = min(1.0, successful_runs / max(total_coins_count, 1))
                        else:
                            overall_score = performance_summary.get('success_rate', 0) / 100.0
                            overall_confidence = min(1.0, successful_runs / max(total_coins_count, 1))
                        
                        save_global_strategy_results(
                            overall_score=overall_score,
                            overall_confidence=overall_confidence,
                            top_performers=top_performers[:20]  # 상위 20개
                        )
                        logger.info("✅ 글로벌 전략 결과 저장 완료")
                    except Exception as e:
                        logger.warning(f"⚠️ 글로벌 전략 결과 저장 실패: {e}")
                else:
                    logger.warning("⚠️ 파이프라인 성능 요약 실패")
                
                logger.info("🎉 통합된 파이프라인 결과 요약 완료!")
                
            except Exception as e:
                logger.error(f"❌ 파이프라인 결과 요약 실패: {e}", exc_info=True)
        else:
            logger.warning("⚠️ 성공한 코인이 없어 파이프라인 결과 요약을 건너뜁니다.")
        
        # 🧹 데이터베이스 정리
        try:
            logger.info("\n🧹 데이터베이스 정리 시작...")
            try:
                from rl_pipeline.db.connection_pool import cleanup_all_database_files
                cleanup_all_database_files()
            except ImportError:
                # 간단한 DB 정리 (rl_candles.db는 제외 - 읽기 전용 원천 데이터)
                logger.info("📊 간단한 DB 정리 수행...")
                import sqlite3
                for db_path in [STRATEGIES_DB_PATH]:  # CANDLES_DB_PATH 제외!
                    try:
                        conn = sqlite3.connect(db_path)
                        conn.execute("VACUUM")
                        conn.close()
                        logger.info(f"✅ {db_path} 정리 완료")
                    except Exception as db_e:
                        logger.warning(f"⚠️ {db_path} 정리 실패: {db_e}")
            logger.info("✅ 데이터베이스 정리 완료!")
        except Exception as e:
            logger.warning(f"⚠️ 데이터베이스 정리 실패: {e}")
        
        return {
            "total_runs": total_coins_count,
            "successful_runs": successful_runs,
            "failed_runs": len(failed_runs),
            "skipped_runs": len(skipped_coins),
            "failed_combinations": failed_runs,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ 메인 실행 실패: {e}")
        return {"error": f"메인 실행 실패: {e}"}

def generate_global_strategies_only(
    coin_filter: Optional[List[str]] = None,
    enable_training: bool = False
):
    """
    글로벌 전략만 독립적으로 생성

    Args:
        coin_filter: 특정 코인만 필터링 (None이면 모든 코인)
        enable_training: 글로벌 학습 실행 여부 (기본값: False)
    """
    try:
        _configure_logging()
        logger.info("🌍 글로벌 전략 생성 실행 시작 (Synthesizer 방식)")

        # 🔥 검증 함수 import
        from rl_pipeline.pipelines.orchestrator import (
            validate_global_strategy_pool,
            validate_global_strategy_patterns,
            validate_global_strategy_quality
        )

        # GlobalStrategySynthesizer 사용
        from rl_pipeline.strategy.global_synthesizer import create_global_synthesizer

        # 🔥 디버그 세션 생성
        session_manager = SessionManager()
        session_id = session_manager.create_session(
            coins=coin_filter or ["ALL"],
            intervals=["global"],
            config={"enable_training": enable_training}
        )
        
        # Synthesizer 초기화
        db_path = config.STRATEGIES_DB
        
        # DB에서 실제 사용 가능한 인터벌 조회 (하드코딩 제거)
        available_combinations = get_available_coins_and_intervals()
        intervals = sorted(list({itv for _, itv in available_combinations}))
        
        # 사용 가능한 인터벌이 없으면 기본값 사용
        if not intervals:
            intervals = config.UNIFIED_INTERVALS
        
        seed = 123  # 재현성을 위한 seed
        
        synthesizer = create_global_synthesizer(db_path, intervals, seed)
        
        # 코인 필터링
        if coin_filter:
            logger.info(f"📋 코인 필터: {coin_filter}")
            coins = coin_filter
        else:
            # 사용 가능한 모든 코인 가져오기
            coin_interval_combinations = get_available_coins_and_intervals()
            coins = list(set([c for c, _ in coin_interval_combinations]))
            logger.info(f"📊 발견된 코인: {len(coins)}개")
        
        # 7단계 Synthesizer 파이프라인 실행
        logger.info("📊 1단계: 개별 전략 수집...")
        # 필터 조건 완화: min_trades=1 (최소 1개 거래), max_dd=1.0 (100% 허용)
        pool = synthesizer.load_pool(coins=coins, min_trades=1, max_dd=1.0)

        # 🔥 1단계 검증: 전략 풀 검증
        pool_validation = validate_global_strategy_pool(
            pool=pool,
            coins=coins,
            intervals=intervals,
            min_strategies_per_interval=10
        )

        logger.info(f"📊 전략 풀 검증 완료")
        logger.info(f"   └─ 검증 통과: {pool_validation['valid']}")
        logger.info(f"   └─ 데이터 품질 점수: {pool_validation.get('quality_score', 0)}/100")
        logger.info(f"   └─ 총 전략 수: {pool_validation['stats'].get('total_strategies', 0)}개")
        logger.info(f"   └─ 인터벌 커버리지: {pool_validation['stats'].get('intervals_covered', 0)}/{pool_validation['stats'].get('intervals_expected', 0)}")

        if pool_validation['issues']:
            logger.error(f"❌ 전략 풀 검증 실패:")
            for issue in pool_validation['issues']:
                logger.error(f"   └─ {issue}")

        if pool_validation['warnings']:
            logger.warning(f"⚠️ 전략 풀 경고:")
            for warning in pool_validation['warnings']:
                logger.warning(f"   └─ {warning}")

        # 🔥 디버그 로그 저장
        try:
            from rl_pipeline.monitoring.simulation_debugger import SimulationDebugger
            debugger = SimulationDebugger(session_id=session_id)
            debugger.log({
                'event': 'global_strategy_pool_validation',
                'validation_result': {
                    'valid': pool_validation['valid'],
                    'quality_score': pool_validation.get('quality_score', 0),
                    'total_strategies': pool_validation['stats'].get('total_strategies', 0),
                    'intervals_covered': pool_validation['stats'].get('intervals_covered', 0),
                    'num_issues': len(pool_validation['issues']),
                    'num_warnings': len(pool_validation['warnings'])
                },
                'issues': pool_validation['issues'],
                'warnings': pool_validation['warnings']
            })
        except Exception as debug_error:
            logger.debug(f"⚠️ 검증 결과 디버그 로깅 실패: {debug_error}")

        if not pool:
            logger.warning("⚠️ 수집된 개별 전략 없음, 폴백만 생성")
            final = synthesizer.apply_fallbacks({})
            synthesizer.save(final)

            # 🔥 세션 종료
            session_manager.end_session(session_id, summary={
                'status': 'fallback_only',
                'strategies_generated': sum(len(s) for s in final.values())
            })

            return {"success": True, "count": sum(len(s) for s in final.values())}

        logger.info("📊 2단계: 전략 표준화...")
        std_pool = synthesizer.standardize(pool)

        logger.info("📊 3단계: 공통 패턴 추출...")
        patterns = synthesizer.extract_common_patterns(std_pool)

        # 🔥 3단계 검증: 패턴 검증
        pattern_validation = validate_global_strategy_patterns(
            patterns=patterns,
            min_patterns_per_interval=3
        )

        logger.info(f"📊 패턴 추출 검증 완료")
        logger.info(f"   └─ 검증 통과: {pattern_validation['valid']}")
        logger.info(f"   └─ 품질 점수: {pattern_validation.get('quality_score', 0)}/100")
        logger.info(f"   └─ 총 패턴 수: {pattern_validation['stats'].get('total_patterns', 0)}개")

        if pattern_validation['issues']:
            logger.error(f"❌ 패턴 검증 실패:")
            for issue in pattern_validation['issues']:
                logger.error(f"   └─ {issue}")

        if pattern_validation['warnings']:
            logger.warning(f"⚠️ 패턴 경고:")
            for warning in pattern_validation['warnings']:
                logger.warning(f"   └─ {warning}")

        # 🔥 디버그 로그 저장
        try:
            debugger.log({
                'event': 'global_strategy_pattern_validation',
                'validation_result': {
                    'valid': pattern_validation['valid'],
                    'quality_score': pattern_validation.get('quality_score', 0),
                    'total_patterns': pattern_validation['stats'].get('total_patterns', 0),
                    'num_issues': len(pattern_validation['issues']),
                    'num_warnings': len(pattern_validation['warnings'])
                },
                'issues': pattern_validation['issues'],
                'warnings': pattern_validation['warnings']
            })
        except Exception as debug_error:
            logger.debug(f"⚠️ 패턴 검증 결과 디버그 로깅 실패: {debug_error}")
        
        logger.info("📊 4단계: 글로벌 전략 조립...")
        assembled = synthesizer.assemble_global_strategies(patterns)
        
        logger.info("📊 5단계: 빠른 샌티백테스트...")
        tested = synthesizer.quick_sanity_backtest(assembled)
        
        logger.info("📊 6단계: 폴백 적용...")
        final = synthesizer.apply_fallbacks(tested)

        # 🔥 7단계 전: 최종 품질 검증
        final_validation = validate_global_strategy_quality(
            final_strategies=final,
            intervals=intervals,
            min_strategies_per_interval=5
        )

        logger.info(f"📊 최종 글로벌 전략 품질 검증 완료")
        logger.info(f"   └─ 검증 통과: {final_validation['valid']}")
        logger.info(f"   └─ 품질 점수: {final_validation.get('quality_score', 0)}/100")
        logger.info(f"   └─ 총 전략 수: {final_validation['stats'].get('total_strategies', 0)}개")
        logger.info(f"   └─ 인터벌당 평균: {final_validation['stats'].get('avg_strategies_per_interval', 0)}개")

        if final_validation['issues']:
            logger.error(f"❌ 최종 품질 검증 실패:")
            for issue in final_validation['issues']:
                logger.error(f"   └─ {issue}")

        if final_validation['warnings']:
            logger.warning(f"⚠️ 최종 품질 경고:")
            for warning in final_validation['warnings']:
                logger.warning(f"   └─ {warning}")

        # 🔥 인터벌별 상세 통계 로깅
        interval_dist = final_validation['stats'].get('interval_distribution', {})
        if interval_dist:
            logger.info(f"📊 인터벌별 최종 전략 통계:")
            for interval, stat in interval_dist.items():
                logger.info(f"   └─ {interval}: {stat['strategy_count']}개 전략")

        # 🔥 디버그 로그 저장
        try:
            debugger.log({
                'event': 'global_strategy_quality_validation',
                'validation_result': {
                    'valid': final_validation['valid'],
                    'quality_score': final_validation.get('quality_score', 0),
                    'total_strategies': final_validation['stats'].get('total_strategies', 0),
                    'avg_strategies_per_interval': final_validation['stats'].get('avg_strategies_per_interval', 0),
                    'num_issues': len(final_validation['issues']),
                    'num_warnings': len(final_validation['warnings'])
                },
                'issues': final_validation['issues'],
                'warnings': final_validation['warnings'],
                'interval_distribution': interval_dist
            })
        except Exception as debug_error:
            logger.debug(f"⚠️ 최종 검증 결과 디버그 로깅 실패: {debug_error}")

        logger.info("📊 7단계: 글로벌 전략 저장...")
        synthesizer.save(final)

        total_strategies = sum(len(strategies) for strategies in final.values())
        logger.info(f"✅ 레짐별 대표 전략 생성 완료: {total_strategies}개")

        # 🌟 추가: 세밀한 구간화 기반 글로벌 예측값 생성
        binned_predictions_count = 0
        logger.info("\n📊 [추가] 세밀한 구간화 기반 글로벌 예측값 생성")
        try:
            from rl_pipeline.strategy.binned_global_synthesizer import create_binned_global_synthesizer
            
            binned_synthesizer = create_binned_global_synthesizer(
                source_db_path=db_path,
                output_db_path=db_path,
                intervals=intervals,
                seed=seed
            )
            
            # 세밀한 구간화 파이프라인 실행
            binned_result = binned_synthesizer.run_synthesis(
                min_trades=5,
                max_dd=0.8,
                min_samples=2
            )
            
            if binned_result['success']:
                binned_predictions_count = binned_result['output_predictions']
                logger.info(f"✅ 구간화 기반 글로벌 예측값 생성 완료: {binned_predictions_count}개")
                for interval, count in binned_result['interval_stats'].items():
                    logger.info(f"    ● {interval}: {count}개")
            else:
                logger.warning(f"⚠️ 구간화 기반 글로벌 예측값 생성 실패: {binned_result.get('error')}")
        
        except Exception as be:
            logger.warning(f"⚠️ 구간화 기반 글로벌 예측값 생성 실패: {be}")
            import traceback
            logger.warning(traceback.format_exc())

        # 🔥 세션 종료
        session_manager.end_session(session_id, summary={
            'status': 'success',
            'strategies_generated': total_strategies,
            'binned_predictions_generated': binned_predictions_count,
            'pool_quality_score': pool_validation.get('quality_score', 0),
            'pattern_quality_score': pattern_validation.get('quality_score', 0),
            'final_quality_score': final_validation.get('quality_score', 0),
            'overall_quality': round((
                pool_validation.get('quality_score', 0) +
                pattern_validation.get('quality_score', 0) +
                final_validation.get('quality_score', 0)
            ) / 3, 2)
        })

        logger.info(f"\n✨ 글로벌 전략 생성 총계:")
        logger.info(f"   📊 구간화 기반 예측값: {binned_predictions_count}개 (global_strategy_predictions)")
        logger.info(f"   📊 레짐별 대표 전략: {total_strategies}개 (global_strategies)")
        
        result = {
            "success": True,
            "count": total_strategies,
            "binned_predictions_count": binned_predictions_count,
            "details": {
                "intervals": list(final.keys()),
                "strategies_per_interval": {k: len(v) for k, v in final.items()}
            }
        }
        
        return result
            
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 생성 실패: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import argparse
    
    # Argument parser 설정
    parser = argparse.ArgumentParser(description='Absolute Zero System')
    parser.add_argument('--global-only', action='store_true', 
                        help='글로벌 전략만 생성 (self-play 결과 기반)')
    parser.add_argument('--coins', nargs='+', default=None,
                        help='특정 코인만 필터링 (예: --coins BTC ETH)')
    args = parser.parse_args()
    
    # 로깅 설정 (간결한 형식)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 🔥 시작 전 DB 연결 정리 (락 방지)
    try:
        cleanup_all_database_files()
    except Exception as e:
        logger.warning(f"⚠️ 초기 DB 정리 중 오류 (무시 가능): {e}")
    
    # --global-only 플래그 확인
    if args.global_only:
        logger.info("🌍 글로벌 전략 생성 모드")
        result = generate_global_strategies_only(coin_filter=args.coins)
        
        if result.get("success"):
            logger.info(f"✅ 글로벌 전략 생성 완료: {result.get('count', 0)}개")
            sys.exit(0)
        else:
            logger.error(f"❌ 글로벌 전략 생성 실패: {result.get('reason', 'unknown')}")
            sys.exit(1)
    else:
        # 메인 실행 (전체 self-play 포함)
        result = main()
        
        if "error" in result:
            logger.error(f"❌ 실행 실패: {result['error']}")
            sys.exit(1)
        else:
            logger.info(f"✅ 실행 완료: {result['successful_runs']}/{result['total_runs']} 성공")
            sys.exit(0)
