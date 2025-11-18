"""
Absolute Zero 시스템 - 설정 및 환경 구성 모듈
환경 변수, 경로 설정, 로깅 구성 등을 관리
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# 경고 및 환경 설정
# ============================================================================

# NumPy overflow/underflow 경고 숨김
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# TensorFlow Protobuf 버전 경고 숨김 (JAX 로드 시 발생하는 경고, 기능 영향 없음)
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Sharding info not provided.*', category=UserWarning)

# JAX TPU/ROCm 백엔드 방지 및 CUDA 강제 사용
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

# ============================================================================
# 경로 설정
# ============================================================================

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

# 환경변수 파일 로드
env_path = PROJECT_ROOT / 'rl_pipeline_config.env'
if env_path.exists():
    load_dotenv(env_path)

# Docker 환경 경로 설정
WORKSPACE_ROOT = os.getenv('WORKSPACE_ROOT', '/workspace')
AUTO_TRADER_ROOT = os.getenv('AUTO_TRADER_ROOT', '/workspace')
RL_PIPELINE_ROOT = os.getenv('RL_PIPELINE_ROOT', '/workspace/rl_pipeline')
DATA_STORAGE_PATH = os.getenv('DATA_STORAGE_PATH', '/workspace/data_storage')

# ============================================================================
# 실행 규모/범위 설정 (환경변수로 제어)
# ============================================================================

AZ_INTERVALS = os.getenv('AZ_INTERVALS')  # 예: "15m,30m,240m,1d"
AZ_CANDLE_DAYS = int(os.getenv('AZ_CANDLE_DAYS', '60'))  # 기본 60일 (신생 코인은 가능한 데이터만큼 사용)
AZ_ALLOW_FALLBACK = os.getenv('AZ_ALLOW_FALLBACK', 'false').lower() == 'true'
AZ_FALLBACK_PAIRS = os.getenv('AZ_FALLBACK_PAIRS', '')  # 예: "BTC:15m;ETH:15m"

# Self-play 및 전략 풀 설정 (환경변수로 제어)
AZ_SELFPLAY_EPISODES = int(os.getenv('AZ_SELFPLAY_EPISODES', '200'))  # Self-play 에피소드 수
AZ_SELFPLAY_AGENTS_PER_EPISODE = int(os.getenv('AZ_SELFPLAY_AGENTS_PER_EPISODE', '4'))  # 에피소드당 에이전트 수
AZ_STRATEGY_POOL_SIZE = int(os.getenv('AZ_STRATEGY_POOL_SIZE', '15000'))  # DB에서 로드할 최대 전략 수

# 점진적 통합: 예측 실현 Self-play 비율 (0.0-1.0)
PREDICTIVE_SELFPLAY_RATIO = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))  # 기본값: 20%

# 데이터베이스 경로 설정
STRATEGIES_DB_PATH = os.path.join(DATA_STORAGE_PATH, 'rl_strategies.db')
CANDLES_DB_PATH = os.path.join(DATA_STORAGE_PATH, 'rl_candles.db')
# learning_results.db는 이제 rl_strategies.db로 통합됨 (core/env.py 참조)
LEARNING_RESULTS_DB_PATH = STRATEGIES_DB_PATH

# ============================================================================
# 디버그 설정
# ============================================================================

AZ_DEBUG = os.getenv('AZ_DEBUG', 'false').lower() == 'true'
AZ_LOG_FILE = os.getenv('AZ_LOG_FILE', str(PROJECT_ROOT / 'absolute_zero_debug.log'))
AZ_SIMULATION_VERBOSE = os.getenv('AZ_SIMULATION_VERBOSE', 'false').lower() == 'true'

# 환경변수 설명:
# AZ_DEBUG=true: 모든 DEBUG 로그 출력 (매우 상세)
# AZ_SIMULATION_VERBOSE=true: 시뮬레이션 상세 로그 출력 (전략별 RSI/MACD/Volume 로그)

# ============================================================================
# 로깅 설정
# ============================================================================

logger = logging.getLogger(__name__)

def configure_logging():
    """로깅 설정을 구성합니다."""
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

# ============================================================================
# 저장소 준비 함수
# ============================================================================

def ensure_file_exists(db_path: str) -> None:
    """DB 파일이 없으면 생성 (원천 데이터 DB는 제외)"""
    import sqlite3

    try:
        parent = os.path.dirname(db_path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
                logger.debug(f"📂 DB 디렉토리 생성: {parent}")
            except Exception as dir_err:
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
                logger.debug(f"⚠️ DB 파일 생성 시도 실패 (무시 가능): {db_path} - {create_err}")
    except Exception as e:
        logger.warning(f"⚠️ DB 파일 준비 중 오류 (무시 가능): {db_path} - {e}")

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

        # LEARNING_RESULTS_DB_PATH는 이제 STRATEGIES_DB_PATH와 동일하므로 중복 제거
        for path in (CANDLES_DB_PATH, STRATEGIES_DB_PATH):
            ensure_file_exists(path)
    except Exception as e:
        logger.error(f"❌ 저장소 사전 준비 실패: {e}")

# ============================================================================
# 기타 설정
# ============================================================================

# 전략 필터링 활성화 여부
ENABLE_STRATEGY_FILTERING = os.getenv('ENABLE_STRATEGY_FILTERING', 'false').lower() == 'true'

# 최소 캔들 데이터 요구사항
MIN_CANDLES_PER_INTERVAL = {
    '15m': 672,  # 7일 최소 데이터
    '30m': 336,
    '240m': 42,
    '1d': 7
}