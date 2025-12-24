"""
설정 모듈 - 환경 변수 및 상수 설정

이 모듈은 시그널 선택 시스템의 모든 설정과 환경 변수를 관리합니다.
- 데이터베이스 경로 설정
- 시스템 설정 (스레드 수, 캐시 크기 등)
"""

import os

# 경로 설정 (더 안정적인 방법)
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)  # signal_selector/
_trade_dir = os.path.dirname(_current_dir)  # trade/
workspace_dir = os.path.dirname(_trade_dir)  # auto_trader 루트

# 🚀 시스템 설정
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))
CACHE_SIZE = int(os.getenv('CACHE_SIZE', '50000'))
ENABLE_CROSS_COIN_LEARNING = os.getenv('ENABLE_CROSS_COIN_LEARNING', 'true').lower() == 'true'

# 🚫 GPU 가속 설정 (사용하지 않음, 호환성 유지를 위해 변수는 남겨둠)
USE_GPU_ACCELERATION = False
JAX_PLATFORM_NAME = 'cpu'

# 🆕 DB 경로 설정 (Windows 환경 지원 및 DATA_STORAGE_PATH 우선)
# 환경 변수에서 경로를 가져오되, Windows 경로가 아닌 경우(/workspace/...) 자동으로 로컬 경로로 변환
_env_storage = os.getenv('DATA_STORAGE_PATH')
if _env_storage and (_env_storage.startswith('/workspace') or _env_storage.startswith('\\workspace')):
     # Docker 경로를 로컬 경로로 변환 (Windows 환경일 경우)
    if os.name == 'nt':
         _default_market_storage = os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage')
         DATA_STORAGE_PATH = _default_market_storage
    else:
         DATA_STORAGE_PATH = _env_storage
else:
    DATA_STORAGE_PATH = _env_storage

if not DATA_STORAGE_PATH:
    # 기본 경로 탐색: market/coin_market/data_storage 우선, 없으면 root/data_storage
    _default_market_storage = os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage')
    if os.path.isdir(_default_market_storage):
        DATA_STORAGE_PATH = _default_market_storage
    else:
        DATA_STORAGE_PATH = os.path.join(workspace_dir, 'data_storage')

# 🔥 [Fix] 캔들 DB도 환경변수 우선 사용 (run_learning.py / run_trading.py 호환)
_env_candles_db = os.getenv('RL_DB_PATH') or os.getenv('CANDLES_DB_PATH')
if _env_candles_db:
    CANDLES_DB_PATH = _env_candles_db
else:
    # 기본값: trade_candles.db (트레이딩 환경)
    CANDLES_DB_PATH = os.path.join(DATA_STORAGE_PATH, 'trade_candles.db')

# 🔥 [Fix] 환경변수 우선 사용 (run_learning.py와 호환성 보장)
# 🔧 디렉토리 모드 지원: 환경변수가 .db로 끝나지 않거나 디렉토리인 경우 디렉토리로 간주
_env_strategy_db = os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
if _env_strategy_db:
    STRATEGIES_DB_PATH = _env_strategy_db
else:
    STRATEGIES_DB_PATH = os.path.join(DATA_STORAGE_PATH, 'learning_strategies')

_env_trading_db = os.getenv('TRADING_DB_PATH') or os.getenv('TRADING_SYSTEM_DB_PATH')
if _env_trading_db:
    TRADING_SYSTEM_DB_PATH = _env_trading_db
else:
    TRADING_SYSTEM_DB_PATH = os.path.join(DATA_STORAGE_PATH, 'trading_system.db')
DB_PATH = TRADING_SYSTEM_DB_PATH

# 🆕 자체 데이터베이스 연결 시스템 (rl_pipeline 충돌 방지)
DB_POOL_AVAILABLE = True
CONFLICT_MANAGER_AVAILABLE = True

# 🆕 크로스 코인 학습 설정
CROSS_COIN_AVAILABLE = os.getenv('CROSS_COIN_AVAILABLE', 'true').lower() == 'true'

# 🚀 최적화된 성능 설정
PERFORMANCE_CONFIG = {
    'ENABLE_BATCH_PROCESSING': True,
    'BATCH_SIZE': 50,
    'MAX_WORKERS': 8,
    'ENABLE_CACHING': True,
    'CACHE_TTL': 300,
    'ENABLE_PROGRESS_TRACKING': True,
    'LOG_DETAILED_METRICS': True,
    'OPTIMIZE_240M': True,
    'REDUCE_DB_QUERIES': True,
    'USE_BATCH_QUERIES': True,
    'ENABLE_CONNECTION_POOL': True,
    'ENABLE_PREPARED_STATEMENTS': True,
    'MEMORY_OPTIMIZATION': True
}

# 🚀 딥러닝 AI 모델 관련 설정 (비활성화)
# learning_engine 모듈이 존재하지 않으므로 기본 비활성화 처리
AI_MODEL_AVAILABLE = False
SYNERGY_LEARNING_AVAILABLE = False
PolicyTrainer = None
GlobalLearningManager = None
SymbolFinetuningManager = None
ShortTermLongTermSynergyLearner = None
ReliabilityScoreCalculator = None
ContinuousLearningManager = None
RoutingPatternAnalyzer = None
ContextualLearningManager = None
analyze_strategy_quality = None

# print("ℹ️ AI Learning Engine 비활성화 (모듈 미포함)")

# 🆕 변동성 시스템
try:
    from rl_pipeline.utils.coin_volatility import CoinVolatilityCalculator
    VOLATILITY_SYSTEM_AVAILABLE = True
except ImportError:
    VOLATILITY_SYSTEM_AVAILABLE = False


# 🔧 코인별 전략 DB 경로 함수 (strategy_signal_generator 순환 import 방지)
def get_coin_strategy_db_path(coin: str = None) -> str:
    """개별 코인의 전략 DB 경로 반환 (Directory Mode 지원)
    
    Args:
        coin: 코인 심볼 (예: 'BTC', 'ETH')
        
    Returns:
        DB 파일 경로 (예: /workspace/.../learning_strategies/btc_strategies.db)
    """
    base_path = STRATEGIES_DB_PATH
    
    # 1. 디렉토리 모드인지 확인 (확장자가 .db가 아니거나, 실제 디렉토리인 경우)
    is_directory = not base_path.endswith('.db') or os.path.isdir(base_path)
    
    if is_directory:
        if not coin:
            # 코인이 지정되지 않았는데 디렉토리 모드인 경우, 기본/공용 파일 반환
            return os.path.join(base_path, 'common_strategies.db')
        
        # 코인별 파일명 생성 (소문자 변환)
        return os.path.join(base_path, f"{coin.lower()}_strategies.db")
    
    # 2. 단일 파일 모드 (기존 호환성)
    return base_path
