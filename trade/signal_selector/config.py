"""
설정 모듈 - 환경 변수 및 상수 설정

이 모듈은 시그널 선택 시스템의 모든 설정과 환경 변수를 관리합니다.
- 고성능 시스템 설정 (GPU 가속, 캐시 크기 등)
- 데이터베이스 경로 설정
- AI 모델 import 및 초기화
"""

import os

# 경로 설정 (더 안정적인 방법)
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)  # signal_selector/
_trade_dir = os.path.dirname(_current_dir)  # trade/
workspace_dir = os.path.dirname(_trade_dir)  # auto_trader 루트

# 🚀 고성능 시스템 설정
USE_GPU_ACCELERATION = os.getenv('USE_GPU_ACCELERATION', 'true').lower() == 'true'
JAX_PLATFORM_NAME = os.getenv('JAX_PLATFORM_NAME', 'gpu')
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))
CACHE_SIZE = int(os.getenv('CACHE_SIZE', '50000'))
ENABLE_CROSS_COIN_LEARNING = os.getenv('ENABLE_CROSS_COIN_LEARNING', 'true').lower() == 'true'

# 🆕 DB 경로 설정 (Windows 환경 지원)
CANDLES_DB_PATH = os.getenv('CANDLES_DB_PATH', os.path.join(workspace_dir, 'data_storage', 'rl_candles.db'))
STRATEGIES_DB_PATH = os.getenv('STRATEGIES_DB_PATH', os.path.join(workspace_dir, 'data_storage', 'learning_results.db'))
TRADING_SYSTEM_DB_PATH = os.path.join(workspace_dir, 'data_storage', 'trading_system.db')
DB_PATH = TRADING_SYSTEM_DB_PATH

# 🆕 자체 데이터베이스 연결 시스템 (rl_pipeline 충돌 방지)
DB_POOL_AVAILABLE = True
CONFLICT_MANAGER_AVAILABLE = True

# 🆕 크로스 코인 학습 설정
CROSS_COIN_AVAILABLE = os.getenv('CROSS_COIN_AVAILABLE', 'false').lower() == 'true'

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

# 🚀 GPU 가속 설정
if USE_GPU_ACCELERATION:
    try:
        import jax
        import logging as std_logging
        jax_logger = std_logging.getLogger('jax._src.xla_bridge')
        jax_logger.setLevel(std_logging.ERROR)
        
        os.environ.setdefault('JAX_PLATFORM_NAME', JAX_PLATFORM_NAME)
        os.environ.setdefault('XLA_PLATFORM_NAME', JAX_PLATFORM_NAME)
        
        jax.config.update('jax_platform_name', JAX_PLATFORM_NAME)
        print(f"🚀 GPU 가속 활성화: {JAX_PLATFORM_NAME}")
    except ImportError:
        print("⚠️ JAX를 import할 수 없습니다. CPU 모드로 실행됩니다.")
        USE_GPU_ACCELERATION = False
        JAX_PLATFORM_NAME = 'cpu'
        jax = None

# 🆕 AI 모델 import
try:
    from learning_engine import (
        PolicyTrainer, GlobalLearningManager, SymbolFinetuningManager, 
        ShortTermLongTermSynergyLearner, ReliabilityScoreCalculator,
        ContinuousLearningManager, RoutingPatternAnalyzer, 
        ContextualLearningManager, analyze_strategy_quality
    )
    AI_MODEL_AVAILABLE = True
    SYNERGY_LEARNING_AVAILABLE = True
    print("✅ learning_engine 고급 기능 로드 완료")
except ImportError:
    AI_MODEL_AVAILABLE = False
    SYNERGY_LEARNING_AVAILABLE = False
    print("⚠️ AI 모델을 import할 수 없습니다. 기본 시그널 계산만 사용됩니다.")
    PolicyTrainer = None
    GlobalLearningManager = None
    SymbolFinetuningManager = None
    ShortTermLongTermSynergyLearner = None
    ReliabilityScoreCalculator = None
    ContinuousLearningManager = None
    RoutingPatternAnalyzer = None
    ContextualLearningManager = None
    analyze_strategy_quality = None

