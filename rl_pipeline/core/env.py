"""
Absolute Zero 시스템 환경 설정 중앙화
모든 설정의 단일 출처로 AI가 길을 잃지 않게 함
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 환경변수 로드
env_path = os.path.join(os.path.dirname(__file__), '..', 'rl_pipeline_config.env')
load_dotenv(env_path)


class ConfigProfile:
    """설정 프로파일 - 사용 목적별 최적 설정"""
    
    MINIMAL = {
        # 최소 설정 (초보자용/테스트용)
        'STRATEGIES_PER_COMBINATION': 10,
        'AZ_SELFPLAY_EPISODES': 50,
        'ENABLE_HYBRID': False,
        'ENABLE_AUTO_TRAINING': False,
        'MAX_WORKERS': 2,
        'BATCH_SIZE': 25,
    }
    
    DEVELOPMENT = {
        # 개발 환경 (테스트용)
        'STRATEGIES_PER_COMBINATION': 50,
        'AZ_SELFPLAY_EPISODES': 100,
        'ENABLE_HYBRID': False,
        'ENABLE_AUTO_TRAINING': True,
        'MAX_WORKERS': 4,
        'BATCH_SIZE': 50,
    }
    
    PRODUCTION = {
        # 프로덕션 환경 (실전용)
        'STRATEGIES_PER_COMBINATION': 500,
        'AZ_SELFPLAY_EPISODES': 200,
        'ENABLE_HYBRID': True,
        'ENABLE_AUTO_TRAINING': True,
        'MAX_WORKERS': 16,
        'BATCH_SIZE': 100,
    }


class Config:
    """모든 설정의 단일 출처"""
    
    def __init__(self, profile: str = 'auto', auto_detect: bool = True):
        """
        Args:
            profile: 'minimal', 'development', 'production', 'auto'
            auto_detect: 자동 감지 활성화 여부
        """
        # 기본 설정 로드
        self._load_from_env()
        
        # 프로파일 및 자동 감지 적용
        if profile == 'auto' and auto_detect:
            from rl_pipeline.core.config_auto import AutoConfigDetector
            optimal = AutoConfigDetector.generate_optimal_config()
            self._apply_optimal_config(optimal)
        elif profile in ('minimal', 'development', 'production'):
            profile_config = getattr(ConfigProfile, profile.upper())
            self._apply_profile_config(profile_config)
            if auto_detect:
                from rl_pipeline.core.config_auto import AutoConfigDetector
                optimal = AutoConfigDetector.generate_optimal_config()
                self._apply_optimal_config(optimal)
        
        # 설정 검증
        self._validate()
    
    def _load_from_env(self):
        """환경변수에서 설정 로드 (초기화 시 1회 실행)"""
        # 워크스페이스 경로
        self.WORKSPACE_ROOT = os.getenv('WORKSPACE_ROOT', '/workspace')
        self.AUTO_TRADER_ROOT = os.getenv('AUTO_TRADER_ROOT', '/workspace')
        self.RL_PIPELINE_ROOT = os.getenv('RL_PIPELINE_ROOT', '/workspace/rl_pipeline')
        
        # 성능 설정
        self.MAX_WORKERS = min(os.cpu_count() or 4, int(os.getenv('MAX_WORKERS', '16')))
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
        self.CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '300'))
        self.DB_TIMEOUT = float(os.getenv('DB_TIMEOUT', '30.0'))
        
        # 시뮬레이션 설정
        self.STRATEGIES_PER_COMBINATION = int(os.getenv('STRATEGIES_PER_COMBINATION', '500'))
        self.LONG_TERM_DAYS = int(os.getenv('LONG_TERM_DAYS', '30'))
        self.SHORT_TERM_DAYS = int(os.getenv('SHORT_TERM_DAYS', '14'))
        self.SIMULATION_SAMPLE_RATIO = float(os.getenv('SIMULATION_SAMPLE_RATIO', '0.1'))
        
        # 전략 생성 비율 설정 (Enhanced 모드)
        self.INTELLIGENT_STRATEGY_RATIO = float(os.getenv('INTELLIGENT_STRATEGY_RATIO', '0.5'))
        self.GRID_SEARCH_STRATEGY_RATIO = float(os.getenv('GRID_SEARCH_STRATEGY_RATIO', '0.2'))
        self.DIRECTION_SPECIALIZED_RATIO = float(os.getenv('DIRECTION_SPECIALIZED_RATIO', '0.3'))
        
        # 인터벌 설정
        self.UNIFIED_INTERVALS = ['15m', '30m', '240m', '1d']
        
        # 분석 설정
        self.ENABLE_FRACTAL_ANALYSIS = os.getenv('ENABLE_FRACTAL_ANALYSIS', 'true').lower() == 'true'
        self.ENABLE_DNA_EXTRACTION = os.getenv('ENABLE_DNA_EXTRACTION', 'true').lower() == 'true'
        
        # 성능 모니터링 설정
        self.PERFORMANCE_LOG_INTERVAL = int(os.getenv('PERFORMANCE_LOG_INTERVAL', '120'))
        
        # 데이터베이스 연결 풀 설정
        self.DB_MAX_CONNECTIONS = int(os.getenv('DB_MAX_CONNECTIONS', '50'))
        self.DB_CONNECTION_TIMEOUT = float(os.getenv('DB_CONNECTION_TIMEOUT', '60.0'))
        self.DB_BATCH_MAX_CONNECTIONS = int(os.getenv('DB_BATCH_MAX_CONNECTIONS', '200'))
        
        # 하이브리드 설정
        self.ENABLE_HYBRID = os.getenv('USE_HYBRID', 'false').lower() == 'true'
        self.ENABLE_AUTO_TRAINING = os.getenv('ENABLE_AUTO_TRAINING', 'false').lower() == 'true'

    # 🔥 동적 속성: 환경변수 변경을 실시간 반영 (엔진화 필수)
    @property
    def DATA_STORAGE_PATH(self):
        """데이터 저장소 경로 (동적)"""
        # 1. 환경변수 우선
        _env_storage = os.getenv('DATA_STORAGE_PATH')
        if _env_storage:
            return _env_storage
            
        # 2. 전략 DB 경로 기반 추론
        _strat_db = os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
        if _strat_db:
            return os.path.dirname(_strat_db)
            
        # 3. 폴백: market/coin_market/data_storage 우선 확인 (프로젝트 구조 인식)
        # 현재 위치에서 상대 경로로 market/coin_market 찾기 시도
        current_dir = os.getcwd()
        
        # case A: 루트에서 실행 시
        potential_path = os.path.join(current_dir, 'market', 'coin_market', 'data_storage')
        if os.path.exists(os.path.dirname(potential_path)): # coin_market 폴더가 있으면
            return potential_path
            
        # case B: market/coin_market 내부에서 실행 시 (이미 처리되겠지만)
        if 'coin_market' in current_dir:
            # 상위로 올라가서 data_storage 찾기 등 복잡한 로직보다는
            # 보통 run_learning.py가 환경변수를 설정하므로 여기까지 올 일이 적음
            pass

        # 4. 최후의 수단 (현재 디렉토리)
        return os.path.join(current_dir, 'data_storage')

    @property
    def RL_DB(self):
        """RL 캔들 DB 경로 (동적)"""
        return os.getenv('RL_DB_PATH', os.getenv('CANDLES_DB_PATH', os.path.join(self.DATA_STORAGE_PATH, 'rl_candles.db')))

    @property
    def STRATEGIES_DB(self):
        """전략 DB 경로 (동적 - 파일 또는 디렉토리)"""
        # 🔧 기본값을 디렉토리 모드로 변경 (learning_strategies 폴더)
        path = os.getenv('STRATEGY_DB_PATH', os.getenv('STRATEGIES_DB_PATH', os.path.join(self.DATA_STORAGE_PATH, 'learning_strategies')))
        
        # 🔥 강제 보정: rl_strategies.db가 경로에 포함되어 있으면 learning_strategies로 교체 (레거시 호환성)
        if 'rl_strategies.db' in path:
            path = path.replace('rl_strategies.db', 'learning_strategies')
            
        return path

    # 🔒 글로벌 전략용 예약어 (이 이름의 코인이 생기면 prefix 추가)
    RESERVED_DB_NAMES = {'common', 'global', 'shared', 'system', '_global'}
    
    def get_strategy_db_path(self, coin: str = None) -> str:
        """코인별 전략 DB 경로 반환 (Directory Mode 지원)
        
        Args:
            coin: 코인 심볼 (예: 'BTC', 'ETH')
            
        Returns:
            DB 파일 경로
        """
        base_path = self.STRATEGIES_DB
        
        # 1. 디렉토리 모드인지 확인 (확장자가 .db가 아니거나, 실제 디렉토리인 경우)
        is_directory = not base_path.endswith('.db') or os.path.isdir(base_path)
        
        if is_directory:
            if not coin:
                # 코인이 지정되지 않았는데 디렉토리 모드인 경우, 기본/공용 파일 반환
                return os.path.join(base_path, 'common_strategies.db')
            
            # 🔒 예약어 충돌 방지: common, global 등의 코인명은 prefix 추가
            coin_lower = coin.lower()
            if coin_lower in self.RESERVED_DB_NAMES:
                # 예: common → coin_common_strategies.db (글로벌용 common_strategies.db와 구분)
                return os.path.join(base_path, f"coin_{coin_lower}_strategies.db")
            
            # 코인별 파일명 생성 (소문자 변환)
            return os.path.join(base_path, f"{coin_lower}_strategies.db")
        
        # 2. 단일 파일 모드 (기존 호환성)
        return base_path

    @property
    def LEARNING_RESULTS_DB_PATH(self):
        """학습 결과 DB 경로 (동적) - 전략 DB와 통합됨"""
        base_path = self.STRATEGIES_DB
        
        # 디렉토리 모드인 경우, 공용 파일(common_strategies.db)을 반환
        # 학습 결과, 파이프라인 로그 등은 코인에 종속되지 않는 경우가 많거나,
        # 중앙에서 관리하는 것이 편하므로 공용 DB에 저장
        is_directory = not base_path.endswith('.db') or os.path.isdir(base_path)
        if is_directory:
            # common_strategies.db가 없으면 생성하도록 유도할 수 있지만,
            # 여기서는 경로만 반환
            return os.path.join(base_path, 'common_strategies.db')
            
        return base_path
    
    def _apply_profile_config(self, profile_config: Dict[str, Any]):
        """프로파일 설정 적용"""
        for key, value in profile_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _apply_optimal_config(self, optimal: Dict[str, Any]):
        """최적 설정 적용 (자동 감지 결과)"""
        for key, value in optimal.items():
            if key.startswith('detected_'):
                # 감지된 정보는 속성으로 저장
                setattr(self, key, value)
            elif hasattr(self, key):
                # 기존 속성이 있으면 업데이트 (환경변수보다 우선)
                setattr(self, key, value)
    
    def _validate(self):
        """설정 검증"""
        errors = []
        warnings = []
        
        # 필수 설정 확인
        if not self.STRATEGIES_DB:
            errors.append("STRATEGIES_DB_PATH 필수")
        
        # 권장 설정 확인
        if self.MAX_WORKERS > 32:
            warnings.append("MAX_WORKERS가 너무 큼 (권장: 16 이하)")
        elif self.MAX_WORKERS < 1:
            errors.append("MAX_WORKERS는 1 이상이어야 함")
        
        if self.BATCH_SIZE > 1000:
            warnings.append("BATCH_SIZE가 너무 큼 (권장: 50-200)")
        
        # 하이브리드 모드 확인
        if self.ENABLE_HYBRID:
            try:
                import jax
                devices = jax.devices()
                has_gpu = any('gpu' in str(d).lower() for d in devices)
                if not has_gpu:
                    warnings.append(
                        "하이브리드 모드 활성화되었으나 GPU를 감지할 수 없습니다. "
                        "CPU로 실행되면 성능이 저하될 수 있습니다."
                    )
            except ImportError:
                errors.append(
                    "하이브리드 모드 활성화되었으나 JAX가 설치되지 않았습니다. "
                    "pip install jax flax 실행 필요"
                )
        
        if errors:
            raise ValueError(f"설정 오류: {', '.join(errors)}")
        
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")
    
    def print_summary(self):
        """설정 요약 출력"""
        print("📋 현재 설정 요약:")
        print(f"  - CPU 코어: {getattr(self, 'detected_cpu_cores', 'N/A')}")
        print(f"  - 메모리: {getattr(self, 'detected_memory_gb', 'N/A'):.1f} GB" if hasattr(self, 'detected_memory_gb') else "  - 메모리: N/A")
        print(f"  - GPU: {'✅' if getattr(self, 'detected_gpu', False) else '❌'}")
        print(f"  - MAX_WORKERS: {self.MAX_WORKERS}")
        print(f"  - BATCH_SIZE: {self.BATCH_SIZE}")
        print(f"  - 하이브리드 모드: {'✅' if getattr(self, 'ENABLE_HYBRID', False) else '❌'}")

# 전역 설정 인스턴스 (레거시 호환성 - 기본 동작)
config = Config(profile='auto', auto_detect=True)

# 레거시 호환성을 위한 별칭 (이제는 동적 프로퍼티 접근)
# 주의: 모듈 레벨 변수는 import 시점에 고정되므로, 가능한 config.속성 으로 접근하는 것이 좋습니다.
WORKSPACE_ROOT = config.WORKSPACE_ROOT
AUTO_TRADER_ROOT = config.AUTO_TRADER_ROOT
RL_PIPELINE_ROOT = config.RL_PIPELINE_ROOT
# 아래 변수들은 이제 프로퍼티이므로 값 복사가 됨. 동적 반영을 위해선 config 객체 사용 권장
DATA_STORAGE_PATH = config.DATA_STORAGE_PATH
CANDLES_DB_PATH = config.RL_DB
STRATEGIES_DB_PATH = config.STRATEGIES_DB
WORK_DIR = config.AUTO_TRADER_ROOT
