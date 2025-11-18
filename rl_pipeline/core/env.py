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
        """환경변수에서 설정 로드"""
        # 데이터베이스 경로
        self.RL_DB = os.getenv('CANDLES_DB_PATH', '/workspace/data_storage/rl_candles.db')
        self.STRATEGIES_DB = os.getenv('STRATEGIES_DB_PATH', '/workspace/data_storage/rl_strategies.db')
        # learning_results.db는 이제 rl_strategies.db로 통합됨
        self.LEARNING_RESULTS_DB_PATH = self.STRATEGIES_DB
        
        # 워크스페이스 경로
        self.WORKSPACE_ROOT = os.getenv('WORKSPACE_ROOT', '/workspace')
        self.AUTO_TRADER_ROOT = os.getenv('AUTO_TRADER_ROOT', '/workspace')
        self.RL_PIPELINE_ROOT = os.getenv('RL_PIPELINE_ROOT', '/workspace/rl_pipeline')
        self.DATA_STORAGE_PATH = os.getenv('DATA_STORAGE_PATH', '/workspace/data_storage')
        
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

# 레거시 호환성을 위한 별칭
WORKSPACE_ROOT = config.WORKSPACE_ROOT
AUTO_TRADER_ROOT = config.AUTO_TRADER_ROOT
RL_PIPELINE_ROOT = config.RL_PIPELINE_ROOT
DATA_STORAGE_PATH = config.DATA_STORAGE_PATH
CANDLES_DB_PATH = config.RL_DB
STRATEGIES_DB_PATH = config.STRATEGIES_DB
WORK_DIR = config.AUTO_TRADER_ROOT
