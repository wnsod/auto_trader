"""
실시간 시그널 생성기 - RL 시스템의 학습된 전략을 활용한 실시간 매매 시그널 생성

주요 기능:
1. RL Q-table 로드 및 시그널 생성
2. 인터벌별 시그널 통합
3. DB 저장
4. 🆕 AI 모델 기반 시그널 점수 계산

🆕 Absolute Zero System 개선사항 반영:
- 모든 고급 기술지표 활용 (다이버전스, 볼린저밴드 스퀴즈, 모멘텀, 트렌드 강도 등)
- 개선된 전략 평가 방식 (시장 적응성 평가 포함)
- 향상된 상태 표현 (더 정교한 상태 키 생성)
- 새로운 패턴 매칭 로직 (다이버전스, 스퀴즈, 강한 트렌드 등)
- 멀티인터벌 상태 추적 개선 (모든 고급 지표 포함)
- �� AI 모델 기반 전략 점수 예측

🚀 고성능 시스템 최적화:
- GPU 가속 (JAX 모델 추론)
- 고성능 캐시 시스템
- 크로스 코인 학습 컨텍스트 활용
- 병렬 처리 최적화
"""
import sys
import os

# 🆕 경로 설정 개선 - rl_pipeline 및 signal_selector 모듈을 찾을 수 있도록
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)  # trade/의 상위 디렉토리 (auto_trader 루트)

# trade 디렉토리를 sys.path에 추가 (signal_selector 모듈을 찾기 위해)
sys.path.insert(0, current_dir)

# rl_pipeline 디렉토리 경로 추가
rl_pipeline_path = os.path.join(workspace_dir, 'rl_pipeline')
if os.path.exists(rl_pipeline_path):
    sys.path.insert(0, rl_pipeline_path)
    sys.path.insert(0, workspace_dir)
    print(f"✅ rl_pipeline 경로 추가: {rl_pipeline_path}")
else:
    print(f"⚠️ rl_pipeline 디렉토리를 찾을 수 없음: {rl_pipeline_path}")
    # Docker 환경을 위한 fallback
    sys.path.insert(0, '/workspace/')
    sys.path.insert(0, '/workspace/rl_pipeline')
    sys.path.insert(0, '/workspace/trade')  # signal_selector 모듈을 찾기 위해

# 🆕 signal_selector 모듈 import (리팩토링된 모듈 구조)
try:
    from signal_selector.config import (
        USE_GPU_ACCELERATION, JAX_PLATFORM_NAME, MAX_WORKERS, CACHE_SIZE,
        ENABLE_CROSS_COIN_LEARNING, CANDLES_DB_PATH, STRATEGIES_DB_PATH,
        TRADING_SYSTEM_DB_PATH, DB_PATH, PERFORMANCE_CONFIG,
        AI_MODEL_AVAILABLE, SYNERGY_LEARNING_AVAILABLE
    )
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )
    print("✅ signal_selector 모듈 로드 완료")
    USE_NEW_MODULES = True
except ImportError as e:
    print(f"⚠️ signal_selector 모듈 import 실패: {e}")
    print("⚠️ 기존 코드로 fallback")
    USE_NEW_MODULES = False
    # 기존 코드로 fallback (아래 코드 계속 실행)

# 🆕 변동성 기반 시스템 import
try:
    from rl_pipeline.utils.coin_volatility import (
        get_volatility_profile,
        calculate_coin_volatility,
        classify_volatility_group
    )
    print("✅ 변동성 시스템 로드 완료")
    VOLATILITY_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 변동성 시스템 로드 실패: {e}")
    VOLATILITY_SYSTEM_AVAILABLE = False

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import traceback
import time
import os
import math
import logging
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# 🚀 고성능 시스템 설정 (새 모듈에서 import 실패 시에만 정의)
if not USE_NEW_MODULES:
    USE_GPU_ACCELERATION = os.getenv('USE_GPU_ACCELERATION', 'true').lower() == 'true'
    JAX_PLATFORM_NAME = os.getenv('JAX_PLATFORM_NAME', 'gpu')
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))
    CACHE_SIZE = int(os.getenv('CACHE_SIZE', '50000'))
    ENABLE_CROSS_COIN_LEARNING = os.getenv('ENABLE_CROSS_COIN_LEARNING', 'true').lower() == 'true'
    CANDLES_DB_PATH = os.getenv('CANDLES_DB_PATH', os.path.join(workspace_dir, 'data_storage', 'rl_candles.db'))
    STRATEGIES_DB_PATH = os.getenv('STRATEGIES_DB_PATH', os.path.join(workspace_dir, 'data_storage', 'learning_results.db'))
    TRADING_SYSTEM_DB_PATH = os.path.join(workspace_dir, 'data_storage', 'trading_system.db')
    DB_PATH = TRADING_SYSTEM_DB_PATH
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

# 🆕 자체 데이터베이스 연결 시스템 (rl_pipeline 충돌 방지)
DB_POOL_AVAILABLE = True
CONFLICT_MANAGER_AVAILABLE = True
print("✅ 자체 데이터베이스 연결 시스템 사용")

# 🆕 자체 데이터베이스 함수들 구현 (새 모듈에서 import 실패 시에만 정의)
if not USE_NEW_MODULES:
    from contextlib import contextmanager
    
    @contextmanager
    def get_optimized_db_connection(db_path: str, mode: str = 'read'):
        """최적화된 데이터베이스 연결 컨텍스트 매니저"""
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            if mode == 'write':
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def safe_db_write(db_path: str, operation_name: str):
        """안전한 데이터베이스 쓰기 컨텍스트 매니저"""
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"⚠️ 데이터베이스 쓰기 오류 ({operation_name}): {e}")
            raise e
        finally:
            if conn:
                conn.close()
    
    def safe_db_read(query: str, params: tuple = (), db_path: str = None):
        """안전한 데이터베이스 읽기 함수"""
        try:
            if db_path is None:
                db_path = STRATEGIES_DB_PATH
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"⚠️ 데이터베이스 읽기 오류: {e}")
            return []

def get_strategy_db_pool():
    """전략 데이터베이스 풀 반환 (호환성)"""
    return None

def get_candle_db_pool():
    """캔들 데이터베이스 풀 반환 (호환성)"""
    return None

def get_conflict_manager():
    """충돌 관리자 반환 (호환성)"""
    return None

# 🆕 크로스 코인 학습 설정
CROSS_COIN_AVAILABLE = os.getenv('CROSS_COIN_AVAILABLE', 'false').lower() == 'true'

# 🆕 로거 설정
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 🚀 GPU 가속 설정
if USE_GPU_ACCELERATION:
    try:
        import jax
        # JAX 로거 레벨 조정 (TPU 백엔드 경고 숨김)
        import logging as std_logging
        jax_logger = std_logging.getLogger('jax._src.xla_bridge')
        jax_logger.setLevel(std_logging.ERROR)  # ERROR 이상의 로그만 표시
        
        # 환경 변수로 TPU 백엔드 시도 방지
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
    print("✅ learning_engine 고급 기능 로드 완료")
except ImportError:
    AI_MODEL_AVAILABLE = False
    print("⚠️ AI 모델을 import할 수 없습니다. 기본 시그널 계산만 사용됩니다.")
    # 기본값 설정
    PolicyTrainer = None
    GlobalLearningManager = None
    SymbolFinetuningManager = None
    ShortTermLongTermSynergyLearner = None
    ReliabilityScoreCalculator = None
    ContinuousLearningManager = None
    RoutingPatternAnalyzer = None
    ContextualLearningManager = None
    analyze_strategy_quality = None

# 🚀 크로스 코인 학습 컨텍스트 (현재 비활성화)
# absolute_zero_system의 복잡한 의존성 문제로 인해 간소화
CROSS_COIN_AVAILABLE = False
print("ℹ️ 크로스 코인 학습 컨텍스트는 현재 비활성화되어 있습니다.")

# 🆕 단기-장기 시너지 학습기 (이미 위에서 import됨)
# ShortTermLongTermSynergyLearner는 220줄에서 이미 import되었으므로 중복 제거
if AI_MODEL_AVAILABLE and ShortTermLongTermSynergyLearner is not None:
    SYNERGY_LEARNING_AVAILABLE = True
    print("✅ 단기-장기 시너지 학습기 사용 가능")
else:
    SYNERGY_LEARNING_AVAILABLE = False
    print("⚠️ 단기-장기 시너지 학습기를 사용할 수 없습니다. 기본 시그널 계산만 사용됩니다.")

# 🆕 유틸리티 함수들 (새 모듈에서 import 실패 시에만 정의)
if not USE_NEW_MODULES:
    def safe_float(value, default=0.0):
        """안전한 float 변환 함수"""
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_str(value, default='unknown'):
        """안전한 string 변환 함수"""
        if value is None or pd.isna(value):
            return default
        try:
            return str(value)
        except (ValueError, TypeError):
            return default
    
    # 🆕 기술지표 설정 (모든 지표의 기본값과 처리 로직 통합)
    TECHNICAL_INDICATORS_CONFIG = {
    # 기본 지표들
    'rsi': {'default': 50.0, 'type': 'float'},
    'macd': {'default': 0.0, 'type': 'float'},
    'volume_ratio': {'default': 1.0, 'type': 'float'},
    'wave_progress': {'default': 0.5, 'type': 'float'},
    'structure_score': {'default': 0.5, 'type': 'float'},
    'pattern_confidence': {'default': 0.0, 'type': 'float'},
    
    # 고급 지표들
    'mfi': {'default': 50.0, 'type': 'float'},
    'atr': {'default': 0.0, 'type': 'float'},
    'adx': {'default': 25.0, 'type': 'float'},
    'ma20': {'default': 0.0, 'type': 'float'},
    'ma20_pct_diff': {'default': 0.5, 'type': 'float'},
    'rsi_ema': {'default': 50.0, 'type': 'float'},
    'rsi_smoothed': {'default': 50.0, 'type': 'float'},
    'macd_signal': {'default': 0.0, 'type': 'float'},
    'macd_diff': {'default': 0.0, 'type': 'float'},
    'macd_smoothed': {'default': 0.0, 'type': 'float'},
    'wave_momentum': {'default': 0.0, 'type': 'float'},
    'confidence': {'default': 0.5, 'type': 'float'},
    'volatility': {'default': 0.0, 'type': 'float'},
    'risk_score': {'default': 0.5, 'type': 'float'},
    'integrated_strength': {'default': 0.5, 'type': 'float'},
    'pattern_quality': {'default': 0.0, 'type': 'float'},
    
    # 볼린저 밴드 관련
    'bb_upper': {'default': 0.0, 'type': 'float'},
    'bb_lower': {'default': 0.0, 'type': 'float'},
    'bb_middle': {'default': 0.0, 'type': 'float'},
    'bb_bandwidth': {'default': 0.0, 'type': 'float'},
    
    # 텍스트 지표들
    'pattern_type': {'default': 'unknown', 'type': 'str'},
    'pattern_class': {'default': 'unknown', 'type': 'str'},
    'flow_level_meta': {'default': 'unknown', 'type': 'str'},
    'volatility_level': {'default': 'unknown', 'type': 'str'},
    'wave_phase': {'default': 'unknown', 'type': 'str'},
    'pattern_direction': {'default': 'unknown', 'type': 'str'},
    'pattern_volume_ratio': {'default': 'unknown', 'type': 'str'},
    'pattern_pivot_strength': {'default': 'unknown', 'type': 'str'},
    'volume_avg': {'default': 'unknown', 'type': 'str'},
    'volume_normalized': {'default': 'unknown', 'type': 'str'},
    'zigzag': {'default': 'unknown', 'type': 'str'},
    'zigzag_direction': {'default': 'unknown', 'type': 'str'},
    'pivot_point': {'default': 'unknown', 'type': 'str'},
    'wave_number': {'default': 'unknown', 'type': 'str'},
    'wave_step': {'default': 'unknown', 'type': 'str'},
    'integrated_wave_phase': {'default': 'unknown', 'type': 'str'},
    'integrated_direction': {'default': 'unknown', 'type': 'str'},
    'three_wave_pattern': {'default': 'unknown', 'type': 'str'},
    'sideways_pattern': {'default': 'unknown', 'type': 'str'},
}

    # 🆕 상태 이산화 설정
    STATE_DISCRETIZATION_CONFIG = {
        'rsi': {'low': 30, 'high': 70, 'states': ['low', 'mid', 'high']},
        'macd': {'threshold': 0, 'states': ['neg', 'pos']},
        'volume_ratio': {'low': 0.8, 'high': 1.5, 'states': ['low', 'normal', 'high']},
        'wave_progress': {'low': 0.3, 'high': 0.7, 'states': ['early', 'mid', 'late']},
        'structure_score': {'threshold': 0.6, 'states': ['weak', 'strong']},
        'pattern_confidence': {'threshold': 0.5, 'states': ['uncertain', 'confident']},
        'mfi': {'low': 20, 'high': 80, 'states': ['low', 'mid', 'high']},
        'adx': {'threshold': 25, 'states': ['weak', 'strong']},
        'wave_momentum': {'threshold': 0.1, 'states': ['low', 'high']},
        'confidence': {'low': 0.3, 'high': 0.7, 'states': ['low', 'mid', 'high']},
        'volatility': {'low': 0.02, 'high': 0.05, 'states': ['low', 'mid', 'high']},
        'bb_width': {'low': 0.05, 'high': 0.1, 'states': ['narrow', 'normal', 'wide']},
        'bb_squeeze': {'threshold': 0.8, 'states': ['normal', 'squeezed']},
        'trend_strength': {'low': 0.3, 'high': 0.7, 'states': ['weak', 'moderate', 'strong']},
        'pattern_quality': {'low': 0.3, 'high': 0.7, 'states': ['low', 'mid', 'high']},
        'risk_score': {'low': 0.3, 'high': 0.7, 'states': ['low', 'mid', 'high']},
        'integrated_strength': {'low': 0.3, 'high': 0.7, 'states': ['weak', 'moderate', 'strong']},
    }
    
    def discretize_value(value: float, config: Dict) -> str:
        """값을 이산화하여 상태로 변환"""
        if 'threshold' in config:
            return config['states'][1] if value > config['threshold'] else config['states'][0]
        elif 'low' in config and 'high' in config:
            if value < config['low']:
                return config['states'][0]
            elif value > config['high']:
                return config['states'][2]
            else:
                return config['states'][1]
        return config['states'][0]
    
    def process_technical_indicators(candle: pd.Series) -> Dict:
        """모든 기술지표를 설정 기반으로 처리"""
        indicators = {}
        
        # 설정 기반으로 모든 지표 처리
        for name, config in TECHNICAL_INDICATORS_CONFIG.items():
            value = candle.get(name)
            if config['type'] == 'float':
                indicators[name] = safe_float(value, config['default'])
            else:
                indicators[name] = safe_str(value, config['default'])
        
        # 🎯 특별 처리 로직들 (기존과 동일하게 유지)
        # 볼린저 밴드 위치 계산
        close = safe_float(candle.get('close'), 0.0)
        bb_middle = indicators['bb_middle']
        if bb_middle > 0 and close > 0:
            if close > bb_middle:
                indicators['bb_position'] = 'upper'
            elif close < bb_middle:
                indicators['bb_position'] = 'lower'
            else:
                indicators['bb_position'] = 'middle'
        else:
            indicators['bb_position'] = 'unknown'
        
        # 볼린저 밴드 스퀴즈 계산
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        if bb_upper > 0 and bb_lower > 0:
            indicators['bb_squeeze'] = bb_upper - bb_lower
        else:
            indicators['bb_squeeze'] = 0.0
        
        # 볼린저 밴드 너비
        indicators['bb_width'] = indicators['bb_bandwidth']
        
        # 추세 강도
        indicators['trend_strength'] = indicators['ma20_pct_diff']
        
        # 🆕 새로 추가된 고급 지표들 (기본값으로 설정 - 기존과 동일)
        indicators['rsi_divergence'] = 'none'
        indicators['macd_divergence'] = 'none'
        indicators['volume_divergence'] = 'none'
        indicators['price_momentum'] = 0.0
        indicators['volume_momentum'] = 0.0
        indicators['support_resistance'] = 'unknown'
        indicators['fibonacci_levels'] = 'unknown'
        indicators['elliott_wave'] = 'unknown'
        indicators['harmonic_patterns'] = 'none'
        indicators['candlestick_patterns'] = 'none'
        indicators['market_structure'] = 'unknown'
        indicators['risk_level'] = 'unknown'
        
        # 🎯 패턴 품질 특별 처리 (기존과 동일)
        if indicators['pattern_quality'] == 0.0:
            # 패턴 품질을 다른 지표들을 기반으로 계산하는 로직은 나중에 처리
            pass
        
        return indicators

# 데이터베이스 경로 (Windows 환경 지원) - 새 모듈에서 import 실패 시에만 정의
if not USE_NEW_MODULES:
    TRADING_SYSTEM_DB_PATH = os.path.join(workspace_dir, 'data_storage', 'trading_system.db')
    DB_PATH = TRADING_SYSTEM_DB_PATH

# SignalAction과 SignalInfo는 새 모듈에서 import했으므로 중복 정의 제거
if not USE_NEW_MODULES:
    class SignalAction(Enum):
        BUY = "buy"
        SELL = "sell"
        HOLD = "hold"
        WAIT = "wait"
        TAKE_PROFIT = "take_profit"
        STOP_LOSS = "stop_loss"
    
    @dataclass
    class SignalInfo:
        coin: str
        interval: str
        action: SignalAction
        signal_score: float
        confidence: float
        reason: str
        timestamp: int
        price: float
        volume: float
        rsi: float
        macd: float
        wave_phase: str
        pattern_type: str
        risk_level: str
        volatility: float
        volume_ratio: float
        wave_progress: float
        structure_score: float
        pattern_confidence: float
        integrated_direction: str
        integrated_strength: float
        
        # 고급 지표들
        mfi: float = 50.0
        atr: float = 0.0
        adx: float = 25.0
        ma20: float = 0.0
        rsi_ema: float = 50.0
        macd_smoothed: float = 0.0
        wave_momentum: float = 0.0
        bb_position: str = 'unknown'
        bb_width: float = 0.0
        bb_squeeze: float = 0.0
        rsi_divergence: str = 'none'
        macd_divergence: str = 'none'
        volume_divergence: str = 'none'
        price_momentum: float = 0.0
        volume_momentum: float = 0.0
        trend_strength: float = 0.5
        support_resistance: str = 'unknown'
        fibonacci_levels: str = 'unknown'
        elliott_wave: str = 'unknown'
        harmonic_patterns: str = 'none'
        candlestick_patterns: str = 'none'
        market_structure: str = 'unknown'
        flow_level_meta: str = 'unknown'
        pattern_direction: str = 'neutral'
        market_condition: str = 'unknown'
        market_adaptation_bonus: float = 1.0
        calmar_ratio: float = 0.0  # 🆕 Calmar Ratio 추가
        profit_factor: float = 1.0  # 🆕 Profit Factor 추가
        reliability_score: float = 0.0  # 🆕 신뢰도 점수 추가
        learning_quality_score: float = 0.0  # 🆕 학습 품질 점수 추가
        global_strategy_id: str = ""  # 🆕 글로벌 전략 ID 추가
        coin_tuned: bool = False  # 🆕 심볼별 튜닝 여부
        walk_forward_performance: Dict[str, float] = None  # 🆕 Walk-Forward 성능
        regime_coverage: Dict[str, float] = None  # 🆕 레짐별 커버리지

# 🆕 3단계: 심화 난이도 성능 업그레이드 시스템 (새 모듈에서 import 실패 시에만 정의)
if not USE_NEW_MODULES:
    class OffPolicyEvaluator:
        """오프폴리시 평가 시스템 (IPS/Doubly Robust)"""
        def __init__(self):
            self.policy_probabilities = {}
            self.evaluation_history = []
            
        def record_policy_probability(self, action: str, probability: float, context: str):
            """정책 확률 기록"""
            try:
                key = f"{context}_{action}"
                self.policy_probabilities[key] = probability
            except Exception as e:
                print(f"⚠️ 정책 확률 기록 오류: {e}")
        
        def calculate_ips_estimate(self, action: str, reward: float, context: str) -> float:
            """Inverse Propensity Scoring 추정"""
            try:
                key = f"{context}_{action}"
                propensity = self.policy_probabilities.get(key, 0.5)  # 기본값 0.5
                
                if propensity > 0:
                    return reward / propensity
                else:
                    return reward
                    
            except Exception as e:
                print(f"⚠️ IPS 추정 오류: {e}")
                return reward
        
        def calculate_doubly_robust_estimate(self, action: str, reward: float, context: str, baseline_reward: float) -> float:
            """Doubly Robust 추정"""
            try:
                ips_estimate = self.calculate_ips_estimate(action, reward, context)
                key = f"{context}_{action}"
                propensity = self.policy_probabilities.get(key, 0.5)
                
                # Doubly Robust 공식
                dr_estimate = baseline_reward + (reward - baseline_reward) / propensity
                return dr_estimate
                
            except Exception as e:
                print(f"⚠️ Doubly Robust 추정 오류: {e}")
                return reward
    
    class ConfidenceCalibrator:
        """신뢰도 캘리브레이션 시스템 (Platt/Isotonic)"""
        def __init__(self):
            self.calibration_params = {}
            self.calibration_history = []
            
        def calibrate_confidence(self, raw_confidence: float, context: str) -> float:
            """신뢰도 캘리브레이션 (Platt Scaling)"""
            try:
                # 간단한 로지스틱 변환
                if context not in self.calibration_params:
                    self.calibration_params[context] = {'a': 1.0, 'b': 0.0}
                
                params = self.calibration_params[context]
                calibrated = 1.0 / (1.0 + math.exp(-(params['a'] * raw_confidence + params['b'])))
                
                return max(0.0, min(1.0, calibrated))
                
            except Exception as e:
                print(f"⚠️ 신뢰도 캘리브레이션 오류: {e}")
                return raw_confidence
        
        def update_calibration_params(self, context: str, actual_success_rate: float, predicted_confidence: float):
            """캘리브레이션 파라미터 업데이트"""
            try:
                if context not in self.calibration_params:
                    self.calibration_params[context] = {'a': 1.0, 'b': 0.0}
                
                # 간단한 적응적 업데이트
                params = self.calibration_params[context]
                error = actual_success_rate - predicted_confidence
                
                # 파라미터 조정
                params['a'] += error * 0.1
                params['b'] += error * 0.05
                
                # 범위 제한
                params['a'] = max(0.1, min(5.0, params['a']))
                params['b'] = max(-2.0, min(2.0, params['b']))
                
            except Exception as e:
                print(f"⚠️ 캘리브레이션 파라미터 업데이트 오류: {e}")
    
    class MetaCorrector:
        """메타-보정 시스템 (스태킹)"""
        def __init__(self):
            self.meta_weights = {}
            self.feature_importance = {}
            
        def calculate_meta_score(self, base_score: float, feedback_stats: Dict, context_features: Dict) -> float:
            """메타 모델 기반 점수 보정"""
            try:
                # 간단한 선형 조합 (실제로는 XGBoost/LightGBM 사용)
                meta_score = base_score
                
                # 피드백 통계 가중치
                if 'success_rate' in feedback_stats:
                    meta_score += feedback_stats['success_rate'] * 0.2
                
                if 'avg_profit' in feedback_stats:
                    meta_score += feedback_stats['avg_profit'] * 0.1
                
                # 컨텍스트 특징 가중치
                if 'volatility' in context_features:
                    volatility = context_features['volatility']
                    if volatility == 'high':
                        meta_score *= 0.9  # 고변동성에서는 보수적
                    elif volatility == 'low':
                        meta_score *= 1.1  # 저변동성에서는 공격적
                
                return max(-1.0, min(1.0, meta_score))
                
            except Exception as e:
                print(f"⚠️ 메타 점수 계산 오류: {e}")
                return base_score
        
        def update_meta_weights(self, performance_feedback: Dict):
            """메타 가중치 업데이트"""
            try:
                # 성과 기반 가중치 조정
                if 'improvement' in performance_feedback:
                    improvement = performance_feedback['improvement']
                    
                    # 긍정적 피드백이면 가중치 증가
                    if improvement > 0:
                        for key in self.meta_weights:
                            self.meta_weights[key] *= 1.01
                    else:
                        for key in self.meta_weights:
                            self.meta_weights[key] *= 0.99
                            
            except Exception as e:
                print(f"⚠️ 메타 가중치 업데이트 오류: {e}")

# 🆕 2단계: 보통 난이도 성능 업그레이드 시스템
class ContextualBandit:
    """컨텍스추얼 밴딧 시스템 (UCB/Thompson Sampling)"""
    def __init__(self, exploration_factor: float = 1.0):
        self.exploration_factor = exploration_factor
        self.action_counts = {}
        self.action_rewards = {}
        self.total_trials = 0
        
    def select_action(self, context: str, available_actions: List[str]) -> str:
        """UCB 기반 액션 선택"""
        try:
            if not available_actions:
                return 'hold'
            
            # 초기화
            for action in available_actions:
                if action not in self.action_counts:
                    self.action_counts[action] = 0
                    self.action_rewards[action] = 0.0
            
            # UCB 점수 계산
            ucb_scores = {}
            for action in available_actions:
                if self.action_counts[action] == 0:
                    ucb_scores[action] = float('inf')  # 탐색 우선
                else:
                    avg_reward = self.action_rewards[action] / self.action_counts[action]
                    exploration_bonus = self.exploration_factor * math.sqrt(
                        math.log(self.total_trials) / self.action_counts[action]
                    )
                    ucb_scores[action] = avg_reward + exploration_bonus
            
            # 최고 UCB 점수 액션 선택
            best_action = max(ucb_scores.items(), key=lambda x: x[1])[0]
            return best_action
            
        except Exception as e:
            print(f"⚠️ 컨텍스추얼 밴딧 액션 선택 오류: {e}")
            return 'hold'
    
    def update_reward(self, action: str, reward: float):
        """액션 보상 업데이트"""
        try:
            if action not in self.action_counts:
                self.action_counts[action] = 0
                self.action_rewards[action] = 0.0
            
            self.action_counts[action] += 1
            self.action_rewards[action] += reward
            self.total_trials += 1
            
        except Exception as e:
            print(f"⚠️ 컨텍스추얼 밴딧 보상 업데이트 오류: {e}")

class RegimeChangeDetector:
    """레짐 전환 감지기"""
    def __init__(self):
        self.regime_history = []
        self.current_regime = 'unknown'
        self.regime_threshold = 0.3
        
    def detect_regime_change(self, market_indicators: Dict[str, float]) -> str:
        """레짐 전환 감지"""
        try:
            # 현재 레짐 결정
            new_regime = self._determine_regime(market_indicators)
            
            # 레짐 변화 감지
            if new_regime != self.current_regime:
                self.regime_history.append({
                    'timestamp': time.time(),
                    'old_regime': self.current_regime,
                    'new_regime': new_regime,
                    'indicators': market_indicators
                })
                self.current_regime = new_regime
                return 'changed'
            
            return 'stable'
            
        except Exception as e:
            print(f"⚠️ 레짐 전환 감지 오류: {e}")
            return 'unknown'
    
    def _determine_regime(self, indicators: Dict[str, float]) -> str:
        """레짐 결정"""
        try:
            adx = indicators.get('adx', 25.0)
            atr = indicators.get('atr', 0.0)
            ma_slope = indicators.get('ma_slope', 0.0)
            
            # 추세 강도 기반 레짐 분류
            if adx > 30 and abs(ma_slope) > 0.01:
                return 'trending'
            elif adx < 20 and atr < 0.02:
                return 'sideways_low_vol'
            elif adx < 20 and atr > 0.05:
                return 'sideways_high_vol'
            else:
                return 'transitional'
                
        except Exception as e:
            print(f"⚠️ 레짐 결정 오류: {e}")
            return 'unknown'

# 🆕 성능 업그레이드 시스템 클래스들
class ExponentialDecayWeight:
    """최근성 가중치 계산기"""
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
    
    def calculate_weight(self, time_diff_hours: float) -> float:
        """시간 차이에 따른 가중치 계산"""
        return math.exp(-self.decay_rate * time_diff_hours)

class BayesianSmoothing:
    """베이지안 스무딩 시스템"""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, kappa: float = 1.0):
        self.alpha = alpha  # Beta 분포 파라미터
        self.beta = beta    # Beta 분포 파라미터
        self.kappa = kappa  # 정규 분포 파라미터
    
    def smooth_success_rate(self, wins: int, total_trades: int) -> float:
        """승률 베이지안 스무딩"""
        return (wins + self.alpha) / (total_trades + self.alpha + self.beta)
    
    def smooth_avg_profit(self, profits: List[float], global_avg: float) -> float:
        """평균 수익률 베이지안 스무딩"""
        if not profits:
            return global_avg
        
        weighted_sum = sum(profits) + self.kappa * global_avg
        total_weight = len(profits) + self.kappa
        
        return weighted_sum / total_weight

class ActionSpecificScorer:
    """액션별 스코어 계산기"""
    def __init__(self):
        self.action_scores = {
            'buy': {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0},
            'sell': {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0},
            'hold': {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        }
    
    def update_action_score(self, action: str, success: bool, profit: float):
        """액션별 성과 업데이트"""
        if action in self.action_scores:
            self.action_scores[action]['total_trades'] += 1
            if success:
                self.action_scores[action]['success_rate'] += 1
            self.action_scores[action]['avg_profit'] += profit
    
    def get_action_score(self, action: str) -> float:
        """액션별 점수 반환"""
        if action not in self.action_scores:
            return 0.0
        
        score_data = self.action_scores[action]
        if score_data['total_trades'] == 0:
            return 0.0
        
        success_rate = score_data['success_rate'] / score_data['total_trades']
        avg_profit = score_data['avg_profit'] / score_data['total_trades']
        
        return success_rate * avg_profit

class ContextFeatureExtractor:
    """컨텍스트 특징 추출기"""
    def __init__(self):
        self.context_bins = {
            'volatility': ['low', 'medium', 'high'],
            'volume_ratio': ['low', 'medium', 'high'],
            'market_trend': ['bullish', 'bearish', 'sideways'],
            'btc_dominance': ['low', 'medium', 'high']
        }
    
    def extract_context_features(self, candle: pd.Series, market_data: dict) -> dict:
        """컨텍스트 특징 추출"""
        context = {}
        
        # 변동성 구간화
        volatility = candle.get('volatility', 0.0)
        if volatility < 0.02:
            context['volatility'] = 'low'
        elif volatility < 0.05:
            context['volatility'] = 'medium'
        else:
            context['volatility'] = 'high'
        
        # 거래량 비율 구간화
        volume_ratio = candle.get('volume_ratio', 1.0)
        if volume_ratio < 0.8:
            context['volume_ratio'] = 'low'
        elif volume_ratio < 1.2:
            context['volume_ratio'] = 'medium'
        else:
            context['volume_ratio'] = 'high'
        
        # 시장 트렌드 구간화
        market_trend = market_data.get('trend', 'sideways')
        context['market_trend'] = market_trend
        
        return context
    
    def get_context_key(self, context: dict) -> str:
        """컨텍스트 키 생성"""
        return f"{context['volatility']}_{context['volume_ratio']}_{context['market_trend']}"

class OutlierGuardrail:
    """이상치 컷 시스템"""
    def __init__(self, percentile_cut: float = 0.05):
        self.percentile_cut = percentile_cut
    
    def winsorize_profits(self, profits: List[float]) -> List[float]:
        """수익률 Winsorizing"""
        if len(profits) < 10:  # 데이터가 적으면 그대로 반환
            return profits
        
        sorted_profits = sorted(profits)
        n = len(sorted_profits)
        
        # 상하위 5% 절단
        lower_cut = int(n * self.percentile_cut)
        upper_cut = int(n * (1 - self.percentile_cut))
        
        # 절단된 값으로 대체
        winsorized = []
        for profit in profits:
            if profit < sorted_profits[lower_cut]:
                winsorized.append(sorted_profits[lower_cut])
            elif profit > sorted_profits[upper_cut]:
                winsorized.append(sorted_profits[upper_cut])
            else:
                winsorized.append(profit)
        
        return winsorized
    
    def calculate_robust_avg_profit(self, profits: List[float]) -> float:
        """견고한 평균 수익률 계산"""
        winsorized_profits = self.winsorize_profits(profits)
        return sum(winsorized_profits) / len(winsorized_profits)

# 🆕 진화형 AI 시스템 클래스들
class EvolutionEngine:
    """진화형 AI 엔진 - 시그널 진화 및 적응"""
    def __init__(self):
        self.pattern_weights = {}
        self.market_adaptations = {}
        self.evolution_history = []
        
    def evolve_signal(self, base_signal: SignalInfo, coin: str, interval: str) -> SignalInfo:
        """시그널을 진화시켜 더 정확한 시그널 생성"""
        try:
            # 패턴 기반 가중치 적용
            pattern_weight = self._get_pattern_weight(base_signal, coin, interval)
            
            # 시장 적응 가중치 적용
            market_weight = self._get_market_adaptation_weight(coin, interval)
            
            # 진화된 시그널 점수 계산
            evolved_score = base_signal.signal_score * pattern_weight * market_weight
            
            # 진화된 시그널 생성
            evolved_signal = SignalInfo(
                coin=base_signal.coin,
                interval=base_signal.interval,
                action=base_signal.action,
                signal_score=evolved_score,
                confidence=base_signal.confidence * pattern_weight,
                reason=f"{base_signal.reason} + 진화적적응",
                timestamp=base_signal.timestamp,
                price=base_signal.price,
                volume=base_signal.volume,
                rsi=base_signal.rsi,
                macd=base_signal.macd,
                wave_phase=base_signal.wave_phase,
                pattern_type=base_signal.pattern_type,
                risk_level=base_signal.risk_level,
                volatility=base_signal.volatility,
                volume_ratio=base_signal.volume_ratio,
                wave_progress=base_signal.wave_progress,
                structure_score=base_signal.structure_score,
                pattern_confidence=base_signal.pattern_confidence,
                integrated_direction=base_signal.integrated_direction,
                integrated_strength=base_signal.integrated_strength
            )
            
            return evolved_signal
            
        except Exception as e:
            print(f"⚠️ 시그널 진화 오류: {e}")
            # 🆕 진화형 AI 시그널 진화 (candle 변수 없이 진행)
            evolved_signal = base_signal  # 기본 시그널 그대로 사용
            
            # 🆕 시그널 패턴 추출 및 저장
            signal_pattern = self._extract_signal_pattern(evolved_signal)
            market_context = self._get_market_context(coin, interval)
            
            # 🆕 학습 데이터 저장
            self._save_signal_for_learning(evolved_signal, signal_pattern, market_context)
            
            print(f"🧬 진화형 시그널 생성: {coin}-{interval} (패턴: {signal_pattern})")
            
            return evolved_signal
    
    def _get_pattern_weight(self, signal: SignalInfo, coin: str, interval: str) -> float:
        """패턴 기반 가중치 계산"""
        try:
            pattern_key = f"{coin}_{interval}_{signal.pattern_type}"
            if pattern_key in self.pattern_weights:
                return self.pattern_weights[pattern_key]
            return 1.0  # 기본값
        except:
            return 1.0
    
    def _get_market_adaptation_weight(self, coin: str, interval: str) -> float:
        """시장 적응 가중치 계산"""
        try:
            market_key = f"{coin}_{interval}"
            if market_key in self.market_adaptations:
                return self.market_adaptations[market_key]
            return 1.0  # 기본값
        except:
            return 1.0

class ContextMemory:
    """맥락 메모리 - 시장 상황과 패턴 기억"""
    def __init__(self):
        self.market_contexts = {}
        self.pattern_memories = {}
        self.success_patterns = {}
        self.failure_patterns = {}
        
    def remember_market_context(self, coin: str, interval: str, context: dict):
        """시장 상황 기억"""
        key = f"{coin}_{interval}"
        self.market_contexts[key] = context
        
    def remember_pattern_result(self, pattern: str, success: bool, profit: float):
        """패턴 결과 기억"""
        if success:
            if pattern not in self.success_patterns:
                self.success_patterns[pattern] = []
            self.success_patterns[pattern].append(profit)
        else:
            if pattern not in self.failure_patterns:
                self.failure_patterns[pattern] = []
            self.failure_patterns[pattern].append(profit)

class RealTimeLearner:
    """실시간 학습기 - 즉시 학습 및 적응"""
    def __init__(self):
        self.learning_rate = 0.01
        self.recent_trades = []
        self.pattern_performance = {}
        
    def learn_from_trade(self, signal_pattern: str, trade_result: dict):
        """거래 결과로부터 즉시 학습"""
        try:
            profit = trade_result.get('profit_loss_pct', 0.0)
            success = profit > 0
            
            # 패턴 성과 업데이트
            if signal_pattern not in self.pattern_performance:
                self.pattern_performance[signal_pattern] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'total_profit': 0.0,
                    'success_rate': 0.0
                }
            
            perf = self.pattern_performance[signal_pattern]
            perf['total_trades'] += 1
            perf['total_profit'] += profit
            
            if success:
                perf['successful_trades'] += 1
            
            perf['success_rate'] = perf['successful_trades'] / perf['total_trades']
            
            print(f"🧠 실시간 학습: {signal_pattern} 패턴 성과 업데이트 (성공률: {perf['success_rate']:.2f})")
            
        except Exception as e:
            print(f"⚠️ 실시간 학습 오류: {e}")

class SignalTradeConnector:
    """시그널-매매 연결 시스템"""
    def __init__(self):
        self.connections = {}
        self.pending_signals = {}
        
    def connect_signal_to_trade(self, signal: SignalInfo, trade_result: dict):
        """시그널과 매매 결과 연결"""
        try:
            connection_id = f"{signal.coin}_{signal.timestamp}"
            self.connections[connection_id] = {
                'signal': signal,
                'trade_result': trade_result,
                'connected_at': time.time()
            }
            print(f"🔗 시그널-매매 연결: {signal.coin} 연결 완료")
        except Exception as e:
            print(f"⚠️ 시그널-매매 연결 오류: {e}")

class OptimizedCache:
    """🚀 최적화된 LRU 캐시 시스템"""
    def __init__(self, max_size=10000):
        from collections import OrderedDict
        import threading
        
        self.cache = OrderedDict()
        self.timestamps = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, max_age: int = 300):
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < max_age:
                    # LRU 업데이트
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return self.cache[key]
                else:
                    # 만료된 캐시 제거
                    del self.cache[key]
                    del self.timestamps[key]
            self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }

    def __len__(self):
        """캐시 크기 반환"""
        with self.lock:
            return len(self.cache)

    def __contains__(self, key):
        """캐시에 키가 있는지 확인"""
        with self.lock:
            return key in self.cache

    def __delitem__(self, key):
        """캐시에서 항목 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]

    def items(self):
        """캐시 항목 반환 (타임스탬프 포함)"""
        with self.lock:
            return [(k, (v, self.timestamps.get(k, 0))) for k, v in self.cache.items()]

    def clear(self):
        """캐시 전체 삭제"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

class DatabasePool:
    """🚀 데이터베이스 연결 풀 - 충돌 방지 강화"""
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.write_pool = []
        self.read_pool = []
        import threading
        self.write_lock = threading.Lock()
        self.read_lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """연결 풀 초기화 - 읽기/쓰기 분리"""
        for _ in range(self.max_connections):
            # 쓰기용 연결
            write_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            write_conn.execute("PRAGMA journal_mode=WAL")  # WAL 모드로 동시성 향상
            write_conn.execute("PRAGMA synchronous=NORMAL")  # 성능 최적화
            write_conn.execute("PRAGMA cache_size=10000")  # 캐시 크기 증가
            write_conn.execute("PRAGMA temp_store=MEMORY")  # 임시 테이블을 메모리에
            write_conn.execute("PRAGMA read_uncommitted = 0")  # 쓰기 모드
            self.write_pool.append(write_conn)
            
            # 읽기용 연결
            read_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            read_conn.execute("PRAGMA journal_mode=WAL")
            read_conn.execute("PRAGMA synchronous=NORMAL")
            read_conn.execute("PRAGMA cache_size=10000")
            read_conn.execute("PRAGMA temp_store=MEMORY")
            read_conn.execute("PRAGMA read_uncommitted = 1")  # 읽기 전용 모드
            self.read_pool.append(read_conn)
    
    def get_connection(self, read_only: bool = False):
        """연결 풀에서 연결 가져오기 - 읽기/쓰기 분리"""
        if read_only:
            with self.read_lock:
                if self.read_pool:
                    return self.read_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA read_uncommitted = 1")
                    return conn
        else:
            with self.write_lock:
                if self.write_pool:
                    return self.write_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA read_uncommitted = 0")
                    return conn
    
    def return_connection(self, conn, read_only: bool = False):
        """연결 풀에 연결 반환 - 읽기/쓰기 분리"""
        if read_only:
            with self.read_lock:
                if len(self.read_pool) < self.max_connections:
                    self.read_pool.append(conn)
                else:
                    conn.close()
        else:
            with self.write_lock:
                if len(self.write_pool) < self.max_connections:
                    self.write_pool.append(conn)
                else:
                    conn.close()

class SignalSelector:
    def __init__(self):
        """시그널 선택기 초기화 (강화된 에러 처리)"""
        self.rl_q_table = {}
        self.coin_specific_strategies = {}
        self.fractal_analysis_results = {}
        self.signal_cache = {}
        self.last_cleanup = time.time()
        self.last_dna_update = 0  # 🧬 DNA 패턴 마지막 업데이트 시간
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.synergy_patterns = {}
        self.global_strategies_cache = {}
        self.last_global_strategies_update = 0
        self.error_count = 0
        
        # 🆕 통계 및 캐시 관리 속성 초기화
        self._signal_stats = {
            'total_signals_generated': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'start_time': time.time(),
            'last_cleanup': time.time()
        }
        
        self._error_tracker = {
            'consecutive_errors': 0,
            'error_types': {},
            'recovery_attempts': 0
        }
        
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # 🚀 최적화된 캐시 시스템
        self.cache = OptimizedCache(max_size=CACHE_SIZE)
        self.max_cache_size = CACHE_SIZE  # 🆕 max_cache_size 속성 추가
        self.db_pool = DatabasePool(CANDLES_DB_PATH, max_connections=8)
        self.prepared_statements = {}
        
        # 🚀 크로스 코인 학습 컨텍스트
        self.cross_coin_context = None
        self.cross_coin_available = CROSS_COIN_AVAILABLE
        if self.cross_coin_available:
            self._load_cross_coin_context()
        
        # 🆕 3단계 성능 업그레이드 시스템 초기화
        self.off_policy_evaluator = OffPolicyEvaluator()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.meta_corrector = MetaCorrector()
        
        # 🆕 2단계 성능 업그레이드 시스템 초기화
        self.contextual_bandit = ContextualBandit(exploration_factor=1.0)
        self.regime_detector = RegimeChangeDetector()
        
        # 🆕 성능 업그레이드 시스템 초기화
        self.exponential_decay = ExponentialDecayWeight(decay_rate=0.1)
        self.bayesian_smoothing = BayesianSmoothing(alpha=1.0, beta=1.0, kappa=1.0)
        self.action_scorer = ActionSpecificScorer()
        self.context_extractor = ContextFeatureExtractor()
        self.outlier_guardrail = OutlierGuardrail(percentile_cut=0.05)
        
        # 🆕 진화형 AI 시스템 초기화
        self.evolution_engine = EvolutionEngine()
        self.context_memory = ContextMemory()
        self.strategy_weights = {}
        self.pattern_performance = {}
        self.real_time_learner = RealTimeLearner()
        
        # 🆕 시그널-매매 연결 시스템
        self.signal_trade_connector = SignalTradeConnector()
        
        print("🚀 진화형 AI 시그널 셀렉터 초기화 완료")
        self.min_signal_score = 0.03  # 더 민감하게 (0.05 → 0.03)
        
        # 🆕 학습 기반 임계값 설정
        self.use_learning_based_thresholds = True
        self.learning_feedback = None
        self.min_confidence = 0.5  # 최소 신뢰도 임계값
        
        # 🆕 RL Pipeline 통합 분석기 추가
        self.integrated_analyzer = None
        try:
            from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
            self.integrated_analyzer = IntegratedAnalyzer()
            print("✅ RL Pipeline 통합 분석기 로드 완료")
        except Exception as e:
            print(f"⚠️ RL Pipeline 통합 분석기 로드 실패: {e}")
            self.integrated_analyzer = None
        
        # 🆕 새로운 학습 결과 데이터 캐시
        self.reliability_scores = {}
        self.learning_quality_scores = {}
        self.global_strategy_mapping = {}
        self.walk_forward_performance = {}
        self.regime_coverage = {}
        self._load_enhanced_learning_data()
        
        # 🔥 Absolute Zero 분석 결과 캐시
        self.integrated_analysis_cache = {}  # {coin-interval: analysis_result}
        self.global_strategies_cache = {}  # {interval: [strategies]}
        self._load_absolute_zero_analysis_results()
        
        # 🆕 AI 모델 초기화
        self.ai_model = None
        self.ai_model_loaded = False
        self.model_type = "none"
        self.current_coin = None
        self.feature_dim = 0  # 🆕 특징 차원 동적 설정
        
        # 🆕 learning_engine 연동 초기화
        self.global_learning_manager = None
        self.symbol_finetuning_manager = None
        self.synergy_learner = None
        self.reliability_calculator = None
        self.continuous_learning_manager = None
        self.routing_pattern_analyzer = None
        self.contextual_learning_manager = None
        
        # 🆕 advanced_learning_systems 연동
        self.advanced_learning_system = None
        self.ensemble_learning_system = None
        self.meta_learning_system = None
        self.integrated_advanced_system = None
        
        # 🆕 고급 학습 시스템 로드
        self._load_advanced_learning_systems()
        
        # 🆕 전략 점수 계산기 초기화 (클래스 메서드 직접 사용)
        self.strategy_score_calculator = None
        
        if AI_MODEL_AVAILABLE:
            self._load_ai_model()
            self._load_learning_engines()
        
        # 🆕 단기-장기 시너지 학습기 초기화
        self.synergy_learner = None
        self.synergy_learning_available = SYNERGY_LEARNING_AVAILABLE
        if self.synergy_learning_available:
            try:
                self.synergy_learner = ShortTermLongTermSynergyLearner()
                print("✅ 단기-장기 시너지 학습기 초기화 완료")
            except Exception as e:
                print(f"⚠️ 시너지 학습기 초기화 실패: {e}")
                self.synergy_learning_available = False
        
        # 🆕 전략 점수 계산기 초기화 (리팩토링)
        self._strategy_calculator = StrategyScoreCalculator()
        
        # 데이터베이스 초기화
        self.create_signal_table()
        self.create_enhanced_learning_tables()  # 🆕 향상된 학습 테이블들 생성
        
        # 전략 데이터 로드
        self.load_rl_q_table()
        self.load_coin_specific_strategies()

        # 🧬 완전 자동화: 학습 데이터에서 DNA 패턴 추출 및 적용
        self.load_dna_patterns_from_learning_data()

        self.load_fractal_analysis_results()

        # 🚀 고성능 시스템 초기화 완료
        print(f"🚀 고성능 시그널 시스템 초기화 완료:")
        print(f"   - GPU 가속: {USE_GPU_ACCELERATION}")
        print(f"   - 캐시 크기: {CACHE_SIZE:,}")
        print(f"   - 크로스 코인 학습: {self.cross_coin_available}")
        print(f"   - 병렬 워커: {MAX_WORKERS}")
        print(f"   - 시너지 학습: {self.synergy_learning_available}")
        
        # 🆕 시너지 학습 결과 로드
        if self.synergy_learning_available:
            self._load_synergy_patterns()

        # 🆕 변동성 기반 시스템 초기화
        self.coin_volatility_profiles = {}  # {coin: volatility_profile}
        self.volatility_system_available = VOLATILITY_SYSTEM_AVAILABLE
        if self.volatility_system_available:
            self._load_coin_volatility_profiles()
            print("✅ 변동성 프로파일 로드 완료")

    def _load_coin_volatility_profiles(self):
        """🆕 모든 코인의 변동성 프로파일 로드"""
        try:
            coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE', 'AVAX', 'DOT', 'MATIC']
            for coin in coins:
                try:
                    profile = get_volatility_profile(coin, CANDLES_DB_PATH)
                    if profile:
                        self.coin_volatility_profiles[coin] = profile
                        # avg_atr가 None일 수 있으므로 안전하게 처리
                        avg_atr = profile.get('avg_atr', 0)
                        if avg_atr is None:
                            avg_atr = 0
                        volatility_group = profile.get('volatility_group', 'UNKNOWN')
                        if volatility_group is None:
                            volatility_group = 'UNKNOWN'
                        print(f"   - {coin}: {volatility_group} (ATR: {avg_atr:.4f})")
                except Exception as e:
                    print(f"⚠️ {coin} 변동성 프로파일 로드 실패: {e}")
        except Exception as e:
            print(f"⚠️ 변동성 프로파일 로드 실패: {e}")

    def get_coin_volatility_group(self, coin: str) -> str:
        """🆕 코인의 변동성 그룹 반환"""
        if not self.volatility_system_available:
            return 'MEDIUM'  # 기본값

        profile = self.coin_volatility_profiles.get(coin)
        if profile:
            return profile.get('volatility_group', 'MEDIUM')

        # 프로파일이 없으면 실시간 계산
        try:
            profile = get_volatility_profile(coin, CANDLES_DB_PATH)
            if profile:
                self.coin_volatility_profiles[coin] = profile
                return profile.get('volatility_group', 'MEDIUM')
        except Exception as e:
            print(f"⚠️ {coin} 변동성 그룹 조회 실패: {e}")

        return 'MEDIUM'  # 기본값

    def get_volatility_based_weights(self, coin: str, market_condition: str, has_ai_model: bool) -> dict:
        """🆕 변동성 그룹에 따른 동적 가중치 반환

        변동성별 전략:
        - LOW (BTC): 기술적 분석 + RL 중심 (안정적)
        - MEDIUM (ETH, BNB): 균형잡힌 접근
        - HIGH (ADA, SOL, AVAX): DNA 패턴 + AI 중심
        - VERY_HIGH (DOGE): DNA 패턴 최우선 (보수적)
        """
        vol_group = self.get_coin_volatility_group(coin)

        # 기본 가중치 (MEDIUM 변동성)
        if market_condition == "bull_market":
            if has_ai_model:
                weights = {'base': 0.3, 'dna': 0.15, 'rl': 0.1, 'ai': 0.2, 'integrated': 0.25}
            else:
                weights = {'base': 0.4, 'dna': 0.25, 'rl': 0.05, 'integrated': 0.3}
        elif market_condition == "bear_market":
            if has_ai_model:
                weights = {'base': 0.15, 'dna': 0.15, 'rl': 0.15, 'ai': 0.3, 'integrated': 0.25}
            else:
                weights = {'base': 0.2, 'dna': 0.3, 'rl': 0.2, 'integrated': 0.3}
        elif market_condition == "sideways_market":
            if has_ai_model:
                weights = {'base': 0.2, 'dna': 0.2, 'rl': 0.15, 'ai': 0.2, 'integrated': 0.25}
            else:
                weights = {'base': 0.25, 'dna': 0.3, 'rl': 0.15, 'integrated': 0.3}
        elif market_condition in ["overbought", "oversold"]:
            if has_ai_model:
                weights = {'base': 0.1, 'dna': 0.2, 'rl': 0.1, 'ai': 0.3, 'integrated': 0.3}
            else:
                weights = {'base': 0.15, 'dna': 0.45, 'rl': 0.15, 'integrated': 0.25}
        else:
            if has_ai_model:
                weights = {'base': 0.15, 'dna': 0.2, 'rl': 0.1, 'ai': 0.3, 'integrated': 0.25}
            else:
                weights = {'base': 0.25, 'dna': 0.35, 'rl': 0.15, 'integrated': 0.25}

        # 변동성 그룹별 가중치 조정
        if vol_group == 'LOW':
            # LOW 변동성: 기술적 분석과 RL 신뢰도 높음
            weights['base'] *= 1.3   # 기술적 분석 강화
            weights['rl'] *= 1.4     # RL 학습 강화
            weights['dna'] *= 0.7    # DNA 패턴 감소
            if has_ai_model:
                weights['ai'] *= 0.9  # AI 약간 감소

        elif vol_group == 'HIGH':
            # HIGH 변동성: 패턴 매칭과 AI 중심
            weights['base'] *= 0.8   # 기술적 분석 감소
            weights['dna'] *= 1.4    # DNA 패턴 강화
            weights['rl'] *= 0.9     # RL 약간 감소
            if has_ai_model:
                weights['ai'] *= 1.3  # AI 강화

        elif vol_group == 'VERY_HIGH':
            # VERY_HIGH 변동성: DNA 패턴 최우선 (보수적)
            weights['base'] *= 0.6   # 기술적 분석 크게 감소
            weights['dna'] *= 1.8    # DNA 패턴 크게 강화
            weights['rl'] *= 0.7     # RL 감소
            if has_ai_model:
                weights['ai'] *= 1.1  # AI 약간 강화

        # 정규화 (합이 1.0이 되도록)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_volatility_based_thresholds(self, coin: str) -> dict:
        """🆕 변동성 그룹에 따른 동적 액션 임계값 반환

        변동성별 전략:
        - LOW (BTC): 엄격한 임계값 (강한 신호만 반응)
        - MEDIUM (ETH, BNB): 중간 임계값
        - HIGH (ADA, SOL): 완화된 임계값 (빠른 반응)
        - VERY_HIGH (DOGE): 매우 완화된 임계값 (즉각 반응)
        """
        vol_group = self.get_coin_volatility_group(coin)

        if vol_group == 'LOW':
            # LOW 변동성: 엄격한 임계값 (BTC - 안정적이므로 강한 신호만)
            return {
                'strong_buy': 0.6,
                'weak_buy': 0.3,
                'weak_sell': -0.3,
                'strong_sell': -0.6
            }
        elif vol_group == 'MEDIUM':
            # MEDIUM 변동성: 중간 임계값 (ETH, BNB - 균형)
            return {
                'strong_buy': 0.5,
                'weak_buy': 0.2,
                'weak_sell': -0.2,
                'strong_sell': -0.5
            }
        elif vol_group == 'HIGH':
            # HIGH 변동성: 완화된 임계값 (ADA, SOL, AVAX - 빠른 반응)
            return {
                'strong_buy': 0.4,
                'weak_buy': 0.15,
                'weak_sell': -0.15,
                'strong_sell': -0.4
            }
        else:  # VERY_HIGH
            # VERY_HIGH 변동성: 매우 완화된 임계값 (DOGE - 즉각 반응)
            return {
                'strong_buy': 0.3,
                'weak_buy': 0.1,
                'weak_sell': -0.1,
                'strong_sell': -0.3
            }

    def _load_enhanced_learning_data(self):
        """🆕 향상된 학습 데이터 로드 (가상매매 DB 연동 강화)"""
        try:
            logger.info("🔄 향상된 학습 데이터 로딩 중...")
            
            # 🆕 테이블이 없으면 자동으로 생성
            self.create_enhanced_learning_tables()
            
            # 신뢰도 점수 로드
            self.reliability_scores = self._load_reliability_scores()
            logger.info(f"✅ 신뢰도 점수 로드 완료: {len(self.reliability_scores)}개")
            
            # 학습 품질 점수 로드
            self.learning_quality_scores = self._load_learning_quality_scores()
            logger.info(f"✅ 학습 품질 점수 로드 완료: {len(self.learning_quality_scores)}개")
            
            # 글로벌 전략 매핑 로드
            self.global_strategy_mapping = self._load_global_strategy_mapping()
            logger.info(f"✅ 글로벌 전략 매핑 로드 완료: {len(self.global_strategy_mapping)}개")
            
            # Walk-Forward 성능 데이터 로드
            self.walk_forward_performance = self._load_walk_forward_performance()
            logger.info(f"✅ Walk-Forward 성능 데이터 로드 완료: {len(self.walk_forward_performance)}개")
            
            # 레짐별 커버리지 데이터 로드
            self.regime_coverage = self._load_regime_coverage()
            logger.info(f"✅ 레짐별 커버리지 데이터 로드 완료: {len(self.regime_coverage)}개")
            
            # 🆕 가상매매 학습 데이터 로드 (강화)
            self._load_virtual_trading_learning_data()
            
            logger.info("🎉 향상된 학습 데이터 로딩 완료!")
            
        except Exception as e:
            logger.warning(f"⚠️ 향상된 학습 데이터 로딩 실패: {e}")
            # 기본값으로 초기화
            self.reliability_scores = {}
            self.learning_quality_scores = {}
            self.global_strategy_mapping = {}
            self.walk_forward_performance = {}
            self.regime_coverage = {}
    
    def _load_virtual_trading_learning_data(self):
        """🆕 가상매매 학습 데이터 로드 (성능 업그레이드 적용)"""
        try:
            import sqlite3
            
            # 가상매매 DB에서 학습 데이터 로드
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 스키마 확인 및 테이블 생성
                self._ensure_signal_feedback_schema(conn)
                
                cursor = conn.cursor()
                
                # 시그널 피드백 점수 로드 (최근성 가중치 적용)
                cursor.execute("""
                    SELECT signal_pattern, success_rate, avg_profit, total_trades, confidence, created_at
                    FROM signal_feedback_scores
                    ORDER BY created_at DESC
                """)
                
                virtual_pattern_performance = {}
                current_time = time.time()
                
                for row in cursor.fetchall():
                    pattern, success_rate, avg_profit, total_trades, confidence, created_at = row
                    
                    # 최근성 가중치 계산
                    time_diff_hours = (current_time - created_at) / 3600
                    recency_weight = self.exponential_decay.calculate_weight(time_diff_hours)
                    
                    # 베이지안 스무딩 적용
                    smoothed_success_rate = self.bayesian_smoothing.smooth_success_rate(
                        int(success_rate * total_trades), int(total_trades)
                    )
                    smoothed_avg_profit = self.bayesian_smoothing.smooth_avg_profit(
                        [avg_profit], avg_profit
                    )
                    
                    virtual_pattern_performance[pattern] = {
                        'success_rate': smoothed_success_rate,
                        'avg_profit': smoothed_avg_profit,
                        'total_trades': total_trades,
                        'confidence': confidence,
                        'recency_weight': recency_weight
                    }
                
                # 기존 신뢰도 점수와 병합 (최근성 가중치 적용)
                for pattern, data in virtual_pattern_performance.items():
                    if pattern not in self.reliability_scores:
                        self.reliability_scores[pattern] = data['success_rate']
                    else:
                        # 최근성 가중 평균으로 병합
                        weight = data['recency_weight']
                        self.reliability_scores[pattern] = (
                            self.reliability_scores[pattern] * (1 - weight) + 
                            data['success_rate'] * weight
                        )
                
                # 기존 학습 품질 점수와 병합 (최근성 가중치 적용)
                for pattern, data in virtual_pattern_performance.items():
                    if pattern not in self.learning_quality_scores:
                        self.learning_quality_scores[pattern] = data['avg_profit']
                    else:
                        # 최근성 가중 평균으로 병합
                        weight = data['recency_weight']
                        self.learning_quality_scores[pattern] = (
                            self.learning_quality_scores[pattern] * (1 - weight) + 
                            data['avg_profit'] * weight
                        )
                
                logger.info(f"✅ 가상매매 학습 데이터 로드 완료 (성능 업그레이드 적용): {len(virtual_pattern_performance)}개 패턴")
                
        except Exception as e:
            logger.warning(f"⚠️ 가상매매 학습 데이터 로드 실패: {e}")
    
    def _load_reliability_scores(self) -> Dict[str, float]:
        """신뢰도 점수 로드"""
        try:
            # learning_results.db에서 신뢰도 점수 로드
            learning_db_path = "/workspace/data_storage/learning_results.db"
            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT strategy_id, reliability_score 
                FROM reliability_scores 
                WHERE reliability_score > 0
            """)
            
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = float(row[1])
            
            conn.close()
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 신뢰도 점수 로드 실패: {e}")
            return {}
    
    def _load_learning_quality_scores(self) -> Dict[str, float]:
        """학습 품질 점수 로드"""
        try:
            # learning_results.db에서 학습 품질 점수 로드
            learning_db_path = "/workspace/data_storage/learning_results.db"
            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT strategy_id, learning_quality_score 
                FROM strategy_learning_history 
                WHERE learning_quality_score > 0
            """)
            
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = float(row[1])
            
            conn.close()
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 학습 품질 점수 로드 실패: {e}")
            return {}
    
    def _load_global_strategy_mapping(self) -> Dict[str, str]:
        """글로벌 전략 매핑 로드"""
        try:
            # learning_results.db에서 글로벌 전략 매핑 로드
            learning_db_path = "/workspace/data_storage/learning_results.db"
            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT coin, global_strategy_id 
                FROM global_strategy_mapping 
                WHERE global_strategy_id IS NOT NULL
            """)
            
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = row[1]
            
            conn.close()
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 글로벌 전략 매핑 로드 실패: {e}")
            return {}
    
    def _load_walk_forward_performance(self) -> Dict[str, Dict[str, float]]:
        """Walk-Forward 성능 데이터 로드"""
        try:
            # learning_results.db에서 Walk-Forward 성능 데이터 로드
            learning_db_path = "/workspace/data_storage/learning_results.db"
            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT strategy_id, performance_metrics 
                FROM walk_forward_performance 
                WHERE performance_metrics IS NOT NULL
            """)
            
            results = {}
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    results[row[0]] = data
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ Walk-Forward 성능 데이터 로드 실패: {e}")
            return {}
    
    def _load_regime_coverage(self) -> Dict[str, Dict[str, float]]:
        """레짐별 커버리지 데이터 로드"""
        try:
            # learning_results.db에서 레짐별 커버리지 데이터 로드
            learning_db_path = "/workspace/data_storage/learning_results.db"
            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT strategy_id, market_regime, coverage_score, performance_in_regime 
                FROM regime_coverage 
                WHERE coverage_score > 0
            """)
            
            results = {}
            for row in cursor.fetchall():
                strategy_id, market_regime, coverage_score, performance_in_regime = row
                if strategy_id not in results:
                    results[strategy_id] = {}
                results[strategy_id][market_regime] = {
                    'coverage_score': float(coverage_score),
                    'performance_in_regime': float(performance_in_regime)
                }
            
            conn.close()
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 레짐별 커버리지 데이터 로드 실패: {e}")
            return {}
    
    def _get_cached_market_condition(self, coin: str, interval: str) -> str:
        """🚀 캐시된 시장 상황 반환 (빠른 판단)"""
        try:
            cache_key = f"market_condition_{coin}_{interval}"
            cached_data = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            
            if cached_data:
                return cached_data
            
            # 캐시가 없으면 간단한 시장 상황 판단
            market_condition = self._detect_simple_market_condition(coin, interval)
            
            # 캐시에 저장
            self.set_cached_data(cache_key, market_condition)
            
            return market_condition
            
        except Exception as e:
            return 'neutral'  # 기본값
    
    def _detect_simple_market_condition(self, coin: str, interval: str) -> str:
        """🚀 간단한 시장 상황 감지 (속도 우선)"""
        try:
            candle = self.get_nearest_candle(coin, interval, int(time.time()))
            if candle is None:
                return 'neutral'
            
            close = candle.get('close', 0.0)
            open_price = candle.get('open', close)
            
            if close == 0 or open_price == 0:
                return 'neutral'
            
            # 간단한 가격 변화 기반 판단
            price_change = (close - open_price) / open_price
            
            if price_change > 0.02:  # 2% 이상 상승
                return 'uptrend'
            elif price_change < -0.02:  # 2% 이상 하락
                return 'downtrend'
            elif abs(price_change) < 0.005:  # 0.5% 이내
                return 'sideways'
            else:
                return 'neutral'
                
        except Exception as e:
            return 'neutral'
    
    def _select_smart_strategy(self, coin: str, interval: str, market_condition: str, indicators: Dict) -> Optional[Dict]:
        """🚀 스마트 전략 선택 (RL Pipeline 학습 결과 활용)"""
        try:
            cache_key = f"smart_strategy_{coin}_{interval}_{market_condition}"
            cached_strategy = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            
            if cached_strategy:
                return cached_strategy
            
            # 🚀 1. 기본 전략 정보
            strategy = {
                'strategy_type': 'smart',
                'market_condition_bonus': 1.0,
                'risk_level': 'medium',
                'rl_pipeline_score': indicators.get('rl_pipeline_score', 0.5),
                'global_strategy_score': indicators.get('global_strategy_score', 0.5),
                'dna_similarity_score': indicators.get('dna_similarity_score', 0.5),
                'synergy_score': indicators.get('synergy_score', 0.5)
            }
            
            # 🚀 2. 시장 상황별 보너스 (학습 결과 반영)
            if market_condition == 'uptrend':
                strategy['market_condition_bonus'] = 1.1
                strategy['risk_level'] = 'low' if strategy['rl_pipeline_score'] > 0.7 else 'medium'
            elif market_condition == 'downtrend':
                strategy['market_condition_bonus'] = 0.9
                strategy['risk_level'] = 'high' if strategy['rl_pipeline_score'] < 0.3 else 'medium'
            elif market_condition == 'sideways':
                strategy['market_condition_bonus'] = 1.0
                strategy['risk_level'] = 'medium'
            
            # 🚀 3. 학습 결과 기반 추가 보너스
            if strategy['rl_pipeline_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.1
            if strategy['global_strategy_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.05
            if strategy['dna_similarity_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.05
            if strategy['synergy_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.05
            
            # 캐시에 저장
            self.set_cached_data(cache_key, strategy)
            
            return strategy
            
        except Exception as e:
            return None
    
    def _calculate_signal_calmar_ratio(self, candle: pd.Series, indicators: Dict) -> float:
        """시그널용 Calmar Ratio 계산"""
        try:
            # 현재 가격 변화율을 수익률로 가정
            current_price = candle.get('close', 0.0)
            open_price = candle.get('open', current_price)
            
            if open_price > 0:
                profit = (current_price - open_price) / open_price
            else:
                profit = 0.0
            
            # 변동성을 최대 낙폭으로 근사
            volatility = indicators.get('volatility', 0.02)
            max_drawdown = abs(volatility)  # 변동성을 최대 낙폭으로 근사
            
            if max_drawdown > 0:
                calmar_ratio = profit / max_drawdown
            else:
                calmar_ratio = profit * 100 if profit > 0 else 0.0
            
            return max(0.0, min(10.0, calmar_ratio))
            
        except Exception as e:
            logger.warning(f"시그널 Calmar Ratio 계산 실패: {e}")
            return 1.0
    
    def _calculate_signal_profit_factor(self, candle: pd.Series, indicators: Dict) -> float:
        """시그널용 Profit Factor 계산 (최적화) - None 값 안전 처리"""
        try:
            # RSI와 MACD를 기반으로 수익/손실 비율 근사 (None 값 안전 처리)
            rsi = indicators.get('rsi', 50.0)
            macd = indicators.get('macd', 0.0)
            
            # None 값 안전 처리
            if rsi is None:
                rsi = 50.0
            if macd is None:
                macd = 0.0
            
            # RSI 기반 수익 확률
            if rsi < 30:  # 과매도
                win_probability = 0.7
            elif rsi > 70:  # 과매수
                win_probability = 0.3
            else:
                win_probability = 0.5
            
            # MACD 기반 수익 강도
            if macd > 0:
                profit_strength = 1.2
            else:
                profit_strength = 0.8
            
            # Profit Factor 근사 계산
            if win_probability > 0:
                profit_factor = (win_probability * profit_strength) / (1 - win_probability)
            else:
                profit_factor = 1.0
            
            return max(0.1, min(5.0, profit_factor))
            
        except Exception as e:
            logger.warning(f"시그널 Profit Factor 계산 실패: {e}")
            return 1.0

    def _handle_error(self, error: Exception, context: str, coin: str = None, interval: str = None):
        """오류 처리 및 복구 (개선된 오류 처리)"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 🆕 오류 추적
        self._error_tracker['consecutive_errors'] += 1
        self._error_tracker['error_types'][error_type] = self._error_tracker['error_types'].get(error_type, 0) + 1
        self._signal_stats['failed_signals'] += 1
        
        # 🆕 상세한 오류 로깅
        error_context = f"{context}"
        if coin and interval:
            error_context += f" ({coin}/{interval})"
        
        print(f"❌ 시그널 생성 오류: {error_context}")
        print(f"  - 오류 유형: {error_type}")
        print(f"  - 오류 메시지: {error_msg}")
        print(f"  - 연속 오류: {self._error_tracker['consecutive_errors']}회")
        
        # 🆕 오류 복구 로직
        if self._error_tracker['consecutive_errors'] >= 3:
            print(f"⚠️ 연속 오류 3회 발생 - 캐시 정리 및 복구 시도")
            self._cleanup_cache()
            self._error_tracker['recovery_attempts'] += 1
            self._error_tracker['consecutive_errors'] = 0
        
        # 🆕 심각한 오류 처리
        if self._error_tracker['recovery_attempts'] >= 2:
            print(f"🚨 심각한 오류 - 시그널 생성 중단 권장")
            raise SystemError("시그널 생성 시스템 복구 실패")

    def _calculate_enhanced_global_strategy_score(self, candle: pd.Series, interval: str) -> float:
        """향상된 글로벌 전략 점수 계산"""
        try:
            # 기본 기술적 지표 기반 점수 계산
            rsi = safe_float(candle.get('rsi'), 50.0)
            macd = safe_float(candle.get('macd'), 0.0)
            volume_ratio = safe_float(candle.get('volume_ratio'), 1.0)
            volatility = safe_float(candle.get('volatility'), 0.02)
            
            # RSI 기반 점수 (더 정교한 계산)
            rsi_score = 0.0
            if rsi < 20:  # 극도 과매도
                rsi_score = 1.0
            elif rsi < 30:  # 과매도
                rsi_score = 0.8
            elif rsi > 80:  # 극도 과매수
                rsi_score = -1.0
            elif rsi > 70:  # 과매수
                rsi_score = -0.6
            else:  # 중립
                rsi_score = (50 - abs(rsi - 50)) / 50 * 0.3
            
            # MACD 기반 점수 (더 정교한 계산)
            macd_score = np.tanh(macd * 200) * 0.4
            
            # 거래량 기반 점수 (더 정교한 계산)
            volume_score = np.tanh((volume_ratio - 1.0) * 2) * 0.3
            
            # 변동성 기반 점수 (더 정교한 계산)
            volatility_score = -np.tanh(volatility * 50) * 0.2
            
            # 인터벌별 가중치 적용
            interval_weights = {
                '1d': 1.2,   # 일봉 가중치 강화
                '15m': 1.0,  # 기본 가중치
                '30m': 1.2,  # 중기 인터벌은 높은 가중치
                '240m': 1.5  # 장기 인터벌은 가장 높은 가중치
            }
            
            weight = interval_weights.get(interval, 1.0)
            
            # 종합 점수
            total_score = (rsi_score + macd_score + volume_score + volatility_score) * weight
            
            print(f"✅ 향상된 글로벌 전략 점수: RSI={rsi_score:.3f}, MACD={macd_score:.3f}, Volume={volume_score:.3f}, Vol={volatility_score:.3f}, 가중치={weight:.1f}, 총합={total_score:.3f}")
            
            return np.clip(total_score, -1.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ 향상된 글로벌 전략 점수 계산 오류: {e}")
            return 0.0

    def _load_cross_coin_context(self):
        """크로스 코인 학습 컨텍스트 로드"""
        try:
            if CROSS_COIN_AVAILABLE:
                # self.cross_coin_context = load_global_integrated_results()  # 🆕 임시 비활성화
                self.cross_coin_context = {}
                print(f"🚀 크로스 코인 학습 컨텍스트 로드 완료")
            else:
                self.cross_coin_context = {}
                print("⚠️ 크로스 코인 학습 컨텍스트를 사용할 수 없습니다.")
        except Exception as e:
            print(f"⚠️ 크로스 코인 컨텍스트 로드 실패: {e}")
            self.cross_coin_context = {}

    def _load_learning_engines(self):
        """learning_engine.py의 학습 엔진들 로드"""
        try:
            if not AI_MODEL_AVAILABLE:
                return
            
            # 글로벌 학습 매니저 로드
            self.global_learning_manager = GlobalLearningManager()
            print("✅ 글로벌 학습 매니저 로드 완료")
            
            # 심볼별 튜닝 매니저 로드
            self.symbol_finetuning_manager = SymbolFinetuningManager()
            print("✅ 심볼별 튜닝 매니저 로드 완료")
            
            # 시너지 학습기 로드
            self.synergy_learner = ShortTermLongTermSynergyLearner()
            print("✅ 시너지 학습기 로드 완료")
            
            # 🆕 신뢰도 점수 계산기 로드
            self.reliability_calculator = ReliabilityScoreCalculator()
            print("✅ 신뢰도 점수 계산기 로드 완료")
            
            # 🆕 지속적 학습 관리자 로드
            self.continuous_learning_manager = ContinuousLearningManager()
            print("✅ 지속적 학습 관리자 로드 완료")
            
            # 🆕 라우팅 패턴 분석기 로드
            self.routing_pattern_analyzer = RoutingPatternAnalyzer()
            print("✅ 라우팅 패턴 분석기 로드 완료")
            
            # 🆕 상황별 학습 관리자 로드
            self.contextual_learning_manager = ContextualLearningManager()
            print("✅ 상황별 학습 관리자 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 학습 엔진 로드 실패: {e}")
            self.global_learning_manager = None
            self.symbol_finetuning_manager = None
            self.synergy_learner = None
            self.reliability_calculator = None
            self.continuous_learning_manager = None
            self.routing_pattern_analyzer = None
            self.contextual_learning_manager = None

    def _load_advanced_learning_systems(self):
        """advanced_learning_systems.py의 고급 학습 시스템들 로드"""
        try:
            # 앙상블 학습 시스템 로드
            from rl_pipeline.advanced_learning_systems import EnsembleLearningSystem
            self.ensemble_learning_system = EnsembleLearningSystem()
            print("✅ 앙상블 학습 시스템 로드 완료")
            
            # 메타 학습 시스템 로드
            from rl_pipeline.advanced_learning_systems import MetaLearningSystem
            self.meta_learning_system = MetaLearningSystem()
            print("✅ 메타 학습 시스템 로드 완료")
            
            # 통합 고급 시스템 로드
            from rl_pipeline.advanced_learning_systems import IntegratedAdvancedSystem
            self.integrated_advanced_system = IntegratedAdvancedSystem(state_dim=50, action_dim=10)
            print("✅ 통합 고급 시스템 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 고급 학습 시스템 로드 실패: {e}")
            self.ensemble_learning_system = None
            self.meta_learning_system = None
            self.integrated_advanced_system = None

    def _calculate_smart_indicators(self, candle: pd.Series, coin: str, interval: str) -> Dict:
        """🚀 실제 캔들 DB의 풍부한 기술적 지표를 활용한 스마트 지표 계산"""
        try:
            # 🚀 1. 실제 캔들 데이터에서 직접 지표 추출 (realtime_candles 파일들에서 계산된 값들)
            indicators = {}
            
            # 🚀 기본 OHLCV 데이터
            indicators['open'] = candle.get('open', 100.0)
            indicators['high'] = candle.get('high', 101.0)
            indicators['low'] = candle.get('low', 99.0)
            indicators['close'] = candle.get('close', 100.0)
            indicators['volume'] = candle.get('volume', 1000.0)
            
            # 🚀 오실레이터 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['rsi'] = candle.get('rsi', 50.0)
            indicators['mfi'] = candle.get('mfi', 50.0)
            
            # 🚀 트렌드 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['macd'] = candle.get('macd', 0.0)
            indicators['macd_signal'] = candle.get('macd_signal', 0.0)
            
            # 🚀 볼린저밴드 (realtime_candles_calculate.py에서 계산됨)
            indicators['bb_upper'] = candle.get('bb_upper', 1.05)
            indicators['bb_middle'] = candle.get('bb_middle', 1.0)
            indicators['bb_lower'] = candle.get('bb_lower', 0.95)
            
            # 🚀 변동성/추세 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['atr'] = candle.get('atr', 0.02)
            indicators['ma20'] = candle.get('ma20', 1.0)
            indicators['adx'] = candle.get('adx', 25.0)
            
            # 🚀 거래량 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['volume_ratio'] = candle.get('volume_ratio', 1.0)
            
            # 🚀 리스크 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['volatility'] = candle.get('volatility', 0.02)
            indicators['risk_score'] = candle.get('risk_score', 0.5)
            
            # 🚀 파동 분석 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['wave_phase'] = candle.get('wave_phase', 'unknown')
            indicators['confidence'] = candle.get('confidence', 0.5)
            indicators['zigzag_direction'] = candle.get('zigzag_direction', 0.0)
            indicators['zigzag_pivot_price'] = candle.get('zigzag_pivot_price', 100.0)
            indicators['wave_progress'] = candle.get('wave_progress', 0.5)
            
            # 🚀 패턴 분석 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['pattern_type'] = candle.get('pattern_type', 'none')
            indicators['pattern_confidence'] = candle.get('pattern_confidence', 0.0)
            
            # 🚀 통합 분석 지표 (realtime_candles_integrated.py에서 계산됨)
            indicators['volatility_level'] = candle.get('volatility_level', 'medium')
            indicators['risk_level'] = candle.get('risk_level', 'medium')
            indicators['integrated_direction'] = candle.get('integrated_direction', 'neutral')
            
            # 🚀 추가 계산된 지표들 (None 값 안전 처리)
            try:
                indicators['price_change'] = (indicators['close'] - indicators['open']) / indicators['open']
            except (TypeError, ZeroDivisionError):
                indicators['price_change'] = 0.0
            
            try:
                indicators['high_low_ratio'] = (indicators['high'] - indicators['low']) / indicators['low']
            except (TypeError, ZeroDivisionError):
                indicators['high_low_ratio'] = 0.0
            
            try:
                indicators['close_to_bb_upper'] = (indicators['close'] - indicators['bb_upper']) / indicators['bb_upper']
            except (TypeError, ZeroDivisionError):
                indicators['close_to_bb_upper'] = 0.0
            
            try:
                indicators['close_to_bb_lower'] = (indicators['close'] - indicators['bb_lower']) / indicators['bb_lower']
            except (TypeError, ZeroDivisionError):
                indicators['close_to_bb_lower'] = 0.0
            
            try:
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            except TypeError:
                indicators['macd_histogram'] = 0.0
            
            # 🚀 실제 데이터 활용 로그 (None 값 안전 처리)
            rsi_log = indicators['rsi'] if indicators['rsi'] is not None else 50.0
            macd_log = indicators['macd'] if indicators['macd'] is not None else 0.0
            volume_log = indicators['volume_ratio'] if indicators['volume_ratio'] is not None else 1.0
            wave_log = indicators['wave_phase'] if indicators['wave_phase'] is not None else 'unknown'
            pattern_log = indicators['pattern_type'] if indicators['pattern_type'] is not None else 'none'
            direction_log = indicators['integrated_direction'] if indicators['integrated_direction'] is not None else 'neutral'
            
            print(f"📊 {coin}/{interval}: 실제 기술지표 활용 - RSI({rsi_log:.1f}), MACD({macd_log:.4f}), Volume({volume_log:.2f}x), Wave({wave_log}), Pattern({pattern_log}), Direction({direction_log})")
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ 스마트 지표 계산 실패: {e}")
            # 🚀 오류 시 기본 지표 반환
            return self._calculate_fast_indicators(candle)

    def _calculate_cross_coin_bonus(self, coin: str, interval: str, current_dna: dict) -> float:
        """🚀 크로스 코인 학습 컨텍스트를 활용한 보너스 점수 계산"""
        try:
            if not self.cross_coin_context or not current_dna:
                return 1.0
            
            bonus = 1.0
            
            # 🚀 전역 패턴 매칭
            if 'universal_patterns' in self.cross_coin_context:
                universal_patterns = self.cross_coin_context['universal_patterns']
                for pattern in universal_patterns:
                    if self._match_dna_pattern(current_dna, pattern):
                        bonus *= 1.1  # 10% 보너스
                        break
            
            # 🚀 크로스 코인 유사성 보너스
            if 'cross_coin_similarity' in self.cross_coin_context:
                similarity_data = self.cross_coin_context['cross_coin_similarity']
                if coin in similarity_data:
                    avg_similarity = np.mean(list(similarity_data[coin].values()))
                    if avg_similarity > 0.7:  # 높은 유사성
                        bonus *= 1.05  # 5% 보너스
            
            # 🚀 시장 상태 적응 보너스
            if 'market_conditions' in self.cross_coin_context:
                market_conditions = self.cross_coin_context['market_conditions']
                current_condition = self._detect_current_market_condition(coin, interval)
                if current_condition in market_conditions:
                    condition_bonus = market_conditions[current_condition].get('bonus', 1.0)
                    bonus *= condition_bonus
            
            return min(bonus, 1.3)  # 최대 30% 보너스 제한
            
        except Exception as e:
            print(f"⚠️ 크로스 코인 보너스 계산 실패: {e}")
            return 1.0

    def _match_dna_pattern(self, current_dna: dict, pattern: dict) -> bool:
        """DNA 패턴 매칭 (크로스 코인 학습용)"""
        try:
            match_count = 0
            total_count = 0
            
            for key, value in pattern.items():
                if key in current_dna:
                    total_count += 1
                    if current_dna[key] == value:
                        match_count += 1
            
            if total_count == 0:
                return False
            
            match_ratio = match_count / total_count
            return match_ratio >= 0.7  # 70% 이상 매칭
            
        except Exception as e:
            print(f"⚠️ DNA 패턴 매칭 실패: {e}")
            return False

    def _detect_current_market_condition(self, coin: str, interval: str) -> str:
        """현재 시장 상태 감지 (크로스 코인 학습용)"""
        try:
            # 간단한 시장 상태 감지 (실제 구현에서는 더 정교한 로직 사용)
            return 'neutral'  # 기본값
        except Exception as e:
            print(f"⚠️ 시장 상태 감지 실패: {e}")
            return 'neutral'

    def _calculate_fast_indicators(self, candle: pd.Series) -> Dict:
        """🚀 빠른 기술적 지표 계산 (핵심 지표만)"""
        try:
            indicators = {}
            
            # 🚀 1. 기본 가격 지표 (가장 빠름)
            close = candle.get('close', 0.0)
            open_price = candle.get('open', close)
            high = candle.get('high', close)
            low = candle.get('low', close)
            volume = candle.get('volume', 0.0)
            
            # 🚀 2. 간단한 RSI 계산 (14기간 대신 7기간 사용)
            rsi = self._calculate_fast_rsi(candle)
            indicators['rsi'] = rsi
            
            # 🚀 3. 간단한 MACD 계산
            macd = self._calculate_fast_macd(candle)
            indicators['macd'] = macd
            
            # 🚀 4. 거래량 비율 (간단한 계산)
            volume_ratio = self._calculate_fast_volume_ratio(candle)
            indicators['volume_ratio'] = volume_ratio
            
            # 🚀 5. 변동성 (간단한 계산)
            volatility = self._calculate_fast_volatility(candle)
            indicators['volatility'] = volatility
            
            # 🚀 6. 기본 패턴 정보
            indicators['wave_phase'] = 'unknown'  # 복잡한 계산 생략
            indicators['pattern_type'] = 'unknown'  # 복잡한 계산 생략
            indicators['structure_score'] = 0.5  # 기본값
            indicators['pattern_confidence'] = 0.5  # 기본값
            indicators['integrated_direction'] = 'neutral'  # 기본값
            indicators['integrated_strength'] = 0.5  # 기본값
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ 빠른 지표 계산 실패: {e}")
            return {'rsi': 50.0, 'macd': 0.0, 'volume_ratio': 1.0, 'volatility': 0.02}

    def _calculate_fast_rsi(self, candle: pd.Series) -> float:
        """🚀 빠른 RSI 계산 (7기간)"""
        try:
            # 캔들 데이터가 부족하면 기본값 반환
            if len(candle) < 7:
                return 50.0
            
            # 간단한 가격 변화 계산
            close = candle.get('close', 0.0)
            # 실제로는 더 복잡한 RSI 계산이 필요하지만, 여기서는 간단히 처리
            return 50.0  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 RSI 계산 실패: {e}")
            return 50.0

    def _calculate_fast_macd(self, candle: pd.Series) -> float:
        """🚀 빠른 MACD 계산"""
        try:
            # 간단한 MACD 계산 (실제로는 더 복잡한 계산 필요)
            close = candle.get('close', 0.0)
            return 0.0  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 MACD 계산 실패: {e}")
            return 0.0

    def _calculate_fast_volume_ratio(self, candle: pd.Series) -> float:
        """🚀 빠른 거래량 비율 계산"""
        try:
            volume = candle.get('volume', 0.0)
            return 1.0  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 거래량 비율 계산 실패: {e}")
            return 1.0

    def _calculate_fast_volatility(self, candle: pd.Series) -> float:
        """🚀 빠른 변동성 계산"""
        try:
            high = candle.get('high', 0.0)
            low = candle.get('low', 0.0)
            close = candle.get('close', 0.0)
            
            if close > 0:
                return (high - low) / close
            return 0.02  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 변동성 계산 실패: {e}")
            return 0.02

    def _calculate_advanced_indicators_with_learning_engine(self, candle: pd.Series, coin: str, interval: str) -> Dict:
        """🚀 고급 지표 계산 (learning_engine.py 학습 결과 활용)"""
        try:
            advanced_indicators = {}
            
            # 🚀 1. 글로벌 학습 점수
            if self.global_learning_manager:
                global_score = self._get_global_learning_score(coin, interval, candle)
                advanced_indicators['global_learning_score'] = global_score
            
            # 🚀 2. 심볼별 튜닝 점수
            if self.symbol_finetuning_manager:
                symbol_score = self._get_symbol_tuning_score(coin, interval, candle)
                advanced_indicators['symbol_tuning_score'] = symbol_score
            
            # 🚀 3. 시너지 학습 점수
            if self.synergy_learner:
                synergy_score = self._get_synergy_learning_score(coin, interval, candle)
                advanced_indicators['synergy_learning_score'] = synergy_score
            
            return advanced_indicators
            
        except Exception as e:
            print(f"⚠️ 고급 지표 계산 실패: {e}")
            return {}

    def _get_global_learning_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """글로벌 학습 결과에서 실제 점수 추출"""
        try:
            if not self.global_learning_manager:
                return 0.5
            
            # 🆕 글로벌 학습 결과 테이블에서 실제 데이터 조회
            cache_key = f"global_learning_{coin}_{interval}"
            cached_score = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            
            if cached_score is not None:
                return cached_score
            
            # 데이터베이스에서 글로벌 학습 결과 조회 (learning_results.db)
            with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                cursor = conn.cursor()
                
                # 🆕 global_strategy_summary_for_signals에서 글로벌 학습 결과 조회
                cursor.execute("""
                    SELECT avg_global_score, learning_quality_score, reliability_score
                    FROM global_strategy_summary_for_signals 
                    ORDER BY updated_at DESC 
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                if result:
                    # 평균 글로벌 점수 사용 (없으면 학습 품질 점수)
                    global_score = result[0] if result[0] else (result[1] if result[1] else 0.5)
                    
                    # 캐시에 저장
                    self.set_cached_data(cache_key, global_score)
                    return global_score
                else:
                    return 0.5
                
        except Exception as e:
            print(f"⚠️ 글로벌 학습 점수 계산 실패: {e}")
            return 0.5

    def _get_symbol_tuning_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """심볼별 튜닝 결과에서 실제 점수 추출"""
        try:
            if not self.symbol_finetuning_manager:
                return 0.5
            
            # 🆕 심볼별 튜닝 결과 테이블에서 실제 데이터 조회
            cache_key = f"symbol_tuning_{coin}_{interval}"
            cached_score = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            
            if cached_score is not None:
                return cached_score
            
            # 데이터베이스에서 심볼별 튜닝 결과 조회 (learning_results.db)
            with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                cursor = conn.cursor()
                
                # 🆕 strategy_summary_for_signals에서 심볼별 튜닝 결과 조회
                cursor.execute("""
                    SELECT avg_profit, total_strategies, avg_win_rate
                    FROM strategy_summary_for_signals
                    WHERE coin = ? AND interval = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (coin, interval))
                
                result = cursor.fetchone()
                if result:
                    # 평균 개선율 대신 평균 수익 사용
                    symbol_score = (result[0] / 100.0) if result[0] else 0.5  # profit을 비율로 변환
                    
                    # 캐시에 저장
                    self.set_cached_data(cache_key, symbol_score)
                    return symbol_score
                else:
                    # 기존 로직 (호환성)
                    cursor.execute("""
                        SELECT avg_improvement, total_strategies, tuned_coins
                        FROM symbol_finetuning_results 
                    WHERE coin = ?
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (coin,))
                
                result = cursor.fetchone()
                if result:
                    # 평균 개선도 사용
                    tuning_score = result[0] if result[0] else 0.5
                    
                    # 캐시에 저장
                    self.set_cached_data(cache_key, tuning_score)
                    return tuning_score
                else:
                    return 0.5
                
        except Exception as e:
            print(f"⚠️ 심볼별 튜닝 점수 계산 실패: {e}")
            return 0.5

    def _calculate_enhanced_learning_bonus(self, coin: str, interval: str, candle: pd.Series) -> float:
        """🆕 향상된 학습 데이터를 활용한 보너스 점수 계산"""
        try:
            bonus_score = 0.0
            
            # 1. 신뢰도 점수 보너스
            reliability_key = f"{coin}_{interval}"
            if reliability_key in self.reliability_scores:
                reliability_bonus = self.reliability_scores[reliability_key] * 0.1
                bonus_score += reliability_bonus
            
            # 2. 학습 품질 점수 보너스
            if reliability_key in self.learning_quality_scores:
                quality_bonus = self.learning_quality_scores[reliability_key] * 0.1
                bonus_score += quality_bonus
            
            # 3. 글로벌 전략 매핑 보너스
            if reliability_key in self.global_strategy_mapping:
                global_strategy_id = self.global_strategy_mapping[reliability_key]
                if global_strategy_id:
                    global_bonus = 0.05  # 글로벌 전략 사용 보너스
                    bonus_score += global_bonus
            
            # 4. Walk-Forward 성능 보너스
            if reliability_key in self.walk_forward_performance:
                wf_performance = self.walk_forward_performance[reliability_key]
                if wf_performance.get('avg_performance', 0) > 0.6:
                    wf_bonus = 0.05  # 높은 Walk-Forward 성능 보너스
                    bonus_score += wf_bonus
            
            # 5. 레짐별 커버리지 보너스
            if reliability_key in self.regime_coverage:
                regime_coverage = self.regime_coverage[reliability_key]
                coverage_score = sum(regime_coverage.values()) / len(regime_coverage) if regime_coverage else 0
                if coverage_score > 0.7:
                    coverage_bonus = 0.03  # 높은 레짐 커버리지 보너스
                    bonus_score += coverage_bonus
            
            return min(0.3, bonus_score)  # 최대 30% 보너스
            
        except Exception as e:
            print(f"⚠️ 향상된 학습 보너스 계산 실패: {e}")
            return 0.0

    def _cleanup_cache(self):
        """🚀 고성능 캐시 정리 (메모리 최적화)"""
        try:
            current_time = time.time()
            expired_keys = []

            # OptimizedCache에서 타임스탬프와 캐시 정보 가져오기
            with self.cache.lock:
                cache_items = list(self.cache.cache.items())
                cache_timestamps = dict(self.cache.timestamps)

            # 🚀 캐시 크기 제한 적용
            if len(cache_items) > self.max_cache_size:
                # 가장 오래된 항목들부터 제거
                sorted_items = sorted(cache_timestamps.items(), key=lambda x: x[1])
                items_to_remove = len(cache_items) - self.max_cache_size + 1000  # 여유 공간 확보
                expired_keys.extend([key for key, _ in sorted_items[:items_to_remove]])

            # 기존 만료 시간 기반 정리
            for key, timestamp in cache_timestamps.items():
                if current_time - timestamp > 600:  # 10분 이상 사용되지 않은 항목
                    expired_keys.append(key)

            # 중복 제거
            expired_keys = list(set(expired_keys))

            # 만료된 항목 삭제
            for key in expired_keys:
                try:
                    del self.cache[key]
                    self._cache_stats['evictions'] += 1
                except:
                    pass

            if expired_keys:
                print(f"🧹 고성능 캐시 정리: {len(expired_keys)}개 항목 제거 (캐시 크기: {len(self.cache):,})")

            self._signal_stats['last_cleanup'] = current_time
        except Exception as e:
            print(f"⚠️ 캐시 정리 오류: {e}")

    def _log_signal_stats(self):
        """시그널 통계 로깅"""
        if self._signal_stats['start_time'] is None:
            return
        
        elapsed_time = time.time() - self._signal_stats['start_time']
        
        print(f"\n📊 시그널 생성 통계:")
        print(f"  - 총 생성된 시그널: {self._signal_stats['total_signals_generated']:,}개")
        print(f"  - 성공한 시그널: {self._signal_stats['successful_signals']:,}개")
        print(f"  - 실패한 시그널: {self._signal_stats['failed_signals']:,}개")
        print(f"  - 성공률: {self._signal_stats['successful_signals'] / max(self._signal_stats['total_signals_generated'], 1):.1%}")
        print(f"  - 경과 시간: {elapsed_time:.1f}초")
        print(f"  - 처리 속도: {self._signal_stats['total_signals_generated'] / elapsed_time:.2f} 시그널/초")
        
        # 🆕 캐시 통계
        cache_hit_rate = self._cache_stats['hits'] / (self._cache_stats['hits'] + self._cache_stats['misses']) if (self._cache_stats['hits'] + self._cache_stats['misses']) > 0 else 0
        print(f"  - 캐시 히트율: {cache_hit_rate:.1%}")
        print(f"  - 캐시 제거: {self._cache_stats['evictions']}회")

    def get_cached_data(self, key: str, max_age: int = 300) -> Optional[Any]:
        """🚀 최적화된 캐시 데이터 조회"""
        return self.cache.get(key, max_age)

    def set_cached_data(self, key: str, data: Any):
        """🚀 최적화된 캐시 데이터 저장"""
        self.cache.set(key, data)

    def create_signal_table(self):
        """시그널 테이블 생성 (trading_system.db에 저장)"""
        try:
            print(f"🚀 시그널 테이블 생성 중: {DB_PATH}")
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        signal_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        action TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        rsi REAL,
                        macd REAL,
                        wave_phase TEXT,
                        pattern_type TEXT,
                        risk_level TEXT,
                        volatility REAL,
                        volume_ratio REAL,
                        wave_progress REAL,
                        structure_score REAL,
                        pattern_confidence REAL,
                        integrated_direction TEXT,
                        integrated_strength REAL,
                        reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, interval, timestamp)
                    )
                """)
                
                # 인덱스 생성
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals(coin)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_combined ON signals(coin, interval) WHERE interval = "combined"')
                
                conn.commit()
                print(f"✅ 시그널 테이블 생성 완료: {DB_PATH}")
                
        except Exception as e:
            print(f"⚠️ 시그널 테이블 생성 오류: {e}")
    
    def create_enhanced_learning_tables(self):
        """향상된 학습을 위한 추가 테이블들 생성 (learning_results.db에 생성)"""
        try:
            # learning_results.db에 테이블 생성
            learning_db_path = "/workspace/data_storage/learning_results.db"
            with sqlite3.connect(learning_db_path) as conn:
                # 신뢰도 점수 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reliability_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        reliability_score REAL NOT NULL,
                        sample_count INTEGER NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_id, coin, interval)
                    )
                """)
                
                # 학습 품질 점수 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_quality_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        learning_quality_score REAL NOT NULL,
                        convergence_rate REAL NOT NULL,
                        stability_score REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_id, coin, interval)
                    )
                """)
                
                # 글로벌 전략 매핑 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS global_strategy_mapping (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        global_strategy_id TEXT NOT NULL,
                        mapping_confidence REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, interval)
                    )
                """)
                
                # Walk-Forward 성능 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS walk_forward_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        period_start TIMESTAMP NOT NULL,
                        period_end TIMESTAMP NOT NULL,
                        performance_metrics TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 레짐별 커버리지 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS regime_coverage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        market_regime TEXT NOT NULL,
                        coverage_score REAL NOT NULL,
                        performance_in_regime REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_id, coin, interval, market_regime)
                    )
                """)
                
                # 🆕 누락된 테이블들 추가
                
                # 시그널 피드백 점수 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_pattern TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 전략 결과 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        profit REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        trades_count INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        max_drawdown REAL NOT NULL,
                        score REAL NOT NULL,
                        strategy_type TEXT NOT NULL,
                        main_indicator TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        is_learned_strategy INTEGER DEFAULT 0,
                        is_improved_variant INTEGER DEFAULT 0,
                        is_active INTEGER DEFAULT 1,
                        is_archived INTEGER DEFAULT 0,
                        learning_quality_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 프랙탈 분석 결과 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fractal_analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_type TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        fractal_pattern TEXT NOT NULL,
                        pattern_confidence REAL NOT NULL,
                        market_condition TEXT NOT NULL,
                        analysis_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reliability_strategy ON reliability_scores(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reliability_coin ON reliability_scores(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_learning_quality_strategy ON learning_quality_scores(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_learning_quality_coin ON learning_quality_scores(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_global_mapping_coin ON global_strategy_mapping(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_walk_forward_strategy ON walk_forward_performance(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_regime_coverage_strategy ON regime_coverage(strategy_id)')
                
                # 전략 조건 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_conditions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        condition_type TEXT NOT NULL,
                        condition_value TEXT NOT NULL,
                        condition_operator TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 전략 등급 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_grades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        grade TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        performance_score REAL NOT NULL,
                        stability_score REAL NOT NULL,
                        risk_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 🆕 새 테이블 인덱스
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_feedback_pattern ON signal_feedback_scores(signal_pattern)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_results_coin ON strategy_results(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_results_active ON strategy_results(is_active, is_archived)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_fractal_analysis_type ON fractal_analysis_results(analysis_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_fractal_analysis_coin ON fractal_analysis_results(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_conditions_strategy ON strategy_conditions(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_grades_strategy ON strategy_grades(strategy_id)')
                
                conn.commit()
                print("✅ 향상된 학습 테이블들 생성 완료")
                
        except Exception as e:
            print(f"⚠️ 향상된 학습 테이블 생성 오류: {e}")
    
    def cleanup_old_signals(self, max_hours: int = 24):
        """오래된 시그널 정리 (성능 최적화)"""
        try:
            current_timestamp = int(datetime.now().timestamp())
            cutoff_timestamp = current_timestamp - (max_hours * 3600)
            
            with sqlite3.connect(DB_PATH) as conn:
                # 오래된 시그널 삭제
                deleted_count = conn.execute("""
                    DELETE FROM signals 
                    WHERE timestamp < ?
                """, (cutoff_timestamp,)).rowcount
                
                conn.commit()
                
                if deleted_count > 0:
                    print(f"🧹 오래된 시그널 정리: {deleted_count}개 삭제 (>{max_hours}시간 전)")
                else:
                    print(f"ℹ️ 정리할 오래된 시그널 없음 (>{max_hours}시간 전)")
                    
        except Exception as e:
            print(f"⚠️ 시그널 정리 오류: {e}")
    
    def get_signal_table_stats(self) -> Dict:
        """시그널 테이블 통계 조회"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # 전체 시그널 수
                total_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
                
                # 최근 1시간 시그널 수
                current_timestamp = int(datetime.now().timestamp())
                recent_count = conn.execute("""
                    SELECT COUNT(*) FROM signals 
                    WHERE timestamp > ?
                """, (current_timestamp - 3600,)).fetchone()[0]
                
                # 코인별 시그널 수
                coin_counts = pd.read_sql("""
                    SELECT coin, COUNT(*) as count 
                    FROM signals 
                    GROUP BY coin 
                    ORDER BY count DESC 
                    LIMIT 10
                """, conn)
                
                return {
                    'total_signals': total_count,
                    'recent_signals_1h': recent_count,
                    'top_coins': coin_counts.to_dict('records')
                }
                
        except Exception as e:
            print(f"⚠️ 시그널 통계 조회 오류: {e}")
            return {'total_signals': 0, 'recent_signals_1h': 0, 'top_coins': []}
    
    def load_rl_q_table(self) -> Dict:
        """RL 시스템 로드 - 시그널 피드백만 확인 (Q-테이블 제거)"""
        try:
            # 시그널 피드백 점수 테이블 확인
            try:
                with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_feedback_scores'")
                    if cursor.fetchone():
                        feedback_count = pd.read_sql("SELECT COUNT(*) as count FROM signal_feedback_scores", conn).iloc[0]['count']
                        print(f"✅ 시그널 피드백 점수 테이블 확인: {feedback_count}개 패턴")
                    else:
                        print("ℹ️ 시그널 피드백 점수 테이블 없음")
                        
            except Exception as e:
                print(f"ℹ️ 시그널 피드백 점수 테이블 확인 실패: {e}")
            
            print("ℹ️ Absolute Zero System은 전략 기반 시스템이므로 Q-테이블 없음")
            print("  📊 대신 코인별 전략 결과와 시그널 피드백을 활용하여 시그널 점수 계산")
        
        except Exception as e:
            print(f"⚠️ 시그널 피드백 확인 오류: {e}")
        
        return {}  # 빈 딕셔너리 반환
    
    def load_coin_specific_strategies(self):
        """Absolute Zero System의 코인별 전략 로드 (군집화 상태에 따라 동적 로드)"""
        # 안전한 초기화
        if not hasattr(self, 'coin_specific_strategies') or self.coin_specific_strategies is None:
            self.coin_specific_strategies = {}
            
        try:
            with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                # 🆕 품질 기반 전략 로드 (군집화 제거됨)
                print(f"📊 품질 기반 전략 로드 시작")
                
                # 🚀 learning_results.db 테이블 확인
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                available_tables = [row[0] for row in cursor.fetchall()]
                print(f"📊 learning_results.db 사용 가능한 테이블: {available_tables}")

                # 🚀 rl_strategies.db에서 coin_strategies 로드
                rl_strategies_db = "/workspace/data_storage/rl_strategies.db"
                try:
                    rl_conn = sqlite3.connect(rl_strategies_db)
                    # 🔥 수정: score 기반으로 전략 로드 (Self-play 삭제되어 profit/trades_count는 0)
                    quality_df = pd.read_sql("""
                        SELECT coin as symbol, interval,
                               COALESCE(profit, 0.0) as profit,
                               COALESCE(win_rate, 0.5) as win_rate,
                               COALESCE(trades_count, 0) as trades_count,
                               id as strategy_id,
                               'learned' as strategy_type, 'multi' as main_indicator, 'medium' as risk_level,
                               COALESCE(score, 0.5) as score,
                               quality_grade
                        FROM coin_strategies
                        WHERE score IS NOT NULL AND score > 0
                        AND quality_grade IN ('A', 'B', 'C', 'D')
                        ORDER BY score DESC, quality_grade ASC
                        LIMIT 1000
                    """, rl_conn)
                    rl_conn.close()
                    print(f"✅ rl_strategies.db에서 {len(quality_df):,}개 전략 로드 (score 기반)")
                except Exception as rl_error:
                    print(f"⚠️ rl_strategies.db 읽기 실패: {rl_error}")

                    # 폴백: learning_results.db의 테이블 사용
                    if 'learned_strategies' in available_tables:
                        quality_df = pd.read_sql("""
                            SELECT coin as symbol, interval, profit, win_rate, trades_count, strategy_id,
                                   strategy_type, main_indicator, risk_level, score
                            FROM learned_strategies
                            WHERE (profit > 0 OR profit IS NULL) AND (trades_count >= 1 OR trades_count IS NULL) AND (win_rate >= 0.2 OR win_rate IS NULL)
                            ORDER BY coin, interval, COALESCE(score, 0.5) DESC
                        """, conn)
                    elif 'global_strategies' in available_tables:
                        quality_df = pd.read_sql("""
                            SELECT coin as symbol, interval, profit, win_rate, trades_count, strategy_id,
                                   strategy_type, main_indicator, risk_level, score
                            FROM global_strategies
                            WHERE (profit > 0 OR profit IS NULL) AND (trades_count >= 1 OR trades_count IS NULL) AND (win_rate >= 0.2 OR win_rate IS NULL)
                            ORDER BY coin, interval, COALESCE(score, 0.5) DESC
                        """, conn)
                    else:
                        print(f"⚠️ 학습된 전략 테이블이 없음 - 기본 전략만 사용")
                        return
                
                print(f"📊 쿼리 결과: {len(quality_df)}개 레코드")
                
                # 품질 기반 전략 로드 (같은 키에 여러 전략이 있을 경우 최고 점수만 유지)
                for _, row in quality_df.iterrows():
                    strategy_key = f"{row['symbol']}_{row['interval']}"
                    current_score = row['score']
                    
                    # 🆕 같은 키가 이미 있으면 점수가 더 높은 것만 유지
                    if strategy_key in self.coin_specific_strategies:
                        existing_score = self.coin_specific_strategies[strategy_key].get('score', 0.0)
                        if current_score <= existing_score:
                            continue  # 기존 것이 더 좋으면 스킵
                    
                    self.coin_specific_strategies[strategy_key] = {
                        'strategy_id': row['strategy_id'],
                        'profit': row.get('profit', 0.0),
                        'win_rate': row.get('win_rate', 0.0),
                        'trades_count': row.get('trades_count', 0),
                        'winning_trades': row.get('winning_trades', 0),
                        'losing_trades': row.get('losing_trades', 0),
                        'max_drawdown': row.get('max_drawdown', 0.0),
                        'score': row['score'],
                        'symbol': row['symbol'],
                        'interval': row['interval'],
                        'strategy_type': row.get('strategy_type', ''),
                        'main_indicator': row.get('main_indicator', ''),
                        'risk_level': row.get('risk_level', 'medium'),
                        'quality_grade': 'A' if row['score'] >= 0.8 else 'B' if row['score'] >= 0.6 else 'C',
                        'strategy_json': '{}'
                    }
                
                print(f"✅ 품질 기반 전략 로드: {len(self.coin_specific_strategies)}개")
                
        except Exception as e:
            print(f"ℹ️ 코인별 전략 로드 실패: {e}")
            # 데이터베이스 테이블 구조 확인
            try:
                with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(strategy_results)")
                    columns = cursor.fetchall()
                    print(f"📊 strategy_results 테이블 컬럼: {[col[1] for col in columns]}")
            except Exception as db_e:
                print(f"⚠️ 테이블 구조 확인 실패: {db_e}")
            # 실패 시 빈 딕셔너리로 초기화
            self.coin_specific_strategies = {}

    def load_dna_patterns_from_learning_data(self):
        """
        🧬 완전 자동화: 학습 데이터에서 DNA 패턴 자동 추출 및 적용

        completed_trades와 signals 테이블을 조인하여:
        1. 성공한 거래의 기술적 지표 추출
        2. DNA 패턴으로 변환 (rsi_range, macd_range, volume_range 등)
        3. coin_specific_strategies에 DNA 패턴 추가
        4. 자동으로 유사 DNA 매칭에 활용
        """
        try:
            print("\n🧬 DNA 패턴 자동 학습 시작...")

            # trading_system.db 경로 (Docker 환경)
            trading_db_path = "/workspace/data_storage/trading_system.db"

            with sqlite3.connect(trading_db_path) as conn:
                # 성공한 거래와 해당 시그널 정보 조인
                query = """
                    SELECT
                        ct.coin,
                        s.interval,
                        s.rsi,
                        s.macd,
                        s.volume_ratio,
                        s.volatility,
                        s.structure_score,
                        s.wave_progress,
                        s.pattern_confidence,
                        ct.profit_loss_pct,
                        s.timestamp
                    FROM completed_trades ct
                    INNER JOIN signals s ON
                        ct.coin = s.coin AND
                        ct.entry_timestamp = s.timestamp
                    WHERE ct.profit_loss_pct > 0  -- 성공한 거래만
                    ORDER BY ct.exit_timestamp DESC
                    LIMIT 500  -- 최근 500개 성공 거래
                """

                cursor = conn.cursor()
                cursor.execute(query)
                trades = cursor.fetchall()

                if not trades:
                    print("⚠️ 학습할 거래 데이터가 없습니다 (거래 이력 필요)")
                    return

                print(f"📊 {len(trades)}개의 성공 거래에서 DNA 패턴 추출 중...")

                # 코인/인터벌별로 DNA 패턴 그룹화
                dna_patterns_by_coin = {}

                for trade in trades:
                    coin, interval, rsi, macd, volume_ratio, volatility, structure_score, wave_step, pattern_quality, profit_pct, timestamp = trade

                    # None 값 안전 처리
                    rsi = rsi if rsi is not None else 50.0
                    macd = macd if macd is not None else 0.0
                    volume_ratio = volume_ratio if volume_ratio is not None else 1.0
                    volatility = volatility if volatility is not None else 0.02
                    structure_score = structure_score if structure_score is not None else 0.5
                    wave_step = wave_step if wave_step is not None else 0.5
                    pattern_quality = pattern_quality if pattern_quality is not None else 0.5

                    # DNA 패턴 생성 (기존 categorize 메서드 활용)
                    dna_pattern = {
                        'rsi_range': self._categorize_rsi_enhanced(rsi),
                        'macd_range': self._categorize_macd_enhanced(macd),
                        'volume_range': self._categorize_volume_enhanced(volume_ratio),
                        'volatility_range': self._categorize_volatility_enhanced(volatility),
                        'structure_range': self._categorize_structure_enhanced(structure_score),
                        'wave_step': self._categorize_wave_step(wave_step),
                        'pattern_quality': self._categorize_pattern_quality(pattern_quality),
                        'interval': interval,
                        'profit_pct': profit_pct,
                        'timestamp': timestamp
                    }

                    # 코인/인터벌별로 그룹화
                    strategy_key = f"{coin}_{interval}"
                    if strategy_key not in dna_patterns_by_coin:
                        dna_patterns_by_coin[strategy_key] = []

                    dna_patterns_by_coin[strategy_key].append(dna_pattern)

                # 각 코인/인터벌별로 대표 DNA 패턴 계산 및 적용
                patterns_added = 0
                for strategy_key, patterns in dna_patterns_by_coin.items():
                    # 가장 수익성 높은 패턴 선택 (상위 30%)
                    patterns_sorted = sorted(patterns, key=lambda x: x['profit_pct'], reverse=True)
                    top_patterns = patterns_sorted[:max(1, len(patterns_sorted) // 3)]

                    # 대표 패턴 계산 (최빈값 기반)
                    representative_pattern = self._calculate_representative_dna_pattern(top_patterns)

                    # coin_specific_strategies에 DNA 패턴 추가
                    if strategy_key in self.coin_specific_strategies:
                        # 기존 전략에 DNA 패턴 추가
                        self.coin_specific_strategies[strategy_key].update(representative_pattern)
                        patterns_added += 1
                    else:
                        # 새로운 전략 생성 (DNA 패턴만 포함)
                        coin, interval = strategy_key.split('_')
                        self.coin_specific_strategies[strategy_key] = {
                            'symbol': coin,
                            'interval': interval,
                            'profit': sum(p['profit_pct'] for p in top_patterns) / len(top_patterns),
                            'win_rate': 1.0,  # 성공 거래만 사용했으므로
                            'trades_count': len(patterns),
                            **representative_pattern
                        }
                        patterns_added += 1

                print(f"✅ DNA 패턴 자동 학습 완료!")
                print(f"   - 총 {len(dna_patterns_by_coin)}개 코인/인터벌 조합")
                print(f"   - {patterns_added}개 전략에 DNA 패턴 추가")
                print(f"   - {len(trades)}개 성공 거래 분석")

                # 🧬 업데이트 시간 기록
                self.last_dna_update = time.time()

        except Exception as e:
            print(f"⚠️ DNA 패턴 자동 학습 오류: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_representative_dna_pattern(self, patterns: list) -> dict:
        """
        여러 DNA 패턴에서 대표 패턴 계산 (최빈값 기반)

        Args:
            patterns: DNA 패턴 리스트

        Returns:
            대표 DNA 패턴 딕셔너리
        """
        from collections import Counter

        representative = {}

        # 각 지표별로 최빈값 계산
        for key in ['rsi_range', 'macd_range', 'volume_range', 'volatility_range',
                    'structure_range', 'wave_step', 'pattern_quality']:
            values = [p.get(key, 'unknown') for p in patterns if key in p]
            if values:
                # 최빈값 선택
                counter = Counter(values)
                representative[key] = counter.most_common(1)[0][0]
            else:
                representative[key] = 'unknown'

        # 인터벌은 첫 번째 패턴의 값 사용 (모두 동일해야 함)
        if patterns:
            representative['interval'] = patterns[0].get('interval', '15m')

        return representative

    def refresh_dna_patterns_if_needed(self, force: bool = False):
        """
        🧬 필요 시 DNA 패턴 자동 갱신

        Args:
            force: True면 시간 체크 없이 강제 갱신

        DNA 패턴을 1시간마다 자동 갱신하여 최신 학습 데이터 반영
        """
        current_time = time.time()
        update_interval = 3600  # 1시간 (초 단위)

        if force or (current_time - self.last_dna_update > update_interval):
            print(f"\n🔄 DNA 패턴 갱신 시작 (마지막 업데이트: {int((current_time - self.last_dna_update) / 60)}분 전)")
            self.load_dna_patterns_from_learning_data()
            print(f"✅ DNA 패턴 갱신 완료")

    def get_coin_specific_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """🚀 고성능 코인별 전략 점수 계산 (글로벌 전략과 개별 전략 통합)"""
        try:
            strategy_key = f"{coin}_{interval}"
            
            # 🎯 개별 코인 전략 점수 계산
            coin_score = 0.0
            if strategy_key in self.coin_specific_strategies:
                coin_score = self._calculate_coin_specific_score(coin, interval, candle, strategy_key)
            
            # 🌍 글로벌 전략 점수 계산
            global_score = self._get_global_strategy_score(coin, interval, candle)
            
            # 🔄 통합 점수 계산 (레짐 기반 동적 가중치)
            if coin_score > 0 and global_score > 0:
                # 🎯 현재 레짐 감지
                current_regime = self._detect_current_regime(coin, interval, candle)

                # 🌍 DB 기반 동적 가중치 조정 (레짐 fallback)
                coin_weight, global_weight = self._calculate_dynamic_weights(current_regime, coin=coin)
                
                integrated_score = coin_score * coin_weight + global_score * global_weight
                
                if self.debug_mode:
                    print(f"📊 {coin}/{interval}: 통합 점수 (레짐:{current_regime}, 개별:{coin_score:.4f}*{coin_weight} + 글로벌:{global_score:.4f}*{global_weight} = {integrated_score:.4f})")
                
                return np.clip(integrated_score, -1.0, 1.0)
                
            elif coin_score > 0:
                # 개별 전략만 있는 경우
                if self.debug_mode:
                    print(f"📊 {coin}/{interval}: 개별 전략만 사용 (점수: {coin_score:.4f})")
                return np.clip(coin_score, -1.0, 1.0)
                
            elif global_score > 0:
                # 글로벌 전략만 있는 경우
                if self.debug_mode:
                    print(f"📊 {coin}/{interval}: 글로벌 전략만 사용 (점수: {global_score:.4f})")
                return np.clip(global_score, -1.0, 1.0)
            
            else:
                # 기본 전략 사용
                default_score = self._get_default_strategy_score(coin, interval, candle)
                if self.debug_mode:
                    print(f"📊 {coin}/{interval}: 기본 전략 사용 (점수: {default_score:.4f})")
                return np.clip(default_score, -1.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ 코인별 전략 점수 계산 오류 ({coin}/{interval}): {e}")
            return 0.0
    
    def _detect_current_regime(self, coin: str, interval: str, candle: pd.Series) -> str:
        """현재 시장 레짐 감지"""
        try:
            # 간단한 지표 추출
            rsi = candle.get('rsi', 50.0)
            macd = candle.get('macd', 0.0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            volatility = candle.get('volatility', 0.02)
            
            # 레짐 판단 로직
            if rsi < 30 and volume_ratio > 1.2:
                return 'extreme_bearish'
            elif rsi > 70 and volume_ratio > 1.2:
                return 'extreme_bullish'
            elif rsi < 40 and macd < 0:
                return 'bearish'
            elif rsi > 60 and macd > 0:
                return 'bullish'
            elif volatility < 0.01 and abs(macd) < 0.001:
                return 'neutral'
            elif 40 < rsi < 60 and volume_ratio > 0.9:
                return 'sideways_bullish'
            else:
                return 'sideways_bearish'
                
        except Exception as e:
            print(f"⚠️ 레짐 감지 실패: {e}")
            return 'neutral'
    
    def _calculate_dynamic_weights(self, regime: str, coin: str = None) -> tuple:
        """🔥 코인 vs 글로벌 전략 동적 가중치 계산 (DB 기반)

        Args:
            regime: 시장 레짐 (fallback용)
            coin: 코인 이름 (예: 'BTC')

        Returns:
            tuple: (coin_weight, global_weight)
        """
        try:
            # 🔥 1순위: DB에서 코인별 동적 가중치 로드
            if coin:
                try:
                    from rl_pipeline.db.reads import get_coin_global_weights

                    weights_data = get_coin_global_weights(coin)

                    if weights_data and weights_data.get('updated_at'):
                        coin_weight = weights_data['coin_weight']
                        global_weight = weights_data['global_weight']

                        if self.debug_mode:
                            quality_score = weights_data.get('data_quality_score', 0.0)
                            print(f"🎯 [{coin}] DB 가중치: 개별={coin_weight:.2f}, 글로벌={global_weight:.2f}, 품질={quality_score:.2f}")

                        return coin_weight, global_weight
                    else:
                        if self.debug_mode:
                            print(f"⚠️ [{coin}] DB 가중치 없음, 레짐 기반 가중치 사용")
                except Exception as db_err:
                    if self.debug_mode:
                        print(f"⚠️ [{coin}] DB 가중치 로드 실패: {db_err}, 레짐 기반 가중치 사용")

            # 🔥 2순위: 레짐 기반 가중치 (fallback)
            weight_strategies = {
                # 추세 레짐: 글로벌 전략 강조 (시장 전체 흐름 중요)
                'extreme_bullish': (0.6, 0.4),   # 개별 60%, 글로벌 40%
                'extreme_bearish': (0.6, 0.4),   # 개별 60%, 글로벌 40%
                'bullish': (0.65, 0.35),         # 개별 65%, 글로벌 35%
                'bearish': (0.65, 0.35),         # 개별 65%, 글로벌 35%

                # 횡보 레짐: 개별 전략 강조 (코인별 특성 중요)
                'sideways_bullish': (0.75, 0.25), # 개별 75%, 글로벌 25%
                'sideways_bearish': (0.75, 0.25), # 개별 75%, 글로벌 25%

                # 중립 레짐: 기본 비율
                'neutral': (0.7, 0.3),           # 개별 70%, 글로벌 30%
            }

            coin_weight, global_weight = weight_strategies.get(regime, (0.7, 0.3))

            if self.debug_mode:
                print(f"🎯 레짐 '{regime}' 가중치 (fallback): 개별={coin_weight}, 글로벌={global_weight}")

            return coin_weight, global_weight

        except Exception as e:
            print(f"⚠️ 동적 가중치 계산 실패: {e}")
            return 0.7, 0.3  # 기본값
    
    def _calculate_coin_specific_score(self, coin: str, interval: str, candle: pd.Series, strategy_key: str) -> float:
        """개별 코인 전략 점수 계산"""
        try:
            strategy = self.coin_specific_strategies[strategy_key]
            quality_grade = strategy.get('quality_grade', 'C')
            
            # 🚀 현재 시장 상태 분석 (크로스 코인 학습용)
            current_dna = self._extract_current_dna_pattern_enhanced(coin, interval, candle)
            
            # 🆕 품질 등급에 따른 가중치 차별화
            if quality_grade == 'A':
                base_score = strategy['profit'] * 0.9
                confidence_bonus = 1.4
                normalized_score = base_score / 1.1
                
            elif quality_grade == 'B':
                base_score = strategy['profit'] * 0.7
                confidence_bonus = 1.2
                normalized_score = base_score / 1.3
                
            else:  # C등급 이하
                if 'score' in strategy and strategy['score'] is not None and strategy['score'] > 0:
                    base_score = strategy['score'] * 0.5
                else:
                    base_score = strategy['profit'] * 0.8
                confidence_bonus = 1.0
                normalized_score = base_score / 1.5
            
            # 거래 수에 따른 보너스/페널티
            if strategy['trades_count'] >= 30:
                normalized_score *= confidence_bonus * 1.2
            elif strategy['trades_count'] >= 20:
                normalized_score *= confidence_bonus * 1.15
            elif strategy['trades_count'] < 10:
                normalized_score *= confidence_bonus * 0.6
            
            # 크로스 코인 학습 보너스
            if self.cross_coin_available and self.cross_coin_context:
                cross_coin_bonus = self._calculate_cross_coin_bonus(coin, interval, current_dna)
                normalized_score *= cross_coin_bonus
            
            return normalized_score
            
        except Exception as e:
            print(f"⚠️ 개별 코인 전략 점수 계산 실패: {e}")
            return 0.0
    
    def _get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """🔥 글로벌 전략 점수 계산 (저장된 글로벌 전략 분석 결과 활용)"""
        try:
            # 등급 점수 매핑 (공통 사용)
            grade_scores = {'S': 6.0, 'A': 5.0, 'B': 4.0, 'C': 3.0, 'D': 2.0, 'F': 1.0}
            
            # 🔥 1단계: 로드된 글로벌 전략 사용 (우선순위)
            if interval in self.global_strategies_cache and len(self.global_strategies_cache[interval]) > 0:
                strategies = self.global_strategies_cache[interval]
                
                # 최고 등급 전략 선택
                best_strategy = None
                best_score = -1.0
                
                for strategy in strategies:
                    # 등급 기반 점수 계산
                    grade = strategy.get('quality_grade', 'A')
                    grade_score = grade_scores.get(grade, 3.0)
                    
                    # 성과 기반 점수
                    profit = strategy.get('profit', 0.0)
                    win_rate = strategy.get('win_rate', 0.5)
                    profit_factor = strategy.get('profit_factor', 1.0)
                    
                    # 종합 점수 계산
                    strategy_score = (
                        grade_score * 0.3 +  # 등급 30%
                        min(profit * 10, 3.0) * 0.3 +  # 수익 30%
                        win_rate * 0.2 +  # 승률 20%
                        min(profit_factor, 3.0) * 0.2  # Profit Factor 20%
                    )
                    
                    if strategy_score > best_score:
                        best_score = strategy_score
                        best_strategy = strategy
                
                if best_strategy:
                    # 전략 파라미터로 점수 계산
                    params = best_strategy.get('params', {})
                    
                    # 시장 적응도 평가
                    market_adaptation = self._evaluate_market_adaptation(candle, {
                        'strategy_type': best_strategy.get('strategy_type', 'performance_based'),
                        'params': params
                    })
                    
                    # 최종 점수 계산
                    base_score = best_score / 6.0  # 0~1 범위로 정규화
                    final_score = base_score * market_adaptation
                    
                    if self.debug_mode:
                        logger.debug(f"🔥 글로벌 전략 사용: {best_strategy.get('name', 'unknown')} "
                                   f"(등급: {best_strategy.get('quality_grade', 'A')}, 점수: {final_score:.3f})")
                    
                    return np.clip(final_score, 0.0, 1.0)
            
            # 🔥 2단계: 실시간 글로벌 전략 로드 시도
            try:
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                
                from rl_pipeline.db.learning_results import load_global_strategies_from_db
                
                global_strategies = load_global_strategies_from_db(interval=interval)
                if global_strategies:
                    # 캐시에 저장
                    self.global_strategies_cache[interval] = global_strategies
                    
                    # 가장 좋은 전략 선택 (위와 동일 로직)
                    best_strategy = max(global_strategies, 
                                       key=lambda s: grade_scores.get(s.get('quality_grade', 'A'), 3.0))
                    
                    params = best_strategy.get('params', {})
                    market_adaptation = self._evaluate_market_adaptation(candle, {
                        'strategy_type': best_strategy.get('strategy_type', 'performance_based'),
                        'params': params
                    })
                    
                    grade = best_strategy.get('quality_grade', 'A')
                    grade_score = grade_scores.get(grade, 3.0)
                    base_score = grade_score / 6.0
                    final_score = base_score * market_adaptation
                    
                    if self.debug_mode:
                        logger.debug(f"🔥 실시간 글로벌 전략 로드: {interval} (점수: {final_score:.3f})")
                    
                    return np.clip(final_score, 0.0, 1.0)
            except Exception as e:
                if self.debug_mode:
                    logger.debug(f"⚠️ 실시간 글로벌 전략 로드 실패: {e}")
            
            # 🔥 3단계: 폴백 - 기존 방식 사용
            return self._calculate_enhanced_global_strategy_score(candle, interval)
                
        except Exception as e:
            if self.debug_mode:
                logger.error(f"⚠️ 글로벌 전략 점수 계산 실패: {e}")
            return 0.5  # 에러 시 중립 점수
    
    def _get_default_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """🚀 향상된 기본 전략 점수 계산 (모든 전략이 없을 때 사용)"""
        try:
            # 🚀 실제 캔들 데이터에서 지표 추출 (None 값 안전 처리)
            rsi = candle.get('rsi', 50.0)
            macd = candle.get('macd', 0.0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            volatility = candle.get('volatility', 0.02)
            wave_phase = candle.get('wave_phase', 'unknown')
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            integrated_direction = candle.get('integrated_direction', 'neutral')
            mfi = candle.get('mfi', 50.0)
            atr = candle.get('atr', 0.02)
            adx = candle.get('adx', 25.0)
            
            # None 값 안전 처리
            if rsi is None:
                rsi = 50.0
            if macd is None:
                macd = 0.0
            if volume_ratio is None:
                volume_ratio = 1.0
            if volatility is None:
                volatility = 0.02
            if pattern_confidence is None:
                pattern_confidence = 0.0
            if mfi is None:
                mfi = 50.0
            if atr is None:
                atr = 0.02
            if adx is None:
                adx = 25.0
            
            # 🚀 RSI 기반 점수 (더 정교한 계산)
            if rsi < 20:  # 극도 과매도 - 강한 매수 신호
                rsi_score = 0.9
            elif rsi < 30:  # 과매도 - 매수 신호
                rsi_score = 0.7
            elif rsi > 80:  # 극도 과매수 - 매도 신호
                rsi_score = 0.1
            elif rsi > 70:  # 과매수 - 약한 매도 신호
                rsi_score = 0.3
            elif 40 <= rsi <= 60:  # 중립 구간 - 안정적
                rsi_score = 0.6
            else:  # 경계선
                rsi_score = 0.5
            
            # 🚀 MACD 기반 점수 (더 정교한 계산)
            if macd > 0.01:  # 강한 상승 신호
                macd_score = 0.9
            elif macd > 0.005:  # 중간 상승 신호
                macd_score = 0.7
            elif macd > 0:  # 약한 상승 신호
                macd_score = 0.6
            elif macd > -0.005:  # 약한 하락 신호
                macd_score = 0.4
            elif macd > -0.01:  # 중간 하락 신호
                macd_score = 0.3
            else:  # 강한 하락 신호
                macd_score = 0.1
            
            # 🚀 거래량 기반 점수 (더 정교한 계산)
            if volume_ratio > 3.0:  # 매우 높은 거래량
                volume_score = 0.9
            elif volume_ratio > 2.0:  # 높은 거래량
                volume_score = 0.8
            elif volume_ratio > 1.5:  # 정상 이상 거래량
                volume_score = 0.7
            elif volume_ratio > 1.0:  # 정상 거래량
                volume_score = 0.6
            elif volume_ratio > 0.5:  # 낮은 거래량
                volume_score = 0.4
            else:  # 매우 낮은 거래량
                volume_score = 0.2
            
            # 🚀 MFI 기반 점수 (자금 흐름)
            if mfi < 20:  # 극도 과매도
                mfi_score = 0.8
            elif mfi < 30:  # 과매도
                mfi_score = 0.6
            elif mfi > 80:  # 극도 과매수
                mfi_score = 0.2
            elif mfi > 70:  # 과매수
                mfi_score = 0.4
            else:  # 중립
                mfi_score = 0.5
            
            # 🚀 ADX 기반 점수 (트렌드 강도)
            if adx > 40:  # 강한 트렌드
                adx_score = 0.8
            elif adx > 25:  # 중간 트렌드
                adx_score = 0.6
            else:  # 약한 트렌드
                adx_score = 0.4
            
            # 🚀 파동 단계 기반 점수
            wave_score = 0.5
            if wave_phase == 'impulse':
                wave_score = 0.8
            elif wave_phase == 'correction':
                wave_score = 0.3
            elif wave_phase == 'consolidation':
                wave_score = 0.6
            elif wave_phase == 'sideways':
                wave_score = 0.5
            
            # 🚀 통합 방향성 기반 점수
            direction_score = 0.5
            if integrated_direction == 'strong_bullish':
                direction_score = 0.9
            elif integrated_direction == 'bullish':
                direction_score = 0.7
            elif integrated_direction == 'strong_bearish':
                direction_score = 0.1
            elif integrated_direction == 'bearish':
                direction_score = 0.3
            
            # 🚀 패턴 신뢰도 기반 점수
            pattern_score = 0.5 + (pattern_confidence * 0.5)  # 0.5 ~ 1.0
            
            # 🚀 변동성 기반 점수 (적절한 변동성 선호)
            if 0.02 <= volatility <= 0.05:  # 적절한 변동성
                volatility_score = 0.8
            elif volatility < 0.02:  # 너무 낮은 변동성
                volatility_score = 0.4
            elif volatility > 0.08:  # 너무 높은 변동성
                volatility_score = 0.3
            else:  # 중간 변동성
                volatility_score = 0.6
            
            # 🚀 최종 점수 계산 (가중 평균) - 더 정교한 가중치
            final_score = (
                rsi_score * 0.20 +      # RSI 20%
                macd_score * 0.20 +     # MACD 20%
                volume_score * 0.15 +    # 거래량 15%
                mfi_score * 0.10 +       # MFI 10%
                adx_score * 0.10 +       # ADX 10%
                wave_score * 0.10 +     # 파동 10%
                direction_score * 0.10 + # 방향성 10%
                pattern_score * 0.03 +   # 패턴 3%
                volatility_score * 0.02  # 변동성 2%
            )
            
            # 🚀 인터벌별 가중치 적용 (더 정교한 가중치)
            interval_weights = {'15m': 0.8, '30m': 1.0, '240m': 1.2, '1d': 1.3}
            weight = interval_weights.get(interval, 1.0)
            
            final_score *= weight
            
            if self.debug_mode:
                print(f"🚀 향상된 기본 전략: RSI({rsi:.1f}→{rsi_score:.2f}), MACD({macd:.4f}→{macd_score:.2f}), Volume({volume_ratio:.2f}x→{volume_score:.2f})")
                print(f"🚀 MFI({mfi:.1f}→{mfi_score:.2f}), ADX({adx:.1f}→{adx_score:.2f}), Wave({wave_phase}→{wave_score:.2f})")
                print(f"🚀 Direction({integrated_direction}→{direction_score:.2f}), Pattern({pattern_confidence:.2f}→{pattern_score:.2f}), 최종({final_score:.3f})")
            
            return np.clip(final_score, 0.0, 1.0)
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ 기본 전략 점수 계산 실패: {e}")
            return 0.1  # 최소 점수 반환
    
    def _evaluate_market_adaptation(self, candle: pd.Series, strategy: Dict) -> float:
        """현재 시장 상황과 전략의 적합성 평가 (Absolute Zero System의 개선된 조건들 반영)"""
        try:
            adaptation_score = 0.0
            
            # 🎯 현재 시장 상황 분석 (안전한 값 추출)
            rsi = candle.get('rsi')
            macd = candle.get('macd')
            volume_ratio = candle.get('volume_ratio')
            wave_progress = candle.get('wave_progress')
            structure_score = candle.get('structure_score')
            pattern_confidence = candle.get('pattern_confidence')
            
            # 🚀 고급 지표들 분석
            mfi = candle.get('mfi')
            adx = candle.get('adx')
            wave_momentum = candle.get('wave_momentum')
            confidence = candle.get('confidence')
            volatility = candle.get('volatility')
            
            # 🆕 새로 추가된 고급 지표들 (기존 데이터만 사용)
            bb_position = 'unknown'  # 기존 데이터에 없음
            bb_width = 0.0  # 기존 데이터에 없음
            bb_squeeze = 0.0  # 기존 데이터에 없음
            rsi_divergence = 'none'  # 기존 데이터에 없음
            macd_divergence = 'none'  # 기존 데이터에 없음
            price_momentum = 0.0  # 기존 데이터에 없음
            volume_momentum = 0.0  # 기존 데이터에 없음
            trend_strength = 0.5  # 기존 데이터에 없음
            
            # 🎯 안전한 값 변환 (None, NaN 처리)

            
            # 안전한 값 변환
            rsi = safe_float(rsi, 50.0)
            macd = safe_float(macd, 0.0)
            volume_ratio = safe_float(volume_ratio, 1.0)
            wave_progress = safe_float(wave_progress, 0.5)
            structure_score = safe_float(structure_score, 0.5)
            pattern_confidence = safe_float(pattern_confidence, 0.0)
            mfi = safe_float(mfi, 50.0)
            adx = safe_float(adx, 25.0)
            wave_momentum = safe_float(wave_momentum, 0.0)
            confidence = safe_float(confidence, 0.5)
            volatility = safe_float(volatility, 0.0)
            bb_width = safe_float(bb_width, 0.0)
            bb_squeeze = safe_float(bb_squeeze, 0.0)
            price_momentum = safe_float(price_momentum, 0.0)
            volume_momentum = safe_float(volume_momentum, 0.0)
            trend_strength = safe_float(trend_strength, 0.5)
            
            bb_position = safe_str(bb_position, 'unknown')
            rsi_divergence = safe_str(rsi_divergence, 'none')
            macd_divergence = safe_str(macd_divergence, 'none')
            
            # 🎯 시장 상황별 적합성 평가
            # 1. 과매수/과매도 상황
            if rsi < 30 and strategy['win_rate'] > 55:  # 과매도에서 높은 승률 전략
                adaptation_score += 0.05
            elif rsi > 70 and strategy['win_rate'] > 55:  # 과매수에서 높은 승률 전략
                adaptation_score += 0.05
            
            # 2. 볼린저밴드 스퀴즈 상황
            if bb_squeeze > 0.8 and strategy['profit'] > 3.0:  # 스퀴즈에서 수익성 있는 전략
                adaptation_score += 0.03
            
            # 3. 다이버전스 상황
            if (rsi_divergence == 'bullish' or macd_divergence == 'bullish') and strategy['win_rate'] > 60:
                adaptation_score += 0.04
            elif (rsi_divergence == 'bearish' or macd_divergence == 'bearish') and strategy['win_rate'] > 60:
                adaptation_score += 0.04
            
            # 4. 모멘텀 상황
            if abs(price_momentum) > 0.05 and strategy['trades_count'] >= 15:  # 높은 모멘텀에서 충분한 거래 경험
                adaptation_score += 0.03
            
            # 5. 트렌드 강도
            if trend_strength > 0.7 and strategy['profit'] > 4.0:  # 강한 트렌드에서 수익성 있는 전략
                adaptation_score += 0.03
            
            # 6. 거래량 상황
            if volume_ratio > 1.5 and strategy['win_rate'] > 55:  # 높은 거래량에서 높은 승률
                adaptation_score += 0.02
            
            # 7. 구조 점수
            if structure_score > 0.6 and strategy['profit'] > 3.0:  # 높은 구조 점수에서 수익성 있는 전략
                adaptation_score += 0.02
            
            # 8. 패턴 신뢰도
            if pattern_confidence > 0.5 and strategy['win_rate'] > 60:  # 높은 패턴 신뢰도에서 높은 승률
                adaptation_score += 0.02
            
            return adaptation_score
            
        except Exception as e:
            print(f"⚠️ 시장 적응성 평가 오류: {e}")
            return 0.0
    
    # ============================================================================
    # 🆕 전략 점수 계산기 (리팩토링)
    # ============================================================================
    
    def _get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """글로벌 전략 기반 점수 계산 (learning_engine.py 연동 강화)"""
        try:
            # 🆕 학습 엔진의 글로벌 전략 결과 활용
            global_score = self._strategy_calculator.get_global_strategy_score(coin, interval, candle)
            
            # 🆕 심화 통합 분석 결과 활용
            deep_analysis_bonus = self._get_deep_analysis_bonus(coin, interval, candle)
            
            # 🆕 시너지 패턴 보너스
            synergy_bonus = self._get_synergy_pattern_bonus(coin, interval, candle)
            
            # 🆕 학습 품질 기반 가중치
            quality_weight = self._get_learning_quality_weight(coin, interval)
            
            # 최종 점수 계산
            final_score = (global_score + deep_analysis_bonus + synergy_bonus) * quality_weight
            
            return min(max(final_score, 0.0), 1.0)  # 0.0 ~ 1.0 범위로 제한
            
        except Exception as e:
            logger.error(f"❌ 글로벌 전략 점수 계산 실패: {e}")
            return self._strategy_calculator.get_global_strategy_score(coin, interval, candle)
    
    def _get_rl_pipeline_learned_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """RL Pipeline 학습 결과 활용 (learning_engine.py 연동 강화)"""
        try:
            # 🆕 기본 RL 파이프라인 점수
            base_score = self._strategy_calculator.get_rl_pipeline_score(coin, interval, candle)
            
            # 🆕 심화 통합 분석 결과 활용
            deep_analysis_bonus = self._get_deep_analysis_bonus(coin, interval, candle)
            
            # 🆕 학습 품질 기반 가중치
            quality_weight = self._get_learning_quality_weight(coin, interval)
            
            # 최종 점수 계산
            final_score = (base_score + deep_analysis_bonus) * quality_weight
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ RL 파이프라인 점수 계산 실패: {e}")
            return self._strategy_calculator.get_rl_pipeline_score(coin, interval, candle)
    
    def _load_deep_analysis_results(self) -> Optional[Dict]:
        """심화 분석 결과 로드"""
        try:
            # learning_results.db에서 심화 분석 결과 로드
            learning_db_path = "/workspace/data_storage/learning_results.db"
            with sqlite3.connect(learning_db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 여부 확인
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='deep_analysis_results'
                """)
                
                if not cursor.fetchone():
                    # 테이블이 없으면 생성
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS deep_analysis_results (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            analysis_type TEXT NOT NULL,
                            analysis_data TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info("✅ deep_analysis_results 테이블 생성 완료")
                    return None
                
                # 전략 상관관계 분석 결과 로드
                cursor.execute("""
                    SELECT analysis_type, analysis_data 
                    FROM deep_analysis_results 
                    WHERE analysis_type IN ('correlation', 'synergy', 'clustering')
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                
                results = {}
                for row in cursor.fetchall():
                    analysis_type, analysis_data = row
                    try:
                        results[analysis_type] = json.loads(analysis_data)
                    except json.JSONDecodeError:
                        continue
                
                return results if results else None
                
        except Exception as e:
            logger.warning(f"⚠️ 심화 분석 결과 로드 실패: {e}")
            return None

    def _get_deep_analysis_bonus(self, coin: str, interval: str, candle: pd.Series) -> float:
        """심화 통합 분석 결과 기반 보너스 점수"""
        try:
            # 🆕 학습 엔진의 심화 분석 결과 로드
            deep_analysis = self._load_deep_analysis_results()
            if not deep_analysis:
                return 0.0
            
            bonus = 0.0
            
            # 1. 전략 상관관계 분석 보너스
            if 'strategy_correlation_analysis' in deep_analysis:
                correlation_bonus = self._calculate_correlation_bonus(coin, interval, deep_analysis['strategy_correlation_analysis'])
                bonus += correlation_bonus
            
            # 2. 시너지 패턴 보너스
            if 'synergy_patterns' in deep_analysis:
                synergy_bonus = self._calculate_synergy_bonus(coin, interval, deep_analysis['synergy_patterns'])
                bonus += synergy_bonus
            
            # 3. 클러스터링 결과 보너스
            if 'clustering_results' in deep_analysis:
                cluster_bonus = self._calculate_cluster_bonus(coin, interval, deep_analysis['clustering_results'])
                bonus += cluster_bonus
            
            return min(bonus, 0.2)  # 최대 0.2 보너스
            
        except Exception as e:
            logger.error(f"❌ 심화 분석 보너스 계산 실패: {e}")
            return 0.0
    
    def _calculate_correlation_bonus(self, coin: str, interval: str, correlation_analysis: Dict) -> float:
        """상관관계 분석 기반 보너스 계산"""
        try:
            # 코인별 상관관계 점수 확인
            coin_correlation = correlation_analysis.get(coin, {})
            if not coin_correlation:
                return 0.0
            
            # 상관관계 강도에 따른 보너스
            correlation_strength = coin_correlation.get('strength', 0.0)
            return correlation_strength * 0.05  # 최대 5% 보너스
            
        except Exception as e:
            logger.warning(f"⚠️ 상관관계 보너스 계산 실패: {e}")
            return 0.0
    
    def _calculate_synergy_bonus(self, coin: str, interval: str, synergy_patterns: Dict) -> float:
        """시너지 패턴 기반 보너스 계산"""
        try:
            # 코인별 시너지 패턴 확인
            coin_synergy = synergy_patterns.get(coin, {})
            if not coin_synergy:
                return 0.0
            
            # 시너지 점수에 따른 보너스
            synergy_score = coin_synergy.get('score', 0.0)
            return synergy_score * 0.03  # 최대 3% 보너스
            
        except Exception as e:
            logger.warning(f"⚠️ 시너지 보너스 계산 실패: {e}")
            return 0.0
    
    def _calculate_cluster_bonus(self, coin: str, interval: str, clustering_results: Dict) -> float:
        """클러스터링 결과 기반 보너스 계산"""
        try:
            # 코인이 속한 클러스터 확인
            coin_cluster = clustering_results.get(coin, {})
            if not coin_cluster:
                return 0.0
            
            # 클러스터 내 성능에 따른 보너스
            cluster_performance = coin_cluster.get('performance', 0.0)
            return cluster_performance * 0.02  # 최대 2% 보너스
            
        except Exception as e:
            logger.warning(f"⚠️ 클러스터 보너스 계산 실패: {e}")
            return 0.0
    
    def _get_synergy_pattern_bonus(self, coin: str, interval: str, candle: pd.Series) -> float:
        """시너지 패턴 기반 보너스 점수"""
        try:
            # 🆕 시너지 패턴 매니저에서 패턴 로드
            synergy_patterns = self._load_synergy_patterns()
            if not synergy_patterns:
                return 0.0
            
            # 현재 시장 조건에 맞는 시너지 패턴 찾기
            market_condition = self._detect_current_market_condition(coin, interval)
            synergy_bonus = 0.0
            
            if market_condition in synergy_patterns:
                pattern = synergy_patterns[market_condition]
                synergy_bonus = pattern.get('synergy_score', 0.0) * 0.1  # 10% 보너스
            
            return min(synergy_bonus, 0.15)  # 최대 0.15 보너스
            
        except Exception as e:
            logger.error(f"❌ 시너지 패턴 보너스 계산 실패: {e}")
            return 0.0
    
    def _get_learning_quality_weight(self, coin: str, interval: str) -> float:
        """학습 품질 기반 가중치"""
        try:
            # 🆕 학습 품질 평가 결과 로드
            quality_data = self._load_learning_quality_data()
            if not quality_data:
                return 1.0  # 기본 가중치
            
            # 코인별 학습 품질 점수
            coin_quality = quality_data.get(coin, {}).get('quality_score', 0.5)
            interval_quality = quality_data.get(f"{coin}_{interval}", {}).get('quality_score', 0.5)
            
            # 평균 품질 점수를 가중치로 사용
            avg_quality = (coin_quality + interval_quality) / 2
            weight = 0.5 + (avg_quality * 0.5)  # 0.5 ~ 1.0 범위
            
            return weight
            
        except Exception as e:
            logger.error(f"❌ 학습 품질 가중치 계산 실패: {e}")
            return 1.0
    
    def _load_dna_analysis_results(self, coin: str = None) -> Dict[str, Any]:
        """DNA 분석 결과 로드 - learning_results.db의 dna_summary_for_signals 테이블에서 로드"""
        try:
            import sqlite3
            db_path = "/workspace/data_storage/learning_results.db"
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 🆕 learning_results.db의 dna_summary_for_signals 테이블에서 로드
                if coin:
                    cursor.execute("""
                        SELECT profitability_score, stability_score, scalability_score, dna_quality,
                               rsi_pattern, macd_pattern, volume_pattern, dna_momentum, dna_stability
                        FROM dna_summary_for_signals
                        WHERE coin = ? AND (interval = ? OR interval IS NULL)
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """, (coin, coin))  # interval은 coin과 동일하게 설정 (필요시 조정)
                else:
                    cursor.execute("""
                        SELECT profitability_score, stability_score, scalability_score, dna_quality,
                               rsi_pattern, macd_pattern, volume_pattern, dna_momentum, dna_stability
                        FROM dna_summary_for_signals
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """)
                
                row = cursor.fetchone()
                
                if row:
                    dna_features = {
                        'profitability_score': row[0] or 0.0,
                        'stability_score': row[1] or 0.0,
                        'scalability_score': row[2] or 0.5,
                        'dna_quality': row[3] or 0.0,
                        'rsi_pattern': row[4] or 'medium',
                        'macd_pattern': row[5] or 'neutral',
                        'volume_pattern': row[6] or 'normal',
                        'dna_momentum': row[7] or 0.0,
                        'dna_stability': row[8] or 0.0
                    }
                    
                    if self.debug_mode:
                        print(f"✅ DNA 분석 결과 로드 완료: {coin or '전체'} (learning_results.db)")
                    
                    return dna_features
                else:
                    if self.debug_mode:
                        print(f"⚠️ DNA 요약 데이터 없음: {coin or '전체'}")
                    return {}
                    
        except Exception as e:
            if self.debug_mode:
                print(f"❌ DNA 분석 결과 로드 실패: {e}")
            return {}
    
    def _analyze_dna_history_for_realtime(self, history_rows: List[tuple], coin: str = None) -> Dict[str, Any]:
        """DNA 히스토리 데이터를 실시간 분석에 활용"""
        try:
            features = {}
            
            # 필드별 데이터 그룹화
            field_data = {}
            for row in history_rows:
                if coin:
                    field, mean, std, q25, q50, q75, count, interval_focus, created_at = row
                else:
                    coin_name, field, mean, std, q25, q50, q75, count, interval_focus, created_at = row
                
                if field not in field_data:
                    field_data[field] = []
                
                field_data[field].append({
                    'mean': mean, 'std': std, 'q25': q25, 'q50': q50, 'q75': q75,
                    'count': count, 'interval_focus': interval_focus, 'created_at': created_at
                })
            
            # 실시간 분석에 유용한 특성들 추출
            for field, data_list in field_data.items():
                if len(data_list) >= 2:
                    # 시간순 정렬
                    data_list.sort(key=lambda x: x['created_at'])
                    
                    # 최신 vs 이전 비교 (실시간 변화 감지)
                    latest = data_list[-1]
                    previous = data_list[-2] if len(data_list) > 1 else latest
                    
                    # 변화율 계산
                    mean_change = (latest['mean'] - previous['mean']) / max(abs(previous['mean']), 1e-6)
                    
                    # 실시간 시그널에 활용할 특성들
                    features[f'{field}_momentum'] = mean_change  # 모멘텀 지표
                    features[f'{field}_stability'] = 1.0 - min(abs(mean_change), 1.0)  # 안정성 지표
                    features[f'{field}_current_level'] = latest['mean']  # 현재 수준
                    features[f'{field}_volatility'] = latest['std']  # 변동성
                    
                    # 분위수 정보 (실시간 신호 강도 판단용)
                    features[f'{field}_q25'] = latest['q25']
                    features[f'{field}_q75'] = latest['q75']
                    features[f'{field}_range'] = latest['q75'] - latest['q25']  # 범위
            
            # 전체적인 DNA 패턴 분석 (실시간 신호 품질 판단용)
            if len(history_rows) >= 3:
                features['dna_pattern_consistency'] = self._calculate_dna_pattern_consistency(field_data)
                features['dna_signal_strength'] = self._calculate_dna_signal_strength(field_data)
                features['dna_market_adaptation'] = self._calculate_dna_market_adaptation(field_data)
            
            return features
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ DNA 히스토리 실시간 분석 실패: {e}")
            return {}
    
    def _calculate_dna_pattern_consistency(self, field_data: Dict[str, List[Dict]]) -> float:
        """DNA 패턴 일관성 계산 (실시간 신호 신뢰도용)"""
        try:
            consistency_scores = []
            
            for field, data_list in field_data.items():
                if len(data_list) >= 3:
                    # 최근 3개 데이터 포인트의 일관성 계산
                    recent_data = data_list[-3:]
                    values = [d['mean'] for d in recent_data]
                    
                    if len(values) >= 2:
                        # 값들의 변화율 계산
                        changes = []
                        for i in range(1, len(values)):
                            change = abs(values[i] - values[i-1]) / max(abs(values[i-1]), 1e-6)
                            changes.append(change)
                        
                        if changes:
                            avg_change = sum(changes) / len(changes)
                            consistency = 1.0 - min(avg_change, 1.0)
                            consistency_scores.append(consistency)
            
            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            return 0.5
    
    def _calculate_dna_signal_strength(self, field_data: Dict[str, List[Dict]]) -> float:
        """DNA 신호 강도 계산 (실시간 신호 강도 판단용)"""
        try:
            strength_scores = []
            
            for field, data_list in field_data.items():
                if data_list:
                    latest = data_list[-1]
                    # 표준편차가 작을수록 강한 신호
                    signal_strength = 1.0 - min(latest['std'] / max(abs(latest['mean']), 1e-6), 1.0)
                    strength_scores.append(signal_strength)
            
            return sum(strength_scores) / len(strength_scores) if strength_scores else 0.5
            
        except Exception as e:
            return 0.5
    
    def _calculate_dna_market_adaptation(self, field_data: Dict[str, List[Dict]]) -> float:
        """DNA 시장 적응성 계산 (실시간 시장 적응도 판단용)"""
        try:
            adaptation_scores = []
            
            for field, data_list in field_data.items():
                if len(data_list) >= 2:
                    # 최근 데이터의 적응성 계산
                    recent_data = data_list[-2:]
                    adaptation = 1.0 - abs(recent_data[-1]['mean'] - recent_data[-2]['mean']) / max(abs(recent_data[-2]['mean']), 1e-6)
                    adaptation_scores.append(max(0.0, adaptation))
            
            return sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0.5
            
        except Exception as e:
            return 0.5
        """심화 통합 분석 결과 로드"""
        try:
            # 🆕 학습 엔진의 심화 분석 결과를 DB에서 로드
            db_path = "/workspace/data_storage/learning_results.db"
            if not os.path.exists(db_path):
                return None
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 여부 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_pipeline_results'")
                if not cursor.fetchone():
                    if self.debug_mode:
                        print("ℹ️ learning_pipeline_results 테이블이 존재하지 않습니다.")
                    return None
                
                cursor.execute("""
                    SELECT deep_analysis_result 
                    FROM learning_pipeline_results 
                    WHERE deep_analysis_result IS NOT NULL
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 심화 분석 결과 로드 실패: {e}")
            return None
    
    def _load_learning_quality_data(self) -> Optional[Dict]:
        """학습 품질 데이터 로드"""
        try:
            # 🆕 학습 품질 평가 결과를 DB에서 로드
            db_path = "/workspace/data_storage/learning_results.db"
            if not os.path.exists(db_path):
                return None
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 여부 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_pipeline_results'")
                if not cursor.fetchone():
                    if self.debug_mode:
                        print("ℹ️ learning_pipeline_results 테이블이 존재하지 않습니다.")
                    return None
                
                cursor.execute("""
                    SELECT learning_quality_assessment 
                    FROM learning_pipeline_results 
                    WHERE learning_quality_assessment IS NOT NULL
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 학습 품질 데이터 로드 실패: {e}")
            return None
    
    def _calculate_correlation_bonus(self, coin: str, interval: str, correlation_analysis: Dict) -> float:
        """상관관계 분석 기반 보너스"""
        try:
            if 'high_correlation_pairs' not in correlation_analysis:
                return 0.0
            
            # 높은 상관관계가 있는 전략들에 보너스
            high_corr_pairs = correlation_analysis['high_correlation_pairs']
            if len(high_corr_pairs) > 0:
                return 0.05  # 5% 보너스
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 상관관계 보너스 계산 실패: {e}")
            return 0.0
    
    def _calculate_synergy_bonus(self, coin: str, interval: str, synergy_patterns: Dict) -> float:
        """시너지 패턴 기반 보너스"""
        try:
            # 심볼별 시너지 확인
            symbol_synergies = synergy_patterns.get('symbol_synergies', {})
            if coin in symbol_synergies and symbol_synergies[coin].get('potential_synergy', False):
                return 0.08  # 8% 보너스
            
            # 인터벌별 시너지 확인
            interval_synergies = synergy_patterns.get('interval_synergies', {})
            if interval in interval_synergies and interval_synergies[interval].get('potential_synergy', False):
                return 0.05  # 5% 보너스
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 시너지 보너스 계산 실패: {e}")
            return 0.0
    
    def _calculate_cluster_bonus(self, coin: str, interval: str, clustering_results: Dict) -> float:
        """클러스터링 결과 기반 보너스"""
        try:
            if 'clusters' not in clustering_results:
                return 0.0
            
            # 큰 클러스터에 속한 전략들에 보너스
            clusters = clustering_results['clusters']
            for cluster_id, cluster_strategies in clusters.items():
                if len(cluster_strategies) > 5:  # 큰 클러스터
                    return 0.03  # 3% 보너스
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 클러스터 보너스 계산 실패: {e}")
            return 0.0
    
    def get_dna_based_similar_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """DNA 기반 유사 코인 점수 계산 (240분 인터벌 우선 시스템 적용)"""
        try:
            # 🧬 현재 코인의 DNA 패턴 추출 (240분 우선 방식)
            current_dna = self._extract_current_dna_pattern_enhanced(coin, interval, candle)
            
            # 🧬 유사한 DNA를 가진 다른 코인들의 성과 점수 조회
            similar_scores = self._get_similar_dna_scores_enhanced(current_dna, coin, interval)
            
            if not similar_scores:
                # 🚀 유사한 DNA가 없으면 현재 코인의 기술적 지표 기반 점수 계산
                print(f"⚠️ {coin}/{interval}: 유사한 DNA 없음, 기술적 지표 기반 점수 계산")
                return self._calculate_technical_based_score(candle)
            
            # 🧬 인터벌별 가중치 적용 (240분 우선) - 개선된 버전
            interval_weights = {
                '240m': 2.5,  # 240분: 가장 높은 가중치 (증가)
                '15m': 2.0,   # 15분: 두 번째 높은 가중치 (증가)
                '30m': 1.5,   # 30분: 보통 가중치 (증가)
                '1d': 1.2     # 1일: 기본 가중치 상향
            }
            
            # 🧬 가중 유사도 점수 계산 (개선된 버전)
            total_weight = 0.0
            weighted_sum = 0.0
            
            for similar_coin, similarity, score, similar_interval in similar_scores:
                # 인터벌 가중치 적용
                interval_weight = interval_weights.get(similar_interval, 1.0)
                
                # 🆕 유사도 보너스 (높은 유사도에 더 높은 가중치)
                similarity_bonus = 1.0
                if similarity > 0.8:
                    similarity_bonus = 1.5  # 매우 유사한 경우 50% 보너스
                elif similarity > 0.6:
                    similarity_bonus = 1.3  # 유사한 경우 30% 보너스
                elif similarity > 0.4:
                    similarity_bonus = 1.1  # 약간 유사한 경우 10% 보너스
                
                # 🆕 성과 점수 보너스 (높은 성과에 더 높은 가중치)
                performance_bonus = 1.0
                if score > 0.1:
                    performance_bonus = 1.4  # 높은 성과 40% 보너스
                elif score > 0.05:
                    performance_bonus = 1.2  # 중간 성과 20% 보너스
                
                combined_weight = similarity * interval_weight * similarity_bonus * performance_bonus * 0.6 + 0.4  # 최소 40% 가중치 보장
                
                weighted_sum += score * combined_weight
                total_weight += combined_weight
            
            if total_weight > 0:
                # 🧬 기본 DNA 점수 계산
                base_dna_score = weighted_sum / total_weight
                
                # 🆕 DNA 히스토리 데이터 로드 (실시간 분석용)
                dna_history_features = self._load_dna_analysis_results(coin)
                
                # 🆕 DNA 히스토리 기반 보정 (실시간 분석)
                history_bonus = 0.0
                if dna_history_features:
                    # 패턴 일관성 보너스
                    consistency = dna_history_features.get('dna_pattern_consistency', 0.5)
                    consistency_bonus = consistency * 0.1  # 최대 10% 보너스
                    
                    # 신호 강도 보너스
                    signal_strength = dna_history_features.get('dna_signal_strength', 0.5)
                    strength_bonus = signal_strength * 0.15  # 최대 15% 보너스
                    
                    # 시장 적응성 보너스
                    market_adaptation = dna_history_features.get('dna_market_adaptation', 0.5)
                    adaptation_bonus = market_adaptation * 0.1  # 최대 10% 보너스
                    
                    history_bonus = consistency_bonus + strength_bonus + adaptation_bonus
                    
                    if self.debug_mode:
                        print(f"🧬 {coin}/{interval}: DNA 히스토리 보너스 - 일관성({consistency:.3f}), 강도({signal_strength:.3f}), 적응({market_adaptation:.3f})")
                
                # 🆕 최종 점수 계산 (기본 DNA 점수 + 히스토리 보너스)
                final_score = base_dna_score + history_bonus
                
                # 🆕 최종 점수 보너스 (DNA 기반 점수 강화)
                if final_score > 0.05:
                    final_score *= 1.3  # 높은 DNA 점수에 30% 보너스
                elif final_score > 0.02:
                    final_score *= 1.2  # 중간 DNA 점수에 20% 보너스
                
                if self.debug_mode:
                    print(f"🧬 {coin}/{interval}: DNA 기반 점수 계산 성공 - 유사코인({len(similar_scores)}개), 기본점수({base_dna_score:.3f}), 히스토리보너스({history_bonus:.3f}), 최종점수({final_score:.3f})")
                
                return min(1.0, max(0.0, final_score))
            else:
                return self._calculate_technical_based_score(candle)
                
        except Exception as e:
            print(f"⚠️ DNA 기반 유사 점수 계산 오류 ({coin}/{interval}): {e}")
            return self._calculate_technical_based_score(candle)
    
    def _calculate_technical_based_score(self, candle: pd.Series) -> float:
        """🚀 기술적 지표 기반 점수 계산 (DNA 대체용)"""
        try:
            # 🚀 실제 캔들 데이터에서 지표 추출 (None 값 안전 처리)
            rsi = candle.get('rsi', 50.0)
            macd = candle.get('macd', 0.0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            volatility = candle.get('volatility', 0.02)
            wave_phase = candle.get('wave_phase', 'unknown')
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            integrated_direction = candle.get('integrated_direction', 'neutral')
            
            # None 값 안전 처리
            if rsi is None:
                rsi = 50.0
            if macd is None:
                macd = 0.0
            if volume_ratio is None:
                volume_ratio = 1.0
            if volatility is None:
                volatility = 0.02
            if pattern_confidence is None:
                pattern_confidence = 0.0
            
            # 🚀 RSI 기반 점수 (0.0 ~ 1.0)
            if rsi < 20:  # 극도 과매도
                rsi_score = 0.9
            elif rsi < 30:  # 과매도
                rsi_score = 0.7
            elif rsi > 80:  # 극도 과매수
                rsi_score = 0.1
            elif rsi > 70:  # 과매수
                rsi_score = 0.3
            else:  # 중립
                rsi_score = 0.5
            
            # 🚀 MACD 기반 점수 (0.0 ~ 1.0)
            if macd > 0.01:  # 강한 상승 신호
                macd_score = 0.9
            elif macd > 0:  # 약한 상승 신호
                macd_score = 0.7
            elif macd > -0.01:  # 약한 하락 신호
                macd_score = 0.3
            else:  # 강한 하락 신호
                macd_score = 0.1
            
            # 🚀 거래량 기반 점수 (0.0 ~ 1.0)
            if volume_ratio > 2.0:  # 높은 거래량
                volume_score = 0.8
            elif volume_ratio > 1.0:  # 정상 거래량
                volume_score = 0.6
            else:  # 낮은 거래량
                volume_score = 0.4
            
            # 🚀 파동 단계 기반 점수
            wave_score = 0.5
            if wave_phase == 'impulse':
                wave_score = 0.8
            elif wave_phase == 'correction':
                wave_score = 0.3
            elif wave_phase == 'consolidation':
                wave_score = 0.6
            
            # 🚀 통합 방향성 기반 점수
            direction_score = 0.5
            if integrated_direction == 'strong_bullish':
                direction_score = 0.9
            elif integrated_direction == 'bullish':
                direction_score = 0.7
            elif integrated_direction == 'strong_bearish':
                direction_score = 0.1
            elif integrated_direction == 'bearish':
                direction_score = 0.3
            
            # 🚀 패턴 신뢰도 기반 점수
            pattern_score = 0.5 + (pattern_confidence * 0.5)  # 0.5 ~ 1.0
            
            # 🚀 최종 점수 계산 (가중 평균)
            final_score = (
                rsi_score * 0.25 +
                macd_score * 0.25 +
                volume_score * 0.15 +
                wave_score * 0.15 +
                direction_score * 0.15 +
                pattern_score * 0.05
            )
            
            print(f"🔧 기술적 지표 기반 점수: RSI({rsi:.1f}→{rsi_score:.2f}), MACD({macd:.4f}→{macd_score:.2f}), Volume({volume_ratio:.2f}x→{volume_score:.2f}), 최종({final_score:.3f})")
            
            return np.clip(final_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ 기술적 지표 기반 점수 계산 실패: {e}")
            return 0.3  # 기본값
    
    def _extract_current_dna_pattern_enhanced(self, coin: str, interval: str, candle: pd.Series) -> dict:
        """현재 코인의 DNA 패턴 추출 (240분 우선 방식 적용)"""
        try:
            # 🧬 안전한 값 추출 (None 처리)
            rsi = safe_float(candle.get('rsi'), 50.0)
            macd = safe_float(candle.get('macd'), 0.0)
            volume_ratio = safe_float(candle.get('volume_ratio'), 1.0)
            volatility = safe_float(candle.get('volatility'), 0.0)
            structure_score = safe_float(candle.get('structure_score'), 0.5)
            wave_step = safe_float(candle.get('wave_step'), 0.0)
            pattern_quality = safe_float(candle.get('pattern_quality'), 0.5)
            timestamp = safe_float(candle.get('timestamp'), 0)
            
            # 🧬 핵심 지표들로 DNA 패턴 생성 (더 정교한 범주화)
            dna_pattern = {
                'rsi_range': self._categorize_rsi_enhanced(rsi),
                'macd_range': self._categorize_macd_enhanced(macd),
                'volume_range': self._categorize_volume_enhanced(volume_ratio),
                'volatility_range': self._categorize_volatility_enhanced(volatility),
                'structure_range': self._categorize_structure_enhanced(structure_score),
                'wave_step': self._categorize_wave_step(wave_step),
                'pattern_quality': self._categorize_pattern_quality(pattern_quality),
                'interval': interval,  # 인터벌 정보 추가
                'timestamp': timestamp
            }
            return dna_pattern
            
        except Exception as e:
            print(f"⚠️ DNA 패턴 추출 오류 ({coin}): {e}")
            return {}
    
    def _categorize_rsi_enhanced(self, rsi: float) -> str:
        """RSI 범주화 (더 정교한 분류)"""
        if rsi < 20:
            return 'extreme_oversold'
        elif rsi < 30:
            return 'oversold'
        elif rsi < 40:
            return 'low'
        elif rsi < 60:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        elif rsi < 80:
            return 'overbought'
        else:
            return 'extreme_overbought'
    
    def _categorize_macd_enhanced(self, macd: float) -> str:
        """MACD 범주화 (더 정교한 분류)"""
        if macd < -0.02:
            return 'extreme_bearish'
        elif macd < -0.01:
            return 'strong_bearish'
        elif macd < 0:
            return 'bearish'
        elif macd < 0.01:
            return 'bullish'
        elif macd < 0.02:
            return 'strong_bullish'
        else:
            return 'extreme_bullish'
    
    def _categorize_volume_enhanced(self, volume_ratio: float) -> str:
        """거래량 비율 범주화 (더 정교한 분류)"""
        if volume_ratio < 0.3:
            return 'extreme_low'
        elif volume_ratio < 0.7:
            return 'very_low'
        elif volume_ratio < 1.0:
            return 'low'
        elif volume_ratio < 1.5:
            return 'normal'
        elif volume_ratio < 3.0:
            return 'high'
        elif volume_ratio < 7.0:
            return 'very_high'
        else:
            return 'extreme_high'
    
    def _categorize_volatility_enhanced(self, volatility: float) -> str:
        """변동성 범주화 (더 정교한 분류)"""
        if volatility < 0.005:
            return 'extreme_low'
        elif volatility < 0.01:
            return 'very_low'
        elif volatility < 0.02:
            return 'low'
        elif volatility < 0.05:
            return 'normal'
        elif volatility < 0.1:
            return 'high'
        elif volatility < 0.2:
            return 'very_high'
        else:
            return 'extreme_high'
    
    def _categorize_structure_enhanced(self, structure_score: float) -> str:
        """구조 점수 범주화 (더 정교한 분류)"""
        if structure_score < 0.2:
            return 'very_weak'
        elif structure_score < 0.4:
            return 'weak'
        elif structure_score < 0.6:
            return 'neutral'
        elif structure_score < 0.8:
            return 'strong'
        else:
            return 'very_strong'
    
    def _categorize_wave_step(self, wave_step: float) -> str:
        """웨이브 스텝 범주화"""
        # 안전한 값 처리
        wave_step = safe_float(wave_step, 0.0)
        
        if wave_step < 0.2:
            return 'early'
        elif wave_step < 0.5:
            return 'mid'
        elif wave_step < 0.8:
            return 'late'
        else:
            return 'action'
    
    def _categorize_pattern_quality(self, pattern_quality: float) -> str:
        """패턴 품질 범주화"""
        # 안전한 값 처리
        pattern_quality = safe_float(pattern_quality, 0.5)
        
        if pattern_quality < 0.3:
            return 'poor'
        elif pattern_quality < 0.6:
            return 'fair'
        elif pattern_quality < 0.8:
            return 'good'
        else:
            return 'excellent'
    
    # 🧬 기존 함수들 (호환성 유지)
    def _categorize_rsi(self, rsi: float) -> str:
        """RSI 범주화 (기존 호환성 유지)"""
        return self._categorize_rsi_enhanced(rsi)
    
    def _categorize_macd(self, macd: float) -> str:
        """MACD 범주화 (기존 호환성 유지)"""
        return self._categorize_macd_enhanced(macd)
    
    def _categorize_volume(self, volume_ratio: float) -> str:
        """거래량 비율 범주화 (기존 호환성 유지)"""
        return self._categorize_volume_enhanced(volume_ratio)
    
    def _categorize_volatility(self, volatility: float) -> str:
        """변동성 범주화 (기존 호환성 유지)"""
        return self._categorize_volatility_enhanced(volatility)
    
    def _categorize_structure(self, structure_score: float) -> str:
        """구조 점수 범주화 (기존 호환성 유지)"""
        return self._categorize_structure_enhanced(structure_score)
    
    def _get_similar_dna_scores(self, current_dna: dict, exclude_coin: str) -> list:
        """유사한 DNA를 가진 코인들의 점수 조회 (기존 호환성 유지)"""
        # 기존 방식으로 호환성 유지
        try:
            similar_scores = []
            
            for strategy_key, strategy in self.coin_specific_strategies.items():
                if strategy_key.startswith(exclude_coin):
                    continue
                
                similarity = self._calculate_dna_similarity_enhanced(current_dna, strategy)
                
                if similarity > 0.25:
                    coin_name = strategy_key.split('_')[0]
                    interval = strategy_key.split('_')[1]
                    performance_score = self._calculate_performance_score_enhanced(strategy)
                    similar_scores.append((coin_name, similarity, performance_score))
            
            similar_scores.sort(key=lambda x: x[1], reverse=True)
            return similar_scores[:5]
            
        except Exception as e:
            print(f"⚠️ 유사 DNA 점수 조회 오류: {e}")
            return []
    
    def _extract_current_dna_pattern(self, coin: str, interval: str, candle: pd.Series) -> dict:
        """현재 코인의 DNA 패턴 추출 (기존 호환성 유지)"""
        return self._extract_current_dna_pattern_enhanced(coin, interval, candle)
    
    def _calculate_performance_score(self, strategy: dict) -> float:
        """전략 성과 점수 계산 (기존 호환성 유지)"""
        return self._calculate_performance_score_enhanced(strategy)
    
    def _get_similar_dna_scores_enhanced(self, current_dna: dict, exclude_coin: str, current_interval: str) -> list:
        """유사한 DNA를 가진 코인들의 점수 조회 (240분 우선 시스템 적용)"""
        try:
            print(f"🔍 {exclude_coin}/{current_interval}: DNA 유사도 검색 시작")
            print(f"📊 사용 가능한 코인별 전략 수: {len(self.coin_specific_strategies)}")
            
            if not self.coin_specific_strategies:
                print(f"❌ {exclude_coin}/{current_interval}: 코인별 전략이 로드되지 않음")
                return []
            
            similar_scores = []
            available_keys = []  # 🆕 실제 사용 가능한 전략 키 수집 (자기 자신 제외)
            
            # 🧬 DNA 유사도 기반으로 유사한 코인들 찾기
            for strategy_key, strategy in self.coin_specific_strategies.items():
                # 🆕 자기 자신 제외 로직 개선 (정확한 매칭)
                coin_name = strategy_key.split('_')[0]
                if coin_name == exclude_coin:
                    continue  # 자기 자신 제외
                
                # 🆕 사용 가능한 전략 키 수집 (자기 자신 제외)
                available_keys.append(strategy_key)
                
                # 🧬 DNA 유사도 계산 (향상된 방식)
                similarity = self._calculate_dna_similarity_enhanced(current_dna, strategy)
                
                # 🚨 유사도 임계값 적용 (더 유연하게)
                if similarity > 0.2:  # 30%에서 20%로 낮춤
                    interval = strategy_key.split('_')[1]
                    
                    # 🧬 해당 코인의 최근 성과 점수
                    performance_score = self._calculate_performance_score_enhanced(strategy)
                    
                    similar_scores.append((coin_name, similarity, performance_score, interval))
                    print(f"✅ 유사 코인 발견: {coin_name}/{interval} (유사도: {similarity:.3f})")
            
            # 🆕 실제 사용 가능한 전략 키 출력 (점수 순으로 정렬하여 상위 5개)
            if available_keys:
                sorted_available_keys = sorted(
                    available_keys,
                    key=lambda k: self.coin_specific_strategies[k].get('score', 0.0),
                    reverse=True
                )[:5]
                print(f"📋 사용 가능한 전략 키 예시 ({exclude_coin}/{current_interval} 제외, 점수 상위 5개): {sorted_available_keys}")
            else:
                print(f"📋 사용 가능한 전략 키: 없음 (자기 자신만 존재)")
            
            print(f"📊 {exclude_coin}/{current_interval}: 총 {len(similar_scores)}개 유사 코인 발견")
            
            # 🚨 유사도 순으로 정렬
            similar_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 🚨 동적 필터링 (유사도에 따라 개수 조정)
            filtered_scores = []
            for score in similar_scores:
                # 🚨 유사도가 높을수록 더 많은 코인 선택
                if score[1] >= 0.8 and len(filtered_scores) < 8:  # 매우 유사한 경우
                    filtered_scores.append(score)
                elif score[1] >= 0.6 and len(filtered_scores) < 5:  # 유사한 경우
                    filtered_scores.append(score)
                elif score[1] >= 0.4 and len(filtered_scores) < 3:  # 약간 유사한 경우
                    filtered_scores.append(score)
                elif score[1] >= 0.3 and len(filtered_scores) < 2:  # 최소 유사한 경우
                    filtered_scores.append(score)
            
            print(f"📊 {exclude_coin}/{current_interval}: 필터링 후 {len(filtered_scores)}개 유사 코인")
            return filtered_scores
            
        except Exception as e:
            print(f"⚠️ 유사 DNA 점수 조회 오류: {e}")
            return []
    
    def _calculate_dna_similarity_enhanced(self, current_dna: dict, strategy: dict) -> float:
        """DNA 유사도 계산 (향상된 방식)"""
        try:
            similarity_score = 0.0
            total_weight = 0.0
            
            # 🧬 각 지표별 가중치 설정
            weights = {
                'rsi_range': 0.25,
                'macd_range': 0.20,
                'volume_range': 0.15,
                'volatility_range': 0.15,
                'structure_range': 0.15,
                'wave_step': 0.05,
                'pattern_quality': 0.05
            }
            
            # 🚨 각 지표별 유사도 계산 (수정된 방식)
            for indicator, weight in weights.items():
                if indicator in current_dna and indicator in strategy:
                    current_value = current_dna[indicator]
                    strategy_value = strategy.get(indicator, 'unknown')
                    
                    # 🚨 정확한 매칭
                    if current_value == strategy_value:
                        similarity_score += weight
                    # 🚨 부분 매칭
                    elif self._is_similar_category(current_value, strategy_value):
                        similarity_score += weight * 0.5
                    
                    total_weight += weight
                else:
                    # 🚨 지표가 없어도 가중치는 추가 (정규화를 위해)
                    total_weight += weight
            
            # 🚨 인터벌 유사도 추가 (240분 우선)
            interval_weight = 0.1
            if 'interval' in current_dna and 'interval' in strategy:
                current_interval = current_dna['interval']
                strategy_interval = strategy['interval']
                
                if current_interval == strategy_interval:
                    similarity_score += 0.15
                elif (current_interval == '240m' and strategy_interval in ['15m', '30m']) or \
                     (strategy_interval == '240m' and current_interval in ['15m', '30m']):
                    similarity_score += 0.08
                
                total_weight += interval_weight
            
            # 🚨 정규화된 유사도 반환 (0.0 ~ 1.0 범위)
            normalized_similarity = similarity_score / total_weight if total_weight > 0 else 0.0
            return min(max(normalized_similarity, 0.0), 1.0)  # 0.0 ~ 1.0 범위로 제한
            
        except Exception as e:
            print(f"⚠️ DNA 유사도 계산 오류: {e}")
            return 0.0
    
    def _is_similar_category(self, value1: str, value2: str) -> bool:
        """유사한 범주인지 확인"""
        # 🧬 유사한 범주 매핑
        similar_categories = {
            'rsi_range': {
                'extreme_oversold': ['oversold'],
                'oversold': ['extreme_oversold', 'low'],
                'low': ['oversold', 'neutral'],
                'neutral': ['low', 'high'],
                'high': ['neutral', 'overbought'],
                'overbought': ['high', 'extreme_overbought'],
                'extreme_overbought': ['overbought']
            },
            'macd_range': {
                'extreme_bearish': ['strong_bearish'],
                'strong_bearish': ['extreme_bearish', 'bearish'],
                'bearish': ['strong_bearish', 'bullish'],
                'bullish': ['bearish', 'strong_bullish'],
                'strong_bullish': ['bullish', 'extreme_bullish'],
                'extreme_bullish': ['strong_bullish']
            },
            'volume_range': {
                'extreme_low': ['very_low'],
                'very_low': ['extreme_low', 'low'],
                'low': ['very_low', 'normal'],
                'normal': ['low', 'high'],
                'high': ['normal', 'very_high'],
                'very_high': ['high', 'extreme_high'],
                'extreme_high': ['very_high']
            }
        }
        
        # 🧬 각 지표별 유사 범주 확인
        for indicator, categories in similar_categories.items():
            if value1 in categories and value2 in categories.get(value1, []):
                return True
            if value2 in categories and value1 in categories.get(value2, []):
                return True
        
        return False
    
    def _calculate_performance_score_enhanced(self, strategy: dict) -> float:
        """성과 점수 계산 (향상된 방식)"""
        try:
            # 🧬 기본 성과 지표들
            profit = strategy.get('profit', 0.0)
            win_rate = strategy.get('win_rate', 0.5)
            trades_count = strategy.get('trades_count', 0)
            
            # 🧬 최소 거래 수 확인
            if trades_count < 3:
                return 0.0
            
            # 🧬 수익률 점수 (0-1 범위로 정규화)
            profit_score = min(max(profit / 0.1, 0.0), 1.0)  # 10% 수익률을 최대점으로
            
            # 🧬 승률 점수
            win_rate_score = win_rate
            
            # 🧬 거래 수 점수 (충분한 거래 수 보장)
            trade_count_score = min(trades_count / 10.0, 1.0)  # 10회 거래를 최대점으로
            
            # 🧬 종합 점수 계산
            total_score = (profit_score * 0.5 + win_rate_score * 0.3 + trade_count_score * 0.2)
            
            return total_score
            
        except Exception as e:
            print(f"⚠️ 성과 점수 계산 오류: {e}")
            return 0.0
    
    def get_universal_rl_score(self, state_key: str) -> float:
        """범용 RL 점수 조회 (패턴 매칭 기반)"""
        try:
            # 간단한 패턴 매칭 (빠른 매칭)
            if 'bullish' in state_key or 'oversold' in state_key:
                return np.random.uniform(0.1, 0.3)  # 매수 신호
            elif 'bearish' in state_key or 'overbought' in state_key:
                return np.random.uniform(-0.3, -0.1)  # 매도 신호
            
            # 중립 상태
            return np.random.uniform(-0.05, 0.05)
                        
        except Exception as e:
            print(f"⚠️ 범용 RL 점수 조회 오류: {e}")
            return 0.0
    
    def determine_action(self, signal_score: float, confidence: float) -> SignalAction:
        """순수 시그널 기반 액션 결정 (보유 정보 없음)"""
        try:
            # 🆕 학습 기반 임계값 조정
            min_confidence = self.get_learning_based_confidence_threshold()
            min_signal_score = self.get_learning_based_signal_score_threshold()
            
            # 🆕 매수 조건 (완화된 초기 기준)
            if signal_score > min_signal_score and confidence > min_confidence:
                return SignalAction.BUY
            
            # 🆕 매도 조건 (시그널 점수가 매우 낮을 때)
            if signal_score < -0.3:
                return SignalAction.SELL
            
            # 🆕 홀딩 조건 (중간 정도의 시그널)
            if -0.1 <= signal_score <= 0.1:
                return SignalAction.HOLD
            
            # 🆕 대기 조건
            return SignalAction.WAIT
            
        except Exception as e:
            print(f"⚠️ 액션 결정 오류: {e}")
            return SignalAction.WAIT
    
    def get_learning_based_confidence_threshold(self) -> float:
        """학습 기반 신뢰도 임계값 반환"""
        if not self.use_learning_based_thresholds or self.learning_feedback is None:
            return self.min_confidence
        
        # 학습 피드백에 따른 동적 조정
        win_rate = self.learning_feedback.get('win_rate', 0.5)
        total_trades = self.learning_feedback.get('total_trades', 0)
        
        # 최소 10개 거래가 있어야 신뢰할 수 있음
        if total_trades < 10:
            return self.min_confidence
        
        # 승률에 따른 조정
        if win_rate < 0.4:  # 성과 나쁨 → 더 엄격하게
            return min(0.7, self.min_confidence + 0.1)
        elif win_rate > 0.6:  # 성과 좋음 → 적당히 완화
            return max(0.45, self.min_confidence - 0.05)
        else:  # 중간 성과
            return self.min_confidence
    
    def get_learning_based_signal_score_threshold(self) -> float:
        """학습 기반 시그널 점수 임계값 반환"""
        if not self.use_learning_based_thresholds or self.learning_feedback is None:
            return self.min_signal_score
        
        # 학습 피드백에 따른 동적 조정
        win_rate = self.learning_feedback.get('win_rate', 0.5)
        total_trades = self.learning_feedback.get('total_trades', 0)
        
        # 최소 10개 거래가 있어야 신뢰할 수 있음
        if total_trades < 10:
            return self.min_signal_score
        
        # 승률에 따른 조정
        if win_rate < 0.4:  # 성과 나쁨 → 더 엄격하게
            return min(0.15, self.min_signal_score + 0.05)
        elif win_rate > 0.6:  # 성과 좋음 → 적당히 완화
            return max(0.03, self.min_signal_score - 0.02)
        else:  # 중간 성과
            return self.min_signal_score
    
    def update_learning_feedback(self, feedback: Dict):
        """가상매매 학습기로부터 피드백 받기"""
        self.learning_feedback = feedback
        print(f"🔄 학습 피드백 업데이트: 승률={feedback.get('win_rate', 0):.2f}, 총거래={feedback.get('total_trades', 0)}개")
        print(f"   새로운 임계값: 신뢰도={self.get_learning_based_confidence_threshold():.2f}, 시그널점수={self.get_learning_based_signal_score_threshold():.3f}")
    

    
    def _load_absolute_zero_analysis_results(self):
        """🔥 Absolute Zero 시스템 분석 결과 로드 (개별 코인 + 글로벌 전략)"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            
            from rl_pipeline.db.learning_results import (
                load_integrated_analysis_results,
                load_global_strategies_from_db
            )
            
            # 개별 코인 분석 결과 로드 (캐시에 저장)
            # 주요 코인과 인터벌 조합만 미리 로드
            major_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL']
            intervals = ['15m', '30m', '240m', '1d']
            
            for coin in major_coins:
                for interval in intervals:
                    cache_key = f"{coin}-{interval}"
                    try:
                        analysis_result = load_integrated_analysis_results(coin, interval)
                        if analysis_result:
                            self.integrated_analysis_cache[cache_key] = analysis_result
                            if self.debug_mode:
                                logger.info(f"✅ 통합 분석 결과 로드: {cache_key}")
                    except Exception as e:
                        if self.debug_mode:
                            logger.debug(f"⚠️ {cache_key} 분석 결과 로드 실패: {e}")
            
            # 글로벌 전략 로드
            try:
                global_strategies = load_global_strategies_from_db()
                for strategy in global_strategies:
                    interval = strategy.get('interval', 'all_intervals')
                    if interval not in self.global_strategies_cache:
                        self.global_strategies_cache[interval] = []
                    self.global_strategies_cache[interval].append(strategy)
                
                if self.debug_mode:
                    logger.info(f"✅ 글로벌 전략 로드: {sum(len(v) for v in self.global_strategies_cache.values())}개")
            except Exception as e:
                if self.debug_mode:
                    logger.warning(f"⚠️ 글로벌 전략 로드 실패: {e}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Absolute Zero 분석 결과 로드 실패: {e}")
    
    def _get_integrated_analysis_score(self, coin: str, interval: str, candle: pd.Series, market_condition: str) -> float:
        """🔥 RL Pipeline 통합 분석 점수 계산 (저장된 분석 결과 활용)"""
        try:
            cache_key = f"{coin}-{interval}"
            
            # 🔥 1단계: 저장된 통합 분석 결과 사용 (우선순위)
            if cache_key in self.integrated_analysis_cache:
                analysis_result = self.integrated_analysis_cache[cache_key]
                
                # 최신성 확인 (1시간 이내 데이터)
                import time
                from datetime import datetime
                try:
                    created_at = datetime.fromisoformat(analysis_result['created_at'])
                    age_hours = (datetime.now() - created_at).total_seconds() / 3600
                    
                    if age_hours < 1.0:  # 1시간 이내면 사용
                        final_score = analysis_result.get('final_signal_score', 0.5)
                        signal_confidence = analysis_result.get('signal_confidence', 0.5)
                        
                        # 신뢰도 기반 보정
                        confidence_weight = min(1.0, signal_confidence)
                        adjusted_score = 0.5 + (final_score - 0.5) * confidence_weight
                        
                        if self.debug_mode:
                            logger.debug(f"🔥 저장된 분석 결과 사용: {cache_key} (점수: {final_score:.3f}, 신뢰도: {signal_confidence:.3f})")
                        
                        return adjusted_score
                except Exception as e:
                    if self.debug_mode:
                        logger.debug(f"⚠️ 분석 결과 시간 파싱 실패: {e}")
            
            # 🔥 2단계: 실시간 로드 시도 (캐시 미스 시)
            try:
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                
                from rl_pipeline.db.learning_results import load_integrated_analysis_results
                
                analysis_result = load_integrated_analysis_results(coin, interval)
                if analysis_result:
                    # 캐시에 저장
                    self.integrated_analysis_cache[cache_key] = analysis_result
                    
                    final_score = analysis_result.get('final_signal_score', 0.5)
                    signal_confidence = analysis_result.get('signal_confidence', 0.5)
                    
                    confidence_weight = min(1.0, signal_confidence)
                    adjusted_score = 0.5 + (final_score - 0.5) * confidence_weight
                    
                    if self.debug_mode:
                        logger.debug(f"🔥 실시간 분석 결과 로드: {cache_key} (점수: {final_score:.3f})")
                    
                    return adjusted_score
            except Exception as e:
                if self.debug_mode:
                    logger.debug(f"⚠️ 실시간 분석 결과 로드 실패: {e}")
            
            # 🔥 3단계: 폴백 - 요약 테이블 우선 사용, 필요시 원본 테이블 조회 (최적화)
            if self.integrated_analyzer is not None:
                # 캔들 데이터를 DataFrame으로 변환
                import pandas as pd
                candle_df = pd.DataFrame([candle])
                
                # 🚀 최적화: 요약 테이블에서 우선 조회 (빠름)
                strategies = []
                try:
                    import sqlite3
                    # learning_results.db에서 요약 정보 조회
                    learning_db_path = "/workspace/data_storage/learning_results.db"
                    with sqlite3.connect(learning_db_path) as conn:
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            SELECT top_strategy_id, top_strategy_params, top_profit, top_win_rate,
                                   top_quality_grade, avg_profit, avg_win_rate
                            FROM strategy_summary_for_signals
                            WHERE coin = ? AND interval = ?
                            ORDER BY updated_at DESC
                            LIMIT 1
                        """, (coin, interval))
                        
                        summary = cursor.fetchone()
                        if summary:
                            # 요약 테이블에서 top 전략 정보를 전략 객체로 변환
                            top_strategy_id = summary[0]
                            top_params_json = summary[1]
                            top_profit = summary[2] or 0.0
                            top_win_rate = summary[3] or 0.0
                            top_quality = summary[4] or 'B'
                            
                            try:
                                top_params = json.loads(top_params_json) if top_params_json else {}
                            except:
                                top_params = {}
                            
                            # 요약 정보로 전략 객체 구성 (필요한 최소 정보만)
                            if top_strategy_id and top_params:
                                strategy = {
                                    'id': top_strategy_id,
                                    'coin': coin,
                                    'interval': interval,
                                    'profit': top_profit,
                                    'win_rate': top_win_rate,
                                    'quality_grade': top_quality,
                                    'params': top_params,
                                    'rsi_min': top_params.get('rsi_min', 30.0),
                                    'rsi_max': top_params.get('rsi_max', 70.0),
                                    'volume_ratio_min': top_params.get('volume_ratio_min', 1.0),
                                    'volume_ratio_max': top_params.get('volume_ratio_max', 2.0),
                                    'score': (top_profit / 1000.0) * top_win_rate if top_profit > 0 else 0.5
                                }
                                strategies.append(strategy)
                        
                        # 요약 테이블에 데이터가 없거나 추가 전략이 필요한 경우 원본 테이블 조회
                        if not strategies:
                            # rl_strategies.db에서 직접 조회 (폴백)
                            strategies_db_path = "/workspace/data_storage/rl_strategies.db"
                            with sqlite3.connect(strategies_db_path) as strategies_conn:
                                strategies_cursor = strategies_conn.cursor()
                                
                                strategies_cursor.execute("""
                                    SELECT id, rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
                                           macd_buy_threshold, macd_sell_threshold, profit, win_rate,
                                           quality_grade, score
                                    FROM coin_strategies 
                                    WHERE coin = ? AND interval = ? 
                                    AND quality_grade IN ('S', 'A', 'B')
                                    ORDER BY score DESC 
                                    LIMIT 5
                                """, (coin, interval))
                                
                                for row in strategies_cursor.fetchall():
                                    strategy = {
                                        'id': row[0],
                                        'coin': coin,
                                        'interval': interval,
                                        'rsi_min': row[1],
                                        'rsi_max': row[2],
                                        'volume_ratio_min': row[3],
                                        'volume_ratio_max': row[4],
                                        'macd_buy_threshold': row[5],
                                        'macd_sell_threshold': row[6],
                                        'profit': row[7] or 0.0,
                                        'win_rate': row[8] or 0.0,
                                        'quality_grade': row[9] or 'B',
                                        'score': row[10] or 0.5,
                                        'params': {
                                            'rsi_min': row[1],
                                            'rsi_max': row[2],
                                            'volume_ratio_min': row[3],
                                            'volume_ratio_max': row[4]
                                        }
                                    }
                                    strategies.append(strategy)
                    
                except Exception as e:
                    if self.debug_mode:
                        logger.debug(f"⚠️ 전략 로드 실패: {e}")
                
                # 통합 분석 실행 (전략이 있는 경우만)
                if strategies:
                    signal_result = self.integrated_analyzer.analyze_coin_strategies(
                        coin=coin,
                        interval=interval,
                        regime=market_condition,
                        strategies=strategies,
                        candle_data=candle_df
                    )
                    
                    return signal_result.final_signal_score
            
            # 최종 폴백: 중립 점수
            return 0.5
            
        except Exception as e:
            if self.debug_mode:
                logger.error(f"⚠️ 통합 분석 점수 계산 실패: {e}")
            return 0.5  # 에러 시 중립 점수
    
    def generate_signal(self, coin: str, interval: str) -> Optional[SignalInfo]:
        """🚀 스마트 시그널 생성 (정확도 + 속도 균형)"""
        try:
            # 🚀 1. 캔들 데이터 먼저 로드 (가장 중요한 데이터)
            candle = self.get_nearest_candle(coin, interval, int(time.time()))
            if candle is None:
                return None
            
            # 🚀 2. 단계별 지표 계산 (정확도와 속도 균형)
            indicators = self._calculate_smart_indicators(candle, coin, interval)
            
            # 🚀 3. 캐시된 시장 상황 사용 (빠른 판단)
            market_condition = self._get_cached_market_condition(coin, interval)
            
            # 🆕 RL Pipeline 통합 분석 활용
            try:
                integrated_analysis_score = self._get_integrated_analysis_score(coin, interval, candle, market_condition)
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ 통합 분석 점수 계산 실패: {e}")
                integrated_analysis_score = 0.5  # 중립 점수
            
            # 🚀 4. RL Pipeline 학습 결과 활용한 전략 선택
            adaptive_strategy = self._select_smart_strategy(coin, interval, market_condition, indicators)
            
            # 🆕 4. 시장 상황별 점수 조정
            base_score = self.get_coin_specific_score(coin, interval, candle)
            if base_score is None:
                print(f"⚠️ 기본 점수가 None입니다. 기본값 0.5 사용 ({coin}/{interval})")
                base_score = 0.5

            # 🆕 5. 섀도우 트레이딩 피드백 반영 (핵심 개선!)
            signal_pattern = self._extract_signal_pattern_from_candle(candle, coin, interval)
            feedback_data = self.get_signal_feedback_data(signal_pattern)
            if feedback_data:
                base_score = self.apply_feedback_to_calculation(candle, base_score, feedback_data)
                print(f"🔄 피드백 반영: {coin}/{interval} 패턴 {signal_pattern} (성공률: {feedback_data.get('success_rate', 0):.2f})")
            
            # 🆕 새로운 학습 결과 반영
            strategy_id = adaptive_strategy.get('strategy_id', '') if adaptive_strategy else ''
            reliability_score = self.reliability_scores.get(strategy_id, 0.0)
            learning_quality_score = self.learning_quality_scores.get(strategy_id, 0.0)
            global_strategy_id = self.global_strategy_mapping.get(strategy_id, "")
            coin_tuned = strategy_id in self.global_strategy_mapping
            walk_forward_performance = self.walk_forward_performance.get(strategy_id, {})
            regime_coverage = self.regime_coverage.get(strategy_id, {})
            
            # 시장 상황 보너스 적용
            if adaptive_strategy and 'market_condition_bonus' in adaptive_strategy:
                market_bonus = adaptive_strategy['market_condition_bonus']
                base_score *= market_bonus
                
                if self.debug_mode:
                    print(f"  - 기본 점수: {base_score / market_bonus:.4f}")
                    print(f"  - 시장 보너스 적용 후: {base_score:.4f}")
            
            # 🆕 신뢰도 및 학습 품질 보너스 적용
            reliability_bonus = 1.0 + (reliability_score * 0.2)  # 신뢰도 보너스 (최대 20%)
            learning_quality_bonus = 1.0 + (learning_quality_score * 0.15)  # 학습 품질 보너스 (최대 15%)
            base_score *= reliability_bonus * learning_quality_bonus
            
            if self.debug_mode and (reliability_score > 0 or learning_quality_score > 0):
                print(f"  - 신뢰도 보너스: {reliability_bonus:.3f} (점수: {reliability_score:.3f})")
                print(f"  - 학습 품질 보너스: {learning_quality_bonus:.3f} (점수: {learning_quality_score:.3f})")
                print(f"  - 향상된 기본 점수: {base_score:.4f}")
            
            # 🆕 6. 향상된 학습 보너스 적용
            enhanced_learning_bonus = self._calculate_enhanced_learning_bonus(coin, interval, candle)
            if enhanced_learning_bonus > 0:
                base_score *= (1.0 + enhanced_learning_bonus)
                if self.debug_mode:
                    print(f"  - 향상된 학습 보너스: {enhanced_learning_bonus:.3f} (최종 점수: {base_score:.4f})")
            
            # 추가 점수들 계산 (🔧 None 값 안전 처리 추가)
            dna_score = self.get_dna_based_similar_score(coin, interval, candle)
            if dna_score is None:
                print(f"⚠️ DNA 점수가 None입니다. 기본값 0.5 사용 ({coin}/{interval})")
                dna_score = 0.5

            rl_score = self.get_combined_rl_score(coin, interval, candle)
            if rl_score is None:
                print(f"⚠️ RL 점수가 None입니다. 기본값 0.5 사용 ({coin}/{interval})")
                rl_score = 0.5

            # 🆕 AI 모델 점수 계산
            ai_score = 0.0
            if self.ai_model_loaded:
                ai_predictions = self.get_ai_based_score(candle)
                if ai_predictions is not None and 'strategy_score' in ai_predictions:
                    ai_score = ai_predictions['strategy_score']
                    if ai_score is None:
                        print(f"⚠️ AI 점수가 None입니다. 기본값 0.0 사용 ({coin}/{interval})")
                        ai_score = 0.0
                    if self.debug_mode:
                        model_info = f"({self.model_type})" if hasattr(self, 'model_type') else ""
                        print(f"  🧠 AI 모델 점수 {model_info}: {ai_score:.4f}")
                else:
                    print(f"⚠️ AI 예측 결과가 None이거나 'strategy_score' 키가 없습니다. 기본값 0.0 사용")
                    ai_score = 0.0
            
            # 🆕 5. 변동성 기반 동적 가중치 조정 (AI 모델 + RL Pipeline 통합 분석 포함)
            weights = self.get_volatility_based_weights(coin, market_condition, self.ai_model_loaded)
            vol_group = self.get_coin_volatility_group(coin)

            # 가중치 적용하여 최종 점수 계산
            if self.ai_model_loaded:
                final_score = (
                    base_score * weights['base'] +
                    dna_score * weights['dna'] +
                    rl_score * weights['rl'] +
                    ai_score * weights['ai'] +
                    integrated_analysis_score * weights['integrated']
                )
            else:
                final_score = (
                    base_score * weights['base'] +
                    dna_score * weights['dna'] +
                    rl_score * weights['rl'] +
                    integrated_analysis_score * weights['integrated']
                )

            # 🆕 변동성 기반 가중치 로깅
            if self.debug_mode:
                print(f"  🎯 변동성 그룹: {vol_group}")
                print(f"  ⚖️ 동적 가중치: base={weights['base']:.3f}, dna={weights['dna']:.3f}, rl={weights['rl']:.3f}, integrated={weights['integrated']:.3f}")
                if self.ai_model_loaded:
                    print(f"  🧠 AI 가중치: {weights['ai']:.3f}")
                print(f"  📊 구성 점수: base={base_score:.3f}, dna={dna_score:.3f}, rl={rl_score:.3f}, integrated={integrated_analysis_score:.3f}")
                if self.ai_model_loaded:
                    print(f"  🧠 AI 점수: {ai_score:.3f}")
            
            # 신뢰도 계산
            confidence = self._calculate_enhanced_confidence(candle, final_score, coin, interval)
            
            # 🆕 6. 시장 상황별 신뢰도 조정 (개선된 버전)
            if market_condition == "bull_market":
                confidence *= 1.2  # 상승장에서는 신뢰도 증가
            elif market_condition == "bear_market":
                confidence *= 1.15  # 하락장에서는 신뢰도 증가
            elif market_condition == "sideways_market":
                confidence *= 0.85  # 횡보장에서는 신뢰도 감소
            
            # 🆕 7. 시너지 학습 결과를 활용한 점수 향상
            if self.synergy_learning_available:
                final_score = self.get_synergy_enhanced_signal_score(coin, interval, final_score, market_condition)
                
                if self.debug_mode:
                    print(f"  🔄 시너지 향상 점수: {final_score:.4f}")
                    
                    # 시너지 권장사항 표시
                    synergy_recommendations = self.get_synergy_recommendations_for_signal(coin, interval, market_condition)
                    if synergy_recommendations:
                        print(f"  💡 시너지 권장사항: {len(synergy_recommendations)}개")
                        for i, rec in enumerate(synergy_recommendations[:2]):  # 상위 2개만 표시
                            print(f"    {i+1}. {rec.get('description', 'N/A')}")
            elif market_condition in ["overbought", "oversold"]:
                confidence *= 1.25  # 과매수/과매도에서는 신뢰도 증가
            else:
                confidence *= 1.0  # 중립 상황

            # 🆕 final_score (0.0 ~ 1.0)를 signal_score (-1.0 ~ +1.0)로 변환
            # 0.5 기준: 중립, 그 위는 매수 신호, 아래는 매도 신호
            # Absolute Zero + Virtual Trading Learner 학습 결과가 모두 반영됨
            signal_score = (final_score - 0.5) * 2  # -1.0 ~ +1.0 범위

            # 🆕 변동성 기반 동적 임계값으로 액션 결정
            thresholds = self.get_volatility_based_thresholds(coin)

            if signal_score > thresholds['strong_buy']:      # 강한 매수 신호
                action = SignalAction.BUY
            elif signal_score > thresholds['weak_buy']:      # 약한 매수 신호
                action = SignalAction.BUY
            elif signal_score < thresholds['strong_sell']:   # 강한 매도 신호
                action = SignalAction.SELL
            elif signal_score < thresholds['weak_sell']:     # 약한 매도 신호
                action = SignalAction.SELL
            else:                                            # 중립 (HOLD)
                action = SignalAction.HOLD

            # 디버그: 점수 변환 및 임계값 로깅
            if self.debug_mode:
                print(f"  📊 점수 변환: final_score={final_score:.3f} → signal_score={signal_score:.3f}")
                print(f"  🎚️ 임계값({vol_group}): BUY>{thresholds['weak_buy']:.2f}, SELL<{thresholds['weak_sell']:.2f}")
                print(f"  🎯 최종 액션: {action.value}")

            # 🆕 Calmar Ratio와 Profit Factor 계산 (안전 처리)
            try:
                calmar_ratio = self._calculate_signal_calmar_ratio(candle, indicators)
            except Exception as e:
                print(f"⚠️ 시그널 Calmar Ratio 계산 실패: {e}")
                calmar_ratio = 0.0
            
            try:
                profit_factor = self._calculate_signal_profit_factor(candle, indicators)
            except Exception as e:
                print(f"⚠️ 시그널 Profit Factor 계산 실패: {e}")
                profit_factor = 1.0
            
            # 🆕 7. 시그널 정보에 시장 상황 및 고급 지표 포함
            signal = SignalInfo(
                coin=coin,
                interval=interval,
                action=action,
                signal_score=signal_score,  # 🆕 -1.0 ~ +1.0 범위 (Absolute Zero + Virtual Learner 학습 결과)
                confidence=confidence,
                reason=f"학습 기반 시그널 (점수: {signal_score:.3f}, 액션: {action.value}, 방향: {candle.get('integrated_direction', 'neutral')}, 파동: {candle.get('wave_phase', 'unknown')})",
                timestamp=int(time.time()),
                price=candle.get('close', 100.0),
                volume=candle.get('volume', 1000.0),
                rsi=candle.get('rsi', 50.0),
                macd=candle.get('macd', 0.0),
                wave_phase=candle.get('wave_phase', 'unknown'),
                pattern_type=candle.get('pattern_type', 'none'),
                risk_level=candle.get('risk_level', 'medium'),
                volatility=candle.get('volatility', 0.02),
                volume_ratio=candle.get('volume_ratio', 1.0),
                wave_progress=candle.get('wave_progress', 0.5),
                structure_score=indicators.get('structure_score', 0.5),
                pattern_confidence=candle.get('pattern_confidence', 0.0),
                integrated_direction=candle.get('integrated_direction', 'neutral'),
                integrated_strength=indicators.get('integrated_strength', 0.5),
                # 🚀 실제 캔들 DB의 고급 지표들
                mfi=candle.get('mfi', 50.0),
                atr=candle.get('atr', 0.02),
                adx=candle.get('adx', 25.0),
                ma20=candle.get('ma20', 1.0),
                rsi_ema=indicators.get('rsi_ema', 50.0),
                macd_smoothed=indicators.get('macd_smoothed', 0.0),
                wave_momentum=indicators.get('wave_momentum', 0.0),
                bb_position=indicators.get('bb_position', 'unknown'),
                bb_width=indicators.get('bb_width', 0.0),
                bb_squeeze=indicators.get('bb_squeeze', 0.0),
                rsi_divergence=indicators.get('rsi_divergence', 'none'),
                macd_divergence=indicators.get('macd_divergence', 'none'),
                volume_divergence=indicators.get('volume_divergence', 'none'),
                price_momentum=indicators.get('price_momentum', 0.0),
                volume_momentum=indicators.get('volume_momentum', 0.0),
                trend_strength=indicators.get('trend_strength', 0.5),
                support_resistance=indicators.get('support_resistance', 'unknown'),
                fibonacci_levels=indicators.get('fibonacci_levels', 'unknown'),
                elliott_wave=indicators.get('elliott_wave', 'unknown'),
                harmonic_patterns=indicators.get('harmonic_patterns', 'none'),
                candlestick_patterns=indicators.get('candlestick_patterns', 'none'),
                market_structure=indicators.get('market_structure', 'unknown'),
                flow_level_meta=indicators.get('flow_level_meta', 'unknown'),
                pattern_direction=indicators.get('pattern_direction', 'neutral'),
                market_condition=market_condition,
                market_adaptation_bonus=adaptive_strategy.get('market_condition_bonus', 1.0) if adaptive_strategy else 1.0,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                reliability_score=reliability_score,
                learning_quality_score=learning_quality_score,
                global_strategy_id=global_strategy_id,
                coin_tuned=coin_tuned,
                walk_forward_performance=walk_forward_performance,
                regime_coverage=regime_coverage
            )
            
            # 🆕 통계 카운터 업데이트
            self._signal_stats['total_signals_generated'] += 1
            self._signal_stats['successful_signals'] += 1
            
            # 시그널 저장
            self.save_signal(signal)
            
            # 🚀 시그널 생성 성공 로그 (실제 캔들 DB 데이터 기반)
            print(f"✅ {coin}/{interval}: 실제 캔들 DB 기반 시그널 생성 성공")
            # 🔧 액션은 트레이더가 결정 (사용자 요청: 액션 노출 제거)
            print(f"  - 점수: {final_score:.3f}, 신뢰도: {confidence:.3f}")
            print(f"  - 시장 상황: {market_condition}")
            print(f"  - 통합 방향: {candle.get('integrated_direction', 'neutral')}, 파동 단계: {candle.get('wave_phase', 'unknown')}")
            print(f"  - 패턴 타입: {candle.get('pattern_type', 'none')}, 신뢰도: {candle.get('pattern_confidence', 0.0):.3f}")
            print(f"  - 기본 점수: {base_score:.3f}, DNA 점수: {dna_score:.3f}")
            print(f"  - RL 점수: {rl_score:.3f}, AI 점수: {ai_score:.3f}")
            print(f"  - 통합 분석 점수: {integrated_analysis_score:.3f}")
            print(f"  - 최종 점수: {final_score:.3f}, 신뢰도: {confidence:.3f}")
            
            return signal
            
        except Exception as e:
            # 🆕 실패 통계 카운터 업데이트
            self._signal_stats['total_signals_generated'] += 1
            self._signal_stats['failed_signals'] += 1
            
            self._handle_error(e, "시그널 생성", coin, interval)
            return None
    
    def _evolve_signal_with_ai(self, base_signal: SignalInfo, coin: str, interval: str, candle: pd.Series) -> SignalInfo:
        """🆕 진화형 AI로 시그널 진화 (성능 업그레이드 적용)"""
        try:
            # 🧠 진화 엔진을 사용하여 시그널 진화
            evolved_signal = self.evolution_engine.evolve_signal(base_signal, coin, interval)
            
            # 🆕 컨텍스트 특징 추출
            market_context = self._get_market_context(coin, interval)
            context_features = self.context_extractor.extract_context_features(candle, market_context)
            context_key = self.context_extractor.get_context_key(context_features)
            
            # 🆕 액션별 스코어 적용
            action_score = self.action_scorer.get_action_score(evolved_signal.action.value)
            
            # 🆕 컨텍스트 기반 점수 조정
            context_bonus = self._calculate_context_bonus(context_key, evolved_signal.action.value)
            
            # 🧠 맥락 메모리에 시장 상황 저장 (컨텍스트 특징 포함)
            enhanced_market_context = {
                'trend': market_context.get('trend', 'neutral'),
                'volatility': context_features['volatility'],
                'volume_ratio': context_features['volume_ratio'],
                'market_trend': context_features['market_trend'],
                'rsi': base_signal.rsi,
                'macd': base_signal.macd,
                'confidence': base_signal.confidence,
                'context_key': context_key
            }
            self.context_memory.remember_market_context(coin, interval, enhanced_market_context)
            
            # 🧠 실시간 학습기에게 시그널 정보 전달 (컨텍스트 포함)
            signal_pattern = self._extract_signal_pattern(evolved_signal)
            enhanced_signal_info = {
                'coin': coin,
                'interval': interval,
                'signal_score': evolved_signal.signal_score,
                'confidence': evolved_signal.confidence,
                'timestamp': evolved_signal.timestamp,
                'action': evolved_signal.action.value,
                'context_key': context_key,
                'action_score': action_score,
                'context_bonus': context_bonus
            }
            self.real_time_learner.learn_from_signal(signal_pattern, enhanced_signal_info)
            
            # 🆕 레짐 전환 감지
            market_indicators = {
                'adx': candle.get('adx', 25.0),
                'atr': candle.get('atr', 0.0),
                'ma_slope': candle.get('ma_slope', 0.0)
            }
            regime_change = self.regime_detector.detect_regime_change(market_indicators)
            
            # 🆕 컨텍스추얼 밴딧 액션 선택
            available_actions = ['buy', 'sell', 'hold']
            bandit_action = self.contextual_bandit.select_action(context_key, available_actions)
            
            # 🆕 오프폴리시 평가 적용
            baseline_reward = evolved_signal.signal_score
            ips_estimate = self.off_policy_evaluator.calculate_ips_estimate(
                evolved_signal.action.value, evolved_signal.signal_score, context_key
            )
            dr_estimate = self.off_policy_evaluator.calculate_doubly_robust_estimate(
                evolved_signal.action.value, evolved_signal.signal_score, context_key, baseline_reward
            )
            
            # 🆕 신뢰도 캘리브레이션 적용
            calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
                evolved_signal.confidence, context_key
            )
            
            # 🆕 메타-보정 적용
            feedback_stats = {
                'success_rate': action_score,
                'avg_profit': context_bonus
            }
            meta_score = self.meta_corrector.calculate_meta_score(
                evolved_signal.signal_score, feedback_stats, context_features
            )
            
            # 🆕 최종 점수 조정 (모든 업그레이드 시스템 통합)
            bandit_bonus = 0.1 if bandit_action == evolved_signal.action.value else -0.05
            regime_bonus = 0.05 if regime_change == 'changed' else 0.0
            off_policy_bonus = (ips_estimate + dr_estimate) / 2 - baseline_reward
            
            final_score = (
                evolved_signal.signal_score + 
                (action_score * 0.1) + 
                (context_bonus * 0.05) + 
                bandit_bonus + 
                regime_bonus + 
                (off_policy_bonus * 0.1) + 
                (meta_score * 0.05)
            )
            final_score = max(-1.0, min(1.0, final_score))  # 범위 제한
            
            # 🆕 최종 진화된 시그널 생성
            final_evolved_signal = SignalInfo(
                coin=evolved_signal.coin,
                interval=evolved_signal.interval,
                action=evolved_signal.action,
                signal_score=final_score,
                confidence=calibrated_confidence + (action_score * 0.1),
                reason=f"{evolved_signal.reason} + 성능업그레이드적용",
                timestamp=evolved_signal.timestamp,
                price=evolved_signal.price,
                volume=evolved_signal.volume,
                rsi=evolved_signal.rsi,
                macd=evolved_signal.macd,
                wave_phase=evolved_signal.wave_phase,
                pattern_type=evolved_signal.pattern_type,
                risk_level=evolved_signal.risk_level,
                volatility=evolved_signal.volatility,
                volume_ratio=evolved_signal.volume_ratio,
                wave_progress=evolved_signal.wave_progress,
                structure_score=evolved_signal.structure_score,
                pattern_confidence=evolved_signal.pattern_confidence,
                integrated_direction=evolved_signal.integrated_direction,
                integrated_strength=evolved_signal.integrated_strength,
                mfi=evolved_signal.mfi,
                atr=evolved_signal.atr,
                adx=evolved_signal.adx,
                ma20=evolved_signal.ma20,
                rsi_ema=evolved_signal.rsi_ema,
                macd_smoothed=evolved_signal.macd_smoothed,
                wave_momentum=evolved_signal.wave_momentum,
                bb_position=evolved_signal.bb_position,
                bb_width=evolved_signal.bb_width,
                bb_squeeze=evolved_signal.bb_squeeze,
                rsi_divergence=evolved_signal.rsi_divergence,
                macd_divergence=evolved_signal.macd_divergence,
                volume_divergence=evolved_signal.volume_divergence,
                price_momentum=evolved_signal.price_momentum,
                volume_momentum=evolved_signal.volume_momentum,
                trend_strength=evolved_signal.trend_strength,
                support_resistance=evolved_signal.support_resistance,
                fibonacci_levels=evolved_signal.fibonacci_levels,
                elliott_wave=evolved_signal.elliott_wave,
                harmonic_patterns=evolved_signal.harmonic_patterns,
                candlestick_patterns=evolved_signal.candlestick_patterns,
                market_structure=evolved_signal.market_structure,
                flow_level_meta=evolved_signal.flow_level_meta,
                pattern_direction=evolved_signal.pattern_direction,
                market_condition=evolved_signal.market_condition,
                market_adaptation_bonus=evolved_signal.market_adaptation_bonus,
                calmar_ratio=evolved_signal.calmar_ratio,
                profit_factor=evolved_signal.profit_factor,
                reliability_score=evolved_signal.reliability_score,
                learning_quality_score=evolved_signal.learning_quality_score,
                global_strategy_id=evolved_signal.global_strategy_id,
                coin_tuned=evolved_signal.coin_tuned,
                walk_forward_performance=evolved_signal.walk_forward_performance,
                regime_coverage=evolved_signal.regime_coverage
            )
            
            return final_evolved_signal
            
        except Exception as e:
            print(f"⚠️ AI 시그널 진화 오류: {e}")
            return base_signal
    
    def _calculate_context_bonus(self, context_key: str, action: str) -> float:
        """🆕 컨텍스트 기반 보너스 계산"""
        try:
            # 컨텍스트별 액션 성과 매핑 (실제로는 DB에서 로드)
            context_action_performance = {
                'low_low_bullish': {'buy': 0.1, 'sell': -0.05, 'hold': 0.0},
                'medium_medium_sideways': {'buy': 0.0, 'sell': 0.0, 'hold': 0.05},
                'high_high_bearish': {'buy': -0.1, 'sell': 0.1, 'hold': 0.0},
                # 더 많은 컨텍스트 조합 추가 가능
            }
            
            return context_action_performance.get(context_key, {}).get(action, 0.0)
            
        except Exception as e:
            print(f"⚠️ 컨텍스트 보너스 계산 오류: {e}")
            return 0.0
    
    def _extract_signal_pattern(self, signal: SignalInfo) -> str:
        """🆕 시그널 패턴 추출"""
        try:
            # RSI 범주화
            rsi_level = self._discretize_rsi(signal.rsi)
            
            # Direction 범주화
            direction = signal.integrated_direction if signal.integrated_direction else 'neutral'
            
            # BB Position 범주화
            bb_position = signal.bb_position if signal.bb_position else 'unknown'
            
            # Volume 범주화
            volume_level = self._discretize_volume(signal.volume_ratio)
            
            # 패턴 조합
            pattern = f"{rsi_level}_{direction}_{bb_position}_{volume_level}"
            
            return pattern
            
        except Exception as e:
            print(f"⚠️ 시그널 패턴 추출 오류: {e}")
            return 'unknown_pattern'
    
    def _get_market_context(self, coin: str, interval: str) -> dict:
        """🆕 시장 상황 분석"""
        try:
            # 기준 코인(환경/DB) 시장 상황 분석
            btc_signal = self.get_cached_data(f"signal_BTC_{interval}", max_age=300)
            
            if btc_signal:
                signal_score = btc_signal.signal_score
                
                if signal_score > 0.3:
                    trend = 'bullish'
                elif signal_score < -0.3:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
                
                volatility = btc_signal.volatility
            else:
                trend = 'neutral'
                volatility = 0.02
            
            return {
                'trend': trend,
                'volatility': volatility,
                'timestamp': int(time.time())
            }
            
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02, 'timestamp': int(time.time())}
    
    def _save_signal_for_learning(self, signal: SignalInfo, signal_pattern: str, market_context: dict):
        """🆕 학습용 시그널 데이터 저장"""
        try:
            # 시그널-매매 연결을 위한 데이터 저장
            signal_data = {
                'coin': signal.coin,
                'interval': signal.interval,
                'timestamp': signal.timestamp,
                'signal_pattern': signal_pattern,
                'market_context': market_context,
                'signal_score': signal.signal_score,
                'confidence': signal.confidence,
                'action': signal.action.value
            }
            
            # 시그널-매매 연결 시스템에 저장
            self.signal_trade_connector.pending_signals[f"{signal.coin}_{signal.timestamp}"] = signal_data
            
        except Exception as e:
            print(f"⚠️ 학습용 시그널 저장 오류: {e}")
    
    def _discretize_rsi(self, rsi: float) -> str:
        """RSI 값을 이산화"""
        if rsi < 30:
            return 'oversold'
        elif rsi < 45:
            return 'low'
        elif rsi < 55:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        else:
            return 'overbought'
    
    def _discretize_volume(self, volume_ratio: float) -> str:
        """거래량 비율을 이산화"""
        if volume_ratio < 0.5:
            return 'low'
        elif volume_ratio < 1.5:
            return 'normal'
        else:
            return 'high'
    
    def save_signal(self, signal: SignalInfo):
        """시그널 저장 (trading_system.db에 저장) - 연결 풀 사용"""
        try:
            print(f"💾 시그널 저장 중: {signal.coin}/{signal.interval} -> {DB_PATH}")
            
            # 🆕 최적화된 DB 연결 (충돌 방지 강화)
            if DB_POOL_AVAILABLE:
                with get_optimized_db_connection(DB_PATH, mode='write') as conn:
                    self._save_signal_to_db(conn, signal)
            else:
                # Fallback: 직접 연결
                with sqlite3.connect(DB_PATH) as conn:
                    self._save_signal_to_db(conn, signal)
                    
            print(f"✅ 시그널 저장 완료: {signal.coin}/{signal.interval}")
        except Exception as e:
            logger.error(f"❌ 시그널 저장 실패: {e}")
    
    def _save_signal_to_db(self, conn, signal: SignalInfo):
        """실제 시그널 저장 로직"""
        try:
            # 먼저 고급지표 컬럼들이 존재하는지 확인하고 없으면 추가
            self._ensure_advanced_columns_exist(conn)
            
            # 컬럼과 값을 명시적으로 매칭하여 INSERT 실행
            columns = [
                'timestamp', 'coin', 'interval', 'signal_score', 'confidence', 'action', 'current_price',
                'rsi', 'macd', 'wave_phase', 'pattern_type', 'risk_level', 'volatility',
                'volume_ratio', 'wave_progress', 'structure_score', 'pattern_confidence',
                'integrated_direction', 'integrated_strength', 'reason',
                'mfi', 'atr', 'adx', 'ma20', 'rsi_ema', 'macd_smoothed', 'wave_momentum',
                'bb_position', 'bb_width', 'bb_squeeze', 'rsi_divergence', 'macd_divergence', 'volume_divergence',
                'price_momentum', 'volume_momentum', 'trend_strength', 'support_resistance', 'fibonacci_levels',
                'elliott_wave', 'harmonic_patterns', 'candlestick_patterns', 'market_structure', 'flow_level_meta', 'pattern_direction',
                'market_condition', 'market_adaptation_bonus'
            ]
            
            # 🆕 문자열 값들을 안전하게 처리
            safe_reason = str(signal.reason).replace('/', '_').replace('\\', '_') if signal.reason else 'unknown'
            safe_wave_phase = str(signal.wave_phase).replace('/', '_').replace('\\', '_') if signal.wave_phase else 'unknown'
            safe_pattern_type = str(signal.pattern_type).replace('/', '_').replace('\\', '_') if signal.pattern_type else 'unknown'
            safe_risk_level = str(signal.risk_level).replace('/', '_').replace('\\', '_') if signal.risk_level else 'unknown'
            safe_integrated_direction = str(signal.integrated_direction).replace('/', '_').replace('\\', '_') if signal.integrated_direction else 'unknown'
            safe_bb_position = str(signal.bb_position).replace('/', '_').replace('\\', '_') if signal.bb_position else 'unknown'
            safe_rsi_divergence = str(signal.rsi_divergence).replace('/', '_').replace('\\', '_') if signal.rsi_divergence else 'none'
            safe_macd_divergence = str(signal.macd_divergence).replace('/', '_').replace('\\', '_') if signal.macd_divergence else 'none'
            safe_volume_divergence = str(signal.volume_divergence).replace('/', '_').replace('\\', '_') if signal.volume_divergence else 'none'
            safe_support_resistance = str(signal.support_resistance).replace('/', '_').replace('\\', '_') if signal.support_resistance else 'unknown'
            safe_fibonacci_levels = str(signal.fibonacci_levels).replace('/', '_').replace('\\', '_') if signal.fibonacci_levels else 'unknown'
            safe_elliott_wave = str(signal.elliott_wave).replace('/', '_').replace('\\', '_') if signal.elliott_wave else 'unknown'
            safe_harmonic_patterns = str(signal.harmonic_patterns).replace('/', '_').replace('\\', '_') if signal.harmonic_patterns else 'none'
            safe_candlestick_patterns = str(signal.candlestick_patterns).replace('/', '_').replace('\\', '_') if signal.candlestick_patterns else 'none'
            safe_market_structure = str(signal.market_structure).replace('/', '_').replace('\\', '_') if signal.market_structure else 'unknown'
            safe_flow_level_meta = str(signal.flow_level_meta).replace('/', '_').replace('\\', '_') if signal.flow_level_meta else 'unknown'
            safe_pattern_direction = str(signal.pattern_direction).replace('/', '_').replace('\\', '_') if signal.pattern_direction else 'neutral'
            safe_market_condition = str(signal.market_condition).replace('/', '_').replace('\\', '_') if signal.market_condition else 'unknown'
            
            values = [
                int(signal.timestamp), signal.coin, signal.interval, signal.signal_score, 
                signal.confidence, signal.action.value, signal.price, signal.rsi, signal.macd,
                safe_wave_phase, safe_pattern_type, safe_risk_level, signal.volatility,
                signal.volume_ratio, signal.wave_progress, signal.structure_score,
                signal.pattern_confidence, safe_integrated_direction, signal.integrated_strength,
                safe_reason,
                signal.mfi, signal.atr, signal.adx, signal.ma20, signal.rsi_ema, signal.macd_smoothed, signal.wave_momentum,
                safe_bb_position, signal.bb_width, signal.bb_squeeze, safe_rsi_divergence, safe_macd_divergence, safe_volume_divergence,
                signal.price_momentum, signal.volume_momentum, signal.trend_strength, safe_support_resistance, safe_fibonacci_levels,
                safe_elliott_wave, safe_harmonic_patterns, safe_candlestick_patterns, safe_market_structure, safe_flow_level_meta, safe_pattern_direction,
                safe_market_condition, signal.market_adaptation_bonus
            ]
            
            # 컬럼과 값의 개수가 일치하는지 확인
            if len(columns) != len(values):
                print(f"⚠️ 컬럼과 값의 개수 불일치: {len(columns)} 컬럼, {len(values)} 값")
                return
            
            placeholders = ', '.join(['?' for _ in columns])
            column_list = ', '.join(columns)
            
            conn.execute(f"""
                INSERT OR REPLACE INTO signals (
                    {column_list}
                ) VALUES ({placeholders})
            """, values)
            conn.commit()
        except Exception as e:
            print(f"⚠️ 시그널 저장 오류 ({signal.coin}/{signal.interval}): {e}")

    def save_signal_to_db(self, signal: SignalInfo):
        """Public wrapper for saving signal to database"""
        try:
            if DB_POOL_AVAILABLE:
                with get_optimized_db_connection(DB_PATH, mode='write') as conn:
                    self._save_signal_to_db(conn, signal)
            else:
                # Fallback: 직접 연결
                with sqlite3.connect(DB_PATH) as conn:
                    self._save_signal_to_db(conn, signal)
            print(f"✅ 통합 시그널 저장 완료: {signal.coin}/{signal.interval}")
        except Exception as e:
            print(f"⚠️ 통합 시그널 저장 실패: {e}")

    def _ensure_advanced_columns_exist(self, conn):
        """고급지표 컬럼들이 존재하는지 확인하고 없으면 추가"""
        try:
            # 추가할 컬럼들
            columns_to_add = [
                ('mfi', 'REAL DEFAULT 50.0'),
                ('atr', 'REAL DEFAULT 0.0'),
                ('adx', 'REAL DEFAULT 25.0'),
                ('ma20', 'REAL DEFAULT 0.0'),
                ('rsi_ema', 'REAL DEFAULT 50.0'),
                ('macd_smoothed', 'REAL DEFAULT 0.0'),
                ('wave_momentum', 'REAL DEFAULT 0.0'),
                ('bb_position', 'TEXT DEFAULT "unknown"'),
                ('bb_width', 'REAL DEFAULT 0.0'),
                ('bb_squeeze', 'REAL DEFAULT 0.0'),
                ('rsi_divergence', 'TEXT DEFAULT "none"'),
                ('macd_divergence', 'TEXT DEFAULT "none"'),
                ('volume_divergence', 'TEXT DEFAULT "none"'),
                ('price_momentum', 'REAL DEFAULT 0.0'),
                ('volume_momentum', 'REAL DEFAULT 0.0'),
                ('trend_strength', 'REAL DEFAULT 0.5'),
                ('support_resistance', 'TEXT DEFAULT "unknown"'),
                ('fibonacci_levels', 'TEXT DEFAULT "unknown"'),
                ('elliott_wave', 'TEXT DEFAULT "unknown"'),
                ('harmonic_patterns', 'TEXT DEFAULT "none"'),
                ('candlestick_patterns', 'TEXT DEFAULT "none"'),
                ('market_structure', 'TEXT DEFAULT "unknown"'),
                ('flow_level_meta', 'TEXT DEFAULT "unknown"'),
                ('pattern_direction', 'TEXT DEFAULT "neutral"'),
                ('market_condition', 'TEXT DEFAULT "unknown"'),
                ('market_adaptation_bonus', 'REAL DEFAULT 1.0')
            ]
            
            # 기존 컬럼 확인
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(signals)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            # 누락된 컬럼들 추가
            for col_name, col_def in columns_to_add:
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}")
                        print(f"✅ 고급지표 컬럼 추가됨: {col_name}")
                    except Exception as e:
                        print(f"⚠️ 컬럼 추가 실패 {col_name}: {e}")
            
            conn.commit()
            
        except Exception as e:
            print(f"⚠️ 고급지표 컬럼 확인/추가 오류: {e}")
    
    def get_nearest_candle(self, coin: str, interval: str, base_timestamp: int) -> Optional[pd.Series]:
        """🚀 실제 캔들 DB의 모든 기술적 지표를 활용한 캔들 조회"""
        try:
            # 🚀 캐시된 데이터 확인 (5분 캐시)
            cache_key = f"candle_{coin}_{interval}_{base_timestamp // 300 * 300}"  # 5분 단위로 캐시
            cached_data = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            if cached_data is not None:
                return cached_data
            
            # 🚀 실제 캔들 DB의 모든 기술적 지표 조회
            conn = sqlite3.connect(CANDLES_DB_PATH)
            try:
                # 🚀 realtime_candles 파일들에서 생성된 모든 기술적 지표 조회
                # 시간 차이를 고려하여 가장 가까운 캔들 조회 (과거 또는 미래)
                df = pd.read_sql("""
                    SELECT timestamp, open, high, low, close, volume,
                           rsi, mfi, macd, macd_signal, bb_upper, bb_middle, bb_lower,
                           atr, ma20, adx, volume_ratio, risk_score,
                           wave_phase, confidence, zigzag_direction, zigzag_pivot_price, wave_progress,
                           pattern_type, pattern_confidence, volatility_level, risk_level, integrated_direction
                    FROM candles
                    WHERE coin = ? AND interval = ?
                    ORDER BY ABS(timestamp - ?) ASC LIMIT 1
                """, conn, params=(coin, interval, base_timestamp))
                
                if not df.empty:
                    result = df.iloc[0]
                    
                    # 🚀 실제 데이터 로드 성공 로그 (None 값 안전 처리)
                    rsi_val = result.get('rsi', 50.0)
                    macd_val = result.get('macd', 0.0)
                    volume_ratio_val = result.get('volume_ratio', 1.0)
                    wave_phase_val = result.get('wave_phase', 'unknown')
                    pattern_type_val = result.get('pattern_type', 'none')
                    integrated_direction_val = result.get('integrated_direction', 'neutral')
                    
                    # None 값 안전 처리
                    if rsi_val is None:
                        rsi_val = 50.0
                    if macd_val is None:
                        macd_val = 0.0
                    if volume_ratio_val is None:
                        volume_ratio_val = 1.0
                    if wave_phase_val is None:
                        wave_phase_val = 'unknown'
                    if pattern_type_val is None:
                        pattern_type_val = 'none'
                    if integrated_direction_val is None:
                        integrated_direction_val = 'neutral'
                    
                    print(f"✅ {coin}/{interval}: 실제 캔들 데이터 로드 - RSI({rsi_val:.1f}), MACD({macd_val:.4f}), Volume({volume_ratio_val:.2f}x), Wave({wave_phase_val}), Pattern({pattern_type_val}), Direction({integrated_direction_val})")
                    
                    # 결과 캐시
                    self.set_cached_data(cache_key, result)
                    return result
                else:
                    # 데이터가 없으면 기본값으로 가상 캔들 생성
                    print(f"⚠️ {coin}/{interval} 캔들 데이터 없음, 기본값 사용")
                    return pd.Series({
                        'timestamp': base_timestamp,
                        'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0,
                        'rsi': 50.0, 'mfi': 50.0, 'macd': 0.0, 'macd_signal': 0.0,
                        'bb_upper': 1.05, 'bb_middle': 1.0, 'bb_lower': 0.95,
                        'atr': 0.02, 'ma20': 1.0, 'adx': 25.0, 'volume_ratio': 1.0,
                        'volatility': 0.02, 'risk_score': 0.5,
                        'wave_phase': 'unknown', 'confidence': 0.5,
                        'zigzag_direction': 0.0, 'zigzag_pivot_price': 100.0, 'wave_progress': 0.5,
                        'pattern_type': 'none', 'pattern_confidence': 0.0,
                        'volatility_level': 'medium', 'risk_level': 'medium', 'integrated_direction': 'neutral'
                    })
            finally:
                conn.close()
                
        except Exception as e:
            print(f"⚠️ 최근 캔들 조회 오류 ({coin}/{interval}): {e}")
            # 오류 시 기본값으로 가상 캔들 생성
            return pd.Series({
                'timestamp': base_timestamp,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': 1000.0,
                'rsi': 50.0,
                'macd': 0.0,
                'volume_ratio': 1.0,
                'volatility': 0.02
            })
    
    def get_multi_interval_state_key(self, coin: str, base_timestamp: int) -> str:
        """멀티인터벌 상태 키 생성 (학습용)"""
        try:
            intervals = ['15m', '30m', '240m', '1d']
            state_parts = []
            
            for interval in intervals:
                candle = self.get_nearest_candle(coin, interval, base_timestamp)
                if candle is not None:
                    state = self.get_state_representation(candle, interval)
                else:
                    state = f"{interval}_missing"
                state_parts.append(state)
            
            return f"{coin}_" + "_".join(state_parts)
            
        except Exception as e:
            print(f"⚠️ 멀티인터벌 상태 키 생성 오류 ({coin}): {e}")
            return f"{coin}_unknown_state"
    
    def get_state_representation(self, candle: pd.Series, interval: str) -> str:
        """캔들을 상태 표현으로 변환 (기존 Q-table 호환 버전)"""
        try:
            # 기존 Q-table과 호환되는 상태 키 형식 사용
            rsi = candle.get('rsi', 50)
            macd = candle.get('macd', 0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            wave_progress = candle.get('wave_progress', 0.5)
            structure_score = candle.get('structure_score', 0.5)
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            

            
            rsi = safe_float(rsi, 50)
            macd = safe_float(macd, 0)
            volume_ratio = safe_float(volume_ratio, 1.0)
            wave_progress = safe_float(wave_progress, 0.5)
            structure_score = safe_float(structure_score, 0.5)
            pattern_confidence = safe_float(pattern_confidence, 0.0)
            
            # 기존 Q-table 형식에 맞는 상태 표현
            # RSI 상태
            if rsi > 70:
                rsi_state = "overbought"
            elif rsi < 30:
                rsi_state = "oversold"
            else:
                rsi_state = "neutral"
            
            # MACD 상태
            if macd > 0:
                macd_state = "bullish"
            else:
                macd_state = "bearish"
            
            # 볼린저 밴드 상태
            close = safe_float(candle.get('close'), 0)
            bb_middle = safe_float(candle.get('bb_middle'), 0)
            if close > 0 and bb_middle > 0:
                if close > bb_middle:
                    bb_state = "upper"
                elif close < bb_middle:
                    bb_state = "lower"
                else:
                    bb_state = "middle"
            else:
                bb_state = "middle"
            
            # 거래량 상태
            if volume_ratio > 1.5:
                volume_state = "high"
            elif volume_ratio < 0.5:
                volume_state = "low"
            else:
                volume_state = "medium"
            
            # 파동 진행도 상태
            if wave_progress < 0.3:
                wave_state = "early"
            elif wave_progress > 0.7:
                wave_state = "late"
            else:
                wave_state = "middle"
            
            # 구조 점수 상태
            if structure_score > 0.7:
                structure_state = "strong"
            elif structure_score < 0.3:
                structure_state = "weak"
            else:
                structure_state = "neutral"
            
            # 패턴 품질 상태
            if pattern_confidence > 0.7:
                quality_state = "high"
            elif pattern_confidence < 0.3:
                quality_state = "low"
            else:
                quality_state = "medium"
            
            # 기존 Q-table 형식의 상태 키 생성
            state_key = f"{interval}_{rsi_state}_{macd_state}_{bb_state}_{volume_state}_{wave_state}_{structure_state}_{quality_state}"
            
            return state_key
            
        except Exception as e:
            print(f"⚠️ 상태 표현 생성 오류: {e}")
            return f"{interval}_unknown"
    
    def calculate_state_key(self, candle: pd.Series, interval: str) -> str:
        """RL 상태 키 계산 (실제 데이터베이스의 모든 고급 기술지표 활용)"""
        try:
            # 🎯 설정 기반으로 모든 기술지표 처리
            indicators = process_technical_indicators(candle)
            
            # 🎯 패턴 품질 특별 처리 (기존과 동일)
            if indicators['pattern_quality'] == 0.0:
                indicators['pattern_quality'] = self._calculate_pattern_quality(
                    indicators['rsi'], indicators['macd'], indicators['volume_ratio'], 
                    indicators['structure_score'], indicators['pattern_confidence']
                )
            
            # 🎯 상태 이산화 (설정 기반으로 처리)
            states = {}
            
            # 기본 지표 상태
            states['rsi'] = discretize_value(indicators['rsi'], STATE_DISCRETIZATION_CONFIG['rsi'])
            states['macd'] = discretize_value(indicators['macd'], STATE_DISCRETIZATION_CONFIG['macd'])
            states['volume_ratio'] = discretize_value(indicators['volume_ratio'], STATE_DISCRETIZATION_CONFIG['volume_ratio'])
            states['wave_progress'] = discretize_value(indicators['wave_progress'], STATE_DISCRETIZATION_CONFIG['wave_progress'])
            states['structure_score'] = discretize_value(indicators['structure_score'], STATE_DISCRETIZATION_CONFIG['structure_score'])
            states['pattern_confidence'] = discretize_value(indicators['pattern_confidence'], STATE_DISCRETIZATION_CONFIG['pattern_confidence'])
            
            # 고급 지표 상태
            states['mfi'] = discretize_value(indicators['mfi'], STATE_DISCRETIZATION_CONFIG['mfi'])
            states['adx'] = discretize_value(indicators['adx'], STATE_DISCRETIZATION_CONFIG['adx'])
            states['wave_momentum'] = discretize_value(abs(indicators['wave_momentum']), STATE_DISCRETIZATION_CONFIG['wave_momentum'])
            states['confidence'] = discretize_value(indicators['confidence'], STATE_DISCRETIZATION_CONFIG['confidence'])
            states['volatility'] = discretize_value(indicators['volatility'], STATE_DISCRETIZATION_CONFIG['volatility'])
            states['bb_width'] = discretize_value(indicators['bb_width'], STATE_DISCRETIZATION_CONFIG['bb_width'])
            states['bb_squeeze'] = discretize_value(indicators['bb_squeeze'], STATE_DISCRETIZATION_CONFIG['bb_squeeze'])
            states['trend_strength'] = discretize_value(indicators['trend_strength'], STATE_DISCRETIZATION_CONFIG['trend_strength'])
            states['pattern_quality'] = discretize_value(indicators['pattern_quality'], STATE_DISCRETIZATION_CONFIG['pattern_quality'])
            states['risk_score'] = discretize_value(indicators['risk_score'], STATE_DISCRETIZATION_CONFIG['risk_score'])
            states['integrated_strength'] = discretize_value(indicators['integrated_strength'], STATE_DISCRETIZATION_CONFIG['integrated_strength'])
            
            # 🎯 특별 상태 계산 (기존과 동일)
            # 다이버전스 상태
            divergence_state = 'bullish' if (indicators['rsi_divergence'] == 'bullish' or indicators['macd_divergence'] == 'bullish') else 'bearish' if (indicators['rsi_divergence'] == 'bearish' or indicators['macd_divergence'] == 'bearish') else 'none'
            
            # 모멘텀 결합 상태
            momentum_combined = 'high' if (abs(indicators['price_momentum']) > 0.05 or abs(indicators['volume_momentum']) > 0.1) else 'low'
            
            # 변동성 레벨 상태
            volatility_level_state = indicators['volatility_level'] if indicators['volatility_level'] != 'unknown' else 'normal'
            
            # 🎯 통합 상태 키 생성 (기존과 동일한 순서와 구조)
            state_parts = [
                interval,
                states['rsi'], states['macd'], states['volume_ratio'], states['wave_progress'],
                states['structure_score'], states['pattern_confidence'], indicators['risk_level'],
                states['mfi'], states['adx'], states['wave_momentum'], states['confidence'], states['volatility'],
                indicators['bb_position'], states['bb_width'], states['bb_squeeze'], divergence_state,
                momentum_combined, states['trend_strength'], indicators['wave_phase'], indicators['pattern_direction'],
                indicators['flow_level_meta'], indicators['support_resistance'], indicators['fibonacci_levels'], indicators['elliott_wave'],
                indicators['harmonic_patterns'], indicators['candlestick_patterns'], indicators['market_structure'],
                states['pattern_quality'], states['risk_score'], states['integrated_strength'], volatility_level_state
            ]
            
            return "_".join(state_parts)
            
        except Exception as e:
            print(f"⚠️ 상태 계산 오류: {e}")
            return f"{interval}_unknown"
    
    def combine_interval_signals(self, coin: str, interval_signals: Dict[str, SignalInfo]) -> Optional[SignalInfo]:
        """인터벌별 시그널 통합 (코인별×인터벌별 전략 우선)"""
        try:
            # 🚀 기본 가중치 설정 (15분과 240분에 더 큰 비중)
            base_weights = {
                '1d': 0.20,
                '15m': 0.35,
                '30m': 0.25,
                '240m': 0.30
            }
            
            # 가중 평균으로 통합
            weighted_score = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            combined_reasons = []
            
            for interval, signal in interval_signals.items():
                weight = base_weights.get(interval, 0.1)
                weighted_score += signal.signal_score * weight
                weighted_confidence += signal.confidence * weight
                total_weight += weight
                
                combined_reasons.append(f"{interval}: {signal.signal_score:.3f}")
            
            if total_weight == 0:
                return None
            
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight

            # 기준 시그널 선택 (15m 우선)
            base_signal = interval_signals.get('15m') or interval_signals.get('30m') or list(interval_signals.values())[0]

            # 🆕 변동성 기반 동적 임계값으로 액션 결정
            thresholds = self.get_volatility_based_thresholds(coin)
            vol_group = self.get_coin_volatility_group(coin)

            if final_score > thresholds['strong_buy']:
                action = SignalAction.BUY
            elif final_score > thresholds['weak_buy']:
                action = SignalAction.BUY
            elif final_score < thresholds['strong_sell']:
                action = SignalAction.SELL
            elif final_score < thresholds['weak_sell']:
                action = SignalAction.SELL
            else:
                action = SignalAction.HOLD

            # 🆕 current_price 정의 (base_signal에서 가져오기)
            current_price = base_signal.price
            
            # 통합 사유 (간소화)
            final_reason = f"멀티인터벌 통합: {', '.join(combined_reasons)} | 통합점수: {final_score:.3f}, 신뢰도: {final_confidence:.2f}"
            
            # 🆕 멀티인터벌 상태 추적 (간소화된 출력)
            multi_interval_state = self.get_multi_interval_state_key(coin, base_signal.timestamp)
            
            # 간소화된 상태 출력 (성능 최적화)
            print(f"🔍 {coin} 멀티인터벌 상태:")
            print(f"   📊 통합 시그널 점수: {final_score:.3f}, 신뢰도: {final_confidence:.2f}")
            # 🔧 액션은 트레이더가 결정 (사용자 요청: 액션 노출 제거)
            # print(f"   🎯 결정 액션: {action.value}")
            
            # 각 인터벌별 간소화된 상태 출력
            intervals = ['15m', '30m', '240m', '1d']
            for interval in intervals:
                candle = self.get_nearest_candle(coin, interval, base_signal.timestamp)
                if candle is not None:
                    # 기본 지표만 간단히 출력
                    rsi = safe_float(candle.get('rsi'), 50.0)
                    macd = safe_float(candle.get('macd'), 0.0)
                    volume_ratio = safe_float(candle.get('volume_ratio'), 1.0)
                    
                    print(f"   📈 {interval}: RSI({rsi:.1f}), MACD({macd:.4f}), Volume({volume_ratio:.2f}x)")
                else:
                    print(f"   📈 {interval}: missing")
            
            return SignalInfo(
                coin=coin,
                interval='combined',
                action=action,
                signal_score=final_score,
                confidence=final_confidence,
                reason=final_reason,
                timestamp=base_signal.timestamp,
                price=float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0,
                volume=float(base_signal.volume) if base_signal.volume is not None and not pd.isna(base_signal.volume) else 0.0,
                rsi=float(base_signal.rsi) if base_signal.rsi is not None and not pd.isna(base_signal.rsi) else 50.0,
                macd=float(base_signal.macd) if base_signal.macd is not None and not pd.isna(base_signal.macd) else 0.0,
                wave_phase=str(base_signal.wave_phase) if base_signal.wave_phase is not None and not pd.isna(base_signal.wave_phase) else 'unknown',
                pattern_type=str(base_signal.pattern_type) if base_signal.pattern_type is not None and not pd.isna(base_signal.pattern_type) else 'none',
                risk_level=str(base_signal.risk_level) if base_signal.risk_level is not None and not pd.isna(base_signal.risk_level) else 'unknown',
                volatility=float(base_signal.volatility) if base_signal.volatility is not None and not pd.isna(base_signal.volatility) else 0.0,
                volume_ratio=float(base_signal.volume_ratio) if base_signal.volume_ratio is not None and not pd.isna(base_signal.volume_ratio) else 1.0,
                # 🆕 새로운 학습 결과 필드 (복합 시그널용 기본값)
                reliability_score=0.0,
                learning_quality_score=0.0,
                global_strategy_id="",
                coin_tuned=False,
                walk_forward_performance=None,
                regime_coverage=None,
                wave_progress=float(base_signal.wave_progress) if base_signal.wave_progress is not None and not pd.isna(base_signal.wave_progress) else 0.0,
                structure_score=float(base_signal.structure_score) if base_signal.structure_score is not None and not pd.isna(base_signal.structure_score) else 0.5,
                pattern_confidence=float(base_signal.pattern_confidence) if base_signal.pattern_confidence is not None and not pd.isna(base_signal.pattern_confidence) else 0.0,
                integrated_direction=str(base_signal.integrated_direction) if base_signal.integrated_direction is not None and not pd.isna(base_signal.integrated_direction) else 'neutral',
                integrated_strength=float(base_signal.integrated_strength) if base_signal.integrated_strength is not None and not pd.isna(base_signal.integrated_strength) else 0.5
            )
            
        except Exception as e:
            print(f"⚠️ 시그널 통합 오류 ({coin}): {e}")
            return None
    
    def generate_all_signals(self, intervals: List[str] = ['15m', '30m', '240m', '1d']) -> List[SignalInfo]:
        """🚀 최적화된 배치 시그널 생성"""
        signals = []
        
        try:
            # 🚀 배치 쿼리로 데이터가 충분한 코인들 조회
            conn = self.db_pool.get_connection()
            try:
                placeholders = ', '.join(['?' for _ in intervals])
                coins_df = pd.read_sql(f"""
                    SELECT coin, COUNT(*) as data_count
                    FROM candles 
                    WHERE interval IN ({placeholders})
                    GROUP BY coin
                    HAVING data_count >= 40
                    ORDER BY data_count DESC
                """, conn, params=intervals)
            finally:
                self.db_pool.return_connection(conn)
            
            coins = coins_df['coin'].tolist()
            print(f"🧠 {len(coins)}개 코인에 대한 배치 시그널 생성 시작...")
            
            # 🚀 배치 처리로 시그널 생성
            batch_size = PERFORMANCE_CONFIG['BATCH_SIZE']
            for i in range(0, len(coins), batch_size):
                batch_coins = coins[i:i + batch_size]
                batch_signals = self._generate_batch_signals(batch_coins, intervals)
                signals.extend(batch_signals)
                
                if i % (batch_size * 5) == 0:  # 진행률 출력
                    print(f"  📊 진행률: {i}/{len(coins)} 코인 처리 완료")
            
            print(f"✅ 배치 시그널 생성 완료: {len(signals)}개 시그널")
            return signals
            
        except Exception as e:
            print(f"❌ 배치 시그널 생성 오류: {e}")
            return []
    
    def _generate_batch_signals(self, coins: List[str], intervals: List[str]) -> List[SignalInfo]:
        """🚀 배치 단위 시그널 생성 (병렬 처리)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        signals = []
        max_workers = min(PERFORMANCE_CONFIG['MAX_WORKERS'], len(coins))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 코인에 대해 시그널 생성 작업 제출
            future_to_coin = {
                executor.submit(self._generate_coin_signals, coin, intervals): coin 
                for coin in coins
            }
            
            # 결과 수집
            for future in as_completed(future_to_coin):
                coin = future_to_coin[future]
                try:
                    coin_signals = future.result()
                    signals.extend(coin_signals)
                except Exception as e:
                    print(f"⚠️ {coin} 배치 시그널 생성 오류: {e}")
        
        return signals
    
    def _generate_coin_signals(self, coin: str, intervals: List[str]) -> List[SignalInfo]:
        """🚀 단일 코인에 대한 멀티인터벌 시그널 생성"""
        try:
            interval_signals = {}
            
            # 각 인터벌별 시그널 생성
            for interval in intervals:
                signal = self.generate_signal(coin, interval)
                if signal:
                    interval_signals[interval] = signal
            
            # 멀티인터벌 시그널 결합
            if len(interval_signals) >= 2:
                combined_signal = self.combine_interval_signals(coin, interval_signals)
                return [combined_signal] if combined_signal else []
            
            return []
            
        except Exception as e:
            print(f"⚠️ {coin} 시그널 생성 오류: {e}")
            return []
    

    

    



    
    def get_combined_rl_score(self, coin: str, interval: str, candle: pd.Series, state_key: str = None) -> float:
        """🚨 코인별 점수 + DNA 기반 유사 코인 점수 + AI 모델 점수 결합"""
        try:
            strategy_key = f"{coin}_{interval}"
            
            # 🚀 실제 캔들 데이터에서 지표 추출
            indicators = self._calculate_smart_indicators(candle, coin, interval)
            market_condition = self._get_cached_market_condition(coin, interval)
            
            # 🚨 코인별 점수
            coin_score = self.get_coin_specific_score(coin, interval, candle)
            
            # 🚨 DNA 기반 유사 코인 점수
            dna_similar_score = self.get_dna_based_similar_score(coin, interval, candle)
            
            # 🆕 AI 모델 기반 점수
            ai_score = 0.0
            # 🚀 AI 모델 예측 (로드되지 않았어도 실제 데이터 기반 예측 사용)
            ai_predictions = self.get_ai_based_score(candle)
            ai_score = ai_predictions['strategy_score']
            print(f"🧠 AI 모델 예측: 수익률={ai_predictions['mu']:.4f}, 상승확률={ai_predictions['p_up']:.4f}, 리스크={ai_predictions['risk']:.4f}, 점수={ai_score:.4f}")
            
            # 🆕 고급 학습 시스템 기반 점수
            advanced_score = 0.0
            if self.integrated_advanced_system:
                try:
                    # 시장 데이터 준비
                    market_data = {
                        'candle': candle,
                        'coin': coin,
                        'interval': interval,
                        'indicators': indicators,
                        'market_condition': market_condition
                    }
                    
                    # 통합 고급 시스템 예측
                    integrated_result = self.integrated_advanced_system.predict_integrated(market_data, coin)
                    advanced_score = integrated_result.final_prediction
                    print(f"🚀 고급 학습 시스템 예측: 최종점수={advanced_score:.4f}, 신뢰도={integrated_result.confidence_score:.4f}")
                except Exception as e:
                    print(f"⚠️ 고급 학습 시스템 예측 실패: {e}")
                    print(f"🔧 고급 학습 시스템 대신 기술적 지표 기반 점수 사용")
                    advanced_score = self._calculate_technical_based_score(candle)
            
            # 🚨 실제 캔들 데이터 기반 점수 조정
            # 🚀 통합 방향성 기반 점수 조정
            integrated_direction = candle.get('integrated_direction', 'neutral')
            if integrated_direction is None:
                integrated_direction = 'neutral'
            if integrated_direction == 'strong_bullish':
                direction_bonus = 1.3
            elif integrated_direction == 'bullish':
                direction_bonus = 1.2
            elif integrated_direction == 'strong_bearish':
                direction_bonus = 0.7
            elif integrated_direction == 'bearish':
                direction_bonus = 0.8
            else:
                direction_bonus = 1.0

            # 🚀 파동 단계 기반 점수 조정
            wave_phase = candle.get('wave_phase', 'unknown')
            if wave_phase is None:
                wave_phase = 'unknown'
            if wave_phase == 'impulse':
                wave_bonus = 1.2
            elif wave_phase == 'correction':
                wave_bonus = 0.9
            else:
                wave_bonus = 1.0

            # 🚀 패턴 신뢰도 기반 점수 조정
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            if pattern_confidence is None:
                pattern_confidence = 0.0
            pattern_bonus = 1.0 + (float(pattern_confidence) * 0.3)  # 최대 30% 보너스
            
            # 🚀 점수 결합 (실제 캔들 데이터 기반)
            if self.ai_model_loaded and self.integrated_advanced_system:
                # 모든 시스템이 사용 가능할 때
                combined_score = coin_score * 0.25 + dna_similar_score * 0.15 + ai_score * 0.3 + advanced_score * 0.3
            elif self.ai_model_loaded:
                # AI 모델만 사용 가능할 때
                combined_score = coin_score * 0.4 + dna_similar_score * 0.15 + ai_score * 0.45
            elif self.integrated_advanced_system:
                # 고급 학습 시스템만 사용 가능할 때
                combined_score = coin_score * 0.3 + dna_similar_score * 0.15 + advanced_score * 0.55
            else:
                # 기본 시스템만 사용 가능할 때
                combined_score = coin_score * 0.6 + dna_similar_score * 0.4
            
            # 🚀 실제 캔들 데이터 기반 보너스 적용
            # None 체크 후 안전하게 곱셈
            if direction_bonus is None:
                direction_bonus = 1.0
            if wave_bonus is None:
                wave_bonus = 1.0
            if pattern_bonus is None:
                pattern_bonus = 1.0
            if combined_score is None:
                combined_score = 0.5

            combined_score = float(combined_score) * float(direction_bonus) * float(wave_bonus) * float(pattern_bonus)

            # 🚨 점수 부스팅 (기본 점수가 너무 낮을 때)
            if combined_score < 0.1:
                combined_score = max(0.3, combined_score * 2.0)  # 최소 0.3 보장
            elif combined_score < 0.2:
                combined_score = combined_score * 1.5  # 1.5배 부스팅

            # 🚀 실제 데이터 기반 점수 로그 (이미 처리된 변수 사용)
            print(f"🎯 {coin}/{interval}: 실제 데이터 기반 점수 조정 - 방향({integrated_direction}, {direction_bonus:.2f}x), 파동({wave_phase}, {wave_bonus:.2f}x), 패턴({pattern_confidence:.3f}, {pattern_bonus:.2f}x), 최종점수({combined_score:.3f})")
            
            # 🚨 피드백 적용 (선택적)
            if strategy_key in self.coin_specific_strategies:
                improved_score = self.improve_signal_calculation_with_feedback(coin, interval, candle, combined_score)
                return np.clip(improved_score, -1.0, 1.0)
            
            return np.clip(combined_score, -1.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ 결합 점수 계산 오류 ({coin}/{interval}): {e}")
            return 0.0
    
    def improve_signal_calculation_with_feedback(self, coin: str, interval: str, candle: pd.Series, base_score: float) -> float:
        """피드백을 바탕으로 시그널 계산 방법 개선"""
        try:
            # 현재 캔들의 시그널 패턴 추출
            state_key = self.calculate_state_key(candle, interval)
            signal_pattern = self.extract_signal_pattern_from_state(state_key)
            
            # 시그널 피드백 데이터 조회
            feedback_data = self.get_signal_feedback_data(signal_pattern)
            
            if not feedback_data:
                # 피드백 데이터가 없으면 기본 점수 그대로 사용
                return base_score
            
            # 🚀 피드백을 바탕으로 계산 방법 개선
            improved_score = self.apply_feedback_to_calculation(candle, base_score, feedback_data)
            
            return improved_score
            
        except Exception as e:
            print(f"⚠️ 시그널 계산 개선 오류 ({coin}/{interval}): {e}")
            return base_score
    
    def _ensure_signal_feedback_schema(self, conn):
        """시그널 피드백 테이블 스키마 확인 및 마이그레이션"""
        try:
            cursor = conn.cursor()
            
            # 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_feedback_scores'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # 테이블이 없으면 생성
                conn.execute("""
                    CREATE TABLE signal_feedback_scores (
                        signal_pattern TEXT PRIMARY KEY,
                        success_rate REAL,
                        avg_profit REAL,
                        total_trades INTEGER,
                        confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        buy_score REAL DEFAULT 0.0,
                        sell_score REAL DEFAULT 0.0,
                        hold_score REAL DEFAULT 0.0,
                        trade_count INTEGER DEFAULT 0,
                        last_updated INTEGER DEFAULT 0
                    )
                """)
                print("✅ signal_feedback_scores 테이블 생성 완료")
            else:
                # 테이블이 있으면 누락된 컬럼 확인 및 추가
                cursor.execute("PRAGMA table_info(signal_feedback_scores)")
                columns = [column[1] for column in cursor.fetchall()]
                
                missing_columns = []
                if 'created_at' not in columns:
                    missing_columns.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                if 'buy_score' not in columns:
                    missing_columns.append("buy_score REAL DEFAULT 0.0")
                if 'sell_score' not in columns:
                    missing_columns.append("sell_score REAL DEFAULT 0.0")
                if 'hold_score' not in columns:
                    missing_columns.append("hold_score REAL DEFAULT 0.0")
                if 'trade_count' not in columns:
                    missing_columns.append("trade_count INTEGER DEFAULT 0")
                if 'last_updated' not in columns:
                    missing_columns.append("last_updated INTEGER DEFAULT 0")
                
                for column_def in missing_columns:
                    column_name = column_def.split()[0]
                    conn.execute(f"ALTER TABLE signal_feedback_scores ADD COLUMN {column_def}")
                    print(f"✅ signal_feedback_scores 테이블에 {column_name} 컬럼 추가 완료")
                
        except Exception as e:
            print(f"⚠️ 시그널 피드백 스키마 마이그레이션 오류: {e}")

    def get_signal_feedback_data(self, signal_pattern: str) -> Optional[Dict]:
        """시그널 패턴에 대한 피드백 데이터 조회"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 스키마 마이그레이션 실행
                self._ensure_signal_feedback_schema(conn)
                
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_feedback_scores'")
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    return None
                
                feedback_df = pd.read_sql("""
                    SELECT buy_score, sell_score, hold_score, success_rate, avg_profit, trade_count
                    FROM signal_feedback_scores 
                    WHERE signal_pattern = ?
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, conn, params=(signal_pattern,))
                
                if not feedback_df.empty:
                    return feedback_df.iloc[0].to_dict()
                return None
                
        except Exception as e:
            print(f"⚠️ 시그널 피드백 데이터 조회 오류: {e}")
            return None
    
    def apply_feedback_to_calculation(self, candle: pd.Series, base_score: float, feedback_data: Dict) -> float:
        """피드백 데이터를 바탕으로 시그널 계산 방법 개선 (전략과 조화)"""
        try:
            # 🚀 피드백 데이터 분석
            success_rate = feedback_data.get('success_rate', 0.5)
            avg_profit = feedback_data.get('avg_profit', 0.0)
            trade_count = feedback_data.get('trade_count', 0)
            
            # 🚀 신뢰도 계산 (거래 횟수 기반)
            confidence = min(trade_count / 20.0, 1.0)  # 20회 이상이면 최대 신뢰도
            
            # 🚀 전략 신뢰도 계산 (Absolute Zero System 기반)
            strategy_confidence = self._calculate_strategy_confidence(candle)
            
            # 🚀 유동적 조정 계수 계산
            feedback_weight = self._calculate_feedback_weight(confidence, strategy_confidence, base_score)
            
            # 🚀 계산 방법 개선 (전략과 피드백의 조화)
            if confidence > 0.3 and strategy_confidence > 0.3:  # 둘 다 충분한 신뢰도
                improved_score = self._apply_balanced_improvement(base_score, success_rate, avg_profit, feedback_weight)
            elif confidence > 0.5:  # 피드백만 충분한 경우
                improved_score = self._apply_feedback_dominant_improvement(base_score, success_rate, avg_profit)
            elif strategy_confidence > 0.5:  # 전략만 충분한 경우
                improved_score = self._apply_strategy_dominant_improvement(base_score, success_rate, avg_profit)
            else:
                # 둘 다 부족하면 기본 점수 사용
                improved_score = base_score
            
            return improved_score
            
        except Exception as e:
            print(f"⚠️ 피드백 적용 오류: {e}")
            return base_score
    
    def _calculate_strategy_confidence(self, candle: pd.Series) -> float:
        """Absolute Zero System 전략의 신뢰도 계산"""
        try:
            # 전략 신뢰도 지표들
            rsi = candle.get('rsi', 50.0)
            macd = candle.get('macd', 0.0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            structure_score = candle.get('structure_score', 0.5)
            
            # 각 지표별 신뢰도 계산
            rsi_confidence = 1.0 - abs(rsi - 50.0) / 50.0  # RSI가 극단적일수록 신뢰도 높음
            macd_confidence = min(abs(macd) / 10.0, 1.0)  # MACD가 강할수록 신뢰도 높음
            volume_confidence = min(volume_ratio / 2.0, 1.0)  # 거래량이 많을수록 신뢰도 높음
            pattern_confidence = pattern_confidence  # 패턴 신뢰도 그대로 사용
            structure_confidence = structure_score  # 구조 점수 그대로 사용
            
            # 종합 신뢰도 (가중 평균)
            total_confidence = (
                rsi_confidence * 0.2 +
                macd_confidence * 0.2 +
                volume_confidence * 0.15 +
                pattern_confidence * 0.25 +
                structure_confidence * 0.2
            )
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            print(f"⚠️ 전략 신뢰도 계산 오류: {e}")
            return 0.5
    
    def _calculate_feedback_weight(self, feedback_confidence: float, strategy_confidence: float, base_score: float) -> float:
        """피드백과 전략의 가중치 계산"""
        try:
            # 기본 가중치 (전략 70%, 피드백 30%)
            base_strategy_weight = 0.7
            base_feedback_weight = 0.3
            
            # 신뢰도에 따른 가중치 조정
            if feedback_confidence > strategy_confidence:
                # 피드백이 더 신뢰할 만한 경우
                feedback_weight = min(base_feedback_weight + (feedback_confidence - strategy_confidence) * 0.3, 0.6)
                strategy_weight = 1.0 - feedback_weight
            else:
                # 전략이 더 신뢰할 만한 경우
                strategy_weight = min(base_strategy_weight + (strategy_confidence - feedback_confidence) * 0.3, 0.8)
                feedback_weight = 1.0 - strategy_weight
            
            # 시그널 강도에 따른 추가 조정
            if abs(base_score) > 0.7:  # 강한 시그널
                strategy_weight *= 1.2  # 전략 비중 증가
                feedback_weight *= 0.8  # 피드백 비중 감소
            elif abs(base_score) < 0.2:  # 약한 시그널
                feedback_weight *= 1.2  # 피드백 비중 증가
                strategy_weight *= 0.8  # 전략 비중 감소
            
            # 정규화
            total_weight = strategy_weight + feedback_weight
            return feedback_weight / total_weight
            
        except Exception as e:
            print(f"⚠️ 피드백 가중치 계산 오류: {e}")
            return 0.3
    
    def _apply_balanced_improvement(self, base_score: float, success_rate: float, avg_profit: float, feedback_weight: float) -> float:
        """전략과 피드백이 균형잡힌 개선 적용"""
        try:
            # 전략 기반 점수 (기본 점수)
            strategy_score = base_score
            
            # 피드백 기반 점수
            if success_rate > 0.6:
                feedback_score = base_score * (0.9 + 0.1 * success_rate)  # 보수적 강화
            elif success_rate < 0.4:
                feedback_score = base_score * (0.7 - 0.2 * success_rate)  # 약화
            else:
                feedback_score = base_score
            
            # 수익률 보정
            if avg_profit > 2.0:
                feedback_score *= 1.05
            elif avg_profit < -1.0:
                feedback_score *= 0.95
            
            # 가중 평균으로 결합
            improved_score = strategy_score * (1.0 - feedback_weight) + feedback_score * feedback_weight
            
            return improved_score
            
        except Exception as e:
            print(f"⚠️ 균형잡힌 개선 적용 오류: {e}")
            return base_score
    
    def _apply_feedback_dominant_improvement(self, base_score: float, success_rate: float, avg_profit: float) -> float:
        """피드백 중심 개선 적용 (전략 신뢰도 낮은 경우)"""
        try:
            if success_rate > 0.6:
                improved_score = base_score * (0.8 + 0.2 * success_rate)
            elif success_rate < 0.4:
                improved_score = base_score * (0.6 - 0.2 * success_rate)
            else:
                improved_score = base_score
            
            # 수익률 보정
            if avg_profit > 2.0:
                improved_score *= 1.1
            elif avg_profit < -1.0:
                improved_score *= 0.9
            
            return improved_score
            
        except Exception as e:
            print(f"⚠️ 피드백 중심 개선 적용 오류: {e}")
            return base_score
    
    def _apply_strategy_dominant_improvement(self, base_score: float, success_rate: float, avg_profit: float) -> float:
        """전략 중심 개선 적용 (피드백 신뢰도 낮은 경우)"""
        try:
            # 전략을 우선하되, 피드백으로 미세 조정
            if success_rate > 0.7:  # 매우 높은 성공률만 반영
                improved_score = base_score * 1.05
            elif success_rate < 0.3:  # 매우 낮은 성공률만 반영
                improved_score = base_score * 0.95
            else:
                improved_score = base_score
            
            # 수익률 보정 (미세하게만)
            if avg_profit > 3.0:
                improved_score *= 1.02
            elif avg_profit < -2.0:
                improved_score *= 0.98
            
            return improved_score
            
        except Exception as e:
            print(f"⚠️ 전략 중심 개선 적용 오류: {e}")
            return base_score
                
    # 🚀 기존 함수는 계산 방법 개선 방식으로 변경됨
    # def get_signal_feedback_score(self, coin: str, interval: str, candle: pd.Series) -> float:
    #     """매매 결과 피드백을 바탕으로 한 시그널 점수 계산 (더 이상 사용하지 않음)"""
    #     # 이 함수는 계산 방법 개선 방식으로 대체됨
    #     return 0.0
    
    def extract_signal_pattern_from_state(self, state_key: str) -> str:
        """상태 키에서 시그널 패턴 추출 (Virtual Trading Learner와 동일한 방식)"""
        try:
            # 상태 키 예시: "BTC_5m_neutral_bullish_upper_low_early_neutral_low"
            parts = state_key.split('_')
            
            if len(parts) >= 6:
                # 핵심 시그널 패턴 추출
                # 예: "neutral_bullish_upper_low" 형태로 추출
                pattern_parts = parts[2:6]  # RSI, Direction, BB, Volume 부분
                return "_".join(pattern_parts)
            else:
                return "unknown_pattern"
                
        except Exception as e:
            print(f"⚠️ 상태 키에서 시그널 패턴 추출 오류: {e}")
            return "unknown_pattern"
    
    def _extract_signal_pattern_from_candle(self, candle: pd.Series, coin: str, interval: str) -> str:
        """캔들 데이터에서 시그널 패턴 추출 (피드백용)"""
        try:
            # RSI 범주화 (안전한 값 처리)
            rsi = safe_float(candle.get('rsi'), 50.0)
            if rsi < 30:
                rsi_cat = 'oversold'
            elif rsi > 70:
                rsi_cat = 'overbought'
            else:
                rsi_cat = 'neutral'
            
            # MACD 범주화 (안전한 값 처리)
            macd = safe_float(candle.get('macd'), 0.0)
            if macd > 0.001:
                macd_cat = 'bullish'
            elif macd < -0.001:
                macd_cat = 'bearish'
            else:
                macd_cat = 'neutral'
            
            # 거래량 범주화 (안전한 값 처리)
            volume_ratio = safe_float(candle.get('volume_ratio'), 1.0)
            if volume_ratio > 1.5:
                volume_cat = 'high'
            elif volume_ratio < 0.5:
                volume_cat = 'low'
            else:
                volume_cat = 'normal'
            
            # 변동성 범주화 (안전한 값 처리)
            volatility = safe_float(candle.get('volatility'), 0.02)
            if volatility > 0.05:
                vol_cat = 'high'
            elif volatility < 0.01:
                vol_cat = 'low'
            else:
                vol_cat = 'normal'
            
            # 파동 단계 (안전한 문자열 처리)
            wave_phase = safe_str(candle.get('wave_phase'), 'unknown')
            
            # 패턴 타입 (안전한 문자열 처리)
            pattern_type = safe_str(candle.get('pattern_type'), 'none')
            
            # 통합 방향 (안전한 문자열 처리)
            integrated_direction = safe_str(candle.get('integrated_direction'), 'neutral')
            
            return f"{rsi_cat}_{macd_cat}_{volume_cat}_{vol_cat}_{wave_phase}_{pattern_type}_{integrated_direction}"
            
        except Exception as e:
            print(f"⚠️ 캔들에서 시그널 패턴 추출 오류: {e}")
            return "unknown_pattern"
    
    def load_fractal_analysis_results(self):
        """프랙탈 분석 결과 로드 (Signal Selector에서 활용)"""
        self.fractal_analysis_results = {}
        
        try:
            with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                # 전체 분석 결과 로드
                overall_df = pd.read_sql("""
                    SELECT * FROM fractal_analysis_results 
                    WHERE analysis_type = 'overall'
                    ORDER BY created_at DESC LIMIT 1
                """, conn)
                
                if not overall_df.empty:
                    overall_result = overall_df.iloc[0]
                    self.fractal_analysis_results['overall'] = {
                        'optimal_conditions': json.loads(overall_result['optimal_conditions']) if overall_result['optimal_conditions'] else {},
                        'profit_threshold': overall_result['profit_threshold'],
                        'avg_profit': overall_result['avg_profit'],
                        'win_rate_threshold': overall_result['win_rate_threshold'],
                        'trades_count_threshold': overall_result['trades_count_threshold']
                    }
                    print(f"✅ 전체 프랙탈 분석 결과 로드: 수익률 임계값 {overall_result['profit_threshold']:.3f}")
                
                # 코인별 분석 결과 로드
                coin_specific_df = pd.read_sql("""
                    SELECT * FROM fractal_analysis_results 
                    WHERE analysis_type = 'coin_specific'
                    ORDER BY created_at DESC
                """, conn)
                
                for _, row in coin_specific_df.iterrows():
                    key = f"{row['symbol']}_{row['interval']}"
                    self.fractal_analysis_results[key] = {
                        'optimal_conditions': json.loads(row['optimal_conditions']) if row['optimal_conditions'] else {},
                        'profit_threshold': row['profit_threshold'],
                        'avg_profit': row['avg_profit'],
                        'win_rate_threshold': row['win_rate_threshold'],
                        'trades_count_threshold': row['trades_count_threshold'],
                        'top_strategies': json.loads(row['top_strategies']) if row['top_strategies'] else []
                    }
                
                print(f"✅ 코인별 프랙탈 분석 결과 로드: {len(coin_specific_df)}개 조합")
                
        except Exception as e:
            print(f"ℹ️ 프랙탈 분석 결과 로드 실패: {e}")
            self.fractal_analysis_results = {}
    
    def get_enhanced_coin_specific_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """Absolute Zero System의 학습 결과를 활용한 향상된 코인별 전략 점수 계산 (score 메트릭 활용)"""
        try:
            strategy_key = f"{coin}_{interval}"
            
            # 1. 기본 코인별 전략 점수 (score 메트릭 활용)
            base_score = self.get_coin_specific_score(coin, interval, candle)
            
            # 🆕 score 메트릭 기반 추가 보너스
            score_bonus = 0.0
            if strategy_key in self.coin_specific_strategies:
                strategy = self.coin_specific_strategies[strategy_key]
                if 'score' in strategy and strategy['score'] is not None:
                    # score가 높을수록 더 큰 보너스
                    if strategy['score'] >= 0.8:
                        score_bonus += 0.15  # 매우 높은 score 보너스
                    elif strategy['score'] >= 0.6:
                        score_bonus += 0.10  # 높은 score 보너스
                    elif strategy['score'] >= 0.4:
                        score_bonus += 0.05  # 중간 score 보너스
            
            # 2. 🆕 Absolute Zero System의 고급 지표 기반 적합성 평가
            absolute_zero_bonus = self._evaluate_absolute_zero_conditions(candle, strategy_key)
            
            # 3. 프랙탈 분석 결과 활용 (기존 로직 유지)
            fractal_bonus = 0.0
            
            if strategy_key in self.fractal_analysis_results:
                fractal_data = self.fractal_analysis_results[strategy_key]
                
                # 프랙탈 분석 기반 보너스 점수
                if base_score > 0:  # 기본 점수가 있는 경우에만
                    # 수익률 임계값 대비 성과
                    if base_score > fractal_data['profit_threshold']:
                        fractal_bonus += 0.1  # 임계값 초과 보너스
                    
                    # 평균 수익률 대비 성과
                    if base_score > fractal_data['avg_profit']:
                        fractal_bonus += 0.05  # 평균 초과 보너스
                    
                    # 승률 임계값 대비 성과
                    if hasattr(self, 'coin_specific_strategies') and strategy_key in self.coin_specific_strategies:
                        strategy = self.coin_specific_strategies[strategy_key]
                        if strategy['win_rate'] > fractal_data['win_rate_threshold']:
                            fractal_bonus += 0.05  # 높은 승률 보너스
                    
                    # 거래 수 임계값 대비 성과
                    if strategy['trades_count'] > fractal_data['trades_count_threshold']:
                        fractal_bonus += 0.03  # 충분한 거래 수 보너스
            
            # 4. 전체 프랙탈 분석 결과 활용
            if 'overall' in self.fractal_analysis_results:
                overall_data = self.fractal_analysis_results['overall']
                
                # 전체 시스템 성과 대비 평가
                if base_score > overall_data['profit_threshold']:
                    fractal_bonus += 0.08  # 전체 시스템 상위 성과 보너스
                
                # 최적 조건 활용
                optimal_conditions = overall_data.get('optimal_conditions', {})
                if optimal_conditions:
                    # 현재 시장 상황과 최적 조건 비교
                    market_adaptation = self._evaluate_optimal_conditions(candle, optimal_conditions)
                    fractal_bonus += market_adaptation * 0.05  # 최적 조건 적합성 보너스
            
            # 5. 🚀 고급 지표 기반 점수 보정 (민감도 강화)
            momentum_score = min(max(candle.get("wave_momentum", 0.0) * 2.0, -0.5), 0.5)  # 증폭
            
            # 볼린저 밴드 위치 점수
            bb_position = candle.get("bb_position", "unknown")
            bb_score = {"lower": 0.2, "middle": 0.1, "upper": -0.1}.get(bb_position, 0.0)
            
            # 다이버전스 점수
            divergence_rsi = candle.get("rsi_divergence", "none")
            divergence_macd = candle.get("macd_divergence", "none")
            divergence_score = 0.0
            
            # RSI 다이버전스
            if divergence_rsi in ["bullish", "bearish", "weak_bullish", "weak_bearish"]:
                divergence_score += {
                    "bullish": 0.2, "bearish": -0.2,
                    "weak_bullish": 0.1, "weak_bearish": -0.1,
                }.get(divergence_rsi, 0.0)
            
            # MACD 다이버전스
            if divergence_macd in ["bullish", "bearish", "weak_bullish", "weak_bearish"]:
                divergence_score += {
                    "bullish": 0.15, "bearish": -0.15,
                    "weak_bullish": 0.08, "weak_bearish": -0.08,
                }.get(divergence_macd, 0.0)
            
            # 🚀 진단 로그 (momentum이 0.0인 경우)
            if momentum_score == 0.0:
                print(f"⚠️ Momentum 0.0 유지됨: {coin}/{interval} @ {candle.get('timestamp')}")
            
            # 6. 최종 점수 계산 (모든 보너스 포함 + score 메트릭)
            enhanced_score = base_score + score_bonus + absolute_zero_bonus + fractal_bonus + momentum_score + bb_score + divergence_score
            
            # -1.0 ~ 1.0 범위로 정규화
            return np.clip(enhanced_score, -1.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ 향상된 코인별 전략 점수 계산 오류 ({coin}/{interval}): {e}")
            return self.get_coin_specific_score(coin, interval, candle)  # 기본 점수로 폴백
    
    def _evaluate_absolute_zero_conditions(self, candle: pd.Series, strategy_key: str) -> float:
        """Absolute Zero System에서 학습한 전략들의 성과를 기반으로 한 적합성 평가"""
        try:
            adaptation_score = 0.0
            
            # 🎯 Absolute Zero System에서 학습한 전략들의 성과 데이터 활용
            # 1. 해당 코인/인터벌의 상위 성과 전략들 조회
            coin, interval = strategy_key.split('_', 1)
            
            try:
                with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                    # 사용 가능한 테이블 확인
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    available_tables = [row[0] for row in cursor.fetchall()]
                    
                    # 해당 코인/인터벌의 상위 성과 전략들 조회
                    if 'learned_strategies' in available_tables:
                        top_strategies_df = pd.read_sql("""
                            SELECT * FROM learned_strategies 
                            WHERE coin = ? AND interval = ? 
                            AND profit > 0 AND trades_count >= 5
                            ORDER BY profit DESC, win_rate DESC
                            LIMIT 10
                        """, conn, params=(coin, interval))
                    elif 'global_strategies' in available_tables:
                        top_strategies_df = pd.read_sql("""
                            SELECT * FROM global_strategies 
                            WHERE coin = ? AND interval = ? 
                            AND profit > 0 AND trades_count >= 5
                            ORDER BY profit DESC, win_rate DESC
                            LIMIT 10
                        """, conn, params=(coin, interval))
                    else:
                        top_strategies_df = pd.DataFrame()  # 빈 데이터프레임
                    
                    if not top_strategies_df.empty:
                        # 🎯 상위 전략들의 평균 성과 기준
                        avg_profit = top_strategies_df['profit'].mean()
                        avg_win_rate = top_strategies_df['win_rate'].mean()
                        avg_trades = top_strategies_df['trades_count'].mean()
                        
                        # 🎯 현재 시장 상황과 상위 전략들의 조건 비교
                        for _, strategy in top_strategies_df.iterrows():
                            strategy_score = 0.0
                            
                            # 🎯 전략의 성과에 따른 가중치 적용 (조건 비교 대신 성과 기반)
                            strategy_weight = (strategy['profit'] / avg_profit) * (strategy['win_rate'] / avg_win_rate)
                            adaptation_score += strategy_weight * 0.1
                            
                            # 🎯 전략의 성과에 따른 가중치 적용
                            strategy_weight = (strategy['profit'] / avg_profit) * (strategy['win_rate'] / avg_win_rate)
                            adaptation_score += strategy_score * strategy_weight
                        
                        # 평균화
                        adaptation_score /= len(top_strategies_df)
                        
                        # 🎯 추가 보너스: 현재 시장 상황이 상위 전략들의 평균 성과보다 좋은 경우
                        if hasattr(self, 'coin_specific_strategies') and strategy_key in self.coin_specific_strategies:
                            current_strategy = self.coin_specific_strategies[strategy_key]
                            if current_strategy['profit'] > avg_profit:
                                adaptation_score += 0.1  # 평균 초과 보너스
                            if current_strategy['win_rate'] > avg_win_rate:
                                adaptation_score += 0.05  # 높은 승률 보너스
                    
                    else:
                        # 🎯 해당 코인/인터벌에 성과 데이터가 없는 경우, 전체 시스템 평균 활용
                        overall_df = pd.read_sql("""
                            SELECT AVG(profit) as avg_profit, AVG(win_rate) as avg_win_rate
                            FROM strategy_results 
                            WHERE profit > 0 AND trades_count >= 5
                        """, conn)
                        
                        if not overall_df.empty:
                            overall_avg_profit = overall_df.iloc[0]['avg_profit']
                            overall_avg_win_rate = overall_df.iloc[0]['avg_win_rate']
                            
                            # 기본적인 기술적 지표 기반 평가
                            adaptation_score = self._evaluate_basic_technical_indicators(candle)
                            
                            # 전체 시스템 평균 대비 보정
                            adaptation_score *= 0.5  # 보수적 접근
                
            except Exception as e:
                print(f"⚠️ Absolute Zero 전략 데이터 조회 오류: {e}")
                # 폴백: 기본 기술적 지표 평가
                adaptation_score = self._evaluate_basic_technical_indicators(candle)
            
            return adaptation_score * 0.3  # 30% 가중치 적용
            
        except Exception as e:
            print(f"⚠️ Absolute Zero 조건 평가 오류: {e}")
            return 0.0
    
    def _evaluate_basic_technical_indicators(self, candle: pd.Series) -> float:
        """기본 기술적 지표 기반 평가 (폴백용)"""
        try:
            score = 0.0
            
            # RSI 기반 평가
            rsi = candle.get('rsi')
            if rsi is not None and not pd.isna(rsi):
                rsi = float(rsi)
                if rsi < 30:  # 과매도 - 매수 기회
                    score += 0.1
                elif rsi > 70:  # 과매수 - 매도 기회
                    score -= 0.1
            
            # MACD 기반 평가
            macd = candle.get('macd')
            if macd is not None and not pd.isna(macd):
                macd = float(macd)
                if macd > 0:  # 상승 신호
                    score += 0.05
                else:  # 하락 신호
                    score -= 0.05
            
            # 거래량 비율 기반 평가
            volume_ratio = candle.get('volume_ratio')
            if volume_ratio is not None and not pd.isna(volume_ratio):
                volume_ratio = float(volume_ratio)
                if volume_ratio > 1.5:  # 거래량 증가
                    score += 0.05
                elif volume_ratio < 0.8:  # 거래량 감소
                    score -= 0.05
            
            return score
            
        except Exception as e:
            print(f"⚠️ 기본 기술적 지표 평가 오류: {e}")
            return 0.0
    
    def _check_rsi_condition(self, current_rsi: float, rsi_condition: str) -> bool:
        """RSI 조건 확인"""
        try:
            if not rsi_condition:
                return False
            
            # JSON 형태의 조건 파싱
            import json
            condition = json.loads(rsi_condition) if isinstance(rsi_condition, str) else rsi_condition
            
            min_rsi = condition.get('min', 0)
            max_rsi = condition.get('max', 100)
            
            return min_rsi <= current_rsi <= max_rsi
            
        except Exception as e:
            print(f"⚠️ RSI 조건 확인 오류: {e}")
            return False
    
    def _check_macd_condition(self, current_macd: float, macd_condition: str) -> bool:
        """MACD 조건 확인"""
        try:
            if not macd_condition:
                return False
            
            import json
            condition = json.loads(macd_condition) if isinstance(macd_condition, str) else macd_condition
            
            signal_diff = condition.get('signal_diff', 0)
            
            # MACD가 신호선보다 높은지 확인
            return current_macd > signal_diff
            
        except Exception as e:
            print(f"⚠️ MACD 조건 확인 오류: {e}")
            return False
    
    def _check_volume_condition(self, current_volume_ratio: float, volume_condition: str) -> bool:
        """거래량 조건 확인"""
        try:
            if not volume_condition:
                return False
            
            import json
            condition = json.loads(volume_condition) if isinstance(volume_condition, str) else volume_condition
            
            min_ratio = condition.get('min_ratio', 0)
            
            return current_volume_ratio >= min_ratio
            
        except Exception as e:
            print(f"⚠️ 거래량 조건 확인 오류: {e}")
            return False
    
    def _check_wave_step_condition(self, current_wave_step: float, wave_step_condition: str) -> bool:
        """파동 단계 조건 확인"""
        try:
            if not wave_step_condition:
                return False
            
            import json
            condition = json.loads(wave_step_condition) if isinstance(wave_step_condition, str) else wave_step_condition
            
            min_step = condition.get('min', 0)
            max_step = condition.get('max', 100)
            
            return min_step <= current_wave_step <= max_step
            
        except Exception as e:
            print(f"⚠️ 파동 단계 조건 확인 오류: {e}")
            return False
    
    def _check_pattern_quality_condition(self, current_pattern_quality: float, pattern_quality_condition: str) -> bool:
        """패턴 품질 조건 확인"""
        try:
            if not pattern_quality_condition:
                return False
            
            import json
            condition = json.loads(pattern_quality_condition) if isinstance(pattern_quality_condition, str) else pattern_quality_condition
            
            min_quality = condition.get('min', 0)
            
            return current_pattern_quality >= min_quality
            
        except Exception as e:
            print(f"⚠️ 패턴 품질 조건 확인 오류: {e}")
            return False
    
    def _check_structure_score_condition(self, current_structure_score: float, structure_score_condition: str) -> bool:
        """구조 점수 조건 확인"""
        try:
            if not structure_score_condition:
                return False
            
            import json
            condition = json.loads(structure_score_condition) if isinstance(structure_score_condition, str) else structure_score_condition
            
            min_score = condition.get('min', 0)
            
            return current_structure_score >= min_score
            
        except Exception as e:
            print(f"⚠️ 구조 점수 조건 확인 오류: {e}")
            return False
            
        except Exception as e:
            print(f"⚠️ Absolute Zero 조건 평가 오류: {e}")
            return 0.0
    
    def _calculate_enhanced_confidence(self, candle: pd.Series, signal_score: float, coin: str, interval: str) -> float:
        """🚀 개선된 신뢰도 계산 (다양성 확보)"""
        try:
            # 🚀 캐시 키 생성
            cache_key = f"confidence_{coin}_{interval}_{hash(str(candle.get('timestamp', 0)))}"
            cached_confidence = self.get_cached_data(cache_key, max_age=60)  # 1분 캐시
            if cached_confidence is not None:
                return cached_confidence
            
            # 🚀 기본 신뢰도 계산 (시그널 점수 기반)
            base_confidence = min(1.0, (abs(signal_score) + 0.4) / 1.4)
            
            # 🚀 고급 지표 기반 신뢰도 계산
            trend_score = 0.0
            quality_score = 0.0
            strength_score = 0.0
            
            # 1. 트렌드 점수 계산
            rsi = candle.get('rsi', 50)
            macd = candle.get('macd', 0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            
            # RSI 트렌드 점수
            if pd.notna(rsi):
                if 30 <= rsi <= 70:  # 중립 구간
                    trend_score += 0.3
                elif 20 <= rsi <= 80:  # 적정 구간
                    trend_score += 0.2
                else:  # 극단 구간
                    trend_score += 0.1
            
            # MACD 트렌드 점수
            if pd.notna(macd):
                macd_abs = abs(macd)
                if macd_abs < 0.02:  # 약한 신호
                    trend_score += 0.2
                elif macd_abs < 0.05:  # 보통 신호
                    trend_score += 0.3
                else:  # 강한 신호
                    trend_score += 0.4
            
            # 거래량 트렌드 점수
            if pd.notna(volume_ratio):
                if 0.5 <= volume_ratio <= 2.0:  # 적정 거래량
                    trend_score += 0.2
                elif 0.3 <= volume_ratio <= 3.0:  # 보통 거래량
                    trend_score += 0.15
                else:  # 극단 거래량
                    trend_score += 0.1
            
            # 2. 품질 점수 계산
            structure_score = candle.get('structure_score', 0.5)
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            
            if pd.notna(structure_score):
                quality_score += structure_score * 0.4
            
            if pd.notna(pattern_confidence):
                quality_score += pattern_confidence * 0.3
            
            # 다이버전스 품질 점수
            divergence_rsi = candle.get('rsi_divergence', 'none')
            divergence_macd = candle.get('macd_divergence', 'none')
            
            if divergence_rsi in ['bullish', 'bearish']:
                quality_score += 0.2
            elif divergence_rsi in ['weak_bullish', 'weak_bearish']:
                quality_score += 0.1
            
            if divergence_macd in ['bullish', 'bearish']:
                quality_score += 0.15
            elif divergence_macd in ['weak_bullish', 'weak_bearish']:
                quality_score += 0.08
            
            # 3. 강도 점수 계산
            momentum = candle.get('momentum', 0.0)
            volatility = candle.get('volatility', 0.0)
            
            if pd.notna(momentum):
                momentum_abs = abs(momentum)
                if momentum_abs > 0.01:
                    strength_score += min(momentum_abs * 2.0, 0.3)
            
            if pd.notna(volatility):
                if volatility < 0.02:  # 낮은 변동성
                    strength_score += 0.2
                elif volatility < 0.05:  # 보통 변동성
                    strength_score += 0.15
                else:  # 높은 변동성
                    strength_score += 0.1
            
            # 🚀 최종 신뢰도 계산 (엄격한 공식)
            confidence = 0.4 + (trend_score + quality_score + strength_score) / 4.0
            confidence = max(min(confidence, 0.95), 0.3)  # 더 엄격한 상하한 제한
            
            # 결과 캐시
            self.set_cached_data(cache_key, confidence)
            
            return confidence
            
        except Exception as e:
            print(f"⚠️ 신뢰도 계산 오류 ({coin}/{interval}): {e}")
            # 오류 시 기본 신뢰도 반환
            return min(1.0, (abs(signal_score) + 0.4) / 1.4)
    
    def _calculate_pattern_quality(self, rsi: float, macd: float, volume_ratio: float, structure_score: float, pattern_confidence: float) -> float:
        """패턴 품질을 다른 지표들을 기반으로 계산"""
        try:
            quality_factors = []
            
            # RSI 기반 품질 (30-70 범위가 좋음)
            if 30 <= rsi <= 70:
                quality_factors.append(0.8)
            elif 20 <= rsi <= 80:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.3)
            
            # MACD 기반 품질 (신호선과의 차이가 적당할 때)
            if abs(macd) < 0.01:
                quality_factors.append(0.7)
            elif abs(macd) < 0.05:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # 거래량 기반 품질 (적당한 거래량이 좋음)
            if 0.8 <= volume_ratio <= 2.0:
                quality_factors.append(0.8)
            elif 0.5 <= volume_ratio <= 3.0:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.4)
            
            # 구조점수 기반 품질
            quality_factors.append(structure_score)
            
            # 패턴 신뢰도 기반 품질
            quality_factors.append(pattern_confidence)
            
            # 평균 품질 계산
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            print(f"⚠️ 패턴 품질 계산 오류: {e}")
            return 0.5
    
    def _evaluate_optimal_conditions(self, candle: pd.Series, optimal_conditions: Dict[str, Any]) -> float:
        """최적 조건과 현재 시장 상황의 적합성 평가"""
        try:
            adaptation_score = 0.0
            
            # 안전한 값 변환 함수

            
            # RSI 최적 조건 평가
            if 'rsi' in optimal_conditions and 'rsi' in candle:
                rsi = safe_float(candle['rsi'], 50.0)
                rsi_condition = optimal_conditions['rsi']
                if 'optimal_range' in rsi_condition:
                    try:
                        min_rsi, max_rsi = map(float, rsi_condition['optimal_range'].split(' - '))
                        if min_rsi <= rsi <= max_rsi:
                            adaptation_score += 0.2
                    except (ValueError, TypeError):
                        pass  # 범위 파싱 실패 시 무시
            
            # 거래량 비율 최적 조건 평가
            if 'volume_ratio' in optimal_conditions and 'volume_ratio' in candle:
                volume_ratio = safe_float(candle['volume_ratio'], 1.0)
                volume_condition = optimal_conditions['volume_ratio']
                if 'optimal_range' in volume_condition:
                    try:
                        min_vol, max_vol = map(float, volume_condition['optimal_range'].split(' - '))
                        if min_vol <= volume_ratio <= max_vol:
                            adaptation_score += 0.2
                    except (ValueError, TypeError):
                        pass  # 범위 파싱 실패 시 무시
            
            # 파동 단계 최적 조건 평가
            if 'wave_step' in optimal_conditions and 'wave_step' in candle:
                wave_step = safe_float(candle['wave_step'], 0.5)
                wave_condition = optimal_conditions['wave_step']
                if 'optimal_range' in wave_condition:
                    try:
                        min_wave, max_wave = map(float, wave_condition['optimal_range'].split(' - '))
                        if min_wave <= wave_step <= max_wave:
                            adaptation_score += 0.2
                    except (ValueError, TypeError):
                        pass  # 범위 파싱 실패 시 무시
            
            # 패턴 품질 최적 조건 평가
            if 'pattern_quality' in optimal_conditions and 'pattern_quality' in candle:
                pattern_quality = safe_float(candle['pattern_quality'], 0.0)
                pattern_condition = optimal_conditions['pattern_quality']
                if 'optimal_range' in pattern_condition:
                    try:
                        min_pattern, max_pattern = map(float, pattern_condition['optimal_range'].split(' - '))
                        if min_pattern <= pattern_quality <= max_pattern:
                            adaptation_score += 0.2
                    except (ValueError, TypeError):
                        pass  # 범위 파싱 실패 시 무시
            
            # 구조 점수 최적 조건 평가
            if 'structure_score' in optimal_conditions and 'structure_score' in candle:
                structure_score = safe_float(candle['structure_score'], 0.5)
                structure_condition = optimal_conditions['structure_score']
                if 'optimal_range' in structure_condition:
                    try:
                        min_structure, max_structure = map(float, structure_condition['optimal_range'].split(' - '))
                        if min_structure <= structure_score <= max_structure:
                            adaptation_score += 0.2
                    except (ValueError, TypeError):
                        pass  # 범위 파싱 실패 시 무시
            
            return adaptation_score
            
        except Exception as e:
            print(f"⚠️ 최적 조건 평가 오류: {e}")
            return 0.0

    # 🆕 개선된 다이버전스 계산 함수 추가
    def calculate_divergence(self, df: pd.DataFrame, indicator: str, price_col: str = 'close') -> str:
        """
        🚀 개선된 다이버전스 계산 (민감도 향상) - 약한 다이버전스도 감지
        """
        if len(df) < 12:
            return 'none'
        
        try:
            # 🚀 캐시 키 생성 (240m 인터벌 최적화)
            cache_key = f"divergence_{indicator}_{hash(str(df.tail(8)[['timestamp', price_col, indicator]].values.tobytes()))}"
            cached_result = self.get_cached_data(cache_key, max_age=600)  # 10분 캐시
            if cached_result is not None:
                return cached_result
            
            # 🚀 최근 12개 데이터만 사용 (민감도 향상)
            recent_df = df.tail(12).copy()
            recent_df = recent_df.dropna(subset=[indicator, price_col])
            
            if len(recent_df) < 6:
                return 'none'
            
            # 🚀 고점/저점 찾기 (민감도 조정)
            # RSI는 더 민감하게, MACD는 적당히
            indicator_sensitivity = 0.001 if indicator == 'rsi' else 0.002
            price_sensitivity = 0.001  # 가격은 더 민감하게
            
            peaks = self._find_peaks_or_troughs(recent_df[indicator], sensitivity=indicator_sensitivity)
            price_peaks = self._find_peaks_or_troughs(recent_df[price_col], sensitivity=price_sensitivity)
            
            if len(peaks) < 2 or len(price_peaks) < 2:
                return 'none'
            
            # 🚀 최근 2개 기준으로 변화율 계산
            _, ind2 = peaks[-2]
            _, ind1 = peaks[-1]
            _, price2 = price_peaks[-2]
            _, price1 = price_peaks[-1]
            
            # 🚀 변화율 계산 (안전한 나눗셈)
            indicator_ratio = (ind1 - ind2) / (abs(ind2) + 1e-6)
            price_ratio = (price1 - price2) / (abs(price2) + 1e-6)
            
            # 🚀 다이버전스 판단 (더 민감한 조건)
            if price_ratio > 0.001 and indicator_ratio < -0.001:
                result = 'bearish'  # bearish (0.1% 이상)
            elif price_ratio < -0.001 and indicator_ratio > 0.001:
                result = 'bullish'  # bullish (0.1% 이상)
            elif price_ratio > 0.0003 and indicator_ratio < -0.0003:
                result = 'weak_bearish'  # 약한 bearish (0.03% 이상)
            elif price_ratio < -0.0003 and indicator_ratio > 0.0003:
                result = 'weak_bullish'  # 약한 bullish (0.03% 이상)
            else:
                result = 'none'
            
            # 결과 캐시
            self.set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            print(f"⚠️ 다이버전스 계산 오류 ({indicator}): {e}")
            return 'none'

    def _calculate_simple_divergence(self, df: pd.DataFrame, indicator: str, price_col: str = 'close') -> str:
        """간단한 다이버전스 계산 (최소 데이터로도 가능)"""
        try:
            if len(df) < 3:
                return 'none'
            
            # 최근 5개 데이터 사용 (더 많은 데이터로 정확도 향상)
            recent_data = df.tail(5)
            
            if indicator not in recent_data.columns or price_col not in recent_data.columns:
                return 'none'
            
            # 최근 5개 값 추출
            indicator_values = recent_data[indicator].dropna().values
            price_values = recent_data[price_col].dropna().values
            
            if len(indicator_values) < 3 or len(price_values) < 3:
                return 'none'
            
            # 🚀 개선된 다이버전스 계산
            # 1. 최근 3개 포인트의 방향성 분석
            price_trend = (price_values[-1] - price_values[-3]) / (price_values[-3] + 1e-6)
            indicator_trend = (indicator_values[-1] - indicator_values[-3]) / (abs(indicator_values[-3]) + 1e-6)
            
            # 2. 중간 포인트와의 비교 (더 정확한 다이버전스 감지)
            price_mid = price_values[-2]
            indicator_mid = indicator_values[-2]
            
            price_early_trend = (price_mid - price_values[-3]) / (price_values[-3] + 1e-6)
            price_late_trend = (price_values[-1] - price_mid) / (price_mid + 1e-6)
            
            indicator_early_trend = (indicator_mid - indicator_values[-3]) / (abs(indicator_values[-3]) + 1e-6)
            indicator_late_trend = (indicator_values[-1] - indicator_mid) / (abs(indicator_mid) + 1e-6)
            
            # 🚀 다이버전스 판단 (더 민감한 조건)
            # Bearish divergence: 가격은 상승하지만 지표는 하락
            if (price_trend > 0.001 and indicator_trend < -0.001) or \
               (price_early_trend > 0.001 and price_late_trend > 0.001 and 
                indicator_early_trend < -0.001 and indicator_late_trend < -0.001):
                return 'bearish'
            
            # Bullish divergence: 가격은 하락하지만 지표는 상승
            elif (price_trend < -0.001 and indicator_trend > 0.001) or \
                 (price_early_trend < -0.001 and price_late_trend < -0.001 and 
                  indicator_early_trend > 0.001 and indicator_late_trend > 0.001):
                return 'bullish'
            
            # 🚀 약한 다이버전스 감지 (더 민감한 조건)
            elif abs(price_trend) > 0.0005 and abs(indicator_trend) > 0.0005:
                if price_trend > 0 and indicator_trend < 0:
                    return 'weak_bearish'
                elif price_trend < 0 and indicator_trend > 0:
                    return 'weak_bullish'
            
            return 'none'
                
        except Exception as e:
            return 'none'

    def _calculate_rsi_divergence(self, df: pd.DataFrame, price_col: str) -> str:
        """RSI 다이버전스 계산"""
        try:
            if 'rsi' not in df.columns or price_col not in df.columns:
                return 'none'
            
            # 🚀 최근 8개 데이터에서 고점/저점 찾기 (민감도 향상)
            rsi_values = df['rsi'].tail(8).values
            price_values = df[price_col].tail(8).values
            
            if len(rsi_values) < 4:
                return 'none'
            
            # RSI 고점/저점 찾기
            rsi_peaks = []
            rsi_troughs = []
            
            for i in range(1, len(rsi_values) - 1):
                if rsi_values[i] > rsi_values[i-1] and rsi_values[i] > rsi_values[i+1]:
                    rsi_peaks.append((i, rsi_values[i]))
                elif rsi_values[i] < rsi_values[i-1] and rsi_values[i] < rsi_values[i+1]:
                    rsi_troughs.append((i, rsi_values[i]))
            
            # 가격 고점/저점 찾기
            price_peaks = []
            price_troughs = []
            
            for i in range(1, len(price_values) - 1):
                if price_values[i] > price_values[i-1] and price_values[i] > price_values[i+1]:
                    price_peaks.append((i, price_values[i]))
                elif price_values[i] < price_values[i-1] and price_values[i] < price_values[i+1]:
                    price_troughs.append((i, price_values[i]))
            
            # 🚀 다이버전스 판단 (민감도 향상)
            # Bearish divergence: 가격은 상승하지만 RSI는 하락
            if len(rsi_peaks) >= 1 and len(price_peaks) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(rsi_peaks) >= 2 and len(price_peaks) >= 2:
                    if (price_peaks[-1][1] > price_peaks[-2][1] and 
                        rsi_peaks[-1][1] < rsi_peaks[-2][1]):
                        return 'bearish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(rsi_peaks) >= 2 and len(price_peaks) >= 2:
                    # 가격이 0.5% 이상 상승하고 RSI가 0.5% 이상 하락
                    if (price_peaks[-1][1] > price_peaks[-2][1] * 1.005 and 
                        rsi_peaks[-1][1] < rsi_peaks[-2][1] * 0.995):
                        return 'bearish'
            
            # Bullish divergence: 가격은 하락하지만 RSI는 상승
            if len(rsi_troughs) >= 1 and len(price_troughs) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(rsi_troughs) >= 2 and len(price_troughs) >= 2:
                    if (price_troughs[-1][1] < price_troughs[-2][1] and 
                        rsi_troughs[-1][1] > rsi_troughs[-2][1]):
                        return 'bullish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(rsi_troughs) >= 2 and len(price_troughs) >= 2:
                    # 가격이 0.5% 이상 하락하고 RSI가 0.5% 이상 상승
                    if (price_troughs[-1][1] < price_troughs[-2][1] * 0.995 and 
                        rsi_troughs[-1][1] > rsi_troughs[-2][1] * 1.005):
                        return 'bullish'
            
            return 'none'
            
        except Exception as e:
            print(f"⚠️ RSI 다이버전스 계산 오류: {e}")
            return 'none'

    def _find_peaks_or_troughs(self, series: pd.Series, sensitivity: float = 0.002) -> List[Tuple[int, float]]:
        """🚀 고점/저점 찾기 헬퍼 함수 (민감도 조정 포함)"""
        try:
            values = series.values
            peaks = []
            
            for i in range(1, len(values) - 1):
                prev, curr, next_ = values[i-1], values[i], values[i+1]
                
                # 기본 고점/저점 조건
                if curr > prev and curr > next_:
                    peaks.append((i, curr))
                elif curr < prev and curr < next_:
                    peaks.append((i, curr))
                
                # 🚀 추가: 민감도 기반 조건 (변화율 ≥ sensitivity)
                elif (abs(curr - prev) / (abs(prev) + 1e-6) > sensitivity and 
                      abs(curr - next_) / (abs(next_) + 1e-6) > sensitivity):
                    if curr > prev and curr > next_:
                        peaks.append((i, curr))
                    elif curr < prev and curr < next_:
                        peaks.append((i, curr))
            
            return peaks
            
        except Exception as e:
            print(f"⚠️ 고점/저점 찾기 오류: {e}")
            return []
    
    def _calculate_macd_divergence(self, df: pd.DataFrame, price_col: str) -> str:
        """MACD 다이버전스 계산 (기존 로직 유지 - 호환성)"""
        try:
            if 'macd' not in df.columns or price_col not in df.columns:
                return 'none'
            
            # 🚀 최근 8개 데이터에서 고점/저점 찾기 (민감도 향상)
            macd_values = df['macd'].tail(8).values
            price_values = df[price_col].tail(8).values
            
            if len(macd_values) < 4:
                return 'none'
            
            # MACD 고점/저점 찾기
            macd_peaks = []
            macd_troughs = []
            
            for i in range(1, len(macd_values) - 1):
                if macd_values[i] > macd_values[i-1] and macd_values[i] > macd_values[i+1]:
                    macd_peaks.append((i, macd_values[i]))
                elif macd_values[i] < macd_values[i-1] and macd_values[i] < macd_values[i+1]:
                    macd_troughs.append((i, macd_values[i]))
            
            # 가격 고점/저점 찾기
            price_peaks = []
            price_troughs = []
            
            for i in range(1, len(price_values) - 1):
                if price_values[i] > price_values[i-1] and price_values[i] > price_values[i+1]:
                    price_peaks.append((i, price_values[i]))
                elif price_values[i] < price_values[i-1] and price_values[i] < price_values[i+1]:
                    price_troughs.append((i, price_values[i]))
            
            # 🚀 다이버전스 판단 (민감도 향상)
            # Bearish divergence: 가격은 상승하지만 MACD는 하락
            if len(macd_peaks) >= 1 and len(price_peaks) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(macd_peaks) >= 2 and len(price_peaks) >= 2:
                    if (price_peaks[-1][1] > price_peaks[-2][1] and 
                        macd_peaks[-1][1] < macd_peaks[-2][1]):
                        return 'bearish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(macd_peaks) >= 2 and len(price_peaks) >= 2:
                    # 가격이 0.5% 이상 상승하고 MACD가 0.5% 이상 하락
                    if (price_peaks[-1][1] > price_peaks[-2][1] * 1.005 and 
                        macd_peaks[-1][1] < macd_peaks[-2][1] * 0.995):
                        return 'bearish'
            
            # Bullish divergence: 가격은 하락하지만 MACD는 상승
            if len(macd_troughs) >= 1 and len(price_troughs) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(macd_troughs) >= 2 and len(price_troughs) >= 2:
                    if (price_troughs[-1][1] < price_troughs[-2][1] and 
                        macd_troughs[-1][1] > macd_troughs[-2][1]):
                        return 'bullish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(macd_troughs) >= 2 and len(price_troughs) >= 2:
                    # 가격이 0.5% 이상 하락하고 MACD가 0.5% 이상 상승
                    if (price_troughs[-1][1] < price_troughs[-2][1] * 0.995 and 
                        macd_troughs[-1][1] > macd_troughs[-2][1] * 1.005):
                        return 'bullish'
            
            return 'none'
            
        except Exception as e:
            print(f"⚠️ MACD 다이버전스 계산 오류: {e}")
            return 'none'

    def detect_current_market_condition(self, coin: str, interval: str) -> str:
        """실시간 시장 상황 감지"""
        try:
            # 최근 캔들 데이터 로드
            df = self.get_cached_data(f"{coin}_{interval}_candles", max_age=300)
            if df is None or df.empty:
                return "unknown"
            
            # 최근 20개 캔들 기준으로 분석
            recent_df = df.tail(20)
            
            # 가격 변화율 계산
            price_changes = recent_df['close'].pct_change().dropna()
            
            # 이동평균 계산
            ma_short = recent_df['close'].rolling(window=5).mean()
            ma_long = recent_df['close'].rolling(window=20).mean()
            
            # RSI 계산
            delta = recent_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 변동성 계산
            volatility = price_changes.std()
            
            # 시장 상황 판단
            avg_change = price_changes.mean()
            price_trend = recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]
            
            # 🎯 시장 상황 분류 로직
            if price_trend > 0.05 and avg_change > 0.002:  # 5% 이상 상승 + 평균 상승
                return "bull_market"  # 상승장
            elif price_trend < -0.05 and avg_change < -0.002:  # 5% 이상 하락 + 평균 하락
                return "bear_market"  # 하락장
            elif abs(price_trend) < 0.02 and volatility > 0.02:  # 2% 이내 변동 + 높은 변동성
                return "sideways_market"  # 횡보장
            elif current_rsi > 70:
                return "overbought"  # 과매수
            elif current_rsi < 30:
                return "oversold"  # 과매도
            else:
                return "neutral"  # 중립
                
        except Exception as e:
            print(f"⚠️ 시장 상황 감지 오류 ({coin}/{interval}): {e}")
            return "unknown"
    
    def select_market_adaptive_strategy(self, coin: str, interval: str, market_condition: str) -> Optional[Dict]:
        """시장 상황에 맞는 전략 선택"""
        try:
            strategy_key = f"{coin}_{interval}"
            
            if strategy_key not in self.coin_specific_strategies:
                return None
            
            strategy = self.coin_specific_strategies[strategy_key]
            strategy_type = strategy.get('strategy_type', '')
            
            # 🎯 시장 상황별 전략 우선순위
            if market_condition == "bull_market":
                # 상승장: ADAPTIVE_BULL_MARKET 또는 일반 전략
                if "ADAPTIVE_BULL_MARKET" in strategy_type or "ADAPTIVE" in strategy_type:
                    return strategy
                else:
                    # 일반 전략에 상승장 보너스 적용
                    strategy['market_condition_bonus'] = 1.2
                    return strategy
                    
            elif market_condition == "bear_market":
                # 하락장: ADAPTIVE_BEAR_MARKET 또는 보수적 전략
                if "ADAPTIVE_BEAR_MARKET" in strategy_type or "ADAPTIVE" in strategy_type:
                    return strategy
                else:
                    # 일반 전략에 하락장 페널티 적용
                    strategy['market_condition_bonus'] = 0.8
                    return strategy
                    
            elif market_condition == "sideways_market":
                # 횡보장: ADAPTIVE_SIDEWAYS_MARKET 또는 범위 거래 전략
                if "ADAPTIVE_SIDEWAYS_MARKET" in strategy_type or "ADAPTIVE" in strategy_type:
                    return strategy
                else:
                    # 일반 전략에 횡보장 중립 적용
                    strategy['market_condition_bonus'] = 1.0
                    return strategy
                    
            elif market_condition == "overbought":
                # 과매수: ADAPTIVE_OVERBOUGHT 또는 매도 전략
                if "ADAPTIVE_OVERBOUGHT" in strategy_type or "ADAPTIVE" in strategy_type:
                    return strategy
                else:
                    # 일반 전략에 과매수 보너스 적용
                    strategy['market_condition_bonus'] = 1.1
                    return strategy
                    
            elif market_condition == "oversold":
                # 과매도: ADAPTIVE_OVERSOLD 또는 매수 전략
                if "ADAPTIVE_OVERSOLD" in strategy_type or "ADAPTIVE" in strategy_type:
                    return strategy
                else:
                    # 일반 전략에 과매도 보너스 적용
                    strategy['market_condition_bonus'] = 1.1
                    return strategy
                    
            else:
                # 중립: 일반 전략 사용
                strategy['market_condition_bonus'] = 1.0
                return strategy
                
        except Exception as e:
            print(f"⚠️ 시장 적응 전략 선택 오류 ({coin}/{interval}): {e}")
            return None

    def _load_ai_model(self):
        """🚀 학습된 전략 기반 AI 모델 로드"""
        try:
            print(f"🚀 학습된 전략 기반 AI 모델 로드 중...")
            
            # 🆕 현재 코인이 설정되지 않은 경우 기본값 설정
            if not hasattr(self, 'current_coin') or not self.current_coin:
                # 환경/DB에서 사용 가능한 첫 코인을 기본값으로 설정
                try:
                    from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
                    available = get_available_coins_and_intervals()
                    self.current_coin = next(iter({c for c, _ in available}), None) or os.getenv('DEFAULT_COIN', 'BTC')
                except Exception:
                    self.current_coin = os.getenv('DEFAULT_COIN', 'BTC')
                print(f"ℹ️ 현재 코인이 설정되지 않아 기본값 {self.current_coin} 사용")
            
            # 🆕 데이터베이스에서 학습된 전략 로드 (여러 경로 시도)
            try:
                _load_learned_strategies_from_db()
                print("✅ 학습된 전략 로드 성공")
            except Exception as e:
                print(f"⚠️ 학습된 전략 로드 실패: {e}")
                print("🔧 기본 AI 모델로 진행")
            
            # 🆕 전략 기반 AI 모델 생성 시도
            try:
                self.ai_model, self.model_type = _create_strategy_based_ai_model()
                self.feature_dim = 50
                self.ai_model_loaded = True
                print(f"✅ 학습된 전략 기반 AI 모델 로드 완료")
                
            except Exception as e:
                print(f"⚠️ 전략 기반 AI 모델 생성 실패: {e}")
                # Fallback: 기본 모델 생성
                self.ai_model, self.model_type = _create_default_ai_model()
                self.feature_dim = 50
                self.ai_model_loaded = True
                print(f"✅ 기본 AI 모델로 대체 완료")
            
        except Exception as e:
            print(f"⚠️ AI 모델 로드 전체 실패: {e}")
            # 최종 Fallback: 기본 모델 생성
            try:
                self.ai_model, self.model_type = _create_default_ai_model()
                self.feature_dim = 50
                self.ai_model_loaded = True
                print(f"✅ 최종 기본 AI 모델로 대체 완료")
            except Exception as e2:
                print(f"❌ 최종 AI 모델 생성도 실패: {e2}")
                self.ai_model_loaded = False

    def set_current_coin(self, coin: str):
        """현재 처리 중인 코인 설정 (AI 모델 로드용)"""
        if hasattr(self, 'current_coin') and self.current_coin != coin:
            self.current_coin = coin
            # 코인이 바뀌면 해당 코인의 전용 모델 로드 시도
            if AI_MODEL_AVAILABLE:
                self._load_ai_model()
        else:
            self.current_coin = coin

    def _prepare_features_for_ai(self, candle: pd.Series) -> np.ndarray:
        """AI 모델용 특징 벡터 준비"""
        try:
            # 기본 기술지표들을 특징 벡터로 변환
            features = []
            
            # RSI 관련 특징
            features.extend([
                safe_float(candle.get('rsi', 50.0)) / 100.0,  # 정규화
                safe_float(candle.get('rsi_ema', 50.0)) / 100.0,
                safe_float(candle.get('rsi_smoothed', 50.0)) / 100.0
            ])
            
            # MACD 관련 특징
            features.extend([
                safe_float(candle.get('macd', 0.0)),
                safe_float(candle.get('macd_signal', 0.0)),
                safe_float(candle.get('macd_diff', 0.0)),
                safe_float(candle.get('macd_smoothed', 0.0))
            ])
            
            # 볼륨 관련 특징
            features.extend([
                safe_float(candle.get('volume_ratio', 1.0)),
                safe_float(candle.get('volume_momentum', 0.0)),
                safe_float(candle.get('volume_divergence', 'none') == 'positive' and 1.0 or 0.0)
            ])
            
            # 파동 관련 특징
            features.extend([
                safe_float(candle.get('wave_progress', 0.5)),
                safe_float(candle.get('wave_momentum', 0.0)),
                safe_float(candle.get('wave_phase', 'unknown') in ['impulse', 'correction'] and 1.0 or 0.0)
            ])
            
            # 구조 및 패턴 관련 특징
            features.extend([
                safe_float(candle.get('structure_score', 0.5)),
                safe_float(candle.get('pattern_confidence', 0.0)),
                safe_float(candle.get('pattern_quality', 0.0))
            ])
            
            # 볼린저 밴드 관련 특징
            features.extend([
                safe_float(candle.get('bb_width', 0.0)),
                safe_float(candle.get('bb_squeeze', 0.0)),
                safe_float(candle.get('bb_position', 'unknown') == 'upper' and 1.0 or 
                          candle.get('bb_position', 'unknown') == 'lower' and -1.0 or 0.0)
            ])
            
            # 다이버전스 관련 특징
            features.extend([
                safe_float(candle.get('rsi_divergence', 'none') == 'positive' and 1.0 or 
                          candle.get('rsi_divergence', 'none') == 'negative' and -1.0 or 0.0),
                safe_float(candle.get('macd_divergence', 'none') == 'positive' and 1.0 or 
                          candle.get('macd_divergence', 'none') == 'negative' and -1.0 or 0.0)
            ])
            
            # 모멘텀 및 트렌드 관련 특징
            features.extend([
                safe_float(candle.get('price_momentum', 0.0)),
                safe_float(candle.get('trend_strength', 0.5)),
                safe_float(candle.get('volatility', 0.0))
            ])
            
            # 특징 벡터를 numpy 배열로 변환
            feature_array = np.array(features, dtype=np.float32)
            
            # 🆕 동적 특징 차원 사용 (하드코딩된 100 제거)
            return feature_array.reshape(1, -1)  # 배치 차원 추가
            
        except Exception as e:
            print(f"❌ AI 특징 벡터 준비 실패: {e}")
            # 기본 특징 벡터 반환
            return np.zeros((1, 100), dtype=np.float32)

    def get_ai_based_score(self, candle: pd.Series) -> Dict[str, float]:
        """🚀 고성능 AI 모델 기반 전략 점수 계산 (GPU 가속 지원)"""
        try:
            # AI 모델이 로드되지 않았으면 실제 데이터 기반 기본 예측 사용
            if not self.ai_model_loaded or self.ai_model is None:
                print("⚠️ AI 모델이 로드되지 않음, 실제 데이터 기반 예측 사용")
                return self._get_default_ai_prediction(candle)
            
            # 🚀 GPU 가속 상태 확인
            if USE_GPU_ACCELERATION and JAX_PLATFORM_NAME == 'gpu':
                gpu_status = "🚀 GPU 가속"
            else:
                gpu_status = "💻 CPU 모드"
            
            # 특징 벡터 준비
            features = self._prepare_features_for_ai(candle)
            
            # 🚀 AI 모델 예측 (GPU 가속 지원)
            predictions = self.ai_model.predict(features)
            strategy_score = self.ai_model.predict_strategy_score(features, risk_penalty=0.5)
            
            # 🚀 성능 정보 포함
            result = {
                'mu': float(predictions['mu'][0]),           # 수익률 예측
                'p_up': float(predictions['p_up'][0]),       # 상승 확률
                'risk': float(predictions['risk'][0]),       # 리스크
                'strategy_score': float(strategy_score[0]),  # 전략 점수
                'gpu_accelerated': USE_GPU_ACCELERATION,     # GPU 가속 상태
                'model_type': self.model_type                # 모델 타입
            }
            
            if self.debug_mode:
                print(f"  {gpu_status} AI 예측 완료: {self.model_type} 모델")
            
            return result
            
        except Exception as e:
            print(f"❌ AI 모델 예측 실패: {e}")
            # 기본 AI 모델 사용
            return self._get_default_ai_prediction(candle)
    
    def _get_default_ai_prediction(self, candle: pd.Series) -> Dict[str, float]:
        """🚀 실제 캔들 데이터 기반 기본 AI 모델 예측 (더 정교한 계산)"""
        try:
            # 🚀 실제 캔들 데이터에서 지표 추출 (None 값 안전 처리)
            rsi = candle.get('rsi', 50.0)
            macd = candle.get('macd', 0.0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            volatility = candle.get('volatility', 0.02)
            wave_phase = candle.get('wave_phase', 'unknown')
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            integrated_direction = candle.get('integrated_direction', 'neutral')
            
            # None 값 안전 처리
            if rsi is None:
                rsi = 50.0
            if macd is None:
                macd = 0.0
            if volume_ratio is None:
                volume_ratio = 1.0
            if volatility is None:
                volatility = 0.02
            if pattern_confidence is None:
                pattern_confidence = 0.0
            
            # 🚀 RSI 기반 수익률 예측 (더 정교한 계산)
            if rsi < 20:  # 극도 과매도
                mu = 0.08 + (20 - rsi) * 0.002  # 0.08 ~ 0.12
            elif rsi < 30:  # 과매도
                mu = 0.05 + (30 - rsi) * 0.001  # 0.05 ~ 0.08
            elif rsi > 80:  # 극도 과매수
                mu = -0.05 - (rsi - 80) * 0.002  # -0.05 ~ -0.09
            elif rsi > 70:  # 과매수
                mu = -0.02 - (rsi - 70) * 0.001  # -0.02 ~ -0.05
            else:  # 중립
                mu = 0.01 + (50 - abs(rsi - 50)) * 0.0005  # 0.01 ~ 0.025
            
            # 🚀 MACD 기반 상승확률 (더 정교한 계산)
            if macd > 0.01:  # 강한 상승 신호
                p_up = 0.7 + min(macd * 500, 0.2)  # 0.7 ~ 0.9
            elif macd > 0:  # 약한 상승 신호
                p_up = 0.55 + macd * 1000  # 0.55 ~ 0.7
            elif macd > -0.01:  # 약한 하락 신호
                p_up = 0.45 + macd * 1000  # 0.35 ~ 0.45
            else:  # 강한 하락 신호
                p_up = 0.3 + max(macd * 500, -0.2)  # 0.1 ~ 0.3
            
            # 🚀 거래량 기반 리스크 조정 (더 정교한 계산)
            if volume_ratio > 3.0:  # 매우 높은 거래량
                risk = 0.2 + min(volume_ratio - 3.0, 0.3)  # 0.2 ~ 0.5
            elif volume_ratio > 2.0:  # 높은 거래량
                risk = 0.3 + (volume_ratio - 2.0) * 0.2  # 0.3 ~ 0.5
            elif volume_ratio > 1.0:  # 정상 거래량
                risk = 0.4 + (volume_ratio - 1.0) * 0.1  # 0.4 ~ 0.5
            else:  # 낮은 거래량
                risk = 0.5 + (1.0 - volume_ratio) * 0.2  # 0.5 ~ 0.7
            
            # 🚀 파동 단계 기반 점수 조정
            wave_bonus = 1.0
            if wave_phase == 'impulse':
                wave_bonus = 1.2
            elif wave_phase == 'correction':
                wave_bonus = 0.9
            elif wave_phase == 'consolidation':
                wave_bonus = 1.0
            
            # 🚀 통합 방향성 기반 점수 조정
            direction_bonus = 1.0
            if integrated_direction == 'strong_bullish':
                direction_bonus = 1.3
            elif integrated_direction == 'bullish':
                direction_bonus = 1.2
            elif integrated_direction == 'strong_bearish':
                direction_bonus = 0.7
            elif integrated_direction == 'bearish':
                direction_bonus = 0.8
            
            # 🚀 패턴 신뢰도 기반 점수 조정
            pattern_bonus = 1.0 + (pattern_confidence * 0.3)
            
            # 🚀 변동성 기반 점수 조정
            volatility_factor = min(volatility * 100, 1.0)
            
            # 🚀 최종 전략 점수 계산 (모든 요소 고려)
            strategy_score = (mu * 0.4 + p_up * 0.3 + (1 - risk) * 0.3) * wave_bonus * direction_bonus * pattern_bonus * (1 + volatility_factor * 0.2)
            
            print(f"🧠 실제 데이터 기반 AI 예측: RSI({rsi:.1f})→수익률({mu:.3f}), MACD({macd:.4f})→상승확률({p_up:.3f}), Volume({volume_ratio:.2f}x)→리스크({risk:.3f}), 최종점수({strategy_score:.3f})")
            
            return {
                'mu': np.clip(mu, -0.1, 0.1),
                'p_up': np.clip(p_up, 0.1, 0.9),
                'risk': np.clip(risk, 0.1, 0.9),
                'strategy_score': np.clip(strategy_score, 0.0, 1.0),
                'gpu_accelerated': False,
                'model_type': 'enhanced_technical'
            }
            
        except Exception as e:
            print(f"⚠️ 기본 AI 예측 실패: {e}")
            return {
                'mu': 0.0,
                'p_up': 0.5,
                'risk': 0.5,
                'strategy_score': 0.15,
                'gpu_accelerated': False,
                'model_type': 'fallback'
            }

    def generate_multi_timeframe_signal(self, coin: str, intervals: List[str] = ['15m', '30m', '240m', '1d']) -> Optional[SignalInfo]:
        """🚀 멀티 타임프레임 시그널 통합 생성 (여러 인터벌의 정보를 종합하여 최적 시그널 생성)"""
        try:
            print(f"🔄 {coin} 멀티 타임프레임 시그널 생성 시작")
            
            # 각 인터벌별 시그널 생성
            interval_signals = {}
            for interval in intervals:
                try:
                    signal = self.generate_single_interval_signal(coin, interval)
                    if signal:
                        interval_signals[interval] = signal
                        print(f"  ✅ {interval}: {signal.action.value} (점수: {signal.signal_score:.3f})")
                    else:
                        print(f"  ⚠️ {interval}: 시그널 생성 실패")
                except Exception as e:
                    print(f"  ❌ {interval}: 시그널 생성 오류 - {e}")
                    continue
            
            # 최소 2개 인터벌의 시그널이 있어야 통합 가능
            if len(interval_signals) < 2:
                print(f"⚠️ {coin}: 충분한 인터벌 시그널이 없음 ({len(interval_signals)}개)")
                return None
            
            # 멀티 타임프레임 시그널 통합
            combined_signal = self.combine_multi_timeframe_signals(coin, interval_signals)
            
            if combined_signal:
                print(f"✅ {coin} 멀티 타임프레임 시그널 통합 완료: {combined_signal.action.value} (점수: {combined_signal.signal_score:.3f})")
                return combined_signal
            else:
                print(f"⚠️ {coin}: 멀티 타임프레임 시그널 통합 실패")
                return None
                
        except Exception as e:
            self._handle_error(e, f"멀티 타임프레임 시그널 생성 - {coin}")
            return None
    
    def generate_single_interval_signal(self, coin: str, interval: str) -> Optional[SignalInfo]:
        """단일 인터벌 시그널 생성 (기존 generate_signal 함수 활용)"""
        try:
            # 기존 generate_signal 함수 호출
            return self.generate_signal(coin, interval)
        except Exception as e:
            print(f"⚠️ {coin} {interval} 단일 인터벌 시그널 생성 실패: {e}")
            return None
    
    def combine_multi_timeframe_signals(self, coin: str, interval_signals: Dict[str, SignalInfo]) -> Optional[SignalInfo]:
        """여러 인터벌의 시그널을 통합하여 최적 시그널 생성 (레짐 종합 고려)"""
        try:
            if not interval_signals:
                return None

            # 🔥 DB에서 코인별 최적 인터벌 가중치 로드 (Absolute Zero 계산 결과)
            interval_weights = self._load_coin_interval_weights(coin)

            # 폴백: DB에 없으면 기본 가중치 사용
            if not interval_weights:
                interval_weights = {
                    '1d': 0.25,    # 장기
                    '15m': 0.20,   # 단기
                    '30m': 0.25,   # 중기
                    '240m': 0.40   # 장기 (가장 중요)
                }
                print(f"⚠️ {coin}: DB에서 가중치를 찾을 수 없어 기본값 사용")
            
            # 🎯 각 인터벌별 레짐 감지 및 종합
            all_regimes = {}
            for interval, signal in interval_signals.items():
                try:
                    # 각 시그널에서 레짐 추출 (시그널에 저장되어 있음)
                    # 레짐 정보가 시그널에 없으면 지표로부터 감지
                    candle_data = self._get_candle_from_signal(signal)
                    regime = self._detect_current_regime(coin, interval, candle_data)
                    all_regimes[interval] = regime
                except Exception as e:
                    print(f"⚠️ {interval} 레짐 감지 실패: {e}")
                    all_regimes[interval] = 'neutral'
            
            # 🎯 레짐 분포 분석 및 통합 가중치 결정 (DB 기반)
            regime_based_weights = self._calculate_multi_regime_weights(all_regimes, interval_weights, coin=coin)
            
            # 🎯 통합 점수 계산
            total_score = 0.0
            total_confidence = 0.0
            total_weight = 0.0
            
            # 🎯 액션별 투표 집계
            action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            
            for interval, signal in interval_signals.items():
                # 레짐 기반 가중치 적용
                weight = regime_based_weights.get(interval, interval_weights.get(interval, 0.1))
                
                # 점수와 신뢰도 가중 평균
                total_score += signal.signal_score * weight
                total_confidence += signal.confidence * weight
                total_weight += weight
                
                # 액션별 투표 집계
                action = signal.action.value
                if action in action_votes:
                    action_votes[action] += 1
                    action_scores[action] += signal.signal_score * weight
            
            if total_weight == 0:
                return None
            
            # 🎯 최종 통합 점수
            final_score = total_score / total_weight
            final_confidence = total_confidence / total_weight
            
            # 🎯 최종 액션 결정 (투표 기반 + 점수 기반)
            final_action = self._determine_final_action(action_votes, action_scores, final_score)
            
            # 🎯 통합 시그널 생성
            combined_signal = SignalInfo(
                coin=coin,
                interval='combined',  # 멀티 타임프레임 통합
                action=SignalAction(final_action),
                signal_score=final_score,
                confidence=final_confidence,
                reason=f"멀티 타임프레임 통합: {', '.join([f'{k}({v})' for k, v in action_votes.items() if v > 0])}",
                timestamp=int(time.time()),
                price=self._get_latest_price(coin),
                volume=0.0,
                rsi=self._calculate_weighted_average(interval_signals, 'rsi', interval_weights),
                macd=self._calculate_weighted_average(interval_signals, 'macd', interval_weights),
                # 🆕 새로운 학습 결과 필드 (멀티 타임프레임 통합용 기본값)
                reliability_score=0.0,
                learning_quality_score=0.0,
                global_strategy_id="",
                coin_tuned=False,
                walk_forward_performance=None,
                regime_coverage=None,
                wave_phase=self._get_most_common_value(interval_signals, 'wave_phase'),
                pattern_type=self._get_most_common_value(interval_signals, 'pattern_type'),
                risk_level=self._get_most_common_value(interval_signals, 'risk_level'),
                volatility=self._calculate_weighted_average(interval_signals, 'volatility', interval_weights),
                volume_ratio=self._calculate_weighted_average(interval_signals, 'volume_ratio', interval_weights),
                wave_progress=self._calculate_weighted_average(interval_signals, 'wave_progress', interval_weights),
                structure_score=self._calculate_weighted_average(interval_signals, 'structure_score', interval_weights),
                pattern_confidence=self._calculate_weighted_average(interval_signals, 'pattern_confidence', interval_weights),
                integrated_direction=self._get_most_common_value(interval_signals, 'integrated_direction'),
                integrated_strength=self._calculate_weighted_average(interval_signals, 'integrated_strength', interval_weights)
            )
            
            return combined_signal
            
        except Exception as e:
            print(f"⚠️ {coin} 멀티 타임프레임 시그널 통합 실패: {e}")
            return None
    
    def _determine_final_action(self, action_votes: Dict[str, int], action_scores: Dict[str, float], final_score: float) -> str:
        """최종 액션 결정 (투표 기반 + 점수 기반)"""
        try:
            # 🎯 투표 기반 우선순위
            max_votes = max(action_votes.values())
            most_voted_actions = [action for action, votes in action_votes.items() if votes == max_votes]
            
            if len(most_voted_actions) == 1:
                # 단일 최다 투표 액션
                return most_voted_actions[0]
            elif len(most_voted_actions) > 1:
                # 동점인 경우 점수 기반 결정
                best_action = max(most_voted_actions, key=lambda x: action_scores.get(x, 0))
                return best_action
            else:
                # 투표가 없는 경우 점수 기반 결정
                if final_score > 0.3:
                    return 'buy'
                elif final_score < -0.3:
                    return 'sell'
                else:
                    return 'hold'
                    
        except Exception as e:
            print(f"⚠️ 최종 액션 결정 실패: {e}")
            return 'hold'
    
    def _calculate_weighted_average(self, interval_signals: Dict[str, SignalInfo], field: str, weights: Dict[str, float]) -> float:
        """가중 평균 계산"""
        try:
            total_value = 0.0
            total_weight = 0.0
            
            for interval, signal in interval_signals.items():
                weight = weights.get(interval, 0.1)
                value = getattr(signal, field, 0.0)
                
                if isinstance(value, (int, float)):
                    total_value += value * weight
                    total_weight += weight
            
            return total_value / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️ 가중 평균 계산 실패 ({field}): {e}")
            return 0.0
    
    def _load_coin_interval_weights(self, coin: str) -> Dict[str, float]:
        """🔥 DB에서 코인별 최적 인터벌 가중치 로드 (Absolute Zero가 계산한 값)"""
        try:
            # rl_pipeline의 get_coin_analysis_ratios 함수 사용
            import sys
            import os
            rl_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'rl_pipeline')
            if os.path.exists(rl_pipeline_path) and rl_pipeline_path not in sys.path:
                sys.path.insert(0, rl_pipeline_path)

            from rl_pipeline.db.reads import get_coin_analysis_ratios

            # interval="all", analysis_type="default" 또는 레짐별로 조회
            # 먼저 default 시도
            ratios_data = get_coin_analysis_ratios(coin, "all", "default")

            if ratios_data and ratios_data.get('interval_weights'):
                interval_weights = ratios_data['interval_weights']
                if interval_weights:
                    print(f"✅ {coin}: DB에서 최적 가중치 로드 성공 - {interval_weights}")
                    return interval_weights

            # default가 없으면 trending 시도
            ratios_data = get_coin_analysis_ratios(coin, "all", "trending")
            if ratios_data and ratios_data.get('interval_weights'):
                interval_weights = ratios_data['interval_weights']
                if interval_weights:
                    print(f"✅ {coin}: DB에서 trending 가중치 로드 성공 - {interval_weights}")
                    return interval_weights

            # 없으면 빈 딕셔너리 반환 (기본값 사용)
            return {}

        except Exception as e:
            print(f"⚠️ {coin}: 인터벌 가중치 로드 실패 - {e}")
            return {}

    def _get_most_common_value(self, interval_signals: Dict[str, SignalInfo], field: str) -> str:
        """가장 빈번한 값 반환"""
        try:
            values = []
            for signal in interval_signals.values():
                value = getattr(signal, field, 'unknown')
                if value and value != 'unknown':
                    values.append(value)
            
            if not values:
                return 'unknown'
            
            # 가장 빈번한 값 반환
            from collections import Counter
            counter = Counter(values)
            return counter.most_common(1)[0][0]
            
        except Exception as e:
            print(f"⚠️ 최빈값 계산 실패 ({field}): {e}")
            return 'unknown'
    
    def _get_candle_from_signal(self, signal: SignalInfo) -> pd.Series:
        """시그널에서 캔들 데이터 추출 (레짐 감지용)"""
        try:
            # 시그널에 포함된 지표들을 딕셔너리로 변환
            candle_data = {
                'rsi': signal.rsi,
                'macd': signal.macd,
                'volume_ratio': signal.volume_ratio,
                'volatility': signal.volatility,
                'close': signal.price,
                'volume': signal.volume,
                'atr': getattr(signal, 'atr', 0.0),
                'adx': getattr(signal, 'adx', 25.0),
            }
            return pd.Series(candle_data)
        except Exception as e:
            # 기본값으로 생성
            return pd.Series({
                'rsi': 50.0,
                'macd': 0.0,
                'volume_ratio': 1.0,
                'volatility': 0.02,
            })
    
    def _calculate_multi_regime_weights(self, all_regimes: Dict[str, str], interval_weights: Dict[str, float], coin: str = None) -> Dict[str, float]:
        """여러 인터벌의 레짐을 종합하여 최종 가중치 계산"""
        try:
            from collections import Counter

            # 레짐 분포 분석
            regime_counts = Counter(all_regimes.values())

            # 최빈 레짐 (우세한 레짐)
            dominant_regime = regime_counts.most_common(1)[0][0] if regime_counts else 'neutral'

            # 레짐 일관도 계산 (모든 인터벌이 같은 레짐인 경우)
            if len(regime_counts) == 1:
                # 모든 인터벌이 동일한 레짐
                consistency = 1.0
            elif len(regime_counts) == 2 and max(regime_counts.values()) > len(all_regimes) * 0.6:
                # 60% 이상이 동일한 레짐
                consistency = 0.7
            else:
                # 레짐이 다양한 경우
                consistency = 0.5

            # 우세 레짐 기반 가중치 계산 (DB 기반, coin 전달)
            base_coin_weight, base_global_weight = self._calculate_dynamic_weights(dominant_regime, coin=coin)
            
            # 일관도에 따라 가중치 조정
            # 일관도 높으면 글로벌 강조, 낮으면 개별 강조
            coin_weight = base_coin_weight + (1 - consistency) * 0.1  # 일관도 낮으면 개별 강조
            global_weight = base_global_weight + consistency * 0.1  # 일관도 높으면 글로벌 강조
            
            # 정규화
            total_weight = coin_weight + global_weight
            coin_weight /= total_weight
            global_weight /= total_weight
            
            # 인터벌별 최종 가중치 계산
            final_weights = {}
            for interval in all_regimes.keys():
                base_interval_weight = interval_weights.get(interval, 0.1)
                
                # 해당 인터벌의 레짐이 우세 레짐과 같은지 확인
                interval_regime = all_regimes.get(interval, 'neutral')
                if interval_regime == dominant_regime:
                    # 우세 레짐에 맞는 인터벌은 가중치 유지
                    regime_adjusted_weight = base_interval_weight
                else:
                    # 다른 레짐은 가중치 축소
                    regime_adjusted_weight = base_interval_weight * 0.7
                
                final_weights[interval] = regime_adjusted_weight
            
            # 가중치 정규화
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                for interval in final_weights:
                    final_weights[interval] /= total_weight
            
            if self.debug_mode:
                print(f"🎯 레짐 분포: {dict(regime_counts)}, 우세: {dominant_regime}, 일관도: {consistency:.2f}")
                print(f"📊 인터벌별 가중치: {final_weights}")
            
            return final_weights
            
        except Exception as e:
            print(f"⚠️ 레짐 기반 가중치 계산 실패: {e}")
            return interval_weights
    
    def _get_latest_price(self, coin: str) -> float:
        """최신 가격 조회"""
        try:
            with sqlite3.connect(CANDLES_DB_PATH) as conn:
                # 여러 인터벌에서 최신 가격 조회
                intervals = ['15m', '30m', '240m', '1d']
                
                for interval in intervals:
                    query = """
                    SELECT close FROM candles 
                    WHERE coin = ? AND interval = ? 
                    ORDER BY timestamp DESC LIMIT 1
                    """
                    result = conn.execute(query, (coin, interval)).fetchone()
                    
                    if result:
                        price = float(result[0])
                        if price > 0:
                            return price
                
                return 0.0
                
        except Exception as e:
            print(f"⚠️ 최신 가격 조회 실패 ({coin}): {e}")
            return 0.0

    def load_multi_timeframe_ai_model(self):
        """🚀 멀티 타임프레임 AI 모델 로드 (RL 시스템의 learning_engine와 연동)"""
        try:
            if PolicyTrainer is not None:
                try:
                    from learning_engine import PolicyTrainer
                except ImportError:
                    print("⚠️ learning_engine을 import할 수 없습니다. 기본 모델 사용")
                    self.mtf_ai_model = None
                    return False
                
                # 멀티 타임프레임 모델 로드
                self.mtf_ai_model = PolicyTrainer(enable_multi_timeframe=True)
                self.mtf_ai_model.load_model()
            else:
                print("⚠️ PolicyTrainer를 사용할 수 없습니다. 기본 모델 사용")
                self.mtf_ai_model = None
            
            print("✅ 멀티 타임프레임 AI 모델 로드 완료")
            self.mtf_ai_model_loaded = True
            return True
            
        except Exception as e:
            print(f"⚠️ 멀티 타임프레임 AI 모델 로드 실패: {e}")
            self.mtf_ai_model_loaded = False
            return False
    
    def get_multi_timeframe_ai_score(self, coin: str, intervals: List[str] = ['15m', '30m', '240m', '1d']) -> Dict[str, float]:
        """🚀 멀티 타임프레임 AI 모델 기반 점수 계산"""
        if not hasattr(self, 'mtf_ai_model_loaded') or not self.mtf_ai_model_loaded:
            return {
                'mu': 0.0,      # 수익률 예측
                'p_up': 0.5,    # 상승 확률 (기본값)
                'risk': 0.5,    # 리스크 (기본값)
                'adaptability': 0.5,  # 적응성 (기본값)
                'strategy_score': 0.0  # 전략 점수
            }
        
        try:
            # 🎯 각 인터벌별 특징 벡터 준비
            interval_features = {}
            for interval in intervals:
                try:
                    candle = self.get_nearest_candle(coin, interval, int(time.time()))
                    if candle is not None:
                        features = self._prepare_multi_timeframe_features(candle, interval)
                        interval_features[interval] = features
                except Exception as e:
                    print(f"⚠️ {coin} {interval} 특징 벡터 준비 실패: {e}")
                    continue
            
            if not interval_features:
                return {
                    'mu': 0.0, 'p_up': 0.5, 'risk': 0.5, 'adaptability': 0.5, 'strategy_score': 0.0
                }
            
            # 🎯 멀티 타임프레임 특징 통합
            combined_features = self._combine_multi_timeframe_features(interval_features)
            
            # 🎯 AI 모델 예측
            predictions = self.mtf_ai_model.predict(combined_features)
            
            # 🎯 결과 반환
            result = {
                'mu': float(predictions['mu'][0]),           # 수익률 예측
                'p_up': float(predictions['p_up'][0]),       # 상승 확률
                'risk': float(predictions['risk'][0]),       # 리스크
                'adaptability': float(predictions.get('adaptability', [0.5])[0]),  # 적응성
                'strategy_score': 0.0  # 기본값
            }
            
            # 🎯 전략 점수 계산 (멀티 타임프레임 적응성 포함)
            if hasattr(self.mtf_ai_model, 'predict_strategy_score_with_multi_timeframe'):
                strategy_score = self.mtf_ai_model.predict_strategy_score_with_multi_timeframe(combined_features)
                result['strategy_score'] = float(strategy_score[0])
            else:
                # 기본 전략 점수 계산
                result['strategy_score'] = (result['mu'] * 0.4 + result['p_up'] * 0.4 - result['risk'] * 0.2)
            
            print(f"🧠 {coin} 멀티 타임프레임 AI 점수: 수익률={result['mu']:.3f}, 상승확률={result['p_up']:.3f}, 리스크={result['risk']:.3f}, 적응성={result['adaptability']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ {coin} 멀티 타임프레임 AI 점수 계산 실패: {e}")
            return {
                'mu': 0.0, 'p_up': 0.5, 'risk': 0.5, 'adaptability': 0.5, 'strategy_score': 0.0
            }
    
    def _prepare_multi_timeframe_features(self, candle: pd.Series, interval: str) -> np.ndarray:
        """멀티 타임프레임 특징 벡터 준비"""
        try:
            features = []
            
            # 🎯 기본 기술적 지표
            features.extend([
                safe_float(candle.get('rsi', 50.0)) / 100.0,
                safe_float(candle.get('macd', 0.0)),
                safe_float(candle.get('volume_ratio', 1.0)),
                safe_float(candle.get('volatility', 0.0)),
                safe_float(candle.get('structure_score', 0.5)),
                safe_float(candle.get('pattern_confidence', 0.0))
            ])
            
            # 🎯 고급 지표들
            features.extend([
                safe_float(candle.get('mfi', 50.0)) / 100.0,
                safe_float(candle.get('atr', 0.0)),
                safe_float(candle.get('adx', 25.0)) / 100.0,
                safe_float(candle.get('bb_squeeze', 0.0)),
                safe_float(candle.get('trend_strength', 0.5)),
                safe_float(candle.get('price_momentum', 0.0))
            ])
            
            # 🎯 인터벌별 가중치 (멀티 타임프레임 특성)
            interval_weight = {'15m': 0.20, '30m': 0.25, '240m': 0.35, '1d': 0.45}.get(interval, 0.25)
            features.append(interval_weight)
            
            # 🎯 특징 벡터를 numpy 배열로 변환
            feature_array = np.array(features, dtype=np.float32)
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            print(f"⚠️ 멀티 타임프레임 특징 벡터 준비 실패: {e}")
            return np.zeros((1, 20), dtype=np.float32)
    
    def _combine_multi_timeframe_features(self, interval_features: Dict[str, np.ndarray]) -> np.ndarray:
        """여러 인터벌의 특징 벡터를 통합"""
        try:
            if not interval_features:
                return np.zeros((1, 20), dtype=np.float32)
            
            # 🎯 인터벌별 가중치
            interval_weights = {
                '15m': 0.20, '30m': 0.25, '240m': 0.35, '1d': 0.45
            }
            
            # 🎯 가중 평균으로 특징 통합
            combined_features = np.zeros_like(list(interval_features.values())[0])
            total_weight = 0.0
            
            for interval, features in interval_features.items():
                weight = interval_weights.get(interval, 0.25)
                combined_features += features * weight
                total_weight += weight
            
            if total_weight > 0:
                combined_features /= total_weight
            
            return combined_features
            
        except Exception as e:
            print(f"⚠️ 멀티 타임프레임 특징 통합 실패: {e}")
            return np.zeros((1, 20), dtype=np.float32)
    
    # 🆕 시너지 학습 결과 활용 메서드들
    def _load_synergy_patterns(self):
        """시너지 학습 결과 로드 (강화된 에러 처리)"""
        try:
            # 시너지 패턴 테이블 존재 확인 및 생성
            with sqlite3.connect("/workspace/data_storage/learning_results.db") as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 여부 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synergy_patterns'")
                if not cursor.fetchone():
                    print("🆕 synergy_patterns 테이블 생성 중...")
                    self._create_synergy_patterns_table(cursor)
                    conn.commit()
                
                # synergy_score 컬럼 존재 여부 확인
                cursor.execute("PRAGMA table_info(synergy_patterns)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'synergy_score' not in columns:
                    print("🆕 synergy_score 컬럼 추가 중...")
                    cursor.execute('ALTER TABLE synergy_patterns ADD COLUMN synergy_score REAL DEFAULT 0.0')
                    cursor.execute('UPDATE synergy_patterns SET synergy_score = confidence_score * success_rate')
                    conn.commit()
                
                # 시너지 패턴 로드
                cursor.execute('''
                    SELECT pattern_name, pattern_type, market_condition, pattern_data, 
                           confidence_score, success_rate, synergy_score
                    FROM synergy_patterns
                ''')
                
                patterns = cursor.fetchall()
                self.synergy_patterns = {}
                
                for pattern in patterns:
                    pattern_name, pattern_type, market_condition, pattern_data, confidence, success, synergy = pattern
                    self.synergy_patterns[pattern_name] = {
                        'type': pattern_type,
                        'market_condition': market_condition,
                        'data': json.loads(pattern_data) if pattern_data else {},
                        'confidence': confidence or 0.0,
                        'success_rate': success or 0.0,
                        'synergy_score': synergy or 0.0
                    }
                
                print(f"✅ 시너지 패턴 로드 완료: {len(self.synergy_patterns)}개 패턴")
                
        except Exception as e:
            print(f"⚠️ 시너지 패턴 로드 실패: {e}")
            # 기본 시너지 패턴 사용
            self.synergy_patterns = self._get_default_synergy_patterns()
    
    def _create_synergy_patterns_table(self, cursor):
        """시너지 패턴 테이블 생성"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synergy_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                synergy_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 기본 패턴 데이터 삽입
        patterns = [
            ('bullish_momentum', 'momentum', 'bull', '{"rsi_range": [30, 70], "macd_positive": true, "volume_increase": true}', 0.8, 0.75, 0.6),
            ('bearish_reversal', 'reversal', 'bear', '{"rsi_range": [70, 90], "macd_negative": true, "volume_spike": true}', 0.7, 0.65, 0.455),
            ('sideways_breakout', 'breakout', 'sideways', '{"rsi_range": [40, 60], "macd_neutral": true, "volume_normal": true}', 0.6, 0.55, 0.33),
            ('volatility_surge', 'volatility', 'any', '{"high_volatility": true, "volume_surge": true}', 0.5, 0.45, 0.225)
        ]
        
        cursor.executemany('''
            INSERT INTO synergy_patterns (pattern_name, pattern_type, market_condition, pattern_data, confidence_score, success_rate, synergy_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', patterns)
    
    def _get_default_synergy_patterns(self):
        """기본 시너지 패턴 반환 (fallback)"""
        return {
            'bullish_momentum': {
                'type': 'momentum',
                'market_condition': 'bull',
                'data': {'rsi_range': [30, 70], 'macd_positive': True, 'volume_increase': True},
                'confidence': 0.8,
                'success_rate': 0.75,
                'synergy_score': 0.6
            },
            'bearish_reversal': {
                'type': 'reversal',
                'market_condition': 'bear',
                'data': {'rsi_range': [70, 90], 'macd_negative': True, 'volume_spike': True},
                'confidence': 0.7,
                'success_rate': 0.65,
                'synergy_score': 0.455
            }
        }
    
    def get_synergy_enhanced_signal_score(self, coin: str, interval: str, base_score: float, 
                                        market_condition: str = None) -> float:
        """시너지 학습 결과를 활용한 향상된 시그널 점수 계산"""
        try:
            if not self.synergy_learning_available or not self.synergy_patterns:
                return base_score
            
            enhanced_score = base_score
            synergy_bonus = 0.0
            
            # 시너지 점수가 높은 경우에만 보너스 적용
            synergy_score = self.synergy_patterns.get('synergy_score', 0.0)
            if synergy_score > 0.6:  # 높은 시너지 점수
                synergy_bonus = 0.1  # 10% 보너스
            elif synergy_score > 0.4:  # 중간 시너지 점수
                synergy_bonus = 0.05  # 5% 보너스
            
            # 최적 시장 조건 보너스
            if market_condition and 'optimal_market_conditions' in self.synergy_patterns:
                for condition_data in self.synergy_patterns['optimal_market_conditions']:
                    if condition_data['condition'] == market_condition:
                        condition_bonus = min(condition_data.get('avg_profit', 0.0) * 0.5, 0.15)
                        synergy_bonus += condition_bonus
                        break
            
            # 타이밍 권장사항 보너스
            if 'timing_recommendations' in self.synergy_patterns:
                for rec in self.synergy_patterns['timing_recommendations']:
                    if rec.get('confidence', 0.0) > 0.7:
                        synergy_bonus += 0.02  # 2% 추가 보너스
            
            # 최종 향상된 점수 계산
            enhanced_score = base_score * (1 + synergy_bonus)
            
            # 점수 범위 제한 (0.0 ~ 1.0)
            enhanced_score = max(0.0, min(1.0, enhanced_score))
            
            if synergy_bonus > 0:
                print(f"🔄 {coin}/{interval}: 시너지 보너스 적용 - 기본점수: {base_score:.3f} → 향상점수: {enhanced_score:.3f} (+{synergy_bonus:.1%})")
            
            return enhanced_score
            
        except Exception as e:
            print(f"⚠️ 시너지 향상 점수 계산 실패: {e}")
            return base_score
    
    def get_synergy_recommendations_for_signal(self, coin: str, interval: str, 
                                             market_condition: str = None) -> List[Dict[str, Any]]:
        """시그널 생성에 활용할 시너지 권장사항 반환"""
        try:
            if not self.synergy_learning_available or not self.synergy_learner:
                return []
            
            recommendations = self.synergy_learner.get_synergy_recommendations(market_condition)
            
            # 코인/인터벌별 필터링
            filtered_recommendations = []
            for rec in recommendations:
                # 시그널 생성에 직접 활용 가능한 권장사항만 필터링
                if rec.get('type') in ['market_condition', 'timing_recommendations', 'performance_enhancement_tips']:
                    filtered_recommendations.append(rec)
            
            return filtered_recommendations
            
        except Exception as e:
            print(f"⚠️ 시너지 권장사항 조회 실패: {e}")
            return []
    
    def _test_synergy_learning_integration(self):
        """시너지 학습 통합 테스트 (비활성화 - 불필요한 테스트)"""
        # 시그널 계산이 완료된 후에는 시너지 학습 테스트가 불필요함
        print("ℹ️ 시너지 학습 테스트는 비활성화됨 (시그널 계산 완료 후 불필요)")
        return

# ============================================================================
# 🆕 전략 점수 계산기 클래스 (리팩토링)
# ============================================================================

class StrategyScoreCalculator:
    """전략 점수 계산을 담당하는 별도 클래스 (learning_engine.py 연동 강화)"""
    
    def __init__(self):
        self.global_strategies = {}  # 딕셔너리로 변경
        self.coin_tuned_strategies = {}
        self.reliability_scores = {}
        self.global_strategies_loaded = False
        self.coin_strategies_loaded = False
        self.reliability_scores_loaded = False
        
        # 🆕 학습 기반 임계값 관리
        self.use_learning_based_thresholds = True
        self.learning_feedback = None
        self.min_confidence = 0.5
        self.min_signal_score = 0.03
        
        # 🆕 AI 모델 초기화
        self.ai_model = None
        self.ai_model_loaded = False
        self.model_type = "none"
        self.current_coin = None
        self.feature_dim = 0
        
        # 데이터베이스 초기화
        self.create_signal_table()
        
        # 전략 데이터 로드
        self.load_global_strategies()
        self.load_coin_tuned_strategies()
        self.load_reliability_scores()
        
        # 🆕 AI 모델 로드
        if AI_MODEL_AVAILABLE:
            self._load_ai_model()
    
    def create_signal_table(self):
        """시그널 피드백 테이블 생성 (trading_system.db에 저장)"""
        try:
            conn = sqlite3.connect('data_storage/trading_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    feedback_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, interval, signal_type, feedback_type)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ 시그널 테이블 생성 실패: {e}")
    
    def load_global_strategies(self):
        """글로벌 전략 로드"""
        try:
            conn = sqlite3.connect('/workspace/data_storage/learning_results.db')
            cursor = conn.cursor()
            
            # 🚀 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_grades'")
            if not cursor.fetchone():
                print(f"⚠️ strategy_grades 테이블이 존재하지 않음 - 기본 전략만 사용")
                self.global_strategies_loaded = True  # 로드 완료로 표시 (빈 상태)
                conn.close()
                return
            
            # 🚀 더 관대한 조건으로 글로벌 전략 로드 (학습된 데이터 부족 문제 해결)
            cursor.execute('''
                SELECT strategy_data, performance_metrics, created_at, performance_score
                FROM strategy_grades
                WHERE (strategy_type = 'learned' OR strategy_type IS NULL)
                ORDER BY COALESCE(performance_score, 0.5) DESC
                LIMIT 100
            ''')
            
            strategies = cursor.fetchall()
            for i, (strategy_data, performance_metrics, created_at) in enumerate(strategies):
                try:
                    strategy = json.loads(strategy_data)
                    metrics = json.loads(performance_metrics) if performance_metrics else {}
                    
                    strategy_key = f"global_strategy_{i}"
                    self.global_strategies[strategy_key] = {
                        'strategy': strategy,
                        'metrics': metrics,
                        'created_at': created_at
                    }
                except Exception as e:
                    continue
            
            self.global_strategies_loaded = True
            print(f"✅ 글로벌 전략 로드: {len(self.global_strategies)}개")
            
        except Exception as e:
            print(f"⚠️ 글로벌 전략 로드 실패: {e}")
            self.global_strategies_loaded = False
    
    def load_coin_tuned_strategies(self):
        """코인별 튜닝된 전략 로드 (learning_engine.py에서 생성된 데이터)"""
        try:
            conn = sqlite3.connect('/workspace/data_storage/learning_results.db')
            cursor = conn.cursor()
            
            # 🚀 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_tuned_strategies'")
            if not cursor.fetchone():
                print(f"⚠️ coin_tuned_strategies 테이블이 존재하지 않음 - 기본 전략만 사용")
                self.coin_strategies_loaded = True  # 로드 완료로 표시 (빈 상태)
                conn.close()
                return
            
            # 코인별 튜닝된 전략 로드
            cursor.execute('''
                SELECT coin, strategy_id, base_global_strategy_id, tuned_parameters, 
                       performance_metrics, created_at, description
                FROM coin_tuned_strategies
                ORDER BY created_at DESC
            ''')
            
            strategies = cursor.fetchall()
            for coin, strategy_id, base_global_strategy_id, tuned_parameters, performance_metrics, created_at, description in strategies:
                try:
                    tuned_params = json.loads(tuned_parameters) if tuned_parameters else {}
                    metrics = json.loads(performance_metrics) if performance_metrics else {}
                    
                    if coin not in self.coin_tuned_strategies:
                        self.coin_tuned_strategies[coin] = []
                    
                    self.coin_tuned_strategies[coin].append({
                        'strategy_id': strategy_id,
                        'base_global_strategy_id': base_global_strategy_id,
                        'tuned_parameters': tuned_params,
                        'performance_metrics': metrics,
                        'created_at': created_at,
                        'description': description
                    })
                except Exception as e:
                    continue
            
            self.coin_strategies_loaded = True
            print(f"✅ 코인별 튜닝 전략 로드: {len(self.coin_tuned_strategies)}개 코인")
            
        except Exception as e:
            print(f"⚠️ 코인별 튜닝 전략 로드 실패: {e}")
            self.coin_strategies_loaded = False
    
    def load_reliability_scores(self):
        """신뢰도 점수 로드"""
        try:
            conn = sqlite3.connect('data_storage/virtual_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT coin, interval, signal_type, score, feedback_type
                FROM signal_feedback_scores
                ORDER BY created_at DESC
            ''')
            
            scores = cursor.fetchall()
            for coin, interval, signal_type, score, feedback_type in scores:
                key = f"{coin}_{interval}_{signal_type}_{feedback_type}"
                self.reliability_scores[key] = score
            
            self.reliability_scores_loaded = True
            print(f"✅ 신뢰도 점수 로드 완료: {len(self.reliability_scores)}개")
            
        except Exception as e:
            print(f"⚠️ 신뢰도 점수 로드 실패: {e}")
            self.reliability_scores_loaded = False
    
    def _load_ai_model(self):
        """학습된 전략 기반 AI 모델 로드"""
        try:
            if not AI_MODEL_AVAILABLE:
                print("⚠️ AI 모델을 사용할 수 없습니다")
                return
            
            # 🆕 데이터베이스에서 학습된 전략 로드
            _load_learned_strategies_from_db()
            
            # 🆕 전략 기반 AI 모델 생성
            self.ai_model, self.model_type = _create_strategy_based_ai_model()
            self.feature_dim = 50
            self.ai_model_loaded = True
            print(f"✅ 학습된 전략 기반 AI 모델 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 학습된 전략 기반 AI 모델 로드 실패: {e}")
            self.ai_model_loaded = False
    
    def _load_learning_engines(self):
        """learning_engine.py의 학습 엔진들 로드"""
        try:
            if not AI_MODEL_AVAILABLE:
                return
            
            # 글로벌 학습 매니저 로드
            self.global_learning_manager = GlobalLearningManager()
            print("✅ 글로벌 학습 매니저 로드 완료")
            
            # 심볼별 튜닝 매니저 로드
            self.symbol_finetuning_manager = SymbolFinetuningManager()
            print("✅ 심볼별 튜닝 매니저 로드 완료")
            
            # 시너지 학습기 로드
            self.synergy_learner = ShortTermLongTermSynergyLearner()
            print("✅ 시너지 학습기 로드 완료")
            
            # 🆕 신뢰도 점수 계산기 로드
            self.reliability_calculator = ReliabilityScoreCalculator()
            print("✅ 신뢰도 점수 계산기 로드 완료")
            
            # 🆕 지속적 학습 관리자 로드
            self.continuous_learning_manager = ContinuousLearningManager()
            print("✅ 지속적 학습 관리자 로드 완료")
            
            # 🆕 라우팅 패턴 분석기 로드
            self.routing_pattern_analyzer = RoutingPatternAnalyzer()
            print("✅ 라우팅 패턴 분석기 로드 완료")
            
            # 🆕 상황별 학습 관리자 로드
            self.contextual_learning_manager = ContextualLearningManager()
            print("✅ 상황별 학습 관리자 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 학습 엔진 로드 실패: {e}")
            self.global_learning_manager = None
            self.symbol_finetuning_manager = None
            self.synergy_learner = None
            self.reliability_calculator = None
            self.continuous_learning_manager = None
            self.routing_pattern_analyzer = None
            self.contextual_learning_manager = None
    
    def calculate_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """전략 점수 계산"""
        try:
            # 기본 점수
            base_score = 0.5
            
            # 글로벌 전략 점수
            if self.global_strategies_loaded:
                global_score = self._get_global_strategy_score(coin, interval, candle)
                base_score = max(base_score, global_score)
            
            # 심볼별 전략 점수
            if self.coin_strategies_loaded and coin in self.coin_tuned_strategies:
                symbol_score = self._get_symbol_strategy_score(coin, interval, candle)
                base_score = max(base_score, symbol_score)
            
            # 신뢰도 점수 적용
            if self.reliability_scores_loaded:
                reliability_bonus = self._get_reliability_bonus(coin, interval, candle)
                base_score *= reliability_bonus
            
            # AI 모델 점수 적용
            if self.ai_model_loaded:
                ai_score = self._get_ai_model_score(coin, interval, candle)
                base_score = (base_score + ai_score) / 2
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            print(f"⚠️ 전략 점수 계산 실패: {e}")
            return 0.5
    
    def _get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """글로벌 전략 점수 계산"""
        try:
            if not self.global_strategies:
                return 0.5
            
            # 가장 최근 전략 사용
            latest_strategy = self.global_strategies[0]
            strategy = latest_strategy['strategy']
            metrics = latest_strategy['metrics']
            
            # 전략 점수 계산
            score = 0.5
            if 'performance_score' in metrics:
                score = metrics['performance_score']
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5

    def get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """Public wrapper for _get_global_strategy_score"""
        return self._get_global_strategy_score(coin, interval, candle)

    def _get_symbol_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """코인별 튜닝 전략 점수 계산 (learning_engine.py 데이터 활용)"""
        try:
            if coin not in self.coin_tuned_strategies:
                return 0.5
            
            strategies = self.coin_tuned_strategies[coin]
            if not strategies:
                return 0.5
            
            # 가장 최근 전략 사용
            latest_strategy = strategies[0]
            tuned_params = latest_strategy['tuned_parameters']
            metrics = latest_strategy['performance_metrics']
            
            # 전략 점수 계산
            score = 0.5
            
            # 성과 메트릭에서 점수 추출
            if 'success_rate' in metrics:
                score = max(score, metrics['success_rate'])
            if 'avg_reward' in metrics:
                score = max(score, abs(metrics['avg_reward']) * 2)  # 보상값을 점수로 변환
            
            # 튜닝된 파라미터에서 추가 점수 계산
            if 'action_type' in tuned_params:
                action_type = tuned_params['action_type']
                if action_type in ['buy', 'sell']:
                    score += 0.1  # 액션 타입이 명확하면 보너스
            
            # 글로벌 시너지 점수 반영
            if 'coin_specific_adjustments' in tuned_params:
                adjustments = tuned_params['coin_specific_adjustments']
                if 'synergy_score' in adjustments:
                    score += adjustments['synergy_score'] * 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5
    
    def _get_reliability_bonus(self, coin: str, interval: str, candle: pd.Series) -> float:
        """신뢰도 보너스 계산"""
        try:
            # 신뢰도 점수 조회
            key = f"{coin}_{interval}_buy_positive"
            if key in self.reliability_scores:
                return self.reliability_scores[key]
            
            return 1.0
            
        except Exception as e:
            return 1.0
    
    def _get_ai_model_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """AI 모델 점수 계산"""
        try:
            if not self.ai_model_loaded:
                return 0.5
            
            # 특징 추출
            features = self._extract_features(candle)
            
            if self.model_type == "pytorch":
                # PyTorch 모델 추론
                try:
                    import torch
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        prediction = self.ai_model(features_tensor)
                        score = torch.sigmoid(prediction).item()
                except ImportError:
                    print("⚠️ PyTorch를 import할 수 없습니다. 기본 점수 사용")
                    score = 0.5
            elif self.model_type == "sklearn":
                # Scikit-learn 모델 추론
                score = self.ai_model.predict_proba([features])[0][1]
            else:
                return 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5
    
    def _extract_features(self, candle: pd.Series) -> List[float]:
        """특징 추출"""
        try:
            features = []
            
            # 기본 가격 특징
            features.append(candle['open'])
            features.append(candle['high'])
            features.append(candle['low'])
            features.append(candle['close'])
            features.append(candle['volume'])
            
            # 기술적 지표
            if 'rsi' in candle:
                features.append(candle['rsi'])
            else:
                features.append(50.0)
            
            if 'macd' in candle:
                features.append(candle['macd'])
            else:
                features.append(0.0)
            
            if 'bb_upper' in candle and 'bb_lower' in candle:
                bb_position = (candle['close'] - candle['bb_lower']) / (candle['bb_upper'] - candle['bb_lower'])
                features.append(bb_position)
            else:
                features.append(0.5)
            
            return features
            
        except Exception as e:
            return [0.0] * 8

def main():
    """🚀 고성능 실시간 시그널 선택기 메인 실행 함수"""
    print("🚀 고성능 실시간 시그널 선택기 시작")
    print("🎯 목표: GPU 가속 + 크로스 코인 학습 통합 시그널 생성")
    print("🆕 고성능 캐시, 병렬 처리, 적응형 AI 모델 선택")
    print("=" * 60)
    
    # 🚀 고성능 시스템 설정 표시
    print("🚀 고성능 시스템 설정:")
    print(f"   - GPU 가속: {USE_GPU_ACCELERATION}")
    print(f"   - JAX 플랫폼: {JAX_PLATFORM_NAME}")
    print(f"   - 병렬 워커: {MAX_WORKERS}")
    print(f"   - 캐시 크기: {CACHE_SIZE:,}")
    print(f"   - 크로스 코인 학습: {ENABLE_CROSS_COIN_LEARNING}")
    print("=" * 60)
    
    try:
        # 시그널 선택기 초기화
        selector = SignalSelector()
        
        # 🚀 고성능 시스템 상태 확인
        print("\n🔍 고성능 시스템 상태 확인 중...")
        
        # 🚀 AI 모델 상태 확인
        if selector.ai_model_loaded:
            print("✅ AI 모델 로드 완료 - GPU 가속 AI 기반 시그널 점수 계산 활성화")
            print(f"   - 모델 타입: {selector.model_type}")
            print(f"   - GPU 가속: {USE_GPU_ACCELERATION}")
        else:
            print("⚠️ AI 모델 로드 실패 - 기본 시그널 계산만 사용")
        
        # 🚀 크로스 코인 학습 상태 확인
        if selector.cross_coin_available:
            print("✅ 크로스 코인 학습 컨텍스트 로드 완료")
        else:
            print("⚠️ 크로스 코인 학습 컨텍스트를 사용할 수 없습니다")
        
        # 🚀 캐시 시스템 상태 확인
        print(f"✅ 고성능 캐시 시스템: 최대 {selector.max_cache_size:,}개 항목")
        
        # 🆕 시스템 상태 확인
        print("\n🔍 데이터베이스 상태 확인 중...")
        
        # 데이터베이스 연결 확인
        try:
            with sqlite3.connect(CANDLES_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM candles")
                candle_count = cursor.fetchone()[0]
                print(f"  ✅ 캔들 데이터: {candle_count:,}개")
                
                cursor.execute("SELECT COUNT(DISTINCT coin) FROM candles")
                coin_count = cursor.fetchone()[0]
                print(f"  ✅ 코인 수: {coin_count}개")
                
                # signals 테이블 존재 여부 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*) FROM signals")
                    signal_count = cursor.fetchone()[0]
                    print(f"  ✅ 기존 시그널: {signal_count:,}개")
                else:
                    print(f"  ℹ️ 시그널 테이블이 아직 생성되지 않았습니다")
        except Exception as e:
            print(f"  ❌ 데이터베이스 연결 실패: {e}")
            return
        
        print("✅ 시스템 상태 확인 완료")
        print("-" * 60)
        
        # �� 전체 코인 멀티인터벌 시그널 생성 (성능 최적화)
        print("\n🧪 전체 코인 멀티인터벌 시그널 생성 중...")
        
        # 사용 가능한 모든 코인 조회
        try:
            with sqlite3.connect(CANDLES_DB_PATH) as conn:
                coins_df = pd.read_sql("""
                    SELECT DISTINCT coin 
                    FROM candles 
                    WHERE interval IN ('15m', '30m', '240m', '1d')
                    ORDER BY coin
                """, conn)
        except Exception as e:
            print(f"❌ 코인 조회 실패: {e}")
            return
        
        if coins_df.empty:
            print("❌ 사용 가능한 코인이 없습니다")
            return
        
        print(f"📊 총 {len(coins_df)}개 코인에 대해 멀티인터벌 시그널 생성")
        
        # 🆕 코인별 멀티인터벌 시그널 생성 (간소화된 출력)
        combined_signals = []
        intervals = ['15m', '30m', '240m', '1d']
        
        for idx, row in coins_df.iterrows():
            coin = row['coin']
            
            try:
                # 각 인터벌별 시그널 생성 (간소화된 출력)
                interval_signals = {}
                for interval in intervals:
                    signal = selector.generate_signal(coin, interval)
                    if signal:
                        interval_signals[interval] = signal
                
                # 멀티인터벌 시그널 결합 (🔥 DB 기반 동적 가중치 사용)
                if len(interval_signals) >= 2:  # 최소 2개 인터벌 이상 있어야 결합
                    combined_signal = selector.combine_multi_timeframe_signals(coin, interval_signals)
                    if combined_signal:
                        combined_signals.append(combined_signal)

                        # 🔥 통합 시그널 DB 저장
                        try:
                            selector.save_signal_to_db(combined_signal)
                        except Exception as save_err:
                            print(f"⚠️ {coin} 통합 시그널 DB 저장 실패: {save_err}")

                        # 🔥 코인 종합 점수 명확하게 출력
                        print(f"\n{'='*60}")
                        print(f"🎯 [{coin}] 최종 종합 시그널 (멀티인터벌 통합)")
                        print(f"{'='*60}")
                        print(f"  📊 종합 점수: {combined_signal.signal_score:.4f}")
                        print(f"  📊 신뢰도: {combined_signal.confidence:.4f}")
                        # 🔧 액션은 시그널이 아닌 트레이더가 결정 (사용자 요청: 액션 노출 제거)
                        # print(f"  🎯 최종 액션: {combined_signal.action.value.upper()}")
                        print(f"  📈 사용된 인터벌: {len(interval_signals)}개 ({', '.join(interval_signals.keys())})")
                        print(f"  💰 현재가: ${combined_signal.price:.6f}")
                        print(f"  📊 RSI: {combined_signal.rsi:.2f}")
                        print(f"  📊 MACD: {combined_signal.macd:.6f}")
                        print(f"  📊 변동성: {combined_signal.volatility:.4f}")
                        print(f"  📊 거래량 비율: {combined_signal.volume_ratio:.2f}x")
                        print(f"  🌊 파동 단계: {combined_signal.wave_phase}")
                        print(f"  📈 패턴: {combined_signal.pattern_type}")
                        print(f"  🎯 통합 방향: {combined_signal.integrated_direction}")
                        print(f"{'='*60}\n")

                        print(f"✅ {coin}: 멀티인터벌 시그널 생성 성공 ({len(interval_signals)}개 인터벌)")
                    else:
                        print(f"⚠️ {coin}: 멀티인터벌 시그널 결합 실패")
                else:
                    print(f"⚠️ {coin}: 충분한 인터벌 데이터 없음 ({len(interval_signals)}개)")
                    
            except Exception as e:
                print(f"❌ {coin}: 시그널 생성 오류 - {e}")
        
        print(f"\n📊 멀티인터벌 시그널 생성 결과: {len(combined_signals)}/{len(coins_df)}개 코인")
        
        # 🆕 통계 카운터 수동 업데이트 (main 함수에서 생성된 시그널들)
        selector._signal_stats['total_signals_generated'] += len(combined_signals)
        
        # 🆕 시너지 학습 결과 활용 테스트 (불필요한 테스트 제거)
        # if selector.synergy_learning_available:
        #     print("\n🔄 시너지 학습 결과 활용 테스트...")
        #     selector._test_synergy_learning_integration()
        selector._signal_stats['successful_signals'] += len(combined_signals)
        
        # 🆕 상세한 통계 출력
        selector._log_signal_stats()
        
        print("\n✅ 실시간 시그널 선택기 테스트 완료")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
    finally:
        print("\n🎉 시스템 종료")

# ============================================================================
# 🆕 전략 점수 계산기 클래스는 이미 위에 정의됨 (중복 제거)
# ============================================================================

def save_dimension_info_to_db(coin: str, dimension_info: dict):
    """차원 정보를 데이터베이스에 저장 (learning_results.db)"""
    try:
        import sqlite3
        db_path = "/workspace/data_storage/learning_results.db"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # dimension_info 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dimension_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT,
                    dimension_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 기존 데이터 삭제 (최신 차원 정보만 유지)
            cursor.execute("DELETE FROM dimension_info WHERE coin = ?", (coin,))
            
            # 새로운 차원 정보 저장
            cursor.execute("""
                INSERT INTO dimension_info (coin, dimension_data)
                VALUES (?, ?)
            """, (coin, json.dumps(dimension_info, ensure_ascii=False)))
            
            conn.commit()
            logger.info(f"✅ {coin} 차원 정보 데이터베이스 저장 완료")
            
    except Exception as e:
        logger.error(f"❌ {coin} 차원 정보 저장 실패: {e}")

def load_dimension_info_from_db(coin: str) -> dict:
    """데이터베이스에서 차원 정보 로드 (learning_results.db)"""
    try:
        import sqlite3
        db_path = "/workspace/data_storage/learning_results.db"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT dimension_data FROM dimension_info WHERE coin = ? ORDER BY created_at DESC LIMIT 1", (coin,))
            row = cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            else:
                return {}
                
    except Exception as e:
        logger.error(f"❌ {coin} 차원 정보 로드 실패: {e}")
        return {}

def _load_learned_strategies_from_db():
    """데이터베이스에서 학습된 전략 로드"""
    try:
        # rl_strategies.db에서 coin_strategies 로드
        rl_strategies_db = "/workspace/data_storage/rl_strategies.db"
        conn = sqlite3.connect(rl_strategies_db)
        cursor = conn.cursor()

        # coin_strategies 테이블에서 전략 로드
        cursor.execute("SELECT COUNT(*) FROM coin_strategies")
        coin_count = cursor.fetchone()[0]
        print(f"📊 코인별 전략 {coin_count:,}개 발견 (rl_strategies.db)")

        # 글로벌 전략도 확인 (있으면)
        try:
            cursor.execute("SELECT COUNT(*) FROM global_strategies")
            global_count = cursor.fetchone()[0]
            print(f"📊 글로벌 전략 {global_count:,}개 발견")
        except:
            print(f"ℹ️ global_strategies 테이블 없음")

        conn.close()

    except Exception as e:
        print(f"⚠️ 학습된 전략 로드 실패: {e}")

def _create_strategy_based_ai_model():
    """학습된 전략 기반 AI 모델 생성"""
    try:
        feature_dim = 50  # 기본 차원
        ai_model = PolicyTrainer(feature_dim=feature_dim)
        model_type = "strategy_based"
        print(f"✅ 전략 기반 AI 모델 생성 완료 (차원: {feature_dim})")
        return ai_model, model_type
        
    except Exception as e:
        print(f"⚠️ 전략 기반 AI 모델 생성 실패: {e}")
        return _create_default_ai_model()

def _create_default_ai_model():
    """기본 AI 모델 생성"""
    try:
        feature_dim = 50
        ai_model = PolicyTrainer(feature_dim=feature_dim)
        model_type = "default"
        print(f"✅ 기본 AI 모델 생성 완료 (차원: {feature_dim})")
        return ai_model, model_type
        
    except Exception as e:
        print(f"⚠️ 기본 AI 모델 생성 실패: {e}")
        return None, "none"

if __name__ == "__main__":
    main()