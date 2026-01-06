"""
core 관련 Mixin 클래스
SignalSelector의 core 기능을 담당합니다.
"""



# === 공통 import ===
import os
import sys
import logging
import traceback
import time
import json
import math
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

# signal_selector 내부 모듈 - 순환 참조 방지를 위해 지연 임포트 사용
# 필요한 타입 정의만 상단에 유지하거나 상단 임포트 최소화
try:
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir, MAX_WORKERS,
        VOLATILITY_SYSTEM_AVAILABLE
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    # ⚠️ evaluators 임포트를 여기서 제거 (순환 참조의 주범)
except ImportError:
    # 직접 실행 시 경로 추가 로직은 유지하되 임포트 최소화
    _current = os.path.dirname(os.path.abspath(__file__))
    _signal_selector = os.path.dirname(_current)
    _trade = os.path.dirname(_signal_selector)
    sys.path.insert(0, _trade)
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir, MAX_WORKERS,
        VOLATILITY_SYSTEM_AVAILABLE
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )

# 헬퍼 클래스 임포트 제거 (메소드 내부로 이동)
# ThompsonSamplingLearner 임포트 제거 (메소드 내부로 이동)

# StrategyScoreCalculator import는 순환 참조 방지를 위해 __init__ 내부에서 수행합니다.


class CoreMixin:
    """
    CoreMixin - core 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

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
        try:
            from signal_selector.evaluators import OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
            self.off_policy_evaluator = OffPolicyEvaluator()
            self.confidence_calibrator = ConfidenceCalibrator()
            self.meta_corrector = MetaCorrector()
        except ImportError:
            self.off_policy_evaluator, self.confidence_calibrator, self.meta_corrector = None, None, None
        
        # 🆕 2단계 성능 업그레이드 시스템 초기화
        try:
            from signal_selector.helpers import (
                ContextualBandit, RegimeChangeDetector, ExponentialDecayWeight,
                BayesianSmoothing, ActionSpecificScorer, ContextFeatureExtractor,
                OutlierGuardrail, EvolutionEngine, ContextMemory, RealTimeLearner,
                SignalTradeConnector
            )
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
            self.real_time_learner = RealTimeLearner()
            
            # 🆕 시그널-매매 연결 시스템
            self.signal_trade_connector = SignalTradeConnector()
        except ImportError:
            # 필수 헬퍼 클래스들에 대해 기본값 또는 None 처리
            pass
        
        self.strategy_weights = {}
        self.pattern_performance = {}
        
        # 🆕 Thompson Sampling 학습기 (Closed Loop Learning)
        try:
            from trade.core.thompson import ThompsonSamplingLearner
            self.thompson_sampler = ThompsonSamplingLearner(db_path=STRATEGIES_DB_PATH)
        except Exception as e:
            print(f"⚠️ ThompsonSamplingLearner 초기화 실패: {e}")
            self.thompson_sampler = None
        
        print("🚀 진화형 AI 시그널 셀렉터 초기화 완료")
        self.min_signal_score = 0.02  # 0.03 -> 0.02 (보수성 완화)
        
        # 🆕 학습 기반 임계값 설정
        self.use_learning_based_thresholds = True
        self.learning_feedback = None
        self.min_confidence = 0.2  # 0.5 -> 0.2 (시그널 희석 고려하여 완화)
        
        # 🆕 통합 분석기 추가 (rl_pipeline 의존성 제거)
        self.integrated_analyzer = None
        try:
            from trade.core.data_utils import get_integrated_analyzer
            self.integrated_analyzer = get_integrated_analyzer()
            print("✅ RL Pipeline 통합 분석기 로드 완료")
        except Exception as e:
            print(f"⚠️ 통합 분석기 로드 실패: {e}")
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
        self._supervisor_cache = {}  # 🆕 MetaCognitiveSupervisor 캐시 (속도 최적화)
        import threading
        self._cache_lock = threading.Lock()  # 🆕 캐시 접근용 락
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
        try:
            from signal_selector.scoring import StrategyScoreCalculator
            self._strategy_calculator = StrategyScoreCalculator()
        except Exception as e:
            print(f"⚠️ StrategyScoreCalculator를 로드할 수 없습니다: {e}")
            self._strategy_calculator = None
        
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
    
    def _categorize_volume(self, volume_ratio: float) -> str:
        """거래량 비율 범주화 (기존 호환성 유지)"""
        return self._categorize_volume_enhanced(volume_ratio)
    
    def _categorize_structure(self, structure_score: float) -> str:
        """구조 점수 범주화 (기존 호환성 유지)"""
        return self._categorize_structure_enhanced(structure_score)
    
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
    
    def determine_action(self, signal_score: float, confidence: float, coin: str = None, interval: str = None) -> SignalAction:
        """순수 시그널 기반 액션 결정 (보유 정보 없음)"""
        try:
            # 🆕 학습 기반 임계값 조정 (캔들 신뢰도 연동 포함)
            min_confidence = self.get_learning_based_confidence_threshold()
            
            # ScoringMixin에 정의된 메서드 호출
            if hasattr(self, 'get_learning_based_signal_score_threshold'):
                min_signal_score = self.get_learning_based_signal_score_threshold(coin, interval)
            else:
                min_signal_score = self.min_signal_score
            
            # 🆕 매수 조건 (동적 임계값 적용)
            if signal_score >= min_signal_score and confidence >= min_confidence:
                return SignalAction.BUY
            
            # 🆕 매도 조건 (시그널 점수가 매우 낮을 때)
            if signal_score <= -min_signal_score:
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
    
    def update_learning_feedback(self, feedback: Dict):
        """가상매매 학습기로부터 피드백 받기"""
        self.learning_feedback = feedback
        print(f"🔄 학습 피드백 업데이트: 승률={feedback.get('win_rate', 0):.2f}, 총거래={feedback.get('total_trades', 0)}개")
        print(f"   새로운 임계값: 신뢰도={self.get_learning_based_confidence_threshold():.2f}, 시그널점수={self.get_learning_based_signal_score_threshold():.3f}")
    

    
    def _discretize_volume(self, volume_ratio: float) -> str:
        """거래량 비율을 이산화 (None-Safe)"""
        if volume_ratio is None: return 'normal'
        try:
            val = float(volume_ratio)
            if val < 0.5:
                return 'low'
            elif val < 1.5:
                return 'normal'
            else:
                return 'high'
        except:
            return 'normal'
    
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
                ('market_adaptation_bonus', 'REAL DEFAULT 1.0'),
                ('target_price', 'REAL DEFAULT 0.0'),  # 🆕 예상 목표가
                ('source_type', "TEXT DEFAULT 'quant'")  # 🆕 소스 타입
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
    
    def get_recent_candles(self, coin: str, interval: str, limit: int = 30) -> pd.DataFrame:
        """🚀 최근 캔들 데이터 조회 (히스토리 분석용)"""
        try:
            # 🚀 캐시된 데이터 확인
            cache_key = f"recent_candles_{coin}_{interval}_{limit}_{int(time.time() // 60)}"  # 1분 캐시
            cached_data = self.get_cached_data(cache_key, max_age=60)
            if cached_data is not None:
                return cached_data
            
            conn = sqlite3.connect(CANDLES_DB_PATH)
            try:
                # 필요한 컬럼만 조회
                df = pd.read_sql(f"""
                    SELECT timestamp, open, high, low, close, volume,
                           rsi, macd, macd_signal, volume_ratio, 
                           bb_upper, bb_middle, bb_lower, adx
                    FROM candles
                    WHERE symbol = ? AND interval = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, conn, params=(coin, interval, limit))
                
                if not df.empty:
                    # 시간순 정렬 (과거 -> 현재)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    self.set_cached_data(cache_key, df)
                    return df
                return pd.DataFrame()
            finally:
                conn.close()
                
        except Exception as e:
            print(f"⚠️ 최근 캔들 조회 오류 ({coin}/{interval}): {e}")
            return pd.DataFrame()

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
                    WHERE symbol = ? AND interval = ?
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
    
    
    def set_current_coin(self, coin: str):
        """현재 처리 중인 코인 설정 (AI 모델 로드용)"""
        if hasattr(self, 'current_coin') and self.current_coin != coin:
            self.current_coin = coin
            # 코인이 바뀌면 해당 코인의 전용 모델 로드 시도
            if AI_MODEL_AVAILABLE:
                self._load_ai_model()
        else:
            self.current_coin = coin

    def _determine_final_action(self, action_votes: Dict[str, int], action_scores: Dict[str, float], final_score: float, coin: str = None, interval: str = None) -> str:
        """최종 액션 결정 (투표 기반 + 점수 기반) - 자율 임계값 적용"""
        try:
            # 🆕 동적 임계값 가져오기 (0.30 -> 0.12로 현실화하여 BUY 기회 확대)
            min_score = 0.12
            if hasattr(self, 'get_learning_based_signal_score_threshold'):
                min_score = self.get_learning_based_signal_score_threshold(coin, interval)
                # 학습 임계값이 너무 높으면(0.3 이상) 강제로 0.15 정도로 캡핑하여 매매 기회 확보
                min_score = min(min_score, 0.15)

            # 🎯 [보수성 완화] 강력한 점수가 있을 경우 투표보다 우선시
            # 기존 1.3배Multiplier는 현재 점수 분포에 비해 너무 가혹하므로 제거
            if final_score >= min_score:
                return 'buy'
            elif final_score <= -min_score:
                return 'sell'

            # 🎯 투표 기반 우선순위
            max_votes = max(action_votes.values())
            most_voted_actions = [action for action, votes in action_votes.items() if votes == max_votes]
            
            if len(most_voted_actions) == 1:
                # 단일 최다 투표 액션
                if most_voted_actions[0] == 'hold':
                    # HOLD가 많더라도 점수가 임계값의 70%를 넘으면 공격적으로 BUY 검토
                    if final_score >= min_score * 0.7: return 'buy'
                    if final_score <= -min_score * 0.7: return 'sell'
                return most_voted_actions[0]
            elif len(most_voted_actions) > 1:
                # 동점인 경우 점수 기반 결정
                best_action = max(most_voted_actions, key=lambda x: action_scores.get(x, 0))
                return best_action
            else:
                # 투표가 없는 경우 점수 기반 결정
                if final_score >= min_score:
                    return 'buy'
                elif final_score <= -min_score:
                    return 'sell'
                else:
                    return 'hold'
                    
        except Exception as e:
            print(f"⚠️ 최종 액션 결정 실패: {e}")
            return 'hold'
    
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
    
    def _get_latest_price(self, coin: str) -> float:
        """최신 가격 조회"""
        try:
            with sqlite3.connect(CANDLES_DB_PATH) as conn:
                # 여러 인터벌에서 최신 가격 조회
                intervals = ['15m', '30m', '240m', '1d']
                
                for interval in intervals:
                    query = """
                    SELECT close FROM candles 
                    WHERE symbol = ? AND interval = ? 
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
    
    def _test_synergy_learning_integration(self):
        """시너지 학습 통합 테스트 (비활성화 - 불필요한 테스트)"""
        # 시그널 계산이 완료된 후에는 시너지 학습 테스트가 불필요함
        print("ℹ️ 시너지 학습 테스트는 비활성화됨 (시그널 계산 완료 후 불필요)")
        return

# ============================================================================
# 🆕 전략 점수 계산기 클래스 (리팩토링)
# ============================================================================


