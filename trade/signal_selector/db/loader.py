"""
db_loader 관련 Mixin 클래스
SignalSelector의 db_loader 기능을 담당합니다.
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

# signal_selector 내부 모듈
try:
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )
except ImportError:
    # 직접 실행 시 경로 추가
    _current = os.path.dirname(os.path.abspath(__file__))
    _signal_selector = os.path.dirname(_current)
    _trade = os.path.dirname(_signal_selector)
    sys.path.insert(0, _trade)
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )

# 헬퍼 클래스 import (core에서만 필요)
try:
    from signal_selector.helpers import (
        ContextualBandit, RegimeChangeDetector, ExponentialDecayWeight,
        BayesianSmoothing, ActionSpecificScorer, ContextFeatureExtractor,
        OutlierGuardrail, EvolutionEngine, ContextMemory, RealTimeLearner,
        SignalTradeConnector
    )
except ImportError:
    pass  # 헬퍼가 필요없는 Mixin에서는 무시


class DBLoaderMixin:
    """
    DBLoaderMixin - db_loader 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

    def _load_coin_volatility_profiles(self):
        """🆕 모든 코인의 변동성 프로파일 로드 (DB에서 동적으로 조회)"""
        try:
            # 🆕 데이터베이스에서 실제 존재하는 코인 목록 동적 조회
            coins = []
            try:
                with sqlite3.connect(CANDLES_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT DISTINCT symbol as coin 
                        FROM candles 
                        WHERE symbol IS NOT NULL
                        ORDER BY symbol
                    """)
                    coins = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                print(f"⚠️ 코인 목록 조회 실패: {e}")
                # 폴백: 빈 리스트 (나중에 개별 코인 조회 시 자동으로 처리됨)
                coins = []
            
            if not coins:
                print("ℹ️ 변동성 프로파일 로드할 코인이 없습니다 (데이터 수집 중일 수 있음)")
                return
            
            print(f"📊 {len(coins)}개 코인의 변동성 프로파일 로드 중...")
            
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
    
    def _resolve_db_path(self, base_path, coin=None, is_common=True):
        """DB 경로 해석 (디렉토리 모드 지원)"""
        if os.path.isdir(base_path):
            if is_common:
                # 공용 DB (common_strategies.db)
                return os.path.join(base_path, "common_strategies.db")
            elif coin:
                # 개별 코인 DB
                return os.path.join(base_path, f"{coin.lower()}_strategies.db")
            else:
                # 코인 지정 안됨 + 디렉토리 모드 -> 공용 DB 반환
                return os.path.join(base_path, "common_strategies.db")
        return base_path

    def _load_reliability_scores(self) -> Dict[str, float]:
        """신뢰도 점수 로드 (reliability_scores 테이블)"""
        try:
            # 🆕 DB 경로 설정 (절대 경로 사용 + 디렉토리 모드 지원)
            try:
                from signal_selector.config import STRATEGIES_DB_PATH
                base_path = STRATEGIES_DB_PATH
            except ImportError:
                # 폴백: DATA_STORAGE_PATH 환경변수 사용 (디렉토리 모드 기본)
                data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                base_path = os.getenv('STRATEGY_DB_PATH', os.path.join(data_storage, "learning_strategies"))
            
            learning_db_path = self._resolve_db_path(base_path, is_common=True)
            
            # 🆕 파일 존재 여부 확인
            if not os.path.exists(learning_db_path):
                logger.debug(f"ℹ️ 신뢰도 점수 DB 파일이 없습니다: {learning_db_path} (정상 - 아직 데이터 없음)")
                return {}
            
            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            # 🆕 테이블 존재 여부 먼저 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='reliability_scores'
            """)
            if not cursor.fetchone():
                conn.close()
                logger.debug(f"ℹ️ reliability_scores 테이블이 없습니다 (정상 - 아직 데이터 없음)")
                return {}
            
            cursor.execute("""
                SELECT strategy_id, reliability_score 
                FROM reliability_scores 
                WHERE reliability_score > 0 AND strategy_id IS NOT NULL
            """)
            
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = float(row[1])
            
            conn.close()
            return results
            
        except Exception as e:
            # 🆕 "unable to open database file" 오류는 조용히 무시
            if "unable to open database file" in str(e).lower():
                logger.debug(f"ℹ️ 신뢰도 점수 DB 접근 불가 (정상 - 아직 데이터 없음): {e}")
                return {}
            logger.warning(f"⚠️ 신뢰도 점수 로드 실패: {e}")
            return {}
    
    def _load_learning_quality_scores(self) -> Dict[str, float]:
        """학습 품질 점수 로드 (simulation_results 테이블에서 조회)"""
        try:
            # learning_strategies.db에서 학습 품질 점수 로드
            from signal_selector.config import STRATEGIES_DB_PATH
            learning_db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(learning_db_path):
                return {}

            conn = sqlite3.connect(learning_db_path)
            cursor = conn.cursor()
            
            # 🆕 simulation_results 테이블에서 learning_quality_score 조회
            # strategy_learning_history 테이블은 존재하지 않으므로 simulation_results 사용
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='simulation_results'
            """)
            if not cursor.fetchone():
                conn.close()
                # 테이블이 없으면 조용히 빈 딕셔너리 반환
                return {}
            
            # 🆕 simulation_results 테이블에서 strategy_id와 learning_quality_score 조회
            # 최신 결과만 사용 (같은 strategy_id가 여러 개면 최신 것 사용)
            cursor.execute("""
                SELECT strategy_id, learning_quality_score 
                FROM (
                    SELECT strategy_id, learning_quality_score,
                           ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY created_at DESC) as rn
                    FROM simulation_results 
                    WHERE learning_quality_score > 0 AND strategy_id IS NOT NULL
                )
                WHERE rn = 1
            """)
            
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = float(row[1])
            
            conn.close()
            return results
            
        except Exception as e:
            # 🆕 테이블이 없는 경우는 조용히 무시 (경고 제거)
            if "no such table" in str(e).lower():
                return {}
            # 🆕 윈도우 함수 미지원 시 단순 조회로 fallback
            if "window" in str(e).lower() or "ROW_NUMBER" in str(e).upper():
                try:
                    conn = sqlite3.connect(learning_db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT strategy_id, MAX(learning_quality_score) as learning_quality_score
                        FROM simulation_results 
                        WHERE learning_quality_score > 0 AND strategy_id IS NOT NULL
                        GROUP BY strategy_id
                    """)
                    results = {}
                    for row in cursor.fetchall():
                        results[row[0]] = float(row[1])
                    conn.close()
                    return results
                except Exception as fallback_e:
                    return {}
            logger.warning(f"⚠️ 학습 품질 점수 로드 실패: {e}")
            return {}
    
    def _load_global_strategy_mapping(self) -> Dict[str, str]:
        """글로벌 전략 매핑 로드"""
        try:
            # learning_strategies.db에서 글로벌 전략 매핑 로드
            from signal_selector.config import STRATEGIES_DB_PATH
            learning_db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(learning_db_path):
                return {}

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
            # learning_strategies.db에서 Walk-Forward 성능 데이터 로드
            from signal_selector.config import STRATEGIES_DB_PATH
            learning_db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(learning_db_path):
                return {}

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
            # learning_strategies.db에서 레짐별 커버리지 데이터 로드
            from signal_selector.config import STRATEGIES_DB_PATH
            learning_db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(learning_db_path):
                return {}

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
    
    def _load_cross_coin_context(self):
        """크로스 코인 학습 컨텍스트 로드"""
        try:
            if CROSS_COIN_AVAILABLE:
                # self.cross_coin_context = load_global_integrated_results()  # 🆕 임시 비활성화
                self.cross_coin_context = {}
                print(f"🚀 크로스 코인 학습 컨텍스트 로드 완료")
            else:
                # 크로스 코인 학습은 의도적으로 비활성화됨 (복잡한 의존성 문제로 인해 간소화)
                self.cross_coin_context = {}
                # 조용히 처리 (상태 확인 시 정보 메시지 출력)
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
            try:
                # 경로 수정: rl_pipeline.advanced_learning_systems -> rl_pipeline.analysis.advanced_learning_systems
                # 클래스명 수정: EnsembleLearningSystem -> JAXEnsembleLearningSystem
                from rl_pipeline.analysis.advanced_learning_systems import JAXEnsembleLearningSystem
                self.ensemble_learning_system = JAXEnsembleLearningSystem()
                print("✅ 앙상블 학습 시스템 로드 완료 (JAXEnsembleLearningSystem)")
            except ImportError as e:
                print(f"⚠️ 앙상블 학습 시스템 모듈을 찾을 수 없음: {e}")
                self.ensemble_learning_system = None
            except Exception as e:
                print(f"⚠️ 앙상블 학습 시스템 초기화 실패: {e}")
                self.ensemble_learning_system = None
            
            # 메타 학습 시스템 로드 (현재 미구현으로 비활성화)
            self.meta_learning_system = None
            # try:
            #     from rl_pipeline.analysis.advanced_learning_systems import MetaLearningSystem
            #     self.meta_learning_system = MetaLearningSystem()
            #     print("✅ 메타 학습 시스템 로드 완료")
            # except ImportError:
            #     self.meta_learning_system = None
            
            # 통합 고급 시스템 로드 (현재 미구현으로 비활성화)
            self.integrated_advanced_system = None
            # try:
            #     from rl_pipeline.analysis.advanced_learning_systems import IntegratedAdvancedSystem
            #     self.integrated_advanced_system = IntegratedAdvancedSystem(state_dim=50, action_dim=10)
            #     print("✅ 통합 고급 시스템 로드 완료")
            # except ImportError:
            #     self.integrated_advanced_system = None
            
        except Exception as e:
            print(f"⚠️ 고급 학습 시스템 로드 실패: {e}")
            self.ensemble_learning_system = None
            self.meta_learning_system = None
            self.integrated_advanced_system = None

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
            from signal_selector.config import STRATEGIES_DB_PATH
            
            # 🔧 TARGET_COINS 환경변수에서 대상 코인 목록 가져오기
            target_coins_str = os.getenv('TARGET_COINS', '')
            target_coins = [c.strip().upper() for c in target_coins_str.split(',') if c.strip()] if target_coins_str else []
            
            # 🆕 디렉토리 모드 지원: 개별 코인 DB 파일 찾기 (TARGET_COINS 필터링)
            db_files = []
            
            # 🔥 [Fix] STRATEGIES_DB_PATH가 파일 경로(common_strategies.db)인 경우, 디렉토리로 변환하여 스캔
            scan_dir = STRATEGIES_DB_PATH
            if not os.path.isdir(scan_dir) and scan_dir.endswith('.db'):
                scan_dir = os.path.dirname(scan_dir)
                
            if os.path.isdir(scan_dir):
                # *_strategies.db 패턴의 파일들 중 TARGET_COINS에 해당하는 것만 찾음
                for f in os.listdir(scan_dir):
                    if f.endswith('_strategies.db') and f != 'common_strategies.db':
                        # 파일명에서 코인 이름 추출 (예: btc_strategies.db -> BTC)
                        coin_name = f.replace('_strategies.db', '').upper()
                        
                        # TARGET_COINS가 비어있으면 모든 코인 로드, 아니면 필터링
                        if not target_coins or coin_name in target_coins:
                            db_files.append(os.path.join(scan_dir, f))
            else:
                # 기존 단일 파일 모드 (디렉토리가 아니고 .db로 끝나지도 않는 경우, 혹은 존재하지 않는 경우)
                if os.path.exists(STRATEGIES_DB_PATH):
                    db_files.append(STRATEGIES_DB_PATH)
            
            if not db_files:
                if target_coins:
                    print(f"ℹ️ TARGET_COINS({', '.join(target_coins)})에 해당하는 전략 DB 파일이 없습니다.")
                else:
                    print("⚠️ 로드할 전략 DB 파일이 없습니다.")
                return

            print(f"📊 {len(db_files)}개 전략 DB 파일 로드 시작")

            for db_path in db_files:
                try:
                    with sqlite3.connect(db_path) as conn:
                        # 🚀 DB 테이블 확인
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        available_tables = [row[0] for row in cursor.fetchall()]
                        
                        quality_df = pd.DataFrame()
                        
                        # 테이블 우선순위: coin_strategies > learned_strategies > global_strategies
                        if 'coin_strategies' in available_tables:
                            # 🚀 [Integrated] 모든 유의미한 전략 조회 (등급순 정렬)
                            # S, A, B 등급 우선, 그 다음 수익률 순
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
                                AND (lifecycle_status = 'ACTIVE' OR lifecycle_status IS NULL)
                                -- 수익이 0 이상인 전략은 모두 로드 후보 (등급 관계 없음)
                                AND COALESCE(profit, 0) >= 0
                                ORDER BY 
                                    CASE COALESCE(quality_grade, 'F')
                                        WHEN 'S' THEN 0
                                        WHEN 'A' THEN 1
                                        WHEN 'B' THEN 2
                                        WHEN 'C' THEN 3
                                        WHEN 'D' THEN 4
                                        ELSE 5
                                    END ASC,
                                    score DESC
                                LIMIT 2000
                            """, conn)
                                
                        elif 'learned_strategies' in available_tables:
                             quality_df = pd.read_sql("""
                                SELECT coin as symbol, interval, profit, win_rate, trades_count, strategy_id,
                                       strategy_type, main_indicator, risk_level, score
                                FROM learned_strategies
                                WHERE (profit > 0 OR profit IS NULL) AND (trades_count >= 1 OR trades_count IS NULL) AND (win_rate >= 0.2 OR win_rate IS NULL)
                                ORDER BY coin, interval, COALESCE(score, 0.5) DESC
                            """, conn)
                        elif 'strategies' in available_tables:
                             # 🆕 리그 시스템 지원: league 컬럼 확인
                             cursor.execute("PRAGMA table_info(strategies)")
                             cols = [c[1] for c in cursor.fetchall()]
                             has_league = 'league' in cols
                             
                             # 🔥 MFE/MAE 통계 테이블 존재 확인 (방어 로직)
                             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_label_stats'")
                             has_mfe_stats = cursor.fetchone() is not None
                             
                             if has_mfe_stats:
                                 # 🔥 MFE/MAE 완전 전환: EntryScore 기반 정렬 및 필터링
                                 league_filter = "AND (s.league = 'major' OR s.league IS NULL)" if has_league else ""
                                 
                                 quality_df = pd.read_sql(f"""
                                    SELECT 
                                        s.symbol, s.interval, 
                                        COALESCE(s.profit, 0.0) as profit, 
                                        COALESCE(s.win_rate, 0.5) as win_rate,
                                        COALESCE(s.trades_count, 0) as trades_count, 
                                        s.id as strategy_id,
                                        'learned' as strategy_type, 'multi' as main_indicator, 'medium' as risk_level,
                                        COALESCE(s.score, 0.5) as score,
                                        s.quality_grade,
                                        ls.rmax_p90, ls.rmin_p10, ls.n_signals,
                                        (COALESCE(ls.rmax_p90, 0) - 1.5 * ABS(COALESCE(ls.rmin_p10, 0))) as entry_score,
                                        ABS(COALESCE(ls.rmin_p10, 0)) as risk_score
                                    FROM strategies s
                                    LEFT JOIN strategy_label_stats ls 
                                        ON s.id = ls.strategy_id 
                                        AND s.symbol = ls.coin 
                                        AND s.interval = ls.interval
                                    WHERE 
                                        (ls.rmax_p90 IS NULL OR (ls.rmax_p90 - 1.5 * ABS(COALESCE(ls.rmin_p10, 0))) >= -0.01)
                                        AND COALESCE(s.profit, 0) >= 0
                                        {league_filter}
                                    ORDER BY 
                                        CASE WHEN ls.rmax_p90 IS NOT NULL THEN 0 ELSE 1 END ASC,
                                        (COALESCE(ls.rmax_p90, 0) - 1.5 * ABS(COALESCE(ls.rmin_p10, 0))) DESC,
                                        s.profit DESC
                                    LIMIT 2000
                                """, conn)
                             else:
                                 # 🔧 방어 로직: MFE/MAE 테이블 없으면 기존 방식으로 fallback
                                 where_clause = "WHERE COALESCE(profit, 0) >= 0"
                                 if has_league:
                                     where_clause += " AND (league = 'major' OR league IS NULL)"
                                 
                                 quality_df = pd.read_sql(f"""
                                    SELECT symbol, interval, profit, win_rate, trades_count, id as strategy_id,
                                           'learned' as strategy_type, 'multi' as main_indicator, 'medium' as risk_level,
                                           COALESCE(score, 0.5) as score,
                                           quality_grade,
                                           NULL as rmax_p90, NULL as rmin_p10, NULL as n_signals,
                                           NULL as entry_score, NULL as risk_score
                                    FROM strategies
                                    {where_clause}
                                    ORDER BY 
                                        CASE COALESCE(quality_grade, 'F')
                                            WHEN 'S' THEN 0 WHEN 'A' THEN 1 WHEN 'B' THEN 2
                                            WHEN 'C' THEN 3 WHEN 'D' THEN 4 ELSE 5
                                        END ASC,
                                        score DESC
                                    LIMIT 2000
                                """, conn)
                        
                        if not quality_df.empty:
                            # 품질 기반 전략 로드
                            mfe_strategy_count = 0
                            for _, row in quality_df.iterrows():
                                strategy_key = f"{row['symbol']}_{row['interval']}"
                                current_score = row['score']
                                
                                # 🆕 리스트 형태로 저장하여 여러 전략 지원 (레짐/상황별)
                                if strategy_key not in self.coin_specific_strategies:
                                    self.coin_specific_strategies[strategy_key] = []
                                elif isinstance(self.coin_specific_strategies[strategy_key], dict):
                                    # 기존 딕셔너리를 리스트로 변환 (하위 호환성)
                                    self.coin_specific_strategies[strategy_key] = [self.coin_specific_strategies[strategy_key]]
                                
                                # 🔥 MFE/MAE 지표 안전하게 추출 (None 처리)
                                entry_score = row.get('entry_score') if 'entry_score' in row.index else None
                                risk_score = row.get('risk_score') if 'risk_score' in row.index else None
                                rmax_p90 = row.get('rmax_p90') if 'rmax_p90' in row.index else None
                                rmin_p10 = row.get('rmin_p10') if 'rmin_p10' in row.index else None
                                n_signals = row.get('n_signals') if 'n_signals' in row.index else None
                                
                                if entry_score is not None and not pd.isna(entry_score):
                                    mfe_strategy_count += 1
                                
                                # 모든 전략 추가 (덮어쓰기 아님)
                                strategy_data = {
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
                                    'quality_grade': row.get('quality_grade', 'B'),
                                    'strategy_json': '{}',
                                    # 🔥 MFE/MAE 지표 추가
                                    'entry_score': entry_score if not pd.isna(entry_score) else None,
                                    'risk_score': risk_score if not pd.isna(risk_score) else None,
                                    'rmax_p90': rmax_p90 if not pd.isna(rmax_p90) else None,
                                    'rmin_p10': rmin_p10 if not pd.isna(rmin_p10) else None,
                                    'n_signals': int(n_signals) if n_signals and not pd.isna(n_signals) else None
                                }
                                self.coin_specific_strategies[strategy_key].append(strategy_data)
                except Exception as e:
                    # 파일이 없는 경우 정보 메시지로 표시 (첫 실행 시 정상)
                    if "unable to open database file" in str(e).lower():
                        print(f"ℹ️ {os.path.basename(db_path)}: 아직 학습된 전략이 없습니다 (run_learning.py 실행 필요)")
                    else:
                        print(f"⚠️ DB 파일 로드 실패 ({os.path.basename(db_path)}): {e}")
                    continue
                
            # 🚀 [Log] 코인별 전략 로드 상세 현황 출력
            total_strategy_count = 0
            total_mfe_count = 0
            if self.coin_specific_strategies:
                coin_counts = {}
                for key, strategies in self.coin_specific_strategies.items():
                    # 리스트인 경우와 딕셔너리인 경우 모두 처리
                    if isinstance(strategies, list):
                        count = len(strategies)
                        if count > 0:
                            symbol = strategies[0].get('symbol', 'UNKNOWN')
                            # 🔥 MFE/MAE 전략 개수 집계
                            total_mfe_count += sum(1 for s in strategies if s.get('entry_score') is not None)
                        else:
                            continue
                    else:
                        count = 1
                        symbol = strategies.get('symbol', 'UNKNOWN')
                        if strategies.get('entry_score') is not None:
                            total_mfe_count += 1
                        
                    coin_counts[symbol] = coin_counts.get(symbol, 0) + count
                    total_strategy_count += count
                
                # 상위 5개 또는 전체 출력
                count_str_list = [f"{coin}: {count}개" for coin, count in sorted(coin_counts.items())]
                if len(count_str_list) > 10:
                    print(f"📊 코인별 전략 현황 (총 {len(coin_counts)}개 코인): {', '.join(count_str_list[:10])} ...")
                else:
                    print(f"📊 코인별 전략 현황: {', '.join(count_str_list)}")

            print(f"✅ 총 {total_strategy_count}개 전략 로드 완료 (MFE/MAE: {total_mfe_count}개, 기존: {total_strategy_count - total_mfe_count}개)")
                
        except Exception as e:
            print(f"ℹ️ 코인별 전략 로드 전체 실패: {e}")
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
            try:
                from signal_selector.config import TRADING_SYSTEM_DB_PATH
                trading_db_path = TRADING_SYSTEM_DB_PATH
            except ImportError:
                # 폴백: DATA_STORAGE_PATH 환경변수 사용
                data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                trading_db_path = os.getenv('TRADING_DB_PATH', os.path.join(data_storage, "trading_system.db"))
            
            # 🆕 DB 파일 존재 여부 확인
            if not os.path.exists(trading_db_path):
                print(f"ℹ️ trading_system.db 파일이 없습니다: {trading_db_path} (정상 - 아직 데이터 없음)")
                return

            with sqlite3.connect(trading_db_path) as conn:
                cursor = conn.cursor()
                
                # 🆕 테이블 존재 여부 확인
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('completed_trades', 'signals')
                """)
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                if 'completed_trades' not in existing_tables:
                    print(f"ℹ️ completed_trades 테이블이 없습니다 (정상 - 아직 거래 이력 없음)")
                    return
                
                if 'signals' not in existing_tables:
                    print(f"ℹ️ signals 테이블이 없습니다 (정상 - 아직 시그널 이력 없음)")
                    return
                
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

                cursor.execute(query)
                trades = cursor.fetchall()

                if not trades:
                    # 첫 실행 시 정상 - 거래가 완료되면 자동으로 DNA 학습 시작
                    print("ℹ️ DNA 패턴 학습 대기 중 (거래 완료 후 자동 학습됨)")
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
                        # 기존 전략들에 DNA 패턴 추가 (리스트 지원)
                        strategies = self.coin_specific_strategies[strategy_key]
                        if isinstance(strategies, list):
                            for strategy in strategies:
                                strategy.update(representative_pattern)
                        elif isinstance(strategies, dict):
                             strategies.update(representative_pattern)
                        patterns_added += 1
                    else:
                        # 새로운 전략 생성 (DNA 패턴만 포함)
                        coin, interval = strategy_key.split('_')
                        new_strategy = {
                            'symbol': coin,
                            'interval': interval,
                            'profit': sum(p['profit_pct'] for p in top_patterns) / len(top_patterns),
                            'win_rate': 1.0,  # 성공 거래만 사용했으므로
                            'trades_count': len(patterns),
                            **representative_pattern
                        }
                        # 리스트로 초기화
                        self.coin_specific_strategies[strategy_key] = [new_strategy]
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

    def _load_deep_analysis_results(self) -> Optional[Dict]:
        """심화 분석 결과 로드"""
        try:
            # learning_strategies.db에서 심화 분석 결과 로드
            from signal_selector.config import STRATEGIES_DB_PATH
            learning_db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(learning_db_path):
                return None

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

    def _load_dna_analysis_results(self, coin: str = None) -> Dict[str, Any]:
        """DNA 분석 결과 로드 - dna_summary_for_signals 테이블에서 로드"""
        try:
            import sqlite3
            from signal_selector.config import STRATEGIES_DB_PATH
            # 공용 DB에서 로드 (dna_summary는 보통 공용 DB에 저장됨)
            db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(db_path):
                return {}

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 🆕 learning_strategies.db의 dna_summary_for_signals 테이블에서 로드
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
                        print(f"✅ DNA 분석 결과 로드 완료: {coin or '전체'} (learning_strategies.db)")
                    
                    return dna_features
                else:
                    if self.debug_mode:
                        print(f"⚠️ DNA 요약 데이터 없음: {coin or '전체'}")
                    return {}
                    
        except Exception as e:
            if self.debug_mode:
                print(f"❌ DNA 분석 결과 로드 실패: {e}")
            return {}
    
    def _load_learning_quality_data(self) -> Optional[Dict]:
        """학습 품질 데이터 로드"""
        try:
            # 🆕 학습 품질 평가 결과를 DB에서 로드
            from signal_selector.config import STRATEGIES_DB_PATH
            db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
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
            # 🆕 데이터베이스에서 실제 존재하는 코인 목록 동적 조회
            major_coins = []
            try:
                with sqlite3.connect(CANDLES_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT DISTINCT symbol as coin 
                        FROM candles 
                        WHERE symbol IS NOT NULL
                        ORDER BY symbol
                        LIMIT 20  -- 성능을 위해 상위 20개만 미리 로드
                    """)
                    major_coins = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                if self.debug_mode:
                    logger.debug(f"⚠️ 코인 목록 조회 실패: {e}")
                # 폴백: 빈 리스트 (나중에 개별 코인 조회 시 자동으로 처리됨)
                major_coins = []
            
            intervals = ['15m', '30m', '240m', '1d']
            
            if not major_coins:
                if self.debug_mode:
                    logger.debug("ℹ️ 통합 분석 결과 로드할 코인이 없습니다 (데이터 수집 중일 수 있음)")
                return
            
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
    
    def load_fractal_analysis_results(self):
        """프랙탈 분석 결과 로드 (Signal Selector에서 활용)"""
        self.fractal_analysis_results = {}
        
        try:
            from signal_selector.config import STRATEGIES_DB_PATH
            db_path = self._resolve_db_path(STRATEGIES_DB_PATH, is_common=True)
            
            if not os.path.exists(db_path):
                return

            with sqlite3.connect(db_path) as conn:
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
    
    def _load_synergy_patterns(self):
        """시너지 학습 결과 로드 (강화된 에러 처리)"""
        # 🔧 이미 로드되어 있으면 스킵 (중복 로그 방지)
        if hasattr(self, 'synergy_patterns') and self.synergy_patterns:
            return self.synergy_patterns
        
        try:
            # 시너지 패턴 테이블 존재 확인 및 생성
            from signal_selector.config import STRATEGIES_DB_PATH
            
            # 🔧 디렉토리 모드 지원: 폴더면 common_strategies.db 사용
            db_path = STRATEGIES_DB_PATH
            if os.path.isdir(db_path):
                db_path = os.path.join(db_path, 'common_strategies.db')
            
            if not os.path.exists(db_path):
                print(f"ℹ️ 시너지 패턴 DB 파일이 없습니다: {db_path} (정상 - 아직 학습 데이터 없음)")
                self.synergy_patterns = self._get_default_synergy_patterns()
                return
            
            with sqlite3.connect(db_path) as conn:
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
    

