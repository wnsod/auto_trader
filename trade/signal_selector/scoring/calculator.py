"""
scoring 관련 Mixin 클래스
SignalSelector의 scoring 기능을 담당합니다.
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


class ScoringMixin:
    """
    ScoringMixin - scoring 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

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

    def _calculate_fast_volume_ratio(self, candle: pd.Series) -> float:
        """🚀 빠른 거래량 비율 계산"""
        try:
            volume = candle.get('volume', 0.0)
            return 1.0  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 거래량 비율 계산 실패: {e}")
            return 1.0

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
            
            # 데이터베이스에서 글로벌 학습 결과 조회 (learning_strategies.db)
            try:
                from signal_selector.config import STRATEGIES_DB_PATH
                db_path = STRATEGIES_DB_PATH
            except ImportError:
                # 폴백: 환경변수 사용
                data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                db_path = os.getenv('STRATEGY_DB_PATH', os.path.join(data_storage, 'learning_strategies'))
            
            # 🔧 디렉토리 모드 지원: 폴더인 경우 common_strategies.db 사용
            if os.path.isdir(db_path):
                db_path = os.path.join(db_path, 'common_strategies.db')
            
            if not os.path.exists(db_path):
                return 0.5
                
            with sqlite3.connect(db_path) as conn:
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
            
            # 데이터베이스에서 심볼별 튜닝 결과 조회 (learning_strategies.db)
            try:
                from signal_selector.config import STRATEGIES_DB_PATH
                db_path = STRATEGIES_DB_PATH
            except ImportError:
                # 폴백: 환경변수 사용
                data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                db_path = os.getenv('STRATEGY_DB_PATH', os.path.join(data_storage, 'learning_strategies'))
            
            # 🔧 디렉토리 모드 지원: 폴더인 경우 common_strategies.db 사용
            if os.path.isdir(db_path):
                db_path = os.path.join(db_path, 'common_strategies.db')
            
            if not os.path.exists(db_path):
                return 0.5
                
            with sqlite3.connect(db_path) as conn:
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

                # 🌍 DB 기반 동적 가중치 조정 (interval_profiles 우선, 레짐 fallback)
                coin_weight, global_weight = self._calculate_dynamic_weights(current_regime, coin=coin, interval=interval)
                
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
    
    def _calculate_dynamic_weights(self, regime: str, coin: str = None, interval: str = None) -> tuple:
        """🔥 코인 vs 글로벌 전략 동적 가중치 계산 (interval_profiles 기반)

        Args:
            regime: 시장 레짐 (fallback용)
            coin: 코인 이름 (예: 'BTC')
            interval: 인터벌 (예: '15m', '1d')

        Returns:
            tuple: (coin_weight, global_weight)
        """
        try:
            # 🔥 1순위: interval_profiles 기반 인터벌별 가중치 (최우선)
            if interval:
                try:
                    from rl_pipeline.core.interval_profiles import get_interval_role
                    
                    interval_role = get_interval_role(interval)
                    
                    if interval_role:
                        # 인터벌별 역할에 따른 가중치 차별화
                        if interval_role == "Macro Regime":  # 1d: 시장 전체 흐름 중요
                            coin_weight, global_weight = (0.3, 0.7)  # 개별 30%, 글로벌 70%
                            if self.debug_mode:
                                print(f"🎯 [{interval}] {interval_role}: 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                        elif interval_role == "Trend Structure":  # 240m: 중기 추세
                            coin_weight, global_weight = (0.4, 0.6)  # 개별 40%, 글로벌 60%
                            if self.debug_mode:
                                print(f"🎯 [{interval}] {interval_role}: 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                        elif interval_role == "Micro Trend":  # 30m: 추세 확인
                            coin_weight, global_weight = (0.5, 0.5)  # 개별 50%, 글로벌 50%
                            if self.debug_mode:
                                print(f"🎯 [{interval}] {interval_role}: 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                        elif interval_role == "Execution":  # 15m: 매매 타이밍
                            coin_weight, global_weight = (0.7, 0.3)  # 개별 70%, 글로벌 30%
                            if self.debug_mode:
                                print(f"🎯 [{interval}] {interval_role}: 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                except ImportError:
                    if self.debug_mode:
                        print(f"⚠️ interval_profiles 모듈 없음, 다음 우선순위 사용")
                except Exception as ip_err:
                    if self.debug_mode:
                        print(f"⚠️ interval_profiles 로드 실패: {ip_err}, 다음 우선순위 사용")

            # 🔥 2순위: 변동성 그룹 기반 가중치 (기초적인 기반)
            if coin:
                try:
                    vol_group = self.get_coin_volatility_group(coin)
                    
                    if vol_group:
                        # 변동성 그룹별 가중치 차별화
                        if vol_group == 'LOW':  # BTC 등: 안정적이므로 코인별 특성 중요
                            coin_weight, global_weight = (0.75, 0.25)  # 개별 75%, 글로벌 25%
                            if self.debug_mode:
                                print(f"🎯 [{coin}] 변동성 그룹 '{vol_group}': 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                        elif vol_group == 'MEDIUM':  # ETH, BNB 등: 균형
                            coin_weight, global_weight = (0.60, 0.40)  # 개별 60%, 글로벌 40%
                            if self.debug_mode:
                                print(f"🎯 [{coin}] 변동성 그룹 '{vol_group}': 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                        elif vol_group == 'HIGH':  # ADA, SOL 등: 변동성이 크므로 시장 전체 흐름 중요
                            coin_weight, global_weight = (0.50, 0.50)  # 개별 50%, 글로벌 50%
                            if self.debug_mode:
                                print(f"🎯 [{coin}] 변동성 그룹 '{vol_group}': 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                        elif vol_group == 'VERY_HIGH':  # DOGE 등: 매우 불안정하므로 시장 전체 흐름 중요
                            coin_weight, global_weight = (0.40, 0.60)  # 개별 40%, 글로벌 60%
                            if self.debug_mode:
                                print(f"🎯 [{coin}] 변동성 그룹 '{vol_group}': 개별={coin_weight}, 글로벌={global_weight}")
                            return coin_weight, global_weight
                except Exception as vol_err:
                    if self.debug_mode:
                        print(f"⚠️ [{coin}] 변동성 그룹 로드 실패: {vol_err}, 다음 우선순위 사용")

            # 🔥 3순위: DB에서 코인별 동적 가중치 로드
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

            # 🔥 4순위: 레짐 기반 가중치 (fallback)
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
        """개별 코인 전략 점수 계산 (다중 전략 지원)"""
        try:
            strategies = self.coin_specific_strategies[strategy_key]
            
            # 단일 전략(딕셔너리)인 경우 리스트로 변환
            if isinstance(strategies, dict):
                strategies = [strategies]
            
            if not strategies:
                return 0.0
            
            # 🚀 현재 시장 상태 분석 (크로스 코인 학습용)
            current_dna = self._extract_current_dna_pattern_enhanced(coin, interval, candle)
            
            best_normalized_score = 0.0
            
            # 🆕 모든 전략을 평가하여 가장 높은 점수 사용 (앙상블 효과)
            for strategy in strategies:
                quality_grade = strategy.get('quality_grade', 'C')
                
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
                
                if normalized_score > best_normalized_score:
                    best_normalized_score = normalized_score
            
            return best_normalized_score
            
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
    
    def _calculate_performance_score(self, strategy: dict) -> float:
        """전략 성과 점수 계산 (기존 호환성 유지)"""
        return self._calculate_performance_score_enhanced(strategy)
    
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
                    # learning_strategies.db에서 요약 정보 조회
                    try:
                        from signal_selector.config import STRATEGIES_DB_PATH
                        learning_db_path = STRATEGIES_DB_PATH
                    except ImportError:
                        # 폴백: 환경변수 사용
                        data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                        learning_db_path = os.getenv('STRATEGY_DB_PATH', os.path.join(data_storage, 'learning_strategies'))
                    
                    # 🔧 디렉토리 모드 지원
                    if os.path.isdir(learning_db_path):
                        learning_db_path = os.path.join(learning_db_path, 'common_strategies.db')
                    
                    if not os.path.exists(learning_db_path):
                        pass  # 파일 없으면 빈 전략으로 진행
                    else:
                        pass  # 정상 진행
                        
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
                            # learning_strategies.db에서 직접 조회 (폴백)
                            strategies_db_path = None
                            try:
                                from signal_selector.config import STRATEGIES_DB_PATH
                                strategies_db_path = STRATEGIES_DB_PATH
                            except ImportError:
                                pass

                            if not strategies_db_path:
                                import os
                                # 환경변수 우선 사용 (폴백 경로도 환경변수 기반으로 수정)
                                default_storage = os.getenv('DATA_STORAGE_PATH', "/workspace/data_storage")
                                strategies_db_path = os.getenv('STRATEGY_DB_PATH', os.path.join(default_storage, "learning_strategies"))
                            
                            # 🆕 디렉토리 모드 지원: 폴더인 경우 해당 코인의 DB 파일 선택
                            real_db_path = strategies_db_path
                            if os.path.isdir(strategies_db_path):
                                # 코인별 DB 파일명 규칙: {coin_lower}_strategies.db
                                coin_db_name = f"{coin.lower()}_strategies.db"
                                real_db_path = os.path.join(strategies_db_path, coin_db_name)
                                
                                # 파일이 없으면 공용 DB 확인 (common_strategies.db)
                                if not os.path.exists(real_db_path):
                                    common_path = os.path.join(strategies_db_path, "common_strategies.db")
                                    if os.path.exists(common_path):
                                        real_db_path = common_path
                            
                            with sqlite3.connect(real_db_path) as strategies_conn:
                                strategies_cursor = strategies_conn.cursor()
                                
                                # Phase 2: strategy_grades를 Source of Truth로 우선 사용
                                strategies_cursor.execute("""
                                    SELECT
                                        cs.id, cs.rsi_min, cs.rsi_max, cs.volume_ratio_min, cs.volume_ratio_max,
                                        cs.macd_buy_threshold, cs.macd_sell_threshold,
                                        cs.profit, cs.win_rate, cs.quality_grade, cs.score,
                                        sg.grade_score, sg.total_return, sg.predictive_accuracy, sg.grade
                                    FROM coin_strategies cs
                                    LEFT JOIN strategy_grades sg
                                        ON cs.id = sg.strategy_id
                                        AND cs.coin = sg.coin
                                        AND cs.interval = sg.interval
                                    WHERE cs.coin = ? AND cs.interval = ?
                                    AND cs.quality_grade IN ('S', 'A', 'B')
                                    ORDER BY
                                        COALESCE(sg.grade_score, cs.score) DESC
                                    LIMIT 5
                                """, (coin, interval))

                                for row in strategies_cursor.fetchall():
                                    # Phase 2: grade_score 우선 사용, 없으면 coin_strategies의 스냅샷 사용
                                    grade_score = row[11]  # sg.grade_score
                                    total_return = row[12]  # sg.total_return
                                    predictive_accuracy = row[13]  # sg.predictive_accuracy

                                    # 성과 데이터: strategy_grades가 있으면 우선 사용
                                    profit = total_return if total_return is not None else (row[7] or 0.0)
                                    win_rate = predictive_accuracy if predictive_accuracy is not None else (row[8] or 0.0)
                                    score = grade_score if grade_score is not None else (row[10] or 0.5)

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
                                        'profit': profit,
                                        'win_rate': win_rate,
                                        'quality_grade': row[14] or row[9] or 'B',  # sg.grade 우선, 없으면 cs.quality_grade
                                        'score': score,
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
    
    def _calculate_target_price(self, candle: pd.Series, action: SignalAction, indicators: Dict, avg_profit_pct: float = 0.0) -> float:
        """예상 목표가 계산 (ATR 및 볼린저 밴드 + 🆕 학습된 평균 수익률 활용)
        
        🔧 개선사항:
        - 최소 기대 수익률 1.5% 보장 (슬리피지/수수료 커버)
        - 볼린저밴드 저항선 제한 완화 (참조만, 제한 안 함)
        - 횡보장에서도 적절한 목표가 설정
        """
        try:
            current_price = safe_float(candle.get('close'), 0.0)
            if current_price == 0:
                return 0.0
                
            atr = indicators.get('atr', 0.0)
            bb_upper = indicators.get('bb_upper', 0.0)
            bb_lower = indicators.get('bb_lower', 0.0)
            
            # 🔧 최소 변동성을 1.5%로 상향 (슬리피지/수수료 커버 + 여유)
            min_volatility = current_price * 0.015
            volatility = max(atr * 2.0, min_volatility)
            
            # 🆕 최소 기대 수익률 1.5% 보장용 목표가
            min_target_buy = current_price * 1.015
            min_target_sell = current_price * 0.985
            
            if action == SignalAction.BUY:
                # 1. 기술적 목표가: 볼린저 밴드 상단 또는 현재가 + 변동성
                tech_target = max(bb_upper, current_price + volatility) if bb_upper > current_price else current_price + volatility
                
                # 2. 학습 기반 목표가 반영
                if avg_profit_pct > 0:
                    # 학습된 수익률만큼 목표 설정
                    learned_target = current_price * (1 + avg_profit_pct / 100.0)
                    
                    # 🔧 개선: 볼린저밴드 제한 제거, 학습값과 기술값 중 더 높은 값 선택
                    # (적극적인 목표 설정으로 기회 확대)
                    if avg_profit_pct >= 1.0:
                        # 학습된 수익률이 충분하면 학습값 우선
                        target = learned_target
                    else:
                        # 학습 수익률이 낮으면 기술적 목표와 평균
                        target = max(tech_target, learned_target)
                else:
                    target = tech_target
                
                # 🆕 최소 기대 수익률 1.5% 보장
                target = max(target, min_target_buy)
                    
                return target
                
            elif action == SignalAction.SELL:
                # 1. 기술적 목표가
                tech_target = min(bb_lower, current_price - volatility) if bb_lower > 0 and bb_lower < current_price else current_price - volatility
                
                # 2. 학습 기반 목표가 (매도 시 수익률은 가격 하락)
                if avg_profit_pct > 0:
                    learned_target = current_price * (1 - avg_profit_pct / 100.0)
                    
                    # 🔧 개선: 볼린저밴드 제한 제거
                    if avg_profit_pct >= 1.0:
                        target = learned_target
                    else:
                        target = min(tech_target, learned_target)
                else:
                    target = tech_target
                
                # 🆕 최소 기대 수익률 1.5% 보장 (매도 시 가격 하락 방향)
                target = min(target, min_target_sell)
                    
                return max(0.0, target)
                
            return 0.0
            
        except Exception as e:
            return 0.0

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
    
    def get_combined_rl_score(self, coin: str, interval: str, candle: pd.Series, state_key: str = None) -> float:
        """🚨 코인별 점수 + DNA 기반 유사 코인 점수 + AI 모델 점수 결합"""
        try:
            strategy_key = f"{coin}_{interval}"
            
            # 🚀 실제 캔들 데이터에서 지표 추출 (verbose=False로 중복 로그 방지)
            indicators = self._calculate_smart_indicators(candle, coin, interval, verbose=False)
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
                    advanced_score = self._calculate_technical_based_score(candle, verbose=False)
            
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
                # 🌟 [Update] 과감한 글로벌 개입 (Regime Based Weighting)
                # 시장 상황(Regime)에 따라 글로벌 전략(AI/Advanced) 비중을 동적으로 조절
                
                # 1. 시장 상황 판단 (통합 방향성 기반)
                is_crisis = integrated_direction in ['strong_bearish', 'bearish']
                is_opportunity = integrated_direction in ['strong_bullish']
                
                if is_crisis:
                    # 🚨 위기 상황: 개별 전략(멘붕 가능성) 축소, 글로벌 지능(위기 관리) 대폭 확대
                    # 개별: 0.4 (Coin+DNA) / 글로벌: 0.6 (AI+Advanced)
                    combined_score = coin_score * 0.25 + dna_similar_score * 0.15 + ai_score * 0.3 + advanced_score * 0.3
                    print(f"🌪️ 위기 상황 감지: 글로벌 전략 비중 확대 (60%)")
                elif is_opportunity:
                    # 🚀 기회 상황: 개별 전략과 글로벌 지능 균형 (적극적 수익 추구)
                    # 개별: 0.5 / 글로벌: 0.5
                    combined_score = coin_score * 0.3 + dna_similar_score * 0.2 + ai_score * 0.25 + advanced_score * 0.25
                    print(f"🚀 기회 상황 감지: 적극적 수익 추구 (50:50)")
                else:
                    # ⚖️ 평시 상황: 개별 전략 우선 (기존 유지)
                    # 개별: 0.7 / 글로벌: 0.3
                    combined_score = coin_score * 0.45 + dna_similar_score * 0.25 + ai_score * 0.15 + advanced_score * 0.15
                    print(f"⚖️ 평시 상황: 개별 전략 우선 (70%)")
                    
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
    
    def get_ai_based_score(self, candle: pd.Series) -> Dict[str, float]:
        """🚀 AI 모델 예측 (현재 비활성화되어 기본 예측 사용)"""
        # 딥러닝 AI 모델이 비활성화되었으므로 바로 기본 예측 로직 사용
        return self._get_default_ai_prediction(candle)
    
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
    

