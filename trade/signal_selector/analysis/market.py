"""
market 관련 Mixin 클래스
SignalSelector의 market 기능을 담당합니다.
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


class MarketAnalysisMixin:
    """
    MarketAnalysisMixin - market 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

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
        """🆕 변동성 그룹 및 자율 신뢰도 기반 동적 액션 임계값 반환"""
        try:
            # 1. 자율 주행 엔진에서 베이스 문턱값 가져오기 (캔들 신뢰도 연동)
            base_threshold = 0.30
            if hasattr(self, 'get_learning_based_signal_score_threshold'):
                # 통합 시그널 판단을 위한 베이스 문턱값 조회
                base_threshold = self.get_learning_based_signal_score_threshold(coin, 'combined')

            vol_group = self.get_coin_volatility_group(coin)

            # 2. 변동성 그룹별 조정 계수 적용
            if vol_group == 'LOW':
                multiplier = 1.5  # BTC 등은 더 확실한 신호 필요
            elif vol_group == 'MEDIUM':
                multiplier = 1.0  # ETH 등은 표준
            elif vol_group == 'HIGH':
                multiplier = 0.7  # SOL 등은 더 공격적으로
            else:  # VERY_HIGH
                multiplier = 0.5  # 밈코인 등은 즉각 반응

            # 최종 임계값 산출 (최소 0.1, 최대 0.6 범위 제한)
            adj_threshold = max(0.1, min(0.6, base_threshold * multiplier))
            
            return {
                'strong_buy': adj_threshold * 2.0,
                'weak_buy': adj_threshold,
                'weak_sell': -adj_threshold,
                'strong_sell': -adj_threshold * 2.0
            }
        except Exception:
            return {
                'strong_buy': 0.5, 'weak_buy': 0.25, 'weak_sell': -0.25, 'strong_sell': -0.5
            }

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
    
    def _detect_current_market_condition(self, coin: str, interval: str) -> str:
        """현재 시장 상태 감지 (크로스 코인 학습용)"""
        try:
            # 간단한 시장 상태 감지 (실제 구현에서는 더 정교한 로직 사용)
            return 'neutral'  # 기본값
        except Exception as e:
            print(f"⚠️ 시장 상태 감지 실패: {e}")
            return 'neutral'

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
    
    def _categorize_volatility(self, volatility: float) -> str:
        """변동성 범주화 (기존 호환성 유지)"""
        return self._categorize_volatility_enhanced(volatility)
    
    def _get_market_context(self, coin: str, interval: str) -> dict:
        """🆕 시장 상황 분석"""
        try:
            # [엔진화] 하드코딩된 BTC 대신 환경변수 또는 DB의 대장 코인 시그널 사용
            leader_coin = os.getenv('MARKET_LEADER', 'BTC')
            btc_signal = self.get_cached_data(f"signal_{leader_coin}_{interval}", max_age=300)
            
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
    
    def _evaluate_absolute_zero_conditions(self, candle: pd.Series, strategy_key: str) -> float:
        """Absolute Zero System에서 학습한 전략들의 성과를 기반으로 한 적합성 평가"""
        try:
            adaptation_score = 0.0
            
            # 🎯 Absolute Zero System에서 학습한 전략들의 성과 데이터 활용
            # 1. 해당 코인/인터벌의 상위 성과 전략들 조회
            coin, interval = strategy_key.split('_', 1)
            
            try:
                from signal_selector.config import STRATEGIES_DB_PATH
                with sqlite3.connect(STRATEGIES_DB_PATH) as conn:
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
                            WHERE symbol = ? AND interval = ? 
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
    def detect_current_market_condition(self, coin: str, interval: str) -> str:
        """🆕 설계 반영: 캔들 DB에서 공인된 BTC 7단계 레짐 정보를 직접 로드 (계산 로직 통합)"""
        try:
            # 🎯 DB에서 최신 공인 레짐 로드 시도
            regime = 'neutral'
            
            try:
                with sqlite3.connect(CANDLES_DB_PATH) as conn:
                    cursor = conn.cursor()
                    # [엔진화] 하드코딩된 BTC 대신, DB에서 가장 최신 레짐 데이터가 있는 대표 코인을 찾음
                    cursor.execute("""
                        SELECT regime_label, symbol FROM candles 
                        WHERE regime_label IS NOT NULL
                        ORDER BY timestamp DESC, volume DESC LIMIT 1
                    """)
                    row = cursor.fetchone()
                    if row:
                        regime = str(row[0] or 'neutral').lower().replace(' ', '_')
            except Exception:
                # DB 조회 실패 시 analyzer 폴백
                if hasattr(self, 'market_regime_manager'):
                    info = self.market_regime_manager.analyze_market_regime()
                    regime = info.get('regime', 'neutral').lower().replace(' ', '_')
            
            return regime
                
        except Exception as e:
            # print(f"⚠️ 시장 상황 감지 오류: {e}")
            return "neutral"
    
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
    

