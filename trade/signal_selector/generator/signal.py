"""
signal_gen 관련 Mixin 클래스
SignalSelector의 signal_gen 기능을 담당합니다.
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
        ENABLE_CROSS_COIN_LEARNING, workspace_dir, get_coin_strategy_db_path
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
    # 🆕 추세 분석기 임포트
    from trade.core.trajectory_analyzer import get_real_trajectory_analyzer, TrendType
    # 🆕 메타 인지 감독관 임포트 (rl_pipeline 의존성 제거)
    try:
        from trade.core.data_utils import SimpleMetaSupervisor as MetaCognitiveSupervisor
    except ImportError:
        MetaCognitiveSupervisor = None
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


class SignalGeneratorMixin:
    """
    SignalGeneratorMixin - signal_gen 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

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
    
    def _get_global_fallback_signal(self, coin: str, interval: str, reason: str = "데이터 부족") -> SignalInfo:
        """⚠️ 글로벌 전략 기반 대체 시그널 생성 (Fallback) - 데이터 부족 시 안전장치"""
        try:
            # 1. 글로벌 시장 상황 확인 (캐시된 값 활용)
            market_condition = self._get_cached_market_condition(coin, interval)
            
            # 2. 기본값 설정 (보수적 접근)
            action = SignalAction.HOLD
            score = 0.0
            confidence = 0.1  # 낮은 신뢰도
            
            # 3. 시장 상황에 따른 미세 조정 (완전 맹탕보다는 나은 가이드)
            if market_condition == "bull_market":
                score = 0.1  # 약한 긍정
            elif market_condition == "bear_market":
                score = -0.1 # 약한 부정
            elif market_condition == "overbought":
                score = -0.05 # 약간의 주의
            elif market_condition == "oversold":
                score = 0.05 # 약간의 반등 기대
                
            # 4. 안전한 기본 시그널 객체 생성
            try:
                from trade.core.database import get_latest_candle_timestamp
                timestamp = get_latest_candle_timestamp()
            except:
                timestamp = int(time.time())
                
            return SignalInfo(
                coin=coin,
                interval=interval,
                action=action,
                signal_score=score,
                confidence=confidence,
                reason=f"⚠️ 글로벌 전략 대체 ({reason}) - {market_condition}",
                timestamp=timestamp,
                price=0.0,  # 가격 불명
                volume=0.0,
                rsi=50.0,
                macd=0.0,
                wave_phase='unknown',
                pattern_type='none',
                risk_level='medium',
                volatility=0.0,
                volume_ratio=1.0,
                reliability_score=0.0,
                learning_quality_score=0.0,
                global_strategy_id="global_fallback",
                coin_tuned=False,
                walk_forward_performance={},
                regime_coverage={},
                wave_progress=0.0,
                structure_score=0.5,
                pattern_confidence=0.0,
                integrated_direction='neutral',
                integrated_strength=0.5,
                target_price=0.0,
                source_type='fallback'
            )
        except Exception as e:
            print(f"❌ Fallback 시그널 생성마저 실패: {e}")
            return None

    def generate_signal(self, coin: str, interval: str, save: bool = True) -> Optional[SignalInfo]:
        """🚀 스마트 시그널 생성 (정확도 + 속도 균형)"""
        try:
            # 🚀 [Fix] PC 시각이 아닌 DB 최신 캔들 시각을 "현재"로 정의
            try:
                from trade.core.database import get_latest_candle_timestamp
                db_now = get_latest_candle_timestamp()
            except:
                db_now = int(time.time())

            # 🚀 1. 캔들 데이터 먼저 로드 (가장 중요한 데이터)
            candle = self.get_nearest_candle(coin, interval, db_now)
            if candle is None:
                print(f"⚠️ {coin}/{interval}: 캔들 데이터 부족 -> 글로벌 전략 Fallback 시도")
                return self._get_global_fallback_signal(coin, interval, "캔들 데이터 없음")
            
            # 🆕 1-1. 가격 궤적 분석 (Pre-buy Trajectory Analysis) 통합
            # 매수 전이라도 최근 20개 캔들의 흐름을 분석하여 현재 추세와 위치 파악
            trend_info = self._analyze_price_trajectory(coin, interval)
            
            # 🚀 2. 단계별 지표 계산 (정확도와 속도 균형)
            try:
                indicators = self._calculate_smart_indicators(candle, coin, interval)
            except Exception as ind_err:
                print(f"⚠️ {coin}/{interval}: 지표 계산 실패 ({ind_err}) -> 글로벌 전략 Fallback 시도")
                return self._get_global_fallback_signal(coin, interval, "지표 계산 실패")
            
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

            # 🆕 궤적 분석 결과 반영 (매수 전 필터링 및 가중치 부여)
            if trend_info:
                # 1. 하락 추세 위험 관리 (완화: 급락 중에도 지지 확인 시 기회 포착 가능)
                if trend_info['trend_type'] in ['strong_down', 'peak_reversal']:
                    # 박스 하단 지지 중이라면 삭감 완화
                    if trend_info.get('is_low_support'):
                        base_score *= 0.8  # 0.5 -> 0.8 (덜 깎음)
                    else:
                        base_score *= 0.6  # 0.5 -> 0.6
                
                # 2. 횡보장 고점/저점 전략 (박스권 매매 활성화)
                elif trend_info['trend_type'] == 'sideways':
                    if trend_info.get('is_high_resistance'):
                        base_score *= 0.8  # 0.7 -> 0.8
                    elif trend_info.get('is_low_support'):
                        base_score *= 1.3  # 1.2 -> 1.3 (가산점 강화)
                
                # 3. 상승 추세 강화
                elif trend_info['trend_type'] in ['strong_up', 'up']:
                    base_score *= 1.25  # 1.15 -> 1.25 (추세 추종 강화)

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

            # 🆕 AI 모델 점수 계산 (🔧 ai_model_loaded 조건 제거 - 기본 예측 항상 사용)
            ai_score = 0.0
            try:
                # 🔧 calculator.py와 동일하게 조건 없이 호출 (기본 예측 로직 사용)
                ai_predictions = self.get_ai_based_score(candle)
                if ai_predictions is not None and 'strategy_score' in ai_predictions:
                    ai_score = ai_predictions['strategy_score']
                    if ai_score is None:
                        ai_score = 0.0
            except Exception as ai_err:
                    ai_score = 0.0
            
            # 🆕 5. 변동성 기반 동적 가중치 조정 (AI 모델 + RL Pipeline 통합 분석 포함)
            weights = self.get_volatility_based_weights(coin, market_condition, self.ai_model_loaded)
            
            # 🆕 시그널 출처 결정 (가중치 기반)
            current_source_type = 'quant'
            if self.ai_model_loaded and weights.get('ai', 0) >= 0.2:
                current_source_type = 'hybrid'
            
            vol_group = self.get_coin_volatility_group(coin)

            # 🆕 점수 정규화 (-1.0 ~ +1.0 범위 통일)
            def _unit_to_symmetric(score: Optional[float]) -> float:
                if score is None:
                    return 0.0
                # 이미 -1.0 ~ 1.0 범위로 추정되는 값은 그대로 사용
                if -1.0 <= score <= 1.0 and score < 0:
                    return score
                # 0.0 ~ 1.0 범위 값은 -1.0 ~ 1.0로 변환
                if 0.0 <= score <= 1.0:
                    return (score - 0.5) * 2.0
                # 그 외 값은 안전하게 클리핑
                return max(-1.0, min(1.0, score))
            
            norm_base = max(-1.0, min(1.0, base_score if base_score is not None else 0.0))
            norm_dna = _unit_to_symmetric(dna_score)
            norm_rl = _unit_to_symmetric(rl_score)
            norm_integrated = _unit_to_symmetric(integrated_analysis_score)
            norm_ai = _unit_to_symmetric(ai_score) if ai_score > 0 else 0.0
            
            # 🚀 [Aggressive Component Boosting] 
            # 개별 지표(DNA, RL 등) 중 하나라도 매우 강력하면(절대값 0.6 이상), 
            # 평균에 의해 희석되지 않도록 해당 지표의 정규화 점수를 1.5배 증폭
            # 💡 [Alpha Guardian] norm_ai는 결정에서 제외하기 위해 루프에서 제거
            for score_val in [norm_dna, norm_rl]:
                if abs(score_val) >= 0.6:
                    if score_val > 0: norm_base = (norm_base + score_val) / 1.5 # 베이스 점수를 강한 지표 방향으로 끌어올림
                    if self.debug_mode:
                        print(f"  🔥 강력한 개별 지표 감지(점수 {score_val:.2f}) -> 점수 희석 방지 보정 적용")

            # 💡 [Alpha Guardian] AI 점수는 가중치를 0으로 설정하여 최종 결정에서 완전히 배제
            # 텍스트 기록을 위해 norm_ai 계산은 유지하되, 합산에는 반영하지 않음
            final_score = (
                norm_base * weights['base'] +
                norm_dna * weights['dna'] +
                norm_rl * weights['rl'] +
                norm_integrated * weights['integrated']
            )
            
            # 🆕 메타 인지 감독관(Meta-Cognitive Supervisor) 개입
            # 시장 상황(Regime)과 실제 성과(Performance) 간의 괴리를 감지하여 점수 보정
            if MetaCognitiveSupervisor:
                try:
                    db_path = get_coin_strategy_db_path(coin)
                    
                    # 캐시된 supervisor 사용 (없으면 생성) - Thread-safe access
                    with self._cache_lock:
                        if db_path not in self._supervisor_cache:
                            self._supervisor_cache[db_path] = MetaCognitiveSupervisor(db_path)
                        supervisor = self._supervisor_cache[db_path]
                    
                    # 현재 레짐에 따른 보정 계수 산출
                    meta_corrections = supervisor.analyze_performance_discrepancy(coin, interval, market_condition)
                    
                    # 현재 선택된 전략의 타입 파악
                    current_strategy_type = 'trend' # 기본값
                    if adaptive_strategy and 'strategy_type' in adaptive_strategy:
                        st_type = adaptive_strategy['strategy_type'].lower()
                        if 'trend' in st_type: current_strategy_type = 'trend'
                        elif 'rever' in st_type: current_strategy_type = 'reversion'
                        elif 'vol' in st_type: current_strategy_type = 'volatility'
                    
                    # 보정 계수 적용
                    correction_factor = meta_corrections.get(current_strategy_type, 1.0)
                    
                    if correction_factor != 1.0:
                        final_score *= correction_factor
                        if self.debug_mode:
                            print(f"  🧠 메타 인지 보정: {current_strategy_type} 전략 성과 괴리 감지 -> {correction_factor:.2f}배 적용")
                            
                except Exception as meta_e:
                    if self.debug_mode:
                        # 🆕 '실패' 대신 '준비 중'으로 순화하고, 구체적인 원인은 디버그 모드에서만 출력
                        if "no such table" in str(meta_e):
                            print(f"  ℹ️ 메타 인지 분석 대기 중 ({coin}): 학습 요약 데이터 생성 전입니다.")
                        else:
                            print(f"  ⚠️ 메타 인지 보정 건너뜀: {meta_e}")

            final_score = max(-1.0, min(1.0, final_score))
            
            # 🆕 [Reality Check] 하락 모멘텀 감지 및 점수 보정 (공격적 완화: 눌림목 포착 강화)
            price_momentum = indicators.get('price_momentum', 0.0)
            
            # 1. 강한 하락 모멘텀 발생 시 (-3% 이상 급락으로 기준 완화)
            if price_momentum < -0.03:
                if final_score > 0: # 매수 관점이었다면
                    if self.debug_mode:
                        print(f"  📉 {coin}: 강한 급락 감지({price_momentum:.3f}) -> 점수 보정 (눌림목 검토)")
                    
                    # 지지선 근처거나 RSI가 낮으면 삭감하지 않음 (눌림목 매수 기회)
                    if indicators.get('rsi_ema', 50) < 35 or indicators.get('bb_position') == 'lower':
                        final_score *= 0.9 # 거의 안 깎음
                    else:
                        final_score *= 0.7 # 일반적인 경우 소폭 삭감
                elif final_score > -0.5: # 매도 관점이었다면
                    final_score -= 0.2 # 매도 강도 강화
            
            # 2. 하락 다이버전스 발생 시 (공격적 완화)
            if indicators.get('rsi_divergence') == 'bearish' or indicators.get('macd_divergence') == 'bearish':
                # 시그널이 아주 강하면(0.4 이상) 다이버전스를 무시하고 공격적으로 진입
                if final_score > 0.4:
                    final_score *= 0.9 # 10%만 보정
                elif final_score > 0.2:
                    final_score *= 0.7 # 30% 보정 (0.15 고정 대신 가중치 유지)
            
            # 보정된 점수 재조정
            final_score = max(-1.0, min(1.0, final_score))

            # 🆕 변동성 기반 가중치 로깅
            if self.debug_mode:
                print(f"  🎯 변동성 그룹: {vol_group}")
                print(f"  ⚖️ 동적 가중치: base={weights['base']:.3f}, dna={weights['dna']:.3f}, rl={weights['rl']:.3f}, integrated={weights['integrated']:.3f}")
                if self.ai_model_loaded:
                    print(f"  🧠 AI 가중치: {weights.get('ai', 0.0):.3f}")
                print(f"  📊 구성 점수(정규화): base={norm_base:.3f}, dna={norm_dna:.3f}, rl={norm_rl:.3f}, integrated={norm_integrated:.3f}")
                if self.ai_model_loaded:
                    print(f"  🧠 AI 점수(정규화): {norm_ai:.3f}")
            
            # 신뢰도 계산 (이미 -1.0 ~ +1.0 범위 점수 사용)
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
                base_for_synergy = (final_score + 1.0) / 2.0  # 0.0 ~ 1.0 범위로 변환
                enhanced_synergy_score = self.get_synergy_enhanced_signal_score(
                    coin, interval, base_for_synergy, market_condition
                )
                final_score = max(-1.0, min(1.0, (enhanced_synergy_score * 2.0) - 1.0))
                
                if self.debug_mode:
                    print(f"  🔄 시너지 향상 점수(대칭): {final_score:.4f}")
                    
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
            
            # 🆕 final_score는 이미 -1.0 ~ +1.0 범위
            signal_score = final_score
            
            # 🆕 [통합] 학습 데이터(Thompson 점수) 반영
            thompson_bonus = 0.0
            try:
                from trade.core.thompson import get_thompson_calculator
                calc = get_thompson_calculator()
                if calc:
                    # 시그널 패턴 추출
                    signal_pattern = self._extract_signal_pattern_from_candle(candle, coin, interval)
                    # Thompson 점수 조회
                    result = calc.sample_success_rate(signal_pattern)
                    thompson_rate = result[0] if isinstance(result, tuple) else float(result)
                    
                    # Thompson 점수를 시그널 점수에 반영 (30% 가중치)
                    # Thompson 점수가 높으면(0.6 이상) 시그널 점수에 보너스, 낮으면(0.4 이하) 페널티
                    if thompson_rate >= 0.6:
                        thompson_bonus = (thompson_rate - 0.5) * 0.3  # 최대 +0.03 보너스
                    elif thompson_rate <= 0.4:
                        thompson_bonus = (thompson_rate - 0.5) * 0.3  # 최대 -0.03 페널티
                    
                    signal_score = max(-1.0, min(1.0, signal_score + thompson_bonus))
                    
                    if self.debug_mode and abs(thompson_bonus) > 0.01:
                        print(f"  🎰 Thompson 반영: {thompson_rate:.3f} -> {thompson_bonus:+.3f} 보정 (최종: {signal_score:.3f})")
            except Exception as e:
                if self.debug_mode:
                    print(f"  ⚠️ Thompson 점수 조회 실패: {e}")

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
                timestamp=db_now, # 🚀 [Fix] DB 최신 캔들 시각 부여
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
                regime_coverage=regime_coverage,
                target_price=0.0,  # 임시 값 (아래에서 계산)
                source_type=current_source_type  # 🆕 설정
            )
            
            # 🆕 궤적 정보를 시그널 객체에 추가 저장 (멀티인터벌 통합 시 참조)
            if trend_info:
                signal.trend_type = trend_info['trend_type']
                signal.position_in_range = trend_info['position_in_range']
                signal.trend_velocity = trend_info['velocity']
            
            # 🆕 target_price 계산 (전략 기반 우선 + 지표 기반 폴백)
            calculated_target = 0.0
            try:
                # 🆕 학습된 평균 수익률 조회 (Thompson Sampling)
                avg_profit_pct = 0.0
                if hasattr(self, 'thompson_sampler') and self.thompson_sampler:
                    try:
                        # 임시 시그널 객체로 패턴 추출 (signal은 위에서 생성됨)
                        current_pattern = self._extract_signal_pattern(signal)
                        stats = self.thompson_sampler.get_pattern_stats(current_pattern)
                        if stats:
                            avg_profit_pct = stats.get('avg_profit', 0.0)
                            if self.debug_mode and avg_profit_pct != 0:
                                print(f"  🧠 학습된 수익률 반영: {current_pattern[:20]}... -> {avg_profit_pct:.2f}%")
                    except Exception:
                        pass

                # 1. 기본 지표 기반 목표가 계산 (학습된 수익률 반영)
                calculated_target = self._calculate_target_price(candle, action, indicators, avg_profit_pct)
                
                # 2. 전략 기반 목표가로 보정 (Top 전략의 평균 수익률 활용)
                if adaptive_strategy and adaptive_strategy.get('expected_profit_pct', 0) > 0:
                    expected_pct = adaptive_strategy['expected_profit_pct']
                    current_price = candle.get('close', 0.0)
                    
                    if action == SignalAction.BUY:
                        strategy_target = current_price * (1 + expected_pct)
                        # 전략 목표가가 있으면 우선 사용 (데이터 기반 예측)
                        calculated_target = strategy_target
                        if self.debug_mode:
                            print(f"  🎯 전략 기반 목표가: {strategy_target:.2f} (기대수익: {expected_pct*100:.2f}%)")
                            
                    elif action == SignalAction.SELL:
                        # 매도(공매도/청산) 시 하락 목표가
                        strategy_target = current_price * (1 - expected_pct)
                        calculated_target = strategy_target
                        if self.debug_mode:
                            print(f"  🎯 전략 기반 하락목표: {strategy_target:.2f} (기대하락: {expected_pct*100:.2f}%)")
            except Exception as e:
                print(f"⚠️ 목표가 계산 오류: {e}")
            
            # 🆕 목표가 유효성 검증 (현재가의 ±50% 범위 내에만 유효)
            current_price = candle.get('close', 0.0)
            if current_price > 0 and calculated_target > 0:
                ratio = calculated_target / current_price
                if ratio < 0.5 or ratio > 2.0:
                    # 비정상 목표가 (현재가의 50%~200% 범위 밖) → 기본 계산으로 대체
                    if action == SignalAction.BUY:
                        calculated_target = current_price * 1.03  # +3% 기본 목표
                    elif action == SignalAction.SELL:
                        calculated_target = current_price * 0.97  # -3% 기본 목표
                    else:
                        calculated_target = 0.0  # HOLD는 목표가 없음
            
            # signal 객체에 target_price 설정
            signal.target_price = calculated_target
            
            # 🆕 통계 카운터 업데이트
            self._signal_stats['total_signals_generated'] += 1
            self._signal_stats['successful_signals'] += 1
            
            # 시그널 저장 (조건부)
            if save:
                self.save_signal(signal)
            
            # 🚀 시그널 생성 성공 로그 (실제 캔들 DB 데이터 기반)
            display_score = (final_score + 1.0) / 2.0  # -1~+1 → 0~1 변환
            
            # 🆕 궤적 정보 포함된 단일 라인 요약 로그
            traj_summary = ""
            if trend_info:
                traj_summary = f" | 🌊 {trend_info['trend_type']} ({trend_info['position_in_range']:.1%})"
            
            print(f"✅ {coin}/{interval}: 점수 {display_score:.3f} | 신뢰 {confidence:.2f}{traj_summary}")
            
            if self.debug_mode:
                print(f"  - 시장 상황: {market_condition}")
                print(f"  - 통합 방향: {candle.get('integrated_direction', 'neutral')}, 파동 단계: {candle.get('wave_phase', 'unknown')}")
                
                # 🚨 NoneType 안전 처리
                pattern_conf = candle.get('pattern_confidence')
                if pattern_conf is None: pattern_conf = 0.0
                
                print(f"  - 패턴 타입: {candle.get('pattern_type', 'none')}, 신뢰도: {pattern_conf:.3f}")
                print(f"  - 기본 점수: {base_score:.3f}, DNA 점수: {dna_score:.3f}")
            print(f"  - RL 점수: {rl_score:.3f}, AI 점수: {ai_score:.3f}")
            # ✅ 인터벌별 통합 분석 점수 로드 및 사용 점수/DB 점수 병기
            try:
                from trade.core.database import get_learning_data
                itg_data = get_learning_data(coin, interval, 'integrated_analysis_results')
                raw_itg_score = itg_data.get('ensemble_score', integrated_analysis_score) if itg_data else integrated_analysis_score
            except Exception:
                raw_itg_score = integrated_analysis_score

            # integrated_analysis_score는 품질 필터 적용 후 실제 사용값
            if abs(raw_itg_score - integrated_analysis_score) >= 1e-3:
                print(f"  - 통합 분석 점수({interval}): {integrated_analysis_score:.3f} (사용) / {raw_itg_score:.3f} (DB)")
            else:
                print(f"  - 통합 분석 점수({interval}): {integrated_analysis_score:.3f}")
            
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

            # 🆕 멀티 타임프레임 추세(궤적) 통합 분석 및 점수 보정
            # 장기(1d, 240m)와 단기(15m, 30m)의 추세 조화를 확인합니다.
            long_trend = interval_signals.get('1d', interval_signals.get('240m'))
            short_trend = interval_signals.get('15m', interval_signals.get('30m'))
            
            if long_trend and hasattr(long_trend, 'trend_type') and short_trend and hasattr(short_trend, 'trend_type'):
                lt = long_trend.trend_type
                st = short_trend.trend_type
                pos = getattr(short_trend, 'position_in_range', 0.5)
                
                # 1. 최고의 조합: 장기 상승 + 단기 저점 눌림목
                if lt in ['strong_up', 'up'] and (st == 'sideways' and pos < 0.3):
                    final_score *= 1.25
                    final_reason += " | 🎯 장기 상승 중 단기 저점(눌림목) 감지"
                
                # 2. 강력 추세: 장/단기 모두 상승 (추세 추종)
                elif lt in ['strong_up', 'up'] and st in ['strong_up', 'up']:
                    final_score *= 1.15
                    final_confidence *= 1.1
                    final_reason += " | 🚀 장/단기 상승 추세 정렬"
                
                # 3. 위험 조합: 장기 하락 + 단기 반등 (데드캣 바운스 경계)
                elif lt in ['strong_down', 'down'] and st in ['strong_up', 'up']:
                    final_score *= 0.6
                    final_reason += " | ⚠️ 장기 하락 중 일시적 반등 주의"
                
                # 4. 박스권 돌파 전조: 장기 상승 + 단기 고점 유지 (에너지 응축)
                elif lt in ['strong_up', 'up'] and (st == 'sideways' and pos > 0.8):
                    final_score *= 1.1
                    final_reason += " | ⚡ 장기 추세 기반 박스권 상단 돌파 전조"

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
            
            # 🆕 멀티인터벌 상태 추적 (상세 정보 포함)
            multi_interval_state = self.get_multi_interval_state_key(coin, base_signal.timestamp)
            
            # 상세 상태 출력 (판단 근거 명확화)
            print(f"\n🎯 [{coin}] 최종 종합 시그널 (멀티인터벌 통합)")
            print("============================================================")
            print(f"  📊 종합 점수: {final_score:.4f} | 신뢰도: {final_confidence:.2%}")
            
            # 🆕 추세/궤적 요약 정보 추가
            if hasattr(combined_signal, 'trend_type'):
                trend_emoji = {
                    'strong_up': '🚀', 'up': '📈', 'sideways': '↔️', 
                    'down': '📉', 'strong_down': '📉🔥', 'peak_reversal': '⚠️'
                }.get(combined_signal.trend_type, '⚪')
                
                pos_desc = "하단(지지)" if combined_signal.position_in_range < 0.3 else \
                           "상단(저항)" if combined_signal.position_in_range > 0.7 else "중간"
                
                print(f"  🌊 통합 추세: {trend_emoji} {combined_signal.trend_type.upper()} ({pos_desc})")
                print(f"  📉 궤적 위치: {combined_signal.position_in_range:.1%} (박스권 내 위치)")
            
            print(f"  💰 현재가: {current_price:,.2f}원")
            if hasattr(combined_signal, 'target_price') and combined_signal.target_price > 0:
                expected_ret = ((combined_signal.target_price - current_price) / current_price) * 100
                print(f"  🎯 예상 목표: {combined_signal.target_price:,.2f}원 ({expected_ret:+.2f}%)")
            
            print(f"  📈 분석 근거: {final_reason}")
            print("============================================================")
            
            # 🆕 target_price 계산 (가장 강한 시그널의 target_price 사용 또는 평균)
            target_price = 0.0
            current_price = base_signal.price if base_signal.price else 0.0
            if interval_signals and current_price > 0:
                # 각 인터벌 시그널의 target_price 중 유효한 값들의 평균 계산
                # 🔧 유효성 검증 추가: 현재가의 50%~200% 범위 내만 유효
                valid_target_prices = []
                for sig in interval_signals.values():
                    if sig and hasattr(sig, 'target_price') and sig.target_price > 0:
                        ratio = sig.target_price / current_price if current_price > 0 else 0
                        if 0.5 <= ratio <= 2.0:  # 합리적 범위만 포함
                            valid_target_prices.append(sig.target_price)
                
                if valid_target_prices:
                    target_price = sum(valid_target_prices) / len(valid_target_prices)
                elif hasattr(base_signal, 'target_price') and base_signal.target_price > 0:
                    ratio = base_signal.target_price / current_price if current_price > 0 else 0
                    if 0.5 <= ratio <= 2.0:
                        target_price = base_signal.target_price
            
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
                integrated_strength=float(base_signal.integrated_strength) if base_signal.integrated_strength is not None and not pd.isna(base_signal.integrated_strength) else 0.5,
                target_price=target_price  # 🆕 예상 목표가 추가
            )
            
        except Exception as e:
            print(f"⚠️ 시그널 통합 오류 ({coin}): {e}")
            return None
    
    def _analyze_price_trajectory(self, coin: str, interval: str, lookback: int = 20) -> Optional[Dict]:
        """🆕 매수 전 종목에 대한 가격 궤적(Trajectory) 분석"""
        try:
            # 1. 최근 캔들 데이터 조회
            conn = self.db_pool.get_connection()
            try:
                query = """
                    SELECT close, volume FROM candles 
                    WHERE symbol = ? AND interval = ?
                    ORDER BY timestamp DESC LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(coin, interval, lookback))
            finally:
                self.db_pool.return_connection(conn)

            if df.empty or len(df) < 10:
                return None
            
            prices = df['close'].tolist()
            current_p = prices[0]  # 최신 가격
            
            # 2. 궤적 분석 (TrajectoryAnalyzer 로직 활용)
            max_p = max(prices)
            min_p = min(prices)
            range_width_pct = ((max_p - min_p) / min_p) * 100
            
            # 박스권 내 위치 (0.0: 저점, 1.0: 고점)
            position_in_range = (current_p - min_p) / (max_p - min_p) if max_p > min_p else 0.5
            
            # 최근 기울기(Velocity) 계산 (선형 회귀)
            x = np.arange(len(prices))
            y = np.array(prices[::-1]) # 과거 -> 현재 순으로 정렬
            y_norm = y / y[0] # 정규화
            slope = np.polyfit(x, y_norm, 1)[0]
            velocity = slope * 100 # 샘플당 변화율(%)
            
            # 3. 추세 유형 결정
            if velocity > 0.4: trend_type = 'strong_up'
            elif velocity > 0.15: trend_type = 'up'
            elif velocity < -0.4: trend_type = 'strong_down'
            elif velocity < -0.15: trend_type = 'down'
            else: trend_type = 'sideways'
            
            return {
                'trend_type': trend_type,
                'velocity': velocity,
                'position_in_range': position_in_range,
                'range_width_pct': range_width_pct,
                'is_low_support': position_in_range < 0.2 and range_width_pct > 1.5,
                'is_high_resistance': position_in_range > 0.8 and range_width_pct > 1.5
            }
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ {coin}/{interval} 궤적 분석 실패: {e}")
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
                    SELECT symbol as coin, COUNT(*) as data_count
                    FROM candles 
                    WHERE interval IN ({placeholders})
                    GROUP BY symbol
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
                # print("✅ signal_feedback_scores 테이블 생성 완료")  # 로그 간소화
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
                    try:
                        conn.execute(f"ALTER TABLE signal_feedback_scores ADD COLUMN {column_def}")
                    except:
                        pass  # 컬럼 추가 실패는 무시
                
        except Exception as e:
            # 🆕 DB 접근 오류는 조용히 처리 (로그 스팸 방지)
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                pass  # 스키마 마이그레이션 오류는 조용히 무시

    def get_signal_feedback_data(self, signal_pattern: str) -> Optional[Dict]:
        """시그널 패턴에 대한 피드백 데이터 조회"""
        try:
            # 🆕 DB 파일 존재 여부 먼저 확인
            if not TRADING_SYSTEM_DB_PATH or not os.path.exists(TRADING_SYSTEM_DB_PATH):
                return None  # DB 파일 없으면 조용히 None 반환
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_feedback_scores'")
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    return None
                
                # 🆕 통일된 스키마에 맞게 쿼리 수정
                feedback_df = pd.read_sql("""
                    SELECT 
                        COALESCE(buy_score, 0.0) as buy_score,
                        COALESCE(sell_score, 0.0) as sell_score,
                        COALESCE(hold_score, 0.0) as hold_score,
                        success_rate,
                        avg_profit,
                        COALESCE(trade_count, total_trades, 0) as trade_count,
                        confidence,
                        score
                    FROM signal_feedback_scores 
                    WHERE signal_pattern = ?
                    ORDER BY updated_at DESC, last_updated DESC
                    LIMIT 1
                """, conn, params=(signal_pattern,))
                
                if not feedback_df.empty:
                    return feedback_df.iloc[0].to_dict()
                return None
                
        except Exception as e:
            # 🆕 DB 접근 오류는 조용히 처리 (로그 스팸 방지)
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 시그널 피드백 데이터 조회 오류: {e}")
            return None
    
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

    def _get_default_ai_prediction(self, candle: pd.Series, verbose: bool = False) -> Dict[str, float]:
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
            
            # 🚀 MACD 기반 상승확률 (더 정교한 계산 및 안전장치 추가)
            if macd > 0.01:  # 강한 상승 신호
                p_up = 0.7 + min(macd * 10, 0.25)  # 계수 조정
            elif macd > 0:  # 약한 상승 신호
                p_up = 0.55 + min(macd * 15, 0.15)  # 계수 조정
            elif macd > -0.01:  # 약한 하락 신호
                p_up = 0.45 + max(macd * 15, -0.15)  # 계수 조정 (음수 계산 안전화)
            else:  # 강한 하락 신호
                p_up = 0.3 + max(macd * 5, -0.25)  # 계수 조정

            # 안전장치: 확률은 0~1 사이
            p_up = max(0.05, min(0.95, p_up))
            
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
            
            if verbose:
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

    def generate_multi_timeframe_signal(self, coin: str, intervals: List[str] = ['15m', '30m', '240m', '1d'], save: bool = True) -> Optional[SignalInfo]:
        """🚀 멀티 타임프레임 시그널 통합 생성 (여러 인터벌의 정보를 종합하여 최적 시그널 생성)"""
        try:
            print(f"🔄 {coin} 멀티 타임프레임 시그널 생성 시작")
            
            # 각 인터벌별 시그널 생성
            interval_signals = {}
            for interval in intervals:
                try:
                    # 🆕 내부 호출 시 save=False로 중복 저장 방지
                    signal = self.generate_single_interval_signal(coin, interval, save=False)
                    if signal:
                        interval_signals[interval] = signal
                        # 🆕 개별 인터벌 시그널 저장 여부 결정 (가독성을 위해 silent=True)
                        if save and hasattr(self, 'save_signal'):
                            try:
                                self.save_signal(signal, silent=True)
                            except Exception: pass
                                
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
            combined_signal = self.combine_multi_timeframe_signals(coin, interval_signals, save=save)
            
            if combined_signal:
                # 🆕 combined 시그널 저장 여부 결정
                if save and hasattr(self, 'save_signal'):
                    try:
                        self.save_signal(combined_signal)
                    except Exception: pass

                print(f"✅ {coin} 멀티 타임프레임 시그널 통합 완료: {combined_signal.action.value} (점수: {combined_signal.signal_score:.3f})")
                return combined_signal
            else:
                print(f"⚠️ {coin}: 멀티 타임프레임 시그널 통합 실패")
                return None
                
        except Exception as e:
            self._handle_error(e, f"멀티 타임프레임 시그널 생성 - {coin}")
            return None
    
    def generate_single_interval_signal(self, coin: str, interval: str, save: bool = True) -> Optional[SignalInfo]:
        """단일 인터벌 시그널 생성 (기존 generate_signal 함수 활용)"""
        try:
            # 기존 generate_signal 함수 호출
            return self.generate_signal(coin, interval, save=save)
        except Exception as e:
            print(f"⚠️ {coin} {interval} 단일 인터벌 시그널 생성 실패: {e}")
            return None
    
    def _get_previous_signals(self, coin: str, intervals: List[str], lookback_count: int = 3) -> Dict[str, List[Dict]]:
        """이전 시그널 히스토리 조회 (연속성 분석용)
        
        Args:
            coin: 코인 심볼
            intervals: 조회할 인터벌 목록
            lookback_count: 조회할 이전 시그널 개수
            
        Returns:
            {interval: [{timestamp, signal_score, action}, ...]} 형태의 딕셔너리
        """
        try:
            previous_signals = {}
            
            with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
                for interval in intervals:
                    query = """
                        SELECT timestamp, signal_score, action
                        FROM signals
                        WHERE coin = ? AND interval = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    cursor = conn.execute(query, (coin, interval, lookback_count))
                    rows = cursor.fetchall()
                    
                    if rows:
                        previous_signals[interval] = [
                            {'timestamp': r[0], 'signal_score': r[1], 'action': r[2]}
                            for r in rows
                        ]
                    else:
                        previous_signals[interval] = []
                        
            return previous_signals
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ 이전 시그널 조회 실패: {e}")
            return {iv: [] for iv in intervals}
    
    def _calculate_signal_continuity(self, current_signal: SignalInfo, previous_signals: List[Dict]) -> float:
        """시그널 연속성 점수 계산
        
        이전 시그널들과의 방향성 일치도를 계산하여 연속성 점수 반환
        - 일관된 방향: 높은 점수 (신뢰도 상승)
        - 급격한 반전: 낮은 점수 (신중해야 함)
        
        Returns:
            0.0 ~ 1.0 사이의 연속성 점수
        """
        if not previous_signals:
            return 0.5  # 이전 데이터 없으면 중립
        
        current_score = current_signal.signal_score
        current_direction = 1 if current_score > 0.05 else (-1 if current_score < -0.05 else 0)
        
        continuity_scores = []
        
        for i, prev in enumerate(previous_signals):
            prev_score = prev.get('signal_score', 0.0)
            prev_direction = 1 if prev_score > 0.05 else (-1 if prev_score < -0.05 else 0)
            
            # 방향성 일치도 (같으면 1, 반대면 0, 중립 포함시 0.5)
            if current_direction == 0 or prev_direction == 0:
                direction_match = 0.5
            elif current_direction == prev_direction:
                direction_match = 1.0
            else:
                direction_match = 0.0
            
            # 점수 변화량 (급격한 변화일수록 페널티)
            score_change = abs(current_score - prev_score)
            stability_score = max(0.0, 1.0 - score_change * 2.0)  # 0.5 이상 변화시 0점
            
            # 최근일수록 가중치 높음 (가장 최근 1.0, 그 다음 0.7, 0.5...)
            recency_weight = 1.0 / (i + 1)
            
            # 종합 연속성 점수 (방향 60% + 안정성 40%)
            combined = (direction_match * 0.6 + stability_score * 0.4) * recency_weight
            continuity_scores.append(combined)
        
        # 가중 평균
        total_weight = sum(1.0 / (i + 1) for i in range(len(previous_signals)))
        return sum(continuity_scores) / total_weight if total_weight > 0 else 0.5
    
    def combine_multi_timeframe_signals(self, coin: str, interval_signals: Dict[str, SignalInfo], save: bool = True) -> Optional[SignalInfo]:
        """여러 인터벌의 시그널을 통합하여 최적 시그널 생성 (레짐 종합 + 분석 비율 활용)"""
        try:
            if not interval_signals:
                return None
            
            # 🔥 이전 시그널 히스토리 조회 (연속성 분석용)
            intervals_list = list(interval_signals.keys())
            previous_signals = self._get_previous_signals(coin, intervals_list, lookback_count=3)

            # 🔥 DB에서 코인별 전체 분석 비율 로드 (Absolute Zero 분석 결과)
            analysis_ratios = self._load_coin_analysis_ratios(coin)
            
            # 인터벌 가중치
            interval_weights = analysis_ratios.get('interval_weights', {})
            
            # 🆕 분석 모듈별 가중치 (프렉탈, 멀티타임프레임, 교차지표)
            fractal_ratios = analysis_ratios.get('fractal_ratios', {})
            multi_timeframe_ratios = analysis_ratios.get('multi_timeframe_ratios', {})
            indicator_cross_ratios = analysis_ratios.get('indicator_cross_ratios', {})
            optimal_modules = analysis_ratios.get('optimal_modules', {})
            performance_score = analysis_ratios.get('performance_score', 0.0)

            # 🎯 [동적 가중치] 상위 인터벌 방향성에 따라 하위 인터벌 가중치 조정
            # - 방향성(1d) + 스윙(240m)이 명확하면 → 타이밍(15m) 가중치 높임 (적극 매매)
            # - 방향성이 불명확하면 → 타이밍 가중치 낮춤 (보수적 매매)
            
            # 1. 상위 인터벌 방향성 확인
            direction_clarity = 0.0  # -1.0 (약세) ~ +1.0 (강세)
            direction_strength = 0.0  # 0.0 (불명확) ~ 1.0 (매우 명확)
            
            # 1d (방향성) 시그널 확인
            if '1d' in interval_signals:
                sig_1d = interval_signals['1d']
                score_1d = getattr(sig_1d, 'signal_score', 0.0)
                direction_clarity += score_1d * 0.5  # 50% 기여
                direction_strength += abs(score_1d) * 0.5
            
            # 240m (스윙) 시그널 확인
            if '240m' in interval_signals:
                sig_240m = interval_signals['240m']
                score_240m = getattr(sig_240m, 'signal_score', 0.0)
                direction_clarity += score_240m * 0.5  # 50% 기여
                direction_strength += abs(score_240m) * 0.5
            
            # 2. 방향성 명확도에 따른 동적 가중치 계산
            # direction_strength: 0.0 ~ 1.0 (상위 인터벌 신호 강도)
            # 강도가 높을수록 타이밍(15m) 가중치 증가 (더 적극적 매매)
            timing_boost = min(0.15, direction_strength * 0.3)  # 최대 15% 증가
            
            # 방향 일치 여부 확인 (1d와 240m가 같은 방향이면 추가 부스트)
            direction_aligned = False
            if '1d' in interval_signals and '240m' in interval_signals:
                score_1d = getattr(interval_signals['1d'], 'signal_score', 0.0)
                score_240m = getattr(interval_signals['240m'], 'signal_score', 0.0)
                if (score_1d > 0 and score_240m > 0) or (score_1d < 0 and score_240m < 0):
                    direction_aligned = True
                    timing_boost += 0.05  # 방향 일치 시 추가 5%
            
            # 3. 최종 동적 가중치 계산
            if not interval_weights:
                # 기본 가중치 (방향성 불명확할 때)
                base_weights = {
                    '1d': 0.30,    # 방향성 (Macro Regime)
                    '240m': 0.30,  # 스윙 (Swing)
                    '30m': 0.25,   # 모멘텀 (Momentum)
                    '15m': 0.15    # 타이밍 (Execution)
                }
                
                # 동적 조정 적용
                interval_weights = base_weights.copy()
                if timing_boost > 0:
                    # 타이밍 가중치 증가, 다른 것들은 비례 감소
                    interval_weights['15m'] = base_weights['15m'] + timing_boost
                    reduction_per_other = timing_boost / 3
                    interval_weights['1d'] = max(0.15, base_weights['1d'] - reduction_per_other)
                    interval_weights['240m'] = max(0.15, base_weights['240m'] - reduction_per_other)
                    interval_weights['30m'] = max(0.10, base_weights['30m'] - reduction_per_other)
                
                # 디버그 출력 (방향성 정보 포함)
                if self.debug_mode:
                    dir_str = "🟢 상승" if direction_clarity > 0.1 else ("🔴 하락" if direction_clarity < -0.1 else "⚪ 중립")
                    align_str = "✅ 일치" if direction_aligned else "❌ 불일치"
                    print(f"📊 {coin}: 방향 {dir_str} (강도: {direction_strength:.2f}), 1d/240m {align_str} → 타이밍 가중치: {interval_weights['15m']:.2f}")
            elif self.debug_mode:
                # 🆕 학습된 가중치 사용 중임을 표시
                print(f"✅ {coin}: 학습된 분석 비율 사용 (성능점수: {performance_score:.2f})")
            
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
            
            # 🔥 [Update] 지표 성격에 따른 인터벌 가중치 분리 (사용자 요청 반영)
            # 1. 추세/구조 판단용 (Trend/Structure) -> 장기 관점 중시 (기존 가중치 활용)
            trend_weights = regime_based_weights.copy() if regime_based_weights else interval_weights.copy()
            
            # 2. 타이밍/변동성 판단용 (Timing/Volatility) -> 단기 관점 중시
            # 단기 인터벌일수록 높은 가중치를 부여
            momentum_weights = {}
            if '15m' in interval_signals: momentum_weights['15m'] = 0.40
            if '30m' in interval_signals: momentum_weights['30m'] = 0.30
            if '240m' in interval_signals: momentum_weights['240m'] = 0.20
            if '1d' in interval_signals: momentum_weights['1d'] = 0.10
            
            # 가중치 합이 1이 되도록 정규화 (존재하는 인터벌만 고려)
            total_m_weight = sum(momentum_weights.values())
            if total_m_weight > 0:
                momentum_weights = {k: v / total_m_weight for k, v in momentum_weights.items()}
            else:
                # 단기 인터벌이 없으면 기존 가중치 그대로 사용
                momentum_weights = trend_weights.copy()

            # 🎯 통합 점수 계산
            total_score = 0.0
            total_confidence = 0.0
            total_weight = 0.0
            
            # 🔥 [NEW] 동적 영향도 기반 가중치 계산 (시그널 품질 반영)
            # 고정 가중치가 아닌, 각 인터벌의 시그널 품질에 따라 동적으로 영향도 결정
            dynamic_weights = {}
            influence_details = {}  # 디버깅용 상세 정보
            
            for interval, signal in interval_signals.items():
                # 🎯 동적 영향도 구성 요소
                # 1. 시그널 강도 (0~1): 명확한 방향성일수록 높은 영향도
                signal_strength = min(1.0, abs(signal.signal_score) * 2.0)  # 0.5 -> 1.0 매핑
                
                # 2. 신뢰도 (0~1): 시그널 자체의 신뢰도
                confidence_factor = signal.confidence if hasattr(signal, 'confidence') else 0.5
                
                # 3. 패턴 신뢰도 (0~1): 패턴 인식 정확도
                pattern_conf = getattr(signal, 'pattern_confidence', 0.0)
                pattern_factor = pattern_conf if pattern_conf > 0 else 0.5
                
                # 4. 파동 진행도 (0~1): 파동 분석의 진행 단계 (초기/중기/말기)
                wave_progress = getattr(signal, 'wave_progress', 0.5)
                # 파동 초기(0.2 이하)나 말기(0.8 이상)에서 시그널이 더 명확
                wave_clarity = 1.0 - abs(wave_progress - 0.5) * 1.5  # 중간(0.5)일 때 0.25, 끝단에서 1.0
                wave_clarity = max(0.3, min(1.0, wave_clarity))
                
                # 5. 구조 점수 (0~1): 시장 구조 분석 점수
                structure_score = getattr(signal, 'structure_score', 0.5)
                
                # 🔥 6. 시그널 연속성 점수 (0~1): 이전 시그널과의 방향성 일치도
                # 일관된 방향 = 높은 점수 (신뢰도 상승), 급격한 반전 = 낮은 점수 (신중)
                prev_sigs = previous_signals.get(interval, [])
                continuity_score = self._calculate_signal_continuity(signal, prev_sigs)
                
                # 🔥 동적 영향도 계산 (가중 합산)
                # 시그널 강도(35%) + 신뢰도(20%) + 연속성(15%) + 패턴 신뢰도(12%) + 파동 명확도(10%) + 구조 점수(8%)
                dynamic_influence = (
                    signal_strength * 0.35 +
                    confidence_factor * 0.20 +
                    continuity_score * 0.15 +  # 🆕 연속성 추가
                    pattern_factor * 0.12 +
                    wave_clarity * 0.10 +
                    structure_score * 0.08
                )
                
                # 기본 가중치와 동적 영향도 결합
                base_weight = trend_weights.get(interval, 0.2)
                # 동적 영향도가 0.5 미만이면 기본 가중치 감소, 0.5 이상이면 증가
                adjusted_weight = base_weight * (0.5 + dynamic_influence)  # 0.5x ~ 1.5x 범위
                
                dynamic_weights[interval] = adjusted_weight
                influence_details[interval] = {
                    'strength': signal_strength,
                    'confidence': confidence_factor,
                    'continuity': continuity_score,  # 🆕 연속성 추가
                    'pattern': pattern_factor,
                    'wave': wave_clarity,
                    'structure': structure_score,
                    'influence': dynamic_influence,
                    'final_weight': adjusted_weight
                }
            
            # 🎯 동적 가중치 정규화 (합이 1이 되도록)
            total_dynamic_weight = sum(dynamic_weights.values())
            if total_dynamic_weight > 0:
                dynamic_weights = {k: v / total_dynamic_weight for k, v in dynamic_weights.items()}
            
            # 디버그 모드일 때 동적 영향도 상세 출력
            if self.debug_mode:
                print(f"🔬 {coin} 동적 영향도 분석:")
                for iv, details in influence_details.items():
                    sig = interval_signals[iv]
                    print(f"   {iv}: 강도={details['strength']:.2f}, 신뢰={details['confidence']:.2f}, "
                          f"연속={details['continuity']:.2f}, 패턴={details['pattern']:.2f}, 파동={details['wave']:.2f} -> "
                          f"영향도={details['influence']:.2f}, 가중치={dynamic_weights[iv]:.2f} (점수: {sig.signal_score:.3f})")
            
            # 🆕 [보수성 완화] Max-Boosting 전략 (동적 가중치 기반)
            boosted_weights = dynamic_weights.copy()
            max_abs_score = 0.0
            best_interval = None
            
            for interval, signal in interval_signals.items():
                abs_score = abs(signal.signal_score)
                if abs_score > max_abs_score:
                    max_abs_score = abs_score
                    best_interval = interval
            
            # 🚀 [Aggressive] 의미 있는 신호가 포착되면 해당 인터벌에 파격적인 가중치 (Max Boosting)
            if max_abs_score >= 0.4 and best_interval:
                boosted_weights[best_interval] *= 2.5  # 동적 가중치에 2.5배 부스팅
                if self.debug_mode:
                    print(f"🚀 {coin}: {best_interval}에서 강력한 시그널({max_abs_score:.3f}) 감지 -> 동적 가중치 부스팅 적용")
            
            # 🎯 액션별 투표 집계
            action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            
            for interval, signal in interval_signals.items():
                # 부스팅된 가중치 적용
                weight = boosted_weights.get(interval, 0.1)
                
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
            
            # 🚀 희석(Dilution) 방지: 대부분 같은 방향이면 최고점을 더 반영
            all_scores = [sig.signal_score for sig in interval_signals.values()]
            positive_count = len([s for s in all_scores if s > 0])
            if positive_count >= len(all_scores) * 0.75:  # 75% 이상이 상방이면
                max_sig = max(all_scores)
                final_score = (final_score * 0.3) + (max_sig * 0.7)
            elif positive_count == 0 and len(all_scores) > 0:  # 모두 하방이면
                min_sig = min(all_scores)
                final_score = (final_score * 0.3) + (min_sig * 0.7)
            
            # 🆕 [추가 보정] 방향성 일치 보너스 및 희석 방지 (Aggregrate Agreement)
            buy_votes = action_votes.get('buy', 0)
            sell_votes = action_votes.get('sell', 0)
            total_valid_intervals = len(interval_signals)
            
            # 🆕 [Aggressive Integration] 
            # 1. 방향성이 일치하면 가장 높은 점수를 더 많이 반영 (희석 방지)
            if (buy_votes >= total_valid_intervals * 0.5 and final_score > 0) or \
               (sell_votes >= total_valid_intervals * 0.5 and final_score < 0):
                # 일치하는 방향의 최고 점수 비중을 더 높임 (희석 최소화)
                target_max = max_abs_score if final_score > 0 else -max_abs_score
                final_score = (final_score * 0.4) + (target_max * 0.6) # 평균보다 최고점 중시
                if self.debug_mode:
                    print(f"  🔥 방향성 합의 보정(Aggressive): 최종 점수 {final_score:.3f}")

            # 2. 압도적 일치 시 추가 보너스 (파격 상향)
            if buy_votes >= total_valid_intervals * 0.75:  # 75% 이상이 매수 의견이면
                final_score *= 1.8  # 1.3 -> 1.8 상향
                final_confidence = min(0.98, final_confidence * 1.25)
            elif sell_votes >= total_valid_intervals * 0.75:  # 75% 이상이 매도 의견이면
                final_score *= 1.8  # 1.3 -> 1.8 상향
                final_confidence = min(0.98, final_confidence * 1.25)
            
            # 🆕 [Absolute Zero 분석 비율 적용] 학습된 분석 모듈 가중치로 점수 보정
            analysis_adjustment = self._apply_analysis_ratios_adjustment(
                coin=coin,
                interval_signals=interval_signals,
                base_score=final_score,
                fractal_ratios=fractal_ratios,
                multi_timeframe_ratios=multi_timeframe_ratios,
                indicator_cross_ratios=indicator_cross_ratios,
                optimal_modules=optimal_modules,
                performance_score=performance_score
            )
            
            if analysis_adjustment != 0.0:
                final_score = final_score + analysis_adjustment
                if self.debug_mode:
                    print(f"  📊 {coin}: 분석 비율 보정 적용 ({analysis_adjustment:+.3f}) -> {final_score:.3f}")
            
            # 🎯 최종 액션 결정 (투표 기반 + 점수 기반) - 자율 임계값 연동
            final_action = self._determine_final_action(action_votes, action_scores, final_score, coin, 'combined')

            # 🆕 target_price 계산 (추세 중심 가중치 적용)
            target_price = 0.0
            current_price = self._get_latest_price(coin)
            if interval_signals and current_price > 0:
                valid_targets = []
                valid_weights = []
                for interval, sig in interval_signals.items():
                    if sig and hasattr(sig, 'target_price') and sig.target_price > 0:
                        # 🔧 유효성 검증: 현재가의 50%~200% 범위 내만 유효
                        ratio = sig.target_price / current_price
                        if 0.5 <= ratio <= 2.0:
                            valid_targets.append(sig.target_price)
                            valid_weights.append(trend_weights.get(interval, 1.0))
                
                if valid_targets:
                    # 가중 평균 목표가
                    target_price = np.average(valid_targets, weights=valid_weights)
            
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
                # 🚀 [Update] 지표별 성격에 따른 가중치 차별화 적용
                # 단기 반응성 지표 -> momentum_weights (단기 중시)
                rsi=self._calculate_weighted_average(interval_signals, 'rsi', momentum_weights),
                volatility=self._calculate_weighted_average(interval_signals, 'volatility', momentum_weights),
                volume_ratio=self._calculate_weighted_average(interval_signals, 'volume_ratio', momentum_weights),
                
                # 추세 지속성 지표 -> trend_weights (장기 중시)
                macd=self._calculate_weighted_average(interval_signals, 'macd', trend_weights),
                wave_phase=self._get_most_common_value(interval_signals, 'wave_phase'), # 범주형은 최빈값
                pattern_type=self._get_most_common_value(interval_signals, 'pattern_type'),
                risk_level=self._get_most_common_value(interval_signals, 'risk_level'),
                
                # 학습/신뢰도 지표 -> trend_weights
                reliability_score=0.0,
                learning_quality_score=0.0,
                global_strategy_id="",
                coin_tuned=False,
                walk_forward_performance=None,
                regime_coverage=None,
                
                # 구조적 지표 -> trend_weights
                wave_progress=self._calculate_weighted_average(interval_signals, 'wave_progress', trend_weights),
                structure_score=self._calculate_weighted_average(interval_signals, 'structure_score', trend_weights),
                pattern_confidence=self._calculate_weighted_average(interval_signals, 'pattern_confidence', trend_weights),
                integrated_direction=self._get_most_common_value(interval_signals, 'integrated_direction'),
                integrated_strength=self._calculate_weighted_average(interval_signals, 'integrated_strength', trend_weights),
                
                target_price=target_price,  # 🆕 가중 평균 목표가
                source_type='hybrid',  # 🆕 멀티 타임프레임 통합은 항상 하이브리드 성격
                
                # 🚀 고급 지표 통합 (누락 방지 및 정보 보존)
                # 1. 자금 흐름 및 모멘텀 (단기 중시)
                mfi=self._calculate_weighted_average(interval_signals, 'mfi', momentum_weights),
                price_momentum=self._calculate_weighted_average(interval_signals, 'price_momentum', momentum_weights),
                volume_momentum=self._calculate_weighted_average(interval_signals, 'volume_momentum', momentum_weights),
                wave_momentum=self._calculate_weighted_average(interval_signals, 'wave_momentum', momentum_weights),
                
                # 2. 추세 및 변동성 상세 (장기 중시)
                adx=self._calculate_weighted_average(interval_signals, 'adx', trend_weights),
                atr=self._calculate_weighted_average(interval_signals, 'atr', trend_weights),
                trend_strength=self._calculate_weighted_average(interval_signals, 'trend_strength', trend_weights),
                bb_width=self._calculate_weighted_average(interval_signals, 'bb_width', trend_weights),
                bb_squeeze=self._calculate_weighted_average(interval_signals, 'bb_squeeze', trend_weights),
                
                # 3. 구조적 상태 (범주형 - 최빈값)
                bb_position=self._get_most_common_value(interval_signals, 'bb_position'),
                market_structure=self._get_most_common_value(interval_signals, 'market_structure'),
                elliott_wave=self._get_most_common_value(interval_signals, 'elliott_wave'),
                harmonic_patterns=self._get_most_common_value(interval_signals, 'harmonic_patterns'),
                rsi_divergence=self._get_most_common_value(interval_signals, 'rsi_divergence'),
                macd_divergence=self._get_most_common_value(interval_signals, 'macd_divergence'),
                support_resistance=self._get_most_common_value(interval_signals, 'support_resistance'),
                market_condition=self._get_most_common_value(interval_signals, 'market_condition')
            )
            
            # 🔧 ADX 극단값 보정 (0.00 또는 100.00은 데이터 부족/오류 가능성)
            if combined_signal.adx is not None:
                if combined_signal.adx >= 99.0 or combined_signal.adx <= 0.1:
                    # 변동성 기반 추정
                    vol = combined_signal.volatility if combined_signal.volatility else 0.02
                    est_adx = 20.0 + (vol * 1000)
                    combined_signal.adx = min(80.0, max(10.0, est_adx))
            
            return combined_signal
            
        except Exception as e:
            print(f"⚠️ {coin} 멀티 타임프레임 시그널 통합 실패: {e}")
            return None
    
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
    
    def _apply_analysis_ratios_adjustment(
        self,
        coin: str,
        interval_signals: Dict[str, Any],
        base_score: float,
        fractal_ratios: Dict[str, float],
        multi_timeframe_ratios: Dict[str, float],
        indicator_cross_ratios: Dict[str, float],
        optimal_modules: Dict[str, float],
        performance_score: float
    ) -> float:
        """
        🆕 Absolute Zero 분석 비율을 활용한 점수 보정
        
        - fractal_ratios: 프렉탈 분석 비율 (추세 지속성)
        - multi_timeframe_ratios: 멀티 타임프레임 분석 비율 (방향 일치도)
        - indicator_cross_ratios: 교차 지표 분석 비율 (RSI/MACD 일치도)
        - optimal_modules: 최적 분석 모듈 가중치
        - performance_score: 전체 성능 점수
        
        Returns:
            점수 보정값 (-0.2 ~ +0.2 범위)
        """
        try:
            adjustment = 0.0
            
            # 분석 비율이 없으면 보정 없이 반환
            if not any([fractal_ratios, multi_timeframe_ratios, indicator_cross_ratios, optimal_modules]):
                return 0.0
            
            # 1. 프렉탈 분석 보정 (추세 지속성 기반)
            # 프렉탈 점수가 높으면 추세가 지속될 가능성이 높음
            if fractal_ratios:
                fractal_score = fractal_ratios.get('score', fractal_ratios.get('fractal_score', 0.5))
                # 추세 방향과 점수 방향이 일치하면 보너스
                if (base_score > 0 and fractal_score > 0.6) or (base_score < 0 and fractal_score < 0.4):
                    adjustment += 0.03 * abs(fractal_score - 0.5)
                elif (base_score > 0 and fractal_score < 0.4) or (base_score < 0 and fractal_score > 0.6):
                    adjustment -= 0.02 * abs(fractal_score - 0.5)
            
            # 2. 멀티 타임프레임 일치도 보정
            # 여러 타임프레임의 방향이 일치하면 신뢰도 증가
            if multi_timeframe_ratios:
                mtf_consistency = multi_timeframe_ratios.get('consistency', 
                                   multi_timeframe_ratios.get('direction_consistency', 0.5))
                if mtf_consistency > 0.7:
                    # 방향 일치도가 높으면 점수 방향 유지 + 약간 증폭
                    adjustment += 0.04 * (mtf_consistency - 0.5) * (1 if base_score > 0 else -1) * abs(base_score)
                elif mtf_consistency < 0.3:
                    # 방향이 불일치하면 점수 축소
                    adjustment -= 0.02 * (0.5 - mtf_consistency)
            
            # 3. 교차 지표 분석 보정 (6개 핵심 지표: rsi, macd, mfi, atr, adx, bb)
            # IntegratedAnalyzer에서 학습한 지표별 비율을 활용
            if indicator_cross_ratios:
                # 6개 핵심 지표 비율 추출 (기본값 0.5)
                rsi_weight = indicator_cross_ratios.get('rsi', 0.5)
                macd_weight = indicator_cross_ratios.get('macd', 0.5)
                mfi_weight = indicator_cross_ratios.get('mfi', 0.5)
                atr_weight = indicator_cross_ratios.get('atr', 0.5)
                adx_weight = indicator_cross_ratios.get('adx', 0.5)
                bb_weight = indicator_cross_ratios.get('bb', 0.5)
                
                # 시그널에서 지표 값 추출하여 방향 일치도 계산
                indicator_signals = []
                for interval, sig in interval_signals.items():
                    rsi = getattr(sig, 'rsi', 50) or 50
                    macd = getattr(sig, 'macd', 0) or 0
                    mfi = getattr(sig, 'mfi', 50) or 50
                    
                    # 각 지표의 매수/매도 신호 판단
                    rsi_signal = 1 if rsi < 35 else (-1 if rsi > 65 else 0)  # 매수/매도/중립
                    macd_signal = 1 if macd > 0 else (-1 if macd < 0 else 0)
                    mfi_signal = 1 if mfi < 30 else (-1 if mfi > 70 else 0)
                    
                    # 가중 합산
                    weighted_signal = (
                        rsi_signal * rsi_weight +
                        macd_signal * macd_weight +
                        mfi_signal * mfi_weight
                    )
                    indicator_signals.append(weighted_signal)
                
                if indicator_signals:
                    # 지표 신호 방향 일치도
                    avg_indicator_signal = sum(indicator_signals) / len(indicator_signals)
                    
                    # base_score와 지표 신호가 같은 방향이면 보너스
                    if (base_score > 0 and avg_indicator_signal > 0.3) or \
                       (base_score < 0 and avg_indicator_signal < -0.3):
                        # 지표 일치 보너스 (최대 0.05)
                        indicator_bonus = min(0.05, abs(avg_indicator_signal) * 0.05)
                        adjustment += indicator_bonus
                    elif (base_score > 0 and avg_indicator_signal < -0.3) or \
                         (base_score < 0 and avg_indicator_signal > 0.3):
                        # 지표 불일치 페널티 (최대 -0.03)
                        adjustment -= min(0.03, abs(avg_indicator_signal) * 0.03)
            
            # 4. 최적 모듈 가중치 적용
            if optimal_modules:
                # 가장 신뢰할 수 있는 모듈 확인
                best_module = max(optimal_modules.items(), key=lambda x: x[1], default=(None, 0))
                if best_module[0] and best_module[1] > 0.3:
                    # 최적 모듈 가중치가 높으면 약간의 신뢰도 보너스
                    adjustment += 0.02 * best_module[1]
            
            # 5. 전체 성능 점수 보정
            if performance_score > 0.6:
                # 과거 성능이 좋았으면 신뢰도 증가
                adjustment += 0.02 * (performance_score - 0.5)
            elif performance_score < 0.4:
                # 과거 성능이 좋지 않았으면 보수적으로
                adjustment -= 0.01 * (0.5 - performance_score)
            
            # 최종 보정 범위 제한 (-0.2 ~ +0.2)
            adjustment = max(-0.2, min(0.2, adjustment))
            
            return adjustment
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ {coin}: 분석 비율 보정 실패 - {e}")
            return 0.0

