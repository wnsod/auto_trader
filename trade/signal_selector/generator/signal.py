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

    def generate_signal(self, coin: str, interval: str) -> Optional[SignalInfo]:
        """🚀 스마트 시그널 생성 (정확도 + 속도 균형)"""
        try:
            # 🚀 1. 캔들 데이터 먼저 로드 (가장 중요한 데이터)
            candle = self.get_nearest_candle(coin, interval, int(time.time()))
            if candle is None:
                print(f"⚠️ {coin}/{interval}: 캔들 데이터 부족 -> 글로벌 전략 Fallback 시도")
                return self._get_global_fallback_signal(coin, interval, "캔들 데이터 없음")
            
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
            # 🔧 AI 점수가 실제로 계산되었으면 사용 (ai_model_loaded 조건 제거)
            norm_ai = _unit_to_symmetric(ai_score) if ai_score > 0 else 0.0
            
            # 🔧 AI 점수가 있으면 가중치에 포함하여 계산
            has_ai_score = ai_score > 0
            if has_ai_score:
                final_score = (
                    norm_base * weights['base'] +
                    norm_dna * weights['dna'] +
                    norm_rl * weights['rl'] +
                    norm_ai * weights.get('ai', 0.15) +
                    norm_integrated * weights['integrated']
                )
            else:
                final_score = (
                    norm_base * weights['base'] +
                    norm_dna * weights['dna'] +
                    norm_rl * weights['rl'] +
                    norm_integrated * weights['integrated']
                )
            
            # 🆕 메타 인지 감독관(Meta-Cognitive Supervisor) 개입
            # 시장 상황(Regime)과 실제 성과(Performance) 간의 괴리를 감지하여 점수 보정
            try:
                from rl_pipeline.analysis.meta_supervisor import MetaCognitiveSupervisor
                from signal_selector.config import get_coin_strategy_db_path
                
                db_path = get_coin_strategy_db_path(coin)
                supervisor = MetaCognitiveSupervisor(db_path)
                
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
                        
            except ImportError:
                pass # 모듈이 없거나 경로 문제 시 무시
            except Exception as meta_e:
                if self.debug_mode:
                    print(f"⚠️ 메타 인지 보정 실패: {meta_e}")

            final_score = max(-1.0, min(1.0, final_score))
            
            # 🆕 [Reality Check] 하락 모멘텀 감지 및 점수 보정 (낙관 편향 방지)
            price_momentum = indicators.get('price_momentum', 0.0)
            
            # 1. 강한 하락 모멘텀 발생 시 (-2% 이상 급락)
            if price_momentum < -0.02:
                if final_score > 0: # 매수 관점이었다면
                    if self.debug_mode:
                        print(f"  📉 {coin}: 하락 모멘텀 감지({price_momentum:.3f}) -> 점수 하향 조정")
                    final_score *= 0.5 # 매수 점수 반토막 (신중하게)
                elif final_score > -0.5: # 매도 관점이었다면
                    final_score -= 0.2 # 매도 강도 강화 (더 강하게 매도)

            # 2. 하락 다이버전스 발생 시
            if indicators.get('rsi_divergence') == 'bearish' or indicators.get('macd_divergence') == 'bearish':
                if final_score > 0.2:
                    final_score = 0.1 # 매수 보류 수준으로 낮춤
            
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
                regime_coverage=regime_coverage,
                target_price=0.0,  # 임시 값 (아래에서 계산)
                source_type=current_source_type  # 🆕 설정
            )
            
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
            
            # 시그널 저장
            self.save_signal(signal)
            
            # 🚀 시그널 생성 성공 로그 (실제 캔들 DB 데이터 기반)
            print(f"✅ {coin}/{interval}: 실제 캔들 DB 기반 시그널 생성 성공")
            # 🔧 점수를 0~1 범위로 변환하여 표시 (직관성 향상)
            display_score = (final_score + 1.0) / 2.0  # -1~+1 → 0~1 변환
            print(f"  - 점수: {display_score:.3f}, 신뢰도: {confidence:.3f}")
            print(f"  - 시장 상황: {market_condition}")
            print(f"  - 통합 방향: {candle.get('integrated_direction', 'neutral')}, 파동 단계: {candle.get('wave_phase', 'unknown')}")
            
            # 🚨 NoneType 안전 처리
            pattern_conf = candle.get('pattern_confidence')
            if pattern_conf is None: pattern_conf = 0.0
            
            print(f"  - 패턴 타입: {candle.get('pattern_type', 'none')}, 신뢰도: {pattern_conf:.3f}")
            print(f"  - 기본 점수: {base_score:.3f}, DNA 점수: {dna_score:.3f}")
            print(f"  - RL 점수: {rl_score:.3f}, AI 점수: {ai_score:.3f}")
            print(f"  - 통합 분석 점수: {integrated_analysis_score:.3f}")
            print(f"  - 최종 점수: {display_score:.3f}, 신뢰도: {confidence:.3f}")
            
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
                        # 🆕 개별 인터벌 시그널도 DB에 저장 (실전매매 등에서 재사용 가능하도록)
                        if hasattr(self, 'save_signal'):
                            try:
                                self.save_signal(signal)
                            except Exception as e:
                                print(f"  ⚠️ {interval} 시그널 저장 실패: {e}")
                                
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
                # 🆕 combined 시그널도 DB에 저장 (실전매매 등에서 재사용 가능하도록)
                if hasattr(self, 'save_signal'):
                    try:
                        self.save_signal(combined_signal)
                    except Exception as e:
                        print(f"  ⚠️ {coin} combined 시그널 저장 실패: {e}")

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
                # 첫 실행 시 정상 - 학습 후 자동으로 최적 가중치 적용됨
                print(f"ℹ️ {coin}: 기본 인터벌 가중치 사용 (학습 데이터 축적 후 최적화됨)")
            
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
            
            # 🎯 액션별 투표 집계
            action_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            
            for interval, signal in interval_signals.items():
                # 레짐 기반 가중치 적용 (기본적으로 추세 중심)
                weight = trend_weights.get(interval, 0.1)
                
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
    

