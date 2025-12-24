"""
레짐 기반 라우팅 시스템
새로운 파이프라인의 3단계: 레짐 기반 라우팅
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from rl_pipeline.core.regime_classifier import classify_regime_from_old

# 디버거 import
try:
    from rl_pipeline.monitoring import RoutingDebugger
    DEBUGGER_AVAILABLE = True
except ImportError:
    RoutingDebugger = None
    DEBUGGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """7단계 시장 레짐"""
    EXTREME_BEARISH = "extreme_bearish"
    BEARISH = "bearish"
    SIDEWAYS_BEARISH = "sideways_bearish"
    NEUTRAL = "neutral"
    SIDEWAYS_BULLISH = "sideways_bullish"
    BULLISH = "bullish"
    EXTREME_BULLISH = "extreme_bullish"

@dataclass
class RegimeRoutingResult:
    """레짐 라우팅 결과"""
    coin: str
    interval: str
    regime: str
    routed_strategy: Dict[str, Any]
    routing_confidence: float
    routing_score: float
    regime_performance: float
    regime_adaptation: float
    created_at: str
    predictive_accuracy: float = 0.0  # 🔥 예측 정확도 (백테스트에서 계산)
    backtest_result: Optional[Dict[str, Any]] = None  # 🔥 백테스트 결과 (예측 정확도 포함)

class RegimeRouter:
    """레짐 기반 라우터"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.regime_weights = {
            'extreme_bearish': 0.1,
            'bearish': 0.2,
            'sideways_bearish': 0.15,
            'neutral': 0.2,
            'sideways_bullish': 0.15,
            'bullish': 0.2,
            'extreme_bullish': 0.1
        }

        # 백테스트용 데이터 저장
        self.regime_timeline = None
        self.candle_data = None

        # 🔥 디버거 초기화
        self.routing_debug = None
        if DEBUGGER_AVAILABLE and session_id:
            try:
                self.routing_debug = RoutingDebugger(session_id=session_id)
                logger.info(f"✅ RoutingDebugger 초기화 완료 (session: {session_id})")
            except Exception as e:
                logger.warning(f"⚠️ RoutingDebugger 초기화 실패: {e}")

        logger.info("🚀 레짐 라우터 초기화 완료 (백테스트 기능 활성화)")
    
    def detect_current_regime(self, coin: str, interval: str, candle_data: pd.DataFrame) -> Tuple[str, float, float]:
        """현재 시장 레짐 감지 및 전환 확률 반환

        Returns:
            Tuple[str, float, float]: (regime_label, confidence, regime_transition_prob)
        """
        try:
            if candle_data.empty or len(candle_data) < 20:
                return "ranging", 0.5, 0.0  # 기본값: ranging (3단계 레짐)

            # 🆕 캔들 데이터에 'regime' 컬럼이 있으면 DB에서 로드된 레짐 사용 (3단계 매핑된 값)
            if 'regime' in candle_data.columns:
                latest_regime = candle_data['regime'].iloc[-1]
                if pd.notna(latest_regime):
                    regime_label = str(latest_regime)

                    # 신뢰도는 'regime_confidence' 컬럼에서 가져오거나 기본값 사용
                    confidence = 0.8  # 기본값
                    if 'regime_confidence' in candle_data.columns:
                        latest_confidence = candle_data['regime_confidence'].iloc[-1]
                        if pd.notna(latest_confidence):
                            confidence = float(latest_confidence)

                    # 🔥 regime_transition_prob 읽기 (캔들 데이터에서)
                    regime_transition_prob = 0.0
                    if 'regime_transition_prob' in candle_data.columns:
                        latest_transition_prob = candle_data['regime_transition_prob'].iloc[-1]
                        if pd.notna(latest_transition_prob):
                            regime_transition_prob = float(latest_transition_prob)

                    # 전환 확률이 높으면 경고
                    if regime_transition_prob > 0.15:
                        logger.warning(f"⚠️ {coin}-{interval} 높은 레짐 전환 확률 감지: {regime_transition_prob:.2%} "
                                     f"(현재 레짐: {regime_label})")

                    logger.info(f"📊 {coin}-{interval} 레짐 로드 (DB): {regime_label} (신뢰도: {confidence:.2f}, "
                               f"전환 확률: {regime_transition_prob:.2%})")

                    # 🔥 디버거 로깅
                    if self.routing_debug:
                        try:
                            self.routing_debug.log_regime_detected(
                                coin=coin,
                                interval=interval,
                                regime=regime_label,
                                confidence=confidence,
                                transition_prob=regime_transition_prob,
                                indicators={}
                            )
                        except Exception as debug_err:
                            logger.debug(f"디버거 로깅 실패 (무시): {debug_err}")

                    return regime_label, confidence, regime_transition_prob

            # 🔄 폴백: 'regime' 컬럼이 없으면 기존 방식으로 계산
            logger.debug(f"⚠️ {coin}-{interval} 'regime' 컬럼 없음 - 레짐 재계산")

            # 최근 데이터로 레짐 계산
            recent_data = candle_data.tail(20)

            # 레짐 감지 로직 (7단계)
            regime_score = self._calculate_regime_score(recent_data)
            regime_label_7stage = self._classify_regime(regime_score)
            confidence = self._calculate_regime_confidence(regime_score)

            # 🆕 7단계 → 3단계 매핑 적용
            regime_label = classify_regime_from_old(regime_label_7stage)

            # 🔥 regime_transition_prob 읽기 (캔들 데이터에서)
            regime_transition_prob = 0.0
            if 'regime_transition_prob' in candle_data.columns:
                latest_transition_prob = candle_data['regime_transition_prob'].iloc[-1]
                if pd.notna(latest_transition_prob):
                    regime_transition_prob = float(latest_transition_prob)

            # 전환 확률이 높으면 경고
            if regime_transition_prob > 0.15:
                logger.warning(f"⚠️ {coin}-{interval} 높은 레짐 전환 확률 감지: {regime_transition_prob:.2%} "
                             f"(현재 레짐: {regime_label})")

            logger.info(f"📊 {coin}-{interval} 레짐 계산: {regime_label_7stage} → {regime_label} (신뢰도: {confidence:.2f}, "
                       f"전환 확률: {regime_transition_prob:.2%})")

            # 🔥 디버거 로깅
            if self.routing_debug:
                try:
                    self.routing_debug.log_regime_detected(
                        coin=coin,
                        interval=interval,
                        regime=regime_label,
                        confidence=confidence,
                        transition_prob=regime_transition_prob,
                        indicators={}
                    )
                except Exception as debug_err:
                    logger.debug(f"디버거 로깅 실패 (무시): {debug_err}")

            return regime_label, confidence, regime_transition_prob

        except Exception as e:
            logger.error(f"❌ 레짐 감지 실패: {e}")
            return "ranging", 0.5, 0.0  # 기본값: ranging (3단계 레짐)
    
    def _calculate_regime_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """레짐 점수 계산"""
        try:
            # 가격 트렌드 분석
            price_trend = self._analyze_price_trend(data)
            
            # 변동성 분석
            volatility = self._analyze_volatility(data)
            
            # 거래량 분석
            volume_pattern = self._analyze_volume_pattern(data)
            
            # 기술적 지표 분석
            technical_signals = self._analyze_technical_signals(data)
            
            return {
                'price_trend': price_trend,
                'volatility': volatility,
                'volume_pattern': volume_pattern,
                'technical_signals': technical_signals
            }
            
        except Exception as e:
            logger.error(f"❌ 레짐 점수 계산 실패: {e}")
            return {'price_trend': 0.5, 'volatility': 0.5, 'volume_pattern': 0.5, 'technical_signals': 0.5}
    
    def _analyze_price_trend(self, data: pd.DataFrame) -> float:
        """가격 트렌드 분석"""
        try:
            if 'close' not in data.columns:
                return 0.5
            
            closes = data['close'].dropna()
            if len(closes) < 5:
                return 0.5
            
            # 단기/장기 이동평균 비교
            short_ma = closes.tail(5).mean()
            long_ma = closes.tail(20).mean() if len(closes) >= 20 else closes.mean()
            
            # 트렌드 강도 계산
            trend_strength = (short_ma - long_ma) / long_ma
            
            # 0.0 ~ 1.0 범위로 정규화
            normalized_trend = (trend_strength + 1.0) / 2.0
            return max(0.0, min(1.0, normalized_trend))
            
        except Exception as e:
            logger.error(f"❌ 가격 트렌드 분석 실패: {e}")
            return 0.5
    
    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """변동성 분석"""
        try:
            if 'close' not in data.columns:
                return 0.5
            
            closes = data['close'].dropna()
            if len(closes) < 5:
                return 0.5
            
            # 변동성 계산 (표준편차 기반)
            returns = closes.pct_change().dropna()
            volatility = returns.std()
            
            # 변동성 수준 정규화 (0.0 ~ 1.0)
            normalized_volatility = min(1.0, volatility * 100)  # 1% = 1.0
            return normalized_volatility
            
        except Exception as e:
            logger.error(f"❌ 변동성 분석 실패: {e}")
            return 0.5
    
    def _analyze_volume_pattern(self, data: pd.DataFrame) -> float:
        """거래량 패턴 분석"""
        try:
            if 'volume' not in data.columns:
                return 0.5
            
            volumes = data['volume'].dropna()
            if len(volumes) < 5:
                return 0.5
            
            # 거래량 트렌드 분석
            recent_volume = volumes.tail(5).mean()
            avg_volume = volumes.mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 거래량 패턴 점수 (0.0 ~ 1.0)
            volume_score = min(1.0, volume_ratio / 2.0)  # 2배 = 1.0
            return volume_score
            
        except Exception as e:
            logger.error(f"❌ 거래량 패턴 분석 실패: {e}")
            return 0.5
    
    def _analyze_technical_signals(self, data: pd.DataFrame) -> float:
        """기술적 지표 분석"""
        try:
            signals = []
            
            # RSI 분석
            if 'rsi' in data.columns:
                rsi_values = data['rsi'].dropna()
                if len(rsi_values) > 0:
                    latest_rsi = rsi_values.iloc[-1]
                    # RSI를 0.0 ~ 1.0으로 정규화
                    rsi_signal = latest_rsi / 100.0
                    signals.append(rsi_signal)
            
            # MACD 분석
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd_values = data['macd'].dropna()
                macd_signal_values = data['macd_signal'].dropna()
                if len(macd_values) > 0 and len(macd_signal_values) > 0:
                    latest_macd = macd_values.iloc[-1]
                    latest_macd_signal = macd_signal_values.iloc[-1]
                    macd_diff = latest_macd - latest_macd_signal
                    # MACD 차이를 0.0 ~ 1.0으로 정규화
                    macd_signal = (macd_diff + 0.1) / 0.2  # -0.1 ~ 0.1 범위를 0.0 ~ 1.0으로
                    macd_signal = max(0.0, min(1.0, macd_signal))
                    signals.append(macd_signal)
            
            if not signals:
                return 0.5
            
            # 평균 신호 계산
            return sum(signals) / len(signals)
            
        except Exception as e:
            logger.error(f"❌ 기술적 지표 분석 실패: {e}")
            return 0.5
    
    def _classify_regime(self, regime_score: Dict[str, float]) -> str:
        """레짐 점수를 레짐 라벨로 분류"""
        try:
            # 종합 점수 계산
            overall_score = (
                regime_score['price_trend'] * 0.4 +
                regime_score['volatility'] * 0.2 +
                regime_score['volume_pattern'] * 0.2 +
                regime_score['technical_signals'] * 0.2
            )
            
            # 레짐 분류
            if overall_score < 0.15:
                return "extreme_bearish"
            elif overall_score < 0.3:
                return "bearish"
            elif overall_score < 0.4:
                return "sideways_bearish"
            elif overall_score < 0.6:
                return "neutral"
            elif overall_score < 0.7:
                return "sideways_bullish"
            elif overall_score < 0.85:
                return "bullish"
            else:
                return "extreme_bullish"
                
        except Exception as e:
            logger.error(f"❌ 레짐 분류 실패: {e}")
            return "neutral"
    
    def _calculate_regime_confidence(self, regime_score: Dict[str, float]) -> float:
        """레짐 신뢰도 계산"""
        try:
            # 각 점수의 일관성 계산
            scores = list(regime_score.values())
            score_variance = np.var(scores)
            
            # 낮은 분산 = 높은 신뢰도
            confidence = max(0.0, 1.0 - score_variance)
            
            return confidence
            
        except Exception as e:
            logger.error(f"❌ 레짐 신뢰도 계산 실패: {e}")
            return 0.5
    
    def route_strategies(self, coin: str, interval: str, strategies: List[Dict[str, Any]], 
                        candle_data: pd.DataFrame, use_accumulated_data: bool = True) -> List[RegimeRoutingResult]:
        """전략들을 레짐에 따라 라우팅 - 시간순 레짐 추적 및 실제 백테스트 검증
        
        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategies: 전략 리스트
            candle_data: 캔들 데이터
            use_accumulated_data: 누적 데이터 활용 여부 (여러 번 수행 시 이전 결과 반영) 🔥
        """
        try:
            logger.info(f"🔄 {coin}-{interval} 레짐 기반 라우팅 시작: {len(strategies)}개 전략")

            # 🔥 디버거 로깅: 라우팅 시작
            current_regime, _, _ = self.detect_current_regime(coin, interval, candle_data)
            if self.routing_debug:
                try:
                    self.routing_debug.log_routing_start(
                        coin=coin,
                        interval=interval,
                        regime=current_regime,
                        num_strategies=len(strategies)
                    )
                except Exception as debug_err:
                    logger.debug(f"디버거 로깅 실패 (무시): {debug_err}")

            # 🔥 시간순 레짐 추적: 전체 기간 동안 레짐 변화 추적
            regime_timeline = self._track_regime_timeline(candle_data)
            logger.info(f"📊 {len(regime_timeline)}개 시점의 레짐 추적 완료")
            
            # 🔥 레짐별 데이터 필터링 준비 (백테스트용)
            self.regime_timeline = regime_timeline
            self.candle_data = candle_data
            
            # 🔥 이전 레짐 라우팅 결과 로드 (누적 데이터 활용)
            accumulated_regime_performance = {}
            if use_accumulated_data:
                accumulated_regime_performance = self._load_accumulated_regime_performance(coin, interval)
                if accumulated_regime_performance:
                    logger.info(f"📊 누적 레짐 성능 데이터 로드: {len(accumulated_regime_performance)}개 전략")
            
            # 레짐별 전략 라우팅 및 검증
            routing_results = []
            
            # 🔥 병렬 백테스트를 위한 작업 준비
            all_regimes = ['extreme_bearish', 'bearish', 'sideways_bearish', 'neutral', 
                          'sideways_bullish', 'bullish', 'extreme_bullish']
            
            # 🔥 병렬 백테스트 실행 (전략별로 모든 레짐에 대해)
            # 🔥 워커 수를 보수적으로 설정 (메모리/CPU 과부하 방지)
            # 환경변수로 제어 가능하지만, 기본값은 2개로 제한
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # 🔥 보수적 설정: 최소 1개, 최대 4개 (메모리 사용량 제어)
            default_workers = min(4, max(1, cpu_count // 6))  # CPU 코어의 1/6 정도만 사용
            max_workers = min(
                int(os.getenv('REGIME_ROUTING_MAX_WORKERS', str(default_workers))),
                len(strategies) * len(all_regimes),
                4  # 🔥 최대 4개로 제한 (메모리 안전)
            )
            logger.info(f"🚀 병렬 백테스트 시작: {len(strategies)}개 전략 × {len(all_regimes)}개 레짐 = {len(strategies) * len(all_regimes)}개 작업 (워커: {max_workers}개, CPU: {cpu_count}코어)")
            
            # 전략별 레짐 평가 결과 저장
            strategy_regime_results = {}
            
            # 🔥 메모리 사용량 모니터링 (선택적)
            try:
                import psutil
                process = psutil.Process()
                initial_memory_mb = process.memory_info().rss / 1024 / 1024
                logger.debug(f"📊 초기 메모리 사용량: {initial_memory_mb:.1f}MB")
            except ImportError:
                psutil = None
                logger.debug("📊 psutil 없음, 메모리 모니터링 건너뜀")
            
            # 🔥 배치 크기 제한: 한 번에 너무 많은 작업을 제출하지 않도록 제한
            # (메모리 사용량을 제어하기 위해)
            batch_size = max_workers  # 🔥 워커 수와 동일하게 제한 (메모리 안전)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 백테스트 작업 제출 (배치 단위로 제한)
                future_to_task = {}
                submitted_count = 0
                
                for strategy_idx, strategy in enumerate(strategies):
                    strategy_id = strategy.get('id') or strategy.get('strategy_id') or str(hash(json.dumps(strategy, sort_keys=True)))
                    strategy_regime_results[strategy_id] = {}
                    
                    for regime_type in all_regimes:
                        # 🔥 배치 크기 제한: 너무 많은 작업이 대기 중이면 일부 완료 대기
                        if submitted_count >= batch_size:
                            # 일부 작업 완료 대기 (최대 10개)
                            completed_futures = [f for f in future_to_task.keys() if f.done()]
                            if len(completed_futures) < max_workers:
                                # 아직 완료된 작업이 적으면 잠시 대기
                                import time
                                time.sleep(0.01)  # 10ms 대기
                        
                        # 백테스트 작업 제출
                        future = executor.submit(
                            self._evaluate_strategy_for_regime,
                            strategy, regime_type, use_accumulated_data,
                            strategy_id, accumulated_regime_performance
                        )
                        future_to_task[future] = (strategy_id, regime_type, strategy_idx)
                        submitted_count += 1
                
                # 결과 수집
                completed = 0
                total_tasks = len(future_to_task)
                for future in as_completed(future_to_task):
                    strategy_id, regime_type, strategy_idx = future_to_task[future]
                    try:
                        result = future.result()
                        if strategy_id not in strategy_regime_results:
                            strategy_regime_results[strategy_id] = {}
                        strategy_regime_results[strategy_id][regime_type] = result
                        completed += 1
                        if completed % 50 == 0:
                            logger.debug(f"📊 백테스트 진행: {completed}/{total_tasks} 완료 ({completed*100//total_tasks}%)")
                    except Exception as e:
                        logger.warning(f"⚠️ 백테스트 실패 ({strategy_id}-{regime_type}): {e}")
                        # 실패 시 기본값 설정
                        if strategy_id not in strategy_regime_results:
                            strategy_regime_results[strategy_id] = {}
                        strategy_regime_results[strategy_id][regime_type] = {
                            'fitness': 0.5,
                            'performance': 0.0,
                            'adaptation': 0.5,
                            'total_score': 0.25,
                            'backtest_result': None
                        }
            
            logger.info(f"✅ 병렬 백테스트 완료: {completed}/{total_tasks} 작업")
            
            # 🔥 메모리 사용량 모니터링 (완료 후)
            try:
                if psutil:
                    final_memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_increase = final_memory_mb - initial_memory_mb
                    logger.debug(f"📊 최종 메모리 사용량: {final_memory_mb:.1f}MB (증가: {memory_increase:+.1f}MB)")
                    if memory_increase > 300:  # 🔥 300MB 이상 증가 시 경고 (임계값 낮춤)
                        logger.warning(f"⚠️ 메모리 사용량이 크게 증가했습니다: {memory_increase:.1f}MB (워커 수: {max_workers}개)")
            except:
                pass
            
            # 🔥 전략별로 최적 레짐 선택 및 라우팅 결과 생성
            for strategy_idx, strategy in enumerate(strategies):
                strategy_id = strategy.get('id') or strategy.get('strategy_id') or str(hash(json.dumps(strategy, sort_keys=True)))
                regime_scores = strategy_regime_results.get(strategy_id, {})
                
                if not regime_scores:
                    logger.warning(f"⚠️ 전략 {strategy_id}의 레짐 평가 결과 없음")
                    continue
                
                # 최적 레짐 선택
                optimal_regime = max(regime_scores.keys(), key=lambda r: regime_scores[r]['total_score'])
                current_regime_info = regime_timeline[-1] if regime_timeline else {'regime': 'neutral', 'confidence': 0.5}
                regime_confidence = current_regime_info.get('confidence', 0.5)
                
                # 최적 레짐 정보
                optimal_info = regime_scores[optimal_regime]
                
                # 🔥 최적 레짐의 백테스트 결과를 결과에 포함
                optimal_backtest = regime_scores[optimal_regime].get('backtest_result')
                
                # 🔥 예측 정확도 추출 (백테스트 결과에서)
                predictive_accuracy = 0.0
                if optimal_backtest and 'predictive_accuracy' in optimal_backtest:
                    predictive_accuracy = optimal_backtest['predictive_accuracy']
                
                routing_result = RegimeRoutingResult(
                    coin=coin,
                    interval=interval,
                    regime=optimal_regime,
                    routed_strategy=strategy,
                    routing_confidence=regime_confidence,
                    routing_score=optimal_info['total_score'],
                    regime_performance=optimal_info['performance'],
                    regime_adaptation=optimal_info['adaptation'],
                    created_at=datetime.now().isoformat(),
                    predictive_accuracy=predictive_accuracy,  # 🔥 예측 정확도 전달
                    backtest_result=optimal_backtest  # 🔥 백테스트 결과 전달
                )
                
                routing_results.append(routing_result)
            
            # 성능 지표 출력
            if routing_results:
                # 레짐별 집계 (백테스트 결과 포함)
                regime_stats = {}
                for idx, result in enumerate(routing_results):
                    # 전략의 최적 레짐에 대한 백테스트 결과 가져오기
                    strategy = result.routed_strategy
                    optimal_regime = result.regime
                    
                    # 🔥 최적 레짐에 대한 백테스트 결과 수집 (이미 계산된 경우 재사용)
                    backtest_result = getattr(result, 'backtest_result', None)
                    if not backtest_result:
                        # 백테스트 결과가 없으면 다시 계산
                        backtest_result = self._backtest_strategy_by_regime(strategy, optimal_regime)
                    
                    if optimal_regime not in regime_stats:
                        regime_stats[optimal_regime] = {
                            'count': 0,
                            'total_score': 0.0,
                            'total_performance': 0.0,
                            'total_confidence': 0.0,
                            # 🔥 백테스트 결과 집계
                            'total_trades': 0,
                            'total_profit': 0.0,
                            'total_wins': 0,
                            'backtest_count': 0  # 유효한 백테스트 결과 수
                        }
                    regime_stats[optimal_regime]['count'] += 1
                    regime_stats[optimal_regime]['total_score'] += result.routing_score
                    regime_stats[optimal_regime]['total_performance'] += result.regime_performance
                    regime_stats[optimal_regime]['total_confidence'] += result.routing_confidence
                    
                    # 🔥 백테스트 결과 집계
                    if backtest_result and backtest_result.get('trades', 0) > 0:
                        regime_stats[optimal_regime]['total_trades'] += backtest_result.get('trades', 0)
                        regime_stats[optimal_regime]['total_profit'] += backtest_result.get('profit', 0.0)
                        regime_stats[optimal_regime]['total_wins'] += backtest_result.get('wins', 0)
                        regime_stats[optimal_regime]['backtest_count'] += 1
                
                # 평균 계산 및 상세 로그 출력
                for regime, stats in regime_stats.items():
                    avg_score = stats['total_score'] / stats['count']
                    avg_perf = stats['total_performance'] / stats['count']
                    avg_conf = stats['total_confidence'] / stats['count']
                    
                    # 🔥 백테스트 통계 계산
                    if stats['backtest_count'] > 0:
                        avg_trades = stats['total_trades'] / stats['backtest_count']
                        avg_profit = stats['total_profit'] / stats['backtest_count']
                        total_trades = stats['total_trades']
                        total_wins = stats['total_wins']
                        win_rate = total_wins / total_trades if total_trades > 0 else 0.0
                        avg_profit_per_trade = avg_profit / avg_trades if avg_trades > 0 else 0.0
                        
                        # 수익비 계산 (Profit Factor: 총 수익 / 총 손실)
                        # 단순화: 양수 수익률 기반으로 추정
                        profit_factor = 1.0 + (avg_profit * 10) if avg_profit > 0 else 0.5
                        profit_factor = max(0.0, min(5.0, profit_factor))  # 0 ~ 5 범위 제한
                        
                        logger.info(f"📊 레짐 {regime}: {stats['count']}개 전략, 평균 점수 {avg_score:.3f}, "
                                  f"평균 성능 {avg_perf:.3f}, 신뢰도 {avg_conf:.3f}")
                        logger.info(f"   💰 백테스트 성과 (유효 {stats['backtest_count']}개): "
                                  f"거래 {total_trades}회, 승률 {win_rate:.1%}, "
                                  f"평균 수익률 {avg_profit:.2%}, 수익비 {profit_factor:.2f}, "
                                  f"거래당 수익 {avg_profit_per_trade:.4f}%")
                    else:
                        logger.info(f"📊 레짐 {regime}: {stats['count']}개 전략, 평균 점수 {avg_score:.3f}, "
                                  f"평균 성능 {avg_perf:.3f}, 신뢰도 {avg_conf:.3f} (백테스트 데이터 없음)")
            
            # 🔥 현재 레짐 감지 및 전환 확률 읽기
            current_regime, regime_confidence, regime_transition_prob = self.detect_current_regime(coin, interval, candle_data)

            # ✅ 레짐 라우팅 결과 저장은 orchestrator에서 centralized save 함수로 처리됨
            # (save_regime_routing_results가 rl_strategies.db에 올바른 스키마로 저장)

            # 🔥 디버거 로깅: 라우팅 종료
            if self.routing_debug:
                try:
                    # 통계 계산
                    total_strategies = len(routing_results)
                    routed_strategies = len([r for r in routing_results if r.routing_score > 0.5])

                    self.routing_debug.log_routing_end(
                        coin=coin,
                        interval=interval,
                        regime=current_regime,
                        total_strategies=total_strategies,
                        routed_strategies=routed_strategies,
                        avg_routing_score=sum([r.routing_score for r in routing_results]) / total_strategies if total_strategies > 0 else 0.0
                    )
                except Exception as debug_err:
                    logger.debug(f"디버거 로깅 실패 (무시): {debug_err}")

            logger.info(f"✅ {coin}-{interval} 레짐 라우팅 완료: {len(routing_results)}개 결과 (🔥 백테스트 검증 완료)")
            return routing_results
            
        except Exception as e:
            logger.error(f"❌ 레짐 라우팅 실패: {e}")
            return []
    
    def _load_accumulated_regime_performance(self, coin: str, interval: str, days: int = 30) -> Dict[str, Dict[str, float]]:
        """이전 레짐 라우팅 결과 로드 (누적 성능 데이터) 🔥"""
        try:
            from rl_pipeline.core.env import LEARNING_RESULTS_DB_PATH
            from rl_pipeline.db.connection_pool import get_optimized_db_connection

            # strategy_id -> {regime -> 평균 성능}
            accumulated = {}

            # learning_results는 strategies로 통합됨
            with get_optimized_db_connection("strategies") as conn:
                cursor = conn.cursor()
                
                # 최근 N일간 레짐 라우팅 결과 조회
                cursor.execute("""
                    SELECT 
                        routed_strategy,
                        regime,
                        AVG(regime_performance) as avg_performance,
                        COUNT(*) as test_count
                    FROM regime_routing_results
                    WHERE symbol = ? AND interval = ?
                      AND created_at >= datetime('now', '-' || ? || ' days')
                    GROUP BY routed_strategy, regime
                    HAVING test_count >= 1
                """, (coin, interval, days))
                
                results = cursor.fetchall()
                
                for row in results:
                    try:
                        strategy_json = row[0]
                        regime = row[1]
                        avg_performance = row[2]
                        test_count = row[3]
                        
                        # 전략 ID 추출 (JSON에서)
                        strategy_data = json.loads(strategy_json)
                        strategy_id = strategy_data.get('id') or strategy_data.get('strategy_id') or str(hash(json.dumps(strategy_data, sort_keys=True)))
                        
                        if strategy_id not in accumulated:
                            accumulated[strategy_id] = {}
                        
                        accumulated[strategy_id][regime] = float(avg_performance)
                        
                    except Exception as e:
                        logger.debug(f"⚠️ 누적 데이터 파싱 실패: {e}")
                        continue
                
                logger.debug(f"📊 누적 레짐 성능 로드: {len(accumulated)}개 전략, {sum(len(v) for v in accumulated.values())}개 레짐 매핑")
            
            return accumulated
            
        except Exception as e:
            logger.debug(f"⚠️ 누적 레짐 성능 로드 실패: {e}")
            return {}
    
    def _save_regime_routing_results(
        self,
        coin: str,
        interval: str,
        regime_detected: str,
        regime_confidence: float,
        regime_transition_prob: float,
        matched_strategies: int
    ) -> bool:
        """레짐 라우팅 결과를 DB에 저장 - rl_strategies.db에 저장"""
        try:
            from rl_pipeline.db.connection_pool import get_optimized_db_connection

            # 🔥 rl_strategies DB에 저장 (learning_results는 strategies로 통합됨)
            with get_optimized_db_connection("strategies") as conn:
                cursor = conn.cursor()

                # 🔥 실제 테이블 스키마에 맞게 수정 (regime_detected, regime_confidence, regime_transition_prob, matched_strategies)
                cursor.execute("""
                    INSERT INTO regime_routing_results (
                        coin, interval, regime_detected, regime_confidence,
                        regime_transition_prob, matched_strategies, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    coin, interval, regime_detected, regime_confidence,
                    regime_transition_prob, matched_strategies
                ))

                conn.commit()
                logger.info(f"✅ 레짐 라우팅 결과 저장: {coin}-{interval}, 레짐={regime_detected}, "
                          f"전환 확률={regime_transition_prob:.2%}, 매칭 전략={matched_strategies}개")
                return True

        except Exception as e:
            logger.error(f"❌ 레짐 라우팅 결과 저장 실패: {e}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")
            return False
    
    def get_strategy_regime_history(self, coin: str, interval: str, strategy_id: str) -> Dict[str, Any]:
        """전략의 레짐 매핑 이력 조회 (여러 번 수행 시 어떤 레짐에 배치되었는지) 🔥"""
        try:
            from rl_pipeline.core.env import LEARNING_RESULTS_DB_PATH
            from rl_pipeline.db.connection_pool import get_optimized_db_connection

            regime_history = {}

            # learning_results는 strategies로 통합됨
            with get_optimized_db_connection("strategies") as conn:
                cursor = conn.cursor()
                
                # 전략의 레짐 라우팅 이력 조회
                cursor.execute("""
                    SELECT 
                        regime,
                        COUNT(*) as count,
                        AVG(regime_performance) as avg_performance,
                        AVG(routing_score) as avg_score,
                        MAX(created_at) as last_seen
                    FROM regime_routing_results
                    WHERE symbol = ? AND interval = ?
                      AND (routed_strategy LIKE ? OR routed_strategy LIKE ?)
                    GROUP BY regime
                    ORDER BY count DESC
                """, (coin, interval, f'%"id":"{strategy_id}"%', f'%"strategy_id":"{strategy_id}"%'))
                
                results = cursor.fetchall()
                
                for row in results:
                    regime = row[0]
                    count = row[1]
                    avg_performance = row[2]
                    avg_score = row[3]
                    last_seen = row[4]
                    
                    regime_history[regime] = {
                        'count': count,
                        'avg_performance': float(avg_performance) if avg_performance else 0.0,
                        'avg_score': float(avg_score) if avg_score else 0.0,
                        'last_seen': last_seen
                    }
            
            return regime_history
            
        except Exception as e:
            logger.debug(f"⚠️ 전략 레짐 이력 조회 실패: {e}")
            return {}
    
    def _track_regime_timeline(self, candle_data: pd.DataFrame, window_size: int = 20) -> List[Dict[str, Any]]:
        """시간순 레짐 추적: 전체 기간 동안 레짐 변화를 추적"""
        try:
            regime_timeline = []

            # 🆕 캔들 데이터에 'regime' 컬럼이 있으면 DB 레짐 데이터 사용 (3단계 매핑됨)
            if 'regime' in candle_data.columns:
                logger.info("✅ DB에서 로드된 레짐 데이터 사용 (3단계 매핑됨)")

                for i in range(len(candle_data)):
                    row = candle_data.iloc[i]

                    # DB 레짐 데이터 사용
                    regime = str(row['regime']) if pd.notna(row['regime']) else 'ranging'

                    # 신뢰도 로드 (있으면)
                    confidence = 0.8  # 기본값
                    if 'regime_confidence' in candle_data.columns:
                        if pd.notna(row['regime_confidence']):
                            confidence = float(row['regime_confidence'])

                    regime_timeline.append({
                        'regime': regime,
                        'confidence': confidence,
                        'score': {},  # DB 데이터 사용 시 점수는 비어있음
                        'timestamp': candle_data.index[i] if hasattr(candle_data.index[i], 'isoformat') else str(candle_data.index[i])
                    })

                logger.info(f"📊 레짐 추적 (DB): {len(regime_timeline)}개 시점, 레짐 분포: {self._get_regime_distribution(regime_timeline)}")
                return regime_timeline

            # 🔄 폴백: 'regime' 컬럼이 없으면 기존 방식으로 계산 후 7→3 매핑
            logger.debug("⚠️ 'regime' 컬럼 없음 - 레짐 재계산 후 매핑")

            # Rolling window로 시간별 레짐 감지
            for i in range(window_size, len(candle_data)):
                window_data = candle_data.iloc[max(0, i-window_size):i]

                regime_score = self._calculate_regime_score(window_data)
                regime_label_7stage = self._classify_regime(regime_score)
                confidence = self._calculate_regime_confidence(regime_score)

                # 🆕 7단계 → 3단계 매핑 적용
                regime = classify_regime_from_old(regime_label_7stage)

                regime_timeline.append({
                    'regime': regime,
                    'confidence': confidence,
                    'score': regime_score,
                    'timestamp': candle_data.index[i] if hasattr(candle_data.index[i], 'isoformat') else str(candle_data.index[i])
                })

            logger.info(f"📊 레짐 추적 (계산): {len(regime_timeline)}개 시점, 레짐 분포: {self._get_regime_distribution(regime_timeline)}")
            return regime_timeline

        except Exception as e:
            logger.error(f"❌ 레짐 타임라인 추적 실패: {e}")
            return []
    
    def _get_regime_distribution(self, regime_timeline: List[Dict[str, Any]]) -> Dict[str, int]:
        """레짐 분포 계산"""
        distribution = {}
        for item in regime_timeline:
            regime = item['regime']
            distribution[regime] = distribution.get(regime, 0) + 1
        return distribution
    
    def _evaluate_strategy_regime_fitness(self, strategy: Dict[str, Any], regime: str) -> float:
        """전략의 레짐 적합성 평가"""
        try:
            # 전략 파라미터 추출
            rsi_min = strategy.get('rsi_min', 30)
            rsi_max = strategy.get('rsi_max', 70)
            stop_loss = strategy.get('stop_loss_pct', 0.02)
            take_profit = strategy.get('take_profit_pct', 0.05)
            
            # 레짐별 적합성 점수 계산
            fitness_scores = {
                'extreme_bearish': self._calculate_bearish_fitness(rsi_min, rsi_max, stop_loss, take_profit),
                'bearish': self._calculate_bearish_fitness(rsi_min, rsi_max, stop_loss, take_profit),
                'sideways_bearish': self._calculate_sideways_fitness(rsi_min, rsi_max, stop_loss, take_profit),
                'neutral': self._calculate_neutral_fitness(rsi_min, rsi_max, stop_loss, take_profit),
                'sideways_bullish': self._calculate_sideways_fitness(rsi_min, rsi_max, stop_loss, take_profit),
                'bullish': self._calculate_bullish_fitness(rsi_min, rsi_max, stop_loss, take_profit),
                'extreme_bullish': self._calculate_bullish_fitness(rsi_min, rsi_max, stop_loss, take_profit)
            }
            
            return fitness_scores.get(regime, 0.5)
            
        except Exception as e:
            logger.error(f"❌ 레짐 적합성 평가 실패: {e}")
            return 0.5
    
    def _calculate_bearish_fitness(self, rsi_min: float, rsi_max: float, stop_loss: float, take_profit: float) -> float:
        """베어리시 레짐 적합성 계산"""
        # 베어리시에서는 낮은 RSI 매수, 높은 손절, 낮은 익절이 유리
        rsi_score = (30 - rsi_min) / 20.0  # 낮은 RSI 매수 선호
        stop_loss_score = min(1.0, stop_loss / 0.05)  # 높은 손절 선호
        take_profit_score = (0.03 - take_profit) / 0.02  # 낮은 익절 선호
        
        return (rsi_score + stop_loss_score + take_profit_score) / 3.0
    
    def _calculate_bullish_fitness(self, rsi_min: float, rsi_max: float, stop_loss: float, take_profit: float) -> float:
        """불리시 레짐 적합성 계산"""
        # 불리시에서는 높은 RSI 매도, 낮은 손절, 높은 익절이 유리
        rsi_score = (rsi_max - 70) / 20.0  # 높은 RSI 매도 선호
        stop_loss_score = (0.02 - stop_loss) / 0.01  # 낮은 손절 선호
        take_profit_score = (take_profit - 0.03) / 0.02  # 높은 익절 선호
        
        return (rsi_score + stop_loss_score + take_profit_score) / 3.0
    
    def _calculate_sideways_fitness(self, rsi_min: float, rsi_max: float, stop_loss: float, take_profit: float) -> float:
        """사이드웨이 레짐 적합성 계산"""
        # 사이드웨이에서는 균형잡힌 파라미터가 유리
        rsi_range_score = 1.0 - abs((rsi_min + rsi_max) / 2.0 - 50) / 50.0  # 중간 RSI 선호
        stop_loss_score = 1.0 - abs(stop_loss - 0.02) / 0.02  # 중간 손절 선호
        take_profit_score = 1.0 - abs(take_profit - 0.04) / 0.02  # 중간 익절 선호
        
        return (rsi_range_score + stop_loss_score + take_profit_score) / 3.0
    
    def _calculate_neutral_fitness(self, rsi_min: float, rsi_max: float, stop_loss: float, take_profit: float) -> float:
        """중립 레짐 적합성 계산"""
        # 중립에서는 보수적인 파라미터가 유리
        return self._calculate_sideways_fitness(rsi_min, rsi_max, stop_loss, take_profit)
    
    def _calculate_routing_score(self, strategy: Dict[str, Any], regime: str, regime_fitness: float) -> float:
        """라우팅 점수 계산"""
        try:
            # 전략의 기본 성능 점수
            base_performance = strategy.get('profit', 0.0)
            win_rate = strategy.get('win_rate', 0.5)
            trades_count = strategy.get('trades_count', 0)
            
            # 기본 성능 정규화
            performance_score = max(0.0, min(1.0, (base_performance + 0.1) / 0.2))  # -0.1 ~ 0.1을 0.0 ~ 1.0으로
            win_rate_score = win_rate
            trades_score = min(1.0, trades_count / 10.0)  # 10회 거래 = 1.0
            
            # 종합 점수 계산
            routing_score = (
                performance_score * 0.3 +
                win_rate_score * 0.3 +
                trades_score * 0.2 +
                regime_fitness * 0.2
            )
            
            return routing_score
            
        except Exception as e:
            logger.error(f"❌ 라우팅 점수 계산 실패: {e}")
            return 0.5
    
    def _predict_regime_performance(self, strategy: Dict[str, Any], regime: str) -> float:
        """레짐별 성능 예측 - 실제 백테스트 기반 🔥"""
        try:
            # 🔥 실제 백테스트 실행 (레짐별 데이터 필터링)
            backtest_result = self._backtest_strategy_by_regime(strategy, regime)
            return self._predict_regime_performance_with_backtest(strategy, regime, backtest_result)
            
        except Exception as e:
            logger.debug(f"⚠️ {regime} 백테스트 실패, 폴백 사용: {e}")
            return self._predict_regime_performance_fallback(strategy, regime)
    
    def _predict_regime_performance_with_backtest(self, strategy: Dict[str, Any], regime: str, backtest_result: Optional[Dict[str, Any]]) -> float:
        """레짐별 성능 예측 (백테스트 결과 받아서 처리) 🔥"""
        try:
            if backtest_result and backtest_result.get('trades', 0) > 0:
                # 실제 백테스트 결과 사용
                actual_profit = backtest_result.get('profit', 0.0)
                actual_win_rate = backtest_result.get('win_rate', 0.0)
                trades = backtest_result.get('trades', 0)
                
                # 종합 성능 점수 (실제 백테스트 기반)
                # 수익률 정규화: -0.1 ~ 0.1 → 0.0 ~ 1.0
                normalized_profit = max(0.0, min(1.0, (actual_profit + 0.1) / 0.2))
                performance_score = (normalized_profit * 0.6 + actual_win_rate * 0.4)
                
                logger.debug(f"  🔥 {regime} 백테스트: {trades}거래, 수익 {actual_profit:.2%}, 승률 {actual_win_rate:.1%}")
                
                return max(0.0, min(1.0, performance_score))
            else:
                # 백테스트 데이터 부족 시 기존 방식 사용 (폴백)
                return self._predict_regime_performance_fallback(strategy, regime)
            
        except Exception as e:
            logger.debug(f"⚠️ {regime} 백테스트 결과 처리 실패: {e}")
            return self._predict_regime_performance_fallback(strategy, regime)
    
    def _predict_regime_performance_fallback(self, strategy: Dict[str, Any], regime: str) -> float:
        """레짐별 성능 예측 (폴백: 이론적 계산)"""
        try:
            # 기본 성능
            base_profit = strategy.get('profit', 0.0)
            base_win_rate = strategy.get('win_rate', 0.5)
            
            # 레짐별 성능 조정 계수
            regime_multipliers = {
                'extreme_bearish': 0.8,
                'bearish': 0.9,
                'sideways_bearish': 0.95,
                'neutral': 1.0,
                'sideways_bullish': 1.05,
                'bullish': 1.1,
                'extreme_bullish': 1.2
            }
            
            multiplier = regime_multipliers.get(regime, 1.0)
            
            # 예측 성능 계산
            predicted_profit = base_profit * multiplier
            predicted_win_rate = min(1.0, base_win_rate * multiplier)
            
            # 종합 성능 점수
            performance_score = (predicted_profit + predicted_win_rate) / 2.0
            
            return max(0.0, min(1.0, performance_score))
            
        except Exception as e:
            logger.error(f"❌ 레짐 성능 예측 실패: {e}")
            return 0.5
    
    def _evaluate_strategy_for_regime(self, strategy: Dict[str, Any], regime_type: str, 
                                      use_accumulated_data: bool, strategy_id: str,
                                      accumulated_regime_performance: Dict[str, Any]) -> Dict[str, Any]:
        """🔥 병렬 실행용: 전략의 특정 레짐에 대한 평가 (헬퍼 함수)"""
        try:
            regime_fitness = self._evaluate_strategy_regime_fitness(strategy, regime_type)
            
            # 🔥 백테스트 결과 가져오기 (성능 평가와 로깅을 위해)
            backtest_result = self._backtest_strategy_by_regime(strategy, regime_type)
            regime_performance = self._predict_regime_performance_with_backtest(strategy, regime_type, backtest_result)
            regime_adaptation = self._calculate_regime_adaptation(strategy, regime_type)
            
            # 🔥 누적 성능 데이터 반영 (여러 번 수행 시)
            if use_accumulated_data and strategy_id in accumulated_regime_performance:
                if regime_type in accumulated_regime_performance[strategy_id]:
                    historical_performance = accumulated_regime_performance[strategy_id][regime_type]
                    # 누적 성능 가중 평균 (최근 70% + 이전 30%)
                    regime_performance = regime_performance * 0.7 + historical_performance * 0.3
            
            # 종합 점수
            total_score = (regime_fitness * 0.4 + regime_performance * 0.4 + regime_adaptation * 0.2)
            
            return {
                'fitness': regime_fitness,
                'performance': regime_performance,
                'adaptation': regime_adaptation,
                'total_score': total_score,
                'backtest_result': backtest_result
            }
        except Exception as e:
            logger.debug(f"⚠️ 레짐 평가 실패 ({strategy_id}-{regime_type}): {e}")
            return {
                'fitness': 0.5,
                'performance': 0.0,
                'adaptation': 0.5,
                'total_score': 0.25,
                'backtest_result': None
            }
    
    def _backtest_strategy_by_regime(self, strategy: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """레짐별 실제 백테스트 실행 (캐싱 적용) 🔥"""
        try:
            # 레짐별 데이터 필터링
            regime_data = self._filter_data_by_regime(regime)
            
            if regime_data is None or len(regime_data) < 20:  # 🔥 최소 데이터 요구사항 완화 (50 → 20)
                # 최소 데이터 부족
                logger.debug(f"⚠️ {regime} 레짐 데이터 부족: {len(regime_data) if regime_data is not None else 0}개 (최소 20개 필요)")
                return None
            
            # 🔥 캐시 확인 (성능 최적화)
            from rl_pipeline.analysis.backtest_cache import get_backtest_cache
            cache = get_backtest_cache()
            cached_result = cache.get(strategy, regime_data, regime)
            
            if cached_result:
                logger.debug(f"✅ 백테스트 캐시 히트: {regime}")
                return cached_result
            
            # 백테스트 실행
            from rl_pipeline.strategy.router import execute_simple_backtest
            
            trades, profit, wins, predictive_accuracy = execute_simple_backtest(strategy, regime_data)
            
            win_rate = wins / trades if trades > 0 else 0.0
            
            result = {
                'trades': trades,
                'profit': profit,
                'wins': wins,
                'win_rate': win_rate,
                'predictive_accuracy': predictive_accuracy,  # 🔥 예측 정확도 추가
                'data_points': len(regime_data)
            }
            
            # 🔥 캐시 저장
            cache.set(strategy, regime_data, result, regime)
            
            return result
            
        except Exception as e:
            logger.debug(f"⚠️ {regime} 백테스트 실행 실패: {e}")
            return None
    
    def _filter_data_by_regime(self, target_regime: str) -> Optional[pd.DataFrame]:
        """레짐별 캔들 데이터 필터링 🔥"""
        try:
            if not hasattr(self, 'regime_timeline') or not hasattr(self, 'candle_data'):
                return None
            
            if self.regime_timeline is None or len(self.regime_timeline) == 0:
                return None
            
            if self.candle_data is None or len(self.candle_data) == 0:
                return None
            
            # 🔥 7단계 레짐을 3단계로 매핑 (백테스트용)
            # 레짐 타임라인은 3단계 레짐(ranging/trending)을 사용하므로 매핑 필요
            from rl_pipeline.core.regime_classifier import REGIME_MAPPING
            # 역방향 매핑: 7단계 레짐 → 3단계 레짐
            regime_7to3 = {
                'extreme_bearish': 'trending',
                'bearish': 'trending',
                'sideways_bearish': 'ranging',
                'neutral': 'ranging',
                'sideways_bullish': 'ranging',
                'bullish': 'trending',
                'extreme_bullish': 'trending'
            }
            mapped_regime = regime_7to3.get(target_regime, target_regime)
            
            # 레짐 타임라인에서 해당 레짐인 시점 찾기
            regime_indices = []
            window_size = 20  # 레짐 추적 시 사용한 window_size와 동일
            
            for i, regime_info in enumerate(self.regime_timeline):
                timeline_regime = regime_info.get('regime')
                # 3단계 레짐과 매핑된 레짐 비교
                if timeline_regime == mapped_regime:
                    # 해당 시점의 인덱스 계산 (regime_timeline은 window_size부터 시작)
                    data_index = window_size + i
                    if data_index < len(self.candle_data):
                        regime_indices.append(data_index)
            
            if len(regime_indices) == 0:
                # 해당 레짐 데이터 없음
                logger.debug(f"⚠️ {target_regime} (매핑: {mapped_regime}) 레짐 데이터 없음")
                return None
            
            # 레짐별 데이터 필터링
            # 연속된 구간으로 그룹화하여 더 많은 데이터 확보
            regime_data_list = []
            for idx in regime_indices:
                # 각 시점 주변 데이터도 포함 (윈도우 확장)
                start_idx = max(0, idx - 5)
                end_idx = min(len(self.candle_data), idx + 5)
                regime_data_list.append(self.candle_data.iloc[start_idx:end_idx])
            
            if len(regime_data_list) == 0:
                return None
            
            # 데이터 결합 및 중복 제거
            regime_data = pd.concat(regime_data_list, ignore_index=False)
            regime_data = regime_data.drop_duplicates()
            regime_data = regime_data.sort_index()
            
            logger.debug(f"  📊 {target_regime} 필터링: {len(regime_indices)}개 시점 → {len(regime_data)}개 캔들")
            
            return regime_data if len(regime_data) >= 20 else None
            
        except Exception as e:
            logger.debug(f"⚠️ {target_regime} 데이터 필터링 실패: {e}")
            return None
    
    def _calculate_regime_adaptation(self, strategy: Dict[str, Any], regime: str) -> float:
        """레짐 적응도 계산"""
        try:
            # 전략의 유연성 지표들
            rsi_range = strategy.get('rsi_max', 70) - strategy.get('rsi_min', 30)
            stop_loss = strategy.get('stop_loss_pct', 0.02)
            take_profit = strategy.get('take_profit_pct', 0.05)
            
            # 적응도 점수 계산
            rsi_adaptation = min(1.0, rsi_range / 40.0)  # 넓은 RSI 범위 = 높은 적응도
            risk_adaptation = 1.0 - abs(stop_loss - 0.02) / 0.02  # 중간 손절 = 높은 적응도
            reward_adaptation = 1.0 - abs(take_profit - 0.04) / 0.02  # 중간 익절 = 높은 적응도
            
            adaptation_score = (rsi_adaptation + risk_adaptation + reward_adaptation) / 3.0
            
            return adaptation_score
            
        except Exception as e:
            logger.error(f"❌ 레짐 적응도 계산 실패: {e}")
            return 0.5

def create_regime_routing_strategies(coin: str, interval: str, strategies: List[Dict[str, Any]], 
                                   candle_data: pd.DataFrame, use_accumulated_data: bool = True) -> List[RegimeRoutingResult]:
    """레짐 라우팅 전략 생성
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        strategies: 전략 리스트
        candle_data: 캔들 데이터
        use_accumulated_data: 누적 데이터 활용 여부 (여러 번 수행 시 이전 결과 반영) 🔥
    """
    try:
        router = RegimeRouter()
        return router.route_strategies(coin, interval, strategies, candle_data, use_accumulated_data=use_accumulated_data)
        
    except Exception as e:
        logger.error(f"❌ 레짐 라우팅 전략 생성 실패: {e}")
        return []

def route_strategies_by_regime(coin: str, interval: str, strategies: List[Dict[str, Any]], 
                             candle_data: pd.DataFrame, use_accumulated_data: bool = True) -> Dict[str, List[RegimeRoutingResult]]:
    """레짐별 전략 라우팅
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        strategies: 전략 리스트
        candle_data: 캔들 데이터
        use_accumulated_data: 누적 데이터 활용 여부 (여러 번 수행 시 이전 결과 반영) 🔥
    """
    try:
        router = RegimeRouter()
        routing_results = router.route_strategies(coin, interval, strategies, candle_data, use_accumulated_data=use_accumulated_data)
        
        # 레짐별로 그룹화
        regime_groups = {}
        for result in routing_results:
            regime = result.regime
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(result)
        
        return regime_groups
        
    except Exception as e:
        logger.error(f"❌ 레짐별 전략 라우팅 실패: {e}")
        return {}

def analyze_regime_performance(routing_results: List[RegimeRoutingResult]) -> Dict[str, Any]:
    """레짐 성능 분석"""
    try:
        if not routing_results:
            return {}
        
        # 레짐별 성능 집계
        regime_stats = {}
        
        for result in routing_results:
            regime = result.regime
            if regime not in regime_stats:
                regime_stats[regime] = {
                    'count': 0,
                    'total_score': 0.0,
                    'total_performance': 0.0,
                    'total_adaptation': 0.0,
                    'confidences': []
                }
            
            regime_stats[regime]['count'] += 1
            regime_stats[regime]['total_score'] += result.routing_score
            regime_stats[regime]['total_performance'] += result.regime_performance
            regime_stats[regime]['total_adaptation'] += result.regime_adaptation
            regime_stats[regime]['confidences'].append(result.routing_confidence)
        
        # 평균 계산
        analysis_result = {}
        for regime, stats in regime_stats.items():
            count = stats['count']
            analysis_result[regime] = {
                'strategy_count': count,
                'avg_routing_score': stats['total_score'] / count,
                'avg_performance': stats['total_performance'] / count,
                'avg_adaptation': stats['total_adaptation'] / count,
                'avg_confidence': sum(stats['confidences']) / len(stats['confidences'])
            }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ 레짐 성능 분석 실패: {e}")
        return {}

def get_strategy_multi_regime_analysis(coin: str, interval: str, strategy_id: str) -> Dict[str, Any]:
    """전략의 다중 레짐 적합성 분석 (여러 번 수행 시 어떤 레짐에서 좋은 성능을 보였는지) 🔥"""
    try:
        router = RegimeRouter()
        history = router.get_strategy_regime_history(coin, interval, strategy_id)
        
        if not history:
            return {'strategy_id': strategy_id, 'regime_count': 0, 'multi_regime': False}
        
        # 레짐별 성능 분석
        regime_count = len(history)
        best_regime = max(history.keys(), key=lambda r: history[r]['avg_performance'])
        worst_regime = min(history.keys(), key=lambda r: history[r]['avg_performance'])
        
        # 다중 레짐 적합성 판단 (2개 이상 레짐에서 좋은 성능)
        good_performance_count = sum(1 for h in history.values() if h['avg_performance'] > 0.6)
        multi_regime = good_performance_count >= 2
        
        return {
            'strategy_id': strategy_id,
            'regime_count': regime_count,
            'regimes': list(history.keys()),
            'best_regime': best_regime,
            'best_performance': history[best_regime]['avg_performance'],
            'worst_regime': worst_regime,
            'worst_performance': history[worst_regime]['avg_performance'],
            'good_performance_count': good_performance_count,
            'multi_regime': multi_regime,
            'regime_history': history
        }
        
    except Exception as e:
        logger.error(f"❌ 다중 레짐 분석 실패: {e}")
        return {'strategy_id': strategy_id, 'error': str(e)}

def calculate_regime_routing_quality(routing_results: List[RegimeRoutingResult]) -> float:
    """레짐 라우팅 품질 점수 계산 🔥"""
    try:
        if not routing_results:
            return 0.0
        
        # 레짐별 전략 수 균형 점수 (모든 레짐에 전략이 분산되어 있는지)
        regime_counts = {}
        for result in routing_results:
            regime = result.regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # 레짐 수 (7개 레짐 중 몇 개에 전략이 있는지)
        regimes_covered = len(regime_counts)
        coverage_score = regimes_covered / 7.0  # 7개 레짐 중 몇 개 커버하는지
        
        # 레짐별 전략 수 균형도 (표준편차가 낮을수록 좋음)
        if len(regime_counts) > 0:
            avg_count = sum(regime_counts.values()) / len(regime_counts)
            variance = sum((count - avg_count) ** 2 for count in regime_counts.values()) / len(regime_counts)
            balance_score = max(0.0, 1.0 - (variance / (avg_count ** 2 + 1)))  # 표준편차가 작을수록 1.0에 가까움
        else:
            balance_score = 0.0
        
        # 평균 라우팅 점수
        avg_routing_score = sum(r.routing_score for r in routing_results) / len(routing_results)
        
        # 평균 성능 점수
        avg_performance = sum(r.regime_performance for r in routing_results) / len(routing_results)
        
        # 종합 품질 점수
        quality_score = (
            coverage_score * 0.2 +      # 레짐 커버리지 20%
            balance_score * 0.2 +        # 레짐 균형도 20%
            avg_routing_score * 0.3 +    # 평균 라우팅 점수 30%
            avg_performance * 0.3        # 평균 성능 30%
        )
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception as e:
        logger.error(f"❌ 레짐 라우팅 품질 계산 실패: {e}")
        return 0.5

def route_strategies_with_iteration_control(
    coin: str,
    interval: str,
    strategies: List[Dict[str, Any]],
    candle_data: pd.DataFrame,
    max_iterations: int = 5,
    quality_threshold: float = 0.75,
    improvement_threshold: float = 0.02,
    min_iterations: int = 1,
    use_accumulated_data: bool = True
) -> Tuple[List[RegimeRoutingResult], Dict[str, Any]]:
    """레짐 라우팅 반복 제어 실행 (성능 개선이 멈추면 종료) 🔥
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        strategies: 전략 리스트
        candle_data: 캔들 데이터
        max_iterations: 최대 반복 횟수
        quality_threshold: 품질 임계값 (이상이면 조기 종료)
        improvement_threshold: 개선 임계값 (이하이면 종료)
        min_iterations: 최소 반복 횟수
        use_accumulated_data: 누적 데이터 활용 여부
    
    Returns:
        (최종 라우팅 결과, 반복 정보)
    """
    try:
        logger.info(f"🔄 {coin}-{interval} 레짐 라우팅 반복 제어 시작 (최대 {max_iterations}회)")
        
        router = RegimeRouter()
        best_results = []
        best_quality = 0.0
        iteration_info = {
            'iterations_performed': 0,
            'quality_history': [],
            'improvement_history': [],
            'early_stopped': False,
            'stop_reason': None
        }
        
        previous_quality = 0.0
        
        for iteration in range(max_iterations):
            try:
                logger.info(f"🔄 레짐 라우팅 반복 {iteration + 1}/{max_iterations}")
                
                # 레짐 라우팅 실행
                current_results = router.route_strategies(
                    coin, interval, strategies, candle_data,
                    use_accumulated_data=use_accumulated_data
                )
                
                if not current_results:
                    logger.warning(f"⚠️ 반복 {iteration + 1}: 라우팅 결과 없음")
                    continue
                
                # 품질 점수 계산
                current_quality = calculate_regime_routing_quality(current_results)
                improvement = current_quality - previous_quality
                
                iteration_info['iterations_performed'] += 1
                iteration_info['quality_history'].append(current_quality)
                iteration_info['improvement_history'].append(improvement)
                
                logger.info(f"📊 반복 {iteration + 1} 품질: {current_quality:.3f} (개선: {improvement:+.3f})")
                
                # 최고 결과 업데이트
                if current_quality > best_quality:
                    best_quality = current_quality
                    best_results = current_results
                    logger.info(f"✅ 최고 품질 업데이트: {best_quality:.3f}")
                
                # 조기 종료 조건 1: 품질 임계값 달성
                if current_quality >= quality_threshold:
                    logger.info(f"🎯 품질 임계값 달성 ({current_quality:.3f} >= {quality_threshold:.3f}) - 조기 종료")
                    iteration_info['early_stopped'] = True
                    iteration_info['stop_reason'] = 'quality_threshold'
                    break
                
                # 조기 종료 조건 2: 개선도 미미 (최소 반복 횟수 충족 시)
                if iteration >= min_iterations - 1 and improvement < improvement_threshold:
                    logger.info(f"🎯 개선도 미미 ({improvement:.3f} < {improvement_threshold:.3f}) - 조기 종료")
                    iteration_info['early_stopped'] = True
                    iteration_info['stop_reason'] = 'improvement_threshold'
                    break
                
                previous_quality = current_quality
                
            except Exception as e:
                logger.error(f"❌ 반복 {iteration + 1} 실패: {e}")
                continue
        
        # 최종 결과
        final_results = best_results if best_results else current_results if 'current_results' in locals() else []
        iteration_info['final_quality'] = best_quality
        
        logger.info(f"✅ 레짐 라우팅 반복 완료: {iteration_info['iterations_performed']}회, "
                   f"최종 품질 {best_quality:.3f} ({iteration_info['stop_reason'] or '최대 반복 도달'})")
        
        return final_results, iteration_info
        
    except Exception as e:
        logger.error(f"❌ 레짐 라우팅 반복 제어 실패: {e}")
        return [], {'error': str(e)}