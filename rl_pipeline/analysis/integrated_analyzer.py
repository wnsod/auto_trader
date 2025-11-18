import logging
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 🔥 디버그 시스템 import (안전한 fallback)
try:
    from rl_pipeline.monitoring import RoutingDebugger, AnalysisDebugger
    DEBUG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ 디버그 로깅 시스템을 사용할 수 없습니다")
    DEBUG_AVAILABLE = False
    RoutingDebugger = None
    AnalysisDebugger = None

# ---------------------------------------------------------------------
# 외부 시스템 의존 모듈 (없을 수 있으므로 안전 가드)
# ---------------------------------------------------------------------
try:
    from rl_pipeline.analysis.learning_engine import (
        JAXPolicyTrainer,
        JAXGPUSimulation,
        JAXPerformanceMonitor,
        get_jax_policy_trainer,
        get_jax_gpu_simulation,
        get_jax_performance_monitor,
    )
    from rl_pipeline.analysis.advanced_learning_systems import (
        JAXEnsembleLearningSystem,
        JAXPPOSystem,
        get_jax_ensemble_system,
        get_jax_ppo_system,
    )
    LEARNING_SYSTEMS_AVAILABLE = True
except Exception as e:
    # 🔥 수정: 미구현 모듈은 debug 레벨로 변경 (선택적 기능)
    logger.debug(f"[통합분석기] 학습 시스템 모듈 미사용 (선택적): {e}")
    LEARNING_SYSTEMS_AVAILABLE = False

# 분석 유틸 (이전 외부 모듈이 제거되었을 수 있어 더미 제공)
def run_fractal_analysis(*_args, **_kwargs) -> Dict[str, float]:
    return {"fractal_score": 0.5, "complexity": 0.5}

def score_synergy(*_args, **_kwargs) -> Dict[str, float]:
    return {"synergy_score": 0.5, "correlation": 0.5}

def analyze_coin_indicator_correlations(*_args, **_kwargs) -> Dict[str, Any]:
    return {"correlation_score": 0.5, "indicators": []}

# ---------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------
@dataclass
class CoinSignalScore:
    """코인별 신호 스코어 산출 결과"""
    coin: str
    interval: str
    regime: str
    fractal_score: float
    multi_timeframe_score: float
    indicator_cross_score: float
    ensemble_score: float
    ensemble_confidence: float
    final_signal_score: float
    signal_action: str
    signal_confidence: float
    created_at: str

@dataclass
class GlobalSignalScore:
    """전역(글로벌) 신호 스코어 결과"""
    overall_score: float
    overall_confidence: float
    policy_improvement: float
    convergence_rate: float
    top_performers: List[str]
    top_coins: List[str]
    top_intervals: List[str]
    created_at: str

# ---------------------------------------------------------------------
# 메인 분석기
# ---------------------------------------------------------------------
class IntegratedAnalyzer:
    """learning_engine + advanced_learning_systems 통합 분석기"""

    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        통합 분석기 초기화

        Args:
            session_id: 디버그 세션 ID (옵션)
        """
        self.learning_systems_available = False
        if LEARNING_SYSTEMS_AVAILABLE:
            try:
                self.ensemble_system = get_jax_ensemble_system()
                self.policy_trainer = get_jax_policy_trainer()
                self.gpu_simulation = get_jax_gpu_simulation()
                self.performance_monitor = get_jax_performance_monitor()
                self.learning_systems_available = True
            except Exception as e:
                # 🔥 수정: debug 레벨로 변경
                logger.debug(f"[통합분석기] 학습 시스템 초기화 건너뜀 (선택적): {e}")
                self.learning_systems_available = False

        # 🔥 디버거 초기화
        self.routing_debug = None
        self.analysis_debug = None
        if DEBUG_AVAILABLE and session_id:
            try:
                self.routing_debug = RoutingDebugger(session_id=session_id)
                self.analysis_debug = AnalysisDebugger(session_id=session_id)
                logger.debug(f"✅ Analysis/Routing 디버거 초기화 완료 (session: {session_id})")
            except Exception as e:
                logger.warning(f"⚠️ Analysis/Routing 디버거 초기화 실패: {e}")

        logger.info("✅ 통합 분석기 초기화 완료")

    def _normalize_interval(self, interval: Any) -> str:
        """인터벌을 문자열로 정규화

        Args:
            interval: 인터벌 (str 또는 tuple)

        Returns:
            str: 정규화된 인터벌 문자열
        """
        if isinstance(interval, tuple):
            # ('SOL', '15m') -> '15m'
            return interval[1] if len(interval) > 1 else str(interval[0])
        return str(interval)

    # ------------------------------
    # 코인별 전략 분석 (단일 인터벌)
    # ------------------------------
    def analyze_coin_strategies(
        self,
        coin: str,
        interval: str,
        regime: str,
        strategies: List[Dict[str, Any]],
        candle_data: pd.DataFrame,
    ) -> CoinSignalScore:
        """코인별 전략/지표/레짐을 종합하여 스코어/액션을 산출"""
        try:
            logger.info(f"[{coin}-{interval}] 전략 분석 시작 (전략 {len(strategies)}개)")

            # 저장된 최적 비율 불러오기
            stored_ratios: Dict[str, Any] = {}
            try:
                from rl_pipeline.db.reads import get_coin_analysis_ratios  # type: ignore
                stored_ratios = get_coin_analysis_ratios(coin, interval, regime) or {}
            except Exception as e:
                logger.debug(f"[{coin}] 저장된 분석 비율 조회 실패(무시): {e}")

            if stored_ratios.get("updated_at"):
                logger.info(f"[{coin}] '{regime}' 저장된 최적 분석 비율 사용: {stored_ratios['updated_at']}")
                analysis_modules = stored_ratios.get("optimal_modules", {})
                fractal_ratios = stored_ratios.get("fractal_ratios", {})
                multi_timeframe_ratios = stored_ratios.get("multi_timeframe_ratios", {})
                indicator_cross_ratios = stored_ratios.get("indicator_cross_ratios", {})
            else:
                logger.info(f"[{coin}] '{regime}' 최적 분석 비율 계산")
                analysis_modules = self._select_optimal_analysis_modules(coin, interval, regime, candle_data)
                fractal_ratios = self._get_coin_optimal_fractal_intervals(coin, regime)
                multi_timeframe_ratios = self._get_coin_optimal_multi_timeframe_ratios(coin, regime)
                indicator_cross_ratios = self._get_coin_optimal_indicator_cross_ratios(coin, regime)
                # 저장 시도
                self._save_coin_analysis_ratios(
                    coin, interval, regime, analysis_modules,
                    fractal_ratios, multi_timeframe_ratios, indicator_cross_ratios
                )

            analysis_results: Dict[str, float] = {}

            # 1) 프랙탈
            if "fractal" in analysis_modules:
                analysis_results["fractal"] = self._analyze_fractal_patterns_with_ratios(
                    coin, interval, candle_data, fractal_ratios
                )
            else:
                analysis_results["fractal"] = 0.5

            # 2) 다중시간대
            if "multi_timeframe" in analysis_modules:
                analysis_results["multi_timeframe"] = self._analyze_multi_timeframe_with_ratios(
                    coin, interval, candle_data, multi_timeframe_ratios
                )
            else:
                analysis_results["multi_timeframe"] = 0.5

            # 3) 지표 교차/상관
            if "indicator_cross" in analysis_modules:
                analysis_results["indicator_cross"] = self._analyze_indicator_correlations_with_ratios(
                    coin, interval, candle_data, indicator_cross_ratios
                )
            else:
                analysis_results["indicator_cross"] = 0.5

            # 4) 코인 특화
            if "coin_specific" in analysis_modules:
                analysis_results["coin_specific"] = self._analyze_coin_specific_patterns(coin, interval, candle_data)
            else:
                analysis_results["coin_specific"] = 0.5

            # 5) 변동성 특화
            if "volatility" in analysis_modules:
                analysis_results["volatility"] = self._analyze_volatility_patterns(coin, interval, candle_data)
            else:
                analysis_results["volatility"] = 0.5

            # 6) 거래량 특화
            if "volume" in analysis_modules:
                analysis_results["volume"] = self._analyze_volume_patterns(coin, interval, candle_data)
            else:
                analysis_results["volume"] = 0.5

            # 7) 앙상블 예측
            if self.learning_systems_available:
                try:
                    ensemble_result = self.ensemble_system.predict_ensemble(
                        {
                            "coin": coin,
                            "interval": interval,
                            "regime": regime,
                            "strategies": strategies,
                            "analysis_results": analysis_results,
                            "selected_modules": list(analysis_modules.keys()),
                            "close": candle_data["close"].tolist() if "close" in candle_data.columns else [],
                            "volume": candle_data["volume"].tolist() if "volume" in candle_data.columns else [],
                        }
                    )
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.warning(f"[{coin}] 앙상블 예측 실패: {e}")
                    logger.debug(f"앙상블 예측 상세 에러:\n{error_details}")
                    ensemble_result = self._create_default_ensemble_result()
                    logger.debug(f"[{coin}] 기본 앙상블 결과 사용: 예측={ensemble_result.ensemble_prediction:.3f}, 신뢰도={ensemble_result.confidence_score:.3f}")
            else:
                ensemble_result = self._create_default_ensemble_result()

            # 8) GPU 시뮬레이션 (상위 5개 전략)
            simulation_results: List[Dict[str, Any]] = []
            if self.learning_systems_available:
                try:
                    for strategy in strategies[:5]:
                        sim = self.gpu_simulation.simulate_strategy_with_jax(
                            strategy=strategy,
                            market_data=self._get_market_data(coin, interval, candle_data),
                            strategy_id=strategy.get("strategy_id", "unknown"),
                        )
                        simulation_results.append(sim)
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.warning(f"[{coin}] 시뮬레이션 실패(무시): {e}")
                    logger.debug(f"시뮬레이션 상세 에러:\n{error_details}")
                    simulation_results = []
                    logger.debug(f"[{coin}] 시뮬레이션 결과 없음으로 진행 (상위 {len(strategies)}개 전략 중 {min(5, len(strategies))}개 시도)")

            # 9) 레짐 가중 최종 스코어
            if candle_data is None or candle_data.empty:
                candle_data = pd.DataFrame(
                    {
                        "close": [100.0],
                        "volume": [1_000_000.0],
                        "regime_confidence": [0.5],
                        "regime_transition_prob": [0.5],
                    }
                )

            final_signal_score = self._calculate_final_signal_score_with_regime(
                analysis_results,
                analysis_modules,
                ensemble_result,
                simulation_results,
                coin,
                interval,
                regime,
                candle_data,
            )
            
            # 🆕 등급별 전략 신뢰도 기반 점수 조정 (안정성 우선)
            confidence = self._calculate_strategy_confidence(strategies, candle_data)
            
            if confidence is not None:
                # 신뢰도가 낮으면 보수적으로 판단
                if confidence > 0.5:
                    # 신뢰도 매우 높음 - 점수 그대로 사용
                    logger.info(f"✅ 높은 신뢰도: {confidence:.2%} - 점수 유지")
                elif confidence > 0.3:
                    # 신뢰도 중간 - 약간 보수적으로 조정
                    final_signal_score = 0.5 + (final_signal_score - 0.5) * 0.7
                    logger.info(f"⚠️ 중간 신뢰도: {confidence:.2%} - 점수 조정 (0.7배)")
                else:
                    # 신뢰도 매우 낮음 - 강제 HOLD
                    final_signal_score = 0.5
                    logger.info(f"🚫 낮은 신뢰도: {confidence:.2%} - 강제 HOLD (안전)")
            
            final_signal_score = max(0.0, min(1.0, final_signal_score))

            # 10) 액션/신뢰도
            signal_action = self._determine_signal_action(final_signal_score, regime, confidence)
            ensemble_conf = (
                ensemble_result["confidence_score"]
                if isinstance(ensemble_result, dict)
                else getattr(ensemble_result, "confidence_score", 0.5)
            )
            ensemble_pred = (
                ensemble_result["ensemble_prediction"]
                if isinstance(ensemble_result, dict)
                else getattr(ensemble_result, "ensemble_prediction", 0.5)
            )
            signal_confidence = self._calculate_signal_confidence(ensemble_conf, simulation_results)

            result = CoinSignalScore(
                coin=coin,
                interval=interval,
                regime=regime,
                fractal_score=analysis_results.get("fractal", 0.5),
                multi_timeframe_score=analysis_results.get("multi_timeframe", 0.5),
                indicator_cross_score=analysis_results.get("indicator_cross", 0.5),
                ensemble_score=float(ensemble_pred),
                ensemble_confidence=float(ensemble_conf),
                final_signal_score=float(final_signal_score),
                signal_action=signal_action,
                signal_confidence=float(signal_confidence),
                created_at=datetime.now().isoformat(),
            )

            logger.info(f"✅ [{coin}-{interval}] 분석 완료 → {signal_action} (점수: {final_signal_score:.3f})")

            # 🔥 디버거 로깅
            if self.analysis_debug:
                try:
                    # 분석 결과 로깅
                    self.analysis_debug.log_interval_strategy_score(
                        coin=coin,
                        interval=interval,
                        strategy_score=final_signal_score,
                        num_strategies=len(strategies),
                        regime=regime
                    )

                    # 각 분석 모듈 결과 로깅
                    if "fractal" in analysis_results:
                        self.analysis_debug.log_fractal_analysis(
                            coin=coin,
                            interval=interval,
                            fractal_score=analysis_results["fractal"],
                            fractal_ratios={},
                            trend_strength=0.0
                        )

                    if "multi_timeframe" in analysis_results:
                        self.analysis_debug.log_multi_timeframe_analysis(
                            coin=coin,
                            interval=interval,
                            multi_tf_score=analysis_results["multi_timeframe"],
                            timeframe_ratios={},
                            alignment_score=0.0
                        )

                    if "indicator_cross" in analysis_results:
                        self.analysis_debug.log_indicator_cross_analysis(
                            coin=coin,
                            interval=interval,
                            indicator_score=analysis_results["indicator_cross"],
                            indicator_ratios={},
                            num_crosses=0
                        )

                    # 신뢰도 로깅
                    self.analysis_debug.log_interval_confidence(
                        coin=coin,
                        interval=interval,
                        strategy_score=final_signal_score,
                        fractal_score=analysis_results.get("fractal", 0.5),
                        multi_tf_score=analysis_results.get("multi_timeframe", 0.5),
                        indicator_score=analysis_results.get("indicator_cross", 0.5),
                        ensemble_score=float(ensemble_pred),
                        ensemble_confidence=float(ensemble_conf),
                        signal_confidence=float(signal_confidence),
                        signal_action=signal_action
                    )
                except Exception as debug_err:
                    logger.debug(f"디버거 로깅 실패 (무시): {debug_err}")

            return result

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ [{coin}-{interval}] 코인 분석 실패: {e}")
            logger.debug(f"상세 에러 정보:\n{error_details}")
            # 기본값 반환하되 에러 정보 포함
            default_result = self._create_default_coin_signal_score(coin, interval, regime)
            logger.warning(f"⚠️ [{coin}-{interval}] 기본값 반환: {default_result.signal_action} (점수: {default_result.final_signal_score:.3f})")
            return default_result

    # ------------------------------
    # 다중 인터벌 전략 분석 (개선된 프랙탈/멀티타임프레임/지표 교차 분석)
    # ------------------------------
    def analyze_multi_interval_strategies(
        self,
        coin: str,
        regime: str,
        strategies: List[Dict[str, Any]],
        multi_interval_candle_data: Dict[str, pd.DataFrame],
    ) -> CoinSignalScore:
        """🔥 여러 인터벌의 전략 점수를 종합하여 최종 시그널 점수 계산
        
        Args:
            coin: 코인 심볼
            regime: 현재 레짐
            strategies: 전략 리스트
            multi_interval_candle_data: 인터벌별 캔들 데이터 {interval: DataFrame}
        
        Returns:
            CoinSignalScore: 최종 시그널 점수 및 액션
        """
        try:
            logger.info(f"🔥 [{coin}] 다중 인터벌 통합 분석 시작: {list(multi_interval_candle_data.keys())} (전략 {len(strategies)}개)")
            
            # 실제 전달받은 인터벌 데이터 사용 (하드코딩 제거)
            available_intervals = [iv for iv in multi_interval_candle_data.keys() 
                                  if iv in multi_interval_candle_data and not multi_interval_candle_data[iv].empty]
            
            if not available_intervals:
                logger.warning(f"⚠️ [{coin}] 사용 가능한 인터벌 데이터가 없습니다")
                # 단일 인터벌 분석으로 폴백
                if strategies:
                    first_interval = list(multi_interval_candle_data.keys())[0] if multi_interval_candle_data else '15m'
                    first_candle = multi_interval_candle_data.get(first_interval, pd.DataFrame())
                    return self.analyze_coin_strategies(coin, first_interval, regime, strategies, first_candle)
                else:
                    return self._create_default_coin_signal_score(coin, '15m', regime)
            
            logger.info(f"📊 [{coin}] 분석 대상 인터벌: {available_intervals}")
            
            # 0-1단계: 🔥 각 인터벌별 레짐 감지 (개선 1단계)
            interval_regimes: Dict[str, Tuple[str, float]] = {}
            for interval in available_intervals:
                candle_data = multi_interval_candle_data[interval]
                interval_str = self._normalize_interval(interval)
                try:
                    from rl_pipeline.routing.regime_router import RegimeRouter
                    router = RegimeRouter()
                    detected_regime, regime_confidence, regime_transition_prob = router.detect_current_regime(coin, interval_str, candle_data)
                    interval_regimes[interval] = (detected_regime, regime_confidence)
                    logger.info(f"  📊 [{coin}-{interval_str}] 레짐 감지: {detected_regime} (신뢰도: {regime_confidence:.3f})")
                except Exception as e:
                    logger.warning(f"⚠️ [{coin}-{interval_str}] 레짐 감지 실패: {e}")
                    interval_regimes[interval] = (regime, 0.5)  # 기본값으로 입력 레짐 사용
            
            # 0-2단계: 레짐 일치도 계산 및 메인 레짐 결정
            regime_alignment, main_regime = self._calculate_regime_alignment(interval_regimes)
            logger.info(f"📊 [{coin}] 레짐 일치도: {regime_alignment:.3f}, 메인 레짐: {main_regime}")
            
            # 0-3단계: 🔥 고등급 전략 맥락 분석 (새로운 접근)
            high_grade_strategies = [s for s in strategies if s.get('grade') in ['S', 'A'] or s.get('quality_grade') in ['S', 'A']]
            if high_grade_strategies:
                logger.info(f"🔥 [{coin}] 고등급 전략 맥락 분석 시작: {len(high_grade_strategies)}개")
                context_analysis = self._analyze_strategy_context(
                    coin, high_grade_strategies, multi_interval_candle_data
                )
                logger.info(f"📊 [{coin}] 맥락 분석 완료: 다른 인터벌 성과 및 지표 상태 파악")
            else:
                context_analysis = {}
            
            # 1단계: 각 인터벌별 전략 점수 및 신뢰도 점수 계산
            interval_results: Dict[str, Dict[str, float]] = {}

            for interval in available_intervals:
                candle_data = multi_interval_candle_data[interval]
                interval_str = self._normalize_interval(interval)

                try:
                    # 1-1) 🔥 인터벌별 전략 점수 계산 (매수/매도 그룹 분리)
                    # 매수 그룹과 매도 그룹의 점수를 각각 계산
                    buy_strategy_score = self._calculate_interval_strategy_score_by_direction(
                        strategies, candle_data, 'buy'
                    )
                    sell_strategy_score = self._calculate_interval_strategy_score_by_direction(
                        strategies, candle_data, 'sell'
                    )
                    
                    # 🔥 최종 전략 점수: 매수 그룹 점수와 매도 그룹 점수를 조합
                    # 매수 점수는 0.5 이상일 때 상승 신호, 매도 점수는 0.5 이하일 때 하락 신호
                    # 매수 점수가 높으면 상승 신호 강화, 매도 점수가 낮으면 하락 신호 강화
                    strategy_score = 0.5 + (buy_strategy_score - 0.5) * 0.6 - (sell_strategy_score - 0.5) * 0.4
                    strategy_score = max(0.0, min(1.0, strategy_score))
                    
                    logger.debug(f"🔥 {interval_str} 전략 점수 조합: 매수={buy_strategy_score:.3f}, 매도={sell_strategy_score:.3f}, 최종={strategy_score:.3f}")

                    # 1-2) 프랙탈 분석 (인터벌 신뢰도)
                    fractal_ratios = self._get_coin_optimal_fractal_intervals(coin, regime)
                    fractal_score = self._analyze_fractal_patterns_with_ratios(coin, interval_str, candle_data, fractal_ratios)

                    # 1-3) 멀티타임프레임 분석 (인터벌 신뢰도) - 방향성 해석 추가
                    multi_timeframe_ratios = self._get_coin_optimal_multi_timeframe_ratios(coin, regime)
                    base_multi_timeframe_score = self._analyze_multi_timeframe_with_ratios(coin, interval_str, candle_data, multi_timeframe_ratios)
                    
                    # 🔥 방향성 해석: 매수 신호용 상승 추세, 매도 신호용 하락 추세
                    # 추세 방향 계산
                    st_trend = self._calculate_short_term_trend(candle_data)
                    mt_trend = self._calculate_medium_term_trend(candle_data)
                    lt_trend = self._calculate_long_term_trend(candle_data)
                    avg_trend = (st_trend + mt_trend + lt_trend) / 3.0
                    
                    # 매수 그룹 점수가 높으면 상승 추세일 때 신뢰도 증가
                    if buy_strategy_score > 0.5:
                        # 상승 추세 (avg_trend > 0)일 때 신뢰도 증가
                        trend_adjustment = max(0.0, avg_trend) * 0.2  # 최대 20% 증가
                        buy_multi_timeframe_score = min(1.0, base_multi_timeframe_score + trend_adjustment)
                    else:
                        buy_multi_timeframe_score = base_multi_timeframe_score
                    
                    # 매도 그룹 점수가 높으면 하락 추세일 때 신뢰도 증가
                    if sell_strategy_score > 0.5:
                        # 하락 추세 (avg_trend < 0)일 때 신뢰도 증가
                        trend_adjustment = max(0.0, -avg_trend) * 0.2  # 최대 20% 증가
                        sell_multi_timeframe_score = min(1.0, base_multi_timeframe_score + trend_adjustment)
                    else:
                        sell_multi_timeframe_score = base_multi_timeframe_score
                    
                    # 🔥 최종 멀티타임프레임 점수: 매수/매도 그룹 점수에 따라 가중 평균
                    if buy_strategy_score > 0.5 and sell_strategy_score < 0.5:
                        multi_timeframe_score = buy_multi_timeframe_score * 0.7 + sell_multi_timeframe_score * 0.3
                    elif sell_strategy_score > 0.5 and buy_strategy_score < 0.5:
                        multi_timeframe_score = buy_multi_timeframe_score * 0.3 + sell_multi_timeframe_score * 0.7
                    else:
                        multi_timeframe_score = (buy_multi_timeframe_score + sell_multi_timeframe_score) / 2.0

                    # 1-4) 지표 교차 분석 (인터벌 신뢰도) - 방향성 해석 추가
                    indicator_cross_ratios = self._get_coin_optimal_indicator_cross_ratios(coin, regime)
                    base_indicator_cross_score = self._analyze_indicator_correlations_with_ratios(coin, interval_str, candle_data, indicator_cross_ratios)
                    
                    # 🔥 방향성 해석: 매수/매도 신호별로 지표 해석
                    # RSI와 MACD 방향성 확인
                    rsi_buy_signal = False
                    rsi_sell_signal = False
                    macd_buy_signal = False
                    macd_sell_signal = False
                    
                    if not candle_data.empty and len(candle_data) > 0:
                        # 최신 RSI 확인
                        if 'rsi' in candle_data.columns:
                            latest_rsi = candle_data['rsi'].iloc[-1] if not candle_data['rsi'].isna().iloc[-1] else 50.0
                            rsi_buy_signal = latest_rsi < 40  # 과매도 영역
                            rsi_sell_signal = latest_rsi > 60  # 과매수 영역
                        
                        # 최신 MACD 확인
                        if 'macd' in candle_data.columns and 'macd_signal' in candle_data.columns:
                            latest_macd = candle_data['macd'].iloc[-1] if not candle_data['macd'].isna().iloc[-1] else 0.0
                            latest_signal = candle_data['macd_signal'].iloc[-1] if not candle_data['macd_signal'].isna().iloc[-1] else 0.0
                            macd_buy_signal = latest_macd > latest_signal and latest_macd > 0  # MACD 상승 크로스
                            macd_sell_signal = latest_macd < latest_signal and latest_macd < 0  # MACD 하락 크로스
                    
                    # 매수 그룹 점수가 높으면 매수 신호 지표일 때 신뢰도 증가
                    buy_indicator_signals = sum([rsi_buy_signal, macd_buy_signal])
                    if buy_strategy_score > 0.5 and buy_indicator_signals > 0:
                        indicator_adjustment = (buy_indicator_signals / 2.0) * 0.15  # 최대 15% 증가
                        buy_indicator_cross_score = min(1.0, base_indicator_cross_score + indicator_adjustment)
                    else:
                        buy_indicator_cross_score = base_indicator_cross_score
                    
                    # 매도 그룹 점수가 높으면 매도 신호 지표일 때 신뢰도 증가
                    sell_indicator_signals = sum([rsi_sell_signal, macd_sell_signal])
                    if sell_strategy_score > 0.5 and sell_indicator_signals > 0:
                        indicator_adjustment = (sell_indicator_signals / 2.0) * 0.15  # 최대 15% 증가
                        sell_indicator_cross_score = min(1.0, base_indicator_cross_score + indicator_adjustment)
                    else:
                        sell_indicator_cross_score = base_indicator_cross_score
                    
                    # 🔥 최종 지표 교차 점수: 매수/매도 그룹 점수에 따라 가중 평균
                    if buy_strategy_score > 0.5 and sell_strategy_score < 0.5:
                        indicator_cross_score = buy_indicator_cross_score * 0.7 + sell_indicator_cross_score * 0.3
                    elif sell_strategy_score > 0.5 and buy_strategy_score < 0.5:
                        indicator_cross_score = buy_indicator_cross_score * 0.3 + sell_indicator_cross_score * 0.7
                    else:
                        indicator_cross_score = (buy_indicator_cross_score + sell_indicator_cross_score) / 2.0

                    # 1-5) 🔥 전략별 맥락 분석 결과 기반 신뢰도 계산 (매수/매도 그룹 분리)
                    # 매수 그룹 맥락 신뢰도
                    buy_strategies_list = [s for s in strategies if self._classify_strategy_direction(s) == 'buy']
                    buy_context_confidence = self._calculate_context_based_confidence(
                        interval_str, context_analysis, buy_strategies_list
                    ) if buy_strategies_list else 0.5
                    
                    # 매도 그룹 맥락 신뢰도
                    sell_strategies_list = [s for s in strategies if self._classify_strategy_direction(s) == 'sell']
                    sell_context_confidence = self._calculate_context_based_confidence(
                        interval_str, context_analysis, sell_strategies_list
                    ) if sell_strategies_list else 0.5
                    
                    # 🔥 최종 맥락 신뢰도: 매수/매도 그룹 신뢰도의 가중 평균
                    # 매수 점수가 높으면 매수 맥락 신뢰도에 더 높은 가중치
                    if buy_strategy_score > 0.5 and sell_strategy_score < 0.5:
                        context_confidence = buy_context_confidence * 0.7 + sell_context_confidence * 0.3
                    elif sell_strategy_score > 0.5 and buy_strategy_score < 0.5:
                        context_confidence = buy_context_confidence * 0.3 + sell_context_confidence * 0.7
                    else:
                        context_confidence = (buy_context_confidence + sell_context_confidence) / 2.0

                    # 1-6) 🔥 레짐 일치도 기반 신뢰도 조정 (개선 1단계)
                    interval_regime, regime_conf = interval_regimes.get(interval, (regime, 0.5))
                    regime_consistency = self._calculate_regime_consistency_penalty(
                        interval_regime, main_regime, regime_alignment
                    )

                    # 1-7) 🔥 레짐별 동적 가중치 계산 (개선 3단계)
                    dynamic_weights = self._calculate_dynamic_analysis_weights(
                        interval_regime, coin, interval_str
                    )
                    
                    # 1-8) 인터벌 종합 신뢰도 (동적 가중치 적용)
                    base_interval_confidence = (
                        fractal_score * dynamic_weights['fractal'] +
                        multi_timeframe_score * dynamic_weights['multi_timeframe'] +
                        indicator_cross_score * dynamic_weights['indicator_cross'] +
                        context_confidence * dynamic_weights['context']
                    )
                    # 레짐 불일치 시 신뢰도 조정 (0.8 ~ 1.0 배율)
                    interval_confidence = base_interval_confidence * regime_consistency
                    
                    interval_results[interval] = {
                        'strategy_score': strategy_score,
                        'buy_strategy_score': buy_strategy_score,  # 🔥 매수 그룹 점수
                        'sell_strategy_score': sell_strategy_score,  # 🔥 매도 그룹 점수
                        'fractal_score': fractal_score,
                        'multi_timeframe_score': multi_timeframe_score,
                        'buy_multi_timeframe_score': buy_multi_timeframe_score,  # 🔥 매수 그룹 멀티타임프레임 점수
                        'sell_multi_timeframe_score': sell_multi_timeframe_score,  # 🔥 매도 그룹 멀티타임프레임 점수
                        'indicator_cross_score': indicator_cross_score,
                        'buy_indicator_cross_score': buy_indicator_cross_score,  # 🔥 매수 그룹 지표 교차 점수
                        'sell_indicator_cross_score': sell_indicator_cross_score,  # 🔥 매도 그룹 지표 교차 점수
                        'context_confidence': context_confidence,  # 맥락 분석 신뢰도
                        'buy_context_confidence': buy_context_confidence,  # 🔥 매수 그룹 맥락 신뢰도
                        'sell_context_confidence': sell_context_confidence,  # 🔥 매도 그룹 맥락 신뢰도
                        'regime': interval_regime,  # 🔥 인터벌별 레짐
                        'regime_confidence': regime_conf,  # 🔥 레짐 신뢰도
                        'regime_consistency': regime_consistency,  # 🔥 레짐 일치도 배율
                        'dynamic_weights': dynamic_weights,  # 🔥 동적 가중치 (개선 3단계)
                        'interval_confidence': interval_confidence,
                    }
                    
                    logger.info(f"  📊 {interval_str}: 전략점수={strategy_score:.3f}, 신뢰도={interval_confidence:.3f} "
                              f"(프랙탈={fractal_score:.3f}×{dynamic_weights['fractal']:.2f}, 멀티TF={multi_timeframe_score:.3f}×{dynamic_weights['multi_timeframe']:.2f}, "
                              f"지표교차={indicator_cross_score:.3f}×{dynamic_weights['indicator_cross']:.2f}, 맥락={context_confidence:.3f}×{dynamic_weights['context']:.2f}, 레짐일치={regime_consistency:.3f})")

                except Exception as e:
                    logger.warning(f"⚠️ [{coin}-{interval_str}] 인터벌 분석 실패: {e}")
                    interval_results[interval] = {
                        'strategy_score': 0.5,
                        'buy_strategy_score': 0.5,  # 🔥 매수 그룹 점수
                        'sell_strategy_score': 0.5,  # 🔥 매도 그룹 점수
                        'fractal_score': 0.5,
                        'multi_timeframe_score': 0.5,
                        'buy_multi_timeframe_score': 0.5,  # 🔥 매수 그룹 멀티타임프레임 점수
                        'sell_multi_timeframe_score': 0.5,  # 🔥 매도 그룹 멀티타임프레임 점수
                        'indicator_cross_score': 0.5,
                        'buy_indicator_cross_score': 0.5,  # 🔥 매수 그룹 지표 교차 점수
                        'sell_indicator_cross_score': 0.5,  # 🔥 매도 그룹 지표 교차 점수
                        'context_confidence': 0.5,
                        'buy_context_confidence': 0.5,  # 🔥 매수 그룹 맥락 신뢰도
                        'sell_context_confidence': 0.5,  # 🔥 매도 그룹 맥락 신뢰도
                        'regime': regime,
                        'regime_confidence': 0.5,
                        'regime_consistency': 0.8,
                        'dynamic_weights': {'fractal': 0.25, 'multi_timeframe': 0.25, 'indicator_cross': 0.25, 'context': 0.25},
                        'interval_confidence': 0.5,
                    }
            
            # 2단계: 인터벌별 가중치 계산 (신뢰도 기반)
            interval_weights: Dict[str, float] = {}
            total_confidence = sum(result['interval_confidence'] for result in interval_results.values())
            
            if total_confidence > 0:
                for interval, result in interval_results.items():
                    # 신뢰도 기반 가중치 (정규화)
                    weight = result['interval_confidence'] / total_confidence
                    interval_weights[interval] = weight
            else:
                # 신뢰도가 모두 0이면 균등 가중치
                for interval in available_intervals:
                    interval_weights[interval] = 1.0 / len(available_intervals)
            
            # 🔥 소숫점 정리 (3자리) - numpy 타입을 float로 변환
            formatted_weights = {k: float(round(v, 3)) for k, v in interval_weights.items()}
            logger.info(f"📊 [{coin}] 인터벌별 가중치: {formatted_weights}")

            # 🔥 인터벌 가중치 DB에 저장 (코인별 최적 가중치)
            try:
                self._save_interval_weights(coin, main_regime, interval_weights)
            except Exception as e:
                logger.debug(f"⚠️ [{coin}] 인터벌 가중치 저장 실패(무시): {e}")

            # 3단계: 최종 시그널 점수 계산 (가중 평균)
            final_signal_score = 0.0
            for interval in available_intervals:
                strategy_score = interval_results[interval]['strategy_score']
                weight = interval_weights[interval]
                final_signal_score += strategy_score * weight
                logger.debug(f"  {interval}: {strategy_score:.3f} × {weight:.3f} = {strategy_score * weight:.3f}")
            
            final_signal_score = max(0.0, min(1.0, final_signal_score))
            logger.info(f"🔥 [{coin}] 최종 시그널 점수: {final_signal_score:.3f} (인터벌 가중 평균)")
            
            # 4단계: 액션 결정 (메인 레짐 사용)
            # 전체 인터벌의 평균 신뢰도 먼저 계산
            avg_confidence = sum(r['interval_confidence'] for r in interval_results.values()) / len(interval_results) if interval_results else 0.5
            signal_action = self._determine_signal_action(final_signal_score, main_regime, avg_confidence)
            
            # 5단계: 종합 신뢰도 계산
            avg_fractal = sum(r['fractal_score'] for r in interval_results.values()) / len(interval_results)
            avg_multi_timeframe = sum(r['multi_timeframe_score'] for r in interval_results.values()) / len(interval_results)
            avg_indicator_cross = sum(r['indicator_cross_score'] for r in interval_results.values()) / len(interval_results)
            
            # 전체 인터벌의 평균 신뢰도
            avg_confidence = sum(r['interval_confidence'] for r in interval_results.values()) / len(interval_results)
            signal_confidence = avg_confidence
            
            # 결과 생성
            result = CoinSignalScore(
                coin=coin,
                interval='all_intervals',  # 🔥 다중 인터벌 통합 분석은 'all_intervals'로 저장 (Paper Trading 조회 호환)
                regime=main_regime,  # 🔥 메인 레짐 사용
                fractal_score=avg_fractal,
                multi_timeframe_score=avg_multi_timeframe,
                indicator_cross_score=avg_indicator_cross,
                ensemble_score=final_signal_score,  # 최종 점수를 앙상블 점수로 사용
                ensemble_confidence=avg_confidence,
                final_signal_score=final_signal_score,
                signal_action=signal_action,
                signal_confidence=signal_confidence,
                created_at=datetime.now().isoformat(),
            )
            
            logger.info(f"✅ [{coin}] 다중 인터벌 통합 분석 완료 → {signal_action} (점수: {final_signal_score:.3f}, 신뢰도: {signal_confidence:.3f})")
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ [{coin}] 다중 인터벌 분석 실패: {e}")
            logger.debug(f"상세 에러 정보:\n{error_details}")
            return self._create_default_coin_signal_score(coin, '15m', regime)
    
    def _classify_strategy_direction(self, strategy: Dict[str, Any]) -> str:
        """🔥 전략을 매수/매도 그룹으로 분류 (strategy_type 우선 버전)
    
        Args:
            strategy: 전략 딕셔너리
    
        Returns:
            'buy', 'sell', 또는 'neutral'
        """
        try:
            # ⭐ 1. strategy_type 우선 확인 (가장 정확한 정보)
            strategy_type = strategy.get('strategy_type', '').lower()
    
            if strategy_type:
                # oversold = 과매도 = 매수 기회
                if 'oversold' in strategy_type or strategy_type == 'buy':
                    return 'buy'
    
                # overbought = 과매수 = 매도 기회
                elif 'overbought' in strategy_type or strategy_type == 'sell':
                    return 'sell'
    
                # mean_reversion = 평균 회귀 -> RSI 기반 판단
                elif 'mean_reversion' in strategy_type or 'reversion' in strategy_type:
                    rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
                    # 평균 회귀는 극단에서 반대 방향
                    if rsi_midpoint < 40:
                        return 'buy'  # 낮은 RSI에서 반등 기대
                    elif rsi_midpoint > 60:
                        return 'sell'  # 높은 RSI에서 하락 기대
                    else:
                        return 'neutral'
    
                # trend_following = 추세 추종 -> MACD/ADX 기반 판단
                elif 'trend' in strategy_type:
                    macd_buy = strategy.get('macd_buy_threshold', 0.0)
                    macd_sell = strategy.get('macd_sell_threshold', 0.0)
    
                    # MACD 차이로 추세 방향 판단
                    if macd_buy > macd_sell + 0.01:
                        return 'buy'  # 상승 추세 추종
                    elif macd_sell < macd_buy - 0.01:
                        return 'sell'  # 하락 추세 추종
                    else:
                        # RSI로 2차 판단
                        rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
                        if rsi_midpoint < 48:
                            return 'buy'
                        elif rsi_midpoint > 52:
                            return 'sell'
                        else:
                            return 'neutral'
    
                # hybrid나 기타 타입은 다음 단계로
                # (여기서는 패스)
    
            # 2. 전략 ID/이름 기반 분류 (strategy_type 없을 때)
            buy_score = 0.0
            sell_score = 0.0
    
            strategy_id = strategy.get('id', '')
            if 'oversold' in strategy_id.lower():
                buy_score += 0.8
            elif 'overbought' in strategy_id.lower():
                sell_score += 0.8
            elif 'buy' in strategy_id.lower():
                buy_score += 0.5
            elif 'sell' in strategy_id.lower():
                sell_score += 0.5
    
            # 3. 명시적 방향성 특화 전략 확인
            pattern_source = strategy.get('pattern_source', '')
            if pattern_source == 'direction_specialized':
                direction = strategy.get('direction', '')
                if direction == 'BUY':
                    buy_score += 1.0
                elif direction == 'SELL':
                    sell_score += 1.0
    
            # 4. RSI 기반 분류 (중앙값과 범위 활용)
            rsi_min = strategy.get('rsi_min', 30.0)
            rsi_max = strategy.get('rsi_max', 70.0)
            rsi_midpoint = (rsi_min + rsi_max) / 2.0
            rsi_range = rsi_max - rsi_min
    
            if rsi_midpoint < 50:
                buy_score += (50 - rsi_midpoint) / 50.0
            elif rsi_midpoint > 50:
                sell_score += (rsi_midpoint - 50) / 50.0
    
            # RSI 범위 특화
            if rsi_range < 30:
                specialization_bonus = (30 - rsi_range) / 30.0 * 0.3
                if rsi_midpoint < 50:
                    buy_score += specialization_bonus
                else:
                    sell_score += specialization_bonus
    
            # 극단적 RSI
            if rsi_min < 30:
                buy_score += (30 - rsi_min) / 30.0 * 0.5
            if rsi_max > 70:
                sell_score += (rsi_max - 70) / 30.0 * 0.5
    
            # 5. MACD 기준
            macd_buy_threshold = strategy.get('macd_buy_threshold', 0.0)
            macd_sell_threshold = strategy.get('macd_sell_threshold', 0.0)
    
            if macd_buy_threshold > 0:
                buy_score += min(macd_buy_threshold * 10, 0.5)
            if macd_sell_threshold < 0:
                sell_score += min(abs(macd_sell_threshold) * 10, 0.5)
    
            macd_diff = macd_buy_threshold - macd_sell_threshold
            if macd_diff > 0.02:
                buy_score += 0.2
            elif macd_diff < -0.02:
                sell_score += 0.2
    
            # 6. 볼륨 기준
            volume_ratio_min = strategy.get('volume_ratio_min', 1.0)
            if volume_ratio_min > 1.5:
                if rsi_midpoint < 50:
                    buy_score += (volume_ratio_min - 1.0) * 0.2
                else:
                    sell_score += (volume_ratio_min - 1.0) * 0.2
    
            # 7. MFI
            mfi_min = strategy.get('mfi_min', 20.0)
            mfi_max = strategy.get('mfi_max', 80.0)
            mfi_midpoint = (mfi_min + mfi_max) / 2.0
    
            if mfi_midpoint < 50:
                buy_score += (50 - mfi_midpoint) / 100.0
            elif mfi_midpoint > 50:
                sell_score += (mfi_midpoint - 50) / 100.0
    
            # 8. 최종 분류 (임계값 0.05)
            score_diff = abs(buy_score - sell_score)
    
            if buy_score > sell_score and score_diff > 0.05:
                return 'buy'
            elif sell_score > buy_score and score_diff > 0.05:
                return 'sell'
            else:
                # RSI 중앙값으로 최종 결정
                if rsi_midpoint < 48:
                    return 'buy'
                elif rsi_midpoint > 52:
                    return 'sell'
                else:
                    return 'neutral'
    
        except Exception as e:
            logger.debug(f"전략 방향 분류 실패 (무시): {e}")
            # 에러 시 기본 분류
            try:
                rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
                if rsi_midpoint < 48:
                    return 'buy'
                elif rsi_midpoint > 52:
                    return 'sell'
            except:
                pass
            return 'neutral'
    

    def _calculate_interval_strategy_score(
        self,
        strategies: List[Dict[str, Any]],
        candle_data: pd.DataFrame,
    ) -> float:
        """인터벌별 전략 점수 계산 (기존 방식 - 하위 호환성 유지)
        
        Args:
            strategies: 전략 리스트
            candle_data: 캔들 데이터
        
        Returns:
            float: 0.0 ~ 1.0 사이의 전략 점수
        """
        try:
            if not strategies or candle_data.empty:
                return 0.5
            
            scores: List[float] = []
            
            for strategy in strategies:
                try:
                    # 전략 등급 기반 점수
                    grade = strategy.get('grade', 'C')
                    grade_scores = {'S': 0.95, 'A': 0.85, 'B': 0.75, 'C': 0.65, 'D': 0.55, 'F': 0.45}
                    grade_score = grade_scores.get(grade, 0.5)
                    
                    # 성능 지표 기반 점수
                    performance = strategy.get('performance_metrics', {})
                    if isinstance(performance, str):
                        import json
                        performance = json.loads(performance) if performance else {}
                    
                    win_rate = performance.get('win_rate', 0.5)
                    profit = performance.get('profit', 0.0)
                    
                    # 수익률 정규화 (-0.2 ~ 0.2 → 0.0 ~ 1.0)
                    normalized_profit = max(0.0, min(1.0, (profit + 0.2) / 0.4))
                    
                    # 등급 점수와 성능 점수 종합
                    strategy_score = (grade_score * 0.6 + win_rate * 0.25 + normalized_profit * 0.15)
                    
                    # 현재 시장 조건에서 전략 조건 만족 여부 체크
                    if self._check_strategy_condition(strategy, candle_data):
                        strategy_score *= 1.1  # 조건 만족시 10% 보너스
                    
                    scores.append(max(0.0, min(1.0, strategy_score)))
                    
                except Exception as e:
                    logger.debug(f"전략 점수 계산 실패 (무시): {e}")
                    continue
            
            if not scores:
                return 0.5
            
            # 상위 50% 전략의 평균 점수 사용 (노이즈 제거)
            scores_sorted = sorted(scores, reverse=True)
            top_half = scores_sorted[:max(1, len(scores_sorted) // 2)]
            avg_score = sum(top_half) / len(top_half)
            
            return max(0.0, min(1.0, avg_score))
            
        except Exception as e:
            logger.error(f"인터벌 전략 점수 계산 실패: {e}")
            return 0.5
    
    def _calculate_interval_strategy_score_by_direction(
        self,
        strategies: List[Dict[str, Any]],
        candle_data: pd.DataFrame,
        direction: str = 'buy'
    ) -> float:
        """🔥 방향별 전략 점수 계산 (매수 그룹 또는 매도 그룹)
        
        Args:
            strategies: 전략 리스트
            candle_data: 캔들 데이터
            direction: 'buy' 또는 'sell'
        
        Returns:
            float: 0.0 ~ 1.0 사이의 전략 점수
        """
        try:
            if not strategies or candle_data.empty:
                return 0.5
            
            # 전략을 방향별로 분류
            buy_strategies = []
            sell_strategies = []
            neutral_strategies = []
            
            for strategy in strategies:
                strategy_direction = self._classify_strategy_direction(strategy)
                if strategy_direction == 'buy':
                    buy_strategies.append(strategy)
                elif strategy_direction == 'sell':
                    sell_strategies.append(strategy)
                else:
                    neutral_strategies.append(strategy)
            
            # 요청한 방향의 전략 선택
            if direction == 'buy':
                target_strategies = buy_strategies
                # 매수 전략이 없으면 중립 전략도 포함 (하위 호환성)
                if not target_strategies:
                    target_strategies = neutral_strategies
            elif direction == 'sell':
                target_strategies = sell_strategies
                # 매도 전략이 없으면 중립 전략도 포함 (하위 호환성)
                if not target_strategies:
                    target_strategies = neutral_strategies
            else:
                target_strategies = strategies  # 전체 전략 사용
            
            if not target_strategies:
                return 0.5
            
            scores: List[float] = []
            
            for strategy in target_strategies:
                try:
                    # 전략 등급 기반 점수
                    grade = strategy.get('grade', 'C')
                    grade_scores = {'S': 0.95, 'A': 0.85, 'B': 0.75, 'C': 0.65, 'D': 0.55, 'F': 0.45}
                    grade_score = grade_scores.get(grade, 0.5)
                    
                    # 성능 지표 기반 점수
                    performance = strategy.get('performance_metrics', {})
                    if isinstance(performance, str):
                        import json
                        performance = json.loads(performance) if performance else {}
                    
                    win_rate = performance.get('win_rate', 0.5)
                    profit = performance.get('profit', 0.0)
                    
                    # 수익률 정규화 (-0.2 ~ 0.2 → 0.0 ~ 1.0)
                    normalized_profit = max(0.0, min(1.0, (profit + 0.2) / 0.4))
                    
                    # 등급 점수와 성능 점수 종합
                    strategy_score = (grade_score * 0.6 + win_rate * 0.25 + normalized_profit * 0.15)
                    
                    # 현재 시장 조건에서 전략 조건 만족 여부 체크
                    if self._check_strategy_condition(strategy, candle_data):
                        strategy_score *= 1.1  # 조건 만족시 10% 보너스
                    
                    scores.append(max(0.0, min(1.0, strategy_score)))
                    
                except Exception as e:
                    logger.debug(f"전략 점수 계산 실패 (무시): {e}")
                    continue
            
            if not scores:
                return 0.5
            
            # 상위 50% 전략의 평균 점수 사용 (노이즈 제거)
            scores_sorted = sorted(scores, reverse=True)
            top_half = scores_sorted[:max(1, len(scores_sorted) // 2)]
            avg_score = sum(top_half) / len(top_half)
            
            logger.debug(f"🔥 {direction} 그룹 전략 점수: {avg_score:.3f} ({len(target_strategies)}개 전략 중 {len(scores)}개 계산)")
            
            return max(0.0, min(1.0, avg_score))
            
        except Exception as e:
            logger.error(f"방향별 전략 점수 계산 실패: {e}")
            return 0.5
    
    def _analyze_strategy_context(
        self,
        coin: str,
        high_grade_strategies: List[Dict[str, Any]],
        multi_interval_candle_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """🔥 고등급 전략의 맥락 분석: 다른 인터벌 및 지표 상태 분석
        
        분석 내용:
        1. A등급 전략이 다른 인터벌(30m, 240m, 1d)에서는 어떻게 작동했는가?
        2. A등급 전략에서 핵심 지표(예: RSI 30 이하)가 좋았다면, 
           그때 다른 지표들(MACD, MFI, ATR 등)은 어떤 상태였는가?
        """
        try:
            context_result = {
                'cross_interval_performance': {},  # {strategy_id: {interval: performance}}
                'indicator_correlations': {},  # {strategy_id: {indicator: correlation}}
                'contextual_patterns': []  # 발견된 패턴들
            }
            
            # 원본 인터벌 추출 (A등급을 받은 인터벌)
            source_intervals = {}
            for strategy in high_grade_strategies:
                strategy_id = strategy.get('id') or strategy.get('strategy_id')
                if not strategy_id:
                    continue
                
                # 전략의 원본 인터벌 추출
                original_interval = strategy.get('interval', '15m')
                source_intervals[strategy_id] = original_interval
            
            # 1) 다른 인터벌에서의 성과 분석
            from rl_pipeline.strategy.router import execute_simple_backtest
            
            for strategy in high_grade_strategies[:10]:  # 상위 10개만 분석 (성능 고려)
                strategy_id = strategy.get('id') or strategy.get('strategy_id')
                if not strategy_id:
                    continue
                
                original_interval = source_intervals.get(strategy_id, '15m')
                context_result['cross_interval_performance'][strategy_id] = {}
                
                # 다른 인터벌에서 백테스트
                for test_interval, test_candle_data in multi_interval_candle_data.items():
                    if test_interval == original_interval:
                        continue  # 원본 인터벌은 스킵
                    
                    if test_candle_data.empty or len(test_candle_data) < 20:
                        continue
                    
                    try:
                        trades, profit, wins, predictive_accuracy = execute_simple_backtest(strategy, test_candle_data)
                        win_rate = wins / trades if trades > 0 else 0.0
                        
                        context_result['cross_interval_performance'][strategy_id][test_interval] = {
                            'trades': trades,
                            'profit': profit,
                            'win_rate': win_rate,
                            'performance_score': profit * 0.6 + win_rate * 0.4
                        }
                        
                        logger.debug(f"  📊 {strategy_id}: {original_interval}→{test_interval} "
                                   f"(거래={trades}, 수익={profit:.2%}, 승률={win_rate:.1%})")
                    except Exception as e:
                        logger.debug(f"⚠️ {strategy_id} {test_interval} 백테스트 실패: {e}")
                        continue
            
            # 2) 지표 간 상관관계 분석 (A등급 전략이 좋았을 때의 지표 상태)
            for strategy in high_grade_strategies[:10]:
                strategy_id = strategy.get('id') or strategy.get('strategy_id')
                if not strategy_id:
                    continue
                
                original_interval = source_intervals.get(strategy_id, '15m')
                original_candles = multi_interval_candle_data.get(original_interval)
                
                if original_candles is None or original_candles.empty:
                    continue
                
                # 전략의 핵심 조건 추출
                strategy_params = self._extract_strategy_indicators(strategy)
                if not strategy_params:
                    continue
                
                # 핵심 지표가 조건을 만족하는 시점 찾기
                indicator_states = self._find_indicator_states_when_condition_met(
                    strategy, original_candles, strategy_params
                )
                
                if indicator_states:
                    context_result['indicator_correlations'][strategy_id] = indicator_states
                    
                    logger.debug(f"  📊 {strategy_id}: 핵심 조건 만족 시점 {len(indicator_states)}개 발견")
                    
                    # 패턴 발견
                    pattern = self._detect_contextual_pattern(strategy_params, indicator_states)
                    if pattern:
                        context_result['contextual_patterns'].append({
                            'strategy_id': strategy_id,
                            'pattern': pattern,
                            'confidence': pattern.get('confidence', 0.5)
                        })
            
            logger.info(f"🔥 [{coin}] 맥락 분석 완료: "
                       f"{len(context_result['cross_interval_performance'])}개 전략 인터벌 분석, "
                       f"{len(context_result['indicator_correlations'])}개 전략 지표 분석, "
                       f"{len(context_result['contextual_patterns'])}개 패턴 발견")
            
            return context_result
            
        except Exception as e:
            logger.error(f"❌ 맥락 분석 실패: {e}")
            return {}
    
    def _extract_strategy_indicators(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """전략에서 핵심 지표 조건 추출"""
        try:
            params = {}
            
            # strategy_conditions에서 추출
            if 'strategy_conditions' in strategy:
                conditions = strategy['strategy_conditions']
                if isinstance(conditions, str):
                    import json
                    conditions = json.loads(conditions) if conditions else {}
                
                if isinstance(conditions, dict):
                    params.update(conditions)
            
            # 직접 필드에서 추출
            indicator_keys = ['rsi_min', 'rsi_max', 'macd_buy_threshold', 'macd_sell_threshold',
                            'volume_ratio_min', 'volume_ratio_max', 'mfi_min', 'mfi_max',
                            'atr_threshold', 'adx_threshold']
            
            for key in indicator_keys:
                if key in strategy:
                    params[key] = strategy[key]
            
            return params
        except Exception as e:
            logger.debug(f"지표 추출 실패: {e}")
            return {}
    
    def _find_indicator_states_when_condition_met(
        self,
        strategy: Dict[str, Any],
        candle_data: pd.DataFrame,
        strategy_params: Dict[str, Any],
        lookback_period: int = 50
    ) -> List[Dict[str, Any]]:
        """전략 조건이 만족되는 시점에서 다른 지표들의 상태 찾기"""
        try:
            states = []
            
            if candle_data.empty or len(candle_data) < lookback_period:
                return states
            
            # 최근 lookback_period 동안의 데이터 분석
            recent_data = candle_data.iloc[-lookback_period:]
            
            for idx, row in recent_data.iterrows():
                condition_met = True
                indicator_state = {}
                
                # RSI 조건 체크
                if 'rsi_min' in strategy_params and 'rsi' in row.index:
                    rsi_val = row.get('rsi', 50)
                    rsi_min = strategy_params.get('rsi_min', 0)
                    rsi_max = strategy_params.get('rsi_max', 100)
                    
                    if not (rsi_min <= rsi_val <= rsi_max):
                        condition_met = False
                    else:
                        indicator_state['rsi'] = rsi_val
                
                # MACD 조건 체크
                if 'macd_buy_threshold' in strategy_params and 'macd' in row.index:
                    macd_val = row.get('macd', 0)
                    macd_threshold = strategy_params.get('macd_buy_threshold', 0)
                    
                    if macd_val <= macd_threshold:
                        condition_met = False
                    else:
                        indicator_state['macd'] = macd_val
                
                # 조건 만족 시, 다른 지표들의 상태 기록
                if condition_met:
                    state = {
                        'timestamp': row.get('timestamp', idx),
                        'price': row.get('close', 0),
                    }
                    
                    # 다른 지표들의 상태 추가
                    for indicator in ['rsi', 'macd', 'macd_signal', 'mfi', 'atr', 'adx', 
                                    'bb_upper', 'bb_middle', 'bb_lower', 'volume_ratio']:
                        if indicator in row.index:
                            state[indicator] = row.get(indicator, None)
                    
                    state.update(indicator_state)
                    states.append(state)
            
            return states
            
        except Exception as e:
            logger.debug(f"지표 상태 찾기 실패: {e}")
            return []
    
    def _detect_contextual_pattern(
        self,
        strategy_params: Dict[str, Any],
        indicator_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """지표 상태들에서 패턴 발견"""
        try:
            if not indicator_states or len(indicator_states) < 5:
                return None
            
            # RSI가 낮을 때 (30 이하) 다른 지표들의 평균 상태
            rsi_low_states = [s for s in indicator_states if s.get('rsi', 50) <= 35]
            if len(rsi_low_states) >= 3:
                avg_macd = sum(s.get('macd', 0) for s in rsi_low_states) / len(rsi_low_states)
                avg_mfi = sum(s.get('mfi', 50) for s in rsi_low_states) / len(rsi_low_states)
                avg_atr = sum(s.get('atr', 0) for s in rsi_low_states) / len(rsi_low_states)
                
                pattern = {
                    'type': 'rsi_low_indicator_state',
                    'rsi_range': 'low',
                    'avg_macd': avg_macd,
                    'avg_mfi': avg_mfi,
                    'avg_atr': avg_atr,
                    'sample_size': len(rsi_low_states),
                    'confidence': min(1.0, len(rsi_low_states) / 10.0)
                }
                
                logger.debug(f"  🔍 패턴 발견: RSI 낮을 때 MACD={avg_macd:.4f}, MFI={avg_mfi:.1f}, ATR={avg_atr:.4f}")
                return pattern
            
            return None
            
        except Exception as e:
            logger.debug(f"패턴 발견 실패: {e}")
            return None
    
    def _calculate_context_based_confidence(
        self,
        interval: str,
        context_analysis: Dict[str, Any],
        strategies: List[Dict[str, Any]],
    ) -> float:
        """🔥 전략별 맥락 분석 결과를 기반으로 인터벌 신뢰도 계산
        
        예: 
        - 고등급 전략이 15분에서 A등급, 30분에서도 좋은 성과 → 30분 인터벌 신뢰도 ↑
        - 고등급 전략이 15분에서 A등급, 240분에서 나쁜 성과 → 240분 인터벌 신뢰도 ↓
        
        Returns:
            float: 0.0 ~ 1.0 사이의 맥락 기반 신뢰도
        """
        try:
            if not context_analysis or 'cross_interval_performance' not in context_analysis:
                return 0.5
            
            cross_perf = context_analysis.get('cross_interval_performance', {})
            if not cross_perf:
                return 0.5
            
            # 이 인터벌에서 테스트된 전략들의 성과 수집
            interval_performances: List[float] = []
            high_grade_count = 0
            good_perf_count = 0
            
            for strategy in strategies[:10]:  # 상위 10개만 고려
                strategy_id = strategy.get('id') or strategy.get('strategy_id')
                if not strategy_id or strategy_id not in cross_perf:
                    continue
                
                # 고등급 전략인지 확인
                grade = strategy.get('grade') or strategy.get('quality_grade', 'C')
                is_high_grade = grade in ['S', 'A']
                
                if is_high_grade:
                    high_grade_count += 1
                
                # 이 인터벌에서의 성과
                interval_perf = cross_perf[strategy_id].get(interval)
                if interval_perf:
                    perf_score = interval_perf.get('performance_score', 0.5)
                    interval_performances.append(perf_score)
                    
                    # 고등급 전략이고 성과가 좋으면
                    if is_high_grade and perf_score > 0.6:
                        good_perf_count += 1
            
            if not interval_performances:
                return 0.5
            
            # 평균 성과 점수
            avg_perf = sum(interval_performances) / len(interval_performances)
            
            # 고등급 전략 비율 및 좋은 성과 비율
            high_grade_ratio = high_grade_count / len(interval_performances) if interval_performances else 0.0
            good_perf_ratio = good_perf_count / max(1, high_grade_count) if high_grade_count > 0 else 0.0
            
            # 🔥 통계적 유의성 검증 강화 (개선 5단계)
            # 1. 표본 크기 검증
            sample_size = len(interval_performances)
            sample_size_factor = min(1.0, sample_size / 10.0)  # 최소 10개 이상 권장
            
            # 2. 분산 검증 (일관성)
            if len(interval_performances) >= 3:
                import numpy as np
                std_dev = float(np.std(interval_performances))
                consistency_factor = max(0.5, 1.0 - std_dev)  # 분산이 낮으면 높은 점수
            else:
                consistency_factor = 0.5  # 표본 부족 시 보수적
            
            # 3. 통계적 유의성 검증 (최소 전략 수)
            min_strategies_required = 3
            significance_factor = 1.0 if high_grade_count >= min_strategies_required else 0.7
            
            # 맥락 기반 신뢰도 계산 (통계적 유의성 반영)
            # 1. 평균 성과가 높으면 신뢰도 ↑
            perf_component = avg_perf
            
            # 2. 고등급 전략이 많고 성과가 좋으면 신뢰도 ↑
            grade_component = high_grade_ratio * good_perf_ratio
            
            # 3. 통계적 유의성 반영
            base_confidence = (
                perf_component * 0.5 +
                grade_component * 0.3 +
                consistency_factor * 0.2  # 일관성 추가
            )
            
            # 통계적 유의성 및 표본 크기 적용
            context_confidence = base_confidence * sample_size_factor * significance_factor
            
            logger.debug(f"  🔥 {interval} 맥락 신뢰도: {context_confidence:.3f} "
                        f"(평균성과={avg_perf:.3f}, 고등급비율={high_grade_ratio:.2f}, 좋은성과비율={good_perf_ratio:.2f}, "
                        f"표본크기={sample_size}, 일관성={consistency_factor:.2f}, 유의성={significance_factor:.2f})")
            
            return max(0.0, min(1.0, context_confidence))
            
        except Exception as e:
            logger.debug(f"맥락 기반 신뢰도 계산 실패: {e}")
            return 0.5
    
    def _calculate_regime_alignment(
        self, interval_regimes: Dict[str, Tuple[str, float]]
    ) -> Tuple[float, str]:
        """🔥 인터벌별 레짐 일치도 계산 및 메인 레짐 결정 (개선 1단계)
        
        Args:
            interval_regimes: {interval: (regime, confidence)} 딕셔너리
        
        Returns:
            (일치도 점수, 메인 레짐)
        """
        try:
            if not interval_regimes:
                return 0.5, "neutral"
            
            # 레짐별 가중치 계산 (신뢰도 기반)
            regime_weights: Dict[str, float] = {}
            for interval, (regime, conf) in interval_regimes.items():
                if regime not in regime_weights:
                    regime_weights[regime] = 0.0
                regime_weights[regime] += conf
            
            # 총합으로 정규화
            total_weight = sum(regime_weights.values())
            if total_weight > 0:
                regime_weights = {k: v / total_weight for k, v in regime_weights.items()}
            
            # 메인 레짐 결정 (가장 높은 가중치)
            main_regime = max(regime_weights.items(), key=lambda x: x[1])[0]
            
            # 일치도 계산: 같은 레짐인 인터벌들의 신뢰도 합
            alignment_score = regime_weights.get(main_regime, 0.0)
            
            # 추가 보정: 인터벌 수가 많을수록 일치도 계산에 반영
            num_intervals = len(interval_regimes)
            if num_intervals >= 3:
                # 최소 2개 이상의 인터벌이 같은 레짐이면 보너스
                same_regime_count = sum(1 for _, (r, _) in interval_regimes.items() if r == main_regime)
                if same_regime_count >= 2:
                    alignment_score = min(1.0, alignment_score * 1.1)  # 10% 보너스
            
            logger.debug(f"  📊 레짐 일치도 계산: 메인={main_regime}, 일치도={alignment_score:.3f}, 레짐분포={regime_weights}")
            
            return alignment_score, main_regime
            
        except Exception as e:
            logger.debug(f"레짐 일치도 계산 실패: {e}")
            return 0.5, "neutral"
    
    def _calculate_regime_consistency_penalty(
        self, interval_regime: str, main_regime: str, regime_alignment: float
    ) -> float:
        """🔥 레짐 불일치 시 신뢰도 조정 배율 계산 (개선 1단계)
        
        Args:
            interval_regime: 인터벌의 레짐
            main_regime: 메인 레짐
            regime_alignment: 전체 레짐 일치도
        
        Returns:
            0.8 ~ 1.0 사이의 배율
        """
        try:
            # 레짐이 일치하면 패널티 없음
            if interval_regime == main_regime:
                return 1.0
            
            # 레짐 일치도에 따라 패널티 조정
            # 일치도가 높으면 불일치 인터벌에 대한 패널티도 완화
            if regime_alignment >= 0.8:
                # 대부분 일치하므로 약간의 패널티만 (0.95)
                return 0.95
            elif regime_alignment >= 0.6:
                # 보통 일치도 (0.9)
                return 0.9
            else:
                # 낮은 일치도 (0.8)
                return 0.8
            
        except Exception as e:
            logger.debug(f"레짐 일치도 패널티 계산 실패: {e}")
            return 0.9  # 기본값
    
    def _calculate_dynamic_analysis_weights(
        self, regime: str, coin: str, interval: str
    ) -> Dict[str, float]:
        """🔥 레짐/코인/인터벌 특성에 따른 동적 가중치 계산 (개선 3단계)
        
        Args:
            regime: 레짐
            coin: 코인 심볼
            interval: 인터벌
        
        Returns:
            {'fractal': ..., 'multi_timeframe': ..., 'indicator_cross': ..., 'context': ...}
        """
        try:
            # 기본 가중치 (균등)
            base_weights = {
                'fractal': 0.25,
                'multi_timeframe': 0.25,
                'indicator_cross': 0.25,
                'context': 0.25
            }
            
            # 레짐별 조정
            regime_adjustments = self._get_regime_analysis_adjustments(regime)
            
            # 코인 특성별 조정
            coin_adjustments = self._get_coin_analysis_adjustments(coin)
            
            # 인터벌별 조정
            interval_adjustments = self._get_interval_analysis_adjustments(interval)
            
            # 모든 조정 적용
            final_weights = {}
            for key in base_weights.keys():
                adjustment = (
                    regime_adjustments.get(key, 0.0) +
                    coin_adjustments.get(key, 0.0) +
                    interval_adjustments.get(key, 0.0)
                )
                final_weights[key] = max(0.1, min(0.5, base_weights[key] + adjustment))
            
            # 정규화 (합이 1.0이 되도록)
            total = sum(final_weights.values())
            if total > 0:
                final_weights = {k: v / total for k, v in final_weights.items()}
            
            logger.debug(f"  📊 동적 가중치 [{coin}-{interval}-{regime}]: {final_weights}")
            return final_weights
            
        except Exception as e:
            logger.debug(f"동적 가중치 계산 실패: {e}")
            return {'fractal': 0.25, 'multi_timeframe': 0.25, 'indicator_cross': 0.25, 'context': 0.25}
    
    def _get_regime_analysis_adjustments(self, regime: str) -> Dict[str, float]:
        """레짐별 분석 가중치 조정"""
        adjustments = {k: 0.0 for k in ('fractal', 'multi_timeframe', 'indicator_cross', 'context')}
        
        if regime in ('extreme_bullish', 'extreme_bearish'):
            # 극단적 레짐 → 변동성/프랙탈 중시
            adjustments['fractal'] = 0.15
            adjustments['multi_timeframe'] = 0.1
            adjustments['indicator_cross'] = -0.05
            adjustments['context'] = -0.05
        
        elif regime in ('bullish', 'bearish'):
            # 명확한 추세 → 멀티타임프레임 중시
            adjustments['multi_timeframe'] = 0.15
            adjustments['indicator_cross'] = 0.05
            adjustments['fractal'] = -0.05
            adjustments['context'] = -0.05
        
        elif regime in ('sideways_bullish', 'sideways_bearish'):
            # 횡보 → 지표 교차 중시
            adjustments['indicator_cross'] = 0.15
            adjustments['context'] = 0.05
            adjustments['fractal'] = -0.1
            adjustments['multi_timeframe'] = -0.1
        
        elif regime == 'neutral':
            # 중립 → 맥락 분석 중시
            adjustments['context'] = 0.1
            adjustments['indicator_cross'] = 0.05
            adjustments['fractal'] = -0.05
            adjustments['multi_timeframe'] = -0.05
        
        return adjustments
    
    def _get_coin_analysis_adjustments(self, coin: str) -> Dict[str, float]:
        """코인 특성별 분석 가중치 조정"""
        adjustments = {k: 0.0 for k in ('fractal', 'multi_timeframe', 'indicator_cross', 'context')}
        
        # 코인 이름 길이로 간단히 분류 (실제로는 더 정교한 분류 가능)
        coin_len = len(coin)
        
        if coin_len <= 4:
            # 주요 코인 → 프랙탈/멀티타임프레임 중시
            adjustments['fractal'] = 0.05
            adjustments['multi_timeframe'] = 0.05
        elif coin_len == 5:
            # 고성능 코인 → 지표 교차 중시
            adjustments['indicator_cross'] = 0.05
            adjustments['context'] = 0.02
        else:
            # 알트코인 → 맥락 분석 중시
            adjustments['context'] = 0.05
            adjustments['indicator_cross'] = 0.02
        
        return adjustments
    
    def _get_interval_analysis_adjustments(self, interval: str) -> Dict[str, float]:
        """인터벌별 분석 가중치 조정"""
        adjustments = {k: 0.0 for k in ('fractal', 'multi_timeframe', 'indicator_cross', 'context')}
        
        if interval in ('15m', '30m'):
            # 단기 인터벌 → 지표 교차 중시
            adjustments['indicator_cross'] = 0.05
            adjustments['context'] = 0.02
            adjustments['fractal'] = -0.03
            adjustments['multi_timeframe'] = -0.02
        
        elif interval in ('240m', '4h'):
            # 중기 인터벌 → 멀티타임프레임 중시
            adjustments['multi_timeframe'] = 0.08
            adjustments['fractal'] = 0.02
            adjustments['indicator_cross'] = -0.05
        
        elif interval in ('1d', '1w'):
            # 장기 인터벌 → 프랙탈 중시
            adjustments['fractal'] = 0.1
            adjustments['multi_timeframe'] = 0.02
            adjustments['indicator_cross'] = -0.05
            adjustments['context'] = -0.02
        
        return adjustments
    
    def _calculate_interval_strategy_score_with_context(
        self,
        strategies: List[Dict[str, Any]],
        candle_data: pd.DataFrame,
        interval: str,
        context_analysis: Dict[str, Any],
    ) -> float:
        """맥락 분석을 반영한 인터벌별 전략 점수 계산"""
        try:
            # 기본 점수 계산
            base_score = self._calculate_interval_strategy_score(strategies, candle_data)
            
            # 맥락 분석이 없으면 기본 점수 반환
            if not context_analysis or 'cross_interval_performance' not in context_analysis:
                return base_score
            
            # 맥락 분석 기반 보정
            context_adjustment = 0.0
            
            for strategy in strategies[:10]:  # 상위 10개만
                strategy_id = strategy.get('id') or strategy.get('strategy_id')
                if not strategy_id:
                    continue
                
                # 이 인터벌에서의 성과 확인
                if strategy_id in context_analysis.get('cross_interval_performance', {}):
                    interval_perf = context_analysis['cross_interval_performance'][strategy_id].get(interval)
                    if interval_perf:
                        perf_score = interval_perf.get('performance_score', 0.5)
                        # 다른 인터벌에서도 좋으면 보정
                        if perf_score > 0.6:
                            context_adjustment += 0.05
                        elif perf_score < 0.4:
                            context_adjustment -= 0.03
            
            # 맥락 보정 적용 (최대 ±10%)
            context_adjustment = max(-0.1, min(0.1, context_adjustment / len(strategies)))
            adjusted_score = base_score + context_adjustment
            
            logger.debug(f"  📊 {interval} 맥락 보정: {base_score:.3f} → {adjusted_score:.3f} (보정={context_adjustment:+.3f})")
            
            return max(0.0, min(1.0, adjusted_score))
            
        except Exception as e:
            logger.debug(f"맥락 기반 점수 계산 실패, 기본 점수 사용: {e}")
            return self._calculate_interval_strategy_score(strategies, candle_data)

    # ------------------------------
    # 글로벌 전략 분석
    # ------------------------------
    def analyze_global_strategies(
        self,
        global_strategies: List[Dict[str, Any]],
        all_coin_results: List[CoinSignalScore],
    ) -> GlobalSignalScore:
        """여러 코인/인터벌 조합을 단일 전역 시그널로 요약"""
        try:
            logger.info(f"[Global] 전략 분석 시작 (전략 {len(global_strategies)}개)")

            # 🔥 글로벌 전략을 매수/매도 그룹으로 분리
            buy_global_strategies = []
            sell_global_strategies = []
            neutral_global_strategies = []
            
            for strategy in global_strategies:
                direction = self._classify_strategy_direction(strategy)
                if direction == 'buy':
                    buy_global_strategies.append(strategy)
                elif direction == 'sell':
                    sell_global_strategies.append(strategy)
                else:
                    neutral_global_strategies.append(strategy)
            
            logger.info(f"  📊 글로벌 전략 분류: 매수 {len(buy_global_strategies)}개, 매도 {len(sell_global_strategies)}개, 중립 {len(neutral_global_strategies)}개")

            all_combinations = self._get_all_combinations(all_coin_results)

            # 정책 훈련으로 전역 패턴 추론
            if self.learning_systems_available:
                try:
                    policy_result = self.policy_trainer.train_policy(all_combinations)
                except Exception as e:
                    logger.warning(f"[Global] 정책 추론 실패: {e}")
                    policy_result = {"policy_improvement": 0.0, "convergence_rate": 0.0}
            else:
                policy_result = {"policy_improvement": 0.0, "convergence_rate": 0.0}

            # 성능 모니터링
            if self.learning_systems_available:
                try:
                    fitness_scores = [combo.get("signal_score", 0.0) for combo in all_combinations]
                    performance_result = self.performance_monitor.analyze_jax_performance(
                        population=all_combinations, fitness_scores=fitness_scores
                    )
                except Exception as e:
                    logger.warning(f"[Global] 성능 분석 실패: {e}")
                    performance_result = {"mean_fitness": 0.5, "std_fitness": 0.1}
            else:
                performance_result = {"mean_fitness": 0.5, "std_fitness": 0.1}

            top_performers = self._get_top_performers(all_combinations)
            top_coins = self._get_top_coins(all_coin_results)
            top_intervals = self._get_top_intervals(all_coin_results)

            result = GlobalSignalScore(
                overall_score=float(performance_result.get("mean_fitness", 0.0)),
                overall_confidence=float(performance_result.get("std_fitness", 0.0)),
                policy_improvement=float(policy_result.get("policy_improvement", 0.0)),
                convergence_rate=float(policy_result.get("convergence_rate", 0.0)),
                top_performers=top_performers,
                top_coins=top_coins,
                top_intervals=top_intervals,
                created_at=datetime.now().isoformat(),
            )
            logger.info(f"✅ [Global] 분석 완료: 평균 점수 {result.overall_score:.3f} (매수: {len(buy_global_strategies)}개, 매도: {len(sell_global_strategies)}개)")
            return result

        except Exception as e:
            logger.error(f"❌ [Global] 전략 분석 실패: {e}")
            return self._create_default_global_signal_score()

    # ------------------------------------------------------------------
    # 세부 분석 로직 (프랙탈/다중시간대/지표/코인특화 등)
    # ------------------------------------------------------------------
    def _analyze_fractal_patterns(self, coin: str, interval: str, candle_data: pd.DataFrame) -> float:
        try:
            # 🔥 수정: 더미 함수 대신 실제 계산 로직 바로 사용
            return self._calculate_basic_fractal_score(candle_data)
        except Exception as e:
            logger.error(f"[{coin}] 프랙탈 분석 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.5

    def _analyze_multi_timeframe(self, coin: str, interval: str, candle_data: pd.DataFrame) -> float:
        try:
            if candle_data.empty or len(candle_data) < 20:
                return 0.5
            st = self._calculate_short_term_trend(candle_data)
            mt = self._calculate_medium_term_trend(candle_data)
            lt = self._calculate_long_term_trend(candle_data)
            align = self._calculate_trend_alignment(st, mt, lt)
            score = (align + abs(st) + abs(mt) + abs(lt)) / 4.0
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"[{coin}] 다중시간대 분석 실패: {e}")
            return 0.5

    def _analyze_indicator_correlations(self, coin: str, interval: str, candle_data: pd.DataFrame) -> float:
        try:
            # 🔥 수정: 더미 함수 대신 실제 계산 로직 바로 사용
            return self._calculate_basic_indicator_score(candle_data)
        except Exception as e:
            logger.error(f"[{coin}] 지표 상관 분석 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.5

    # ---- 기본 보조 계산들
    def _calculate_short_term_trend(self, data: pd.DataFrame) -> float:
        try:
            if "close" not in data.columns or len(data) < 5:
                return 0.0
            closes = data["close"].dropna()
            if len(closes) < 5:
                return 0.0
            short_ma = closes.tail(5).mean()
            prev_short_ma = closes.tail(10).head(5).mean() if len(closes) >= 10 else short_ma
            trend = (short_ma - prev_short_ma) / prev_short_ma if prev_short_ma > 0 else 0.0
            return max(-1.0, min(1.0, float(trend)))
        except Exception as e:
            logger.error(f"단기 추세 계산 실패: {e}")
            return 0.0

    def _calculate_medium_term_trend(self, data: pd.DataFrame) -> float:
        try:
            if "close" not in data.columns or len(data) < 10:
                return 0.0
            closes = data["close"].dropna()
            if len(closes) < 10:
                return 0.0
            medium_ma = closes.tail(10).mean()
            prev_medium_ma = closes.tail(20).head(10).mean() if len(closes) >= 20 else medium_ma
            trend = (medium_ma - prev_medium_ma) / prev_medium_ma if prev_medium_ma > 0 else 0.0
            return max(-1.0, min(1.0, float(trend)))
        except Exception as e:
            logger.error(f"중기 추세 계산 실패: {e}")
            return 0.0

    def _calculate_long_term_trend(self, data: pd.DataFrame) -> float:
        try:
            if "close" not in data.columns or len(data) < 20:
                return 0.0
            closes = data["close"].dropna()
            if len(closes) < 20:
                return 0.0
            long_ma = closes.tail(20).mean()
            prev_long_ma = closes.tail(40).head(20).mean() if len(closes) >= 40 else long_ma
            trend = (long_ma - prev_long_ma) / prev_long_ma if prev_long_ma > 0 else 0.0
            return max(-1.0, min(1.0, float(trend)))
        except Exception as e:
            logger.error(f"장기 추세 계산 실패: {e}")
            return 0.0

    def _calculate_trend_alignment(self, st: float, mt: float, lt: float) -> float:
        try:
            signs = [st, mt, lt]
            pos = sum(1 for v in signs if v > 0)
            neg = sum(1 for v in signs if v < 0)
            return max(pos, neg) / 3.0
        except Exception as e:
            logger.error(f"추세 정렬 계산 실패: {e}")
            return 0.5

    def _calculate_basic_fractal_score(self, data: pd.DataFrame) -> float:
        try:
            if "close" not in data.columns or len(data) < 10:
                return 0.5
            changes = data["close"].pct_change().dropna()
            vol = float(changes.std())
            return min(1.0, vol * 100.0)  # 1% 표준편차 ≈ 1.0
        except Exception as e:
            logger.error(f"기본 프랙탈 점수 계산 실패: {e}")
            return 0.5

    def _calculate_basic_indicator_score(self, data: pd.DataFrame) -> float:
        try:
            if data is None or data.empty:
                return 0.5
            scores: List[float] = []

            # RSI
            if "rsi" in data.columns:
                rsi_values = data["rsi"].dropna()
                if len(rsi_values) > 0:
                    scores.append(float(rsi_values.iloc[-1]) / 100.0)

            # MACD
            if "macd" in data.columns and "macd_signal" in data.columns:
                macd_values = data["macd"].dropna()
                macd_signal_values = data["macd_signal"].dropna()
                if len(macd_values) > 0 and len(macd_signal_values) > 0:
                    diff = float(macd_values.iloc[-1] - macd_signal_values.iloc[-1])
                    macd_score = (diff + 0.1) / 0.2  # -0.1~0.1 -> 0~1
                    macd_score = max(0.0, min(1.0, macd_score))
                    scores.append(macd_score)

            return (sum(scores) / len(scores)) if scores else 0.5
        except Exception as e:
            logger.error(f"기본 지표 점수 계산 실패: {e}")
            return 0.5

    # ---- 마켓/시뮬레이션/최종 결합
    def _get_market_data(self, coin: str, interval: str, candle_data: pd.DataFrame) -> Dict[str, Any]:
        """시장 데이터 추출 - 공통 함수 사용"""
        try:
            from rl_pipeline.core.utils import extract_market_data_from_candles
            return extract_market_data_from_candles(candle_data)
        except Exception as e:
            logger.error(f"[{coin}] 시장 데이터 추출 실패: {e}")
            return {"close": [100.0], "volume": [1_000_000.0]}

    def _calculate_final_signal_score(
        self,
        fractal_score: float,
        multi_timeframe_score: float,
        indicator_cross_score: float,
        ensemble_result: Any,
        simulation_results: List[Dict[str, Any]],
        coin: Optional[str] = None,
        interval: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> float:
        """(레거시) 단일 값 기반 최종 점수 계산 - 유지용"""
        try:
            weights = self._calculate_dynamic_weights(
                fractal_score, multi_timeframe_score, indicator_cross_score,
                ensemble_result, simulation_results, coin, interval, regime
            )
            ensemble_score = (
                ensemble_result.ensemble_prediction
                if hasattr(ensemble_result, "ensemble_prediction")
                else (ensemble_result.get("ensemble_prediction", 0.5) if isinstance(ensemble_result, dict) else 0.5)
            )

            simulation_score = 0.5
            if simulation_results:
                profits = [sim.get("profit", 0.0) for sim in simulation_results]
                win_rates = [sim.get("win_rate", 0.0) for sim in simulation_results]
                if profits and win_rates:
                    simulation_score = (sum(profits) / len(profits) + sum(win_rates) / len(win_rates)) / 2.0

            final_score = (
                fractal_score * weights["fractal"]
                + multi_timeframe_score * weights["multi_timeframe"]
                + indicator_cross_score * weights["indicator_cross"]
                + ensemble_score * weights["ensemble"]
                + simulation_score * weights["simulation"]
            )
            return max(0.0, min(1.0, float(final_score)))
        except Exception as e:
            logger.error(f"최종 신호 점수 계산 실패: {e}")
            return 0.5

    def _calculate_dynamic_weights(
        self,
        fractal_score: float,
        multi_timeframe_score: float,
        indicator_cross_score: float,
        ensemble_result: Any,
        simulation_results: List[Dict[str, Any]],
        coin: Optional[str] = None,
        interval: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        try:
            base = {"fractal": 0.2, "multi_timeframe": 0.2, "indicator_cross": 0.2, "ensemble": 0.3, "simulation": 0.1}
            adj = self._analyze_weight_adjustments(
                fractal_score, multi_timeframe_score, indicator_cross_score,
                ensemble_result, simulation_results, coin, interval, regime
            )
            dyn = {k: max(0.05, min(0.5, base[k] + adj.get(k, 0.0))) for k in base}
            s = sum(dyn.values())
            dyn = {k: v / s for k, v in dyn.items()} if s > 0 else base
            logger.info(f"가중치(동적): {dyn}")
            return dyn
        except Exception as e:
            logger.error(f"가중치 계산 실패: {e}")
            return {"fractal": 0.2, "multi_timeframe": 0.2, "indicator_cross": 0.2, "ensemble": 0.3, "simulation": 0.1}

    def _analyze_weight_adjustments(
        self,
        fractal_score: float,
        multi_timeframe_score: float,
        indicator_cross_score: float,
        ensemble_result: Any,
        simulation_results: List[Dict[str, Any]],
        coin: Optional[str] = None,
        interval: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        try:
            adj = {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "ensemble", "simulation")}

            # 레짐 기반
            if regime:
                for k, v in self._get_regime_based_adjustments(regime).items():
                    adj[k] += v

            # 스코어 기반
            if fractal_score > 0.7: adj["fractal"] += 0.03
            elif fractal_score < 0.3: adj["fractal"] -= 0.03
            if multi_timeframe_score > 0.7: adj["multi_timeframe"] += 0.03
            elif multi_timeframe_score < 0.3: adj["multi_timeframe"] -= 0.03
            if indicator_cross_score > 0.7: adj["indicator_cross"] += 0.03
            elif indicator_cross_score < 0.3: adj["indicator_cross"] -= 0.03

            # 앙상블 신뢰도
            conf = None
            if hasattr(ensemble_result, "confidence_score"):
                conf = getattr(ensemble_result, "confidence_score", None)
            elif isinstance(ensemble_result, dict):
                conf = ensemble_result.get("confidence_score", None)
            if conf is not None:
                if conf > 0.8: adj["ensemble"] += 0.1
                elif conf < 0.3: adj["ensemble"] -= 0.1

            # 시뮬레이션 성능
            if simulation_results:
                avg_p = sum(sim.get("total_return", 0.0) for sim in simulation_results) / len(simulation_results)
                avg_w = sum(sim.get("win_rate", 0.0) for sim in simulation_results) / len(simulation_results)
                if avg_p > 0.05 and avg_w > 0.6:
                    adj["simulation"] += 0.05; adj["ensemble"] += 0.02
                elif avg_p < -0.02 or avg_w < 0.4:
                    adj["simulation"] -= 0.05; adj["ensemble"] -= 0.02

            # 코인/인터벌 특성
            if coin:
                for k, v in self._get_coin_based_adjustments(coin).items():
                    adj[k] += v
            if interval:
                for k, v in self._get_interval_based_adjustments(interval).items():
                    adj[k] += v

            return adj
        except Exception as e:
            logger.error(f"가중치 조정 분석 실패: {e}")
            return {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "ensemble", "simulation")}

    # ---- 레짐 계열 조정표
    def _get_regime_based_adjustments(self, regime: str) -> Dict[str, float]:
        adj = {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "ensemble", "simulation")}
        if regime in ("extreme_bullish", "extreme_bearish"):
            adj["fractal"] += 0.05; adj["multi_timeframe"] += 0.05; adj["ensemble"] -= 0.05
        elif regime in ("bullish", "bearish"):
            adj["indicator_cross"] += 0.05; adj["ensemble"] += 0.05
        elif regime in ("sideways_bullish", "sideways_bearish"):
            adj["simulation"] += 0.05; adj["ensemble"] += 0.03
        return adj

    def _get_regime_fractal_adjustments(self, regime: str) -> Dict[str, float]:
        base = {"15m": 0.0, "30m": 0.0, "240m": 0.0, "1d": 0.0}
        if regime in ("extreme_bullish", "extreme_bearish"):
            base.update({"15m": 0.1, "240m": 0.1, "1d": 0.1})
        elif regime in ("bullish", "bearish"):
            base["240m"] = 0.1; base["1d"] = 0.1
        elif regime in ("sideways_bullish", "sideways_bearish"):
            base.update({"15m": 0.1, "30m": 0.1})
        elif regime == "neutral":
            base.update({"30m": 0.05, "240m": 0.05, "1d": 0.05})
        return base

    def _get_regime_multi_timeframe_adjustments(self, regime: str) -> Dict[str, float]:
        """인터벌별 다중시간대 분석 가중치"""
        adj = {"15m": 0.0, "30m": 0.0, "240m": 0.0, "1d": 0.0}
        if regime in ("extreme_bullish", "extreme_bearish"):
            adj["240m"] = 0.2; adj["1d"] = 0.2  # 장기 인터벌 강조
        elif regime in ("bullish", "bearish"):
            adj["30m"] = 0.2; adj["240m"] = 0.1; adj["1d"] = 0.1  # 중기/장기 인터벌 강조
        elif regime in ("sideways_bullish", "sideways_bearish"):
            adj["15m"] = 0.2  # 중기 인터벌 강조
        elif regime == "neutral":
            adj["15m"] = adj["30m"] = adj["240m"] = adj["1d"] = 0.05
        return adj

    def _get_regime_indicator_adjustments(self, regime: str) -> Dict[str, float]:
        adj = {"rsi": 0.0, "macd": 0.0, "mfi": 0.0, "atr": 0.0, "adx": 0.0, "bb": 0.0}
        if regime in ("extreme_bullish", "extreme_bearish"):
            adj["rsi"] = 0.2; adj["bb"] = 0.2; adj["atr"] = 0.15
        elif regime in ("bullish", "bearish"):
            adj["macd"] = 0.2; adj["adx"] = 0.15; adj["mfi"] = 0.1
        elif regime in ("sideways_bullish", "sideways_bearish"):
            adj["rsi"] = 0.1; adj["bb"] = 0.1; adj["atr"] = 0.1
        elif regime == "neutral":
            adj["rsi"] = adj["macd"] = adj["mfi"] = adj["atr"] = adj["adx"] = adj["bb"] = 0.05
        return adj

    # ---- 저장/조회/비율 계산
    def _save_coin_analysis_ratios(
        self,
        coin: str,
        interval: str,
        regime: str,
        analysis_modules: Dict[str, float],
        fractal_ratios: Dict[str, float],
        multi_timeframe_ratios: Dict[str, float],
        indicator_cross_ratios: Dict[str, float],
    ) -> bool:
        try:
            from rl_pipeline.db.writes import save_coin_analysis_ratios  # type: ignore
            ratios_data = {
                "fractal_ratios": fractal_ratios,
                "multi_timeframe_ratios": multi_timeframe_ratios,
                "indicator_cross_ratios": indicator_cross_ratios,
                "coin_specific_ratios": {},
                "volatility_ratios": {},
                "volume_ratios": {},
                "optimal_modules": analysis_modules,
                "performance_score": 0.0,
                "accuracy_score": 0.0,
            }
            # regime을 analysis_type으로 사용
            return bool(save_coin_analysis_ratios(coin, interval, regime, ratios_data))
        except Exception as e:
            logger.debug(f"[{coin}] 분석 비율 저장 실패(무시): {e}")
            return False

    def _save_interval_weights(
        self,
        coin: str,
        regime: str,
        interval_weights: Dict[str, float],
    ) -> bool:
        """🔥 코인별 인터벌 가중치를 DB에 저장 (Signal Selector에서 사용)"""
        try:
            from rl_pipeline.db.writes import save_coin_analysis_ratios  # type: ignore

            # 코인별로 하나의 레코드만 저장 (interval="all"로 통합)
            ratios_data = {
                "fractal_ratios": {},
                "multi_timeframe_ratios": {},
                "indicator_cross_ratios": {},
                "coin_specific_ratios": {},
                "volatility_ratios": {},
                "volume_ratios": {},
                "optimal_modules": {},
                "interval_weights": interval_weights,  # 🔥 핵심 데이터
                "performance_score": 0.0,
                "accuracy_score": 0.0,
            }

            # interval="all"로 저장하여 멀티 인터벌 가중치임을 표시
            result = save_coin_analysis_ratios(coin, "all", regime, ratios_data)
            if result:
                # 🔥 소숫점 정리 (3자리) - numpy 타입을 float로 변환
                formatted_weights = {k: float(round(v, 3)) for k, v in interval_weights.items()}
                logger.info(f"✅ [{coin}] 인터벌 가중치 저장 완료: {formatted_weights}")
            return bool(result)
        except Exception as e:
            logger.debug(f"[{coin}] 인터벌 가중치 저장 실패(무시): {e}")
            return False

    def _classify_coin_type(self, coin: str) -> str:
        # 하드코딩 심볼 제거: 간단 휴리스틱(길이/문자) 기반 분류
        sym = coin.upper()
        if sym.endswith('USD') or sym.endswith('USDT') or sym.endswith('USDC'):
            return "stable_coin"
        if len(sym) <= 4:
            return "major_coin"
        if len(sym) == 5:
            return "high_performance"
        return "alt_coin"
    
    def _select_optimal_analysis_modules(self, coin: str, interval: str, regime: str, candle_data: pd.DataFrame) -> Dict[str, float]:
        """최적 분석 모듈 선택"""
        try:
            modules = {"fractal": 0.6, "multi_timeframe": 0.7, "indicator_cross": 0.6, "coin_specific": 0.5, "volatility": 0.4, "volume": 0.4}
            
            # 코인 특성 기반 조정
            characteristics = self._analyze_coin_characteristics(coin, candle_data)
            if characteristics.get("is_major_coin"):
                modules["fractal"] = 0.8
                modules["multi_timeframe"] = 0.9
                modules["indicator_cross"] = 0.7
            elif characteristics.get("is_high_volatility"):
                modules["volatility"] = 0.9
                modules["coin_specific"] = 0.8
            elif characteristics.get("is_low_volume"):
                modules["volume"] = 0.9
                modules["coin_specific"] = 0.7
            else:
                modules["coin_specific"] = 0.9
                modules["volatility"] = 0.7
            
            # 인터벌 조정
            interval_adj = self._get_interval_module_adjustments(interval)
            for module, adjustment in interval_adj.items():
                if module in modules:
                    modules[module] += adjustment
            
            # 레짐 조정
            regime_adj = self._get_regime_module_adjustments(regime)
            for module, adjustment in regime_adj.items():
                if module in modules:
                    modules[module] += adjustment
            
            # 0.4 이상만 선택
            selected = {k: v for k, v in modules.items() if v >= 0.4}
            
            # 최소 3개 보장
            if len(selected) < 3:
                sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
                for module, score in sorted_modules:
                    if module not in selected and len(selected) < 3:
                        selected[module] = score
            
            # 소수점 2자리로 포맷팅
            selected_formatted = {k: round(v, 2) for k, v in selected.items()}
            logger.info(f"[{coin}] 선택된 분석 모듈: {selected_formatted}")
            return selected
        except Exception as e:
            logger.error(f"[{coin}] 분석 모듈 선택 실패: {e}")
            return {"fractal": 0.6, "multi_timeframe": 0.6, "indicator_cross": 0.6}
    
    def _analyze_coin_characteristics(self, coin: str, candle_data: pd.DataFrame) -> Dict[str, Any]:
        """코인 특성 분석"""
        try:
            char = {"is_major_coin": False, "is_high_volatility": False, "is_low_volume": False, "is_altcoin": False, "volatility_score": 0.0, "volume_score": 0.0}
            char["is_major_coin"] = len(coin) <= 4
            
            if not candle_data.empty:
                if "atr" in candle_data.columns:
                    avg_atr = candle_data["atr"].mean()
                    char["volatility_score"] = avg_atr
                    char["is_high_volatility"] = avg_atr > 0.05
                
                if "volume_ratio" in candle_data.columns:
                    avg_volume_ratio = candle_data["volume_ratio"].mean()
                    char["volume_score"] = avg_volume_ratio
                    char["is_low_volume"] = avg_volume_ratio < 0.8
            
            char["is_altcoin"] = not char["is_major_coin"]
            return char
        except Exception as e:
            logger.error(f"[{coin}] 코인 특성 분석 실패: {e}")
            return {"is_major_coin": False, "is_high_volatility": False, "is_low_volume": False, "is_altcoin": True, "volatility_score": 0.0, "volume_score": 0.0}
    
    def _get_interval_module_adjustments(self, interval: str) -> Dict[str, float]:
        """인터벌별 분석 모듈 조정"""
        adj = {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "coin_specific", "volatility", "volume")}
        if interval in ("15m",):
            adj["indicator_cross"] = 0.2
            adj["volume"] = 0.1
        elif interval in ("1h", "4h"):
            adj["multi_timeframe"] = 0.2
            adj["volatility"] = 0.1
        elif interval in ("1d", "1w"):
            adj["fractal"] = 0.2
            adj["coin_specific"] = 0.1
        return adj
    
    def _get_regime_module_adjustments(self, regime: str) -> Dict[str, float]:
        """레짐별 분석 모듈 조정"""
        adj = {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "coin_specific", "volatility", "volume")}
        if regime in ("extreme_bullish", "extreme_bearish"):
            adj["fractal"] = 0.2
            adj["volatility"] = 0.2
        elif regime in ("bullish", "bearish"):
            adj["multi_timeframe"] = 0.2
            adj["indicator_cross"] = 0.1
        elif regime in ("sideways_bullish", "sideways_bearish"):
            adj["coin_specific"] = 0.2
            adj["volume"] = 0.1
        return adj
    
    def _get_coin_based_adjustments(self, coin: str) -> Dict[str, float]:
        """코인별 가중치 조정"""
        adj = {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "ensemble", "simulation")}
        if len(coin) <= 4:
            adj["fractal"] += 0.02
            adj["multi_timeframe"] += 0.02
        else:
            adj["ensemble"] += 0.03
            adj["simulation"] += 0.02
        return adj
    
    def _get_interval_based_adjustments(self, interval: str) -> Dict[str, float]:
        """인터벌별 가중치 조정"""
        adj = {k: 0.0 for k in ("fractal", "multi_timeframe", "indicator_cross", "ensemble", "simulation")}
        if interval in ("15m",):
            adj["indicator_cross"] = 0.03
            adj["simulation"] = 0.02
        elif interval in ("1h", "4h"):
            adj["multi_timeframe"] = 0.03
            adj["ensemble"] = 0.02
        elif interval in ("1d", "1w"):
            adj["fractal"] = 0.03
            adj["multi_timeframe"] = 0.02
        return adj

    def _get_coin_optimal_fractal_intervals(self, coin: str, regime: str = "neutral") -> Dict[str, float]:
        try:
            # 코인 비하드코딩: 코인 유형별 기본 비율만 정의 (모든 코인 적용)
            type_map = {
                "major_coin": {"15m": 0.45, "30m": 0.55, "240m": 0.7, "1d": 0.8},
                "high_performance": {"15m": 0.6, "30m": 0.7, "240m": 0.55, "1d": 0.6},
                "exchange_coin": {"15m": 0.55, "30m": 0.65, "240m": 0.6, "1d": 0.65},
                "academic_coin": {"15m": 0.35, "30m": 0.45, "240m": 0.75, "1d": 0.8},
                "alt_coin": {"15m": 0.5, "30m": 0.6, "240m": 0.7, "1d": 0.75},
            }
            ctype = self._classify_coin_type(coin)
            base = type_map.get(ctype, type_map["alt_coin"])

            adj = self._get_regime_fractal_adjustments(regime)
            out = {}
            for k, v in base.items():
                out[k] = round(max(0.1, min(1.0, v + adj.get(k, 0.0))), 2)
            logger.info(f"[{coin}] {regime} 프랙탈 비율: {out}")
            return out
        except Exception as e:
            logger.error(f"[{coin}] 프랙탈 비율 계산 실패: {e}")
            return {"15m": 0.5, "30m": 0.5, "240m": 0.5, "1d": 0.5}

    def _get_coin_optimal_multi_timeframe_ratios(self, coin: str, regime: str = "neutral") -> Dict[str, float]:
        """다중시간대: 각 인터벌별 추세 분석 (프랙탈과 구분 - 추세 단계별 가중치)"""
        try:
            # 코인 비하드코딩: 코인 유형별 기본 비율만 정의 (모든 코인 적용)
            type_map = {
                "major_coin": {"15m": 0.5, "30m": 0.65, "240m": 0.8, "1d": 0.85},
                "high_performance": {"15m": 0.7, "30m": 0.6, "240m": 0.4, "1d": 0.45},
                "exchange_coin": {"15m": 0.6, "30m": 0.65, "240m": 0.5, "1d": 0.55},
                "academic_coin": {"15m": 0.4, "30m": 0.55, "240m": 0.9, "1d": 0.9},
                "alt_coin": {"15m": 0.6, "30m": 0.7, "240m": 0.7, "1d": 0.75},
            }
            ctype = self._classify_coin_type(coin)
            base = type_map.get(ctype, type_map["alt_coin"])

            # 레짐 조정
            adj = self._get_regime_multi_timeframe_adjustments(regime)
            out = {k: round(max(0.1, min(1.0, base.get(k, 0.5) + adj.get(k, 0.0))), 2) for k in base}
            logger.info(f"[{coin}] {regime} 다중시간대 비율 (인터벌별): {out}")
            return out
        except Exception as e:
            logger.error(f"[{coin}] 다중시간대 비율 계산 실패: {e}")
            return {"15m": 0.6, "30m": 0.7, "240m": 0.7, "1d": 0.75}

    def _get_coin_optimal_indicator_cross_ratios(self, coin: str, regime: str = "neutral") -> Dict[str, float]:
        try:
            ctype = self._classify_coin_type(coin)
            # 하드코딩 제거: 유형별 기본 비율 사용
            type_map = {
                "major_coin": {"rsi": 0.62, "macd": 0.66, "mfi": 0.58, "atr": 0.52, "adx": 0.62, "bb": 0.6},
                "high_performance": {"rsi": 0.65, "macd": 0.65, "mfi": 0.6, "atr": 0.55, "adx": 0.6, "bb": 0.58},
                "stable_coin": {"rsi": 0.55, "macd": 0.6, "mfi": 0.6, "atr": 0.45, "adx": 0.55, "bb": 0.55},
                "alt_coin": {"rsi": 0.6, "macd": 0.6, "mfi": 0.6, "atr": 0.5, "adx": 0.6, "bb": 0.6},
            }
            base = type_map.get(ctype, type_map["alt_coin"])

            adj = self._get_regime_indicator_adjustments(regime)
            out = {k: round(max(0.1, min(1.0, base.get(k, 0.5) + adj.get(k, 0.0))), 2) for k in base}
            logger.info(f"[{coin}] {regime} 지표 교차 비율: {out}")
            return out
        except Exception as e:
            logger.error(f"[{coin}] 지표 교차 비율 계산 실패: {e}")
            return {"rsi": 0.6, "macd": 0.6, "mfi": 0.6, "atr": 0.5, "adx": 0.6, "bb": 0.6}

    # ---- 비율 적용 분석
    def _analyze_fractal_patterns_with_ratios(
        self, coin: str, interval: str, candle_data: pd.DataFrame, fractal_ratios: Dict[str, float]
    ) -> float:
        """🔥 개선된 프랙탈 분석: 실제 프랙탈 패턴 검증"""
        try:
            if candle_data.empty or len(candle_data) < 20:
                return 0.5
            
            # 기본 인터벌 가중치
            w = float(fractal_ratios.get(interval, 0.5))
            
            # 1) 기본 변동성 점수
            base_score = self._calculate_basic_fractal_score(candle_data)
            
            # 2) 🔥 실제 프랙탈 패턴 검증 (상위/하위 프랙탈)
            fractal_pattern_score = self._detect_fractal_patterns(candle_data)
            
            # 3) 프랙탈 일관성 점수 (패턴이 명확한지)
            fractal_consistency = self._calculate_fractal_consistency(candle_data)
            
            # 종합 점수: 기본점수(40%) + 패턴점수(35%) + 일관성(25%)
            combined_score = (
                base_score * 0.4 +
                fractal_pattern_score * 0.35 +
                fractal_consistency * 0.25
            )
            
            # 인터벌 가중치 적용
            score = max(0.0, min(1.0, combined_score * w))
            logger.debug(f"[{coin}-{interval}] 프랙탈 점수: {score:.3f} (기본={base_score:.3f}, 패턴={fractal_pattern_score:.3f}, 일관성={fractal_consistency:.3f}, 가중치={w:.2f})")
            return score
        except Exception as e:
            logger.error(f"[{coin}] 프랙탈(비율) 분석 실패: {e}")
            return 0.5
    
    def _detect_fractal_patterns(self, candle_data: pd.DataFrame, period: int = 5) -> float:
        """프랙탈 패턴 감지 (상위 프랙탈/하위 프랙탈)"""
        try:
            if len(candle_data) < period * 2 + 1:
                return 0.5
            
            high = candle_data['high'].values if 'high' in candle_data.columns else candle_data['close'].values
            low = candle_data['low'].values if 'low' in candle_data.columns else candle_data['close'].values
            
            fractal_up_count = 0
            fractal_down_count = 0
            
            # 상위 프랙탈: 중앙값이 양쪽보다 높음
            # 하위 프랙탈: 중앙값이 양쪽보다 낮음
            for i in range(period, len(candle_data) - period):
                center_high = high[i]
                center_low = low[i]
                
                # 양쪽 period 개의 캔들 확인
                left_high = max(high[i-period:i]) if i >= period else center_high
                right_high = max(high[i+1:i+1+period]) if i + period < len(high) else center_high
                
                left_low = min(low[i-period:i]) if i >= period else center_low
                right_low = min(low[i+1:i+1+period]) if i + period < len(low) else center_low
                
                # 상위 프랙탈
                if center_high > left_high and center_high > right_high:
                    fractal_up_count += 1
                
                # 하위 프랙탈
                if center_low < left_low and center_low < right_low:
                    fractal_down_count += 1
            
            total_fractals = fractal_up_count + fractal_down_count
            if total_fractals == 0:
                return 0.5
            
            # 프랙탈 밀도 정규화 (0.0 ~ 1.0)
            max_possible = (len(candle_data) - period * 2) // 2
            density = min(1.0, total_fractals / max_possible if max_possible > 0 else 0.0)
            
            return density
        except Exception as e:
            logger.debug(f"프랙탈 패턴 감지 실패: {e}")
            return 0.5
    
    def _calculate_fractal_consistency(self, candle_data: pd.DataFrame) -> float:
        """프랙탈 패턴의 일관성 계산"""
        try:
            if len(candle_data) < 20:
                return 0.5
            
            # 변동성 패턴의 일관성 (표준편차의 안정성)
            changes = candle_data['close'].pct_change().dropna()
            if len(changes) < 10:
                return 0.5
            
            # 변동성의 변동성 (CV: Coefficient of Variation)
            rolling_std = changes.rolling(window=10).std()
            if rolling_std.std() == 0:
                return 0.5
            
            # 일관성: 변동성이 일정하면 높은 점수 (CV가 낮으면 일관성 높음)
            cv = rolling_std.std() / rolling_std.mean() if rolling_std.mean() > 0 else 1.0
            consistency = max(0.0, min(1.0, 1.0 - min(cv, 2.0) / 2.0))  # CV 0~2 → 점수 1.0~0.0
            
            return consistency
        except Exception as e:
            logger.debug(f"프랙탈 일관성 계산 실패: {e}")
            return 0.5

    def _analyze_multi_timeframe_with_ratios(
        self, coin: str, interval: str, candle_data: pd.DataFrame, ratios: Dict[str, float]
    ) -> float:
        """🔥 개선된 다중시간대 분석: 추세 일치도 및 강도 종합 평가"""
        try:
            if candle_data.empty or len(candle_data) < 20:
                return 0.5
            
            # 현재 인터벌의 가중치 가져오기
            w = float(ratios.get(interval, 0.5))
            
            # 1) 각 인터벌 내에서 추세 분석 (short/medium/long 추세를 종합)
            short_score = self._analyze_short_timeframe(candle_data)
            medium_score = self._analyze_medium_timeframe(candle_data)
            long_score = self._analyze_long_timeframe(candle_data)
            
            # 2) 🔥 추세 일치도 계산 (단기/중기/장기가 같은 방향인지)
            trend_alignment = self._calculate_trend_alignment(candle_data)
            
            # 3) 🔥 추세 강도 계산 (명확한 추세가 있는지)
            trend_strength = self._calculate_trend_strength(candle_data)
            
            # 추세별 가중 평균
            trend_score = (short_score * 0.3 + medium_score * 0.4 + long_score * 0.3)
            
            # 종합 점수: 추세점수(60%) + 일치도(25%) + 강도(15%)
            combined_score = (
                trend_score * 0.6 +
                trend_alignment * 0.25 +
                trend_strength * 0.15
            )
            
            # 인터벌 가중치 적용
            score = max(0.0, min(1.0, combined_score * w))
            logger.debug(f"[{coin}-{interval}] 다중시간대 점수: {score:.3f} (추세={trend_score:.3f}, 일치도={trend_alignment:.3f}, 강도={trend_strength:.3f}, 가중치={w:.2f})")
            return score
        except Exception as e:
            logger.error(f"[{coin}] 다중시간대(비율) 분석 실패: {e}")
            return 0.5
    
    def _calculate_trend_alignment(self, candle_data: pd.DataFrame) -> float:
        """단기/중기/장기 추세 일치도 계산"""
        try:
            if len(candle_data) < 50:
                return 0.5
            
            close = candle_data['close'].values
            
            # 단기 추세 (5캔들)
            short_trend = 1 if close[-1] > close[-5] else -1
            # 중기 추세 (20캔들)
            medium_trend = 1 if close[-1] > close[-20] else -1
            # 장기 추세 (50캔들)
            long_trend = 1 if close[-1] > close[-min(50, len(close)-1)] else -1
            
            # 모두 같은 방향이면 높은 점수
            alignment_count = sum([short_trend == medium_trend, medium_trend == long_trend, short_trend == long_trend])
            alignment_score = alignment_count / 3.0  # 0.0 ~ 1.0
            
            return alignment_score
        except Exception as e:
            logger.debug(f"추세 일치도 계산 실패: {e}")
            return 0.5
    
    def _calculate_trend_strength(self, candle_data: pd.DataFrame) -> float:
        """추세 강도 계산"""
        try:
            if len(candle_data) < 20:
                return 0.5
            
            # 이동평균 기울기 계산
            close = candle_data['close'].values
            ma_short = pd.Series(close).rolling(window=5).mean()
            ma_long = pd.Series(close).rolling(window=20).mean()
            
            if len(ma_short.dropna()) < 2 or len(ma_long.dropna()) < 2:
                return 0.5
            
            # 최근 기울기
            short_slope = (ma_short.iloc[-1] - ma_short.iloc[-2]) / ma_short.iloc[-2] if ma_short.iloc[-2] > 0 else 0
            long_slope = (ma_long.iloc[-1] - ma_long.iloc[-2]) / ma_long.iloc[-2] if ma_long.iloc[-2] > 0 else 0
            
            # 기울기의 절대값으로 강도 측정
            strength = (abs(short_slope) + abs(long_slope)) * 100  # % 변환
            normalized_strength = min(1.0, strength * 10)  # 0.1% 변화 → 1.0 점수
            
            return normalized_strength
        except Exception as e:
            logger.debug(f"추세 강도 계산 실패: {e}")
            return 0.5

    def _analyze_indicator_correlations_with_ratios(
        self, coin: str, interval: str, candle_data: pd.DataFrame, ratios: Dict[str, float]
    ) -> float:
        """🔥 개선된 지표 교차 분석: 다중 지표 신호 일치도 및 교차 신호 강도 평가"""
        try:
            if candle_data.empty:
                return 0.5
            
            parts: Dict[str, float] = {}
            indicator_scores: Dict[str, float] = {}
            
            # 각 지표별 패턴 분석
            if ratios.get("rsi", 0) > 0 and "rsi" in candle_data.columns:
                rsi_score = self._analyze_rsi_patterns(candle_data)
                indicator_scores["rsi"] = rsi_score
                parts["rsi"] = rsi_score * float(ratios["rsi"])
            
            if ratios.get("macd", 0) > 0 and "macd" in candle_data.columns:
                macd_score = self._analyze_macd_patterns(candle_data)
                indicator_scores["macd"] = macd_score
                parts["macd"] = macd_score * float(ratios["macd"])
            
            if ratios.get("mfi", 0) > 0 and "mfi" in candle_data.columns:
                mfi_score = self._analyze_mfi_patterns(candle_data)
                indicator_scores["mfi"] = mfi_score
                parts["mfi"] = mfi_score * float(ratios["mfi"])
            
            if ratios.get("atr", 0) > 0 and "atr" in candle_data.columns:
                atr_score = self._analyze_atr_patterns(candle_data)
                indicator_scores["atr"] = atr_score
                parts["atr"] = atr_score * float(ratios["atr"])
            
            if ratios.get("adx", 0) > 0 and "adx" in candle_data.columns:
                adx_score = self._analyze_adx_patterns(candle_data)
                indicator_scores["adx"] = adx_score
                parts["adx"] = adx_score * float(ratios["adx"])
            
            if ratios.get("bb", 0) > 0 and "bb_width" in candle_data.columns:
                bb_score = self._analyze_bb_patterns(candle_data)
                indicator_scores["bb"] = bb_score
                parts["bb"] = bb_score * float(ratios["bb"])

            if not parts:
                return 0.5
            
            # 1) 기본 가중 평균 점수
            wsum = sum(ratios.values())
            base_score = sum(parts.values()) / wsum if wsum > 0 else 0.5
            
            # 2) 🔥 지표 간 신호 일치도 계산
            signal_alignment = self._calculate_indicator_alignment(indicator_scores)
            
            # 3) 🔥 교차 신호 강도 계산 (여러 지표가 동시에 강한 신호를 주는지)
            cross_signal_strength = self._calculate_cross_signal_strength(indicator_scores)
            
            # 종합 점수: 기본점수(50%) + 일치도(30%) + 교차강도(20%)
            combined_score = (
                base_score * 0.5 +
                signal_alignment * 0.3 +
                cross_signal_strength * 0.2
            )
            
            score = max(0.0, min(1.0, combined_score))
            logger.debug(f"[{coin}-{interval}] 지표 교차 점수: {score:.3f} (기본={base_score:.3f}, 일치도={signal_alignment:.3f}, 교차강도={cross_signal_strength:.3f})")
            return score
        except Exception as e:
            logger.error(f"[{coin}] 지표 교차(비율) 분석 실패: {e}")
            return 0.5
    
    def _calculate_indicator_alignment(self, indicator_scores: Dict[str, float]) -> float:
        """지표 간 신호 일치도 계산"""
        try:
            if not indicator_scores or len(indicator_scores) < 2:
                return 0.5
            
            scores = list(indicator_scores.values())
            
            # 신호 강도 기준 (0.7 이상이면 강한 신호)
            strong_signals = [s for s in scores if s >= 0.7]
            weak_signals = [s for s in scores if s <= 0.3]
            
            # 강한 신호들이 많으면 일치도 높음
            alignment_score = len(strong_signals) / len(scores) if scores else 0.5
            
            # 또는 신호들이 비슷한 수준이면 일치도 높음 (표준편차가 낮으면)
            if len(scores) >= 2:
                import numpy as np
                std = np.std(scores)
                similarity = max(0.0, 1.0 - std * 2)  # 표준편차 0.5 이상이면 점수 낮음
                alignment_score = (alignment_score + similarity) / 2.0
            
            return alignment_score
        except Exception as e:
            logger.debug(f"지표 일치도 계산 실패: {e}")
            return 0.5
    
    def _calculate_cross_signal_strength(self, indicator_scores: Dict[str, float]) -> float:
        """교차 신호 강도 계산 (여러 지표가 동시에 강한 신호를 주는지)"""
        try:
            if not indicator_scores:
                return 0.5
            
            scores = list(indicator_scores.values())
            
            # 평균 점수가 높고, 최소값도 어느 정도 높으면 강한 교차 신호
            avg_score = sum(scores) / len(scores) if scores else 0.5
            min_score = min(scores) if scores else 0.5
            
            # 평균 70% 이상, 최소 50% 이상이면 강한 교차 신호
            if avg_score >= 0.7 and min_score >= 0.5:
                cross_strength = 0.9
            elif avg_score >= 0.6 and min_score >= 0.4:
                cross_strength = 0.7
            elif avg_score >= 0.5:
                cross_strength = 0.5
            else:
                cross_strength = 0.3
            
            return cross_strength
        except Exception as e:
            logger.debug(f"교차 신호 강도 계산 실패: {e}")
            return 0.5

    # ---- 각 지표별 간단 패턴 스코어
    def _analyze_short_timeframe(self, candle_data: pd.DataFrame) -> float:
        try:
            if len(candle_data) < 5:
                return 0.5
            recent = candle_data["close"].pct_change().iloc[-5:].dropna()
            return min(1.0, abs(float(recent.mean())) * 10.0) if len(recent) else 0.5
        except Exception:
            return 0.5

    def _analyze_medium_timeframe(self, candle_data: pd.DataFrame) -> float:
        try:
            if len(candle_data) < 20:
                return 0.5
            recent = candle_data["close"].iloc[-20:]
            trend = (float(recent.iloc[-1]) - float(recent.iloc[0])) / float(recent.iloc[0])
            return min(1.0, abs(trend) * 5.0)
        except Exception:
            return 0.5

    def _analyze_long_timeframe(self, candle_data: pd.DataFrame) -> float:
        try:
            if len(candle_data) < 50:
                return 0.5
            vol = float(candle_data["close"].pct_change().std())
            return min(1.0, vol * 20.0)
        except Exception:
            return 0.5

    def _analyze_rsi_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "rsi" not in candle_data.columns:
                return 0.5
            r = float(candle_data["rsi"].dropna().iloc[-1])
            if r > 70 or r < 30:
                return 0.8
            return 0.5
        except Exception:
            return 0.5

    def _analyze_macd_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "macd" not in candle_data.columns:
                return 0.5
            macd = candle_data["macd"].dropna()
            if len(macd) < 2:
                return 0.5
            recent, prev = float(macd.iloc[-1]), float(macd.iloc[-2])
            if recent > prev and recent > 0:
                return 0.8  # 매수 신호
            if recent < prev and recent < 0:
                return 0.8  # 매도 신호
            return 0.5
        except Exception:
            return 0.5

    def _analyze_bb_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "bb_width" not in candle_data.columns:
                return 0.5
            w = candle_data["bb_width"].dropna()
            if len(w) < 2:
                return 0.5
            recent = float(w.iloc[-1]); avg = float(w.mean())
            if recent > avg * 1.5:
                return 0.8  # 변동성 확장
            if recent < avg * 0.5:
                return 0.8  # 변동성 수축
            return 0.5
        except Exception:
            return 0.5

    def _analyze_mfi_patterns(self, candle_data: pd.DataFrame) -> float:
        """MFI (Money Flow Index) 패턴 분석"""
        try:
            if "mfi" not in candle_data.columns:
                return 0.5
            mfi = candle_data["mfi"].dropna()
            if len(mfi) < 2:
                return 0.5
            recent = float(mfi.iloc[-1])
            if recent < 20:
                return 0.8  # 과매도 신호
            if recent > 80:
                return 0.8  # 과매수 신호
            return 0.5
        except Exception:
            return 0.5

    def _analyze_atr_patterns(self, candle_data: pd.DataFrame) -> float:
        """ATR (Average True Range) 패턴 분석"""
        try:
            if "atr" not in candle_data.columns:
                return 0.5
            atr = candle_data["atr"].dropna()
            if len(atr) < 2:
                return 0.5
            recent = float(atr.iloc[-5:].mean())
            avg = float(atr.mean())
            if recent > avg * 1.5:
                return 0.8  # 높은 변동성
            if recent < avg * 0.5:
                return 0.8  # 낮은 변동성
            return 0.5
        except Exception:
            return 0.5

    def _analyze_adx_patterns(self, candle_data: pd.DataFrame) -> float:
        """ADX (Average Directional Index) 패턴 분석"""
        try:
            if "adx" not in candle_data.columns:
                return 0.5
            adx = candle_data["adx"].dropna()
            if len(adx) < 2:
                return 0.5
            recent = float(adx.iloc[-1])
            if recent > 40:
                return 0.8  # 강한 추세
            if recent < 20:
                return 0.8  # 약한 추세 (횡보)
            return 0.5
        except Exception:
            return 0.5

    # ---- 코인 특화 패턴 (간단 판별)
    def _analyze_coin_specific_patterns(self, coin: str, interval: str, candle_data: pd.DataFrame) -> float:
        try:
            # 하드코딩 심볼 제거: 모든 코인은 일반 패턴 분석 사용
            return float(self._analyze_generic_altcoin_patterns(candle_data))
        except Exception as e:
            logger.error(f"[{coin}] 코인 특화 패턴 분석 실패: {e}")
            return 0.5

    def _analyze_btc_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "rsi" in candle_data.columns:
                return min(1.0, abs(float(candle_data["rsi"].iloc[-1]) - 50.0) / 50.0)
            return 0.5
        except Exception:
            return 0.5

    def _analyze_eth_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "volume_ratio" in candle_data.columns:
                return min(1.0, float(candle_data["volume_ratio"].iloc[-1]) / 2.0)
            return 0.5
        except Exception:
            return 0.5

    def _analyze_bnb_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "macd" in candle_data.columns:
                return min(1.0, abs(float(candle_data["macd"].iloc[-1])) * 10.0)
            return 0.5
        except Exception:
            return 0.5

    def _analyze_ada_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "bb_width" in candle_data.columns:
                return min(1.0, float(candle_data["bb_width"].iloc[-1]) * 5.0)
            return 0.5
        except Exception:
            return 0.5

    def _analyze_sol_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "atr" in candle_data.columns:
                return min(1.0, float(candle_data["atr"].iloc[-1]) * 20.0)
            return 0.5
        except Exception:
            return 0.5

    def _analyze_generic_altcoin_patterns(self, candle_data: pd.DataFrame) -> float:
        try:
            if "close" in candle_data.columns and len(candle_data) > 1:
                prev = float(candle_data["close"].iloc[-2])
                curr = float(candle_data["close"].iloc[-1])
                return min(1.0, abs((curr - prev) / prev) * 10.0)
            return 0.5
        except Exception:
            return 0.5

    def _analyze_volatility_patterns(self, coin: str, interval: str, candle_data: pd.DataFrame) -> float:
        try:
            if candle_data.empty or "atr" not in candle_data.columns:
                return 0.5
            atr = candle_data["atr"].dropna()
            if len(atr) < 2:
                return 0.5
            recent = float(atr.iloc[-5:].mean())
            hist = float(atr.mean())
            ratio = (recent / hist) if hist > 0 else 1.0
            return min(1.0, ratio)
        except Exception as e:
            logger.error(f"[{coin}] 변동성 패턴 분석 실패: {e}")
            return 0.5

    def _analyze_volume_patterns(self, coin: str, interval: str, candle_data: pd.DataFrame) -> float:
        try:
            if candle_data.empty or "volume_ratio" not in candle_data.columns:
                return 0.5
            vr = candle_data["volume_ratio"].dropna()
            if len(vr) < 2:
                return 0.5
            recent = float(vr.iloc[-5:].mean()); hist = float(vr.mean())
            ratio = (recent / hist) if hist > 0 else 1.0
            return min(1.0, ratio)
        except Exception as e:
            logger.error(f"[{coin}] 거래량 패턴 분석 실패: {e}")
            return 0.5

    # ---- 최종 결합 (레짐 가중)
    def _calculate_final_signal_score_dynamic(
        self,
        analysis_results: Dict[str, float],
        analysis_modules: Dict[str, float],
        ensemble_result: Any,
        simulation_results: List[Dict[str, Any]],
        coin: str,
        interval: str,
        regime: str,
    ) -> float:
        try:
            if not analysis_modules or not analysis_results:
                return 0.5
            total_w = sum(analysis_modules.values()) or 1.0
            weighted = 0.0
            matched = 0
            for m, s in analysis_results.items():
                if m in analysis_modules:
                    weighted += float(s) * (analysis_modules[m] / total_w)
                    matched += 1
            if matched == 0:
                weighted = 0.5

            ensemble_score = (
                ensemble_result.ensemble_prediction if hasattr(ensemble_result, "ensemble_prediction")
                else (ensemble_result.get("ensemble_prediction", 0.5) if isinstance(ensemble_result, dict) else 0.5)
            )
            sim_score = 0.5
            if simulation_results:
                profits = [sim.get("total_return", 0.0) for sim in simulation_results]
                win_rates = [sim.get("win_rate", 0.0) for sim in simulation_results]
                if profits and win_rates:
                    sim_score = (sum(profits) / len(profits) + sum(win_rates) / len(win_rates)) / 2.0

            # 분석:0.6 / 앙상블:0.3 / 시뮬:0.1
            final_score = weighted * 0.6 + float(ensemble_score) * 0.3 + float(sim_score) * 0.1
            logger.info(f"[{coin}-{interval}] 최종(기본) 점수: {final_score:.3f}")
            return max(0.0, min(1.0, float(final_score)))
        except Exception as e:
            logger.error(f"[{coin}] 전략 최종 점수 계산 실패: {e}")
            return 0.5

    def _calculate_final_signal_score_with_regime(
        self,
        analysis_results: Dict[str, float],
        analysis_modules: Dict[str, float],
        ensemble_result: Any,
        simulation_results: List[Dict[str, Any]],
        coin: str,
        interval: str,
        regime: str,
        candle_data: pd.DataFrame,
    ) -> float:
        try:
            base = self._calculate_final_signal_score_dynamic(
                analysis_results, analysis_modules, ensemble_result, simulation_results, coin, interval, regime
            )
            mult = self._get_regime_signal_multiplier(regime, candle_data)
            trans_p = self._get_regime_transition_probability(candle_data)
            conf = self._get_regime_confidence(candle_data)

            score = base * mult
            if trans_p > 0.7:
                score *= 0.8
            if conf < 0.5:
                score *= 0.9
            logger.info(f"[{coin}-{interval}] 레짐 적용 점수: {score:.3f} (base {base:.3f}, mult {mult:.2f}, trans {trans_p:.2f}, conf {conf:.2f})")
            return max(0.0, min(1.0, float(score)))
        except Exception as e:
            logger.error(f"[{coin}] 레짐 최종 점수 계산 실패: {e}")
            return 0.5

    def _get_regime_signal_multiplier(self, regime: str, candle_data: pd.DataFrame) -> float:
        try:
            base = {
                "extreme_bullish": 1.3,
                "extreme_bearish": 1.3,
                "bullish": 1.1,
                "bearish": 1.1,
                "sideways_bullish": 0.9,
                "sideways_bearish": 0.9,
                "neutral": 0.8,
            }.get(regime, 1.0)
            if not candle_data.empty and "regime_confidence" in candle_data.columns:
                conf = float(candle_data["regime_confidence"].iloc[-1])
                base += 0.2 * (conf - 0.5)
            return max(0.5, min(1.5, float(base)))
        except Exception as e:
            logger.error(f"레짐 멀티플라이어 계산 실패: {e}")
            return 1.0

    def _get_regime_transition_probability(self, candle_data: pd.DataFrame) -> float:
        try:
            if candle_data.empty or "regime_transition_prob" not in candle_data.columns:
                return 0.5
            return float(candle_data["regime_transition_prob"].iloc[-1])
        except Exception as e:
            logger.error(f"레짐 전환확률 계산 실패: {e}")
            return 0.5

    def _get_regime_confidence(self, candle_data: pd.DataFrame) -> float:
        try:
            if candle_data.empty or "regime_confidence" not in candle_data.columns:
                return 0.5
            return float(candle_data["regime_confidence"].iloc[-1])
        except Exception as e:
            logger.error(f"레짐 신뢰도 계산 실패: {e}")
            return 0.5

    def _determine_signal_action(self, signal_score: float, regime: str, confidence: float = None) -> str:
        """
        신호 액션 결정 (예측 + 실행 신뢰도 고려)

        Args:
            signal_score: 예측 점수 (0~1)
            regime: 시장 레짐
            confidence: 실행 신뢰도 (None이면 score만으로 결정)
        """
        try:
            # 🔥 실행 신뢰도 임계값 (환경변수로 조정 가능)
            import os
            MIN_CONFIDENCE_FOR_TRADE = float(os.getenv('MIN_CONFIDENCE_FOR_TRADE', '0.65'))
            MIN_CONFIDENCE_FOR_STRONG_TRADE = float(os.getenv('MIN_CONFIDENCE_FOR_STRONG_TRADE', '0.75'))

            # 레짐별 임계값
            thr = {
                "extreme_bearish": {"buy": 0.3, "sell": 0.7},
                "bearish": {"buy": 0.4, "sell": 0.6},
                "sideways_bearish": {"buy": 0.45, "sell": 0.55},
                "neutral": {"buy": 0.5, "sell": 0.5},
                "sideways_bullish": {"buy": 0.55, "sell": 0.45},
                "bullish": {"buy": 0.6, "sell": 0.4},
                "extreme_bullish": {"buy": 0.7, "sell": 0.3},
            }.get(regime, {"buy": 0.5, "sell": 0.5})

            # 예측 결정
            if signal_score >= thr["buy"]:
                predicted_action = "BUY"
            elif signal_score <= thr["sell"]:
                predicted_action = "SELL"
            else:
                predicted_action = "HOLD"

            # 🔥 실행 신뢰도 기반 최종 결정
            if confidence is not None:
                if predicted_action != "HOLD":
                    # 거래 신호가 있을 때만 신뢰도 체크
                    if confidence < MIN_CONFIDENCE_FOR_TRADE:
                        # 신뢰도 부족 - 실행하지 않음
                        logger.debug(f"🚫 낮은 신뢰도({confidence:.2%}) - {predicted_action} → HOLD")
                        return "HOLD"
                    elif confidence < MIN_CONFIDENCE_FOR_STRONG_TRADE:
                        # 중간 신뢰도 - 약한 신호만 허용
                        signal_strength = abs(signal_score - 0.5)
                        if signal_strength < 0.15:  # 약한 신호
                            logger.debug(f"⚠️ 중간 신뢰도({confidence:.2%}) + 약한 신호 - {predicted_action} → HOLD")
                            return "HOLD"

            return predicted_action

        except Exception as e:
            logger.error(f"❌ 신호 액션 결정 실패: {e}")
            return "HOLD"

    def _calculate_signal_confidence(self, ensemble_confidence: float, simulation_results: List[Dict[str, Any]]) -> float:
        try:
            ens = float(ensemble_confidence) if ensemble_confidence is not None else 0.5
            sim_conf = 0.5
            if simulation_results:
                profits = [sim.get("profit", 0.0) for sim in simulation_results]
                if profits:
                    var = float(np.var(profits))
                    sim_conf = max(0.0, 1.0 - var)
            return max(0.0, min(1.0, (ens + sim_conf) / 2.0))
        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    def _calculate_strategy_confidence(self, strategies: List[Dict[str, Any]], candle_data: pd.DataFrame) -> Optional[float]:
        """전략 등급별 조건 만족률로 신뢰도 계산 (안정성 우선)"""
        try:
            if not strategies or len(strategies) < 5 or candle_data.empty:
                return None
            
            # 높은 등급 전략과 낮은 등급 전략 분리
            high_grade_strategies = [s for s in strategies if s.get('grade') == 'high']
            low_grade_strategies = [s for s in strategies if s.get('grade') == 'low']
            
            if len(high_grade_strategies) < 3 or len(low_grade_strategies) < 2:
                return None
            
            # 🔥 실제 전략 조건 만족 여부 체크
            high_satisfied = 0
            for strategy in high_grade_strategies:
                # 전략이 현재 시장 조건에서 매수 신호를 주는지 판단
                if self._check_strategy_condition(strategy, candle_data):
                    high_satisfied += 1
            
            low_satisfied = 0
            for strategy in low_grade_strategies:
                # 낮은 등급 전략도 조건 체크 (반대 신호)
                if self._check_strategy_condition(strategy, candle_data):
                    low_satisfied += 1
            
            # 만족률 계산
            high_satisfaction_rate = high_satisfied / len(high_grade_strategies) if high_grade_strategies else 0
            low_satisfaction_rate = low_satisfied / len(low_grade_strategies) if low_grade_strategies else 0
            
            # 신뢰도 = 높은등급 만족률 - 낮은등급 만족률
            # 높은등급이 많이 만족하고, 낮은등급이 적게 만족하면 신뢰도 ↑
            confidence = high_satisfaction_rate - low_satisfaction_rate
            
            logger.info(f"📊 신뢰도 계산: 높은등급 {high_satisfied}/{len(high_grade_strategies)}={high_satisfaction_rate:.1%}, "
                       f"낮은등급 {low_satisfied}/{len(low_grade_strategies)}={low_satisfaction_rate:.1%}, "
                       f"신뢰도={confidence:.1%}")
            
            return confidence
            
        except Exception as e:
            logger.error(f"전략 신뢰도 계산 실패: {e}")
            return None
    
    def _check_strategy_condition(self, strategy: Dict[str, Any], candle_data: pd.DataFrame) -> bool:
        """전략이 현재 시장 조건에서 매수 조건을 만족하는지 체크"""
        try:
            # 전략 파라미터 가져오기
            params = {}
            if 'params' in strategy:
                params_raw = strategy.get('params')
                if isinstance(params_raw, str):
                    params = json.loads(params_raw)
                elif isinstance(params_raw, dict):
                    params = params_raw
            else:
                # 필드에서 직접 가져오기
                for key in ['rsi_min', 'rsi_max', 'macd_threshold', 'volume_ratio_min']:
                    if key in strategy:
                        params[key] = strategy[key]
            
            if not params:
                return False
            
            # 현재 시장 지표 가져오기
            conditions_met = 0
            total_conditions = 0
            
            # RSI 조건 체크
            if 'rsi_min' in params or 'rsi_max' in params:
                if 'rsi' in candle_data.columns:
                    current_rsi = float(candle_data['rsi'].iloc[-1])
                    rsi_min = params.get('rsi_min', 0)
                    rsi_max = params.get('rsi_max', 100)
                    
                    if rsi_min <= current_rsi <= rsi_max:
                        conditions_met += 1
                    total_conditions += 1
            
            # MACD 조건 체크
            if 'macd_threshold' in params:
                if 'macd' in candle_data.columns:
                    current_macd = float(candle_data['macd'].iloc[-1])
                    threshold = params.get('macd_threshold', 0)
                    
                    if current_macd > threshold:
                        conditions_met += 1
                    total_conditions += 1
            
            # Volume Ratio 조건 체크
            if 'volume_ratio_min' in params:
                if 'volume_ratio' in candle_data.columns:
                    current_volume = float(candle_data['volume_ratio'].iloc[-1])
                    volume_min = params.get('volume_ratio_min', 0)
                    
                    if current_volume >= volume_min:
                        conditions_met += 1
                    total_conditions += 1
            
            # 대부분의 조건을 만족하면 True
            if total_conditions == 0:
                return False
            
            satisfaction_rate = conditions_met / total_conditions
            return satisfaction_rate >= 0.6  # 60% 이상 만족하면 OK
            
        except Exception as e:
            logger.debug(f"전략 조건 체크 실패: {e}")
            return False
    
    def _quick_strategy_score(self, strategy_params: Dict[str, Any]) -> float:
        """전략 파라미터로 간단한 점수 계산 (레거시)"""
        try:
            # 기본 점수
            score = 0.5
            
            # 전략 성과를 기반으로 조정
            profit = strategy_params.get('profit', 0)
            win_rate = strategy_params.get('win_rate', 0.5)
            
            # 좋은 전략이면 점수 상승
            if profit > 0 and win_rate > 0.5:
                score = 0.5 + (win_rate - 0.5) * 0.5 + min(profit / 10000, 0.2)
            elif profit < 0 or win_rate < 0.5:
                score = 0.5 - (0.5 - win_rate) * 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5

    # ---- 집계/랭킹
    def _get_all_combinations(self, coin_results: List[CoinSignalScore]) -> List[Dict[str, Any]]:
        try:
            combos: List[Dict[str, Any]] = []
            for r in coin_results or []:
                combos.append(
                    {
                        "coin": r.coin,
                        "interval": r.interval,
                        "regime": r.regime,
                        "signal_score": r.final_signal_score,
                        "signal_action": r.signal_action,
                        "confidence": r.signal_confidence,
                        "fractal_score": r.fractal_score,
                        "multi_timeframe_score": r.multi_timeframe_score,
                        "indicator_cross_score": r.indicator_cross_score,
                        "ensemble_score": r.ensemble_score,
                    }
                )
            return combos
        except Exception as e:
            logger.error(f"조합 생성 실패: {e}")
            return []

    def _get_top_performers(self, combinations: List[Dict[str, Any]]) -> List[str]:
        try:
            if not combinations:
                return []
            ranked = sorted(combinations, key=lambda x: x.get("signal_score", 0.0), reverse=True)
            return [
                f"{c.get('coin','Unknown')}-{c.get('interval','Unknown')}-{c.get('regime','Unknown')}"
                for c in ranked[:5]
            ]
        except Exception as e:
            logger.error(f"최고 성능 추출 실패: {e}")
            return []

    def _get_top_coins(self, coin_results: List[CoinSignalScore]) -> List[str]:
        try:
            if not coin_results:
                return []
            bucket: Dict[str, List[float]] = {}
            for r in coin_results:
                bucket.setdefault(r.coin, []).append(r.final_signal_score)
            avg = {k: sum(v) / len(v) for k, v in bucket.items()}
            return [k for k, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:3]]
        except Exception as e:
            logger.error(f"Top 코인 추출 실패: {e}")
            return []

    def _get_top_intervals(self, coin_results: List[CoinSignalScore]) -> List[str]:
        try:
            if not coin_results:
                return []
            bucket: Dict[str, List[float]] = {}
            for r in coin_results:
                bucket.setdefault(r.interval, []).append(r.final_signal_score)
            avg = {k: sum(v) / len(v) for k, v in bucket.items()}
            return [k for k, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:3]]
        except Exception as e:
            logger.error(f"Top 인터벌 추출 실패: {e}")
            return []

    # ---- 기본 객체 생성자
    def _create_default_coin_signal_score(self, coin: str, interval: str, regime: str) -> CoinSignalScore:
        return CoinSignalScore(
            coin=coin,
            interval=interval,
            regime=regime,
            fractal_score=0.5,
            multi_timeframe_score=0.5,
            indicator_cross_score=0.5,
            ensemble_score=0.5,
            ensemble_confidence=0.5,
            final_signal_score=0.5,
            signal_action="HOLD",
            signal_confidence=0.5,
            created_at=datetime.now().isoformat(),
        )

    def _create_default_ensemble_result(self) -> Any:
        class _Default:
            def __init__(self):
                self.ensemble_prediction = 0.5
                self.confidence_score = 0.5
        return _Default()

    def _create_default_global_signal_score(self) -> GlobalSignalScore:
        return GlobalSignalScore(
            overall_score=0.5,
            overall_confidence=0.5,
            policy_improvement=0.0,
            convergence_rate=0.0,
            top_performers=[],
            top_coins=[],
            top_intervals=[],
            created_at=datetime.now().isoformat(),
        )

# ---------------------------------------------------------------------
# 외부 노출 편의 함수
# ---------------------------------------------------------------------
def analyze_coin_strategies(
    coin: str, interval: str, regime: str, strategies: List[Dict[str, Any]], candle_data: pd.DataFrame
) -> CoinSignalScore:
    analyzer = IntegratedAnalyzer()
    return analyzer.analyze_coin_strategies(coin, interval, regime, strategies, candle_data)

def analyze_multi_interval_strategies(
    coin: str, regime: str, strategies: List[Dict[str, Any]], multi_interval_candle_data: Dict[str, pd.DataFrame]
) -> CoinSignalScore:
    """🔥 여러 인터벌의 전략 점수를 종합하여 최종 시그널 점수 계산 (외부 노출 함수)"""
    analyzer = IntegratedAnalyzer()
    return analyzer.analyze_multi_interval_strategies(coin, regime, strategies, multi_interval_candle_data)

def analyze_global_strategies(
    global_strategies: List[Dict[str, Any]], all_coin_results: List[CoinSignalScore]
) -> GlobalSignalScore:
    analyzer = IntegratedAnalyzer()
    return analyzer.analyze_global_strategies(global_strategies, all_coin_results)

def calculate_signal_scores(coin_results: List[CoinSignalScore]) -> Dict[str, Any]:
    try:
        if not coin_results:
            return {}
        total = len(coin_results)
        avg_score = sum(r.final_signal_score for r in coin_results) / total
        avg_conf = sum(r.signal_confidence for r in coin_results) / total
        action_counts: Dict[str, int] = {}
        for r in coin_results:
            action_counts[r.signal_action] = action_counts.get(r.signal_action, 0) + 1
        return {
            "total_count": total,
            "avg_score": float(avg_score),
            "avg_confidence": float(avg_conf),
            "action_distribution": action_counts,
        }
    except Exception as e:
        logger.error(f"신호 점수 집계 실패: {e}")
        return {}

def generate_final_recommendations(
    coin_results: List[CoinSignalScore], global_result: GlobalSignalScore
) -> Dict[str, Any]:
    try:
        rec: Dict[str, Any] = {
            "summary": {
                "total_coins_analyzed": len(coin_results),
                "overall_score": global_result.overall_score,
                "overall_confidence": global_result.overall_confidence,
                "top_coins": global_result.top_coins,
                "top_intervals": global_result.top_intervals,
            },
            "coin_recommendations": [],
            "global_recommendations": [],
            "created_at": datetime.now().isoformat(),
        }
        for r in coin_results:
            if r.final_signal_score > 0.6:
                rec["coin_recommendations"].append(
                    {
                        "coin": r.coin,
                        "interval": r.interval,
                        "regime": r.regime,
                        "action": r.signal_action,
                        "score": r.final_signal_score,
                        "confidence": r.signal_confidence,
                    }
                )
        if global_result.overall_score > 0.6:
            rec["global_recommendations"].append(
                {
                    "type": "positive_market",
                    "message": f"전체 시장이 긍정적입니다 (점수: {global_result.overall_score:.3f})",
                    "confidence": global_result.overall_confidence,
                }
            )
        elif global_result.overall_score < 0.4:
            rec["global_recommendations"].append(
                {
                    "type": "negative_market",
                    "message": f"전체 시장이 신중해야 합니다 (점수: {global_result.overall_score:.3f})",
                    "confidence": global_result.overall_confidence,
                }
            )
        else:
            rec["global_recommendations"].append(
                {
                    "type": "neutral_market",
                    "message": f"전체 시장이 중립적입니다 (점수: {global_result.overall_score:.3f})",
                    "confidence": global_result.overall_confidence,
                }
            )
        return rec
    except Exception as e:
        logger.error(f"최종 추천 생성 실패: {e}")
        return {
            "summary": {
                "total_coins_analyzed": len(coin_results) if coin_results else 0,
                "overall_score": 0.5,
                "overall_confidence": 0.5,
                "top_coins": [],
                "top_intervals": [],
            },
            "coin_recommendations": [],
            "global_recommendations": [],
            "created_at": datetime.now().isoformat(),
        }