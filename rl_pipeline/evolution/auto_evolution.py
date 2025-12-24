"""
🧬 Auto Evolution System - 자동 진화 시스템

종목별로 정확도에 따라 자동으로 Phase를 전환하는 시스템:
- Phase 1: 통계 기반 (MFE/MAE EntryScore) - 기본
- Phase 2: MFE/MAE 예측 모델 (XGBoost/LightGBM) - 데이터 충분 시
- Phase 3: 타이밍 최적화 (RL Agent) - 고정확도 달성 시
"""

import os
import sys
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .phase_manager import PhaseManager, Phase, get_phase_manager
from .accuracy_tracker import AccuracyTracker, get_accuracy_tracker

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """시그널 계산 결과"""
    direction: str           # 'buy', 'sell', 'hold'
    score: float             # 0.0 ~ 1.0
    confidence: float        # 신뢰도
    entry_score: float       # MFE/MAE 기반 진입 점수
    phase: Phase             # 사용된 Phase
    predicted_mfe: float     # 예측 MFE
    predicted_mae: float     # 예측 MAE
    metadata: Dict[str, Any] # 추가 정보


class AutoEvolutionSystem:
    """
    🧬 자동 진화 시스템
    
    종목별로 최적의 Phase를 자동으로 선택하여 시그널을 계산합니다.
    
    사용법:
    ```python
    evolution = AutoEvolutionSystem()
    
    # 시그널 계산 (Phase 자동 선택)
    result = evolution.calculate_signal(
        coin="BTC",
        interval="15m",
        candle_data=df,
        strategy=strategy_dict
    )
    
    # 결과 확인
    print(f"Phase: {result.phase.name}")
    print(f"Direction: {result.direction}")
    print(f"Score: {result.score}")
    ```
    """
    
    def __init__(
        self,
        phase_manager: Optional[PhaseManager] = None,
        accuracy_tracker: Optional[AccuracyTracker] = None
    ):
        """
        Args:
            phase_manager: Phase 관리자 (None이면 싱글톤 사용)
            accuracy_tracker: 정확도 추적기 (None이면 싱글톤 사용)
        """
        self.phase_manager = phase_manager or get_phase_manager()
        self.accuracy_tracker = accuracy_tracker or get_accuracy_tracker()
        
        # Phase별 계산기 로드 (지연 로딩)
        self._phase_calculators = {}
    
    def calculate_signal(
        self,
        coin: str,
        interval: str,
        candle_data: Any,  # DataFrame
        strategy: Dict[str, Any],
        force_phase: Optional[Phase] = None
    ) -> SignalResult:
        """
        시그널 계산 (Phase 자동 선택)
        
        Args:
            coin: 코인명
            interval: 인터벌
            candle_data: 캔들 데이터 (DataFrame)
            strategy: 전략 딕셔너리
            force_phase: 강제 Phase 지정 (테스트용)
            
        Returns:
            SignalResult: 시그널 계산 결과
        """
        # Phase 결정
        if force_phase is not None:
            current_phase = force_phase
        else:
            current_phase = self.phase_manager.get_phase(coin, interval)
        
        # Phase별 시그널 계산
        if current_phase == Phase.STATISTICAL:
            result = self._calculate_phase1_statistical(
                coin, interval, candle_data, strategy
            )
        elif current_phase == Phase.PREDICTIVE:
            result = self._calculate_phase2_predictive(
                coin, interval, candle_data, strategy
            )
        elif current_phase == Phase.TIMING_OPTIMIZED:
            result = self._calculate_phase3_timing(
                coin, interval, candle_data, strategy
            )
        else:
            # 기본: Phase 1 사용
            result = self._calculate_phase1_statistical(
                coin, interval, candle_data, strategy
            )
        
        # Phase 정보 추가
        result.phase = current_phase
        result.metadata["calculated_at"] = datetime.now().isoformat()
        
        return result
    
    def _calculate_phase1_statistical(
        self,
        coin: str,
        interval: str,
        candle_data: Any,
        strategy: Dict[str, Any]
    ) -> SignalResult:
        """
        Phase 1: 통계 기반 시그널 계산
        
        MFE/MAE EntryScore를 사용한 기본 계산
        """
        try:
            # 전략에서 MFE/MAE 통계 추출
            entry_score = strategy.get('entry_score', 0.0)
            risk_score = strategy.get('risk_score', 0.0)
            mfe_mean = strategy.get('mfe_mean', 0.0)
            mae_mean = strategy.get('mae_mean', 0.0)
            
            # 방향 결정
            strategy_type = strategy.get('strategy_type', '')
            if 'buy' in strategy_type.lower():
                direction = 'buy'
            elif 'sell' in strategy_type.lower():
                direction = 'sell'
            else:
                # EntryScore 기반 방향 결정
                if entry_score > 0.01:
                    direction = 'buy'
                elif entry_score < -0.01:
                    direction = 'sell'
                else:
                    direction = 'hold'
            
            # 점수 계산 (0 ~ 1 정규화)
            # EntryScore가 -0.05 ~ 0.05 범위라고 가정
            score = (entry_score + 0.05) / 0.10
            score = max(0.0, min(1.0, score))
            
            # 신뢰도: 데이터 양과 일관성 기반
            n_signals = strategy.get('n_signals', 0)
            confidence = min(1.0, n_signals / 100.0)
            
            return SignalResult(
                direction=direction,
                score=score,
                confidence=confidence,
                entry_score=entry_score,
                phase=Phase.STATISTICAL,
                predicted_mfe=mfe_mean,
                predicted_mae=mae_mean,
                metadata={
                    "method": "statistical",
                    "risk_score": risk_score,
                    "n_signals": n_signals
                }
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 1 계산 실패: {e}")
            return self._fallback_result()
    
    def _calculate_phase2_predictive(
        self,
        coin: str,
        interval: str,
        candle_data: Any,
        strategy: Dict[str, Any]
    ) -> SignalResult:
        """
        Phase 2: 예측 모델 기반 시그널 계산
        
        XGBoost/LightGBM으로 MFE/MAE 예측
        """
        try:
            # 예측 모델 로드 (없으면 Phase 1로 폴백)
            model = self._get_predictive_model(coin, interval)
            
            if model is None:
                logger.debug(f"⚠️ {coin}/{interval} 예측 모델 없음, Phase 1 폴백")
                return self._calculate_phase1_statistical(
                    coin, interval, candle_data, strategy
                )
            
            # 특성 추출
            features = self._extract_features(candle_data, strategy)
            
            # MFE/MAE 예측
            predicted_mfe, predicted_mae = model.predict(features)
            
            # 예측 EntryScore 계산
            # EntryScore = P90(MFE) - k * |P10(MAE)|
            k = 1.5  # 리스크 회피 계수
            predicted_entry_score = predicted_mfe - k * abs(predicted_mae)
            
            # 방향 결정
            if predicted_entry_score > 0.01:
                direction = 'buy'
            elif predicted_entry_score < -0.01:
                direction = 'sell'
            else:
                direction = 'hold'
            
            # 점수 계산
            score = (predicted_entry_score + 0.05) / 0.10
            score = max(0.0, min(1.0, score))
            
            # 신뢰도: 모델 예측 신뢰도 + 데이터 품질
            model_confidence = model.get_confidence() if hasattr(model, 'get_confidence') else 0.7
            n_signals = strategy.get('n_signals', 0)
            data_confidence = min(1.0, n_signals / 100.0)
            confidence = (model_confidence + data_confidence) / 2
            
            return SignalResult(
                direction=direction,
                score=score,
                confidence=confidence,
                entry_score=predicted_entry_score,
                phase=Phase.PREDICTIVE,
                predicted_mfe=predicted_mfe,
                predicted_mae=predicted_mae,
                metadata={
                    "method": "predictive",
                    "model_type": type(model).__name__,
                    "model_confidence": model_confidence
                }
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 2 계산 실패, Phase 1 폴백: {e}")
            return self._calculate_phase1_statistical(
                coin, interval, candle_data, strategy
            )
    
    def _calculate_phase3_timing(
        self,
        coin: str,
        interval: str,
        candle_data: Any,
        strategy: Dict[str, Any]
    ) -> SignalResult:
        """
        Phase 3: 타이밍 최적화 기반 시그널 계산
        
        RL Agent로 진입/청산 타이밍 최적화
        """
        try:
            # RL 에이전트 로드 (없으면 Phase 2로 폴백)
            agent = self._get_rl_agent(coin, interval)
            
            if agent is None:
                logger.debug(f"⚠️ {coin}/{interval} RL 에이전트 없음, Phase 2 폴백")
                return self._calculate_phase2_predictive(
                    coin, interval, candle_data, strategy
                )
            
            # 상태 구성
            state = self._construct_state(candle_data, strategy)
            
            # 에이전트 행동 결정
            action, action_prob = agent.select_action(state)
            
            # 행동 해석
            if action == 0:
                direction = 'hold'
            elif action == 1:
                direction = 'buy'
            elif action == 2:
                direction = 'sell'
            else:
                direction = 'hold'
            
            # MFE/MAE 예측 (Phase 2 모델 활용)
            model = self._get_predictive_model(coin, interval)
            if model:
                features = self._extract_features(candle_data, strategy)
                predicted_mfe, predicted_mae = model.predict(features)
            else:
                # 전략 통계 사용
                predicted_mfe = strategy.get('mfe_mean', 0.0)
                predicted_mae = strategy.get('mae_mean', 0.0)
            
            # 점수: 에이전트 확신도 기반
            score = float(action_prob)
            
            # EntryScore 계산
            k = 1.5
            entry_score = predicted_mfe - k * abs(predicted_mae)
            
            # 신뢰도: 에이전트 학습 상태 기반
            agent_confidence = agent.get_confidence() if hasattr(agent, 'get_confidence') else 0.8
            
            return SignalResult(
                direction=direction,
                score=score,
                confidence=agent_confidence,
                entry_score=entry_score,
                phase=Phase.TIMING_OPTIMIZED,
                predicted_mfe=predicted_mfe,
                predicted_mae=predicted_mae,
                metadata={
                    "method": "timing_optimized",
                    "action": action,
                    "action_prob": action_prob,
                    "agent_type": type(agent).__name__
                }
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 3 계산 실패, Phase 2 폴백: {e}")
            return self._calculate_phase2_predictive(
                coin, interval, candle_data, strategy
            )
    
    def _get_predictive_model(self, coin: str, interval: str) -> Optional[Any]:
        """예측 모델 로드 (XGBoost/LightGBM)"""
        try:
            # 캐시 확인
            key = f"pred_{coin}_{interval}"
            if key in self._phase_calculators:
                return self._phase_calculators[key]
            
            # 🔥 모델 파일 경로 (엔진화 - 환경변수 우선)
            data_storage_path = os.getenv('DATA_STORAGE_PATH')
            if not data_storage_path:
                # 환경변수 없으면 STRATEGY_DB_PATH에서 추론
                from rl_pipeline.core.env import config
                strategy_db = config.STRATEGIES_DB or os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
                if strategy_db:
                    data_storage_path = os.path.dirname(strategy_db)
                else:
                    # 최종 fallback: 현재 작업 디렉토리
                    data_storage_path = os.path.join(os.getcwd(), 'data_storage')
            
            model_dir = os.path.join(data_storage_path, 'models', 'predictive')
            model_path = os.path.join(model_dir, f'{coin}_{interval}_mfe_mae.pkl')
            
            if not os.path.exists(model_path):
                return None
            
            # 모델 로드
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self._phase_calculators[key] = model
            return model
            
        except Exception as e:
            logger.debug(f"⚠️ 예측 모델 로드 실패: {e}")
            return None
    
    def _get_rl_agent(self, coin: str, interval: str) -> Optional[Any]:
        """RL 에이전트 로드"""
        try:
            # 캐시 확인
            key = f"rl_{coin}_{interval}"
            if key in self._phase_calculators:
                return self._phase_calculators[key]
            
            # 🔥 에이전트 파일 경로 (엔진화 - 환경변수 우선)
            data_storage_path = os.getenv('DATA_STORAGE_PATH')
            if not data_storage_path:
                # 환경변수 없으면 STRATEGY_DB_PATH에서 추론
                from rl_pipeline.core.env import config
                strategy_db = config.STRATEGIES_DB or os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
                if strategy_db:
                    data_storage_path = os.path.dirname(strategy_db)
                else:
                    # 최종 fallback: 현재 작업 디렉토리
                    data_storage_path = os.path.join(os.getcwd(), 'data_storage')
            
            agent_dir = os.path.join(data_storage_path, 'models', 'rl_agents')
            agent_path = os.path.join(agent_dir, f'{coin}_{interval}_timing_agent.pkl')
            
            if not os.path.exists(agent_path):
                return None
            
            # 에이전트 로드
            import pickle
            with open(agent_path, 'rb') as f:
                agent = pickle.load(f)
            
            self._phase_calculators[key] = agent
            return agent
            
        except Exception as e:
            logger.debug(f"⚠️ RL 에이전트 로드 실패: {e}")
            return None
    
    def _extract_features(self, candle_data: Any, strategy: Dict) -> Any:
        """캔들 데이터에서 특성 추출"""
        import numpy as np
        
        try:
            # 기본 특성
            features = []
            
            # 캔들 데이터가 DataFrame인 경우
            if hasattr(candle_data, 'iloc'):
                latest = candle_data.iloc[-1]
                
                # 가격 특성
                features.append(latest.get('close', 0))
                features.append(latest.get('high', 0))
                features.append(latest.get('low', 0))
                features.append(latest.get('volume', 0))
                
                # 기술 지표
                features.append(latest.get('rsi', 50))
                features.append(latest.get('macd', 0))
                features.append(latest.get('bb_upper', 0))
                features.append(latest.get('bb_lower', 0))
            
            # 전략 특성
            features.append(strategy.get('rsi_min', 30))
            features.append(strategy.get('rsi_max', 70))
            features.append(strategy.get('entry_score', 0))
            features.append(strategy.get('risk_score', 0))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"⚠️ 특성 추출 실패: {e}")
            return np.zeros((1, 12))
    
    def _construct_state(self, candle_data: Any, strategy: Dict) -> Any:
        """RL 에이전트용 상태 구성"""
        import numpy as np
        
        try:
            # 특성 추출 후 상태 벡터로 변환
            features = self._extract_features(candle_data, strategy)
            
            # 추가 상태 정보
            position_state = 0  # 현재 포지션 (0: 없음, 1: 롱, 2: 숏)
            
            state = np.concatenate([features.flatten(), [position_state]])
            return state
            
        except Exception as e:
            logger.warning(f"⚠️ 상태 구성 실패: {e}")
            return np.zeros(13)
    
    def _fallback_result(self) -> SignalResult:
        """폴백 결과 (계산 실패 시)"""
        return SignalResult(
            direction='hold',
            score=0.5,
            confidence=0.0,
            entry_score=0.0,
            phase=Phase.STATISTICAL,
            predicted_mfe=0.0,
            predicted_mae=0.0,
            metadata={"method": "fallback", "reason": "calculation_failed"}
        )
    
    def record_result(
        self,
        coin: str,
        interval: str,
        prediction_id: str,
        result: SignalResult,
        actual_direction: Optional[str] = None,
        actual_mfe: Optional[float] = None,
        actual_mae: Optional[float] = None,
        actual_pnl: Optional[float] = None
    ) -> None:
        """
        예측 결과 기록 및 평가
        
        실제 결과가 주어지면 정확도 측정에 반영됩니다.
        """
        # 예측 기록
        self.accuracy_tracker.record_prediction(
            prediction_id=prediction_id,
            coin=coin,
            interval=interval,
            phase=int(result.phase),
            predicted_direction=result.direction,
            predicted_mfe=result.predicted_mfe,
            predicted_mae=result.predicted_mae,
            entry_score=result.entry_score,
            confidence=result.confidence,
            metadata=result.metadata
        )
        
        # 실제 결과가 있으면 평가
        if actual_direction is not None:
            self.accuracy_tracker.update_actual_result(
                prediction_id=prediction_id,
                actual_direction=actual_direction,
                actual_mfe=actual_mfe or 0.0,
                actual_mae=actual_mae or 0.0,
                actual_pnl=actual_pnl or 0.0
            )
            
            # Phase 관리자에 결과 기록
            self.phase_manager.record_prediction(
                coin=coin,
                interval=interval,
                predicted_direction=result.direction,
                actual_direction=actual_direction,
                confidence=result.confidence
            )
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """전체 진화 현황 요약"""
        return self.phase_manager.get_summary()
    
    def get_symbol_status(self, coin: str, interval: str) -> Dict[str, Any]:
        """종목별 상태 조회"""
        state = self.phase_manager.get_state(coin, interval)
        accuracy_stats = self.accuracy_tracker.get_accuracy_stats(coin, interval)
        phase_comparison = self.accuracy_tracker.get_phase_comparison(coin, interval)
        
        return {
            "coin": coin,
            "interval": interval,
            "current_phase": state.current_phase.name,
            "current_accuracy": state.current_accuracy,
            "recent_accuracy": state.recent_accuracy,
            "total_predictions": state.total_predictions,
            "consecutive_fails": state.consecutive_fails,
            "last_promotion": state.last_promotion.isoformat() if state.last_promotion else None,
            "last_demotion": state.last_demotion.isoformat() if state.last_demotion else None,
            "accuracy_stats": accuracy_stats,
            "phase_comparison": phase_comparison
        }


# 싱글톤 인스턴스
_auto_evolution: Optional[AutoEvolutionSystem] = None


def get_auto_evolution() -> AutoEvolutionSystem:
    """AutoEvolutionSystem 싱글톤 인스턴스 반환"""
    global _auto_evolution
    if _auto_evolution is None:
        _auto_evolution = AutoEvolutionSystem()
    return _auto_evolution


def run_evolution_check(coins: list = None, intervals: list = None) -> Dict[str, Any]:
    """
    모든 종목의 Phase 상태 체크 및 업데이트
    
    absolute_zero_system.py에서 호출됩니다.
    
    Args:
        coins: 체크할 코인 리스트 (None이면 전체)
        intervals: 체크할 인터벌 리스트 (None이면 전체)
        
    Returns:
        진화 상태 요약
    """
    evolution = get_auto_evolution()
    
    # 오래된 기록 정리
    evolution.accuracy_tracker.cleanup_old_records(days=90)
    
    # 현황 요약 반환
    summary = evolution.get_evolution_summary()
    
    logger.info(f"🧬 진화 시스템 현황:")
    logger.info(f"   총 종목: {summary['total_symbols']}개")
    logger.info(f"   Phase 분포: {summary['distribution']}")
    logger.info(f"   Phase별 평균 정확도: {summary['avg_accuracies']}")
    
    if summary.get('top_performers'):
        logger.info(f"   🏆 상위 성과:")
        for perf in summary['top_performers'][:3]:
            logger.info(f"      {perf['symbol']}: Phase {perf['phase']}, 정확도 {perf['accuracy']:.1%}")
    
    return summary

