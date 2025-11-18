"""
하이브리드 정책 에이전트
규칙 기반 + 신경망 기반 통합
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from rl_pipeline.simulation.agent import StrategyAgent
from rl_pipeline.simulation.market_models import MarketState, Action

logger = logging.getLogger(__name__)


class HybridPolicyAgent(StrategyAgent):
    """
    하이브리드 정책 에이전트
    
    규칙 기반 정책과 신경망 정책을 결합하여 의사결정
    - 명확한 신호: 규칙 기반 즉시 결정
    - 애매한 구간: 신경망 정책 사용
    - 폴백: 신경망 부재/실패 시 규칙 기반
    """
    
    def __init__(
        self,
        agent_id: str,
        strategy_params: Dict[str, Any],
        neural_policy: Optional[Dict[str, Any]] = None,
        use_neural_threshold: float = 0.3,
        enable_neural: bool = False,
        max_latency_ms: float = 10.0
    ):
        """
        초기화
        
        Args:
            agent_id: 에이전트 ID
            strategy_params: 전략 파라미터 (규칙 기반용)
            neural_policy: 신경망 정책 모델 (None이면 신경망 미사용)
            use_neural_threshold: 신경망 사용 최소 신뢰도 (0~1)
            enable_neural: 신경망 활성화 여부
            max_latency_ms: 최대 허용 지연 시간 (밀리초)
        """
        # 기존 StrategyAgent 초기화
        super().__init__(agent_id, strategy_params)
        
        self.neural_policy = neural_policy
        self.use_neural_threshold = use_neural_threshold
        self.enable_neural = enable_neural and (neural_policy is not None)
        self.max_latency_ms = max_latency_ms
        
        # 의사결정 로그 (디버깅/분석용)
        self.decision_log: List[Dict[str, Any]] = []
        self.stats = {
            'rule_decisions': 0,
            'neural_decisions': 0,
            'neural_errors': 0,
            'clear_signal_count': 0
        }
    
    def decide_action(self, market_state: MarketState) -> Action:
        """
        하이브리드 의사결정
        
        흐름:
        1. 명확한 신호 체크 (규칙 기반)
        2. 신경망 사용 가능하고 활성화된 경우 신경망 판단
        3. 기본값: 규칙 기반 (기존 로직)
        
        Returns:
            Action: BUY/SELL/HOLD
        """
        try:
            # 1. 명확한 신호 체크 (규칙 기반, 빠른 처리)
            clear_action = self._check_clear_signals(market_state)
            if clear_action is not None:
                self._log_decision('rule', 'clear_signal', clear_action, 1.0)
                self.stats['clear_signal_count'] += 1
                self.stats['rule_decisions'] += 1
                return clear_action
            
            # 2. 신경망 사용 가능하고 활성화된 경우
            if self.enable_neural and self.neural_policy is not None:
                try:
                    import time
                    start_time = time.time()
                    
                    # 🔥 평가 단계에서는 deterministic=True로 일관된 액션 생성
                    # 학습 단계에서는 deterministic=False로 탐험 허용
                    neural_result = self._get_neural_action(market_state, deterministic=True)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # 지연 시간 체크
                    if latency_ms > self.max_latency_ms:
                        logger.warning(f"⚠️ 신경망 지연 시간 초과: {latency_ms:.2f}ms > {self.max_latency_ms}ms, 규칙으로 폴백")
                        self.stats['neural_errors'] += 1
                        rule_action = super().decide_action(market_state)
                        self._log_decision('rule', 'latency_exceeded', rule_action, 0.0)
                        self.stats['rule_decisions'] += 1
                        return rule_action
                    
                    # 🔥 신뢰도 체크 완화: 평가 단계에서는 신경망을 더 적극적으로 사용
                    # use_neural_threshold를 낮춰서 신경망 사용 빈도 증가
                    effective_threshold = max(0.1, self.use_neural_threshold * 0.5)  # 최소 0.1, 기본값의 50%
                    
                    if neural_result['confidence'] >= effective_threshold:
                        self._log_decision(
                            'neural',
                            f"confidence_{neural_result['confidence']:.2f}",
                            neural_result['action'],
                            neural_result['confidence']
                        )
                        self.stats['neural_decisions'] += 1
                        return neural_result['action']
                    else:
                        # 신뢰도 낮으면 규칙으로 폴백
                        logger.debug(f"신경망 신뢰도 낮음: {neural_result['confidence']:.2f} < {effective_threshold:.2f}")
                
                except Exception as e:
                    logger.warning(f"⚠️ 신경망 추론 실패, 규칙으로 폴백: {e}")
                    self.stats['neural_errors'] += 1
            
            # 3. 기본값: 규칙 기반 (기존 로직 사용)
            rule_action = super().decide_action(market_state)
            self._log_decision('rule', 'default', rule_action, 0.5)
            self.stats['rule_decisions'] += 1
            return rule_action
            
        except Exception as e:
            logger.error(f"❌ 하이브리드 의사결정 실패: {e}")
            # 최종 폴백: 규칙 기반
            rule_action = super().decide_action(market_state)
            self._log_decision('rule', 'error_fallback', rule_action, 0.0)
            return rule_action
    
    def _check_clear_signals(self, market_state: MarketState) -> Optional[Action]:
        """
        명확한 신호 체크 (규칙 기반)
        
        매우 강한 신호는 즉시 결정하여 처리 속도 향상
        
        Returns:
            Action 또는 None
        """
        try:
            # 매우 강한 매수 신호
            # 조건: RSI 매우 낮음 + MACD 강한 상승 + 거래량 급증 + 레짐 신뢰도 높음
            if (market_state.rsi < 20 and 
                market_state.macd > market_state.macd_signal * 1.5 and
                market_state.volume_ratio > 2.0 and
                market_state.regime_confidence > 0.7 and
                market_state.regime_stage >= 4):  # 중립 이상
                return Action.BUY
            
            # 매우 강한 매도 신호
            # 조건: RSI 매우 높음 + MACD 강한 하락 + 레짐 신뢰도 높음
            if (market_state.rsi > 80 and
                market_state.macd < market_state.macd_signal * 0.5 and
                market_state.regime_confidence > 0.7 and
                market_state.regime_stage <= 4):  # 중립 이하
                return Action.SELL
            
            return None
            
        except Exception as e:
            logger.debug(f"명확한 신호 체크 실패: {e}")
            return None
    
    def _get_neural_action(self, market_state: MarketState, deterministic: bool = False) -> Dict[str, Any]:
        """
        신경망으로 액션 결정
        
        Args:
            market_state: 시장 상태
            deterministic: True면 최대 확률 액션, False면 샘플링 (평가 시 True 권장)
        
        Returns:
            {
                'action': Action,
                'confidence': float,  # 0~1
                'action_probs': np.ndarray,  # (3,) 액션별 확률
                'value': float  # 상태 가치
            }
        """
        from rl_pipeline.hybrid.features import build_state_vector
        from rl_pipeline.hybrid.neural_policy_jax import apply
        import jax.random as jrandom
        
        # 상태 벡터 변환
        state_vec = build_state_vector(market_state)
        
        # JAX 랜덤 키 생성 (에이전트별 고유 키)
        agent_hash = hash(self.agent_id) % (2**31)
        rng_key = jrandom.PRNGKey(agent_hash)
        
        # 🔥 신경망 추론 (평가 시 deterministic=True로 일관된 액션 생성)
        result = apply(self.neural_policy, state_vec, rng_key, deterministic=deterministic)
        
        # Action enum으로 변환
        action_map = {
            0: Action.HOLD,
            1: Action.BUY,
            2: Action.SELL
        }
        
        return {
            'action': action_map.get(result['action'], Action.HOLD),
            'confidence': result['confidence'],
            'action_probs': result['action_probs'],
            'value': result['value']
        }
    
    def _log_decision(self, method: str, reason: str, action: Action, confidence: float):
        """
        의사결정 로그 저장 (디버깅/분석용)
        
        Args:
            method: 'rule' or 'neural'
            reason: 결정 사유
            action: 선택된 액션
            confidence: 신뢰도 (0~1)
        """
        self.decision_log.append({
            'method': method,
            'reason': reason,
            'action': action.value,
            'confidence': confidence
        })
        
        # 로그가 너무 길어지면 최근 N개만 유지
        if len(self.decision_log) > 1000:
            self.decision_log = self.decision_log[-500:]
    
    def get_stats(self) -> Dict[str, Any]:
        """의사결정 통계 반환"""
        total = self.stats['rule_decisions'] + self.stats['neural_decisions']
        
        return {
            **self.stats,
            'total_decisions': total,
            'rule_ratio': self.stats['rule_decisions'] / total if total > 0 else 0.0,
            'neural_ratio': self.stats['neural_decisions'] / total if total > 0 else 0.0,
        }

