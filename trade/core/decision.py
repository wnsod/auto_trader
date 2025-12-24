"""
매매 의사결정 모듈 (Decision Maker)
- 실전/가상 매매에서 공통으로 사용하는 진입/청산 판단 로직
- Thompson Sampling, Risk Management, AI Model 등을 종합하여 판단
"""

import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from trade.virtual_trade_learner import ThompsonSamplingLearner, SignalInfo

# 로깅 설정
logger = logging.getLogger(__name__)

class DecisionMaker:
    """통합 의사결정 시스템"""
    
    def __init__(self, strategy_db_path: str = None):
        # Thompson Sampling 학습기 초기화
        self.thompson_sampler = ThompsonSamplingLearner(db_path=strategy_db_path)
        print("🧠 [DecisionMaker] 의사결정 시스템 초기화 완료 (Thompson Sampling 연동)")

    def decide_buy(self, signal: SignalInfo, is_simulation: bool = False) -> Tuple[bool, float, str]:
        """
        매수 여부 결정 (통합 로직)
        Returns: (실행여부, 최종점수, 사유)
        """
        try:
            # 1. 기대 수익률 및 변동성 체크 (1차 필터)
            # 슬리피지 고려하여 최소 1% 이상 기대 수익률 확보 필요 (사용자 요청 반영)
            is_viable, viability_reason = self._check_viability(signal)
            if not is_viable:
                return False, 0.0, f"기대 수익률 또는 변동성 미달 ({viability_reason})"

            # 2. 쉐도우 트레이딩 감지 (강제 진입 후보)
            is_shadow_forced = False
            # 쫄보 전략(HOLD)인데 점수가 높으면 강제 진입 검토
            # signal.action.value == 0 (HOLD) 가정
            # SignalAction Enum 처리가 필요할 수 있으므로 문자열/값 모두 고려
            is_hold = str(signal.action).upper() == 'HOLD' or signal.action == 'hold' or getattr(signal.action, 'value', None) == 0
            
            if is_hold and abs(signal.signal_score) > 0.7:
                is_shadow_forced = True
                # print(f"👻 쉐도우 트레이딩 후보 감지: {signal.coin}")

            # 3. Thompson Sampling 학습기 판단 (핵심)
            if self.thompson_sampler:
                # 패턴 추출
                pattern = self._extract_signal_pattern(signal)
                
                # 학습기에게 문의
                should_execute, score, reason = self.thompson_sampler.should_execute_action(
                    signal_pattern=pattern,
                    signal_score=signal.signal_score,
                    action_type='buy'
                )
                
                # 쉐도우 트레이딩 강제성 부여
                # Thompson 점수가 최악(-0.5 미만)만 아니면 강제 집행
                if is_shadow_forced and not should_execute:
                    if score > -0.5:
                        should_execute = True
                        reason += " [Shadow Trading Forced]"
                        
                return should_execute, score, reason

            # 학습기가 없는 경우 (Fallback)
            if is_shadow_forced:
                return True, signal.signal_score, "Shadow Trading Forced (No Learner)"
                
            return True, signal.signal_score, "기본 실행 (학습기 없음)"

        except Exception as e:
            logger.error(f"매수 판단 중 오류: {e}")
            return True, signal.signal_score, "오류로 인한 안전 실행"

    def decide_sell(self, signal: SignalInfo) -> Tuple[bool, float, str]:
        """매도 여부 결정"""
        try:
            if self.thompson_sampler:
                pattern = self._extract_signal_pattern(signal)
                return self.thompson_sampler.should_execute_action(
                    signal_pattern=pattern,
                    signal_score=signal.signal_score,
                    action_type='sell'
                )
            return True, signal.signal_score, "기본 매도 실행"
        except Exception as e:
            logger.error(f"매도 판단 중 오류: {e}")
            return True, signal.signal_score, "오류로 인한 안전 매도"

    def _check_viability(self, signal: SignalInfo) -> Tuple[bool, str]:
        """기대 수익률 및 변동성 기반 타당성 검사
        
        🔧 개선사항:
        - 목표가 계산에서 이미 1.5% 보장하므로 기준 완화 (0.5%)
        - 비교 연산자 <= 로 변경 (정확히 같으면 통과)
        """
        try:
            # 목표가와 현재가가 유효할 때만 검사
            if hasattr(signal, 'target_price') and signal.target_price > 0 and signal.price > 0:
                expected_profit_pct = ((signal.target_price - signal.price) / signal.price) * 100
                
                # 목표가 망상 방지 (50% 이상은 10%로 보정)
                if expected_profit_pct > 50.0:
                    expected_profit_pct = 10.0
                    signal.target_price = signal.price * 1.10
                
                # 🔧🔧 대폭 완화된 최소 요구 수익률 (가상매매 활성화)
                # 가상매매가 너무 보수적이어서 거래가 안됨 → 0.3%로 대폭 낮춤
                volatility = getattr(signal, 'volatility', 0.02) or 0.02
                min_expected_profit = max(0.3, volatility * 100 * 0.2)  # 0.5% → 0.3%
                
                # 🔧 < 사용 (정확히 같으면 통과, 미만일 때만 거부)
                if expected_profit_pct < min_expected_profit:
                    return False, f"기대수익 {expected_profit_pct:.2f}% < 최소 {min_expected_profit:.2f}%"
            
            return True, "OK"
        except Exception as e:
            return True, f"검사 오류({e})로 인한 통과"  # 계산 불가 시 통과 (안전)

    def _extract_signal_pattern(self, signal: SignalInfo) -> str:
        """시그널에서 패턴 문자열 추출 (일관성 보장)"""
        try:
            # RSI
            rsi = getattr(signal, 'rsi', 50.0)
            rsi_state = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            
            # MACD
            macd = getattr(signal, 'macd', 0.0)
            if macd > 0.01: macd_state = 'strong_bullish'
            elif macd > 0: macd_state = 'bullish'
            elif macd < -0.01: macd_state = 'strong_bearish'
            else: macd_state = 'bearish'
            
            # Volume
            vol_ratio = getattr(signal, 'volume_ratio', 1.0)
            if vol_ratio > 2.0: vol_state = 'very_high'
            elif vol_ratio > 1.5: vol_state = 'high'
            elif vol_ratio < 0.5: vol_state = 'low'
            else: vol_state = 'normal'
            
            # Confidence
            conf = getattr(signal, 'confidence', 0.5)
            if conf > 0.8: conf_state = 'very_high'
            elif conf > 0.6: conf_state = 'high'
            elif conf < 0.4: conf_state = 'low'
            else: conf_state = 'medium'
            
            return f"{signal.coin}_{rsi_state}_{macd_state}_{vol_state}_{conf_state}"
            
        except Exception:
            return f"{signal.coin}_unknown"

