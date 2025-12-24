"""
통합 의사결정 점수 (Judgement Score) 시스템
- Signal, Thompson, Risk, Post-Trade 등 다양한 판단 요소를 단일 점수로 통합
- 실전/가상 매매에서 동일한 기준으로 의사결정 수행
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json

class DecisionType(Enum):
    PROMOTE = "PROMOTE"   # 강력 매수 / 승격
    HOLD = "HOLD"         # 관망 / 보류
    KILL = "KILL"         # 매도 / 폐기
    EXPLORE = "EXPLORE"   # 탐험적 진입 (Leap of Faith)

@dataclass
class JudgementComponents:
    """판단 점수 구성 요소"""
    signal_score: float = 0.0       # 시그널 점수 (0.0 ~ 1.0 정규화)
    thompson_score: float = 0.0     # 톰슨 샘플링 점수 (성공 확률)
    risk_score: float = 0.0         # 리스크 점수 (낮을수록 좋음, 1.0 - Risk)
    market_score: float = 0.0       # 시장 상황 점수
    post_trade_score: float = 0.0   # 사후 평가 점수 (MFE/MAE 기반)
    
    # 가중치 (기본값)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'signal': 0.4,
        'thompson': 0.3,
        'risk': 0.2,
        'market': 0.1,
        'post_trade': 0.0 # 사후 평가는 진입 시점에는 0일 수 있음
    })

@dataclass
class JudgementResult:
    """최종 판단 결과"""
    score: float                    # 최종 통합 점수 (0.0 ~ 1.0)
    decision: DecisionType          # 최종 결정
    components: JudgementComponents # 세부 점수
    reasons: List[str]              # 결정 사유 목록
    meta_data: Dict[str, Any] = field(default_factory=dict) # 추가 메타 데이터

class JudgementSystem:
    """통합 의사결정 시스템"""
    
    @staticmethod
    def evaluate(
        signal_info: Any,           # SignalInfo 객체 or Dict
        thompson_prob: float,       # Thompson Sampling 확률 (없으면 0.5)
        risk_level: str,            # 리스크 레벨 ('low', 'medium', 'high')
        market_context: Dict,       # 시장 상황 ({'trend': ..., 'volatility': ...})
        is_exploration: bool = False # 탐험 모드 여부
    ) -> JudgementResult:
        
        # 1. 컴포넌트 점수 계산
        # 시그널 점수 정규화 (-1~1 -> 0~1)
        raw_signal = getattr(signal_info, 'signal_score', 0.0) if hasattr(signal_info, 'signal_score') else signal_info.get('signal_score', 0.0)
        signal_score = (raw_signal + 1) / 2
        
        # 리스크 점수 (Low=1.0, Medium=0.6, High=0.2)
        risk_map = {'low': 1.0, 'medium': 0.6, 'high': 0.2, 'unknown': 0.5}
        risk_score = risk_map.get(str(risk_level).lower(), 0.5)
        
        # 시장 점수 (Bullish=1.0, Neutral=0.5, Bearish=0.2)
        trend = market_context.get('trend', 'neutral')
        market_map = {'bullish': 1.0, 'neutral': 0.5, 'bearish': 0.2}
        market_score = market_map.get(trend, 0.5)
        
        components = JudgementComponents(
            signal_score=signal_score,
            thompson_score=thompson_prob,
            risk_score=risk_score,
            market_score=market_score
        )
        
        # 2. 가중 평균 계산
        total_weight = sum(components.weights.values())
        weighted_sum = (
            components.signal_score * components.weights['signal'] +
            components.thompson_score * components.weights['thompson'] +
            components.risk_score * components.weights['risk'] +
            components.market_score * components.weights['market']
        )
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # 3. 결정 내리기
        reasons = []
        decision = DecisionType.HOLD
        
        # 기본 승격 기준
        if final_score >= 0.7:
            decision = DecisionType.PROMOTE
            reasons.append(f"높은 통합 점수 ({final_score:.2f})")
        elif final_score < 0.3:
            decision = DecisionType.KILL
            reasons.append(f"낮은 통합 점수 ({final_score:.2f})")
        
        # 4. 탐험적 진입 (Leap of Faith) 오버라이드
        # 점수가 낮아도 시그널이 매우 강력하고 신뢰도가 높으면 탐험
        confidence = getattr(signal_info, 'confidence', 0.0) if hasattr(signal_info, 'confidence') else signal_info.get('confidence', 0.0)
        
        if decision == DecisionType.HOLD:
            if raw_signal >= 0.4 and confidence >= 0.6:
                decision = DecisionType.EXPLORE
                reasons.append("강력한 시그널 기반 탐험적 진입 (Leap of Faith)")
                # 탐험 모드일 경우 점수 보정 (최소 0.5)
                final_score = max(final_score, 0.5)
        
        return JudgementResult(
            score=final_score,
            decision=decision,
            components=components,
            reasons=reasons
        )

