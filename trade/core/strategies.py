"""
🎯 매매 전략 중앙 관리 모듈

10가지 매매 전략 정의 및 Thompson Sampling 기반 전략 선택 시스템

전략 목록:
1. trend     - 추세 추종: 확인된 추세를 따라가기
2. bottom    - 저점 매수: 바닥에서 사서 기다리기
3. scalp     - 급등 스캘핑: 급등 시 빠른 진입/청산
4. swing     - 스윙 트레이딩: 파동의 시작에서 끝까지
5. revert    - 평균 회귀: 극단값에서 평균 복귀 기대
6. breakout  - 브레이크아웃: 박스권 돌파 시 진입
7. dca       - 분할 매수: 장기 상승 + 단기 하락 시 추가 매수
8. momentum  - 모멘텀: 강한 추세에 편승
9. counter   - 역추세: 과열 시 반전 노리기
10. range    - 레인지: 박스권 내 반복 매매
"""

import os
import sqlite3
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum


# ============================================================================
# 전략 타입 정의
# ============================================================================
class StrategyType(Enum):
    """매매 전략 타입"""
    TREND = "trend"           # 추세 추종
    BOTTOM = "bottom"         # 저점 매수
    SCALP = "scalp"           # 급등 스캘핑
    SWING = "swing"           # 스윙 트레이딩
    REVERT = "revert"         # 평균 회귀
    BREAKOUT = "breakout"     # 브레이크아웃
    DCA = "dca"               # 분할 매수
    MOMENTUM = "momentum"     # 모멘텀
    COUNTER = "counter"       # 역추세
    RANGE = "range"           # 레인지
    
    @classmethod
    def all_types(cls) -> List[str]:
        return [s.value for s in cls]


# ============================================================================
# 전략별 청산 규칙 정의
# ============================================================================
@dataclass
class StrategyExitRules:
    """전략별 청산 규칙"""
    take_profit_pct: float      # 익절 %
    stop_loss_pct: float        # 손절 %
    max_holding_hours: int      # 최대 보유 시간
    trailing_stop: bool         # 트레일링 스탑 사용 여부
    trailing_trigger_pct: float # 트레일링 시작 수익률 %
    trailing_distance_pct: float # 트레일링 간격 %
    partial_take_profit: bool   # 분할 익절 여부
    description: str            # 전략 설명


# 전략별 청산 규칙 정의
STRATEGY_EXIT_RULES: Dict[str, StrategyExitRules] = {
    "trend": StrategyExitRules(
        take_profit_pct=15.0,
        stop_loss_pct=5.0,
        max_holding_hours=168,      # 7일
        trailing_stop=True,
        trailing_trigger_pct=5.0,
        trailing_distance_pct=3.0,
        partial_take_profit=True,
        description="추세 추종: 확인된 추세를 따라감"
    ),
    "bottom": StrategyExitRules(
        take_profit_pct=30.0,
        stop_loss_pct=8.0,
        max_holding_hours=336,      # 14일
        trailing_stop=False,
        trailing_trigger_pct=0.0,
        trailing_distance_pct=0.0,
        partial_take_profit=False,
        description="저점 매수: 바닥에서 사서 장기 보유"
    ),
    "scalp": StrategyExitRules(
        take_profit_pct=1.5,
        stop_loss_pct=1.0,
        max_holding_hours=4,        # 4시간
        trailing_stop=False,
        trailing_trigger_pct=0.0,
        trailing_distance_pct=0.0,
        partial_take_profit=False,
        description="급등 스캘핑: 빠른 진입/청산"
    ),
    "swing": StrategyExitRules(
        take_profit_pct=20.0,
        stop_loss_pct=6.0,
        max_holding_hours=240,      # 10일
        trailing_stop=True,
        trailing_trigger_pct=8.0,
        trailing_distance_pct=4.0,
        partial_take_profit=True,
        description="스윙 트레이딩: 파동 전체 캡처"
    ),
    "revert": StrategyExitRules(
        take_profit_pct=5.0,
        stop_loss_pct=3.0,
        max_holding_hours=48,       # 2일
        trailing_stop=False,
        trailing_trigger_pct=0.0,
        trailing_distance_pct=0.0,
        partial_take_profit=False,
        description="평균 회귀: 극단값에서 평균 복귀"
    ),
    "breakout": StrategyExitRules(
        take_profit_pct=12.0,
        stop_loss_pct=4.0,
        max_holding_hours=120,      # 5일
        trailing_stop=True,
        trailing_trigger_pct=6.0,
        trailing_distance_pct=3.0,
        partial_take_profit=False,
        description="브레이크아웃: 박스권 돌파 후 추세 추종"
    ),
    "dca": StrategyExitRules(
        take_profit_pct=25.0,
        stop_loss_pct=15.0,         # 넓은 손절 (분할 매수 고려)
        max_holding_hours=504,      # 21일
        trailing_stop=True,
        trailing_trigger_pct=10.0,
        trailing_distance_pct=5.0,
        partial_take_profit=True,
        description="분할 매수: 평균 단가 낮추며 장기 보유"
    ),
    "momentum": StrategyExitRules(
        take_profit_pct=10.0,
        stop_loss_pct=4.0,
        max_holding_hours=72,       # 3일
        trailing_stop=True,
        trailing_trigger_pct=4.0,
        trailing_distance_pct=2.0,
        partial_take_profit=False,
        description="모멘텀: 강한 추세에 빠르게 편승"
    ),
    "counter": StrategyExitRules(
        take_profit_pct=8.0,
        stop_loss_pct=4.0,
        max_holding_hours=24,       # 1일
        trailing_stop=False,
        trailing_trigger_pct=0.0,
        trailing_distance_pct=0.0,
        partial_take_profit=False,
        description="역추세: 과열 시 반전 포착"
    ),
    "range": StrategyExitRules(
        take_profit_pct=4.0,
        stop_loss_pct=2.5,
        max_holding_hours=48,       # 2일
        trailing_stop=False,
        trailing_trigger_pct=0.0,
        trailing_distance_pct=0.0,
        partial_take_profit=False,
        description="레인지: 박스권 내 지지/저항 매매"
    ),
}


# ============================================================================
# 전략별 진입 임계값
# ============================================================================
STRATEGY_ENTRY_THRESHOLDS: Dict[str, float] = {
    "trend": 0.40,      # 보수적 (확인된 추세)
    "bottom": 0.35,     # 중간 (저점 확인)
    "scalp": 0.50,      # 높음 (빠른 판단 필요)
    "swing": 0.35,      # 중간
    "revert": 0.45,     # 높음 (역방향이라 신중)
    "breakout": 0.45,   # 높음 (돌파 확인)
    "dca": 0.25,        # 낮음 (분할 매수라 관대)
    "momentum": 0.40,   # 중간
    "counter": 0.50,    # 높음 (역방향이라 신중)
    "range": 0.30,      # 낮음 (박스권 내 반복)
}


# ============================================================================
# 🆕 전략별 횡보(Sideways) 정책
# ============================================================================
@dataclass
class SidewaysPolicy:
    """전략별 횡보 시장 대응 정책"""
    exempt_from_switch: bool        # 횡보 갈아타기 체크 면제 여부
    patience_multiplier: float      # patience_hours 배율 (1.0 = 기본, 2.0 = 2배)
    exempt_from_peak_sell: bool     # 횡보 고점 매도 면제 여부
    min_profit_for_peak_sell: float # 횡보 고점 매도 시 최소 수익률 (면제 아닐 경우)
    description: str                # 정책 설명


# 전략별 횡보 정책 정의
# 충돌 수준: 🔴 높음 -> 완전 면제, 🟠 중간 -> 부분 완화, 🟡 낮음 -> 약간 완화, 🟢 없음 -> 기본
STRATEGY_SIDEWAYS_POLICY: Dict[str, SidewaysPolicy] = {
    # 🔴 높음 - 횡보에서 작동하는 전략 (완전 면제)
    "range": SidewaysPolicy(
        exempt_from_switch=True,      # 갈아타기 면제
        patience_multiplier=999.0,    # 사실상 무제한
        exempt_from_peak_sell=False,  # 고점 매도는 허용 (레인지 전략의 핵심)
        min_profit_for_peak_sell=2.0, # 2% 이상이면 고점 매도
        description="🔴 레인지: 횡보 전략이므로 갈아타기 면제, 고점 매도는 허용"
    ),
    
    # 🟠 중간 - 횡보 구간에서 기다려야 하는 전략 (부분 완화)
    "revert": SidewaysPolicy(
        exempt_from_switch=True,      # 평균 회귀 대기 중
        patience_multiplier=3.0,      # 3배 인내
        exempt_from_peak_sell=False,  # 고점 매도 허용 (회귀 완료 시점)
        min_profit_for_peak_sell=3.0, # 3% 이상이면 매도
        description="🟠 평균회귀: 횡보에서 평균 회귀 대기, patience 3배"
    ),
    "bottom": SidewaysPolicy(
        exempt_from_switch=True,      # 저점에서 기다리는 중
        patience_multiplier=4.0,      # 4배 인내 (장기 보유 전략)
        exempt_from_peak_sell=True,   # 고점 매도 면제 (상승 대기)
        min_profit_for_peak_sell=0.0, # N/A
        description="🟠 저점매수: 상승 대기, 횡보 체크 면제"
    ),
    "dca": SidewaysPolicy(
        exempt_from_switch=True,      # 분할 매수 축적 중
        patience_multiplier=5.0,      # 5배 인내 (장기 축적)
        exempt_from_peak_sell=True,   # 고점 매도 면제 (장기 보유)
        min_profit_for_peak_sell=0.0, # N/A
        description="🟠 분할매수: 장기 축적, 횡보 체크 면제"
    ),
    
    # 🟡 낮음 - 약간의 완화 필요
    "swing": SidewaysPolicy(
        exempt_from_switch=False,     # 갈아타기 허용
        patience_multiplier=2.0,      # 2배 인내
        exempt_from_peak_sell=False,  # 고점 매도 허용
        min_profit_for_peak_sell=5.0, # 5% 이상 수익에서만 매도 (파동 캡처)
        description="🟡 스윙: patience 2배, 고점 매도 수익률 기준 상향"
    ),
    "counter": SidewaysPolicy(
        exempt_from_switch=False,     # 갈아타기 허용
        patience_multiplier=1.5,      # 1.5배 인내
        exempt_from_peak_sell=False,  # 고점 매도 허용 (역추세 반전 확인)
        min_profit_for_peak_sell=2.0, # 2% 이상이면 매도
        description="🟡 역추세: patience 1.5배"
    ),
    
    # 🟢 없음 - 횡보 체크 유지 (추세 필요 전략)
    "trend": SidewaysPolicy(
        exempt_from_switch=False,     # 추세 없으면 교체
        patience_multiplier=1.0,      # 기본 인내
        exempt_from_peak_sell=False,  # 고점 매도 허용
        min_profit_for_peak_sell=1.0, # 기본 1%
        description="🟢 추세추종: 횡보 시 교체 권장"
    ),
    "momentum": SidewaysPolicy(
        exempt_from_switch=False,     # 모멘텀 없으면 교체
        patience_multiplier=0.8,      # 더 빠른 판단 (모멘텀은 속도)
        exempt_from_peak_sell=False,  # 고점 매도 허용
        min_profit_for_peak_sell=1.0, # 기본 1%
        description="🟢 모멘텀: 횡보 시 빠른 교체"
    ),
    "breakout": SidewaysPolicy(
        exempt_from_switch=False,     # 돌파 대기, 하지만 너무 길면 교체
        patience_multiplier=1.2,      # 약간의 추가 인내 (돌파 대기)
        exempt_from_peak_sell=True,   # 고점 매도 면제 (돌파 기대)
        min_profit_for_peak_sell=0.0, # N/A
        description="🟢 돌파: 횡보 돌파 대기, 고점 매도 면제"
    ),
    "scalp": SidewaysPolicy(
        exempt_from_switch=False,     # 빠른 판단
        patience_multiplier=0.5,      # 더 빠른 판단 (스캘핑)
        exempt_from_peak_sell=False,  # 고점 매도 적극 활용
        min_profit_for_peak_sell=0.5, # 0.5%라도 매도 (빠른 청산)
        description="🟢 스캘핑: 횡보 시 빠른 청산"
    ),
}


def get_sideways_policy(strategy_type: str) -> SidewaysPolicy:
    """전략별 횡보 정책 조회"""
    return STRATEGY_SIDEWAYS_POLICY.get(
        strategy_type, 
        STRATEGY_SIDEWAYS_POLICY.get("trend")  # 기본값: trend 정책
    )


def should_exempt_from_sideways_switch(strategy_type: str) -> bool:
    """횡보 갈아타기 면제 여부 확인"""
    policy = get_sideways_policy(strategy_type)
    return policy.exempt_from_switch if policy else False


# ============================================================================
# 🆕 전략별 레짐 조정 계수
# ============================================================================
# 각 전략이 특정 레짐에서 얼마나 효과적인지를 나타내는 계수
# > 1.0: 해당 레짐에서 더 인내 (전략에 유리)
# < 1.0: 해당 레짐에서 덜 인내 (전략에 불리, 빠른 청산/교체)
# = 1.0: 기본 (전략 자체 배율 적용)

STRATEGY_REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # 🔴 range: 횡보/중립 레짐에서 최적, 추세 레짐에서 비효율
    "range": {
        "extreme_bearish": 0.3,    # 강한 하락 → 레인지 깨짐
        "bearish": 0.5,            # 하락 → 비효율
        "sideways_bearish": 1.2,   # 약세 횡보 → 레인지 기회
        "neutral": 1.5,            # 중립 → 최적!
        "sideways_bullish": 1.2,   # 강세 횡보 → 레인지 기회
        "bullish": 0.5,            # 상승 → 비효율
        "extreme_bullish": 0.3,    # 강한 상승 → 레인지 깨짐
    },
    
    # 🟠 revert: 극단 레짐에서 평균 회귀 기대
    "revert": {
        "extreme_bearish": 1.5,    # 과매도 → 회귀 기대
        "bearish": 1.2,            # 약세 → 회귀 기회
        "sideways_bearish": 1.0,
        "neutral": 0.8,            # 중립 → 회귀할 곳 없음
        "sideways_bullish": 1.0,
        "bullish": 1.2,            # 강세 → 조정 기대
        "extreme_bullish": 1.5,    # 과매수 → 회귀 기대
    },
    
    # 🟠 bottom: 하락 레짐에서 저점 매수 기회
    "bottom": {
        "extreme_bearish": 1.8,    # 극도 약세 → 저점 기회!
        "bearish": 1.5,            # 약세 → 저점 탐색
        "sideways_bearish": 1.2,   # 약세 횡보 → 바닥 확인 중
        "neutral": 1.0,
        "sideways_bullish": 0.7,   # 상승 전환 → 이미 올랐을 수 있음
        "bullish": 0.4,            # 상승 중 → 저점 아님
        "extreme_bullish": 0.2,    # 급등 → 저점 전략 부적합
    },
    
    # 🟠 dca: 하락 레짐에서 분할 매수 기회
    "dca": {
        "extreme_bearish": 1.5,    # 극도 약세 → 적립 기회
        "bearish": 1.3,            # 약세 → 분할 매수 적기
        "sideways_bearish": 1.2,
        "neutral": 1.1,
        "sideways_bullish": 1.0,
        "bullish": 0.8,            # 상승 중 → 추가 매수 신중
        "extreme_bullish": 0.5,    # 급등 → 고점 매수 위험
    },
    
    # 🟡 swing: 방향 있는 레짐에서 파동 캡처
    "swing": {
        "extreme_bearish": 1.0,
        "bearish": 1.3,            # 하락 파동 캡처
        "sideways_bearish": 0.8,
        "neutral": 0.6,            # 중립 → 파동 없음
        "sideways_bullish": 0.8,
        "bullish": 1.3,            # 상승 파동 캡처
        "extreme_bullish": 1.0,
    },
    
    # 🟡 counter: 극단 레짐에서 반전 노림
    "counter": {
        "extreme_bearish": 1.8,    # 극도 약세 → 반전 기대!
        "bearish": 1.2,
        "sideways_bearish": 0.8,
        "neutral": 0.5,            # 중립 → 반전할 곳 없음
        "sideways_bullish": 0.8,
        "bullish": 1.2,
        "extreme_bullish": 1.8,    # 극도 강세 → 조정 기대!
    },
    
    # 🟢 trend: 추세 레짐에서 최적
    "trend": {
        "extreme_bearish": 0.8,    # 극단은 반전 위험
        "bearish": 1.3,            # 하락 추세 추종
        "sideways_bearish": 0.7,
        "neutral": 0.5,            # 중립 → 추세 없음
        "sideways_bullish": 0.7,
        "bullish": 1.3,            # 상승 추세 추종
        "extreme_bullish": 0.8,    # 극단은 반전 위험
    },
    
    # 🟢 momentum: 강세 레짐에서 최적
    "momentum": {
        "extreme_bearish": 0.3,    # 하락 모멘텀은 위험
        "bearish": 0.5,
        "sideways_bearish": 0.6,
        "neutral": 0.4,            # 중립 → 모멘텀 없음
        "sideways_bullish": 0.8,
        "bullish": 1.5,            # 상승 모멘텀 활용!
        "extreme_bullish": 1.3,    # 급등 모멘텀
    },
    
    # 🟢 breakout: 횡보 후 돌파 대기
    "breakout": {
        "extreme_bearish": 0.5,
        "bearish": 0.7,
        "sideways_bearish": 1.3,   # 횡보 → 돌파 대기
        "neutral": 1.2,            # 중립 → 돌파 대기
        "sideways_bullish": 1.3,   # 횡보 → 돌파 대기
        "bullish": 1.0,            # 이미 돌파됨
        "extreme_bullish": 0.8,
    },
    
    # 🟢 scalp: 레짐 영향 적음 (빠른 매매)
    "scalp": {
        "extreme_bearish": 0.7,    # 변동성 크지만 위험
        "bearish": 0.9,
        "sideways_bearish": 1.0,
        "neutral": 1.0,
        "sideways_bullish": 1.0,
        "bullish": 1.1,
        "extreme_bullish": 1.2,    # 변동성 활용
    },
}


def get_regime_adjustment(strategy_type: str, regime: str) -> float:
    """
    전략+레짐 조합에 따른 조정 계수 반환
    
    Args:
        strategy_type: 전략 타입
        regime: 시장 레짐 (7개 중 하나)
    
    Returns:
        조정 계수 (1.0 = 기본)
    """
    # 레짐 이름 정규화
    regime_lower = regime.lower() if regime else 'neutral'
    
    # 간단한 레짐 이름 매핑 (호환성)
    if 'extreme' in regime_lower and 'bear' in regime_lower:
        regime_key = 'extreme_bearish'
    elif 'extreme' in regime_lower and 'bull' in regime_lower:
        regime_key = 'extreme_bullish'
    elif 'sideways' in regime_lower and 'bear' in regime_lower:
        regime_key = 'sideways_bearish'
    elif 'sideways' in regime_lower and 'bull' in regime_lower:
        regime_key = 'sideways_bullish'
    elif 'bear' in regime_lower:
        regime_key = 'bearish'
    elif 'bull' in regime_lower:
        regime_key = 'bullish'
    else:
        regime_key = 'neutral'
    
    # 전략별 레짐 조정 계수 조회
    strategy_adjustments = STRATEGY_REGIME_ADJUSTMENTS.get(strategy_type, {})
    return strategy_adjustments.get(regime_key, 1.0)


def get_patience_multiplier(strategy_type: str, regime: str = None) -> float:
    """
    전략별 patience 배율 조회 (레짐 반영)
    
    Args:
        strategy_type: 전략 타입
        regime: 시장 레짐 (선택사항)
    
    Returns:
        최종 patience 배율
    """
    policy = get_sideways_policy(strategy_type)
    base_multiplier = policy.patience_multiplier if policy else 1.0
    
    # 레짐 조정 적용
    if regime:
        regime_adjustment = get_regime_adjustment(strategy_type, regime)
        final_multiplier = base_multiplier * regime_adjustment
        return max(0.2, min(10.0, final_multiplier))  # 0.2 ~ 10.0 범위 제한
    
    return base_multiplier


def should_sideways_peak_sell(strategy_type: str, profit_pct: float, regime: str = None) -> Tuple[bool, str]:
    """
    횡보 고점 매도 여부 결정 (레짐 반영)
    
    Args:
        strategy_type: 전략 타입
        profit_pct: 현재 수익률
        regime: 시장 레짐 (선택사항)
    
    Returns:
        (should_sell, reason)
    """
    policy = get_sideways_policy(strategy_type)
    
    if policy.exempt_from_peak_sell:
        return False, f"전략({strategy_type}) 횡보 고점 매도 면제"
    
    # 🆕 레짐에 따른 최소 수익률 조정
    min_profit = policy.min_profit_for_peak_sell
    
    if regime:
        regime_lower = regime.lower() if regime else 'neutral'
        
        # 강세 레짐에서는 더 높은 수익률에서만 매도 (추가 상승 기대)
        if 'bull' in regime_lower and 'extreme' not in regime_lower:
            min_profit *= 1.5  # 50% 상향
        elif 'extreme_bullish' in regime_lower:
            min_profit *= 0.8  # 극단 강세에서는 빠른 익절
        # 약세 레짐에서는 빠른 익절 (하락 전환 우려)
        elif 'bear' in regime_lower:
            min_profit *= 0.7  # 30% 하향
    
    if profit_pct >= min_profit:
        return True, f"전략({strategy_type}) 횡보 고점 매도 ({profit_pct:.1f}% >= {min_profit:.1f}%)"
    
    return False, f"전략({strategy_type}) 수익률 부족 ({profit_pct:.1f}% < {min_profit:.1f}%)"


def get_strategy_regime_compatibility(strategy_type: str, regime: str) -> Tuple[float, str]:
    """
    전략과 레짐의 호환성 점수 반환 (정보 제공용)
    
    Returns:
        (compatibility_score, description)
        - 1.5+: 최적 조합
        - 1.0~1.5: 좋음
        - 0.7~1.0: 보통
        - 0.5~0.7: 비효율
        - <0.5: 부적합
    """
    adjustment = get_regime_adjustment(strategy_type, regime)
    
    if adjustment >= 1.5:
        return adjustment, f"🟢 최적 ({strategy_type} + {regime})"
    elif adjustment >= 1.0:
        return adjustment, f"🟡 좋음 ({strategy_type} + {regime})"
    elif adjustment >= 0.7:
        return adjustment, f"🟠 보통 ({strategy_type} + {regime})"
    elif adjustment >= 0.5:
        return adjustment, f"🔴 비효율 ({strategy_type} + {regime})"
    else:
        return adjustment, f"⛔ 부적합 ({strategy_type} + {regime})"


# ============================================================================
# 🆕 레짐 변화 감지 시스템
# ============================================================================
class RegimeChangeDetector:
    """
    시장 레짐 변화를 감지하고 전략 재평가를 트리거하는 시스템
    
    Usage:
        detector = RegimeChangeDetector()
        changed, old, new = detector.check_regime_change('bullish')
        if changed:
            # 전략 재평가 로직 실행
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.last_regime = 'neutral'
        self.regime_history: List[Dict] = []  # 레짐 변화 이력
        self.change_count = 0
        self.last_change_timestamp = 0
    
    def check_regime_change(self, current_regime: str) -> Tuple[bool, str, str]:
        """
        레짐 변화 여부 확인
        
        Args:
            current_regime: 현재 시장 레짐
            
        Returns:
            (changed, old_regime, new_regime)
        """
        if not current_regime:
            return False, self.last_regime, self.last_regime
        
        # 레짐 정규화
        normalized = self._normalize_regime(current_regime)
        
        if normalized != self.last_regime:
            old_regime = self.last_regime
            self.last_regime = normalized
            self.change_count += 1
            self.last_change_timestamp = int(time.time())
            
            # 이력 저장
            self.regime_history.append({
                'timestamp': self.last_change_timestamp,
                'from': old_regime,
                'to': normalized,
                'change_count': self.change_count
            })
            
            # 최대 100개 이력 유지
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            print(f"🔄 [레짐 변화] {old_regime.upper()} → {normalized.upper()} (총 {self.change_count}회 변화)")
            return True, old_regime, normalized
        
        return False, self.last_regime, normalized
    
    def _normalize_regime(self, regime: str) -> str:
        """레짐 이름 정규화"""
        regime_lower = regime.lower() if regime else 'neutral'
        
        if 'extreme' in regime_lower and 'bear' in regime_lower:
            return 'extreme_bearish'
        elif 'extreme' in regime_lower and 'bull' in regime_lower:
            return 'extreme_bullish'
        elif 'sideways' in regime_lower and 'bear' in regime_lower:
            return 'sideways_bearish'
        elif 'sideways' in regime_lower and 'bull' in regime_lower:
            return 'sideways_bullish'
        elif 'bear' in regime_lower:
            return 'bearish'
        elif 'bull' in regime_lower:
            return 'bullish'
        return 'neutral'
    
    def get_regime_stability(self) -> Tuple[float, str]:
        """
        현재 레짐 안정성 평가 (잦은 변화 = 불안정)
        
        Returns:
            (stability_score, description)
            - 1.0: 매우 안정 (최근 변화 없음)
            - 0.5: 보통
            - 0.0: 불안정 (잦은 변화)
        """
        if len(self.regime_history) < 2:
            return 1.0, "안정 (데이터 부족)"
        
        # 최근 6시간 내 변화 횟수
        now = int(time.time())
        recent_changes = sum(
            1 for h in self.regime_history 
            if now - h.get('timestamp', 0) < 6 * 3600
        )
        
        if recent_changes >= 5:
            return 0.0, f"⚠️ 불안정 (6시간 내 {recent_changes}회 변화)"
        elif recent_changes >= 3:
            return 0.3, f"🟠 다소 불안정 (6시간 내 {recent_changes}회 변화)"
        elif recent_changes >= 1:
            return 0.7, f"🟡 보통 (6시간 내 {recent_changes}회 변화)"
        else:
            return 1.0, "🟢 안정 (최근 변화 없음)"
    
    def should_reevaluate_strategies(self, current_regime: str) -> Tuple[bool, str]:
        """
        전략 재평가 필요 여부 판단
        
        Returns:
            (should_reevaluate, reason)
        """
        changed, old, new = self.check_regime_change(current_regime)
        
        if not changed:
            return False, ""
        
        # 레짐 그룹 변화 체크 (bearish <-> neutral <-> bullish)
        regime_groups = {
            'extreme_bearish': 'bearish_group',
            'bearish': 'bearish_group',
            'sideways_bearish': 'bearish_group',
            'neutral': 'neutral_group',
            'sideways_bullish': 'bullish_group',
            'bullish': 'bullish_group',
            'extreme_bullish': 'bullish_group'
        }
        
        old_group = regime_groups.get(old, 'neutral_group')
        new_group = regime_groups.get(new, 'neutral_group')
        
        if old_group != new_group:
            return True, f"📊 레짐 그룹 변화: {old_group} → {new_group} (전략 재평가 필요)"
        
        # 극단 레짐 전환
        if 'extreme' in new:
            return True, f"⚡ 극단 레짐 진입: {new} (전략 재평가 권장)"
        
        return False, f"ℹ️ 레짐 미세 변화: {old} → {new} (재평가 불필요)"
    
    def get_recommended_strategies_for_regime(self, regime: str) -> List[str]:
        """
        현재 레짐에 추천되는 전략 목록 반환
        """
        regime_key = self._normalize_regime(regime)
        
        # 각 전략의 레짐 조정 계수 확인
        strategy_scores = []
        for strategy_type in STRATEGY_REGIME_ADJUSTMENTS.keys():
            adj = STRATEGY_REGIME_ADJUSTMENTS[strategy_type].get(regime_key, 1.0)
            strategy_scores.append((strategy_type, adj))
        
        # 조정 계수가 높은 순으로 정렬
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 5개 추천
        return [s[0] for s in strategy_scores[:5] if s[1] >= 1.0]


# 싱글톤 인스턴스 접근 함수
def get_regime_detector() -> RegimeChangeDetector:
    """레짐 변화 감지기 싱글톤 인스턴스 반환"""
    return RegimeChangeDetector()


# ============================================================================
# 전략 매칭 함수들
# ============================================================================
def calc_trend_match(direction: str, adx: float, signal_continuity: float, 
                     macd: float = 0, rsi: float = 50) -> Dict[str, float]:
    """
    추세 추종 전략 적합도 계산
    
    조건: direction=Bullish + ADX 높음 + 인터벌 정렬
    """
    score = 0.0
    
    # 방향성 확인 (40%)
    dir_lower = direction.lower() if direction else 'neutral'
    if 'bull' in dir_lower or 'long' in dir_lower or 'up' in dir_lower:
        score += 0.4
    elif 'neutral' in dir_lower:
        score += 0.1
    
    # ADX 추세 강도 (30%)
    adx = adx or 25
    if adx > 40:
        score += 0.3
    elif adx > 30:
        score += 0.2
    elif adx > 25:
        score += 0.1
    
    # 인터벌 정렬도 (20%)
    score += signal_continuity * 0.2
    
    # MACD 추세 확인 (10%)
    if macd > 0.001:
        score += 0.1
    elif macd > 0:
        score += 0.05
    
    return {"match": round(score, 3), "strategy": "trend"}


def calc_bottom_match(rsi: float, wave: str, direction: str, 
                      macd: float = 0, volume_ratio: float = 1.0) -> Dict[str, float]:
    """
    저점 매수 전략 적합도 계산
    
    조건: RSI < 35 + wave=accumulation + 방향 전환 조짐
    """
    score = 0.0
    wave_lower = (wave or 'unknown').lower()
    dir_lower = (direction or 'neutral').lower()
    rsi = rsi or 50
    
    # RSI 과매도 (40%)
    if rsi < 25:
        score += 0.4
    elif rsi < 30:
        score += 0.35
    elif rsi < 35:
        score += 0.25
    elif rsi < 40:
        score += 0.1
    
    # Wave Phase (30%)
    if wave_lower == 'accumulation':
        score += 0.3
    elif wave_lower == 'markdown':
        score += 0.15  # 하락 중이지만 반등 가능
    elif wave_lower == 'sideways':
        score += 0.1
    
    # 방향 확인 (20%) - 아직 상승 안 했지만 바닥 근처
    if 'neutral' in dir_lower:
        score += 0.2   # 중립이면 반등 기대
    elif 'bear' in dir_lower:
        score += 0.15  # 하락 중이지만 저점 가능
    elif 'bull' in dir_lower:
        score += 0.05  # 이미 상승 시작
    
    # MACD 반등 조짐 (10%)
    if macd > -0.001 and macd < 0.002:  # 0 근처에서 상향 전환
        score += 0.1
    elif macd > -0.005:
        score += 0.05
    
    return {"match": round(score, 3), "strategy": "bottom"}


def calc_scalp_match(volume_ratio: float, candle_data: Dict, 
                     rsi: float = 50, macd: float = 0) -> Dict[str, float]:
    """
    급등 스캘핑 전략 적합도 계산
    
    조건: volume > 2.5x + 단기 급등 + RSI 과열 아님
    """
    score = 0.0
    volume_ratio = volume_ratio or 1.0
    rsi = rsi or 50
    
    # 거래량 급증 (40%)
    if volume_ratio > 3.0:
        score += 0.4
    elif volume_ratio > 2.5:
        score += 0.35
    elif volume_ratio > 2.0:
        score += 0.25
    elif volume_ratio > 1.5:
        score += 0.1
    
    # 단기 급등 (30%) - 캔들 데이터에서 계산
    recent_change = candle_data.get('recent_change_pct', 0)
    if recent_change > 3.0:
        score += 0.3
    elif recent_change > 2.0:
        score += 0.25
    elif recent_change > 1.0:
        score += 0.15
    elif recent_change > 0.5:
        score += 0.05
    
    # RSI 과열 아님 (20%) - 너무 높으면 스캘핑 위험
    if 40 <= rsi <= 70:
        score += 0.2
    elif 30 <= rsi <= 75:
        score += 0.1
    elif rsi > 75:
        score += 0.0  # 과열 시 감점
    
    # 상승 모멘텀 (10%)
    if macd > 0.002:
        score += 0.1
    elif macd > 0:
        score += 0.05
    
    return {"match": round(score, 3), "strategy": "scalp"}


def calc_swing_match(wave: str, candle_data: Dict, 
                     direction: str = 'neutral', adx: float = 25) -> Dict[str, float]:
    """
    스윙 트레이딩 전략 적합도 계산
    
    조건: wave 전환 감지 (accumulation → markup)
    """
    score = 0.0
    wave_lower = (wave or 'unknown').lower()
    
    # Wave Phase 전환 (50%)
    wave_transition = candle_data.get('wave_transition', '')
    if 'accumulation_to_markup' in wave_transition:
        score += 0.5
    elif wave_lower == 'markup' and candle_data.get('wave_progress', 0) < 0.3:
        score += 0.4   # markup 초기 단계
    elif wave_lower == 'accumulation':
        score += 0.25  # 전환 대기
    elif wave_lower == 'markup':
        score += 0.15  # 이미 진행 중
    
    # ADX 추세 형성 (25%)
    adx = adx or 25
    if 25 <= adx <= 40:  # 적당한 추세 (너무 강하면 스윙 어려움)
        score += 0.25
    elif 20 <= adx <= 50:
        score += 0.15
    
    # 방향 일치 (25%)
    dir_lower = (direction or 'neutral').lower()
    if 'bull' in dir_lower:
        score += 0.25
    elif 'neutral' in dir_lower:
        score += 0.1
    
    return {"match": round(score, 3), "strategy": "swing"}


def calc_revert_match(rsi: float, pattern: str, adx: float, 
                      volume_ratio: float = 1.0) -> Dict[str, float]:
    """
    평균 회귀 전략 적합도 계산
    
    조건: RSI 극단값 + sideways + 낮은 ADX
    """
    score = 0.0
    pattern_lower = (pattern or 'unknown').lower()
    rsi = rsi or 50
    adx = adx or 25
    
    # RSI 극단값 (40%)
    if rsi < 25 or rsi > 75:
        score += 0.4
    elif rsi < 30 or rsi > 70:
        score += 0.3
    elif rsi < 35 or rsi > 65:
        score += 0.15
    
    # 횡보 패턴 (30%)
    if 'sideways' in pattern_lower or 'range' in pattern_lower:
        score += 0.3
    elif 'consolidation' in pattern_lower:
        score += 0.2
    
    # 낮은 ADX (30%) - 추세 없음 = 회귀 가능성
    if adx < 20:
        score += 0.3
    elif adx < 25:
        score += 0.2
    elif adx < 30:
        score += 0.1
    
    return {"match": round(score, 3), "strategy": "revert"}


def calc_breakout_match(pattern: str, volume_ratio: float, direction: str,
                        adx: float = 25, candle_data: Dict = None) -> Dict[str, float]:
    """
    브레이크아웃 전략 적합도 계산
    
    조건: sideways 후 + volume 급증 + 방향 발생
    """
    score = 0.0
    pattern_lower = (pattern or 'unknown').lower()
    dir_lower = (direction or 'neutral').lower()
    volume_ratio = volume_ratio or 1.0
    candle_data = candle_data or {}
    
    # 이전 횡보 + 현재 돌파 (40%)
    was_sideways = candle_data.get('was_sideways', False)
    if was_sideways and ('uptrend' in pattern_lower or 'bull' in dir_lower):
        score += 0.4
    elif 'sideways' in pattern_lower and volume_ratio > 2.0:
        score += 0.3  # 돌파 직전
    elif was_sideways:
        score += 0.15
    
    # 거래량 급증 (30%)
    if volume_ratio > 2.5:
        score += 0.3
    elif volume_ratio > 2.0:
        score += 0.25
    elif volume_ratio > 1.5:
        score += 0.15
    
    # 방향 발생 (30%)
    if 'bull' in dir_lower or 'up' in dir_lower:
        score += 0.3
    elif 'neutral' not in dir_lower:
        score += 0.15  # 어떤 방향이든 발생
    
    return {"match": round(score, 3), "strategy": "breakout"}


def calc_dca_match(direction: str, rsi: float, interval: str,
                   signal_score: float = 0, existing_position: bool = False) -> Dict[str, float]:
    """
    분할 매수 (DCA) 전략 적합도 계산
    
    조건: 장기 방향 상승 + 단기 하락 시 추가 매수
    """
    score = 0.0
    dir_lower = (direction or 'neutral').lower()
    rsi = rsi or 50
    interval_lower = (interval or '15m').lower()
    
    # 장기 방향 상승 (35%)
    if 'bull' in dir_lower:
        score += 0.35
    elif 'neutral' in dir_lower:
        score += 0.2
    
    # 단기 조정 (RSI) (35%)
    if 35 <= rsi <= 45:  # 조정 구간
        score += 0.35
    elif 30 <= rsi <= 50:
        score += 0.25
    elif rsi < 30:
        score += 0.15  # 과매도는 저점매수가 나음
    
    # 인터벌 (15%) - 장기 인터벌에서 더 적합
    if '1d' in interval_lower or '240' in interval_lower:
        score += 0.15
    elif '30' in interval_lower:
        score += 0.1
    
    # 기존 포지션 존재 시 추가 매수 적합 (15%)
    if existing_position:
        score += 0.15
    
    return {"match": round(score, 3), "strategy": "dca"}


def calc_momentum_match(adx: float, direction: str, volume_ratio: float,
                        macd: float = 0, rsi: float = 50) -> Dict[str, float]:
    """
    모멘텀 전략 적합도 계산
    
    조건: ADX > 40 + 강한 방향 + 거래량 증가
    """
    score = 0.0
    dir_lower = (direction or 'neutral').lower()
    adx = adx or 25
    volume_ratio = volume_ratio or 1.0
    rsi = rsi or 50
    
    # ADX 강한 추세 (40%)
    if adx > 50:
        score += 0.4
    elif adx > 40:
        score += 0.35
    elif adx > 35:
        score += 0.25
    elif adx > 30:
        score += 0.1
    
    # 방향 일치 (30%)
    if 'bull' in dir_lower or 'long' in dir_lower:
        score += 0.3
    elif 'bear' in dir_lower or 'short' in dir_lower:
        score += 0.25  # 숏 모멘텀도 가능
    
    # 거래량 (20%)
    if volume_ratio > 2.0:
        score += 0.2
    elif volume_ratio > 1.5:
        score += 0.15
    elif volume_ratio > 1.2:
        score += 0.1
    
    # MACD 강도 (10%)
    if abs(macd) > 0.005:
        score += 0.1
    elif abs(macd) > 0.002:
        score += 0.05
    
    return {"match": round(score, 3), "strategy": "momentum"}


def calc_counter_match(rsi: float, candle_data: Dict, 
                       volume_ratio: float = 1.0, adx: float = 25) -> Dict[str, float]:
    """
    역추세 전략 적합도 계산
    
    조건: RSI 과열 + 다이버전스 징후 + 거래량 감소
    """
    score = 0.0
    rsi = rsi or 50
    volume_ratio = volume_ratio or 1.0
    candle_data = candle_data or {}
    
    # RSI 과열 (40%)
    if rsi > 80:
        score += 0.4
    elif rsi > 75:
        score += 0.35
    elif rsi > 70:
        score += 0.25
    elif rsi < 20:
        score += 0.35  # 과매도에서 반등 기대
    elif rsi < 25:
        score += 0.25
    
    # 다이버전스 (30%)
    has_divergence = candle_data.get('has_divergence', False)
    if has_divergence:
        score += 0.3
    
    # 거래량 감소 (20%) - 추세 약화 신호
    if volume_ratio < 0.7:
        score += 0.2
    elif volume_ratio < 0.9:
        score += 0.15
    elif volume_ratio < 1.0:
        score += 0.1
    
    # ADX 약화 (10%)
    adx = adx or 25
    adx_declining = candle_data.get('adx_declining', False)
    if adx_declining:
        score += 0.1
    
    return {"match": round(score, 3), "strategy": "counter"}


def calc_range_match(adx: float, pattern: str, candle_data: Dict,
                     rsi: float = 50, volume_ratio: float = 1.0) -> Dict[str, float]:
    """
    레인지 트레이딩 전략 적합도 계산
    
    조건: ADX < 20 (횡보) + 지지/저항선 근처
    """
    score = 0.0
    pattern_lower = (pattern or 'unknown').lower()
    adx = adx or 25
    rsi = rsi or 50
    candle_data = candle_data or {}
    
    # 낮은 ADX (40%)
    if adx < 15:
        score += 0.4
    elif adx < 20:
        score += 0.35
    elif adx < 25:
        score += 0.25
    elif adx < 30:
        score += 0.1
    
    # 횡보 패턴 (30%)
    if 'sideways' in pattern_lower or 'range' in pattern_lower:
        score += 0.3
    elif 'consolidation' in pattern_lower:
        score += 0.2
    
    # 지지/저항 근처 (20%)
    near_support = candle_data.get('near_support', False)
    near_resistance = candle_data.get('near_resistance', False)
    if near_support:
        score += 0.2   # 지지선 근처 = 매수
    elif near_resistance:
        score += 0.15  # 저항선 근처 = 매도 (또는 관망)
    
    # RSI 중립 구간 (10%)
    if 40 <= rsi <= 60:
        score += 0.1
    
    return {"match": round(score, 3), "strategy": "range"}


# ============================================================================
# 통합 전략 평가 함수
# ============================================================================
def evaluate_all_strategies(signal_data: Dict, candle_data: Dict = None) -> Dict[str, Dict]:
    """
    모든 전략의 적합도를 평가
    
    Args:
        signal_data: 시그널 정보 (rsi, macd, wave_phase, direction 등)
        candle_data: 캔들 데이터 (최근 변화율, 지지/저항 등)
    
    Returns:
        전략별 적합도 딕셔너리
        {
            'trend': {'match': 0.65, 'strategy': 'trend'},
            'bottom': {'match': 0.82, 'strategy': 'bottom'},
            ...
        }
    """
    candle_data = candle_data or {}
    
    # 시그널 데이터 추출
    rsi = signal_data.get('rsi') or 50
    macd = signal_data.get('macd') or 0
    adx = signal_data.get('adx') or 25
    volume_ratio = signal_data.get('volume_ratio') or 1.0
    wave = signal_data.get('wave_phase') or 'unknown'
    pattern = signal_data.get('pattern_type') or 'unknown'
    direction = signal_data.get('integrated_direction') or 'neutral'
    signal_continuity = signal_data.get('signal_continuity') or 0.5
    interval = signal_data.get('interval') or '15m'
    signal_score = signal_data.get('signal_score') or 0
    existing_position = signal_data.get('existing_position', False)
    
    results = {}
    
    # 1. 추세 추종
    results['trend'] = calc_trend_match(direction, adx, signal_continuity, macd, rsi)
    
    # 2. 저점 매수
    results['bottom'] = calc_bottom_match(rsi, wave, direction, macd, volume_ratio)
    
    # 3. 급등 스캘핑
    results['scalp'] = calc_scalp_match(volume_ratio, candle_data, rsi, macd)
    
    # 4. 스윙 트레이딩
    results['swing'] = calc_swing_match(wave, candle_data, direction, adx)
    
    # 5. 평균 회귀
    results['revert'] = calc_revert_match(rsi, pattern, adx, volume_ratio)
    
    # 6. 브레이크아웃
    results['breakout'] = calc_breakout_match(pattern, volume_ratio, direction, adx, candle_data)
    
    # 7. 분할 매수
    results['dca'] = calc_dca_match(direction, rsi, interval, signal_score, existing_position)
    
    # 8. 모멘텀
    results['momentum'] = calc_momentum_match(adx, direction, volume_ratio, macd, rsi)
    
    # 9. 역추세
    results['counter'] = calc_counter_match(rsi, candle_data, volume_ratio, adx)
    
    # 10. 레인지
    results['range'] = calc_range_match(adx, pattern, candle_data, rsi, volume_ratio)
    
    return results


def get_top_strategies(strategy_scores: Dict[str, Dict], 
                       top_n: int = 3, 
                       min_match: float = 0.3) -> List[Dict]:
    """
    상위 N개 전략 반환
    
    Args:
        strategy_scores: evaluate_all_strategies() 결과
        top_n: 반환할 전략 수
        min_match: 최소 적합도
    
    Returns:
        상위 전략 리스트 [{'strategy': 'bottom', 'match': 0.82}, ...]
    """
    filtered = [
        {'strategy': k, 'match': v['match']}
        for k, v in strategy_scores.items()
        if v['match'] >= min_match
    ]
    
    sorted_strategies = sorted(filtered, key=lambda x: x['match'], reverse=True)
    return sorted_strategies[:top_n]


def get_strategy_description(strategy_type: str) -> str:
    """전략 설명 반환"""
    rules = STRATEGY_EXIT_RULES.get(strategy_type)
    if rules:
        return rules.description
    return f"Unknown strategy: {strategy_type}"


def get_exit_rules(strategy_type: str) -> StrategyExitRules:
    """전략별 청산 규칙 반환"""
    return STRATEGY_EXIT_RULES.get(strategy_type, STRATEGY_EXIT_RULES['trend'])


# ============================================================================
# 전략 피드백 DB 관련 함수
# ============================================================================
def create_strategy_feedback_table(db_path: str):
    """전략 피드백 테이블 생성"""
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_type TEXT NOT NULL,
                market_condition TEXT DEFAULT 'unknown',
                signal_pattern TEXT DEFAULT 'unknown',
                
                -- Thompson Sampling 파라미터
                alpha INTEGER DEFAULT 1,
                beta INTEGER DEFAULT 1,
                
                -- 성과 통계
                total_trades INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.5,
                avg_profit REAL DEFAULT 0.0,
                avg_holding_hours REAL DEFAULT 0.0,
                max_profit REAL DEFAULT 0.0,
                max_loss REAL DEFAULT 0.0,
                
                -- 메타 정보
                last_updated INTEGER,
                created_at INTEGER,
                
                UNIQUE(strategy_type, market_condition, signal_pattern)
            )
        """)
        
        # 인덱스 생성
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_feedback_type ON strategy_feedback(strategy_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_feedback_market ON strategy_feedback(market_condition)")
        conn.commit()
        
        print("✅ strategy_feedback 테이블 생성/확인 완료")


def update_strategy_feedback(db_path: str, strategy_type: str, market_condition: str,
                             signal_pattern: str, success: bool, profit_pct: float,
                             holding_hours: float = 0, feedback_type: str = 'trade'):
    """
    전략 피드백 업데이트 (Thompson Sampling)
    
    Args:
        feedback_type: 'entry' (진입 정확도), 'exit' (청산 정확도), 
                       'switch' (전환 성공률), 'trade' (전체 매매)
    """
    now = int(time.time())
    
    # feedback_type을 signal_pattern에 포함하여 분리 저장
    full_pattern = f"{signal_pattern}_{feedback_type}" if feedback_type != 'trade' else signal_pattern
    
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        # 🆕 feedback_type 컬럼 마이그레이션
        try:
            cursor = conn.execute("PRAGMA table_info(strategy_feedback)")
            cols = [c[1] for c in cursor.fetchall()]
            if 'feedback_type' not in cols:
                conn.execute("ALTER TABLE strategy_feedback ADD COLUMN feedback_type TEXT DEFAULT 'trade'")
                conn.commit()
        except:
            pass
        
        # 기존 레코드 확인
        cursor = conn.execute("""
            SELECT alpha, beta, total_trades, success_count, avg_profit, avg_holding_hours,
                   max_profit, max_loss
            FROM strategy_feedback
            WHERE strategy_type = ? AND market_condition = ? AND signal_pattern = ?
        """, (strategy_type, market_condition, full_pattern))
        
        row = cursor.fetchone()
        
        if row:
            alpha, beta, total, success_cnt, avg_profit, avg_holding, max_profit, max_loss = row
            
            # Thompson Sampling 업데이트
            if success:
                alpha += 1
                success_cnt += 1
            else:
                beta += 1
            
            total += 1
            new_avg_profit = (avg_profit * (total - 1) + profit_pct) / total
            new_avg_holding = (avg_holding * (total - 1) + holding_hours) / total
            new_max_profit = max(max_profit, profit_pct)
            new_max_loss = min(max_loss, profit_pct)
            new_success_rate = alpha / (alpha + beta)
            
            conn.execute("""
                UPDATE strategy_feedback SET
                    alpha = ?, beta = ?, total_trades = ?, success_count = ?,
                    success_rate = ?, avg_profit = ?, avg_holding_hours = ?,
                    max_profit = ?, max_loss = ?, last_updated = ?, feedback_type = ?
                WHERE strategy_type = ? AND market_condition = ? AND signal_pattern = ?
            """, (alpha, beta, total, success_cnt, new_success_rate, new_avg_profit, 
                  new_avg_holding, new_max_profit, new_max_loss, now, feedback_type,
                  strategy_type, market_condition, full_pattern))
        else:
            # 새 레코드 삽입
            alpha = 2 if success else 1
            beta = 1 if success else 2
            
            conn.execute("""
                INSERT INTO strategy_feedback 
                (strategy_type, market_condition, signal_pattern, alpha, beta,
                 total_trades, success_count, success_rate, avg_profit, avg_holding_hours,
                 max_profit, max_loss, last_updated, created_at, feedback_type)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (strategy_type, market_condition, full_pattern, alpha, beta,
                  1 if success else 0, alpha / (alpha + beta), profit_pct, holding_hours,
                  profit_pct, profit_pct, now, now, feedback_type))
        
        conn.commit()


def get_strategy_success_rate(db_path: str, strategy_type: str, 
                              market_condition: str = None,
                              signal_pattern: str = None) -> Tuple[float, float]:
    """
    전략의 학습된 성공률 조회 (Thompson Sampling)
    
    Returns:
        (sampled_rate, confidence) - 샘플링된 성공률과 신뢰도
    """
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            # 가장 구체적인 매칭부터 시도
            queries = []
            
            if market_condition and signal_pattern:
                queries.append(("""
                    SELECT alpha, beta, total_trades FROM strategy_feedback
                    WHERE strategy_type = ? AND market_condition = ? AND signal_pattern = ?
                """, (strategy_type, market_condition, signal_pattern)))
            
            if market_condition:
                queries.append(("""
                    SELECT SUM(alpha), SUM(beta), SUM(total_trades) FROM strategy_feedback
                    WHERE strategy_type = ? AND market_condition = ?
                """, (strategy_type, market_condition)))
            
            queries.append(("""
                SELECT SUM(alpha), SUM(beta), SUM(total_trades) FROM strategy_feedback
                WHERE strategy_type = ?
            """, (strategy_type,)))
            
            for query, params in queries:
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
                
                if row and row[0] and row[1]:
                    alpha, beta, total = row[0], row[1], row[2] or 0
                    
                    # Thompson Sampling: Beta 분포에서 샘플링
                    sampled_rate = np.random.beta(alpha, beta)
                    
                    # 신뢰도: 데이터 축적량 기반
                    confidence = min(1.0, total / 50.0)  # 50회 이상이면 신뢰도 1.0
                    
                    return (round(sampled_rate, 3), round(confidence, 2))
            
            # 데이터 없음 - 기본값 + 탐색 보너스
            return (0.5 + np.random.uniform(-0.1, 0.1), 0.1)
            
    except Exception as e:
        print(f"⚠️ 전략 성공률 조회 오류: {e}")
        return (0.5, 0.1)


def get_market_strategy_preference(db_path: str, market_condition: str) -> Dict[str, float]:
    """
    시장 조건별 전략 선호도 반환
    
    Returns:
        {strategy_type: preference_score, ...}
    """
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.execute("""
                SELECT strategy_type, 
                       SUM(success_count) as wins,
                       SUM(total_trades) as total,
                       AVG(avg_profit) as avg_profit
                FROM strategy_feedback
                WHERE market_condition = ?
                GROUP BY strategy_type
                ORDER BY wins DESC
            """, (market_condition,))
            
            results = {}
            for row in cursor.fetchall():
                strategy, wins, total, avg_profit = row
                if total > 0:
                    win_rate = wins / total
                    # 선호도 = 승률 * 0.6 + 정규화된 수익률 * 0.4
                    profit_factor = max(0, min(1, (avg_profit + 10) / 20))  # -10% ~ +10% → 0 ~ 1
                    preference = win_rate * 0.6 + profit_factor * 0.4
                    results[strategy] = round(preference, 3)
            
            return results
            
    except Exception as e:
        print(f"⚠️ 시장 전략 선호도 조회 오류: {e}")
        return {}


# ============================================================================
# 전략 선택 통합 함수
# ============================================================================
def select_best_strategies(signal_data: Dict, candle_data: Dict,
                           db_path: str, market_condition: str = 'unknown',
                           top_n: int = 3) -> List[Dict]:
    """
    최적의 전략을 선택 (규칙 기반 + 학습 기반 혼합)
    
    Args:
        signal_data: 시그널 정보
        candle_data: 캔들 데이터
        db_path: 학습 DB 경로
        market_condition: 현재 시장 상태
        top_n: 반환할 전략 수
    
    Returns:
        [
            {
                'strategy': 'bottom',
                'match': 0.82,          # 규칙 기반 적합도
                'learned_rate': 0.71,   # 학습된 성공률
                'confidence': 0.8,       # 학습 신뢰도
                'final_score': 0.65,    # 최종 점수 (match * learned_rate * confidence_factor)
                'threshold': 0.35,      # 진입 임계값
                'should_enter': True,   # 진입 여부
            },
            ...
        ]
    """
    # 1. 규칙 기반 적합도 계산
    strategy_scores = evaluate_all_strategies(signal_data, candle_data)
    
    # 2. 시장 상태별 선호도 가져오기
    market_prefs = get_market_strategy_preference(db_path, market_condition)
    
    # 3. 각 전략에 학습된 성공률 추가
    results = []
    signal_pattern = signal_data.get('pattern', 'unknown')
    
    for strategy_type, score_data in strategy_scores.items():
        match_score = score_data['match']
        
        # 학습된 성공률 조회
        learned_rate, confidence = get_strategy_success_rate(
            db_path, strategy_type, market_condition, signal_pattern
        )
        
        # 시장 선호도 반영
        market_pref = market_prefs.get(strategy_type, 0.5)
        
        # 최종 점수 계산
        # = 규칙 적합도(40%) + 학습 성공률(40%) + 시장 선호도(20%)
        # × 신뢰도 가중치
        confidence_weight = 0.5 + (confidence * 0.5)  # 0.5 ~ 1.0
        
        final_score = (
            match_score * 0.4 +
            learned_rate * 0.4 +
            market_pref * 0.2
        ) * confidence_weight
        
        # 진입 임계값 확인
        threshold = STRATEGY_ENTRY_THRESHOLDS.get(strategy_type, 0.4)
        should_enter = final_score >= threshold and match_score >= 0.25
        
        results.append({
            'strategy': strategy_type,
            'match': match_score,
            'learned_rate': learned_rate,
            'confidence': confidence,
            'market_pref': market_pref,
            'final_score': round(final_score, 3),
            'threshold': threshold,
            'should_enter': should_enter,
            'exit_rules': asdict(get_exit_rules(strategy_type)),
            'description': get_strategy_description(strategy_type),
        })
    
    # 4. 최종 점수로 정렬
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return results[:top_n]


# ============================================================================
# 유틸리티 함수
# ============================================================================
def print_strategy_summary(strategies: List[Dict]):
    """전략 선택 결과 출력"""
    print("\n" + "=" * 70)
    print("🎯 전략 선택 결과")
    print("=" * 70)
    
    for i, s in enumerate(strategies, 1):
        enter_mark = "✅" if s['should_enter'] else "❌"
        print(f"\n{i}. [{s['strategy'].upper()}] {enter_mark}")
        print(f"   📊 규칙 적합도: {s['match']:.2f}")
        print(f"   📈 학습 성공률: {s['learned_rate']:.2f} (신뢰도: {s['confidence']:.2f})")
        print(f"   🎯 최종 점수: {s['final_score']:.3f} (임계값: {s['threshold']:.2f})")
        print(f"   💡 {s['description']}")
    
    print("\n" + "=" * 70)


def get_all_strategy_types() -> List[str]:
    """모든 전략 타입 반환"""
    return StrategyType.all_types()


def serialize_strategy_scores(strategy_scores: Dict[str, Dict]) -> str:
    """전략 점수를 JSON 문자열로 직렬화"""
    simplified = {k: v['match'] for k, v in strategy_scores.items()}
    return json.dumps(simplified)


def deserialize_strategy_scores(json_str: str) -> Dict[str, float]:
    """JSON 문자열에서 전략 점수 복원"""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}


# ============================================================================
# 🆕 전략별 검증 함수 (각 전략마다 성공/실패 기준이 다름!)
# ============================================================================
@dataclass
class StrategyValidationResult:
    """전략별 검증 결과"""
    strategy_type: str
    is_success: bool
    profit_pct: float
    validation_reason: str
    validation_horizon: str  # 'short', 'mid', 'long'
    confidence: float = 1.0  # 검증 신뢰도 (데이터 충분성)


def validate_trend_strategy(entry_price: float, candle_window: 'pd.DataFrame', 
                            is_long: bool, target_pct: float = 15.0) -> StrategyValidationResult:
    """
    추세 추종 전략 검증
    
    성공 기준: 추세가 지속되어 목표가 도달 OR 트레일링 스탑 수준 달성
    실패 기준: 추세 반전 (손절선 도달)
    """
    if candle_window.empty:
        return StrategyValidationResult('trend', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    low_min = candle_window['low'].min()
    final_close = candle_window['close'].iloc[-1]
    
    if is_long:
        max_profit = ((high_max - entry_price) / entry_price) * 100
        final_profit = ((final_close - entry_price) / entry_price) * 100
        max_drawdown = ((low_min - entry_price) / entry_price) * 100
        
        # 성공: 목표가 달성 또는 트레일링 스탑 트리거 후 수익 실현
        if max_profit >= target_pct:
            return StrategyValidationResult('trend', True, final_profit, f'목표가 달성 (+{max_profit:.1f}%)', 'long', 1.0)
        elif max_profit >= 5.0 and final_profit >= max_profit * 0.7:
            return StrategyValidationResult('trend', True, final_profit, f'트레일링 수익 실현 (+{final_profit:.1f}%)', 'mid', 0.8)
        elif max_drawdown <= -5.0:
            return StrategyValidationResult('trend', False, final_profit, f'추세 반전 손절 ({max_drawdown:.1f}%)', 'short', 1.0)
        else:
            # 아직 진행 중
            return StrategyValidationResult('trend', final_profit > 0, final_profit, '추세 진행 중', 'mid', 0.5)
    else:
        # 숏 포지션 (역방향)
        max_profit = ((entry_price - low_min) / entry_price) * 100
        final_profit = ((entry_price - final_close) / entry_price) * 100
        return StrategyValidationResult('trend', max_profit >= target_pct * 0.5, final_profit, '숏 추세', 'mid', 0.7)


def validate_bottom_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                             max_holding_hours: float = 336) -> StrategyValidationResult:
    """
    저점 매수 전략 검증
    
    성공 기준: 결국 entry_price보다 상승 (시간이 오래 걸려도 OK)
    실패 기준: 더 큰 하락 후 회복 안 됨 (진짜 바닥이 아니었음)
    
    핵심: 시간보다 "최종 결과"가 중요!
    """
    if candle_window.empty:
        return StrategyValidationResult('bottom', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    low_min = candle_window['low'].min()
    final_close = candle_window['close'].iloc[-1]
    
    max_profit = ((high_max - entry_price) / entry_price) * 100
    final_profit = ((final_close - entry_price) / entry_price) * 100
    max_drawdown = ((low_min - entry_price) / entry_price) * 100
    
    # 저점 매수는 "결국 올라갔느냐"가 핵심
    if final_profit >= 10.0:
        return StrategyValidationResult('bottom', True, final_profit, f'저점 반등 성공 (+{final_profit:.1f}%)', 'long', 1.0)
    elif max_profit >= 20.0:
        return StrategyValidationResult('bottom', True, max_profit * 0.7, f'고점 달성 후 조정 (+{max_profit:.1f}% 달성)', 'long', 0.9)
    elif max_drawdown <= -15.0 and final_profit < 0:
        # 더 큰 하락 = 저점이 아니었음
        return StrategyValidationResult('bottom', False, final_profit, f'진짜 저점 아님 (추가 하락 {max_drawdown:.1f}%)', 'long', 1.0)
    elif final_profit > 0:
        return StrategyValidationResult('bottom', True, final_profit, f'소폭 반등 (+{final_profit:.1f}%)', 'mid', 0.7)
    else:
        # 아직 회복 안 됨 - 판단 보류
        return StrategyValidationResult('bottom', False, final_profit, '회복 대기 중', 'mid', 0.3)


def validate_scalp_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                            target_pct: float = 1.5, max_hours: float = 4) -> StrategyValidationResult:
    """
    스캘핑 전략 검증
    
    성공 기준: 빠르게 목표가 달성 (4시간 이내)
    실패 기준: 손절선 도달 OR 시간 초과
    
    핵심: "빠른 수익 실현"이 핵심! 시간이 오래 걸리면 실패
    """
    if candle_window.empty:
        return StrategyValidationResult('scalp', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    low_min = candle_window['low'].min()
    final_close = candle_window['close'].iloc[-1]
    
    max_profit = ((high_max - entry_price) / entry_price) * 100
    final_profit = ((final_close - entry_price) / entry_price) * 100
    max_drawdown = ((low_min - entry_price) / entry_price) * 100
    
    # 캔들 수로 시간 추정 (15분봉 기준)
    num_candles = len(candle_window)
    
    # 빠른 성공 (처음 몇 개 캔들 안에 달성)
    if num_candles <= 16:  # 4시간 (15분봉 16개)
        if max_profit >= target_pct:
            return StrategyValidationResult('scalp', True, target_pct, f'빠른 수익 실현 (+{max_profit:.2f}%)', 'short', 1.0)
        elif max_drawdown <= -1.0:
            return StrategyValidationResult('scalp', False, max_drawdown, f'빠른 손절 ({max_drawdown:.2f}%)', 'short', 1.0)
    
    # 시간 초과 (스캘핑 실패)
    if num_candles > 16:
        if final_profit > 0:
            return StrategyValidationResult('scalp', False, final_profit, f'시간 초과 (스캘핑 실패, +{final_profit:.2f}%)', 'mid', 0.8)
        else:
            return StrategyValidationResult('scalp', False, final_profit, f'시간 초과 & 손실 ({final_profit:.2f}%)', 'mid', 1.0)
    
    return StrategyValidationResult('scalp', False, final_profit, '진행 중', 'short', 0.3)


def validate_swing_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                            wave_phase_series: list = None) -> StrategyValidationResult:
    """
    스윙 전략 검증
    
    성공 기준: 파동의 상당 부분 캡처 (markup 전체 또는 대부분)
    실패 기준: 파동 초반에 손절
    
    핵심: 파동 사이클 전체를 탄 것인지 확인
    """
    if candle_window.empty:
        return StrategyValidationResult('swing', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    low_min = candle_window['low'].min()
    final_close = candle_window['close'].iloc[-1]
    
    max_profit = ((high_max - entry_price) / entry_price) * 100
    final_profit = ((final_close - entry_price) / entry_price) * 100
    
    # 스윙은 중간 정도 수익을 목표
    if max_profit >= 15.0:
        if final_profit >= max_profit * 0.6:
            return StrategyValidationResult('swing', True, final_profit, f'파동 캡처 성공 (+{final_profit:.1f}%)', 'long', 1.0)
        else:
            return StrategyValidationResult('swing', True, final_profit, f'파동 고점 후 조정 (+{max_profit:.1f}% → +{final_profit:.1f}%)', 'long', 0.7)
    elif max_profit >= 8.0:
        return StrategyValidationResult('swing', True, final_profit, f'소규모 스윙 성공 (+{max_profit:.1f}%)', 'mid', 0.8)
    elif final_profit <= -6.0:
        return StrategyValidationResult('swing', False, final_profit, f'스윙 손절 ({final_profit:.1f}%)', 'short', 1.0)
    else:
        return StrategyValidationResult('swing', final_profit > 0, final_profit, '스윙 진행 중', 'mid', 0.5)


def validate_revert_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                             entry_rsi: float = 50) -> StrategyValidationResult:
    """
    평균 회귀 전략 검증
    
    성공 기준: 극단값에서 평균(RSI 50 근처)으로 복귀
    실패 기준: 극단으로 더 갔음 (회귀 실패)
    
    핵심: "방향 전환" 확인
    """
    if candle_window.empty:
        return StrategyValidationResult('revert', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    final_close = candle_window['close'].iloc[-1]
    final_profit = ((final_close - entry_price) / entry_price) * 100
    
    # 과매도에서 진입했으면 (RSI < 30) 상승 기대
    # 과매수에서 진입했으면 (RSI > 70) 하락 기대 (숏 또는 매도)
    expected_up = entry_rsi < 40
    
    if expected_up:
        if final_profit >= 3.0:
            return StrategyValidationResult('revert', True, final_profit, f'평균 회귀 성공 (+{final_profit:.1f}%)', 'short', 1.0)
        elif final_profit <= -5.0:
            return StrategyValidationResult('revert', False, final_profit, f'회귀 실패 (더 하락 {final_profit:.1f}%)', 'short', 1.0)
    else:
        # 과매수 상태에서 숏/매도 기대
        if final_profit <= -3.0:  # 하락 = 숏 성공
            return StrategyValidationResult('revert', True, -final_profit, f'과매수 회귀 성공', 'short', 1.0)
    
    return StrategyValidationResult('revert', abs(final_profit) < 2.0, final_profit, '회귀 진행 중', 'short', 0.5)


def validate_breakout_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                               was_sideways: bool = True) -> StrategyValidationResult:
    """
    돌파 전략 검증
    
    성공 기준: 돌파 후 추세 지속 (거짓 돌파 아님)
    실패 기준: 되돌림 (거짓 돌파)
    
    핵심: 돌파 후 "지지/저항 전환" 확인
    """
    if candle_window.empty:
        return StrategyValidationResult('breakout', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    low_min = candle_window['low'].min()
    final_close = candle_window['close'].iloc[-1]
    
    max_profit = ((high_max - entry_price) / entry_price) * 100
    final_profit = ((final_close - entry_price) / entry_price) * 100
    max_drawdown = ((low_min - entry_price) / entry_price) * 100
    
    # 돌파 후 추세 지속 확인
    if max_profit >= 8.0 and final_profit >= 4.0:
        return StrategyValidationResult('breakout', True, final_profit, f'진짜 돌파 (+{final_profit:.1f}%)', 'mid', 1.0)
    elif max_drawdown <= -4.0:
        return StrategyValidationResult('breakout', False, final_profit, f'거짓 돌파 (되돌림 {max_drawdown:.1f}%)', 'short', 1.0)
    elif max_profit >= 5.0:
        return StrategyValidationResult('breakout', True, final_profit, f'돌파 진행 중 (+{max_profit:.1f}% 달성)', 'mid', 0.7)
    
    return StrategyValidationResult('breakout', final_profit > 0, final_profit, '돌파 확인 중', 'short', 0.4)


def validate_range_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                            support_price: float = 0, resistance_price: float = 0) -> StrategyValidationResult:
    """
    레인지 전략 검증
    
    성공 기준: 지지선에서 매수 → 저항선 근처에서 매도 (또는 그 반대)
    실패 기준: 박스권 이탈 (손절)
    
    핵심: 박스권 내 왕복 성공 여부
    """
    if candle_window.empty:
        return StrategyValidationResult('range', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    low_min = candle_window['low'].min()
    final_close = candle_window['close'].iloc[-1]
    
    max_profit = ((high_max - entry_price) / entry_price) * 100
    final_profit = ((final_close - entry_price) / entry_price) * 100
    max_drawdown = ((low_min - entry_price) / entry_price) * 100
    
    # 레인지는 소폭 수익 목표
    if max_profit >= 3.0:
        return StrategyValidationResult('range', True, min(final_profit, 4.0), f'레인지 반등 성공 (+{max_profit:.1f}%)', 'short', 1.0)
    elif max_drawdown <= -3.5:
        return StrategyValidationResult('range', False, final_profit, f'박스권 이탈 ({max_drawdown:.1f}%)', 'short', 1.0)
    elif abs(final_profit) < 2.0:
        return StrategyValidationResult('range', True, final_profit, '박스권 유지', 'short', 0.6)
    
    return StrategyValidationResult('range', final_profit > 0, final_profit, '레인지 진행 중', 'short', 0.5)


def validate_momentum_strategy(entry_price: float, candle_window: 'pd.DataFrame') -> StrategyValidationResult:
    """
    모멘텀 전략 검증
    
    성공 기준: 강한 추세에서 빠르게 수익 실현
    실패 기준: 모멘텀 소진 (급반전)
    """
    if candle_window.empty:
        return StrategyValidationResult('momentum', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    high_max = candle_window['high'].max()
    final_close = candle_window['close'].iloc[-1]
    
    max_profit = ((high_max - entry_price) / entry_price) * 100
    final_profit = ((final_close - entry_price) / entry_price) * 100
    
    if max_profit >= 8.0:
        return StrategyValidationResult('momentum', True, final_profit, f'모멘텀 성공 (+{max_profit:.1f}%)', 'mid', 1.0)
    elif final_profit <= -4.0:
        return StrategyValidationResult('momentum', False, final_profit, f'모멘텀 소진 ({final_profit:.1f}%)', 'short', 1.0)
    
    return StrategyValidationResult('momentum', final_profit > 0, final_profit, '모멘텀 진행 중', 'short', 0.5)


def validate_counter_strategy(entry_price: float, candle_window: 'pd.DataFrame',
                              entry_rsi: float = 50) -> StrategyValidationResult:
    """
    역추세 전략 검증
    
    성공 기준: 과열에서 반전 발생
    실패 기준: 과열이 더 심해짐 (추세 지속)
    """
    if candle_window.empty:
        return StrategyValidationResult('counter', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    final_close = candle_window['close'].iloc[-1]
    low_min = candle_window['low'].min()
    high_max = candle_window['high'].max()
    
    # 과매수에서 역추세 (하락 기대)
    if entry_rsi > 70:
        drop_pct = ((entry_price - low_min) / entry_price) * 100
        if drop_pct >= 5.0:
            return StrategyValidationResult('counter', True, drop_pct, f'과매수 반전 성공 (-{drop_pct:.1f}%)', 'short', 1.0)
        elif ((high_max - entry_price) / entry_price) * 100 >= 5.0:
            return StrategyValidationResult('counter', False, 0, '역추세 실패 (추세 지속)', 'short', 1.0)
    # 과매도에서 역추세 (상승 기대)
    elif entry_rsi < 30:
        final_profit = ((final_close - entry_price) / entry_price) * 100
        if final_profit >= 5.0:
            return StrategyValidationResult('counter', True, final_profit, f'과매도 반전 성공 (+{final_profit:.1f}%)', 'short', 1.0)
    
    return StrategyValidationResult('counter', False, 0, '반전 대기 중', 'short', 0.3)


def validate_dca_strategy(entry_price: float, avg_price: float, 
                          candle_window: 'pd.DataFrame') -> StrategyValidationResult:
    """
    분할 매수 (DCA) 전략 검증
    
    성공 기준: 평균 단가 기준 수익 실현
    실패 기준: 평균 단가도 회복 못함
    
    핵심: "평균 단가"가 기준!
    """
    if candle_window.empty:
        return StrategyValidationResult('dca', False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    final_close = candle_window['close'].iloc[-1]
    high_max = candle_window['high'].max()
    
    # 평균 단가 기준 수익률
    avg_profit = ((final_close - avg_price) / avg_price) * 100 if avg_price > 0 else 0
    max_profit = ((high_max - avg_price) / avg_price) * 100 if avg_price > 0 else 0
    
    if max_profit >= 15.0:
        return StrategyValidationResult('dca', True, avg_profit, f'DCA 성공 (+{max_profit:.1f}% 달성)', 'long', 1.0)
    elif avg_profit >= 5.0:
        return StrategyValidationResult('dca', True, avg_profit, f'평균 단가 수익 (+{avg_profit:.1f}%)', 'mid', 0.8)
    elif avg_profit <= -10.0:
        return StrategyValidationResult('dca', False, avg_profit, f'평균 단가 손실 ({avg_profit:.1f}%)', 'long', 1.0)
    
    return StrategyValidationResult('dca', avg_profit > 0, avg_profit, 'DCA 진행 중', 'mid', 0.5)


# ============================================================================
# 통합 검증 함수
# ============================================================================
def validate_strategy_signal(strategy_type: str, entry_price: float, 
                             candle_window: 'pd.DataFrame', **kwargs) -> StrategyValidationResult:
    """
    전략 타입에 따라 적절한 검증 함수 호출
    
    Args:
        strategy_type: 전략 타입
        entry_price: 진입가
        candle_window: 검증용 캔들 데이터 (시간순 정렬)
        **kwargs: 전략별 추가 파라미터 (is_long, entry_rsi, avg_price 등)
    
    Returns:
        StrategyValidationResult
    """
    is_long = kwargs.get('is_long', True)
    entry_rsi = kwargs.get('entry_rsi', 50)
    avg_price = kwargs.get('avg_price', entry_price)
    was_sideways = kwargs.get('was_sideways', False)
    
    validators = {
        'trend': lambda: validate_trend_strategy(entry_price, candle_window, is_long),
        'bottom': lambda: validate_bottom_strategy(entry_price, candle_window),
        'scalp': lambda: validate_scalp_strategy(entry_price, candle_window),
        'swing': lambda: validate_swing_strategy(entry_price, candle_window),
        'revert': lambda: validate_revert_strategy(entry_price, candle_window, entry_rsi),
        'breakout': lambda: validate_breakout_strategy(entry_price, candle_window, was_sideways),
        'dca': lambda: validate_dca_strategy(entry_price, avg_price, candle_window),
        'momentum': lambda: validate_momentum_strategy(entry_price, candle_window),
        'counter': lambda: validate_counter_strategy(entry_price, candle_window, entry_rsi),
        'range': lambda: validate_range_strategy(entry_price, candle_window),
    }
    
    validator = validators.get(strategy_type)
    if validator:
        return validator()
    
    # 기본값: 일반 수익률 기반 검증
    if candle_window.empty:
        return StrategyValidationResult(strategy_type, False, 0.0, '데이터 부족', 'unknown', 0.0)
    
    final_close = candle_window['close'].iloc[-1]
    final_profit = ((final_close - entry_price) / entry_price) * 100
    return StrategyValidationResult(strategy_type, final_profit > 0, final_profit, '기본 검증', 'mid', 0.5)
