#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 트레이딩 로직 모듈

trade_executor.py와 virtual_trade_executor.py에서 공통으로 사용하는 로직을 중앙화합니다.

포함 기능:
1. 시장 컨텍스트 조회 (get_market_context)
2. 매수 임계값 계산 (calculate_buy_thresholds)
3. 7단계 레짐 관련 유틸리티
4. Thompson 점수 기반 판단 로직
"""

import os
import time
import sqlite3
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

# ============================================================================
# 7단계 레짐 정의 (rl_pipeline/core/regime_classifier.py와 동기화)
# ============================================================================

REGIME_STAGES = {
    1: "extreme_bearish",    # RSI < 20
    2: "bearish",            # RSI 20-40
    3: "sideways_bearish",   # RSI 40-50
    4: "neutral",            # RSI 45-55
    5: "sideways_bullish",   # RSI 50-60
    6: "bullish",            # RSI 60-80
    7: "extreme_bullish"     # RSI > 80
}

VALID_REGIMES = [
    "extreme_bearish",
    "bearish", 
    "sideways_bearish",
    "neutral",
    "sideways_bullish",
    "bullish",
    "extreme_bullish"
]

# 레짐 그룹 (분석용)
REGIME_GROUPS = {
    "bearish_group": ["extreme_bearish", "bearish", "sideways_bearish"],
    "neutral_group": ["neutral"],
    "bullish_group": ["sideways_bullish", "bullish", "extreme_bullish"]
}


def normalize_regime(regime: str) -> str:
    """
    레짐 이름을 7단계 표준으로 정규화
    
    Args:
        regime: 원본 레짐 이름 (다양한 형식 허용)
    
    Returns:
        7단계 레짐 중 하나 (기본값: 'neutral')
    """
    if not regime:
        return 'neutral'
    
    regime = regime.lower().replace(' ', '_').replace('-', '_')
    
    # 이미 유효한 레짐이면 그대로 반환
    if regime in VALID_REGIMES:
        return regime
    
    # 레거시 레짐 매핑
    if 'extreme' in regime and 'bear' in regime:
        return 'extreme_bearish'
    elif 'bear' in regime and 'side' in regime:
        return 'sideways_bearish'
    elif 'bear' in regime:
        return 'bearish'
    elif 'extreme' in regime and 'bull' in regime:
        return 'extreme_bullish'
    elif 'bull' in regime and 'side' in regime:
        return 'sideways_bullish'
    elif 'bull' in regime:
        return 'bullish'
    elif 'sideways' in regime or 'side' in regime:
        return 'neutral'
    
    return 'neutral'


def get_regime_group(regime: str) -> str:
    """
    레짐이 속한 그룹 반환 (bearish_group, neutral_group, bullish_group)
    """
    regime = normalize_regime(regime)
    
    for group, members in REGIME_GROUPS.items():
        if regime in members:
            return group
    
    return 'neutral_group'


def get_regime_severity(regime: str) -> int:
    """
    레짐의 강도 반환 (1=extreme_bearish ~ 7=extreme_bullish)
    """
    regime = normalize_regime(regime)
    
    for stage, name in REGIME_STAGES.items():
        if name == regime:
            return stage
    
    return 4  # neutral


# ============================================================================
# 시장 컨텍스트 조회 (캐시 적용)
# ============================================================================

_market_context_cache = {'data': None, 'timestamp': 0}
_MARKET_CONTEXT_CACHE_TTL = 60  # 1분 캐시


def get_market_context(force_refresh: bool = False) -> Dict[str, Any]:
    """
    현재 시장 상황 분석 (7단계 레짐 포함)
    
    Returns:
        {
            'trend': str,          # 레짐 이름 (7단계)
            'regime': str,         # trend와 동일 (하위 호환성)
            'regime_stage': int,   # 레짐 단계 (1~7)
            'regime_group': str,   # 레짐 그룹 (bearish/neutral/bullish)
            'volatility': float,   # 변동성
            'score': float,        # 시장 점수 (0~1)
            'breadth': str,        # 시장 폭 (narrow/normal/wide)
            'timestamp': int
        }
    """
    global _market_context_cache
    
    now = time.time()
    if not force_refresh and _market_context_cache['data'] and (now - _market_context_cache['timestamp'] < _MARKET_CONTEXT_CACHE_TTL):
        return _market_context_cache['data']
    
    # 기본값
    regime = 'neutral'
    volatility = 0.02
    score = 0.5
    
    try:
        from trade.core.database import CANDLES_DB_PATH, get_db_connection
        
        # 🔧 [Fix] get_db_connection 사용 (Docker/Windows 경로 호환성)
        with get_db_connection(CANDLES_DB_PATH, read_only=True, timeout=5.0) as conn:
            # DB에서 가장 최신 레짐 데이터 조회
            cursor = conn.execute("""
                SELECT regime_label, volatility, score, symbol
                FROM candles 
                WHERE regime_label IS NOT NULL
                ORDER BY timestamp DESC, volume DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                regime = normalize_regime(str(row[0] or 'neutral'))
                volatility = float(row[1]) if row[1] is not None else 0.02
                raw_score = float(row[2]) if row[2] is not None else 50.0
                score = raw_score / 100.0 if raw_score > 1.0 else raw_score
    except Exception as e:
        # DB 접근 오류 시 조용히 기본값 사용 (로그는 최초 1회만)
        pass
    
    context = {
        'trend': regime,
        'regime': regime,
        'regime_stage': get_regime_severity(regime),
        'regime_group': get_regime_group(regime),
        'volatility': volatility,
        'score': score,
        'breadth': 'normal',
        'timestamp': int(now)
    }
    
    _market_context_cache = {'data': context, 'timestamp': now}
    return context


# ============================================================================
# 매수 임계값 계산 (시장 상황 기반)
# ============================================================================

@dataclass
class BuyThresholds:
    """매수 임계값 설정"""
    min_signal_score: float           # 신규 매수 최소 시그널 점수
    min_signal_score_additional: float # 추매 최소 시그널 점수
    min_thompson_score: float         # 최소 Thompson 점수
    description: str                  # 설명


def calculate_buy_thresholds(
    market_context: Optional[Dict] = None,
    signal_continuity: float = 0.5,
    dynamic_influence: float = 0.5,
    learning_weight: float = 0.3
) -> BuyThresholds:
    """
    시장 상황과 학습 성숙도에 따른 매수 임계값 계산
    
    Args:
        market_context: 시장 컨텍스트 (None이면 자동 조회)
        signal_continuity: 시그널 연속성 (0~1)
        dynamic_influence: 동적 영향도 (0~1)
        learning_weight: 학습 가중치 (0~0.7)
    
    Returns:
        BuyThresholds 객체
    """
    if market_context is None:
        market_context = get_market_context()
    
    regime = market_context.get('regime', 'neutral')
    regime_group = get_regime_group(regime)
    
    # 기본 임계값
    BASE_MIN_SIGNAL_SCORE = 0.05
    BASE_MIN_SIGNAL_SCORE_ADDITIONAL = 0.15
    BASE_MIN_THOMPSON_SCORE = 0.10
    
    # 학습 성숙도가 높으면 Thompson(학습) 기준을 약간 낮춤
    thompson_maturity_adj = learning_weight * -0.03  # 최대 -2.1%
    
    # 시그널 연속성/영향도에 따른 임계값 조정
    continuity_adjustment = 0.0
    if signal_continuity > 0.7 and dynamic_influence > 0.6:
        continuity_adjustment = -0.02  # 임계값 낮춤 (더 쉽게 진입)
    elif signal_continuity < 0.3:
        continuity_adjustment = +0.05  # 급반전 시 임계값 높임
    
    # 레짐별 임계값 조정
    if regime == 'extreme_bearish':
        # 극심한 약세: 매우 엄격한 기준
        min_signal = BASE_MIN_SIGNAL_SCORE + 0.12 + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL + 0.15
        min_thompson = BASE_MIN_THOMPSON_SCORE + 0.18 + thompson_maturity_adj
        desc = f"극심한 약세장: 매수 기준 강화"
    elif regime == 'bearish':
        min_signal = BASE_MIN_SIGNAL_SCORE + 0.08 + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL + 0.12
        min_thompson = BASE_MIN_THOMPSON_SCORE + 0.12 + thompson_maturity_adj
        desc = f"약세장: 매수 기준 강화"
    elif regime == 'sideways_bearish':
        min_signal = BASE_MIN_SIGNAL_SCORE + 0.05 + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL + 0.08
        min_thompson = BASE_MIN_THOMPSON_SCORE + 0.08 + thompson_maturity_adj
        desc = f"약세 횡보장: 매수 기준 약간 강화"
    elif regime == 'sideways_bullish':
        min_signal = BASE_MIN_SIGNAL_SCORE - 0.01 + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL - 0.03
        min_thompson = BASE_MIN_THOMPSON_SCORE - 0.03 + thompson_maturity_adj
        desc = f"강세 횡보장: 매수 기준 약간 완화"
    elif regime == 'bullish':
        min_signal = BASE_MIN_SIGNAL_SCORE - 0.02 + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL - 0.05
        min_thompson = BASE_MIN_THOMPSON_SCORE - 0.05 + thompson_maturity_adj
        desc = f"강세장: 매수 기준 완화"
    elif regime == 'extreme_bullish':
        min_signal = BASE_MIN_SIGNAL_SCORE - 0.03 + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL - 0.07
        min_thompson = BASE_MIN_THOMPSON_SCORE - 0.07 + thompson_maturity_adj
        desc = f"극강세장: 매수 기준 크게 완화"
    else:  # neutral
        min_signal = BASE_MIN_SIGNAL_SCORE + continuity_adjustment
        min_signal_add = BASE_MIN_SIGNAL_SCORE_ADDITIONAL
        min_thompson = BASE_MIN_THOMPSON_SCORE + thompson_maturity_adj
        desc = f"중립장: 기본 기준"
    
    return BuyThresholds(
        min_signal_score=max(0.01, min_signal),
        min_signal_score_additional=max(0.05, min_signal_add),
        min_thompson_score=max(0.03, min_thompson),
        description=desc
    )


# ============================================================================
# 레짐별 매매 전략 가이드
# ============================================================================

def get_regime_trading_strategy(regime: str) -> Dict[str, Any]:
    """
    레짐별 권장 매매 전략 반환
    
    Returns:
        {
            'buy_aggressiveness': float,    # 매수 공격성 (0~1)
            'sell_defensiveness': float,    # 매도 방어성 (0~1)
            'position_size_factor': float,  # 포지션 크기 배수
            'stop_loss_tightness': float,   # 손절 타이트함 (0~1)
            'take_profit_target': float,    # 익절 목표 배수
            'advice': str                   # 전략 조언
        }
    """
    regime = normalize_regime(regime)
    
    strategies = {
        'extreme_bearish': {
            'buy_aggressiveness': 0.1,
            'sell_defensiveness': 0.9,
            'position_size_factor': 0.3,
            'stop_loss_tightness': 0.9,
            'take_profit_target': 0.5,
            'advice': '매수 자제, 현금 보유 우선. 매우 강한 시그널만 진입.'
        },
        'bearish': {
            'buy_aggressiveness': 0.3,
            'sell_defensiveness': 0.7,
            'position_size_factor': 0.5,
            'stop_loss_tightness': 0.7,
            'take_profit_target': 0.7,
            'advice': '보수적 매수, 빠른 익절. 추세 반전 시그널 주시.'
        },
        'sideways_bearish': {
            'buy_aggressiveness': 0.4,
            'sell_defensiveness': 0.6,
            'position_size_factor': 0.6,
            'stop_loss_tightness': 0.6,
            'take_profit_target': 0.8,
            'advice': '횡보 구간 매매. 지지선 근처 매수, 저항선 근처 매도.'
        },
        'neutral': {
            'buy_aggressiveness': 0.5,
            'sell_defensiveness': 0.5,
            'position_size_factor': 0.7,
            'stop_loss_tightness': 0.5,
            'take_profit_target': 1.0,
            'advice': '기본 전략 유지. 방향성 확인 후 진입.'
        },
        'sideways_bullish': {
            'buy_aggressiveness': 0.6,
            'sell_defensiveness': 0.4,
            'position_size_factor': 0.8,
            'stop_loss_tightness': 0.5,
            'take_profit_target': 1.1,
            'advice': '적극적 매수 준비. 돌파 시 추가 진입 고려.'
        },
        'bullish': {
            'buy_aggressiveness': 0.7,
            'sell_defensiveness': 0.3,
            'position_size_factor': 0.9,
            'stop_loss_tightness': 0.4,
            'take_profit_target': 1.3,
            'advice': '추세 추종 매매. 조정 시 추가 매수, 익절은 여유있게.'
        },
        'extreme_bullish': {
            'buy_aggressiveness': 0.5,  # 오히려 신중하게 (과열 주의)
            'sell_defensiveness': 0.5,
            'position_size_factor': 0.7,
            'stop_loss_tightness': 0.6,
            'take_profit_target': 1.5,
            'advice': '과열 주의! 신규 진입 신중, 보유 물량 일부 익절 고려.'
        }
    }
    
    return strategies.get(regime, strategies['neutral'])


# ============================================================================
# Thompson 점수 기반 판단 유틸리티
# ============================================================================

def should_execute_buy(
    signal_score: float,
    thompson_score: float,
    thresholds: BuyThresholds,
    expected_profit: float = 0.0,
    is_additional_buy: bool = False
) -> Tuple[bool, str]:
    """
    매수 실행 여부 판단 (공통 로직)
    
    Args:
        signal_score: 시그널 점수
        thompson_score: Thompson Sampling 점수
        thresholds: 매수 임계값
        expected_profit: 예상 수익률 (%)
        is_additional_buy: 추매 여부
    
    Returns:
        (should_buy: bool, reason: str)
    """
    min_signal = thresholds.min_signal_score_additional if is_additional_buy else thresholds.min_signal_score
    
    # 시그널 점수 체크
    if signal_score < min_signal:
        return False, f"시그널 점수 부족: {signal_score:.3f} < {min_signal:.2f}"
    
    # Thompson 점수 체크
    if thompson_score < thresholds.min_thompson_score:
        return False, f"Thompson 점수 부족: {thompson_score:.3f} < {thresholds.min_thompson_score:.2f}"
    
    # 기대수익률 체크
    if expected_profit < 0:
        return False, f"기대수익률 음수: {expected_profit:.2f}%"
    
    reason = f"시그널: {signal_score:.3f}, Thompson: {thompson_score:.2f}, 기대수익: {expected_profit:.2f}%"
    return True, reason


def calculate_combined_score(
    signal_score: float,
    thompson_score: float,
    signal_weight: float = 0.6,
    learning_weight: float = 0.4
) -> float:
    """
    시그널 점수와 Thompson 점수의 가중 평균 계산
    
    Args:
        signal_score: 시그널 점수
        thompson_score: Thompson Sampling 점수
        signal_weight: 시그널 가중치
        learning_weight: 학습 가중치 (=1-signal_weight)
    
    Returns:
        combined_score: 통합 점수
    """
    # 가중치 정규화
    total = signal_weight + learning_weight
    if total > 0:
        signal_weight = signal_weight / total
        learning_weight = learning_weight / total
    else:
        signal_weight = 0.6
        learning_weight = 0.4
    
    return (signal_score * signal_weight) + (thompson_score * learning_weight)
