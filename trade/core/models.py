#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
트레이딩 시스템 공통 데이터 모델
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

class SignalAction(Enum):
    """시그널 액션 열거형"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    SKIP = "skip"

@dataclass
class SignalInfo:
    """시그널 정보 데이터클래스"""
    coin: str
    interval: str
    action: SignalAction
    signal_score: float
    confidence: float
    reason: str
    timestamp: int
    price: float = 0.0
    volume: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    wave_phase: str = 'unknown'
    pattern_type: str = 'none'
    risk_level: str = 'medium'
    volatility: float = 0.02
    volume_ratio: float = 1.0
    wave_progress: float = 0.0
    structure_score: float = 0.5
    pattern_confidence: float = 0.0
    integrated_direction: str = 'neutral'
    integrated_strength: float = 0.0
    # Absolute Zero System 고급 지표들
    mfi: float = 50.0
    atr: float = 0.0
    adx: float = 25.0
    ma20: float = 0.0
    rsi_ema: float = 50.0
    macd_smoothed: float = 0.0
    wave_momentum: float = 0.0
    bb_position: str = 'unknown'
    bb_width: float = 0.0
    bb_squeeze: float = 0.0
    rsi_divergence: str = 'none'
    macd_divergence: str = 'none'
    volume_divergence: str = 'none'
    price_momentum: float = 0.0
    volume_momentum: float = 0.0
    trend_strength: float = 0.5
    support_resistance: str = 'unknown'
    fibonacci_levels: str = 'unknown'
    elliott_wave: str = 'unknown'
    harmonic_patterns: str = 'none'
    candlestick_patterns: str = 'none'
    market_structure: str = 'unknown'
    flow_level_meta: str = 'unknown'
    pattern_direction: str = 'neutral'
    target_price: float = 0.0
    initial_target_price: float = 0.0
    source_type: str = 'quant'
    tick_size: float = 0.0 # 🆕 호가 단위 (Tick Size) 추가
    # 🆕 Absolute Zero System 정밀 분석 점수
    fractal_score: float = 0.5
    mtf_score: float = 0.5
    cross_score: float = 0.5
    # 🆕 전략 시스템 필드
    strategy_scores: dict = None  # 전략별 점수 {strategy: {match: 0.5, ...}}
    recommended_strategy: str = 'trend'  # 추천 전략
    strategy_match: float = 0.5  # 전략 적합도

@dataclass
class VirtualPosition:
    """가상 포지션 정보 데이터클래스"""
    coin: str
    entry_price: float
    quantity: float
    entry_timestamp: int
    entry_signal_score: float
    current_price: float
    profit_loss_pct: float
    holding_duration: int
    max_profit_pct: float
    max_loss_pct: float
    stop_loss_price: float
    take_profit_price: float
    last_updated: int
    target_price: float = 0.0
    initial_target_price: float = 0.0
    pattern_type: str = 'none'
    entry_confidence: float = 0.0
    # 🆕 Absolute Zero System 정밀 분석 점수
    fractal_score: float = 0.5
    mtf_score: float = 0.5
    cross_score: float = 0.5

