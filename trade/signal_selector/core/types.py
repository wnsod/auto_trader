"""
타입 정의 모듈 - SignalInfo, SignalAction 등

이 모듈은 시그널 선택 시스템에서 사용하는 핵심 데이터 타입을 정의합니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class SignalAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"


@dataclass
class SignalInfo:
    coin: str
    interval: str
    action: SignalAction
    signal_score: float
    confidence: float
    reason: str
    timestamp: int
    price: float
    volume: float
    rsi: float
    macd: float
    wave_phase: str
    pattern_type: str
    risk_level: str
    volatility: float
    volume_ratio: float
    wave_progress: float
    structure_score: float
    pattern_confidence: float
    integrated_direction: str
    integrated_strength: float
    
    # 고급 지표들
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
    market_condition: str = 'unknown'
    market_adaptation_bonus: float = 1.0
    calmar_ratio: float = 0.0
    profit_factor: float = 1.0
    reliability_score: float = 0.0
    learning_quality_score: float = 0.0
    global_strategy_id: str = ""
    coin_tuned: bool = False
    walk_forward_performance: Optional[Dict[str, float]] = field(default=None)
    regime_coverage: Optional[Dict[str, float]] = field(default=None)

