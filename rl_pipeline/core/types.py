"""
Absolute Zero 시스템 공용 타입 정의
모든 모듈에서 사용하는 공통 타입과 DTO 정의
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# 공용 Enum들
class PositionState(Enum):
    NO_POSITION = "no_position"
    LONG_POSITION = "long_position"
    SHORT_POSITION = "short_position"

class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    LONG = "long"
    SHORT = "short"

class RunStep(Enum):
    STRATEGY_GENERATION = "strategy_generation"
    SIMULATION = "simulation"
    DNA_ANALYSIS = "dna_analysis"
    FRACTAL_ANALYSIS = "fractal_analysis"
    SYNERGY_ANALYSIS = "synergy_analysis"
    METADATA_SYNC = "metadata_sync"

# 핵심 DTO들
@dataclass
class Strategy:
    """전략 DTO - 모듈간 계약"""
    id: str
    params: Dict[str, Any]
    version: str
    coin: str
    interval: str
    created_at: datetime
    complexity_score: float = 0.0
    confidence: float = 0.0
    strategy_type: str = "hybrid"

    # 🆕 레짐 (ranging, trending, volatile)
    regime: str = "ranging"

    # 전략 파라미터들
    rsi_min: float = 30.0
    rsi_max: float = 70.0
    volume_ratio_min: float = 1.0
    volume_ratio_max: float = 2.0
    macd_buy_threshold: float = 0.0
    macd_sell_threshold: float = 0.0
    # 🆕 추가 지표들 (min/max 관리)
    mfi_min: float = 20.0
    mfi_max: float = 80.0
    atr_min: float = 0.01
    atr_max: float = 0.05
    adx_min: float = 15.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    position_size: float = 0.01
    max_trades: int = 100
    min_trades: int = 3
    win_rate_threshold: float = 0.4
    profit_threshold: float = 0.0
    ma_period: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    pattern_confidence: float = 0.5

    # 패턴 소스
    pattern_source: str = 'unknown'

    # 향상 타입
    enhancement_type: str = 'none'

    # 조건 속성들
    rsi_condition: Dict[str, float] = None
    volume_condition: Dict[str, float] = None
    atr_condition: Dict[str, float] = None
    
    # 🚀 통합 분석 메타데이터 (그룹 조합, OR 조건 등)
    metadata: Dict[str, Any] = None

    def get(self, key: str, default: Any = None) -> Any:
        """dict 인터페이스 호환을 위한 헬퍼 (Orchestrator 등에서 사용)"""
        if hasattr(self, key):
            value = getattr(self, key)
            if value is not None:
                return value
        if isinstance(self.params, dict):
            return self.params.get(key, default)
        return default

@dataclass
class ReplayReport:
    """시뮬레이션 결과 DTO"""
    run_id: str
    coin: str
    interval: str
    profit_factor: float
    sharpe_ratio: float
    total_return: float
    trades: int
    win_rate: float
    max_drawdown: float
    avg_profit_per_trade: float
    by_trade: List[Dict[str, Any]]
    execution_time: float = 0.0

@dataclass
class Position:
    """포지션 DTO"""
    entry_time: datetime
    entry_price: float
    entry_index: int
    position_type: PositionState
    stop_loss_price: float
    take_profit_price: float
    max_hold_periods: int
    current_hold_periods: int = 0

@dataclass
class SimulationState:
    """시뮬레이션 상태 DTO"""
    current_index: int
    current_price: float
    position: Optional[Position] = None
    total_profit: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

@dataclass
class CandleData:
    """캔들 데이터 DTO"""
    coin: str
    interval: str
    data: pd.DataFrame
    indicators_computed: bool = False
    cache_key: str = ""

@dataclass
class DNAAnalysis:
    """DNA 분석 결과 DTO"""
    coin: str
    interval: str
    dna_patterns: Dict[str, Any]
    top_strategies_count: int
    analysis_timestamp: datetime
    quality_score: float = 0.0

@dataclass
class FractalAnalysis:
    """프랙탈 분석 결과 DTO"""
    coin: str
    interval: str
    fractal_score: float
    pattern_distribution: Dict[str, Any]
    pruned_strategies_count: int
    analysis_timestamp: datetime

@dataclass
class SynergyAnalysis:
    """시너지 분석 결과 DTO"""
    coin: str
    interval: str
    synergy_score: float
    synergy_patterns: Dict[str, Any]
    analysis_timestamp: datetime

@dataclass
class PerformanceMetrics:
    """성능 지표 DTO"""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    gpu_usage: Optional[float] = None
    timestamp: datetime = None

@dataclass
class RunMetadata:
    """실행 메타데이터 DTO"""
    run_id: str
    coin: str
    interval: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    strategies_count: int = 0
    successful_strategies: int = 0
    error_count: int = 0

# ============================================================================
# Result 타입 - 명시적 에러 처리
# ============================================================================

@dataclass
class Result:
    """제네릭 Result 타입 - 성공/실패를 명시적으로 표현"""
    success: bool
    error: Optional[str] = None
    data: Optional[Any] = None
    
    @classmethod
    def success(cls, data: Any = None):
        """성공 결과 생성"""
        return cls(success=True, data=data)
    
    @classmethod
    def failure(cls, error: str, data: Any = None):
        """실패 결과 생성"""
        return cls(success=False, error=error, data=data)
    
    def is_success(self) -> bool:
        """성공 여부 확인"""
        return self.success
    
    def is_failure(self) -> bool:
        """실패 여부 확인"""
        return not self.success

@dataclass
class StrategyResult(Result):
    """전략 생성 결과"""
    strategy: Optional[Strategy] = None
    
    @classmethod
    def success(cls, strategy: Strategy):
        """성공 결과 생성"""
        return cls(success=True, strategy=strategy, data=strategy)
    
    @classmethod
    def failure(cls, error: str, strategy: Optional[Strategy] = None):
        """실패 결과 생성"""
        return cls(success=False, error=error, strategy=strategy, data=strategy)

@dataclass
class StrategyMetrics:
    """전략 성능 지표 DTO"""
    profit: float = 0.0
    profit_percent: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    calmar_ratio: float = 0.0
    avg_profit_per_trade: float = 0.0