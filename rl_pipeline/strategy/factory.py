"""
Strategy Factory 모듈 - 전략 객체 생성을 위한 팩토리 함수들
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Strategy:
    """전략 데이터 클래스"""
    coin: str
    interval: str
    strategy_type: str
    params: Dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.name is None:
            self.name = f"{self.strategy_type}_{self.coin}_{self.interval}"
        
        if self.description is None:
            self.description = f"{self.strategy_type} strategy for {self.coin} {self.interval}"

def make_strategy(params: Dict[str, Any], coin: str, interval: str, strategy_type: str = "custom") -> Strategy:
    """
    전략 객체를 생성하는 팩토리 함수
    
    Args:
        params: 전략 파라미터 딕셔너리
        coin: 코인 심볼
        interval: 시간 간격
        strategy_type: 전략 타입
        
    Returns:
        Strategy 객체
    """
    logger.debug(f"📊 전략 생성: {strategy_type} for {coin} {interval}")
    
    # 파라미터 검증
    required_params = ['rsi_min', 'rsi_max', 'volume_ratio_min', 'volume_ratio_max']
    for param in required_params:
        if param not in params:
            logger.warning(f"⚠️ 필수 파라미터 {param}가 없습니다. 기본값 사용")
            if param == 'rsi_min':
                params[param] = 30.0
            elif param == 'rsi_max':
                params[param] = 70.0
            elif param == 'volume_ratio_min':
                params[param] = 1.0
            elif param == 'volume_ratio_max':
                params[param] = 2.0
    
    # 전략 타입에 따른 기본값 설정
    if strategy_type == "range_trading":
        params.setdefault('bb_period', 20)
        params.setdefault('bb_std', 2.0)
    elif strategy_type == "mean_reversion":
        params.setdefault('ma_period', 20)
        params.setdefault('bb_period', 20)
    elif strategy_type == "trend_following":
        params.setdefault('ma_period', 20)
        params.setdefault('macd_buy_threshold', 0.0)
        params.setdefault('macd_sell_threshold', 0.0)
    elif strategy_type == "volume_spike":
        params.setdefault('volume_ratio_min', 1.5)
        params.setdefault('volume_ratio_max', 3.0)
    
    # 공통 기본값
    params.setdefault('stop_loss_pct', 0.02)
    params.setdefault('take_profit_pct', 0.04)
    params.setdefault('position_size', 0.01)
    
    return Strategy(
        coin=coin,
        interval=interval,
        strategy_type=strategy_type,
        params=params
    )

def create_range_trading_strategy(coin: str, interval: str, **kwargs) -> Strategy:
    """범위 거래 전략 생성"""
    params = {
        'rsi_min': kwargs.get('rsi_min', 30.0),
        'rsi_max': kwargs.get('rsi_max', 70.0),
        'volume_ratio_min': kwargs.get('volume_ratio_min', 1.0),
        'volume_ratio_max': kwargs.get('volume_ratio_max', 2.0),
        'bb_period': kwargs.get('bb_period', 20),
        'bb_std': kwargs.get('bb_std', 2.0),
        'stop_loss_pct': kwargs.get('stop_loss_pct', 0.02),
        'take_profit_pct': kwargs.get('take_profit_pct', 0.04),
    }
    return make_strategy(params, coin, interval, "range_trading")

def create_mean_reversion_strategy(coin: str, interval: str, **kwargs) -> Strategy:
    """평균 회귀 전략 생성"""
    params = {
        'rsi_min': kwargs.get('rsi_min', 25.0),
        'rsi_max': kwargs.get('rsi_max', 75.0),
        'volume_ratio_min': kwargs.get('volume_ratio_min', 1.2),
        'volume_ratio_max': kwargs.get('volume_ratio_max', 2.5),
        'ma_period': kwargs.get('ma_period', 20),
        'bb_period': kwargs.get('bb_period', 20),
        'stop_loss_pct': kwargs.get('stop_loss_pct', 0.025),
        'take_profit_pct': kwargs.get('take_profit_pct', 0.05),
    }
    return make_strategy(params, coin, interval, "mean_reversion")

def create_trend_following_strategy(coin: str, interval: str, **kwargs) -> Strategy:
    """추세 추종 전략 생성"""
    params = {
        'rsi_min': kwargs.get('rsi_min', 40.0),
        'rsi_max': kwargs.get('rsi_max', 80.0),
        'volume_ratio_min': kwargs.get('volume_ratio_min', 1.0),
        'volume_ratio_max': kwargs.get('volume_ratio_max', 2.0),
        'ma_period': kwargs.get('ma_period', 20),
        'macd_buy_threshold': kwargs.get('macd_buy_threshold', 0.0),
        'macd_sell_threshold': kwargs.get('macd_sell_threshold', 0.0),
        'stop_loss_pct': kwargs.get('stop_loss_pct', 0.015),
        'take_profit_pct': kwargs.get('take_profit_pct', 0.06),
    }
    return make_strategy(params, coin, interval, "trend_following")

def create_volume_spike_strategy(coin: str, interval: str, **kwargs) -> Strategy:
    """볼륨 스파이크 전략 생성"""
    params = {
        'rsi_min': kwargs.get('rsi_min', 35.0),
        'rsi_max': kwargs.get('rsi_max', 75.0),
        'volume_ratio_min': kwargs.get('volume_ratio_min', 1.5),
        'volume_ratio_max': kwargs.get('volume_ratio_max', 3.0),
        'stop_loss_pct': kwargs.get('stop_loss_pct', 0.015),
        'take_profit_pct': kwargs.get('take_profit_pct', 0.03),
        'position_size': kwargs.get('position_size', 0.01),
    }
    return make_strategy(params, coin, interval, "volume_spike")

