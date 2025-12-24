"""
Trade Core 모듈

가상매매와 실전매매에서 공통으로 사용하는 핵심 로직
"""

# 시장 분석
from trade.core.market import MarketAnalyzer

# 리스크 관리
from trade.core.risk import RiskManager

# 청산 파라미터 (학습 기반)
from trade.core.exit_params import (
    should_take_profit,
    should_stop_loss,
    get_exit_params,
    STRATEGY_DB_PATH as EXIT_PARAMS_DB_PATH
)

# Thompson Sampling
from trade.core.thompson import (
    ThompsonScoreCalculator,
    ThompsonScore,
    get_thompson_calculator,
    get_thompson_score,
    get_thompson_score_from_pattern,
    should_execute_trade,
    extract_signal_pattern
)

__all__ = [
    # Market
    'MarketAnalyzer',
    
    # Risk
    'RiskManager',
    
    # Exit Params
    'should_take_profit',
    'should_stop_loss',
    'get_exit_params',
    'EXIT_PARAMS_DB_PATH',
    
    # Thompson
    'ThompsonScoreCalculator',
    'ThompsonScore',
    'get_thompson_calculator',
    'get_thompson_score',
    'get_thompson_score_from_pattern',
    'should_execute_trade',
    'extract_signal_pattern',
]

