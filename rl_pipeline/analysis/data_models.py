"""
분석 데이터 모델
데이터 클래스 정의
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class CoinSignalScore:
    """코인별 신호 스코어 산출 결과"""
    coin: str
    interval: str
    regime: str
    fractal_score: float
    multi_timeframe_score: float
    indicator_cross_score: float
    ensemble_score: float
    ensemble_confidence: float
    final_signal_score: float
    signal_action: str
    signal_confidence: float
    created_at: str


@dataclass
class GlobalSignalScore:
    """전역(글로벌) 신호 스코어 결과"""
    overall_score: float
    overall_confidence: float
    policy_improvement: float
    convergence_rate: float
    top_performers: List[str]
    top_coins: List[str]
    top_intervals: List[str]
    created_at: str

