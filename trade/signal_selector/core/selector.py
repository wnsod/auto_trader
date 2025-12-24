"""
SignalSelector 통합 클래스

모든 Mixin들을 상속받아 완전한 SignalSelector를 구성합니다.
"""
import os
import sys

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
signal_selector_dir = os.path.dirname(current_dir)
trade_dir = os.path.dirname(signal_selector_dir)
workspace_dir = os.path.dirname(trade_dir)

sys.path.insert(0, trade_dir)
sys.path.insert(0, workspace_dir)

# Mixin imports
from signal_selector.core.base import CoreMixin
from signal_selector.db.loader import DBLoaderMixin
from signal_selector.db.writer import DBWriterMixin
from signal_selector.analysis.technical import TechnicalAnalysisMixin
from signal_selector.analysis.market import MarketAnalysisMixin
from signal_selector.scoring.calculator import ScoringMixin
from signal_selector.generator.signal import SignalGeneratorMixin
from signal_selector.cache.manager import CacheMixin
from signal_selector.strategy.manager import StrategyMixin

# 설정 imports
from signal_selector.config import (
    CANDLES_DB_PATH, STRATEGIES_DB_PATH, CACHE_SIZE
)
from signal_selector.utils import OptimizedCache, DatabasePool


class SignalSelector(
    CoreMixin,
    DBLoaderMixin,
    DBWriterMixin,
    TechnicalAnalysisMixin,
    MarketAnalysisMixin,
    ScoringMixin,
    SignalGeneratorMixin,
    CacheMixin,
    StrategyMixin
):
    """
    SignalSelector - 통합 시그널 선택 시스템

    모든 기능이 Mixin 클래스로 분리되어 있으며,
    이 클래스가 모든 Mixin을 상속받아 완전한 기능을 제공합니다.

    Mixin 구성:
    - CoreMixin: 초기화 및 핵심 로직
    - DBLoaderMixin: 데이터베이스 로딩
    - DBWriterMixin: 데이터베이스 저장
    - TechnicalAnalysisMixin: 기술적 분석
    - MarketAnalysisMixin: 시장 분석
    - ScoringMixin: 점수 계산
    - SignalGeneratorMixin: 시그널 생성
    - CacheMixin: 캐시 관리
    - StrategyMixin: 전략 관리
    """

    def __init__(self):
        """SignalSelector 초기화 - 모든 Mixin 초기화 호출"""
        # CoreMixin.__init__ 호출
        super().__init__()

        # 추가 초기화 (캐시 및 DB 풀)
        self.cache = OptimizedCache(max_size=CACHE_SIZE)
        self.max_cache_size = CACHE_SIZE
        self.db_pool = DatabasePool(CANDLES_DB_PATH, max_connections=8)
        self.prepared_statements = {}

        print("✅ SignalSelector 초기화 완료 (Mixin 구조)")


# 외부에서 import 가능하도록 export
__all__ = ['SignalSelector']
