"""
통합 전략 생성 Factory
아키텍처 개선: 인터페이스 명확화를 위한 단일 진입점 제공
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from rl_pipeline.core.types import Strategy
from rl_pipeline.strategy.common import StrategyCreationHelper


class StrategyType(Enum):
    """전략 타입 명시적 정의"""
    INTELLIGENT = "intelligent"
    INTELLIGENT_WITH_TYPE = "intelligent_with_type"
    ENHANCED_MARKET_ADAPTIVE = "enhanced_market_adaptive"
    GUIDED_RANDOM = "guided_random"
    BASIC = "basic"
    GRID_SEARCH = "grid_search"
    DIRECTION_SPECIALIZED = "direction_specialized"


@dataclass
class StrategyCreationConfig:
    """전략 생성 설정 (통일된 인터페이스)"""
    coin: str
    interval: str
    num_strategies: int
    strategy_type: StrategyType = StrategyType.INTELLIGENT
    strategy_type_param: Optional[str] = None  # "long_term", "short_term" 등
    candle_data: Optional[pd.DataFrame] = None
    use_data_based_ranges: bool = True
    enable_enhancements: bool = True
    
    def validate(self) -> None:
        """설정 검증"""
        StrategyCreationHelper.validate_params(
            self.coin,
            self.interval,
            self.num_strategies,
            self.candle_data
        )


class StrategyFactory:
    """
    통합 전략 생성 Factory
    
    사용 예:
        factory = StrategyFactory()
        config = StrategyCreationConfig(
            coin='BTC',
            interval='15m',
            num_strategies=100,
            strategy_type=StrategyType.INTELLIGENT
        )
        strategies = factory.create(config)
    """
    
    def __init__(self):
        self._creators = {
            StrategyType.INTELLIGENT: self._create_intelligent,
            StrategyType.INTELLIGENT_WITH_TYPE: self._create_intelligent_with_type,
            StrategyType.ENHANCED_MARKET_ADAPTIVE: self._create_enhanced_market_adaptive,
            StrategyType.GUIDED_RANDOM: self._create_guided_random,
            StrategyType.BASIC: self._create_basic,
            StrategyType.GRID_SEARCH: self._create_grid_search,
            StrategyType.DIRECTION_SPECIALIZED: self._create_direction_specialized,
        }
    
    def create(self, config: StrategyCreationConfig) -> List[Strategy]:
        """
        전략 생성 (단일 진입점)
        
        Args:
            config: 전략 생성 설정
            
        Returns:
            생성된 전략 리스트
        """
        config.validate()
        
        # 타입별 생성자 호출
        creator = self._creators.get(config.strategy_type)
        if not creator:
            raise ValueError(f"알 수 없는 전략 타입: {config.strategy_type}")
        
        # 공통 로직: 전략 생성
        strategies = creator(config)
        
        # 공통 로직: 중복 제거
        strategies = self._remove_duplicates(strategies)
        
        # 공통 로직: DB 저장 (선택적)
        if config.enable_enhancements:
            StrategyCreationHelper.save_batch(strategies)
        
        return strategies
    
    def _create_intelligent(self, config: StrategyCreationConfig) -> List[Strategy]:
        """지능형 전략 생성"""
        from rl_pipeline.strategy.creator import create_intelligent_strategies
        if config.candle_data is None:
            raise ValueError("지능형 전략 생성에는 candle_data가 필요합니다")
        return create_intelligent_strategies(
            coin=config.coin,
            interval=config.interval,
            num_strategies=config.num_strategies,
            df=config.candle_data
        )
    
    def _create_intelligent_with_type(self, config: StrategyCreationConfig) -> List[Strategy]:
        """타입별 지능형 전략 생성"""
        from rl_pipeline.strategy.creator import create_intelligent_strategies_with_type
        if config.candle_data is None:
            raise ValueError("타입별 지능형 전략 생성에는 candle_data가 필요합니다")
        return create_intelligent_strategies_with_type(
            coin=config.coin,
            interval=config.interval,
            num_strategies=config.num_strategies,
            df=config.candle_data,
            strategy_type=config.strategy_type_param or "general"
        )
    
    def _create_enhanced_market_adaptive(self, config: StrategyCreationConfig) -> List[Strategy]:
        """향상된 시장 적응형 전략 생성"""
        from rl_pipeline.strategy.creator import create_enhanced_market_adaptive_strategy
        strategies = []
        for i in range(config.num_strategies):
            strategy = create_enhanced_market_adaptive_strategy(
                coin=config.coin,
                interval=config.interval,
                index=i
            )
            strategies.append(strategy)
        return strategies
    
    def _create_guided_random(self, config: StrategyCreationConfig) -> List[Strategy]:
        """가이드 랜덤 전략 생성"""
        from rl_pipeline.strategy.creator import create_guided_random_strategy, classify_market_condition
        from rl_pipeline.core.regime_strategy_manager import get_target_regime_for_generation

        strategies = []
        df = config.candle_data if config.candle_data is not None else pd.DataFrame()

        # 시장 상황 및 레짐 결정
        market_condition = classify_market_condition(df) if not df.empty else "neutral"
        regime = get_target_regime_for_generation(config.coin, config.interval)

        for i in range(config.num_strategies):
            strategy = create_guided_random_strategy(
                coin=config.coin,
                interval=config.interval,
                df=df,
                market_condition=market_condition,
                index=i,
                regime=regime
            )
            strategies.append(strategy)
        return strategies
    
    def _create_basic(self, config: StrategyCreationConfig) -> List[Strategy]:
        """기본 전략 생성"""
        from rl_pipeline.strategy.creator import create_basic_strategy
        strategies = []
        for i in range(config.num_strategies):
            strategy = create_basic_strategy(
                coin=config.coin,
                interval=config.interval,
                index=i
            )
            strategies.append(strategy)
        return strategies
    
    def _create_grid_search(self, config: StrategyCreationConfig) -> List[Strategy]:
        """그리드 서치 전략 생성"""
        try:
            from rl_pipeline.strategy.creator_enhancements import create_grid_search_strategies
            if config.candle_data is None:
                raise ValueError("그리드 서치 전략 생성에는 candle_data가 필요합니다")
            return create_grid_search_strategies(
                coin=config.coin,
                interval=config.interval,
                num_strategies=config.num_strategies,
                df=config.candle_data
            )
        except ImportError:
            raise NotImplementedError("그리드 서치 전략 생성은 creator_enhancements 모듈이 필요합니다")
    
    def _create_direction_specialized(self, config: StrategyCreationConfig) -> List[Strategy]:
        """방향별 특화 전략 생성"""
        try:
            from rl_pipeline.strategy.creator_enhancements import create_direction_specialized_strategies
            if config.candle_data is None:
                raise ValueError("방향별 특화 전략 생성에는 candle_data가 필요합니다")
            return create_direction_specialized_strategies(
                coin=config.coin,
                interval=config.interval,
                num_strategies=config.num_strategies,
                df=config.candle_data
            )
        except ImportError:
            raise NotImplementedError("방향별 특화 전략 생성은 creator_enhancements 모듈이 필요합니다")
    
    def _remove_duplicates(self, strategies: List[Strategy]) -> List[Strategy]:
        """중복 전략 제거"""
        try:
            from rl_pipeline.strategy.creator_enhancements import filter_duplicate_strategies
            return filter_duplicate_strategies(strategies)
        except ImportError:
            # 개선 모듈 없으면 기본 중복 제거
            seen = set()
            unique = []
            for s in strategies:
                key = (s.coin, s.interval, str(s.strategy_conditions))
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            return unique


# 레거시 호환성: 기존 함수를 Factory로 래핑
def create_strategies(
    coin: str,
    interval: str,
    num_strategies: int,
    strategy_type: str = "intelligent",
    df: Optional[pd.DataFrame] = None
) -> List[Strategy]:
    """
    레거시 호환 함수
    
    Args:
        coin: 코인 심볼
        interval: 시간 간격
        num_strategies: 생성할 전략 수
        strategy_type: 전략 타입 ('intelligent', 'basic' 등)
        df: 캔들 데이터 (선택적)
        
    Returns:
        생성된 전략 리스트
    """
    factory = StrategyFactory()
    try:
        strategy_type_enum = StrategyType(strategy_type)
    except ValueError:
        # 기본값 사용
        strategy_type_enum = StrategyType.INTELLIGENT
    
    config = StrategyCreationConfig(
        coin=coin,
        interval=interval,
        num_strategies=num_strategies,
        strategy_type=strategy_type_enum,
        candle_data=df
    )
    return factory.create(config)

