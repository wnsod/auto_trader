"""
전략 생성 공통 로직
코드 품질 개선: 중복 코드 제거를 위한 공통 헬퍼
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from rl_pipeline.core.types import Strategy
from datetime import datetime


class StrategyCreationHelper:
    """전략 생성 공통 헬퍼"""
    
    @staticmethod
    def validate_params(
        coin: str,
        interval: str,
        num_strategies: int,
        df: Optional[pd.DataFrame] = None
    ) -> None:
        """파라미터 검증 (통합)"""
        if not coin:
            raise ValueError("coin은 필수입니다")
        if not interval:
            raise ValueError("interval은 필수입니다")
        if num_strategies <= 0:
            raise ValueError("num_strategies는 1 이상이어야 합니다")
        if df is not None and df.empty:
            raise ValueError("캔들 데이터가 비어있습니다")
    
    @staticmethod
    def check_duplicates(
        new_strategy: Strategy,
        existing_strategies: List[Strategy]
    ) -> bool:
        """중복 체크 (통합)"""
        try:
            from rl_pipeline.strategy.creator_enhancements import generate_strategy_hash
            new_hash = generate_strategy_hash(new_strategy)
            for existing in existing_strategies:
                existing_hash = generate_strategy_hash(existing)
                if new_hash == existing_hash:
                    return True
            return False
        except (ImportError, AttributeError):
            # 해시 생성 실패 시 기본 비교
            for existing in existing_strategies:
                if (new_strategy.coin == existing.coin and
                    new_strategy.interval == existing.interval and
                    str(new_strategy.strategy_conditions) == str(existing.strategy_conditions)):
                    return True
            return False
    
    @staticmethod
    def save_batch(
        strategies: List[Strategy],
        batch_size: int = 100
    ) -> int:
        """배치 저장 (통합)"""
        from rl_pipeline.strategy.manager import save_strategies_to_db
        
        saved_count = 0
        for i in range(0, len(strategies), batch_size):
            batch = strategies[i:i+batch_size]
            saved = save_strategies_to_db(batch)
            saved_count += len(saved) if saved else 0
        
        return saved_count
    
    @staticmethod
    def create_strategy_base(
        coin: str,
        interval: str,
        conditions: Dict[str, Any],
        strategy_id: Optional[str] = None
    ) -> Strategy:
        """전략 기본 구조 생성 (통합)"""
        return Strategy(
            id=strategy_id or f"strategy_{datetime.now().timestamp()}",
            coin=coin,
            interval=interval,
            strategy_conditions=conditions,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

