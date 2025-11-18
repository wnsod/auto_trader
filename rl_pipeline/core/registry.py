"""
Registry 모듈 - 지표 등록을 위한 데코레이터
"""

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

def register_indicator(name: str) -> Callable:
    """
    지표 등록 데코레이터 (더미 구현)
    
    Args:
        name: 지표 이름
        
    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        """실제 데코레이터 함수"""
        logger.debug(f"📊 지표 등록: {name}")
        return func
    
    return decorator

