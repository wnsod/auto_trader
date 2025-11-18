"""
유틸리티 헬퍼 함수들
"""

from typing import Union, Any
import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    안전한 float 변환 함수
    
    Args:
        value: 변환할 값
        default: 변환 실패 시 기본값
    
    Returns:
        float 값 (변환 실패 시 default 반환)
    """
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = 'unknown') -> str:
    """
    안전한 string 변환 함수
    
    Args:
        value: 변환할 값
        default: 변환 실패 시 기본값
    
    Returns:
        string 값 (변환 실패 시 default 반환)
    """
    if value is None or pd.isna(value):
        return default
    try:
        return str(value)
    except (ValueError, TypeError):
        return default

