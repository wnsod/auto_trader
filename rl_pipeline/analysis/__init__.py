"""
분석 모듈
새로운 파이프라인에 맞춘 통합 분석 인터페이스
"""

import logging

logger = logging.getLogger(__name__)

# 새로운 통합분석기 (새로운 파이프라인의 핵심)
try:
    from .integrated_analyzer import IntegratedAnalyzer, analyze_strategies, analyze_global_strategies
    INTEGRATED_ANALYZER_AVAILABLE = True
except ImportError as e:
    # 🔥 필수 모듈이므로 경고 유지 (logger 사용)
    logger.warning(f"⚠️ 통합분석기 import 실패: {e}")
    INTEGRATED_ANALYZER_AVAILABLE = False

__all__ = [
    # 새로운 통합분석기 (새로운 파이프라인의 핵심)
    "IntegratedAnalyzer", "analyze_strategies", "analyze_global_strategies"
]