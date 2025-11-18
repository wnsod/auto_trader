"""
로깅 유틸리티 모듈
구조화된 로깅 및 로그 레벨 관리
"""

import logging
import time
from functools import wraps
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# 로깅 레벨 가이드라인
# DEBUG: 개발 중 디버깅용 (프로덕션에서는 비활성화)
# INFO: 중요한 진행 상황
# WARNING: 예상 가능한 문제 (계속 진행 가능)
# ERROR: 심각한 문제 (처리 필요)


def log_strategy_creation(coin: str, interval: str, count: int, success: bool):
    """전략 생성 로깅 통합"""
    if success:
        logger.info(f"✅ {coin}-{interval}: {count}개 전략 생성 완료")
    else:
        logger.error(f"❌ {coin}-{interval}: 전략 생성 실패")


def log_pipeline_step(step: str, coin: str, interval: str, success: bool, details: Optional[Dict[str, Any]] = None):
    """파이프라인 단계 로깅"""
    status = "✅" if success else "❌"
    message = f"{status} {coin}-{interval}: {step}"
    
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        message += f" ({detail_str})"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


def log_performance(operation: str, duration_ms: float, details: Optional[Dict[str, Any]] = None):
    """성능 로깅"""
    message = f"⏱️ {operation}: {duration_ms:.2f}ms"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        message += f" ({detail_str})"
    logger.debug(message)


@contextmanager
def log_execution_time(operation: str, details: Optional[Dict[str, Any]] = None):
    """실행 시간 측정 컨텍스트 매니저"""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        log_performance(operation, duration_ms, details)


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    level: str = "error"
):
    """컨텍스트와 함께 에러 로깅"""
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    message = f"❌ {context_str}: {str(error)}"
    
    if level == "error":
        logger.error(message, exc_info=True)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.debug(message, exc_info=True)


def log_batch_operation(
    operation: str,
    total: int,
    success: int,
    failed: int,
    details: Optional[Dict[str, Any]] = None
):
    """배치 작업 로깅"""
    success_rate = (success / total * 100) if total > 0 else 0
    message = f"📊 {operation}: {success}/{total} 성공 ({success_rate:.1f}%)"
    
    if failed > 0:
        message += f", {failed}개 실패"
    
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        message += f" ({detail_str})"
    
    if success == total:
        logger.info(message)
    elif success > 0:
        logger.warning(message)
    else:
        logger.error(message)


def log_db_operation(operation: str, table: str, count: int, success: bool):
    """데이터베이스 작업 로깅"""
    status = "✅" if success else "❌"
    logger.info(f"{status} {operation}: {count}개 행 -> {table}")


def log_strategy_grade_update(
    strategy_id: str,
    old_grade: str,
    new_grade: str,
    reason: Optional[str] = None
):
    """전략 등급 업데이트 로깅"""
    message = f"📈 {strategy_id}: {old_grade} → {new_grade}"
    if reason:
        message += f" ({reason})"
    logger.info(message)


def log_selfplay_result(
    coin: str,
    interval: str,
    episodes: int,
    avg_win_rate: float,
    avg_profit: float,
    details: Optional[Dict[str, Any]] = None
):
    """Self-play 결과 로깅"""
    message = f"🎮 {coin}-{interval} Self-play: {episodes} 에피소드, 승률 {avg_win_rate:.1%}, 수익 {avg_profit:.2f}"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        message += f" ({detail_str})"
    logger.info(message)


def suppress_debug_logs():
    """DEBUG 로그 비활성화 (프로덕션용)"""
    logging.getLogger().setLevel(logging.INFO)


def enable_debug_logs():
    """DEBUG 로그 활성화 (개발용)"""
    logging.getLogger().setLevel(logging.DEBUG)

