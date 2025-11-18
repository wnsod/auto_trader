"""
Paper Trading 간단 테스트
"""
import sys
sys.path.insert(0, '/workspace')

import logging
from rl_pipeline.validation.auto_paper_trading import auto_start_paper_trading_after_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Paper Trading 테스트 시작")

    # ADA 코인으로 Paper Trading 시작
    result = auto_start_paper_trading_after_pipeline(
        coin="ADA",
        intervals=["15m"],
        duration_days=1  # 1일 테스트
    )

    logger.info(f"📊 결과: {result}")

    if result and result.get('status') == 'started':
        logger.info("✅ Paper Trading 세션 시작 성공")
    else:
        logger.error(f"❌ Paper Trading 세션 시작 실패: {result}")
