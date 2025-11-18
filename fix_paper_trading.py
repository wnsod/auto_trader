"""
Paper Trading 문제 해결 스크립트
1. Paper Trading 모니터 실행하여 거래 생성
2. 오래된 세션 정리
3. 결과 검증
"""
import sys
sys.path.insert(0, '/workspace')

import logging
from rl_pipeline.validation.auto_paper_trading import run_paper_trading_monitor, AutoPaperTrading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🔧 Paper Trading 문제 해결 시작")
    logger.info("=" * 80)

    # 1. 현재 활성 세션 확인
    logger.info("\n📊 Step 1: 현재 활성 세션 확인")
    auto_paper = AutoPaperTrading()
    sessions = auto_paper.get_active_sessions()
    logger.info(f"   - 활성 세션: {len(sessions)}개")

    # 2. 오래된 세션 정리 (14일 이상)
    logger.info("\n🧹 Step 2: 오래된 세션 정리")
    cleaned = auto_paper.cleanup_old_sessions(days_old=14)
    logger.info(f"   - 정리된 세션: {cleaned}개")

    # 3. Paper Trading 모니터 실행 (1회)
    logger.info("\n🚀 Step 3: Paper Trading 모니터 실행")
    logger.info("   - 신호 체크 및 거래 실행 중...")
    run_paper_trading_monitor()

    # 4. 결과 확인
    logger.info("\n📊 Step 4: 실행 결과 확인")
    import sqlite3

    conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
    cursor = conn.cursor()

    # 총 거래 수
    cursor.execute("SELECT COUNT(*) FROM paper_trading_trades")
    total_trades = cursor.fetchone()[0]
    logger.info(f"   - 총 거래 수: {total_trades}개")

    # 최근 거래
    cursor.execute("""
        SELECT session_id, coin, action, price, size, timestamp
        FROM paper_trading_trades
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    recent_trades = cursor.fetchall()

    if recent_trades:
        logger.info("\n   📝 최근 거래:")
        for trade in recent_trades:
            session_id, coin, action, price, size, timestamp = trade
            logger.info(f"      - {coin} {action} {size:.4f} @ ${price:.2f} ({timestamp})")
    else:
        logger.warning("   ⚠️ 거래 없음")

    conn.close()

    logger.info("\n" + "=" * 80)
    logger.info("✅ Paper Trading 문제 해결 완료")
    logger.info("=" * 80)
