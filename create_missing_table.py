"""
누락된 integrated_analysis_results 테이블 생성
"""
import sys
sys.path.insert(0, '/workspace')

from rl_pipeline.db.schema import create_integrated_analysis_results_table
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 integrated_analysis_results 테이블 생성 시작")
    result = create_integrated_analysis_results_table()

    if result:
        logger.info("✅ 테이블 생성 성공")

        # 테이블 확인
        import sqlite3
        conn = sqlite3.connect('/workspace/data_storage/learning_results.db')
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        logger.info(f"📊 현재 테이블 목록: {[t[0] for t in tables]}")

        # integrated_analysis_results 테이블 스키마 확인
        schema = conn.execute("PRAGMA table_info(integrated_analysis_results)").fetchall()
        logger.info(f"📋 integrated_analysis_results 스키마:")
        for col in schema:
            logger.info(f"   - {col[1]}: {col[2]}")

        conn.close()
    else:
        logger.error("❌ 테이블 생성 실패")
        sys.exit(1)
