"""
Paper Trading DB 경로 테스트
"""
import sys
sys.path.insert(0, '/workspace')

import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # auto_paper_trading.py의 DB 경로 로직 테스트
    db_path = os.getenv('RL_STRATEGIES_DB_PATH', 'data_storage/rl_strategies.db')

    logger.info(f"🔍 Paper Trading DB 경로: {db_path}")
    logger.info(f"📂 절대 경로: /workspace/{db_path}")

    full_path = f"/workspace/{db_path}"

    if os.path.exists(full_path):
        logger.info(f"✅ DB 파일 존재 확인")

        # integrated_analysis_results 테이블 확인
        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()

        # 테이블 존재 여부 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='integrated_analysis_results'")
        table_exists = cursor.fetchone()

        if table_exists:
            logger.info(f"✅ integrated_analysis_results 테이블 존재")

            # 스키마 확인
            schema = cursor.execute("PRAGMA table_info(integrated_analysis_results)").fetchall()
            logger.info(f"📋 테이블 스키마 ({len(schema)}개 컬럼):")
            for col in schema:
                logger.info(f"   - {col[1]}: {col[2]}")

            # 데이터 개수 확인
            cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
            count = cursor.fetchone()[0]
            logger.info(f"📊 저장된 레코드 수: {count}개")

            if count > 0:
                # 샘플 데이터 조회
                cursor.execute("SELECT coin, interval, signal_action, final_signal_score, created_at FROM integrated_analysis_results ORDER BY created_at DESC LIMIT 5")
                rows = cursor.fetchall()
                logger.info(f"\n📝 최근 데이터 샘플:")
                for row in rows:
                    logger.info(f"   - {row[0]}-{row[1]}: {row[2]} (점수: {row[3]}, 시각: {row[4]})")
        else:
            logger.error(f"❌ integrated_analysis_results 테이블 없음")

        conn.close()
    else:
        logger.error(f"❌ DB 파일이 존재하지 않음: {full_path}")

        # learning_results.db 확인
        old_path = "/workspace/data_storage/learning_results.db"
        if os.path.exists(old_path):
            logger.warning(f"⚠️ 이전 DB 파일 발견: {old_path}")

            conn = sqlite3.connect(old_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
            count = cursor.fetchone()[0]
            logger.info(f"   └─ 레코드 수: {count}개")
            conn.close()
