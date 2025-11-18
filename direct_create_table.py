"""
직접 SQL로 integrated_analysis_results 테이블 생성
"""
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = '/workspace/data_storage/learning_results.db'

if __name__ == "__main__":
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        logger.info(f"🚀 {DB_PATH}에 테이블 생성 시작")

        # integrated_analysis_results 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrated_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                regime TEXT NOT NULL,

                -- 분석 결과
                fractal_score REAL DEFAULT 0.0,
                multi_timeframe_score REAL DEFAULT 0.0,
                indicator_cross_score REAL DEFAULT 0.0,

                -- JAX 앙상블 결과
                ensemble_score REAL DEFAULT 0.0,
                ensemble_confidence REAL DEFAULT 0.0,

                -- 최종 시그널 점수
                final_signal_score REAL DEFAULT 0.0,
                signal_confidence REAL DEFAULT 0.0,
                signal_action TEXT DEFAULT 'hold',

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_integrated_analysis_coin_interval
            ON integrated_analysis_results(coin, interval)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_integrated_analysis_created_at
            ON integrated_analysis_results(created_at DESC)
        """)

        conn.commit()
        logger.info("✅ integrated_analysis_results 테이블 생성 완료")

        # 테이블 확인
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        logger.info(f"📊 현재 테이블 목록 ({len(tables)}개):")
        for t in tables:
            logger.info(f"   - {t[0]}")

        # 스키마 확인
        schema = conn.execute("PRAGMA table_info(integrated_analysis_results)").fetchall()
        logger.info(f"\n📋 integrated_analysis_results 스키마:")
        for col in schema:
            logger.info(f"   - {col[1]}: {col[2]}")

        conn.close()

    except Exception as e:
        logger.error(f"❌ 테이블 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
