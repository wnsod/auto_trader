#!/usr/bin/env python3
"""
integrated_analysis_results 테이블 스키마 마이그레이션 스크립트

기존 스키마:
  id, coin, interval, signal_action, signal_score, confidence, created_at

새로운 스키마:
  id, coin, interval, regime, fractal_score, multi_timeframe_score,
  indicator_cross_score, ensemble_score, ensemble_confidence,
  final_signal_score, signal_confidence, signal_action, created_at

마이그레이션 방법:
1. 기존 데이터를 임시 테이블로 백업
2. 기존 테이블 삭제
3. 새로운 스키마로 테이블 재생성
4. 기존 데이터를 새 테이블로 복사 (컬럼 매핑)
5. 임시 테이블 삭제
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_integrated_analysis_results(db_path: str) -> bool:
    """integrated_analysis_results 테이블 스키마 마이그레이션"""
    try:
        # DB 파일 존재 확인
        if not Path(db_path).exists():
            logger.error(f"❌ DB 파일이 존재하지 않습니다: {db_path}")
            return False

        logger.info(f"🚀 DB 마이그레이션 시작: {db_path}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 1. 기존 테이블 존재 확인
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='integrated_analysis_results'
            """)
            if not cursor.fetchone():
                logger.info("ℹ️ integrated_analysis_results 테이블이 존재하지 않습니다. 새로 생성합니다.")
                create_new_table(cursor)
                conn.commit()
                logger.info("✅ 새 테이블 생성 완료")
                return True

            # 2. 기존 스키마 확인
            cursor.execute("PRAGMA table_info(integrated_analysis_results)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            logger.info(f"📊 기존 컬럼: {existing_columns}")

            # 3. 이미 새 스키마인지 확인
            required_columns = {
                'regime', 'fractal_score', 'multi_timeframe_score',
                'indicator_cross_score', 'ensemble_score', 'ensemble_confidence',
                'final_signal_score', 'signal_confidence'
            }
            if required_columns.issubset(existing_columns):
                logger.info("✅ 테이블이 이미 새 스키마로 되어 있습니다. 마이그레이션 불필요.")
                return True

            # 4. 기존 데이터 개수 확인
            cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
            data_count = cursor.fetchone()[0]
            logger.info(f"📊 기존 데이터 개수: {data_count}개")

            # 5. 기존 데이터를 임시 테이블로 백업
            logger.info("💾 기존 데이터 백업 중...")
            cursor.execute("""
                CREATE TABLE integrated_analysis_results_backup AS
                SELECT * FROM integrated_analysis_results
            """)
            logger.info(f"✅ {data_count}개 데이터 백업 완료")

            # 6. 기존 테이블 삭제
            logger.info("🗑️ 기존 테이블 삭제 중...")
            cursor.execute("DROP TABLE integrated_analysis_results")
            logger.info("✅ 기존 테이블 삭제 완료")

            # 7. 새 스키마로 테이블 생성
            logger.info("🏗️ 새 테이블 생성 중...")
            create_new_table(cursor)
            logger.info("✅ 새 테이블 생성 완료")

            # 8. 기존 데이터를 새 테이블로 복사
            logger.info("📥 데이터 마이그레이션 중...")

            # 기존 데이터의 컬럼 구조 확인
            cursor.execute("PRAGMA table_info(integrated_analysis_results_backup)")
            backup_columns = [row[1] for row in cursor.fetchall()]

            if data_count > 0:
                # 컬럼 매핑 (구 스키마 → 새 스키마)
                # signal_score → final_signal_score
                # confidence → signal_confidence
                if 'signal_score' in backup_columns and 'confidence' in backup_columns:
                    cursor.execute("""
                        INSERT INTO integrated_analysis_results
                        (coin, interval, regime, fractal_score, multi_timeframe_score,
                         indicator_cross_score, ensemble_score, ensemble_confidence,
                         final_signal_score, signal_confidence, signal_action, created_at)
                        SELECT
                            coin,
                            interval,
                            'neutral' AS regime,
                            0.0 AS fractal_score,
                            0.0 AS multi_timeframe_score,
                            0.0 AS indicator_cross_score,
                            0.0 AS ensemble_score,
                            0.0 AS ensemble_confidence,
                            signal_score AS final_signal_score,
                            confidence AS signal_confidence,
                            signal_action,
                            created_at
                        FROM integrated_analysis_results_backup
                    """)
                    logger.info(f"✅ {data_count}개 데이터 마이그레이션 완료")
                else:
                    logger.warning("⚠️ 기존 데이터 구조가 예상과 다릅니다. 데이터 마이그레이션을 건너뜁니다.")

            # 9. 백업 테이블 삭제
            logger.info("🗑️ 백업 테이블 삭제 중...")
            cursor.execute("DROP TABLE integrated_analysis_results_backup")
            logger.info("✅ 백업 테이블 삭제 완료")

            # 10. 새 테이블 데이터 개수 확인
            cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
            new_data_count = cursor.fetchone()[0]
            logger.info(f"📊 마이그레이션 후 데이터 개수: {new_data_count}개")

            conn.commit()
            logger.info("✅ 마이그레이션 완료!")

            # 11. 결과 검증
            if new_data_count == data_count:
                logger.info("✅ 데이터 무결성 확인 완료")
            else:
                logger.warning(f"⚠️ 데이터 개수 불일치: {data_count} → {new_data_count}")

            return True

    except Exception as e:
        logger.error(f"❌ 마이그레이션 실패: {e}")
        import traceback
        logger.error(f"상세 에러:\n{traceback.format_exc()}")
        return False


def create_new_table(cursor: sqlite3.Cursor):
    """새 스키마로 테이블 생성"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS integrated_analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            interval TEXT NOT NULL,
            regime TEXT NOT NULL DEFAULT 'neutral',

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
        CREATE INDEX IF NOT EXISTS idx_integrated_analysis_final_signal_score
        ON integrated_analysis_results(final_signal_score DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_integrated_analysis_created_at
        ON integrated_analysis_results(created_at DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_integrated_analysis_regime
        ON integrated_analysis_results(regime)
    """)


if __name__ == "__main__":
    import sys

    # DB 경로 (기본값: rl_pipeline/data_storage/rl_strategies.db)
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # 도커 내부 경로
        db_path = "/workspace/data_storage/rl_strategies.db"

        # 로컬 경로로 대체 (파일이 없으면)
        if not Path(db_path).exists():
            db_path = "./data_storage/rl_strategies.db"

    logger.info(f"🎯 대상 DB: {db_path}")

    success = migrate_integrated_analysis_results(db_path)

    if success:
        logger.info("🎉 마이그레이션 성공!")
        sys.exit(0)
    else:
        logger.error("💥 마이그레이션 실패!")
        sys.exit(1)
