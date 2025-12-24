"""
리그 시스템 스키마 업데이트
- strategies 테이블에 league 컬럼 추가
- 기존 컬럼 확인 및 안전한 마이그레이션
"""

import sqlite3
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

def add_league_column_to_strategies(db_path: str):
    """strategies 테이블에 league 컬럼 추가"""
    if not os.path.exists(db_path):
        logger.warning(f"⚠️ DB 파일이 존재하지 않음: {db_path}")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 테이블 컬럼 확인
            cursor.execute("PRAGMA table_info(strategies)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'league' not in columns:
                logger.info(f"🔧 {db_path}: league 컬럼 추가 중...")
                # league 컬럼 추가 (기본값: 'minor')
                cursor.execute("ALTER TABLE strategies ADD COLUMN league TEXT DEFAULT 'minor'")
                
                # 인덱스 추가 (빠른 조회를 위해)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategies_league ON strategies(league)")
                conn.commit()
                logger.info(f"✅ {db_path}: league 컬럼 추가 완료")
            else:
                logger.debug(f"ℹ️ {db_path}: league 컬럼 이미 존재함")
                
    except Exception as e:
        logger.error(f"❌ {db_path}: league 컬럼 추가 실패: {e}")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 전략 DB 디렉토리 탐색 및 업데이트
    base_dir = "market/kr_market/data_storage/learning_strategies"
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            if filename.endswith("_strategies.db") or filename == "common_strategies.db":
                db_path = os.path.join(base_dir, filename)
                add_league_column_to_strategies(db_path)
    else:
        logger.warning(f"⚠️ 전략 디렉토리를 찾을 수 없음: {base_dir}")

