"""
WAL (Write-Ahead Logging) 유틸리티 모듈
WAL 체크포인트 및 관련 작업 관리
"""

import sqlite3
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def checkpoint_wal_with_retry(
    conn: sqlite3.Connection,
    mode: str = "TRUNCATE",
    max_retries: int = 3,
    base_wait_time: float = 0.5
) -> bool:
    """
    WAL 체크포인트를 재시도 로직과 함께 실행
    
    Args:
        conn: SQLite 연결
        mode: 체크포인트 모드 ("PASSIVE", "FULL", "RESTART", "TRUNCATE")
        max_retries: 최대 재시도 횟수
        base_wait_time: 기본 대기 시간 (초)
    
    Returns:
        성공 여부
    """
    for attempt in range(max_retries):
        try:
            cursor = conn.cursor()
            
            # PASSIVE 모드로 먼저 시도 (읽기 전용, 락 없음)
            if attempt == 0 and mode != "PASSIVE":
                try:
                    result = cursor.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                    if result and result[0] == 0:  # 0 = 성공
                        logger.debug("✅ WAL 체크포인트 (PASSIVE) 성공")
                        return True
                except Exception as passive_error:
                    logger.debug(f"⚠️ WAL 체크포인트 (PASSIVE) 실패: {passive_error}")
            
            # 요청된 모드로 체크포인트 실행
            result = cursor.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
            conn.commit()
            
            if result and result[0] == 0:  # 0 = 성공
                logger.debug(f"✅ WAL 체크포인트 ({mode}) 성공")
                return True
            else:
                # 결과 코드: 0=성공, 1=SQLITE_BUSY, 2=SQLITE_LOCKED
                error_code = result[0] if result else -1
                logger.debug(f"⚠️ WAL 체크포인트 ({mode}) 결과 코드: {error_code}")
                
        except sqlite3.OperationalError as db_error:
            error_msg = str(db_error)
            logger.warning(f"⚠️ WAL 체크포인트 실패 (시도 {attempt + 1}/{max_retries}): {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (attempt + 1)
                logger.debug(f"⏳ {wait_time:.1f}초 후 재시도...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ WAL 체크포인트 최종 실패: {error_msg}")
                return False
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ WAL 체크포인트 예외 발생 (시도 {attempt + 1}/{max_retries}): {error_msg}")
            
            if attempt < max_retries - 1:
                time.sleep(base_wait_time)
            else:
                logger.error(f"❌ WAL 체크포인트 최종 실패: {error_msg}")
                return False
    
    return False


def get_wal_size(db_path: str) -> Optional[int]:
    """WAL 파일 크기 조회 (바이트)"""
    try:
        import os
        wal_path = f"{db_path}-wal"
        if os.path.exists(wal_path):
            return os.path.getsize(wal_path)
        return 0
    except Exception as e:
        logger.debug(f"⚠️ WAL 파일 크기 조회 실패: {e}")
        return None

