"""
데이터베이스 유틸리티 모듈
"""

import sqlite3
from contextlib import contextmanager
from typing import Tuple, Optional

# DB 경로는 config.py에서 import
try:
    from ..config import CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH, DB_PATH
except ImportError:
    # fallback: 직접 계산
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    workspace_dir = os.path.dirname(current_dir)
    CANDLES_DB_PATH = os.getenv('CANDLES_DB_PATH', os.path.join(workspace_dir, 'data_storage', 'realtime_candles.db'))
    STRATEGIES_DB_PATH = os.getenv('STRATEGIES_DB_PATH', os.path.join(workspace_dir, 'data_storage', 'learning_results.db'))
    TRADING_SYSTEM_DB_PATH = os.path.join(workspace_dir, 'data_storage', 'trading_system.db')
    DB_PATH = TRADING_SYSTEM_DB_PATH


@contextmanager
def get_optimized_db_connection(db_path: str, mode: str = 'read'):
    """
    최적화된 데이터베이스 연결 컨텍스트 매니저
    
    Args:
        db_path: 데이터베이스 경로
        mode: 연결 모드 ('read' 또는 'write')
    
    Yields:
        SQLite 연결 객체
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        if mode == 'write':
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


@contextmanager
def safe_db_write(db_path: str, operation_name: str):
    """
    안전한 데이터베이스 쓰기 컨텍스트 매니저
    
    Args:
        db_path: 데이터베이스 경로
        operation_name: 작업 이름 (로깅용)
    
    Yields:
        SQLite 연결 객체
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"⚠️ 데이터베이스 쓰기 오류 ({operation_name}): {e}")
        raise e
    finally:
        if conn:
            conn.close()


def get_strategy_db_pool():
    """전략 데이터베이스 풀 반환 (호환성)"""
    return None


def get_candle_db_pool():
    """캔들 데이터베이스 풀 반환 (호환성)"""
    return None


def get_conflict_manager():
    """충돌 관리자 반환 (호환성)"""
    return None


def safe_db_read(query: str, params: Tuple = (), db_path: Optional[str] = None):
    """
    안전한 데이터베이스 읽기 함수
    
    Args:
        query: SQL 쿼리 문자열
        params: 쿼리 파라미터 튜플
        db_path: 데이터베이스 경로 (None이면 STRATEGIES_DB_PATH 사용)
    
    Returns:
        쿼리 결과 리스트 (오류 시 빈 리스트)
    """
    try:
        if db_path is None:
            db_path = STRATEGIES_DB_PATH
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"⚠️ 데이터베이스 읽기 오류: {e}")
        return []


def safe_db_write_func(query: str, params: Tuple = (), db_path: Optional[str] = None) -> bool:
    """
    안전한 데이터베이스 쓰기 함수
    
    Args:
        query: SQL 쿼리 문자열
        params: 쿼리 파라미터 튜플
        db_path: 데이터베이스 경로 (None이면 STRATEGIES_DB_PATH 사용)
    
    Returns:
        성공 여부 (bool)
    """
    try:
        if db_path is None:
            db_path = STRATEGIES_DB_PATH
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ 데이터베이스 쓰기 오류: {e}")
        return False

