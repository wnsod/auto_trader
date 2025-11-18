"""
데이터베이스 연결 풀 모듈 - 읽기/쓰기 분리 연결 풀
"""

import sqlite3
import threading
from typing import Optional


class DatabasePool:
    """
    🚀 데이터베이스 연결 풀 - 충돌 방지 강화
    
    읽기/쓰기 연결을 분리하여 동시성 문제를 방지하고 성능을 최적화합니다.
    """
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.write_pool = []
        self.read_pool = []
        self.write_lock = threading.Lock()
        self.read_lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """연결 풀 초기화 - 읽기/쓰기 분리"""
        for _ in range(self.max_connections):
            # 쓰기용 연결
            write_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            write_conn.execute("PRAGMA journal_mode=WAL")  # WAL 모드로 동시성 향상
            write_conn.execute("PRAGMA synchronous=NORMAL")  # 성능 최적화
            write_conn.execute("PRAGMA cache_size=10000")  # 캐시 크기 증가
            write_conn.execute("PRAGMA temp_store=MEMORY")  # 임시 테이블을 메모리에
            write_conn.execute("PRAGMA read_uncommitted = 0")  # 쓰기 모드
            self.write_pool.append(write_conn)
            
            # 읽기용 연결
            read_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            read_conn.execute("PRAGMA journal_mode=WAL")
            read_conn.execute("PRAGMA synchronous=NORMAL")
            read_conn.execute("PRAGMA cache_size=10000")
            read_conn.execute("PRAGMA temp_store=MEMORY")
            read_conn.execute("PRAGMA read_uncommitted = 1")  # 읽기 전용 모드
            self.read_pool.append(read_conn)
    
    def get_connection(self, read_only: bool = False) -> sqlite3.Connection:
        """
        연결 풀에서 연결 가져오기 - 읽기/쓰기 분리
        
        Args:
            read_only: 읽기 전용 연결 여부
        
        Returns:
            SQLite 연결 객체
        """
        if read_only:
            with self.read_lock:
                if self.read_pool:
                    return self.read_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA read_uncommitted = 1")
                    return conn
        else:
            with self.write_lock:
                if self.write_pool:
                    return self.write_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA read_uncommitted = 0")
                    return conn
    
    def return_connection(self, conn: sqlite3.Connection, read_only: bool = False) -> None:
        """
        연결 풀에 연결 반환 - 읽기/쓰기 분리
        
        Args:
            conn: 반환할 연결 객체
            read_only: 읽기 전용 연결 여부
        """
        if read_only:
            with self.read_lock:
                if len(self.read_pool) < self.max_connections:
                    self.read_pool.append(conn)
                else:
                    conn.close()
        else:
            with self.write_lock:
                if len(self.write_pool) < self.max_connections:
                    self.write_pool.append(conn)
                else:
                    conn.close()

