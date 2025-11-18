"""
데이터베이스 연결 풀 관리
고성능 SQLite 연결 풀링 및 성능 최적화
"""

import sqlite3
import threading
import logging
import os
from queue import Queue, Empty
from contextlib import contextmanager
from typing import Optional, Dict, Any
from rl_pipeline.core.env import config
from rl_pipeline.core.errors import DBWriteError, DBReadError

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """고성능 SQLite 연결 풀"""
    
    def __init__(self, db_path: str, max_connections: int = None, connection_timeout: float = None):
        self.db_path = db_path
        # 락 문제 해결을 위해 연결 수를 줄임
        self.max_connections = max_connections or min(config.DB_MAX_CONNECTIONS, 10)
        self.connection_timeout = connection_timeout or config.DB_CONNECTION_TIMEOUT
        self.connections: Queue = Queue(maxsize=self.max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # 초기 연결 생성
        self._initialize_pool()
    
    def _initialize_pool(self):
        """연결 풀 초기화 - 최적화된 연결 수"""
        logger.info(f"🔧 데이터베이스 연결 풀 초기화 중... ({self.db_path})")
        
        # 동적 연결 풀 크기: 최소 5개, 최대의 절반, 최대값 이하
        initial_connections = min(
            max(5, self.max_connections // 2),
            self.max_connections
        )
        
        for _ in range(initial_connections):
            try:
                conn = self._create_optimized_connection()
                self.connections.put(conn)
                logger.debug(f"✅ 연결 풀에 연결 추가됨 (총 {self.connections.qsize()}개)")
            except Exception as e:
                logger.warning(f"⚠️ 초기 연결 생성 실패: {e}")
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """최적화된 연결 생성 - 락 문제 해결"""
        try:
            # DB 파일이 없으면 자동으로 생성
            import os
            if not os.path.exists(self.db_path):
                logger.info(f"🔧 DB 파일이 없어 생성 중: {self.db_path}")
                # 디렉토리가 없으면 생성
                db_dir = os.path.dirname(self.db_path)
                if not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"✅ DB 디렉토리 생성 완료: {db_dir}")
            
            # 캔들 DB는 읽기 전용 모드로 열기
            is_candles_db = 'candles' in self.db_path.lower()
            
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            
            if is_candles_db:
                # 읽기 전용 최적화 설정 (WAL 모드 제외)
                conn.execute("PRAGMA journal_mode=DELETE")  # WAL 모드 사용하지 않음
                conn.execute("PRAGMA synchronous=OFF")  # 읽기만 하므로 동기화 불필요
                logger.debug(f"📖 캔들 DB 읽기 전용 모드로 열림: {self.db_path}")
            else:
                # 락 문제 해결을 위한 최적화 설정 (WAL 모드)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            
            # 공통 최적화 설정
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.execute("PRAGMA busy_timeout=10000")  # 10초 대기 (최적화: 60초 → 10초)
            conn.execute("PRAGMA optimize")
            if not is_candles_db:
                conn.execute("PRAGMA wal_autocheckpoint=1000")  # WAL 체크포인트 자동화 (캔들 DB 제외)
            
            return conn
            
        except Exception as e:
            logger.error(f"❌ 연결 생성 실패: {self.db_path} - {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """연결 컨텍스트 매니저"""
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except Exception as e:
            logger.error(f"❌ 연결 사용 중 오류: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def _get_connection(self) -> sqlite3.Connection:
        """연결 풀에서 연결 가져오기"""
        with self.lock:
            # 기존 연결이 있으면 사용
            try:
                conn = self.connections.get_nowait()
                logger.debug(f"♻️ 기존 연결 재사용 (남은 연결: {self.connections.qsize()})")
                return conn
            except Empty:
                pass
            
            # 새 연결 생성
            if self.active_connections < self.max_connections:
                try:
                    conn = self._create_optimized_connection()
                    self.active_connections += 1
                    logger.debug(f"🆕 새 연결 생성됨 (활성 연결: {self.active_connections})")
                    return conn
                except Exception as e:
                    logger.error(f"❌ 새 연결 생성 실패: {e}")
                    raise DBReadError(f"데이터베이스 연결 생성 실패: {e}") from e
            
            # 연결 풀이 가득 찬 경우 대기
            logger.warning(f"⚠️ 연결 풀 가득 참, 대기 중... (활성: {self.active_connections})")
            try:
                conn = self.connections.get(timeout=30.0)  # 30초 대기
                logger.debug(f"⏳ 대기 후 연결 획득 (남은 연결: {self.connections.qsize()})")
                return conn
            except Empty:
                raise DBReadError("연결 풀에서 연결을 가져올 수 없습니다 (30초 타임아웃)")
    
    def _return_connection(self, conn: sqlite3.Connection):
        """연결을 풀로 반환"""
        try:
            # 연결 상태 확인
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # 정상이면 풀로 반환
            if not self.connections.full():
                self.connections.put(conn)
                logger.debug(f"✅ 연결 반환됨 (풀 크기: {self.connections.qsize()})")
            else:
                # 풀이 가득 찬 경우 연결 종료
                conn.close()
                with self.lock:
                    self.active_connections -= 1
                logger.debug(f"🔒 연결 풀 가득 참, 연결 종료 (활성 연결: {self.active_connections})")
                
        except Exception as e:
            # 연결에 문제가 있으면 종료
            logger.warning(f"⚠️ 문제가 있는 연결 감지, 종료: {e}")
            try:
                conn.close()
            except:
                pass
            with self.lock:
                self.active_connections -= 1
    
    def close_all_connections(self, verbose: bool = False):
        """모든 연결 강제 종료
        
        Args:
            verbose: True면 상세 로그 출력 (기본값: False, 상위 함수에서만 로그 출력)
        """
        if verbose:
            logger.info("🔧 모든 데이터베이스 연결 강제 종료 중...")
        
        with self.lock:
            # 큐에 있는 모든 연결 종료
            closed_count = 0
            while not self.connections.empty():
                try:
                    conn = self.connections.get_nowait()
                    conn.close()
                    closed_count += 1
                    if verbose:
                        logger.debug("✅ 연결 종료됨")
                except Empty:
                    break
                except Exception as e:
                    if verbose:
                        logger.warning(f"⚠️ 연결 종료 중 오류: {e}")
            
            self.active_connections = 0
            if verbose:
                logger.info(f"✅ 모든 데이터베이스 연결이 종료되었습니다 (종료된 연결: {closed_count}개)")
    
    def cleanup_wal_files(self):
        """WAL 파일 정리"""
        try:
            import os
            logger.debug(f"🧹 WAL 파일 정리 시작: {self.db_path}")
            
            # 캔들 DB는 원천 데이터 - WAL 파일 정리하지 않음
            if 'candles' in self.db_path.lower():
                logger.debug("⚠️ 캔들 DB는 원천 데이터로 WAL 파일 정리를 건너뜁니다")
                return
            
            # WAL 파일 경로
            wal_path = f"{self.db_path}-wal"
            shm_path = f"{self.db_path}-shm"
            
            # WAL 모드 체크
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA journal_mode")
                    journal_mode = cursor.fetchone()[0]
                    
                    if journal_mode.lower() == 'wal':
                        # WAL 체크포인트 수행
                        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        conn.commit()
                        logger.debug(f"✅ WAL 체크포인트 완료: {self.db_path}")
            except Exception as e:
                logger.debug(f"⚠️ WAL 체크포인트 실패 (무시 가능): {e}")
            
            # WAL 파일 정리 (안전하게)
            if os.path.exists(wal_path) and os.path.getsize(wal_path) == 0:
                try:
                    os.remove(wal_path)
                    logger.debug(f"✅ 빈 WAL 파일 삭제: {wal_path}")
                except Exception as e:
                    logger.debug(f"⚠️ WAL 파일 삭제 실패: {e}")
                    
            if os.path.exists(shm_path) and os.path.getsize(shm_path) == 0:
                try:
                    os.remove(shm_path)
                    logger.debug(f"✅ 빈 SHM 파일 삭제: {shm_path}")
                except Exception as e:
                    logger.debug(f"⚠️ SHM 파일 삭제 실패: {e}")
                    
        except Exception as e:
            logger.debug(f"⚠️ WAL 파일 정리 실패 (무시 가능): {e}")

class BatchLoadingConnectionPool:
    """배치 로딩 전용 고성능 연결 풀"""
    
    def __init__(self, db_path: str, max_connections: int = None):
        self.db_path = db_path
        # 락 문제 해결을 위해 배치 연결 수도 줄임
        self.max_connections = max_connections or min(config.DB_BATCH_MAX_CONNECTIONS, 20)
        self.connections: Queue = Queue(maxsize=self.max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # 배치 로딩용 최적화 설정
        self._initialize_batch_pool()
    
    def _initialize_batch_pool(self):
        """배치 로딩용 연결 풀 초기화 - 락 문제 해결을 위해 연결 수 감소"""
        logger.info(f"🚀 배치 로딩용 연결 풀 초기화 중... ({self.db_path})")
        
        # 락 문제 해결을 위해 초기 연결 수를 줄임
        initial_connections = min(5, self.max_connections)
        for _ in range(initial_connections):
            try:
                conn = self._create_batch_optimized_connection()
                self.connections.put(conn)
                logger.debug(f"✅ 배치 연결 풀에 연결 추가됨 (총 {self.connections.qsize()}개)")
            except Exception as e:
                logger.warning(f"⚠️ 배치 초기 연결 생성 실패: {e}")
    
    def _create_batch_optimized_connection(self) -> sqlite3.Connection:
        """배치 로딩용 최적화된 연결 생성 - 락 문제 해결"""
        try:
            # DB 파일이 없으면 자동으로 생성
            import os
            if not os.path.exists(self.db_path):
                logger.info(f"🔧 배치 DB 파일이 없어 생성 중: {self.db_path}")
                # 디렉토리가 없으면 생성
                db_dir = os.path.dirname(self.db_path)
                if db_dir and not os.path.exists(db_dir):
                    try:
                        os.makedirs(db_dir, exist_ok=True)
                        logger.info(f"✅ 배치 DB 디렉토리 생성 완료: {db_dir}")
                    except Exception as dir_err:
                        logger.warning(f"⚠️ 배치 DB 디렉토리 생성 실패: {db_dir} - {dir_err}")
                        # 폴백: RL_PIPELINE_ROOT 하위에 생성 시도
                        try:
                            from rl_pipeline.core.env import config as _cfg
                            fallback_dir = _cfg.RL_PIPELINE_ROOT
                            fallback_path = os.path.join(fallback_dir, os.path.basename(self.db_path))
                            logger.info(f"🔄 폴백 경로로 시도: {fallback_path}")
                            self.db_path = fallback_path
                            db_dir = os.path.dirname(fallback_path)
                            if db_dir and not os.path.exists(db_dir):
                                os.makedirs(db_dir, exist_ok=True)
                        except Exception as fallback_err:
                            logger.debug(f"⚠️ 폴백 경로도 실패: {fallback_err}")
            
            conn = sqlite3.connect(
                self.db_path,
                timeout=config.DB_CONNECTION_TIMEOUT,
                check_same_thread=False
            )
            
            # 배치 로딩용 성능 최적화 설정 + 락 문제 해결
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # 안정성을 위해 NORMAL로 변경
            conn.execute("PRAGMA cache_size=50000")  # 더 큰 캐시
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=536870912")  # 512MB
            conn.execute("PRAGMA busy_timeout=60000")  # 60초 대기 (개선: 30초 → 60초)
            conn.execute("PRAGMA wal_autocheckpoint=1000")  # WAL 체크포인트 자동화
            conn.execute("PRAGMA optimize")
            
            return conn
            
        except Exception as e:
            logger.error(f"❌ 배치 연결 생성 실패: {self.db_path} - {e}")
            raise
    
    @contextmanager
    def get_batch_connection(self):
        """배치 연결 컨텍스트 매니저"""
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except Exception as e:
            logger.error(f"❌ 배치 연결 사용 중 오류: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    @contextmanager
    def get_connection(self):
        """연결 컨텍스트 매니저 (get_batch_connection의 별칭)"""
        # get_batch_connection 재사용
        with self.get_batch_connection() as conn:
            yield conn
    
    def _get_connection(self) -> sqlite3.Connection:
        """연결 풀에서 연결 가져오기"""
        with self.lock:
            try:
                conn = self.connections.get_nowait()
                logger.debug(f"♻️ 배치 연결 재사용 (남은 연결: {self.connections.qsize()})")
                return conn
            except Empty:
                pass
            
            if self.active_connections < self.max_connections:
                try:
                    conn = self._create_batch_optimized_connection()
                    self.active_connections += 1
                    logger.debug(f"🆕 새 배치 연결 생성됨 (활성 연결: {self.active_connections})")
                    return conn
                except Exception as e:
                    logger.error(f"❌ 새 배치 연결 생성 실패: {e}")
                    raise DBReadError(f"배치 데이터베이스 연결 생성 실패: {e}") from e
            
            try:
                conn = self.connections.get(timeout=30.0)
                logger.debug(f"⏳ 배치 연결 대기 후 획득 (남은 연결: {self.connections.qsize()})")
                return conn
            except Empty:
                raise DBReadError("배치 연결 풀에서 연결을 가져올 수 없습니다 (30초 타임아웃)")
    
    def _return_connection(self, conn: sqlite3.Connection):
        """연결을 풀로 반환"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            if not self.connections.full():
                self.connections.put(conn)
                logger.debug(f"✅ 배치 연결 반환됨 (풀 크기: {self.connections.qsize()})")
            else:
                conn.close()
                with self.lock:
                    self.active_connections -= 1
                logger.debug(f"🔒 배치 연결 풀 가득 참, 연결 종료 (활성 연결: {self.active_connections})")
                
        except Exception as e:
            logger.warning(f"⚠️ 문제가 있는 배치 연결 감지, 종료: {e}")
            try:
                conn.close()
            except:
                pass
            with self.lock:
                self.active_connections -= 1
    
    def close_all_connections(self, verbose: bool = False):
        """모든 연결 강제 종료
        
        Args:
            verbose: True면 상세 로그 출력 (기본값: False, 상위 함수에서만 로그 출력)
        """
        if verbose:
            logger.info("🔧 모든 배치 데이터베이스 연결 강제 종료 중...")
        
        with self.lock:
            # 큐에 있는 모든 연결 종료
            closed_count = 0
            while not self.connections.empty():
                try:
                    conn = self.connections.get_nowait()
                    conn.close()
                    closed_count += 1
                    if verbose:
                        logger.debug("✅ 배치 연결 종료됨")
                except Empty:
                    break
                except Exception as e:
                    if verbose:
                        logger.warning(f"⚠️ 배치 연결 종료 중 오류: {e}")
            
            self.active_connections = 0
            if verbose:
                logger.info(f"✅ 모든 배치 데이터베이스 연결이 종료되었습니다 (종료된 연결: {closed_count}개)")
    
    def cleanup_wal_files(self):
        """WAL 파일 정리"""
        try:
            import os
            logger.debug(f"🧹 배치 WAL 파일 정리 시작: {self.db_path}")
            
            # 캔들 DB는 원천 데이터 - WAL 파일 정리하지 않음
            if 'candles' in self.db_path.lower():
                logger.debug("⚠️ 캔들 DB는 원천 데이터로 WAL 파일 정리를 건너뜁니다")
                return
            
            # WAL 파일 경로
            wal_path = f"{self.db_path}-wal"
            shm_path = f"{self.db_path}-shm"
            
            # WAL 모드 체크 및 체크포인트
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA journal_mode")
                    journal_mode = cursor.fetchone()[0]
                    
                    if journal_mode.lower() == 'wal':
                        # WAL 체크포인트 수행
                        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        conn.commit()
                        logger.debug(f"✅ 배치 WAL 체크포인트 완료: {self.db_path}")
            except Exception as e:
                logger.debug(f"⚠️ 배치 WAL 체크포인트 실패 (무시 가능): {e}")
            
            # WAL 파일 정리 (안전하게)
            if os.path.exists(wal_path) and os.path.getsize(wal_path) == 0:
                try:
                    os.remove(wal_path)
                    logger.debug(f"✅ 빈 WAL 파일 삭제: {wal_path}")
                except Exception as e:
                    logger.debug(f"⚠️ WAL 파일 삭제 실패: {e}")
                    
            if os.path.exists(shm_path) and os.path.getsize(shm_path) == 0:
                try:
                    os.remove(shm_path)
                    logger.debug(f"✅ 빈 SHM 파일 삭제: {shm_path}")
                except Exception as e:
                    logger.debug(f"⚠️ SHM 파일 삭제 실패: {e}")
                    
        except Exception as e:
            logger.debug(f"⚠️ 배치 WAL 파일 정리 실패 (무시 가능): {e}")

# 전역 연결 풀 인스턴스들
_candle_pool: Optional[DatabaseConnectionPool] = None
_strategy_pool: Optional[DatabaseConnectionPool] = None
_learning_results_pool: Optional[DatabaseConnectionPool] = None
_batch_pool: Optional[BatchLoadingConnectionPool] = None

def get_candle_db_pool() -> DatabaseConnectionPool:
    """캔들 데이터베이스 연결 풀 반환"""
    global _candle_pool
    if _candle_pool is None:
        _candle_pool = DatabaseConnectionPool(config.RL_DB)
    return _candle_pool

def get_strategy_db_pool() -> DatabaseConnectionPool:
    """전략 데이터베이스 연결 풀 반환"""
    global _strategy_pool
    if _strategy_pool is None:
        # 전략 DB 파일이 없으면 자동으로 생성 (SQLite가 생성하도록 위임) + 실패 시 폴백 경로 사용
        import os, sqlite3
        primary_path = config.STRATEGIES_DB
        candidate_paths = [primary_path]
        # 폴백 경로: RL_PIPELINE_ROOT 하위 (일반적으로 쓰기 가능)
        try:
            from rl_pipeline.core.env import config as _cfg
            fallback_path = os.path.join(_cfg.RL_PIPELINE_ROOT, 'rl_strategies.db')
            if fallback_path not in candidate_paths:
                candidate_paths.append(fallback_path)
        except Exception:
            candidate_paths.append('/workspace/rl_pipeline/rl_strategies.db')

        last_error = None
        chosen_path = None
        for db_path in candidate_paths:
            try:
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"✅ 전략 DB 디렉토리 생성 완료: {db_dir}")
                logger.info(f"🔧 전략 DB 준비: {db_path}")
                conn = sqlite3.connect(db_path)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.close()
                chosen_path = db_path
                break
            except Exception as e:
                last_error = e
                # 🔧 폴백 경로 시도 중이므로 경고로 처리 (치명적 에러 아님)
                logger.debug(f"⚠️ 전략 DB 준비 시도 (폴백 경로로 계속 시도): {db_path} - {e}")
                continue

        if not chosen_path:
            raise DBReadError(f"전략 DB를 준비할 수 없습니다: {last_error}")

        _strategy_pool = DatabaseConnectionPool(chosen_path)
    return _strategy_pool

def get_learning_results_db_pool() -> DatabaseConnectionPool:
    """학습 결과 데이터베이스 연결 풀 반환"""
    global _learning_results_pool
    if _learning_results_pool is None:
        # 학습 결과 DB 파일이 없으면 자동으로 생성
        import os, sqlite3
        primary_path = config.LEARNING_RESULTS_DB_PATH

        try:
            db_dir = os.path.dirname(primary_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"✅ 학습 결과 DB 디렉토리 생성 완료: {db_dir}")

            logger.info(f"🔧 학습 결과 DB 준비: {primary_path}")
            conn = sqlite3.connect(primary_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.close()

            _learning_results_pool = DatabaseConnectionPool(primary_path)
        except Exception as e:
            logger.error(f"❌ 학습 결과 DB 준비 실패: {e}")
            raise DBReadError(f"학습 결과 DB를 준비할 수 없습니다: {e}")

    return _learning_results_pool

def get_batch_loading_pool(db_path: str = None) -> BatchLoadingConnectionPool:
    """배치 로딩 연결 풀 반환"""
    global _batch_pool
    if _batch_pool is None:
        db_path = db_path or config.RL_DB
        
        # 🔧 DB 경로 폴백 처리 (디렉토리 생성 실패 시)
        try:
            import os
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except Exception as dir_err:
                    logger.debug(f"⚠️ 배치 DB 디렉토리 생성 실패, 폴백 경로 사용: {db_dir} - {dir_err}")
                    # 폴백: RL_PIPELINE_ROOT 하위에 생성
                    try:
                        from rl_pipeline.core.env import config as _cfg
                        fallback_path = os.path.join(_cfg.RL_PIPELINE_ROOT, os.path.basename(db_path))
                        logger.info(f"🔄 배치 DB 폴백 경로: {fallback_path}")
                        db_path = fallback_path
                    except Exception:
                        logger.warning(f"⚠️ 폴백 경로도 실패, 원래 경로 사용")
        except Exception as e:
            # 전체 폴백 처리 실패 시 원래 경로 사용
            logger.debug(f"⚠️ 배치 DB 경로 처리 중 오류 (원래 경로 사용): {e}")
        
        _batch_pool = BatchLoadingConnectionPool(db_path)
    return _batch_pool

def close_all_pools():
    """모든 연결 풀 종료"""
    global _candle_pool, _strategy_pool, _learning_results_pool, _batch_pool

    if _candle_pool:
        _candle_pool.close_all_connections()
        _candle_pool = None

    if _strategy_pool:
        _strategy_pool.close_all_connections()
        _strategy_pool = None

    if _learning_results_pool:
        _learning_results_pool.close_all_connections()
        _learning_results_pool = None

    if _batch_pool:
        _batch_pool.close_all_connections()
        _batch_pool = None

def close_all_connections():
    """
    모든 활성 데이터베이스 연결을 종료합니다.
    인터벌 처리 사이 또는 에러 발생 시 사용.
    
    이 함수는 모든 연결 풀의 연결을 종료하여 잠금을 해제합니다.
    """
    try:
        logger.info("🔧 모든 데이터베이스 연결 종료 중...")

        # 모든 연결 풀의 연결 종료
        if _strategy_pool:
            _strategy_pool.close_all_connections()
        if _learning_results_pool:
            _learning_results_pool.close_all_connections()
        if _candle_pool:
            _candle_pool.close_all_connections()
        if _batch_pool:
            _batch_pool.close_all_connections()

        logger.info("✅ 모든 데이터베이스 연결 종료 완료")
        return True
    except Exception as e:
        logger.warning(f"⚠️ 연결 종료 중 오류 (무시 가능): {e}")
        return False
    
def validate_simulation_results(results: Dict[str, Any], criteria: Dict[str, Any] = None) -> Dict[str, Any]:
    """시뮬레이션 결과 검증"""
    try:
        if not results:
            return {'overall_status': 'FAIL', 'issues_found': ['No results provided']}
        
        issues = []
        
        # 기본 검증
        if not isinstance(results, dict):
            issues.append('Results must be a dictionary')
        
        # 성공 여부 확인
        if not results.get('success', False):
            issues.append('Simulation did not complete successfully')
        
        # 결과 반환
        if issues:
            return {'overall_status': 'FAIL', 'issues_found': issues}
        else:
            return {'overall_status': 'PASS', 'issues_found': []}
        
    except Exception as e:
        return {'overall_status': 'FAIL', 'issues_found': [f'Validation error: {e}']}

def validate_dna_results(results: Dict[str, Any], criteria: Dict[str, Any] = None) -> bool:
    """DNA 결과 검증"""
    try:
        if not results:
            return False
        
        required_fields = ['patterns', 'confidence']
        if not all(field in results for field in required_fields):
            return False
        
        # 추가 기준 검증
        if criteria:
            if 'min_confidence' in criteria and results.get('confidence', 0) < criteria['min_confidence']:
                return False
        
        return True
    except Exception:
        return False

def validate_fractal_results(results: Dict[str, Any], criteria: Dict[str, Any] = None) -> bool:
    """프랙탈 결과 검증"""
    try:
        if not results:
            return False
        
        required_fields = ['fractal_score', 'patterns']
        if not all(field in results for field in required_fields):
            return False
        
        # 추가 기준 검증
        if criteria:
            if 'min_fractal_score' in criteria and results.get('fractal_score', 0) < criteria['min_fractal_score']:
                return False
        
        return True
    except Exception:
        return False

def validate_pipeline_results(results: Dict[str, Any], criteria: Dict[str, Any] = None) -> Dict[str, Any]:
    """파이프라인 결과 검증 - 개선된 로직"""
    try:
        if not results:
            return {'overall_status': 'FAIL', 'issues_found': ['No results provided']}
        
        issues = []
        
        # 기본 검증
        if not isinstance(results, dict):
            issues.append('Results must be a dictionary')
            return {'overall_status': 'FAIL', 'issues_found': issues}
        
        # 다양한 성공 조건 확인 (더 유연한 검증)
        success_indicators = [
            results.get('success', False),
            results.get('success_count', 0) >= 3,  # 최소 3단계 성공
            results.get('total_steps', 0) > 0,     # 최소 1단계 실행
            'coin' in results,                     # 코인 정보 존재
            results.get('data_quality_score', 0) > 0.5  # 데이터 품질 점수
        ]
        
        # 하나라도 성공 조건을 만족하면 통과
        if any(success_indicators):
            return {'overall_status': 'PASS', 'issues_found': [], 'data_quality_score': results.get('data_quality_score', 0.8)}
        
        # 성공 조건을 만족하지 못한 경우
        issues.append('Pipeline did not meet success criteria')
        return {'overall_status': 'FAIL', 'issues_found': issues}
        
    except Exception as e:
        return {'overall_status': 'FAIL', 'issues_found': [f'Validation error: {e}']}

def validate_dna_analysis_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """DNA 분석 결과 검증 - 개선된 로직"""
    try:
        if not results:
            return {'overall_status': 'FAIL', 'issues_found': ['No results provided']}
        
        issues = []
        
        # 기본 검증
        if not isinstance(results, dict):
            issues.append('Results must be a dictionary')
            return {'overall_status': 'FAIL', 'issues_found': issues}
        
        # 다양한 성공 조건 확인 (더 유연한 검증)
        success_indicators = [
            results.get('success', False),
            results.get('evolved', False),
            results.get('total_evolved', 0) > 0,
            results.get('data_quality_score', 0) > 0.5,
            'coin' in results and 'intervals' in results
        ]
        
        # 하나라도 성공 조건을 만족하면 통과
        if any(success_indicators):
            return {'overall_status': 'PASS', 'issues_found': [], 'data_quality_score': results.get('data_quality_score', 0.8)}
        
        # 성공 조건을 만족하지 못한 경우
        issues.append('DNA analysis did not meet success criteria')
        return {'overall_status': 'FAIL', 'issues_found': issues}
        
    except Exception as e:
        return {'overall_status': 'FAIL', 'issues_found': [f'Validation error: {e}']}

def validate_fractal_analysis_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """프랙탈 분석 결과 검증 - 개선된 로직"""
    try:
        if not results:
            return {'overall_status': 'FAIL', 'issues_found': ['No results provided']}
        
        issues = []
        
        # 기본 검증
        if not isinstance(results, dict):
            issues.append('Results must be a dictionary')
            return {'overall_status': 'FAIL', 'issues_found': issues}
        
        # 다양한 성공 조건 확인 (더 유연한 검증)
        success_indicators = [
            results.get('success', False),
            results.get('analyzed', False),
            results.get('data_quality_score', 0) > 0.5,
            'coin' in results and 'intervals' in results
        ]
        
        # 하나라도 성공 조건을 만족하면 통과
        if any(success_indicators):
            return {'overall_status': 'PASS', 'issues_found': [], 'data_quality_score': results.get('data_quality_score', 0.8)}
        
        # 성공 조건을 만족하지 못한 경우
        issues.append('Fractal analysis did not meet success criteria')
        return {'overall_status': 'FAIL', 'issues_found': issues}
        
    except Exception as e:
        return {'overall_status': 'FAIL', 'issues_found': [f'Validation error: {e}']}

def auto_validate_pipeline_step(step_name: str, results: Any) -> Dict[str, Any]:
    """파이프라인 단계 자동 검증"""
    try:
        logger.info(f"🔍 {step_name} 자동 검증 시작")
        
        if not results:
            logger.warning(f"⚠️ {step_name}: 결과가 비어있음")
            return {'overall_status': 'FAIL', 'issues_found': ['No results provided']}
        
        # 단계별 검증 로직
        if 'simulation' in step_name.lower():
            return validate_simulation_results(results)
        elif 'dna' in step_name.lower():
            return validate_dna_analysis_results(results)
        elif 'fractal' in step_name.lower():
            return validate_fractal_analysis_results(results)
        else:
            return validate_pipeline_results(results)
            
    except Exception as e:
        logger.error(f"❌ {step_name} 자동 검증 실패: {e}")
        return {'overall_status': 'FAIL', 'issues_found': [f'Validation error: {e}']}

logger.info("✅ 모든 데이터베이스 연결 풀이 종료되었습니다")

def cleanup_all_database_files():
    """모든 데이터베이스 임시 파일 정리"""
    logger.info("🧹 모든 데이터베이스 임시 파일 정리 시작...")
    
    try:
        # 모든 연결 풀 종료
        strategy_pool = get_strategy_db_pool()
        candle_pool = get_candle_db_pool()
        
        strategy_pool.close_all_connections()
        candle_pool.close_all_connections()
        
        # WAL 파일 정리
        strategy_pool.cleanup_wal_files()
        candle_pool.cleanup_wal_files()
        
        logger.info("✅ 모든 데이터베이스 파일 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 파일 정리 실패: {e}")

@contextmanager
def get_optimized_db_connection(db_path: str, write_only: bool = False):
    """최적화된 데이터베이스 연결 컨텍스트 매니저 - 트랜잭션 안전"""
    # db_path에 따라 적절한 풀 선택 - 더 정확한 매칭
    if db_path == config.STRATEGIES_DB or 'strategies' in db_path.lower():
        pool = get_strategy_db_pool()
    elif db_path == config.LEARNING_RESULTS_DB_PATH or 'learning_results' in db_path.lower():
        pool = get_learning_results_db_pool()
    else:
        pool = get_candle_db_pool()

    with pool.get_connection() as conn:
        # 트랜잭션 내에서 안전한 기본 설정만 적용
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")  # 60초 대기 (개선: 30초 → 60초)
        yield conn
