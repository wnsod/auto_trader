import os
import sqlite3
from typing import Generator, Optional

# 기본 DB 경로 설정 (프로젝트 루트 기준)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 시장별 DB 경로 정의 (확장성을 위해 딕셔너리로 관리 가능하지만, 명시적으로 선언)
COIN_DB_PATH = os.path.join(BASE_DIR, "market", "coin_market", "data_storage", "trading_system.db")
KR_DB_PATH = os.path.join(BASE_DIR, "market", "kr_market", "data_storage", "trading_system.db")
US_DB_PATH = os.path.join(BASE_DIR, "market", "us_market", "data_storage", "trading_system.db")
FOREX_DB_PATH = os.path.join(BASE_DIR, "market", "forex_market", "data_storage", "trading_system.db")
BOND_DB_PATH = os.path.join(BASE_DIR, "market", "bond_market", "data_storage", "trading_system.db")
COMMODITY_DB_PATH = os.path.join(BASE_DIR, "market", "commodity_market", "data_storage", "trading_system.db")

def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """SQLite DB 연결 생성 (읽기 전용). 파일이 없으면 None 반환."""
    if not os.path.exists(db_path):
        # print(f"⚠️ DB Not Found: {db_path}") # 디버그 로그 노이즈 제거
        return None
    
    try:
        # URI 모드로 열어서 읽기 전용으로 접근 (Lock 방지 및 안전성)
        # check_same_thread=False 추가: FastAPI의 멀티스레드 환경(Thread Pool)에서 커넥션 공유 허용
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Connection Failed: {db_path} - {e}")
        return None

# --- Dependency Injection Functions (FastAPI Depends용) ---

def get_coin_db():
    conn = get_db_connection(COIN_DB_PATH)
    try:
        yield conn
    finally:
        if conn: conn.close()

def get_kr_db():
    conn = get_db_connection(KR_DB_PATH)
    try:
        yield conn
    finally:
        if conn: conn.close()

def get_us_db():
    conn = get_db_connection(US_DB_PATH)
    try:
        yield conn
    finally:
        if conn: conn.close()

def get_forex_db():
    conn = get_db_connection(FOREX_DB_PATH)
    try:
        yield conn
    finally:
        if conn: conn.close()

def get_bond_db():
    conn = get_db_connection(BOND_DB_PATH)
    try:
        yield conn
    finally:
        if conn: conn.close()

def get_commodity_db():
    conn = get_db_connection(COMMODITY_DB_PATH)
    try:
        yield conn
    finally:
        if conn: conn.close()
