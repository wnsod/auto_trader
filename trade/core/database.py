#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
트레이딩 시스템 데이터베이스 통합 관리 모듈
"""

import sqlite3
import os
import time
import traceback
from typing import Dict, List, Optional, Any

# 📂 데이터 저장소 및 DB 경로 설정 (환경 변수 우선, 엔진 모드)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def finalize_path(path):
    """경로를 절대 경로로 변환 (Docker/Windows 호환성 강화)"""
    if not path: return None
    
    # 🚀 Windows 호스트에서 /workspace 경로가 들어올 경우 로컬 경로로 변환
    if os.name == 'nt' and isinstance(path, str):
        if path.startswith('/workspace') or path.startswith('\\workspace'):
            rel_path = path.replace('/workspace', '', 1).replace('\\workspace', '', 1).lstrip('/\\')
            return os.path.abspath(os.path.join(_BASE_DIR, rel_path))
    
    return os.path.abspath(path)

# 1. 데이터 저장소 루트
DATA_DIR = finalize_path(os.environ.get('DATA_STORAGE_PATH'))
if not DATA_DIR:
    DATA_DIR = finalize_path(os.path.join(_BASE_DIR, 'market', 'coin_market', 'data_storage'))

# 2. 매매 시스템 DB (trading_system.db)
TRADING_SYSTEM_DB_PATH = finalize_path(os.environ.get('TRADING_SYSTEM_DB_PATH'))
if not TRADING_SYSTEM_DB_PATH:
    TRADING_SYSTEM_DB_PATH = os.path.join(DATA_DIR, 'trading_system.db')

# 3. 전략/학습 DB (Thompson 및 글로벌 전략 공용)
_STRATEGY_ENV = os.environ.get('STRATEGY_DB_PATH')
if _STRATEGY_ENV:
    _STRATEGY_ENV = finalize_path(_STRATEGY_ENV)
    if os.path.isdir(_STRATEGY_ENV):
        candidate = os.path.join(_STRATEGY_ENV, 'common_strategies.db')
        if not os.path.exists(candidate):
            alt_candidate = os.path.join(_STRATEGY_ENV, 'learning_strategies.db')
            if os.path.exists(alt_candidate):
                candidate = alt_candidate
        STRATEGY_DB_PATH = candidate
    else:
        STRATEGY_DB_PATH = _STRATEGY_ENV
else:
    STRATEGY_DB_PATH = os.path.join(DATA_DIR, 'learning_strategies', 'common_strategies.db')

if not os.environ.get('GLOBAL_STRATEGY_DB_PATH'):
    os.environ['GLOBAL_STRATEGY_DB_PATH'] = STRATEGY_DB_PATH

# 4. 캔들 DB (trade_candles.db)
CANDLES_DB_PATH = finalize_path(os.environ.get('CANDLES_DB_PATH'))
if not CANDLES_DB_PATH:
    CANDLES_DB_PATH = os.path.join(DATA_DIR, 'trade_candles.db')

# 📁 디렉토리 존재 보장
for path in [TRADING_SYSTEM_DB_PATH, STRATEGY_DB_PATH, CANDLES_DB_PATH]:
    if not path: continue
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

def apply_sqlite_pragmas(conn: sqlite3.Connection, read_only: bool = False):
    """SQLite 성능 및 안정성 최적화 설정"""
    try:
        if not read_only:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
    except: pass

def get_db_connection(db_path: str, read_only: bool = True, **kwargs) -> sqlite3.Connection:
    """DB 연결 객체 생성 및 최적화 설정 적용"""
    try:
        abs_path = os.path.abspath(db_path)
        timeout = kwargs.get('timeout', 15.0)
        
        if read_only:
            if not os.path.exists(abs_path):
                return sqlite3.connect(abs_path, timeout=timeout)
            uri_path = abs_path.replace("\\", "/")
            if not uri_path.startswith("/"): uri_path = "/" + uri_path
            conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=timeout)
        else:
            conn = sqlite3.connect(abs_path, timeout=timeout)
            
        apply_sqlite_pragmas(conn, read_only=read_only)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            apply_sqlite_pragmas(conn, read_only=read_only)
            conn.row_factory = sqlite3.Row
            return conn
        except:
            raise sqlite3.OperationalError(f"DB 연결 실패 ({db_path}): {e}")

def get_latest_candle_timestamp() -> int:
    """캔들 DB에서 가장 최신 데이터의 타임스탬프를 가져옴 (시스템의 절대 기준 시각)"""
    try:
        # 🚀 [Fix] PC 시각(time.time)이 아닌 오직 DB 데이터만 기준
        with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
            row = conn.execute("SELECT MAX(timestamp) FROM candles").fetchone()
            if row and row[0]:
                return int(row[0])
    except Exception as e:
        print(f"⚠️ 기준 시각 조회 실패: {e}")
    # DB에 데이터가 아예 없는 경우에만 시스템 시각으로 폴백 (최초 실행 대비)
    return int(time.time())

# 🚀 [Performance] 메모리 캐시
_LEARNING_CACHE = {}
_CACHE_EXPIRY = 60 # 1분

def get_learning_data(coin: str, interval: str, table: str = 'integrated_analysis_results') -> Optional[Dict]:
    """학습 결과 데이터를 읽어옴 (통합 분석 결과는 공용 DB 우선, 전략은 코인 DB 우선)"""
    cache_key = f"{coin}_{interval}_{table}"
    
    # 1. 캐시 확인
    now = time.time()
    if cache_key in _LEARNING_CACHE:
        cache_data, expiry = _LEARNING_CACHE[cache_key]
        if now < expiry:
            return cache_data

    # 2. DB 조회 경로 결정
    strat_dir = os.path.dirname(STRATEGY_DB_PATH)
    coin_db_path = os.path.join(strat_dir, f"{coin.lower()}_strategies.db")
    
    # 🎯 [핵심 보정] 통합 분석 결과는 보통 common_strategies.db에 저장됨
    if table == 'integrated_analysis_results':
        target_dbs = [STRATEGY_DB_PATH, coin_db_path]
    else:
        target_dbs = [coin_db_path, STRATEGY_DB_PATH]
    
    for target_db in target_dbs:
        if not target_db or not os.path.exists(target_db):
            continue
            
        try:
            uri_path = os.path.abspath(target_db).replace("\\", "/")
            if not uri_path.startswith("/"): uri_path = "/" + uri_path
            
            with sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if not cursor.fetchone():
                    continue

                cursor.execute(f"PRAGMA table_info({table})")
                cols = [r[1].lower() for r in cursor.fetchall()]
                coin_col = 'symbol' if 'symbol' in cols else 'coin' if 'coin' in cols else None
                
                if not coin_col:
                    continue

                query = f"SELECT * FROM {table} WHERE ({coin_col} = ? OR {coin_col} = ?) AND interval = ? ORDER BY created_at DESC LIMIT 1"
                cursor.execute(query, (coin.upper(), coin.lower(), interval))
                row = cursor.fetchone()
                
                if row:
                    result = dict(row)
                    _LEARNING_CACHE[cache_key] = (result, now + _CACHE_EXPIRY)
                    return result
        except Exception:
            continue
            
    return None

def save_trade_decision(decision_data: Dict):
    """가상/실전 매매 결정을 DB에 저장"""
    try:
        coin = decision_data.get('coin')
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
            conn.execute("DELETE FROM virtual_trade_decisions WHERE coin = ? AND processed = 0", (coin,))
            conn.execute("""
                INSERT INTO virtual_trade_decisions (
                    coin, timestamp, decision, signal_score, confidence, current_price,
                    target_price, expected_profit_pct, thompson_score, thompson_approved,
                    regime_score, regime_name, viability_passed, reason,
                    is_holding, entry_price, profit_loss_pct, trend_type,
                    wave_phase, integrated_direction, processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                decision_data.get('coin'),
                decision_data.get('timestamp', int(time.time())),
                decision_data.get('decision', 'skip'),
                decision_data.get('signal_score', 0.0),
                decision_data.get('confidence', 0.0),
                decision_data.get('current_price', 0.0),
                decision_data.get('target_price', 0.0),
                decision_data.get('expected_profit_pct', 0.0),
                decision_data.get('thompson_score', 0.0),
                1 if decision_data.get('thompson_approved', False) else 0,
                decision_data.get('regime_score', 0.5),
                decision_data.get('regime_name', 'Neutral'),
                1 if decision_data.get('viability_passed', False) else 0,
                decision_data.get('reason', ''),
                1 if decision_data.get('is_holding', False) else 0,
                decision_data.get('entry_price', 0.0),
                decision_data.get('profit_loss_pct', 0.0),
                decision_data.get('trend_type', ''),
                decision_data.get('wave_phase', 'unknown'),
                decision_data.get('integrated_direction', 'neutral')
            ))
            conn.commit()
    except Exception as e:
        print(f"⚠️ 의사결정 저장 오류: {e}")

def save_trade_history(trade_record: Dict, table_name: str = 'virtual_trade_history'):
    """거래 내역 저장"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
                columns = ', '.join(trade_record.keys())
                placeholders = ', '.join(['?' for _ in trade_record])
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                conn.execute(query, tuple(trade_record.values()))
                conn.commit()
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                print(f"🚨 DB 저장 최종 실패: {e}")
                traceback.print_exc()
    return False
