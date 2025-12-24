from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Optional
import sqlite3
from pydantic import BaseModel
from api.database import get_coin_db

router = APIRouter(
    prefix="/api/coin",
    tags=["coin_market"]
)

# --- Pydantic Models ---
class Position(BaseModel):
    coin: str
    entry_price: float
    current_price: float
    profit_loss_pct: float
    quantity: float
    entry_timestamp: int
    holding_duration: int

class TradeHistory(BaseModel):
    coin: str
    action: str
    profit_loss_pct: float
    entry_price: float
    exit_price: float
    entry_timestamp: int
    exit_timestamp: int

class PerformanceStats(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit_pct: float
    active_positions: int
    timestamp: int

# --- Endpoints ---

@router.get("/positions", response_model=List[Position])
def get_active_positions(db: sqlite3.Connection = Depends(get_coin_db)):
    """현재 보유 중인 가상 포지션 조회"""
    query = """
        SELECT coin, entry_price, current_price, profit_loss_pct, quantity, 
               entry_timestamp, holding_duration
        FROM virtual_positions
        ORDER BY profit_loss_pct DESC
    """
    cursor = db.execute(query)
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

@router.get("/history", response_model=List[TradeHistory])
def get_trade_history(limit: int = 50, db: sqlite3.Connection = Depends(get_coin_db)):
    """최근 거래 내역 조회"""
    query = """
        SELECT coin, action, profit_loss_pct, entry_price, exit_price,
               entry_timestamp, exit_timestamp
        FROM virtual_trade_history
        ORDER BY exit_timestamp DESC
        LIMIT ?
    """
    cursor = db.execute(query, (limit,))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

@router.get("/stats", response_model=PerformanceStats)
def get_performance_stats(db: sqlite3.Connection = Depends(get_coin_db)):
    """최신 성과 통계 조회"""
    query = """
        SELECT total_trades, winning_trades, losing_trades, win_rate, 
               total_profit_pct, active_positions, timestamp
        FROM virtual_performance_stats
        ORDER BY timestamp DESC
        LIMIT 1
    """
    cursor = db.execute(query)
    row = cursor.fetchone()
    
    if not row:
        # 데이터가 없을 경우 기본값 반환
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "total_profit_pct": 0.0, "active_positions": 0,
            "timestamp": 0
        }
    return dict(row)

@router.get("/summary")
def get_market_summary(db: sqlite3.Connection = Depends(get_coin_db)):
    """대시보드 상단 요약 정보 (간단 조회)"""
    # 1. 총 포지션 수 및 수익/손실 수
    pos_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN profit_loss_pct > 0 THEN 1 ELSE 0 END) as profit_count,
            SUM(CASE WHEN profit_loss_pct <= 0 THEN 1 ELSE 0 END) as loss_count,
            AVG(profit_loss_pct) as avg_pnl
        FROM virtual_positions
    """
    pos_row = db.execute(pos_query).fetchone()
    
    # 2. 최근 24시간 거래 횟수 (약식 계산)
    # 현재 시간 - 24시간 (초 단위)
    import time
    one_day_ago = int(time.time()) - 86400
    vol_query = "SELECT COUNT(*) as count FROM virtual_trade_history WHERE exit_timestamp > ?"
    vol_row = db.execute(vol_query, (one_day_ago,)).fetchone()

    return {
        "active_positions": pos_row["total"],
        "profitable_positions": pos_row["profit_count"] or 0,
        "losing_positions": pos_row["loss_count"] or 0,
        "average_pnl": pos_row["avg_pnl"] or 0.0,
        "recent_trades_24h": vol_row["count"]
    }

