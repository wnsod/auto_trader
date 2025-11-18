"""
베이스라인 전략 비교 모듈
- Buy & Hold 전략과 현재 전략 비교
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def compare_with_baseline(
    coin: str,
    interval: str,
    test_period_days: int = 30,
    db_path: str = "data_storage/rl_strategies.db",
    candles_db_path: str = "data_storage/rl_candles.db"
) -> Dict:
    """현재 전략 vs 베이스라인 전략 비교"""
    
    try:
        # 1. Buy & Hold 전략
        buy_hold_return = calculate_buy_hold_return(
            coin, interval, test_period_days, candles_db_path
        )
        
        # 2. 현재 전략 평균
        current_strategy_return = calculate_current_strategy_avg(
            coin, interval, db_path
        )
        
        # 3. 비교
        comparison = {
            'coin': coin,
            'interval': interval,
            'buy_hold_return': buy_hold_return,
            'current_strategy_return': current_strategy_return,
            'difference': current_strategy_return - buy_hold_return if current_strategy_return is not None else None,
            'outperforms': (current_strategy_return > buy_hold_return) if current_strategy_return is not None else False
        }
        
        logger.info(f"\n{coin}-{interval} 베이스라인 비교:")
        logger.info(f"  Buy & Hold: {buy_hold_return:.2f}%")
        if current_strategy_return is not None:
            logger.info(f"  현재 전략: {current_strategy_return:.2f}%")
            logger.info(f"  차이: {comparison['difference']:.2f}%")
            
            if comparison['outperforms']:
                logger.info(f"  ✅ 현재 전략이 우수")
            else:
                logger.warning(f"  ❌ Buy & Hold가 더 나음 (전략 재설계 필요)")
        else:
            logger.warning(f"  현재 전략 데이터 없음")
        
        return comparison
    
    except Exception as e:
        logger.error(f"❌ 베이스라인 비교 실패: {e}")
        return {
            'coin': coin,
            'interval': interval,
            'error': str(e)
        }


def calculate_buy_hold_return(
    coin: str,
    interval: str,
    days: int,
    candles_db_path: str = "data_storage/rl_candles.db"
) -> float:
    """Buy & Hold 수익률 계산"""
    
    try:
        conn = sqlite3.connect(candles_db_path)

        # 최근 N일 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 캔들 수 계산 (인터벌에 따라)
        interval_minutes = {
            '15m': 15,
            '30m': 30,
            '240m': 240,
            '4h': 240,
            '1d': 1440
        }

        minutes = interval_minutes.get(interval, 15)
        expected_candles = int((days * 24 * 60) / minutes)

        # 🔥 수정: 실제 테이블 구조에 맞게 쿼리 변경
        query = """
            SELECT timestamp, close
            FROM candles
            WHERE coin = ? AND interval = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(coin, interval, expected_candles))
        conn.close()
        
        if len(df) < 2:
            logger.warning(f"데이터 부족: {len(df)}개 캔들")
            return 0.0
        
        # 시작 가격과 종료 가격
        start_price = df['close'].iloc[-1]  # 가장 오래된 데이터
        end_price = df['close'].iloc[0]      # 가장 최근 데이터
        
        # 수익률 계산
        return_pct = ((end_price - start_price) / start_price) * 100
        
        logger.debug(f"Buy & Hold: {start_price:.2f} → {end_price:.2f} ({return_pct:.2f}%)")
        
        return return_pct
    
    except Exception as e:
        logger.error(f"❌ Buy & Hold 계산 실패: {e}")
        return 0.0


def calculate_current_strategy_avg(
    coin: str,
    interval: str,
    db_path: str = "data_storage/rl_strategies.db"
) -> Optional[float]:
    """현재 전략 평균 수익률"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT AVG(profit) FROM coin_strategies
            WHERE coin = ? AND interval = ?
            AND profit IS NOT NULL
        """, (coin, interval))
        
        result = cursor.fetchone()[0]
        conn.close()
        
        return float(result) if result is not None else None
    
    except Exception as e:
        logger.error(f"❌ 현재 전략 평균 계산 실패: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 테스트
    coins = ['BTC', 'ETH', 'BNB']
    intervals = ['15m', '30m', '240m', '1d']
    
    for coin in coins:
        for interval in intervals:
            try:
                comparison = compare_with_baseline(coin, interval, test_period_days=30)
            except Exception as e:
                logger.error(f"{coin}-{interval} 비교 실패: {e}")

