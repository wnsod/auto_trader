
import sqlite3
import os
import sys
import pandas as pd
from datetime import datetime

# 경로 설정
BASE_DIR = r"c:\auto_trader\market\coin_market\data_storage"
CANDLES_DB = os.path.join(BASE_DIR, "learning_candles.db")
STRATEGIES_DB_OLD = os.path.join(BASE_DIR, "learning_strategies.db")
STRATEGIES_DB_NEW = os.path.join(BASE_DIR, "learning_strategies.db")

def analyze_candles_db(db_path):
    print(f"\n🔍 Analyzing {db_path}...")
    if not os.path.exists(db_path):
        print("❌ File not found.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
        
        if ('candles',) in tables:
            # 레코드 수
            cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM candles")
            count, min_ts, max_ts = cursor.fetchone()
            print(f"Total Candles: {count}")
            print(f"Time Range: {min_ts} ~ {max_ts}")
            
            if count > 0:
                # 데이터 샘플 및 이상치 확인
                df = pd.read_sql_query("SELECT * FROM candles ORDER BY timestamp DESC LIMIT 100", conn)
                print("\nLatest 5 Candles:")
                print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head().to_string())
                
                # 0이나 음수 값 확인
                cursor.execute("SELECT COUNT(*) FROM candles WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0")
                invalid_ohlc = cursor.fetchone()[0]
                print(f"\nInvalid OHLC (<= 0): {invalid_ohlc}")
                
                cursor.execute("SELECT COUNT(*) FROM candles WHERE volume < 0")
                invalid_vol = cursor.fetchone()[0]
                print(f"Invalid Volume (< 0): {invalid_vol}")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_strategies_db(db_path):
    print(f"\n🔍 Analyzing {db_path}...")
    if not os.path.exists(db_path):
        print("❌ File not found.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"Tables: {tables}")
        
        target_table = None
        if 'strategies' in tables:
            target_table = 'strategies'
        elif 'strategies' in tables:
            target_table = 'strategies'
            
        if target_table:
            print(f"\nTarget Table: {target_table}")
            
            # 전체 전략 수
            cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
            count = cursor.fetchone()[0]
            print(f"Total Strategies: {count}")
            
            if count > 0:
                # 성과 통계
                cursor.execute(f"SELECT AVG(win_rate), AVG(profit_factor), AVG(trades_count) FROM {target_table}")
                avg_wr, avg_pf, avg_trades = cursor.fetchone()
                print(f"Avg Win Rate: {avg_wr}")
                print(f"Avg Profit Factor: {avg_pf}")
                print(f"Avg Trades Count: {avg_trades}")
                
                # 거래 횟수 0인 전략 수
                cursor.execute(f"SELECT COUNT(*) FROM {target_table} WHERE trades_count = 0 OR trades_count IS NULL")
                zero_trades = cursor.fetchone()[0]
                print(f"Strategies with 0 trades: {zero_trades} ({(zero_trades/count)*100:.1f}%)")
                
                # 파라미터 샘플 확인 (params 컬럼이 있다고 가정)
                # 컬럼 목록 확인
                cursor.execute(f"PRAGMA table_info({target_table})")
                columns = [c[1] for c in cursor.fetchall()]
                
                print(f"\nColumns: {columns}")
                
                # 샘플 데이터 조회
                df = pd.read_sql_query(f"SELECT * FROM {target_table} LIMIT 5", conn)
                print("\nSample Strategies:")
                print(df.to_string())

        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=== DB Analysis Start ===")
    analyze_candles_db(CANDLES_DB)
    analyze_strategies_db(STRATEGIES_DB_NEW) # learning_strategies.db (현재 사용 중인 것으로 추정)
    analyze_strategies_db(STRATEGIES_DB_OLD) # learning_strategies.db (이전 버전 또는 백업)
    print("\n=== DB Analysis End ===")

