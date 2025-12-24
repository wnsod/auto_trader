
import sqlite3
import os
import pandas as pd

# Check BTC strategies DB
db_path = r"market/coin_market/data_storage/learning_strategies/btc_strategies.db"
print(f"🔍 Checking {db_path}...")

if not os.path.exists(db_path):
    print("❌ File not found.")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"Tables: {tables}")
        
        # Check strategies count
        if 'strategies' in tables:
            cursor.execute("SELECT count(*) FROM strategies")
            count = cursor.fetchone()[0]
            print(f" strategies count: {count}")
            
            if count > 0:
                cursor.execute("SELECT symbol, interval, count(*) FROM strategies GROUP BY symbol, interval")
                print(" Strategies by coin/interval:")
                for row in cursor.fetchall():
                    print(f"  {row}")
        
        if 'strategies' in tables: # View or table
             # Check if it is a view
            cursor.execute("SELECT type FROM sqlite_master WHERE name='strategies'")
            obj_type = cursor.fetchone()[0]
            print(f" strategies type: {obj_type}")

            cursor.execute("SELECT count(*) FROM strategies")
            count = cursor.fetchone()[0]
            print(f" strategies count: {count}")

        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n🔍 Checking common_strategies.db...")
common_db_path = r"market/coin_market/data_storage/learning_strategies/common_strategies.db"
if os.path.exists(common_db_path):
    try:
        conn = sqlite3.connect(common_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(f"Tables: {[t[0] for t in cursor.fetchall()]}")
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ common_strategies.db not found.")

