"""Test params field structure"""
import sqlite3
import json

db_path = '/workspace/data_storage/rl_strategies.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check coin_strategies
print("=== coin_strategies ===")
cursor.execute('SELECT id, rsi_min, rsi_max, params FROM coin_strategies LIMIT 1')
row = cursor.fetchone()
if row:
    print(f'ID: {row[0]}')
    print(f'rsi_min (column): {row[1]}')
    print(f'rsi_max (column): {row[2]}')
    if row[3]:
        params = json.loads(row[3])
        print(f'params.rsi_min: {params.get("rsi_min")}')
        print(f'params.rsi_max: {params.get("rsi_max")}')
    else:
        print('params: NULL')
else:
    print('No data')

print("\n=== global_strategies ===")
cursor.execute('SELECT id, regime, params FROM global_strategies LIMIT 1')
row = cursor.fetchone()
if row:
    print(f'ID: {row[0]}')
    print(f'regime: {row[1]}')
    if row[2]:
        params = json.loads(row[2])
        print(f'params.rsi_min: {params.get("rsi_min")}')
        print(f'params.rsi_max: {params.get("rsi_max")}')
        print(f'params keys: {list(params.keys())[:10]}')
    else:
        print('params: NULL')
else:
    print('No data yet')

conn.close()
