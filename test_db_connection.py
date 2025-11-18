#!/usr/bin/env python
"""DB 연결 테스트"""
import sys
sys.path.append('/workspace')

from rl_pipeline.db.candle_data import CandleDataManager
from rl_pipeline.core.env import config

print("=" * 80)
print("Database Connection Test")
print("=" * 80)
print()

print("Environment Configuration:")
print(f"  CANDLES_DB_PATH: {config.RL_DB}")
print()

mgr = CandleDataManager()
print(f"CandleDataManager:")
print(f"  DB Path: {mgr.db_path}")
print()

try:
    candles = mgr.get_candles('BTC', '15m')
    print("✓ SUCCESS: Database connection successful!")
    print(f"  BTC-15m candles loaded: {len(candles)}")
except Exception as e:
    print("✗ FAILED: Database connection failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
