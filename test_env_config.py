#!/usr/bin/env python
"""환경 변수 설정 테스트"""
import sys
sys.path.append('/workspace')

from rl_pipeline.core.env import config

print("=" * 80)
print("Environment Variable Configuration Test")
print("=" * 80)
print()

print("Database Paths:")
print(f"  CANDLES_DB_PATH:          {config.RL_DB}")
print(f"  STRATEGIES_DB_PATH:       {config.STRATEGIES_DB}")
print(f"  ML_CANDLES_DB_PATH:       {config.ML_CANDLES_DB}")
print(f"  LEARNING_RESULTS_DB_PATH: {config.LEARNING_RESULTS_DB}")
print()

# DB 파일 존재 확인
import os

print("Database File Existence:")
print(f"  CANDLES_DB:          {'EXISTS' if os.path.exists(config.RL_DB) else 'NOT FOUND'}")
print(f"  STRATEGIES_DB:       {'EXISTS' if os.path.exists(config.STRATEGIES_DB) else 'NOT FOUND'}")
print(f"  ML_CANDLES_DB:       {'EXISTS' if os.path.exists(config.ML_CANDLES_DB) else 'NOT FOUND'}")
print(f"  LEARNING_RESULTS_DB: {'EXISTS' if os.path.exists(config.LEARNING_RESULTS_DB) else 'NOT FOUND'}")
print()

# 간단한 DB 연결 테스트
import sqlite3

try:
    conn = sqlite3.connect(config.RL_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM candles LIMIT 1")
    count = cursor.fetchone()[0]
    conn.close()

    print(f"✓ SUCCESS: Connected to {config.RL_DB}")
    print(f"  Candles count: {count}")
except Exception as e:
    print(f"✗ FAILED: Could not connect to {config.RL_DB}")
    print(f"  Error: {e}")

print()
print("=" * 80)
