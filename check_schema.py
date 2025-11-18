#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# rl_episode_summary 스키마
print("=" * 70)
print("rl_episode_summary 테이블 스키마:")
print("=" * 70)
cursor.execute("PRAGMA table_info(rl_episode_summary)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]:30s} {col[2]:15s} NOT NULL={col[3]} DEFAULT={col[4]}")
print()

# 샘플 데이터
print("샘플 데이터 (최근 3개):")
cursor.execute("SELECT * FROM rl_episode_summary LIMIT 3")
rows = cursor.fetchall()
for row in rows:
    print(f"  {row[:5]}")  # 처음 5개 컬럼만
print()

print("=" * 70)
print("rl_episodes 테이블 스키마:")
print("=" * 70)
cursor.execute("PRAGMA table_info(rl_episodes)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]:30s} {col[2]:15s} NOT NULL={col[3]} DEFAULT={col[4]}")
print()

# 샘플 데이터
print("샘플 데이터 (최근 3개):")
cursor.execute("SELECT * FROM rl_episodes LIMIT 3")
rows = cursor.fetchall()
for row in rows:
    print(f"  {row[:5]}")  # 처음 5개 컬럼만

conn.close()
