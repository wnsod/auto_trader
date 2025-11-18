#!/usr/bin/env python
"""에피소드 생성 시간 확인"""
import sys
sys.path.append('/workspace')

import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')

df = pd.read_sql_query("""
    SELECT
        coin,
        interval,
        MAX(ts_entry) as latest_ts,
        MIN(ts_entry) as earliest_ts,
        COUNT(*) as cnt
    FROM rl_episodes
    GROUP BY coin, interval
    ORDER BY coin, interval
""", conn)

df['latest_time'] = pd.to_datetime(df['latest_ts'], unit='s')
df['earliest_time'] = pd.to_datetime(df['earliest_ts'], unit='s')
df['hours_ago'] = (datetime.now() - df['latest_time']).dt.total_seconds() / 3600

print("=" * 100)
print("📊 에피소드 생성 시간 분석")
print("=" * 100)
print()
print(df[['coin', 'interval', 'cnt', 'latest_time', 'hours_ago']].to_string(index=False))
print()
print("=" * 100)

# 최근 1시간 이내 데이터가 있는지 확인
recent_data = df[df['hours_ago'] < 1.0]
if len(recent_data) > 0:
    print(f"✅ 최근 1시간 이내 생성된 에피소드: {len(recent_data)}개 코인-인터벌")
else:
    print(f"⚠️ 최근 1시간 이내 생성된 에피소드 없음")
    print(f"   - 가장 최근 에피소드: {df['hours_ago'].min():.1f}시간 전")
    print(f"   → 수정된 코드가 아직 실행되지 않았음!")

conn.close()
