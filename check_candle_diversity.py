#!/usr/bin/env python
"""캔들 다양성 간단 검증 - DB만 확인"""
import sys
sys.path.append('/workspace')

import sqlite3
import pandas as pd
from collections import Counter

print("=" * 80)
print("캔들 다양성 DB 검증")
print("=" * 80)
print()

# DB 연결
db_path = '/workspace/data_storage/rl_strategies.db'
conn = sqlite3.connect(db_path)

# 1. 코인별 최근 데이터 조회
query = """
SELECT
    coin,
    interval,
    COUNT(*) as total_episodes,
    COUNT(DISTINCT entry_price) as unique_prices,
    COUNT(DISTINCT ts_entry) as unique_timestamps,
    MIN(entry_price) as min_price,
    MAX(entry_price) as max_price
FROM rl_episodes
WHERE coin IN ('ADA', 'SOL', 'BNB', 'XRP')
GROUP BY coin, interval
ORDER BY coin, interval
"""

summary_df = pd.read_sql_query(query, conn)

print("📊 코인별 다양성 요약:")
print()
print(summary_df.to_string(index=False))
print()
print("=" * 80)

# 2. 각 코인별 상세 검증
for _, row in summary_df.iterrows():
    coin = row['coin']
    interval = row['interval']
    total = row['total_episodes']
    unique_prices = row['unique_prices']
    unique_ts = row['unique_timestamps']

    if total == 0:
        continue

    price_diversity = (unique_prices / total * 100) if total > 0 else 0
    ts_diversity = (unique_ts / total * 100) if total > 0 else 0

    print(f"\n📌 {coin}-{interval}:")
    print(f"   - 총 에피소드: {total}개")
    print(f"   - 고유 가격: {unique_prices}개 ({price_diversity:.1f}%)")
    print(f"   - 고유 타임스탬프: {unique_ts}개 ({ts_diversity:.1f}%)")

    # 판정
    if unique_prices <= 2 or unique_ts <= 2:
        print(f"   ❌ 실패: 다양성 없음 (같은 캔들 반복 사용)")
    elif price_diversity < 30 or ts_diversity < 30:
        print(f"   ⚠️ 경고: 다양성 부족 ({price_diversity:.1f}% / {ts_diversity:.1f}%)")
    else:
        print(f"   ✅ 통과: 캔들 다양성 확보됨")

    # 최근 20개 에피소드의 entry_price 분포 확인
    detail_query = f"""
    SELECT entry_price, COUNT(*) as cnt
    FROM rl_episodes
    WHERE coin = '{coin}' AND interval = '{interval}'
    ORDER BY ts_entry DESC
    LIMIT 100
    """

    detail_df = pd.read_sql_query(detail_query, conn)
    if len(detail_df) > 0:
        price_counts = detail_df['entry_price'].value_counts()
        print(f"   - 최근 100개 에피소드 가격 분포 (상위 5개):")
        for price, count in list(price_counts.head(5).items()):
            print(f"      {price:.4f}: {count}회")

print()
print("=" * 80)

# 3. 전체 평가
total_episodes = summary_df['total_episodes'].sum()
if total_episodes == 0:
    print("⚠️ 에피소드 데이터가 없습니다.")
else:
    avg_price_diversity = (summary_df['unique_prices'] / summary_df['total_episodes'] * 100).mean()
    avg_ts_diversity = (summary_df['unique_timestamps'] / summary_df['total_episodes'] * 100).mean()

    print(f"전체 평균 다양성:")
    print(f"  - 가격 다양성: {avg_price_diversity:.1f}%")
    print(f"  - 타임스탬프 다양성: {avg_ts_diversity:.1f}%")
    print()

    if avg_price_diversity >= 30 and avg_ts_diversity >= 30:
        print("✅ 전체 통과: 캔들 다양성 확보됨")
    elif avg_price_diversity <= 5 or avg_ts_diversity <= 5:
        print("❌ 전체 실패: 캔들 다양성 없음 (대부분 동일한 캔들 사용)")
    else:
        print("⚠️ 부분 통과: 일부 다양성 있으나 개선 필요")

print("=" * 80)

conn.close()
