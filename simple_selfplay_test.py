#!/usr/bin/env python
"""간단한 Self-play 테스트 - 더미 데이터로 DB 저장 확인"""
import sys
sys.path.append('/workspace')

import os
os.environ['PREDICTIVE_SELFPLAY_EPISODES'] = '1'

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

print("=" * 80)
print("간단한 Self-play 테스트 (더미 데이터)")
print("=" * 80)
print()

# 1. 더미 캔들 데이터 생성
print("1. 더미 캔들 데이터 생성...")
num_candles = 500
timestamps = []
closes = []
base_time = datetime.now() - timedelta(hours=num_candles)
base_price = 1000.0

for i in range(num_candles):
    timestamps.append(int((base_time + timedelta(hours=i)).timestamp()))
    price_change = np.random.uniform(-0.02, 0.02)
    base_price = base_price * (1 + price_change)
    closes.append(base_price)

candle_data = pd.DataFrame({
    'timestamp': timestamps,
    'close': closes,
    'high': [c * 1.01 for c in closes],
    'low': [c * 0.99 for c in closes],
    'open': closes,
    'volume': [np.random.uniform(1000, 10000) for _ in range(num_candles)],
    'rsi': [np.random.uniform(30, 70) for _ in range(num_candles)],
    'macd': [np.random.uniform(-5, 5) for _ in range(num_candles)],
    'macd_signal': [np.random.uniform(-5, 5) for _ in range(num_candles)],
    'volume_ratio': [np.random.uniform(0.8, 1.5) for _ in range(num_candles)]
})

print(f"   ✓ 캔들 데이터: {len(candle_data)}개")
print()

# 2. DB 기존 상태 확인
db_path = '/workspace/data_storage/rl_strategies.db'
print("2. DB 상태 확인...")

try:
    conn = sqlite3.connect(db_path, timeout=10.0)
    cursor = conn.cursor()

    # 테이블 존재 확인
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_episodes'")
    table_exists = cursor.fetchone()

    if table_exists:
        cursor.execute("SELECT COUNT(*) FROM rl_episodes")
        count_before = cursor.fetchone()[0]
        print(f"   ✓ rl_episodes 테이블 존재")
        print(f"   - 기존 에피소드: {count_before}개")
    else:
        count_before = 0
        print(f"   ⚠️ rl_episodes 테이블 없음 (자동 생성될 예정)")

    conn.close()
except Exception as e:
    print(f"   ⚠️ DB 확인 실패: {e}")
    count_before = 0

print()

# 3. Orchestrator 실행
print("3. Orchestrator 파이프라인 실행...")
try:
    from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

    orch = IntegratedPipelineOrchestrator(session_id="simple_test")

    result = orch.run_complete_pipeline(
        coin='TEST',
        interval='15m',
        candle_data=candle_data
    )

    print(f"   ✓ 파이프라인 완료")
    print()

except Exception as e:
    print(f"   ⚠️ 파이프라인 실행 중 에러: {e}")
    # 일부 에러는 무시하고 계속 진행
    print()

# 4. DB에 저장된 데이터 확인
print("4. DB에 저장된 데이터 확인...")
import time
time.sleep(1)  # DB 쓰기 완료 대기

try:
    conn = sqlite3.connect(db_path, timeout=10.0)
    cursor = conn.cursor()

    # 테이블 확인
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_episodes'")
    if not cursor.fetchone():
        print("   ❌ rl_episodes 테이블이 생성되지 않음")
        conn.close()
        sys.exit(1)

    # TEST 코인 데이터 조회
    cursor.execute("SELECT COUNT(*) FROM rl_episodes WHERE coin='TEST'")
    test_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM rl_episodes")
    total_count = cursor.fetchone()[0]

    new_episodes = total_count - count_before

    print(f"   - 기존 에피소드: {count_before}개")
    print(f"   - 현재 에피소드: {total_count}개")
    print(f"   - 새로 생성: {new_episodes}개")
    print(f"   - TEST 코인: {test_count}개")
    print()

    if test_count > 0:
        # TEST 코인 데이터의 다양성 확인
        df = pd.read_sql_query("""
            SELECT
                entry_price,
                ts_entry,
                strategy_id
            FROM rl_episodes
            WHERE coin='TEST' AND interval='15m'
            ORDER BY ts_entry DESC
        """, conn)

        unique_prices = df['entry_price'].nunique()
        unique_timestamps = df['ts_entry'].nunique()
        total = len(df)

        price_diversity_pct = (unique_prices / total * 100) if total > 0 else 0
        ts_diversity_pct = (unique_timestamps / total * 100) if total > 0 else 0

        print("=" * 80)
        print("📊 캔들 다양성 검증 결과 (실제 DB 데이터)")
        print("=" * 80)
        print()
        print(f"   entry_price 다양성:")
        print(f"   - 총 에피소드: {total}개")
        print(f"   - 고유 가격: {unique_prices}개")
        print(f"   - 다양성 비율: {price_diversity_pct:.1f}%")

        if total > 0:
            print(f"   - 가격 범위: {df['entry_price'].min():.4f} ~ {df['entry_price'].max():.4f}")

        print()

        print(f"   timestamp 다양성:")
        print(f"   - 고유 타임스탬프: {unique_timestamps}개")
        print(f"   - 다양성 비율: {ts_diversity_pct:.1f}%")
        print()

        # 가격 분포
        from collections import Counter
        price_counts = Counter(df['entry_price'].values)
        print(f"   가격 분포 (상위 10개):")
        for price, count in list(price_counts.most_common(10)):
            print(f"      {price:.4f}: {count}회")

        print()
        print("=" * 80)

        # 최종 판정
        if price_diversity_pct >= 30 and ts_diversity_pct >= 30:
            print("✅ 전체 통과: 캔들 다양성 확보됨!")
            print("   → 수정된 코드가 정상 작동함 (각 전략이 다른 캔들 사용)")
        elif unique_prices > 1:
            print(f"⚠️ 부분 통과: 다양성 {price_diversity_pct:.1f}% (목표: 30% 이상)")
        else:
            print("❌ 실패: 모두 같은 가격 사용")
            print("   → 수정이 제대로 반영되지 않음")

        print("=" * 80)
    else:
        print("   ⚠️ TEST 코인 데이터가 생성되지 않음")

    conn.close()

except Exception as e:
    print(f"   ❌ DB 확인 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("테스트 완료")
