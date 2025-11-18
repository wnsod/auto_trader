#!/usr/bin/env python
"""캔들 다양성 테스트 - 각 전략이 다른 캔들 사용하는지 검증"""
import sys
sys.path.append('/workspace')

import os
import sqlite3
import pandas as pd
from collections import Counter

# 환경 변수 설정 (1 에피소드만)
os.environ['PREDICTIVE_SELFPLAY_EPISODES'] = '1'
os.environ['PREDICTIVE_SELFPLAY_MIN_EPISODES'] = '1'

print("=" * 80)
print("캔들 다양성 테스트")
print("=" * 80)
print()

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

# Orchestrator 초기화
orch = IntegratedPipelineOrchestrator(session_id="test_candle_diversity")

# ADA-15m 1 에피소드만 실행
print("📊 ADA-15m 예측 Self-play 시작 (1 에피소드)")
print()

try:
    results = orch.run_predictive_selfplay_for_coin(
        coin="ADA",
        intervals=["15m"]
    )

    print()
    print("=" * 80)
    print("✅ Self-play 완료")
    print("=" * 80)
    print()

    if results and results.get('success'):
        print(f"  - 평균 정확도: {results.get('avg_accuracy', 0)*100:.1f}%")
        print(f"  - 에피소드 수: {results.get('total_episodes', 0)}")

    print()
    print("=" * 80)
    print("📊 DB 검증 시작")
    print("=" * 80)
    print()

    # DB 연결
    db_path = '/workspace/data_storage/rl_strategies.db'
    conn = sqlite3.connect(db_path)

    # 1. 최근 예측 데이터 조회
    query = """
    SELECT
        episode_id,
        strategy_id,
        entry_price,
        predicted_conf,
        horizon_k,
        ts_entry
    FROM rl_episodes
    WHERE coin = 'ADA' AND interval = '15m'
    ORDER BY ts_entry DESC
    LIMIT 200
    """

    df = pd.read_sql_query(query, conn)

    print(f"1. 조회된 예측 수: {len(df)}개")
    print()

    # 2. entry_price 다양성 검증
    unique_prices = df['entry_price'].nunique()
    price_counts = Counter(df['entry_price'].values)

    print(f"2. entry_price 다양성:")
    print(f"   - 고유 가격 수: {unique_prices}개 (전체 {len(df)}개 중)")
    print(f"   - 다양성 비율: {unique_prices / len(df) * 100:.1f}%")
    print()

    if unique_prices <= 2:
        print("   ❌ 실패: entry_price가 너무 적음 (대부분 동일)")
        print(f"   - 가격 분포: {dict(list(price_counts.most_common(5)))}")
    elif unique_prices < len(df) * 0.3:
        print("   ⚠️ 경고: entry_price 다양성 부족")
        print(f"   - 가격 분포 (상위 10개): {dict(list(price_counts.most_common(10)))}")
    else:
        print("   ✅ 통과: entry_price가 다양함")
        print(f"   - 가격 범위: {df['entry_price'].min():.4f} ~ {df['entry_price'].max():.4f}")
        print(f"   - 상위 5개 가격: {dict(list(price_counts.most_common(5)))}")

    print()

    # 3. timestamp 다양성 검증
    unique_timestamps = df['ts_entry'].nunique()
    ts_counts = Counter(df['ts_entry'].values)

    print(f"3. timestamp 다양성:")
    print(f"   - 고유 타임스탬프 수: {unique_timestamps}개 (전체 {len(df)}개 중)")
    print(f"   - 다양성 비율: {unique_timestamps / len(df) * 100:.1f}%")
    print()

    if unique_timestamps <= 2:
        print("   ❌ 실패: timestamp가 너무 적음 (대부분 동일)")
        print(f"   - 타임스탬프 분포: {dict(list(ts_counts.most_common(5)))}")
    elif unique_timestamps < len(df) * 0.3:
        print("   ⚠️ 경고: timestamp 다양성 부족")
        print(f"   - 타임스탬프 분포 (상위 10개): {dict(list(ts_counts.most_common(10)))}")
    else:
        print("   ✅ 통과: timestamp가 다양함")
        # 시간 간격 계산
        timestamps_sorted = sorted(df['ts_entry'].unique())
        if len(timestamps_sorted) > 1:
            time_diffs = [timestamps_sorted[i+1] - timestamps_sorted[i] for i in range(len(timestamps_sorted)-1)]
            avg_diff = sum(time_diffs) / len(time_diffs)
            print(f"   - 평균 시간 간격: {avg_diff / 60:.1f}분")
            print(f"   - 최소/최대 간격: {min(time_diffs) / 60:.1f}분 / {max(time_diffs) / 60:.1f}분")

    print()

    # 4. 전략별 분산도 확인
    strategy_price_diversity = df.groupby('strategy_id')['entry_price'].nunique()

    print(f"4. 전략별 가격 다양성:")
    print(f"   - 평균 고유 가격 수/전략: {strategy_price_diversity.mean():.2f}")

    if strategy_price_diversity.mean() > 1.0:
        print("   ⚠️ 주의: 같은 전략이 여러 가격 사용 (에피소드마다 다른 캔들?)")
    else:
        print("   ✅ 정상: 각 전략이 1개 가격 사용 (에피소드 내에서 일관성 유지)")

    print()

    # 5. 최종 결과 요약
    query_summary = """
    SELECT
        MIN(entry_price) as min_price,
        MAX(entry_price) as max_price,
        AVG(entry_price) as avg_price,
        MIN(ts_entry) as min_ts,
        MAX(ts_entry) as max_ts
    FROM rl_episodes
    WHERE coin = 'ADA' AND interval = '15m'
    """

    summary = pd.read_sql_query(query_summary, conn)

    print("=" * 80)
    print("📊 최종 검증 결과")
    print("=" * 80)
    print()
    print(f"가격 범위: {summary['min_price'].iloc[0]:.4f} ~ {summary['max_price'].iloc[0]:.4f}")
    print(f"타임스탬프 범위: {pd.to_datetime(summary['min_ts'].iloc[0], unit='s')} ~ {pd.to_datetime(summary['max_ts'].iloc[0], unit='s')}")
    print()

    # 전체 평가
    if unique_prices >= len(df) * 0.3 and unique_timestamps >= len(df) * 0.3:
        print("✅ 전체 통과: 캔들 다양성 확보됨")
    elif unique_prices <= 2 or unique_timestamps <= 2:
        print("❌ 전체 실패: 캔들 다양성 없음 (대부분 동일한 캔들 사용)")
    else:
        print("⚠️ 부분 통과: 일부 다양성 있으나 개선 필요")

    print("=" * 80)

    conn.close()

except Exception as e:
    print(f"❌ 테스트 실행 중 오류: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
