#!/usr/bin/env python
"""새로운 DB로 수정 사항 테스트"""
import sys
sys.path.append('/workspace')

import os
os.environ['PREDICTIVE_SELFPLAY_EPISODES'] = '1'

print("=" * 80)
print("새로운 DB로 수정 사항 테스트")
print("=" * 80)
print()

# 1. DB 스키마 초기화
print("1. DB 스키마 초기화...")
try:
    from rl_pipeline.db.rl_init import initialize_database
    initialize_database()
    print("   ✓ DB 스키마 초기화 완료")
    print()
except Exception as e:
    print(f"   ⚠️ 초기화 중 에러 (무시 가능): {e}")
    print()

# 2. 캔들 데이터 로드
print("2. 캔들 데이터 준비...")
try:
    from rl_pipeline.db.rl_reads import load_candles

    # rl_candles.db에서 데이터 로드
    candle_data = load_candles(coin='ADA', interval='15m', limit=500)

    if candle_data is None or len(candle_data) == 0:
        print("   ⚠️ rl_candles.db에서 데이터를 로드할 수 없음")
        print("   → 더미 데이터 생성")

        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # 더미 캔들 데이터 생성
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
        print(f"   ✓ 더미 캔들 데이터 생성: {len(candle_data)}개")
    else:
        print(f"   ✓ 캔들 데이터 로드 완료: {len(candle_data)}개")

    print()
except Exception as e:
    print(f"   ❌ 캔들 데이터 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Orchestrator로 파이프라인 실행
print("3. 통합 파이프라인 실행 (수정된 코드)...")
try:
    from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

    orch = IntegratedPipelineOrchestrator(session_id="fresh_db_test")

    result = orch.run_complete_pipeline(
        coin='ADA',
        interval='15m',
        candle_data=candle_data
    )

    print(f"   ✓ 파이프라인 실행 완료")
    print()

except Exception as e:
    print(f"   ⚠️ 파이프라인 실행 중 에러: {e}")
    import traceback
    traceback.print_exc()
    print()
    # 에러가 발생해도 계속 진행 (DB에 일부 데이터가 저장되었을 수 있음)

# 4. DB에 저장된 데이터 검증
print("4. DB에 저장된 데이터 검증...")
try:
    import sqlite3
    import pandas as pd
    from collections import Counter

    conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db', timeout=10.0)

    # 전체 에피소드 조회
    df = pd.read_sql_query("""
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
    """, conn)

    if len(df) == 0:
        print("   ⚠️ DB에 저장된 에피소드가 없습니다")
        print("   → Self-play가 실행되지 않았거나 에러 발생")
    else:
        print(f"   ✓ 저장된 에피소드: {len(df)}개")
        print()

        # 다양성 검증
        unique_prices = df['entry_price'].nunique()
        unique_timestamps = df['ts_entry'].nunique()

        price_diversity_pct = (unique_prices / len(df) * 100)
        ts_diversity_pct = (unique_timestamps / len(df) * 100)

        print("=" * 80)
        print("📊 캔들 다양성 검증 결과")
        print("=" * 80)
        print()
        print(f"   entry_price 다양성:")
        print(f"   - 총 에피소드: {len(df)}개")
        print(f"   - 고유 가격: {unique_prices}개")
        print(f"   - 다양성 비율: {price_diversity_pct:.1f}%")
        print(f"   - 가격 범위: {df['entry_price'].min():.4f} ~ {df['entry_price'].max():.4f}")
        print()

        print(f"   timestamp 다양성:")
        print(f"   - 고유 타임스탬프: {unique_timestamps}개")
        print(f"   - 다양성 비율: {ts_diversity_pct:.1f}%")
        print()

        # 가격 분포 확인
        price_counts = Counter(df['entry_price'].values)
        print(f"   가격 분포 (상위 10개):")
        for price, count in list(price_counts.most_common(10)):
            print(f"      {price:.4f}: {count}회")

        print()
        print("=" * 80)

        # 최종 판정
        if price_diversity_pct >= 30 and ts_diversity_pct >= 30:
            print("✅ 전체 통과: 캔들 다양성 확보됨!")
            print("   → 각 전략이 다른 캔들을 사용함")
        elif unique_prices > 1:
            print(f"⚠️ 부분 통과: 다양성 {price_diversity_pct:.1f}% (목표: 30% 이상)")
            print("   → 일부 다양성은 있으나 개선 필요")
        else:
            print("❌ 실패: 모든 전략이 같은 캔들 사용")
            print("   → 수정이 제대로 반영되지 않음")

        print("=" * 80)

    conn.close()

except Exception as e:
    print(f"   ❌ DB 검증 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("테스트 완료")
