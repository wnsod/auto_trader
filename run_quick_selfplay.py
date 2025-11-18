#!/usr/bin/env python
"""빠른 Self-play 테스트 - DB에 실제 데이터 생성"""
import sys
sys.path.append('/workspace')

import os
os.environ['PREDICTIVE_SELFPLAY_EPISODES'] = '1'  # 1 에피소드만
os.environ['PREDICTIVE_SELFPLAY_MIN_EPISODES'] = '1'

print("=" * 80)
print("빠른 Self-play 테스트 (수정된 코드 검증)")
print("=" * 80)
print()

# 1. DB 접근 가능 확인
print("1. DB 접근 확인...")
try:
    import sqlite3
    import time

    db_path = '/workspace/data_storage/rl_strategies.db'

    # 짧은 timeout으로 시도
    conn = sqlite3.connect(db_path, timeout=5.0)
    cursor = conn.cursor()

    # 간단한 쿼리 테스트
    cursor.execute("SELECT COUNT(*) FROM rl_episodes WHERE coin='ADA' AND interval='15m'")
    count_before = cursor.fetchone()[0]
    conn.close()

    print(f"   ✓ DB 접근 성공")
    print(f"   ✓ ADA-15m 기존 에피소드: {count_before}개")
    print()

except Exception as e:
    print(f"   ❌ DB 접근 실패: {e}")
    print(f"   → 다른 프로세스가 DB를 사용 중일 수 있음")
    print()
    sys.exit(1)

# 2. Orchestrator로 Self-play 실행
print("2. Self-play 실행 (ADA-15m, 1 에피소드)...")
print()

try:
    from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

    orch = IntegratedPipelineOrchestrator(session_id="test_fix_verification")

    # run_complete_pipeline 사용
    print("   - 캔들 데이터 로드 중...")

    # 캔들 데이터 로드
    from rl_pipeline.db.rl_reads import load_candles
    candle_data = load_candles(coin='ADA', interval='15m', limit=500)

    if candle_data is None or len(candle_data) == 0:
        print("   ❌ 캔들 데이터 없음")
        sys.exit(1)

    print(f"   ✓ 캔들 데이터: {len(candle_data)}개")
    print()

    print("   - 파이프라인 실행 중...")
    result = orch.run_complete_pipeline(
        coin='ADA',
        interval='15m',
        candle_data=candle_data
    )

    print()
    print(f"   ✓ 파이프라인 완료")
    print(f"   - 결과: {result}")
    print()

except Exception as e:
    print(f"   ❌ Self-play 실행 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. DB에 새 데이터가 저장되었는지 확인
print("3. DB 변화 확인...")
try:
    conn = sqlite3.connect(db_path, timeout=5.0)
    cursor = conn.cursor()

    # 최근 에피소드 수 확인
    cursor.execute("SELECT COUNT(*) FROM rl_episodes WHERE coin='ADA' AND interval='15m'")
    count_after = cursor.fetchone()[0]

    new_episodes = count_after - count_before

    print(f"   - 이전 에피소드: {count_before}개")
    print(f"   - 현재 에피소드: {count_after}개")
    print(f"   - 새로 생성: {new_episodes}개")
    print()

    if new_episodes > 0:
        # 최근 생성된 에피소드의 entry_price 다양성 확인
        cursor.execute(f"""
            SELECT
                COUNT(DISTINCT entry_price) as unique_prices,
                COUNT(*) as total,
                MIN(entry_price) as min_price,
                MAX(entry_price) as max_price
            FROM (
                SELECT entry_price
                FROM rl_episodes
                WHERE coin='ADA' AND interval='15m'
                ORDER BY ts_entry DESC
                LIMIT {new_episodes}
            )
        """)

        diversity = cursor.fetchone()
        unique_prices, total, min_price, max_price = diversity

        diversity_pct = (unique_prices / total * 100) if total > 0 else 0

        print("=" * 80)
        print("📊 새로 생성된 데이터 다양성 검증")
        print("=" * 80)
        print()
        print(f"   - 총 에피소드: {total}개")
        print(f"   - 고유 가격: {unique_prices}개")
        print(f"   - 다양성 비율: {diversity_pct:.1f}%")
        print(f"   - 가격 범위: {min_price:.4f} ~ {max_price:.4f}")
        print()

        if diversity_pct >= 30:
            print("   ✅ 통과: 캔들 다양성 확보됨!")
            print("   → 수정된 코드가 정상 작동함")
        elif unique_prices > 1:
            print(f"   ⚠️ 부분 통과: 다양성 {diversity_pct:.1f}% (목표: 30% 이상)")
        else:
            print("   ❌ 실패: 모두 같은 가격 사용")

        print("=" * 80)
    else:
        print("   ⚠️ 새로 생성된 에피소드 없음")

    conn.close()

except Exception as e:
    print(f"   ❌ DB 확인 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✅ 테스트 완료")
print("=" * 80)
