#!/usr/bin/env python
"""2️⃣ 통합 파이프라인 실행 검증"""
import sys
sys.path.append('/workspace')

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator
from rl_pipeline.data.candle_loader import load_candle_data_for_coin, get_available_coins_and_intervals

print("=" * 70)
print("2️⃣  통합 파이프라인 실행 검증")
print("=" * 70)
print()

# Step 1: 테스트 코인 및 인터벌 선택
print("Step 1: 테스트 설정")
print("-" * 70)

try:
    available = get_available_coins_and_intervals()
    coins = sorted(list({c for c, _ in available}))

    if not coins:
        print("❌ 사용 가능한 코인이 없습니다")
        sys.exit(1)

    test_coin = coins[0]
    test_intervals = sorted([i for c, i in available if c == test_coin])

    # 가장 작은 인터벌 하나만 테스트 (빠른 검증을 위해)
    test_interval = test_intervals[0] if test_intervals else '15m'

    print(f"✅ 테스트 코인: {test_coin}")
    print(f"✅ 테스트 인터벌: {test_interval}")
    print(f"   (전체 인터벌: {test_intervals})")
    print()

except Exception as e:
    print(f"❌ 설정 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: 캔들 데이터 로드
print("Step 2: 캔들 데이터 로드")
print("-" * 70)

try:
    candle_data = load_candle_data_for_coin(test_coin, [test_interval])

    if not candle_data or (test_coin, test_interval) not in candle_data:
        print(f"❌ {test_coin}-{test_interval} 캔들 데이터 로드 실패")
        sys.exit(1)

    df = candle_data[(test_coin, test_interval)]
    print(f"✅ 캔들 데이터 로드 성공: {len(df)}개")
    print()

except Exception as e:
    print(f"❌ 캔들 데이터 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: 파이프라인 오케스트레이터 초기화
print("Step 3: 파이프라인 오케스트레이터 초기화")
print("-" * 70)

try:
    orchestrator = IntegratedPipelineOrchestrator(session_id=None)
    print("✅ IntegratedPipelineOrchestrator 초기화 성공")
    print()

except Exception as e:
    print(f"❌ 오케스트레이터 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 파이프라인 실행 (축소 버전)
print("Step 4: 통합 파이프라인 실행 (축소 버전)")
print("-" * 70)
print(f"실행 중: {test_coin}-{test_interval}...")
print("⏳ 전략 생성 → Self-Play → 롤업 평가 진행 중...")
print()

try:
    result = orchestrator.run_partial_pipeline(
        test_coin,
        test_interval,
        df
    )

    print("✅ 파이프라인 실행 완료!")
    print()

    # 결과 출력
    print("Step 5: 실행 결과 확인")
    print("-" * 70)

    print(f"   코인: {result.coin}")
    print(f"   인터벌: {result.interval}")
    print(f"   상태: {result.status}")
    print(f"   생성된 전략 수: {result.strategies_created}")
    print(f"   Self-Play 에피소드: {result.selfplay_episodes}")
    print(f"   레짐 감지: {result.regime_detected}")
    print(f"   라우팅 결과: {result.routing_results}")
    print(f"   시그널 액션: {result.signal_action}")
    print(f"   시그널 점수: {result.signal_score:.3f}")
    print(f"   실행 시간: {result.execution_time:.2f}초")
    print()

    # 검증
    if result.status == "success":
        print("✅ 파이프라인 상태: SUCCESS")
    else:
        print(f"⚠️  파이프라인 상태: {result.status}")

    if result.strategies_created > 0:
        print(f"✅ 전략 생성: {result.strategies_created}개")
    else:
        print("⚠️  전략 생성: 0개 (문제 가능)")

    if result.selfplay_episodes > 0:
        print(f"✅ Self-Play 실행: {result.selfplay_episodes}개 에피소드")
    else:
        print("⚠️  Self-Play 미실행")

    print()

except Exception as e:
    print(f"❌ 파이프라인 실행 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("✅ 통합 파이프라인 검증 완료!")
print("=" * 70)
