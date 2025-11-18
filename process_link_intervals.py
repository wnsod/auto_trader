#!/usr/bin/env python
"""
LINK 코인의 나머지 인터벌 처리 (30m, 240m, 1d)
그 다음 통합 분석까지
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.data.candle_loader import load_candle_data_for_coin
from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

COIN = 'LINK'
INTERVALS = ['30m', '240m', '1d']  # 15m은 이미 완료

print("=" * 70)
print(f"{COIN} 나머지 인터벌 처리")
print("=" * 70)
print(f"인터벌: {', '.join(INTERVALS)}")
print()

# 캔들 데이터 로드
print(f"📥 {COIN} 캔들 데이터 로드 중...")
all_candle_data = load_candle_data_for_coin(COIN, INTERVALS)

print(f"✅ 캔들 데이터 로드 완료:")
for (coin, interval), df in all_candle_data.items():
    print(f"   {interval}: {len(df)}개")
print()

# Orchestrator 초기화
orchestrator = IntegratedPipelineOrchestrator()

# 각 인터벌 처리
pipeline_results = []
for interval in INTERVALS:
    print("=" * 70)
    print(f"{COIN}-{interval} 처리 시작")
    print("=" * 70)

    candle_data = all_candle_data.get((COIN, interval))
    if candle_data is None or candle_data.empty:
        print(f"⚠️ {interval} 캔들 데이터 없음, 건너뜀")
        continue

    try:
        # 전략 생성 → 예측 self-play → 롤업 → 등급
        print(f"\n1. {interval} 파이프라인 실행 중...")
        result = orchestrator.run_partial_pipeline(COIN, interval, candle_data)
        pipeline_results.append(result)

        print(f"\n✅ {COIN}-{interval} 처리 완료")
        print(f"   상태: {result.status}")

    except Exception as e:
        print(f"\n❌ {COIN}-{interval} 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        continue

    print()

# 15m 결과도 포함해서 통합 분석
print("=" * 70)
print("전체 인터벌 통합 분석")
print("=" * 70)
print()

# 15m 캔들 데이터도 로드
all_intervals = ['15m', '30m', '240m', '1d']
all_candle_data_full = load_candle_data_for_coin(COIN, all_intervals)

print(f"📥 전체 인터벌 캔들 데이터:")
for (coin, interval), df in all_candle_data_full.items():
    print(f"   {interval}: {len(df)}개")
print()

try:
    print("🔍 통합 분석 실행 중...")
    # run_integrated_analysis_all_intervals는 PipelineResult 리스트를 받음
    # 하지만 15m 결과가 없으므로, 일단 현재 결과로만 진행
    final_result = orchestrator.run_integrated_analysis_all_intervals(
        COIN,
        pipeline_results,
        all_candle_data_full
    )

    print(f"\n✅ 통합 분석 완료")
    print(f"   시그널 액션: {final_result.signal_action}")
    print(f"   시그널 점수: {final_result.signal_score:.3f}")

except Exception as e:
    print(f"\n❌ 통합 분석 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("처리 완료")
print("=" * 70)
