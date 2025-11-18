#!/usr/bin/env python
"""
Orchestrator v1 통합 테스트

LINK 코인의 기존 처리 결과를 사용하여 통합 분석 v1 실행
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator, PipelineResult
from rl_pipeline.data.candle_loader import load_candle_data_for_coin

COIN = 'LINK'
INTERVALS = ['15m', '30m', '240m', '1d']

print("=" * 70)
print("Orchestrator v1 통합 테스트")
print("=" * 70)
print(f"코인: {COIN}")
print(f"인터벌: {', '.join(INTERVALS)}")
print()

# Orchestrator 초기화
orchestrator = IntegratedPipelineOrchestrator()

# 캔들 데이터 로드
print("📥 캔들 데이터 로드 중...")
all_candle_data = load_candle_data_for_coin(COIN, INTERVALS)

print(f"✅ 캔들 데이터 로드 완료:")
for (coin, interval), df in all_candle_data.items():
    print(f"   {interval}: {len(df)}개")
print()

# 더미 PipelineResult 생성 (실제로는 각 인터벌 처리 후 생성됨)
# 여기서는 이미 처리된 LINK 데이터가 DB에 있으므로, 더미로 생성
pipeline_results = []
for interval in INTERVALS:
    result = PipelineResult(
        coin=COIN,
        interval=interval,
        status='completed',
        strategies_created=100,  # 더미 값
        regime_detected='neutral',
        signal_action='HOLD',
        signal_score=0.5
    )
    pipeline_results.append(result)

print(f"📊 더미 PipelineResult 생성: {len(pipeline_results)}개")
print()

# 통합 분석 실행
print("=" * 70)
print("통합 분석 v1 실행")
print("=" * 70)
print()

try:
    # run_integrated_analysis_all_intervals 호출
    final_result = orchestrator.run_integrated_analysis_all_intervals(
        coin=COIN,
        pipeline_results=pipeline_results,
        all_candle_data=all_candle_data
    )

    print("✅ 통합 분석 완료")
    print()
    print("=" * 70)
    print("결과")
    print("=" * 70)
    print()

    # v0 호환 필드
    print("📊 v0 호환 출력:")
    print(f"  signal_action:       {final_result.signal_action}")
    print(f"  signal_score:        {final_result.signal_score:.3f}")

    # v1 추가 필드 (있으면)
    if hasattr(final_result, 'direction'):
        print()
        print("📊 v1 상세 출력:")
        print(f"  방향:               {final_result.direction}")
        print(f"  타이밍:             {final_result.timing}")
        print(f"  확신도:             {final_result.signal_confidence:.3f}")
        print(f"  크기:               {final_result.signal_score:.3f}")
        print(f"  기간:               {final_result.horizon}")

    if hasattr(final_result, 'v1_reason'):
        print()
        print("📊 v1 이유:")
        import json
        print(json.dumps(final_result.v1_reason, indent=2, ensure_ascii=False))

    print()
    print("=" * 70)
    print("시그널 해석")
    print("=" * 70)
    print()

    if final_result.signal_action == 'BUY':
        print(f"🟢 매수 신호")
        print(f"   포지션 크기: {final_result.signal_score * 100:.1f}%")
        if hasattr(final_result, 'horizon'):
            print(f"   보유 기간: {final_result.horizon}")
        if hasattr(final_result, 'signal_confidence'):
            print(f"   확신도: {final_result.signal_confidence * 100:.1f}%")

    elif final_result.signal_action == 'SELL':
        print(f"🔴 매도 신호")
        print(f"   포지션 크기: {final_result.signal_score * 100:.1f}%")
        if hasattr(final_result, 'horizon'):
            print(f"   보유 기간: {final_result.horizon}")
        if hasattr(final_result, 'signal_confidence'):
            print(f"   확신도: {final_result.signal_confidence * 100:.1f}%")

    elif final_result.signal_action == 'HOLD':
        print(f"🟡 관망 신호")
        if hasattr(final_result, 'direction') and hasattr(final_result, 'timing'):
            print(f"   방향: {final_result.direction}, 타이밍: {final_result.timing}")

    print()

except Exception as e:
    print(f"❌ 통합 분석 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("테스트 완료")
print("=" * 70)
