#!/usr/bin/env python
"""
Orchestrator v1 통합 검증

v1 로직이 제대로 작동하는지 상세 검증
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator, PipelineResult
from rl_pipeline.data.candle_loader import load_candle_data_for_coin
from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1
import json

COIN = 'LINK'
INTERVALS = ['15m', '30m', '240m', '1d']

print("=" * 70)
print("Orchestrator v1 통합 검증")
print("=" * 70)
print()

# 1. v1 직접 호출 (기준)
print("1️⃣  v1 직접 호출 (기준)")
print("-" * 70)

analyzer_v1 = IntegratedAnalyzerV1()
v1_direct = analyzer_v1.analyze(COIN)

print(f"방향:     {v1_direct['direction']}")
print(f"타이밍:   {v1_direct['timing']}")
print(f"크기:     {v1_direct['size']:.3f}")
print(f"확신도:   {v1_direct['confidence']:.3f}")
print(f"기간:     {v1_direct['horizon']}")
print()
print("이유:")
print(json.dumps(v1_direct['reason'], indent=2, ensure_ascii=False))
print()

# 2. Orchestrator를 통한 호출
print("=" * 70)
print("2️⃣  Orchestrator를 통한 호출")
print("-" * 70)

orchestrator = IntegratedPipelineOrchestrator()

# 캔들 데이터 로드
all_candle_data = load_candle_data_for_coin(COIN, INTERVALS)

# 더미 PipelineResult
pipeline_results = []
for interval in INTERVALS:
    result = PipelineResult(
        coin=COIN,
        interval=interval,
        status='completed',
        strategies_created=100,
        regime_detected='neutral',
        signal_action='HOLD',
        signal_score=0.5
    )
    pipeline_results.append(result)

# 통합 분석 실행
final_result = orchestrator.run_integrated_analysis_all_intervals(
    coin=COIN,
    pipeline_results=pipeline_results,
    all_candle_data=all_candle_data
)

print(f"signal_action:       {final_result.signal_action}")
print(f"signal_score:        {final_result.signal_score:.3f}")

# PipelineResult의 모든 속성 출력
print()
print("PipelineResult 모든 속성:")
for attr in dir(final_result):
    if not attr.startswith('_'):
        try:
            val = getattr(final_result, attr)
            if not callable(val):
                print(f"  {attr:20s} = {val}")
        except:
            pass

print()

# 3. 비교 검증
print("=" * 70)
print("3️⃣  비교 검증")
print("-" * 70)

# signal_action 매핑 확인
expected_action = 'BUY' if v1_direct['direction'] == 'LONG' and v1_direct['timing'] == 'NOW' else 'HOLD'
actual_action = final_result.signal_action

print(f"✅ signal_action 매핑:")
print(f"   v1: direction={v1_direct['direction']}, timing={v1_direct['timing']}")
print(f"   예상: {expected_action}")
print(f"   실제: {actual_action}")
print(f"   {'✅ 일치' if expected_action == actual_action else '❌ 불일치'}")
print()

# signal_score 매핑 확인
expected_score = v1_direct['size']
actual_score = final_result.signal_score

print(f"✅ signal_score 매핑:")
print(f"   v1 size: {expected_score:.3f}")
print(f"   실제:    {actual_score:.3f}")
print(f"   {'✅ 일치' if abs(expected_score - actual_score) < 0.001 else '❌ 불일치'}")
print()

# 4. 최종 판정
print("=" * 70)
print("4️⃣  최종 판정")
print("=" * 70)
print()

all_ok = True

if expected_action != actual_action:
    print("❌ signal_action 매핑 실패")
    all_ok = False

if abs(expected_score - actual_score) > 0.001:
    print("❌ signal_score 매핑 실패")
    all_ok = False

if all_ok:
    print("✅ Orchestrator v1 통합 성공!")
    print()
    print("📊 최종 시그널:")
    print(f"   액션:     {final_result.signal_action}")
    print(f"   포지션:   {final_result.signal_score * 100:.1f}%")
    print()
    print("📈 v1 상세:")
    print(f"   방향:     {v1_direct['direction']}")
    print(f"   타이밍:   {v1_direct['timing']}")
    print(f"   확신도:   {v1_direct['confidence'] * 100:.1f}%")
    print(f"   기간:     {v1_direct['horizon']}")
else:
    print("❌ Orchestrator v1 통합 실패")

print()
print("=" * 70)
