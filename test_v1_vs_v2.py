#!/usr/bin/env python
"""
v1 vs v2 비교 테스트

v2가 v1 기본 파라미터로 v1과 동일한 결과를 내는지 검증
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1
from rl_pipeline.analysis.integrated_analysis_v2 import IntegratedAnalyzerV2, V2Parameters

COIN = 'LINK'

print("=" * 70)
print("v1 vs v2 비교 테스트")
print("=" * 70)
print()

# v1 실행
print("1️⃣  v1 실행")
print("-" * 70)

analyzer_v1 = IntegratedAnalyzerV1()
result_v1 = analyzer_v1.analyze(COIN)

print(f"방향:     {result_v1['direction']}")
print(f"타이밍:   {result_v1['timing']}")
print(f"크기:     {result_v1['size']:.3f}")
print(f"확신도:   {result_v1['confidence']:.3f}")
print(f"기간:     {result_v1['horizon']}")
print()

# v2 실행 (v1 기본 파라미터)
print("2️⃣  v2 실행 (v1 기본 파라미터)")
print("-" * 70)

params_v1_default = V2Parameters()  # v1 기본값
analyzer_v2 = IntegratedAnalyzerV2(params_v1_default)
result_v2 = analyzer_v2.analyze(COIN)

print(f"방향:     {result_v2['direction']}")
print(f"타이밍:   {result_v2['timing']}")
print(f"크기:     {result_v2['size']:.3f}")
print(f"확신도:   {result_v2['confidence']:.3f}")
print(f"기간:     {result_v2['horizon']}")
print()

# 비교
print("=" * 70)
print("3️⃣  비교 결과")
print("=" * 70)
print()

all_match = True

# 방향 비교
if result_v1['direction'] == result_v2['direction']:
    print(f"✅ 방향 일치:     {result_v1['direction']}")
else:
    print(f"❌ 방향 불일치:   v1={result_v1['direction']}, v2={result_v2['direction']}")
    all_match = False

# 타이밍 비교
if result_v1['timing'] == result_v2['timing']:
    print(f"✅ 타이밍 일치:   {result_v1['timing']}")
else:
    print(f"❌ 타이밍 불일치: v1={result_v1['timing']}, v2={result_v2['timing']}")
    all_match = False

# 크기 비교 (소수점 오차 허용)
size_diff = abs(result_v1['size'] - result_v2['size'])
if size_diff < 0.001:
    print(f"✅ 크기 일치:     {result_v1['size']:.3f} (차이: {size_diff:.6f})")
else:
    print(f"❌ 크기 불일치:   v1={result_v1['size']:.3f}, v2={result_v2['size']:.3f}, 차이={size_diff:.6f}")
    all_match = False

# 확신도 비교
confidence_diff = abs(result_v1['confidence'] - result_v2['confidence'])
if confidence_diff < 0.001:
    print(f"✅ 확신도 일치:   {result_v1['confidence']:.3f} (차이: {confidence_diff:.6f})")
else:
    print(f"❌ 확신도 불일치: v1={result_v1['confidence']:.3f}, v2={result_v2['confidence']:.3f}, 차이={confidence_diff:.6f}")
    all_match = False

# 기간 비교
if result_v1['horizon'] == result_v2['horizon']:
    print(f"✅ 기간 일치:     {result_v1['horizon']}")
else:
    print(f"❌ 기간 불일치:   v1={result_v1['horizon']}, v2={result_v2['horizon']}")
    all_match = False

print()

# 최종 판정
print("=" * 70)
if all_match:
    print("✅ v2가 v1과 동일한 결과를 생성합니다!")
    print("   → v2 구현 성공, 파라미터 학습 준비 완료")
else:
    print("❌ v2가 v1과 다른 결과를 생성합니다!")
    print("   → v2 구현 검토 필요")
print("=" * 70)
