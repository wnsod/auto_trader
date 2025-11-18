#!/usr/bin/env python
"""
통합 분석 v1 테스트 및 검증
"""
import sys
sys.path.append('/workspace')

import sqlite3
from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1, analyze_coin

COIN = 'LINK'

print("=" * 70)
print("통합 분석 v1 테스트")
print("=" * 70)
print()

# ==================== 1. 데이터 로드 테스트 ====================
print("1️⃣  데이터 로드 테스트")
print("-" * 70)

analyzer = IntegratedAnalyzerV1()
interval_data = analyzer._load_interval_data(COIN)

for interval in ['15m', '30m', '240m', '1d']:
    if interval_data.get(interval) and interval_data[interval]:
        data = interval_data[interval]
        print(f"\n{interval}:")
        print(f"  전체 전략: {data['total_count']}개")
        print(f"  필터링 후: {data['filtered_count']}개")
        print(f"  가중 점수: {data['weighted_score']:.4f} ({data['weighted_score']*100:.2f}%)")

        # 등급 분포
        grade_dist = {}
        for s in data['strategies']:
            grade = s['grade']
            grade_dist[grade] = grade_dist.get(grade, 0) + 1

        grade_str = ', '.join([f"{g}:{c}" for g, c in sorted(grade_dist.items())])
        print(f"  등급 분포: {grade_str}")
    else:
        print(f"\n{interval}: 데이터 없음")

print()

# ==================== 2. Layer 1 테스트 (방향 결정) ====================
print("2️⃣  Layer 1: 방향 결정 테스트")
print("-" * 70)

direction, strength, reason = analyzer._determine_direction(interval_data)

print(f"\n방향: {direction}")
print(f"강도: {strength:.3f}")
print(f"\n상세:")
print(f"  1d 점수:     {reason['1d_score']:.4f} ({reason['1d_score']*100:.2f}%)")
print(f"  240m 점수:   {reason['240m_score']:.4f} ({reason['240m_score']*100:.2f}%)")
print(f"  가중 점수:   {reason['weighted_score']:.4f} ({reason['weighted_score']*100:.2f}%)")
print(f"  임계값:      ±{reason['threshold']*100:.2f}%")

print()

# ==================== 3. Layer 2 테스트 (타이밍 결정) ====================
print("3️⃣  Layer 2: 타이밍 결정 테스트")
print("-" * 70)

timing, timing_conf, timing_reason = analyzer._determine_timing(interval_data)

print(f"\n타이밍: {timing}")
print(f"확신도: {timing_conf:.3f}")
print(f"\n상세:")
print(f"  30m 점수:    {timing_reason['30m_score']:.4f} ({timing_reason['30m_score']*100:.2f}%)")
print(f"  15m 점수:    {timing_reason['15m_score']:.4f} ({timing_reason['15m_score']*100:.2f}%)")
print(f"  가중 점수:   {timing_reason['weighted_score']:.4f} ({timing_reason['weighted_score']*100:.2f}%)")
print(f"  임계값:      ±{timing_reason['threshold']*100:.2f}%")

print()

# ==================== 4. Layer 3 테스트 (확신도 및 크기) ====================
print("4️⃣  Layer 3: 확신도 및 크기 테스트")
print("-" * 70)

convergence = analyzer._check_convergence(interval_data)
confidence = analyzer._calculate_confidence(strength, timing_conf, interval_data)
size = analyzer._calculate_position_size(confidence, strength)
horizon = analyzer._determine_horizon(direction, timing, interval_data)

print(f"\n수렴도:      {convergence:.3f}")
print(f"종합 확신도: {confidence:.3f}")
print(f"포지션 크기: {size:.3f}")
print(f"거래 기간:   {horizon}")

print()

# ==================== 5. 발산 감지 테스트 ====================
print("5️⃣  발산 감지 테스트")
print("-" * 70)

divergence = analyzer._detect_divergence(interval_data)

print(f"\n발산 여부:     {divergence['is_divergent']}")
print(f"장기 점수:     {divergence['long_term_score']:.4f} ({divergence['long_term_score']*100:.2f}%)")
print(f"단기 점수:     {divergence['short_term_score']:.4f} ({divergence['short_term_score']*100:.2f}%)")

if divergence['is_divergent']:
    print("\n⚠️  장기/단기 방향 불일치 감지!")
    if divergence['long_term_score'] > 0 > divergence['short_term_score']:
        print("  → 장기 상승 추세이지만 단기 조정 중")
    else:
        print("  → 장기 하락 추세이지만 단기 반등 중")

print()

# ==================== 6. 전체 통합 분석 테스트 ====================
print("6️⃣  전체 통합 분석 테스트")
print("-" * 70)

result = analyze_coin(COIN)

print(f"\n✅ 통합 분석 결과:")
print(f"  방향:     {result['direction']}")
print(f"  타이밍:   {result['timing']}")
print(f"  크기:     {result['size']:.3f}")
print(f"  확신도:   {result['confidence']:.3f}")
print(f"  기간:     {result['horizon']}")

print()

# ==================== 7. 의사결정 해석 ====================
print("7️⃣  의사결정 해석")
print("-" * 70)

print("\n📊 트레이딩 시그널:")

if result['direction'] == 'NEUTRAL' or result['timing'] == 'WAIT':
    print("  🟡 관망 - 거래하지 않음")
    if result['direction'] == 'NEUTRAL':
        print("     이유: 방향 불명확")
    if result['timing'] == 'WAIT':
        print("     이유: 타이밍 부적절")

elif result['direction'] == 'LONG' and result['timing'] == 'NOW':
    print(f"  🟢 매수 진입 - {result['size']*100:.1f}% 포지션")
    print(f"     기간: {result['horizon']} 기준")
    print(f"     확신도: {result['confidence']*100:.1f}%")

elif result['direction'] == 'SHORT' and result['timing'] == 'NOW':
    print(f"  🔴 매도 진입 - {result['size']*100:.1f}% 포지션")
    print(f"     기간: {result['horizon']} 기준")
    print(f"     확신도: {result['confidence']*100:.1f}%")

elif result['timing'] == 'EXIT':
    print(f"  ⛔ 청산 신호")
    print(f"     이유: 단기 인터벌 반전")

print()

# ==================== 8. 이전 로직과 비교 ====================
print("8️⃣  v0 vs v1 비교")
print("-" * 70)

# v0: 단순 평균
simple_avg = sum([
    interval_data[i]['weighted_score']
    for i in ['15m', '30m', '240m', '1d']
    if interval_data.get(i) and interval_data[i]
]) / 4

print(f"\nv0 (단순 평균):    {simple_avg:.4f} ({simple_avg*100:.2f}%)")
print(f"v1 (계층 분석):")
print(f"  방향 점수:       {reason['weighted_score']:.4f} ({reason['weighted_score']*100:.2f}%)")
print(f"  타이밍 점수:     {timing_reason['weighted_score']:.4f} ({timing_reason['weighted_score']*100:.2f}%)")

print(f"\n차이:")
print(f"  v0는 모든 인터벌을 동등하게 평균")
print(f"  v1은 장기=방향, 단기=타이밍으로 분리")
print(f"  → 1d 상승 추세에서 15m으로 여러 번 매매 가능")

print()
print("=" * 70)
print("테스트 완료")
print("=" * 70)

# ==================== 9. DB 검증 ====================
print("\n9️⃣  데이터베이스 검증")
print("-" * 70)

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 전체 통계
cursor.execute("""
    SELECT COUNT(*) FROM strategy_grades WHERE coin=?
""", (COIN,))
total_grades = cursor.fetchone()[0]

cursor.execute("""
    SELECT grade, COUNT(*) FROM strategy_grades
    WHERE coin=?
    GROUP BY grade
    ORDER BY
        CASE grade
            WHEN 'S' THEN 1
            WHEN 'A' THEN 2
            WHEN 'B' THEN 3
            WHEN 'C' THEN 4
            WHEN 'D' THEN 5
            WHEN 'F' THEN 6
        END
""", (COIN,))
grade_dist_all = cursor.fetchall()

print(f"\n전체 등급 분포 ({total_grades}개):")
for grade, count in grade_dist_all:
    pct = count / total_grades * 100
    print(f"  {grade}: {count:3d}개 ({pct:5.1f}%)")

# D/F 필터링 확인
d_f_count = sum(count for grade, count in grade_dist_all if grade in ['D', 'F'])
filtered_count = total_grades - d_f_count

print(f"\n필터링:")
print(f"  D/F 등급: {d_f_count}개 제외")
print(f"  사용 전략: {filtered_count}개")

conn.close()

print()
print("=" * 70)
print("✅ 모든 테스트 완료")
print("=" * 70)
