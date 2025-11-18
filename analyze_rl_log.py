#!/usr/bin/env python
"""RL 로그 분석 스크립트"""

with open('C:\\auto_trader\\rl_pipeline\\test_rl_log.txt', encoding='utf-8') as f:
    lines = f.readlines()

print("=" * 70)
print("RL Pipeline 로그 분석")
print("=" * 70)
print()

print(f"📊 전체 통계:")
print(f"  - 총 라인 수: {len(lines):,}개")
print()

# WARNING 분석
warnings = [l for l in lines if 'WARNING' in l]
print(f"⚠️  WARNING 메시지:")
print(f"  - 총 개수: {len(warnings)}개")

# WARNING 종류별 분류
direction_warnings = [w for w in warnings if '방향 재평가' in w]
other_warnings = [w for w in warnings if '방향 재평가' not in w]

print(f"  - 방향 재평가: {len(direction_warnings)}개")
print(f"  - 기타: {len(other_warnings)}개")
print()

# 기타 WARNING 상세
if other_warnings:
    print("  기타 WARNING 내용:")
    for w in other_warnings[:10]:
        print(f"    {w.strip()}")
    if len(other_warnings) > 10:
        print(f"    ... 외 {len(other_warnings) - 10}개")
print()

# 주요 이벤트 카운트
events = {
    '전략 생성 완료': [l for l in lines if '전략 생성 완료' in l or '시장 분석 기반 전략 생성 완료' in l],
    'Self-play 완료': [l for l in lines if 'Self-play' in l and '완료' in l],
    '롤업 및 등급 평가 완료': [l for l in lines if '롤업 및 등급 평가 완료' in l or '롤업/등급 평가 완료' in l],
    'WAL 체크포인트': [l for l in lines if 'WAL 체크포인트 완료' in l],
    '통합 분석 완료': [l for l in lines if '통합분석 완료' in l or '통합 분석 완료' in l],
    'Paper Trading 시작': [l for l in lines if 'Paper Trading 시작' in l or 'Paper Trading 세션 생성 완료' in l],
}

print("📈 주요 이벤트:")
for event_name, event_lines in events.items():
    print(f"  - {event_name}: {len(event_lines)}개")
print()

# 인터벌별 분석
intervals = {}
for line in lines:
    for interval in ['15m', '30m', '240m', '1d']:
        if f'ADA-{interval}' in line and '파이프라인 실행' in line:
            intervals[interval] = intervals.get(interval, 0) + 1

if intervals:
    print("📊 인터벌별 실행:")
    for interval, count in sorted(intervals.items()):
        print(f"  - ADA-{interval}: {count}회")
    print()

# 최종 결과 확인
final_lines = lines[-50:]
final_status = []
for line in final_lines:
    if '완료' in line or '성공' in line or '실패' in line:
        final_status.append(line.strip())

print("✅ 최종 상태:")
# 핵심 최종 메시지만 추출
key_final = [s for s in final_status if any(kw in s for kw in ['파이프라인 완료', '통합 파이프라인 성공', 'Paper Trading', '처리 성공'])]
for status in key_final[-10:]:
    print(f"  {status}")
print()

print("=" * 70)
print("✅ 로그 분석 완료!")
print("=" * 70)
