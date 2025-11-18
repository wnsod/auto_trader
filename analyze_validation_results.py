"""
검증 결과 상세 분석 스크립트
"""
import json
import os
from datetime import datetime
from collections import Counter, defaultdict

def analyze_validation_results():
    """검증 결과 상세 분석"""

    print("="*70)
    print("📊 Validation System 상세 분석 리포트")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    validation_log = '/workspace/rl_pipeline/validation/reports/validation_log.jsonl'

    if not os.path.exists(validation_log):
        print("❌ 검증 로그 파일이 없습니다.")
        return

    # 모든 검증 로그 읽기
    validations = []
    with open(validation_log, 'r') as f:
        for line in f:
            try:
                validations.append(json.loads(line.strip()))
            except:
                continue

    if not validations:
        print("⚠️ 검증 로그가 비어있습니다.")
        return

    print(f"\n📈 전체 검증 통계:")
    print(f"  • 총 검증 수: {len(validations)}")

    # 상태별 집계
    status_counts = Counter(v['status'] for v in validations)
    print(f"  • 상태별:")
    for status, count in status_counts.most_common():
        pct = count / len(validations) * 100
        icon = '✅' if status == 'passed' else '⚠️' if status == 'warning' else '❌'
        print(f"    {icon} {status}: {count} ({pct:.1f}%)")

    # 컴포넌트별 분석
    component_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'warning': 0, 'failed': 0})
    for v in validations:
        comp = v.get('component', 'Unknown')
        status = v['status']
        component_stats[comp]['total'] += 1
        component_stats[comp][status] += 1

    print(f"\n🔍 컴포넌트별 분석:")
    for comp, stats in component_stats.items():
        success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"\n  📦 {comp}:")
        print(f"    • 총 검증: {stats['total']}")
        print(f"    • 성공률: {success_rate:.1f}%")
        print(f"    • 통과: {stats['passed']}, 경고: {stats['warning']}, 실패: {stats['failed']}")

    # 최근 실패/경고 이슈 분석
    recent_issues = []
    for v in validations[-50:]:  # 최근 50개
        if v['status'] in ['warning', 'failed'] and 'issues' in v:
            for issue in v['issues']:
                recent_issues.append({
                    'component': v.get('component'),
                    'coin': v.get('context', {}).get('coin'),
                    'interval': v.get('context', {}).get('interval'),
                    'check': issue.get('check'),
                    'message': issue.get('message'),
                    'severity': issue.get('severity'),
                    'timestamp': v.get('timestamp')
                })

    # 이슈 빈도 분석
    if recent_issues:
        print(f"\n🔴 주요 문제 패턴 (최근 50개 검증):")

        # 체크별 이슈 빈도
        check_counts = Counter(i['check'] for i in recent_issues)
        print(f"\n  📌 가장 빈번한 문제:")
        for check, count in check_counts.most_common(5):
            print(f"    • {check}: {count}회")

        # 코인별 이슈 빈도
        coin_issues = Counter(i['coin'] for i in recent_issues if i['coin'])
        if coin_issues:
            print(f"\n  🪙 코인별 이슈:")
            for coin, count in coin_issues.most_common():
                print(f"    • {coin}: {count}회")

    # 개선 권장사항
    print(f"\n💡 개선 권장사항:")

    issues_to_fix = []

    # 전략 수 문제
    if 'strategy_count_range' in [i['check'] for i in recent_issues]:
        issues_to_fix.append("전략 생성 수가 여전히 임계값 미달 (최소 100개 필요)")
        print(f"  1️⃣ 전략 생성 수 추가 증가 필요")
        print(f"     → VAL_MIN_STRATEGIES를 30으로 낮추거나")
        print(f"     → 실제 생성 전략 수를 증가")

    # 라우팅 결과 문제
    if 'routing_results_not_empty' in [i['check'] for i in recent_issues]:
        issues_to_fix.append("라우팅 결과가 비어있음")
        print(f"  2️⃣ 라우팅 결과 생성 확인 필요")
        print(f"     → 레짐 라우터가 정상 작동하는지 확인")

    # 백테스트 문제
    if 'backtest_presence' in [i['check'] for i in recent_issues]:
        issues_to_fix.append("백테스트 결과 누락")
        print(f"  3️⃣ 백테스트 실행 확인 필요")
        print(f"     → ENABLE_BACKTEST 설정 확인")

    # 자동 복구 통계
    auto_fixed_count = sum(v.get('statistics', {}).get('auto_fixed', 0) for v in validations)
    if auto_fixed_count > 0:
        print(f"\n🔧 자동 복구 통계:")
        print(f"  • 자동 수정된 이슈: {auto_fixed_count}개")

    # 신뢰도 수준 변화 추적
    trust_levels = []
    for v in validations:
        context = v.get('context', {})
        if 'trust_level' in context:
            trust_levels.append({
                'timestamp': v.get('timestamp'),
                'level': context['trust_level']
            })

    if trust_levels:
        print(f"\n📈 신뢰도 수준 변화:")
        latest_trust = trust_levels[-1]['level'] if trust_levels else 'Unknown'
        print(f"  • 현재 신뢰도: {latest_trust}")

    # 실행 중인 백그라운드 프로세스 확인
    print(f"\n⚙️ 시스템 상태:")
    try:
        import subprocess
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            cwd='/workspace'
        )
        if 'python' in result.stdout:
            print(f"  🟢 Python 프로세스 실행 중")
        else:
            print(f"  🔴 실행 중인 Python 프로세스 없음")
    except:
        print(f"  ⚠️ 프로세스 상태 확인 불가")

    return issues_to_fix

if __name__ == "__main__":
    issues = analyze_validation_results()

    if issues:
        print(f"\n❗ 해결이 필요한 주요 이슈:")
        for idx, issue in enumerate(issues, 1):
            print(f"  {idx}. {issue}")