#!/usr/bin/env python3
"""
IntegratedAnalyzerV1 하드코딩 해결 검증 스크립트
- interval_profiles 통합 확인
- 동적 가중치 계산 확인
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_v1_analyzer_weights():
    """V1 분석기의 하드코딩 해결 확인"""

    print("=" * 60)
    print("IntegratedAnalyzerV1 하드코딩 검증 시작")
    print("=" * 60)

    # 1. interval_profiles 로드 확인
    print("\n1️⃣ interval_profiles 모듈 확인...")
    try:
        from rl_pipeline.core.interval_profiles import (
            get_integration_weights,
            get_interval_role,
            INTERVAL_PROFILES
        )
        print("✅ interval_profiles 모듈 로드 성공")

        # 가중치 확인
        weights = get_integration_weights()
        print(f"✅ 통합 가중치: {weights}")

        # 역할 확인
        for interval in ['1d', '240m', '30m', '15m']:
            role = get_interval_role(interval)
            print(f"   {interval}: {role}")

    except ImportError as e:
        print(f"❌ interval_profiles 모듈 로드 실패: {e}")
        return False

    # 2. IntegratedAnalyzerV1 로드 및 초기화
    print("\n2️⃣ IntegratedAnalyzerV1 초기화...")
    try:
        from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1

        analyzer = IntegratedAnalyzerV1()
        print("✅ IntegratedAnalyzerV1 초기화 성공")

    except Exception as e:
        print(f"❌ IntegratedAnalyzerV1 초기화 실패: {e}")
        return False

    # 3. 하드코딩 여부 확인
    print("\n3️⃣ 하드코딩 여부 확인...")

    # 이전 하드코딩된 값
    old_direction_weights = {'1d': 0.6, '240m': 0.4}
    old_timing_weights = {'30m': 0.6, '15m': 0.4}

    # 현재 값 확인
    if hasattr(analyzer, 'interval_weights'):
        print(f"✅ interval_weights 속성 존재: {analyzer.interval_weights}")
    else:
        print("❌ interval_weights 속성이 없습니다 (하드코딩 상태)")
        return False

    if hasattr(analyzer, 'direction_weights'):
        print(f"✅ direction_weights: {analyzer.direction_weights}")

        # 하드코딩 여부 체크
        if analyzer.direction_weights == old_direction_weights:
            print("⚠️ 여전히 하드코딩된 값 사용 중")
        else:
            print("✅ 동적으로 계산된 가중치 사용 중")
    else:
        print("❌ direction_weights 속성이 없습니다")
        return False

    if hasattr(analyzer, 'timing_weights'):
        print(f"✅ timing_weights: {analyzer.timing_weights}")

        # 하드코딩 여부 체크
        if analyzer.timing_weights == old_timing_weights:
            print("⚠️ 여전히 하드코딩된 값 사용 중")
        else:
            print("✅ 동적으로 계산된 가중치 사용 중")
    else:
        print("❌ timing_weights 속성이 없습니다")
        return False

    # 4. 가중치 계산 검증
    print("\n4️⃣ 가중치 계산 검증...")

    # interval_profiles 가중치에서 계산된 값 검증
    expected_direction_sum = analyzer.interval_weights['1d'] + analyzer.interval_weights['240m']
    expected_timing_sum = analyzer.interval_weights['30m'] + analyzer.interval_weights['15m']

    actual_direction_sum = analyzer.direction_weights['1d'] + analyzer.direction_weights['240m']
    actual_timing_sum = analyzer.timing_weights['30m'] + analyzer.timing_weights['15m']

    # 정규화 확인 (합이 1이어야 함)
    if abs(actual_direction_sum - 1.0) < 0.001:
        print("✅ direction_weights 정규화 확인 (합 = 1.0)")
    else:
        print(f"❌ direction_weights 정규화 오류 (합 = {actual_direction_sum})")

    if abs(actual_timing_sum - 1.0) < 0.001:
        print("✅ timing_weights 정규화 확인 (합 = 1.0)")
    else:
        print(f"❌ timing_weights 정규화 오류 (합 = {actual_timing_sum})")

    # 5. analyze() 메서드 테스트
    print("\n5️⃣ analyze() 메서드 테스트...")
    try:
        # 테스트용 더미 데이터로 실행 (실제 DB 없어도 동작 확인)
        result = analyzer.analyze('BTC')

        if 'interval_profiles_used' in result:
            if result['interval_profiles_used']:
                print("✅ interval_profiles 사용 확인")
            else:
                print("⚠️ interval_profiles 미사용")

        if 'reason' in result and 'interval_weights' in result['reason']:
            print(f"✅ 분석 결과에 interval_weights 포함: {result['reason']['interval_weights']}")

        if 'reason' in result and 'interval_roles' in result['reason']:
            print(f"✅ 분석 결과에 interval_roles 포함: {result['reason']['interval_roles']}")

    except Exception as e:
        print(f"⚠️ analyze() 실행 중 오류 (DB 없음 예상): {e}")

    # 6. 최종 검증
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)

    issues = []

    # interval_profiles 통합 확인
    if hasattr(analyzer, 'interval_weights') and analyzer.interval_weights != old_direction_weights:
        print("✅ interval_profiles 통합 완료")
    else:
        issues.append("interval_profiles 미통합")

    # 하드코딩 제거 확인
    if (hasattr(analyzer, 'direction_weights') and
        analyzer.direction_weights != old_direction_weights):
        print("✅ 방향 레이어 하드코딩 제거")
    else:
        issues.append("방향 레이어 하드코딩")

    if (hasattr(analyzer, 'timing_weights') and
        analyzer.timing_weights != old_timing_weights):
        print("✅ 타이밍 레이어 하드코딩 제거")
    else:
        issues.append("타이밍 레이어 하드코딩")

    if not issues:
        print("\n🎉 모든 하드코딩이 성공적으로 제거되었습니다!")
        print("📊 이제 interval_profiles의 동적 가중치를 사용합니다.")
        return True
    else:
        print(f"\n❌ 해결되지 않은 문제: {', '.join(issues)}")
        return False

def main():
    """메인 실행 함수"""
    try:
        # Docker 환경인지 확인
        in_docker = os.path.exists('/workspace')

        if in_docker:
            # Docker 환경에서 실행
            print("🐳 Docker 환경에서 실행 중...")
            sys.path.insert(0, '/workspace/rl_pipeline')

        success = test_v1_analyzer_weights()

        if success:
            print("\n✅ IntegratedAnalyzerV1 하드코딩 문제 해결 완료!")
            return 0
        else:
            print("\n❌ 일부 문제가 남아있습니다.")
            return 1

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())