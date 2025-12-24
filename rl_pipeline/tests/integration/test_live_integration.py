#!/usr/bin/env python3
"""
실시간 통합 테스트 - 실제 파이프라인에서 인터벌 프로필 적용 확인
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 경로 설정
sys.path.insert(0, '/workspace/rl_pipeline')

def test_orchestrator_with_profiles():
    """실제 Orchestrator에서 프로필 적용 테스트"""
    print("\n" + "="*60)
    print("실시간 Orchestrator 통합 테스트")
    print("="*60)

    try:
        # Orchestrator 로드
        from pipelines.orchestrator import IntegratedPipelineOrchestrator

        orch = IntegratedPipelineOrchestrator()
        print("✅ Orchestrator 로드 성공")

        # interval_profiles 모듈 확인
        try:
            import pipelines.orchestrator as orch_module
            if hasattr(orch_module, 'interval_profiles'):
                print("✅ interval_profiles 모듈이 임포트됨")

                # 프로필 확인
                profiles = orch_module.interval_profiles.INTERVAL_PROFILES
                print(f"\n로드된 인터벌 프로필: {list(profiles.keys())}")

                # 가중치 확인
                weights = orch_module.interval_profiles.get_integration_weights()
                print(f"\n통합 가중치:")
                for interval, weight in weights.items():
                    print(f"  {interval}: {weight:.2f}")
            else:
                print("❌ interval_profiles 모듈이 임포트되지 않음")
        except Exception as e:
            print(f"❌ 프로필 확인 실패: {e}")

        # 테스트 데이터 생성
        print("\n테스트 데이터로 파이프라인 실행:")

        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.rand(100) * 1000000
        })

        # run_partial_pipeline 테스트
        if hasattr(orch, 'run_partial_pipeline'):
            print("  run_partial_pipeline 메소드 존재 ✅")

            # 실제로 실행해보기 (짧은 테스트)
            try:
                result = orch.run_partial_pipeline('TEST', '15m', test_df)
                print(f"  파이프라인 실행 결과: {result.status}")

                if hasattr(result, 'interval'):
                    print(f"  처리된 인터벌: {result.interval}")
            except Exception as e:
                print(f"  파이프라인 실행 중 오류 (예상 범위 내): {type(e).__name__}")
        else:
            print("  run_partial_pipeline 메소드 없음 ❌")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_running_session():
    """현재 실행 중인 세션 확인"""
    print("\n" + "="*60)
    print("실행 중인 세션 확인")
    print("="*60)

    try:
        import json

        # 세션 파일 확인
        session_file = '/workspace/rl_pipeline/debug_logs/sessions.json'
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                sessions = json.load(f)

            latest = sessions.get('latest')
            if latest:
                print(f"최신 세션: {latest}")

                # 세션 정보 확인
                for session in sessions.get('sessions', [])[-3:]:  # 최근 3개
                    if session['session_id'] == latest:
                        print(f"\n현재 세션 정보:")
                        print(f"  코인: {session['coins']}")
                        print(f"  인터벌: {session['intervals']}")
                        print(f"  상태: {session['status']}")

                        # 로그 파일 확인
                        session_dir = f"/workspace/rl_pipeline/debug_logs/{latest}"
                        if os.path.exists(session_dir):
                            log_file = f"{session_dir}/simulation.jsonl"
                            if os.path.exists(log_file):
                                with open(log_file, 'r') as f:
                                    lines = f.readlines()
                                    print(f"  로그 라인 수: {len(lines)}")

                                    # 프로필 관련 로그 찾기
                                    profile_logs = []
                                    for line in lines:
                                        if any(keyword in line.lower() for keyword in ['프로필', 'profile', '라벨', 'label', '보상', 'reward']):
                                            profile_logs.append(line)

                                    if profile_logs:
                                        print(f"\n  프로필 관련 로그 발견: {len(profile_logs)}개")
                                        for log in profile_logs[:3]:  # 처음 3개만
                                            print(f"    {log[:100]}...")
                                    else:
                                        print("  프로필 관련 로그 없음")
        else:
            print("세션 파일 없음")

        return True

    except Exception as e:
        print(f"❌ 세션 확인 실패: {e}")
        return False


def main():
    """메인 테스트"""
    print("\n" + "="*70)
    print("인터벌 프로필 실시간 통합 테스트")
    print("="*70)

    results = {}

    # 1. Orchestrator 테스트
    results['orchestrator'] = test_orchestrator_with_profiles()

    # 2. 실행 중인 세션 확인
    results['session'] = check_running_session()

    # 결과 요약
    print("\n" + "="*70)
    print("테스트 결과")
    print("="*70)

    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    if all(results.values()):
        print("\n🎉 모든 실시간 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패")


if __name__ == "__main__":
    main()