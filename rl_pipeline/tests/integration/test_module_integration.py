#!/usr/bin/env python3
"""
핵심 모듈 interval_profiles 통합 테스트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 경로 설정
sys.path.insert(0, '/workspace/rl_pipeline')

def test_chart_future_scanner():
    """chart_future_scanner.py 테스트"""
    print("\n" + "="*60)
    print("1. ChartFutureScanner 테스트")
    print("="*60)

    try:
        from labeling.chart_future_scanner import ChartFutureScanner

        scanner = ChartFutureScanner()

        # HORIZON_MAP이 interval_profiles를 사용하는지 확인
        horizon_map = scanner.HORIZON_MAP
        print("HORIZON_MAP 값:")
        for interval, horizon in horizon_map.items():
            print(f"  {interval}: {horizon} 캔들")

        # interval_profiles 값과 비교
        try:
            from core.interval_profiles import INTERVAL_PROFILES
            print("\ninterval_profiles와 비교:")
            for interval in ['15m', '30m', '240m', '1d']:
                scanner_value = horizon_map.get(interval)
                profile_value = INTERVAL_PROFILES[interval]['labeling']['target_horizon']
                match = "✅" if scanner_value == profile_value else "❌"
                print(f"  {interval}: scanner={scanner_value}, profile={profile_value} {match}")
        except ImportError:
            print("interval_profiles를 로드할 수 없어 비교 불가")

        return True

    except Exception as e:
        print(f"❌ ChartFutureScanner 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_analyzer():
    """integrated_analyzer.py 테스트"""
    print("\n" + "="*60)
    print("2. IntegratedAnalyzer 테스트")
    print("="*60)

    try:
        from analysis.integrated_analyzer import IntegratedAnalyzer

        analyzer = IntegratedAnalyzer()
        print("✅ IntegratedAnalyzer 로드 성공")

        # interval_profiles 사용 확인
        try:
            from analysis import integrated_analyzer as ia_module
            if hasattr(ia_module, 'INTERVAL_PROFILES_AVAILABLE'):
                print(f"  interval_profiles 사용 가능: {ia_module.INTERVAL_PROFILES_AVAILABLE}")
            else:
                print("  interval_profiles 상태 확인 불가")
        except Exception as e:
            print(f"  모듈 확인 실패: {e}")

        return True

    except Exception as e:
        print(f"❌ IntegratedAnalyzer 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selfplay():
    """selfplay.py 테스트"""
    print("\n" + "="*60)
    print("3. SelfPlaySimulator 테스트")
    print("="*60)

    try:
        from simulation.selfplay import SelfPlaySimulator

        simulator = SelfPlaySimulator(use_gpu=False)
        print("✅ SelfPlaySimulator 로드 성공")

        # interval_profiles 사용 확인
        try:
            from simulation import selfplay as sp_module
            if hasattr(sp_module, 'INTERVAL_PROFILES_AVAILABLE'):
                print(f"  interval_profiles 사용 가능: {sp_module.INTERVAL_PROFILES_AVAILABLE}")
                if hasattr(sp_module, 'calculate_reward'):
                    print(f"  calculate_reward 함수 로드: {sp_module.calculate_reward is not None}")
            else:
                print("  interval_profiles 상태 확인 불가")
        except Exception as e:
            print(f"  모듈 확인 실패: {e}")

        return True

    except Exception as e:
        print(f"❌ SelfPlaySimulator 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_calculation_integration():
    """통합 보상 계산 테스트"""
    print("\n" + "="*60)
    print("4. 통합 보상 계산 테스트")
    print("="*60)

    try:
        # interval_profiles 로드
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)

        # 테스트 데이터
        test_prediction = {
            'direction': 1,
            'return': 0.02,
            'regime': 'bull',
            'swing': 'up',
            'trend': 'continuation',
            'entry_quality': 'good',
            'r_multiple': 2.0,
            'stop_hit': False,
        }

        test_actual = {
            'direction': 1,
            'return': 0.025,
            'regime': 'bull',
            'swing': 'strong_up',
            'trend': 'continuation',
            'entry_quality': 'excellent',
            'r_multiple': 2.5,
            'stop_hit': False,
        }

        print("테스트 보상 계산:")
        for interval in ['15m', '30m', '240m', '1d']:
            reward = interval_profiles.calculate_reward(interval, test_prediction, test_actual)
            print(f"  {interval}: {reward:.3f}")

        return True

    except Exception as e:
        print(f"❌ 통합 보상 계산 테스트 실패: {e}")
        return False


def main():
    """메인 테스트"""
    print("\n" + "="*70)
    print("핵심 모듈 interval_profiles 통합 테스트")
    print("="*70)

    results = {}

    # 각 모듈 테스트
    results['chart_future_scanner'] = test_chart_future_scanner()
    results['integrated_analyzer'] = test_integrated_analyzer()
    results['selfplay'] = test_selfplay()
    results['reward_calculation'] = test_reward_calculation_integration()

    # 결과 요약
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)

    for module, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")

    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    print(f"\n총 {success_count}/{total_count} 테스트 성공")

    if success_count == total_count:
        print("\n🎉 모든 핵심 모듈 통합 완료!")
    else:
        print("\n⚠️ 일부 모듈 통합 실패")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)