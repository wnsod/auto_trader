#!/usr/bin/env python3
"""
인터벌별 기능 통합 테스트
"""

import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# 경로 설정
sys.path.insert(0, '/workspace/rl_pipeline')

def test_label_generation():
    """실제 캔들 데이터로 라벨 생성 테스트"""
    print("\n" + "="*60)
    print("1. 라벨 생성 테스트")
    print("="*60)

    try:
        # interval_profiles 모듈 로드
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)

        # 실제 캔들 데이터 로드 (DB에서)
        conn = sqlite3.connect('/workspace/data_storage/rl_candles.db')

        test_results = {}

        for interval in ['15m', '30m', '240m', '1d']:
            print(f"\n[{interval}] 테스트:")

            # SOL 데이터 로드
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = 'SOL' AND interval = '{interval}'
            ORDER BY timestamp DESC
            LIMIT 500
            """

            df = pd.read_sql_query(query, conn)

            if len(df) == 0:
                print(f"  ⚠️ 데이터 없음")
                continue

            print(f"  - 캔들 수: {len(df)}개")

            # 라벨 생성
            labeled_df = interval_profiles.generate_labels(df.copy(), interval)

            # 결과 확인
            profile = interval_profiles.INTERVAL_PROFILES[interval]
            print(f"  - 라벨 타입: {profile['labeling']['label_type']}")
            print(f"  - 예측 기간: {profile['labeling']['target_horizon']} 캔들")

            # 라벨 분포
            label_counts = labeled_df['label'].value_counts()
            print("  - 라벨 분포:")
            for label, count in label_counts.items():
                pct = count/len(labeled_df)*100
                print(f"    {label}: {count}개 ({pct:.1f}%)")

            # NaN 체크
            nan_count = labeled_df['label'].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️ NaN 라벨: {nan_count}개")

            test_results[interval] = {
                'success': True,
                'label_type': profile['labeling']['label_type'],
                'label_counts': label_counts.to_dict()
            }

        conn.close()

        # 전체 결과
        success_count = sum(1 for r in test_results.values() if r.get('success'))
        print(f"\n✅ 라벨 생성 테스트: {success_count}/4 성공")

        return test_results

    except Exception as e:
        print(f"❌ 라벨 생성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_reward_calculation():
    """보상 계산 테스트"""
    print("\n" + "="*60)
    print("2. 보상 계산 테스트")
    print("="*60)

    try:
        # interval_profiles 모듈 로드
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)

        # 테스트 케이스들
        test_cases = [
            {
                'name': '완벽한 예측',
                'prediction': {
                    'direction': 1, 'return': 0.03, 'regime': 'bull',
                    'swing': 'up', 'trend': 'continuation',
                    'entry_quality': 'excellent', 'r_multiple': 3.0,
                    'r_max': 0.03, 'r_min': 0,
                    'trend_continues': True, 'reversal': False,
                    'volatility': 0.02, 'stop_hit': False
                },
                'actual': {
                    'direction': 1, 'return': 0.03, 'regime': 'bull',
                    'swing': 'strong_up', 'trend': 'continuation',
                    'entry_quality': 'excellent', 'r_multiple': 3.0,
                    'r_max': 0.03, 'r_min': 0,
                    'trend_continues': True, 'reversal': False,
                    'volatility': 0.02, 'stop_hit': False
                }
            },
            {
                'name': '방향 틀림',
                'prediction': {
                    'direction': 1, 'return': 0.02, 'regime': 'bull',
                    'swing': 'up', 'trend': 'continuation',
                    'entry_quality': 'good', 'r_multiple': 2.0,
                    'r_max': 0.02, 'r_min': 0,
                    'trend_continues': True, 'reversal': False,
                    'volatility': 0.01, 'stop_hit': False
                },
                'actual': {
                    'direction': -1, 'return': -0.02, 'regime': 'bear',
                    'swing': 'down', 'trend': 'reversal',
                    'entry_quality': 'poor', 'r_multiple': -1.0,
                    'r_max': 0, 'r_min': -0.02,
                    'trend_continues': False, 'reversal': True,
                    'volatility': 0.02, 'stop_hit': True
                }
            }
        ]

        for test_case in test_cases:
            print(f"\n테스트: {test_case['name']}")

            for interval in ['15m', '30m', '240m', '1d']:
                try:
                    reward = interval_profiles.calculate_reward(
                        interval,
                        test_case['prediction'],
                        test_case['actual']
                    )

                    # 보상 가중치 확인
                    profile = interval_profiles.INTERVAL_PROFILES[interval]
                    weights = profile['reward_weights']

                    print(f"  {interval}: {reward:.3f} (가중치: {list(weights.keys())})")

                except Exception as e:
                    print(f"  {interval}: ❌ 오류 - {e}")

        print("\n✅ 보상 계산 테스트 완료")
        return True

    except Exception as e:
        print(f"❌ 보상 계산 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_integration():
    """orchestrator.py에서 프로필 사용 확인"""
    print("\n" + "="*60)
    print("3. Orchestrator 통합 테스트")
    print("="*60)

    try:
        # orchestrator에서 interval_profiles 임포트 확인
        with open('/workspace/rl_pipeline/pipelines/orchestrator.py', 'r') as f:
            content = f.read()

        checks = {
            'import 확인': 'import rl_pipeline.core.interval_profiles as interval_profiles' in content,
            'generate_labels 사용': 'interval_profiles.generate_labels' in content,
            'calculate_reward 사용': 'interval_profiles.calculate_reward' in content,
            'get_integration_weights 사용': 'interval_profiles.get_integration_weights' in content,
            'get_interval_role 사용': 'interval_profiles.get_interval_role' in content
        }

        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")

        success = all(checks.values())
        if success:
            print("\n✅ Orchestrator 통합 확인 완료")
        else:
            print("\n⚠️ 일부 기능이 통합되지 않음")

        return success

    except Exception as e:
        print(f"❌ Orchestrator 통합 테스트 실패: {e}")
        return False


def test_integration_weights():
    """통합 분석 가중치 테스트"""
    print("\n" + "="*60)
    print("4. 통합 분석 가중치 테스트")
    print("="*60)

    try:
        # interval_profiles 모듈 로드
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)

        weights = interval_profiles.get_integration_weights()

        print("\n가중치 분배:")
        total = 0
        for interval, weight in weights.items():
            role = interval_profiles.get_interval_role(interval)
            print(f"  {interval}: {weight:.2f} ({weight*100:.0f}%)")
            print(f"    역할: {role}")
            total += weight

        print(f"\n합계: {total:.2f}")

        if abs(total - 1.0) < 0.001:
            print("✅ 가중치 합계 100% 확인")
            return True
        else:
            print(f"❌ 가중치 합계 오류: {total*100:.1f}%")
            return False

    except Exception as e:
        print(f"❌ 가중치 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "="*70)
    print("인터벌별 프로필 기능 통합 테스트")
    print("="*70)

    results = {}

    # 1. 라벨 생성 테스트
    label_results = test_label_generation()
    results['labels'] = len(label_results) > 0

    # 2. 보상 계산 테스트
    results['rewards'] = test_reward_calculation()

    # 3. Orchestrator 통합 테스트
    results['orchestrator'] = test_orchestrator_integration()

    # 4. 가중치 테스트
    results['weights'] = test_integration_weights()

    # 최종 결과
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)

    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")

    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    print(f"\n총 {success_count}/{total_count} 테스트 성공")

    if success_count == total_count:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)