#!/usr/bin/env python3
"""
ISSUES_FIXED_REPORT.md 개선사항 테스트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 경로 설정
sys.path.insert(0, '/workspace/rl_pipeline')

def test_normalize_interval():
    """1. 인터벌 이름 정규화 테스트"""
    print("\n" + "="*60)
    print("1. 인터벌 정규화 테스트")
    print("="*60)

    try:
        # 직접 파일 경로로 임포트
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)
        normalize_interval = interval_profiles.normalize_interval

        # 테스트 케이스
        test_cases = [
            # (입력, 기대값, 설명)
            ('15m', '15m', '표준 형식'),
            ('15M', '15m', '대문자'),
            ('30m', '30m', '표준 형식'),
            ('30M', '30m', '대문자'),
            ('240m', '240m', '표준 형식'),
            ('240M', '240m', '대문자'),
            ('4h', '240m', '4h → 240m 변환'),
            ('4H', '240m', '4H → 240m 변환'),
            ('1d', '1d', '표준 형식'),
            ('1D', '1d', '대문자'),
            (' 15m ', '15m', '공백 제거'),
            ('4hour', '240m', '4hour → 240m'),
        ]

        success_count = 0
        for input_val, expected, desc in test_cases:
            try:
                result = normalize_interval(input_val)
                if result == expected:
                    print(f"  ✅ '{input_val}' → '{result}' ({desc})")
                    success_count += 1
                else:
                    print(f"  ❌ '{input_val}' → '{result}' (기대: '{expected}')")
            except Exception as e:
                print(f"  ❌ '{input_val}' 오류: {e}")

        # 잘못된 형식 테스트
        invalid_cases = ['5m', '10m', '1h', '2d', 'invalid', '', None]
        error_count = 0
        print("\n잘못된 형식 테스트:")
        for invalid in invalid_cases:
            try:
                if invalid is None:
                    continue  # None은 다른 처리
                result = normalize_interval(invalid)
                print(f"  ❌ '{invalid}' → '{result}' (에러가 발생해야 함)")
            except ValueError as e:
                print(f"  ✅ '{invalid}' → ValueError: 예상대로 에러 발생")
                error_count += 1
            except Exception as e:
                print(f"  ⚠️ '{invalid}' → 예상치 못한 에러: {e}")

        print(f"\n결과: 정상 케이스 {success_count}/{len(test_cases)} 성공")
        print(f"      에러 케이스 {error_count}/{len(invalid_cases)-1} 성공")

        return success_count == len(test_cases) and error_count >= 3

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nan_handling():
    """2. NaN 값 처리 테스트"""
    print("\n" + "="*60)
    print("2. NaN 값 처리 테스트")
    print("="*60)

    try:
        # 직접 파일 경로로 임포트
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)
        generate_labels = interval_profiles.generate_labels

        # 테스트 데이터 생성 (마지막 20개가 NaN이 되도록)
        n_rows = 100
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1D'),
            'open': np.random.randn(n_rows).cumsum() + 100,
            'high': np.random.randn(n_rows).cumsum() + 101,
            'low': np.random.randn(n_rows).cumsum() + 99,
            'close': np.random.randn(n_rows).cumsum() + 100,
            'volume': np.random.rand(n_rows) * 1000000
        })

        success_count = 0
        for interval in ['15m', '30m', '240m', '1d']:
            try:
                labeled_df = generate_labels(test_df, interval)

                # unknown 라벨 확인
                unknown_count = (labeled_df['label'] == 'unknown').sum()
                total_count = len(labeled_df)

                print(f"\n{interval} 테스트:")
                print(f"  전체 행: {total_count}")
                print(f"  'unknown' 라벨: {unknown_count}개")

                # 라벨 분포
                label_counts = labeled_df['label'].value_counts()
                for label, count in label_counts.items():
                    print(f"  {label}: {count}개 ({count/total_count*100:.1f}%)")

                # NaN이 없어야 함
                nan_count = labeled_df['label'].isna().sum()
                if nan_count == 0:
                    print(f"  ✅ NaN 값 없음 (모두 처리됨)")
                    success_count += 1
                else:
                    print(f"  ❌ NaN 값 {nan_count}개 남아있음")

            except Exception as e:
                print(f"  ❌ {interval} 오류: {e}")

        print(f"\n결과: {success_count}/4 인터벌 성공")
        return success_count == 4

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_required_columns():
    """3. 필수 컬럼 체크 테스트"""
    print("\n" + "="*60)
    print("3. 필수 컬럼 체크 테스트")
    print("="*60)

    try:
        # 직접 파일 경로로 임포트
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)
        generate_labels = interval_profiles.generate_labels

        # 필수 컬럼이 없는 데이터
        df_missing = pd.DataFrame({
            'open': [1, 2, 3, 4, 5],
            'volume': [100, 200, 300, 400, 500]
        })

        # 각 인터벌별 테스트
        test_results = []

        for interval in ['1d', '30m', '240m', '15m']:
            try:
                result = generate_labels(df_missing, interval)
                print(f"  ❌ {interval}: ValueError가 발생해야 하는데 성공함")
                test_results.append(False)
            except ValueError as e:
                if '필수 컬럼 누락' in str(e):
                    print(f"  ✅ {interval}: 예상대로 필수 컬럼 누락 에러")
                    test_results.append(True)
                else:
                    print(f"  ⚠️ {interval}: ValueError지만 다른 메시지: {e}")
                    test_results.append(False)
            except Exception as e:
                print(f"  ❌ {interval}: 예상치 못한 에러: {e}")
                test_results.append(False)

        # close만 있는 데이터 (1d, 30m은 통과해야 함)
        df_close_only = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })

        print("\nclose 컬럼만 있는 경우:")
        for interval in ['1d', '30m']:
            try:
                result = generate_labels(df_close_only, interval)
                print(f"  ✅ {interval}: 성공 (close만 필요)")
                test_results.append(True)
            except Exception as e:
                print(f"  ❌ {interval}: 실패 (close만 있어도 되는데): {e}")
                test_results.append(False)

        # 240m, 15m은 high, low도 필요
        for interval in ['240m', '15m']:
            try:
                result = generate_labels(df_close_only, interval)
                print(f"  ❌ {interval}: high, low 없어도 성공함 (에러가 발생해야 함)")
                test_results.append(False)
            except ValueError as e:
                print(f"  ✅ {interval}: 예상대로 필수 컬럼 누락 에러")
                test_results.append(True)
            except Exception as e:
                print(f"  ⚠️ {interval}: 예상치 못한 에러: {e}")
                test_results.append(False)

        success_count = sum(test_results)
        total_count = len(test_results)
        print(f"\n결과: {success_count}/{total_count} 테스트 성공")
        return success_count == total_count

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_reward_type_validation():
    """4. 보상 계산 타입 검증 테스트"""
    print("\n" + "="*60)
    print("4. 보상 계산 타입 검증 테스트")
    print("="*60)

    try:
        # 직접 파일 경로로 임포트
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)
        calculate_reward = interval_profiles.calculate_reward

        test_results = []

        # dict가 아닌 타입 테스트
        invalid_types = [
            ("string", {}),
            (123, {}),
            ([1, 2, 3], {}),
            ({}, "string"),
            ({}, 456),
            ({}, [4, 5, 6])
        ]

        for pred, actual in invalid_types:
            try:
                result = calculate_reward('1d', pred, actual)
                print(f"  ❌ {type(pred).__name__}, {type(actual).__name__}: ValueError가 발생해야 함")
                test_results.append(False)
            except ValueError as e:
                if 'dict 타입이어야 합니다' in str(e):
                    print(f"  ✅ {type(pred).__name__}, {type(actual).__name__}: 예상대로 타입 에러")
                    test_results.append(True)
                else:
                    print(f"  ⚠️ ValueError지만 다른 메시지: {e}")
                    test_results.append(False)
            except Exception as e:
                print(f"  ❌ 예상치 못한 에러: {e}")
                test_results.append(False)

        # 정상 케이스
        valid_pred = {'direction': 1, 'return': 0.02, 'regime': 'bull'}
        valid_actual = {'direction': 1, 'return': 0.025, 'regime': 'bull'}

        try:
            result = calculate_reward('1d', valid_pred, valid_actual)
            print(f"\n  ✅ 정상 dict 입력: 보상 = {result:.3f}")
            test_results.append(True)
        except Exception as e:
            print(f"\n  ❌ 정상 dict 입력 실패: {e}")
            test_results.append(False)

        success_count = sum(test_results)
        total_count = len(test_results)
        print(f"\n결과: {success_count}/{total_count} 테스트 성공")
        return success_count == total_count

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_horizon_map_caching():
    """5. HORIZON_MAP 캐싱 테스트"""
    print("\n" + "="*60)
    print("5. HORIZON_MAP 캐싱 테스트")
    print("="*60)

    try:
        from labeling.chart_future_scanner import ChartFutureScanner

        scanner = ChartFutureScanner()

        # 첫 번째 호출
        horizon_map1 = scanner.HORIZON_MAP
        print(f"첫 번째 호출: {horizon_map1}")

        # 두 번째 호출 (캐시된 값 사용)
        horizon_map2 = scanner.HORIZON_MAP
        print(f"두 번째 호출: {horizon_map2}")

        # 같은 객체인지 확인 (캐싱 확인)
        if horizon_map1 is horizon_map2:
            print("  ✅ 캐싱 확인: 같은 객체 반환")
            cache_test = True
        else:
            print("  ❌ 캐싱 실패: 다른 객체 반환")
            cache_test = False

        # 캐시 속성 확인
        if hasattr(scanner, '_horizon_map_cache'):
            if scanner._horizon_map_cache is not None:
                print(f"  ✅ 캐시 저장 확인: {scanner._horizon_map_cache}")
            else:
                print("  ⚠️ 캐시가 None")
        else:
            print("  ❌ 캐시 속성 없음")

        return cache_test

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_exception_handling():
    """6. 예외 처리 세분화 테스트"""
    print("\n" + "="*60)
    print("6. 예외 처리 세분화 테스트")
    print("="*60)

    try:
        # 직접 파일 경로로 임포트
        spec = __import__('importlib.util').util.spec_from_file_location(
            "interval_profiles",
            "/workspace/rl_pipeline/core/interval_profiles.py"
        )
        interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(interval_profiles)
        generate_labels = interval_profiles.generate_labels
        calculate_reward = interval_profiles.calculate_reward

        test_results = []

        # ValueError 테스트
        try:
            generate_labels(pd.DataFrame(), 'invalid_interval')
        except ValueError as e:
            print(f"  ✅ ValueError 발생 (잘못된 인터벌): {str(e)[:50]}...")
            test_results.append(True)
        except Exception as e:
            print(f"  ❌ 다른 예외 발생: {type(e).__name__}")
            test_results.append(False)

        # TypeError 테스트 (간접적)
        try:
            calculate_reward('1d', None, {})
        except (ValueError, TypeError) as e:
            print(f"  ✅ 타입 관련 예외 발생: {type(e).__name__}")
            test_results.append(True)
        except Exception as e:
            print(f"  ❌ 예상치 못한 예외: {type(e).__name__}")
            test_results.append(False)

        success_count = sum(test_results)
        print(f"\n결과: {success_count}/{len(test_results)} 예외 처리 테스트 성공")
        return success_count == len(test_results)

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def main():
    """메인 테스트"""
    print("\n" + "="*70)
    print("ISSUES_FIXED_REPORT 개선사항 테스트")
    print("="*70)

    results = {}

    # 각 개선사항 테스트
    results['normalize_interval'] = test_normalize_interval()
    results['nan_handling'] = test_nan_handling()
    results['required_columns'] = test_required_columns()
    results['reward_type_validation'] = test_reward_type_validation()
    results['horizon_map_caching'] = test_horizon_map_caching()
    results['exception_handling'] = test_exception_handling()

    # 결과 요약
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)

    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")

    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    print(f"\n총 {success_count}/{total_count} 개선사항 검증 성공")

    if success_count == total_count:
        print("\n🎉 모든 개선사항이 정상적으로 작동합니다!")
    else:
        print("\n⚠️ 일부 개선사항에 문제가 있습니다.")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)