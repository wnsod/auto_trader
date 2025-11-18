"""
리팩토링 테스트 스크립트
absolute_zero_system과 orchestrator의 리팩토링된 모듈 테스트
"""

import sys
import os
import logging
from pathlib import Path

# 경로 설정
sys.path.insert(0, os.path.dirname(__file__))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_absolute_zero_modules():
    """absolute_zero 패키지 모듈 테스트"""
    print("\n" + "=" * 60)
    print("[TEST] Absolute Zero 모듈 테스트 시작")
    print("=" * 60)

    try:
        # 1. 설정 모듈 테스트
        print("\n[1] 설정 모듈 (az_config) 테스트...")
        from absolute_zero.az_config import (
            configure_logging,
            ensure_storage_ready,
            AZ_DEBUG,
            STRATEGIES_DB_PATH,
            MIN_CANDLES_PER_INTERVAL
        )
        print(f"   [OK] configure_logging: {configure_logging.__name__}")
        print(f"   [OK] AZ_DEBUG: {AZ_DEBUG}")
        print(f"   [OK] STRATEGIES_DB_PATH: {STRATEGIES_DB_PATH}")
        print(f"   [OK] MIN_CANDLES_PER_INTERVAL: {list(MIN_CANDLES_PER_INTERVAL.keys())}")

        # 2. 유틸리티 모듈 테스트
        print("\n2️⃣ 유틸리티 모듈 (az_utils) 테스트...")
        from absolute_zero.az_utils import (
            sort_intervals,
            format_time_duration,
            validate_environment
        )

        # 인터벌 정렬 테스트
        intervals = ['1d', '15m', '240m', '30m']
        sorted_intervals = sort_intervals(intervals)
        print(f"   ✅ sort_intervals: {intervals} → {sorted_intervals}")

        # 환경 검증
        env_valid = validate_environment()
        print(f"   ✅ validate_environment: {env_valid}")

        # 3. 분석 모듈 테스트
        print("\n3️⃣ 분석 모듈 (az_analysis) 테스트...")
        from absolute_zero.az_analysis import (
            calculate_fractal_score,
            validate_strategy_quality,
            analyze_strategy_distribution
        )

        # 샘플 전략으로 테스트
        sample_strategies = [
            {'win_rate': 0.6, 'profit_factor': 1.2, 'sharpe_ratio': 1.5, 'trades': 100},
            {'win_rate': 0.55, 'profit_factor': 1.1, 'sharpe_ratio': 1.2, 'trades': 80},
            {'win_rate': 0.65, 'profit_factor': 1.3, 'sharpe_ratio': 1.7, 'trades': 120}
        ]

        fractal = calculate_fractal_score(sample_strategies)
        print(f"   ✅ calculate_fractal_score: {fractal:.3f}")

        quality = validate_strategy_quality(sample_strategies[0])
        print(f"   ✅ validate_strategy_quality: {quality}")

        # 4. 글로벌 전략 모듈 테스트
        print("\n4️⃣ 글로벌 전략 모듈 (az_global_strategies) 테스트...")
        from absolute_zero.az_global_strategies import generate_global_strategies_only
        print(f"   ✅ generate_global_strategies_only: {generate_global_strategies_only.__name__}")

        # 5. 메인 모듈 테스트
        print("\n5️⃣ 메인 모듈 (az_main) 테스트...")
        from absolute_zero.az_main import run_absolute_zero, main
        print(f"   ✅ run_absolute_zero: {run_absolute_zero.__name__}")
        print(f"   ✅ main: {main.__name__}")

        print("\n✅ Absolute Zero 모듈 테스트 성공!")
        return True

    except Exception as e:
        print(f"\n❌ Absolute Zero 모듈 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_modules():
    """orchestrator 리팩토링 모듈 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Orchestrator 모듈 테스트 시작")
    print("=" * 60)

    try:
        # 1. 검증 모듈 테스트
        print("\n1️⃣ 검증 모듈 (validators) 테스트...")
        from pipelines.orchestrator_refactored.validators import (
            validate_selfplay_result,
            validate_global_strategy_pool,
            validate_global_strategy_quality
        )

        # 샘플 self-play 결과로 테스트
        sample_result = {
            'cycle_results': [{'episode': 1, 'accuracy': 0.75}],
            'episodes': 1,
            'avg_accuracy': 0.75,
            'best_accuracy': 0.75,
            'strategy_count': 50
        }

        validation = validate_selfplay_result(sample_result, 'BTC', '15m')
        print(f"   ✅ validate_selfplay_result: valid={validation['valid']}, issues={len(validation['issues'])}")

        # 2. 데이터 모델 테스트
        print("\n2️⃣ 데이터 모델 (models) 테스트...")
        from pipelines.orchestrator_refactored.models import (
            PipelineResult,
            SelfPlayConfig,
            StrategyPoolConfig,
            ValidationResult
        )

        # PipelineResult 테스트
        result = PipelineResult(coin='BTC', interval='15m')
        result.update_status('running')
        print(f"   ✅ PipelineResult: coin={result.coin}, status={result.status}")

        # SelfPlayConfig 테스트
        sp_config = SelfPlayConfig.from_env()
        print(f"   ✅ SelfPlayConfig: episodes={sp_config.episodes}, early_stop={sp_config.early_stop}")

        # ValidationResult 테스트
        val_result = ValidationResult(valid=True)
        val_result.add_warning("테스트 경고")
        print(f"   ✅ ValidationResult: valid={val_result.valid}, warnings={len(val_result.warnings)}")

        print("\n✅ Orchestrator 모듈 테스트 성공!")
        return True

    except Exception as e:
        print(f"\n❌ Orchestrator 모듈 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_refactored_import():
    """리팩토링된 absolute_zero_system_refactored.py 테스트"""
    print("\n" + "=" * 60)
    print("🧪 리팩토링된 absolute_zero_system 테스트")
    print("=" * 60)

    try:
        # 리팩토링된 파일 import
        from absolute_zero_system_refactored import (
            run_absolute_zero,
            generate_global_strategies_only,
            calculate_global_analysis_data
        )

        print("   ✅ run_absolute_zero 함수 import 성공")
        print("   ✅ generate_global_strategies_only 함수 import 성공")
        print("   ✅ calculate_global_analysis_data 함수 import 성공")

        print("\n✅ 리팩토링된 시스템 import 테스트 성공!")
        return True

    except Exception as e:
        print(f"\n❌ 리팩토링된 시스템 import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 70)
    print("🚀 리팩토링 테스트 시작")
    print("=" * 70)

    results = {
        'absolute_zero_modules': False,
        'orchestrator_modules': False,
        'refactored_import': False
    }

    # 각 테스트 실행
    results['absolute_zero_modules'] = test_absolute_zero_modules()
    results['orchestrator_modules'] = test_orchestrator_modules()
    results['refactored_import'] = test_refactored_import()

    # 최종 결과 출력
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 모든 테스트 통과! 리팩토링 성공!")
    else:
        print("⚠️ 일부 테스트 실패. 코드 확인 필요.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)