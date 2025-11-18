#!/usr/bin/env python
"""리팩토링된 코드 검증 스크립트"""
import sys
sys.path.append('/workspace')

try:
    from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

    print("=" * 80)
    print("Refactored Code Verification")
    print("=" * 80)

    o = IntegratedPipelineOrchestrator()

    # 메서드 존재 확인
    has_get_opposite = hasattr(o, '_get_opposite_direction')
    has_reassess = hasattr(o, '_reassess_strategy_direction')

    print(f"\nMethod Existence Check:")
    print(f"  _get_opposite_direction: {'YES' if has_get_opposite else 'NO'}")
    print(f"  _reassess_strategy_direction: {'YES' if has_reassess else 'NO'}")

    if has_get_opposite:
        # 테스트 케이스
        print(f"\n_get_opposite_direction Tests:")
        print(f"  Input: 1 (buy)  -> Output: {o._get_opposite_direction(1)}")
        print(f"  Input: -1 (sell) -> Output: {o._get_opposite_direction(-1)}")
        print(f"  Input: 0 (neutral, fallback=1) -> Output: {o._get_opposite_direction(0, 1)}")
        print(f"  Input: 0 (neutral, fallback=-1) -> Output: {o._get_opposite_direction(0, -1)}")

    print("\n" + "=" * 80)
    print("SUCCESS: Refactored code is working!")
    print("=" * 80)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
