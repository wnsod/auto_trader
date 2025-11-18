#!/usr/bin/env python
"""빠른 SOL 테스트 - 1 에피소드만 실행"""
import sys
sys.path.append('/workspace')

import os

# 환경 변수 설정 (최소 에피소드로)
os.environ['PREDICTIVE_SELFPLAY_EPISODES'] = '1'  # 1 에피소드만
os.environ['PREDICTIVE_SELFPLAY_MIN_EPISODES'] = '1'

print("=" * 80)
print("빠른 SOL 테스트 (1 에피소드)")
print("=" * 80)
print()

from rl_pipeline.pipelines.orchestrator import Orchestrator

# Orchestrator 초기화
orch = Orchestrator(session_id="test_sol_quick")

# SOL 코인으로 학습 실행 (15m 인터벌만)
try:
    results = orch.run_predictive_selfplay_for_coin(
        coin="SOL",
        intervals=["15m"]
    )

    print()
    print("=" * 80)
    if results and results.get('success'):
        print("✅ 테스트 성공!")
        print(f"  - 코인: SOL")
        print(f"  - 인터벌: 15m")
        print(f"  - 평균 정확도: {results.get('avg_accuracy', 0)*100:.1f}%")
        print(f"  - 에피소드 수: {results.get('total_episodes', 0)}")
    else:
        print("⚠️ 테스트 실패")
        print(f"  - 에러: {results.get('error', '알 수 없음')}")
    print("=" * 80)
except Exception as e:
    print(f"❌ 테스트 실행 중 오류: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
