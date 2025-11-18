#!/usr/bin/env python3
"""간단한 Phase 4 테스트"""

import sys
import os
from pathlib import Path

workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

print("=" * 60)
print("Phase 4 간단 테스트")
print("=" * 60)

try:
    from rl_pipeline.hybrid.ml_runner import (
        PredictionFeedbackRunner,
        PREDICTION_ERROR_THRESHOLD_LOW,
        PREDICTION_WEIGHT_HIGH
    )
    print("✅ Import 성공")
    
    runner = PredictionFeedbackRunner()
    print("✅ PredictionFeedbackRunner 생성 성공")
    
    # 예측 오차 계산 테스트
    error = runner.calculate_prediction_error(
        predicted_target=0.01,
        actual_move_pct=0.012,
        predicted_dir=1,
        actual_dir=1
    )
    print(f"✅ 예측 오차 계산: {error:.4f}")
    
    # 가중치 테스트
    episodes = [{'state': [1.0], 'action': 1, 'reward': 0.5}]
    errors = [PREDICTION_ERROR_THRESHOLD_LOW * 0.5]
    weighted = runner.apply_feedback_weights(episodes, errors)
    print(f"✅ 가중치 부여: {weighted[0].get('prediction_weight', 'N/A')}")
    
    print("=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

