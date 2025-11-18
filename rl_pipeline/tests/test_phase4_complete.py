#!/usr/bin/env python3
"""
Phase 4 완전한 테스트
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rl_pipeline.hybrid.ml_runner import (
    PredictionFeedbackRunner,
    WeightedEpisodeData,
    process_prediction_feedback,
    PREDICTION_ERROR_THRESHOLD_LOW,
    PREDICTION_ERROR_THRESHOLD_HIGH,
    PREDICTION_WEIGHT_HIGH,
    PREDICTION_WEIGHT_LOW,
    PREDICTION_WEIGHT_DEFAULT
)

logging.basicConfig(level=logging.WARNING)  # 경고만 출력
logger = logging.getLogger(__name__)

def test_all():
    print("=" * 60)
    print("Phase 4: ml_runner.py 완전한 테스트")
    print("=" * 60)
    
    all_passed = True
    
    # 테스트 1: 예측 오차 계산
    print("\n[테스트 1] 예측 오차 계산")
    try:
        runner = PredictionFeedbackRunner()
        error = runner.calculate_prediction_error(0.01, 0.012, 1, 1)
        if error < 0.01:
            print("  ✅ 통과")
        else:
            print(f"  ❌ 실패: error={error}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        all_passed = False
    
    # 테스트 2: 가중치 부여
    print("\n[테스트 2] 가중치 부여")
    try:
        runner = PredictionFeedbackRunner()
        episodes = [{'state': [1.0], 'action': 1, 'reward': 0.5}]
        errors = [PREDICTION_ERROR_THRESHOLD_LOW * 0.5]
        weighted = runner.apply_feedback_weights(episodes, errors)
        if weighted[0]['prediction_weight'] == PREDICTION_WEIGHT_HIGH:
            print("  ✅ 통과")
        else:
            print(f"  ❌ 실패: weight={weighted[0]['prediction_weight']}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        all_passed = False
    
    # 테스트 3: 학습 데이터 변환
    print("\n[테스트 3] 학습 데이터 변환")
    try:
        runner = PredictionFeedbackRunner()
        weighted_episodes = [
            {'state': [1.0, 2.0], 'action': 1, 'reward': 0.5, 
             'prediction_weight': 1.5, 'prediction_error': 0.001}
        ]
        training_data = runner.convert_to_training_data(weighted_episodes)
        if training_data and len(training_data.states) > 0:
            print("  ✅ 통과")
        else:
            print("  ❌ 실패: 데이터 변환 실패")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        all_passed = False
    
    # 테스트 4: 통합
    print("\n[테스트 4] 통합 테스트")
    try:
        online_results = {'status': 'success', 'segment_results': [[]]}
        predictions = [
            {
                'timestamp': 1000,
                'predicted_dir': 1,
                'predicted_target': 0.01,
                'predicted_horizon': 10,
                'predicted_conf': 0.8,
                'actual_dir': 1,
                'actual_move_pct': 0.012,
                'actual_horizon': 8
            }
        ]
        training_data = process_prediction_feedback(online_results, predictions)
        if training_data and len(training_data.states) > 0:
            print("  ✅ 통과")
        else:
            print("  ❌ 실패: 통합 처리 실패")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 결과
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Phase 4 모든 테스트 통과!")
        print("=" * 60)
        return True
    else:
        print("❌ 일부 테스트 실패")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)

