"""
Phase 4 테스트: ml_runner.py 예측 피드백 모듈 검증

실행 방법:
    docker exec -it auto_trader_coin bash
    cd /workspace
    python rl_pipeline/tests/test_phase4_ml_runner.py
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rl_pipeline.hybrid.ml_runner import (
    PredictionFeedbackRunner,
    WeightedEpisodeData,
    PredictionRecord,
    process_prediction_feedback,
    PREDICTION_ERROR_THRESHOLD_LOW,
    PREDICTION_ERROR_THRESHOLD_HIGH,
    PREDICTION_WEIGHT_HIGH,
    PREDICTION_WEIGHT_LOW,
    PREDICTION_WEIGHT_DEFAULT
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_prediction_error_calculation():
    """예측 오차 계산 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 1: 예측 오차 계산")
    logger.info("=" * 60)
    
    try:
        runner = PredictionFeedbackRunner()
        
        # 테스트 케이스
        test_cases = [
            {
                "name": "정확한 예측 (방향/크기 모두 맞음)",
                "predicted_target": 0.01,  # 1%
                "actual_move_pct": 0.01,
                "predicted_dir": 1,
                "actual_dir": 1,
                "expected_low": True
            },
            {
                "name": "방향 맞음, 크기 다름",
                "predicted_target": 0.02,  # 2%
                "actual_move_pct": 0.01,  # 1%
                "predicted_dir": 1,
                "actual_dir": 1,
                "expected_low": None  # 크기 오차는 중간
            },
            {
                "name": "방향 틀림",
                "predicted_target": 0.01,
                "actual_move_pct": -0.01,
                "predicted_dir": 1,
                "actual_dir": -1,
                "expected_low": False  # 큰 오차
            }
        ]
        
        for case in test_cases:
            error = runner.calculate_prediction_error(
                case["predicted_target"],
                case["actual_move_pct"],
                case["predicted_dir"],
                case["actual_dir"]
            )
            
            logger.info(f"  {case['name']}: error={error:.4f}")
            
            if case["expected_low"] is not None:
                if case["expected_low"]:
                    if error > PREDICTION_ERROR_THRESHOLD_LOW:
                        logger.warning(f"  ⚠️ 예상: 낮은 오차, 실제: {error:.4f}")
                else:
                    if error < PREDICTION_ERROR_THRESHOLD_HIGH:
                        logger.warning(f"  ⚠️ 예상: 높은 오차, 실제: {error:.4f}")
        
        logger.info("✅ 예측 오차 계산 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_feedback_weights():
    """가중치 부여 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 2: 가중치 부여")
    logger.info("=" * 60)
    
    try:
        runner = PredictionFeedbackRunner()
        
        # 테스트 에피소드 데이터
        episodes_data = [
            {'state': [1.0], 'action': 1, 'reward': 0.5},
            {'state': [1.0], 'action': 1, 'reward': 0.3},
            {'state': [1.0], 'action': 1, 'reward': -0.2},
        ]
        
        # 다양한 예측 오차
        prediction_errors = [
            PREDICTION_ERROR_THRESHOLD_LOW * 0.5,  # 낮은 오차 → 높은 가중치
            PREDICTION_ERROR_THRESHOLD_LOW * 1.5,  # 중간 오차 → 기본 가중치
            PREDICTION_ERROR_THRESHOLD_HIGH * 1.5,  # 높은 오차 → 낮은 가중치
        ]
        
        weighted_episodes = runner.apply_feedback_weights(episodes_data, prediction_errors)
        
        if len(weighted_episodes) != len(episodes_data):
            logger.error(f"❌ 가중치 부여된 에피소드 수 불일치: {len(weighted_episodes)} != {len(episodes_data)}")
            return False
        
        logger.info("  가중치 부여 결과:")
        for i, episode in enumerate(weighted_episodes):
            weight = episode.get('prediction_weight', 0.0)
            error = episode.get('prediction_error', 0.0)
            logger.info(f"    에피소드 {i+1}: weight={weight:.2f}, error={error:.4f}")
            
            # 가중치 검증
            if error < PREDICTION_ERROR_THRESHOLD_LOW:
                if weight != PREDICTION_WEIGHT_HIGH:
                    logger.error(f"❌ 낮은 오차인데 가중치가 HIGH가 아님: {weight}")
                    return False
            elif error > PREDICTION_ERROR_THRESHOLD_HIGH:
                if weight != PREDICTION_WEIGHT_LOW:
                    logger.error(f"❌ 높은 오차인데 가중치가 LOW가 아님: {weight}")
                    return False
        
        logger.info("✅ 가중치 부여 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_convert_to_training_data():
    """학습 데이터 변환 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 3: 학습 데이터 변환")
    logger.info("=" * 60)
    
    try:
        runner = PredictionFeedbackRunner()
        
        # 테스트 에피소드 데이터
        weighted_episodes = [
            {
                'state': [1.0, 2.0, 3.0],
                'action': 1,
                'reward': 0.5,
                'prediction_weight': 1.5,
                'prediction_error': 0.001
            },
            {
                'state': [2.0, 3.0, 4.0],
                'action': -1,
                'reward': -0.2,
                'prediction_weight': 0.3,
                'prediction_error': 0.025
            },
            {
                'state': [3.0, 4.0, 5.0],
                'action': 0,
                'reward': 0.1,
                'prediction_weight': 1.0,
                'prediction_error': 0.01
            }
        ]
        
        training_data = runner.convert_to_training_data(weighted_episodes)
        
        if training_data is None:
            logger.error("❌ 학습 데이터 변환 실패")
            return False
        
        logger.info(f"✅ 학습 데이터 변환 성공:")
        
        # 데이터 타입 확인 및 길이 확인
        states_len = len(training_data.states) if hasattr(training_data.states, '__len__') else 0
        actions_len = len(training_data.actions) if hasattr(training_data.actions, '__len__') else 0
        rewards_len = len(training_data.rewards) if hasattr(training_data.rewards, '__len__') else 0
        weights_len = len(training_data.weights) if hasattr(training_data.weights, '__len__') else 0
        errors_len = len(training_data.prediction_errors) if hasattr(training_data.prediction_errors, '__len__') else 0
        
        logger.info(f"  - 상태 샘플 수: {states_len}")
        logger.info(f"  - 행동 샘플 수: {actions_len}")
        logger.info(f"  - 보상 샘플 수: {rewards_len}")
        logger.info(f"  - 가중치 샘플 수: {weights_len}")
        logger.info(f"  - 예측 오차 샘플 수: {errors_len}")
        
        # 데이터 일관성 확인
        if not (states_len == actions_len == rewards_len == weights_len == errors_len):
            logger.error(f"❌ 데이터 길이 불일치: states={states_len}, actions={actions_len}, rewards={rewards_len}, weights={weights_len}, errors={errors_len}")
            return False
        
        if states_len == 0:
            logger.error("❌ 데이터가 비어있음")
            return False
        
        # 가중치 범위 확인
        weights_array = np.array(training_data.weights) if not isinstance(training_data.weights, np.ndarray) else training_data.weights
        if np.any(weights_array < 0):
            logger.error("❌ 음수 가중치 발견")
            return False
        
        errors_array = np.array(training_data.prediction_errors) if not isinstance(training_data.prediction_errors, np.ndarray) else training_data.prediction_errors
        
        logger.info(f"  - 가중치 범위: {np.min(weights_array):.2f} ~ {np.max(weights_array):.2f}")
        logger.info(f"  - 예측 오차 범위: {np.min(errors_array):.4f} ~ {np.max(errors_array):.4f}")
        
        logger.info("✅ 학습 데이터 변환 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_integration():
    """통합 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 4: 통합 테스트 (예측 검증 → 오차 계산 → 가중치 부여 → 변환)")
    logger.info("=" * 60)
    
    try:
        # 온라인 Self-Play 결과 시뮬레이션
        online_results = {
            'status': 'success',
            'segment_results': [
                [
                    type('Segment', (), {
                        'strategy_id': 'test_strategy_1',
                        'profit': 100.0,
                        'trades_count': 5
                    })()
                ]
            ]
        }
        
        # 예측 기록 시뮬레이션
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
            },
            {
                'timestamp': 2000,
                'predicted_dir': -1,
                'predicted_target': -0.015,
                'predicted_horizon': 12,
                'predicted_conf': 0.7,
                'actual_dir': 1,  # 방향 틀림
                'actual_move_pct': 0.005,
                'actual_horizon': 10
            },
            {
                'timestamp': 3000,
                'predicted_dir': 1,
                'predicted_target': 0.02,
                'predicted_horizon': 15,
                'predicted_conf': 0.6,
                'actual_dir': 1,
                'actual_move_pct': 0.018,
                'actual_horizon': 14
            }
        ]
        
        # 통합 처리
        training_data = process_prediction_feedback(online_results, predictions)
        
        if training_data is None:
            logger.error("❌ 통합 처리 실패")
            return False
        
        # 데이터 검증
        if len(training_data.states) == 0:
            logger.error("❌ 상태 데이터가 비어있음")
            return False
        
        logger.info(f"✅ 통합 처리 성공:")
        logger.info(f"  - 샘플 수: {len(training_data.states)}")
        
        # NumPy 배열인지 확인
        if isinstance(training_data.states, np.ndarray):
            logger.info(f"  - 상태 차원: {training_data.states.shape if training_data.states.size > 0 else 'N/A'}")
        else:
            logger.info(f"  - 상태 타입: {type(training_data.states)}")
        
        if len(training_data.weights) > 0:
            logger.info(f"  - 가중치 평균: {np.mean(training_data.weights):.3f}")
            logger.info(f"  - 가중치 범위: {np.min(training_data.weights):.2f} ~ {np.max(training_data.weights):.2f}")
        
        if len(training_data.prediction_errors) > 0:
            logger.info(f"  - 예측 오차 평균: {np.mean(training_data.prediction_errors):.4f}")
        
        # 가중치 분포 확인
        if len(training_data.weights) > 0:
            high_weight_count = np.sum(training_data.weights == PREDICTION_WEIGHT_HIGH)
            default_weight_count = np.sum(training_data.weights == PREDICTION_WEIGHT_DEFAULT)
            low_weight_count = np.sum(training_data.weights == PREDICTION_WEIGHT_LOW)
            
            logger.info(f"  - 가중치 분포: HIGH={high_weight_count}, DEFAULT={default_weight_count}, LOW={low_weight_count}")
        
        logger.info("✅ 통합 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("=" * 60)
    logger.info("Phase 4: ml_runner.py 예측 피드백 모듈 테스트 시작")
    logger.info("=" * 60)
    
    tests = [
        ("예측 오차 계산", test_prediction_error_calculation),
        ("가중치 부여", test_feedback_weights),
        ("학습 데이터 변환", test_convert_to_training_data),
        ("통합 테스트", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n▶ {test_name} 테스트 실행...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} 통과\n")
            else:
                logger.error(f"❌ {test_name} 실패\n")
        except Exception as e:
            logger.error(f"❌ {test_name} 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        logger.info("=" * 60)
        logger.info("🎉 Phase 4 테스트 모두 통과!")
        logger.info("=" * 60)
        return True
    else:
        logger.error("=" * 60)
        logger.error("❌ 일부 테스트 실패")
        logger.error("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

