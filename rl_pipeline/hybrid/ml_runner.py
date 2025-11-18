"""
예측 피드백 모듈 (Phase 4)
온라인 Self-Play 결과를 재학습 데이터로 변환하고 가중치 부여

기능:
1. 온라인 Self-Play 결과에서 예측 검증
2. 예측 오차 계산
3. 예측 오차 기반 가중치 부여
4. auto_trainer.py에 전달할 데이터 준비
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# 환경변수
PREDICTION_ERROR_THRESHOLD_LOW = float(os.getenv('PREDICTION_ERROR_THRESHOLD_LOW', '0.005'))  # 0.5%
PREDICTION_ERROR_THRESHOLD_HIGH = float(os.getenv('PREDICTION_ERROR_THRESHOLD_HIGH', '0.02'))  # 2.0%
PREDICTION_WEIGHT_HIGH = float(os.getenv('PREDICTION_WEIGHT_HIGH', '1.5'))  # 높은 가중치
PREDICTION_WEIGHT_LOW = float(os.getenv('PREDICTION_WEIGHT_LOW', '0.3'))  # 낮은 가중치
PREDICTION_WEIGHT_DEFAULT = float(os.getenv('PREDICTION_WEIGHT_DEFAULT', '1.0'))  # 기본 가중치


@dataclass
class WeightedEpisodeData:
    """ml_runner → auto_trainer 전달 데이터"""
    states: np.ndarray  # 시장 상태 벡터
    actions: np.ndarray  # 행동 (BUY/SELL/HOLD)
    rewards: np.ndarray  # 보상
    weights: np.ndarray  # 예측 정확도 기반 가중치
    prediction_errors: np.ndarray  # 예측 오차 (메타데이터)


@dataclass
class PredictionRecord:
    """예측 기록"""
    timestamp: int
    predicted_dir: int  # +1/-1/0
    predicted_target: float  # 목표 변동률
    predicted_horizon: int  # 목표 캔들 수
    predicted_conf: float  # 예측 확신도
    actual_dir: Optional[int] = None  # 실제 방향
    actual_move_pct: Optional[float] = None  # 실제 변동률
    actual_horizon: Optional[int] = None  # 실제 도달 캔들 수
    prediction_error: Optional[float] = None  # 예측 오차


class PredictionFeedbackRunner:
    """예측 피드백 러너"""
    
    def __init__(self):
        """초기화"""
        self.prediction_records: List[PredictionRecord] = []
        self.validation_results: List[Dict[str, Any]] = []
        
        logger.info("✅ Prediction Feedback Runner 초기화 완료")
    
    def validate_predictions(
        self,
        online_results: Dict[str, Any],
        predictions: List[Dict[str, Any]]
    ) -> List[PredictionRecord]:
        """
        온라인 Self-Play에서 예측 검증
        
        Args:
            online_results: 온라인 Self-Play 결과 (세그먼트 결과 포함)
            predictions: 예측 기록 리스트
        
        Returns:
            검증된 예측 기록 리스트
        """
        try:
            validated_records = []
            segment_results = online_results.get('segment_results', [])
            
            # 세그먼트 결과에서 실제 결과 추출
            actual_results = {}
            
            for segment_list in segment_results:
                if isinstance(segment_list, list):
                    for segment in segment_list:
                        # 세그먼트 정보에서 실제 결과 추출
                        # 실제 구현 시 세그먼트 결과 구조에 맞게 수정 필요
                        strategy_id = getattr(segment, 'strategy_id', None)
                        if strategy_id:
                            actual_results[strategy_id] = {
                                'profit': getattr(segment, 'profit', 0.0),
                                'trades_count': getattr(segment, 'trades_count', 0)
                            }
            
            # 예측과 실제 결과 매칭 및 검증
            for pred in predictions:
                record = PredictionRecord(
                    timestamp=pred.get('timestamp', 0),
                    predicted_dir=pred.get('predicted_dir', 0),
                    predicted_target=pred.get('predicted_target', 0.0),
                    predicted_horizon=pred.get('predicted_horizon', 0),
                    predicted_conf=pred.get('predicted_conf', 0.5)
                )
                
                # 실제 결과 매칭 (예측 오차 계산을 위해)
                # 실제 구현 시 timestamp나 인덱스 기반 매칭
                if 'actual_dir' in pred:
                    record.actual_dir = pred['actual_dir']
                    record.actual_move_pct = pred.get('actual_move_pct', 0.0)
                    record.actual_horizon = pred.get('actual_horizon', 0)
                    
                    # 예측 오차 계산
                    record.prediction_error = self.calculate_prediction_error(
                        predicted_target=record.predicted_target,
                        actual_move_pct=record.actual_move_pct or 0.0,
                        predicted_dir=record.predicted_dir,
                        actual_dir=record.actual_dir or 0
                    )
                
                validated_records.append(record)
            
            self.prediction_records.extend(validated_records)
            logger.info(f"✅ {len(validated_records)}개 예측 검증 완료")
            
            return validated_records
            
        except Exception as e:
            logger.error(f"❌ 예측 검증 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def calculate_prediction_errors(
        self,
        predictions: List[PredictionRecord],
        actuals: List[Dict[str, Any]]
    ) -> List[float]:
        """
        예측 오차 계산
        
        Args:
            predictions: 예측 기록 리스트
            actuals: 실제 결과 리스트
        
        Returns:
            예측 오차 리스트 (절대값, 퍼센트)
        """
        try:
            errors = []
            
            for pred, actual in zip(predictions, actuals):
                predicted_target = pred.predicted_target
                actual_move_pct = actual.get('actual_move_pct', 0.0)
                predicted_dir = pred.predicted_dir
                actual_dir = actual.get('actual_dir', 0)
                
                error = self.calculate_prediction_error(
                    predicted_target, actual_move_pct, predicted_dir, actual_dir
                )
                
                errors.append(error)
            
            logger.debug(f"✅ {len(errors)}개 예측 오차 계산 완료")
            return errors
            
        except Exception as e:
            logger.error(f"❌ 예측 오차 계산 실패: {e}")
            return []
    
    def calculate_prediction_error(
        self,
        predicted_target: float,
        actual_move_pct: float,
        predicted_dir: int,
        actual_dir: int
    ) -> float:
        """
        단일 예측 오차 계산
        
        Args:
            predicted_target: 예측 목표 변동률
            actual_move_pct: 실제 변동률
            predicted_dir: 예측 방향 (+1/-1/0)
            actual_dir: 실제 방향 (+1/-1/0)
        
        Returns:
            예측 오차 (절대값, 퍼센트)
        """
        try:
            # 방향 오차 확인
            if predicted_dir != actual_dir:
                # 방향이 틀리면 큰 오차
                return abs(actual_move_pct) + abs(predicted_target) + 0.05  # 추가 페널티
            
            # 방향이 맞으면 크기 오차만 계산
            error = abs(predicted_target - actual_move_pct)
            
            return error
            
        except Exception as e:
            logger.error(f"❌ 예측 오차 계산 실패: {e}")
            return 1.0  # 기본값 (큰 오차)
    
    def apply_feedback_weights(
        self,
        episodes_data: List[Dict[str, Any]],
        prediction_errors: List[float]
    ) -> List[Dict[str, Any]]:
        """
        가중치 부여
        
        Args:
            episodes_data: 에피소드 데이터 리스트
            prediction_errors: 예측 오차 리스트
        
        Returns:
            가중치가 부여된 에피소드 데이터 리스트
        """
        try:
            weighted_episodes = []
            
            for episode, error in zip(episodes_data, prediction_errors):
                # 예측 오차에 따른 가중치 계산
                if error < PREDICTION_ERROR_THRESHOLD_LOW:
                    # 예측이 정확함 → 높은 가중치
                    weight = PREDICTION_WEIGHT_HIGH
                elif error > PREDICTION_ERROR_THRESHOLD_HIGH:
                    # 예측이 부정확함 → 낮은 가중치
                    weight = PREDICTION_WEIGHT_LOW
                else:
                    # 중간 → 기본 가중치
                    weight = PREDICTION_WEIGHT_DEFAULT
                
                # 에피소드에 가중치 추가
                weighted_episode = episode.copy()
                weighted_episode['prediction_weight'] = weight
                weighted_episode['prediction_error'] = error
                
                weighted_episodes.append(weighted_episode)
            
            logger.info(f"✅ {len(weighted_episodes)}개 에피소드에 가중치 부여 완료")
            logger.debug(f"  가중치 분포: HIGH={sum(1 for e in weighted_episodes if e['prediction_weight'] == PREDICTION_WEIGHT_HIGH)}, "
                        f"DEFAULT={sum(1 for e in weighted_episodes if e['prediction_weight'] == PREDICTION_WEIGHT_DEFAULT)}, "
                        f"LOW={sum(1 for e in weighted_episodes if e['prediction_weight'] == PREDICTION_WEIGHT_LOW)}")
            
            return weighted_episodes
            
        except Exception as e:
            logger.error(f"❌ 가중치 부여 실패: {e}")
            return episodes_data  # 실패 시 원본 반환
    
    def convert_to_training_data(
        self,
        weighted_episodes: List[Dict[str, Any]]
    ) -> Optional[WeightedEpisodeData]:
        """
        auto_trainer에 전달할 데이터 준비
        
        Args:
            weighted_episodes: 가중치가 부여된 에피소드 데이터 리스트
        
        Returns:
            WeightedEpisodeData 객체 (None이면 변환 실패)
        """
        try:
            if not weighted_episodes:
                logger.warning("⚠️ 변환할 에피소드 데이터가 없습니다")
                return None
            
            # 상태, 행동, 보상, 가중치, 예측 오차 추출
            states_list = []
            actions_list = []
            rewards_list = []
            weights_list = []
            errors_list = []
            
            for episode in weighted_episodes:
                # 상태 벡터 추출 (에피소드 구조에 맞게 수정 필요)
                state = episode.get('state', [])
                if isinstance(state, (list, np.ndarray)):
                    states_list.append(state)
                else:
                    # 기본 상태 벡터 생성
                    states_list.append(np.array([0.0] * 10))  # 기본값
                
                # 행동 추출
                action = episode.get('action', 0)
                if isinstance(action, (int, float)):
                    actions_list.append(int(action))
                else:
                    actions_list.append(0)  # 기본값
                
                # 보상 추출
                reward = episode.get('reward', 0.0)
                rewards_list.append(float(reward))
                
                # 가중치 추출
                weight = episode.get('prediction_weight', PREDICTION_WEIGHT_DEFAULT)
                weights_list.append(float(weight))
                
                # 예측 오차 추출
                error = episode.get('prediction_error', 0.0)
                errors_list.append(float(error))
            
            # NumPy 배열로 변환
            states = np.array(states_list) if states_list else np.array([])
            actions = np.array(actions_list) if actions_list else np.array([])
            rewards = np.array(rewards_list) if rewards_list else np.array([])
            weights = np.array(weights_list) if weights_list else np.array([])
            errors = np.array(errors_list) if errors_list else np.array([])
            
            logger.info(f"✅ 학습 데이터 변환 완료: {len(states)}개 샘플")
            
            return WeightedEpisodeData(
                states=states,
                actions=actions,
                rewards=rewards,
                weights=weights,
                prediction_errors=errors
            )
            
        except Exception as e:
            logger.error(f"❌ 학습 데이터 변환 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_online_selfplay_results(
        self,
        online_results: Dict[str, Any],
        predictions: List[Dict[str, Any]]
    ) -> Optional[WeightedEpisodeData]:
        """
        온라인 Self-Play 결과를 처리하여 학습 데이터 생성 (통합 함수)
        
        Args:
            online_results: 온라인 Self-Play 결과
            predictions: 예측 기록 리스트
        
        Returns:
            WeightedEpisodeData 객체 (None이면 처리 실패)
        """
        try:
            # 1. 예측 검증
            validated = self.validate_predictions(online_results, predictions)
            
            if not validated:
                logger.warning("⚠️ 검증된 예측이 없습니다")
                return None
            
            # 2. 예측 오차 계산 (이미 validate_predictions에서 계산됨)
            errors = [r.prediction_error or 1.0 for r in validated]
            
            # 3. 에피소드 데이터 준비 (예측 기록을 에피소드 형식으로 변환)
            episodes_data = []
            for record in validated:
                episode = {
                    'state': self._record_to_state(record),
                    'action': record.predicted_dir,
                    'reward': 0.0,  # 실제 보상은 online_results에서 추출 가능
                    'timestamp': record.timestamp
                }
                episodes_data.append(episode)
            
            # 4. 가중치 부여
            weighted_episodes = self.apply_feedback_weights(episodes_data, errors)
            
            # 5. 학습 데이터 변환
            training_data = self.convert_to_training_data(weighted_episodes)
            
            return training_data
            
        except Exception as e:
            logger.error(f"❌ 온라인 Self-Play 결과 처리 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _record_to_state(self, record: PredictionRecord) -> np.ndarray:
        """예측 기록을 상태 벡터로 변환"""
        try:
            # 기본 상태 벡터 생성 (실제 구현 시 더 풍부한 정보 포함)
            state = np.array([
                float(record.predicted_dir),
                float(record.predicted_target),
                float(record.predicted_horizon),
                float(record.predicted_conf),
                float(record.actual_dir or 0),
                float(record.actual_move_pct or 0.0),
                float(record.actual_horizon or 0),
                float(record.prediction_error or 0.0),
                0.0,  # 예약
                0.0   # 예약
            ])
            return state
            
        except Exception as e:
            logger.error(f"❌ 상태 벡터 변환 실패: {e}")
            return np.array([0.0] * 10)


def process_prediction_feedback(
    online_results: Dict[str, Any],
    predictions: List[Dict[str, Any]]
) -> Optional[WeightedEpisodeData]:
    """
    예측 피드백 처리 (편의 함수)
    
    Args:
        online_results: 온라인 Self-Play 결과
        predictions: 예측 기록 리스트
    
    Returns:
        WeightedEpisodeData 객체
    """
    runner = PredictionFeedbackRunner()
    return runner.process_online_selfplay_results(online_results, predictions)

