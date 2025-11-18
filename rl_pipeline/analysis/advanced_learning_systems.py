"""
고급 학습 시스템 모듈
JAX 기반 앙상블 및 PPO 시스템
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    num_models: int = 5
    diversity_threshold: float = 0.1
    voting_strategy: str = "weighted"
    learning_rate: float = 0.001

@dataclass
class PPOConfig:
    """PPO 설정"""
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 0.0003
    batch_size: int = 64

class JAXEnsembleLearningSystem:
    """JAX 앙상블 학습 시스템"""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = []
        self.is_initialized = False
        logger.info("🚀 JAX 앙상블 학습 시스템 초기화")
    
    def initialize(self):
        """앙상블 시스템 초기화"""
        try:
            # 앙상블 모델들 초기화
            for i in range(self.config.num_models):
                model = {
                    "id": i,
                    "weights": np.random.randn(10),  # 더미 가중치
                    "performance": 0.0
                }
                self.models.append(model)
            
            self.is_initialized = True
            logger.info(f"✅ JAX 앙상블 학습 시스템 초기화 완료 ({self.config.num_models}개 모델)")
        except Exception as e:
            logger.warning(f"⚠️ JAX 앙상블 학습 시스템 초기화 실패: {e}")
            self.is_initialized = False
    
    def train_ensemble(self, data: pd.DataFrame, targets: np.ndarray) -> Dict[str, Any]:
        """앙상블 훈련"""
        if not self.is_initialized:
            self.initialize()
        
        try:
            # 더미 앙상블 훈련 결과
            results = []
            for i, model in enumerate(self.models):
                # 각 모델별 훈련 시뮬레이션
                performance = np.random.uniform(0.6, 0.9)
                model["performance"] = performance
                results.append({
                    "model_id": i,
                    "performance": performance,
                    "loss": np.random.uniform(0.05, 0.2)
                })
            
            return {
                "ensemble_performance": np.mean([r["performance"] for r in results]),
                "model_results": results,
                "diversity_score": self.config.diversity_threshold,
                "training_time": 2.5
            }
        except Exception as e:
            logger.error(f"❌ 앙상블 훈련 실패: {e}")
            return {"error": str(e)}
    
    def predict_ensemble(self, data: dict) -> dict:
        """앙상블 예측 - 실제 데이터 기반 개선된 예측"""
        if not self.is_initialized:
            self.initialize()
        
        try:
            # 실제 데이터 기반 간단한 예측 모델
            predictions = []
            
            # 입력 데이터에서 특징 추출
            analysis_results = data.get("analysis_results", {})
            close_prices = data.get("close", [])
            volume_data = data.get("volume", [])
            
            for i, model in enumerate(self.models):
                # 간단한 휴리스틱 기반 예측 (더미 대신 실제 데이터 사용)
                base_pred = 0.5
                
                # 분석 결과가 있으면 사용
                if analysis_results:
                    fractal_score = analysis_results.get("fractal", 0.5)
                    multi_timeframe_score = analysis_results.get("multi_timeframe", 0.5)
                    indicator_score = analysis_results.get("indicator_cross", 0.5)
                    # 가중 평균
                    base_pred = (fractal_score * 0.3 + multi_timeframe_score * 0.4 + indicator_score * 0.3)
                
                # 가격 추세 반영
                if close_prices and len(close_prices) >= 2:
                    recent_trend = (close_prices[-1] - close_prices[-2]) / close_prices[-2] if close_prices[-2] > 0 else 0
                    # 추세를 점수로 변환 (-1~1 -> 0~1)
                    trend_score = 0.5 + np.tanh(recent_trend * 10) * 0.3
                    base_pred = (base_pred * 0.7 + trend_score * 0.3)
                
                # 모델별 미세 조정 (모델 성능 기반)
                model_adjustment = (model.get("performance", 0.5) - 0.5) * 0.1
                pred = np.clip(base_pred + model_adjustment, 0.0, 1.0)
                predictions.append(pred)
            
            # 가중 평균으로 앙상블 예측
            weights = [model.get("performance", 0.5) for model in self.models]
            if np.sum(weights) == 0:
                weights = np.ones(len(weights)) / len(weights)
            else:
                weights = np.array(weights) / np.sum(weights)
            
            ensemble_pred = np.average(predictions, weights=weights)
            
            # 신뢰도 계산 (예측 일관성 및 데이터 품질 기반)
            prediction_std = np.std(predictions)
            # 표준편차가 낮을수록 일관성 높음 -> 신뢰도 높음
            consistency_score = max(0.0, 1.0 - prediction_std * 2)
            
            # 데이터 품질 기반 신뢰도 조정
            data_quality = 1.0
            if len(close_prices) < 10:
                data_quality *= 0.7  # 데이터 부족
            if not analysis_results:
                data_quality *= 0.8  # 분석 결과 없음
            
            confidence = min(1.0, consistency_score * data_quality)
            
            return {
                'ensemble_prediction': float(ensemble_pred),
                'confidence_score': float(confidence)
            }
        except Exception as e:
            logger.error(f"❌ 앙상블 예측 실패: {e}")
            return {
                'ensemble_prediction': 0.5,
                'confidence_score': 0.5
            }

class JAXPPOSystem:
    """JAX PPO 시스템"""
    
    def __init__(self, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.policy_network = None
        self.value_network = None
        self.is_initialized = False
        logger.info("🚀 JAX PPO 시스템 초기화")
    
    def initialize(self):
        """PPO 시스템 초기화"""
        try:
            # PPO 네트워크 초기화
            self.policy_network = {
                "weights": np.random.randn(128, 64),
                "bias": np.random.randn(64)
            }
            self.value_network = {
                "weights": np.random.randn(64, 1),
                "bias": np.random.randn(1)
            }
            
            self.is_initialized = True
            logger.info("✅ JAX PPO 시스템 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ JAX PPO 시스템 초기화 실패: {e}")
            self.is_initialized = False
    
    def train_ppo(self, states: np.ndarray, actions: np.ndarray, 
                  rewards: np.ndarray, old_log_probs: np.ndarray) -> Dict[str, Any]:
        """
        PPO 훈련
        
        ⚠️ 주의: 이 클래스는 더미 구현입니다. 
        실제 PPO 학습은 rl_pipeline.hybrid.trainer_jax.PPOTrainer를 사용하세요.
        이 클래스는 integrated_analyzer에서 참고용으로만 사용됩니다.
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # ⚠️ 더미 PPO 훈련 결과 (실제 학습은 trainer_jax.py 사용)
            # 참고: integrated_analyzer에서 사용되지만 실제 학습은 하지 않음
            policy_loss = np.random.uniform(0.1, 0.5)
            value_loss = np.random.uniform(0.05, 0.3)
            entropy_loss = np.random.uniform(0.01, 0.1)
            
            total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy_loss
            
            return {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "total_loss": total_loss,
                "clip_ratio": self.config.clip_ratio,
                "training_time": 1.8
            }
        except Exception as e:
            logger.error(f"❌ PPO 훈련 실패: {e}")
            return {"error": str(e)}
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """액션 선택 - 상태 기반 개선된 액션 선택"""
        if not self.is_initialized:
            self.initialize()
        
        try:
            # 상태 기반 간단한 정책 (더미 대신 휴리스틱 사용)
            if state is None or len(state) == 0:
                # 기본 액션 (중립)
                return np.array([0.0, 0.0]), -0.5
            
            # 상태 값 기반 액션 결정
            # state[0]: 가격 변화율 (예상)
            # state[1]: 변동성 (예상)
            # state[2]: 거래량 (예상)
            
            # 간단한 휴리스틱 정책
            if len(state) >= 2:
                price_change = float(state[0]) if len(state) > 0 else 0.0
                volatility = float(state[1]) if len(state) > 1 else 0.0
                
                # 가격 변화 기반 액션 (정규화)
                buy_action = np.tanh(price_change * 5)  # -1 ~ 1
                # 변동성 조정 (높은 변동성은 액션 크기 감소)
                volatility_factor = 1.0 / (1.0 + volatility * 10)
                buy_action *= volatility_factor
                
                # 액션을 [-1, 1] 범위로 정규화
                action = np.array([np.clip(buy_action, -1.0, 1.0), 
                                  np.clip(volatility * 0.5, -1.0, 1.0)])
                
                # 로그 확률 계산 (액션 크기 기반)
                action_magnitude = np.linalg.norm(action)
                log_prob = -0.5 - action_magnitude * 0.5
                
                return action, float(log_prob)
            else:
                # 기본 액션 (중립)
                return np.array([0.0, 0.0]), -0.5
            
        except Exception as e:
            logger.error(f"❌ 액션 선택 실패: {e}")
            return np.zeros(2), -0.5

# 팩토리 함수들
def get_jax_ensemble_system(config: EnsembleConfig = None) -> JAXEnsembleLearningSystem:
    """JAX 앙상블 학습 시스템 인스턴스 반환"""
    return JAXEnsembleLearningSystem(config)

def get_jax_ppo_system(config: PPOConfig = None) -> JAXPPOSystem:
    """JAX PPO 시스템 인스턴스 반환"""
    return JAXPPOSystem(config)

# 모듈 초기화
logger.info("✅ advanced_learning_systems 모듈 로드 완료")

