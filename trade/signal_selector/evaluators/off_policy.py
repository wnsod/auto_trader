"""
Off-Policy Evaluator 모듈 - 오프폴리시 평가 시스템
"""

from typing import Dict, List


class OffPolicyEvaluator:
    """
    오프폴리시 평가 시스템 (IPS/Doubly Robust)
    
    Inverse Propensity Scoring (IPS)와 Doubly Robust 추정을 사용하여
    오프폴리시 평가를 수행합니다.
    """
    def __init__(self):
        self.policy_probabilities: Dict[str, float] = {}
        self.evaluation_history: List[Dict] = []
        
    def record_policy_probability(self, action: str, probability: float, context: str) -> None:
        """
        정책 확률 기록
        
        Args:
            action: 액션 문자열
            probability: 정책 확률
            context: 컨텍스트 문자열
        """
        try:
            key = f"{context}_{action}"
            self.policy_probabilities[key] = probability
        except Exception as e:
            print(f"⚠️ 정책 확률 기록 오류: {e}")
    
    def calculate_ips_estimate(self, action: str, reward: float, context: str) -> float:
        """
        Inverse Propensity Scoring 추정
        
        Args:
            action: 액션 문자열
            reward: 보상 값
            context: 컨텍스트 문자열
        
        Returns:
            IPS 추정값
        """
        try:
            key = f"{context}_{action}"
            propensity = self.policy_probabilities.get(key, 0.5)  # 기본값 0.5
            
            if propensity > 0:
                return reward / propensity
            else:
                return reward
                
        except Exception as e:
            print(f"⚠️ IPS 추정 오류: {e}")
            return reward
    
    def calculate_doubly_robust_estimate(self, action: str, reward: float, context: str, baseline_reward: float) -> float:
        """
        Doubly Robust 추정
        
        Args:
            action: 액션 문자열
            reward: 보상 값
            context: 컨텍스트 문자열
            baseline_reward: 베이스라인 보상 값
        
        Returns:
            Doubly Robust 추정값
        """
        try:
            ips_estimate = self.calculate_ips_estimate(action, reward, context)
            key = f"{context}_{action}"
            propensity = self.policy_probabilities.get(key, 0.5)
            
            # Doubly Robust 공식
            dr_estimate = baseline_reward + (reward - baseline_reward) / propensity
            return dr_estimate
            
        except Exception as e:
            print(f"⚠️ Doubly Robust 추정 오류: {e}")
            return reward

