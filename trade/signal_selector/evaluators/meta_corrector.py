"""
Meta Corrector 모듈 - 메타-보정 시스템 (스태킹)
"""

from typing import Dict, Any


class MetaCorrector:
    """
    메타-보정 시스템 (스태킹)
    
    메타 모델을 사용하여 기본 점수를 보정하고 개선합니다.
    """
    def __init__(self):
        self.meta_weights: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}
        
    def calculate_meta_score(self, base_score: float, feedback_stats: Dict[str, Any], context_features: Dict[str, Any]) -> float:
        """
        메타 모델 기반 점수 보정
        
        Args:
            base_score: 기본 점수
            feedback_stats: 피드백 통계 딕셔너리
            context_features: 컨텍스트 특징 딕셔너리
        
        Returns:
            보정된 메타 점수 (-1.0 ~ 1.0)
        """
        try:
            # 간단한 선형 조합 (실제로는 XGBoost/LightGBM 사용)
            meta_score = base_score
            
            # 피드백 통계 가중치
            if 'success_rate' in feedback_stats:
                meta_score += feedback_stats['success_rate'] * 0.2
            
            if 'avg_profit' in feedback_stats:
                meta_score += feedback_stats['avg_profit'] * 0.1
            
            # 컨텍스트 특징 가중치
            if 'volatility' in context_features:
                volatility = context_features['volatility']
                if volatility == 'high':
                    meta_score *= 0.9  # 고변동성에서는 보수적
                elif volatility == 'low':
                    meta_score *= 1.1  # 저변동성에서는 공격적
            
            return max(-1.0, min(1.0, meta_score))
            
        except Exception as e:
            print(f"⚠️ 메타 점수 계산 오류: {e}")
            return base_score
    
    def update_meta_weights(self, performance_feedback: Dict[str, Any]) -> None:
        """
        메타 가중치 업데이트
        
        Args:
            performance_feedback: 성과 피드백 딕셔너리
        """
        try:
            # 성과 기반 가중치 조정
            if 'improvement' in performance_feedback:
                improvement = performance_feedback['improvement']
                
                # 긍정적 피드백이면 가중치 증가
                if improvement > 0:
                    for key in self.meta_weights:
                        self.meta_weights[key] *= 1.01
                else:
                    for key in self.meta_weights:
                        self.meta_weights[key] *= 0.99
                        
        except Exception as e:
            print(f"⚠️ 메타 가중치 업데이트 오류: {e}")

