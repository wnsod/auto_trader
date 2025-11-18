"""
Confidence Calibrator 모듈 - 신뢰도 캘리브레이션 시스템
"""

import math
from typing import Dict, List


class ConfidenceCalibrator:
    """
    신뢰도 캘리브레이션 시스템 (Platt/Isotonic)
    
    Platt Scaling을 사용하여 신뢰도를 캘리브레이션합니다.
    """
    def __init__(self):
        self.calibration_params: Dict[str, Dict[str, float]] = {}
        self.calibration_history: List[Dict] = []
        
    def calibrate_confidence(self, raw_confidence: float, context: str) -> float:
        """
        신뢰도 캘리브레이션 (Platt Scaling)
        
        Args:
            raw_confidence: 원본 신뢰도 값
            context: 컨텍스트 문자열
        
        Returns:
            캘리브레이션된 신뢰도 값 (0.0 ~ 1.0)
        """
        try:
            # 간단한 로지스틱 변환
            if context not in self.calibration_params:
                self.calibration_params[context] = {'a': 1.0, 'b': 0.0}
            
            params = self.calibration_params[context]
            calibrated = 1.0 / (1.0 + math.exp(-(params['a'] * raw_confidence + params['b'])))
            
            return max(0.0, min(1.0, calibrated))
            
        except Exception as e:
            print(f"⚠️ 신뢰도 캘리브레이션 오류: {e}")
            return raw_confidence
    
    def update_calibration_params(self, context: str, actual_success_rate: float, predicted_confidence: float) -> None:
        """
        캘리브레이션 파라미터 업데이트
        
        Args:
            context: 컨텍스트 문자열
            actual_success_rate: 실제 성공률
            predicted_confidence: 예측된 신뢰도
        """
        try:
            if context not in self.calibration_params:
                self.calibration_params[context] = {'a': 1.0, 'b': 0.0}
            
            # 간단한 적응적 업데이트
            params = self.calibration_params[context]
            error = actual_success_rate - predicted_confidence
            
            # 파라미터 조정
            params['a'] += error * 0.1
            params['b'] += error * 0.05
            
            # 범위 제한
            params['a'] = max(0.1, min(5.0, params['a']))
            params['b'] = max(-2.0, min(2.0, params['b']))
            
        except Exception as e:
            print(f"⚠️ 캘리브레이션 파라미터 업데이트 오류: {e}")

