"""
과적합 방지 모듈 (Phase 5)
학습/검증 데이터 분할 및 조기 종료

기능:
1. 데이터 분할 (train/validation/test)
2. 검증 성과 하락 감지
3. 조기 종료
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 환경변수
TRAIN_SPLIT_RATIO = float(os.getenv('TRAIN_SPLIT_RATIO', '0.8'))  # 80%
VALIDATION_SPLIT_RATIO = float(os.getenv('VALIDATION_SPLIT_RATIO', '0.1'))  # 10%
TEST_SPLIT_RATIO = float(os.getenv('TEST_SPLIT_RATIO', '0.1'))  # 10%
EARLY_STOP_ENABLED = os.getenv('EARLY_STOP_ENABLED', 'true').lower() == 'true'
EARLY_STOP_PATIENCE = int(os.getenv('EARLY_STOP_PATIENCE', '3'))  # 3회 연속 하락 시 종료


class OverfittingPrevention:
    """과적합 방지 시스템"""
    
    def __init__(self):
        """초기화"""
        self.validation_history: List[float] = []
        self.stagnation_count = 0
        
        logger.info("✅ Overfitting Prevention 초기화 완료")
    
    def split_data(
        self,
        total_size: int,
        train_ratio: float = TRAIN_SPLIT_RATIO,
        validation_ratio: float = VALIDATION_SPLIT_RATIO
    ) -> Tuple[int, int, int]:
        """
        데이터를 학습/검증/테스트로 분할
        
        Args:
            total_size: 전체 데이터 크기
            train_ratio: 학습 데이터 비율
            validation_ratio: 검증 데이터 비율
        
        Returns:
            (train_end, validation_end, test_end) 인덱스
        """
        try:
            train_end = int(total_size * train_ratio)
            validation_end = int(total_size * (train_ratio + validation_ratio))
            test_end = total_size
            
            logger.info(f"✅ 데이터 분할: train=[0~{train_end}], "
                       f"validation=[{train_end}~{validation_end}], "
                       f"test=[{validation_end}~{test_end}]")
            
            return train_end, validation_end, test_end
            
        except Exception as e:
            logger.error(f"❌ 데이터 분할 실패: {e}")
            # 기본값: 전체를 학습으로
            return total_size, total_size, total_size
    
    def check_validation_performance(
        self,
        current_performance: float,
        threshold_improvement: float = 0.0
    ) -> Tuple[bool, bool]:
        """
        검증 성과 확인 및 조기 종료 판단
        
        Args:
            current_performance: 현재 검증 성과
            threshold_improvement: 개선 임계값
        
        Returns:
            (should_early_stop, is_improving) 튜플
        """
        try:
            if not EARLY_STOP_ENABLED:
                return False, True
            
            # 이전 성과와 비교
            if not self.validation_history:
                self.validation_history.append(current_performance)
                return False, True
            
            previous_performance = self.validation_history[-1]
            improvement = current_performance - previous_performance
            
            is_improving = improvement >= threshold_improvement
            
            if is_improving:
                # 개선됨
                self.stagnation_count = 0
                self.validation_history.append(current_performance)
                return False, True
            else:
                # 하락 또는 정체
                self.stagnation_count += 1
                self.validation_history.append(current_performance)
                
                should_stop = self.stagnation_count >= EARLY_STOP_PATIENCE
                
                if should_stop:
                    logger.warning(f"⚠️ 조기 종료 조건 충족: {self.stagnation_count}회 연속 하락/정체")
                
                return should_stop, False
                
        except Exception as e:
            logger.error(f"❌ 검증 성과 확인 실패: {e}")
            return False, True
    
    def reset(self):
        """상태 초기화"""
        self.validation_history = []
        self.stagnation_count = 0
        logger.debug("✅ 과적합 방지 상태 초기화")

