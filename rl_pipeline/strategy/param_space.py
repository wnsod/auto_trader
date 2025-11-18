"""
Parameter Space 모듈 - 전략 파라미터 샘플링을 위한 함수들
"""

import random
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

def sample_param_grid(param_ranges: Dict[str, Tuple[float, float]], n_samples: int = 10) -> List[Dict[str, float]]:
    """
    파라미터 그리드에서 샘플을 생성하는 함수 (더미 구현)
    
    Args:
        param_ranges: 파라미터 이름과 (min, max) 범위의 딕셔너리
        n_samples: 생성할 샘플 수
        
    Returns:
        파라미터 샘플 리스트
    """
    logger.debug(f"📊 파라미터 그리드 샘플링: {len(param_ranges)} 파라미터, {n_samples} 샘플")
    
    samples = []
    for _ in range(n_samples):
        sample = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            sample[param_name] = random.uniform(min_val, max_val)
        samples.append(sample)
    
    return samples

