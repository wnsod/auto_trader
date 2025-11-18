"""
경험 리플레이 버퍼
PPO 학습을 위한 경험 저장 및 샘플링
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    경험 리플레이 버퍼
    
    PPO는 on-policy이므로 일반적으로 리플레이 버퍼를 사용하지 않지만,
    구현 선택 사항으로 제공
    """
    
    def __init__(self, capacity: int = 100000):
        """
        초기화
        
        Args:
            capacity: 버퍼 용량
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.size = 0
    
    def add(self, experience: Dict[str, Any]):
        """
        경험 추가
        
        Args:
            experience: {
                'state': np.ndarray,
                'action': int,
                'reward': float,
                'next_state': np.ndarray,
                'done': bool,
                'log_prob': float,  # 액션 선택 시 로그 확률
                'value': float  # 상태 가치
            }
        """
        self.buffer.append(experience)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        배치 샘플링
        
        Args:
            batch_size: 배치 크기
        
        Returns:
            경험 배치 리스트
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """버퍼 비우기"""
        self.buffer.clear()
        self.size = 0
    
    def __len__(self) -> int:
        return len(self.buffer)

