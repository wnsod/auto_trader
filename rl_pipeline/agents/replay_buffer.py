"""
Experience Replay Buffer
강화학습의 경험을 저장하고 샘플링
"""

import numpy as np
from typing import Dict, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity: int = 10000, state_dim: int = 20):
        """
        Args:
            capacity: 버퍼 최대 크기
            state_dim: 상태 차원
        """
        self.capacity = capacity
        self.state_dim = state_dim

        # 버퍼 (deque로 자동 오버플로우 처리)
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

        self.size = 0
        logger.info(f"✅ Replay Buffer 초기화 (capacity={capacity})")

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        경험 추가 (s, a, r, s', done)

        Args:
            state: 현재 상태 [state_dim]
            action: 선택한 행동 (0, 1, 2)
            reward: 보상
            next_state: 다음 상태 [state_dim]
            done: 에피소드 종료 여부
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        랜덤 배치 샘플링

        Args:
            batch_size: 배치 크기
        Returns:
            batch: {
                'states': [batch, state_dim],
                'actions': [batch],
                'rewards': [batch],
                'next_states': [batch, state_dim],
                'dones': [batch]
            }
        """
        if self.size < batch_size:
            raise ValueError(f"버퍼 크기({self.size})가 배치 크기({batch_size})보다 작습니다")

        # 랜덤 인덱스 선택
        indices = np.random.choice(self.size, batch_size, replace=False)

        # 배치 구성
        batch = {
            'states': np.array([self.states[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_states': np.array([self.next_states[i] for i in indices]),
            'dones': np.array([self.dones[i] for i in indices])
        }

        return batch

    def __len__(self):
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """배치 샘플링 가능 여부"""
        return self.size >= batch_size


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    buffer = ReplayBuffer(capacity=1000, state_dim=10)

    # 더미 경험 추가
    for i in range(500):
        state = np.random.randn(10)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = (i % 100 == 99)  # 100 스텝마다 종료

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # 배치 샘플링
    batch = buffer.sample(batch_size=64)
    print(f"Batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    print("✅ Replay Buffer 테스트 완료!")
