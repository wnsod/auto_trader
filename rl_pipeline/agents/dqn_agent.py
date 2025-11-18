"""
DQN (Deep Q-Network) Agent
진짜 강화학습 구현
"""

import jax
import jax.numpy as jnp
from jax import random, jit, grad
import flax
import flax.linen as nn
import optax
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    """DQN 설정"""
    state_dim: int = 20  # 상태 차원 (RSI, MACD 등)
    action_dim: int = 3  # 행동 개수 (HOLD, BUY, SELL)
    hidden_dims: List[int] = None  # 은닉층 차원
    learning_rate: float = 0.001
    gamma: float = 0.99  # 할인 계수
    epsilon_start: float = 1.0  # 탐험 시작 확률
    epsilon_end: float = 0.01  # 탐험 종료 확률
    epsilon_decay: float = 0.995  # 탐험 감소율
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_freq: int = 100  # 타겟 네트워크 업데이트 빈도

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class QNetwork(nn.Module):
    """Q-Network (가치 함수 근사)"""
    hidden_dims: List[int]
    action_dim: int
    training: bool = False  # 학습 모드 여부

    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        State → Q-values for each action

        Args:
            x: State vector [batch, state_dim]
            training: 학습 모드 (Dropout 활성화)
        Returns:
            Q-values [batch, action_dim]
        """
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            # Dropout은 학습 시에만
            if training:
                x = nn.Dropout(0.2)(x, deterministic=False)

        # 출력: 각 행동의 Q-value
        q_values = nn.Dense(self.action_dim)(x)
        return q_values


class DQNAgent:
    """DQN 에이전트 - 진짜 강화학습!"""

    def __init__(self, config: DQNConfig, seed: int = 42):
        self.config = config
        self.rng = random.PRNGKey(seed)

        # Q-Network 초기화
        self.q_network = QNetwork(
            hidden_dims=config.hidden_dims,
            action_dim=config.action_dim
        )

        # 더미 입력으로 파라미터 초기화
        dummy_state = jnp.zeros((1, config.state_dim))
        self.rng, init_rng = random.split(self.rng)
        self.params = self.q_network.init(init_rng, dummy_state)

        # Target Network (안정적인 학습을 위해)
        self.target_params = self.params

        # Optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # 탐험 확률
        self.epsilon = config.epsilon_start

        # 학습 스텝 카운터
        self.train_steps_count = 0

        logger.info(f"✅ DQN Agent 초기화 완료")
        logger.info(f"   State dim: {config.state_dim}")
        logger.info(f"   Action dim: {config.action_dim}")
        logger.info(f"   Hidden dims: {config.hidden_dims}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy 정책으로 행동 선택

        Args:
            state: 현재 상태 [state_dim]
            training: 학습 모드 여부
        Returns:
            action: 선택된 행동 (0=HOLD, 1=BUY, 2=SELL)
        """
        if training and np.random.random() < self.epsilon:
            # 탐험: 랜덤 행동
            return np.random.randint(0, self.config.action_dim)

        # 활용: Q-value가 가장 높은 행동
        state_batch = jnp.array([state])  # [1, state_dim]
        q_values = self.q_network.apply(self.params, state_batch, training=False)[0]  # [action_dim]

        return int(jnp.argmax(q_values))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        현재 상태의 Q-values 반환

        Args:
            state: 현재 상태 [state_dim]
        Returns:
            q_values: [action_dim]
        """
        state_batch = jnp.array([state])
        q_values = self.q_network.apply(self.params, state_batch, training=False)[0]
        return np.array(q_values)

    def _loss_fn(self, params, target_params, states, actions, rewards, next_states, dones, gamma, rng):
        """
        DQN Loss 계산 (Bellman Equation)

        Loss = (Q(s,a) - (r + γ * max_a' Q_target(s',a')))^2
        """
        # 현재 Q-values (학습 모드, RNG 전달)
        q_values = self.q_network.apply(params, states, training=True, rngs={'dropout': rng})  # [batch, action_dim]
        q_values_selected = jnp.take_along_axis(
            q_values,
            actions[:, None],
            axis=1
        ).squeeze()  # [batch]

        # 타겟 Q-values (다음 상태, 추론 모드)
        next_q_values = self.q_network.apply(target_params, next_states, training=False)  # [batch, action_dim]
        next_q_max = jnp.max(next_q_values, axis=1)  # [batch]

        # TD Target
        targets = rewards + gamma * next_q_max * (1 - dones)

        # MSE Loss
        loss = jnp.mean((q_values_selected - targets) ** 2)

        return loss

    def train_step(self, batch: Dict[str, np.ndarray]) -> float:
        """
        한 번의 학습 스텝

        Args:
            batch: {
                'states': [batch, state_dim],
                'actions': [batch],
                'rewards': [batch],
                'next_states': [batch, state_dim],
                'dones': [batch]
            }
        Returns:
            loss: 학습 손실
        """
        # NumPy → JAX
        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'], dtype=jnp.int32)
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'], dtype=jnp.float32)

        # RNG 생성 (Dropout용)
        self.rng, dropout_rng = random.split(self.rng)

        # Gradient 계산
        loss, grads = jax.value_and_grad(self._loss_fn)(
            self.params,
            self.target_params,
            states,
            actions,
            rewards,
            next_states,
            dones,
            self.config.gamma,
            dropout_rng
        )

        # Optimizer 업데이트
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        # Epsilon 감소 (탐험 줄이기)
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

        # Target Network 업데이트
        self.train_steps_count += 1
        if self.train_steps_count % self.config.target_update_freq == 0:
            self.target_params = self.params
            logger.debug(f"🔄 Target network 업데이트 (step {self.train_steps_count})")

        return float(loss)

    def save(self, path: str):
        """정책 저장"""
        import pickle
        save_dict = {
            'params': self.params,
            'target_params': self.target_params,
            'opt_state': self.opt_state,
            'epsilon': self.epsilon,
            'train_steps_count': self.train_steps_count,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"✅ DQN Agent 저장: {path}")

    def load(self, path: str):
        """정책 로드"""
        import pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.params = save_dict['params']
        self.target_params = save_dict['target_params']
        self.opt_state = save_dict['opt_state']
        self.epsilon = save_dict['epsilon']
        self.train_steps_count = save_dict.get('train_steps_count', save_dict.get('train_step', 0))
        logger.info(f"✅ DQN Agent 로드: {path}")
        logger.info(f"   Train step: {self.train_steps_count}, Epsilon: {self.epsilon:.4f}")


if __name__ == "__main__":
    # 간단한 테스트
    logging.basicConfig(level=logging.INFO)

    config = DQNConfig(state_dim=20, action_dim=3)
    agent = DQNAgent(config)

    # 더미 상태로 테스트
    state = np.random.randn(20)
    action = agent.select_action(state, training=True)
    q_values = agent.get_q_values(state)

    print(f"State: {state[:5]}...")
    print(f"Action: {action} (0=HOLD, 1=BUY, 2=SELL)")
    print(f"Q-values: {q_values}")

    # 학습 테스트
    batch = {
        'states': np.random.randn(64, 20),
        'actions': np.random.randint(0, 3, 64),
        'rewards': np.random.randn(64),
        'next_states': np.random.randn(64, 20),
        'dones': np.random.randint(0, 2, 64)
    }

    loss = agent.train_step(batch)
    print(f"Training loss: {loss:.4f}")
    print("✅ DQN Agent 테스트 완료!")
