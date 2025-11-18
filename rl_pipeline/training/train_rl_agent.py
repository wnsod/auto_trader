"""
DQN Agent 학습 루프
진짜 강화학습 트레이닝!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import numpy as np
import pandas as pd
from typing import Dict, List

from rl_pipeline.agents.dqn_agent import DQNAgent, DQNConfig
from rl_pipeline.agents.replay_buffer import ReplayBuffer
from rl_pipeline.simulation.rl_environment import TradingEnvironment

logger = logging.getLogger(__name__)


def train_dqn_agent(
    candle_data: pd.DataFrame,
    num_episodes: int = 100,
    state_dim: int = 20,
    save_path: str = "models/dqn_agent.pkl",
    log_interval: int = 10
) -> Dict:
    """
    DQN Agent 학습

    Args:
        candle_data: 캔들 데이터 (RSI, MACD 등 포함)
        num_episodes: 학습 에피소드 수
        state_dim: 상태 차원
        save_path: 모델 저장 경로
        log_interval: 로그 출력 간격

    Returns:
        학습 결과 통계
    """
    # DQN Agent 생성
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=3,  # HOLD, BUY, SELL
        hidden_dims=[128, 64, 32],
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=10000
    )

    agent = DQNAgent(config)
    buffer = ReplayBuffer(capacity=config.buffer_size, state_dim=state_dim)

    logger.info("🚀 DQN 학습 시작")
    logger.info(f"   에피소드: {num_episodes}")
    logger.info(f"   캔들 데이터: {len(candle_data)}개")

    # 학습 통계
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    episode_win_rates = []
    losses = []

    # 학습 루프
    for episode in range(num_episodes):
        env = TradingEnvironment(candle_data, state_dim=state_dim)
        state = env.reset()

        episode_reward = 0.0
        episode_loss = []
        done = False

        # 한 에피소드 실행
        while not done:
            # 행동 선택 (Epsilon-greedy)
            action = agent.select_action(state, training=True)

            # 환경에서 스텝 실행
            next_state, reward, done, info = env.step(action)

            # Experience Replay Buffer에 저장
            buffer.add(state, action, reward, next_state, done)

            episode_reward += reward

            # 학습 (버퍼가 충분히 차면)
            if buffer.is_ready(config.batch_size):
                batch = buffer.sample(config.batch_size)
                loss = agent.train_step(batch)
                episode_loss.append(loss)

            state = next_state

        # 에피소드 통계
        stats = env.get_episode_stats()
        episode_rewards.append(episode_reward)
        episode_returns.append(stats['total_return'])
        episode_trades.append(stats['total_trades'])
        episode_win_rates.append(stats['win_rate'])

        if episode_loss:
            losses.append(np.mean(episode_loss))

        # 로깅
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_return = np.mean(episode_returns[-log_interval:])
            avg_win_rate = np.mean(episode_win_rates[-log_interval:])
            avg_loss = np.mean(losses[-log_interval:]) if losses else 0

            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"  Avg Reward: {avg_reward:.2f}")
            logger.info(f"  Avg Return: {avg_return:.2%}")
            logger.info(f"  Avg Win Rate: {avg_win_rate:.2%}")
            logger.info(f"  Avg Loss: {avg_loss:.4f}")
            logger.info(f"  Epsilon: {agent.epsilon:.4f}")
            logger.info(f"  Buffer Size: {len(buffer)}")

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    logger.info("✅ DQN 학습 완료!")
    logger.info(f"   최종 Epsilon: {agent.epsilon:.4f}")
    logger.info(f"   총 학습 스텝: {agent.train_steps_count}")
    logger.info(f"   모델 저장: {save_path}")

    # 결과 요약
    results = {
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_win_rates': episode_win_rates,
        'losses': losses,
        'final_epsilon': agent.epsilon,
        'total_train_steps': agent.train_steps_count
    }

    return results


def load_candle_data_from_db(coin: str, interval: str, limit: int = 5000) -> pd.DataFrame:
    """
    DB에서 캔들 데이터 로드

    Args:
        coin: 코인 (예: "ADA")
        interval: 인터벌 (예: "15m")
        limit: 최대 캔들 수

    Returns:
        캔들 데이터 DataFrame
    """
    import sqlite3
    from rl_pipeline.core.env import config

    with sqlite3.connect(config.RL_DB) as conn:
        query = f"""
            SELECT
                timestamp, open, high, low, close, volume,
                rsi, macd, macd_signal, mfi, atr, adx,
                bb_upper, bb_middle, bb_lower,
                volume_ratio, regime_label, regime_stage
            FROM candles
            WHERE coin = ? AND interval = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(coin, interval, limit))

    # 역순 정렬 (오래된 것부터)
    df = df.iloc[::-1].reset_index(drop=True)

    # 결측치 처리
    df['rsi'] = df['rsi'].fillna(50.0)
    df['macd'] = df['macd'].fillna(0.0)
    df['macd_signal'] = df['macd_signal'].fillna(0.0)
    df['mfi'] = df['mfi'].fillna(50.0)
    df['atr'] = df['atr'].fillna(0.02)
    df['adx'] = df['adx'].fillna(25.0)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    df['regime_stage'] = df['regime_stage'].fillna(3)

    # BB 계산 (없으면)
    if df['bb_upper'].isna().any():
        df['bb_middle'] = df['close']
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98

    logger.info(f"✅ 캔들 데이터 로드: {coin}-{interval} ({len(df)}개)")

    return df


if __name__ == "__main__":
    print("사용법: python rl_pipeline/training/train_all_dqn_agents.py")
