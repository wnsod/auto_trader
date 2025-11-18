"""
강화학습 환경 (Trading Environment)
시장 시뮬레이션 + Reward 계산
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLState:
    """RL 상태"""
    # 시장 지표
    rsi: float
    macd: float
    macd_signal: float
    mfi: float
    atr: float
    adx: float
    volume_ratio: float

    # 볼린저 밴드
    bb_upper: float
    bb_middle: float
    bb_lower: float

    # 레짐 정보
    regime_stage: int  # 0-6
    regime_confidence: float

    # 포지션 상태
    position_pnl: float  # 현재 포지션 수익률
    holding_time: int  # 보유 시간 (캔들 개수)

    # 가격 변화
    price_change_1: float  # 1캔들 전 대비 변화율
    price_change_5: float  # 5캔들 전 대비 변화율

    def to_vector(self) -> np.ndarray:
        """상태를 벡터로 변환 (Neural Network 입력용)"""
        return np.array([
            self.rsi / 100.0,  # 0-1로 정규화
            self.macd,
            self.macd_signal,
            self.mfi / 100.0,
            self.atr,
            self.adx / 100.0,
            self.volume_ratio,
            self.bb_upper,
            self.bb_middle,
            self.bb_lower,
            self.regime_stage / 6.0,  # 0-1로 정규화
            self.regime_confidence,
            self.position_pnl,
            self.holding_time / 100.0,  # 정규화
            self.price_change_1,
            self.price_change_5,
            # 파생 피처
            1.0 if self.position_pnl > 0 else 0.0,  # 수익 중 여부
            1.0 if self.macd > self.macd_signal else 0.0,  # MACD 크로스
            (self.bb_upper - self.bb_lower) / self.bb_middle,  # BB 폭
            1.0 if self.regime_stage >= 4 else 0.0,  # 상승장 여부
        ])


class TradingEnvironment:
    """거래 환경 (RL Environment)"""

    def __init__(
        self,
        candle_data: pd.DataFrame,
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,  # 0.1% 수수료
        state_dim: int = 20
    ):
        """
        Args:
            candle_data: 캔들 데이터 (RSI, MACD 등 포함)
            initial_balance: 초기 자본
            trading_fee: 거래 수수료
            state_dim: 상태 벡터 차원
        """
        self.candle_data = candle_data
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.state_dim = state_dim

        # 현재 상태
        self.current_step = 0
        self.balance = initial_balance
        self.position = None  # {'entry_price': ..., 'size': ..., 'entry_step': ...}
        self.total_trades = 0
        self.winning_trades = 0

        # 에피소드 기록
        self.episode_reward = 0.0
        self.episode_trades = []

        self.max_steps = len(candle_data) - 1

        logger.debug(f"Trading Environment 초기화 (steps={self.max_steps})")

    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.total_trades = 0
        self.winning_trades = 0
        self.episode_reward = 0.0
        self.episode_trades = []

        return self._get_state().to_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        행동 실행

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
        Returns:
            next_state: 다음 상태 [state_dim]
            reward: 보상
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        current_candle = self.candle_data.iloc[self.current_step]
        price = current_candle['close']

        reward = 0.0
        info = {'action': action, 'price': price}

        # 행동 실행
        if action == 1 and self.position is None:  # BUY
            # 매수
            size = self.balance * 0.95  # 95% 투자 (5% 여유)
            cost = size * (1 + self.trading_fee)

            if cost <= self.balance:
                self.position = {
                    'entry_price': price,
                    'size': size / price,  # 코인 개수
                    'entry_step': self.current_step
                }
                self.balance -= cost
                self.total_trades += 1
                info['trade'] = 'BUY'

        elif action == 2 and self.position is not None:  # SELL
            # 매도
            sell_value = self.position['size'] * price
            net_value = sell_value * (1 - self.trading_fee)

            # 수익률 계산
            profit = net_value - (self.position['size'] * self.position['entry_price'])
            profit_pct = profit / (self.position['size'] * self.position['entry_price'])

            self.balance += net_value

            # 보상 계산
            reward = profit_pct * 100.0  # 수익률을 보상으로

            # 승리/패배 기록
            if profit > 0:
                self.winning_trades += 1

            # 거래 기록
            self.episode_trades.append({
                'entry_price': self.position['entry_price'],
                'exit_price': price,
                'profit_pct': profit_pct,
                'holding_time': self.current_step - self.position['entry_step']
            })

            self.position = None
            info['trade'] = 'SELL'
            info['profit_pct'] = profit_pct

        # HOLD 페널티 (너무 오래 보유 시)
        if self.position is not None:
            holding_time = self.current_step - self.position['entry_step']
            if holding_time > 50:  # 50 캔들 이상 보유
                reward -= 0.01  # 작은 페널티

        # 다음 스텝으로
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        # 에피소드 종료 시 미청산 포지션 강제 청산
        if done and self.position is not None:
            price = self.candle_data.iloc[self.current_step - 1]['close']
            sell_value = self.position['size'] * price
            net_value = sell_value * (1 - self.trading_fee)
            profit = net_value - (self.position['size'] * self.position['entry_price'])
            profit_pct = profit / (self.position['size'] * self.position['entry_price'])

            self.balance += net_value
            reward += profit_pct * 100.0
            self.position = None

        self.episode_reward += reward

        next_state = self._get_state().to_vector()

        return next_state, reward, done, info

    def _get_state(self) -> RLState:
        """현재 상태 생성"""
        if self.current_step >= len(self.candle_data):
            self.current_step = len(self.candle_data) - 1

        candle = self.candle_data.iloc[self.current_step]

        # 포지션 정보
        if self.position is not None:
            current_price = candle['close']
            position_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
            holding_time = self.current_step - self.position['entry_step']
        else:
            position_pnl = 0.0
            holding_time = 0

        # 가격 변화
        price_change_1 = 0.0
        price_change_5 = 0.0
        if self.current_step > 0:
            prev_price = self.candle_data.iloc[self.current_step - 1]['close']
            price_change_1 = (candle['close'] - prev_price) / prev_price

        if self.current_step > 4:
            prev_price_5 = self.candle_data.iloc[self.current_step - 5]['close']
            price_change_5 = (candle['close'] - prev_price_5) / prev_price_5

        return RLState(
            rsi=candle.get('rsi', 50.0),
            macd=candle.get('macd', 0.0),
            macd_signal=candle.get('macd_signal', 0.0),
            mfi=candle.get('mfi', 50.0),
            atr=candle.get('atr', 0.02),
            adx=candle.get('adx', 25.0),
            volume_ratio=candle.get('volume_ratio', 1.0),
            bb_upper=candle.get('bb_upper', candle['close'] * 1.02),
            bb_middle=candle.get('bb_middle', candle['close']),
            bb_lower=candle.get('bb_lower', candle['close'] * 0.98),
            regime_stage=candle.get('regime_stage', 3),
            regime_confidence=candle.get('regime_confidence', 0.5),
            position_pnl=position_pnl,
            holding_time=holding_time,
            price_change_1=price_change_1,
            price_change_5=price_change_5
        )

    def get_episode_stats(self) -> Dict:
        """에피소드 통계"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        final_balance = self.balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        return {
            'total_reward': self.episode_reward,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'final_balance': final_balance,
            'total_return': total_return,
            'trades': self.episode_trades
        }


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    # 더미 캔들 데이터 생성
    np.random.seed(42)
    n_candles = 1000

    candle_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n_candles) * 0.01) + 100,
        'rsi': np.random.uniform(20, 80, n_candles),
        'macd': np.random.randn(n_candles) * 0.1,
        'macd_signal': np.random.randn(n_candles) * 0.1,
        'mfi': np.random.uniform(20, 80, n_candles),
        'atr': np.random.uniform(0.01, 0.05, n_candles),
        'adx': np.random.uniform(15, 40, n_candles),
        'volume_ratio': np.random.uniform(0.5, 2.0, n_candles),
        'bb_upper': 0,
        'bb_middle': 0,
        'bb_lower': 0,
        'regime_stage': np.random.randint(0, 7, n_candles),
        'regime_confidence': np.random.uniform(0.3, 0.9, n_candles)
    })

    candle_data['bb_middle'] = candle_data['close']
    candle_data['bb_upper'] = candle_data['close'] * 1.02
    candle_data['bb_lower'] = candle_data['close'] * 0.98

    # 환경 생성
    env = TradingEnvironment(candle_data)

    # 에피소드 실행
    state = env.reset()
    print(f"Initial state: {state[:5]}...")

    total_reward = 0
    for i in range(100):
        action = np.random.randint(0, 3)  # 랜덤 행동
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    stats = env.get_episode_stats()
    print(f"\nEpisode Stats:")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Win rate: {stats['win_rate']:.2%}")
    print(f"  Total return: {stats['total_return']:.2%}")
    print("✅ Trading Environment 테스트 완료!")
