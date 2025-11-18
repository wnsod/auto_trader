"""
Chart Future Scanner - Phase 1 라벨링 시스템
실제 캔들 데이터를 기반으로 전략 신호의 미래 수익/손실 라벨 생성
"""
import sys
import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.db.connection_pool import get_strategy_db_pool

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SignalLabel:
    """신호 라벨 데이터"""
    ts: int
    coin: str
    interval: str
    regime_tag: str
    strategy_id: str
    signal_type: str
    horizon: int
    r_max: float
    k_max: int
    r_min: float
    k_min: int
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

class ChartFutureScanner:
    """차트 미래 스캐너 - 신호별 라벨링"""

    # 인터벌별 스캔 길이 (H)
    HORIZON_MAP = {
        '15m': 40,   # 10시간
        '30m': 40,   # 20시간
        '240m': 20,  # 3.3일
        '1d': 15     # 15일
    }

    def __init__(self,
                 candle_db_path: str = '/workspace/data_storage/rl_candles.db',
                 strategy_db_path: str = '/workspace/data_storage/rl_strategies.db',
                 fee_bps: float = 10.0,
                 slippage_bps: float = 5.0):
        """
        Args:
            candle_db_path: 캔들 DB 경로
            strategy_db_path: 전략 DB 경로
            fee_bps: 수수료 (basis points)
            slippage_bps: 슬리피지 (basis points)
        """
        self.candle_db_path = candle_db_path
        self.strategy_db_path = strategy_db_path
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    def load_candles(self, coin: str, interval: str) -> pd.DataFrame:
        """캔들 데이터 로드"""
        conn = sqlite3.connect(self.candle_db_path)
        try:
            query = """
            SELECT timestamp, open, high, low, close, volume,
                   rsi, macd, macd_signal, mfi, adx, atr,
                   bb_upper, bb_middle, bb_lower, volume_ratio,
                   regime_stage, regime_label
            FROM candles
            WHERE coin = ? AND interval = ?
            ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=(coin, interval))

            if len(df) > 0:
                logger.debug(f"  {coin} {interval}: {len(df)}개 캔들 로드")

            return df
        finally:
            conn.close()

    def load_strategies(self, coin: str, interval: str) -> List[Dict]:
        """전략 로드"""
        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, coin, interval, regime,
                       rsi_min, rsi_max,
                       macd_buy_threshold, macd_sell_threshold,
                       mfi_min, mfi_max,
                       volume_ratio_min, volume_ratio_max,
                       stop_loss_pct, take_profit_pct,
                       strategy_conditions
                FROM coin_strategies
                WHERE coin = ? AND interval = ?
            """, (coin, interval))

            columns = [desc[0] for desc in cursor.description]
            strategies = []
            for row in cursor.fetchall():
                strategies.append(dict(zip(columns, row)))

            return strategies

    def calculate_signal(self, candle: pd.Series, strategy: Dict) -> Optional[str]:
        """
        전략 신호 계산

        Returns:
            'BUY', 'SELL', or None
        """
        try:
            rsi = candle['rsi']
            macd = candle['macd']
            mfi = candle['mfi']
            volume_ratio = candle['volume_ratio']

            # RSI 조건
            if pd.notna(strategy.get('rsi_min')) and pd.notna(strategy.get('rsi_max')):
                if not (strategy['rsi_min'] <= rsi <= strategy['rsi_max']):
                    return None

            # MFI 조건
            if pd.notna(strategy.get('mfi_min')) and pd.notna(strategy.get('mfi_max')):
                if not (strategy['mfi_min'] <= mfi <= strategy['mfi_max']):
                    return None

            # Volume 조건
            if pd.notna(strategy.get('volume_ratio_min')):
                if volume_ratio < strategy['volume_ratio_min']:
                    return None

            # MACD 기반 신호
            macd_buy_threshold = strategy.get('macd_buy_threshold', 0)
            macd_sell_threshold = strategy.get('macd_sell_threshold', 0)

            # 신호 판단 로직
            # 1. MACD 기반
            if macd is not None and pd.notna(macd):
                if macd > macd_sell_threshold:
                    return 'BUY'
                elif macd < macd_buy_threshold:
                    return 'SELL'

            # 2. RSI 기반 (보조)
            if rsi is not None and pd.notna(rsi):
                if rsi < 30:
                    return 'BUY'
                elif rsi > 70:
                    return 'SELL'

            return None

        except Exception as e:
            logger.debug(f"Signal calculation error: {e}")
            return None

    def scan_future(self,
                    df: pd.DataFrame,
                    signal_idx: int,
                    signal_type: str,
                    horizon: int) -> Tuple[float, int, float, int]:
        """
        미래 캔들 스캔하여 r_max, k_max, r_min, k_min 계산

        Args:
            df: 캔들 데이터
            signal_idx: 신호 발생 인덱스
            signal_type: 'BUY' or 'SELL'
            horizon: 스캔 길이

        Returns:
            (r_max, k_max, r_min, k_min)
        """
        entry_price = df.iloc[signal_idx]['close']

        # 미래 구간 (신호 발생 다음 캔들부터)
        future_start = signal_idx + 1
        future_end = min(signal_idx + 1 + horizon, len(df))

        if future_start >= len(df):
            return 0.0, 0, 0.0, 0

        future_df = df.iloc[future_start:future_end]

        # 수수료/슬리피지 계산
        cost_bps = self.fee_bps + self.slippage_bps
        cost_rate = cost_bps / 10000.0

        # BUY 신호: 상승 = 수익, 하락 = 손실
        # SELL 신호: 하락 = 수익, 상승 = 손실

        r_max = -float('inf')
        k_max = 0
        r_min = float('inf')
        k_min = 0

        for k, (idx, candle) in enumerate(future_df.iterrows(), start=1):
            high = candle['high']
            low = candle['low']

            if signal_type == 'BUY':
                # 최대 수익: high 기준
                r_up = (high - entry_price) / entry_price - cost_rate
                if r_up > r_max:
                    r_max = r_up
                    k_max = k

                # 최대 손실: low 기준
                r_down = (low - entry_price) / entry_price - cost_rate
                if r_down < r_min:
                    r_min = r_down
                    k_min = k

            else:  # SELL
                # SELL은 가격 하락이 수익
                # 최대 수익: low 기준 (가격 하락)
                r_down = (entry_price - low) / entry_price - cost_rate
                if r_down > r_max:
                    r_max = r_down
                    k_max = k

                # 최대 손실: high 기준 (가격 상승)
                r_up = (entry_price - high) / entry_price - cost_rate
                if r_up < r_min:
                    r_min = r_up
                    k_min = k

        return r_max, k_max, r_min, k_min

    def process_coin_interval(self, coin: str, interval: str) -> List[SignalLabel]:
        """코인×인터벌 라벨링 처리"""
        logger.info(f"\n📊 {coin} {interval} 라벨링 시작...")

        # 1. 캔들 데이터 로드
        df = self.load_candles(coin, interval)
        if len(df) == 0:
            logger.warning(f"  ⚠️ {coin} {interval}: 캔들 데이터 없음")
            return []

        logger.info(f"  ✅ 캔들: {len(df)}개")

        # 2. 전략 로드
        strategies = self.load_strategies(coin, interval)
        if len(strategies) == 0:
            logger.warning(f"  ⚠️ {coin} {interval}: 전략 없음")
            return []

        logger.info(f"  ✅ 전략: {len(strategies)}개")

        # 3. 스캔 길이
        horizon = self.HORIZON_MAP.get(interval, 20)

        # 4. 각 전략별 신호 계산 및 라벨링
        labels = []
        total_signals = 0

        for strategy in strategies:
            strategy_id = strategy['id']
            regime = strategy.get('regime', 'ranging')

            # 각 캔들에 대해 신호 계산
            for idx in range(len(df) - horizon):  # 미래 스캔 가능한 구간만
                candle = df.iloc[idx]

                signal_type = self.calculate_signal(candle, strategy)
                if signal_type is None:
                    continue

                # 미래 스캔
                r_max, k_max, r_min, k_min = self.scan_future(
                    df, idx, signal_type, horizon
                )

                # 라벨 생성
                label = SignalLabel(
                    ts=int(candle['timestamp']),
                    coin=coin,
                    interval=interval,
                    regime_tag=regime,
                    strategy_id=strategy_id,
                    signal_type=signal_type,
                    horizon=horizon,
                    r_max=r_max,
                    k_max=k_max,
                    r_min=r_min,
                    k_min=k_min,
                    fee_bps=self.fee_bps,
                    slippage_bps=self.slippage_bps
                )

                labels.append(label)
                total_signals += 1

        logger.info(f"  ✅ {coin} {interval}: {total_signals}개 신호 라벨링 완료")

        return labels

    def save_labels(self, labels: List[SignalLabel]) -> int:
        """라벨 DB 저장"""
        if len(labels) == 0:
            return 0

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            insert_query = """
            INSERT INTO strategy_signal_labels
            (ts, coin, interval, regime_tag, strategy_id, signal_type,
             horizon, r_max, k_max, r_min, k_min, fee_bps, slippage_bps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            data = [
                (l.ts, l.coin, l.interval, l.regime_tag, l.strategy_id, l.signal_type,
                 l.horizon, l.r_max, l.k_max, l.r_min, l.k_min, l.fee_bps, l.slippage_bps)
                for l in labels
            ]

            cursor.executemany(insert_query, data)
            conn.commit()

        return len(labels)

    def run_full_labeling(self,
                          coins: Optional[List[str]] = None,
                          intervals: Optional[List[str]] = None) -> Dict[str, int]:
        """전체 라벨링 실행"""
        logger.info("🚀 Chart Future Scanner 시작\n")

        # 기본값: 모든 코인/인터벌
        if coins is None:
            coins = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'SOL', 'XRP']

        if intervals is None:
            intervals = ['15m', '30m', '240m', '1d']

        results = {}
        total_labels = 0

        for coin in coins:
            for interval in intervals:
                key = f"{coin}_{interval}"

                try:
                    labels = self.process_coin_interval(coin, interval)
                    saved = self.save_labels(labels)
                    results[key] = saved
                    total_labels += saved

                    logger.info(f"  💾 저장: {saved}개 라벨")

                except Exception as e:
                    logger.error(f"  ❌ {coin} {interval} 실패: {e}", exc_info=True)
                    results[key] = 0

        logger.info(f"\n🎉 라벨링 완료: 총 {total_labels}개 라벨 생성")

        return results

def main():
    """메인 실행 함수"""
    # 환경변수로 제어 가능
    quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'

    scanner = ChartFutureScanner()

    if quick_test:
        # 빠른 테스트: BTC 15m만
        logger.info("🧪 빠른 테스트 모드: BTC 15m만 실행")
        results = scanner.run_full_labeling(
            coins=['BTC'],
            intervals=['15m']
        )
    else:
        # 전체 실행
        results = scanner.run_full_labeling()

    # 결과 요약
    logger.info("\n📊 라벨링 결과 요약:")
    for key, count in sorted(results.items()):
        logger.info(f"  {key}: {count}개")

if __name__ == "__main__":
    main()
