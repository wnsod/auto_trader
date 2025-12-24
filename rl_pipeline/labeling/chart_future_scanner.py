"""
Chart Future Scanner - Phase 1 라벨링 시스템
실제 캔들 데이터를 기반으로 전략 신호의 미래 수익/손실 라벨 생성 (MFE/MAE)
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
from rl_pipeline.core.env import config

# 🔥 인터벌 프로필 사용
try:
    from rl_pipeline.core.interval_profiles import INTERVAL_PROFILES, generate_labels
except ImportError:
    logging.getLogger(__name__).warning("interval_profiles 모듈을 찾을 수 없습니다. 기본값 사용")
    INTERVAL_PROFILES = None
    generate_labels = None

# 로깅 설정
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

    def __init__(self,
                 candle_db_path: str = None,
                 strategy_db_path: str = None,
                 fee_bps: float = 10.0,
                 slippage_bps: float = 5.0):
        """
        Args:
            candle_db_path: 캔들 DB 경로
            strategy_db_path: 전략 DB 경로
            fee_bps: 수수료 (basis points)
            slippage_bps: 슬리피지 (basis points)
        """
        self.candle_db_path = candle_db_path or config.RL_DB
        self.strategy_db_path = strategy_db_path or config.STRATEGIES_DB
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        # 🔥 HORIZON_MAP 캐싱 (성능 최적화)
        self._horizon_map_cache = None
        
        # 테이블 보장
        self._ensure_tables()

    def _ensure_tables(self):
        """필요한 테이블 생성"""
        try:
            # 🔥 [Fix] 명시된 DB 경로 사용
            pool = get_strategy_db_pool(self.strategy_db_path)
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # strategy_signal_labels 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_signal_labels (
                        ts INTEGER,
                        coin TEXT,
                        interval TEXT,
                        regime_tag TEXT,
                        strategy_id TEXT,
                        signal_type TEXT,
                        horizon INTEGER,
                        r_max REAL,
                        k_max INTEGER,
                        r_min REAL,
                        k_min INTEGER,
                        fee_bps REAL,
                        slippage_bps REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (strategy_id, ts)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_coin_int ON strategy_signal_labels(coin, interval)")
                
                # strategy_label_stats 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_label_stats (
                        coin TEXT,
                        interval TEXT,
                        regime_tag TEXT,
                        strategy_id TEXT,
                        rmax_mean REAL,
                        rmax_median REAL,
                        rmax_p75 REAL,
                        rmax_p90 REAL,
                        rmin_mean REAL,
                        rmin_median REAL,
                        rmin_p25 REAL,
                        rmin_p10 REAL,
                        kmax_mean REAL,
                        kmax_median INTEGER,
                        kmin_mean REAL,
                        kmin_median INTEGER,
                        pf REAL,
                        win_rate REAL,
                        mdd REAL,
                        n_signals INTEGER,
                        last_updated INTEGER,
                        PRIMARY KEY (coin, interval, regime_tag, strategy_id)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_coin_int ON strategy_label_stats(coin, interval)")
                
                conn.commit()
                logger.info("✅ MFE/MAE 라벨링 테이블 생성 완료")
        except Exception as e:
            logger.warning(f"테이블 생성 실패 (무시 가능): {e}")

    # 🔥 인터벌별 스캔 길이 (interval_profiles 우선 사용, 캐싱 적용)
    @property
    def HORIZON_MAP(self):
        """인터벌별 스캔 길이 반환 (캐싱됨)"""
        if self._horizon_map_cache is None:
            if INTERVAL_PROFILES:
                # interval_profiles에서 target_horizon 사용
                self._horizon_map_cache = {
                    interval: profile['labeling']['target_horizon']
                    for interval, profile in INTERVAL_PROFILES.items()
                }
            else:
                # 폴백: 기존 값 사용
                self._horizon_map_cache = {
                    '15m': 40,   # 10시간
                    '30m': 40,   # 20시간
                    '240m': 20,  # 3.3일
                    '1d': 15     # 15일
                }
        return self._horizon_map_cache

    def load_candles(self, coin: str, interval: str) -> pd.DataFrame:
        """캔들 데이터 로드"""
        conn = sqlite3.connect(self.candle_db_path)
        try:
            # 컬럼 확인 (regime_stage 등이 없을 수 있음)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info(candles)")
            columns = [info[1] for info in cursor.fetchall()]
            
            select_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # 필수 지표 (있으면 로드)
            optional_cols = ['rsi', 'macd', 'macd_signal', 'mfi', 'adx', 'atr', 
                           'bb_upper', 'bb_middle', 'bb_lower', 'volume_ratio', 
                           'regime_stage', 'regime_label']
            for col in optional_cols:
                if col in columns:
                    select_cols.append(col)
            
            query = f"""
            SELECT {', '.join(select_cols)}
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=(coin, interval))

            if len(df) > 0:
                logger.debug(f"  {coin} {interval}: {len(df)}개 캔들 로드")

            return df
        finally:
            conn.close()

    def load_strategies(self, coin: str, interval: str) -> List[Dict]:
        """전략 로드 (symbol/coin 컬럼 호환)"""
        try:
            # 🔥 [Fix] 명시된 DB 경로 사용 (코인별 DB에서 전략 조회)
            pool = get_strategy_db_pool(self.strategy_db_path)
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 🔥 컬럼 이름 확인 (coin vs symbol)
                cursor.execute("PRAGMA table_info(strategies)")
                cols = [c[1] for c in cursor.fetchall()]
                
                # symbol 컬럼 사용 (대부분의 경우)
                coin_col = 'symbol' if 'symbol' in cols else 'coin'
                
                # 필수 컬럼만 선택 (없는 컬럼은 제외)
                base_cols = ['id', 'interval', 'regime']
                optional_cols = ['rsi_min', 'rsi_max', 'macd_buy_threshold', 'macd_sell_threshold',
                               'mfi_min', 'mfi_max', 'volume_ratio_min', 'volume_ratio_max',
                               'stop_loss_pct', 'take_profit_pct', 'strategy_conditions']
                
                select_cols = [f'{coin_col} as coin'] + base_cols
                for col in optional_cols:
                    if col in cols:
                        select_cols.append(col)
                
                query = f"""
                    SELECT {', '.join(select_cols)}
                    FROM strategies
                    WHERE {coin_col} = ? AND interval = ?
                """
                cursor.execute(query, (coin, interval))

                columns = [desc[0] for desc in cursor.description]
                strategies = []
                for row in cursor.fetchall():
                    strategies.append(dict(zip(columns, row)))

                return strategies
                
        except sqlite3.OperationalError as e:
            logger.warning(f"전략 로드 실패 (테이블/컬럼 없을 수 있음): {e}")
            return []
        except Exception as e:
            logger.error(f"전략 로드 중 예외: {e}")
            return []

    def calculate_signal(self, candle: pd.Series, strategy: Dict) -> Optional[str]:
        """
        전략 신호 계산
        
        Returns:
            'BUY', 'SELL', or None
        """
        try:
            # 필수 지표가 없으면 패스
            rsi = candle.get('rsi')
            macd = candle.get('macd')
            
            if rsi is None and macd is None:
                return None
            
            mfi = candle.get('mfi')
            volume_ratio = candle.get('volume_ratio')

            # RSI 조건
            if rsi is not None and pd.notna(strategy.get('rsi_min')) and pd.notna(strategy.get('rsi_max')):
                if not (strategy['rsi_min'] <= rsi <= strategy['rsi_max']):
                    return None

            # MFI 조건
            if mfi is not None and pd.notna(strategy.get('mfi_min')) and pd.notna(strategy.get('mfi_max')):
                if not (strategy['mfi_min'] <= mfi <= strategy['mfi_max']):
                    return None

            # Volume 조건
            if volume_ratio is not None and pd.notna(strategy.get('volume_ratio_min')):
                if volume_ratio < strategy['volume_ratio_min']:
                    return None

            # MACD 기반 신호
            macd_buy_threshold = strategy.get('macd_buy_threshold', 0) or 0
            macd_sell_threshold = strategy.get('macd_sell_threshold', 0) or 0

            # 신호 판단 로직
            # 1. MACD 기반
            if macd is not None and pd.notna(macd):
                if macd > macd_sell_threshold:
                    return 'BUY'
                elif macd < macd_buy_threshold:
                    return 'SELL'

            # 2. RSI 기반 (보조 - MACD가 없거나 중립일 때)
            if rsi is not None and pd.notna(rsi):
                if rsi < 30:
                    return 'BUY'
                elif rsi > 70:
                    return 'SELL'

            return None

        except Exception as e:
            # logger.debug(f"Signal calculation error: {e}")
            return None

    def scan_future(self,
                    df: pd.DataFrame,
                    signal_idx: int,
                    signal_type: str,
                    horizon: int) -> Tuple[float, int, float, int]:
        """
        미래 캔들 스캔하여 r_max, k_max, r_min, k_min 계산 (MFE/MAE)
        
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
                # 최대 수익 (MFE): high 기준
                r_up = (high - entry_price) / entry_price - cost_rate
                if r_up > r_max:
                    r_max = r_up
                    k_max = k

                # 최대 손실 (MAE): low 기준
                r_down = (low - entry_price) / entry_price - cost_rate
                if r_down < r_min:
                    r_min = r_down
                    k_min = k

            else:  # SELL
                # SELL은 가격 하락이 수익
                # 최대 수익 (MFE): low 기준 (가격 하락)
                r_down = (entry_price - low) / entry_price - cost_rate
                if r_down > r_max:
                    r_max = r_down
                    k_max = k

                # 최대 손실 (MAE): high 기준 (가격 상승)
                r_up = (entry_price - high) / entry_price - cost_rate
                if r_up < r_min:
                    r_min = r_up
                    k_min = k

        # 초기값이 그대로면 0으로 처리
        if r_max == -float('inf'): r_max = 0.0
        if r_min == float('inf'): r_min = 0.0
        
        return r_max, k_max, r_min, k_min

    def _get_last_labeled_ts(self, coin: str, interval: str) -> int:
        """마지막 라벨링된 timestamp 조회"""
        try:
            pool = get_strategy_db_pool(self.strategy_db_path)
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(ts) FROM strategy_signal_labels
                    WHERE coin = ? AND interval = ?
                """, (coin, interval))
                result = cursor.fetchone()
                return result[0] if result and result[0] else 0
        except:
            return 0

    def process_coin_interval(self, coin: str, interval: str, incremental: bool = True) -> List[SignalLabel]:
        """코인×인터벌 라벨링 처리 (🔥 완전 벡터화 + 증분 처리)
        
        Args:
            coin: 코인 심볼
            interval: 인터벌
            incremental: 증분 처리 여부 (기본 True)
        """
        logger.info(f"\n📊 {coin} {interval} 라벨링 시작...")

        # 1. 캔들 데이터 로드
        df = self.load_candles(coin, interval)
        if len(df) == 0:
            logger.warning(f"  ⚠️ {coin} {interval}: 캔들 데이터 없음")
            return []

        original_count = len(df)
        
        # 🔥 [증분 처리] 마지막 라벨링 시점 이후 캔들만 처리
        if incremental:
            last_ts = self._get_last_labeled_ts(coin, interval)
            if last_ts > 0:
                # horizon 만큼 여유를 두고 필터링 (MFE/MAE 계산에 필요)
                horizon = self.HORIZON_MAP.get(interval, 20)
                buffer_rows = horizon * 2  # 안전 버퍼
                
                # last_ts 이전 캔들 중 buffer_rows 개만 유지 + 이후 전체
                df_before = df[df['timestamp'] <= last_ts].tail(buffer_rows)
                df_after = df[df['timestamp'] > last_ts]
                
                if len(df_after) == 0:
                    logger.info(f"  ✅ 캔들: {original_count}개 (새 캔들 없음 - 스킵)")
                    return []
                
                df = pd.concat([df_before, df_after], ignore_index=True)
                logger.info(f"  ✅ 캔들: {original_count}개 → {len(df)}개 (증분: 새 캔들 {len(df_after)}개)")
            else:
                logger.info(f"  ✅ 캔들: {len(df)}개 (첫 라벨링)")
        else:
            logger.info(f"  ✅ 캔들: {len(df)}개 (전체 처리)")

        # 2. 전략 로드
        strategies = self.load_strategies(coin, interval)
        if len(strategies) == 0:
            logger.warning(f"  ⚠️ {coin} {interval}: 전략 없음 (신규 전략 생성 후 재시도 필요)")
            return []

        logger.info(f"  ✅ 전략: {len(strategies)}개")

        # 3. 스캔 길이
        horizon = self.HORIZON_MAP.get(interval, 20)
        n_candles = len(df)
        n_strategies = len(strategies)
        scan_range = n_candles - horizon
        
        if scan_range <= 0:
            logger.warning(f"  ⚠️ 캔들 수가 horizon보다 적음")
            return []

        # 4. 🔥 [벡터화] numpy 배열 준비
        timestamps = df['timestamp'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # 지표 배열 (없으면 None)
        rsi_arr = df['rsi'].values if 'rsi' in df.columns else None
        macd_arr = df['macd'].values if 'macd' in df.columns else None
        mfi_arr = df['mfi'].values if 'mfi' in df.columns else None
        vol_ratio_arr = df['volume_ratio'].values if 'volume_ratio' in df.columns else None
        
        # 5. 🔥 [벡터화] 전략 파라미터를 배열로 변환
        strategy_ids = []
        regimes = []
        rsi_mins = np.full(n_strategies, -np.inf)
        rsi_maxs = np.full(n_strategies, np.inf)
        mfi_mins = np.full(n_strategies, -np.inf)
        mfi_maxs = np.full(n_strategies, np.inf)
        vol_ratio_mins = np.full(n_strategies, -np.inf)
        macd_buy_thresholds = np.zeros(n_strategies)
        macd_sell_thresholds = np.zeros(n_strategies)
        
        for i, s in enumerate(strategies):
            strategy_ids.append(s['id'])
            regimes.append(s.get('regime', 'ranging'))
            
            if s.get('rsi_min') is not None:
                rsi_mins[i] = s['rsi_min']
            if s.get('rsi_max') is not None:
                rsi_maxs[i] = s['rsi_max']
            if s.get('mfi_min') is not None:
                mfi_mins[i] = s['mfi_min']
            if s.get('mfi_max') is not None:
                mfi_maxs[i] = s['mfi_max']
            if s.get('volume_ratio_min') is not None:
                vol_ratio_mins[i] = s['volume_ratio_min']
            macd_buy_thresholds[i] = s.get('macd_buy_threshold', 0) or 0
            macd_sell_thresholds[i] = s.get('macd_sell_threshold', 0) or 0

        # 6. 🔥 [벡터화] 조건 매트릭스 계산 (M strategies × N candles)
        # 각 (전략, 캔들) 조합에 대해 조건 충족 여부를 한 번에 계산
        
        # 기본 마스크: 모든 조합 True
        valid_mask = np.ones((n_strategies, scan_range), dtype=bool)
        
        # RSI 조건 (Broadcasting: (M,1) vs (N,) → (M,N))
        if rsi_arr is not None:
            rsi_candles = rsi_arr[:scan_range]  # (N,)
            valid_mask &= (rsi_candles >= rsi_mins[:, None]) & (rsi_candles <= rsi_maxs[:, None])
        
        # MFI 조건
        if mfi_arr is not None:
            mfi_candles = mfi_arr[:scan_range]
            valid_mask &= (mfi_candles >= mfi_mins[:, None]) & (mfi_candles <= mfi_maxs[:, None])
        
        # Volume Ratio 조건
        if vol_ratio_arr is not None:
            vol_candles = vol_ratio_arr[:scan_range]
            valid_mask &= (vol_candles >= vol_ratio_mins[:, None])
        
        # 7. 🔥 [벡터화] 신호 타입 결정 (BUY/SELL)
        # signal_matrix: (M, N) - 0=No Signal, 1=BUY, -1=SELL
        signal_matrix = np.zeros((n_strategies, scan_range), dtype=np.int8)
        
        if macd_arr is not None:
            macd_candles = macd_arr[:scan_range]  # (N,)
            # BUY: macd > sell_threshold
            buy_mask = macd_candles > macd_sell_thresholds[:, None]
            # SELL: macd < buy_threshold
            sell_mask = macd_candles < macd_buy_thresholds[:, None]
            
            signal_matrix[buy_mask & valid_mask] = 1   # BUY
            signal_matrix[sell_mask & valid_mask] = -1  # SELL
        
        # RSI 기반 보조 신호 (MACD 신호 없는 경우만)
        if rsi_arr is not None:
            rsi_candles = rsi_arr[:scan_range]
            no_signal_mask = (signal_matrix == 0) & valid_mask
            
            rsi_buy = rsi_candles < 30
            rsi_sell = rsi_candles > 70
            
            signal_matrix[no_signal_mask & rsi_buy] = 1
            signal_matrix[no_signal_mask & rsi_sell] = -1

        # 8. 🔥 [벡터화] MFE/MAE 사전 계산 (Sliding Window)
        # 각 캔들 idx에서 horizon 기간 내의 max(high), min(low) 미리 계산
        cost_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        
        # Rolling max high, min low 계산 (horizon 윈도우)
        # future_max_high[i] = max(high[i+1:i+1+horizon])
        # future_min_low[i] = min(low[i+1:i+1+horizon])
        future_max_high = np.zeros(scan_range)
        future_min_low = np.zeros(scan_range)
        future_k_max = np.zeros(scan_range, dtype=np.int32)  # MFE 도달 시점
        future_k_min = np.zeros(scan_range, dtype=np.int32)  # MAE 도달 시점
        
        for idx in range(scan_range):
            future_highs = highs[idx + 1:idx + 1 + horizon]
            future_lows = lows[idx + 1:idx + 1 + horizon]
            
            if len(future_highs) > 0:
                future_max_high[idx] = np.max(future_highs)
                future_min_low[idx] = np.min(future_lows)
                future_k_max[idx] = np.argmax(future_highs) + 1
                future_k_min[idx] = np.argmin(future_lows) + 1
        
        # 9. 🔥 [벡터화] 라벨 생성 (신호가 있는 조합만)
        labels = []
        
        # 신호가 있는 (전략 인덱스, 캔들 인덱스) 찾기
        signal_indices = np.where(signal_matrix != 0)
        strategy_indices = signal_indices[0]
        candle_indices = signal_indices[1]
        
        logger.info(f"  📊 신호 발생: {len(strategy_indices)}개 (전체 조합 중)")
        
        for i in range(len(strategy_indices)):
            s_idx = strategy_indices[i]
            c_idx = candle_indices[i]
            
            signal_type = 'BUY' if signal_matrix[s_idx, c_idx] == 1 else 'SELL'
            entry_price = closes[c_idx]
            
            # MFE/MAE 계산
            if signal_type == 'BUY':
                # BUY: 상승이 수익
                r_max = (future_max_high[c_idx] - entry_price) / entry_price - cost_rate
                r_min = (future_min_low[c_idx] - entry_price) / entry_price - cost_rate
                k_max = future_k_max[c_idx]
                k_min = future_k_min[c_idx]
            else:
                # SELL: 하락이 수익
                r_max = (entry_price - future_min_low[c_idx]) / entry_price - cost_rate
                r_min = (entry_price - future_max_high[c_idx]) / entry_price - cost_rate
                k_max = future_k_min[c_idx]
                k_min = future_k_max[c_idx]
            
            label = SignalLabel(
                ts=int(timestamps[c_idx]),
                coin=coin,
                interval=interval,
                regime_tag=regimes[s_idx],
                strategy_id=strategy_ids[s_idx],
                signal_type=signal_type,
                horizon=horizon,
                r_max=float(r_max),
                k_max=int(k_max),
                r_min=float(r_min),
                k_min=int(k_min),
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps
            )
            labels.append(label)

        logger.info(f"  ✅ {coin} {interval}: {len(labels)}개 신호 라벨링 완료")

        return labels

    def save_labels(self, labels: List[SignalLabel]) -> int:
        """라벨 DB 저장"""
        if len(labels) == 0:
            return 0

        # 🔥 [Fix] 명시된 DB 경로 사용
        pool = get_strategy_db_pool(self.strategy_db_path)
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            insert_query = """
            INSERT OR REPLACE INTO strategy_signal_labels
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
                          intervals: Optional[List[str]] = None,
                          incremental: bool = True) -> Dict[str, int]:
        """전체 라벨링 실행
        
        Args:
            coins: 대상 코인 목록
            intervals: 대상 인터벌 목록
            incremental: 증분 처리 여부 (기본 True - 새 캔들만 처리)
        """
        mode = "증분" if incremental else "전체"
        logger.info(f"🚀 Chart Future Scanner 시작 (MFE/MAE 라벨링 - {mode} 모드)\n")

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
                    labels = self.process_coin_interval(coin, interval, incremental=incremental)
                    saved = self.save_labels(labels)
                    results[key] = saved
                    total_labels += saved

                    if saved > 0:
                        logger.info(f"  💾 저장: {saved}개 라벨")

                except Exception as e:
                    logger.error(f"  ❌ {coin} {interval} 실패: {e}")
                    results[key] = 0

        logger.info(f"\n🎉 라벨링 완료: 총 {total_labels}개 라벨 생성")

        return results

def main():
    """메인 실행 함수"""
    # 환경변수로 제어 가능
    quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
