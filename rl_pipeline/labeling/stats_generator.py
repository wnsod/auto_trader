"""
통계 생성기 - Phase 1 라벨링 통계
strategy_signal_labels → strategy_label_stats
"""
import sys
import os
import logging
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.db.connection_pool import get_strategy_db_pool
from rl_pipeline.core.env import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LabelStats:
    """라벨 통계"""
    coin: str
    interval: str
    regime_tag: str
    strategy_id: str
    rmax_mean: float
    rmax_median: float
    rmax_p75: float
    rmax_p90: float
    rmin_mean: float
    rmin_median: float
    rmin_p25: float
    rmin_p10: float
    kmax_mean: float
    kmax_median: int
    kmin_mean: float
    kmin_median: int
    pf: float
    win_rate: float
    mdd: float
    n_signals: int
    last_updated: int

class StatsGenerator:
    """라벨링 통계 생성기"""

    def __init__(self, strategy_db_path: str = None):
        self.strategy_db_path = strategy_db_path or config.STRATEGIES_DB
        self._ensure_tables()

    def _ensure_tables(self):
        """통계 테이블 생성"""
        try:
            # 🔥 [Fix] 명시된 DB 경로 사용
            pool = get_strategy_db_pool(self.strategy_db_path)
            with pool.get_connection() as conn:
                cursor = conn.cursor()
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
                conn.commit()
        except Exception as e:
            logger.warning(f"통계 테이블 생성 실패 (무시 가능): {e}")

    def calculate_stats(self,
                       coin: str,
                       interval: str,
                       regime_tag: str,
                       strategy_id: str) -> LabelStats:
        """
        전략×레짐×인터벌별 통계 계산
        
        Returns:
            LabelStats 객체
        """
        # 🔥 [Fix] 명시된 DB 경로 사용
        pool = get_strategy_db_pool(self.strategy_db_path)
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 라벨 데이터 조회
            try:
                cursor.execute("""
                    SELECT r_max, r_min, k_max, k_min
                    FROM strategy_signal_labels
                    WHERE coin = ? AND interval = ? AND regime_tag = ? AND strategy_id = ?
                """, (coin, interval, regime_tag, strategy_id))
                
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                # 테이블이 없을 경우
                rows = []

            if len(rows) == 0:
                # 데이터 없음
                return LabelStats(
                    coin=coin,
                    interval=interval,
                    regime_tag=regime_tag,
                    strategy_id=strategy_id,
                    rmax_mean=0.0,
                    rmax_median=0.0,
                    rmax_p75=0.0,
                    rmax_p90=0.0,
                    rmin_mean=0.0,
                    rmin_median=0.0,
                    rmin_p25=0.0,
                    rmin_p10=0.0,
                    kmax_mean=0.0,
                    kmax_median=0,
                    kmin_mean=0.0,
                    kmin_median=0,
                    pf=0.0,
                    win_rate=0.0,
                    mdd=0.0,
                    n_signals=0,
                    last_updated=int(datetime.now().timestamp())
                )

            # numpy 배열로 변환
            r_max = np.array([row[0] for row in rows])
            r_min = np.array([row[1] for row in rows])
            k_max = np.array([row[2] for row in rows])
            k_min = np.array([row[3] for row in rows])

            # r_max 통계
            rmax_mean = float(np.mean(r_max))
            rmax_median = float(np.median(r_max))
            rmax_p75 = float(np.percentile(r_max, 75))
            rmax_p90 = float(np.percentile(r_max, 90))

            # r_min 통계
            rmin_mean = float(np.mean(r_min))
            rmin_median = float(np.median(r_min))
            rmin_p25 = float(np.percentile(r_min, 25))
            rmin_p10 = float(np.percentile(r_min, 10))

            # k_max, k_min 통계
            kmax_mean = float(np.mean(k_max))
            kmax_median = int(np.median(k_max))
            kmin_mean = float(np.mean(k_min))
            kmin_median = int(np.median(k_min))

            # Profit Factor 계산
            # PF = 총이익 / 총손실
            total_profit = np.sum(r_max[r_max > 0])
            total_loss = np.abs(np.sum(r_min[r_min < 0]))
            pf = float(total_profit / total_loss) if total_loss > 0 else 0.0

            # Win Rate 계산
            wins = np.sum(r_max > 0)
            win_rate = float(wins / len(r_max)) if len(r_max) > 0 else 0.0

            # MDD 근사 (r_min의 최소값)
            mdd = float(np.min(r_min))

            # 표본 수
            n_signals = len(rows)

            return LabelStats(
                coin=coin,
                interval=interval,
                regime_tag=regime_tag,
                strategy_id=strategy_id,
                rmax_mean=rmax_mean,
                rmax_median=rmax_median,
                rmax_p75=rmax_p75,
                rmax_p90=rmax_p90,
                rmin_mean=rmin_mean,
                rmin_median=rmin_median,
                rmin_p25=rmin_p25,
                rmin_p10=rmin_p10,
                kmax_mean=kmax_mean,
                kmax_median=kmax_median,
                kmin_mean=kmin_mean,
                kmin_median=kmin_median,
                pf=pf,
                win_rate=win_rate,
                mdd=mdd,
                n_signals=n_signals,
                last_updated=int(datetime.now().timestamp())
            )

    def save_stats(self, stats: LabelStats) -> bool:
        """통계 저장 (UPSERT)"""
        # 🔥 [Fix] 명시된 DB 경로 사용
        pool = get_strategy_db_pool(self.strategy_db_path)
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO strategy_label_stats
                (coin, interval, regime_tag, strategy_id,
                 rmax_mean, rmax_median, rmax_p75, rmax_p90,
                 rmin_mean, rmin_median, rmin_p25, rmin_p10,
                 kmax_mean, kmax_median, kmin_mean, kmin_median,
                 pf, win_rate, mdd, n_signals, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(coin, interval, regime_tag, strategy_id)
                DO UPDATE SET
                    rmax_mean = excluded.rmax_mean,
                    rmax_median = excluded.rmax_median,
                    rmax_p75 = excluded.rmax_p75,
                    rmax_p90 = excluded.rmax_p90,
                    rmin_mean = excluded.rmin_mean,
                    rmin_median = excluded.rmin_median,
                    rmin_p25 = excluded.rmin_p25,
                    rmin_p10 = excluded.rmin_p10,
                    kmax_mean = excluded.kmax_mean,
                    kmax_median = excluded.kmax_median,
                    kmin_mean = excluded.kmin_mean,
                    kmin_median = excluded.kmin_median,
                    pf = excluded.pf,
                    win_rate = excluded.win_rate,
                    mdd = excluded.mdd,
                    n_signals = excluded.n_signals,
                    last_updated = excluded.last_updated
            """, (
                stats.coin, stats.interval, stats.regime_tag, stats.strategy_id,
                stats.rmax_mean, stats.rmax_median, stats.rmax_p75, stats.rmax_p90,
                stats.rmin_mean, stats.rmin_median, stats.rmin_p25, stats.rmin_p10,
                stats.kmax_mean, stats.kmax_median, stats.kmin_mean, stats.kmin_median,
                stats.pf, stats.win_rate, stats.mdd, stats.n_signals, stats.last_updated
            ))

            conn.commit()

        return True

    def generate_all_stats(self) -> int:
        """모든 통계 생성 (🔥 벡터화 버전)"""
        logger.info("🚀 통계 생성 시작\n")

        # 🔥 [Fix] 명시된 DB 경로 사용
        pool = get_strategy_db_pool(self.strategy_db_path)
        with pool.get_connection() as conn:
            try:
                # 🔥 [벡터화] 모든 라벨 데이터를 한 번에 로드
                logger.info("  📥 라벨 데이터 로딩 중...")
                df = pd.read_sql("""
                    SELECT coin, interval, regime_tag, strategy_id, r_max, r_min, k_max, k_min
                    FROM strategy_signal_labels
                """, conn)
                
                if len(df) == 0:
                    logger.warning("라벨 데이터가 없습니다.")
                    return 0
                
                logger.info(f"  ✅ {len(df):,}개 라벨 로드 완료")
                
            except sqlite3.OperationalError:
                logger.warning("라벨 테이블(strategy_signal_labels)이 없습니다.")
                return 0

        # 🔥 [벡터화] pandas groupby로 통계 일괄 계산
        logger.info("  📊 통계 계산 중 (벡터화)...")
        
        grouped = df.groupby(['coin', 'interval', 'regime_tag', 'strategy_id'])
        
        # 집계 함수 정의
        def calc_stats(g):
            r_max = g['r_max'].values
            r_min = g['r_min'].values
            k_max = g['k_max'].values
            k_min = g['k_min'].values
            n = len(g)
            
            if n == 0:
                return pd.Series({
                    'rmax_mean': 0.0, 'rmax_median': 0.0, 'rmax_p75': 0.0, 'rmax_p90': 0.0,
                    'rmin_mean': 0.0, 'rmin_median': 0.0, 'rmin_p25': 0.0, 'rmin_p10': 0.0,
                    'kmax_mean': 0.0, 'kmax_median': 0, 'kmin_mean': 0.0, 'kmin_median': 0,
                    'pf': 0.0, 'win_rate': 0.0, 'mdd': 0.0, 'n_signals': 0
                })
            
            # 통계 계산
            total_profit = np.sum(r_max[r_max > 0])
            total_loss = np.abs(np.sum(r_min[r_min < 0]))
            pf = total_profit / total_loss if total_loss > 0 else 0.0
            wins = np.sum(r_max > 0)
            win_rate = wins / n if n > 0 else 0.0
            
            return pd.Series({
                'rmax_mean': np.mean(r_max),
                'rmax_median': np.median(r_max),
                'rmax_p75': np.percentile(r_max, 75),
                'rmax_p90': np.percentile(r_max, 90),
                'rmin_mean': np.mean(r_min),
                'rmin_median': np.median(r_min),
                'rmin_p25': np.percentile(r_min, 25),
                'rmin_p10': np.percentile(r_min, 10),
                'kmax_mean': np.mean(k_max),
                'kmax_median': int(np.median(k_max)),
                'kmin_mean': np.mean(k_min),
                'kmin_median': int(np.median(k_min)),
                'pf': pf,
                'win_rate': win_rate,
                'mdd': np.min(r_min),
                'n_signals': n
            })
        
        stats_df = grouped.apply(calc_stats, include_groups=False).reset_index()
        
        logger.info(f"  ✅ {len(stats_df):,}개 조합 통계 계산 완료")
        
        # 최소 표본 수 필터링
        n_min = 10
        stats_df = stats_df[stats_df['n_signals'] >= n_min]
        logger.info(f"  📋 최소 표본({n_min}개) 충족: {len(stats_df):,}개")
        
        if len(stats_df) == 0:
            logger.warning("표본 수 기준을 충족하는 조합이 없습니다.")
            return 0

        # 🔥 [벡터화] 배치 INSERT
        logger.info("  💾 통계 저장 중 (배치)...")
        
        now_ts = int(datetime.now().timestamp())
        stats_df['last_updated'] = now_ts
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # UPSERT 쿼리
            insert_query = """
                INSERT INTO strategy_label_stats
                (coin, interval, regime_tag, strategy_id,
                 rmax_mean, rmax_median, rmax_p75, rmax_p90,
                 rmin_mean, rmin_median, rmin_p25, rmin_p10,
                 kmax_mean, kmax_median, kmin_mean, kmin_median,
                 pf, win_rate, mdd, n_signals, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(coin, interval, regime_tag, strategy_id)
                DO UPDATE SET
                    rmax_mean = excluded.rmax_mean,
                    rmax_median = excluded.rmax_median,
                    rmax_p75 = excluded.rmax_p75,
                    rmax_p90 = excluded.rmax_p90,
                    rmin_mean = excluded.rmin_mean,
                    rmin_median = excluded.rmin_median,
                    rmin_p25 = excluded.rmin_p25,
                    rmin_p10 = excluded.rmin_p10,
                    kmax_mean = excluded.kmax_mean,
                    kmax_median = excluded.kmax_median,
                    kmin_mean = excluded.kmin_mean,
                    kmin_median = excluded.kmin_median,
                    pf = excluded.pf,
                    win_rate = excluded.win_rate,
                    mdd = excluded.mdd,
                    n_signals = excluded.n_signals,
                    last_updated = excluded.last_updated
            """
            
            # 데이터 준비
            data = [
                (row['coin'], row['interval'], row['regime_tag'], row['strategy_id'],
                 row['rmax_mean'], row['rmax_median'], row['rmax_p75'], row['rmax_p90'],
                 row['rmin_mean'], row['rmin_median'], row['rmin_p25'], row['rmin_p10'],
                 row['kmax_mean'], row['kmax_median'], row['kmin_mean'], row['kmin_median'],
                 row['pf'], row['win_rate'], row['mdd'], row['n_signals'], row['last_updated'])
                for _, row in stats_df.iterrows()
            ]
            
            cursor.executemany(insert_query, data)
            conn.commit()

        logger.info(f"\n🎉 통계 생성 완료: 총 {len(stats_df):,}개")

        return len(stats_df)

def main():
    """메인 실행 함수"""
    generator = StatsGenerator()
    count = generator.generate_all_stats()

    # 결과 검증
    logger.info("\n📊 통계 검증:")

    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM strategy_label_stats")
            total = cursor.fetchone()[0]
            logger.info(f"  총 통계 레코드: {total}개")

            cursor.execute("SELECT coin, interval, COUNT(*) FROM strategy_label_stats GROUP BY coin, interval")
            logger.info("\n  코인 x 인터벌별:")
            for row in cursor.fetchall():
                logger.info(f"    {row[0]} {row[1]}: {row[2]}개")

            # 통계 샘플
            cursor.execute("""
                SELECT coin, interval, strategy_id, n_signals, pf, win_rate, rmax_mean, rmin_mean
                FROM strategy_label_stats
                ORDER BY pf DESC
                LIMIT 5
            """)
            logger.info("\n  상위 PF 전략 (TOP 5):")
            for row in cursor.fetchall():
                logger.info(f"    {row[0]} {row[1]} {row[2][:30]}...: PF={row[4]:.2f}, WR={row[5]*100:.1f}%, r_max={row[6]*100:.2f}%, r_min={row[7]*100:.2f}% (n={row[3]})")
        except sqlite3.OperationalError:
            logger.warning("통계 테이블 조회 실패 (생성되지 않았을 수 있음)")

if __name__ == "__main__":
    main()
