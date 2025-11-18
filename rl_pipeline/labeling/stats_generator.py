"""
통계 생성기 - Phase 1 라벨링 통계
strategy_signal_labels → strategy_label_stats
"""
import sys
import os
import logging
import sqlite3
import numpy as np
from typing import Dict, List, Tuple
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

    def __init__(self, strategy_db_path: str = '/workspace/data_storage/rl_strategies.db'):
        self.strategy_db_path = strategy_db_path

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
        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 라벨 데이터 조회
            cursor.execute("""
                SELECT r_max, r_min, k_max, k_min
                FROM strategy_signal_labels
                WHERE coin = ? AND interval = ? AND regime_tag = ? AND strategy_id = ?
            """, (coin, interval, regime_tag, strategy_id))

            rows = cursor.fetchall()

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
        pool = get_strategy_db_pool()
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
        """모든 통계 생성"""
        logger.info("🚀 통계 생성 시작\n")

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 모든 (coin, interval, regime_tag, strategy_id) 조합 조회
            cursor.execute("""
                SELECT DISTINCT coin, interval, regime_tag, strategy_id
                FROM strategy_signal_labels
                ORDER BY coin, interval, regime_tag, strategy_id
            """)

            combinations = cursor.fetchall()

        logger.info(f"✅ {len(combinations)}개 조합 발견\n")

        total_saved = 0

        for coin, interval, regime_tag, strategy_id in combinations:
            try:
                stats = self.calculate_stats(coin, interval, regime_tag, strategy_id)

                # 최소 표본 수 체크 (N_min)
                n_min = 10  # 최소 10개 신호 필요

                if stats.n_signals < n_min:
                    logger.debug(f"  ⚠️ {coin} {interval} {regime_tag} {strategy_id[:30]}...: 표본 부족 ({stats.n_signals}개)")
                    continue

                self.save_stats(stats)
                total_saved += 1

                if total_saved % 100 == 0:
                    logger.info(f"  📊 {total_saved}개 통계 저장 완료...")

            except Exception as e:
                logger.error(f"  ❌ {coin} {interval} {regime_tag} {strategy_id} 실패: {e}")

        logger.info(f"\n🎉 통계 생성 완료: 총 {total_saved}개")

        return total_saved

def main():
    """메인 실행 함수"""
    generator = StatsGenerator()
    count = generator.generate_all_stats()

    # 결과 검증
    logger.info("\n📊 통계 검증:")

    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

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

if __name__ == "__main__":
    main()
