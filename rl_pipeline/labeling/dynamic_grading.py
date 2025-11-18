"""
동적 등급화 시스템 - 백분위 기반 (고정 임계값 없음)
strategy_label_stats → strategy_grades
"""
import sys
import os
import logging
import sqlite3
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

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
class GradeConfig:
    """등급 설정"""
    # 가중치
    w1_pf: float = 0.35
    w2_rmax_mean: float = 0.25
    w3_rmin_mean: float = 0.20
    w4_hitrate: float = 0.15
    w5_rmax_std: float = 0.03
    w6_latency: float = 0.02

    # 백분위 컷오프
    percentiles: Dict[str, float] = None

    # 최소 표본 수
    n_min: Dict[str, int] = None

    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = {
                'S': 95,  # 상위 5%
                'A': 85,  # 상위 15%
                'B': 70,  # 상위 30%
                'C': 50,  # 상위 50%
                'D': 20,  # 하위 30%
                'F': 10   # 하위 10%
            }

        if self.n_min is None:
            self.n_min = {
                '15m': 120,
                '30m': 100,
                '240m': 60,
                '1d': 40
            }

class DynamicGrader:
    """동적 등급화기"""

    def __init__(self, config: Optional[GradeConfig] = None):
        self.config = config or GradeConfig()

    def calculate_grade_score(self, stats: Dict) -> float:
        """
        grade_score 계산

        Formula:
            grade_score =
              w1 * PF +
              w2 * mean(r_max) -
              w3 * mean(|r_min|) +
              w4 * win_rate -
              w5 * (latency penalty) -
              w6 * (volatility penalty)
        """
        pf = stats['pf']
        rmax_mean = stats['rmax_mean']
        rmin_mean = abs(stats['rmin_mean'])  # 절대값
        win_rate = stats['win_rate']
        kmax_mean = stats['kmax_mean']

        # Latency penalty (빠를수록 좋음, 정규화)
        # 15m 기준 40캔들이 최대, 빠를수록 점수 높음
        max_horizon = 40
        latency_penalty = kmax_mean / max_horizon

        # Volatility는 rmax의 표준편차를 사용하려면 추가 계산 필요
        # 여기서는 간단히 0으로 설정 (추후 개선 가능)
        volatility_penalty = 0.0

        score = (
            self.config.w1_pf * pf +
            self.config.w2_rmax_mean * rmax_mean * 100 -  # % 단위로 변환
            self.config.w3_rmin_mean * rmin_mean * 100 +
            self.config.w4_hitrate * win_rate -
            self.config.w5_rmax_std * volatility_penalty -
            self.config.w6_latency * latency_penalty
        )

        return score

    def assign_grade(self, score: float, percentiles_map: Dict[str, float]) -> str:
        """
        백분위를 기반으로 등급 할당

        Args:
            score: grade_score
            percentiles_map: {'grade': percentile_value, ...}

        Returns:
            등급 (S/A/B/C/D/F)
        """
        # 등급 순서대로 체크 (S → A → B → C → D → F)
        grade_order = ['S', 'A', 'B', 'C', 'D', 'F']

        for grade in grade_order:
            if grade in percentiles_map:
                if score >= percentiles_map[grade]:
                    return grade

        return 'F'

    def grade_by_regime_interval(self, regime_tag: str, interval: str) -> int:
        """
        레짐×인터벌별로 독립적으로 등급화

        Returns:
            등급 부여된 전략 수
        """
        logger.info(f"\n📊 {regime_tag} × {interval} 등급화 중...")

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 해당 그룹의 모든 통계 조회
            cursor.execute("""
                SELECT coin, interval, regime_tag, strategy_id,
                       rmax_mean, rmax_median, rmax_p75, rmax_p90,
                       rmin_mean, rmin_median, rmin_p25, rmin_p10,
                       kmax_mean, kmax_median, kmin_mean, kmin_median,
                       pf, win_rate, mdd, n_signals
                FROM strategy_label_stats
                WHERE regime_tag = ? AND interval = ?
            """, (regime_tag, interval))

            rows = cursor.fetchall()

            if len(rows) == 0:
                logger.warning(f"  ⚠️ {regime_tag} {interval}: 통계 없음")
                return 0

            logger.info(f"  ✅ {len(rows)}개 전략 발견")

            # 최소 표본 수 체크
            n_min = self.config.n_min.get(interval, 50)

            # grade_score 계산
            scores = []
            valid_strategies = []

            for row in rows:
                stats = {
                    'coin': row[0],
                    'interval': row[1],
                    'regime_tag': row[2],
                    'strategy_id': row[3],
                    'rmax_mean': row[4],
                    'rmax_median': row[5],
                    'rmax_p75': row[6],
                    'rmax_p90': row[7],
                    'rmin_mean': row[8],
                    'rmin_median': row[9],
                    'rmin_p25': row[10],
                    'rmin_p10': row[11],
                    'kmax_mean': row[12],
                    'kmax_median': row[13],
                    'kmin_mean': row[14],
                    'kmin_median': row[15],
                    'pf': row[16],
                    'win_rate': row[17],
                    'mdd': row[18],
                    'n_signals': row[19]
                }

                # 표본 수 체크
                if stats['n_signals'] < n_min:
                    logger.debug(f"    ⚠️ {stats['strategy_id'][:30]}...: 표본 부족 ({stats['n_signals']}개)")
                    continue

                score = self.calculate_grade_score(stats)
                scores.append(score)
                valid_strategies.append(stats)

            if len(scores) == 0:
                logger.warning(f"  ⚠️ {regime_tag} {interval}: 유효한 전략 없음 (표본 부족)")
                return 0

            logger.info(f"  ✅ 유효 전략: {len(scores)}개")

            # numpy 배열로 변환
            scores_array = np.array(scores)

            # 백분위 계산
            percentiles_values = {}
            for grade, pct in self.config.percentiles.items():
                percentiles_values[grade] = np.percentile(scores_array, pct)

            logger.info(f"  📊 백분위 컷오프:")
            for grade in ['S', 'A', 'B', 'C', 'D', 'F']:
                if grade in percentiles_values:
                    logger.info(f"    {grade}: {percentiles_values[grade]:.3f}")

            # 등급 할당 및 저장
            saved_count = 0

            for stats, score in zip(valid_strategies, scores):
                grade = self.assign_grade(score, percentiles_values)

                # explain JSON 생성
                percentile_rank = (scores_array < score).sum() / len(scores_array) * 100
                explain = {
                    'pf': round(stats['pf'], 3),
                    'win_rate': round(stats['win_rate'], 3),
                    'rmax_mean': round(stats['rmax_mean'], 4),
                    'rmin_mean': round(stats['rmin_mean'], 4),
                    'n_signals': stats['n_signals'],
                    'percentile': round(percentile_rank, 1)
                }

                # strategy_grades 저장
                cursor.execute("""
                    INSERT INTO strategy_grades
                    (strategy_id, interval, regime_tag, grade_score, grade, explain)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_id, interval, regime_tag)
                    DO UPDATE SET
                        grade_score = excluded.grade_score,
                        grade = excluded.grade,
                        explain = excluded.explain
                """, (
                    stats['strategy_id'],
                    stats['interval'],
                    stats['regime_tag'],
                    score,
                    grade,
                    json.dumps(explain)
                ))

                saved_count += 1

            conn.commit()

            logger.info(f"  💾 {saved_count}개 등급 저장 완료")

            return saved_count

    def grade_all(self) -> Dict[str, int]:
        """전체 등급화 실행"""
        logger.info("🚀 동적 등급화 시작\n")

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 모든 (regime_tag, interval) 조합 조회
            cursor.execute("""
                SELECT DISTINCT regime_tag, interval
                FROM strategy_label_stats
                ORDER BY regime_tag, interval
            """)

            combinations = cursor.fetchall()

        logger.info(f"✅ {len(combinations)}개 레짐×인터벌 조합 발견\n")

        results = {}
        total_graded = 0

        for regime_tag, interval in combinations:
            key = f"{regime_tag}_{interval}"
            count = self.grade_by_regime_interval(regime_tag, interval)
            results[key] = count
            total_graded += count

        logger.info(f"\n🎉 등급화 완료: 총 {total_graded}개 전략 등급 부여")

        return results

def main():
    """메인 실행 함수"""
    grader = DynamicGrader()
    results = grader.grade_all()

    # 결과 검증
    logger.info("\n📊 등급화 결과 검증:")

    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM strategy_grades")
        total = cursor.fetchone()[0]
        logger.info(f"  총 등급 레코드: {total}개")

        cursor.execute("SELECT grade, COUNT(*) FROM strategy_grades GROUP BY grade ORDER BY grade")
        logger.info("\n  등급 분포:")
        for row in cursor.fetchall():
            logger.info(f"    {row[0]}: {row[1]}개")

        # 상위 등급 샘플
        cursor.execute("""
            SELECT strategy_id, interval, regime_tag, grade, grade_score, explain
            FROM strategy_grades
            WHERE grade IN ('S', 'A')
            ORDER BY grade_score DESC
            LIMIT 5
        """)
        logger.info("\n  상위 등급 전략 (S/A 등급):")
        for row in cursor.fetchall():
            explain = json.loads(row[5]) if row[5] else {}
            logger.info(f"    [{row[3]}] {row[0][:40]}... (score={row[4]:.3f})")
            logger.info(f"         {row[2]} {row[1]}: PF={explain.get('pf', 0):.2f}, WR={explain.get('win_rate', 0)*100:.1f}%, n={explain.get('n_signals', 0)}")

if __name__ == "__main__":
    main()
