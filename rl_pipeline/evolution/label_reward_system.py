"""
Label-Based Reward System - Phase 3
라벨링 통계를 RL 보상으로 변환하는 시스템
"""
import sys
import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

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
class RewardWeights:
    """보상 가중치 설정"""
    w_pf: float = 0.3          # Profit Factor
    w_rmax: float = 0.25       # 평균 최대수익
    w_rmin: float = 0.2        # 평균 최대손실 (페널티)
    w_win_rate: float = 0.15   # 승률
    w_latency: float = 0.05    # 지연 시간 (빠를수록 좋음)
    w_sample: float = 0.05     # 표본 수 (많을수록 신뢰)

@dataclass
class StrategyReward:
    """전략 보상 정보"""
    strategy_id: str
    coin: str
    interval: str
    regime_tag: str

    # 보상 구성요소
    pf_reward: float
    rmax_reward: float
    rmin_penalty: float
    win_rate_reward: float
    latency_penalty: float
    sample_bonus: float

    # 최종 보상
    total_reward: float
    normalized_reward: float  # 0~1 범위로 정규화

    # 추가 정보
    grade: str
    confidence: float

class LabelRewardSystem:
    """라벨 기반 보상 시스템"""

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()

        # 정규화 파라미터 (경험적 값)
        self.norm_params = {
            'pf_max': 3.0,        # PF 3.0 이상은 최대 보상
            'rmax_max': 0.15,     # 15% 이상은 최대 보상
            'rmin_max': 0.05,     # 5% 손실까지는 페널티 중간
            'kmax_ref': 20,       # 20캔들이 기준
            'n_ref': 100          # 100개 샘플이 기준
        }

    def calculate_reward(self,
                        coin: str,
                        interval: str,
                        regime_tag: str,
                        strategy_id: str) -> Optional[StrategyReward]:
        """
        전략의 보상 계산

        Args:
            coin: 코인명
            interval: 인터벌
            regime_tag: 레짐 태그
            strategy_id: 전략 ID

        Returns:
            StrategyReward 또는 None (통계 없음)
        """
        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 통계 조회
            cursor.execute("""
                SELECT rmax_mean, rmin_mean, kmax_mean, pf, win_rate, n_signals
                FROM strategy_label_stats
                WHERE coin = ? AND interval = ? AND regime_tag = ? AND strategy_id = ?
            """, (coin, interval, regime_tag, strategy_id))

            stats_row = cursor.fetchone()

            if not stats_row:
                return None

            rmax_mean, rmin_mean, kmax_mean, pf, win_rate, n_signals = stats_row

            # 등급 조회
            cursor.execute("""
                SELECT grade, grade_score
                FROM strategy_grades
                WHERE strategy_id = ? AND interval = ? AND regime_tag = ?
            """, (strategy_id, interval, regime_tag))

            grade_row = cursor.fetchone()
            grade = grade_row[0] if grade_row else 'F'
            grade_score = grade_row[1] if grade_row else 0.0

        # 1. PF 보상 (1.0 이상이 좋음)
        pf_reward = self._calculate_pf_reward(pf)

        # 2. r_max 보상 (높을수록 좋음)
        rmax_reward = self._calculate_rmax_reward(rmax_mean)

        # 3. r_min 페널티 (손실 작을수록 좋음)
        rmin_penalty = self._calculate_rmin_penalty(rmin_mean)

        # 4. Win rate 보상
        win_rate_reward = self._calculate_win_rate_reward(win_rate)

        # 5. Latency 페널티 (빠를수록 좋음)
        latency_penalty = self._calculate_latency_penalty(kmax_mean)

        # 6. Sample 보너스 (많을수록 신뢰)
        sample_bonus = self._calculate_sample_bonus(n_signals)

        # 총 보상 계산
        total_reward = (
            self.weights.w_pf * pf_reward +
            self.weights.w_rmax * rmax_reward -
            self.weights.w_rmin * rmin_penalty +
            self.weights.w_win_rate * win_rate_reward -
            self.weights.w_latency * latency_penalty +
            self.weights.w_sample * sample_bonus
        )

        # 정규화 (0~1 범위)
        # total_reward는 대략 -0.5 ~ 1.5 범위
        normalized_reward = (total_reward + 0.5) / 2.0
        normalized_reward = max(0.0, min(1.0, normalized_reward))

        # 신뢰도 계산
        confidence = self._calculate_confidence(n_signals, grade)

        return StrategyReward(
            strategy_id=strategy_id,
            coin=coin,
            interval=interval,
            regime_tag=regime_tag,
            pf_reward=pf_reward,
            rmax_reward=rmax_reward,
            rmin_penalty=rmin_penalty,
            win_rate_reward=win_rate_reward,
            latency_penalty=latency_penalty,
            sample_bonus=sample_bonus,
            total_reward=total_reward,
            normalized_reward=normalized_reward,
            grade=grade,
            confidence=confidence
        )

    def _calculate_pf_reward(self, pf: float) -> float:
        """PF 보상 계산"""
        # PF 1.0 = 0 보상, 3.0+ = 1.0 보상
        return min((pf - 1.0) / (self.norm_params['pf_max'] - 1.0), 1.0)

    def _calculate_rmax_reward(self, rmax_mean: float) -> float:
        """r_max 보상 계산"""
        # 0% = 0 보상, 15%+ = 1.0 보상
        return min(rmax_mean / self.norm_params['rmax_max'], 1.0)

    def _calculate_rmin_penalty(self, rmin_mean: float) -> float:
        """r_min 페널티 계산"""
        # rmin은 음수, 절대값이 클수록 페널티
        # 0% = 0 페널티, 5%+ = 1.0 페널티
        return min(abs(rmin_mean) / self.norm_params['rmin_max'], 1.0)

    def _calculate_win_rate_reward(self, win_rate: float) -> float:
        """승률 보상 계산"""
        # 50% = 0 보상, 100% = 1.0 보상
        return max((win_rate - 0.5) / 0.5, 0.0)

    def _calculate_latency_penalty(self, kmax_mean: float) -> float:
        """지연 페널티 계산"""
        # 빠를수록 좋음
        # 0캔들 = 0 페널티, 20캔들+ = 1.0 페널티
        return min(kmax_mean / self.norm_params['kmax_ref'], 1.0)

    def _calculate_sample_bonus(self, n_signals: int) -> float:
        """표본 보너스 계산"""
        # 많을수록 신뢰
        # 0개 = 0 보너스, 100개+ = 1.0 보너스
        return min(n_signals / self.norm_params['n_ref'], 1.0)

    def _calculate_confidence(self, n_signals: int, grade: str) -> float:
        """신뢰도 계산"""
        # 표본 수 기반
        n_score = min(n_signals / 200.0, 1.0)

        # 등급 기반
        grade_scores = {'S': 1.0, 'A': 0.9, 'B': 0.8, 'C': 0.7, 'D': 0.5, 'F': 0.3}
        grade_score = grade_scores.get(grade, 0.5)

        # 가중 평균
        confidence = 0.6 * n_score + 0.4 * grade_score

        return confidence

    def calculate_batch_rewards(self,
                                strategies: list[Tuple[str, str, str, str]]
                               ) -> Dict[str, StrategyReward]:
        """
        여러 전략의 보상 일괄 계산

        Args:
            strategies: [(coin, interval, regime_tag, strategy_id), ...]

        Returns:
            {strategy_id: StrategyReward, ...}
        """
        results = {}

        for coin, interval, regime_tag, strategy_id in strategies:
            reward = self.calculate_reward(coin, interval, regime_tag, strategy_id)
            if reward:
                results[strategy_id] = reward

        return results

def main():
    """테스트 함수"""
    logger.info("🚀 Label-Based Reward System 테스트\n")

    reward_system = LabelRewardSystem()

    # 상위 등급 전략으로 테스트
    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT s.coin, s.interval, s.regime_tag, s.strategy_id, g.grade
            FROM strategy_label_stats s
            JOIN strategy_grades g
                ON s.strategy_id = g.strategy_id
                AND s.interval = g.interval
                AND s.regime_tag = g.regime_tag
            WHERE g.grade IN ('S', 'A', 'B', 'C')
            ORDER BY g.grade_score DESC
            LIMIT 10
        """)

        test_strategies = cursor.fetchall()

    logger.info(f"✅ {len(test_strategies)}개 전략 테스트\n")

    for coin, interval, regime_tag, strategy_id, grade in test_strategies:
        reward = reward_system.calculate_reward(coin, interval, regime_tag, strategy_id)

        if reward:
            logger.info(f"📊 [{grade}] {coin} {interval} {regime_tag}")
            logger.info(f"   전략: {strategy_id[:50]}...")
            logger.info(f"   보상 구성:")
            logger.info(f"     PF: {reward.pf_reward:.3f}")
            logger.info(f"     R_max: {reward.rmax_reward:.3f}")
            logger.info(f"     R_min penalty: {reward.rmin_penalty:.3f}")
            logger.info(f"     Win rate: {reward.win_rate_reward:.3f}")
            logger.info(f"     Latency penalty: {reward.latency_penalty:.3f}")
            logger.info(f"     Sample bonus: {reward.sample_bonus:.3f}")
            logger.info(f"   ✅ Total Reward: {reward.total_reward:.3f}")
            logger.info(f"   ✅ Normalized: {reward.normalized_reward:.3f}")
            logger.info(f"   ✅ Confidence: {reward.confidence:.1%}\n")

    logger.info("🎉 테스트 완료!")

if __name__ == "__main__":
    main()
