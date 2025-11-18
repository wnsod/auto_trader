"""
Signal Runtime Adapter - Phase 4
실시간 신호 발생 시 라벨링 통계 기반으로 TP/SL/보유기간/사이징 자동 산출
"""
import sys
import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json

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
class SignalParameters:
    """신호 파라미터"""
    tp: float              # Take Profit
    sl: float              # Stop Loss
    target_hold: int       # 목표 보유 캔들 수
    size: float            # 포지션 크기 (1.0 = 기준)
    grade: str             # 전략 등급
    confidence: float      # 신뢰도 (0~1)

    # 추가 정보
    pf: float              # Profit Factor
    win_rate: float        # 승률
    n_signals: int         # 표본 수
    rmax_mean: float       # 평균 최대수익
    rmin_mean: float       # 평균 최대손실

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'tp': round(self.tp, 4),
            'sl': round(self.sl, 4),
            'target_hold': self.target_hold,
            'size': round(self.size, 2),
            'grade': self.grade,
            'confidence': round(self.confidence, 3),
            'pf': round(self.pf, 2),
            'win_rate': round(self.win_rate, 3),
            'n_signals': self.n_signals,
            'rmax_mean': round(self.rmax_mean, 4),
            'rmin_mean': round(self.rmin_mean, 4)
        }

@dataclass
class AdapterConfig:
    """Adapter 설정"""
    # TP/SL 보수성 (0.5 = 평균의 50% 사용)
    tp_conservatism: float = 0.7
    sl_conservatism: float = 1.3

    # 최소 기준
    min_n_signals: int = 30      # 최소 표본 수
    min_grade: str = 'C'         # 최소 등급
    min_pf: float = 1.0          # 최소 Profit Factor

    # 포지션 사이징
    base_size: float = 1.0       # 기본 포지션 크기
    grade_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.grade_multipliers is None:
            self.grade_multipliers = {
                'S': 1.5,
                'A': 1.3,
                'B': 1.1,
                'C': 1.0,
                'D': 0.7,
                'F': 0.5
            }

class SignalRuntimeAdapter:
    """실시간 신호 어댑터"""

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()

    def get_signal_parameters(self,
                             coin: str,
                             interval: str,
                             regime_tag: str,
                             strategy_id: str) -> Optional[SignalParameters]:
        """
        신호 파라미터 조회

        Args:
            coin: 코인명
            interval: 인터벌
            regime_tag: 레짐 태그
            strategy_id: 전략 ID

        Returns:
            SignalParameters 또는 None (기준 미달)
        """
        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 1. 통계 조회
            cursor.execute("""
                SELECT rmax_mean, rmax_median, rmax_p75, rmax_p90,
                       rmin_mean, rmin_median, rmin_p25, rmin_p10,
                       kmax_mean, kmax_median,
                       pf, win_rate, mdd, n_signals
                FROM strategy_label_stats
                WHERE coin = ? AND interval = ? AND regime_tag = ? AND strategy_id = ?
            """, (coin, interval, regime_tag, strategy_id))

            stats_row = cursor.fetchone()

            if not stats_row:
                logger.debug(f"통계 없음: {coin} {interval} {regime_tag} {strategy_id[:30]}...")
                return None

            # 통계 파싱
            stats = {
                'rmax_mean': stats_row[0],
                'rmax_median': stats_row[1],
                'rmax_p75': stats_row[2],
                'rmax_p90': stats_row[3],
                'rmin_mean': stats_row[4],
                'rmin_median': stats_row[5],
                'rmin_p25': stats_row[6],
                'rmin_p10': stats_row[7],
                'kmax_mean': stats_row[8],
                'kmax_median': stats_row[9],
                'pf': stats_row[10],
                'win_rate': stats_row[11],
                'mdd': stats_row[12],
                'n_signals': stats_row[13]
            }

            # 2. 등급 조회
            cursor.execute("""
                SELECT grade, grade_score, explain
                FROM strategy_grades
                WHERE strategy_id = ? AND interval = ? AND regime_tag = ?
            """, (strategy_id, interval, regime_tag))

            grade_row = cursor.fetchone()

            if not grade_row:
                logger.debug(f"등급 없음: {strategy_id[:30]}...")
                grade = 'F'
                grade_score = 0.0
            else:
                grade = grade_row[0]
                grade_score = grade_row[1]

        # 3. 필터링 검증
        if not self._validate_stats(stats, grade):
            return None

        # 4. TP/SL 계산
        tp = self._calculate_tp(stats)
        sl = self._calculate_sl(stats)

        # 5. 목표 보유 시간 계산
        target_hold = self._calculate_target_hold(stats)

        # 6. 포지션 사이즈 계산
        size = self._calculate_size(grade, stats)

        # 7. 신뢰도 계산
        confidence = self._calculate_confidence(stats, grade)

        return SignalParameters(
            tp=tp,
            sl=sl,
            target_hold=target_hold,
            size=size,
            grade=grade,
            confidence=confidence,
            pf=stats['pf'],
            win_rate=stats['win_rate'],
            n_signals=stats['n_signals'],
            rmax_mean=stats['rmax_mean'],
            rmin_mean=stats['rmin_mean']
        )

    def _validate_stats(self, stats: Dict, grade: str) -> bool:
        """통계 검증"""
        # 최소 표본 수
        if stats['n_signals'] < self.config.min_n_signals:
            logger.debug(f"표본 부족: {stats['n_signals']} < {self.config.min_n_signals}")
            return False

        # 최소 등급
        grade_order = ['F', 'D', 'C', 'B', 'A', 'S']
        if grade_order.index(grade) < grade_order.index(self.config.min_grade):
            logger.debug(f"등급 미달: {grade} < {self.config.min_grade}")
            return False

        # 최소 PF
        if stats['pf'] < self.config.min_pf:
            logger.debug(f"PF 미달: {stats['pf']} < {self.config.min_pf}")
            return False

        return True

    def _calculate_tp(self, stats: Dict) -> float:
        """Take Profit 계산"""
        # 보수적 접근: rmax_mean의 70% 사용 (설정 가능)
        # 또는 rmax_p75 사용 (75% 이상 도달한 값)
        tp = stats['rmax_p75'] * self.config.tp_conservatism

        # 최소 TP (0.5%)
        tp = max(tp, 0.005)

        return tp

    def _calculate_sl(self, stats: Dict) -> float:
        """Stop Loss 계산"""
        # 보수적 접근: rmin_p25의 130% 사용 (더 넓은 손절)
        # rmin은 음수이므로 절대값 사용
        sl = stats['rmin_p25'] * self.config.sl_conservatism

        # 최대 SL (-5%)
        sl = max(sl, -0.05)

        return sl

    def _calculate_target_hold(self, stats: Dict) -> int:
        """목표 보유 캔들 수 계산"""
        # kmax_median 사용 (중앙값이 평균보다 robust)
        target_hold = int(stats['kmax_median'])

        # 최소 2캔들, 최대 100캔들
        target_hold = max(2, min(target_hold, 100))

        return target_hold

    def _calculate_size(self, grade: str, stats: Dict) -> float:
        """포지션 크기 계산"""
        # 등급별 기본 배수
        grade_mult = self.config.grade_multipliers.get(grade, 1.0)

        # PF 기반 조정 (PF가 높을수록 크게)
        pf_mult = min(stats['pf'] / 2.0, 1.5)

        # 승률 기반 조정
        wr_mult = 0.5 + stats['win_rate']  # 승률 50% = 1.0x, 100% = 1.5x

        # 최종 사이즈
        size = self.config.base_size * grade_mult * pf_mult * wr_mult

        # 범위 제한 (0.3 ~ 2.0)
        size = max(0.3, min(size, 2.0))

        return size

    def _calculate_confidence(self, stats: Dict, grade: str) -> float:
        """신뢰도 계산 (0~1)"""
        # 표본 수 기반 (많을수록 높음)
        n_score = min(stats['n_signals'] / 200.0, 1.0)

        # 등급 기반
        grade_scores = {'S': 1.0, 'A': 0.9, 'B': 0.8, 'C': 0.7, 'D': 0.5, 'F': 0.3}
        grade_score = grade_scores.get(grade, 0.5)

        # PF 기반
        pf_score = min(stats['pf'] / 3.0, 1.0)

        # 승률 기반
        wr_score = stats['win_rate']

        # 가중 평균
        confidence = (
            n_score * 0.2 +
            grade_score * 0.3 +
            pf_score * 0.2 +
            wr_score * 0.3
        )

        return confidence

def main():
    """테스트 함수"""
    adapter = SignalRuntimeAdapter()

    logger.info("🚀 Signal Runtime Adapter 테스트\n")

    # 1. 상위 등급 전략 조회
    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT s.coin, s.interval, s.regime_tag, s.strategy_id, g.grade, g.grade_score
            FROM strategy_label_stats s
            JOIN strategy_grades g
                ON s.strategy_id = g.strategy_id
                AND s.interval = g.interval
                AND s.regime_tag = g.regime_tag
            WHERE g.grade IN ('S', 'A', 'B')
            ORDER BY g.grade_score DESC
            LIMIT 10
        """)

        test_strategies = cursor.fetchall()

    logger.info(f"✅ {len(test_strategies)}개 전략으로 테스트\n")

    # 2. 각 전략별 파라미터 조회
    success_count = 0

    for coin, interval, regime_tag, strategy_id, grade, score in test_strategies:
        logger.info(f"📊 [{grade}] {coin} {interval} {regime_tag}")
        logger.info(f"   전략: {strategy_id[:50]}... (score={score:.3f})")

        params = adapter.get_signal_parameters(coin, interval, regime_tag, strategy_id)

        if params:
            logger.info(f"   ✅ 파라미터 생성 성공:")
            logger.info(f"      TP: {params.tp*100:.2f}% | SL: {params.sl*100:.2f}%")
            logger.info(f"      Target Hold: {params.target_hold} candles")
            logger.info(f"      Size: {params.size:.2f}x | Confidence: {params.confidence:.1%}")
            logger.info(f"      PF: {params.pf:.2f} | WR: {params.win_rate*100:.1f}% | n: {params.n_signals}")
            success_count += 1
        else:
            logger.warning(f"   ⚠️ 파라미터 생성 실패 (필터링)")

        logger.info("")

    logger.info(f"🎉 테스트 완료: {success_count}/{len(test_strategies)} 성공")

if __name__ == "__main__":
    main()
