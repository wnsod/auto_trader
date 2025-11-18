"""
전략 등급 계산 통합 모듈
모든 등급 계산 로직을 한 곳에서 관리

개선 사항:
- 예측 정확도 중심 평가
- 코인-인터벌-레짐별 상대평가
- 가중치 기반 종합 점수
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from rl_pipeline.core.types import StrategyMetrics
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GradeCriteria:
    """등급 기준 설정"""
    profit_percent_min: float
    win_rate_min: float
    sharpe_min: float
    max_dd_max: float
    profit_factor_min: float


class StrategyGrading:
    """전략 등급 계산 통합 클래스"""
    
    # 등급 기준 정의 (암호화폐 트레이딩에 최적화된 현실적 기준)
    GRADE_CRITERIA = {
        'S': GradeCriteria(
            profit_percent_min=5.0,      # 월 5%+ 수익
            win_rate_min=0.45,           # 승률 45%+ (암호화폐는 승률보다 손익비 중요)
            sharpe_min=1.2,              # Sharpe 1.2+
            max_dd_max=0.15,             # 최대 낙폭 15% 이하
            profit_factor_min=2.0        # 수익팩터 2.0+
        ),
        'A': GradeCriteria(
            profit_percent_min=2.0,      # 월 2%+ 수익
            win_rate_min=0.40,           # 승률 40%+
            sharpe_min=0.8,              # Sharpe 0.8+
            max_dd_max=0.20,             # 최대 낙폭 20% 이하
            profit_factor_min=1.5        # 수익팩터 1.5+
        ),
        'B': GradeCriteria(
            profit_percent_min=0.5,      # 월 0.5%+ 수익 (손실 방지)
            win_rate_min=0.35,           # 승률 35%+
            sharpe_min=0.3,              # Sharpe 0.3+
            max_dd_max=0.30,             # 최대 낙폭 30% 이하
            profit_factor_min=1.2        # 수익팩터 1.2+
        ),
        'C': GradeCriteria(
            profit_percent_min=-1.0,     # 월 -1% 이내 (소폭 손실 허용)
            win_rate_min=0.30,           # 승률 30%+
            sharpe_min=0.0,              # Sharpe 0+ (최소한 랜덤보다 나음)
            max_dd_max=0.40,             # 최대 낙폭 40% 이하
            profit_factor_min=1.0        # 수익팩터 1.0+
        ),
        'D': GradeCriteria(
            profit_percent_min=-3.0,     # 월 -3% 이내
            win_rate_min=0.25,           # 승률 25%+
            sharpe_min=-0.5,             # Sharpe -0.5+ (큰 마이너스 아님)
            max_dd_max=0.50,             # 최대 낙폭 50% 이하
            profit_factor_min=0.7        # 수익팩터 0.7+
        ),
    }
    
    @staticmethod
    def calculate_grade(
        profit_percent: float,
        win_rate: float,
        sharpe: float,
        max_dd: float,
        profit_factor: float,
        is_initial_learning: bool = False,
        trades_count: int = 0
    ) -> str:
        """
        전략 등급 계산 (통합 로직)

        Args:
            profit_percent: 수익률 (%)
            win_rate: 승률 (0.0 ~ 1.0)
            sharpe: 샤프 비율
            max_dd: 최대 낙폭 (0.0 ~ 1.0)
            profit_factor: 수익 팩터
            is_initial_learning: 초기 학습 단계 여부
            trades_count: 거래 횟수 (통계적 유의성 판단)

        Returns:
            등급 ('S', 'A', 'B', 'C', 'D', 'F')
        """
        # 거래 횟수가 너무 적으면 신뢰할 수 없음
        if trades_count > 0 and trades_count < 5:
            logger.warning(f"거래 횟수 부족 ({trades_count}건) - 통계적 신뢰도 낮음")
            # 거래가 너무 적으면 더 엄격하게 평가
            is_initial_learning = True

        if is_initial_learning:
            return StrategyGrading._calculate_initial_learning_grade(
                win_rate, profit_factor
            )
        else:
            return StrategyGrading._calculate_standard_grade(
                profit_percent, win_rate, sharpe, max_dd, profit_factor
            )
    
    @staticmethod
    def calculate_grade_from_metrics(
        metrics: StrategyMetrics,
        is_initial_learning: bool = False
    ) -> str:
        """StrategyMetrics 객체로부터 등급 계산"""
        return StrategyGrading.calculate_grade(
            profit_percent=metrics.profit_percent,
            win_rate=metrics.win_rate,
            sharpe=metrics.sharpe_ratio,
            max_dd=metrics.max_drawdown,
            profit_factor=metrics.profit_factor,
            is_initial_learning=is_initial_learning
        )
    
    @staticmethod
    def _calculate_standard_grade(
        profit_percent: float,
        win_rate: float,
        sharpe: float,
        max_dd: float,
        profit_factor: float
    ) -> str:
        """표준 등급 계산"""
        # S등급부터 순차적으로 확인
        for grade in ['S', 'A', 'B', 'C', 'D']:
            criteria = StrategyGrading.GRADE_CRITERIA[grade]
            
            if (profit_percent >= criteria.profit_percent_min and
                win_rate >= criteria.win_rate_min and
                sharpe >= criteria.sharpe_min and
                max_dd <= criteria.max_dd_max and
                profit_factor >= criteria.profit_factor_min):
                return grade
        
        # 모든 기준을 만족하지 않으면 F등급
        return 'F'
    
    @staticmethod
    def _calculate_initial_learning_grade(
        win_rate: float,
        profit_factor: float
    ) -> str:
        """
        초기 학습 모드: 더 관대한 기준
        거래 횟수가 적거나 학습 초기 단계에서 사용
        """
        # 수익팩터가 1.0 이상이면서 승률이 합리적이면 좋은 평가
        if profit_factor >= 1.5 and win_rate >= 0.35:
            return 'B'
        elif profit_factor >= 1.2 and win_rate >= 0.30:
            return 'C'
        elif profit_factor >= 0.9 and win_rate >= 0.25:
            return 'D'
        # 수익팩터만으로도 어느 정도 평가
        elif profit_factor >= 1.0:
            return 'D'
        else:
            return 'F'
    
    @staticmethod
    def get_grade_score(grade: str) -> float:
        """등급을 점수로 변환 (0.0 ~ 1.0)"""
        grade_scores = {
            'S': 1.0,
            'A': 0.8,
            'B': 0.6,
            'C': 0.4,
            'D': 0.2,
            'F': 0.0,
            'UNKNOWN': 0.5
        }
        return grade_scores.get(grade, 0.5)
    
    @staticmethod
    def is_grade_acceptable(grade: str, min_grade: str = 'C') -> bool:
        """등급이 허용 가능한지 확인"""
        grade_order = ['F', 'D', 'C', 'B', 'A', 'S']
        try:
            grade_idx = grade_order.index(grade)
            min_idx = grade_order.index(min_grade)
            return grade_idx >= min_idx
        except ValueError:
            return False


@dataclass
class StrategyScore:
    """전략 종합 점수"""
    strategy_id: str
    coin: str
    interval: str
    regime: str

    # 결과 지표
    profit_percent: float
    win_rate: float
    sharpe: float
    max_dd: float
    profit_factor: float

    # 예측 정확도 지표
    prediction_accuracy: float  # 신호 방향과 실제 가격 방향 일치율
    signal_precision: float     # 신호 정밀도 (실제 수익 거래 비율)

    # 종합 점수
    composite_score: float

    # 등급
    grade: str = 'C'


class PredictionMetrics:
    """예측 정확도 계산 유틸리티"""

    @staticmethod
    def calculate_prediction_accuracy(
        win_rate: float,
        profit_factor: float,
        trades_count: int = 0
    ) -> float:
        """
        예측 정확도 계산 (신호 방향과 실제 가격 방향 일치율 추정)

        현재는 win_rate와 profit_factor를 기반으로 추정
        향후 실제 거래 데이터에서 방향 일치율을 직접 계산 가능

        Args:
            win_rate: 승률 (0.0 ~ 1.0)
            profit_factor: 수익 팩터
            trades_count: 거래 횟수

        Returns:
            예측 정확도 (0.0 ~ 1.0)
        """
        # 승률 기반 기본 정확도
        base_accuracy = win_rate

        # 수익팩터가 높으면 예측 품질이 좋다고 판단
        # profit_factor > 1.0: 수익 > 손실
        if profit_factor > 1.0:
            pf_bonus = min(0.15, (profit_factor - 1.0) * 0.1)
            base_accuracy = min(1.0, base_accuracy + pf_bonus)
        else:
            pf_penalty = max(-0.15, (profit_factor - 1.0) * 0.15)
            base_accuracy = max(0.0, base_accuracy + pf_penalty)

        # 거래 횟수가 적으면 신뢰도 낮춤
        if trades_count > 0 and trades_count < 10:
            confidence_factor = trades_count / 10.0
            base_accuracy *= confidence_factor

        return max(0.0, min(1.0, base_accuracy))

    @staticmethod
    def calculate_signal_precision(
        profit_percent: float,
        win_rate: float,
        trades_count: int = 0
    ) -> float:
        """
        신호 정밀도 계산 (신호 발생 후 실제 수익 거래 비율)

        Args:
            profit_percent: 수익률 (%)
            win_rate: 승률
            trades_count: 거래 횟수

        Returns:
            신호 정밀도 (0.0 ~ 1.0)
        """
        # 수익률이 양수이고 승률이 높으면 정밀도 높음
        if profit_percent > 0:
            precision = win_rate * (1.0 + min(0.2, profit_percent / 100.0))
        else:
            precision = win_rate * (1.0 + max(-0.3, profit_percent / 100.0))

        # 거래 횟수 고려
        if trades_count > 0 and trades_count < 10:
            precision *= (trades_count / 10.0)

        return max(0.0, min(1.0, precision))


class RelativeGrading:
    """상대평가 기반 등급 시스템"""

    # 등급 비율 설정 (옵션 A: 완만한 피라미드 구조)
    GRADE_PERCENTILES = {
        'S': (0.95, 1.00),   # 상위 5% (진짜 최고만)
        'A': (0.80, 0.95),   # 상위 5~20% (우수)
        'B': (0.45, 0.80),   # 상위 20~55% (주력 풀)
        'C': (0.20, 0.45),   # 상위 55~80% (평균)
        'D': (0.10, 0.20),   # 상위 80~90% (경고)
        'F': (0.00, 0.10),   # 하위 10% (제거 대상)
    }

    # 가중치 설정 (예측 정확도 중심)
    WEIGHTS = {
        'prediction_accuracy': 0.35,   # 예측 정확도 (가장 중요)
        'profit': 0.25,                # 수익률
        'signal_precision': 0.20,      # 신호 정밀도
        'sharpe': 0.10,                # 샤프 비율
        'max_dd': 0.10,                # 낙폭 (낮을수록 좋음)
    }

    @staticmethod
    def calculate_composite_score(
        profit_percent: float,
        win_rate: float,
        sharpe: float,
        max_dd: float,
        profit_factor: float,
        prediction_accuracy: Optional[float] = None,
        signal_precision: Optional[float] = None,
        trades_count: int = 0
    ) -> float:
        """
        종합 점수 계산 (가중치 기반)

        Returns:
            종합 점수 (0.0 ~ 1.0)
        """
        # 예측 정확도가 없으면 계산
        if prediction_accuracy is None:
            prediction_accuracy = PredictionMetrics.calculate_prediction_accuracy(
                win_rate, profit_factor, trades_count
            )

        if signal_precision is None:
            signal_precision = PredictionMetrics.calculate_signal_precision(
                profit_percent, win_rate, trades_count
            )

        # 각 지표를 0~1 범위로 정규화
        profit_normalized = RelativeGrading._normalize_profit(profit_percent)
        sharpe_normalized = RelativeGrading._normalize_sharpe(sharpe)
        dd_normalized = 1.0 - min(1.0, max(0.0, max_dd))  # 낙폭은 낮을수록 좋음

        # 가중치 합산
        weights = RelativeGrading.WEIGHTS
        composite_score = (
            weights['prediction_accuracy'] * prediction_accuracy +
            weights['profit'] * profit_normalized +
            weights['signal_precision'] * signal_precision +
            weights['sharpe'] * sharpe_normalized +
            weights['max_dd'] * dd_normalized
        )

        return max(0.0, min(1.0, composite_score))

    @staticmethod
    def _normalize_profit(profit_percent: float) -> float:
        """수익률을 0~1로 정규화 (-10% ~ +20% 범위 가정)"""
        # -10% = 0.0, 0% = 0.5, +20% = 1.0
        normalized = (profit_percent + 10.0) / 30.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _normalize_sharpe(sharpe: float) -> float:
        """Sharpe를 0~1로 정규화 (-1.0 ~ +3.0 범위 가정)"""
        # -1.0 = 0.0, 0.0 = 0.25, 1.0 = 0.5, 3.0 = 1.0
        normalized = (sharpe + 1.0) / 4.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def assign_grades_by_group(
        strategies: List[Dict[str, Any]],
        coin: str,
        interval: str,
        regime: str
    ) -> List[StrategyScore]:
        """
        코인-인터벌-레짐별 그룹 내 상대평가로 등급 부여

        Args:
            strategies: 전략 리스트 (딕셔너리)
            coin: 코인
            interval: 인터벌
            regime: 레짐

        Returns:
            StrategyScore 리스트 (등급 포함)
        """
        if not strategies:
            return []

        # 각 전략의 종합 점수 계산
        strategy_scores = []
        for s in strategies:
            try:
                # 필수 지표 추출
                profit_percent = s.get('profit', 0.0) / 100.0  # 달러 → %로 변환 (10000 = 100%)
                win_rate = s.get('win_rate', 0.0)
                sharpe = s.get('sharpe', 0.0)
                max_dd = s.get('max_dd', 0.5)
                profit_factor = s.get('profit_factor', 1.0)
                trades_count = s.get('trades', 0)

                # 예측 지표 추출 (없으면 계산)
                prediction_accuracy = s.get('prediction_accuracy')
                signal_precision = s.get('signal_precision')

                # 종합 점수 계산
                composite_score = RelativeGrading.calculate_composite_score(
                    profit_percent=profit_percent,
                    win_rate=win_rate,
                    sharpe=sharpe,
                    max_dd=max_dd,
                    profit_factor=profit_factor,
                    prediction_accuracy=prediction_accuracy,
                    signal_precision=signal_precision,
                    trades_count=trades_count
                )

                # 계산된 예측 지표 (없었던 경우)
                if prediction_accuracy is None:
                    prediction_accuracy = PredictionMetrics.calculate_prediction_accuracy(
                        win_rate, profit_factor, trades_count
                    )
                if signal_precision is None:
                    signal_precision = PredictionMetrics.calculate_signal_precision(
                        profit_percent, win_rate, trades_count
                    )

                score_obj = StrategyScore(
                    strategy_id=s.get('id', s.get('strategy_id', 'unknown')),
                    coin=coin,
                    interval=interval,
                    regime=regime,
                    profit_percent=profit_percent,
                    win_rate=win_rate,
                    sharpe=sharpe,
                    max_dd=max_dd,
                    profit_factor=profit_factor,
                    prediction_accuracy=prediction_accuracy,
                    signal_precision=signal_precision,
                    composite_score=composite_score,
                    grade='C'  # 초기값
                )
                strategy_scores.append(score_obj)

            except Exception as e:
                logger.warning(f"전략 점수 계산 실패: {e}")
                continue

        if not strategy_scores:
            return []

        # 종합 점수 기준 정렬 (내림차순)
        strategy_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # 백분위수 기반 등급 부여
        total_count = len(strategy_scores)
        for idx, score in enumerate(strategy_scores):
            percentile = 1.0 - (idx / total_count)  # 상위 비율

            # 등급 결정
            for grade, (lower, upper) in RelativeGrading.GRADE_PERCENTILES.items():
                if lower <= percentile < upper:
                    score.grade = grade
                    break

        return strategy_scores

    @staticmethod
    def batch_assign_grades(
        all_strategies: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]
    ) -> Dict[str, List[StrategyScore]]:
        """
        모든 코인-인터벌-레짐 조합에 대해 등급 부여

        Args:
            all_strategies: {coin: {interval: {regime: [strategies]}}}

        Returns:
            {group_key: [StrategyScore]}
        """
        all_scores = {}

        for coin, intervals in all_strategies.items():
            for interval, regimes in intervals.items():
                for regime, strategies in regimes.items():
                    group_key = f"{coin}-{interval}-{regime}"

                    scores = RelativeGrading.assign_grades_by_group(
                        strategies, coin, interval, regime
                    )

                    if scores:
                        all_scores[group_key] = scores
                        logger.info(
                            f"📊 {group_key}: {len(scores)}개 전략 등급 부여 완료 "
                            f"(S: {sum(1 for s in scores if s.grade == 'S')}, "
                            f"A: {sum(1 for s in scores if s.grade == 'A')}, "
                            f"B: {sum(1 for s in scores if s.grade == 'B')})"
                        )

        return all_scores

