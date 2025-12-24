"""
전략 등급 계산 통합 모듈
모든 등급 계산 로직을 한 곳에서 관리

개선 사항:
- MFE/MAE 기반 Gate Score (EntryScore) 도입 (최우선 평가 기준)
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


@dataclass
class MFEStats:
    """MFE/MAE 통계 데이터"""
    rmax_mean: float
    rmax_p90: float
    rmin_mean: float
    rmin_p10: float
    coverage_n: int


class MFEGrading:
    """MFE/MAE 기반 평가 (Absolute Zero 핵심 로직)"""
    
    @staticmethod
    def calculate_scores(stats: MFEStats, k: float = 1.5) -> Tuple[float, float, float]:
        """
        MFE/MAE 스코어 계산
        
        Returns:
            (entry_score, risk_score, edge_score)
        """
        # rmin은 음수이므로 절대값 처리
        abs_rmin_p10 = abs(stats.rmin_p10)
        abs_rmin_mean = abs(stats.rmin_mean)
        
        # 1. 진입 점수: EntryScore = P90(MFE) - k * abs(P10(MAE))
        # 상방 포텐셜에서 하방 리스크(k배 가중)를 뺀 값. 
        # 양수여야 진입 가치가 있음.
        entry_score = stats.rmax_p90 - (k * abs_rmin_p10)
        
        # 2. 리스크 점수: RiskScore = abs(P10(MAE))
        # 하방 꼬리 위험. 낮을수록 좋음.
        risk_score = abs_rmin_p10
        
        # 3. 기대값 점수: Edge = E[MFE] - k * abs(E[MAE])
        # 평균적인 우위.
        edge_score = stats.rmax_mean - (k * abs_rmin_mean)
        
        return entry_score, risk_score, edge_score

    @staticmethod
    def determine_grade(entry_score: float, risk_score: float, coverage_n: int) -> str:
        """MFE/MAE 기반 등급 산정"""
        
        if coverage_n < 20:
            return 'UNKNOWN' # 표본 부족
            
        # 등급 기준 (단위: %, 0.01 = 1%)
        # S급: EntryScore > 2% (수수료/리스크 제하고도 2% 먹을 구간) AND Risk < 3%
        if entry_score >= 0.02 and risk_score <= 0.03:
            return 'S'
        # A급: EntryScore > 1% AND Risk < 5%
        elif entry_score >= 0.01 and risk_score <= 0.05:
            return 'A'
        # B급: EntryScore > 0.5% (최소한의 엣지)
        elif entry_score >= 0.005:
            return 'B'
        # C급: EntryScore >= 0 (본전치기는 가능)
        elif entry_score >= 0.0:
            return 'C'
        # D급: EntryScore > -1% (약간 손해)
        elif entry_score > -0.01:
            return 'D'
        # F급: 진입하면 손해
        else:
            return 'F'
    
    @staticmethod
    def validate_direction_by_mfe(entry_score: float, min_entry_score: float = 0.0) -> bool:
        """
        🔥 MFE 기반 방향성 유효성 검증
        
        EntryScore가 기준 이상이면 해당 방향으로 진입할 가치가 있음.
        음수면 해당 방향은 손해 → 방향 무효.
        
        Args:
            entry_score: 진입 점수 (= P90(MFE) - k * abs(P10(MAE)))
            min_entry_score: 최소 요구 점수 (기본: 0, 손익분기)
            
        Returns:
            True if 방향 유효, False if 방향 무효 (neutral로 처리해야 함)
        """
        return entry_score >= min_entry_score
    
    @staticmethod
    def get_directional_confidence(entry_score: float, edge_score: float) -> float:
        """
        🔥 방향성 신뢰도 계산 (0.0 ~ 1.0)
        
        EntryScore와 Edge를 기반으로 해당 방향에 대한 신뢰도 계산.
        승률 개선의 핵심: 신뢰도가 낮으면 신호 억제.
        
        Args:
            entry_score: 진입 점수
            edge_score: 기대값 점수
            
        Returns:
            신뢰도 (0.0 ~ 1.0)
        """
        if entry_score < 0:
            return 0.0  # 손해 구간은 신뢰도 0
        
        # EntryScore 기반 기본 신뢰도 (0 ~ 3%를 0 ~ 1로 정규화)
        base_confidence = min(1.0, entry_score / 0.03)
        
        # Edge 보정 (양수면 보너스, 음수면 페널티)
        edge_bonus = 0.0
        if edge_score > 0:
            edge_bonus = min(0.2, edge_score / 0.02)  # 최대 20% 보너스
        elif edge_score < 0:
            edge_bonus = max(-0.3, edge_score / 0.01)  # 최대 30% 페널티
        
        confidence = max(0.0, min(1.0, base_confidence + edge_bonus))
        return round(confidence, 3)


def get_strategy_mfe_stats(strategy_id: str, db_path: str = None) -> Optional[MFEStats]:
    """
    🔥 전략의 MFE/MAE 통계 로드 (DB에서)
    
    Args:
        strategy_id: 전략 ID
        db_path: DB 경로 (없으면 환경변수 사용)
        
    Returns:
        MFEStats 또는 None
    """
    import os
    import sqlite3
    
    try:
        if db_path is None:
            db_path = os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
        
        if not db_path:
            return None
        
        # 디렉토리인 경우 common_strategies.db 사용
        if os.path.isdir(db_path):
            db_path = os.path.join(db_path, 'common_strategies.db')
        
        if not os.path.exists(db_path):
            return None
        
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            # strategy_label_stats 테이블에서 조회
            cursor.execute("""
                SELECT rmax_mean, rmax_p90, rmin_mean, rmin_p10, coverage_n
                FROM strategy_label_stats
                WHERE strategy_id = ?
            """, (strategy_id,))
            
            row = cursor.fetchone()
            if row:
                return MFEStats(
                    rmax_mean=row[0] or 0.0,
                    rmax_p90=row[1] or 0.0,
                    rmin_mean=row[2] or 0.0,
                    rmin_p10=row[3] or 0.0,
                    coverage_n=row[4] or 0
                )
        
        return None
        
    except Exception as e:
        logger.debug(f"⚠️ MFE 통계 로드 실패 ({strategy_id}): {e}")
        return None


class StrategyGrading:
    """전략 등급 계산 통합 클래스"""
    
    # 기존 기준 유지 (레거시 호환 및 보조 지표용)
    GRADE_CRITERIA = {
        'S': GradeCriteria(profit_percent_min=5.0, win_rate_min=0.45, sharpe_min=1.2, max_dd_max=0.15, profit_factor_min=2.0),
        'A': GradeCriteria(profit_percent_min=2.0, win_rate_min=0.40, sharpe_min=0.8, max_dd_max=0.20, profit_factor_min=1.5),
        'B': GradeCriteria(profit_percent_min=0.5, win_rate_min=0.35, sharpe_min=0.3, max_dd_max=0.30, profit_factor_min=1.2),
        'C': GradeCriteria(profit_percent_min=-1.0, win_rate_min=0.30, sharpe_min=0.0, max_dd_max=0.40, profit_factor_min=1.0),
        'D': GradeCriteria(profit_percent_min=-3.0, win_rate_min=0.25, sharpe_min=-0.5, max_dd_max=0.50, profit_factor_min=0.7),
    }
    
    @staticmethod
    def calculate_grade(
        profit_percent: float,
        win_rate: float,
        sharpe: float,
        max_dd: float,
        profit_factor: float,
        is_initial_learning: bool = False,
        trades_count: int = 0,
        mfe_stats: Optional[MFEStats] = None  # 🔥 MFE 통계 추가
    ) -> str:
        """
        전략 등급 계산 (통합 로직)
        
        우선순위:
        1. MFE/MAE 기반 등급 (데이터가 있을 경우)
        2. 기존 성과 기반 등급
        """
        
        # 1. MFE/MAE 기반 평가 (최우선)
        if mfe_stats and mfe_stats.coverage_n >= 20:
            entry_score, risk_score, _ = MFEGrading.calculate_scores(mfe_stats)
            mfe_grade = MFEGrading.determine_grade(entry_score, risk_score, mfe_stats.coverage_n)
            
            # MFE 등급이 유효하면 반환
            if mfe_grade != 'UNKNOWN':
                return mfe_grade

        # 2. 기존 성과 기반 평가 (Fallback)
        # 거래 횟수가 너무 적으면 신뢰할 수 없음
        if trades_count > 0 and trades_count < 5:
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
        """초기 학습 모드: 더 관대한 기준"""
        if profit_factor >= 1.5 and win_rate >= 0.35:
            return 'B'
        elif profit_factor >= 1.2 and win_rate >= 0.30:
            return 'C'
        elif profit_factor >= 0.9 and win_rate >= 0.25:
            return 'D'
        elif profit_factor >= 1.0:
            return 'D'
        else:
            return 'F'
    
    @staticmethod
    def get_grade_score(grade: str) -> float:
        """등급을 점수로 변환 (0.0 ~ 1.0)"""
        grade_scores = {
            'S': 1.0, 'A': 0.8, 'B': 0.6, 'C': 0.4, 'D': 0.2, 'F': 0.0, 'UNKNOWN': 0.5
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
    
    # 🔥 MFE/MAE 점수 추가
    entry_score: float = 0.0
    risk_score: float = 0.0

    # 종합 점수
    composite_score: float = 0.0

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
        """예측 정확도 계산 (신호 방향과 실제 가격 방향 일치율 추정)"""
        base_accuracy = win_rate
        if profit_factor > 1.0:
            pf_bonus = min(0.15, (profit_factor - 1.0) * 0.1)
            base_accuracy = min(1.0, base_accuracy + pf_bonus)
        else:
            pf_penalty = max(-0.15, (profit_factor - 1.0) * 0.15)
            base_accuracy = max(0.0, base_accuracy + pf_penalty)

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
        """신호 정밀도 계산 (신호 발생 후 실제 수익 거래 비율)"""
        if profit_percent > 0:
            precision = win_rate * (1.0 + min(0.2, profit_percent / 100.0))
        else:
            precision = win_rate * (1.0 + max(-0.3, profit_percent / 100.0))

        if trades_count > 0 and trades_count < 10:
            precision *= (trades_count / 10.0)

        return max(0.0, min(1.0, precision))


class RelativeGrading:
    """상대평가 기반 등급 시스템"""
    # (기존 로직 유지)
    GRADE_PERCENTILES = {
        'S': (0.95, 1.01),
        'A': (0.80, 0.95),
        'B': (0.45, 0.80),
        'C': (0.20, 0.45),
        'D': (0.10, 0.20),
        'F': (0.00, 0.10),
    }

    WEIGHTS = {
        'prediction_accuracy': 0.35,
        'profit': 0.25,
        'signal_precision': 0.20,
        'sharpe': 0.10,
        'max_dd': 0.10,
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
        trades_count: int = 0,
        entry_score: Optional[float] = None  # 🔥 MFE 점수 반영
    ) -> float:
        """종합 점수 계산 (가중치 기반)"""
        if prediction_accuracy is None:
            prediction_accuracy = PredictionMetrics.calculate_prediction_accuracy(
                win_rate, profit_factor, trades_count
            )

        if signal_precision is None:
            signal_precision = PredictionMetrics.calculate_signal_precision(
                profit_percent, win_rate, trades_count
            )

        profit_normalized = RelativeGrading._normalize_profit(profit_percent)
        sharpe_normalized = RelativeGrading._normalize_sharpe(sharpe)
        dd_normalized = 1.0 - min(1.0, max(0.0, max_dd))

        weights = RelativeGrading.WEIGHTS
        
        # 기본 점수
        composite_score = (
            weights['prediction_accuracy'] * prediction_accuracy +
            weights['profit'] * profit_normalized +
            weights['signal_precision'] * signal_precision +
            weights['sharpe'] * sharpe_normalized +
            weights['max_dd'] * dd_normalized
        )
        
        # 🔥 EntryScore가 있으면 보너스/페널티 적용
        if entry_score is not None:
            # entry_score는 대략 -0.05 ~ +0.05 범위
            # 0.01(1%) 당 10% 가산점
            bonus = entry_score * 10.0
            composite_score += bonus

        return max(0.0, min(1.0, composite_score))

    @staticmethod
    def _normalize_profit(profit_percent: float) -> float:
        normalized = (profit_percent + 10.0) / 30.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _normalize_sharpe(sharpe: float) -> float:
        normalized = (sharpe + 1.0) / 4.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def assign_grades_by_group(
        strategies: List[Dict[str, Any]],
        coin: str,
        interval: str,
        regime: str
    ) -> List[StrategyScore]:
        """코인-인터벌-레짐별 그룹 내 상대평가로 등급 부여"""
        if not strategies:
            return []

        strategy_scores = []
        for s in strategies:
            try:
                profit_percent = s.get('profit', 0.0) / 100.0
                win_rate = s.get('win_rate', 0.0)
                sharpe = s.get('sharpe', 0.0)
                max_dd = s.get('max_dd', 0.5)
                profit_factor = s.get('profit_factor', 1.0)
                trades_count = s.get('trades', 0)

                prediction_accuracy = s.get('prediction_accuracy')
                signal_precision = s.get('signal_precision')
                
                # MFE 통계 추출 (있다면)
                entry_score = s.get('entry_score')
                risk_score = s.get('risk_score', 0.0)

                composite_score = RelativeGrading.calculate_composite_score(
                    profit_percent=profit_percent,
                    win_rate=win_rate,
                    sharpe=sharpe,
                    max_dd=max_dd,
                    profit_factor=profit_factor,
                    prediction_accuracy=prediction_accuracy,
                    signal_precision=signal_precision,
                    trades_count=trades_count,
                    entry_score=entry_score
                )

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
                    entry_score=entry_score if entry_score is not None else 0.0,
                    risk_score=risk_score,
                    composite_score=composite_score,
                    grade='C'
                )
                strategy_scores.append(score_obj)

            except Exception as e:
                logger.warning(f"전략 점수 계산 실패: {e}")
                continue

        if not strategy_scores:
            return []

        strategy_scores.sort(key=lambda x: x.composite_score, reverse=True)

        total_count = len(strategy_scores)
        for idx, score in enumerate(strategy_scores):
            percentile = 1.0 - (idx / total_count)
            for grade, (lower, upper) in RelativeGrading.GRADE_PERCENTILES.items():
                if lower <= percentile < upper:
                    score.grade = grade
                    break

        return strategy_scores

    @staticmethod
    def batch_assign_grades(
        all_strategies: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]
    ) -> Dict[str, List[StrategyScore]]:
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
                        logger.info(f"📊 {group_key}: {len(scores)}개 등급 부여 완료")
        return all_scores
