"""
Validation 모듈 전용 디버그 로거
- Walk-Forward 검증 추적
- 과적합 감지
- 레짐별 성능 비교
- A/B 테스트 결과
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .debug_logger import DebugLogger


class ValidationDebugger(DebugLogger):
    """Validation 모듈 전용 디버거"""

    def __init__(self, session_id: str = None):
        super().__init__("validation", session_id)

        # 검증 통계
        self.validation_stats = {
            "total_validations": 0,
            "overfitting_detected": 0,
            "consistency_issues": 0,
            "ab_tests": 0
        }

    def log_walkforward_start(
        self,
        coin: str,
        interval: str,
        train_ratio: float,
        total_data_points: int,
        train_points: int,
        test_points: int
    ):
        """
        Walk-Forward 검증 시작

        Args:
            coin: 코인 심볼
            interval: 인터벌
            train_ratio: 학습 데이터 비율
            total_data_points: 전체 데이터 포인트
            train_points: 학습 데이터 포인트
            test_points: 테스트 데이터 포인트
        """
        self.validation_stats["total_validations"] += 1

        self.log({
            "event": "walkforward_start",
            "coin": coin,
            "interval": interval,
            "train_ratio": float(train_ratio),
            "data_split": {
                "total": total_data_points,
                "train": train_points,
                "test": test_points,
                "train_percentage": float(train_points / total_data_points * 100),
                "test_percentage": float(test_points / total_data_points * 100)
            },
            "message": f"🔍 Walk-Forward 검증 시작: {coin}-{interval}"
        })

    def log_train_phase_result(
        self,
        coin: str,
        interval: str,
        mode: str,
        episodes: int,
        trades: int,
        win_rate: float,
        total_pnl: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float
    ):
        """
        학습 단계 결과 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            mode: 모드 (HYBRID, RULE)
            episodes: 에피소드 수
            trades: 거래 수
            win_rate: 승률
            total_pnl: 총 PnL
            profit_factor: Profit Factor
            sharpe_ratio: Sharpe Ratio
            max_drawdown: 최대 낙폭
        """
        self.log({
            "event": "train_phase_result",
            "coin": coin,
            "interval": interval,
            "mode": mode,
            "phase": "training",
            "performance": {
                "episodes": episodes,
                "trades": trades,
                "win_rate": float(win_rate),
                "total_pnl": float(total_pnl),
                "profit_factor": float(profit_factor),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "avg_pnl_per_trade": float(total_pnl / trades) if trades > 0 else 0.0
            }
        })

    def log_test_phase_result(
        self,
        coin: str,
        interval: str,
        mode: str,
        episodes: int,
        trades: int,
        win_rate: float,
        total_pnl: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float
    ):
        """
        테스트 단계 결과 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            mode: 모드 (HYBRID, RULE)
            episodes: 에피소드 수
            trades: 거래 수
            win_rate: 승률
            total_pnl: 총 PnL
            profit_factor: Profit Factor
            sharpe_ratio: Sharpe Ratio
            max_drawdown: 최대 낙폭
        """
        self.log({
            "event": "test_phase_result",
            "coin": coin,
            "interval": interval,
            "mode": mode,
            "phase": "testing",
            "performance": {
                "episodes": episodes,
                "trades": trades,
                "win_rate": float(win_rate),
                "total_pnl": float(total_pnl),
                "profit_factor": float(profit_factor),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "avg_pnl_per_trade": float(total_pnl / trades) if trades > 0 else 0.0
            }
        })

    def log_overfitting_detection(
        self,
        coin: str,
        interval: str,
        train_pf: float,
        test_pf: float,
        pf_ratio: float,
        overfitting: bool,
        threshold: float = 0.5
    ):
        """
        과적합 감지 로깅 (중요!)

        Args:
            coin: 코인 심볼
            interval: 인터벌
            train_pf: 학습 Profit Factor
            test_pf: 테스트 Profit Factor
            pf_ratio: PF 비율 (test/train)
            overfitting: 과적합 여부
            threshold: 과적합 판정 임계값
        """
        if overfitting:
            self.validation_stats["overfitting_detected"] += 1

        self.log({
            "event": "overfitting_detection",
            "coin": coin,
            "interval": interval,
            "train_pf": float(train_pf),
            "test_pf": float(test_pf),
            "pf_ratio": float(pf_ratio),
            "overfitting": overfitting,
            "threshold": float(threshold),
            "severity": self._calculate_overfitting_severity(train_pf, test_pf),
            "message": "⚠️ 과적합 감지" if overfitting else "✅ 과적합 없음"
        }, level="WARNING" if overfitting else "INFO")

    def _calculate_overfitting_severity(self, train_pf: float, test_pf: float) -> str:
        """과적합 심각도 계산"""
        if train_pf == 0:
            return "unknown"

        ratio = test_pf / train_pf if train_pf > 0 else 0

        if ratio > 0.8:
            return "none"
        elif ratio > 0.5:
            return "mild"
        elif ratio > 0.3:
            return "moderate"
        else:
            return "severe"

    def log_walkforward_end(
        self,
        coin: str,
        interval: str,
        train_result: Dict[str, Any],
        test_result: Dict[str, Any],
        overfitting: bool
    ):
        """
        Walk-Forward 검증 종료

        Args:
            coin: 코인 심볼
            interval: 인터벌
            train_result: 학습 결과
            test_result: 테스트 결과
            overfitting: 과적합 여부
        """
        self.log({
            "event": "walkforward_end",
            "coin": coin,
            "interval": interval,
            "train_result": train_result,
            "test_result": test_result,
            "overfitting": overfitting,
            "generalization_gap": {
                "pnl_gap": float(train_result.get("total_pnl", 0) - test_result.get("total_pnl", 0)),
                "win_rate_gap": float(train_result.get("win_rate", 0) - test_result.get("win_rate", 0)),
                "sharpe_gap": float(train_result.get("sharpe_ratio", 0) - test_result.get("sharpe_ratio", 0))
            },
            "message": f"✅ Walk-Forward 완료: {'과적합 감지' if overfitting else '정상'}"
        })

    def log_multiperiod_validation_start(self, coin: str, interval: str, periods: List[Dict[str, Any]]):
        """
        다중 기간 검증 시작

        Args:
            coin: 코인 심볼
            interval: 인터벌
            periods: 검증 기간 리스트
        """
        self.log({
            "event": "multiperiod_validation_start",
            "coin": coin,
            "interval": interval,
            "num_periods": len(periods),
            "periods": periods,
            "message": f"🔍 다중 기간 검증 시작: {len(periods)}개 기간"
        })

    def log_period_validation_result(
        self,
        coin: str,
        interval: str,
        period_name: str,
        start_date: str,
        end_date: str,
        performance: Dict[str, Any]
    ):
        """
        기간별 검증 결과

        Args:
            coin: 코인 심볼
            interval: 인터벌
            period_name: 기간명 (예: "2024-Q1")
            start_date: 시작일
            end_date: 종료일
            performance: 성능 지표
        """
        self.log({
            "event": "period_validation_result",
            "coin": coin,
            "interval": interval,
            "period": {
                "name": period_name,
                "start_date": start_date,
                "end_date": end_date
            },
            "performance": performance
        })

    def log_consistency_analysis(
        self,
        coin: str,
        interval: str,
        period_performances: List[Dict[str, Any]],
        consistency_score: float,
        is_consistent: bool
    ):
        """
        일관성 분석 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            period_performances: 기간별 성능 리스트
            consistency_score: 일관성 점수
            is_consistent: 일관성 여부
        """
        if not is_consistent:
            self.validation_stats["consistency_issues"] += 1

        # 성능 변동성 계산
        pnls = [p.get("total_pnl", 0) for p in period_performances]
        win_rates = [p.get("win_rate", 0) for p in period_performances]

        self.log({
            "event": "consistency_analysis",
            "coin": coin,
            "interval": interval,
            "num_periods": len(period_performances),
            "consistency_score": float(consistency_score),
            "is_consistent": is_consistent,
            "variability": {
                "pnl_std": float(np.std(pnls)),
                "pnl_range": float(np.max(pnls) - np.min(pnls)),
                "win_rate_std": float(np.std(win_rates)),
                "positive_periods": int(np.sum(np.array(pnls) > 0)),
                "negative_periods": int(np.sum(np.array(pnls) < 0))
            },
            "message": "✅ 성능 일관성 양호" if is_consistent else "⚠️ 성능 일관성 부족"
        }, level="INFO" if is_consistent else "WARNING")

    def log_regime_performance_comparison(
        self,
        coin: str,
        interval: str,
        regime_performances: Dict[str, Dict[str, Any]]
    ):
        """
        레짐별 성능 비교 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            regime_performances: 레짐별 성능
                {
                    "bullish": {"pnl": 100, "win_rate": 0.6, ...},
                    "bearish": {"pnl": -50, "win_rate": 0.4, ...},
                    ...
                }
        """
        # 최고/최악 레짐 찾기
        best_regime = max(regime_performances.keys(), key=lambda k: regime_performances[k].get("total_pnl", 0))
        worst_regime = min(regime_performances.keys(), key=lambda k: regime_performances[k].get("total_pnl", 0))

        # 레짐 간 성능 차이
        pnls = [p.get("total_pnl", 0) for p in regime_performances.values()]
        performance_gap = np.max(pnls) - np.min(pnls)

        self.log({
            "event": "regime_performance_comparison",
            "coin": coin,
            "interval": interval,
            "regime_performances": regime_performances,
            "best_regime": {
                "name": best_regime,
                "pnl": float(regime_performances[best_regime].get("total_pnl", 0)),
                "win_rate": float(regime_performances[best_regime].get("win_rate", 0))
            },
            "worst_regime": {
                "name": worst_regime,
                "pnl": float(regime_performances[worst_regime].get("total_pnl", 0)),
                "win_rate": float(regime_performances[worst_regime].get("win_rate", 0))
            },
            "performance_gap": float(performance_gap),
            "warnings": {
                "large_gap": performance_gap > 100,
                "negative_regimes": [k for k, v in regime_performances.items() if v.get("total_pnl", 0) < 0]
            }
        })

    def log_ab_test_start(self, coin: str, interval: str, mode_a: str, mode_b: str):
        """
        A/B 테스트 시작

        Args:
            coin: 코인 심볼
            interval: 인터벌
            mode_a: 모드 A (예: HYBRID)
            mode_b: 모드 B (예: RULE)
        """
        self.validation_stats["ab_tests"] += 1

        self.log({
            "event": "ab_test_start",
            "coin": coin,
            "interval": interval,
            "mode_a": mode_a,
            "mode_b": mode_b,
            "message": f"🔬 A/B 테스트 시작: {mode_a} vs {mode_b}"
        })

    def log_ab_test_result(
        self,
        coin: str,
        interval: str,
        mode_a: str,
        mode_b: str,
        result_a: Dict[str, Any],
        result_b: Dict[str, Any],
        winner: str,
        improvement_percentage: float
    ):
        """
        A/B 테스트 결과 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            mode_a: 모드 A
            mode_b: 모드 B
            result_a: A 결과
            result_b: B 결과
            winner: 승자
            improvement_percentage: 개선 비율
        """
        self.log({
            "event": "ab_test_result",
            "coin": coin,
            "interval": interval,
            "comparison": {
                "mode_a": mode_a,
                "mode_b": mode_b,
                "result_a": result_a,
                "result_b": result_b
            },
            "winner": winner,
            "improvement_percentage": float(improvement_percentage),
            "differences": {
                "pnl_diff": float(result_a.get("total_pnl", 0) - result_b.get("total_pnl", 0)),
                "win_rate_diff": float(result_a.get("win_rate", 0) - result_b.get("win_rate", 0)),
                "sharpe_diff": float(result_a.get("sharpe_ratio", 0) - result_b.get("sharpe_ratio", 0))
            },
            "message": f"🏆 승자: {winner} (+{improvement_percentage:.1f}%)"
        })

    def log_validation_summary(self, summary: Dict[str, Any]):
        """
        검증 요약 로깅

        Args:
            summary: 전체 검증 요약
        """
        self.log({
            "event": "validation_summary",
            "summary": summary,
            "statistics": self.validation_stats,
            "message": "✅ 검증 완료"
        })

        # 통계 저장
        self.stats.update(self.validation_stats)
        self.save_stats()
