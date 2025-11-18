"""
Analysis 모듈 전용 디버그 로거
- 통합 분석 추적
- 인터벌 가중치 계산
- 신호 점수 산출
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .debug_logger import DebugLogger


class AnalysisDebugger(DebugLogger):
    """Analysis 모듈 전용 디버거"""

    def __init__(self, session_id: str = None):
        super().__init__("analysis", session_id)

        # 분석 통계
        self.analysis_stats = {
            "total_analyses": 0,
            "multi_interval_analyses": 0
        }

    def log_integrated_analysis_start(
        self,
        coin: str,
        intervals: List[str],
        num_strategies: int,
        regime: str
    ):
        """
        통합 분석 시작

        Args:
            coin: 코인 심볼
            intervals: 인터벌 리스트
            num_strategies: 전략 수
            regime: 레짐
        """
        self.analysis_stats["total_analyses"] += 1
        if len(intervals) > 1:
            self.analysis_stats["multi_interval_analyses"] += 1

        self.log({
            "event": "integrated_analysis_start",
            "coin": coin,
            "intervals": intervals,
            "num_strategies": num_strategies,
            "regime": regime,
            "message": f"🔥 통합 분석 시작: {coin} ({len(intervals)}개 인터벌)"
        })

    def log_interval_strategy_score(
        self,
        coin: str,
        interval: str,
        strategy_score: float,
        num_strategies: int,
        grade_distribution: Dict[str, int]
    ):
        """
        인터벌별 전략 점수 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategy_score: 전략 점수
            num_strategies: 전략 수
            grade_distribution: 등급 분포
        """
        self.log({
            "event": "interval_strategy_score",
            "coin": coin,
            "interval": interval,
            "strategy_score": float(strategy_score),
            "num_strategies": num_strategies,
            "grade_distribution": grade_distribution
        }, level="DEBUG")

    def log_fractal_analysis(
        self,
        coin: str,
        interval: str,
        fractal_score: float,
        fractal_ratios: Dict[str, float],
        detected_patterns: List[str]
    ):
        """
        프랙탈 분석 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            fractal_score: 프랙탈 점수
            fractal_ratios: 프랙탈 비율
            detected_patterns: 감지된 패턴
        """
        self.log({
            "event": "fractal_analysis",
            "coin": coin,
            "interval": interval,
            "fractal_score": float(fractal_score),
            "fractal_ratios": {k: float(v) for k, v in fractal_ratios.items()},
            "detected_patterns": detected_patterns,
            "num_patterns": len(detected_patterns)
        }, level="DEBUG")

    def log_multi_timeframe_analysis(
        self,
        coin: str,
        interval: str,
        multi_tf_score: float,
        timeframe_ratios: Dict[str, float],
        alignment: float
    ):
        """
        다중 시간대 분석 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            multi_tf_score: 다중 시간대 점수
            timeframe_ratios: 시간대별 비율
            alignment: 정렬 점수
        """
        self.log({
            "event": "multi_timeframe_analysis",
            "coin": coin,
            "interval": interval,
            "multi_tf_score": float(multi_tf_score),
            "timeframe_ratios": {k: float(v) for k, v in timeframe_ratios.items()},
            "alignment": float(alignment)
        }, level="DEBUG")

    def log_indicator_cross_analysis(
        self,
        coin: str,
        interval: str,
        indicator_score: float,
        indicator_ratios: Dict[str, float],
        crosses_detected: Dict[str, bool]
    ):
        """
        지표 교차 분석 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            indicator_score: 지표 점수
            indicator_ratios: 지표별 비율
            crosses_detected: 감지된 교차
        """
        self.log({
            "event": "indicator_cross_analysis",
            "coin": coin,
            "interval": interval,
            "indicator_score": float(indicator_score),
            "indicator_ratios": {k: float(v) for k, v in indicator_ratios.items()},
            "crosses_detected": crosses_detected,
            "num_crosses": sum(1 for v in crosses_detected.values() if v)
        }, level="DEBUG")

    def log_interval_confidence(
        self,
        coin: str,
        interval: str,
        strategy_score: float,
        fractal_score: float,
        multi_tf_score: float,
        indicator_score: float,
        context_confidence: float,
        regime_consistency: float,
        interval_confidence: float,
        dynamic_weights: Dict[str, float]
    ):
        """
        인터벌 신뢰도 로깅 (중요!)

        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategy_score: 전략 점수
            fractal_score: 프랙탈 점수
            multi_tf_score: 다중 TF 점수
            indicator_score: 지표 점수
            context_confidence: 맥락 신뢰도
            regime_consistency: 레짐 일치도
            interval_confidence: 최종 인터벌 신뢰도
            dynamic_weights: 동적 가중치
        """
        self.log({
            "event": "interval_confidence",
            "coin": coin,
            "interval": interval,
            "scores": {
                "strategy": float(strategy_score),
                "fractal": float(fractal_score),
                "multi_timeframe": float(multi_tf_score),
                "indicator_cross": float(indicator_score),
                "context": float(context_confidence)
            },
            "regime_consistency": float(regime_consistency),
            "interval_confidence": float(interval_confidence),
            "dynamic_weights": {k: float(v) for k, v in dynamic_weights.items()},
            "weighted_contribution": {
                "fractal": float(fractal_score * dynamic_weights.get("fractal", 0)),
                "multi_tf": float(multi_tf_score * dynamic_weights.get("multi_timeframe", 0)),
                "indicator": float(indicator_score * dynamic_weights.get("indicator_cross", 0)),
                "context": float(context_confidence * dynamic_weights.get("context", 0))
            }
        })

    def log_interval_weights(
        self,
        coin: str,
        interval_weights: Dict[str, float],
        normalization_method: str = "confidence_based"
    ):
        """
        인터벌 가중치 로깅 (중요!)

        Args:
            coin: 코인 심볼
            interval_weights: 인터벌별 가중치
            normalization_method: 정규화 방법
        """
        # 가중치 분포 분석
        weights = list(interval_weights.values())
        max_weight = max(weights) if weights else 0
        min_weight = min(weights) if weights else 0

        self.log({
            "event": "interval_weights",
            "coin": coin,
            "weights": {k: float(v) for k, v in interval_weights.items()},
            "normalization_method": normalization_method,
            "weight_statistics": {
                "max": float(max_weight),
                "min": float(min_weight),
                "range": float(max_weight - min_weight),
                "std": float(np.std(weights)),
                "dominant_interval": max(interval_weights.keys(), key=lambda k: interval_weights[k])
            },
            "warnings": {
                "highly_skewed": max_weight > 0.7,  # 한 인터벌이 70% 이상
                "low_diversity": np.std(weights) < 0.05
            }
        })

    def log_final_signal_calculation(
        self,
        coin: str,
        interval_scores: Dict[str, float],
        interval_weights: Dict[str, float],
        final_signal_score: float,
        signal_action: str,
        signal_confidence: float
    ):
        """
        최종 신호 계산 로깅 (핵심!)

        Args:
            coin: 코인 심볼
            interval_scores: 인터벌별 점수
            interval_weights: 인터벌별 가중치
            final_signal_score: 최종 신호 점수
            signal_action: 신호 액션
            signal_confidence: 신호 신뢰도
        """
        # 각 인터벌의 기여도 계산
        contributions = {
            interval: float(score * interval_weights.get(interval, 0))
            for interval, score in interval_scores.items()
        }

        self.log({
            "event": "final_signal_calculation",
            "coin": coin,
            "interval_scores": {k: float(v) for k, v in interval_scores.items()},
            "interval_weights": {k: float(v) for k, v in interval_weights.items()},
            "interval_contributions": contributions,
            "final_signal_score": float(final_signal_score),
            "signal_action": signal_action,
            "signal_confidence": float(signal_confidence),
            "calculation_breakdown": [
                f"{interval}: {score:.3f} × {interval_weights.get(interval, 0):.3f} = {contributions[interval]:.3f}"
                for interval, score in interval_scores.items()
            ],
            "message": f"🔥 최종 신호: {signal_action} (점수: {final_signal_score:.3f}, 신뢰도: {signal_confidence:.3f})"
        })

    def log_analysis_comparison(
        self,
        coin: str,
        current_analysis: Dict[str, Any],
        previous_analysis: Dict[str, Any] = None
    ):
        """
        분석 결과 비교 로깅

        Args:
            coin: 코인 심볼
            current_analysis: 현재 분석 결과
            previous_analysis: 이전 분석 결과 (옵션)
        """
        if not previous_analysis:
            self.log({
                "event": "analysis_comparison",
                "coin": coin,
                "current_analysis": current_analysis,
                "message": "첫 번째 분석 (비교 대상 없음)"
            })
            return

        # 변화 계산
        score_change = current_analysis.get("final_signal_score", 0) - previous_analysis.get("final_signal_score", 0)
        action_changed = current_analysis.get("signal_action") != previous_analysis.get("signal_action")

        self.log({
            "event": "analysis_comparison",
            "coin": coin,
            "current": {
                "score": float(current_analysis.get("final_signal_score", 0)),
                "action": current_analysis.get("signal_action"),
                "confidence": float(current_analysis.get("signal_confidence", 0))
            },
            "previous": {
                "score": float(previous_analysis.get("final_signal_score", 0)),
                "action": previous_analysis.get("signal_action"),
                "confidence": float(previous_analysis.get("signal_confidence", 0))
            },
            "changes": {
                "score_change": float(score_change),
                "action_changed": action_changed,
                "confidence_change": float(
                    current_analysis.get("signal_confidence", 0) -
                    previous_analysis.get("signal_confidence", 0)
                )
            },
            "warnings": {
                "large_score_change": abs(score_change) > 0.3,
                "action_flip": action_changed
            }
        })

    def log_analysis_summary(self, summary: Dict[str, Any]):
        """
        분석 요약 로깅

        Args:
            summary: 전체 분석 요약
        """
        self.log({
            "event": "analysis_summary",
            "summary": summary,
            "statistics": self.analysis_stats,
            "message": "✅ 분석 완료"
        })

        # 통계 저장
        self.stats.update(self.analysis_stats)
        self.save_stats()
