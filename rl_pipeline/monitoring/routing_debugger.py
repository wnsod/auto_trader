"""
Routing 모듈 전용 디버그 로거
- 레짐 감지 추적
- 전략 라우팅 결정
- 레짐 전환 감지
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .debug_logger import DebugLogger


class RoutingDebugger(DebugLogger):
    """Routing 모듈 전용 디버거"""

    def __init__(self, session_id: str = None):
        super().__init__("routing", session_id)

        # 라우팅 통계
        self.routing_stats = {
            "total_regime_detections": 0,
            "regime_changes": 0,
            "total_strategies_routed": 0
        }

    def log_regime_detection_start(self, coin: str, interval: str, candle_count: int):
        """
        레짐 감지 시작

        Args:
            coin: 코인 심볼
            interval: 인터벌
            candle_count: 캔들 데이터 개수
        """
        self.routing_stats["total_regime_detections"] += 1

        self.log({
            "event": "regime_detection_start",
            "coin": coin,
            "interval": interval,
            "candle_count": candle_count,
            "message": f"🔍 레짐 감지 시작: {coin}-{interval}"
        })

    def log_regime_indicators(
        self,
        coin: str,
        interval: str,
        indicators: Dict[str, float],
        candle_data: Dict[str, Any]
    ):
        """
        레짐 감지용 지표 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            indicators: 지표 값들 (rsi, macd, atr, adx 등)
            candle_data: 최근 캔들 데이터
        """
        self.log({
            "event": "regime_indicators",
            "coin": coin,
            "interval": interval,
            "indicators": {k: float(v) for k, v in indicators.items()},
            "recent_price": {
                "close": float(candle_data.get("close", 0)),
                "high": float(candle_data.get("high", 0)),
                "low": float(candle_data.get("low", 0)),
                "volume": float(candle_data.get("volume", 0))
            }
        }, level="DEBUG")

    def log_regime_detected(
        self,
        coin: str,
        interval: str,
        regime: str,
        confidence: float,
        transition_probability: float,
        indicators: Dict[str, float],
        previous_regime: Optional[str] = None
    ):
        """
        레짐 감지 결과 로깅 (중요!)

        Args:
            coin: 코인 심볼
            interval: 인터벌
            regime: 감지된 레짐
            confidence: 신뢰도
            transition_probability: 전환 확률
            indicators: 사용된 지표들
            previous_regime: 이전 레짐 (있으면)
        """
        regime_changed = previous_regime and previous_regime != regime
        if regime_changed:
            self.routing_stats["regime_changes"] += 1

        self.log({
            "event": "regime_detected",
            "coin": coin,
            "interval": interval,
            "regime": regime,
            "confidence": float(confidence),
            "transition_probability": float(transition_probability),
            "previous_regime": previous_regime,
            "regime_changed": regime_changed,
            "indicators": {k: float(v) for k, v in indicators.items()},
            "message": f"📊 레짐: {regime} (신뢰도: {confidence:.2f})" +
                      (f" [변경: {previous_regime} → {regime}]" if regime_changed else "")
        }, level="WARNING" if regime_changed else "INFO")

    def log_routing_start(
        self,
        coin: str,
        interval: str,
        regime: str,
        num_strategies: int
    ):
        """
        전략 라우팅 시작

        Args:
            coin: 코인 심볼
            interval: 인터벌
            regime: 현재 레짐
            num_strategies: 라우팅할 전략 수
        """
        self.log({
            "event": "routing_start",
            "coin": coin,
            "interval": interval,
            "regime": regime,
            "num_strategies": num_strategies,
            "message": f"🔄 전략 라우팅 시작: {num_strategies}개"
        })

    def log_strategy_routing_decision(
        self,
        coin: str,
        interval: str,
        strategy_id: str,
        strategy_grade: str,
        regime: str,
        routed: bool,
        reason: str,
        score: float = None
    ):
        """
        개별 전략 라우팅 결정 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategy_id: 전략 ID
            strategy_grade: 전략 등급
            regime: 레짐
            routed: 라우팅 여부
            reason: 결정 이유
            score: 전략 점수 (옵션)
        """
        if routed:
            self.routing_stats["total_strategies_routed"] += 1

        self.log({
            "event": "strategy_routing_decision",
            "coin": coin,
            "interval": interval,
            "strategy": {
                "id": strategy_id,
                "grade": strategy_grade,
                "score": float(score) if score is not None else None
            },
            "regime": regime,
            "routed": routed,
            "reason": reason
        }, level="DEBUG")

    def log_routing_end(
        self,
        coin: str,
        interval: str,
        regime: str,
        total_strategies: int,
        routed_strategies: int,
        grade_distribution: Dict[str, int]
    ):
        """
        전략 라우팅 종료 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            regime: 레짐
            total_strategies: 전체 전략 수
            routed_strategies: 라우팅된 전략 수
            grade_distribution: 등급별 분포
        """
        self.log({
            "event": "routing_end",
            "coin": coin,
            "interval": interval,
            "regime": regime,
            "total_strategies": total_strategies,
            "routed_strategies": routed_strategies,
            "routing_ratio": float(routed_strategies / total_strategies) if total_strategies > 0 else 0,
            "grade_distribution": grade_distribution,
            "message": f"✅ 라우팅 완료: {routed_strategies}/{total_strategies}개"
        })

    def log_regime_alignment(
        self,
        coin: str,
        intervals: List[str],
        interval_regimes: Dict[str, str],
        alignment_score: float,
        main_regime: str
    ):
        """
        다중 인터벌 레짐 일치도 로깅

        Args:
            coin: 코인 심볼
            intervals: 인터벌 리스트
            interval_regimes: 인터벌별 레짐
            alignment_score: 일치도 점수
            main_regime: 메인 레짐
        """
        # 레짐 분포
        regime_counts = {}
        for regime in interval_regimes.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        self.log({
            "event": "regime_alignment",
            "coin": coin,
            "num_intervals": len(intervals),
            "interval_regimes": interval_regimes,
            "regime_distribution": regime_counts,
            "alignment_score": float(alignment_score),
            "main_regime": main_regime,
            "warnings": {
                "low_alignment": alignment_score < 0.6,
                "conflicting_regimes": len(regime_counts) > 2
            },
            "message": f"📊 레짐 일치도: {alignment_score:.2f} (메인: {main_regime})"
        })

    def log_backtesting_result(
        self,
        coin: str,
        interval: str,
        regime: str,
        strategy_id: str,
        backtest_result: Dict[str, Any]
    ):
        """
        백테스트 결과 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            regime: 레짐
            strategy_id: 전략 ID
            backtest_result: 백테스트 결과
        """
        self.log({
            "event": "backtesting_result",
            "coin": coin,
            "interval": interval,
            "regime": regime,
            "strategy_id": strategy_id,
            "result": backtest_result
        }, level="DEBUG")

    def log_routing_summary(self, summary: Dict[str, Any]):
        """
        라우팅 요약 로깅

        Args:
            summary: 전체 라우팅 요약
        """
        self.log({
            "event": "routing_summary",
            "summary": summary,
            "statistics": self.routing_stats,
            "message": "✅ 라우팅 완료"
        })

        # 통계 저장
        self.stats.update(self.routing_stats)
        self.save_stats()
