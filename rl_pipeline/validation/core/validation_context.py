"""
검증 컨텍스트 - 검증 시 필요한 메타데이터 및 설정 관리
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class ValidationContext:
    """검증 실행 컨텍스트"""

    # 기본 정보
    coin: str
    interval: str
    stage: str  # strategy_generation, selfplay, routing, analysis, etc.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 검증 설정
    trust_level: str = "PARANOID"
    enable_auto_fix: bool = True
    enable_comparison: bool = True
    comparison_lookback_days: int = 7

    # 실행 환경
    pipeline_run_id: Optional[str] = None
    session_id: Optional[str] = None

    # 이전 실행 정보 (비교용)
    previous_results: Optional[Dict[str, Any]] = None
    historical_stats: Optional[Dict[str, Any]] = None

    # 임계값 설정
    thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "min_strategies": 100,
        "max_strategies": 20000,
        "min_success_rate": 0.3,
        "max_success_rate": 0.9,
        "min_prediction_accuracy": 0.4,
        "max_prediction_accuracy": 0.8,
        "min_profit_rate": -0.5,
        "max_profit_rate": 2.0,
        "rsi_range": (2, 100),
        "macd_signal_range": (1, 50),
        "volume_multiplier_range": (0.5, 10.0),
        "atr_multiplier_range": (0.1, 5.0)
    })

    # 검증 옵션
    validation_options: Dict[str, bool] = field(default_factory=lambda: {
        "check_data_types": True,
        "check_ranges": True,
        "check_statistics": True,
        "check_consistency": True,
        "check_relationships": True,
        "compare_with_history": True,
        "detect_anomalies": True
    })

    # 추가 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_perform_deep_check(self) -> bool:
        """신뢰도 레벨에 따라 심층 검사 여부 결정"""
        return self.trust_level in ["PARANOID", "CAUTIOUS"]

    def should_compare_with_history(self) -> bool:
        """과거 데이터와 비교 여부"""
        return (self.enable_comparison and
                self.validation_options.get("compare_with_history", True) and
                self.trust_level in ["PARANOID", "CAUTIOUS", "MODERATE"])

    def get_threshold(self, key: str, default: Any = None) -> Any:
        """임계값 조회"""
        return self.thresholds.get(key, default)

    def update_from_trust_level(self, trust_level: str):
        """신뢰도 레벨에 따라 검증 옵션 자동 조정"""
        self.trust_level = trust_level

        if trust_level == "PARANOID":
            # 모든 검증 활성화
            for key in self.validation_options:
                self.validation_options[key] = True

        elif trust_level == "CAUTIOUS":
            # 대부분 검증 활성화
            self.validation_options["check_data_types"] = True
            self.validation_options["check_ranges"] = True
            self.validation_options["check_statistics"] = True
            self.validation_options["check_consistency"] = True
            self.validation_options["check_relationships"] = False
            self.validation_options["compare_with_history"] = True
            self.validation_options["detect_anomalies"] = True

        elif trust_level == "MODERATE":
            # 중요 검증만
            self.validation_options["check_data_types"] = True
            self.validation_options["check_ranges"] = True
            self.validation_options["check_statistics"] = False
            self.validation_options["check_consistency"] = True
            self.validation_options["check_relationships"] = False
            self.validation_options["compare_with_history"] = True
            self.validation_options["detect_anomalies"] = False

        elif trust_level == "CONFIDENT":
            # 핵심 검증만
            self.validation_options["check_data_types"] = False
            self.validation_options["check_ranges"] = True
            self.validation_options["check_statistics"] = False
            self.validation_options["check_consistency"] = True
            self.validation_options["check_relationships"] = False
            self.validation_options["compare_with_history"] = False
            self.validation_options["detect_anomalies"] = False

        else:  # TRUSTED
            # 최소 검증
            self.validation_options["check_data_types"] = False
            self.validation_options["check_ranges"] = True
            self.validation_options["check_statistics"] = False
            self.validation_options["check_consistency"] = False
            self.validation_options["check_relationships"] = False
            self.validation_options["compare_with_history"] = False
            self.validation_options["detect_anomalies"] = False