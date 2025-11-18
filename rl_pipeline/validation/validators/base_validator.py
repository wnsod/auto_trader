"""
기본 Validator 추상 클래스
모든 Validator가 상속받아야 하는 기본 인터페이스
"""

import sqlite3
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from ..core.validation_context import ValidationContext
from ..core.validation_result import ValidationResult, ValidationIssue, ValidationStatus, ValidationSeverity

logger = logging.getLogger(__name__)

class BaseValidator(ABC):
    """기본 Validator 클래스"""

    def __init__(self, db_connections: Dict[str, str] = None):
        """초기화

        Args:
            db_connections: 데이터베이스 연결 정보
                - strategies: 전략 DB 경로
                - learning_results: 학습 결과 DB 경로
        """
        self.db_connections = db_connections or {}
        self.component_name = self.__class__.__name__.replace("Validator", "")

    @abstractmethod
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        """검증 실행 - 서브클래스에서 구현"""
        pass

    def create_result(self, context: ValidationContext) -> ValidationResult:
        """검증 결과 객체 생성"""
        validation_id = f"{self.component_name}_{context.coin}_{context.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return ValidationResult(
            component=self.component_name,
            validation_id=validation_id,
            metadata={
                "coin": context.coin,
                "interval": context.interval,
                "stage": context.stage,
                "trust_level": context.trust_level
            }
        )

    def check_range(self, value: float, min_val: float, max_val: float,
                   name: str, severity: ValidationSeverity = ValidationSeverity.HIGH) -> Optional[ValidationIssue]:
        """범위 검증"""
        if min_val <= value <= max_val:
            return ValidationIssue(
                check_name=f"{name}_range",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"{name} is within valid range",
                expected=f"[{min_val}, {max_val}]",
                actual=value
            )
        else:
            return ValidationIssue(
                check_name=f"{name}_range",
                status=ValidationStatus.FAILED,
                severity=severity,
                message=f"{name} out of valid range",
                expected=f"[{min_val}, {max_val}]",
                actual=value,
                deviation=min(abs(value - min_val), abs(value - max_val)),
                suggestion=f"Adjust {name} to be within [{min_val}, {max_val}]"
            )

    def check_not_none(self, value: Any, name: str) -> ValidationIssue:
        """None 체크"""
        if value is not None:
            return ValidationIssue(
                check_name=f"{name}_not_none",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"{name} is not None"
            )
        else:
            return ValidationIssue(
                check_name=f"{name}_not_none",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"{name} is None",
                suggestion=f"Ensure {name} is properly initialized"
            )

    def check_list_not_empty(self, lst: List, name: str,
                           min_size: int = 1) -> ValidationIssue:
        """리스트 비어있지 않은지 체크"""
        if lst and len(lst) >= min_size:
            return ValidationIssue(
                check_name=f"{name}_not_empty",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"{name} has {len(lst)} items"
            )
        else:
            return ValidationIssue(
                check_name=f"{name}_not_empty",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.HIGH,
                message=f"{name} is empty or too small",
                expected=f">= {min_size} items",
                actual=len(lst) if lst else 0,
                suggestion=f"Ensure {name} has at least {min_size} items"
            )

    def compare_with_history(self, current_value: float, historical_values: List[float],
                            metric_name: str, tolerance: float = 0.3) -> Optional[ValidationIssue]:
        """과거 데이터와 비교"""
        if not historical_values:
            return None

        mean = np.mean(historical_values)
        std = np.std(historical_values)

        if std == 0:
            # 표준편차가 0이면 평균과 같은지만 체크
            if abs(current_value - mean) < 0.001:
                return ValidationIssue(
                    check_name=f"{metric_name}_historical_comparison",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message=f"{metric_name} consistent with history"
                )

        z_score = abs((current_value - mean) / std) if std > 0 else 0

        # 3-sigma rule
        if z_score > 3:
            return ValidationIssue(
                check_name=f"{metric_name}_historical_comparison",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"{metric_name} significantly deviates from historical average",
                expected=f"Close to {mean:.4f} (±{std:.4f})",
                actual=current_value,
                deviation=z_score,
                suggestion=f"Investigate why {metric_name} is {z_score:.1f} standard deviations from normal"
            )
        else:
            return ValidationIssue(
                check_name=f"{metric_name}_historical_comparison",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"{metric_name} within historical norms",
                expected=f"{mean:.4f} (±{std:.4f})",
                actual=current_value
            )

    def detect_anomalies(self, values: List[float], name: str) -> List[ValidationIssue]:
        """이상치 탐지 (IQR 방법)"""
        issues = []

        if len(values) < 4:
            return issues

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [v for v in values if v < lower_bound or v > upper_bound]

        if outliers:
            issues.append(ValidationIssue(
                check_name=f"{name}_anomaly_detection",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Found {len(outliers)} outliers in {name}",
                expected=f"Values within [{lower_bound:.4f}, {upper_bound:.4f}]",
                actual=f"{len(outliers)} outliers: {outliers[:5]}...",  # 최대 5개만 표시
                suggestion=f"Review outlier values in {name} for potential data issues"
            ))
        else:
            issues.append(ValidationIssue(
                check_name=f"{name}_anomaly_detection",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"No anomalies detected in {name}"
            ))

        return issues

    def get_historical_data(self, db_path: str, query: str,
                           params: Tuple = ()) -> List[Dict[str, Any]]:
        """과거 데이터 조회"""
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get historical data: {e}")
            return []

    def check_consistency(self, data: Dict[str, Any], rules: List[Tuple[str, callable]]) -> List[ValidationIssue]:
        """일관성 규칙 검증

        Args:
            data: 검증할 데이터
            rules: (규칙명, 검증함수) 튜플 리스트
        """
        issues = []

        for rule_name, rule_func in rules:
            try:
                is_valid, message = rule_func(data)
                if is_valid:
                    issues.append(ValidationIssue(
                        check_name=f"consistency_{rule_name}",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"Consistency rule '{rule_name}' passed"
                    ))
                else:
                    issues.append(ValidationIssue(
                        check_name=f"consistency_{rule_name}",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.HIGH,
                        message=message or f"Consistency rule '{rule_name}' failed",
                        suggestion=f"Review data consistency for {rule_name}"
                    ))
            except Exception as e:
                issues.append(ValidationIssue(
                    check_name=f"consistency_{rule_name}",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Could not verify consistency rule '{rule_name}': {str(e)}"
                ))

        return issues

    def auto_fix(self, issue: ValidationIssue, data: Any) -> Tuple[bool, Any, str]:
        """자동 복구 시도

        Returns:
            (성공여부, 수정된데이터, 수정내용설명)
        """
        # 기본 구현은 복구하지 않음
        # 서브클래스에서 구체적인 복구 로직 구현
        return False, data, ""