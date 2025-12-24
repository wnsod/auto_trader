"""
전략 생성 데이터 검증
strategies 테이블 및 전략 파라미터 검증
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from .base_validator import BaseValidator
from ..core.validation_context import ValidationContext
from ..core.validation_result import ValidationResult, ValidationIssue, ValidationStatus, ValidationSeverity

logger = logging.getLogger(__name__)

class StrategyValidator(BaseValidator):
    """전략 생성 및 저장 데이터 검증"""

    def __init__(self, db_connections: Dict[str, str] = None):
        super().__init__(db_connections)
        self.strategies_db = db_connections.get('strategies') if db_connections else None

    def validate(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """전략 데이터 검증

        Args:
            data: {
                'strategies': List[Dict],  # 생성된 전략 리스트
                'count': int,              # 전략 개수
                'saved_count': int,        # DB에 저장된 개수
                'coin': str,
                'interval': str
            }
        """
        result = self.create_result(context)
        start_time = datetime.now()

        try:
            # 1. 데이터 존재 여부 검증
            if context.validation_options.get("check_data_types", True):
                self._validate_data_structure(data, result, context)

            # 2. 전략 개수 검증
            if context.validation_options.get("check_ranges", True):
                self._validate_strategy_count(data, result, context)

            # 3. 전략 파라미터 검증
            if context.validation_options.get("check_consistency", True):
                self._validate_strategy_parameters(data, result, context)

            # 4. DB 저장 검증
            if self.strategies_db and context.validation_options.get("check_consistency", True):
                self._validate_db_storage(data, result, context)

            # 5. 과거 데이터와 비교
            if context.should_compare_with_history() and self.strategies_db:
                self._compare_with_history(data, result, context)

            # 6. 이상치 탐지
            if context.validation_options.get("detect_anomalies", True):
                self._detect_anomalies(data, result, context)

        except Exception as e:
            logger.error(f"Strategy validation error: {e}")
            result.add_issue(ValidationIssue(
                check_name="strategy_validation",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation failed with error: {str(e)}",
                suggestion="Check validation logs for details"
            ))

        # 검증 소요 시간 기록
        result.validation_duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    def _validate_data_structure(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """데이터 구조 검증"""
        # 필수 필드 체크
        required_fields = ['strategies', 'count', 'coin', 'interval']
        for field in required_fields:
            issue = self.check_not_none(data.get(field), field)
            result.add_issue(issue)

        # strategies 리스트 검증
        if 'strategies' in data:
            strategies = data['strategies']
            issue = self.check_list_not_empty(strategies, "strategies",
                                             min_size=context.get_threshold("min_strategies", 100))
            result.add_issue(issue)

            # 각 전략의 필수 필드 검증 (샘플링)
            if strategies:
                sample_size = min(10, len(strategies))
                for i in range(sample_size):
                    strategy = strategies[i]
                    for field in ['rsi_period', 'macd_fast', 'macd_slow', 'volume_multiplier']:
                        if field not in strategy:
                            result.add_issue(ValidationIssue(
                                check_name=f"strategy_field_{field}",
                                status=ValidationStatus.FAILED,
                                severity=ValidationSeverity.HIGH,
                                message=f"Strategy #{i} missing required field: {field}",
                                suggestion=f"Ensure all strategies have {field} parameter"
                            ))

    def _validate_strategy_count(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """전략 개수 검증"""
        count = data.get('count', 0)
        saved_count = data.get('saved_count', 0)

        # 전략 개수 범위 체크
        min_strategies = context.get_threshold("min_strategies", 100)
        max_strategies = context.get_threshold("max_strategies", 20000)

        issue = self.check_range(count, min_strategies, max_strategies,
                                "strategy_count", ValidationSeverity.HIGH)
        result.add_issue(issue)

        # 저장 개수와 생성 개수 일치 검증
        if saved_count > 0:
            if abs(count - saved_count) > count * 0.1:  # 10% 이상 차이
                result.add_issue(ValidationIssue(
                    check_name="strategy_save_consistency",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Mismatch between created ({count}) and saved ({saved_count}) strategies",
                    expected=count,
                    actual=saved_count,
                    deviation=abs(count - saved_count),
                    suggestion="Check database save operation for failures"
                ))
            else:
                result.add_issue(ValidationIssue(
                    check_name="strategy_save_consistency",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message="Strategy save count matches creation count"
                ))

    def _validate_strategy_parameters(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """전략 파라미터 범위 검증"""
        strategies = data.get('strategies', [])

        if not strategies:
            return

        # 파라미터 범위
        param_ranges = {
            'rsi_period': context.get_threshold("rsi_range", (2, 100)),
            'rsi_oversold': (10, 50),
            'rsi_overbought': (50, 90),
            'macd_fast': (3, 30),
            'macd_slow': (10, 50),
            'macd_signal': context.get_threshold("macd_signal_range", (1, 50)),
            'volume_multiplier': context.get_threshold("volume_multiplier_range", (0.5, 10.0)),
            'atr_multiplier': context.get_threshold("atr_multiplier_range", (0.1, 5.0))
        }

        # 샘플링하여 검증 (성능 고려)
        sample_size = min(100, len(strategies))
        sample_indices = range(0, len(strategies), max(1, len(strategies) // sample_size))

        invalid_params = []
        for idx in sample_indices:
            strategy = strategies[idx]
            for param, (min_val, max_val) in param_ranges.items():
                if param in strategy:
                    value = strategy[param]
                    if not (min_val <= value <= max_val):
                        invalid_params.append({
                            'strategy_idx': idx,
                            'param': param,
                            'value': value,
                            'range': (min_val, max_val)
                        })

        if invalid_params:
            result.add_issue(ValidationIssue(
                check_name="strategy_parameter_ranges",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Found {len(invalid_params)} parameters out of range",
                actual=f"First invalid: {invalid_params[0]}" if invalid_params else None,
                suggestion="Review parameter generation logic for boundary issues"
            ))
        else:
            result.add_issue(ValidationIssue(
                check_name="strategy_parameter_ranges",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="All sampled strategy parameters within valid ranges"
            ))

        # RSI 일관성 체크 (oversold < overbought)
        consistency_errors = 0
        for strategy in strategies[:sample_size]:
            if 'rsi_oversold' in strategy and 'rsi_overbought' in strategy:
                if strategy['rsi_oversold'] >= strategy['rsi_overbought']:
                    consistency_errors += 1

        if consistency_errors > 0:
            result.add_issue(ValidationIssue(
                check_name="rsi_consistency",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.HIGH,
                message=f"{consistency_errors} strategies have RSI oversold >= overbought",
                suggestion="Fix RSI parameter generation logic"
            ))

    def _validate_db_storage(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """데이터베이스 저장 검증"""
        try:
            coin = data['coin']
            interval = data['interval']

            with sqlite3.connect(self.strategies_db) as conn:
                # DB에서 실제 저장된 전략 수 확인
                cursor = conn.execute("""
                    SELECT COUNT(*) as count,
                           AVG(CAST(json_extract(parameters, '$.rsi_period') AS REAL)) as avg_rsi,
                           AVG(performance_score) as avg_score
                    FROM strategies
                    WHERE symbol = ? AND interval = ?
                    AND created_at > datetime('now', '-1 hour')
                """, (coin, interval))

                row = cursor.fetchone()
                db_count = row[0] if row else 0
                avg_rsi = row[1] if row and row[1] else 0
                avg_score = row[2] if row and row[2] else 0

                expected_count = data.get('saved_count', data.get('count', 0))

                if abs(db_count - expected_count) > expected_count * 0.05:  # 5% 허용 오차
                    result.add_issue(ValidationIssue(
                        check_name="db_storage_verification",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.MEDIUM,
                        message=f"DB count mismatch: expected {expected_count}, found {db_count}",
                        expected=expected_count,
                        actual=db_count,
                        suggestion="Check for database write failures or duplicates"
                    ))
                else:
                    result.add_issue(ValidationIssue(
                        check_name="db_storage_verification",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"DB storage verified: {db_count} strategies saved"
                    ))

                # 평균값 합리성 체크
                if avg_rsi > 0:
                    result.add_issue(self.check_range(avg_rsi, 10, 50, "average_rsi_period",
                                                     ValidationSeverity.LOW))

        except Exception as e:
            result.add_issue(ValidationIssue(
                check_name="db_storage_verification",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Could not verify DB storage: {str(e)}"
            ))

    def _compare_with_history(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """과거 데이터와 비교"""
        try:
            coin = data['coin']
            interval = data['interval']
            current_count = data.get('count', 0)

            # 과거 7일간 전략 생성 히스토리 조회
            historical_data = self.get_historical_data(
                self.strategies_db,
                """
                SELECT COUNT(*) as count, DATE(created_at) as date
                FROM strategies
                WHERE symbol = ? AND interval = ?
                AND created_at > datetime('now', '-7 days')
                GROUP BY DATE(created_at)
                """,
                (coin, interval)
            )

            if historical_data:
                historical_counts = [d['count'] for d in historical_data]
                comparison_issue = self.compare_with_history(
                    current_count,
                    historical_counts,
                    "strategy_count"
                )
                if comparison_issue:
                    result.add_issue(comparison_issue)

        except Exception as e:
            logger.warning(f"Historical comparison failed: {e}")

    def _detect_anomalies(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """이상치 탐지"""
        strategies = data.get('strategies', [])

        if not strategies:
            return

        # RSI period 분포 이상치 체크
        rsi_periods = [s.get('rsi_period', 0) for s in strategies if 'rsi_period' in s]
        if rsi_periods:
            anomaly_issues = self.detect_anomalies(rsi_periods, "rsi_period_distribution")
            for issue in anomaly_issues:
                result.add_issue(issue)

        # Volume multiplier 분포 이상치 체크
        volume_mults = [s.get('volume_multiplier', 0) for s in strategies if 'volume_multiplier' in s]
        if volume_mults:
            anomaly_issues = self.detect_anomalies(volume_mults, "volume_multiplier_distribution")
            for issue in anomaly_issues:
                result.add_issue(issue)