"""
레짐 라우팅 결과 검증
시장 레짐 판단, 전략 라우팅, 백테스트 결과 검증
"""

import os
import sqlite3
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base_validator import BaseValidator
from ..core.validation_context import ValidationContext
from ..core.validation_result import ValidationResult, ValidationIssue, ValidationStatus, ValidationSeverity

logger = logging.getLogger(__name__)

class RoutingValidator(BaseValidator):
    """레짐 기반 라우팅 결과 검증"""

    def validate(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """라우팅 결과 검증

        Args:
            data: {
                'routing_results': List[Dict],    # 라우팅 결과
                'regime': str,                     # 감지된 레짐
                'selected_strategies': List[Dict], # 선택된 전략
                'backtest_results': Dict,          # 백테스트 결과
                'signal_scores': List[float],      # 신호 점수
                'coin': str,
                'interval': str
            }
        """
        result = self.create_result(context)
        start_time = datetime.now()

        try:
            # 1. 데이터 구조 검증
            if context.validation_options.get("check_data_types", True):
                self._validate_data_structure(data, result, context)

            # 2. 레짐 판단 검증
            if context.validation_options.get("check_consistency", True):
                self._validate_regime_detection(data, result, context)

            # 3. 전략 라우팅 검증
            if context.validation_options.get("check_relationships", True):
                self._validate_strategy_routing(data, result, context)

            # 4. 백테스트 결과 검증
            if context.validation_options.get("check_statistics", True):
                self._validate_backtest_results(data, result, context)

            # 5. 신호 점수 검증
            if context.validation_options.get("check_ranges", True):
                self._validate_signal_scores(data, result, context)

            # 6. 과거 데이터와 비교
            if context.should_compare_with_history():
                self._compare_with_history(data, result, context)

            # 7. 시간 일관성 검증
            if context.validation_options.get("check_consistency", True):
                self._validate_temporal_consistency(data, result, context)

        except Exception as e:
            logger.error(f"Routing validation error: {e}")
            result.add_issue(ValidationIssue(
                check_name="routing_validation",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation failed with error: {str(e)}",
                suggestion="Check validation logs for details"
            ))

        result.validation_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    def _validate_data_structure(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """데이터 구조 검증"""
        required_fields = ['routing_results', 'regime', 'coin', 'interval']

        for field in required_fields:
            issue = self.check_not_none(data.get(field), field)
            result.add_issue(issue)

        # 라우팅 결과 리스트 검증
        if 'routing_results' in data:
            routing_results = data['routing_results']
            issue = self.check_list_not_empty(routing_results, "routing_results", min_size=1)
            result.add_issue(issue)

    def _validate_regime_detection(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """레짐 판단 검증"""
        regime = data.get('regime', '').lower()
        valid_regimes = ['bullish', 'bearish', 'neutral', 'volatile', 'trending', 'ranging']

        if regime not in valid_regimes:
            result.add_issue(ValidationIssue(
                check_name="regime_validity",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.HIGH,
                message=f"Invalid regime detected: {regime}",
                expected=f"One of {valid_regimes}",
                actual=regime,
                suggestion="Check regime detection logic"
            ))
        else:
            result.add_issue(ValidationIssue(
                check_name="regime_validity",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"Valid regime detected: {regime}"
            ))

        # 라우팅 결과와 레짐 일관성 체크
        routing_results = data.get('routing_results', [])
        if routing_results:
            # 각 라우팅 결과의 레짐과 일치하는지 확인
            inconsistent_regimes = []
            for i, rr in enumerate(routing_results[:10]):  # 샘플 체크
                if 'regime' in rr and rr['regime'].lower() != regime:
                    inconsistent_regimes.append((i, rr['regime']))

            if inconsistent_regimes:
                result.add_issue(ValidationIssue(
                    check_name="regime_consistency",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Found {len(inconsistent_regimes)} routing results with different regimes",
                    actual=str(inconsistent_regimes[:3]),
                    suggestion="Ensure regime is consistently applied"
                ))

    def _validate_strategy_routing(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """전략 라우팅 검증"""
        selected_strategies = data.get('selected_strategies', [])
        routing_results = data.get('routing_results', [])
        regime = data.get('regime', '').lower()

        # 선택된 전략 수 체크
        if selected_strategies:
            strategy_count = len(selected_strategies)

            # 레짐별 적정 전략 수 (경험적 값)
            expected_ranges = {
                'bullish': (5, 50),
                'bearish': (5, 50),
                'neutral': (10, 100),
                'volatile': (3, 30),
                'trending': (5, 40),
                'ranging': (10, 60)
            }

            if regime in expected_ranges:
                min_strategies, max_strategies = expected_ranges[regime]
                if not (min_strategies <= strategy_count <= max_strategies):
                    result.add_issue(ValidationIssue(
                        check_name="strategy_count_for_regime",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.MEDIUM,
                        message=f"Unusual strategy count for {regime} regime: {strategy_count}",
                        expected=f"[{min_strategies}, {max_strategies}]",
                        actual=strategy_count,
                        suggestion=f"Review strategy selection criteria for {regime} market"
                    ))
                else:
                    result.add_issue(ValidationIssue(
                        check_name="strategy_count_for_regime",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"Appropriate strategy count for {regime}: {strategy_count}"
                    ))

        # 라우팅 결과 품질 체크
        if routing_results:
            high_confidence_count = 0
            low_performance_count = 0

            for rr in routing_results[:50]:  # 샘플
                if 'confidence' in rr and rr['confidence'] > 0.7:
                    high_confidence_count += 1
                if 'performance_score' in rr and rr['performance_score'] < 0:
                    low_performance_count += 1

            confidence_rate = high_confidence_count / min(50, len(routing_results))
            poor_performance_rate = low_performance_count / min(50, len(routing_results))

            if confidence_rate < 0.3:
                result.add_issue(ValidationIssue(
                    check_name="routing_confidence",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Low confidence in routing: only {confidence_rate:.1%} have high confidence",
                    suggestion="Review routing criteria and strategy quality"
                ))

            if poor_performance_rate > 0.5:
                result.add_issue(ValidationIssue(
                    check_name="routing_performance",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.HIGH,
                    message=f"Many poor performers: {poor_performance_rate:.1%} have negative scores",
                    suggestion="Filter out low-quality strategies before routing"
                ))

    def _validate_backtest_results(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """백테스트 결과 검증"""
        backtest = data.get('backtest_results', {})

        if not backtest:
            result.add_issue(ValidationIssue(
                check_name="backtest_presence",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.HIGH,
                message="No backtest results found",
                suggestion="Ensure backtest is performed after routing"
            ))
            return

        # 주요 메트릭 검증
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']

        for metric in metrics:
            if metric not in backtest:
                result.add_issue(ValidationIssue(
                    check_name=f"backtest_{metric}",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Missing backtest metric: {metric}"
                ))
                continue

            value = backtest[metric]

            # 메트릭별 합리적 범위 체크
            if metric == 'total_return':
                issue = self.check_range(value, -0.5, 2.0, metric, ValidationSeverity.MEDIUM)
                result.add_issue(issue)

            elif metric == 'sharpe_ratio':
                issue = self.check_range(value, -2.0, 3.0, metric, ValidationSeverity.LOW)
                result.add_issue(issue)

            elif metric == 'max_drawdown':
                issue = self.check_range(value, -1.0, 0, metric, ValidationSeverity.MEDIUM)
                result.add_issue(issue)

            elif metric == 'win_rate':
                issue = self.check_range(value, 0.2, 0.8, metric, ValidationSeverity.MEDIUM)
                result.add_issue(issue)

            elif metric == 'total_trades':
                min_trades = 10
                if value < min_trades:
                    result.add_issue(ValidationIssue(
                        check_name=f"backtest_{metric}",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.HIGH,
                        message=f"Too few trades in backtest: {value}",
                        expected=f">= {min_trades}",
                        actual=value,
                        suggestion="Increase backtest period or review signal generation"
                    ))

    def _validate_signal_scores(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """신호 점수 검증"""
        signal_scores = data.get('signal_scores', [])

        if not signal_scores:
            return

        # 점수 범위 체크 (0-1)
        invalid_scores = [s for s in signal_scores if not (0 <= s <= 1)]

        if invalid_scores:
            result.add_issue(ValidationIssue(
                check_name="signal_score_range",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.HIGH,
                message=f"Found {len(invalid_scores)} signal scores out of [0,1] range",
                actual=f"Invalid scores: {invalid_scores[:5]}",
                suggestion="Normalize signal scores to [0,1] range"
            ))
        else:
            result.add_issue(ValidationIssue(
                check_name="signal_score_range",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="All signal scores within valid range"
            ))

        # 점수 분포 체크
        if len(signal_scores) > 10:
            mean_score = np.mean(signal_scores)
            std_score = np.std(signal_scores)

            # 모든 점수가 너무 비슷한 경우 (분산이 너무 작음)
            if std_score < 0.01:
                result.add_issue(ValidationIssue(
                    check_name="signal_score_diversity",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Signal scores lack diversity (std={std_score:.4f})",
                    suggestion="Review scoring logic - may not be discriminative enough"
                ))

            # 평균이 극단적인 경우
            if mean_score < 0.2 or mean_score > 0.8:
                result.add_issue(ValidationIssue(
                    check_name="signal_score_distribution",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Signal scores skewed: mean={mean_score:.2f}",
                    suggestion="Review scoring calibration"
                ))

    def _validate_temporal_consistency(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """시간적 일관성 검증"""
        routing_results = data.get('routing_results', [])

        if not routing_results:
            return

        # 타임스탬프 순서 체크
        timestamps = []
        for rr in routing_results:
            if 'timestamp' in rr:
                try:
                    ts = datetime.fromisoformat(rr['timestamp'])
                    timestamps.append(ts)
                except:
                    pass

        if len(timestamps) > 1:
            # 시간 순서가 올바른지 체크
            is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

            if not is_sorted:
                result.add_issue(ValidationIssue(
                    check_name="temporal_ordering",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.HIGH,
                    message="Routing results not in chronological order",
                    suggestion="Sort results by timestamp"
                ))

            # 시간 갭 체크
            gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
                   for i in range(len(timestamps)-1)]

            if gaps:
                max_gap = max(gaps)
                if max_gap > 3600:  # 1시간 이상 갭
                    result.add_issue(ValidationIssue(
                        check_name="temporal_gaps",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.MEDIUM,
                        message=f"Large time gap detected: {max_gap/3600:.1f} hours",
                        suggestion="Check for missing data periods"
                    ))

    def _compare_with_history(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """과거 라우팅 결과와 비교"""
        if not self.db_connections or 'learning_results' not in self.db_connections:
            return

        try:
            coin = data.get('coin')
            interval = data.get('interval')
            regime = data.get('regime')
            current_count = len(data.get('routing_results', []))

            # 같은 레짐의 과거 라우팅 결과 조회
            historical_data = self.get_historical_data(
                self.db_connections['learning_results'],
                """
                SELECT routing_count, performance_score
                FROM regime_routing_results
                WHERE coin = ? AND interval = ? AND regime = ?
                AND created_at > datetime('now', '-14 days')
                ORDER BY created_at DESC
                LIMIT 10
                """,
                (coin, interval, regime)
            )

            if historical_data:
                hist_counts = [d['routing_count'] for d in historical_data]
                if hist_counts:
                    count_issue = self.compare_with_history(
                        current_count, hist_counts, f"{regime}_routing_count"
                    )
                    if count_issue:
                        result.add_issue(count_issue)

        except Exception as e:
            logger.warning(f"Historical comparison failed: {e}")