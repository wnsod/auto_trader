"""
Self-play 결과 검증
에피소드 결과, 예측 정확도, 승률 등 검증
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

class SelfPlayValidator(BaseValidator):
    """Self-play 진화 결과 검증"""

    def validate(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """Self-play 결과 검증

        Args:
            data: {
                'episodes': List[Dict],          # 에피소드 결과
                'total_episodes': int,            # 총 에피소드 수
                'evolved_strategies': List[Dict], # 진화된 전략
                'prediction_accuracy': float,     # 예측 정확도
                'average_return': float,          # 평균 수익률
                'win_rate': float,               # 승률
                'coin': str,
                'interval': str
            }
        """
        result = self.create_result(context)
        start_time = datetime.now()

        try:
            # Self-play가 비활성화된 경우 건너뛰기
            if not data or data.get('skipped'):
                result.add_issue(ValidationIssue(
                    check_name="selfplay_enabled",
                    status=ValidationStatus.SKIPPED,
                    severity=ValidationSeverity.INFO,
                    message="Self-play was skipped (ENABLE_SELFPLAY=false)"
                ))
                result.validation_duration_ms = 0
                return result

            # 1. 데이터 구조 검증
            if context.validation_options.get("check_data_types", True):
                self._validate_data_structure(data, result, context)

            # 2. 에피소드 결과 검증
            if context.validation_options.get("check_consistency", True):
                self._validate_episode_results(data, result, context)

            # 3. 예측 정확도 검증
            if context.validation_options.get("check_ranges", True):
                self._validate_prediction_accuracy(data, result, context)

            # 4. 수익률 및 승률 검증
            if context.validation_options.get("check_statistics", True):
                self._validate_performance_metrics(data, result, context)

            # 5. 전략 진화 검증
            if context.validation_options.get("check_relationships", True):
                self._validate_strategy_evolution(data, result, context)

            # 6. 과거 데이터와 비교
            if context.should_compare_with_history():
                self._compare_with_history(data, result, context)

            # 7. 이상치 탐지
            if context.validation_options.get("detect_anomalies", True):
                self._detect_anomalies(data, result, context)

        except Exception as e:
            logger.error(f"Self-play validation error: {e}")
            result.add_issue(ValidationIssue(
                check_name="selfplay_validation",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation failed with error: {str(e)}",
                suggestion="Check validation logs for details"
            ))

        result.validation_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    def _validate_data_structure(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """데이터 구조 검증"""
        required_fields = ['episodes', 'total_episodes', 'evolved_strategies']

        for field in required_fields:
            issue = self.check_not_none(data.get(field), field)
            result.add_issue(issue)

        # 에피소드 리스트 검증
        if 'episodes' in data:
            episodes = data['episodes']
            min_episodes = 10  # 최소 에피소드 수
            issue = self.check_list_not_empty(episodes, "episodes", min_size=min_episodes)
            result.add_issue(issue)

    def _validate_episode_results(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """에피소드 결과 일관성 검증"""
        episodes = data.get('episodes', [])
        total_episodes = data.get('total_episodes', 0)

        if not episodes:
            return

        # 에피소드 수 일치 검증
        if len(episodes) != total_episodes:
            result.add_issue(ValidationIssue(
                check_name="episode_count_consistency",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Episode count mismatch: list has {len(episodes)}, total says {total_episodes}",
                expected=total_episodes,
                actual=len(episodes),
                suggestion="Check episode collection logic"
            ))

        # 각 에피소드 결과 검증 (샘플링)
        sample_size = min(20, len(episodes))
        invalid_episodes = []

        for i in range(sample_size):
            episode = episodes[i]

            # 필수 필드 체크
            required_fields = ['agent_id', 'return', 'steps', 'win']
            for field in required_fields:
                if field not in episode:
                    invalid_episodes.append((i, f"missing {field}"))
                    break

            # 수익률 범위 체크
            if 'return' in episode:
                ret = episode['return']
                if not (-1.0 <= ret <= 5.0):  # -100% ~ +500% 허용
                    invalid_episodes.append((i, f"invalid return: {ret}"))

            # 스텝 수 체크
            if 'steps' in episode:
                steps = episode['steps']
                if not (1 <= steps <= 10000):
                    invalid_episodes.append((i, f"invalid steps: {steps}"))

        if invalid_episodes:
            result.add_issue(ValidationIssue(
                check_name="episode_data_validity",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Found {len(invalid_episodes)} invalid episodes",
                actual=str(invalid_episodes[:5]),  # 처음 5개만
                suggestion="Review episode data generation"
            ))
        else:
            result.add_issue(ValidationIssue(
                check_name="episode_data_validity",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="All sampled episodes have valid data"
            ))

    def _validate_prediction_accuracy(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """예측 정확도 검증"""
        pred_accuracy = data.get('prediction_accuracy')

        if pred_accuracy is not None:
            # 예측 정확도 범위 (0.4 ~ 0.8이 정상적)
            min_acc = context.get_threshold("min_prediction_accuracy", 0.4)
            max_acc = context.get_threshold("max_prediction_accuracy", 0.8)

            issue = self.check_range(pred_accuracy, min_acc, max_acc,
                                    "prediction_accuracy", ValidationSeverity.MEDIUM)
            result.add_issue(issue)

            # 너무 높은 정확도 경고 (과적합 가능성)
            if pred_accuracy > 0.75:
                result.add_issue(ValidationIssue(
                    check_name="prediction_accuracy_overfitting",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Prediction accuracy {pred_accuracy:.2%} may indicate overfitting",
                    actual=pred_accuracy,
                    suggestion="Review training data diversity and model complexity"
                ))

            # 너무 낮은 정확도 경고
            if pred_accuracy < 0.45:
                result.add_issue(ValidationIssue(
                    check_name="prediction_accuracy_low",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.HIGH,
                    message=f"Prediction accuracy {pred_accuracy:.2%} is below random chance",
                    actual=pred_accuracy,
                    suggestion="Review feature engineering and model architecture"
                ))

    def _validate_performance_metrics(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """성능 메트릭 검증"""
        avg_return = data.get('average_return', 0)
        win_rate = data.get('win_rate', 0)

        # 평균 수익률 검증
        if avg_return is not None:
            min_return = context.get_threshold("min_profit_rate", -0.5)
            max_return = context.get_threshold("max_profit_rate", 2.0)

            issue = self.check_range(avg_return, min_return, max_return,
                                    "average_return", ValidationSeverity.MEDIUM)
            result.add_issue(issue)

        # 승률 검증 (0.3 ~ 0.7이 정상적)
        if win_rate is not None:
            issue = self.check_range(win_rate, 0.2, 0.8, "win_rate", ValidationSeverity.MEDIUM)
            result.add_issue(issue)

        # 수익률과 승률의 일관성 체크
        if avg_return is not None and win_rate is not None:
            # 높은 승률인데 음의 수익률인 경우
            if win_rate > 0.6 and avg_return < 0:
                result.add_issue(ValidationIssue(
                    check_name="performance_consistency",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.HIGH,
                    message=f"Inconsistent metrics: high win rate ({win_rate:.1%}) but negative return ({avg_return:.2f})",
                    suggestion="Check risk-reward ratio and position sizing"
                ))

            # 낮은 승률인데 양의 수익률인 경우 (정상적일 수 있음 - 손절/익절 비율)
            if win_rate < 0.4 and avg_return > 0.1:
                result.add_issue(ValidationIssue(
                    check_name="performance_consistency",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message=f"Low win rate ({win_rate:.1%}) but positive return ({avg_return:.2f}) - good risk management"
                ))

    def _validate_strategy_evolution(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """전략 진화 검증"""
        original_count = data.get('original_strategy_count', 0)
        evolved_strategies = data.get('evolved_strategies', [])

        if not evolved_strategies:
            result.add_issue(ValidationIssue(
                check_name="strategy_evolution",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.HIGH,
                message="No evolved strategies produced",
                suggestion="Check self-play evolution logic"
            ))
            return

        # 진화 후 전략 수 변화 체크
        evolved_count = len(evolved_strategies)
        if original_count > 0:
            reduction_rate = 1 - (evolved_count / original_count)

            if reduction_rate > 0.9:  # 90% 이상 감소
                result.add_issue(ValidationIssue(
                    check_name="strategy_evolution_reduction",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.HIGH,
                    message=f"Excessive strategy reduction: {original_count} → {evolved_count} ({reduction_rate:.1%} reduction)",
                    expected=f"< 50% reduction",
                    actual=f"{reduction_rate:.1%} reduction",
                    suggestion="Review selection criteria - may be too strict"
                ))
            else:
                result.add_issue(ValidationIssue(
                    check_name="strategy_evolution_reduction",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message=f"Strategy evolution: {original_count} → {evolved_count} strategies"
                ))

        # 진화된 전략 품질 체크 (샘플)
        sample_size = min(10, len(evolved_strategies))
        high_quality_count = 0

        for strategy in evolved_strategies[:sample_size]:
            if 'quality_grade' in strategy:
                if strategy['quality_grade'] in ['S', 'A']:
                    high_quality_count += 1

        quality_rate = high_quality_count / sample_size if sample_size > 0 else 0

        if quality_rate < 0.2:  # 20% 미만이 고품질
            result.add_issue(ValidationIssue(
                check_name="evolved_strategy_quality",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Low quality rate in evolved strategies: {quality_rate:.1%} are S/A grade",
                suggestion="Review evolution fitness function"
            ))
        else:
            result.add_issue(ValidationIssue(
                check_name="evolved_strategy_quality",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"Good quality rate: {quality_rate:.1%} are S/A grade"
            ))

    def _compare_with_history(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """과거 Self-play 결과와 비교"""
        if not self.db_connections or 'learning_results' not in self.db_connections:
            return

        try:
            coin = data.get('coin')
            interval = data.get('interval')
            current_accuracy = data.get('prediction_accuracy', 0)
            current_return = data.get('average_return', 0)

            # 과거 7일간 Self-play 결과 조회
            historical_data = self.get_historical_data(
                self.db_connections['learning_results'],
                """
                SELECT prediction_accuracy, average_return
                FROM selfplay_results
                WHERE coin = ? AND interval = ?
                AND created_at > datetime('now', '-7 days')
                ORDER BY created_at DESC
                LIMIT 10
                """,
                (coin, interval)
            )

            if historical_data:
                # 예측 정확도 비교
                hist_accuracies = [d['prediction_accuracy'] for d in historical_data if d['prediction_accuracy']]
                if hist_accuracies and current_accuracy:
                    accuracy_issue = self.compare_with_history(
                        current_accuracy, hist_accuracies, "prediction_accuracy"
                    )
                    if accuracy_issue:
                        result.add_issue(accuracy_issue)

                # 수익률 비교
                hist_returns = [d['average_return'] for d in historical_data if d['average_return'] is not None]
                if hist_returns and current_return is not None:
                    return_issue = self.compare_with_history(
                        current_return, hist_returns, "average_return"
                    )
                    if return_issue:
                        result.add_issue(return_issue)

        except Exception as e:
            logger.warning(f"Historical comparison failed: {e}")

    def _detect_anomalies(self, data: Dict[str, Any], result: ValidationResult, context: ValidationContext):
        """이상치 탐지"""
        episodes = data.get('episodes', [])

        if not episodes:
            return

        # 에피소드 수익률 분포 이상치
        returns = [e.get('return', 0) for e in episodes if 'return' in e]
        if returns:
            anomaly_issues = self.detect_anomalies(returns, "episode_returns")
            for issue in anomaly_issues:
                result.add_issue(issue)

        # 스텝 수 분포 이상치
        steps = [e.get('steps', 0) for e in episodes if 'steps' in e]
        if steps:
            anomaly_issues = self.detect_anomalies(steps, "episode_steps")
            for issue in anomaly_issues:
                result.add_issue(issue)