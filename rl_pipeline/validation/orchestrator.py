"""
검증 오케스트레이터 - 전체 검증 시스템 통합 관리
absolute_zero_system.py와 통합되는 메인 인터페이스
"""

import os
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

from .core.trust_manager import TrustManager, TrustLevel
from .core.validation_context import ValidationContext
from .core.validation_result import ValidationResult, ValidationStatus
from .validators.strategy_validator import StrategyValidator
from .validators.selfplay_validator import SelfPlayValidator
from .validators.routing_validator import RoutingValidator
from .recovery.recovery_engine import RecoveryEngine
from .reports.validation_reporter import ValidationReporter

logger = logging.getLogger(__name__)

class ValidationOrchestrator:
    """검증 시스템 통합 오케스트레이터"""

    def __init__(self, db_connections: Dict[str, str] = None, enable_auto_fix: bool = True):
        """초기화

        Args:
            db_connections: 데이터베이스 연결 정보
                - strategies: 전략 DB 경로
                - learning_results: 학습 결과 DB 경로
            enable_auto_fix: 자동 복구 활성화 여부
        """
        # 기본 DB 경로 설정
        if db_connections is None:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_storage')
            db_connections = {
                'strategies': os.path.join(base_path, 'learning_strategies.db'),
                # learning_results는 이제 learning_strategies.db로 통합됨
                'learning_results': os.path.join(base_path, 'learning_strategies.db')
            }

        self.db_connections = db_connections
        self.enable_auto_fix = enable_auto_fix

        # 컴포넌트 초기화
        self.trust_manager = TrustManager()
        self.recovery_engine = RecoveryEngine(enable_auto_fix=enable_auto_fix)
        self.reporter = ValidationReporter()

        # Validator 초기화
        self.validators = {
            'strategy': StrategyValidator(db_connections),
            'selfplay': SelfPlayValidator(db_connections),
            'routing': RoutingValidator(db_connections),
        }

        # 통계
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'auto_fixed': 0
        }

    def validate_pipeline_stage(self, stage: str, data: Dict[str, Any],
                               coin: str, interval: str,
                               pipeline_run_id: Optional[str] = None) -> ValidationResult:
        """파이프라인 스테이지별 검증 실행

        Args:
            stage: 스테이지 이름 (strategy_generation, selfplay, routing, analysis)
            data: 검증할 데이터
            coin: 코인 심볼
            interval: 시간 인터벌
            pipeline_run_id: 파이프라인 실행 ID

        Returns:
            ValidationResult: 검증 결과
        """
        logger.info(f"🔍 Starting validation for {stage} - {coin}/{interval}")

        # 신뢰도 레벨 조회
        component = self._get_component_name(stage)
        trust_level = self.trust_manager.get_trust_level(component)

        logger.info(f"   Trust level: {trust_level.name}")

        # 컨텍스트 생성
        context = ValidationContext(
            coin=coin,
            interval=interval,
            stage=stage,
            trust_level=trust_level.name,
            enable_auto_fix=self.enable_auto_fix,
            pipeline_run_id=pipeline_run_id
        )

        # 신뢰도에 따른 검증 옵션 자동 조정
        context.update_from_trust_level(trust_level.name)

        # Validator 선택 및 실행
        validator = self._select_validator(stage)

        if validator is None:
            logger.warning(f"No validator for stage: {stage}")
            return self._create_skipped_result(stage, context)

        # 검증 실행
        try:
            start_time = time.time()
            result = validator.validate(data, context)
            validation_time = (time.time() - start_time) * 1000

            result.validation_duration_ms = validation_time
            logger.info(f"   Validation completed in {validation_time:.0f}ms")

        except Exception as e:
            logger.error(f"Validation error for {stage}: {e}")
            result = self._create_error_result(stage, context, str(e))

        # 검증 실패 시 복구 시도
        recovery_result = None
        if not result.is_successful() and self.enable_auto_fix:
            logger.info(f"🔧 Attempting recovery for {stage}")
            recovery_result = self.recovery_engine.attempt_recovery(result, data)

            if recovery_result['recovered']:
                logger.info(f"✅ Recovery successful for {stage}")
                result.overall_status = ValidationStatus.FIXED
                self.validation_stats['auto_fixed'] += 1

                # 복구된 데이터 반환
                data.update(recovery_result['fixed_data'])

        # 신뢰도 업데이트
        self.trust_manager.update_trust(
            component,
            result.is_successful(),
            failure_reason=self._get_failure_reason(result),
            details={'stage': stage, 'coin': coin, 'interval': interval}
        )

        # 리포트 저장
        self.reporter.save_validation_result(
            result,
            recovery_result,
            {'coin': coin, 'interval': interval, 'stage': stage}
        )

        # 통계 업데이트
        self._update_stats(result)

        # 결과 요약 로그
        self._log_result_summary(result, stage, trust_level)

        return result

    def validate_full_pipeline(self, pipeline_results: Dict[str, Any],
                              coin: str, intervals: List[str]) -> Dict[str, ValidationResult]:
        """전체 파이프라인 결과 검증

        Args:
            pipeline_results: 각 스테이지별 결과 데이터
            coin: 코인 심볼
            intervals: 인터벌 리스트

        Returns:
            Dict[str, ValidationResult]: 스테이지별 검증 결과
        """
        all_results = {}

        # 각 인터벌별로 검증
        for interval in intervals:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Validating pipeline for {coin}/{interval}")
            logger.info(f"{'='*60}")

            interval_results = {}

            # 1. 전략 생성 검증
            if 'strategies' in pipeline_results:
                strategy_data = pipeline_results['strategies'].get(interval, {})
                strategy_data.update({'coin': coin, 'interval': interval})
                interval_results['strategy'] = self.validate_pipeline_stage(
                    'strategy_generation',
                    strategy_data,
                    coin,
                    interval
                )

            # 2. Self-play 검증
            if 'selfplay' in pipeline_results:
                selfplay_data = pipeline_results['selfplay'].get(interval, {})
                selfplay_data.update({'coin': coin, 'interval': interval})
                interval_results['selfplay'] = self.validate_pipeline_stage(
                    'selfplay',
                    selfplay_data,
                    coin,
                    interval
                )

            # 3. 라우팅 검증
            if 'routing' in pipeline_results:
                routing_data = pipeline_results['routing'].get(interval, {})
                routing_data.update({'coin': coin, 'interval': interval})
                interval_results['routing'] = self.validate_pipeline_stage(
                    'routing',
                    routing_data,
                    coin,
                    interval
                )

            all_results[interval] = interval_results

        # 전체 요약
        self._log_overall_summary(all_results, coin)

        return all_results

    def _select_validator(self, stage: str):
        """스테이지에 맞는 Validator 선택"""
        stage_mapping = {
            'strategy_generation': self.validators['strategy'],
            'strategy': self.validators['strategy'],
            'selfplay': self.validators['selfplay'],
            'self_play': self.validators['selfplay'],
            'routing': self.validators['routing'],
            'regime_routing': self.validators['routing'],
        }

        return stage_mapping.get(stage.lower())

    def _get_component_name(self, stage: str) -> str:
        """스테이지명을 컴포넌트명으로 변환"""
        mapping = {
            'strategy_generation': 'Strategy',
            'strategy': 'Strategy',
            'selfplay': 'SelfPlay',
            'self_play': 'SelfPlay',
            'routing': 'Routing',
            'regime_routing': 'Routing',
            'analysis': 'Analysis',
            'paper_trading': 'PaperTrading',
            'global_strategy': 'GlobalStrategy'
        }

        return mapping.get(stage.lower(), stage)

    def _create_skipped_result(self, stage: str, context: ValidationContext) -> ValidationResult:
        """스킵된 검증 결과 생성"""
        result = ValidationResult(
            component=self._get_component_name(stage),
            validation_id=f"skip_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_status=ValidationStatus.SKIPPED
        )
        return result

    def _create_error_result(self, stage: str, context: ValidationContext, error: str) -> ValidationResult:
        """에러 검증 결과 생성"""
        result = ValidationResult(
            component=self._get_component_name(stage),
            validation_id=f"error_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_status=ValidationStatus.FAILED
        )
        result.metadata['error'] = error
        return result

    def _get_failure_reason(self, result: ValidationResult) -> Optional[str]:
        """검증 실패 이유 추출"""
        failed_issues = result.get_failed_issues()
        if failed_issues:
            return failed_issues[0].message
        return None

    def _update_stats(self, result: ValidationResult):
        """통계 업데이트"""
        self.validation_stats['total_validations'] += 1

        if result.is_successful():
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1

    def _log_result_summary(self, result: ValidationResult, stage: str, trust_level: TrustLevel):
        """검증 결과 요약 로그"""
        status_emoji = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.SKIPPED: "⏭️",
            ValidationStatus.FIXED: "🔧"
        }

        emoji = status_emoji.get(result.overall_status, "❓")

        logger.info(f"{emoji} {stage} validation: {result.overall_status.value}")
        logger.info(f"   Success rate: {result.get_success_rate():.1%} "
                   f"({result.passed_checks}/{result.total_checks} checks)")

        # Critical issues 로그
        if result.has_critical_issues():
            critical = result.get_failed_issues()[:3]
            logger.warning(f"   🔴 Critical issues found:")
            for issue in critical:
                logger.warning(f"      - {issue.check_name}: {issue.message}")

        # 신뢰도 변화 로그
        new_trust = self.trust_manager.get_trust_level(self._get_component_name(stage))
        if new_trust != trust_level:
            if new_trust.value > trust_level.value:
                logger.info(f"   📈 Trust level improved: {trust_level.name} → {new_trust.name}")
            else:
                logger.warning(f"   📉 Trust level decreased: {trust_level.name} → {new_trust.name}")

    def _log_overall_summary(self, all_results: Dict[str, Dict[str, ValidationResult]], coin: str):
        """전체 검증 요약"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Overall Validation Summary for {coin}")
        logger.info(f"{'='*60}")

        total_checks = 0
        total_passed = 0
        total_failed = 0
        critical_count = 0

        for interval, results in all_results.items():
            interval_checks = 0
            interval_passed = 0

            for stage, result in results.items():
                total_checks += result.total_checks
                total_passed += result.passed_checks
                total_failed += result.failed_checks
                interval_checks += result.total_checks
                interval_passed += result.passed_checks

                if result.has_critical_issues():
                    critical_count += 1

            interval_rate = (interval_passed / interval_checks * 100) if interval_checks > 0 else 0
            logger.info(f"   {interval}: {interval_rate:.1f}% success "
                       f"({interval_passed}/{interval_checks} checks)")

        overall_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0

        logger.info(f"\n   Total: {overall_rate:.1f}% success rate")
        logger.info(f"   Checks: {total_checks} (✅ {total_passed}, ❌ {total_failed})")

        if critical_count > 0:
            logger.warning(f"   ⚠️ {critical_count} stages with critical issues")

        # 시스템 건강도
        health = self._get_system_health(overall_rate, critical_count)
        logger.info(f"\n   System Health: {health}")

        # 통계
        logger.info(f"\n   Cumulative Stats:")
        logger.info(f"   - Total validations: {self.validation_stats['total_validations']}")
        logger.info(f"   - Successful: {self.validation_stats['successful_validations']}")
        logger.info(f"   - Failed: {self.validation_stats['failed_validations']}")
        logger.info(f"   - Auto-fixed: {self.validation_stats['auto_fixed']}")

    def _get_system_health(self, success_rate: float, critical_count: int) -> str:
        """시스템 건강도 평가"""
        if critical_count > 0:
            return "🔴 Critical - Immediate attention required"
        elif success_rate >= 95:
            return "🟢 Excellent"
        elif success_rate >= 90:
            return "🟡 Good"
        elif success_rate >= 80:
            return "🟠 Fair - Monitoring required"
        else:
            return "🔴 Poor - Investigation needed"

    def get_validation_stats(self) -> Dict[str, Any]:
        """검증 통계 조회"""
        return {
            **self.validation_stats,
            'trust_levels': self.trust_manager.get_global_stats(),
            'recovery_stats': self.recovery_engine.get_recovery_stats(),
            'recent_failures': self.reporter.get_recent_failures(5)
        }

    def generate_report(self) -> Dict[str, Any]:
        """종합 리포트 생성"""
        return self.reporter.generate_daily_report()

    def reset_component_trust(self, component: str):
        """특정 컴포넌트 신뢰도 초기화 (코드 수정 후)"""
        self.trust_manager.reset_component(component)
        logger.info(f"🔄 Reset trust level for {component}")


# absolute_zero_system.py와 통합을 위한 간편 함수
def create_validation_orchestrator(enable_auto_fix: bool = True) -> ValidationOrchestrator:
    """검증 오케스트레이터 생성 헬퍼 함수"""
    return ValidationOrchestrator(enable_auto_fix=enable_auto_fix)


def validate_absolute_zero_stage(stage_name: str, data: Dict[str, Any],
                                coin: str, interval: str,
                                orchestrator: Optional[ValidationOrchestrator] = None) -> ValidationResult:
    """Absolute Zero 시스템 스테이지 검증 헬퍼 함수

    Args:
        stage_name: 스테이지 이름
        data: 검증할 데이터
        coin: 코인 심볼
        interval: 시간 인터벌
        orchestrator: 기존 오케스트레이터 (없으면 새로 생성)

    Returns:
        ValidationResult: 검증 결과
    """
    if orchestrator is None:
        orchestrator = create_validation_orchestrator()

    return orchestrator.validate_pipeline_stage(stage_name, data, coin, interval)