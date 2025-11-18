"""
검증 결과 데이터 구조 및 유틸리티
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

class ValidationStatus(Enum):
    """검증 상태"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    FIXED = "fixed"  # 자동 복구됨

class ValidationSeverity(Enum):
    """문제 심각도"""
    CRITICAL = "critical"  # 시스템 중단 필요
    HIGH = "high"         # 즉시 조치 필요
    MEDIUM = "medium"     # 주의 필요
    LOW = "low"          # 참고사항
    INFO = "info"        # 정보성

@dataclass
class ValidationIssue:
    """검증 이슈 상세 정보"""
    check_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    expected: Any = None
    actual: Any = None
    deviation: Optional[float] = None
    location: Optional[str] = None  # 코드 위치 (file:line)
    suggestion: Optional[str] = None
    auto_fixed: bool = False
    fix_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "deviation": self.deviation,
            "location": self.location,
            "suggestion": self.suggestion,
            "auto_fixed": self.auto_fixed,
            "fix_details": self.fix_details
        }

@dataclass
class ValidationResult:
    """검증 결과 종합"""

    # 기본 정보
    component: str
    validation_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 전체 상태
    overall_status: ValidationStatus = ValidationStatus.PASSED
    issues: List[ValidationIssue] = field(default_factory=list)

    # 통계
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    skipped_checks: int = 0
    auto_fixed_count: int = 0

    # 성능 메트릭
    validation_duration_ms: Optional[float] = None

    # 과거 비교
    comparison_with_previous: Optional[Dict[str, Any]] = None

    # 추가 데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue):
        """이슈 추가 및 통계 업데이트"""
        self.issues.append(issue)
        self.total_checks += 1

        if issue.status == ValidationStatus.PASSED:
            self.passed_checks += 1
        elif issue.status == ValidationStatus.FAILED:
            self.failed_checks += 1
            # 심각도가 CRITICAL이면 전체 상태를 FAILED로
            if issue.severity == ValidationSeverity.CRITICAL:
                self.overall_status = ValidationStatus.FAILED
            elif self.overall_status != ValidationStatus.FAILED:
                self.overall_status = ValidationStatus.WARNING
        elif issue.status == ValidationStatus.WARNING:
            self.warning_checks += 1
            if self.overall_status == ValidationStatus.PASSED:
                self.overall_status = ValidationStatus.WARNING
        elif issue.status == ValidationStatus.SKIPPED:
            self.skipped_checks += 1
        elif issue.status == ValidationStatus.FIXED:
            self.auto_fixed_count += 1
            self.passed_checks += 1  # Fixed도 성공으로 간주

        if issue.auto_fixed:
            self.auto_fixed_count += 1

    def is_successful(self) -> bool:
        """검증 성공 여부"""
        return self.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    def has_critical_issues(self) -> bool:
        """치명적 이슈 존재 여부"""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """심각도별 이슈 조회"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_failed_issues(self) -> List[ValidationIssue]:
        """실패한 검증 항목 조회"""
        return [issue for issue in self.issues if issue.status == ValidationStatus.FAILED]

    def get_success_rate(self) -> float:
        """성공률 계산"""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks + self.auto_fixed_count) / self.total_checks

    def get_fix_suggestions(self) -> List[Dict[str, Any]]:
        """수정 제안 목록"""
        suggestions = []
        for issue in self.issues:
            if issue.suggestion and issue.status == ValidationStatus.FAILED:
                suggestions.append({
                    "check": issue.check_name,
                    "location": issue.location,
                    "problem": issue.message,
                    "suggestion": issue.suggestion,
                    "severity": issue.severity.value
                })
        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """전체 결과를 딕셔너리로 변환 (JSON 저장용)"""
        return {
            "validation_id": self.validation_id,
            "component": self.component,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "statistics": {
                "total_checks": self.total_checks,
                "passed": self.passed_checks,
                "failed": self.failed_checks,
                "warnings": self.warning_checks,
                "skipped": self.skipped_checks,
                "auto_fixed": self.auto_fixed_count,
                "success_rate": self.get_success_rate()
            },
            "has_critical_issues": self.has_critical_issues(),
            "validation_duration_ms": self.validation_duration_ms,
            "issues": [issue.to_dict() for issue in self.issues],
            "fix_suggestions": self.get_fix_suggestions(),
            "comparison": self.comparison_with_previous,
            "metadata": self.metadata
        }

    def get_summary(self) -> str:
        """요약 문자열 생성"""
        status_emoji = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.SKIPPED: "⏭️",
            ValidationStatus.FIXED: "🔧"
        }

        summary = f"""
{status_emoji[self.overall_status]} Validation Result for {self.component}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: {self.overall_status.value.upper()}
Success Rate: {self.get_success_rate():.1%}

Checks: {self.total_checks} total
  ✅ Passed: {self.passed_checks}
  ❌ Failed: {self.failed_checks}
  ⚠️  Warnings: {self.warning_checks}
  🔧 Auto-fixed: {self.auto_fixed_count}
"""

        if self.failed_checks > 0:
            summary += "\nFailed Checks:\n"
            for issue in self.get_failed_issues()[:5]:  # 최대 5개만 표시
                summary += f"  • {issue.check_name}: {issue.message}\n"
                if issue.suggestion:
                    summary += f"    💡 {issue.suggestion}\n"

        return summary