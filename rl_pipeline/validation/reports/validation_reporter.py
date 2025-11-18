"""
검증 리포트 생성 및 저장
Claude가 읽을 수 있는 형식으로 검증 결과 저장
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..core.validation_result import ValidationResult, ValidationStatus
from ..recovery.recovery_engine import CodeSuggestion

logger = logging.getLogger(__name__)

class ValidationReporter:
    """검증 결과 리포터"""

    def __init__(self, output_dir: str = None):
        """초기화

        Args:
            output_dir: 리포트 저장 디렉토리
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'reports'
            )

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 리포트 파일 경로
        self.jsonl_path = os.path.join(output_dir, 'validation_log.jsonl')
        self.summary_path = os.path.join(output_dir, 'validation_summary.md')
        self.suggestions_path = os.path.join(output_dir, 'code_suggestions.json')

    def save_validation_result(self, result: ValidationResult,
                               recovery_result: Optional[Dict[str, Any]] = None,
                               context: Optional[Dict[str, Any]] = None):
        """검증 결과 저장

        Args:
            result: 검증 결과
            recovery_result: 복구 시도 결과
            context: 추가 컨텍스트 정보
        """
        # JSONL 형식으로 저장 (Claude가 읽기 좋은 형식)
        self._save_to_jsonl(result, recovery_result, context)

        # Markdown 요약 업데이트
        self._update_summary(result, recovery_result)

        # 코드 제안사항 저장
        if recovery_result and 'suggestions' in recovery_result:
            self._save_suggestions(recovery_result['suggestions'])

    def _save_to_jsonl(self, result: ValidationResult,
                       recovery_result: Optional[Dict],
                       context: Optional[Dict]):
        """JSONL 형식으로 저장"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "validation_id": result.validation_id,
            "component": result.component,
            "status": result.overall_status.value,
            "success_rate": result.get_success_rate(),
            "has_critical_issues": result.has_critical_issues(),
            "context": context or {},
            "statistics": {
                "total_checks": result.total_checks,
                "passed": result.passed_checks,
                "failed": result.failed_checks,
                "warnings": result.warning_checks,
                "auto_fixed": result.auto_fixed_count
            },
            "issues": [
                {
                    "check": issue.check_name,
                    "status": issue.status.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "expected": issue.expected,
                    "actual": issue.actual,
                    "suggestion": issue.suggestion,
                    "auto_fixed": issue.auto_fixed
                }
                for issue in result.issues
                if issue.status != ValidationStatus.PASSED  # PASSED는 제외하여 크기 줄임
            ],
            "recovery": recovery_result if recovery_result else None
        }

        try:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            logger.debug(f"Validation result saved to {self.jsonl_path}")

        except Exception as e:
            logger.error(f"Failed to save validation result: {e}")

    def _update_summary(self, result: ValidationResult, recovery_result: Optional[Dict]):
        """Markdown 요약 업데이트"""
        try:
            # 기존 요약 로드 또는 새로 생성
            if os.path.exists(self.summary_path):
                with open(self.summary_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = self._create_summary_header()

            # 새 엔트리 추가
            new_entry = self._create_summary_entry(result, recovery_result)

            # 요약 통계 업데이트
            content = self._update_summary_stats(content, result)

            # 최신 엔트리를 상단에 추가
            insertion_point = content.find("## Recent Validations")
            if insertion_point != -1:
                insertion_point = content.find("\n", insertion_point) + 1
                content = content[:insertion_point] + new_entry + "\n" + content[insertion_point:]
            else:
                content += "\n## Recent Validations\n\n" + new_entry

            # 파일 저장
            with open(self.summary_path, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            logger.error(f"Failed to update summary: {e}")

    def _create_summary_header(self) -> str:
        """요약 헤더 생성"""
        return f"""# Absolute Zero System Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Health Overview

| Metric | Value |
|--------|-------|
| Total Validations | 0 |
| Success Rate | 0.0% |
| Critical Issues | 0 |
| Auto-Fixed Issues | 0 |
| Last Updated | {datetime.now().isoformat()} |

## Component Status

| Component | Trust Level | Success Rate | Last Check |
|-----------|-------------|--------------|------------|

"""

    def _create_summary_entry(self, result: ValidationResult, recovery_result: Optional[Dict]) -> str:
        """요약 엔트리 생성"""
        status_emoji = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.SKIPPED: "⏭️",
            ValidationStatus.FIXED: "🔧"
        }

        emoji = status_emoji.get(result.overall_status, "❓")

        entry = f"""### {emoji} {result.component} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

- **Validation ID**: `{result.validation_id}`
- **Status**: {result.overall_status.value}
- **Success Rate**: {result.get_success_rate():.1%}
- **Checks**: Total {result.total_checks} (✅ {result.passed_checks}, ❌ {result.failed_checks}, ⚠️ {result.warning_checks})
"""

        # Critical issues 표시
        if result.has_critical_issues():
            critical_issues = result.get_failed_issues()
            entry += "\n#### 🔴 Critical Issues:\n"
            for issue in critical_issues[:3]:  # 최대 3개만
                entry += f"- **{issue.check_name}**: {issue.message}\n"
                if issue.suggestion:
                    entry += f"  - 💡 {issue.suggestion}\n"

        # Recovery 결과 표시
        if recovery_result and recovery_result.get('recovered'):
            entry += "\n#### 🔧 Auto-Recovery:\n"
            entry += f"- Recovery successful: {recovery_result['recovered']}\n"
            if 'actions' in recovery_result:
                for action in recovery_result['actions'][:3]:
                    if action.get('success'):
                        entry += f"- ✅ {action.get('description', 'Action completed')}\n"

        entry += "\n---\n"

        return entry

    def _update_summary_stats(self, content: str, result: ValidationResult) -> str:
        """요약 통계 업데이트"""
        # 간단한 구현 - 실제로는 DB에서 집계하는 것이 좋음
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if "| Total Validations |" in line:
                # 숫자 추출 및 증가
                try:
                    parts = line.split('|')
                    current = int(parts[2].strip())
                    lines[i] = f"| Total Validations | {current + 1} |"
                except:
                    pass

            elif "| Last Updated |" in line:
                lines[i] = f"| Last Updated | {datetime.now().isoformat()} |"

        return '\n'.join(lines)

    def _save_suggestions(self, suggestions: List[CodeSuggestion]):
        """코드 제안사항 저장"""
        if not suggestions:
            return

        try:
            # 기존 제안사항 로드
            if os.path.exists(self.suggestions_path):
                with open(self.suggestions_path, 'r', encoding='utf-8') as f:
                    all_suggestions = json.load(f)
            else:
                all_suggestions = []

            # 새 제안사항 추가
            for suggestion in suggestions:
                all_suggestions.append({
                    "timestamp": datetime.now().isoformat(),
                    "file_path": suggestion.file_path,
                    "line_number": suggestion.line_number,
                    "issue_type": suggestion.issue_type,
                    "severity": suggestion.severity,
                    "confidence": suggestion.confidence,
                    "explanation": suggestion.explanation,
                    "suggested_code": suggestion.suggested_code
                })

            # 최근 100개만 유지
            all_suggestions = all_suggestions[-100:]

            # 저장
            with open(self.suggestions_path, 'w', encoding='utf-8') as f:
                json.dump(all_suggestions, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save code suggestions: {e}")

    def generate_daily_report(self) -> Dict[str, Any]:
        """일일 리포트 생성"""
        try:
            # 오늘의 검증 결과 집계
            today = datetime.now().date()
            today_results = []

            if os.path.exists(self.jsonl_path):
                with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        entry_date = datetime.fromisoformat(entry['timestamp']).date()
                        if entry_date == today:
                            today_results.append(entry)

            # 통계 계산
            if today_results:
                total = len(today_results)
                success_count = sum(1 for r in today_results if r['status'] == 'passed')
                critical_count = sum(1 for r in today_results if r['has_critical_issues'])
                auto_fixed = sum(r['statistics']['auto_fixed'] for r in today_results)

                avg_success_rate = sum(r['success_rate'] for r in today_results) / total

                # 컴포넌트별 통계
                component_stats = {}
                for r in today_results:
                    comp = r['component']
                    if comp not in component_stats:
                        component_stats[comp] = {
                            'total': 0,
                            'passed': 0,
                            'failed': 0,
                            'success_rates': []
                        }

                    component_stats[comp]['total'] += 1
                    if r['status'] == 'passed':
                        component_stats[comp]['passed'] += 1
                    else:
                        component_stats[comp]['failed'] += 1
                    component_stats[comp]['success_rates'].append(r['success_rate'])

                # 평균 계산
                for comp in component_stats:
                    rates = component_stats[comp]['success_rates']
                    component_stats[comp]['avg_success_rate'] = sum(rates) / len(rates)
                    del component_stats[comp]['success_rates']  # 리스트 제거

                return {
                    'date': today.isoformat(),
                    'total_validations': total,
                    'success_count': success_count,
                    'success_rate': success_count / total,
                    'critical_issues': critical_count,
                    'auto_fixed_issues': auto_fixed,
                    'average_success_rate': avg_success_rate,
                    'component_stats': component_stats,
                    'most_problematic': min(component_stats.items(),
                                           key=lambda x: x[1]['avg_success_rate'])[0]
                                       if component_stats else None
                }
            else:
                return {
                    'date': today.isoformat(),
                    'message': 'No validations performed today'
                }

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return {'error': str(e)}

    def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 실패 항목 조회"""
        failures = []

        try:
            if os.path.exists(self.jsonl_path):
                with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                    # 파일 끝에서부터 읽기 (최신 항목)
                    lines = f.readlines()
                    for line in reversed(lines):
                        if len(failures) >= limit:
                            break

                        entry = json.loads(line)
                        if entry['status'] == 'failed' or entry['has_critical_issues']:
                            failures.append({
                                'timestamp': entry['timestamp'],
                                'component': entry['component'],
                                'validation_id': entry['validation_id'],
                                'issues': entry['issues'][:3],  # 상위 3개 이슈만
                                'success_rate': entry['success_rate']
                            })

        except Exception as e:
            logger.error(f"Failed to get recent failures: {e}")

        return failures