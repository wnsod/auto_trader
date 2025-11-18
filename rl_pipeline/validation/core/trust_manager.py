"""
신뢰도 기반 적응형 검증 관리자
검증 성공률에 따라 검증 깊이를 자동으로 조절
"""

import os
import json
import sqlite3
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    """검증 신뢰도 레벨"""
    PARANOID = 0     # 초기 단계: 모든 것을 철저히 검증 (성공률 < 80%)
    CAUTIOUS = 1     # 신뢰도 낮음: 대부분 검증 (성공률 80-90%)
    MODERATE = 2     # 신뢰도 중간: 중요 포인트만 (성공률 90-95%)
    CONFIDENT = 3    # 신뢰도 높음: 핵심만 검증 (성공률 95-98%)
    TRUSTED = 4      # 신뢰도 매우 높음: 최소 검증 (성공률 > 98%)

    @classmethod
    def from_success_rate(cls, success_rate: float) -> 'TrustLevel':
        """성공률에 따른 신뢰도 레벨 결정"""
        if success_rate < 0.80:
            return cls.PARANOID
        elif success_rate < 0.90:
            return cls.CAUTIOUS
        elif success_rate < 0.95:
            return cls.MODERATE
        elif success_rate < 0.98:
            return cls.CONFIDENT
        else:
            return cls.TRUSTED

@dataclass
class ValidationMetric:
    """검증 메트릭 데이터"""
    component: str
    success_count: int = 0
    total_count: int = 0
    consecutive_success: int = 0
    consecutive_failure: int = 0
    last_failure_time: Optional[str] = None
    last_failure_reason: Optional[str] = None
    current_trust_level: str = "PARANOID"
    success_rate: float = 0.0

    def update_metrics(self, success: bool, failure_reason: str = None):
        """메트릭 업데이트"""
        self.total_count += 1

        if success:
            self.success_count += 1
            self.consecutive_success += 1
            self.consecutive_failure = 0
        else:
            self.consecutive_failure += 1
            self.consecutive_success = 0
            self.last_failure_time = datetime.now().isoformat()
            self.last_failure_reason = failure_reason

        # 성공률 재계산
        if self.total_count > 0:
            self.success_rate = self.success_count / self.total_count

        # 신뢰도 레벨 업데이트
        self.current_trust_level = TrustLevel.from_success_rate(self.success_rate).name

class TrustManager:
    """신뢰도 관리자 - 검증 히스토리 추적 및 신뢰도 레벨 관리"""

    # 신뢰도 조정을 위한 임계값
    PROMOTION_THRESHOLD = 20    # 연속 성공 시 승급
    DEMOTION_THRESHOLD = 3      # 연속 실패 시 강등
    MIN_SAMPLES = 10             # 최소 샘플 수
    RECOVERY_PERIOD_DAYS = 7    # 실패 후 회복 기간

    def __init__(self, db_path: str = None):
        """초기화"""
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'validation_metrics.db'
            )
        self.db_path = db_path
        self.metrics: Dict[str, ValidationMetric] = {}
        self._init_database()
        self._load_metrics()

    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    component TEXT PRIMARY KEY,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    consecutive_success INTEGER DEFAULT 0,
                    consecutive_failure INTEGER DEFAULT 0,
                    last_failure_time TEXT,
                    last_failure_reason TEXT,
                    current_trust_level TEXT DEFAULT 'PARANOID',
                    success_rate REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 상세 로그 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    details TEXT,
                    trust_level TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 인덱스 생성
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_component
                ON validation_history(component, created_at DESC)
            """)

    def _load_metrics(self):
        """저장된 메트릭 로드"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT component, success_count, total_count,
                       consecutive_success, consecutive_failure,
                       last_failure_time, last_failure_reason,
                       current_trust_level, success_rate
                FROM validation_metrics
            """)

            for row in cursor.fetchall():
                self.metrics[row[0]] = ValidationMetric(
                    component=row[0],
                    success_count=row[1],
                    total_count=row[2],
                    consecutive_success=row[3],
                    consecutive_failure=row[4],
                    last_failure_time=row[5],
                    last_failure_reason=row[6],
                    current_trust_level=row[7],
                    success_rate=row[8]
                )

    def _save_metric(self, metric: ValidationMetric):
        """메트릭 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO validation_metrics
                (component, success_count, total_count,
                 consecutive_success, consecutive_failure,
                 last_failure_time, last_failure_reason,
                 current_trust_level, success_rate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.component,
                metric.success_count,
                metric.total_count,
                metric.consecutive_success,
                metric.consecutive_failure,
                metric.last_failure_time,
                metric.last_failure_reason,
                metric.current_trust_level,
                metric.success_rate,
                datetime.now().isoformat()
            ))

    def get_trust_level(self, component: str) -> TrustLevel:
        """컴포넌트의 현재 신뢰도 레벨 조회"""
        if component not in self.metrics:
            # 새로운 컴포넌트는 PARANOID로 시작
            self.metrics[component] = ValidationMetric(component=component)
            self._save_metric(self.metrics[component])
            return TrustLevel.PARANOID

        metric = self.metrics[component]

        # 최소 샘플 수 미달 시 PARANOID
        if metric.total_count < self.MIN_SAMPLES:
            return TrustLevel.PARANOID

        # 최근 실패가 있었다면 회복 기간 체크
        if metric.last_failure_time:
            last_failure = datetime.fromisoformat(metric.last_failure_time)
            if datetime.now() - last_failure < timedelta(days=self.RECOVERY_PERIOD_DAYS):
                # 회복 기간 중에는 한 단계 낮은 레벨 적용
                current_level = TrustLevel[metric.current_trust_level]
                if current_level.value > 0:
                    return TrustLevel(current_level.value - 1)

        return TrustLevel[metric.current_trust_level]

    def update_trust(self, component: str, success: bool,
                    failure_reason: str = None, details: Dict[str, Any] = None):
        """검증 결과에 따른 신뢰도 업데이트"""
        if component not in self.metrics:
            self.metrics[component] = ValidationMetric(component=component)

        metric = self.metrics[component]
        old_level = TrustLevel[metric.current_trust_level]

        # 메트릭 업데이트
        metric.update_metrics(success, failure_reason)

        # 신뢰도 레벨 조정 로직
        new_level = self._adjust_trust_level(metric, old_level)
        metric.current_trust_level = new_level.name

        # 데이터베이스에 저장
        self._save_metric(metric)
        self._save_history(component, success, failure_reason, details, new_level)

        # 레벨 변경 시 로깅
        if old_level != new_level:
            logger.info(f"🎚️ {component} 신뢰도 레벨 변경: {old_level.name} → {new_level.name}")
            logger.info(f"   성공률: {metric.success_rate:.1%}, 연속 성공: {metric.consecutive_success}")

    def _adjust_trust_level(self, metric: ValidationMetric, current_level: TrustLevel) -> TrustLevel:
        """신뢰도 레벨 조정 로직"""
        # 연속 실패 시 즉시 강등
        if metric.consecutive_failure >= self.DEMOTION_THRESHOLD:
            if current_level.value > 0:
                logger.warning(f"⚠️ {metric.component}: 연속 {metric.consecutive_failure}회 실패로 신뢰도 강등")
                return TrustLevel(max(0, current_level.value - 2))  # 두 단계 강등

        # 연속 성공 시 점진적 승급
        if metric.consecutive_success >= self.PROMOTION_THRESHOLD:
            if current_level.value < 4 and metric.success_rate >= 0.95:
                logger.info(f"✅ {metric.component}: 연속 {metric.consecutive_success}회 성공으로 신뢰도 승급")
                return TrustLevel(min(4, current_level.value + 1))

        # 성공률 기반 자동 조정
        suggested_level = TrustLevel.from_success_rate(metric.success_rate)

        # 급격한 변화 방지 (한 번에 한 단계씩만 이동)
        if abs(suggested_level.value - current_level.value) > 1:
            if suggested_level.value > current_level.value:
                return TrustLevel(current_level.value + 1)
            else:
                return TrustLevel(current_level.value - 1)

        return suggested_level

    def _save_history(self, component: str, success: bool,
                     failure_reason: str, details: Dict, trust_level: TrustLevel):
        """검증 히스토리 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO validation_history
                (component, validation_type, success, failure_reason, details, trust_level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                component,
                details.get('validation_type', 'general') if details else 'general',
                success,
                failure_reason,
                json.dumps(details) if details else None,
                trust_level.name
            ))

    def get_component_stats(self, component: str) -> Dict[str, Any]:
        """컴포넌트의 상세 통계 조회"""
        if component not in self.metrics:
            return {"error": f"No metrics for component: {component}"}

        metric = self.metrics[component]

        # 최근 검증 히스토리 조회
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT success, failure_reason, created_at
                FROM validation_history
                WHERE component = ?
                ORDER BY created_at DESC
                LIMIT 20
            """, (component,))

            recent_history = [
                {
                    "success": bool(row[0]),
                    "failure_reason": row[1],
                    "timestamp": row[2]
                }
                for row in cursor.fetchall()
            ]

        return {
            "component": component,
            "current_trust_level": metric.current_trust_level,
            "success_rate": metric.success_rate,
            "total_validations": metric.total_count,
            "consecutive_success": metric.consecutive_success,
            "consecutive_failure": metric.consecutive_failure,
            "last_failure": {
                "time": metric.last_failure_time,
                "reason": metric.last_failure_reason
            } if metric.last_failure_time else None,
            "recent_history": recent_history,
            "recommended_action": self._get_recommendation(metric)
        }

    def _get_recommendation(self, metric: ValidationMetric) -> str:
        """메트릭 기반 권장 조치"""
        if metric.consecutive_failure >= self.DEMOTION_THRESHOLD:
            return "🔴 긴급 점검 필요 - 연속 실패 발생"
        elif metric.success_rate < 0.80:
            return "⚠️ 코드 개선 필요 - 낮은 성공률"
        elif metric.success_rate < 0.95:
            return "📊 모니터링 지속 - 개선 중"
        else:
            return "✅ 안정적 운영 중"

    def get_global_stats(self) -> Dict[str, Any]:
        """전체 시스템 통계"""
        total_components = len(self.metrics)

        if total_components == 0:
            return {"status": "No validation data yet"}

        avg_success_rate = sum(m.success_rate for m in self.metrics.values()) / total_components

        trust_distribution = {}
        for level in TrustLevel:
            count = sum(1 for m in self.metrics.values() if m.current_trust_level == level.name)
            trust_distribution[level.name] = count

        problematic_components = [
            {
                "component": m.component,
                "success_rate": m.success_rate,
                "consecutive_failure": m.consecutive_failure
            }
            for m in self.metrics.values()
            if m.success_rate < 0.90 or m.consecutive_failure >= 2
        ]

        return {
            "total_components": total_components,
            "average_success_rate": avg_success_rate,
            "trust_distribution": trust_distribution,
            "problematic_components": problematic_components,
            "system_health": self._get_system_health(avg_success_rate)
        }

    def _get_system_health(self, avg_success_rate: float) -> str:
        """시스템 전체 건강도 평가"""
        if avg_success_rate >= 0.95:
            return "🟢 Excellent"
        elif avg_success_rate >= 0.90:
            return "🟡 Good"
        elif avg_success_rate >= 0.80:
            return "🟠 Fair"
        else:
            return "🔴 Poor - Immediate attention required"

    def reset_component(self, component: str):
        """특정 컴포넌트의 신뢰도 초기화 (코드 수정 후 사용)"""
        if component in self.metrics:
            logger.info(f"🔄 {component} 신뢰도 초기화 (코드 수정 후)")
            self.metrics[component] = ValidationMetric(component=component)
            self._save_metric(self.metrics[component])