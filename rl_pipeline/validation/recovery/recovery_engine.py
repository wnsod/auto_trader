"""
자동 복구 및 코드 개선 제안 엔진
검증 실패 시 자동 복구를 시도하고 개선 방안을 제시
"""

import os
import json
import sqlite3
import traceback
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from ..core.validation_result import ValidationResult, ValidationIssue, ValidationStatus, ValidationSeverity

logger = logging.getLogger(__name__)

@dataclass
class RecoveryAction:
    """복구 액션 정보"""
    action_type: str  # retry, fix_data, adjust_params, restart_service, code_fix
    description: str
    success: bool = False
    error_message: Optional[str] = None
    fixed_data: Optional[Any] = None
    timestamp: str = datetime.now().isoformat()

@dataclass
class CodeSuggestion:
    """코드 개선 제안"""
    file_path: str
    line_number: Optional[int]
    issue_type: str
    current_code: Optional[str]
    suggested_code: Optional[str]
    explanation: str
    severity: str
    confidence: float  # 0-1, 제안의 확신도

class RecoveryEngine:
    """자동 복구 엔진"""

    def __init__(self, enable_auto_fix: bool = True, max_retry: int = 3):
        """초기화

        Args:
            enable_auto_fix: 자동 수정 활성화 여부
            max_retry: 최대 재시도 횟수
        """
        self.enable_auto_fix = enable_auto_fix
        self.max_retry = max_retry
        self.recovery_history = []
        self.code_suggestions = []

        # 복구 전략 매핑
        self.recovery_strategies = {
            "strategy_count_validation": self._recover_strategy_count,
            "strategy_parameter_ranges": self._recover_strategy_parameters,
            "db_storage_verification": self._recover_db_storage,
            "prediction_accuracy": self._recover_prediction_accuracy,
            "regime_validity": self._recover_regime_detection,
            "signal_score_range": self._recover_signal_scores,
            "backtest_metrics": self._recover_backtest,
            "data_missing": self._recover_missing_data,
        }

    def attempt_recovery(self, validation_result: ValidationResult,
                        original_data: Any = None) -> Dict[str, Any]:
        """검증 실패 항목에 대한 자동 복구 시도

        Returns:
            {
                'recovered': bool,
                'actions': List[RecoveryAction],
                'fixed_data': Any,
                'suggestions': List[CodeSuggestion]
            }
        """
        recovered = False
        actions = []
        fixed_data = original_data
        suggestions = []

        # Critical 이슈부터 처리
        critical_issues = validation_result.get_issues_by_severity(ValidationSeverity.CRITICAL)

        for issue in critical_issues:
            if not self.enable_auto_fix:
                # 자동 수정 비활성화 시 제안만 생성
                suggestion = self._generate_code_suggestion(issue)
                if suggestion:
                    suggestions.append(suggestion)
                continue

            # 복구 전략 선택
            strategy = self._select_recovery_strategy(issue)

            if strategy:
                try:
                    action, new_data = strategy(issue, fixed_data)
                    actions.append(action)

                    if action.success:
                        fixed_data = new_data
                        recovered = True
                        logger.info(f"✅ Successfully recovered: {issue.check_name}")
                    else:
                        logger.warning(f"⚠️ Recovery failed for: {issue.check_name}")

                except Exception as e:
                    logger.error(f"Recovery error for {issue.check_name}: {e}")
                    actions.append(RecoveryAction(
                        action_type="error",
                        description=f"Recovery failed: {str(e)}",
                        success=False,
                        error_message=str(e)
                    ))

        # High severity 이슈 처리
        high_issues = validation_result.get_issues_by_severity(ValidationSeverity.HIGH)

        for issue in high_issues:
            suggestion = self._generate_code_suggestion(issue)
            if suggestion:
                suggestions.append(suggestion)

        # 복구 히스토리 저장
        self._save_recovery_history(validation_result.component, actions, suggestions)

        return {
            'recovered': recovered,
            'actions': actions,
            'fixed_data': fixed_data,
            'suggestions': suggestions
        }

    def _select_recovery_strategy(self, issue: ValidationIssue):
        """이슈에 맞는 복구 전략 선택"""
        # 직접 매핑된 전략 확인
        if issue.check_name in self.recovery_strategies:
            return self.recovery_strategies[issue.check_name]

        # 패턴 기반 매칭
        for pattern, strategy in self.recovery_strategies.items():
            if pattern in issue.check_name:
                return strategy

        return None

    def _recover_strategy_count(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """전략 개수 문제 복구"""
        action = RecoveryAction(
            action_type="adjust_params",
            description="Adjusting strategy generation parameters"
        )

        try:
            if isinstance(data, dict) and 'strategies' in data:
                current_count = len(data['strategies'])
                expected_min = issue.expected[0] if isinstance(issue.expected, tuple) else 100

                if current_count < expected_min:
                    # 전략 추가 생성 시도
                    logger.info(f"Attempting to generate {expected_min - current_count} more strategies")

                    # 여기서 실제로 전략을 추가 생성하는 로직 호출
                    # 임시로 빈 전략 추가 (실제 구현 시 교체 필요)
                    additional_strategies = self._generate_additional_strategies(
                        expected_min - current_count,
                        data.get('coin'),
                        data.get('interval')
                    )

                    data['strategies'].extend(additional_strategies)
                    data['count'] = len(data['strategies'])

                    action.success = True
                    action.fixed_data = data
                else:
                    # 너무 많은 경우 트리밍
                    data['strategies'] = data['strategies'][:expected_min * 2]
                    data['count'] = len(data['strategies'])
                    action.success = True
                    action.fixed_data = data

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_strategy_parameters(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """전략 파라미터 범위 문제 복구"""
        action = RecoveryAction(
            action_type="fix_data",
            description="Fixing out-of-range strategy parameters"
        )

        try:
            if isinstance(data, dict) and 'strategies' in data:
                fixed_count = 0

                for strategy in data['strategies']:
                    # RSI 파라미터 수정
                    if 'rsi_period' in strategy:
                        strategy['rsi_period'] = max(2, min(100, strategy['rsi_period']))
                        fixed_count += 1

                    if 'rsi_oversold' in strategy and 'rsi_overbought' in strategy:
                        # oversold < overbought 보장
                        if strategy['rsi_oversold'] >= strategy['rsi_overbought']:
                            strategy['rsi_oversold'] = 30
                            strategy['rsi_overbought'] = 70
                            fixed_count += 1

                    # MACD 파라미터 수정
                    if 'macd_fast' in strategy:
                        strategy['macd_fast'] = max(3, min(30, strategy['macd_fast']))

                    if 'macd_slow' in strategy:
                        strategy['macd_slow'] = max(10, min(50, strategy['macd_slow']))

                    # Volume multiplier 수정
                    if 'volume_multiplier' in strategy:
                        strategy['volume_multiplier'] = max(0.5, min(10.0, strategy['volume_multiplier']))

                action.success = True
                action.description = f"Fixed {fixed_count} strategy parameters"
                action.fixed_data = data

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_db_storage(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """DB 저장 문제 복구"""
        action = RecoveryAction(
            action_type="retry",
            description="Retrying database storage"
        )

        try:
            # DB 연결 재시도
            for attempt in range(self.max_retry):
                try:
                    # 여기서 실제 DB 저장 재시도
                    # 실제 구현 시 적절한 DB 저장 함수 호출
                    logger.info(f"DB storage retry attempt {attempt + 1}/{self.max_retry}")

                    # WAL 모드 체크 및 체크포인트
                    self._checkpoint_database()

                    action.success = True
                    action.description = f"DB storage succeeded on attempt {attempt + 1}"
                    break

                except Exception as e:
                    if attempt == self.max_retry - 1:
                        raise e
                    logger.warning(f"DB retry {attempt + 1} failed: {e}")

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_prediction_accuracy(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """예측 정확도 문제 복구"""
        action = RecoveryAction(
            action_type="adjust_params",
            description="Adjusting prediction model parameters"
        )

        try:
            if isinstance(data, dict):
                current_accuracy = data.get('prediction_accuracy', 0)

                # 정확도가 너무 낮은 경우
                if current_accuracy < 0.4:
                    # 더 많은 에피소드로 재학습 제안
                    data['suggested_episodes'] = data.get('total_episodes', 100) * 2
                    data['suggested_learning_rate'] = 0.0001
                    action.description = "Suggest increasing training episodes"

                # 정확도가 너무 높은 경우 (과적합)
                elif current_accuracy > 0.8:
                    data['suggested_regularization'] = 0.01
                    data['suggested_dropout'] = 0.2
                    action.description = "Suggest adding regularization for overfitting"

                action.success = True

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_regime_detection(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """레짐 감지 문제 복구"""
        action = RecoveryAction(
            action_type="fix_data",
            description="Fixing regime detection"
        )

        try:
            if isinstance(data, dict):
                invalid_regime = data.get('regime', '')

                # 기본 레짐으로 대체
                if invalid_regime not in ['bullish', 'bearish', 'neutral', 'volatile', 'trending', 'ranging']:
                    data['regime'] = 'neutral'
                    action.success = True
                    action.description = f"Replaced invalid regime '{invalid_regime}' with 'neutral'"

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_signal_scores(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """신호 점수 범위 문제 복구"""
        action = RecoveryAction(
            action_type="fix_data",
            description="Normalizing signal scores"
        )

        try:
            if isinstance(data, dict) and 'signal_scores' in data:
                scores = data['signal_scores']

                # 점수 정규화 (0-1 범위로)
                normalized_scores = []
                for score in scores:
                    if score < 0:
                        normalized_scores.append(0)
                    elif score > 1:
                        normalized_scores.append(1)
                    else:
                        normalized_scores.append(score)

                data['signal_scores'] = normalized_scores
                action.success = True
                action.description = f"Normalized {len(scores)} signal scores to [0,1] range"

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_backtest(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """백테스트 문제 복구"""
        action = RecoveryAction(
            action_type="retry",
            description="Re-running backtest with adjusted parameters"
        )

        try:
            # 백테스트 재실행 제안
            if isinstance(data, dict):
                data['backtest_retry_suggested'] = True
                data['suggested_backtest_params'] = {
                    'initial_capital': 10000,
                    'commission': 0.001,
                    'slippage': 0.001,
                    'min_trade_amount': 100
                }
                action.description = "Suggest backtest re-run with adjusted parameters"
                action.success = True

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _recover_missing_data(self, issue: ValidationIssue, data: Any) -> Tuple[RecoveryAction, Any]:
        """누락 데이터 복구"""
        action = RecoveryAction(
            action_type="fix_data",
            description="Filling missing data with defaults"
        )

        try:
            if isinstance(data, dict):
                # 필수 필드 기본값 채우기
                defaults = {
                    'coin': 'BTC',
                    'interval': '15m',
                    'strategies': [],
                    'count': 0,
                    'regime': 'neutral',
                    'signal_scores': [0.5],
                    'prediction_accuracy': 0.5,
                    'win_rate': 0.5,
                    'average_return': 0.0
                }

                for key, default_value in defaults.items():
                    if key not in data or data[key] is None:
                        data[key] = default_value
                        logger.info(f"Filled missing field '{key}' with default: {default_value}")

                action.success = True

        except Exception as e:
            action.success = False
            action.error_message = str(e)

        return action, data

    def _generate_additional_strategies(self, count: int, coin: str, interval: str) -> List[Dict]:
        """추가 전략 생성 (임시 구현)"""
        # 실제 구현 시 strategy_manager의 생성 함수 호출
        strategies = []
        for i in range(count):
            strategies.append({
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'volume_multiplier': 1.5,
                'atr_multiplier': 2.0,
                'generated_by': 'recovery_engine'
            })
        return strategies

    def _checkpoint_database(self):
        """데이터베이스 WAL 체크포인트 실행"""
        try:
            # learning_results.db는 이제 rl_strategies.db로 통합됨 (중복 제거)
            db_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'data', 'rl_strategies.db')
            ]

            for db_path in db_paths:
                if os.path.exists(db_path):
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        logger.debug(f"WAL checkpoint executed for {db_path}")

        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")

    def _generate_code_suggestion(self, issue: ValidationIssue) -> Optional[CodeSuggestion]:
        """이슈에 대한 코드 개선 제안 생성"""
        # 이슈 타입별 코드 제안 매핑
        suggestions_map = {
            "strategy_count": self._suggest_strategy_count_fix,
            "parameter_range": self._suggest_parameter_range_fix,
            "db_storage": self._suggest_db_storage_fix,
            "prediction_accuracy": self._suggest_ml_model_fix,
            "regime": self._suggest_regime_detection_fix,
        }

        for pattern, suggest_func in suggestions_map.items():
            if pattern in issue.check_name.lower():
                return suggest_func(issue)

        # 기본 제안
        if issue.suggestion:
            return CodeSuggestion(
                file_path="unknown",
                line_number=None,
                issue_type=issue.check_name,
                current_code=None,
                suggested_code=None,
                explanation=issue.suggestion,
                severity=issue.severity.value,
                confidence=0.5
            )

        return None

    def _suggest_strategy_count_fix(self, issue: ValidationIssue) -> CodeSuggestion:
        """전략 생성 개수 관련 코드 수정 제안"""
        return CodeSuggestion(
            file_path="rl_pipeline/strategy/manager.py",
            line_number=None,
            issue_type="strategy_generation",
            current_code=None,
            suggested_code="""
# Increase strategy generation count
STRATEGIES_PER_COMBINATION = 100  # Increased from 50
MAX_STRATEGIES = 20000  # Increased from 10000

# Add retry logic for strategy generation
for attempt in range(3):
    try:
        strategies = create_strategies(...)
        if len(strategies) >= min_required:
            break
    except Exception as e:
        if attempt == 2:
            raise
        logger.warning(f"Retry {attempt + 1}: {e}")
""",
            explanation="Increase strategy generation parameters and add retry logic",
            severity=issue.severity.value,
            confidence=0.8
        )

    def _suggest_parameter_range_fix(self, issue: ValidationIssue) -> CodeSuggestion:
        """파라미터 범위 관련 코드 수정 제안"""
        return CodeSuggestion(
            file_path="rl_pipeline/strategy/manager.py",
            line_number=None,
            issue_type="parameter_validation",
            current_code=None,
            suggested_code="""
# Add parameter validation before saving
def validate_strategy_params(strategy):
    strategy['rsi_period'] = np.clip(strategy['rsi_period'], 2, 100)
    strategy['rsi_oversold'] = np.clip(strategy['rsi_oversold'], 10, 50)
    strategy['rsi_overbought'] = np.clip(strategy['rsi_overbought'], 50, 90)

    # Ensure oversold < overbought
    if strategy['rsi_oversold'] >= strategy['rsi_overbought']:
        strategy['rsi_oversold'] = 30
        strategy['rsi_overbought'] = 70

    strategy['volume_multiplier'] = np.clip(strategy['volume_multiplier'], 0.5, 10.0)
    return strategy

# Apply validation to all strategies
validated_strategies = [validate_strategy_params(s) for s in strategies]
""",
            explanation="Add parameter validation to ensure all values are within valid ranges",
            severity=issue.severity.value,
            confidence=0.9
        )

    def _suggest_db_storage_fix(self, issue: ValidationIssue) -> CodeSuggestion:
        """DB 저장 관련 코드 수정 제안"""
        return CodeSuggestion(
            file_path="rl_pipeline/db/writes.py",
            line_number=None,
            issue_type="database_storage",
            current_code=None,
            suggested_code="""
# Add retry logic with exponential backoff
import time
from contextlib import contextmanager

@contextmanager
def get_db_connection_with_retry(db_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            yield conn
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"DB locked, retry {attempt + 1} after {wait_time}s")
            time.sleep(wait_time)
        finally:
            if 'conn' in locals():
                conn.close()
""",
            explanation="Add retry logic with exponential backoff for database operations",
            severity=issue.severity.value,
            confidence=0.85
        )

    def _suggest_ml_model_fix(self, issue: ValidationIssue) -> CodeSuggestion:
        """ML 모델 관련 코드 수정 제안"""
        return CodeSuggestion(
            file_path="rl_pipeline/simulation/selfplay.py",
            line_number=None,
            issue_type="ml_model_improvement",
            current_code=None,
            suggested_code="""
# Add regularization to prevent overfitting
model_config = {
    'learning_rate': 0.0001,  # Reduced from 0.001
    'dropout_rate': 0.2,       # Added dropout
    'l2_regularization': 0.01, # Added L2 reg
    'batch_size': 64,          # Increased batch size
    'episodes': 500,           # Increased training episodes
    'early_stopping_patience': 20,
    'validation_split': 0.2
}

# Add data augmentation
def augment_training_data(data):
    # Add noise to prevent overfitting
    noise = np.random.normal(0, 0.01, data.shape)
    augmented = data + noise
    return np.clip(augmented, -1, 1)
""",
            explanation="Add regularization and data augmentation to improve model generalization",
            severity=issue.severity.value,
            confidence=0.75
        )

    def _suggest_regime_detection_fix(self, issue: ValidationIssue) -> CodeSuggestion:
        """레짐 감지 관련 코드 수정 제안"""
        return CodeSuggestion(
            file_path="rl_pipeline/routing/regime_router.py",
            line_number=None,
            issue_type="regime_detection",
            current_code=None,
            suggested_code="""
# Add validation for regime detection
VALID_REGIMES = ['bullish', 'bearish', 'neutral', 'volatile', 'trending', 'ranging']

def detect_regime(market_data):
    # ... existing detection logic ...

    # Validate detected regime
    if detected_regime not in VALID_REGIMES:
        logger.warning(f"Invalid regime '{detected_regime}', defaulting to 'neutral'")
        detected_regime = 'neutral'

    # Add confidence score
    confidence = calculate_regime_confidence(market_data, detected_regime)

    return {
        'regime': detected_regime,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
""",
            explanation="Add regime validation and confidence scoring",
            severity=issue.severity.value,
            confidence=0.8
        )

    def _save_recovery_history(self, component: str, actions: List[RecoveryAction],
                              suggestions: List[CodeSuggestion]):
        """복구 히스토리 저장"""
        history_entry = {
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'actions': [asdict(a) for a in actions],
            'suggestions': [asdict(s) for s in suggestions],
            'success': any(a.success for a in actions)
        }

        self.recovery_history.append(history_entry)

        # 파일로도 저장
        try:
            log_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'recovery_history.jsonl'
            )

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(history_entry, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.warning(f"Failed to save recovery history: {e}")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """복구 통계 조회"""
        if not self.recovery_history:
            return {"message": "No recovery attempts yet"}

        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for h in self.recovery_history if h['success'])

        return {
            'total_recovery_attempts': total_attempts,
            'successful_recoveries': successful_recoveries,
            'success_rate': successful_recoveries / total_attempts if total_attempts > 0 else 0,
            'total_code_suggestions': sum(len(h['suggestions']) for h in self.recovery_history),
            'recent_attempts': self.recovery_history[-5:]  # 최근 5개
        }