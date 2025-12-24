"""
Orchestrator 데이터 모델
파이프라인 결과 및 데이터 구조 정의
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    coin: str
    interval: str
    strategies_created: int = 0
    selfplay_episodes: int = 0
    regime_detected: str = "neutral"
    routing_results: int = 0
    signal_score: float = 0.0
    signal_action: str = "HOLD"
    execution_time: float = 0.0
    status: str = "pending"
    created_at: str = ""
    selfplay_result: Optional[Dict[str, Any]] = None  # self-play 결과 저장
    strategies: Optional[List[Dict]] = None  # 코인별 전략
    coin_analysis: Optional[Dict] = None  # 코인별 분석 결과
    regime_routing: Optional[Dict] = None  # 레짐 라우팅 결과
    error_message: Optional[str] = None  # 에러 메시지

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def update_status(self, status: str, message: Optional[str] = None):
        """상태 업데이트"""
        self.status = status
        if message:
            self.error_message = message
        if not self.created_at:
            self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@dataclass
class SelfPlayConfig:
    """Self-play 설정"""
    episodes: int = 200
    agents_per_episode: int = 4
    learning_rate: float = 0.1
    early_stop: bool = True
    early_stop_patience: int = 15
    early_stop_accuracy: float = 0.85
    min_improvement: float = 0.01
    min_episodes: int = 20
    predictive_ratio: float = 0.2

    @classmethod
    def from_env(cls):
        """환경변수에서 설정 로드"""
        import os
        return cls(
            episodes=int(os.getenv('AZ_SELFPLAY_EPISODES', '200')),
            agents_per_episode=int(os.getenv('AZ_SELFPLAY_AGENTS_PER_EPISODE', '4')),
            learning_rate=float(os.getenv('PREDICTIVE_SELFPLAY_LEARNING_RATE', '0.1')),
            early_stop=os.getenv('PREDICTIVE_SELFPLAY_EARLY_STOP', 'true').lower() == 'true',
            early_stop_patience=int(os.getenv('PREDICTIVE_SELFPLAY_EARLY_STOP_PATIENCE', '15')),
            early_stop_accuracy=float(os.getenv('PREDICTIVE_SELFPLAY_EARLY_STOP_ACCURACY', '0.85')),
            min_improvement=float(os.getenv('PREDICTIVE_SELFPLAY_MIN_IMPROVEMENT', '0.01')),
            min_episodes=int(os.getenv('PREDICTIVE_SELFPLAY_MIN_EPISODES', '20')),
            predictive_ratio=float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
        )


@dataclass
class StrategyPoolConfig:
    """전략 풀 설정"""
    pool_size: int = 15000
    min_trades: int = 1
    max_dd: float = 1.0
    min_win_rate: float = 0.3
    min_profit_factor: float = 0.8

    @classmethod
    def from_env(cls):
        """환경변수에서 설정 로드"""
        import os
        return cls(
            pool_size=int(os.getenv('AZ_STRATEGY_POOL_SIZE', '15000'))
        )


@dataclass
class ValidationResult:
    """검증 결과"""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    quality_score: int = 0

    def add_issue(self, issue: str):
        """이슈 추가"""
        self.issues.append(issue)
        self.valid = False

    def add_warning(self, warning: str):
        """경고 추가"""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)