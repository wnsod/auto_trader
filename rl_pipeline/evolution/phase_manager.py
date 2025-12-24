"""
🧬 Phase Manager - 종목별 진화 단계 관리

종목(코인+인터벌)별로 현재 Phase를 관리하고 승격/강등을 결정합니다.
- Phase 1: 통계 기반 (MFE/MAE EntryScore)
- Phase 2: MFE/MAE 예측 모델 (XGBoost/LightGBM)
- Phase 3: 타이밍 최적화 (RL Agent)
"""

import os
import sys
import logging
import sqlite3
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
import json

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)


class Phase(IntEnum):
    """진화 단계 정의"""
    STATISTICAL = 1      # 통계 기반 (MFE/MAE)
    PREDICTIVE = 2       # 예측 모델 (XGBoost/LightGBM)
    TIMING_OPTIMIZED = 3 # 타이밍 최적화 (RL Agent)


@dataclass
class PhaseThresholds:
    """Phase 승격/강등 임계값"""
    # Phase 1 → Phase 2 승격 조건
    promote_1_to_2_accuracy: float = 0.60   # 60% 이상 정확도
    promote_1_to_2_samples: int = 100       # 최소 100개 샘플
    
    # Phase 2 → Phase 3 승격 조건
    promote_2_to_3_accuracy: float = 0.70   # 70% 이상 정확도
    promote_2_to_3_samples: int = 200       # 최소 200개 샘플
    
    # 강등 조건 (연속 N회 기준 미달)
    demote_accuracy_drop: float = 0.10      # 10% 이상 하락 시
    demote_consecutive_fails: int = 3       # 연속 3회 실패
    
    # 최소 유지 기간 (성급한 강등 방지)
    min_phase_duration_hours: int = 24      # 최소 24시간 유지


@dataclass
class PhaseState:
    """종목별 Phase 상태"""
    coin: str
    interval: str
    current_phase: Phase = Phase.STATISTICAL
    accuracy_history: List[float] = field(default_factory=list)
    last_promotion: Optional[datetime] = None
    last_demotion: Optional[datetime] = None
    consecutive_fails: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0
    phase_start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def current_accuracy(self) -> float:
        """현재 정확도 계산"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    @property
    def recent_accuracy(self) -> float:
        """최근 정확도 (마지막 20개 기준)"""
        if not self.accuracy_history:
            return 0.0
        recent = self.accuracy_history[-20:]
        return sum(recent) / len(recent)


class PhaseManager:
    """
    🧬 종목별 Phase 관리자
    
    종목마다 독립적으로 Phase를 관리하므로:
    - BTC/15m: Phase 3 (데이터 충분, 정확도 높음)
    - NEW_COIN/1h: Phase 1 (데이터 부족, 통계만 사용)
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        thresholds: Optional[PhaseThresholds] = None
    ):
        """
        Args:
            db_path: Phase 상태 저장 DB 경로
            thresholds: 승격/강등 임계값
        """
        self.thresholds = thresholds or PhaseThresholds()
        self.states: Dict[str, PhaseState] = {}  # key: "coin_interval"
        
        # DB 경로 설정
        if db_path:
            self.db_path = db_path
        else:
            try:
                from rl_pipeline.core.env import config
                self.db_path = config.LEARNING_RESULTS_DB_PATH
            except:
                self.db_path = None
        
        # DB에서 기존 상태 로드
        self._load_states_from_db()
    
    def _get_key(self, coin: str, interval: str) -> str:
        """종목 키 생성"""
        return f"{coin}_{interval}"
    
    def _ensure_table(self) -> None:
        """Phase 상태 테이블 생성"""
        if not self.db_path:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_phases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    current_phase INTEGER DEFAULT 1,
                    accuracy_history TEXT DEFAULT '[]',
                    last_promotion TEXT,
                    last_demotion TEXT,
                    consecutive_fails INTEGER DEFAULT 0,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    phase_start_time TEXT,
                    metadata TEXT DEFAULT '{}',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, interval)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evolution_coin_interval 
                ON evolution_phases(coin, interval)
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"⚠️ Phase 테이블 생성 실패: {e}")
    
    def _load_states_from_db(self) -> None:
        """DB에서 Phase 상태 로드"""
        if not self.db_path:
            return
            
        self._ensure_table()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT coin, interval, current_phase, accuracy_history,
                       last_promotion, last_demotion, consecutive_fails,
                       total_predictions, correct_predictions, phase_start_time, metadata
                FROM evolution_phases
            """)
            
            for row in cursor.fetchall():
                coin, interval, phase, acc_hist, last_prom, last_dem, \
                    cons_fails, total_pred, correct_pred, phase_start, meta = row
                
                key = self._get_key(coin, interval)
                
                # JSON 파싱
                try:
                    accuracy_history = json.loads(acc_hist) if acc_hist else []
                except:
                    accuracy_history = []
                
                try:
                    metadata = json.loads(meta) if meta else {}
                except:
                    metadata = {}
                
                # datetime 파싱
                last_promotion = datetime.fromisoformat(last_prom) if last_prom else None
                last_demotion = datetime.fromisoformat(last_dem) if last_dem else None
                phase_start_time = datetime.fromisoformat(phase_start) if phase_start else datetime.now()
                
                self.states[key] = PhaseState(
                    coin=coin,
                    interval=interval,
                    current_phase=Phase(phase),
                    accuracy_history=accuracy_history,
                    last_promotion=last_promotion,
                    last_demotion=last_demotion,
                    consecutive_fails=cons_fails,
                    total_predictions=total_pred,
                    correct_predictions=correct_pred,
                    phase_start_time=phase_start_time,
                    metadata=metadata
                )
            
            conn.close()
            logger.info(f"✅ {len(self.states)}개 종목 Phase 상태 로드 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 상태 로드 실패: {e}")
    
    def _save_state_to_db(self, state: PhaseState) -> None:
        """Phase 상태를 DB에 저장"""
        if not self.db_path:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO evolution_phases
                (coin, interval, current_phase, accuracy_history, last_promotion,
                 last_demotion, consecutive_fails, total_predictions, correct_predictions,
                 phase_start_time, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.coin,
                state.interval,
                int(state.current_phase),
                json.dumps(state.accuracy_history[-100:]),  # 최근 100개만 저장
                state.last_promotion.isoformat() if state.last_promotion else None,
                state.last_demotion.isoformat() if state.last_demotion else None,
                state.consecutive_fails,
                state.total_predictions,
                state.correct_predictions,
                state.phase_start_time.isoformat(),
                json.dumps(state.metadata),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 상태 저장 실패: {e}")
    
    def get_phase(self, coin: str, interval: str) -> Phase:
        """종목의 현재 Phase 조회"""
        key = self._get_key(coin, interval)
        
        if key not in self.states:
            # 새 종목은 Phase 1부터 시작
            self.states[key] = PhaseState(coin=coin, interval=interval)
            self._save_state_to_db(self.states[key])
        
        return self.states[key].current_phase
    
    def get_state(self, coin: str, interval: str) -> PhaseState:
        """종목의 전체 상태 조회"""
        key = self._get_key(coin, interval)
        
        if key not in self.states:
            self.states[key] = PhaseState(coin=coin, interval=interval)
            self._save_state_to_db(self.states[key])
        
        return self.states[key]
    
    def record_prediction(
        self,
        coin: str,
        interval: str,
        predicted_direction: str,  # 'buy', 'sell', 'hold'
        actual_direction: str,      # 'buy', 'sell', 'hold'
        confidence: float = 1.0
    ) -> None:
        """
        예측 결과 기록
        
        Args:
            coin: 코인명
            interval: 인터벌
            predicted_direction: 예측 방향
            actual_direction: 실제 방향
            confidence: 예측 신뢰도 (가중치)
        """
        state = self.get_state(coin, interval)
        
        # 예측 카운트 업데이트
        state.total_predictions += 1
        
        is_correct = predicted_direction == actual_direction
        if is_correct:
            state.correct_predictions += 1
            state.accuracy_history.append(1.0 * confidence)
            state.consecutive_fails = 0
        else:
            state.accuracy_history.append(0.0)
            state.consecutive_fails += 1
        
        # 최근 100개만 유지
        if len(state.accuracy_history) > 100:
            state.accuracy_history = state.accuracy_history[-100:]
        
        # Phase 업데이트 체크
        self._check_phase_transition(state)
        
        # DB 저장
        self._save_state_to_db(state)
    
    def _check_phase_transition(self, state: PhaseState) -> None:
        """Phase 승격/강등 체크"""
        current_phase = state.current_phase
        accuracy = state.recent_accuracy
        samples = state.total_predictions
        
        # 최소 유지 기간 체크
        hours_in_phase = (datetime.now() - state.phase_start_time).total_seconds() / 3600
        if hours_in_phase < self.thresholds.min_phase_duration_hours:
            return  # 아직 평가하기 이름
        
        # 🔼 승격 체크
        if current_phase == Phase.STATISTICAL:
            if (accuracy >= self.thresholds.promote_1_to_2_accuracy and 
                samples >= self.thresholds.promote_1_to_2_samples):
                self._promote(state, Phase.PREDICTIVE)
                
        elif current_phase == Phase.PREDICTIVE:
            if (accuracy >= self.thresholds.promote_2_to_3_accuracy and 
                samples >= self.thresholds.promote_2_to_3_samples):
                self._promote(state, Phase.TIMING_OPTIMIZED)
        
        # 🔽 강등 체크
        if state.consecutive_fails >= self.thresholds.demote_consecutive_fails:
            if current_phase == Phase.TIMING_OPTIMIZED:
                self._demote(state, Phase.PREDICTIVE)
            elif current_phase == Phase.PREDICTIVE:
                self._demote(state, Phase.STATISTICAL)
    
    def _promote(self, state: PhaseState, new_phase: Phase) -> None:
        """Phase 승격"""
        old_phase = state.current_phase
        state.current_phase = new_phase
        state.last_promotion = datetime.now()
        state.phase_start_time = datetime.now()
        state.consecutive_fails = 0
        
        logger.info(
            f"🔼 {state.coin}/{state.interval} Phase 승격: "
            f"{old_phase.name} → {new_phase.name} "
            f"(정확도: {state.recent_accuracy:.1%})"
        )
    
    def _demote(self, state: PhaseState, new_phase: Phase) -> None:
        """Phase 강등"""
        old_phase = state.current_phase
        state.current_phase = new_phase
        state.last_demotion = datetime.now()
        state.phase_start_time = datetime.now()
        state.consecutive_fails = 0
        
        logger.warning(
            f"🔽 {state.coin}/{state.interval} Phase 강등: "
            f"{old_phase.name} → {new_phase.name} "
            f"(연속 실패: {self.thresholds.demote_consecutive_fails}회)"
        )
    
    def force_phase(self, coin: str, interval: str, phase: Phase) -> None:
        """Phase 강제 설정 (테스트/디버그용)"""
        state = self.get_state(coin, interval)
        old_phase = state.current_phase
        state.current_phase = phase
        state.phase_start_time = datetime.now()
        state.consecutive_fails = 0
        self._save_state_to_db(state)
        
        logger.info(f"⚙️ {coin}/{interval} Phase 강제 설정: {old_phase.name} → {phase.name}")
    
    def get_all_states(self) -> Dict[str, PhaseState]:
        """모든 종목의 Phase 상태 조회"""
        return self.states.copy()
    
    def get_phase_distribution(self) -> Dict[Phase, int]:
        """Phase별 종목 수 통계"""
        distribution = {phase: 0 for phase in Phase}
        for state in self.states.values():
            distribution[state.current_phase] += 1
        return distribution
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 현황 요약"""
        distribution = self.get_phase_distribution()
        
        # Phase별 평균 정확도
        phase_accuracies = {phase: [] for phase in Phase}
        for state in self.states.values():
            if state.total_predictions > 0:
                phase_accuracies[state.current_phase].append(state.current_accuracy)
        
        avg_accuracies = {}
        for phase, accs in phase_accuracies.items():
            if accs:
                avg_accuracies[phase.name] = sum(accs) / len(accs)
            else:
                avg_accuracies[phase.name] = 0.0
        
        return {
            "total_symbols": len(self.states),
            "distribution": {p.name: c for p, c in distribution.items()},
            "avg_accuracies": avg_accuracies,
            "top_performers": self._get_top_performers(5)
        }
    
    def _get_top_performers(self, n: int = 5) -> List[Dict]:
        """상위 성과 종목"""
        sorted_states = sorted(
            self.states.values(),
            key=lambda s: (s.current_phase, s.current_accuracy),
            reverse=True
        )
        
        return [
            {
                "symbol": f"{s.coin}/{s.interval}",
                "phase": s.current_phase.name,
                "accuracy": round(s.current_accuracy, 3),
                "predictions": s.total_predictions
            }
            for s in sorted_states[:n]
        ]


# 싱글톤 인스턴스
_phase_manager: Optional[PhaseManager] = None


def get_phase_manager() -> PhaseManager:
    """PhaseManager 싱글톤 인스턴스 반환"""
    global _phase_manager
    if _phase_manager is None:
        _phase_manager = PhaseManager()
    return _phase_manager


def reset_phase_manager() -> None:
    """PhaseManager 인스턴스 리셋 (테스트용)"""
    global _phase_manager
    _phase_manager = None

