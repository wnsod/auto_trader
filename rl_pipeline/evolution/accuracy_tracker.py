"""
📊 Accuracy Tracker - 정확도 추적 시스템

실제 거래 결과와 예측을 비교하여 Phase별 정확도를 측정합니다.
"""

import os
import sys
import logging
import sqlite3
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """예측 기록"""
    prediction_id: str
    coin: str
    interval: str
    phase: int
    
    # 예측 정보
    predicted_direction: str  # 'buy', 'sell', 'hold'
    predicted_mfe: float      # 예측 MFE
    predicted_mae: float      # 예측 MAE
    entry_score: float        # 진입 점수
    confidence: float         # 신뢰도
    
    # 실제 결과 (나중에 업데이트)
    actual_direction: Optional[str] = None
    actual_mfe: Optional[float] = None
    actual_mae: Optional[float] = None
    actual_pnl: Optional[float] = None
    
    # 메타데이터
    timestamp: datetime = field(default_factory=datetime.now)
    evaluated: bool = False
    evaluation_time: Optional[datetime] = None
    
    @property
    def is_direction_correct(self) -> Optional[bool]:
        """방향 예측 정확도"""
        if self.actual_direction is None:
            return None
        return self.predicted_direction == self.actual_direction
    
    @property
    def mfe_error(self) -> Optional[float]:
        """MFE 예측 오차"""
        if self.actual_mfe is None:
            return None
        return abs(self.actual_mfe - self.predicted_mfe)
    
    @property
    def mae_error(self) -> Optional[float]:
        """MAE 예측 오차"""
        if self.actual_mae is None:
            return None
        return abs(self.actual_mae - self.predicted_mae)


class AccuracyTracker:
    """
    📊 정확도 추적 시스템
    
    - 예측 기록
    - 실제 결과와 비교
    - Phase별 정확도 계산
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: 추적 데이터 저장 DB 경로
        """
        if db_path:
            self.db_path = db_path
        else:
            try:
                from rl_pipeline.core.env import config
                self.db_path = config.LEARNING_RESULTS_DB_PATH
            except:
                self.db_path = None
        
        self._ensure_table()
    
    def _ensure_table(self) -> None:
        """테이블 생성"""
        if not self.db_path:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    phase INTEGER DEFAULT 1,
                    
                    -- 예측 정보
                    predicted_direction TEXT,
                    predicted_mfe REAL,
                    predicted_mae REAL,
                    entry_score REAL,
                    confidence REAL,
                    
                    -- 실제 결과
                    actual_direction TEXT,
                    actual_mfe REAL,
                    actual_mae REAL,
                    actual_pnl REAL,
                    
                    -- 메타데이터
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    evaluated INTEGER DEFAULT 0,
                    evaluation_time TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_coin_interval 
                ON prediction_tracking(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_evaluated 
                ON prediction_tracking(evaluated)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_timestamp 
                ON prediction_tracking(timestamp)
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"⚠️ 추적 테이블 생성 실패: {e}")
    
    def record_prediction(
        self,
        prediction_id: str,
        coin: str,
        interval: str,
        phase: int,
        predicted_direction: str,
        predicted_mfe: float,
        predicted_mae: float,
        entry_score: float,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        예측 기록
        
        Args:
            prediction_id: 고유 예측 ID
            coin: 코인명
            interval: 인터벌
            phase: 현재 Phase
            predicted_direction: 예측 방향
            predicted_mfe: 예측 MFE
            predicted_mae: 예측 MAE
            entry_score: 진입 점수
            confidence: 신뢰도
            metadata: 추가 메타데이터
        """
        if not self.db_path:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO prediction_tracking
                (prediction_id, coin, interval, phase, predicted_direction,
                 predicted_mfe, predicted_mae, entry_score, confidence,
                 timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                coin,
                interval,
                phase,
                predicted_direction,
                predicted_mfe,
                predicted_mae,
                entry_score,
                confidence,
                datetime.now().isoformat(),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ 예측 기록 실패: {e}")
    
    def update_actual_result(
        self,
        prediction_id: str,
        actual_direction: str,
        actual_mfe: float,
        actual_mae: float,
        actual_pnl: float
    ) -> bool:
        """
        실제 결과 업데이트
        
        Args:
            prediction_id: 예측 ID
            actual_direction: 실제 방향
            actual_mfe: 실제 MFE
            actual_mae: 실제 MAE
            actual_pnl: 실제 PnL
            
        Returns:
            성공 여부
        """
        if not self.db_path:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE prediction_tracking
                SET actual_direction = ?,
                    actual_mfe = ?,
                    actual_mae = ?,
                    actual_pnl = ?,
                    evaluated = 1,
                    evaluation_time = ?
                WHERE prediction_id = ?
            """, (
                actual_direction,
                actual_mfe,
                actual_mae,
                actual_pnl,
                datetime.now().isoformat(),
                prediction_id
            ))
            
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            return updated
            
        except Exception as e:
            logger.warning(f"⚠️ 실제 결과 업데이트 실패: {e}")
            return False
    
    def get_accuracy_stats(
        self,
        coin: str,
        interval: str,
        phase: Optional[int] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        정확도 통계 조회
        
        Args:
            coin: 코인명
            interval: 인터벌
            phase: Phase 필터 (None이면 전체)
            days: 조회 기간 (일)
            
        Returns:
            정확도 통계 딕셔너리
        """
        if not self.db_path:
            return {}
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = (datetime.now() - timedelta(days=days)).isoformat()
            
            # 기본 쿼리
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_direction = actual_direction THEN 1 ELSE 0 END) as correct,
                    AVG(ABS(predicted_mfe - actual_mfe)) as avg_mfe_error,
                    AVG(ABS(predicted_mae - actual_mae)) as avg_mae_error,
                    AVG(actual_pnl) as avg_pnl,
                    AVG(confidence) as avg_confidence
                FROM prediction_tracking
                WHERE coin = ? AND interval = ? AND evaluated = 1 AND timestamp >= ?
            """
            
            params = [coin, interval, since]
            
            if phase is not None:
                query += " AND phase = ?"
                params.append(phase)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                total, correct, mfe_err, mae_err, avg_pnl, avg_conf = row
                
                stats = {
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "direction_accuracy": correct / total if total > 0 else 0.0,
                    "avg_mfe_error": mfe_err or 0.0,
                    "avg_mae_error": mae_err or 0.0,
                    "avg_pnl": avg_pnl or 0.0,
                    "avg_confidence": avg_conf or 0.0
                }
            else:
                stats = {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "direction_accuracy": 0.0,
                    "avg_mfe_error": 0.0,
                    "avg_mae_error": 0.0,
                    "avg_pnl": 0.0,
                    "avg_confidence": 0.0
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️ 정확도 통계 조회 실패: {e}")
            return {}
    
    def get_phase_comparison(
        self,
        coin: str,
        interval: str,
        days: int = 30
    ) -> Dict[int, Dict[str, Any]]:
        """
        Phase별 정확도 비교
        
        Args:
            coin: 코인명
            interval: 인터벌
            days: 조회 기간
            
        Returns:
            Phase별 통계 딕셔너리
        """
        if not self.db_path:
            return {}
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT 
                    phase,
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_direction = actual_direction THEN 1 ELSE 0 END) as correct,
                    AVG(actual_pnl) as avg_pnl,
                    AVG(confidence) as avg_confidence
                FROM prediction_tracking
                WHERE coin = ? AND interval = ? AND evaluated = 1 AND timestamp >= ?
                GROUP BY phase
                ORDER BY phase
            """, (coin, interval, since))
            
            comparison = {}
            for row in cursor.fetchall():
                phase, total, correct, avg_pnl, avg_conf = row
                comparison[phase] = {
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "direction_accuracy": correct / total if total > 0 else 0.0,
                    "avg_pnl": avg_pnl or 0.0,
                    "avg_confidence": avg_conf or 0.0
                }
            
            conn.close()
            return comparison
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 비교 조회 실패: {e}")
            return {}
    
    def get_pending_evaluations(
        self,
        older_than_minutes: int = 60
    ) -> List[Dict]:
        """
        평가 대기 중인 예측 조회
        
        Args:
            older_than_minutes: N분 이상 지난 예측만
            
        Returns:
            평가 대기 예측 리스트
        """
        if not self.db_path:
            return []
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = (datetime.now() - timedelta(minutes=older_than_minutes)).isoformat()
            
            cursor.execute("""
                SELECT prediction_id, coin, interval, phase,
                       predicted_direction, predicted_mfe, predicted_mae,
                       entry_score, timestamp
                FROM prediction_tracking
                WHERE evaluated = 0 AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT 100
            """, (cutoff,))
            
            pending = []
            for row in cursor.fetchall():
                pending.append({
                    "prediction_id": row[0],
                    "coin": row[1],
                    "interval": row[2],
                    "phase": row[3],
                    "predicted_direction": row[4],
                    "predicted_mfe": row[5],
                    "predicted_mae": row[6],
                    "entry_score": row[7],
                    "timestamp": row[8]
                })
            
            conn.close()
            return pending
            
        except Exception as e:
            logger.warning(f"⚠️ 대기 평가 조회 실패: {e}")
            return []
    
    def cleanup_old_records(self, days: int = 90) -> int:
        """
        오래된 기록 정리
        
        Args:
            days: N일 이상 지난 기록 삭제
            
        Returns:
            삭제된 레코드 수
        """
        if not self.db_path:
            return 0
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                DELETE FROM prediction_tracking
                WHERE timestamp < ? AND evaluated = 1
            """, (cutoff,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted > 0:
                logger.info(f"🧹 {deleted}개 오래된 예측 기록 삭제")
            
            return deleted
            
        except Exception as e:
            logger.warning(f"⚠️ 기록 정리 실패: {e}")
            return 0


# 싱글톤 인스턴스
_accuracy_tracker: Optional[AccuracyTracker] = None


def get_accuracy_tracker() -> AccuracyTracker:
    """AccuracyTracker 싱글톤 인스턴스 반환"""
    global _accuracy_tracker
    if _accuracy_tracker is None:
        _accuracy_tracker = AccuracyTracker()
    return _accuracy_tracker

