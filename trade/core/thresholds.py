"""
🎯 시그널 임계값 중앙 관리 모듈

모든 매매 로직에서 사용하는 시그널 점수 임계값을 중앙에서 관리합니다.
시그널 계산 방식이 변경되어도 이 파일만 수정하면 전체 시스템에 반영됩니다.

주요 기능:
1. 시그널 점수 정규화 (-1.0 ~ 1.0)
2. 동적 임계값 계산 (백분위 기반)
3. 매매 결정 임계값 관리
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import time
import sqlite3


# ============================================================================
# 기본 임계값 설정 (시그널 점수 범위: -1.0 ~ 1.0 기준)
# ============================================================================
@dataclass
class SignalThresholds:
    """시그널 점수 기반 매매 결정 임계값"""
    
    # 🔥 매수 임계값 (높을수록 보수적)
    strong_buy: float = 0.5      # 강한 매수 시그널
    buy: float = 0.3             # 일반 매수 시그널
    weak_buy: float = 0.1        # 약한 매수 시그널 (탐색적 매수)
    buy_candidate: float = 0.05  # 매수 후보 (모니터링)
    
    # 🔥 매도 임계값 (낮을수록 공격적 매도)
    strong_sell: float = -0.5    # 강한 매도 시그널
    sell: float = -0.3           # 일반 매도 시그널
    weak_sell: float = -0.1      # 약한 매도 시그널
    
    # 🔥 홀딩 구간
    hold_min: float = -0.1       # 홀딩 최소 점수
    hold_max: float = 0.1        # 홀딩 최대 점수
    
    # 🔥 우선순위 결정 임계값
    priority_high: float = 0.4   # 높은 우선순위
    priority_medium: float = 0.2 # 중간 우선순위
    priority_low: float = -0.2   # 낮은 우선순위
    
    # 🔥 손절 조정 임계값
    stop_loss_lenient: float = 0.8   # 손절 관대 (강한 시그널)
    stop_loss_moderate: float = 0.6  # 손절 보통
    stop_loss_strict: float = 0.3    # 손절 엄격 (약한 시그널)
    
    # 🔥 톰슨 샘플링 임계값
    thompson_min: float = 0.4    # 톰슨 점수 최소
    
    # 🔥 신규 패턴 탐색 임계값
    new_pattern_min: float = 0.10  # 신규 패턴 탐색 최소 점수


# 글로벌 임계값 인스턴스
DEFAULT_THRESHOLDS = SignalThresholds()


# ============================================================================
# 시그널 점수 정규화 함수
# ============================================================================
def normalize_signal_score(raw_score: float, 
                           min_val: float = -1.0, 
                           max_val: float = 1.0,
                           target_min: float = -1.0,
                           target_max: float = 1.0) -> float:
    """
    시그널 점수를 지정된 범위로 정규화
    
    Args:
        raw_score: 원본 점수
        min_val: 원본 점수의 최소값 (예상)
        max_val: 원본 점수의 최대값 (예상)
        target_min: 목표 최소값 (기본 -1.0)
        target_max: 목표 최대값 (기본 1.0)
    
    Returns:
        정규화된 점수 (target_min ~ target_max 범위)
    """
    if max_val == min_val:
        return 0.0
    
    # 원본 범위 → 0~1 → 목표 범위
    normalized = (raw_score - min_val) / (max_val - min_val)
    result = normalized * (target_max - target_min) + target_min
    
    # 범위 제한
    return max(target_min, min(target_max, result))


def clip_signal_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """시그널 점수를 지정된 범위 내로 제한"""
    return max(min_val, min(max_val, score))


# ============================================================================
# 동적 임계값 계산 (백분위 기반)
# ============================================================================
class DynamicThresholdCalculator:
    """최근 시그널 점수 분포 기반 동적 임계값 계산기"""
    
    def __init__(self, 
                 window_size: int = 1000, 
                 decay_hours: float = 24.0,
                 db_path: Optional[str] = None):
        self.window_size = window_size
        self.decay_hours = decay_hours
        self.db_path = db_path
        self.score_history: List[float] = []
        self.last_update = 0
        self._cached_thresholds: Optional[SignalThresholds] = None
        self._cache_time = 0
        self._cache_ttl = 300  # 5분 캐시
    
    def add_score(self, score: float):
        """새로운 시그널 점수 추가"""
        self.score_history.append(score)
        if len(self.score_history) > self.window_size:
            self.score_history = self.score_history[-self.window_size:]
        self._cached_thresholds = None  # 캐시 무효화
    
    def load_from_db(self, hours_back: int = 24) -> List[float]:
        """DB에서 최근 시그널 점수 로드"""
        if not self.db_path:
            return []
        
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cutoff = int(time.time()) - (hours_back * 3600)
                cursor = conn.execute("""
                    SELECT signal_score FROM signals 
                    WHERE timestamp > ? AND signal_score IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (cutoff, self.window_size))
                scores = [row[0] for row in cursor.fetchall() if row[0] is not None]
                self.score_history = scores
                self.last_update = int(time.time())
                return scores
        except Exception:
            return []
    
    def get_percentile_thresholds(self) -> SignalThresholds:
        """백분위 기반 동적 임계값 계산"""
        now = time.time()
        
        # 캐시 확인
        if self._cached_thresholds and (now - self._cache_time) < self._cache_ttl:
            return self._cached_thresholds
        
        # 데이터 부족 시 기본값 반환
        if len(self.score_history) < 50:
            return DEFAULT_THRESHOLDS
        
        scores = np.array(self.score_history)
        
        thresholds = SignalThresholds(
            # 매수 임계값 (상위 백분위)
            strong_buy=float(np.percentile(scores, 90)),    # 상위 10%
            buy=float(np.percentile(scores, 75)),           # 상위 25%
            weak_buy=float(np.percentile(scores, 60)),      # 상위 40%
            buy_candidate=float(np.percentile(scores, 55)), # 상위 45%
            
            # 매도 임계값 (하위 백분위)
            strong_sell=float(np.percentile(scores, 10)),   # 하위 10%
            sell=float(np.percentile(scores, 25)),          # 하위 25%
            weak_sell=float(np.percentile(scores, 40)),     # 하위 40%
            
            # 홀딩 구간 (중앙값 기준)
            hold_min=float(np.percentile(scores, 40)),
            hold_max=float(np.percentile(scores, 60)),
            
            # 우선순위 (사분위)
            priority_high=float(np.percentile(scores, 80)),
            priority_medium=float(np.percentile(scores, 60)),
            priority_low=float(np.percentile(scores, 40)),
            
            # 손절 관련은 고정값 유지 (안전성)
            stop_loss_lenient=0.8,
            stop_loss_moderate=0.6,
            stop_loss_strict=0.3,
            
            # 기타
            thompson_min=float(np.percentile(scores, 50)),
            new_pattern_min=float(np.percentile(scores, 55))
        )
        
        self._cached_thresholds = thresholds
        self._cache_time = now
        return thresholds


# ============================================================================
# 편의 함수 (글로벌 임계값 접근)
# ============================================================================
def get_thresholds() -> SignalThresholds:
    """기본 임계값 반환"""
    return DEFAULT_THRESHOLDS


def get_buy_threshold(level: str = 'normal') -> float:
    """매수 임계값 반환
    
    Args:
        level: 'strong', 'normal', 'weak', 'candidate'
    """
    t = DEFAULT_THRESHOLDS
    if level == 'strong':
        return t.strong_buy
    elif level == 'weak':
        return t.weak_buy
    elif level == 'candidate':
        return t.buy_candidate
    return t.buy


def get_sell_threshold(level: str = 'normal') -> float:
    """매도 임계값 반환
    
    Args:
        level: 'strong', 'normal', 'weak'
    """
    t = DEFAULT_THRESHOLDS
    if level == 'strong':
        return t.strong_sell
    elif level == 'weak':
        return t.weak_sell
    return t.sell


def is_buy_signal(score: float, level: str = 'normal') -> bool:
    """매수 시그널 여부 판단"""
    return score >= get_buy_threshold(level)


def is_sell_signal(score: float, level: str = 'normal') -> bool:
    """매도 시그널 여부 판단"""
    return score <= get_sell_threshold(level)


def is_hold_signal(score: float) -> bool:
    """홀딩 시그널 여부 판단"""
    t = DEFAULT_THRESHOLDS
    return t.hold_min <= score <= t.hold_max


def get_signal_action(score: float) -> str:
    """시그널 점수 → 액션 변환
    
    Returns:
        'strong_buy', 'buy', 'weak_buy', 'hold', 'weak_sell', 'sell', 'strong_sell'
    """
    t = DEFAULT_THRESHOLDS
    
    if score >= t.strong_buy:
        return 'strong_buy'
    elif score >= t.buy:
        return 'buy'
    elif score >= t.weak_buy:
        return 'weak_buy'
    elif score <= t.strong_sell:
        return 'strong_sell'
    elif score <= t.sell:
        return 'sell'
    elif score <= t.weak_sell:
        return 'weak_sell'
    else:
        return 'hold'


def get_priority_level(score: float) -> str:
    """시그널 점수 → 우선순위 레벨
    
    Returns:
        'high', 'medium', 'low', 'none'
    """
    t = DEFAULT_THRESHOLDS
    
    if score > t.priority_high:
        return 'high'
    elif score > t.priority_medium:
        return 'medium'
    elif score > t.priority_low:
        return 'low'
    else:
        return 'none'


def get_stop_loss_adjustment(signal_score: float) -> float:
    """시그널 점수 기반 손절 조정값 반환
    
    Returns:
        양수: 손절 완화 (%), 음수: 손절 강화 (%)
    """
    t = DEFAULT_THRESHOLDS
    
    if signal_score >= t.stop_loss_lenient:
        return 3.0   # 매우 높은 시그널: 손절을 3% 더 관대하게
    elif signal_score >= t.stop_loss_moderate:
        return 1.5   # 높은 시그널: 손절을 1.5% 더 관대하게
    elif signal_score <= t.stop_loss_strict:
        return -1.5  # 낮은 시그널: 손절을 1.5% 더 엄격하게
    else:
        return 0.0   # 중립


# ============================================================================
# 시그널 점수 통계 유틸리티
# ============================================================================
def calculate_score_stats(scores: List[float]) -> Dict[str, float]:
    """시그널 점수 통계 계산"""
    if not scores:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    
    arr = np.array(scores)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'p10': float(np.percentile(arr, 10)),
    }


def print_threshold_info():
    """현재 임계값 정보 출력 (디버깅용)"""
    t = DEFAULT_THRESHOLDS
    print("=" * 50)
    print("📊 시그널 임계값 설정")
    print("=" * 50)
    print(f"  🔼 강한 매수: {t.strong_buy}")
    print(f"  📈 일반 매수: {t.buy}")
    print(f"  📊 약한 매수: {t.weak_buy}")
    print(f"  📍 매수 후보: {t.buy_candidate}")
    print("-" * 50)
    print(f"  📉 약한 매도: {t.weak_sell}")
    print(f"  📉 일반 매도: {t.sell}")
    print(f"  🔽 강한 매도: {t.strong_sell}")
    print("-" * 50)
    print(f"  ⏸️ 홀딩 구간: {t.hold_min} ~ {t.hold_max}")
    print("=" * 50)
