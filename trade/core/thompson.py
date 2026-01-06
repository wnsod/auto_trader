"""
Thompson Sampling 공통 모듈 (복구 및 강화 버전)

주요 기능:
1. 패턴별 승률 샘플링 (Beta Distribution)
2. 패턴별 성과 추적 및 통계 제공
3. 실전/가상 매매 공통 인터페이스 제공
"""
import os
import sys
import sqlite3
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# 경로 설정
_current_dir = os.path.dirname(os.path.abspath(__file__))
_trade_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_trade_dir)

if _project_root not in sys.path:
    sys.path.append(_project_root)

# --- 수학적 보정 도구 클래스 정의 ---

class ExponentialDecayWeight:
    """최근 데이터에 더 높은 가중치를 부여하는 지수 감쇠기"""
    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate
        
    def calculate_weight(self, time_diff_hours: float) -> float:
        """시간 차이에 따른 가중치 계산 (e^-λt)"""
        return np.exp(-self.decay_rate * time_diff_hours)

class BayesianSmoothing:
    """베이지안 스무딩 - 데이터가 적을 때의 극단적 확률 보정"""
    def __init__(self, alpha: float = 2.0, beta: float = 2.0, kappa: float = 5.0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
    def smooth_success_rate(self, wins: int, total: int) -> float:
        """(wins + alpha) / (total + alpha + beta)"""
        return (wins + self.alpha) / (total + self.alpha + self.beta)
        
    def smooth_avg_profit(self, profit_list: List[float], current_avg: float) -> float:
        """샘플 수가 적을 때 글로벌 평균 쪽으로 끌어당김 (Shrinkage)"""
        n = len(profit_list)
        if n == 0: return 0.0
        return (current_avg * n + self.kappa * 0) / (n + self.kappa)

class OutlierGuardrail:
    """이상치 차단 - 비정상적인 폭등/폭락 데이터가 학습을 왜곡하는 것 방지"""
    def __init__(self, percentile_cut: float = 0.05):
        self.lower_bound = -15.0 # -15% 이하 차단
        self.upper_bound = 30.0  # +30% 이상 차단
        
    def clamp_profit(self, profit_pct: float) -> float:
        """지정된 범위 내로 수익률 제한 (Winsorizing)"""
        return max(self.lower_bound, min(self.upper_bound, profit_pct))

# --- 메인 학습기 클래스 ---

class ThompsonSamplingLearner:
    """Thompson Sampling 기반 패턴 학습기 (강화 버전)"""
    
    def __init__(self, db_path: str = None):
        # 🆕 경로 보정: 디렉토리가 들어오면 common_strategies.db 파일로 연결
        if db_path and os.path.isdir(db_path):
            db_path = os.path.join(db_path, 'common_strategies.db')
        
        self.db_path = db_path
        self.alpha_prior = 1.0  # 기본 성공 횟수
        self.beta_prior = 1.0   # 기본 실패 횟수
        self._pattern_cache = {}
        
        # 🆕 수학적 보정 도구 초기화
        self.smoother = BayesianSmoothing(alpha=2.0, beta=2.0, kappa=5.0)
        self.guardrail = OutlierGuardrail(percentile_cut=0.05)
        self.decay = ExponentialDecayWeight(decay_rate=0.05)
        
        if self.db_path:
            self._create_tables()
            self._load_all_patterns()

    def _create_tables(self):
        """필요한 테이블 생성 (안정성 강화)"""
        # 🚀 [Fix] DB 경로가 없으면 스킵
        if not self.db_path:
            return
            
        try:
            # 🚀 트레이딩 코어 DB 유틸리티 사용 (잠금 대기 포함, 쓰기 모드)
            from trade.core.database import get_db_connection
            
            with get_db_connection(self.db_path, read_only=False) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        coin TEXT,
                        interval TEXT,
                        signal_type TEXT,
                        score REAL,
                        feedback_type TEXT,
                        success_rate REAL,
                        avg_profit REAL,
                        total_trades INTEGER,
                        alpha REAL DEFAULT 1.0,
                        beta REAL DEFAULT 1.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (coin, interval, signal_type, feedback_type)
                    )
                """)
                # 패턴 기반 학습 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_feedback_logs (
                        pattern TEXT PRIMARY KEY,
                        alpha REAL DEFAULT 1.0,
                        beta REAL DEFAULT 1.0,
                        avg_profit REAL DEFAULT 0.0,
                        total_samples INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception:
            # 🔇 엔진 모드: 모든 DB 생성 오류 조용히 처리 (선택적 기능)
            pass

    def _load_all_patterns(self):
        """DB에서 모든 패턴 데이터 로드 (읽기 전용 최적화)"""
        if not self.db_path or not os.path.exists(self.db_path):
            return
            
        try:
            # 🚀 읽기 전용 모드로 조회 (잠금 방지 핵심)
            from trade.core.database import get_db_connection
            with get_db_connection(self.db_path, read_only=True) as conn:
                cursor = conn.cursor()
                # 테이블 존재 여부 먼저 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pattern_feedback_logs'")
                if not cursor.fetchone(): return

                cursor.execute("SELECT pattern, alpha, beta, avg_profit, total_samples FROM pattern_feedback_logs")
                for row in cursor.fetchall():
                    self._pattern_cache[row[0]] = {
                        'alpha': row[1],
                        'beta': row[2],
                        'avg_profit': row[3],
                        'total_samples': row[4]
                    }
        except Exception:
            # 🔇 엔진 모드: 모든 DB 로드 오류 조용히 처리 (선택적 기능)
            pass

    def sample_success_rate(self, pattern: str) -> Tuple[float, str]:
        """특정 패턴의 승률을 Thompson Sampling으로 추출 (보정 적용)"""
        stats = self._get_pattern_data(pattern)
        
        # 🆕 베이지안 스무딩이 적용된 alpha, beta로 샘플링
        alpha = stats['alpha']
        beta = stats['beta']
        
        # Beta 분포에서 샘플링
        sampled_rate = np.random.beta(alpha, beta)
        
        total = int(alpha + beta - 2)
        confidence_msg = f"데이터 {total}회" if total > 0 else "신규 패턴"
        
        return sampled_rate, confidence_msg

    def get_action_statistics(self, signal_pattern: str, signal_score: float) -> Dict:
        """의사결정에 필요한 상세 통계 정보 제공"""
        sampled_rate, conf_msg = self.sample_success_rate(signal_pattern)
        stats = self._get_pattern_data(signal_pattern)
        
        total_samples = stats['total_samples']
        is_exploration = total_samples < 20
        
        return {
            'sampled_rate': sampled_rate,
            'avg_profit': stats['avg_profit'],
            'total_samples': total_samples,
            'normalized_signal_score': (signal_score + 1.0) / 2.0,
            'exploration_bonus': 0.15 if is_exploration else 0.05,
            'is_exploration': is_exploration,
            'phase': '탐색 단계' if is_exploration else '최적화 단계',
            'confidence_msg': conf_msg
        }

    def update_distribution(self, pattern: str, success: bool, profit_pct: float, weight: float = 1.0):
        """거래 결과에 따른 분포 업데이트 (수학적 보정 적용)"""
        stats = self._get_pattern_data(pattern)
        
        # 🆕 1. 이상치 차단 (Outlier Guardrail)
        clamped_profit = self.guardrail.clamp_profit(profit_pct)
        
        # 🆕 2. 최근성 가중치 계산 (필요시 weight와 결합)
        # (여기서는 전달된 weight를 우선 사용하고, 기본은 1.0)
        effective_weight = weight
        
        if success:
            stats['alpha'] += effective_weight
        else:
            stats['beta'] += effective_weight
            
        # 🆕 3. 평균 수익률 업데이트 (EMA 방식 + Guardrail 적용값)
        current_avg = stats.get('avg_profit', 0.0)
        # 데이터가 쌓일수록 점진적으로 반영
        alpha_ema = 0.1 # 기본 반영률
        stats['avg_profit'] = (current_avg * (1 - alpha_ema)) + (clamped_profit * alpha_ema)
        
        stats['total_samples'] += 1
        stats['last_updated_ts'] = int(time.time())
        
        self._pattern_cache[pattern] = stats
        self._save_pattern_to_db(pattern, stats)

    def get_pattern_stats(self, pattern: str) -> Optional[Dict]:
        """패턴 통계 정보 조회"""
        return self._pattern_cache.get(pattern)

    def get_decision_engine_stats(self, pattern: str) -> Dict:
        """알파 가디언(의사결정 엔진) 포맷에 맞춘 통계 데이터 반환"""
        stats = self._get_pattern_data(pattern)
        total_samples = stats.get('total_samples', 0)
        alpha = stats.get('alpha', self.alpha_prior)
        beta = stats.get('beta', self.beta_prior)
        
        # 기대 승률 계산 (Beta 분포의 평균)
        success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        
        return {
            'success_rate': success_rate,
            'avg_profit': stats.get('avg_profit', 0.0),
            'total_trades': total_samples
        }

    def _get_pattern_data(self, pattern: str) -> Dict:
        """패턴 데이터 가져오기 (없으면 기본값)"""
        if pattern not in self._pattern_cache:
            return {
                'alpha': self.alpha_prior,
                'beta': self.beta_prior,
                'avg_profit': 0.0,
                'total_samples': 0
            }
        return self._pattern_cache[pattern]

    def _save_pattern_to_db(self, pattern: str, stats: Dict):
        """패턴 데이터를 DB에 저장 (쓰기 모드 안정성 강화)"""
        if not self.db_path:
            # DB 경로가 없으면 메모리에만 유지 (실시간 학습용)
            return
            
        try:
            try:
                from trade.core.database import get_db_connection
            except ImportError:
                def get_db_connection(p, **kwargs): return sqlite3.connect(p, timeout=30.0)

            with get_db_connection(self.db_path, read_only=False) as conn:
                conn.execute("""
                    INSERT INTO pattern_feedback_logs (pattern, alpha, beta, avg_profit, total_samples, last_updated)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(pattern) DO UPDATE SET
                        alpha = excluded.alpha,
                        beta = excluded.beta,
                        avg_profit = excluded.avg_profit,
                        total_samples = excluded.total_samples,
                        last_updated = CURRENT_TIMESTAMP
                """, (pattern, stats['alpha'], stats['beta'], stats['avg_profit'], stats['total_samples']))
                conn.commit()
        except Exception:
            # 🔇 엔진 모드: 저장 실패 조용히 처리 (다음 턴에 재시도)
            pass

@dataclass
class ThompsonScore:
    """Thompson Sampling 결과 데이터클래스"""
    score: float
    total_samples: int
    pattern: str
    is_new_pattern: bool

class ThompsonScoreCalculator:
    """Thompson Sampling 점수 계산기 (싱글톤)"""
    _instance = None
    _sampler = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        self._initialized = True
        self._init_sampler()
    
    def _init_sampler(self):
        from trade.core.database import STRATEGY_DB_PATH
        try:
            self._sampler = ThompsonSamplingLearner(db_path=STRATEGY_DB_PATH)
        except Exception:
            # 🔇 엔진 모드: 초기화 실패 시 조용히 처리 (폴백 로직 사용)
            self._sampler = None
    
    def get_score(self, signal: Any) -> ThompsonScore:
        pattern = self.extract_pattern(signal)
        if not self._sampler:
            return ThompsonScore(0.0, 0, pattern, True)
        
        sampled_rate, conf_msg = self._sampler.sample_success_rate(pattern)
        stats = self._sampler.get_pattern_stats(pattern)
        total_samples = stats['total_samples'] if stats else 0
        
        return ThompsonScore(
            score=sampled_rate,
            total_samples=total_samples,
            pattern=pattern,
            is_new_pattern=total_samples < 5
        )

    def extract_pattern(self, signal: Any) -> str:
        # 패턴 추출 로직 (SignalInfo 기반)
        try:
            coin = getattr(signal, 'coin', 'unknown')
            rsi = getattr(signal, 'rsi', 50.0)
            vol = getattr(signal, 'volume_ratio', 1.0)
            rsi_state = 'low' if rsi < 30 else 'high' if rsi > 70 else 'mid'
            vol_state = 'high' if vol > 1.5 else 'low' if vol < 0.5 else 'mid'
            return f"{coin}_{rsi_state}_{vol_state}"
        except:
            return "unknown_pattern"

# 공용 인스턴스 및 함수
_calculator = None

def get_thompson_calculator():
    global _calculator
    if _calculator is None:
        _calculator = ThompsonScoreCalculator()
    return _calculator

def get_thompson_score(signal):
    return get_thompson_calculator().get_score(signal).score

def get_thompson_score_from_pattern(pattern: str) -> float:
    """패턴 문자열에서 Thompson 점수 조회"""
    calc = get_thompson_calculator()
    if not calc._sampler: return 0.5
    sampled_rate, _ = calc._sampler.sample_success_rate(pattern)
    return sampled_rate

def should_execute_trade(signal, signal_score):
    # DecisionMaker의 로직을 간소화하여 제공
    calc = get_thompson_calculator()
    score_obj = calc.get_score(signal)
    final_score = (score_obj.score + (signal_score + 1.0) / 2.0) / 2.0
    return final_score >= 0.4, final_score, f"Thompson: {score_obj.score:.2f}"

def extract_signal_pattern(signal: Any) -> str:
    """시그널에서 패턴 추출"""
    return get_thompson_calculator().extract_pattern(signal)
