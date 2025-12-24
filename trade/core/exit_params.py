"""
학습된 청산 파라미터 조회 모듈

virtual_trade_learner.py에서 학습한 optimal_tp_ratio, optimal_sl_ratio를
가상매매/실전매매에서 사용할 수 있도록 제공

사용 위치:
- trade/virtual_trade_executor.py
- trade/trade_executor.py
"""

import os
import sqlite3
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# DB 경로 설정
_DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'market', 'coin_market', 'data_storage'
)

STRATEGY_DB_PATH = os.getenv('STRATEGY_DB_PATH')
if STRATEGY_DB_PATH and os.path.isdir(STRATEGY_DB_PATH):
    STRATEGY_DB_PATH = os.path.join(STRATEGY_DB_PATH, 'common_strategies.db')
elif not STRATEGY_DB_PATH:
    STRATEGY_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'learning_strategies', 'common_strategies.db')


@dataclass
class ExitParams:
    """청산 파라미터"""
    optimal_tp_ratio: float = 2.0  # 기본 TP 비율 (목표수익 / 리스크)
    optimal_sl_ratio: float = 1.0  # 기본 SL 비율
    samples: int = 0               # 학습 샘플 수
    confidence: float = 0.0        # 신뢰도 (샘플 수 기반)
    
    @property
    def is_learned(self) -> bool:
        """학습된 파라미터인지 여부"""
        return self.samples >= 5  # 최소 5회 이상 학습되어야 신뢰


# 캐시 (성능 최적화)
_exit_params_cache: Dict[str, ExitParams] = {}
_cache_timestamp: int = 0
_CACHE_TTL = 300  # 5분


def get_exit_params(signal_pattern: str) -> ExitParams:
    """
    패턴별 학습된 청산 파라미터 조회
    
    Args:
        signal_pattern: 시그널 패턴 (예: 'oversold_bullish_high_up')
        
    Returns:
        ExitParams: 학습된 청산 파라미터 (없으면 기본값)
    """
    import time
    global _exit_params_cache, _cache_timestamp
    
    # 캐시 만료 체크
    current_time = int(time.time())
    if current_time - _cache_timestamp > _CACHE_TTL:
        _exit_params_cache.clear()
        _cache_timestamp = current_time
    
    # 캐시 히트
    if signal_pattern in _exit_params_cache:
        return _exit_params_cache[signal_pattern]
    
    # DB 조회
    params = _load_exit_params_from_db(signal_pattern)
    _exit_params_cache[signal_pattern] = params
    
    return params


def _load_exit_params_from_db(signal_pattern: str) -> ExitParams:
    """DB에서 청산 파라미터 로드"""
    try:
        if not os.path.exists(STRATEGY_DB_PATH):
            return ExitParams()
        
        with sqlite3.connect(STRATEGY_DB_PATH, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            # 테이블 존재 여부 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='pattern_exit_params'
            """)
            if not cursor.fetchone():
                return ExitParams()
            
            # 파라미터 조회
            cursor.execute("""
                SELECT optimal_tp_ratio, optimal_sl_ratio, samples
                FROM pattern_exit_params
                WHERE signal_pattern = ?
            """, (signal_pattern,))
            
            row = cursor.fetchone()
            if row:
                tp_ratio, sl_ratio, samples = row
                confidence = min(samples / 20.0, 1.0)  # 20회 기준 최대 신뢰도
                return ExitParams(
                    optimal_tp_ratio=tp_ratio or 2.0,
                    optimal_sl_ratio=sl_ratio or 1.0,
                    samples=samples or 0,
                    confidence=confidence
                )
            
    except Exception as e:
        print(f"⚠️ 청산 파라미터 로드 오류: {e}")
    
    return ExitParams()


def should_take_profit(
    profit_pct: float,
    max_profit_pct: float,
    signal_pattern: str,
    entry_volatility: float = 0.02
) -> Tuple[bool, str]:
    """
    학습 기반 익절 판단
    
    Args:
        profit_pct: 현재 수익률 (%)
        max_profit_pct: 최고 수익률 (%)
        signal_pattern: 시그널 패턴
        entry_volatility: 진입 시 변동성
        
    Returns:
        Tuple[bool, str]: (익절 여부, 사유)
    """
    # 🔒 안전장치: +50% 이상 무조건 익절 (하드코딩)
    if profit_pct >= 50.0:
        return True, "safety_take_profit_50pct"
    
    # 학습된 파라미터 조회
    params = get_exit_params(signal_pattern)
    
    if params.is_learned:
        # 🎓 학습 기반 익절
        # 변동성 기반 목표 수익 계산
        base_target = entry_volatility * 100 * params.optimal_tp_ratio
        target_profit = max(base_target, 3.0)  # 최소 3%
        
        # 목표 수익 도달 시 익절
        if profit_pct >= target_profit:
            return True, f"learned_tp_{target_profit:.1f}pct"
        
        # 트레일링 스탑: 최고점 대비 1/3 반납 시 익절
        if max_profit_pct >= 10.0:
            retracement = max_profit_pct - profit_pct
            allowed_retracement = max_profit_pct / 3  # 1/3 허용
            
            if retracement >= allowed_retracement:
                return True, f"trailing_stop_{max_profit_pct:.1f}pct_peak"
    else:
        # 🔧 기본 익절 로직 (학습 전)
        # 10% 수익 시 부분 익절 고려
        if profit_pct >= 10.0:
            return True, "default_tp_10pct"
        
        # 트레일링: 최고점 20% 도달 후 5% 반납 시
        if max_profit_pct >= 20.0 and profit_pct <= (max_profit_pct - 5.0):
            return True, "default_trailing_20pct"
        
        # 트레일링: 최고점 10% 도달 후 3% 반납 시
        if max_profit_pct >= 10.0 and profit_pct <= (max_profit_pct - 3.0):
            return True, "default_trailing_10pct"
    
    return False, "hold"


def should_stop_loss(
    profit_pct: float,
    signal_pattern: str,
    entry_volatility: float = 0.02,
    holding_hours: float = 0
) -> Tuple[bool, str]:
    """
    학습 기반 손절 판단
    
    Args:
        profit_pct: 현재 수익률 (%)
        signal_pattern: 시그널 패턴
        entry_volatility: 진입 시 변동성
        holding_hours: 보유 시간 (시간)
        
    Returns:
        Tuple[bool, str]: (손절 여부, 사유)
    """
    # 🔒 안전장치: -10% 이하 무조건 손절 (하드코딩)
    if profit_pct <= -10.0:
        return True, "safety_stop_loss_10pct"
    
    # 학습된 파라미터 조회
    params = get_exit_params(signal_pattern)
    
    if params.is_learned:
        # 🎓 학습 기반 손절
        # 변동성 기반 손절 라인 계산
        base_stop = entry_volatility * 100 * params.optimal_sl_ratio
        stop_loss_line = max(base_stop, 2.0)  # 최소 2%
        stop_loss_line = min(stop_loss_line, 8.0)  # 최대 8%
        
        if profit_pct <= -stop_loss_line:
            return True, f"learned_sl_{stop_loss_line:.1f}pct"
    else:
        # 🔧 기본 손절 로직 (학습 전)
        # 시간에 따른 동적 손절 (오래 보유할수록 더 넉넉하게)
        if holding_hours < 2:
            # 2시간 미만: -5% 손절
            if profit_pct <= -5.0:
                return True, "default_sl_5pct_early"
        elif holding_hours < 12:
            # 12시간 미만: -7% 손절
            if profit_pct <= -7.0:
                return True, "default_sl_7pct_mid"
        else:
            # 12시간 이상: -8% 손절
            if profit_pct <= -8.0:
                return True, "default_sl_8pct_late"
    
    return False, "hold"


def get_trailing_stop_params(signal_pattern: str) -> Dict:
    """
    패턴별 트레일링 스탑 파라미터
    
    Returns:
        Dict: {'activation_pct': 활성화 수익률, 'trailing_pct': 추적 비율}
    """
    params = get_exit_params(signal_pattern)
    
    if params.is_learned and params.confidence > 0.5:
        # 학습 기반: TP 비율에 따라 트레일링 파라미터 조정
        # TP 비율이 높으면 → 더 오래 버티도록 설정
        tp_ratio = params.optimal_tp_ratio
        
        if tp_ratio >= 3.0:
            # 공격적 패턴: 높은 활성화, 넓은 추적
            return {'activation_pct': 15.0, 'trailing_pct': 5.0}
        elif tp_ratio >= 2.0:
            # 보통 패턴
            return {'activation_pct': 10.0, 'trailing_pct': 3.0}
        else:
            # 보수적 패턴
            return {'activation_pct': 5.0, 'trailing_pct': 2.0}
    
    # 기본값
    return {'activation_pct': 10.0, 'trailing_pct': 3.0}

