#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
트레이딩 의사결정 전략 (Advanced)
횡보장 박스권 매매, 돌파 감지 및 슬리피지 최적화

🔥 공통 원칙:
- 시그널의 action(BUY/SELL)은 "미보유 상태 기준"으로 생성됨
- 보유 중인 코인에 대해서는 signal_score와 보유 정보(수익률, 보유시간 등)를 종합 판단
- 시그널이 SELL이어도 바로 매도하지 않음! decide_final_action 결과만 존중

🆕 학습 기반 매도 전략:
- 매수는 기술적 분석(시그널)으로 진입
- 매도는 학습 데이터(경험) 70% + 시그널 30% 비중으로 판단
"""

import os
import sqlite3
from typing import Dict, Optional, Any, Tuple
from trade.core.models import SignalInfo, SignalAction

# ============================================================================
# 🆕 경험 기반 신뢰도 성숙 시스템 (Experience-Based Trust Maturation)
# ============================================================================

# 학습 성숙도 캐시
_LEARNING_MATURITY_CACHE = {'data': None, 'timestamp': 0}
_MATURITY_CACHE_TTL = 600  # 10분 캐시

def get_learning_maturity() -> Dict[str, Any]:
    """
    학습 성숙도 계산 (거래 횟수, 패턴 샘플, 학습 기간, 수익비 기반)
    
    Returns:
        {
            'maturity_score': 0.0~1.0,      # 전체 성숙도 점수
            'total_trades': int,             # 총 거래 횟수
            'avg_samples_per_pattern': float, # 패턴별 평균 샘플 수
            'learning_days': int,            # 학습 기간 (일)
            'profit_ratio': float,           # 수익비 (총수익/총손실)
            'stage': 'initial' | 'growing' | 'mature'
        }
    """
    import time
    global _LEARNING_MATURITY_CACHE
    
    # 캐시 확인
    if (_LEARNING_MATURITY_CACHE['data'] is not None and 
        time.time() - _LEARNING_MATURITY_CACHE['timestamp'] < _MATURITY_CACHE_TTL):
        return _LEARNING_MATURITY_CACHE['data']
    
    # 기본값 (초기 상태)
    result = {
        'maturity_score': 0.0,
        'total_trades': 0,
        'avg_samples_per_pattern': 0.0,
        'learning_days': 0,
        'profit_ratio': 1.0,
        'stage': 'initial'
    }
    
    try:
        global_db = os.environ.get('GLOBAL_STRATEGY_DB_PATH')
        if not global_db or not os.path.exists(global_db):
            # 기본 DB 경로 시도
            fallback_paths = [
                'market/coin_market/data_storage/learning_strategies/common_strategies.db',
                'C:/auto_trader/market/coin_market/data_storage/learning_strategies/common_strategies.db'
            ]
            for path in fallback_paths:
                if os.path.exists(path):
                    global_db = path
                    break
        
        if not global_db or not os.path.exists(global_db):
            return result
        
        with sqlite3.connect(global_db, timeout=5.0) as conn:
            # 1. 총 거래 횟수 (Thompson 분포 업데이트 기록 기반)
            cursor = conn.execute("""
                SELECT SUM(alpha + beta - 2) as total_trades
                FROM thompson_distributions
                WHERE alpha + beta > 2
            """)
            row = cursor.fetchone()
            total_trades = int(row[0]) if row and row[0] else 0
            result['total_trades'] = total_trades
            
            # 2. 패턴별 평균 샘플 수
            cursor = conn.execute("""
                SELECT AVG(sample_count) as avg_samples
                FROM optimal_thresholds
                WHERE sample_count > 0
            """)
            row = cursor.fetchone()
            avg_samples = float(row[0]) if row and row[0] else 0.0
            result['avg_samples_per_pattern'] = avg_samples
            
            # 3. 학습 기간 (가장 오래된 피드백 데이터 기준)
            try:
                from trade.core.database import TRADING_SYSTEM_DB_PATH
                with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=5.0) as trade_conn:
                    cursor = trade_conn.execute("""
                        SELECT MIN(exit_timestamp), MAX(exit_timestamp)
                        FROM virtual_trade_feedback
                        WHERE exit_timestamp > 0
                    """)
                    row = cursor.fetchone()
                    if row and row[0] and row[1]:
                        learning_days = max(1, (row[1] - row[0]) // 86400)
                        result['learning_days'] = learning_days
            except:
                result['learning_days'] = max(1, total_trades // 10)  # 추정값
            
            # 4. 수익비 계산 (총수익 / 총손실)
            try:
                from trade.core.database import TRADING_SYSTEM_DB_PATH
                with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=5.0) as trade_conn:
                    cursor = trade_conn.execute("""
                        SELECT 
                            SUM(CASE WHEN profit_loss_pct > 0 THEN profit_loss_pct ELSE 0 END) as total_profit,
                            SUM(CASE WHEN profit_loss_pct < 0 THEN ABS(profit_loss_pct) ELSE 0 END) as total_loss
                        FROM virtual_trade_feedback
                    """)
                    row = cursor.fetchone()
                    if row:
                        total_profit = row[0] if row[0] else 0.0
                        total_loss = row[1] if row[1] else 1.0  # 0 나누기 방지
                        profit_ratio = total_profit / max(total_loss, 1.0)
                        result['profit_ratio'] = round(profit_ratio, 2)
            except:
                result['profit_ratio'] = 1.0
        
        # 5. 성숙도 점수 계산 (0.0 ~ 1.0)
        # 각 요소별 점수 (0~1)
        trade_score = min(1.0, total_trades / 100)  # 100회 이상이면 1.0
        sample_score = min(1.0, avg_samples / 15)    # 15개 이상이면 1.0
        days_score = min(1.0, result['learning_days'] / 7)  # 7일 이상이면 1.0
        
        # 수익비 점수 (0.5~1.5 → 0~1)
        profit_score = min(1.0, max(0.0, (result['profit_ratio'] - 0.5) / 1.0))
        
        # 가중 평균 (거래 횟수 40%, 샘플 수 25%, 학습 기간 20%, 수익비 15%)
        maturity = (trade_score * 0.40 + 
                   sample_score * 0.25 + 
                   days_score * 0.20 + 
                   profit_score * 0.15)
        
        result['maturity_score'] = round(maturity, 3)
        
        # 6. 단계 결정
        if maturity < 0.3:
            result['stage'] = 'initial'
        elif maturity < 0.6:
            result['stage'] = 'growing'
        else:
            result['stage'] = 'mature'
        
        # 캐시 저장
        _LEARNING_MATURITY_CACHE = {'data': result, 'timestamp': time.time()}
        
    except Exception as e:
        pass  # 오류 시 기본값 사용
    
    return result


def get_dynamic_weights(
    for_buy: bool = True,
    signal_confidence: float = 0.5,
    pattern_confidence: float = 0.5,
    interval_alignment: float = 0.5
) -> Tuple[float, float, str]:
    """
    🆕 이중 신뢰도 기반 동적 가중치 계산
    
    Args:
        for_buy: True = 매수 결정용, False = 매도 결정용
        signal_confidence: 시그널 품질/신뢰도 (0.0 ~ 1.0)
            - RSI 극단값, 인터벌 일치, 변동성 등 기반
        pattern_confidence: 패턴별 학습 신뢰도 (0.0 ~ 1.0)
            - 샘플 수, 수익 일관성 기반
        interval_alignment: 인터벌 방향 일치도 (0.0 ~ 1.0)
            - 1d/240m/30m/15m 방향이 일치할수록 높음
    
    Returns:
        (signal_weight, learning_weight, stage_description)
        - signal_weight: 시그널 점수 가중치 (0.0 ~ 1.0)
        - learning_weight: 학습 데이터 가중치 (0.0 ~ 1.0)
        - stage_description: 현재 단계 설명
        
    🎯 이중 신뢰도 시나리오:
        - 시그널↑ 학습↑ → 적극 매매 (타이밍 가중치↑)
        - 시그널↓ 학습↑ → 학습 기반 매매
        - 시그널↑ 학습↓ → 시그널 기반 매매
        - 시그널↓ 학습↓ → 보수적 매매 (HOLD 우선)
    """
    maturity = get_learning_maturity()
    stage = maturity['stage']
    score = maturity['maturity_score']
    profit_ratio = maturity['profit_ratio']
    
    # =========================================================================
    # [1단계] 전역 성숙도 기반 기본 가중치
    # =========================================================================
    if stage == 'initial':
        base_signal = 0.80
        base_learning = 0.20
        stage_desc = "초기"
    elif stage == 'growing':
        base_signal = 0.55
        base_learning = 0.45
        stage_desc = "성장"
    else:  # mature
        base_signal = 0.30
        base_learning = 0.70
        stage_desc = "성숙"
    
    # =========================================================================
    # [2단계] 수익비 패널티 (나쁜 경험에 덜 의존)
    # =========================================================================
    profit_penalty = 1.0
    if profit_ratio < 0.7:
        profit_penalty = 0.5
    elif profit_ratio < 1.0:
        profit_penalty = 0.8
    
    # =========================================================================
    # 🆕 [3단계] 이중 신뢰도 기반 동적 조정
    # =========================================================================
    # 신뢰도 조정 계수 계산
    signal_trust = signal_confidence * 0.6 + interval_alignment * 0.4  # 시그널 + 인터벌 일치
    learning_trust = pattern_confidence * profit_penalty  # 패턴 신뢰도 × 수익비
    
    # 신뢰도 차이에 따른 가중치 시프트
    trust_diff = signal_trust - learning_trust  # -1.0 ~ +1.0
    
    # 시프트 계수: 최대 ±20% 조정
    shift_factor = trust_diff * 0.20
    
    # 기본 가중치에 시프트 적용
    adjusted_signal = base_signal + shift_factor
    adjusted_learning = base_learning - shift_factor
    
    # 경계 제한 (최소 20%, 최대 80%)
    adjusted_signal = max(0.20, min(0.80, adjusted_signal))
    adjusted_learning = max(0.20, min(0.80, adjusted_learning))
    
    # 합이 1.0이 되도록 정규화
    total = adjusted_signal + adjusted_learning
    adjusted_signal = adjusted_signal / total
    adjusted_learning = adjusted_learning / total
    
    # =========================================================================
    # 🆕 [4단계] 양쪽 신뢰도 모두 높으면 적극 매매 플래그
    # =========================================================================
    both_confident = signal_trust > 0.6 and learning_trust > 0.6
    both_uncertain = signal_trust < 0.4 and learning_trust < 0.4
    
    confidence_desc = ""
    if both_confident:
        confidence_desc = " 🟢확신"
    elif both_uncertain:
        confidence_desc = " 🟡신중"
    elif signal_trust > learning_trust + 0.2:
        confidence_desc = " 📊시그널↑"
    elif learning_trust > signal_trust + 0.2:
        confidence_desc = " 📚학습↑"
    
    # =========================================================================
    # [5단계] 매수/매도별 미세 조정
    # =========================================================================
    if not for_buy:
        # 매도는 학습 데이터를 조금 더 신뢰 (+5%)
        adjusted_learning = min(0.75, adjusted_learning + 0.05)
        adjusted_signal = 1.0 - adjusted_learning
    
    desc = f"{stage_desc} (시그널:{signal_trust:.2f}/학습:{learning_trust:.2f}){confidence_desc}"
    
    return (round(adjusted_signal, 2), round(adjusted_learning, 2), desc)


# ============================================================================
# 🆕 학습 데이터 기반 매도 전략 (Learning-Based Exit Strategy)
# ============================================================================

# 학습 데이터 캐시 (메모리 최적화)
_LEARNED_THRESHOLDS_CACHE = {}
_CACHE_EXPIRY = 300  # 5분 캐시

def get_learned_exit_thresholds(pattern: str) -> Dict[str, Any]:
    """
    학습된 매도 임계값 로드 (optimal_thresholds 테이블)
    
    Returns:
        {
            'optimal_stop_loss': -3.5,      # 학습된 최적 손절선
            'optimal_take_profit': 5.2,     # 학습된 최적 익절선
            'optimal_holding_hours': 4.0,   # 학습된 최적 보유 기간
            'avg_mfe': 6.1,                 # 평균 최대 유리 변동
            'avg_mae': 2.8,                 # 평균 최대 불리 변동
            'sample_count': 25,             # 학습 샘플 수
            'confidence': 0.75              # 신뢰도 (샘플 수 기반)
        }
    """
    import time
    
    # 캐시 확인
    cache_key = pattern.split('_')[0] if pattern else 'unknown'  # 기본 패턴만 사용
    if cache_key in _LEARNED_THRESHOLDS_CACHE:
        cached = _LEARNED_THRESHOLDS_CACHE[cache_key]
        if time.time() - cached['timestamp'] < _CACHE_EXPIRY:
            return cached['data']
    
    # 기본값
    defaults = {
        'optimal_stop_loss': -5.0,
        'optimal_take_profit': 5.0,
        'optimal_holding_hours': 6.0,
        'avg_mfe': 5.0,
        'avg_mae': 3.0,
        'sample_count': 0,
        'confidence': 0.0
    }
    
    try:
        global_db = os.environ.get('GLOBAL_STRATEGY_DB_PATH')
        if not global_db or not os.path.exists(global_db):
            return defaults
        
        with sqlite3.connect(global_db, timeout=5.0) as conn:
            # 1. optimal_thresholds에서 손절/익절 임계값 로드
            cursor = conn.execute("""
                SELECT optimal_stop_loss, optimal_take_profit, avg_mae, avg_mfe, sample_count
                FROM optimal_thresholds
                WHERE pattern = ?
            """, (cache_key,))
            row = cursor.fetchone()
            
            if row:
                defaults['optimal_stop_loss'] = row[0] if row[0] else -5.0
                defaults['optimal_take_profit'] = row[1] if row[1] else 5.0
                defaults['avg_mae'] = row[2] if row[2] else 3.0
                defaults['avg_mfe'] = row[3] if row[3] else 5.0
                defaults['sample_count'] = row[4] if row[4] else 0
                defaults['confidence'] = min(1.0, defaults['sample_count'] / 20)  # 20건 이상이면 신뢰도 1.0
            
            # 2. 보유 기간 학습 데이터 (Thompson Sampling 결과)
            try:
                cursor = conn.execute("""
                    SELECT alpha, beta 
                    FROM thompson_distributions
                    WHERE pattern LIKE ?
                    ORDER BY (alpha + beta) DESC
                    LIMIT 1
                """, (f"{cache_key}_holding%",))
                holding_row = cursor.fetchone()
                if holding_row:
                    # Thompson 분포에서 추정 (알파가 높을수록 좋은 결과)
                    alpha, beta = holding_row
                    if alpha + beta > 5:  # 충분한 샘플
                        # 알파가 높을수록 오래 보유하는 게 좋았다는 의미
                        if alpha > beta:
                            defaults['optimal_holding_hours'] = 8.0  # 장기 보유 권장
                        else:
                            defaults['optimal_holding_hours'] = 4.0  # 단기 익절 권장
            except:
                pass
        
        # 캐시 저장
        _LEARNED_THRESHOLDS_CACHE[cache_key] = {
            'data': defaults,
            'timestamp': time.time()
        }
        
    except Exception as e:
        pass  # DB 오류 시 기본값 사용
    
    return defaults


def calculate_learning_based_sell_score(
    profit_loss_pct: float,
    holding_hours: float,
    max_profit_pct: float,
    learned_thresholds: Dict[str, Any]
) -> Tuple[float, str]:
    """
    학습 데이터 기반 매도 점수 계산 (0.0 ~ 1.0)
    
    Returns:
        (sell_score, reason)
        - sell_score: 0.0 = 절대 매도 금지, 1.0 = 즉시 매도
    """
    sell_score = 0.0
    reasons = []
    
    optimal_tp = learned_thresholds['optimal_take_profit']
    optimal_sl = learned_thresholds['optimal_stop_loss']
    optimal_holding = learned_thresholds['optimal_holding_hours']
    avg_mfe = learned_thresholds['avg_mfe']
    confidence = learned_thresholds['confidence']
    
    # 학습 데이터 신뢰도가 낮으면 (샘플 부족) 기본 로직 사용
    if confidence < 0.3:
        return (0.0, "학습 데이터 부족")
    
    # 1. 익절 조건 평가 (학습된 MFE 기준)
    if profit_loss_pct > 0:
        # 학습된 평균 MFE의 80%에 도달하면 익절 점수 급상승
        mfe_ratio = profit_loss_pct / max(avg_mfe, 1.0)
        if mfe_ratio >= 0.8:
            sell_score += 0.5 * confidence
            reasons.append(f"MFE 80% 도달({profit_loss_pct:.1f}%/{avg_mfe:.1f}%)")
        elif mfe_ratio >= 0.5:
            sell_score += 0.2 * confidence
        
        # 학습된 익절선 도달
        if profit_loss_pct >= optimal_tp:
            sell_score += 0.3 * confidence
            reasons.append(f"학습된 익절선({optimal_tp:.1f}%) 도달")
    
    # 2. 손절 조건 평가 (학습된 MAE 기준)
    if profit_loss_pct < 0:
        # 학습된 손절선 도달
        if profit_loss_pct <= optimal_sl:
            sell_score += 0.6 * confidence
            reasons.append(f"학습된 손절선({optimal_sl:.1f}%) 도달")
    
    # 3. 보유 기간 평가
    if optimal_holding > 0:
        holding_ratio = holding_hours / optimal_holding
        
        # 최적 보유 기간 초과 + 수익 중
        if holding_ratio > 1.2 and profit_loss_pct > 0:
            sell_score += 0.3 * confidence
            reasons.append(f"최적 보유기간({optimal_holding:.1f}h) 초과")
        
        # 최적 보유 기간 초과 + 손실 중 (빠른 손절)
        elif holding_ratio > 1.5 and profit_loss_pct < 0:
            sell_score += 0.4 * confidence
            reasons.append(f"장기 손실 보유({holding_hours:.1f}h)")
    
    # 4. 수익 반납 감지 (최고점 대비)
    if max_profit_pct > 2.0 and profit_loss_pct < (max_profit_pct * 0.5):
        sell_score += 0.4 * confidence
        reasons.append(f"수익 반납(최고 {max_profit_pct:.1f}% → 현재 {profit_loss_pct:.1f}%)")
    
    # 점수 상한
    sell_score = min(1.0, sell_score)
    reason = " | ".join(reasons) if reasons else "학습 기반 평가"
    
    return (sell_score, reason)


def should_sell_holding_position(
    signal_score: float,
    profit_loss_pct: float,
    max_profit_pct: float,
    holding_hours: float,
    tick_size: float = 0.0,
    current_price: float = 0.0,
    trend_analysis: Any = None,
    signal_continuity: float = 0.5,  # 🆕 시그널 연속성 (0~1)
    dynamic_influence: float = 0.5   # 🆕 동적 영향도 (0~1)
) -> tuple:
    """
    🔥 [공통 기준] 보유 중인 코인에 대한 매도 여부 판단
    
    시그널의 action(BUY/SELL)이 아니라 signal_score와 보유 정보를 종합 판단합니다.
    이 함수는 virtual_trade_executor.py와 trade_executor.py 모두에서 사용됩니다.
    
    Args:
        signal_continuity: 이전 시그널과의 방향성 일치도 (0=급격한 반전, 1=일관된 방향)
        dynamic_influence: 시그널 품질 기반 동적 영향도 (0=저품질, 1=고품질)
    
    Returns:
        (should_sell: bool, reason: str)
    """
    # 🆕 [0단계] 호가 해상도 필터 (Tick-Aware Noise Filter)
    def is_significant_move(target_pct: float) -> bool:
        if tick_size <= 0 or current_price <= 0: return True
        tick_ratio = (tick_size / current_price * 100)
        if tick_ratio < 0.5: return True  # 고해상도 자산
        move_abs = abs(target_pct / 100 * current_price)
        ticks = move_abs / tick_size
        return ticks >= 3.0  # 최소 3틱 변동 확인
    
    # [1단계] 하드 룰 (절대 보호)
    if profit_loss_pct >= 50.0:
        return (True, '대박 수익 달성 (+50%)')
    if profit_loss_pct <= -10.0:
        return (True, '손절선 도달 (-10%)')
    
    # [2단계] 수익 반납 보호 (Trailing Stop)
    if max_profit_pct >= 2.0 and profit_loss_pct < (max_profit_pct * 0.5):
        if is_significant_move(max_profit_pct - profit_loss_pct):
            return (True, f'수익 반납 감지 (최고 {max_profit_pct:.1f}% → 현재 {profit_loss_pct:.1f}%)')
    
    # [3단계] 손실 장기화 + 하락 추세
    if profit_loss_pct < -1.0 and holding_hours >= 4.0:
        if trend_analysis and hasattr(trend_analysis, 'trend_type'):
            if trend_analysis.trend_type.value in ['bearish', 'weak_bearish', 'strong_down', 'down']:
                return (True, f'손실 장기화 + 하락 추세 ({holding_hours:.1f}h, {profit_loss_pct:.1f}%)')
    
    # [4단계] 극단적 리스크 점수 (신호가 매우 강하게 SELL일 때만)
    if signal_score < -0.5:
        # 🆕 연속성이 낮으면(급격한 반전) 더 신중하게 판단
        if signal_continuity < 0.3:
            # 급격한 반전이지만 이전에 강한 상승 신호였다면 한 번 더 확인
            return (True, f'리스크 점수 임계값 초과 ({signal_score:.3f}, 급반전 주의)')
        return (True, f'리스크 점수 임계값 초과 ({signal_score:.3f})')
    
    # [5단계] 기술적 점수 미달 (-0.35 이하)
    # 🔥 [수정] -0.3 → -0.35로 강화 (일관성 유지)
    # 시그널이 단순히 SELL이라고 팔지 않음. 점수가 -0.35 이하일 때만 매도 고려
    if signal_score < -0.35:
        # 🆕 연속성/영향도에 따른 조정
        # 연속성이 높으면(일관된 하락) 매도 신뢰, 낮으면(급반전) 신중
        adjusted_threshold = -0.35
        if signal_continuity < 0.3:
            adjusted_threshold = -0.45  # 급반전 시 더 엄격한 기준
        elif signal_continuity > 0.7 and dynamic_influence > 0.6:
            adjusted_threshold = -0.30  # 일관된 고품질 신호 시 더 빠른 대응
        
        if signal_score < adjusted_threshold:
            if is_significant_move(profit_loss_pct):
                return (True, f'기술적 점수 미달 ({signal_score:.3f}, 연속성: {signal_continuity:.2f})')
    
    # 매도하지 않음
    return (False, '')


def decide_final_action(
    coin: str,
    signal_score: float,
    profit_loss_pct: float,
    max_profit_pct: float,
    signal_pattern: str,
    market_adjustment: float,
    holding_hours: float = 0.0,
    trend_analysis: Any = None,
    learned_threshold: Optional[float] = None,
    ai_decision: str = 'hold',
    tick_size: float = 0.0,
    current_price: float = 0.0,
    signal_continuity: float = 0.5,
    dynamic_influence: float = 0.5
) -> str:
    """
    🔥 계층적 의사결정 (학습 기반 매도 전략 적용)
    
    핵심 원칙:
    - 매수: 기술적 분석(시그널) 100%
    - 매도: 학습 데이터 70% + 시그널 30% (시그널 널뛰기 방지)
    
    Args:
        signal_continuity: 이전 시그널과의 방향성 일치도 (0=급격한 반전, 1=일관된 방향)
        dynamic_influence: 시그널 품질 기반 동적 영향도 (0=저품질, 1=고품질)
    """
    
    # 🆕 [0단계] 호가 해상도 필터 (Tick-Aware Noise Filter)
    MIN_JITTER_TICKS = 3
    tick_ratio = (tick_size / current_price * 100) if tick_size > 0 and current_price > 0 else 0.0
    is_low_resolution = tick_ratio > 0.5
    
    def is_significant_move(target_pct: float) -> bool:
        if not is_low_resolution or tick_size <= 0: return True
        move_abs = abs(target_pct / 100 * current_price)
        ticks = move_abs / tick_size
        return ticks >= MIN_JITTER_TICKS

    # =========================================================================
    # [1단계] 하드 룰 (절대 보호) - 학습/시그널 무관하게 즉시 실행
    # =========================================================================
    if profit_loss_pct >= 50.0: return 'take_profit'
    if profit_loss_pct <= -10.0: return 'stop_loss'

    # =========================================================================
    # 🆕 [2단계] 학습 기반 매도 점수 계산 (Learning-Based Exit)
    # =========================================================================
    learned_thresholds = get_learned_exit_thresholds(signal_pattern)
    learning_sell_score, learning_reason = calculate_learning_based_sell_score(
        profit_loss_pct=profit_loss_pct,
        holding_hours=holding_hours,
        max_profit_pct=max_profit_pct,
        learned_thresholds=learned_thresholds
    )
    
    # 🆕 시그널 기반 매도 점수 변환 (-1~1 → 0~1 매도 점수)
    # signal_score가 음수일수록 매도 점수 높음
    signal_sell_score = 0.0
    if signal_score < 0:
        signal_sell_score = min(1.0, abs(signal_score) / 0.5)  # -0.5 이하면 1.0
    
    # 🆕 [핵심] 동적 가중치 - 학습 성숙도에 따라 조정 (최대 70%)
    SIGNAL_WEIGHT, LEARNING_WEIGHT, maturity_desc = get_dynamic_weights(for_buy=False)
    
    # 추가로 패턴별 학습 신뢰도 반영 (신뢰도 낮으면 학습 가중치 감소)
    pattern_confidence = learned_thresholds.get('confidence', 0.0)
    if pattern_confidence < 0.3:
        # 패턴별 샘플 부족 시 전역 성숙도만 사용
        LEARNING_WEIGHT = max(0.20, LEARNING_WEIGHT * 0.5)
        SIGNAL_WEIGHT = 1.0 - LEARNING_WEIGHT
    
    combined_sell_score = (learning_sell_score * LEARNING_WEIGHT) + (signal_sell_score * SIGNAL_WEIGHT)
    
    # 매도 임계값 (0.5 이상이면 매도 고려)
    SELL_DECISION_THRESHOLD = 0.5
    
    # =========================================================================
    # [3단계] 추세 분석 기반 전략 (학습/시그널보다 우선)
    # =========================================================================
    if trend_analysis and trend_analysis.confidence >= 0.5:
        trend_type = trend_analysis.trend_type.value
        reason = trend_analysis.reason
        
        # 횡보장(Sideways) 전략
        if trend_type == 'sideways':
            if trend_analysis.volatility < 0.015:
                return 'hold'
            if trend_analysis.should_sell_early and '고점' in reason:
                if '거래량' in reason or '돌파' in reason:
                    return 'hold'
                if profit_loss_pct >= 1.0:
                    print(f"📊 {coin}: 횡보 고점 익절 (학습+시그널: {combined_sell_score:.2f})")
                    return 'sell'
            if trend_analysis.should_hold_strong and '저점' in reason:
                return 'hold'

        # 상승 추세 - 학습 데이터가 매도 권장해도 추세 추종
        if trend_type in ['strong_up', 'up'] and profit_loss_pct > 0:
            # 학습 점수가 매우 높지 않으면(0.7 미만) 상승 추세 유지
            if combined_sell_score < 0.7:
                return 'hold'

    # =========================================================================
    # 🆕 [4단계] 학습 기반 매도 판단 (핵심)
    # =========================================================================
    if combined_sell_score >= SELL_DECISION_THRESHOLD:
        # 틱 노이즈 필터 적용
        if not is_significant_move(profit_loss_pct):
            print(f"🛡️ {coin}: 학습 매도 신호({combined_sell_score:.2f})지만 호가 노이즈로 보류")
            return 'hold'
        
        # 수익 중 매도 (익절)
        if profit_loss_pct > 0:
            print(f"📈 {coin}: 학습 기반 익절 (점수: {combined_sell_score:.2f} = 학습 {learning_sell_score:.2f}×{LEARNING_WEIGHT:.0%} + 시그널 {signal_sell_score:.2f}×{SIGNAL_WEIGHT:.0%})")
            print(f"   └ {learning_reason}")
            return 'take_profit'
        
        # 손실 중 매도 (손절)
        else:
            # 🆕 손절은 더 신중하게 (학습 점수가 0.6 이상일 때만)
            if combined_sell_score >= 0.6:
                print(f"📉 {coin}: 학습 기반 손절 (점수: {combined_sell_score:.2f} = 학습 {learning_sell_score:.2f}×{LEARNING_WEIGHT:.0%} + 시그널 {signal_sell_score:.2f}×{SIGNAL_WEIGHT:.0%})")
                print(f"   └ {learning_reason}")
                return 'stop_loss'
    
    # =========================================================================
    # [5단계] 시간 기반 청산 (학습 데이터로 커버 안 되는 경우)
    # =========================================================================
    if holding_hours >= 36.0 and profit_loss_pct >= 0.0:
        print(f"⏳ {coin}: 36시간 이상 보유 (좀비 포지션) - 본전 이상 탈출")
        return 'sell'
    elif holding_hours >= 24.0 and profit_loss_pct >= 3.0:
        print(f"⏳ {coin}: 24시간 이상 보유 - 3% 익절")
        return 'take_profit'
    elif holding_hours >= 12.0 and profit_loss_pct >= 5.0:
        print(f"⏳ {coin}: 12시간 이상 보유 - 5% 익절")
        return 'take_profit'

    # =========================================================================
    # [6단계] 수익 반납 보호 (Trailing Stop)
    # =========================================================================
    if max_profit_pct >= 3.0 and profit_loss_pct < (max_profit_pct * 0.5):
        if is_significant_move(max_profit_pct - profit_loss_pct):
            print(f"📉 {coin}: 수익 반납 임계점 (최고 {max_profit_pct:.1f}% → 현재 {profit_loss_pct:.1f}%)")
            return 'sell'

    # =========================================================================
    # [7단계] 매수 판단 (동적 가중치 적용)
    # =========================================================================
    signal_buy_weight, learning_buy_weight, _ = get_dynamic_weights(for_buy=True)
    
    # 기본 매수 임계값
    buy_threshold = 0.4
    if signal_continuity > 0.7 and dynamic_influence > 0.6:
        buy_threshold = 0.35
    elif signal_continuity < 0.3:
        buy_threshold = 0.50
    
    # 🆕 학습 데이터 기반 매수 조정 (성숙 단계에서만)
    if learning_buy_weight >= 0.40:  # 학습 비중이 40% 이상일 때만 적용
        # 학습된 패턴 성과가 나쁘면 매수 기준 상향
        if pattern_confidence >= 0.3:
            optimal_tp = learned_thresholds.get('optimal_take_profit', 5.0)
            optimal_sl = learned_thresholds.get('optimal_stop_loss', -5.0)
            
            # 수익비가 나쁜 패턴이면 (익절가 < 손절폭) 더 엄격한 매수 기준
            if optimal_tp < abs(optimal_sl) * 0.8:
                adjustment = learning_buy_weight * 0.1  # 최대 7% 상향
                buy_threshold = min(0.55, buy_threshold + adjustment)
                print(f"   📚 {coin}: 학습 기반 매수 기준 상향 ({buy_threshold:.2f}, 패턴 수익비 낮음)")
    
    if signal_score > buy_threshold: 
        return 'buy'
    
    return 'hold'
