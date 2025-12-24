#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
가상매매 학습기 (순수 피드백 제공자)
RL 학습 부분 제거, 성과 데이터 수집 및 피드백 제공만 담당
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import time
import threading
from collections import defaultdict
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 🆕 변동성 시스템 import
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rl_pipeline'))
    from utils.coin_volatility import get_volatility_profile
    VOLATILITY_SYSTEM_AVAILABLE = True
except ImportError:
    VOLATILITY_SYSTEM_AVAILABLE = False
    print("⚠️ 변동성 시스템을 사용할 수 없습니다")

# 데이터베이스 경로
# 기존: ../data_storage -> 변경: market/coin_market/data_storage
_DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'market', 'coin_market', 'data_storage')
try:
    os.makedirs(_DEFAULT_DB_DIR, exist_ok=True)
except OSError:
    pass

DB_PATH = os.getenv('RL_DB_PATH', os.path.join(_DEFAULT_DB_DIR, 'trade_candles.db'))
# 🆕 DB 경로 분리 (Strategy vs Trading) + 디렉토리 모드 지원
# 전략 DB (Brain): signal_feedback_scores, evolution_results 등
# 🔧 환경변수가 디렉토리면 common_strategies.db 사용
# 🆕 [Fix] 환경 변수 경로가 Windows에서 Docker 경로(/workspace/...)로 인식될 경우 로컬 경로로 강제 변환
_env_strategy_base = os.getenv('STRATEGY_DB_PATH')
_default_strategy_base = os.path.join(os.getenv('DATA_STORAGE_PATH', _DEFAULT_DB_DIR), 'learning_strategies')

if _env_strategy_base and (_env_strategy_base.startswith('/workspace') or _env_strategy_base.startswith('\\workspace')):
    if os.name == 'nt':
         _strategy_base = _default_strategy_base
    else:
         _strategy_base = _env_strategy_base
else:
    _strategy_base = _env_strategy_base or _default_strategy_base

# print(f"🔧 [VirtualLearner] 전략 DB 베이스 경로 확인: {_strategy_base}")

if os.path.isdir(_strategy_base) or not _strategy_base.endswith('.db'):
    STRATEGY_DB_PATH = os.path.join(_strategy_base, 'common_strategies.db')
    # print(f"   -> 디렉토리 모드 감지 (또는 확장자 없음). DB 파일: {STRATEGY_DB_PATH}")
else:
    STRATEGY_DB_PATH = _strategy_base
    # print(f"   -> 단일 파일 모드. DB 파일: {STRATEGY_DB_PATH}")

# 🆕 DB 디렉토리 자동 생성 (에러 방지 및 로그 강화)
try:
    db_dir = os.path.dirname(STRATEGY_DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    # print(f"   ✅ DB 디렉토리 확인/생성 완료: {db_dir}")
except OSError as e:
    print(f"   ❌ DB 디렉토리 생성 실패: {e}")
    # 권한 문제 등으로 실패해도 일단 진행 (치명적 에러는 나중에 connect에서 발생)
# 매매 DB (Records): completed_trades, virtual_trade_feedback 등
TRADING_DB_PATH = os.getenv('TRADING_DB_PATH', os.path.join(os.getenv('DATA_STORAGE_PATH', _DEFAULT_DB_DIR), 'trading_system.db'))

# 호환성을 위해 기본 변수는 매매 DB를 가리키도록 설정 (하지만 개별 함수에서 적절한 DB를 사용해야 함)
TRADING_SYSTEM_DB_PATH = TRADING_DB_PATH

def get_db_path_for_table(table_name: str) -> str:
    """테이블 이름에 따라 적절한 DB 경로 반환"""
    # ⚠️ 피드백/시그널 테이블은 트레이딩 DB에 저장
    strategy_tables = ['evolution_results', 'learning_checkpoint', 'multi_timeframe_analysis']
    trading_tables = ['signal_feedback_scores', 'signals']

    if table_name in trading_tables:
        return TRADING_SYSTEM_DB_PATH
    if table_name in strategy_tables:
        return STRATEGY_DB_PATH
    return TRADING_DB_PATH

# 안전한 타입 변환 함수들
def safe_float(value, default=0.0):
    """안전한 float 변환"""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_str(value, default='unknown'):
    """안전한 string 변환"""
    try:
        if value is None or pd.isna(value):
            return default
        return str(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """안전한 int 변환"""
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

# 시그널 액션 열거형
class SignalAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"

# 시그널 정보 데이터클래스
@dataclass
class SignalInfo:
    coin: str  # symbol -> coin
    interval: str
    action: SignalAction
    signal_score: float
    confidence: float
    reason: str
    timestamp: int
    price: float
    volume: float
    rsi: float
    macd: float
    wave_phase: str
    pattern_type: str
    risk_level: str
    volatility: float
    volume_ratio: float
    wave_progress: float
    structure_score: float
    pattern_confidence: float
    integrated_direction: str
    integrated_strength: float
    # 🆕 Absolute Zero System의 새로운 고급 지표들
    mfi: float = 50.0
    atr: float = 0.0
    adx: float = 25.0
    ma20: float = 0.0
    rsi_ema: float = 50.0
    macd_smoothed: float = 0.0
    wave_momentum: float = 0.0
    bb_position: str = 'unknown'
    bb_width: float = 0.0
    bb_squeeze: float = 0.0
    rsi_divergence: str = 'none'
    macd_divergence: str = 'none'
    volume_divergence: str = 'none'
    price_momentum: float = 0.0
    volume_momentum: float = 0.0
    trend_strength: float = 0.5
    support_resistance: str = 'unknown'
    fibonacci_levels: str = 'unknown'
    elliott_wave: str = 'unknown'
    harmonic_patterns: str = 'none'
    candlestick_patterns: str = 'none'
    market_structure: str = 'unknown'
    flow_level_meta: str = 'unknown'
    pattern_direction: str = 'neutral'
    target_price: float = 0.0  # 🆕 예상 목표가 (AI/기술적 분석 기반)
    source_type: str = 'quant' # 🆕 시그널 출처 (quant, ai, hybrid)

# 가상 포지션 정보 데이터클래스
@dataclass
class VirtualPosition:
    """가상 포지션 정보"""
    coin: str
    entry_price: float
    quantity: float
    entry_timestamp: int
    entry_signal_score: float
    current_price: float
    profit_loss_pct: float
    holding_duration: int
    max_profit_pct: float
    max_loss_pct: float
    stop_loss_price: float
    take_profit_price: float
    last_updated: int

# 🆕 성능 업그레이드 시스템 클래스들
class PostTradeEvaluator:
    """매매 사후 평가기 - 매도/손절 후 가격 흐름을 추적하여 판단의 질 평가"""
    def __init__(self):
        self.tracked_trades = {}  # trade_id: {action, exit_price, ...}
        self.tracking_duration = 24 * 3600  # 24시간 동안 추적
        
        # 🆕 [성능 최적화] 배치 처리를 위한 버퍼
        self.pending_penalties = [] # (signal_pattern, penalty_type, severity)
    
    def add_trade(self, trade_data: dict):
        """추적 대상 거래 추가"""
        try:
            action = trade_data.get('action')
            # 매도, 손절, 익절 거래를 모두 추적
            if action in ['sell', 'stop_loss', 'take_profit']:
                trade_id = f"{trade_data['coin']}_{trade_data['entry_timestamp']}"
                # 🆕 초기 목표가 우선 사용 (없으면 최종 목표가 사용)
                initial_target = trade_data.get('initial_target_price', 0)
                final_target = trade_data.get('target_price', 0)
                target_price = initial_target if initial_target > 0 else final_target
                
                self.tracked_trades[trade_id] = {
                    'coin': trade_data['coin'],
                    'action': action,
                    'exit_price': trade_data.get('exit_price', 0),
                    'target_price': target_price, # 평가 기준 목표가
                    'initial_target_price': initial_target, # 기록용
                    'final_target_price': final_target, # 기록용
                    'exit_timestamp': trade_data['exit_timestamp'],
                    'signal_pattern': trade_data.get('signal_pattern', 'unknown'),
                    'lowest_price_after_exit': trade_data.get('exit_price', 0),  # 추적 기간 중 최저가
                    'highest_price_after_exit': trade_data.get('exit_price', 0), # 추적 기간 중 최고가
                    'status': 'tracking'
                }
                print(f"👀 사후 평가 시작: {trade_data['coin']} ({action} @ {trade_data.get('exit_price', 0)})")
        except Exception as e:
            print(f"⚠️ 추적 추가 오류: {e}")

    def check_evaluations(self, current_prices: dict):
        """현재가와 비교하여 매매 판단 평가"""
        try:
            current_time = int(time.time())
            completed_tracks = []
            
            for trade_id, data in self.tracked_trades.items():
                # 시간 만료 체크 (평가 종료)
                if current_time - data['exit_timestamp'] > self.tracking_duration:
                    self._finalize_evaluation(data)
                    completed_tracks.append(trade_id)
                    continue
                
                coin = data['coin']
                if coin in current_prices:
                    current_price = current_prices[coin]
                    
                    # 최저가/최고가 업데이트
                    if current_price < data['lowest_price_after_exit']:
                        data['lowest_price_after_exit'] = current_price
                    if current_price > data['highest_price_after_exit']:
                        data['highest_price_after_exit'] = current_price
                    
                    # 🚀 즉각적인 평가 (극단적인 경우 바로 피드백)
                    self._evaluate_immediate_reaction(data, current_price)
            
            # 완료된 추적 제거
            for trade_id in completed_tracks:
                del self.tracked_trades[trade_id]
                
        except Exception as e:
            print(f"⚠️ 평가 확인 오류: {e}")

    def _evaluate_immediate_reaction(self, data: dict, current_price: float):
        """즉각적인 시장 반응 평가 (심각한 실수나 대박 감지)"""
        exit_price = data['exit_price']
        action = data['action']
        signal_pattern = data.get('signal_pattern', 'unknown')
        
        # 1. 손절 후 급반등 감지 (Panic Sell)
        if action == 'stop_loss':
            rebound_pct = ((current_price - exit_price) / exit_price) * 100
            if rebound_pct >= 5.0:  # 손절하자마자 5% 이상 반등
                print(f"😱 패닉 셀 감지! {data['coin']}: 손절 후 {rebound_pct:.2f}% 급반등 (패널티 강화)")
                # 해당 패턴에 대한 즉각적인 패널티 부여
                self._apply_pattern_penalty(signal_pattern, penalty_type='panic_sell', severity=rebound_pct)

        # 2. 익절 후 추가 폭등 감지 (Too Early Exit)
        elif action in ['sell', 'take_profit']:
            missed_pct = ((current_price - exit_price) / exit_price) * 100
            target_price = data.get('target_price', 0)
            
            # 목표가 도달했거나 10% 이상 추가 상승 시
            if (target_price > 0 and current_price >= target_price) or missed_pct >= 10.0:
                print(f"😅 조기 매도 감지! {data['coin']}: 매도 후 {missed_pct:.2f}% 추가 상승 (기회 비용)")
                # 해당 패턴의 '참을성' 가중치 증가 피드백
                self._apply_pattern_penalty(signal_pattern, penalty_type='early_exit', severity=missed_pct)

    def evaluate_profit_retracement(self, trade_data: dict):
        """🆕 수익 반납(Profit Retracement) 평가 - 익절 기회 놓침 학습"""
        try:
            max_profit = trade_data.get('max_profit_pct', 0.0)
            final_profit = trade_data.get('profit_loss_pct', 0.0)
            signal_pattern = trade_data.get('signal_pattern', 'unknown')
            
            # 5% 이상 수익이 났었는데, 최종적으로 1% 미만으로 마감한 경우
            # (욕심 부리다 익절 타이밍 놓침)
            if max_profit >= 5.0 and final_profit < 1.0:
                retracement = max_profit - final_profit
                print(f"📉 수익 반납 감지! {trade_data['coin']}: 최고 {max_profit:.1f}% -> 마감 {final_profit:.1f}% (놓친 수익 {retracement:.1f}%)")
                
                # '적당히 먹고 나오기' 학습을 위해 패널티 부여
                # 패널티 타입: missed_opportunity
                self._apply_pattern_penalty(signal_pattern, penalty_type='missed_opportunity', severity=retracement)
                
        except Exception as e:
            print(f"⚠️ 수익 반납 평가 오류: {e}")

    def evaluate_bull_trap(self, trade_data: dict):
        """🆕 설거지(Bull Trap) 평가 - 진입 타점 실패 학습"""
        try:
            max_profit = trade_data.get('max_profit_pct', 0.0)
            final_profit = trade_data.get('profit_loss_pct', 0.0)
            signal_pattern = trade_data.get('signal_pattern', 'unknown')
            
            # 매수 후 한 번도 0.3% 이상 오르지 못하고, 결국 -3% 이상 손실 본 경우
            # (사자마자 물림 -> 명백한 진입 실패)
            if max_profit < 0.3 and final_profit <= -3.0:
                print(f"🪤 설거지(Bull Trap) 감지! {trade_data['coin']}: 최고 {max_profit:.1f}% -> 마감 {final_profit:.1f}% (진입 타점 실패)")
                
                # '잘못된 진입' 학습을 위해 강력한 패널티 부여
                self._apply_pattern_penalty(signal_pattern, penalty_type='entry_fail', severity=abs(final_profit))
                
        except Exception as e:
            print(f"⚠️ 설거지 평가 오류: {e}")

    def evaluate_time_efficiency(self, trade_data: dict):
        """🆕 시간 가성비 평가 - 기회비용 학습"""
        try:
            duration_hours = trade_data.get('holding_duration', 0) / 3600
            final_profit = trade_data.get('profit_loss_pct', 0.0)
            signal_pattern = trade_data.get('signal_pattern', 'unknown')
            
            # 48시간 이상 걸려서 2% 미만 수익 (너무 오래 걸린 짤짤이)
            # 승리로 기록되지만 가중치를 낮춰야 함
            if duration_hours >= 48 and 0 < final_profit < 2.0:
                print(f"🐌 가성비 꽝! {trade_data['coin']}: {duration_hours:.1f}시간 동안 {final_profit:.1f}% 수익 (자금 회전율 저하)")
                
                # 성공했지만 점수를 깎아서 더 빠른 패턴을 선호하게 유도
                self._apply_pattern_penalty(signal_pattern, penalty_type='low_efficiency', severity=duration_hours)
                
        except Exception as e:
            print(f"⚠️ 시간 가성비 평가 오류: {e}")

    def _finalize_evaluation(self, data: dict):
        """추적 기간 종료 후 최종 평가"""
        exit_price = data['exit_price']
        lowest_price = data['lowest_price_after_exit']
        action = data['action']
        
        # 1. 신의 손절 (Smart Cut) 평가
        # 손절했는데 그 뒤로 가격이 더 많이 빠졌다면? -> 아주 잘한 행동!
        if action == 'stop_loss':
            max_drop_pct = ((lowest_price - exit_price) / exit_price) * 100
            if max_drop_pct <= -10.0:  # 팔고 나서 10% 이상 더 빠짐
                print(f"🛡️ 신의 손절! {data['coin']}: 손절 후 {max_drop_pct:.2f}% 추가 폭락 (대형 손실 방어)")
                # 손실 거래였지만, '성공적인 방어'로 기록하여 점수 보정 (+1.0 보상)
                self._record_smart_cut_feedback(data)

        # 2. 신의 익절 (Perfect Exit) 평가
        # 익절했는데 그 뒤로 가격이 빠졌다면? -> 고점 매도 성공!
        elif action in ['sell', 'take_profit']:
            max_drop_pct = ((lowest_price - exit_price) / exit_price) * 100
            if max_drop_pct <= -5.0: # 팔고 나서 5% 이상 빠짐
                print(f"🌟 신의 한 수! {data['coin']}: 매도 후 {max_drop_pct:.2f}% 하락 (고점 매도 성공)")
                # 익절 점수에 추가 보너스 부여 (+3.0 보상)
                self._record_perfect_exit_feedback(data)

        # 🆕 [Adaptive Exit] 최적 청산 파라미터 업데이트
        self._update_optimal_exit_params(data)

    def _update_optimal_exit_params(self, data: dict):
        """사후 평가 결과를 바탕으로 최적 TP/SL 비율 업데이트"""
        try:
            signal_pattern = data.get('signal_pattern', 'unknown')
            if signal_pattern == 'unknown': return

            exit_price = data['exit_price']
            highest_price = data['highest_price_after_exit']
            lowest_price = data['lowest_price_after_exit']
            entry_price = data.get('entry_price', 0)
            
            if entry_price <= 0: return

            # MFE (진입가 대비 최고가 수익률)
            mfe_pct = ((highest_price - entry_price) / entry_price) * 100
            # MAE (진입가 대비 최저가 수익률 - 손절 라인 체크용)
            mae_pct = ((lowest_price - entry_price) / entry_price) * 100
            
            # 실제 실현 수익률
            realized_profit_pct = data['profit_loss_pct']

            with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # 기존 파라미터 조회
                cursor.execute("SELECT optimal_tp_ratio, samples FROM pattern_exit_params WHERE signal_pattern = ?", (signal_pattern,))
                row = cursor.fetchone()
                
                current_tp_ratio = 2.0
                samples = 0
                
                if row:
                    current_tp_ratio, samples = row
                
                # 🆕 TP 조정 로직 (Adaptive TP)
                # 팔고 나서 더 올랐다면 (놓친 수익이 큼) -> TP 상향
                # MFE가 실현 수익의 1.5배 이상이었다면 더 버텼어야 함
                missed_profit = mfe_pct - realized_profit_pct
                
                new_tp_ratio = current_tp_ratio
                
                if missed_profit > 5.0: # 5% 이상 더 갈 수 있었음
                    # 과감하게 상향 (0.1 ~ 0.5)
                    adjustment = min(missed_profit / 20.0, 0.5)
                    new_tp_ratio += adjustment
                    print(f"📈 [TP 학습] {signal_pattern}: 너무 일찍 매도 (놓친 수익 {missed_profit:.1f}%) -> TP 비율 상향 ({current_tp_ratio:.2f} -> {new_tp_ratio:.2f})")
                elif missed_profit < 1.0 and realized_profit_pct > 0:
                    # 거의 고점에서 팔았음 -> 유지하거나 미세하게 하향 (안전빵)
                    pass
                
                # 이동평균 업데이트
                samples += 1
                updated_tp = (current_tp_ratio * (samples - 1) + new_tp_ratio) / samples
                
                cursor.execute("""
                    INSERT INTO pattern_exit_params (signal_pattern, optimal_tp_ratio, samples, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(signal_pattern) DO UPDATE SET
                    optimal_tp_ratio = excluded.optimal_tp_ratio,
                    samples = excluded.samples,
                    updated_at = excluded.updated_at
                """, (signal_pattern, updated_tp, samples, int(time.time())))
                
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 최적 청산 파라미터 업데이트 오류: {e}")

    def _apply_pattern_penalty(self, signal_pattern: str, penalty_type: str, severity: float):
        """패턴에 대한 페널티/보상 적용 (배치 큐에 추가)"""
        # 🚀 [성능 최적화] 즉시 DB 업데이트 대신 큐에 추가
        self.pending_penalties.append((signal_pattern, penalty_type, severity))

    def flush_penalties(self):
        """🚀 [성능 최적화] 큐에 쌓인 패널티 일괄 DB 업데이트"""
        if not self.pending_penalties:
            return
            
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=60.0) as conn:
                # 테이블이 없을 수 있으므로 보정
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_pattern TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor = conn.cursor()
                count = 0
                
                # 중복 패턴 합치기 (최적화)
                updates = {}
                for pattern, p_type, sev in self.pending_penalties:
                    key = (pattern, p_type)
                    if key not in updates:
                        updates[key] = []
                    updates[key].append(sev)
                
                for (signal_pattern, penalty_type), severities in updates.items():
                    # 평균 severity 사용
                    avg_severity = sum(severities) / len(severities)
                    
                    # 기존 패턴 조회
                    cursor.execute("""
                        SELECT success_rate, avg_profit, total_trades, confidence
                        FROM signal_feedback_scores WHERE signal_pattern = ?
                    """, (signal_pattern,))
                    
                    result = cursor.fetchone()
                    if not result:
                        # 신규 패턴 삽입
                        base_success = max(0.0, 0.5 - avg_severity/200)  # 0~0.5 사이 초기화
                        cursor.execute("""
                            INSERT INTO signal_feedback_scores (signal_pattern, success_rate, avg_profit, total_trades, confidence, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (signal_pattern, base_success, 0.0, 1, 0.5, int(time.time()), int(time.time())))
                        print(f"🆕 패턴 초기화: {signal_pattern} (success_rate={base_success:.2f})")
                        count += 1
                        continue

                    success_rate, avg_profit, total_trades, confidence = result
                    new_success_rate = success_rate
                    
                    # 페널티 타입에 따른 조정 (로직 동일)
                    if penalty_type == 'panic_sell':
                        adjustment = min(avg_severity / 100, 0.1)
                        new_success_rate = max(0, success_rate - adjustment)
                        print(f"📉 패닉셀 페널티 (Batch): {signal_pattern} 승률 {success_rate:.2f} → {new_success_rate:.2f}")
                    elif penalty_type == 'early_exit':
                        # 조기 매도는 승률 유지, 로그만 출력 (복잡성 감소)
                        print(f"📊 조기매도 피드백 (Batch): {signal_pattern} 놓친 수익 {avg_severity:.2f}%")
                    elif penalty_type == 'stagnant':
                        adjustment = 0.1
                        new_success_rate = max(0, success_rate - adjustment)
                        print(f"🐌 침체구간 페널티 (Batch): {signal_pattern} 승률 {success_rate:.2f} → {new_success_rate:.2f}")
                    elif penalty_type == 'missed_opportunity':
                        adjustment = 0.05
                        new_success_rate = max(0, success_rate - adjustment)
                        print(f"💸 수익반납 페널티 (Batch): {signal_pattern} 승률 {success_rate:.2f} → {new_success_rate:.2f}")
                    elif penalty_type == 'entry_fail':
                        adjustment = 0.15
                        new_success_rate = max(0, success_rate - adjustment)
                        print(f"🪤 진입실패 페널티 (Batch): {signal_pattern} 승률 {success_rate:.2f} → {new_success_rate:.2f}")
                    elif penalty_type == 'low_efficiency':
                        adjustment = 0.03
                        new_success_rate = max(0, success_rate - adjustment)
                        print(f"🐌 가성비저하 페널티 (Batch): {signal_pattern} 승률 {success_rate:.2f} → {new_success_rate:.2f}")

                    if new_success_rate != success_rate:
                        cursor.execute("""
                            UPDATE signal_feedback_scores
                            SET success_rate = ?, updated_at = ?
                            WHERE signal_pattern = ?
                        """, (new_success_rate, int(time.time()), signal_pattern))
                        count += 1
                
                conn.commit()
                if count > 0:
                    print(f"✅ {count}건의 패턴 페널티 일괄 업데이트 완료")
                
                # 큐 비우기
                self.pending_penalties = []
                
        except Exception as e:
            print(f"⚠️ 패턴 페널티 일괄 적용 오류: {e}")

    def _record_smart_cut_feedback(self, data: dict):
        """손실 방어 성공 피드백 기록 - 손절 후 가격이 더 하락한 경우"""
        try:
            signal_pattern = data.get('signal_pattern', 'unknown')
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=60.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_pattern TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor = conn.cursor()
                
                # 기존 패턴 조회
                cursor.execute("""
                    SELECT success_rate, avg_profit, total_trades, confidence
                    FROM signal_feedback_scores WHERE signal_pattern = ?
                """, (signal_pattern,))
                
                result = cursor.fetchone()
                
                if result:
                    success_rate, avg_profit, total_trades, confidence = result
                    
                    # 손실 방어 성공 보상: 승률과 신뢰도 증가
                    new_success_rate = min(1.0, success_rate + 0.05)  # 5% 승률 증가
                    new_confidence = min(1.0, confidence + 0.1)  # 10% 신뢰도 증가
                    
                    cursor.execute("""
                        UPDATE signal_feedback_scores
                        SET success_rate = ?, confidence = ?, updated_at = ?
                        WHERE signal_pattern = ?
                    """, (new_success_rate, new_confidence, int(time.time()), signal_pattern))
                    conn.commit()
                    
                    print(f"🛡️ 손실 방어 성공 기록: {signal_pattern} (승률 +5%, 신뢰도 +10%)")
                else:
                    # 신규 패턴 삽입 (기본값)
                    cursor.execute("""
                        INSERT INTO signal_feedback_scores (signal_pattern, success_rate, avg_profit, total_trades, confidence, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (signal_pattern, 0.55, 0.0, 1, 0.6, int(time.time()), int(time.time())))
                    conn.commit()
                    print(f"🆕 손실 방어 패턴 신규 등록: {signal_pattern}")
                    
        except Exception as e:
            print(f"⚠️ 손실 방어 피드백 기록 오류: {e}")

    def _record_perfect_exit_feedback(self, data: dict):
        """고점 매도 성공 피드백 기록 - 매도 후 가격이 하락한 경우"""
        try:
            signal_pattern = data.get('signal_pattern', 'unknown')
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=60.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_pattern TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor = conn.cursor()
                
                # 기존 패턴 조회
                cursor.execute("""
                    SELECT success_rate, avg_profit, total_trades, confidence
                    FROM signal_feedback_scores WHERE signal_pattern = ?
                """, (signal_pattern,))
                
                result = cursor.fetchone()
                
                if result:
                    success_rate, avg_profit, total_trades, confidence = result
                    
                    # 고점 매도 성공 보상: 승률과 평균 수익률 증가
                    new_success_rate = min(1.0, success_rate + 0.08)  # 8% 승률 증가
                    new_avg_profit = avg_profit + 1.0  # 평균 수익률 +1%
                    new_confidence = min(1.0, confidence + 0.15)  # 15% 신뢰도 증가
                    
                    cursor.execute("""
                        UPDATE signal_feedback_scores
                        SET success_rate = ?, avg_profit = ?, confidence = ?, updated_at = ?
                        WHERE signal_pattern = ?
                    """, (new_success_rate, new_avg_profit, new_confidence, int(time.time()), signal_pattern))
                    conn.commit()
                    
                    print(f"🌟 고점 매도 성공 기록: {signal_pattern} (승률 +8%, 수익률 +1%, 신뢰도 +15%)")
                else:
                    cursor.execute("""
                        INSERT INTO signal_feedback_scores (signal_pattern, success_rate, avg_profit, total_trades, confidence, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (signal_pattern, 0.60, 1.0, 1, 0.65, int(time.time()), int(time.time())))
                    conn.commit()
                    print(f"🆕 고점 매도 패턴 신규 등록: {signal_pattern}")
                    
        except Exception as e:
            print(f"⚠️ 고점 매도 피드백 기록 오류: {e}")

class ExponentialDecayWeight:
    """최근성 가중치 계산기"""
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
    
    def calculate_weight(self, time_diff_hours: float) -> float:
        """시간 차이에 따른 가중치 계산"""
        import math
        return math.exp(-self.decay_rate * time_diff_hours)

class BayesianSmoothing:
    """베이지안 스무딩 시스템"""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, kappa: float = 1.0):
        self.alpha = alpha  # Beta 분포 파라미터
        self.beta = beta    # Beta 분포 파라미터
        self.kappa = kappa  # 정규 분포 파라미터
    
    def smooth_success_rate(self, wins: int, total_trades: int) -> float:
        """승률 베이지안 스무딩"""
        return (wins + self.alpha) / (total_trades + self.alpha + self.beta)
    
    def smooth_avg_profit(self, profits: List[float], global_avg: float) -> float:
        """평균 수익률 베이지안 스무딩"""
        if not profits:
            return global_avg
        
        weighted_sum = sum(profits) + self.kappa * global_avg
        total_weight = len(profits) + self.kappa
        
        return weighted_sum / total_weight

class OutlierGuardrail:
    """이상치 컷 시스템"""
    def __init__(self, percentile_cut: float = 0.05):
        self.percentile_cut = percentile_cut
    
    def winsorize_profits(self, profits: List[float]) -> List[float]:
        """수익률 Winsorizing"""
        if len(profits) < 10:  # 데이터가 적으면 그대로 반환
            return profits
        
        sorted_profits = sorted(profits)
        n = len(sorted_profits)
        
        # 상하위 5% 절단
        lower_cut = int(n * self.percentile_cut)
        upper_cut = int(n * (1 - self.percentile_cut))
        
        # 절단된 값으로 대체
        winsorized = []
        for profit in profits:
            if profit < sorted_profits[lower_cut]:
                winsorized.append(sorted_profits[lower_cut])
            elif profit > sorted_profits[upper_cut]:
                winsorized.append(sorted_profits[upper_cut])
            else:
                winsorized.append(profit)
        
        return winsorized
    
    def calculate_robust_avg_profit(self, profits: List[float]) -> float:
        """견고한 평균 수익률 계산"""
        winsorized_profits = self.winsorize_profits(profits)
        return sum(winsorized_profits) / len(winsorized_profits)

class RecencyWeightedAggregator:
    """최근성 가중치 집계기"""
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
        self.exponential_decay = ExponentialDecayWeight(decay_rate)
    
    def aggregate_with_recency_weights(self, trades: List[Dict]) -> Dict[str, float]:
        """최근성 가중치로 집계"""
        current_time = time.time()
        
        weighted_success_rate = 0.0
        weighted_avg_profit = 0.0
        total_weight = 0.0
        
        for trade in trades:
            time_diff_hours = (current_time - trade['timestamp']) / 3600
            weight = self.exponential_decay.calculate_weight(time_diff_hours)
            
            if trade['success']:
                weighted_success_rate += weight
            weighted_avg_profit += weight * trade['profit']
            total_weight += weight
        
        if total_weight == 0:
            return {'success_rate': 0.0, 'avg_profit': 0.0}
        
        return {
            'success_rate': weighted_success_rate / total_weight,
            'avg_profit': weighted_avg_profit / total_weight
        }

class BayesianSmoothingApplier:
    """베이지안 스무딩 적용기"""
    def __init__(self):
        self.bayesian_smoothing = BayesianSmoothing()
        self.global_stats = {'avg_success_rate': 0.5, 'avg_profit': 0.0}
    
    def apply_bayesian_smoothing(self, pattern_stats: Dict[str, float]) -> Dict[str, float]:
        """베이지안 스무딩 적용"""
        smoothed_stats = {}
        
        # 승률 스무딩
        if 'success_rate' in pattern_stats and 'total_trades' in pattern_stats:
            smoothed_stats['success_rate'] = self.bayesian_smoothing.smooth_success_rate(
                int(pattern_stats['success_rate'] * pattern_stats['total_trades']),
                int(pattern_stats['total_trades'])
            )
        
        # 평균 수익률 스무딩
        if 'avg_profit' in pattern_stats:
            smoothed_stats['avg_profit'] = self.bayesian_smoothing.smooth_avg_profit(
                [pattern_stats['avg_profit']], 
                self.global_stats['avg_profit']
            )
        
        return smoothed_stats

class OutlierGuardrailApplier:
    """이상치 컷 적용기"""
    def __init__(self):
        self.outlier_guardrail = OutlierGuardrail()
    
    def apply_outlier_guardrail(self, profits: List[float]) -> float:
        """이상치 컷 적용"""
        return self.outlier_guardrail.calculate_robust_avg_profit(profits)

# 🆕 [Confidence Calibration] 신뢰도 교정 시스템
class CalibrationTracker:
    """
    예측된 신뢰도와 실제 결과의 오차를 추적하여 '겸손함'을 학습하는 시스템
    - Brier Score 개념 활용
    - Binning 방식을 사용하여 신뢰도 구간별 정확도 측정
    """
    def __init__(self, db_path: str = None):
        self.db_path = db_path or STRATEGY_DB_PATH
        # 신뢰도 구간 (0.0~0.1, 0.1~0.2 ... 0.9~1.0)
        self.bins = {i: {'correct': 0, 'total': 0} for i in range(10)}
        self._load_calibration_data()

    def _load_calibration_data(self):
        """DB에서 교정 데이터 로드"""
        try:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS confidence_calibration (
                        bin_index INTEGER PRIMARY KEY,
                        correct_count INTEGER DEFAULT 0,
                        total_count INTEGER DEFAULT 0,
                        updated_at INTEGER
                    )
                """)
                conn.commit()
                
                cursor.execute("SELECT bin_index, correct_count, total_count FROM confidence_calibration")
                for row in cursor.fetchall():
                    idx, correct, total = row
                    self.bins[idx] = {'correct': correct, 'total': total}
        except Exception as e:
            print(f"⚠️ Calibration 데이터 로드 오류: {e}")

    def update(self, predicted_confidence: float, is_success: bool):
        """
        예측 결과 업데이트
        Args:
            predicted_confidence: AI가 예측한 신뢰도 (0.0 ~ 1.0)
            is_success: 실제 성공 여부 (수익 발생 여부)
        """
        try:
            # 0.05 -> 0번 bin, 0.95 -> 9번 bin
            bin_idx = min(int(predicted_confidence * 10), 9)
            
            self.bins[bin_idx]['total'] += 1
            if is_success:
                self.bins[bin_idx]['correct'] += 1
            
            # DB 업데이트
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute("""
                    INSERT INTO confidence_calibration (bin_index, correct_count, total_count, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(bin_index) DO UPDATE SET
                    correct_count = excluded.correct_count,
                    total_count = excluded.total_count,
                    updated_at = excluded.updated_at
                """, (bin_idx, self.bins[bin_idx]['correct'], self.bins[bin_idx]['total'], int(time.time())))
                conn.commit()
                
            # 로그 출력 (디버깅용)
            # accuracy = self.bins[bin_idx]['correct'] / self.bins[bin_idx]['total']
            # print(f"🔧 신뢰도 교정: 예측 {predicted_confidence:.2f} -> 구간[{bin_idx}] 실제 정확도 {accuracy:.2f}")
            
        except Exception as e:
            print(f"⚠️ Calibration 업데이트 오류: {e}")

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """
        원래 신뢰도를 교정된 신뢰도로 변환
        예: AI가 0.9라고 했지만, 과거 0.9 구간의 실제 성공률이 0.6이라면 -> 0.6 반환 (겸손해짐)
        데이터가 부족하면(total < 5) 원래 값과 가중 평균
        """
        bin_idx = min(int(raw_confidence * 10), 9)
        bin_data = self.bins[bin_idx]
        
        if bin_data['total'] < 5:
            # 데이터 부족 시: 원래 값 그대로 사용 (혹은 약간 보수적으로)
            return raw_confidence
        
        actual_accuracy = bin_data['correct'] / bin_data['total']
        
        # 급격한 변화 방지를 위해 가중 평균 (원래 값 30% + 실제 결과 70%)
        calibrated = (raw_confidence * 0.3) + (actual_accuracy * 0.7)
        
        # 하한선 설정 (너무 낮아지면 매매 아예 안 하므로 최소 0.2 등 설정 가능하나, 여기선 그대로)
        return calibrated


# 🆕 Thompson Sampling 기반 강화학습 시스템 (시간 감쇠 + 학습 단계 인지)
class ThompsonSamplingLearner:
    """
    Thompson Sampling 기반 액션 결정 시스템
    - 탐색(Exploration)과 활용(Exploitation)의 자연스러운 균형
    - Beta 분포를 사용하여 불확실성 고려
    - 🆕 시간 감쇠: 오래된 데이터의 영향력 자동 감소
    - 🆕 학습 단계 인지: 초기 탐색 기간에는 Thompson 영향력 감소
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or STRATEGY_DB_PATH
        self.pattern_distributions = {}  # {pattern: {'alpha': float, 'beta': float, 'avg_profit': float}}
        self.exploration_bonus = 0.15  # 🔧 새 패턴 탐색 보너스 (0.1 → 0.15 상향)
        self.min_samples_for_confidence = 5  # 신뢰할 수 있는 최소 샘플 수
        
        # 🆕 시간 감쇠 설정 (Cold Start Problem 해결)
        self.decay_rate = 0.98  # 하루마다 2% 감쇠 (0.98^30 ≈ 0.55, 한 달 후 55%)
        self.decay_period_hours = 24  # 24시간마다 감쇠 적용
        self.min_alpha_beta = 1.0  # alpha/beta 최소값 (균등 분포로 리셋되는 것 방지)
        self.max_sample_age_days = 30  # 30일 이상 된 데이터는 영향력 크게 감소
        
        # 🆕 학습 단계 인지
        self.exploration_phase_samples = 20  # 20회 미만이면 탐색 단계
        self.exploration_phase_weight = 0.3  # 탐색 단계에서 Thompson 영향력 (30%)
        
        # DB에서 기존 분포 로드 (시간 감쇠 적용)
        self._load_distributions_from_db()
        
        print(f"🎰 Thompson Sampling 학습 시스템 초기화 완료 (패턴 {len(self.pattern_distributions)}개 로드)")
        print(f"   ⏳ 시간 감쇠: {(1-self.decay_rate)*100:.1f}%/일, 탐색 단계: {self.exploration_phase_samples}회 미만")
    
    def _load_distributions_from_db(self):
        """DB에서 패턴별 분포 로드 (🆕 시간 감쇠 적용)"""
        try:
            # 🆕 [Fix] 로드 시에는 STRATEGY_DB_PATH 대신, 명시적인 로컬 경로를 우선 시도
            # (Thompson Sampling은 보통 common_strategies.db에 저장됨)
            target_db_path = self.db_path
            
            # self.db_path가 디렉토리이거나 존재하지 않으면, 기본 common_strategies.db 경로 시도
            if not target_db_path or os.path.isdir(target_db_path):
                 # trade_executor와 같은 방식으로 경로 추론
                 _current_dir = os.path.dirname(os.path.abspath(__file__))
                 _root_dir = os.path.dirname(_current_dir)
                 target_db_path = os.path.join(_root_dir, 'market', 'coin_market', 'data_storage', 'learning_strategies', 'common_strategies.db')
            
            # 연결 시도
            with sqlite3.connect(target_db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 확인 및 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS thompson_sampling_distributions (
                        signal_pattern TEXT PRIMARY KEY,
                        alpha REAL NOT NULL DEFAULT 1.0,
                        beta REAL NOT NULL DEFAULT 1.0,
                        avg_profit REAL DEFAULT 0.0,
                        total_samples INTEGER DEFAULT 0,
                        last_updated INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                
                # 기존 데이터 로드 (last_updated 포함)
                cursor.execute("""
                    SELECT signal_pattern, alpha, beta, avg_profit, total_samples, last_updated
                    FROM thompson_sampling_distributions
                """)
                
                current_time = int(time.time())
                decayed_count = 0
                
                for row in cursor.fetchall():
                    pattern, alpha, beta, avg_profit, total_samples, last_updated = row
                    
                    # 🆕 시간 감쇠 적용
                    if last_updated:
                        hours_since_update = (current_time - last_updated) / 3600
                        days_since_update = hours_since_update / 24
                        
                        # 오래된 데이터일수록 alpha/beta를 균등 분포(1,1)에 가깝게 감쇠
                        decay_factor = self.decay_rate ** days_since_update
                        
                        # alpha와 beta를 감쇠 (균등 분포 방향으로)
                        # 새 값 = 1 + (기존 값 - 1) * decay_factor
                        decayed_alpha = self.min_alpha_beta + (alpha - self.min_alpha_beta) * decay_factor
                        decayed_beta = self.min_alpha_beta + (beta - self.min_alpha_beta) * decay_factor
                        
                        # 최소값 보장
                        alpha = max(decayed_alpha, self.min_alpha_beta)
                        beta = max(decayed_beta, self.min_alpha_beta)
                        
                        if decay_factor < 0.95:  # 5% 이상 감쇠됨
                            decayed_count += 1
                    
                    self.pattern_distributions[pattern] = {
                        'alpha': alpha,
                        'beta': beta,
                        'avg_profit': avg_profit,
                        'total_samples': total_samples,
                        'last_updated': last_updated or current_time
                    }
                
                if decayed_count > 0:
                    print(f"   ⏳ {decayed_count}개 패턴에 시간 감쇠 적용됨 (오래된 데이터 영향력 감소)")
                    
        except Exception as e:
            print(f"⚠️ Thompson Sampling 분포 로드 오류: {e} (DB: {target_db_path})")
    
    def update_distribution(self, signal_pattern: str, success: bool, profit_pct: float, 
                           weight: float = 1.0):
        """
        거래 결과로 분포 업데이트 (핵심 학습 로직)
        - 성공 시: alpha += weight (성공 횟수 증가)
        - 실패 시: beta += weight (실패 횟수 증가)
        - 🆕 수익 질(Quality)에 따른 Reward 가중치 차등 적용 (gpt.md 피드백 반영)
        """
        try:
            current_time = int(time.time())
            
            if signal_pattern not in self.pattern_distributions:
                # 새 패턴: 사전 분포 Beta(1, 1) = 균등 분포
                self.pattern_distributions[signal_pattern] = {
                    'alpha': 1.0,
                    'beta': 1.0,
                    'avg_profit': 0.0,
                    'total_samples': 0,
                    'last_updated': current_time
                }
            
            dist = self.pattern_distributions[signal_pattern]
            
            # 🆕 업데이트 전 시간 감쇠 적용 (오래된 데이터 영향력 감소)
            if 'last_updated' in dist and dist['last_updated']:
                hours_since_update = (current_time - dist['last_updated']) / 3600
                days_since_update = hours_since_update / 24
                
                if days_since_update > 1:  # 하루 이상 지났으면 감쇠
                    decay_factor = self.decay_rate ** days_since_update
                    dist['alpha'] = self.min_alpha_beta + (dist['alpha'] - self.min_alpha_beta) * decay_factor
                    dist['beta'] = self.min_alpha_beta + (dist['beta'] - self.min_alpha_beta) * decay_factor
            
            # 🆕 [Reward Shaping] 수익의 질에 따른 가중치 조절
            # 목표: "찔끔 먹는 것"보다 "확실하게 먹는 것"을 선호하고, "크게 잃는 것"을 극도로 기피하도록 유도
            
            magnitude_bonus = 0.0
            if success:
                # 대승(5% 이상)에는 큰 가중치, 소승(1% 미만)에는 작은 가중치
                # 예: 10% 수익 -> 1.0 + min(2.0, 1.0) = 2.0배 반영
                magnitude_bonus = min(abs(profit_pct) / 5.0, 1.0) 
            else:
                # 대패(-5% 이하)에는 매우 큰 가중치 (뼈저리게 느끼도록)
                # 예: -10% 손실 -> 1.0 + min(4.0, 2.0) = 3.0배 반영
                magnitude_bonus = min(abs(profit_pct) / 5.0, 2.0)
                
                # 🆕 [Bull Trap Defense] 상승장 패턴이 실패하면 더 큰 페널티 (속임수 학습 강화)
                if "bullish" in signal_pattern and "high" in signal_pattern:
                    magnitude_bonus += 0.5  # 가중치 0.5 추가 (불트랩 경계)
                
            final_weight = weight * (1.0 + magnitude_bonus)
            
            # Beta 분포 업데이트
            if success:
                dist['alpha'] += final_weight
            else:
                dist['beta'] += final_weight
            
            # 평균 수익률 업데이트 (가중 이동 평균)
            dist['total_samples'] += 1
            n = dist['total_samples']
            # 최근 데이터에 더 높은 가중치 (지수 이동 평균)
            ema_weight = 0.2  # 새 데이터 20% 반영
            dist['avg_profit'] = dist['avg_profit'] * (1 - ema_weight) + profit_pct * ema_weight
            
            # 마지막 업데이트 시간 기록
            dist['last_updated'] = current_time
            
            # DB에 저장
            self._save_distribution_to_db(signal_pattern, dist)
            
            # 🆕 저품질 패턴 청소 (Garbage Collection) - DB 삭제 기능 추가
            # Unknown/None 패턴이면서, 표본이 충분한데 성과가 나쁘면 영구 삭제
            is_garbage_pattern = "unknown" in signal_pattern.lower() or "none" in signal_pattern.lower()
            if is_garbage_pattern and dist['total_samples'] > 30:
                win_rate = dist['alpha'] / (dist['alpha'] + dist['beta'])
                if win_rate < 0.35 or dist['avg_profit'] < -2.0: # 기준 완화 (확실한 쓰레기만)
                     print(f"🧹 쓰레기 패턴 영구 삭제: {signal_pattern} (승률 {win_rate:.2f}, 수익 {dist['avg_profit']:.2f}%)")
                     del self.pattern_distributions[signal_pattern]
                     
                     # DB에서도 삭제
                     with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                         conn.execute("DELETE FROM thompson_sampling_distributions WHERE signal_pattern = ?", (signal_pattern,))
                         conn.commit()
                     return
            
            # 학습 결과 출력
            expected_success_rate = dist['alpha'] / (dist['alpha'] + dist['beta'])
            weight_str = f" (가중치 {final_weight:.1f}x)" if final_weight != weight else ""
            print(f"🎰 Thompson 업데이트: {signal_pattern[:30]}... "
                  f"({'✅' if success else '❌'}){weight_str} → "
                  f"기대승률 {expected_success_rate:.1%}, 평균수익 {dist['avg_profit']:.2f}%")
            
        except Exception as e:
            print(f"⚠️ Thompson Sampling 분포 업데이트 오류: {e}")
    
    def _save_distribution_to_db(self, signal_pattern: str, dist: dict):
        """분포를 DB에 저장"""
        try:
            # 🆕 [Fix] 저장 시에도 명시적인 로컬 경로 사용
            target_db_path = self.db_path
            if not target_db_path or os.path.isdir(target_db_path):
                 _current_dir = os.path.dirname(os.path.abspath(__file__))
                 _root_dir = os.path.dirname(_current_dir)
                 target_db_path = os.path.join(_root_dir, 'market', 'coin_market', 'data_storage', 'learning_strategies', 'common_strategies.db')

            with sqlite3.connect(target_db_path, timeout=60.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO thompson_sampling_distributions
                    (signal_pattern, alpha, beta, avg_profit, total_samples, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    signal_pattern,
                    dist['alpha'],
                    dist['beta'],
                    dist['avg_profit'],
                    dist['total_samples'],
                    int(time.time())
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ Thompson Sampling 분포 저장 오류: {e} (DB: {target_db_path})")
    
    def sample_success_rate(self, signal_pattern: str) -> Tuple[float, str]:
        """
        Thompson Sampling: Beta 분포에서 승률 샘플링
        - 데이터 적으면 분산 높음 → 탐색 유도
        - 데이터 많으면 분산 낮음 → 활용 위주
        """
        if signal_pattern not in self.pattern_distributions:
            # 새 패턴: 긍정적 초기화 (Beta(2, 1) -> Mean 0.66) - 전략 신뢰
            sampled = np.random.beta(2, 1)
            return sampled, "🔍 새 패턴 탐색 (Optimistic)"
        
        dist = self.pattern_distributions[signal_pattern]
        
        # Beta 분포에서 샘플링 (핵심!)
        sampled = np.random.beta(dist['alpha'], dist['beta'])
        
        # 신뢰도 메시지
        total = dist['alpha'] + dist['beta'] - 2  # 사전 분포 제외
        if total < self.min_samples_for_confidence:
            confidence_msg = f"⚠️ 데이터 부족 ({int(total)}회)"
        else:
            expected = dist['alpha'] / (dist['alpha'] + dist['beta'])
            confidence_msg = f"📊 기대승률 {expected:.0%} ({int(total)}회)"
        
        return sampled, confidence_msg
    
    def should_execute_action(self, signal_pattern: str, signal_score: float, 
                              action_type: str = 'buy') -> Tuple[bool, float, str]:
        """
        Thompson Sampling 기반 액션 실행 여부 결정
        🆕 학습 단계 인지: 초기에는 Thompson 영향력 감소 (시그널 학습 중일 때 보호)
        
        Returns:
            (실행 여부, 최종 점수, 이유)
        """
        # 1. Thompson Sampling으로 승률 샘플링
        sampled_rate, sample_msg = self.sample_success_rate(signal_pattern)
        
        # 2. 평균 수익률 고려
        avg_profit = 0.0
        total_samples = 0
        if signal_pattern in self.pattern_distributions:
            avg_profit = self.pattern_distributions[signal_pattern].get('avg_profit', 0.0)
            total_samples = self.pattern_distributions[signal_pattern].get('total_samples', 0)
        
        # 🆕 3. 학습 단계 인지 - Thompson Sampling 영향력 동적 조정
        # 초기 탐색 단계: 시그널 점수를 더 신뢰 (시그널 학습이 진행 중)
        # 충분한 데이터: Thompson Sampling을 더 신뢰 (검증된 패턴)
        if total_samples < self.exploration_phase_samples:
            # 🔍 탐색 단계: 시그널 점수 비중 ↑, Thompson 비중 ↓
            # "시그널이 아직 학습 중이니까, Thompson의 부정적 판단을 덜 신뢰"
            signal_weight = 0.7  # 시그널 70%
            thompson_weight = 0.2  # Thompson 20%
            profit_weight = 0.1  # 수익률 10%
            phase_msg = f"🔍 탐색단계({total_samples}회)"
        else:
            # 📊 활용 단계: Thompson 비중 ↑
            signal_weight = 0.4  # 시그널 40%
            thompson_weight = 0.4  # Thompson 40%
            profit_weight = 0.2  # 수익률 20%
            phase_msg = f"📊 활용단계({total_samples}회)"
        
        # 4. 복합 점수 계산 (동적 가중치)
        profit_bonus = min(max(avg_profit / 5.0, -1.0), 1.0) * 0.5 + 0.5  # [0, 1] 범위
        
        # 🔧 시그널 점수 정규화 (-1~+1 → 0~1)
        # strategy_signal_generator에서 생성된 점수가 -1~+1 범위
        # Thompson Sampling에서는 0~1 범위로 변환하여 사용
        normalized_signal_score = (signal_score + 1.0) / 2.0  # -1→0, 0→0.5, +1→1
        
        final_score = (
            normalized_signal_score * signal_weight +
            sampled_rate * thompson_weight +
            profit_bonus * profit_weight
        )

        # 🆕 탐색 단계 보너스 추가 (전략 신뢰)
        if total_samples < self.exploration_phase_samples:
            final_score += self.exploration_bonus
        
        # 5. 임계값 기반 결정
        if action_type == 'buy':
            # 🆕🆕 탐색 단계에서는 임계값 대폭 완화 (가상매매 활성화)
            # 시그널 점수가 0.1~0.3 범위로 낮아서 기존 임계값으로는 진입이 어려움
            threshold = 0.40
            if total_samples < self.exploration_phase_samples:
                threshold = 0.30  # 🔧 0.40 → 0.30으로 낮춤 (탐색 촉진)
                
            if final_score >= threshold:
                reason = f"✅ 매수 실행 | 점수={final_score:.2f} ({phase_msg}, {sample_msg})"
                return True, final_score, reason
            else:
                reason = f"⏸️ 매수 보류 | 점수={final_score:.2f} ({phase_msg}, {sample_msg})"
                return False, final_score, reason
        
        elif action_type == 'sell':
            threshold = 0.4  # 매도 임계값 (더 낮음 - 손실 방지 우선)
            if final_score >= threshold:
                reason = f"✅ 매도 실행 | 점수={final_score:.2f} ({phase_msg})"
                return True, final_score, reason
            else:
                reason = f"⏸️ 매도 보류 | 점수={final_score:.2f} ({phase_msg})"
                return False, final_score, reason
        
        # 기본: 실행
        return True, final_score, f"기본 실행 | 점수={final_score:.2f}"
    
    def get_pattern_stats(self, signal_pattern: str) -> Optional[Dict]:
        """패턴 통계 조회"""
        if signal_pattern not in self.pattern_distributions:
            return None
        
        dist = self.pattern_distributions[signal_pattern]
        expected_rate = dist['alpha'] / (dist['alpha'] + dist['beta'])
        
        return {
            'expected_success_rate': expected_rate,
            'alpha': dist['alpha'],
            'beta': dist['beta'],
            'avg_profit': dist['avg_profit'],
            'total_samples': dist['total_samples'],
            'confidence': min(dist['total_samples'] / 20.0, 1.0)  # 20회 기준
        }
    
    def get_exploration_stats(self) -> Dict:
        """탐색/활용 통계 (🆕 학습 단계 정보 포함)"""
        total_patterns = len(self.pattern_distributions)
        total_samples = sum(d.get('total_samples', 0) for d in self.pattern_distributions.values())
        
        # 패턴별 학습 단계 분류
        exploration_patterns = sum(
            1 for d in self.pattern_distributions.values()
            if d.get('total_samples', 0) < self.exploration_phase_samples
        )
        exploitation_patterns = total_patterns - exploration_patterns
        
        confident_patterns = sum(
            1 for d in self.pattern_distributions.values()
            if d.get('total_samples', 0) >= self.min_samples_for_confidence
        )
        
        return {
            'total_patterns': total_patterns,
            'total_samples': total_samples,
            'confident_patterns': confident_patterns,
            'exploration_patterns': exploration_patterns,  # 🔍 탐색 단계 패턴 수
            'exploitation_patterns': exploitation_patterns,  # 📊 활용 단계 패턴 수
            'exploration_ratio': exploration_patterns / max(total_patterns, 1),
            'decay_rate': f"{(1-self.decay_rate)*100:.1f}%/일",  # 시간 감쇠율
            'exploration_threshold': self.exploration_phase_samples  # 탐색→활용 전환 기준
        }


# 🆕 진화형 AI 시스템 클래스들
class RealTimeLearner:
    """실시간 학습기 - 즉시 학습 및 적응"""
    def __init__(self):
        self.learning_rate = 0.01
        self.recent_trades = []
        self.pattern_performance = {}
        
    def learn_from_trade(self, signal_pattern: str, trade_result: dict):
        """거래 결과로부터 즉시 학습"""
        try:
            profit = trade_result.get('profit_loss_pct', 0.0)
            success = profit > 0
            
            # 패턴 성과 업데이트
            if signal_pattern not in self.pattern_performance:
                self.pattern_performance[signal_pattern] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'total_profit': 0.0,
                    'success_rate': 0.0
                }
            
            perf = self.pattern_performance[signal_pattern]
            perf['total_trades'] += 1
            perf['total_profit'] += profit
            
            if success:
                perf['successful_trades'] += 1
            
            perf['success_rate'] = perf['successful_trades'] / perf['total_trades']
            
            print(f"🧠 실시간 학습: {signal_pattern} 패턴 성과 업데이트 (성공률: {perf['success_rate']:.2f})")
            
        except Exception as e:
            print(f"⚠️ 실시간 학습 오류: {e}")

    def learn_from_ongoing_drawdown(self, signal_pattern: str, current_profit: float):
        """🚀 [Update] 실패를 통한 빠른 학습: 진행 중인 거래가 위험할 때 즉시 피드백"""
        try:
            # 손실이 -2%를 넘어가면 즉시 위험 신호 학습
            if current_profit < -2.0:
                print(f"🚨 실시간 위험 감지: {signal_pattern} 패턴이 {current_profit:.2f}% 손실 중! 즉시 피드백 반영")
                
                if signal_pattern not in self.pattern_performance:
                    self.pattern_performance[signal_pattern] = {
                        'total_trades': 0,
                        'successful_trades': 0,
                        'total_profit': 0.0,
                        'success_rate': 0.5,  # 초기값 50%
                        'risk_warnings': 0,  # 위험 경고 횟수 추적
                        'cumulative_drawdown': 0.0  # 누적 손실 추적
                    }
                
                perf = self.pattern_performance[signal_pattern]
                
                # 위험 경고 횟수 증가 (정수 유지)
                perf['risk_warnings'] = perf.get('risk_warnings', 0) + 1
                perf['cumulative_drawdown'] = perf.get('cumulative_drawdown', 0.0) + current_profit
                
                # 승률 조정: 경고 횟수와 누적 손실 기반으로 가중치 적용
                if perf['total_trades'] > 0:
                    # 위험 가중치: 경고 횟수에 따라 승률 페널티
                    risk_penalty = min(0.3, perf['risk_warnings'] * 0.05)  # 최대 30% 페널티
                    base_success_rate = perf['successful_trades'] / perf['total_trades']
                    perf['success_rate'] = max(0, base_success_rate - risk_penalty)
                else:
                    # 거래 이력이 없으면 초기 승률에서 페널티만 적용
                    risk_penalty = min(0.3, perf['risk_warnings'] * 0.05)
                    perf['success_rate'] = max(0, 0.5 - risk_penalty)
                
                print(f"📉 {signal_pattern} 패턴 신뢰도 하향 조정 -> {perf['success_rate']:.2f} (경고 {perf['risk_warnings']}회)")
                
        except Exception as e:
            print(f"⚠️ 실시간 위험 학습 오류: {e}")

class PatternAnalyzer:
    """패턴 분석기 - 거래 패턴 분석 및 개선점 도출"""
    def __init__(self):
        self.pattern_database = {}
        self.success_patterns = {}
        self.failure_patterns = {}
        
    def get_pattern_performance(self) -> dict:
        """패턴별 성과 반환 (DB에서 최신 데이터 로드)"""
        try:
            with sqlite3.connect(STRATEGY_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT signal_pattern, success_rate, avg_profit, total_trades, confidence
                    FROM signal_feedback_scores
                    ORDER BY total_trades DESC
                """)
                
                pattern_performance = {}
                for row in cursor.fetchall():
                    pattern, success_rate, avg_profit, total_trades, confidence = row
                    pattern_performance[pattern] = {
                        'success_rate': success_rate,
                        'avg_profit': avg_profit,
                        'total_trades': total_trades,
                        'confidence': confidence
                    }
                
                return pattern_performance
                
        except Exception as e:
            print(f"⚠️ 패턴 성과 조회 오류: {e}")
            return {}
        
    def analyze_pattern(self, trade_data: dict) -> dict:
        """거래 패턴 분석"""
        try:
            # 시그널 패턴 추출
            signal_pattern = self._extract_signal_pattern(trade_data)
            
            # 시장 상황 분석
            market_context = self._analyze_market_context(trade_data)
            
            # 성과 분석
            performance = self._analyze_performance(trade_data)
            
            # 패턴 분석 결과
            analysis_result = {
                'signal_pattern': signal_pattern,
                'market_context': market_context,
                'performance': performance,
                'timestamp': int(time.time())
            }
            
            # 패턴 데이터베이스 업데이트
            self.pattern_database[signal_pattern] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            print(f"⚠️ 패턴 분석 오류: {e}")
            return {}
    
    def _extract_signal_pattern(self, trade_data: dict) -> str:
        """시그널 패턴 추출 (정보 유실 방지 및 강제화)"""
        # 1. 🆕 이미 저장된 패턴이 있으면 우선 사용 (Executor에서 생성한 Rich Pattern)
        # 단, 'unknown'이 포함된 패턴은 가능한 한 다시 추출 시도
        existing_pattern = trade_data.get('signal_pattern')
        if existing_pattern and isinstance(existing_pattern, str) and existing_pattern.lower() != 'unknown' and 'unknown' not in existing_pattern.lower():
             return existing_pattern

        try:
            # 2. 🆕 패턴 타입(pattern_type) 활용
            pattern_prefix = ""
            pattern_type = trade_data.get('pattern_type') or trade_data.get('entry_pattern_type')
            if pattern_type and pattern_type != 'none':
                pattern_prefix = f"{pattern_type}_"

            # 3. RSI 범주화 (safe_float 사용)
            rsi = safe_float(trade_data.get('rsi') or trade_data.get('entry_rsi'), 50.0)
            rsi_level = self._discretize_rsi(rsi)
            
            # 4. MACD 범주화 (safe_float 사용)
            macd = safe_float(trade_data.get('macd') or trade_data.get('entry_macd'), 0.0)
            macd_level = self._discretize_macd(macd)
            
            # 5. 볼륨 범주화 (safe_float 사용)
            volume_ratio = safe_float(trade_data.get('volume_ratio') or trade_data.get('entry_volume_ratio'), 1.0)
            volume_level = self._discretize_volume(volume_ratio)
            
            # 6. 추세 방향 (Direction) 추가
            direction = trade_data.get('integrated_direction') or trade_data.get('entry_integrated_direction') or 'neutral'
            
            # 패턴 조합 (예: double_bottom_oversold_bullish_high_up)
            pattern = f"{pattern_prefix}{rsi_level}_{macd_level}_{volume_level}_{direction}"
            
            return pattern
            
        except Exception as e:
            # 🚨 [Fallback] 오류 발생 시에도 최소한의 패턴 정보 생성 시도 (절대 unknown 반환 안함)
            try:
                # 가능한 모든 정보를 긁어모아 키 생성
                coin = trade_data.get('coin', 'unknown')
                action = trade_data.get('action', 'unknown')
                score = trade_data.get('entry_signal_score', 0.0)
                
                # 점수대별 범주화 (10점 단위)
                score_level = f"s{int(score * 10)}"
                
                return f"{coin}_{action}_{score_level}_fallback"
            except:
                print(f"⚠️ 시그널 패턴 추출 치명적 오류: {e}")
                return f"emergency_fallback_{int(time.time())}"
    
    def _discretize_rsi(self, rsi: float) -> str:
        """RSI 값을 이산화"""
        if rsi < 30:
            return 'oversold'
        elif rsi < 45:
            return 'low'
        elif rsi < 55:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        else:
            return 'overbought'
    
    def _discretize_macd(self, macd: float) -> str:
        """MACD 값을 이산화"""
        if macd > 0.1:
            return 'strong_bullish'
        elif macd > 0:
            return 'bullish'
        elif macd > -0.1:
            return 'bearish'
        else:
            return 'strong_bearish'
    
    def _discretize_volume(self, volume_ratio: float) -> str:
        """거래량 비율을 이산화"""
        if volume_ratio < 0.5:
            return 'low'
        elif volume_ratio < 1.5:
            return 'normal'
        else:
            return 'high'
    
    def _analyze_market_context(self, trade_data: dict) -> dict:
        """시장 상황 분석"""
        try:
            # 기본 시장 상황
            market_context = {
                'trend': 'neutral',
                'volatility': trade_data.get('volatility', 0.02),
                'volume_trend': 'normal',
                'timestamp': int(time.time())
            }
            
            return market_context
            
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02, 'timestamp': int(time.time())}
    
    def _analyze_performance(self, trade_data: dict) -> dict:
        """성과 분석"""
        try:
            profit_loss_pct = trade_data.get('profit_loss_pct', 0.0)
            holding_duration = trade_data.get('holding_duration', 0)
            
            performance = {
                'profit_loss_pct': profit_loss_pct,
                'holding_duration': holding_duration,
                'success': profit_loss_pct > 0,
                'efficiency': profit_loss_pct / max(holding_duration, 1) if holding_duration > 0 else 0
            }
            
            return performance
            
        except Exception as e:
            print(f"⚠️ 성과 분석 오류: {e}")
            return {'profit_loss_pct': 0.0, 'success': False}

class FeedbackProcessor:
    """피드백 처리기 - 거래 결과 피드백 처리"""
    def __init__(self):
        self.feedback_queue = []
        self.processed_feedback = {}
        
    def process_feedback(self, trade_data: dict) -> dict:
        """거래 결과 피드백 처리"""
        try:
            # 🆕 피드백 데이터 준비 (목표가 달성 여부 포함)
            feedback_data = {
                'coin': trade_data.get('coin', 'unknown'),
                'entry_timestamp': trade_data.get('entry_timestamp', 0),
                'exit_timestamp': trade_data.get('exit_timestamp', 0),
                'profit_loss_pct': trade_data.get('profit_loss_pct', 0.0),
                'holding_duration': trade_data.get('holding_duration', 0),
                'signal_pattern': trade_data.get('signal_pattern', 'unknown'),
                'market_context': trade_data.get('market_context', {}),
                'processed_at': int(time.time())
            }
            
            # 목표가 정보가 있으면 달성 여부 평가
            # 🆕 초기 목표가를 기준으로 평가해야 "예측 정확도"를 제대로 알 수 있음
            target_price = trade_data.get('initial_target_price', 0)
            if target_price == 0:
                target_price = trade_data.get('target_price', 0)

            if target_price > 0:
                entry_price = trade_data.get('entry_price', 0)
                exit_price = trade_data.get('exit_price', 0)
                # target_price 변수는 위에서 이미 설정됨
                
                # 목표가 달성 여부 (매수 기준)
                if entry_price > 0:
                    # 목표 수익률
                    target_profit_pct = ((target_price - entry_price) / entry_price) * 100
                    # 실제 수익률
                    actual_profit_pct = trade_data.get('profit_loss_pct', 0.0)
                    
                    # 목표가의 80% 이상 도달했으면 성공으로 간주
                    if target_profit_pct != 0:
                        target_hit = actual_profit_pct >= (target_profit_pct * 0.8)
                        target_accuracy = actual_profit_pct / target_profit_pct
                    else:
                        target_hit = False
                        target_accuracy = 0.0
                    
                    feedback_data['target_hit'] = target_hit
                    feedback_data['target_accuracy'] = target_accuracy
                    
                    if target_hit:
                        print(f"🎯 목표가 적중! (초기예상: {target_profit_pct:.2f}%, 실제: {actual_profit_pct:.2f}%)")
                    else:
                        print(f"📉 목표가 미달 (초기예상: {target_profit_pct:.2f}%, 실제: {actual_profit_pct:.2f}%)")

            # 피드백 큐에 추가
            self.feedback_queue.append(feedback_data)
            
            # 처리된 피드백 저장
            feedback_id = f"{feedback_data['coin']}_{feedback_data['entry_timestamp']}"
            self.processed_feedback[feedback_id] = feedback_data
            
            # 🆕 패턴 정보가 unknown인 경우, 시그널 점수로 추정하여 로그 가독성 향상
            display_pattern = feedback_data['signal_pattern']
            if display_pattern in ['unknown', 'none']:
                entry_score = trade_data.get('entry_signal_score', 0.0)
                if entry_score > 0.3:
                    display_pattern = "bullish_high_(est)"
                elif entry_score < -0.3:
                    display_pattern = "bearish_high_(est)"
                else:
                    display_pattern = "neutral_low_(est)"
            
            print(f"📊 피드백 처리: {feedback_data['coin']} 패턴 {display_pattern}")
            
            return feedback_data
            
        except Exception as e:
            print(f"⚠️ 피드백 처리 오류: {e}")
            return {}
    
    def get_feedback_summary(self) -> dict:
        """피드백 요약 정보"""
        try:
            total_feedback = len(self.processed_feedback)
            successful_trades = sum(1 for f in self.processed_feedback.values() if f.get('profit_loss_pct', 0) > 0)
            total_profit = sum(f.get('profit_loss_pct', 0) for f in self.processed_feedback.values())
            
            summary = {
                'total_trades': total_feedback,
                'successful_trades': successful_trades,
                'success_rate': successful_trades / max(total_feedback, 1),
                'total_profit': total_profit,
                'avg_profit': total_profit / max(total_feedback, 1)
            }
            
            return summary
            
        except Exception as e:
            print(f"⚠️ 피드백 요약 오류: {e}")
            return {'total_trades': 0, 'success_rate': 0.0, 'total_profit': 0.0}

class EvolutionEngine:
    """진화 엔진 - 학습 결과를 바탕으로 시스템 진화"""
    def __init__(self):
        self.evolution_history = []
        self.performance_trends = {}
        
    def get_evolution_summary(self) -> dict:
        """진화 결과 요약 반환 (DB에서 최신 데이터 로드)"""
        try:
            with sqlite3.connect(STRATEGY_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()

                # 🔧 테이블 존재 여부 확인
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='evolution_results'
                """)
                table_exists = cursor.fetchone() is not None

                evolution_summary = {
                    'recent_evolutions': [],
                    'current_direction': 'stable',
                    'performance_trend': 'neutral',
                    'total_evolutions': len(self.evolution_history)
                }

                if table_exists:
                    cursor.execute("""
                        SELECT evolution_direction, changes, performance_trend, created_at
                        FROM evolution_results
                        ORDER BY created_at DESC
                        LIMIT 10
                    """)

                    for row in cursor.fetchall():
                        direction, changes, trend, created_at = row
                        evolution_summary['recent_evolutions'].append({
                            'direction': direction,
                            'changes': changes,
                            'trend': trend,
                            'created_at': created_at
                        })

                    # 최근 진화 방향 결정
                    if evolution_summary['recent_evolutions']:
                        latest = evolution_summary['recent_evolutions'][0]
                        evolution_summary['current_direction'] = latest['direction']
                        evolution_summary['performance_trend'] = latest['trend']

                return evolution_summary

        except Exception as e:
            print(f"⚠️ 진화 결과 조회 오류: {e}")
            return {
                'recent_evolutions': [],
                'current_direction': 'stable',
                'performance_trend': 'neutral',
                'total_evolutions': len(self.evolution_history)
            }
        
    def evolve_system(self, feedback_summary: dict) -> dict:
        """시스템 진화"""
        try:
            # 성과 트렌드 분석
            performance_trend = self._analyze_performance_trend(feedback_summary)
            
            # 진화 방향 결정
            evolution_direction = self._determine_evolution_direction(performance_trend)
            
            # 진화 실행
            evolution_result = self._execute_evolution(evolution_direction)
            
            # 진화 기록
            evolution_record = {
                'timestamp': int(time.time()),
                'performance_trend': performance_trend,
                'evolution_direction': evolution_direction,
                'evolution_result': evolution_result
            }

            self.evolution_history.append(evolution_record)

            # 🆕 DB에 진화 결과 저장
            try:
                with sqlite3.connect(STRATEGY_DB_PATH, timeout=60.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO evolution_results
                        (evolution_direction, changes, performance_trend, win_rate, avg_profit, total_trades, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        evolution_direction,
                        json.dumps(evolution_result.get('changes', {})),
                        json.dumps(performance_trend),
                        performance_trend.get('success_rate', 0.0),
                        performance_trend.get('avg_profit', 0.0),
                        feedback_summary.get('total_trades', 0),
                        int(time.time())
                    ))
                    conn.commit()
            except Exception as e:
                print(f"⚠️ 진화 결과 저장 오류: {e}")

            print(f"🧬 시스템 진화: {evolution_direction} 방향으로 진화 실행")
            
            return evolution_result
            
        except Exception as e:
            print(f"⚠️ 시스템 진화 오류: {e}")
            return {}
    
    def _analyze_performance_trend(self, feedback_summary: dict) -> dict:
        """성과 트렌드 분석"""
        try:
            success_rate = feedback_summary.get('success_rate', 0.0)
            avg_profit = feedback_summary.get('avg_profit', 0.0)
            
            # 트렌드 분석
            if success_rate > 0.6 and avg_profit > 0.05:
                trend = 'excellent'
            elif success_rate > 0.5 and avg_profit > 0.02:
                trend = 'good'
            elif success_rate > 0.4 and avg_profit > 0:
                trend = 'average'
            else:
                trend = 'poor'
            
            return {
                'trend': trend,
                'success_rate': success_rate,
                'avg_profit': avg_profit,
                'timestamp': int(time.time())
            }
            
        except Exception as e:
            print(f"⚠️ 성과 트렌드 분석 오류: {e}")
            return {'trend': 'unknown', 'success_rate': 0.0, 'avg_profit': 0.0}
    
    def _determine_evolution_direction(self, performance_trend: dict) -> str:
        """진화 방향 결정"""
        try:
            trend = performance_trend.get('trend', 'unknown')
            
            if trend == 'excellent':
                return 'maintain_and_optimize'
            elif trend == 'good':
                return 'gradual_improvement'
            elif trend == 'average':
                return 'moderate_enhancement'
            else:
                return 'major_overhaul'
                
        except Exception as e:
            print(f"⚠️ 진화 방향 결정 오류: {e}")
            return 'maintain_and_optimize'
    
    def _execute_evolution(self, evolution_direction: str) -> dict:
        """진화 실행"""
        try:
            evolution_result = {
                'direction': evolution_direction,
                'executed_at': int(time.time()),
                'changes': []
            }
            
            if evolution_direction == 'maintain_and_optimize':
                evolution_result['changes'] = ['현재 성과 유지', '세부 최적화']
            elif evolution_direction == 'gradual_improvement':
                evolution_result['changes'] = ['점진적 개선', '안정성 강화']
            elif evolution_direction == 'moderate_enhancement':
                evolution_result['changes'] = ['중간 수준 개선', '리스크 관리 강화']
            else:
                evolution_result['changes'] = ['대폭 개선', '전략 재검토']
            
            return evolution_result
            
        except Exception as e:
            print(f"⚠️ 진화 실행 오류: {e}")
            return {'direction': 'unknown', 'changes': []}

class SignalTradeConnector:
    """시그널-매매 연결 시스템"""
    def __init__(self):
        self.connections = {}
        self.pending_signals = {}
        
    def connect_signal_to_trade(self, signal: SignalInfo, trade_result: dict):
        """시그널과 매매 결과 연결"""
        try:
            connection_id = f"{signal.coin}_{signal.timestamp}"
            self.connections[connection_id] = {
                'signal': signal,
                'trade_result': trade_result,
                'connected_at': time.time()
            }
            print(f"🔗 시그널-매매 연결: {signal.coin} 연결 완료")
        except Exception as e:
            print(f"⚠️ 시그널-매매 연결 오류: {e}")

class MarketInsightMiner:
    """시장 통찰 발굴기 - 전체 코인의 급등/급락에서 교훈 학습
    
    🆕 [개선] 캔들 데이터 기반으로 단순화:
    1. 전체 코인에서 급등/급락 코인 발견 (candles 테이블)
    2. 해당 코인을 매수했었나? → 매수 여부에 따라 학습
       - 안 샀는데 급등 → "놓친 기회" 학습
       - 안 샀는데 급락 → "잘한 관망" 학습
    """
    def __init__(self, learner):
        self.learner = learner
        self.db_path = learner.TRADING_SYSTEM_DB_PATH
        self.min_rise_threshold = 5.0   # 5% 이상 상승 (놓친 기회)
        self.min_drop_threshold = -5.0  # 5% 이상 하락 (잘한 관망)
        self.lookback_hours = 12        # 최근 12시간 데이터 확인
        self.processed_insights = set()  # 중복 학습 방지

    def mine_insights(self, current_prices: Dict[str, float]):
        """전체 코인 캔들 기반 급등/급락 학습"""
        try:
            current_time = int(time.time())
            start_time = current_time - (self.lookback_hours * 3600)
            
            # 캔들 DB 경로
            candles_db_path = os.environ.get('RL_DB_PATH', DB_PATH)
            if not os.path.exists(candles_db_path):
                print(f"⚠️ 캔들 DB를 찾을 수 없습니다: {candles_db_path}")
                return
            
            # 1. 전체 코인의 가격 변동 조회
            with sqlite3.connect(candles_db_path, timeout=60.0) as conn:
                # lookback 기간 시작 시점의 종가와 기간 내 최고가/최저가 조회
                query = """
                    WITH first_candles AS (
                        SELECT symbol, close as start_price, MIN(timestamp) as first_ts
                        FROM candles
                        WHERE timestamp >= ? AND timestamp < ? + 3600
                        GROUP BY symbol
                    ),
                    price_range AS (
                        SELECT 
                            symbol,
                            MAX(high) as max_high,
                            MIN(low) as min_low
                        FROM candles
                        WHERE timestamp >= ?
                        GROUP BY symbol
                    )
                    SELECT 
                        f.symbol,
                        f.start_price,
                        p.max_high,
                        p.min_low
                    FROM first_candles f
                    JOIN price_range p ON f.symbol = p.symbol
                    WHERE f.start_price > 0
                """
                df = pd.read_sql(query, conn, params=(start_time, start_time, start_time))
            
            if df.empty:
                return

            # 2. 가상매매에서 현재 보유 중인 코인 / 최근 매수한 코인 조회
            held_coins = self._get_held_or_traded_coins(start_time)
            
            opportunity_count = 0
            avoidance_count = 0
            
            for _, row in df.iterrows():
                coin = row['symbol']
                start_price = safe_float(row['start_price'])
                max_high = safe_float(row['max_high'])
                min_low = safe_float(row['min_low'])
                
                if start_price <= 0:
                    continue
                
                # 중복 방지 (날짜+코인 기준, 시간 단위)
                insight_id = f"{coin}_{start_time // 3600}"
                if insight_id in self.processed_insights:
                    continue
                
                # 수익률 계산
                max_profit_pct = ((max_high - start_price) / start_price) * 100
                max_loss_pct = ((min_low - start_price) / start_price) * 100
                
                # 이미 매수한 코인은 제외 (놓친 기회/잘한 관망이 아님)
                if coin in held_coins:
                    continue
                
                # 패턴 생성 (🆕 급등/급락 직전 시점의 시그널 데이터 활용)
                pattern = self._create_pattern(coin, start_price, max_high, min_low, max_profit_pct, max_loss_pct, start_time)
                
                # 🕵️ 놓친 기회: 안 샀는데 급등
                if max_profit_pct >= self.min_rise_threshold:
                    print(f"🕵️ [놓친 기회] {coin}: {self.lookback_hours}시간 내 +{max_profit_pct:.2f}% 급등! (미보유)")
                    
                    # 학습: "이런 상황에서는 샀어야 했다" → 성공 케이스로 학습
                    self.learner.thompson_sampler.update_distribution(
                        pattern, success=True, profit_pct=max_profit_pct, weight=1.0
                    )
                    opportunity_count += 1
                    self.processed_insights.add(insight_id)
                    
                    self.learner.log_system_event("WARN", "Learner", 
                        f"🕵️ {coin} 놓친 기회 (+{max_profit_pct:.1f}%) → 패턴 학습 강화", {
                            "pattern": pattern,
                            "missed_profit": max_profit_pct,
                            "max_price": max_high
                        })

                # 🛡️ 잘한 관망: 안 샀는데 급락
                elif max_loss_pct <= self.min_drop_threshold:
                    print(f"🛡️ [잘한 관망] {coin}: {self.lookback_hours}시간 내 {max_loss_pct:.2f}% 급락! (미보유)")
                    
                    # 학습: "이런 상황에서 안 산 게 잘한 것" → 실패 케이스로 학습
                    self.learner.thompson_sampler.update_distribution(
                        pattern, success=False, profit_pct=max_loss_pct, weight=1.0
                    )
                    avoidance_count += 1
                    self.processed_insights.add(insight_id)
                    
                    self.learner.log_system_event("INFO", "Learner", 
                        f"🛡️ {coin} 하락 회피 ({max_loss_pct:.1f}%) → 방어적 판단 강화", {
                            "pattern": pattern,
                            "avoided_loss": max_loss_pct
                        })
            
            if opportunity_count > 0 or avoidance_count > 0:
                print(f"🧠 시장 통찰 학습: 놓친 기회 {opportunity_count}건, 잘한 관망 {avoidance_count}건")
            
            # 오래된 캐시 정리 (24시간 이상)
            self._cleanup_old_insights()
                
        except Exception as e:
            print(f"⚠️ 시장 통찰 발굴 오류: {e}")

    def _get_held_or_traded_coins(self, since_timestamp: int) -> set:
        """가상매매에서 보유 중이거나 최근 매매한 코인 목록 조회"""
        held_coins = set()
        try:
            with sqlite3.connect(TRADING_DB_PATH, timeout=10.0) as conn:
                cursor = conn.cursor()
                
                # 1. 현재 보유 중인 코인
                try:
                    cursor.execute("""
                        SELECT DISTINCT coin FROM virtual_positions 
                        WHERE is_open = 1
                    """)
                    for row in cursor.fetchall():
                        held_coins.add(row[0])
                except:
                    pass
                
                # 2. lookback 기간 내 매매한 코인 (매수/매도 모두)
                try:
                    cursor.execute("""
                        SELECT DISTINCT coin FROM virtual_trade_history 
                        WHERE entry_timestamp >= ? OR exit_timestamp >= ?
                    """, (since_timestamp, since_timestamp))
                    for row in cursor.fetchall():
                        held_coins.add(row[0])
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ 보유 코인 조회 오류: {e}")
        
        return held_coins

    def _create_pattern(self, coin: str, start_price: float, max_high: float, 
                       min_low: float, profit_pct: float, loss_pct: float,
                       start_timestamp: int = None) -> str:
        """🆕 급등/급락 직전의 시장 상황을 패턴으로 생성 (예측에 활용 가능!)
        
        핵심: "결과"가 아닌 "직전 상황"을 패턴으로 만들어야 예측 가능
        - 급등 직전에 RSI가 낮았다 → 다음에 RSI 낮으면 매수 고려
        - 급락 직전에 RSI가 높았다 → 다음에 RSI 높으면 매수 주의
        """
        try:
            # 🆕 급등/급락 직전 시점의 시그널 데이터 조회
            pre_signal = self._get_pre_move_signal(coin, start_timestamp)
            
            if pre_signal:
                # 시그널 데이터가 있으면 → 정확한 "직전 상황" 패턴 생성
                rsi = safe_float(pre_signal.get('rsi', 50))
                macd = safe_float(pre_signal.get('macd', 0))
                volume_ratio = safe_float(pre_signal.get('volume_ratio', 1.0))
                
                # RSI 범주화
                rsi_level = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
                
                # MACD 범주화
                macd_level = 'bullish' if macd > 0.01 else 'bearish' if macd < -0.01 else 'flat'
                
                # 거래량 범주화
                vol_level = 'high_vol' if volume_ratio > 2.0 else 'low_vol' if volume_ratio < 0.5 else 'normal_vol'
                
                # 결과 태그 (학습용)
                result_tag = 'SURGE' if profit_pct >= 5.0 else 'CRASH' if loss_pct <= -5.0 else 'FLAT'
                
                # 🎯 예측 가능한 패턴: "직전 상황_결과"
                # 예: PRE_oversold_bullish_high_vol_SURGE → "과매도+상승신호+거래량↑ → 급등"
                return f"PRE_{rsi_level}_{macd_level}_{vol_level}_{result_tag}"
            
            else:
                # 시그널 데이터가 없으면 → 기본 캔들 기반 패턴 (fallback)
                volatility = ((max_high - min_low) / start_price) * 100
                vol_level = 'high_vol' if volatility > 15 else 'med_vol' if volatility > 7 else 'low_vol'
                
                if profit_pct >= 5.0:
                    direction = 'surge'
                elif loss_pct <= -5.0:
                    direction = 'crash'
                else:
                    direction = 'neutral'
                
                return f"CANDLE_{vol_level}_{direction}"
                
        except Exception as e:
            return f"INSIGHT_error_{coin[:3]}"

    def _get_pre_move_signal(self, coin: str, timestamp: int) -> Optional[Dict]:
        """🆕 급등/급락 직전 시점의 시그널 데이터 조회"""
        if not timestamp:
            return None
            
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # 급등/급락 시작 시점 전후 1시간 내의 시그널 조회
                query = """
                    SELECT rsi, macd, volume_ratio, signal_score, confidence
                    FROM signals
                    WHERE coin = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY ABS(timestamp - ?) ASC
                    LIMIT 1
                """
                cursor = conn.execute(query, (
                    coin, 
                    timestamp - 3600,  # 1시간 전
                    timestamp + 1800,  # 30분 후 (약간의 여유)
                    timestamp
                ))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'rsi': row[0],
                        'macd': row[1],
                        'volume_ratio': row[2],
                        'signal_score': row[3],
                        'confidence': row[4]
                    }
        except:
            pass
        
        return None

    def _cleanup_old_insights(self):
        """24시간 이상 지난 학습 기록 정리"""
        try:
            current_hour = int(time.time()) // 3600
            old_insights = [k for k in self.processed_insights 
                          if '_' in k and int(k.split('_')[-1]) < current_hour - 24]
            for k in old_insights:
                self.processed_insights.discard(k)
        except:
            pass

# 🚫 RL 학습 클래스 제거됨 - 순수 피드백 제공자로 변경

class VirtualTradingLearner:
    """가상매매 순수 피드백 제공자 (증분 학습 시스템)"""
    
    def __init__(self):
        print("🚀 최적화된 피드백 처리 시스템 초기화 중...")
        
        # 🆕 DB 경로 인스턴스 변수 설정 (MarketInsightMiner 등에서 참조)
        self.TRADING_SYSTEM_DB_PATH = TRADING_SYSTEM_DB_PATH
        
        # 🚀 최적화된 학습 범위 설정
        self.max_hours_back = int(os.getenv('VIRTUAL_LEARNING_MAX_HOURS', '6'))  # 기본 6시간
        self.batch_size = int(os.getenv('VIRTUAL_LEARNING_BATCH_SIZE', '100'))   # 기본 100개 (증가)
        self.max_processing_time = int(os.getenv('VIRTUAL_LEARNING_MAX_TIME', '30'))  # 기본 30초
        
        # 🚀 실시간 학습용 설정 (더 빠른 처리)
        self.realtime_max_hours = int(os.getenv('VIRTUAL_LEARNING_REALTIME_HOURS', '2'))  # 기본 2시간
        self.realtime_batch_size = int(os.getenv('VIRTUAL_LEARNING_REALTIME_BATCH', '50'))  # 기본 50개 (증가)
        self.realtime_max_time = int(os.getenv('VIRTUAL_LEARNING_REALTIME_TIME', '15'))  # 기본 15초
        
        # 🆕 증분 학습 시스템 설정
        self.incremental_learning = True  # 증분 학습 활성화
        self.last_learning_timestamp = 0  # 마지막 학습 시점
        self.learning_checkpoint = {}  # 학습 체크포인트
        self.processed_trade_ids = set()  # 처리된 거래 ID 추적
        self.learning_episode = 0  # 학습 에피소드 번호
        
        # 🚀 성능 최적화 설정
        self.cache_size = 1000
        self.cache_ttl = 300  # 5분 캐시
        self.feedback_cache = {}
        self.last_cache_cleanup = time.time()
        
        # 🚀 배치 처리 설정
        self.feedback_batch = []
        self.last_batch_process = time.time()
        self.batch_interval = 60  # 1분마다 배치 처리
        
        # 🆕 성능 업그레이드 시스템 초기화
        self.recency_aggregator = RecencyWeightedAggregator(decay_rate=0.1)
        self.bayesian_applier = BayesianSmoothingApplier()
        self.outlier_applier = OutlierGuardrailApplier()
        self.post_trade_evaluator = PostTradeEvaluator()  # 🆕 매매 사후 평가기
        
        # 🆕 진화형 AI 시스템 초기화
        self.real_time_learner = RealTimeLearner()
        self.pattern_analyzer = PatternAnalyzer()
        self.feedback_processor = FeedbackProcessor()
        self.evolution_engine = EvolutionEngine()
        
        # 🆕 시그널-매매 연결 시스템
        self.signal_trade_connector = SignalTradeConnector()
        
        # 🆕 [Confidence Calibration] 신뢰도 교정기 초기화
        self.calibration_tracker = CalibrationTracker()

        # 🎰 Thompson Sampling 강화학습 시스템 초기화
        self.thompson_sampler = ThompsonSamplingLearner(db_path=STRATEGY_DB_PATH)
        
        # 🆕 시장 통찰 발굴기 (놓친 기회 학습)
        self.market_miner = MarketInsightMiner(self)

        # 🚀 [추가] Unknown 패턴 재학습 실행 (초기화 시 1회 시도)
        self._relearn_unknown_trades()
        
        print(f"📊 진화형 AI 피드백 처리 설정:")
        print(f"  📦 배치 크기: {self.batch_size}개 (증가)")
        print(f"  ⏱️ 처리 시간 제한: {self.max_processing_time}초")
        print(f"  🚀 캐시 시스템: 활성화")
        print(f"  📦 배치 처리: 활성화")
        
        # 테이블 생성
        self.create_learning_tables()
        
        # 🆕 [중복 학습 방지] 이미 처리된 거래 ID 로드
        self._load_processed_trades()
        
        print("✅ 피드백 처리 시스템 초기화 완료!")
    
    def _load_processed_trades(self):
        """이미 학습된 거래 ID 목록 로드 (중복 학습 방지)"""
        try:
            print(f"📂 [DEBUG] 학습 내역 로드 시작 (DB: {TRADING_DB_PATH})")
            with sqlite3.connect(TRADING_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                # virtual_trade_feedback 테이블이 존재하는지 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='virtual_trade_feedback'")
                if not cursor.fetchone():
                    print("⚠️ virtual_trade_feedback 테이블이 존재하지 않습니다.")
                    return

                # 최근 10000개 정도의 처리된 거래 로드 (coin, entry_timestamp 조합)
                cursor.execute("""
                    SELECT coin, entry_timestamp 
                    FROM virtual_trade_feedback 
                    ORDER BY id DESC LIMIT 10000
                """)
                rows = cursor.fetchall()
                
                for coin, entry_ts in rows:
                    trade_id = f"{coin}_{entry_ts}"
                    self.processed_trade_ids.add(trade_id)
                    
            print(f"📦 이미 학습된 거래 {len(self.processed_trade_ids)}건 로드 완료 (중복 방지)")
            
        except Exception as e:
            print(f"⚠️ 처리된 거래 로드 실패: {e}")

    def _relearn_unknown_trades(self):
        """🚀 과거 Unknown 패턴 거래에 대한 재학습 (패턴 복원)"""
        print("🔄 과거 Unknown 거래 패턴 복원 및 재학습 시작...")
        try:
            with sqlite3.connect(TRADING_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # signal_pattern 컬럼 확인
                cursor.execute("PRAGMA table_info(virtual_trade_history)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'signal_pattern' not in columns:
                    print("⚠️ virtual_trade_history에 signal_pattern 컬럼이 없어 재학습을 건너뜁니다.")
                    return

                # Unknown이거나 NULL인 거래 조회 (none 문자열 포함)
                query = """
                    SELECT rowid, coin, entry_timestamp, profit_loss_pct, entry_signal_score
                    FROM virtual_trade_history 
                    WHERE signal_pattern IS NULL 
                       OR signal_pattern = 'unknown' 
                       OR signal_pattern = 'unknown_pattern'
                       OR signal_pattern = 'none'
                """
                cursor.execute(query)
                unknown_trades = cursor.fetchall()
            
            if not unknown_trades:
                print("✅ 복원할 Unknown 거래가 없습니다.")
                return
                
            print(f"🔍 총 {len(unknown_trades)}개의 Unknown 거래 발견. 패턴 복원 시도...")
            
            restored_count = 0
            
            # signals 테이블 위치 확인
            try:
                signal_db_path = get_db_path_for_table('signals')
            except:
                signal_db_path = TRADING_SYSTEM_DB_PATH # fallback

            with sqlite3.connect(signal_db_path, timeout=60.0) as signal_conn:
                with sqlite3.connect(TRADING_DB_PATH, timeout=60.0) as update_conn:
                    for rowid, coin, entry_timestamp, profit_loss_pct, entry_score in unknown_trades:
                        try:
                            # 1. 당시 시그널 조회 (오차 범위 5분)
                            query = """
                                SELECT * FROM signals 
                                WHERE coin = ? AND timestamp BETWEEN ? AND ?
                                ORDER BY ABS(timestamp - ?) ASC LIMIT 1
                            """
                            time_margin = 300 # 5분
                            df_sig = pd.read_sql(query, signal_conn, params=(
                                coin, entry_timestamp - time_margin, entry_timestamp + time_margin, entry_timestamp
                            ))
                            
                            restored_pattern = None
                            
                            if not df_sig.empty:
                                # 패턴 추출
                                signal_row = df_sig.iloc[0]
                                restored_pattern = self.pattern_analyzer._extract_signal_pattern(signal_row)
                            else:
                                # 시그널이 없으면 점수 기반 추정 패턴 생성
                                score_s = 'high' if entry_score > 0.05 else 'low' if entry_score < 0.01 else 'medium'
                                restored_pattern = f"SRC_RESTORED_unknown_unknown_unknown_medium_{score_s}"
                            
                            if restored_pattern and restored_pattern != 'unknown':
                                # 2. 재학습 (Thompson Sampling)
                                success = profit_loss_pct > 0
                                self.thompson_sampler.update_distribution(
                                    restored_pattern, success=success, profit_pct=profit_loss_pct, weight=1.0
                                )
                                
                                # 3. DB 업데이트
                                update_conn.execute("""
                                    UPDATE virtual_trade_history 
                                    SET signal_pattern = ? 
                                    WHERE rowid = ?
                                """, (restored_pattern, rowid))
                                
                                restored_count += 1
                                
                        except Exception as e:
                            continue
                    
                    update_conn.commit()
            
            print(f"✨ {restored_count}개의 거래 패턴 복원 및 재학습 완료!")
            
        except Exception as e:
            print(f"⚠️ 재학습 중 오류 발생: {e}")
    
    def create_learning_tables(self):
        """학습 관련 테이블 생성"""
        try:
            # 1. 전략 DB 테이블 (signal_feedback_scores, evolution_results 등)
            with sqlite3.connect(STRATEGY_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # 시그널 피드백 점수 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        signal_pattern TEXT PRIMARY KEY,
                        success_rate REAL DEFAULT 0.5,
                        avg_profit REAL DEFAULT 0.0,
                        total_trades INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0.0,
                        updated_at INTEGER
                    )
                """)

                # 진화 결과 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evolution_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evolution_direction TEXT,
                        changes TEXT,
                        performance_trend TEXT,
                        win_rate REAL,
                        avg_profit REAL,
                        total_trades INTEGER,
                        created_at INTEGER
                    )
                """)
                
                # 학습 체크포인트 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_checkpoint (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at INTEGER
                    )
                """)

                # 🆕 [Adaptive Exit] 패턴별 최적 청산 파라미터 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_exit_params (
                        signal_pattern TEXT PRIMARY KEY,
                        optimal_tp_ratio REAL DEFAULT 2.0, -- Risk Reward Ratio (Target / Risk)
                        optimal_sl_ratio REAL DEFAULT 1.0, -- 보통 1.0 (Stop Loss 배수)
                        avg_mfe REAL DEFAULT 0.0, -- 평균 최대 수익폭 (Maximum Favorable Excursion)
                        avg_mae REAL DEFAULT 0.0, -- 평균 최대 손실폭 (Maximum Adverse Excursion)
                        samples INTEGER DEFAULT 0,
                        updated_at INTEGER
                    )
                """)
                
                conn.commit()
                
            # 2. 매매 DB 테이블 (virtual_trade_feedback 등)
            with sqlite3.connect(TRADING_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # 가상 매매 피드백 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_trade_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT,
                        entry_timestamp INTEGER,
                        exit_timestamp INTEGER,
                        profit_loss_pct REAL,
                        signal_pattern TEXT,
                        target_hit INTEGER DEFAULT 0,
                        target_accuracy REAL DEFAULT 0.0,
                        processed_at INTEGER,
                        entry_confidence REAL DEFAULT 0.0,
                        exit_confidence REAL DEFAULT 0.0
                    )
                """)
                
                # 🆕 컬럼 마이그레이션 (기존 DB 호환성)
                cursor.execute("PRAGMA table_info(virtual_trade_feedback)")
                cols = [c[1] for c in cursor.fetchall()]
                if 'entry_confidence' not in cols:
                    try: 
                        cursor.execute("ALTER TABLE virtual_trade_feedback ADD COLUMN entry_confidence REAL DEFAULT 0.0")
                    except: pass
                
                if 'exit_confidence' not in cols:
                    try: 
                        cursor.execute("ALTER TABLE virtual_trade_feedback ADD COLUMN exit_confidence REAL DEFAULT 0.0")
                    except: pass
                    
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 학습 테이블 생성 오류: {e}")

    def process_feedback(self):
        """외부에서 호출 가능한 피드백 처리 메서드 (wrapper)"""
        print("🔄 수동 피드백 처리 요청 실행...")
        self._execute_real_time_learning()
        self._execute_system_evolution()
        self._cleanup_old_data()

    def _execute_real_time_learning(self):
        """실시간 학습 실행"""
        try:
            # 최근 완료된 거래 조회 (completed_trades from TRADING_DB_PATH)
            # 여기서는 가상 매매 기록을 가져와서 학습
            with sqlite3.connect(TRADING_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # 아직 처리되지 않은(processed_at이 없는) 가상 매매 기록 조회
                # 실제로는 별도의 processed_at 필드를 두거나, last_learning_timestamp 이후의 데이터를 조회
                # 여기서는 간단히 최근 데이터 조회 후 메모리 상의 processed_trade_ids로 필터링
                
                # 가상 매매 기록 테이블이 'virtual_trade_history'라고 가정 (completed_trades는 실전일 수 있음)
                # 확인 필요: virtual_trade_executor가 어디에 저장하는지.
                # 보통 virtual_trade_history 테이블을 사용함.
                
                # 테이블 존재 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='virtual_trade_history'")
                if not cursor.fetchone():
                    return

                query = """
                    SELECT * FROM virtual_trade_history 
                    WHERE exit_timestamp > ?
                    ORDER BY exit_timestamp ASC
                """
                
                # 최근 24시간 데이터 조회 (안전장치)
                start_time = max(self.last_learning_timestamp, int(time.time()) - 24*3600)
                df = pd.read_sql(query, conn, params=(start_time,))
                
            if df.empty:
                return
                
            new_trades_count = 0
            
            for _, row in df.iterrows():
                trade_id = f"{row['coin']}_{row['entry_timestamp']}"
                if trade_id in self.processed_trade_ids:
                    continue
                
                # 거래 데이터 구성
                trade_data = {
                    'coin': row['coin'],
                    'entry_timestamp': row['entry_timestamp'],
                    'exit_timestamp': row['exit_timestamp'],
                    'profit_loss_pct': row['profit_loss_pct'],
                    'holding_duration': row['exit_timestamp'] - row['entry_timestamp'],
                    'entry_signal_score': row.get('entry_signal_score', 0),
                    'entry_confidence': row.get('entry_confidence', 0.0), # 🆕 신뢰도 추가
                    'signal_pattern': row.get('signal_pattern', 'unknown'),
                    'action': row.get('exit_reason', 'sell'), # exit_reason을 action으로 매핑
                    'exit_price': row['exit_price'],
                    'entry_price': row['entry_price']
                }
                
                success = trade_data['profit_loss_pct'] > 0
                
                # 1. Thompson Sampling 업데이트
                signal_pattern = trade_data.get('signal_pattern', 'unknown')
                if signal_pattern and signal_pattern != 'unknown':
                    self.thompson_sampler.update_distribution(
                        signal_pattern, 
                        success=success, 
                        profit_pct=trade_data['profit_loss_pct'],
                        weight=1.0 # 가상 매매 가중치
                    )
                
                # 🆕 1-1. 신뢰도 교정 (Confidence Calibration) 업데이트
                entry_confidence = trade_data.get('entry_confidence', 0.0) 
                if entry_confidence > 0:
                    self.calibration_tracker.update(entry_confidence, success)
                    # print(f"🔧 신뢰도 교정 업데이트: 예측 {entry_confidence:.2f} -> 결과 {'✅' if success else '❌'}")
                
                # 2. 실시간 학습기 업데이트
                self.real_time_learner.learn_from_trade(signal_pattern, trade_data)
                
                # 3. 사후 평가기 등록
                self.post_trade_evaluator.add_trade(trade_data)
                
                # 🆕 4. 학습 완료 기록 (DB에 저장하여 재시작 시에도 중복 방지)
                self._record_processed_trade(trade_data, signal_pattern)
                
                self.processed_trade_ids.add(trade_id)
                new_trades_count += 1
                self.last_learning_timestamp = max(self.last_learning_timestamp, row['exit_timestamp'])
            
            if new_trades_count > 0:
                print(f"📚 실시간 학습 완료: {new_trades_count}개 거래 학습")
                # Unknown 패턴 비율 경고
                unknown_count = sum(1 for trade in df.to_dict('records') if trade.get('signal_pattern', 'unknown') == 'unknown')
                if unknown_count > 0:
                    print(f"⚠️ 주의: {unknown_count}개 거래의 패턴 정보를 찾을 수 없습니다. (DB 컬럼 업데이트 필요)")
            
        except Exception as e:
            print(f"⚠️ 실시간 학습 실행 오류: {e}")

    def log_system_event(self, level: str, component: str, message: str, details: dict = None):
        """🆕 시스템 로그 DB 저장 (대시보드 노출용)"""
        try:
            with sqlite3.connect(TRADING_DB_PATH, timeout=10.0) as conn:
                cursor = conn.cursor()
                created_at = datetime.now().isoformat()
                timestamp = int(time.time())
                detail_json = json.dumps(details, ensure_ascii=False) if details else "{}"
                
                cursor.execute("""
                    INSERT INTO system_logs (level, component, message, details, created_at, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (level, component, message, detail_json, created_at, timestamp))
                conn.commit()
        except Exception as e:
            # 로그 저장 실패는 치명적이지 않으므로 출력만 함
            print(f"⚠️ 시스템 로그 저장 실패: {e}")

    def _record_processed_trade(self, trade_data: dict, signal_pattern: str):
        """학습 완료된 거래를 DB에 기록 (virtual_trade_feedback)"""
        try:
            with sqlite3.connect(TRADING_DB_PATH, timeout=60.0) as conn:
                cursor = conn.cursor()
                
                # 🚀 [Fix] NOT NULL 제약조건 해결을 위해 필수 컬럼 모두 포함
                entry_signal_score = trade_data.get('entry_signal_score', 0.0)
                # exit_signal_score가 trade_data에 없으면 entry_signal_score로 대체하거나 0.0 사용
                exit_signal_score = trade_data.get('exit_signal_score', 0.0)
                entry_confidence = trade_data.get('entry_confidence', 0.0)
                exit_confidence = trade_data.get('exit_confidence', 0.0) # 🆕 exit_confidence 추가
                entry_price = trade_data.get('entry_price', 0.0)
                exit_price = trade_data.get('exit_price', 0.0)
                holding_duration = trade_data.get('holding_duration', 0)
                action = trade_data.get('action', 'sell')
                
                # 컬럼 존재 여부 확인 (동적 대응)
                cursor.execute("PRAGMA table_info(virtual_trade_feedback)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # 기본 쿼리 구성
                cols = ['coin', 'entry_timestamp', 'exit_timestamp', 'profit_loss_pct', 'signal_pattern', 'processed_at']
                vals = [
                    trade_data['coin'], trade_data['entry_timestamp'], trade_data['exit_timestamp'], 
                    trade_data['profit_loss_pct'], signal_pattern, int(time.time())
                ]
                
                # 추가 컬럼 동적 바인딩
                if 'entry_signal_score' in columns:
                    cols.append('entry_signal_score')
                    vals.append(entry_signal_score)

                if 'entry_confidence' in columns:
                    cols.append('entry_confidence')
                    vals.append(entry_confidence)

                if 'exit_signal_score' in columns:
                    cols.append('exit_signal_score')
                    vals.append(exit_signal_score)

                if 'exit_confidence' in columns: # 🆕 exit_confidence 컬럼 처리
                    cols.append('exit_confidence')
                    vals.append(exit_confidence)
                
                if 'entry_price' in columns:
                    cols.append('entry_price')
                    vals.append(entry_price)
                    
                if 'exit_price' in columns:
                    cols.append('exit_price')
                    vals.append(exit_price)
                    
                if 'holding_duration' in columns:
                    cols.append('holding_duration')
                    vals.append(holding_duration)
                    
                if 'action' in columns:
                    cols.append('action')
                    vals.append(action)
                
                placeholders = ', '.join(['?' for _ in cols])
                columns_str = ', '.join(cols)
                
                query = f"INSERT INTO virtual_trade_feedback ({columns_str}) VALUES ({placeholders})"
                
                cursor.execute(query, vals)
                conn.commit()

        # 🆕 [AI Learning Log] 학습 로그 기록 (코인명 포함)
            profit = trade_data['profit_loss_pct']
            coin = trade_data.get('coin', 'Unknown')
            result_str = "성공" if profit > 0 else "실패"
            
            # 🆕 [실제 수행 결과 반영] 
            # 실제 매매 결과가 긍정적이었다면 "실제 경험을 통한 확신"을 로그에 남김
            exp_msg = ""
            if profit > 5.0:
                 exp_msg = " (🚀 대박 실전 경험!)"
            elif profit < -5.0:
                 exp_msg = " (😭 뼈아픈 실전 교훈...)"

            log_msg = f"[{coin}] 매매 복기 완료: {result_str} ({profit:+.2f}%) → 패턴 학습 업데이트{exp_msg}"
            
            self.log_system_event("INFO", "Learner", log_msg, {
                "pattern": signal_pattern,
                "profit": profit
            })
                
        except Exception as e:
            # 🚨 에러 무시하지 않고 출력 (원인 파악용)
            print(f"⚠️ 학습 내역 저장 실패: {e}")

    def _execute_system_evolution(self):
        """시스템 진화 실행"""
        try:
            # 1시간마다 실행
            current_time = int(time.time())
            if current_time - self.last_batch_process < 3600:
                return

            # 피드백 요약
            summary = self.feedback_processor.get_feedback_summary()
            
            # 진화 엔진 실행
            if summary['total_trades'] > 10:
                self.evolution_engine.evolve_system(summary)
                
            self.last_batch_process = current_time
            
        except Exception as e:
            print(f"⚠️ 시스템 진화 실행 오류: {e}")
    
    def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            current_time = int(time.time())
            if current_time - self.last_cache_cleanup < 3600: # 1시간마다
                return
            
            # 캐시 정리
            self.feedback_cache = {}
            
            # 사후 평가기 정리 (자동으로 됨)
            
            self.last_cache_cleanup = current_time
            print("🧹 데이터 정리 완료")
            
        except Exception as e:
            print(f"⚠️ 데이터 정리 오류: {e}")

    def _update_realtime_executor_data(self):
        """실전 매매 실행기용 데이터 업데이트 (파일 등 공유)"""
        # 여기서는 파일 시스템이나 DB를 통해 실전 매매 봇이 읽을 수 있는 형태로 데이터를 내보내는 로직
        # 이미 DB(STRATEGY_DB_PATH)에 저장하고 있으므로, 실전 봇이 거기서 읽으면 됨.
        pass

    def print_learning_status(self):
        """학습 상태 출력 (변동 사항이 있거나 1시간 경과 시 출력)"""
        try:
            current_time = int(time.time())
            # 1시간마다 또는 강제 출력 필요 시에만 출력
            if not hasattr(self, '_last_status_print'):
                self._last_status_print = 0
            
            if current_time - self._last_status_print < 3600:
                return
            
            stats = self.thompson_sampler.get_exploration_stats()
            print(f"\\n📊 [학습 상태] 패턴: {stats['total_patterns']}개, 샘플: {stats['total_samples']}회")
            print(f"   탐색: {stats['exploration_patterns']}개 ({stats['exploration_ratio']:.1%}), 활용: {stats['exploitation_patterns']}개")
            
            self._last_status_print = current_time
            
        except Exception as e:
            print(f"⚠️ 상태 출력 오류: {e}")

    def run_once(self):
        """1회 학습 실행 (run_trading.py 등에서 호출용)"""
        print("🚀 가상매매 학습 (1회 실행) 시작")
        try:
            # 1. 실시간 학습
            self._execute_real_time_learning()
            
            # 2. 시장 통찰 학습 (놓친 기회) & 사후 평가
            try:
                # 현재가 조회 (DB에서 최신 캔들로 대체)
                current_prices = {}
                
                # 안전하게 DB 경로 확인 (환경변수 우선)
                # 🚀 [Fix] trade_candles.db 경로를 명시적으로 탐색하여 설정
                _current_dir = os.path.dirname(os.path.abspath(__file__))
                _root_dir = os.path.dirname(os.path.dirname(_current_dir))
                _trade_candles_path = os.path.join(_root_dir, 'market', 'coin_market', 'data_storage', 'trade_candles.db')
                
                candles_db_path = os.environ.get('RL_DB_PATH')
                
                if not candles_db_path:
                    if os.path.exists(_trade_candles_path):
                        candles_db_path = _trade_candles_path
                    else:
                        candles_db_path = DB_PATH # fallback to default
                
                if os.path.exists(candles_db_path):
                    # print(f"📊 [DEBUG] 현재가 조회 DB: {candles_db_path}")
                    with sqlite3.connect(candles_db_path, timeout=60.0) as conn:
                        # 🆕 캔들 DB의 최신 timestamp 기준으로 조회 (시간대 문제 해결)
                        cursor = conn.cursor()
                        cursor.execute("SELECT MAX(timestamp) FROM candles")
                        max_ts = cursor.fetchone()[0] or int(time.time())
                        
                        # 최신 캔들 기준 60분 이내 데이터 조회
                        df_prices = pd.read_sql("""
                            SELECT symbol as coin, close FROM candles 
                            WHERE timestamp >= ?
                            GROUP BY symbol
                        """, conn, params=(max_ts - 3600,))
                        
                        for _, row in df_prices.iterrows():
                            current_prices[row['coin']] = row['close']
                else:
                    print(f"⚠️ 캔들 DB 파일을 찾을 수 없습니다: {candles_db_path}")
                
                if current_prices:
                    # 🚀 MFE/MAE 기반 사후 평가 수행 (진행 중인 추적 업데이트)
                    completed = self.post_trade_evaluator.check_evaluations(current_prices)
                    
                    # 🆕 [성능 최적화] 쌓인 패널티 일괄 DB 업데이트
                    self.post_trade_evaluator.flush_penalties()
                    
                    if completed:
                        print(f"✅ {len(completed)}건의 거래 사후 평가 완료 (MFE/MAE 분석)")
                    
                    # 시장 통찰 (놓친 기회)
                    self.market_miner.mine_insights(current_prices)
                else:
                    print("⚠️ 현재가 데이터를 조회할 수 없어 사후 평가를 건너뜁니다.")
                    
            except Exception as e:
                print(f"⚠️ 시장 데이터 처리 오류: {e}")

            # 3. 시스템 진화
            self._execute_system_evolution()
            
            # 4. 정리
            self._cleanup_old_data()
            
            # 5. 상태 출력 (강제 출력)
            stats = self.thompson_sampler.get_exploration_stats()
            print(f"📊 [학습 상태] 패턴: {stats['total_patterns']}개, 샘플: {stats['total_samples']}회")
            
        except Exception as e:
            print(f"⚠️ 학습 실행 오류: {e}")
            
        print("✅ 가상매매 학습 완료")

    def run(self):
        """메인 학습 루프 (데몬 모드)"""
        print("🚀 가상매매 학습기 시작 (데몬 모드)")
        
        # Heartbeat 초기화
        last_heartbeat = time.time()
        
        while True:
            try:
                self.run_once()
                
                # 💓 생존 신고 (10분마다)
                if time.time() - last_heartbeat > 600:
                    print(f"💓 [생존신고] 학습기 정상 작동 중... (현재: {time.strftime('%H:%M:%S')})")
                    last_heartbeat = time.time()
                
                # 대기
                time.sleep(60) # 1분 대기
                
            except KeyboardInterrupt:
                print("🛑 학습기 종료")
                break
            except Exception as e:
                print(f"⚠️ 메인 루프 오류: {e}")
                time.sleep(60)

if __name__ == "__main__":
    learner = VirtualTradingLearner()
    # run_trading.py에서 호출할 때는 1회만 실행하고 종료해야 함
    # 데몬 모드가 필요한 경우 별도 인자나 환경변수로 처리 가능하지만, 
    # 현재 구조상 기본 동작을 run_once로 변경하는 것이 안전함
    if os.environ.get('LEARNER_DAEMON_MODE', 'false').lower() == 'true':
        learner.run()
    else:
        learner.run_once()
