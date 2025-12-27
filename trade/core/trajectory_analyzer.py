#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 Trajectory Analyzer - 수익률 추적 및 추세 분석 모듈

매수 이후부터 매도까지의 수익률 변화를 추적하고 분석합니다.

주요 기능:
1. 수익률 히스토리 기록 (매 사이클마다)
2. 추세 분석 (상승/하락/횡보 감지)
3. 고점 대비 하락(Drawdown) 계산
4. 연속 하락/상승 횟수 감지
5. 조기 매도/홀딩 신호 생성
6. 학습용 추세 패턴 제공
"""

import os
import sys
import sqlite3
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

# DB 경로 설정
_DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'market', 'coin_market', 'data_storage')
TRADING_SYSTEM_DB_PATH = os.getenv('TRADING_DB_PATH', os.path.join(os.getenv('DATA_STORAGE_PATH', _DEFAULT_DB_DIR), 'trading_system.db'))


class TrendType(Enum):
    """추세 유형"""
    STRONG_UP = "strong_up"       # 강한 상승
    UP = "up"                      # 상승
    SIDEWAYS = "sideways"          # 횡보
    DOWN = "down"                  # 하락
    STRONG_DOWN = "strong_down"   # 강한 하락
    RECOVERING = "recovering"      # 회복 중 (고점 대비 하락 후 반등)
    PEAK_REVERSAL = "peak_reversal" # 고점 반전 (고점 찍고 하락 시작)


@dataclass
class TrendAnalysis:
    """추세 분석 결과"""
    trend_type: TrendType           # 추세 유형
    consecutive_drops: int          # 연속 하락 횟수
    consecutive_rises: int          # 연속 상승 횟수
    max_profit_pct: float           # 최고 수익률
    current_profit_pct: float       # 현재 수익률
    drawdown_pct: float             # 고점 대비 하락률
    profit_velocity: float          # 수익률 변화 속도 (최근 기울기)
    profit_acceleration: float      # 수익률 변화 가속도
    should_sell_early: bool         # 조기 매도 권장 여부
    should_hold_strong: bool        # 강한 홀딩 권장 여부
    confidence: float               # 분석 신뢰도 (0-1)
    reason: str                     # 분석 사유
    history_count: int              # 히스토리 개수


class TrajectoryAnalyzer:
    """수익률 추적 및 추세 분석기"""
    
    def __init__(self, db_path: str = None, is_virtual: bool = True):
        """
        Args:
            db_path: DB 경로 (None이면 기본 경로 사용)
            is_virtual: 가상매매 여부 (테이블명 구분용)
        """
        self.db_path = db_path or TRADING_SYSTEM_DB_PATH
        self.is_virtual = is_virtual
        self.table_prefix = "virtual_" if is_virtual else "real_"
        self._ensure_tables()
    
    def _ensure_tables(self):
        """히스토리 테이블 생성"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                
                # 수익률 히스토리 테이블
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}profit_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        profit_pct REAL NOT NULL,
                        signal_score REAL DEFAULT 0.0,
                        current_price REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        max_profit_pct REAL DEFAULT 0.0,
                        min_profit_pct REAL DEFAULT 0.0,
                        holding_hours REAL DEFAULT 0.0,
                        market_regime TEXT DEFAULT 'neutral',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성 (빠른 조회용)
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}profit_history_coin_ts 
                    ON {self.table_prefix}profit_history(coin, timestamp DESC)
                """)
                
                # 추세 패턴 학습 테이블 (학습기용)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}trajectory_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        entry_timestamp INTEGER NOT NULL,
                        exit_timestamp INTEGER NOT NULL,
                        trajectory_type TEXT NOT NULL,
                        peak_profit_pct REAL NOT NULL,
                        final_profit_pct REAL NOT NULL,
                        peak_to_exit_drop REAL DEFAULT 0.0,
                        consecutive_drops_at_exit INTEGER DEFAULT 0,
                        total_samples INTEGER DEFAULT 0,
                        optimal_exit_timing TEXT,
                        pattern_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ Trajectory 테이블 생성 오류: {e}")
    
    def record_profit_snapshot(self, coin: str, profit_pct: float, current_price: float,
                                entry_price: float, signal_score: float = 0.0,
                                max_profit_pct: float = None, min_profit_pct: float = None,
                                holding_hours: float = 0.0, market_regime: str = 'neutral') -> bool:
        """
        현재 수익률 스냅샷 기록
        
        Args:
            coin: 코인 심볼
            profit_pct: 현재 수익률
            current_price: 현재 가격
            entry_price: 진입 가격
            signal_score: 시그널 점수
            max_profit_pct: 최대 수익률 (None이면 자동 계산)
            min_profit_pct: 최소 수익률 (None이면 자동 계산)
            holding_hours: 보유 시간
            market_regime: 시장 레짐
            
        Returns:
            성공 여부
        """
        try:
            timestamp = int(time.time())
            
            # 이전 히스토리에서 max/min 계산
            if max_profit_pct is None or min_profit_pct is None:
                prev_max, prev_min = self._get_prev_max_min(coin)
                if max_profit_pct is None:
                    max_profit_pct = max(profit_pct, prev_max)
                if min_profit_pct is None:
                    min_profit_pct = min(profit_pct, prev_min)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute(f"""
                    INSERT INTO {self.table_prefix}profit_history 
                    (coin, timestamp, profit_pct, signal_score, current_price, entry_price,
                     max_profit_pct, min_profit_pct, holding_hours, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (coin, timestamp, profit_pct, signal_score, current_price, entry_price,
                      max_profit_pct, min_profit_pct, holding_hours, market_regime))
                conn.commit()
            
            return True
            
        except Exception as e:
            print(f"⚠️ 수익률 스냅샷 기록 오류 ({coin}): {e}")
            return False
    
    def _get_prev_max_min(self, coin: str) -> Tuple[float, float]:
        """이전 히스토리에서 최대/최소 수익률 조회"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT MAX(max_profit_pct), MIN(min_profit_pct)
                    FROM {self.table_prefix}profit_history
                    WHERE coin = ?
                """, (coin,))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return row[0], row[1]
        except:
            pass
        return 0.0, 0.0
    
    def analyze_trend(self, coin: str, lookback: int = 10) -> TrendAnalysis:
        """
        수익률 추세 분석
        
        Args:
            coin: 코인 심볼
            lookback: 분석할 히스토리 개수 (최근 N개)
            
        Returns:
            TrendAnalysis 객체
        """
        try:
            history = self._get_recent_history(coin, lookback)
            
            if len(history) < 2:
                return TrendAnalysis(
                    trend_type=TrendType.SIDEWAYS,
                    consecutive_drops=0,
                    consecutive_rises=0,
                    max_profit_pct=history[0]['max_profit_pct'] if history else 0.0,
                    current_profit_pct=history[0]['profit_pct'] if history else 0.0,
                    drawdown_pct=0.0,
                    profit_velocity=0.0,
                    profit_acceleration=0.0,
                    should_sell_early=False,
                    should_hold_strong=False,
                    confidence=0.3,
                    reason="히스토리 부족",
                    history_count=len(history)
                )
            
            # 수익률 시계열 추출 (최신 → 과거 순으로 저장되어 있으므로 역순)
            profits = [h['profit_pct'] for h in reversed(history)]
            max_profit = max(h['max_profit_pct'] for h in history)
            current_profit = history[0]['profit_pct']  # 가장 최신
            
            # 연속 하락/상승 횟수 계산
            consecutive_drops = self._count_consecutive_changes(profits, direction='down')
            consecutive_rises = self._count_consecutive_changes(profits, direction='up')
            
            # 고점 대비 하락률
            drawdown_pct = max_profit - current_profit if max_profit > 0 else 0.0
            
            # 변화 속도 (기울기) 계산 - 선형 회귀
            velocity = self._calculate_velocity(profits)
            
            # 가속도 계산 (속도의 변화)
            acceleration = self._calculate_acceleration(profits)
            
            # 추세 유형 결정
            trend_type, reason = self._determine_trend_type(
                profits, velocity, acceleration, drawdown_pct, max_profit, current_profit
            )
            
            # 조기 매도/강한 홀딩 판단
            should_sell_early, should_hold_strong, decision_reason = self._make_trade_recommendation(
                trend_type, consecutive_drops, consecutive_rises, 
                drawdown_pct, max_profit, current_profit, velocity
            )
            
            # 신뢰도 계산 (히스토리 개수 기반)
            confidence = min(1.0, len(history) / 10)
            
            return TrendAnalysis(
                trend_type=trend_type,
                consecutive_drops=consecutive_drops,
                consecutive_rises=consecutive_rises,
                max_profit_pct=max_profit,
                current_profit_pct=current_profit,
                drawdown_pct=drawdown_pct,
                profit_velocity=velocity,
                profit_acceleration=acceleration,
                should_sell_early=should_sell_early,
                should_hold_strong=should_hold_strong,
                confidence=confidence,
                reason=decision_reason or reason,
                history_count=len(history)
            )
            
        except Exception as e:
            print(f"⚠️ 추세 분석 오류 ({coin}): {e}")
            return TrendAnalysis(
                trend_type=TrendType.SIDEWAYS,
                consecutive_drops=0,
                consecutive_rises=0,
                max_profit_pct=0.0,
                current_profit_pct=0.0,
                drawdown_pct=0.0,
                profit_velocity=0.0,
                profit_acceleration=0.0,
                should_sell_early=False,
                should_hold_strong=False,
                confidence=0.0,
                reason=f"분석 오류: {e}",
                history_count=0
            )
    
    def _get_recent_history(self, coin: str, limit: int) -> List[Dict]:
        """최근 히스토리 조회 (최신순)"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_prefix}profit_history
                    WHERE coin = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (coin, limit))
                return [dict(row) for row in cursor.fetchall()]
        except:
            return []
    
    def _count_consecutive_changes(self, profits: List[float], direction: str = 'down') -> int:
        """연속 변화 횟수 계산 (가장 최근부터)"""
        if len(profits) < 2:
            return 0
        
        count = 0
        for i in range(len(profits) - 1, 0, -1):
            diff = profits[i] - profits[i-1]
            
            if direction == 'down' and diff < -0.1:  # 0.1% 이상 하락
                count += 1
            elif direction == 'up' and diff > 0.1:   # 0.1% 이상 상승
                count += 1
            else:
                break  # 연속성 끊김
        
        return count
    
    def _calculate_velocity(self, profits: List[float]) -> float:
        """수익률 변화 속도 (기울기) 계산"""
        if len(profits) < 2:
            return 0.0
        
        # 간단한 선형 회귀 기울기
        x = np.arange(len(profits))
        y = np.array(profits)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def _calculate_acceleration(self, profits: List[float]) -> float:
        """수익률 변화 가속도 계산"""
        if len(profits) < 3:
            return 0.0
        
        # 1차 미분 (속도)
        velocities = np.diff(profits)
        
        # 2차 미분 (가속도)
        if len(velocities) < 2:
            return 0.0
        
        accelerations = np.diff(velocities)
        return float(np.mean(accelerations))
    
    def _determine_trend_type(self, profits: List[float], velocity: float, 
                               acceleration: float, drawdown: float,
                               max_profit: float, current_profit: float) -> Tuple[TrendType, str]:
        """추세 유형 결정"""
        
        # 고점 반전 감지 (고점 찍고 하락 시작)
        if max_profit > 5.0 and drawdown > 2.0 and velocity < -0.2:
            return TrendType.PEAK_REVERSAL, f"고점 {max_profit:.1f}%에서 {drawdown:.1f}% 하락 중"
        
        # 강한 하락
        if velocity < -0.5 or (drawdown > 3.0 and velocity < -0.2):
            return TrendType.STRONG_DOWN, f"급락 (속도: {velocity:.2f}%/샘플)"
        
        # 하락
        if velocity < -0.15:
            return TrendType.DOWN, f"하락 추세 (속도: {velocity:.2f}%/샘플)"
        
        # 회복 중 (고점 대비 하락 후 반등)
        if drawdown > 2.0 and velocity > 0.1 and acceleration > 0:
            return TrendType.RECOVERING, f"회복 중 (고점 대비 -{drawdown:.1f}%, 속도: +{velocity:.2f}%)"
        
        # 강한 상승
        if velocity > 0.5:
            return TrendType.STRONG_UP, f"급등 (속도: +{velocity:.2f}%/샘플)"
        
        # 상승
        if velocity > 0.15:
            return TrendType.UP, f"상승 추세 (속도: +{velocity:.2f}%/샘플)"
        
        # 횡보
        return TrendType.SIDEWAYS, f"횡보 (속도: {velocity:.2f}%/샘플)"
    
    def _make_trade_recommendation(self, trend_type: TrendType, 
                                    consecutive_drops: int, consecutive_rises: int,
                                    drawdown: float, max_profit: float, 
                                    current_profit: float, velocity: float) -> Tuple[bool, bool, str]:
        """
        매매 권장 사항 결정
        
        Returns:
            (should_sell_early, should_hold_strong, reason)
        """
        
        # 🔴 조기 매도 권장 조건
        
        # 1. 고점 반전: 5% 이상 수익 후 2% 이상 하락
        if trend_type == TrendType.PEAK_REVERSAL:
            return True, False, f"🔴 고점 반전! ({max_profit:.1f}% → {current_profit:.1f}%)"
        
        # 2. 연속 3회 이상 하락
        if consecutive_drops >= 3:
            return True, False, f"🔴 연속 {consecutive_drops}회 하락"
        
        # 3. 강한 하락 추세
        if trend_type == TrendType.STRONG_DOWN and drawdown > 1.5:
            return True, False, f"🔴 급락 중 (고점 대비 -{drawdown:.1f}%)"
        
        # 4. 고점 대비 큰 하락 (이미 수익이 있었는데 많이 반납)
        if max_profit > 8.0 and drawdown > 5.0:
            return True, False, f"🔴 수익 대량 반납 ({max_profit:.1f}% → {current_profit:.1f}%)"
        
        # 🟢 강한 홀딩 권장 조건
        
        # 1. 강한 상승 추세
        if trend_type == TrendType.STRONG_UP:
            return False, True, f"🟢 강한 상승 중! (속도: +{velocity:.2f}%)"
        
        # 2. 연속 상승
        if consecutive_rises >= 3:
            return False, True, f"🟢 연속 {consecutive_rises}회 상승"
        
        # 3. 회복 중 (하락 후 반등)
        if trend_type == TrendType.RECOVERING:
            return False, True, f"🟢 회복 중 (반등 속도: +{velocity:.2f}%)"
        
        # 4. 상승 추세 유지
        if trend_type == TrendType.UP:
            return False, True, f"🟢 상승 추세 유지"
        
        # 🟡 횡보 전략: 고점에서 매도, 저점에서 홀딩/매수 (슬리피지 고려)
        if trend_type == TrendType.SIDEWAYS:
            # 🆕 슬리피지 고려: 거래 비용 (수수료 0.1% + 슬리피지 0.05%) * 2 (매수+매도) = 약 0.3%
            # 최소 순수익: 0.5% 이상 필요 (안전 마진 포함)
            MIN_NET_PROFIT = 0.5  # 최소 순수익 0.5%
            MIN_RANGE = 1.5  # 최소 변동폭 1.5% (고점-저점 차이)
            
            # 횡보 범위 계산 (고점 - 저점)
            range_size = max_profit - (current_profit - drawdown) if drawdown > 0 else max_profit
            
            # 🆕 최소 변동폭 체크: 범위가 너무 작으면 거래하지 않음
            if range_size < MIN_RANGE:
                return False, False, f"🟡 횡보 범위 부족 ({range_size:.1f}% < {MIN_RANGE}%) - 거래 비용 고려하여 홀딩"
            
            # 고점 근처 판단: 현재 수익률이 최고점의 70% 이상이면 매도 고려
            if max_profit > MIN_NET_PROFIT * 2:  # 최소 순수익의 2배 이상 수익이 있었던 경우만
                profit_ratio = current_profit / max_profit if max_profit > 0 else 0
                
                # 🆕 고점 근처 (최고점의 70% 이상) + 최소 순수익 확보 가능: 매도 고려
                # 현재 수익률이 최소 순수익(0.5%) 이상이고, 고점의 70% 이상이면 매도
                if profit_ratio >= 0.7 and current_profit >= MIN_NET_PROFIT:
                    # 고점과 현재의 차이가 충분한지 확인 (최소 0.3% 이상 차이)
                    profit_from_peak = max_profit - current_profit
                    if profit_from_peak <= 0.3:  # 고점과 너무 가까우면 아직 기다림
                        return False, False, f"🟡 횡보 고점 근처 대기 ({current_profit:.1f}% / 최고 {max_profit:.1f}%, 차이: {profit_from_peak:.1f}%)"
                    return True, False, f"🟡 횡보 고점 근처 ({current_profit:.1f}% / 최고 {max_profit:.1f}%) - 매도 고려 (순수익: {current_profit - MIN_NET_PROFIT:.1f}%)"
                
                # 🆕 저점 근처 (최고점 대비 30% 이하 또는 손실) + 하락 여지 충분: 홀딩/추매 고려
                # 저점에서 매수할 경우 최소 순수익을 낼 수 있는지 확인
                elif profit_ratio <= 0.3 or current_profit < 0:
                    # 저점에서 매수 후 고점에서 매도 시 예상 순수익 계산
                    potential_profit = max_profit - (current_profit - drawdown) if drawdown > 0 else max_profit - current_profit
                    if potential_profit >= MIN_NET_PROFIT * 2:  # 최소 순수익의 2배 이상 가능하면 추매 고려
                        return False, True, f"🟡 횡보 저점 근처 ({current_profit:.1f}% / 최고 {max_profit:.1f}%) - 홀딩/추매 고려 (예상수익: {potential_profit:.1f}%)"
                    else:
                        return False, False, f"🟡 횡보 저점 근처 ({current_profit:.1f}% / 최고 {max_profit:.1f}%) - 수익 여지 부족 (예상: {potential_profit:.1f}%)"
            
            # 횡보 중간 구간: 중립
            return False, False, f"🟡 횡보 중 ({current_profit:.1f}%, 범위: {max_profit:.1f}% ~ {current_profit - drawdown:.1f}%, 폭: {range_size:.1f}%)"
        
        # 중립 (조건 미해당)
        return False, False, "⚪ 중립 (추세 불명확)"
    
    def clear_coin_history(self, coin: str):
        """특정 코인의 히스토리 삭제 (매도 완료 시 호출)"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute(f"""
                    DELETE FROM {self.table_prefix}profit_history
                    WHERE coin = ?
                """, (coin,))
                conn.commit()
        except Exception as e:
            print(f"⚠️ 히스토리 삭제 오류 ({coin}): {e}")
    
    def save_trajectory_pattern(self, coin: str, entry_timestamp: int, exit_timestamp: int,
                                 peak_profit: float, final_profit: float,
                                 trajectory_type: str, pattern_data: Dict = None,
                                 include_full_history: bool = False):
        """
        거래 완료 시 추세 패턴 저장 (학습용)
        
        Args:
            coin: 코인 심볼
            entry_timestamp: 진입 시점
            exit_timestamp: 청산 시점
            peak_profit: 최고 수익률
            final_profit: 최종 수익률
            trajectory_type: 추세 유형
            pattern_data: 추가 패턴 데이터 (JSON)
            include_full_history: 전체 히스토리 포함 여부 (학습용)
        """
        try:
            peak_to_exit_drop = peak_profit - final_profit
            
            # 최종 분석 결과 조회
            analysis = self.analyze_trend(coin, lookback=20)
            
            # 🆕 전체 히스토리 포함 (학습용)
            full_history = None
            if include_full_history:
                full_history = self.get_coin_full_history(coin)
            
            # 패턴 데이터에 전체 히스토리 추가
            combined_pattern_data = pattern_data or {}
            if full_history:
                combined_pattern_data['full_history'] = full_history
                combined_pattern_data['history_count'] = len(full_history)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # 🆕 컬럼 존재 확인 및 추가
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({self.table_prefix}trajectory_patterns)")
                cols = [c[1] for c in cursor.fetchall()]
                if 'is_learned' not in cols:
                    try:
                        cursor.execute(f"ALTER TABLE {self.table_prefix}trajectory_patterns ADD COLUMN is_learned INTEGER DEFAULT 0")
                    except: pass
                
                conn.execute(f"""
                    INSERT INTO {self.table_prefix}trajectory_patterns
                    (coin, entry_timestamp, exit_timestamp, trajectory_type, peak_profit_pct, 
                     final_profit_pct, peak_to_exit_drop, consecutive_drops_at_exit, 
                     total_samples, optimal_exit_timing, pattern_json, is_learned)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    coin, entry_timestamp, exit_timestamp, trajectory_type,
                    peak_profit, final_profit, peak_to_exit_drop,
                    analysis.consecutive_drops, analysis.history_count,
                    self._determine_optimal_exit_timing(peak_profit, final_profit, peak_to_exit_drop),
                    json.dumps(combined_pattern_data, ensure_ascii=False)
                ))
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 추세 패턴 저장 오류: {e}")
    
    def _determine_optimal_exit_timing(self, peak: float, final: float, drop: float) -> str:
        """최적 청산 타이밍 판단"""
        if drop > 3.0 and peak > 5.0:
            return "peak"  # 고점에서 청산했어야 함
        elif drop > 1.5 and peak > 3.0:
            return "early_drop"  # 하락 초기에 청산했어야 함
        elif final > peak * 0.8:
            return "optimal"  # 최적 타이밍
        else:
            return "late"  # 너무 늦게 청산
    
    def get_trajectory_learning_data(self, limit: int = 1000) -> List[Dict]:
        """학습용 추세 패턴 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_prefix}trajectory_patterns
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except:
            return []
    
    def get_coin_full_history(self, coin: str) -> List[Dict]:
        """특정 코인의 전체 히스토리 조회 (학습용)"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_prefix}profit_history
                    WHERE coin = ?
                    ORDER BY timestamp ASC
                """, (coin,))
                return [dict(row) for row in cursor.fetchall()]
        except:
            return []
    
    def get_unlearned_patterns(self, limit: int = 100) -> List[Dict]:
        """🆕 아직 학습되지 않은 패턴 조회 (학습기용)"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # is_learned 컬럼 존재 확인
                cursor.execute(f"PRAGMA table_info({self.table_prefix}trajectory_patterns)")
                cols = [c[1] for c in cursor.fetchall()]
                
                if 'is_learned' in cols:
                    cursor.execute(f"""
                        SELECT * FROM {self.table_prefix}trajectory_patterns
                        WHERE is_learned = 0
                        ORDER BY exit_timestamp ASC
                        LIMIT ?
                    """, (limit,))
                else:
                    cursor.execute(f"""
                        SELECT * FROM {self.table_prefix}trajectory_patterns
                        ORDER BY exit_timestamp ASC
                        LIMIT ?
                    """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
        except:
            return []
    
    def mark_pattern_as_learned(self, pattern_id: int):
        """🆕 패턴을 학습 완료로 표시"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute(f"""
                    UPDATE {self.table_prefix}trajectory_patterns
                    SET is_learned = 1
                    WHERE id = ?
                """, (pattern_id,))
                conn.commit()
        except Exception as e:
            print(f"⚠️ 패턴 학습 표시 오류: {e}")
    
    def cleanup_learned_coin_history(self, coin: str):
        """🆕 학습 완료된 코인의 히스토리 삭제"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # 해당 코인의 미학습 패턴이 있는지 확인
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({self.table_prefix}trajectory_patterns)")
                cols = [c[1] for c in cursor.fetchall()]
                
                has_unlearned = False
                if 'is_learned' in cols:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {self.table_prefix}trajectory_patterns
                        WHERE coin = ? AND is_learned = 0
                    """, (coin,))
                    has_unlearned = cursor.fetchone()[0] > 0
                
                # 미학습 패턴이 없으면 히스토리 삭제
                if not has_unlearned:
                    cursor.execute(f"""
                        DELETE FROM {self.table_prefix}profit_history
                        WHERE coin = ?
                    """, (coin,))
                    conn.commit()
                    
        except Exception as e:
            print(f"⚠️ 학습 후 히스토리 삭제 오류 ({coin}): {e}")
    
    def cleanup_old_data(self, days: int = 30):
        """오래된 히스토리 데이터 정리"""
        try:
            cutoff = int(time.time()) - (days * 24 * 3600)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # 히스토리 테이블 정리
                cursor = conn.execute(f"""
                    DELETE FROM {self.table_prefix}profit_history
                    WHERE timestamp < ?
                """, (cutoff,))
                history_deleted = cursor.rowcount
                
                # 패턴 테이블은 더 오래 보관 (90일)
                pattern_cutoff = int(time.time()) - (90 * 24 * 3600)
                cursor = conn.execute(f"""
                    DELETE FROM {self.table_prefix}trajectory_patterns
                    WHERE exit_timestamp < ?
                """, (pattern_cutoff,))
                pattern_deleted = cursor.rowcount
                
                conn.commit()
                
                if history_deleted > 0 or pattern_deleted > 0:
                    print(f"🧹 오래된 데이터 정리: 히스토리 {history_deleted}개, 패턴 {pattern_deleted}개 삭제")
                    
        except Exception as e:
            print(f"⚠️ 데이터 정리 오류: {e}")


# 싱글톤 인스턴스 (편의용)
_virtual_analyzer = None
_real_analyzer = None


def get_virtual_trajectory_analyzer() -> TrajectoryAnalyzer:
    """가상매매용 분석기 인스턴스 반환"""
    global _virtual_analyzer
    if _virtual_analyzer is None:
        _virtual_analyzer = TrajectoryAnalyzer(is_virtual=True)
    return _virtual_analyzer


def get_real_trajectory_analyzer() -> TrajectoryAnalyzer:
    """실전매매용 분석기 인스턴스 반환"""
    global _real_analyzer
    if _real_analyzer is None:
        _real_analyzer = TrajectoryAnalyzer(is_virtual=False)
    return _real_analyzer

