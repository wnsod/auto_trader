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
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data_storage', 'realtime_candles.db')
# 🆕 통합 트레이딩 시스템 DB 경로 (섀도우 + 실전 매매) - 통일된 경로
TRADING_SYSTEM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_storage', 'trading_system.db')

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
    coin: str
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

class PatternAnalyzer:
    """패턴 분석기 - 거래 패턴 분석 및 개선점 도출"""
    def __init__(self):
        self.pattern_database = {}
        self.success_patterns = {}
        self.failure_patterns = {}
        
    def get_pattern_performance(self) -> dict:
        """패턴별 성과 반환 (DB에서 최신 데이터 로드)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
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
        """시그널 패턴 추출"""
        try:
            # RSI 범주화
            rsi = trade_data.get('rsi', 50.0)
            rsi_level = self._discretize_rsi(rsi)
            
            # MACD 범주화
            macd = trade_data.get('macd', 0.0)
            macd_level = self._discretize_macd(macd)
            
            # 볼륨 범주화
            volume_ratio = trade_data.get('volume_ratio', 1.0)
            volume_level = self._discretize_volume(volume_ratio)
            
            # 패턴 조합
            pattern = f"{rsi_level}_{macd_level}_{volume_level}"
            
            return pattern
            
        except Exception as e:
            print(f"⚠️ 시그널 패턴 추출 오류: {e}")
            return 'unknown_pattern'
    
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
            # 피드백 데이터 준비
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
            
            # 피드백 큐에 추가
            self.feedback_queue.append(feedback_data)
            
            # 처리된 피드백 저장
            feedback_id = f"{feedback_data['coin']}_{feedback_data['entry_timestamp']}"
            self.processed_feedback[feedback_id] = feedback_data
            
            print(f"📊 피드백 처리: {feedback_data['coin']} 패턴 {feedback_data['signal_pattern']}")
            
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
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
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
                with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
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

# 🚫 RL 학습 클래스 제거됨 - 순수 피드백 제공자로 변경

class VirtualTradingLearner:
    """가상매매 순수 피드백 제공자 (증분 학습 시스템)"""
    
    def __init__(self):
        print("🚀 최적화된 피드백 처리 시스템 초기화 중...")
        
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
        
        # 🆕 진화형 AI 시스템 초기화
        self.real_time_learner = RealTimeLearner()
        self.pattern_analyzer = PatternAnalyzer()
        self.feedback_processor = FeedbackProcessor()
        self.evolution_engine = EvolutionEngine()
        
        # 🆕 시그널-매매 연결 시스템
        self.signal_trade_connector = SignalTradeConnector()
        
        print(f"📊 진화형 AI 피드백 처리 설정:")
        print(f"  📦 배치 크기: {self.batch_size}개 (증가)")
        print(f"  ⏱️ 처리 시간 제한: {self.max_processing_time}초")
        print(f"  🚀 캐시 시스템: 활성화")
        print(f"  📦 배치 처리: 활성화")
        
        # 테이블 생성
        self.create_learning_tables()
        
        print("✅ 피드백 처리 시스템 초기화 완료!")
    
    def create_learning_tables(self):
        """학습 관련 테이블 생성"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 완료된 거래 추적 테이블 (중복 학습 방지)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS completed_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        entry_timestamp INTEGER NOT NULL,
                        exit_timestamp INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        profit_loss_pct REAL NOT NULL,
                        action TEXT NOT NULL,
                        holding_duration INTEGER NOT NULL,
                        is_learned BOOLEAN DEFAULT FALSE,
                        learning_episode INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, entry_timestamp, exit_timestamp)
                    )
                """)
                
                # 가상매매 성과 통계 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_performance_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        total_trades INTEGER NOT NULL,
                        successful_trades INTEGER NOT NULL,
                        failed_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_episodes INTEGER NOT NULL,
                        successful_episodes INTEGER NOT NULL,
                        failed_episodes INTEGER NOT NULL,
                        episode_win_rate REAL NOT NULL,
                        avg_episode_profit REAL NOT NULL,
                        epsilon REAL NOT NULL,
                        q_table_size INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 🆕 시그널 피드백 점수 테이블 (변동성 그룹별 분리)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        signal_pattern TEXT NOT NULL,
                        volatility_group TEXT NOT NULL DEFAULT 'MEDIUM',
                        success_rate REAL,
                        avg_profit REAL,
                        total_trades INTEGER,
                        confidence REAL,
                        updated_at INTEGER,
                        PRIMARY KEY (signal_pattern, volatility_group)
                    )
                """)

                # 🆕 변동성 그룹 컬럼 마이그레이션 (기존 테이블에 컬럼 추가)
                try:
                    conn.execute("""
                        ALTER TABLE signal_feedback_scores
                        ADD COLUMN volatility_group TEXT DEFAULT 'MEDIUM'
                    """)
                except sqlite3.OperationalError:
                    # 컬럼이 이미 존재하면 무시
                    pass

                # 🆕 가상매매 피드백 데이터 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_trade_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        entry_timestamp INTEGER NOT NULL,
                        exit_timestamp INTEGER NOT NULL,
                        entry_signal_score REAL NOT NULL,
                        exit_signal_score REAL NOT NULL,
                        entry_confidence REAL NOT NULL,
                        exit_confidence REAL NOT NULL,
                        profit_loss_pct REAL NOT NULL,
                        holding_duration INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        is_learned BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # print("✅ 학습 테이블 생성 완료")  # 제거됨
                
        except Exception as e:
            # print(f"⚠️ 학습 테이블 생성 오류: {e}")  # 제거됨
            pass
    
    def load_signal_from_db(self, coin: str, timestamp: int) -> Optional[SignalInfo]:
        """DB에서 시그널 정보 로드"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql("""
                    SELECT * FROM signals 
                    WHERE coin = ? AND timestamp = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, conn, params=(coin, timestamp))
                
                if df.empty:
                    return None
                
                row = df.iloc[0]
                
                # SignalInfo 객체 생성 (모든 고급 지표 포함)
                signal = SignalInfo(
                    coin=row['coin'],
                    interval=row['interval'],
                    action=SignalAction(row['action']),
                    signal_score=safe_float(row['signal_score']),
                    confidence=safe_float(row['confidence']),
                    reason=safe_str(row['reason']),
                    timestamp=safe_int(row['timestamp']),
                    price=safe_float(row['price']),
                    volume=safe_float(row['volume']),
                    rsi=safe_float(row['rsi']),
                    macd=safe_float(row['macd']),
                    wave_phase=safe_str(row['wave_phase']),
                    pattern_type=safe_str(row['pattern_type']),
                    risk_level=safe_str(row['risk_level']),
                    volatility=safe_float(row['volatility']),
                    volume_ratio=safe_float(row['volume_ratio']),
                    wave_progress=safe_float(row['wave_progress']),
                    structure_score=safe_float(row['structure_score']),
                    pattern_confidence=safe_float(row['pattern_confidence']),
                    integrated_direction=safe_str(row['integrated_direction']),
                    integrated_strength=safe_float(row['integrated_strength']),
                    # 🆕 고급 지표들
                    mfi=safe_float(row.get('mfi', 50.0)),
                    atr=safe_float(row.get('atr', 0.0)),
                    adx=safe_float(row.get('adx', 25.0)),
                    ma20=safe_float(row.get('ma20', 0.0)),
                    rsi_ema=safe_float(row.get('rsi_ema', 50.0)),
                    macd_smoothed=safe_float(row.get('macd_smoothed', 0.0)),
                    wave_momentum=safe_float(row.get('wave_momentum', 0.0)),
                    bb_position=safe_str(row.get('bb_position', 'unknown')),
                    bb_width=safe_float(row.get('bb_width', 0.0)),
                    bb_squeeze=safe_float(row.get('bb_squeeze', 0.0)),
                    rsi_divergence=safe_str(row.get('rsi_divergence', 'none')),
                    macd_divergence=safe_str(row.get('macd_divergence', 'none')),
                    volume_divergence=safe_str(row.get('volume_divergence', 'none')),
                    price_momentum=safe_float(row.get('price_momentum', 0.0)),
                    volume_momentum=safe_float(row.get('volume_momentum', 0.0)),
                    trend_strength=safe_float(row.get('trend_strength', 0.5)),
                    support_resistance=safe_str(row.get('support_resistance', 'unknown')),
                    fibonacci_levels=safe_str(row.get('fibonacci_levels', 'unknown')),
                    elliott_wave=safe_str(row.get('elliott_wave', 'unknown')),
                    harmonic_patterns=safe_str(row.get('harmonic_patterns', 'none')),
                    candlestick_patterns=safe_str(row.get('candlestick_patterns', 'none')),
                    market_structure=safe_str(row.get('market_structure', 'unknown')),
                    flow_level_meta=safe_str(row.get('flow_level_meta', 'unknown')),
                    pattern_direction=safe_str(row.get('pattern_direction', 'neutral'))
                )
                
                return signal
                
        except Exception as e:
            print(f"⚠️ 시그널 로드 오류 ({coin}): {e}")
            return None
    
    def save_learning_history(self, signal: SignalInfo, action: str, profit_loss_pct: float, 
                            holding_duration: int):
        """학습 히스토리 저장 (피드백 데이터용)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO virtual_trade_feedback 
                    (coin, entry_timestamp, exit_timestamp, entry_signal_score, exit_signal_score,
                     entry_confidence, exit_confidence, profit_loss_pct, holding_duration, action)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.coin,
                    signal.timestamp,
                    int(datetime.now().timestamp()),
                    signal.signal_score,
                    signal.signal_score,  # 간단히 동일한 값 사용
                    signal.confidence,
                    signal.confidence,  # 간단히 동일한 값 사용
                    profit_loss_pct,
                    holding_duration,
                    action
                ))
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 학습 히스토리 저장 오류: {e}")
    
    def print_learning_status(self):
        """학습 상태 출력"""
        try:
            stats = self.get_completed_trades_stats()
            
            print(f"🧠 가상매매 피드백 상태:")
            print(f"  📊 총 완료된 거래: {stats['total_trades']}개")
            print(f"  ✅ 성공 거래: {stats['successful_trades']}개")
            print(f"  ❌ 실패 거래: {stats['failed_trades']}개")
            print(f"  🎯 승률: {stats['win_rate']:.1f}%")
            print(f"  📈 평균 수익률: {stats['avg_profit']:.2f}%")
            
        except Exception as e:
            print(f"⚠️ 학습 상태 출력 오류: {e}")
    
    def get_learning_range_info(self) -> Dict:
        """학습 범위 정보 반환"""
        return {
            'max_hours_back': self.max_hours_back,
            'batch_size': self.batch_size,
            'max_processing_time': self.max_processing_time,
            'realtime_max_hours': self.realtime_max_hours,
            'realtime_batch_size': self.realtime_batch_size,
            'realtime_max_time': self.realtime_max_time
        }
    
    def start_learning(self):
        """한 번 실행되는 피드백 처리 (run_trade_pipeline.py에서 반복 호출)"""
        print("🧠 피드백 처리 시작 (run_trade_pipeline.py에서 반복 호출)")
        
        try:
            # 피드백 처리 실행
            self.process_feedback()
            
            # 성과 통계 출력
            self.print_learning_status()
            
            print("✅ 피드백 처리 완료")
            
        except Exception as e:
            print(f"⚠️ 피드백 처리 오류: {e}")
    

    
    def save_completed_trade(self, coin: str, entry_timestamp: int, exit_timestamp: int, 
                           entry_price: float, exit_price: float, profit_loss_pct: float, 
                           action: str, holding_duration: int) -> bool:
        """완료된 거래 저장"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO completed_trades 
                    (coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                     profit_loss_pct, action, holding_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                      profit_loss_pct, action, holding_duration))
                conn.commit()
                return True
                
        except Exception as e:
            print(f"⚠️ 완료된 거래 저장 오류: {e}")
            return False
    
    def mark_trade_as_learned(self, coin: str, entry_timestamp: int, exit_timestamp: int, 
                            learning_episode: int) -> bool:
        """거래를 학습 완료로 표시"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    UPDATE completed_trades 
                    SET is_learned = TRUE, learning_episode = ?
                    WHERE coin = ? AND entry_timestamp = ? AND exit_timestamp = ?
                """, (learning_episode, coin, entry_timestamp, exit_timestamp))
                conn.commit()
                return True
                
        except Exception as e:
            print(f"⚠️ 거래 학습 완료 표시 오류: {e}")
            return False
    
    def get_unlearned_completed_trades(self, max_hours_back: int = None, batch_size: int = 50) -> List[Dict]:
        """미학습 완료된 거래 조회 (시간 범위 제한 없음)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 시간 범위 제한 제거 - 모든 미학습 거래 조회
                df = pd.read_sql("""
                    SELECT 
                        coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                        profit_loss_pct, action, holding_duration
                    FROM completed_trades 
                    WHERE is_learned = FALSE
                    ORDER BY exit_timestamp DESC
                    LIMIT ?
                """, conn, params=(batch_size,))
                
                trades = []
                for _, row in df.iterrows():
                    # 🆕 수익률과 보유시간 재계산
                    profit_loss_pct = safe_float(row['profit_loss_pct'])
                    holding_duration = safe_int(row['holding_duration'])
                    
                    trade = {
                        'coin': row['coin'],
                        'entry_timestamp': safe_int(row['entry_timestamp']),
                        'exit_timestamp': safe_int(row['exit_timestamp']),
                        'entry_price': safe_float(row['entry_price']),
                        'exit_price': safe_float(row['exit_price']),
                        'profit_loss_pct': profit_loss_pct,
                        'action': row['action'],
                        'holding_duration': holding_duration
                    }
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            print(f"⚠️ 미학습 거래 조회 오류: {e}")
            return []
    
    def load_trade_feedback_data(self, max_hours_back: int = None, batch_size: int = 50) -> List[Dict]:
        """거래 피드백 데이터 로드 (시간 범위 제한 없음)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 시간 범위 제한 제거 - 모든 미학습 피드백 데이터 조회
                df = pd.read_sql("""
                    SELECT 
                        coin, entry_timestamp, exit_timestamp, entry_signal_score, exit_signal_score,
                        entry_confidence, exit_confidence, profit_loss_pct, holding_duration, action
                    FROM virtual_trade_feedback 
                    WHERE is_learned = FALSE
                    ORDER BY exit_timestamp DESC
                    LIMIT ?
                """, conn, params=(batch_size,))
                
                feedback_data = []
                for _, row in df.iterrows():
                    feedback = {
                        'coin': row['coin'],
                        'entry_timestamp': safe_int(row['entry_timestamp']),
                        'exit_timestamp': safe_int(row['exit_timestamp']),
                        'entry_signal_score': safe_float(row['entry_signal_score']),
                        'exit_signal_score': safe_float(row['exit_signal_score']),
                        'entry_confidence': safe_float(row['entry_confidence']),
                        'exit_confidence': safe_float(row['exit_confidence']),
                        'profit_loss_pct': safe_float(row['profit_loss_pct']),
                        'holding_duration': safe_int(row['holding_duration']),
                        'action': row['action']
                    }
                    feedback_data.append(feedback)
                
                return feedback_data
                
        except Exception as e:
            print(f"⚠️ 거래 피드백 데이터 로드 오류: {e}")
            return []
    
    def is_trade_already_learned(self, coin: str, entry_timestamp: int, exit_timestamp: int) -> bool:
        """거래가 이미 학습되었는지 확인"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                df = pd.read_sql("""
                    SELECT is_learned FROM completed_trades 
                    WHERE coin = ? AND entry_timestamp = ? AND exit_timestamp = ?
                """, conn, params=(coin, entry_timestamp, exit_timestamp))
                
                if not df.empty:
                    return bool(df.iloc[0]['is_learned'])
                return False
                
        except Exception as e:
            print(f"⚠️ 거래 학습 상태 확인 오류: {e}")
            return False
    
    def get_nearest_candle(self, coin: str, interval: str, base_timestamp: int) -> Optional[pd.Series]:
        """가장 가까운 캔들 데이터 조회"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # 기준 타임스탬프 전후 1시간 범위에서 검색
                time_range = 3600  # 1시간
                start_time = base_timestamp - time_range
                end_time = base_timestamp + time_range
                
                df = pd.read_sql("""
                    SELECT * FROM candles 
                    WHERE coin = ? AND interval = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY ABS(timestamp - ?) ASC
                    LIMIT 1
                """, conn, params=(coin, interval, start_time, end_time, base_timestamp))
                
                if not df.empty:
                    return df.iloc[0]
                return None
                
        except Exception as e:
            print(f"⚠️ 캔들 데이터 조회 오류 ({coin}/{interval}): {e}")
            return None
    
    def get_multi_interval_state_key(self, coin: str, base_timestamp: int) -> str:
        """멀티인터벌 상태 키 생성"""
        try:
            intervals = ['15m', '30m', '240m', '1d']
            state_parts = [coin]
            
            for interval in intervals:
                candle = self.get_nearest_candle(coin, interval, base_timestamp)
                if candle is not None:
                    # 간단한 상태 표현
                    rsi = safe_float(candle.get('rsi', 50))
                    macd = safe_float(candle.get('macd', 0))
                    volume_ratio = safe_float(candle.get('volume_ratio', 1))
                    
                    rsi_state = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
                    macd_state = 'bullish' if macd > 0 else 'bearish'
                    volume_state = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.8 else 'normal'
                    
                    interval_state = f"{interval}_{rsi_state}_{macd_state}_{volume_state}"
                    state_parts.append(interval_state)
                else:
                    state_parts.append(f"{interval}_unknown")
            
            return "_".join(state_parts)
            
        except Exception as e:
            print(f"⚠️ 멀티인터벌 상태 키 생성 오류: {e}")
            return f"{coin}_unknown_state"
    
    def get_state_representation(self, candle: pd.Series, interval: str) -> str:
        """캔들 데이터를 상태 표현으로 변환"""
        try:
            # 안전한 타입 변환
            def safe_float(value, default=0.0):
                try:
                    if value is None or pd.isna(value):
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # 기본 지표들
            rsi = safe_float(candle.get('rsi', 50))
            macd = safe_float(candle.get('macd', 0))
            volume_ratio = safe_float(candle.get('volume_ratio', 1))
            wave_progress = safe_float(candle.get('wave_progress', 0.5))
            volatility = safe_float(candle.get('volatility', 0.03))
            structure_score = safe_float(candle.get('structure_score', 0.5))
            pattern_confidence = safe_float(candle.get('pattern_confidence', 0.5))
            
            # 이산화
            rsi_state = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            macd_state = 'bullish' if macd > 0 else 'bearish'
            volume_state = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.8 else 'normal'
            wave_state = 'late' if wave_progress > 0.7 else 'early' if wave_progress < 0.3 else 'mid'
            risk_state = 'high' if volatility > 0.05 else 'low' if volatility < 0.02 else 'mid'
            structure_state = 'high' if structure_score > 0.7 else 'low' if structure_score < 0.3 else 'mid'
            confidence_state = 'high' if pattern_confidence > 0.7 else 'low' if pattern_confidence < 0.3 else 'mid'
            
            # 상태 키 생성
            state_parts = [
                interval,
                rsi_state, macd_state, volume_state, wave_state, risk_state,
                structure_state, confidence_state
            ]
            
            return "_".join(state_parts)
            
        except Exception as e:
            print(f"⚠️ 상태 표현 생성 오류: {e}")
            return f"{interval}_unknown_state"
    
    def process_feedback(self):
        """🧬 진화형 AI 피드백 처리 (증분 학습 + 실시간 학습 + 진화 + 실전매매 연동)"""
        try:
            print("🧬 진화형 AI 피드백 처리 시작...")
            start_time = time.time()
            
            # 🆕 증분 학습 체크포인트 로드
            self._load_learning_checkpoint()
            
            # 🚀 1. 증분 학습: 새로운 거래만 처리
            if self.incremental_learning:
                new_trades = self._get_incremental_trades()
                if new_trades:
                    print(f"📊 증분 학습: 새로운 거래 {len(new_trades)}개 처리")
                    processed_count = self._process_trades_with_ai(new_trades, start_time)
                    print(f"✅ 증분 학습 완료: {processed_count}개")
                    
                    # 학습 체크포인트 업데이트
                    self._update_learning_checkpoint()
                else:
                    print("ℹ️ 증분 학습: 새로운 거래가 없습니다.")
            else:
                # 🚀 기존 방식: 완료된 거래 피드백 처리 (배치)
                unlearned_trades = self.get_unlearned_completed_trades(
                    batch_size=self.batch_size
                )
                
                if unlearned_trades:
                    print(f"📊 처리할 완료된 거래: {len(unlearned_trades)}개")
                    
                    # 🆕 진화형 AI 배치 처리
                    processed_count = self._process_trades_with_ai(unlearned_trades, start_time)
                    print(f"✅ 진화형 AI 거래 피드백 처리: {processed_count}개")
                else:
                    print("ℹ️ 처리할 완료된 거래가 없습니다.")
            
            # 🚀 2. 가상매매 피드백 데이터 처리 (배치)
            feedback_data = self.load_trade_feedback_data(
                batch_size=self.batch_size
            )
            
            if feedback_data:
                print(f"📊 처리할 가상매매 피드백 데이터: {len(feedback_data)}개")
                
                # 🆕 진화형 AI 시그널 성능 분석
                self._analyze_signal_performance_with_ai(feedback_data)
                
                print(f"✅ 가상매매 피드백 데이터 처리 완료")
            else:
                print("ℹ️ 처리할 가상매매 피드백 데이터가 없습니다.")
            
            # 🆕 3. 실시간 학습 실행
            self._execute_real_time_learning()
            
            # 🆕 4. 시스템 진화 실행
            self._execute_system_evolution()
            
            # 🆕 5. 데이터 정리 (주기적)
            self._cleanup_old_data()
            
            # 🆕 6. 실전매매 연동 데이터 업데이트
            self._update_realtime_executor_data()
            
            # 7. 성과 통계 출력
            self.print_learning_status()
            
        except Exception as e:
            print(f"⚠️ 진화형 AI 피드백 처리 오류: {e}")
    
    def _load_learning_checkpoint(self):
        """학습 체크포인트 로드"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                cursor = conn.cursor()
                
                # 학습 체크포인트 테이블 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_checkpoint (
                        id INTEGER PRIMARY KEY,
                        last_learning_timestamp INTEGER,
                        learning_episode INTEGER,
                        processed_trade_count INTEGER,
                        last_cleanup_timestamp INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 🆕 진화 결과 테이블 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evolution_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evolution_direction TEXT NOT NULL,
                        changes TEXT,
                        performance_trend TEXT,
                        win_rate REAL,
                        avg_profit REAL,
                        total_trades INTEGER,
                        created_at INTEGER NOT NULL
                    )
                """)
                
                # 최신 체크포인트 조회
                cursor.execute("""
                    SELECT last_learning_timestamp, learning_episode, processed_trade_count, last_cleanup_timestamp
                    FROM learning_checkpoint
                    ORDER BY id DESC LIMIT 1
                """)
                
                result = cursor.fetchone()
                if result:
                    self.last_learning_timestamp = result[0]
                    self.learning_episode = result[1]
                    # processed_trade_count는 별도로 관리
                    print(f"📊 학습 체크포인트 로드: 마지막 학습={self.last_learning_timestamp}, 에피소드={self.learning_episode}")
                else:
                    # 첫 실행인 경우
                    self.last_learning_timestamp = int(time.time()) - (self.max_hours_back * 3600)
                    self.learning_episode = 0
                    print("🆕 첫 학습 실행: 체크포인트 초기화")
                    
        except Exception as e:
            print(f"⚠️ 학습 체크포인트 로드 오류: {e}")
            self.last_learning_timestamp = int(time.time()) - (self.max_hours_back * 3600)
            self.learning_episode = 0
    
    def _get_incremental_trades(self):
        """증분 학습용 새로운 거래만 조회"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 마지막 학습 시점 이후의 새로운 거래만 조회
                query = """
                    SELECT * FROM completed_trades 
                    WHERE exit_timestamp > ? AND learning_episode IS NULL
                    ORDER BY exit_timestamp ASC
                    LIMIT ?
                """
                
                df = pd.read_sql(query, conn, params=(self.last_learning_timestamp, self.batch_size))
                
                if not df.empty:
                    print(f"📊 증분 학습 대상: {len(df)}개 새로운 거래 (마지막 학습: {datetime.fromtimestamp(self.last_learning_timestamp)})")
                
                return df.to_dict('records') if not df.empty else []
                
        except Exception as e:
            print(f"⚠️ 증분 거래 조회 오류: {e}")
            return []
    
    def _update_learning_checkpoint(self):
        """학습 체크포인트 업데이트"""
        try:
            current_time = int(time.time())
            self.learning_episode += 1
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                cursor = conn.cursor()
                
                # 체크포인트 저장
                cursor.execute("""
                    INSERT INTO learning_checkpoint 
                    (last_learning_timestamp, learning_episode, processed_trade_count, last_cleanup_timestamp)
                    VALUES (?, ?, ?, ?)
                """, (current_time, self.learning_episode, 0, current_time))
                
                conn.commit()
                print(f"✅ 학습 체크포인트 업데이트: 에피소드 {self.learning_episode}")
                
        except Exception as e:
            print(f"⚠️ 학습 체크포인트 업데이트 오류: {e}")
    
    def _cleanup_old_data(self):
        """오래된 데이터 정리 (주기적)"""
        try:
            current_time = int(time.time())
            
            # 24시간마다 정리 실행
            if current_time - self.last_learning_timestamp < 86400:
                return
            
            print("🧹 오래된 데이터 정리 시작...")
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                cursor = conn.cursor()
                
                # 7일 이상 된 완료된 거래 정리
                cleanup_timestamp = current_time - (7 * 86400)
                
                cursor.execute("""
                    DELETE FROM completed_trades 
                    WHERE exit_timestamp < ?
                """, (cleanup_timestamp,))
                
                deleted_trades = cursor.rowcount
                
                # 30일 이상 된 피드백 데이터 정리
                feedback_cleanup_timestamp = current_time - (30 * 86400)
                
                cursor.execute("""
                    DELETE FROM virtual_trade_feedback 
                    WHERE exit_timestamp < ?
                """, (feedback_cleanup_timestamp,))
                
                deleted_feedback = cursor.rowcount
                
                conn.commit()
                
                if deleted_trades > 0 or deleted_feedback > 0:
                    print(f"✅ 데이터 정리 완료: 거래 {deleted_trades}개, 피드백 {deleted_feedback}개 삭제")
                
        except Exception as e:
            print(f"⚠️ 데이터 정리 오류: {e}")
    
    def _process_trades_with_ai(self, trades: List[Dict], start_time: float) -> int:
        """🆕 진화형 AI로 거래 배치 처리"""
        try:
            processed_count = 0
            
            for trade in trades:
                # 처리 시간 체크
                if time.time() - start_time > self.max_processing_time:
                    print(f"⏰ 처리 시간 초과로 중단: {processed_count}개 처리 완료")
                    break
                
                try:
                    # 🆕 최근성 가중치로 거래 데이터 전처리
                    trade_with_timestamp = {
                        'timestamp': trade.get('entry_timestamp', time.time()),
                        'success': trade.get('profit_loss_pct', 0) > 0,
                        'profit': trade.get('profit_loss_pct', 0),
                        'action': trade.get('action', 'unknown'),
                        'coin': trade.get('coin', 'unknown'),
                        'signal_pattern': trade.get('signal_pattern', 'unknown')
                    }
                    
                    # 🆕 베이지안 스무딩 적용
                    pattern_stats = {
                        'success_rate': 1.0 if trade_with_timestamp['success'] else 0.0,
                        'avg_profit': trade_with_timestamp['profit'],
                        'total_trades': 1
                    }
                    smoothed_stats = self.bayesian_applier.apply_bayesian_smoothing(pattern_stats)
                    
                    # 🆕 이상치 컷 적용
                    robust_profit = self.outlier_applier.apply_outlier_guardrail([trade_with_timestamp['profit']])
                    
                    # 🆕 패턴 분석
                    pattern_analysis = self.pattern_analyzer.analyze_pattern(trade)
                    
                    # 🆕 피드백 처리
                    feedback_data = self.feedback_processor.process_feedback(trade)
                    
                    # 🆕 실시간 학습 (성능 업그레이드 적용)
                    signal_pattern = pattern_analysis.get('signal_pattern', 'unknown')
                    enhanced_trade_data = {
                        'trade_result': trade,
                        'pattern_analysis': pattern_analysis,
                        'feedback_result': feedback_data,
                        'smoothed_stats': smoothed_stats,
                        'robust_profit': robust_profit,
                        'recency_weight': self.recency_aggregator.exponential_decay.calculate_weight(
                            (time.time() - trade_with_timestamp['timestamp']) / 3600
                        )
                    }
                    self.real_time_learner.learn_from_trade(signal_pattern, enhanced_trade_data)
                    
                    # 거래를 학습 완료로 표시
                    self.mark_trade_as_learned(
                        trade['coin'],
                        trade['entry_timestamp'],
                        trade['exit_timestamp'],
                        processed_count + 1
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 거래 처리 오류 ({trade.get('coin', 'unknown')}): {e}")
                    continue
            
            return processed_count
            
        except Exception as e:
            print(f"⚠️ AI 거래 배치 처리 오류: {e}")
            return 0
    
    def _analyze_signal_performance_with_ai(self, feedback_data: List[Dict]):
        """🆕 진화형 AI로 시그널 성능 분석"""
        try:
            # 패턴별 성과 분석
            pattern_performance = {}
            
            for data in feedback_data:
                signal_pattern = data.get('signal_pattern', 'unknown')
                
                if signal_pattern not in pattern_performance:
                    pattern_performance[signal_pattern] = {
                        'total_trades': 0,
                        'successful_trades': 0,
                        'total_profit': 0.0,
                        'avg_profit': 0.0,
                        'success_rate': 0.0
                    }
                
                perf = pattern_performance[signal_pattern]
                perf['total_trades'] += 1
                perf['total_profit'] += data.get('profit_loss_pct', 0.0)
                
                if data.get('profit_loss_pct', 0.0) > 0:
                    perf['successful_trades'] += 1
            
            # 성과 계산
            for pattern, perf in pattern_performance.items():
                if perf['total_trades'] > 0:
                    perf['avg_profit'] = perf['total_profit'] / perf['total_trades']
                    perf['success_rate'] = perf['successful_trades'] / perf['total_trades']
            
            print(f"📊 진화형 AI 시그널 성과 분석 완료: {len(pattern_performance)}개 패턴")
            
        except Exception as e:
            print(f"⚠️ AI 시그널 성능 분석 오류: {e}")
    
    def _execute_real_time_learning(self):
        """🆕 실시간 학습 실행"""
        try:
            # 실시간 학습기에서 패턴 성과 업데이트
            for pattern, perf in self.real_time_learner.pattern_performance.items():
                if perf['total_trades'] > 0:
                    print(f"🧠 실시간 학습: {pattern} 패턴 성과 (성공률: {perf['success_rate']:.2f})")
            
        except Exception as e:
            print(f"⚠️ 실시간 학습 실행 오류: {e}")
    
    def _update_realtime_executor_data(self):
        """🆕 실전매매 연동 데이터 업데이트"""
        try:
            print("🔄 실전매매 연동 데이터 업데이트 중...")
            
            # 학습된 패턴 성능을 실전매매에 반영
            pattern_performance = self.pattern_analyzer.get_pattern_performance()
            
            # 시그널 피드백 점수를 실전매매에 반영
            signal_feedback = self.feedback_processor.get_feedback_summary()
            
            # 진화 결과를 실전매매에 반영
            evolution_result = self.evolution_engine.get_evolution_summary()
            
            print(f"✅ 실전매매 연동 데이터 업데이트 완료")
            print(f"   - 패턴 성능: {len(pattern_performance)}개")
            print(f"   - 시그널 피드백: {len(signal_feedback)}개")
            print(f"   - 진화 결과: {len(evolution_result)}개")
            
        except Exception as e:
            print(f"⚠️ 실전매매 연동 데이터 업데이트 오류: {e}")
    
    def _execute_system_evolution(self):
        """🆕 시스템 진화 실행"""
        try:
            # 피드백 요약 정보 수집
            feedback_summary = self.feedback_processor.get_feedback_summary()
            
            # 시스템 진화 실행
            evolution_result = self.evolution_engine.evolve_system(feedback_summary)
            
            if evolution_result:
                print(f"🧬 시스템 진화 완료: {evolution_result.get('direction', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ 시스템 진화 실행 오류: {e}")
    
    def _process_trades_batch(self, trades: List[Dict], start_time: float) -> int:
        """🚀 배치 거래 처리"""
        processed_count = 0
        batch_size = 20
        
        try:
            for i in range(0, len(trades), batch_size):
                batch = trades[i:i + batch_size]
                
                # 시간 제한 체크
                if time.time() - start_time > self.max_processing_time:
                    print(f"⏰ 시간 제한 도달 ({self.max_processing_time}초)")
                    break
                
                # 배치로 거래 처리
                with sqlite3.connect(self.db_path) as conn:
                    for trade in batch:
                        try:
                            # 거래를 학습 완료로 표시
                            conn.execute("""
                                UPDATE completed_trades 
                                SET learned = 1, learning_episode = 0
                                WHERE coin = ? AND entry_timestamp = ? AND exit_timestamp = ?
                            """, (trade['coin'], trade['entry_timestamp'], trade['exit_timestamp']))
                            
                            # 시그널 패턴 분석 및 피드백 점수 업데이트
                            self.update_signal_feedback_score(trade)
                            processed_count += 1
                            
                        except Exception as e:
                            print(f"⚠️ 거래 피드백 처리 오류 ({trade['coin']}): {e}")
                            continue
                    
                    conn.commit()
                
                # 진행률 출력
                if (i + batch_size) % 50 == 0:
                    print(f"  📈 진행률: {min(i + batch_size, len(trades))}/{len(trades)}")
            
        except Exception as e:
            print(f"⚠️ 배치 거래 처리 오류: {e}")
        
        return processed_count
    
    def _analyze_signal_performance_batch(self, feedback_data: List[Dict]):
        """🚀 배치 시그널 성능 분석"""
        try:
            # 시그널 패턴별로 그룹화
            pattern_groups = {}
            for trade in feedback_data:
                pattern = self.extract_signal_pattern_from_trade(trade)
                if pattern:
                    if pattern not in pattern_groups:
                        pattern_groups[pattern] = []
                    pattern_groups[pattern].append(trade)
            
            # 배치로 성능 분석
            with sqlite3.connect(self.db_path) as conn:
                for pattern, trades in pattern_groups.items():
                    try:
                        # 성능 지표 계산
                        profits = [trade['profit_loss_pct'] for trade in trades]
                        success_count = sum(1 for p in profits if p > 0)
                        success_rate = success_count / len(profits) if profits else 0
                        avg_profit = sum(profits) / len(profits) if profits else 0
                        
                        # 피드백 점수 업데이트
                        conn.execute("""
                            INSERT OR REPLACE INTO signal_feedback_scores 
                            (signal_pattern, success_rate, avg_profit, total_trades, confidence, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (pattern, success_rate, avg_profit, len(profits), success_rate, int(time.time())))
                        
                    except Exception as e:
                        print(f"⚠️ 패턴 {pattern} 분석 오류: {e}")
                        continue
                
                conn.commit()
            
        except Exception as e:
            print(f"⚠️ 배치 시그널 성능 분석 오류: {e}")
    
    def get_completed_trades_stats(self) -> Dict:
        """완료된 거래 통계 반환 (virtual_trade_history 테이블에서 조회)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 virtual_trade_history 테이블에서 전체 통계 조회
                df = pd.read_sql("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN profit_loss_pct > 0 THEN 1 ELSE 0 END) as successful_trades,
                        SUM(CASE WHEN profit_loss_pct <= 0 THEN 1 ELSE 0 END) as failed_trades,
                        AVG(profit_loss_pct) as avg_profit
                    FROM virtual_trade_history
                """, conn)
                
                if not df.empty and not pd.isna(df.iloc[0]['total_trades']):
                    row = df.iloc[0]
                    total_trades = safe_int(row['total_trades'])
                    successful_trades = safe_int(row['successful_trades']) if not pd.isna(row['successful_trades']) else 0
                    failed_trades = safe_int(row['failed_trades']) if not pd.isna(row['failed_trades']) else 0
                    avg_profit = safe_float(row['avg_profit']) if not pd.isna(row['avg_profit']) else 0.0
                    
                    win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0.0
                    
                    return {
                        'total_trades': total_trades,
                        'successful_trades': successful_trades,
                        'failed_trades': failed_trades,
                        'win_rate': win_rate,
                        'avg_profit': avg_profit
                    }
                
                return {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0
                }
                
        except Exception as e:
            print(f"⚠️ 완료된 거래 통계 조회 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0
            }
    
    def update_signal_feedback_score(self, trade: Dict):
        """거래 결과를 바탕으로 시그널 피드백 점수 업데이트"""
        try:
            # 시그널 패턴 추출
            signal_pattern = self.extract_signal_pattern_from_trade(trade)
            
            if not signal_pattern:
                return
            
            # 기존 피드백 점수 조회
            current_feedback = self.get_signal_feedback_score(signal_pattern)
            
            # 새로운 거래 결과 반영
            profit_loss_pct = trade['profit_loss_pct']
            is_success = profit_loss_pct > 0
            
            # 피드백 점수 업데이트
            if current_feedback:
                # 기존 데이터가 있는 경우 평균 계산
                total_trades = current_feedback['total_trades'] + 1
                success_rate = ((current_feedback['success_rate'] * current_feedback['total_trades']) + (1 if is_success else 0)) / total_trades
                avg_profit = ((current_feedback['avg_profit'] * current_feedback['total_trades']) + profit_loss_pct) / total_trades
                confidence = min(1.0, total_trades / 10.0)  # 최대 10개 거래 기준
            else:
                # 새로운 패턴
                total_trades = 1
                success_rate = 1.0 if is_success else 0.0
                avg_profit = profit_loss_pct
                confidence = 0.1  # 낮은 신뢰도로 시작
            
            # DB에 업데이트
            self.save_signal_feedback_score(signal_pattern, success_rate, avg_profit, total_trades, confidence)
            
        except Exception as e:
            print(f"⚠️ 시그널 피드백 점수 업데이트 오류: {e}")
    
    def extract_signal_pattern_from_trade(self, trade: Dict) -> Optional[str]:
        """거래에서 시그널 패턴 추출"""
        try:
            coin = trade['coin']
            entry_timestamp = trade['entry_timestamp']
            
            # 진입 시점의 시그널 정보 조회
            signal = self.load_signal_from_db(coin, entry_timestamp)
            
            if not signal:
                return None
            
            # 시그널 패턴 생성
            pattern_parts = [
                f"rsi_{self.discretize_rsi(signal.rsi)}",
                f"macd_{self.discretize_macd(signal.macd)}",
                f"volume_{self.discretize_volume(signal.volume_ratio)}",
                f"confidence_{self.discretize_confidence(signal.confidence)}",
                f"score_{self.discretize_score(signal.signal_score)}"
            ]
            
            return "_".join(pattern_parts)
            
        except Exception as e:
            print(f"⚠️ 시그널 패턴 추출 오류: {e}")
            return None
    
    def discretize_rsi(self, rsi: float) -> str:
        """RSI 이산화"""
        if rsi < 30:
            return "oversold"
        elif rsi > 70:
            return "overbought"
        else:
            return "neutral"
    
    def discretize_macd(self, macd: float) -> str:
        """MACD 이산화"""
        if macd > 0.01:
            return "strong_bullish"
        elif macd > 0:
            return "bullish"
        elif macd < -0.01:
            return "strong_bearish"
        else:
            return "bearish"
    
    def discretize_volume(self, volume_ratio: float) -> str:
        """거래량 이산화"""
        if volume_ratio > 2.0:
            return "very_high"
        elif volume_ratio > 1.5:
            return "high"
        elif volume_ratio < 0.5:
            return "low"
        else:
            return "normal"
    
    def discretize_confidence(self, confidence: float) -> str:
        """신뢰도 이산화"""
        if confidence > 0.8:
            return "very_high"
        elif confidence > 0.6:
            return "high"
        elif confidence < 0.4:
            return "low"
        else:
            return "medium"
    
    def discretize_score(self, score: float) -> str:
        """시그널 점수 이산화"""
        if score > 0.1:
            return "very_high"
        elif score > 0.05:
            return "high"
        elif score < 0.01:
            return "low"
        else:
            return "medium"
    
    def get_signal_feedback_score(self, signal_pattern: str) -> Optional[Dict]:
        """시그널 패턴의 피드백 점수 조회"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                df = pd.read_sql("""
                    SELECT success_rate, avg_profit, total_trades, confidence
                    FROM signal_feedback_scores 
                    WHERE signal_pattern = ?
                """, conn, params=(signal_pattern,))
                
                if not df.empty:
                    row = df.iloc[0]
                    return {
                        'success_rate': safe_float(row['success_rate']),
                        'avg_profit': safe_float(row['avg_profit']),
                        'total_trades': safe_int(row['total_trades']),
                        'confidence': safe_float(row['confidence'])
                    }
                return None
                
        except Exception as e:
            print(f"⚠️ 시그널 피드백 점수 조회 오류: {e}")
            return None
    
    def save_signal_feedback_score(self, signal_pattern: str, success_rate: float, 
                                 avg_profit: float, total_trades: int, confidence: float):
        """시그널 피드백 점수 저장"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO signal_feedback_scores 
                    (signal_pattern, success_rate, avg_profit, total_trades, confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (signal_pattern, success_rate, avg_profit, total_trades, confidence, int(time.time())))
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 시그널 피드백 점수 저장 오류: {e}")
    
    def analyze_signal_performance(self, feedback_data: List[Dict]):
        """시그널 성과 분석 및 개선점 도출"""
        try:
            if not feedback_data:
                return
            
            # 성과 통계 계산
            total_trades = len(feedback_data)
            successful_trades = sum(1 for f in feedback_data if f['profit_loss_pct'] > 0)
            avg_profit = sum(f['profit_loss_pct'] for f in feedback_data) / total_trades
            
            # 시그널 점수별 성과 분석
            score_performance = {}
            confidence_performance = {}
            
            for feedback in feedback_data:
                entry_score = feedback['entry_signal_score']
                entry_confidence = feedback['entry_confidence']
                profit = feedback['profit_loss_pct']
                
                # 점수 구간별 성과
                score_range = self.get_score_range(entry_score)
                if score_range not in score_performance:
                    score_performance[score_range] = {'trades': [], 'avg_profit': 0.0}
                score_performance[score_range]['trades'].append(profit)
                
                # 신뢰도 구간별 성과
                confidence_range = self.get_confidence_range(entry_confidence)
                if confidence_range not in confidence_performance:
                    confidence_performance[confidence_range] = {'trades': [], 'avg_profit': 0.0}
                confidence_performance[confidence_range]['trades'].append(profit)
            
            # 평균 수익률 계산
            for range_name, data in score_performance.items():
                if data['trades']:
                    data['avg_profit'] = sum(data['trades']) / len(data['trades'])
            
            for range_name, data in confidence_performance.items():
                if data['trades']:
                    data['avg_profit'] = sum(data['trades']) / len(data['trades'])
            
            # 개선점 출력
            print(f"📊 시그널 성과 분석 결과:")
            print(f"  📈 총 거래: {total_trades}개, 성공: {successful_trades}개, 승률: {successful_trades/total_trades*100:.1f}%")
            print(f"  💰 평균 수익률: {avg_profit:.2f}%")
            
            print(f"  🎯 점수별 성과:")
            for range_name, data in score_performance.items():
                print(f"    {range_name}: {data['avg_profit']:.2f}% ({len(data['trades'])}개)")
            
            print(f"  🔍 신뢰도별 성과:")
            for range_name, data in confidence_performance.items():
                print(f"    {range_name}: {data['avg_profit']:.2f}% ({len(data['trades'])}개)")
            
        except Exception as e:
            print(f"⚠️ 시그널 성과 분석 오류: {e}")
    
    def get_score_range(self, score: float) -> str:
        """시그널 점수 구간 분류"""
        if score > 0.1:
            return "very_high"
        elif score > 0.05:
            return "high"
        elif score > 0.02:
            return "medium"
        else:
            return "low"
    
    def get_confidence_range(self, confidence: float) -> str:
        """신뢰도 구간 분류"""
        if confidence > 0.8:
            return "very_high"
        elif confidence > 0.6:
            return "high"
        elif confidence > 0.4:
            return "medium"
        else:
            return "low"

    def analyze_multi_timeframe_signal_performance(self):
        """🚀 멀티 타임프레임 시그널 성과 분석 (시그널 계산 개선을 위한 피드백)"""
        try:
            print("🔄 멀티 타임프레임 시그널 성과 분석 시작")
            
            # 🎯 1. combined 시그널 vs 개별 인터벌 시그널 성과 비교
            combined_performance = self._analyze_combined_signal_performance()
            
            # 🎯 2. 인터벌별 시그널 성과 분석
            interval_performance = self._analyze_interval_signal_performance()
            
            # 🎯 3. 시그널 통합 방식별 성과 분석
            integration_performance = self._analyze_signal_integration_performance()
            
            # 🎯 4. 시그널 계산 개선 제안
            improvement_suggestions = self._generate_signal_improvement_suggestions(
                combined_performance, interval_performance, integration_performance
            )
            
            # 🎯 5. 결과 저장 및 요약
            self._save_multi_timeframe_analysis_results(
                combined_performance, interval_performance, integration_performance, improvement_suggestions
            )
            
            print("✅ 멀티 타임프레임 시그널 성과 분석 완료")
            return {
                'combined_performance': combined_performance,
                'interval_performance': interval_performance,
                'integration_performance': integration_performance,
                'improvement_suggestions': improvement_suggestions
            }
            
        except Exception as e:
            print(f"⚠️ 멀티 타임프레임 시그널 성과 분석 실패: {e}")
            return {}
    
    def _analyze_combined_signal_performance(self) -> Dict[str, Any]:
        """combined 시그널 성과 분석"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🎯 combined 시그널 기반 거래 성과 분석
                query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss_pct > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit_loss_pct) as avg_profit,
                    AVG(holding_duration) as avg_holding_duration,
                    AVG(entry_signal_score) as avg_signal_score
                FROM trade_feedback 
                WHERE signal_pattern LIKE '%combined%'
                AND exit_timestamp >= ?
                """
                
                # 최근 30일 데이터 분석
                thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
                df = pd.read_sql(query, conn, params=(thirty_days_ago,))
                
                if df.empty or df.iloc[0]['total_trades'] == 0:
                    return {'status': 'no_data', 'message': 'combined 시그널 거래 데이터가 없습니다'}
                
                row = df.iloc[0]
                win_rate = (row['winning_trades'] / row['total_trades']) * 100
                
                return {
                    'status': 'success',
                    'total_trades': int(row['total_trades']),
                    'winning_trades': int(row['winning_trades']),
                    'win_rate': float(win_rate),
                    'avg_profit': float(row['avg_profit']),
                    'avg_holding_duration': float(row['avg_holding_duration']),
                    'avg_signal_score': float(row['avg_signal_score'])
                }
                
        except Exception as e:
            print(f"⚠️ combined 시그널 성과 분석 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _analyze_interval_signal_performance(self) -> Dict[str, Any]:
        """인터벌별 시그널 성과 분석"""
        try:
            intervals = ['15m', '30m', '240m', '1d']
            interval_results = {}
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                for interval in intervals:
                    query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN profit_loss_pct > 0 THEN 1 ELSE 0 END) as winning_trades,
                        AVG(profit_loss_pct) as avg_profit,
                        AVG(holding_duration) as avg_holding_duration
                    FROM trade_feedback 
                    WHERE signal_pattern LIKE ?
                    AND exit_timestamp >= ?
                    """
                    
                    # 최근 30일 데이터 분석
                    thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
                    df = pd.read_sql(query, conn, params=(f'%{interval}%', thirty_days_ago))
                    
                    if not df.empty and df.iloc[0]['total_trades'] > 0:
                        row = df.iloc[0]
                        win_rate = (row['winning_trades'] / row['total_trades']) * 100
                        
                        interval_results[interval] = {
                            'total_trades': int(row['total_trades']),
                            'winning_trades': int(row['winning_trades']),
                            'win_rate': float(win_rate),
                            'avg_profit': float(row['avg_profit']),
                            'avg_holding_duration': float(row['avg_holding_duration'])
                        }
                    else:
                        interval_results[interval] = {'status': 'no_data'}
            
            return interval_results
            
        except Exception as e:
            print(f"⚠️ 인터벌별 시그널 성과 분석 실패: {e}")
            return {}
    
    def _analyze_signal_integration_performance(self) -> Dict[str, Any]:
        """시그널 통합 방식별 성과 분석"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🎯 투표 기반 vs 점수 기반 통합 방식 성과 비교
                query = """
                SELECT 
                    signal_pattern,
                    COUNT(*) as total_trades,
                    AVG(profit_loss_pct) as avg_profit,
                    AVG(CASE WHEN profit_loss_pct > 0 THEN 1 ELSE 0 END) as win_rate
                FROM trade_feedback 
                WHERE signal_pattern LIKE '%combined%'
                AND exit_timestamp >= ?
                GROUP BY signal_pattern
                ORDER BY avg_profit DESC
                """
                
                # 최근 30일 데이터 분석
                thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
                df = pd.read_sql(query, conn, params=(thirty_days_ago,))
                
                integration_results = {}
                for _, row in df.iterrows():
                    pattern = row['signal_pattern']
                    integration_results[pattern] = {
                        'total_trades': int(row['total_trades']),
                        'avg_profit': float(row['avg_profit']),
                        'win_rate': float(row['win_rate'] * 100)
                    }
                
                return integration_results
                
        except Exception as e:
            print(f"⚠️ 시그널 통합 방식별 성과 분석 실패: {e}")
            return {}
    
    def _generate_signal_improvement_suggestions(self, combined_perf: Dict, interval_perf: Dict, integration_perf: Dict) -> List[str]:
        """시그널 계산 개선 제안 생성"""
        suggestions = []
        
        try:
            # 🎯 1. combined 시그널 성과 분석 기반 제안
            if combined_perf.get('status') == 'success':
                if combined_perf.get('win_rate', 0) < 50:
                    suggestions.append("🚨 combined 시그널 승률이 낮습니다. 인터벌별 가중치 조정이 필요합니다.")
                
                if combined_perf.get('avg_profit', 0) < 0:
                    suggestions.append("📉 combined 시그널 평균 수익률이 음수입니다. 시그널 통합 로직 개선이 필요합니다.")
            
            # 🎯 2. 인터벌별 성과 분석 기반 제안
            best_interval = None
            best_performance = -999
            
            for interval, perf in interval_perf.items():
                if perf.get('status') != 'no_data':
                    performance = perf.get('avg_profit', 0) * perf.get('win_rate', 0)
                    if performance > best_performance:
                        best_performance = performance
                        best_interval = interval
            
            if best_interval:
                suggestions.append(f"🏆 {best_interval} 인터벌이 가장 좋은 성과를 보입니다. 이 인터벌의 가중치를 높이는 것을 고려하세요.")
            
            # 🎯 3. 시그널 통합 방식별 성과 분석 기반 제안
            if integration_perf:
                best_pattern = max(integration_perf.keys(), 
                                 key=lambda x: integration_perf[x].get('avg_profit', -999))
                
                if integration_perf[best_pattern].get('avg_profit', 0) > 0:
                    suggestions.append(f"💡 '{best_pattern}' 패턴이 가장 좋은 성과를 보입니다. 이 패턴을 더 자주 사용하도록 조정하세요.")
            
            # 🎯 4. 일반적인 개선 제안
            suggestions.extend([
                "🔧 시그널 점수 임계값을 동적으로 조정하여 시장 상황에 적응하도록 개선하세요.",
                "📊 멀티 타임프레임 시그널 통합 시 시장 변동성을 고려한 가중치 조정이 필요합니다.",
                "🧠 AI 모델의 적응성 점수를 시그널 통합에 더 적극적으로 활용하세요.",
                "⏰ 시장 상황별로 다른 인터벌 조합을 사용하는 적응적 전략을 고려하세요."
            ])
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️ 개선 제안 생성 실패: {e}")
            return ["⚠️ 개선 제안 생성 중 오류가 발생했습니다."]
    
    def _save_multi_timeframe_analysis_results(self, combined_perf: Dict, interval_perf: Dict, 
                                             integration_perf: Dict, suggestions: List[str]):
        """멀티 타임프레임 분석 결과 저장"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🎯 분석 결과 테이블 생성
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS multi_timeframe_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        combined_performance TEXT,
                        interval_performance TEXT,
                        integration_performance TEXT,
                        improvement_suggestions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 🎯 결과 저장
                conn.execute("""
                    INSERT INTO multi_timeframe_analysis 
                    (combined_performance, interval_performance, integration_performance, improvement_suggestions)
                    VALUES (?, ?, ?, ?)
                """, (
                    json.dumps(combined_perf, ensure_ascii=False),
                    json.dumps(interval_perf, ensure_ascii=False),
                    json.dumps(integration_perf, ensure_ascii=False),
                    json.dumps(suggestions, ensure_ascii=False)
                ))
                
                conn.commit()
                print("✅ 멀티 타임프레임 분석 결과 저장 완료")
                
        except Exception as e:
            print(f"⚠️ 분석 결과 저장 실패: {e}")
    
    def get_signal_improvement_recommendations(self) -> List[str]:
        """시그널 계산 개선을 위한 구체적 권장사항 조회"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🎯 최신 분석 결과 조회
                query = """
                SELECT improvement_suggestions 
                FROM multi_timeframe_analysis 
                ORDER BY analysis_date DESC 
                LIMIT 1
                """
                
                result = conn.execute(query).fetchone()
                
                if result and result[0]:
                    suggestions = json.loads(result[0])
                    return suggestions
                else:
                    return ["📊 아직 멀티 타임프레임 분석이 수행되지 않았습니다. analyze_multi_timeframe_signal_performance()를 실행하세요."]
                    
        except Exception as e:
            print(f"⚠️ 개선 권장사항 조회 실패: {e}")
            return ["⚠️ 개선 권장사항 조회 중 오류가 발생했습니다."]

def main():
    """🚀 메인 실행 함수 - 멀티 타임프레임 시그널 성과 분석 포함"""
    print("🆕 가상매매 학습기 시작")
    
    # 학습기 초기화
    learner = VirtualTradingLearner()
    
    try:
        print("\n🚀 [STEP 1] 기존 학습 데이터 분석")
        learner.print_learning_status()
        
        print("\n🚀 [STEP 2] 멀티 타임프레임 시그널 성과 분석")
        print("=" * 60)
        
        # 🎯 멀티 타임프레임 시그널 성과 분석 실행
        analysis_results = learner.analyze_multi_timeframe_signal_performance()
        
        if analysis_results:
            print("\n📊 멀티 타임프레임 시그널 성과 분석 결과:")
            print("-" * 40)
            
            # 🎯 1. Combined 시그널 성과
            combined_perf = analysis_results.get('combined_performance', {})
            if combined_perf.get('status') == 'success':
                print(f"🎯 Combined 시그널 성과:")
                print(f"  - 총 거래: {combined_perf.get('total_trades', 0)}회")
                print(f"  - 승률: {combined_perf.get('win_rate', 0):.1f}%")
                print(f"  - 평균 수익률: {combined_perf.get('avg_profit', 0):+.2f}%")
                print(f"  - 평균 보유시간: {combined_perf.get('avg_holding_duration', 0)/3600:.1f}시간")
                print(f"  - 평균 시그널 점수: {combined_perf.get('avg_signal_score', 0):.3f}")
            else:
                print(f"⚠️ Combined 시그널 성과: {combined_perf.get('message', '데이터 없음')}")
            
            # 🎯 2. 인터벌별 성과
            interval_perf = analysis_results.get('interval_performance', {})
            print(f"\n📊 인터벌별 시그널 성과:")
            for interval, perf in interval_perf.items():
                if perf.get('status') != 'no_data':
                    print(f"  - {interval}: 거래 {perf.get('total_trades', 0)}회, 승률 {perf.get('win_rate', 0):.1f}%, 수익률 {perf.get('avg_profit', 0):+.2f}%")
                else:
                    print(f"  - {interval}: 데이터 없음")
            
            # 🎯 3. 시그널 통합 방식별 성과
            integration_perf = analysis_results.get('integration_performance', {})
            if integration_perf:
                print(f"\n🔧 시그널 통합 방식별 성과:")
                for pattern, perf in integration_perf.items():
                    print(f"  - {pattern}: 거래 {perf.get('total_trades', 0)}회, 승률 {perf.get('win_rate', 0):.1f}%, 수익률 {perf.get('avg_profit', 0):+.2f}%")
            
            # 🎯 4. 개선 제안
            suggestions = analysis_results.get('improvement_suggestions', [])
            if suggestions:
                print(f"\n💡 시그널 계산 개선 제안:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
            
            print("\n" + "=" * 60)
            
            # 🎯 5. 개선 권장사항 조회
            print("\n🚀 [STEP 3] 시그널 계산 개선 권장사항")
            recommendations = learner.get_signal_improvement_recommendations()
            
            if recommendations:
                print("📋 최신 개선 권장사항:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
        else:
            print("⚠️ 멀티 타임프레임 시그널 성과 분석 실패")
        
        print("\n🚀 [STEP 4] 기존 학습 피드백 처리")
        learner.process_feedback()
        
        print("\n🚀 [STEP 5] 최종 학습 상태 확인")
        learner.print_learning_status()
        
    except Exception as e:
        print(f"⚠️ 메인 실행 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ 가상매매 학습기 완료!")

if __name__ == "__main__":
    main() 