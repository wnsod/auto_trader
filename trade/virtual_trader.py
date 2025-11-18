"""
가상매매 시뮬레이터 - 시그널 기반 가상 거래 실행

주요 기능:
1. 시그널 셀렉터에서 생성된 시그널을 실시간으로 읽기
2. 가상매매 시뮬레이션으로 거래 실행 (매수/매도/홀딩/익절/손절)
3. 포지션 관리 및 손익 계산
4. 거래 결과를 DB에 저장하여 학습기에서 활용
5. 실시간 포트폴리오 모니터링

🆕 Absolute Zero System 개선사항 반영:
- 모든 고급 기술지표 활용 (다이버전스, 볼린저밴드 스퀴즈, 모멘텀, 트렌드 강도 등)
- 개선된 시그널 정보 구조 (새로운 고급 지표들 포함)
- 향상된 상태 표현 (더 정교한 상태 키 생성)
- 새로운 패턴 매칭 로직 (다이버전스, 스퀴즈, 강한 트렌드 등)
"""
import sys
sys.path.insert(0, '/workspace/')

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import traceback
import time
import threading
from queue import Queue
import signal
import os
from market_name_utils import get_korean_name

# 데이터베이스 경로
DB_PATH = "/workspace/data_storage/realtime_candles.db"
# 🆕 통합 트레이딩 시스템 DB 경로 (섀도우 + 실전 매매) - 통일된 경로
TRADING_SYSTEM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_storage', 'trading_system.db')

class SignalAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"

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
    # 🆕 Absolute Zero System의 새로운 고급 지표들 (기본값으로 설정)
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

class ActionPerformanceTracker:
    """액션별 성과 추적기"""
    def __init__(self):
        self.action_performance = {
            'buy': {'trades': 0, 'wins': 0, 'total_profit': 0.0},
            'sell': {'trades': 0, 'wins': 0, 'total_profit': 0.0},
            'hold': {'trades': 0, 'wins': 0, 'total_profit': 0.0}
        }
    
    def record_action_result(self, action: str, profit: float, success: bool):
        """액션 결과 기록"""
        if action in self.action_performance:
            self.action_performance[action]['trades'] += 1
            self.action_performance[action]['total_profit'] += profit
            if success:
                self.action_performance[action]['wins'] += 1
    
    def get_action_performance(self, action: str) -> dict:
        """액션별 성과 반환"""
        if action not in self.action_performance:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        perf = self.action_performance[action]
        if perf['trades'] == 0:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        return {
            'success_rate': perf['wins'] / perf['trades'],
            'avg_profit': perf['total_profit'] / perf['trades'],
            'total_trades': perf['trades']
        }

class ContextRecorder:
    """컨텍스트 기록기"""
    def __init__(self):
        self.trade_contexts = {}
    
    def record_trade_context(self, trade_id: str, context: dict):
        """거래 컨텍스트 기록"""
        self.trade_contexts[trade_id] = {
            'timestamp': time.time(),
            'context': context
        }
    
    def get_trade_context(self, trade_id: str) -> dict:
        """거래 컨텍스트 조회"""
        return self.trade_contexts.get(trade_id, {})

class OutlierGuardrailApplier:
    """이상치 컷 적용기"""
    def __init__(self):
        self.outlier_guardrail = OutlierGuardrail()
    
    def apply_outlier_guardrail(self, profits: List[float]) -> float:
        """이상치 컷 적용"""
        return self.outlier_guardrail.calculate_robust_avg_profit(profits)

# 🆕 진화형 AI 시스템 클래스들
class AIDecisionEngine:
    """AI 의사결정 엔진 - 지능형 거래 결정"""
    def __init__(self):
        self.decision_history = []
        self.pattern_recognition = {}
        self.market_memory = {}
        
    def make_trading_decision(self, signal: SignalInfo, current_price: float, market_context: dict) -> str:
        """지능형 거래 결정"""
        try:
            # 패턴 인식 기반 결정
            pattern_score = self._analyze_pattern(signal)
            
            # 시장 맥락 기반 결정
            context_score = self._analyze_market_context(market_context)
            
            # 리스크 평가
            risk_score = self._evaluate_risk(signal, current_price)
            
            # 최종 결정
            decision = self._make_final_decision(pattern_score, context_score, risk_score, signal)
            
            # 결정 기록
            self.decision_history.append({
                'timestamp': time.time(),
                'signal': signal,
                'decision': decision,
                'scores': {
                    'pattern': pattern_score,
                    'context': context_score,
                    'risk': risk_score
                }
            })
            
            return decision
            
        except Exception as e:
            print(f"⚠️ AI 의사결정 오류: {e}")
            return 'HOLD'
    
    def _analyze_pattern(self, signal: SignalInfo) -> float:
        """패턴 분석"""
        try:
            # RSI 패턴 분석
            rsi_score = self._analyze_rsi_pattern(signal.rsi)
            
            # MACD 패턴 분석
            macd_score = self._analyze_macd_pattern(signal.macd)
            
            # 볼륨 패턴 분석
            volume_score = self._analyze_volume_pattern(signal.volume_ratio)
            
            # 종합 패턴 점수
            pattern_score = (rsi_score + macd_score + volume_score) / 3
            
            return pattern_score
            
        except Exception as e:
            print(f"⚠️ 패턴 분석 오류: {e}")
            return 0.5
    
    def _analyze_rsi_pattern(self, rsi: float) -> float:
        """RSI 패턴 분석"""
        if rsi < 30:
            return 0.8  # 과매도 - 매수 기회
        elif rsi < 45:
            return 0.6  # 낮은 RSI - 약간의 매수 기회
        elif rsi < 55:
            return 0.5  # 중립
        elif rsi < 70:
            return 0.4  # 높은 RSI - 약간의 매도 기회
        else:
            return 0.2  # 과매수 - 매도 기회
    
    def _analyze_macd_pattern(self, macd: float) -> float:
        """MACD 패턴 분석"""
        if macd > 0.1:
            return 0.8  # 강한 상승 모멘텀
        elif macd > 0:
            return 0.6  # 약한 상승 모멘텀
        elif macd > -0.1:
            return 0.4  # 약한 하락 모멘텀
        else:
            return 0.2  # 강한 하락 모멘텀
    
    def _analyze_volume_pattern(self, volume_ratio: float) -> float:
        """볼륨 패턴 분석"""
        if volume_ratio > 2.0:
            return 0.8  # 높은 거래량 - 강한 신호
        elif volume_ratio > 1.5:
            return 0.7  # 증가한 거래량
        elif volume_ratio > 0.8:
            return 0.5  # 정상 거래량
        else:
            return 0.3  # 낮은 거래량 - 약한 신호
    
    def _analyze_market_context(self, market_context: dict) -> float:
        """시장 맥락 분석"""
        try:
            trend = market_context.get('trend', 'neutral')
            volatility = market_context.get('volatility', 0.02)
            
            # 트렌드 기반 점수
            if trend == 'bullish':
                trend_score = 0.7
            elif trend == 'bearish':
                trend_score = 0.3
            else:
                trend_score = 0.5
            
            # 변동성 기반 점수 (적당한 변동성이 좋음)
            if 0.01 < volatility < 0.05:
                vol_score = 0.8
            elif volatility < 0.01:
                vol_score = 0.4  # 너무 낮은 변동성
            else:
                vol_score = 0.3  # 너무 높은 변동성
            
            return (trend_score + vol_score) / 2
            
        except Exception as e:
            print(f"⚠️ 시장 맥락 분석 오류: {e}")
            return 0.5
    
    def _evaluate_risk(self, signal: SignalInfo, current_price: float) -> float:
        """리스크 평가"""
        try:
            # 신호 신뢰도 기반 리스크
            confidence_risk = 1.0 - signal.confidence
            
            # 변동성 기반 리스크
            volatility_risk = min(signal.volatility * 10, 1.0)
            
            # 종합 리스크 점수 (낮을수록 좋음)
            risk_score = 1.0 - (confidence_risk + volatility_risk) / 2
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            print(f"⚠️ 리스크 평가 오류: {e}")
            return 0.5
    
    def _make_final_decision(self, pattern_score: float, context_score: float, risk_score: float, signal: SignalInfo) -> str:
        """최종 거래 결정"""
        try:
            # 가중 평균 점수 계산
            final_score = (pattern_score * 0.4 + context_score * 0.3 + risk_score * 0.3)
            
            # 신호 점수와 결합
            combined_score = (final_score + signal.signal_score) / 2
            
            # 결정 임계값
            if combined_score > 0.7:
                return 'BUY'
            elif combined_score < 0.3:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            print(f"⚠️ 최종 결정 오류: {e}")
            return 'HOLD'

class MarketAnalyzer:
    """시장 분석기 - 시장 상황 실시간 분석"""
    def __init__(self):
        self.market_conditions = {}
        self.trend_analysis = {}
        
    def analyze_market_condition(self, coin: str, interval: str) -> dict:
        """시장 상황 분석"""
        try:
            # 기본 시장 상황
            market_condition = {
                'trend': 'neutral',
                'volatility': 0.02,
                'volume_trend': 'normal',
                'momentum': 'neutral',
                'timestamp': int(time.time())
            }
            
            # 코인별 시장 상황 업데이트
            key = f"{coin}_{interval}"
            self.market_conditions[key] = market_condition
            
            return market_condition
            
        except Exception as e:
            print(f"⚠️ 시장 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02, 'timestamp': int(time.time())}

class RiskManager:
    """리스크 관리자 - 포지션 리스크 관리"""
    def __init__(self):
        self.risk_limits = {}
        self.position_risks = {}
        
    def calculate_position_risk(self, coin: str, position: VirtualPosition, current_price: float) -> float:
        """포지션 리스크 계산"""
        try:
            # 현재 손익
            current_pnl = (current_price - position.entry_price) / position.entry_price
            
            # 최대 손실
            max_loss = abs(position.max_loss_pct) / 100
            
            # 리스크 점수 (0-1, 높을수록 위험)
            risk_score = min(max_loss / 0.1, 1.0)  # 10% 손실을 최대 위험으로 설정
            
            return risk_score
            
        except Exception as e:
            print(f"⚠️ 포지션 리스크 계산 오류: {e}")
            return 0.5
    
    def should_close_position(self, coin: str, position: VirtualPosition, current_price: float, stop_loss_pct: float, take_profit_pct: float) -> bool:
        """포지션 종료 여부 판단 (파라미터 주입 방식)"""
        try:
            # 손절/익절 조건 확인
            if position.profit_loss_pct <= -stop_loss_pct:
                return True
            
            if position.profit_loss_pct >= take_profit_pct:
                return True
            
            # 리스크 기반 종료
            risk_score = self.calculate_position_risk(coin, position, current_price)
            if risk_score > 0.8:  # 80% 이상 위험시 종료
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ 포지션 종료 판단 오류: {e}")
            return False

class LearningFeedback:
    """학습 피드백 시스템 - 거래 결과 학습"""
    def __init__(self):
        self.trade_feedback = {}
        self.pattern_performance = {}
        
    def record_trade_result(self, coin: str, trade_result: dict):
        """거래 결과 기록"""
        try:
            # 거래 결과 저장
            trade_id = f"{coin}_{trade_result.get('entry_timestamp', 0)}"
            self.trade_feedback[trade_id] = trade_result
            
            # 패턴 성과 업데이트
            signal_pattern = trade_result.get('signal_pattern', 'unknown')
            if signal_pattern not in self.pattern_performance:
                self.pattern_performance[signal_pattern] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'total_profit': 0.0
                }
            
            perf = self.pattern_performance[signal_pattern]
            perf['total_trades'] += 1
            perf['total_profit'] += trade_result.get('profit_loss_pct', 0.0)
            
            if trade_result.get('profit_loss_pct', 0.0) > 0:
                perf['successful_trades'] += 1
            
            print(f"📊 거래 결과 기록: {coin} 패턴 {signal_pattern} 성과 업데이트")
            
        except Exception as e:
            print(f"⚠️ 거래 결과 기록 오류: {e}")

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

class VirtualTrader:
    """가상매매 시뮬레이터 (전체 코인 대상 + 무제한 포지션)"""
    
    def __init__(self):
        """🚀 최적화된 가상매매 시뮬레이터 초기화"""
        self.positions = {}
        self.max_positions = int(os.getenv('MAX_POSITIONS', '100'))  # 환경변수로 제한 가능
        self.min_confidence = 0.3  # 30% (완화된 기준)
        self.min_signal_score = 0.3  # 30% (완화된 기준)
        self.stop_loss_pct = 10.0  # 10% 손절
        self.take_profit_pct = 50.0  # 50% 익절
        self.max_holding_hours = None  # 제거됨
        
        # 🆕 통계 변수 초기화 (누락된 부분)
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_pct = 0.0
        self.max_drawdown = 0.0
        self.trade_history = []
        
        # 🚀 성능 최적화 설정
        self.batch_size = 20
        self.position_update_batch = []
        self.last_batch_update = time.time()
        self.cleanup_interval = 300  # 5분마다 정리
        self.max_position_age = 86400 * 7  # 7일 후 강제 정리
        
        # 🚀 캐시 시스템
        self.price_cache = {}
        
        # 🆕 성능 업그레이드 시스템 초기화
        self.action_tracker = ActionPerformanceTracker()
        self.context_recorder = ContextRecorder()
        self.outlier_applier = OutlierGuardrailApplier()
        
        # 🆕 진화형 AI 시스템 초기화
        self.ai_decision_engine = AIDecisionEngine()
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        self.learning_feedback = LearningFeedback()
        
        # 🆕 시그널-매매 연결 시스템
        self.signal_trade_connector = SignalTradeConnector()
        
        print("🚀 진화형 AI 가상 트레이더 초기화 완료")
        self.cache_ttl = 60  # 1분 캐시
        
        # 🆕 데이터베이스 경로 설정
        self.db_path = TRADING_SYSTEM_DB_PATH
        
        # 🆕 거래 테이블 생성
        self.create_trading_tables()
        
        # 🆕 기존 포지션 로드
        self.load_positions_from_db()
        
        # 🆕 0원 진입가 포지션들 수정
        self._fix_zero_entry_prices()
        
        # 🆕 대상 코인 목록 (전체 코인)
        self.target_coins = self._get_all_available_coins()
        
        # 🆕 과도한 포지션 정리
        self._cleanup_excessive_positions()
        
        print(f"🚀 가상매매 시뮬레이터 시작")
    
    def _cleanup_excessive_positions(self):
        """과도한 포지션 정리"""
        if len(self.positions) > 50:  # 50개 이상이면 정리
            closed_count = 0
            for coin in list(self.positions.keys()):
                try:
                    # 🆕 현재가로 포지션 종료
                    current_price = self._get_latest_price(coin)
                    if current_price > 0:
                        self._close_position(coin, current_price, int(datetime.now().timestamp()), 'cleanup')
                        closed_count += 1
                except Exception as e:
                    pass
            
            if closed_count > 0:
                print(f"🔄 {closed_count}개 포지션 정리 완료")
    
    def _get_all_available_coins(self) -> List[str]:
        """전체 사용 가능한 코인 목록 조회 (거래량 제한 없음)"""
        try:
            # 🔧 signals 테이블은 TRADING_SYSTEM_DB_PATH에 있음
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 더 유연한 조회: combined가 없으면 모든 interval에서 조회
                query = """
                    SELECT DISTINCT coin FROM signals
                    WHERE timestamp > ?
                    ORDER BY coin
                """
                # 최근 24시간 내 시그널이 있는 코인들
                cutoff_time = int((datetime.now() - timedelta(hours=24)).timestamp())
                df = pd.read_sql(query, conn, params=(cutoff_time,))
                
                coins = df['coin'].tolist()
                print(f"📈 전체 대상 코인 수: {len(coins)}개")
                
                # 🆕 만약 코인이 없으면 candles 테이블에서 조회
                if not coins:
                    print("🔄 signals 테이블에 코인이 없어 candles 테이블에서 조회...")
                    # 🔧 realtime_candles.db에서 조회
                    with sqlite3.connect(DB_PATH) as candles_conn:
                        candles_query = """
                            SELECT DISTINCT coin FROM candles 
                            WHERE timestamp > ?
                            ORDER BY coin
                            LIMIT 50
                        """
                        candles_df = pd.read_sql(candles_query, candles_conn, params=(cutoff_time,))
                        coins = candles_df['coin'].tolist()
                        print(f"📈 candles 테이블에서 조회된 코인 수: {len(coins)}개")
                
                return coins
                
        except Exception as e:
            print(f"⚠️ 전체 코인 목록 조회 오류: {e}")
            # 기본 코인 목록 반환
            # DB 기반 사용 가능 코인 반환 (하드코딩 제거)
            try:
                from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
                available = get_available_coins_and_intervals()
                coins = sorted(list({c for c, _ in available}))
                return coins or ['BTC']
            except Exception:
                return [os.getenv('DEFAULT_COIN', 'BTC')]
    
    def can_open_position(self, coin: str) -> bool:
        """새로운 포지션 열기 가능 여부 확인 (무제한 포지션)"""
        # 🆕 무제한 포지션: 이미 보유 중이지 않으면 가능
        return coin not in self.positions
    
    def get_new_signals(self, max_hours_back: int = 24, batch_size: int = 100) -> List[SignalInfo]:
        """🚀 새로운 시그널 조회 - 멀티 타임프레임 combined 시그널 우선 처리"""
        try:
            # 🔧 signals 테이블은 TRADING_SYSTEM_DB_PATH에 있음
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 최근 N시간 내의 시그널 조회
                current_time = int(datetime.now().timestamp())
                time_threshold = current_time - (max_hours_back * 3600)

                # 🎯 1순위: combined 시그널 (멀티 타임프레임 통합)
                combined_query = """
                SELECT * FROM signals
                WHERE timestamp >= ? AND interval = 'combined'
                AND (coin, timestamp) IN (
                    SELECT coin, MAX(timestamp)
                    FROM signals
                    WHERE timestamp >= ? AND interval = 'combined'
                    GROUP BY coin
                )
                ORDER BY timestamp DESC
                LIMIT ?
                """

                combined_df = pd.read_sql(combined_query, conn, params=(time_threshold, time_threshold, batch_size))

                if len(combined_df) > 0:
                    print(f"📊 combined 시그널 {len(combined_df)}개 조회 (멀티 타임프레임 통합)")
                    return self._convert_df_to_signals(combined_df)

                # 🎯 2순위: combined 시그널이 없으면 각 코인별로 최신 시그널 생성
                print("🔄 combined 시그널이 없어 각 코인별 최신 시그널 조회...")

                # 사용 가능한 코인 목록 조회
                coins_query = """
                SELECT DISTINCT coin FROM signals 
                WHERE timestamp >= ?
                ORDER BY coin
                """
                coins_df = pd.read_sql(coins_query, conn, params=(time_threshold,))
                
                if coins_df.empty:
                    print("⚠️ 사용 가능한 코인이 없습니다")
                    return []
                
                # 🎯 각 코인별로 최신 시그널 조회
                signals = []
                for coin in coins_df['coin'].head(batch_size):
                    try:
                        # 🆕 멀티 타임프레임 시그널 생성 시도
                        from realtime_signal_selector import SignalSelector
                        selector = SignalSelector()
                        
                        # 멀티 타임프레임 시그널 생성
                        mtf_signal = selector.generate_multi_timeframe_signal(coin)
                        
                        if mtf_signal:
                            signals.append(mtf_signal)
                            print(f"  ✅ {coin}: 멀티 타임프레임 시그널 생성 성공")
                        else:
                            # 실패 시 기존 시그널 조회
                            fallback_signal = self._get_fallback_signal(conn, coin, time_threshold)
                            if fallback_signal:
                                signals.append(fallback_signal)
                                print(f"  ⚠️ {coin}: 기존 시그널 사용")
                    except Exception as e:
                        print(f"  ❌ {coin}: 시그널 생성 실패 - {e}")
                        continue
                
                print(f"📊 총 {len(signals)}개 시그널 처리 완료")
                return signals
                
        except Exception as e:
            print(f"⚠️ 시그널 조회 오류: {e}")
            return []
    
    def _get_fallback_signal(self, conn, coin: str, time_threshold: int) -> Optional[SignalInfo]:
        """기존 시그널 조회 (fallback)"""
        try:
            fallback_query = """
            SELECT * FROM signals 
            WHERE coin = ? AND timestamp >= ?
            ORDER BY timestamp DESC LIMIT 1
            """
            
            fallback_df = pd.read_sql(fallback_query, conn, params=(coin, time_threshold))
            
            if fallback_df.empty:
                return None
            
            row = fallback_df.iloc[0]
            return self._create_signal_from_row(row)
            
        except Exception as e:
            print(f"⚠️ {coin} fallback 시그널 조회 실패: {e}")
            return None
    
    def _create_signal_from_row(self, row) -> SignalInfo:
        """DB 행을 SignalInfo 객체로 변환"""
        try:
            return SignalInfo(
                coin=row['coin'],
                interval=row['interval'],
                action=SignalAction(row['action']),
                signal_score=float(row['signal_score']),
                confidence=float(row['confidence']),
                reason=row['reason'],
                timestamp=int(row['timestamp']),
                price=float(row['current_price']),
                volume=0.0,
                rsi=float(row['rsi']),
                macd=float(row['macd']),
                wave_phase=row['wave_phase'],
                pattern_type=row['pattern_type'],
                risk_level=row['risk_level'],
                volatility=float(row['volatility']),
                volume_ratio=float(row['volume_ratio']),
                wave_progress=float(row['wave_progress']),
                structure_score=float(row['structure_score']),
                pattern_confidence=float(row['pattern_confidence']),
                integrated_direction=row['integrated_direction'],
                integrated_strength=float(row['integrated_strength'])
            )
        except Exception as e:
            print(f"⚠️ 시그널 객체 생성 실패: {e}")
            return None
    
    def _convert_df_to_signals(self, df: pd.DataFrame) -> List[SignalInfo]:
        """DataFrame을 SignalInfo 리스트로 변환"""
        signals = []
        for _, row in df.iterrows():
            try:
                signal = self._create_signal_from_row(row)
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"⚠️ 시그널 변환 실패: {e}")
                continue
        return signals
    
    def create_trading_tables(self):
        """거래 관련 테이블 생성"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 가상매매 포지션 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        entry_timestamp INTEGER NOT NULL,
                        entry_signal_score REAL NOT NULL,
                        current_price REAL NOT NULL,
                        profit_loss_pct REAL NOT NULL,
                        holding_duration INTEGER NOT NULL,
                        max_profit_pct REAL NOT NULL,
                        max_loss_pct REAL NOT NULL,
                        stop_loss_price REAL NOT NULL,
                        take_profit_price REAL NOT NULL,
                        last_updated INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin)
                    )
                """)
                
                # 가상매매 거래 히스토리 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_trade_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        profit_loss_pct REAL NOT NULL,
                        action TEXT NOT NULL,
                        entry_timestamp INTEGER NOT NULL,
                        exit_timestamp INTEGER NOT NULL,
                        holding_duration INTEGER NOT NULL,
                        entry_signal_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 가상매매 성과 통계 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_performance_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        total_profit_pct REAL NOT NULL,
                        max_drawdown_pct REAL NOT NULL,
                        active_positions INTEGER NOT NULL,
                        total_episodes INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 🆕 가상매매 피드백 테이블 (학습용 상세 정보)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        profit_loss_pct REAL NOT NULL,
                        holding_duration INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        entry_timestamp INTEGER NOT NULL,
                        exit_timestamp INTEGER NOT NULL,
                        entry_signal_score REAL NOT NULL,
                        entry_confidence REAL,
                        entry_rsi REAL,
                        entry_macd REAL,
                        entry_volume_ratio REAL,
                        entry_wave_phase TEXT,
                        entry_pattern_type TEXT,
                        entry_risk_level TEXT,
                        entry_volatility REAL,
                        entry_structure_score REAL,
                        entry_pattern_confidence REAL,
                        entry_integrated_direction TEXT,
                        entry_integrated_strength REAL,
                        market_conditions TEXT,
                        signal_pattern TEXT,
                        is_learned BOOLEAN DEFAULT FALSE,
                        learning_episode INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 🆕 가상매매 Q-table 테이블 (시그널 계산기에서 사용)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_trading_q_table (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state_key TEXT NOT NULL,
                        action TEXT NOT NULL,
                        q_value REAL NOT NULL,
                        episode_count INTEGER DEFAULT 1,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(state_key, action)
                    )
                """)
                
                # 인덱스 생성
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_positions_coin ON virtual_positions(coin)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_trade_history_coin ON virtual_trade_history(coin)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_trade_history_timestamp ON virtual_trade_history(exit_timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_performance_timestamp ON virtual_performance_stats(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_feedback_coin ON trade_feedback(coin)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_feedback_timestamp ON trade_feedback(entry_timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_trading_q_table_state ON virtual_trading_q_table(state_key)')
                
                # 🆕 기존 테이블에 누락된 컬럼 추가 (마이그레이션)
                try:
                    conn.execute("ALTER TABLE virtual_performance_stats ADD COLUMN total_episodes INTEGER DEFAULT 0")
                    print("✅ virtual_performance_stats 테이블에 total_episodes 컬럼 추가 완료")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e):
                        print("ℹ️ total_episodes 컬럼이 이미 존재합니다")
                    else:
                        print(f"⚠️ 컬럼 추가 중 오류: {e}")
                
                conn.commit()
                print("✅ 거래 테이블 생성 완료")
                
        except Exception as e:
            print(f"⚠️ 거래 테이블 생성 오류: {e}")
    
    def load_signal_from_db(self, coin: str, timestamp: int) -> Optional[SignalInfo]:
        """DB에서 시그널 로드 (Absolute Zero System의 새로운 고급 지표들 포함)"""
        try:
            # 🔧 signals 테이블은 TRADING_SYSTEM_DB_PATH에 있음
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🚨 더 유연한 시그널 검색 (combined가 없으면 다른 interval도 검색)
                df = pd.read_sql("""
                    SELECT * FROM signals 
                    WHERE coin = ? 
                    AND timestamp BETWEEN ? - 7200 AND ? + 7200  -- 2시간 범위로 확장
                    ORDER BY 
                        CASE WHEN interval = 'combined' THEN 1 ELSE 2 END,  -- combined 우선
                        ABS(timestamp - ?) ASC 
                    LIMIT 1
                """, conn, params=(coin, timestamp, timestamp, timestamp))
                
                if df.empty:
                    print(f"⚠️ {coin}: 진입 시점({timestamp}) 근처의 시그널을 찾을 수 없음")
                    return None
                
                row = df.iloc[0]
                print(f"✅ {coin}: 시그널 로드 성공 (타임스탬프: {row['timestamp']}, 진입: {timestamp}, interval: {row['interval']})")
                
                return SignalInfo(
                    coin=row['coin'],
                    interval=row['interval'],
                    action=SignalAction(row['action']),
                    signal_score=row['signal_score'],
                    confidence=row['confidence'],
                    reason=row['reason'],
                    timestamp=row['timestamp'],
                    price=row['current_price'],
                    volume=0,
                    rsi=row['rsi'],
                    macd=row['macd'],
                    wave_phase=row['wave_phase'],
                    pattern_type=row['pattern_type'],
                    risk_level=row['risk_level'],
                    volatility=row['volatility'],
                    volume_ratio=row['volume_ratio'],
                    wave_progress=0.0,  # 기본값
                    structure_score=row['structure_score'],
                    pattern_confidence=row['pattern_confidence'],
                    integrated_direction=row['integrated_direction'],
                    integrated_strength=row['integrated_strength'],
                    # 🆕 Absolute Zero System의 새로운 고급 지표들
                    mfi=row.get('mfi', 50.0),
                    atr=row.get('atr', 0.0),
                    adx=row.get('adx', 25.0),
                    ma20=row.get('ma20', 0.0),
                    rsi_ema=row.get('rsi_ema', 50.0),
                    macd_smoothed=row.get('macd_smoothed', 0.0),
                    wave_momentum=row.get('wave_momentum', 0.0),
                    bb_position=row.get('bb_position', 'unknown'),
                    bb_width=row.get('bb_width', 0.0),
                    bb_squeeze=row.get('bb_squeeze', 0.0),
                    rsi_divergence=row.get('rsi_divergence', 'none'),
                    macd_divergence=row.get('macd_divergence', 'none'),
                    volume_divergence=row.get('volume_divergence', 'none'),
                    price_momentum=row.get('price_momentum', 0.0),
                    volume_momentum=row.get('volume_momentum', 0.0),
                    trend_strength=row.get('trend_strength', 0.5),
                    support_resistance=row.get('support_resistance', 'unknown'),
                    fibonacci_levels=row.get('fibonacci_levels', 'unknown'),
                    elliott_wave=row.get('elliott_wave', 'unknown'),
                    harmonic_patterns=row.get('harmonic_patterns', 'none'),
                    candlestick_patterns=row.get('candlestick_patterns', 'none'),
                    market_structure=row.get('market_structure', 'unknown'),
                    flow_level_meta=row.get('flow_level_meta', 'unknown'),
                    pattern_direction=row.get('pattern_direction', 'neutral')
                )
                
        except Exception as e:
            print(f"⚠️ 시그널 로드 오류 ({coin}): {e}")
            return None
    
    def update_position(self, coin: str, current_price: float, timestamp: int) -> Optional[str]:
        """포지션 업데이트 및 액션 결정"""
        if coin not in self.positions:
            return None
        
        # 🆕 캔들 테이블에서 최신 현재가 조회
        try:
            latest_price = self._get_latest_price(coin)
            if latest_price > 0:
                current_price = latest_price
            else:
                # 캔들 데이터가 없으면 시그널의 price 사용
                pass
        except Exception as e:
            print(f"⚠️ 현재가 조회 오류 ({coin}): {e}")
            # 오류 시 시그널의 price 사용
        
        position = self.positions[coin]
        position.current_price = current_price
        
        # 🆕 타임스탬프 타입 안전성 보장
        try:
            entry_timestamp = int(position.entry_timestamp) if position.entry_timestamp is not None else 0
            current_timestamp = int(timestamp) if timestamp is not None else 0
            position.holding_duration = current_timestamp - entry_timestamp
            position.last_updated = current_timestamp
        except (ValueError, TypeError) as e:
            print(f"⚠️ 타임스탬프 변환 오류 ({coin}): {e}")
            position.holding_duration = 0
            position.last_updated = int(datetime.now().timestamp())
        
        # 수익률 계산
        if position.entry_price > 0:
            profit_loss_pct = (current_price - position.entry_price) / position.entry_price * 100
        else:
            print(f"⚠️ {coin}: 진입가가 0이므로 수익률 계산 불가")
            profit_loss_pct = 0.0
        
        position.profit_loss_pct = profit_loss_pct
        
        # 최대 수익/손실 업데이트
        if profit_loss_pct > position.max_profit_pct:
            position.max_profit_pct = profit_loss_pct
        if profit_loss_pct < position.max_loss_pct:
            position.max_loss_pct = profit_loss_pct
        
        # 액션 결정
        action = self._determine_position_action(position, current_price, timestamp)
        
        if action in ['take_profit', 'stop_loss', 'sell']:
            self._close_position(coin, current_price, timestamp, action)
        
        # DB에 포지션 업데이트
        self.update_position_in_db(coin)
        
        return action
    
    def _determine_position_action(self, position: VirtualPosition, current_price: float, timestamp: int) -> str:
        """포지션 액션 결정 (시그널 점수 중심 + 학습 기반 동적 리스크 관리 + 적응적 고급 지표 활용)"""
        # 🆕 현재 시그널 정보 조회
        current_signal = self._get_current_signal_info(position.coin)
        
        if not current_signal:
            # 시그널이 없으면 기본 홀딩
            return 'hold'
        
        # 🎯 핵심: 시그널 점수가 주요 기준
        signal_score = current_signal.signal_score
        confidence = current_signal.confidence
        
        # 🆕 적응적 고급 지표 분석
        adaptive_analysis = self._analyze_adaptive_indicators(current_signal)
        
        # 🆕 학습 기반 동적 손절 강도 계산
        stop_loss_strength = self._calculate_adaptive_stop_loss_strength(position, current_signal)
        
        # 🆕 시그널 점수 기반 매매 결정 (핵심 로직)
        if signal_score < -0.5 and confidence > 0.6:
            # 강한 매도 시그널
            return 'sell'
        
        elif signal_score < -0.3 and confidence > 0.5:
            # 매도 시그널
            return 'sell'
        
        elif signal_score < -0.2 and confidence > 0.4:
            # 약한 매도 시그널 (손절 고려)
            if position.profit_loss_pct < -3.0:  # 손실이 있는 경우
                return 'stop_loss'
            else:
                return 'hold'
        
        elif signal_score < 0.0 and position.profit_loss_pct > 5.0:
            # 수익이 있지만 시그널이 약해진 경우 (익절 고려)
            return 'take_profit'
        
        elif signal_score < 0.0 and position.profit_loss_pct < -5.0:
            # 손실이 있고 시그널이 약해진 경우 (손절 고려)
            return 'stop_loss'
        
        # 🆕 홀딩 (시그널이 중립적이거나 약간 양호)
        return 'hold'
    
    def _analyze_adaptive_indicators(self, signal: SignalInfo) -> Dict:
        """적응적 고급 지표 분석"""
        try:
            # 🎯 시장 상황 분석
            market_context = self._get_market_context()
            
            # 🎯 고급 지표 분석
            advanced_indicators = {
                'mfi': signal.mfi,
                'atr': signal.atr,
                'adx': signal.adx,
                'rsi_divergence': signal.rsi_divergence,
                'macd_divergence': signal.macd_divergence,
                'bb_squeeze': signal.bb_squeeze,
                'trend_strength': signal.trend_strength,
                'price_momentum': signal.price_momentum,
                'volume_momentum': signal.volume_momentum
            }
            
            # 🎯 시장 상황에 따른 적응적 분석
            analysis_result = {
                'market_trend': market_context['trend'],
                'market_volatility': market_context['volatility'],
                'technical_score': 0.0,
                'risk_level': 'medium'
            }
            
            # 🎯 시장 상황별 적응적 가중치 적용
            if market_context['trend'] == 'bullish':
                # 상승장에서는 다이버전스와 트렌드 강도에 높은 가중치
                if signal.rsi_divergence == 'bullish' or signal.macd_divergence == 'bullish':
                    analysis_result['technical_score'] += 0.15
                
                if signal.trend_strength > 0.7:
                    analysis_result['technical_score'] += 0.12
                    
            elif market_context['trend'] == 'bearish':
                # 하락장에서는 볼린저밴드 스퀴즈와 모멘텀에 높은 가중치
                if signal.bb_squeeze > 0.8:
                    analysis_result['technical_score'] += 0.10
                
                if abs(signal.price_momentum) > 0.05:
                    analysis_result['technical_score'] += 0.08
                    
            else:  # 중립장
                # 중립장에서는 균형잡힌 분석
                if signal.rsi_divergence == 'bullish' or signal.macd_divergence == 'bullish':
                    analysis_result['technical_score'] += 0.10
                
                if signal.trend_strength > 0.7:
                    analysis_result['technical_score'] += 0.08
                
                if signal.bb_squeeze > 0.8:
                    analysis_result['technical_score'] += 0.05
            
            # 🎯 변동성에 따른 조정
            if market_context['volatility'] > 0.05:  # 고변동성
                analysis_result['technical_score'] *= 0.8
                analysis_result['risk_level'] = 'high'
            elif market_context['volatility'] < 0.02:  # 저변동성
                analysis_result['technical_score'] *= 1.2
                analysis_result['risk_level'] = 'low'
            
            return analysis_result
            
        except Exception as e:
            print(f"⚠️ 적응적 지표 분석 오류: {e}")
            return {
                'market_trend': 'neutral',
                'market_volatility': 0.02,
                'technical_score': 0.0,
                'risk_level': 'medium'
            }
    
    def _calculate_adaptive_stop_loss_strength(self, position: VirtualPosition, signal: SignalInfo) -> float:
        """학습 기반 동적 손절 강도 계산"""
        try:
            coin = position.coin
            
            # 🎯 코인별 과거 손절 성과 분석
            stop_loss_performance = self._analyze_stop_loss_performance(coin)
            
            # 🎯 현재 시그널 강도
            signal_strength = abs(signal.signal_score)
            
            # 🎯 시장 변동성
            market_volatility = self._get_market_volatility()
            
            # 🎯 기본 손절 강도 (50%)
            base_strength = 50.0
            
            # 🎯 성과 기반 조정
            if stop_loss_performance > 0.7:  # 손절이 효과적이었던 경우
                base_strength += 20.0  # 손절 강도 증가
            elif stop_loss_performance < 0.3:  # 손절이 비효과적이었던 경우
                base_strength -= 15.0  # 손절 강도 감소
            
            # 🎯 시그널 강도 기반 조정
            if signal_strength > 0.5:  # 강한 매도 시그널
                base_strength += 15.0
            elif signal_strength < 0.2:  # 약한 매도 시그널
                base_strength -= 10.0
            
            # 🎯 변동성 기반 조정
            if market_volatility > 0.05:  # 고변동성
                base_strength += 10.0  # 고변동성에서는 손절 강화
            elif market_volatility < 0.02:  # 저변동성
                base_strength -= 5.0  # 저변동성에서는 손절 완화
            
            return max(30.0, min(80.0, base_strength))  # 30~80% 범위로 제한
            
        except Exception as e:
            print(f"⚠️ 동적 손절 강도 계산 오류: {e}")
            return 50.0  # 기본값 반환
    
    def _get_market_context(self) -> Dict:
        """시장 상황 분석"""
        try:
            # 🎯 기준 코인(환경/DB) 시장 상황 분석
            base_coin = None
            try:
                from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
                available = get_available_coins_and_intervals()
                base_coin = next(iter({c for c, _ in available}), None)
            except Exception:
                base_coin = None
            btc_signal = self._get_current_signal_info(base_coin or os.getenv('DEFAULT_COIN', 'BTC'))
            
            if btc_signal:
                signal_score = btc_signal.signal_score
                
                if signal_score > 0.3:
                    trend = 'bullish'
                elif signal_score < -0.3:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
                
                # 변동성은 간단히 계산 (실제로는 더 복잡한 계산 필요)
                volatility = 0.02  # 기본값
            else:
                trend = 'neutral'
                volatility = 0.02
            
            return {
                'trend': trend,
                'volatility': volatility
            }
            
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02}
    
    def _analyze_stop_loss_performance(self, coin: str) -> float:
        """코인별 손절 성과 분석"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 최근 30일간 손절 거래 분석
                thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
                
                df = pd.read_sql("""
                    SELECT profit_loss_pct FROM virtual_trade_history 
                    WHERE coin = ? AND exit_timestamp > ? 
                    AND action IN ('stop_loss', 'sell')
                    ORDER BY exit_timestamp DESC
                """, conn, params=(coin, thirty_days_ago))
                
                if df.empty:
                    return 0.5  # 손절 내역 없으면 중립
                
                # 손절 후 추가 하락 여부 분석
                avg_stop_loss = df['profit_loss_pct'].mean()
                
                # 손절이 효과적이었는지 판단 (-10% 이상 손절이면 효과적)
                if avg_stop_loss < -10.0:
                    return 0.8  # 효과적
                elif avg_stop_loss > -5.0:
                    return 0.2  # 비효과적
                else:
                    return 0.5  # 중간
                
        except Exception as e:
            print(f"⚠️ 손절 성과 분석 오류 ({coin}): {e}")
            return 0.5
    
    def _get_market_volatility(self) -> float:
        """시장 변동성 계산"""
        try:
            # 기준 코인 변동성 계산 (간단한 구현)
            base_coin = None
            try:
                from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
                available = get_available_coins_and_intervals()
                base_coin = next(iter({c for c, _ in available}), None)
            except Exception:
                base_coin = None
            btc_signal = self._get_current_signal_info(base_coin or os.getenv('DEFAULT_COIN', 'BTC'))
            
            if btc_signal:
                # 실제로는 더 복잡한 변동성 계산이 필요
                return 0.02  # 기본값
            else:
                return 0.02
                
        except Exception as e:
            print(f"⚠️ 시장 변동성 계산 오류: {e}")
            return 0.02
    
    def _get_current_signal_info(self, coin: str) -> Optional[SignalInfo]:
        """현재 코인의 시그널 정보 조회"""
        try:
            # 🔧 signals 테이블은 TRADING_SYSTEM_DB_PATH에 있음
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                df = pd.read_sql("""
                    SELECT * FROM signals 
                    WHERE coin = ? AND interval = 'combined'
                    ORDER BY timestamp DESC LIMIT 1
                """, conn, params=(coin,))
                
                if df.empty:
                    return None
                
                row = df.iloc[0]
                # 🆕 고급지표 필드들을 DB에서 가져오기 (기본값 대신)
                mfi = row.get('mfi', 50.0)
                atr = row.get('atr', 0.0)
                adx = row.get('adx', 25.0)
                ma20 = row.get('ma20', 0.0)
                rsi_ema = row.get('rsi_ema', 50.0)
                macd_smoothed = row.get('macd_smoothed', 0.0)
                wave_momentum = row.get('wave_momentum', 0.0)
                bb_position = row.get('bb_position', 'unknown')
                bb_width = row.get('bb_width', 0.0)
                bb_squeeze = row.get('bb_squeeze', 0.0)
                rsi_divergence = row.get('rsi_divergence', 'none')
                macd_divergence = row.get('macd_divergence', 'none')
                volume_divergence = row.get('volume_divergence', 'none')
                price_momentum = row.get('price_momentum', 0.0)
                volume_momentum = row.get('volume_momentum', 0.0)
                trend_strength = row.get('trend_strength', 0.5)
                support_resistance = row.get('support_resistance', 'unknown')
                fibonacci_levels = row.get('fibonacci_levels', 'unknown')
                elliott_wave = row.get('elliott_wave', 'unknown')
                harmonic_patterns = row.get('harmonic_patterns', 'none')
                candlestick_patterns = row.get('candlestick_patterns', 'none')
                market_structure = row.get('market_structure', 'unknown')
                flow_level_meta = row.get('flow_level_meta', 'unknown')
                pattern_direction = row.get('pattern_direction', 'neutral')
                
                return SignalInfo(
                    coin=row['coin'],
                    interval=row['interval'],
                    action=SignalAction(row['action']),
                    signal_score=row['signal_score'],
                    confidence=row['confidence'],
                    reason=row['reason'],
                    timestamp=row['timestamp'],
                    price=row['current_price'],
                    volume=0,
                    rsi=row['rsi'],
                    macd=row['macd'],
                    wave_phase=row['wave_phase'],
                    pattern_type=row['pattern_type'],
                    risk_level=row['risk_level'],
                    volatility=row['volatility'],
                    volume_ratio=row['volume_ratio'],
                    wave_progress=row['wave_progress'],
                    structure_score=row['structure_score'],
                    pattern_confidence=row['pattern_confidence'],
                    integrated_direction=row['integrated_direction'],
                    integrated_strength=row['integrated_strength'],
                    # 🆕 실제 DB에서 가져온 고급지표 값들
                    mfi=mfi,
                    atr=atr,
                    adx=adx,
                    ma20=ma20,
                    rsi_ema=rsi_ema,
                    macd_smoothed=macd_smoothed,
                    wave_momentum=wave_momentum,
                    bb_position=bb_position,
                    bb_width=bb_width,
                    bb_squeeze=bb_squeeze,
                    rsi_divergence=rsi_divergence,
                    macd_divergence=macd_divergence,
                    volume_divergence=volume_divergence,
                    price_momentum=price_momentum,
                    volume_momentum=volume_momentum,
                    trend_strength=trend_strength,
                    support_resistance=support_resistance,
                    fibonacci_levels=fibonacci_levels,
                    elliott_wave=elliott_wave,
                    harmonic_patterns=harmonic_patterns,
                    candlestick_patterns=candlestick_patterns,
                    market_structure=market_structure,
                    flow_level_meta=flow_level_meta,
                    pattern_direction=pattern_direction
                )
                
        except Exception as e:
            print(f"⚠️ 현재 시그널 조회 오류 ({coin}): {e}")
            return None
    
    def _analyze_coin_performance(self, coin: str) -> float:
        """코인별 과거 거래 성과 분석"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 최근 30일간 해당 코인의 거래 성과 분석
                thirty_days_ago = int(datetime.now().timestamp()) - (30 * 24 * 3600)
                
                df = pd.read_sql("""
                    SELECT profit_loss_pct, action, holding_duration 
                    FROM virtual_trade_history 
                    WHERE coin = ? AND exit_timestamp >= ?
                    ORDER BY exit_timestamp DESC
                """, conn, params=(coin, thirty_days_ago))
                
                if df.empty:
                    return 0.0  # 과거 거래 없으면 중립
                
                # 평균 수익률 계산
                avg_profit = df['profit_loss_pct'].mean()
                
                # 승률 계산
                win_rate = len(df[df['profit_loss_pct'] > 0]) / len(df)
                
                # 🆕 성과 기반 손절 조정
                if avg_profit > 5.0 and win_rate > 0.6:
                    return 2.0  # 좋은 성과: 손절을 2% 더 관대하게
                elif avg_profit < -5.0 or win_rate < 0.3:
                    return -2.0  # 나쁜 성과: 손절을 2% 더 엄격하게
                else:
                    return 0.0  # 중간 성과: 중립
                    
        except Exception as e:
            print(f"⚠️ 코인 성과 분석 오류 ({coin}): {e}")
            return 0.0
    
    def _get_signal_based_stop_loss(self, signal_score: float) -> float:
        """시그널 점수 기반 손절 조정"""
        try:
            # 🆕 시그널 점수가 높을수록 손절을 더 관대하게
            if signal_score >= 0.8:
                return 3.0  # 매우 높은 시그널: 손절을 3% 더 관대하게
            elif signal_score >= 0.6:
                return 1.5  # 높은 시그널: 손절을 1.5% 더 관대하게
            elif signal_score <= 0.3:
                return -1.5  # 낮은 시그널: 손절을 1.5% 더 엄격하게
            else:
                return 0.0  # 중간 시그널: 중립
                
        except Exception as e:
            print(f"⚠️ 시그널 기반 손절 계산 오류: {e}")
            return 0.0
    
    def _get_time_based_stop_loss(self, holding_duration: int) -> float:
        """보유 시간 기반 손절 조정"""
        try:
            holding_hours = holding_duration / 3600
            
            # 🆕 보유 시간이 길수록 손절을 더 관대하게 (장기 투자 신뢰)
            if holding_hours >= 12:
                return 2.0  # 12시간 이상 보유: 손절을 2% 더 관대하게
            elif holding_hours >= 6:
                return 1.0  # 6시간 이상 보유: 손절을 1% 더 관대하게
            elif holding_hours <= 1:
                return -1.0  # 1시간 이하 보유: 손절을 1% 더 엄격하게
            else:
                return 0.0  # 중간 보유 시간: 중립
                
        except Exception as e:
            print(f"⚠️ 시간 기반 손절 계산 오류: {e}")
            return 0.0
    
    def _close_position(self, coin: str, price: float, timestamp: int, action: str):
        """포지션 종료"""
        position = self.positions[coin]
        
        # 🚨 수익률 재계산 (정확한 계산 보장)
        if position.entry_price > 0:
            profit_loss_pct = ((price - position.entry_price) / position.entry_price) * 100
        else:
            profit_loss_pct = 0.0
            print(f"⚠️ {coin}: 진입가가 0이므로 수익률을 0으로 설정")
        
        # 통계 업데이트
        self.total_trades += 1
        if profit_loss_pct > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.total_profit_pct += profit_loss_pct
        
        # 최대 손실 업데이트
        if profit_loss_pct < self.max_drawdown:
            self.max_drawdown = profit_loss_pct
        
        # 🚨 보유시간 정확히 계산
        try:
            entry_timestamp = int(position.entry_timestamp) if position.entry_timestamp is not None else timestamp
            exit_timestamp = int(timestamp) if timestamp is not None else entry_timestamp
            actual_holding_duration = exit_timestamp - entry_timestamp
        except (ValueError, TypeError) as e:
            print(f"⚠️ {coin} 보유시간 계산 오류: {e}")
            actual_holding_duration = 0
        
        # 거래 히스토리 기록
        trade_record = {
            'coin': coin,
            'entry_price': position.entry_price,
            'exit_price': price,
            'quantity': position.quantity,
            'profit_loss_pct': profit_loss_pct,
            'action': action,
            'entry_timestamp': entry_timestamp,
            'exit_timestamp': exit_timestamp,
            'holding_duration': actual_holding_duration,
            'entry_signal_score': position.entry_signal_score
        }
        self.trade_history.append(trade_record)
        
        # 🆕 액션별 성과 추적
        success = profit_loss_pct > 0
        self.action_tracker.record_action_result(action, profit_loss_pct, success)
        
        # 🆕 컨텍스트 기록
        trade_id = f"{coin}_{entry_timestamp}_{exit_timestamp}"
        context = {
            'volatility': getattr(position, 'volatility', 0.0),
            'volume_ratio': getattr(position, 'volume_ratio', 1.0),
            'market_trend': self._get_market_context().get('trend', 'unknown'),
            'action': action,
            'profit_loss_pct': profit_loss_pct
        }
        self.context_recorder.record_trade_context(trade_id, context)
        
        # 🆕 학습 피드백에 거래 결과 기록
        self.learning_feedback.record_trade_result(coin, {
            'trade_record': trade_record,
            'context': context,
            'action_performance': self.action_tracker.get_action_performance(action)
        })
        
        # 🆕 시그널-매매 연결
        signal_pattern = self._extract_signal_pattern_for_feedback(position.entry_signal_score)
        self.signal_trade_connector.connect_signal_to_trade(signal_pattern, trade_record)
        
        # DB에 거래 히스토리 저장
        self.save_trade_to_db(trade_record)
        
        # 🆕 학습용 completed_trades 테이블에도 저장
        self.save_completed_trade_for_learning(trade_record)
        
        # 🆕 가상매매 피드백 테이블에 상세 정보 저장
        self.save_trade_feedback_for_learning(trade_record)
        
        # DB에서 포지션 삭제
        self.delete_position_from_db(coin)
        
        # 포지션 제거
        del self.positions[coin]
        
        action_name = {
            'take_profit': "익절",
            'stop_loss': "손절", 
            'sell': "매도"
        }.get(action, "매도")
        
        print(f"🆕 포지션 종료: {get_korean_name(coin)} {action_name} @ {self._format_price(price)}원 (수익률: {profit_loss_pct:+.2f}%)")
    
    def save_position_to_db(self, coin: str):
        """포지션을 DB에 저장"""
        try:
            position = self.positions[coin]
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO virtual_positions 
                    (coin, entry_price, quantity, entry_timestamp, entry_signal_score, 
                     current_price, profit_loss_pct, holding_duration, max_profit_pct, 
                     max_loss_pct, stop_loss_price, take_profit_price, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    coin, position.entry_price, position.quantity, position.entry_timestamp,
                    position.entry_signal_score, position.current_price, position.profit_loss_pct,
                    position.holding_duration, position.max_profit_pct, position.max_loss_pct,
                    position.stop_loss_price, position.take_profit_price, position.last_updated
                ))
                conn.commit()
        except Exception as e:
            pass
    
    def update_position_in_db(self, coin: str):
        """포지션 정보를 DB에서 업데이트"""
        try:
            position = self.positions[coin]
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    UPDATE virtual_positions SET
                    current_price = ?, profit_loss_pct = ?, holding_duration = ?,
                    max_profit_pct = ?, max_loss_pct = ?, last_updated = ?
                    WHERE coin = ?
                """, (
                    position.current_price, position.profit_loss_pct, position.holding_duration,
                    position.max_profit_pct, position.max_loss_pct, position.last_updated, coin
                ))
                conn.commit()
        except Exception as e:
            pass
    
    def delete_position_from_db(self, coin: str):
        """포지션을 DB에서 삭제"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("DELETE FROM virtual_positions WHERE coin = ?", (coin,))
                conn.commit()
        except Exception as e:
            pass
    
    def save_trade_to_db(self, trade_record: Dict):
        """거래 내역을 DB에 저장"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO virtual_trade_history
                    (coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                     profit_loss_pct, action, holding_duration, entry_signal_score, quantity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record['coin'], trade_record['entry_timestamp'], trade_record['exit_timestamp'],
                    trade_record['entry_price'], trade_record['exit_price'], trade_record['profit_loss_pct'],
                    trade_record['action'], trade_record['holding_duration'], trade_record['entry_signal_score'],
                    trade_record.get('quantity', 1.0)
                ))
                conn.commit()
                print(f"✅ 거래 기록 저장: {trade_record['coin']} {trade_record['action']}")
        except Exception as e:
            print(f"⚠️ 거래 기록 저장 실패 ({trade_record['coin']}): {e}")
    
    def save_completed_trade_for_learning(self, trade_record: Dict):
        """완료된 거래를 학습용으로 저장"""
        try:
            # 🆕 이미 저장된 거래인지 확인
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                existing = conn.execute("""
                    SELECT 1 FROM virtual_learning_trades 
                    WHERE coin = ? AND entry_timestamp = ? AND exit_timestamp = ?
                """, (trade_record['coin'], trade_record['entry_timestamp'], trade_record['exit_timestamp'])).fetchone()
                
                if existing:
                    return  # 이미 저장된 거래는 건너뛰기
            
            # 🆕 새로운 학습용 거래 저장
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO virtual_learning_trades 
                    (coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                     profit_loss_pct, action, holding_duration, entry_signal_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record['coin'], trade_record['entry_timestamp'], trade_record['exit_timestamp'],
                    trade_record['entry_price'], trade_record['exit_price'], trade_record['profit_loss_pct'],
                    trade_record['action'], trade_record['holding_duration'], trade_record['entry_signal_score']
                ))
                conn.commit()
        except Exception as e:
            pass
    
    def save_trade_feedback_for_learning(self, trade_record: Dict):
        """거래 피드백을 학습용으로 저장"""
        try:
            # 🆕 진입 시점의 시그널 정보 로드
            entry_signal = self.load_signal_from_db(trade_record['coin'], trade_record['entry_timestamp'])
            
            # 🆕 시장 상황 분석
            market_conditions = self._get_market_context()
            
            # 🆕 시그널 패턴 추출
            signal_pattern = self._extract_signal_pattern_for_feedback(entry_signal) if entry_signal else 'unknown_pattern'
            
            # 🆕 피드백 저장
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO virtual_trade_feedback 
                    (coin, entry_price, exit_price, profit_loss_pct, holding_duration, action,
                     entry_timestamp, exit_timestamp, entry_signal_score, entry_confidence,
                     entry_rsi, entry_macd, entry_volume_ratio, entry_wave_phase, entry_pattern_type,
                     entry_risk_level, entry_volatility, entry_structure_score, entry_pattern_confidence,
                     entry_integrated_direction, entry_integrated_strength, market_conditions, signal_pattern)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record['coin'], trade_record['entry_price'], trade_record['exit_price'],
                    trade_record['profit_loss_pct'], trade_record['holding_duration'], trade_record['action'],
                    trade_record['entry_timestamp'], trade_record['exit_timestamp'], trade_record['entry_signal_score'],
                    entry_signal.confidence if entry_signal else 0.5,
                    entry_signal.rsi if entry_signal else 50.0,
                    entry_signal.macd if entry_signal else 0.0,
                    entry_signal.volume_ratio if entry_signal else 1.0,
                    entry_signal.wave_phase if entry_signal else 'unknown',
                    entry_signal.pattern_type if entry_signal else 'none',
                    entry_signal.risk_level if entry_signal else 'unknown',
                    entry_signal.volatility if entry_signal else 0.0,
                    entry_signal.structure_score if entry_signal else 0.5,
                    entry_signal.pattern_confidence if entry_signal else 0.0,
                    entry_signal.integrated_direction if entry_signal else 'neutral',
                    entry_signal.integrated_strength if entry_signal else 0.5,
                    json.dumps(market_conditions) if market_conditions else '{}',
                    signal_pattern
                ))
                conn.commit()
        except Exception as e:
            pass
    
    def load_positions_from_db(self):
        """DB에서 포지션 로드"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                df = pd.read_sql("SELECT * FROM virtual_positions", conn)
                
                self.positions = {}
                fixed_count = 0
                
                for _, row in df.iterrows():
                    try:
                        # 🆕 타임스탬프를 정수로 변환하여 타입 불일치 문제 해결 (바이너리 데이터 처리 추가)
                        entry_timestamp = self._safe_convert_to_int(row['entry_timestamp'])
                        last_updated = self._safe_convert_to_int(row['last_updated'])
                        
                        # 🆕 진입가가 0인 경우 복구
                        entry_price = self._safe_convert_to_float(row['entry_price'])
                        current_price = self._safe_convert_to_float(row['current_price'])
                        
                        if entry_price == 0.0:
                            # 🆕 최신 가격으로 복구
                            latest_price = self._get_latest_price(row['coin'])
                            if latest_price > 0:
                                entry_price = latest_price
                                current_price = latest_price
                                fixed_count += 1
                                print(f"🔧 {row['coin']} 진입가 복구: 0.00원 → {self._format_price(latest_price)}원")
                        
                        # 🆕 현재가도 0인 경우 복구
                        if current_price == 0.0:
                            latest_price = self._get_latest_price(row['coin'])
                            if latest_price > 0:
                                current_price = latest_price
                        
                        # 🆕 손절가와 익절가도 진입가 기반으로 재계산
                        stop_loss_price = self._safe_convert_to_float(row['stop_loss_price'])
                        take_profit_price = self._safe_convert_to_float(row['take_profit_price'])
                        
                        if entry_price > 0:
                            if stop_loss_price == 0.0:
                                stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
                            if take_profit_price == 0.0:
                                take_profit_price = entry_price * (1 + self.take_profit_pct / 100)
                        
                        self.positions[row['coin']] = VirtualPosition(
                            coin=row['coin'],
                            entry_price=entry_price,
                            quantity=self._safe_convert_to_float(row['quantity']),
                            entry_timestamp=entry_timestamp,
                            entry_signal_score=self._safe_convert_to_float(row['entry_signal_score']),
                            current_price=current_price,
                            profit_loss_pct=self._safe_convert_to_float(row['profit_loss_pct']),
                            holding_duration=self._safe_convert_to_int(row['holding_duration']),
                            max_profit_pct=self._safe_convert_to_float(row['max_profit_pct']),
                            max_loss_pct=self._safe_convert_to_float(row['max_loss_pct']),
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            last_updated=last_updated
                        )
                        
                        # 🆕 수정된 포지션을 DB에 저장
                        if entry_price > 0 and (row['entry_price'] == 0.0 or row['current_price'] == 0.0):
                            self.save_position_to_db(row['coin'])
                            
                    except Exception as row_error:
                        print(f"⚠️ 포지션 로드 오류 ({row.get('coin', 'unknown')}): {row_error}")
                        continue
                
                print(f"✅ {len(self.positions)}개 포지션 로드 완료")
                if fixed_count > 0:
                    print(f"🔧 {fixed_count}개 포지션의 가격 정보 복구 완료")
                
        except Exception as e:
            print(f"⚠️ 포지션 로드 오류: {e}")
            self.positions = {}
    
    def _fix_zero_entry_prices(self):
        """0원 진입가 포지션들을 수정"""
        fixed_count = 0
        for coin, position in list(self.positions.items()):
            needs_fix = False
            
            # 🆕 진입가가 0인 경우 수정
            if position.entry_price == 0.0:
                latest_price = self._get_latest_price(coin)
                if latest_price > 0:
                    position.entry_price = latest_price
                    position.current_price = latest_price
                    needs_fix = True
                    print(f"🔧 {coin} 진입가 수정: 0.00원 → {self._format_price(latest_price)}원")
            
            # 🆕 현재가가 0인 경우 수정
            if position.current_price == 0.0:
                latest_price = self._get_latest_price(coin)
                if latest_price > 0:
                    position.current_price = latest_price
                    needs_fix = True
                    print(f"🔧 {coin} 현재가 수정: 0.00원 → {self._format_price(latest_price)}원")
            
            # 🆕 손절가/익절가가 0인 경우 수정
            if position.entry_price > 0:
                if position.stop_loss_price == 0.0:
                    position.stop_loss_price = position.entry_price * (1 - self.stop_loss_pct / 100)
                    needs_fix = True
                    print(f"🔧 {coin} 손절가 수정: 0.00원 → {self._format_price(position.stop_loss_price)}원")
                
                if position.take_profit_price == 0.0:
                    position.take_profit_price = position.entry_price * (1 + self.take_profit_pct / 100)
                    needs_fix = True
                    print(f"🔧 {coin} 익절가 수정: 0.00원 → {self._format_price(position.take_profit_price)}원")
            
            # 🆕 수정된 포지션을 DB에 저장
            if needs_fix:
                self.save_position_to_db(coin)
                fixed_count += 1
        
        if fixed_count > 0:
            print(f"✅ {fixed_count}개 포지션의 가격 정보 수정 완료")
    
    def _safe_convert_to_int(self, value) -> int:
        """안전한 정수 변환 (바이너리 데이터 처리)"""
        try:
            if value is None:
                return 0
            if isinstance(value, bytes):
                # 바이너리 데이터를 문자열로 디코딩 후 정수 변환 시도
                try:
                    decoded = value.decode('utf-8')
                    return int(decoded)
                except (UnicodeDecodeError, ValueError):
                    # 디코딩 실패 시 현재 타임스탬프로 대체
                    return int(datetime.now().timestamp())
            if isinstance(value, str):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            return 0
        except (ValueError, TypeError):
            return 0
    
    def _safe_convert_to_float(self, value) -> float:
        """안전한 실수 변환 (바이너리 데이터 처리)"""
        try:
            if value is None:
                return 0.0
            if isinstance(value, bytes):
                # 바이너리 데이터인 경우 0.0으로 대체
                return 0.0
            if isinstance(value, str):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _get_historical_price(self, coin: str, timestamp: int) -> float:
        """과거 특정 시점의 가격 조회"""
        try:
            with sqlite3.connect(DB_PATH) as conn:  # 🔧 realtime_candles.db 사용
                # 🆕 가장 가까운 시점의 캔들 조회
                query = """
                SELECT close FROM candles 
                WHERE coin = ? AND timestamp <= ? 
                ORDER BY timestamp DESC LIMIT 1
                """
                result = conn.execute(query, (coin, timestamp)).fetchone()
                
                if result:
                    return float(result[0])
                else:
                    return 0.0
                    
        except Exception as e:
            return 0.0
    
    def _get_latest_price(self, coin: str) -> float:
        """🚀 최적화된 최신 가격 조회 (캐시 + 배치 처리)"""
        try:
            # 🚀 캐시된 가격 확인
            cache_key = f"price_{coin}"
            if cache_key in self.price_cache:
                cached_data = self.price_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['price']
            
            # 🚀 배치 업데이트가 필요한 경우
            current_time = time.time()
            if (current_time - self.last_batch_update > 30 or 
                len(self.position_update_batch) >= self.batch_size):
                self._update_price_batch()
            
            # 개별 조회 (배치에 없는 경우)
            with sqlite3.connect(DB_PATH) as conn:  # 🔧 realtime_candles.db 사용
                # 🚀 최적화된 쿼리: 여러 인터벌을 한 번에 조회
                intervals = ['15m', '30m', '240m', '1d']
                placeholders = ', '.join(['?' for _ in intervals])
                
                query = f"""
                SELECT interval, close FROM (
                    SELECT interval, close, 
                           ROW_NUMBER() OVER (PARTITION BY interval ORDER BY timestamp DESC) as rn
                    FROM candles 
                    WHERE coin = ? AND interval IN ({placeholders})
                ) ranked
                WHERE rn = 1 AND close > 0
                ORDER BY 
                    CASE interval 
                        WHEN '15m' THEN 1 
                        WHEN '30m' THEN 2 
                        WHEN '240m' THEN 3 
                        WHEN '1d' THEN 4 
                    END
                LIMIT 1
                """
                
                result = conn.execute(query, (coin, *intervals)).fetchone()
                
                if result:
                    price = float(result[1])
                    # 캐시에 저장
                    self.price_cache[cache_key] = {
                        'price': price,
                        'timestamp': time.time()
                    }
                    return price
                
                return 0.0
                
        except Exception as e:
            print(f"⚠️ {coin} 가격 조회 오류: {e}")
            return 0.0
    
    def _update_price_batch(self):
        """🚀 배치 가격 업데이트"""
        try:
            if not self.position_update_batch:
                return
            
            # 배치로 가격 조회
            coins = list(set(self.position_update_batch))
            placeholders = ', '.join(['?' for _ in coins])
            
            with sqlite3.connect(DB_PATH) as conn:  # 🔧 realtime_candles.db 사용
                df = pd.read_sql(f"""
                    SELECT coin, close FROM (
                        SELECT coin, close, 
                               ROW_NUMBER() OVER (PARTITION BY coin ORDER BY timestamp DESC) as rn
                        FROM candles 
                        WHERE coin IN ({placeholders}) AND interval = '15m'
                    ) ranked
                    WHERE rn = 1 AND close > 0
                """, conn, params=coins)
                
                # 캐시 업데이트
                current_time = time.time()
                for _, row in df.iterrows():
                    cache_key = f"price_{row['coin']}"
                    self.price_cache[cache_key] = {
                        'price': float(row['close']),
                        'timestamp': current_time
                    }
            
            # 배치 초기화
            self.position_update_batch.clear()
            self.last_batch_update = current_time
            
        except Exception as e:
            print(f"⚠️ 배치 가격 업데이트 오류: {e}")
    
    def _format_price(self, price: float) -> str:
        """가격 포맷팅: 1원 미만은 소수점 4자리, 100원 미만은 소수점 2자리, 100원 이상은 천단위 콤마"""
        try:
            if price == 0:
                return "0"
            
            # 1원 미만인 경우 소수점 4자리까지 정확히 표시
            if price < 1.0:
                return f"{price:.4f}"
            
            # 1원 이상 100원 미만인 경우 소수점 2자리까지 표시
            if price < 100.0:
                return f"{price:.2f}"
            
            # 100원 이상인 경우 천단위 콤마 추가
            return f"{int(price):,}"
                
        except Exception as e:
            return f"{price}"
    
    def open_position(self, coin: str, price: float, signal_score: float, timestamp: int) -> bool:
        """포지션 열기"""
        try:
            # 🆕 이미 보유 중인지 확인
            if coin in self.positions:
                return False
            
            print(f"✅ {coin} 포지션 열기 가능 확인됨")
            
            # 🎯 진입가는 시그널에서 전달받은 price를 그대로 사용
            entry_price = price
            
            # 🆕 현재가 조회 (수익률 계산용)
            try:
                current_price = self._get_latest_price(coin)
                if current_price > 0:
                    pass
                else:
                    current_price = entry_price  # 현재가 조회 실패 시 진입가 사용
            except Exception as e:
                current_price = entry_price  # 오류 시 진입가 사용
            
            try:
                self.positions[coin] = VirtualPosition(
                    coin=coin,
                    entry_price=entry_price,  # 진입가는 시그널에서 전달받은 가격
                    quantity=1.0,  # 수량은 1로 고정 (수익률 계산용)
                    entry_timestamp=timestamp,
                    entry_signal_score=signal_score,
                    current_price=current_price,  # 현재가는 별도로 조회한 가격
                    profit_loss_pct=0.0,
                    holding_duration=0,
                    max_profit_pct=0.0,
                    max_loss_pct=0.0,
                    stop_loss_price=entry_price * (1 - self.stop_loss_pct / 100),  # 10% 손절
                    take_profit_price=entry_price * (1 + self.take_profit_pct / 100),  # 50% 익절
                    last_updated=timestamp
                )
                
                # 🆕 DB에 저장
                self.save_position_to_db(coin)
                
                print(f"🆕 포지션 열기: {get_korean_name(coin)} @ {self._format_price(entry_price)}원")
                return True
                
            except Exception as e:
                return False
                
        except Exception as e:
            return False
    
    def process_signal(self, signal: SignalInfo):
        """🆕 순수 시그널과 보유 정보를 조합하여 최종 액션 결정"""
        current_time = int(datetime.now().timestamp())
        current_price = signal.price
        
        # 🆕 보유 중인 포지션이 있는 경우
        if signal.coin in self.positions:
            # 🆕 포지션을 최신 시장 데이터로 업데이트
            try:
                latest_price = self._get_latest_price(signal.coin)
                if latest_price > 0:
                    self.update_position(signal.coin, latest_price, current_time)
                    current_price = latest_price
            except Exception as e:
                print(f"⚠️ {signal.coin} 포지션 업데이트 오류: {e}")

            position = self.positions[signal.coin]

            # 🎯 시그널 액션에 따라 처리 (realtime_signal_selector가 이미 정교하게 계산함)
            if signal.action == SignalAction.SELL:
                self._close_position(signal.coin, current_price, current_time, 'sell')
                print(f"{get_korean_name(signal.coin)} : 매도 (시그널) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간 (점수: {signal.signal_score:.3f})")
            elif signal.action == SignalAction.HOLD:
                print(f"{get_korean_name(signal.coin)} : 홀딩 {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간")
            elif signal.action == SignalAction.BUY:
                print(f"{get_korean_name(signal.coin)} : 보유 중 {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간 (추가 매수 신호 무시)")

            # 🎯 추가 안전 장치: 극단적 손익 시 강제 청산
            if position.profit_loss_pct >= 50.0:  # 익절
                self._close_position(signal.coin, current_price, current_time, 'take_profit')
                print(f"{get_korean_name(signal.coin)} : 매도 (익절) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간")
            elif position.profit_loss_pct <= -10.0:  # 손절
                self._close_position(signal.coin, current_price, current_time, 'stop_loss')
                print(f"{get_korean_name(signal.coin)} : 매도 (손절) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간")

        # 🆕 미보유 시 BUY 시그널이면 매수
        elif signal.action == SignalAction.BUY:
            if self.can_open_position(signal.coin):
                if self.open_position(signal.coin, current_price, signal.signal_score, current_time):
                    print(f"{get_korean_name(signal.coin)} : 매수 {self._format_price(current_price)}원 (시그널점수: {signal.signal_score:.3f}, 신뢰도: {signal.confidence:.2f})")
    
    def _combine_signal_with_position(self, signal: SignalInfo, position: VirtualPosition, current_price: float) -> str:
        """🆕 순수 시그널과 보유 정보를 조합하여 최종 액션 결정 (실전매매와 동일한 로직)"""
        try:
            signal_score = signal.signal_score
            confidence = signal.confidence
            profit_loss_pct = position.profit_loss_pct
            
            # 🎯 익절 조건 (수익률 50% 이상) - 실전매매와 동일
            if profit_loss_pct >= 50.0:
                return 'take_profit'
            
            # 🎯 손절 조건 (손실 10% 이상) - 실전매매와 동일
            if profit_loss_pct <= -10.0:
                return 'stop_loss'
            
            # 🎯 학습 기반 매도 조건 (시그널 점수 기반) - 실전매매와 동일
            if signal_score < -0.5:  # 강한 매도 시그널
                return 'sell'
            elif signal_score < -0.3:  # 매도 시그널
                return 'sell'
            elif signal_score < -0.2:
                return 'sell'
            elif signal_score < -0.1:
                return 'sell'
            
            # 🎯 학습 기반 매수 조건 (시그널 점수 기반) - 실전매매와 동일
            elif signal_score > 0.5:  # 강한 매수 시그널
                return 'buy'
            elif signal_score > 0.3:  # 매수 시그널
                return 'buy'
            elif signal_score > 0.2:
                return 'buy'
            elif signal_score > 0.1:
                return 'buy'
            
            # 🎯 중립 구간 (홀딩)
            else:
                return 'hold'
                
        except Exception as e:
            print(f"⚠️ 시그널-포지션 결합 오류: {e}")
            return 'hold'
    
    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약 (DB에서 전체 거래 내역 조회)"""
        try:
            # 🆕 DB에서 전체 거래 내역 조회하여 정확한 통계 계산
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                df = pd.read_sql("""
                    SELECT * FROM virtual_trade_history 
                    ORDER BY exit_timestamp DESC
                """, conn)
                
                if df.empty:
                    # 거래 내역이 없으면 기본값 반환
                    return {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate_pct': 0.0,
                        'total_profit_pct': 0.0,
                        'max_drawdown_pct': 0.0,
                        'active_positions': len(self.positions),
                        'max_positions': self.max_positions
                    }
                
                # 전체 통계 계산
                total_trades = len(df)
                winning_trades = len(df[df['profit_loss_pct'] > 0])
                losing_trades = len(df[df['profit_loss_pct'] <= 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_profit_pct = df['profit_loss_pct'].sum()
                max_drawdown_pct = df['profit_loss_pct'].min() if len(df) > 0 else 0.0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate_pct': win_rate,
                    'total_profit_pct': total_profit_pct,
                    'max_drawdown_pct': max_drawdown_pct,
                    'active_positions': len(self.positions),
                    'max_positions': self.max_positions
                }
        except Exception as e:
            print(f"⚠️ 포트폴리오 요약 조회 오류: {e}")
            # 오류 발생 시 인스턴스 변수 사용 (폴백)
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            return {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate_pct': win_rate,
                'total_profit_pct': self.total_profit_pct,
                'max_drawdown_pct': self.max_drawdown,
                'active_positions': len(self.positions),
                'max_positions': self.max_positions
            }
    
    def save_performance_stats(self):
        """성과 통계 저장"""
        try:
            portfolio_stats = self.get_portfolio_summary()
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO virtual_performance_stats (
                        timestamp, total_trades, winning_trades, losing_trades,
                        win_rate, total_profit_pct, max_drawdown_pct, active_positions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(datetime.now().timestamp()),
                    portfolio_stats['total_trades'],
                    portfolio_stats['winning_trades'],
                    portfolio_stats['losing_trades'],
                    portfolio_stats['win_rate_pct'],
                    portfolio_stats['total_profit_pct'],
                    portfolio_stats['max_drawdown_pct'],
                    portfolio_stats['active_positions']
                ))
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 성과 통계 저장 오류: {e}")
    
    def print_trading_status(self):
        """거래 상태 출력 (간소화)"""
        try:
            # 🆕 포트폴리오 통계 계산
            portfolio_stats = self.get_portfolio_summary()
            
            # 🆕 성과 통계 저장
            self.save_performance_stats()
            
            # 🆕 간단한 상태 출력
            print(f"📊 총거래: {portfolio_stats['total_trades']}회, 승률: {portfolio_stats['win_rate_pct']:.1f}%, 수익률: {portfolio_stats['total_profit_pct']:.2f}%, 활성: {portfolio_stats['active_positions']}개")
            
        except Exception as e:
            print(f"⚠️ 거래 상태 출력 오류: {e}")
            # 기본 정보라도 출력
            print(f"📊 활성 포지션: {len(self.positions)}개")
    
    def print_24h_performance_report(self):
        """24시간 성과 리포트 출력"""
        try:
            current_timestamp = int(datetime.now().timestamp())
            day_ago_timestamp = current_timestamp - (24 * 3600)
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 24시간 내 거래 히스토리 조회
                df = pd.read_sql("""
                    SELECT * FROM virtual_trade_history 
                    WHERE exit_timestamp >= ?
                    ORDER BY exit_timestamp DESC
                """, conn, params=(day_ago_timestamp,))
                
                if df.empty:
                    print(f"\n📊 24시간 성과 리포트 ({datetime.fromtimestamp(day_ago_timestamp).strftime('%Y-%m-%d %H:%M')} ~ {datetime.now().strftime('%Y-%m-%d %H:%M')})")
                    print(f"  📈 총 거래: 0회")
                    print(f"  ✅ 승리: 0회")
                    print(f"  ❌ 패배: 0회")
                    print(f"  🎯 승률: 0.0%")
                    print(f"  📊 총 수익률: +0.00%")
                    print(f"  📊 평균 수익률: +0.00%")
                    print(f"  📈 최고 수익: +0.00%")
                    print(f"  📉 최대 손실: +0.00%")
                    print(f"\n🔍 액션별 상세 통계:")
                    print(f"  거래 내역 없음")
                    print(f"\n📋 완료된 거래 내역 (24시간):")
                    print(f"  거래 내역 없음")
                    return
                
                # 24시간 통계 계산
                total_trades_24h = len(df)
                winning_trades_24h = len(df[df['profit_loss_pct'] > 0])
                losing_trades_24h = len(df[df['profit_loss_pct'] <= 0])
                win_rate_24h = (winning_trades_24h / total_trades_24h * 100) if total_trades_24h > 0 else 0
                total_profit_24h = df['profit_loss_pct'].sum()
                avg_profit_24h = df['profit_loss_pct'].mean()
                max_profit_24h = df['profit_loss_pct'].max()
                max_loss_24h = df['profit_loss_pct'].min()
                
                # 액션별 통계
                action_stats = df.groupby('action').agg({
                    'profit_loss_pct': ['count', 'sum', 'mean'],
                    'holding_duration': 'mean'
                }).round(2)
                
                print(f"\n📊 24시간 성과 리포트 ({datetime.fromtimestamp(day_ago_timestamp).strftime('%Y-%m-%d %H:%M')} ~ {datetime.now().strftime('%Y-%m-%d %H:%M')})")
                print(f"  📈 총 거래: {total_trades_24h}회")
                print(f"  ✅ 승리: {winning_trades_24h}회")
                print(f"  ❌ 패배: {losing_trades_24h}회")
                print(f"  🎯 승률: {win_rate_24h:.1f}%")
                print(f"  📊 총 수익률: {total_profit_24h:+.2f}%")
                print(f"  📊 평균 수익률: {avg_profit_24h:+.2f}%")
                print(f"  📈 최고 수익: {max_profit_24h:+.2f}%")
                print(f"  📉 최대 손실: {max_loss_24h:+.2f}%")
                
                # 액션별 상세 통계
                print(f"\n🔍 액션별 상세 통계:")
                for action in df['action'].unique():
                    action_df = df[df['action'] == action]
                    action_count = len(action_df)
                    action_profit = action_df['profit_loss_pct'].sum()
                    action_avg = action_df['profit_loss_pct'].mean()
                    action_win_rate = (len(action_df[action_df['profit_loss_pct'] > 0]) / action_count * 100) if action_count > 0 else 0
                    
                    action_name = {
                        'buy': '매수',
                        'sell': '매도',
                        'take_profit': '익절',
                        'stop_loss': '손절'
                    }.get(action, action)
                    
                    print(f"  {action_name}: {action_count}회, 수익률 {action_profit:+.2f}%, 평균 {action_avg:+.2f}%, 승률 {action_win_rate:.1f}%")
                
                # 🆕 완료된 거래 내역 모두 출력
                print(f"\n📋 완료된 거래 내역 (24시간):")
                for _, trade in df.iterrows():
                    # 🚨 타임스탬프 안전 변환 및 검증
                    entry_timestamp = self._safe_convert_to_int(trade['entry_timestamp'])
                    exit_timestamp = self._safe_convert_to_int(trade['exit_timestamp'])
                    holding_duration = self._safe_convert_to_int(trade['holding_duration'])
                    
                    # 🚨 보유시간 재계산 (정확성 보장)
                    if entry_timestamp > 0 and exit_timestamp > 0:
                        actual_holding_duration = exit_timestamp - entry_timestamp
                        holding_hours = actual_holding_duration / 3600  # 초를 시간으로 변환
                    else:
                        holding_hours = holding_duration / 3600 if holding_duration > 0 else 0.0
                    
                    # 🚨 수익률 재계산 (정확성 보장)
                    entry_price = self._safe_convert_to_float(trade['entry_price'])
                    exit_price = self._safe_convert_to_float(trade['exit_price'])
                    
                    if entry_price > 0:
                        actual_profit_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        actual_profit_pct = trade['profit_loss_pct']
                    
                    entry_time = datetime.fromtimestamp(entry_timestamp).strftime('%H:%M') if entry_timestamp > 0 else "00:00"
                    exit_time = datetime.fromtimestamp(exit_timestamp).strftime('%H:%M') if exit_timestamp > 0 else "00:00"
                    
                    # 액션 이름 한글화
                    action_name = {
                        'buy': '매수',
                        'sell': '매도',
                        'take_profit': '익절',
                        'stop_loss': '손절',
                        'cleanup': '정리'
                    }.get(trade['action'], trade['action'])
                    
                    # 수익률에 따른 이모지
                    profit_emoji = "🟢" if actual_profit_pct > 0 else "🔴"
                    
                    # 🆕 진입가와 종료가 포맷팅
                    entry_price_str = self._format_price(entry_price)
                    exit_price_str = self._format_price(exit_price)
                    
                    print(f"  {profit_emoji} {get_korean_name(trade['coin'])}: {action_name} | "
                          f"진입 {entry_time} @ {entry_price_str}원 → 종료 {exit_time} @ {exit_price_str}원 | "
                          f"보유 {holding_hours:.1f}시간 | "
                          f"수익률 {actual_profit_pct:+.2f}%")
                
                # 🆕 코인별 누적 수익률 계산 (완료된 거래만)
                coin_profit_summary = df.groupby('coin').agg({
                    'profit_loss_pct': 'sum',
                    'coin': 'count'
                }).rename(columns={'coin': 'trade_count'})
                
                # 상위 수익 코인 (완료된 거래 기준)
                top_profit_coins = coin_profit_summary.sort_values('profit_loss_pct', ascending=False).head(5)
                if not top_profit_coins.empty:
                    print(f"\n🏆 상위 수익 코인 (완료된 거래 기준):")
                    for coin, row in top_profit_coins.iterrows():
                        print(f"  {get_korean_name(coin)}: {row['profit_loss_pct']:+.2f}% (거래 {row['trade_count']}회)")
                
                # 상위 손실 코인 (완료된 거래 기준)
                top_loss_coins = coin_profit_summary.sort_values('profit_loss_pct', ascending=True).head(5)
                if not top_loss_coins.empty:
                    print(f"\n📉 상위 손실 코인 (완료된 거래 기준):")
                    for coin, row in top_loss_coins.iterrows():
                        print(f"  {get_korean_name(coin)}: {row['profit_loss_pct']:+.2f}% (거래 {row['trade_count']}회)")
                
        except Exception as e:
            print(f"⚠️ 24시간 성과 리포트 오류: {e}")
    
    def start_trading(self):
        """거래 시작"""
        self.is_running = True
        print("🚀 가상매매 시뮬레이터 시작!")
        
        # 기존 포지션 로드
        self.load_positions_from_db()
        
        try:
            while self.is_running:
                # 새로운 시그널로 거래 실행
                new_signals = self.get_new_signals(max_hours_back=6, batch_size=50)
                
                if new_signals:
                    for signal in new_signals:
                        self.process_signal(signal)
                
                # 성과 통계 저장 (5분마다)
                if int(time.time()) % 300 == 0:
                    self.save_performance_stats()
                
                # 거래 상태 출력 (10분마다)
                if int(time.time()) % 600 == 0:
                    self.print_trading_status()
                
                # 대기
                time.sleep(self.trading_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ 거래 중단 요청")
        except Exception as e:
            print(f"⚠️ 거래 오류: {e}")
            traceback.print_exc()
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """거래 중단"""
        self.is_running = False
        
        # 최종 성과 통계 저장
        self.save_performance_stats()
        
        # 최종 상태 출력
        self.print_trading_status()
        
        print("✅ 가상매매 시뮬레이터 완료!")
    
    def _calculate_adaptive_buy_bonus(self, signal: SignalInfo) -> float:
        """적응적 매수 보너스 계산"""
        try:
            # 🎯 시장 상황 분석
            market_context = self._get_market_context()
            
            bonus = 0.0
            
            # 🎯 시장 상황에 따른 적응적 가중치
            if market_context['trend'] == 'bullish':
                # 상승장에서는 다이버전스와 트렌드 강도에 더 높은 가중치
                if signal.rsi_divergence == 'bullish' or signal.macd_divergence == 'bullish':
                    bonus += 0.15  # 상승장에서 다이버전스 보너스 증가
                
                if signal.trend_strength > 0.7:
                    bonus += 0.12  # 상승장에서 트렌드 보너스 증가
            
            elif market_context['trend'] == 'bearish':
                # 하락장에서는 볼린저밴드 스퀴즈와 모멘텀에 더 높은 가중치
                if signal.bb_squeeze > 0.8:
                    bonus += 0.10  # 하락장에서 스퀴즈 보너스 증가
                
                if abs(signal.price_momentum) > 0.05:
                    bonus += 0.08  # 하락장에서 모멘텀 보너스 증가
            
            else:  # 중립장
                # 중립장에서는 균형잡힌 보너스
                if signal.rsi_divergence == 'bullish' or signal.macd_divergence == 'bullish':
                    bonus += 0.10
                
                if signal.trend_strength > 0.7:
                    bonus += 0.08
                
                if signal.bb_squeeze > 0.8:
                    bonus += 0.05
            
            # 🎯 변동성에 따른 보너스 조정
            volatility = market_context.get('volatility', 0.02)
            if volatility > 0.05:  # 고변동성
                bonus *= 0.8  # 고변동성에서는 보너스 감소
            elif volatility < 0.02:  # 저변동성
                bonus *= 1.2  # 저변동성에서는 보너스 증가
            
            return min(bonus, 0.1)  # 최대 10% 보너스 제한 (더 엄격하게)
            
        except Exception as e:
            print(f"⚠️ 적응적 매수 보너스 계산 오류: {e}")
            return 0.0
    
    def _get_dynamic_buy_threshold(self, coin: str) -> float:
        """학습 기반 동적 매수 임계값 조정 (RL 시스템 연동)"""
        try:
            # 🎯 코인별 과거 성과 분석
            performance_score = self._analyze_coin_performance(coin)
            
            # 🎯 시장 상황 분석
            market_score = self._analyze_market_conditions()
            
            # 🎯 기본 임계값 (0.1 - 학습된 전략 신뢰)
            base_threshold = 0.1
            
            # 🎯 성과 기반 조정 (매우 작은 조정)
            if performance_score > 0.7:  # 좋은 성과
                base_threshold -= 0.01  # 임계값 낮춤 (더 쉽게 매수)
            elif performance_score < 0.3:  # 나쁜 성과
                base_threshold += 0.02  # 임계값 높임 (더 엄격하게 매수)
            
            # 🎯 시장 상황 기반 조정 (매우 작은 조정)
            if market_score > 0.7:  # 좋은 시장 상황
                base_threshold -= 0.01
            elif market_score < 0.3:  # 나쁜 시장 상황
                base_threshold += 0.01
            
            # 🆕 RL 학습 결과 기반 추가 조정 (향후 구현)
            # rl_adjustment = self._get_rl_based_threshold_adjustment(coin)
            # base_threshold += rl_adjustment
            
            return max(0.05, min(0.3, base_threshold))  # 0.05~0.3 범위로 제한 (학습된 전략 신뢰)
            
        except Exception as e:
            print(f"⚠️ 동적 매수 임계값 계산 오류 ({coin}): {e}")
            return 0.1  # 기본값 반환 (학습된 전략 신뢰)
    
    def _get_rl_based_threshold_adjustment(self, coin: str) -> float:
        """RL 시스템 학습 결과 기반 임계값 조정 (향후 구현)"""
        try:
            # 🎯 RL 시스템의 Q-테이블에서 해당 코인의 학습 결과 조회
            # 🎯 승률, 평균 수익률 등 기반으로 임계값 조정
            # 🎯 현재는 0.0 반환 (향후 RL 시스템과 연동 시 구현)
            return 0.0
            
        except Exception as e:
            print(f"⚠️ RL 기반 임계값 조정 오류 ({coin}): {e}")
            return 0.0
    
    def _analyze_coin_performance(self, coin: str) -> float:
        """코인별 과거 거래 성과 분석"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 최근 30일간 거래 성과 분석
                thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
                
                df = pd.read_sql("""
                    SELECT profit_loss_pct FROM virtual_trade_history 
                    WHERE coin = ? AND exit_timestamp > ?
                    ORDER BY exit_timestamp DESC
                """, conn, params=(coin, thirty_days_ago))
                
                if df.empty:
                    return 0.5  # 거래 내역 없으면 중립
                
                # 평균 수익률
                avg_profit = df['profit_loss_pct'].mean()
                
                # 승률
                win_rate = len(df[df['profit_loss_pct'] > 0]) / len(df)
                
                # 성과 점수 계산 (0~1)
                performance_score = (avg_profit + 10) / 20 * 0.6 + win_rate * 0.4
                
                return max(0.0, min(1.0, performance_score))
                
        except Exception as e:
            print(f"⚠️ 코인 성과 분석 오류 ({coin}): {e}")
            return 0.5
    
    def _extract_signal_pattern_for_feedback(self, signal: SignalInfo) -> str:
        """시그널에서 피드백 학습용 패턴 추출 (realtime_signal_selector와 동일한 방식)"""
        try:
            if not signal:
                print(f"⚠️ 시그널이 None이므로 unknown_pattern 반환")
                return 'unknown_pattern'
            
            # 🚀 핵심 시그널 패턴 추출 (RSI, Direction, BB, Volume 기반)
            rsi_level = self._discretize_rsi(signal.rsi)
            direction = signal.integrated_direction if hasattr(signal, 'integrated_direction') and signal.integrated_direction else 'neutral'
            bb_position = signal.bb_position if hasattr(signal, 'bb_position') and signal.bb_position else 'unknown'
            volume_level = self._discretize_volume(signal.volume_ratio)
            
            # 🚨 기본값 검증 및 수정
            if not direction or direction == '' or direction == 'unknown':
                direction = 'neutral'
            if not bb_position or bb_position == '' or bb_position == 'unknown':
                bb_position = 'middle'  # unknown 대신 middle 사용
            if not volume_level or volume_level == '':
                volume_level = 'normal'
            
            # 패턴 조합
            pattern = f"{rsi_level}_{direction}_{bb_position}_{volume_level}"
            print(f"🧬 패턴 추출: {signal.coin} = {pattern} (RSI: {signal.rsi:.1f}, Direction: {direction}, BB: {bb_position}, Volume: {signal.volume_ratio:.2f})")
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
    
    def _discretize_volume(self, volume_ratio: float) -> str:
        """거래량 비율을 이산화"""
        if volume_ratio < 0.5:
            return 'low'
        elif volume_ratio < 1.5:
            return 'normal'
        else:
            return 'high'
    
    def _analyze_market_conditions(self) -> float:
        """전체 시장 상황 분석"""
        try:
            # 상위 10개 코인의 평균 시그널 점수로 시장 상황 판단
            try:
                from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
                available = get_available_coins_and_intervals()
                top_coins = sorted(list({c for c, _ in available}))
            except Exception:
                top_coins = [os.getenv('DEFAULT_COIN', 'BTC')]
            
            total_score = 0.0
            valid_count = 0
            
            for coin in top_coins:
                signal = self._get_current_signal_info(coin)
                if signal:
                    total_score += signal.signal_score
                    valid_count += 1
            
            if valid_count > 0:
                avg_score = total_score / valid_count
                # -1~1 범위를 0~1 범위로 변환
                market_score = (avg_score + 1) / 2
                return max(0.0, min(1.0, market_score))
            else:
                return 0.5
                
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류: {e}")
            return 0.5

def main():
    """메인 실행 함수"""
    print("🆕 가상매매 시뮬레이터 시작")
    
    # 시뮬레이터 초기화
    trader = VirtualTrader()
    
    try:
        print("\n🚀 [STEP 1] 보유 포지션 확인")
        if trader.positions:
            # 🆕 보유 포지션들의 최신 시장 데이터로 업데이트
            current_time = int(datetime.now().timestamp())
            for coin in list(trader.positions.keys()):
                try:
                    latest_price = trader._get_latest_price(coin)
                    if latest_price > 0:
                        trader.update_position(coin, latest_price, current_time)
                except Exception as e:
                    print(f"⚠️ {coin} 포지션 업데이트 오류: {e}")
            
            # 🆕 진입가가 0인 포지션들 수정
            fixed_count = 0
            for coin, position in trader.positions.items():
                if position.entry_price == 0.0:
                    latest_price = trader._get_latest_price(coin)
                    if latest_price > 0:
                        position.entry_price = latest_price
                        position.current_price = latest_price
                        trader.save_position_to_db(coin)
                        fixed_count += 1
                        print(f"🔧 {get_korean_name(coin)} 진입가 수정: 0.00원 → {trader._format_price(latest_price)}원")
            
            # 🆕 보유 코인 상세 정보 출력
            print("보유 코인:")
            for coin, position in trader.positions.items():
                print(f"   {get_korean_name(coin)}: 진입가 {trader._format_price(position.entry_price)}원, 수익률 {position.profit_loss_pct:+.2f}%, 보유시간 {position.holding_duration//3600}시간")
        else:
            print("📊 보유 포지션 없음")

        print("\n🚀 [STEP 2] 시그널 데이터 조회")
        new_signals = trader.get_new_signals(max_hours_back=24, batch_size=1000)
        
        if new_signals:
            print(f"📊 {len(new_signals)}개 새 시그널 처리 중...")
            
            print("\n🚀 [STEP 3] 순수 시그널과 보유 정보 조합하여 최종 액션 결정")
            
            # 🆕 중복 처리 방지를 위한 세트
            processed_coins = set()
            
            for signal in new_signals:
                # 🆕 이미 처리된 코인은 건너뛰기
                if signal.coin in processed_coins:
                    continue
                
                trader.process_signal(signal)
                processed_coins.add(signal.coin)
            
            print("✅ 가상매매 거래 실행 완료")
        else:
            print("ℹ️ 새로운 시그널이 없습니다.")
        
        print("\n🚀 [STEP 4] 최종 포지션 상태 확인")
        trader.print_trading_status()
        
        print("\n🚀 [STEP 5] 24시간 성과 리포트 출력")
        trader.print_24h_performance_report()
        
    except Exception as e:
        print(f"⚠️ 거래 실행 오류: {e}")
        traceback.print_exc()
    
    print("✅ 가상매매 시뮬레이터 완료!")

if __name__ == "__main__":
    main() 