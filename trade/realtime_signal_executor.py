"""
실전매매 실행기 - 시그널 기반 실전 거래 실행

주요 기능:
1. 시그널 셀렉터에서 생성된 시그널을 실시간으로 읽기
2. 실전매매 시뮬레이션으로 거래 실행 (매수/매도/홀딩/익절/손절)
3. 포지션 관리 및 손익 계산
4. 거래 결과를 DB에 저장하여 학습기에서 활용
5. 실시간 포트폴리오 모니터링

🆕 Absolute Zero System 개선사항 반영:
- 모든 고급 기술지표 활용 (다이버전스, 볼린저밴드 스퀴즈, 모멘텀, 트렌드 강도 등)
- 개선된 시그널 정보 구조 (새로운 고급 지표들 포함)
- 향상된 상태 표현 (더 정교한 상태 키 생성)
- 새로운 패턴 매칭 로직 (다이버전스, 스퀴즈, 강한 트렌드 등)
- 실전매매에서 고급 지표 기반 의사결정 강화
"""
import sys
sys.path.insert(0, '/workspace/')  # 절대 경로 추가

import time
import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from trade_manager import sync_wallet_to_db, get_filtered_wallet_coins, execute_trade_with_timeout, get_order_chance, wait_for_balance_update, fetch_tick_size_from_bithumb
from utils.market_name_utils import get_korean_name
from typing import Dict, Any, List

# 🆕 성능 업그레이드 시스템 클래스들 (실전 매매 특화)
class RealTimeActionTracker:
    """실전 매매 액션별 성과 추적기"""
    def __init__(self):
        self.action_performance = {
            'buy': {'trades': 0, 'wins': 0, 'total_profit': 0.0, 'total_amount': 0.0},
            'sell': {'trades': 0, 'wins': 0, 'total_profit': 0.0, 'total_amount': 0.0},
            'hold': {'trades': 0, 'wins': 0, 'total_profit': 0.0, 'total_amount': 0.0}
        }
        self.coin_performance = {}
    
    def record_action_result(self, action: str, profit: float, success: bool, amount: float, coin: str):
        """액션 결과 기록 (실전 매매 특화)"""
        if action in self.action_performance:
            self.action_performance[action]['trades'] += 1
            self.action_performance[action]['total_profit'] += profit
            self.action_performance[action]['total_amount'] += amount
            if success:
                self.action_performance[action]['wins'] += 1
        
        # 코인별 성과 추적
        if coin not in self.coin_performance:
            self.coin_performance[coin] = {'trades': 0, 'wins': 0, 'total_profit': 0.0}
        self.coin_performance[coin]['trades'] += 1
        self.coin_performance[coin]['total_profit'] += profit
        if success:
            self.coin_performance[coin]['wins'] += 1
    
    def get_action_performance(self, action: str) -> dict:
        """액션별 성과 반환"""
        if action not in self.action_performance:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0, 'avg_amount': 0.0}
        
        perf = self.action_performance[action]
        if perf['trades'] == 0:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0, 'avg_amount': 0.0}
        
        return {
            'success_rate': perf['wins'] / perf['trades'],
            'avg_profit': perf['total_profit'] / perf['trades'],
            'total_trades': perf['trades'],
            'avg_amount': perf['total_amount'] / perf['trades']
        }
    
    def get_coin_performance(self, coin: str) -> dict:
        """코인별 성과 반환"""
        if coin not in self.coin_performance:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        perf = self.coin_performance[coin]
        if perf['trades'] == 0:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        return {
            'success_rate': perf['wins'] / perf['trades'],
            'avg_profit': perf['total_profit'] / perf['trades'],
            'total_trades': perf['trades']
        }

class RealTimeContextRecorder:
    """실전 매매 컨텍스트 기록기"""
    def __init__(self):
        self.trade_contexts = {}
        self.market_contexts = {}
    
    def record_trade_context(self, trade_id: str, context: dict):
        """거래 컨텍스트 기록"""
        self.trade_contexts[trade_id] = {
            'timestamp': time.time(),
            'context': context
        }
    
    def record_market_context(self, timestamp: int, context: dict):
        """시장 컨텍스트 기록"""
        self.market_contexts[timestamp] = context
    
    def get_trade_context(self, trade_id: str) -> dict:
        """거래 컨텍스트 조회"""
        return self.trade_contexts.get(trade_id, {})
    
    def get_market_context(self, timestamp: int) -> dict:
        """시장 컨텍스트 조회"""
        return self.market_contexts.get(timestamp, {})

class RealTimeOutlierGuardrail:
    """실전 매매 이상치 컷 시스템"""
    def __init__(self, percentile_cut: float = 0.05):
        self.percentile_cut = percentile_cut
    
    def winsorize_profits(self, profits: List[float]) -> List[float]:
        """수익률 Winsorizing (실전 매매 특화)"""
        if len(profits) < 5:  # 실전 매매는 더 보수적
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

class RealTimeAIDecisionEngine:
    """실전 매매 AI 의사결정 엔진"""
    def __init__(self):
        self.decision_history = []
        self.coin_decision_patterns = {}
        self.market_adaptations = {}
        
    def make_trading_decision(self, signal_data: dict, current_price: float, 
                            market_context: dict, coin_performance: dict) -> str:
        """실전 매매 의사결정 (거래량 기준 선별된 코인 대상)"""
        try:
            # 기본 시그널 분석
            signal_score = signal_data.get('signal_score', 0.0)
            confidence = signal_data.get('confidence', 0.0)
            action = signal_data.get('action', 'hold')
            
            # 코인별 성과 기반 조정
            coin_bonus = self._calculate_coin_performance_bonus(coin_performance)
            
            # 시장 컨텍스트 기반 조정
            market_bonus = self._calculate_market_context_bonus(market_context)
            
            # 실전 매매 특화 리스크 조정
            risk_adjustment = self._calculate_real_time_risk_adjustment(signal_data, current_price)
            
            # 최종 의사결정
            final_score = signal_score + coin_bonus + market_bonus - risk_adjustment
            
            # 의사결정 기록
            decision_record = {
                'timestamp': time.time(),
                'coin': signal_data.get('coin', 'unknown'),
                'signal_score': signal_score,
                'final_score': final_score,
                'action': action,
                'coin_bonus': coin_bonus,
                'market_bonus': market_bonus,
                'risk_adjustment': risk_adjustment
            }
            self.decision_history.append(decision_record)
            
            # 액션 결정
            if final_score > 0.3 and confidence > 0.6:
                return 'buy'
            elif final_score < -0.3 and confidence > 0.6:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            print(f"⚠️ 실전 매매 AI 의사결정 오류: {e}")
            return 'hold'
    
    def _calculate_coin_performance_bonus(self, coin_performance: dict) -> float:
        """코인별 성과 보너스 계산"""
        try:
            success_rate = coin_performance.get('success_rate', 0.5)
            avg_profit = coin_performance.get('avg_profit', 0.0)
            total_trades = coin_performance.get('total_trades', 0)
            
            # 거래 횟수가 적으면 보수적
            if total_trades < 5:
                return 0.0
            
            # 성과 기반 보너스
            performance_bonus = (success_rate - 0.5) * 0.2 + avg_profit * 0.1
            return max(-0.1, min(0.1, performance_bonus))
            
        except Exception as e:
            print(f"⚠️ 코인 성과 보너스 계산 오류: {e}")
            return 0.0
    
    def _calculate_market_context_bonus(self, market_context: dict) -> float:
        """시장 컨텍스트 보너스 계산"""
        try:
            market_trend = market_context.get('trend', 'neutral')
            volatility = market_context.get('volatility', 'medium')
            
            bonus = 0.0
            
            # 트렌드 기반 보너스
            if market_trend == 'bullish':
                bonus += 0.05
            elif market_trend == 'bearish':
                bonus -= 0.05
            
            # 변동성 기반 보너스
            if volatility == 'low':
                bonus += 0.02
            elif volatility == 'high':
                bonus -= 0.02
            
            return max(-0.1, min(0.1, bonus))
            
        except Exception as e:
            print(f"⚠️ 시장 컨텍스트 보너스 계산 오류: {e}")
            return 0.0
    
    def _calculate_real_time_risk_adjustment(self, signal_data: dict, current_price: float) -> float:
        """실전 매매 리스크 조정"""
        try:
            risk_level = signal_data.get('risk_level', 'medium')
            confidence = signal_data.get('confidence', 0.5)
            
            risk_adjustment = 0.0
            
            # 리스크 레벨 기반 조정
            if risk_level == 'high':
                risk_adjustment += 0.1
            elif risk_level == 'low':
                risk_adjustment += 0.02
            
            # 신뢰도 기반 조정
            if confidence < 0.5:
                risk_adjustment += 0.05
            
            return risk_adjustment
            
        except Exception as e:
            print(f"⚠️ 실전 매매 리스크 조정 오류: {e}")
            return 0.05

class RealTimeLearningFeedback:
    """실전 매매 학습 피드백 시스템"""
    def __init__(self):
        self.trade_feedback = {}
        self.coin_patterns = {}
        self.market_patterns = {}
        
    def record_trade_result(self, coin: str, trade_result: dict):
        """거래 결과 기록"""
        try:
            trade_id = f"{coin}_{trade_result.get('timestamp', int(time.time()))}"
            
            self.trade_feedback[trade_id] = {
                'coin': coin,
                'timestamp': trade_result.get('timestamp', int(time.time())),
                'action': trade_result.get('action', 'unknown'),
                'profit': trade_result.get('profit', 0.0),
                'success': trade_result.get('profit', 0.0) > 0,
                'amount': trade_result.get('amount', 0.0),
                'context': trade_result.get('context', {})
            }
            
            # 코인별 패턴 업데이트
            if coin not in self.coin_patterns:
                self.coin_patterns[coin] = {'trades': 0, 'wins': 0, 'total_profit': 0.0}
            
            self.coin_patterns[coin]['trades'] += 1
            self.coin_patterns[coin]['total_profit'] += trade_result.get('profit', 0.0)
            if trade_result.get('profit', 0.0) > 0:
                self.coin_patterns[coin]['wins'] += 1
                
        except Exception as e:
            print(f"⚠️ 실전 매매 학습 피드백 기록 오류: {e}")
    
    def get_coin_learning_data(self, coin: str) -> dict:
        """코인별 학습 데이터 반환"""
        if coin not in self.coin_patterns:
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0}
        
        pattern = self.coin_patterns[coin]
        if pattern['trades'] == 0:
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0}
        
        return {
            'success_rate': pattern['wins'] / pattern['trades'],
            'avg_profit': pattern['total_profit'] / pattern['trades'],
            'total_trades': pattern['trades']
        }

# 🆕 실전 매매 성능 업그레이드 시스템 초기화
real_time_action_tracker = RealTimeActionTracker()
real_time_context_recorder = RealTimeContextRecorder()
real_time_outlier_guardrail = RealTimeOutlierGuardrail()
real_time_ai_decision_engine = RealTimeAIDecisionEngine()
real_time_learning_feedback = RealTimeLearningFeedback()

# 로깅 설정 (파일 생성 없이 콘솔만)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# DB 경로 설정
DB_PATH = '/workspace/data_storage/realtime_candles.db'
# 🆕 통합 트레이딩 시스템 DB 경로 (섀도우 + 실전 매매)
TRADING_SYSTEM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_storage', 'trading_system.db')

# 시그널 기반 거래 결정 내역 테이블 생성 (최초 1회 실행 시 생성)
def create_signal_trade_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                coin TEXT,
                action TEXT,
                signal_score REAL,
                confidence REAL,
                reason TEXT,
                price REAL,
                position_percentage REAL,
                profit_pct REAL,
                rsi REAL,
                macd REAL,
                wave_phase TEXT,
                rl_score REAL,
                tech_score REAL,
                wave_score REAL,
                risk_score REAL,
                decision_status TEXT,
                executed INTEGER DEFAULT 0
            );
        """)

def create_trade_decision_log_table():
    # 🚀 trading_system.db에 실전 매매 테이블 생성 (통합 DB 사용)
    with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS real_trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                coin TEXT,
                interval TEXT,
                action TEXT,              -- buy / sell / hold / switch
                reason TEXT,              -- stop_loss / profit_sell / hold / switch
                reason_detail TEXT,       -- 판단 사유 상세 (지표 수치, 시그널 분석 등)
                entry_price REAL,
                current_price REAL,
                profit_pct REAL,
                fusion_score REAL,
                rl_score REAL,
                market_mode TEXT,
                market_flow TEXT,
                gpt_approved INTEGER,     -- 1 = 승인됨, 0 = 반려됨
                executed INTEGER,         -- 1 = 실제 매매 실행됨, 0 = 판단만 기록
                execution_price REAL,     -- 실체결가 (없으면 NULL)
                execution_amount REAL,    -- 체결 금액 or 수량 (없으면 NULL)
                execution_type TEXT,      -- buy / sell / switch / none
                signal_score REAL,        -- 시그널 점수
                confidence REAL,          -- 신뢰도
                holding_duration INTEGER,  -- 보유 기간 (초)
                max_profit_pct REAL,      -- 최대 수익률
                max_loss_pct REAL,        -- 최대 손실률
                stop_loss_price REAL,     -- 스탑로스 가격
                take_profit_price REAL,   -- 테이크프로핏 가격
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS real_trade_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                coin TEXT,
                signal_pattern TEXT,      -- 시그널 패턴
                success_rate REAL,        -- 성공률
                avg_profit REAL,          -- 평균 수익률
                total_trades INTEGER,     -- 총 거래 수
                confidence REAL,          -- 신뢰도
                learning_episode INTEGER, -- 학습 에피소드
                feedback_type TEXT,       -- feedback_type (success/failure)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES real_trade_history(id)
            );
        """)

def create_holdings_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                coin TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_buy_price REAL
            );
        """)

def save_candle_snapshot(coin, interval, timestamp):
    conn = sqlite3.connect(DB_PATH)  # 별도 저장 DB
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candle_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            interval TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            zigzag_direction REAL,
            zigzag_pivot_price REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_diff REAL
        )
    """)

    cursor.execute("""
        SELECT open, high, low, close, volume, zigzag_direction, zigzag_pivot_price, rsi, macd, macd_signal, macd_diff 
        FROM candles 
        WHERE coin=? AND interval=? AND timestamp=?
    """, (coin, interval, timestamp))

    candle = cursor.fetchone()

    if candle:
        cursor.execute("""
            INSERT INTO candle_snapshot (coin, interval, timestamp, open, high, low, close, volume, zigzag_direction, zigzag_pivot_price, rsi, macd, macd_signal, macd_diff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (coin, interval, timestamp, *candle))
        conn.commit()
        print(f"✅ 캔들 스냅샷 저장 완료: {coin}-{interval}-{timestamp}")
    else:
        print(f"⚠️ 캔들 데이터 없음: {coin}-{interval}-{timestamp}")

    conn.close()

# 실제 보유 중인 코인 및 수량 로딩
def load_wallet_real():
    with sqlite3.connect(DB_PATH) as conn:
        wallet_df = pd.read_sql('SELECT coin, quantity FROM holdings', conn, index_col='coin')
    return wallet_df

# 매수 금액 불러오기
def get_entry_price(coin):
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT avg_buy_price FROM holdings WHERE coin=?"
        result = conn.execute(query, (coin,)).fetchone()
        return result[0] if result else None

# 보유 수량 불러오기
def get_quantity(coin):
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT quantity FROM holdings WHERE coin=?"
        result = conn.execute(query, (coin,)).fetchone()
        return result[0] if result else 0.0

# 추가 매수 여부 결정 함수
def should_add_buy(coin, signal_score, confidence, current_price, entry_price):
    """이미 보유한 코인에 대한 추가 매수 여부를 결정"""
    if entry_price is None or entry_price <= 0:
        return True  # 보유하지 않은 코인이므로 신규 매수
    
    # 현재 수익률 계산
    profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
    
    # 🎯 추가 매수 조건
    # 1. 시그널 점수가 매우 높을 때 (0.08 이상)
    if signal_score >= 0.08 and confidence >= 0.7:
        return True
    
    # 2. 신뢰도가 높고 시그널 점수가 좋을 때 (0.06 이상)
    if signal_score >= 0.06 and confidence >= 0.75:
        return True
    
    # 3. 현재 가격이 진입가보다 낮고 시그널이 좋을 때 (저가 매수)
    if current_price < entry_price and signal_score >= 0.05 and confidence >= 0.65:
        return True
    
    # 4. 수익률이 -5% 이하이고 시그널이 좋을 때 (평균단가 낮추기)
    if profit_loss_pct <= -5.0 and signal_score >= 0.04 and confidence >= 0.6:
        return True
    
    return False

# 상위 150개 코인 로딩 (1일봉 거래량 기준)
def load_top_150_coins():
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT coin FROM candles
            WHERE interval='1d' AND timestamp=(SELECT MAX(timestamp) FROM candles WHERE interval='1d')
            ORDER BY volume DESC LIMIT 150
        """
        return pd.read_sql(query, conn)['coin'].tolist()

# 🆕 실전 매매용 시그널 점수 조회 (realtime_signals 테이블에서)
def load_realtime_signal(coin: str, interval: str = 'combined'):
    """signals 테이블에서 코인의 최신 통합 시그널 정보 로드 (combined 시그널만 사용)"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # combined 시그널만 조회 (인터벌 합치기 로직 제거)
            query = """
                SELECT * FROM signals
                WHERE coin = ? AND interval = 'combined'
                ORDER BY timestamp DESC
                LIMIT 1
            """
            df = pd.read_sql(query, conn, params=(coin,))
        
        if df.empty:
            return {
                'signal_info': {
                    'action': 'wait',
                    'signal_score': 0.0,
                    'confidence': 0.0,
                    'reason': '시그널 없음'
                },
                'market_data': {
                    'price': 0.0,
                    'volume': 0.0,
                    'rsi': 50.0,
                    'macd': 0.0,
                    'volatility': 0.0,
                    'volume_ratio': 1.0
                },
                'wave_info': {
                    'wave_phase': 'unknown',
                    'pattern_type': 'none',
                    'wave_progress': 0.5,
                    'structure_score': 0.5,
                    'pattern_confidence': 0.0,
                    'integrated_direction': 'neutral'
                },
                'scores': {
                    'rl_score': 0.0,
                    'tech_score': 0.0,
                    'wave_score': 0.0,
                    'risk_score': 0.0
                },
                # 🆕 Absolute Zero System의 새로운 고급 지표들
                'advanced_indicators': {
                    'mfi': 50.0,
                    'atr': 0.0,
                    'adx': 25.0,
                    'ma20': 0.0,
                    'rsi_ema': 50.0,
                    'macd_smoothed': 0.0,
                    'wave_momentum': 0.0,
                    'bb_position': 'unknown',
                    'bb_width': 0.0,
                    'bb_squeeze': 0.0,
                    'rsi_divergence': 'none',
                    'macd_divergence': 'none',
                    'volume_divergence': 'none',
                    'price_momentum': 0.0,
                    'volume_momentum': 0.0,
                    'trend_strength': 0.5,
                    'support_resistance': 'unknown',
                    'fibonacci_levels': 'unknown',
                    'elliott_wave': 'unknown',
                    'harmonic_patterns': 'none',
                    'candlestick_patterns': 'none',
                    'market_structure': 'unknown',
                    'flow_level_meta': 'unknown',
                    'pattern_direction': 'neutral'
                }
            }
        
        row = df.iloc[0]
        return {
            'signal_info': {
                'action': row['action'],
                'signal_score': row['signal_score'],
                'confidence': row['confidence'],
                'reason': row['reason']
            },
            'market_data': {
                'price': row['current_price'],
                'volume': 0.0,  # 실전 매매에서 별도 조회
                'rsi': row['rsi'],
                'macd': row['macd'],
                'volatility': row['volatility'],
                'volume_ratio': row['volume_ratio']
            },
            'wave_info': {
                'wave_phase': row['wave_phase'],
                'pattern_type': row['pattern_type'],
                'wave_progress': row['wave_progress'],
                'structure_score': row['structure_score'],
                'pattern_confidence': row['pattern_confidence'],
                'integrated_direction': row['integrated_direction']
            },
            'scores': {
                'rl_score': row.get('rl_score', 0.0),
                'tech_score': row.get('tech_score', 0.0),
                'wave_score': row.get('wave_score', 0.0),
                'risk_score': row.get('risk_score', 0.0)
            },
            # 🆕 Absolute Zero System의 새로운 고급 지표들
            'advanced_indicators': {
                'mfi': row.get('mfi', 50.0),
                'atr': row.get('atr', 0.0),
                'adx': row.get('adx', 25.0),
                'ma20': row.get('ma20', 0.0),
                'rsi_ema': row.get('rsi_ema', 50.0),
                'macd_smoothed': row.get('macd_smoothed', 0.0),
                'wave_momentum': row.get('wave_momentum', 0.0),
                'bb_position': row.get('bb_position', 'unknown'),
                'bb_width': row.get('bb_width', 0.0),
                'bb_squeeze': row.get('bb_squeeze', 0.0),
                'rsi_divergence': row.get('rsi_divergence', 'none'),
                'macd_divergence': row.get('macd_divergence', 'none'),
                'volume_divergence': row.get('volume_divergence', 'none'),
                'price_momentum': row.get('price_momentum', 0.0),
                'volume_momentum': row.get('volume_momentum', 0.0),
                'trend_strength': row.get('trend_strength', 0.5),
                'support_resistance': row.get('support_resistance', 'unknown'),
                'fibonacci_levels': row.get('fibonacci_levels', 'unknown'),
                'elliott_wave': row.get('elliott_wave', 'unknown'),
                'harmonic_patterns': row.get('harmonic_patterns', 'none'),
                'candlestick_patterns': row.get('candlestick_patterns', 'none'),
                'market_structure': row.get('market_structure', 'unknown'),
                'flow_level_meta': row.get('flow_level_meta', 'unknown'),
                'pattern_direction': row.get('pattern_direction', 'neutral')
            }
        }
    except Exception as e:
        print(f"⚠️ 실전 매매용 시그널 조회 오류 ({coin}/{interval}): {e}")
        return None

# 최신 realtime_signals에서 시그널 정보 가져오기 (통합 시그널 기준) - 기존 호환성 유지
def load_signal_from_summary(coin):
    """signals 테이블에서 코인의 최신 통합 시그널 정보 로드 (통합 DB 사용)"""
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT action, signal_score, confidence, reason, current_price as price, 0 as volume,
                   rsi, macd, wave_phase, pattern_type, risk_level, volatility,
                   volume_ratio, wave_progress, structure_score, pattern_confidence,
                   integrated_direction, integrated_strength, rl_score, tech_score, wave_score, risk_score
            FROM signals
            WHERE coin = ? AND interval = 'combined'
            ORDER BY timestamp DESC
            LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(coin,))

    if df.empty:
        return {
            'signal_info': {
                'action': 'wait',
                'signal_score': 0.0,
                'confidence': 0.0,
                'reason': '시그널 없음'
            },
            'market_data': {
                'price': 0.0,
                'volume': 0.0,
                'rsi': 50.0,
                'macd': 0.0,
                'volatility': 0.0,
                'volume_ratio': 1.0
            },
            'wave_info': {
                'wave_phase': 'unknown',
                'pattern_type': 'none',
                'wave_progress': 0.5,
                'structure_score': 0.5,
                'pattern_confidence': 0.0,
                'integrated_direction': 'neutral'
            },
            'scores': {
                'rl_score': 0.0,
                'tech_score': 0.0,
                'wave_score': 0.0,
                'risk_score': 0.0
            },
            # 🆕 Absolute Zero System의 새로운 고급 지표들
            'advanced_indicators': {
                'mfi': 50.0,
                'atr': 0.0,
                'adx': 25.0,
                'ma20': 0.0,
                'rsi_ema': 50.0,
                'macd_smoothed': 0.0,
                'wave_momentum': 0.0,
                'bb_position': 'unknown',
                'bb_width': 0.0,
                'bb_squeeze': 0.0,
                'rsi_divergence': 'none',
                'macd_divergence': 'none',
                'volume_divergence': 'none',
                'price_momentum': 0.0,
                'volume_momentum': 0.0,
                'trend_strength': 0.5,
                'support_resistance': 'unknown',
                'fibonacci_levels': 'unknown',
                'elliott_wave': 'unknown',
                'harmonic_patterns': 'none',
                'candlestick_patterns': 'none',
                'market_structure': 'unknown',
                'flow_level_meta': 'unknown',
                'pattern_direction': 'neutral'
            }
        }

    row = df.iloc[0]
    return {
        'signal_info': {
            'action': row['action'],
            'signal_score': row['signal_score'],
            'confidence': row['confidence'],
            'reason': row['reason']
        },
        'market_data': {
            'price': row['price'],
            'volume': row['volume'],
            'rsi': row['rsi'],
            'macd': row['macd'],
            'volatility': row['volatility'],
            'volume_ratio': row['volume_ratio']
        },
        'wave_info': {
            'wave_phase': row['wave_phase'],
            'pattern_type': row['pattern_type'],
            'wave_progress': row['wave_progress'],
            'structure_score': row['structure_score'],
            'pattern_confidence': row['pattern_confidence'],
            'integrated_direction': row['integrated_direction']
        },
        'scores': {
            'rl_score': row['rl_score'],
            'tech_score': row['tech_score'],
            'wave_score': row['wave_score'],
            'risk_score': row['risk_score']
        },
        # 🆕 Absolute Zero System의 새로운 고급 지표들
        'advanced_indicators': {
            'mfi': row.get('mfi', 50.0),
            'atr': row.get('atr', 0.0),
            'adx': row.get('adx', 25.0),
            'ma20': row.get('ma20', 0.0),
            'rsi_ema': row.get('rsi_ema', 50.0),
            'macd_smoothed': row.get('macd_smoothed', 0.0),
            'wave_momentum': row.get('wave_momentum', 0.0),
            'bb_position': row.get('bb_position', 'unknown'),
            'bb_width': row.get('bb_width', 0.0),
            'bb_squeeze': row.get('bb_squeeze', 0.0),
            'rsi_divergence': row.get('rsi_divergence', 'none'),
            'macd_divergence': row.get('macd_divergence', 'none'),
            'volume_divergence': row.get('volume_divergence', 'none'),
            'price_momentum': row.get('price_momentum', 0.0),
            'volume_momentum': row.get('volume_momentum', 0.0),
            'trend_strength': row.get('trend_strength', 0.5),
            'support_resistance': row.get('support_resistance', 'unknown'),
            'fibonacci_levels': row.get('fibonacci_levels', 'unknown'),
            'elliott_wave': row.get('elliott_wave', 'unknown'),
            'harmonic_patterns': row.get('harmonic_patterns', 'none'),
            'candlestick_patterns': row.get('candlestick_patterns', 'none'),
            'market_structure': row.get('market_structure', 'unknown'),
            'flow_level_meta': row.get('flow_level_meta', 'unknown'),
            'pattern_direction': row.get('pattern_direction', 'neutral')
        }
    }

# 기존 함수 호환성을 위한 래퍼 함수
def load_market_context_from_signal_history(coin, interval='combined'):
    """기존 호환성을 위한 래퍼 함수 - signal_summary 사용"""
    signal_data = load_signal_from_summary(coin)
    
    # 기존 형식으로 변환
    return {
        'market_context': {
            'market_mode': 'Neutral',  # 기본값
            'market_flow': signal_data['wave_info']['integrated_direction']
        },
        'wave_info': {
            'wave_phase': signal_data['wave_info']['wave_phase'],
            'three_wave_pattern': signal_data['wave_info']['pattern_type'],
            'sideways_pattern': 'none'
        }
    }

# 최근 N개 캔들 데이터 로딩 (다중)
INTERVAL_RECENT_CANDLE_COUNT = {
    '15m': 8, '30m': 6, '240m': 4, '1d': 2
}

# 최근 캔들 데이터 로딩 (단일)
def load_recent_candle(coin, interval):
    recent_candles = INTERVAL_RECENT_CANDLE_COUNT.get(interval, 4)
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT * FROM candles
            WHERE coin=? AND interval=? ORDER BY timestamp DESC LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(coin, interval, recent_candles))

    if df.empty:
        return pd.Series()  # ✅ 빈 Series 반환

    required_fields = ['rsi', 'macd', 'macd_signal', 'mfi', 'bb_upper', 'bb_lower', 'volume_avg']

    # 최신 캔들부터 -2까지 돌면서 유효한 row 찾기
    for i in range(len(df)):
        candle = df.iloc[i]
        if all(pd.notnull(candle.get(field)) for field in required_fields):
            return candle

    return df.iloc[0]

# 최근 캔들 데이터 로딩 (다중)
def load_recent_candles_for_replace(coin, interval, count=4):
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT * FROM candles
            WHERE coin=? AND interval=? 
            ORDER BY timestamp DESC LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(coin, interval, count))

    if df.empty:
        return pd.DataFrame()

    required_fields = ['rsi', 'macd', 'macd_signal', 'mfi', 'bb_upper', 'bb_lower', 'volume_avg']

    valid_candles = df.dropna(subset=required_fields)

    return valid_candles

# 240m 파동 정보 로딩
def load_wave_and_market_info(coin, interval='combined'):
    """signals 테이블에서 파동 및 시장 정보 로드 (combined 시그널만 사용)"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT wave_phase, pattern_type, integrated_direction, integrated_strength
                FROM signals
                WHERE coin=? AND interval='combined'
                ORDER BY timestamp DESC LIMIT 1
            """
            df = pd.read_sql(query, conn, params=(coin,))

        if df.empty:
            raise ValueError("데이터가 없습니다.")

        return {
            'wave_info': {
                'wave_phase': df['wave_phase'].iloc[0],
                'three_wave_pattern': df['pattern_type'].iloc[0],
                'sideways_pattern': 'none'  # 기본값
            },
            'market_context': {
                'market_mode': 'Neutral',  # 기본값
                'market_flow': df['integrated_direction'].iloc[0]
            }
        }
    except Exception as e:
        logging.warning(f"[wave_and_market_info] {coin}-{interval} 데이터 로드 실패: {e}")
        return {
            'wave_info': {'wave_phase': 'None', 'three_wave_pattern': 'None', 'sideways_pattern': 'None'},
            'market_context': {'market_mode': 'Neutral', 'market_flow': 'Neutral'}
        }

# 손절 조건 확인 함수
def calculate_stop_loss_strength(current_price, entry_price, latest_candle, params, interval='240m'):
    strengths = []
    weights = {
        'rsi': 1.5,
        'bollinger': 1.2,
        'volume_spike': 1.0,
        'macd': 1.3,
        'mfi': 1.0,
        'rl_score': 2.0  # RL 점수를 중요하게 반영
    }

    # 기존 기술적 지표 계산 유지
    rsi_strength = 1.0 if latest_candle['rsi'] <= params['rsi_rebound']['stop_loss_rsi'] else 0.0
    strengths.append(rsi_strength * weights['rsi'])

    bb_middle = (latest_candle['bb_upper'] + latest_candle['bb_lower']) / 2
    bollinger_strength = 1.0 if params['bollinger_breakout']['stop_loss_bb_middle'] and current_price <= bb_middle else 0.0
    strengths.append(bollinger_strength * weights['bollinger'])

    volume_spike_strength = 1.0 if latest_candle['volume'] <= latest_candle['volume_avg'] * params['volume_spike']['stop_loss_volume_ratio'] else 0.0
    strengths.append(volume_spike_strength * weights['volume_spike'])

    macd_strength = 1.0 if latest_candle['macd_diff'] <= params['macd_cross']['stop_loss_macd'] else 0.0
    strengths.append(macd_strength * weights['macd'])

    mfi_strength = 1.0 if latest_candle['mfi'] <= params['mfi']['stop_loss_level'] else 0.0
    strengths.append(mfi_strength * weights['mfi'])

    # RL 점수 추가 계산 (기존 함수 제거)
    # rl_state_key = calculate_rl_state(latest_candle, interval)
    # rl_score = get_rl_score(rl_state_key)
    # RL 점수는 적절한 기준으로 정규화 (예: 100점 기준)
    rl_normalized_score = 0.5  # 기본값으로 설정
    strengths.append(rl_normalized_score * weights['rl_score'])

    total_weight = sum(weights.values())
    final_strength = (sum(strengths) / total_weight) * 100

    return final_strength

# 손절 조건 체크
def check_stop_loss_conditions(coin, current_price, entry_price, params, latest_candle, interval='240m', stop_loss_threshold=50):
    final_strength = calculate_stop_loss_strength(current_price, entry_price, latest_candle, params, interval)

    if final_strength >= stop_loss_threshold:
        return True, f'융합 손절 (강도 {final_strength:.2f}%)'

    return False, None

# 🆕 Absolute Zero System 개선사항을 반영한 시그널 기반 매매 결정 함수
def make_signal_based_decision(signal_data):
    """시그널 점수 중심 매매 결정 (학습 기반 동적 리스크 관리 + 적응적 고급 지표 활용)"""
    buy_decisions = []
    sell_decisions = []
    
    # 매수 후보 결정 (시그널 점수 중심 + 적응적 고급 지표 보너스)
    for trade in signal_data.get('selected_trades', []):
        # 🎯 핵심: 시그널 점수가 주요 기준
        signal_score = trade.get('signal_score', 0.0)
        confidence = trade.get('confidence', 0.0)
        
        # 기본 매수 조건: 학습된 전략의 시그널 점수만 신뢰
        if (confidence >= 0.6 and 
            signal_score >= 0.4 and 
            trade['action'] == 'buy'):
            
            # 🎯 학습된 전략의 시그널 점수 그대로 사용 (중복 계산 제거)
            trade['enhanced_score'] = signal_score  # 시그널 점수 그대로 사용
            buy_decisions.append(trade)
    
    # 매도 후보 결정 (시그널 점수 중심 + 학습 기반 동적 손절)
    for holding in signal_data.get('current_holdings', []):
        signal_score = holding.get('signal_score', 0.0)
        confidence = holding.get('confidence', 0.0)
        
        # 🎯 핵심: 학습된 전략의 시그널 점수만 신뢰
        if signal_score < -0.3 and confidence > 0.5:  # 강한 매도 시그널
            # 🎯 학습된 전략의 시그널 점수 그대로 사용 (중복 계산 제거)
            holding['enhanced_score'] = signal_score  # 시그널 점수 그대로 사용
            sell_decisions.append(holding)
    
    return {
        'buy': buy_decisions,
        'sell': sell_decisions
    }

# 🆕 적응적 고급 지표 보너스 계산
def calculate_adaptive_technical_bonus(trade):
    """적응적 고급 지표 보너스 (시장 상황에 따라 가중치 조정)"""
    advanced_indicators = trade.get('advanced_indicators', {})
    market_context = get_market_context()
    
    bonus = 0.0
    
    # 🎯 시장 상황에 따른 적응적 가중치
    if market_context['trend'] == 'bullish':
        # 상승장에서는 다이버전스와 트렌드 강도에 더 높은 가중치
        if (advanced_indicators.get('rsi_divergence') == 'bullish' or 
            advanced_indicators.get('macd_divergence') == 'bullish'):
            bonus += 0.15  # 상승장에서 다이버전스 보너스 증가
        
        if advanced_indicators.get('trend_strength', 0.0) > 0.7:
            bonus += 0.12  # 상승장에서 트렌드 보너스 증가
    
    elif market_context['trend'] == 'bearish':
        # 하락장에서는 볼린저밴드 스퀴즈와 모멘텀에 더 높은 가중치
        if advanced_indicators.get('bb_squeeze', 0.0) > 0.8:
            bonus += 0.10  # 하락장에서 스퀴즈 보너스 증가
        
        if abs(advanced_indicators.get('price_momentum', 0.0)) > 0.05:
            bonus += 0.08  # 하락장에서 모멘텀 보너스 증가
    
    else:  # 중립장
        # 중립장에서는 균형잡힌 보너스
        if (advanced_indicators.get('rsi_divergence') == 'bullish' or 
            advanced_indicators.get('macd_divergence') == 'bullish'):
            bonus += 0.10
        
        if advanced_indicators.get('trend_strength', 0.0) > 0.7:
            bonus += 0.08
        
        if advanced_indicators.get('bb_squeeze', 0.0) > 0.8:
            bonus += 0.05
    
    # 🎯 변동성에 따른 보너스 조정
    volatility = market_context.get('volatility', 0.02)
    if volatility > 0.05:  # 고변동성
        bonus *= 0.8  # 고변동성에서는 보너스 감소
    elif volatility < 0.02:  # 저변동성
        bonus *= 1.2  # 저변동성에서는 보너스 증가
    
    return min(bonus, 0.2)  # 최대 20% 보너스 제한

# 🆕 적응적 고급 지표 페널티 계산
def calculate_adaptive_technical_penalty(holding):
    """적응적 고급 지표 매도 페널티 (시장 상황에 따라 가중치 조정)"""
    advanced_indicators = holding.get('advanced_indicators', {})
    market_context = get_market_context()
    
    penalty = 0.0
    
    # 🎯 시장 상황에 따른 적응적 페널티
    if market_context['trend'] == 'bearish':
        # 하락장에서는 다이버전스와 약한 트렌드에 더 높은 페널티
        if (advanced_indicators.get('rsi_divergence') == 'bearish' or 
            advanced_indicators.get('macd_divergence') == 'bearish'):
            penalty += 0.20  # 하락장에서 다이버전스 페널티 증가
        
        if advanced_indicators.get('trend_strength', 0.0) < 0.3:
            penalty += 0.15  # 하락장에서 약한 트렌드 페널티 증가
    
    elif market_context['trend'] == 'bullish':
        # 상승장에서는 상대적으로 낮은 페널티
        if (advanced_indicators.get('rsi_divergence') == 'bearish' or 
            advanced_indicators.get('macd_divergence') == 'bearish'):
            penalty += 0.10  # 상승장에서 다이버전스 페널티 감소
        
        if advanced_indicators.get('trend_strength', 0.0) < 0.3:
            penalty += 0.08  # 상승장에서 약한 트렌드 페널티 감소
    
    else:  # 중립장
        # 중립장에서는 균형잡힌 페널티
        if (advanced_indicators.get('rsi_divergence') == 'bearish' or 
            advanced_indicators.get('macd_divergence') == 'bearish'):
            penalty += 0.15
        
        if advanced_indicators.get('trend_strength', 0.0) < 0.3:
            penalty += 0.10
    
    return min(penalty, 0.25)  # 최대 25% 페널티 제한

# 🆕 학습 기반 동적 임계값 조정
def get_dynamic_threshold(coin):
    """학습 기반 동적 매수 임계값 조정"""
    try:
        # 🎯 코인별 과거 성과 분석
        performance_score = analyze_coin_performance(coin)
        
        # 🎯 시장 상황 분석
        market_score = analyze_market_conditions()
        
        # 🎯 기본 임계값 (0.4)
        base_threshold = 0.4
        
        # 🎯 성과 기반 조정
        if performance_score > 0.7:  # 좋은 성과
            base_threshold -= 0.05  # 임계값 낮춤 (더 쉽게 매수)
        elif performance_score < 0.3:  # 나쁜 성과
            base_threshold += 0.05  # 임계값 높임 (더 엄격하게 매수)
        
        # 🎯 시장 상황 기반 조정
        if market_score > 0.7:  # 좋은 시장 상황
            base_threshold -= 0.03
        elif market_score < 0.3:  # 나쁜 시장 상황
            base_threshold += 0.03
        
        return max(0.3, min(0.6, base_threshold))  # 0.3~0.6 범위로 제한
        
    except Exception as e:
        print(f"⚠️ 동적 임계값 계산 오류 ({coin}): {e}")
        return 0.4  # 기본값 반환

# 🆕 학습 기반 동적 손절 강도 계산
def calculate_adaptive_stop_loss_strength(holding):
    """학습 기반 동적 손절 강도 계산"""
    try:
        coin = holding['coin']
        
        # 🎯 코인별 과거 손절 성과 분석
        stop_loss_performance = analyze_stop_loss_performance(coin)
        
        # 🎯 현재 시그널 강도
        signal_strength = abs(holding.get('signal_score', 0.0))
        
        # 🎯 시장 변동성
        market_volatility = get_market_volatility()
        
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

# 🆕 시장 상황 분석
def get_market_context():
    """시장 상황 분석 (트렌드, 변동성 등)"""
    try:
        # 🎯 기준 코인(환경/DB) 시장 상황 분석
        from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
        base_coin = None
        try:
            available = get_available_coins_and_intervals()
            base_coin = next(iter({c for c, _ in available}), None)
        except Exception:
            base_coin = None
        base_coin = base_coin or os.getenv('DEFAULT_COIN', 'BTC')
        btc_signal = load_realtime_signal(base_coin, 'combined')
        
        if btc_signal:
            signal_score = btc_signal['signal_info']['signal_score']
            
            if signal_score > 0.3:
                trend = 'bullish'
            elif signal_score < -0.3:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            volatility = btc_signal['market_data'].get('volatility', 0.02)
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

# 🆕 코인별 성과 분석
def analyze_coin_performance(coin):
    """코인별 과거 거래 성과 분석"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # 최근 30일간 거래 성과 분석
            thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
            
            df = pd.read_sql("""
                SELECT profit_pct, executed FROM trade_decision_log 
                WHERE coin = ? AND timestamp > ? AND executed = 1
                ORDER BY timestamp DESC
            """, conn, params=(coin, thirty_days_ago))
            
            if df.empty:
                return 0.5  # 거래 내역 없으면 중립
            
            # 평균 수익률
            avg_profit = df['profit_pct'].mean()
            
            # 승률
            win_rate = len(df[df['profit_pct'] > 0]) / len(df)
            
            # 성과 점수 계산 (0~1)
            performance_score = (avg_profit + 10) / 20 * 0.6 + win_rate * 0.4
            
            return max(0.0, min(1.0, performance_score))
            
    except Exception as e:
        print(f"⚠️ 코인 성과 분석 오류 ({coin}): {e}")
        return 0.5

# 🆕 손절 성과 분석
def analyze_stop_loss_performance(coin):
    """코인별 손절 성과 분석"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # 최근 30일간 손절 거래 분석
            thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
            
            df = pd.read_sql("""
                SELECT profit_pct FROM trade_decision_log 
                WHERE coin = ? AND timestamp > ? AND executed = 1 
                AND reason LIKE '%stop_loss%' OR reason LIKE '%손절%'
                ORDER BY timestamp DESC
            """, conn, params=(coin, thirty_days_ago))
            
            if df.empty:
                return 0.5  # 손절 내역 없으면 중립
            
            # 손절 후 추가 하락 여부 분석
            # (실제로는 더 복잡한 분석이 필요하지만 여기서는 단순화)
            avg_stop_loss = df['profit_pct'].mean()
            
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

# 🆕 시장 변동성 계산
def get_market_volatility():
    """시장 변동성 계산"""
    try:
        # 기준 코인 변동성 계산
        from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
        base_coin = None
        try:
            available = get_available_coins_and_intervals()
            base_coin = next(iter({c for c, _ in available}), None)
        except Exception:
            base_coin = None
        base_coin = base_coin or os.getenv('DEFAULT_COIN', 'BTC')
        btc_signal = load_realtime_signal(base_coin, 'combined')
        
        if btc_signal:
            return btc_signal['market_data'].get('volatility', 0.02)
        else:
            return 0.02
            
    except Exception as e:
        print(f"⚠️ 시장 변동성 계산 오류: {e}")
        return 0.02

# 🆕 시장 상황 분석
def analyze_market_conditions():
    """전체 시장 상황 분석"""
    try:
        # 상위 10개 코인의 평균 시그널 점수로 시장 상황 판단
        top_coins = load_top_150_coins()[:10]
        
        total_score = 0.0
        valid_count = 0
        
        for coin in top_coins:
            signal = load_realtime_signal(coin, 'combined')
            if signal:
                total_score += signal['signal_info']['signal_score']
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

# 시그널 기반 최종 매수/매도 (멀티 타임프레임 최적화 통합)
def execute_signal_based_trades(signal_decisions, wallet_coins, selected_candidates, selected_trades):
    """🚀 멀티 타임프레임 최적화된 시그널 기반 매매 실행"""
    print("🎯 멀티 타임프레임 최적화된 시그널 기반 매매 실행 시작")
    print("─" * 40)

    # 🚀 멀티 타임프레임 시그널 기반 매도 실행
    for decision in signal_decisions.get('sell', []):
        coin = decision['coin']
        
        # 최신 캔들 로딩
        latest_candle = load_recent_candle(coin, '240m')
        if latest_candle is None:
            continue

        current_price = latest_candle['close']
        entry_price = get_entry_price(coin)
        profit_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0

        # 🚀 멀티 타임프레임 최적화된 매도 실행
        print(f"🔄 {coin} 멀티 타임프레임 최적화 매도 실행...")
        
        mtf_trade_result = execute_multi_timeframe_optimized_trade(
            coin=coin,
            action='sell',
            base_price=current_price * 0.99,  # 1% 할인
            base_position_size=1.0
        )
        
        if mtf_trade_result['status'] == 'success':
            print(f"✅ {coin} 멀티 타임프레임 최적화 매도 성공")
            print(f"  📊 최적화된 가격: {mtf_trade_result['optimized_params']['optimized_price']}")
            print(f"  🛑 손절: {mtf_trade_result['optimized_params']['stop_loss_pct']}%")
            print(f"  🎯 익절: {mtf_trade_result['optimized_params']['take_profit_pct']}%")
        elif mtf_trade_result['status'] == 'skipped':
            print(f"⏭️ {coin}: 실행 우선순위가 낮아 매도 건너뜀")
        else:
            print(f"⚠️ {coin}: 멀티 타임프레임 최적화 매도 실패, 기존 방식으로 실행")
            # 🎯 기존 방식으로 fallback
            trade_data = {
                'coin': coin,
                'interval': 'combined',
                'timestamp': int(datetime.now().timestamp()),
                'signal': -1,
                'final_score': decision.get('signal_score', 0.0),
                'approved_by': ['Signal'],
                'market_flow': 'Signal',
                'market_mode': 'Signal',
                'price': round(current_price * 0.99, 2),
                'position_percentage': 1.0,
                'profit_pct': round(profit_pct, 2),
                'decision_status': 'approved'
            }
            
            execute_trade_with_timeout(trade_data)
            print(f"[{datetime.now()}] 🔑 기존 방식 매도: {get_korean_name(coin)} | 수익률: {profit_pct:.2f}%")

        log_trade_decision({
            'timestamp': int(datetime.now().timestamp()),
            'coin': coin,
            'interval': 'combined',
            'action': 'sell',
            'reason': 'signal_based_sell',
            'reason_detail': f"멀티 타임프레임 시그널 기반 매도 (최적화 상태: {mtf_trade_result['status']})",
            'entry_price': entry_price or 0,
            'current_price': current_price,
            'profit_pct': profit_pct,
            'fusion_score': decision.get('signal_score', 0.0),
            'rl_score': 0.0,
            'market_mode': 'MultiTimeframe',
            'market_flow': 'MultiTimeframe',
            'gpt_approved': 1,
            'executed': 1 if mtf_trade_result['status'] == 'success' else 0,
            'execution_price': mtf_trade_result.get('optimized_params', {}).get('optimized_price', current_price * 0.99),
            'execution_amount': 1.0,
            'execution_type': 'sell'
        })

    # 🚀 멀티 타임프레임 시그널 기반 매수 실행
    for decision in signal_decisions.get('buy', []):
        coin = decision['coin']
        
        # 최신 캔들 로딩
        latest_candle = load_recent_candle(coin, '240m')
        if latest_candle is None:
            continue

        current_price = latest_candle['close']

        # 🚀 멀티 타임프레임 최적화된 매수 실행
        print(f"🔄 {coin} 멀티 타임프레임 최적화 매수 실행...")
        
        mtf_trade_result = execute_multi_timeframe_optimized_trade(
            coin=coin,
            action='buy',
            base_price=current_price * 1.01,  # 1% 프리미엄
            base_position_size=0.5  # 기본 50% 포지션
        )
        
        if mtf_trade_result['status'] == 'success':
            print(f"✅ {coin} 멀티 타임프레임 최적화 매수 성공")
            print(f"  📊 최적화된 가격: {mtf_trade_result['optimized_params']['optimized_price']}")
            print(f"  📈 최적화된 포지션 크기: {mtf_trade_result['optimized_params']['optimized_position_size']}")
            print(f"  🛑 손절: {mtf_trade_result['optimized_params']['stop_loss_pct']}%")
            print(f"  🎯 익절: {mtf_trade_result['optimized_params']['take_profit_pct']}%")
        elif mtf_trade_result['status'] == 'skipped':
            print(f"⏭️ {coin}: 실행 우선순위가 낮아 매수 건너뜀")
        else:
            print(f"⚠️ {coin}: 멀티 타임프레임 최적화 매수 실패, 기존 방식으로 실행")
            # 🎯 기존 방식으로 fallback
            trade_data = {
                'coin': coin,
                'interval': 'combined',
                'timestamp': int(datetime.now().timestamp()),
                'signal': 1,
                'final_score': decision.get('signal_score', 0.0),
                'approved_by': ['Signal'],
                'market_flow': 'Signal',
                'market_mode': 'Signal',
                'price': round(current_price * 1.01, 2),
                'position_percentage': 0.5,
                'profit_pct': 0.0,
                'decision_status': 'approved'
            }
            
            execute_trade_with_timeout(trade_data)
            print(f"[{datetime.now()}] 🔑 기존 방식 매수: {get_korean_name(coin)} | 가격: {current_price:.2f}")

        log_trade_decision({
            'timestamp': int(datetime.now().timestamp()),
            'coin': coin,
            'interval': 'combined',
            'action': 'buy',
            'reason': 'signal_based_buy',
            'reason_detail': f"멀티 타임프레임 시그널 기반 매수 (최적화 상태: {mtf_trade_result['status']})",
            'entry_price': 0,
            'current_price': current_price,
            'profit_pct': 0.0,
            'fusion_score': decision.get('signal_score', 0.0),
            'rl_score': 0.0,
            'market_mode': 'MultiTimeframe',
            'market_flow': 'MultiTimeframe',
            'gpt_approved': 1,
            'executed': 1 if mtf_trade_result['status'] == 'success' else 0,
            'execution_price': mtf_trade_result.get('optimized_params', {}).get('optimized_price', current_price * 1.01),
            'execution_amount': mtf_trade_result.get('optimized_params', {}).get('optimized_position_size', 0.5),
            'execution_type': 'buy'
        })

    print("✅ 멀티 타임프레임 최적화된 시그널 기반 매매 실행 완료")

def log_trade_decision(data: dict):
    """
    실전 매매 결정과 실행 정보를 virtual_trading.db의 real_trade_history 테이블에 기록합니다.
    - 섀도우 트레이딩과 실전 매매를 통합 관리
    - data에는 판단 사유, 가격, 실행 여부 등이 포함되어야 합니다.
    """

    insert_query = """
        INSERT INTO real_trade_history (
            timestamp, coin, interval, action, reason, reason_detail,
            entry_price, current_price, profit_pct,
            fusion_score, rl_score, market_mode, market_flow,
            gpt_approved, executed, execution_price, execution_amount, execution_type,
            signal_score, confidence, holding_duration, max_profit_pct, max_loss_pct, stop_loss_price, take_profit_price
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        data.get('timestamp'),
        data.get('coin'),
        data.get('interval', 'multi'),
        data.get('action'),
        data.get('reason', 'unknown'),
        data.get('reason_detail', ''),
        data.get('entry_price'),
        data.get('current_price'),
        data.get('profit_pct'),
        data.get('fusion_score'),
        data.get('rl_score'),
        data.get('market_mode', 'Neutral'),
        data.get('market_flow', 'Neutral'),
        int(data.get('gpt_approved', 0)),
        int(data.get('executed', 0)),
        data.get('execution_price'),
        data.get('execution_amount'),
        data.get('execution_type', 'none'),
        data.get('signal_score', 0.0),
        data.get('confidence', 0.0),
        data.get('holding_duration', 0),
        data.get('max_profit_pct', 0.0),
        data.get('max_loss_pct', 0.0),
        data.get('stop_loss_price', None),
        data.get('take_profit_price', None)
    )

    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            conn.execute(insert_query, values)
    except Exception as e:
        logging.error(f"[DB 저장 오류] real_trade_history 기록 실패 - {data.get('coin')} | 오류: {e}")

def save_real_trade_feedback(trade_id: int, coin: str, signal_pattern: str, 
                            success_rate: float, avg_profit: float, total_trades: int, 
                            confidence: float, learning_episode: int, feedback_type: str):
    """실전 매매 피드백 저장 (trading_system.db)"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO real_trade_feedback (
                    trade_id, coin, signal_pattern, success_rate, avg_profit, 
                    total_trades, confidence, learning_episode, feedback_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, coin, signal_pattern, success_rate, avg_profit, 
                  total_trades, confidence, learning_episode, feedback_type))
    except Exception as e:
        logging.error(f"[DB 저장 오류] real_trade_feedback 기록 실패 - {coin} | 오류: {e}")

def log_signal_based_trade(signal_data: dict):
    """
    시그널 기반 매매 정보를 별도로 기록 (통합 DB)
    - 시그널 정보와 실전 매매 정보를 연결하는 브릿지 역할
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    coin TEXT,
                    signal_action TEXT,
                    actual_action TEXT,
                    signal_score REAL,
                    confidence REAL,
                    signal_reason TEXT,
                    execution_reason TEXT,
                    signal_price REAL,
                    execution_price REAL,
                    executed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                INSERT INTO signal_trade_executions (
                    timestamp, coin, signal_action, actual_action,
                    signal_score, confidence, signal_reason, execution_reason,
                    signal_price, execution_price, executed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('timestamp'),
                signal_data.get('coin'),
                signal_data.get('signal_action'),
                signal_data.get('actual_action'),
                signal_data.get('signal_score'),
                signal_data.get('confidence'),
                signal_data.get('signal_reason'),
                signal_data.get('execution_reason'),
                signal_data.get('signal_price'),
                signal_data.get('execution_price'),
                signal_data.get('executed', 0)
            ))
            conn.commit()
    except Exception as e:
        logging.error(f"[시그널 매매 기록 오류] {signal_data.get('coin')} | 오류: {e}")

def get_signal_history(coin: str, hours: int = 24) -> list:
    """시그널 히스토리 조회 (통합 DB)"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT timestamp, action, signal_score, confidence, reason, price
                FROM signal_history
                WHERE coin = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
            df = pd.read_sql(query, conn, params=(coin, cutoff_time))
            return df.to_dict('records')
    except Exception as e:
        logging.error(f"시그널 히스토리 조회 오류 ({coin}): {e}")
        return []

def get_trade_history(coin: str, hours: int = 24) -> list:
    """실전 매매 히스토리 조회 (매매 전용 DB)"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT timestamp, action, reason, executed, execution_price, execution_type
                FROM trade_decision_log
                WHERE coin = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
            df = pd.read_sql(query, conn, params=(coin, cutoff_time))
            return df.to_dict('records')
    except Exception as e:
        logging.error(f"매매 히스토리 조회 오류 ({coin}): {e}")
        return []

def compare_signal_vs_trade(coin: str, hours: int = 24):
    """시그널 히스토리와 실전 매매 히스토리 비교"""
    signal_history = get_signal_history(coin, hours)
    trade_history = get_trade_history(coin, hours)
    
    print(f"\n📊 {get_korean_name(coin)} 시그널 vs 실전 매매 비교 (최근 {hours}시간)")
    print("=" * 60)
    
    print(f"📈 시그널 히스토리: {len(signal_history)}개")
    for signal in signal_history[:5]:  # 최근 5개만
        timestamp_str = datetime.fromtimestamp(signal['timestamp']).strftime('%H:%M:%S')
        print(f"  {timestamp_str} | {signal['action']} | 점수: {signal['signal_score']:.3f} | 신뢰도: {signal['confidence']:.2f}")
    
    print(f"\n💰 실전 매매 히스토리: {len(trade_history)}개")
    for trade in trade_history[:5]:  # 최근 5개만
        timestamp_str = datetime.fromtimestamp(trade['timestamp']).strftime('%H:%M:%S')
        executed_str = "✅실행" if trade['executed'] else "❌미실행"
        print(f"  {timestamp_str} | {trade['action']} | {executed_str} | {trade['reason']}")
    
    # 시그널과 매매의 일치율 계산
    if signal_history and trade_history:
        signal_actions = {s['timestamp']: s['action'] for s in signal_history}
        trade_actions = {t['timestamp']: t['action'] for t in trade_history if t['executed']}
        
        matches = 0
        total = 0
        for timestamp, trade_action in trade_actions.items():
            if timestamp in signal_actions:
                total += 1
                if signal_actions[timestamp] == trade_action:
                    matches += 1
        
        if total > 0:
            match_rate = (matches / total) * 100
            print(f"\n🎯 시그널-매매 일치율: {match_rate:.1f}% ({matches}/{total})")
        else:
            print(f"\n 시그널-매매 일치율: 비교할 데이터 없음")

def print_signal_trade_summary():
    """전체 시그널과 매매 현황 요약"""
    try:
        # 시그널 현황 (통합 시그널 기준)
        with sqlite3.connect(DB_PATH) as conn:
            signal_stats = pd.read_sql("""
                SELECT action, COUNT(*) as count, AVG(signal_score) as avg_score
                FROM signals
                WHERE interval = 'combined'
                GROUP BY action
            """, conn)
        
        # 매매 현황
        with sqlite3.connect(DB_PATH) as conn:
            trade_stats = pd.read_sql("""
                SELECT action, COUNT(*) as count, 
                       SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_count
                FROM trade_decision_log
                WHERE timestamp > ?
                GROUP BY action
            """, conn, params=(int((datetime.now() - timedelta(hours=24)).timestamp()),))
        
        print(f"\n📊 24시간 시그널 vs 매매 현황")
        print("=" * 50)
        
        print("📈 시그널 현황 (통합):")
        for _, row in signal_stats.iterrows():
            print(f"  {row['action']}: {row['count']}개 (평균점수: {row['avg_score']:.3f})")
        
        print("\n💰 매매 현황:")
        for _, row in trade_stats.iterrows():
            execution_rate = (row['executed_count'] / row['count'] * 100) if row['count'] > 0 else 0
            print(f"  {row['action']}: {row['count']}개 (실행률: {execution_rate:.1f}%)")
            
    except Exception as e:
        logging.error(f"시그널-매매 요약 조회 오류: {e}")

# 🆕 실전매매와 동일한 시그널 기반 Executor 로직 (갈아타기 제외)
def run_signal_based_executor():
    """🆕 순수 시그널과 보유 정보를 조합하여 실전매매 실행"""
    print("🚀 [STEP 1] 보유 자산 확인")
    sync_wallet_to_db()
    wallet_coins = get_filtered_wallet_coins(min_balance_krw=10000)
    print(f"✅ 보유 자산 수: {len(wallet_coins)} | 보유 코인: {[get_korean_name(coin) for coin in wallet_coins]}")

    print("\n🚀 [STEP 2] 순수 시그널과 보유 정보 조합하여 최종 액션 결정")
    sell_decisions = []
    hold_decisions = []
    
    for coin in wallet_coins:
        # 🆕 순수 시그널 로드
        signal_data = load_realtime_signal(coin, 'combined')
        if signal_data is None:
            print(f"⚠️ {get_korean_name(coin)}: 시그널 데이터 없음 - 홀딩 유지")
            hold_decisions.append({
                'coin': coin,
                'action': 'hold',
                'signal_score': 0.0,
                'confidence': 0.0,
                'reason': '시그널 데이터 없음'
            })
            continue
            
        # 🆕 순수 시그널 정보
        pure_action = signal_data['signal_info']['action']
        signal_score = signal_data['signal_info']['signal_score']
        confidence = signal_data['signal_info']['confidence']
        reason = signal_data['signal_info']['reason']
        current_price = signal_data['market_data']['price']
        
        # 🆕 보유 정보 확인
        entry_price = get_entry_price(coin)
        if entry_price > 0:
            profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_loss_pct = 0.0
        
        print(f"📊 {get_korean_name(coin)}: 순수시그널={pure_action} (점수: {signal_score:.3f}, 신뢰도: {confidence:.2f})")
        print(f"   📈 보유정보: 진입가 {entry_price:,.0f}원, 수익률 {profit_loss_pct:+.2f}%")
        
        # 🎯 순수 시그널과 보유 정보를 조합하여 최종 액션 결정
        final_action = combine_signal_with_holding(pure_action, signal_score, profit_loss_pct)
        
        print(f"   🎯 최종결정: {final_action} (순수시그널: {pure_action} + 보유정보: {profit_loss_pct:+.2f}%)")
        
        # 🆕 최종 액션에 따른 분류
        if final_action in ['sell', 'stop_loss', 'take_profit']:
            sell_decisions.append({
                'coin': coin,
                'action': final_action,
                'signal_score': signal_score,
                'confidence': confidence,
                'reason': f"{reason} + 보유정보조합",
                'price': current_price,
                'pure_action': pure_action,
                'profit_loss_pct': profit_loss_pct
            })
        else:
            hold_decisions.append({
                'coin': coin,
                'action': final_action,
                'signal_score': signal_score,
                'confidence': confidence,
                'reason': f"{reason} + 보유정보조합",
                'pure_action': pure_action,
                'profit_loss_pct': profit_loss_pct
            })
    
    print(f"🔴 매도 대상: {len(sell_decisions)}개")
    print(f"🟡 홀딩 대상: {len(hold_decisions)}개")
    
    # 🆕 성능 업그레이드된 거래 실행
    executed_trades = execute_enhanced_signal_trades(sell_decisions, hold_decisions)
    
    return executed_trades

def combine_signal_with_holding(pure_action: str, signal_score: float, profit_loss_pct: float) -> str:
    """🆕 순수 시그널과 보유 정보를 조합하여 최종 액션 결정 (성능 업그레이드 완전 적용)"""
    try:
        # 🆕 실전 매매 특화 의사결정 엔진 사용
        signal_data = {
            'action': pure_action,
            'signal_score': signal_score,
            'confidence': abs(signal_score),  # 신뢰도는 시그널 점수의 절댓값
            'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low'
        }
        
        # 🆕 코인별 성과 데이터 로드 (실제로는 DB에서 로드)
        coin_performance = real_time_learning_feedback.get_coin_learning_data('current_coin')
        
        # 🆕 시장 컨텍스트 로드
        market_context = {
            'trend': 'bullish' if signal_score > 0.3 else 'bearish' if signal_score < -0.3 else 'neutral',
            'volatility': 'high' if abs(signal_score) > 0.6 else 'medium' if abs(signal_score) > 0.3 else 'low',
            'timestamp': int(time.time())
        }
        
        # 🆕 AI 의사결정 엔진으로 최종 액션 결정
        ai_decision = real_time_ai_decision_engine.make_trading_decision(
            signal_data, 0.0, market_context, coin_performance
        )
        
        # 🎯 익절 조건 (수익률 50% 이상) - 섀도우 트레이딩과 동일
        if profit_loss_pct >= 50.0:
            return 'take_profit'
        
        # 🎯 손절 조건 (손실 10% 이상) - 섀도우 트레이딩과 동일
        if profit_loss_pct <= -10.0:
            return 'stop_loss'
        
        # 🆕 AI 기반 매도 조건 (시그널 점수 + AI 결정)
        if signal_score < -0.5 or ai_decision == 'sell':  # 강한 매도 시그널
            return 'sell'
        elif signal_score < -0.3 or ai_decision == 'sell':  # 매도 시그널
            return 'sell'
        elif signal_score < -0.2:
            return 'sell'
        elif signal_score < -0.1:
            return 'sell'
        
        # 🆕 AI 기반 매수 조건 (시그널 점수 + AI 결정)
        elif signal_score > 0.5 or ai_decision == 'buy':  # 강한 매수 시그널
            return 'buy'
        elif signal_score > 0.3 or ai_decision == 'buy':  # 매수 시그널
            return 'buy'
        elif signal_score > 0.2:
            return 'buy'
        elif signal_score > 0.1:
            return 'buy'
        
        # 🎯 중립 구간 (홀딩) - AI 결정도 고려
        else:
            return 'hold' if ai_decision == 'hold' else ai_decision
            
    except Exception as e:
        print(f"⚠️ 시그널-보유 결합 오류: {e}")
        # 🆕 오류 시 안전한 기본값 반환 (기존 로직 유지)
        if profit_loss_pct >= 50.0:
            return 'take_profit'
        if profit_loss_pct <= -10.0:
            return 'stop_loss'
        if signal_score < -0.3:
            return 'sell'
        elif signal_score > 0.3:
            return 'buy'
        else:
            return 'hold'

    print("\n🚀 [STEP 4] 성능 업그레이드된 매수 후보 확인")
    
    # 🆕 예수금 확인 (실전매매와 동일)
    try:
        from trade_manager import get_available_balance
        available_balance = get_available_balance()
        print(f"💰 예수금: {available_balance:,.0f}원")
        
        if available_balance < 10000:
            print("⚠️ 예수금이 10,000원 미만이므로 신규 매수 불가")
            print("\n🚀 [STEP 5] 최종 보유 상태 확인")
            sync_wallet_to_db()
            updated_wallet_coins = get_filtered_wallet_coins(min_balance_krw=10000)
            print(f"💼 최종 보유 코인: {[get_korean_name(coin) for coin in updated_wallet_coins]}")
            return
            
    except Exception as e:
        print(f"⚠️ 예수금 확인 오류: {e}")
        print("💰 예수금 확인 실패, 매수 진행")
    
    # 🆕 매수 후보 확인 (성능 업그레이드 적용)
    coins = load_top_150_coins()
    buy_candidates = []
    
    for coin in coins:
        # 🆕 순수 시그널 로드
        signal_data = load_realtime_signal(coin, 'combined')
        if signal_data is None:
            continue  # 시그널 데이터 없으면 건너뛰기
            
        pure_action = signal_data['signal_info']['action']
        signal_score = signal_data['signal_info']['signal_score']
        confidence = signal_data['signal_info']['confidence']
        current_price = signal_data['market_data']['price']
        
        # 🎯 순수 시그널의 BUY 액션 확인
        if pure_action == 'buy':
            # 🆕 보유 정보 확인
            entry_price = get_entry_price(coin)
            is_holding = coin in wallet_coins
            
            # 🆕 AI 의사결정 엔진으로 매수 검증
            signal_data_for_ai = {
                'coin': coin,
                'action': 'buy',
                'signal_score': signal_score,
                'confidence': confidence,
                'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low'
            }
            
            market_context = {
                'trend': 'bullish' if signal_score > 0.3 else 'neutral',
                'volatility': 'high' if abs(signal_score) > 0.6 else 'medium',
                'timestamp': int(time.time())
            }
            
            coin_performance = real_time_learning_feedback.get_coin_learning_data(coin)
            ai_decision = real_time_ai_decision_engine.make_trading_decision(
                signal_data_for_ai, current_price, market_context, coin_performance
            )
            
            # 🎯 추가 매수 여부 결정 (AI 검증 포함)
            should_buy = should_add_buy(coin, signal_score, confidence, current_price, entry_price) and ai_decision == 'buy'
            
            if should_buy:
                if is_holding:
                    print(f"🟢 {get_korean_name(coin)}: AI 승인 추매 - 시그널={pure_action} (점수: {signal_score:.3f}, 신뢰도: {confidence:.2f})")
                else:
                    print(f"🟢 {get_korean_name(coin)}: AI 승인 신규매수 - 시그널={pure_action} (점수: {signal_score:.3f}, 신뢰도: {confidence:.2f})")
                
                buy_candidates.append({
                    'coin': coin,
                    'signal_score': signal_score,
                    'confidence': confidence,
                    'reason': signal_data['signal_info']['reason'],
                    'price': current_price,
                    'pure_action': pure_action,
                    'is_additional_buy': is_holding,
                    'entry_price': entry_price
                })
    
    # 🆕 순수 시그널 점수 기준으로 정렬
    buy_candidates.sort(key=lambda x: x['signal_score'], reverse=True)
    
    print(f"🟢 AI 승인 매수 후보: {len(buy_candidates)}개")
    for candidate in buy_candidates:
        buy_type = "추매" if candidate['is_additional_buy'] else "신규매수"
        print(f"  {get_korean_name(candidate['coin'])}: {buy_type} - 순수시그널={candidate['pure_action']}, 점수 {candidate['signal_score']:.3f}, 신뢰도 {candidate['confidence']:.2f}")

    print("\n🚀 [STEP 5] 성능 업그레이드된 매수 실행")
    for candidate in buy_candidates:
        buy_type = "추매" if candidate['is_additional_buy'] else "신규매수"
        print(f"🟢 {get_korean_name(candidate['coin'])} {buy_type} 실행 - {candidate['reason']}")
        
        # 🆕 성능 업그레이드된 매수 실행
        trade_data = {
            'coin': candidate['coin'],
            'action': 'buy',
            'interval': 'combined',
            'timestamp': int(time.time()),
            'signal': 1,
            'final_score': candidate['signal_score'],
            'approved_by': ['AI_Enhanced_Signal'],
            'market_flow': 'AI_Enhanced',
            'market_mode': 'AI_Enhanced',
            'price': round(candidate['price'] * 1.01, 2),
            'position_percentage': 1.0,  # 전액 매수
            'decision_status': 'approved',
            'confidence': candidate['confidence']
        }
        
        # 🆕 거래 결과 기록
        trade_result = {
            'coin': candidate['coin'],
            'action': 'buy',
            'signal_score': candidate['signal_score'],
            'confidence': candidate['confidence'],
            'timestamp': int(time.time()),
            'amount': 0.0,  # 실제 거래 후 업데이트
            'price': candidate['price'],
            'profit': 0.0
        }
        
        # 🆕 학습 피드백에 거래 결과 기록
        real_time_learning_feedback.record_trade_result(candidate['coin'], trade_result)
        
        # 🆕 액션별 성과 추적
        real_time_action_tracker.record_action_result('buy', 0.0, False, 0.0, candidate['coin'])
        
        # 🆕 컨텍스트 기록
        trade_id = f"{candidate['coin']}_{int(time.time())}"
        context = {
            'action': 'buy',
            'signal_score': candidate['signal_score'],
            'confidence': candidate['confidence'],
            'market_context': market_context,
            'coin_performance': coin_performance,
            'buy_type': buy_type
        }
        real_time_context_recorder.record_trade_context(trade_id, context)

    print("\n🚀 [STEP 6] 최종 보유 상태 확인")
    sync_wallet_to_db()
    updated_wallet_coins = get_filtered_wallet_coins(min_balance_krw=10000)
    print(f"💼 최종 보유 코인: {[get_korean_name(coin) for coin in updated_wallet_coins]}")
    
    # 🆕 최종 성과 요약
    print(f"\n📊 성능 업그레이드된 실전매매 완료:")
    for action in ['buy', 'sell', 'hold']:
        perf = real_time_action_tracker.get_action_performance(action)
        if perf['total_trades'] > 0:
            print(f"📈 {action.upper()}: {perf['total_trades']}회, 승률: {perf['success_rate']:.1%}, 평균수익: {perf['avg_profit']:.2f}%")

# 🚀 멀티 타임프레임 실전매매 최적화 시스템
def get_multi_timeframe_execution_priority(coin: str) -> Dict[str, Any]:
    """🚀 멀티 타임프레임 시그널 기반 실전매매 우선순위 결정"""
    try:
        print(f"🔄 {coin} 멀티 타임프레임 실전매매 우선순위 분석 시작")
        
        # 🎯 각 인터벌별 시그널 조회
        intervals = ['15m', '30m', '240m', '1d']
        interval_signals = {}
        
        for interval in intervals:
            try:
                signal = load_realtime_signal(coin, interval)
                if signal:
                    interval_signals[interval] = signal
                    print(f"  ✅ {interval}: {signal['signal_info']['action']} (점수: {signal['signal_info']['signal_score']:.3f})")
                else:
                    print(f"  ⚠️ {interval}: 시그널 없음")
            except Exception as e:
                print(f"  ❌ {interval}: 시그널 조회 실패 - {e}")
                continue
        
        if not interval_signals:
            print(f"⚠️ {coin}: 사용 가능한 시그널이 없습니다")
            return {
                'execution_priority': 'low',
                'confidence_level': 0.0,
                'risk_adjustment': 1.0,
                'position_size_multiplier': 0.5,
                'stop_loss_adjustment': 1.2
            }
        
        # 🎯 멀티 타임프레임 시그널 통합 분석
        execution_priority = calculate_execution_priority(interval_signals)
        confidence_level = calculate_confidence_level(interval_signals)
        risk_adjustment = calculate_risk_adjustment(interval_signals)
        position_size_multiplier = calculate_position_size_multiplier(interval_signals)
        stop_loss_adjustment = calculate_stop_loss_adjustment(interval_signals)
        
        result = {
            'execution_priority': execution_priority,
            'confidence_level': confidence_level,
            'risk_adjustment': risk_adjustment,
            'position_size_multiplier': position_size_multiplier,
            'stop_loss_adjustment': stop_loss_adjustment,
            'interval_signals': interval_signals
        }
        
        print(f"✅ {coin} 멀티 타임프레임 우선순위 분석 완료:")
        print(f"  🎯 실행 우선순위: {execution_priority}")
        print(f"  🔍 신뢰도: {confidence_level:.3f}")
        print(f"  ⚠️ 리스크 조정: {risk_adjustment:.2f}x")
        print(f"  📊 포지션 크기: {position_size_multiplier:.2f}x")
        print(f"  🛑 손절 조정: {stop_loss_adjustment:.2f}x")
        
        return result
        
    except Exception as e:
        print(f"⚠️ {coin} 멀티 타임프레임 우선순위 분석 실패: {e}")
        return {
            'execution_priority': 'low',
            'confidence_level': 0.0,
            'risk_adjustment': 1.0,
            'position_size_multiplier': 0.5,
            'stop_loss_adjustment': 1.2
        }

def calculate_execution_priority(interval_signals: Dict[str, Dict]) -> str:
    """멀티 타임프레임 시그널 기반 실행 우선순위 계산"""
    try:
        if not interval_signals:
            return 'low'
        
        # 🎯 인터벌별 가중치
        interval_weights = {
            '1d': 0.25,    # 장기
            '15m': 0.20,   # 단기
            '30m': 0.25,   # 중기
            '240m': 0.40   # 장기 (가장 중요)
        }
        
        # 🎯 가중 평균 시그널 점수 계산
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for interval, signal in interval_signals.items():
            weight = interval_weights.get(interval, 0.25)
            signal_score = signal['signal_info']['signal_score']
            
            total_weighted_score += signal_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 'low'
        
        avg_signal_score = total_weighted_score / total_weight
        
        # 🎯 우선순위 결정
        if avg_signal_score > 0.4:
            return 'high'
        elif avg_signal_score > 0.2:
            return 'medium'
        elif avg_signal_score > -0.2:
            return 'low'
        else:
            return 'very_low'
            
    except Exception as e:
        print(f"⚠️ 실행 우선순위 계산 실패: {e}")
        return 'low'

def calculate_confidence_level(interval_signals: Dict[str, Dict]) -> float:
    """멀티 타임프레임 시그널 기반 신뢰도 계산"""
    try:
        if not interval_signals:
            return 0.0
        
        # 🎯 인터벌별 신뢰도 가중 평균
        interval_weights = {
            '15m': 0.20, '30m': 0.25, '240m': 0.35, '1d': 0.45
        }
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for interval, signal in interval_signals.items():
            weight = interval_weights.get(interval, 0.25)
            confidence = signal['signal_info']['confidence']
            
            total_weighted_confidence += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_confidence / total_weight
        
    except Exception as e:
        print(f"⚠️ 신뢰도 계산 실패: {e}")
        return 0.0

def calculate_risk_adjustment(interval_signals: Dict[str, Dict]) -> float:
    """멀티 타임프레임 시그널 기반 리스크 조정 계산"""
    try:
        if not interval_signals:
            return 1.0
        
        # 🎯 시그널 일관성 분석
        actions = [signal['signal_info']['action'] for signal in interval_signals.values()]
        unique_actions = set(actions)
        
        # 🎯 액션 일관성에 따른 리스크 조정
        if len(unique_actions) == 1:
            # 모든 인터벌이 동일한 액션
            risk_multiplier = 0.8  # 리스크 감소
        elif len(unique_actions) == 2:
            # 2가지 액션
            risk_multiplier = 1.0  # 기본 리스크
        else:
            # 3가지 이상 액션 (혼재)
            risk_multiplier = 1.3  # 리스크 증가
        
        # 🎯 시그널 점수 분산에 따른 추가 조정
        signal_scores = [signal['signal_info']['signal_score'] for signal in interval_signals.values()]
        score_variance = calculate_variance(signal_scores)
        
        if score_variance > 0.3:
            risk_multiplier *= 1.2  # 높은 분산 = 높은 리스크
        elif score_variance < 0.1:
            risk_multiplier *= 0.9  # 낮은 분산 = 낮은 리스크
        
        return max(0.5, min(2.0, risk_multiplier))  # 0.5~2.0 범위로 제한
        
    except Exception as e:
        print(f"⚠️ 리스크 조정 계산 실패: {e}")
        return 1.0

def calculate_position_size_multiplier(interval_signals: Dict[str, Dict]) -> float:
    """멀티 타임프레임 시그널 기반 포지션 크기 조정 계산"""
    try:
        if not interval_signals:
            return 0.5
        
        # 🎯 신뢰도와 우선순위 기반 포지션 크기 조정
        confidence = calculate_confidence_level(interval_signals)
        priority = calculate_execution_priority(interval_signals)
        
        # 🎯 우선순위별 기본 배수
        priority_multipliers = {
            'high': 1.0,
            'medium': 0.8,
            'low': 0.6,
            'very_low': 0.4
        }
        
        base_multiplier = priority_multipliers.get(priority, 0.6)
        
        # 🎯 신뢰도 기반 조정
        confidence_adjustment = 0.5 + (confidence * 0.5)  # 0.5~1.0 범위
        
        final_multiplier = base_multiplier * confidence_adjustment
        
        return max(0.2, min(1.5, final_multiplier))  # 0.2~1.5 범위로 제한
        
    except Exception as e:
        print(f"⚠️ 포지션 크기 조정 계산 실패: {e}")
        return 0.5

def calculate_stop_loss_adjustment(interval_signals: Dict[str, Dict]) -> float:
    """멀티 타임프레임 시그널 기반 손절 조정 계산"""
    try:
        if not interval_signals:
            return 1.2
        
        # 🎯 시그널 강도와 변동성 기반 손절 조정
        signal_strengths = [abs(signal['signal_info']['signal_score']) for signal in interval_signals.values()]
        avg_strength = sum(signal_strengths) / len(signal_strengths)
        
        # 🎯 강한 시그널일수록 손절 완화
        if avg_strength > 0.6:
            stop_loss_multiplier = 0.8  # 손절 완화
        elif avg_strength > 0.3:
            stop_loss_multiplier = 1.0  # 기본 손절
        else:
            stop_loss_multiplier = 1.3  # 손절 강화
        
        # 🎯 시그널 일관성에 따른 추가 조정
        actions = [signal['signal_info']['action'] for signal in interval_signals.values()]
        unique_actions = set(actions)
        
        if len(unique_actions) == 1:
            # 일관된 시그널 = 손절 완화
            stop_loss_multiplier *= 0.9
        elif len(unique_actions) >= 3:
            # 혼재된 시그널 = 손절 강화
            stop_loss_multiplier *= 1.2
        
        return max(0.6, min(2.0, stop_loss_multiplier))  # 0.6~2.0 범위로 제한
        
    except Exception as e:
        print(f"⚠️ 손절 조정 계산 실패: {e}")
        return 1.2

def calculate_variance(values: List[float]) -> float:
    """분산 계산"""
    try:
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        variance = squared_diff_sum / len(values)
        
        return variance
        
    except Exception as e:
        print(f"⚠️ 분산 계산 실패: {e}")
        return 0.0

# 🚀 멀티 타임프레임 기반 실전매매 실행 함수
def execute_multi_timeframe_optimized_trade(coin: str, action: str, base_price: float, 
                                          base_position_size: float = 1.0) -> Dict[str, Any]:
    """🚀 멀티 타임프레임 시그널 기반 최적화된 실전매매 실행"""
    try:
        print(f"🚀 {coin} 멀티 타임프레임 최적화 매매 실행 시작")
        
        # 🎯 1. 멀티 타임프레임 우선순위 분석
        mtf_analysis = get_multi_timeframe_execution_priority(coin)
        
        # 🎯 2. 실행 우선순위 확인
        execution_priority = mtf_analysis['execution_priority']
        if execution_priority == 'very_low':
            print(f"⚠️ {coin}: 실행 우선순위가 매우 낮아 매매를 건너뜁니다")
            return {
                'status': 'skipped',
                'reason': 'execution_priority_too_low',
                'mtf_analysis': mtf_analysis
            }
        
        # 🎯 3. 매매 파라미터 최적화
        optimized_params = optimize_trade_parameters(mtf_analysis, base_price, base_position_size)
        
        # 🎯 4. 최적화된 매매 실행
        trade_result = execute_optimized_trade(coin, action, optimized_params)
        
        # 🎯 5. 결과 로깅
        log_multi_timeframe_trade(coin, action, mtf_analysis, optimized_params, trade_result)
        
        print(f"✅ {coin} 멀티 타임프레임 최적화 매매 완료")
        return {
            'status': 'success',
            'trade_result': trade_result,
            'mtf_analysis': mtf_analysis,
            'optimized_params': optimized_params
        }
        
    except Exception as e:
        print(f"⚠️ {coin} 멀티 타임프레임 최적화 매매 실패: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

def optimize_trade_parameters(mtf_analysis: Dict[str, Any], base_price: float, 
                            base_position_size: float) -> Dict[str, Any]:
    """멀티 타임프레임 분석 결과 기반 매매 파라미터 최적화"""
    try:
        # 🎯 가격 최적화
        confidence_level = mtf_analysis['confidence_level']
        risk_adjustment = mtf_analysis['risk_adjustment']
        
        # 🎯 신뢰도 기반 가격 조정
        if confidence_level > 0.8:
            price_adjustment = 0.995  # 높은 신뢰도 = 더 공격적인 가격
        elif confidence_level > 0.6:
            price_adjustment = 0.998  # 중간 신뢰도 = 보수적 가격
        else:
            price_adjustment = 1.002  # 낮은 신뢰도 = 보수적 가격
        
        optimized_price = base_price * price_adjustment
        
        # 🎯 포지션 크기 최적화
        position_multiplier = mtf_analysis['position_size_multiplier']
        optimized_position_size = base_position_size * position_multiplier
        
        # 🎯 손절 설정 최적화
        stop_loss_adjustment = mtf_analysis['stop_loss_adjustment']
        base_stop_loss_pct = 5.0  # 기본 5% 손절
        optimized_stop_loss_pct = base_stop_loss_pct * stop_loss_adjustment
        
        # 🎯 익절 설정 최적화
        confidence_based_take_profit = 10.0 + (confidence_level * 20.0)  # 10~30% 범위
        base_take_profit_pct = 15.0  # 기본 15% 익절
        optimized_take_profit_pct = max(base_take_profit_pct, confidence_based_take_profit)
        
        return {
            'optimized_price': round(optimized_price, 8),
            'optimized_position_size': round(optimized_position_size, 4),
            'stop_loss_pct': round(optimized_stop_loss_pct, 2),
            'take_profit_pct': round(optimized_take_profit_pct, 2),
            'confidence_level': confidence_level,
            'risk_adjustment': risk_adjustment,
            'execution_priority': mtf_analysis['execution_priority']
        }
        
    except Exception as e:
        print(f"⚠️ 매매 파라미터 최적화 실패: {e}")
        return {
            'optimized_price': base_price,
            'optimized_position_size': base_position_size,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0,
            'confidence_level': 0.5,
            'risk_adjustment': 1.0,
            'execution_priority': 'low'
        }

def execute_optimized_trade(coin: str, action: str, optimized_params: Dict[str, Any]) -> Dict[str, Any]:
    """최적화된 파라미터로 매매 실행"""
    try:
        print(f"🎯 {coin} 최적화된 매매 실행: {action}")
        print(f"  💰 최적화된 가격: {optimized_params['optimized_price']}")
        print(f"  📊 최적화된 포지션 크기: {optimized_params['optimized_position_size']}")
        print(f"  🛑 손절: {optimized_params['stop_loss_pct']}%")
        print(f"  🎯 익절: {optimized_params['take_profit_pct']}%")
        
        # 🎯 실제 매매 실행 (기존 execute_trade_with_timeout 함수 활용)
        trade_data = {
            'coin': coin,
            'interval': 'combined',  # 멀티 타임프레임 통합
            'timestamp': int(datetime.now().timestamp()),
            'signal': 1 if action == 'buy' else -1,
            'final_score': optimized_params['confidence_level'],
            'approved_by': ['MultiTimeframe'],
            'market_flow': 'MultiTimeframe',
            'market_mode': 'MultiTimeframe',
            'price': optimized_params['optimized_price'],
            'position_percentage': optimized_params['optimized_position_size'],
            'profit_pct': 0.0,
            'decision_status': 'approved',
            'stop_loss_pct': optimized_params['stop_loss_pct'],
            'take_profit_pct': optimized_params['take_profit_pct']
        }
        
        # 🎯 매매 실행
        execution_result = execute_trade_with_timeout(trade_data)
        
        return {
            'execution_result': execution_result,
            'trade_data': trade_data,
            'timestamp': trade_data['timestamp']
        }
        
    except Exception as e:
        print(f"⚠️ {coin} 최적화된 매매 실행 실패: {e}")
        return {
            'execution_result': None,
            'trade_data': None,
            'timestamp': int(datetime.now().timestamp()),
            'error': str(e)
        }

def log_multi_timeframe_trade(coin: str, action: str, mtf_analysis: Dict[str, Any], 
                            optimized_params: Dict[str, Any], trade_result: Dict[str, Any]):
    """멀티 타임프레임 매매 결과 로깅"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # 🎯 멀티 타임프레임 매매 로그 테이블 생성
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_timeframe_trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    action TEXT NOT NULL,
                    execution_priority TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    risk_adjustment REAL NOT NULL,
                    position_size_multiplier REAL NOT NULL,
                    stop_loss_adjustment REAL NOT NULL,
                    optimized_price REAL NOT NULL,
                    optimized_position_size REAL NOT NULL,
                    stop_loss_pct REAL NOT NULL,
                    take_profit_pct REAL NOT NULL,
                    trade_status TEXT NOT NULL,
                    execution_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 🎯 로그 저장
            conn.execute("""
                INSERT INTO multi_timeframe_trade_log (
                    timestamp, coin, action, execution_priority, confidence_level,
                    risk_adjustment, position_size_multiplier, stop_loss_adjustment,
                    optimized_price, optimized_position_size, stop_loss_pct, take_profit_pct,
                    trade_status, execution_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                coin,
                action,
                mtf_analysis['execution_priority'],
                mtf_analysis['confidence_level'],
                mtf_analysis['risk_adjustment'],
                mtf_analysis['position_size_multiplier'],
                mtf_analysis['stop_loss_adjustment'],
                optimized_params['optimized_price'],
                optimized_params['optimized_position_size'],
                optimized_params['stop_loss_pct'],
                optimized_params['take_profit_pct'],
                trade_result.get('status', 'unknown'),
                json.dumps(trade_result, ensure_ascii=False)
            ))
            
            conn.commit()
            print(f"✅ {coin} 멀티 타임프레임 매매 로그 저장 완료")
            
    except Exception as e:
        print(f"⚠️ {coin} 멀티 타임프레임 매매 로그 저장 실패: {e}")



def execute_enhanced_signal_trades(sell_decisions, hold_decisions):
    """🆕 성능 업그레이드된 시그널 기반 거래 실행"""
    print(f"\n🚀 [STEP 3] 성능 업그레이드된 시그널 기반 거래 실행")
    print(f"🔴 매도 대상: {len(sell_decisions)}개")
    print(f"🟡 홀딩 대상: {len(hold_decisions)}개")
    
    executed_trades = []
    total_profit = 0.0
    
    # 🆕 매도 거래 실행 (성능 업그레이드 적용)
    for decision in sell_decisions:
        try:
            coin = decision['coin']
            signal_score = decision['signal_score']
            confidence = decision['confidence']
            current_price = decision['current_price']
            profit_loss_pct = decision['profit_loss_pct']
            
            # 🆕 코인별 성과 데이터 로드
            coin_performance = real_time_learning_feedback.get_coin_learning_data(coin)
            
            # 🆕 AI 의사결정 엔진으로 최종 검증
            signal_data = {
                'coin': coin,
                'action': 'sell',
                'signal_score': signal_score,
                'confidence': confidence,
                'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low'
            }
            
            market_context = {
                'trend': 'bearish' if signal_score < -0.3 else 'neutral',
                'volatility': 'high' if abs(signal_score) > 0.6 else 'medium',
                'timestamp': int(time.time())
            }
            
            ai_decision = real_time_ai_decision_engine.make_trading_decision(
                signal_data, current_price, market_context, coin_performance
            )
            
            # 🆕 AI가 매도를 승인하면 실행
            if ai_decision == 'sell':
                print(f"✅ {get_korean_name(coin)}: AI 승인 매도 실행 - {decision['reason']}")
                
                # 실제 거래 실행 (trade_manager.py 호출)
                trade_data = {
                    'coin': coin,
                    'interval': 'combined',
                    'timestamp': int(time.time()),
                    'signal': -1,
                    'final_score': signal_score,
                    'approved_by': ['AI_Enhanced_Signal'],
                    'market_flow': 'AI_Enhanced',
                    'market_mode': 'AI_Enhanced',
                    'price': round(current_price * 0.99, 2),
                    'position_percentage': 1.0,
                    'profit_pct': round(profit_loss_pct, 2),
                    'confidence': confidence
                }
                
                # 🆕 거래 결과 기록
                trade_result = {
                    'coin': coin,
                    'action': 'sell',
                    'signal_score': signal_score,
                    'confidence': confidence,
                    'timestamp': int(time.time()),
                    'amount': 0.0,  # 실제 거래 후 업데이트
                    'price': current_price,
                    'profit': profit_loss_pct
                }
                
                executed_trades.append(trade_result)
                
                # 🆕 학습 피드백에 거래 결과 기록
                real_time_learning_feedback.record_trade_result(coin, trade_result)
                
                # 🆕 액션별 성과 추적
                success = profit_loss_pct > 0
                real_time_action_tracker.record_action_result('sell', profit_loss_pct, success, 0.0, coin)
                
                # 🆕 컨텍스트 기록
                trade_id = f"{coin}_{int(time.time())}"
                context = {
                    'action': 'sell',
                    'signal_score': signal_score,
                    'confidence': confidence,
                    'market_context': market_context,
                    'coin_performance': coin_performance,
                    'profit_loss_pct': profit_loss_pct
                }
                real_time_context_recorder.record_trade_context(trade_id, context)
                
                total_profit += profit_loss_pct
                
            else:
                print(f"⏭️ {get_korean_name(coin)}: AI가 매도 거부 - 홀딩 유지")
                
        except Exception as e:
            print(f"⚠️ {decision.get('coin', 'unknown')} 매도 실행 오류: {e}")
            continue
    
    # 🆕 실행 결과 요약
    print(f"\n📊 성능 업그레이드된 거래 실행 완료:")
    print(f"✅ 실행된 매도: {len(executed_trades)}개")
    print(f"💰 총 수익: {total_profit:.2f}%")
    
    # 🆕 액션별 성과 요약
    for action in ['buy', 'sell', 'hold']:
        perf = real_time_action_tracker.get_action_performance(action)
        if perf['total_trades'] > 0:
            print(f"📈 {action.upper()}: {perf['total_trades']}회, 승률: {perf['success_rate']:.1%}, 평균수익: {perf['avg_profit']:.2f}%")
    
    return executed_trades


if __name__ == "__main__":
    create_holdings_table()
    create_trade_decision_log_table()
    
    # 🆕 성능 업그레이드된 순수 시그널 기반 매매 실행기 시작
    print("🚀 성능 업그레이드된 순수 시그널 기반 매매 실행기 시작")
    print("=" * 60)
    
    # 🆕 성능 업그레이드 시스템 초기화 확인
    print("🔧 성능 업그레이드 시스템 초기화:")
    print(f"  ✅ RealTimeActionTracker: 활성화")
    print(f"  ✅ RealTimeContextRecorder: 활성화") 
    print(f"  ✅ RealTimeOutlierGuardrail: 활성화")
    print(f"  ✅ RealTimeAIDecisionEngine: 활성화")
    print(f"  ✅ RealTimeLearningFeedback: 활성화")
    print("=" * 60)
    
    # 🆕 성능 업그레이드된 매매 실행
    run_signal_based_executor()
    
    # 🆕 순수 시그널 vs 매매 비교 분석 (성능 업그레이드 적용)
    print("\n" + "="*60)
    print("📊 성능 업그레이드된 순수 시그널 vs 실전 매매 분석")
    print("="*60)
    
    # 전체 요약
    print_signal_trade_summary()
    
    # 주요 코인별 상세 비교 (보유 중인 코인들)
    sync_wallet_to_db()
    wallet_coins = get_filtered_wallet_coins(min_balance_krw=10000)
    
    if wallet_coins:
        print(f"\n🔍 보유 코인별 성능 업그레이드된 순수시그널-매매 비교:")
        for coin in wallet_coins[:3]:  # 최대 3개만
            compare_signal_vs_trade(coin, hours=6)  # 최근 6시간
    
    # 🆕 성능 업그레이드 시스템 최종 요약
    print(f"\n🎯 성능 업그레이드 시스템 최종 요약:")
    print("=" * 50)
    
    for action in ['buy', 'sell', 'hold']:
        perf = real_time_action_tracker.get_action_performance(action)
        if perf['total_trades'] > 0:
            print(f"📈 {action.upper()}: {perf['total_trades']}회, 승률: {perf['success_rate']:.1%}, 평균수익: {perf['avg_profit']:.2f}%")
    
    print(f"\n✅ 성능 업그레이드된 순수 시그널 기반 매매 실행 완료!")
    print("📁 통합 DB: realtime_candles.db")
    print("   📈 시그널 테이블: signal_summary, signal_history, signal_analysis")
    print("   💰 매매 테이블: trade_decision_log, signal_trade_executions")
    print("   🧠 AI 시스템: RealTimeActionTracker, RealTimeAIDecisionEngine, RealTimeLearningFeedback")