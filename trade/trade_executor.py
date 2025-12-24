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
import os

# 현재 스크립트의 디렉토리를 path에 추가하여 같은 폴더 내의 모듈을 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 상위 디렉토리(프로젝트 루트)도 추가
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 🔥 [추가] os 모듈 import (중복 제거됨)
import time
import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta

# 🔧 [경로 수정] trade_manager는 trade 패키지 내에 있음
try:
    from trade.trade_manager import sync_wallet_to_db, get_filtered_wallet_coins, execute_trade_with_timeout, get_order_chance, wait_for_balance_update, fetch_tick_size_from_bithumb, execute_trades_parallel, get_available_balance
except ImportError:
    # 하위 호환성 (trade 폴더가 path에 있는 경우)
    from trade_manager import sync_wallet_to_db, get_filtered_wallet_coins, execute_trade_with_timeout, get_order_chance, wait_for_balance_update, fetch_tick_size_from_bithumb, execute_trades_parallel, get_available_balance

# 🔧 [경로 수정] market_analyzer에서 한국어 이름 조회 가져오기
try:
    from market.coin_market.market_analyzer import get_korean_name
except ImportError:
    print("⚠️ market_analyzer 로드 실패 - 기본 get_korean_name 사용")
    def get_korean_name(symbol):
        return symbol
from typing import Dict, Any, List

# 🆕 Thompson Sampling 학습기 임포트 (가상/실전 매매 일치화)
try:
    from trade.virtual_trade_learner import ThompsonSamplingLearner
except ImportError:
    print("⚠️ ThompsonSamplingLearner 로드 실패")
    ThompsonSamplingLearner = None

# 🆕 학습된 청산 파라미터 모듈 (가상매매와 동일한 매매 기법 적용)
try:
    from trade.core.exit_params import should_take_profit, should_stop_loss, get_exit_params
    LEARNED_EXIT_AVAILABLE = True
except ImportError:
    LEARNED_EXIT_AVAILABLE = False
    print("⚠️ 학습된 청산 파라미터 모듈 로드 실패 - 기본 청산 로직 사용")

# DB 경로 설정 (전역 변수로 미리 설정)
# 1. 시그널/캔들 DB (환경변수 우선, 없으면 trade_candles.db 사용)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT, 'market', 'coin_market', 'data_storage')

try:
    os.makedirs(_DEFAULT_DB_DIR, exist_ok=True)
except OSError:
    pass

# 🆕 전략 DB 경로 설정 (virtual_trade_learner와 동일한 로직 사용)
_env_strategy_base = os.getenv('STRATEGY_DB_PATH')
_default_strategy_base = os.path.join(_DEFAULT_DB_DIR, 'learning_strategies')

if _env_strategy_base and (_env_strategy_base.startswith('/workspace') or _env_strategy_base.startswith('\\workspace')):
    if os.name == 'nt':
         _strategy_base = _default_strategy_base
    else:
         _strategy_base = _env_strategy_base
else:
    _strategy_base = _env_strategy_base or _default_strategy_base

if os.path.isdir(_strategy_base) or not _strategy_base.endswith('.db'):
    STRATEGY_DB_PATH = os.path.join(_strategy_base, 'common_strategies.db')
else:
    STRATEGY_DB_PATH = _strategy_base

# 🆕 trade_candles.db 우선 사용
_trade_candles_path = os.path.join(_DEFAULT_DB_DIR, 'trade_candles.db')
# ⚠️ realtime_candles.db는 더 이상 사용하지 않음 (trade_candles.db로 통일)
_default_candle_db = _trade_candles_path

DB_PATH = os.getenv('RL_DB_PATH')
if not DB_PATH:
    DB_PATH = _default_candle_db

# 🆕 통합 트레이딩 시스템 DB 경로 (섀도우 + 실전 매매)
DEFAULT_TRADING_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'trading_system.db')
TRADING_SYSTEM_DB_PATH = os.getenv('TRADING_DB_PATH')
if not TRADING_SYSTEM_DB_PATH:
    TRADING_SYSTEM_DB_PATH = DEFAULT_TRADING_DB_PATH


def load_virtual_trade_decisions(max_age_minutes: int = 30) -> Dict[str, Dict]:
    """🆕🆕 가상매매 결정 테이블에서 최신 결정 읽기
    
    가상매매에서 모든 분석(레짐, Thompson Sampling, 기대수익률 등)을 완료한 결과를 읽어옴
    실전매매에서는 이 결정을 그대로 사용하여 매매 실행
    
    Returns:
        Dict[str, Dict]: coin -> decision_data 매핑
    """
    try:
        cutoff_time = int(time.time()) - (max_age_minutes * 60)
        
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            query = """
                SELECT coin, timestamp, decision, signal_score, confidence, current_price,
                       target_price, expected_profit_pct, thompson_score, thompson_approved,
                       regime_score, regime_name, viability_passed, reason,
                       is_holding, entry_price, profit_loss_pct
                FROM virtual_trade_decisions
                WHERE timestamp > ? AND processed = 0
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql(query, conn, params=(cutoff_time,))
            
            if df.empty:
                return {}
            
            # 코인별로 가장 최신 결정만 사용
            decisions = {}
            for _, row in df.iterrows():
                coin = row['coin']
                if coin not in decisions:  # 첫 번째(최신) 결정만 사용
                    decisions[coin] = {
                        'coin': coin,
                        'timestamp': row['timestamp'],
                        'decision': row['decision'],
                        'signal_score': row['signal_score'],
                        'confidence': row['confidence'],
                        'current_price': row['current_price'],
                        'target_price': row['target_price'],
                        'expected_profit_pct': row['expected_profit_pct'],
                        'thompson_score': row['thompson_score'],
                        'thompson_approved': bool(row['thompson_approved']),
                        'regime_score': row['regime_score'],
                        'regime_name': row['regime_name'],
                        'viability_passed': bool(row['viability_passed']),
                        'reason': row['reason'],
                        'is_holding': bool(row['is_holding']),
                        'entry_price': row['entry_price'],
                        'profit_loss_pct': row['profit_loss_pct']
                    }
            
            return decisions
            
    except Exception as e:
        print(f"⚠️ 가상매매 결정 로드 오류: {e}")
        return {}


def mark_decision_processed(coin: str, timestamp: int):
    """🆕 가상매매 결정을 처리 완료로 표시"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            conn.execute("""
                UPDATE virtual_trade_decisions
                SET processed = 1
                WHERE coin = ? AND timestamp = ?
            """, (coin, timestamp))
            conn.commit()
    except Exception as e:
        print(f"⚠️ 결정 처리 완료 표시 오류: {e}")

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
    
    def record_action_result(self, action: str, profit: float, success: bool, amount: float, symbol: str):
        """액션 결과 기록 (실전 매매 특화)"""
        if action in self.action_performance:
            self.action_performance[action]['trades'] += 1
            self.action_performance[action]['total_profit'] += profit
            self.action_performance[action]['total_amount'] += amount
            if success:
                self.action_performance[action]['wins'] += 1
        
        # 코인별 성과 추적
        if symbol not in self.coin_performance:
            self.coin_performance[symbol] = {'trades': 0, 'wins': 0, 'total_profit': 0.0}
        self.coin_performance[symbol]['trades'] += 1
        self.coin_performance[symbol]['total_profit'] += profit
        if success:
            self.coin_performance[symbol]['wins'] += 1
    
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
    
    def get_coin_performance(self, symbol: str) -> dict:
        """코인별 성과 반환"""
        if symbol not in self.coin_performance:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        perf = self.coin_performance[symbol]
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
                'symbol': signal_data.get('symbol', 'unknown'),
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
        # 🚀 초기화 시 DB에서 과거 성과 로드
        self.load_history_from_db()
        
    def load_history_from_db(self):
        """DB에서 과거 거래 기록을 로드하여 학습 상태 복원"""
        try:
            # 통합 DB 경로 사용 (없으면 전역 변수 참조 시도)
            db_path = TRADING_SYSTEM_DB_PATH if 'TRADING_SYSTEM_DB_PATH' in globals() else DB_PATH
            
            with sqlite3.connect(db_path) as conn:
                # real_trade_history 테이블이 있는지 확인
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='real_trade_history'")
                if not cursor.fetchone():
                    # 테이블이 없으면 trade_decision_log 확인 (구버전 호환)
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_decision_log'")
                    if not cursor.fetchone():
                        return
                    table_name = 'trade_decision_log'
                else:
                    table_name = 'real_trade_history'
                
                # 최근 1000개 거래 내역 로드 (실행된 것만)
                query = f"""
                    SELECT coin, profit_pct, action 
                    FROM {table_name}
                    WHERE executed = 1 AND profit_pct IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 1000
                """
                rows = cursor.execute(query).fetchall()
                
                for coin, profit, action in rows:
                    if coin not in self.coin_patterns:
                        self.coin_patterns[coin] = {'trades': 0, 'wins': 0, 'total_profit': 0.0}
                    
                    self.coin_patterns[coin]['trades'] += 1
                    self.coin_patterns[coin]['total_profit'] += profit
                    if profit > 0:
                        self.coin_patterns[coin]['wins'] += 1
                        
            print(f"✅ [RealTimeLearningFeedback] 과거 거래 {len(rows)}건 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 과거 데이터 로드 중 오류 (무시하고 진행): {e}")

    def record_trade_result(self, symbol: str, trade_result: dict):
        """거래 결과 기록"""
        try:
            trade_id = f"{symbol}_{trade_result.get('timestamp', int(time.time()))}"
            
            self.trade_feedback[trade_id] = {
                'symbol': symbol,
                'timestamp': trade_result.get('timestamp', int(time.time())),
                'action': trade_result.get('action', 'unknown'),
                'profit': trade_result.get('profit', 0.0),
                'success': trade_result.get('profit', 0.0) > 0,
                'amount': trade_result.get('amount', 0.0),
                'context': trade_result.get('context', {})
            }
            
            # 코인별 패턴 업데이트
            if symbol not in self.coin_patterns:
                self.coin_patterns[symbol] = {'trades': 0, 'wins': 0, 'total_profit': 0.0}
            
            self.coin_patterns[symbol]['trades'] += 1
            self.coin_patterns[symbol]['total_profit'] += trade_result.get('profit', 0.0)
            if trade_result.get('profit', 0.0) > 0:
                self.coin_patterns[symbol]['wins'] += 1
                
        except Exception as e:
            print(f"⚠️ 실전 매매 학습 피드백 기록 오류: {e}")
    
    def get_coin_learning_data(self, symbol: str) -> dict:
        """코인별 학습 데이터 반환"""
        if symbol not in self.coin_patterns:
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0}
        
        pattern = self.coin_patterns[symbol]
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

# 시그널 기반 거래 결정 내역 테이블 생성 (최초 1회 실행 시 생성)
def create_signal_trade_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                symbol TEXT,
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
                symbol TEXT,
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
        
        # 🆕 보유 시간 전용 테이블 (가벼움, 매도 시 삭제)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS current_position_times (
                coin TEXT PRIMARY KEY,
                buy_timestamp INTEGER NOT NULL,
                entry_price REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

def create_holdings_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                symbol TEXT PRIMARY KEY,
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
            symbol TEXT NOT NULL,
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
        wallet_df = pd.read_sql('SELECT symbol, quantity FROM holdings', conn, index_col='symbol')
    return wallet_df

# 🆕 가격 포맷팅 헬퍼 함수 (소수점 자릿수 동적 결정)
def format_price(price: float) -> str:
    """가격에 따라 적절한 소수점 자릿수로 포맷팅
    
    - 1원 미만: 소수점 4자리 (예: 0.5912)
    - 1~100원: 소수점 2자리 (예: 19.40)
    - 100원 이상: 소수점 0자리 + 천단위 콤마 (예: 4,544)
    """
    if price < 1:
        return f"{price:.4f}"
    elif price < 100:
        return f"{price:.2f}"
    else:
        return f"{price:,.0f}"

# 매수 금액 불러오기
def get_entry_price(symbol):
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT avg_buy_price FROM holdings WHERE symbol=?"
        result = conn.execute(query, (symbol,)).fetchone()
        # 결과가 없으면 None이 아니라 0.0 반환 (타입 안전성 보장)
        return result[0] if result and result[0] is not None else 0.0

# 보유 수량 불러오기
def get_quantity(symbol):
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT quantity FROM holdings WHERE symbol=?"
        result = conn.execute(query, (symbol,)).fetchone()
        return result[0] if result else 0.0

# 추가 매수 여부 결정 함수
def should_add_buy(coin, signal_score, confidence, current_price, entry_price):
    """이미 보유한 코인에 대한 추가 매수(피라미딩) 여부를 결정
    
    ⚠️ 물타기(손실 중 추매) 금지 - 수익 중일 때만 추매 허용
    """
    if entry_price is None or entry_price <= 0:
        return True  # 보유하지 않은 코인이므로 신규 매수
    
    # 현재 수익률 계산
    profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
    
    # ❌ 손실 중이면 추매 금지 (물타기 금지)
    if profit_loss_pct < 0:
        return False
    
    # 🎯 피라미딩 조건 (수익 중일 때만)
    # 1. 수익률 1% 이상 + 시그널 점수 높을 때
    if profit_loss_pct >= 1.0 and signal_score >= 0.06 and confidence >= 0.7:
        return True
    
    # 2. 수익률 3% 이상 + 시그널 점수 양호할 때
    if profit_loss_pct >= 3.0 and signal_score >= 0.05 and confidence >= 0.65:
        return True
    
    return False


# 🆕🆕 보유 시간 관리 함수들 (current_position_times 테이블)
def record_position_buy_time(coin: str, entry_price: float = 0.0):
    """매수 시 보유 시간 기록"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO current_position_times (coin, buy_timestamp, entry_price)
                VALUES (?, ?, ?)
            """, (coin, int(time.time()), entry_price))
            conn.commit()
    except Exception as e:
        logging.warning(f"보유 시간 기록 오류 ({coin}): {e}")


def remove_position_time(coin: str):
    """매도 시 보유 시간 기록 삭제"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            conn.execute("DELETE FROM current_position_times WHERE coin = ?", (coin,))
            conn.commit()
    except Exception as e:
        logging.warning(f"보유 시간 삭제 오류 ({coin}): {e}")


def get_holding_duration(coin: str) -> int:
    """코인의 보유 시간(초) 조회
    
    조회 순서:
    1. current_position_times (실전매매 전용, 가벼움)
    2. virtual_positions (가상매매 기록)
    3. 둘 다 없으면 기본값 24시간
    
    Returns:
        보유 시간(초)
    """
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            cursor = conn.cursor()
            current_time = int(time.time())
            
            # 1. 실전매매 보유 시간 테이블에서 조회 (가장 정확)
            cursor.execute("""
                SELECT buy_timestamp FROM current_position_times 
                WHERE coin = ?
            """, (coin,))
            
            row = cursor.fetchone()
            if row and row[0]:
                buy_timestamp = row[0]
                holding_seconds = current_time - buy_timestamp
                return max(0, holding_seconds)
            
            # 2. 가상매매 포지션에서 조회 (fallback)
            cursor.execute("""
                SELECT entry_timestamp FROM virtual_positions 
                WHERE coin = ?
            """, (coin,))
            
            row = cursor.fetchone()
            if row and row[0]:
                entry_timestamp = row[0]
                holding_seconds = current_time - entry_timestamp
                return max(0, holding_seconds)
            
            # 3. 기록 없으면 기본값 24시간 (갈아타기 조건 체크 가능하도록)
            return 24 * 3600  # 24시간
            
    except Exception as e:
        logging.warning(f"보유 시간 조회 오류 ({coin}): {e}")
        return 24 * 3600  # 오류 시에도 기본값 반환


# 🆕🆕 갈아타기 조건 체크 함수들 (횡보/손실장기화/목표미달)
def check_switch_condition(coin: str, profit_pct: float, holding_hours: float, 
                           target_price: float = 0, current_price: float = 0) -> tuple:
    """갈아타기 조건 체크 (3가지 조건 중 하나라도 충족하면 True)
    
    Returns:
        (should_switch: bool, reason: str, switch_type: str)
    """
    # 1. 횡보 감지: 12시간+ 보유 & 수익률 ±2% 이내
    if holding_hours >= 12.0 and -2.0 <= profit_pct <= 2.0:
        return True, f"횡보 감지 ({holding_hours:.1f}시간, {profit_pct:+.2f}%)", "sideways"
    
    # 2. 손실 장기화: 24시간+ 보유 & 손실 -3% 이하 지속
    if holding_hours >= 24.0 and profit_pct <= -3.0:
        return True, f"손실 장기화 ({holding_hours:.1f}시간, {profit_pct:+.2f}%)", "stagnant_loss"
    
    # 3. 목표 미달: 예상 시간 2배 경과 & 목표 50% 미달
    if target_price > 0 and current_price > 0 and holding_hours >= 24.0:
        # 목표가까지 남은 비율 계산
        target_distance_pct = ((target_price - current_price) / current_price) * 100
        # 목표의 50% 이상 남아있고, 24시간 이상 경과했으면 목표 미달로 판정
        if target_distance_pct > 2.0:  # 목표까지 2% 이상 남음
            return True, f"목표 미달 ({holding_hours:.1f}시간, 목표까지 {target_distance_pct:.1f}%)", "target_miss"
    
    return False, "", ""


def find_best_switch_target(virtual_decisions: dict, wallet_coins: list, 
                            current_coin: str, min_signal_score: float = 0.3) -> dict:
    """갈아타기 대상 코인 찾기 (시그널 점수 기반)
    
    조건:
    1. 가상매매에서 'buy' 결정된 코인
    2. 시그널 점수 > min_signal_score
    3. 아직 보유하지 않은 코인
    4. Thompson 점수 양호 (0.4 이상)
    
    Returns:
        {'coin': str, 'signal_score': float, ...} 또는 None
    """
    best_candidate = None
    best_score = 0
    
    for coin, decision in virtual_decisions.items():
        # 조건 1: 가상매매에서 'buy' 결정된 코인만
        if decision['decision'] != 'buy':
            continue
        
        # 조건 2: 시그널 점수 기준
        signal_score = decision['signal_score']
        if signal_score < min_signal_score:
            continue
        
        # 조건 3: 이미 보유 중인 코인은 제외
        if coin in wallet_coins:
            continue
        
        # 현재 코인과 같으면 제외
        if coin == current_coin:
            continue
        
        # 조건 4: Thompson 점수 체크 (0.4 이상)
        thompson_score = decision.get('thompson_score', 0)
        if thompson_score < 0.4:
            continue
        
        # 가장 좋은 시그널 점수 코인 선택
        if signal_score > best_score:
            best_candidate = {
                'coin': coin,
                'signal_score': signal_score,
                'expected_profit_pct': decision.get('expected_profit_pct', 0),
                'thompson_score': thompson_score,
                'current_price': decision.get('current_price', 0),
                'target_price': decision.get('target_price', 0),
                'reason': f"시그널 점수 {signal_score:.3f}, Thompson {thompson_score:.2f}",
                'decision_timestamp': decision.get('timestamp', 0)
            }
            best_score = signal_score
    
    return best_candidate


# 🆕 갈아타기 후보 찾기 (기존 - 수익 중일 때 점수 차이 기반)
def find_switch_candidate(current_coin: str, current_profit_pct: float, current_signal_score: float,
                          holding_duration_hours: float, virtual_decisions: dict, wallet_coins: list) -> dict:
    """갈아타기 대상 코인 찾기 (수익 중일 때 점수 차이 기반)
    
    조건:
    1. 현재 코인이 수익 중 (+1% 이상)
    2. 보유 시간 충분 (2시간 이상)
    3. 신규 코인이 가상매매에서 'buy' 결정됨
    4. 신규 코인 시그널이 현재 코인보다 압도적으로 좋음
    
    Returns:
        {'coin': str, 'signal_score': float, 'reason': str} 또는 None
    """
    # 조건 1: 수익 중이어야 함
    if current_profit_pct < 1.0:
        return None
    
    # 조건 2: 보유 시간 기준 최소 임계값 (시간에 따라 점수 차이 요구사항 완화)
    if holding_duration_hours < 2.0:
        return None  # 너무 이름
    
    # 보유 시간에 따른 점수 차이 임계값 조정
    if holding_duration_hours >= 12:
        min_score_diff = 0.3  # 12시간 이상: 0.3 차이면 갈아타기
    elif holding_duration_hours >= 6:
        min_score_diff = 0.4  # 6~12시간: 0.4 차이
    else:
        min_score_diff = 0.5  # 2~6시간: 0.5 차이 (신중하게)
    
    best_candidate = None
    best_score_diff = 0
    
    for coin, decision in virtual_decisions.items():
        # 조건 3: 가상매매에서 'buy' 결정된 코인만
        if decision['decision'] != 'buy':
            continue
        
        # 이미 보유 중인 코인은 제외
        if coin in wallet_coins:
            continue
        
        # 현재 코인과 같으면 제외
        if coin == current_coin:
            continue
        
        # 조건 4: 시그널 점수 차이 계산
        new_signal_score = decision['signal_score']
        score_diff = new_signal_score - current_signal_score
        
        if score_diff >= min_score_diff and score_diff > best_score_diff:
            best_candidate = {
                'coin': coin,
                'signal_score': new_signal_score,
                'score_diff': score_diff,
                'expected_profit_pct': decision['expected_profit_pct'],
                'thompson_score': decision['thompson_score'],
                'current_price': decision['current_price'],
                'reason': f"점수 차이 {score_diff:.2f} (현재 {current_signal_score:.2f} → 신규 {new_signal_score:.2f})",
                'decision_timestamp': decision['timestamp']
            }
            best_score_diff = score_diff
    
    return best_candidate


# 🆕 일일 갈아타기 횟수 조회
def get_daily_switch_count() -> int:
    """오늘 갈아타기 횟수 조회"""
    try:
        today_start = int(time.time()) - (int(time.time()) % 86400)  # 오늘 00:00
        
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM real_trade_history 
                WHERE action = 'switch' AND timestamp >= ?
            """, (today_start,))
            
            row = cursor.fetchone()
            return row[0] if row else 0
            
    except Exception as e:
        return 0


# 🆕 수집된 코인 전체 로딩 (1일봉 거래량 기준 정렬, 상위 40%)
MARKET_ANALYSIS_RATIO = 0.40  # 시장 분석 대상 비율 (40%)

def load_target_coins():
    """거래량 상위 40% 코인 로딩 (엔진 확장성 고려, 비율 기반)"""
    with sqlite3.connect(DB_PATH) as conn:
        # 1. 전체 코인 수 조회
        total_query = """
            SELECT COUNT(DISTINCT symbol) as cnt FROM candles
            WHERE interval='1d' AND timestamp=(SELECT MAX(timestamp) FROM candles WHERE interval='1d')
        """
        total_df = pd.read_sql(total_query, conn)
        total_coins = total_df['cnt'].iloc[0] if not total_df.empty else 0
        
        # 2. 상위 40% 계산 (최소 50개, 최대 500개)
        target_count = int(total_coins * MARKET_ANALYSIS_RATIO)
        target_count = max(50, min(target_count, 500))
        
        # 3. 거래량 상위 코인 조회
        query = """
            SELECT symbol FROM candles
            WHERE interval='1d' AND timestamp=(SELECT MAX(timestamp) FROM candles WHERE interval='1d')
            ORDER BY volume DESC
            LIMIT ?
        """
        coins = pd.read_sql(query, conn, params=(target_count,))['symbol'].tolist()
        
        print(f"📊 실전 매매 대상: 전체 {total_coins}개 중 상위 {len(coins)}개 ({MARKET_ANALYSIS_RATIO*100:.0f}%)")
        return coins

# 🆕 실전 매매용 시그널 점수 조회 (realtime_signals 테이블에서)
def load_realtime_signal(symbol: str, interval: str = 'combined'):
    """signals 테이블에서 코인의 최신 통합 시그널 정보 로드 (combined 시그널만 사용)"""
    try:
        # 🚀 trading_system.db 사용
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            # combined 시그널만 조회 (인터벌 합치기 로직 제거)
            # symbol 우선 조회, 없으면 coin 조회 (호환성)
            try:
                query = """
                    SELECT * FROM signals
                    WHERE symbol = ? AND interval = 'combined'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                df = pd.read_sql(query, conn, params=(symbol,))
            except:
                query = """
                    SELECT * FROM signals
                    WHERE coin = ? AND interval = 'combined'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                df = pd.read_sql(query, conn, params=(symbol,))
        
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
        print(f"⚠️ 실전 매매용 시그널 조회 오류 ({symbol}/{interval}): {e}")
        return None

# 최신 realtime_signals에서 시그널 정보 가져오기 (통합 시그널 기준) - 기존 호환성 유지
def load_signal_from_summary(coin):
    """signals 테이블에서 코인의 최신 통합 시그널 정보 로드 (통합 DB 사용)"""
    with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
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
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
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

# 🆕 시장 상황 분석 캐시 (성능 최적화)
_market_context_cache = {'data': None, 'timestamp': 0}
_MARKET_CONTEXT_CACHE_TTL = 300  # 5분 캐시

# 🆕 시장 상황 분석 (Core 위임 + 캐싱)
def get_market_context():
    """시장 상황 분석 (트렌드, 변동성 등) - Core 모듈 사용 + 캐싱"""
    global _market_context_cache
    
    try:
        # 🚀 캐시 확인 (5분 TTL)
        current_time = time.time()
        if (_market_context_cache['data'] is not None and 
            current_time - _market_context_cache['timestamp'] < _MARKET_CONTEXT_CACHE_TTL):
            return _market_context_cache['data']
        
        # 🆕 Core MarketAnalyzer 사용 (거래량 상위 40% 코인 기준)
        from trade.core.market import MarketAnalyzer
        analyzer = MarketAnalyzer(db_path=os.getenv('TRADING_SYSTEM_DB_PATH'))
        result = analyzer.analyze_market_regime()
        
        regime = result.get('regime', 'Neutral')
        volatility = result.get('volatility', 0.02)
        score = result.get('score', 0.5)
        
        # Trend 매핑 (레짐 기반)
        regime_lower = regime.lower()
        if 'bullish' in regime_lower or 'bull' in regime_lower:
            trend = 'bullish'
        elif 'bearish' in regime_lower or 'bear' in regime_lower:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        context = {
            'trend': trend,
            'volatility': volatility,
            'regime': regime,
            'score': score
        }
        
        # 🚀 캐시 저장
        _market_context_cache = {'data': context, 'timestamp': current_time}
        
        return context
        
    except Exception as e:
        print(f"⚠️ 시장 상황 분석 오류 (Core 연동): {e}")
        return {'trend': 'neutral', 'volatility': 0.02}

# 🆕 코인별 성과 분석 (유지 - 필요한 경우 Core로 이동 고려)
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

# 🆕 손절 성과 분석 (유지)
def analyze_stop_loss_performance(coin):
    # ... (기존 로직 유지) ...
    try:
        with sqlite3.connect(DB_PATH) as conn:
            thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp())
            df = pd.read_sql("""
                SELECT profit_pct FROM trade_decision_log 
                WHERE coin = ? AND timestamp > ? AND executed = 1 
                AND reason LIKE '%stop_loss%' OR reason LIKE '%손절%'
                ORDER BY timestamp DESC
            """, conn, params=(coin, thirty_days_ago))
            if df.empty: return 0.5
            avg_stop_loss = df['profit_pct'].mean()
            if avg_stop_loss < -10.0: return 0.8
            elif avg_stop_loss > -5.0: return 0.2
            else: return 0.5
    except Exception as e:
        return 0.5

# 🆕 시장 변동성 계산 (Core 위임)
def get_market_volatility():
    """시장 변동성 계산 - Core 모듈 사용"""
    try:
        context = get_market_context()
        return context.get('volatility', 0.02)
    except Exception:
        return 0.02

# 🆕 시장 상황 분석 (Core 위임)
def analyze_market_conditions():
    """전체 시장 상황 분석 - Core 모듈 사용"""
    try:
        from trade.core.market import MarketAnalyzer
        analyzer = MarketAnalyzer(db_path=os.getenv('TRADING_SYSTEM_DB_PATH'))
        result = analyzer.analyze_market_regime()
        return result.get('score', 0.5)
            
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
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            # signal_history 테이블이 없으면 signals 테이블에서 조회
            try:
                query = """
                    SELECT timestamp, action, signal_score, confidence, reason, price
                    FROM signal_history
                    WHERE coin = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """
                cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
                df = pd.read_sql(query, conn, params=(coin, cutoff_time))
            except:
                # signals 테이블 사용
                query = """
                    SELECT timestamp, action, signal_score, confidence, reason, current_price as price
                    FROM signals
                    WHERE coin = ? AND interval = 'combined' AND timestamp > ?
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
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            # real_trade_history 테이블 조회 (trade_decision_log 대체)
            query = """
                SELECT timestamp, action, reason, executed, execution_price, execution_type
                FROM real_trade_history
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
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            signal_stats = pd.read_sql("""
                SELECT action, COUNT(*) as count, AVG(signal_score) as avg_score
                FROM signals
                WHERE interval = 'combined'
                GROUP BY action
            """, conn)
        
        # 매매 현황 (real_trade_history 테이블 사용)
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            trade_stats = pd.read_sql("""
                SELECT action, COUNT(*) as count, 
                       SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_count
                FROM real_trade_history
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
    """🆕🆕 개선된 실전매매 실행 (판단/실행 분리 + 우선순위 기반)
    
    [개선된 흐름]
    - STEP 1: 현재 상태 수집 (지갑, 예수금, 보유 코인)
    - STEP 2: 전체 판단 (실행 X) - 매도/홀딩/갈아타기/신규매수 판단만
    - STEP 3: 우선순위 기반 실행
        1순위: 손절 (즉시 실행)
        2순위: 갈아타기 (매도→매수 원자적)
        3순위: 일반 매도/익절
        4순위: 신규 매수 (예수금 확인 후)
    - STEP 4: 결과 검증
    """
    
    # ═══════════════════════════════════════════════════════════════
    # 🚀 [STEP 1] 현재 상태 수집
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("🚀 [STEP 1] 현재 상태 수집")
    print("=" * 60)
    
    sync_wallet_to_db()
    wallet_info = get_filtered_wallet_coins(min_balance_krw=10000, return_dict=True)
    wallet_coins = list(wallet_info.keys())
    initial_balance = get_available_balance()
    
    print(f"💼 보유 자산: {len(wallet_coins)}개 | {[get_korean_name(coin) for coin in wallet_coins]}")
    print(f"💰 예수금: {initial_balance:,.0f}원")

    # ═══════════════════════════════════════════════════════════════
    # 🚀 [STEP 2] 전체 판단 (실행 X) - 매도/홀딩/갈아타기/신규매수 판단만
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🚀 [STEP 2] 전체 판단 (실행 없이 판단만)")
    print("=" * 60)
    
    # 결과 저장용 리스트
    stop_loss_decisions = []    # 1순위: 손절 (즉시 실행)
    switch_decisions = []       # 2순위: 갈아타기
    sell_decisions = []         # 3순위: 일반 매도/익절
    hold_decisions = []         # 홀딩 유지
    buy_candidates = []         # 4순위: 신규 매수
    
    # 🆕 가상매매 결정 로드 (Thompson 점수 참조용)
    virtual_decisions = load_virtual_trade_decisions(max_age_minutes=30)
    print(f"📥 가상매매 학습 데이터: {len(virtual_decisions)}개")
    
    # ─────────────────────────────────────────────────────────────────
    # [2-1] 보유 코인 판단: 매도/홀딩/갈아타기
    # ─────────────────────────────────────────────────────────────────
    print(f"\n📊 [2-1] 보유 코인 {len(wallet_coins)}개 판단 중...")
    
    for coin in wallet_coins:
        coin_info = wallet_info.get(coin, {})
        entry_price = coin_info.get('entry_price', 0.0)
        wallet_current_price = coin_info.get('current_price', 0.0)
        
        # 실전매매 독립적 시그널 계산
        signal_data = load_realtime_signal(coin, 'combined')
        
        # 가상매매 결정 (참고용)
        virtual_decision_ref = virtual_decisions.get(coin, {}).get('decision', 'N/A')
        virtual_thompson = virtual_decisions.get(coin, {}).get('thompson_score', 0.0)
        virtual_regime = virtual_decisions.get(coin, {}).get('regime_name', 'N/A')
        target_price_ref = virtual_decisions.get(coin, {}).get('target_price', 0)
        
        if signal_data is None:
            if coin in virtual_decisions:
                decision = virtual_decisions[coin]
                signal_score = decision['signal_score']
                confidence = decision['confidence']
                reason = f"(가상매매 참조) {decision['reason']}"
                current_price = wallet_current_price if wallet_current_price > 0 else decision['current_price']
                pure_action = decision['decision']
            else:
                print(f"⚠️ {get_korean_name(coin)}: 시그널 없음 → 홀딩 유지")
                hold_decisions.append({
                    'coin': coin, 'action': 'hold', 'signal_score': 0.0,
                    'confidence': 0.0, 'reason': '시그널 없음', 'profit_loss_pct': 0.0
                })
                continue
        else:
            signal_score = signal_data['signal_info']['signal_score']
            confidence = signal_data['signal_info']['confidence']
            reason = signal_data['signal_info'].get('reason', 'signal_based')
            current_price = wallet_current_price if wallet_current_price > 0 else signal_data['market_data']['price']
            pure_action = signal_data['signal_info'].get('action', 'hold')
        
        # 수익률 계산
        profit_loss_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 and current_price > 0 else 0.0
        
        # 보유 시간 조회
        holding_seconds = get_holding_duration(coin)
        holding_hours = holding_seconds / 3600
        
        # 로그 출력
        holding_info = f"진입가 {format_price(entry_price)}원, 수익률 {profit_loss_pct:+.2f}%, 보유 {holding_hours:.1f}h" if entry_price > 0 else "보유 중"
        print(f"📊 {get_korean_name(coin)}: {holding_info}")
        print(f"   📈 시그널: {pure_action} (점수: {signal_score:.3f})")
        print(f"   📥 참고: Thompson {virtual_thompson:.2f}, 레짐 {virtual_regime}")
        
        # 최종 액션 판단 (학습 기반 + 트레일링스탑 + 익절/손절)
        final_action = combine_signal_with_holding(
            coin=coin,
            pure_action=pure_action,
            signal_score=signal_score,
            profit_loss_pct=profit_loss_pct,
            signal_pattern=reason if reason else 'unknown',
            max_profit_pct=max(profit_loss_pct, 0.0),
            entry_volatility=0.02,
            holding_hours=holding_hours
        )
        
        # 공통 결정 데이터
        decision_data = {
            'coin': coin,
            'action': final_action,
            'signal_score': signal_score,
            'confidence': confidence,
            'reason': reason,
            'current_price': current_price,
            'entry_price': entry_price,
            'pure_action': pure_action,
            'profit_loss_pct': profit_loss_pct,
            'holding_hours': holding_hours,
            'decision_timestamp': int(time.time())
        }
        
        # ═══ 분류 ═══
        # 1순위: 손절 (stop_loss)
        if final_action == 'stop_loss':
            print(f"   🔴 판단: 손절 (1순위)")
            stop_loss_decisions.append(decision_data)
        
        # 2순위: 갈아타기 조건 체크
        elif final_action in ['hold'] and profit_loss_pct < 3.0:
            # 갈아타기 조건 체크 (횡보/손실장기화/목표미달)
            should_switch, switch_reason, switch_type = check_switch_condition(
                coin=coin,
                profit_pct=profit_loss_pct,
                holding_hours=holding_hours,
                target_price=target_price_ref,
                current_price=current_price
            )
            
            if should_switch:
                # 대안 코인 찾기
                target = find_best_switch_target(
                    virtual_decisions=virtual_decisions,
                    wallet_coins=wallet_coins,
                    current_coin=coin,
                    min_signal_score=0.25
                )
                
                if target:
                    print(f"   🔄 판단: 갈아타기 (2순위) → {get_korean_name(target['coin'])}")
                    decision_data['switch_reason'] = switch_reason
                    decision_data['switch_type'] = switch_type
                    decision_data['target'] = target
                    switch_decisions.append(decision_data)
                else:
                    print(f"   ⏸️ 판단: 갈아타기 조건 충족하나 대안 없음 → 홀딩")
                    hold_decisions.append(decision_data)
            else:
                print(f"   🟡 판단: 홀딩")
                hold_decisions.append(decision_data)
        
        # 3순위: 일반 매도/익절
        elif final_action in ['sell', 'take_profit', 'partial_sell']:
            print(f"   🟢 판단: {final_action} (3순위)")
            sell_decisions.append(decision_data)
        
        # 홀딩
        else:
            print(f"   🟡 판단: 홀딩")
            hold_decisions.append(decision_data)
    
    # ─────────────────────────────────────────────────────────────────
    # [2-2] 신규 매수 + 추가 매수 후보 판단
    # ─────────────────────────────────────────────────────────────────
    print(f"\n📊 [2-2] 매수 후보 판단 (신규 + 추매)...")
    
    MIN_SIGNAL_SCORE = 0.20           # 신규 매수 최소 시그널 점수
    MIN_SIGNAL_SCORE_ADDITIONAL = 0.35  # 추가 매수 최소 시그널 점수 (더 높음)
    MIN_THOMPSON_SCORE = 0.45
    MAX_SIGNAL_CANDIDATES = 5
    
    top_volume_coins = load_target_coins()
    print(f"📊 분석 대상: {len(top_volume_coins)}개 (거래량 상위 40%)")
    
    # 이미 매수 예정인 코인 추적 (갈아타기 대상 포함)
    pending_buy_coins = set()
    for sw in switch_decisions:
        if 'target' in sw:
            pending_buy_coins.add(sw['target']['coin'])
    
    analyzed_count = 0
    for coin in top_volume_coins:
        # 갈아타기 대상이면 스킵 (같은 사이클 내 중복 방지)
        if coin in pending_buy_coins:
            continue
        
        # 🆕 보유 중인 코인은 추가 매수 조건 체크
        is_additional_buy = coin in wallet_coins
        
        signal_data = load_realtime_signal(coin, 'combined')
        if signal_data is None:
            continue
        
        analyzed_count += 1
        
        signal_score = signal_data['signal_info'].get('signal_score', 0)
        confidence = signal_data['signal_info'].get('confidence', 0)
        current_price = signal_data['market_data'].get('price', 0)
        pure_action = signal_data['signal_info'].get('action', 'hold')
        target_price = signal_data['signal_info'].get('target_price', 0)
        
        # 가상매매 참조 (Thompson 점수)
        virtual_ref = virtual_decisions.get(coin, {})
        thompson_score = virtual_ref.get('thompson_score', 0.5)
        regime_name = virtual_ref.get('regime_name', 'Neutral')
        expected_profit = virtual_ref.get('expected_profit_pct', 0)
        
        if expected_profit == 0 and target_price > 0 and current_price > 0:
            expected_profit = ((target_price - current_price) / current_price) * 100
        
        # ═══════════════════════════════════════════════════════════════
        # 🆕 추가 매수 조건 (보유 중인 코인)
        # ═══════════════════════════════════════════════════════════════
        if is_additional_buy:
            coin_info = wallet_info.get(coin, {})
            entry_price = coin_info.get('entry_price', 0.0)
            wallet_current_price = coin_info.get('current_price', 0.0)
            
            # 현재 수익률
            current_profit_pct = ((wallet_current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
            
            # 보유 시간
            holding_seconds = get_holding_duration(coin)
            holding_hours = holding_seconds / 3600
            
            # 추가 매수 조건:
            # 1. 시그널 점수가 높음 (0.35 이상)
            # 2. 현재 수익률이 양수 (수익 중)
            # 3. 보유 시간이 1시간 이상 (너무 빨리 추매 방지)
            # 4. Thompson 점수가 충분히 높음
            if (signal_score >= MIN_SIGNAL_SCORE_ADDITIONAL and 
                current_profit_pct >= 0.5 and 
                holding_hours >= 1.0 and 
                thompson_score >= MIN_THOMPSON_SCORE):
                
                buy_candidates.append({
                    'coin': coin,
                    'signal_score': signal_score,
                    'confidence': confidence,
                    'reason': 'additional_buy_high_signal',
                    'price': current_price,
                    'pure_action': pure_action,
                    'is_additional_buy': True,
                    'entry_price': entry_price,
                    'current_profit_pct': current_profit_pct,
                    'target_price': target_price,
                    'expected_profit_pct': expected_profit,
                    'thompson_score': thompson_score,
                    'regime_name': regime_name,
                    'decision_timestamp': int(time.time())
                })
                print(f"   🔵 {get_korean_name(coin)}: 추매 후보 (점수: {signal_score:.3f}, 현수익: {current_profit_pct:+.2f}%)")
            continue  # 추가 매수 조건 체크 후 다음 코인으로
        
        # ═══════════════════════════════════════════════════════════════
        # 신규 매수 조건 체크
        # ═══════════════════════════════════════════════════════════════
        if signal_score < MIN_SIGNAL_SCORE:
            continue
        if thompson_score < MIN_THOMPSON_SCORE:
            continue
        if expected_profit < 0:
            continue
        if current_price <= 0:
            continue
        
        buy_candidates.append({
            'coin': coin,
            'signal_score': signal_score,
            'confidence': confidence,
            'reason': 'signal_based_new_buy',
            'price': current_price,
            'pure_action': pure_action,
            'is_additional_buy': False,
            'entry_price': 0,
            'target_price': target_price,
            'expected_profit_pct': expected_profit,
            'thompson_score': thompson_score,
            'regime_name': regime_name,
            'decision_timestamp': int(time.time())
        })
    
    # 시그널 점수 기준 정렬 후 상위 N개
    buy_candidates.sort(key=lambda x: x['signal_score'], reverse=True)
    buy_candidates = buy_candidates[:MAX_SIGNAL_CANDIDATES]
    
    print(f"✅ 분석 완료: {analyzed_count}개 중 {len(buy_candidates)}개 조건 충족")
    
    # ─────────────────────────────────────────────────────────────────
    # [2-3] 판단 요약
    # ─────────────────────────────────────────────────────────────────
    new_buy_count = len([c for c in buy_candidates if not c.get('is_additional_buy', False)])
    additional_buy_count = len([c for c in buy_candidates if c.get('is_additional_buy', False)])
    
    print(f"\n📋 [2-3] 판단 요약")
    print(f"   🔴 손절: {len(stop_loss_decisions)}개 (1순위)")
    print(f"   🔄 갈아타기: {len(switch_decisions)}개 (2순위)")
    print(f"   🟢 매도/익절: {len(sell_decisions)}개 (3순위)")
    print(f"   🟡 홀딩: {len(hold_decisions)}개")
    print(f"   🔵 신규매수: {new_buy_count}개 / 추매: {additional_buy_count}개 (4순위)")
    
    # ═══════════════════════════════════════════════════════════════
    # 🚀 [STEP 3] 우선순위 기반 실행
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🚀 [STEP 3] 우선순위 기반 실행")
    print("=" * 60)
    
    executed_trades = []
    executed_buy_coins = set()  # 이미 매수된 코인 추적 (중복 방지)
    
    # ─────────────────────────────────────────────────────────────────
    # [3-1] 1순위: 손절 실행 (즉시)
    # ─────────────────────────────────────────────────────────────────
    if stop_loss_decisions:
        print(f"\n🔴 [3-1] 손절 실행 ({len(stop_loss_decisions)}개)")
        stop_loss_results = execute_enhanced_signal_trades(stop_loss_decisions, [])
        executed_trades.extend(stop_loss_results)
        
        for dec in stop_loss_decisions:
            remove_position_time(dec['coin'])
            print(f"   ✅ {get_korean_name(dec['coin'])} 손절 완료 (수익률: {dec['profit_loss_pct']:+.2f}%)")
    
    # ─────────────────────────────────────────────────────────────────
    # [3-2] 2순위: 갈아타기 실행 (매도→매수 원자적)
    # ─────────────────────────────────────────────────────────────────
    MAX_DAILY_SWITCHES = 5
    daily_switch_count = get_daily_switch_count()
    
    if switch_decisions and daily_switch_count < MAX_DAILY_SWITCHES:
        print(f"\n🔄 [3-2] 갈아타기 실행 (남은 횟수: {MAX_DAILY_SWITCHES - daily_switch_count}회)")
        
        # 시그널 점수 기준 정렬
        switch_decisions.sort(key=lambda x: x['target']['signal_score'], reverse=True)
        
        for sw in switch_decisions:
            if daily_switch_count >= MAX_DAILY_SWITCHES:
                print(f"   ⚠️ 갈아타기 일일 한도 도달")
                break
            
            from_coin = sw['coin']
            target = sw['target']
            to_coin = target['coin']
            
            # 이미 매수된 코인이면 스킵
            if to_coin in executed_buy_coins:
                print(f"   ⏭️ {get_korean_name(to_coin)} 이미 매수됨 - 스킵")
                continue
            
            print(f"   🔄 {get_korean_name(from_coin)} → {get_korean_name(to_coin)}")
            print(f"      사유: {sw.get('switch_reason', 'unknown')}")
            
            # 매도 실행
            sell_trade_data = {
                'coin': from_coin,
                'action': 'switch',
                'interval': 'combined',
                'timestamp': int(time.time()),
                'signal': -1,
                'final_score': sw['signal_score'],
                'approved_by': ['Switch_Position'],
                'market_flow': 'Switch',
                'market_mode': 'Switch',
                'position_percentage': 1.0,
                'decision_status': 'approved',
                'confidence': 0.9
            }
            
            sell_success = execute_trade_with_timeout(sell_trade_data)
            
            if sell_success:
                print(f"      ✅ {get_korean_name(from_coin)} 매도 완료")
                remove_position_time(from_coin)
                
                # DB 기록
                log_trade_decision({
                    'timestamp': int(time.time()),
                    'coin': from_coin,
                    'interval': 'combined',
                    'action': 'switch',
                    'reason': 'position_switch',
                    'reason_detail': f"갈아타기: {get_korean_name(to_coin)}로 이동",
                    'entry_price': sw.get('entry_price', 0),
                    'current_price': sw['current_price'],
                    'profit_pct': sw['profit_loss_pct'],
                    'fusion_score': sw['signal_score'],
                    'rl_score': 0.0,
                    'market_mode': 'Switch',
                    'market_flow': 'Switch',
                    'gpt_approved': 1,
                    'executed': 1,
                    'execution_price': sw['current_price'],
                    'execution_amount': 0,
                    'execution_type': 'switch_sell',
                    'signal_score': sw['signal_score'],
                    'confidence': 0.9,
                    'holding_duration': int(sw['holding_hours'] * 3600)
                })
                
                time.sleep(0.5)
                
                # 매수 실행
                available_balance = get_available_balance()
                buy_amount = min(available_balance * 0.995, 5_000_000.0)
                
                if buy_amount >= 1_000_000:
                    buy_trade_data = {
                        'coin': to_coin,
                        'action': 'buy',
                        'interval': 'combined',
                        'timestamp': int(time.time()),
                        'signal': 1,
                        'final_score': target['signal_score'],
                        'approved_by': ['Switch_Position'],
                        'market_flow': 'Switch',
                        'market_mode': 'Switch',
                        'price': buy_amount,
                        'position_percentage': None,
                        'decision_status': 'approved',
                        'confidence': 0.9,
                        'ord_type': 'price'
                    }
                    
                    buy_success = execute_trade_with_timeout(buy_trade_data)
                    
                    if buy_success:
                        print(f"      ✅ {get_korean_name(to_coin)} 매수 완료")
                        record_position_buy_time(to_coin, target.get('current_price', 0))
                        executed_buy_coins.add(to_coin)
                        daily_switch_count += 1
                        
                        log_trade_decision({
                            'timestamp': int(time.time()),
                            'coin': to_coin,
                            'interval': 'combined',
                            'action': 'switch',
                            'reason': 'position_switch',
                            'reason_detail': f"갈아타기: {get_korean_name(from_coin)}에서 이동",
                            'entry_price': 0,
                            'current_price': target.get('current_price', 0),
                            'profit_pct': 0.0,
                            'fusion_score': target['signal_score'],
                            'rl_score': 0.0,
                            'market_mode': 'Switch',
                            'market_flow': 'Switch',
                            'gpt_approved': 1,
                            'executed': 1,
                            'execution_price': target.get('current_price', 0),
                            'execution_amount': buy_amount,
                            'execution_type': 'switch_buy',
                            'signal_score': target['signal_score'],
                            'confidence': 0.9
                        })
                        
                        if 'decision_timestamp' in target:
                            mark_decision_processed(to_coin, target['decision_timestamp'])
                        
                        print(f"      🎉 갈아타기 완료!")
                    else:
                        print(f"      ❌ {get_korean_name(to_coin)} 매수 실패")
                else:
                    print(f"      ⚠️ 예수금 부족 ({buy_amount:,.0f}원 < 100만원)")
            else:
                print(f"      ❌ {get_korean_name(from_coin)} 매도 실패")
    elif switch_decisions:
        print(f"\n⚠️ 갈아타기 일일 한도 초과 ({daily_switch_count}/{MAX_DAILY_SWITCHES})")
    
    # ─────────────────────────────────────────────────────────────────
    # [3-3] 3순위: 일반 매도/익절 실행
    # ─────────────────────────────────────────────────────────────────
    if sell_decisions:
        print(f"\n🟢 [3-3] 매도/익절 실행 ({len(sell_decisions)}개)")
        sell_results = execute_enhanced_signal_trades(sell_decisions, [])
        executed_trades.extend(sell_results)
        
        for dec in sell_decisions:
            remove_position_time(dec['coin'])
            print(f"   ✅ {get_korean_name(dec['coin'])} 매도 완료 ({dec['action']}, 수익률: {dec['profit_loss_pct']:+.2f}%)")
    
    # ─────────────────────────────────────────────────────────────────
    # [3-4] 4순위: 신규 매수 + 추가 매수 실행
    # ─────────────────────────────────────────────────────────────────
    new_buy_candidates = [c for c in buy_candidates if not c.get('is_additional_buy', False)]
    additional_buy_candidates = [c for c in buy_candidates if c.get('is_additional_buy', False)]
    print(f"\n🔵 [3-4] 매수 검토 (신규: {len(new_buy_candidates)}개, 추매: {len(additional_buy_candidates)}개)")
    
    try:
        # 예수금 확인
        available_balance = get_available_balance()
        print(f"   💰 예수금: {available_balance:,.0f}원")
        
        if available_balance >= 1_000_000 and buy_candidates:
            # 🆕 신규 매수: 갈아타기에서 매수한 코인 제외 (같은 사이클 중복 방지)
            # 🆕 추가 매수: 원래 보유 중인 코인이므로 executed_buy_coins 체크 불필요
            remaining_candidates = []
            for c in buy_candidates:
                if c.get('is_additional_buy', False):
                    # 추가 매수: 갈아타기로 새로 산 코인이 아니면 허용
                    if c['coin'] not in executed_buy_coins:
                        remaining_candidates.append(c)
                else:
                    # 신규 매수: 갈아타기로 이미 산 코인이면 제외
                    if c['coin'] not in executed_buy_coins:
                        remaining_candidates.append(c)
            
            if remaining_candidates:
                buy_trade_data_list = []
                buy_trade_contexts = []
                virtual_balance = available_balance
                
                for candidate in remaining_candidates:
                    coin = candidate['coin']
                    is_additional = candidate.get('is_additional_buy', False)
                    buy_type = "추매" if is_additional else "신규매수"
                    
                    # 최대 500만원, 최소 100만원
                    buy_amount = min(virtual_balance * 0.995, 5_000_000.0)
                    
                    if buy_amount < 1_000_000:
                        print(f"   ⚠️ 예수금 부족 ({buy_amount:,.0f}원 < 100만원) - 중단")
                        break
                    
                    print(f"   🟢 {get_korean_name(coin)} {buy_type} 준비 - {buy_amount:,.0f}원")
                    
                    trade_data = {
                        'coin': coin,
                        'action': 'buy',
                        'interval': 'combined',
                        'timestamp': int(time.time()),
                        'signal': 1,
                        'final_score': candidate['signal_score'],
                        'approved_by': ['AI_Enhanced_Signal'],
                        'market_flow': 'AI_Enhanced',
                        'market_mode': 'AI_Enhanced',
                        'price': buy_amount,
                        'position_percentage': None,
                        'decision_status': 'approved',
                        'confidence': candidate['confidence'],
                        'ord_type': 'price'
                    }
                    
                    buy_trade_data_list.append(trade_data)
                    buy_trade_contexts.append(candidate)
                    virtual_balance -= buy_amount
                
                # 병렬 매수 실행
                if buy_trade_data_list:
                    print(f"   🚀 {len(buy_trade_data_list)}개 매수 주문 실행")
                    execution_results = execute_trades_parallel(buy_trade_data_list)
                    
                    for i, success in enumerate(execution_results):
                        candidate = buy_trade_contexts[i]
                        coin = candidate['coin']
                        is_additional = candidate.get('is_additional_buy', False)
                        buy_type = "추매" if is_additional else "신규매수"
                        
                        if success:
                            # 거래 기록
                            trade_result = {
                                'coin': coin,
                                'action': 'buy',
                                'signal_score': candidate['signal_score'],
                                'confidence': candidate['confidence'],
                                'timestamp': int(time.time()),
                                'amount': 0.0,
                                'price': candidate['price'],
                                'profit': 0.0
                            }
                            
                            real_time_learning_feedback.record_trade_result(coin, trade_result)
                            real_time_action_tracker.record_action_result('buy', 0.0, False, 0.0, coin)
                            
                            trade_id = f"{coin}_{int(time.time())}"
                            context = {
                                'action': 'buy',
                                'signal_score': candidate['signal_score'],
                                'confidence': candidate['confidence'],
                                'regime_name': candidate.get('regime_name', 'Neutral'),
                                'thompson_score': candidate.get('thompson_score', 0.0),
                                'buy_type': buy_type
                            }
                            real_time_context_recorder.record_trade_context(trade_id, context)
                            
                            # 추매의 경우 현재 수익률 정보 포함
                            if is_additional:
                                reason_detail = f"추매 (점수: {candidate['signal_score']:.3f}, 현수익: {candidate.get('current_profit_pct', 0):+.2f}%)"
                            else:
                                reason_detail = f"신규매수 (Thompson: {candidate.get('thompson_score', 0):.2f}, 기대수익: {candidate.get('expected_profit_pct', 0):.2f}%)"
                            
                            log_trade_decision({
                                'timestamp': int(time.time()),
                                'coin': coin,
                                'interval': 'combined',
                                'action': 'buy',
                                'reason': candidate['reason'],
                                'reason_detail': reason_detail,
                                'entry_price': candidate.get('entry_price', 0),
                                'current_price': candidate['price'],
                                'profit_pct': candidate.get('current_profit_pct', 0.0),
                                'fusion_score': candidate['signal_score'],
                                'rl_score': 0.0,
                                'market_mode': candidate.get('regime_name', 'Neutral'),
                                'market_flow': 'Signal_Based',
                                'gpt_approved': 1,
                                'executed': 1,
                                'execution_price': candidate['price'],
                                'execution_amount': 0,
                                'execution_type': 'additional_buy' if is_additional else 'buy',
                                'signal_score': candidate['signal_score'],
                                'confidence': candidate['confidence']
                            })
                            
                            if 'decision_timestamp' in candidate:
                                mark_decision_processed(coin, candidate['decision_timestamp'])
                            
                            # 🆕 추가 매수의 경우 보유 시간 기록 업데이트 불필요
                            if not is_additional:
                                record_position_buy_time(coin, candidate['price'])
                            
                            executed_buy_coins.add(coin)
                            
                            print(f"   ✅ {get_korean_name(coin)} {buy_type} 완료")
                        else:
                            print(f"   ❌ {get_korean_name(coin)} {buy_type} 실패")
            else:
                print("   ℹ️ 매수 가능한 후보 없음 (이미 처리됨)")
        elif available_balance < 1_000_000:
            print("   ⚠️ 예수금 부족 (100만원 미만)")
        else:
            print("   ℹ️ 매수 후보 없음")
    
    except Exception as e:
        print(f"   ⚠️ 신규 매수 중 오류: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 🚀 [STEP 4] 결과 검증
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🚀 [STEP 4] 결과 검증")
    print("=" * 60)
    
    sync_wallet_to_db()
    updated_wallet_coins = get_filtered_wallet_coins(min_balance_krw=10000)
    final_balance = get_available_balance()
    
    print(f"💼 최종 보유 코인: {[get_korean_name(coin) for coin in updated_wallet_coins]}")
    print(f"💰 최종 예수금: {final_balance:,.0f}원")
    
    # 실행 요약
    executed_new_buys = len([c for c in buy_candidates if c['coin'] in executed_buy_coins and not c.get('is_additional_buy', False)])
    executed_additional_buys = len([c for c in buy_candidates if c['coin'] in executed_buy_coins and c.get('is_additional_buy', False)])
    
    print(f"\n📊 실행 요약:")
    print(f"   🔴 손절: {len(stop_loss_decisions)}건")
    print(f"   🔄 갈아타기: {len([s for s in switch_decisions if s.get('target')])}건")
    print(f"   🟢 매도/익절: {len(sell_decisions)}건")
    print(f"   🔵 신규매수: {executed_new_buys}건 / 추매: {executed_additional_buys}건")
    
    # 성과 추적
    for action in ['buy', 'sell', 'hold']:
        perf = real_time_action_tracker.get_action_performance(action)
        if perf['total_trades'] > 0:
            print(f"   📈 {action.upper()}: {perf['total_trades']}회, 승률: {perf['success_rate']:.1%}")
    
    print("\n✅ 실전매매 사이클 완료!")
    
    return executed_trades

def combine_signal_with_holding(coin: str, pure_action: str, signal_score: float, profit_loss_pct: float, 
                                 signal_pattern: str = 'unknown', max_profit_pct: float = None,
                                 entry_volatility: float = 0.02, holding_hours: float = 0) -> str:
    """🆕 순수 시그널과 보유 정보를 조합하여 최종 액션 결정 (학습 기반 매매 기법 적용)"""
    try:
        # max_profit_pct가 없으면 현재 수익률 사용
        if max_profit_pct is None:
            max_profit_pct = max(profit_loss_pct, 0.0)
        
        # ═══════════════════════════════════════════════════════════════
        # 🔒 [최우선] 안전장치 (절대 변경 불가 - 하드코딩)
        # ═══════════════════════════════════════════════════════════════
        if profit_loss_pct >= 50.0:
            print(f"🔒 {coin} 안전장치 익절 (+50% 도달)")
            return 'take_profit'
        
        if profit_loss_pct <= -10.0:
            print(f"🔒 {coin} 안전장치 손절 (-10% 도달)")
            return 'stop_loss'
        
        # ═══════════════════════════════════════════════════════════════
        # 🎓 [학습 기반] 청산 판단 (virtual_trade_learner에서 학습한 기법 적용)
        # ═══════════════════════════════════════════════════════════════
        if LEARNED_EXIT_AVAILABLE:
            try:
                # 🎓 학습 기반 익절 체크
                should_tp, tp_reason = should_take_profit(
                    profit_pct=profit_loss_pct,
                    max_profit_pct=max_profit_pct,
                    signal_pattern=signal_pattern,
                    entry_volatility=entry_volatility
                )
                if should_tp:
                    print(f"🎓 {coin} 학습 기반 익절 ({tp_reason})")
                    if 'trailing' in tp_reason:
                        return 'partial_sell'  # 트레일링 스탑은 부분 매도
                    return 'take_profit'
                
                # 🎓 학습 기반 손절 체크
                should_sl, sl_reason = should_stop_loss(
                    profit_pct=profit_loss_pct,
                    signal_pattern=signal_pattern,
                    entry_volatility=entry_volatility,
                    holding_hours=holding_hours
                )
                if should_sl:
                    print(f"🎓 {coin} 학습 기반 손절 ({sl_reason})")
                    return 'stop_loss'
                    
            except Exception as e:
                # 학습 기반 청산 오류 시 기본 로직으로 fallback
                print(f"⚠️ 학습 기반 청산 판단 오류: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # [기존 로직] AI 기반 매매 판단 (학습 기반 청산이 아닌 경우)
        # ═══════════════════════════════════════════════════════════════
        
        # 🆕 실전 매매 특화 의사결정 엔진 사용
        signal_data = {
            'action': pure_action,
            'signal_score': signal_score,
            'confidence': abs(signal_score),  # 신뢰도는 시그널 점수의 절댓값
            'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low'
        }
        
        # 🆕 코인별 성과 데이터 로드 (실제로는 DB에서 로드)
        coin_performance = real_time_learning_feedback.get_coin_learning_data(coin)
        
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
    """🆕 성능 업그레이드된 시그널 기반 거래 실행 (병렬 처리 적용)"""
    print(f"\n🚀 [STEP 3] 성능 업그레이드된 시그널 기반 거래 실행")
    print(f"🔴 매도 대상: {len(sell_decisions)}개")
    print(f"🟡 홀딩 대상: {len(hold_decisions)}개")
    
    executed_trades = []
    total_profit = 0.0
    
    sell_trade_data_list = []
    sell_trade_contexts = []

    # 🆕 매도 거래 준비 (성능 업그레이드 적용)
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
            
            # 🆕 진짜 시장 상황 분석 (Core 모듈 연동)
            real_market_context = get_market_context()
            
            market_context = {
                'trend': real_market_context.get('trend', 'neutral'),
                'volatility': 'high' if real_market_context.get('volatility', 0.02) > 0.05 else 'medium',
                'timestamp': int(time.time())
            }
            
            ai_decision = real_time_ai_decision_engine.make_trading_decision(
                signal_data, current_price, market_context, coin_performance
            )
            
            # 🔒 [핵심 수정] 손절(stop_loss)은 AI 의사결정 무시하고 무조건 실행!
            is_stop_loss = decision['action'] == 'stop_loss'
            is_forced_sell = decision['action'] in ['stop_loss', 'take_profit']  # 익절도 강제
            
            # 🆕 AI가 매도를 승인하거나, 손절/익절이면 무조건 실행
            if ai_decision == 'sell' or decision['action'] == 'partial_sell' or is_forced_sell:
                if is_stop_loss:
                    print(f"🔒 {get_korean_name(coin)}: 손절 강제 실행! (AI 의사결정 무시)")
                elif is_forced_sell:
                    print(f"🔒 {get_korean_name(coin)}: 익절 강제 실행!")
                else:
                    print(f"✅ {get_korean_name(coin)}: AI 승인 매도 준비 - {decision['reason']}")
                
                # 🎯 분할 매도 로직 적용 (부분 익절 시 50% 매도)
                if decision['action'] == 'partial_sell':
                    position_pct = 0.5
                    reason_detail = f"부분 익절 (수익률 {profit_loss_pct:.2f}%)"
                else:
                    position_pct = 1.0
                    reason_detail = f"전량 매도 (수익률 {profit_loss_pct:.2f}%)"
                
                # 실제 거래 데이터 생성
                trade_data = {
                    'coin': coin,
                    'interval': 'combined',
                    'timestamp': int(time.time()),
                    'signal': -1,
                    'final_score': signal_score,
                    'approved_by': ['AI_Enhanced_Signal'],
                    'market_flow': 'AI_Enhanced',
                    'market_mode': 'AI_Enhanced',
                    'ord_type': 'market',  # 🔧 시장가 매도 (지정가 체결 실패 방지)
                    'position_percentage': position_pct,
                    'profit_pct': round(profit_loss_pct, 2),
                    'confidence': confidence
                }
                
                sell_trade_data_list.append(trade_data)
                
                # 컨텍스트 저장을 위한 데이터 보관
                sell_trade_contexts.append({
                    'coin': coin,
                    'signal_score': signal_score,
                    'confidence': confidence,
                    'current_price': current_price,
                    'profit_loss_pct': profit_loss_pct,
                    'market_context': market_context,
                    'coin_performance': coin_performance,
                    'reason_detail': reason_detail
                })

            else:
                print(f"⏭️ {get_korean_name(coin)}: AI가 매도 거부 - 홀딩 유지")
                
        except Exception as e:
            print(f"⚠️ {decision.get('coin', 'unknown')} 매도 준비 중 오류: {e}")
            continue
    
    # 🚀 매도 주문 병렬 실행
    if sell_trade_data_list:
        print(f"🚀 총 {len(sell_trade_data_list)}개 매도 주문 일괄 전송 및 병렬 처리 시작")
        execution_results = execute_trades_parallel(sell_trade_data_list)
        
        # 결과 처리
        for i, success in enumerate(execution_results):
            if success:
                ctx = sell_trade_contexts[i]
                coin = ctx['coin']
                profit_loss_pct = ctx['profit_loss_pct']
                
                # 🆕 거래 결과 기록
                trade_result = {
                    'coin': coin,
                    'action': 'sell',
                    'signal_score': ctx['signal_score'],
                    'confidence': ctx['confidence'],
                    'timestamp': int(time.time()),
                    'amount': 0.0,  # 실제 거래 후 업데이트
                    'price': ctx['current_price'],
                    'profit': profit_loss_pct
                }
                
                executed_trades.append(trade_result)
                
                # 🆕 학습 피드백에 거래 결과 기록
                real_time_learning_feedback.record_trade_result(coin, trade_result)
                
                # 🆕 액션별 성과 추적
                success_trade = profit_loss_pct > 0
                real_time_action_tracker.record_action_result('sell', profit_loss_pct, success_trade, 0.0, coin)
                
                # 🆕 컨텍스트 기록
                trade_id = f"{coin}_{int(time.time())}"
                context = {
                    'action': 'sell',
                    'signal_score': ctx['signal_score'],
                    'confidence': ctx['confidence'],
                    'market_context': ctx['market_context'],
                    'coin_performance': ctx['coin_performance'],
                    'profit_loss_pct': profit_loss_pct
                }
                real_time_context_recorder.record_trade_context(trade_id, context)

                # 🆕 [복구] DB에 매매 결정 기록 (real_trade_history)
                log_trade_decision({
                    'timestamp': int(time.time()),
                    'coin': coin,
                    'interval': 'combined',
                    'action': 'sell',
                    'reason': 'signal_based_sell',
                    'reason_detail': ctx.get('reason_detail', f"AI 승인 매도 (수익률: {profit_loss_pct:.2f}%)"),
                    'entry_price': 0, # 매도 시 진입가 조회 필요하면 추가
                    'current_price': ctx['current_price'],
                    'profit_pct': profit_loss_pct,
                    'fusion_score': ctx['signal_score'],
                    'rl_score': 0.0,
                    'market_mode': 'AI_Enhanced',
                    'market_flow': 'AI_Enhanced',
                    'gpt_approved': 1,
                    'executed': 1,
                    'execution_price': ctx['current_price'], # 추정치
                    'execution_amount': 0.0,
                    'execution_type': 'sell',
                    'signal_score': ctx['signal_score'],
                    'confidence': ctx['confidence']
                })
                
                total_profit += profit_loss_pct
                
                # 🆕 보유 시간 기록 삭제 (매도 성공 시)
                remove_position_time(coin)
                
                print(f"✅ {get_korean_name(coin)} 매도 처리 완료 (수익률: {profit_loss_pct:.2f}%)")
            else:
                coin = sell_trade_data_list[i]['coin']
                print(f"❌ {get_korean_name(coin)} 매도 실패 (타임아웃 또는 API 오류)")

    
    return executed_trades


if __name__ == "__main__":
    create_holdings_table()
    create_trade_decision_log_table()
    
    print("🚀 실전매매 실행기 시작")
    print("=" * 60)
    
    # 매매 실행
    run_signal_based_executor()
    
    print("\n✅ 실전매매 실행 완료!")