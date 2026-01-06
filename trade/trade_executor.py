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
import math
import numpy as np
import pandas as pd
import time
import json
import os
import sys
import logging
import traceback
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# 🆕 공통 코어 모듈 임포트 (가상매매와 동일하게 정합성 유지)
from trade.core.learner.connector import SignalTradeConnector
from trade.core.learner.analyzer import PatternAnalyzer
from trade.core.learner.realtime import RealTimeLearner
from trade.core.thompson import get_thompson_calculator, ThompsonSamplingLearner, OutlierGuardrail, BayesianSmoothing
from trade.core.executor.strategy import decide_final_action, get_dynamic_weights, get_learning_maturity
from trade.core.trading import (
    get_market_context as get_common_market_context,
    calculate_buy_thresholds, BuyThresholds,
    normalize_regime, get_regime_severity, get_regime_trading_strategy,
    should_execute_buy, calculate_combined_score, VALID_REGIMES
)
from trade.core.decision import get_ai_decision_engine
from trade.core.models import SignalInfo, SignalAction
from trade.core.sequence_analyzer import SequenceAnalyzer
from trade.core.thresholds import (
    get_thresholds, get_buy_threshold, get_sell_threshold,
    get_priority_level, get_stop_loss_adjustment, is_buy_signal
)

# 🆕 전략 시스템 임포트
try:
    from trade.core.strategies import (
        evaluate_all_strategies, select_best_strategies, get_top_strategies,
        get_exit_rules, get_strategy_description, update_strategy_feedback,
        get_strategy_success_rate, create_strategy_feedback_table,
        STRATEGY_EXIT_RULES, STRATEGY_ENTRY_THRESHOLDS, StrategyType,
        serialize_strategy_scores, deserialize_strategy_scores,
        get_regime_adjustment, get_sideways_policy  # 🆕 레짐 조정 함수
    )
    STRATEGY_SYSTEM_AVAILABLE = True
    print("✅ 전략 시스템 로드 완료 (10가지 매매 전략)")
except ImportError as e:
    STRATEGY_SYSTEM_AVAILABLE = False
    print(f"⚠️ 전략 시스템 로드 실패: {e}")

# 🧬 전략 진화 시스템 임포트 (가상매매와 동일)
try:
    from trade.core.strategy_evolution import (
        get_evolution_manager, update_evolution_stats, get_strategy_level,
        get_best_evolved_strategy, EvolutionLevel
    )
    EVOLUTION_SYSTEM_AVAILABLE = True
    print("✅ 전략 진화 시스템 로드 완료 (4단계 진화)")
except ImportError as e:
    EVOLUTION_SYSTEM_AVAILABLE = False
    print(f"⚠️ 전략 진화 시스템 로드 실패: {e}")

# 🔧 [경로 수정] trade_manager는 trade 패키지 내에 있음
try:
    from trade.trade_manager import sync_wallet_to_db, get_filtered_wallet_coins, execute_trade_with_timeout, get_order_chance, wait_for_balance_update, fetch_tick_size_from_bithumb, execute_trades_parallel, get_available_balance, print_trade_summary_24h
except ImportError:
    # 하위 호환성 (trade 폴더가 path에 있는 경우)
    from trade_manager import sync_wallet_to_db, get_filtered_wallet_coins, execute_trade_with_timeout, get_order_chance, wait_for_balance_update, fetch_tick_size_from_bithumb, execute_trades_parallel, get_available_balance, print_trade_summary_24h

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
    from trade.core.thompson import ThompsonSamplingLearner
    # 🆕 전역 인스턴스 생성 (DB 연결 재사용)
    _thompson_learner_instance = None
    def get_thompson_learner():
        global _thompson_learner_instance
        if _thompson_learner_instance is None:
            # 실전 매매 DB 경로 사용 (core.database에서 가져옴)
            from trade.core.database import STRATEGY_DB_PATH
            _thompson_learner_instance = ThompsonSamplingLearner(db_path=STRATEGY_DB_PATH)
        return _thompson_learner_instance
except ImportError:
    print("⚠️ ThompsonSamplingLearner 로드 실패")
    ThompsonSamplingLearner = None
    def get_thompson_learner():
        return None

# 🆕 학습된 청산 파라미터 모듈 (가상매매와 동일한 매매 기법 적용)
try:
    from trade.core.exit_params import should_take_profit, should_stop_loss, get_exit_params, get_learned_sell_threshold
    LEARNED_EXIT_AVAILABLE = True
except ImportError:
    LEARNED_EXIT_AVAILABLE = False
    print("⚠️ 학습된 청산 파라미터 모듈 로드 실패 - 기본 청산 로직 사용")
    def get_learned_sell_threshold(*args, **kwargs):
        return None

# 🆕 Trajectory Analyzer - 수익률 추적 및 추세 분석
try:
    from trade.core.trajectory_analyzer import get_real_trajectory_analyzer, TrendType
    TRAJECTORY_ANALYZER_AVAILABLE = True
except ImportError:
    TRAJECTORY_ANALYZER_AVAILABLE = False
    print("⚠️ Trajectory Analyzer 로드 실패 - 추세 분석 비활성화")

# DB 경로 설정 (trade.core.database에서 중앙화된 설정 로드)
try:
    from trade.core.database import TRADING_SYSTEM_DB_PATH, STRATEGY_DB_PATH, CANDLES_DB_PATH
    DB_PATH = CANDLES_DB_PATH
except ImportError:
    # 하위 호환성 및 대체 로직
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT, 'market', 'coin_market', 'data_storage')
    TRADING_SYSTEM_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'trading_system.db')
    STRATEGY_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'learning_strategies', 'common_strategies.db')
    DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'trade_candles.db')


def load_virtual_trade_decisions(max_age_minutes: int = 30, reference_ts: int = None) -> Dict[str, Dict]:
    """🆕🆕 가상매매 결정 테이블에서 최신 결정 읽기 (DB 최신 시각 기준)"""
    try:
        from trade.core.database import get_db_connection
        # 🚀 [Fix] 기준 시각 설정 (없으면 현재 시스템 시각)
        now = reference_ts if reference_ts else int(time.time())
        cutoff_time = now - (max_age_minutes * 60)
        
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=True) as conn:
            # 🆕 컬럼 존재 여부 확인 후 쿼리 생성
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(virtual_trade_decisions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            select_fields = [
                "coin", "timestamp", "decision", "signal_score", "confidence", "current_price",
                "target_price", "expected_profit_pct", "thompson_score", "thompson_approved",
                "regime_score", "regime_name", "viability_passed", "reason",
                "is_holding", "entry_price", "profit_loss_pct"
            ]
            
            if 'wave_phase' in columns:
                select_fields.append("wave_phase")
            if 'integrated_direction' in columns:
                select_fields.append("integrated_direction")
                
            query = f"""
                SELECT {', '.join(select_fields)}
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
                        'profit_loss_pct': row['profit_loss_pct'],
                        'wave_phase': row.get('wave_phase', 'unknown'),
                        'integrated_direction': row.get('integrated_direction', 'neutral')
                    }
            
            return decisions
            
    except Exception as e:
        print(f"⚠️ 가상매매 결정 로드 오류: {e}")
        return {}


def mark_decision_processed(coin: str, timestamp: int):
    """🆕 가상매매 결정을 처리 완료로 표시 (쓰기 모드 안정성 강화)"""
    try:
        from trade.core.database import get_db_connection
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
            conn.execute("""
                UPDATE virtual_trade_decisions
                SET processed = 1
                WHERE coin = ? AND timestamp = ?
            """, (coin, timestamp))
            conn.commit()
    except Exception as e:
        print(f"⚠️ 결정 처리 완료 표시 오류: {e}")

# 🆕 실전 매매용 정밀 분석 및 학습 로직 통합 (trade.core 활용)
# 가상매매에서 검증된 고정밀 학습 로직이 실전에도 동일하게 적용됩니다.

# 🆕 실전 매매 성능 업그레이드 시스템 초기화 (코어 모듈 연동)
# 가상매매와 동일한 정밀 분석 도구 사용
pattern_analyzer = PatternAnalyzer()
thompson_sampler = get_thompson_learner()
real_time_learner = RealTimeLearner(thompson_sampler)

# 🛡️ 알파 가디언 활성화/비활성화 설정
ENABLE_ALPHA_GUARDIAN = os.getenv('ENABLE_ALPHA_GUARDIAN', 'true').lower() == 'true'

# 글로벌 AI 엔진 인스턴스 (공통 모듈 연동)
if ENABLE_ALPHA_GUARDIAN:
    real_time_ai_decision_engine = get_ai_decision_engine()
    print("🛡️ 알파 가디언 활성화됨 (실전 매매)")
else:
    real_time_ai_decision_engine = None
    print("ℹ️ 알파 가디언 비활성화됨 (실전 매매, ENABLE_ALPHA_GUARDIAN=false)")

# 로깅 설정 (파일 생성 없이 콘솔만)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 시그널 기반 거래 결정 내역 테이블 생성 (최초 1회 실행 시 생성)
def create_signal_trade_table():
    try:
        from trade.core.database import get_db_connection
        with get_db_connection(DB_PATH, read_only=False) as conn:
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
            conn.commit()
    except Exception as e:
        print(f"⚠️ 시그널 매매 내역 테이블 생성 오류: {e}")

def create_trade_decision_log_table():
    # 🚀 trading_system.db에 실전 매매 테이블 생성 (통합 DB 사용, 쓰기 모드 안정성 강화)
    try:
        from trade.core.database import get_db_connection
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
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
            conn.commit()
    except Exception as e:
        print(f"⚠️ 실전 매매 테이블 생성 오류: {e}")

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
def format_price(price: float, return_float: bool = False) -> Any:
    """🆕 빗썸 KRW 마켓 호가 단위(Tick Size) 규정을 준수한 포맷팅 및 반올림
    
    - 1원 미만: 소수점 4자리 (0.0001 단위)
    - 1원 이상 ~ 10원 미만: 소수점 3자리 (0.001 단위)
    - 10원 이상 ~ 100원 미만: 소수점 2자리 (0.01 단위)
    - 100원 이상 ~ 1,000원 미만: 소수점 1자리 (0.1 단위)
    - 1,000원 이상: 소수점 없음 (1원 단위)
    """
    if price is None or price <= 0: return 0.0 if return_float else "0"
    
    if price < 1:
        # 1원 미만: 0.0001 단위
        rounded = round(price, 4)
        return rounded if return_float else f"{rounded:.4f}"
    elif price < 10:
        # 1원 ~ 10원: 0.001 단위
        rounded = round(price, 3)
        return rounded if return_float else f"{rounded:.3f}"
    elif price < 100:
        # 10원 ~ 100원: 0.01 단위
        rounded = round(price, 2)
        return rounded if return_float else f"{rounded:.2f}"
    elif price < 1000:
        # 100원 ~ 1,000원: 0.1 단위
        rounded = round(price, 1)
        return rounded if return_float else f"{rounded:.1f}"
    else:
        # 1,000원 이상: 1원 단위 (고가 코인은 5원/10원 단위이나 소수점 제거가 핵심)
        rounded = float(int(round(price, 0)))
        return rounded if return_float else f"{int(rounded):,}"

def round_to_tick(price: float) -> float:
    """가격을 빗썸 호가 단위로 반올림하여 float로 반환"""
    return format_price(price, return_float=True)

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
    """🆕 설계 반영: 수익 중 추가 매수(피라미딩)의 자율 판단"""
    if entry_price is None or entry_price <= 0: return True # 신규 매수 허용
    
    profit_pct = ((current_price - entry_price) / entry_price) * 100
    
    # ❌ 손실 중 물타기 금지 (원칙 유지)
    if profit_pct < 0: return False
    
    # 🎯 시장 상황 및 알파 가디언 성향 연동
    market_context = get_market_context()
    buy_bias = 0.0
    try:
        from trade.core.decision import get_ai_decision_engine
        guardian = get_ai_decision_engine()
        buy_bias = guardian.get_meta_bias().get('buy_threshold_offset', 0.0)
    except: pass

    # 기본 추매 문턱 (0.15)을 알파 가디언 성향으로 보정
    min_add_score = 0.15 + buy_bias
    
    # 불장(Bullish)일수록 더 낮은 수익권에서도 적극적으로 피라미딩
    min_profit_threshold = 1.0 if market_context['trend'] == 'bullish' else 2.5

    if profit_pct >= min_profit_threshold and signal_score >= min_add_score and confidence >= 0.65:
        return True
    
    return False


# 🆕🆕 보유 시간 관리 함수들 (current_position_times 테이블)
def record_position_buy_time(coin: str, entry_price: float = 0.0, 
                            entry_strategy: str = 'trend', strategy_match: float = 0.5,
                            evolution_level: int = 1, evolved_params: str = ''):
    """매수 시 보유 시간 및 전략 정보 기록 (진화 레벨 포함)"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            from trade.core.database import get_latest_candle_timestamp
            db_now = get_latest_candle_timestamp()
            
            # 🆕 전략 + 진화 관련 컬럼 마이그레이션
            cursor = conn.execute("PRAGMA table_info(current_position_times)")
            cols = [c[1] for c in cursor.fetchall()]
            strategy_cols = {
                'entry_strategy': "TEXT DEFAULT 'trend'",
                'current_strategy': "TEXT DEFAULT 'trend'",
                'strategy_match': "REAL DEFAULT 0.5",
                'strategy_switch_count': "INTEGER DEFAULT 0",
                'strategy_switch_history': "TEXT DEFAULT ''",
                # 🧬 진화 시스템 필드
                'evolution_level': "INTEGER DEFAULT 1",
                'evolved_params': "TEXT DEFAULT ''"
            }
            for col, col_type in strategy_cols.items():
                if col not in cols:
                    try:
                        conn.execute(f"ALTER TABLE current_position_times ADD COLUMN {col} {col_type}")
                    except:
                        pass
            
            conn.execute("""
                INSERT OR REPLACE INTO current_position_times 
                (coin, buy_timestamp, entry_price, entry_strategy, current_strategy, 
                 strategy_match, strategy_switch_count, evolution_level, evolved_params)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (coin, db_now, entry_price, entry_strategy, entry_strategy, 
                  strategy_match, evolution_level, evolved_params))
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


def get_position_strategy_info(coin: str) -> dict:
    """포지션의 전략 정보 조회 (진화 레벨 포함)"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT entry_strategy, current_strategy, strategy_match, 
                       strategy_switch_count, strategy_switch_history,
                       evolution_level, evolved_params
                FROM current_position_times WHERE coin = ?
            """, (coin,))
            row = cursor.fetchone()
            if row:
                return {
                    'entry_strategy': row[0] or 'trend',
                    'current_strategy': row[1] or row[0] or 'trend',
                    'strategy_match': row[2] or 0.5,
                    'strategy_switch_count': row[3] or 0,
                    'strategy_switch_history': row[4] or '',
                    # 🧬 진화 시스템 정보
                    'evolution_level': row[5] if len(row) > 5 and row[5] else 1,
                    'evolved_params': row[6] if len(row) > 6 and row[6] else ''
                }
    except:
        pass
    return {'entry_strategy': 'trend', 'current_strategy': 'trend', 
            'strategy_match': 0.5, 'strategy_switch_count': 0, 'strategy_switch_history': '',
            'evolution_level': 1, 'evolved_params': ''}


def check_strategy_switch_real(coin: str, profit_pct: float, holding_hours: float) -> tuple:
    """
    🆕 실제 매매용 전략 전환 확인
    
    Returns:
        (should_switch, new_strategy, reason)
    """
    import json
    
    strategy_info = get_position_strategy_info(coin)
    current_strat = strategy_info['current_strategy']
    switch_count = strategy_info['strategy_switch_count']
    
    # 전환 횟수 제한 (최대 2회 - 실제 매매는 더 보수적)
    if switch_count >= 2:
        return False, None, None
    
    new_strategy = None
    reason = None
    
    # 스캘핑 → 스윙/추세
    if current_strat == 'scalp':
        if holding_hours > 4.0 and profit_pct >= 0:
            new_strategy = 'swing'
            reason = f"스캘핑 시간 초과 ({holding_hours:.1f}h), 스윙 전환"
        elif holding_hours > 4.0 and profit_pct >= 3.0:
            new_strategy = 'trend'
            reason = f"수익 중 시간 초과 (+{profit_pct:.1f}%), 추세 전환"
    
    # 저점 매수 → 추세
    elif current_strat == 'bottom':
        if profit_pct >= 10.0:
            new_strategy = 'trend'
            reason = f"저점 반등 확인 (+{profit_pct:.1f}%), 추세 전환"
    
    # 스윙 → 추세
    elif current_strat == 'swing':
        if profit_pct >= 20.0:
            new_strategy = 'trend'
            reason = f"파동 연장 (+{profit_pct:.1f}%), 추세 전환"
    
    if new_strategy:
        # DB 업데이트
        try:
            history = json.loads(strategy_info['strategy_switch_history']) if strategy_info['strategy_switch_history'] else []
            history.append({
                'from': current_strat, 'to': new_strategy, 
                'reason': reason, 'profit_at_switch': profit_pct,
                'ts': int(time.time())
            })
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    UPDATE current_position_times SET
                        current_strategy = ?, 
                        strategy_switch_count = strategy_switch_count + 1,
                        strategy_switch_history = ?
                    WHERE coin = ?
                """, (new_strategy, json.dumps(history), coin))
                conn.commit()
            
            print(f"   🔄 {coin}: 전략 전환! [{current_strat.upper()}] → [{new_strategy.upper()}]")
            print(f"      📋 이유: {reason}")
            
        except Exception as e:
            logging.warning(f"전략 전환 DB 업데이트 오류: {e}")
        
        return True, new_strategy, reason
    
    return False, None, None


def record_strategy_feedback_real(coin: str, profit_pct: float, success: bool, holding_hours: float):
    """🆕 실제 매매 완료 시 전략 분리 학습"""
    try:
        from trade.core.strategies import update_strategy_feedback
        from trade.core.database import STRATEGY_DB_PATH
        
        strategy_info = get_position_strategy_info(coin)
        entry_strategy = strategy_info['entry_strategy']
        exit_strategy = strategy_info['current_strategy']
        switch_count = strategy_info['strategy_switch_count']
        
        pattern = f"{coin}_real_trade"
        
        # 1️⃣ 진입 전략 학습
        update_strategy_feedback(
            db_path=STRATEGY_DB_PATH,
            strategy_type=entry_strategy,
            market_condition='real_trade',
            signal_pattern=pattern,
            success=success,
            profit_pct=profit_pct,
            holding_hours=holding_hours,
            feedback_type='entry'
        )
        
        # 2️⃣ 청산 전략 학습 (전환된 경우)
        if switch_count > 0 and exit_strategy != entry_strategy:
            update_strategy_feedback(
                db_path=STRATEGY_DB_PATH,
                strategy_type=exit_strategy,
                market_condition='real_trade',
                signal_pattern=pattern,
                success=success,
                profit_pct=profit_pct,
                holding_hours=holding_hours,
                feedback_type='exit'
            )
            
            # 3️⃣ 전환 성공률 학습
            switch_key = f"{entry_strategy}_to_{exit_strategy}"
            update_strategy_feedback(
                db_path=STRATEGY_DB_PATH,
                strategy_type=switch_key,
                market_condition='real_trade',
                signal_pattern=pattern,
                success=success,
                profit_pct=profit_pct,
                holding_hours=holding_hours,
                feedback_type='switch'
            )
            
            print(f"   📚 [{entry_strategy}→{exit_strategy}] 전략 전환 학습: {'✅' if success else '❌'} ({profit_pct:+.2f}%)")
        else:
            print(f"   📚 [{entry_strategy}] 전략 학습: {'✅' if success else '❌'} ({profit_pct:+.2f}%)")
            
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"전략 피드백 기록 오류: {e}")


def get_holding_duration(coin: str) -> int:
    """코인의 보유 시간(초) 조회"""
    try:
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            cursor = conn.cursor()
            from trade.core.database import get_latest_candle_timestamp
            current_time = get_latest_candle_timestamp()
            
            # 1. 실전매매 보유 시간 테이블에서 조회
            cursor.execute("""
                SELECT buy_timestamp FROM current_position_times 
                WHERE coin = ?
            """, (coin,))
            
            row = cursor.fetchone()
            if row and row[0]:
                return max(0, current_time - row[0])
            
            # 2. 가상매매 포지션에서 조회 (fallback)
            cursor.execute("""
                SELECT entry_timestamp FROM virtual_positions 
                WHERE coin = ?
            """, (coin,))
            
            row = cursor.fetchone()
            if row and row[0]:
                return max(0, current_time - row[0])
            
            return 24 * 3600  # 기본값 24시간
        
    except Exception as e:
        logging.warning(f"보유 시간 조회 오류 ({coin}): {e}")
        return 24 * 3600


# 🆕🆕 갈아타기 조건 체크 함수 (지능형 보완 + 전략별 횡보 정책 + 레짐 반영)
def check_switch_condition(coin: str, profit_pct: float, holding_hours: float, 
                           target_price: float = 0, current_price: float = 0,
                           market_score: float = 0.5, trend_analysis = None,
                           strategy_type: str = None, market_regime: str = None) -> tuple:
    """🆕 설계 반영: 시장 상황에 따른 자율적 종목 교체 판단 (전략별 횡보 정책 + 레짐 적용)"""
    
    # 🆕 전략별 횡보 정책 로드
    try:
        from trade.core.strategies import get_sideways_policy, should_exempt_from_sideways_switch
        strategy_policy_available = True
    except ImportError:
        strategy_policy_available = False
    
    # 🆕 전략별 횡보 갈아타기 면제 체크 (레짐 반영)
    if strategy_policy_available and strategy_type:
        # 🆕 전략+레짐 호환성 체크
        if market_regime:
            from trade.core.strategies import get_strategy_regime_compatibility
            compatibility, compat_desc = get_strategy_regime_compatibility(strategy_type, market_regime)
            # 호환성 매우 낮으면 (< 0.5) 면제 전략이라도 교체 고려
            if compatibility < 0.5:
                return True, f"{compat_desc} - 전략 부적합", "strategy_regime_mismatch"
        
        if should_exempt_from_sideways_switch(strategy_type):
            # 면제 전략이라도 최대 보유 시간은 체크
            from trade.core.strategies import STRATEGY_EXIT_RULES
            exit_rules = STRATEGY_EXIT_RULES.get(strategy_type)
            if exit_rules and holding_hours >= exit_rules.max_holding_hours:
                return True, f"전략({strategy_type}) 최대 보유 시간 초과 ({holding_hours:.0f}h/{exit_rules.max_holding_hours}h)", "strategy_max_holding"
            # 횡보 체크 스킵 (but 손실 장기화는 체크)
            if profit_pct <= -8.0:  # 심각한 손실은 전략 무관 청산
                return True, f"전략({strategy_type}) 손실 한도 초과 ({profit_pct:.1f}%)", "strategy_stop_loss"
            return False, "", ""
    
    # 🎯 시장 상황에 따른 '인내심(Patience)' 동적 계산
    # 시장 점수가 높을수록(1.0에 가까울수록) 인내심을 낮춰 빠르게 주도주로 교체
    # 시장 점수 0.8+ (강한 불장) -> 4시간만 횡보해도 교체
    # 시장 점수 0.5  (중립)     -> 12시간 횡보 시 교체
    # 시장 점수 0.2- (하락장)   -> 24시간까지 견딤
    
    patience_hours = 24.0 * (1.1 - market_score) # 0.5일 때 약 14시간, 0.8일 때 약 7시간
    
    # 🆕 전략별 patience 배율 적용 (레짐 반영)
    if strategy_policy_available and strategy_type:
        from trade.core.strategies import get_patience_multiplier
        patience_multiplier = get_patience_multiplier(strategy_type, regime=market_regime)
        patience_hours *= patience_multiplier
    
    # 🎯 전문가 지능 반영: 중장기 전문가(240m_mid)의 신뢰도가 높으면 인내심 2배 강화
    # 이 종목이 결국 갈 것이라는 '전문가적 확신'이 있다면 횡보를 더 견딥니다.
    expert_reliability = 0.5
    try:
        # 🆕 설계 반영: SignalSelector 엔진을 로드하는 대신 DB에서 직접 신뢰도 조회
        with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(is_correct) FROM prediction_events 
                WHERE coin = ? AND type = '240m_mid' AND status = 'completed'
                ORDER BY expire_timestamp DESC LIMIT 30
            """, (coin,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                expert_reliability = float(row[0])
    except Exception as e:
        logging.debug(f"신뢰도 직접 조회 실패: {e}")
        expert_reliability = 0.5
        
    if expert_reliability >= 0.65:
        patience_hours *= 2.0
        # print(f"🛡️ {coin}: 전문가 확신 감지 (신뢰도 {expert_reliability:.2f}) -> 인내심 {patience_hours:.1f}h 확장")

    # 🆕 전략별 최대 patience 확장 (기존 48시간 -> 전략에 따라 최대 336시간)
    max_patience = 48.0
    if strategy_policy_available and strategy_type:
        from trade.core.strategies import STRATEGY_EXIT_RULES
        exit_rules = STRATEGY_EXIT_RULES.get(strategy_type)
        if exit_rules:
            max_patience = min(exit_rules.max_holding_hours, 336.0)  # 전략 최대 보유 시간까지
    
    patience_hours = max(4.0, min(max_patience, patience_hours))

    # 1. 횡보 감지 (자율 인내심 적용)
    if holding_hours >= patience_hours and -1.5 <= profit_pct <= 1.5:
        return True, f"시장 상황 대비 정체 ({holding_hours:.1f}h/{patience_hours:.1f}h, {profit_pct:+.2f}%)", "relative_weakness"
    
    # 2. 상대적 약세 감지 (시장 주도주 소외)
    # 시장은 달리고 있는데 내 코인만 멈춰있을 때 (기준 시간도 시장 상황에 연동)
    outcast_threshold = patience_hours / 2.0
    if market_score > 0.7 and profit_pct < 0.5 and holding_hours >= outcast_threshold:
        return True, f"주도주 소외 감지 ({holding_hours:.1f}h/{outcast_threshold:.1f}h)", "market_outcast"

    # 3. 추세 피로도 분석 (기존 유지)
    if trend_analysis and trend_analysis.history_count >= 5:
        if trend_analysis.should_sell_early and profit_pct > 0.5:
            return True, f"상승 에너지 고갈 (추세 피로)", "trajectory_fatigue"
    
    # 4. 손실 장기화 및 목표 미달 (시장이 좋을수록 더 엄격하게)
    if holding_hours >= patience_hours * 2.0:
        if profit_pct <= -3.0:
            return True, f"손실 장기화 방어", "stagnant_loss"
        if target_price > 0 and current_price > 0:
            target_distance_pct = ((target_price - current_price) / current_price) * 100
            if target_distance_pct > 2.0:
                return True, f"목표 달성 지연", "target_miss"
    
    return False, "", ""


def find_best_switch_target(virtual_decisions: dict, wallet_coins: list, 
                            current_coin: str, min_signal_score: float = 0.2,
                            top_volume_coins: list = None) -> dict:
    """🆕 설계 반영: 종목 교체 시 임계값 완화 (0.3 -> 0.2)"""
    """갈아타기 대상 코인 찾기 (학습 결과 중심)"""
    best_candidate = None
    best_score = 0
    
    if top_volume_coins is not None:
        top_volume_set = set(top_volume_coins)
    
    for coin, decision in virtual_decisions.items():
        if top_volume_coins is not None and coin not in top_volume_set:
            continue
            
        if decision['decision'] != 'buy':
            continue
        
        signal_score = decision['signal_score']
        if signal_score < min_signal_score:
            continue
        
        if coin in wallet_coins or coin == current_coin:
            continue
        
        thompson_score = decision.get('thompson_score', 0)
        t = get_thresholds()
        if thompson_score < t.thompson_min:
            continue
        
        if signal_score > best_score:
            best_candidate = {
                'coin': coin,
                'signal_score': signal_score,
                'expected_profit_pct': decision.get('expected_profit_pct', 0),
                'thompson_score': thompson_score,
                'current_price': decision.get('current_price', 0),
                'reason': f"시그널 {signal_score:.3f}, Thompson {thompson_score:.2f}",
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
    """signals 테이블에서 코인의 최신 통합 시그널 정보 로드 (combined 시그널만 사용, 읽기 전용 강화)"""
    try:
        from trade.core.database import get_db_connection
        # 🚀 trading_system.db 사용
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=True) as conn:
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
                    'integrated_direction': 'neutral',
                    # 🆕 동적 영향도 정보 추가 (기본값)
                    'signal_continuity': 0.5,
                    'dynamic_influence': 0.5
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
        
        # 🆕 틱 사이즈 정보 로드
        from trade.trade_manager import get_bithumb_tick_size
        current_price = row['current_price']
        tick_size = get_bithumb_tick_size(current_price)

        return {
            'signal_info': {
                'action': row['action'],
                'signal_score': row['signal_score'],
                'confidence': row['confidence'],
                'reason': row['reason']
            },
            'market_data': {
                'price': current_price,
                'tick_size': tick_size, # 🆕 틱 사이즈 정보 추가
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
                'integrated_direction': row['integrated_direction'],
                # 🆕 동적 영향도 정보 추가
                'signal_continuity': row.get('signal_continuity', 0.5),
                'dynamic_influence': row.get('dynamic_influence', 0.5)
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

def _get_recent_candles(coin, interval, count=5):
    """🆕 DB에서 최근 N개의 캔들 데이터 로드 (Sequence 분석용)"""
    try:
        from trade.core.database import CANDLES_DB_PATH, get_db_connection
        with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
            query = """
                SELECT timestamp, open, high, low, close, volume, rsi
                FROM candles 
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            df = pd.read_sql(query, conn, params=(coin, interval, count))
            return df if not df.empty else None
    except Exception as e:
        # coin -> symbol 마이그레이션 대응
        try:
            from trade.core.database import CANDLES_DB_PATH, get_db_connection
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume, rsi
                    FROM candles 
                    WHERE coin = ? AND interval = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(coin, interval, count))
                return df if not df.empty else None
        except:
            print(f"⚠️ {coin}/{interval} 최근 캔들 로드 실패: {e}")
    return None

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
    """🆕 설계 반영: 자율 손절 외에 실전 매매 최후의 보루(Hard Rule) 적용"""
    if entry_price and entry_price > 0:
        profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 🚨 [Hard Rule] 무조건 익절 +50%
        if profit_loss_pct >= 50.0:
            return True, f"🚀 강제 익절 달성 (+{profit_loss_pct:.1f}%)"
            
        # 🚨 [Hard Rule] 무조건 손절 -10%
        if profit_loss_pct <= -10.0:
            return True, f"😭 강제 손절 집행 ({profit_loss_pct:.1f}%)"

    final_strength = calculate_stop_loss_strength(current_price, entry_price, latest_candle, params, interval)

    if final_strength >= stop_loss_threshold:
        return True, f'융합 손절 (강도 {final_strength:.2f}%)'

    return False, None

# 🆕 Absolute Zero System 개선사항을 반영한 시그널 기반 매매 결정 함수
def make_signal_based_decision(signal_data):
    """🆕 [로직 동기화] 가상 매매(VirtualTrader)와 100% 동일한 판단 로직 적용"""
    try:
        from trade.core.decision import get_ai_decision_engine
        from trade.core.thompson import get_thompson_calculator
        
        # 🎯 1. 실전용 독자 필터 대신 통합 엔진 로직 사용
        guardian = get_ai_decision_engine()
        thompson = get_thompson_calculator()
        market_context = get_market_context()
        
        buy_decisions = []
        sell_decisions = []
        
        # 매수 후보 결정
        for trade in signal_data.get('selected_trades', []):
            coin = trade['coin']
            pattern = trade.get('signal_pattern', 'unknown')
            interval = trade.get('interval', 'combined')
            
            # 🆕 [5-Candle Sequence Analysis] 추가 검증
            recent_candles = _get_recent_candles(coin, interval)
            if recent_candles is not None and len(recent_candles) >= 5:
                analysis = SequenceAnalyzer.analyze_sequence(recent_candles, interval)
                if not analysis['passed']:
                    print(f"  ✋ {coin} 매수 보류 (흐름분석 부적합): {analysis['reason']}")
                    continue
                if analysis['score_mod'] != 1.0:
                    print(f"  🌊 {coin} 흐름분석 반영: {analysis['reason']} (보정계수: {analysis['score_mod']:.2f})")
                    trade['signal_score'] = trade.get('signal_score', 0.0) * analysis['score_mod']
            
            # Thompson 점수
            res = thompson.sample_success_rate(pattern)
            sampled_rate = res[0] if isinstance(res, tuple) else float(res)
            
            # 알파 가디언 결정 (참고용으로만 사용)
            ai_res = guardian.make_trading_decision(
                signal_data={
                    **trade,
                    'wave_phase': trade.get('wave_phase', 'unknown'),
                    'integrated_direction': trade.get('integrated_direction', 'neutral')
                },
                current_price=trade['price'],
                market_context=market_context,
                coin_performance={'profit_rate': analyze_coin_performance(coin)}
            )
            
            ai_decision = ai_res.get('decision', 'hold').lower() if isinstance(ai_res, dict) else str(ai_res).lower()
            ai_score = ai_res.get('final_score', 0.0) if isinstance(ai_res, dict) else 0.0
            ai_reason = ai_res.get('reason', '분석 완료') if isinstance(ai_res, dict) else '분석 완료'
            
            # 💡 [Alpha Guardian] AI 판단 결과는 참고용 로그로만 남기고, 실제 매매 결정에는 참여하지 않음
            # [Sync] 가상과 동일하게 Thompson 0.3 이상이면 승인
            if sampled_rate >= 0.3:
                trade['enhanced_score'] = trade.get('signal_score', 0.0)
                trade['ai_decision_ref'] = ai_decision
                trade['ai_score_ref'] = ai_score
                trade['ai_reason_ref'] = ai_reason
                buy_decisions.append(trade)
                
        # 매도 후보 결정
        for holding in signal_data.get('current_holdings', []):
            coin = holding['coin']
            
            ai_res = guardian.make_trading_decision(
                signal_data={
                    **holding,
                    'wave_phase': holding.get('wave_phase', 'unknown'),
                    'integrated_direction': holding.get('integrated_direction', 'neutral')
                },
                current_price=holding['price'],
                market_context=market_context,
                coin_performance={'profit_rate': analyze_coin_performance(coin)}
            )
            
            ai_decision = ai_res.get('decision', 'hold').lower() if isinstance(ai_res, dict) else str(ai_res).lower()
            
            # 💡 [Alpha Guardian] 매도 결정에서도 AI는 참고용으로만 사용
            # 실제 매도는 시그널 생성기에서 생성된 SELL 액션에 따름
            if holding.get('action') == 'sell':
                holding['enhanced_score'] = holding.get('signal_score', 0.0)
                sell_decisions.append(holding)
                
        return {
            'buy': buy_decisions,
            'sell': sell_decisions
        }
        
    except Exception as e:
        print(f"⚠️ 실전 통합 판단 로직 오류: {e}")
        return {'buy': [], 'sell': []}

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

# 🆕 적응적 고급 지표 페널티 계산 (알파 가디언 바이어스 연동)
def calculate_adaptive_technical_penalty(holding):
    """🆕 설계 반영: 알파 가디언의 리스크 바이어스와 연동된 자율 페널티"""
    advanced_indicators = holding.get('advanced_indicators', {})
    market_context = get_market_context()
    
    # 🎯 알파 가디언의 현재 리스크 성향 가져오기
    risk_multiplier = 1.0
    try:
        from trade.core.decision import get_ai_decision_engine
        guardian = get_ai_decision_engine()
        bias = guardian.get_meta_bias()
        risk_multiplier = bias.get('risk_weight_multiplier', 1.0)
    except:
        pass

    penalty = 0.0
    
    # 🎯 시장 상황에 따른 베이스 페널티
    if market_context['trend'] == 'bearish':
        # 하락장에서 다이버전스/약세 트렌드 감지 시
        if (advanced_indicators.get('rsi_divergence') == 'bearish' or 
            advanced_indicators.get('macd_divergence') == 'bearish'):
            penalty += 0.15 * risk_multiplier # 자율 가중치 적용
        
        if advanced_indicators.get('trend_strength', 0.0) < 0.3:
            penalty += 0.10 * risk_multiplier
    
    elif market_context['trend'] == 'bullish':
        # 상승장에서는 페널티를 대폭 낮게 유지
        if (advanced_indicators.get('rsi_divergence') == 'bearish' or 
            advanced_indicators.get('macd_divergence') == 'bearish'):
            penalty += 0.08 * risk_multiplier
    
    else:  # 중립장
        if (advanced_indicators.get('rsi_divergence') == 'bearish' or 
            advanced_indicators.get('macd_divergence') == 'bearish'):
            penalty += 0.10 * risk_multiplier
    
    return min(penalty, 0.30)  # 최대 30%까지 유동적으로 적용

# 🆕 학습 기반 동적 임계값 조정
def get_dynamic_threshold(coin):
    """🆕 설계 반영: 시그널 생성기의 자율 임계값과 동기화 (전략적 신뢰)"""
    try:
        # 🎯 실전 매매 엔진 또한 시그널 생성기가 이미 동적 임계값을 통과했음을 전제로 합니다.
        # 기존의 보수적인 0.4 기준을 완화하여 시그널의 BUY 결정을 최대한 존중합니다.
        
        # 최소한의 안전장치(0.2)만 유지합니다. (시그널의 임계값 0.15~0.45 대응)
        return 0.2
        
    except Exception as e:
        print(f"⚠️ 동적 임계값 계산 오류 ({coin}): {e}")
        return 0.3  # 실패 시 약간 보수적으로 반환

# 🆕 자율형 동적 손절 강도 계산
def calculate_adaptive_stop_loss_strength(holding):
    """🆕 설계 반영: 시장 변동성 및 알파 가디언 성향과 연동된 자율 손절 강도"""
    try:
        coin = holding['coin']
        
        # 🎯 알파 가디언의 리스크 성향 반영
        risk_multiplier = 1.0
        try:
            from trade.core.decision import get_ai_decision_engine
            guardian = get_ai_decision_engine()
            risk_multiplier = guardian.get_meta_bias().get('risk_weight_multiplier', 1.0)
        except:
            pass
            
        # 🎯 코인별 과거 손절 성과 (기존 유지)
        stop_loss_performance = analyze_stop_loss_performance(coin)
        
        # 🎯 현재 시그널 강도 및 시장 변동성
        signal_strength = abs(holding.get('signal_score', 0.0))
        market_volatility = get_market_volatility()
        
        # 🎯 베이스 강도 계산 (기본 50% -> 리스크 성향에 따라 동적 시작)
        # 리스크 multiplier가 높을수록(보수적일수록) 손절 강도를 높여 더 빨리 손절함
        base_strength = 50.0 * risk_multiplier 
        
        # 🎯 성과 기반 보정
        if stop_loss_performance > 0.7: base_strength += 15.0
        elif stop_loss_performance < 0.3: base_strength -= 10.0
        
        # 🎯 시그널 및 변동성 보정
        if signal_strength > 0.6: base_strength += 10.0
        if market_volatility > 0.05: base_strength += 10.0
        
        return max(20.0, min(90.0, base_strength)) # 20~90% 범위 자율 조절
        
    except Exception as e:
        print(f"⚠️ 자율 손절 강도 계산 오류: {e}")
        return 50.0

# 🆕 시장 상황 분석 (Core 위임)
def get_market_context():
    """
    🆕 공통 모듈(trade.core.trading) 사용
    7단계 레짐 정보 포함: regime_stage, regime_group 추가
    """
    # 공통 모듈 호출 (캐싱 및 7단계 레짐 정규화 포함)
    context = get_common_market_context()
    
    # 7단계 레짐 정보 정규화 보장
    regime = normalize_regime(context.get('regime', 'neutral'))
    context['regime'] = regime
    context['trend'] = regime
    context['regime_stage'] = get_regime_severity(regime)
    
    return context

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
            
            # 🆕 [전략 시스템] 전략 분리 학습
            if STRATEGY_SYSTEM_AVAILABLE:
                holding_hours = get_holding_duration(coin) / 3600.0
                record_strategy_feedback_real(
                    coin=coin, profit_pct=profit_pct, 
                    success=(profit_pct > 0), holding_hours=holding_hours
                )
            
            remove_position_time(coin)
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
        from trade.core.database import get_db_connection
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
            conn.execute(insert_query, values)
    except Exception as e:
        logging.error(f"[DB 저장 오류] real_trade_history 기록 실패 - {data.get('coin')} | 오류: {e}")

def save_real_trade_feedback(trade_id: int, coin: str, signal_pattern: str, 
                            success_rate: float, avg_profit: float, total_trades: int, 
                            confidence: float, learning_episode: int, feedback_type: str):
    """실전 매매 피드백 저장 (trading_system.db, 쓰기 모드 안정성 강화)"""
    try:
        from trade.core.database import get_db_connection
        with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
            conn.execute("""
                INSERT INTO real_trade_feedback (
                    trade_id, coin, signal_pattern, success_rate, avg_profit, 
                    total_trades, confidence, learning_episode, feedback_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, coin, signal_pattern, success_rate, avg_profit, 
                  total_trades, confidence, learning_episode, feedback_type))
            conn.commit()
    except Exception as e:
        logging.error(f"[DB 저장 오류] real_trade_feedback 기록 실패 - {coin} | 오류: {e}")

def log_signal_based_trade(signal_data: dict):
    """
    시그널 기반 매매 정보를 별도로 기록 (통합 DB, 쓰기 모드 안정성 강화)
    - 시그널 정보와 실전 매매 정보를 연결하는 브릿지 역할
    """
    try:
        from trade.core.database import get_db_connection
        with get_db_connection(DB_PATH, read_only=False) as conn:
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
    """🆕🆕 개선된 실전매매 실행 (판단/실행 분리 + 우선순위 기반)"""
    
    # 🔥 [Critical] 기준 시각 설정 (DB 최신 캔들 기준)
    try:
        from trade.core.database import get_latest_candle_timestamp
        db_now = get_latest_candle_timestamp()
    except:
        db_now = int(time.time())
    
    # 🆕 [전략 시스템] 전략 피드백 테이블 초기화
    if STRATEGY_SYSTEM_AVAILABLE:
        try:
            create_strategy_feedback_table(TRADING_SYSTEM_DB_PATH)
        except Exception as e:
            print(f"⚠️ 전략 피드백 테이블 초기화 오류 (무시됨): {e}")
    
    print(f"🕒 실전매매 기준 시각 (DB): {db_now} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(db_now))})")
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
    
    # 🆕 가상매매 결정 로드 (최신 캔들 시각 기준 30분 이내)
    virtual_decisions = load_virtual_trade_decisions(max_age_minutes=30, reference_ts=db_now)
    print(f"📥 가상매매 학습 데이터: {len(virtual_decisions)}개")
    
    # 🆕 거래량 상위 40% 코인 미리 로드 (갈아타기 필터링용)
    top_volume_coins = load_target_coins()
    print(f"📊 거래량 필터: 상위 {len(top_volume_coins)}개 코인만 매수 가능")
    
    # ─────────────────────────────────────────────────────────────────
    # [2-1] 보유 코인 판단: 매도/홀딩/갈아타기
    # ─────────────────────────────────────────────────────────────────
    print(f"\n📊 [2-1] 보유 코인 {len(wallet_coins)}개 판단 중...")
    
    # 🎯 시장 레짐 정보 조회 및 출력 (공통 정보이므로 한 번만 출력)
    market_context = get_market_context()
    market_regime = market_context.get('regime', 'Neutral')
    market_score = market_context.get('score', 0.5)
    
    # 🆕 레짐 변화 감지 및 전략 재평가
    regime_changed = False
    recommended_strategies = []
    try:
        from trade.core.strategies import get_regime_detector
        detector = get_regime_detector()
        should_reevaluate, reason = detector.should_reevaluate_strategies(market_regime)
        stability, stability_desc = detector.get_regime_stability()
        
        if should_reevaluate:
            regime_changed = True
            recommended_strategies = detector.get_recommended_strategies_for_regime(market_regime)
            print(f"🔄 {reason}")
            print(f"   📋 현재 레짐에 추천 전략: {', '.join(recommended_strategies[:3])}")
        
        print(f"📊 시장 레짐: {market_regime} (점수: {market_score:.2f}, 안정성: {stability:.1f})")
    except Exception as e:
        print(f"📊 시장 레짐: {market_regime} (점수: {market_score:.2f})")
    
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
            
            # 🆕 파동 및 통합 방향 정보 추출
            wave_info = signal_data.get('wave_info', {})
            wave_phase = wave_info.get('wave_phase', 'unknown')
            integrated_direction = wave_info.get('integrated_direction', 'neutral')
        
        # 수익률 계산
        profit_loss_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 and current_price > 0 else 0.0
        
        # 보유 시간 조회
        holding_seconds = get_holding_duration(coin)
        holding_hours = holding_seconds / 3600
        
        # 🆕 [전략 시스템] 전략 전환 확인
        if STRATEGY_SYSTEM_AVAILABLE:
            switched, new_strategy, switch_reason = check_strategy_switch_real(
                coin, profit_loss_pct, holding_hours
            )
            if switched:
                print(f"   📋 전략 전환 이유: {switch_reason}")
        
        # 🆕 수익률 스냅샷 기록 (추세 분석용)
        trend_analysis = None
        if TRAJECTORY_ANALYZER_AVAILABLE:
            try:
                trajectory_analyzer = get_real_trajectory_analyzer()
                trajectory_analyzer.record_profit_snapshot(
                    coin=coin,
                    profit_pct=profit_loss_pct,
                    current_price=current_price,
                    entry_price=entry_price,
                    signal_score=signal_score,
                    holding_hours=holding_hours,
                    market_regime=market_regime
                )
                # 추세 분석 실행
                trend_analysis = trajectory_analyzer.analyze_trend(coin, lookback=10)
            except Exception as e:
                print(f"⚠️ {coin} 추세 분석 오류: {e}")
        
        # 🆕 추세 분석 결과 (데이터 준비)
        trend_summary = f"({trend_analysis.trend_type.value})" if trend_analysis and trend_analysis.history_count >= 3 else ""
        
        # 최종 액션 판단 (알파 가디언 + 공통 전략 엔진)
        # 🆕 알파 가디언 판단을 먼저 수행하여 로그 출력 보장
        ai_action = 'hold'
        ai_score = 0.0
        ai_reason = '알파 가디언 분석 완료'
        
        if real_time_ai_decision_engine:
            try:
                signal_data_for_ai = {
                    'coin': coin,
                    'action': pure_action,
                    'signal_score': signal_score,
                    'confidence': abs(signal_score),
                    'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low',
                    'wave_phase': wave_phase,
                    'integrated_direction': integrated_direction
                }
                market_context = get_market_context()
                
                ai_res = real_time_ai_decision_engine.make_trading_decision(
                    signal_data=signal_data_for_ai,
                    current_price=current_price,
                    market_context=market_context,
                    # 🆕 Thompson 기반 패턴 성과 조회 (정밀 분석용)
                    coin_performance=thompson_sampler.get_decision_engine_stats(coin)
                )
                
                # 🆕 딕셔너리 형태로 반환되므로 처리
                if isinstance(ai_res, dict):
                    ai_action = ai_res.get('decision', 'hold')
                    ai_score = ai_res.get('final_score', 0.0)
                    ai_reason = ai_res.get('reason', '알파 가디언 분석 완료')
                else:
                    ai_action = ai_res if isinstance(ai_res, str) else 'hold'
                    ai_score = 0.0
                    ai_reason = '알파 가디언 분석 완료'
            except Exception as e:
                print(f"   ⚠️ 알파 가디언 판단 오류 ({coin}): {e}")
                ai_action = 'hold'
                ai_score = 0.0
                ai_reason = f'알파 가디언 분석 오류: {str(e)[:50]}'
        else:
            ai_action = 'hold'
            ai_score = 0.0
            ai_reason = '알파 가디언 비활성화됨'
        
        # 🆕 호가 해상도 필터를 위한 틱 사이즈 로드
        from trade.trade_manager import get_bithumb_tick_size
        tick_size = get_bithumb_tick_size(current_price)

        # 🆕 전략 정보 조회 (전략별 청산 규칙 적용용)
        strategy_info = get_position_strategy_info(coin)
        current_strategy = strategy_info.get('current_strategy', 'trend')
        
        final_action, action_reason = combine_signal_with_holding(
            coin=coin,
            pure_action=pure_action,
            signal_score=signal_score,
            profit_loss_pct=profit_loss_pct,
            signal_pattern=reason if reason else 'unknown',
            max_profit_pct=max(profit_loss_pct, 0.0) if trend_analysis is None else trend_analysis.max_profit_pct,
            entry_volatility=0.02,
            holding_hours=holding_hours,
            trend_analysis=trend_analysis,
            ai_decision=ai_action,  # 🆕 알파 가디언 판단 결과 전달
            tick_size=tick_size,
            current_price=current_price,
            current_strategy=current_strategy  # 🆕 전략별 청산 규칙 적용
        )

        # 🆕 통합 상세 로그 (가상매매와 포맷 통일) - 코인명 + 최종판단 먼저 출력
        print(f"📊 {get_korean_name(coin)}: 최종판단={final_action.upper()} (점수: {signal_score:.3f})")
        
        # 🆕 액션 사유 출력 (전략별 청산 등 상세 사유)
        if action_reason:
            print(f"   {action_reason}")
        
        # 🆕 알파 가디언 판단 로그를 요약 바로 아래 출력 (가독성 개선)
        print(f"   🛡️ [알파 가디언] 판단: {ai_action.upper()} (점수: {ai_score:.3f})")
        print(f"   💬 근거: {ai_reason}")
        
        print(f"   📈 보유정보: {format_price(entry_price)}원 → {format_price(current_price)}원 ({profit_loss_pct:+.2f}%, {holding_hours:.1f}h)")
        
        # 가상매매 참조 데이터 표시
        if virtual_thompson > 0 or virtual_decision_ref != 'N/A':
            target_status = ""
            if target_price_ref > 0:
                dist_pct = ((target_price_ref - current_price) / current_price) * 100
                target_status = f", 목표까지 {dist_pct:+.2f}%"
            print(f"   📥 가상참조: {virtual_decision_ref.upper()}, Thompson {virtual_thompson:.2f}{target_status}")

        if trend_analysis and trend_analysis.history_count >= 3:
            print(f"   📉 추세분석: {trend_analysis.trend_type.value} ({trend_analysis.reason})")
            if trend_analysis.should_sell_early: print(f"   ⚠️ 조기 매도 권장!")
            if trend_analysis.should_hold_strong: print(f"   💪 강한 홀딩 권장!")
        
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
            'decision_timestamp': int(time.time()),
            'wave_phase': wave_phase,
            'integrated_direction': integrated_direction
        }
        
        # ═══ 분류 ═══
        # 1순위: 손절 (stop_loss)
        if final_action == 'stop_loss':
            stop_loss_decisions.append(decision_data)
        
        # 2순위: 갈아타기 조건 체크
        elif final_action in ['hold'] and profit_loss_pct < 3.0:
            # 🆕 전략 정보 조회 (횡보 정책 적용용)
            strategy_info = get_position_strategy_info(coin)
            current_strategy = strategy_info.get('current_strategy', 'trend')
            
            # 🆕 개선된 갈아타기 조건 체크 (시장 점수 및 추세 반영 + 전략별 횡보 정책 + 레짐)
            should_switch, switch_reason, switch_type = check_switch_condition(
                coin=coin,
                profit_pct=profit_loss_pct,
                holding_hours=holding_hours,
                target_price=target_price_ref,
                current_price=current_price,
                market_score=market_score,
                trend_analysis=trend_analysis,
                strategy_type=current_strategy,  # 🆕 전략별 횡보 정책 적용
                market_regime=market_regime      # 🆕 레짐 반영
            )
            
            if should_switch:
                # 🆕 대안 코인 찾기 (학습 결과 중심)
                target = find_best_switch_target(
                    virtual_decisions=virtual_decisions,
                    wallet_coins=wallet_coins,
                    current_coin=coin,
                    min_signal_score=0.25,
                    top_volume_coins=top_volume_coins
                )
                
                if target:
                    # 🛡️ 알파 가디언 갈아타기 판단 (대상 코인 매수 승인 여부)
                    to_coin = target['coin']
                    target_signal_score = target.get('signal_score', 0.0)
                    target_confidence = virtual_decisions.get(to_coin, {}).get('confidence', 0.5)
                    target_current_price = target.get('current_price', 0.0)
                    
                    # 🛡️ 알파 가디언 갈아타기 판단 (참고용으로만 사용)
                    if real_time_ai_decision_engine:
                        signal_data_for_ai = {
                            'coin': to_coin,
                            'action': 'buy',
                            'signal_score': target_signal_score,
                            'confidence': target_confidence,
                            'risk_level': 'high' if abs(target_signal_score) > 0.7 else 'medium' if abs(target_signal_score) > 0.4 else 'low'
                        }
                        
                        ai_res = real_time_ai_decision_engine.make_trading_decision(
                            signal_data=signal_data_for_ai,
                            current_price=target_current_price,
                            market_context=market_context,
                            coin_performance=thompson_sampler.get_decision_engine_stats(to_coin)
                        )
                        
                        if isinstance(ai_res, dict):
                            ai_action = ai_res.get('decision', 'hold')
                            ai_score = ai_res.get('final_score', 0.0)
                            ai_reason = ai_res.get('reason', '알파 가디언 분석 완료')
                        else:
                            ai_action = str(ai_res)
                            ai_score = 0.0
                            ai_reason = '알파 가디언 분석 완료'
                        
                        # 💡 [Alpha Guardian] 갈아타기 결정에서도 AI는 참고용으로만 사용
                        if ai_action != 'buy':
                            print(f"   🛡️ [알파 가디언 (참고용)] 갈아타기 거부 ({ai_action.upper()}, 점수: {ai_score:.3f}) - {ai_reason}")
                        else:
                            print(f"   🛡️ [알파 가디언 (참고용)] 갈아타기 승인 (점수: {ai_score:.3f}) - {ai_reason}")
                        
                        target['ai_action'] = ai_action
                        target['ai_score'] = ai_score
                        target['ai_reason'] = ai_reason
                    else:
                        target['ai_action'] = 'buy'
                        target['ai_score'] = 0.0
                        target['ai_reason'] = '알파 가디언 비활성화됨'
                    
                    print(f"   🔄 갈아타기 대상 감지 → {get_korean_name(target['coin'])}")
                    decision_data['switch_reason'] = switch_reason
                    decision_data['switch_type'] = switch_type
                    decision_data['target'] = target
                    switch_decisions.append(decision_data)
                else:
                    hold_decisions.append(decision_data)
            else:
                hold_decisions.append(decision_data)
    
        # 3순위: 일반 매도/익절
        elif final_action in ['sell', 'take_profit', 'partial_sell']:
            sell_decisions.append(decision_data)
        
        # 홀딩
        else:
            hold_decisions.append(decision_data)

    # ─────────────────────────────────────────────────────────────────
    # [2-2] 신규 매수 + 추가 매수 후보 판단
    # ─────────────────────────────────────────────────────────────────
    print(f"\n📊 [2-2] 매수 후보 판단 (신규 + 추매)...")
    
    # 🎯 시장 상황 조회 (매수 결정에 반영)
    market_context = get_market_context()
    market_regime = market_context.get('regime', 'Neutral')
    market_trend = market_context.get('trend', 'neutral')
    market_score = market_context.get('score', 0.5)
    
    # 🎯 시장 상황에 따른 매수 임계값 조정
    regime_lower = market_regime.lower() if market_regime else 'neutral'
    is_bearish = 'bearish' in regime_lower or market_trend == 'bearish'
    is_extreme_bearish = 'extreme_bearish' in regime_lower
    is_bullish = 'bullish' in regime_lower or market_trend == 'bullish'
    
    # 기본 임계값 (보수성 완화를 위해 하향 조정)
    BASE_MIN_SIGNAL_SCORE = 0.05
    BASE_MIN_SIGNAL_SCORE_ADDITIONAL = 0.15
    BASE_MIN_THOMPSON_SCORE = 0.10
    
    # 🆕 [이중 신뢰도 동적 가중치] 전역 수준 기본 계산 (개별 코인별로 재계산됨)
    signal_weight, learning_weight, maturity_desc = get_dynamic_weights(for_buy=True)
    print(f"   📊 동적 가중치: {maturity_desc} (시그널 {signal_weight:.0%} / 학습 {learning_weight:.0%})")
    
    # 🆕 학습 성숙도가 높으면 Thompson(학습) 기준을 약간 낮춤 (경험 신뢰)
    thompson_maturity_adj = learning_weight * -0.03  # 최대 -2.1% (성숙시)
    
    # 시장 상황에 따른 임계값 조정
    if is_extreme_bearish:
        # 극심한 하락장: 매우 엄격한 기준 (매수 거의 차단)
        MIN_SIGNAL_SCORE = BASE_MIN_SIGNAL_SCORE + 0.10  # 0.15
        MIN_SIGNAL_SCORE_ADDITIONAL = BASE_MIN_SIGNAL_SCORE_ADDITIONAL + 0.15  # 0.30
        MIN_THOMPSON_SCORE = BASE_MIN_THOMPSON_SCORE + 0.15 + thompson_maturity_adj
        print(f"   ⚠️ 극심한 하락장 감지: 매수 기준 강화 (시그널: {MIN_SIGNAL_SCORE:.2f}, Thompson: {MIN_THOMPSON_SCORE:.2f})")
    elif is_bearish:
        # 하락장: 엄격한 기준
        MIN_SIGNAL_SCORE = BASE_MIN_SIGNAL_SCORE + 0.05  # 0.10
        MIN_SIGNAL_SCORE_ADDITIONAL = BASE_MIN_SIGNAL_SCORE_ADDITIONAL + 0.08  # 0.23
        MIN_THOMPSON_SCORE = BASE_MIN_THOMPSON_SCORE + 0.08 + thompson_maturity_adj
        print(f"   ⚠️ 하락장 감지: 매수 기준 강화 (시그널: {MIN_SIGNAL_SCORE:.2f}, Thompson: {MIN_THOMPSON_SCORE:.2f})")
    elif is_bullish:
        # 상승장: 완화된 기준 (더 쉽게 매수)
        MIN_SIGNAL_SCORE = BASE_MIN_SIGNAL_SCORE - 0.02  # 0.03
        MIN_SIGNAL_SCORE_ADDITIONAL = BASE_MIN_SIGNAL_SCORE_ADDITIONAL - 0.05  # 0.10
        MIN_THOMPSON_SCORE = BASE_MIN_THOMPSON_SCORE - 0.05 + thompson_maturity_adj
        print(f"   ✅ 상승장 감지: 매수 기준 완화 (시그널: {MIN_SIGNAL_SCORE:.2f}, Thompson: {MIN_THOMPSON_SCORE:.2f})")
    else:
        # 중립장: 기본 기준
        MIN_SIGNAL_SCORE = BASE_MIN_SIGNAL_SCORE
        MIN_SIGNAL_SCORE_ADDITIONAL = BASE_MIN_SIGNAL_SCORE_ADDITIONAL
        MIN_THOMPSON_SCORE = BASE_MIN_THOMPSON_SCORE + thompson_maturity_adj
        print(f"   ➡️ 중립장: 기본 기준 (시그널: {MIN_SIGNAL_SCORE:.2f}, Thompson: {MIN_THOMPSON_SCORE:.2f})")
    
    # 🎯 예수금에 따라 최대 매수 가능 개수 계산
    available_balance_for_calc = get_available_balance()
    MIN_BALANCE_REQUIRED = 1_000_000  # 최소 예수금 (100만원 이하면 매수 안함)
    MAX_BALANCE_FOR_SINGLE = 2_000_000  # 단일 매수 최대 금액: 200만원
    
    # 예수금이 100만원 초과이고 200만원 이하면 1개 매수 가능 (예수금 전액 사용)
    # 예수금이 200만원 초과면 200만원씩 여러 개 매수 가능
    if available_balance_for_calc > MIN_BALANCE_REQUIRED:
        if available_balance_for_calc <= MAX_BALANCE_FOR_SINGLE:
            # 100만원 초과 ~ 200만원 이하: 1개 매수 (예수금 전액 사용)
            MAX_SIGNAL_CANDIDATES = 1
        else:
            # 200만원 초과: 200만원씩 여러 개 매수 가능
            max_buy_count = int(available_balance_for_calc / MAX_BALANCE_FOR_SINGLE)
            # 최대 10개로 제한 (너무 많이 매수하는 것 방지)
            MAX_SIGNAL_CANDIDATES = min(max_buy_count, 10)
    else:
        MAX_SIGNAL_CANDIDATES = 0  # 예수금 부족 (100만원 이하)
    
    # top_volume_coins는 이미 위에서 로드됨 (중복 로드 방지)
    print(f"📊 분석 대상: {len(top_volume_coins)}개 (거래량 상위 40%)")
    if available_balance_for_calc > MIN_BALANCE_REQUIRED:
        if available_balance_for_calc <= MAX_BALANCE_FOR_SINGLE:
            print(f"💰 예수금 기반 최대 매수 가능: {MAX_SIGNAL_CANDIDATES}개 (예수금: {available_balance_for_calc:,.0f}원, 예수금 전액 사용)")
        else:
            print(f"💰 예수금 기반 최대 매수 가능: {MAX_SIGNAL_CANDIDATES}개 (예수금: {available_balance_for_calc:,.0f}원, 매수당: {MAX_BALANCE_FOR_SINGLE:,.0f}원)")
    else:
        print(f"💰 예수금 부족: {available_balance_for_calc:,.0f}원 (최소 필요: {MIN_BALANCE_REQUIRED:,.0f}원)")
    
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
        
        # 🆕 파동 및 통합 방향 정보 추출
        wave_info = signal_data.get('wave_info', {})
        wave_phase = wave_info.get('wave_phase', 'unknown')
        integrated_direction = wave_info.get('integrated_direction', 'neutral')
                
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
            
            # 추가 매수 조건 (시장 상황 반영):
            # 1. 시그널 점수가 높음 (시장 상황에 따라 조정된 기준)
            # 2. 현재 수익률이 양수 (수익 중)
            # 3. 보유 시간이 1시간 이상 (너무 빨리 추매 방지)
            # 4. Thompson 점수가 충분히 높음 (시장 상황에 따라 조정된 기준)
            # 5. 🆕 극심한 하락장에서는 추매 차단
            if is_extreme_bearish:
                # 극심한 하락장에서는 추매도 차단 (현금 보유 우선)
                continue
                            
            if (signal_score >= MIN_SIGNAL_SCORE_ADDITIONAL and 
                current_profit_pct >= 0.5 and 
                holding_hours >= 1.0 and 
                thompson_score >= MIN_THOMPSON_SCORE):
                
                # 🛡️ 알파 가디언 추매 판단 (참고용으로만 사용)
                if real_time_ai_decision_engine:
                    signal_data_for_ai = {
                        'coin': coin,
                        'action': pure_action,
                        'signal_score': signal_score,
                        'confidence': confidence,
                        'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low',
                        'wave_phase': wave_phase,
                        'integrated_direction': integrated_direction
                    }
                    
                    ai_res = real_time_ai_decision_engine.make_trading_decision(
                        signal_data=signal_data_for_ai,
                        current_price=wallet_current_price,
                        market_context=market_context,
                        coin_performance=thompson_sampler.get_decision_engine_stats(coin)
                    )
                    
                    if isinstance(ai_res, dict):
                        ai_action = ai_res.get('decision', 'hold')
                        ai_score = ai_res.get('final_score', 0.0)
                        ai_reason = ai_res.get('reason', '알파 가디언 분석 완료')
                    else:
                        ai_action = str(ai_res)
                        ai_score = 0.0
                        ai_reason = '알파 가디언 분석 완료'
                    
                    # 💡 [Alpha Guardian] 추매 결정에서도 AI는 참고용으로만 사용
                    if ai_action != 'buy':
                        print(f"   🛡️ [알파 가디언 (참고용)] 추매 거부 ({ai_action.upper()}, 점수: {ai_score:.3f}) - {ai_reason}")
                    else:
                        print(f"   🛡️ [알파 가디언 (참고용)] 추매 승인 (점수: {ai_score:.3f}) - {ai_reason}")
                else:
                    ai_action = 'buy'
                    ai_score = 0.0
                    ai_reason = '알파 가디언 비활성화됨'
                
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
                    'ai_action': ai_action,  # 🆕 알파 가디언 판단 추가
                    'ai_score': ai_score,  # 🆕 알파 가디언 점수 추가
                    'ai_reason': ai_reason,  # 🆕 알파 가디언 근거 추가
                    'decision_timestamp': int(time.time())
                })
                print(f"   🔵 {get_korean_name(coin)}: 추매 후보 (점수: {signal_score:.3f}, 현수익: {current_profit_pct:+.2f}%)")
            continue  # 추가 매수 조건 체크 후 다음 코인으로
        
        # ═══════════════════════════════════════════════════════════════
        # 신규 매수 조건 체크 (시장 상황 반영)
        # ═══════════════════════════════════════════════════════════════
        # 🆕 [이중 신뢰도] 개별 코인별 신뢰도 계산
        # 시그널 신뢰도: 시그널 강도 + 신뢰도
        sig_strength = min(1.0, abs(signal_score) * 2.0)
        signal_conf = (sig_strength + confidence) / 2.0
        
        # 패턴 학습 신뢰도: Thompson 점수 기반
        pattern_learning_conf = min(1.0, thompson_score + 0.3)
        
        # 인터벌 방향 일치도: integrated_direction 기반
        direction_score = 0.7 if integrated_direction in ['up', 'strong_up'] else (0.3 if integrated_direction in ['down', 'strong_down'] else 0.5)
        interval_align = direction_score
        
        # 개별 코인별 동적 가중치 계산
        coin_signal_w, coin_learning_w, coin_weight_desc = get_dynamic_weights(
            for_buy=True,
            signal_confidence=signal_conf,
            pattern_confidence=pattern_learning_conf,
            interval_alignment=interval_align
        )
        
        # 🎯 시장 상황이 극심한 하락장이면 추가 필터링
        if is_extreme_bearish:
            # 극심한 하락장에서는 기대수익률도 더 높게 요구
            if expected_profit < 3.0:  # 기본 0% → 3% 이상 요구
                continue
        
        # 🆕 이중 신뢰도 기반 임계값 조정
        # 양쪽 신뢰도 모두 높으면 더 적극적 매매 (임계값 완화)
        both_confident = signal_conf > 0.6 and pattern_learning_conf > 0.6
        adjusted_min_signal = MIN_SIGNAL_SCORE - 0.02 if both_confident else MIN_SIGNAL_SCORE
        adjusted_min_thompson = MIN_THOMPSON_SCORE - 0.03 if both_confident else MIN_THOMPSON_SCORE
        
        if signal_score < adjusted_min_signal:
            continue
        if thompson_score < adjusted_min_thompson:
            continue
        if expected_profit < 0:
            continue
        if current_price <= 0:
            continue
        
        # 🛡️ 알파 가디언 매수 판단
        if real_time_ai_decision_engine:
            # 🆕 파동 및 방향 정보 추출
            wave_info = signal_data.get('wave_info', {})
            wave_phase = wave_info.get('wave_phase', 'unknown')
            integrated_direction = wave_info.get('integrated_direction', 'neutral')

            signal_data_for_ai = {
                'coin': coin,
                'action': pure_action,
                'signal_score': signal_score,
                'confidence': confidence,
                'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low',
                'wave_phase': wave_phase,
                'integrated_direction': integrated_direction
            }
            
            ai_res = real_time_ai_decision_engine.make_trading_decision(
                signal_data=signal_data_for_ai,
                current_price=current_price,
                market_context=market_context,
                coin_performance=thompson_sampler.get_decision_engine_stats(coin)
            )
            
            # 🆕 딕셔너리 형태로 반환되므로 처리
            if isinstance(ai_res, dict):
                ai_action = ai_res.get('decision', 'hold')
                ai_score = ai_res.get('final_score', 0.0)
                ai_reason = ai_res.get('reason', '알파 가디언 분석 완료')
            else:
                ai_action = ai_res
                ai_score = 0.0
                ai_reason = '알파 가디언 분석 완료'
            
            # 🆕 알파 가디언이 매수를 승인하지 않더라도 로그만 남기고 계속 진행
            # 🔧 코인 정보 포함하여 로그 출력
            if ai_action != 'buy':
                print(f"   🟢 {get_korean_name(coin)}: 시그널 {signal_score:.3f}, Thompson {thompson_score:.2f}, 기대수익 {expected_profit:.2f}%")
                print(f"      🛡️ [알파 가디언] {ai_action.upper()} (점수: {ai_score:.3f}) - {ai_reason}")
            else:
                print(f"   🟢 {get_korean_name(coin)}: 시그널 {signal_score:.3f}, Thompson {thompson_score:.2f}, 기대수익 {expected_profit:.2f}%")
                print(f"      🛡️ [알파 가디언] BUY 승인 (점수: {ai_score:.3f})")
        else:
            ai_action = 'buy'
            ai_score = 0.0
            ai_reason = '알파 가디언 비활성화됨'
            # 🔧 알파 가디언 비활성화 시에도 코인 정보 출력
            print(f"   🟢 {get_korean_name(coin)}: 시그널 {signal_score:.3f}, Thompson {thompson_score:.2f}, 기대수익 {expected_profit:.2f}%")
        
        # 🆕 [전략 시스템] 전략 선택
        strategy_type = 'trend'  # 기본 전략
        strategy_match = 0.5
        if STRATEGY_SYSTEM_AVAILABLE:
            # 시그널에서 추천 전략 추출
            signal_info = signal_data.get('signal_info', {})
            if signal_info.get('recommended_strategy'):
                strategy_type = signal_info['recommended_strategy']
                strategy_match = signal_info.get('strategy_match', 0.5)
            elif signal_info.get('strategy_scores'):
                # 직접 전략 점수에서 최적 전략 선택
                strategy_scores_raw = signal_info['strategy_scores']
                if isinstance(strategy_scores_raw, str):
                    strategy_scores_raw = deserialize_strategy_scores(strategy_scores_raw)
                if strategy_scores_raw:
                    best_strat = max(strategy_scores_raw.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1].get('match', 0))
                    strategy_type = best_strat[0]
                    strategy_match = best_strat[1] if isinstance(best_strat[1], (int, float)) else best_strat[1].get('match', 0.5)
            
            # 전략별 학습 데이터 기반 신뢰도 조회
            strat_rate, strat_conf = get_strategy_success_rate(
                db_path=TRADING_SYSTEM_DB_PATH,
                strategy_type=strategy_type,
                market_condition=regime_name
            )
            print(f"      🎯 [{strategy_type.upper()}] 전략 선택 (적합도: {strategy_match:.2f}, 학습 성공률: {strat_rate:.2f})")
        
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
            'ai_action': ai_action,  # 🆕 알파 가디언 판단 추가
            'ai_score': ai_score,  # 🆕 알파 가디언 점수 추가
            'ai_reason': ai_reason,  # 🆕 알파 가디언 근거 추가
            'decision_timestamp': int(time.time()),
            'strategy_type': strategy_type,  # 🆕 매매 전략
            'strategy_match': strategy_match  # 🆕 전략 적합도
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
                # 예수금이 200만원 이하면 예수금 전액 사용, 200만원 초과면 200만원 사용
                if available_balance <= MAX_BALANCE_FOR_SINGLE:
                    buy_amount = available_balance * 0.995
                else:
                    buy_amount = MAX_BALANCE_FOR_SINGLE * 0.995
                
                if buy_amount > MIN_BALANCE_REQUIRED:
                    # 🛡️ 알파 가디언 갈아타기 매수 최종 확인 (매도 성공 후 재확인)
                    if real_time_ai_decision_engine:
                        target_signal_score = target.get('signal_score', 0.0)
                        target_confidence = virtual_decisions.get(to_coin, {}).get('confidence', 0.5)
                        target_current_price = target.get('current_price', 0.0)
                        
                        signal_data_for_ai = {
                            'coin': to_coin,
                            'action': 'buy',
                            'signal_score': target_signal_score,
                            'confidence': target_confidence,
                            'risk_level': 'high' if abs(target_signal_score) > 0.7 else 'medium' if abs(target_signal_score) > 0.4 else 'low'
                        }
                        
                        ai_res = real_time_ai_decision_engine.make_trading_decision(
                            signal_data=signal_data_for_ai,
                            current_price=target_current_price,
                            market_context=market_context,
                            # 🆕 Thompson 기반 패턴 성과 조회 (정밀 분석용)
                            coin_performance=thompson_sampler.get_decision_engine_stats(to_coin)
                        )
                        
                        # 🆕 딕셔너리 형태로 반환되므로 처리
                        if isinstance(ai_res, dict):
                            ai_action = ai_res.get('decision', 'hold')
                            ai_score = ai_res.get('final_score', 0.0)
                            ai_reason = ai_res.get('reason', '알파 가디언 분석 완료')
                        else:
                            ai_action = ai_res
                            ai_score = 0.0
                            ai_reason = '알파 가디언 분석 완료'
                        
                        # 🆕 알파 가디언 판단 로그 출력 (참고용)
                        if ai_action != 'buy':
                            print(f"      🛡️ [알파 가디언 (참고용)] 갈아타기 매수 보류 권고 ({ai_action.upper()}, 점수: {ai_score:.3f}) - {ai_reason}")
                        else:
                            print(f"      🛡️ [알파 가디언 (참고용)] 갈아타기 매수 승인 (점수: {ai_score:.3f}) - {ai_reason}")
                        
                        # 🚀 [결정] 알파 가디언 판단과 무관하게 갈아타기 매수 실행 (결정권 박탈)
                    
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
                        # 🆕 전략 정보 포함
                        entry_strategy = target.get('recommended_strategy', 'trend')
                        strategy_match = target.get('strategy_match', 0.5)
                        
                        # 🧬 진화 레벨 조회
                        evolution_level = 1
                        evolved_params = ''
                        if EVOLUTION_SYSTEM_AVAILABLE:
                            try:
                                regime = get_market_context().get('regime', 'neutral')
                                evolution_level = get_strategy_level(entry_strategy, regime)
                            except:
                                pass
                        
                        record_position_buy_time(to_coin, target.get('current_price', 0), 
                                                entry_strategy, strategy_match,
                                                evolution_level, evolved_params)
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
        
        # 🆕 개별 매도 결과는 execute_enhanced_signal_trades 내부에서 상세히 출력함
        # 불필요한 중복 로그 제거 및 실제 체결 여부와 무관한 '매도 완료' 출력 방지
    
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
        
        if available_balance > 1_000_000 and buy_candidates:
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
                    
                    # 매수 금액 계산:
                    # - 예수금이 200만원 이하면 예수금 전액 사용
                    # - 예수금이 200만원 초과면 200만원씩 사용
                    if virtual_balance <= MAX_BALANCE_FOR_SINGLE:
                        buy_amount = virtual_balance * 0.995  # 예수금 전액 사용
                    else:
                        buy_amount = MAX_BALANCE_FOR_SINGLE * 0.995  # 200만원씩
                    
                    if virtual_balance <= MIN_BALANCE_REQUIRED:
                        print(f"   ⚠️ 예수금 부족 (남은 예수금: {virtual_balance:,.0f}원 <= {MIN_BALANCE_REQUIRED:,.0f}원) - 중단")
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
                        
                        # 🆕 실전 매매 실시간 학습 (진행 중인 거래 위험 감지)
                        # buy 시점에는 profit이 0이므로 learn_from_trade 생략 (필요 시 learn_from_ongoing_drawdown 사용)
                        
                        # 🆕 시그널-매매 연결 (인과관계 정밀 추적)
                        # candidate['signal'] 이 SignalInfo 객체라고 가정 (아니라면 candidate 활용)
                        sig_info = candidate.get('signal')
                        if not sig_info:
                            sig_info = SignalInfo(
                                coin,                                      # coin
                                candidate.get('interval', 'combined'),     # interval
                                SignalAction.BUY,                          # action
                                float(candidate['signal_score']),          # signal_score
                                float(candidate.get('confidence', 0.5)),   # confidence
                                candidate.get('reason', 'Signal_Buy'),     # reason
                                int(time.time())                           # timestamp
                            )
                            # 선택적 필드 설정
                            sig_info.price = float(candidate.get('price', 0.0))
                        
                        SignalTradeConnector().connect_signal_to_trade(sig_info, trade_result)
                        
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
                            # 🆕 전략 정보 포함
                            entry_strategy = candidate.get('recommended_strategy', 'trend')
                            strategy_match = candidate.get('strategy_match', 0.5)
                            
                            # 🧬 진화 레벨 조회
                            evolution_level = 1
                            evolved_params = ''
                            if EVOLUTION_SYSTEM_AVAILABLE:
                                try:
                                    regime = get_market_context().get('regime', 'neutral')
                                    evolution_level = get_strategy_level(entry_strategy, regime)
                                except:
                                    pass
                            
                            record_position_buy_time(coin, candidate['price'], 
                                                    entry_strategy, strategy_match,
                                                    evolution_level, evolved_params)
                        
                        executed_buy_coins.add(coin)
                        
                        print(f"   ✅ {get_korean_name(coin)} {buy_type} 완료")
                    else:
                        print(f"   ❌ {get_korean_name(coin)} {buy_type} 실패")
            else:
                print("   ℹ️ 매수 가능한 후보 없음 (이미 처리됨)")
        elif available_balance <= 1_000_000:
            print("   ⚠️ 예수금 부족 (100만원 이하)")
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
    
    # 🆕 Thompson Sampling 기반 성과 출력
    for action in ['buy', 'sell']:
        # 코인별 통합 성과 (ALL 코인 기준)
        stats = thompson_sampler.get_decision_engine_stats('ALL') # 또는 특정 패턴
        if stats['total_trades'] > 0:
            print(f"   📈 {action.upper()} 학습 지식: {stats['total_trades']}회 완료, 기대승률: {stats['success_rate']:.1%}")
    
    # 🆕 24시간 빗썸 거래내역 출력
    try:
        print_trade_summary_24h()
    except Exception as e:
        print(f"⚠️ 24시간 거래내역 조회 오류: {e}")
    
    print("\n✅ 실전매매 사이클 완료!")
            
    return executed_trades

def combine_signal_with_holding(coin: str, pure_action: str, signal_score: float, profit_loss_pct: float, 
                                 signal_pattern: str = 'unknown', max_profit_pct: float = None,
                                 entry_volatility: float = 0.02, holding_hours: float = 0,
                                 trend_analysis = None, ai_decision: str = 'hold',
                                 tick_size: float = 0.0, current_price: float = 0.0,
                                 signal_continuity: float = 0.5, dynamic_influence: float = 0.5,
                                 current_strategy: str = 'trend') -> Tuple[str, str]:
    """🆕 통합된 계층적 의사결정 전략 적용 (수익 보호 우선 + 전략별 청산 규칙)
    
    🔥 공통 원칙 (trade/core/executor/strategy.py 참조):
    - 시그널의 action(BUY/SELL)이 아니라 signal_score와 보유 정보를 종합 판단
    - should_sell_holding_position() 공통 함수 사용
    - 🆕 전략별 청산 규칙 (STRATEGY_EXIT_RULES) 적용
    
    Args:
        ai_decision: 알파 가디언 판단 결과 (함수 호출 전에 이미 판단됨)
        tick_size: 호가 단위
        current_price: 현재가
        signal_continuity: 이전 시그널과의 방향성 일치도 (0~1)
        dynamic_influence: 시그널 품질 기반 동적 영향도 (0~1)
        current_strategy: 🆕 현재 적용 중인 전략 (전략별 청산 규칙 적용)
    
    Returns:
        Tuple[str, str]: (action, reason) - 액션과 상세 사유
    """
    try:
        # 1. 최고 수익률 관리
        if max_profit_pct is None:
            max_profit_pct = max(profit_loss_pct, 0.0)
        
        # 🆕 [전략별 청산 규칙] 전략 시스템 로드
        strategy_exit_rules = None
        try:
            from trade.core.strategies import STRATEGY_EXIT_RULES
            strategy_exit_rules = STRATEGY_EXIT_RULES.get(current_strategy, STRATEGY_EXIT_RULES.get('trend'))
        except ImportError:
            pass
        
        # 🆕 [전략별 손익절 체크] 기본 청산 체크 전에 전략 규칙 우선 적용
        if strategy_exit_rules:
            # 전략별 익절 체크
            if profit_loss_pct >= strategy_exit_rules.take_profit_pct:
                reason = f"✅ 전략({current_strategy}) 익절 도달 ({profit_loss_pct:.1f}% >= {strategy_exit_rules.take_profit_pct}%)"
                return 'take_profit', reason
            
            # 전략별 손절 체크
            if profit_loss_pct <= -strategy_exit_rules.stop_loss_pct:
                reason = f"🛑 전략({current_strategy}) 손절 도달 ({profit_loss_pct:.1f}% <= -{strategy_exit_rules.stop_loss_pct}%)"
                return 'stop_loss', reason
            
            # 전략별 최대 보유 시간 체크
            if holding_hours >= strategy_exit_rules.max_holding_hours:
                reason = f"⏰ 전략({current_strategy}) 보유 시간 초과 ({holding_hours:.0f}h >= {strategy_exit_rules.max_holding_hours}h)"
                return 'sell', reason
            
            # 🆕 전략별 트레일링 스탑 체크
            if strategy_exit_rules.trailing_stop and max_profit_pct >= strategy_exit_rules.trailing_trigger_pct:
                trailing_stop_price = max_profit_pct - strategy_exit_rules.trailing_distance_pct
                if profit_loss_pct <= trailing_stop_price:
                    reason = f"📉 전략({current_strategy}) 트레일링 스탑 ({profit_loss_pct:.1f}% <= 최고 {max_profit_pct:.1f}% - {strategy_exit_rules.trailing_distance_pct}%)"
                    return 'sell', reason
        
        # 🔥 [공통 기준 적용] should_sell_holding_position 호출
        # 시그널 action이 아니라 signal_score + 보유 정보로 판단
        from trade.core.executor.strategy import should_sell_holding_position
        should_sell, sell_reason = should_sell_holding_position(
            signal_score=signal_score,
            profit_loss_pct=profit_loss_pct,
            max_profit_pct=max_profit_pct,
            holding_hours=holding_hours,
            tick_size=tick_size,
            current_price=current_price,
            trend_analysis=trend_analysis,
            signal_continuity=signal_continuity,  # 🆕 연속성 전달
            dynamic_influence=dynamic_influence   # 🆕 영향도 전달
        )
        
        if should_sell:
            # 손절/익절 구분
            if '손절' in sell_reason or '-10%' in sell_reason:
                return 'stop_loss', f"🚨 {sell_reason}"
            elif '익절' in sell_reason or '+50%' in sell_reason:
                return 'take_profit', f"🚨 {sell_reason}"
            return 'sell', f"🚨 {sell_reason}"
        
        # 🆕 market_adjustment 제거: 알파 가디언이 시장 상황별 meta_bias로 자동 학습하므로
        market_adjustment = 1.0
        
        # 💡 [Alpha Guardian] AI 판단은 참고용으로만 사용하며 결정권은 박탈
        ai_action = 'hold' # AI 판단을 의사결정 엔진에 전달하지 않음

        # 4. 학습된 매도 임계값 조회
        learned_threshold = None
        if LEARNED_EXIT_AVAILABLE and signal_pattern != 'unknown':
            learned_threshold = get_learned_sell_threshold(
                signal_pattern=signal_pattern,
                profit_loss_pct=profit_loss_pct,
                max_profit_pct=max_profit_pct,
                min_success_rate=0.5,
                min_samples=3
            )

        # 5. 공통 전략 엔진 호출 (최종 의사결정)
        final_action = decide_final_action(
            coin=coin,
            signal_score=signal_score,
            profit_loss_pct=profit_loss_pct,
            max_profit_pct=max_profit_pct,
            signal_pattern=signal_pattern,
            market_adjustment=market_adjustment,
            holding_hours=holding_hours,
            trend_analysis=trend_analysis,
            learned_threshold=learned_threshold,
            ai_decision='hold', # 💡 AI 결정 무시
            tick_size=tick_size,
            current_price=current_price,
            signal_continuity=signal_continuity,  # 🆕 연속성 전달
            dynamic_influence=dynamic_influence   # 🆕 영향도 전달
        )
            
        return final_action, ""  # 기본 액션은 사유 없음

    except Exception as e:
        print(f"⚠️ 의사결정 결합 오류 ({coin}): {e}")
        import traceback
        traceback.print_exc()
        return pure_action, f"오류: {e}"

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
        
        # 🎯 우선순위 결정 (중앙 관리 임계값 사용)
        t = get_thresholds()
        if avg_signal_score > t.priority_high:
            return 'high'
        elif avg_signal_score > t.priority_medium:
            return 'medium'
        elif avg_signal_score > t.priority_low:
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
            coin_performance = thompson_sampler.get_decision_engine_stats(coin)
            
            # 🆕 AI 의사결정 엔진으로 최종 검증
            signal_data = {
                'coin': coin,
                'action': 'sell',
                'signal_score': signal_score,
                'confidence': confidence,
                'risk_level': 'high' if abs(signal_score) > 0.7 else 'medium' if abs(signal_score) > 0.4 else 'low',
                'wave_phase': decision.get('wave_phase', 'unknown'),
                'integrated_direction': decision.get('integrated_direction', 'neutral')
            }
            
            # 🆕 진짜 시장 상황 분석 (Core 모듈 연동)
            real_market_context = get_market_context()
            
            market_context = {
                'trend': real_market_context.get('trend', 'neutral'),
                'volatility': 'high' if real_market_context.get('volatility', 0.02) > 0.05 else 'medium',
                'timestamp': int(time.time())
            }
            
            # 🛡️ 알파 가디언 판단 (매도 시점)
            if real_time_ai_decision_engine:
                ai_res = real_time_ai_decision_engine.make_trading_decision(
                    signal_data=signal_data,
                    current_price=current_price,
                    market_context=market_context,
                    coin_performance=coin_performance
                )
                # 🆕 딕셔너리 형태로 반환되므로 처리
                if isinstance(ai_res, dict):
                    ai_action = ai_res.get('decision', 'hold')
                    ai_score = ai_res.get('final_score', 0.0)
                    ai_reason = ai_res.get('reason', '알파 가디언 분석 완료')
                else:
                    ai_action = ai_res
                    ai_score = 0.0
                    ai_reason = '알파 가디언 분석 완료'
            else:
                ai_action = 'hold'
                ai_score = 0.0
                ai_reason = '알파 가디언 비활성화됨'
            
            # 🆕 알파 가디언 판단 로그 (매도 시점)
            print(f"   🛡️ [알파 가디언 매도 판단] {ai_action.upper()} (점수: {ai_score:.3f})")
            print(f"   💬 근거: {ai_reason}")
            
            # 🔒 [핵심 수정] 손절(stop_loss) 및 익절(take_profit)은 AI 의사결정 무시하고 무조건 실행!
            is_stop_loss = decision['action'] == 'stop_loss'
            is_take_profit = decision['action'] == 'take_profit'
            
            # 🆕 매도 실행 조건 확인
            should_execute_sell = False
            
            if is_stop_loss or is_take_profit:
                # 손절/익절은 무조건 실행 (AI 판단 무시)
                should_execute_sell = True
                if is_stop_loss:
                    print(f"🔒 {get_korean_name(coin)}: 손절 강제 실행! (AI 의사결정 무시)")
                elif is_take_profit:
                    print(f"🔒 {get_korean_name(coin)}: 익절 강제 실행!")
            elif decision['action'] == 'partial_sell':
                # 부분 매도는 항상 실행
                should_execute_sell = True
                print(f"✅ {get_korean_name(coin)}: 부분 매도 실행 (알파 가디언: {ai_action.upper()})")
            elif decision['action'] == 'sell':
                # 🆕 일반 매도는 알파 가디언 판단을 참고만 함 (결정권 박탈)
                should_execute_sell = True
                if ai_action == 'sell':
                    print(f"✅ {get_korean_name(coin)}: 알파 가디언 승인 매도 - {decision.get('reason', 'N/A')}")
                else:
                    print(f"⚠️ {get_korean_name(coin)}: 알파 가디언 매도 보류 권고했지만 전략 엔진 판단으로 매도 ({decision.get('reason', 'N/A')})")
            else:
                # 🆕 'hold' 등의 경우. 시그널이 아주 나쁘면 매도 보완 로직은 combine_signal_with_holding에서 처리됨.
                pass
            
            # 🆕 매도 실행
            if should_execute_sell:
                
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
                
                # 🆕 실전 매매 실시간 학습 (매도 시 지식 업데이트)
                real_time_learner.learn_from_trade(coin, profit_loss_pct)
                
                # Thompson Sampling 지식 즉시 업데이트 (수익/손실 패턴 학습)
                # 매도 품질 평가(Evaluator)와 연계 가능
                success_trade = profit_loss_pct > 0
                thompson_sampler.update_distribution(
                    pattern=coin, # 정밀 패턴 추출 가능 시 교체 추천
                    success=success_trade,
                    profit_pct=profit_loss_pct
                )
                
                # 🆕 시그널-매매 연결 (인과관계 추적)
                # SignalTradeConnector().connect_signal_to_trade(sig_info, trade_result)
                
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
                
                # 🆕 추세 패턴 저장 (학습용) - 히스토리는 학습기에서 정리
                if TRAJECTORY_ANALYZER_AVAILABLE:
                    try:
                        trajectory_analyzer = get_real_trajectory_analyzer()
                        # 추세 패턴 저장 (전체 히스토리 포함)
                        trajectory_analyzer.save_trajectory_pattern(
                            coin=coin,
                            entry_timestamp=ctx.get('entry_timestamp', int(time.time())),
                            exit_timestamp=int(time.time()),
                            peak_profit=ctx.get('max_profit_pct', profit_loss_pct),
                            final_profit=profit_loss_pct,
                            trajectory_type=ctx.get('action', 'sell'),
                            include_full_history=True  # 🆕 전체 히스토리 포함
                        )
                        # ⚠️ 히스토리 삭제는 학습기(virtual_trade_learner)에서 수행
                    except Exception as e:
                        print(f"⚠️ {coin} 추세 패턴 저장 오류: {e}")
                
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