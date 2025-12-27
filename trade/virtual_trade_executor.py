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
import os
import sys

# 현재 스크립트의 디렉토리를 path에 추가하여 같은 폴더 내의 모듈을 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 상위 디렉토리(프로젝트 루트)도 추가
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import logging
# 🔇 [JAX 로그 억제] TPU 초기화 실패 경고 등 불필요한 로그 숨김
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

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
# from trade.trade_executor import (
#    get_market_regime_manager,
#    get_portfolio_risk_manager,
# )

# 🆕 학습된 청산 파라미터 모듈
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
    from trade.core.trajectory_analyzer import get_virtual_trajectory_analyzer, TrendType
    TRAJECTORY_ANALYZER_AVAILABLE = True
except ImportError:
    TRAJECTORY_ANALYZER_AVAILABLE = False
    print("⚠️ Trajectory Analyzer 로드 실패 - 추세 분석 비활성화")

# 🆕 Thompson Sampling 공통 모듈
try:
    from trade.core.thompson import get_thompson_calculator, get_thompson_score as core_get_thompson_score
    THOMPSON_CORE_AVAILABLE = True
except ImportError:
    THOMPSON_CORE_AVAILABLE = False
    print("⚠️ Thompson 공통 모듈 로드 실패 - 로컬 구현 사용")

try:
    # 🆕 공통 마켓 분석기 사용 (한국어 이름 조회 등)
    from market.coin_market.market_analyzer import get_korean_name
except ImportError:
    print("⚠️ market_analyzer 로드 실패 - 기본 get_korean_name 사용")
# 🆕 한국어 이름 조회 유틸리티 import
try:
    from market.coin_market.market_analyzer import get_korean_name
except ImportError:
    print("⚠️ market_analyzer 모듈 로드 실패 - 한글 이름 미지원")
    def get_korean_name(symbol):
        return symbol

# 🆕 트레이딩 코어 매니저 (통합 관리)
try:
    from trade.core.manager import CoreManager
    CORE_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ 트레이딩 코어 매니저를 로드할 수 없습니다. (CoreManager)")
    CORE_MANAGER_AVAILABLE = False

# 🎰 Thompson Sampling 학습기 import
try:
    from virtual_trade_learner import VirtualTradingLearner, ThompsonSamplingLearner, STRATEGY_DB_PATH
    THOMPSON_SAMPLING_AVAILABLE = True
except ImportError:
    THOMPSON_SAMPLING_AVAILABLE = False
    print("⚠️ Thompson Sampling 학습기를 로드할 수 없습니다")

# 🆕 공통 코어 모듈 임포트 (두뇌 통합)
from trade.core.ai import AIDecisionEngine
from trade.core.risk import RiskManager, OutlierGuardrail
from trade.core.tracker import ActionPerformanceTracker, ContextRecorder, LearningFeedback
from trade.core.market import MarketAnalyzer
# 🆕 통합 의사결정 시스템 (가상/실전 로직 일치화)
try:
    from trade.core.judgement import JudgementSystem, DecisionType
    JUDGEMENT_AVAILABLE = True
except ImportError:
    JUDGEMENT_AVAILABLE = False
    print("⚠️ 통합 의사결정 시스템(JudgementSystem) 로드 실패")

from trade.signal_selector.core.types import SignalInfo, SignalAction

# 데이터베이스 경로
# 1. 캔들/시그널 DB (이전 호환성 유지)
# 기존: /workspace/data_storage -> 변경: 동적 경로 market/coin_market/data_storage
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT, 'market', 'coin_market', 'data_storage')
try:
    os.makedirs(_DEFAULT_DB_DIR, exist_ok=True)
except OSError:
    pass

# 🆕 trade_candles.db 우선 사용 로직 (trade_executor와 통일)
_trade_candles_path = os.path.join(_DEFAULT_DB_DIR, 'trade_candles.db')
# ⚠️ realtime_candles.db는 더 이상 사용하지 않음
_default_candle_db = _trade_candles_path

DB_PATH = os.getenv('RL_DB_PATH')
if not DB_PATH:
    # 환경변수가 없을 경우에만 기본 경로 사용 (하드코딩 방지)
    # 1. trade_candles.db (단일화)
    DB_PATH = _default_candle_db

# 2. 통합 트레이딩 시스템 DB (섀도우 + 실전 매매)
# 프로젝트 루트 찾기 (trade/virtual_trade_executor.py -> trade -> root)
# PROJECT_ROOT는 위에서 이미 정의됨

# 기본 경로: market/coin_market/data_storage/trading_system.db
DEFAULT_TRADING_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'trading_system.db')

# 환경변수가 있으면 우선 사용, 없으면 기본 경로 사용
TRADING_SYSTEM_DB_PATH = os.getenv('TRADING_DB_PATH')
if not TRADING_SYSTEM_DB_PATH:
    TRADING_SYSTEM_DB_PATH = DEFAULT_TRADING_DB_PATH

# 디렉토리가 없으면 생성 (안전장치)
try:
    os.makedirs(os.path.dirname(TRADING_SYSTEM_DB_PATH), exist_ok=True)
except Exception:
    pass

@dataclass
class VirtualPosition:
    """가상 포지션 정보"""
    coin: str  # symbol -> coin
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
    target_price: float = 0.0  # 🆕 예상 목표가 (동적 갱신됨)
    initial_target_price: float = 0.0  # 🆕 초기 예상 목표가 (진입 시점 기록용, 불변)
    pattern_type: str = 'none'  # 🆕 패턴 정보 저장 (학습용)
    entry_confidence: float = 0.0  # 🆕 진입 신뢰도

# 🆕 성능 업그레이드 시스템 클래스들
# 🆕 성능 업그레이드 시스템 클래스들 -> trade.core.* 로 이동됨
# (OutlierGuardrail, ActionPerformanceTracker, ContextRecorder 등)

# 🆕 진화형 AI 시스템 클래스들 -> trade.core.* 로 이동됨
# (AIDecisionEngine, MarketAnalyzer, RiskManager, LearningFeedback 등)

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
        
        # 🆕 [성능 최적화] SignalSelector 재사용 (지연 초기화)
        self.signal_selector = None
        
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
        
        # 🆕 트레이딩 코어 매니저 초기화 (AI, Risk, Market, Tracker 통합)
        if CORE_MANAGER_AVAILABLE:
            self.core = CoreManager()
            # 주요 모듈 바로가기 (기존 코드 호환성)
            self.ai_decision_engine = self.core.ai_engine
            self.market_analyzer = self.core.market_analyzer
            self.risk_manager = self.core.risk_manager
            self.learning_feedback = self.core.learning_feedback
            self.action_tracker = self.core.action_tracker
            self.context_recorder = self.core.context_recorder
            self.outlier_guardrail = self.core.outlier_guardrail
        else:
            # Fallback (CoreManager 없을 때)
            self.ai_decision_engine = AIDecisionEngine()
            self.market_analyzer = MarketAnalyzer()
            self.risk_manager = RiskManager()
            self.learning_feedback = LearningFeedback()
            self.action_tracker = ActionPerformanceTracker()
            self.context_recorder = ContextRecorder()
            self.outlier_guardrail = OutlierGuardrail()
        
        # 🧭 시장 국면 & 포트폴리오 리스크 매니저 (실전과 동일하게 재사용)
        # 🆕 CoreManager로 통합됨
        if hasattr(self, 'core') and self.core:
             self.market_regime_manager = self.core.market_analyzer
             # RiskManager가 PortfolioRiskManager 기능도 포함하는지 확인 필요하나 일단 할당
             self.portfolio_risk_manager = getattr(self.core, 'portfolio_risk_manager', self.core.risk_manager)
        else:
             self.market_regime_manager = self.market_analyzer
             self.portfolio_risk_manager = self.risk_manager

        self._regime_cache = {'ts': 0, 'data': None}
        
        # 🆕 [v2] 통합 의사결정 시스템 (DecisionMaker) 도입
        from trade.core.decision import DecisionMaker
        self.decision_maker = DecisionMaker(strategy_db_path=STRATEGY_DB_PATH)
        
        # 🆕 시그널-매매 연결 시스템
        self.signal_trade_connector = SignalTradeConnector()
        
        # 🆕 [Dashboard] 시스템 로거 초기화
        self._init_system_logger()

        # 🚀 트레일링 스탑을 위한 상태 추적
        # {coin_symbol: {'max_profit_pct': float}}
        self.position_tracking_state = {}
        
        # 🎰 Thompson Sampling - DecisionMaker 내부로 위임되어 직접 사용하지 않음
        # 하지만 기존 코드 호환성을 위해 None으로 두거나, DecisionMaker의 것을 참조
        self.thompson_sampler = self.decision_maker.thompson_sampler
        
        print("🚀 진화형 AI 가상 트레이더 초기화 완료 (DecisionMaker v2 탑재)")
        self.cache_ttl = 60  # 1분 캐시
        
        # 🆕 데이터베이스 경로 설정
        self.db_path = TRADING_SYSTEM_DB_PATH
        
        # 🆕 거래 테이블 생성
        self.create_trading_tables()
        
        # 🆕 DB 마이그레이션 (pattern_type 컬럼 추가)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(virtual_positions)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'pattern_type' not in columns:
                    print("🔄 virtual_positions 테이블에 pattern_type 컬럼 추가 중...")
                    conn.execute("ALTER TABLE virtual_positions ADD COLUMN pattern_type TEXT DEFAULT 'none'")
                    conn.commit()
                if 'target_price' not in columns:
                    print("🔄 virtual_positions 테이블에 target_price 컬럼 추가 중...")
                    conn.execute("ALTER TABLE virtual_positions ADD COLUMN target_price REAL DEFAULT 0.0")
                    conn.commit()
                if 'initial_target_price' not in columns:
                    print("🔄 virtual_positions 테이블에 initial_target_price 컬럼 추가 중...")
                    conn.execute("ALTER TABLE virtual_positions ADD COLUMN initial_target_price REAL DEFAULT 0.0")
                    conn.commit()
        except Exception as e:
            print(f"⚠️ DB 마이그레이션 오류: {e}")

        # 🆕 기존 포지션 로드
        self.load_positions_from_db()
        
        # 🆕 0원 진입가 포지션들 수정
        self._fix_zero_entry_prices()
        
        # 🆕 대상 코인 목록 (전체 코인)
        self.target_coins = self._get_all_available_coins()
        
        # 🆕 과도한 포지션 정리
        self._cleanup_excessive_positions()
        
        # 🆕 펀더멘탈 데이터 사전 로드 (CoreManager 위임)
        if CORE_MANAGER_AVAILABLE:
            self.core.prefetch_market_data()
        
        print(f"🚀 가상매매 시뮬레이터 시작")
    
    def _init_system_logger(self):
        """[Dashboard] 시스템 로거 초기화 (DB 연결 확인)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 테이블이 없으면 생성 (create_trading_tables에서 생성하지만 안전장치)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        level TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_status (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"⚠️ 시스템 로거 초기화 실패: {e}")

    def log_system_event(self, level: str, component: str, message: str, details: dict = None):
        """[Dashboard] 시스템 이벤트 기록 (봇의 생각 저장)"""
        try:
            timestamp = int(datetime.now().timestamp())
            details_json = json.dumps(details) if details else "{}"
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO system_logs (timestamp, level, component, message, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (timestamp, level, component, message, details_json))
                
                # 최신 로그 100개만 유지 (DB 비대화 방지)
                # 가끔씩 정리 (확률적 실행으로 성능 부하 분산)
                if timestamp % 100 == 0:
                    conn.execute("DELETE FROM system_logs WHERE id NOT IN (SELECT id FROM system_logs ORDER BY id DESC LIMIT 100)")
                conn.commit()
        except Exception:
            pass # 로깅 실패는 무시 (메인 로직 영향 최소화)

    def update_system_status(self, key: str, value: str):
        """[Dashboard] 시스템 상태 업데이트 (실시간 상태 공유)"""
        try:
            timestamp = int(datetime.now().timestamp())
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO system_status (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, value, timestamp))
                conn.commit()
        except Exception:
            pass
    
    def _validate_and_cleanup_positions(self):
        """포지션 유효성 검증 및 유의/상폐 종목 청산"""
        try:
            print("🧹 포지션 유효성 검증 및 청산 작업 시작...")
            
            # 1. 유의 종목 리스트 조회 (실패 시 환경변수 폴백)
            warning_list = []
            try:
                from market.coin_market.market_analyzer import get_market_warning_list
                warning_list = [c.upper() for c in get_market_warning_list()]
                if warning_list:
                    print(f"⚠️ 유의 종목 리스트({len(warning_list)}개): {', '.join(warning_list[:5])}...")
                else:
                    print("⚠️ 유의 종목 리스트가 비어있습니다.")
            except ImportError:
                print("⚠️ market_analyzer 모듈을 찾을 수 없어 유의 종목 확인을 건너뜁니다.")
            except Exception as e:
                print(f"⚠️ 유의 종목 조회 실패: {e}")
            
            # 🆕 폴백: 환경변수 FORCE_WARNING_COINS로 강제 지정 (쉼표 구분)
            if not warning_list:
                forced = os.getenv("FORCE_WARNING_COINS", "")
                if forced:
                    warning_list = [c.strip().upper() for c in forced.split(",") if c.strip()]
                    print(f"⚠️ 환경변수 기반 유의 종목 적용: {', '.join(warning_list)}")

            # 2. 포지션 검증 및 청산
            coins_to_remove = []
            current_timestamp = int(time.time())
            
            for coin, position in list(self.positions.items()):
                reason = None
                
                # A. 유의 종목 확인 (대소문자 구분 없이 확인)
                if coin.upper() in warning_list:
                    reason = 'caution_coin'
                    print(f"🚨 {coin}: 유의 종목 지정으로 인한 강제 청산")
                
                # B. 상폐/데이터 오류 확인 (가격이 0이거나 업데이트 안됨)
                elif position.current_price <= 0:
                    # 최신 가격 재조회 시도
                    latest_price = self._get_latest_price(coin)
                    if latest_price <= 0:
                        reason = 'invalid_price'
                        print(f"🚨 {coin}: 유효하지 않은 가격(0원)으로 인한 강제 청산")
                    else:
                        # 가격 복구
                        position.current_price = latest_price
                
                # C. 청산 실행
                if reason:
                    self._close_position(coin, position.current_price, current_timestamp, 'cleanup', reason)
                    coins_to_remove.append(coin)
            
            if coins_to_remove:
                print(f"✅ 포지션 정리 완료: {len(coins_to_remove)}개 청산됨 ({', '.join(coins_to_remove)})")
            else:
                print("✅ 정리할 포지션이 없습니다.")
            
        except Exception as e:
            print(f"⚠️ 포지션 정리 중 오류 발생: {e}")
            traceback.print_exc()

    def _cleanup_excessive_positions(self):
        """(Deprecated) 포지션 정리 - _validate_and_cleanup_positions 사용 권장"""
        self._validate_and_cleanup_positions()
    
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
                    # 🔧 trade_candles.db에서 조회
                    with sqlite3.connect(DB_PATH) as candles_conn:
                        candles_query = """
                            SELECT DISTINCT symbol as coin FROM candles 
                            WHERE timestamp > ?
                            ORDER BY symbol
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

    def _get_market_regime_info(self) -> Dict[str, any]:
        """시장 국면 정보를 5분 캐싱하여 재사용"""
        try:
            now = time.time()
            if self._regime_cache['data'] and (now - self._regime_cache['ts'] < 300):
                return self._regime_cache['data']
            # 🆕 인자 제거 (Core MarketAnalyzer는 인자 없이 자동 감지)
            info = self.market_regime_manager.analyze_market_regime()
            self._regime_cache = {'ts': now, 'data': info}
            return info
        except Exception as e:
            print(f"⚠️ 시장 국면 조회 오류: {e}")
            return {'score': 50, 'regime': 'neutral', 'breadth': 50.0}
    
    def can_open_position(self, coin: str) -> bool:
        """새로운 포지션 열기 가능 여부 확인 (무제한 포지션 + 🆕 펀더멘탈 체크 + 🆕 서킷 브레이커)"""
        # 1. 기본 체크: 이미 보유 중인지
        if coin in self.positions:
            return False

        # 🆕 2. [Circuit Breaker] 연속 손실 코인 차단 (TEMCO 사태 방지)
        # "잃었던 놈한테 또 잃는" 바보 같은 짓을 방지
        if self._check_consecutive_losses(coin):
            # print(f"⛔ {coin}: 연속 손실 과다로 인한 쿨다운 진입 (Circuit Breaker)")
            return False

        # 🆕 3. 펀더멘탈 체크 (CoreManager 위임)
        if CORE_MANAGER_AVAILABLE:
            try:
                fund_data = self.core.get_fundamental_data(coin)
                if fund_data:
                    score = self.core.calculate_fundamental_score(fund_data)
                    
                    # 🚨 필터링 제거: 점수가 낮아도 진입 허용 (정보 수집 목적)
                    # if score < 30:
                    #     return False
            except Exception:
                pass

        # 🆕 무제한 포지션: 이미 보유 중이지 않으면 가능
        return True
    
    def _check_consecutive_losses(self, coin: str) -> bool:
        """연속 손실 발생 여부 확인 (Circuit Breaker) - 🚫 비활성화됨"""
        return False  # 사용자의 요청으로 서킷브레이커 비활성화
        
        # try:
        #     with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
        #         # 최근 5회 거래 내역 조회
        #         cursor = conn.execute("""
        #             SELECT profit_loss_pct FROM virtual_trade_history
        #             WHERE coin = ?
        #             ORDER BY exit_timestamp DESC
        #             LIMIT 5
        #         """, (coin,))
        #         trades = [row[0] for row in cursor.fetchall()]
        #         
        #         if len(trades) < 3:
        #             return False
        #         
        #         # 최근 3회 연속 손실이고, 누적 손실이 -5% 이상이면 차단
        #         recent_3 = trades[:3]
        #         if all(p < 0 for p in recent_3) and sum(recent_3) < -5.0:
        #             return True
        #         
        #         return False
        # except Exception:
        #     return False
    
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
                AND coin NOT GLOB '[0-9]*' -- 🚨 숫자형 코인(오류 데이터) 제외
                AND (coin, timestamp) IN (
                    SELECT coin, MAX(timestamp)
                    FROM signals
                    WHERE timestamp >= ? AND interval = 'combined'
                    AND coin NOT GLOB '[0-9]*' -- 🚨 서브쿼리에서도 제외
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

                # 사용 가능한 코인 목록 조회 (숫자형 코인 제외)
                coins_query = """
                SELECT DISTINCT coin FROM signals 
                WHERE timestamp >= ? AND coin NOT GLOB '[0-9]*'
                ORDER BY coin
                """
                coins_df = pd.read_sql(coins_query, conn, params=(time_threshold,))
                
                if coins_df.empty:
                    print("⚠️ 사용 가능한 코인이 없습니다")
                    return []
                
                # 🆕 [성능 최적화] SignalSelector 인스턴스 재사용
                if self.signal_selector is None:
                    try:
                        from trade.strategy_signal_generator import SignalSelector
                        self.signal_selector = SignalSelector()
                        print("✅ SignalSelector 인스턴스 생성 완료 (재사용)")
                    except ImportError:
                        print("⚠️ SignalSelector 로드 실패")
                
                # 🎯 각 코인별로 최신 시그널 조회
                signals = []
                for coin in coins_df['coin'].head(batch_size):
                    try:
                        # 🆕 멀티 타임프레임 시그널 생성 시도
                        try:
                            # 재사용 가능한 selector 활용
                            if self.signal_selector:
                                mtf_signal = self.signal_selector.generate_multi_timeframe_signal(coin)
                            else:
                                # fallback (selector 없을 때)
                                from trade.strategy_signal_generator import SignalSelector
                                temp_selector = SignalSelector()
                                mtf_signal = temp_selector.generate_multi_timeframe_signal(coin)
                        except Exception:
                            # selector 오류 시 None 처리
                            mtf_signal = None
                        
                        if mtf_signal:
                            signals.append(mtf_signal)
                            print(f"  ✅ {coin}: 멀티 타임프레임 시그널 생성 성공")
                            # 쉐도우 트레이딩 판단은 _decide_buy_with_thompson으로 이동됨
                                
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
                integrated_strength=float(row['integrated_strength']),
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
                pattern_direction=row.get('pattern_direction', 'neutral'),
                target_price=row.get('target_price', 0.0)  # 🆕 예상 목표가 로드
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
                # 🚀 [성능 최적화] WAL 모드 활성화 (동시성 향상 및 파일 손상 방지)
                conn.execute("PRAGMA journal_mode=WAL;")
                
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
                        target_price REAL DEFAULT 0.0,
                        initial_target_price REAL DEFAULT 0.0,
                        pattern_type TEXT DEFAULT 'none',
                        entry_confidence REAL DEFAULT 0.0,
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
                        entry_confidence REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 🆕 컬럼 마이그레이션 (기존 DB 호환성)
                cursor = conn.cursor()
                
                # virtual_positions 컬럼 확인
                cursor.execute("PRAGMA table_info(virtual_positions)")
                v_cols = [c[1] for c in cursor.fetchall()]
                if 'entry_confidence' not in v_cols:
                    try:
                        cursor.execute("ALTER TABLE virtual_positions ADD COLUMN entry_confidence REAL DEFAULT 0.0")
                    except: pass
                if 'initial_target_price' not in v_cols:
                    try:
                        cursor.execute("ALTER TABLE virtual_positions ADD COLUMN initial_target_price REAL DEFAULT 0.0")
                    except: pass
                    
                # virtual_trade_history 컬럼 확인
                cursor.execute("PRAGMA table_info(virtual_trade_history)")
                h_cols = [c[1] for c in cursor.fetchall()]
                if 'entry_confidence' not in h_cols:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN entry_confidence REAL DEFAULT 0.0")
                    except: pass
                # 🆕 시장 조건 컬럼 추가 (전이학습 필터용)
                if 'volatility_regime' not in h_cols:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN volatility_regime TEXT DEFAULT 'medium'")
                    except: pass
                if 'volume_regime' not in h_cols:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN volume_regime TEXT DEFAULT 'medium'")
                    except: pass
                if 'market_regime' not in h_cols:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN market_regime TEXT DEFAULT 'neutral'")
                    except: pass
                
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
                
                # 🆕 [Dashboard] 시스템 로그 테이블 (봇의 생각 기록)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        level TEXT NOT NULL,  -- INFO, WARN, JUDGEMENT
                        component TEXT NOT NULL, -- Scanner, Executor, RiskManager
                        message TEXT NOT NULL,
                        details TEXT, -- JSON form
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 🆕 [Dashboard] 시스템 상태 테이블 (실시간 상태 공유)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_status (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                """)

                # 🆕 완료된 거래 테이블 (학습기 연동용, 여기서도 생성 보장)
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
                        target_price REAL DEFAULT 0.0,
                        is_learned BOOLEAN DEFAULT FALSE,
                        learning_episode INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, entry_timestamp, exit_timestamp)
                    )
                """)
                
                # 🆕🆕 가상매매 결정 테이블 (실전매매에서 읽기용)
                # - 가상매매에서 모든 분석(레짐, Thompson, 기대수익률 등) 후 결정 저장
                # - 실전매매에서는 이 테이블만 읽어서 매매 실행
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_trade_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        decision TEXT NOT NULL,           -- buy / sell / hold / skip
                        signal_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        current_price REAL NOT NULL,
                        target_price REAL DEFAULT 0.0,
                        expected_profit_pct REAL DEFAULT 0.0,
                        thompson_score REAL DEFAULT 0.0,
                        thompson_approved INTEGER DEFAULT 0,
                        regime_score REAL DEFAULT 0.5,
                        regime_name TEXT DEFAULT 'Neutral',
                        viability_passed INTEGER DEFAULT 0,
                        reason TEXT,
                        is_holding INTEGER DEFAULT 0,     -- 가상매매에서 보유 중인지
                        entry_price REAL DEFAULT 0.0,     -- 보유 중일 때 진입가
                        profit_loss_pct REAL DEFAULT 0.0, -- 보유 중일 때 수익률
                        processed INTEGER DEFAULT 0,      -- 실전매매에서 처리했는지
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, timestamp)
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
                # 🆕 가상매매 결정 테이블 인덱스
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_trade_decisions_coin ON virtual_trade_decisions(coin)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_trade_decisions_timestamp ON virtual_trade_decisions(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_virtual_trade_decisions_processed ON virtual_trade_decisions(processed)')
                
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
    
    def update_position(self, coin: str, current_price: float, timestamp: int, save_db: bool = True, execute_action: bool = True) -> Optional[str]:
        """포지션 업데이트 및 액션 결정 (execute_action=False이면 단순 업데이트만 수행)"""
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
            
            # [Debug] 보유 시간 계산 검증 (이상치 발견 시 로그)
            if position.holding_duration < 0:
                print(f"⚠️ {coin} 보유 시간 음수 발생: {position.holding_duration}s (Entry: {entry_timestamp}, Current: {current_timestamp})")
            elif position.holding_duration < 3600 and (current_timestamp - entry_timestamp) > 86400:
                 # 1일 이상 차이나는데 1시간 미만으로 계산된 경우
                 print(f"⚠️ {coin} 보유 시간 계산 의심: {position.holding_duration}s (Entry: {entry_timestamp}, Current: {current_timestamp})")
                 
        except (ValueError, TypeError) as e:
            print(f"⚠️ 타임스탬프 변환 오류 ({coin}): {e}")
            position.holding_duration = 0
            position.last_updated = int(datetime.now().timestamp())
        
        # 수익률 계산 (현재가가 유효할 때만)
        if position.entry_price > 0 and current_price > 0:
            profit_loss_pct = (current_price - position.entry_price) / position.entry_price * 100
            position.profit_loss_pct = profit_loss_pct
            
            # 최대 수익/손실 업데이트
            if profit_loss_pct > position.max_profit_pct:
                position.max_profit_pct = profit_loss_pct
            if profit_loss_pct < position.max_loss_pct:
                position.max_loss_pct = profit_loss_pct
        elif current_price <= 0:
            # 현재가가 0이거나 음수인 경우 수익률 업데이트 건너뜀 (데이터 오류 방지)
            pass
        else:
            print(f"⚠️ {coin}: 진입가가 0이므로 수익률 계산 불가")
            profit_loss_pct = 0.0
            position.profit_loss_pct = 0.0
        
        # 🆕 [Dynamic Target] 최신 시그널 기반 목표가 갱신
        # 7시간 전의 목표가를 계속 유지하는 것은 불합리함. 시장 상황에 따라 목표가도 변해야 함.
        try:
            current_signal = self._get_current_signal_info(coin)
            # 신규 목표가가 존재하고 유효할 때
            if current_signal and hasattr(current_signal, 'target_price') and current_signal.target_price > 0:
                # 목표가가 현재가보다 높을 때만 업데이트 (매수 관점 유지 시)
                if current_signal.target_price > current_price and current_signal.target_price != position.target_price:
                    # 기존 목표가 대비 변화율이 크면 로그 출력 (선택사항)
                    # if abs(current_signal.target_price - position.target_price) / position.target_price > 0.01:
                    #     print(f"  🎯 {coin}: 목표가 갱신 {position.target_price:.0f} -> {current_signal.target_price:.0f}")
                    position.target_price = current_signal.target_price
        except Exception:
            pass

        # 🆕 추세 분석 실행
        trend_analysis = None
        if TRAJECTORY_ANALYZER_AVAILABLE and execute_action:
            try:
                trajectory_analyzer = get_virtual_trajectory_analyzer()
                # 수익률 스냅샷 기록
                regime_info = self._get_market_regime_info()
                trajectory_analyzer.record_profit_snapshot(
                    coin=coin,
                    profit_pct=position.profit_loss_pct,
                    current_price=current_price,
                    entry_price=position.entry_price,
                    signal_score=position.entry_signal_score,
                    max_profit_pct=position.max_profit_pct,
                    min_profit_pct=position.max_loss_pct,
                    holding_hours=position.holding_duration / 3600,
                    market_regime=regime_info.get('regime', 'neutral')
                )
                # 추세 분석 실행
                trend_analysis = trajectory_analyzer.analyze_trend(coin, lookback=10)
            except Exception as e:
                pass  # 추세 분석 오류는 무시
        
        # 🆕 추세 분석 결과 출력 (실전매매와 동일, update_position에서)
        if trend_analysis and trend_analysis.history_count >= 3:
            trend_type_str = trend_analysis.trend_type.value
            reason_str = trend_analysis.reason
            # 추세 타입을 한글로 변환
            trend_map = {
                'up': '상승',
                'down': '하락',
                'sideways': '횡보',
                'peak_reversal': '고점반전',
                'strong_up': '강한상승',
                'strong_down': '강한하락',
                'neutral': '중립'
            }
            trend_kr = trend_map.get(trend_type_str, trend_type_str)
            
            # 신뢰도에 따른 표시
            if trend_analysis.confidence >= 0.7:
                confidence_icon = "🟢"
            elif trend_analysis.confidence >= 0.5:
                confidence_icon = "🟡"
            else:
                confidence_icon = "⚪"
            
            print(f"   📉 추세: {trend_kr} ({confidence_icon} {reason_str})")
            if trend_analysis.should_sell_early:
                print(f"   ⚠️ 조기 매도 권장!")
            elif trend_analysis.should_hold_strong:
                print(f"   💪 강한 홀딩 권장!")

        # 액션 결정 (execute_action이 True일 때만 수행)
        if execute_action:
            action = self._determine_position_action(position, current_price, timestamp, trend_analysis)
            
            if action in ['take_profit', 'stop_loss', 'sell', 'cleanup']:
                reason = ''
                if action == 'cleanup':
                    reason = 'stagnant_48h'
                    
                self._close_position(coin, current_price, timestamp, action, reason)
        else:
            action = 'hold' # 단순 업데이트만 수행 시 기본 액션
        
        # DB에 포지션 업데이트 (옵션)
        # 매도/청산이 아닐 때만 업데이트 (이미 close_position에서 처리됨)
        if save_db and action not in ['take_profit', 'stop_loss', 'sell', 'cleanup']:
            self.update_position_in_db(coin)
        
        return action
    
    def _determine_position_action(self, position: VirtualPosition, current_price: float, timestamp: int, trend_analysis=None) -> str:
        """포지션 액션 결정 (시그널 점수 중심 + 학습 기반 동적 리스크 관리 + 적응적 고급 지표 활용 + 추세 분석)"""
        
        # 🆕 [우선순위 0] 좀비 포지션 청산 (지루함 탈출 강화)
        holding_duration = timestamp - position.entry_timestamp
        holding_hours = holding_duration / 3600
        
        # ═══════════════════════════════════════════════════════════════
        # 🆕 [추세 분석 기반] 조기 매도/강한 홀딩 판단 (적극적 활용)
        # ═══════════════════════════════════════════════════════════════
        trend_sell_signal = False  # 추세 기반 매도 신호 플래그
        trend_sell_reason = ""
        trend_hold_signal = False  # 추세 기반 홀딩 신호 플래그
        trend_hold_reason = ""
        trend_pattern_adjustment = 0.0  # 학습된 추세 패턴 기반 조정값
        
        if trend_analysis is not None and trend_analysis.confidence >= 0.5:
            trend_type = trend_analysis.trend_type.value
            
            # 🆕 조기 매도 권장: should_sell_early가 True이면 무조건 매도 신호
            if trend_analysis.should_sell_early:
                trend_sell_signal = True
                trend_sell_reason = trend_analysis.reason
                self.log_system_event("JUDGEMENT", "Executor", 
                    f"⚠️ {position.coin} 추세 경고: {trend_sell_reason} (추세: {trend_type})", 
                    {"roi": position.profit_loss_pct, "max_roi": trend_analysis.max_profit_pct, "trend_type": trend_type})
                
                # 🆕 학습된 추세 패턴 조회 (Thompson Sampling)
                if self.thompson_sampler:
                    try:
                        trajectory_pattern = f"trajectory_{trend_type}"
                        pattern_stats = self.thompson_sampler.get_pattern_stats(trajectory_pattern)
                        
                        if pattern_stats:
                            success_rate = pattern_stats.get('success_rate', 0.5)
                            avg_profit = pattern_stats.get('avg_profit', 0.0)
                            
                            # 성공률이 낮거나 평균 수익이 음수면 더 적극적으로 매도
                            if success_rate < 0.4 or avg_profit < -2.0:
                                trend_pattern_adjustment = 0.25  # 임계값을 0.25 더 완화
                                self.log_system_event("LEARNING", "Executor", 
                                    f"📚 {position.coin} 학습된 추세 패턴: {trajectory_pattern} (성공률: {success_rate:.2f}, 평균수익: {avg_profit:.2f}%) → 더 적극적 매도",
                                    {"pattern": trajectory_pattern, "success_rate": success_rate, "avg_profit": avg_profit})
                    except Exception as e:
                        pass  # 학습 데이터 없으면 무시
            
            # 🆕 강한 홀딩 권장: 상승 추세 지속 또는 횡보 저점
            if trend_analysis.should_hold_strong:
                trend_hold_signal = True
                trend_hold_reason = trend_analysis.reason
                
                # 상승 추세에서는 매도 신호 무시하고 홀딩
                if trend_type in ['strong_up', 'up', 'recovering']:
                    self.log_system_event("JUDGEMENT", "Executor", 
                        f"💪 {position.coin} 추세 우선 홀딩 (상승 추세 지속: {trend_hold_reason})",
                        {"roi": position.profit_loss_pct, "trend_type": trend_type})
                    return 'hold'
                
                # 🆕 횡보 저점 근처: 홀딩 유지 (추매 기회)
                elif trend_type == 'sideways' and '저점' in trend_hold_reason:
                    self.log_system_event("JUDGEMENT", "Executor", 
                        f"💪 {position.coin} 횡보 저점 근처 - 홀딩 유지 (추매 기회: {trend_hold_reason})",
                        {"roi": position.profit_loss_pct, "trend_type": trend_type})
                    return 'hold'
        
        # ═══════════════════════════════════════════════════════════════
        # 🔒 [최우선] 안전장치 (절대 변경 불가 - 하드코딩)
        # ═══════════════════════════════════════════════════════════════
        if position.profit_loss_pct >= 50.0:
            self.log_system_event("JUDGEMENT", "Executor", f"🔒 {position.coin} 안전장치 익절 (+50% 도달)", {"roi": position.profit_loss_pct})
            return 'take_profit'
        
        if position.profit_loss_pct <= -10.0:
            self.log_system_event("JUDGEMENT", "Executor", f"🔒 {position.coin} 안전장치 손절 (-10% 도달)", {"roi": position.profit_loss_pct})
            return 'stop_loss'
        
        # ═══════════════════════════════════════════════════════════════
        # 🎓 [학습 기반] 청산 판단 (virtual_trade_learner에서 학습한 기법 적용 + 시장 상황 반영)
        # ═══════════════════════════════════════════════════════════════
        if LEARNED_EXIT_AVAILABLE:
            try:
                # 🎯 시장 상황 조회 (매도 결정에 반영)
                regime_info = self._get_market_regime_info()
                market_regime = regime_info.get('regime', 'Neutral')
                market_trend = regime_info.get('trend', 'neutral')
                
                # 🎯 시장 상황에 따른 매도 조정 계수 계산
                regime_lower = market_regime.lower() if market_regime else 'neutral'
                is_bearish = 'bearish' in regime_lower or market_trend == 'bearish'
                is_extreme_bearish = 'extreme_bearish' in regime_lower
                is_bullish = 'bullish' in regime_lower or market_trend == 'bullish'
                
                # 매도 조정 계수 (하락장일수록 더 적극적으로 매도)
                if is_extreme_bearish:
                    market_adjustment = 0.7  # 30% 완화 (더 쉽게 매도)
                elif is_bearish:
                    market_adjustment = 0.85  # 15% 완화
                elif is_bullish:
                    market_adjustment = 1.2  # 20% 강화 (더 확실한 신호에서만)
                else:
                    market_adjustment = 1.0  # 중립
                
                # 패턴 추출 (현재 시그널에서)
                current_signal = self._get_current_signal_info(position.coin)
                signal_pattern = self._extract_signal_pattern(current_signal) if current_signal else 'unknown'
                entry_volatility = getattr(position, 'entry_volatility', 0.02)
                max_profit_pct = getattr(position, 'max_profit_pct', position.profit_loss_pct)
                
                # 🎓 학습 기반 익절 체크 (시장 상황 조정 적용)
                should_tp, tp_reason = should_take_profit(
                    profit_pct=position.profit_loss_pct,
                    max_profit_pct=max_profit_pct,
                    signal_pattern=signal_pattern,
                    entry_volatility=entry_volatility,
                    market_adjustment=market_adjustment
                )
                if should_tp:
                    self.log_system_event("JUDGEMENT", "Executor", 
                        f"🎓 {position.coin} 학습 기반 익절 ({tp_reason}, 조정: {market_adjustment:.2f}x)", 
                        {"roi": position.profit_loss_pct, "max_roi": max_profit_pct})
                    return 'take_profit'
                
                # 🎓 학습 기반 손절 체크 (시장 상황 조정 적용)
                should_sl, sl_reason = should_stop_loss(
                    profit_pct=position.profit_loss_pct,
                    signal_pattern=signal_pattern,
                    entry_volatility=entry_volatility,
                    holding_hours=holding_hours,
                    market_adjustment=market_adjustment
                )
                if should_sl:
                    self.log_system_event("JUDGEMENT", "Executor", 
                        f"🎓 {position.coin} 학습 기반 손절 ({sl_reason}, 조정: {market_adjustment:.2f}x)", 
                        {"roi": position.profit_loss_pct, "holding_hours": holding_hours})
                    return 'stop_loss'
                    
            except Exception as e:
                # 학습 기반 청산 오류 시 기본 로직으로 fallback
                pass
        
        # ═══════════════════════════════════════════════════════════════
        # [기존 로직] 좀비 포지션 청산
        # ═══════════════════════════════════════════════════════════════
        
        # 1. 48시간 이상: 수익이 없으면(0% 이하) 무조건 청산 (희망 고문 종료)
        if holding_hours >= 48 and position.profit_loss_pct <= 0.0:
            self.log_system_event("JUDGEMENT", "Executor", f"{position.coin} 48시간 보유 중. 수익 없음. 청산 결정.", {"holding_hours": holding_hours})
            return 'cleanup'
            
        # 2. 24시간 이상: 지지부진(-2% ~ +2%)하면 청산 (단, 초강력 시그널은 제외)
        if holding_hours >= 24 and -2.0 <= position.profit_loss_pct <= 2.0:
            current_signal_check = self._get_current_signal_info(position.coin)
            # 시그널 점수가 0.7 이상(매우 강력)인 경우에만 한 번 더 기회를 줌
            if not current_signal_check or current_signal_check.signal_score < 0.7:
                 self.log_system_event("JUDGEMENT", "Executor", f"{position.coin} 24시간 지지부진. 기회비용 확보 위해 정리.", {"roi": position.profit_loss_pct})
                 return 'cleanup'

        # 🆕 현재 시그널 정보 조회
        current_signal = self._get_current_signal_info(position.coin)
        
        if not current_signal:
            # 시그널이 없으면 기본 홀딩
            return 'hold'
        
        # 🎯 핵심: 시그널 점수가 주요 기준
        signal_score = current_signal.signal_score
        confidence = current_signal.confidence

        # 🆕 [Reality Check] 포지션 상황에 따른 점수 보정 (인지 부조화 해결)
        # 시그널은 매수라고 하지만, 내 계좌가 파란불(-3% 이상)이라면 시그널을 의심해야 함
        if position.profit_loss_pct < -3.0:
            # 손실 중일 때는 매수 관점을 약화시키고 매도 관점을 강화
            if signal_score > 0:
                signal_score *= 0.5 # 긍정적 뷰 반토막
            else:
                signal_score -= 0.1 # 부정적 뷰 강화
            confidence += 0.1 # 위기 상황이므로 판단에 대한 가중치 증가
            
            # 로그 기록 (고민 중)
            if timestamp % 300 == 0: # 5분에 한번만 기록
                self.log_system_event("WARN", "RiskManager", f"{position.coin} 손실 구간(-3%↓). 시그널 신뢰도 하향 조정.", {"origin_score": current_signal.signal_score, "adj_score": signal_score})
            
        if position.profit_loss_pct < -5.0:
            # 손실이 커지면 더 비관적으로 판단
            signal_score -= 0.2
            if timestamp % 300 == 0:
                self.log_system_event("WARN", "RiskManager", f"⚠️ {position.coin} 손실 -5% 돌파. 비관적 관점 강화.", {"roi": position.profit_loss_pct})


        # 🆕 추세 정보를 시그널 점수에 직접 반영
        original_signal_score = signal_score
        
        if trend_analysis is not None and trend_analysis.confidence >= 0.5:
            # 조기 매도 권장 시 시그널 점수를 더 부정적으로 조정
            if trend_analysis.should_sell_early:
                # 🆕 횡보 고점 근처: 더 적극적으로 매도 (고점에서 이익 실현)
                if trend_type == 'sideways' and '고점' in trend_analysis.reason:
                    # 횡보 고점에서는 수익 보호보다 이익 실현 우선
                    if signal_score > 0:
                        signal_score -= 0.25  # 양수 시그널을 더 약화
                    else:
                        signal_score -= 0.2  # 음수 시그널을 더 강화
                    if timestamp % 300 == 0:  # 5분에 한번만 로그
                        self.log_system_event("JUDGEMENT", "Executor", 
                            f"📉 {position.coin} 횡보 고점 근처 - 적극 매도 고려: {original_signal_score:.3f} → {signal_score:.3f} ({trend_analysis.reason})",
                            {"original_score": original_signal_score, "adjusted_score": signal_score, "reason": trend_analysis.reason})
                else:
                    # 일반 조기 매도
                    if signal_score > 0:
                        signal_score -= 0.2  # 양수 시그널을 약화
                    else:
                        signal_score -= 0.15  # 음수 시그널을 강화
                    if timestamp % 300 == 0:  # 5분에 한번만 로그
                        self.log_system_event("JUDGEMENT", "Executor", 
                            f"📉 {position.coin} 추세 기반 시그널 점수 조정: {original_signal_score:.3f} → {signal_score:.3f}",
                            {"original_score": original_signal_score, "adjusted_score": signal_score})
            
            # 강한 홀딩 권장 시 시그널 점수를 더 긍정적으로 조정
            elif trend_analysis.should_hold_strong:
                # 시그널 점수가 음수면 더 긍정적으로, 양수면 더 강하게
                if signal_score < 0:
                    signal_score += 0.2  # 음수 시그널을 약화
                else:
                    signal_score += 0.1  # 양수 시그널을 강화
                if timestamp % 300 == 0:  # 5분에 한번만 로그
                    self.log_system_event("JUDGEMENT", "Executor", 
                        f"📈 {position.coin} 추세 기반 시그널 점수 조정: {original_signal_score:.3f} → {signal_score:.3f}",
                        {"original_score": original_signal_score, "adjusted_score": signal_score})
        
        # 🆕 적응적 고급 지표 분석
        adaptive_analysis = self._analyze_adaptive_indicators(current_signal)
        
        # 🆕 학습 기반 동적 손절 강도 계산
        stop_loss_strength = self._calculate_adaptive_stop_loss_strength(position, current_signal)
        
        # 🆕 AI 의사결정 엔진 활용 (조정된 시그널 점수 반영을 위해 시그널 복사)
        coin_performance = self.learning_feedback.get_coin_learning_data(position.coin)
        market_context = self._get_market_context()
        
        # 시그널 점수가 조정되었으면 임시로 시그널 객체 수정
        if signal_score != original_signal_score:
            from copy import copy
            adjusted_signal = copy(current_signal)
            adjusted_signal.signal_score = signal_score
            ai_decision = self.ai_decision_engine.make_trading_decision(
                adjusted_signal, current_price, market_context, coin_performance
            )
        else:
            ai_decision = self.ai_decision_engine.make_trading_decision(
                current_signal, current_price, market_context, coin_performance
            )
        
        # 🆕 [v2] 통합 Judgement System 기반 매도 판단 (청산 로직 강화, CoreManager 위임)
        if CORE_MANAGER_AVAILABLE:
            # Thompson 확률 조회
            thompson_prob = 0.5
            if self.thompson_sampler:
                pattern = self._extract_signal_pattern(current_signal)
                thompson_prob, _ = self.thompson_sampler.sample_success_rate(pattern)
            
            # CoreManager 평가
            judgement = self.core.evaluate_signal(
                signal_info=current_signal,
                thompson_prob=thompson_prob,
                mode='simulation'
            )
            
            # KILL 판정이면 강력 매도
            if judgement.decision == DecisionType.KILL:
                if position.profit_loss_pct < -2.0:
                    return 'stop_loss'
                return 'sell'
        
        # 🆕 시그널 점수 기반 매매 결정 (핵심 로직)
        source_type = getattr(current_signal, 'source_type', 'quant')
        
        # 🆕 [학습 기반 매도] 패턴별 최적 매도 시그널 점수 임계값 조회
        signal_pattern = self._extract_signal_pattern(current_signal) if current_signal else 'unknown'
        learned_threshold = None
        
        if LEARNED_EXIT_AVAILABLE and signal_pattern != 'unknown':
            # 학습된 최적 임계값 조회 (성공률 50% 이상, 샘플 3회 이상)
            learned_threshold = get_learned_sell_threshold(
                signal_pattern=signal_pattern,
                profit_loss_pct=position.profit_loss_pct,
                min_success_rate=0.5,
                min_samples=3
            )
        
        # 1. 하이브리드 시그널인 경우 AI의 매도 의견을 적극 수용
        if source_type == 'hybrid' and ai_decision == 'SELL':
            return 'sell'
        
        # 🆕 추세 경고가 있으면 매도 임계값을 더 보수적으로 조정
        trend_adjustment = 0.0
        if trend_sell_signal:
            # 기본 조정값 + 학습된 패턴 기반 추가 조정
            trend_adjustment = 0.15 + trend_pattern_adjustment  # 기본 0.15 + 패턴 기반 추가
            self.log_system_event("JUDGEMENT", "Executor", 
                f"⚠️ {get_korean_name(position.coin)} 추세 경고 반영: 매도 임계값 {trend_adjustment:.2f} 완화 (기본: 0.15, 패턴: {trend_pattern_adjustment:.2f})",
                {"trend_adjustment": trend_adjustment, "pattern_adjustment": trend_pattern_adjustment})
        
        # 🆕 [학습 기반 매도] 학습된 임계값이 있으면 우선 사용 (조정된 시그널 점수 사용)
        if learned_threshold is not None:
            # 학습된 임계값에 추세 경고 반영
            adjusted_learned_threshold = learned_threshold + trend_adjustment
            if signal_score < adjusted_learned_threshold:  # 🆕 조정된 시그널 점수 사용
                self.log_system_event("JUDGEMENT", "Executor", 
                    f"📚 {get_korean_name(position.coin)}: 학습 기반 매도 (패턴: {signal_pattern}, "
                    f"임계값: {learned_threshold:.2f} → 조정: {adjusted_learned_threshold:.2f}, "
                    f"현재: {signal_score:.2f} (원본: {original_signal_score:.2f}), 추세경고: {trend_sell_reason}",
                    {"pattern": signal_pattern, "threshold": adjusted_learned_threshold, "score": signal_score})
                return 'sell'
        else:
            # 학습 데이터가 없으면 기본 임계값 사용 (추세 경고 반영)
            base_threshold = -0.5
            adjusted_threshold = base_threshold + trend_adjustment
            if signal_score < adjusted_threshold and confidence > 0.6:  # 🆕 조정된 시그널 점수 사용
                # 강한 매도 시그널
                if trend_sell_signal:
                    self.log_system_event("JUDGEMENT", "Executor", 
                        f"📉 {get_korean_name(position.coin)}: 매도 (시그널: {signal_score:.2f} (원본: {original_signal_score:.2f}), 추세경고: {trend_sell_reason})",
                        {"score": signal_score, "original_score": original_signal_score})
                return 'sell'
            
            elif signal_score < -0.3 and (confidence > 0.5 or ai_decision == 'SELL'):  # 🆕 조정된 시그널 점수 사용
                # 매도 시그널 (AI 동의 시 완화)
                return 'sell'
            
            elif signal_score < -0.2 and confidence > 0.4:  # 🆕 조정된 시그널 점수 사용
                # 약한 매도 시그널 (손절 고려)
                if position.profit_loss_pct < -3.0:  # 손실이 있는 경우
                    return 'stop_loss'
                else:
                    # AI가 매도 의견이면 매도
                    if ai_decision == 'SELL':
                        return 'sell'
                    return 'hold'
        
        if signal_score < 0.0 and position.profit_loss_pct > 5.0:
            # 수익이 있지만 시그널이 약해진 경우 (익절 고려)
            if ai_decision == 'SELL':
                 return 'take_profit'
            return 'take_profit'
        
        if signal_score < 0.0 and position.profit_loss_pct < -5.0:
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
        """학습 기반 동적 손절 강도 계산 (Core RiskManager 위임)"""
        try:
            stop_loss_performance = self._analyze_stop_loss_performance(position.coin)
            market_volatility = self._get_market_volatility()
            
            return self.risk_manager.calculate_adaptive_stop_loss_strength(
                position.coin, signal, market_volatility, stop_loss_performance
            )
        except Exception as e:
            print(f"⚠️ 동적 손절 강도 계산 오류: {e}")
            return 50.0
    
    def _get_market_context(self) -> Dict:
        """🆕 시장 상황 분석 (Market Regime 기반 통합)"""
        try:
            # 🆕 Market Regime 결과 활용 (거래량 상위 40% 코인 기준)
            regime_info = self._get_market_regime_info()
            
            # 레짐 → 트렌드 변환
            regime = regime_info.get('regime', 'Neutral').lower()
            score = regime_info.get('score', 0.5)
            volatility = regime_info.get('volatility', 0.02)
            
            # 트렌드 분류 (레짐 기반)
            if 'bullish' in regime or 'bull' in regime:
                trend = 'bullish'
            elif 'bearish' in regime or 'bear' in regime:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'trend': trend,
                'volatility': volatility,
                'regime': regime_info.get('regime', 'Neutral'),
                'score': score
            }
            
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02, 'regime': 'Neutral', 'score': 0.5}
    
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
                    harmonic_patterns=row.get('harmonic_patterns', 'none'),
                    candlestick_patterns=row.get('candlestick_patterns', 'none'),
                    market_structure=row.get('market_structure', 'unknown'),
                    flow_level_meta=row.get('flow_level_meta', 'unknown'),
                    pattern_direction=row.get('pattern_direction', 'neutral'),
                    target_price=row.get('target_price', 0.0),  # 🆕 예상 목표가 로드
                    source_type=row.get('source_type', 'quant')  # 🆕 소스 타입 로드
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
    
    def _close_position(self, coin: str, price: float, timestamp: int, action: str, reason: str = ''):
        """포지션 종료"""
        if coin not in self.positions:
            return
            
        # 🚨 가격 데이터 오류 방지 (0원 매도 방지)
        if price <= 0:
            print(f"⚠️ {coin}: 매도 가격 오류 ({price}원) - 매도 취소")
            return

        # 🆕 [Realistic Friction] 슬리피지(Slippage) 0.1% 적용 (현실적 마찰력)
        # 시장가 매도 시 호가 갭과 수수료를 고려하여 체결가를 0.1% 낮게 잡음
        exit_price = price * 0.999

        position = self.positions[coin]
        
        # 🚨 수익률 재계산 (정확한 계산 보장)
        if position.entry_price > 0:
            profit_loss_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
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
            'exit_price': exit_price,
            'quantity': position.quantity,
            'profit_loss_pct': profit_loss_pct,
            'action': action,
            'entry_timestamp': entry_timestamp,
            'exit_timestamp': exit_timestamp,
            'holding_duration': actual_holding_duration,
            'entry_signal_score': position.entry_signal_score,
            'target_price': getattr(position, 'target_price', 0.0),  # 🆕 예상 목표가 포함
            'initial_target_price': getattr(position, 'initial_target_price', getattr(position, 'target_price', 0.0)),  # 🆕 초기 목표가
            'signal_pattern': getattr(position, 'pattern_type', 'unknown'), # 🆕 패턴 정보 추가
            'entry_confidence': getattr(position, 'entry_confidence', 0.0), # 🆕 신뢰도 추가
            'reason': reason # 🆕 사유 추가
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
            'profit_loss_pct': profit_loss_pct,
            'reason': reason # 🆕 사유 추가
        }
        self.context_recorder.record_trade_context(trade_id, context)
        
        # 🆕 학습 피드백에 거래 결과 기록
        self.learning_feedback.record_trade_result(coin, {
            'trade_record': trade_record,
            'context': context,
            'action_performance': self.action_tracker.get_action_performance(action)
        })
        
        # 🆕 시그널-매매 연결 (🔧 수정: 현재 시그널 로드하여 패턴 추출)
        current_signal = self._get_current_signal_info(coin)
        exit_signal_score = 0.0  # 기본값
        if current_signal:
            signal_pattern = self._extract_signal_pattern_for_feedback(current_signal)
            exit_signal_score = current_signal.signal_score  # 🆕 매도 시점의 시그널 점수 기록
            self.signal_trade_connector.connect_signal_to_trade(current_signal, trade_record)
        else:
            # 시그널이 없으면 기본 패턴 사용
            signal_pattern = f"{coin}_score_{int(position.entry_signal_score * 100)}"
        
        # 🆕 매도 시점의 시그널 점수를 trade_record에 추가
        trade_record['exit_signal_score'] = exit_signal_score
        
        # 🎰 Thompson Sampling 분포 업데이트 (강화학습 핵심!)
        self._update_thompson_on_trade_close(coin, signal_pattern, success, profit_loss_pct)
        
        # DB에 거래 히스토리 저장
        self.save_trade_to_db(trade_record)
        
        # 🆕 학습용 completed_trades 테이블에도 저장
        self.save_completed_trade_for_learning(trade_record)
        
        # 🆕 가상매매 피드백 테이블에 상세 정보 저장
        self.save_trade_feedback_for_learning(trade_record)
        
        # 🆕 추세 패턴 저장 및 히스토리 정리 (매도 완료 시)
        if TRAJECTORY_ANALYZER_AVAILABLE:
            try:
                trajectory_analyzer = get_virtual_trajectory_analyzer()
                # 추세 패턴 저장 (전체 히스토리 포함)
                trajectory_analyzer.save_trajectory_pattern(
                    coin=coin,
                    entry_timestamp=entry_timestamp,
                    exit_timestamp=exit_timestamp,
                    peak_profit=position.max_profit_pct,
                    final_profit=profit_loss_pct,
                    trajectory_type=action,
                    pattern_data={
                        'signal_pattern': signal_pattern,
                        'holding_hours': actual_holding_duration / 3600,
                        'entry_confidence': getattr(position, 'entry_confidence', 0.0)
                    },
                    include_full_history=True  # 🆕 전체 히스토리 포함
                )
                # ⚠️ 히스토리 삭제는 학습기(virtual_trade_learner)에서 수행
            except Exception as e:
                pass  # 추세 패턴 저장 오류는 무시
        
        # DB에서 포지션 삭제
        self.delete_position_from_db(coin)
        
        # 🆕 DB 삭제 확인 및 재시도 (좀비 포지션 방지)
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                check = conn.execute("SELECT 1 FROM virtual_positions WHERE coin = ?", (coin,)).fetchone()
                if check:
                    print(f"⚠️ {coin}: DB 삭제 실패 (좀비 포지션 감지) - 강제 삭제 재시도")
                    conn.execute("DELETE FROM virtual_positions WHERE coin = ?", (coin,))
                    conn.commit()
        except Exception as e:
            print(f"⚠️ DB 삭제 확인 중 오류 ({coin}): {e}")

        # 포지션 제거
        if coin in self.positions:
            del self.positions[coin]
        
        action_name = {
            'take_profit': "익절",
            'stop_loss': "손절", 
            'sell': "매도",
            'cleanup': "청산"
        }.get(action, "매도")
        
        print(f"🆕 포지션 종료: {get_korean_name(coin)} {action_name} @ {self._format_price(price)}원 (수익률: {profit_loss_pct:+.2f}%) {reason}")
    
    def save_position_to_db(self, coin: str):
        """포지션을 DB에 저장"""
        try:
            position = self.positions[coin]
            # 🚨 숫자형 코인 심볼 안전 처리
            safe_coin = str(coin)
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO virtual_positions 
                    (coin, entry_price, quantity, entry_timestamp, entry_signal_score, 
                     current_price, profit_loss_pct, holding_duration, max_profit_pct, 
                     max_loss_pct, stop_loss_price, take_profit_price, last_updated,
                     target_price, pattern_type, initial_target_price, entry_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    safe_coin, position.entry_price, position.quantity, position.entry_timestamp,
                    position.entry_signal_score, position.current_price, position.profit_loss_pct,
                    position.holding_duration, position.max_profit_pct, position.max_loss_pct,
                    position.stop_loss_price, position.take_profit_price, position.last_updated,
                    getattr(position, 'target_price', 0.0), getattr(position, 'pattern_type', 'none'),
                    getattr(position, 'initial_target_price', 0.0), getattr(position, 'entry_confidence', 0.0)
                ))
                conn.commit()
        except Exception as e:
            print(f"❌ DB 저장 오류 (save_position_to_db): {e}")
    
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
    
    def save_trade_decision(self, decision_data: Dict):
        """🆕🆕 가상매매 결정을 DB에 저장 (실전매매에서 읽기용)
        
        decision_data:
            - coin: 코인 심볼
            - timestamp: 결정 시간
            - decision: 'buy' / 'sell' / 'hold' / 'skip'
            - signal_score: 시그널 점수
            - confidence: 신뢰도
            - current_price: 현재가
            - target_price: 목표가
            - expected_profit_pct: 기대 수익률
            - thompson_score: Thompson Sampling 점수
            - thompson_approved: Thompson Sampling 승인 여부
            - regime_score: 시장 레짐 점수
            - regime_name: 시장 레짐 이름
            - viability_passed: 기대수익률 필터 통과 여부
            - reason: 결정 사유
            - is_holding: 가상매매에서 보유 중인지
            - entry_price: 보유 중일 때 진입가
            - profit_loss_pct: 보유 중일 때 수익률
        """
        try:
            coin = decision_data.get('coin')
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕🆕 [핵심] 해당 코인의 이전 미처리 결정 삭제 (오래된 'buy' 결정 무효화)
                # → 같은 코인에 대해 최신 결정만 유지되도록 보장
                conn.execute("""
                    DELETE FROM virtual_trade_decisions 
                    WHERE coin = ? AND processed = 0
                """, (coin,))
                
                conn.execute("""
                    INSERT INTO virtual_trade_decisions (
                        coin, timestamp, decision, signal_score, confidence, current_price,
                        target_price, expected_profit_pct, thompson_score, thompson_approved,
                        regime_score, regime_name, viability_passed, reason,
                        is_holding, entry_price, profit_loss_pct, processed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    decision_data.get('coin'),
                    decision_data.get('timestamp', int(datetime.now().timestamp())),
                    decision_data.get('decision', 'skip'),
                    decision_data.get('signal_score', 0.0),
                    decision_data.get('confidence', 0.0),
                    decision_data.get('current_price', 0.0),
                    decision_data.get('target_price', 0.0),
                    decision_data.get('expected_profit_pct', 0.0),
                    decision_data.get('thompson_score', 0.0),
                    1 if decision_data.get('thompson_approved', False) else 0,
                    decision_data.get('regime_score', 0.5),
                    decision_data.get('regime_name', 'Neutral'),
                    1 if decision_data.get('viability_passed', False) else 0,
                    decision_data.get('reason', ''),
                    1 if decision_data.get('is_holding', False) else 0,
                    decision_data.get('entry_price', 0.0),
                    decision_data.get('profit_loss_pct', 0.0)
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ 가상매매 결정 저장 오류: {e}")
    
    def save_trade_to_db(self, trade_record: Dict):
        """거래 내역을 DB에 저장"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 signal_pattern 컬럼이 없으면 추가
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(virtual_trade_history)")
                columns = [col[1] for col in cursor.fetchall()]
                if columns and 'signal_pattern' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN signal_pattern TEXT")
                    except:
                        pass
                if columns and 'initial_target_price' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN initial_target_price REAL DEFAULT 0.0")
                    except:
                        pass
                if columns and 'entry_confidence' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN entry_confidence REAL DEFAULT 0.0")
                    except:
                        pass
                if columns and 'exit_signal_score' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN exit_signal_score REAL DEFAULT 0.0")
                    except:
                        pass
                # 🆕 시장 조건 컬럼 추가 (전이학습 필터용)
                if 'volatility_regime' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN volatility_regime TEXT DEFAULT 'medium'")
                    except: pass
                if 'volume_regime' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN volume_regime TEXT DEFAULT 'medium'")
                    except: pass
                if 'market_regime' not in columns:
                    try:
                        cursor.execute("ALTER TABLE virtual_trade_history ADD COLUMN market_regime TEXT DEFAULT 'neutral'")
                    except: pass
                
                # 🆕 시장 조건 정보 조회
                regime_info = self._get_market_regime_info()
                volatility_regime = 'medium'
                volume_regime = 'medium'
                market_regime = regime_info.get('regime', 'Neutral')
                
                # 변동성 레짐 판단 (간단한 버전)
                volatility = regime_info.get('volatility', 0.02)
                if volatility > 0.03:
                    volatility_regime = 'high'
                elif volatility < 0.01:
                    volatility_regime = 'low'
                
                # 거래량 레짐 판단 (간단한 버전)
                volume_ratio = regime_info.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:
                    volume_regime = 'high'
                elif volume_ratio < 0.7:
                    volume_regime = 'low'
                
                # 🆕 signal_pattern 저장 추가
                conn.execute("""
                    INSERT INTO virtual_trade_history
                    (coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                     profit_loss_pct, action, holding_duration, entry_signal_score, exit_signal_score, quantity, signal_pattern, initial_target_price, entry_confidence,
                     volatility_regime, volume_regime, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record['coin'], trade_record['entry_timestamp'], trade_record['exit_timestamp'],
                    trade_record['entry_price'], trade_record['exit_price'], trade_record['profit_loss_pct'],
                    trade_record['action'], trade_record['holding_duration'], trade_record['entry_signal_score'],
                    trade_record.get('exit_signal_score', 0.0),  # 🆕 매도 시점의 시그널 점수 (정규화 후 기본값 0.0)
                    trade_record.get('quantity', 1.0),
                    trade_record.get('signal_pattern', 'unknown'),
                    trade_record.get('initial_target_price', 0.0),
                    trade_record.get('entry_confidence', 0.0),
                    volatility_regime, volume_regime, market_regime  # 🆕 시장 조건
                ))
                conn.commit()
                print(f"✅ 거래 기록 저장: {trade_record['coin']} {trade_record['action']}")
        except Exception as e:
            print(f"⚠️ 거래 기록 저장 실패 ({trade_record['coin']}): {e}")
    
    def save_completed_trade_for_learning(self, trade_record: Dict):
        """완료된 거래를 학습용으로 저장 (virtual_learning_trades + completed_trades)"""
        try:
            target_price = trade_record.get('target_price', 0.0)
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 🆕 1. virtual_learning_trades 테이블에 저장
                try:
                    existing = conn.execute("""
                        SELECT 1 FROM virtual_learning_trades 
                        WHERE coin = ? AND entry_timestamp = ? AND exit_timestamp = ?
                    """, (trade_record['coin'], trade_record['entry_timestamp'], trade_record['exit_timestamp'])).fetchone()
                    
                    if not existing:
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
                except Exception as e:
                    pass  # virtual_learning_trades 저장 실패는 무시
            
                # 🆕 2. completed_trades 테이블에 저장 (직접 저장으로 성능 최적화)
                try:
                    # 🆕 target_price 컬럼이 없으면 추가 (마이그레이션)
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(completed_trades)")
                    columns = [col[1] for col in cursor.fetchall()]
                    if columns and 'target_price' not in columns:
                        cursor.execute("ALTER TABLE completed_trades ADD COLUMN target_price REAL DEFAULT 0.0")
                    
                    conn.execute("""
                        INSERT OR IGNORE INTO completed_trades 
                        (coin, entry_timestamp, exit_timestamp, entry_price, exit_price,
                         profit_loss_pct, action, holding_duration, target_price)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_record['coin'], trade_record['entry_timestamp'], trade_record['exit_timestamp'],
                        trade_record['entry_price'], trade_record['exit_price'], trade_record['profit_loss_pct'],
                        trade_record['action'], trade_record['holding_duration'], target_price
                    ))
                    conn.commit()
                except Exception as e:
                    pass # completed_trades 저장 실패는 무시
                    
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
        """DB에서 포지션 로드 (이미 종료된 포지션 제외)"""
        try:
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 1. 현재 열려있는 포지션 로드
                df = pd.read_sql("SELECT * FROM virtual_positions", conn)
                
                # 2. 이미 종료된 거래 내역 로드 (중복 로드 방지용)
                # 🆕 entry/exit를 모두 비교할 수 있도록 매핑 생성 (동일 초 재진입 보호)
                history_df = pd.read_sql("SELECT coin, entry_timestamp, exit_timestamp FROM virtual_trade_history", conn)
                
                closed_map = {}
                for _, hrow in history_df.iterrows():
                    coin_h = str(hrow['coin'])
                    et = self._safe_convert_to_int(hrow.get('entry_timestamp', 0))
                    xt = self._safe_convert_to_int(hrow.get('exit_timestamp', 0))
                    if coin_h not in closed_map:
                        closed_map[coin_h] = []
                    closed_map[coin_h].append((et, xt))
                
                self.positions = {}
                fixed_count = 0
                skipped_count = 0
                
                for _, row in df.iterrows():
                    try:
                        # 🚨 숫자형 코인 심볼 안전 처리
                        coin_symbol = str(row['coin'])
                        
                        # 🆕 잘못된 코인 심볼(숫자만 있는 경우) 감지 및 삭제
                        # 예: '541973', '458' 등 잘못된 데이터가 DB에 들어간 경우
                        if coin_symbol.isdigit():
                            print(f"🗑️ 잘못된 코인 심볼 발견(숫자): {coin_symbol} - DB에서 영구 삭제합니다.")
                            self.delete_position_from_db(coin_symbol)
                            skipped_count += 1
                            continue
                            
                        # 🆕 타임스탬프를 정수로 변환하여 타입 불일치 문제 해결 (바이너리 데이터 처리 추가)
                        entry_timestamp = self._safe_convert_to_int(row['entry_timestamp'])
                        
                        # 🚫 필터링: (coin, entry_ts)와 정확히 일치하고, exit_ts 이후로 last_updated가 갱신되지 않은 경우만 좀비로 판단
                        last_updated = self._safe_convert_to_int(row['last_updated'])
                        is_zombie = False
                        if coin_symbol in closed_map:
                            for et, xt in closed_map[coin_symbol]:
                                if et == entry_timestamp and xt > 0 and last_updated <= xt:
                                    is_zombie = True
                                    zombie_exit_ts = xt
                                    break
                        if is_zombie:
                            print(f"🧟 {coin_symbol}: 종료 이력과 동일한 포지션 감지 → 정리 (entry={entry_timestamp}, exit={zombie_exit_ts}, last_updated={last_updated})")
                            self.delete_position_from_db(coin_symbol)
                            skipped_count += 1
                            continue
                        
                        # 🆕 진입가가 0인 경우 복구
                        entry_price = self._safe_convert_to_float(row['entry_price'])
                        current_price = self._safe_convert_to_float(row['current_price'])
                        
                        if entry_price == 0.0:
                            # 🆕 최신 가격으로 복구
                            latest_price = self._get_latest_price(coin_symbol)
                            if latest_price > 0:
                                entry_price = latest_price
                                current_price = latest_price
                                fixed_count += 1
                                print(f"🔧 {coin_symbol} 진입가 복구: 0.00원 → {self._format_price(latest_price)}원")
                        
                        # 🆕 현재가도 0인 경우 복구
                        if current_price == 0.0:
                            latest_price = self._get_latest_price(coin_symbol)
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
                        
                        # 🆕 target_price와 initial_target_price 처리 (마이그레이션)
                        target_price = self._safe_convert_to_float(row.get('target_price', 0.0))
                        initial_target_price = self._safe_convert_to_float(row.get('initial_target_price', 0.0))
                        
                        # initial_target_price가 0이면 target_price로 채움 (과거 데이터 호환성)
                        if initial_target_price == 0.0 and target_price > 0.0:
                            initial_target_price = target_price

                        self.positions[coin_symbol] = VirtualPosition(
                            coin=coin_symbol,
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
                            last_updated=last_updated,
                            target_price=target_price,
                            initial_target_price=initial_target_price,
                            pattern_type=str(row.get('pattern_type', 'none')),
                            entry_confidence=self._safe_convert_to_float(row.get('entry_confidence', 0.0))
                        )
                        
                        # 🆕 수정된 포지션을 DB에 저장 (타입 안전 비교)
                        original_entry = self._safe_convert_to_float(row['entry_price'])
                        original_current = self._safe_convert_to_float(row['current_price'])
                        
                        if entry_price > 0 and (original_entry == 0.0 or original_current == 0.0):
                            self.save_position_to_db(coin_symbol)
                            
                    except Exception as row_error:
                        print(f"⚠️ 포지션 로드 오류 ({row.get('coin', 'unknown')}): {row_error}")
                        continue
                
                print(f"✅ {len(self.positions)}개 포지션 로드 완료")
                if skipped_count > 0:
                    print(f"🧹 {skipped_count}개의 좀비 포지션(이미 종료됨)을 정리했습니다.")
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
            with sqlite3.connect(DB_PATH, timeout=60.0) as conn:  # 🔧 trade_candles.db 사용
                # 🆕 가장 가까운 시점의 캔들 조회
                query = """
                SELECT close FROM candles 
                WHERE symbol = ? AND timestamp <= ? 
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
            with sqlite3.connect(DB_PATH, timeout=60.0) as conn:  # 🔧 trade_candles.db 사용
                # 🚀 동적 인터벌 감지 및 최적화된 쿼리 생성
                try:
                    # 해당 코인의 모든 인터벌 조회
                    avail_intervals_df = pd.read_sql(
                        "SELECT DISTINCT interval FROM candles WHERE symbol = ?", 
                        conn, params=(coin,)
                    )
                    avail_intervals = avail_intervals_df['interval'].tolist()
                except Exception:
                    avail_intervals = []

                if not avail_intervals:
                    return 0.0

                # 인터벌 정렬 (단기 -> 장기)
                def get_minutes(iv):
                    iv = iv.lower()
                    try:
                        if iv.endswith('m'): return int(iv[:-1])
                        if iv.endswith('h'): return int(iv[:-1]) * 60
                        if iv.endswith('d'): return int(iv[:-1]) * 1440
                        if iv.endswith('w'): return int(iv[:-1]) * 10080
                    except: pass
                    return 999999
                
                # 우선순위: 15m(단기) > 30m > ... (가장 짧은 인터벌의 최신가를 현재가로 간주)
                sorted_intervals = sorted(avail_intervals, key=get_minutes)
                
                # 쿼리 생성
                placeholders = ', '.join(['?' for _ in sorted_intervals])
                
                # CASE문 동적 생성
                case_parts = []
                for idx, iv in enumerate(sorted_intervals):
                    case_parts.append(f"WHEN '{iv}' THEN {idx+1}")
                order_case = "\n".join(case_parts)
                
                query = f"""
                SELECT interval, close FROM (
                    SELECT interval, close, 
                           ROW_NUMBER() OVER (PARTITION BY interval ORDER BY timestamp DESC) as rn
                    FROM candles 
                    WHERE symbol = ? AND interval IN ({placeholders})
                ) ranked
                WHERE rn = 1 AND close > 0
                ORDER BY 
                    CASE interval 
                        {order_case}
                        ELSE 999
                    END
                LIMIT 1
                """
                
                params = [coin] + sorted_intervals
                result = conn.execute(query, params).fetchone()
                
                if result:
                    price = float(result[1])
                    # 캐시에 저장 (0보다 클 때만)
                    if price > 0:
                        self.price_cache[cache_key] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                    return price
                
                return 0.0
                
        except Exception as e:
            print(f"⚠️ {coin} 가격 조회 오류: {e}")
            return 0.0
    
    def prefetch_prices(self, coins: List[str]):
        """🚀 [성능 최적화] 코인 목록에 대한 가격을 미리 조회하여 캐시에 저장 (배치 처리)"""
        if not coins:
            return
        
        try:
            # 중복 제거
            unique_coins = list(set(coins))
            
            # 이미 배치 큐에 있는 것들은 제외
            existing_batch = set(self.position_update_batch)
            new_coins = [c for c in unique_coins if c not in existing_batch]
            
            if not new_coins:
                return

            print(f"⚡ {len(new_coins)}개 코인 가격 미리 조회 (Prefetch)...")
            
            # 배치 큐에 추가
            self.position_update_batch.extend(new_coins)
            
            # 강제로 배치 업데이트 실행
            self._update_price_batch()
            
        except Exception as e:
            print(f"⚠️ 가격 미리 조회(Prefetch) 오류: {e}")
    
    def _update_price_batch(self):
        """🚀 배치 가격 업데이트"""
        try:
            if not self.position_update_batch:
                return
            
            # 배치로 가격 조회
            coins = list(set(self.position_update_batch))
            placeholders = ', '.join(['?' for _ in coins])
            
            with sqlite3.connect(DB_PATH, timeout=60.0) as conn:  # 🔧 trade_candles.db 사용
                # 🚀 동적 인터벌 대응: 특정 인터벌(15m) 고정 없이 가장 최신 캔들 사용
                df = pd.read_sql(f"""
                    SELECT symbol as coin, close FROM (
                        SELECT symbol, close, 
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM candles 
                        WHERE symbol IN ({placeholders})
                    ) ranked
                    WHERE rn = 1 AND close > 0
                """, conn, params=coins)
                
                # 캐시 업데이트
                current_time = time.time()
                for _, row in df.iterrows():
                    price = float(row['close'])
                    if price > 0:
                        cache_key = f"price_{row['coin']}"
                        self.price_cache[cache_key] = {
                            'price': price,
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
    
    def update_all_positions(self):
        """🚀 모든 보유 포지션 일괄 업데이트 (배치 처리로 성능 최적화)"""
        if not self.positions:
            return

        print(f"🔄 {len(self.positions)}개 포지션 일괄 업데이트 중...")
        current_time = int(datetime.now().timestamp())
        
        # 1. 업데이트할 코인 목록
        coins = list(self.positions.keys())
        
        # 2. 배치 단위로 가격 조회 및 업데이트
        batch_size = 50
        for i in range(0, len(coins), batch_size):
            batch_coins = coins[i:i+batch_size]
            
            # 배치 가격 조회를 위해 position_update_batch에 추가
            self.position_update_batch.extend(batch_coins)
            self._update_price_batch()  # 배치 가격 조회 실행 (캐시에 저장됨)
            
            # 각 코인 업데이트 (캐시된 가격 사용, DB 저장은 생략하여 속도 향상, 매매 실행 안 함)
            for coin in batch_coins:
                try:
                    latest_price = self._get_latest_price(coin)
                    if latest_price > 0:
                        # 🚀 [성능 최적화] save_db=False, execute_action=False
                        self.update_position(coin, latest_price, current_time, save_db=False, execute_action=False)
                except Exception as e:
                    print(f"⚠️ {coin} 포지션 업데이트 오류: {e}")
        
        # 🆕 포지션 요약 정보 출력
        try:
            total_positions = len(self.positions)
            if total_positions > 0:
                up_count = sum(1 for p in self.positions.values() if p.profit_loss_pct > 0)
                down_count = sum(1 for p in self.positions.values() if p.profit_loss_pct < 0)
                
                # 수익률 합계 및 평균 계산
                total_profit_sum = sum(p.profit_loss_pct for p in self.positions.values())
                avg_profit = total_profit_sum / total_positions
                
                # 최고/최저 수익률 종목 찾기
                sorted_positions = sorted(self.positions.values(), key=lambda x: x.profit_loss_pct, reverse=True)
                best_coin = sorted_positions[0]
                worst_coin = sorted_positions[-1]
                
                print(f"\n📊 [포지션 요약] 총 {total_positions}개 보유 (🔺상승: {up_count} / 🔻하락: {down_count})")
                print(f"   💰 총 수익률 합계: {total_profit_sum:+.2f}%")
                print(f"   💰 평균 수익률: {avg_profit:+.2f}%")
                print(f"   🏆 최고 수익: {best_coin.coin} ({best_coin.profit_loss_pct:+.2f}%)")
                print(f"   📉 최저 수익: {worst_coin.coin} ({worst_coin.profit_loss_pct:+.2f}%)\n")
        except Exception as e:
            print(f"⚠️ 포지션 요약 출력 오류: {e}")
                    
    def open_position(self, coin: str, price: float, signal_score: float, timestamp: int, signal: SignalInfo = None) -> bool:
        """포지션 열기"""
        try:
            # 🆕 이미 보유 중인지 확인
            if coin in self.positions:
                return False
            
            # 🧭 시장 국면 필터 (5분 캐시)
            regime_info = self._get_market_regime_info()
            regime_score = regime_info.get('score', 50)
            regime_tag = regime_info.get('regime', 'neutral')
            
            # 🚨 [수정] 시장 국면 점수가 0-1 범위일 때 0.25 미만이면 경고
            # MarketAnalyzer가 0-1 범위 점수를 반환하므로 비교 기준 수정
            # 🆕 로그는 process_signal()에서 통일된 형식으로 출력하므로 여기서는 생략
            if regime_score < 0.25:  # 🔧 25 → 0.25 (0-1 범위 대응)
                pass  # 로그는 process_signal()에서 출력
                # return False  <-- 실전과 동일하게 차단 해제
            
            # 🧯 포트폴리오 상관관계 필터
            current_holdings = list(self.positions.keys())
            if current_holdings:
                try:
                    risk_check = self.portfolio_risk_manager.check_correlation_risk(
                        coin, current_holdings, threshold=0.8
                    )
                    if not risk_check.get('safe', True):
                        # 상관관계 과다 시 거부 (로그는 상위에서 처리)
                        return False
                except Exception as corr_err:
                    # 상관관계 체크 실패 시 경고만 출력하고 진행 (차단하지 않음)
                    pass  # 로그 생략 - 너무 잦은 출력 방지
            
            # 🎯 [수정] 진입가는 최신가를 우선 사용 (15분 캔들 기준)
            entry_price = price # 기본값
            
            # 🆕 최신가 재확인 (process_signal에서 넘어왔더라도 한번 더 확인)
            try:
                latest = self._get_latest_price(coin)
                if latest > 0:
                    entry_price = latest
            except Exception:
                pass
            
            # 🆕 [Realistic Friction] 슬리피지(Slippage) 0.1% 적용 (현실적 마찰력)
            # 시장가 매수 시 호가 갭과 수수료를 고려하여 진입가를 0.1% 높게 잡음
            entry_price = entry_price * 1.001
            
            # 🆕 현재가 조회 (수익률 계산용)
            current_price = entry_price
            
            # 🆕 target_price 추출 (signal에서 가져오기)
            target_price = 0.0
            pattern_type = 'none'
            entry_confidence = 0.0 # 🆕 진입 신뢰도 초기화
            
            if signal:
                if hasattr(signal, 'target_price'):
                    target_price = signal.target_price if signal.target_price is not None else 0.0
                if hasattr(signal, 'pattern_type'):
                    pattern_type = signal.pattern_type if signal.pattern_type is not None else 'none'
                if hasattr(signal, 'confidence'): # 🆕 신뢰도 추출
                    entry_confidence = signal.confidence if signal.confidence is not None else 0.0
            
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
                    last_updated=timestamp,
                    target_price=target_price,  # 🆕 예상 목표가 저장
                    initial_target_price=target_price, # 🆕 초기 목표가 불변 기록
                    pattern_type=pattern_type,   # 🆕 패턴 정보 저장
                    entry_confidence=entry_confidence # 🆕 진입 신뢰도 저장
                )
                
                # 🆕 DB에 저장
                self.save_position_to_db(coin)
                
                # 🆕 목표가 정보 출력
                target_info = ""
                if signal and signal.target_price > 0:
                    expected_profit = ((signal.target_price - entry_price) / entry_price) * 100
                    target_info = f" (목표가: {self._format_price(signal.target_price)}원, 예상: {expected_profit:+.2f}%)"
                
                print(f"🆕 포지션 열기: {get_korean_name(coin)} @ {self._format_price(entry_price)}원{target_info}")

                # 🆕 [로그 강화] 매수 시점의 상세 정보를 virtual_trade_history에 'buy' 액션으로 기록
                # 대시보드에서 RSI나 Score를 보여주기 위해 action 필드나 별도 필드 활용
                # action 필드에 "buy | Score:0.85 | RSI:32.5" 형태로 저장하여 프론트에서 파싱 유도
                try:
                    buy_action_detail = "buy"
                    if signal:
                        buy_action_detail += f" | Score:{signal.signal_score:.2f}"
                        if signal.rsi > 0: buy_action_detail += f" | RSI:{signal.rsi:.1f}"
                        if hasattr(signal, 'pattern_type') and signal.pattern_type != 'none':
                            buy_action_detail += f" | Pat:{signal.pattern_type}"
                    
                    with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                        conn.execute("""
                            INSERT INTO virtual_trade_history 
                            (coin, entry_price, exit_price, profit_loss_pct, holding_duration, action, entry_timestamp, exit_timestamp, created_at, quantity, entry_signal_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            coin, entry_price, 0.0, 0.0, 0, buy_action_detail, 
                            timestamp, 0, datetime.now().isoformat(), 1.0, signal_score
                        ))
                        conn.commit()
                except Exception as e:
                    print(f"⚠️ 매수 로그 저장 실패: {e}")

                return True
                
            except Exception as e:
                print(f"  ❌ {get_korean_name(coin)}: 포지션 생성 오류 - {e}")
                return False
                
        except Exception as e:
            print(f"  ❌ {get_korean_name(coin)}: open_position 예외 - {e}")
            return False
    
    def process_signal(self, signal: SignalInfo):
        """🆕 순수 시그널과 보유 정보를 조합하여 최종 액션 결정 (트레일링 스탑 적용)"""
        current_time = int(datetime.now().timestamp())
        
        # 🆕 [AI Signal Log] 유의미한 시그널 감지 시 로그 기록
        # (매수 시그널이거나 점수가 절댓값 0.5 이상인 강한 시그널일 때)
        if signal.action == SignalAction.BUY or abs(signal.signal_score) >= 0.5:
             # 중복 로그 방지를 위해 1분 내 동일 코인 시그널은 패스 (구현 복잡하므로 일단 생략)
             # 로그 메시지: "📡 BTC 상승 시그널 감지 (Score 0.85)"
             direction = "상승" if signal.signal_score > 0 else "하락"
             log_msg = f"{get_korean_name(signal.coin)} {direction} 시그널 감지 (Score {signal.signal_score:.2f})"
             
             # 시스템 로그 저장 (컴포넌트: Signal) -> API에서 아이콘 매핑 필요
             self.log_system_event("INFO", "MarketAnalyzer", log_msg, {"score": signal.signal_score})
        
        # ⏰ [유효기간 체크] 생성된 지 60분 지난 시그널은 폐기 (뒷북 방지)
        signal_age_minutes = (current_time - signal.timestamp) / 60
        if signal_age_minutes > 60:
            # print(f"⌛ {get_korean_name(signal.coin)}: 시그널 유효기간 만료 ({int(signal_age_minutes)}분 경과) - 패스")
            return

        # ⚡ [최신가 갱신] 시그널 가격 대신 최신 15분 캔들 가격 조회 (사용자 피드백 반영)
        current_price = signal.price
        try:
            latest = self._get_latest_price(signal.coin)
            if latest > 0:
                current_price = latest
            else:
                # 최신가를 못 가져오면 시그널 가격이라도 써야겠지만, 
                # 데이터가 멈춘 상태일 수 있으므로 너무 오래된 시그널이면 거부
                if signal_age_minutes > 15:
                    print(f"⚠️ {signal.coin}: 최신가 조회 실패 & 시그널 15분 경과 - 진입 거부")
                    return
        except Exception:
            pass
        
        # 🛡️ [괴리율 체크] 시그널 가격 vs 현재가 3% 이상 차이면 진입 거부 (급등락/뒷북)
        if signal.action == SignalAction.BUY and signal.price > 0:
            price_diff_pct = abs((current_price - signal.price) / signal.price) * 100
            if price_diff_pct > 3.0:
                print(f"📉 {get_korean_name(signal.coin)}: 가격 괴리율 과다 ({price_diff_pct:.1f}%) - 진입 보류 (시그널 {self._format_price(signal.price)} vs 현재 {self._format_price(current_price)})")
                return

        # 🚨 가격 데이터 유효성 검사 (0원 방지)
        if current_price <= 0:
            # 최신가 조회 재시도 (캐시 또는 DB)
            try:
                latest = self._get_latest_price(signal.coin)
                if latest > 0:
                    current_price = latest
                else:
                    print(f"⚠️ {signal.coin}: 현재가 오류 (0원) - 시그널 처리 건너뜀")
                    return
            except Exception:
                return
        
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

            # 🚨 포지션 업데이트 중 청산(cleanup)되어 삭제되었을 수 있으므로 재확인
            if signal.coin not in self.positions:
                return

            position = self.positions[signal.coin]

            # 🆕 시장 레짐 정보 조회 (결정 저장용)
            regime_info = self._get_market_regime_info()
            
            # 🆕 Thompson 점수 조회 (보유 중인 코인에도 실제 점수 표시)
            thompson_score = self._get_thompson_score(signal)
            
            # 🆕 수익률 스냅샷 기록 및 추세 분석
            trend_analysis = None
            if TRAJECTORY_ANALYZER_AVAILABLE:
                try:
                    trajectory_analyzer = get_virtual_trajectory_analyzer()
                    trajectory_analyzer.record_profit_snapshot(
                        coin=signal.coin,
                        profit_pct=position.profit_loss_pct,
                        current_price=current_price,
                        entry_price=position.entry_price,
                        signal_score=signal.signal_score,
                        max_profit_pct=position.max_profit_pct,
                        min_profit_pct=position.max_loss_pct,
                        holding_hours=position.holding_duration / 3600,
                        market_regime=regime_info.get('regime', 'neutral')
                    )
                    # 추세 분석 실행
                    trend_analysis = trajectory_analyzer.analyze_trend(signal.coin, lookback=10)
                except Exception as e:
                    print(f"⚠️ {signal.coin} 추세 분석 오류: {e}")
            
            # 🆕 추세 분석 결과 출력 (실전매매와 동일)
            if trend_analysis and trend_analysis.history_count >= 3:
                trend_type_str = trend_analysis.trend_type.value
                reason_str = trend_analysis.reason
                # 추세 타입을 한글로 변환 (간단한 매핑)
                trend_map = {
                    'up': '상승',
                    'down': '하락',
                    'sideways': '횡보',
                    'peak_reversal': '고점반전',
                    'strong_up': '강한상승',
                    'strong_down': '강한하락',
                    'neutral': '중립'
                }
                trend_kr = trend_map.get(trend_type_str, trend_type_str)
                
                # 신뢰도에 따른 표시
                if trend_analysis.confidence >= 0.7:
                    confidence_icon = "🟢"
                elif trend_analysis.confidence >= 0.5:
                    confidence_icon = "🟡"
                else:
                    confidence_icon = "⚪"
                
                print(f"   📉 추세: {trend_kr} ({confidence_icon} {reason_str})")
                if trend_analysis.should_sell_early:
                    print(f"   ⚠️ 조기 매도 권장!")
                elif trend_analysis.should_hold_strong:
                    print(f"   💪 강한 홀딩 권장!")

            # 🎯 시그널 액션에 따라 처리 (realtime_signal_selector가 이미 정교하게 계산함)
            if signal.action == SignalAction.SELL:
                # 🆕🆕 가상매매 결정 저장 (실전매매에서 읽기용)
                self.save_trade_decision({
                    'coin': signal.coin,
                    'timestamp': current_time,
                    'decision': 'sell',
                    'signal_score': signal.signal_score,
                    'confidence': signal.confidence,
                    'current_price': current_price,
                    'target_price': getattr(signal, 'target_price', 0.0),
                    'expected_profit_pct': 0.0,
                    'thompson_score': thompson_score,  # 🆕 실제 점수 사용
                    'thompson_approved': True,
                    'regime_score': regime_info.get('score', 0.5),
                    'regime_name': regime_info.get('regime', 'Neutral'),
                    'viability_passed': True,
                    'reason': '시그널 기반 매도',
                    'is_holding': True,
                    'entry_price': position.entry_price,
                    'profit_loss_pct': position.profit_loss_pct
                })
                
                self._close_position(signal.coin, current_price, current_time, 'sell')
                print(f"{get_korean_name(signal.coin)} : 매도 (시그널) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간 (점수: {signal.signal_score:.3f})")
            elif signal.action == SignalAction.HOLD:
                # 🆕🆕 가상매매 결정 저장 (실전매매에서 읽기용)
                self.save_trade_decision({
                    'coin': signal.coin,
                    'timestamp': current_time,
                    'decision': 'hold',
                    'signal_score': signal.signal_score,
                    'confidence': signal.confidence,
                    'current_price': current_price,
                    'target_price': getattr(signal, 'target_price', 0.0),
                    'expected_profit_pct': ((getattr(signal, 'target_price', 0) - current_price) / current_price * 100) if getattr(signal, 'target_price', 0) > 0 and current_price > 0 else 0.0,
                    'thompson_score': thompson_score,  # 🆕 실제 점수 사용
                    'thompson_approved': True,
                    'regime_score': regime_info.get('score', 0.5),
                    'regime_name': regime_info.get('regime', 'Neutral'),
                    'viability_passed': True,
                    'reason': '홀딩 유지',
                    'is_holding': True,
                    'entry_price': position.entry_price,
                    'profit_loss_pct': position.profit_loss_pct
                })
                
                # 🆕 통일된 로그 형식 (trade_executor.py와 동일)
                target_status = ""
                if signal.target_price > 0:
                    dist_pct = ((signal.target_price - current_price) / current_price) * 100
                    target_status = f", 목표까지 {dist_pct:+.2f}%"
                print(f"📊 {get_korean_name(signal.coin)}: 가상매매결정=hold (점수: {signal.signal_score:.3f})")
                print(f"   📈 보유정보: 진입가 {self._format_price(position.entry_price)}원, 수익률 {position.profit_loss_pct:+.2f}%")
                print(f"   📥 Thompson: {thompson_score:.2f}, {position.holding_duration//3600}시간{target_status}")
            elif signal.action == SignalAction.BUY:
                # 보유 중일 때 추가 매수 신호는 무시 (🆕 'hold'로 저장하여 실전매매에 알림)
                self.save_trade_decision({
                    'coin': signal.coin,
                    'timestamp': current_time,
                    'decision': 'hold',  # 🆕 이미 보유 중이므로 hold
                    'signal_score': signal.signal_score,
                    'confidence': signal.confidence,
                    'current_price': current_price,
                    'target_price': getattr(signal, 'target_price', 0.0),
                    'expected_profit_pct': ((getattr(signal, 'target_price', 0) - current_price) / current_price * 100) if getattr(signal, 'target_price', 0) > 0 and current_price > 0 else 0.0,
                    'thompson_score': thompson_score,  # 🆕 실제 점수 사용
                    'thompson_approved': False,
                    'regime_score': regime_info.get('score', 0.5),
                    'regime_name': regime_info.get('regime', 'Neutral'),
                    'viability_passed': False,
                    'reason': '이미 보유 중 (추매 불가)',
                    'is_holding': True,
                    'entry_price': position.entry_price,
                    'profit_loss_pct': position.profit_loss_pct
                })

            # 🚀 트레일링 스탑 (Trailing Stop) 로직
            # max_profit_pct는 update_position에서 이미 갱신됨
            max_profit = position.max_profit_pct
            profit_pct = position.profit_loss_pct
            
            # 1. 수익 20% 이상 도달 후, 고점 대비 5% 하락 시 익절 (15% 수익 확보)
            if max_profit >= 20.0 and profit_pct <= (max_profit - 5.0):
                self.log_system_event("JUDGEMENT", "Executor", f"{signal.coin} 트레일링 스탑 발동! (최고 {max_profit:.1f}% -> 현재 {profit_pct:.1f}%)", {"max_profit": max_profit, "current_profit": profit_pct})
                self._close_position(signal.coin, current_price, current_time, 'trailing_stop')
                print(f"📉 {get_korean_name(signal.coin)} : 매도 (트레일링 스탑) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {profit_pct:+.2f}% (최고 {max_profit:.1f}%)")
                
            # 2. 수익 10% 이상 도달 후, 고점 대비 3% 하락 시 익절 (7% 수익 확보)
            elif max_profit >= 10.0 and profit_pct <= (max_profit - 3.0):
                self.log_system_event("JUDGEMENT", "Executor", f"{signal.coin} 트레일링 스탑 발동 (최고 {max_profit:.1f}% -> 현재 {profit_pct:.1f}%)", {"max_profit": max_profit, "current_profit": profit_pct})
                self._close_position(signal.coin, current_price, current_time, 'trailing_stop')
                print(f"📉 {get_korean_name(signal.coin)} : 매도 (트레일링 스탑) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {profit_pct:+.2f}% (최고 {max_profit:.1f}%)")
                
            # 3. 수익 5% 이상 도달 후, 본전(0.5% 이하) 위협 시 익절 (수익 보전)
            elif max_profit >= 5.0 and profit_pct <= 0.5:
                self.log_system_event("WARN", "Executor", f"{signal.coin} 수익 반납 위기. 본전 탈출 시도.", {"max_profit": max_profit})
                self._close_position(signal.coin, current_price, current_time, 'trailing_stop')
                print(f"🛡️ {get_korean_name(signal.coin)} : 매도 (수익 보전) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {profit_pct:+.2f}% (최고 {max_profit:.1f}%)")

            # 🎯 추가 안전 장치: 극단적 손익 시 강제 청산 (기존 로직)
            elif position.profit_loss_pct >= 50.0:  # 익절
                self.log_system_event("JUDGEMENT", "Executor", f"🎉 {signal.coin} 대박 수익 달성 (+50%). 익절 확정.", {"roi": 50.0})
                self._close_position(signal.coin, current_price, current_time, 'take_profit')
                print(f"{get_korean_name(signal.coin)} : 매도 (익절) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간")
            elif position.profit_loss_pct <= -10.0:  # 손절
                self.log_system_event("WARN", "RiskManager", f"😭 {signal.coin} 손절 원칙 실행 (-10%).", {"roi": position.profit_loss_pct})
                self._close_position(signal.coin, current_price, current_time, 'stop_loss')
                print(f"{get_korean_name(signal.coin)} : 매도 (손절) {self._format_price(position.entry_price)}원 → {self._format_price(current_price)}원, {position.profit_loss_pct:+.2f}%, {position.holding_duration//3600}시간")

        # 🆕 미보유 시 BUY 시그널이면 매수 (🎰 Thompson Sampling 적용)
        elif signal.action == SignalAction.BUY:
            if self.can_open_position(signal.coin):
                # 🎰 Thompson Sampling으로 매수 실행 여부 결정
                should_buy, final_score, reason = self._decide_buy_with_thompson(signal)
                
                # 🆕🆕 [수정] 순수 Thompson 점수 별도 조회 (복합 점수와 구분)
                # final_score는 시그널+Thompson+수익률 가중합이므로, 순수 Thompson 점수 조회
                pure_thompson_score = self._get_thompson_score(signal)
                
                # 🆕 시장 레짐 정보 조회
                regime_info = self._get_market_regime_info()
                
                # 🆕🆕 [버그 수정] 가상매매 결정 저장은 open_position() 결과 후에!
                # open_position()이 실패하면 decision='skip'으로 저장해야 함
                final_decision = 'skip'  # 기본값
                
                # 🆕 통일된 로그 형식 (trade_executor.py와 동일)
                expected_profit_pct = ((getattr(signal, 'target_price', 0) - current_price) / current_price * 100) if getattr(signal, 'target_price', 0) > 0 and current_price > 0 else 0.0
                
                if should_buy:
                    # 실제 포지션 열기 시도
                    position_opened = self.open_position(signal.coin, current_price, signal.signal_score, current_time, signal)
                    
                    if position_opened:
                        final_decision = 'buy'  # 성공 시에만 'buy'
                        print(f"📊 {get_korean_name(signal.coin)}: 가상매매결정=buy (점수: {signal.signal_score:.3f})")
                        print(f"   📥 Thompson: {pure_thompson_score:.2f}, 기대수익: {expected_profit_pct:.2f}%")
                    else:
                        # open_position() 실패 - skip으로 처리
                        print(f"📊 {get_korean_name(signal.coin)}: 가상매매결정=skip (점수: {signal.signal_score:.3f})")
                        print(f"   📥 Thompson: {pure_thompson_score:.2f}")
                        print(f"   ⛔ 포지션 열기 실패")
                        reason = f"포지션 열기 실패 (원인: {reason})"
                else:
                    print(f"📊 {get_korean_name(signal.coin)}: 가상매매결정=skip (점수: {signal.signal_score:.3f})")
                    print(f"   📥 Thompson: {pure_thompson_score:.2f}, 기대수익: {expected_profit_pct:.2f}%")
                    print(f"   ✋ 매수 보류: {reason}")
                
                # 🆕🆕 가상매매 결정 저장 (실전매매에서 읽기용) - 최종 결과 기준!
                decision_data = {
                    'coin': signal.coin,
                    'timestamp': current_time,
                    'decision': final_decision,  # 실제 결과 기준
                    'signal_score': signal.signal_score,
                    'confidence': signal.confidence,
                    'current_price': current_price,
                    'target_price': getattr(signal, 'target_price', 0.0),
                    'expected_profit_pct': ((getattr(signal, 'target_price', 0) - current_price) / current_price * 100) if getattr(signal, 'target_price', 0) > 0 and current_price > 0 else 0.0,
                    'thompson_score': pure_thompson_score,  # 🆕🆕 순수 Thompson 점수 사용
                    'thompson_approved': should_buy,
                    'regime_score': regime_info.get('score', 0.5),
                    'regime_name': regime_info.get('regime', 'Neutral'),
                    'viability_passed': final_decision == 'buy',  # 실제 결과 기준
                    'reason': reason,
                    'is_holding': False,
                    'entry_price': 0.0,
                    'profit_loss_pct': 0.0
                }
                self.save_trade_decision(decision_data)
            else:
                # 진입 불가 사유 출력 (로그 강화)
                if signal.coin in self.positions:
                    pass # 이미 보유 중인 경우는 너무 잦은 로그 방지를 위해 생략
                else:
                    print(f"  ⛔ {get_korean_name(signal.coin)}: 진입 조건 미달 (서킷브레이커 등)")
                    
                # 🆕🆕 can_open_position=False여도 'skip'으로 저장 (이전 'buy' 결정 무효화)
                regime_info = self._get_market_regime_info()
                skip_thompson_score = self._get_thompson_score(signal)  # 🆕 실제 점수 조회
                self.save_trade_decision({
                    'coin': signal.coin,
                    'timestamp': current_time,
                    'decision': 'skip',
                    'signal_score': signal.signal_score,
                    'confidence': signal.confidence,
                    'current_price': current_price,
                    'target_price': getattr(signal, 'target_price', 0.0),
                    'expected_profit_pct': 0.0,
                    'thompson_score': skip_thompson_score,  # 🆕 실제 점수 사용
                    'thompson_approved': False,
                    'regime_score': regime_info.get('score', 0.5),
                    'regime_name': regime_info.get('regime', 'Neutral'),
                    'viability_passed': False,
                    'reason': '진입 조건 미달',
                    'is_holding': signal.coin in self.positions,
                    'entry_price': 0.0,
                    'profit_loss_pct': 0.0
                })
        
        # 🆕🆕 [핵심 수정] 미보유 + HOLD/SELL 시그널일 때도 'skip' 저장
        # → 이전에 저장된 오래된 'buy' 결정을 무효화
        else:
            regime_info = self._get_market_regime_info()
            else_thompson_score = self._get_thompson_score(signal)  # 🆕 실제 점수 조회
            self.save_trade_decision({
                'coin': signal.coin,
                'timestamp': current_time,
                'decision': 'skip',  # 미보유인데 HOLD/SELL 시그널이므로 매수 안함
                'signal_score': signal.signal_score,
                'confidence': signal.confidence,
                'current_price': current_price,
                'target_price': getattr(signal, 'target_price', 0.0),
                'expected_profit_pct': 0.0,
                'thompson_score': else_thompson_score,  # 🆕 실제 점수 사용
                'thompson_approved': False,
                'regime_score': regime_info.get('score', 0.5),
                'regime_name': regime_info.get('regime', 'Neutral'),
                'viability_passed': False,
                'reason': f'미보유 상태에서 {signal.action.value} 시그널',
                'is_holding': False,
                'entry_price': 0.0,
                'profit_loss_pct': 0.0
            })
    
    def _get_thompson_score(self, signal: SignalInfo) -> float:
        """현재 시그널의 Thompson Sampling 점수 조회 (보유 중인 코인에도 사용)
        
        🆕 로컬 인스턴스 우선 사용, 공통 모듈 fallback
        """
        try:
            # 🆕 로컬 thompson_sampler 우선 사용 (이미 초기화된 인스턴스)
            if self.thompson_sampler is not None:
                # 패턴 추출
                pattern = self._extract_signal_pattern(signal)
                
                # Thompson Sampling에서 확률 샘플링
                # 🔧 sample_success_rate는 (float, str)을 반환 - 문자열은 신뢰도 메시지
                result = self.thompson_sampler.sample_success_rate(pattern)
                
                # 결과가 튜플이면 첫 번째 값(점수)만 사용
                if isinstance(result, tuple):
                    sampled_rate = result[0]
                else:
                    sampled_rate = float(result)
                
                return sampled_rate
            
            # Fallback: 공통 모듈 사용
            if THOMPSON_CORE_AVAILABLE:
                return core_get_thompson_score(signal)
            
            return 0.0
        except Exception as e:
            print(f"⚠️ Thompson 점수 조회 오류: {e}")
            return 0.0
    
    def _decide_buy_with_thompson(self, signal: SignalInfo) -> Tuple[bool, float, str]:
        """매수 실행 여부 결정 (DecisionMaker 위임 + 시장 상황 반영) 및 Thinking 로그 기록"""
        # 🎯 시장 상황 조회 (매수 결정에 반영)
        market_context = self._get_market_context()
        market_regime = market_context.get('regime', 'Neutral')
        market_trend = market_context.get('trend', 'neutral')
        market_score = market_context.get('score', 0.5)
        
        # 🎯 시장 상황 분석
        regime_lower = market_regime.lower() if market_regime else 'neutral'
        is_bearish = 'bearish' in regime_lower or market_trend == 'bearish'
        is_extreme_bearish = 'extreme_bearish' in regime_lower
        is_bullish = 'bullish' in regime_lower or market_trend == 'bullish'
        
        # DecisionMaker로 매수 결정
        should_buy, final_score, reason = self.decision_maker.decide_buy(signal, is_simulation=True)
        
        # 🎯 시장 상황에 따른 추가 필터링 (실전매매와 동일한 기준)
        if should_buy:
            # 극심한 하락장: 매우 엄격한 기준
            if is_extreme_bearish:
                # 시그널 점수와 Thompson 점수가 매우 높아야만 허용
                if signal.signal_score < 0.35 or final_score < 0.60:
                    should_buy = False
                    reason = f"극심한 하락장: 기준 강화 (시그널 {signal.signal_score:.2f} < 0.35 또는 점수 {final_score:.2f} < 0.60)"
            
            # 하락장: 엄격한 기준
            elif is_bearish:
                if signal.signal_score < 0.28 or final_score < 0.53:
                    should_buy = False
                    reason = f"하락장: 기준 강화 (시그널 {signal.signal_score:.2f} < 0.28 또는 점수 {final_score:.2f} < 0.53)"
            
            # 상승장: 완화된 기준 (DecisionMaker 결과 그대로 사용)
            # 중립장: 기본 기준 (DecisionMaker 결과 그대로 사용)
        
        # 🆕 [AI Thinking Log] 매수 결정 시 사고 과정 기록
        if should_buy:
            # 패턴 정보 추출 (로그용)
            pattern_desc = self._extract_signal_pattern(signal)
            
            # 사람이 읽기 쉬운 메시지로 변환
            # 예: "MA_UP_RSI_LOW" -> "상승추세+눌림목" (간단화)
            short_reason = reason.split('(')[0].strip() # "패턴 승률 우수" 등
            
            # 로그 메시지 생성
            log_msg = f"{get_korean_name(signal.coin)} 매수 판단: {short_reason} (Score {signal.signal_score:.2f})"

            # 시스템 로그에 저장 (컴포넌트: Thinking)
            self.log_system_event("INFO", "Strategy", log_msg, {
                "signal_score": signal.signal_score,
                "reason": reason,
                "pattern": pattern_desc
            })
            
        return should_buy, final_score, reason
    
    def _extract_signal_pattern(self, signal: SignalInfo) -> str:
        """시그널에서 패턴 문자열 추출 (DecisionMaker 위임 + 시장 상황 반영)"""
        base_pattern = self.decision_maker._extract_signal_pattern(signal)
        
        # 🆕 [Context Aware Learning] 시장 상황(Regime)을 패턴에 결합
        # 예: BULL_RSI_LOW_MACD_UP (상승장에서의 해당 패턴)
        # 시장 상황이 없으면 NEUTRAL 사용
        # [Fix] self.market_regime이 항상 최신인지 확인 필요. analyze_market_regime() 결과로 업데이트 권장
        regime = getattr(self, 'market_regime', 'NEUTRAL').upper()
        
        # [Optimization] 혹시 market_regime이 초기값(None/Neutral)이라면 캐시된 매니저에서 가져오기 시도
        if regime in ['NEUTRAL', 'NONE'] and hasattr(self, 'market_regime_manager'):
             try:
                 # 너무 잦은 호출 방지를 위해, 메모리에 저장된 최근 값을 쓰거나 가끔 갱신
                 # 여기서는 안전하게 현재 속성값 사용 (run_trading_cycle에서 업데이트됨)
                 pass
             except:
                 pass
        
        # 변동성 상태도 결합 (HighVol / LowVol)
        # signal.volatility가 있으면 활용
        vol_state = ""
        if hasattr(signal, 'volatility'):
            if signal.volatility > 0.05: # 변동성 5% 이상
                vol_state = "_HIGHVOL"
            elif signal.volatility < 0.01: # 변동성 1% 이하
                vol_state = "_LOWVOL"
                
        # 최종 패턴: REGIME_VOLSTATE_BASEPATTERN
        # 예: BULL_HIGHVOL_RSI_LOW_MACD_UP
        final_pattern = f"{regime}{vol_state}_{base_pattern}"
        
        return final_pattern
    
    def _update_thompson_on_trade_close(self, coin: str, signal_pattern: str, 
                                        success: bool, profit_pct: float):
        """거래 종료 시 Thompson Sampling 분포 업데이트"""
        try:
            if self.thompson_sampler is not None:
                self.thompson_sampler.update_distribution(signal_pattern, success, profit_pct)
        except Exception as e:
            print(f"⚠️ Thompson Sampling 업데이트 오류: {e}")
    
    def _combine_signal_with_position(self, signal: SignalInfo, position: VirtualPosition, current_price: float, trend_analysis=None) -> str:
        """🆕 순수 시그널과 보유 정보를 조합하여 최종 액션 결정 (실전매매와 동일한 로직, 트레일링 스탑 포함 + 시장 상황 반영 + 추세 분석)"""
        try:
            signal_score = signal.signal_score
            confidence = signal.confidence
            profit_loss_pct = position.profit_loss_pct
            
            # 🎯 시장 상황 조회 (매도 결정에 반영)
            regime_info = self._get_market_regime_info()
            market_regime = regime_info.get('regime', 'Neutral')
            market_trend = regime_info.get('trend', 'neutral')
            
            # 🎯 시장 상황에 따른 매도 조정 계수 계산
            regime_lower = market_regime.lower() if market_regime else 'neutral'
            is_bearish = 'bearish' in regime_lower or market_trend == 'bearish'
            is_extreme_bearish = 'extreme_bearish' in regime_lower
            is_bullish = 'bullish' in regime_lower or market_trend == 'bullish'
            
            # 매도 조정 계수 (하락장일수록 더 적극적으로 매도)
            if is_extreme_bearish:
                market_adjustment = 0.7  # 30% 완화 (더 쉽게 매도)
            elif is_bearish:
                market_adjustment = 0.85  # 15% 완화
            elif is_bullish:
                market_adjustment = 1.2  # 20% 강화 (더 확실한 신호에서만)
            else:
                market_adjustment = 1.0  # 중립
            
            # ═══════════════════════════════════════════════════════════════
            # 🆕 [추세 분석 기반] 조기 매도/강한 홀딩 판단 (참고 정보로만 사용)
            # ═══════════════════════════════════════════════════════════════
            # 추세 분석은 "경고" 신호로만 사용하고, 실제 매도는 학습 기반 로직으로 결정
            trend_sell_signal = False  # 추세 기반 매도 신호 플래그
            trend_sell_reason = ""
            
            if trend_analysis is not None and trend_analysis.confidence >= 0.5:
                # 조기 매도 권장: 고점 반전, 연속 하락 등
                if trend_analysis.should_sell_early:
                    # 고점 반전 감지 (학습 기반 매도 로직에서 고려)
                    if trend_analysis.trend_type.value == 'peak_reversal':
                        trend_sell_signal = True
                        trend_sell_reason = f"고점 반전 감지 ({trend_analysis.reason})"
                        print(f"   ⚠️ {get_korean_name(signal.coin)} 추세 경고: {trend_sell_reason}")
                    # 연속 하락 감지
                    elif trend_analysis.consecutive_drops >= 3:
                        trend_sell_signal = True
                        trend_sell_reason = f"연속 {trend_analysis.consecutive_drops}회 하락"
                        print(f"   ⚠️ {get_korean_name(signal.coin)} 추세 경고: {trend_sell_reason}")
                    # 강한 하락 추세 감지
                    elif trend_analysis.trend_type.value == 'strong_down':
                        trend_sell_signal = True
                        trend_sell_reason = f"급락 감지 ({trend_analysis.reason})"
                        print(f"   ⚠️ {get_korean_name(signal.coin)} 추세 경고: {trend_sell_reason}")
                
                # 강한 홀딩 권장: 상승 추세 지속
                if trend_analysis.should_hold_strong:
                    # 상승 추세에서는 매도 신호 무시하고 홀딩
                    if signal.action == SignalAction.SELL and trend_analysis.trend_type.value in ['strong_up', 'up']:
                        print(f"💪 {get_korean_name(signal.coin)} 추세 우선 홀딩 (상승 추세 지속: {trend_analysis.reason})")
                        return 'hold'
            
            # 🚀 트레일링 스탑 (Trailing Stop) 로직 (시장 상황 조정 적용)
            max_profit = position.max_profit_pct
            
            # 1. 수익 20% 이상 도달 후, 고점 대비 5% 하락 시 익절 (조정된 기준)
            trailing_20_threshold = 20.0 * market_adjustment
            trailing_retrace_20 = 5.0 / market_adjustment
            if max_profit >= trailing_20_threshold and profit_loss_pct <= (max_profit - trailing_retrace_20):
                print(f"📉 {get_korean_name(signal.coin)}: 트레일링 스탑 (최고 {max_profit:.1f}% -> 현재 {profit_loss_pct:.1f}%, 조정: {market_adjustment:.2f}x)")
                return 'trailing_stop'
                
            # 2. 수익 10% 이상 도달 후, 고점 대비 3% 하락 시 익절 (조정된 기준)
            trailing_10_threshold = 10.0 * market_adjustment
            trailing_retrace_10 = 3.0 / market_adjustment
            if max_profit >= trailing_10_threshold and profit_loss_pct <= (max_profit - trailing_retrace_10):
                print(f"📉 {get_korean_name(signal.coin)}: 트레일링 스탑 (최고 {max_profit:.1f}% -> 현재 {profit_loss_pct:.1f}%, 조정: {market_adjustment:.2f}x)")
                return 'trailing_stop'
                
            # 3. 수익 5% 이상 도달 후, 본전(0.5% 이하) 위협 시 익절 (조정된 기준)
            trailing_5_threshold = 5.0 * market_adjustment
            if max_profit >= trailing_5_threshold and profit_loss_pct <= 0.5:
                print(f"🛡️ {get_korean_name(signal.coin)}: 수익 보전 매도 (최고 {max_profit:.1f}% -> 현재 {profit_loss_pct:.1f}%, 조정: {market_adjustment:.2f}x)")
                return 'trailing_stop'
            
            # 🎯 익절 조건 (수익률 50% 이상) - 실전매매와 동일
            if profit_loss_pct >= 50.0:
                return 'take_profit'
            
            # 🎯 손절 조건 (손실 10% 이상) - 실전매매와 동일
            if profit_loss_pct <= -10.0:
                return 'stop_loss'
            
            # 🎯 🆕 [학습 기반 매도] 패턴별 최적 매도 시그널 점수 임계값 조회
            signal_pattern = self._extract_signal_pattern(signal) if hasattr(self, '_extract_signal_pattern') else 'unknown'
            learned_threshold = None
            
            if LEARNED_EXIT_AVAILABLE and signal_pattern != 'unknown':
                # 학습된 최적 임계값 조회 (성공률 50% 이상, 샘플 3회 이상)
                learned_threshold = get_learned_sell_threshold(
                    signal_pattern=signal_pattern,
                    profit_loss_pct=profit_loss_pct,
                    min_success_rate=0.5,
                    min_samples=3
                )
            
            # 🎯 시장 상황에 따른 매도 시그널 임계값 조정
            BASE_SELL_THRESHOLDS = [-0.5, -0.3, -0.2, -0.1]
            adjusted_sell_thresholds = [t * market_adjustment for t in BASE_SELL_THRESHOLDS]
            
            # 🆕 추세 경고가 있으면 매도 임계값을 더 보수적으로 조정 (0.1~0.2 완화)
            trend_adjustment = 0.0
            if trend_sell_signal:
                trend_adjustment = 0.15  # 임계값을 0.15 완화 (더 쉽게 매도)
                print(f"   ⚠️ {get_korean_name(signal.coin)} 추세 경고 반영: 매도 임계값 {trend_adjustment:.2f} 완화")
            
            # 🎯 🆕 학습 기반 매도 조건 (학습된 임계값 우선 사용)
            if learned_threshold is not None:
                # 학습된 임계값이 있으면 그것을 사용 (시장 상황 조정 + 추세 경고 반영)
                adjusted_learned_threshold = (learned_threshold + trend_adjustment) * market_adjustment
                if signal_score < adjusted_learned_threshold:
                    print(f"📚 {get_korean_name(signal.coin)}: 학습 기반 매도 (패턴: {signal_pattern}, "
                          f"임계값: {learned_threshold:.2f} → 조정: {adjusted_learned_threshold:.2f}, "
                          f"현재: {signal_score:.2f}, 추세경고: {trend_sell_reason})")
                    return 'sell'
            else:
                # 학습 데이터가 없으면 기본 임계값 사용 (추세 경고 반영)
                adjusted_sell_thresholds = [t + trend_adjustment for t in adjusted_sell_thresholds]
                if signal_score < adjusted_sell_thresholds[0]:  # 강한 매도 시그널
                    if trend_sell_signal:
                        print(f"📉 {get_korean_name(signal.coin)}: 매도 (시그널: {signal_score:.2f}, 추세경고: {trend_sell_reason})")
                    return 'sell'
                elif signal_score < adjusted_sell_thresholds[1]:  # 매도 시그널
                    if trend_sell_signal:
                        print(f"📉 {get_korean_name(signal.coin)}: 매도 (시그널: {signal_score:.2f}, 추세경고: {trend_sell_reason})")
                    return 'sell'
                elif signal_score < adjusted_sell_thresholds[2]:
                    if trend_sell_signal:
                        print(f"📉 {get_korean_name(signal.coin)}: 매도 (시그널: {signal_score:.2f}, 추세경고: {trend_sell_reason})")
                    return 'sell'
                elif signal_score < adjusted_sell_thresholds[3]:
                    if trend_sell_signal:
                        print(f"📉 {get_korean_name(signal.coin)}: 매도 (시그널: {signal_score:.2f}, 추세경고: {trend_sell_reason})")
                    return 'sell'
            
            # 🎯 학습 기반 매수 조건 (시그널 점수 기반) - 실전매매와 동일
            if signal_score > 0.5:  # 강한 매수 시그널
                return 'buy'
            elif signal_score > 0.3:  # 매수 시그널
                return 'buy'
            elif signal_score > 0.2:
                return 'buy'
            elif signal_score > 0.1:
                return 'buy'
            
            # 🎯 중립 구간 (홀딩)
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
    
    def print_active_positions(self):
        """보유 중인 포지션 요약 출력 (수익률 순)"""
        if not self.positions:
            print("📊 보유 포지션 없음")
            return
            
        print(f"📊 보유 포지션: {len(self.positions)}개")
        
        # 수익률 내림차순 정렬
        sorted_positions = sorted(self.positions.items(), key=lambda x: x[1].profit_loss_pct, reverse=True)
        
        for coin, position in sorted_positions:
            try:
                holding_hours = max(0, position.holding_duration) // 3600
                
                # 가독성을 위한 가격 포맷팅 (소수점 처리)
                entry_price = position.entry_price
                current_price = position.current_price
                
                fmt_entry = f"{entry_price:,.2f}" if entry_price < 100 else f"{entry_price:,.0f}"
                fmt_current = f"{current_price:,.2f}" if current_price < 100 else f"{current_price:,.0f}"
                
                # 진입 시간 포맷팅 (MM-DD HH:MM)
                entry_time_str = datetime.fromtimestamp(position.entry_timestamp).strftime('%m-%d %H:%M')
                holding_time_str = f"{int(holding_hours)}시간 {int((position.holding_duration % 3600) // 60)}분"
                
                print(
                    f"  - {get_korean_name(coin)}: 진입가 {fmt_entry}원, 현재가 {fmt_current}원, "
                    f"수익률 {position.profit_loss_pct:+.2f}%, 매수시간 {entry_time_str}, 보유시간 {holding_time_str}"
                )
            except Exception as e:
                print(f"⚠️ 포지션 출력 오류 ({coin}): {e}")
    
    def print_24h_performance_report(self):
        """24시간 성과 리포트 출력"""
        try:
            current_timestamp = int(datetime.now().timestamp())
            day_ago_timestamp = current_timestamp - (24 * 3600)
            
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                # 24시간 내 거래 히스토리 조회 (🚨 비정상적인 -100% 손실 데이터 제외)
                df = pd.read_sql("""
                    SELECT * FROM virtual_trade_history 
                    WHERE exit_timestamp >= ? 
                    AND profit_loss_pct > -99.0  -- 데이터 오류 필터링
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
                print(f"\n📋 완료된 거래 내역 (24시간, 매수 시간순):")
                
                # 매수 시간 기준 오름차순 정렬 (과거 -> 최신)
                sorted_df = df.sort_values('entry_timestamp', ascending=True)
                
                for _, trade in sorted_df.iterrows():
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
            
            # 🆕 시장 레짐 분석 (Regime 활용)
            market_analysis = self._analyze_market_conditions()
            market_regime = market_analysis.get('regime', 'Neutral')
            
            bonus = 0.0
            
            # 🎯 시장 상황에 따른 적응적 가중치 (Regime 반영)
            if market_context['trend'] == 'bullish' or market_regime == 'Bull':
                # 상승장에서는 다이버전스와 트렌드 강도에 더 높은 가중치
                if signal.rsi_divergence == 'bullish' or signal.macd_divergence == 'bullish':
                    bonus += 0.15  # 상승장에서 다이버전스 보너스 증가
                
                if signal.trend_strength > 0.7:
                    bonus += 0.12  # 상승장에서 트렌드 보너스 증가
            
            elif market_context['trend'] == 'bearish' or market_regime == 'Bear':
                # 하락장에서는 볼린저밴드 스퀴즈와 모멘텀에 더 높은 가중치
                if signal.bb_squeeze > 0.8:
                    bonus += 0.10  # 하락장에서 스퀴즈 보너스 증가
                
                if abs(signal.price_momentum) > 0.05:
                    bonus += 0.08  # 하락장에서 모멘텀 보너스 증가
            
            else:  # 중립장 (Neutral or Volatile)
                # Volatile Regime이면 보너스 대폭 축소 (위험 관리)
                if market_regime == 'Volatile':
                     return 0.0 # 변동성 장에서는 보너스 없음 (보수적 접근)

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
            
            # 🎯 시장 상황 분석 (여기서 시스템 상태 업데이트됨)
            market_analysis = self._analyze_market_conditions()
            market_score = market_analysis.get('score', 0.5)
            
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
    
    def _analyze_market_conditions(self) -> Dict:
        """전체 시장 상황 정밀 분석 (Core 모듈 위임)"""
        try:
            # 🆕 Core MarketAnalyzer 사용 (중복 로직 제거)
            if not hasattr(self, 'core_analyzer') or self.core_analyzer is None:
                from trade.core.market import MarketAnalyzer
                self.core_analyzer = MarketAnalyzer(db_path=TRADING_SYSTEM_DB_PATH)
            
            result = self.core_analyzer.analyze_market_regime()
            
            # 상태 업데이트
            self.update_system_status("market_regime", result.get('regime', 'Neutral'))
            self.update_system_status("market_score", f"{result.get('score', 0.5):.2f}")

            return result
                
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류 (Core): {e}")
            return {'score': 0.5, 'regime': 'Neutral', 'volatility': 0.0}

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🚨 [LEGAL NOTICE] 가상 매매 시뮬레이션 시스템 시작")
    print("="*60)
    print("1. 본 시스템은 AI 학습 및 시장 모니터링 목적의 시뮬레이터입니다.")
    print("2. 생성되는 모든 신호와 거래 내역은 '가상(Virtual)'입니다.")
    print("3. 실제 금전적 투자를 권유하거나 자문하지 않습니다.")
    print("="*60 + "\n")
    
    print("🆕 가상매매 시뮬레이터 시작")
    
    # 시뮬레이터 초기화
    trader = VirtualTrader()
    
    try:
        print("\n🚀 [STEP 1] 보유 포지션 확인")
        # 🆕 포지션 유효성 검증 및 청산 (가장 먼저 수행)
        trader._validate_and_cleanup_positions()
        
        if trader.positions:
            # 🆕 보유 포지션들의 최신 시장 데이터로 일괄 업데이트 (배치 처리)
            trader.update_all_positions()
            
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
            print("보유 코인 (수익률 순):")
            # 수익률 높은 순으로 정렬
            sorted_positions = sorted(trader.positions.items(), key=lambda item: item[1].profit_loss_pct, reverse=True)
            
            for coin, position in sorted_positions:
                holding_seconds = max(0, position.holding_duration)
                holding_hours = holding_seconds // 3600
                holding_minutes = (holding_seconds % 3600) // 60
                
                buy_timestamp_str = "-"
                try:
                    if position.entry_timestamp:
                        buy_timestamp_str = datetime.fromtimestamp(position.entry_timestamp).strftime("%m-%d %H:%M")
                except Exception:
                    buy_timestamp_str = "정보없음"
                
                print(
                    f"   {get_korean_name(coin)}: 진입가 {trader._format_price(position.entry_price)}원, "
                    f"현재가 {trader._format_price(position.current_price)}원, "
                    f"수익률 {position.profit_loss_pct:+.2f}%, 매수시간 {buy_timestamp_str}, "
                    f"보유시간 {holding_hours}시간 {holding_minutes:02d}분"
                )
        else:
            print("📊 보유 포지션 없음")

        print("\n🚀 [STEP 2] 신규 매수 및 보유 코인 매도 진행")
        
        # 🎯 시장 레짐 정보 조회 및 출력 (공통 정보이므로 한 번만 출력)
        regime_info = trader._get_market_regime_info()
        market_regime = regime_info.get('regime', 'Neutral')
        market_score = regime_info.get('score', 0.5)
        print(f"📊 시장 레짐: {market_regime} (점수: {market_score:.2f})")
        
        new_signals = trader.get_new_signals(max_hours_back=24, batch_size=1000)
        
        if new_signals:
            # [Dashboard] 스캔 상태 업데이트
            coins_str = ", ".join([s.coin for s in new_signals[:5]])
            if len(new_signals) > 5: coins_str += "..."
            trader.update_system_status("scanning_coins", coins_str)

            print(f"📊 {len(new_signals)}개 시그널 기반 거래 실행 중...")
            
            # 🆕 [성능 최적화] 시그널 처리 전 가격 일괄 조회 (Prefetch)
            try:
                signal_coins = [s.coin for s in new_signals]
                trader.prefetch_prices(signal_coins)
            except Exception as e:
                print(f"⚠️ Prefetch 오류: {e}")
            
            # 🆕 중복 처리 방지를 위한 세트
            processed_coins = set()
            
            for signal in new_signals:
                # 🆕 이미 처리된 코인은 건너뛰기
                if signal.coin in processed_coins:
                    continue
                
                trader.process_signal(signal)
                processed_coins.add(signal.coin)
            
            print("✅ 거래 실행 완료")
        else:
            print("ℹ️ 새로운 시그널이 없습니다.")
        
        print("\n🚀 [STEP 3] 최종 보유 내역 확인")
        trader.print_active_positions()
        
        print("\n🚀 [STEP 4] 24시간 성과 리포트 출력")
        trader.print_24h_performance_report()
        
        # 🆕 [STEP 5] 사후 분석 및 피드백 학습 (손절 후 반등 체크 등)
        print("\n🚀 [STEP 5] 학습 피드백 처리 (사후 분석)")
        if THOMPSON_SAMPLING_AVAILABLE:
            try:
                # 학습기 인스턴스 생성 및 피드백 처리 실행
                learner = VirtualTradingLearner()
                learner.process_feedback()
            except Exception as e:
                print(f"⚠️ 피드백 처리 오류: {e}")
        else:
            print("ℹ️ 학습기가 활성화되지 않아 피드백 처리를 건너뜁니다.")
        
    except Exception as e:
        print(f"⚠️ 거래 실행 오류: {e}")
        traceback.print_exc()
    
    print("✅ 가상매매 시뮬레이터 완료!")

if __name__ == "__main__":
    main() 