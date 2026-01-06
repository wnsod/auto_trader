"""
db_writer 관련 Mixin 클래스
SignalSelector의 db_writer 기능을 담당합니다.
"""



# === 공통 import ===
import os
import sys
import logging
import traceback
import time
import json
import math
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

# signal_selector 내부 모듈
try:
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir, DB_POOL_AVAILABLE
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )
except ImportError:
    # 직접 실행 시 경로 추가
    _current = os.path.dirname(os.path.abspath(__file__))
    _signal_selector = os.path.dirname(_current)
    _trade = os.path.dirname(_signal_selector)
    sys.path.insert(0, _trade)
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir, DB_POOL_AVAILABLE
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )

# 헬퍼 클래스 import (core에서만 필요)
try:
    from signal_selector.helpers import (
        ContextualBandit, RegimeChangeDetector, ExponentialDecayWeight,
        BayesianSmoothing, ActionSpecificScorer, ContextFeatureExtractor,
        OutlierGuardrail, EvolutionEngine, ContextMemory, RealTimeLearner,
        SignalTradeConnector
    )
except ImportError:
    pass  # 헬퍼가 필요없는 Mixin에서는 무시


class DBWriterMixin:
    """
    DBWriterMixin - db_writer 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

    def create_signal_table(self):
        """시그널 테이블 생성 (엔진 모드에서는 생략 가능하도록 보호)"""
        if os.environ.get('ENGINE_READ_ONLY') == 'true':
            return
            
        try:
            print(f"🚀 시그널 테이블 생성 중: {DB_PATH}")
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        signal_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        action TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        rsi REAL,
                        macd REAL,
                        wave_phase TEXT,
                        pattern_type TEXT,
                        risk_level TEXT,
                        volatility REAL,
                        volume_ratio REAL,
                        wave_progress REAL,
                        structure_score REAL,
                        pattern_confidence REAL,
                        integrated_direction TEXT,
                        integrated_strength REAL,
                        reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        target_price REAL DEFAULT 0.0,
                        source_type TEXT DEFAULT 'quant',
                        UNIQUE(coin, interval, timestamp)
                    )
                """)
                
                # 인덱스 생성
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals(coin)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_combined ON signals(coin, interval) WHERE interval = "combined"')
                
                # 🆕 [증분 검증] validated_at 컬럼 마이그레이션 (없으면 추가)
                cursor = conn.execute("PRAGMA table_info(signals)")
                cols = [row[1] for row in cursor.fetchall()]
                if 'validated_at' not in cols:
                    conn.execute("ALTER TABLE signals ADD COLUMN validated_at INTEGER DEFAULT NULL")
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_unvalidated ON signals(validated_at) WHERE validated_at IS NULL')
                
                conn.commit()
                print(f"✅ 시그널 테이블 생성 완료: {DB_PATH}")
                
        except Exception as e:
            print(f"⚠️ 시그널 테이블 생성 오류: {e}")
    
    def create_enhanced_learning_tables(self):
        """향상된 학습을 위한 추가 테이블들 생성 (엔진 모드 보호)"""
        if os.environ.get('ENGINE_READ_ONLY') == 'true':
            return
            
        try:
            # learning_strategies.db에 테이블 생성
            # 🔧 디렉토리 모드 지원: 폴더면 common_strategies.db 사용
            learning_db_path = STRATEGIES_DB_PATH
            if os.path.isdir(learning_db_path):
                learning_db_path = os.path.join(learning_db_path, 'common_strategies.db')
            
            # 디렉토리가 없으면 생성
            db_dir = os.path.dirname(learning_db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            from trade.core.database import get_db_connection
            with get_db_connection(learning_db_path, read_only=False) as conn:
                # 신뢰도 점수 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reliability_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        reliability_score REAL NOT NULL,
                        sample_count INTEGER NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_id, coin, interval)
                    )
                """)
                
                # 학습 품질 점수 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_quality_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        learning_quality_score REAL NOT NULL,
                        convergence_rate REAL NOT NULL,
                        stability_score REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_id, coin, interval)
                    )
                """)
                
                # 글로벌 전략 매핑 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS global_strategy_mapping (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        global_strategy_id TEXT NOT NULL,
                        mapping_confidence REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, interval)
                    )
                """)
                
                # Walk-Forward 성능 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS walk_forward_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        period_start TIMESTAMP NOT NULL,
                        period_end TIMESTAMP NOT NULL,
                        performance_metrics TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 레짐별 커버리지 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS regime_coverage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        market_regime TEXT NOT NULL,
                        coverage_score REAL NOT NULL,
                        performance_in_regime REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_id, coin, interval, market_regime)
                    )
                """)
                
                # 🆕 누락된 테이블들 추가
                
                # 🆕 통일된 스키마로 시그널 피드백 점수 테이블 생성
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL DEFAULT 'combined',
                        signal_pattern TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        score REAL,  -- strategy_calculator용 (confidence와 동일 값)
                        feedback_type TEXT,  -- strategy_calculator용
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, interval, signal_pattern, feedback_type)
                    )
                """)
                
                # 🆕 기존 컬럼이 없으면 추가 (마이그레이션)
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(signal_feedback_scores)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'coin' not in columns:
                    try:
                        cursor.execute("ALTER TABLE signal_feedback_scores ADD COLUMN coin TEXT DEFAULT 'unknown'")
                        cursor.execute("ALTER TABLE signal_feedback_scores ADD COLUMN interval TEXT DEFAULT 'combined'")
                        cursor.execute("ALTER TABLE signal_feedback_scores ADD COLUMN score REAL")
                        cursor.execute("ALTER TABLE signal_feedback_scores ADD COLUMN feedback_type TEXT")
                        # 기존 데이터에 기본값 설정
                        cursor.execute("UPDATE signal_feedback_scores SET coin = 'unknown', interval = 'combined', score = confidence, feedback_type = 'unknown' WHERE coin IS NULL")
                        conn.commit()
                    except Exception as e:
                        pass  # 마이그레이션 오류는 조용히 무시
                
                # 전략 결과 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        profit REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        trades_count INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        max_drawdown REAL NOT NULL,
                        score REAL NOT NULL,
                        strategy_type TEXT NOT NULL,
                        main_indicator TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        is_learned_strategy INTEGER DEFAULT 0,
                        is_improved_variant INTEGER DEFAULT 0,
                        is_active INTEGER DEFAULT 1,
                        is_archived INTEGER DEFAULT 0,
                        learning_quality_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 프랙탈 분석 결과 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fractal_analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_type TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        fractal_pattern TEXT NOT NULL,
                        pattern_confidence REAL NOT NULL,
                        market_condition TEXT NOT NULL,
                        analysis_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reliability_strategy ON reliability_scores(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reliability_coin ON reliability_scores(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_learning_quality_strategy ON learning_quality_scores(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_learning_quality_coin ON learning_quality_scores(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_global_mapping_coin ON global_strategy_mapping(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_walk_forward_strategy ON walk_forward_performance(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_regime_coverage_strategy ON regime_coverage(strategy_id)')
                
                # 전략 조건 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_conditions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        condition_type TEXT NOT NULL,
                        condition_value TEXT NOT NULL,
                        condition_operator TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 전략 등급 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_grades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        grade TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        performance_score REAL NOT NULL,
                        stability_score REAL NOT NULL,
                        risk_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 🆕 새 테이블 인덱스
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_feedback_pattern ON signal_feedback_scores(signal_pattern)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_results_coin ON strategy_results(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_results_active ON strategy_results(is_active, is_archived)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_fractal_analysis_type ON fractal_analysis_results(analysis_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_fractal_analysis_coin ON fractal_analysis_results(coin, interval)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_conditions_strategy ON strategy_conditions(strategy_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_grades_strategy ON strategy_grades(strategy_id)')
                
                conn.commit()
                print("✅ 향상된 학습 테이블들 생성 완료")
                
        except Exception as e:
            print(f"⚠️ 향상된 학습 테이블 생성 오류: {e}")
    
    def _save_signal_for_learning(self, signal: SignalInfo, signal_pattern: str, market_context: dict):
        """🆕 학습용 시그널 데이터 저장"""
        try:
            # 시그널-매매 연결을 위한 데이터 저장
            signal_data = {
                'coin': signal.coin,
                'interval': signal.interval,
                'timestamp': signal.timestamp,
                'signal_pattern': signal_pattern,
                'market_context': market_context,
                'signal_score': signal.signal_score,
                'confidence': signal.confidence,
                'action': signal.action.value
            }
            
            # 시그널-매매 연결 시스템에 저장
            self.signal_trade_connector.pending_signals[f"{signal.coin}_{signal.timestamp}"] = signal_data
            
        except Exception as e:
            print(f"⚠️ 학습용 시그널 저장 오류: {e}")
    
    def save_signal(self, signal: SignalInfo, silent: bool = False):
        """시그널 저장 (trading_system.db에 저장) - 연결 풀 사용"""
        try:
            if not silent:
                print(f"💾 시그널 저장 중: {signal.coin}/{signal.interval} -> {DB_PATH}")
            
            # 🆕 최적화된 DB 연결 (충돌 방지 강화)
            if DB_POOL_AVAILABLE:
                with get_optimized_db_connection(DB_PATH, mode='write') as conn:
                    self._save_signal_to_db(conn, signal)
            else:
                # Fallback: 직접 연결
                with sqlite3.connect(DB_PATH) as conn:
                    self._save_signal_to_db(conn, signal)
            
            if not silent:
                print(f"✅ 시그널 저장 완료: {signal.coin}/{signal.interval}")
        except Exception as e:
            logger.error(f"❌ 시그널 저장 실패: {e}")

    def save_signals_batch(self, signals: List[SignalInfo]):
        """🚀 [Speed] 대량의 시그널을 하나의 트랜잭션으로 일괄 저장"""
        if not signals: return
        
        try:
            start_t = time.time()
            print(f"📡 {len(signals)}개 시그널 일괄 저장 시작...")
            
            # 🆕 최적화된 DB 연결
            if DB_POOL_AVAILABLE:
                with get_optimized_db_connection(DB_PATH, mode='write') as conn:
                    # 트랜잭션 수동 관리 (성능 극대화)
                    conn.execute("BEGIN TRANSACTION")
                    for sig in signals:
                        self._save_signal_to_db(conn, sig, commit=False)
                    conn.commit()
            else:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("BEGIN TRANSACTION")
                    for sig in signals:
                        self._save_signal_to_db(conn, sig, commit=False)
                    conn.commit()
            
            print(f"✅ 일괄 저장 완료: {len(signals)}개 | 소요: {time.time() - start_t:.3f}s")
        except Exception as e:
            logger.error(f"❌ 시그널 일괄 저장 실패: {e}")

    def _save_signal_to_db(self, conn, signal: SignalInfo, commit: bool = True):
        """실제 시그널 저장 로직 (commit 옵션 추가)"""
        try:
            # 🚨 [Safety] 코인 심볼 유효성 검사 (숫자형 코인 방지)
            if str(signal.coin).isdigit():
                return

            # 먼저 고급지표 컬럼들이 존재하는지 확인하고 없으면 추가
            self._ensure_advanced_columns_exist(conn)
            
            # 컬럼과 값을 명시적으로 매칭하여 INSERT 실행
            columns = [
                'timestamp', 'coin', 'interval', 'signal_score', 'confidence', 'action', 'current_price',
                'rsi', 'macd', 'wave_phase', 'pattern_type', 'risk_level', 'volatility',
                'volume_ratio', 'wave_progress', 'structure_score', 'pattern_confidence',
                'integrated_direction', 'integrated_strength', 'reason',
                'mfi', 'atr', 'adx', 'ma20', 'rsi_ema', 'macd_smoothed', 'wave_momentum',
                'bb_position', 'bb_width', 'bb_squeeze', 'rsi_divergence', 'macd_divergence', 'volume_divergence',
                'price_momentum', 'volume_momentum', 'trend_strength', 'support_resistance', 'fibonacci_levels',
                'elliott_wave', 'harmonic_patterns', 'candlestick_patterns', 'market_structure', 'flow_level_meta', 'pattern_direction',
                'market_condition', 'market_adaptation_bonus', 'target_price', 'source_type'  # 🆕 소스 타입 추가
            ]
            
            # 🆕 문자열 값들을 안전하게 처리
            safe_reason = str(signal.reason).replace('/', '_').replace('\\', '_') if signal.reason else 'unknown'
            safe_wave_phase = str(signal.wave_phase).replace('/', '_').replace('\\', '_') if signal.wave_phase else 'unknown'
            safe_pattern_type = str(signal.pattern_type).replace('/', '_').replace('\\', '_') if signal.pattern_type else 'unknown'
            safe_risk_level = str(signal.risk_level).replace('/', '_').replace('\\', '_') if signal.risk_level else 'unknown'
            safe_integrated_direction = str(signal.integrated_direction).replace('/', '_').replace('\\', '_') if signal.integrated_direction else 'unknown'
            safe_bb_position = str(signal.bb_position).replace('/', '_').replace('\\', '_') if signal.bb_position else 'unknown'
            safe_rsi_divergence = str(signal.rsi_divergence).replace('/', '_').replace('\\', '_') if signal.rsi_divergence else 'none'
            safe_macd_divergence = str(signal.macd_divergence).replace('/', '_').replace('\\', '_') if signal.macd_divergence else 'none'
            safe_volume_divergence = str(signal.volume_divergence).replace('/', '_').replace('\\', '_') if signal.volume_divergence else 'none'
            safe_support_resistance = str(signal.support_resistance).replace('/', '_').replace('\\', '_') if signal.support_resistance else 'unknown'
            safe_fibonacci_levels = str(signal.fibonacci_levels).replace('/', '_').replace('\\', '_') if signal.fibonacci_levels else 'unknown'
            safe_elliott_wave = str(signal.elliott_wave).replace('/', '_').replace('\\', '_') if signal.elliott_wave else 'unknown'
            safe_harmonic_patterns = str(signal.harmonic_patterns).replace('/', '_').replace('\\', '_') if signal.harmonic_patterns else 'none'
            safe_candlestick_patterns = str(signal.candlestick_patterns).replace('/', '_').replace('\\', '_') if signal.candlestick_patterns else 'none'
            safe_market_structure = str(signal.market_structure).replace('/', '_').replace('\\', '_') if signal.market_structure else 'unknown'
            safe_flow_level_meta = str(signal.flow_level_meta).replace('/', '_').replace('\\', '_') if signal.flow_level_meta else 'unknown'
            safe_pattern_direction = str(signal.pattern_direction).replace('/', '_').replace('\\', '_') if signal.pattern_direction else 'neutral'
            safe_market_condition = str(signal.market_condition).replace('/', '_').replace('\\', '_') if signal.market_condition else 'unknown'
            
            # 🆕 target_price 안전 처리
            target_price = getattr(signal, 'target_price', 0.0) if hasattr(signal, 'target_price') else 0.0
            if target_price is None or pd.isna(target_price):
                target_price = 0.0
            
            # 🆕 심볼 정규화: KRW- 제거 (저장 시 표준화)
            coin_symbol = str(signal.coin)
            if coin_symbol.startswith('KRW-'):
                coin_symbol = coin_symbol.replace('KRW-', '')

            values = [
                int(signal.timestamp), coin_symbol, signal.interval, signal.signal_score, 
                signal.confidence, signal.action.value, signal.price, signal.rsi, signal.macd,
                safe_wave_phase, safe_pattern_type, safe_risk_level, signal.volatility,
                signal.volume_ratio, signal.wave_progress, signal.structure_score,
                signal.pattern_confidence, safe_integrated_direction, signal.integrated_strength,
                safe_reason,
                signal.mfi, signal.atr, signal.adx, signal.ma20, signal.rsi_ema, signal.macd_smoothed, signal.wave_momentum,
                safe_bb_position, signal.bb_width, signal.bb_squeeze, safe_rsi_divergence, safe_macd_divergence, safe_volume_divergence,
                signal.price_momentum, signal.volume_momentum, signal.trend_strength, safe_support_resistance, safe_fibonacci_levels,
                safe_elliott_wave, safe_harmonic_patterns, safe_candlestick_patterns, safe_market_structure, safe_flow_level_meta, safe_pattern_direction,
                safe_market_condition, signal.market_adaptation_bonus, target_price, signal.source_type  # 🆕 소스 타입 추가
            ]
            
            # 컬럼과 값의 개수가 일치하는지 확인
            if len(columns) != len(values):
                print(f"⚠️ 컬럼과 값의 개수 불일치: {len(columns)} 컬럼, {len(values)} 값")
                return
            
            placeholders = ', '.join(['?' for _ in columns])
            column_list = ', '.join(columns)
            
            conn.execute(f"""
                INSERT OR REPLACE INTO signals (
                    {column_list}
                ) VALUES ({placeholders})
            """, values)
            
            if commit:
                conn.commit()
        except Exception as e:
            print(f"⚠️ 시그널 저장 오류 ({signal.coin}/{signal.interval}): {e}")

    def save_signal_to_db(self, signal: SignalInfo):
        """Public wrapper for saving signal to database"""
        try:
            if DB_POOL_AVAILABLE:
                with get_optimized_db_connection(DB_PATH, mode='write') as conn:
                    self._save_signal_to_db(conn, signal)
            else:
                # Fallback: 직접 연결
                with sqlite3.connect(DB_PATH) as conn:
                    self._save_signal_to_db(conn, signal)
            print(f"✅ 통합 시그널 저장 완료: {signal.coin}/{signal.interval}")
        except Exception as e:
            print(f"⚠️ 통합 시그널 저장 실패: {e}")

    def _create_synergy_patterns_table(self, cursor):
        """시너지 패턴 테이블 생성"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synergy_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                synergy_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 기본 패턴 데이터 삽입
        patterns = [
            ('bullish_momentum', 'momentum', 'bull', '{"rsi_range": [30, 70], "macd_positive": true, "volume_increase": true}', 0.8, 0.75, 0.6),
            ('bearish_reversal', 'reversal', 'bear', '{"rsi_range": [70, 90], "macd_negative": true, "volume_spike": true}', 0.7, 0.65, 0.455),
            ('sideways_breakout', 'breakout', 'sideways', '{"rsi_range": [40, 60], "macd_neutral": true, "volume_normal": true}', 0.6, 0.55, 0.33),
            ('volatility_surge', 'volatility', 'any', '{"high_volatility": true, "volume_surge": true}', 0.5, 0.45, 0.225)
        ]
        
        cursor.executemany('''
            INSERT INTO synergy_patterns (pattern_name, pattern_type, market_condition, pattern_data, confidence_score, success_rate, synergy_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', patterns)
    

