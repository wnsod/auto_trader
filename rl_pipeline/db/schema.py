"""
데이터베이스 스키마 관리 및 마이그레이션
DDL/마이그레이션, 인덱스 관리
"""

import sqlite3
import logging
import os
from typing import Dict, List, Any, Optional
from rl_pipeline.db.connection_pool import get_strategy_db_pool, get_candle_db_pool
from rl_pipeline.core.errors import DBWriteError
from rl_pipeline.core.env import config

logger = logging.getLogger(__name__)

def ensure_indexes() -> bool:
    """인덱스 존재 여부 확인 및 생성"""
    try:
        # 캔들 데이터베이스 인덱스
        candle_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_candles_coin_interval ON candles(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_coin_interval_timestamp ON candles(coin, interval, timestamp)"
        ]
        
        # 전략 데이터베이스 인덱스
        strategy_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_strategies_coin_interval ON coin_strategies(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_profit ON coin_strategies(profit DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_win_rate ON coin_strategies(win_rate DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON coin_strategies(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_dna_coin ON strategy_dna(coin)",
            "CREATE INDEX IF NOT EXISTS idx_fractal_coin_interval ON fractal_analysis(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_synergy_coin_interval ON synergy_analysis(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_indicator_analysis_coin_interval ON indicator_analysis(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_indicator_analysis_type ON indicator_analysis(analysis_type)",
            # 🆕 AI 학습 최적화 인덱스
            "CREATE INDEX IF NOT EXISTS idx_episode_coin_interval ON learning_episodes(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_episode_start_time ON learning_episodes(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_state_episode_id ON learning_states(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_state_timestamp ON learning_states(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_action_state_id ON learning_actions(state_id)",
            "CREATE INDEX IF NOT EXISTS idx_reward_action_id ON learning_rewards(action_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_strategy ON strategy_performance_history(strategy_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_coin_interval ON strategy_performance_history(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_comparison_strategies ON strategy_comparison_matrix(strategy_a_id, strategy_b_id)",
            "CREATE INDEX IF NOT EXISTS idx_training_model_type ON model_training_data(model_type)",
            "CREATE INDEX IF NOT EXISTS idx_tracking_model_id ON model_performance_tracking(model_id)",
            "CREATE INDEX IF NOT EXISTS idx_global_dna_type ON global_dna_analysis(analysis_type)",
            # 🚀 코인별 분석 비율 최적화 인덱스
            "CREATE INDEX IF NOT EXISTS idx_coin_analysis_ratios_coin ON coin_analysis_ratios(coin)",
            "CREATE INDEX IF NOT EXISTS idx_coin_analysis_ratios_interval ON coin_analysis_ratios(interval)",
            "CREATE INDEX IF NOT EXISTS idx_coin_analysis_ratios_coin_interval ON coin_analysis_ratios(coin, interval)",
            "CREATE INDEX IF NOT EXISTS idx_coin_analysis_ratios_updated_at ON coin_analysis_ratios(updated_at)",
            # 🔥 코인 vs 글로벌 가중치 최적화 인덱스
            "CREATE INDEX IF NOT EXISTS idx_coin_global_weights_coin ON coin_global_weights(coin)",
            "CREATE INDEX IF NOT EXISTS idx_coin_global_weights_updated_at ON coin_global_weights(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_global_fractal_type ON global_fractal_analysis(analysis_type)",
            "CREATE INDEX IF NOT EXISTS idx_global_synergy_type ON global_synergy_analysis(analysis_type)",
            "CREATE INDEX IF NOT EXISTS idx_global_models_type ON global_learning_models(model_type)"
        ]
        
        # 🔒 캔들 DB는 원천 데이터 - 인덱스 생성하지 않음 (읽기 전용)
        logger.debug("⚠️ 캔들 DB는 원천 데이터로 인덱스 생성을 건너뜁니다 (rl_candles.db는 읽기 전용)")
        # candle_pool = get_candle_db_pool()를 사용하지 않음 - 원천 데이터 보호
        
        # 전략 인덱스 생성 (테이블 존재 여부 먼저 확인)
        try:
            strategy_pool = get_strategy_db_pool()
            with strategy_pool.get_connection() as conn:
                cursor = conn.cursor()
                for index_query in strategy_indexes:
                    try:
                        # 테이블 존재 여부 확인
                        table_name = index_query.split(" ON ")[1].split("(")[0].strip()
                        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                        if cursor.fetchone():
                            cursor.execute(index_query)
                        else:
                            logger.debug(f"⚠️ {table_name} 테이블이 존재하지 않아 인덱스 생성을 건너뜁니다")
                    except Exception as e:
                        logger.warning(f"⚠️ 전략 인덱스 생성 건너뜀: {e}")
                conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ 전략 인덱스 생성 실패 (계속 진행): {e}")
        
        logger.info("✅ 인덱스 확인 및 생성 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 인덱스 생성 실패: {e}")
        # 인덱스 생성 실패는 치명적이지 않으므로 경고만 하고 계속 진행
        logger.warning("⚠️ 인덱스 없이 계속 진행합니다")
        return False

def create_candles_table() -> bool:
    """캔들 테이블 생성"""
    try:
        pool = get_candle_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                mfi REAL,
                adx REAL,
                atr REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                volume_ratio REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, interval, timestamp)
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 캔들 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 캔들 테이블 생성 실패: {e}")
        raise DBWriteError(f"캔들 테이블 생성 실패: {e}") from e

def create_coin_strategies_table() -> bool:
    """코인 전략 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS coin_strategies (
                id TEXT PRIMARY KEY,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                strategy_type TEXT DEFAULT 'hybrid',
                strategy_conditions TEXT DEFAULT '{}',
                rsi_min REAL DEFAULT 30.0,
                rsi_max REAL DEFAULT 70.0,
                volume_ratio_min REAL DEFAULT 1.0,
                volume_ratio_max REAL DEFAULT 2.0,
                macd_buy_threshold REAL DEFAULT 0.0,
                macd_sell_threshold REAL DEFAULT 0.0,
                mfi_min REAL DEFAULT 20.0,
                mfi_max REAL DEFAULT 80.0,
                atr_min REAL DEFAULT 0.01,
                atr_max REAL DEFAULT 0.05,
                adx_min REAL DEFAULT 15.0,
                stop_loss_pct REAL DEFAULT 0.02,
                take_profit_pct REAL DEFAULT 0.04,
                profit REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                trades_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                max_drawdown REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                calmar_ratio REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                avg_profit_per_trade REAL DEFAULT 0.0,
                quality_grade TEXT DEFAULT NULL,
                complexity_score REAL DEFAULT 0.0,
                score REAL DEFAULT 0.0,
                ma_period INTEGER DEFAULT 20,
                bb_period INTEGER DEFAULT 20,
                bb_std REAL DEFAULT 2.0,
                market_condition TEXT DEFAULT 'neutral',
                pattern_confidence REAL DEFAULT 0.5,
                pattern_source TEXT DEFAULT 'unknown',
                enhancement_type TEXT DEFAULT 'standard',
                is_active INTEGER DEFAULT 1,
                params TEXT DEFAULT '{}',
                parent_id TEXT DEFAULT NULL,
                regime TEXT DEFAULT NULL,
                similarity_classification TEXT DEFAULT NULL,
                similarity_score REAL DEFAULT NULL,
                parent_strategy_id TEXT DEFAULT NULL
            )
            """

            cursor.execute(create_table_query)

            # 🆕 증분 학습: 기존 테이블에 컬럼 추가 (마이그레이션)
            try:
                cursor.execute("ALTER TABLE coin_strategies ADD COLUMN similarity_classification TEXT DEFAULT NULL")
                logger.info("✅ similarity_classification 컬럼 추가")
            except Exception as e:
                if "duplicate column" in str(e).lower():
                    pass  # 이미 존재
                else:
                    logger.debug(f"similarity_classification 컬럼 추가 실패 (무시): {e}")

            try:
                cursor.execute("ALTER TABLE coin_strategies ADD COLUMN similarity_score REAL DEFAULT NULL")
                logger.info("✅ similarity_score 컬럼 추가")
            except Exception as e:
                if "duplicate column" in str(e).lower():
                    pass
                else:
                    logger.debug(f"similarity_score 컬럼 추가 실패 (무시): {e}")

            try:
                cursor.execute("ALTER TABLE coin_strategies ADD COLUMN parent_strategy_id TEXT DEFAULT NULL")
                logger.info("✅ parent_strategy_id 컬럼 추가")
            except Exception as e:
                if "duplicate column" in str(e).lower():
                    pass
                else:
                    logger.debug(f"parent_strategy_id 컬럼 추가 실패 (무시): {e}")

            conn.commit()

            logger.info("✅ 코인 전략 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 코인 전략 테이블 생성 실패: {e}")
        raise DBWriteError(f"코인 전략 테이블 생성 실패: {e}") from e

def create_selfplay_results_table() -> bool:
    """Self-play 결과 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS selfplay_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                episodes INTEGER NOT NULL,
                results TEXT NOT NULL,
                summary TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(coin, interval, episodes, results)
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ Self-play 결과 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ Self-play 결과 테이블 생성 실패: {e}")
        raise DBWriteError(f"Self-play 결과 테이블 생성 실패: {e}") from e

def create_strategy_dna_table() -> bool:
    """전략 DNA 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS strategy_dna (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT,
                dna_patterns TEXT DEFAULT '{}',
                dna_data TEXT,
                quality_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, interval)
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 전략 DNA 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 전략 DNA 테이블 생성 실패: {e}")
        raise DBWriteError(f"전략 DNA 테이블 생성 실패: {e}") from e

def create_fractal_analysis_table() -> bool:
    """프랙탈 분석 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS fractal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                analysis_type TEXT NOT NULL DEFAULT 'fractal_pattern',
                fractal_score REAL DEFAULT 0.0,
                pattern_distribution TEXT,
                pruned_strategies_count INTEGER DEFAULT 0,
                total_strategies INTEGER DEFAULT 0,
                avg_profit REAL DEFAULT 0.0,
                avg_win_rate REAL DEFAULT 0.0,
                optimal_rsi_min REAL DEFAULT 30.0,
                optimal_rsi_max REAL DEFAULT 70.0,
                optimal_volume_ratio REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, interval)
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 프랙탈 분석 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 프랙탈 분석 테이블 생성 실패: {e}")
        raise DBWriteError(f"프랙탈 분석 테이블 생성 실패: {e}") from e

def create_synergy_analysis_table() -> bool:
    """시너지 분석 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS synergy_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                synergy_score REAL DEFAULT 0.0,
                synergy_patterns TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, interval)
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 시너지 분석 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 시너지 분석 테이블 생성 실패: {e}")
        raise DBWriteError(f"시너지 분석 테이블 생성 실패: {e}") from e

def create_runs_table() -> bool:
    """실행 이력 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    coin TEXT,
                    interval TEXT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    status TEXT DEFAULT 'running',
                    strategies_count INTEGER DEFAULT 0,
                    successful_strategies INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    notes TEXT DEFAULT '',
                    completed_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 실행 이력 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 실행 이력 테이블 생성 실패: {e}")
        raise DBWriteError(f"실행 이력 테이블 생성 실패: {e}") from e

def create_simulation_results_table() -> bool:
    """시뮬레이션 결과 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                total_return REAL DEFAULT 0.0,
                profit REAL DEFAULT 0.0,
                trades_count INTEGER DEFAULT 0,
                profit_loss_ratio REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                calmar_ratio REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                avg_profit_per_trade REAL DEFAULT 0.0,
                final_balance REAL DEFAULT 0.0,
                initial_balance REAL DEFAULT 10000.0,
                simulation_duration INTEGER DEFAULT 0,
                market_volatility REAL DEFAULT 0.0,
                trend_strength REAL DEFAULT 0.0,
                volume_profile TEXT DEFAULT 'normal',
                price_momentum REAL DEFAULT 0.0,
                rsi_avg REAL DEFAULT 50.0,
                macd_signal_strength REAL DEFAULT 0.0,
                bb_position REAL DEFAULT 0.5,
                learning_quality_score REAL DEFAULT 0.5,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES coin_strategies(id)
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 시뮬레이션 결과 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 시뮬레이션 결과 테이블 생성 실패: {e}")
        raise DBWriteError(f"시뮬레이션 결과 테이블 생성 실패: {e}") from e

def create_indicator_analysis_table() -> bool:
    """지표 분석 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS indicator_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                analysis_result TEXT NOT NULL,
                total_trades_analyzed INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ 지표 분석 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 지표 분석 테이블 생성 실패: {e}")
        raise DBWriteError(f"지표 분석 테이블 생성 실패: {e}") from e

def create_dna_analysis_table() -> bool:
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS dna_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                analysis_result TEXT NOT NULL,
                evolved BOOLEAN DEFAULT FALSE,
                total_evolved INTEGER DEFAULT 0,
                data_quality_score REAL DEFAULT 0.0,
                analysis_results TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            logger.info("✅ DNA 분석 테이블 생성 완료")
            
            # 기존 테이블에 누락된 컬럼 추가 (마이그레이션)
            try:
                # analysis_type 컬럼 추가
                cursor.execute("ALTER TABLE dna_analysis ADD COLUMN analysis_type TEXT")
                logger.info("✅ dna_analysis 테이블에 analysis_type 컬럼 추가 완료")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug("analysis_type 컬럼이 이미 존재함")
                else:
                    logger.warning(f"analysis_type 컬럼 추가 실패: {e}")
            
            try:
                # analysis_result 컬럼 추가
                cursor.execute("ALTER TABLE dna_analysis ADD COLUMN analysis_result TEXT")
                logger.info("✅ dna_analysis 테이블에 analysis_result 컬럼 추가 완료")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug("analysis_result 컬럼이 이미 존재함")
                else:
                    logger.warning(f"analysis_result 컬럼 추가 실패: {e}")
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"❌ DNA 분석 테이블 생성 실패: {e}")
        raise DBWriteError(f"DNA 분석 테이블 생성 실패: {e}") from e

def create_global_strategies_table() -> bool:
    """글로벌 전략 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS global_strategies (
                id TEXT PRIMARY KEY,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                params TEXT NOT NULL,
                name TEXT,
                description TEXT,
                dna_hash TEXT,
                source_type TEXT DEFAULT 'synthesized',
                profit REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.5,
                trades_count INTEGER DEFAULT 0,
                quality_grade TEXT DEFAULT 'A',
                market_condition TEXT DEFAULT 'neutral',
                sharpe_ratio REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                regime TEXT DEFAULT NULL,
                rsi_zone TEXT DEFAULT NULL,
                volatility_level TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                meta TEXT,
                strategy_id TEXT,
                analysis_type TEXT,
                analysis_result TEXT,
                global_dna_pattern TEXT,
                global_fractal_score REAL DEFAULT 0.0,
                global_synergy_score REAL DEFAULT 0.0,
                performance_score REAL DEFAULT 0.0,
                similarity_classification TEXT DEFAULT NULL,
                similarity_score REAL DEFAULT NULL,
                parent_strategy_id TEXT DEFAULT NULL,
                zone_key TEXT DEFAULT NULL,
                source_coin TEXT DEFAULT NULL,
                source_strategy_id TEXT DEFAULT NULL,
                FOREIGN KEY (strategy_id) REFERENCES coin_strategies(id)
            )
            """
            
            cursor.execute(create_table_query)
            
            # 🚀 코인별 분석 비율 테이블 생성
            create_coin_analysis_ratios_query = """
            CREATE TABLE IF NOT EXISTS coin_analysis_ratios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                fractal_ratios TEXT DEFAULT '{}',
                multi_timeframe_ratios TEXT DEFAULT '{}',
                indicator_cross_ratios TEXT DEFAULT '{}',
                coin_specific_ratios TEXT DEFAULT '{}',
                volatility_ratios TEXT DEFAULT '{}',
                volume_ratios TEXT DEFAULT '{}',
                optimal_modules TEXT DEFAULT '{}',
                interval_weights TEXT DEFAULT '{}',
                performance_score REAL DEFAULT 0.0,
                accuracy_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, interval, analysis_type)
            )
            """
            
            cursor.execute(create_coin_analysis_ratios_query)

            # 🔥 코인 vs 글로벌 전략 가중치 테이블 생성
            create_coin_global_weights_query = """
            CREATE TABLE IF NOT EXISTS coin_global_weights (
                coin TEXT PRIMARY KEY,
                coin_weight REAL DEFAULT 0.7,
                global_weight REAL DEFAULT 0.3,
                coin_score REAL DEFAULT 0.0,
                global_score REAL DEFAULT 0.0,
                data_quality_score REAL DEFAULT 0.0,
                coin_strategy_count INTEGER DEFAULT 0,
                global_strategy_count INTEGER DEFAULT 0,
                coin_avg_profit REAL DEFAULT 0.0,
                global_avg_profit REAL DEFAULT 0.0,
                coin_win_rate REAL DEFAULT 0.0,
                global_win_rate REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_coin_global_weights_query)
            conn.commit()

            logger.info("✅ 글로벌 전략 테이블 생성 완료")
            logger.info("✅ 코인별 분석 비율 테이블 생성 완료")
            logger.info("✅ 코인 vs 글로벌 가중치 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 테이블 생성 실패: {e}")
        raise DBWriteError(f"글로벌 전략 테이블 생성 실패: {e}") from e

def create_predictive_rl_tables() -> bool:
    """예측형 강화학습 시스템 테이블 생성"""
    try:
        logger.info("🔧 예측형 강화학습 테이블 생성 시작...")
        
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # PRAGMA 설정
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            # 1. rl_episodes 테이블 (예측 발표)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_episodes (
                  episode_id     TEXT PRIMARY KEY,
                  ts_entry       INTEGER NOT NULL,
                  coin           TEXT NOT NULL,
                  interval       TEXT NOT NULL,
                  strategy_id    TEXT NOT NULL,
                  state_key      TEXT NOT NULL,
                  predicted_dir  INTEGER NOT NULL,          -- -1/0/+1
                  predicted_conf REAL    NOT NULL,          -- 0~1
                  entry_price    REAL    NOT NULL,
                  target_move_pct REAL   NOT NULL,
                  horizon_k      INTEGER NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_episodes_1 
                ON rl_episodes(coin, interval, ts_entry)
            """)
            
            # 2. rl_steps 테이블 (스텝별 검증)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_steps (
                  episode_id       TEXT NOT NULL,
                  ts               INTEGER NOT NULL,
                  event            TEXT    NOT NULL,        -- TP/SL/expiry/hold/scalein/scaleout
                  price            REAL    NOT NULL,
                  ret_raw          REAL,
                  ret_signed       REAL,
                  dd_pct_norm      REAL,
                  actual_move_pct  REAL,
                  prox             REAL,
                  dir_correct      INTEGER,
                  reward_dir       REAL,
                  reward_price     REAL,
                  reward_time      REAL,
                  reward_trade     REAL,
                  reward_calib     REAL,
                  reward_risk      REAL,
                  reward_total     REAL,
                  PRIMARY KEY (episode_id, ts)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_steps_1 ON rl_steps(ts)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_steps_2 ON rl_steps(episode_id)
            """)
            
            # 3. rl_episode_summary 테이블 (에피소드 요약)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_episode_summary (
                  episode_id    TEXT PRIMARY KEY,
                  ts_exit       INTEGER,
                  first_event   TEXT,
                  t_hit         INTEGER,
                  realized_ret_signed REAL,
                  total_reward  REAL,
                  acc_flag      INTEGER,
                  coin          TEXT,
                  interval      TEXT,
                  strategy_id   TEXT,
                  source_type   TEXT DEFAULT 'predictive'
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_episode_summary_1 
                ON rl_episode_summary(coin, interval, ts_exit)
            """)
            
            # 4. strategy_grades 테이블 (등급)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_grades (
                  strategy_id TEXT, 
                  coin TEXT, 
                  interval TEXT,
                  total_return REAL,
                  win_rate REAL,
                  predictive_accuracy REAL,
                  grade_score REAL,
                  grade TEXT,
                  updated_at INTEGER,
                  PRIMARY KEY (strategy_id, coin, interval)
                )
            """)
            
            # 7. realtime_predictions 테이블 (실시간 예측 캐시)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS realtime_predictions (
                  ts INTEGER NOT NULL,
                  coin TEXT NOT NULL,
                  interval TEXT NOT NULL,
                  state_key TEXT NOT NULL,
                  predicted_dir INTEGER NOT NULL,
                  predicted_conf REAL NOT NULL,
                  entry_price REAL NOT NULL,
                  target_move_pct REAL NOT NULL,
                  horizon_k INTEGER NOT NULL,
                  p_up REAL,
                  e_ret REAL,
                  prox_est REAL,
                  regime TEXT,
                  source TEXT,
                  PRIMARY KEY (coin, interval, ts)
                )
            """)
            
            # 8. realtime_predictions 뷰 생성
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_realtime_targets AS
                SELECT
                  ts, coin, interval, state_key,
                  predicted_dir, predicted_conf,
                  entry_price, target_move_pct,
                  (entry_price * (1 + target_move_pct)) AS target_price,
                  horizon_k, p_up, e_ret, prox_est, regime, source
                FROM realtime_predictions
            """)
            
            conn.commit()
            logger.info("✅ 예측형 강화학습 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 예측형 강화학습 테이블 생성 실패: {e}")
        raise DBWriteError(f"예측형 강화학습 테이블 생성 실패: {e}") from e

def setup_database_tables() -> bool:
    """모든 데이터베이스 테이블 초기화"""
    try:
        logger.info("🔧 데이터베이스 테이블 초기화 시작...")
        
        # 실제 사용되는 테이블만 생성 (기존 테이블 활용)
        create_candles_table()
        create_coin_strategies_table()
        create_strategy_dna_table()
        create_fractal_analysis_table()
        create_synergy_analysis_table()
        create_runs_table()
        # create_simulation_results_table()  # 🔴 제거: 테스트용, 미사용
        # indicator_analysis 테이블 제거 (미사용)
        create_selfplay_results_table()

        # 🆕 증분 학습용 전략 학습 이력 테이블 생성
        create_strategy_training_history_table()

        # 필요한 테이블만 추가 생성 (기존 테이블 활용)
        # create_market_condition_tables()  # 🔴 제거: 레거시, 미사용
        create_global_strategies_table()  # coin_analysis_ratios 테이블 포함
        
        # 🆕 예측형 강화학습 테이블 생성
        create_predictive_rl_tables()

        # 🆕 Absolute Zero Phase 1 테이블 생성 (라벨링 시스템)
        create_absolute_zero_phase1_tables()

        # 🆕 누락된 테이블 생성 (run_records)
        # 🔥 Absolute Zero에 필요한 테이블 생성
        try:
            create_run_records_table()
            create_regime_routing_results_table()  # 🔥 활성화: Absolute Zero에 필요
            create_integrated_analysis_results_table()  # 🔥 활성화: Paper Trading에 필요
            logger.info("✅ 누락된 테이블 생성 완료")
        except Exception as e:
            logger.warning(f"⚠️ 누락된 테이블 생성 실패(무시 가능): {e}")
        
        # 🔥 누락된 핵심 테이블 생성 (rl_strategy_rollup, rl_state_ensemble)
        try:
            create_strategy_rollup_table()
            create_state_ensemble_table()
            logger.info("✅ 핵심 롤업 테이블 생성 완료")
        except Exception as e:
            logger.warning(f"⚠️ 핵심 롤업 테이블 생성 실패(무시 가능): {e}")
        
        # 🆕 하이브리드 정책 시스템 테이블 생성
        # policy_models와 evaluation_results는 실제 사용되므로 항상 생성
        try:
            create_essential_hybrid_tables()
            add_hybrid_columns_to_strategies()
            logger.info("✅ 하이브리드 정책 시스템 필수 테이블 생성 완료")
        except Exception as e:
            logger.warning(f"⚠️ 하이브리드 필수 테이블 생성 실패(무시 가능): {e}")
        
        # 🔥 Phase 1: 온라인 진화 시스템 스키마 마이그레이션
        try:
            migrate_online_evolution_schema()
            logger.info("✅ 온라인 진화 시스템 스키마 마이그레이션 완료")
        except Exception as e:
            logger.warning(f"⚠️ 온라인 진화 시스템 스키마 마이그레이션 실패(무시 가능): {e}")
        
        # training_runs 테이블은 선택적 생성 (현재 미사용)
        enable_training_runs = os.getenv('ENABLE_TRAINING_RUNS_TABLE', 'false').lower() == 'true'
        if enable_training_runs:
            try:
                create_training_runs_table()
                logger.info("✅ training_runs 테이블 생성 완료")
            except Exception as e:
                logger.warning(f"⚠️ training_runs 테이블 생성 실패(무시 가능): {e}")
        else:
            logger.debug("ℹ️ training_runs 테이블 생성 건너뜀 (ENABLE_TRAINING_RUNS_TABLE=false)")
        
        # 기존 테이블 마이그레이션
        migrate_coin_strategies_table()
        # migrate_simulation_results_table()  # 🔴 제거: 테이블 삭제됨, 불필요한 경고 방지
        migrate_global_strategies_table()
        migrate_rl_episode_summary_table()  # 🔥 옵션 A: source_type 컬럼 추가
        migrate_coin_analysis_ratios_table()  # 🔥 interval_weights 컬럼 추가
        migrate_coin_global_weights_table()  # 🔥 coin vs global 가중치 테이블 생성

        # 인덱스 생성
        ensure_indexes()
        
        logger.info("✅ 모든 데이터베이스 테이블 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 테이블 초기화 실패: {e}")
        raise DBWriteError(f"데이터베이스 테이블 초기화 실패: {e}") from e

def migrate_coin_strategies_table() -> bool:
    """coin_strategies 테이블 마이그레이션 - 누락된 컬럼 추가"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 기존 테이블에 누락된 컬럼만 추가 (이미 존재하는 컬럼은 제외)
            columns_to_add = [
                ("strategy_conditions", "TEXT DEFAULT '{}'"),  # 로그 오류 해결을 위해 필요한 컬럼
                ("ma_period", "INTEGER DEFAULT 20"),
                ("bb_period", "INTEGER DEFAULT 20"),
                ("bb_std", "REAL DEFAULT 2.0"),
                ("market_condition", "TEXT DEFAULT 'neutral'"),
                ("pattern_confidence", "REAL DEFAULT 0.5"),
                ("pattern_source", "TEXT DEFAULT 'unknown'"),
                ("enhancement_type", "TEXT DEFAULT 'standard'"),
                ("is_active", "INTEGER DEFAULT 1"),
                ("params", "TEXT DEFAULT '{}'"),  # 전략 파라미터 저장용
                # 🆕 핵심 지표 min/max 컬럼 추가
                ("mfi_min", "REAL DEFAULT 20.0"),
                ("mfi_max", "REAL DEFAULT 80.0"),
                ("atr_min", "REAL DEFAULT 0.01"),
                ("atr_max", "REAL DEFAULT 0.05"),
                ("adx_min", "REAL DEFAULT 15.0"),
                # 🔥 Phase 1: 온라인 진화 시스템용 컬럼 추가
                ("parent_id", "TEXT"),
                ("version", "INTEGER DEFAULT 1"),
                ("last_train_end_idx", "INTEGER"),
                ("online_pf", "REAL DEFAULT 0.0"),
                ("online_return", "REAL DEFAULT 0.0"),
                ("online_mdd", "REAL DEFAULT 0.0"),
                ("online_updates_count", "INTEGER DEFAULT 0"),
                ("consistency_score", "REAL DEFAULT 0.0")
            ]
            
            for column_name, column_def in columns_to_add:
                try:
                    cursor.execute(f"ALTER TABLE coin_strategies ADD COLUMN {column_name} {column_def}")
                    logger.info(f"✅ 컬럼 추가 완료: {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"컬럼 이미 존재: {column_name}")
                    else:
                        logger.warning(f"컬럼 추가 실패: {column_name} - {e}")
            
            # strategy_dna 테이블에 dna_data 컬럼 추가
            try:
                cursor.execute("ALTER TABLE strategy_dna ADD COLUMN dna_data TEXT")
                logger.info("✅ strategy_dna 테이블에 dna_data 컬럼 추가 완료")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug("dna_data 컬럼이 이미 존재함")
                else:
                    logger.warning(f"dna_data 컬럼 추가 실패: {e}")
            
            # fractal_analysis 테이블에 누락된 컬럼들 추가
            fractal_columns = [
                ("analysis_type", "TEXT NOT NULL DEFAULT 'fractal_pattern'"),
                ("total_strategies", "INTEGER DEFAULT 0"),
                ("avg_profit", "REAL DEFAULT 0.0"),
                ("avg_win_rate", "REAL DEFAULT 0.0"),
                ("optimal_rsi_min", "REAL DEFAULT 30.0"),
                ("optimal_rsi_max", "REAL DEFAULT 70.0"),
                ("optimal_volume_ratio", "REAL DEFAULT 1.0")
            ]
            
            for column_name, column_def in fractal_columns:
                try:
                    cursor.execute(f"ALTER TABLE fractal_analysis ADD COLUMN {column_name} {column_def}")
                    logger.info(f"✅ fractal_analysis 컬럼 추가 완료: {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"fractal_analysis 컬럼 이미 존재: {column_name}")
                    else:
                        logger.warning(f"fractal_analysis 컬럼 추가 실패: {column_name} - {e}")
            
            conn.commit()
            logger.info("✅ coin_strategies 테이블 마이그레이션 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ coin_strategies 테이블 마이그레이션 실패: {e}")
        return False

def migrate_simulation_results_table() -> bool:
    """simulation_results 테이블 마이그레이션 - 누락된 컬럼 추가"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 누락된 컬럼들 추가 (학습 데이터용 컬럼 포함)
            columns_to_add = [
                ("total_return", "REAL DEFAULT 0.0"),
                ("profit", "REAL DEFAULT 0.0"),
                ("trades_count", "INTEGER DEFAULT 0"),
                # 학습 데이터 확보를 위한 추가 컬럼들
                ("market_volatility", "REAL DEFAULT 0.0"),
                ("trend_strength", "REAL DEFAULT 0.0"),
                ("volume_profile", "TEXT DEFAULT 'normal'"),
                ("price_momentum", "REAL DEFAULT 0.0"),
                ("rsi_avg", "REAL DEFAULT 50.0"),
                ("macd_signal_strength", "REAL DEFAULT 0.0"),
                ("bb_position", "REAL DEFAULT 0.5"),
                ("learning_quality_score", "REAL DEFAULT 0.5")
            ]
            
            for column_name, column_def in columns_to_add:
                try:
                    cursor.execute(f"ALTER TABLE simulation_results ADD COLUMN {column_name} {column_def}")
                    logger.info(f"✅ simulation_results 테이블에 {column_name} 컬럼 추가 완료")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"{column_name} 컬럼이 이미 존재함")
                    else:
                        logger.warning(f"{column_name} 컬럼 추가 실패: {e}")
            
            conn.commit()
            logger.info("✅ simulation_results 테이블 마이그레이션 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ simulation_results 테이블 마이그레이션 실패: {e}")
        return False

def migrate_rl_episode_summary_table() -> bool:
    """rl_episode_summary 테이블에 source_type 컬럼 추가"""
    try:
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # source_type 컬럼 존재 여부 확인
            cursor.execute("PRAGMA table_info(rl_episode_summary)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'source_type' not in columns:
                try:
                    cursor.execute("ALTER TABLE rl_episode_summary ADD COLUMN source_type TEXT DEFAULT 'predictive'")
                    conn.commit()
                    logger.info("✅ rl_episode_summary.source_type 컬럼 추가 완료")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.debug("⚠️ rl_episode_summary.source_type 컬럼 이미 존재")
            else:
                logger.debug("⚠️ rl_episode_summary.source_type 컬럼 이미 존재")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ rl_episode_summary 테이블 마이그레이션 실패: {e}")
        return False

def migrate_global_strategies_table() -> bool:
    """global_strategies 테이블 마이그레이션 - 누락된 컬럼 추가"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_strategies'")
            if not cursor.fetchone():
                logger.debug("global_strategies 테이블이 존재하지 않음 (새로 생성될 예정)")
                return True
            
            # 누락된 컬럼들 추가 (실제 사용되는 컬럼들)
            columns_to_add = [
                ("coin", "TEXT"),
                ("interval", "TEXT"),
                ("strategy_type", "TEXT"),
                ("params", "TEXT"),
                ("name", "TEXT"),
                ("description", "TEXT"),
                ("dna_hash", "TEXT"),
                ("source_type", "TEXT DEFAULT 'synthesized'"),
                ("profit", "REAL DEFAULT 0.0"),
                ("profit_factor", "REAL DEFAULT 0.0"),
                ("win_rate", "REAL DEFAULT 0.5"),
                ("trades_count", "INTEGER DEFAULT 0"),
                ("quality_grade", "TEXT DEFAULT 'A'"),
                ("market_condition", "TEXT DEFAULT 'neutral'"),
                ("sharpe_ratio", "REAL DEFAULT 0.0"),
                ("max_drawdown", "REAL DEFAULT 0.0"),
                ("regime", "TEXT DEFAULT NULL"),
                ("rsi_zone", "TEXT DEFAULT NULL"),
                ("volatility_level", "TEXT DEFAULT NULL"),
                ("created_at", "TEXT"),
                ("updated_at", "TEXT"),
                ("meta", "TEXT"),
                ("similarity_classification", "TEXT DEFAULT NULL"),
                ("similarity_score", "REAL DEFAULT NULL"),
                ("parent_strategy_id", "TEXT DEFAULT NULL"),
                ("zone_key", "TEXT DEFAULT NULL"),
                ("source_coin", "TEXT DEFAULT NULL"),
                ("source_strategy_id", "TEXT DEFAULT NULL")
            ]
            
            for column_name, column_def in columns_to_add:
                try:
                    cursor.execute(f"ALTER TABLE global_strategies ADD COLUMN {column_name} {column_def}")
                    logger.info(f"✅ global_strategies 테이블에 {column_name} 컬럼 추가 완료")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"global_strategies.{column_name} 컬럼이 이미 존재함")
                    else:
                        logger.warning(f"global_strategies.{column_name} 컬럼 추가 실패: {e}")
            
            conn.commit()
            logger.info("✅ global_strategies 테이블 마이그레이션 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ global_strategies 테이블 마이그레이션 실패: {e}")
        return False

def migrate_coin_analysis_ratios_table() -> bool:
    """coin_analysis_ratios 테이블 마이그레이션 - interval_weights 컬럼 추가"""
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_analysis_ratios'")
            if not cursor.fetchone():
                logger.debug("coin_analysis_ratios 테이블이 존재하지 않음 (새로 생성될 예정)")
                return True

            # interval_weights 컬럼 추가
            try:
                cursor.execute("ALTER TABLE coin_analysis_ratios ADD COLUMN interval_weights TEXT DEFAULT '{}'")
                logger.info("✅ coin_analysis_ratios 테이블에 interval_weights 컬럼 추가 완료")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug("coin_analysis_ratios.interval_weights 컬럼이 이미 존재함")
                else:
                    logger.warning(f"coin_analysis_ratios.interval_weights 컬럼 추가 실패: {e}")

            conn.commit()
            logger.info("✅ coin_analysis_ratios 테이블 마이그레이션 완료")
            return True

    except Exception as e:
        logger.error(f"❌ coin_analysis_ratios 테이블 마이그레이션 실패: {e}")
        return False

def migrate_coin_global_weights_table() -> bool:
    """coin_global_weights 테이블 생성 및 마이그레이션"""
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_global_weights'")
            if cursor.fetchone():
                logger.debug("coin_global_weights 테이블이 이미 존재함")
                return True

            # 테이블 생성 (create_global_strategies_table에서 생성되지 않은 경우)
            create_query = """
            CREATE TABLE IF NOT EXISTS coin_global_weights (
                coin TEXT PRIMARY KEY,
                coin_weight REAL DEFAULT 0.7,
                global_weight REAL DEFAULT 0.3,
                coin_score REAL DEFAULT 0.0,
                global_score REAL DEFAULT 0.0,
                data_quality_score REAL DEFAULT 0.0,
                coin_strategy_count INTEGER DEFAULT 0,
                global_strategy_count INTEGER DEFAULT 0,
                coin_avg_profit REAL DEFAULT 0.0,
                global_avg_profit REAL DEFAULT 0.0,
                coin_win_rate REAL DEFAULT 0.0,
                global_win_rate REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_query)

            conn.commit()
            logger.info("✅ coin_global_weights 테이블 마이그레이션 완료")
            return True

    except Exception as e:
        logger.error(f"❌ coin_global_weights 테이블 마이그레이션 실패: {e}")
        return False

def migrate() -> bool:
    """데이터베이스 마이그레이션 실행"""
    try:
        logger.info("🔄 데이터베이스 마이그레이션 시작...")
        
        # 기존 테이블 구조 확인 및 업데이트
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 기존 테이블에 누락된 컬럼 추가
            migrations = [
                # coin_strategies 테이블 마이그레이션
                "ALTER TABLE coin_strategies ADD COLUMN strategy_type TEXT DEFAULT 'hybrid'",
                "ALTER TABLE coin_strategies ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP",
                
                # strategy_dna 테이블 마이그레이션
                "ALTER TABLE strategy_dna ADD COLUMN interval TEXT",
                
                # 기타 마이그레이션들...
            ]
            
            for migration in migrations:
                try:
                    cursor.execute(migration)
                    logger.debug(f"✅ 마이그레이션 실행: {migration}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e):
                        logger.debug(f"⚠️ 컬럼이 이미 존재함: {migration}")
                    else:
                        logger.warning(f"⚠️ 마이그레이션 실패: {migration} -> {e}")
            
            conn.commit()
        
        logger.info("✅ 데이터베이스 마이그레이션 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 마이그레이션 실패: {e}")
        raise DBWriteError(f"데이터베이스 마이그레이션 실패: {e}") from e

def check_database_integrity(db_path: str = None) -> bool:
    """데이터베이스 무결성 검사"""
    try:
        if db_path:
            pool = get_candle_db_pool() if 'candles' in db_path else get_strategy_db_pool()
        else:
            pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] == "ok":
                logger.info("✅ 데이터베이스 무결성 검사 통과")
                return True
            else:
                logger.error(f"❌ 데이터베이스 무결성 검사 실패: {result[0]}")
                return False
                
    except Exception as e:
        logger.error(f"❌ 데이터베이스 무결성 검사 실패: {e}")
        return False

def create_market_condition_tables() -> bool:
    """시장 상황 분석 테이블 생성"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 시장 상황 분석 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_condition_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                confidence REAL NOT NULL,
                analysis_data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, interval, market_condition)
            )
            """)
            
            # 미사용 테이블 제거됨:
            # - dna_market_analysis (strategy_dna로 대체)
            # - fractal_market_analysis (fractal_analysis로 대체)
            # - routing_market_analysis (regime_routing_results로 대체)
            
            conn.commit()
            logger.info("✅ 시장 상황 분석 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 시장 상황 분석 테이블 생성 실패: {e}")
        return False

def repair_database(db_path: str = None) -> bool:
    """손상된 데이터베이스 복구 시도"""
    try:
        if db_path:
            pool = get_candle_db_pool() if 'candles' in db_path else get_strategy_db_pool()
        else:
            pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # VACUUM 실행으로 데이터베이스 최적화
            cursor.execute("VACUUM")
            
            # REINDEX 실행으로 인덱스 재구성
            cursor.execute("REINDEX")
            
            logger.info("✅ 데이터베이스 복구 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 데이터베이스 복구 실패: {e}")
        return False

def create_essential_hybrid_tables() -> bool:
    """하이브리드 정책 시스템 필수 테이블 생성 (항상 생성됨)
    
    실제 사용되는 테이블:
    - policy_models: trainer_jax.py에서 모델 저장
    - evaluation_results: evaluator.py에서 평가 결과 저장
    """
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # policy_models 테이블 (trainer_jax.py에서 사용)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS policy_models (
                model_id TEXT PRIMARY KEY,
                algo TEXT NOT NULL DEFAULT 'PPO',
                features_ver TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ckpt_path TEXT NOT NULL,
                notes TEXT
            )
            """)
            
            # evaluation_results 테이블 (evaluator.py에서 사용)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                eval_id TEXT PRIMARY KEY,
                model_id TEXT,
                mode TEXT NOT NULL,
                asset TEXT NOT NULL,
                interval TEXT NOT NULL,
                period_from DATETIME NOT NULL,
                period_to DATETIME NOT NULL,
                profit_factor REAL,
                total_return REAL,
                win_rate REAL,
                mdd REAL,
                sharpe REAL,
                trades INTEGER,
                latency_ms_p95 REAL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES policy_models(model_id)
            )
            """)
            
            # hybrid_models 테이블 (auto_trainer.py에서 사용)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hybrid_models (
                model_id TEXT PRIMARY KEY,
                coin TEXT NOT NULL,
                interval TEXT,
                status TEXT NOT NULL DEFAULT 'training',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                epochs INTEGER,
                final_loss REAL,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES policy_models(model_id)
            )
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_policy_models_algo ON policy_models(algo)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_policy_models_created ON policy_models(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_model ON evaluation_results(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_mode ON evaluation_results(mode)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_asset_interval ON evaluation_results(asset, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hybrid_models_coin ON hybrid_models(coin)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hybrid_models_status ON hybrid_models(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hybrid_models_created ON hybrid_models(created_at)")
            
            conn.commit()
            logger.info("✅ 하이브리드 정책 시스템 필수 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 하이브리드 정책 필수 테이블 생성 실패: {e}")
        return False

def create_training_runs_table() -> bool:
    """training_runs 테이블 생성 (선택적, 현재 미사용)"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # training_runs 테이블 (현재 미사용, 향후 확장용)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                start_at DATETIME NOT NULL,
                end_at DATETIME,
                epochs INTEGER,
                steps INTEGER,
                reward_scale REAL,
                entropy_coef REAL,
                lr REAL,
                train_return REAL,
                train_pf REAL,
                loss_pi REAL,
                loss_vf REAL,
                FOREIGN KEY (model_id) REFERENCES policy_models(model_id)
            )
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_start ON training_runs(start_at)")
            
            conn.commit()
            logger.info("✅ training_runs 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ training_runs 테이블 생성 실패: {e}")
        return False

def add_hybrid_columns_to_strategies() -> bool:
    """기존 전략 테이블에 하이브리드 컬럼 추가"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # coin_strategies 테이블에 컬럼 추가 (없는 경우만)
            try:
                cursor.execute("ALTER TABLE coin_strategies ADD COLUMN hybrid_score REAL")
                logger.debug("✅ coin_strategies.hybrid_score 컬럼 추가")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                logger.debug("⚠️ coin_strategies.hybrid_score 컬럼 이미 존재")
            
            try:
                cursor.execute("ALTER TABLE coin_strategies ADD COLUMN model_id TEXT")
                logger.debug("✅ coin_strategies.model_id 컬럼 추가")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                logger.debug("⚠️ coin_strategies.model_id 컬럼 이미 존재")
            
            # global_strategies 테이블에도 추가 (테이블이 있는 경우)
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_strategies'")
                if cursor.fetchone():
                    try:
                        cursor.execute("ALTER TABLE global_strategies ADD COLUMN hybrid_score REAL")
                        logger.debug("✅ global_strategies.hybrid_score 컬럼 추가")
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            raise
                        logger.debug("⚠️ global_strategies.hybrid_score 컬럼 이미 존재")
                    
                    try:
                        cursor.execute("ALTER TABLE global_strategies ADD COLUMN model_id TEXT")
                        logger.debug("✅ global_strategies.model_id 컬럼 추가")
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            raise
                        logger.debug("⚠️ global_strategies.model_id 컬럼 이미 존재")
            except Exception as e:
                logger.debug(f"⚠️ global_strategies 테이블 처리 중 오류 (계속 진행): {e}")
            
            conn.commit()
            logger.info("✅ 하이브리드 컬럼 추가 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ 하이브리드 컬럼 추가 실패: {e}")
        return False

def create_strategy_lineage_table() -> bool:
    """strategy_lineage 테이블 생성 - 전략 진화 계보 추적"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS strategy_lineage (
                child_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                mutation_desc TEXT,
                segment_range TEXT,  -- JSON: {"start_idx": 100, "end_idx": 200}
                improvement_flag INTEGER DEFAULT 0,  -- 0: 개선 없음, 1: 개선됨
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (child_id),
                FOREIGN KEY (parent_id) REFERENCES coin_strategies(id),
                FOREIGN KEY (child_id) REFERENCES coin_strategies(id)
            )
            """
            
            cursor.execute(create_table_query)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_parent 
                ON strategy_lineage(parent_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_created 
                ON strategy_lineage(created_at DESC)
            """)
            
            conn.commit()
            logger.info("✅ strategy_lineage 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ strategy_lineage 테이블 생성 실패: {e}")
        return False

def create_segment_scores_table() -> bool:
    """segment_scores 테이블 생성 - 세그먼트별 성과 기록"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS segment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                market TEXT NOT NULL,  -- 코인 정보
                interval TEXT NOT NULL,
                start_idx INTEGER NOT NULL,
                end_idx INTEGER NOT NULL,
                start_timestamp INTEGER,  -- 디버깅용
                end_timestamp INTEGER,
                profit REAL DEFAULT 0.0,
                pf REAL DEFAULT 0.0,
                sharpe REAL DEFAULT 0.0,
                mdd REAL DEFAULT 0.0,
                trades_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES coin_strategies(id)
            )
            """
            
            cursor.execute(create_table_query)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_segment_scores_strategy 
                ON segment_scores(strategy_id, market, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_segment_scores_range 
                ON segment_scores(start_idx, end_idx)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_segment_scores_timestamp 
                ON segment_scores(start_timestamp, end_timestamp)
            """)
            
            conn.commit()
            logger.info("✅ segment_scores 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ segment_scores 테이블 생성 실패: {e}")
        return False

def create_run_records_table() -> bool:
    """run_records 테이블 생성 - 실행 기록 추적"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_records (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    coin TEXT,
                    interval TEXT,
                    strategies_count INTEGER DEFAULT 0,
                    successful_strategies INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_records_coin_interval 
                ON run_records(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_records_status 
                ON run_records(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_records_created_at 
                ON run_records(created_at DESC)
            """)
            
            conn.commit()
            logger.info("✅ run_records 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ run_records 테이블 생성 실패: {e}")
        return False

def create_regime_routing_results_table() -> bool:
    """regime_routing_results 테이블 생성 - 레짐 라우팅 결과 저장"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regime_routing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime_detected TEXT NOT NULL,
                    regime_confidence REAL DEFAULT 0.5,
                    regime_transition_prob REAL DEFAULT 0.0,
                    matched_strategies INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_regime_routing_coin_interval 
                ON regime_routing_results(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_regime_routing_transition_prob 
                ON regime_routing_results(regime_transition_prob DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_regime_routing_created_at 
                ON regime_routing_results(created_at DESC)
            """)
            
            conn.commit()
            logger.info("✅ regime_routing_results 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ regime_routing_results 테이블 생성 실패: {e}")
        return False

def create_integrated_analysis_results_table() -> bool:
    """integrated_analysis_results 테이블 생성 - 통합 분석 결과 저장"""
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS integrated_analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime TEXT NOT NULL DEFAULT 'neutral',

                    -- 분석 결과
                    fractal_score REAL DEFAULT 0.0,
                    multi_timeframe_score REAL DEFAULT 0.0,
                    indicator_cross_score REAL DEFAULT 0.0,

                    -- JAX 앙상블 결과
                    ensemble_score REAL DEFAULT 0.0,
                    ensemble_confidence REAL DEFAULT 0.0,

                    -- 최종 시그널 점수
                    final_signal_score REAL DEFAULT 0.0,
                    signal_confidence REAL DEFAULT 0.0,
                    signal_action TEXT DEFAULT 'hold',

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_integrated_analysis_coin_interval
                ON integrated_analysis_results(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_integrated_analysis_final_signal_score
                ON integrated_analysis_results(final_signal_score DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_integrated_analysis_created_at
                ON integrated_analysis_results(created_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_integrated_analysis_regime
                ON integrated_analysis_results(regime)
            """)
            
            conn.commit()
            logger.info("✅ integrated_analysis_results 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ integrated_analysis_results 테이블 생성 실패: {e}")
        return False

def migrate_online_evolution_schema() -> bool:
    """온라인 진화 시스템 스키마 마이그레이션 (Phase 1)"""
    try:
        logger.info("🔄 온라인 진화 시스템 스키마 마이그레이션 시작...")
        
        # 1. coin_strategies 테이블에 온라인 진화 컬럼 추가
        result1 = migrate_coin_strategies_table()
        
        # 2. strategy_lineage 테이블 생성
        result2 = create_strategy_lineage_table()
        
        # 3. segment_scores 테이블 생성
        result3 = create_segment_scores_table()
        
        if result1 and result2 and result3:
            logger.info("✅ 온라인 진화 시스템 스키마 마이그레이션 완료")
            return True
        else:
            logger.warning("⚠️ 일부 마이그레이션 실패")
            return False

    except Exception as e:
        logger.error(f"❌ 온라인 진화 시스템 스키마 마이그레이션 실패: {e}")
        return False

def create_strategy_rollup_table() -> bool:
    """rl_strategy_rollup 테이블 생성 - 전략별 롤업 통계"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_strategy_rollup (
                    strategy_id TEXT PRIMARY KEY,
                    coin TEXT,
                    interval TEXT,
                    episodes_trained INTEGER DEFAULT 0,
                    avg_ret REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    predictive_accuracy REAL DEFAULT 0.0,
                    avg_dd REAL DEFAULT 0.0,
                    total_episodes INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0.0,
                    avg_reward REAL DEFAULT 0.0,
                    avg_profit_factor REAL DEFAULT 0.0,
                    avg_sharpe_ratio REAL DEFAULT 0.0,
                    best_episode_reward REAL DEFAULT 0.0,
                    worst_episode_reward REAL DEFAULT 0.0,
                    grade TEXT DEFAULT 'UNKNOWN',
                    updated_at INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rollup_strategy 
                ON rl_strategy_rollup(strategy_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rollup_coin 
                ON rl_strategy_rollup(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rollup_grade 
                ON rl_strategy_rollup(grade)
            """)
            
            conn.commit()
            logger.info("✅ rl_strategy_rollup 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ rl_strategy_rollup 테이블 생성 실패: {e}")
        return False

def create_state_ensemble_table() -> bool:
    """rl_state_ensemble 테이블 생성 - 상태 앙상블 예측"""
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_state_ensemble (
                    state_key TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    acc_up INTEGER DEFAULT 0,
                    acc_down INTEGER DEFAULT 0,
                    acc_total REAL DEFAULT 0.0,
                    p_up_smooth REAL DEFAULT 0.0,
                    e_ret_smooth REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    last_updated INTEGER DEFAULT 0,
                    state_id TEXT,
                    timestamp TIMESTAMP,
                    ensemble_prediction REAL DEFAULT 0.0,
                    strategy_count INTEGER DEFAULT 0,
                    top_strategies TEXT,
                    market_regime TEXT,
                    rsi REAL,
                    volume_ratio REAL,
                    atr REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (coin, interval, state_key)
                )
            """)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ensemble_coin
                ON rl_state_ensemble(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ensemble_time
                ON rl_state_ensemble(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ensemble_regime
                ON rl_state_ensemble(market_regime)
            """)

            conn.commit()
            logger.info("✅ rl_state_ensemble 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌ rl_state_ensemble 테이블 생성 실패: {e}")
        return False

def create_absolute_zero_phase1_tables() -> bool:
    """Absolute Zero Phase 1 테이블 생성 - 라벨링 시스템"""
    try:
        logger.info("🔧 Absolute Zero Phase 1 테이블 생성 시작...")
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 1. strategy_signal_labels 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_signal_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime_tag TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    r_max REAL NOT NULL,
                    k_max INTEGER NOT NULL,
                    r_min REAL NOT NULL,
                    k_min INTEGER NOT NULL,
                    fee_bps REAL DEFAULT 10.0,
                    slippage_bps REAL DEFAULT 5.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_labels_coin_interval
                ON strategy_signal_labels(coin, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_labels_regime
                ON strategy_signal_labels(regime_tag)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_labels_strategy
                ON strategy_signal_labels(strategy_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_labels_ts
                ON strategy_signal_labels(ts)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_labels_composite
                ON strategy_signal_labels(coin, interval, regime_tag, strategy_id)
            """)

            # 2. strategy_label_stats 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_label_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime_tag TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    rmax_mean REAL DEFAULT 0.0,
                    rmax_median REAL DEFAULT 0.0,
                    rmax_p75 REAL DEFAULT 0.0,
                    rmax_p90 REAL DEFAULT 0.0,
                    rmin_mean REAL DEFAULT 0.0,
                    rmin_median REAL DEFAULT 0.0,
                    rmin_p25 REAL DEFAULT 0.0,
                    rmin_p10 REAL DEFAULT 0.0,
                    kmax_mean REAL DEFAULT 0.0,
                    kmax_median INTEGER DEFAULT 0,
                    kmin_mean REAL DEFAULT 0.0,
                    kmin_median INTEGER DEFAULT 0,
                    pf REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    mdd REAL DEFAULT 0.0,
                    n_signals INTEGER DEFAULT 0,
                    last_updated INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, interval, regime_tag, strategy_id)
                )
            """)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_label_stats_composite
                ON strategy_label_stats(coin, interval, regime_tag, strategy_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_label_stats_pf
                ON strategy_label_stats(pf DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_label_stats_n_signals
                ON strategy_label_stats(n_signals DESC)
            """)

            # 3. strategy_grades 테이블은 이미 create_predictive_rl_tables()에 존재하므로
            # 컬럼 추가만 수행
            try:
                cursor.execute("ALTER TABLE strategy_grades ADD COLUMN explain TEXT")
                logger.info("✅ strategy_grades.explain 컬럼 추가 완료")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"strategy_grades.explain 컬럼 추가 실패: {e}")
                else:
                    logger.debug("strategy_grades.explain 컬럼 이미 존재")

            conn.commit()
            logger.info("✅ Absolute Zero Phase 1 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌ Absolute Zero Phase 1 테이블 생성 실패: {e}")
        return False

def create_mtf_tables() -> bool:
    """MTF (Multi-Timeframe) 분석 테이블 생성"""
    try:
        logger.info("🔧 MTF 분석 테이블 생성 시작...")
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 1. mtf_signal_context 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mtf_signal_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_ts INTEGER NOT NULL,
                    base_interval TEXT NOT NULL,
                    base_strategy_id TEXT NOT NULL,
                    base_regime TEXT NOT NULL,
                    htf_interval TEXT NOT NULL,
                    htf_regime TEXT NOT NULL,
                    htf_trend_state TEXT NOT NULL,
                    htf_vol_bucket INTEGER NOT NULL,
                    align_sign INTEGER NOT NULL,
                    scale_ratio REAL NOT NULL,
                    coherence REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(base_ts, base_interval, base_strategy_id, htf_interval)
                )
            """)

            # mtf_signal_context 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_context_base_ts
                ON mtf_signal_context(base_ts)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_context_base_interval
                ON mtf_signal_context(base_interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_context_base_strategy
                ON mtf_signal_context(base_strategy_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_context_htf_interval
                ON mtf_signal_context(htf_interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_context_regimes
                ON mtf_signal_context(base_regime, htf_regime)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_context_composite
                ON mtf_signal_context(base_interval, htf_interval, base_regime, htf_regime)
            """)

            # 2. mtf_stats_by_pair 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mtf_stats_by_pair (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_interval TEXT NOT NULL,
                    htf_interval TEXT NOT NULL,
                    regime_combo TEXT NOT NULL,
                    align_rate_mean REAL DEFAULT 0.0,
                    scale_ratio_mean REAL DEFAULT 0.0,
                    coherence_mean REAL DEFAULT 0.0,
                    n_pairs INTEGER DEFAULT 0,
                    last_updated INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(base_interval, htf_interval, regime_combo)
                )
            """)

            # mtf_stats_by_pair 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_stats_base_interval
                ON mtf_stats_by_pair(base_interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_stats_htf_interval
                ON mtf_stats_by_pair(htf_interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_stats_regime_combo
                ON mtf_stats_by_pair(regime_combo)
            """)

            conn.commit()

        logger.info("✅ MTF 분석 테이블 생성 완료")
        return True

    except Exception as e:
        logger.error(f"❌ MTF 분석 테이블 생성 실패: {e}")
        return False


def create_strategy_training_history_table() -> bool:
    """전략 학습 이력 테이블 생성 (증분 학습용)"""
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # strategy_training_history 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_training_history (
                    strategy_id TEXT PRIMARY KEY,
                    trained_at DATETIME,
                    training_episodes INTEGER DEFAULT 0,
                    avg_accuracy REAL DEFAULT 0.0,
                    parent_strategy_id TEXT DEFAULT NULL,
                    similarity_score REAL DEFAULT 0.0,
                    training_source TEXT DEFAULT 'trained',
                    policy_data TEXT DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (strategy_id) REFERENCES coin_strategies(id)
                )
            """)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_history_trained_at
                ON strategy_training_history(trained_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_history_parent
                ON strategy_training_history(parent_strategy_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_history_source
                ON strategy_training_history(training_source)
            """)

            conn.commit()

        logger.info("✅ 전략 학습 이력 테이블 생성 완료")
        return True

    except Exception as e:
        logger.error(f"❌ 전략 학습 이력 테이블 생성 실패: {e}")
        return False
