"""
데이터베이스 스키마 - 엔진화를 위한 범용 스키마
DDL/마이그레이션, 인덱스 관리

핵심 설계:
- symbol: 범용 심볼 키 (coin 대신 사용)
- market_type: 시장 유형 (COIN, US_STOCK, KR_STOCK)
- market: 거래소/시장 (BITHUMB, NYSE, KOSPI 등)
- 테이블 구조 정리 (GPT.md 참조)
"""

import sqlite3
import logging
import os
from typing import Dict, List, Any, Optional
from rl_pipeline.db.connection_pool import get_strategy_db_pool, get_candle_db_pool
from rl_pipeline.core.errors import DBWriteError
from rl_pipeline.core.env import config

logger = logging.getLogger(__name__)

# ============================================================================
# 상수 정의
# ============================================================================

# 마켓 타입 상수
class MarketType:
    COIN = "COIN"
    US_STOCK = "US_STOCK"
    KR_STOCK = "KR_STOCK"

# 마켓 상수 (거래소/시장)
class Market:
    # 코인
    BITHUMB = "BITHUMB"
    BINANCE = "BINANCE"
    UPBIT = "UPBIT"
    # 미장
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    # 국장
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"

# 기본값 (코인장)
DEFAULT_MARKET_TYPE = MarketType.COIN
DEFAULT_MARKET = Market.BITHUMB


# ============================================================================
# 핵심 테이블 생성 함수
# ============================================================================

def create_strategies_table_impl(db_path: str = None) -> bool:
    """
    전략 테이블 생성 (구 strategies → strategies)
    strategy_definitions + strategy_performance_backtest 통합
    """
    try:
        # db_path가 명시되면 해당 경로의 DB에 테이블 생성
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,

                -- 범용 키 (엔진화 핵심)
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 전략 정의
                strategy_type TEXT DEFAULT 'hybrid',
                strategy_family TEXT DEFAULT NULL,
                strategy_conditions TEXT DEFAULT '{}',
                params TEXT DEFAULT '{}',
                description TEXT DEFAULT NULL,

                -- 지표 파라미터
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
                ma_period INTEGER DEFAULT 20,
                bb_period INTEGER DEFAULT 20,
                bb_std REAL DEFAULT 2.0,

                -- 백테스트 성과 (strategy_performance_backtest 통합)
                profit REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                trades_count INTEGER DEFAULT 0,
                max_drawdown REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                calmar_ratio REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                avg_profit_per_trade REAL DEFAULT 0.0,
                avg_mfe REAL DEFAULT 0.0,
                avg_mae REAL DEFAULT 0.0,

                -- 등급/점수
                quality_grade TEXT DEFAULT NULL,
                complexity_score REAL DEFAULT 0.0,
                score REAL DEFAULT 0.0,

                -- 시장 상태
                market_condition TEXT DEFAULT 'neutral',
                regime TEXT DEFAULT NULL,
                pattern_confidence REAL DEFAULT 0.5,
                pattern_source TEXT DEFAULT 'unknown',

                -- 메타데이터
                enhancement_type TEXT DEFAULT 'standard',
                
                -- 리그 시스템
                league TEXT DEFAULT 'minor', -- major, minor
                league_score REAL DEFAULT 0.0, -- 승강제 평가 점수
                
                is_active INTEGER DEFAULT 1,
                version INTEGER DEFAULT 1,
                parent_id TEXT DEFAULT NULL,
                parent_strategy_id TEXT DEFAULT NULL,
                similarity_classification TEXT DEFAULT NULL,
                similarity_score REAL DEFAULT NULL,

                -- 온라인 진화 시스템
                last_train_end_idx INTEGER DEFAULT NULL,
                online_pf REAL DEFAULT 0.0,
                online_return REAL DEFAULT 0.0,
                online_mdd REAL DEFAULT 0.0,
                online_updates_count INTEGER DEFAULT 0,
                consistency_score REAL DEFAULT 0.0,

                -- 하이브리드 정책
                hybrid_score REAL DEFAULT NULL,
                model_id TEXT DEFAULT NULL,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅ strategies 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌ strategies 테이블 생성 실패: {e}")
        raise DBWriteError(f"strategies 테이블 생성 실패: {e}") from e


def create_strategy_performance_rl_table(db_path: str = None) -> bool:
    """
    RL 성과 테이블 생성
    구 rl_strategy_rollup의 역할
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS strategy_performance_rl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- RL 성과 지표
                episodes_trained INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0,
                avg_ret REAL DEFAULT 0.0,
                rl_win_rate REAL DEFAULT 0.0,
                avg_profit_factor REAL DEFAULT 0.0,
                avg_sharpe_ratio REAL DEFAULT 0.0,
                best_episode_reward REAL DEFAULT 0.0,
                worst_episode_reward REAL DEFAULT 0.0,

                -- 예측 정확도
                predictive_accuracy REAL DEFAULT 0.0,
                avg_dd REAL DEFAULT 0.0,

                -- 등급
                grade TEXT DEFAULT 'UNKNOWN',
                grade_score REAL DEFAULT 0.0,

                -- 메타데이터
                training_run_id TEXT DEFAULT NULL,
                meta_json TEXT DEFAULT NULL,

                -- 타임스탬프
                last_trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at INTEGER DEFAULT 0,

                UNIQUE(strategy_id, market_type, market, symbol, interval)
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅ strategy_performance_rl 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌ strategy_performance_rl 테이블 생성 실패: {e}")
        raise DBWriteError(f"strategy_performance_rl 테이블 생성 실패: {e}") from e


def create_strategy_grades_table(db_path: str = None) -> bool:
    """
    전략 등급 테이블 생성
    시그널 계산/전략 선택의 공식 성적표
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS strategy_grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 등급 정보
                grade TEXT NOT NULL DEFAULT 'C',
                grade_score REAL NOT NULL DEFAULT 0.5,
                grade_basis TEXT NOT NULL DEFAULT 'BACKTEST',

                -- 성과 지표
                total_return REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                predictive_accuracy REAL DEFAULT 0.0,
                mdd REAL DEFAULT 0.0,

                -- 설명
                explain TEXT DEFAULT NULL,

                -- 타임스탬프
                graded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at INTEGER DEFAULT 0,

                UNIQUE(strategy_id, market_type, market, symbol, interval)
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅strategy_grades 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌strategy_grades 테이블 생성 실패: {e}")
        raise DBWriteError(f"strategy_grades 테이블 생성 실패: {e}") from e


def create_rl_episodes_table(db_path: str = None) -> bool:
    """
    RL 에피소드 테이블 생성 (예측 발표)
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS rl_episodes (
                episode_id TEXT PRIMARY KEY,
                ts_entry INTEGER NOT NULL,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 예측 정보
                strategy_id TEXT NOT NULL,
                state_key TEXT NOT NULL,
                predicted_dir INTEGER NOT NULL,
                predicted_conf REAL NOT NULL,
                entry_price REAL NOT NULL,
                target_move_pct REAL NOT NULL,
                horizon_k INTEGER NOT NULL
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_episodes_symbol_interval
                ON rl_episodes(symbol, interval, ts_entry)
            """)

            conn.commit()

            logger.info("✅rl_episodes 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌rl_episodes 테이블 생성 실패: {e}")
        raise DBWriteError(f"rl_episodes 테이블 생성 실패: {e}") from e


def create_rl_episode_summary_table(db_path: str = None) -> bool:
    """
    RL 에피소드 요약 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS rl_episode_summary (
                episode_id TEXT PRIMARY KEY,
                ts_exit INTEGER,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 결과 정보
                strategy_id TEXT,
                first_event TEXT,
                t_hit INTEGER,
                realized_ret_signed REAL,
                total_reward REAL,
                acc_flag INTEGER,
                source_type TEXT DEFAULT 'predictive'
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_episode_summary_symbol_interval
                ON rl_episode_summary(symbol, interval, ts_exit)
            """)

            conn.commit()

            logger.info("✅rl_episode_summary 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌rl_episode_summary 테이블 생성 실패: {e}")
        raise DBWriteError(f"rl_episode_summary 테이블 생성 실패: {e}") from e


def create_rl_steps_table(db_path: str = None) -> bool:
    """
    RL 스텝 테이블 생성 (스텝별 검증)
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS rl_steps (
                episode_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                event TEXT NOT NULL,
                price REAL NOT NULL,
                ret_raw REAL,
                ret_signed REAL,
                dd_pct_norm REAL,
                actual_move_pct REAL,
                prox REAL,
                dir_correct INTEGER,
                reward_dir REAL,
                reward_price REAL,
                reward_time REAL,
                reward_trade REAL,
                reward_calib REAL,
                reward_risk REAL,
                reward_total REAL,
                PRIMARY KEY (episode_id, ts)
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_steps_ts ON rl_steps(ts)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_steps_episode ON rl_steps(episode_id)")

            conn.commit()

            logger.info("✅rl_steps 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌rl_steps 테이블 생성 실패: {e}")
        raise DBWriteError(f"rl_steps 테이블 생성 실패: {e}") from e


def create_rl_state_ensemble_table(db_path: str = None) -> bool:
    """
    상태 앙상블 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS rl_state_ensemble (
                state_key TEXT NOT NULL,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 앙상블 정보
                acc_up INTEGER DEFAULT 0,
                acc_down INTEGER DEFAULT 0,
                acc_total REAL DEFAULT 0.0,
                p_up_smooth REAL DEFAULT 0.0,
                e_ret_smooth REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                last_updated INTEGER DEFAULT 0,

                -- 추가 정보
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

                PRIMARY KEY (market_type, market, symbol, interval, state_key)
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ensemble_symbol
                ON rl_state_ensemble(symbol, interval)
            """)

            conn.commit()

            logger.info("✅rl_state_ensemble 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌rl_state_ensemble 테이블 생성 실패: {e}")
        raise DBWriteError(f"rl_state_ensemble 테이블 생성 실패: {e}") from e


def create_global_strategies_table(db_path: str = None) -> bool:
    """
    글로벌 전략 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS global_strategies (
                id TEXT PRIMARY KEY,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 전략 정보
                strategy_type TEXT NOT NULL,
                params TEXT NOT NULL,
                name TEXT,
                description TEXT,
                dna_hash TEXT,
                source_type TEXT DEFAULT 'synthesized',

                -- 성과 지표
                profit REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.5,
                trades_count INTEGER DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,

                -- 등급
                quality_grade TEXT DEFAULT 'A',

                -- 시장 상태
                market_condition TEXT DEFAULT 'neutral',
                regime TEXT DEFAULT NULL,
                rsi_zone TEXT DEFAULT NULL,
                volatility_level TEXT DEFAULT NULL,

                -- 글로벌 분석
                global_dna_pattern TEXT,
                global_fractal_score REAL DEFAULT 0.0,
                global_synergy_score REAL DEFAULT 0.0,
                performance_score REAL DEFAULT 0.0,

                -- 메타데이터
                zone_key TEXT DEFAULT NULL,
                source_symbol TEXT DEFAULT NULL,
                source_strategy_id TEXT DEFAULT NULL,
                similarity_classification TEXT DEFAULT NULL,
                similarity_score REAL DEFAULT NULL,
                parent_strategy_id TEXT DEFAULT NULL,
                hybrid_score REAL DEFAULT NULL,
                model_id TEXT DEFAULT NULL,
                meta TEXT,

                -- 타임스탬프
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅global_strategies 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌global_strategies 테이블 생성 실패: {e}")
        raise DBWriteError(f"global_strategies 테이블 생성 실패: {e}") from e


def create_analysis_ratios_table(db_path: str = None) -> bool:
    """
    분석 비율 테이블 생성 (구 coin_analysis_ratios)
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS analysis_ratios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 분석 정보
                analysis_type TEXT NOT NULL,
                fractal_ratios TEXT DEFAULT '{}',
                multi_timeframe_ratios TEXT DEFAULT '{}',
                indicator_cross_ratios TEXT DEFAULT '{}',
                symbol_specific_ratios TEXT DEFAULT '{}',
                volatility_ratios TEXT DEFAULT '{}',
                volume_ratios TEXT DEFAULT '{}',
                optimal_modules TEXT DEFAULT '{}',
                interval_weights TEXT DEFAULT '{}',

                -- 점수
                performance_score REAL DEFAULT 0.0,
                accuracy_score REAL DEFAULT 0.0,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(market_type, market, symbol, interval, analysis_type)
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅analysis_ratios 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌analysis_ratios 테이블 생성 실패: {e}")
        raise DBWriteError(f"analysis_ratios 테이블 생성 실패: {e}") from e


def create_symbol_global_weights_table(db_path: str = None) -> bool:
    """
    심볼별 글로벌 가중치 테이블 생성 (구 coin_global_weights)
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS symbol_global_weights (
                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,

                -- 가중치
                symbol_weight REAL DEFAULT 0.7,
                global_weight REAL DEFAULT 0.3,
                symbol_score REAL DEFAULT 0.0,
                global_score REAL DEFAULT 0.0,
                data_quality_score REAL DEFAULT 0.0,

                -- 전략 수
                symbol_strategy_count INTEGER DEFAULT 0,
                global_strategy_count INTEGER DEFAULT 0,

                -- 성과
                symbol_avg_profit REAL DEFAULT 0.0,
                global_avg_profit REAL DEFAULT 0.0,
                symbol_win_rate REAL DEFAULT 0.0,
                global_win_rate REAL DEFAULT 0.0,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (market_type, market, symbol)
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅symbol_global_weights 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌symbol_global_weights 테이블 생성 실패: {e}")
        raise DBWriteError(f"symbol_global_weights 테이블 생성 실패: {e}") from e


def create_run_records_table(db_path: str = None) -> bool:
    """
    실행 기록 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS run_records (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                message TEXT,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT,
                interval TEXT,

                -- 실행 정보
                strategies_count INTEGER DEFAULT 0,
                successful_strategies INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,

                -- 타임스탬프
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_records_symbol_interval
                ON run_records(symbol, interval)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_records_status
                ON run_records(status)
            """)

            conn.commit()

            logger.info("✅run_records 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌run_records 테이블 생성 실패: {e}")
        raise DBWriteError(f"run_records 테이블 생성 실패: {e}") from e


def create_runs_table(db_path: str = None) -> bool:
    """
    실행 이력 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT,
                interval TEXT,

                -- 실행 정보
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                status TEXT DEFAULT 'running',
                strategies_count INTEGER DEFAULT 0,
                successful_strategies INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                completed_at DATETIME,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time)")

            conn.commit()

            logger.info("✅runs 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌runs 테이블 생성 실패: {e}")
        raise DBWriteError(f"runs 테이블 생성 실패: {e}") from e


def create_policy_models_table(db_path: str = None) -> bool:
    """
    정책 모델 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS policy_models (
                model_id TEXT PRIMARY KEY,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',

                -- 모델 정보
                algo TEXT NOT NULL DEFAULT 'PPO',
                features_ver TEXT NOT NULL,
                ckpt_path TEXT NOT NULL,
                notes TEXT,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_policy_models_algo ON policy_models(algo)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_policy_models_created ON policy_models(created_at)")

            conn.commit()

            logger.info("✅policy_models 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌policy_models 테이블 생성 실패: {e}")
        raise DBWriteError(f"policy_models 테이블 생성 실패: {e}") from e


def create_evaluation_results_table(db_path: str = None) -> bool:
    """
    평가 결과 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS evaluation_results (
                eval_id TEXT PRIMARY KEY,
                model_id TEXT,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',

                -- 평가 정보
                mode TEXT NOT NULL,
                asset TEXT NOT NULL,
                interval TEXT NOT NULL,
                period_from DATETIME NOT NULL,
                period_to DATETIME NOT NULL,

                -- 성과 지표
                profit_factor REAL,
                total_return REAL,
                win_rate REAL,
                mdd REAL,
                sharpe REAL,
                trades INTEGER,
                latency_ms_p95 REAL,
                notes TEXT,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (model_id) REFERENCES policy_models(model_id)
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_model ON evaluation_results(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_mode ON evaluation_results(mode)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_asset_interval ON evaluation_results(asset, interval)")

            conn.commit()

            logger.info("✅evaluation_results 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌evaluation_results 테이블 생성 실패: {e}")
        raise DBWriteError(f"evaluation_results 테이블 생성 실패: {e}") from e


def create_strategy_training_history_table(db_path: str = None) -> bool:
    """
    전략 학습 이력 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS strategy_training_history (
                strategy_id TEXT PRIMARY KEY,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',

                -- 학습 정보
                trained_at DATETIME,
                training_episodes INTEGER DEFAULT 0,
                avg_accuracy REAL DEFAULT 0.0,
                parent_strategy_id TEXT DEFAULT NULL,
                similarity_score REAL DEFAULT 0.0,
                training_source TEXT DEFAULT 'trained',
                policy_data TEXT DEFAULT NULL,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_history_trained_at
                ON strategy_training_history(trained_at)
            """)

            conn.commit()

            logger.info("✅strategy_training_history 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌strategy_training_history 테이블 생성 실패: {e}")
        raise DBWriteError(f"strategy_training_history 테이블 생성 실패: {e}") from e


def create_integrated_analysis_results_table(db_path: str = None) -> bool:
    """
    통합 분석 결과 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS integrated_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 분석 결과
                regime TEXT NOT NULL DEFAULT 'neutral',
                fractal_score REAL DEFAULT 0.0,
                multi_timeframe_score REAL DEFAULT 0.0,
                indicator_cross_score REAL DEFAULT 0.0,

                -- JAX 앙상블 결과
                ensemble_score REAL DEFAULT 0.0,
                ensemble_confidence REAL DEFAULT 0.0,

                -- 최종 시그널
                final_signal_score REAL DEFAULT 0.0,
                signal_confidence REAL DEFAULT 0.0,
                signal_action TEXT DEFAULT 'hold',

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_integrated_analysis_symbol_interval
                ON integrated_analysis_results(symbol, interval)
            """)

            conn.commit()

            logger.info("✅integrated_analysis_results 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌integrated_analysis_results 테이블 생성 실패: {e}")
        raise DBWriteError(f"integrated_analysis_results 테이블 생성 실패: {e}") from e


def migrate_integrated_analysis_results_table(db_path: str = None) -> bool:
    """
    integrated_analysis_results 테이블 마이그레이션 (coin -> symbol)
    """
    try:
        pool = get_strategy_db_pool(db_path)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 기존 테이블 컬럼 확인
            cursor.execute("PRAGMA table_info(integrated_analysis_results)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # 2. coin 컬럼이 있고 symbol이 없는 경우 (매우 구버전) -> symbol 추가 및 데이터 복사
            if 'coin' in columns and 'symbol' not in columns:
                logger.info("🔧 integrated_analysis_results: coin -> symbol 마이그레이션 시작")
                cursor.execute("ALTER TABLE integrated_analysis_results ADD COLUMN symbol TEXT")
                cursor.execute("UPDATE integrated_analysis_results SET symbol = coin")
                logger.info("✅ symbol 컬럼 추가 및 데이터 복사 완료")
            
            # 3. symbol만 있고 coin이 없는 경우 (신버전) -> coin 뷰 생성 필요 없음 (이미 다른 뷰에서 처리)
            # 하지만 호환성을 위해 coin 컬럼을 가상으로라도 제공해야 한다면 뷰를 생성해야 함
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ integrated_analysis_results 마이그레이션 실패 (무시 가능): {e}")
        return False



def create_pipeline_execution_logs_table(db_path: str = None) -> bool:
    """
    파이프라인 실행 로그 테이블 생성
    """
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS pipeline_execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- 범용 키
                market_type TEXT NOT NULL DEFAULT 'COIN',
                market TEXT NOT NULL DEFAULT 'BITHUMB',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,

                -- 실행 정보
                strategies_created INTEGER DEFAULT 0,
                selfplay_episodes INTEGER DEFAULT 0,
                regime_detected TEXT,
                routing_results TEXT,
                signal_score REAL,
                signal_action TEXT,
                execution_time REAL,
                status TEXT,

                -- 타임스탬프
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """

            cursor.execute(create_table_query)
            conn.commit()

            logger.info("✅pipeline_execution_logs 테이블 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌pipeline_execution_logs 테이블 생성 실패: {e}")
        raise DBWriteError(f"pipeline_execution_logs 테이블 생성 실패: {e}") from e


# ============================================================================
# 인덱스 생성
# ============================================================================

def ensure_indexes_impl(db_path: str = None) -> bool:
    """테이블 인덱스 생성"""
    try:
        strategy_indexes = [
            # strategies 테이블
            "CREATE INDEX IF NOT EXISTS idx_strategies_symbol_interval ON strategies(symbol, interval)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_market_type ON strategies(market_type)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_profit ON strategies(profit DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_win_rate ON strategies(win_rate DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON strategies(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_quality_grade ON strategies(quality_grade)",

            # strategy_performance_rl 테이블
            "CREATE INDEX IF NOT EXISTS idx_perf_rl_symbol_interval ON strategy_performance_rl(symbol, interval)",
            "CREATE INDEX IF NOT EXISTS idx_perf_rl_grade ON strategy_performance_rl(grade)",

            # strategy_grades 테이블
            "CREATE INDEX IF NOT EXISTS idx_grades_symbol_interval ON strategy_grades(symbol, interval)",
            "CREATE INDEX IF NOT EXISTS idx_grades_grade ON strategy_grades(grade)",

            # global_strategies 테이블
            "CREATE INDEX IF NOT EXISTS idx_global_symbol_interval ON global_strategies(symbol, interval)",

            # analysis_ratios 테이블
            "CREATE INDEX IF NOT EXISTS idx_analysis_symbol_interval ON analysis_ratios(symbol, interval)",
        ]

        try:
            strategy_pool = get_strategy_db_pool(db_path)
            with strategy_pool.get_connection() as conn:
                cursor = conn.cursor()
                for index_query in strategy_indexes:
                    try:
                        table_name = index_query.split(" ON ")[1].split("(")[0].strip()
                        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                        if cursor.fetchone():
                            cursor.execute(index_query)
                        else:
                            logger.debug(f"⚠️ {table_name} 테이블이 존재하지 않아 인덱스 생성을 건너뜁니다")
                    except Exception as e:
                        logger.warning(f"⚠️ 인덱스 생성 건너뜀: {e}")
                conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ 인덱스 생성 실패 (계속 진행): {e}")

        logger.info("✅인덱스 확인 및 생성 완료")
        return True

    except Exception as e:
        logger.error(f"❌인덱스 생성 실패: {e}")
        return False


# ============================================================================
# 통합 초기화 함수
# ============================================================================

def migrate_strategies_league_columns(db_path: str = None) -> None:
    """strategies 테이블에 리그 시스템 컬럼 추가"""
    try:
        pool = get_strategy_db_pool(db_path)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 현재 컬럼 목록 조회
            cursor.execute("PRAGMA table_info(strategies)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # 추가할 컬럼 정의
            new_columns = {
                'league': "TEXT DEFAULT 'minor'",
                'league_score': "REAL DEFAULT 0.0"
            }
            
            for col, definition in new_columns.items():
                if col not in existing_columns:
                    try:
                        alter_query = f"ALTER TABLE strategies ADD COLUMN {col} {definition}"
                        cursor.execute(alter_query)
                        logger.info(f"✅ strategies 테이블 컬럼 추가: {col} (리그 시스템)")
                    except Exception as alter_err:
                        logger.warning(f"⚠️ 컬럼 추가 실패 ({col}): {alter_err}")
            
            # 인덱스 추가 (리그별 조회를 빠르게)
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategies_league ON strategies(league)")
            except Exception as e:
                logger.warning(f"⚠️ 리그 인덱스 생성 실패: {e}")
                
            conn.commit()
            
    except Exception as e:
        logger.error(f"❌ 리그 시스템 마이그레이션 실패: {e}")


def setup_database_tables_impl(db_path: str = None) -> bool:
    """모든 데이터베이스 테이블 초기화"""
    try:
        logger.info(f"🔧 데이터베이스 테이블 초기화 시작... (경로: {db_path or '기본'})")

        # 핵심 테이블 (GPT.md 기반)
        create_strategies_table_impl(db_path)                    # 구 strategies
        create_strategy_performance_rl_table(db_path)       # 구 rl_strategy_rollup
        create_strategy_grades_table(db_path)               # 전략 등급

        # RL 테이블
        create_rl_episodes_table(db_path)
        create_rl_episode_summary_table(db_path)
        create_rl_steps_table(db_path)
        create_rl_state_ensemble_table(db_path)

        # 글로벌 전략
        create_global_strategies_table(db_path)

        # 분석 테이블
        create_analysis_ratios_table(db_path)               # 구 coin_analysis_ratios
        create_symbol_global_weights_table(db_path)         # 구 coin_global_weights
        create_integrated_analysis_results_table(db_path)

        # 실행 기록
        create_run_records_table(db_path)
        create_runs_table(db_path)
        create_pipeline_execution_logs_table(db_path)

        # 모델/학습
        create_policy_models_table(db_path)
        create_evaluation_results_table(db_path)
        create_strategy_training_history_table(db_path)

        # v1 호환성 마이그레이션 실행
        migrate_strategies_table(db_path)
        
        # 🔥 v2 생애주기 마이그레이션 실행
        migrate_strategies_lifecycle_columns(db_path)
        
        # 🔥 v3 리그 시스템 마이그레이션 실행 (New)
        migrate_strategies_league_columns(db_path)

        # 🔥 호환성 뷰 생성 (반드시 필요)
        create_compatibility_views(db_path)

        # 인덱스 생성
        ensure_indexes_impl(db_path)

        logger.info("✅모든 데이터베이스 테이블 초기화 완료")
        return True

    except Exception as e:
        logger.error(f"❌데이터베이스 테이블 초기화 실패: {e}")
        raise DBWriteError(f"데이터베이스 테이블 초기화 실패: {e}") from e


# ============================================================================
# 호환성 별칭 (기존 코드 호환)
# ============================================================================

# 기존 코드에서 strategies를 사용하는 경우를 위한 뷰 생성
def create_compatibility_views(db_path: str = None) -> bool:
    """v1 호환성을 위한 뷰 생성 (strategies → strategies)"""
    try:
        pool = get_strategy_db_pool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # strategies 뷰 (symbol을 coin으로 별칭)
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS strategies AS
                SELECT
                    id,
                    symbol AS coin,
                    interval,
                    strategy_type,
                    strategy_conditions,
                    params,
                    rsi_min, rsi_max,
                    volume_ratio_min, volume_ratio_max,
                    macd_buy_threshold, macd_sell_threshold,
                    mfi_min, mfi_max,
                    atr_min, atr_max,
                    adx_min,
                    stop_loss_pct, take_profit_pct,
                    ma_period, bb_period, bb_std,
                    profit, win_rate, trades_count,
                    max_drawdown, sharpe_ratio, calmar_ratio,
                    profit_factor, avg_profit_per_trade,
                    quality_grade, complexity_score, score,
                    market_condition, regime,
                    pattern_confidence, pattern_source,
                    enhancement_type, is_active,
                    version, parent_id, parent_strategy_id,
                    similarity_classification, similarity_score,
                    last_train_end_idx, online_pf, online_return,
                    online_mdd, online_updates_count, consistency_score,
                    hybrid_score, model_id,
                    created_at, updated_at,
                    market_type, market
                FROM strategies
                WHERE market_type = 'COIN'
            """)

            # rl_strategy_rollup 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS rl_strategy_rollup AS
                SELECT
                    strategy_id,
                    symbol AS coin,
                    interval,
                    episodes_trained,
                    avg_ret,
                    rl_win_rate AS win_rate,
                    predictive_accuracy,
                    avg_dd,
                    episodes_trained AS total_episodes,
                    avg_ret AS total_profit,
                    avg_reward,
                    avg_profit_factor,
                    avg_sharpe_ratio,
                    best_episode_reward,
                    worst_episode_reward,
                    grade,
                    updated_at,
                    last_trained_at AS last_updated,
                    created_at,
                    market_type, market
                FROM strategy_performance_rl
                WHERE market_type = 'COIN'
            """)

            # coin_analysis_ratios 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS coin_analysis_ratios AS
                SELECT
                    id,
                    symbol AS coin,
                    interval,
                    analysis_type,
                    fractal_ratios,
                    multi_timeframe_ratios,
                    indicator_cross_ratios,
                    symbol_specific_ratios AS coin_specific_ratios,
                    volatility_ratios,
                    volume_ratios,
                    optimal_modules,
                    interval_weights,
                    performance_score,
                    accuracy_score,
                    created_at,
                    updated_at,
                    market_type, market
                FROM analysis_ratios
                WHERE market_type = 'COIN'
            """)

            # coin_global_weights 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS coin_global_weights AS
                SELECT
                    symbol AS coin,
                    symbol_weight AS coin_weight,
                    global_weight,
                    symbol_score AS coin_score,
                    global_score,
                    data_quality_score,
                    symbol_strategy_count AS coin_strategy_count,
                    global_strategy_count,
                    symbol_avg_profit AS coin_avg_profit,
                    global_avg_profit,
                    symbol_win_rate AS coin_win_rate,
                    global_win_rate,
                    created_at,
                    updated_at,
                    market_type, market
                FROM symbol_global_weights
                WHERE market_type = 'COIN'
            """)

            # integrated_analysis_results 뷰 (호환성) - coin 컬럼 별칭 제공
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS coin_integrated_analysis_results AS
                SELECT
                    id,
                    symbol AS coin,
                    market_type, market, symbol, interval,
                    regime, fractal_score, multi_timeframe_score, indicator_cross_score,
                    ensemble_score, ensemble_confidence,
                    final_signal_score, signal_confidence, signal_action,
                    created_at
                FROM integrated_analysis_results
                WHERE market_type = 'COIN'
            """)

            conn.commit()

            logger.info("✅ 호환성 뷰 생성 완료")
            return True

    except Exception as e:
        logger.error(f"❌ 호환성 뷰 생성 실패: {e}")
        return False


# ============================================================================
# 외부 API 함수 (기존 코드에서 import 호환)
# ============================================================================

# 기존 코드 호환성을 위한 함수 별칭
def setup_database_tables(db_path: str = None) -> bool:
    """데이터베이스 테이블 초기화"""
    result = setup_database_tables_impl(db_path)
    # 호환성 뷰 생성
    create_compatibility_views(db_path)
    return result


def ensure_indexes(db_path: str = None) -> bool:
    """인덱스 생성"""
    return ensure_indexes_impl(db_path)


def create_compatibility_views(db_path: str = None) -> bool:
    """호환성 뷰 생성 (모든 뷰 한 번에)"""
    try:
        pool = get_strategy_db_pool(db_path)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # strategies 뷰 (symbol을 coin으로 별칭)
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS strategies AS
                SELECT 
                    id,
                    symbol AS coin,
                    interval,
                    strategy_type,
                    strategy_conditions,
                    params,
                    rsi_min, rsi_max,
                    volume_ratio_min, volume_ratio_max,
                    macd_buy_threshold, macd_sell_threshold,
                    mfi_min, mfi_max,
                    atr_min, atr_max,
                    adx_min,
                    stop_loss_pct, take_profit_pct,
                    ma_period, bb_period, bb_std,
                    profit, win_rate, trades_count,
                    max_drawdown, sharpe_ratio, calmar_ratio,
                    profit_factor, avg_profit_per_trade,
                    avg_mfe, avg_mae,
                    quality_grade, complexity_score, score,
                    market_condition, regime,
                    pattern_confidence, pattern_source,
                    enhancement_type, is_active,
                    version, parent_id, parent_strategy_id,
                    similarity_classification, similarity_score,
                    last_train_end_idx, online_pf, online_return,
                    online_mdd, online_updates_count, consistency_score,
                    hybrid_score, model_id,
                    created_at, updated_at,
                    market_type, market
                FROM strategies
                WHERE market_type = 'COIN'
            """)
            
            # rl_strategy_rollup 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS rl_strategy_rollup AS
                SELECT 
                    strategy_id,
                    symbol AS coin,
                    interval,
                    episodes_trained,
                    avg_ret,
                    rl_win_rate AS win_rate,
                    predictive_accuracy,
                    avg_dd,
                    episodes_trained AS total_episodes,
                    avg_ret AS total_profit,
                    avg_reward,
                    avg_profit_factor,
                    avg_sharpe_ratio,
                    best_episode_reward,
                    worst_episode_reward,
                    grade,
                    updated_at,
                    last_trained_at AS last_updated,
                    created_at,
                    market_type, market
                FROM strategy_performance_rl
                WHERE market_type = 'COIN'
            """)
            
            # coin_analysis_ratios 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS coin_analysis_ratios AS
                SELECT 
                    id,
                    symbol AS coin,
                    interval,
                    analysis_type,
                    fractal_ratios,
                    multi_timeframe_ratios,
                    indicator_cross_ratios,
                    symbol_specific_ratios AS coin_specific_ratios,
                    volatility_ratios,
                    volume_ratios,
                    optimal_modules,
                    interval_weights,
                    performance_score,
                    accuracy_score,
                    created_at,
                    updated_at,
                    market_type, market
                FROM analysis_ratios
                WHERE market_type = 'COIN'
            """)
            
            # coin_global_weights 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS coin_global_weights AS
                SELECT 
                    symbol AS coin,
                    symbol_weight AS coin_weight,
                    global_weight,
                    symbol_score AS coin_score,
                    created_at,
                    market_type, market
                FROM symbol_global_weights
                WHERE market_type = 'COIN'
            """)
            
            # coin_integrated_analysis_results 뷰
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS coin_integrated_analysis_results AS
                SELECT 
                    id,
                    symbol AS coin,
                    interval,
                    regime,
                    signal_score,
                    signal_action,
                    fractal_score,
                    multi_tf_score,
                    indicator_cross_score,
                    final_signal_score,
                    direction_strength,
                    timing_confidence,
                    regime_consistency,
                    meta_score,
                    volatility_adjustment,
                    created_at,
                    market_type, market
                FROM integrated_analysis_results
                WHERE market_type = 'COIN'
            """)
            
            conn.commit()
            logger.info("✅ 호환성 뷰 생성 완료 (strategies 등)")
            return True
            
    except Exception as e:
        logger.error(f"❌ 호환성 뷰 생성 실패: {e}")
        return False

def create_strategies_table(db_path: str = None) -> bool:
    """v1 호환성 - strategies 테이블 생성 후 뷰 생성"""
    result = create_strategies_table_impl(db_path)
    create_compatibility_views(db_path)
    return result


# 기존 코드에서 사용하는 기타 함수들의 호환성 별칭
def create_candles_table(db_path: str = None) -> bool:
    """캔들 테이블은 원천 데이터 - 생성하지 않음"""
    logger.info("⚠️ 캔들 테이블은 원천 데이터로 생성하지 않습니다 (rl_candles.db는 읽기 전용)")
    return True


def create_selfplay_results_table(db_path: str = None) -> bool:
    """rl_episodes/rl_episode_summary로 대체"""
    logger.debug("⚠️ selfplay_results는 rl_episodes로 대체됨")
    return True


def create_strategy_dna_table(db_path: str = None) -> bool:
    """strategies 테이블의 params에 통합"""
    logger.debug("⚠️ strategy_dna는 strategies.params에 통합됨")
    return True


def create_fractal_analysis_table(db_path: str = None) -> bool:
    """analysis_ratios로 대체"""
    logger.debug("⚠️ fractal_analysis는 analysis_ratios로 대체됨")
    return True


def create_synergy_analysis_table(db_path: str = None) -> bool:
    """analysis_ratios로 대체"""
    logger.debug("⚠️ synergy_analysis는 analysis_ratios로 대체됨")
    return True


def create_runs_table_compat(db_path: str = None) -> bool:
    """호환성 - create_runs_table 호출"""
    return create_runs_table(db_path)


def migrate() -> bool:
    """마이그레이션 불필요 (새 스키마)"""
    logger.info("✅ 새 스키마 사용 중 - 마이그레이션 불필요")
    return True


def check_database_integrity(db_path: str = None) -> bool:
    """데이터베이스 무결성 검사"""
    try:
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


def repair_database(db_path: str = None) -> bool:
    """손상된 데이터베이스 복구 시도"""
    try:
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


# v1 호환성 마이그레이션 함수
def migrate_strategies_table(db_path: str = None) -> bool:
    """strategies 테이블에 누락된 컬럼 추가 (마이그레이션)"""
    try:
        pool = get_strategy_db_pool(db_path)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 현재 컬럼 목록 조회
            cursor.execute("PRAGMA table_info(strategies)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # 추가할 컬럼 정의
            columns_to_add = {
                'league': "TEXT DEFAULT 'minor'",
                'league_score': "REAL DEFAULT 0.0",
                'consistency_score': "REAL DEFAULT 0.0",
                'lifecycle_status': "TEXT DEFAULT 'QUARANTINE'",
                'failure_assumption': "TEXT DEFAULT NULL",
                'hybrid_score': "REAL DEFAULT NULL",
                'model_id': "TEXT DEFAULT NULL",
                'market_type': "TEXT NOT NULL DEFAULT 'COIN'",
                'market': "TEXT NOT NULL DEFAULT 'BITHUMB'",
                'avg_mfe': "REAL DEFAULT 0.0",
                'avg_mae': "REAL DEFAULT 0.0"
            }
            
            added_count = 0
            for col, definition in columns_to_add.items():
                if col not in existing_columns:
                    try:
                        alter_query = f"ALTER TABLE strategies ADD COLUMN {col} {definition}"
                        cursor.execute(alter_query)
                        logger.info(f"✅ strategies 테이블에 '{col}' 컬럼 추가 완료")
                        added_count += 1
                    except Exception as alter_err:
                        logger.warning(f"⚠️ 컬럼 추가 실패 ({col}): {alter_err}")
            
            if added_count > 0:
                conn.commit()
                # 뷰 재생성
                create_compatibility_views(db_path)
                logger.info(f"✅ 총 {added_count}개 컬럼 마이그레이션 완료")
            
            return True
    except Exception as e:
        logger.error(f"❌ strategies 마이그레이션 실패: {e}")
        return False

def migrate_strategies_lifecycle_columns(db_path: str = None) -> bool:
    """v2 마이그레이션 - 전략 생애주기(Lifecycle) 및 메타정보 컬럼 추가"""
    try:
        pool = get_strategy_db_pool(db_path)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 컬럼 확인
            cursor.execute("PRAGMA table_info(strategies)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # 1. lifecycle_status 추가 (기본값: QUARANTINE)
            # 기존 전략은 호환성을 위해 'ACTIVE'나 'CANDIDATE'로 둘 수도 있지만,
            # 엄격한 관리를 위해 일괄적으로 NULL 또는 별도 처리가 필요함.
            # 여기서는 스키마 호환을 위해 'ACTIVE'로 초기화하되, 신규 전략은 'QUARANTINE'으로 들어감.
            if 'lifecycle_status' not in columns:
                logger.info("🔧 strategies: lifecycle_status 컬럼 추가")
                # 기존 데이터는 ACTIVE로 설정 (갑작스러운 차단 방지), 신규 기본값은 QUARANTINE
                cursor.execute("ALTER TABLE strategies ADD COLUMN lifecycle_status TEXT DEFAULT 'QUARANTINE'")
                
                # 기존 데이터 마이그레이션 (선택적: 기존 전략을 모두 ACTIVE로 승격)
                cursor.execute("UPDATE strategies SET lifecycle_status = 'ACTIVE' WHERE lifecycle_status IS 'QUARANTINE'")
                logger.info("   └─ 기존 전략들을 'ACTIVE' 상태로 초기화했습니다.")

            # 2. failure_assumption 추가
            if 'failure_assumption' not in columns:
                logger.info("🔧 strategies: failure_assumption 컬럼 추가")
                cursor.execute("ALTER TABLE strategies ADD COLUMN failure_assumption TEXT DEFAULT NULL")

            conn.commit()
            
            # 뷰 재생성
            create_compatibility_views(db_path)
            
            return True
            
    except sqlite3.OperationalError as oe:
        if "database is locked" in str(oe):
            logger.warning(f"⚠️ DB 잠금으로 인해 마이그레이션 보류 (다음 실행 시 재시도): {oe}")
            return False
        logger.error(f"❌ strategies 생애주기 마이그레이션 실패 (Operational): {oe}")
        return False
    except Exception as e:
        logger.error(f"❌ strategies 생애주기 마이그레이션 실패: {e}")
        return False

def migrate_global_strategies_table() -> bool:
    """v1 호환성 - 마이그레이션 불필요"""
    return True

def migrate_coin_analysis_ratios_table() -> bool:
    """v1 호환성 - 마이그레이션 불필요"""
    return True

def migrate_coin_global_weights_table() -> bool:
    """v1 호환성 - 마이그레이션 불필요"""
    return True

def migrate_online_evolution_system() -> bool:
    """v1 호환성 - 마이그레이션 불필요"""
    return True
