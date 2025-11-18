"""
통합 데이터베이스 초기화 스크립트
- trading_system.db: 모든 거래 시스템 테이블
- realtime_candles.db: 캔들 데이터
"""
import sqlite3
import os
from pathlib import Path

# 데이터베이스 경로
BASE_DIR = Path(__file__).parent.parent / 'data_storage'
TRADING_SYSTEM_DB = BASE_DIR / 'trading_system.db'
REALTIME_CANDLES_DB = BASE_DIR / 'realtime_candles.db'

def initialize_trading_system_db():
    """trading_system.db 초기화"""
    print("📊 trading_system.db 초기화 중...")

    # 디렉토리 생성
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(TRADING_SYSTEM_DB) as conn:
        cursor = conn.cursor()

        # 1. signals 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                action TEXT NOT NULL,
                signal_score REAL,
                confidence REAL,
                reason TEXT,
                timestamp INTEGER NOT NULL,
                price REAL,
                volume REAL,
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
                mfi REAL,
                atr REAL,
                adx REAL,
                ma20 REAL,
                rsi_ema REAL,
                macd_smoothed REAL,
                wave_momentum REAL,
                bb_position TEXT,
                bb_width REAL,
                bb_squeeze REAL,
                rsi_divergence TEXT,
                macd_divergence TEXT,
                volume_divergence TEXT,
                price_momentum REAL,
                volume_momentum REAL,
                trend_strength REAL,
                support_resistance TEXT,
                fibonacci_levels TEXT,
                elliott_wave TEXT,
                harmonic_patterns TEXT,
                candlestick_patterns TEXT,
                market_structure TEXT,
                flow_level_meta TEXT,
                pattern_direction TEXT
            )
        """)

        # 2. virtual_positions 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS virtual_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_timestamp INTEGER NOT NULL,
                entry_signal_score REAL,
                stop_loss_price REAL,
                take_profit_price REAL,
                max_profit_pct REAL DEFAULT 0,
                max_loss_pct REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin)
            )
        """)

        # 3. completed_trades 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS completed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_timestamp INTEGER NOT NULL,
                exit_timestamp INTEGER NOT NULL,
                profit_loss_pct REAL NOT NULL,
                holding_duration INTEGER NOT NULL,
                entry_signal_score REAL,
                exit_signal_score REAL,
                exit_reason TEXT,
                max_profit_pct REAL,
                max_loss_pct REAL,
                learning_episode INTEGER,
                total_episodes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 4. virtual_trade_feedback 테이블
        cursor.execute("""
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

        # 5. learning_checkpoint 테이블
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

        # 6. evolution_results 테이블
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

        # 7. multi_timeframe_analysis 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_timeframe_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                combined_signal_score REAL,
                combined_confidence REAL,
                interval_15m_score REAL,
                interval_30m_score REAL,
                interval_240m_score REAL,
                interval_1d_score REAL,
                trade_outcome TEXT,
                profit_loss_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 8. signal_feedback_scores 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                signal_score REAL,
                outcome TEXT,
                profit_loss_pct REAL,
                timestamp INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 9. virtual_performance_stats 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS virtual_performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_profit REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                updated_at INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 10. virtual_trading_q_table 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS virtual_trading_q_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_key TEXT NOT NULL UNIQUE,
                action TEXT NOT NULL,
                q_value REAL DEFAULT 0,
                visit_count INTEGER DEFAULT 0,
                updated_at INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 11. trade_feedback 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT,
                entry_timestamp INTEGER NOT NULL,
                exit_timestamp INTEGER NOT NULL,
                profit_loss_pct REAL NOT NULL,
                signal_quality REAL,
                pattern_accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        print("✅ trading_system.db 초기화 완료!")
        print(f"   - 위치: {TRADING_SYSTEM_DB}")


def initialize_realtime_candles_db():
    """realtime_candles.db 초기화"""
    print("\n📊 realtime_candles.db 초기화 중...")

    with sqlite3.connect(REALTIME_CANDLES_DB) as conn:
        cursor = conn.cursor()

        # candles 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(coin, interval, timestamp)
            )
        """)

        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_coin_interval_timestamp
            ON candles(coin, interval, timestamp)
        """)

        # dqn_training_results 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dqn_training_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                episode INTEGER NOT NULL,
                total_reward REAL,
                total_return REAL,
                win_rate REAL,
                avg_profit REAL,
                total_trades INTEGER,
                epsilon REAL,
                train_steps INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        print("✅ realtime_candles.db 초기화 완료!")
        print(f"   - 위치: {REALTIME_CANDLES_DB}")


def verify_databases():
    """데이터베이스 검증"""
    print("\n🔍 데이터베이스 검증 중...")

    # trading_system.db 검증
    with sqlite3.connect(TRADING_SYSTEM_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n✅ trading_system.db 테이블 ({len(tables)}개):")
        for table in sorted(tables):
            print(f"   - {table}")

    # realtime_candles.db 검증
    with sqlite3.connect(REALTIME_CANDLES_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n✅ realtime_candles.db 테이블 ({len(tables)}개):")
        for table in sorted(tables):
            print(f"   - {table}")


def main():
    """메인 실행"""
    print("=" * 60)
    print("🚀 트레이딩 시스템 데이터베이스 초기화")
    print("=" * 60)

    try:
        initialize_trading_system_db()
        initialize_realtime_candles_db()
        verify_databases()

        print("\n" + "=" * 60)
        print("✅ 모든 데이터베이스 초기화 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
