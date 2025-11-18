#!/usr/bin/env python
"""Paper Trading 상태 확인"""
import sys
sys.path.append('/workspace')

import sqlite3
import os

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def check_paper_trading():
    print("=" * 70)
    print("📊 Paper Trading 상태 확인")
    print("=" * 70)
    print()

    # 1. DB 파일 존재 확인
    if not os.path.exists(DB_PATH):
        print(f"❌ DB 파일 없음: {DB_PATH}")
        return

    print(f"✅ DB 파일 존재: {DB_PATH}")
    print()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2. 테이블 존재 확인
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE '%paper%'
    """)
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print("❌ Paper Trading 테이블 없음")
        print()
        print("💡 테이블을 생성해야 합니다:")
        print("   python -m rl_pipeline.validation.auto_paper_trading")
        conn.close()
        return

    print(f"✅ Paper Trading 테이블: {len(tables)}개")
    for table in tables:
        print(f"   - {table}")
    print()

    # 3. 세션 개수 확인
    try:
        cursor.execute("SELECT COUNT(*) FROM paper_trading_sessions")
        total_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM paper_trading_sessions WHERE status='running'")
        active_sessions = cursor.fetchone()[0]

        print(f"📊 세션 통계:")
        print(f"   총 세션: {total_sessions}개")
        print(f"   활성 세션: {active_sessions}개")
        print()

        # 4. 최근 세션 조회
        if total_sessions > 0:
            cursor.execute("""
                SELECT session_id, coin, interval, status, start_time,
                       initial_capital, current_capital
                FROM paper_trading_sessions
                ORDER BY start_time DESC
                LIMIT 5
            """)

            print(f"최근 세션 5개:")
            print(f"{'Session ID':<40} {'코인':<8} {'인터벌':<8} {'상태':<10} {'시작':<20}")
            print("-" * 70)

            for row in cursor.fetchall():
                session_id = row[0]
                coin = row[1]
                interval = row[2]
                status = row[3]
                start_time = row[4][:19] if row[4] else 'N/A'

                print(f"{session_id:<40} {coin:<8} {interval:<8} {status:<10} {start_time:<20}")
            print()

            # 5. 성과 데이터 확인
            cursor.execute("SELECT COUNT(*) FROM paper_trading_performance")
            perf_count = cursor.fetchone()[0]

            print(f"성과 데이터: {perf_count}개")

            if perf_count > 0:
                cursor.execute("""
                    SELECT p.session_id, s.coin, s.interval,
                           p.total_return, p.total_trades, p.win_rate
                    FROM paper_trading_performance p
                    JOIN paper_trading_sessions s ON p.session_id = s.session_id
                    ORDER BY p.last_updated DESC
                    LIMIT 5
                """)

                print()
                print(f"최근 성과 5개:")
                print(f"{'코인':<8} {'인터벌':<8} {'수익률':>10} {'거래':>8} {'승률':>8}")
                print("-" * 70)

                for row in cursor.fetchall():
                    coin = row[1]
                    interval = row[2]
                    total_return = row[3] or 0
                    total_trades = row[4] or 0
                    win_rate = row[5] or 0

                    print(f"{coin:<8} {interval:<8} {total_return:>9.2f}% {total_trades:>8} {win_rate:>7.1%}")
            print()

        else:
            print("⚠️  세션이 하나도 없습니다!")
            print()
            print("💡 Paper Trading 세션을 생성하려면:")
            print("   1. Absolute Zero 파이프라인 실행:")
            print("      python rl_pipeline/absolute_zero_system.py")
            print()
            print("   2. 또는 수동 세션 생성:")
            print("      python -m rl_pipeline.validation.auto_paper_trading")
            print()

    except sqlite3.OperationalError as e:
        print(f"❌ 테이블 쿼리 실패: {e}")
        print()
        print("💡 테이블을 생성해야 합니다:")
        print("   python -m rl_pipeline.validation.auto_paper_trading")

    conn.close()

    # 6. 환경 변수 확인
    print("=" * 70)
    print("🔧 환경 변수")
    print("=" * 70)

    enable_paper = os.getenv('ENABLE_AUTO_PAPER_TRADING', 'true')
    duration_days = os.getenv('PAPER_TRADING_DURATION_DAYS', '30')

    print(f"ENABLE_AUTO_PAPER_TRADING: {enable_paper}")
    print(f"PAPER_TRADING_DURATION_DAYS: {duration_days}")
    print()

    if enable_paper.lower() != 'true':
        print("⚠️  Paper Trading 자동 실행이 비활성화되어 있습니다!")
        print("   활성화하려면 .env 파일에서:")
        print("   ENABLE_AUTO_PAPER_TRADING=true")
    else:
        print("✅ Paper Trading 자동 실행 활성화됨")

    print()

if __name__ == '__main__':
    check_paper_trading()
