#!/usr/bin/env python
"""글로벌 전략 생성 상태 확인"""
import sys
sys.path.append('/workspace')

import sqlite3
import os

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def check_global_strategies():
    print("=" * 70)
    print("📊 글로벌 전략 생성 상태 확인")
    print("=" * 70)
    print()

    if not os.path.exists(DB_PATH):
        print(f"❌ DB 파일 없음: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. global_strategies 테이블 확인
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='global_strategies'
    """)

    if not cursor.fetchone():
        print("❌ global_strategies 테이블 없음")
        conn.close()
        return

    print("✅ global_strategies 테이블 존재")
    print()

    # 2. 글로벌 전략 개수
    cursor.execute("SELECT COUNT(*) FROM global_strategies")
    global_count = cursor.fetchone()[0]

    print(f"글로벌 전략 개수: {global_count}개")
    print()

    # 3. coin_strategies 개수 (원천 데이터)
    try:
        cursor.execute("SELECT COUNT(*) FROM coin_strategies")
        coin_count = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT coin FROM coin_strategies")
        coins = [row[0] for row in cursor.fetchall()]

        print(f"코인별 전략 개수: {coin_count}개")
        print(f"코인 목록: {', '.join(coins) if coins else '(없음)'}")
        print()
    except sqlite3.OperationalError:
        print("⚠️  coin_strategies 테이블 없음")
        print()

    # 4. 최근 글로벌 전략 조회
    if global_count > 0:
        cursor.execute("""
            SELECT strategy_id, regime, rsi_zone, market_condition,
                   volatility_level, profit, win_rate, created_at
            FROM global_strategies
            ORDER BY created_at DESC
            LIMIT 10
        """)

        print("최근 글로벌 전략 10개:")
        print(f"{'Strategy ID':<20} {'Regime':<10} {'RSI':<12} {'Market':<12} {'Vol':<10} {'Profit':>8} {'WinRate':>8}")
        print("-" * 100)

        for row in cursor.fetchall():
            strategy_id = row[0][:18]
            regime = row[1] or 'N/A'
            rsi_zone = row[2] or 'N/A'
            market = row[3] or 'N/A'
            volatility = row[4] or 'N/A'
            profit = row[5] or 0
            win_rate = row[6] or 0

            print(f"{strategy_id:<20} {regime:<10} {rsi_zone:<12} {market:<12} {volatility:<10} {profit:>7.2f}% {win_rate:>7.1%}")
        print()

        # 5. Zone별 분포 확인
        cursor.execute("""
            SELECT regime, COUNT(*)
            FROM global_strategies
            GROUP BY regime
        """)
        regime_dist = cursor.fetchall()

        if regime_dist:
            print("Regime 분포:")
            for regime, count in regime_dist:
                print(f"  {regime}: {count}개")
            print()

        cursor.execute("""
            SELECT rsi_zone, COUNT(*)
            FROM global_strategies
            GROUP BY rsi_zone
        """)
        rsi_dist = cursor.fetchall()

        if rsi_dist:
            print("RSI Zone 분포:")
            for rsi, count in rsi_dist:
                print(f"  {rsi}: {count}개")
            print()

    else:
        print("⚠️  글로벌 전략이 하나도 없습니다!")
        print()
        print("💡 글로벌 전략을 생성하려면:")
        print("   1. 여러 코인에 대해 전략 생성 완료")
        print("   2. create_global_strategies_from_results() 실행")
        print()

    conn.close()

    # 6. 글로벌 전략 생성 함수 확인
    print("=" * 70)
    print("🔍 글로벌 전략 생성 함수 위치")
    print("=" * 70)
    print()
    print("진입점: rl_pipeline/strategy/creator.py:3593-3725")
    print("  └─> create_global_strategies_from_results()")
    print()
    print("핵심 로직: rl_pipeline/strategy/zone_based_global_creator.py")
    print("  └─> create_zone_based_global_strategies()")
    print("  └─> save_global_strategies_to_db()")
    print()

if __name__ == '__main__':
    check_global_strategies()
