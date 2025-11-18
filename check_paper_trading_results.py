"""
Paper Trading 결과 확인 스크립트
"""
import sqlite3
import json
from datetime import datetime, timedelta

def check_paper_trading_results():
    """Paper Trading 결과 확인"""

    print("="*70)
    print("📈 Paper Trading 결과 확인")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Paper Trading 세션 확인
    try:
        conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
        c = conn.cursor()

        # Paper Trading 세션 조회
        c.execute("""
            SELECT session_id, coin, interval, created_at, status
            FROM paper_trading_sessions
            WHERE datetime(created_at) >= datetime('now', '-1 hour')
            ORDER BY created_at DESC
            LIMIT 10
        """)

        sessions = c.fetchall()

        if sessions:
            print(f"\n📊 최근 Paper Trading 세션 (1시간 내):")
            print("-"*50)
            for sess_id, coin, interval, created, status in sessions:
                print(f"  • {coin}-{interval}: {sess_id[:20]}...")
                print(f"    생성: {created[:19]}, 상태: {status or 'running'}")
        else:
            print(f"\n⚠️ 최근 1시간 내 Paper Trading 세션이 없습니다.")

        # Paper Trading 결과 조회
        c.execute("""
            SELECT coin, interval,
                   COUNT(*) as trade_count,
                   SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as win_count,
                   AVG(profit_loss) as avg_profit,
                   SUM(profit_loss) as total_profit,
                   MAX(created_at) as latest_trade
            FROM paper_trading_results
            WHERE datetime(created_at) >= datetime('now', '-1 hour')
            GROUP BY coin, interval
            ORDER BY latest_trade DESC
        """)

        results = c.fetchall()

        if results:
            print(f"\n💰 Paper Trading 성과 (1시간):")
            print("-"*50)

            total_trades = 0
            total_wins = 0
            total_profit = 0

            for coin, interval, trades, wins, avg_profit, tot_profit, latest in results:
                win_rate = (wins / trades * 100) if trades > 0 else 0
                print(f"\n  🪙 {coin}-{interval}:")
                print(f"    • 거래 수: {trades}")
                print(f"    • 승률: {win_rate:.1f}% ({wins}/{trades})")
                print(f"    • 평균 수익: {avg_profit:.4f}" if avg_profit else "    • 평균 수익: N/A")
                print(f"    • 총 수익: {tot_profit:.4f}" if tot_profit else "    • 총 수익: N/A")
                print(f"    • 최근 거래: {latest[:19]}")

                total_trades += trades
                total_wins += wins or 0
                total_profit += tot_profit or 0

            # 전체 요약
            if total_trades > 0:
                overall_win_rate = total_wins / total_trades * 100
                print(f"\n📈 전체 Paper Trading 요약:")
                print(f"  • 총 거래: {total_trades}")
                print(f"  • 전체 승률: {overall_win_rate:.1f}%")
                print(f"  • 전체 수익: {total_profit:.4f}")
        else:
            print(f"\n⚠️ Paper Trading 결과가 아직 생성되지 않았습니다.")
            print(f"   (Paper Trading은 백그라운드에서 자동 실행됩니다)")

        # 예측 정확도 확인 (rl_episodes)
        c.execute("""
            SELECT coin, interval,
                   COUNT(*) as episode_count,
                   AVG(CASE WHEN predicted_direction = actual_direction THEN 1.0 ELSE 0.0 END) as accuracy
            FROM rl_episodes
            WHERE datetime(created_at) >= datetime('now', '-1 hour')
            GROUP BY coin, interval
        """)

        episodes = c.fetchall()

        if episodes:
            print(f"\n🎯 예측 정확도 (rl_episodes):")
            print("-"*50)
            for coin, interval, ep_count, accuracy in episodes:
                if accuracy is not None:
                    print(f"  • {coin}-{interval}: {accuracy*100:.1f}% ({ep_count}개 에피소드)")

        conn.close()

    except Exception as e:
        print(f"❌ Paper Trading 결과 조회 실패: {e}")
        return

    # 실행 중인 프로세스 확인
    print(f"\n⚙️ Paper Trading 프로세스 상태:")
    print("-"*50)

    try:
        import subprocess
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            cwd='/workspace'
        )

        if 'paper_trading' in result.stdout.lower():
            print(f"  🟢 Paper Trading 프로세스 실행 중")
        else:
            print(f"  🟡 Paper Trading 프로세스 감지되지 않음")
            print(f"     (백그라운드 실행 또는 대기 상태일 수 있음)")
    except:
        pass

    # 권장사항
    print(f"\n💡 Paper Trading 권장사항:")
    print("-"*50)

    if not sessions:
        print("  1. Paper Trading 세션 생성 확인 필요")
        print("     → absolute_zero_improved.py 실행 시 자동 생성됨")

    if not results:
        print("  2. Paper Trading 결과 대기")
        print("     → 보통 5-10분 후 첫 결과 생성")
        print("     → 30분 이상 실행 시 신뢰할 수 있는 통계")

    print("  3. Paper Trading 모니터링")
    print("     → 주기적으로 결과 확인 (15분 간격)")
    print("     → 승률 50% 이상 전략 식별")
    print("     → 수익률 높은 코인/인터벌 조합 파악")

if __name__ == "__main__":
    check_paper_trading_results()