"""
시스템 전체 실행 상태 모니터링 스크립트
"""
import sqlite3
from datetime import datetime, timedelta
import json
import os

def check_system_status():
    """시스템 전체 실행 상태 확인"""

    print("="*70)
    print("🔍 Absolute Zero System 전체 실행 상태 모니터링")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. 최근 처리된 코인 현황
    try:
        conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
        c = conn.cursor()

        # 최근 24시간 내 처리
        c.execute("""
            SELECT coin, COUNT(*) as strategy_count,
                   COUNT(DISTINCT interval) as interval_count,
                   MAX(created_at) as latest_update
            FROM coin_strategies
            WHERE datetime(created_at) >= datetime('now', '-1 day')
            GROUP BY coin
            ORDER BY latest_update DESC
        """)

        recent_coins = c.fetchall()

        print("\n📊 최근 24시간 내 처리된 코인:")
        print("-"*50)
        for coin, strat_cnt, int_cnt, latest in recent_coins:
            print(f"  🪙 {coin}: {strat_cnt} strategies, {int_cnt} intervals")
            print(f"     마지막: {latest[:19]}")

        print(f"\n  ✅ 총 {len(recent_coins)}개 코인 처리됨")

        # 2. 전체 통계
        c.execute("""
            SELECT
                COUNT(DISTINCT coin) as total_coins,
                COUNT(DISTINCT interval) as total_intervals,
                COUNT(*) as total_strategies,
                MIN(created_at) as first_record,
                MAX(created_at) as last_record
            FROM coin_strategies
        """)

        stats = c.fetchone()
        print(f"\n📈 전체 통계:")
        print(f"  • 총 코인 수: {stats[0]}")
        print(f"  • 총 인터벌 수: {stats[1]}")
        print(f"  • 총 전략 수: {stats[2]:,}")
        print(f"  • 첫 기록: {stats[3][:19]}")
        print(f"  • 최근 기록: {stats[4][:19]}")

        # 3. 인터벌별 전략 수
        c.execute("""
            SELECT interval, COUNT(*) as cnt
            FROM coin_strategies
            WHERE datetime(created_at) >= datetime('now', '-1 day')
            GROUP BY interval
            ORDER BY cnt DESC
        """)

        interval_stats = c.fetchall()
        if interval_stats:
            print(f"\n📐 인터벌별 전략 수 (24시간):")
            for interval, cnt in interval_stats:
                print(f"  • {interval}: {cnt} strategies")

        conn.close()

    except Exception as e:
        print(f"❌ 전략 DB 확인 실패: {e}")

    # 4. 세션 로그 확인
    try:
        session_file = '/workspace/rl_pipeline/debug_logs/sessions.json'
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                sessions = json.load(f)

            print(f"\n🗂️ 디버그 세션 현황:")
            print(f"  • 총 세션 수: {len(sessions.get('sessions', []))}")

            # 최근 5개 세션
            recent_sessions = sessions.get('sessions', [])[-5:]
            if recent_sessions:
                print(f"  • 최근 5개 세션:")
                for sess in recent_sessions:
                    status = sess.get('status', 'unknown')
                    symbol = '🟢' if status == 'completed' else '🔴' if status == 'failed' else '🟡'
                    print(f"    {symbol} {sess['session_id'][:30]}... ({status})")
    except Exception as e:
        print(f"❌ 세션 로그 확인 실패: {e}")

    # 5. 검증 결과 요약
    try:
        val_log = '/workspace/rl_pipeline/validation/reports/validation_log.jsonl'
        if os.path.exists(val_log):
            with open(val_log, 'r') as f:
                lines = f.readlines()

            # 최근 10개 검증 결과
            recent_validations = []
            for line in lines[-10:]:
                try:
                    recent_validations.append(json.loads(line))
                except:
                    pass

            if recent_validations:
                print(f"\n✅ 최근 검증 결과 (최근 10건):")
                passed = sum(1 for v in recent_validations if v['status'] == 'passed')
                warning = sum(1 for v in recent_validations if v['status'] == 'warning')
                failed = sum(1 for v in recent_validations if v['status'] == 'failed')

                print(f"  • 통과: {passed}")
                print(f"  • 경고: {warning}")
                print(f"  • 실패: {failed}")
                print(f"  • 성공률: {passed/len(recent_validations)*100:.1f}%")
    except Exception as e:
        print(f"❌ 검증 로그 확인 실패: {e}")

    # 6. 실행 권장사항
    print(f"\n🚀 실행 권장사항:")
    if len(recent_coins) == 0:
        print("  ⚠️ 최근 24시간 내 처리된 코인이 없습니다.")
        print("  → absolute_zero_improved.py 실행을 권장합니다.")
    elif len(recent_coins) < 5:
        print(f"  ⚠️ 일부 코인만 처리됨 ({len(recent_coins)}개)")
        print("  → 전체 코인 대상으로 실행을 권장합니다.")
    else:
        print(f"  ✅ 정상 실행 중 ({len(recent_coins)}개 코인 처리)")

    print("="*70)

if __name__ == "__main__":
    check_system_status()