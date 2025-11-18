#!/usr/bin/env python3
"""시그널 점수 및 액션 확인"""
import sqlite3
import sys

DB_PATH = '/workspace/data_storage/trading_system.db'

print("=" * 80)
print("최근 Combined 시그널 확인 (변동성 시스템 작동 여부)")
print("=" * 80)

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 최근 combined 시그널 조회
    cursor.execute("""
        SELECT coin, signal_score, action, confidence,
               rsi, macd, volatility, volume_ratio,
               datetime(timestamp, 'unixepoch', 'localtime') as time
        FROM signals
        WHERE interval = 'combined'
        ORDER BY timestamp DESC
        LIMIT 10
    """)

    results = cursor.fetchall()

    if not results:
        print("⚠️ 시그널 데이터가 없습니다")
        sys.exit(0)

    print(f"\n총 {len(results)}개의 시그널 확인:\n")

    for row in results:
        coin, signal_score, action, confidence, rsi, macd, volatility, volume_ratio, time = row

        print(f"🪙 {coin}")
        print(f"   📊 Signal Score: {signal_score:.4f}")
        print(f"   🎯 Action: {action}")
        print(f"   ✅ Confidence: {confidence:.3f}")
        print(f"   📈 RSI: {rsi:.2f}, MACD: {macd:.4f}")
        print(f"   💨 Volatility: {volatility:.4f}, Volume: {volume_ratio:.2f}x")
        print(f"   ⏰ Time: {time}")
        print()

    # 액션별 통계
    print("=" * 80)
    print("액션별 통계")
    print("=" * 80)

    cursor.execute("""
        SELECT action, COUNT(*) as count,
               AVG(signal_score) as avg_score,
               MIN(signal_score) as min_score,
               MAX(signal_score) as max_score
        FROM signals
        WHERE interval = 'combined'
        AND timestamp > strftime('%s', 'now', '-1 hour')
        GROUP BY action
    """)

    action_stats = cursor.fetchall()

    for action, count, avg_score, min_score, max_score in action_stats:
        print(f"\n{action}: {count}개")
        print(f"   평균 점수: {avg_score:.4f}")
        print(f"   범위: {min_score:.4f} ~ {max_score:.4f}")

    conn.close()

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
