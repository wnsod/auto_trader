#!/usr/bin/env python
"""
1단계 검증: 전략 생성 확인
"""
import sqlite3

COIN = 'LINK'
INTERVAL = '15m'

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

print("=" * 70)
print("1단계 검증: 전략 생성")
print("=" * 70)
print()

# 총 전략 수
cursor.execute("SELECT COUNT(*) FROM coin_strategies WHERE coin=? AND interval=?", (COIN, INTERVAL))
total = cursor.fetchone()[0]
print(f"📊 총 {COIN}-{INTERVAL} 전략: {total}개")
print()

if total > 0:
    # strategy_type별 분포
    cursor.execute("""
        SELECT strategy_type, COUNT(*)
        FROM coin_strategies
        WHERE coin=? AND interval=?
        GROUP BY strategy_type
        ORDER BY COUNT(*) DESC
    """, (COIN, INTERVAL))
    types = cursor.fetchall()
    print("전략 타입별 분포:")
    for stype, count in types:
        print(f"  {stype:20s}: {count:3d}개")
    print()

    # regime별 분포
    cursor.execute("""
        SELECT regime, COUNT(*)
        FROM coin_strategies
        WHERE coin=? AND interval=?
        GROUP BY regime
        ORDER BY COUNT(*) DESC
    """, (COIN, INTERVAL))
    regimes = cursor.fetchall()
    print("레짐별 분포:")
    for regime, count in regimes:
        regime_name = regime if regime else 'NULL'
        print(f"  {regime_name:20s}: {count:3d}개")
    print()

    # market_condition별 분포
    cursor.execute("""
        SELECT market_condition, COUNT(*)
        FROM coin_strategies
        WHERE coin=? AND interval=?
        GROUP BY market_condition
        ORDER BY COUNT(*) DESC
    """, (COIN, INTERVAL))
    conditions = cursor.fetchall()
    print("시장 상황별 분포:")
    for cond, count in conditions:
        cond_name = cond if cond else 'NULL'
        print(f"  {cond_name:20s}: {count:3d}개")
    print()

    # 샘플 전략 5개
    cursor.execute("""
        SELECT id, strategy_type, regime, rsi_min, rsi_max, take_profit_pct, stop_loss_pct
        FROM coin_strategies
        WHERE coin=? AND interval=?
        LIMIT 5
    """, (COIN, INTERVAL))
    samples = cursor.fetchall()
    print("샘플 전략 5개:")
    for sid, stype, regime, rsi_min, rsi_max, tp, sl in samples:
        print(f"  {sid}")
        print(f"    타입: {stype}, 레짐: {regime}")
        print(f"    RSI: [{rsi_min:.1f}, {rsi_max:.1f}], TP: {tp:.2f}%, SL: {sl:.2f}%")
    print()

    # 파라미터 범위 확인
    cursor.execute("""
        SELECT
            AVG(rsi_min), AVG(rsi_max),
            AVG(take_profit_pct), AVG(stop_loss_pct),
            AVG(volume_ratio_min), AVG(volume_ratio_max)
        FROM coin_strategies
        WHERE coin=? AND interval=?
    """, (COIN, INTERVAL))
    avg_rsi_min, avg_rsi_max, avg_tp, avg_sl, avg_vol_min, avg_vol_max = cursor.fetchone()
    print("평균 파라미터:")
    print(f"  RSI: [{avg_rsi_min:.1f}, {avg_rsi_max:.1f}]")
    print(f"  TP: {avg_tp:.2f}%, SL: {avg_sl:.2f}%")
    print(f"  Volume 비율: [{avg_vol_min:.2f}, {avg_vol_max:.2f}]")

    print()
    print("✅ 1단계 검증 완료: 전략 생성 성공")
else:
    print("❌ 전략이 생성되지 않았습니다.")

conn.close()
