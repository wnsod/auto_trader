#!/usr/bin/env python
"""
1단계: 전략 생성 및 검증
"""
import sys
sys.path.append('/workspace')

import sqlite3
from rl_pipeline.data.candle_loader import load_candle_data_for_coin
from rl_pipeline.strategy.creator import create_coin_strategies

# 테스트 설정
COIN = 'LINK'
INTERVAL = '15m'

print("=" * 70)
print("1단계: 전략 생성")
print("=" * 70)
print()

# 캔들 데이터 로드
print(f"📥 {COIN}-{INTERVAL} 캔들 데이터 로드 중...")
candle_data_dict = load_candle_data_for_coin(COIN, [INTERVAL])

if (COIN, INTERVAL) not in candle_data_dict:
    print(f"❌ 캔들 데이터를 찾을 수 없습니다.")
    sys.exit(1)

candle_data = candle_data_dict[(COIN, INTERVAL)]
print(f"✅ {len(candle_data)}개 캔들 로드")
print(f"   최신 캔들: {candle_data['timestamp'].max()}")
print(f"   최신 가격: {candle_data['close'].iloc[-1]:,.0f}원")
print()

# 기존 LINK 전략 삭제
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 테이블 목록 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%strateg%'")
strategy_tables = [r[0] for r in cursor.fetchall()]
print(f"전략 관련 테이블: {strategy_tables}")
print()

# coin_strategies 테이블에서 삭제
if 'coin_strategies' in strategy_tables:
    cursor.execute("DELETE FROM coin_strategies WHERE coin = ? AND interval = ?", (COIN, INTERVAL))
    conn.commit()
    print(f"✅ 기존 {COIN}-{INTERVAL} 전략 삭제")

conn.close()

# 전략 생성
print(f"\n🔨 {COIN}-{INTERVAL} 전략 생성 중...")
num_created = create_coin_strategies(
    coin=COIN,
    intervals=[INTERVAL],
    all_candle_data=candle_data_dict
)

print(f"✅ {num_created}개 전략 생성 완료")
print()

# 검증
print("=" * 70)
print("전략 데이터 검증")
print("=" * 70)
print()

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# coin_strategies 테이블 확인
if 'coin_strategies' in strategy_tables:
    cursor.execute("""
        SELECT COUNT(*) FROM coin_strategies
        WHERE coin = ? AND interval = ?
    """, (COIN, INTERVAL))
    total_strategies = cursor.fetchone()[0]
    print(f"📊 총 전략 수: {total_strategies}개")

    # 방향별 분포
    cursor.execute("""
        SELECT direction, COUNT(*)
        FROM coin_strategies
        WHERE coin = ? AND interval = ?
        GROUP BY direction
    """, (COIN, INTERVAL))
    directions = cursor.fetchall()
    print(f"\n방향별 분포:")
    for direction, count in directions:
        print(f"  {direction}: {count}개")

    # 레짐별 분포
    cursor.execute("""
        SELECT regime, COUNT(*)
        FROM coin_strategies
        WHERE coin = ? AND interval = ?
        GROUP BY regime
    """, (COIN, INTERVAL))
    regimes = cursor.fetchall()
    print(f"\n레짐별 분포:")
    for regime, count in regimes:
        print(f"  {regime}: {count}개")

    # 샘플 전략 5개
    cursor.execute("""
        SELECT strategy_id, direction, regime
        FROM coin_strategies
        WHERE coin = ? AND interval = ?
        LIMIT 5
    """, (COIN, INTERVAL))
    samples = cursor.fetchall()
    print(f"\n샘플 전략:")
    for sid, direction, regime in samples:
        print(f"  {sid[:60]}")
        print(f"    방향: {direction}, 레짐: {regime}")

else:
    print("⚠️ coin_strategies 테이블을 찾을 수 없습니다.")

conn.close()

print()
print("=" * 70)
print("1단계 완료")
print("=" * 70)
