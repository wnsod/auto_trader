#!/usr/bin/env python
"""1️⃣ 데이터 수집 기능 검증"""
import sys
sys.path.append('/workspace')

from rl_pipeline.data.candle_loader import load_candle_data_for_coin, get_available_coins_and_intervals

print("=" * 70)
print("1️⃣ 데이터 수집 기능 검증")
print("=" * 70)
print()

# Step 1: 사용 가능한 코인 조회
print("Step 1: 사용 가능한 코인 조회")
print("-" * 70)

try:
    available = get_available_coins_and_intervals()
    coins = sorted(list({c for c, _ in available}))

    if coins:
        print(f"✅ 사용 가능한 코인: {len(coins)}개")
        for coin in coins[:5]:  # 최대 5개만 표시
            intervals_for_coin = sorted([i for c, i in available if c == coin])
            print(f"   {coin}: {intervals_for_coin}")

        if len(coins) > 5:
            print(f"   ... 외 {len(coins) - 5}개 코인")
    else:
        print("❌ 사용 가능한 코인이 없습니다")
        sys.exit(1)
except Exception as e:
    print(f"❌ 코인 조회 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 2: 테스트 코인 선택
test_coin = coins[0]
print(f"Step 2: 테스트 코인 선택: {test_coin}")
print("-" * 70)

# 해당 코인의 인터벌 목록
test_intervals = sorted([i for c, i in available if c == test_coin])
print(f"인터벌 목록: {test_intervals}")
print()

# Step 3: 캔들 데이터 로드
print("Step 3: 캔들 데이터 로드")
print("-" * 70)

try:
    candle_data = load_candle_data_for_coin(test_coin, test_intervals)

    if candle_data:
        print(f"✅ 캔들 데이터 로드 성공: {len(candle_data)}개 인터벌")
        print()

        # 각 인터벌별 데이터 개수 확인
        for (coin, interval), df in candle_data.items():
            print(f"   {coin} - {interval}:")
            print(f"     캔들 수: {len(df)}개")

            # 최소 요구사항 확인
            min_candles_per_interval = {
                '15m': 672,
                '30m': 336,
                '240m': 42,
                '1d': 7
            }

            min_required = min_candles_per_interval.get(interval, 100)

            if len(df) >= min_required:
                print(f"     상태: ✅ 충분 (최소 {min_required}개 필요)")
            else:
                print(f"     상태: ⚠️ 부족 (최소 {min_required}개 필요, 현재 {len(df)}개)")

            # 데이터 샘플 확인 (최근 3개)
            if len(df) > 0:
                print(f"     최근 데이터 샘플:")
                recent_df = df.tail(3)
                for idx, row in recent_df.iterrows():
                    timestamp = row.get('timestamp', row.get('candle_date_time_kst', 'N/A'))
                    close = row.get('trade_price', row.get('close', 0))
                    print(f"       {timestamp}: close={close:.2f}")

            print()
    else:
        print("❌ 캔들 데이터 로드 실패")
        sys.exit(1)

except Exception as e:
    print(f"❌ 캔들 데이터 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 데이터 충분성 종합 평가
print("Step 4: 데이터 충분성 종합 평가")
print("-" * 70)

total_candles = sum(len(df) for df in candle_data.values())
print(f"전체 캔들 수: {total_candles}개")

insufficient_count = 0
for (coin, interval), df in candle_data.items():
    min_candles_per_interval = {
        '15m': 672,
        '30m': 336,
        '240m': 42,
        '1d': 7
    }
    min_required = min_candles_per_interval.get(interval, 100)

    if len(df) < min_required:
        insufficient_count += 1

if insufficient_count == 0:
    print("✅ 모든 인터벌이 최소 요구사항 충족")
else:
    print(f"⚠️ {insufficient_count}개 인터벌이 최소 요구사항 미충족")

print()
print("=" * 70)
print("✅ 데이터 수집 기능 검증 완료!")
print("=" * 70)
