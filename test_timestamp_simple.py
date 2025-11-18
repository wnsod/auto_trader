#!/usr/bin/env python
import sys
sys.path.append('/workspace')

import pandas as pd
from rl_pipeline.data.candle_loader import load_candle_data_for_coin
from datetime import datetime

print("=" * 60)
print("타임스탬프 변환 테스트")
print("=" * 60)
print()

# LINK-15m 캔들 데이터 로드
print("📥 LINK-15m 캔들 데이터 로드 중...")
candle_data = load_candle_data_for_coin('LINK', ['15m'])

if ('LINK', '15m') not in candle_data:
    print("❌ 캔들 데이터를 찾을 수 없습니다.")
    sys.exit(1)

df = candle_data[('LINK', '15m')]
print(f"✅ {len(df)}개 캔들 로드 완료")
print()

# 타임스탬프 검증
print("🔍 타임스탬프 검증:")
print(f"  DataFrame shape: {df.shape}")
print(f"  timestamp 컬럼 타입: {df['timestamp'].dtype}")
print()

# 최근 3개 캔들 확인
print("최근 3개 캔들:")
recent_3 = df.tail(3)
for idx, row in recent_3.iterrows():
    ts_val = row['timestamp']
    print(f"  타입: {type(ts_val)}")
    print(f"  값: {ts_val}")
    if isinstance(ts_val, pd.Timestamp):
        unix_ts = int(ts_val.timestamp())
        print(f"  Unix 타임스탬프: {unix_ts}")
        print(f"  isinstance(pd.Timestamp): True")
    else:
        print(f"  isinstance(pd.Timestamp): False")
    print(f"  close: {row['close']}")
    print()

# 최근 캔들의 타임스탬프 추출 테스트
print("🎯 타임스탬프 추출 로직 테스트:")
ts_value = df['timestamp'].iloc[-1]
print(f"  ts_value 타입: {type(ts_value)}")
print(f"  ts_value 값: {ts_value}")

if isinstance(ts_value, pd.Timestamp):
    ts_entry = int(ts_value.timestamp())
    print(f"  ✅ pandas.Timestamp 감지!")
    print(f"  ts_entry = {ts_entry}")
    entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  진입 시간: {entry_time}")
else:
    ts_entry = int(ts_value)
    print(f"  ⚠️ pandas.Timestamp 아님!")
    print(f"  ts_entry = {ts_entry}")

print()

# 타임스탬프 유효성 검증
now = int(datetime.now().timestamp())
past_days = (now - ts_entry) / 86400

print("📊 타임스탬프 유효성 검증:")
print(f"  현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (ts={now})")
print(f"  진입 시간: {datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M:%S')} (ts={ts_entry})")
print(f"  시간 차이: {past_days:.1f}일")

if ts_entry < 1000:
    print(f"  ❌ 타임스탬프 오류: ts_entry={ts_entry} (너무 작음)")
elif past_days > 365:
    print(f"  ❌ 타임스탬프 오류: {past_days:.1f}일 전 데이터 (너무 오래됨)")
elif past_days < 0:
    print(f"  ❌ 타임스탬프 오류: 미래 시간 ({-past_days:.1f}일 후)")
elif past_days > 7:
    print(f"  ⚠️ 경고: {past_days:.1f}일 전 데이터 (조금 오래됨)")
else:
    print(f"  ✅ 타임스탬프 정상: {past_days:.1f}일 전 데이터")

print()
print("=" * 60)
print("테스트 완료")
print("=" * 60)
