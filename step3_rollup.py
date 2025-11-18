#!/usr/bin/env python
"""
3단계: 롤업 실행
"""
import sys
sys.path.append('/workspace')

import sqlite3
from rl_pipeline.engine.rollup_batch import run_full_rollup_and_grades

# 테스트 설정
COIN = 'LINK'
INTERVAL = '15m'

print("=" * 70)
print("3단계: 롤업 및 등급 측정")
print("=" * 70)
print()

# 기존 롤업 데이터 삭제
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("DELETE FROM rl_strategy_rollup WHERE coin=? AND interval=?", (COIN, INTERVAL))
cursor.execute("DELETE FROM strategy_grades WHERE coin=? AND interval=?", (COIN, INTERVAL))
conn.commit()
print(f"✅ 기존 {COIN}-{INTERVAL} 롤업 및 등급 데이터 삭제")
conn.close()
print()

# 롤업 및 등급 측정 실행
print("📊 롤업 및 등급 측정 실행 중...")
try:
    run_full_rollup_and_grades()
    print("✅ 롤업 및 등급 측정 완료")
except Exception as e:
    print(f"⚠️ 롤업 실행 중 오류: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("3단계 완료")
print("=" * 70)
