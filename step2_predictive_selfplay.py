#!/usr/bin/env python
"""
2단계: 예측 Self-play 실행
"""
import sys
sys.path.append('/workspace')

import sqlite3
from rl_pipeline.data.candle_loader import load_candle_data_for_coin
from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

# 테스트 설정
COIN = 'LINK'
INTERVAL = '15m'
EPISODES_PER_STRATEGY = 10  # 전략당 10개 에피소드

print("=" * 70)
print("2단계: 예측 Self-play")
print("=" * 70)
print()

# 기존 예측 에피소드 삭제
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# rl_episode_summary 테이블 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_episode_summary'")
if cursor.fetchone():
    cursor.execute("DELETE FROM rl_episode_summary WHERE episode_id LIKE ?", (f"pred_{COIN}_{INTERVAL}_%",))
    print(f"✅ 기존 {COIN}-{INTERVAL} 에피소드 삭제 (summary)")

# rl_episodes 테이블 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_episodes'")
if cursor.fetchone():
    cursor.execute("DELETE FROM rl_episodes WHERE coin=? AND interval=?", (COIN, INTERVAL))
    print(f"✅ 기존 {COIN}-{INTERVAL} 에피소드 삭제 (episodes)")

conn.commit()
conn.close()
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
print()

# Orchestrator 초기화
orchestrator = IntegratedPipelineOrchestrator()

# 예측 self-play 실행
print(f"🎮 예측 Self-play 실행 중...")
print(f"   설정: 전략당 {EPISODES_PER_STRATEGY}개 에피소드")
print()

try:
    # run_partial_pipeline 호출
    result = orchestrator.run_partial_pipeline(COIN, INTERVAL, candle_data)
    print(f"✅ 파이프라인 실행 완료")
    print(f"   생성된 에피소드: {result.metrics.get('episodes_generated', 0)}개")
except Exception as e:
    print(f"⚠️ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("2단계 완료")
print("=" * 70)
