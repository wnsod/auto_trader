#!/usr/bin/env python
import sys
sys.path.append('/workspace')

import sqlite3
import pandas as pd
from datetime import datetime
from rl_pipeline.data.candle_loader import load_candle_data_for_coin
from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

print("=" * 70)
print("예측 self-play 간단 테스트")
print("=" * 70)
print()

# 이전 테스트 데이터 삭제
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()
cursor.execute("DELETE FROM rl_episodes WHERE episode_id LIKE 'pred_LINK_15m_%'")
cursor.execute("DELETE FROM rl_episode_summary WHERE episode_id LIKE 'pred_LINK_15m_%'")
conn.commit()
print("✅ 이전 테스트 데이터 삭제")
conn.close()

# 캔들 데이터 로드
print("\n📥 LINK-15m 캔들 데이터 로드 중...")
candle_data_dict = load_candle_data_for_coin('LINK', ['15m'])
if ('LINK', '15m') not in candle_data_dict:
    print("❌ 캔들 데이터를 찾을 수 없습니다.")
    sys.exit(1)

candle_data = candle_data_dict[('LINK', '15m')]
print(f"✅ {len(candle_data)}개 캔들 로드 완료")

# LINK-15m 전략 조회
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# coin_strategies 테이블 존재 여부 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_strategies'")
if cursor.fetchone():
    cursor.execute("""
        SELECT strategy_id, direction, regime
        FROM coin_strategies
        WHERE coin = 'LINK' AND interval = '15m'
        LIMIT 5
    """)
    strategies = cursor.fetchall()
    print(f"\n📋 LINK-15m 전략: {len(strategies)}개")
else:
    print("\n⚠️ coin_strategies 테이블이 없습니다. 전략 없이 테스트 진행...")
    strategies = []

conn.close()

if not strategies:
    print("⚠️ 전략이 없어서 예측을 생성할 수 없습니다.")
    print("   실제 파이프라인은 전략이 있어야 예측 self-play가 작동합니다.")
    sys.exit(0)

# Orchestrator로 예측 생성
print("\n🚀 예측 생성 중...")
orchestrator = IntegratedPipelineOrchestrator()

# _create_predictions_with_policy 직접 호출
predictions = orchestrator._create_predictions_with_policy(
    coin='LINK',
    interval='15m',
    candle_data=candle_data,
    strategies=[{'id': s[0], 'direction': s[1], 'regime': s[2]} for s in strategies[:5]],
    num_episodes_per_strategy=1  # 전략당 1개씩
)

print(f"✅ {len(predictions)}개 예측 생성 완료")

# 예측 결과 시뮬레이션
if predictions:
    print("\n🎯 시뮬레이션 실행 중...")
    orchestrator._check_prediction_results(
        coin='LINK',
        interval='15m',
        candle_data=candle_data,
        predictions=predictions,
        candle_seconds=900  # 15분 = 900초
    )
    print("✅ 시뮬레이션 완료")

# 결과 확인
print("\n" + "=" * 70)
print("결과 검증")
print("=" * 70)
print()

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN first_event = 'TP' THEN 1 ELSE 0 END) as tp,
        SUM(CASE WHEN first_event = 'SL' THEN 1 ELSE 0 END) as sl,
        SUM(CASE WHEN first_event = 'expiry' THEN 1 ELSE 0 END) as expiry,
        AVG(realized_ret_signed) as avg_ret
    FROM rl_episode_summary
    WHERE episode_id LIKE 'pred_LINK_15m_%'
""")

total, tp, sl, expiry, avg_ret = cursor.fetchone()

if total and total > 0:
    print(f"📊 전체 에피소드: {total}개")
    print(f"  TP: {tp}개 ({tp/total*100:.1f}%)")
    print(f"  SL: {sl}개 ({sl/total*100:.1f}%)")
    print(f"  만료: {expiry}개 ({expiry/total*100:.1f}%)")
    if avg_ret:
        print(f"  평균 수익률: {avg_ret:.4f} ({avg_ret*100:.2f}%)")
    print()

    # 샘플 에피소드 확인
    cursor.execute("""
        SELECT
            e.episode_id, e.ts_entry, e.entry_price, e.target_move_pct,
            s.first_event, s.t_hit, s.realized_ret_signed
        FROM rl_episodes e
        LEFT JOIN rl_episode_summary s ON e.episode_id = s.episode_id
        WHERE e.episode_id LIKE 'pred_LINK_15m_%'
        ORDER BY e.ts_entry DESC
        LIMIT 5
    """)

    episodes = cursor.fetchall()
    print("샘플 에피소드:")
    for ep_id, ts_entry, entry_price, target_pct, first_event, t_hit, ret in episodes:
        entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M')
        print(f"  {entry_time} @ {entry_price:,.0f}원 → {first_event} (t={t_hit}) 수익률: {ret:.4f}")

else:
    print("❌ 에피소드가 생성되지 않았습니다.")

conn.close()

print("\n" + "=" * 70)
print("테스트 완료")
print("=" * 70)
