#!/usr/bin/env python
import sys
sys.path.append('/workspace')

import sqlite3
from datetime import datetime

# 이전 예측 에피소드 삭제
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("DELETE FROM rl_episodes WHERE episode_id LIKE 'pred_LINK_15m_%'")
cursor.execute("DELETE FROM rl_episode_summary WHERE episode_id LIKE 'pred_LINK_15m_%'")
conn.commit()
print("✅ 이전 테스트 데이터 삭제 완료")
conn.close()

# 예측 self-play 실행
print("\n" + "=" * 70)
print("예측 self-play 테스트: LINK-15m")
print("=" * 70)
print()

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator
from argparse import Namespace

orchestrator = IntegratedPipelineOrchestrator()

args = Namespace(
    mode='train',
    coins=['LINK'],
    skip_candle_fetch=True,
    skip_strategy_creation=True,
    skip_predictive_rl=False,  # 예측 self-play 실행
    skip_rollup=True,
    skip_grading=True
)

# 5개 에피소드만 생성
orchestrator.run_partial_pipeline(args, {'15m': 5})

print("\n" + "=" * 70)
print("결과 검증")
print("=" * 70)
print()

# 결과 확인
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT
        e.episode_id, e.ts_entry, e.entry_price, e.target_move_pct, e.horizon_k,
        e.predicted_dir,
        s.ts_exit, s.first_event, s.t_hit, s.realized_ret_signed
    FROM rl_episodes e
    LEFT JOIN rl_episode_summary s ON e.episode_id = s.episode_id
    WHERE e.episode_id LIKE 'pred_LINK_15m_%'
    ORDER BY e.ts_entry DESC
    LIMIT 10
""")

episodes = cursor.fetchall()

if episodes:
    print(f"📊 생성된 에피소드: {len(episodes)}개\n")

    tp_count = sum(1 for ep in episodes if ep[7] == 'TP')
    sl_count = sum(1 for ep in episodes if ep[7] == 'SL')
    expiry_count = sum(1 for ep in episodes if ep[7] == 'expiry')

    print(f"결과 분포:")
    print(f"  TP: {tp_count}개 ({tp_count/len(episodes)*100:.1f}%)")
    print(f"  SL: {sl_count}개 ({sl_count/len(episodes)*100:.1f}%)")
    print(f"  만료: {expiry_count}개 ({expiry_count/len(episodes)*100:.1f}%)")
    print()

    print("샘플 에피소드:")
    for ep in episodes[:3]:
        (ep_id, ts_entry, entry_price, target_move_pct, horizon_k,
         predicted_dir, ts_exit, first_event, t_hit, realized_ret) = ep

        entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M')

        print(f"\n  {ep_id[:60]}...")
        print(f"    진입: {entry_time} @ {entry_price:,.0f}원")
        print(f"    목표: {target_move_pct*100:.1f}%, 기간: {horizon_k}캔들")
        print(f"    결과: {first_event} (t={t_hit}) 수익률: {realized_ret:.4f}")

    # 타임스탬프 유효성 검증
    first_ep = episodes[0]
    ts_entry = first_ep[1]
    now = int(datetime.now().timestamp())
    days_ago = (now - ts_entry) / 86400

    print("\n타임스탬프 검증:")
    if ts_entry < 1000:
        print(f"  ❌ 오류: ts_entry={ts_entry} (너무 작음)")
    elif days_ago > 365:
        print(f"  ❌ 오류: {days_ago:.1f}일 전 (너무 오래됨)")
    elif days_ago < 0:
        print(f"  ❌ 오류: 미래 시간")
    else:
        print(f"  ✅ 정상: {days_ago:.1f}일 전 데이터")

else:
    print("❌ 에피소드가 생성되지 않았습니다.")

conn.close()

print("\n" + "=" * 70)
print("테스트 완료")
print("=" * 70)
