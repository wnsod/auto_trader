import sys
sys.path.append('/workspace')

import sqlite3
from datetime import datetime

# 데이터베이스에서 최근 전략 1개 선택
conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# LINK-15m 전략 1개만 선택
cursor.execute("""
    SELECT id, coin, interval, regime, direction
    FROM rl_strategy
    WHERE coin = 'LINK' AND interval = '15m'
    LIMIT 1
""")

strategy = cursor.fetchone()
if not strategy:
    print("❌ 전략을 찾을 수 없습니다.")
    sys.exit(1)

strat_id, coin, interval, regime, direction = strategy
print(f"테스트 전략: {strat_id}")
print(f"코인: {coin}, 인터벌: {interval}, 레짐: {regime}, 방향: {direction}")
print()

# 이전 에피소드 삭제
cursor.execute("DELETE FROM rl_episode_summary WHERE id LIKE ?", (f"pred_{coin}_{interval}_%",))
conn.commit()
print(f"✅ 이전 에피소드 삭제 완료")
print()

# 예측 self-play 실행
from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator

orchestrator = IntegratedPipelineOrchestrator()

# run_partial_pipeline 인자
from argparse import Namespace
args = Namespace(
    mode='train',
    coins=[coin],
    skip_candle_fetch=True,
    skip_strategy_creation=True,
    skip_predictive_rl=False,  # 예측 self-play 실행
    skip_rollup=True,
    skip_grading=True
)

print("🚀 예측 self-play 시작...")
orchestrator.run_partial_pipeline(args, {interval: 5})  # 5개 에피소드만 생성
print()

# 결과 검증
cursor.execute("""
    SELECT id, ts_entry, entry_price, first_event
    FROM rl_episode_summary
    WHERE id LIKE ?
    ORDER BY ts_entry DESC
    LIMIT 5
""", (f"pred_{coin}_{interval}_%",))

episodes = cursor.fetchall()
print(f"📊 생성된 에피소드: {len(episodes)}개")
print()

for ep_id, ts_entry, entry_price, first_event in episodes[:3]:
    entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Episode: {ep_id}")
    print(f"  진입 시간: {entry_time} (ts={ts_entry})")
    print(f"  진입 가격: {entry_price:,.0f}원")
    print(f"  첫 이벤트: {first_event}")
    print()

# 타임스탬프 검증
if episodes:
    ts_entry = episodes[0][1]
    now = int(datetime.now().timestamp())
    past_days = (now - ts_entry) / 86400

    if ts_entry < 1000:  # 1970년대
        print(f"❌ 타임스탬프 오류: ts_entry={ts_entry} (너무 작음)")
    elif past_days > 365:  # 1년 이상 과거
        print(f"❌ 타임스탬프 오류: {past_days:.1f}일 전 데이터")
    elif past_days < 0:  # 미래
        print(f"❌ 타임스탬프 오류: 미래 시간 ({-past_days:.1f}일 후)")
    else:
        print(f"✅ 타임스탬프 검증 성공: {past_days:.1f}일 전 데이터")

conn.close()
