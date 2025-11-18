"""
간단한 selfplay 테스트 - 수정사항 검증
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.simulation.selfplay import run_self_play_test
from rl_pipeline.monitoring import SessionManager
from rl_pipeline.db.candle_reader import load_coin_interval_candles
import json

# 세션 생성
session_manager = SessionManager()
session_id = session_manager.create_session(coins=['BTC'], intervals=['240m'])
print(f'✅ Session created: {session_id}')

# BTC 240m 캔들 데이터 로드
print(f'📊 Loading BTC 240m candle data...')
candle_data = load_coin_interval_candles('BTC', '240m')
print(f'✅ Loaded {len(candle_data)} candles')

# 간단한 전략 파라미터
strategy_params_list = [
    {'rsi_min': 30, 'rsi_max': 70, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05},
    {'rsi_min': 25, 'rsi_max': 75, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06},
    {'rsi_min': 35, 'rsi_max': 65, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.04},
    {'rsi_min': 28, 'rsi_max': 72, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.055},
]

print(f'🚀 Starting BTC 240m selfplay test (10 episodes, 4 agents)...')
result = run_self_play_test(
    strategy_params_list=strategy_params_list,
    episodes=10,  # 200 -> 10으로 줄임
    candle_data=candle_data,
    coin='BTC',
    interval='240m',
    session_id=session_id
)

# 결과 출력
if result.get('status') == 'success':
    summary = result.get('summary', {})
    print(f'\n📊 테스트 결과:')
    print(f"  - 평균 PnL: {summary.get('avg_pnl', 0):.2f}")
    print(f"  - 평균 승률: {summary.get('avg_win_rate', 0):.2%}")
    print(f"  - 총 거래: {summary.get('total_trades', 0)}")
    print(f"  - Best Agent PnL: {summary.get('best_agent_pnl', 0):.2f}")

    # 디버그 파일 확인
    debug_dir = f'/workspace/rl_pipeline/debug_logs/{session_id}'
    print(f'\n📁 Debug logs: {debug_dir}')

    # simulation.jsonl에서 statistics 확인
    try:
        import subprocess
        stats_cmd = f"tail -1 {debug_dir}/simulation.jsonl"
        stats_output = subprocess.check_output(['bash', '-c', stats_cmd], text=True)
        stats_data = json.loads(stats_output)
        statistics = stats_data.get('statistics', {})
        print(f'\n📈 Statistics 필드:')
        print(f"  - total_episodes: {statistics.get('total_episodes', 0)}")
        print(f"  - total_trades: {statistics.get('total_trades', 0)}")
        print(f"  - winning_trades: {statistics.get('winning_trades', 0)}")
        print(f"  - losing_trades: {statistics.get('losing_trades', 0)}")

        if statistics.get('total_episodes', 0) > 0:
            print(f'\n✅ Statistics 필드가 정상적으로 업데이트되었습니다!')
        else:
            print(f'\n⚠️ Statistics 필드가 여전히 0입니다.')
    except Exception as e:
        print(f'⚠️ Statistics 확인 실패: {e}')

    print('\n✅ 테스트 완료!')
else:
    print(f'\n❌ 테스트 실패: {result.get("error", "Unknown error")}')
