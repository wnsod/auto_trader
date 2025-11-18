"""
240m 인터벌만 테스트 - 강제 청산 검증
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator
from rl_pipeline.monitoring import SessionManager
from rl_pipeline.db.reads import load_strategies_pool
import json

# 세션 생성
session_manager = SessionManager()
session_id = session_manager.create_session(coins=['BTC'], intervals=['240m'])
print(f'✅ Session created: {session_id}')

# 오케스트레이터 초기화
orchestrator = IntegratedPipelineOrchestrator(session_id=session_id)

# BTC 240m만 실행 (에피소드 20개로 줄임)
print(f'🚀 Starting BTC 240m selfplay (20 episodes)...')

# execute_coin_pipeline 대신 직접 selfplay 실행
from rl_pipeline.simulation.selfplay import run_self_play_test
from rl_pipeline.db.candle_data_loader import load_coin_interval_candles_with_regime

# 캔들 데이터 로드
candle_data, _ = load_coin_interval_candles_with_regime('BTC', '240m')
print(f'✅ Loaded {len(candle_data)} candles for BTC 240m')

# 전략 로드
strategies = load_strategies_pool(limit=4)
print(f'✅ Loaded {len(strategies)} strategies')

# Self-play 실행
result = run_self_play_test(
    strategy_params_list=strategies,
    episodes=20,  # 20개 에피소드만
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
else:
    print(f'\n❌ 테스트 실패: {result.get("error", "Unknown error")}')

print('\n✅ 테스트 완료!')
