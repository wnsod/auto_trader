"""
예측 Self-play 수정 테스트
LINK-15m 코인 하나만 짧게 실행
"""
import sys
import os
sys.path.insert(0, '/workspace')

import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 환경변수 설정 (짧은 테스트)
os.environ['PREDICTIVE_SELFPLAY_EPISODES'] = '5'  # 5개 에피소드만
os.environ['ENABLE_PREDICTIVE_SELFPLAY'] = 'true'
os.environ['AZ_DEBUG'] = 'false'

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator
from rl_pipeline.data.candle_loader import load_candle_data_for_coin

def main():
    coin = 'LINK'
    interval = '15m'

    logger.info(f"🧪 테스트 시작: {coin}-{interval}")
    logger.info(f"📊 예측 Self-play 에피소드: 5개")

    # 캔들 데이터 로드
    logger.info(f"📊 캔들 데이터 로드 중...")
    all_candle_data = load_candle_data_for_coin(coin, [interval])

    if not all_candle_data:
        logger.error("❌ 캔들 데이터 로드 실패")
        return

    candle_data = all_candle_data.get((coin, interval))
    if candle_data is None or candle_data.empty:
        logger.error("❌ 캔들 데이터 없음")
        return

    logger.info(f"✅ 캔들 데이터 로드 완료: {len(candle_data)}개")
    logger.info(f"   최신 종가: {candle_data['close'].iloc[-1]:,.0f}원")

    # 오케스트레이터 초기화
    orchestrator = IntegratedPipelineOrchestrator(session_id=None)

    # 파이프라인 실행
    logger.info(f"🔄 파이프라인 실행 중...")
    result = orchestrator.run_partial_pipeline(coin, interval, candle_data)

    logger.info(f"✅ 파이프라인 완료")
    logger.info(f"   전략 수: {result.strategies_created}개")
    logger.info(f"   에피소드: {result.selfplay_episodes}개")
    logger.info(f"   상태: {result.status}")

    # 결과 확인
    logger.info(f"\n📊 결과 확인 중...")

    import sqlite3
    conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
    cursor = conn.cursor()

    # 에피소드 통계
    cursor.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN s.first_event = 'TP' THEN 1 ELSE 0 END) as tp_count,
            SUM(CASE WHEN s.first_event = 'SL' THEN 1 ELSE 0 END) as sl_count,
            SUM(CASE WHEN s.first_event = 'expiry' THEN 1 ELSE 0 END) as expiry_count,
            AVG(s.realized_ret_signed) as avg_return,
            MIN(s.realized_ret_signed) as min_return,
            MAX(s.realized_ret_signed) as max_return
        FROM rl_episodes e
        JOIN rl_episode_summary s ON e.episode_id = s.episode_id
        WHERE e.coin = ? AND e.interval = ?
    ''', (coin, interval))

    row = cursor.fetchone()
    total, tp, sl, expiry, avg_ret, min_ret, max_ret = row

    print("\n" + "=" * 80)
    print(f"📊 {coin}-{interval} 에피소드 결과")
    print("=" * 80)
    print(f"총 에피소드: {total}개")
    print(f"  TP 도달: {tp}개 ({tp/total*100 if total > 0 else 0:.1f}%)")
    print(f"  SL 도달: {sl}개 ({sl/total*100 if total > 0 else 0:.1f}%)")
    print(f"  만료: {expiry}개 ({expiry/total*100 if total > 0 else 0:.1f}%)")
    print(f"\n수익률:")
    print(f"  평균: {avg_ret*100 if avg_ret else 0:.4f}%")
    print(f"  최소: {min_ret*100 if min_ret else 0:.4f}%")
    print(f"  최대: {max_ret*100 if max_ret else 0:.4f}%")

    # 샘플 5개 조회
    cursor.execute('''
        SELECT
            e.strategy_id,
            e.predicted_dir,
            e.entry_price,
            s.first_event,
            s.realized_ret_signed
        FROM rl_episodes e
        JOIN rl_episode_summary s ON e.episode_id = s.episode_id
        WHERE e.coin = ? AND e.interval = ?
        ORDER BY e.ts_entry DESC
        LIMIT 5
    ''', (coin, interval))

    print(f"\n샘플 에피소드 (최근 5개):")
    print("-" * 80)
    for sid, pred_dir, entry_price, event, ret in cursor.fetchall():
        dir_str = "BUY" if pred_dir == 1 else ("SELL" if pred_dir == -1 else "HOLD")
        print(f"{sid[:30]:30s} | {dir_str:4s} | {entry_price:12,.0f}원 | {event:6s} | {ret*100:7.4f}%")

    conn.close()

    print("=" * 80)
    logger.info("✅ 테스트 완료")

if __name__ == "__main__":
    main()
