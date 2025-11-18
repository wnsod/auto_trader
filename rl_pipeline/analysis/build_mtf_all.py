"""
모든 코인 및 인터벌에 대해 MTF 컨텍스트 생성
"""
import sys
import os
import logging
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.analysis.build_mtf_context import MTFContextBuilder
from rl_pipeline.db.connection_pool import get_strategy_db_pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_coins() -> List[str]:
    """라벨링된 모든 코인 조회"""
    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT coin FROM strategy_signal_labels ORDER BY coin")
        return [row[0] for row in cursor.fetchall()]

def main():
    """모든 코인/인터벌에 대해 MTF 컨텍스트 생성"""
    logger.info("🚀 전체 코인 MTF 컨텍스트 생성 시작\n")

    # 코인 목록
    coins = get_all_coins()
    logger.info(f"📊 대상 코인: {coins}")

    # Base intervals (15m, 30m)
    base_intervals = ['15m', '30m']

    # HTF intervals (240m, 1d)
    htf_intervals = ['240m', '1d']

    logger.info(f"📊 Base intervals: {base_intervals}")
    logger.info(f"📊 HTF intervals: {htf_intervals}\n")

    # 빌더 초기화
    builder = MTFContextBuilder()

    # 전체 통계
    total_processed = 0
    total_saved = 0

    # 각 코인별로 처리
    for coin in coins:
        logger.info(f"\n{'='*80}")
        logger.info(f"🪙 {coin} 처리 시작")
        logger.info(f"{'='*80}")

        for base_interval in base_intervals:
            result = builder.build_and_save_for_coin_interval(
                coin=coin,
                base_interval=base_interval,
                htf_intervals=htf_intervals,
                limit=None  # 전체 처리
            )

            total_processed += result['processed']
            total_saved += result['saved']

            logger.info(f"  ✅ {coin} {base_interval}: "
                       f"{result['processed']}개 처리, {result['saved']}개 저장")

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 전체 결과")
    logger.info(f"{'='*80}")
    logger.info(f"  처리: {total_processed:,}개")
    logger.info(f"  저장: {total_saved:,}개")

    # MTF 통계 갱신
    logger.info(f"\n{'='*80}")
    logger.info("📊 MTF 통계 갱신 중...")
    logger.info(f"{'='*80}")
    builder.update_mtf_stats()

    logger.info(f"\n🎉 전체 MTF 컨텍스트 생성 완료!")

if __name__ == "__main__":
    main()
