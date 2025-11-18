"""
모든 코인에 대해 Absolute Zero System 실행
"""
import sys
import os
import time

# 경로 추가
sys.path.insert(0, '/workspace/rl_pipeline')

# 개선된 Absolute Zero 시스템 import
from absolute_zero_improved import run_absolute_zero, _configure_logging
import logging

def run_for_all_coins():
    """모든 사용 가능한 코인에 대해 시스템 실행"""

    # 로깅 설정
    _configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("🚀 전체 코인 대상 Absolute Zero System 실행")
    logger.info("="*60)

    # 캔들 데이터가 있는 코인 목록 가져오기
    from rl_pipeline.data.candle_loader import get_available_coins_and_intervals

    try:
        available = get_available_coins_and_intervals()
        coins = sorted(list(set(c for c, _ in available)))

        logger.info(f"📊 발견된 코인: {len(coins)}개")
        logger.info(f"   코인 목록: {', '.join(coins[:10])}...")

        # 실행할 인터벌 설정
        intervals = ['15m', '30m', '240m', '1d']

        # 각 코인에 대해 실행
        success_count = 0
        failed_coins = []

        for idx, coin in enumerate(coins, 1):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🪙 [{idx}/{len(coins)}] {coin} 처리 시작...")
                logger.info(f"{'='*60}")

                # 코인별 실행
                result = run_absolute_zero(
                    coin=coin,
                    intervals=intervals,
                    n_strategies=200  # 개선된 전략 수 사용
                )

                if result and not result.get('error'):
                    success_count += 1
                    logger.info(f"✅ {coin} 처리 완료")
                else:
                    failed_coins.append(coin)
                    logger.warning(f"⚠️ {coin} 처리 실패: {result.get('error', 'Unknown error')}")

                # 코인 간 짧은 대기 (시스템 부하 방지)
                if idx < len(coins):
                    time.sleep(2)

            except Exception as e:
                logger.error(f"❌ {coin} 처리 중 오류: {e}")
                failed_coins.append(coin)
                continue

        # 최종 결과 요약
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 전체 실행 완료")
        logger.info(f"{'='*60}")
        logger.info(f"   성공: {success_count}/{len(coins)} 코인")
        logger.info(f"   실패: {len(failed_coins)} 코인")

        if failed_coins:
            logger.info(f"   실패한 코인: {', '.join(failed_coins)}")

        return success_count, failed_coins

    except Exception as e:
        logger.error(f"❌ 전체 실행 실패: {e}")
        return 0, []

if __name__ == "__main__":
    success, failed = run_for_all_coins()

    # 종료 코드 설정 (모두 성공 시 0, 일부 실패 시 1)
    sys.exit(0 if len(failed) == 0 else 1)