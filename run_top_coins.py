"""
상위 주요 코인에 대해 Absolute Zero System 실행
"""
import sys
import os
import time

# 경로 추가
sys.path.insert(0, '/workspace/rl_pipeline')

# 개선된 Absolute Zero 시스템 import
from absolute_zero_improved import run_absolute_zero, _configure_logging
import logging

def run_for_top_coins(coin_limit=5):
    """상위 N개 코인에 대해 시스템 실행"""

    # 로깅 설정
    _configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info(f"🚀 상위 {coin_limit}개 코인 대상 Absolute Zero System 실행")
    logger.info("="*60)

    # 주요 코인 리스트 (시가총액 기준)
    top_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK']

    # 캔들 데이터가 있는 코인 확인
    from rl_pipeline.data.candle_loader import get_available_coins_and_intervals

    try:
        available = get_available_coins_and_intervals()
        available_coins = set(c for c, _ in available)

        # 실제로 사용 가능한 상위 코인들
        coins_to_run = []
        for coin in top_coins:
            if coin in available_coins:
                coins_to_run.append(coin)
                if len(coins_to_run) >= coin_limit:
                    break

        logger.info(f"📊 실행할 코인: {', '.join(coins_to_run)}")

        # 실행할 인터벌 설정 (간략 테스트를 위해 2개만)
        intervals = ['15m', '240m']  # 단기와 장기 각 1개

        # 실행 결과 추적
        results = {}
        success_count = 0

        for idx, coin in enumerate(coins_to_run, 1):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🪙 [{idx}/{len(coins_to_run)}] {coin} 처리 시작...")
                logger.info(f"   인터벌: {', '.join(intervals)}")
                logger.info(f"{'='*60}")

                start_time = time.time()

                # 코인별 실행
                result = run_absolute_zero(
                    coin=coin,
                    intervals=intervals,
                    n_strategies=200  # 개선된 전략 수
                )

                elapsed_time = time.time() - start_time

                if result and not result.get('error'):
                    success_count += 1
                    results[coin] = {
                        'status': 'success',
                        'time': elapsed_time,
                        'strategies': result.get('total_strategies', 0)
                    }
                    logger.info(f"✅ {coin} 처리 완료 (소요시간: {elapsed_time:.1f}초)")
                else:
                    results[coin] = {
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error'),
                        'time': elapsed_time
                    }
                    logger.warning(f"⚠️ {coin} 처리 실패: {result.get('error', 'Unknown error')}")

                # 코인 간 짧은 대기
                if idx < len(coins_to_run):
                    logger.info(f"   다음 코인 처리까지 3초 대기...")
                    time.sleep(3)

            except Exception as e:
                logger.error(f"❌ {coin} 처리 중 오류: {e}")
                results[coin] = {
                    'status': 'error',
                    'error': str(e)
                }
                continue

        # 최종 결과 요약
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 실행 결과 요약")
        logger.info(f"{'='*60}")

        for coin, result in results.items():
            status_icon = '✅' if result['status'] == 'success' else '❌'
            logger.info(f"{status_icon} {coin}: {result['status']}")
            if 'time' in result:
                logger.info(f"   소요시간: {result['time']:.1f}초")
            if 'strategies' in result:
                logger.info(f"   생성 전략: {result['strategies']}개")
            if 'error' in result:
                logger.info(f"   오류: {result['error']}")

        logger.info(f"\n📈 전체 통계:")
        logger.info(f"   성공: {success_count}/{len(coins_to_run)} 코인")
        logger.info(f"   실패: {len(coins_to_run) - success_count} 코인")

        total_time = sum(r.get('time', 0) for r in results.values())
        logger.info(f"   총 소요시간: {total_time:.1f}초")

        return results

    except Exception as e:
        logger.error(f"❌ 실행 실패: {e}")
        return {}

if __name__ == "__main__":
    results = run_for_top_coins(coin_limit=5)

    # 모든 코인이 성공했는지 확인
    all_success = all(r['status'] == 'success' for r in results.values())
    sys.exit(0 if all_success else 1)