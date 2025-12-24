"""
RL 에피소드 Pruning 유틸리티

전략당 최대 에피소드 수를 제한하여 DB 크기를 관리합니다.
- 전략당 최대 N개 에피소드만 유지 (기본 10,000개)
- 오래된 에피소드부터 삭제 (ts_entry ASC)
- rl_episodes와 rl_episode_summary를 동시에 정리하여 1:1 매핑 유지
- --dry-run 옵션으로 삭제 대상만 확인 가능

사용 예시:
    python rl_pipeline/tools/prune_rl_episodes.py --max-episodes-per-strategy 10000 --dry-run
    python rl_pipeline/tools/prune_rl_episodes.py --max-episodes-per-strategy 5000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.core.env import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_strategy_episode_counts() -> List[Tuple[str, str, str, int]]:
    """
    전략별 에피소드 수 집계

    Returns:
        List of (coin, interval, strategy_id, episode_count)
    """
    try:
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    coin,
                    interval,
                    strategy_id,
                    COUNT(*) as episode_count
                FROM rl_episodes
                GROUP BY coin, interval, strategy_id
                ORDER BY episode_count DESC
            """)

            results = cursor.fetchall()
            logger.info(f"📊 전략별 에피소드 집계 완료: {len(results)}개 전략")

            return results

    except Exception as e:
        logger.error(f"❌ 전략별 에피소드 집계 실패: {e}")
        return []


def prune_strategy_episodes(
    coin: str,
    interval: str,
    strategy_id: str,
    max_episodes: int,
    dry_run: bool = False
) -> int:
    """
    특정 전략의 오래된 에피소드 삭제

    Args:
        coin: 코인 심볼
        interval: 인터벌
        strategy_id: 전략 ID
        max_episodes: 최대 에피소드 수
        dry_run: True이면 삭제하지 않고 로그만 출력

    Returns:
        삭제된 에피소드 수
    """
    try:
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            # 현재 에피소드 수 확인
            cursor.execute("""
                SELECT COUNT(*) FROM rl_episodes
                WHERE symbol = ? AND interval = ? AND strategy_id = ?
            """, (coin, interval, strategy_id))

            current_count = cursor.fetchone()[0]

            if current_count <= max_episodes:
                return 0

            delete_count = current_count - max_episodes

            if dry_run:
                logger.info(
                    f"🔍 [DRY-RUN] {coin}-{interval} {strategy_id}: "
                    f"{delete_count}개 삭제 예정 ({current_count} → {max_episodes})"
                )
                return delete_count

            # 오래된 에피소드 ID 조회
            cursor.execute("""
                SELECT episode_id FROM rl_episodes
                WHERE symbol = ? AND interval = ? AND strategy_id = ?
                ORDER BY ts_entry ASC
                LIMIT ?
            """, (coin, interval, strategy_id, delete_count))

            episode_ids = [row[0] for row in cursor.fetchall()]

            if not episode_ids:
                return 0

            # rl_episodes 삭제
            placeholders = ','.join(['?' for _ in episode_ids])
            cursor.execute(
                f"DELETE FROM rl_episodes WHERE episode_id IN ({placeholders})",
                episode_ids
            )
            episodes_deleted = cursor.rowcount

            # rl_episode_summary 삭제 (1:1 매핑 유지)
            cursor.execute(
                f"DELETE FROM rl_episode_summary WHERE episode_id IN ({placeholders})",
                episode_ids
            )
            summary_deleted = cursor.rowcount

            conn.commit()

            logger.info(
                f"🗑️ {coin}-{interval} {strategy_id}: "
                f"{episodes_deleted}개 에피소드 삭제 ({current_count} → {max_episodes}), "
                f"요약 {summary_deleted}개 삭제"
            )

            return episodes_deleted

    except Exception as e:
        logger.error(f"❌ {coin}-{interval} {strategy_id} 에피소드 삭제 실패: {e}")
        return 0


def prune_all_strategies(max_episodes: int = 10000, dry_run: bool = False) -> Dict:
    """
    모든 전략의 에피소드 정리

    Args:
        max_episodes: 전략당 최대 에피소드 수
        dry_run: True이면 삭제하지 않고 로그만 출력

    Returns:
        통계 정보 딕셔너리
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 RL 에피소드 Pruning 시작")
    logger.info(f"{'='*80}")
    logger.info(f"📌 최대 에피소드 수: {max_episodes:,}개")
    logger.info(f"📌 모드: {'DRY-RUN (삭제하지 않음)' if dry_run else '실제 삭제'}")

    # 전략별 에피소드 수 집계
    strategy_counts = get_strategy_episode_counts()

    if not strategy_counts:
        logger.warning("⚠️ 에피소드가 없습니다")
        return {}

    # 통계
    total_strategies = len(strategy_counts)
    total_episodes_before = sum(count for _, _, _, count in strategy_counts)
    strategies_to_prune = [
        (coin, interval, sid, count)
        for coin, interval, sid, count in strategy_counts
        if count > max_episodes
    ]

    logger.info(f"\n📊 현재 상태:")
    logger.info(f"   총 전략 수: {total_strategies:,}개")
    logger.info(f"   총 에피소드 수: {total_episodes_before:,}개")
    logger.info(f"   정리 대상 전략: {len(strategies_to_prune):,}개")

    if not strategies_to_prune:
        logger.info("✅ 모든 전략이 최대 에피소드 수 이하입니다")
        return {
            'total_strategies': total_strategies,
            'total_episodes_before': total_episodes_before,
            'total_episodes_after': total_episodes_before,
            'strategies_pruned': 0,
            'episodes_deleted': 0
        }

    # 삭제 예정 에피소드 수 계산
    episodes_to_delete = sum(
        count - max_episodes
        for _, _, _, count in strategies_to_prune
    )

    logger.info(f"   삭제 예정 에피소드: {episodes_to_delete:,}개")
    logger.info(f"   예상 최종 에피소드 수: {total_episodes_before - episodes_to_delete:,}개")
    logger.info(f"   예상 감소율: {episodes_to_delete / total_episodes_before * 100:.1f}%")

    # 각 전략별로 정리
    logger.info(f"\n{'='*80}")
    logger.info("🔄 전략별 에피소드 정리 시작...")
    logger.info(f"{'='*80}")

    total_deleted = 0
    strategies_pruned = 0

    for coin, interval, strategy_id, count in strategies_to_prune:
        deleted = prune_strategy_episodes(
            coin, interval, strategy_id, max_episodes, dry_run
        )
        if deleted > 0:
            total_deleted += deleted
            strategies_pruned += 1

    total_episodes_after = total_episodes_before - total_deleted

    # 결과 요약
    logger.info(f"\n{'='*80}")
    logger.info("✅ RL 에피소드 Pruning 완료")
    logger.info(f"{'='*80}")
    logger.info(f"📊 결과 요약:")
    logger.info(f"   정리된 전략 수: {strategies_pruned:,}개")
    logger.info(f"   삭제된 에피소드: {total_deleted:,}개")
    logger.info(f"   에피소드 수 변화: {total_episodes_before:,} → {total_episodes_after:,}")
    logger.info(f"   감소율: {total_deleted / total_episodes_before * 100:.1f}%")

    if dry_run:
        logger.info(f"\n💡 실제 삭제하려면 --dry-run 옵션을 제거하세요")

    return {
        'total_strategies': total_strategies,
        'total_episodes_before': total_episodes_before,
        'total_episodes_after': total_episodes_after,
        'strategies_pruned': strategies_pruned,
        'episodes_deleted': total_deleted,
        'reduction_rate': total_deleted / total_episodes_before if total_episodes_before > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description='RL 에피소드 Pruning 유틸리티',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전략당 최대 10,000개 에피소드 유지 (dry-run)
  python prune_rl_episodes.py --max-episodes-per-strategy 10000 --dry-run

  # 전략당 최대 5,000개 에피소드 유지 (실제 삭제)
  python prune_rl_episodes.py --max-episodes-per-strategy 5000

  # 전략당 최대 20,000개 에피소드 유지
  python prune_rl_episodes.py --max-episodes-per-strategy 20000
        """
    )

    parser.add_argument(
        '--max-episodes-per-strategy',
        type=int,
        default=10000,
        help='전략당 최대 에피소드 수 (기본값: 10000)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='삭제하지 않고 로그만 출력'
    )

    args = parser.parse_args()

    # Pruning 실행
    results = prune_all_strategies(
        max_episodes=args.max_episodes_per_strategy,
        dry_run=args.dry_run
    )

    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
