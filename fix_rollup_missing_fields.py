"""
롤업 누락 필드 계산 및 업데이트
- avg_sharpe_ratio
- avg_profit_factor
- total_profit
- avg_reward
- best/worst_episode_reward
"""
import sys
sys.path.insert(0, '/workspace')

import sqlite3
import logging
from datetime import datetime
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def calculate_sharpe_ratio(returns):
    """Sharpe Ratio 계산"""
    if not returns or len(returns) < 2:
        return 0.0

    avg_return = sum(returns) / len(returns)

    # 표준편차 계산
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0:
        return 0.0

    # Sharpe Ratio (무위험 수익률 = 0 가정)
    return avg_return / std_dev


def calculate_profit_factor(returns):
    """Profit Factor 계산 (총 이익 / 총 손실)"""
    if not returns:
        return 0.0

    total_profit = sum(r for r in returns if r > 0)
    total_loss = abs(sum(r for r in returns if r < 0))

    if total_loss == 0:
        return float('inf') if total_profit > 0 else 0.0

    return total_profit / total_loss


def update_rollup_missing_fields():
    """롤업 테이블의 누락된 필드 계산 및 업데이트"""

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        logger.info("🔄 롤업 누락 필드 계산 시작...")

        # 모든 전략 조회
        cursor.execute("""
            SELECT strategy_id, coin, interval
            FROM rl_strategy_rollup
        """)
        strategies = cursor.fetchall()

        logger.info(f"📊 처리할 전략: {len(strategies)}개")

        updated_count = 0

        for strategy_id, coin, interval in strategies:
            try:
                # 해당 전략의 최근 에피소드 데이터 조회 (최근 30일)
                cutoff_ts = int(datetime.now().timestamp() - (30 * 86400))

                cursor.execute("""
                    SELECT
                        realized_ret_signed,
                        total_reward
                    FROM rl_episode_summary
                    WHERE strategy_id = ?
                      AND coin = ?
                      AND interval = ?
                      AND ts_exit >= ?
                """, (strategy_id, coin, interval, cutoff_ts))

                episodes = cursor.fetchall()

                if not episodes:
                    continue

                # 데이터 추출
                returns = [e[0] for e in episodes if e[0] is not None]
                rewards = [e[1] for e in episodes if e[1] is not None]

                # 계산
                sharpe_ratio = calculate_sharpe_ratio(returns) if returns else 0.0
                profit_factor = calculate_profit_factor(returns) if returns else 0.0
                total_profit = sum(returns) if returns else 0.0
                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                best_reward = max(rewards) if rewards else 0.0
                worst_reward = min(rewards) if rewards else 0.0
                total_episodes = len(episodes)

                # 업데이트
                cursor.execute("""
                    UPDATE rl_strategy_rollup
                    SET avg_sharpe_ratio = ?,
                        avg_profit_factor = ?,
                        total_profit = ?,
                        avg_reward = ?,
                        best_episode_reward = ?,
                        worst_episode_reward = ?,
                        total_episodes = ?,
                        last_updated = ?
                    WHERE strategy_id = ?
                      AND coin = ?
                      AND interval = ?
                """, (
                    sharpe_ratio,
                    profit_factor,
                    total_profit,
                    avg_reward,
                    best_reward,
                    worst_reward,
                    total_episodes,
                    datetime.now().isoformat(),
                    strategy_id,
                    coin,
                    interval
                ))

                updated_count += 1

                if updated_count % 100 == 0:
                    logger.info(f"⏳ 진행 중: {updated_count}/{len(strategies)}")
                    conn.commit()

            except Exception as e:
                logger.error(f"❌ {strategy_id} 업데이트 실패: {e}")
                continue

        conn.commit()

        logger.info(f"✅ 롤업 필드 업데이트 완료: {updated_count}개 전략")

        # 결과 확인
        cursor.execute("""
            SELECT
                AVG(avg_sharpe_ratio) as avg_sharpe,
                AVG(avg_profit_factor) as avg_pf,
                COUNT(*) as total,
                SUM(CASE WHEN avg_sharpe_ratio != 0 THEN 1 ELSE 0 END) as non_zero_sharpe
            FROM rl_strategy_rollup
        """)

        result = cursor.fetchone()
        logger.info(f"\n📊 업데이트 결과:")
        logger.info(f"   - 총 전략 수: {result[2]}")
        logger.info(f"   - Sharpe Ratio != 0인 전략: {result[3]}")
        logger.info(f"   - 평균 Sharpe Ratio: {result[0]:.4f}")
        logger.info(f"   - 평균 Profit Factor: {result[1]:.4f}")

        conn.close()

        return updated_count

    except Exception as e:
        logger.error(f"❌ 롤업 필드 업데이트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


if __name__ == "__main__":
    updated = update_rollup_missing_fields()
    logger.info(f"\n✅ 총 {updated}개 전략 업데이트 완료")
