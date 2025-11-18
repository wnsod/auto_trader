"""
상세 데이터 품질 분석
"""
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def analyze_strategy_performance():
    """전략 성능 상세 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("📊 RL Strategy Rollup 성능 분석")
    logger.info("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Grade별 성능 통계
    cursor.execute("""
        SELECT
            grade,
            COUNT(*) as cnt,
            AVG(avg_ret) as avg_return,
            AVG(win_rate) as avg_win_rate,
            AVG(predictive_accuracy) as avg_accuracy,
            AVG(avg_sharpe_ratio) as avg_sharpe
        FROM rl_strategy_rollup
        GROUP BY grade
        ORDER BY
            CASE grade
                WHEN 'S' THEN 1
                WHEN 'A' THEN 2
                WHEN 'B' THEN 3
                WHEN 'C' THEN 4
                WHEN 'D' THEN 5
                WHEN 'F' THEN 6
                ELSE 7
            END
    """)

    logger.info("\n📈 Grade별 평균 성능:")
    logger.info(f"  {'Grade':<8}{'Count':>8}{'Avg Return':>15}{'Win Rate':>12}{'Accuracy':>12}{'Sharpe':>12}")
    logger.info("  " + "-" * 75)

    for row in cursor.fetchall():
        grade, cnt, avg_ret, win_rate, accuracy, sharpe = row
        logger.info(
            f"  {grade:<8}{cnt:>8,}{avg_ret:>14.4f}%{win_rate:>11.2f}%{accuracy:>11.2f}%{sharpe or 0:>11.4f}"
        )

    # 코인별 최고 성능 전략
    logger.info("\n\n💎 코인별 최고 성능 전략 (Top 3):")
    cursor.execute("""
        SELECT coin, interval, grade, avg_ret, win_rate, predictive_accuracy
        FROM rl_strategy_rollup
        WHERE grade IN ('S', 'A')
        ORDER BY avg_ret DESC
        LIMIT 10
    """)

    logger.info(f"  {'Coin':<8}{'Interval':<10}{'Grade':<8}{'Return':>12}{'Win Rate':>12}{'Accuracy':>12}")
    logger.info("  " + "-" * 70)

    for row in cursor.fetchall():
        coin, interval, grade, ret, win_rate, acc = row
        logger.info(
            f"  {coin:<8}{interval:<10}{grade:<8}{ret:>11.4f}%{win_rate:>11.2f}%{acc:>11.2f}%"
        )

    conn.close()


def analyze_paper_trading():
    """Paper Trading 상세 분석"""
    logger.info("\n\n" + "=" * 80)
    logger.info("📊 Paper Trading 상세 분석")
    logger.info("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 세션별 수익률 분석
    cursor.execute("""
        SELECT
            coin,
            interval,
            initial_capital,
            current_capital,
            (current_capital - initial_capital) / initial_capital * 100 as profit_pct,
            status,
            start_time
        FROM paper_trading_sessions
        ORDER BY profit_pct DESC
        LIMIT 10
    """)

    logger.info("\n📈 상위 10개 Paper Trading 세션:")
    logger.info(f"  {'Coin':<8}{'Interval':<10}{'Initial':>12}{'Current':>12}{'Profit %':>12}{'Status':<10}")
    logger.info("  " + "-" * 75)

    for row in cursor.fetchall():
        coin, interval, initial, current, profit, status, start_time = row
        logger.info(
            f"  {coin:<8}{interval:<10}{initial:>12,.2f}{current:>12,.2f}{profit:>11.2f}%{status:<10}"
        )

    # 하위 10개 세션
    cursor.execute("""
        SELECT
            coin,
            interval,
            initial_capital,
            current_capital,
            (current_capital - initial_capital) / initial_capital * 100 as profit_pct,
            status
        FROM paper_trading_sessions
        ORDER BY profit_pct ASC
        LIMIT 10
    """)

    logger.info("\n📉 하위 10개 Paper Trading 세션:")
    logger.info(f"  {'Coin':<8}{'Interval':<10}{'Initial':>12}{'Current':>12}{'Profit %':>12}{'Status':<10}")
    logger.info("  " + "-" * 75)

    for row in cursor.fetchall():
        coin, interval, initial, current, profit, status = row
        logger.info(
            f"  {coin:<8}{interval:<10}{initial:>12,.2f}{current:>12,.2f}{profit:>11.2f}%{status:<10}"
        )

    # 코인별 평균 수익률
    cursor.execute("""
        SELECT
            coin,
            COUNT(*) as session_count,
            AVG((current_capital - initial_capital) / initial_capital * 100) as avg_profit,
            MIN((current_capital - initial_capital) / initial_capital * 100) as min_profit,
            MAX((current_capital - initial_capital) / initial_capital * 100) as max_profit
        FROM paper_trading_sessions
        WHERE initial_capital > 0
        GROUP BY coin
        ORDER BY avg_profit DESC
    """)

    logger.info("\n\n📊 코인별 Paper Trading 성과:")
    logger.info(f"  {'Coin':<8}{'Sessions':>10}{'Avg Profit':>15}{'Min':>12}{'Max':>12}")
    logger.info("  " + "-" * 65)

    for row in cursor.fetchall():
        coin, cnt, avg_profit, min_profit, max_profit = row
        logger.info(
            f"  {coin:<8}{cnt:>10}{avg_profit:>14.2f}%{min_profit:>11.2f}%{max_profit:>11.2f}%"
        )

    conn.close()


def analyze_episodes_quality():
    """에피소드 데이터 품질 분석"""
    logger.info("\n\n" + "=" * 80)
    logger.info("📊 RL Episodes 데이터 품질 분석")
    logger.info("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 에피소드 요약 통계
    cursor.execute("""
        SELECT
            coin,
            interval,
            COUNT(*) as episode_count,
            AVG(total_reward) as avg_reward,
            AVG(realized_ret_signed) as avg_return,
            SUM(CASE WHEN acc_flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy_pct
        FROM rl_episode_summary
        GROUP BY coin, interval
        ORDER BY avg_reward DESC
        LIMIT 15
    """)

    logger.info("\n📈 코인/인터벌별 에피소드 성능 (Top 15):")
    logger.info(f"  {'Coin':<8}{'Interval':<10}{'Episodes':>10}{'Avg Reward':>15}{'Avg Return':>15}{'Accuracy':>12}")
    logger.info("  " + "-" * 85)

    for row in cursor.fetchall():
        coin, interval, cnt, reward, ret, acc = row
        logger.info(
            f"  {coin:<8}{interval:<10}{cnt:>10,}{reward:>14.4f}{ret:>14.4f}%{acc:>11.2f}%"
        )

    # 이상치 탐지 - 비정상적으로 높거나 낮은 보상
    cursor.execute("""
        SELECT
            episode_id,
            coin,
            interval,
            total_reward,
            realized_ret_signed,
            first_event
        FROM rl_episode_summary
        WHERE total_reward < -1.0 OR total_reward > 2.0
        ORDER BY total_reward DESC
        LIMIT 10
    """)

    anomalies = cursor.fetchall()
    if anomalies:
        logger.info("\n\n⚠️  이상 보상값 탐지 (reward < -1.0 or > 2.0):")
        logger.info(f"  {'Episode ID':<30}{'Coin':<8}{'Interval':<10}{'Reward':>12}{'Return':>12}{'Event':<10}")
        logger.info("  " + "-" * 95)

        for row in anomalies:
            ep_id, coin, interval, reward, ret, event = row
            logger.info(
                f"  {ep_id[:28]:<30}{coin:<8}{interval:<10}{reward:>12.4f}{ret:>11.4f}%{event:<10}"
            )
    else:
        logger.info("\n\n✅ 이상 보상값 없음 (모든 보상값이 정상 범위 내)")

    conn.close()


if __name__ == "__main__":
    try:
        analyze_strategy_performance()
        analyze_paper_trading()
        analyze_episodes_quality()

        logger.info("\n\n" + "=" * 80)
        logger.info("✅ 상세 데이터 품질 분석 완료")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 분석 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
