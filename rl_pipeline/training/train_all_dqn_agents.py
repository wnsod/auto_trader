"""
실전용 멀티 코인/인터벌 DQN 학습 시스템
모든 코인-인터벌 조합에 대해 DQN 에이전트 학습
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple
import json

from rl_pipeline.training.train_rl_agent import train_dqn_agent, load_candle_data_from_db
from rl_pipeline.core.env import config

logger = logging.getLogger(__name__)


def get_available_coin_intervals() -> List[Tuple[str, str, int]]:
    """
    DB에서 학습 가능한 코인-인터벌 조합 조회

    Returns:
        [(coin, interval, candle_count), ...]
    """
    with sqlite3.connect(config.RL_DB) as conn:
        result = conn.execute("""
            SELECT coin, interval, COUNT(*) as cnt
            FROM candles
            GROUP BY coin, interval
            HAVING cnt >= 500  -- 최소 500개 캔들 필요
            ORDER BY coin, interval
        """).fetchall()

    return result


def save_training_results(
    coin: str,
    interval: str,
    results: Dict,
    model_path: str,
    training_time: float
):
    """
    학습 결과를 DB에 저장

    Args:
        coin: 코인
        interval: 인터벌
        results: train_dqn_agent 결과
        model_path: 모델 저장 경로
        training_time: 학습 시간 (초)
    """
    with sqlite3.connect(config.RL_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dqn_training_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                interval TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                num_episodes INTEGER,
                avg_reward REAL,
                avg_return REAL,
                avg_win_rate REAL,
                final_epsilon REAL,
                total_train_steps INTEGER,
                model_path TEXT,
                training_time REAL,
                UNIQUE(coin, interval, trained_at)
            )
        """)

        conn.execute("""
            INSERT OR REPLACE INTO dqn_training_results
            (coin, interval, trained_at, num_episodes, avg_reward, avg_return,
             avg_win_rate, final_epsilon, total_train_steps, model_path, training_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            coin,
            interval,
            datetime.now().isoformat(),
            len(results['episode_rewards']),
            float(sum(results['episode_rewards']) / len(results['episode_rewards'])),
            float(sum(results['episode_returns']) / len(results['episode_returns'])),
            float(sum(results['episode_win_rates']) / len(results['episode_win_rates'])),
            float(results['final_epsilon']),
            int(results['total_train_steps']),
            model_path,
            training_time
        ))

        conn.commit()

    logger.info(f"✅ 학습 결과 DB 저장 완료: {coin}-{interval}")


def train_single_agent(
    coin: str,
    interval: str,
    num_episodes: int = 100,
    candle_limit: int = None
) -> Dict:
    """
    단일 코인-인터벌 조합에 대해 DQN 학습

    Args:
        coin: 코인 (예: "BTC")
        interval: 인터벌 (예: "15m")
        num_episodes: 학습 에피소드 수
        candle_limit: 사용할 캔들 최대 개수 (None=전체)

    Returns:
        학습 결과
    """
    import time
    start_time = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 DQN 학습 시작: {coin}-{interval}")
    logger.info(f"{'='*60}")

    # 1. 데이터 로드
    candle_data = load_candle_data_from_db(coin, interval, limit=candle_limit)
    logger.info(f"   캔들 데이터: {len(candle_data)}개")

    # 2. 학습 실행
    model_path = f"models/dqn_{coin.lower()}_{interval}.pkl"

    results = train_dqn_agent(
        candle_data=candle_data,
        num_episodes=num_episodes,
        save_path=model_path,
        log_interval=max(10, num_episodes // 10)
    )

    training_time = time.time() - start_time

    # 3. 결과 저장
    save_training_results(coin, interval, results, model_path, training_time)

    # 4. 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ {coin}-{interval} 학습 완료!")
    logger.info(f"{'='*60}")
    logger.info(f"   평균 보상: {sum(results['episode_rewards'])/len(results['episode_rewards']):.2f}")
    logger.info(f"   평균 수익률: {sum(results['episode_returns'])/len(results['episode_returns']):.2%}")
    logger.info(f"   평균 승률: {sum(results['episode_win_rates'])/len(results['episode_win_rates']):.2%}")
    logger.info(f"   최종 Epsilon: {results['final_epsilon']:.4f}")
    logger.info(f"   학습 스텝: {results['total_train_steps']}")
    logger.info(f"   학습 시간: {training_time:.1f}초")
    logger.info(f"   모델 경로: {model_path}")

    return results


def train_all_agents(
    coins: List[str] = None,
    intervals: List[str] = None,
    num_episodes: int = 100,
    candle_limit: int = None
):
    """
    모든 코인-인터벌 조합에 대해 DQN 학습

    Args:
        coins: 학습할 코인 목록 (None=전체)
        intervals: 학습할 인터벌 목록 (None=전체)
        num_episodes: 에피소드 수
        candle_limit: 캔들 제한
    """
    import time
    total_start = time.time()

    # 사용 가능한 조합 조회
    available = get_available_coin_intervals()

    # 필터링
    if coins:
        available = [(c, i, cnt) for c, i, cnt in available if c in coins]
    if intervals:
        available = [(c, i, cnt) for c, i, cnt in available if i in intervals]

    logger.info(f"\n{'#'*60}")
    logger.info(f"# 실전용 DQN 멀티 에이전트 학습 시작")
    logger.info(f"{'#'*60}")
    logger.info(f"학습 대상: {len(available)}개 조합")
    logger.info(f"에피소드: {num_episodes}")
    logger.info(f"캔들 제한: {candle_limit if candle_limit else '전체'}\n")

    results_summary = []

    for idx, (coin, interval, cnt) in enumerate(available, 1):
        logger.info(f"\n[{idx}/{len(available)}] {coin}-{interval} (캔들: {cnt:,}개)")

        try:
            results = train_single_agent(
                coin=coin,
                interval=interval,
                num_episodes=num_episodes,
                candle_limit=candle_limit
            )

            results_summary.append({
                'coin': coin,
                'interval': interval,
                'success': True,
                'avg_return': sum(results['episode_returns'])/len(results['episode_returns']),
                'avg_win_rate': sum(results['episode_win_rates'])/len(results['episode_win_rates'])
            })

        except Exception as e:
            logger.error(f"❌ {coin}-{interval} 학습 실패: {e}")
            results_summary.append({
                'coin': coin,
                'interval': interval,
                'success': False,
                'error': str(e)
            })

    # 전체 결과 요약
    total_time = time.time() - total_start

    logger.info(f"\n{'#'*60}")
    logger.info(f"# 전체 학습 완료!")
    logger.info(f"{'#'*60}")
    logger.info(f"총 학습 시간: {total_time/60:.1f}분")
    logger.info(f"성공: {sum(1 for r in results_summary if r['success'])}/{len(results_summary)}")

    # 성능 순위
    logger.info(f"\n=== 성능 순위 (평균 수익률) ===")
    successful = [r for r in results_summary if r['success']]
    successful.sort(key=lambda x: x['avg_return'], reverse=True)

    for idx, r in enumerate(successful[:10], 1):
        logger.info(f"{idx:2d}. {r['coin']:5s}-{r['interval']:4s}: "
                   f"수익률 {r['avg_return']:+.2%}, 승률 {r['avg_win_rate']:.2%}")

    # 결과 JSON 저장
    with open('models/dqn_training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\n✅ 요약 저장: models/dqn_training_summary.json")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='DQN 멀티 에이전트 학습 (DB 기반)')
    parser.add_argument('--coins', nargs='+', help='학습할 코인 필터 (선택사항, 미지정시 DB의 모든 코인)')
    parser.add_argument('--intervals', nargs='+', help='학습할 인터벌 필터 (선택사항, 미지정시 모든 인터벌)')
    parser.add_argument('--episodes', type=int, default=100, help='에피소드 수 (기본: 100)')
    parser.add_argument('--candle-limit', type=int, help='캔들 최대 개수 (빠른 테스트시 사용, 예: 500)')

    args = parser.parse_args()

    # DB에 있는 코인-인터벌 조합으로 자동 학습
    train_all_agents(
        coins=args.coins,
        intervals=args.intervals,
        num_episodes=args.episodes,
        candle_limit=args.candle_limit
    )
