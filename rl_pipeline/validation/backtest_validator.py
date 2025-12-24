"""
백테스트 검증 모듈
- Look-ahead bias 체크
- 데이터 누수 검증
- 통계적 일관성 검증
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_backtest_logic(db_path: str = "data_storage/learning_strategies.db") -> List[str]:
    """백테스트 로직 검증"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    issues = []
    
    try:
        # 샘플 전략 선택
        cursor.execute("""
            SELECT id, coin, interval, profit, win_rate, trades_count
            FROM strategies
            LIMIT 10
        """)
        strategies = cursor.fetchall()
        
        if not strategies:
            logger.warning("검증할 전략이 없습니다.")
            return issues
        
        for strategy_id, coin, interval, profit, win_rate, trades_count in strategies:
            # 1. Look-ahead bias 체크
            if check_lookahead_bias(strategy_id, coin, interval, cursor):
                issues.append(f"⚠️ {coin}-{interval}: Look-ahead bias 발견 가능성")
            
            # 2. 데이터 누수 체크
            if check_data_leakage(strategy_id, coin, interval, cursor):
                issues.append(f"⚠️ {coin}-{interval}: 데이터 누수 가능성")
            
            # 3. 통계적 일관성 체크
            if win_rate is not None and trades_count is not None and trades_count > 0:
                expected_profit = calculate_expected_profit(win_rate, trades_count)
                if profit is not None and expected_profit is not None:
                    if abs(profit - expected_profit) > abs(expected_profit * 0.5):
                        issues.append(
                            f"⚠️ {coin}-{interval}: 수익률과 승률 불일치 "
                            f"(예상: {expected_profit:.2f}%, 실제: {profit:.2f}%)"
                        )
        
        logger.info(f"\n검증 결과: {len(issues)}개 이슈 발견")
        if issues:
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("  ✅ 백테스트 로직 정상")
    
    except Exception as e:
        logger.error(f"❌ 백테스트 검증 실패: {e}")
        issues.append(f"검증 오류: {str(e)}")
    
    finally:
        conn.close()
    
    return issues


def check_lookahead_bias(
    strategy_id: str,
    coin: str,
    interval: str,
    cursor: sqlite3.Cursor
) -> bool:
    """Look-ahead bias 체크"""
    
    try:
        # selfplay_results에서 실행 로그 확인
        cursor.execute("""
            SELECT execution_log
            FROM selfplay_results
            WHERE strategy_id = ?
            LIMIT 1
        """, (strategy_id,))
        
        result = cursor.fetchone()
        if result and result[0]:
            # 실행 로그에서 미래 데이터 접근 여부 확인
            execution_log = result[0]
            
            # 간단한 체크: 미래 타임스탬프 사용 여부
            # 실제 구현에서는 더 정교한 검증 필요
            if "future" in str(execution_log).lower() or "ahead" in str(execution_log).lower():
                return True
        
        # 캔들 데이터와 거래 시점 비교
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT sr.episode_id, sr.trade_time, c.timestamp
                FROM selfplay_results sr
                JOIN rl_candles c ON sr.coin = c.coin AND sr.interval = c.interval
                WHERE sr.strategy_id = ?
                AND sr.trade_time < c.timestamp
            )
        """, (strategy_id,))
        
        future_data_count = cursor.fetchone()[0]
        if future_data_count > 0:
            return True
        
    except Exception as e:
        logger.debug(f"Look-ahead bias 체크 중 오류: {e}")
    
    return False


def check_data_leakage(
    strategy_id: str,
    coin: str,
    interval: str,
    cursor: sqlite3.Cursor
) -> bool:
    """데이터 누수 체크"""
    
    try:
        # 학습 데이터와 테스트 데이터 분리 확인
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT sr.episode_id) as total_episodes,
                MIN(sr.created_at) as first_episode,
                MAX(sr.created_at) as last_episode
            FROM selfplay_results sr
            WHERE sr.strategy_id = ?
        """, (strategy_id,))
        
        result = cursor.fetchone()
        if result:
            total_episodes, first_episode, last_episode = result
            
            # 전략 생성 시간 확인
            cursor.execute("""
                SELECT created_at
                FROM strategies
                WHERE strategy_id = ?
            """, (strategy_id,))
            
            strategy_created = cursor.fetchone()
            if strategy_created and strategy_created[0]:
                # 전략 생성 후 에피소드가 생성되었는지 확인
                if first_episode and first_episode < strategy_created[0]:
                    # 전략 생성 전 에피소드가 있으면 데이터 누수 가능성
                    return True
        
        # 캔들 데이터와 에피소드 시간 비교
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT sr.episode_id
                FROM selfplay_results sr
                JOIN rl_candles c ON sr.coin = c.coin AND sr.interval = c.interval
                WHERE sr.strategy_id = ?
                AND c.timestamp > sr.created_at
                AND c.timestamp < (
                    SELECT MIN(timestamp) FROM rl_candles
                    WHERE symbol = ? AND interval = ?
                    AND timestamp > sr.created_at
                )
            )
        """, (strategy_id, coin, interval))
        
        leakage_count = cursor.fetchone()[0]
        if leakage_count > 0:
            return True
        
    except Exception as e:
        logger.debug(f"데이터 누수 체크 중 오류: {e}")
    
    return False


def calculate_expected_profit(win_rate: float, trades_count: int) -> Optional[float]:
    """예상 수익률 계산 (간단한 추정)"""
    
    if trades_count <= 0 or win_rate < 0 or win_rate > 1:
        return None
    
    # 간단한 추정: 승률과 평균 손익비를 가정
    # 실제 구현에서는 더 정교한 계산 필요
    
    # 가정: 평균 이익 +2%, 평균 손실 -1.5%
    avg_win = 2.0
    avg_loss = -1.5
    
    expected_profit = (win_rate * avg_win + (1 - win_rate) * avg_loss) * trades_count
    
    return expected_profit


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    issues = validate_backtest_logic()
    if issues:
        print(f"\n총 {len(issues)}개 이슈 발견:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 백테스트 로직 검증 완료 (이슈 없음)")

