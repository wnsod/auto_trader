"""
롤업 배치 작업
예측형 강화학습 시스템의 전략별 통계 집계
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.engine.adaptive_rollup import (
    calculate_adaptive_rollup_days,
    create_adaptive_rollup_view
)
from rl_pipeline.core.errors import DBWriteError
from rl_pipeline.pipelines.selfplay_adaptive import get_adaptive_predictive_ratio

logger = logging.getLogger(__name__)


def run_rollup_batch(
    coin: Optional[str] = None,
    interval: Optional[str] = None,
    days: Optional[int] = None
) -> Dict[str, Any]:
    """
    롤업 배치 실행
    
    최근 N일간의 에피소드 결과를 집계하여 전략별 통계 계산
    
    Args:
        coin: 특정 코인만 처리 (None이면 전체)
        interval: 특정 인터벌만 처리 (None이면 전체)
        days: 롤업 기간 (None이면 적응형 기간 사용)
    
    Returns:
        집계 결과 딕셔너리
    """
    try:
        logger.info("🔄 롤업 배치 작업 시작...")
        
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 처리할 코인/인터벌 조합 조회
            if coin and interval:
                combinations = [(coin, interval)]
            else:
                combinations = get_coin_interval_combinations(cursor, coin, interval)
            
            logger.info(f"📊 처리할 조합: {len(combinations)}개")
            
            total_processed = 0
            total_strategies = 0
            
            for coin_item, interval_item in combinations:
                try:
                    # 적응형 롤업 기간 계산
                    if days is None:
                        optimal_days = calculate_adaptive_rollup_days(
                            coin_item, interval_item, conn
                        )
                    else:
                        optimal_days = days
                    
                    logger.info(f"🔄 {coin_item}-{interval_item} 롤업 처리 (기간: {optimal_days}일)...")
                    
                    # 전략별 롤업 계산
                    rollup_count = compute_strategy_rollup(
                        coin_item, interval_item, optimal_days, conn
                    )
                    
                    total_processed += 1
                    total_strategies += rollup_count
                    
                    logger.info(f"✅ {coin_item}-{interval_item} 롤업 완료: {rollup_count}개 전략")

                    # 상태별 예측 정확도 앙상블 계산 (옵션)
                    try:
                        compute_state_ensemble(conn, coin_item, interval_item)
                    except Exception as e:
                        logger.debug(f"⚠️ 상태 앙상블 계산 실패(무시): {e}")
                    
                except Exception as e:
                    logger.error(f"❌ {coin_item}-{interval_item} 롤업 실패: {e}", exc_info=True)
                    continue
            
            logger.info(f"✅ 롤업 배치 완료: {total_processed}개 조합, {total_strategies}개 전략")
            
            return {
                "success": True,
                "combinations_processed": total_processed,
                "strategies_updated": total_strategies
            }
            
    except Exception as e:
        logger.error(f"❌ 롤업 배치 실패: {e}", exc_info=True)
        raise DBWriteError(f"롤업 배치 실패: {e}") from e


def get_coin_interval_combinations(
    cursor,
    coin_filter: Optional[str] = None,
    interval_filter: Optional[str] = None
) -> List[tuple]:
    """
    처리할 코인/인터벌 조합 조회
    
    Args:
        cursor: DB 커서
        coin_filter: 코인 필터
        interval_filter: 인터벌 필터
    
    Returns:
        (coin, interval) 튜플 리스트
    """
    try:
        if coin_filter and interval_filter:
            return [(coin_filter, interval_filter)]
        
        from rl_pipeline.core.utils import safe_query, table_exists
        
        # rl_episodes 테이블 존재 확인
        if not table_exists(cursor, "rl_episodes"):
            logger.warning("⚠️ rl_episodes 테이블이 존재하지 않음, 빈 결과 반환")
            return []
        
        # rl_episodes 테이블에서 고유 조합 조회 (안전한 쿼리 사용)
        if coin_filter:
            query = "SELECT DISTINCT coin, interval FROM rl_episodes WHERE coin = ?"
            results = safe_query(cursor, query, (coin_filter,), table_name="rl_episodes")
        elif interval_filter:
            query = "SELECT DISTINCT coin, interval FROM rl_episodes WHERE interval = ?"
            results = safe_query(cursor, query, (interval_filter,), table_name="rl_episodes")
        else:
            query = "SELECT DISTINCT coin, interval FROM rl_episodes"
            results = safe_query(cursor, query, (), table_name="rl_episodes")
        
        return [(row[0], row[1]) for row in results]
        
    except Exception as e:
        logger.error(f"❌ 코인/인터벌 조합 조회 실패: {e}", exc_info=True)
        return []


def compute_strategy_rollup(
    coin: str,
    interval: str,
    days: int,
    conn
) -> int:
    """
    전략별 롤업 계산 및 저장
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        days: 롤업 기간 (일)
        conn: DB 연결
    
    Returns:
        업데이트된 전략 수
    """
    try:
        cursor = conn.cursor()
        
        # 적응형 뷰 생성 또는 직접 쿼리
        cutoff_ts = int((datetime.now().timestamp() - (days * 86400)))
        
        # 🔥 옵션 A: adaptive_ratio 조회
        try:
            adaptive_ratio = get_adaptive_predictive_ratio(coin, interval)
        except Exception as e:
            logger.debug(f"⚠️ adaptive_ratio 조회 실패, 기본값 사용: {e}")
            adaptive_ratio = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
        
        # 🆕 시간 가중치 활성화 여부 체크
        use_time_weighting = os.getenv('ENABLE_TIME_WEIGHTED_ROLLUP', 'true').lower() == 'true'
        
        if use_time_weighting:
            # 🆕 시간 가중 평균 사용 (최근 데이터에 더 높은 가중치)
            # 가중치: exp(-decay_rate * days_ago)
            # 최근 7일: 가중치 ~1.0, 20일 전: 가중치 ~0.3
            current_ts = int(datetime.now().timestamp())
            decay_rate = float(os.getenv('ROLLUP_TIME_DECAY_RATE', '0.05'))  # 기본값: 0.05 (5% per day)
            
            # 🔥 옵션 A: source_type으로 구분하여 계산 (Python에서 가중 평균 계산)
            # 1단계: 예측 self-play 데이터 조회
            predictive_query = """
                SELECT
                    s.strategy_id,
                    COUNT(*) AS pred_count,
                    SUM(s.realized_ret_signed * EXP(-? * (? - s.ts_exit) / 86400.0)) 
                        / NULLIF(SUM(EXP(-? * (? - s.ts_exit) / 86400.0)), 0) AS pred_avg_ret,
                    SUM(CASE WHEN s.realized_ret_signed > 0 THEN 1.0 ELSE 0.0 END 
                        * EXP(-? * (? - s.ts_exit) / 86400.0)) 
                        / NULLIF(SUM(EXP(-? * (? - s.ts_exit) / 86400.0)), 0) AS pred_win_rate,
                    SUM(CASE 
                        WHEN s.acc_flag IS NOT NULL THEN CAST(s.acc_flag AS REAL)
                        WHEN s.first_event = 'TP' THEN 1.0
                        WHEN s.first_event = 'expiry' AND s.realized_ret_signed > 0 THEN 0.5
                        ELSE 0.0 
                    END * EXP(-? * (? - s.ts_exit) / 86400.0))
                        / NULLIF(SUM(EXP(-? * (? - s.ts_exit) / 86400.0)), 0) AS pred_acc
                FROM rl_episode_summary s
                WHERE s.coin = ? AND s.interval = ? 
                  AND s.ts_exit >= ?
                  AND (s.source_type = 'predictive' OR s.source_type = 'regime_routing' OR s.source_type IS NULL)
                GROUP BY s.strategy_id
            """
            cursor.execute(predictive_query, (
                decay_rate, current_ts, decay_rate, current_ts,  # pred_avg_ret
                decay_rate, current_ts, decay_rate, current_ts,  # pred_win_rate
                decay_rate, current_ts, decay_rate, current_ts,  # pred_acc
                coin, interval, cutoff_ts
            ))
            predictive_results = cursor.fetchall()
            
            # 2단계: 시뮬레이션 self-play 데이터 조회
            simulation_query = """
                SELECT
                    s.strategy_id,
                    COUNT(*) AS sim_count,
                    SUM(s.realized_ret_signed * EXP(-? * (? - s.ts_exit) / 86400.0)) 
                        / NULLIF(SUM(EXP(-? * (? - s.ts_exit) / 86400.0)), 0) AS sim_avg_ret,
                    SUM(CASE WHEN s.realized_ret_signed > 0 THEN 1.0 ELSE 0.0 END 
                        * EXP(-? * (? - s.ts_exit) / 86400.0)) 
                        / NULLIF(SUM(EXP(-? * (? - s.ts_exit) / 86400.0)), 0) AS sim_win_rate
                FROM rl_episode_summary s
                WHERE s.coin = ? AND s.interval = ? 
                  AND s.ts_exit >= ?
                  AND s.source_type = 'simulation'
                GROUP BY s.strategy_id
            """
            cursor.execute(simulation_query, (
                decay_rate, current_ts, decay_rate, current_ts,  # sim_avg_ret
                decay_rate, current_ts, decay_rate, current_ts,  # sim_win_rate
                coin, interval, cutoff_ts
            ))
            simulation_results = cursor.fetchall()
            
            # 3단계: avg_dd 계산 (전체 데이터)
            avg_dd_query = """
                SELECT
                    s.strategy_id,
                    SUM(ABS(s.realized_ret_signed) * EXP(-? * (? - s.ts_exit) / 86400.0))
                        / NULLIF(SUM(EXP(-? * (? - s.ts_exit) / 86400.0)), 0) AS avg_dd
                FROM rl_episode_summary s
                WHERE s.coin = ? AND s.interval = ? 
                  AND s.ts_exit >= ?
                GROUP BY s.strategy_id
            """
            cursor.execute(avg_dd_query, (
                decay_rate, current_ts, decay_rate, current_ts,
                coin, interval, cutoff_ts
            ))
            avg_dd_results = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 🔥 4-1단계: segment_scores에서 온라인 Self-play 성과 조회 (통합)
            try:
                from rl_pipeline.core.utils import table_exists
                if table_exists(cursor, "segment_scores"):
                    online_cutoff_timestamp = int(datetime.now().timestamp() - (days * 86400))
                    online_query = """
                        SELECT
                            strategy_id,
                            COUNT(*) AS online_count,
                            SUM(profit * EXP(-? * (? - COALESCE(end_timestamp, created_at)) / 86400.0)) 
                                / NULLIF(SUM(EXP(-? * (? - COALESCE(end_timestamp, created_at)) / 86400.0)), 0) AS online_avg_ret,
                            SUM(CASE WHEN profit > 0 THEN 1.0 ELSE 0.0 END 
                                * EXP(-? * (? - COALESCE(end_timestamp, created_at)) / 86400.0)) 
                                / NULLIF(SUM(EXP(-? * (? - COALESCE(end_timestamp, created_at)) / 86400.0)), 0) AS online_win_rate,
                            AVG(pf) AS online_pf
                        FROM segment_scores
                        WHERE market = ? AND interval = ?
                          AND (COALESCE(end_timestamp, created_at) >= ? OR created_at >= datetime('now', '-' || ? || ' days'))
                        GROUP BY strategy_id
                    """
                    # SQLite timestamp 처리
                    cursor.execute("""
                        SELECT
                            strategy_id,
                            COUNT(*) AS online_count,
                            SUM(profit * EXP(-? * (strftime('%s', 'now') - CAST(COALESCE(end_timestamp, strftime('%s', created_at)) AS INTEGER)) / 86400.0)) 
                                / NULLIF(SUM(EXP(-? * (strftime('%s', 'now') - CAST(COALESCE(end_timestamp, strftime('%s', created_at)) AS INTEGER)) / 86400.0)), 0) AS online_avg_ret,
                            SUM(CASE WHEN profit > 0 THEN 1.0 ELSE 0.0 END 
                                * EXP(-? * (strftime('%s', 'now') - CAST(COALESCE(end_timestamp, strftime('%s', created_at)) AS INTEGER)) / 86400.0)) 
                                / NULLIF(SUM(EXP(-? * (strftime('%s', 'now') - CAST(COALESCE(end_timestamp, strftime('%s', created_at)) AS INTEGER)) / 86400.0)), 0) AS online_win_rate,
                            AVG(pf) AS online_pf
                        FROM segment_scores
                        WHERE market = ? AND interval = ?
                          AND (CAST(COALESCE(end_timestamp, strftime('%s', created_at)) AS INTEGER) >= ? OR created_at >= datetime('now', '-' || ? || ' days'))
                        GROUP BY strategy_id
                    """, (
                        decay_rate, decay_rate,  # online_avg_ret
                        decay_rate, decay_rate,  # online_win_rate
                        coin, interval, online_cutoff_timestamp, days
                    ))
                    online_results = cursor.fetchall()
                    online_dict = {row[0]: {
                        'count': row[1], 'avg_ret': (row[2] or 0.0) * 100,  # 퍼센트로 변환
                        'win_rate': row[3] or 0.0, 'pf': row[4] or 0.0
                    } for row in online_results}
                    logger.debug(f"✅ {coin}-{interval}: 온라인 Self-play 세그먼트 {sum(o['count'] for o in online_dict.values())}개 발견")
                else:
                    online_dict = {}
                    logger.debug(f"⚠️ segment_scores 테이블 없음 (온라인 Self-play 성과 제외)")
            except Exception as e:
                logger.debug(f"⚠️ 온라인 Self-play 성과 조회 실패: {e}")
                online_dict = {}
            
            # 4단계: Python에서 데이터 병합 및 가중 평균 계산
            pred_dict = {row[0]: {
                'count': row[1], 'avg_ret': row[2] or 0.0, 
                'win_rate': row[3] or 0.0, 'acc': row[4] or 0.0
            } for row in predictive_results}
            
            sim_dict = {row[0]: {
                'count': row[1], 'avg_ret': row[2] or 0.0, 
                'win_rate': row[3] or 0.0
            } for row in simulation_results}
            
            # 모든 전략 ID 수집 (온라인 Self-play 포함)
            all_strategy_ids = set(pred_dict.keys()) | set(sim_dict.keys()) | set(online_dict.keys())
            
            # 결과 생성 (온라인 Self-play 성과 통합)
            results = []
            for strategy_id in all_strategy_ids:
                pred = pred_dict.get(strategy_id, {'count': 0, 'avg_ret': 0.0, 'win_rate': 0.0, 'acc': 0.0})
                sim = sim_dict.get(strategy_id, {'count': 0, 'avg_ret': 0.0, 'win_rate': 0.0})
                online = online_dict.get(strategy_id, {'count': 0, 'avg_ret': 0.0, 'win_rate': 0.0, 'pf': 0.0})
                
                # 에피소드 수 합산 (온라인 Self-play 포함)
                episodes_trained = pred['count'] + sim['count'] + online['count']
                
                # 🔥 온라인 Self-play 성과 가중치 계산 (세그먼트 기반, 더 정확한 성과 반영)
                # 온라인 Self-play가 있으면 우선적으로 반영 (최근 성과)
                online_weight = min(0.4, online['count'] / max(episodes_trained, 1) * 0.5) if online['count'] > 0 else 0.0
                remaining_weight = 1.0 - online_weight
                
                # 가중 평균 계산 (온라인 Self-play 우선)
                if online_weight > 0:
                    avg_ret = (online['avg_ret'] * online_weight + 
                              sim['avg_ret'] * (remaining_weight * (1.0 - adaptive_ratio)) + 
                              pred['avg_ret'] * (remaining_weight * adaptive_ratio))
                    win_rate = (online['win_rate'] * online_weight + 
                               sim['win_rate'] * (remaining_weight * (1.0 - adaptive_ratio)) + 
                               pred['win_rate'] * (remaining_weight * adaptive_ratio))
                else:
                    # 온라인 Self-play 없으면 기존 방식
                    avg_ret = sim['avg_ret'] * (1.0 - adaptive_ratio) + pred['avg_ret'] * adaptive_ratio
                    win_rate = sim['win_rate'] * (1.0 - adaptive_ratio) + pred['win_rate'] * adaptive_ratio
                
                predictive_accuracy = pred['acc']  # 예측 정확도는 예측 self-play만
                avg_dd = avg_dd_results.get(strategy_id, 0.0) or 0.0
                updated_at = int(datetime.now().timestamp())
                
                results.append((
                    strategy_id, coin, interval,
                    episodes_trained, avg_ret, win_rate, predictive_accuracy, avg_dd, updated_at
                ))
        else:
            # 기존 방식: 단순 평균 (시간 가중치 없음) - Python에서 가중 평균 계산
            # 1단계: 예측 self-play 데이터 + 레짐 라우팅 데이터
            predictive_query = """
                SELECT
                    s.strategy_id,
                    COUNT(*) AS pred_count,
                    AVG(s.realized_ret_signed) AS pred_avg_ret,
                    AVG(CASE WHEN s.realized_ret_signed > 0 THEN 1.0 ELSE 0.0 END) AS pred_win_rate,
                    AVG(CASE 
                        WHEN s.acc_flag IS NOT NULL THEN CAST(s.acc_flag AS REAL)
                        WHEN s.first_event = 'TP' THEN 1.0
                        WHEN s.first_event = 'expiry' AND s.realized_ret_signed > 0 THEN 0.5
                        ELSE 0.0 
                    END) AS pred_acc
                FROM rl_episode_summary s
                WHERE s.coin = ? AND s.interval = ?
                  AND s.ts_exit >= ?
                  AND (s.source_type = 'predictive' OR s.source_type = 'regime_routing' OR s.source_type IS NULL)
                GROUP BY s.strategy_id
            """
            cursor.execute(predictive_query, (coin, interval, cutoff_ts))
            predictive_results = cursor.fetchall()
            
            # 2단계: 시뮬레이션 self-play 데이터
            simulation_query = """
                SELECT
                    s.strategy_id,
                    COUNT(*) AS sim_count,
                    AVG(s.realized_ret_signed) AS sim_avg_ret,
                    AVG(CASE WHEN s.realized_ret_signed > 0 THEN 1.0 ELSE 0.0 END) AS sim_win_rate
                FROM rl_episode_summary s
                WHERE s.coin = ? AND s.interval = ?
                  AND s.ts_exit >= ?
                  AND s.source_type = 'simulation'
                GROUP BY s.strategy_id
            """
            cursor.execute(simulation_query, (coin, interval, cutoff_ts))
            simulation_results = cursor.fetchall()
            
            # 3단계: avg_dd 계산
            avg_dd_query = """
                SELECT
                    s.strategy_id,
                    AVG(ABS(s.realized_ret_signed)) AS avg_dd
                FROM rl_episode_summary s
                WHERE s.coin = ? AND s.interval = ?
                  AND s.ts_exit >= ?
                GROUP BY s.strategy_id
            """
            cursor.execute(avg_dd_query, (coin, interval, cutoff_ts))
            avg_dd_results = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 🔥 4-1단계: segment_scores에서 온라인 Self-play 성과 조회 (통합)
            try:
                from rl_pipeline.core.utils import table_exists
                if table_exists(cursor, "segment_scores"):
                    online_query = """
                        SELECT
                            strategy_id,
                            COUNT(*) AS online_count,
                            AVG(profit) * 100 AS online_avg_ret,
                            AVG(CASE WHEN profit > 0 THEN 1.0 ELSE 0.0 END) AS online_win_rate,
                            AVG(pf) AS online_pf
                        FROM segment_scores
                        WHERE market = ? AND interval = ?
                          AND (created_at >= datetime('now', '-' || ? || ' days') OR end_timestamp >= ?)
                        GROUP BY strategy_id
                    """
                    cursor.execute(online_query, (coin, interval, days, cutoff_ts))
                    online_results = cursor.fetchall()
                    online_dict = {row[0]: {
                        'count': row[1], 'avg_ret': row[2] or 0.0,
                        'win_rate': row[3] or 0.0, 'pf': row[4] or 0.0
                    } for row in online_results}
                    logger.debug(f"✅ {coin}-{interval}: 온라인 Self-play 세그먼트 {sum(o['count'] for o in online_dict.values())}개 발견")
                else:
                    online_dict = {}
            except Exception as e:
                logger.debug(f"⚠️ 온라인 Self-play 성과 조회 실패: {e}")
                online_dict = {}
            
            # 4단계: Python에서 병합 및 가중 평균
            pred_dict = {row[0]: {
                'count': row[1], 'avg_ret': row[2] or 0.0, 
                'win_rate': row[3] or 0.0, 'acc': row[4] or 0.0
            } for row in predictive_results}
            
            sim_dict = {row[0]: {
                'count': row[1], 'avg_ret': row[2] or 0.0, 
                'win_rate': row[3] or 0.0
            } for row in simulation_results}
            
            all_strategy_ids = set(pred_dict.keys()) | set(sim_dict.keys()) | set(online_dict.keys())
            
            results = []
            for strategy_id in all_strategy_ids:
                pred = pred_dict.get(strategy_id, {'count': 0, 'avg_ret': 0.0, 'win_rate': 0.0, 'acc': 0.0})
                sim = sim_dict.get(strategy_id, {'count': 0, 'avg_ret': 0.0, 'win_rate': 0.0})
                online = online_dict.get(strategy_id, {'count': 0, 'avg_ret': 0.0, 'win_rate': 0.0, 'pf': 0.0})
                
                episodes_trained = pred['count'] + sim['count'] + online['count']
                
                # 온라인 Self-play 성과 통합
                online_weight = min(0.4, online['count'] / max(episodes_trained, 1) * 0.5) if online['count'] > 0 else 0.0
                remaining_weight = 1.0 - online_weight
                
                if online_weight > 0:
                    avg_ret = (online['avg_ret'] * online_weight + 
                              sim['avg_ret'] * (remaining_weight * (1.0 - adaptive_ratio)) + 
                              pred['avg_ret'] * (remaining_weight * adaptive_ratio))
                    win_rate = (online['win_rate'] * online_weight + 
                               sim['win_rate'] * (remaining_weight * (1.0 - adaptive_ratio)) + 
                               pred['win_rate'] * (remaining_weight * adaptive_ratio))
                else:
                    avg_ret = sim['avg_ret'] * (1.0 - adaptive_ratio) + pred['avg_ret'] * adaptive_ratio
                    win_rate = sim['win_rate'] * (1.0 - adaptive_ratio) + pred['win_rate'] * adaptive_ratio
                
                predictive_accuracy = pred['acc']
                avg_dd = avg_dd_results.get(strategy_id, 0.0) or 0.0
                updated_at = int(datetime.now().timestamp())
                
                results.append((
                    strategy_id, coin, interval,
                    episodes_trained, avg_ret, win_rate, predictive_accuracy, avg_dd, updated_at
                ))
        
        # 결과 확인 (이미 Python에서 생성됨)
        if not results:
            logger.info(f"⚠️ {coin}-{interval}: 롤업할 데이터 없음 (Self-play 또는 레짐 라우팅 데이터 없음)")
            logger.info(f"   💡 Self-play가 비활성화되어 있거나 rl_episode_summary에 데이터가 없습니다")
            logger.info(f"   💡 롤업은 Self-play 또는 레짐 라우팅 결과가 있을 때만 수행됩니다")
            return 0
        
        # rl_strategy_rollup 테이블에 저장
        updated_count = 0
        for row in results:
            strategy_id, coin_item, interval_item, episodes_trained, avg_ret, win_rate, \
            predictive_accuracy, avg_dd, updated_at = row

            # NULL 처리
            avg_ret = avg_ret if avg_ret is not None else 0.0
            win_rate = win_rate if win_rate is not None else 0.0
            predictive_accuracy = predictive_accuracy if predictive_accuracy is not None else 0.0
            avg_dd = avg_dd if avg_dd is not None else 0.0

            # 🔥 Sharpe Ratio, Profit Factor 등 추가 계산
            import math
            sharpe_ratio = 0.0
            profit_factor = 0.0
            total_profit = 0.0
            avg_reward = 0.0
            best_reward = 0.0
            worst_reward = 0.0
            total_episodes = episodes_trained

            try:
                # 에피소드 데이터 조회
                cursor.execute("""
                    SELECT realized_ret_signed, total_reward
                    FROM rl_episode_summary
                    WHERE strategy_id = ? AND coin = ? AND interval = ?
                      AND ts_exit >= ?
                """, (strategy_id, coin_item, interval_item, cutoff_ts))

                episode_data = cursor.fetchall()

                if episode_data:
                    returns = [e[0] for e in episode_data if e[0] is not None]
                    rewards = [e[1] for e in episode_data if e[1] is not None]

                    # Sharpe Ratio 계산
                    if len(returns) >= 2:
                        avg_return = sum(returns) / len(returns)
                        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
                        std_dev = math.sqrt(variance) if variance > 0 else 0.0
                        sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0.0

                    # Profit Factor 계산
                    total_gains = sum(r for r in returns if r > 0)
                    total_losses = abs(sum(r for r in returns if r < 0))
                    profit_factor = total_gains / total_losses if total_losses > 0 else (float('inf') if total_gains > 0 else 0.0)
                    if profit_factor == float('inf'):
                        profit_factor = 999.0  # DB 저장용 제한

                    # 기타 통계
                    total_profit = sum(returns)
                    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                    best_reward = max(rewards) if rewards else 0.0
                    worst_reward = min(rewards) if rewards else 0.0
                    total_episodes = len(episode_data)

            except Exception as e:
                logger.debug(f"⚠️ 추가 통계 계산 실패 (기본값 사용): {e}")

            cursor.execute("""
                INSERT OR REPLACE INTO rl_strategy_rollup (
                    strategy_id, coin, interval,
                    episodes_trained, avg_ret, win_rate, predictive_accuracy, avg_dd,
                    total_episodes, total_profit, avg_reward,
                    avg_profit_factor, avg_sharpe_ratio,
                    best_episode_reward, worst_episode_reward,
                    updated_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, coin_item, interval_item,
                episodes_trained, avg_ret, win_rate, predictive_accuracy, avg_dd,
                total_episodes, total_profit, avg_reward,
                profit_factor, sharpe_ratio,
                best_reward, worst_reward,
                updated_at, datetime.now().isoformat()
            ))

            updated_count += 1
        
        conn.commit()
        
        logger.info(f"✅ {coin}-{interval} 롤업 저장 완료: {updated_count}개 전략")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"❌ 전략별 롤업 계산 실패: {e}", exc_info=True)
        conn.rollback()
        return 0


def compute_strategy_grades(
    coin: Optional[str] = None,
    interval: Optional[str] = None
) -> int:
    """
    전략 등급 계산 및 저장
    
    rl_strategy_rollup 데이터를 기반으로 strategy_grades 계산
    
    Args:
        coin: 특정 코인만 처리
        interval: 특정 인터벌만 처리
    
    Returns:
        업데이트된 전략 수
    """
    try:
        logger.info("🔄 전략 등급 계산 시작...")
        
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 🔥 레짐 정보를 포함하여 전략 조회 (레짐별 상대평가를 위함)
            if coin and interval:
                query = """
                    SELECT r.strategy_id, r.coin, r.interval,
                           r.avg_ret AS total_return,
                           r.win_rate, r.predictive_accuracy,
                           COALESCE(c.regime, 'UNKNOWN') as regime
                    FROM rl_strategy_rollup r
                    LEFT JOIN coin_strategies c
                        ON r.strategy_id = c.id
                        AND r.coin = c.coin
                        AND r.interval = c.interval
                    WHERE r.coin = ? AND r.interval = ?
                """
                cursor.execute(query, (coin, interval))
            elif coin:
                query = """
                    SELECT r.strategy_id, r.coin, r.interval,
                           r.avg_ret AS total_return,
                           r.win_rate, r.predictive_accuracy,
                           COALESCE(c.regime, 'UNKNOWN') as regime
                    FROM rl_strategy_rollup r
                    LEFT JOIN coin_strategies c
                        ON r.strategy_id = c.id
                        AND r.coin = c.coin
                        AND r.interval = c.interval
                    WHERE r.coin = ?
                """
                cursor.execute(query, (coin,))
            elif interval:
                query = """
                    SELECT r.strategy_id, r.coin, r.interval,
                           r.avg_ret AS total_return,
                           r.win_rate, r.predictive_accuracy,
                           COALESCE(c.regime, 'UNKNOWN') as regime
                    FROM rl_strategy_rollup r
                    LEFT JOIN coin_strategies c
                        ON r.strategy_id = c.id
                        AND r.coin = c.coin
                        AND r.interval = c.interval
                    WHERE r.interval = ?
                """
                cursor.execute(query, (interval,))
            else:
                query = """
                    SELECT r.strategy_id, r.coin, r.interval,
                           r.avg_ret AS total_return,
                           r.win_rate, r.predictive_accuracy,
                           COALESCE(c.regime, 'UNKNOWN') as regime
                    FROM rl_strategy_rollup r
                    LEFT JOIN coin_strategies c
                        ON r.strategy_id = c.id
                        AND r.coin = c.coin
                        AND r.interval = c.interval
                """
                cursor.execute(query)

            results = cursor.fetchall()

            if not results:
                logger.warning("⚠️ 등급 계산할 데이터 없음")
                return 0

            # 1단계: 레짐별로 그룹화하여 점수 계산
            from collections import defaultdict
            regime_strategies = defaultdict(list)

            for row in results:
                strategy_id, coin_item, interval_item, total_return, win_rate, predictive_accuracy, regime = row

                # NULL 처리
                total_return = total_return if total_return is not None else 0.0
                win_rate = win_rate if win_rate is not None else 0.0
                predictive_accuracy = predictive_accuracy if predictive_accuracy is not None else 0.0
                regime = regime if regime else 'UNKNOWN'

                # 🔥 등급 점수 계산 (예측 정확도 없을 때 대체 방법 사용)
                # 예측 정확도가 기본값(0.0)이거나 너무 낮으면 대체 평가 사용
                has_valid_predictive_accuracy = predictive_accuracy > 0.01  # 1% 이상이면 유효
                
                if has_valid_predictive_accuracy:
                    # 예측 정확도가 있으면 기존 방식 사용
                    grade_score = (
                        predictive_accuracy * 0.6 +  # 예측 정확도 60%
                        win_rate * 0.25 +            # 승률 25%
                        min(abs(total_return) / 0.1, 1.0) * 0.15  # 수익률 15%
                    )
                else:
                    # 🔥 예측 정확도가 없을 때: 레짐 라우팅 점수 기반 평가
                    # 승률과 수익률에 더 높은 가중치 부여
                    grade_score = (
                        win_rate * 0.50 +            # 승률 50% (증가)
                        min(abs(total_return) / 0.1, 1.0) * 0.30 +  # 수익률 30% (증가)
                        min(win_rate * 2.0, 1.0) * 0.20  # 승률 기반 보너스 20%
                    )
                    # 예측 정확도가 없어도 기본 점수 부여 (너무 낮은 점수 방지)
                    grade_score = max(grade_score, 0.20)  # 최소 0.20 점수 보장
                
                grade_score = max(0.0, min(1.0, grade_score))

                regime_strategies[regime].append({
                    'strategy_id': strategy_id,
                    'coin': coin_item,
                    'interval': interval_item,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'predictive_accuracy': predictive_accuracy,
                    'grade_score': grade_score,
                    'regime': regime
                })

            # 2단계: 레짐별로 상대평가 등급 부여
            all_graded_strategies = []

            for regime, strategies in regime_strategies.items():
                # 레짐 내에서 점수 기준 정렬
                strategies.sort(key=lambda x: x['grade_score'], reverse=True)

                # 레짐 내 상대평가
                regime_count = len(strategies)
                logger.info(f"📊 레짐별 상대평가: {regime} → {regime_count}개 전략")

                for idx, strategy_info in enumerate(strategies):
                    percentile = (idx + 1) / regime_count

                    # 상대평가 등급 결정 (레짐 내)
                    if percentile <= 0.10:
                        grade = 'S'
                    elif percentile <= 0.30:
                        grade = 'A'
                    elif percentile <= 0.50:
                        grade = 'B'
                    elif percentile <= 0.70:
                        grade = 'C'
                    elif percentile <= 0.90:
                        grade = 'D'
                    else:
                        grade = 'F'

                    strategy_info['grade'] = grade
                    all_graded_strategies.append(strategy_info)

            # 3단계: 등급 저장
            updated_count = 0
            for strategy_info in all_graded_strategies:
                strategy_id = strategy_info['strategy_id']
                coin_item = strategy_info['coin']
                interval_item = strategy_info['interval']
                total_return = strategy_info['total_return']
                win_rate = strategy_info['win_rate']
                predictive_accuracy = strategy_info['predictive_accuracy']
                grade_score = strategy_info['grade_score']
                grade = strategy_info['grade']
                
                # 저장
                updated_at = int(datetime.now().timestamp())
                
                # 1. strategy_grades 테이블에 저장 (있는 경우)
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO strategy_grades (
                            strategy_id, coin, interval,
                            total_return, win_rate, predictive_accuracy,
                            grade_score, grade, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        strategy_id, coin_item, interval_item,
                        total_return, win_rate, predictive_accuracy,
                        grade_score, grade, updated_at
                    ))
                except Exception as e:
                    # strategy_grades 테이블이 없으면 무시 (선택적)
                    logger.debug(f"⚠️ strategy_grades 테이블 없음 또는 에러: {e}")
                
                # 2. rl_strategy_rollup 테이블의 grade도 업데이트
                try:
                    cursor.execute("""
                        UPDATE rl_strategy_rollup
                        SET grade = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE strategy_id = ? AND coin = ? AND interval = ?
                    """, (grade, strategy_id, coin_item, interval_item))
                except Exception as e:
                    logger.debug(f"⚠️ rl_strategy_rollup grade 업데이트 실패: {e}")
                
                # 3. coin_strategies 테이블의 quality_grade도 업데이트 (직접 동기화 개선)
                # 🔥 주의: 이 부분은 각 전략별로 업데이트하지만, 
                # 마지막에 coin/interval 기준 일괄 동기화를 수행하므로 여기서는 시도만 함
                try:
                    from rl_pipeline.core.utils import table_exists
                    if table_exists(cursor, "coin_strategies"):
                        # 🔥 방법 1: 정확한 ID 매칭
                        cursor.execute("""
                            UPDATE coin_strategies
                            SET quality_grade = ?, updated_at = datetime('now')
                            WHERE id = ? AND coin = ? AND interval = ?
                        """, (grade, strategy_id, coin_item, interval_item))
                        
                        # 🔥 방법 2: ID가 정확히 일치하지 않는 경우, 부분 매칭 시도
                        if cursor.rowcount == 0:
                            # strategy_id의 마지막 부분으로 매칭 (타임스탬프 제외)
                            # 예: 0G_15m_ai_momentum_breakout_1762347626_12000_v2_20251105_130044
                            #     → 0G_15m_ai_momentum_breakout_1762347626_12000_v2
                            strategy_base = '_'.join(strategy_id.split('_')[:-2]) if '_' in strategy_id else strategy_id
                            cursor.execute("""
                                UPDATE coin_strategies
                                SET quality_grade = ?, updated_at = datetime('now')
                                WHERE coin = ? AND interval = ?
                                  AND (quality_grade IS NULL OR quality_grade = 'UNKNOWN')
                                  AND id LIKE ?
                                LIMIT 1
                            """, (grade, coin_item, interval_item, f"{strategy_base}%"))
                except Exception as e:
                    logger.debug(f"⚠️ coin_strategies quality_grade 개별 업데이트 실패: {e}")
                
                updated_count += 1
            
            # 🔥 4. coin/interval 기준 일괄 동기화 (누락된 등급 채우기)
            # rl_strategy_rollup에 등급이 있지만 coin_strategies에 NULL/UNKNOWN인 경우
            try:
                from rl_pipeline.core.utils import table_exists
                if table_exists(cursor, "coin_strategies") and table_exists(cursor, "rl_strategy_rollup"):
                    # coin/interval 조합별로 처리
                    cursor.execute("""
                        SELECT DISTINCT coin, interval 
                        FROM rl_strategy_rollup 
                        WHERE grade IS NOT NULL AND grade != 'UNKNOWN'
                    """)
                    coin_interval_pairs = cursor.fetchall()
                    
                    batch_sync_count = 0
                    for coin_item, interval_item in coin_interval_pairs:
                        # 해당 coin/interval의 가장 높은 등급으로 업데이트
                        # 등급 우선순위: S > A > B > C > D > F
                        cursor.execute("""
                            UPDATE coin_strategies
                            SET quality_grade = (
                                SELECT grade FROM rl_strategy_rollup
                                WHERE rl_strategy_rollup.coin = coin_strategies.coin
                                  AND rl_strategy_rollup.interval = coin_strategies.interval
                                  AND rl_strategy_rollup.grade IS NOT NULL
                                  AND rl_strategy_rollup.grade != 'UNKNOWN'
                                ORDER BY CASE grade
                                    WHEN 'S' THEN 1
                                    WHEN 'A' THEN 2
                                    WHEN 'B' THEN 3
                                    WHEN 'C' THEN 4
                                    WHEN 'D' THEN 5
                                    WHEN 'F' THEN 6
                                    ELSE 7
                                END
                                LIMIT 1
                            ),
                            updated_at = datetime('now')
                            WHERE coin = ? AND interval = ?
                              AND (quality_grade IS NULL OR quality_grade = 'UNKNOWN')
                              AND EXISTS (
                                  SELECT 1 FROM rl_strategy_rollup
                                  WHERE rl_strategy_rollup.coin = coin_strategies.coin
                                    AND rl_strategy_rollup.interval = coin_strategies.interval
                                    AND rl_strategy_rollup.grade IS NOT NULL
                                    AND rl_strategy_rollup.grade != 'UNKNOWN'
                              )
                        """, (coin_item, interval_item))
                        
                        if cursor.rowcount > 0:
                            batch_sync_count += cursor.rowcount
                    
                    if batch_sync_count > 0:
                        logger.info(f"✅ coin/interval 기준 일괄 동기화: {batch_sync_count}개 전략 등급 업데이트")
            except Exception as e:
                logger.debug(f"⚠️ coin/interval 기준 일괄 동기화 실패: {e}")
            
            conn.commit()
            
            logger.info(f"✅ 전략 등급 계산 완료: {updated_count}개 전략 (일괄 동기화 포함)")
            
            return updated_count
            
    except Exception as e:
        logger.error(f"❌ 전략 등급 계산 실패: {e}")
        raise DBWriteError(f"전략 등급 계산 실패: {e}") from e


def _calculate_grade_text(grade_score: float, predictive_accuracy: float) -> str:
    """
    등급 점수로부터 등급 텍스트 계산 (완화된 기준, 상대평가)
    🔥 예측 정확도 없을 때 대체 평가 방법 사용

    Args:
        grade_score: 등급 점수 (0.0 ~ 1.0)
        predictive_accuracy: 예측 정확도 (0.0 ~ 1.0)

    Returns:
        등급 텍스트 (S/A/B/C/D/F)
    """
    # 🔥 예측 정확도가 없을 때 대체 평가
    has_valid_predictive_accuracy = predictive_accuracy > 0.01  # 1% 이상이면 유효
    
    if not has_valid_predictive_accuracy:
        # 🔥 예측 정확도 없을 때: grade_score만으로 평가 (완화된 기준)
        if grade_score >= 0.80:
            return 'A'
        elif grade_score >= 0.65:
            return 'B'
        elif grade_score >= 0.50:
            return 'C'
        elif grade_score >= 0.35:
            return 'D'
        else:
            return 'F'
    
    # 🔥 예측 정확도가 있을 때: 기존 방식 (완화된 기준)
    if predictive_accuracy >= 0.65 and grade_score >= 0.70:
        return 'S'
    elif predictive_accuracy >= 0.58 and grade_score >= 0.60:
        return 'A'
    elif predictive_accuracy >= 0.52 and grade_score >= 0.50:
        return 'B'
    elif predictive_accuracy >= 0.48 and grade_score >= 0.40:
        return 'C'
    elif predictive_accuracy >= 0.35 and grade_score >= 0.25:
        return 'D'
    else:
        return 'F'


def run_full_rollup_and_grades(
    coin: Optional[str] = None,
    interval: Optional[str] = None
) -> Dict[str, Any]:
    """
    전체 롤업 및 등급 계산 (편의 함수)
    
    1. 롤업 배치 실행
    2. 전략 등급 계산
    
    Args:
        coin: 특정 코인만 처리
        interval: 특정 인터벌만 처리
    
    Returns:
        실행 결과
    """
    try:
        logger.info("🔄 전체 롤업 및 등급 계산 시작...")
        
        # 1. 롤업 배치
        rollup_result = run_rollup_batch(coin=coin, interval=interval)
        
        # 2. 전략 등급 계산
        grades_count = compute_strategy_grades(coin=coin, interval=interval)
        
        logger.info("✅ 전체 롤업 및 등급 계산 완료")
        
        return {
            "success": True,
            "rollup": rollup_result,
            "grades_updated": grades_count
        }
        
    except Exception as e:
        logger.error(f"❌ 전체 롤업 및 등급 계산 실패: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def compute_state_ensemble(conn, coin: str, interval: str) -> bool:
    """상태별 예측 정확도 앙상블 계산 및 저장 (rl_state_ensemble)"""
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                e.state_key,
                SUM(CASE WHEN s.acc_flag = 1 AND e.predicted_dir = 1 THEN 1 ELSE 0 END) AS acc_up,
                SUM(CASE WHEN s.acc_flag = 1 AND e.predicted_dir = -1 THEN 1 ELSE 0 END) AS acc_down,
                AVG(COALESCE(s.acc_flag, 0)) AS acc_total,
                AVG(CASE WHEN e.predicted_dir = 1 THEN 1.0 ELSE 0.0 END) AS p_up_smooth,
                AVG(COALESCE(s.realized_ret_signed, 0.0)) AS e_ret_smooth,
                MAX(COALESCE(s.ts_exit, strftime('%s','now'))) AS last_updated
            FROM rl_episodes e
            JOIN rl_episode_summary s ON e.episode_id = s.episode_id
            WHERE e.coin = ? AND e.interval = ?
            GROUP BY e.state_key
            """,
            (coin, interval),
        )

        rows = cursor.fetchall()
        if not rows:
            return True

        for (state_key, acc_up, acc_down, acc_total, p_up, e_ret, last_updated) in rows:
            total_episodes = max(1, (acc_up or 0) + (acc_down or 0))
            confidence = min(total_episodes / 100.0, 1.0)
            cursor.execute(
                """
                INSERT OR REPLACE INTO rl_state_ensemble (
                    coin, interval, state_key,
                    acc_up, acc_down, acc_total,
                    p_up_smooth, e_ret_smooth, confidence, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    coin,
                    interval,
                    state_key,
                    float(acc_up or 0),
                    float(acc_down or 0),
                    float(acc_total or 0.0),
                    float(p_up or 0.0),
                    float(e_ret or 0.0),
                    float(confidence),
                    int(last_updated or 0),
                ),
            )

        conn.commit()
        logger.info(f"✅ 상태 앙상블 계산 완료: {coin}-{interval}, {len(rows)}개 상태")
        return True
    except Exception as e:
        logger.error(f"❌ 상태 앙상블 계산 실패: {e}")
        return False


if __name__ == "__main__":
    # 테스트
    print("롤업 배치 테스트:")
    
    # 테스트: 전체 롤업 실행
    result = run_full_rollup_and_grades()
    print(f"롤업 결과: {result}")

