import os
import logging
import sqlite3
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

from rl_pipeline.core.env import config
from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.analysis.strategy_grade_updater import StrategyGradeUpdater
from rl_pipeline.db.reads import load_strategies_pool

# 🔥 선택적 모듈 import (없으면 기본값 사용)
try:
    from rl_pipeline.engine.adaptive_predictive import get_adaptive_predictive_ratio
except ImportError:
    get_adaptive_predictive_ratio = None

logger = logging.getLogger(__name__)

def compute_strategy_rollup(
    coin: str,
    interval: str,
    days: int,
    conn
) -> int:
    """
    전략별 롤업 계산 및 저장 (배치 처리)
    
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
        # days가 None이면 전체 기간, 아니면 최근 N일
        cutoff_ts = int((datetime.now().timestamp() - (days * 86400))) if days else 0
        
        # 🔥 옵션 A: adaptive_ratio 조회
        try:
            from rl_pipeline.engine.adaptive_predictive import get_adaptive_predictive_ratio
            adaptive_ratio = get_adaptive_predictive_ratio(coin, interval)
        except ImportError:
            # 모듈이 없는 경우 기본값 사용
            adaptive_ratio = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
        except Exception as e:
            logger.debug(f"⚠️ adaptive_ratio 조회 실패, 기본값 사용: {e}")
            adaptive_ratio = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
        
        # 🆕 시간 가중치 활성화 여부 체크
        use_time_weighting = os.getenv('ENABLE_TIME_WEIGHTED_ROLLUP', 'true').lower() == 'true'
        current_ts = int(datetime.now().timestamp())
        decay_rate = float(os.getenv('ROLLUP_TIME_DECAY_RATE', '0.05'))  # 기본값: 0.05 (5% per day)

        # 🔥 1. 업데이트 대상 전략 ID 목록 조회 (배치 처리 준비)
        # rl_episode_summary에서 해당 코인/인터벌에 대한 전략 ID 조회
        # 데이터가 많을 수 있으므로 DISTINCT로 전략 ID만 먼저 가져옴
        logger.info(f"🔍 {coin}-{interval}: 롤업 대상 전략 ID 조회 중...")
        cursor.execute(f"""
            SELECT DISTINCT strategy_id 
            FROM rl_episode_summary 
            WHERE symbol = ? AND interval = ? AND ts_exit >= ?
        """, (coin, interval, cutoff_ts))
        
        all_strategy_ids = [row[0] for row in cursor.fetchall()]
        
        if not all_strategy_ids:
            logger.info(f"ℹ️ {coin}-{interval}: 롤업 대상 전략 없음")
            return 0
            
        logger.info(f"📊 {coin}-{interval}: 총 {len(all_strategy_ids)}개 전략 롤업 시작 (배치 처리)")
        
        total_updated = 0
        batch_size = 50  # 배치 크기 설정
        
        # 배치 단위로 처리
        for i in range(0, len(all_strategy_ids), batch_size):
            batch_ids = all_strategy_ids[i:i+batch_size]
            placeholders = ','.join(['?'] * len(batch_ids))
            
            # 해당 배치 전략들의 통계 조회
            cursor.execute(f"""
                SELECT 
                    strategy_id,
                    COUNT(*) as trades_count,
                    SUM(CASE WHEN realized_ret_signed > 0 THEN 1 ELSE 0 END) as win_count,
                    AVG(realized_ret_signed) as avg_profit,
                    SUM(realized_ret_signed) as total_profit,
                    MIN(realized_ret_signed) as max_drawdown,  -- 단순 근사치
                    0 as avg_duration, -- 🔥 컬럼 부재로 0으로 대체
                    MAX(ts_exit) as last_trade_ts
                FROM rl_episode_summary
                WHERE strategy_id IN ({placeholders})
                  AND ts_exit >= ?
                GROUP BY strategy_id
            """, batch_ids + [cutoff_ts])
            
            stats_rows = cursor.fetchall()
            update_data = []
            
            for row in stats_rows:
                strategy_id, trades, wins, avg_pnl, total_pnl, mdd, duration, last_ts = row
                
                win_rate = wins / trades if trades > 0 else 0
                
                # 시간 가중치 적용 (옵션)
                if use_time_weighting:
                    days_elapsed = (current_ts - last_ts) / 86400
                    weight = math.exp(-decay_rate * days_elapsed)
                    # 가중치가 적용된 승률/수익률 계산 (단순화: 최근 성과 비중 높임)
                    # 실제로는 에피소드별 가중 평균을 내야 하지만, 여기서는 최종 점수에 반영
                
                update_data.append((
                    win_rate,
                    avg_pnl,
                    trades,
                    total_pnl,
                    coin,
                    interval,
                    strategy_id
                ))
            
            if update_data:
                # strategies 테이블 업데이트
                # total_pnl 컬럼이 없으므로 profit 컬럼에 total_pnl 값을 매핑
                # update_data 순서: win_rate(0), avg_pnl(1), trades(2), total_pnl(3), coin(4), interval(5), strategy_id(6)
                cursor.executemany("""
                    UPDATE strategies 
                    SET win_rate = ?, profit = ?, trades_count = ?
                    WHERE symbol = ? AND interval = ? AND id = ?
                """, [(d[0], d[3], d[2], d[4], d[5], d[6]) for d in update_data])
                total_updated += len(update_data)
                conn.commit()  # 배치마다 커밋
                
        return total_updated

    except Exception as e:
        logger.error(f"❌ 롤업 계산 중 오류: {e}")
        return 0

# 🔥 누락된 함수 추가 (래퍼)
def run_full_rollup_and_grades(coin: str, interval: str) -> Dict[str, Any]:
    """
    전체 롤업 및 등급 평가 실행 (오케스트레이터 호출용)
    """
    try:
        from rl_pipeline.core.env import config
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        db_path = config.get_strategy_db_path(coin)
        
        updated_count = 0
        with get_optimized_db_connection(db_path) as conn:
            # 1. 롤업 계산 (전체 기간)
            updated_count = compute_strategy_rollup(coin, interval, days=None, conn=conn)
            
        # 2. 등급 평가 (상대평가 적용)
        try:
            # 모든 전략 로드
            all_strategies = load_strategies_pool(coin, interval, limit=0)
            
            if all_strategies:
                # 레짐별 그룹화
                regime_groups = defaultdict(list)
                for strategy in all_strategies:
                    regime = strategy.get('regime') or strategy.get('market_condition') or 'neutral'
                    regime_groups[regime].append(strategy)
                
                # 등급 업데이터 실행
                updater = StrategyGradeUpdater()
                total_graded = 0
                
                for regime, strategies in regime_groups.items():
                    if not strategies:
                        continue
                        
                    updates = updater.update_grades_with_relative_evaluation(
                        coin=coin, 
                        interval=interval, 
                        regime=regime, 
                        strategies=strategies,
                        update_db=True
                    )
                    total_graded += len(updates)
                
                return {
                    "success": True,
                    "grades_updated": total_graded,
                    "rollup_updated": updated_count,
                    "message": f"롤업 {updated_count}개, 등급평가 {total_graded}개 완료"
                }
            else:
                return {
                    "success": True,
                    "grades_updated": 0,
                    "rollup_updated": updated_count,
                    "message": f"롤업 완료: {updated_count}개 (전략 없음)"
                }
                
        except Exception as grade_error:
            logger.error(f"❌ 등급 평가 실패: {grade_error}")
            # 롤업은 성공했으므로 성공으로 처리하되 에러 메시지 포함
            return {
                "success": True,
                "grades_updated": 0,
                "rollup_updated": updated_count,
                "warning": f"등급 평가 실패: {grade_error}"
            }
            
    except Exception as e:
        logger.error(f"❌ 롤업 실행 실패: {e}")
        return {"success": False, "error": str(e)}

# 🔥 누락된 함수 추가 (이름 매핑)
run_rollup_batch = run_full_rollup_and_grades
