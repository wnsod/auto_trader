"""
전략 필터링 모듈
- F 등급 전략 제거
- 성과 기준 필터링
- 상위 전략만 유지
"""

import sqlite3
import argparse
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def remove_low_grade_strategies(db_path: str = "data_storage/rl_strategies.db") -> int:
    """낮은 등급 또는 등급이 없는 전략 제거
    
    삭제 대상:
    - quality_grade = 'F' (F 등급)
    - quality_grade IS NULL (등급 없음)
    - quality_grade = 'UNKNOWN' (알 수 없음)
    """
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 삭제 대상 전략 조회 (F 등급, NULL, UNKNOWN)
        cursor.execute("""
            SELECT id, coin, interval, quality_grade
            FROM coin_strategies
            WHERE quality_grade = 'F' 
               OR quality_grade IS NULL 
               OR quality_grade = 'UNKNOWN'
        """)
        low_grade_strategies = cursor.fetchall()
        
        # 등급별 통계
        grade_stats = {}
        for row in low_grade_strategies:
            grade = row[3] if row[3] else 'NULL'
            grade_stats[grade] = grade_stats.get(grade, 0) + 1
        
        logger.info(f"발견된 낮은 등급/등급 없음 전략: {len(low_grade_strategies)}개")
        for grade, count in grade_stats.items():
            logger.info(f"  - {grade}: {count}개")
        
        if not low_grade_strategies:
            logger.info("제거할 낮은 등급 전략이 없습니다.")
            return 0
        
        # 백업 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coin_strategies_backup AS
            SELECT * FROM coin_strategies 
            WHERE quality_grade = 'F' 
               OR quality_grade IS NULL 
               OR quality_grade = 'UNKNOWN'
        """)
        
        # 삭제 대상 전략 ID 목록
        low_grade_ids = [row[0] for row in low_grade_strategies]
        
        # 관련 데이터도 삭제
        if low_grade_ids:
            placeholders = ','.join(['?' for _ in low_grade_ids])
            cursor.execute(f"""
                DELETE FROM selfplay_results
                WHERE strategy_id IN ({placeholders})
            """, low_grade_ids)
        
        # 낮은 등급/등급 없음 전략 삭제
        cursor.execute("""
            DELETE FROM coin_strategies 
            WHERE quality_grade = 'F' 
               OR quality_grade IS NULL 
               OR quality_grade = 'UNKNOWN'
        """)
        deleted_count = cursor.rowcount
        
        conn.commit()
        
        logger.info(f"✅ {deleted_count}개 낮은 등급/등급 없음 전략 제거 완료")
        return deleted_count
    
    except Exception as e:
        logger.error(f"❌ 낮은 등급 전략 제거 실패: {e}")
        conn.rollback()
        return 0
    
    finally:
        conn.close()


def filter_by_performance(
    db_path: str = "data_storage/rl_strategies.db",
    min_profit_factor: float = 1.2,
    min_win_rate: float = 0.55,
    min_sharpe: float = 0.5
) -> int:
    """성과 기준으로 전략 필터링"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 조건에 맞지 않는 전략 조회
        cursor.execute("""
            SELECT COUNT(*) FROM coin_strategies
            WHERE profit_factor < ? OR profit_factor IS NULL
               OR win_rate < ? OR win_rate IS NULL
               OR sharpe_ratio < ? OR sharpe_ratio IS NULL
        """, (min_profit_factor, min_win_rate, min_sharpe))
        
        low_perf_count = cursor.fetchone()[0]
        
        logger.info(f"성과 기준 미달 전략: {low_perf_count}개")
        
        if low_perf_count == 0:
            logger.info("제거할 저성과 전략이 없습니다.")
            return 0
        
        # 백업 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coin_strategies_low_perf AS
            SELECT * FROM coin_strategies
            WHERE profit_factor < ? OR profit_factor IS NULL
               OR win_rate < ? OR win_rate IS NULL
               OR sharpe_ratio < ? OR sharpe_ratio IS NULL
        """, (min_profit_factor, min_win_rate, min_sharpe))
        
        # 저성과 전략 ID 목록
        cursor.execute("""
            SELECT id FROM coin_strategies
            WHERE profit_factor < ? OR profit_factor IS NULL
               OR win_rate < ? OR win_rate IS NULL
               OR sharpe_ratio < ? OR sharpe_ratio IS NULL
        """, (min_profit_factor, min_win_rate, min_sharpe))
        
        low_perf_ids = [row[0] for row in cursor.fetchall()]
        
        # 관련 데이터 삭제
        if low_perf_ids:
            placeholders = ','.join(['?' for _ in low_perf_ids])
            cursor.execute(f"""
                DELETE FROM selfplay_results
                WHERE strategy_id IN ({placeholders})
            """, low_perf_ids)
        
        # 저성과 전략 삭제
        cursor.execute("""
            DELETE FROM coin_strategies
            WHERE profit_factor < ? OR profit_factor IS NULL
               OR win_rate < ? OR win_rate IS NULL
               OR sharpe_ratio < ? OR sharpe_ratio IS NULL
        """, (min_profit_factor, min_win_rate, min_sharpe))
        
        conn.commit()
        
        logger.info(f"✅ {low_perf_count}개 저성과 전략 제거 완료")
        return low_perf_count
    
    except Exception as e:
        logger.error(f"❌ 저성과 전략 제거 실패: {e}")
        conn.rollback()
        return 0
    
    finally:
        conn.close()


def keep_top_strategies(
    db_path: str = "data_storage/rl_strategies.db",
    top_percent: float = 0.1  # 상위 10%
) -> Tuple[int, int]:
    """각 코인/인터벌별 상위 전략만 유지"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_kept = 0
    total_removed = 0
    
    try:
        # 코인/인터벌 조합 가져오기
        cursor.execute("SELECT DISTINCT coin, interval FROM coin_strategies")
        combinations = cursor.fetchall()
        
        if not combinations:
            logger.warning("전략이 없습니다.")
            return 0, 0
        
        for coin, interval in combinations:
            # 전체 전략 수
            cursor.execute("""
                SELECT COUNT(*) FROM coin_strategies
                WHERE coin = ? AND interval = ?
            """, (coin, interval))
            total_count = cursor.fetchone()[0]
            
            keep_count = max(5, int(total_count * top_percent))  # 최소 5개
            
            # 상위 전략 ID 가져오기
            cursor.execute("""
                SELECT id FROM coin_strategies
                WHERE coin = ? AND interval = ?
                ORDER BY
                    COALESCE(profit_factor, 0) DESC,
                    COALESCE(sharpe_ratio, 0) DESC,
                    COALESCE(win_rate, 0) DESC
                LIMIT ?
            """, (coin, interval, keep_count))
            
            keep_ids = [row[0] for row in cursor.fetchall()]
            
            if not keep_ids:
                continue
            
            # 나머지 삭제
            placeholders = ','.join(['?' for _ in keep_ids])
            cursor.execute(f"""
                DELETE FROM coin_strategies
                WHERE coin = ? AND interval = ?
                  AND id NOT IN ({placeholders})
            """, (coin, interval, *keep_ids))
            
            removed = cursor.rowcount
            total_kept += len(keep_ids)
            total_removed += removed
            
            logger.info(f"  {coin}-{interval}: {len(keep_ids)}개 유지, {removed}개 제거")
        
        conn.commit()
        
        logger.info(f"\n✅ 총 {total_kept}개 전략 유지, {total_removed}개 제거")
        return total_kept, total_removed
    
    except Exception as e:
        logger.error(f"❌ 상위 전략 선별 실패: {e}")
        conn.rollback()
        return 0, 0
    
    finally:
        conn.close()


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description='전략 필터링')
    parser.add_argument('--action', choices=['remove_f_grade', 'filter_performance', 'keep_top'],
                       required=True, help='실행할 액션')
    parser.add_argument('--db_path', default='data_storage/rl_strategies.db',
                       help='데이터베이스 경로')
    parser.add_argument('--min_pf', type=float, default=1.2,
                       help='최소 Profit Factor')
    parser.add_argument('--min_wr', type=float, default=0.55,
                       help='최소 Win Rate')
    parser.add_argument('--min_sharpe', type=float, default=0.5,
                       help='최소 Sharpe Ratio')
    parser.add_argument('--top_percent', type=float, default=0.1,
                       help='유지할 상위 비율')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.action == 'remove_f_grade':
        remove_low_grade_strategies(args.db_path)
    elif args.action == 'filter_performance':
        filter_by_performance(args.db_path, args.min_pf, args.min_wr, args.min_sharpe)
    elif args.action == 'keep_top':
        keep_top_strategies(args.db_path, args.top_percent)


if __name__ == "__main__":
    main()

