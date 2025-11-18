"""
Phase 1 테스트: DB 스키마 확장 검증

실행 방법:
    docker exec -it auto_trader_coin bash
    cd /workspace
    python -m pytest rl_pipeline/tests/test_phase1_db_schema.py -v
    또는
    python rl_pipeline/tests/test_phase1_db_schema.py
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path

# 프로젝트 루트를 경로에 추가
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rl_pipeline.db.schema import (
    migrate_online_evolution_schema,
    create_strategy_lineage_table,
    create_segment_scores_table,
    setup_database_tables,
    create_coin_strategies_table
)
from rl_pipeline.db.connection_pool import get_strategy_db_pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_coin_strategies_columns():
    """coin_strategies 테이블에 온라인 진화 컬럼이 추가되었는지 확인"""
    logger.info("=" * 60)
    logger.info("테스트 1: coin_strategies 테이블 컬럼 확인")
    logger.info("=" * 60)
    
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 테이블 정보 조회
            cursor.execute("PRAGMA table_info(coin_strategies)")
            columns = cursor.fetchall()
            
            # 컬럼명 리스트 추출
            column_names = [col[1] for col in columns]
            
            # 필요한 컬럼들
            required_columns = [
                'parent_id',
                'version',
                'last_train_end_idx',
                'online_pf',
                'online_return',
                'online_mdd',
                'online_updates_count',
                'consistency_score'
            ]
            
            logger.info(f"✅ coin_strategies 테이블 총 컬럼 수: {len(column_names)}")
            
            missing_columns = []
            for col in required_columns:
                if col in column_names:
                    logger.info(f"  ✅ {col} 컬럼 존재")
                else:
                    logger.error(f"  ❌ {col} 컬럼 누락")
                    missing_columns.append(col)
            
            if missing_columns:
                logger.error(f"❌ 누락된 컬럼: {missing_columns}")
                return False
            
            logger.info("✅ 모든 필수 컬럼이 존재합니다")
            return True
            
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_strategy_lineage_table():
    """strategy_lineage 테이블이 생성되었는지 확인"""
    logger.info("=" * 60)
    logger.info("테스트 2: strategy_lineage 테이블 확인")
    logger.info("=" * 60)
    
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 테이블 존재 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='strategy_lineage'
            """)
            result = cursor.fetchone()
            
            if not result:
                logger.error("❌ strategy_lineage 테이블이 존재하지 않습니다")
                return False
            
            logger.info("✅ strategy_lineage 테이블 존재 확인")
            
            # 테이블 구조 확인
            cursor.execute("PRAGMA table_info(strategy_lineage)")
            columns = cursor.fetchall()
            
            logger.info(f"✅ strategy_lineage 테이블 컬럼 수: {len(columns)}")
            for col in columns:
                logger.info(f"  - {col[1]} ({col[2]})")
            
            # 인덱스 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='strategy_lineage'
            """)
            indexes = cursor.fetchall()
            
            logger.info(f"✅ 인덱스 수: {len(indexes)}")
            for idx in indexes:
                logger.info(f"  - {idx[0]}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_segment_scores_table():
    """segment_scores 테이블이 생성되었는지 확인"""
    logger.info("=" * 60)
    logger.info("테스트 3: segment_scores 테이블 확인")
    logger.info("=" * 60)
    
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 테이블 존재 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='segment_scores'
            """)
            result = cursor.fetchone()
            
            if not result:
                logger.error("❌ segment_scores 테이블이 존재하지 않습니다")
                return False
            
            logger.info("✅ segment_scores 테이블 존재 확인")
            
            # 테이블 구조 확인
            cursor.execute("PRAGMA table_info(segment_scores)")
            columns = cursor.fetchall()
            
            logger.info(f"✅ segment_scores 테이블 컬럼 수: {len(columns)}")
            for col in columns:
                logger.info(f"  - {col[1]} ({col[2]})")
            
            # 필수 컬럼 확인
            column_names = [col[1] for col in columns]
            required_columns = ['market', 'interval', 'start_timestamp', 'end_timestamp']
            
            for col in required_columns:
                if col in column_names:
                    logger.info(f"  ✅ {col} 컬럼 존재")
                else:
                    logger.error(f"  ❌ {col} 컬럼 누락")
                    return False
            
            # 인덱스 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='segment_scores'
            """)
            indexes = cursor.fetchall()
            
            logger.info(f"✅ 인덱스 수: {len(indexes)}")
            for idx in indexes:
                logger.info(f"  - {idx[0]}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_insert_sample_data():
    """새 테이블에 샘플 데이터 삽입 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 4: 샘플 데이터 삽입 테스트")
    logger.info("=" * 60)
    
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. coin_strategies에 테스트 전략 생성 (이미 있으면 스킵)
            cursor.execute("""
                SELECT id FROM coin_strategies 
                WHERE id = 'test_strategy_001'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO coin_strategies (
                        id, coin, interval, parent_id, version,
                        online_pf, online_return, consistency_score
                    ) VALUES (
                        'test_strategy_001', 'BTC', '15m', NULL, 1,
                        1.5, 0.1, 0.8
                    )
                """)
                logger.info("✅ 테스트 전략 생성")
            
            # 2. strategy_lineage에 테스트 데이터 삽입
            cursor.execute("""
                SELECT child_id FROM strategy_lineage 
                WHERE child_id = 'test_strategy_002'
            """)
            if not cursor.fetchone():
                # 자식 전략 생성
                cursor.execute("""
                    INSERT INTO coin_strategies (
                        id, coin, interval, parent_id, version
                    ) VALUES (
                        'test_strategy_002', 'BTC', '15m', 'test_strategy_001', 2
                    )
                """)
                
                cursor.execute("""
                    INSERT INTO strategy_lineage (
                        child_id, parent_id, mutation_desc, improvement_flag
                    ) VALUES (
                        'test_strategy_002', 'test_strategy_001', 
                        'rsi_min: 30->32', 1
                    )
                """)
                logger.info("✅ strategy_lineage 테스트 데이터 삽입")
            
            # 3. segment_scores에 테스트 데이터 삽입
            cursor.execute("""
                SELECT id FROM segment_scores 
                WHERE strategy_id = 'test_strategy_001' AND start_idx = 0
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO segment_scores (
                        strategy_id, market, interval,
                        start_idx, end_idx, start_timestamp, end_timestamp,
                        profit, pf, sharpe, mdd, trades_count
                    ) VALUES (
                        'test_strategy_001', 'BTC', '15m',
                        0, 1000, 1699000000, 1699001000,
                        100.0, 1.5, 2.0, 0.05, 25
                    )
                """)
                logger.info("✅ segment_scores 테스트 데이터 삽입")
            
            conn.commit()
            logger.info("✅ 모든 샘플 데이터 삽입 완료")
            
            # 조회 테스트
            cursor.execute("""
                SELECT COUNT(*) FROM segment_scores 
                WHERE strategy_id = 'test_strategy_001'
            """)
            count = cursor.fetchone()[0]
            logger.info(f"✅ segment_scores 조회 성공: {count}개 레코드")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("=" * 60)
    logger.info("Phase 1: DB 스키마 확장 테스트 시작")
    logger.info("=" * 60)
    
    # 먼저 기본 테이블 생성 (없을 경우)
    logger.info("\n🔄 기본 테이블 생성 중...")
    try:
        # coin_strategies 테이블이 없으면 생성
        create_coin_strategies_table()
        logger.info("✅ 기본 테이블 생성 완료")
    except Exception as e:
        logger.warning(f"⚠️ 기본 테이블 생성 중 오류 (무시 가능): {e}")
    
    # 마이그레이션 실행
    logger.info("\n🔄 마이그레이션 실행 중...")
    success = migrate_online_evolution_schema()
    
    if not success:
        logger.error("❌ 마이그레이션 실패")
        return False
    
    logger.info("✅ 마이그레이션 완료\n")
    
    # 테스트 실행
    tests = [
        ("coin_strategies 컬럼 확인", test_coin_strategies_columns),
        ("strategy_lineage 테이블 확인", test_strategy_lineage_table),
        ("segment_scores 테이블 확인", test_segment_scores_table),
        ("샘플 데이터 삽입", test_insert_sample_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n▶ {test_name} 테스트 실행...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} 통과\n")
            else:
                logger.error(f"❌ {test_name} 실패\n")
        except Exception as e:
            logger.error(f"❌ {test_name} 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        logger.info("=" * 60)
        logger.info("🎉 Phase 1 테스트 모두 통과!")
        logger.info("=" * 60)
        return True
    else:
        logger.error("=" * 60)
        logger.error("❌ 일부 테스트 실패")
        logger.error("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

