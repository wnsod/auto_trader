"""
하이브리드 시스템 초기화 스크립트
DB 테이블 생성 및 초기 설정
"""

import logging
import os
import sys

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rl_pipeline.db.schema import create_hybrid_policy_tables, add_hybrid_columns_to_strategies
from rl_pipeline.db.connection_pool import get_strategy_db_pool

logger = logging.getLogger(__name__)


def init_hybrid_system() -> bool:
    """
    하이브리드 시스템 초기화
    
    Returns:
        성공 여부
    """
    try:
        logger.info("🚀 하이브리드 시스템 초기화 시작")
        
        # 1. DB 테이블 생성
        logger.info("📊 하이브리드 정책 테이블 생성 중...")
        success1 = create_hybrid_policy_tables()
        
        # 2. 기존 전략 테이블에 컬럼 추가
        logger.info("📊 기존 전략 테이블에 하이브리드 컬럼 추가 중...")
        success2 = add_hybrid_columns_to_strategies()
        
        if success1 and success2:
            logger.info("✅ 하이브리드 시스템 초기화 완료")
            return True
        else:
            logger.warning("⚠️ 일부 초기화 작업 실패 (계속 진행 가능)")
            return False
            
    except Exception as e:
        logger.error(f"❌ 하이브리드 시스템 초기화 실패: {e}")
        return False


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = init_hybrid_system()
    sys.exit(0 if success else 1)

