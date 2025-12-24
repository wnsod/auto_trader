"""Phase 1 DB 초기화 스크립트"""
import sys
import os
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/rl_pipeline')

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from rl_pipeline.db.schema import setup_database_tables

def main():
    print("\n" + "="*80)
    print("Phase 1 DB 초기화")
    print("="*80)

    try:
        # DB 초기화 (뷰 포함)
        result = setup_database_tables()

        if result:
            print("\n✅ DB 초기화 성공 (v_active_strategies 뷰 포함)")
            print("\n💡 검증 실행:")
            print("   python /workspace/rl_pipeline/tools/verify_phase1.py")
        else:
            print("\n❌ DB 초기화 실패")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
