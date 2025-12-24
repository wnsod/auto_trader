"""Phase 1 검증 스크립트"""
import sqlite3
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from rl_pipeline.core.env import Config

config = Config()
DB_PATH = config.STRATEGIES_DB

def verify_phase1():
    print("\n" + "="*80)
    print("Phase 1 검증 시작")
    print("="*80)
    print(f"📁 DB 경로: {DB_PATH}")

    if not os.path.exists(DB_PATH):
        print(f"❌ DB 파일 없음: {DB_PATH}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 뷰 확인
    print("\n1. v_active_strategies 뷰 확인...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='v_active_strategies'")
    view_result = cursor.fetchone()
    if view_result:
        print(f"   ✅ 뷰 존재: {view_result[0]}")

        # 뷰 테스트 쿼리
        try:
            cursor.execute("SELECT COUNT(*) FROM v_active_strategies")
            count = cursor.fetchone()[0]
            print(f"   ✅ 활성 전략 수: {count:,}개")
        except Exception as e:
            print(f"   ❌ 뷰 조회 실패: {e}")
    else:
        print("   ❌ 뷰 없음 - setup_database_tables() 실행 필요")

    # 2. RL 에피소드 수 확인
    print("\n2. RL 에피소드 현황...")
    try:
        cursor.execute("SELECT COUNT(*) FROM rl_episodes")
        episodes_count = cursor.fetchone()[0]
        print(f"   📊 총 에피소드 수: {episodes_count:,}개")

        # 전략별 에피소드 수
        cursor.execute("""
            SELECT COUNT(*) as strategy_count
            FROM (
                SELECT strategy_id, COUNT(*) as episode_count
                FROM rl_episodes
                GROUP BY coin, interval, strategy_id
                HAVING episode_count > 10000
            )
        """)
        over_limit = cursor.fetchone()[0]
        print(f"   ⚠️ 10,000개 초과 전략: {over_limit:,}개")

        if over_limit > 0:
            print(f"   💡 Pruning 권장: python rl_pipeline/tools/prune_rl_episodes.py --max-episodes-per-strategy 10000")
    except Exception as e:
        print(f"   ❌ 에피소드 확인 실패: {e}")

    # 3. 테이블/뷰 개수
    print("\n3. DB 구조...")
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'")
    view_count = cursor.fetchone()[0]
    print(f"   📊 테이블 수: {table_count}개")
    print(f"   📊 뷰 수: {view_count}개")

    # 4. 주요 테이블 row 수
    print("\n4. 주요 테이블 데이터 현황...")
    tables_to_check = [
        "strategies",
        "rl_episodes",
        "rl_episode_summary",
        "rl_strategy_rollup",
        "strategy_grades",
        "global_strategies",
        "coin_global_weights"
    ]

    for table in tables_to_check:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            status = "✅" if count > 0 else "⚠️ "
            print(f"   {status} {table}: {count:,}개")
        except sqlite3.OperationalError:
            print(f"   ❌ {table}: 테이블 없음")

    conn.close()

    print("\n" + "="*80)
    print("Phase 1 검증 완료")
    print("="*80)
    print("\n💡 다음 단계:")
    print("   1. RL 에피소드 Pruning (선택): python rl_pipeline/tools/prune_rl_episodes.py --dry-run")
    print("   2. Phase 2 진행: Source of Truth 통일")
    print()

    return True

if __name__ == "__main__":
    verify_phase1()
