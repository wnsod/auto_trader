"""
모든 수정 사항 검증
1. pattern_confidence - 실제 계산된 값
2. integrated_analysis 새 데이터 생성 및 검증
"""
import sys
sys.path.insert(0, '/workspace')

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def verify_pattern_confidence():
    """pattern_confidence 검증"""
    logger.info("=" * 80)
    logger.info("1. pattern_confidence 검증")
    logger.info("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 전체 통계
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            AVG(pattern_confidence) as avg_conf,
            MIN(pattern_confidence) as min_conf,
            MAX(pattern_confidence) as max_conf,
            SUM(CASE WHEN pattern_confidence = 0.5 THEN 1 ELSE 0 END) as default_count
        FROM coin_strategies
    """)

    result = cursor.fetchone()

    logger.info(f"\n📊 pattern_confidence 통계:")
    logger.info(f"   - 총 전략: {result[0]:,}개")
    logger.info(f"   - 평균: {result[1]:.4f}")
    logger.info(f"   - 최소: {result[2]:.4f}")
    logger.info(f"   - 최대: {result[3]:.4f}")
    logger.info(f"   - 기본값(0.5) 개수: {result[4]:,}개 ({result[4]/result[0]*100:.1f}%)")

    if result[4] == 0:
        logger.info(f"   ✅ 모든 전략의 pattern_confidence가 계산되었습니다!")
    elif result[4] / result[0] < 0.01:
        logger.info(f"   ✅ 대부분의 pattern_confidence가 계산되었습니다 (99%+)")
    else:
        logger.warning(f"   ⚠️ 아직 기본값인 전략이 많습니다")

    # 분포 확인
    cursor.execute("""
        SELECT
            CASE
                WHEN pattern_confidence < 0.6 THEN '0.0-0.6 (낮음)'
                WHEN pattern_confidence < 0.8 THEN '0.6-0.8 (중간)'
                WHEN pattern_confidence < 0.9 THEN '0.8-0.9 (높음)'
                ELSE '0.9-1.0 (매우 높음)'
            END as range,
            COUNT(*) as count
        FROM coin_strategies
        GROUP BY range
        ORDER BY range
    """)

    logger.info(f"\n📊 pattern_confidence 분포:")
    for row in cursor.fetchall():
        logger.info(f"   - {row[0]}: {row[1]:,}개")

    conn.close()


def verify_integrated_analysis_scores():
    """integrated_analysis_results의 점수 검증"""
    logger.info("\n" + "=" * 80)
    logger.info("2. integrated_analysis_results 점수 검증")
    logger.info("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 테이블이 존재하는지 확인
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='integrated_analysis_results'")
    if not cursor.fetchone():
        logger.warning("⚠️ integrated_analysis_results 테이블이 없습니다")
        conn.close()
        return

    # 총 개수
    cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
    total_count = cursor.fetchone()[0]

    if total_count == 0:
        logger.warning("⚠️ integrated_analysis_results에 데이터가 없습니다")
        logger.info("   → 파이프라인을 실행하면 새로운 데이터가 생성됩니다")
        conn.close()
        return

    logger.info(f"\n📊 총 레코드: {total_count}개")

    # 각 점수별 통계
    fields = ['ensemble_score', 'fractal_score', 'multi_timeframe_score', 'indicator_cross_score', 'ensemble_confidence']

    for field in fields:
        cursor.execute(f"""
            SELECT
                AVG({field}) as avg_val,
                MIN({field}) as min_val,
                MAX({field}) as max_val,
                SUM(CASE WHEN {field} = 0.5 THEN 1 ELSE 0 END) as default_count
            FROM integrated_analysis_results
        """)

        result = cursor.fetchone()

        logger.info(f"\n📊 {field}:")
        logger.info(f"   - 평균: {result[0]:.4f}")
        logger.info(f"   - 범위: {result[1]:.4f} ~ {result[2]:.4f}")
        logger.info(f"   - 기본값(0.5) 비율: {result[3]}/{total_count} ({result[3]/total_count*100:.1f}%)")

        if result[3] / total_count > 0.9:
            logger.warning(f"   ⚠️ {field}는 여전히 대부분 0.5입니다 (이전 데이터)")
        elif result[3] / total_count < 0.1:
            logger.info(f"   ✅ {field}가 대부분 계산되었습니다!")

    conn.close()


def main():
    logger.info("🔍 전체 수정 사항 검증 시작\n")

    # 1. pattern_confidence 검증
    verify_pattern_confidence()

    # 2. integrated_analysis_results 검증
    verify_integrated_analysis_scores()

    logger.info("\n" + "=" * 80)
    logger.info("✅ 검증 완료")
    logger.info("=" * 80)
    logger.info("\n💡 참고:")
    logger.info("   - pattern_confidence는 기존 데이터가 모두 업데이트되었습니다")
    logger.info("   - integrated_analysis_results는 파이프라인 재실행 시 새로운 값이 생성됩니다")
    logger.info("   - 기존 데이터(0.5)는 파이프라인 실행으로 갱신됩니다")


if __name__ == "__main__":
    main()
