"""
rl_strategies.db 데이터 검증 스크립트
모든 테이블의 데이터 품질, 일관성, 이상치를 정밀 검증
"""
import sqlite3
import json
from datetime import datetime
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def validate_database():
    """데이터베이스 전체 검증"""

    logger.info("=" * 80)
    logger.info("🔍 RL_STRATEGIES.DB 데이터 검증 시작")
    logger.info("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'db_path': DB_PATH,
        'tables': {},
        'issues': [],
        'warnings': [],
        'summary': {}
    }

    # 1. 전체 테이블 목록 및 기본 통계
    logger.info("\n📊 1. 전체 테이블 목록 및 레코드 수")
    logger.info("-" * 80)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    total_records = 0
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        total_records += count

        logger.info(f"  {'✅' if count > 0 else '⚠️ '} {table:40} {count:>10,}개")

        validation_results['tables'][table] = {
            'record_count': count,
            'issues': [],
            'warnings': []
        }

        if count == 0 and table not in ['sqlite_sequence']:
            validation_results['warnings'].append(f"{table} 테이블이 비어있음")

    logger.info("-" * 80)
    logger.info(f"  총 테이블 수: {len(tables)}개")
    logger.info(f"  총 레코드 수: {total_records:,}개")

    # 2. 주요 테이블 상세 검증
    logger.info("\n📋 2. 주요 테이블 상세 검증")
    logger.info("=" * 80)

    # 2-1. coin_strategies 검증
    validate_coin_strategies(cursor, validation_results)

    # 2-2. integrated_analysis_results 검증
    validate_integrated_analysis(cursor, validation_results)

    # 2-3. selfplay_evolution_results 검증
    validate_selfplay_results(cursor, validation_results)

    # 2-4. global_strategies 검증
    validate_global_strategies(cursor, validation_results)

    # 2-5. paper_trading_sessions 검증
    validate_paper_trading(cursor, validation_results)

    # 2-6. rl_episodes 검증
    validate_rl_episodes(cursor, validation_results)

    # 2-7. strategy_grades 검증
    validate_strategy_grades(cursor, validation_results)

    # 3. 데이터 일관성 검증
    logger.info("\n🔗 3. 데이터 일관성 검증")
    logger.info("=" * 80)
    validate_data_consistency(cursor, validation_results)

    # 4. 최종 요약
    logger.info("\n📊 4. 검증 요약")
    logger.info("=" * 80)
    print_validation_summary(validation_results)

    conn.close()

    # 결과 저장
    with open('/workspace/db_validation_report.json', 'w') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ 검증 완료! 상세 보고서: /workspace/db_validation_report.json")

    return validation_results


def validate_coin_strategies(cursor, results):
    """coin_strategies 테이블 검증"""
    logger.info("\n2-1. coin_strategies 테이블")
    logger.info("-" * 80)

    table = 'coin_strategies'
    table_results = results['tables'][table]

    # 기본 통계
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    # 코인별 통계
    cursor.execute(f"""
        SELECT coin, COUNT(*) as cnt
        FROM {table}
        GROUP BY coin
        ORDER BY cnt DESC
    """)
    coin_stats = cursor.fetchall()
    logger.info(f"  📊 코인별 전략 수:")
    for row in coin_stats[:10]:  # 상위 10개
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")

    # 인터벌별 통계
    cursor.execute(f"""
        SELECT interval, COUNT(*) as cnt
        FROM {table}
        GROUP BY interval
        ORDER BY cnt DESC
    """)
    interval_stats = cursor.fetchall()
    logger.info(f"  📊 인터벌별 전략 수:")
    for row in interval_stats:
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")

    # 성능 데이터 통계
    cursor.execute(f"""
        SELECT
            AVG(profit) as avg_profit,
            MIN(profit) as min_profit,
            MAX(profit) as max_profit,
            AVG(win_rate) as avg_win_rate,
            COUNT(CASE WHEN profit IS NULL THEN 1 END) as null_profit,
            COUNT(CASE WHEN win_rate IS NULL THEN 1 END) as null_win_rate
        FROM {table}
    """)
    perf = cursor.fetchone()

    logger.info(f"  💰 성능 통계:")
    logger.info(f"     - 평균 수익: {perf[0]:.2f}" if perf[0] else "     - 평균 수익: N/A")
    logger.info(f"     - 수익 범위: {perf[1]:.2f} ~ {perf[2]:.2f}" if perf[1] else "     - 수익 범위: N/A")
    logger.info(f"     - 평균 승률: {perf[3]:.4f}" if perf[3] else "     - 평균 승률: N/A")
    logger.info(f"     - Profit NULL: {perf[4]:,}개")
    logger.info(f"     - Win Rate NULL: {perf[5]:,}개")

    # 이상치 탐지
    issues = []

    # 비정상적인 수익률 (±100% 초과)
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE ABS(profit) > 10000
    """)
    abnormal_profit = cursor.fetchone()[0]
    if abnormal_profit > 0:
        issues.append(f"비정상적인 수익률 (±100% 초과): {abnormal_profit}개")

    # 승률이 0~1 범위 밖
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE win_rate < 0 OR win_rate > 1
    """)
    invalid_win_rate = cursor.fetchone()[0]
    if invalid_win_rate > 0:
        issues.append(f"잘못된 승률 범위: {invalid_win_rate}개")

    # 날짜 범위
    cursor.execute(f"SELECT MIN(created_at), MAX(created_at) FROM {table}")
    date_range = cursor.fetchone()
    logger.info(f"  📅 생성 날짜 범위: {date_range[0]} ~ {date_range[1]}")

    table_results['statistics'] = {
        'coin_count': len(coin_stats),
        'interval_count': len(interval_stats),
        'avg_profit': float(perf[0]) if perf[0] else None,
        'avg_win_rate': float(perf[3]) if perf[3] else None
    }

    if issues:
        logger.info(f"  ⚠️  이슈: {len(issues)}개")
        for issue in issues:
            logger.info(f"     - {issue}")
        table_results['issues'].extend(issues)


def validate_integrated_analysis(cursor, results):
    """integrated_analysis_results 테이블 검증"""
    logger.info("\n2-2. integrated_analysis_results 테이블")
    logger.info("-" * 80)

    table = 'integrated_analysis_results'
    table_results = results['tables'][table]

    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    # 코인/인터벌별 통계
    cursor.execute(f"""
        SELECT coin, interval, COUNT(*) as cnt
        FROM {table}
        GROUP BY coin, interval
        ORDER BY coin, interval
    """)
    stats = cursor.fetchall()
    logger.info(f"  📊 코인/인터벌별 분석 결과 수: {len(stats)}개 조합")

    # 시그널 액션 분포
    cursor.execute(f"""
        SELECT signal_action, COUNT(*) as cnt
        FROM {table}
        GROUP BY signal_action
        ORDER BY cnt DESC
    """)
    signal_stats = cursor.fetchall()
    logger.info(f"  📊 시그널 분포:")
    for row in signal_stats:
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")

    # 점수 통계
    cursor.execute(f"""
        SELECT
            AVG(final_signal_score) as avg_score,
            MIN(final_signal_score) as min_score,
            MAX(final_signal_score) as max_score,
            AVG(signal_confidence) as avg_confidence
        FROM {table}
    """)
    score_stats = cursor.fetchone()

    logger.info(f"  📊 점수 통계:")
    logger.info(f"     - 평균 시그널 점수: {score_stats[0]:.4f}" if score_stats[0] else "     - 평균 시그널 점수: N/A")
    logger.info(f"     - 점수 범위: {score_stats[1]:.4f} ~ {score_stats[2]:.4f}" if score_stats[1] else "     - 점수 범위: N/A")
    logger.info(f"     - 평균 신뢰도: {score_stats[3]:.4f}" if score_stats[3] else "     - 평균 신뢰도: N/A")

    # 날짜 범위
    cursor.execute(f"SELECT MIN(created_at), MAX(created_at) FROM {table}")
    date_range = cursor.fetchone()
    logger.info(f"  📅 생성 날짜 범위: {date_range[0]} ~ {date_range[1]}")

    # 이상치 체크
    issues = []

    # 점수가 0~1 범위 밖
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE final_signal_score < 0 OR final_signal_score > 1
    """)
    invalid_score = cursor.fetchone()[0]
    if invalid_score > 0:
        issues.append(f"잘못된 시그널 점수 범위: {invalid_score}개")

    if issues:
        logger.info(f"  ⚠️  이슈: {len(issues)}개")
        for issue in issues:
            logger.info(f"     - {issue}")
        table_results['issues'].extend(issues)


def validate_selfplay_results(cursor, results):
    """selfplay_evolution_results 테이블 검증"""
    logger.info("\n2-3. selfplay_evolution_results 테이블")
    logger.info("-" * 80)

    table = 'selfplay_evolution_results'

    # 테이블 존재 여부 확인
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
    if not cursor.fetchone():
        logger.info(f"  ℹ️  테이블이 존재하지 않음")
        return

    table_results = results['tables'].get(table, {})

    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    logger.info(f"  📊 총 Self-play 결과: {total:,}개")

    # 코인별 통계
    cursor.execute(f"""
        SELECT coin, COUNT(*) as cnt
        FROM {table}
        GROUP BY coin
        ORDER BY cnt DESC LIMIT 10
    """)
    coin_stats = cursor.fetchall()
    logger.info(f"  📊 코인별 Self-play 수:")
    for row in coin_stats:
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")


def validate_global_strategies(cursor, results):
    """global_strategies 테이블 검증"""
    logger.info("\n2-4. global_strategies 테이블")
    logger.info("-" * 80)

    table = 'global_strategies'
    table_results = results['tables'][table]

    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    logger.info(f"  📊 총 글로벌 전략: {total:,}개")

    # 인터벌별 통계
    cursor.execute(f"""
        SELECT interval, COUNT(*) as cnt
        FROM {table}
        GROUP BY interval
        ORDER BY cnt DESC
    """)
    interval_stats = cursor.fetchall()
    logger.info(f"  📊 인터벌별 글로벌 전략:")
    for row in interval_stats:
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")

    # 날짜 범위
    cursor.execute(f"SELECT MIN(created_at), MAX(created_at) FROM {table}")
    date_range = cursor.fetchone()
    logger.info(f"  📅 생성 날짜 범위: {date_range[0]} ~ {date_range[1]}")


def validate_paper_trading(cursor, results):
    """paper_trading_sessions 테이블 검증"""
    logger.info("\n2-5. paper_trading_sessions 테이블")
    logger.info("-" * 80)

    table = 'paper_trading_sessions'
    table_results = results['tables'][table]

    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    logger.info(f"  📊 총 Paper Trading 세션: {total:,}개")

    # 상태별 통계
    cursor.execute(f"""
        SELECT status, COUNT(*) as cnt
        FROM {table}
        GROUP BY status
    """)
    status_stats = cursor.fetchall()
    logger.info(f"  📊 세션 상태:")
    for row in status_stats:
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")

    # 거래 통계 (paper_trading_trades 테이블과 조인)
    cursor.execute("""
        SELECT COUNT(*) as total_trades
        FROM paper_trading_trades
    """)
    trade_count = cursor.fetchone()[0]

    # 세션별 수익률 통계
    cursor.execute(f"""
        SELECT
            AVG((current_capital - initial_capital) / initial_capital * 100) as avg_profit_pct,
            MIN((current_capital - initial_capital) / initial_capital * 100) as min_profit_pct,
            MAX((current_capital - initial_capital) / initial_capital * 100) as max_profit_pct
        FROM {table}
        WHERE initial_capital > 0
    """)
    profit_stats = cursor.fetchone()

    logger.info(f"  📊 거래 통계:")
    logger.info(f"     - 총 거래 수: {trade_count:,}개")
    if profit_stats[0] is not None:
        logger.info(f"     - 평균 수익률: {profit_stats[0]:.2f}%")
        logger.info(f"     - 수익률 범위: {profit_stats[1]:.2f}% ~ {profit_stats[2]:.2f}%")


def validate_rl_episodes(cursor, results):
    """rl_episodes 테이블 검증"""
    logger.info("\n2-6. rl_episodes 테이블")
    logger.info("-" * 80)

    table = 'rl_episodes'
    table_results = results['tables'][table]

    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    logger.info(f"  📊 총 RL 에피소드: {total:,}개")

    # 코인/인터벌별 통계
    cursor.execute(f"""
        SELECT coin, interval, COUNT(*) as cnt
        FROM {table}
        GROUP BY coin, interval
        ORDER BY cnt DESC LIMIT 10
    """)
    stats = cursor.fetchall()
    logger.info(f"  📊 코인/인터벌별 에피소드 (상위 10개):")
    for row in stats:
        logger.info(f"     - {row[0]}-{row[1]:5} {row[2]:>6,}개")

    # 보상 통계 (rl_episode_summary에서 조회)
    cursor.execute("""
        SELECT
            AVG(total_reward) as avg_reward,
            MIN(total_reward) as min_reward,
            MAX(total_reward) as max_reward,
            COUNT(*) as cnt_with_reward
        FROM rl_episode_summary
        WHERE total_reward IS NOT NULL
    """)
    reward_stats = cursor.fetchone()

    if reward_stats[0] is not None:
        logger.info(f"  📊 보상 통계 (rl_episode_summary):")
        logger.info(f"     - 보상 레코드 수: {reward_stats[3]:,}개")
        logger.info(f"     - 평균 보상: {reward_stats[0]:.4f}")
        logger.info(f"     - 보상 범위: {reward_stats[1]:.4f} ~ {reward_stats[2]:.4f}")


def validate_strategy_grades(cursor, results):
    """strategy_grades 테이블 검증"""
    logger.info("\n2-7. strategy_grades 테이블")
    logger.info("-" * 80)

    table = 'strategy_grades'
    table_results = results['tables'][table]

    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]

    if total == 0:
        logger.info("  ⚠️  테이블이 비어있음")
        return

    logger.info(f"  📊 총 전략 등급: {total:,}개")

    # 등급별 분포
    cursor.execute(f"""
        SELECT grade, COUNT(*) as cnt
        FROM {table}
        GROUP BY grade
        ORDER BY grade
    """)
    grade_stats = cursor.fetchall()
    logger.info(f"  📊 등급별 분포:")
    for row in grade_stats:
        logger.info(f"     - {row[0]:10} {row[1]:>6,}개")


def validate_data_consistency(cursor, results):
    """데이터 일관성 검증"""

    issues = []

    # 1. coin_strategies와 integrated_analysis 코인 일치 확인
    cursor.execute("""
        SELECT DISTINCT coin FROM coin_strategies
    """)
    strategy_coins = set(row[0] for row in cursor.fetchall())

    cursor.execute("""
        SELECT DISTINCT coin FROM integrated_analysis_results
    """)
    analysis_coins = set(row[0] for row in cursor.fetchall())

    missing_analysis = strategy_coins - analysis_coins
    if missing_analysis:
        issue = f"전략은 있으나 통합 분석 결과가 없는 코인: {missing_analysis}"
        logger.info(f"  ⚠️  {issue}")
        issues.append(issue)

    extra_analysis = analysis_coins - strategy_coins
    if extra_analysis:
        issue = f"통합 분석 결과만 있고 전략이 없는 코인: {extra_analysis}"
        logger.info(f"  ⚠️  {issue}")
        issues.append(issue)

    # 2. NULL 값 체크
    for table in ['coin_strategies', 'integrated_analysis_results']:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()

        for col in columns:
            col_name = col[1]
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL")
            null_count = cursor.fetchone()[0]

            if null_count > 0:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                total = cursor.fetchone()[0]
                null_pct = (null_count / total * 100) if total > 0 else 0

                if null_pct > 50:  # 50% 이상 NULL이면 경고
                    issue = f"{table}.{col_name}: {null_pct:.1f}% NULL ({null_count}/{total})"
                    logger.info(f"  ⚠️  {issue}")
                    results['warnings'].append(issue)

    if not issues:
        logger.info(f"  ✅ 데이터 일관성 검증 통과")

    results['consistency_check'] = {
        'strategy_coins': list(strategy_coins),
        'analysis_coins': list(analysis_coins),
        'issues': issues
    }


def print_validation_summary(results):
    """검증 요약 출력"""

    total_issues = len(results['issues'])
    total_warnings = len(results['warnings'])

    # 테이블별 이슈 집계
    table_issues = sum(len(t.get('issues', [])) for t in results['tables'].values())
    table_warnings = sum(len(t.get('warnings', [])) for t in results['tables'].values())

    total_issues += table_issues
    total_warnings += table_warnings

    logger.info(f"  총 테이블: {len(results['tables'])}개")
    logger.info(f"  총 레코드: {sum(t['record_count'] for t in results['tables'].values()):,}개")
    logger.info(f"  이슈: {total_issues}개")
    logger.info(f"  경고: {total_warnings}개")

    if total_issues == 0 and total_warnings == 0:
        logger.info("\n  ✅ 모든 검증 통과!")
    elif total_issues == 0:
        logger.info(f"\n  ⚠️  경고 {total_warnings}개 발견 (심각하지 않음)")
    else:
        logger.info(f"\n  ❌ 이슈 {total_issues}개 발견 - 확인 필요")

    results['summary'] = {
        'total_tables': len(results['tables']),
        'total_records': sum(t['record_count'] for t in results['tables'].values()),
        'total_issues': total_issues,
        'total_warnings': total_warnings,
        'status': 'PASS' if total_issues == 0 else 'FAIL'
    }


if __name__ == "__main__":
    try:
        validate_database()
    except Exception as e:
        logger.error(f"\n❌ 검증 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
