"""
데이터베이스에서 계산되어야 하는데 편의상 고정값이 들어간 필드 찾기
설계상 고정값이 맞는 경우는 제외
"""
import sqlite3
import sys
from collections import Counter

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def analyze_field_distribution(cursor, table, field, sample_size=1000):
    """필드 값의 분포 분석"""
    cursor.execute(f"""
        SELECT {field}, COUNT(*) as cnt
        FROM {table}
        WHERE {field} IS NOT NULL
        GROUP BY {field}
        ORDER BY cnt DESC
        LIMIT 20
    """)
    return cursor.fetchall()

def check_all_identical(cursor, table, field):
    """모든 값이 동일한지 체크"""
    cursor.execute(f"""
        SELECT COUNT(DISTINCT {field}) as unique_count,
               COUNT(*) as total_count
        FROM {table}
        WHERE {field} IS NOT NULL
    """)
    result = cursor.fetchone()
    unique_count, total_count = result

    if total_count == 0:
        return None

    if unique_count == 1:
        cursor.execute(f"SELECT {field} FROM {table} WHERE {field} IS NOT NULL LIMIT 1")
        value = cursor.fetchone()[0]
        return {'all_identical': True, 'value': value, 'count': total_count}

    # 값의 90% 이상이 동일한 경우도 체크
    cursor.execute(f"""
        SELECT {field}, COUNT(*) as cnt
        FROM {table}
        WHERE {field} IS NOT NULL
        GROUP BY {field}
        ORDER BY cnt DESC
        LIMIT 1
    """)
    most_common = cursor.fetchone()
    if most_common and most_common[1] / total_count > 0.9:
        return {
            'all_identical': False,
            'mostly_identical': True,
            'value': most_common[0],
            'count': most_common[1],
            'total': total_count,
            'percentage': most_common[1] / total_count * 100
        }

    return None

def find_suspicious_patterns():
    """의심스러운 패턴 찾기"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print("🔍 편의상 고정값이 들어간 필드 찾기")
    print("=" * 80)

    suspicious_fields = []

    # 1. strategy_grades 테이블 - 등급 점수들
    print("\n📊 1. strategy_grades - 등급 점수 분석")
    print("-" * 80)

    fields_to_check = [
        'grade_score',
        'total_return',
        'win_rate',
        'predictive_accuracy'
    ]

    for field in fields_to_check:
        result = check_all_identical(cursor, 'strategy_grades', field)
        if result:
            if result.get('all_identical'):
                print(f"  ⚠️ {field}: 모든 값이 {result['value']} (총 {result['count']}개)")
                suspicious_fields.append(('strategy_grades', field, result['value'], 'all_identical'))
            elif result.get('mostly_identical'):
                print(f"  ⚠️ {field}: {result['percentage']:.1f}%가 {result['value']} ({result['count']}/{result['total']})")
                suspicious_fields.append(('strategy_grades', field, result['value'], f"mostly_{result['percentage']:.1f}%"))

    # 2. integrated_analysis_results - 신호 점수들
    print("\n📊 2. integrated_analysis_results - 신호 점수 분석")
    print("-" * 80)

    fields_to_check = [
        'final_signal_score',
        'ensemble_score',
        'fractal_score',
        'multi_timeframe_score',
        'indicator_cross_score',
        'ensemble_confidence',
        'signal_confidence'
    ]

    for field in fields_to_check:
        result = check_all_identical(cursor, 'integrated_analysis_results', field)
        if result:
            if result.get('all_identical'):
                print(f"  ⚠️ {field}: 모든 값이 {result['value']} (총 {result['count']}개)")
                suspicious_fields.append(('integrated_analysis_results', field, result['value'], 'all_identical'))
            elif result.get('mostly_identical'):
                print(f"  ⚠️ {field}: {result['percentage']:.1f}%가 {result['value']} ({result['count']}/{result['total']})")
                suspicious_fields.append(('integrated_analysis_results', field, result['value'], f"mostly_{result['percentage']:.1f}%"))
        else:
            # 분포 확인
            dist = analyze_field_distribution(cursor, 'integrated_analysis_results', field)
            if dist and len(dist) <= 3:  # 3개 이하의 값만 존재
                print(f"  ℹ️ {field}: {len(dist)}개의 고유값만 존재")
                for val, cnt in dist[:5]:
                    print(f"     - {val}: {cnt}개")

    # 3. coin_strategies - 전략 파라미터 및 메타데이터
    print("\n📊 3. coin_strategies - 전략 메타데이터 분석")
    print("-" * 80)

    # regime 분포 체크
    print("  🔹 regime 분포:")
    cursor.execute("""
        SELECT regime, COUNT(*) as cnt
        FROM coin_strategies
        GROUP BY regime
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"     - {row[0]}: {row[1]}개")

    # similarity_score 체크
    result = check_all_identical(cursor, 'coin_strategies', 'similarity_score')
    if result:
        if result.get('all_identical'):
            print(f"  ⚠️ similarity_score: 모든 값이 {result['value']} (총 {result['count']}개)")
            suspicious_fields.append(('coin_strategies', 'similarity_score', result['value'], 'all_identical'))
        elif result.get('mostly_identical'):
            print(f"  ⚠️ similarity_score: {result['percentage']:.1f}%가 {result['value']} ({result['count']}/{result['total']})")

    # 4. rl_strategy_rollup - 집계 메트릭
    print("\n📊 4. rl_strategy_rollup - 집계 메트릭 분석")
    print("-" * 80)

    fields_to_check = [
        'avg_ret',
        'win_rate',
        'predictive_accuracy',
        'avg_dd',
        'total_profit',
        'avg_reward'
    ]

    for field in fields_to_check:
        # 0.0 값의 비율 체크
        cursor.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN {field} = 0.0 THEN 1 ELSE 0 END) as zero_count
            FROM rl_strategy_rollup
            WHERE {field} IS NOT NULL
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            zero_pct = result[1] / result[0] * 100
            if zero_pct > 50:  # 50% 이상이 0.0
                print(f"  ⚠️ {field}: {zero_pct:.1f}%가 0.0 ({result[1]}/{result[0]})")
                suspicious_fields.append(('rl_strategy_rollup', field, 0.0, f"zero_{zero_pct:.1f}%"))

    # 5. rl_episode_summary - 에피소드 메트릭
    print("\n📊 5. rl_episode_summary - 에피소드 메트릭 분석")
    print("-" * 80)

    fields_to_check = [
        'total_reward',
        'realized_ret_signed',
        'acc_flag'
    ]

    for field in fields_to_check:
        # 0.0 값의 비율 체크 (에피소드는 0이 자연스러울 수 있음)
        cursor.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN {field} = 0.0 THEN 1 ELSE 0 END) as zero_count,
                AVG({field}) as avg_val,
                MIN({field}) as min_val,
                MAX({field}) as max_val
            FROM rl_episode_summary
            WHERE {field} IS NOT NULL
            LIMIT 1000
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            zero_pct = result[1] / result[0] * 100
            avg_val = result[2]
            min_val = result[3]
            max_val = result[4]

            # min = max이면 모든 값이 동일
            if min_val == max_val and result[0] > 100:
                print(f"  ⚠️ {field}: 모든 값이 {min_val} (총 {result[0]}개)")
                suspicious_fields.append(('rl_episode_summary', field, min_val, 'all_identical'))
            elif zero_pct > 80:
                print(f"  ⚠️ {field}: {zero_pct:.1f}%가 0.0 (평균: {avg_val:.6f}, 범위: {min_val:.6f}~{max_val:.6f})")

    # 6. 특정 의심 패턴 체크 - 0.5, 0.7 같은 placeholder 값
    print("\n📊 6. 의심스러운 Placeholder 값 체크 (0.5, 0.7, 1.0 등)")
    print("-" * 80)

    placeholder_values = [0.5, 0.7, 0.8, 1.0]

    tables_fields = [
        ('strategy_grades', ['grade_score', 'total_return', 'win_rate', 'predictive_accuracy']),
        ('integrated_analysis_results', ['ensemble_score', 'fractal_score', 'multi_timeframe_score', 'ensemble_confidence', 'signal_confidence']),
        ('coin_strategies', ['similarity_score', 'consistency_score', 'pattern_confidence'])
    ]

    for table, fields in tables_fields:
        for field in fields:
            for placeholder in placeholder_values:
                cursor.execute(f"""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN {field} = ? THEN 1 ELSE 0 END) as placeholder_count
                    FROM {table}
                    WHERE {field} IS NOT NULL
                """, (placeholder,))
                result = cursor.fetchone()
                if result and result[0] > 0:
                    pct = result[1] / result[0] * 100
                    if pct > 50:
                        print(f"  ⚠️ {table}.{field}: {pct:.1f}%가 {placeholder} ({result[1]}/{result[0]})")
                        if (table, field, placeholder, f"placeholder_{pct:.1f}%") not in suspicious_fields:
                            suspicious_fields.append((table, field, placeholder, f"placeholder_{pct:.1f}%"))

    # 요약
    print("\n" + "=" * 80)
    print("📋 의심스러운 필드 요약")
    print("=" * 80)

    if suspicious_fields:
        print(f"\n총 {len(suspicious_fields)}개의 의심스러운 필드 발견:\n")
        for table, field, value, reason in suspicious_fields:
            print(f"  ⚠️ {table}.{field}")
            print(f"     - 값: {value}")
            print(f"     - 이유: {reason}")
            print()
    else:
        print("\n✅ 의심스러운 고정값 패턴이 발견되지 않았습니다.")

    conn.close()

    return suspicious_fields


if __name__ == "__main__":
    find_suspicious_patterns()
