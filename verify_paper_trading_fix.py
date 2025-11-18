"""Paper Trading 경고 수정 검증 스크립트"""
import sqlite3
import sys

def verify_fix():
    """수정 사항 검증"""
    db_path = '/workspace/data_storage/learning_results.db'

    print("=" * 70)
    print("Paper Trading 경고 수정 검증")
    print("=" * 70)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. 테이블 존재 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='integrated_analysis_results'")
        if not cursor.fetchone():
            print("❌ integrated_analysis_results 테이블이 존재하지 않습니다.")
            return False
        print("✅ integrated_analysis_results 테이블 존재 확인")

        # 2. 전체 데이터 확인
        cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
        total_count = cursor.fetchone()[0]
        print(f"\n📊 전체 레코드 수: {total_count}개")

        # 3. interval 값 분포 확인
        cursor.execute("""
            SELECT interval, COUNT(*) as cnt
            FROM integrated_analysis_results
            GROUP BY interval
            ORDER BY cnt DESC
        """)
        print("\n📊 Interval 값 분포:")
        interval_types = {}
        for row in cursor.fetchall():
            interval_types[row[0]] = row[1]
            print(f"  - {row[0]}: {row[1]}개")

        # 4. 코인별 interval 값 확인
        cursor.execute("SELECT DISTINCT coin FROM integrated_analysis_results ORDER BY coin")
        coins = [row[0] for row in cursor.fetchall()]
        print(f"\n📊 코인 목록: {', '.join(coins)}")

        # 5. 각 코인별 최신 데이터 확인
        print("\n📊 각 코인별 최신 통합 분석 결과:")
        print("-" * 70)

        for coin in coins:
            # all_intervals 조회
            cursor.execute("""
                SELECT interval, signal_action, final_signal_score, created_at
                FROM integrated_analysis_results
                WHERE coin = ? AND interval = 'all_intervals'
                ORDER BY created_at DESC
                LIMIT 1
            """, (coin,))
            all_intervals_row = cursor.fetchone()

            # 개별 인터벌 조회
            cursor.execute("""
                SELECT DISTINCT interval
                FROM integrated_analysis_results
                WHERE coin = ? AND interval != 'all_intervals'
                ORDER BY interval
            """, (coin,))
            individual_intervals = [row[0] for row in cursor.fetchall()]

            print(f"\n{coin}:")
            if all_intervals_row:
                print(f"  ✅ all_intervals: {all_intervals_row[1]} (점수: {all_intervals_row[2]:.4f})")
                print(f"     생성시간: {all_intervals_row[3]}")
            else:
                print(f"  ❌ all_intervals: 데이터 없음")

            if individual_intervals:
                print(f"  ✅ 개별 인터벌: {', '.join(individual_intervals)}")
                # 각 개별 인터벌의 최신 데이터 확인
                for interval in individual_intervals:
                    cursor.execute("""
                        SELECT signal_action, final_signal_score, created_at
                        FROM integrated_analysis_results
                        WHERE coin = ? AND interval = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (coin, interval))
                    interval_row = cursor.fetchone()
                    if interval_row:
                        print(f"     - {interval}: {interval_row[0]} (점수: {interval_row[1]:.4f})")
            else:
                print(f"  ⚠️ 개별 인터벌: 데이터 없음")

        # 6. Paper Trading 조회 시뮬레이션
        print("\n" + "=" * 70)
        print("📊 Paper Trading 조회 시뮬레이션 (LINK 코인)")
        print("=" * 70)

        test_coin = 'LINK'
        test_intervals = ['15m', '30m', '240m', '1d']

        for interval in test_intervals:
            # 개별 인터벌 조회
            cursor.execute("""
                SELECT coin, interval, signal_action, final_signal_score
                FROM integrated_analysis_results
                WHERE coin = ? AND interval = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (test_coin, interval))
            individual_result = cursor.fetchone()

            # all_intervals 폴백 조회
            cursor.execute("""
                SELECT coin, interval, signal_action, final_signal_score
                FROM integrated_analysis_results
                WHERE coin = ? AND interval = 'all_intervals'
                ORDER BY created_at DESC
                LIMIT 1
            """, (test_coin,))
            fallback_result = cursor.fetchone()

            print(f"\n{test_coin}-{interval}:")
            if individual_result:
                print(f"  ✅ 개별 인터벌 조회 성공: {individual_result[2]} (점수: {individual_result[3]:.4f})")
            else:
                print(f"  ⚠️ 개별 인터벌 조회 실패")

            if fallback_result:
                print(f"  ✅ all_intervals 폴백 가능: {fallback_result[2]} (점수: {fallback_result[3]:.4f})")
            else:
                print(f"  ❌ all_intervals 폴백 불가")

            # 최종 판정
            if individual_result or fallback_result:
                print(f"  ✅ Paper Trading 시그널 생성 가능 (경고 없음)")
            else:
                print(f"  ❌ Paper Trading 시그널 생성 불가 (경고 발생 예상)")

        # 7. 최종 결과
        print("\n" + "=" * 70)
        print("최종 검증 결과")
        print("=" * 70)

        # all_intervals 데이터 존재 여부
        has_all_intervals = 'all_intervals' in interval_types

        # 개별 인터벌 데이터 존재 여부
        has_individual_intervals = any(
            interval in interval_types
            for interval in ['15m', '30m', '240m', '1d']
        )

        print(f"\n1. all_intervals 데이터 존재: {'✅ YES' if has_all_intervals else '❌ NO'}")
        print(f"2. 개별 인터벌 데이터 존재: {'✅ YES' if has_individual_intervals else '❌ NO'}")

        if has_all_intervals and has_individual_intervals:
            print("\n✅ 수정 완료! Paper Trading 경고가 발생하지 않을 것으로 예상됩니다.")
            print("   - all_intervals로 통합 분석 결과 저장 확인")
            print("   - 개별 인터벌별 데이터도 저장 확인")
            print("   - Paper Trading이 정상적으로 시그널을 조회할 수 있습니다.")
            return True
        elif has_all_intervals:
            print("\n⚠️ 부분 수정: all_intervals 데이터만 존재")
            print("   - Paper Trading이 all_intervals 폴백으로 동작 가능")
            print("   - 개별 인터벌 데이터가 없어 최적화된 시그널 사용 불가")
            return True
        else:
            print("\n❌ 수정 필요: all_intervals 데이터가 없습니다.")
            print("   - 파이프라인을 다시 실행하여 데이터를 생성해야 합니다.")
            return False

    except Exception as e:
        print(f"\n❌ 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    success = verify_fix()
    sys.exit(0 if success else 1)
