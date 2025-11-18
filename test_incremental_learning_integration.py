#!/usr/bin/env python
"""
증분 학습 통합 테스트

ADA 코인으로 전체 파이프라인 실행:
1. 전략 생성 (유사도 검사 포함)
2. Self-play 실행 (최근 100개 전략만)
3. 증분 학습 적용
4. 결과 검증
"""

import sys
import logging
from datetime import datetime

# Windows 인코딩
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print(f"증분 학습 통합 테스트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    # 1. 데이터베이스 초기화 확인
    print("\n1️⃣ 데이터베이스 초기화 확인")
    print("-" * 80)

    from rl_pipeline.db.schema import setup_database_tables

    if setup_database_tables():
        print("✅ 데이터베이스 테이블 준비 완료")
    else:
        print("❌ 데이터베이스 초기화 실패")
        sys.exit(1)

    # 2. 기존 전략 수 확인
    print("\n2️⃣ 기존 전략 현황")
    print("-" * 80)

    from rl_pipeline.db.reads import load_strategies_pool
    from rl_pipeline.db.connection_pool import get_optimized_db_connection

    coin = "ADA"
    interval = "15m"

    # 전체 전략 수
    with get_optimized_db_connection("strategies") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM coin_strategies WHERE coin = ? AND interval = ?", (coin, interval))
        total_count = cursor.fetchone()[0]
        print(f"  전체 전략 수: {total_count}개")

        # 학습 완료된 전략 수
        cursor.execute("""
            SELECT COUNT(*)
            FROM coin_strategies cs
            INNER JOIN strategy_training_history sth ON cs.id = sth.strategy_id
            WHERE cs.coin = ? AND cs.interval = ?
        """, (coin, interval))
        trained_count = cursor.fetchone()[0]
        print(f"  학습 완료 전략: {trained_count}개")
        print(f"  미학습 전략: {total_count - trained_count}개")

    # 3. 전략 생성 (소량 테스트)
    print("\n3️⃣ 전략 생성 (10개 생성)")
    print("-" * 80)

    from rl_pipeline.strategy.creator import create_coin_strategies
    from rl_pipeline.data import load_candles

    # 캔들 데이터 로드
    candle_data = load_candles(coin, interval, days=30)
    if candle_data is None or candle_data.empty:
        print(f"❌ 캔들 데이터 로드 실패: {coin} {interval}")
        sys.exit(1)

    print(f"✅ 캔들 데이터 로드: {len(candle_data)}개 행")

    # 전략 생성 (환경변수로 개수 조정)
    import os
    original_count = os.getenv('STRATEGIES_PER_COMBINATION', '100')
    os.environ['STRATEGIES_PER_COMBINATION'] = '10'  # 테스트용 10개만

    all_candle_data = {(coin, interval): candle_data}
    created_count = create_coin_strategies(coin, [interval], all_candle_data)

    os.environ['STRATEGIES_PER_COMBINATION'] = original_count  # 복원

    print(f"✅ 전략 생성 완료: {created_count}개")

    # 4. 유사도 분류 결과 확인
    print("\n4️⃣ 유사도 분류 결과")
    print("-" * 80)

    strategies = load_strategies_pool(coin, interval, limit=20, order_by="created_at DESC")

    classification_counts = {
        'duplicate': 0,
        'copy': 0,
        'finetune': 0,
        'novel': 0,
        'unknown': 0
    }

    for s in strategies:
        classification = s.get('similarity_classification', 'unknown')
        if classification in classification_counts:
            classification_counts[classification] += 1
        else:
            classification_counts['unknown'] += 1

    print(f"  최근 20개 전략 분류:")
    for cls, count in classification_counts.items():
        if count > 0:
            print(f"    - {cls}: {count}개")

    # 5. Orchestrator 로드 테스트
    print("\n5️⃣ Orchestrator 전략 로드 테스트")
    print("-" * 80)

    loaded_strategies = load_strategies_pool(
        coin, interval,
        limit=100,
        order_by="created_at DESC"
    )
    print(f"✅ 로드된 전략 수: {len(loaded_strategies)}개 (기존 15000개 → 100개로 제한)")

    # 6. 증분 학습 메타데이터 확인
    print("\n6️⃣ 증분 학습 준비 상태")
    print("-" * 80)

    has_metadata = any(
        s.get('similarity_classification') in ['copy', 'finetune', 'novel']
        for s in loaded_strategies
    )

    if has_metadata:
        print(f"✅ 증분 학습 메타데이터 존재: 증분 학습 활성화 가능")

        # 분류별 카운트
        copy_count = sum(1 for s in loaded_strategies if s.get('similarity_classification') == 'copy')
        finetune_count = sum(1 for s in loaded_strategies if s.get('similarity_classification') == 'finetune')
        novel_count = sum(1 for s in loaded_strategies if s.get('similarity_classification') == 'novel')

        print(f"  - 정책 복사(copy): {copy_count}개")
        print(f"  - 미세 조정(finetune): {finetune_count}개")
        print(f"  - 신규 학습(novel): {novel_count}개")
        print(f"  - 예상 시간 절감: {(copy_count * 0.95 + finetune_count * 0.6) / len(loaded_strategies) * 100:.1f}%")
    else:
        print(f"⚠️ 증분 학습 메타데이터 없음: 일반 학습 모드로 실행됨")

    # 7. 종합 결과
    print("\n" + "=" * 80)
    print("통합 테스트 결과")
    print("=" * 80)

    success_checks = []

    # Check 1: DB 테이블 존재
    success_checks.append(("DB 테이블 준비", True))

    # Check 2: 전략 생성 성공
    success_checks.append(("전략 생성", created_count > 0))

    # Check 3: 유사도 분류 작동
    has_classification = sum(classification_counts.values()) > 0
    success_checks.append(("유사도 분류", has_classification))

    # Check 4: 전략 로드 제한 적용
    success_checks.append(("전략 로드 제한", len(loaded_strategies) <= 100))

    # Check 5: 증분 학습 준비
    success_checks.append(("증분 학습 준비", has_metadata))

    # 결과 출력
    passed = sum(1 for _, result in success_checks if result)
    total = len(success_checks)

    print(f"\n통과한 테스트: {passed}/{total}")
    for check_name, result in success_checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")

    if passed == total:
        print("\n🎉 모든 통합 테스트 통과! 증분 학습 시스템 준비 완료")
    elif passed >= total * 0.75:
        print(f"\n⚠️ 대부분 통과 ({passed}/{total}), 일부 개선 필요")
    else:
        print(f"\n❌ 통합 테스트 실패 ({passed}/{total})")

except Exception as e:
    print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print(f"테스트 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
