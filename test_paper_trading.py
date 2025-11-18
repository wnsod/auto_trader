#!/usr/bin/env python
"""Paper Trading 기능 테스트"""
import sys
sys.path.append('/workspace')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_module_import():
    """모듈 import 테스트"""
    print("=" * 70)
    print("1️⃣  모듈 import 테스트")
    print("=" * 70)
    print()

    try:
        from rl_pipeline.validation.auto_paper_trading import (
            AutoPaperTrading,
            auto_start_paper_trading_after_pipeline,
            run_paper_trading_monitor
        )
        print("✅ AutoPaperTrading 모듈 import 성공")
        print("✅ auto_start_paper_trading_after_pipeline 함수 로드")
        print("✅ run_paper_trading_monitor 함수 로드")
        print()
        return True
    except Exception as e:
        print(f"❌ 모듈 import 실패: {e}")
        print()
        return False


def test_create_instance():
    """AutoPaperTrading 인스턴스 생성 테스트"""
    print("=" * 70)
    print("2️⃣  AutoPaperTrading 인스턴스 생성 테스트")
    print("=" * 70)
    print()

    try:
        from rl_pipeline.validation.auto_paper_trading import AutoPaperTrading

        apt = AutoPaperTrading()
        print(f"✅ AutoPaperTrading 인스턴스 생성 성공")
        print(f"   DB 경로: {apt.db_path}")
        print()
        return True
    except Exception as e:
        print(f"❌ 인스턴스 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_create_session():
    """Paper Trading 세션 생성 테스트"""
    print("=" * 70)
    print("3️⃣  Paper Trading 세션 생성 테스트")
    print("=" * 70)
    print()

    try:
        from rl_pipeline.validation.auto_paper_trading import AutoPaperTrading

        apt = AutoPaperTrading()

        # 테스트 세션 생성
        session_id = apt.start_paper_trading(
            coin='BTC',
            interval='15m',
            initial_capital=100000,
            duration_days=30
        )

        if session_id:
            print(f"✅ Paper Trading 세션 생성 성공")
            print(f"   세션 ID: {session_id}")
            print()
            return True
        else:
            print(f"❌ 세션 생성 실패 (session_id=None)")
            print()
            return False

    except Exception as e:
        print(f"❌ 세션 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_get_active_sessions():
    """활성 세션 조회 테스트"""
    print("=" * 70)
    print("4️⃣  활성 세션 조회 테스트")
    print("=" * 70)
    print()

    try:
        from rl_pipeline.validation.auto_paper_trading import AutoPaperTrading

        apt = AutoPaperTrading()
        sessions = apt.get_active_sessions()

        print(f"✅ 활성 세션 조회 성공: {len(sessions)}개")

        if sessions:
            print()
            print(f"{'Session ID':<40} {'코인':<8} {'인터벌':<8} {'상태':<10}")
            print("-" * 70)
            for s in sessions[:5]:  # 최대 5개만
                print(f"{s['session_id']:<40} {s['coin']:<8} {s['interval']:<8} {s['status']:<10}")
        else:
            print("   (세션 없음)")

        print()
        return True

    except Exception as e:
        print(f"❌ 활성 세션 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_auto_start():
    """auto_start_paper_trading_after_pipeline 테스트"""
    print("=" * 70)
    print("5️⃣  auto_start_paper_trading_after_pipeline 테스트")
    print("=" * 70)
    print()

    try:
        from rl_pipeline.validation.auto_paper_trading import auto_start_paper_trading_after_pipeline

        result = auto_start_paper_trading_after_pipeline(
            coin='ETH',
            intervals=['15m', '30m'],
            duration_days=30
        )

        print(f"✅ auto_start_paper_trading_after_pipeline 실행 성공")
        print(f"   상태: {result.get('status')}")
        print(f"   결과:")

        for r in result.get('results', []):
            print(f"      - {r['coin']}-{r['interval']}: {r['status']}")

        print()
        return result.get('status') == 'started'

    except Exception as e:
        print(f"❌ auto_start_paper_trading_after_pipeline 실패: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    print("\n" * 2)
    print("🔍 Paper Trading 기능 테스트 시작")
    print()

    results = []

    # 1. 모듈 import
    results.append(("모듈 import", test_module_import()))

    # 2. 인스턴스 생성
    results.append(("인스턴스 생성", test_create_instance()))

    # 3. 세션 생성
    results.append(("세션 생성", test_create_session()))

    # 4. 활성 세션 조회
    results.append(("활성 세션 조회", test_get_active_sessions()))

    # 5. auto_start 테스트
    results.append(("auto_start 함수", test_auto_start()))

    # 6. 최종 활성 세션 조회
    results.append(("최종 활성 세션", test_get_active_sessions()))

    # 결과 요약
    print("=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name:<20} {status}")

    print()
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print()

    if passed == total:
        print("🎉 모든 테스트 통과! Paper Trading 정상 동작")
    else:
        print("⚠️  일부 테스트 실패. 로그를 확인하세요.")

    print()


if __name__ == '__main__':
    main()
