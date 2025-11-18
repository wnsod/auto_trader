#!/usr/bin/env python
"""
다중 코인 RL 파이프라인 실행 및 통합 분석

주요 코인들(BTC, ETH, SOL, ADA, XRP)에 대해:
1. RL 파이프라인 실행 (4개 인터벌: 15m, 30m, 240m, 1d)
2. 통합 분석 v1으로 평가
3. 코인별 결과 비교
"""

import sys
sys.path.append('/workspace')

import subprocess
import time
from datetime import datetime
import json
from rl_pipeline.analysis.integrated_analysis_v1 import IntegratedAnalyzerV1

# 학습할 코인 목록
COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']  # 주요 코인 5개
INTERVALS = ['15m', '30m', '240m', '1d']  # 4개 인터벌

# 학습 설정
EPISODES_PER_INTERVAL = 100  # 인터벌당 에피소드 수


def run_coin_training(coin: str):
    """
    단일 코인에 대해 RL 파이프라인 실행

    Args:
        coin: 코인 심볼 (예: BTC, ETH)

    Returns:
        success: 학습 성공 여부
    """
    print("=" * 70)
    print(f"🪙 {coin} 학습 시작")
    print("=" * 70)
    print(f"인터벌: {', '.join(INTERVALS)}")
    print(f"에피소드: {EPISODES_PER_INTERVAL}개/인터벌")
    print()

    start_time = datetime.now()

    # absolute_zero_system.py를 코인별로 실행
    # 환경변수로 코인 지정
    import os
    os.environ['TARGET_COIN'] = coin

    try:
        # 1. 캔들 데이터 수집
        print(f"📊 1/4: {coin} 캔들 데이터 수집 중...")
        result = subprocess.run(
            ['python', '/workspace/rl_pipeline/rl_candles_collector.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )

        if result.returncode != 0:
            print(f"❌ 캔들 수집 실패: {result.stderr[:200]}")
            return False

        print(f"✅ 캔들 수집 완료")

        # 2. 지표 계산
        print(f"📈 2/4: {coin} 지표 계산 중...")
        result = subprocess.run(
            ['python', '/workspace/rl_pipeline/rl_candles_calculate.py'],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"❌ 지표 계산 실패: {result.stderr[:200]}")
            return False

        print(f"✅ 지표 계산 완료")

        # 3. 패턴 계산
        print(f"🔍 3/4: {coin} 패턴/파동 계산 중...")
        result = subprocess.run(
            ['python', '/workspace/rl_pipeline/rl_candles_integrated.py'],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"❌ 패턴 계산 실패: {result.stderr[:200]}")
            return False

        print(f"✅ 패턴 계산 완료")

        # 4. RL 학습 (absolute_zero_system)
        print(f"🧠 4/4: {coin} RL 학습 중 (예상 시간: 1-2시간)...")
        print(f"   → {EPISODES_PER_INTERVAL}개 에피소드 × {len(INTERVALS)}개 인터벌")

        result = subprocess.run(
            ['python', '/workspace/rl_pipeline/absolute_zero_system.py'],
            capture_output=True,
            text=True,
            timeout=7200  # 2시간 타임아웃
        )

        if result.returncode != 0:
            print(f"❌ RL 학습 실패: {result.stderr[:200]}")
            # 로그 출력 (디버깅용)
            if result.stdout:
                print(f"📝 마지막 출력:")
                print(result.stdout[-500:])
            return False

        print(f"✅ RL 학습 완료")

        elapsed = (datetime.now() - start_time).total_seconds()
        print()
        print(f"✅ {coin} 학습 완료! (소요 시간: {elapsed/60:.1f}분)")
        print()

        return True

    except subprocess.TimeoutExpired:
        print(f"❌ {coin} 학습 타임아웃!")
        return False
    except Exception as e:
        print(f"❌ {coin} 학습 오류: {e}")
        return False


def analyze_coin(coin: str):
    """
    코인에 대해 통합 분석 v1 실행

    Args:
        coin: 코인 심볼

    Returns:
        result: 통합 분석 결과 딕셔너리
    """
    try:
        analyzer = IntegratedAnalyzerV1()
        result = analyzer.analyze(coin)
        return result
    except Exception as e:
        print(f"❌ {coin} 분석 오류: {e}")
        return None


def main():
    print("=" * 70)
    print("🚀 다중 코인 RL 파이프라인 & 통합 분석")
    print("=" * 70)
    print(f"대상 코인: {', '.join(COINS)}")
    print(f"인터벌: {', '.join(INTERVALS)}")
    print(f"총 학습 예상 시간: {len(COINS)} × 1~2시간 = {len(COINS)~2*len(COINS)}시간")
    print()

    input("계속하려면 Enter를 누르세요 (Ctrl+C로 취소)...")
    print()

    # 각 코인별 학습 결과
    training_results = {}
    analysis_results = {}

    total_start = datetime.now()

    # 1단계: 각 코인 학습
    print("=" * 70)
    print("📚 1단계: 코인별 RL 학습")
    print("=" * 70)
    print()

    for i, coin in enumerate(COINS, 1):
        print(f"\n[{i}/{len(COINS)}] {coin} 처리 중...")

        success = run_coin_training(coin)
        training_results[coin] = success

        if success:
            print(f"✅ {coin} 학습 성공")
        else:
            print(f"❌ {coin} 학습 실패 - 통합 분석 건너뜀")

        # 다음 코인 전에 잠깐 대기 (시스템 안정화)
        if i < len(COINS):
            print("\n⏳ 다음 코인 준비 중... (10초 대기)")
            time.sleep(10)

    # 2단계: 통합 분석
    print("\n" + "=" * 70)
    print("📊 2단계: 코인별 통합 분석 v1")
    print("=" * 70)
    print()

    for coin in COINS:
        if not training_results.get(coin):
            print(f"⏭️  {coin}: 학습 실패로 건너뜀")
            continue

        print(f"🔍 {coin} 통합 분석 중...")
        result = analyze_coin(coin)

        if result:
            analysis_results[coin] = result
            print(f"   방향: {result['direction']}")
            print(f"   타이밍: {result['timing']}")
            print(f"   크기: {result['size']:.1%}")
            print(f"   확신도: {result['confidence']:.1%}")
            print(f"   기간: {result['horizon']}")
            print()
        else:
            print(f"❌ {coin} 분석 실패")
            print()

    # 3단계: 결과 요약
    print("=" * 70)
    print("📋 3단계: 전체 결과 요약")
    print("=" * 70)
    print()

    # 학습 성공률
    success_count = sum(1 for v in training_results.values() if v)
    print(f"학습 성공: {success_count}/{len(COINS)} ({success_count/len(COINS)*100:.0f}%)")
    print()

    # 통합 분석 결과 테이블
    if analysis_results:
        print(f"{'코인':<8} {'방향':<8} {'타이밍':<8} {'크기':>8} {'확신도':>8} {'기간':<8}")
        print("-" * 70)

        for coin, result in analysis_results.items():
            print(f"{coin:<8} {result['direction']:<8} {result['timing']:<8} "
                  f"{result['size']:>7.1%} {result['confidence']:>7.1%} {result['horizon']:<8}")

        print()

        # 통계
        long_count = sum(1 for r in analysis_results.values() if r['direction'] == 'LONG')
        short_count = sum(1 for r in analysis_results.values() if r['direction'] == 'SHORT')
        hold_count = sum(1 for r in analysis_results.values() if r['direction'] == 'HOLD')

        now_count = sum(1 for r in analysis_results.values() if r['timing'] == 'NOW')
        wait_count = sum(1 for r in analysis_results.values() if r['timing'] == 'WAIT')

        avg_size = sum(r['size'] for r in analysis_results.values()) / len(analysis_results)
        avg_confidence = sum(r['confidence'] for r in analysis_results.values()) / len(analysis_results)

        print("방향 분포:")
        print(f"  LONG: {long_count}개")
        print(f"  SHORT: {short_count}개")
        print(f"  HOLD: {hold_count}개")
        print()

        print("타이밍 분포:")
        print(f"  NOW: {now_count}개")
        print(f"  WAIT: {wait_count}개")
        print()

        print(f"평균 포지션 크기: {avg_size:.1%}")
        print(f"평균 확신도: {avg_confidence:.1%}")
    else:
        print("❌ 통합 분석 결과 없음 (모든 코인 학습 실패)")

    print()

    # 소요 시간
    total_elapsed = (datetime.now() - total_start).total_seconds()
    print(f"총 소요 시간: {total_elapsed/60:.1f}분 ({total_elapsed/3600:.1f}시간)")
    print()

    # 결과 저장
    output_file = f'/workspace/multi_coin_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_data = {
        'coins': COINS,
        'intervals': INTERVALS,
        'training_results': training_results,
        'analysis_results': {
            coin: {
                'direction': result['direction'],
                'timing': result['timing'],
                'size': float(result['size']),
                'confidence': float(result['confidence']),
                'horizon': result['horizon']
            }
            for coin, result in analysis_results.items()
        },
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': total_elapsed
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("=" * 70)
    print(f"결과 저장: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
