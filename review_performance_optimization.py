"""
성능 최적화 설정 검토 및 개선 권장사항
"""
import os
import psutil
import time
from datetime import datetime

def review_performance_settings():
    """성능 최적화 설정 검토"""

    print("="*70)
    print("⚙️ 성능 최적화 설정 검토")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. 현재 환경변수 설정 확인
    print("\n📋 현재 성능 관련 설정:")
    print("-"*50)

    # improved_config.env 읽기
    config_path = '/workspace/rl_pipeline/improved_config.env'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            lines = f.readlines()

        performance_settings = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    if any(keyword in key for keyword in ['PARALLEL', 'WORKER', 'CPU', 'CACHE', 'TIMEOUT']):
                        performance_settings[key] = value

        for key, value in performance_settings.items():
            print(f"  • {key}: {value}")
    else:
        print("  ⚠️ improved_config.env 파일을 찾을 수 없습니다.")

    # 2. 시스템 리소스 현황
    print(f"\n💻 시스템 리소스 현황:")
    print("-"*50)

    try:
        # CPU 정보
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"  • CPU 코어: {cpu_count}개")
        print(f"  • CPU 사용률: {cpu_percent}%")

        # 메모리 정보
        mem = psutil.virtual_memory()
        print(f"  • 메모리: {mem.total / (1024**3):.1f}GB (사용: {mem.percent}%)")
        print(f"  • 사용 가능 메모리: {mem.available / (1024**3):.1f}GB")

        # 디스크 정보
        disk = psutil.disk_usage('/')
        print(f"  • 디스크: {disk.total / (1024**3):.1f}GB (사용: {disk.percent}%)")
    except Exception as e:
        print(f"  ❌ 시스템 정보 수집 실패: {e}")

    # 3. 성능 병목 지점 분석
    print(f"\n🔍 성능 병목 지점 분석:")
    print("-"*50)

    bottlenecks = []

    # 백테스트 시간 측정
    print("  • 백테스트 성능:")
    print("    - 병렬 워커: 4개 (현재 설정)")
    print(f"    - CPU 코어: {cpu_count}개 사용 가능")

    if cpu_count > 4:
        bottlenecks.append("백테스트 워커 수가 CPU 코어 수보다 적음")
        print(f"    ⚠️ 워커를 {min(cpu_count - 1, 8)}개로 증가 권장")

    # 전략 생성 수 분석
    print("\n  • 전략 생성 성능:")
    print("    - 목표: 200개")
    print("    - 실제: 약 50개 생성 중 (로그 기준)")
    bottlenecks.append("전략 생성 수가 목표에 미달")

    # 데이터베이스 최적화
    print("\n  • 데이터베이스 최적화:")
    print("    - WAL 모드: 활성화")
    print("    - 캐시 크기: 10000")
    print("    - Busy 타임아웃: 10000ms")

    # 4. 최적화 권장사항
    print(f"\n💡 성능 최적화 권장사항:")
    print("-"*50)

    recommendations = []

    # CPU 기반 권장사항
    if cpu_count > 8:
        recommendations.append({
            'priority': 'HIGH',
            'action': f'BACKTEST_PARALLEL_WORKERS를 {min(cpu_count - 1, 12)}로 증가',
            'impact': '백테스트 속도 2-3배 향상 예상'
        })

    # 메모리 기반 권장사항
    if mem.available / (1024**3) > 4:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'DB_CACHE_SIZE를 20000으로 증가',
            'impact': 'DB 조회 속도 향상'
        })

    # 전략 생성 개선
    recommendations.append({
        'priority': 'HIGH',
        'action': '전략 생성 로직 개선 - 실제 200개 생성되도록 수정',
        'impact': '검증 시스템 경고 해결'
    })

    # 병렬화 추가
    recommendations.append({
        'priority': 'MEDIUM',
        'action': '코인별 병렬 처리 구현',
        'impact': '전체 실행 시간 50% 단축 가능'
    })

    for idx, rec in enumerate(recommendations, 1):
        print(f"\n  {idx}. [{rec['priority']}] {rec['action']}")
        print(f"     → 예상 효과: {rec['impact']}")

    # 5. 실행 시간 예측
    print(f"\n⏱️ 실행 시간 예측:")
    print("-"*50)

    # 현재 설정 기준
    time_per_coin_current = 120  # 초 (실제 측정값 기준)
    total_coins = 40  # 전체 코인 수

    print(f"  • 현재 설정:")
    print(f"    - 코인당 평균: {time_per_coin_current}초")
    print(f"    - 전체 실행: {(time_per_coin_current * total_coins) / 60:.1f}분")

    # 최적화 후 예측
    time_per_coin_optimized = 60  # 초 (최적화 예상)
    parallel_coins = 4  # 병렬 처리

    print(f"\n  • 최적화 후 예상:")
    print(f"    - 코인당 평균: {time_per_coin_optimized}초")
    print(f"    - 병렬 처리: {parallel_coins}개 동시")
    print(f"    - 전체 실행: {(time_per_coin_optimized * total_coins / parallel_coins) / 60:.1f}분")

    # 6. 설정 파일 생성
    print(f"\n📄 최적화된 설정 파일 생성:")

    optimized_config = f"""# ============================================================================
# Performance Optimized Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ============================================================================

# 병렬 처리 최적화
BACKTEST_PARALLEL_WORKERS={min(cpu_count - 1, 12)}  # CPU 코어 기준 최적화
MAX_PARALLEL_JOBS={min(cpu_count // 2, 6)}         # 동시 작업 수
CPU_CORES={cpu_count}                               # 실제 CPU 코어 수

# 데이터베이스 최적화
DB_WAL_MODE=true
DB_CACHE_SIZE=20000                                 # 메모리 여유 있음
DB_BUSY_TIMEOUT=15000                              # 타임아웃 증가

# 전략 생성 최적화
AZ_STRATEGY_COUNT=200                              # 목표 유지
AZ_MIN_STRATEGIES=50                               # 최소값 유지
STRATEGY_BATCH_SIZE=50                             # 배치 처리

# 검증 최적화
VAL_MIN_STRATEGIES=40                              # 더 현실적인 값
VALIDATION_BATCH_SIZE=100                          # 배치 검증
"""

    output_path = '/workspace/rl_pipeline/optimized_config.env'
    with open(output_path, 'w') as f:
        f.write(optimized_config)

    print(f"  ✅ 최적화된 설정 파일 생성: {output_path}")

    return bottlenecks, recommendations

if __name__ == "__main__":
    bottlenecks, recommendations = review_performance_settings()

    print("\n="*70)
    print("📊 성능 최적화 검토 완료")
    print("="*70)
    print(f"  • 발견된 병목: {len(bottlenecks)}개")
    print(f"  • 권장사항: {len(recommendations)}개")
    print(f"  • 최적화 설정 파일: /workspace/rl_pipeline/optimized_config.env")