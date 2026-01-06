import os
import sys
import glob
import time
import sqlite3
import pandas as pd
from collections import defaultdict

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))

DATA_DIR = os.path.join(ROOT_DIR, "market", "coin_market", "data_storage")
STRATEGIES_DIR = os.path.join(DATA_DIR, "learning_strategies")

# 🎯 마스터 지식 저장소
GLOBAL_OUTPUT_DB = os.path.join(STRATEGIES_DIR, "common_strategies.db")
# 🎯 캔들 소스
CANDLE_DB = os.path.join(DATA_DIR, "learning_strategies.db")

# 2. 실행 환경 변수 설정
os.environ['PYTHONPATH'] = ROOT_DIR
os.environ['RL_DB_PATH'] = CANDLE_DB
os.environ['STRATEGY_DB_PATH'] = STRATEGIES_DIR
os.environ['STRATEGIES_DB_PATH'] = GLOBAL_OUTPUT_DB
os.environ['AZ_INTERVALS'] = "15m,30m,240m,1d"

# 3. 엔진 모듈 경로 추가
sys.path.append(ROOT_DIR)
try:
    from rl_pipeline.strategy.global_synthesizer import create_global_synthesizer
    from rl_pipeline.strategy.binned_global_synthesizer import create_binned_global_synthesizer
    from rl_pipeline.pipelines.orchestrator import (
        validate_global_strategy_pool,
        validate_global_strategy_patterns,
        validate_global_strategy_quality
    )
    print("✅ Absolute Zero 시스템 엔진 로드 완료")
except ImportError as e:
    print(f"❌ 엔진 모듈을 찾을 수 없습니다: {e}")
    sys.exit(1)

def manual_load_pool_optimized(db_files):
    """
    시스템 규격(min_trades=1, max_dd=1.0)에 맞는 데이터를 
    시간대별로 그룹화하여 수집 (validate_global_strategy_pool 호환 규격)
    """
    grouped_pool = defaultdict(list)
    print(f"🔄 {len(db_files)}개 DB에서 시스템 규격 데이터를 수집 중...")
    
    total_count = 0
    for db_path in db_files:
        filename = os.path.basename(db_path)
        if filename in ["common_strategies.db", "learning_strategies.db", "trade_candles.db", "learning_candles.db"]:
            continue
            
        try:
            with sqlite3.connect(db_path) as conn:
                # 💡 [시스템 규격] min_trades=1, max_dd=1.0
                query = "SELECT * FROM strategies WHERE trades_count >= 1 AND max_drawdown <= 1.0"
                df = pd.read_sql(query, conn)
                if not df.empty:
                    for _, row in df.iterrows():
                        interval = row['interval']
                        grouped_pool[interval].append(row.to_dict())
                        total_count += 1
        except Exception:
            continue
    print(f"✅ 수집 완료: 총 {total_count}개 전략 확보")
    return grouped_pool

def run_manual_synthesis():
    print("=" * 60)
    print("🚀 Absolute Zero 시스템 - 글로벌 전략 합성 엔진 (수동 실행)")
    print(f"📍 대상: {GLOBAL_OUTPUT_DB}")
    print("-" * 60)
    
    # 🔥 출력 DB 경로의 부모 디렉토리 존재 확인 및 생성
    output_dir = os.path.dirname(GLOBAL_OUTPUT_DB)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 출력 디렉토리 생성: {output_dir}")
    
    # 4. 데이터 수집 (시간대별 그룹화된 Dict 반환)
    db_files = glob.glob(os.path.join(STRATEGIES_DIR, "*_strategies.db"))
    pool = manual_load_pool_optimized(db_files)
    
    if not pool:
        print("❌ 수집된 전략이 없습니다.")
        return

    # 5. 합성 엔진 초기화
    intervals = ["15m", "30m", "240m", "1d"]
    synthesizer = create_global_synthesizer(GLOBAL_OUTPUT_DB, intervals, seed=123)
    
    # 6. 합성 프로세스 (시스템 7단계 파이프라인 완전 복제)
    try:
        # [Step 1] 검증
        print("📊 1단계: 전략 풀 검증...")
        # 모든 인터벌의 전략에서 코인 리스트 추출
        all_strategies = []
        for interval_strats in pool.values():
            all_strategies.extend(interval_strats)
        
        coins = list(set([s.get('symbol') or s.get('coin') for s in all_strategies]))
        
        # 💡 이제 pool이 Dict 형태이므로 검증 함수가 정상 작동합니다.
        pool_val = validate_global_strategy_pool(pool, coins, intervals, min_strategies_per_interval=10)
        print(f"   └─ 결과: {'✅ 통과' if pool_val['valid'] else '⚠️ 경고발생'}")

        # [Step 2] 표준화
        print("📊 2단계: 전략 표준화...")
        std_pool = synthesizer.standardize(pool)
        
        # [Step 3] 패턴 추출
        print("📊 3단계: 공통 패턴 추출...")
        patterns = synthesizer.extract_common_patterns(std_pool)
        pattern_val = validate_global_strategy_patterns(patterns, min_patterns_per_interval=3)
        print(f"   └─ 결과: {'✅ 통과' if pattern_val['valid'] else '⚠️ 경고발생'}")
        
        # [Step 4-6] 조립 및 백테스트
        print("📊 4-6단계: 글로벌 전략 조립 및 샌티 백테스트...")
        assembled = synthesizer.assemble_global_strategies(patterns)
        tested = synthesizer.quick_sanity_backtest(assembled)
        final = synthesizer.apply_fallbacks(tested)
        
        # [Step 7] 저장
        print(f"📊 7단계: 최종 결과 저장 -> {os.path.basename(GLOBAL_OUTPUT_DB)}")
        synthesizer.save(final)
        
        # 결과 리포트 (기존 방식)
        total_count = sum(len(s) for s in final.values())
        print("\n" + "-" * 60)
        print(f"✅ [방식 1] 레짐별 대표 전략: {total_count}개")
        for itv, strats in final.items():
            print(f"   ● {itv:<5}: {len(strats)}개 레짐 매핑 완료")
        
        # [추가] 세밀한 구간화 기반 글로벌 예측값 생성
        print("\n" + "-" * 60)
        print("📊 [방식 2] 세밀한 구간화 기반 글로벌 예측값 생성...")
        
        binned_predictions_count = 0
        try:
            binned_synthesizer = create_binned_global_synthesizer(
                source_db_path=STRATEGIES_DIR,
                output_db_path=GLOBAL_OUTPUT_DB,
                intervals=intervals,
                seed=123
            )
            
            binned_result = binned_synthesizer.run_synthesis(
                min_trades=5,
                max_dd=0.8,
                min_samples=2
            )
            
            if binned_result['success']:
                binned_predictions_count = binned_result['output_predictions']
                print(f"✅ 구간화 기반 글로벌 예측값: {binned_predictions_count}개")
                for itv, count in binned_result['interval_stats'].items():
                    print(f"   ● {itv:<5}: {count}개 시그널 조건 커버")
            else:
                print(f"⚠️ 구간화 기반 합성 실패: {binned_result.get('error')}")
                
        except Exception as be:
            print(f"⚠️ 구간화 기반 합성 실패: {be}")
        
        # 최종 요약
        print("\n" + "=" * 60)
        print(f"✨ 글로벌 전략 합성 완료!")
        print(f"📊 레짐별 대표 전략 (global_strategies): {total_count}개")
        print(f"📊 구간화 기반 예측값 (global_strategy_predictions): {binned_predictions_count}개")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        print(f"\n❌ 합성 과정 중 오류 발생: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_manual_synthesis()
