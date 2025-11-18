import sys
sys.path.insert(0, '/workspace/')  # 절대 경로 추가

import subprocess
import time
import os
import json
from datetime import datetime

# 상수 및 경로 설정
PIPELINE_INTERVAL_SECONDS = 60  # 매 1분 간격
TRADE_INTERVAL_MINUTES = 1  # 거래 실행 간격 (5분)
DB_PATH = '/workspace/data_storage/realtime_candles.db'
PIPELINE_LOG_FILE = '/workspace/trade/logs/pipeline_log.txt'

# 실행 스크립트 경로
COLLECTOR_SCRIPT = '/workspace/trade/realtime_candles_collector.py'
CALCULATOR_SCRIPT = '/workspace/trade/realtime_candles_calculate.py'
INTEGRATED_SCRIPT = '/workspace/trade/realtime_candles_integrated.py'
SELECTOR_SCRIPT = '/workspace/trade/realtime_signal_selector.py'
EXECUTOR_SCRIPT = '/workspace/trade/realtime_signal_executor.py'
TRADE_MANAGER_SCRIPT = '/workspace/trade/trade_manager.py'
VIRTUAL_TRADER_SCRIPT = '/workspace/trade/virtual_trader.py'
VIRTUAL_LEARNER_SCRIPT = '/workspace/trade/virtual_trading_learner.py'

# 로깅 함수
def log_pipeline_status(message):
    timestamp = datetime.now().isoformat()
    
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(PIPELINE_LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(PIPELINE_LOG_FILE, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{timestamp} - {message}\n")
    print(f"[{timestamp}] {message}")

# 안전한 스크립트 실행 함수 (실행 시간 측정 포함)
def run_script_safe(script_path, step_name=""):
    start_time = time.time()
    try:
        log_pipeline_status(f"🔄 {step_name} 시작: {script_path}")
        subprocess.run(['python', script_path], check=True)
        execution_time = time.time() - start_time
        log_pipeline_status(f"✅ {step_name} 완료: {script_path} (실행시간: {execution_time:.2f}초)")
        return execution_time
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        log_pipeline_status(f"⚠️ {step_name} 오류: {script_path} (실행시간: {execution_time:.2f}초), 내용: {e}")
        return execution_time

# 메인 파이프라인 함수
def main_pipeline():
    log_pipeline_status("🚀 실전 자동매매 파이프라인 시작")
    cycle_count = 0

    while True:
        cycle_count += 1
        cycle_start_time = time.time()
        
        log_pipeline_status(f"🔄 파이프라인 사이클 #{cycle_count} 시작")
        
        # 각 단계별 실행 시간 측정
        step_times = {}
        
        # 1. 캔들 데이터 수집
        step_times['collector'] = run_script_safe(COLLECTOR_SCRIPT, "📥 캔들 데이터 수집")
        
        # 데이터 안정화 대기
        wait_start = time.time()
        print("⏳ 데이터 안정화 대기 (10초)")
        time.sleep(10)
        step_times['wait1'] = time.time() - wait_start

        # 2. 기술지표 및 파동 계산
        step_times['calculator'] = run_script_safe(CALCULATOR_SCRIPT, "🛠️ 기술지표 및 파동 계산")
        
        # 데이터 안정화 대기
        wait_start = time.time()
        print("⏳ 데이터 안정화 대기 (10초)")
        time.sleep(10)
        step_times['wait2'] = time.time() - wait_start

        # 3. 통합 분석 (파동+패턴+프랙탈+통합메타)
        step_times['integrated'] = run_script_safe(INTEGRATED_SCRIPT, "🧠 통합 분석 (파동+패턴+프랙탈)")
        
        # 데이터 안정화 대기
        wait_start = time.time()
        print("⏳ 데이터 안정화 대기 (10초)")
        time.sleep(10)
        step_times['wait3'] = time.time() - wait_start

        # 4. 실시간 신호 생성 + 가상매매
        step_times['selector'] = run_script_safe(SELECTOR_SCRIPT, "📡 실시간 매매 신호 생성")
        
        # 데이터 안정화 대기
        wait_start = time.time()
        print("⏳ 데이터 안정화 대기 (10초)")
        time.sleep(10)
        step_times['wait4'] = time.time() - wait_start

        # 5. 가상매매 시뮬레이션 (virtual_trader.py)
        step_times['virtual_trader'] = run_script_safe(VIRTUAL_TRADER_SCRIPT, "🆕 가상매매 시뮬레이션")
        
        # 데이터 안정화 대기
        wait_start = time.time()
        print("⏳ 데이터 안정화 대기 (10초)")
        time.sleep(10)
        step_times['wait5'] = time.time() - wait_start

        # 6. 가상매매 학습 (virtual_trading_learner.py)
        step_times['virtual_learner'] = run_script_safe(VIRTUAL_LEARNER_SCRIPT, "🧠 가상매매 RL 학습")
        
        # 7. (실전 매매 executor는 우선 제외)
        # step_times['executor'] = run_script_safe(EXECUTOR_SCRIPT, "💰 실전 매매 실행")

        # 전체 사이클 실행 시간 계산
        total_execution_time = time.time() - cycle_start_time
        
        # 실행 시간 요약 로깅
        log_pipeline_status("📊 파이프라인 실행 시간 요약:")
        log_pipeline_status(f"   📥 캔들 수집: {step_times['collector']:.2f}초")
        log_pipeline_status(f"   🛠️ 기술지표 계산: {step_times['calculator']:.2f}초")
        log_pipeline_status(f"   🧠 통합 분석: {step_times['integrated']:.2f}초")
        log_pipeline_status(f"   📡 시그널 생성: {step_times['selector']:.2f}초")
        log_pipeline_status(f"   🆕 가상매매: {step_times['virtual_trader']:.2f}초")
        log_pipeline_status(f"   🧠 가상매매 학습: {step_times['virtual_learner']:.2f}초")
        log_pipeline_status(f"   ⏳ 대기 시간: {step_times['wait1'] + step_times['wait2'] + step_times['wait3'] + step_times['wait4'] + step_times['wait5']:.2f}초")
        log_pipeline_status(f"   🎯 총 실행 시간: {total_execution_time:.2f}초 ({total_execution_time/60:.1f}분)")
        
        # 다음 사이클까지 대기
        log_pipeline_status(f"⏳ 파이프라인 사이클 #{cycle_count} 완료, {TRADE_INTERVAL_MINUTES}분 대기 시작")
        time.sleep(TRADE_INTERVAL_MINUTES * 60)  # 분을 초로 변환

if __name__ == "__main__":
    main_pipeline()