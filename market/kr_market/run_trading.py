import os
import sys
import subprocess
import time
import signal
from datetime import datetime, time as dtime, timedelta, timezone
from dotenv import load_dotenv

# ==========================================
# 🆕 market_analyzer 모듈 임포트 설정
# ==========================================
# KRX 모드에서는 kr_market 폴더 내의 모듈이 필요할 수 있으나,
# 현재 구조상 market_analyzer는 market/coin_market 에 위치해 있습니다.
# KRX 모드에서는 별도의 유의 종목 리스트 로직이나 더미 함수가 필요합니다.
# 여기서는 코인용 모듈을 재활용하거나 예외 처리를 통해 넘어갑니다.

try:
    # 1. 같은 디렉토리 우선 시도
    from market_analyzer import get_market_warning_list_extended, get_all_krw_symbols
except ImportError:
    # 2. 상위 경로 탐색 (market/coin_market/market_analyzer.py 등)
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../coin_market')))
        from market_analyzer import get_market_warning_list_extended, get_all_krw_symbols
    except ImportError:
        # 3. 모듈이 없으면 KRX 전용 더미 함수 정의
        print("⚠️ market_analyzer 모듈을 찾을 수 없어 KRX 기본 함수를 사용합니다.")
        
        def get_market_warning_list_extended():
            # KRX 관리종목 등은 krx_collector 내부에서 처리되거나 별도 로직 필요
            # 여기서는 빈 리스트 반환
            return []

        def get_all_krw_symbols():
            # 전체 종목 리스트는 krx_collector가 처리하지만,
            # TARGET_COINS=ALL 일 때 필요하다면 pykrx 등을 써야 함.
            # 하지만 런너 레벨에서는 일단 빈 리스트나 에러 방지용 리스트 반환
            return []

# ==========================================
# 🚀 트레이딩 인스턴스 실행기 (Trading Runner)
# ==========================================
# 이 파일은 실전/섀도우 트레이딩을 수행합니다.
# 설정 파일(config_trading.env)을 로드하고, trade 폴더의 스크립트를 실행합니다.
# ==========================================

# 1. 설정 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, 'config_trading.env')
load_dotenv(ENV_PATH)

# 2. 경로 설정 (공용 스크립트 위치)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../')) 
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'rl_pipeline/scripts/data_collection')
TRADE_DIR = os.path.join(ROOT_DIR, 'trade')

# 트레이딩 파이프라인 스크립트들
# 1. 수집: rl_pipeline/scripts/data_collection/candles_collector.py (공유)
# 2. 계산: rl_pipeline/scripts/data_collection/candles_calculate.py (공유)
# 3. 통합: rl_pipeline/scripts/data_collection/candles_integrated.py (공유 - 선택 사항)
# 4. 시그널 & 매매: trade 폴더 내 스크립트 (추후 연결)

# 3. 실행 환경 변수 설정
# 데이터 저장소 디렉토리 생성 (절대 경로 보장)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, 'data_storage'))
os.environ['DATA_STORAGE_PATH'] = DATA_DIR  # 하위 프로세스를 위해 가장 먼저 설정

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"📂 데이터 저장소 디렉토리 생성: {DATA_DIR}")

# DB 경로 설정 (4분할 구조 적용 - DATA_DIR 기반으로 통일)
# 1. 매매용 캔들 (경량/최신)
os.environ['RL_DB_PATH'] = os.path.join(DATA_DIR, 'trade_candles.db')
os.environ['CANDLES_DB_PATH'] = os.environ['RL_DB_PATH']  # Signal Selector 호환성

# 2. 학습된 전략/모델 (Brain) - 학습 봇이 만든 것을 공유받아 읽기/쓰기
# 🔧 수정: run_learning.py와 동일한 경로를 사용해야 전략 DB를 올바르게 로드
# 학습 봇이 저장한 경로: /workspace/market/coin_market/data_storage/learning_strategies/
STRATEGY_DIR = os.path.join(DATA_DIR, 'learning_strategies')

# 전략 폴더 및 공용 DB 확인/생성
if not os.path.exists(STRATEGY_DIR):
    try:
        os.makedirs(STRATEGY_DIR, exist_ok=True)
        print(f"📂 전략 저장소 폴더 생성: {STRATEGY_DIR}")
    except Exception as e:
        print(f"⚠️ 전략 저장소 폴더 생성 실패: {e}")

# 1) 전략 저장소 루트 (개별 코인 DB들이 있는 폴더)
os.environ['STRATEGY_DB_PATH'] = STRATEGY_DIR
os.environ['STRATEGIES_DB_PATH'] = STRATEGY_DIR

# 2) 글로벌 전략 DB (공용 전략) - 명시적 설정
COMMON_DB_PATH = os.path.join(STRATEGY_DIR, 'common_strategies.db')
os.environ['GLOBAL_STRATEGY_DB_PATH'] = COMMON_DB_PATH

# 3) 학습 결과 DB (Learning Results) - 하위 호환성 및 명시적 설정
# 일부 구형 모듈이 LEARNING_RESULTS_DB_PATH를 찾을 수 있으므로 공용 DB로 연결
os.environ['LEARNING_RESULTS_DB_PATH'] = COMMON_DB_PATH

# 3. 실전/섀도우 매매 기록 (Records) - 실전 기록 분리
os.environ['TRADING_DB_PATH'] = os.path.join(DATA_DIR, 'trading_system.db')
os.environ['TRADING_SYSTEM_DB_PATH'] = os.environ['TRADING_DB_PATH'] # Executor 호환성

# 3-1. 공통 데이터 저장소 경로 설정 (중복 설정이지만 명시적으로)
os.environ['DATA_STORAGE_PATH'] = DATA_DIR

os.environ['PYTHONPATH'] = ROOT_DIR

# DB 경로 설정 로그 출력
print("-" * 60)
print(f"📊 데이터베이스 경로 설정 (Environment Variables):")
print(f"  📂 DATA_STORAGE: {os.environ['DATA_STORAGE_PATH']}")
print(f"  🕯️ CANDLES_DB:   {os.environ['RL_DB_PATH']}")
print(f"  🧠 STRATEGY_DB:  {os.environ['STRATEGY_DB_PATH']}")
print(f"  📝 TRADING_DB:   {os.environ['TRADING_DB_PATH']}")
print("-" * 60)

# 전역 중단 플래그
_stopped = False

def signal_handler(signum, frame):
    global _stopped
    print("\n\n⏹️ 트레이딩 봇 중단 신호 감지! (안전 종료 중...)")
    _stopped = True

signal.signal(signal.SIGINT, signal_handler)

def check_krx_market_hours():
    """KRX 장 운영 시간 확인 (09:00 ~ 15:30, 평일)"""
    # KST = UTC+9
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    
    # 1. 주말 체크 (0=월, 6=일) -> 5=토, 6=일
    if now.weekday() >= 5:
        return False, "주말 휴장"
        
    # 2. 시간 체크
    current_time = now.time()
    market_start = dtime(9, 0)
    market_end = dtime(15, 30)
    
    if market_start <= current_time <= market_end:
        return True, "장 운영 중 (Regular Session)"
    
    return False, f"장 마감 (현재: {current_time.strftime('%H:%M:%S')})"

def run_step(script_name, script_path):
    """스크립트 실행 도우미"""
    if _stopped: return False
    
    print(f"\n🔄 [Step] {script_name} 실행 중...")
    
    try:
        # 현재 환경변수(os.environ)를 그대로 자식 프로세스에 전달
        result = subprocess.run(
            [sys.executable, script_path], 
            cwd=ROOT_DIR,
            env=os.environ,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} 완료")
            return True
        else:
            print(f"❌ {script_name} 실패 (Exit Code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ 실행 중 치명적 오류: {e}")
        return False

def main():
    print(f"🚀 [[ Auto Trader Trading Bot: {os.getenv('INSTANCE_NAME', 'Unknown')} ]]")
    print(f"📍 작업 공간: {BASE_DIR}")
    print(f"💰 모드: {os.getenv('TRADING_MODE', 'SHADOW')}")
    
    # 1. 초기 타겟 설정 보존 (매 루프마다 이 설정을 기준으로 다시 필터링)
    INITIAL_TARGET_STR = os.getenv('TARGET_COINS', 'ALL')

    print("="*60)
    
    iteration = 1
    
    while not _stopped:
        # 0. 장 운영 시간 체크 (KRX)
        is_open, msg = check_krx_market_hours()
        if not is_open:
            # 장 마감 시에는 1분 간격으로 대기하며 상태 체크
            print(f"\nzzz {msg}. 잠시 대기합니다... (60초)", end='\r')
            time.sleep(60)
            continue

        loop_start_time = time.time()
        print(f"\n🎬 트레이딩 루프 #{iteration} 시작")
        print("-" * 40)

        # 🆕 루프마다 동적 필터링 적용 (가격 변동 반영)
        # 0.003원 -> 0.006원 (재진입), 0.006원 -> 0.003원 (퇴출)
        try:
            # 확장된 유의 종목 리스트 조회 (실시간)
            warning_list = get_market_warning_list_extended()
            
            # 대상 코인 리스트 확보 (초기 설정 기준)
            if INITIAL_TARGET_STR.upper() == 'ALL':
                all_coins = get_all_krw_symbols()
            else:
                all_coins = [c.strip() for c in INITIAL_TARGET_STR.split(',') if c.strip()]
            
            if warning_list:
                # 필터링 수행
                clean_coins = [c for c in all_coins if c not in warning_list]
                removed_count = len(all_coins) - len(clean_coins)
                
                if removed_count > 0:
                    print(f"🛡️ 안전 거래 필터: {removed_count}개 종목 제외 (유의/동전주)")
                    if removed_count <= 10:
                        excluded = [c for c in all_coins if c in warning_list]
                        print(f"   (제외됨: {', '.join(excluded)})")
                    
                    # 환경 변수 업데이트 (현재 루프에 적용)
                    os.environ['TARGET_COINS'] = ','.join(clean_coins)
                else:
                    # 제외할 게 없으면 전체 적용
                    os.environ['TARGET_COINS'] = ','.join(all_coins)
            else:
                # 유의 종목이 없으면 전체 적용
                os.environ['TARGET_COINS'] = ','.join(all_coins)
                
        except Exception as e:
            # print(f"⚠️ 동적 필터링 실패 (이전 설정 유지): {e}")
            # KRX 모드에서 코인용 market_analyzer가 없거나 오동작할 수 있으므로 조용히 넘어감
            pass

        
        # Step 1: 데이터 수집 (KRX 전용 수집기 사용)
        # config_trading.env에 설정된 짧은 기간(DAYS_BACK)만큼만 수집
        if not run_step("KRX 데이터 수집", os.path.join(SCRIPTS_DIR, 'krx_collector.py')):
            time.sleep(5)
        
        # Step 2: 지표 계산 (공용 엔진 사용)
        if not run_step("지표 계산", os.path.join(SCRIPTS_DIR, 'candles_calculate.py')):
            pass

        # Step 3: 통합 분석 (공용 엔진 사용)
        if not run_step("통합 분석", os.path.join(SCRIPTS_DIR, 'candles_integrated.py')):
            pass
            
        # Step 4: 실시간 시그널 생성 (Trading 전용)
        if not run_step("시그널 생성", os.path.join(TRADE_DIR, 'strategy_signal_generator.py')):
            pass

        # Step 5: 가상/실전 매매 (Trading 전용)
        if not run_step("매매 실행", os.path.join(TRADE_DIR, 'virtual_trade_executor.py')):
            pass

        # Step 6: 매매 학습 (Trading 전용)
        if not run_step("매매 학습", os.path.join(TRADE_DIR, 'virtual_trade_learner.py')):
            pass

        # Step 7: 실전/시뮬레이션 매매 (Real/Simulation Executor)
        if not run_step("실전/시뮬레이션 매매", os.path.join(TRADE_DIR, 'trade_executor.py')):
            pass
        
        loop_end_time = time.time()
        duration = loop_end_time - loop_start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print(f"\n✅ 루프 #{iteration} 종료 (소요 시간: {minutes}분 {seconds}초). 대기 중...")
        iteration += 1
        
        wait_time = int(os.getenv('LOOP_WAIT_SECONDS', 10))
        for i in range(wait_time, 0, -1):
            if _stopped: break
            print(f"⏳ {i}초 후 다음 루프...", end='\r')
            time.sleep(1)

if __name__ == "__main__":
    main()
