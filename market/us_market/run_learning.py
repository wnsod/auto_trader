import os
import sys
import subprocess
import time
import signal
from dotenv import load_dotenv
try:
    # 코인 관련 모듈 (KRX 모드에서는 선택 사항)
    from market_analyzer import get_market_warning_list, get_all_krw_symbols
except ImportError:
    # 상위 경로 추가 후 재시도
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from market_analyzer import get_market_warning_list, get_all_krw_symbols
    except ImportError:
        # KRX 모드 등에서 모듈이 아예 없는 경우 더미 함수 처리
        get_market_warning_list = lambda: []
        get_all_krw_symbols = lambda: []

# ==========================================
# 🚀 학습 인스턴스 실행기 (Learning Runner)
# ==========================================
# 이 파일은 특정 시장의 데이터를 수집하고 강화학습 모델을 학습시킵니다.
# 설정 파일(config_learning.env)을 로드합니다.
# ==========================================

# 1. 설정 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 폴더
ENV_PATH = os.path.join(BASE_DIR, 'config_learning.env')
load_dotenv(ENV_PATH)

# 2. 경로 설정 (공용 스크립트 위치)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../')) # 프로젝트 루트 (auto_trader)
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'rl_pipeline/scripts/data_collection')
SYSTEM_SCRIPT = os.path.join(ROOT_DIR, 'rl_pipeline/absolute_zero_system.py')

# 3. 실행 환경 변수 설정 (현재 프로세스 환경에 병합)
# 데이터 저장소 디렉토리 생성
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')
if os.path.exists(DATA_DIR):
    if os.path.isfile(DATA_DIR):
        print(f"⚠️ 경고: {DATA_DIR}가 파일로 존재합니다. 삭제 후 디렉토리를 생성합니다.")
        os.remove(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
    elif not os.path.isdir(DATA_DIR):
        # 심볼릭 링크 등 기타 케이스
        pass
else:
    os.makedirs(DATA_DIR, exist_ok=True)

# DB 경로 설정 (4분할 구조 적용)
# 1. 학습용 캔들 (대용량 - 공유 파일)
os.environ['RL_DB_PATH'] = os.path.join(DATA_DIR, 'learning_candles.db')

# 2. 학습된 전략/모델 (Brain) - KRX 종목별 개별 DB 모드
strategies_dir = os.path.join(DATA_DIR, 'learning_strategies')
if not os.path.exists(strategies_dir):
    os.makedirs(strategies_dir)

# [중요] 디렉토리 경로 지정 -> 시스템이 종목별로 DB 파일 자동 생성 (예: 005930_strategies.db)
os.environ['STRATEGY_DB_PATH'] = strategies_dir
os.environ['STRATEGIES_DB_PATH'] = strategies_dir
os.environ['RL_STRATEGY_DB_PATH'] = strategies_dir

# 임시 잠금 파일 제거 (클린 스타트) - 디렉토리 내 모든 잠금 파일 제거
try:
    for filename in os.listdir(strategies_dir):
        if filename.endswith('-journal') or filename.endswith('-shm') or filename.endswith('-wal'):
            try:
                os.remove(os.path.join(strategies_dir, filename))
            except:
                pass
except:
    pass

# 3. 학습 중 매매 기록 (전략 DB와 같은 디렉토리 내 파일로 관리 권장)
# 여기서는 호환성을 위해 동일하게 설정하되, 엔진 내부에서 파일명 생성
os.environ['TRADING_DB_PATH'] = strategies_dir

# 3-1. 공통 데이터 저장소 경로 설정 (중요: 하위 파이프라인이 올바른 위치를 찾도록 함)
os.environ['DATA_STORAGE_PATH'] = DATA_DIR

# 4. 학습 설정 (실전 모드)
os.environ['ENABLE_STRATEGY_FILTERING'] = 'true'  # 실전 모드: 생존 법칙 및 스트레스 테스트 활성화
os.environ['AZ_CANDLE_DAYS'] = '730'  # 🔥 일봉 2년치(730일) 데이터 로드 강제 설정 (기본값 60일 -> 730일)
# os.environ['STRICT_MODE'] = 'true' # 필요시 추가 설정

# Python Path에 프로젝트 루트 추가 (모듈 임포트 위해)
os.environ['PYTHONPATH'] = ROOT_DIR

# 전역 중단 플래그
_stopped = False

def signal_handler(signum, frame):
    global _stopped
    print("\n\n⏹️ 인스턴스 중단 신호 감지! (종료 중...)")
    _stopped = True

signal.signal(signal.SIGINT, signal_handler)

def run_step(script_name, script_path):
    """스크립트 실행 도우미"""
    if _stopped: return False
    
    print(f"\n🔄 [Step] {script_name} 실행 중...")
    print(f"   📂 DB: {os.environ['RL_DB_PATH']}")
    
    try:
        # 현재 환경변수(os.environ)를 그대로 자식 프로세스에 전달
        result = subprocess.run(
            [sys.executable, script_path], 
            cwd=ROOT_DIR, # 실행 위치는 프로젝트 루트로 (임포트 경로 문제 방지)
            env=os.environ,
            check=False # 에러 나도 여기서 죽지 않고 처리
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
    print(f"🚀 [[ Auto Trader Instance: {os.getenv('INSTANCE_NAME', 'Unknown')} ]]")
    print(f"📍 작업 공간: {BASE_DIR}")
    
    # 🆕 거래유의 종목 필터링 (US 모드에서는 비활성화)
    print("-" * 60)
    print("🛡️ US 주식 모드로 실행 중입니다.")
    print("   (코인 관련 유의종목 필터링은 건너뜁니다)")
    print("="*60)
    
    iteration = 1
    
    while not _stopped:
        print(f"\n🎬 반복 루프 #{iteration} 시작")
        print("-" * 40)
        
        # Step 1: 수집 (Collector) - US 전용 수집기 사용
        if not run_step("US 데이터 수집", os.path.join(SCRIPTS_DIR, 'us_collector.py')):
            time.sleep(5) # 실패 시 잠시 대기
        
        # Step 2: 계산 (Calculate)
        if not run_step("지표 계산", os.path.join(SCRIPTS_DIR, 'candles_calculate.py')):
            pass
            
        # Step 3: 통합 분석 (Integrated)
        if not run_step("통합 분석", os.path.join(SCRIPTS_DIR, 'candles_integrated.py')):
            pass
            
        # Step 4: 시스템 실행 (System)
        # 시스템 스크립트도 DB 경로를 환경변수로 받는지 확인 필요
        if not run_step("전략 시스템", SYSTEM_SCRIPT):
            pass
            
        print(f"\n✅ 반복 #{iteration} 종료. 잠시 대기 후 재시작...")
        iteration += 1
        
        # 대기 시간
        wait_time = int(os.getenv('LOOP_WAIT_SECONDS', 10))
        for i in range(wait_time, 0, -1):
            if _stopped: break
            print(f"⏳ {i}초 후 다음 반복...", end='\r')
            time.sleep(1)

if __name__ == "__main__":
    main()

