import os
import sys
import subprocess
import time
import signal
from dotenv import load_dotenv
try:
    from market_analyzer import get_market_warning_list, get_all_krw_symbols
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from market_analyzer import get_market_warning_list, get_all_krw_symbols

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
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'rl_pipeline', 'scripts', 'data_collection')
# 🔧 [수정] SYSTEM_SCRIPT가 run_trading.py 엔진이 사용하는 경로와 일치하도록 수정
TRADE_DIR = os.path.join(ROOT_DIR, 'trade')
SYSTEM_SCRIPT = os.path.join(ROOT_DIR, 'rl_pipeline', 'absolute_zero_system.py')

# 3. 실행 환경 변수 설정 (현재 프로세스 환경에 병합)
# 데이터 저장소 디렉토리 생성
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')
if os.path.exists(DATA_DIR):
    if os.path.isfile(DATA_DIR):
        print(f"⚠️ 경고: {DATA_DIR}가 파일로 존재합니다. 삭제 후 디렉토리를 생성합니다.")
        os.remove(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
else:
    os.makedirs(DATA_DIR, exist_ok=True)

# DB 경로 설정 (4분할 구조 적용 - 환경변수 우선, 없으면 기본값)
# 1. 학습용 캔들 (대용량 - 공유 파일)
if not os.getenv('RL_DB_PATH'):
    # 기본값: learning_candles.db (학습용)
    os.environ['RL_DB_PATH'] = os.path.join(DATA_DIR, 'learning_candles.db')

# 2. 학습된 전략/모델 (Brain) - 코인별 격리 (Directory Mode)
# 사용자 요청: .../learning_strategies/btc_strategies.db 구조 지원
# STRATEGIES_DB_PATH가 디렉토리를 가리키게 설정 (확장자 없음)
if not os.getenv('STRATEGY_DB_PATH'):
    strategies_dir = os.path.join(DATA_DIR, 'learning_strategies')
    if not os.path.exists(strategies_dir):
        os.makedirs(strategies_dir, exist_ok=True)
    
    os.environ['STRATEGY_DB_PATH'] = strategies_dir
    os.environ['STRATEGIES_DB_PATH'] = strategies_dir # 복수형도 지원 (호환성)

# 3. 학습 중 매매 기록 (전략 DB와 같은 디렉토리 내 파일로 관리 권장)
# 여기서는 호환성을 위해 동일하게 설정하되, 엔진 내부에서 파일명 생성
if not os.getenv('TRADING_DB_PATH'):
    os.environ['TRADING_DB_PATH'] = os.path.join(DATA_DIR, 'trading_system.db')
    os.environ['TRADING_SYSTEM_DB_PATH'] = os.environ['TRADING_DB_PATH']

# 3-1. 공통 데이터 저장소 경로 설정 (중요: 하위 파이프라인이 올바른 위치를 찾도록 함)
if not os.getenv('DATA_STORAGE_PATH'):
    os.environ['DATA_STORAGE_PATH'] = DATA_DIR

# 4. 학습 설정 (실전 모드)
os.environ['ENABLE_STRATEGY_FILTERING'] = 'true'  # 실전 모드: 생존 법칙 및 스트레스 테스트 활성화
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
    
    # 🆕 거래유의 종목 필터링 (수집 단계 전처리)
    print("-" * 60)
    print("🛡️ 안전 학습 필터 적용 중...")
    try:
        warning_list = get_market_warning_list()
        if warning_list:
            print(f"   🚨 거래유의 종목 {len(warning_list)}개 식별됨: {', '.join(warning_list)}")
            
            # 대상 코인 리스트 확보
            target_str = os.getenv('TARGET_COINS', 'ALL')
            if target_str.upper() == 'ALL':
                all_coins = get_all_krw_symbols()
            else:
                all_coins = [c.strip() for c in target_str.split(',') if c.strip()]
            
            # 필터링 수행
            clean_coins = [c for c in all_coins if c not in warning_list]
            removed_count = len(all_coins) - len(clean_coins)
            
            if removed_count > 0:
                print(f"   🧹 {removed_count}개 위험 종목 수집 제외 처리 완료")
                
                # 환경 변수 업데이트 (구체적인 리스트로 덮어쓰기)
                os.environ['TARGET_COINS'] = ','.join(clean_coins)
                print(f"   ✅ 최종 수집 대상: {len(clean_coins)}개 코인")
            else:
                print("   ✅ 제외할 위험 종목이 대상에 포함되어 있지 않습니다.")
        else:
            print("   ✅ 현재 거래유의 종목이 없습니다.")
            
    except Exception as e:
        print(f"⚠️ 필터링 로직 수행 중 오류 (기본 설정으로 진행): {e}")

    print("="*60)
    
    iteration = 1
    
    while not _stopped:
        print(f"\n🎬 반복 루프 #{iteration} 시작")
        print("-" * 40)
        
        # Step 1: 수집 (Collector)
        # 참고: collector도 RL_DB_PATH 환경변수를 지원하도록 수정되어 있어야 함
        # (만약 collector가 아직 환경변수 미지원이라면, 엔진 수정이 필요할 수 있음)
        if not run_step("데이터 수집", os.path.join(SCRIPTS_DIR, 'candles_collector.py')):
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

