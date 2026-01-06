import os
import sys
import subprocess
import time
import signal
from dotenv import load_dotenv
try:
    from market_analyzer import get_market_warning_list_extended, get_all_krw_symbols
except ImportError:
    # 경로 문제로 실패 시 현재 디렉토리 추가
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from market_analyzer import get_market_warning_list_extended, get_all_krw_symbols

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
# 🆕 경로 변환 유틸리티 (Docker /workspace → Windows 절대 경로 호환)
def finalize_path(path):
    if not path: return None
    
    # 🚀 Docker 환경 감지: /workspace가 실제로 존재하면 Docker 환경
    if os.path.exists('/workspace'):
        # Docker 환경 - 경로 변환 없이 그대로 사용
        return os.path.abspath(path)
    
    # 🚀 Windows 호스트에서 직접 실행 시에만 /workspace 경로 변환
    if os.name == 'nt':
        if path.startswith('/workspace') or path.startswith('\\workspace'):
            rel_path = path.replace('/workspace', '', 1).replace('\\workspace', '', 1).lstrip('/\\')
            return os.path.join(ROOT_DIR, rel_path)
        if path.startswith('/') and not path.startswith('//'):
            return os.path.join(ROOT_DIR, path.lstrip('/'))
    
    return os.path.abspath(path)

# 📂 데이터 저장소 디렉토리 결정 (환경 변수 우선)
DATA_DIR = os.environ.get('DATA_STORAGE_PATH')
if not DATA_DIR:
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, 'data_storage'))
    os.environ['DATA_STORAGE_PATH'] = DATA_DIR
else:
    DATA_DIR = finalize_path(DATA_DIR)  # 🚀 os.path.abspath 대신 finalize_path 사용

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📂 데이터 저장소 디렉토리 생성: {DATA_DIR}")

# 🕯️ 1. 매매용 캔들 DB 경로 설정
if not os.environ.get('CANDLES_DB_PATH'):
    os.environ['CANDLES_DB_PATH'] = os.path.join(DATA_DIR, 'trade_candles.db')
os.environ['RL_DB_PATH'] = os.environ['CANDLES_DB_PATH'] # 하위 호환성

# 🧠 2. 전략 저장소 경로 설정
STRATEGY_DIR = os.environ.get('STRATEGY_DB_PATH')
if not STRATEGY_DIR:
    STRATEGY_DIR = os.path.join(DATA_DIR, 'learning_strategies')
    os.environ['STRATEGY_DB_PATH'] = STRATEGY_DIR
    os.environ['STRATEGIES_DB_PATH'] = STRATEGY_DIR
else:
    STRATEGY_DIR = finalize_path(STRATEGY_DIR)  # 🚀 os.path.abspath 대신 finalize_path 사용
    os.environ['STRATEGY_DB_PATH'] = STRATEGY_DIR
    os.environ['STRATEGIES_DB_PATH'] = STRATEGY_DIR

if not os.path.exists(STRATEGY_DIR):
    os.makedirs(STRATEGY_DIR, exist_ok=True)

# 🌐 2-1. 공용 전략 DB 설정 (사용자 지정: common_strategies.db)
if not os.environ.get('GLOBAL_STRATEGY_DB_PATH'):
    os.environ['GLOBAL_STRATEGY_DB_PATH'] = os.path.join(STRATEGY_DIR, 'common_strategies.db')
os.environ['LEARNING_RESULTS_DB_PATH'] = os.environ['GLOBAL_STRATEGY_DB_PATH']

# 📝 3. 실전/가상 매매 시스템 DB 설정
if not os.environ.get('TRADING_SYSTEM_DB_PATH'):
    os.environ['TRADING_SYSTEM_DB_PATH'] = os.path.join(DATA_DIR, 'trading_system.db')
os.environ['TRADING_DB_PATH'] = os.environ['TRADING_SYSTEM_DB_PATH']

# 🐍 PYTHONPATH 설정
if not os.environ.get('PYTHONPATH'):
    os.environ['PYTHONPATH'] = ROOT_DIR
else:
    if ROOT_DIR not in os.environ['PYTHONPATH']:
        os.environ['PYTHONPATH'] = f"{ROOT_DIR}{os.pathsep}{os.environ['PYTHONPATH']}"

# DB 경로 설정 로그 출력
print("-" * 60)
print(f"📊 데이터베이스 경로 설정 (Environment Variables):")
print(f"  📂 DATA_STORAGE: {os.environ['DATA_STORAGE_PATH']}")
print(f"  🕯️ CANDLES_DB:   {os.environ['RL_DB_PATH']}")
print(f"  🧠 STRATEGY_DB:  {os.environ['STRATEGY_DB_PATH']}")
print(f"  📝 TRADING_DB:   {os.environ['TRADING_DB_PATH']}")
print("-" * 60)

# 전역 중단 플래그 관리
_stopped = False

def signal_handler(signum, frame):
    global _stopped
    print("\n\n⏹️ 트레이딩 봇 중단 신호 감지!")
    _stopped = True

signal.signal(signal.SIGINT, signal_handler)

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
            print(f"⚠️ 동적 필터링 실패 (이전 설정 유지): {e}")

        
        # 🚀 [Step 순차 실행] 파이프라인 정합성 유지
        
        # Step 1: 데이터 수집 (공용 엔진 사용)
        if not run_step("데이터 수집", os.path.join(SCRIPTS_DIR, 'candles_collector.py')):
            time.sleep(5)
        
        # Step 2: 지표 계산 (공용 엔진 사용)
        if not run_step("지표 계산", os.path.join(SCRIPTS_DIR, 'candles_calculate.py')):
            pass

        # Step 3: 통합 분석 (공용 엔진 사용)
        if not run_step("통합 분석", os.path.join(SCRIPTS_DIR, 'candles_integrated.py')):
            pass
            
        # Step 4: 실시간 시그널 생성 (최적화된 엔진 순차 실행)
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

