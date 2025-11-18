"""
RL Pipeline Runner - 새로운 4단계 핵심 프로세스
1. 캔들 데이터 수집 (rl_candles_collector.py)
2. 핵심 지표 계산 (rl_candles_calculate.py) 
3. 패턴/파동 계산 (rl_candles_integrated.py)
4. 새로운 파이프라인 실행 (absolute_zero_system.py) - 전략생성 → Self-play진화 → 레짐라우팅 → 통합분석

무제한 반복 실행으로 지속적인 학습 및 전략 개선
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime
import os
import json
import logging
import signal
from dotenv import load_dotenv

# 환경변수 파일 로드
if os.name == 'nt':  # Windows
    env_path = os.path.join(os.path.dirname(__file__), 'rl_pipeline_config.env')
    load_dotenv(env_path)
else:  # Linux/Mac
    load_dotenv('/workspace/rl_pipeline/rl_pipeline_config.env')

# 🔥 전역 중단 플래그
_stopped = False

def _signal_handler(signum, frame):
    """SIGINT (Ctrl+C) 신호 핸들러"""
    global _stopped
    print("\n\n⏹️ 중단 신호 감지됨 (Ctrl+C)")
    print("🔄 현재 작업 완료 후 종료합니다...")
    _stopped = True

class RLPipelineManager:
    """RL 파이프라인 관리자 - 새로운 4단계 핵심 프로세스"""
    
    def __init__(self):
        global _stopped
        _stopped = False
        
        # 🔥 SIGINT 핸들러 등록
        signal.signal(signal.SIGINT, _signal_handler)
        
        # 경로 설정
        if os.name == 'nt':  # Windows
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # auto_trader 루트
        else:  # Linux/Mac
            self.base_dir = "/workspace"
        
        self.start_time = datetime.now()
        
        # 재실행 설정 (환경변수 지원)
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '30'))
    
    def run_pipeline(self):
        """RL 파이프라인 실행 - 새로운 4단계 핵심 프로세스"""
        print("🚀 RL 파이프라인 시작")
        print("=" * 60)
        print("📊 새로운 4단계 프로세스:")
        print("1. 캔들 데이터 수집 (rl_candles_collector.py)")
        print("2. 핵심 지표 계산 (rl_candles_calculate.py)")
        print("3. 패턴/파동 계산 (rl_candles_integrated.py)")
        print("4. 새로운 파이프라인 실행 (absolute_zero_system.py)")
        print("   └─ 전략생성 → Self-play진화 → 레짐라우팅 → 통합분석")
        print("")
        print("🔄 무제한 반복 실행 (캔들 수집부터 다시 시작)")
        print(f"🔄 재실행 설정: 최대 {self.max_retries}회 재시도, {self.retry_delay}초 대기")
        print("⚡ 즉시 다음 반복 실행 (대기 없음)")
        print("💡 대기 시간 설정: PIPELINE_WAIT_SECONDS 환경변수 사용")
        print("=" * 60)
        
        iteration = 1
        
        try:
            while not _stopped:  # 🔥 _stopped 플래그 체크
                print(f"\n🔄 파이프라인 반복 #{iteration}")
                print("=" * 60)
                
                step_results = {}
                
                try:
                    # 새로운 4단계 실행
                    step_results['step1'] = self._run_step_with_retry("캔들 데이터 수집", "rl_pipeline/rl_candles_collector.py")
                    if _stopped: break  # 🔥 중단 체크
                    
                    step_results['step2'] = self._run_step_with_retry("핵심 지표 계산", "rl_pipeline/rl_candles_calculate.py")
                    if _stopped: break  # 🔥 중단 체크
                    
                    step_results['step3'] = self._run_step_with_retry("패턴/파동 계산", "rl_pipeline/rl_candles_integrated.py")
                    if _stopped: break  # 🔥 중단 체크
                    
                    step_results['step4'] = self._run_step_with_retry("새로운 파이프라인 실행", "rl_pipeline/absolute_zero_system.py")
                    if _stopped: break  # 🔥 중단 체크
                    
                    # 반복 완료 후 즉시 다음 반복으로 진행
                    print(f"\n✅ 파이프라인 반복 #{iteration} 완료")
                    print("🔄 즉시 다음 반복 시작...")
                    print("⏹️ 중단하려면 Ctrl+C를 누르세요.")
                    
                    # 선택적 대기 (환경변수로 제어)
                    wait_seconds = int(os.getenv('PIPELINE_WAIT_SECONDS', '0'))
                    if wait_seconds > 0:
                        print(f"⏰ {wait_seconds}초 대기 중...")
                        wait_interval = int(os.getenv('WAIT_INTERVAL', '10'))  # 대기 간격 (기본 10초)
                        for i in range(wait_seconds, 0, -wait_interval):
                            print(f"⏳ {i}초 남음...", end="\r")
                            time.sleep(wait_interval)
                        print("\n🔄 다음 반복 시작!")
                    
                    iteration += 1
                    
                except KeyboardInterrupt:
                    print("\n\n⏹️ 사용자에 의해 파이프라인이 중단되었습니다.")
                    break
                except SystemExit:
                    print("\n\n⏹️ 시스템 종료 신호 감지됨.")
                    break
                except Exception as e:
                    print(f"\n❌ 파이프라인 실행 중 오류 발생: {e}")
                    print("🔄 재시도 중...")
                    time.sleep(self.retry_delay)
                    continue
                finally:
                    if iteration > 1:  # 첫 번째 반복이 아닌 경우에만 요약 출력
                        self._print_iteration_summary(step_results, iteration - 1)
        
        finally:
            if _stopped:
                print("\n\n✅ 파이프라인이 정상적으로 종료되었습니다.")
                print(f"📊 총 {iteration - 1}번의 반복을 완료했습니다.")
                print("=" * 60)
    


    def _run_step_with_retry(self, step_name: str, script_path: str):
        """재실행 기능이 포함된 단계 실행"""
        global _stopped
        
        # 🔥 중단 플래그 체크
        if _stopped:
            return {
                "status": "interrupted", 
                "message": f"{step_name} 중단 요청됨",
                "attempts": 0
            }
        
        print(f"\n📊 {step_name}")
        print("-" * 30)
        print("⚠️ 이 단계는 많은 데이터를 처리하므로 시간이 오래 걸릴 수 있습니다.")
        print("📈 예상 소요시간: 30-90분 (5,925개 태스크, 4개 인터벌)")
        print("🔄 진행률은 실시간으로 표시됩니다.")
        print(f"🔄 재실행 설정: 최대 {self.max_retries}회 재시도")
        print("-" * 30)
        
        for attempt in range(self.max_retries + 1):  # 0부터 시작하므로 +1
            # 🔥 중단 플래그 체크
            if _stopped:
                return {
                    "status": "interrupted", 
                    "message": f"{step_name} 중단 요청됨",
                    "attempts": attempt
                }
            
            try:
                if attempt > 0:
                    # 🔥 중단 플래그 체크
                    if _stopped:
                        return {
                            "status": "interrupted", 
                            "message": f"{step_name} 중단 요청됨",
                            "attempts": attempt
                        }
                    print(f"🔄 {step_name} 재시도 {attempt}/{self.max_retries}...")
                    print(f"⏳ {self.retry_delay}초 대기 중...")
                    # 중단 플래그를 체크하면서 대기
                    for _ in range(self.retry_delay):
                        if _stopped:
                            return {
                                "status": "interrupted", 
                                "message": f"{step_name} 중단 요청됨",
                                "attempts": attempt
                            }
                        time.sleep(1)
                
                # 🆕 스크립트 경로 처리 및 실행
                if script_path.startswith("rl_pipeline/"):
                    # 상대 경로인 경우 절대 경로로 변환
                    full_script_path = os.path.join(self.base_dir, script_path)
                else:
                    full_script_path = script_path
                
                # 스크립트 실행 (신호 전달 가능하도록)
                process = subprocess.Popen([sys.executable, full_script_path], 
                                          cwd=self.base_dir)
                
                # 프로세스 완료 대기 (중단 플래그 체크 포함)
                try:
                    while process.poll() is None:
                        if _stopped:
                            print(f"\n⏹️ {step_name} 프로세스 종료 중...")
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise KeyboardInterrupt("사용자 중단")
                        time.sleep(0.1)  # 짧은 간격으로 체크
                    
                    # 프로세스 종료 코드 확인
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, full_script_path)
                except KeyboardInterrupt:
                    if not _stopped:
                        _stopped = True
                    if process.poll() is None:
                        print(f"\n⏹️ {step_name} 프로세스 종료 중...")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
                
                print(f"✅ {step_name} 완료")
                return {
                    "status": "success", 
                    "message": f"{step_name} 완료",
                    "attempts": attempt + 1
                }
                
            except subprocess.CalledProcessError as e:
                print(f"❌ {step_name} 실패 (시도 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                # 🔥 중단 플래그 체크
                if _stopped:
                    return {
                        "status": "interrupted", 
                        "message": f"{step_name} 중단 요청됨",
                        "attempts": attempt + 1
                    }
                
                if attempt < self.max_retries:
                    print(f"🔄 재시도 예정... ({self.retry_delay}초 후)")
                else:
                    print(f"❌ 최대 재시도 횟수 도달. 다음 단계로 진행합니다.")
                    return {
                        "status": "error", 
                        "message": f"{step_name} 실패 (최대 재시도 횟수 도달): {e}",
                        "attempts": attempt + 1
                    }
                    
            except KeyboardInterrupt:
                print(f"\n⏹️ {step_name} 사용자에 의해 중단됨")
                _stopped = True  # 🔥 전역 플래그 설정
                return {
                    "status": "interrupted", 
                    "message": f"{step_name} 사용자에 의해 중단됨",
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                print(f"❌ {step_name} 예상치 못한 오류 (시도 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                # 🔥 중단 플래그 체크
                if _stopped:
                    return {
                        "status": "interrupted", 
                        "message": f"{step_name} 중단 요청됨",
                        "attempts": attempt + 1
                    }
                
                if attempt < self.max_retries:
                    print(f"🔄 재시도 예정... ({self.retry_delay}초 후)")
                else:
                    print(f"❌ 최대 재시도 횟수 도달. 다음 단계로 진행합니다.")
                    return {
                        "status": "error", 
                        "message": f"{step_name} 예상치 못한 오류 (최대 재시도 횟수 도달): {e}",
                        "attempts": attempt + 1
                    }
    
    def _print_iteration_summary(self, step_results=None, iteration=None):
        """반복 요약 출력"""
        print(f"\n📊 파이프라인 반복 #{iteration} 결과:")
        print("-" * 40)
        
        # 단계별 결과 출력
        if step_results:
            step_names = {
                'step1': '1. 캔들 데이터 수집',
                'step2': '2. 핵심 지표 계산', 
                'step3': '3. 패턴/파동 계산',
                'step4': '4. 새로운 파이프라인 실행'
            }
            
            for step_key, step_name in step_names.items():
                if step_key in step_results:
                    result = step_results[step_key]
                    
                    # result가 딕셔너리가 아닌 경우 처리
                    if not isinstance(result, dict):
                        print(f"  ⚠️ {step_name}: 예상치 못한 결과 타입 ({type(result).__name__})")
                        continue
                    
                    status_icon = "✅" if result.get('status') == 'success' else "❌" if result.get('status') == 'error' else "⚠️" if result.get('status') == 'interrupted' else "⏭️"
                    
                    # 재시도 정보 포함
                    attempts_info = ""
                    if 'attempts' in result:
                        attempts_info = f" (시도 {result['attempts']}회)"
                    
                    print(f"  {status_icon} {step_name}: {result.get('status', 'unknown')}{attempts_info}")
                    
                    if result.get('status') == 'error':
                        print(f"     💬 {result.get('message', '오류 메시지 없음')}")
                    elif result.get('status') == 'success' and 'attempts' in result and result['attempts'] > 1:
                        print(f"     🎉 {result['attempts']}번째 시도에서 성공!")
        
        print("-" * 40)
    


def main():
    """메인 실행 함수 - 새로운 4단계 핵심 프로세스"""
    parser = argparse.ArgumentParser(description='RL 파이프라인 - 새로운 4단계 핵심 프로세스')
    
    # 간단한 옵션들만 유지
    parser.add_argument('--wait-seconds', type=int, default=0,
                       help='반복 간 대기 시간 (초, 기본값: 0)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='최대 재시도 횟수 (기본값: 3)')
    parser.add_argument('--retry-delay', type=int, default=30,
                       help='재시도 간 대기 시간 (초, 기본값: 30)')
    
    args = parser.parse_args()
    
    print("🚀 RL 파이프라인 - 5단계 핵심 프로세스")
    print("🎯 목표: 캔들 수집 → 지표 계산 → 패턴 분석 → 시뮬레이션 → 학습")
    print("=" * 60)
    print("📊 실행 설정:")
    print(f"   - 반복 간 대기: {args.wait_seconds}초")
    print(f"   - 최대 재시도: {args.max_retries}회")
    print(f"   - 재시도 대기: {args.retry_delay}초")
    print("=" * 60)
    
    # 환경변수 설정
    if args.wait_seconds > 0:
        os.environ['PIPELINE_WAIT_SECONDS'] = str(args.wait_seconds)
    os.environ['MAX_RETRIES'] = str(args.max_retries)
    os.environ['RETRY_DELAY'] = str(args.retry_delay)
    
    # 파이프라인 실행
    pipeline_manager = RLPipelineManager()
    pipeline_manager.run_pipeline()


if __name__ == "__main__":
    main()
