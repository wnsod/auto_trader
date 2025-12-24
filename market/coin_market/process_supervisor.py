import subprocess
import time
import signal
import sys

CHECK_INTERVAL = 10  # 초 단위, 죽었는지 체크 주기

PROCESSES = {
    "learning": {
        "cmd": ["python", "/workspace/market/coin_market/run_learning.py"],
        "proc": None,
        "restarts": 0,
    },
    "trading": {
        "cmd": ["python", "/workspace/market/coin_market/run_trading.py"],
        "proc": None,
        "restarts": 0,
    },
}


def start_process(name: str):
    info = PROCESSES[name]
    if info["proc"] is not None and info["proc"].poll() is None:
        # 이미 살아있으면 무시
        return

    print(f"[Supervisor] Starting {name} ...", flush=True)
    proc = subprocess.Popen(
        info["cmd"],
        stdout=subprocess.PIPE,  # 필요 없으면 None으로 변경
        stderr=subprocess.PIPE,  # 필요 없으면 None으로 변경
        text=True,
        bufsize=1,
    )
    info["proc"] = proc
    info["restarts"] += 1
    print(f"[Supervisor] {name} started with PID {proc.pid}, "
          f"restart count={info['restarts']}", flush=True)


def check_and_restart():
    for name, info in PROCESSES.items():
        proc = info["proc"]
        if proc is None:
            # 아직 시작 안됐으면 시작
            start_process(name)
            continue

        ret = proc.poll()
        if ret is not None:
            # 프로세스가 종료됨
            print(f"[Supervisor] {name} exited with code {ret}. Restarting...", flush=True)
            start_process(name)


def terminate_all():
    print("[Supervisor] Terminating all child processes...", flush=True)
    for name, info in PROCESSES.items():
        proc = info["proc"]
        if proc is None:
            continue

        if proc.poll() is None:
            try:
                print(f"[Supervisor] Sending SIGTERM to {name} (PID {proc.pid})", flush=True)
                proc.terminate()
            except Exception as e:
                print(f"[Supervisor] Error terminating {name}: {e}", flush=True)

    # 조금 기다렸다가 강제 종료
    time.sleep(5)
    for name, info in PROCESSES.items():
        proc = info["proc"]
        if proc is None:
            continue
        if proc.poll() is None:
            try:
                print(f"[Supervisor] Sending SIGKILL to {name} (PID {proc.pid})", flush=True)
                proc.kill()
            except Exception as e:
                print(f"[Supervisor] Error killing {name}: {e}", flush=True)


def main():
    print("[Supervisor] process_supervisor.py starting...", flush=True)
    # 최초 기동
    for name in PROCESSES.keys():
        start_process(name)

    try:
        while True:
            check_and_restart()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("[Supervisor] KeyboardInterrupt received. Shutting down.", flush=True)
    finally:
        terminate_all()
        print("[Supervisor] Exit.", flush=True)


if __name__ == "__main__":
    # SIGTERM 처리해서 컨테이너 종료 시 자식 프로세스 깔끔하게 종료
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    main()