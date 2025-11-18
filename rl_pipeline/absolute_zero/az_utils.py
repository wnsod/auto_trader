"""
Absolute Zero 시스템 - 유틸리티 모듈
공통 유틸리티 함수들과 헬퍼 함수들
"""

import logging
import sqlite3
import time
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def sort_intervals(interval_list: List[str]) -> List[str]:
    """
    인터벌을 시간 순서로 정렬: 실제 분 단위로 변환하여 정렬

    Args:
        interval_list: 정렬할 인터벌 리스트

    Returns:
        정렬된 인터벌 리스트
    """
    def get_order_in_minutes(iv):
        """인터벌을 분 단위로 변환하여 정렬 키 생성"""
        iv_lower = iv.lower().strip()

        # 분 단위로 변환
        try:
            if iv_lower.endswith('m'):
                # 분 단위 (예: 15m, 30m, 240m)
                minutes = int(iv_lower[:-1])
                return minutes
            elif iv_lower.endswith('h'):
                # 시간 단위 (예: 1h, 4h)
                hours = int(iv_lower[:-1])
                return hours * 60  # 시간을 분으로 변환
            elif iv_lower.endswith('d'):
                # 일 단위 (예: 1d)
                days = int(iv_lower[:-1])
                return days * 1440  # 일을 분으로 변환
            else:
                # 알 수 없는 형식은 마지막으로
                return 999999
        except (ValueError, AttributeError):
            # 파싱 실패 시 마지막으로
            return 999999

    # 분 단위로 정렬 (안정적 정렬: 같은 값이면 원래 순서 유지)
    return sorted(interval_list, key=lambda x: (get_order_in_minutes(x), x))

def execute_wal_checkpoint(db_path: str, max_retries: int = 3) -> bool:
    """
    SQLite WAL 체크포인트 실행

    Args:
        db_path: 데이터베이스 경로
        max_retries: 최대 재시도 횟수

    Returns:
        성공 여부
    """
    import traceback

    wal_checkpoint_success = False

    for retry in range(max_retries):
        try:
            logger.info(f"🔧 WAL 체크포인트 시도 {retry + 1}/{max_retries}")

            # Connection Pool의 모든 연결 종료 (먼저 실행)
            try:
                from rl_pipeline.db.connection_pool import close_all_connections
                close_all_connections()
                logger.info(f"✅ Connection Pool 종료 완료")
                time.sleep(0.2)  # 종료 대기
            except Exception as pool_error:
                logger.warning(f"⚠️ Connection Pool 종료 실패: {pool_error}")
                logger.debug(f"Connection Pool 종료 실패 상세:\n{traceback.format_exc()}")

            # 짧은 타임아웃으로 연결 시도
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()

            # WAL 체크포인트 실행 (PASSIVE 먼저)
            result_passive = cursor.execute('PRAGMA wal_checkpoint(PASSIVE)').fetchone()
            logger.debug(f"🔧 WAL 체크포인트 PASSIVE 결과: {result_passive}")

            # TRUNCATE 체크포인트 실행
            result = cursor.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            logger.debug(f"🔧 WAL 체크포인트 TRUNCATE 결과: {result}")

            # 결과 확인: (busy, log, checkpointed)
            if result and result[0] == 0:  # busy=0이면 성공
                logger.debug(f"✅ WAL 체크포인트 성공: {result[2]}개 페이지 체크포인트됨")
            elif result and result[0] == 1:  # busy=1이면 다른 연결이 사용 중
                logger.warning(f"⚠️ WAL 체크포인트 busy: 다른 연결이 사용 중 (무시하고 계속)")

            conn.commit()
            conn.close()

            # 추가 대기 (WAL 파일이 실제로 줄어들도록)
            time.sleep(0.3)

            wal_checkpoint_success = True
            logger.info(f"✅ WAL 체크포인트 완료")
            break  # 성공 시 재시도 중단

        except sqlite3.OperationalError as db_error:
            error_msg = str(db_error)
            logger.warning(f"⚠️ WAL 체크포인트 실패 (시도 {retry + 1}/{max_retries}): {error_msg}")
            if retry < max_retries - 1:  # 마지막 재시도가 아니면
                wait_time = (retry + 1) * 0.5
                logger.info(f"⏳ {wait_time:.1f}초 후 재시도...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ WAL 체크포인트 최종 실패: {error_msg}")
                logger.debug(f"WAL 체크포인트 실패 상세:\n{traceback.format_exc()}")
        except Exception as wal_error:
            error_msg = str(wal_error)
            logger.error(f"❌ WAL 체크포인트 예외 발생 (시도 {retry + 1}/{max_retries}): {error_msg}")
            logger.debug(f"WAL 체크포인트 예외 상세:\n{traceback.format_exc()}")
            if retry < max_retries - 1:
                time.sleep(0.5)
            else:
                logger.error(f"❌ WAL 체크포인트 최종 실패: {error_msg}")

    return wal_checkpoint_success

def format_time_duration(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """
    시간 차이를 사람이 읽기 쉬운 형식으로 변환

    Args:
        start_time: 시작 시간
        end_time: 종료 시간 (None이면 현재 시간 사용)

    Returns:
        포맷된 시간 문자열
    """
    if end_time is None:
        end_time = datetime.now()

    duration = end_time - start_time
    seconds = int(duration.total_seconds())

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"

def check_data_sufficiency(candle_data: dict, coin: str) -> tuple[bool, list]:
    """
    캔들 데이터의 충분성 검사

    Args:
        candle_data: 캔들 데이터 딕셔너리
        coin: 코인 심볼

    Returns:
        (충분한지 여부, 부족한 인터벌 리스트)
    """
    from .az_config import MIN_CANDLES_PER_INTERVAL

    insufficient_intervals = []

    for (c, interval), df in candle_data.items():
        min_required = MIN_CANDLES_PER_INTERVAL.get(interval, 100)
        if len(df) < min_required:
            insufficient_intervals.append(f"{interval}({len(df)}개)")

    if insufficient_intervals:
        logger.warning(f"⚠️ {coin}: 신생 코인 감지 - 일부 인터벌 데이터 부족: {', '.join(insufficient_intervals)}")
        logger.info(f"📊 {coin}: 가용 데이터로 진행합니다")

    # 전체 캔들 수 체크
    total_candles = sum(len(df) for df in candle_data.values())
    if total_candles == 0:
        logger.error(f"❌ {coin}: 사용 가능한 캔들 데이터가 없습니다")
        return False, insufficient_intervals

    return True, insufficient_intervals

def create_run_metadata(coin: str, intervals: List[str]) -> dict:
    """
    실행 메타데이터 생성

    Args:
        coin: 코인 심볼
        intervals: 인터벌 리스트

    Returns:
        메타데이터 딕셔너리
    """
    run_id = f"abs_zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_span = datetime.now().strftime('%Y-%m-%d')

    # 여러 interval 사용 시 첫 번째 interval 사용 (또는 ','로 구분된 문자열)
    interval_str = intervals[0] if intervals else "15m"
    if len(intervals) > 1:
        interval_str = ','.join(intervals)  # 여러 interval을 ','로 구분

    return {
        'run_id': run_id,
        'dataset_span': dataset_span,
        'regime': 'mixed',  # 실제로는 시장 분석 결과에 따라 결정
        'coin': coin,
        'interval_str': interval_str
    }

def log_system_info():
    """시스템 정보 로깅"""
    try:
        import platform
        import psutil

        logger.info("=" * 60)
        logger.info("🖥️ 시스템 정보:")
        logger.info(f"  - Python: {platform.python_version()}")
        logger.info(f"  - Platform: {platform.platform()}")
        logger.info(f"  - CPU: {psutil.cpu_count()} cores")
        logger.info(f"  - Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        logger.info(f"  - Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        logger.info("=" * 60)
    except Exception as e:
        logger.debug(f"시스템 정보 로깅 실패: {e}")

def validate_environment() -> bool:
    """
    실행 환경 검증

    Returns:
        환경이 유효한지 여부
    """
    try:
        # 필수 모듈 확인
        required_modules = [
            'numpy',
            'pandas',
            'sqlite3',
            'jax',
            'tensorflow'
        ]

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                logger.error(f"❌ 필수 모듈 누락: {module}")
                return False

        # GPU 사용 가능 여부 확인 (선택사항)
        try:
            import jax
            devices = jax.devices()
            if len(devices) > 0:
                logger.info(f"🎮 JAX 디바이스: {devices}")
        except Exception as e:
            logger.debug(f"JAX 디바이스 확인 실패 (CPU 사용): {e}")

        return True

    except Exception as e:
        logger.error(f"환경 검증 실패: {e}")
        return False