"""
적응형 롤업 시스템
코인별/인터벌별로 최적 롤업 기간을 계산하는 시스템
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from rl_pipeline.db.connection_pool import get_optimized_db_connection

logger = logging.getLogger(__name__)

# 인터벌별 최소 에피소드 기준
INTERVAL_MIN_EPISODES = {
    '15m': 300,
    '30m': 200,
    '240m': 100,   # 장기: 적은 데이터로도 신뢰도 확보
    '1d': 50
}

# 롤업 프로파일 기본값 (심볼 하드코딩 제거)
COIN_ROLLUP_PROFILES = {
    'default': {'standard_period': 20, 'adjustment_factor': 1.0}
}

# 인터벌별 기본 기간
INTERVAL_BASE_PERIODS = {
    '15m': 20,
    '30m': 25,
    '240m': 30,  # 장기 인터벌: 긴 기간
    '1d': 30
}


def calculate_adaptive_rollup_days(
    coin: str,
    interval: str,
    db_connection=None
) -> int:
    """
    코인별 적응형 롤업 기간 계산
    
    결정 요인:
    1. 거래 빈도 (trades_count 기준)
    2. 데이터 품질 (에피소드 수)
    3. 변동성 (시장 특성)
    4. 최소 통계적 신뢰도 확보
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        db_connection: DB 연결 (None이면 새로 생성)
    
    Returns:
        최적 롤업 기간 (일 단위, 7~90일 범위)
    """
    try:
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _calculate_adaptive_rollup_days_impl(coin, interval, conn)
        else:
            return _calculate_adaptive_rollup_days_impl(coin, interval, db_connection)
            
    except Exception as e:
        logger.warning(f"⚠️ 적응형 롤업 기간 계산 실패, 기본값 사용: {e}")
        # 폴백: 인터벌별 기본 기간
        return INTERVAL_BASE_PERIODS.get(interval, 20)


def _calculate_adaptive_rollup_days_impl(
    coin: str,
    interval: str,
    conn
) -> int:
    """적응형 롤업 기간 계산 구현"""
    
    # 1. 최소 에피소드 기준 (인터벌별)
    min_episodes = INTERVAL_MIN_EPISODES.get(interval, 200)
    
    # 2. 최근 에피소드 수 조회 (기본 30일 기준으로 카운트)
    base_days = 30
    recent_episodes = count_recent_episodes(coin, interval, conn, base_days)
    
    logger.debug(f"📊 {coin}-{interval}: 최근 {base_days}일 에피소드 수 = {recent_episodes} (최소 기준: {min_episodes})")
    
    # 3. 코인별 특성 반영
    coin_profile = get_coin_rollup_profile(coin, interval)
    standard_period = coin_profile['standard_period']
    
    # 4. 적응형 기간 계산
    if recent_episodes >= min_episodes:
        # 충분한 데이터: 코인별 최적 기간
        optimal_days = standard_period
        logger.debug(f"✅ {coin}-{interval}: 충분한 데이터 ({recent_episodes} >= {min_episodes}), 표준 기간 사용: {optimal_days}일")
    elif recent_episodes >= min_episodes * 0.7:
        # 약간 부족: 약간 확장
        optimal_days = int(standard_period * 1.3)
        logger.debug(f"⚠️ {coin}-{interval}: 약간 부족 ({recent_episodes} < {min_episodes}), 확장: {optimal_days}일")
    elif recent_episodes >= min_episodes * 0.5:
        # 부족: 확장
        optimal_days = int(standard_period * 2.0)
        logger.debug(f"⚠️ {coin}-{interval}: 데이터 부족 ({recent_episodes} < {min_episodes * 0.7}), 확장: {optimal_days}일")
    else:
        # 매우 부족: 최대 확장 (상한선 90일)
        optimal_days = min(int(standard_period * 3.0), 90)
        logger.debug(f"⚠️ {coin}-{interval}: 매우 부족 ({recent_episodes} < {min_episodes * 0.5}), 최대 확장: {optimal_days}일")
    
    # 5. 코인별 조정
    optimal_days = int(optimal_days * coin_profile['adjustment_factor'])
    
    # 범위 제한 (최소 7일, 최대 90일)
    optimal_days = max(7, min(90, optimal_days))
    
    logger.info(f"🎯 {coin}-{interval} 적응형 롤업 기간: {optimal_days}일 (에피소드 수: {recent_episodes})")
    
    return optimal_days


def count_recent_episodes(
    coin: str,
    interval: str,
    conn,
    base_days: int = 30
) -> int:
    """
    최근 N일간의 에피소드 수 조회
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        conn: DB 연결
        base_days: 기준 일수
    
    Returns:
        에피소드 수
    """
    try:
        cursor = conn.cursor()
        
        # rl_episode_summary 테이블에서 조회
        cutoff_ts = int((datetime.now() - timedelta(days=base_days)).timestamp())
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM rl_episode_summary
            WHERE coin = ? AND interval = ? AND ts_exit >= ?
        """, (coin, interval, cutoff_ts))
        
        result = cursor.fetchone()
        return result[0] if result else 0
        
    except Exception as e:
        logger.debug(f"⚠️ 에피소드 수 조회 실패: {e}")
        return 0


def get_coin_rollup_profile(coin: str, interval: str) -> Dict[str, Any]:
    """
    코인별 롤업 프로파일 조회
    
    기존 시스템의 코인 분류 활용:
    - major_coin (BTC, ETH): 안정적, 표준 기간
    - high_performance (SOL): 빠른 변화, 짧은 기간
    - exchange_coin (BNB): 중간
    - academic_coin (ADA): 장기 전략, 긴 기간
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
    
    Returns:
        프로파일 딕셔너리
    """
    # 인터벌별 기본 기간
    base_period = INTERVAL_BASE_PERIODS.get(interval, 20)
    
    # 코인별 프로파일 조회(하드코딩 제거 → 기본값 사용, 필요 시 환경/DB 분류로 확장)
    coin_profile = COIN_ROLLUP_PROFILES['default']
    
    # 인터벌과 코인 프로파일 결합
    standard_period = base_period  # 인터벌 기본값 사용
    adjustment_factor = coin_profile['adjustment_factor']
    
    return {
        'standard_period': standard_period,
        'adjustment_factor': adjustment_factor,
        'min_episodes_threshold': INTERVAL_MIN_EPISODES.get(interval, 200)
    }


def create_adaptive_rollup_view(
    coin: str,
    interval: str,
    db_connection=None
) -> int:
    """
    코인별 적응형 롤업 뷰 생성
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        db_connection: DB 연결
    
    Returns:
        실제 사용된 롤업 기간 (일)
    """
    try:
        # 적응형 기간 계산
        optimal_days = calculate_adaptive_rollup_days(coin, interval, db_connection)
        
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _create_adaptive_rollup_view_impl(coin, interval, optimal_days, conn)
        else:
            return _create_adaptive_rollup_view_impl(coin, interval, optimal_days, db_connection)
            
    except Exception as e:
        logger.error(f"❌ 적응형 롤업 뷰 생성 실패: {e}")
        return 30  # 폴백


def _create_adaptive_rollup_view_impl(
    coin: str,
    interval: str,
    optimal_days: int,
    conn
) -> int:
    """적응형 롤업 뷰 생성 구현"""
    try:
        cursor = conn.cursor()
        
        # 뷰 이름 (특수문자 제거)
        view_name = f"v_rl_episode_summary_{coin}_{interval}_adaptive".replace('-', '_')
        
        # 기존 뷰 삭제 (있으면)
        cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
        
        # 새 뷰 생성
        query = f"""
        CREATE VIEW {view_name} AS
        SELECT *
        FROM rl_episode_summary
        WHERE coin = '{coin}' 
          AND interval = '{interval}'
          AND ts_exit >= strftime('%s','now','-{optimal_days} days')
        """
        
        cursor.execute(query)
        conn.commit()
        
        logger.info(f"✅ 적응형 롤업 뷰 생성: {view_name} (기간: {optimal_days}일)")
        
        return optimal_days
        
    except Exception as e:
        logger.error(f"❌ 적응형 롤업 뷰 생성 실패: {e}")
        return optimal_days


if __name__ == "__main__":
    # 테스트
    print("적응형 롤업 시스템 테스트:")

    # 테스트 1: 코인 프로파일 조회
    import os
    import sys

    # 환경변수에서 테스트 코인 지정
    test_coin = os.getenv('TEST_COIN')
    if not test_coin:
        print("❌ 테스트 코인을 지정하세요: export TEST_COIN=BTC")
        print("📝 예시: TEST_COIN=BTC python adaptive_rollup.py")
        sys.exit(1)

    profile = get_coin_rollup_profile(test_coin, "15m")
    print(f"{test_coin}-15m 프로파일: {profile}")

    # 테스트 2: 적응형 기간 계산 (DB 없이 기본값만)
    # optimal_days = calculate_adaptive_rollup_days(test_coin, "15m")
    # print(f"{test_coin}-15m 최적 기간: {optimal_days}일")

