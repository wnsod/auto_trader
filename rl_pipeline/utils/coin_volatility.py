"""
변동성 프로파일 관리 시스템
코인의 실제 변동성을 측정하고 자동 분류
"""
import sqlite3
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# 변동성 그룹 정의
VOLATILITY_GROUPS = {
    'LOW': {
        'label': 'LOW',
        'atr_range': (0.0, 0.005),
        'stop_loss': (0.015, 0.025),
        'take_profit': (0.03, 0.05),
        'position_size': (0.15, 0.25),
        'description': '메이저 코인 (BTC 등)'
    },
    'MEDIUM': {
        'label': 'MEDIUM',
        'atr_range': (0.005, 0.007),
        'stop_loss': (0.02, 0.03),
        'take_profit': (0.04, 0.08),
        'position_size': (0.10, 0.18),
        'description': '메이저 알트코인 (ETH, BNB 등)'
    },
    'HIGH': {
        'label': 'HIGH',
        'atr_range': (0.007, 0.009),
        'stop_loss': (0.03, 0.045),
        'take_profit': (0.08, 0.15),
        'position_size': (0.04, 0.10),
        'description': '알트코인 (ADA, SOL, AVAX 등)'
    },
    'VERY_HIGH': {
        'label': 'VERY_HIGH',
        'atr_range': (0.009, 1.0),
        'stop_loss': (0.04, 0.06),
        'take_profit': (0.15, 0.25),
        'position_size': (0.02, 0.06),
        'description': '고변동성 코인 (DOGE, SHIB 등)'
    }
}


def calculate_coin_volatility(db_path: str, coin: str) -> Optional[float]:
    """
    코인의 평균 ATR 계산

    Args:
        db_path: 캔들 데이터베이스 경로
        coin: 코인 심볼

    Returns:
        평균 ATR 값, 데이터 없으면 None
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT AVG(atr) as avg_atr
                FROM candles
                WHERE symbol = ? AND atr IS NOT NULL
            ''', (coin,))

            result = cursor.fetchone()
            if result and result[0]:
                return float(result[0])
            return None

    except Exception as e:
        logger.error(f"변동성 계산 실패 ({coin}): {e}")
        return None


def classify_volatility_group(avg_atr: float) -> str:
    """
    ATR 값을 기반으로 변동성 그룹 분류

    Args:
        avg_atr: 평균 ATR 값

    Returns:
        변동성 그룹 라벨 (LOW, MEDIUM, HIGH, VERY_HIGH)
    """
    for group_name, group_info in VOLATILITY_GROUPS.items():
        atr_min, atr_max = group_info['atr_range']
        if atr_min <= avg_atr < atr_max:
            return group_name

    # 범위 밖이면 가장 가까운 그룹 반환
    if avg_atr < 0.005:
        return 'LOW'
    else:
        return 'VERY_HIGH'


def get_volatility_profile(coin: Optional[str], db_path: str) -> Dict:
    """
    코인의 변동성 프로파일 조회 (자동 계산)

    Args:
        coin: 코인 심볼 (None이면 기본값)
        db_path: 데이터베이스 경로

    Returns:
        변동성 프로파일 딕셔너리
    """
    if not coin:
        # 기본값 (MEDIUM 그룹)
        return {
            'coin': None,
            'avg_atr': None,
            'volatility_group': 'MEDIUM',
            'stop_loss': (0.02, 0.035),
            'take_profit': (0.04, 0.08),
            'position_size': (0.06, 0.15)
        }

    # ATR 계산
    avg_atr = calculate_coin_volatility(db_path, coin)

    if avg_atr is None:
        # 데이터 없으면 기본값 (정상 - 아직 데이터 수집 중일 수 있음)
        logger.debug(f"ℹ️ {coin} ATR 데이터 없음, 기본값(MEDIUM) 사용")
        group = 'MEDIUM'
        avg_atr = 0.0  # 표시용 기본값
    else:
        # 그룹 분류
        group = classify_volatility_group(avg_atr)

    # 그룹 정보 가져오기
    group_info = VOLATILITY_GROUPS[group]

    return {
        'coin': coin,
        'avg_atr': avg_atr,
        'volatility_group': group,
        'stop_loss': group_info['stop_loss'],
        'take_profit': group_info['take_profit'],
        'position_size': group_info['position_size'],
        'description': group_info['description']
    }


def get_all_coin_profiles(db_path: str) -> Dict[str, Dict]:
    """
    모든 코인의 변동성 프로파일 계산

    Args:
        db_path: 데이터베이스 경로

    Returns:
        {coin: profile} 딕셔너리
    """
    profiles = {}

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 모든 코인 조회
            cursor.execute('''
                SELECT DISTINCT symbol as coin
                FROM candles
                WHERE atr IS NOT NULL
            ''')

            coins = [row[0] for row in cursor.fetchall()]

            for coin in coins:
                profile = get_volatility_profile(coin, db_path)
                profiles[coin] = profile

    except Exception as e:
        logger.error(f"전체 프로파일 계산 실패: {e}")

    return profiles


def get_coins_by_volatility_group(db_path: str, group: str) -> List[str]:
    """
    특정 변동성 그룹에 속하는 코인 리스트 반환

    Args:
        db_path: 데이터베이스 경로
        group: 변동성 그룹 (LOW, MEDIUM, HIGH, VERY_HIGH)

    Returns:
        코인 리스트
    """
    profiles = get_all_coin_profiles(db_path)
    return [coin for coin, profile in profiles.items()
            if profile['volatility_group'] == group]


def print_volatility_report(db_path: str):
    """
    변동성 프로파일 리포트 출력

    Args:
        db_path: 데이터베이스 경로
    """
    profiles = get_all_coin_profiles(db_path)

    print('='*70)
    print('📊 변동성 프로파일 리포트')
    print('='*70)

    # 그룹별 분류
    grouped = {}
    for coin, profile in profiles.items():
        group = profile['volatility_group']
        if group not in grouped:
            grouped[group] = []
        grouped[group].append((coin, profile['avg_atr']))

    # 그룹별 출력
    for group in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
        if group not in grouped:
            continue

        coins = grouped[group]
        group_info = VOLATILITY_GROUPS[group]

        print(f'\n🔹 {group} 그룹:')
        print(f'   설명: {group_info["description"]}')
        print(f'   Stop Loss: {group_info["stop_loss"][0]:.1%} ~ {group_info["stop_loss"][1]:.1%}')
        print(f'   Take Profit: {group_info["take_profit"][0]:.1%} ~ {group_info["take_profit"][1]:.1%}')
        print(f'   코인: ', end='')

        coin_strs = [f'{coin}({atr:.4f})' for coin, atr in sorted(coins, key=lambda x: x[1])]
        print(', '.join(coin_strs))

    print('\n' + '='*70)


if __name__ == '__main__':
    # 테스트
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from rl_pipeline.core.env import config

    print("변동성 프로파일 시스템 테스트\n")

    # 전체 리포트
    print_volatility_report(config.RL_DB)

    # 개별 코인 테스트
    print('\n개별 코인 프로파일 조회:')
    for coin in ['BTC', 'ADA', 'DOGE', 'UNKNOWN']:
        profile = get_volatility_profile(coin, config.RL_DB)
        print(f'\n{coin}:')
        print(f'  변동성 그룹: {profile["volatility_group"]}')
        print(f'  평균 ATR: {profile["avg_atr"]:.4f}' if profile['avg_atr'] else '  ATR: 데이터 없음')
        print(f'  Stop Loss: {profile["stop_loss"][0]:.1%} ~ {profile["stop_loss"][1]:.1%}')
