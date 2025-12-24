"""
인터벌별 프로파일 설정
예측형 강화학습 시스템의 인터벌별 파라미터 정의
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 환경변수 파일 로드
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rl_pipeline_config.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

# 인터벌별 프로파일 (하드코딩, .env로 확장 가능)
INTERVAL_PROFILE = {
    "5m": {  # 🔥 5분봉 추가 (15m보다 더 짧고 민감하게)
        "horizon_k": 12,  # 짧은 호흡이므로 1시간(12개) 정도
        "sigma_min": 0.01,
        "risk_gamma": 0.30, # 리스크 패널티 약간 완화
        "alpha_orc": 0.60,  # 기회 비용 비중 높임
        "target_min": 0.003,     # 0.3% (스캘핑 수준)
        "target_max": 0.015,     # 1.5%
        # 🆕 3단계 비율
        "tier_ratios": {
            "tier1": 0.003 / 0.009,  # 0.333
            "tier2": 0.012 / 0.009,  # 1.333
            "tier3": 0.030 / 0.009   # 3.333
        }
    },
    "15m": {
        "horizon_k": 12,  # 🔥 8 → 12로 증가 (더 긴 타임라인으로 가격 변동 기회 확대)
        "sigma_min": 0.01,
        "risk_gamma": 0.35,
        "alpha_orc": 0.55,
        "target_min": 0.005,     # 🔥 0.75% → 0.5%로 감소 (더 작은 목표로 달성 가능성 증가)
        "target_max": 0.020,     # 🔥 2.25% → 2.0%로 감소 (더 현실적인 목표)
        # 🆕 3단계 비율 (base_target 기준, 동적 조정 가능)
        "tier_ratios": {
            "tier1": 0.005 / 0.015,  # 0.5% / 평균(base_target≈1.5%) = 0.333
            "tier2": 0.020 / 0.015,  # 2.0% / 평균(base_target≈1.5%) = 1.333
            "tier3": 0.050 / 0.015   # 5.0% / 평균(base_target≈1.5%) = 3.333
        }
    },
    "30m": {
        "horizon_k": 10,  # 🔥 6 → 10으로 증가
        "sigma_min": 0.01,
        "risk_gamma": 0.40,
        "alpha_orc": 0.50,
        "target_min": 0.006,    # 🔥 0.8% → 0.6%로 감소
        "target_max": 0.018,    # 🔥 2.0% → 1.8%로 감소
        # 🆕 30m: 15m의 약 1.5배 기준으로 비율 설정
        "tier_ratios": {
            "tier1": 0.008 / 0.014,  # 0.8% / 평균(base_target≈1.4%) = 0.571
            "tier2": 0.030 / 0.014,  # 3.0% / 평균(base_target≈1.4%) = 2.143
            "tier3": 0.075 / 0.014   # 7.5% / 평균(base_target≈1.4%) = 5.357
        }
    },
    "240m": {
        "horizon_k": 8,  # 🔥 4 → 8로 증가
        "sigma_min": 0.01,
        "risk_gamma": 0.45,
        "alpha_orc": 0.50,
        "target_min": 0.012,    # 🔥 1.5% → 1.2%로 감소
        "target_max": 0.045,    # 🔥 5.0% → 4.5%로 감소
        # 🆕 240m: 15m의 약 3배 기준으로 비율 설정
        "tier_ratios": {
            "tier1": 0.015 / 0.0325,  # 1.5% / 평균(base_target≈3.25%) = 0.462
            "tier2": 0.060 / 0.0325,  # 6.0% / 평균(base_target≈3.25%) = 1.846
            "tier3": 0.150 / 0.0325   # 15.0% / 평균(base_target≈3.25%) = 4.615
        }
    },
    "1d": {
        "horizon_k": 6,  # 🔥 2 → 6으로 증가
        "sigma_min": 0.01,
        "risk_gamma": 0.50,
        "alpha_orc": 0.50,
        "target_min": 0.015,    # 🔥 2.0% → 1.5%로 감소
        "target_max": 0.070,    # 🔥 8.0% → 7.0%로 감소
        # 🆕 1d: 15m의 약 5배 기준으로 비율 설정
        "tier_ratios": {
            "tier1": 0.025 / 0.050,  # 2.5% / 평균(base_target≈5.0%) = 0.5
            "tier2": 0.100 / 0.050,  # 10.0% / 평균(base_target≈5.0%) = 2.0
            "tier3": 0.250 / 0.050   # 25.0% / 평균(base_target≈5.0%) = 5.0
        }
    }
}

# 환경변수로 오버라이드 가능 (선택적)
def _load_env_overrides():
    """환경변수에서 프로파일 오버라이드 (선택적)"""
    overrides = {}
    for interval in INTERVAL_PROFILE.keys():
        # 예: INTERVAL_5m_HORIZON_K=12 형태로 오버라이드 가능
        horizon_key = f"INTERVAL_{interval.upper().replace('M', '_M')}_HORIZON_K"
        if horizon_key in os.environ:
            try:
                INTERVAL_PROFILE[interval]["horizon_k"] = int(os.getenv(horizon_key))
                logger.info(f"✅ {interval} horizon_k 오버라이드: {os.getenv(horizon_key)}")
            except (ValueError, TypeError):
                logger.warning(f"⚠️ {horizon_key} 값이 유효하지 않음")

# 환경변수 로드 (선택적)
_load_env_overrides()


def get_interval_profile(interval: str) -> Dict[str, Any]:
    """
    인터벌 프로파일 조회
    
    Args:
        interval: 인터벌 문자열 ('15m', '30m', '240m', '1d')
    
    Returns:
        해당 인터벌의 프로파일 딕셔너리
    
    Raises:
        ValueError: 지원하지 않는 인터벌인 경우
    """
    if interval not in INTERVAL_PROFILE:
        supported = ', '.join(INTERVAL_PROFILE.keys())
        raise ValueError(
            f"지원하지 않는 인터벌: {interval}. "
            f"지원 인터벌: {supported}"
        )
    
    return INTERVAL_PROFILE[interval].copy()


def get_horizon_k(interval: str) -> int:
    """인터벌별 판정 기한(캔들 수) 조회"""
    profile = get_interval_profile(interval)
    return profile["horizon_k"]


def get_sigma_min(interval: str) -> float:
    """인터벌별 근접도 σ 하한 조회"""
    profile = get_interval_profile(interval)
    return profile["sigma_min"]


def get_risk_gamma(interval: str) -> float:
    """인터벌별 DD 패널티 강도 조회"""
    profile = get_interval_profile(interval)
    return profile["risk_gamma"]


def get_target_range(interval: str) -> tuple[float, float]:
    """
    인터벌별 목표 변동률 범위 조회
    
    Returns:
        (target_min, target_max) 튜플
    """
    profile = get_interval_profile(interval)
    return (profile["target_min"], profile["target_max"])


def validate_target_move_pct(interval: str, target_move_pct: float) -> bool:
    """
    목표 변동률이 해당 인터벌의 범위 내인지 검증
    
    Args:
        interval: 인터벌
        target_move_pct: 검증할 목표 변동률
    
    Returns:
        유효하면 True, 아니면 False
    """
    target_min, target_max = get_target_range(interval)
    return target_min <= target_move_pct <= target_max


if __name__ == "__main__":
    # 테스트
    print("인터벌 프로파일 테스트:")
    for interval in ["15m", "30m", "240m", "1d"]:
        profile = get_interval_profile(interval)
        print(f"{interval}: horizon_k={profile['horizon_k']}, "
              f"target_range=[{profile['target_min']:.3f}, {profile['target_max']:.3f}]")

