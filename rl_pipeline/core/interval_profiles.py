"""
인터벌별 프로필 설정
- 각 인터벌의 고유 목표와 역할 정의
- ETL은 공통, 라벨링/학습/보상/시그널만 차별화
- 동적 인터벌 매핑 지원 (시장별 캔들 주기 차이 대응)
"""

from typing import Dict, Any, Callable, List, Optional
import pandas as pd
import numpy as np

# ==================== 동적 인터벌 매핑 (역할 할당) ====================

# 역할 우선순위 정의 (시간 순서대로: 장기 -> 단기)
ROLE_PRIORITY = [
    "regime",   # 가장 긴 인터벌: 시장 방향성 / 장기 레짐 구분
    "swing",    # 중간-장기: 중기 추세와 파동 구조 파악
    "trend",    # 중간-단기: 추세 지속/반전 신호 확인
    "timing"    # 가장 짧은 인터벌: 실제 매수·매도 타이밍 최적화
]

# 인터벌 지속 시간(분 단위) 매핑 (정렬 및 비교용)
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60,
    "240m": 240, "4h": 240, "1d": 1440, "1w": 10080
}

def assign_roles_to_intervals(available_intervals: List[str]) -> Dict[str, str]:
    """
    주어진 인터벌 목록을 시간 순으로 정렬하고 역할을 동적으로 할당
    
    Args:
        available_intervals: 사용 가능한 인터벌 목록 (예: ['1d', '15m', '240m', '30m'])
        
    Returns:
        {인터벌: 역할타입} 딕셔너리 (예: {'1d': 'regime', '240m': 'swing', ...})
    """
    # 1. 인터벌 정규화 및 분 단위 변환
    normalized = []
    for interval in available_intervals:
        try:
            norm_int = normalize_interval(interval)
            minutes = INTERVAL_MINUTES.get(norm_int, 0)
            if minutes == 0: # 매핑에 없는 경우 숫자 파싱 시도
                 if 'm' in norm_int:
                     minutes = int(norm_int.replace('m', ''))
                 elif 'd' in norm_int:
                     minutes = int(norm_int.replace('d', '')) * 1440
            normalized.append((norm_int, minutes))
        except ValueError:
            continue
            
    # 2. 시간 순으로 정렬 (긴 순서대로: 내림차순)
    normalized.sort(key=lambda x: x[1], reverse=True)
    
    # 3. 역할 할당
    mapping = {}
    
    # 인터벌 개수에 따른 유연한 할당
    intervals_count = len(normalized)
    
    if intervals_count == 0:
        return {}
    
    # 기본: 위에서부터 순서대로 할당
    for i, (interval, _) in enumerate(normalized):
        if i < len(ROLE_PRIORITY):
            mapping[interval] = ROLE_PRIORITY[i]
        else:
            # 역할 수보다 인터벌이 많으면, 나머지는 모두 'timing' (가장 단기)으로 할당
            mapping[interval] = "timing"
            
    return mapping

# ==================== 인터벌별 프로필 템플릿 ====================

# 각 역할(label_type)별 프로필 템플릿
PROFILE_TEMPLATES = {
    "regime": {
        "name": "Macro Regime",
        "role": "Macro Regime",
        "description": "시장 방향성 / 장기 레짐 구분",
        "objectives": {
            "primary": "향후 5~20일 누적수익률 예측",
            "secondary": ["BULL/BEAR/RANGE 레짐 확률", "레짐 전환 probability"],
        },
        "labeling": {
            "target_horizon": 20,
            "label_type": "regime",
            "regime_thresholds": {"bull": 0.05, "bear": -0.05, "range": (-0.05, 0.05)},
        },
        "reward_weights": {
            "direction_accuracy": 0.4,
            "regime_accuracy": 0.4,
            "return_magnitude": 0.2,
        },
        "integrated_role": "상위 방향성 필터 - 모든 시그널의 Long/Short 우선순위 결정",
        "integration_weight": 0.35,
    },
    "swing": {
        "name": "Trend Structure",
        "role": "Trend Structure / Mid-Cycle",
        "description": "중기 추세와 파동 구조 파악",
        "objectives": {
            "primary": "향후 10~30캔들 최대 상승/하락률",
            "secondary": ["R_max/R_min 계산", "현재 파동 연장/종료 확률"],
        },
        "labeling": {
            "target_horizon": 30,
            "label_type": "swing",
            "swing_thresholds": {
                "strong_up": 0.03, "up": 0.01, "neutral": (-0.01, 0.01),
                "down": -0.01, "strong_down": -0.03
            },
        },
        "reward_weights": {
            "swing_accuracy": 0.35,
            "r_max_min_accuracy": 0.35,
            "trend_continuation": 0.3,
        },
        "integrated_role": "파동 여유 공간 계산 - 1D와 조합해 상승/하락 신뢰도 강화",
        "integration_weight": 0.30,
    },
    "trend": {
        "name": "Micro Trend",
        "role": "Micro Trend / Confirmation",
        "description": "단기 추세의 지속/반전 신호 확인",
        "objectives": {
            "primary": "8~16캔들 추세 지속 확률",
            "secondary": ["추세 반전 확률", "단기 변동성 확대/축소"],
        },
        "labeling": {
            "target_horizon": 16,
            "label_type": "trend",
            "trend_thresholds": {"continuation": 0.7, "reversal": 0.3},
        },
        "reward_weights": {
            "trend_accuracy": 0.4,
            "reversal_detection": 0.3,
            "volatility_prediction": 0.3,
        },
        "integrated_role": "상위 추세 신뢰도 보정 - 반전 가능성 체크",
        "integration_weight": 0.20,
    },
    "timing": {
        "name": "Execution",
        "role": "Execution / Timing",
        "description": "실제 매수·매도 타이밍 최적화",
        "objectives": {
            "primary": "진입 시 Expected R-multiple",
            "secondary": ["Stop-loss hit probability", "Entry quality score"],
        },
        "labeling": {
            "target_horizon": 8,
            "label_type": "timing",
            "timing_thresholds": {
                "excellent": 2.0, "good": 1.5, "neutral": 1.0, "poor": 0.5
            },
        },
        "reward_weights": {
            "entry_quality": 0.5,
            "r_multiple": 0.3,
            "stop_loss_avoidance": 0.2,
        },
        "integrated_role": "최종 실행 판단 - shadow trading/실전 매매 직접 사용",
        "integration_weight": 0.15,
    },
}

# 기본 정적 프로필 (하위 호환성 유지용)
# 동적 매핑이 사용되지 않을 경우 기본값으로 사용됨
INTERVAL_PROFILES = {
    "1d": PROFILE_TEMPLATES["regime"],
    "240m": PROFILE_TEMPLATES["swing"],
    "30m": PROFILE_TEMPLATES["trend"],
    "15m": PROFILE_TEMPLATES["timing"],
}

# ==================== 유틸리티 함수 ====================

def normalize_interval(interval: str) -> str:
    """인터벌 이름을 표준 형식으로 정규화"""
    if not interval:
        raise ValueError("인터벌이 비어있습니다")
    
    interval = interval.lower().strip()
    
    # 이미 표준 키에 있는 경우
    if interval in INTERVAL_PROFILES:
        return interval
        
    # 4h -> 240m 변환
    interval_mapping = {
        '4h': '240m', '4hour': '240m', '4hours': '240m',
    }
    if interval in interval_mapping:
        return interval_mapping[interval]
    
    # 그 외의 경우 그대로 반환 (동적 매핑을 위해 유연하게 허용)
    return interval


# ==================== 라벨링 함수 ====================

def generate_regime_labels(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """1D용 레짐 라벨 생성"""
    returns = df['close'].pct_change(horizon).shift(-horizon)

    labels = pd.Series('range', index=df.index)
    labels[returns > 0.05] = 'bull'
    labels[returns < -0.05] = 'bear'
    
    # NaN 값 처리 (마지막 N개 행)
    labels[returns.isna()] = 'unknown'

    return labels


def generate_swing_labels(df: pd.DataFrame, horizon: int = 30) -> pd.Series:
    """4H용 스윙 라벨 생성"""
    # R_max: 향후 최대 상승률
    r_max = df['high'].rolling(horizon).max().shift(-horizon) / df['close'] - 1
    # R_min: 향후 최대 하락률
    r_min = df['low'].rolling(horizon).min().shift(-horizon) / df['close'] - 1

    # 스윙 라벨 결정
    labels = pd.Series('neutral', index=df.index)
    labels[r_max > 0.03] = 'strong_up'
    labels[(r_max > 0.01) & (r_max <= 0.03)] = 'up'
    labels[r_min < -0.03] = 'strong_down'
    labels[(r_min < -0.01) & (r_min >= -0.03)] = 'down'
    
    # NaN 값 처리 (마지막 N개 행)
    nan_mask = r_max.isna() | r_min.isna()
    labels[nan_mask] = 'unknown'

    return labels


def generate_trend_labels(df: pd.DataFrame, horizon: int = 16) -> pd.Series:
    """30m용 추세 지속/반전 라벨 생성"""
    # 현재 추세 방향
    ma_short = df['close'].rolling(8).mean()
    ma_long = df['close'].rolling(32).mean()
    current_trend = (ma_short > ma_long).astype(int)

    # 미래 추세 방향
    future_trend = current_trend.shift(-horizon)

    # 추세 지속 여부
    labels = pd.Series('neutral', index=df.index)
    labels[current_trend == future_trend] = 'continuation'
    labels[current_trend != future_trend] = 'reversal'
    
    # NaN 값 처리 (마지막 N개 행)
    nan_mask = current_trend.isna() | future_trend.isna()
    labels[nan_mask] = 'unknown'

    return labels


def generate_timing_labels(df: pd.DataFrame, horizon: int = 8) -> pd.Series:
    """15m용 진입 타이밍 라벨 생성"""
    # R-multiple 계산 (진입 후 수익 / 리스크)
    entry_price = df['close']
    stop_loss = df['low'].rolling(20).min()  # 최근 20캔들 최저가

    # 향후 최대 수익
    max_profit = df['high'].rolling(horizon).max().shift(-horizon) - entry_price
    risk = entry_price - stop_loss

    r_multiple = max_profit / risk.where(risk > 0, np.nan)

    # 타이밍 라벨
    labels = pd.Series('neutral', index=df.index)
    labels[r_multiple > 2.0] = 'excellent'
    labels[(r_multiple > 1.5) & (r_multiple <= 2.0)] = 'good'
    labels[r_multiple < 0.5] = 'poor'
    
    # NaN 값 처리 (마지막 N개 행 또는 risk가 0인 경우)
    nan_mask = r_multiple.isna()
    labels[nan_mask] = 'unknown'

    return labels


# ==================== 라벨 생성 메인 함수 ====================

def generate_labels(df: pd.DataFrame, interval: str, role: str = None) -> pd.DataFrame:
    """인터벌별 맞춤 라벨 생성
    
    Args:
        df: 캔들 데이터
        interval: 인터벌
        role: (선택) 강제 지정할 역할 ('regime', 'swing', 'trend', 'timing'). 
              None이면 정적 매핑 사용.

    Returns:
        라벨이 추가된 DataFrame
    """
    # 1. 인터벌 정규화
    interval = normalize_interval(interval)
    
    # 2. DataFrame 복사
    df = df.copy()
    
    # 3. 프로필 및 역할 결정
    if role and role in PROFILE_TEMPLATES:
        profile = PROFILE_TEMPLATES[role]
    else:
        profile = INTERVAL_PROFILES.get(interval, PROFILE_TEMPLATES['timing']) # 기본값 timing
        
    label_type = profile["labeling"]["label_type"]
    horizon = profile["labeling"]["target_horizon"]

    # 4. 라벨 생성
    try:
        if label_type == "regime":
            df['label'] = generate_regime_labels(df, horizon)
        elif label_type == "swing":
            df['label'] = generate_swing_labels(df, horizon)
        elif label_type == "trend":
            df['label'] = generate_trend_labels(df, horizon)
        elif label_type == "timing":
            df['label'] = generate_timing_labels(df, horizon)
        else:
            raise ValueError(f"Unknown label type: {label_type}")
    except (KeyError, ValueError) as e:
        # 실패 시 기본값이나 에러 처리
        raise ValueError(f"라벨 생성 실패 ({interval}, {label_type}): {e}") from e

    # 5. 메타 정보 추가
    df['interval'] = interval
    df['label_type'] = label_type
    df['target_horizon'] = horizon

    return df


# ==================== 보상 함수 ====================

def calculate_regime_reward(prediction: Dict, actual: Dict) -> float:
    """1D 레짐 예측 보상"""
    weights = PROFILE_TEMPLATES["regime"]["reward_weights"]

    reward = 0.0
    if prediction.get('direction') == actual.get('direction'):
        reward += weights['direction_accuracy']
    if prediction.get('regime') == actual.get('regime'):
        reward += weights['regime_accuracy']
    
    pred_return = prediction.get('return', 0)
    actual_return = actual.get('return', 0)
    return_error = abs(pred_return - actual_return)
    reward += weights['return_magnitude'] * max(0, 1 - return_error)

    return reward


def calculate_swing_reward(prediction: Dict, actual: Dict) -> float:
    """4H 스윙 예측 보상"""
    weights = PROFILE_TEMPLATES["swing"]["reward_weights"]

    reward = 0.0
    if prediction.get('swing') == actual.get('swing'):
        reward += weights['swing_accuracy']
    
    r_max_error = abs(prediction.get('r_max', 0) - actual.get('r_max', 0))
    r_min_error = abs(prediction.get('r_min', 0) - actual.get('r_min', 0))
    reward += weights['r_max_min_accuracy'] * max(0, 1 - (r_max_error + r_min_error) / 2)

    if prediction.get('trend_continues') == actual.get('trend_continues'):
        reward += weights['trend_continuation']

    return reward


def calculate_trend_reward(prediction: Dict, actual: Dict) -> float:
    """30m 추세 확인 보상"""
    weights = PROFILE_TEMPLATES["trend"]["reward_weights"]

    reward = 0.0
    if prediction.get('trend') == actual.get('trend'):
        reward += weights['trend_accuracy']
    if prediction.get('reversal') == actual.get('reversal'):
        reward += weights['reversal_detection']
    
    vol_error = abs(prediction.get('volatility', 0) - actual.get('volatility', 0))
    reward += weights['volatility_prediction'] * max(0, 1 - vol_error)

    return reward


def calculate_timing_reward(prediction: Dict, actual: Dict) -> float:
    """15m 타이밍 보상"""
    weights = PROFILE_TEMPLATES["timing"]["reward_weights"]

    reward = 0.0
    if prediction.get('entry_quality') == actual.get('entry_quality'):
        reward += weights['entry_quality']
    
    r_error = abs(prediction.get('r_multiple', 0) - actual.get('r_multiple', 0))
    reward += weights['r_multiple'] * max(0, 1 - r_error / 2)

    if not actual.get('stop_hit', False):
        reward += weights['stop_loss_avoidance']

    return reward


# ==================== 보상 계산 메인 함수 ====================

def calculate_reward(interval: str, prediction: Dict, actual: Dict, role: str = None) -> float:
    """인터벌별 맞춤 보상 계산"""
    interval = normalize_interval(interval)
    
    if not isinstance(prediction, dict) or not isinstance(actual, dict):
        raise ValueError("prediction과 actual은 dict 타입이어야 합니다")

    # 역할 결정
    if role and role in PROFILE_TEMPLATES:
        label_type = PROFILE_TEMPLATES[role]["labeling"]["label_type"]
    else:
        profile = INTERVAL_PROFILES.get(interval, PROFILE_TEMPLATES['timing'])
        label_type = profile["labeling"]["label_type"]

    try:
        if label_type == "regime":
            return calculate_regime_reward(prediction, actual)
        elif label_type == "swing":
            return calculate_swing_reward(prediction, actual)
        elif label_type == "trend":
            return calculate_trend_reward(prediction, actual)
        elif label_type == "timing":
            return calculate_timing_reward(prediction, actual)
        else:
            raise ValueError(f"Unknown label type: {label_type}")
    except (KeyError, TypeError) as e:
        raise ValueError(f"보상 계산 실패 ({interval}, {label_type}): {e}") from e


# ==================== 통합 분석 및 기타 ====================

def get_integration_weights(active_intervals: List[str] = None) -> Dict[str, float]:
    """통합 분석용 인터벌별 가중치 반환 (동적 매핑 반영)"""
    if not active_intervals:
        return {k: v["integration_weight"] for k, v in INTERVAL_PROFILES.items()}
        
    mapping = assign_roles_to_intervals(active_intervals)
    weights = {}
    for interval, role in mapping.items():
        weights[interval] = PROFILE_TEMPLATES[role]["integration_weight"]
    return weights


def get_interval_role(interval: str, active_intervals: List[str] = None) -> str:
    """인터벌의 통합 분석 역할 반환 (동적 매핑 반영)"""
    interval = normalize_interval(interval)
    
    if active_intervals:
         mapping = assign_roles_to_intervals(active_intervals)
         role_key = mapping.get(interval)
         if role_key:
             return PROFILE_TEMPLATES[role_key]["integrated_role"]
             
    # Fallback to static profile
    if interval in INTERVAL_PROFILES:
        return INTERVAL_PROFILES[interval]["integrated_role"]
        
    return "Unknown role"


def get_all_intervals() -> List[str]:
    return list(INTERVAL_PROFILES.keys())


def get_interval_profile(interval: str, active_intervals: List[str] = None) -> Dict[str, Any]:
    """특정 인터벌의 전체 프로필 반환 (동적 매핑 반영)"""
    interval = normalize_interval(interval)
    
    if active_intervals:
        mapping = assign_roles_to_intervals(active_intervals)
        role_key = mapping.get(interval)
        if role_key:
            return PROFILE_TEMPLATES[role_key]
            
    return INTERVAL_PROFILES.get(interval, {})


def validate_interval(interval: str) -> bool:
    try:
        normalize_interval(interval)
        return True
    except ValueError:
        return False


__all__ = [
    'INTERVAL_PROFILES', 'PROFILE_TEMPLATES', 'ROLE_PRIORITY',
    'assign_roles_to_intervals', 'normalize_interval',
    'generate_labels', 'calculate_reward',
    'get_integration_weights', 'get_interval_role',
    'get_all_intervals', 'get_interval_profile', 'validate_interval',
]