"""
이 파일은 전략지표(기술/파동/패턴)를 DB의 OHLCV+기초지표만으로 직접 계산하여
파동, 패턴, 프랙탈, 통합분석 컬럼(wave_step, structure_score, pattern_class, volatility_level, risk_level, integrated_direction, integrated_strength 등)
모두를 갱신합니다.

실행 시 DB의 모든 코인/인터벌에 대해 zigzag/wave/pattern/프랙탈/통합분석을 자동으로 처리합니다.
"""
import sqlite3
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import traceback

# pandas FutureWarning 해결을 위한 설정
pd.set_option('future.no_silent_downcasting', True)

# 데이터베이스 경로 설정
DB_PATH = "/workspace/data_storage/realtime_candles.db"

# 🚀 레짐 계산 상수 정의
REGIME_STAGES = {
    1: "extreme_bearish",    # RSI < 20, 급격한 하락
    2: "bearish",           # RSI 20-40, 하락 추세
    3: "sideways_bearish",  # RSI 40-50, 약한 하락
    4: "neutral",           # RSI 45-55, 횡보
    5: "sideways_bullish",  # RSI 50-60, 약한 상승
    6: "bullish",           # RSI 60-80, 상승 추세
    7: "extreme_bullish"    # RSI > 80, 급격한 상승
}

REGIME_LABELS = {v: k for k, v in REGIME_STAGES.items()}

# 인터벌별 레짐 계산 기준
REGIME_CRITERIA = {
    '15m': {
        'rsi_weight': 0.4,      # 균형
        'macd_weight': 0.3,
        'volume_weight': 0.3,
        'volatility_threshold': 0.025,
        'lookback_period': 15
    },
    '30m': {
        'rsi_weight': 0.5,      # RSI 신뢰도 높음
        'macd_weight': 0.3,
        'volume_weight': 0.2,
        'volatility_threshold': 0.02,
        'lookback_period': 20
    },
    '240m': {
        'rsi_weight': 0.6,      # 가장 신뢰도 높음
        'macd_weight': 0.2,
        'volume_weight': 0.2,
        'volatility_threshold': 0.015,
        'lookback_period': 30,
        'is_primary': True      # 메인 레짐 결정자
    },
    '1d': {
        'rsi_weight': 0.7,
        'macd_weight': 0.2,
        'volume_weight': 0.1,
        'volatility_threshold': 0.01,
        'lookback_period': 60
    }
}

# 레짐 안정화 기본 파라미터 (개선: 민감도 증가)
REGIME_MIN_STAY = 2            # 최소 체류 캔들 수 (3 → 2로 완화)
REGIME_CONF_GATE = 0.4         # 전환 허용 신뢰도 임계값 (0.5 → 0.4로 완화)

# 🚀 심리도 계산 유틸 함수들
def _compute_sentiment_series(df: pd.DataFrame) -> pd.Series:
    """심리도 점수 계산 (-1 ~ +1)"""
    rsi = df.get('rsi', pd.Series(50.0, index=df.index)).fillna(50.0)
    macd = df.get('macd', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_sig = df.get('macd_signal', pd.Series(0.0, index=df.index)).fillna(0.0)
    volr = df.get('volume_ratio', pd.Series(1.0, index=df.index)).fillna(1.0)
    vol = df.get('atr', pd.Series(0.02, index=df.index)).fillna(0.02)
    wave_phase = df.get('wave_phase', pd.Series('unknown', index=df.index)).fillna('unknown')
    patt_conf = df.get('pattern_confidence', pd.Series(0.5, index=df.index)).fillna(0.5)

    # MACD 성분 계산
    macd_mag = (macd.abs() / (macd.abs().ewm(span=20, min_periods=1).mean() + 1e-9)).clip(0, 1)
    macd_side = np.sign(macd - macd_sig).astype(float)

    # 심리도 점수 계산 (가중합)
    sent = (
        0.35 * np.tanh((rsi - 50.0) / 10.0) +
        0.25 * (macd_side * macd_mag) +
        0.20 * np.clip(np.log(volr.replace(0, 1e-9)), -1, 1) +
        0.20 * (wave_phase.isin(['impulse']).astype(float)
                - wave_phase.isin(['correction']).astype(float)) * 0.8 +
        0.10 * (patt_conf - 0.5) -
        0.15 * vol.clip(0, 1)
    )
    return sent.clip(-1, 1)

def _label_sentiment(v: float) -> str:
    """심리도 점수를 라벨로 변환"""
    if v >= 0.6:   return 'very_bullish'
    if v >= 0.3:   return 'bullish'
    if v <= -0.6:  return 'very_bearish'
    if v <= -0.3:  return 'bearish'
    return 'neutral'

# -------------------- 분석 파라미터 --------------------
ZIGZAG_LOOKBACK_MAP = {
    # 5m 제거
    '15m': 3,
    '30m': 3,
    '240m': 2,
    '1d': 2,
    '1w': 1
}
MIN_REQUIREMENTS = {
    'min_zigzag_points': 1,  # 최소 1개로 완화
    'min_unique_pivots': 1,
    'min_wave_progress': 0.001,  # 기준 완화
    'min_pattern_confidence': 0.01  # 기준 완화
}

# -------------------- 분석 함수 (rl_candles_integrated.py 방식) --------------------
def validate_zigzag_data(df: pd.DataFrame, interval: str) -> bool:
    if 'zigzag_direction' not in df.columns:
        return False
    non_zero_directions = (df['zigzag_direction'] != 0).sum()
    if non_zero_directions < MIN_REQUIREMENTS['min_zigzag_points']:
        return False
    return True

def add_zigzag(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    lookback = ZIGZAG_LOOKBACK_MAP.get(interval, 2)  # 기본값 2로 변경
    close = df['close'].values
    zz_direction = [0] * len(close)
    zz_pivot_price = [np.nan] * len(close)  # 전환점 가격만 저장
    
    # None 값 처리: None을 np.nan으로 변환
    close = np.array([np.nan if x is None else x for x in close])
    
    change_count = 0
    
    for i in range(lookback, len(close) - lookback):
        window = close[i - lookback:i + lookback + 1]
        center = close[i]
        
        # None/nan 값 체크
        if pd.isna(center) or pd.isna(window).all():
            continue
            
        # window에서 nan 값 제거 후 최대/최소 계산
        valid_window = window[~pd.isna(window)]
        if len(valid_window) == 0:
            continue
            
        window_max = valid_window.max()
        window_min = valid_window.min()
        
        if center == window_max:
            zz_direction[i] = 1
            zz_pivot_price[i] = center  # 전환점 가격 저장
            change_count += 1
        elif center == window_min:
            zz_direction[i] = -1
            zz_pivot_price[i] = center  # 전환점 가격 저장
            change_count += 1
    
    # 전환점 부족 시 대체 계산
    if change_count < 1:
        # 대체 계산 방식: 단순한 고점/저점 찾기
        zz_direction = [0] * len(close)
        zz_pivot_price = [np.nan] * len(close)
        
        for i in range(1, len(close) - 1):
            current = close[i]
            prev = close[i-1]
            next_val = close[i+1]
            
            if not (pd.isna(current) or pd.isna(prev) or pd.isna(next_val)):
                # 고점 판단 (이전과 다음보다 높음)
                if current > prev and current > next_val:
                    zz_direction[i] = 1
                    zz_pivot_price[i] = current
                    change_count += 1
                # 저점 판단 (이전과 다음보다 낮음)
                elif current < prev and current < next_val:
                    zz_direction[i] = -1
                    zz_pivot_price[i] = current
                    change_count += 1
    
    # zigzag_direction이 모두 0인지 확인
    non_zero_directions = sum(1 for d in zz_direction if d != 0)
    if non_zero_directions == 0:
        # 최소한의 전환점 생성 (첫 번째와 마지막 캔들)
        if len(close) >= 2:
            zz_direction[0] = 1  # 첫 번째를 고점으로
            zz_pivot_price[0] = close[0]
            zz_direction[-1] = -1  # 마지막을 저점으로
            zz_pivot_price[-1] = close[-1]
            change_count = 2
    
    df['zigzag_direction'] = zz_direction
    df['zigzag_pivot_price'] = zz_pivot_price
    return df

def analyze_wave_structure_new(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    zigzag_valid = validate_zigzag_data(df, interval)
    if not zigzag_valid:
        df['wave_number'] = 0
        df['wave_progress'] = 0.5  # 기본값 0.5로 변경
        df['wave_phase'] = 'unknown'
        df['integrated_wave_phase'] = 'unknown'
        df['structure_score'] = 0.5
        df['wave_step'] = 0
        df['pattern_class'] = 'unknown'
        return df
    wave_numbers = []
    current_wave = 0
    last_direction = 0
    for i, direction in enumerate(df['zigzag_direction']):
        if direction != 0 and direction != last_direction:
            current_wave += 1
            last_direction = direction
        wave_numbers.append(current_wave)
    df['wave_number'] = wave_numbers
    wave_progress = []
    lookback = ZIGZAG_LOOKBACK_MAP.get(interval, 2)  # 기본값 2로 변경
    for i in range(len(df)):
        if i < lookback:
            wave_progress.append(0.5)  # 기본값 0.5로 변경
            continue
        current_direction = df['zigzag_direction'].iloc[i]
        if current_direction != 0:
            prev_pivot_idx = i - 1
            while prev_pivot_idx >= 0 and df['zigzag_direction'].iloc[prev_pivot_idx] == 0:
                prev_pivot_idx -= 1
            if prev_pivot_idx >= 0:
                prev_price = df['zigzag_pivot_price'].iloc[prev_pivot_idx]
                current_price = df['close'].iloc[i]
                if pd.notna(prev_price) and prev_price != 0:
                    if current_direction == 1:
                        progress = (current_price - prev_price) / (df['high'].iloc[i] - prev_price + 1e-9)
                    else:
                        progress = (prev_price - current_price) / (prev_price - df['low'].iloc[i] + 1e-9)
                    wave_progress.append(progress.clip(0, 1))
                else:
                    wave_progress.append(0.5)  # 기본값 0.5로 변경
            else:
                wave_progress.append(0.5)  # 기본값 0.5로 변경
        else:
            wave_progress.append(0.5)  # 기본값 0.5로 변경
    df['wave_progress'] = wave_progress
    wave_phases = []
    for i in range(len(df)):
        direction = df['zigzag_direction'].iloc[i]
        progress = df['wave_progress'].iloc[i]
        if direction == 1:
            if progress > 0.7:
                wave_phases.append('impulse')
            elif progress > 0.3:
                wave_phases.append('correction')
            else:
                wave_phases.append('consolidation')
        elif direction == -1:
            if progress > 0.7:
                wave_phases.append('correction')
            elif progress > 0.3:
                wave_phases.append('impulse')
            else:
                wave_phases.append('consolidation')
        else:
            wave_phases.append('unknown')
    df['wave_phase'] = wave_phases
    df['integrated_wave_phase'] = wave_phases
    return df

def analyze_pattern_structure_new(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    zigzag_valid = validate_zigzag_data(df, interval)
    lookback = ZIGZAG_LOOKBACK_MAP.get(interval, 6)
    pattern_types, pattern_qualities, pattern_directions = [], [], []
    pattern_volume_ratios, pattern_pivot_strengths, pattern_start_indices, pattern_end_indices = [], [], [], []
    for i in range(len(df)):
        if i < lookback:
            pattern_types.append('none')
            pattern_qualities.append(0.0)
            pattern_directions.append('neutral')
            pattern_volume_ratios.append(1.0)
            pattern_pivot_strengths.append(0.0)
            pattern_start_indices.append(0)
            pattern_end_indices.append(0)
            continue
        current_direction = df['zigzag_direction'].iloc[i]
        if current_direction != 0 and zigzag_valid:
            pattern_end = i
            pattern_start = i - 1
            while pattern_start >= 0 and df['zigzag_direction'].iloc[pattern_start] == 0:
                pattern_start -= 1
            if pattern_start >= 0:
                pattern_data = df.iloc[pattern_start:pattern_end+1]
                if current_direction == 1:
                    pattern_type = 'ascending_triangle' if len(pattern_data) >= 3 else 'uptrend'
                    pattern_direction = 'bullish'
                elif current_direction == -1:
                    pattern_type = 'descending_triangle' if len(pattern_data) >= 3 else 'downtrend'
                    pattern_direction = 'bearish'
                else:
                    pattern_type = 'sideways'
                    pattern_direction = 'neutral'
                avg_volume_ratio = pattern_data['volume_ratio'].mean() if 'volume_ratio' in pattern_data.columns else 1.0
                avg_volatility = (pattern_data['atr'] / pattern_data['close']).mean() if 'atr' in pattern_data.columns and 'close' in pattern_data.columns else 0.02
                pattern_quality = avg_volume_ratio * avg_volatility * 5  # pattern_confidence 대신 pattern_quality만 계산
                pivot_strength = pattern_data['pivot_point'].sum() / len(pattern_data) if 'pivot_point' in pattern_data.columns else 0.0
                pattern_types.append(pattern_type)
                pattern_qualities.append(pattern_quality)
                pattern_directions.append(pattern_direction)
                pattern_volume_ratios.append(avg_volume_ratio)
                pattern_pivot_strengths.append(pivot_strength)
                pattern_start_indices.append(pattern_start)
                pattern_end_indices.append(pattern_end)
            else:
                pattern_types.append('none')
                pattern_qualities.append(0.0)
                pattern_directions.append('neutral')
                pattern_volume_ratios.append(1.0)
                pattern_pivot_strengths.append(0.0)
                pattern_start_indices.append(0)
                pattern_end_indices.append(0)
        else:
            pattern_types.append('none')
            pattern_qualities.append(0.0)
            pattern_directions.append('neutral')
            pattern_volume_ratios.append(1.0)
            pattern_pivot_strengths.append(0.0)
            pattern_start_indices.append(0)
            pattern_end_indices.append(0)
    df['pattern_type'] = pattern_types
    # ✅ pattern_confidence는 realtime_candles_calculate.py에서 계산됨
    df['pattern_quality'] = pattern_qualities
    df['pattern_direction'] = pattern_directions
    df['pattern_volume_ratio'] = pattern_volume_ratios
    df['pattern_pivot_strength'] = pattern_pivot_strengths
    df['pattern_start_idx'] = pattern_start_indices
    df['pattern_end_idx'] = pattern_end_indices
    return df

def compute_wave_step(df: pd.DataFrame) -> pd.Series:
    """🚀 벡터화 연산으로 최적화된 파동 단계 계산"""
    # 기본값 설정
    wave_num = df.get('wave_number', pd.Series(0, index=df.index))
    wave_progress = df.get('wave_progress', pd.Series(0.0, index=df.index))
    zigzag_direction = df.get('zigzag_direction', pd.Series(0, index=df.index))
    
    # NaN 값 처리
    wave_progress = wave_progress.fillna(0.0)
    wave_num = wave_num.fillna(0)
    zigzag_direction = zigzag_direction.fillna(0)
    
    # 🚀 벡터화된 파동 단계 계산
    wave_step = pd.cut(wave_progress, 
                      bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      include_lowest=True).astype(int)
    
    # 🚀 벡터화된 방향 조정
    wave_step = np.where(zigzag_direction == -1, 11 - wave_step, wave_step)
    
    # 🚀 유효하지 않은 파동은 0으로 설정
    wave_step = np.where((wave_num == 0) | pd.isna(wave_progress), 0, wave_step)
    
    return pd.Series(wave_step, index=df.index)

def evaluate_fractal_structure(df: pd.DataFrame) -> pd.Series:
    structure_scores = []
    for i in range(len(df)):
        base_score = 0.0
        wave_num = df['wave_number'].iloc[i] if 'wave_number' in df.columns else 0
        if wave_num > 0:
            if wave_num <= 3:
                wave_continuity = wave_num / 3.0
            elif wave_num <= 7:
                wave_continuity = 0.5 + (wave_num - 3) / 8.0
            else:
                wave_continuity = 0.75 + (wave_num - 7) / 20.0
            wave_continuity = min(wave_continuity, 1.0)
            base_score += 0.25 * wave_continuity
        wave_progress = df['wave_progress'].iloc[i] if 'wave_progress' in df.columns else 0.0
        if pd.notna(wave_progress):
            if wave_progress < 0.2:
                progress_score = wave_progress * 2.5
            elif wave_progress < 0.4:
                progress_score = 0.5 + (wave_progress - 0.2) * 1.25
            elif wave_progress < 0.6:
                progress_score = 0.75 + (wave_progress - 0.4) * 0.625
            elif wave_progress < 0.8:
                progress_score = 0.875 + (wave_progress - 0.6) * 0.625
            else:
                progress_score = 1.0 - (wave_progress - 0.8) * 2.5
            progress_score = max(0.0, min(1.0, progress_score))
            base_score += 0.2 * progress_score
        if 'pattern_confidence' in df.columns:
            pattern_conf = df['pattern_confidence'].iloc[i]
            if pd.notna(pattern_conf):
                if pattern_conf < 0.3:
                    pattern_score = pattern_conf * 1.5
                elif pattern_conf < 0.6:
                    pattern_score = 0.45 + (pattern_conf - 0.3) * 0.5
                elif pattern_conf < 0.8:
                    pattern_score = 0.6 + (pattern_conf - 0.6) * 1.0
                else:
                    pattern_score = 0.8 + (pattern_conf - 0.8) * 1.0
                base_score += 0.25 * pattern_score
        if 'volume_ratio' in df.columns:
            volume_ratio = df['volume_ratio'].iloc[i]
            if pd.notna(volume_ratio):
                if volume_ratio < 0.5:
                    volume_score = volume_ratio * 1.0
                elif volume_ratio < 1.0:
                    volume_score = 0.5 + (volume_ratio - 0.5) * 0.5
                elif volume_ratio < 2.0:
                    volume_score = 0.75 + (volume_ratio - 1.0) * 0.25
                else:
                    volume_score = 1.0 - (volume_ratio - 2.0) * 0.1
                volume_score = max(0.0, min(1.0, volume_score))
                base_score += 0.15 * volume_score
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[i]
            if pd.notna(rsi):
                if rsi < 20:
                    rsi_score = 0.2
                elif rsi < 30:
                    rsi_score = 0.4
                elif rsi < 45:
                    rsi_score = 0.6
                elif rsi < 55:
                    rsi_score = 0.8
                elif rsi < 70:
                    rsi_score = 0.6
                elif rsi < 80:
                    rsi_score = 0.4
                else:
                    rsi_score = 0.2
                base_score += 0.15 * rsi_score
        structure_scores.append(min(1.0, base_score))
    return pd.Series(structure_scores, index=df.index)

def classify_pattern_structure(df: pd.DataFrame) -> pd.Series:
    pattern_classes = []
    for i in range(len(df)):
        pattern_class = 'unknown'
        pattern_direction = df.get('pattern_direction', pd.Series(['neutral']*len(df))).iloc[i]
        wave_phase = df.get('wave_phase', pd.Series(['unknown']*len(df))).iloc[i]
        
        # ✅ None 값 처리 강화
        if pd.isna(pattern_direction) or pattern_direction is None:
            pattern_direction = 'neutral'
        if pd.isna(wave_phase) or wave_phase is None:
            wave_phase = 'unknown'
        # ✅ wave_step None 값 처리 강화
        if 'wave_step' in df.columns:
            wave_step_val = df['wave_step'].iloc[i]
            if pd.isna(wave_step_val) or wave_step_val is None:
                wave_step = 0
            else:
                try:
                    wave_step = int(wave_step_val)
                except (ValueError, TypeError):
                    wave_step = 0
        else:
            wave_step = 0
        pattern_type = df.get('pattern_type', pd.Series(['none']*len(df))).iloc[i]
        pattern_confidence = df.get('pattern_confidence', pd.Series([0.0]*len(df))).iloc[i]
        
        # ✅ None 값 처리 강화
        if pd.isna(pattern_confidence) or pattern_confidence is None:
            pattern_confidence = 0.0
        else:
            try:
                pattern_confidence = float(pattern_confidence)
            except (ValueError, TypeError):
                pattern_confidence = 0.0
        
        if pattern_direction == 'bullish':
            if wave_phase == 'impulse':
                if wave_step >= 4:
                    if pattern_confidence > 0.7:
                        pattern_class = 'bullish_impulse_late_strong'
                    else:
                        pattern_class = 'bullish_impulse_late'
                elif wave_step >= 2:
                    pattern_class = 'bullish_impulse_mid'
                else:
                    pattern_class = 'bullish_impulse_early'
            elif wave_phase == 'correction':
                pattern_class = 'bullish_correction'
            else:
                pattern_class = 'bullish_consolidation'
        elif pattern_direction == 'bearish':
            if wave_phase == 'impulse':
                if wave_step >= 4:
                    if pattern_confidence > 0.7:
                        pattern_class = 'bearish_impulse_late_strong'
                    else:
                        pattern_class = 'bearish_impulse_late'
                elif wave_step >= 2:
                    pattern_class = 'bearish_impulse_mid'
                else:
                    pattern_class = 'bearish_impulse_early'
            elif wave_phase == 'correction':
                pattern_class = 'bearish_correction'
            else:
                pattern_class = 'bearish_consolidation'
        else:
            if wave_phase == 'consolidation':
                pattern_class = 'sideways_consolidation'
            else:
                if pattern_confidence < 0.3:
                    pattern_class = 'sideways_unknown_low_confidence'
                else:
                    pattern_class = 'sideways_unknown'
        pattern_classes.append(pattern_class)
    return pd.Series(pattern_classes, index=df.index)

def calculate_volatility_level(volatility):
    if pd.isna(volatility):
        return 'unknown'
    if volatility > 0.05:
        return 'high'
    elif volatility > 0.02:
        return 'medium'
    else:
        return 'low'

def calculate_risk_level(risk_score):
    if pd.isna(risk_score):
        return 'unknown'
    if risk_score > 0.7:
        return 'high'
    elif risk_score > 0.4:
        return 'medium'
    else:
        return 'low'

def calculate_flow_level_meta_simple(df):
    """간단한 Flow Level 메타데이터 계산"""
    if len(df) < 5:
        return 'Neutral'
    
    # RSI 기반 추세 판단
    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
    rsi_ema = df['rsi_ema'].iloc[-1] if 'rsi_ema' in df.columns and not pd.isna(df['rsi_ema'].iloc[-1]) else rsi
    
    # MACD 기반 모멘텀 확인
    macd = df['macd'].iloc[-1] if 'macd' in df.columns and not pd.isna(df['macd'].iloc[-1]) else 0
    macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns and not pd.isna(df['macd_signal'].iloc[-1]) else 0
    
    # 거래량 기반 강도 판단
    volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns and not pd.isna(df['volume_ratio'].iloc[-1]) else 1.0
    
    # 추세 판단
    if rsi > 70 and rsi_ema > 65:
        trend = "strong_up"
    elif rsi > 60 and rsi_ema > 55:
        trend = "up"
    elif rsi < 30 and rsi_ema < 35:
        trend = "strong_down"
    elif rsi < 40 and rsi_ema < 45:
        trend = "down"
    else:
        trend = "sideways"
    
    # 모멘텀 판단
    if macd > macd_signal and macd > 0:
        momentum = "bullish"
    elif macd < macd_signal and macd < 0:
        momentum = "bearish"
    else:
        momentum = "neutral"
    
    # 거래량 강도 판단
    if volume_ratio > 2.0:
        volume_strength = "high"
    elif volume_ratio > 1.5:
        volume_strength = "medium"
    else:
        volume_strength = "low"
    
    # 통합 판단
    if trend in ["strong_up", "up"] and momentum == "bullish":
        if volume_strength == "high":
            return "Momentum Bull"
        else:
            return "Pullback Bull"
    elif trend in ["strong_down", "down"] and momentum == "bearish":
        return "Exhaustion Bear"
    else:
        return "Neutral"

# 🚀 레짐 계산 함수들
def calculate_composite_regime_score(df: pd.DataFrame, interval: str) -> pd.Series:
    """인터벌별 복합 지표 기반 레짐 점수 계산"""
    criteria = REGIME_CRITERIA.get(interval, REGIME_CRITERIA['30m'])
    
    # RSI 점수 (0-1 정규화)
    rsi_data = df.get('rsi', pd.Series(50.0, index=df.index)).fillna(50.0)
    rsi_score = (rsi_data - 20) / 60  # 20-80을 0-1로 변환
    rsi_score = rsi_score.clip(0, 1)
    
    # MACD 모멘텀 점수
    macd_data = df.get('macd', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_signal_data = df.get('macd_signal', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_momentum = (macd_data - macd_signal_data).abs()
    macd_score = macd_momentum / (macd_momentum.rolling(20).max() + 1e-9)
    
    # Volume 강도 점수
    volume_data = df.get('volume_ratio', pd.Series(1.0, index=df.index)).fillna(1.0)
    volume_score = volume_data.clip(0, 3) / 3
    
    # Volatility 점수 (안정성 측면)
    atr_data = df.get('atr', pd.Series(0.02, index=df.index)).fillna(0.02)
    close_data = df.get('close', pd.Series(100.0, index=df.index)).fillna(100.0)
    volatility_score = 1 - (atr_data / close_data).clip(0, 0.1) / 0.1
    
    # 가중 평균
    composite_score = (
        criteria['rsi_weight'] * rsi_score +
        criteria['macd_weight'] * macd_score +
        criteria['volume_weight'] * volume_score +
        0.1 * volatility_score  # 안정성 보너스
    )
    
    return composite_score.clip(0, 1)

def classify_regime_stage(composite_score: pd.Series, interval: str) -> pd.Series:
    """복합 점수를 7단계 레짐으로 분류"""
    
    # NaN 값 처리
    composite_score = composite_score.fillna(0.5)  # 기본값: neutral
    
    # 인터벌별 임계값 조정 (중복 제거)
    thresholds = {
        '15m': [0.2, 0.35, 0.5, 0.6, 0.75, 0.9],
        '30m': [0.25, 0.4, 0.55, 0.65, 0.8, 0.95],
        '240m': [0.3, 0.45, 0.6, 0.7, 0.85, 0.99],
        '1d': [0.3, 0.45, 0.6, 0.7, 0.85, 0.99]
    }
    
    thresh = thresholds.get(interval, [0.25, 0.4, 0.55, 0.65, 0.8, 0.95])
    
    # 중복 제거된 bins 생성
    bins = [0] + thresh + [1.0]
    bins = sorted(list(set(bins)))  # 중복 제거 및 정렬
    
    regime_stage = pd.cut(composite_score, 
                         bins=bins,
                         labels=list(range(1, len(bins))),
                         include_lowest=True,
                         duplicates='drop')  # 중복 제거 옵션
    
    # NaN 값 처리 및 정수 변환
    regime_stage = regime_stage.fillna(4)  # 기본값: neutral
    regime_stage = regime_stage.astype(int)
    
    return regime_stage

def calculate_regime_confidence(df: pd.DataFrame, interval: str) -> pd.Series:
    """레짐 신뢰도 계산"""
    criteria = REGIME_CRITERIA.get(interval, REGIME_CRITERIA['30m'])
    lookback = criteria['lookback_period']
    
    if len(df) < lookback:
        return pd.Series(0.5, index=df.index)
    
    # RSI 일관성
    rsi_data = df.get('rsi', pd.Series(50.0, index=df.index)).fillna(50.0)
    rsi_std = rsi_data.rolling(lookback).std()
    rsi_consistency = (1 - rsi_std / 20).clip(0, 1)
    
    # MACD 신호 강도 (과포화 방지 - 롤링 최대치 기준 정규화)
    macd_data = df.get('macd', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_signal_data = df.get('macd_signal', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_diff = (macd_data - macd_signal_data).abs()
    macd_strength = macd_diff.rolling(lookback).mean()
    macd_strength = (macd_strength / (macd_strength.rolling(lookback).max() + 1e-9)).clip(0, 1)
    
    # Volume 일관성
    volume_data = df.get('volume_ratio', pd.Series(1.0, index=df.index)).fillna(1.0)
    volume_std = volume_data.rolling(lookback).std()
    volume_consistency = (1 - volume_std / 2.0).clip(0, 1)
    
    # 종합 신뢰도
    confidence = (
        criteria['rsi_weight'] * rsi_consistency +
        criteria['macd_weight'] * macd_strength +
        criteria['volume_weight'] * volume_consistency
    )
    
    return confidence.clip(0, 1)

def calculate_regime_transition_probability(df: pd.DataFrame, interval: str) -> pd.Series:
    """레짐 전환 확률 계산 (1단계 전환 포함, 차등 가중치 적용)"""
    criteria = REGIME_CRITERIA.get(interval, REGIME_CRITERIA['30m'])
    lookback = criteria['lookback_period']
    
    if len(df) < lookback * 2:
        return pd.Series(0.05, index=df.index)  # 기본값 0.1→0.05로 낮춤
    
    # 점수 계산 및 평활화
    composite_score = calculate_composite_regime_score(df, interval)
    smooth_score = composite_score.ewm(span=lookback, min_periods=1).mean()
    regime_stage = classify_regime_stage(smooth_score, interval)
    
    # 신뢰도 계산
    confidence = calculate_regime_confidence(df, interval)
    
    # 🚀 1단계와 2단계 이상 전환에 차등 가중치 적용
    changes = regime_stage.diff().abs()
    
    # 1단계 전환: 신뢰도 기반 가중치 (0.3 ~ 0.6)
    minor_weight = 0.3 + (confidence * 0.3)  # 신뢰도 높을수록 가중치 증가
    minor_changes = (changes == 1).astype(float) * minor_weight
    
    # 2단계 이상 전환: 전체 가중치 (1.0)
    major_changes = (changes >= 2).astype(float) * 1.0
    
    # 통합 전환 신호
    all_changes = minor_changes + major_changes
    
    # 롤링 평균으로 전환 빈도 계산
    change_frequency = all_changes.rolling(lookback, min_periods=1).mean()
    
    # 상한선 조정 (0.5 → 0.4, 더 넓은 범위 활용)
    transition_prob = change_frequency.clip(0, 0.4)
    
    return transition_prob.fillna(0.05)

# 🚀 ml_candles_calculate.py와 동일한 저장 함수 추가
def save_integrated_indicators_immediate(df: pd.DataFrame, coin: str, interval: str) -> bool:
    """🚀 통합 분석 완료 즉시 저장 - ml_candles_calculate.py의 성공적인 방식 적용"""
    try:
        if df.empty:
            return False
        
        # 📌 소숫점 자리수 통일 적용 (integrated 컬럼) - 4자리로 통일
        rounding_map = {
            'atr': 4,
            'risk_score': 4,
            'integrated_strength': 4,
            'sentiment': 4,  # 🚀 심리도 점수 추가
            'regime_confidence': 4,  # 🚀 레짐 신뢰도 추가
            'regime_transition_prob': 4  # 🚀 레짐 전환 확률 추가
        }

        # 반올림 적용 (존재하는 컬럼만)
        for col, digits in rounding_map.items():
            if col in df.columns:
                df[col] = df[col].round(digits)
        
        # 실제로 테이블에 존재하는 integrated 컬럼 리스트
        integrated_columns = [
            'volatility_level', 'risk_level', 'integrated_direction',
            'sentiment', 'sentiment_label',  # 🚀 심리도 컬럼 추가
            'regime_stage', 'regime_label', 'regime_confidence', 'regime_transition_prob'  # 🚀 레짐 컬럼 추가
        ]

        # 커넥션 열고
        with sqlite3.connect(DB_PATH) as conn:
            # 🚀 SQLite 성능 최적화 설정
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # 동기화 레벨 조정
            conn.execute("PRAGMA cache_size=10000")  # 캐시 크기 증가
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB 메모리 매핑
            
            # 실제로 테이블에 존재하는 컬럼만 확인
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(candles);")
            table_columns = [col[1] for col in cursor.fetchall()]
            
            # DataFrame에 존재하고 테이블에도 존재하는 컬럼만 선택
            existing_columns = [c for c in integrated_columns if c in df.columns and c in table_columns]
            
            # 🔧 누락된 컬럼 자동 추가
            missing_columns = []
            for col in integrated_columns:
                if col in df.columns and col not in table_columns:
                    missing_columns.append(col)
            
            if missing_columns:
                for col in missing_columns:
                    # 컬럼 타입 결정
                    if col in ['volatility_level', 'risk_level', 'integrated_direction', 'sentiment_label', 'regime_label']:
                        col_type = 'TEXT'
                    elif col in ['regime_stage']:
                        col_type = 'INTEGER'
                    else:
                        col_type = 'REAL'
                    
                    try:
                        cursor.execute(f'ALTER TABLE candles ADD COLUMN "{col}" {col_type}')
                    except Exception as e:
                        continue
                
                # 컬럼 추가 후 다시 확인
                cursor.execute("PRAGMA table_info(candles);")
                table_columns = [col[1] for col in cursor.fetchall()]
                existing_columns = [c for c in integrated_columns if c in df.columns and c in table_columns]
            
            if not existing_columns:
                return False
            
            # 🚀 대량 UPDATE 최적화 (기존 데이터 보존하면서 integrated 컬럼만 업데이트)
            total_rows = len(df)
            total_updated = 0
            
            # 🚀 데이터 전처리 (한 번에 처리)
            update_data = []
            for _, row in df.iterrows():
                row_data = []
                for col in existing_columns:
                    if pd.notna(row[col]):
                        value = row[col]
                        # 🚀 컬럼 타입별 안전한 변환
                        if col in ['volatility_level', 'risk_level', 'integrated_direction', 'sentiment_label', 'regime_label']:
                            # TEXT 타입 컬럼 - 문자열로 변환
                            value = str(value) if value is not None else 'unknown'
                        elif col in ['regime_stage']:
                            # INTEGER 타입 컬럼 - 정수로 변환
                            try:
                                value = int(value) if value is not None else 4
                            except (ValueError, TypeError):
                                value = 4  # 기본값: neutral
                        else:
                            # REAL 타입 컬럼 - 숫자로 변환
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                value = 0.0
                        row_data.append(value)
                    else:
                        row_data.append(None)
                
                # 키 값들 추가 (기존 데이터 식별용)
                row_data.extend([row['coin'], row['interval'], row['timestamp']])
                update_data.append(row_data)
            
            if update_data:
                # 🚀 executemany로 배치 업데이트 (기존 데이터는 보존)
                set_clauses = [f'"{col}" = ?' for col in existing_columns]
                
                sql = f"""
                    UPDATE candles 
                    SET {', '.join(set_clauses)}
                    WHERE coin = ? AND interval = ? AND timestamp = ?
                """
                
                cursor.executemany(sql, update_data)
                total_updated = len(update_data)
                conn.commit()
        
        # 🔍 저장 결과 검증
        if total_updated > 0:
            return True
        else:
            return False
        
    except Exception as e:
        print(f"    ❌ 통합 분석 저장 중 오류: {coin}/{interval} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def perform_integrated_analysis(coin: str, interval: str):
    """🚀 9개 통합 컬럼 계산 - volatility_level, risk_level, integrated_direction, sentiment, sentiment_label, regime_stage, regime_label, regime_confidence, regime_transition_prob"""
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA busy_timeout = 30000")
            
            df = pd.read_sql(
                "SELECT * FROM candles WHERE coin=? AND interval=? ORDER BY timestamp",
                conn, params=(coin, interval)
            )
            if df.empty or len(df) < 20:
                return
        
        # 🚀 3개 통합 컬럼만 계산
        # volatility_level 계산
        if 'atr' in df.columns:
            df['volatility_level'] = df['atr'].apply(calculate_volatility_level)
        else:
            df['volatility_level'] = 'unknown'
        
        # risk_level 계산
        if 'risk_score' in df.columns:
            df['risk_level'] = df['risk_score'].apply(calculate_risk_level)
        else:
            df['risk_level'] = 'unknown'
        
        # integrated_direction 계산 (파동 및 패턴 정보 통합 반영)
        if 'rsi' in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
            rsi_data = df['rsi'].fillna(50)
            macd_data = df['macd'].fillna(0)
            macd_signal_data = df['macd_signal'].fillna(0)
            
            # 🚀 기존 RSI/MACD 기반 방향성 계산
            rsi_direction = np.where(rsi_data > 70, 'bearish', 
                                   np.where(rsi_data < 30, 'bullish', 'neutral'))
            macd_direction = np.where(macd_data > macd_signal_data, 'bullish', 'bearish')
            
            # 🚀 파동 및 패턴 정보 통합 반영 (개선사항)
            wave_phase_data = df.get('wave_phase', pd.Series(['unknown'] * len(df)))
            pattern_type_data = df.get('pattern_type', pd.Series(['none'] * len(df)))
            pattern_confidence_data = df.get('pattern_confidence', pd.Series([0.0] * len(df)))
            structure_score_data = df.get('structure_score', pd.Series([0.5] * len(df)))
            
            # 🚀 구조적 신호 조건들
            strong_bullish_condition = (
                (wave_phase_data == 'impulse') & 
                (pattern_type_data.isin(['uptrend', 'strong_uptrend']))
            )
            
            strong_bearish_condition = (
                (wave_phase_data == 'correction') & 
                (pattern_type_data.isin(['downtrend', 'strong_downtrend']))
            )
            
            structural_signal_condition = (
                (pattern_confidence_data > 0.7) & 
                (structure_score_data > 0.7)
            )
            
            # 🚀 통합 방향성 결정 (우선순위: 구조적 신호 > 기존 모멘텀)
            df['integrated_direction'] = np.select([
                strong_bullish_condition,
                strong_bearish_condition,
                structural_signal_condition,
                (rsi_direction == 'bullish') & (macd_direction == 'bullish'),
                (rsi_direction == 'bearish') & (macd_direction == 'bearish'),
                rsi_direction == 'neutral'
            ], [
                'strong_bullish',
                'strong_bearish', 
                'structural_signal',
                'bullish',
                'bearish',
                'neutral'
            ], default='mixed')
        else:
            df['integrated_direction'] = 'neutral'
        
        # 🚀 심리도 계산 및 저장 (개선사항)
        sent_series = _compute_sentiment_series(df)
        df['sentiment'] = sent_series.round(4)
        df['sentiment_label'] = [_label_sentiment(x) for x in sent_series]
        
        # 🚀 레짐 계산 및 저장 (평활화 + 안정화 적용)
        composite_score = calculate_composite_regime_score(df, interval)
        # 1) 점수 평활화(EWM)
        lookback = REGIME_CRITERIA.get(interval, REGIME_CRITERIA['30m'])['lookback_period']
        smooth_score = composite_score.ewm(span=lookback, min_periods=1).mean()
        raw_stage = classify_regime_stage(smooth_score, interval)

        # 2) 최소 체류시간 + 신뢰도 게이트
        conf_series = calculate_regime_confidence(df, interval).round(4)
        stable_stage = raw_stage.copy()
        if len(stable_stage) > 0:
            last = int(stable_stage.iloc[0])
            stay = 1
            for i in range(1, len(stable_stage)):
                cand = int(stable_stage.iloc[i])
                conf = float(conf_series.iloc[i]) if not pd.isna(conf_series.iloc[i]) else 0.5
                if cand != last and (stay < REGIME_MIN_STAY or conf < REGIME_CONF_GATE):
                    stable_stage.iloc[i] = last
                    stay += 1
                else:
                    stable_stage.iloc[i] = cand
                    if cand == last:
                        stay += 1
                    else:
                        last = cand
                        stay = 1

        df['regime_stage'] = stable_stage.astype(int)
        df['regime_label'] = [REGIME_STAGES.get(int(stage), 'neutral') for stage in df['regime_stage']]
        df['regime_confidence'] = conf_series
        df['regime_transition_prob'] = calculate_regime_transition_probability(df, interval).round(4)
        
        # 🚀 NULL 값 방지: 시계열 연속성을 고려한 보간법 적용
        # 숫자형 컬럼: 선형 보간법 사용
        numeric_columns = ['sentiment', 'regime_confidence', 'regime_transition_prob']
        for col in numeric_columns:
            if col in df.columns:
                # 선형 보간법으로 시간적 연속성 유지 (시작 부분은 NULL 유지)
                # 데이터 타입을 명시적으로 변환하여 FutureWarning 방지
                df[col] = pd.to_numeric(df[col], errors='coerce').interpolate(method='linear', limit_direction='forward')
                # 끝 부분만 Forward Fill로 처리 (시작 부분은 NULL 그대로 유지)
                df[col] = df[col].ffill()
        
        # 정수형 컬럼: 이전 값으로 채우기
        integer_columns = ['regime_stage']
        for col in integer_columns:
            if col in df.columns:
                # Forward Fill로 시간적 연속성 유지 (시작 부분은 NULL 유지)
                df[col] = df[col].ffill()
        
        # 텍스트형 컬럼: 이전 값으로 채우기
        text_columns = ['volatility_level', 'risk_level', 'integrated_direction', 'sentiment_label', 'regime_label']
        for col in text_columns:
            if col in df.columns:
                # Forward Fill로 시간적 연속성 유지 (시작 부분은 NULL 유지)
                df[col] = df[col].ffill()
        
        # 🚀 9개 통합 컬럼 저장 (volatility_level, risk_level, integrated_direction, sentiment, sentiment_label, regime_stage, regime_label, regime_confidence, regime_transition_prob)
        save_success = save_integrated_indicators_immediate(df, coin, interval)
        
        if save_success:
            print(f"✅ 통합 분석 완료: {coin}/{interval} → {len(df)}개 업데이트")
        else:
            print(f"❌ 통합 분석 저장 실패: {coin}/{interval}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {coin}/{interval} - {str(e)}")
        import traceback
        traceback.print_exc()
        return

def run_full_integrated_analysis():
    """🚀 9개 통합 컬럼만 계산하는 최적화된 통합분석"""
    print(f"🚀 통합 분석 시작 (9개 컬럼: volatility_level, risk_level, integrated_direction, sentiment, sentiment_label, regime_stage, regime_label, regime_confidence, regime_transition_prob)")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT coin, interval, COUNT(*) as count
            FROM candles 
            WHERE interval IN ('15m', '30m', '240m', '1d')
            AND rsi IS NOT NULL AND macd IS NOT NULL
            GROUP BY coin, interval
            ORDER BY coin, interval
        """)
        coin_intervals = cursor.fetchall()
    
    if not coin_intervals:
        print("⚠️ 처리할 데이터가 없습니다")
        return
    
    total_groups = len(coin_intervals)
    print(f"📊 처리 대상: {total_groups}개 코인/인터벌 그룹")
    
    success_count = 0
    error_count = 0
    batch_size = 10
    
    for i in range(0, len(coin_intervals), batch_size):
        batch = coin_intervals[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(coin_intervals) + batch_size - 1)//batch_size
        
        print(f"🔄 배치 {batch_num}/{total_batches}: {len(batch)}개 그룹 처리 중...")
        
        for coin, interval, count in batch:
            try:
                perform_integrated_analysis(coin, interval)
                success_count += 1
                print(f"✅ 통합 분석 성공: {coin}/{interval}")
            except Exception as e:
                error_count += 1
                print(f"❌ 통합 분석 오류: {coin}/{interval} - {str(e)}")
                continue
        
        import gc
        gc.collect()
        import time
        time.sleep(0.5)
    
    print(f"🎉 통합 분석 완료: 성공 {success_count}개, 실패 {error_count}개")


# ---------------- 실행부 ----------------
if __name__ == '__main__':
    # print('🚀 통합분석(파동+패턴+프랙탈+통합메타) 시작!')  # 제거됨
    run_full_integrated_analysis()
    # print('✅ 통합분석(파동+패턴+프랙탈+통합메타) 완료!')  # 제거됨 