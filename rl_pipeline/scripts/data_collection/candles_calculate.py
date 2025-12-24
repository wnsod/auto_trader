import sys
import os
sys.path.insert(0, "/workspace")

import sqlite3
import pandas as pd
import ta
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
from datetime import datetime

# 데이터베이스 경로 설정 (env에서 가져오기 - 환경변수 오버라이딩 지원)
from rl_pipeline.core.env import config
# 환경 변수 RL_DB_PATH가 있으면 그것을 사용, 없으면 config.RL_DB 사용 (하위 호환성)
DB_PATH = os.getenv('RL_DB_PATH', config.RL_DB)

# 컬럼 존재 여부 체크 및 추가 함수 정의
def ensure_column_exists(conn, table_name, column_name, column_type):
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info({table_name});')
    columns = [col[1] for col in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};')

def validate_data(df):
    """데이터 검증"""
    # 필수 컬럼 검증 (symbol로 변경)
    required_cols = ['timestamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # 하위 호환성: symbol이 없고 coin이 있으면 coin을 symbol로 사용 (임시)
        if 'symbol' in missing_cols and 'coin' in df.columns:
            df.rename(columns={'coin': 'symbol'}, inplace=True)
            return True
        return False
    
    # 값 범위 검증
    if (df['high'] < df['low']).any():
        return False
    
    if (df['volume'] < 0).any():
        return False
    
    return True

def handle_missing_values(df):
    """누락값 처리"""
    # 가격 데이터는 이전 값으로 채우기
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].ffill()
    
    # 거래량은 0으로 채우기
    df['volume'] = df['volume'].fillna(0)
    
    # 기술적 지표는 이전 값으로 채우기
    # ✅ 33개 핵심 컬럼만 유지 (텍스트 컬럼 포함)
    tech_cols = ['rsi', 'mfi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
                 'atr', 'ma20', 'adx', 'volume_ratio', 'risk_score',
                 'wave_phase', 'confidence', 'zigzag_direction', 'zigzag_pivot_price', 'wave_progress',
                 'pattern_type', 'pattern_confidence', 'structure_score']
    df[tech_cols] = df[tech_cols].ffill()
    
    df = df.ffill()
    
    return df

# Wave Phase 분석 함수
def determine_wave_phase(df):
    """개선된 파동 단계 판단 (레짐별 가변 임계값 적용)"""
    if len(df) < 10:
        return 'unknown'
    
    try:
        # 🚀 기술지표 기반 파동 단계 판단
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
        macd = df['macd'].iloc[-1] if 'macd' in df.columns and not pd.isna(df['macd'].iloc[-1]) else 0
        macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns and not pd.isna(df['macd_signal'].iloc[-1]) else 0
        
        # 🚀 파동 진행률 기반 판단
        wave_progress = df['wave_progress'].iloc[-1] if 'wave_progress' in df.columns and not pd.isna(df['wave_progress'].iloc[-1]) else 0.5
        
        # 🚀 가격 모멘텀 기반 판단
        price_momentum = df['close'].pct_change(5).iloc[-1] if len(df) >= 6 else 0
        
        # 🚀 레짐별 가변 임계값 계산 (개선사항)
        if len(df) >= 100:
            rsi_mean = df['rsi'].rolling(100).mean().iloc[-1]
        else:
            rsi_mean = 50
        
        rsi_threshold_high = rsi_mean + 5
        rsi_threshold_low = rsi_mean - 5
        
        # 🚀 종합적인 파동 단계 판단 (가변 임계값 적용)
        if macd > macd_signal and rsi > rsi_threshold_high and wave_progress > 0.6:
            if price_momentum > 0.02:  # 강한 상승 모멘텀
                return 'impulse_strong'
            else:
                return 'impulse'
        elif macd < macd_signal and rsi < rsi_threshold_low and wave_progress < 0.4:
            if price_momentum < -0.02:  # 강한 하락 모멘텀
                return 'correction_strong'
            else:
                return 'correction'
        elif abs(macd - macd_signal) < 0.001 or (rsi_threshold_low <= rsi <= rsi_threshold_high and 0.4 <= wave_progress <= 0.6):
            return 'consolidation'
        else:
            # 기본 가격 기반 판단 (기존 로직)
            if df['close'].iloc[-1] > df['close'].iloc[-2] > df['close'].iloc[-3]:
                return 'uptrend'
            elif df['close'].iloc[-1] < df['close'].iloc[-2] < df['close'].iloc[-3]:
                return 'downtrend'
            else:
                return 'sideways'
                
    except Exception as e:
        return 'unknown'

# 패턴 타입 매핑 함수 (통합 로직과 값 체계 일치)
def _map_basic_pattern_to_trend(ptype: str) -> str:
    """기본 패턴 결과를 통합 패턴으로 매핑"""
    if ptype == 'ABC_Correction_Up':
        return 'uptrend'
    if ptype == 'ABC_Correction_Down':
        return 'downtrend'
    if ptype == 'Sideways':
        return 'sideways_consolidation'
    return 'none'

# Three-Wave 패턴 분석 함수
def identify_three_wave_pattern(df):
    if len(df) < 3:
        return 'None'

    a, b, c = df['close'].iloc[-3], df['close'].iloc[-2], df['close'].iloc[-1]

    if a < b > c > a:
        return 'ABC_Correction_Down'
    elif a > b < c < a:
        return 'ABC_Correction_Up'
    else:
        return 'None'

# Sideways 패턴 분석 함수
def identify_sideways_pattern(df, threshold=0.005):
    recent_prices = df['close'].iloc[-10:]
    if len(recent_prices) < 10:
        return 'None'

    price_range = recent_prices.max() - recent_prices.min()
    avg_price = recent_prices.mean()

    return 'Sideways' if price_range / avg_price < threshold else 'None'

# 🚀 33개 핵심 컬럼만 계산하는 최적화된 함수
def add_technical_and_wave_indicators(df, interval: str = None):
    """
    🚀 33개 핵심 컬럼만 계산 - 테이블 스키마와 완전 일치
    """
    if len(df) < 20:
        print(f"    ⚠️ 유효한 row 수 부족 ({len(df)}개) - 기본값 설정")
        # 33개 핵심 컬럼만 기본값으로 설정
        df['rsi'] = np.nan
        df['mfi'] = np.nan
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_middle'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_position'] = np.nan
        df['bb_width'] = np.nan
        df['atr'] = np.nan
        df['ma20'] = np.nan
        df['adx'] = np.nan
        df['volume_ratio'] = np.nan
        df['risk_score'] = 0.5
        df['wave_phase'] = 'unknown'
        df['confidence'] = 0.5
        df['zigzag_direction'] = 0
        df['zigzag_pivot_price'] = np.nan
        df['wave_progress'] = 0.0
        df['pattern_type'] = 'none'
        df['pattern_confidence'] = 0.0
        return df
    
    # 🚀 1단계: 핵심 오실레이터 (2개)
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['mfi'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
    
    # 🚀 2단계: 핵심 트렌드 (2개) - 정규화된 값으로 계산
    macd = MACD(df['close'])
    df['macd'] = macd.macd() / df['close']  # 가격 대비 비율로 정규화
    df['macd_signal'] = macd.macd_signal() / df['close']  # 가격 대비 비율로 정규화
    
    # 🚀 3단계: 핵심 볼린저밴드 (5개) - 정규화된 값으로 계산
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband() / df['close']  # 가격 대비 비율로 정규화
    df['bb_middle'] = bb.bollinger_mavg() / df['close']  # 가격 대비 비율로 정규화
    df['bb_lower'] = bb.bollinger_lband() / df['close']  # 가격 대비 비율로 정규화
    
    # 🚀 볼린저밴드 추가 지표 계산
    df['bb_position'] = ((1.0 - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)).clip(0, 1)  # 정규화 일관성: close/close = 1
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)  # 밴드 폭 (변동성 측정)
    
    # 🚀 4단계: 핵심 추세/변동성 (3개) - 정규화된 값으로 계산
    # ATR 계산 (가격 대비 비율로 정규화)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = pd.Series(np.max(ranges, axis=1))
    df['atr'] = (true_range.rolling(14).mean()) / df['close']  # 가격 대비 비율로 정규화
    
    # MA20 계산 (가격 대비 비율로 정규화)
    df['ma20'] = df['close'].rolling(window=20).mean() / df['close']  # 가격 대비 비율로 정규화
    
    # ADX 계산 (개선된 로직)
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0))
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), -low_diff, 0))
    tr_smooth = true_range.rolling(14).mean()
    plus_dm_smooth = plus_dm.rolling(14).mean()
    minus_dm_smooth = minus_dm.rolling(14).mean()
    
    # 0으로 나누기 방지
    tr_safe = tr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / tr_safe).fillna(0)
    minus_di = 100 * (minus_dm_smooth / tr_safe).fillna(0)
    
    di_sum = plus_di + minus_di
    # 0으로 나누기 방지
    dx = 100 * np.abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
    dx = dx.fillna(0) # 분모가 0인 경우(변동성 없음) 0으로 처리
    
    df['adx'] = np.clip(dx.rolling(14).mean(), 0, 100).fillna(25.0) # 기본값 25
    
    # 🚀 5단계: 핵심 거래량 (1개)
    volume_avg = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (volume_avg + 1e-9)
    
    # 🚀 6단계: 핵심 리스크 (1개) - ATR 기반 리스크 점수 계산
    # 리스크 점수 계산 (기존 로직 유지)
    rsi_filled = df['rsi'].fillna(50)
    rsi_risk = ((rsi_filled - 50).abs() / 50) * 0.3
    macd_filled = df['macd'].fillna(0)
    macd_rolling_avg = macd_filled.rolling(window=20).mean()
    macd_risk = (macd_filled.abs() / (macd_rolling_avg.abs() + 1e-9)) * 0.3
    volume_ratio_filled = df['volume_ratio'].fillna(1)
    volume_risk = (volume_ratio_filled - 1).clip(lower=0) * 0.4
    base_risk_score = (rsi_risk + macd_risk + volume_risk).clip(0, 1)
    
    # 🚀 ATR 기반 expected volatility 가중치 추가 (개선사항)
    df['expected_vol_risk'] = (df['atr'] * volume_ratio_filled).clip(0, 1)
    df['risk_score'] = (0.5 * base_risk_score + 0.5 * df['expected_vol_risk']).clip(0, 1)
    
    # 🚀 7단계: 핵심 파동 (2개)
    df['wave_phase'] = df.apply(lambda x: determine_wave_phase(df.loc[:x.name]) if x.name > 0 else 'unknown', axis=1)
    
    # 신뢰도 계산
    confidence_factors = []
    rsi_confidence = 1 - abs(rsi_filled - 50) / 50
    confidence_factors.append(rsi_confidence * 0.3)
    macd_confidence = 1 - abs(macd_filled - df['macd_signal'].fillna(0)) / (abs(macd_filled) + abs(df['macd_signal'].fillna(0)) + 1e-9)
    confidence_factors.append(macd_confidence * 0.3)
    volatility_confidence = (1 - df['bb_width']).clip(0, 1)  # volatility 제거 방침에 맞춰 bb_width 사용
    confidence_factors.append(volatility_confidence * 0.2)
    volume_confidence = (1 / (1 + abs(volume_ratio_filled - 1))).clip(0, 1)
    confidence_factors.append(volume_confidence * 0.2)
    df['confidence'] = sum(confidence_factors).clip(0, 1)
    
    # 🚀 8단계: 핵심 파동 분석 (3개)
    if interval is None:
        if 'interval' in df.columns:
            interval = df['interval'].iloc[0]
        else:
            interval = '15m'
    
    df = add_zigzag(df, interval)
    df['wave_progress'] = calculate_wave_progress(df)
    
    # 🚀 9단계: 핵심 패턴 분석 (2개)
    df['pattern_type'] = df.apply(lambda x: identify_three_wave_pattern(df.loc[:x.name]) if x.name > 2 else 'none', axis=1)
    df['pattern_type'] = df['pattern_type'].map(_map_basic_pattern_to_trend).fillna('none')  # 통합 로직과 값 체계 일치
    df['pattern_confidence'] = df.apply(lambda x: _calculate_pattern_confidence_fallback(df.loc[:x.name]) if x.name > 0 else 0.0, axis=1)
    
    # 🚀 NULL 값 방지: 시계열 연속성을 고려한 보간법 적용
    # 숫자형 컬럼: 선형 보간법 사용 (시간적 연속성 유지)
    numeric_columns = ['rsi', 'mfi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 
                      'bb_position', 'bb_width', 'atr', 'ma20', 'adx', 'volume_ratio', 'risk_score',
                      'confidence', 'zigzag_direction', 'zigzag_pivot_price', 'wave_progress', 'pattern_confidence', 'structure_score']
    
    for col in numeric_columns:
        if col in df.columns:
            # 선형 보간법으로 시간적 연속성 유지 (시작 부분은 NULL 유지)
            # 데이터 타입을 명시적으로 변환하여 FutureWarning 방지
            df[col] = pd.to_numeric(df[col], errors='coerce').interpolate(method='linear', limit_direction='forward')
            # 끝 부분만 Forward Fill로 처리 (시작 부분은 NULL 그대로 유지)
            df[col] = df[col].ffill()
    
    # 텍스트형 컬럼: 이전 값으로 채우기 (시간적 연속성 유지)
    text_columns = ['pattern_type', 'wave_phase']
    for col in text_columns:
        if col in df.columns:
            # Forward Fill로 시간적 연속성 유지 (시작 부분은 NULL 유지)
            df[col] = df[col].ffill()
    
    # 🚀 10단계: 구조 점수 계산 (개선사항)
    df['structure_score'] = evaluate_fractal_structure(df)
    
    # 🚀 11단계: 모멘텀 지표 (추가)
    # Price Momentum (10봉 전 대비 변화율)
    df['price_momentum'] = df['close'].pct_change(10).fillna(0) * 100 # 퍼센트 단위로 변환
    
    print(f"    ✅ {interval}: 34개 핵심 컬럼 계산 완료 (총 {len(df)}개 row)")
    return df

# 🚀 33개 컬럼에 포함되지 않는 함수들은 제거됨

# 🚀 33개 컬럼에 포함되지 않는 빠른 지표 계산 함수들은 제거됨

# Zigzag 계산 함수 (candles_calculate.py와 동일한 방식)
def add_zigzag(df: pd.DataFrame, interval: str, percent: float = None) -> pd.DataFrame:
    """
    🚀 최적화된 Zigzag 계산 - 방향성 + 전환점 가격 중심 방식 (candles_calculate.py와 통일)
    
    Args:
        df: DataFrame with OHLCV data
        interval: Time interval ('15m', '30m', '240m', '1d', '1w')
        percent: 기존 퍼센티지 파라미터 (호환성을 위해 유지하지만 사용하지 않음)
    
    Returns:
        DataFrame with zigzag_direction and zigzag_pivot_price columns added
    """
    # ✅ 최소 데이터 수 검증
    if len(df) < 10:  # 더 작은 데이터셋에서도 작동하도록 조정
        print("    ⚠️ 유효한 데이터 수 부족 - zigzag 계산 생략")
        df['zigzag_direction'] = 0
        df['zigzag_pivot_price'] = np.nan
        return df
    
    # ✅ 인터벌별 Lookback 캔들 수 설정 (candles_calculate.py와 동일)
    lookback_map = {
        '15m': 3,   # 약 45분간 기준
        '30m': 3,   # 약 1.5시간 기준
        '240m': 2,  # 약 8시간 기준
        '1d': 2,    # 일봉 기준
    }
    lookback = lookback_map.get(interval, 3)  # 기본값 3
    
    
    # ✅ 슬라이딩 윈도우 기반 Zigzag 계산 (방향성 + 전환점 가격)
    close = df['close'].values
    
    # None 값 처리: None을 np.nan으로 변환
    close = np.array([np.nan if x is None else x for x in close])
    
    zz_direction = [0] * len(close)
    zz_pivot_price = [np.nan] * len(close)  # 전환점 가격만 저장
    
    change_count = 0
    
    
    # ✅ 슬라이딩 윈도우 적용 (방향성 + 전환점 가격 계산)
    for i in range(lookback, len(close) - lookback):
        # 현재 캔들을 중심으로 슬라이딩 윈도우 설정
        window_start = i - lookback
        window_end = i + lookback + 1
        window = close[window_start:window_end]
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
        
        # ✅ 고점 전환점 판단 (현재가가 윈도우 내 최고가)
        if center == window_max:
            zz_direction[i] = 1  # 상승 전환점 (고점)
            zz_pivot_price[i] = center  # 전환점 가격 저장
            change_count += 1
        
        # ✅ 저점 전환점 판단 (현재가가 윈도우 내 최저가)
        elif center == window_min:
            zz_direction[i] = -1  # 하락 전환점 (저점)
            zz_pivot_price[i] = center  # 전환점 가격 저장
            change_count += 1
        
        # ✅ 전환점이 아닌 경우
        else:
            zz_direction[i] = 0  # 유지
            zz_pivot_price[i] = np.nan  # 전환점이 아니므로 NaN
    
    # ✅ 초기 경계 값 처리 (lookback 양 끝)
    for i in range(lookback):
        zz_direction[i] = 0
        zz_pivot_price[i] = np.nan
    
    for i in range(len(close) - lookback, len(close)):
        zz_direction[i] = 0
        zz_pivot_price[i] = np.nan
    
    
    # ✅ 전환점 부족 → 최소 1개 이상 필요 (유효성 검증 기준 완화)
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
        
    
    # ✅ zigzag_direction이 모두 0인지 확인
    non_zero_directions = sum(1 for d in zz_direction if d != 0)
    if non_zero_directions == 0:
        # 최소한의 전환점 생성 (첫 번째와 마지막 캔들)
        if len(close) >= 2:
            zz_direction[0] = 1  # 첫 번째를 고점으로
            zz_pivot_price[0] = close[0]
            zz_direction[-1] = -1  # 마지막을 저점으로
            zz_pivot_price[-1] = close[-1]
            change_count = 2
    
    
    # ✅ 결과 저장 (방향성 + 전환점 가격)
    df['zigzag_direction'] = zz_direction
    df['zigzag_pivot_price'] = zz_pivot_price
    
    return df

# 파동 번호 계산 (방향성 중심 방식으로 개선)
def calculate_wave_numbers(df):
    """Elliott Wave 번호 계산 - 방향성 중심 방식"""
    # ✅ 최소 데이터 수 검증
    if len(df) < 10:
        return pd.Series(0, index=df.index)
    
    # ✅ zigzag_direction 데이터 유효성 확인
    if 'zigzag_direction' not in df.columns:
        return pd.Series(0, index=df.index)
    
    zigzag_direction = df['zigzag_direction']
    
    # ✅ zigzag_direction이 모두 0인지 확인
    non_zero_directions = (zigzag_direction != 0).sum()
    if non_zero_directions == 0:
        return pd.Series(0, index=df.index)
    
    wave_numbers = pd.Series(index=df.index, dtype=int)
    wave_count = 0
    
    # ✅ zigzag_direction을 기반으로 파동 번호 계산
    for i in range(len(df)):
        current_direction = zigzag_direction.iloc[i]
        
        # 방향 변화가 있을 때만 파동 번호 증가
        if current_direction != 0:  # +1 (상승) 또는 -1 (하락)
            wave_count += 1
        
        wave_numbers.iloc[i] = wave_count
    
    # ✅ 파동 번호 유효성 확인
    if wave_count == 0:
        return pd.Series(0, index=df.index)
    elif wave_count < 3:  # 최소 3개 파동
        pass
    else:
        pass
        pass
    
    # ✅ NaN 값 방지를 위해 명시적으로 채우기
    wave_numbers = wave_numbers.fillna(0).astype(int).infer_objects(copy=False)
    return wave_numbers

# 파동 진행률 계산 (개선된 방식 - zigzag 실패 시 대체 계산)
def calculate_wave_progress(df):
    """현재 파동의 진행률 계산 - zigzag 실패 시 대체 계산 방식"""
    # ✅ 최소 데이터 수 검증
    if len(df) < 5:  # 더 작은 데이터셋에서도 작동하도록 조정
        return pd.Series(0.5, index=df.index)  # 기본값 0.5로 변경
    
    # ✅ zigzag_direction 데이터 유효성 확인
    if 'zigzag_direction' not in df.columns:
        return _calculate_wave_progress_fallback(df)
    
    zigzag_direction = df['zigzag_direction']
    
    # ✅ zigzag_direction이 모두 0인지 확인
    non_zero_directions = (zigzag_direction != 0).sum()
    if non_zero_directions == 0:
        return _calculate_wave_progress_fallback(df)
    
    wave_progress = pd.Series(index=df.index, dtype=float)
    
    # ✅ 전환점 가격 기반 진행률 계산 (정확한 가격 정보 활용)
    for i in range(1, len(df)):
        current_direction = df['zigzag_direction'].iloc[i]
        
        # ✅ 실제 전환점이 있을 때만 진행률 계산
        if current_direction != 0:
            # 이전 전환점 찾기
            wave_start = i - 1
            while wave_start > 0 and df['zigzag_direction'].iloc[wave_start] == 0:
                wave_start -= 1
            
            if wave_start > 0:
                # 🚀 전환점 가격 사용 (정확한 가격 정보)
                start_price = df['zigzag_pivot_price'].iloc[wave_start]
                current_price = df['close'].iloc[i]
                
                if pd.notna(start_price) and start_price != 0:
                    # 현재 전환점의 방향에 따라 진행률 계산
                    if current_direction == 1:  # 상승 전환점
                        progress = (current_price - start_price) / (df['high'].iloc[i] - start_price + 1e-9)
                    else:  # 하락 전환점 (current_direction == -1)
                        progress = (start_price - current_price) / (start_price - df['low'].iloc[i] + 1e-9)
                    wave_progress.iloc[i] = progress.clip(0, 1)
    
    # ✅ 진행률 유효성 확인 (기준 완화)
    progress_mean = wave_progress.mean()
    if progress_mean == 0:
        return _calculate_wave_progress_fallback(df)
    elif progress_mean < 0.01:  # 평균 0.01 이상 (기준 완화)
        return _calculate_wave_progress_fallback(df)
    else:
        pass
    
    # ✅ NaN 값 방지를 위해 명시적으로 채우기
    wave_progress = wave_progress.fillna(0.5).astype(float).infer_objects(copy=False)  # 기본값 0.5로 변경
    
    # 🚀 지수평활 적용으로 노이즈 감소 (안정화)
    wave_progress = wave_progress.clip(0, 1)
    wave_progress = wave_progress.ewm(span=5, min_periods=1).mean()
    
    return wave_progress

def _calculate_wave_progress_fallback(df):
    """zigzag 실패 시 대체 파동 진행률 계산"""
    try:
        # 🚀 RSI 기반 파동 진행률 계산 (우선순위 1)
        if 'rsi' in df.columns and not df['rsi'].isna().all():
            rsi = df['rsi'].fillna(50)
            # RSI를 0-1 범위로 정규화 (30-70 범위를 0-1로)
            wave_progress = ((rsi - 30) / (70 - 30)).clip(0, 1)
            # 🚀 지수평활 적용으로 노이즈 감소
            wave_progress = wave_progress.ewm(span=5, min_periods=1).mean()
            return wave_progress
        
        # 🚀 가격 기반 파동 진행률 계산 (슬라이딩 윈도우) (우선순위 2)
        elif 'close' in df.columns:
            close = df['close']
            window_size = min(10, len(df) // 2)  # 더 작은 윈도우 크기
            
            wave_progress = pd.Series(index=df.index, dtype=float)
            
            for i in range(window_size, len(df)):
                window = close.iloc[i-window_size:i+1]
                min_price = window.min()
                max_price = window.max()
                current_price = close.iloc[i]
                
                if max_price > min_price:
                    progress = (current_price - min_price) / (max_price - min_price)
                    wave_progress.iloc[i] = progress
                else:
                    wave_progress.iloc[i] = 0.5  # 중립값
            
            # 앞부분 채우기
            wave_progress.iloc[:window_size] = 0.5
            
            # 🚀 지수평활 적용으로 노이즈 감소
            wave_progress = wave_progress.fillna(0.5).ewm(span=5, min_periods=1).mean()
            
            return wave_progress
        
        # 🚀 MACD 기반 파동 진행률 계산 (우선순위 3)
        elif 'macd' in df.columns and not df['macd'].isna().all():
            macd = df['macd'].fillna(0)
            # MACD를 0-1 범위로 정규화 (절대값 기준)
            max_macd = macd.abs().max()
            if max_macd > 0:
                wave_progress = ((macd + max_macd) / (2 * max_macd)).clip(0, 1)
            else:
                wave_progress = pd.Series(0.5, index=df.index)
            # 🚀 지수평활 적용으로 노이즈 감소
            wave_progress = wave_progress.ewm(span=5, min_periods=1).mean()
            return wave_progress
        
        else:
            return pd.Series(0.5, index=df.index)
            
    except Exception as e:
        return pd.Series(0.5, index=df.index)

def _calculate_pattern_confidence_fallback(df):
    """패턴 감지 실패 시 대체 패턴 신뢰도 계산 (구조 기반과 일관성 기반 분리)"""
    try:
        if len(df) == 0:
            return 0.0
        
        # 마지막 행의 데이터로 계산
        i = len(df) - 1
        score = 0.0
        
        # 🚀 RSI 기반 신뢰도 (30-70 범위가 안정적)
        if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[i]):
            rsi = df['rsi'].iloc[i]
            if 30 <= rsi <= 70:
                score += 0.3
            elif 20 <= rsi <= 80:
                score += 0.2
            else:
                score += 0.1
        
        # 🚀 MACD 기반 신뢰도 (신호선과의 차이가 적당할 때)
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[i]
            macd_signal = df['macd_signal'].iloc[i]
            if not pd.isna(macd) and not pd.isna(macd_signal):
                macd_diff = abs(macd - macd_signal)
                if macd_diff < 0.01:
                    score += 0.3
                elif macd_diff < 0.05:
                    score += 0.2
                else:
                    score += 0.1
        
        # 🚀 거래량 기반 신뢰도 (적당한 거래량이 좋음)
        if 'volume_ratio' in df.columns and not pd.isna(df['volume_ratio'].iloc[i]):
            volume_ratio = df['volume_ratio'].iloc[i]
            if 0.5 <= volume_ratio <= 2.0:
                score += 0.2
            elif 0.3 <= volume_ratio <= 3.0:
                score += 0.15
            else:
                score += 0.1
        
        # 🚀 변동성 기반 신뢰도 (bb_width 사용)
        if 'bb_width' in df.columns and not pd.isna(df['bb_width'].iloc[i]):
            bw = df['bb_width'].iloc[i]
            if bw < 0.02:
                score += 0.2
            elif bw < 0.05:
                score += 0.15
            else:
                score += 0.1
        
        # 🚀 구조 기반 신뢰도 추가 (개선사항)
        if 'structure_score' in df.columns and not pd.isna(df['structure_score'].iloc[i]):
            structure_weight = df['structure_score'].iloc[i] * 0.3
        else:
            structure_weight = 0.15  # 기본값
        
        final_conf = min(1.0, (score + structure_weight))
        return final_conf
        
    except Exception as e:
        return 0.5

def _calculate_pattern_quality_fallback(df):
    """패턴 감지 실패 시 대체 패턴 품질 계산"""
    try:
        quality_scores = []
        
        for i in range(len(df)):
            score = 0.0
            
            # 🚀 구조 점수 기반 품질
            if 'structure_score' in df.columns and not pd.isna(df['structure_score'].iloc[i]):
                structure_score = df['structure_score'].iloc[i]
                score += structure_score * 0.4
            
            # 🚀 파동 진행률 기반 품질
            if 'wave_progress' in df.columns and not pd.isna(df['wave_progress'].iloc[i]):
                wave_progress = df['wave_progress'].iloc[i]
                # 진행률이 0.3-0.7 범위일 때 좋은 품질
                if 0.3 <= wave_progress <= 0.7:
                    score += 0.3
                else:
                    score += 0.15
            
            # 🚀 기술지표 일관성 기반 품질
            if 'rsi' in df.columns and 'macd' in df.columns:
                rsi = df['rsi'].iloc[i]
                macd = df['macd'].iloc[i]
                if not pd.isna(rsi) and not pd.isna(macd):
                    # RSI와 MACD가 같은 방향을 가리킬 때 품질 높음
                    rsi_direction = 1 if rsi > 50 else -1
                    macd_direction = 1 if macd > 0 else -1
                    if rsi_direction == macd_direction:
                        score += 0.3
                    else:
                        score += 0.1
            
            quality_scores.append(min(score, 1.0))
        
        return pd.Series(quality_scores, index=df.index)
        
    except Exception as e:
        return pd.Series(0.5, index=df.index)

# ✅ 고급 분석 함수들은 realtime_candles_integrated.py로 이동됨

def _calculate_pattern_type_fallback(df):
    """패턴 감지 실패 시 대체 패턴 타입 계산"""
    try:
        pattern_types = []
        
        for i in range(len(df)):
            # 🚀 기술지표 기반 패턴 타입 판단
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[i]) else 50
            macd = df['macd'].iloc[i] if 'macd' in df.columns and not pd.isna(df['macd'].iloc[i]) else 0
            macd_signal = df['macd_signal'].iloc[i] if 'macd_signal' in df.columns and not pd.isna(df['macd_signal'].iloc[i]) else 0
            
            # 🚀 파동 진행률 기반 판단
            wave_progress = df['wave_progress'].iloc[i] if 'wave_progress' in df.columns and not pd.isna(df['wave_progress'].iloc[i]) else 0.5
            
            # 🚀 가격 모멘텀 기반 판단
            if i >= 5:
                price_momentum = df['close'].pct_change(5).iloc[i]
            else:
                price_momentum = 0
            
            # 🚀 종합적인 패턴 타입 판단
            if macd > macd_signal and rsi > 60 and wave_progress > 0.7:
                if price_momentum > 0.03:  # 강한 상승
                    pattern_type = 'strong_uptrend'
                else:
                    pattern_type = 'uptrend'
            elif macd < macd_signal and rsi < 40 and wave_progress < 0.3:
                if price_momentum < -0.03:  # 강한 하락
                    pattern_type = 'strong_downtrend'
                else:
                    pattern_type = 'downtrend'
            elif abs(macd - macd_signal) < 0.001 and 40 <= rsi <= 60:
                pattern_type = 'sideways_consolidation'
            elif rsi > 70:
                pattern_type = 'overbought'
            elif rsi < 30:
                pattern_type = 'oversold'
            else:
                # 기본 가격 기반 판단
                if i >= 3:
                    if df['close'].iloc[i] > df['close'].iloc[i-1] > df['close'].iloc[i-2]:
                        pattern_type = 'ascending'
                    elif df['close'].iloc[i] < df['close'].iloc[i-1] < df['close'].iloc[i-2]:
                        pattern_type = 'descending'
                    else:
                        pattern_type = 'sideways'
                else:
                    pattern_type = 'unknown'
            
            pattern_types.append(pattern_type)
        
        return pd.Series(pattern_types, index=df.index)
        
    except Exception as e:
        return pd.Series('unknown', index=df.index)

# 🆕 프랙탈 기반 특성 계산 함수들 (반응형 학습과 동일)
def compute_wave_step(df: pd.DataFrame) -> pd.Series:
    """
    🚀 개선된 파동 단계 계산 - 반응형 학습과 동일한 방식 (0-10)
    
    Args:
        df: DataFrame with wave data
    
    Returns:
        Series with wave step information (0-10 범위)
    """
    wave_steps = []
    
    for i in range(len(df)):
        # wave_number는 30개 컬럼에서 제거됨
        wave_progress = df['wave_progress'].iloc[i]
        zigzag_direction = df['zigzag_direction'].iloc[i]
        
        # None 값 처리 강화
        if pd.isna(wave_progress) or wave_progress is None:
            wave_steps.append(0)
            continue
        
        # wave_progress가 숫자인지 확인
        try:
            wave_progress = float(wave_progress)
        except (ValueError, TypeError):
            wave_steps.append(0)
            continue
        
        # 🚀 파동 진행률에 따른 단계 결정 (더 세밀하게)
        if wave_progress < 0.1:
            wave_step = 1  # 시작 단계
        elif wave_progress < 0.2:
            wave_step = 2  # 초기 진행
        elif wave_progress < 0.3:
            wave_step = 3  # 초기-중간
        elif wave_progress < 0.4:
            wave_step = 4  # 중간 단계
        elif wave_progress < 0.5:
            wave_step = 5  # 중간-후기
        elif wave_progress < 0.6:
            wave_step = 6  # 후기 진행
        elif wave_progress < 0.7:
            wave_step = 7  # 후기-완성
        elif wave_progress < 0.8:
            wave_step = 8  # 완성 단계
        elif wave_progress < 0.9:
            wave_step = 9  # 완성-종료
        else:
            wave_step = 10  # 종료 단계
        
        # 🚀 방향에 따른 조정 (더 세밀하게)
        if zigzag_direction == -1:  # 하락 파동
            wave_step = 11 - wave_step  # 역순으로 조정
        
        # 🚀 추가 보정 요소들
        # 파동 번호 기반 보정 (wave_number는 30개 컬럼에서 제거됨 - 기본값 사용)
        wave_num = 1  # 기본값
        if wave_num > 5:
            wave_step = min(wave_step + 1, 10)  # 높은 파동 번호는 +1
        
        # 패턴 신뢰도 기반 보정
        if 'pattern_confidence' in df.columns:
            pattern_conf = df['pattern_confidence'].iloc[i]
            if pd.notna(pattern_conf) and pattern_conf > 0.7:
                wave_step = min(wave_step + 1, 10)
        
        # RSI 기반 보정
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[i]
            if pd.notna(rsi):
                if rsi > 75:  # 강한 과매수
                    wave_step = min(wave_step + 1, 10)
                elif rsi < 25:  # 강한 과매도
                    wave_step = max(wave_step - 1, 0)
        
        # 거래량 기반 보정
        if 'volume_ratio' in df.columns:
            volume_ratio = df['volume_ratio'].iloc[i]
            if pd.notna(volume_ratio) and volume_ratio > 2.5:  # 매우 높은 거래량
                wave_step = min(wave_step + 1, 10)
        
        wave_steps.append(wave_step)
    
    return pd.Series(wave_steps, index=df.index)

def evaluate_fractal_structure(df: pd.DataFrame) -> pd.Series:
    """
    🚀 개선된 프랙탈 구조 평가 - 반응형 학습과 동일한 방식 (0.0 ~ 1.0, 더 넓은 분포)
    
    Args:
        df: DataFrame with wave and pattern data
    
    Returns:
        Series with structure scores (0.0 ~ 1.0, 더 넓은 분포)
    """
    structure_scores = []
    
    for i in range(len(df)):
        # 기본 점수 초기화
        base_score = 0.0
        
        # 🚀 1. 파동 연속성 점수 (0.25) - 더 세밀한 계산
        # wave_number는 30개 컬럼에서 제거됨 - 기본값 사용
        wave_num = 1  # 기본값
        if wave_num > 0:
            # 연속된 파동이 많을수록 높은 점수 (더 세밀하게)
            if wave_num <= 3:
                wave_continuity = wave_num / 3.0
            elif wave_num <= 7:
                wave_continuity = 0.5 + (wave_num - 3) / 8.0
            else:
                wave_continuity = 0.75 + (wave_num - 7) / 20.0
            wave_continuity = min(wave_continuity, 1.0)
            base_score += 0.25 * wave_continuity
        
        # 🚀 2. 파동 진행률 점수 (0.2) - 더 세밀한 계산
        wave_progress = df['wave_progress'].iloc[i]
        if pd.notna(wave_progress) and wave_progress is not None:
            try:
                wave_progress = float(wave_progress)
                # 진행률별 세밀한 점수 계산
                if wave_progress < 0.2:
                    progress_score = wave_progress * 2.5  # 0~0.5
                elif wave_progress < 0.4:
                    progress_score = 0.5 + (wave_progress - 0.2) * 1.25  # 0.5~0.75
                elif wave_progress < 0.6:
                    progress_score = 0.75 + (wave_progress - 0.4) * 0.625  # 0.75~0.875
                elif wave_progress < 0.8:
                    progress_score = 0.875 + (wave_progress - 0.6) * 0.625  # 0.875~1.0
                else:
                    progress_score = 1.0 - (wave_progress - 0.8) * 2.5  # 1.0~0.5
                progress_score = max(0.0, min(1.0, progress_score))
                base_score += 0.2 * progress_score
            except (ValueError, TypeError):
                pass  # 변환 실패 시 점수 추가하지 않음
        
        # 🚀 3. 패턴 신뢰도 점수 (0.25) - 더 세밀한 계산
        if 'pattern_confidence' in df.columns:
            pattern_conf = df['pattern_confidence'].iloc[i]
            if pd.notna(pattern_conf):
                # 신뢰도별 세밀한 점수 계산
                if pattern_conf < 0.3:
                    pattern_score = pattern_conf * 1.5  # 0~0.45
                elif pattern_conf < 0.6:
                    pattern_score = 0.45 + (pattern_conf - 0.3) * 0.5  # 0.45~0.6
                elif pattern_conf < 0.8:
                    pattern_score = 0.6 + (pattern_conf - 0.6) * 1.0  # 0.6~0.8
                else:
                    pattern_score = 0.8 + (pattern_conf - 0.8) * 1.0  # 0.8~1.0
                base_score += 0.25 * pattern_score
        
        # 🚀 4. 거래량 일관성 점수 (0.15) - 더 세밀한 계산
        if 'volume_ratio' in df.columns:
            volume_ratio = df['volume_ratio'].iloc[i]
            if pd.notna(volume_ratio):
                # 거래량별 세밀한 점수 계산
                if volume_ratio < 0.5:
                    volume_score = volume_ratio * 1.0  # 0~0.5
                elif volume_ratio < 1.0:
                    volume_score = 0.5 + (volume_ratio - 0.5) * 0.5  # 0.5~0.75
                elif volume_ratio < 2.0:
                    volume_score = 0.75 + (volume_ratio - 1.0) * 0.25  # 0.75~1.0
                else:
                    volume_score = 1.0 - (volume_ratio - 2.0) * 0.1  # 1.0~0.8
                volume_score = max(0.0, min(1.0, volume_score))
                base_score += 0.15 * volume_score
        
        # 🚀 5. RSI 기반 점수 (0.15) - 새로운 요소
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[i]
            if pd.notna(rsi):
                # RSI별 세밀한 점수 계산
                if rsi < 20:
                    rsi_score = 0.2  # 매우 과매도
                elif rsi < 30:
                    rsi_score = 0.4  # 과매도
                elif rsi < 45:
                    rsi_score = 0.6  # 약한 하락
                elif rsi < 55:
                    rsi_score = 0.8  # 중립
                elif rsi < 70:
                    rsi_score = 0.6  # 약한 상승
                elif rsi < 80:
                    rsi_score = 0.4  # 과매수
                else:
                    rsi_score = 0.2  # 매우 과매수
                base_score += 0.15 * rsi_score
        
        structure_scores.append(min(1.0, base_score))
    
    return pd.Series(structure_scores, index=df.index)

def classify_pattern_structure(df: pd.DataFrame) -> pd.Series:
    """
    🚀 개선된 패턴 분류 함수 - 반응형 학습과 동일한 다양한 패턴 클래스 생성
    
    Args:
        df: DataFrame with pattern data
    
    Returns:
        Series with pattern class codes
    """
    pattern_classes = []
    
    for i in range(len(df)):
        # 기본 클래스
        pattern_class = 'unknown'
        
        # 패턴 방향 확인
        pattern_direction = df.get('pattern_direction', pd.Series(['neutral']*len(df))).iloc[i]
        wave_phase = df.get('wave_phase', pd.Series(['unknown']*len(df))).iloc[i]
        
        # 파동 단계 확인
        wave_step = 0
        if 'wave_step' in df.columns:
            wave_step_val = df['wave_step'].iloc[i]
            if pd.notna(wave_step_val) and wave_step_val is not None:
                try:
                    wave_step = int(wave_step_val)
                except (ValueError, TypeError):
                    wave_step = 0
        
        # 🚀 추가 패턴 정보 확인
        pattern_type = df.get('pattern_type', pd.Series(['none']*len(df))).iloc[i]
        pattern_confidence = df.get('pattern_confidence', pd.Series([0.0]*len(df))).iloc[i]
        
        # pattern_confidence None 값 처리
        if pd.isna(pattern_confidence) or pattern_confidence is None:
            pattern_confidence = 0.0
        else:
            try:
                pattern_confidence = float(pattern_confidence)
            except (ValueError, TypeError):
                pattern_confidence = 0.0
        
        # 🚀 RSI 위치 확인 (추가 다양성)
        rsi_position = df.get('rsi_position', pd.Series(['neutral']*len(df))).iloc[i]
        
        # 🚀 MACD 크로스 확인 (추가 다양성)
        macd_cross = df.get('macd_cross', pd.Series(['no_cross']*len(df))).iloc[i]
        
        # 🚀 볼린저밴드 터치 확인 (추가 다양성)
        bollinger_touch = df.get('bollinger_touch', pd.Series(['middle']*len(df))).iloc[i]
        
        # 🚀 개선된 패턴 클래스 결정 (더 세밀한 분류)
        if pattern_direction == 'bullish':
            if wave_phase == 'impulse':
                if wave_step >= 4:
                    # 🚀 후기 임펄스에서 세부 분류
                    if pattern_confidence > 0.7:
                        pattern_class = 'bullish_impulse_late_strong'
                    elif rsi_position == 'overbought':
                        pattern_class = 'bullish_impulse_late_overbought'
                    else:
                        pattern_class = 'bullish_impulse_late'
                elif wave_step >= 2:
                    # 🚀 중기 임펄스에서 세부 분류
                    if macd_cross == 'bullish_cross':
                        pattern_class = 'bullish_impulse_mid_macd_cross'
                    elif bollinger_touch == 'upper_touch':
                        pattern_class = 'bullish_impulse_mid_bb_upper'
                    else:
                        pattern_class = 'bullish_impulse_mid'
                else:
                    # 🚀 초기 임펄스에서 세부 분류
                    if rsi_position == 'oversold':
                        pattern_class = 'bullish_impulse_early_oversold'
                    elif pattern_type in ['ascending_triangle', 'uptrend']:
                        pattern_class = 'bullish_impulse_early_trend'
                    else:
                        pattern_class = 'bullish_impulse_early'
            elif wave_phase == 'correction':
                # 🚀 보정 파동에서 세부 분류
                if pattern_type == 'ascending_triangle':
                    pattern_class = 'bullish_correction_triangle'
                elif rsi_position == 'oversold':
                    pattern_class = 'bullish_correction_oversold'
                else:
                    pattern_class = 'bullish_correction'
            else:
                # 🚀 통합에서 세부 분류
                if pattern_type == 'sideways':
                    pattern_class = 'bullish_consolidation_sideways'
                else:
                    pattern_class = 'bullish_consolidation'
        elif pattern_direction == 'bearish':
            if wave_phase == 'impulse':
                if wave_step >= 4:
                    # 🚀 후기 임펄스에서 세부 분류
                    if pattern_confidence > 0.7:
                        pattern_class = 'bearish_impulse_late_strong'
                    elif rsi_position == 'oversold':
                        pattern_class = 'bearish_impulse_late_oversold'
                    else:
                        pattern_class = 'bearish_impulse_late'
                elif wave_step >= 2:
                    # 🚀 중기 임펄스에서 세부 분류
                    if macd_cross == 'bearish_cross':
                        pattern_class = 'bearish_impulse_mid_macd_cross'
                    elif bollinger_touch == 'lower_touch':
                        pattern_class = 'bearish_impulse_mid_bb_lower'
                    else:
                        pattern_class = 'bearish_impulse_mid'
                else:
                    # 🚀 초기 임펄스에서 세부 분류
                    if rsi_position == 'overbought':
                        pattern_class = 'bearish_impulse_early_overbought'
                    elif pattern_type in ['descending_triangle', 'downtrend']:
                        pattern_class = 'bearish_impulse_early_trend'
                    else:
                        pattern_class = 'bearish_impulse_early'
            elif wave_phase == 'correction':
                # 🚀 보정 파동에서 세부 분류
                if pattern_type == 'descending_triangle':
                    pattern_class = 'bearish_correction_triangle'
                elif rsi_position == 'overbought':
                    pattern_class = 'bearish_correction_overbought'
                else:
                    pattern_class = 'bearish_correction'
            else:
                # 🚀 통합에서 세부 분류
                if pattern_type == 'sideways':
                    pattern_class = 'bearish_consolidation_sideways'
                else:
                    pattern_class = 'bearish_consolidation'
        else:  # neutral
            if wave_phase == 'consolidation':
                # 🚀 횡보에서 세부 분류
                if pattern_type == 'sideways':
                    pattern_class = 'sideways_consolidation'
                elif rsi_position == 'neutral':
                    pattern_class = 'sideways_consolidation_neutral'
                else:
                    pattern_class = 'sideways_consolidation'
            else:
                # 🚀 알 수 없는 경우에서도 세부 분류
                if pattern_confidence < 0.3:
                    pattern_class = 'sideways_unknown_low_confidence'
                else:
                    pattern_class = 'sideways_unknown'
        
        pattern_classes.append(pattern_class)
    return pd.Series(pattern_classes, index=df.index)

# 피봇 포인트 계산 (방향성 중심 방식으로 개선)
def calculate_pivot_points(df, interval):
    """피봇 포인트 계산 - 방향성 중심 방식"""
    if len(df) < 5:
        return pd.Series(0, index=df.index)
    
    pivot_points = pd.Series(0, index=df.index)
    
    # zigzag_direction이 있는 경우 방향성 기반으로 피봇 포인트 계산
    if 'zigzag_direction' in df.columns:
        zigzag_direction = df['zigzag_direction']
        
        # zigzag_direction이 유효한 경우에만 피봇 포인트 설정
        valid_pivots = (zigzag_direction != 0)
        pivot_points[valid_pivots] = 1
    
    return pivot_points

# 🚀 파동 분석 관련 함수들 추가
def calculate_wave_characteristics(df):
    """파동 특성 계산"""
    # ATR 기반 변동성은 이미 bb_width로 대체됨 (volatility 컬럼 제거)
    
    # RSI 기반 파동 진행도 계산
    if 'rsi' in df.columns and not df['rsi'].isna().all():
        rsi_normalized = (df['rsi'] - 30) / (70 - 30)  # 30-70 범위를 0-1로 변환
        df['wave_progress'] = rsi_normalized.clip(0, 1)
    else:
        df['wave_progress'] = (df['close'] - df['close'].rolling(window=20).min()) / \
                             (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        df['wave_progress'] = df['wave_progress'].fillna(0.5)
    
    # MACD 기반 파동 단계 결정
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_filled = df['macd'].fillna(0)
        macd_signal_filled = df['macd_signal'].fillna(0)
        macd_diff = macd_filled - macd_signal_filled
        df['wave_phase'] = 'unknown'
        
        # wave_progress None 값 처리 강화
        wave_progress_filled = df['wave_progress'].fillna(0.5)
        
        # MACD 크로스오버 기반 파동 단계 판단
        df.loc[(macd_diff > 0) & (wave_progress_filled > 0.6), 'wave_phase'] = 'impulse'
        df.loc[(macd_diff < 0) & (wave_progress_filled < 0.4), 'wave_phase'] = 'correction'
        df.loc[(abs(macd_diff) < 0.001) | ((wave_progress_filled >= 0.4) & (wave_progress_filled <= 0.6)), 'wave_phase'] = 'consolidation'
    else:
        df['wave_phase'] = 'unknown'
        df.loc[df['wave_progress'] > 0.8, 'wave_phase'] = 'impulse'
        df.loc[df['wave_progress'] < 0.2, 'wave_phase'] = 'correction'
        df.loc[(df['wave_progress'] >= 0.2) & (df['wave_progress'] <= 0.8), 'wave_phase'] = 'consolidation'
    
    # 통합 신뢰도 계산
    confidence_factors = []
    
    # RSI 기반 신뢰도
    if 'rsi' in df.columns:
        rsi_filled = df['rsi'].fillna(50)
        rsi_confidence = 1 - abs(rsi_filled - 50) / 50  # RSI가 50에 가까울수록 신뢰도 높음
        confidence_factors.append(rsi_confidence * 0.3)
    
    # MACD 기반 신뢰도
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_filled = df['macd'].fillna(0)
        macd_signal_filled = df['macd_signal'].fillna(0)
        macd_confidence = 1 - abs(macd_filled - macd_signal_filled) / (abs(macd_filled) + abs(macd_signal_filled) + 1e-9)
        confidence_factors.append(macd_confidence * 0.3)
    
    # 변동성 기반 신뢰도
    volatility_confidence = (1 - df['bb_width']).clip(0, 1)  # volatility 제거 방침에 맞춰 bb_width 사용
    confidence_factors.append(volatility_confidence * 0.2)
    
    # 거래량 기반 신뢰도
    if 'volume_ratio' in df.columns:
        volume_ratio_filled = df['volume_ratio'].fillna(1)
        volume_confidence = (1 / (1 + abs(volume_ratio_filled - 1))).clip(0, 1)
        confidence_factors.append(volume_confidence * 0.2)
    
    # 통합 신뢰도 계산
    if confidence_factors:
        df['confidence'] = sum(confidence_factors)
    else:
        df['confidence'] = 0.5
    
    df['confidence'] = df['confidence'].clip(0, 1)
    
    return df

def flow_level_light(df, interval):
    """Flow Level 계산"""
    window = 100  # 기본 윈도우
    if len(df) < window:
        return "Neutral"

    recent = df.iloc[-window:]
    
    # RSI 기반 추세 판단
    if 'rsi' in recent.columns and not recent['rsi'].isna().all():
        rsi = recent['rsi'].iloc[-1]
        rsi_ema = recent['rsi_ema'].iloc[-1] if 'rsi_ema' in recent.columns else rsi
    else:
        rsi = 50
        rsi_ema = 50
    
    # None 값 처리
    if rsi is None:
        rsi = 50
    if rsi_ema is None:
        rsi_ema = 50
    
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
    
    # MACD 기반 모멘텀 확인
    if 'macd' in recent.columns and not recent['macd'].isna().all():
        macd = recent['macd'].iloc[-1]
        macd_signal = recent['macd_signal'].iloc[-1] if 'macd_signal' in recent.columns else 0
    else:
        macd = 0
        macd_signal = 0
    
    # None 값 처리
    if macd is None:
        macd = 0
    if macd_signal is None:
        macd_signal = 0
    
    if macd > macd_signal and macd > 0:
        momentum = "bullish"
    elif macd < macd_signal and macd < 0:
        momentum = "bearish"
    else:
        momentum = "neutral"
    
    # 거래량 기반 강도 판단
    if 'volume_ratio' in recent.columns and not recent['volume_ratio'].isna().all():
        volume_ratio = recent['volume_ratio'].iloc[-1]
    else:
        volume_ratio = 1.0
    
    # None 값 처리
    if volume_ratio is None:
        volume_ratio = 1.0
    
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

# 🚀 패턴 감지용 피봇 포인트 계산 함수
def calculate_pattern_pivot_points(df, interval):
    """패턴 감지용 피봇 포인트 계산"""
    df = df.copy()
    threshold = 0.03  # 기본 임계값
    
    # 고점/저점 계산
    df['pivot_high'] = 0
    df['pivot_low'] = 0
    df['pivot_strength'] = 0.0
    
    for i in range(2, len(df)-2):
        # 고점 피봇
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i+2]):
            
            # ATR 기반 피봇 강도 계산
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]):
                atr = df['atr'].iloc[i]
                close = df['close'].iloc[i]
                price_change = atr / close if close > 0 else 0
            else:
                price_change = (df['high'].iloc[i] - df['low'].iloc[i-2:i+3].min()) / df['low'].iloc[i-2:i+3].min()
            
            if price_change >= threshold:
                df.loc[df.index[i], 'pivot_high'] = 1
                df.loc[df.index[i], 'pivot_strength'] = price_change
            
        # 저점 피봇
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            
            # ATR 기반 피봇 강도 계산
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]):
                atr = df['atr'].iloc[i]
                close = df['close'].iloc[i]
                price_change = atr / close if close > 0 else 0
            else:
                price_change = (df['high'].iloc[i-2:i+3].max() - df['low'].iloc[i]) / df['low'].iloc[i]
            
            if price_change >= threshold:
                df.loc[df.index[i], 'pivot_low'] = 1
                df.loc[df.index[i], 'pivot_strength'] = price_change
    
    return df

def detect_chart_patterns(df, interval):
    """차트 패턴 감지"""
    window_size = 20  # 기본 윈도우 크기
    
    if len(df) < window_size:
        return []
    
    # 피봇 포인트 계산
    df = calculate_pattern_pivot_points(df, interval)
    patterns = []
    
    # 모든 패턴 감지 조건을 한 번에 계산
    for i in range(len(df) - window_size + 1):
        window_df = df.iloc[i:i + window_size]
        
        # 헤드앤숄더 패턴 감지
        highs = window_df[window_df['pivot_high'] == 1]['high'].values
        if len(highs) >= 3:
            left_shoulder, head, right_shoulder = highs[:3]
            # None 값 처리
            left_shoulder = left_shoulder if left_shoulder is not None else 0
            right_shoulder = right_shoulder if right_shoulder is not None else 0
            if (head > left_shoulder and head > right_shoulder and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                
                # 안전한 패턴 품질 계산
                pivot_strengths = window_df[window_df['pivot_high'] == 1]['pivot_strength'].iloc[:3]
                pivot_volumes = window_df[window_df['pivot_high'] == 1]['volume'].iloc[:3]
                volume_mean = window_df['volume'].mean()
                
                if len(pivot_strengths) > 0 and len(pivot_volumes) > 0 and volume_mean > 0:
                    pattern_quality = (
                        min(pivot_strengths) * 0.4 +
                        min(pivot_volumes / volume_mean) * 0.3 +
                        (1 - window_df['close'].pct_change().std()) * 0.3
                    )
                else:
                    pattern_quality = 0.5  # 기본값
                
                # 안전한 패턴 메트릭 계산
                if len(pivot_volumes) > 0 and volume_mean > 0:
                    pattern_volume_ratio = min(pivot_volumes / volume_mean)
                else:
                    pattern_volume_ratio = 1.0
                
                if len(pivot_strengths) > 0:
                    pattern_pivot_strength = min(pivot_strengths)
                else:
                    pattern_pivot_strength = 0.0
                
                patterns.append({
                    'pattern_type': 'head_and_shoulders',
                    'pattern_confidence': pattern_quality,
                    'pattern_direction': 'bearish',
                    'pattern_start_idx': window_df.index[0],
                    'pattern_end_idx': window_df.index[-1],
                    'pattern_volume_ratio': pattern_volume_ratio,
                    'pattern_pivot_strength': pattern_pivot_strength,
                })
        
        # 더블 탑/바텀 패턴 감지
        if len(highs) >= 2:
            high1 = highs[0] if highs[0] is not None else 0
            high2 = highs[1] if highs[1] is not None else 0
            if high1 > 0 and abs(high1 - high2) / high1 < 0.1:
                # 안전한 패턴 품질 계산
                pivot_strengths = window_df[window_df['pivot_high'] == 1]['pivot_strength'].iloc[:2]
                pivot_volumes = window_df[window_df['pivot_high'] == 1]['volume'].iloc[:2]
                volume_mean = window_df['volume'].mean()
                
                if len(pivot_strengths) > 0 and len(pivot_volumes) > 0 and volume_mean > 0:
                    pattern_quality = (
                        min(pivot_strengths) * 0.4 +
                        min(pivot_volumes / volume_mean) * 0.3 +
                        (1 - window_df['close'].pct_change().std()) * 0.3
                    )
                else:
                    pattern_quality = 0.5  # 기본값
                
                # 안전한 패턴 메트릭 계산
                if len(pivot_volumes) > 0 and volume_mean > 0:
                    pattern_volume_ratio = min(pivot_volumes / volume_mean)
                else:
                    pattern_volume_ratio = 1.0
                
                if len(pivot_strengths) > 0:
                    pattern_pivot_strength = min(pivot_strengths)
                else:
                    pattern_pivot_strength = 0.0
                
                patterns.append({
                    'pattern_type': 'double_top',
                    'pattern_confidence': pattern_quality,
                    'pattern_direction': 'bearish',
                    'pattern_start_idx': window_df.index[0],
                    'pattern_end_idx': window_df.index[-1],
                    'pattern_volume_ratio': pattern_volume_ratio,
                    'pattern_pivot_strength': pattern_pivot_strength,
                })
    
    return patterns

# interval 별 데이터 처리 - 성능 최적화
def process_interval_data(interval, table_name):
    conn = sqlite3.connect(DB_PATH)
    
    # 🚀 SQLite 성능 최적화 설정
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=50000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=536870912")  # 512MB

    essential_columns = [
        # 🏷️ 기본 식별자 (3개) - symbol로 변경
        ('symbol', 'TEXT'), ('interval', 'TEXT'), ('timestamp', 'INTEGER'),
        # 💰 기본 OHLCV (4개)
        ('open', 'REAL'), ('high', 'REAL'), ('low', 'REAL'), ('close', 'REAL'), ('volume', 'REAL'),
        # 📉 핵심 오실레이터 (2개)
        ('rsi', 'REAL'), ('mfi', 'REAL'),
        # 📊 핵심 트렌드 (2개)
        ('macd', 'REAL'), ('macd_signal', 'REAL'),
        # 🌐 핵심 볼린저밴드 (5개)
        ('bb_upper', 'REAL'), ('bb_middle', 'REAL'), ('bb_lower', 'REAL'), ('bb_position', 'REAL'), ('bb_width', 'REAL'),
        # 📈 핵심 추세/변동성 (3개)
        ('atr', 'REAL'), ('ma20', 'REAL'), ('adx', 'REAL'),
        # 📊 핵심 거래량 (1개)
        ('volume_ratio', 'REAL'),
        # ⚠️ 핵심 리스크 (1개)
        ('risk_score', 'REAL'),
        # 🧠 핵심 파동 (2개)
        ('wave_phase', 'TEXT'), ('confidence', 'REAL'),
        # 🔄 핵심 파동 분석 (3개)
        ('zigzag_direction', 'REAL'), ('zigzag_pivot_price', 'REAL'), ('wave_progress', 'REAL'),
        # 🎯 핵심 패턴 분석 (2개)
        ('pattern_type', 'TEXT'), ('pattern_confidence', 'REAL'),
        # 🧠 핵심 통합 분석 (3개)
        ('volatility_level', 'TEXT'), ('risk_level', 'TEXT'), ('integrated_direction', 'TEXT'),
        # 🚀 구조 점수 (개선사항)
        ('structure_score', 'REAL'),
        # 🚀 심리도 분석 (2개)
        ('sentiment', 'REAL'), ('sentiment_label', 'TEXT')
    ]

    # 🚀 배치로 컬럼 추가 (성능 개선)
    for col, col_type in essential_columns:
        ensure_column_exists(conn, table_name, col, col_type)

    # 🚀 종목 목록을 배치로 가져오기 (symbol 컬럼 사용)
    try:
        symbols = pd.read_sql(f'SELECT DISTINCT symbol FROM {table_name} WHERE interval = ?', conn, params=(interval,))['symbol'].tolist()
    except KeyError:
        # 하위 호환성: symbol 컬럼이 없으면 coin 컬럼 시도
        print(f"⚠️ 'symbol' 컬럼 없음, 'coin' 컬럼 시도...")
        symbols = pd.read_sql(f'SELECT DISTINCT coin FROM {table_name} WHERE interval = ?', conn, params=(interval,))['coin'].tolist()
    
    # 🚀 배치 처리 크기 설정
    batch_size = 5
    total_symbols = len(symbols)
    
    print(f"[{datetime.now()}] 🚀 {interval} 처리 시작: {total_symbols}개 종목, 배치 크기: {batch_size}")

    for i in range(0, total_symbols, batch_size):
        batch_symbols = symbols[i:i + batch_size]
        print(f"[{datetime.now()}] 🔄 배치 {i//batch_size + 1}/{(total_symbols + batch_size - 1)//batch_size}: {len(batch_symbols)}개 종목 처리 중...")
        
        for symbol in batch_symbols:
            # 🚀 [KRX 최적화] 무조건 전체 데이터 로드 및 재계산 (Full Recalculation)
            # 증분 계산은 앞부분 지표(MA, RSI 등)에 오차를 누적시킬 수 있으므로,
            # 데이터 정합성을 위해 매번 전체 데이터를 읽어와서 처음부터 다시 계산함.
            
            # 테이블 정보 확인하여 컬럼명 결정
            col_name = 'symbol'
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            cols = [c[1] for c in cursor.fetchall()]
            if 'symbol' not in cols and 'coin' in cols:
                col_name = 'coin'

            # 전체 데이터 로드
            # print(f"[{datetime.now()}] 🔄 {symbol}-{interval}: 전체 데이터 로드 (Full Recalculation)")
            df = pd.read_sql(f'''
                SELECT * FROM {table_name}
                WHERE {col_name}=? AND interval=?
                ORDER BY timestamp ASC
            ''', conn, params=(symbol, interval)).reset_index(drop=True)
            
            # 데이터프레임 컬럼 통일 (coin -> symbol)
            if 'coin' in df.columns and 'symbol' not in df.columns:
                df.rename(columns={'coin': 'symbol'}, inplace=True)

            if df.empty or len(df) < 20:
                continue
    
            # 🚀 2단계: 무조건 계산 진행 (지표 유무 확인 로직 제거)
            # has_indicators 체크 로직 삭제 -> 항상 재계산
            
            new_data_mask = pd.Series([True] * len(df), index=df.index) # 모든 행을 업데이트 대상으로 설정
            
            # 🚀 데이터 검증
            if not validate_data(df):
                continue
            
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # 🚀 누락값 처리
            df = handle_missing_values(df)
        
            # 🚀 31개 핵심 컬럼만 계산
            df = add_technical_and_wave_indicators(df, interval)

            # 🚀 3단계: 전체 데이터 저장 (덮어쓰기)
            # 기존에는 new_data_mask로 필터링했으나, 이제는 전체를 저장함
            save_success = save_technical_indicators_immediate(df, symbol, interval)
                
            if save_success:
                pass
            else:
                pass
    conn.close()

# 🚀 candles_calculate.py의 성공적인 저장 함수 추가
def save_technical_indicators_immediate(df: pd.DataFrame, symbol: str, interval: str) -> bool:
    """🚀 계산 완료 즉시 저장 - candles_calculate.py의 성공적인 방식 적용"""
    try:
        if df.empty:
            return False
        
        
        # 📌 소숫점 자리수 통일 적용 (33개 핵심 컬럼) - 4자리로 통일
        rounding_map = {
            # 오실레이터 (0~100 범위, 정규화된 지표)
            'rsi': 4, 'mfi': 4, 'adx': 4,
            # 트렌드 지표
            'macd': 4, 'macd_signal': 4,
            # 볼린저밴드 (가격 기반 지표 - 코인 가격 0.0001원 단위 고려)
            'bb_upper': 4, 'bb_middle': 4, 'bb_lower': 4, 'bb_position': 4, 'bb_width': 4,
            # 추세/변동성 지표
            'atr': 4, 'ma20': 4,
            # 거래량 지표
            'volume_ratio': 4,
            # 리스크 지표 (0~1 사이의 값)
            'risk_score': 4,
            # Zigzag 관련 지표 (가격 기반 - 코인 가격 0.0001원 단위 고려)
            'zigzag_direction': 4, 'zigzag_pivot_price': 4,
            # 파동 관련 지표
            'wave_progress': 4,
            # 신뢰도 지표
            'confidence': 4, 'pattern_confidence': 4,
            # 구조 점수 지표 (0~1 사이의 값)
            'structure_score': 4
            # ✅ integrated 컬럼들(sentiment 등)은 integrated 파일에서 처리
        }

        # 반올림 적용 (존재하는 컬럼만)
        for col, digits in rounding_map.items():
            if col in df.columns:
                df[col] = df[col].round(digits)
        
        # 🚀 33개 핵심 컬럼만 저장 (테이블 스키마와 완전 일치)
        technical_columns = [
            # 💰 기본 OHLCV (5개)
            'open', 'high', 'low', 'close', 'volume',
            # 📉 핵심 오실레이터 (2개)
            'rsi', 'mfi',
            # 📊 핵심 트렌드 (2개)
            'macd', 'macd_signal',
            # 🌐 핵심 볼린저밴드 (5개)
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
            # 📈 핵심 추세/변동성 (3개)
            'atr', 'ma20', 'adx',
            # 📊 핵심 거래량 (1개)
            'volume_ratio',
            # ⚠️ 핵심 리스크 (1개)
            'risk_score',
            # 🧠 핵심 파동 (2개)
            'wave_phase', 'confidence',
            # 🔄 핵심 파동 분석 (3개)
            'zigzag_direction', 'zigzag_pivot_price', 'wave_progress',
            # 🎯 핵심 패턴 분석 (2개)
            'pattern_type', 'pattern_confidence',
            # 🚀 구조 점수 (1개)
            'structure_score',
            # 🚀 모멘텀 (1개)
            'price_momentum',
            # ✅ integrated 컬럼들(5개)은 integrated 파일에서 처리
            # 'volatility_level', 'risk_level', 'integrated_direction', 'sentiment', 'sentiment_label'
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
            existing_columns = [c for c in technical_columns if c in df.columns and c in table_columns]
            
            # 🔧 누락된 컬럼 자동 추가
            missing_columns = []
            for col in technical_columns:
                if col in df.columns and col not in table_columns:
                    missing_columns.append(col)
            
            if missing_columns:
                for col in missing_columns:
                    # 컬럼 타입 결정
                    if col in ['wave_phase', 'three_wave_pattern', 'sideways_pattern', 'pattern_type', 
                              'pattern_direction', 'pattern_class', 'flow_level_meta', 'sentiment_label']:
                        col_type = 'TEXT'
                    elif col in ['zigzag_direction', 'pattern_start_idx', 'pattern_end_idx']:
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
            existing_columns = [c for c in technical_columns if c in df.columns and c in table_columns]
            
            if not existing_columns:
                return False
            
            
            # 🚀 대량 UPDATE 최적화 (기존 데이터 보존하면서 전략지표만 업데이트)
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
                        if col in ['zigzag_direction', 'pattern_start_idx', 'pattern_end_idx']:
                            try:
                                value = int(value)
                            except (ValueError, TypeError):
                                value = 0
                        elif col in ['wave_phase', 'pattern_type', 'volatility_level', 'risk_level', 'integrated_direction', 'sentiment_label']:
                            # TEXT 타입 컬럼 - 문자열로 변환
                            value = str(value) if value is not None else 'unknown'
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
                # symbol이 없으면 coin 사용
                sym_val = row['symbol'] if 'symbol' in row else row['coin']
                row_data.extend([sym_val, row['interval'], row['timestamp']])
                update_data.append(row_data)
            
            if update_data:
                # 🚀 executemany로 배치 업데이트 (기존 OHLCV 데이터는 보존)
                # DB 컬럼명이 coin인지 symbol인지 확인
                col_name = 'symbol' if 'symbol' in table_columns else 'coin'
                
                set_clauses = [f'"{col}" = ?' for col in existing_columns]
                
                sql = f"""
                    UPDATE candles 
                    SET {', '.join(set_clauses)}
                    WHERE {col_name} = ? AND interval = ? AND timestamp = ?
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
        print(f"    ❌ 즉시 저장 중 오류: {symbol}/{interval} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """🚀 성능 최적화된 메인 함수"""
    import time
    start_time = time.time()
    
    # 환경 변수에서 인터벌 목록 가져오기 (기본값 유지)
    env_intervals = os.getenv('CANDLE_INTERVALS', '15m,30m,240m,1d')
    intervals = [i.strip() for i in env_intervals.split(',')]
    
    total_processed = 0
    
    print(f"[{datetime.now()}] 🚀 기술지표 계산 시작 - {len(intervals)}개 인터벌: {intervals}")
    print(f"[{datetime.now()}] 📂 대상 DB: {DB_PATH}")
    
    for interval in intervals:
        interval_start = time.time()
        print(f"[{datetime.now()}] 🔄 {interval} 캔들 기술지표 계산 중...")
        
        try:
            process_interval_data(interval, 'candles')
            interval_time = time.time() - interval_start
            print(f"[{datetime.now()}] ✅ {interval} 완료 ({interval_time:.1f}초)")
            total_processed += 1
        except Exception as e:
            print(f"[{datetime.now()}] ❌ {interval} 처리 중 오류 발생: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"[{datetime.now()}] 🎉 전략지표 전체 완료! {total_processed}/{len(intervals)} 인터벌 처리됨 (총 {total_time:.1f}초)")

if __name__ == '__main__':
    main()
