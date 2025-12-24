import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import traceback

# pandas FutureWarning 해결을 위한 설정
pd.set_option('future.no_silent_downcasting', True)

# 데이터베이스 경로 설정 (env에서 가져오기)
try:
    from rl_pipeline.core.env import config
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from rl_pipeline.core.env import config
    except ModuleNotFoundError:
        from core.env import config

# 환경 변수 RL_DB_PATH가 있으면 그것을 사용, 없으면 config.RL_DB 사용 (하위 호환성)
DB_PATH = os.getenv('RL_DB_PATH', config.RL_DB)

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

# 인터벌별 레짐 계산 기준 (기본값 및 확장)
REGIME_CRITERIA = {
    '15m': {
        'rsi_weight': 0.4,
        'macd_weight': 0.3,
        'volume_weight': 0.3,
        'volatility_threshold': 0.025,
        'lookback_period': 15
    },
    '30m': {
        'rsi_weight': 0.5,
        'macd_weight': 0.3,
        'volume_weight': 0.2,
        'volatility_threshold': 0.02,
        'lookback_period': 20
    },
    '60m': {  # KRX 등 중단기
        'rsi_weight': 0.55,
        'macd_weight': 0.25,
        'volume_weight': 0.2,
        'volatility_threshold': 0.018,
        'lookback_period': 24
    },
    '240m': {
        'rsi_weight': 0.6,
        'macd_weight': 0.2,
        'volume_weight': 0.2,
        'volatility_threshold': 0.015,
        'lookback_period': 30,
        'is_primary': True
    },
    '1d': {
        'rsi_weight': 0.7,
        'macd_weight': 0.2,
        'volume_weight': 0.1,
        'volatility_threshold': 0.01,
        'lookback_period': 60
    },
    '1w': {  # 장기
        'rsi_weight': 0.8,
        'macd_weight': 0.15,
        'volume_weight': 0.05,
        'volatility_threshold': 0.005,
        'lookback_period': 100
    }
}

# 기본 기준값 (매칭되는 인터벌이 없을 때 사용)
DEFAULT_CRITERIA = {
    'rsi_weight': 0.5,
    'macd_weight': 0.3,
    'volume_weight': 0.2,
    'volatility_threshold': 0.02,
    'lookback_period': 20
}

def get_regime_criteria(interval: str) -> dict:
    """인터벌에 맞는 기준 반환 (없으면 기본값 사용)"""
    return REGIME_CRITERIA.get(interval, DEFAULT_CRITERIA)

# 레짐 안정화 기본 파라미터
REGIME_MIN_STAY = 2
REGIME_CONF_GATE = 0.4

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

    macd_mag = (macd.abs() / (macd.abs().ewm(span=20, min_periods=1).mean() + 1e-9)).clip(0, 1)
    macd_side = np.sign(macd - macd_sig).astype(float)

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
    if v >= 0.6:   return 'very_bullish'
    if v >= 0.3:   return 'bullish'
    if v <= -0.6:  return 'very_bearish'
    if v <= -0.3:  return 'bearish'
    return 'neutral'

# -------------------- 분석 파라미터 --------------------
ZIGZAG_LOOKBACK_MAP = {
    '15m': 3, '30m': 3, '60m': 3, '240m': 2, '1d': 2, '1w': 2
}
MIN_REQUIREMENTS = {
    'min_zigzag_points': 1,
    'min_unique_pivots': 1,
    'min_wave_progress': 0.001,
    'min_pattern_confidence': 0.01
}

# -------------------- 분석 함수 --------------------
def validate_zigzag_data(df: pd.DataFrame, interval: str) -> bool:
    if 'zigzag_direction' not in df.columns:
        return False
    non_zero_directions = (df['zigzag_direction'] != 0).sum()
    if non_zero_directions < MIN_REQUIREMENTS['min_zigzag_points']:
        return False
    return True

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

# 🚀 레짐 계산 함수들
def calculate_composite_regime_score(df: pd.DataFrame, interval: str) -> pd.Series:
    """인터벌별 복합 지표 기반 레짐 점수 계산"""
    criteria = get_regime_criteria(interval)  # 안전하게 가져오기
    
    rsi_data = df.get('rsi', pd.Series(50.0, index=df.index)).fillna(50.0)
    rsi_score = (rsi_data - 20) / 60
    rsi_score = rsi_score.clip(0, 1)
    
    macd_data = df.get('macd', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_signal_data = df.get('macd_signal', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_momentum = (macd_data - macd_signal_data).abs()
    macd_score = macd_momentum / (macd_momentum.rolling(20).max() + 1e-9)
    
    volume_data = df.get('volume_ratio', pd.Series(1.0, index=df.index)).fillna(1.0)
    volume_score = volume_data.clip(0, 3) / 3
    
    atr_data = df.get('atr', pd.Series(0.02, index=df.index)).fillna(0.02)
    close_data = df.get('close', pd.Series(100.0, index=df.index)).fillna(100.0)
    volatility_score = 1 - (atr_data / close_data).clip(0, 0.1) / 0.1
    
    composite_score = (
        criteria['rsi_weight'] * rsi_score +
        criteria['macd_weight'] * macd_score +
        criteria['volume_weight'] * volume_score +
        0.1 * volatility_score
    )
    
    return composite_score.clip(0, 1)

def classify_regime_stage(composite_score: pd.Series, interval: str) -> pd.Series:
    """복합 점수를 7단계 레짐으로 분류"""
    composite_score = composite_score.fillna(0.5)
    
    # 인터벌별 임계값 (없으면 기본값 사용)
    default_thresh = [0.25, 0.4, 0.55, 0.65, 0.8, 0.95]
    thresholds = {
        '15m': [0.2, 0.35, 0.5, 0.6, 0.75, 0.9],
        '30m': [0.25, 0.4, 0.55, 0.65, 0.8, 0.95],
        '60m': [0.25, 0.4, 0.55, 0.65, 0.8, 0.95], # 30m와 동일 적용
        '240m': [0.3, 0.45, 0.6, 0.7, 0.85, 0.99],
        '1d': [0.3, 0.45, 0.6, 0.7, 0.85, 0.99],
        '1w': [0.35, 0.5, 0.6, 0.7, 0.85, 0.99]   # 더 보수적
    }
    
    thresh = thresholds.get(interval, default_thresh)
    
    bins = [0] + thresh + [1.0]
    bins = sorted(list(set(bins)))
    
    regime_stage = pd.cut(composite_score, 
                         bins=bins,
                         labels=list(range(1, len(bins))),
                         include_lowest=True,
                         duplicates='drop')
    
    regime_stage = regime_stage.fillna(4)
    regime_stage = regime_stage.astype(int)
    
    return regime_stage

def calculate_regime_confidence(df: pd.DataFrame, interval: str) -> pd.Series:
    criteria = get_regime_criteria(interval)
    lookback = criteria['lookback_period']
    
    if len(df) < lookback:
        return pd.Series(0.5, index=df.index)
    
    rsi_data = df.get('rsi', pd.Series(50.0, index=df.index)).fillna(50.0)
    rsi_std = rsi_data.rolling(lookback).std()
    rsi_consistency = (1 - rsi_std / 20).clip(0, 1)
    
    macd_data = df.get('macd', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_signal_data = df.get('macd_signal', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_diff = (macd_data - macd_signal_data).abs()
    macd_strength = macd_diff.rolling(lookback).mean()
    macd_strength = (macd_strength / (macd_strength.rolling(lookback).max() + 1e-9)).clip(0, 1)
    
    volume_data = df.get('volume_ratio', pd.Series(1.0, index=df.index)).fillna(1.0)
    volume_std = volume_data.rolling(lookback).std()
    volume_consistency = (1 - volume_std / 2.0).clip(0, 1)
    
    confidence = (
        criteria['rsi_weight'] * rsi_consistency +
        criteria['macd_weight'] * macd_strength +
        criteria['volume_weight'] * volume_consistency
    )
    
    return confidence.clip(0, 1)

def calculate_regime_transition_probability(df: pd.DataFrame, interval: str) -> pd.Series:
    criteria = get_regime_criteria(interval)
    lookback = criteria['lookback_period']
    
    if len(df) < lookback * 2:
        return pd.Series(0.05, index=df.index)
    
    composite_score = calculate_composite_regime_score(df, interval)
    smooth_score = composite_score.ewm(span=lookback, min_periods=1).mean()
    regime_stage = classify_regime_stage(smooth_score, interval)
    
    confidence = calculate_regime_confidence(df, interval)
    
    changes = regime_stage.diff().abs()
    minor_weight = 0.3 + (confidence * 0.3)
    minor_changes = (changes == 1).astype(float) * minor_weight
    major_changes = (changes >= 2).astype(float) * 1.0
    all_changes = minor_changes + major_changes
    
    change_frequency = all_changes.rolling(lookback, min_periods=1).mean()
    transition_prob = change_frequency.clip(0, 0.4)
    
    return transition_prob.fillna(0.05)

def save_integrated_indicators_immediate(df: pd.DataFrame, symbol: str, interval: str) -> bool:
    try:
        if df.empty: return False
        
        integrated_columns = [
            'volatility_level', 'risk_level', 'integrated_direction', 
            'sentiment', 'sentiment_label', 
            'regime_stage', 'regime_label', 'regime_confidence', 'regime_transition_prob'
        ]

        numeric_cols = ['sentiment', 'regime_confidence', 'regime_transition_prob']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(4)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(candles)")
            table_columns = [c[1] for c in cursor.fetchall()]
            
            # 컬럼 자동 추가
            for col in integrated_columns:
                if col not in table_columns:
                    col_type = 'INTEGER' if col == 'regime_stage' else 'REAL' if col in numeric_cols else 'TEXT'
                    try:
                        cursor.execute(f'ALTER TABLE candles ADD COLUMN "{col}" {col_type}')
                    except Exception: pass
            
            cursor.execute("PRAGMA table_info(candles)")
            table_columns = [c[1] for c in cursor.fetchall()]
            
            existing_columns = [c for c in integrated_columns if c in df.columns and c in table_columns]
            if not existing_columns: return False
            
            update_data = []
            col_name = 'symbol' if 'symbol' in table_columns else 'coin'
            
            for _, row in df.iterrows():
                row_data = []
                for col in existing_columns:
                    if pd.notna(row[col]):
                        value = row[col]
                        if col in ['volatility_level', 'risk_level', 'integrated_direction', 'sentiment_label', 'regime_label']:
                            value = str(value)
                        elif col == 'regime_stage':
                            value = int(value)
                        else:
                            value = float(value)
                        row_data.append(value)
                    else:
                        row_data.append(None)
                
                ts_val = int(row['timestamp'])
                sym_val = row['symbol'] if 'symbol' in row else symbol
                row_data.extend([sym_val, row['interval'], ts_val])
                update_data.append(row_data)
            
            if update_data:
                set_clauses = [f'"{col}" = ?' for col in existing_columns]
                sql = f"""
                    UPDATE candles 
                    SET {', '.join(set_clauses)}
                    WHERE {col_name} = ? AND interval = ? AND timestamp = ?
                """
                cursor.executemany(sql, update_data)
                conn.commit()
                return True
            return False
            
    except Exception as e:
        print(f"    ❌ 통합 분석 저장 중 오류: {symbol}/{interval} - {str(e)}")
        return False

def _calculate_integrated_direction_series(df: pd.DataFrame) -> pd.Series:
    rsi = df.get('rsi', pd.Series(50.0, index=df.index)).fillna(50.0)
    macd = df.get('macd', pd.Series(0.0, index=df.index)).fillna(0.0)
    macd_signal = df.get('macd_signal', pd.Series(0.0, index=df.index)).fillna(0.0)
    volume_ratio = df.get('volume_ratio', pd.Series(1.0, index=df.index)).fillna(1.0)
    
    strong_up = rsi > 65
    up = rsi > 53
    strong_down = rsi < 35
    down = rsi < 47
    
    bullish = macd > macd_signal
    bearish = macd < macd_signal
    
    cond_strong_bull = strong_up & bullish
    cond_strong_bear = strong_down & bearish
    cond_bull = up & bullish
    cond_bear = down & bearish
    cond_reversal_up = down & bullish
    cond_reversal_down = up & bearish
    
    conditions = [cond_strong_bull, cond_strong_bear, cond_bull, cond_bear, cond_reversal_up, cond_reversal_down]
    choices = ['Strong Bullish', 'Strong Bearish', 'Bullish', 'Bearish', 'Bullish Reversal', 'Bearish Reversal']
    
    return pd.Series(np.select(conditions, choices, default='Neutral'), index=df.index)

def perform_integrated_analysis(symbol: str, interval: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA journal_mode = WAL")
            
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(candles)")
            cols = [c[1] for c in cursor.fetchall()]
            col_name = 'symbol' if 'symbol' in cols else 'coin'

            df = pd.read_sql(
                f"SELECT * FROM candles WHERE {col_name}=? AND interval=? ORDER BY timestamp",
                conn, params=(symbol, interval)
            )
            if df.empty or len(df) < 20: return False
        
        # 1. Volatility & Risk
        if 'atr' in df.columns and 'close' in df.columns:
            volatility = df['atr'] / df['close']
            df['volatility_level'] = volatility.apply(calculate_volatility_level)
        else:
            df['volatility_level'] = 'unknown'
            
        if 'risk_score' in df.columns:
            df['risk_level'] = df['risk_score'].apply(calculate_risk_level)
        else:
            df['risk_level'] = 'unknown'
            
        # 2. Integrated Direction
        df['integrated_direction'] = _calculate_integrated_direction_series(df)
        
        # 3. Sentiment
        df['sentiment'] = _compute_sentiment_series(df)
        df['sentiment_label'] = df['sentiment'].apply(_label_sentiment)
        
        # 4. Regime Analysis
        composite_score = calculate_composite_regime_score(df, interval)
        df['regime_stage'] = classify_regime_stage(composite_score, interval)
        df['regime_label'] = df['regime_stage'].map(REGIME_STAGES)
        df['regime_confidence'] = calculate_regime_confidence(df, interval)
        df['regime_transition_prob'] = calculate_regime_transition_probability(df, interval)
        
        # Save
        return save_integrated_indicators_immediate(df, symbol, interval)
        
    except Exception as e:
        print(f"❌ 오류 발생: {symbol}/{interval} - {str(e)}")
        return False

def validate_regime_results():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(candles)")
            cols = [c[1] for c in cursor.fetchall()]
            col_name = 'symbol' if 'symbol' in cols else 'coin'
            
            df = pd.read_sql(f"""
                SELECT {col_name}, interval, regime_stage, regime_label, 
                       regime_confidence, regime_transition_prob
                FROM candles 
                WHERE regime_stage IS NOT NULL
                ORDER BY {col_name}, interval, timestamp
            """, conn)
        
        if df.empty:
            print("⚠️ 검증할 레짐 데이터가 없습니다")
            return
        
        if 'symbol' not in df.columns and 'coin' in df.columns:
            df.rename(columns={'coin': 'symbol'}, inplace=True)

        print("\n📊 레짐 검증 리포트")
        print("="*60)
        print(f"✅ 커버리지: {len(df):,}개 레짐 데이터")
        
        # 간단 검증
        valid = df['regime_stage'].between(1, 7).all()
        print(f"✅ 값 범위 (1~7): {'✅ 정상' if valid else '❌ 범위 초과'}")
        
        print("\n📈 레짐 분포:")
        print(df['regime_label'].value_counts())
        print("="*60)
        
    except Exception as e:
        print(f"❌ 레짐 검증 중 오류: {e}")

def run_full_integrated_analysis():
    """전체 통합분석 실행 (유동적 인터벌 감지)"""
    
    print(f"🚀 통합 분석 시작")
    print(f"📂 대상 DB: {DB_PATH}")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(candles)")
        cols = [c[1] for c in cursor.fetchall()]
        col_name = 'symbol' if 'symbol' in cols else 'coin'

        # 🚀 DB에서 실제 존재하는 모든 종목/인터벌 조회 (환경변수 무시하고 DB 기준)
        query = f"""
            SELECT DISTINCT {col_name}, interval, COUNT(*) as count
            FROM candles 
            WHERE rsi IS NOT NULL AND macd IS NOT NULL
            GROUP BY {col_name}, interval
            ORDER BY {col_name}, interval
        """
        cursor.execute(query)
        coin_intervals = cursor.fetchall()
    
    if not coin_intervals:
        print("⚠️ 처리할 데이터가 없습니다 (지표 계산 먼저 수행 필요)")
        return
    
    # 인터벌 목록 추출 (로그용)
    intervals = sorted(list(set([row[1] for row in coin_intervals])))
    print(f"⏱️ 감지된 인터벌: {intervals}")
    
    total_groups = len(coin_intervals)
    print(f"📊 처리 대상: {total_groups}개 종목/인터벌 그룹")
    
    success_count = 0
    error_count = 0
    batch_size = 10
    
    for i in range(0, len(coin_intervals), batch_size):
        batch = coin_intervals[i:i + batch_size]
        print(f"🔄 배치 {i//batch_size + 1}: {len(batch)}개 그룹 처리 중...")
        
        for symbol, interval, count in batch:
            if perform_integrated_analysis(symbol, interval):
                success_count += 1
                print(f"✅ 통합 분석 성공: {symbol}/{interval}")
            else:
                error_count += 1
                print(f"❌ 통합 분석 실패: {symbol}/{interval}")
    
    print(f"🎉 통합 분석 완료: 성공 {success_count}개, 실패 {error_count}개")
    validate_regime_results()

if __name__ == '__main__':
    run_full_integrated_analysis()
