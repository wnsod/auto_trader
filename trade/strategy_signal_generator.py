import sys
import os
import sqlite3
import time
from collections import defaultdict
from typing import Optional

# ============================================================================
# 🔥 [핵심] 환경변수 변환을 모든 모듈 임포트 전에 수행해야 함!
# ============================================================================
# 🆕 경로 설정 (rl_pipeline 및 signal_selector 로드용)
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)

# 🚀 패키지 경로 최적화 (trade 폴더 독립 실행 지원)
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 🆕 rl_pipeline 의존성 제거 - trade 폴더만으로 독립 실행 가능

# 🚀 finalize_path 함수 정의 (Docker 환경 전용)
def finalize_path(path):
    """경로를 절대 경로로 변환 (Docker 환경)"""
    if not path: return path
    return os.path.abspath(path)

# 🔥 엔진 환경 체크 (크로스 플랫폼 호환) - 모든 모듈 임포트 전에 실행!
REQUIRED_ENV_VARS = ['STRATEGY_DB_PATH', 'DATA_STORAGE_PATH', 'CANDLES_DB_PATH', 'TRADING_SYSTEM_DB_PATH']
def check_environment():
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        print(f"❌ [Engine Error] 필수 환경 변수 누락: {', '.join(missing)}")
        sys.exit(1)
    # 🚀 모든 경로 환경변수를 Windows 절대 경로로 변환
    for var in REQUIRED_ENV_VARS:
        os.environ[var] = finalize_path(os.environ[var])
    os.environ['STRATEGIES_DB_PATH'] = os.environ['STRATEGY_DB_PATH']
    os.environ['GLOBAL_STRATEGY_DB_PATH'] = finalize_path(os.environ.get('GLOBAL_STRATEGY_DB_PATH', ''))

# 🔥 [핵심] 환경변수 변환 먼저 실행!
check_environment()

# ============================================================================
# 🔥 이제 환경변수가 올바른 경로로 설정되었으므로 모듈 임포트 시작
# ============================================================================
import pandas as pd
import numpy as np
from functools import lru_cache  # 🚀 [성능] LRU 캐시 추가
from trade.core.sequence_analyzer import SequenceAnalyzer

# 🔥 트레이딩 엔진 전용 DB 유틸리티 임포트 (경로 명시)
try:
    from trade.core.database import get_learning_data, TRADING_SYSTEM_DB_PATH, STRATEGY_DB_PATH, CANDLES_DB_PATH
except ImportError:
    from core.database import get_learning_data, TRADING_SYSTEM_DB_PATH, STRATEGY_DB_PATH, CANDLES_DB_PATH

# 🚀 엔진 전용 모드 설정 (불필요한 DB 쓰기 및 중복 로딩 방지)
os.environ['ENGINE_READ_ONLY'] = 'true'
os.environ['SKIP_REDUNDANT_LOAD'] = 'true'

print(f"READY: 고성능 트레이딩 엔진 가동 (독립 I/O): {os.path.basename(__file__)}")

# ============================================================================
# 🆕 [자가 진화] 증분 검증 시스템 - 시그널 예측 정확도 학습
# ============================================================================
def _extract_technical_pattern(row: dict) -> str:
    """시그널에서 기술적 패턴 추출 (signal_selector와 동일한 형식)"""
    try:
        rsi = float(row.get('rsi', 50) or 50)
        rsi_cat = 'oversold' if rsi < 30 else ('overbought' if rsi > 70 else 'neutral')
        
        macd = float(row.get('macd', 0) or 0)
        macd_cat = 'bullish' if macd > 0.001 else ('bearish' if macd < -0.001 else 'neutral')
        
        direction = str(row.get('integrated_direction', 'neutral') or 'neutral').lower()
        if 'long' in direction or 'bull' in direction or 'up' in direction:
            dir_cat = 'up'
        elif 'short' in direction or 'bear' in direction or 'down' in direction:
            dir_cat = 'down'
        else:
            dir_cat = 'neutral'
        
        wave = str(row.get('wave_phase', 'unknown') or 'unknown').lower()
        wave_cat = wave if wave in ['accumulation', 'markup', 'distribution', 'markdown'] else 'unknown'
        
        interval = str(row.get('interval', 'unknown') or 'unknown')
        return f"{interval}_{rsi_cat}_{macd_cat}_{dir_cat}_{wave_cat}"
    except:
        return "unknown_pattern"


@lru_cache(maxsize=32)  # 🚀 [성능] 인터벌별 초 변환 캐싱
def _get_interval_seconds(interval: str) -> int:
    """인터벌 문자열을 초 단위로 변환"""
    if interval.endswith('m'):
        return int(interval[:-1]) * 60
    elif interval.endswith('h'):
        return int(interval[:-1]) * 3600
    elif interval.endswith('d'):
        return int(interval[:-1]) * 86400
    return 900  # 기본값 15분


@lru_cache(maxsize=32)  # 🚀 [성능] 검증 대기 시간 캐싱
def _get_validation_delay(interval: str) -> int:
    """인터벌별 검증 대기 시간 (결과 확정까지 필요한 시간)"""
    # short horizon (4캔들) 기준으로 대기
    iv_secs = _get_interval_seconds(interval)
    return iv_secs * 4  # 최소 4캔들 대기


def validate_signals_incremental(all_data_df: pd.DataFrame, db_now: int, 
                                  trading_db_path: str, candles_db_path: str,
                                  global_db_path: Optional[str] = None) -> dict:
    """
    🚀 [자가 진화] 증분 시그널 검증 시스템
    
    특징:
    - 미검증 시그널만 대상 (validated_at IS NULL)
    - 이미 로드된 캔들 데이터 활용 (추가 I/O 최소화)
    - 결과 확정 불가능한 시그널은 스킵 (다음 사이클에 재시도)
    - 점진적으로 빨라짐 (초기: 모두 검증 → 이후: 새 시그널만)
    - 🚀 [성능] 벡터 연산 + 배치 처리 최적화
    
    Returns:
        검증 통계 딕셔너리
    """
    from trade.core.database import get_db_connection
    
    stats = {'total_checked': 0, 'validated': 0, 'skipped_pending': 0, 'patterns_saved': 0}
    
    try:
        # 1. 시그널 테이블 컬럼 마이그레이션 (없으면 추가)
        with get_db_connection(trading_db_path, read_only=False) as conn:
            cursor = conn.execute("PRAGMA table_info(signals)")
            cols = [row[1] for row in cursor.fetchall()]
            
            # validated_at 컬럼
            if 'validated_at' not in cols:
                print("🔧 validated_at 컬럼 마이그레이션 중...")
                conn.execute("ALTER TABLE signals ADD COLUMN validated_at INTEGER DEFAULT NULL")
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_unvalidated ON signals(validated_at) WHERE validated_at IS NULL')
                print("✅ validated_at 컬럼 추가 완료")
            
            # 🆕 전략 관련 컬럼 마이그레이션
            if 'recommended_strategy' not in cols:
                print("🔧 recommended_strategy 컬럼 마이그레이션 중...")
                conn.execute("ALTER TABLE signals ADD COLUMN recommended_strategy TEXT DEFAULT NULL")
                print("✅ recommended_strategy 컬럼 추가 완료")
            
            if 'strategy_match' not in cols:
                print("🔧 strategy_match 컬럼 마이그레이션 중...")
                conn.execute("ALTER TABLE signals ADD COLUMN strategy_match REAL DEFAULT NULL")
                print("✅ strategy_match 컬럼 추가 완료")
            
            if 'strategy_scores' not in cols:
                print("🔧 strategy_scores 컬럼 마이그레이션 중...")
                conn.execute("ALTER TABLE signals ADD COLUMN strategy_scores TEXT DEFAULT NULL")
                print("✅ strategy_scores 컬럼 추가 완료")
            
            conn.commit()
        
        # 2. 미검증 시그널 조회 (validated_at IS NULL)
        with get_db_connection(trading_db_path, read_only=True) as conn:
            
            # 🆕 recommended_strategy, strategy_match 컬럼 마이그레이션
            try:
                cursor = conn.execute("PRAGMA table_info(signals)")
                cols = [row[1] for row in cursor.fetchall()]
                # 읽기 전용이므로 마이그레이션은 별도 연결에서
            except:
                cols = []
            
            # 최근 48시간 내 미검증 시그널 조회
            cutoff_ts = db_now - (48 * 3600)
            
            # 🆕 전략 관련 컬럼 포함하여 조회 (없으면 NULL로 처리)
            strategy_cols = ", recommended_strategy, strategy_match" if 'recommended_strategy' in cols else ", NULL as recommended_strategy, NULL as strategy_match"
            
            unvalidated_df = pd.read_sql(f"""
                SELECT id, coin, interval, timestamp, current_price, volatility, 
                       integrated_direction, signal_score, action, target_price,
                       rsi, macd, wave_phase, pattern_type{strategy_cols}
                FROM signals 
                WHERE validated_at IS NULL 
                  AND timestamp > ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 5000
            """, conn, params=(cutoff_ts, db_now))
        
        if unvalidated_df.empty:
            print("ℹ️ 검증 대기 시그널 없음 (모두 검증 완료)")
            return stats
        
        stats['total_checked'] = len(unvalidated_df)
        print(f"\n🔬 [증분 검증] 미검증 시그널 {len(unvalidated_df)}개 발견")
        
        # 2. 캔들 데이터 준비 (이미 로드된 데이터 활용) - 🚀 최적화: 인덱싱 개선
        candle_cache = {}
        if not all_data_df.empty:
            # groupby 후 딕셔너리로 한 번에 변환 (반복적인 copy 제거)
            for (symbol, interval), group in all_data_df.groupby(['symbol', 'interval'], sort=False):
                key = f"{symbol}_{interval}"
                # 🚀 정렬 + 인덱스 설정을 한 번에 (inplace 대신 할당)
                candle_cache[key] = group.sort_values('timestamp').set_index('timestamp')
        
        # 3. 시그널별 검증 - 🚀 최적화: 사전 필터링 + 배치 처리
        stats_by_pattern = defaultdict(lambda: {'correct': 0, 'total': 0, 'profit_sum': 0.0})
        # 🆕 전략별 통계 수집
        stats_by_strategy = defaultdict(lambda: {'correct': 0, 'total': 0, 'profit_sum': 0.0, 'holding_hours': 0.0})
        validated_ids = []
        horizons = {'short': 4, 'mid': 12, 'long': 48}
        
        # 🚀 [성능] 인터벌별 검증 대기 시간을 미리 계산 (반복 호출 제거)
        validation_delays = {}
        for iv in unvalidated_df['interval'].unique():
            target_iv = '15m' if iv == 'combined' else iv
            validation_delays[iv] = _get_validation_delay(target_iv)
        
        # 🚀 [성능] 검증 가능한 시그널만 먼저 필터링 (조기 스킵)
        unvalidated_df = unvalidated_df.copy()
        unvalidated_df['validation_delay'] = unvalidated_df['interval'].map(validation_delays)
        unvalidated_df['can_validate'] = db_now >= (unvalidated_df['timestamp'] + unvalidated_df['validation_delay'])
        
        # 검증 불가능한 시그널 개수 카운트
        stats['skipped_pending'] = (~unvalidated_df['can_validate']).sum()
        
        # 검증 가능한 시그널만 처리
        validatable_df = unvalidated_df[unvalidated_df['can_validate']]
        
        if validatable_df.empty:
            print(f"   ⏳ 검증 대기 중: {stats['skipped_pending']}개 (결과 확정 전)")
            return stats
        
        # 🚀 [성능] iterrows 대신 to_dict('records') 사용 (2~5배 빠름)
        for row in validatable_df.to_dict('records'):
            sig_id = row['id']
            coin = row['coin']
            interval = row['interval']
            t0 = int(row['timestamp'])
            
            # combined은 15m 기준으로 검증
            target_interval = '15m' if interval == 'combined' else interval
            
            # 🚀 [성능] 이미 필터링되었으므로 대기 시간 체크 불필요
            validation_delay = row['validation_delay']
            
            # 캔들 데이터 조회
            cache_key = f"{coin}_{target_interval}"
            if cache_key not in candle_cache:
                # 캐시에 없으면 DB에서 조회
                try:
                    with get_db_connection(candles_db_path, read_only=True) as c_conn:
                        lookback = validation_delay * 2
                        candles = pd.read_sql("""
                            SELECT timestamp, high, low, close 
                            FROM candles 
                            WHERE symbol = ? AND interval = ? 
                              AND timestamp >= ? AND timestamp <= ?
                        """, c_conn, params=(coin, target_interval, t0 - lookback, db_now))
                    if candles.empty:
                        continue
                    candles.set_index('timestamp', inplace=True)
                    candle_cache[cache_key] = candles
                except:
                    continue
            
            candles = candle_cache.get(cache_key)
            if candles is None or candles.empty:
                continue
            
            # 검증 수행
            entry_p = row['current_price']
            vol = row['volatility'] or 0.02
            direction = str(row['integrated_direction'] or '').upper()
            action = str(row['action'] or '').lower()
            is_long = any(x in direction for x in ['LONG', 'BUY', 'BULL']) or action == 'buy'
            
            # target_price 사용 (없으면 volatility로 계산)
            stored_target = row.get('target_price', 0) or 0
            if stored_target > 0 and entry_p > 0:
                ratio = stored_target / entry_p
                target_p = stored_target if 0.5 <= ratio <= 2.0 else entry_p * (1 + vol) if is_long else entry_p * (1 - vol)
            else:
                target_p = entry_p * (1 + vol) if is_long else entry_p * (1 - vol)
            
            signal_score = row.get('signal_score', 0) or 0
            signal_weight = 1.5 if abs(signal_score) > 0.5 else (1.2 if abs(signal_score) > 0.3 else 1.0)
            
            iv_secs = _get_interval_seconds(target_interval)
            any_horizon_validated = False
            
            for p_type, h_count in horizons.items():
                expire_ts = t0 + (h_count * iv_secs)
                if db_now < expire_ts:
                    continue  # 이 horizon은 아직 확정 안됨
                
                try:
                    window = candles[(candles.index >= t0) & (candles.index <= expire_ts)]
                    if window.empty:
                        continue
                    
                    is_hit = (window['high'].max() >= target_p) if is_long else (window['low'].min() <= target_p)
                    
                    # 실제 수익률 계산
                    final_price = window['close'].iloc[-1] if len(window) > 0 else entry_p
                    profit_pct = ((final_price - entry_p) / entry_p * 100) if entry_p > 0 else 0.0
                    if not is_long:
                        profit_pct = -profit_pct
                    
                    # 패턴별 통계 수집
                    tech_pattern = _extract_technical_pattern(row)  # 🚀 이미 dict이므로 to_dict() 불필요
                    expert_key = f"{interval}_{p_type}"
                    
                    if is_hit:
                        stats_by_pattern[tech_pattern]['correct'] += signal_weight
                        stats_by_pattern[expert_key]['correct'] += signal_weight
                    stats_by_pattern[tech_pattern]['total'] += signal_weight
                    stats_by_pattern[tech_pattern]['profit_sum'] += profit_pct * signal_weight
                    stats_by_pattern[expert_key]['total'] += signal_weight
                    
                    # 🆕 전략별 검증 (전략마다 성공/실패 기준이 다름!)
                    strategy_type = row.get('recommended_strategy', None)
                    if strategy_type and strategy_type != 'None' and pd.notna(strategy_type):
                        holding_hours = (expire_ts - t0) / 3600.0
                        strat_key = f"{strategy_type}_{p_type}"  # 예: bottom_short, trend_mid
                        
                        # 🆕 전략별 검증 함수 호출
                        try:
                            from trade.core.strategies import validate_strategy_signal
                            
                            entry_rsi = row.get('rsi', 50) or 50
                            validation_result = validate_strategy_signal(
                                strategy_type=strategy_type,
                                entry_price=entry_p,
                                candle_window=window,
                                is_long=is_long,
                                entry_rsi=entry_rsi
                            )
                            
                            # 전략별 검증 결과 사용
                            strategy_success = validation_result.is_success
                            strategy_profit = validation_result.profit_pct
                            validation_confidence = validation_result.confidence
                            
                            # 신뢰도가 낮은 검증은 가중치 낮춤
                            strat_weight = signal_weight * validation_confidence
                            
                        except ImportError:
                            # 폴백: 기본 수익률 기반 검증
                            strategy_success = is_hit
                            strategy_profit = profit_pct
                            strat_weight = signal_weight
                        
                        if strategy_success:
                            stats_by_strategy[strategy_type]['correct'] += strat_weight
                            stats_by_strategy[strat_key]['correct'] += strat_weight
                        stats_by_strategy[strategy_type]['total'] += strat_weight
                        stats_by_strategy[strategy_type]['profit_sum'] += strategy_profit * strat_weight
                        stats_by_strategy[strategy_type]['holding_hours'] += holding_hours * strat_weight
                        stats_by_strategy[strat_key]['total'] += strat_weight
                        stats_by_strategy[strat_key]['profit_sum'] += strategy_profit * strat_weight
                    
                    any_horizon_validated = True
                except:
                    continue
            
            if any_horizon_validated:
                validated_ids.append(sig_id)
                stats['validated'] += 1
        
        # 4. 검증 완료 표시 (validated_at 업데이트)
        if validated_ids:
            with get_db_connection(trading_db_path, read_only=False) as conn:
                # 배치 업데이트
                conn.execute(f"""
                    UPDATE signals SET validated_at = ? 
                    WHERE id IN ({','.join('?' * len(validated_ids))})
                """, [db_now] + validated_ids)
                conn.commit()
        
        # 5. 학습 결과 저장 (signal_feedback_scores)
        if global_db_path and stats_by_pattern:
            try:
                with sqlite3.connect(global_db_path) as conn:
                    # 테이블 생성 (없으면)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                            coin TEXT, interval TEXT, signal_pattern TEXT,
                            success_rate REAL, avg_profit REAL, total_trades INTEGER,
                            confidence REAL, last_updated INTEGER,
                            PRIMARY KEY (coin, signal_pattern)
                        )
                    """)
                    
                    # avg_profit 컬럼 확인
                    cursor = conn.execute("PRAGMA table_info(signal_feedback_scores)")
                    cols = [r[1] for r in cursor.fetchall()]
                    if 'avg_profit' not in cols:
                        conn.execute("ALTER TABLE signal_feedback_scores ADD COLUMN avg_profit REAL DEFAULT 0.0")
                    
                    for pattern, s in stats_by_pattern.items():
                        if s['total'] < 2:
                            continue
                        accuracy = s['correct'] / s['total']
                        avg_profit = s['profit_sum'] / s['total']
                        pattern_interval = pattern.split('_')[0] if '_' in pattern else 'unknown'
                        confidence = min(1.0, s['total'] / 20.0)
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO signal_feedback_scores 
                            (coin, interval, signal_pattern, success_rate, avg_profit, total_trades, confidence, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, ('ALL', pattern_interval, pattern, accuracy, avg_profit, int(s['total']), confidence, db_now))
                        stats['patterns_saved'] += 1
                    
                    conn.commit()
            except Exception as e:
                print(f"⚠️ 학습 결과 저장 오류: {e}")
        
        # 🆕 6. 전략별 학습 결과 저장 (strategy_feedback 테이블)
        stats['strategies_saved'] = 0
        if stats_by_strategy:
            try:
                # 전략 시스템 임포트 (아직 로드 안 됐을 수 있음)
                try:
                    from trade.core.strategies import update_strategy_feedback, create_strategy_feedback_table
                    strategy_available = True
                except ImportError:
                    strategy_available = False
                
                if strategy_available:
                    # 테이블 생성 확인
                    create_strategy_feedback_table(trading_db_path)
                    
                    for strat_key, s in stats_by_strategy.items():
                        if s['total'] < 2:
                            continue
                        
                        # 전략 타입과 horizon 분리
                        parts = strat_key.split('_')
                        strategy_type = parts[0] if parts else strat_key
                        
                        success = s['correct'] / s['total'] > 0.5
                        avg_profit = s['profit_sum'] / s['total']
                        avg_holding = s['holding_hours'] / s['total'] if s.get('holding_hours', 0) > 0 else 0
                        
                        # 🆕 feedback_type 추가 (시그널 검증은 진입 전략 기준)
                        update_strategy_feedback(
                            db_path=trading_db_path,
                            strategy_type=strategy_type,
                            market_condition='signal_validation',  # 시그널 검증 기반
                            signal_pattern=strat_key,
                            success=success,
                            profit_pct=avg_profit,
                            holding_hours=avg_holding,
                            feedback_type='entry'  # 시그널 검증은 진입 판단 검증
                        )
                        stats['strategies_saved'] += 1
                    
                    print(f"   🎯 전략별 학습: {stats['strategies_saved']}개 전략 업데이트")
            except Exception as e:
                print(f"⚠️ 전략별 학습 저장 오류: {e}")
        
        # 7. 결과 출력
        print(f"   ✅ 검증 완료: {stats['validated']}개 | 대기중: {stats['skipped_pending']}개 | 패턴 저장: {stats['patterns_saved']}개 | 전략 학습: {stats.get('strategies_saved', 0)}개")
        
    except Exception as e:
        print(f"⚠️ 증분 검증 오류: {e}")
    
    return stats


# 🆕 라이브러리 및 GPU/JAX 로드 확인
try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    from signal_selector.config import USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE
    from signal_selector import get_signal_selector
except ImportError:
    USE_GPU_ACCELERATION = True
    AI_MODEL_AVAILABLE = False
    from signal_selector.core.selector import SignalSelector
    def get_signal_selector(): return SignalSelector()

# 🆕 전략 시스템 임포트
try:
    from trade.core.strategies import (
        evaluate_all_strategies, select_best_strategies, get_top_strategies,
        serialize_strategy_scores, create_strategy_feedback_table,
        STRATEGY_EXIT_RULES, STRATEGY_ENTRY_THRESHOLDS, get_strategy_description
    )
    STRATEGY_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 전략 시스템 로드 실패 (기본 모드): {e}")
    STRATEGY_SYSTEM_AVAILABLE = False

# 🧬 전략 진화 시스템 임포트
try:
    from trade.core.strategy_evolution import (
        get_evolution_manager, get_strategy_level, get_best_evolved_strategy,
        EvolutionLevel
    )
    EVOLUTION_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 전략 진화 시스템 로드 실패: {e}")
    EVOLUTION_SYSTEM_AVAILABLE = False


def _prepare_candle_data_for_strategy(coin_data: pd.DataFrame, interval: str) -> dict:
    """캔들 데이터에서 전략 매칭용 추가 정보 추출"""
    candle_info = {
        'recent_change_pct': 0.0,
        'was_sideways': False,
        'wave_transition': '',
        'wave_progress': 0.5,
        'has_divergence': False,
        'adx_declining': False,
        'near_support': False,
        'near_resistance': False,
    }
    
    if coin_data is None or len(coin_data) < 3:
        return candle_info
    
    try:
        # 최근 데이터 (시간순 정렬)
        df = coin_data.sort_values('timestamp', ascending=False).head(10)
        
        if len(df) >= 3:
            # 최근 변화율 (최근 3개 캔들)
            closes = df['close'].values[:3]
            if closes[-1] > 0:
                candle_info['recent_change_pct'] = ((closes[0] - closes[-1]) / closes[-1]) * 100
        
        if len(df) >= 5:
            # 이전 횡보 여부 (ADX 기반)
            adx_values = df['adx'].dropna().values[:5]
            if len(adx_values) >= 3:
                avg_adx = np.mean(adx_values[1:])  # 이전 ADX
                candle_info['was_sideways'] = avg_adx < 25
                candle_info['adx_declining'] = adx_values[0] < avg_adx if len(adx_values) > 1 else False
        
        # Wave Phase 전환 감지
        waves = df['wave_phase'].dropna().values[:3]
        if len(waves) >= 2:
            current_wave = str(waves[0]).lower()
            prev_wave = str(waves[1]).lower()
            if prev_wave == 'accumulation' and current_wave == 'markup':
                candle_info['wave_transition'] = 'accumulation_to_markup'
            elif prev_wave == 'distribution' and current_wave == 'markdown':
                candle_info['wave_transition'] = 'distribution_to_markdown'
        
        # 지지/저항 근처 여부 (RSI 기반 간접 추정)
        rsi = df['rsi'].iloc[0] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[0]) else 50
        candle_info['near_support'] = rsi < 35
        candle_info['near_resistance'] = rsi > 65
        
    except Exception:
        pass
    
    return candle_info


def _calculate_strategy_target_price(current_price: float, strategy_type: str, 
                                      is_long: bool = True, volatility: float = 0.02) -> float:
    """
    🆕 전략별 목표가 계산
    
    각 전략의 take_profit_pct를 기반으로 목표가 계산
    - 전략마다 다른 수익 목표를 가짐 (scalp: 1.5%, bottom: 30%)
    - 변동성과 함께 고려하여 현실적인 목표가 설정
    
    Args:
        current_price: 현재가
        strategy_type: 전략 유형
        is_long: 롱 포지션 여부
        volatility: 변동성 (ATR 기반)
    
    Returns:
        계산된 목표가
    """
    if current_price <= 0:
        return 0.0
    
    try:
        from trade.core.strategies import get_exit_rules
        exit_rules = get_exit_rules(strategy_type)
        take_profit_pct = exit_rules.take_profit_pct
    except (ImportError, AttributeError):
        # 폴백: 기본 2% 목표
        take_profit_pct = 2.0
    
    # 변동성을 고려한 보정 (변동성이 높으면 목표 상향 가능)
    volatility_pct = volatility * 100  # 0.02 -> 2%
    
    # 최소 목표는 전략 기본값, 변동성이 높으면 추가
    # 단, 스캘핑은 변동성 무관하게 고정 목표 유지
    if strategy_type == 'scalp':
        effective_target_pct = take_profit_pct
    else:
        # 변동성이 높으면 목표 약간 상향 (최대 20% 추가)
        vol_bonus = min(volatility_pct * 0.5, take_profit_pct * 0.2)
        effective_target_pct = take_profit_pct + vol_bonus
    
    # 목표가 계산
    if is_long:
        target_price = current_price * (1 + effective_target_pct / 100.0)
    else:
        target_price = current_price * (1 - effective_target_pct / 100.0)
    
    return round(target_price, 2)


def _calculate_strategy_scores_for_signal(signal, coin_data: pd.DataFrame, 
                                          interval: str, db_path: str = None,
                                          regime: str = None) -> dict:
    """
    시그널에 대해 모든 전략 점수 계산 (레짐 반영)
    
    Args:
        signal: 시그널 객체
        coin_data: 캔들 데이터
        interval: 인터벌
        db_path: DB 경로 (미사용)
        regime: 🆕 시장 레짐 (전략-레짐 호환성 적용)
    """
    if not STRATEGY_SYSTEM_AVAILABLE:
        return {}
    
    try:
        # 시그널 데이터 준비
        signal_data = {
            'rsi': getattr(signal, 'rsi', 50),
            'macd': getattr(signal, 'macd', 0),
            'adx': getattr(signal, 'adx', 25),
            'volume_ratio': getattr(signal, 'volume_ratio', 1.0),
            'wave_phase': getattr(signal, 'wave_phase', 'unknown'),
            'pattern_type': getattr(signal, 'pattern_type', 'unknown'),
            'integrated_direction': getattr(signal, 'integrated_direction', 'neutral'),
            'signal_continuity': getattr(signal, 'signal_continuity', 0.5),
            'interval': interval,
            'signal_score': signal.signal_score,
            'pattern': getattr(signal, 'pattern', 'unknown'),
        }
        
        # 캔들 데이터에서 추가 정보 추출
        candle_info = _prepare_candle_data_for_strategy(coin_data, interval)
        
        # 모든 전략 점수 계산
        strategy_scores = evaluate_all_strategies(signal_data, candle_info)
        
        # 🆕 레짐 기반 전략 점수 보정
        if regime and strategy_scores:
            try:
                from trade.core.strategies import get_regime_adjustment, get_strategy_regime_compatibility
                
                for strat_type in strategy_scores:
                    if isinstance(strategy_scores[strat_type], dict) and 'match' in strategy_scores[strat_type]:
                        # 레짐 조정 계수 적용
                        regime_adj = get_regime_adjustment(strat_type, regime)
                        original_match = strategy_scores[strat_type]['match']
                        
                        # 조정된 점수 계산 (0.1 ~ 1.0 범위 유지)
                        adjusted_match = original_match * regime_adj
                        adjusted_match = max(0.1, min(1.0, adjusted_match))
                        
                        strategy_scores[strat_type]['match'] = round(adjusted_match, 3)
                        strategy_scores[strat_type]['regime_adj'] = round(regime_adj, 2)
                        
                        # 호환성 정보 추가
                        compat_score, compat_desc = get_strategy_regime_compatibility(strat_type, regime)
                        strategy_scores[strat_type]['regime_compat'] = round(compat_score, 2)
                        
            except ImportError:
                pass  # 레짐 함수 없으면 무시
        
        return strategy_scores
        
    except Exception as e:
        print(f"⚠️ 전략 점수 계산 오류: {e}")
        return {}

def main():
    """🚀 I/O 병목이 제거된 고성능 시그널 엔진 (순차 방식 + GPU 최적화)"""
    
    # 0. 🔥 [Critical] 기준 시각 설정 (DB 최신 캔들 기준)
    try:
        from trade.core.database import get_latest_candle_timestamp
        db_now = get_latest_candle_timestamp()
    except:
        db_now = int(time.time())
    
    print(f"TIME: 엔진 기준 시각 (DB): {db_now} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(db_now))})")

    # 1. 시스템 초기화
    load_start = time.time()
    try:
        selector = get_signal_selector()
    except Exception as e:
        print(f"WARN: SignalSelector 초기화 중 일부 오류 (연산은 계속 진행): {e}")
        # 초기화 실패 시 폴백 (중요 테이블만이라도 사용 가능하게)
        from signal_selector.core.selector import SignalSelector
        selector = SignalSelector()

    if not selector:
        print("ERROR: SignalSelector 초기화 실패")
        return

    # 🚀 [Stability] 시그널 저장 전용 DB 연결 (WAL 모드 및 타임아웃 1분)
    try:
        from trade.core.database import get_db_connection
        # 🆕 기존에 열려있는 연결이 있다면 닫고 새로 열기 (잠금 해제)
        write_conn = get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False)
    except Exception as e:
        print(f"WARN: 시그널 DB 연결 오류 (폴백 시도): {e}")
        write_conn = sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=60.0)
        write_conn.execute("PRAGMA journal_mode=WAL")
        write_conn.execute("PRAGMA busy_timeout=60000")

    print(f"INFO: 시스템 로딩 완료 ({time.time() - load_start:.1f}s) | GPU 가속: {HAS_JAX}")
    
    # 🚀 [I/O Optimization] 최근 캔들 데이터 고속 일괄 로드
    print("DATA: 모든 코인 최신 데이터 일괄 로딩 중...")
    try:
        # 🚀 읽기 전용 연결 사용 (CANDLES_DB_PATH)
        from trade.core.database import get_db_connection
        read_conn = get_db_connection(CANDLES_DB_PATH, read_only=True)

        with read_conn as conn:
            # 최신 10만건을 가져와서 메모리에서 분석 (I/O 병목 제거의 핵심)
            # 🚀 [Fix] 모든 기술지표 컬럼 포함하여 로드 (N/A 방지)
            # 🆕 [5-Candle Analysis] low, high 컬럼 추가
            all_data_df = pd.read_sql("""
                SELECT symbol, interval, timestamp, close, open, high, low, 
                       rsi, macd, adx, volume_ratio, wave_phase, pattern_type, 
                       integrated_direction, regime_label
                FROM candles 
                ORDER BY timestamp DESC 
                LIMIT 100000
                """, conn)
        print(f"INFO: {len(all_data_df):,}행 데이터 로드 완료")
    except Exception as e:
        print(f"ERROR: 데이터 일괄 로드 실패: {e}")
        return

    # 메모리 내 데이터 그룹화 (분석 대상 필터링)
    coin_intervals_map = defaultdict(list)
    # 🚀 [성능] iterrows → to_dict('records') 최적화
    for row in all_data_df.drop_duplicates(['symbol', 'interval']).to_dict('records'):
        coin_intervals_map[row['symbol']].append(row['interval'])

    loop_start_time = time.time()
    all_signals_to_save = []
    # 🚀 [Fix] 코인 순서를 알파벳 순으로 정렬하여 가시성 확보
    symbols = sorted(list(coin_intervals_map.keys()))
    total_symbols = len(symbols)
    
    print(f"START: 시그널 연산 시작 (순차 실행, 대상 코인: {total_symbols}개)")

    # 🆕 [레짐 기반 전략] 시장 레짐 정보 로드 + 변화 감지
    market_regime = 'neutral'
    regime_changed = False
    recommended_strategies = []
    try:
        from trade.trade_executor import get_market_context
        market_context = get_market_context()
        market_regime = market_context.get('regime', 'neutral')
        market_score = market_context.get('score', 0.5)
        
        # 🆕 레짐 변화 감지
        try:
            from trade.core.strategies import get_regime_detector
            detector = get_regime_detector()
            should_reevaluate, reason = detector.should_reevaluate_strategies(market_regime)
            stability, stability_desc = detector.get_regime_stability()
            
            if should_reevaluate:
                regime_changed = True
                recommended_strategies = detector.get_recommended_strategies_for_regime(market_regime)
                print(f"🔄 {reason}")
                print(f"   📋 추천 전략: {', '.join(recommended_strategies[:3])}")
            
            print(f"📊 시장 레짐: {market_regime.upper()} (점수: {market_score:.2f}, 안정성: {stability:.1f}) - 전략 점수에 반영됨")
        except Exception as det_err:
            print(f"📊 시장 레짐: {market_regime.upper()} (점수: {market_score:.2f}) - 전략 점수에 반영됨")
    except Exception as e:
        print(f"⚠️ 시장 레짐 로드 실패 (기본값 사용): {e}")

    # 🚀 [Performance] 메모리 캐시를 활용한 고속 순차 연산
    for i, coin in enumerate(symbols):
        try:
            interval_signals = {}
            coin_data = all_data_df[all_data_df['symbol'] == coin]
            
            for iv in coin_intervals_map[coin]:
                # save=False로 설정하여 루프 내 DB 쓰기 병목 방지
                sig = selector.generate_signal(coin, iv, save=False)
                if sig:
                    sig.timestamp = db_now # 캔들 시각으로 강제 동기화
                    
                    # 🆕 [5-Candle Sequence Analysis] 적용
                    iv_data = coin_data[coin_data['interval'] == iv]
                    if len(iv_data) >= 5:
                        analysis = SequenceAnalyzer.analyze_sequence(iv_data, iv)
                        if analysis['score_mod'] != 1.0:
                            old_score = sig.signal_score
                            sig.signal_score *= analysis['score_mod']
                            sig.reason += f" | 🌊 흐름분석: {analysis['reason']} (보정 {old_score:.3f} -> {sig.signal_score:.3f})"
                    
                    # 🆕 [디버깅] 각 인터벌별 실제 캔들 지표 값 출력 (핵심 지표 전체 표시)
                    if len(iv_data) > 0:
                        latest = iv_data.iloc[0]
                        
                        # 🔥 핵심 지표 추출 (모든 지표 활용 확인용)
                        rsi_val = latest.get('rsi', 'N/A')
                        close_val = latest.get('close', 'N/A')
                        macd_val = latest.get('macd', 'N/A')
                        volume_ratio = latest.get('volume_ratio', 'N/A')
                        wave_val = latest.get('wave_phase', 'N/A')
                        pattern_val = latest.get('pattern_type', 'N/A')
                        direction_val = latest.get('integrated_direction', 'N/A')
                        adx_val = latest.get('adx', 'N/A')
                        
                        # 시그널 객체에 지표 값 저장 (전략 계산용)
                        sig.rsi = rsi_val if isinstance(rsi_val, (int, float)) and not pd.isna(rsi_val) else 50
                        sig.macd = macd_val if isinstance(macd_val, (int, float)) and not pd.isna(macd_val) else 0
                        sig.adx = adx_val if isinstance(adx_val, (int, float)) and not pd.isna(adx_val) else 25
                        sig.volume_ratio = volume_ratio if isinstance(volume_ratio, (int, float)) and not pd.isna(volume_ratio) else 1.0
                        sig.wave_phase = wave_val if wave_val and wave_val != 'N/A' else 'unknown'
                        sig.pattern_type = pattern_val if pattern_val and pattern_val != 'N/A' else 'unknown'
                        sig.integrated_direction = direction_val if direction_val and direction_val != 'N/A' else 'neutral'
                        
                        # 숫자 타입일 때만 포맷팅
                        def fmt_num(val, decimals=1):
                            if isinstance(val, (int, float)) and not pd.isna(val):
                                return f"{val:.{decimals}f}"
                            return str(val) if val else 'N/A'
                        
                        rsi_str = fmt_num(rsi_val, 1)
                        close_str = f"{close_val:,.0f}" if isinstance(close_val, (int, float)) and not pd.isna(close_val) else str(close_val)
                        macd_str = fmt_num(macd_val, 4)
                        vol_str = f"{volume_ratio:.2f}x" if isinstance(volume_ratio, (int, float)) and not pd.isna(volume_ratio) else str(volume_ratio)
                        adx_str = fmt_num(adx_val, 1)
                        
                        # 🔥 핵심 지표 전체 로그 출력 (실제 점수 계산 근거)
                        print(f"   📈 {coin}/{iv}: RSI={rsi_str}, MACD={macd_str}, ADX={adx_str}, Vol={vol_str}, Wave={wave_val}, Pattern={pattern_val}, Dir={direction_val} -> 점수 {sig.signal_score:.3f}")
                    
                    # 🆕 [전략 시스템] 인터벌별 전략 점수 계산 (레짐 반영)
                    if STRATEGY_SYSTEM_AVAILABLE:
                        strategy_scores = _calculate_strategy_scores_for_signal(sig, iv_data, iv, regime=market_regime)
                        if strategy_scores:
                            sig.strategy_scores = strategy_scores
                            
                            # 🧬 [진화 시스템] 진화 레벨 기반 전략 선택
                            best_strategy = None
                            evolution_level = 1
                            evolved_params = {}
                            
                            if EVOLUTION_SYSTEM_AVAILABLE:
                                try:
                                    signal_data = {'strategy_scores': strategy_scores}
                                    best_strategy, evolution_level, evolved_params = get_best_evolved_strategy(
                                        signal_data, market_regime
                                    )
                                except Exception as evo_err:
                                    pass  # 폴백: 기본 전략 사용
                            
                            # 폴백: 기본 전략 점수 기반 선택
                            if not best_strategy:
                                top_strats = get_top_strategies(strategy_scores, top_n=2, min_match=0.3)
                                if top_strats:
                                    best_strategy = top_strats[0]['strategy']
                            
                            if best_strategy:
                                # 진화 레벨 표시
                                level_emoji = {1: "📘", 2: "📗", 3: "🤖", 4: "🧬"}.get(evolution_level, "📘")
                                top_strats = get_top_strategies(strategy_scores, top_n=2, min_match=0.3)
                                strat_str = ', '.join([f"{s['strategy']}({s['match']:.2f})" for s in top_strats]) if top_strats else best_strategy
                                print(f"      {level_emoji} {coin}/{iv} 전략: {strat_str} (Lv.{evolution_level})")
                                
                                # 🆕 전략 기반 목표가 계산
                                current_price = getattr(sig, 'price', 0) or (close_val if isinstance(close_val, (int, float)) and not pd.isna(close_val) else 0)
                                volatility = getattr(sig, 'volatility', 0.02) or 0.02
                                is_long = str(getattr(sig, 'action', 'buy')).lower() in ['buy', 'long']
                                
                                if current_price > 0:
                                    # 진화된 파라미터가 있으면 사용
                                    if evolved_params.get('take_profit_pct'):
                                        target_pct = evolved_params['take_profit_pct']
                                        sig.target_price = current_price * (1 + target_pct / 100) if is_long else current_price * (1 - target_pct / 100)
                                    else:
                                        sig.target_price = _calculate_strategy_target_price(
                                            current_price=current_price,
                                            strategy_type=best_strategy,
                                            is_long=is_long,
                                            volatility=volatility
                                        )
                                    
                                    sig.recommended_strategy = best_strategy
                                    sig.strategy_match = top_strats[0]['match'] if top_strats else 0.5
                                    
                                    # 🧬 진화 정보 저장
                                    sig.evolution_level = evolution_level
                                    sig.evolved_params = evolved_params
                                    
                                    # 목표 수익률 출력
                                    expected_pct = ((sig.target_price - current_price) / current_price * 100) if current_price > 0 else 0
                                    print(f"      💰 {coin}/{iv} 목표가: {sig.target_price:,.0f}원 ({expected_pct:+.2f}%)")
                    
                    interval_signals[iv] = sig
                    all_signals_to_save.append(sig)
            
            # 멀티 인터벌 통합 시그널
            if len(interval_signals) >= 2:
                # 🆕 [디버깅] 각 인터벌별 실제 점수 + 신뢰도 출력 (통합 전)
                interval_details = []
                for iv, sig in sorted(interval_signals.items()):
                    # 신뢰도와 패턴 신뢰도도 함께 표시 (동적 영향도 요소)
                    conf = getattr(sig, 'confidence', 0.5)
                    pattern_conf = getattr(sig, 'pattern_confidence', 0.0)
                    interval_details.append(f"{iv}:{sig.signal_score:.3f}(신뢰:{conf:.2f})")
                print(f"📊 {coin} 인터벌별 점수: {' | '.join(interval_details)}")
                
                combined_sig = selector.combine_multi_timeframe_signals(coin, interval_signals, save=False)
                if combined_sig:
                    combined_sig.timestamp = db_now # 캔들 시각으로 강제 동기화
                    
                    # 🆕 [전략 시스템] 통합 시그널에 전략 점수 계산
                    if STRATEGY_SYSTEM_AVAILABLE:
                        # 인터벌별 전략 점수 통합 (가중 평균)
                        combined_strategy_scores = {}
                        interval_weights = {'15m': 0.20, '30m': 0.25, '240m': 0.30, '1d': 0.25}
                        
                        for iv, sig in interval_signals.items():
                            if hasattr(sig, 'strategy_scores') and sig.strategy_scores:
                                weight = interval_weights.get(iv, 0.2)
                                for strat_type, score_data in sig.strategy_scores.items():
                                    if strat_type not in combined_strategy_scores:
                                        combined_strategy_scores[strat_type] = {'match': 0.0, 'weight_sum': 0.0}
                                    combined_strategy_scores[strat_type]['match'] += score_data['match'] * weight
                                    combined_strategy_scores[strat_type]['weight_sum'] += weight
                        
                        # 가중 평균 계산
                        for strat_type in combined_strategy_scores:
                            weight_sum = combined_strategy_scores[strat_type]['weight_sum']
                            if weight_sum > 0:
                                combined_strategy_scores[strat_type] = {
                                    'match': round(combined_strategy_scores[strat_type]['match'] / weight_sum, 3),
                                    'strategy': strat_type
                                }
                        
                        combined_sig.strategy_scores = combined_strategy_scores
                        
                        # 상위 전략 출력
                        top_strats = get_top_strategies(combined_strategy_scores, top_n=3, min_match=0.25)
                        if top_strats:
                            strat_str = ', '.join([f"{s['strategy']}({s['match']:.2f})" for s in top_strats])
                            print(f"   🎯 {coin}/combined 추천 전략: {strat_str}")
                            
                            # 최적 전략을 시그널에 저장
                            combined_sig.recommended_strategy = top_strats[0]['strategy']
                            combined_sig.strategy_match = top_strats[0]['match']
                            
                            # 🆕 통합 시그널 목표가 계산 (전략 기반)
                            current_price = getattr(combined_sig, 'price', 0)
                            volatility = getattr(combined_sig, 'volatility', 0.02) or 0.02
                            is_long = str(getattr(combined_sig, 'action', 'buy')).lower() in ['buy', 'long']
                            
                            if current_price > 0:
                                combined_sig.target_price = _calculate_strategy_target_price(
                                    current_price=current_price,
                                    strategy_type=top_strats[0]['strategy'],
                                    is_long=is_long,
                                    volatility=volatility
                                )
                                expected_pct = ((combined_sig.target_price - current_price) / current_price * 100) if current_price > 0 else 0
                                print(f"   💰 {coin}/combined 목표가: {combined_sig.target_price:,.0f}원 ({expected_pct:+.2f}%, {top_strats[0]['strategy']})")
                    
                    all_signals_to_save.append(combined_sig)
                    # 🆕 통합 시그널 로깅 추가 (동적 영향도 기반 최종 결과)
                    rec_strat = getattr(combined_sig, 'recommended_strategy', 'trend')
                    target_info = f", 목표가: {combined_sig.target_price:,.0f}원" if getattr(combined_sig, 'target_price', 0) > 0 else ""
                    print(f"🔗 COMBINED: {coin}/combined: 최종 통합 점수 {combined_sig.signal_score:.3f} | 신뢰도: {combined_sig.confidence:.2f} | 액션: {combined_sig.action.value} | 전략: {rec_strat}{target_info}")
            
            # 진행 상황 출력
            if (i + 1) % 50 == 0 or (i + 1) == total_symbols:
                elapsed = time.time() - loop_start_time
                cps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"PROGRESS: 연산 중... ({i+1}/{total_symbols}) | 속도: {cps:.1f} coin/s")
                
        except Exception as e:
            print(f"ERROR: {coin} 연산 중 오류 발생: {e}")

    # 🚀 [I/O Optimization] 결과값 최종 일괄 저장 (Batch Write)
    if all_signals_to_save:
        save_start = time.time()
        selector.save_signals_batch(all_signals_to_save)
        print(f"SAVE: {len(all_signals_to_save)}개 시그널 일괄 저장 완료 ({time.time() - save_start:.2f}s)")

    loop_elapsed = time.time() - loop_start_time
    print(f"DONE: 시그널 업데이트 완료 (총 소요: {loop_elapsed:.1f}s)")
    
    # =========================================================================
    # 🆕 [자가 진화] 증분 검증 - 과거 시그널 예측 정확도 학습
    # =========================================================================
    # - 시그널 생성 후 실행 (매매 성능에 영향 최소화)
    # - 이미 로드된 캔들 데이터 활용 (추가 I/O 없음)
    # - 미검증 시그널만 대상 (점진적으로 빨라짐)
    # =========================================================================
    print("\n" + "="*60)
    print("🧬 [자가 진화] 시그널 예측 검증 및 학습 시작...")
    print("="*60)
    
    validation_start = time.time()
    global_db = os.environ.get('GLOBAL_STRATEGY_DB_PATH')
    
    validation_stats = validate_signals_incremental(
        all_data_df=all_data_df,
        db_now=db_now,
        trading_db_path=TRADING_SYSTEM_DB_PATH,
        candles_db_path=CANDLES_DB_PATH,
        global_db_path=global_db
    )
    
    validation_elapsed = time.time() - validation_start
    print(f"🧬 [자가 진화] 검증 완료 (소요: {validation_elapsed:.1f}s)")
    print(f"   📊 통계: 검사 {validation_stats.get('total_checked', 0)}개 → 검증 {validation_stats.get('validated', 0)}개, 대기 {validation_stats.get('skipped_pending', 0)}개")
    
    total_elapsed = time.time() - load_start
    print(f"\n🏁 전체 사이클 완료 (총 소요: {total_elapsed:.1f}s)")

if __name__ == "__main__":
    main()
