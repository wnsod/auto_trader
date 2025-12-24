"""
데이터베이스 쓰기 전용 커넥션 및 배치 처리
단일 writer 큐로 배치 insert/update, 트랜잭션 관리

핵심 설계:
- coin → symbol 매핑
- market_type, market 컬럼 추가
- 테이블명 범용화 (strategies → strategies)
"""

import sqlite3
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime
from rl_pipeline.db.connection_pool import get_strategy_db_pool, get_batch_loading_pool
from rl_pipeline.core.errors import DBWriteError
from rl_pipeline.core.utils import safe_json_dumps, _format_decimal_precision

logger = logging.getLogger(__name__)

# ============================================================================
# 상수 정의
# ============================================================================

DEFAULT_MARKET_TYPE = "COIN"
DEFAULT_MARKET = "BITHUMB"


# ============================================================================
# 매핑 유틸리티
# ============================================================================

def _map_coin_to_symbol(row: Dict[str, Any]) -> Dict[str, Any]:
    """coin 컬럼을 symbol로 매핑하고 market_type, market 추가"""
    result = row.copy()

    # coin → symbol 매핑
    if 'coin' in result and 'symbol' not in result:
        result['symbol'] = result.pop('coin')

    # market_type, market 기본값 추가
    if 'market_type' not in result:
        result['market_type'] = DEFAULT_MARKET_TYPE
    if 'market' not in result:
        result['market'] = DEFAULT_MARKET

    return result


def _sanitize_strategy_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """전략 파라미터 현실화 (과도한 조건 자동 완화)"""
    sanitized = row.copy()
    try:
        stype = str(sanitized.get('strategy_type', '')).lower()
        is_sell_strategy = 'sell' in stype and 'buy' not in stype

        def _safe_float(value, default=None):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        vol_min = _safe_float(sanitized.get('volume_ratio_min'))
        vol_max = _safe_float(sanitized.get('volume_ratio_max'))
        if vol_min is not None:
            upper = 2.0 if is_sell_strategy else 3.0
            clamped = min(max(0.4, vol_min), upper)
            if clamped != vol_min and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📉 volume_ratio_min 조정: {vol_min:.3f} → {clamped:.3f} (strategy={stype})")
            sanitized['volume_ratio_min'] = clamped
            vol_min = clamped
        if vol_max is not None:
            upper_max = 3.2 if is_sell_strategy else 4.0
            clamped_max = min(max(vol_max, (vol_min + 0.2) if vol_min is not None else 0.6), upper_max)
            if clamped_max != vol_max and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📉 volume_ratio_max 조정: {vol_max:.3f} → {clamped_max:.3f} (strategy={stype})")
            sanitized['volume_ratio_max'] = clamped_max
        elif vol_min is not None:
            sanitized['volume_ratio_max'] = vol_min + 0.2

        rsi_max = _safe_float(sanitized.get('rsi_max'))
        if rsi_max is not None:
            max_cap = 78.0 if is_sell_strategy else 85.0
            clamped_rsi_max = min(max(rsi_max, 55.0), max_cap)
            if clamped_rsi_max != rsi_max and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📉 rsi_max 조정: {rsi_max:.2f} → {clamped_rsi_max:.2f} (strategy={stype})")
            sanitized['rsi_max'] = clamped_rsi_max

        rsi_min = _safe_float(sanitized.get('rsi_min'))
        if rsi_min is not None and rsi_min >= sanitized.get('rsi_max', rsi_min + 5):
            sanitized['rsi_min'] = sanitized['rsi_max'] - 5.0

        return sanitized
    except Exception as exc:
        logger.warning(f"⚠️ 전략 파라미터 정규화 실패: {exc}")
        return row


def _map_rows_to_schema(rows: List[Dict[str, Any]], table: str) -> List[Dict[str, Any]]:
    """rows 리스트를 스키마에 맞게 매핑"""
    # 매핑 필요한 테이블 목록
    mapped_tables = {
        'strategies', 'strategy_performance_rl', 'strategy_grades',
        'rl_episodes', 'rl_episode_summary', 'rl_state_ensemble',
        'global_strategies', 'analysis_ratios', 'symbol_global_weights',
        'runs', 'run_records', 'pipeline_execution_logs',
        'integrated_analysis_results', 'strategy_training_history'
    }

    # 호환성 뷰 (실제 테이블명으로 변환)
    table_mapping = {
        'strategies': 'strategies',
        'rl_strategy_rollup': 'strategy_performance_rl',
        'coin_analysis_ratios': 'analysis_ratios',
        'coin_global_weights': 'symbol_global_weights'
    }

    # 테이블명 변환
    actual_table = table_mapping.get(table, table)

    # 매핑이 필요한 테이블이면 적용
    if actual_table in mapped_tables:
        mapped = [_map_coin_to_symbol(row) for row in rows]
        if actual_table == 'strategies':
            # strategy_id 키가 없고 id가 있으면 strategy_id로 변환 (DB 스키마가 strategy_id가 아닌 id를 사용하므로 매핑 로직 수정 필요)
            # ⚠️ 실제 strategies 테이블 스키마는 id TEXT PRIMARY KEY로 되어 있음.
            # 따라서 strategy_id -> id로 변환해야 함.
            def _map_strategy_id(r):
                new_r = r.copy()
                # strategy_id 키가 있으면 처리
                if 'strategy_id' in new_r:
                    # id 키가 없으면 strategy_id 값을 id로 이동
                    if 'id' not in new_r:
                        new_r['id'] = new_r['strategy_id']
                    # strategy_id 키는 무조건 제거 (테이블에 없는 컬럼이므로)
                    del new_r['strategy_id']
                return new_r
            
            mapped = [_map_strategy_id(row) for row in mapped]
            mapped = [_sanitize_strategy_row(row) for row in mapped]
        return mapped

    return rows

def execute_query(query: str, params: tuple = (), db_path: str = "strategies") -> bool:
    """쿼리 실행"""
    try:
        pool = get_strategy_db_pool() if db_path == "strategies" else get_batch_loading_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"❌ 쿼리 실행 실패: {e}")
        return False

def write_batch(rows: List[Dict[str, Any]], table: str, db_path: str = None, checkpoint: bool = False, verify: bool = False, max_retries: int = 5) -> int:
    """배치 쓰기
    Args:
        rows: 삽입/업데이트할 행 목록
        table: 대상 테이블명
        db_path: 선택적 DB 경로 (없으면 전략 DB 사용)
        checkpoint: 쓰기 이후 WAL 체크포인트를 강제 수행할지 여부 (기본 False)
        verify: 쓰기 직후 COUNT 검증을 수행할지 여부 (기본 False)
        max_retries: 최대 재시도 횟수
    """
    if not rows:
        return 0

    import time
    import random

    for attempt in range(max_retries):
        try:
            if db_path:
                pool = get_batch_loading_pool(db_path)
            else:
                pool = get_strategy_db_pool()

            # 테이블명 매핑
            table_mapping = {
                'strategies': 'strategies',
                'rl_strategy_rollup': 'strategy_performance_rl',
                'coin_analysis_ratios': 'analysis_ratios',
                'coin_global_weights': 'symbol_global_weights'
            }
            actual_table = table_mapping.get(table, table)

            # 스키마 매핑 (coin → symbol, market_type, market 추가)
            mapped_rows = _map_rows_to_schema(rows, table)

            with pool.get_connection() as conn:
                cursor = conn.cursor()

                # 첫 번째 행으로 컬럼 정보 추출
                columns = list(mapped_rows[0].keys())
                placeholders = ', '.join(['?' for _ in columns])
                query = f"INSERT OR REPLACE INTO {actual_table} ({', '.join(columns)}) VALUES ({placeholders})"
                
                # 배치 실행
                batch_data = []
                for row in mapped_rows:
                    values = []
                    for col in columns:
                        value = row.get(col)
                        # 특정 필드는 포맷팅 적용
                        if col in ['profit', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'profit_factor']:
                            value = _format_decimal_precision(value, col)
                        elif isinstance(value, dict):
                            value = safe_json_dumps(value)
                        values.append(value)
                    batch_data.append(tuple(values))
                
                cursor.executemany(query, batch_data)
                logger.debug(f"🔍 executemany 실행 완료: {len(batch_data)}개 행")

                conn.commit()
                logger.debug(f"🔍 커밋 완료")

                # 고비용 동작은 선택적으로 수행
                if checkpoint:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    logger.debug(f"🔍 WAL 체크포인트 완료 (PASSIVE)")

                if verify:
                    cursor.execute(f"SELECT COUNT(*) FROM {actual_table}")
                    count = cursor.fetchone()[0]
                    logger.info(f"🔍 즉시 확인: {actual_table} 테이블에 {count}개 레코드 존재")

                logger.info(f"✅ 배치 쓰기 완료: {len(mapped_rows)}행 -> {actual_table}")
                return len(mapped_rows)
                
        except Exception as e:
            is_locked = "database is locked" in str(e) or "disk I/O error" in str(e) or "attempt to write a readonly database" in str(e)
            
            if is_locked and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.random()
                logger.warning(f"⚠️ DB 쓰기 일시적 실패 ({attempt+1}/{max_retries}), {wait_time:.2f}초 후 재시도: {e}")
                time.sleep(wait_time)
                
                # 커넥션 풀 리셋 시도 (readonly 에러 등 대응)
                try:
                    if db_path:
                        get_batch_loading_pool(db_path).close_all_connections()
                    else:
                        get_strategy_db_pool().close_all_connections()
                except:
                    pass
            else:
                logger.error(f"❌ 배치 쓰기 최종 실패: {e}")
                raise DBWriteError(f"배치 쓰기 실패 ({table}): {e}") from e

def upsert(data: Dict[str, Any], table: str, key_columns: List[str], db_path: str = None) -> bool:
    """Upsert (INSERT OR REPLACE)"""
    try:
        if db_path:
            pool = get_strategy_db_pool(db_path)
        else:
            pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            columns = list(data.keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = []
            for col in columns:
                value = data.get(col)
                if col in ['profit', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'profit_factor']:
                    value = _format_decimal_precision(value, col)
                elif isinstance(value, dict):
                    value = safe_json_dumps(value)
                values.append(value)
            
            cursor.execute(query, tuple(values))
            conn.commit()
            
            logger.debug(f"✅ Upsert 완료: {table}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Upsert 실패: {e}")
        raise DBWriteError(f"Upsert 실패 ({table}): {e}") from e

def update_strategy_performance(strategy_id: str, performance_data: Dict[str, Any]) -> bool:
    """전략 성과 업데이트"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 성과 데이터 포맷팅
            formatted_data = {}
            for key, value in performance_data.items():
                if key in ['profit', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'profit_factor']:
                    formatted_data[key] = _format_decimal_precision(value, key)
                else:
                    formatted_data[key] = value
            
            # 업데이트 쿼리 생성
            set_clauses = []
            values = []
            for key, value in formatted_data.items():
                set_clauses.append(f"{key} = ?")
                values.append(value)
            
            values.append(strategy_id)
            # strategies → strategies
            query = f"UPDATE strategies SET {', '.join(set_clauses)} WHERE id = ?"
            
            cursor.execute(query, tuple(values))
            conn.commit()
            
            logger.debug(f"✅ 전략 성과 업데이트 완료: {strategy_id}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 전략 성과 업데이트 실패: {e}")
        raise DBWriteError(f"전략 성과 업데이트 실패: {e}") from e

def save_strategy_dna(coin: str, dna_data: Dict[str, Any]) -> bool:
    """전략 DNA 저장"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # DNA 데이터 준비 (interval 추출)
            interval = dna_data.get('interval')
            
            dna_record = {
                'coin': coin,
                'interval': interval,
                'dna_patterns': safe_json_dumps(dna_data.get('dna_patterns', {})),
                'dna_data': safe_json_dumps(dna_data),
                'created_at': pd.Timestamp.now().isoformat(),
                'quality_score': dna_data.get('quality_score', 0.0)
            }
            
            # Upsert 실행
            columns = list(dna_record.keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT OR REPLACE INTO strategy_dna ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = [dna_record[col] for col in columns]
            cursor.execute(query, tuple(values))
            conn.commit()
            
            logger.info(f"✅ 전략 DNA 저장 완료: {coin} {interval}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 전략 DNA 저장 실패: {e}")
        raise DBWriteError(f"전략 DNA 저장 실패: {e}") from e

def save_fractal_analysis(coin: str, interval: str, fractal_data: Dict[str, Any]) -> bool:
    """프랙탈 분석 결과 저장 - 개선된 버전 (모든 컬럼 포함)"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 프랙탈 데이터 준비 (모든 컬럼 포함)
            fractal_record = {
                'coin': coin,
                'interval': interval,
                'analysis_type': fractal_data.get('analysis_type', 'fractal_pattern'),
                'fractal_score': _format_decimal_precision(fractal_data.get('fractal_score', 0.0), 'fractal_score'),
                'pattern_distribution': safe_json_dumps(fractal_data.get('pattern_distribution', {})),
                'pruned_strategies_count': fractal_data.get('pruned_strategies_count', 0),
                'total_strategies': fractal_data.get('total_strategies', 0),
                'avg_profit': fractal_data.get('avg_profit', 0.0),
                'avg_win_rate': fractal_data.get('avg_win_rate', 0.0),
                'optimal_rsi_min': fractal_data.get('optimal_rsi_min', 30.0),
                'optimal_rsi_max': fractal_data.get('optimal_rsi_max', 70.0),
                'optimal_volume_ratio': fractal_data.get('optimal_volume_ratio', 1.0),
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            # Upsert 실행
            columns = list(fractal_record.keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT OR REPLACE INTO fractal_analysis ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = [fractal_record[col] for col in columns]
            cursor.execute(query, tuple(values))
            conn.commit()
            
            logger.info(f"✅ 프랙탈 분석 저장 완료: {coin} {interval} (전체 컬럼 포함)")
            return True
            
    except Exception as e:
        logger.error(f"❌ 프랙탈 분석 저장 실패: {e}")
        raise DBWriteError(f"프랙탈 분석 저장 실패: {e}") from e

def save_synergy_analysis(coin: str, interval: str, synergy_data: Dict[str, Any]) -> bool:
    """시너지 분석 결과 저장"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 시너지 데이터 준비
            synergy_record = {
                'coin': coin,
                'interval': interval,
                'synergy_score': _format_decimal_precision(synergy_data.get('synergy_score', 0.0), 'synergy_score'),
                'synergy_patterns': safe_json_dumps(synergy_data.get('synergy_patterns', {})),
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            # Upsert 실행
            columns = list(synergy_record.keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT OR REPLACE INTO synergy_analysis ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = [synergy_record[col] for col in columns]
            cursor.execute(query, tuple(values))
            conn.commit()
            
            logger.info(f"✅ 시너지 분석 저장 완료: {coin} {interval}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 시너지 분석 저장 실패: {e}")
        raise DBWriteError(f"시너지 분석 저장 실패: {e}") from e

def save_run_metadata(run_metadata: Dict[str, Any],
                     market_type: str = DEFAULT_MARKET_TYPE,
                     market: str = DEFAULT_MARKET) -> bool:
    """실행 메타데이터 저장 (coin → symbol 매핑)"""
    try:
        pool = get_strategy_db_pool()

        # coin → symbol
        symbol = run_metadata.get('symbol', run_metadata.get('coin'))

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 실행 메타데이터 준비
            run_record = {
                'run_id': run_metadata.get('run_id'),
                'market_type': market_type,
                'market': market,
                'symbol': symbol,
                'interval': run_metadata.get('interval'),
                'start_time': run_metadata.get('start_time'),
                'end_time': run_metadata.get('end_time'),
                'status': run_metadata.get('status', 'running'),
                'strategies_count': run_metadata.get('strategies_count', 0),
                'successful_strategies': run_metadata.get('successful_strategies', 0),
                'error_count': run_metadata.get('error_count', 0)
            }

            # Upsert 실행
            columns = list(run_record.keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT OR REPLACE INTO runs ({', '.join(columns)}) VALUES ({placeholders})"

            values = [run_record[col] for col in columns]
            cursor.execute(query, tuple(values))
            conn.commit()

            logger.info(f"✅ 실행 메타데이터 저장 완료: {run_record['run_id']}")
            return True

    except Exception as e:
        logger.error(f"❌ 실행 메타데이터 저장 실패: {e}")
        raise DBWriteError(f"실행 메타데이터 저장 실패: {e}") from e

def delete_strategies(strategy_ids: List[str]) -> int:
    """전략 삭제"""
    if not strategy_ids:
        return 0
    
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            placeholders = ', '.join(['?' for _ in strategy_ids])
            # strategies → strategies
            query = f"DELETE FROM strategies WHERE id IN ({placeholders})"
            
            cursor.execute(query, tuple(strategy_ids))
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"✅ 전략 삭제 완료: {deleted_count}개")
            return deleted_count
            
    except Exception as e:
        logger.error(f"❌ 전략 삭제 실패: {e}")
        raise DBWriteError(f"전략 삭제 실패: {e}") from e

def cleanup_old_data(table: str, days_to_keep: int = 30, db_path: str = None) -> int:
    """오래된 데이터 정리"""
    try:
        if db_path:
            pool = get_strategy_db_pool()
        else:
            pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            query = f"DELETE FROM {table} WHERE created_at < datetime('now', '-{days_to_keep} days')"
            cursor.execute(query)
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"✅ 오래된 데이터 정리 완료: {table}에서 {deleted_count}행 삭제")
            return deleted_count
            
    except Exception as e:
        logger.error(f"❌ 오래된 데이터 정리 실패: {e}")
        raise DBWriteError(f"오래된 데이터 정리 실패: {e}") from e

# ⚠️ DEPRECATED: 미사용 테이블 관련 함수들 (테이블 제거됨)
# - market_condition_analysis: 레거시, 미사용
# - dna_market_analysis (strategy_dna로 대체)

# def save_market_condition_analysis(coin: str, interval: str, market_condition: str,
#                                   confidence: float, analysis_data: Dict[str, Any]) -> bool:
#     """🔴 DEPRECATED: 시장 상황 분석 결과 저장 - market_condition_analysis 테이블 제거됨"""
#     logger.warning("⚠️ save_market_condition_analysis는 더 이상 사용되지 않습니다 (테이블 제거됨)")
#     return False
# - fractal_market_analysis (fractal_analysis로 대체)
# - routing_market_analysis (regime_routing_results로 대체)

def save_dna_by_market_condition(coin: str, interval: str, market_condition: str, 
                               dna_patterns: Dict[str, Any]) -> bool:
    """⚠️ DEPRECATED: 미사용 테이블 (dna_market_analysis) - 함수는 유지하되 동작 안함"""
    logger.debug(f"⚠️ save_dna_by_market_condition 호출됨 (deprecated): {coin}-{interval}")
    return True  # 테이블이 없으므로 성공으로 반환

def save_fractal_by_market_condition(coin: str, interval: str, market_condition: str, 
                                   fractal_features: Dict[str, Any]) -> bool:
    """⚠️ DEPRECATED: 미사용 테이블 (fractal_market_analysis) - 함수는 유지하되 동작 안함"""
    logger.debug(f"⚠️ save_fractal_by_market_condition 호출됨 (deprecated): {coin}-{interval}")
    return True  # 테이블이 없으므로 성공으로 반환

def save_routing_by_market_condition(coin: str, routing_results: Dict[str, Any], 
                                   integrated_routing: Dict[str, Any]) -> bool:
    """⚠️ DEPRECATED: 미사용 테이블 (routing_market_analysis) - 함수는 유지하되 동작 안함"""
    logger.debug(f"⚠️ save_routing_by_market_condition 호출됨 (deprecated): {coin}")
    return True  # 테이블이 없으므로 성공으로 반환

@contextmanager
def transaction(db_path: str = None):
    """트랜잭션 컨텍스트 매니저"""
    from rl_pipeline.db.connection_pool import get_candle_db_pool
    
    # db_path에 따라 적절한 풀 선택
    if db_path and 'candles' in db_path.lower():
        pool = get_candle_db_pool()
    else:
        pool = get_strategy_db_pool()
    
    with pool.get_connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DBWriteError(f"트랜잭션 실패: {e}") from e

def save_coin_analysis_ratios(coin: str, interval: str, analysis_type: str,
                             ratios_data: Dict[str, Any],
                             market_type: str = DEFAULT_MARKET_TYPE,
                             market: str = DEFAULT_MARKET) -> bool:
    """분석 비율을 데이터베이스에 저장 (coin → symbol 매핑)"""
    try:
        pool = get_strategy_db_pool()

        # coin → symbol
        symbol = coin

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 기존 데이터가 있는지 확인
            check_query = """
            SELECT id FROM analysis_ratios
            WHERE market_type = ? AND market = ? AND symbol = ? AND interval = ? AND analysis_type = ?
            """
            cursor.execute(check_query, (market_type, market, symbol, interval, analysis_type))
            existing = cursor.fetchone()

            if existing:
                # 업데이트
                update_query = """
                UPDATE analysis_ratios SET
                    fractal_ratios = ?,
                    multi_timeframe_ratios = ?,
                    indicator_cross_ratios = ?,
                    symbol_specific_ratios = ?,
                    volatility_ratios = ?,
                    volume_ratios = ?,
                    optimal_modules = ?,
                    interval_weights = ?,
                    performance_score = ?,
                    accuracy_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE market_type = ? AND market = ? AND symbol = ? AND interval = ? AND analysis_type = ?
                """

                cursor.execute(update_query, (
                    safe_json_dumps(ratios_data.get('fractal_ratios', {})),
                    safe_json_dumps(ratios_data.get('multi_timeframe_ratios', {})),
                    safe_json_dumps(ratios_data.get('indicator_cross_ratios', {})),
                    safe_json_dumps(ratios_data.get('coin_specific_ratios', ratios_data.get('symbol_specific_ratios', {}))),
                    safe_json_dumps(ratios_data.get('volatility_ratios', {})),
                    safe_json_dumps(ratios_data.get('volume_ratios', {})),
                    safe_json_dumps(ratios_data.get('optimal_modules', {})),
                    safe_json_dumps(ratios_data.get('interval_weights', {})),
                    ratios_data.get('performance_score', 0.0),
                    ratios_data.get('accuracy_score', 0.0),
                    market_type, market, symbol, interval, analysis_type
                ))
            else:
                # 새로 삽입
                insert_query = """
                INSERT INTO analysis_ratios (
                    market_type, market, symbol, interval, analysis_type,
                    fractal_ratios, multi_timeframe_ratios, indicator_cross_ratios,
                    symbol_specific_ratios, volatility_ratios, volume_ratios,
                    optimal_modules, interval_weights, performance_score, accuracy_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor.execute(insert_query, (
                    market_type, market, symbol, interval, analysis_type,
                    safe_json_dumps(ratios_data.get('fractal_ratios', {})),
                    safe_json_dumps(ratios_data.get('multi_timeframe_ratios', {})),
                    safe_json_dumps(ratios_data.get('indicator_cross_ratios', {})),
                    safe_json_dumps(ratios_data.get('coin_specific_ratios', ratios_data.get('symbol_specific_ratios', {}))),
                    safe_json_dumps(ratios_data.get('volatility_ratios', {})),
                    safe_json_dumps(ratios_data.get('volume_ratios', {})),
                    safe_json_dumps(ratios_data.get('optimal_modules', {})),
                    safe_json_dumps(ratios_data.get('interval_weights', {})),
                    ratios_data.get('performance_score', 0.0),
                    ratios_data.get('accuracy_score', 0.0)
                ))

            conn.commit()
            logger.info(f"✅ {symbol} {interval} 분석 비율 저장 완료")
            return True

    except Exception as e:
        logger.error(f"❌ {coin} {interval} 분석 비율 저장 실패: {e}")
        return False

def save_coin_global_weights(coin: str, weights_data: Dict[str, Any],
                            market_type: str = DEFAULT_MARKET_TYPE,
                            market: str = DEFAULT_MARKET,
                            db_path: str = None) -> bool:
    """심볼 vs 글로벌 전략 가중치를 데이터베이스에 저장 (coin → symbol 매핑)

    Args:
        coin: 심볼 이름 (예: 'BTC') - v1 호환성 유지
        weights_data: 가중치 데이터
            - coin_weight/symbol_weight: 개별 심볼 전략 가중치 (0~1)
            - global_weight: 글로벌 전략 가중치 (0~1)
            - coin_score/symbol_score: 심볼 전략 성능 점수
            - global_score: 글로벌 전략 성능 점수
            - data_quality_score: 데이터 품질 점수
            - coin_strategy_count/symbol_strategy_count: 심볼 전략 개수
            - global_strategy_count: 글로벌 전략 개수
            - coin_avg_profit/symbol_avg_profit: 심볼 전략 평균 수익
            - global_avg_profit: 글로벌 전략 평균 수익
            - coin_win_rate/symbol_win_rate: 심볼 전략 승률
            - global_win_rate: 글로벌 전략 승률
        market_type: 마켓 타입 (기본: COIN)
        market: 마켓 (기본: BITHUMB)
        db_path: 선택적 DB 경로 (없으면 전략 DB 사용)

    Returns:
        bool: 저장 성공 여부
    """
    try:
        if db_path:
            pool = get_strategy_db_pool(db_path)
        else:
            pool = get_strategy_db_pool()

        # coin → symbol
        symbol = coin

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 기존 데이터가 있는지 확인
            check_query = "SELECT symbol FROM symbol_global_weights WHERE market_type = ? AND market = ? AND symbol = ?"
            cursor.execute(check_query, (market_type, market, symbol))
            existing = cursor.fetchone()

            # 필드 호환성 (coin_* → symbol_*)
            symbol_weight = weights_data.get('symbol_weight', weights_data.get('coin_weight', 0.7))
            symbol_score = weights_data.get('symbol_score', weights_data.get('coin_score', 0.0))
            symbol_strategy_count = weights_data.get('symbol_strategy_count', weights_data.get('coin_strategy_count', 0))
            symbol_avg_profit = weights_data.get('symbol_avg_profit', weights_data.get('coin_avg_profit', 0.0))
            symbol_win_rate = weights_data.get('symbol_win_rate', weights_data.get('coin_win_rate', 0.0))

            if existing:
                # 업데이트
                update_query = """
                UPDATE symbol_global_weights SET
                    symbol_weight = ?,
                    global_weight = ?,
                    symbol_score = ?,
                    global_score = ?,
                    data_quality_score = ?,
                    symbol_strategy_count = ?,
                    global_strategy_count = ?,
                    symbol_avg_profit = ?,
                    global_avg_profit = ?,
                    symbol_win_rate = ?,
                    global_win_rate = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE market_type = ? AND market = ? AND symbol = ?
                """

                cursor.execute(update_query, (
                    symbol_weight,
                    weights_data.get('global_weight', 0.3),
                    symbol_score,
                    weights_data.get('global_score', 0.0),
                    weights_data.get('data_quality_score', 0.0),
                    symbol_strategy_count,
                    weights_data.get('global_strategy_count', 0),
                    symbol_avg_profit,
                    weights_data.get('global_avg_profit', 0.0),
                    symbol_win_rate,
                    weights_data.get('global_win_rate', 0.0),
                    market_type, market, symbol
                ))
            else:
                # 새로 삽입
                insert_query = """
                INSERT INTO symbol_global_weights (
                    market_type, market, symbol,
                    symbol_weight, global_weight,
                    symbol_score, global_score, data_quality_score,
                    symbol_strategy_count, global_strategy_count,
                    symbol_avg_profit, global_avg_profit,
                    symbol_win_rate, global_win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor.execute(insert_query, (
                    market_type, market, symbol,
                    symbol_weight,
                    weights_data.get('global_weight', 0.3),
                    symbol_score,
                    weights_data.get('global_score', 0.0),
                    weights_data.get('data_quality_score', 0.0),
                    symbol_strategy_count,
                    weights_data.get('global_strategy_count', 0),
                    symbol_avg_profit,
                    weights_data.get('global_avg_profit', 0.0),
                    symbol_win_rate,
                    weights_data.get('global_win_rate', 0.0)
                ))

            conn.commit()
            logger.info(f"✅ {symbol} 심볼 vs 글로벌 가중치 저장 완료 (symbol: {symbol_weight:.2f}, global: {weights_data.get('global_weight', 0.3):.2f})")
            return True

    except Exception as e:
        logger.error(f"❌ {symbol} 심볼 vs 글로벌 가중치 저장 실패: {e}")
        return False