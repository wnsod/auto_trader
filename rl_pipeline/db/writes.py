"""
데이터베이스 쓰기 전용 커넥션 및 배치 처리
단일 writer 큐로 배치 insert/update, 트랜잭션 관리
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

def write_batch(rows: List[Dict[str, Any]], table: str, db_path: str = None, checkpoint: bool = False, verify: bool = False) -> int:
    """배치 쓰기
    Args:
        rows: 삽입/업데이트할 행 목록
        table: 대상 테이블명
        db_path: 선택적 DB 경로 (없으면 전략 DB 사용)
        checkpoint: 쓰기 이후 WAL 체크포인트를 강제 수행할지 여부 (기본 False)
        verify: 쓰기 직후 COUNT 검증을 수행할지 여부 (기본 False)
    """
    if not rows:
        return 0
    
    try:
        if db_path:
            pool = get_batch_loading_pool(db_path)
        else:
            pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 첫 번째 행으로 컬럼 정보 추출
            columns = list(rows[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # 배치 실행
            batch_data = []
            for row in rows:
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
            logger.info(f"🔍 executemany 실행 완료: {len(batch_data)}개 행")

            conn.commit()
            logger.info(f"🔍 커밋 완료")

            # 고비용 동작은 선택적으로 수행
            if checkpoint:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                logger.info(f"🔍 WAL 체크포인트 완료 (PASSIVE)")

            if verify:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"🔍 즉시 확인: {table} 테이블에 {count}개 레코드 존재")
            
            logger.info(f"✅ 배치 쓰기 완료: {len(rows)}행 -> {table}")
            return len(rows)
            
    except Exception as e:
        logger.error(f"❌ 배치 쓰기 실패: {e}")
        raise DBWriteError(f"배치 쓰기 실패 ({table}): {e}") from e

def upsert(data: Dict[str, Any], table: str, key_columns: List[str], db_path: str = None) -> bool:
    """Upsert (INSERT OR REPLACE)"""
    try:
        if db_path:
            pool = get_strategy_db_pool()
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
            query = f"UPDATE coin_strategies SET {', '.join(set_clauses)} WHERE id = ?"
            
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

def save_run_metadata(run_metadata: Dict[str, Any]) -> bool:
    """실행 메타데이터 저장"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 실행 메타데이터 준비
            run_record = {
                'run_id': run_metadata.get('run_id'),
                'coin': run_metadata.get('coin'),
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
            query = f"DELETE FROM coin_strategies WHERE id IN ({placeholders})"
            
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
                             ratios_data: Dict[str, Any]) -> bool:
    """🚀 코인별 분석 비율을 데이터베이스에 저장"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 기존 데이터가 있는지 확인
            check_query = """
            SELECT id FROM coin_analysis_ratios 
            WHERE coin = ? AND interval = ? AND analysis_type = ?
            """
            cursor.execute(check_query, (coin, interval, analysis_type))
            existing = cursor.fetchone()
            
            if existing:
                # 업데이트
                update_query = """
                UPDATE coin_analysis_ratios SET
                    fractal_ratios = ?,
                    multi_timeframe_ratios = ?,
                    indicator_cross_ratios = ?,
                    coin_specific_ratios = ?,
                    volatility_ratios = ?,
                    volume_ratios = ?,
                    optimal_modules = ?,
                    interval_weights = ?,
                    performance_score = ?,
                    accuracy_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE coin = ? AND interval = ? AND analysis_type = ?
                """

                cursor.execute(update_query, (
                    safe_json_dumps(ratios_data.get('fractal_ratios', {})),
                    safe_json_dumps(ratios_data.get('multi_timeframe_ratios', {})),
                    safe_json_dumps(ratios_data.get('indicator_cross_ratios', {})),
                    safe_json_dumps(ratios_data.get('coin_specific_ratios', {})),
                    safe_json_dumps(ratios_data.get('volatility_ratios', {})),
                    safe_json_dumps(ratios_data.get('volume_ratios', {})),
                    safe_json_dumps(ratios_data.get('optimal_modules', {})),
                    safe_json_dumps(ratios_data.get('interval_weights', {})),
                    ratios_data.get('performance_score', 0.0),
                    ratios_data.get('accuracy_score', 0.0),
                    coin, interval, analysis_type
                ))
            else:
                # 새로 삽입
                insert_query = """
                INSERT INTO coin_analysis_ratios (
                    coin, interval, analysis_type,
                    fractal_ratios, multi_timeframe_ratios, indicator_cross_ratios,
                    coin_specific_ratios, volatility_ratios, volume_ratios,
                    optimal_modules, interval_weights, performance_score, accuracy_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor.execute(insert_query, (
                    coin, interval, analysis_type,
                    safe_json_dumps(ratios_data.get('fractal_ratios', {})),
                    safe_json_dumps(ratios_data.get('multi_timeframe_ratios', {})),
                    safe_json_dumps(ratios_data.get('indicator_cross_ratios', {})),
                    safe_json_dumps(ratios_data.get('coin_specific_ratios', {})),
                    safe_json_dumps(ratios_data.get('volatility_ratios', {})),
                    safe_json_dumps(ratios_data.get('volume_ratios', {})),
                    safe_json_dumps(ratios_data.get('optimal_modules', {})),
                    safe_json_dumps(ratios_data.get('interval_weights', {})),
                    ratios_data.get('performance_score', 0.0),
                    ratios_data.get('accuracy_score', 0.0)
                ))
            
            conn.commit()
            logger.info(f"✅ {coin} {interval} 분석 비율 저장 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ {coin} {interval} 분석 비율 저장 실패: {e}")
        return False

def save_coin_global_weights(coin: str, weights_data: Dict[str, Any]) -> bool:
    """🔥 코인 vs 글로벌 전략 가중치를 데이터베이스에 저장

    Args:
        coin: 코인 이름 (예: 'BTC')
        weights_data: 가중치 데이터
            - coin_weight: 개별 코인 전략 가중치 (0~1)
            - global_weight: 글로벌 전략 가중치 (0~1)
            - coin_score: 코인 전략 성능 점수
            - global_score: 글로벌 전략 성능 점수
            - data_quality_score: 데이터 품질 점수
            - coin_strategy_count: 코인 전략 개수
            - global_strategy_count: 글로벌 전략 개수
            - coin_avg_profit: 코인 전략 평균 수익
            - global_avg_profit: 글로벌 전략 평균 수익
            - coin_win_rate: 코인 전략 승률
            - global_win_rate: 글로벌 전략 승률

    Returns:
        bool: 저장 성공 여부
    """
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 기존 데이터가 있는지 확인
            check_query = "SELECT coin FROM coin_global_weights WHERE coin = ?"
            cursor.execute(check_query, (coin,))
            existing = cursor.fetchone()

            if existing:
                # 업데이트
                update_query = """
                UPDATE coin_global_weights SET
                    coin_weight = ?,
                    global_weight = ?,
                    coin_score = ?,
                    global_score = ?,
                    data_quality_score = ?,
                    coin_strategy_count = ?,
                    global_strategy_count = ?,
                    coin_avg_profit = ?,
                    global_avg_profit = ?,
                    coin_win_rate = ?,
                    global_win_rate = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE coin = ?
                """

                cursor.execute(update_query, (
                    weights_data.get('coin_weight', 0.7),
                    weights_data.get('global_weight', 0.3),
                    weights_data.get('coin_score', 0.0),
                    weights_data.get('global_score', 0.0),
                    weights_data.get('data_quality_score', 0.0),
                    weights_data.get('coin_strategy_count', 0),
                    weights_data.get('global_strategy_count', 0),
                    weights_data.get('coin_avg_profit', 0.0),
                    weights_data.get('global_avg_profit', 0.0),
                    weights_data.get('coin_win_rate', 0.0),
                    weights_data.get('global_win_rate', 0.0),
                    coin
                ))
            else:
                # 새로 삽입
                insert_query = """
                INSERT INTO coin_global_weights (
                    coin, coin_weight, global_weight,
                    coin_score, global_score, data_quality_score,
                    coin_strategy_count, global_strategy_count,
                    coin_avg_profit, global_avg_profit,
                    coin_win_rate, global_win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor.execute(insert_query, (
                    coin,
                    weights_data.get('coin_weight', 0.7),
                    weights_data.get('global_weight', 0.3),
                    weights_data.get('coin_score', 0.0),
                    weights_data.get('global_score', 0.0),
                    weights_data.get('data_quality_score', 0.0),
                    weights_data.get('coin_strategy_count', 0),
                    weights_data.get('global_strategy_count', 0),
                    weights_data.get('coin_avg_profit', 0.0),
                    weights_data.get('global_avg_profit', 0.0),
                    weights_data.get('coin_win_rate', 0.0),
                    weights_data.get('global_win_rate', 0.0)
                ))

            conn.commit()
            logger.info(f"✅ {coin} 코인 vs 글로벌 가중치 저장 완료 (coin: {weights_data.get('coin_weight', 0.7):.2f}, global: {weights_data.get('global_weight', 0.3):.2f})")
            return True

    except Exception as e:
        logger.error(f"❌ {coin} 코인 vs 글로벌 가중치 저장 실패: {e}")
        return False