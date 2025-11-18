"""
데이터베이스 읽기 전용 커넥션 및 표준 조회 헬퍼
"""

import json
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from rl_pipeline.db.connection_pool import get_candle_db_pool, get_strategy_db_pool
from rl_pipeline.core.errors import DBReadError
from rl_pipeline.core.utils import safe_json_loads

logger = logging.getLogger(__name__)

def _select_pool_by_query(db_path: str, query: str):
    """쿼리/명시 경로 기준으로 적절한 풀 선택"""
    if db_path:
        return get_candle_db_pool() if 'candles' in db_path else get_strategy_db_pool()
    # 쿼리 기반 휴리스틱: 전략 테이블 키워드가 있으면 전략 DB
    q = (query or '').lower()
    strategy_markers = ['coin_strategies', 'strategy_dna', 'fractal_analysis', 'synergy_analysis', 'runs', 'replay_results', 'simulation_results', 'dna_analysis', 'global_strategies', 'performance_monitoring']
    if any(marker in q for marker in strategy_markers):
        return get_strategy_db_pool()
    return get_candle_db_pool()


def fetch_df(query: str, params: Tuple = (), db_path: str = None) -> pd.DataFrame:
    """데이터프레임 조회"""
    try:
        pool = _select_pool_by_query(db_path, query)
        
        with pool.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            logger.debug(f"✅ 데이터프레임 조회 완료: {len(df)}행")
            return df
            
    except Exception as e:
        logger.error(f"❌ 데이터프레임 조회 실패: {e}")
        raise DBReadError(f"데이터프레임 조회 실패: {e}") from e

def fetch_one(query: str, params: Tuple = (), db_path: str = None) -> Optional[Tuple]:
    """단일 행 조회"""
    try:
        pool = _select_pool_by_query(db_path, query)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            logger.debug(f"✅ 단일 행 조회 완료: {result}")
            return result
            
    except Exception as e:
        logger.error(f"❌ 단일 행 조회 실패: {e}")
        raise DBReadError(f"단일 행 조회 실패: {e}") from e

def fetch_many(query: str, params: Tuple = (), size: int = 1000, db_path: str = None) -> List[Tuple]:
    """여러 행 조회"""
    try:
        pool = _select_pool_by_query(db_path, query)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchmany(size)
            logger.debug(f"✅ 여러 행 조회 완료: {len(results)}행")
            return results
            
    except Exception as e:
        logger.error(f"❌ 여러 행 조회 실패: {e}")
        raise DBReadError(f"여러 행 조회 실패: {e}") from e

def fetch_all(query: str, params: Tuple = (), db_path: str = None) -> List[Tuple]:
    """모든 행 조회"""
    try:
        pool = _select_pool_by_query(db_path, query)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            logger.debug(f"✅ 모든 행 조회 완료: {len(results)}행")
            return results
            
    except Exception as e:
        logger.error(f"❌ 모든 행 조회 실패: {e}")
        raise DBReadError(f"모든 행 조회 실패: {e}") from e

def get_candle_data(coin: str, interval: str, days: int = 30, limit: int = None) -> pd.DataFrame:
    """캔들 데이터 조회"""
    query = """
    SELECT * FROM candles 
    WHERE coin = ? AND interval = ? 
    ORDER BY timestamp DESC
    """
    params = (coin, interval)
    
    if limit:
        query += f" LIMIT {limit}"
    else:
        query += f" LIMIT {days * 24 * 4}"  # 15분 간격 기준
    
    return fetch_df(query, params)

def get_strategy_data(coin: str = None, interval: str = None, limit: int = None) -> pd.DataFrame:
    """전략 데이터 조회"""
    query = "SELECT * FROM coin_strategies WHERE 1=1"
    params = []
    
    if coin:
        query += " AND coin = ?"
        params.append(coin)
    
    if interval:
        query += " AND interval = ?"
        params.append(interval)
    
    query += " ORDER BY created_at DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    return fetch_df(query, tuple(params), db_path="strategies")

def get_top_strategies(coin: str, interval: str, limit: int = 100, min_trades: int = 10) -> pd.DataFrame:
    """상위 전략 조회"""
    query = """
    SELECT * FROM coin_strategies 
    WHERE coin = ? AND interval = ? AND trades_count >= ?
    ORDER BY profit DESC, win_rate DESC
    LIMIT ?
    """
    params = (coin, interval, min_trades, limit)
    
    return fetch_df(query, params, db_path="strategies")

def get_strategy_by_id(strategy_id: str) -> Optional[Dict[str, Any]]:
    """ID로 전략 조회"""
    query = "SELECT * FROM coin_strategies WHERE id = ?"
    result = fetch_one(query, (strategy_id,), db_path="strategies")
    
    if result:
        columns = [desc[0] for desc in fetch_one("PRAGMA table_info(coin_strategies)", db_path="strategies")]
        return dict(zip(columns, result))
    
    return None

def get_dna_data(coin: str = None, limit: int = 100) -> pd.DataFrame:
    """DNA 데이터 조회"""
    query = "SELECT * FROM strategy_dna WHERE 1=1"
    params = []
    
    if coin:
        query += " AND coin = ?"
        params.append(coin)
    
    query += " ORDER BY created_at DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    return fetch_df(query, tuple(params), db_path="strategies")

def get_fractal_data(coin: str = None, limit: int = 100) -> pd.DataFrame:
    """프랙탈 분석 데이터 조회"""
    query = "SELECT * FROM fractal_analysis WHERE 1=1"
    params = []
    
    if coin:
        query += " AND coin = ?"
        params.append(coin)
    
    query += " ORDER BY created_at DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    return fetch_df(query, tuple(params), db_path="strategies")

def get_synergy_data(coin: str = None, limit: int = 100) -> pd.DataFrame:
    """시너지 분석 데이터 조회"""
    query = "SELECT * FROM synergy_analysis WHERE 1=1"
    params = []
    
    if coin:
        query += " AND coin = ?"
        params.append(coin)
    
    query += " ORDER BY created_at DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    return fetch_df(query, tuple(params), db_path="strategies")

def get_performance_data(limit: int = 1000) -> pd.DataFrame:
    """성능 모니터링 데이터 조회"""
    query = """
    SELECT * FROM performance_monitoring 
    ORDER BY timestamp DESC 
    LIMIT ?
    """
    return fetch_df(query, (limit,), db_path="strategies")

def get_run_history(limit: int = 100) -> pd.DataFrame:
    """실행 이력 조회"""
    query = """
    SELECT * FROM runs 
    ORDER BY start_time DESC 
    LIMIT ?
    """
    return fetch_df(query, (limit,), db_path="strategies")

def check_table_exists(table_name: str, db_path: str = None) -> bool:
    """테이블 존재 여부 확인"""
    query = """
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name=?
    """
    result = fetch_one(query, (table_name,), db_path)
    return result is not None

def get_table_info(table_name: str, db_path: str = None) -> List[Dict[str, Any]]:
    """테이블 정보 조회"""
    query = f"PRAGMA table_info({table_name})"
    results = fetch_all(query, db_path=db_path)
    
    columns = ['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']
    return [dict(zip(columns, row)) for row in results]

def load_strategies_by_grade(coin: str, interval: str, grade: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """등급별 전략 조회 (quality_grade 컬럼 사용)"""
    try:
        # grade가 None인 경우 등급이 없는 전략들 조회
        if grade is None:
            query = """
            SELECT * FROM coin_strategies 
            WHERE coin = ? AND interval = ? AND quality_grade IS NULL
            ORDER BY created_at DESC, profit DESC, win_rate DESC
            LIMIT ? OFFSET ?
            """
            params = (coin, interval, limit, offset)
        else:
            # quality_grade 컬럼을 사용하여 조회
            query = """
            SELECT * FROM coin_strategies 
            WHERE coin = ? AND interval = ? AND quality_grade = ?
            ORDER BY profit DESC, win_rate DESC
            LIMIT ? OFFSET ?
            """
            params = (coin, interval, grade, limit, offset)
        
        results = fetch_all(query, params, db_path="strategies")
        
        if results:
            # 컬럼 정보 가져오기
            columns_query = "PRAGMA table_info(coin_strategies)"
            column_info = fetch_all(columns_query, db_path="strategies")
            columns = [col[1] for col in column_info]  # col[1]은 컬럼명
            
            # 딕셔너리 리스트로 변환
            strategies = []
            for row in results:
                strategy_dict = dict(zip(columns, row))
                strategies.append(strategy_dict)
            
            grade_name = "등급없음" if grade is None else f"{grade}등급"
            logger.debug(f"✅ {grade_name} 전략 {len(strategies)}개 조회 완료")
            return strategies
    except Exception as e:
        logger.debug(f"quality_grade 컬럼 조회 실패, 최신 전략들 사용: {e}")
    
    # quality_grade 컬럼이 없거나 조회 실패 시 최신 전략들 반환
    query = """
    SELECT * FROM coin_strategies 
    WHERE coin = ? AND interval = ?
    ORDER BY created_at DESC, profit DESC, win_rate DESC
    LIMIT ?
    """
    params = (coin, interval, limit)
    
    results = fetch_all(query, params, db_path="strategies")
    
    if results:
        # 컬럼 정보 가져오기
        columns_query = "PRAGMA table_info(coin_strategies)"
        column_info = fetch_all(columns_query, db_path="strategies")
        columns = [col[1] for col in column_info]  # col[1]은 컬럼명
        
        # 딕셔너리 리스트로 변환
        strategies = []
        for row in results:
            strategy_dict = dict(zip(columns, row))
            strategies.append(strategy_dict)
        
        logger.debug(f"✅ 최신 전략 {len(strategies)}개 조회 완료")
        return strategies
    
    logger.warning(f"⚠️ {coin} {interval} 전략이 없습니다")
    return []

def load_strategies_by_market_condition(coin: str, interval: str, market_condition: str, limit: int = 100) -> List[Dict[str, Any]]:
    """시장 상황별 전략 조회"""
    try:
        query = """
        SELECT * FROM coin_strategies 
        WHERE coin = ? AND interval = ? AND market_condition = ?
        ORDER BY profit DESC, win_rate DESC
        LIMIT ?
        """
        params = (coin, interval, market_condition, limit)
        
        results = fetch_all(query, params, db_path="strategies")
        
        if results:
            # 컬럼 정보 가져오기
            columns_query = "PRAGMA table_info(coin_strategies)"
            column_info = fetch_all(columns_query, db_path="strategies")
            columns = [col[1] for col in column_info]
            
            # 딕셔너리 리스트로 변환
            strategies = []
            for row in results:
                strategy_dict = dict(zip(columns, row))
                strategies.append(strategy_dict)
            
            logger.info(f"✅ {market_condition} 시장 상황 전략 {len(strategies)}개 조회 완료")
            return strategies
        else:
            logger.warning(f"⚠️ {market_condition} 시장 상황 전략이 없음")
            return []
            
    except Exception as e:
        logger.error(f"❌ 시장 상황별 전략 조회 실패: {e}")
        # 시장 상황별 전략이 없으면 일반 전략 반환
        return load_strategies_by_grade(coin, interval, 'C', limit)

def load_strategies_by_interval_and_market(coin: str, interval: str, market_condition: str, limit: int = 100) -> List[Dict[str, Any]]:
    """인터벌별 시장 상황 전략 조회 (통합 함수)"""
    try:
        # 먼저 시장 상황별 전략 시도
        strategies = load_strategies_by_market_condition(coin, interval, market_condition, limit)
        
        if strategies:
            return strategies
        
        # 시장 상황별 전략이 없으면 등급별 전략 시도
        strategies = load_strategies_by_grade(coin, interval, 'C', limit)
        
        if strategies:
            logger.info(f"🔍 {coin} {interval}: {market_condition} 시장 상황 전략이 없어 C등급 전략 사용")
            return strategies
        
        # 등급별 전략도 없으면 최신 전략 반환
        query = """
        SELECT * FROM coin_strategies 
        WHERE coin = ? AND interval = ?
        ORDER BY created_at DESC, profit DESC, win_rate DESC
        LIMIT ?
        """
        params = (coin, interval, limit)
        
        results = fetch_all(query, params, db_path="strategies")
        
        if results:
            columns_query = "PRAGMA table_info(coin_strategies)"
            column_info = fetch_all(columns_query, db_path="strategies")
            columns = [col[1] for col in column_info]
            strategies = []
            for row in results:
                strategy_dict = dict(zip(columns, row))
                strategies.append(strategy_dict)
            
            logger.info(f"🔍 {coin} {interval}: 최신 전략 {len(strategies)}개 사용")
            return strategies
        else:
            logger.warning(f"⚠️ {coin} {interval} 전략이 전혀 없음")
            return []
            
    except Exception as e:
        logger.error(f"❌ 인터벌별 시장 상황 전략 조회 실패: {e}")
        return []

def load_strategies_pool(coin: str, interval: Optional[str] = None, limit: int = 15000, order_by: str = "id DESC", include_unknown: bool = True) -> List[Dict[str, Any]]:
    """
    DB에서 전략 풀 로드 (공통 함수)
    
    Args:
        coin: 코인 심볼
        interval: 시간대 (None이면 모든 interval)
        limit: 최대 로드할 전략 수 (0이면 제한 없음)
        order_by: 정렬 기준 (예: "id DESC", "created_at DESC")
        include_unknown: UNKNOWN 등급 전략 포함 여부 (기본: True - 모든 전략 포함)
    
    Returns:
        전략 딕셔너리 리스트
    """
    try:
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        strategies = []
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 🔥 UNKNOWN 등급 포함 여부에 따라 쿼리 조건 추가
            # include_unknown=True이면 모든 등급 포함 (기본 동작)
            # include_unknown=False이면 UNKNOWN 제외 (기존 동작)
            
            if interval:
                if include_unknown:
                    # 모든 등급 포함 (UNKNOWN 포함)
                    query = f"""
                        SELECT * FROM coin_strategies 
                        WHERE coin = ? AND interval = ?
                        ORDER BY {order_by}
                    """
                    if limit > 0:
                        query += f" LIMIT ?"
                        cursor.execute(query, (coin, interval, limit))
                    else:
                        cursor.execute(query, (coin, interval))
                else:
                    # UNKNOWN 제외
                    query = f"""
                        SELECT * FROM coin_strategies 
                        WHERE coin = ? AND interval = ? 
                        AND (quality_grade IS NOT NULL AND quality_grade != 'UNKNOWN')
                        ORDER BY {order_by}
                    """
                    if limit > 0:
                        query += f" LIMIT ?"
                        cursor.execute(query, (coin, interval, limit))
                    else:
                        cursor.execute(query, (coin, interval))
            else:
                if include_unknown:
                    # 모든 등급 포함 (UNKNOWN 포함)
                    query = f"""
                        SELECT * FROM coin_strategies 
                        WHERE coin = ?
                        ORDER BY {order_by}
                    """
                    if limit > 0:
                        query += f" LIMIT ?"
                        cursor.execute(query, (coin, limit))
                    else:
                        cursor.execute(query, (coin,))
                else:
                    # UNKNOWN 제외
                    query = f"""
                        SELECT * FROM coin_strategies 
                        WHERE coin = ? 
                        AND (quality_grade IS NOT NULL AND quality_grade != 'UNKNOWN')
                        ORDER BY {order_by}
                    """
                    if limit > 0:
                        query += f" LIMIT ?"
                        cursor.execute(query, (coin, limit))
                    else:
                        cursor.execute(query, (coin,))
            
            results = cursor.fetchall()
            
            # 컬럼 정보 가져오기
            columns_query = "PRAGMA table_info(coin_strategies)"
            columns_info = cursor.execute(columns_query).fetchall()
            columns = [col[1] for col in columns_info]
            
            # 딕셔너리 리스트로 변환
            for row in results:
                strategy_dict = dict(zip(columns, row))
                strategies.append(strategy_dict)
        
        logger.debug(f"✅ {coin}{f'-{interval}' if interval else ''} 전략 {len(strategies)}개 로드 완료")
        return strategies
        
    except Exception as e:
        logger.error(f"❌ 전략 풀 로드 실패: {e}")
        return []


def extract_strategy_params(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    전략 딕셔너리에서 파라미터 추출 (공통 함수)
    
    Args:
        strategy: 전략 딕셔너리
    
    Returns:
        Self-play용 파라미터 딕셔너리
    """
    return {
        'rsi_min': strategy.get('rsi_min', 30),
        'rsi_max': strategy.get('rsi_max', 70),
        'volume_ratio_min': strategy.get('volume_ratio_min', 1.0),
        'volume_ratio_max': strategy.get('volume_ratio_max', 2.0),
        'macd_buy_threshold': strategy.get('macd_buy_threshold', 0.01),
        'macd_sell_threshold': strategy.get('macd_sell_threshold', -0.01),
        'stop_loss_pct': strategy.get('stop_loss_pct', 0.02),
        'take_profit_pct': strategy.get('take_profit_pct', 0.05)
    }


def get_database_status() -> Dict[str, int]:
    """데이터베이스 상태 조회"""
    status = {}
    
    # 주요 테이블들의 행 수 조회
    tables = [
        'candles', 'coin_strategies', 'strategy_dna', 
        'fractal_analysis', 'synergy_analysis', 'runs'
    ]
    
    for table in tables:
        try:
            if check_table_exists(table):
                result = fetch_one(f"SELECT COUNT(*) FROM {table}")
                status[table] = result[0] if result else 0
            else:
                status[table] = 0
        except Exception as e:
            logger.warning(f"⚠️ 테이블 {table} 상태 조회 실패: {e}")
            status[table] = -1
    
    return status

def get_coin_analysis_ratios(coin: str, interval: str, analysis_type: str = "default") -> Dict[str, Any]:
    """🚀 코인별 분석 비율 조회"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT fractal_ratios, multi_timeframe_ratios, indicator_cross_ratios,
                   coin_specific_ratios, volatility_ratios, volume_ratios,
                   optimal_modules, interval_weights, performance_score, accuracy_score, updated_at
            FROM coin_analysis_ratios
            WHERE coin = ? AND interval = ? AND analysis_type = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """

            cursor.execute(query, (coin, interval, analysis_type))
            result = cursor.fetchone()

            if result:
                return {
                    'fractal_ratios': json.loads(result[0]) if result[0] else {},
                    'multi_timeframe_ratios': json.loads(result[1]) if result[1] else {},
                    'indicator_cross_ratios': json.loads(result[2]) if result[2] else {},
                    'coin_specific_ratios': json.loads(result[3]) if result[3] else {},
                    'volatility_ratios': json.loads(result[4]) if result[4] else {},
                    'volume_ratios': json.loads(result[5]) if result[5] else {},
                    'optimal_modules': json.loads(result[6]) if result[6] else {},
                    'interval_weights': json.loads(result[7]) if result[7] else {},
                    'performance_score': result[8],
                    'accuracy_score': result[9],
                    'updated_at': result[10]
                }
            else:
                # 기본값 반환
                return {
                    'fractal_ratios': {'5m': 0.5, '15m': 0.5, '30m': 0.5, '1h': 0.5, '4h': 0.5, '1d': 0.5, '1w': 0.5},
                    'multi_timeframe_ratios': {'short': 0.5, 'medium': 0.5, 'long': 0.5},
                    'indicator_cross_ratios': {'rsi': 0.5, 'macd': 0.5, 'bb': 0.5},
                    'coin_specific_ratios': {'btc': 0.5, 'eth': 0.5, 'altcoin': 0.5},
                    'volatility_ratios': {'low': 0.5, 'medium': 0.5, 'high': 0.5},
                    'volume_ratios': {'low': 0.5, 'medium': 0.5, 'high': 0.5},
                    'optimal_modules': {'fractal': 0.6, 'multi_timeframe': 0.6, 'indicator_cross': 0.6},
                    'interval_weights': {},
                    'performance_score': 0.0,
                    'accuracy_score': 0.0,
                    'updated_at': None
                }

    except Exception as e:
        logger.error(f"❌ {coin} {interval} 분석 비율 조회 실패: {e}")
        # 기본값 반환
        return {
            'fractal_ratios': {'5m': 0.5, '15m': 0.5, '30m': 0.5, '1h': 0.5, '4h': 0.5, '1d': 0.5, '1w': 0.5},
            'multi_timeframe_ratios': {'short': 0.5, 'medium': 0.5, 'long': 0.5},
            'indicator_cross_ratios': {'rsi': 0.5, 'macd': 0.5, 'bb': 0.5},
            'coin_specific_ratios': {'btc': 0.5, 'eth': 0.5, 'altcoin': 0.5},
            'volatility_ratios': {'low': 0.5, 'medium': 0.5, 'high': 0.5},
            'volume_ratios': {'low': 0.5, 'medium': 0.5, 'high': 0.5},
            'optimal_modules': {'fractal': 0.6, 'multi_timeframe': 0.6, 'indicator_cross': 0.6},
            'interval_weights': {},
            'performance_score': 0.0,
            'accuracy_score': 0.0,
            'updated_at': None
        }

def get_all_coin_analysis_ratios(coin: str = None) -> List[Dict[str, Any]]:
    """🚀 모든 코인의 분석 비율 조회 (또는 특정 코인)"""
    try:
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            if coin:
                query = """
                SELECT coin, interval, analysis_type, fractal_ratios, multi_timeframe_ratios,
                       indicator_cross_ratios, coin_specific_ratios, volatility_ratios,
                       volume_ratios, optimal_modules, performance_score, accuracy_score, updated_at
                FROM coin_analysis_ratios 
                WHERE coin = ?
                ORDER BY updated_at DESC
                """
                cursor.execute(query, (coin,))
            else:
                query = """
                SELECT coin, interval, analysis_type, fractal_ratios, multi_timeframe_ratios,
                       indicator_cross_ratios, coin_specific_ratios, volatility_ratios,
                       volume_ratios, optimal_modules, performance_score, accuracy_score, updated_at
                FROM coin_analysis_ratios 
                ORDER BY coin, interval, updated_at DESC
                """
                cursor.execute(query)
            
            results = cursor.fetchall()
            
            analysis_ratios = []
            for result in results:
                analysis_ratios.append({
                    'coin': result[0],
                    'interval': result[1],
                    'analysis_type': result[2],
                    'fractal_ratios': json.loads(result[3]) if result[3] else {},
                    'multi_timeframe_ratios': json.loads(result[4]) if result[4] else {},
                    'indicator_cross_ratios': json.loads(result[5]) if result[5] else {},
                    'coin_specific_ratios': json.loads(result[6]) if result[6] else {},
                    'volatility_ratios': json.loads(result[7]) if result[7] else {},
                    'volume_ratios': json.loads(result[8]) if result[8] else {},
                    'optimal_modules': json.loads(result[9]) if result[9] else {},
                    'performance_score': result[10],
                    'accuracy_score': result[11],
                    'updated_at': result[12]
                })
            
            return analysis_ratios
            
    except Exception as e:
        logger.error(f"❌ 분석 비율 조회 실패: {e}")
        return []

def get_coin_global_weights(coin: str) -> Dict[str, Any]:
    """🔥 코인 vs 글로벌 전략 가중치 조회

    Args:
        coin: 코인 이름 (예: 'BTC')

    Returns:
        Dict[str, Any]: 가중치 데이터
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
            - updated_at: 마지막 업데이트 시간
    """
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
            SELECT coin_weight, global_weight, coin_score, global_score,
                   data_quality_score, coin_strategy_count, global_strategy_count,
                   coin_avg_profit, global_avg_profit, coin_win_rate, global_win_rate,
                   updated_at
            FROM coin_global_weights
            WHERE coin = ?
            """

            cursor.execute(query, (coin,))
            result = cursor.fetchone()

            if result:
                return {
                    'coin_weight': result[0],
                    'global_weight': result[1],
                    'coin_score': result[2],
                    'global_score': result[3],
                    'data_quality_score': result[4],
                    'coin_strategy_count': result[5],
                    'global_strategy_count': result[6],
                    'coin_avg_profit': result[7],
                    'global_avg_profit': result[8],
                    'coin_win_rate': result[9],
                    'global_win_rate': result[10],
                    'updated_at': result[11]
                }
            else:
                # 기본값 반환 (데이터가 없으면 균등 가중치)
                logger.debug(f"⚠️ {coin} 코인의 가중치 데이터가 없습니다. 기본값 사용")
                return {
                    'coin_weight': 0.7,
                    'global_weight': 0.3,
                    'coin_score': 0.0,
                    'global_score': 0.0,
                    'data_quality_score': 0.0,
                    'coin_strategy_count': 0,
                    'global_strategy_count': 0,
                    'coin_avg_profit': 0.0,
                    'global_avg_profit': 0.0,
                    'coin_win_rate': 0.0,
                    'global_win_rate': 0.0,
                    'updated_at': None
                }

    except Exception as e:
        logger.error(f"❌ {coin} 코인 vs 글로벌 가중치 조회 실패: {e}")
        # 기본값 반환
        return {
            'coin_weight': 0.7,
            'global_weight': 0.3,
            'coin_score': 0.0,
            'global_score': 0.0,
            'data_quality_score': 0.0,
            'coin_strategy_count': 0,
            'global_strategy_count': 0,
            'coin_avg_profit': 0.0,
            'global_avg_profit': 0.0,
            'coin_win_rate': 0.0,
            'global_win_rate': 0.0,
            'updated_at': None
        }

def get_all_coin_global_weights() -> List[Dict[str, Any]]:
    """🔥 모든 코인의 가중치 조회"""
    try:
        pool = get_strategy_db_pool()

        with pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
            SELECT coin, coin_weight, global_weight, coin_score, global_score,
                   data_quality_score, coin_strategy_count, global_strategy_count,
                   coin_avg_profit, global_avg_profit, coin_win_rate, global_win_rate,
                   updated_at
            FROM coin_global_weights
            ORDER BY coin
            """

            cursor.execute(query)
            results = cursor.fetchall()

            weights = []
            for result in results:
                weights.append({
                    'coin': result[0],
                    'coin_weight': result[1],
                    'global_weight': result[2],
                    'coin_score': result[3],
                    'global_score': result[4],
                    'data_quality_score': result[5],
                    'coin_strategy_count': result[6],
                    'global_strategy_count': result[7],
                    'coin_avg_profit': result[8],
                    'global_avg_profit': result[9],
                    'coin_win_rate': result[10],
                    'global_win_rate': result[11],
                    'updated_at': result[12]
                })

            return weights

    except Exception as e:
        logger.error(f"❌ 모든 코인 가중치 조회 실패: {e}")
        return []

def fetch_integrated_analysis(
    conn: sqlite3.Connection,
    coin: str,
    interval: str = None
) -> Optional[Dict]:
    """
    통합 분석 결과 조회

    Args:
        conn: SQLite 연결 객체
        coin: 코인 심볼
        interval: 시간대 (None이면 'all_intervals')

    Returns:
        통합 분석 결과 딕셔너리 또는 None
    """
    try:
        cursor = conn.cursor()
        interval_filter = interval if interval else 'all_intervals'

        # 🔥 스키마 확인: learning_results.py의 integrated_analysis_results 테이블 구조
        # 컬럼 순서: coin, interval, regime, fractal_score, multi_timeframe_score, 
        #           indicator_cross_score, ensemble_score, ensemble_confidence,
        #           final_signal_score, signal_confidence, signal_action, created_at
        query = '''
            SELECT coin, interval, signal_action, final_signal_score,
                   fractal_score, multi_timeframe_score, indicator_cross_score,
                   created_at
            FROM integrated_analysis_results
            WHERE coin = ? AND interval = ?
            ORDER BY created_at DESC LIMIT 1
        '''

        try:
            cursor.execute(query, (coin, interval_filter))
            row = cursor.fetchone()

            if not row:
                # 🔥 더 자세한 디버깅 정보 출력
                cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results WHERE coin = ?", (coin,))
                coin_count = cursor.fetchone()[0]
                cursor.execute("SELECT DISTINCT interval FROM integrated_analysis_results WHERE coin = ?", (coin,))
                available_intervals = [r[0] for r in cursor.fetchall()]
                logger.debug(f"⚠️ {coin} {interval_filter} 통합 분석 결과 없음 (코인별 총 {coin_count}개, 사용 가능한 인터벌: {available_intervals})")
                return None

            result = {
                'coin': row[0],
                'interval': row[1],
                'signal': row[2],  # signal_action을 signal로 매핑 (하위 호환성)
                'score': row[3],  # final_signal_score를 score로 매핑
                'fractal_score': row[4],
                'multi_tf_score': row[5],  # multi_timeframe_score를 multi_tf_score로 매핑
                'indicator_cross_score': row[6],
                'created_at': row[7]
            }
        except sqlite3.OperationalError as schema_err:
            # 🔥 스키마 불일치: 컬럼이 없으면 안전하게 처리
            if 'no such column' in str(schema_err).lower():
                logger.warning(f"⚠️ {coin} {interval_filter} 통합 분석 스키마 불일치: {schema_err}")
                # 기본값 반환
                return {
                    'coin': coin,
                    'interval': interval_filter,
                    'signal': 'HOLD',
                    'score': 0.5,
                    'fractal_score': 0.5,
                    'multi_tf_score': 0.5,
                    'indicator_cross_score': 0.5,
                    'created_at': None
                }
            else:
                raise

        logger.debug(f"✅ {coin} {interval_filter} 통합 분석 조회 완료: {result['signal']} ({result['score']:.3f})")
        return result

    except Exception as e:
        logger.error(f"❌ {coin} {interval_filter} 통합 분석 조회 실패: {e}")
        return None