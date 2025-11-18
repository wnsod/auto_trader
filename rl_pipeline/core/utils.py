"""
Absolute Zero 시스템 공용 유틸리티
모든 모듈에서 사용하는 공통 유틸리티 함수들
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import sqlite3

logger = logging.getLogger(__name__)

def safe_json_loads(json_str: str, default_value: Any = None) -> Any:
    """안전한 JSON 파싱"""
    try:
        if json_str is None or json_str == '':
            return default_value
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON 파싱 실패: {json_str} -> {e}")
        return default_value

def safe_json_dumps(data: Any, default_value: str = '{}') -> str:
    """안전한 JSON 직렬화"""
    try:
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.debug(f"JSON 직렬화 실패: {data} -> {e}")
        return default_value

def safe_json_serializer(obj: Any) -> Any:
    """JSON 직렬화를 위한 안전한 변환"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'dtype'):  # pandas 타입 처리
        if 'int' in str(obj.dtype):
            return int(obj)
        elif 'float' in str(obj.dtype):
            return float(obj)
        else:
            return str(obj)
    else:
        return str(obj)

def _safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """안전한 float 변환"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # 날짜/시간 문자열 체크
            if any(char in value for char in ['-', ':', ' ']) and len(value) > 10:
                logger.debug(f"날짜/시간 문자열을 float로 변환 시도: {value} -> 기본값 {default} 반환")
                return default
            return float(value)
        return float(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"float 변환 실패: {value} -> {e} -> 기본값 {default} 반환")
        return default

def _format_decimal_precision(value: Any, field_name: str = 'default') -> float:
    """필드별 적절한 소숫점 자릿수로 포맷팅"""
    try:
        if value is None or pd.isna(value):
            return 0.0
        
        # 필드별 소숫점 자릿수 설정
        precision_map = {
            # 성과 지표 (4자리)
            'profit': 4, 'win_rate': 4, 'max_drawdown': 4,
            'sharpe_ratio': 4, 'calmar_ratio': 4, 'sortino_ratio': 4,
            'var_95': 4, 'var_99': 4, 'profit_factor': 4,
            'recovery_factor': 4, 'avg_profit_per_trade': 6,
            'total_return': 4, 'profit_loss_ratio': 4,
            
            # 전략 파라미터 (3-4자리)
            'score': 4, 'stop_loss_pct': 3, 'take_profit_pct': 3,
            'complexity_score': 4, 'confidence': 4,
            
            # 기술지표 (4자리)
            'rsi': 4, 'mfi': 4, 'adx': 4, 'macd': 4, 'macd_signal': 4,
            'macd_buy_threshold': 4, 'macd_sell_threshold': 4,
            'rsi_min': 4, 'rsi_max': 4, 'volume_ratio_min': 4, 'volume_ratio_max': 4,
            'bb_upper': 4, 'bb_middle': 4, 'bb_lower': 4, 'bb_position': 4,
            'atr': 4, 'volatility': 4, 'volume_ratio': 4,
            'avg_min': 4, 'avg_max': 4, 'min_std': 4, 'max_std': 4,
            'buy_std': 4, 'sell_std': 4, 'pattern_consistency': 4,
            
            # 학습 관련 지표 (4자리)
            'market_volatility': 4, 'trend_strength': 4, 'price_momentum': 4,
            'rsi_avg': 4, 'macd_signal_strength': 4, 'bb_position': 4,
            'learning_quality_score': 4,
            
            # 기타 (4자리 기본)
            'default': 4
        }
        
        precision = precision_map.get(field_name, 4)
        return round(float(value), precision)
        
    except Exception as e:
        logger.debug(f"소숫점 포맷팅 실패: {value} -> {e}")
        return 0.0

def format_simulation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """시뮬레이션 결과의 모든 수치값을 적절한 소숫점으로 정리"""
    try:
        formatted_result = {}
        
        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_result[key] = _format_decimal_precision(value, key)
            else:
                formatted_result[key] = value
                
        return formatted_result
        
    except Exception as e:
        logger.error(f"시뮬레이션 결과 포맷팅 실패: {e}")
        return result

def format_strategy_data(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """전략 데이터의 모든 수치값을 적절한 소숫점으로 정리"""
    try:
        formatted_strategy = {}
        
        for key, value in strategy.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_strategy[key] = _format_decimal_precision(value, key)
            else:
                formatted_strategy[key] = value
                
        return formatted_strategy
        
    except Exception as e:
        logger.error(f"전략 데이터 포맷팅 실패: {e}")
        return strategy

def _safe_parse_timestamp(timestamp_value: Any) -> Optional[datetime]:
    """안전하게 timestamp를 파싱"""
    try:
        if timestamp_value is None:
            return None
        
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        
        if isinstance(timestamp_value, pd.Timestamp):
            return timestamp_value.to_pydatetime()
        
        if isinstance(timestamp_value, str):
            # ISO 형식 파싱 시도
            try:
                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            except ValueError:
                # 다른 형식들 시도
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y%m%d_%H%M%S']:
                    try:
                        return datetime.strptime(timestamp_value, fmt)
                    except ValueError:
                        continue
        
        return None
        
    except Exception as e:
        logger.debug(f"timestamp 파싱 실패: {timestamp_value} -> {e}")
        return None

def make_serializable(obj: Any) -> Any:
    """객체를 직렬화 가능하게 변환"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'dtype'):  # pandas 타입 처리
        if 'int' in str(obj.dtype):
            return int(obj)
        elif 'float' in str(obj.dtype):
            return float(obj)
        else:
            return str(obj)
    else:
        return str(obj)

def ensure_dir(path: str) -> str:
    """디렉토리가 존재하지 않으면 생성"""
    import os
    os.makedirs(path, exist_ok=True)
    return path

def generate_run_id(prefix: str = "abs_zero") -> str:
    """실행 ID 생성"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def log_system_stats(stats: Dict[str, Any]) -> None:
    """시스템 통계 로깅"""
    logger.info("📊 시스템 통계:")
    for key, value in stats.items():
        logger.info(f"  - {key}: {value}")

def update_system_stats(stats: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """시스템 통계 업데이트"""
    for key, value in updates.items():
        if key in stats:
            if isinstance(stats[key], (int, float)):
                stats[key] += value
            else:
                stats[key] = value
        else:
            stats[key] = value
    return stats

def extract_market_data_from_candles(candle_data: pd.DataFrame) -> Dict[str, Any]:
    """캔들 데이터에서 시장 데이터 추출 (공통 함수)
    
    Args:
        candle_data: OHLCV 데이터프레임
        
    Returns:
        시장 데이터 딕셔너리 (close, volume, indicators 등)
    """
    try:
        if candle_data is None or candle_data.empty:
            return {
                "close": [100.0],
                "volume": [1_000_000.0],
                "rsi": [],
                "macd": [],
                "macd_signal": []
            }
        
        market_data: Dict[str, Any] = {
            "close": candle_data["close"].tolist() if "close" in candle_data.columns else [100.0],
            "volume": candle_data["volume"].tolist() if "volume" in candle_data.columns else [1_000_000.0],
        }
        
        # 지표 데이터 추가
        for col in ("rsi", "macd", "macd_signal", "mfi", "atr", "adx", "bb_upper", "bb_middle", "bb_lower"):
            if col in candle_data.columns:
                market_data[col] = candle_data[col].tolist()
        
        return market_data
        
    except Exception as e:
        logger.warning(f"⚠️ 시장 데이터 추출 실패: {e}, 기본값 사용")
        return {
            "close": [100.0],
            "volume": [1_000_000.0],
        }

def table_exists(cursor, table_name: str) -> bool:
    """테이블 존재 여부 확인
    
    Args:
        cursor: DB 커서
        table_name: 확인할 테이블 이름
        
    Returns:
        테이블 존재 여부
    """
    try:
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        return cursor.fetchone() is not None
    except Exception as e:
        logger.warning(f"⚠️ 테이블 존재 확인 실패 ({table_name}): {e}")
        return False

def safe_query(cursor, query: str, params: Tuple = (), table_name: str = None, default_result: Any = None) -> List[Tuple]:
    """테이블 존재 확인 후 안전한 쿼리 실행
    
    Args:
        cursor: DB 커서
        query: 실행할 쿼리
        params: 쿼리 파라미터
        table_name: 확인할 테이블 이름 (자동 추출 시도)
        default_result: 테이블이 없을 때 반환할 기본값
        
    Returns:
        쿼리 결과 (테이블이 없으면 빈 리스트 또는 default_result)
    """
    try:
        # 테이블 이름 자동 추출 (SELECT ... FROM table_name 패턴)
        if table_name is None:
            query_lower = query.lower()
            if 'from' in query_lower:
                parts = query_lower.split('from')
                if len(parts) > 1:
                    table_part = parts[1].strip().split()[0]
                    # 테이블 이름에서 공백, 괄호 제거
                    table_name = table_part.split('(')[0].split()[0].strip()
        
        # 테이블 존재 확인
        if table_name and not table_exists(cursor, table_name):
            logger.warning(f"⚠️ 테이블이 존재하지 않음: {table_name}")
            return [] if default_result is None else default_result
        
        # 쿼리 실행
        cursor.execute(query, params)
        return cursor.fetchall()
        
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        if "no such table" in error_msg or "table" in error_msg and "not found" in error_msg:
            logger.warning(f"⚠️ 테이블이 존재하지 않음: {table_name or '알 수 없음'} ({e})")
            return [] if default_result is None else default_result
        logger.error(f"❌ 쿼리 실행 실패 ({table_name or '알 수 없음'}): {e}")
        return [] if default_result is None else default_result
    except Exception as e:
        logger.error(f"❌ 쿼리 실행 실패: {e}")
        return [] if default_result is None else default_result

def safe_query_one(cursor, query: str, params: Tuple = (), table_name: str = None, default_result: Any = None) -> Optional[Tuple]:
    """테이블 존재 확인 후 안전한 단일 행 쿼리 실행
    
    Args:
        cursor: DB 커서
        query: 실행할 쿼리
        params: 쿼리 파라미터
        table_name: 확인할 테이블 이름
        default_result: 테이블이 없을 때 반환할 기본값
        
    Returns:
        쿼리 결과 (단일 행) 또는 None
    """
    results = safe_query(cursor, query, params, table_name, default_result)
    return results[0] if results else (default_result if default_result is not None else None)