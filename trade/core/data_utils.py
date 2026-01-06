#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
트레이딩 시스템 독립 데이터 유틸리티
rl_pipeline 의존성 없이 트레이딩에 필요한 데이터 조회 기능 제공
"""

import os
import sqlite3
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# 🆕 자체 경로 처리 (순환 임포트 방지 및 독립성 확보)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _finalize_path(path):
    """경로를 절대 경로로 변환 (Docker 환경)"""
    if not path: return None
    return os.path.abspath(path)

def _get_strategy_db_path():
    """전략 DB 경로 동적 결정 (환경변수 → 기본경로 폴백)"""
    
    # 1. GLOBAL_STRATEGY_DB_PATH 환경변수 시도
    db_path = _finalize_path(os.environ.get('GLOBAL_STRATEGY_DB_PATH'))
    if db_path:
        if os.path.isdir(db_path):
            candidate = os.path.join(db_path, 'common_strategies.db')
            if os.path.exists(candidate):
                return candidate
        elif os.path.exists(db_path):
            return db_path
    
    # 2. STRATEGY_DB_PATH 환경변수 시도
    db_path = _finalize_path(os.environ.get('STRATEGY_DB_PATH'))
    if db_path:
        if os.path.isdir(db_path):
            candidate = os.path.join(db_path, 'common_strategies.db')
            if os.path.exists(candidate):
                return candidate
        elif os.path.exists(db_path):
            return db_path
    
    # 3. 기본 경로들 시도
    default_paths = [
        os.path.join(_BASE_DIR, 'market', 'coin_market', 'data_storage', 'learning_strategies', 'common_strategies.db'),
        os.path.join(_BASE_DIR, 'market', 'coin_market', 'data_storage', 'common_strategies.db'),
        os.path.join(_BASE_DIR, 'data_storage', 'learning_strategies', 'common_strategies.db'),
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    return None


def _get_coin_strategy_db_path(coin: str):
    """개별 코인 전략 DB 경로 반환 ({coin}_strategies.db)"""
    
    # 1. STRATEGY_DB_PATH 환경변수 (디렉토리)
    strategies_dir = _finalize_path(os.environ.get('STRATEGY_DB_PATH'))
    if strategies_dir and os.path.isdir(strategies_dir):
        coin_db = os.path.join(strategies_dir, f'{coin}_strategies.db')
        if os.path.exists(coin_db):
            return coin_db
    
    # 2. 기본 경로들 시도
    default_dirs = [
        os.path.join(_BASE_DIR, 'market', 'coin_market', 'data_storage', 'learning_strategies'),
        os.path.join(_BASE_DIR, 'data_storage', 'learning_strategies'),
    ]
    
    for dir_path in default_dirs:
        coin_db = os.path.join(dir_path, f'{coin}_strategies.db')
        if os.path.exists(coin_db):
            return coin_db
    
    return None

def _get_candles_db_path():
    """캔들 DB 경로 동적 결정"""
    db_path = _finalize_path(os.environ.get('CANDLES_DB_PATH'))
    if db_path and os.path.exists(db_path):
        return db_path
    
    # 기본 경로들 시도
    default_paths = [
        os.path.join(_BASE_DIR, 'market', 'coin_market', 'data_storage', 'trade_candles.db'),
        os.path.join(_BASE_DIR, 'data_storage', 'trade_candles.db'),
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    return None

def _get_db_connection(db_path: str, read_only: bool = True):
    """간단한 DB 연결 (외부 의존성 없음)"""
    if not db_path or not os.path.exists(db_path):
        raise FileNotFoundError(f"DB 파일 없음: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000;")
        if not read_only:
            conn.execute("PRAGMA journal_mode=WAL;")
        return conn
    except Exception as e:
        raise Exception(f"DB 연결 실패 ({db_path}): {e}")


def get_available_coins_and_intervals(db_path: str = None) -> List[Tuple[str, str]]:
    """
    캔들 DB에서 사용 가능한 (코인, 인터벌) 조합 조회
    
    rl_pipeline.data.candle_loader.get_available_coins_and_intervals() 대체
    
    Returns:
        List[Tuple[str, str]]: [(코인, 인터벌), ...] 리스트
    """
    target_db = db_path or _get_candles_db_path()
    if not target_db or not os.path.exists(target_db):
        return []
    
    try:
        conn = _get_db_connection(target_db, read_only=True)
        try:
            cursor = conn.cursor()
            
            # candles 테이블에서 고유한 (symbol, interval) 조합 조회
            cursor.execute("""
                SELECT DISTINCT symbol, interval 
                FROM candles 
                WHERE symbol IS NOT NULL AND interval IS NOT NULL
                ORDER BY symbol, interval
            """)
            
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]
        finally:
            conn.close()
            
    except Exception as e:
        # 경로 디버깅 (silent)
        return []


def get_all_available_coins(db_path: str = None) -> List[str]:
    """
    캔들 DB에서 사용 가능한 모든 코인 목록 조회
    
    Returns:
        List[str]: 정렬된 코인 심볼 리스트
    """
    pairs = get_available_coins_and_intervals(db_path)
    coins = sorted(list(set(coin for coin, _ in pairs)))
    return coins


def get_coin_analysis_ratios(coin: str = None, interval: str = 'all') -> List[Dict[str, Any]]:
    """
    분석 비율 조회 (개별 코인 DB 우선 → 글로벌 DB 폴백)
    
    rl_pipeline.db.reads.get_coin_analysis_ratios() 대체
    
    Args:
        coin: 특정 코인 (None이면 전체)
        interval: 인터벌 (기본 'all')
        
    Returns:
        List[Dict]: 분석 비율 정보 리스트
    """
    
    # 🔥 1. 개별 코인 DB에서 먼저 조회 (manual_analysis_ratios.py가 저장하는 위치)
    if coin:
        coin_db_path = _get_coin_strategy_db_path(coin)
        if coin_db_path:
            result = _query_analysis_ratios_from_db(coin_db_path, coin, interval)
            if result:
                return result
    
    # 🔥 2. 폴백: 글로벌 전략 DB에서 조회
    db_path = _get_strategy_db_path()
    if db_path and os.path.exists(db_path):
        result = _query_analysis_ratios_from_db(db_path, coin, interval)
        if result:
            return result
    
    # 3. 데이터 없으면 기본값 반환
    return _get_default_analysis_ratios_list()


def _query_analysis_ratios_from_db(db_path: str, coin: str = None, interval: str = 'all') -> List[Dict[str, Any]]:
    """특정 DB에서 analysis_ratios 조회"""
    try:
        conn = _get_db_connection(db_path, read_only=True)
        try:
            cursor = conn.cursor()
            
            # 테이블 존재 여부 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('analysis_ratios', 'coin_analysis_ratios')
            """)
            available_tables = [row[0] for row in cursor.fetchall()]
            
            if not available_tables:
                return []
            
            table_name = 'analysis_ratios' if 'analysis_ratios' in available_tables else 'coin_analysis_ratios'
            
            # 컬럼명 확인
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            coin_column = 'symbol' if 'symbol' in columns else 'coin'
            
            if coin:
                cursor.execute(f"""
                    SELECT * FROM {table_name} 
                    WHERE {coin_column} = ?
                    ORDER BY updated_at DESC
                """, (coin,))
            else:
                cursor.execute(f"""
                    SELECT * FROM {table_name} 
                    ORDER BY {coin_column}, interval, updated_at DESC
                """)
            
            results = cursor.fetchall()
            if not results:
                return []
                
            column_names = [desc[0] for desc in cursor.description]
            return [dict(zip(column_names, row)) for row in results]
        finally:
            conn.close()
            
    except Exception:
        return []


def get_coin_global_weights(coin: str, interval: str = 'combined') -> Dict[str, float]:
    """
    코인별 글로벌 가중치 조회
    
    rl_pipeline.db.reads.get_coin_global_weights() 대체
    
    Returns:
        Dict[str, float]: 가중치 딕셔너리
    """
    default_weights = {
        'technical': 0.3,
        'wave': 0.2,
        'rl': 0.25,
        'ai': 0.25
    }
    
    db_path = _get_strategy_db_path()
    if not db_path or not os.path.exists(db_path):
        return default_weights
    
    try:
        conn = _get_db_connection(db_path, read_only=True)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='global_weights'
            """)
            
            if not cursor.fetchone():
                return default_weights
            
            cursor.execute("""
                SELECT weight_name, weight_value 
                FROM global_weights 
                WHERE coin = ? AND interval = ?
            """, (coin, interval))
            
            results = cursor.fetchall()
            if results:
                return {row[0]: row[1] for row in results}
            
            # 코인별 가중치가 없으면 기본값 조회
            cursor.execute("""
                SELECT weight_name, weight_value 
                FROM global_weights 
                WHERE coin = 'default'
            """)
            
            results = cursor.fetchall()
            if results:
                return {row[0]: row[1] for row in results}
                
            return default_weights
        finally:
            conn.close()
            
    except Exception:
        return default_weights


def load_global_strategies_from_db(db_path: str = None) -> Dict[str, List[Dict]]:
    """
    글로벌 전략 DB에서 전략 로드
    
    rl_pipeline.db.learning_results.load_global_strategies_from_db() 대체
    
    Returns:
        Dict[str, List[Dict]]: 인터벌별 전략 리스트
    """
    target_db = db_path or _get_strategy_db_path()
    
    if not target_db or not os.path.exists(target_db):
        return {}
    
    try:
        conn = _get_db_connection(target_db, read_only=True)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='global_strategies'
            """)
            
            if not cursor.fetchone():
                return {}
            
            cursor.execute("""
                SELECT interval, strategy_type, params, profit, win_rate, trades_count, regime
                FROM global_strategies
                ORDER BY interval, profit DESC
            """)
            
            results = defaultdict(list)
            for row in cursor.fetchall():
                interval, st_type, params_json, profit, win_rate, trades, regime = row
                try:
                    params = json.loads(params_json) if params_json else {}
                except:
                    params = {}
                    
                results[interval].append({
                    'strategy_type': st_type,
                    'params': params,
                    'profit': profit,
                    'win_rate': win_rate,
                    'trades_count': trades,
                    'regime': regime
                })
            
            return dict(results)
        finally:
            conn.close()
            
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def get_interval_role(interval: str) -> Dict[str, Any]:
    """
    인터벌별 역할 정보 반환
    
    rl_pipeline.core.interval_profiles.get_interval_role() 대체
    
    Returns:
        Dict: 인터벌 역할 정보
    """
    # 표준 인터벌 역할 정의
    INTERVAL_ROLES = {
        '1m': {'role': 'scalping', 'weight': 0.05, 'horizon': 'ultra_short'},
        '5m': {'role': 'scalping', 'weight': 0.10, 'horizon': 'ultra_short'},
        '15m': {'role': 'entry_timing', 'weight': 0.20, 'horizon': 'short'},
        '30m': {'role': 'entry_confirmation', 'weight': 0.25, 'horizon': 'short'},
        '1h': {'role': 'trend_following', 'weight': 0.30, 'horizon': 'medium'},
        '4h': {'role': 'trend_validation', 'weight': 0.35, 'horizon': 'medium'},
        '240m': {'role': 'trend_validation', 'weight': 0.35, 'horizon': 'medium'},
        '1d': {'role': 'macro_trend', 'weight': 0.40, 'horizon': 'long'},
        'daily': {'role': 'macro_trend', 'weight': 0.40, 'horizon': 'long'},
    }
    
    return INTERVAL_ROLES.get(interval.lower(), {
        'role': 'unknown',
        'weight': 0.15,
        'horizon': 'medium'
    })


def _get_default_analysis_ratios_list() -> List[Dict[str, Any]]:
    """기본 분석 비율 리스트 반환"""
    return [{
        'coin': 'default',
        'interval': 'all',
        'fractal_ratios': {'15m': 0.2, '30m': 0.25, '240m': 0.3, '1d': 0.25},
        'multi_timeframe_ratios': {'15m': 0.2, '30m': 0.25, '240m': 0.3, '1d': 0.25},
        'indicator_cross_ratios': {},
        'coin_specific_ratios': {},
        'volatility_ratios': {},
        'volume_ratios': {},
        'optimal_modules': {},
        'performance_score': 0.5,
        'accuracy_score': 0.5
    }]


# 🆕 단순화된 통합 분석기 (rl_pipeline 의존성 제거)
class SimpleIntegratedAnalyzer:
    """
    트레이딩 전용 간소화 통합 분석기
    
    rl_pipeline.analysis.integrated_analyzer.IntegratedAnalyzer 대체
    """
    
    def __init__(self):
        self.enabled = True
    
    def analyze(self, coin: str, interval: str, candle_data: dict) -> Dict[str, Any]:
        """간단한 통합 분석 수행"""
        try:
            rsi = candle_data.get('rsi', 50.0)
            macd = candle_data.get('macd', 0.0)
            wave_phase = candle_data.get('wave_phase', 'neutral')
            
            # 기본 점수 계산
            score = 0.5
            
            # RSI 기반 조정
            if rsi < 30:
                score += 0.15  # 과매도 → 매수 신호
            elif rsi > 70:
                score -= 0.15  # 과매수 → 매도 신호
            
            # 파동 단계 기반 조정
            wave_adjustments = {
                'uptrend': 0.1,
                'downtrend': -0.1,
                'consolidation': 0.0,
                'sideways': 0.0,
                'bullish': 0.1,
                'bearish': -0.1
            }
            score += wave_adjustments.get(wave_phase.lower(), 0.0)
            
            return {
                'score': max(0.0, min(1.0, score)),
                'confidence': 0.7,
                'direction': 'up' if score > 0.5 else ('down' if score < 0.5 else 'neutral'),
                'analysis_type': 'simple_integrated'
            }
            
        except Exception as e:
            return {
                'score': 0.5,
                'confidence': 0.5,
                'direction': 'neutral',
                'analysis_type': 'fallback'
            }


# 🆕 단순화된 메타 감독자 (rl_pipeline 의존성 제거)
class SimpleMetaSupervisor:
    """
    트레이딩 전용 간소화 메타 감독자
    
    rl_pipeline.analysis.meta_supervisor.MetaCognitiveSupervisor 대체
    """
    
    def __init__(self):
        self.enabled = True
    
    def evaluate(self, signal_data: dict) -> Dict[str, Any]:
        """신호 평가"""
        score = signal_data.get('score', 0.5)
        confidence = signal_data.get('confidence', 0.5)
        
        return {
            'approved': confidence > 0.5,
            'adjusted_score': score,
            'meta_confidence': confidence,
            'reason': 'simple_evaluation'
        }


# 편의 함수들
def get_integrated_analyzer():
    """통합 분석기 인스턴스 반환"""
    return SimpleIntegratedAnalyzer()

def get_meta_supervisor():
    """메타 감독자 인스턴스 반환"""
    return SimpleMetaSupervisor()
