"""
strategy 관련 Mixin 클래스
SignalSelector의 strategy 기능을 담당합니다.
"""



# === 공통 import ===
import os
import sys
import logging
import traceback
import time
import json
import math
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

# signal_selector 내부 모듈
try:
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )
except ImportError:
    # 직접 실행 시 경로 추가
    _current = os.path.dirname(os.path.abspath(__file__))
    _signal_selector = os.path.dirname(_current)
    _trade = os.path.dirname(_signal_selector)
    sys.path.insert(0, _trade)
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )

# 헬퍼 클래스 import (core에서만 필요)
try:
    from signal_selector.helpers import (
        ContextualBandit, RegimeChangeDetector, ExponentialDecayWeight,
        BayesianSmoothing, ActionSpecificScorer, ContextFeatureExtractor,
        OutlierGuardrail, EvolutionEngine, ContextMemory, RealTimeLearner,
        SignalTradeConnector
    )
except ImportError:
    pass  # 헬퍼가 필요없는 Mixin에서는 무시


class StrategyMixin:
    """
    StrategyMixin - strategy 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

    def _get_top_strategy_from_db(self, coin: str, interval: str, regimes: List[str] = None) -> Optional[Dict]:
        """전략 DB에서 해당 코인/인터벌/레짐(들)의 최상위 전략 조회"""
        try:
            if regimes is None:
                regimes = ['neutral']
                
            from signal_selector.config import STRATEGIES_DB_PATH
            
            # 코인별 DB 경로 추론 (디렉토리 모드 지원)
            if os.path.isdir(STRATEGIES_DB_PATH):
                db_path = os.path.join(STRATEGIES_DB_PATH, f"{coin.lower()}_strategies.db")
            else:
                db_path = STRATEGIES_DB_PATH

            if not os.path.exists(db_path):
                return None

            with get_optimized_db_connection(db_path) as conn:
                # strategies 테이블에서 승률과 수익금 모두 좋은 전략 1개 조회
                # 🔥 레짐 조건 추가 (해당 레짐들 중 하나 또는 NULL인 전략 조회)
                # 🔥 MFE/MAE 정보도 함께 가져오기
                
                placeholders = ','.join(['?'] * len(regimes))
                query = f"""
                    SELECT profit, win_rate, quality_grade, avg_mfe, avg_mae
                    FROM strategies 
                    WHERE symbol = ? AND interval = ? 
                      AND (regime IN ({placeholders}) OR regime IS NULL)
                    ORDER BY 
                      CASE WHEN regime IN ({placeholders}) THEN 0 ELSE 1 END, -- 해당 레짐 우선
                      win_rate DESC, profit DESC 
                    LIMIT 1
                """
                
                # 파라미터: symbol, interval, *WHERE절 regimes, *ORDER BY절 regimes
                params = [coin, interval] + regimes + regimes
                
                try:
                    cursor = conn.execute(query, params)
                except sqlite3.OperationalError:
                    # avg_mfe/avg_mae 컬럼이 없는 구버전 DB 호환성
                    query_fallback = f"""
                        SELECT profit, win_rate, quality_grade, 0.0 as avg_mfe, 0.0 as avg_mae
                        FROM strategies 
                        WHERE symbol = ? AND interval = ? 
                        ORDER BY win_rate DESC, profit DESC 
                        LIMIT 1
                    """
                    cursor = conn.execute(query_fallback, (coin, interval))
                
                row = cursor.fetchone()
                if row:
                    # profit 값을 대략적인 퍼센트로 변환 (예: 10000 = 100%)
                    return {
                        'profit': row[0],
                        'win_rate': row[1],
                        'grade': row[2],
                        'avg_mfe': row[3] if row[3] else 0.0,
                        'avg_mae': row[4] if row[4] else 0.0,
                        # 🔥 중앙값 기반 평가 지표 (있다면) 추가 고려 가능
                        # 현재는 기존 컬럼 활용
                    }
            return None
        except Exception as e:
            # print(f"⚠️ 전략 조회 실패 ({coin}/{interval}): {e}") # 너무 시끄러울 수 있어 주석
            return None

    def _select_smart_strategy(self, coin: str, interval: str, market_condition: str, indicators: Dict) -> Optional[Dict]:
        """🚀 스마트 전략 선택 (RL Pipeline 학습 결과 활용)"""
        try:
            cache_key = f"smart_strategy_{coin}_{interval}_{market_condition}"
            cached_strategy = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            
            if cached_strategy:
                return cached_strategy
            
            # 🆕 실제 전략 DB에서 Top 전략 조회 (레짐 Fuzzy Matching)
            # market_condition을 레짐 형식으로 변환 (uptrend -> bullish 등)
            target_regimes = ['neutral']
            if market_condition == 'uptrend':
                target_regimes = ['extreme_bullish', 'bullish', 'sideways_bullish', 'neutral']
            elif market_condition == 'downtrend':
                target_regimes = ['extreme_bearish', 'bearish', 'sideways_bearish', 'neutral']
            elif market_condition == 'sideways':
                target_regimes = ['neutral', 'sideways_bullish', 'sideways_bearish']
            
            top_strategy = self._get_top_strategy_from_db(coin, interval, target_regimes)
            expected_profit_pct = 0.0
            
            if top_strategy:
                # 🔥 avg_mfe가 있다면 이를 우선적으로 기대 수익률로 사용
                if top_strategy.get('avg_mfe', 0) > 0:
                    # MFE는 퍼센트 단위로 저장됨 (예: 5.0 = 5%) -> 비율로 변환 (0.05)
                    expected_profit_pct = top_strategy['avg_mfe'] / 100.0
                else:
                    # 기존 profit 기반 추정 (폴백)
                    raw_profit = top_strategy.get('profit', 0)
                    if raw_profit > 0:
                        # 예: raw_profit이 500이면 -> 5% 가정 (조정 필요)
                        expected_profit_pct = min(0.1, max(0.01, raw_profit / 10000.0)) 
            
            # 🚀 1. 기본 전략 정보
            strategy = {
                'strategy_type': 'smart',
                'market_condition_bonus': 1.0,
                'risk_level': 'medium',
                'rl_pipeline_score': indicators.get('rl_pipeline_score', 0.5),
                'global_strategy_score': indicators.get('global_strategy_score', 0.5),
                'dna_similarity_score': indicators.get('dna_similarity_score', 0.5),
                'synergy_score': indicators.get('synergy_score', 0.5),
                'expected_profit_pct': expected_profit_pct, # 🆕 예상 수익률 추가
                'top_strategy_info': top_strategy # 디버깅용
            }
            
            # 🚀 2. 시장 상황별 보너스 (학습 결과 반영)
            if market_condition == 'uptrend':
                strategy['market_condition_bonus'] = 1.1
                strategy['risk_level'] = 'low' if strategy['rl_pipeline_score'] > 0.7 else 'medium'
            elif market_condition == 'downtrend':
                strategy['market_condition_bonus'] = 0.9
                strategy['risk_level'] = 'high' if strategy['rl_pipeline_score'] < 0.3 else 'medium'
            elif market_condition == 'sideways':
                strategy['market_condition_bonus'] = 1.0
                strategy['risk_level'] = 'medium'
            
            # 🚀 3. 학습 결과 기반 추가 보너스
            if strategy['rl_pipeline_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.1
            if strategy['global_strategy_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.05
            if strategy['dna_similarity_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.05
            if strategy['synergy_score'] > 0.8:
                strategy['market_condition_bonus'] *= 1.05
            
            # 캐시에 저장
            self.set_cached_data(cache_key, strategy)
            
            return strategy
            
        except Exception as e:
            return None
    
    def _apply_strategy_dominant_improvement(self, base_score: float, success_rate: float, avg_profit: float) -> float:
        """전략 중심 개선 적용 (피드백 신뢰도 낮은 경우)"""
        try:
            # 전략을 우선하되, 피드백으로 미세 조정
            if success_rate > 0.7:  # 매우 높은 성공률만 반영
                improved_score = base_score * 1.05
            elif success_rate < 0.3:  # 매우 낮은 성공률만 반영
                improved_score = base_score * 0.95
            else:
                improved_score = base_score
            
            # 수익률 보정 (미세하게만)
            if avg_profit > 3.0:
                improved_score *= 1.02
            elif avg_profit < -2.0:
                improved_score *= 0.98
            
            return improved_score
            
        except Exception as e:
            print(f"⚠️ 전략 중심 개선 적용 오류: {e}")
            return base_score
                
    # 🚀 기존 함수는 계산 방법 개선 방식으로 변경됨
    # def get_signal_feedback_score(self, coin: str, interval: str, candle: pd.Series) -> float:
    #     """매매 결과 피드백을 바탕으로 한 시그널 점수 계산 (더 이상 사용하지 않음)"""
    #     # 이 함수는 계산 방법 개선 방식으로 대체됨
    #     return 0.0
    

