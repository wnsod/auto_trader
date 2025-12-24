"""
cache 관련 Mixin 클래스
SignalSelector의 cache 기능을 담당합니다.
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


class CacheMixin:
    """
    CacheMixin - cache 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

    def _get_cached_market_condition(self, coin: str, interval: str) -> str:
        """🚀 캐시된 시장 상황 반환 (빠른 판단)"""
        try:
            cache_key = f"market_condition_{coin}_{interval}"
            cached_data = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            
            if cached_data:
                return cached_data
            
            # 캐시가 없으면 간단한 시장 상황 판단
            market_condition = self._detect_simple_market_condition(coin, interval)
            
            # 캐시에 저장
            self.set_cached_data(cache_key, market_condition)
            
            return market_condition
            
        except Exception as e:
            return 'neutral'  # 기본값
    
    def _cleanup_cache(self):
        """🚀 고성능 캐시 정리 (메모리 최적화)"""
        try:
            current_time = time.time()
            expired_keys = []

            # OptimizedCache에서 타임스탬프와 캐시 정보 가져오기
            with self.cache.lock:
                cache_items = list(self.cache.cache.items())
                cache_timestamps = dict(self.cache.timestamps)

            # 🚀 캐시 크기 제한 적용
            if len(cache_items) > self.max_cache_size:
                # 가장 오래된 항목들부터 제거
                sorted_items = sorted(cache_timestamps.items(), key=lambda x: x[1])
                items_to_remove = len(cache_items) - self.max_cache_size + 1000  # 여유 공간 확보
                expired_keys.extend([key for key, _ in sorted_items[:items_to_remove]])

            # 기존 만료 시간 기반 정리
            for key, timestamp in cache_timestamps.items():
                if current_time - timestamp > 600:  # 10분 이상 사용되지 않은 항목
                    expired_keys.append(key)

            # 중복 제거
            expired_keys = list(set(expired_keys))

            # 만료된 항목 삭제
            for key in expired_keys:
                try:
                    del self.cache[key]
                    self._cache_stats['evictions'] += 1
                except:
                    pass

            if expired_keys:
                print(f"🧹 고성능 캐시 정리: {len(expired_keys)}개 항목 제거 (캐시 크기: {len(self.cache):,})")

            self._signal_stats['last_cleanup'] = current_time
        except Exception as e:
            print(f"⚠️ 캐시 정리 오류: {e}")

    def get_cached_data(self, key: str, max_age: int = 300) -> Optional[Any]:
        """🚀 최적화된 캐시 데이터 조회"""
        return self.cache.get(key, max_age)

    def set_cached_data(self, key: str, data: Any):
        """🚀 최적화된 캐시 데이터 저장"""
        self.cache.set(key, data)

    def cleanup_old_signals(self, max_hours: int = 24):
        """오래된 시그널 정리 (성능 최적화)"""
        try:
            current_timestamp = int(datetime.now().timestamp())
            cutoff_timestamp = current_timestamp - (max_hours * 3600)
            
            with sqlite3.connect(DB_PATH) as conn:
                # 오래된 시그널 삭제
                deleted_count = conn.execute("""
                    DELETE FROM signals 
                    WHERE timestamp < ?
                """, (cutoff_timestamp,)).rowcount
                
                conn.commit()
                
                if deleted_count > 0:
                    print(f"🧹 오래된 시그널 정리: {deleted_count}개 삭제 (>{max_hours}시간 전)")
                else:
                    print(f"ℹ️ 정리할 오래된 시그널 없음 (>{max_hours}시간 전)")
                    
        except Exception as e:
            print(f"⚠️ 시그널 정리 오류: {e}")
    

