"""
백테스트 결과 캐싱 모듈 (성능 최적화)
"""

import logging
import hashlib
import pickle
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
from threading import Lock

logger = logging.getLogger(__name__)

class BacktestCache:
    """백테스트 결과 캐싱 (스레드 안전)"""
    
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Args:
            cache_ttl_hours: 캐시 유효 시간 (기본: 24시간)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._lock = Lock()  # 🔥 스레드 안전성을 위한 Lock
        # 🔥 DEBUG 레벨로 변경 (전역 싱글톤이므로 초기화 로그는 불필요)
        logger.debug(f"🚀 백테스트 캐시 초기화 (TTL: {cache_ttl_hours}시간)")
    
    def _generate_cache_key(
        self,
        strategy: Dict[str, Any],
        candle_data_hash: str,
        regime: Optional[str] = None
    ) -> str:
        """캐시 키 생성"""
        try:
            # 전략 파라미터 추출
            strategy_params = {
                'rsi_min': strategy.get('rsi_min', 30),
                'rsi_max': strategy.get('rsi_max', 70),
                'stop_loss_pct': strategy.get('stop_loss_pct', 0.02),
                'take_profit_pct': strategy.get('take_profit_pct', 0.04),
                'strategy_type': strategy.get('strategy_type', 'unknown')
            }
            
            # 전략 ID 포함 (없으면 파라미터 기반 해시)
            strategy_id = strategy.get('id') or strategy.get('strategy_id', '')
            if strategy_id:
                key_str = f"{strategy_id}:{candle_data_hash}"
            else:
                params_str = str(sorted(strategy_params.items()))
                strategy_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                key_str = f"{strategy_hash}:{candle_data_hash}"
            
            if regime:
                key_str += f":{regime}"
            
            return hashlib.md5(key_str.encode()).hexdigest()
            
        except Exception as e:
            logger.debug(f"캐시 키 생성 실패: {e}")
            return hashlib.md5(str(strategy).encode()).hexdigest()
    
    def _hash_candle_data(self, candle_data) -> str:
        """캔들 데이터 해시 생성 (최근 N개만 사용)"""
        try:
            import pandas as pd
            
            if candle_data is None or len(candle_data) == 0:
                return "empty"
            
            # 최근 100개만 사용 (성능 고려)
            recent_data = candle_data.tail(100) if len(candle_data) > 100 else candle_data
            
            # 중요한 컬럼만 사용
            key_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in key_cols if col in recent_data.columns]
            
            if not available_cols:
                return "no_cols"
            
            # 데이터 요약 (첫/마지막/평균)
            summary = {
                'first_close': float(recent_data['close'].iloc[0]) if 'close' in recent_data.columns else 0,
                'last_close': float(recent_data['close'].iloc[-1]) if 'close' in recent_data.columns else 0,
                'avg_volume': float(recent_data['volume'].mean()) if 'volume' in recent_data.columns else 0,
                'len': len(recent_data)
            }
            
            summary_str = str(sorted(summary.items()))
            return hashlib.md5(summary_str.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.debug(f"캔들 데이터 해시 생성 실패: {e}")
            return "error"
    
    def get(
        self,
        strategy: Dict[str, Any],
        candle_data,
        regime: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """캐시에서 백테스트 결과 조회 (스레드 안전)"""
        try:
            candle_hash = self._hash_candle_data(candle_data)
            cache_key = self._generate_cache_key(strategy, candle_hash, regime)
            
            # 🔥 Lock으로 동시 접근 방지
            with self._lock:
                if cache_key in self.cache:
                    cached_item = self.cache[cache_key]
                    
                    # TTL 확인
                    cache_time = cached_item.get('timestamp')
                    if cache_time:
                        if datetime.now() - cache_time < self.cache_ttl:
                            logger.debug(f"✅ 캐시 히트: {cache_key[:8]}...")
                            return cached_item.get('result')
                        else:
                            # 만료된 캐시 삭제
                            del self.cache[cache_key]
                            logger.debug(f"⏰ 캐시 만료: {cache_key[:8]}...")
            
            return None
            
        except Exception as e:
            logger.debug(f"캐시 조회 실패: {e}")
            return None
    
    def set(
        self,
        strategy: Dict[str, Any],
        candle_data,
        result: Dict[str, Any],
        regime: Optional[str] = None
    ):
        """백테스트 결과 캐시 저장 (스레드 안전)"""
        try:
            candle_hash = self._hash_candle_data(candle_data)
            cache_key = self._generate_cache_key(strategy, candle_hash, regime)
            
            # 🔥 Lock으로 동시 접근 방지
            with self._lock:
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
                
                logger.debug(f"💾 캐시 저장: {cache_key[:8]}...")
                
                # 캐시 크기 제한 (최대 1000개)
                if len(self.cache) > 1000:
                    # 가장 오래된 항목 삭제
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get('timestamp', datetime.min))
                    del self.cache[oldest_key]
                    logger.debug(f"🗑️ 캐시 정리: {oldest_key[:8]}...")
            
        except Exception as e:
            logger.debug(f"캐시 저장 실패: {e}")
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()
        logger.info("🗑️ 백테스트 캐시 전체 삭제")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            'cache_size': len(self.cache),
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600
        }

# 전역 캐시 인스턴스
_global_cache: Optional[BacktestCache] = None

def get_backtest_cache(cache_ttl_hours: int = 24) -> BacktestCache:
    """전역 백테스트 캐시 인스턴스 반환"""
    global _global_cache
    if _global_cache is None:
        _global_cache = BacktestCache(cache_ttl_hours=cache_ttl_hours)
    return _global_cache
