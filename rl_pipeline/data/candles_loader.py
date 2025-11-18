"""
캔들 데이터 로더
캔들 데이터 로딩 및 캐싱 관리
"""

import pandas as pd
import logging
import time
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from rl_pipeline.db.reads import fetch_df, get_candle_data
from rl_pipeline.core.errors import DataLoadError
from rl_pipeline.core.types import CandleData
from rl_pipeline.core.env import config

logger = logging.getLogger(__name__)

class CandlesLoader:
    """캔들 데이터 로더"""
    
    def __init__(self):
        self.cache: Dict[str, CandleData] = {}
        self.cache_timestamps: Dict[str, float] = {}  # 캐시 생성 시간 저장
        self.cache_timeout = config.CACHE_TIMEOUT
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE', '200'))  # 최대 캐시 항목 수
    
    def load_candles(self, coin: str, interval: str, days: int = 30) -> pd.DataFrame:
        """캔들 데이터 로드
        
        Args:
            coin: 코인 심볼 (예: "BTC")
            interval: 시간 간격 (예: "15m")
            days: 로드할 일수
            
        Returns:
            캔들 데이터프레임
            
        Example:
            df = load_candles("BTC", "15m", 30)
        """
        try:
            cache_key = f"{coin}_{interval}_{days}"
            current_time = time.time()
            
            # 캐시 확인 (타임아웃 체크 포함)
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                cache_age = current_time - self.cache_timestamps.get(cache_key, 0)
                
                # 타임아웃 체크
                if cache_age < self.cache_timeout:
                    logger.debug(f"♻️ 캐시된 캔들 데이터 사용: {cache_key} (나이: {cache_age:.1f}초)")
                    return cached_data.data
                else:
                    # 타임아웃된 캐시 제거
                    logger.debug(f"⏰ 캐시 타임아웃: {cache_key} (나이: {cache_age:.1f}초 > {self.cache_timeout}초)")
                    del self.cache[cache_key]
                    del self.cache_timestamps[cache_key]
            
            # 캐시 크기 제한 확인
            if len(self.cache) >= self.max_cache_size:
                self._cleanup_oldest_cache()
            
            # 데이터베이스에서 로드
            df = get_candle_data(coin, interval, days)
            
            if df.empty:
                logger.warning(f"⚠️ 캔들 데이터가 비어있음: {coin} {interval}")
                return df
            
            # 캐시에 저장
            candle_data = CandleData(
                coin=coin,
                interval=interval,
                data=df,
                cache_key=cache_key
            )
            self.cache[cache_key] = candle_data
            self.cache_timestamps[cache_key] = current_time
            
            logger.info(f"✅ 캔들 데이터 로드 완료: {coin} {interval} ({len(df)}행)")
            return df
            
        except Exception as e:
            logger.error(f"❌ 캔들 데이터 로드 실패: {e}")
            raise DataLoadError(f"캔들 데이터 로드 실패 ({coin} {interval}): {e}") from e
    
    def load_candles_batch(self, coins: List[str], intervals: List[str], days: int = 30) -> Dict[Tuple[str, str], pd.DataFrame]:
        """배치로 여러 코인의 캔들 데이터 로드
        
        Args:
            coins: 코인 목록
            intervals: 인터벌 목록
            days: 로드할 일수
            
        Returns:
            {(coin, interval): DataFrame} 형태의 딕셔너리
        """
        try:
            result = {}
            
            for coin in coins:
                for interval in intervals:
                    try:
                        df = self.load_candles(coin, interval, days)
                        result[(coin, interval)] = df
                    except Exception as e:
                        logger.warning(f"⚠️ {coin} {interval} 캔들 데이터 로드 실패: {e}")
                        result[(coin, interval)] = pd.DataFrame()
            
            logger.info(f"✅ 배치 캔들 데이터 로드 완료: {len(result)}개 조합")
            return result
            
        except Exception as e:
            logger.error(f"❌ 배치 캔들 데이터 로드 실패: {e}")
            raise DataLoadError(f"배치 캔들 데이터 로드 실패: {e}") from e
    
    def get_cached_candle_data(self, coin: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """캐시된 캔들 데이터 조회"""
        cache_key = f"{coin}_{interval}_{days}"
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # 요청된 기간만큼 슬라이싱하여 반환
            df = cached_data.data
            if len(df) > days * 24 * 4:  # 15분 간격 기준
                return df.head(days * 24 * 4)
            return df
        
        return None
    
    def _cleanup_oldest_cache(self):
        """가장 오래된 캐시 항목 제거"""
        try:
            if not self.cache_timestamps:
                return
            
            # 타임스탬프 기준으로 정렬하여 가장 오래된 항목 제거
            sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            
            # 가장 오래된 25% 제거 (최소 1개)
            remove_count = max(1, len(sorted_items) // 4)
            removed = 0
            
            for cache_key, _ in sorted_items[:remove_count]:
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    del self.cache_timestamps[cache_key]
                    removed += 1
            
            logger.debug(f"🧹 오래된 캐시 {removed}개 제거 (총 {len(self.cache)}개 남음)")
            
        except Exception as e:
            logger.warning(f"⚠️ 오래된 캐시 정리 실패: {e}")
    
    def cleanup_cached_data(self):
        """캐시된 데이터 정리 (메모리 최적화)"""
        try:
            current_time = time.time()
            
            # 타임아웃된 모든 캐시 항목 제거
            expired_keys = []
            for cache_key, timestamp in self.cache_timestamps.items():
                cache_age = current_time - timestamp
                if cache_age >= self.cache_timeout:
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                if cache_key in self.cache:
                    del self.cache[cache_key]
                if cache_key in self.cache_timestamps:
                    del self.cache_timestamps[cache_key]
            
            if expired_keys:
                logger.info(f"🧹 타임아웃된 캐시 {len(expired_keys)}개 제거")
            
            # 여전히 캐시가 많으면 오래된 것부터 제거
            if len(self.cache) > self.max_cache_size:
                self._cleanup_oldest_cache()
            
        except Exception as e:
            logger.warning(f"⚠️ 캐시 정리 실패: {e}")
    
    def load_candle_data_sequential(self, coin: str, intervals: List[str], days: int = 14) -> Dict[str, pd.DataFrame]:
        """캔들 데이터 순차 로드 (병렬 처리 제거)"""
        try:
            result = {}
            
            for interval in intervals:
                try:
                    df = self.load_candles(coin, interval, days)
                    result[interval] = df
                    logger.debug(f"✅ {coin} {interval} 순차 로드 완료: {len(df)}행")
                except Exception as e:
                    logger.warning(f"⚠️ {coin} {interval} 순차 로드 실패: {e}")
                    result[interval] = pd.DataFrame()
            
            # 주기적으로 캐시 정리
            if len(self.cache) > self.max_cache_size * 0.8:
                self.cleanup_cached_data()
            
            logger.info(f"✅ {coin} 순차 캔들 데이터 로드 완료: {len(result)}개 인터벌")
            return result
            
        except Exception as e:
            logger.error(f"❌ {coin} 순차 캔들 데이터 로드 실패: {e}")
            raise DataLoadError(f"순차 캔들 데이터 로드 실패: {e}") from e

# 전역 인스턴스
_candles_loader: Optional[CandlesLoader] = None

def get_candles_loader() -> CandlesLoader:
    """캔들 로더 인스턴스 반환"""
    global _candles_loader
    if _candles_loader is None:
        _candles_loader = CandlesLoader()
    return _candles_loader

# 편의 함수들
def load_candles(coin: str, interval: str, days: int = 30) -> pd.DataFrame:
    """캔들 데이터 로드 (편의 함수)"""
    loader = get_candles_loader()
    return loader.load_candles(coin, interval, days)

def load_candles_batch(coins: List[str], intervals: List[str], days: int = 30) -> Dict[Tuple[str, str], pd.DataFrame]:
    """배치 캔들 데이터 로드 (편의 함수)"""
    loader = get_candles_loader()
    return loader.load_candles_batch(coins, intervals, days)

def get_cached_candle_data(coin: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """캐시된 캔들 데이터 조회 (편의 함수)"""
    loader = get_candles_loader()
    return loader.get_cached_candle_data(coin, interval, days)

def cleanup_cached_data():
    """캐시 정리 (편의 함수)"""
    loader = get_candles_loader()
    loader.cleanup_cached_data()
