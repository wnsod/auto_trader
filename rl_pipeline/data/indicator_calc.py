"""
지표 계산 모듈 - CPU 최적화
RSI/MACD/MFI/ATR/ADX/BB 등 지표 계산
GPU 실험 결과 CPU가 더 안정적이고 빠름
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from typing import Dict, List, Any, Optional
from rl_pipeline.core.errors import IndicatorError
from rl_pipeline.core.registry import register_indicator

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """CPU 최적화 지표 계산기"""
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, float] = {}  # 캐시 생성 시간 저장
        self.cache_timeout = int(os.getenv('INDICATOR_CACHE_TIMEOUT', '600'))  # 기본 10분
        self.max_cache_size = int(os.getenv('MAX_INDICATOR_CACHE_SIZE', '150'))  # 최대 캐시 항목 수
        logger.info("🚀 CPU 최적화 지표 계산기 초기화")
    
    @register_indicator("rsi")
    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산 (CPU 최적화)
        
        Args:
            prices: 가격 시리즈
            period: RSI 기간
            
        Returns:
            RSI 값들
        """
        try:
            start_time = time.time()
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ CPU RSI 계산 완료: {period}기간, {elapsed:.4f}초")
            return rsi
            
        except Exception as e:
            logger.error(f"❌ RSI 계산 실패: {e}")
            raise IndicatorError(f"RSI 계산 실패: {e}") from e
    
    @register_indicator("macd")
    def compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD 계산 (CPU 최적화)
        
        Args:
            prices: 가격 시리즈
            fast: 빠른 EMA 기간
            slow: 느린 EMA 기간
            signal: 시그널 라인 기간
            
        Returns:
            MACD 딕셔너리 (macd, signal, histogram)
        """
        try:
            start_time = time.time()
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ CPU MACD 계산 완료: {fast}/{slow}/{signal}, {elapsed:.4f}초")
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"❌ MACD 계산 실패: {e}")
            raise IndicatorError(f"MACD 계산 실패: {e}") from e
    
    @register_indicator("mfi")
    def compute_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """MFI (Money Flow Index) 계산 (CPU 최적화)
        
        Args:
            high: 고가 시리즈
            low: 저가 시리즈
            close: 종가 시리즈
            volume: 거래량 시리즈
            period: MFI 기간
            
        Returns:
            MFI 값들
        """
        try:
            start_time = time.time()
            
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = pd.Series(index=money_flow.index, dtype=float)
            negative_flow = pd.Series(index=money_flow.index, dtype=float)
            
            for i in range(1, len(typical_price)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                    negative_flow.iloc[i] = 0
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = 0
                    negative_flow.iloc[i] = money_flow.iloc[i]
                else:
                    positive_flow.iloc[i] = 0
                    negative_flow.iloc[i] = 0
            
            positive_flow_sum = positive_flow.rolling(window=period).sum()
            negative_flow_sum = negative_flow.rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + positive_flow_sum / negative_flow_sum))
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ CPU MFI 계산 완료: {period}기간, {elapsed:.4f}초")
            return mfi
            
        except Exception as e:
            logger.error(f"❌ MFI 계산 실패: {e}")
            raise IndicatorError(f"MFI 계산 실패: {e}") from e
    
    @register_indicator("atr")
    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산 (CPU 최적화)
        
        Args:
            high: 고가 시리즈
            low: 저가 시리즈
            close: 종가 시리즈
            period: ATR 기간
            
        Returns:
            ATR 값들
        """
        try:
            start_time = time.time()
            
            high_low = high - low
            high_close = np.abs(high - close.shift(1))
            low_close = np.abs(low - close.shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ CPU ATR 계산 완료: {period}기간, {elapsed:.4f}초")
            return atr
            
        except Exception as e:
            logger.error(f"❌ ATR 계산 실패: {e}")
            raise IndicatorError(f"ATR 계산 실패: {e}") from e
    
    @register_indicator("adx")
    def compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX (Average Directional Index) 계산 (CPU 최적화)
        
        Args:
            high: 고가 시리즈
            low: 저가 시리즈
            close: 종가 시리즈
            period: ADX 기간
            
        Returns:
            ADX 값들
        """
        try:
            start_time = time.time()
            
            # True Range 계산
            high_low = high - low
            high_close = np.abs(high - close.shift(1))
            low_close = np.abs(low - close.shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Directional Movement 계산
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            # Smoothed values
            atr = true_range.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # ADX 계산
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ CPU ADX 계산 완료: {period}기간, {elapsed:.4f}초")
            return adx
            
        except Exception as e:
            logger.error(f"❌ ADX 계산 실패: {e}")
            raise IndicatorError(f"ADX 계산 실패: {e}") from e
    
    @register_indicator("bb")
    def compute_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산 (CPU 최적화)
        
        Args:
            prices: 가격 시리즈
            period: 이동평균 기간
            std_dev: 표준편차 배수
            
        Returns:
            볼린저 밴드 딕셔너리 (upper, middle, lower)
        """
        try:
            start_time = time.time()
            
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ CPU 볼린저 밴드 계산 완료: {period}기간, {std_dev}σ, {elapsed:.4f}초")
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
            
        except Exception as e:
            logger.error(f"❌ 볼린저 밴드 계산 실패: {e}")
            raise IndicatorError(f"볼린저 밴드 계산 실패: {e}") from e
    
    def ensure_indicators(self, df: pd.DataFrame, coin: str, interval: str) -> pd.DataFrame:
        """데이터프레임에 필요한 지표들을 추가 (CPU 최적화)
        
        Args:
            df: OHLCV 데이터프레임
            coin: 코인 이름
            interval: 인터벌
            
        Returns:
            지표가 추가된 데이터프레임
        """
        try:
            start_time = time.time()
            
            # 캐시 키 생성
            cache_key = f"{coin}_{interval}_{len(df)}"
            current_time = time.time()
            
            # 캐시 확인 (타임아웃 체크 포함)
            if cache_key in self.cache:
                cache_age = current_time - self.cache_timestamps.get(cache_key, 0)
                
                # 타임아웃 체크
                if cache_age < self.cache_timeout:
                    logger.debug(f"📋 캐시에서 지표 로드: {cache_key} (나이: {cache_age:.1f}초)")
                    return self.cache[cache_key]
                else:
                    # 타임아웃된 캐시 제거
                    logger.debug(f"⏰ 지표 캐시 타임아웃: {cache_key} (나이: {cache_age:.1f}초 > {self.cache_timeout}초)")
                    del self.cache[cache_key]
                    if cache_key in self.cache_timestamps:
                        del self.cache_timestamps[cache_key]
            
            # 캐시 크기 제한 확인
            if len(self.cache) >= self.max_cache_size:
                self._cleanup_oldest_cache()
            
            result_df = df.copy()
            
            # 필요한 컬럼 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"⚠️ 누락된 컬럼: {missing_columns}")
                return result_df
            
            # RSI 계산
            if 'rsi' not in result_df.columns:
                result_df['rsi'] = self.compute_rsi(result_df['close'])
            
            # MACD 계산
            if 'macd' not in result_df.columns:
                macd_result = self.compute_macd(result_df['close'])
                result_df['macd'] = macd_result['macd']
                result_df['macd_signal'] = macd_result['signal']
                result_df['macd_histogram'] = macd_result['histogram']
            
            # MFI 계산
            if 'mfi' not in result_df.columns:
                result_df['mfi'] = self.compute_mfi(
                    result_df['high'], result_df['low'], 
                    result_df['close'], result_df['volume']
                )
            
            # ATR 계산
            if 'atr' not in result_df.columns:
                result_df['atr'] = self.compute_atr(
                    result_df['high'], result_df['low'], result_df['close']
                )
            
            # ADX 계산
            if 'adx' not in result_df.columns:
                result_df['adx'] = self.compute_adx(
                    result_df['high'], result_df['low'], result_df['close']
                )
            
            # 볼린저 밴드 계산
            if 'bb_upper' not in result_df.columns:
                bb_result = self.compute_bollinger_bands(result_df['close'])
                result_df['bb_upper'] = bb_result['upper']
                result_df['bb_middle'] = bb_result['middle']
                result_df['bb_lower'] = bb_result['lower']
            
            # 거래량 비율 계산
            if 'volume_ratio' not in result_df.columns:
                volume_ma = result_df['volume'].rolling(window=20).mean()
                result_df['volume_ratio'] = result_df['volume'] / volume_ma
            
            # 캐시에 저장
            self.cache[cache_key] = result_df
            self.cache_timestamps[cache_key] = current_time
            
            elapsed_time = time.time() - start_time
            logger.debug(f"✅ 지표 계산 완료: {coin} {interval} ({elapsed_time:.3f}초)")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ 지표 계산 실패: {e}")
            raise IndicatorError(f"지표 계산 실패: {e}") from e
    
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
            
            logger.debug(f"🧹 오래된 지표 캐시 {removed}개 제거 (총 {len(self.cache)}개 남음)")
            
        except Exception as e:
            logger.warning(f"⚠️ 오래된 지표 캐시 정리 실패: {e}")
    
    def clear_cache(self):
        """캐시 정리"""
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
                logger.info(f"🧹 타임아웃된 지표 캐시 {len(expired_keys)}개 제거")
            
            # 여전히 캐시가 많으면 오래된 것부터 제거
            if len(self.cache) > self.max_cache_size:
                self._cleanup_oldest_cache()
            
            logger.info("🧹 지표 계산 캐시 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 지표 캐시 정리 실패: {e}")

def ensure_indicators(df: pd.DataFrame, coin: str = None, interval: str = None) -> pd.DataFrame:
    """데이터프레임에 필요한 지표들을 추가 (독립 함수)
    
    Args:
        df: OHLCV 데이터프레임
        coin: 코인 이름 (선택사항)
        interval: 인터벌 (선택사항)
        
    Returns:
        지표가 추가된 데이터프레임
    """
    try:
        calculator = IndicatorCalculator()
        return calculator.ensure_indicators(df, coin or 'UNKNOWN', interval or 'UNKNOWN')
    except Exception as e:
        logger.error(f"❌ 지표 계산 실패: {e}")
        return df

def get_gpu_status() -> Dict[str, Any]:
    """GPU 상태 반환 (CPU 전용이므로 항상 False)"""
    return {
        'gpu_available': False,
        'gpu_device': None,
        'optimization_mode': 'CPU_ONLY',
        'reason': 'GPU 실험 결과 CPU가 더 안정적이고 빠름'
    }

def clear_cache():
    """캐시 정리"""
    calculator = get_indicator_calculator()
    calculator.clear_cache()

# 전역 인스턴스
_indicator_calculator: Optional[IndicatorCalculator] = None

def get_indicator_calculator() -> IndicatorCalculator:
    """지표 계산기 인스턴스 반환 (싱글톤)"""
    global _indicator_calculator
    if _indicator_calculator is None:
        _indicator_calculator = IndicatorCalculator()
    return _indicator_calculator

def compute_all_indicators(df: pd.DataFrame, coin: str, interval: str) -> pd.DataFrame:
    """모든 지표 계산 (편의 함수)"""
    calculator = get_indicator_calculator()
    return calculator.ensure_indicators(df, coin, interval)