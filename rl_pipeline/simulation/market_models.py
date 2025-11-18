"""
시장 모델 모듈
- Action: 거래 행동 enum
- MarketState: 시장 상태
- AgentState: 에이전트 상태
- MarketDataGenerator: 시장 데이터 생성기
"""

from datetime import datetime, timedelta
import logging
import random
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class Action(Enum):
    """매매 행동"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class MarketState:
    """시장 상태"""
    timestamp: datetime
    price: float
    volume: float
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    volume_ratio: float
    regime_stage: int  # 1-7 단계 레짐
    regime_label: str  # "extreme_bearish", "bearish", "sideways_bearish", "neutral", "sideways_bullish", "bullish", "extreme_bullish"
    regime_confidence: float  # 레짐 신뢰도
    volatility: float
    # 실제 전략에서 사용하는 추가 지표들
    mfi: float = 50.0      # 자금흐름지수
    atr: float = 0.02      # 평균진정범위
    adx: float = 25.0     # 평균방향성지수

@dataclass
class AgentState:
    """에이전트 상태"""
    balance: float
    position: Optional[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    strategy_params: Dict[str, Any]

class MarketDataGenerator:
    """가상 시장 데이터 생성기"""
    
    def __init__(self, base_price: float = 50000.0):
        self.base_price = base_price
        self.current_price = base_price
        self.current_time = datetime.now()
        self.price_history = [base_price]
        self.volume_history = [1000000.0]
        
        # 시장 패턴 파라미터 (더 현실적인 시장 생성)
        self.trend_strength = np.random.uniform(-0.5, 0.5)  # 트렌드 강도 (-0.5 ~ 0.5)
        self.volatility = np.random.uniform(0.01, 0.05)     # 변동성 (1% ~ 5%)
        self.noise_level = 0.005                            # 노이즈 레벨 감소
        self.trend_duration = np.random.randint(50, 200)    # 트렌드 지속 기간
        self.trend_counter = 0                              # 트렌드 카운터
        
    def generate_next_candle(self) -> MarketState:
        """다음 캔들 데이터 생성"""
        try:
            # 트렌드 지속 기간 체크 및 변경
            self.trend_counter += 1
            if self.trend_counter >= self.trend_duration:
                # 새로운 트렌드 생성
                self.trend_strength = np.random.uniform(-0.5, 0.5)
                self.trend_duration = np.random.randint(50, 200)
                self.trend_counter = 0
                logger.debug(f"🔄 새로운 트렌드 시작: {self.trend_strength:.3f}")
            
            # 가격 움직임 생성 (트렌드 + 노이즈)
            trend_component = self.trend_strength * self.current_price * 0.001
            noise_component = np.random.normal(0, self.volatility * self.current_price)
            
            price_change = trend_component + noise_component
            self.current_price = max(self.current_price + price_change, 1000.0)  # 최소 가격 보장
            
            # 볼륨 생성 (가격 변동에 비례)
            volume_multiplier = 1.0 + abs(price_change) / self.current_price * 10
            current_volume = self.volume_history[-1] * volume_multiplier * np.random.uniform(0.8, 1.2)
            
            # 시간 업데이트
            self.current_time += timedelta(minutes=15)
            
            # 히스토리 업데이트
            self.price_history.append(self.current_price)
            self.volume_history.append(current_volume)
            
            # 기술지표 계산
            rsi = self._calculate_rsi()
            macd, macd_signal = self._calculate_macd()
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands()
            volume_ratio = self._calculate_volume_ratio()
            regime_stage, regime_label, regime_confidence = self._determine_regime()
            volatility = self._calculate_volatility()
            
            # 실제 전략에서 사용하는 추가 지표들 계산
            mfi = self._calculate_mfi()
            atr = self._calculate_atr()
            adx = self._calculate_adx()
            
            return MarketState(
                timestamp=self.current_time,
                price=self.current_price,
                volume=current_volume,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                volume_ratio=volume_ratio,
                regime_stage=regime_stage,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                volatility=volatility,
                mfi=mfi,
                atr=atr,
                adx=adx
            )
            
        except Exception as e:
            logger.error(f"❌ 캔들 데이터 생성 실패: {e}")
            return self._create_default_state()
    
    def _calculate_rsi(self, period: int = 14) -> float:
        """RSI 계산"""
        if len(self.price_history) < period + 1:
            return 50.0
        
        prices = np.array(self.price_history[-period-1:])
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return max(0, min(100, rsi))
    
    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """MACD 계산"""
        if len(self.price_history) < slow:
            return 0.0, 0.0
        
        prices = np.array(self.price_history[-slow:])
        
        # EMA 계산
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = [data[0]]
            for i in range(1, len(data)):
                ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])
            return ema_values[-1]
        
        ema_fast = ema(prices[-fast:], fast)
        ema_slow = ema(prices, slow)
        macd = ema_fast - ema_slow
        
        # MACD 시그널 계산 (간단화)
        macd_signal = macd * 0.9  # 단순화된 시그널
        
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """볼린저 밴드 계산"""
        if len(self.price_history) < period:
            price = self.current_price
            return price * 1.02, price, price * 0.98
        
        prices = np.array(self.price_history[-period:])
        middle = np.mean(prices)
        std = np.std(prices)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def _calculate_volume_ratio(self, period: int = 20) -> float:
        """볼륨 비율 계산"""
        if len(self.volume_history) < period:
            return 1.0
        
        current_volume = self.volume_history[-1]
        avg_volume = np.mean(self.volume_history[-period:])
        
        # Division by zero 및 오버플로우 방지
        if avg_volume <= 0 or not np.isfinite(avg_volume):
            return 1.0
        
        ratio = current_volume / avg_volume
        
        # 무한대 값 방지
        if not np.isfinite(ratio):
            return 1.0
        
        return min(ratio, 100.0)  # 비율 제한
    
    def _determine_regime(self, period: int = 20) -> Tuple[int, str, float]:
        """🚀 새로운 통합 레짐 시스템 사용"""
        try:
            # 새로운 레짐 시스템에서 레짐 정보 가져오기
            # 실제 구현에서는 DB에서 최신 레짐 정보를 가져와야 함
            # 현재는 간단한 폴백 로직 사용
            if len(self.price_history) < period:
                return 4, "neutral", 0.5
            
            prices = np.array(self.price_history[-period:])
            slope = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # RSI 계산
            rsi = self._calculate_rsi()
            
            # MACD 계산
            macd, _ = self._calculate_macd()
            
            # 레짐 분류 (7단계)
            if slope > self.current_price * 0.002 and rsi > 70:
                return 7, "extreme_bullish", 0.9
            elif slope > self.current_price * 0.001 and rsi > 60:
                return 6, "bullish", 0.8
            elif slope > self.current_price * 0.0005 and rsi > 50:
                return 5, "sideways_bullish", 0.7
            elif abs(slope) < self.current_price * 0.0005 and 40 < rsi < 60:
                return 4, "neutral", 0.6
            elif slope < -self.current_price * 0.0005 and rsi < 50:
                return 3, "sideways_bearish", 0.7
            elif slope < -self.current_price * 0.001 and rsi < 40:
                return 2, "bearish", 0.8
            elif slope < -self.current_price * 0.002 and rsi < 30:
                return 1, "extreme_bearish", 0.9
            else:
                return 4, "neutral", 0.5
                
        except Exception as e:
            return 4, "neutral", 0.5
    
    def _calculate_volatility(self, period: int = 20) -> float:
        """변동성 계산"""
        if len(self.price_history) < period:
            return 0.02
        
        prices = np.array(self.price_history[-period:])
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        return max(0.001, min(0.1, volatility))  # 범위 제한
    
    def _calculate_mfi(self, period: int = 14) -> float:
        """자금흐름지수(MFI) 계산"""
        try:
            if len(self.price_history) < period or len(self.volume_history) < period:
                return 50.0
            
            prices = np.array(self.price_history[-period:])
            volumes = np.array(self.volume_history[-period:])
            
            # 전형가격 계산
            typical_prices = prices  # 단순화: 종가 사용
            
            # 자금흐름 계산 (오버플로우 방지)
            positive_flow = np.float64(0)
            negative_flow = np.float64(0)
            
            for i in range(1, len(typical_prices)):
                # 오버플로우 방지: 값이 너무 크면 스케일 조정
                price_value = np.float64(typical_prices[i])
                volume_value = np.float64(volumes[i])
                
                # 값 크기 확인 및 스케일 조정
                if price_value > 1e100 or volume_value > 1e100:
                    # 값이 너무 크면 스킵
                    continue
                
                # 안전한 곱셈
                flow_value = price_value * volume_value
                
                if not np.isfinite(flow_value):
                    continue
                
                if typical_prices[i] > typical_prices[i-1]:
                    # 오버플로우 방지
                    positive_flow = min(positive_flow + flow_value, np.finfo(np.float64).max)
                elif typical_prices[i] < typical_prices[i-1]:
                    negative_flow = min(negative_flow + flow_value, np.finfo(np.float64).max)
            
            # Division by zero 방지
            if negative_flow == 0 or not np.isfinite(positive_flow) or not np.isfinite(negative_flow):
                return 50.0  # 중간값 반환
            
            money_ratio = positive_flow / negative_flow
            
            # 무한대 값 방지
            if not np.isfinite(money_ratio) or money_ratio <= 0:
                return 50.0
            
            mfi = 100 - (100 / (1 + money_ratio))
            
            return max(0, min(100, mfi))
            
        except Exception as e:
            logger.warning(f"⚠️ MFI 계산 실패: {e}")
            return 50.0
    
    def _calculate_atr(self, period: int = 14) -> float:
        """평균진정범위(ATR) 계산"""
        try:
            if len(self.price_history) < period:
                return 0.02
            
            prices = np.array(self.price_history[-period:])
            
            # True Range 계산 (단순화: 가격 변동폭 사용)
            true_ranges = []
            for i in range(1, len(prices)):
                tr = abs(prices[i] - prices[i-1])
                true_ranges.append(tr)
            
            if not true_ranges:
                return 0.02
            
            atr = np.mean(true_ranges) / self.current_price  # 정규화
            
            return max(0.001, min(0.1, atr))
            
        except Exception as e:
            logger.warning(f"⚠️ ATR 계산 실패: {e}")
            return 0.02
    
    def _calculate_adx(self, period: int = 14) -> float:
        """평균방향성지수(ADX) 계산"""
        try:
            if len(self.price_history) < period:
                return 25.0
            
            prices = np.array(self.price_history[-period:])
            
            # 방향성 이동 계산 (단순화)
            positive_dm = 0
            negative_dm = 0
            
            for i in range(1, len(prices)):
                price_change = prices[i] - prices[i-1]
                if price_change > 0:
                    positive_dm += price_change
                elif price_change < 0:
                    negative_dm += abs(price_change)
            
            # ADX 계산 (단순화)
            total_movement = positive_dm + negative_dm
            if total_movement == 0:
                return 25.0
            
            adx = (abs(positive_dm - negative_dm) / total_movement) * 100
            
            return max(0, min(100, adx))
            
        except Exception as e:
            logger.warning(f"⚠️ ADX 계산 실패: {e}")
            return 25.0
    
    def _create_default_state(self) -> MarketState:
        """기본 상태 생성"""
        return MarketState(
            timestamp=self.current_time,
            price=self.current_price,
            volume=1000000.0,
            rsi=50.0,
            macd=0.0,
            macd_signal=0.0,
            bb_upper=self.current_price * 1.02,
            bb_middle=self.current_price,
            bb_lower=self.current_price * 0.98,
            volume_ratio=1.0,
            regime_stage=4,
            regime_label="neutral",
            regime_confidence=0.5,
            volatility=0.02,
            mfi=50.0,
            atr=0.02,
            adx=25.0
        )
    
    def update_market_regime(self, regime_label: str):
        """🚀 새로운 7단계 레짐 시스템으로 시장 체제 변경"""
        if regime_label == "extreme_bullish":
            self.trend_strength = 0.002
            self.volatility = 0.01
        elif regime_label == "bullish":
            self.trend_strength = 0.001
            self.volatility = 0.015
        elif regime_label == "sideways_bullish":
            self.trend_strength = 0.0005
            self.volatility = 0.02
        elif regime_label == "neutral":
            self.trend_strength = 0.0
            self.volatility = 0.02
        elif regime_label == "sideways_bearish":
            self.trend_strength = -0.0005
            self.volatility = 0.02
        elif regime_label == "bearish":
            self.trend_strength = -0.001
            self.volatility = 0.025
        elif regime_label == "extreme_bearish":
            self.trend_strength = -0.002
            self.volatility = 0.03
        else:  # 기본값
            self.trend_strength = 0.0
            self.volatility = 0.02

