"""
단순화된 전략 모듈
- 파라미터 수 감소 (15개 → 4개)
- 과적합 방지
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedStrategy:
    """단순화된 전략 (4개 파라미터만 사용)"""
    
    trend_period: int = 20        # MA 기간
    volatility_period: int = 14   # ATR 기간
    entry_threshold: float = 0.6  # 진입 임계값
    exit_threshold: float = 0.4   # 청산 임계값
    
    def generate_signal(self, market_data: Dict) -> str:
        """신호 생성 (단순화)"""
        
        try:
            prices = market_data.get('close', [])
            if len(prices) < max(self.trend_period, self.volatility_period):
                return "HOLD"
            
            # 1. 트렌드 계산 (MA만 사용)
            ma = np.mean(prices[-self.trend_period:])
            current_price = prices[-1]
            
            trend_strength = (current_price - ma) / ma if ma > 0 else 0.0
            
            # 2. 변동성 계산 (ATR만 사용)
            atr = self._calculate_atr(market_data)
            volatility = atr / current_price if current_price > 0 else 0.0
            
            # 3. 신호 생성 (단순 로직)
            if trend_strength > self.entry_threshold and volatility < 0.05:
                return "BUY"
            elif trend_strength < -self.exit_threshold:
                return "SELL"
            else:
                return "HOLD"
        
        except Exception as e:
            logger.error(f"신호 생성 실패: {e}")
            return "HOLD"
    
    def _calculate_atr(self, market_data: Dict, period: Optional[int] = None) -> float:
        """ATR 계산"""
        
        if period is None:
            period = self.volatility_period
        
        try:
            highs = market_data.get('high', [])
            lows = market_data.get('low', [])
            closes = market_data.get('close', [])
            
            if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
                return 0.0
            
            true_ranges = []
            for i in range(1, len(highs)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(tr)
            
            if len(true_ranges) < period:
                return 0.0
            
            return np.mean(true_ranges[-period:])
        
        except Exception as e:
            logger.error(f"ATR 계산 실패: {e}")
            return 0.0
    
    def get_strategy_params(self) -> Dict:
        """전략 파라미터 반환"""
        return {
            'trend_period': self.trend_period,
            'volatility_period': self.volatility_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold
        }
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return self.get_strategy_params()


def compare_simplified_vs_complex(
    coin: str,
    interval: str,
    test_data: pd.DataFrame,
    complex_strategies: List[Dict]
) -> Dict:
    """단순 전략 vs 복잡 전략 비교"""
    
    try:
        # 단순 전략
        simple_strategy = SimplifiedStrategy()
        
        # 간단한 백테스트
        simple_result = backtest_simplified_strategy(simple_strategy, test_data)
        
        # 복잡 전략 평균
        complex_result = get_average_complex_strategy_result(complex_strategies)
        
        comparison = {
            'coin': coin,
            'interval': interval,
            'simple': simple_result,
            'complex': complex_result,
            'simple_better': simple_result.get('profit', 0) > complex_result.get('profit', 0)
        }
        
        logger.info(f"\n{coin}-{interval} 전략 비교:")
        logger.info(f"  단순 전략: {simple_result.get('profit', 0):.2f}% (파라미터 4개)")
        logger.info(f"  복잡 전략: {complex_result.get('profit', 0):.2f}% (파라미터 15개)")
        
        if comparison['simple_better']:
            logger.info("  ✅ 단순 전략이 우수")
        else:
            logger.info("  ⚠️ 복잡 전략이 우수 (단, 과적합 가능성 고려 필요)")
        
        return comparison
    
    except Exception as e:
        logger.error(f"❌ 전략 비교 실패: {e}")
        return {'error': str(e)}


def backtest_simplified_strategy(
    strategy: SimplifiedStrategy,
    test_data: pd.DataFrame
) -> Dict:
    """단순 전략 백테스트"""
    
    try:
        if test_data.empty or len(test_data) < 50:
            return {'profit': 0.0, 'trades': 0, 'win_rate': 0.0}
        
        trades = []
        position = None
        entry_price = 0.0
        
        # 가격 데이터 준비
        prices = test_data['close'].tolist()
        highs = test_data['high'].tolist()
        lows = test_data['low'].tolist()
        
        market_data = {
            'close': prices,
            'high': highs,
            'low': lows
        }
        
        # 시뮬레이션
        for i in range(max(strategy.trend_period, strategy.volatility_period), len(test_data)):
            # 현재까지의 데이터로 신호 생성
            current_market_data = {
                'close': prices[:i+1],
                'high': highs[:i+1],
                'low': lows[:i+1]
            }
            
            signal = strategy.generate_signal(current_market_data)
            current_price = prices[i]
            
            if signal == "BUY" and position is None:
                position = "LONG"
                entry_price = current_price
            elif signal == "SELL" and position == "LONG":
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_pct': profit_pct
                })
                position = None
        
        # 결과 계산
        if not trades:
            return {'profit': 0.0, 'trades': 0, 'win_rate': 0.0}
        
        total_profit = sum(t['profit_pct'] for t in trades)
        winning_trades = [t for t in trades if t['profit_pct'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        return {
            'profit': total_profit,
            'trades': len(trades),
            'win_rate': win_rate
        }
    
    except Exception as e:
        logger.error(f"❌ 백테스트 실패: {e}")
        return {'profit': 0.0, 'trades': 0, 'win_rate': 0.0}


def get_average_complex_strategy_result(complex_strategies: List[Dict]) -> Dict:
    """복잡 전략 평균 결과"""
    
    if not complex_strategies:
        return {'profit': 0.0, 'trades': 0, 'win_rate': 0.0}
    
    profits = [s.get('profit', 0) for s in complex_strategies if s.get('profit') is not None]
    win_rates = [s.get('win_rate', 0) for s in complex_strategies if s.get('win_rate') is not None]
    
    return {
        'profit': np.mean(profits) if profits else 0.0,
        'trades': sum(s.get('trades_count', 0) for s in complex_strategies),
        'win_rate': np.mean(win_rates) if win_rates else 0.0
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 테스트
    strategy = SimplifiedStrategy()
    print(f"전략 파라미터: {strategy.get_strategy_params()}")
    
    # 예시 데이터
    test_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99
    })
    
    result = backtest_simplified_strategy(strategy, test_data)
    print(f"\n백테스트 결과:")
    print(f"  수익률: {result['profit']:.2f}%")
    print(f"  거래 수: {result['trades']}")
    print(f"  승률: {result['win_rate']:.2%}")

