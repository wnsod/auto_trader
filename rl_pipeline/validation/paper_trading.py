"""
Paper Trading 시스템
- 실전 환경 시뮬레이션
- 실전 투입 가능 여부 판단
"""

from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import logging
import sqlite3
import os
import time

logger = logging.getLogger(__name__)

# DB 경로 (환경변수 또는 기본값)
CANDLES_DB_PATH = os.getenv('CANDLES_DB_PATH', 'data_storage/realtime_candles.db')
TRADING_SYSTEM_DB_PATH = os.getenv('TRADING_SYSTEM_DB_PATH', 'data_storage/trading_system.db')


class PaperTradingSystem:
    """Paper Trading 시스템"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.real_time_data = None
        self.start_time = datetime.now()
    
    def execute_paper_trade(
        self,
        signal: str,  # "BUY", "SELL", "HOLD"
        coin: str,
        price: float,
        size: Optional[float] = None
    ) -> bool:
        """Paper Trading 거래 실행"""
        
        try:
            if signal == "BUY":
                # 포지션 크기 계산 (지정되지 않으면 자본의 10%)
                if size is None:
                    size = (self.capital * 0.1) / price
                
                cost = price * size
                if cost <= self.capital:
                    self.capital -= cost
                    self.positions[coin] = {
                        'size': size,
                        'entry_price': price,
                        'entry_time': datetime.now()
                    }
                    self.trades.append({
                        'type': 'BUY',
                        'coin': coin,
                        'price': price,
                        'size': size,
                        'time': datetime.now()
                    })
                    logger.debug(f"Paper Trading BUY: {coin} {size:.4f} @ ${price:.2f}")
                    return True
                else:
                    logger.warning(f"자본 부족: 필요 ${cost:.2f}, 보유 ${self.capital:.2f}")
                    return False
            
            elif signal == "SELL" and coin in self.positions:
                position = self.positions[coin]
                revenue = price * position['size']
                self.capital += revenue
                profit = (price - position['entry_price']) * position['size']
                
                self.trades.append({
                    'type': 'SELL',
                    'coin': coin,
                    'price': price,
                    'size': position['size'],
                    'profit': profit,
                    'return_pct': (profit / (position['entry_price'] * position['size'])) * 100,
                    'time': datetime.now(),
                    'holding_period': (datetime.now() - position['entry_time']).total_seconds() / 3600
                })
                
                del self.positions[coin]
                logger.debug(f"Paper Trading SELL: {coin} {position['size']:.4f} @ ${price:.2f}, "
                           f"수익: ${profit:.2f} ({self.trades[-1]['return_pct']:.2f}%)")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Paper Trading 거래 실행 실패: {e}")
            return False
    
    def get_performance(self) -> Dict:
        """성과 분석"""
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        completed_trades = [t for t in self.trades if t['type'] == 'SELL']
        winning_trades = [t for t in completed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('profit', 0) <= 0]
        
        # 미청산 포지션 가치 계산
        open_positions_value = 0.0
        for coin, position in self.positions.items():
            # 현재 가격은 마지막 거래 가격으로 가정 (실제로는 실시간 가격 필요)
            open_positions_value += position['entry_price'] * position['size']
        
        total_equity = self.capital + open_positions_value
        
        return {
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'win_rate': len(winning_trades) / len(completed_trades) if completed_trades else 0.0,
            'avg_profit': np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0.0,
            'current_capital': self.capital,
            'total_equity': total_equity,
            'open_positions': len(self.positions),
            'trading_days': (datetime.now() - self.start_time).days
        }
    
    def get_detailed_statistics(self) -> Dict:
        """상세 통계"""
        
        performance = self.get_performance()
        completed_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        if not completed_trades:
            return performance
        
        profits = [t.get('profit', 0) for t in completed_trades]
        returns = [t.get('return_pct', 0) for t in completed_trades]
        holding_periods = [t.get('holding_period', 0) for t in completed_trades if t.get('holding_period')]
        
        # 손익비 계산
        avg_profit = performance['avg_profit']
        avg_loss = abs(performance['avg_loss']) if performance['avg_loss'] < 0 else 0.0
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0.0
        
        # 최대 연속 손실/수익
        max_consecutive_losses = 0
        max_consecutive_wins = 0
        current_losses = 0
        current_wins = 0
        
        for trade in completed_trades:
            if trade.get('profit', 0) > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return {
            **performance,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'max_consecutive_wins': max_consecutive_wins,
            'avg_holding_hours': np.mean(holding_periods) if holding_periods else 0.0,
            'total_profit': sum(profits),
            'best_trade': max(profits) if profits else 0.0,
            'worst_trade': min(profits) if profits else 0.0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns) if returns else 0.0
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """샤프 비율 계산"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # 연율화 (거래당 평균 수익률로 가정)
        if std_return > 0:
            sharpe = mean_return / std_return
            return sharpe
        
        return 0.0
    
    def reset(self):
        """시스템 리셋"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.start_time = datetime.now()
    
    def get_realtime_price(self, coin: str) -> Optional[float]:
        """실시간 가격 조회"""
        try:
            # DB에서 최신 캔들 가격 조회
            intervals = ['15m', '30m', '240m', '1d']
            
            for db_path in [CANDLES_DB_PATH, TRADING_SYSTEM_DB_PATH]:
                try:
                    with sqlite3.connect(db_path) as conn:
                        for interval in intervals:
                            query = """
                                SELECT close FROM candles 
                                WHERE coin = ? AND interval = ? 
                                ORDER BY timestamp DESC LIMIT 1
                            """
                            result = conn.execute(query, (coin, interval)).fetchone()
                            
                            if result and result[0] and result[0] > 0:
                                return float(result[0])
                except Exception:
                    continue
            
            # 폴백: trade_manager 사용
            try:
                from trade.trade_manager import get_latest_price
                price = get_latest_price(coin)
                if price and price > 0:
                    return float(price)
            except Exception:
                pass
            
            logger.warning(f"⚠️ {coin} 실시간 가격 조회 실패")
            return None
        
        except Exception as e:
            logger.error(f"❌ 실시간 가격 조회 오류 ({coin}): {e}")
            return None
    
    def update_open_positions_value(self) -> float:
        """미청산 포지션 가치 업데이트 (실시간 가격 사용)"""
        total_value = 0.0
        for coin, position in self.positions.items():
            current_price = self.get_realtime_price(coin)
            if current_price:
                total_value += current_price * position['size']
            else:
                # 가격 조회 실패 시 진입가로 대체
                total_value += position['entry_price'] * position['size']
        return total_value


def validate_for_live_trading(performance: Dict) -> bool:
    """실전 투입 가능 여부 판단"""
    
    criteria = {
        'min_return': 5.0,      # 최소 5% 수익
        'min_win_rate': 0.55,    # 최소 55% 승률
        'min_trades': 20,        # 최소 20회 거래
        'min_trading_days': 30   # 최소 30일 거래
    }
    
    passed = (
        performance.get('total_return', 0) >= criteria['min_return'] and
        performance.get('win_rate', 0) >= criteria['min_win_rate'] and
        performance.get('total_trades', 0) >= criteria['min_trades'] and
        performance.get('trading_days', 0) >= criteria['min_trading_days']
    )
    
    if passed:
        logger.info("✅ 실전 투입 가능")
        logger.info(f"  수익률: {performance.get('total_return', 0):.2f}%")
        logger.info(f"  승률: {performance.get('win_rate', 0):.2%}")
        logger.info(f"  거래 수: {performance.get('total_trades', 0)}")
    else:
        logger.warning("❌ 실전 투입 불가 (기준 미달)")
        logger.warning(f"  필요 수익률: {criteria['min_return']}% (현재: {performance.get('total_return', 0):.2f}%)")
        logger.warning(f"  필요 승률: {criteria['min_win_rate']:.0%} (현재: {performance.get('win_rate', 0):.2%})")
        logger.warning(f"  필요 거래 수: {criteria['min_trades']} (현재: {performance.get('total_trades', 0)})")
        logger.warning(f"  필요 거래 일수: {criteria['min_trading_days']}일 (현재: {performance.get('trading_days', 0)}일)")
    
    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 테스트
    paper_trader = PaperTradingSystem(initial_capital=100000)
    
    # 모의 거래
    paper_trader.execute_paper_trade("BUY", "BTC", 50000, size=0.1)
    paper_trader.execute_paper_trade("SELL", "BTC", 51000)
    
    performance = paper_trader.get_performance()
    print(f"\n성과:")
    print(f"  수익률: {performance['total_return']:.2f}%")
    print(f"  거래 수: {performance['total_trades']}")
    print(f"  승률: {performance['win_rate']:.2%}")
    
    validate_for_live_trading(performance)

