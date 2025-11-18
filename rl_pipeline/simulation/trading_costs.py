"""
거래 비용 모델
- 수수료, 슬리피지 계산
- 실전 거래 조건 반영
"""

from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class TradingCosts:
    """거래 비용 모델"""
    
    # 거래소 수수료
    maker_fee: float = 0.001  # 0.1% (지정가)
    taker_fee: float = 0.001  # 0.1% (시장가)
    
    # 슬리피지 (시장 충격)
    slippage_rate: float = 0.0005  # 0.05%
    
    # 최소 거래 수수료 (USDT)
    min_fee: float = 0.1
    
    def __post_init__(self):
        """환경변수에서 설정 읽기"""
        self.maker_fee = float(os.getenv("MAKER_FEE", str(self.maker_fee)))
        self.taker_fee = float(os.getenv("TAKER_FEE", str(self.taker_fee)))
        self.slippage_rate = float(os.getenv("SLIPPAGE_RATE", str(self.slippage_rate)))
        self.min_fee = float(os.getenv("MIN_FEE_USDT", str(self.min_fee)))
    
    def calculate_entry_cost(
        self,
        position_size: float,
        price: float,
        order_type: str = "TAKER"
    ) -> float:
        """진입 비용 계산"""
        fee_rate = self.taker_fee if order_type == "TAKER" else self.maker_fee
        
        # 수수료
        commission = position_size * price * fee_rate
        commission = max(commission, self.min_fee)
        
        # 슬리피지
        slippage = position_size * price * self.slippage_rate
        
        return commission + slippage
    
    def calculate_exit_cost(
        self,
        position_size: float,
        price: float,
        order_type: str = "TAKER"
    ) -> float:
        """청산 비용 계산 (진입과 동일)"""
        return self.calculate_entry_cost(position_size, price, order_type)
    
    def calculate_total_cost(
        self,
        position_size: float,
        entry_price: float,
        exit_price: float,
        entry_order_type: str = "TAKER",
        exit_order_type: str = "TAKER"
    ) -> float:
        """총 거래 비용 계산"""
        entry_cost = self.calculate_entry_cost(
            position_size, entry_price, entry_order_type
        )
        exit_cost = self.calculate_exit_cost(
            position_size, exit_price, exit_order_type
        )
        return entry_cost + exit_cost
    
    def adjust_profit(
        self,
        gross_profit: float,
        position_size: float,
        entry_price: float,
        exit_price: float
    ) -> float:
        """순수익 계산 (비용 차감)"""
        total_cost = self.calculate_total_cost(
            position_size, entry_price, exit_price
        )
        net_profit = gross_profit - total_cost
        return net_profit
    
    def get_cost_percentage(
        self,
        position_size: float,
        entry_price: float,
        exit_price: float
    ) -> float:
        """거래 비용 비율 계산 (%)"""
        total_cost = self.calculate_total_cost(
            position_size, entry_price, exit_price
        )
        position_value = position_size * entry_price
        return (total_cost / position_value) * 100 if position_value > 0 else 0.0


# 사용 예시
if __name__ == "__main__":
    costs = TradingCosts()
    
    # 예시 거래
    position_size = 1.0  # 1 BTC
    entry_price = 50000
    exit_price = 51000
    
    entry_cost = costs.calculate_entry_cost(position_size, entry_price)
    exit_cost = costs.calculate_exit_cost(position_size, exit_price)
    total_cost = costs.calculate_total_cost(position_size, entry_price, exit_price)
    cost_pct = costs.get_cost_percentage(position_size, entry_price, exit_price)
    
    print(f"진입 비용: ${entry_cost:.2f}")
    print(f"청산 비용: ${exit_cost:.2f}")
    print(f"총 비용: ${total_cost:.2f} ({cost_pct:.3f}%)")
    
    # 순수익 계산
    gross_profit = (exit_price - entry_price) * position_size
    net_profit = costs.adjust_profit(
        gross_profit, position_size, entry_price, exit_price
    )
    
    print(f"\n총 수익: ${gross_profit:.2f}")
    print(f"순수익: ${net_profit:.2f}")

