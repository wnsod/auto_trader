"""
리스크 관리 엔진 (Core Risk)
- 가상/실전 매매에서 공통으로 사용하는 리스크 관리 로직
- 트레일링 스탑, 포지션 리스크 평가, 이상치 필터링 등
"""
import time
from typing import List, Optional, Dict
from trade.signal_selector.core.types import SignalInfo

class OutlierGuardrail:
    """이상치 컷 시스템"""
    def __init__(self, percentile_cut: float = 0.05):
        self.percentile_cut = percentile_cut
    
    def winsorize_profits(self, profits: List[float]) -> List[float]:
        """수익률 Winsorizing"""
        if len(profits) < 10:  # 데이터가 적으면 그대로 반환
            return profits
        
        sorted_profits = sorted(profits)
        n = len(sorted_profits)
        
        # 상하위 5% 절단
        lower_cut = int(n * self.percentile_cut)
        upper_cut = int(n * (1 - self.percentile_cut))
        
        # 절단된 값으로 대체
        winsorized = []
        for profit in profits:
            if profit < sorted_profits[lower_cut]:
                winsorized.append(sorted_profits[lower_cut])
            elif profit > sorted_profits[upper_cut]:
                winsorized.append(sorted_profits[upper_cut])
            else:
                winsorized.append(profit)
        
        return winsorized
    
    def calculate_robust_avg_profit(self, profits: List[float]) -> float:
        """견고한 평균 수익률 계산"""
        winsorized_profits = self.winsorize_profits(profits)
        if not winsorized_profits:
            return 0.0
        return sum(winsorized_profits) / len(winsorized_profits)

class RiskManager:
    """리스크 관리자 - 포지션 리스크 관리 및 트레일링 스탑"""
    def __init__(self):
        # 트레일링 스탑 상태 추적 {coin: max_profit_pct}
        self.trailing_stop_state = {}
        
    def update_trailing_stop_state(self, coin: str, current_profit_pct: float):
        """트레일링 스탑을 위한 최고 수익률 업데이트"""
        if coin not in self.trailing_stop_state:
            self.trailing_stop_state[coin] = current_profit_pct
        else:
            if current_profit_pct > self.trailing_stop_state[coin]:
                self.trailing_stop_state[coin] = current_profit_pct
                
    def check_trailing_stop(self, coin: str, current_profit_pct: float) -> Optional[str]:
        """트레일링 스탑 조건 확인"""
        max_profit = self.trailing_stop_state.get(coin, current_profit_pct)
        
        # 1. 수익 20% 이상 도달 후, 고점 대비 5% 하락 시 익절
        if max_profit >= 20.0 and current_profit_pct <= (max_profit - 5.0):
            return f"trailing_stop (max: {max_profit:.1f}%, current: {current_profit_pct:.1f}%)"
            
        # 2. 수익 10% 이상 도달 후, 고점 대비 3% 하락 시 익절
        if max_profit >= 10.0 and current_profit_pct <= (max_profit - 3.0):
            return f"trailing_stop (max: {max_profit:.1f}%, current: {current_profit_pct:.1f}%)"
            
        # 3. 수익 5% 이상 도달 후, 본전(0.5% 이하) 위협 시 익절
        if max_profit >= 5.0 and current_profit_pct <= 0.5:
            return f"profit_protect (max: {max_profit:.1f}%, current: {current_profit_pct:.1f}%)"
            
        return None

    def calculate_position_risk(self, entry_price: float, current_price: float, max_loss_pct: float) -> float:
        """포지션 리스크 계산"""
        try:
            if entry_price == 0:
                return 0.5
                
            # 현재 손익
            current_pnl = (current_price - entry_price) / entry_price
            
            # 최대 손실 (절댓값)
            max_loss = abs(max_loss_pct) / 100
            
            # 리스크 점수 (0-1, 높을수록 위험)
            # 10% 손실을 최대 위험(1.0)으로 설정
            risk_score = min(abs(current_pnl) / 0.1, 1.0) if current_pnl < 0 else 0.0
            
            return risk_score
            
        except Exception as e:
            print(f"⚠️ 포지션 리스크 계산 오류: {e}")
            return 0.5
    
    def should_close_position(self, coin: str, profit_loss_pct: float, 
                            stop_loss_pct: float, take_profit_pct: float,
                            entry_price: float, current_price: float) -> Optional[str]:
        """포지션 종료 여부 판단 (손절/익절/리스크)"""
        try:
            # 1. 손절 조건 확인
            if profit_loss_pct <= -stop_loss_pct:
                return "stop_loss"
            
            # 2. 익절 조건 확인
            if profit_loss_pct >= take_profit_pct:
                return "take_profit"
            
            # 3. 리스크 기반 종료 (위험도 80% 초과 시)
            risk_score = self.calculate_position_risk(entry_price, current_price, stop_loss_pct)
            if risk_score > 0.8:
                return "risk_cutoff"
            
            # 4. 트레일링 스탑 확인
            self.update_trailing_stop_state(coin, profit_loss_pct)
            trailing_reason = self.check_trailing_stop(coin, profit_loss_pct)
            if trailing_reason:
                return "trailing_stop"
            
            return None
            
        except Exception as e:
            print(f"⚠️ 포지션 종료 판단 오류: {e}")
            return None

    def check_correlation_risk(self, coin: str, current_holdings: List[str], threshold: float = 0.8) -> Dict:
        """포트폴리오 상관관계 리스크 확인
        
        🆕 단순화: 변동성 기반 분산 체크는 SignalSelector에서 이미 수행 중
        (signal_selector/analysis/market.py의 get_coin_volatility_group)
        여기서는 중복 계산하지 않고, 기본 체크만 수행
        
        Args:
            coin: 신규 매수 코인
            current_holdings: 현재 보유 중인 코인 목록
            threshold: 상관관계 임계값 (기본 0.8)
            
        Returns:
            Dict: {'safe': bool, 'reason': str, 'max_correlation': float}
        """
        # 같은 코인이 이미 보유 중이면 위험
        if coin in current_holdings:
            return {'safe': False, 'reason': 'already_holding', 'max_correlation': 1.0}
        
        # 🆕 변동성/상관관계 기반 분산 체크는 SignalSelector에서 수행
        # → 여기서 중복 계산하지 않음 (signal_selector.analysis.market 참조)
        return {'safe': True, 'reason': 'ok', 'max_correlation': 0.0}
    
    def calculate_adaptive_stop_loss_strength(self, coin: str, signal: SignalInfo, market_volatility: float, performance_score: float) -> float:
        """학습 기반 동적 손절 강도 계산"""
        try:
            # 기본 손절 강도 (50%)
            base_strength = 50.0
            
            # 성과 기반 조정
            if performance_score > 0.7:
                base_strength += 20.0
            elif performance_score < 0.3:
                base_strength -= 15.0
            
            # 시그널 강도 기반 조정
            signal_strength = abs(signal.signal_score)
            if signal_strength > 0.5:
                base_strength += 15.0
            elif signal_strength < 0.2:
                base_strength -= 10.0
            
            # 변동성 기반 조정
            if market_volatility > 0.05:
                base_strength += 10.0
            elif market_volatility < 0.02:
                base_strength -= 5.0
            
            return max(30.0, min(80.0, base_strength))
            
        except Exception:
            return 50.0

