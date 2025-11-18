"""
전략 에이전트 모듈
- StrategyAgent: 전략 기반 에이전트
"""

import os
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from rl_pipeline.simulation.market_models import Action, MarketState, AgentState, MarketDataGenerator
from rl_pipeline.simulation.trading_costs import TradingCosts

logger = logging.getLogger(__name__)

# 환경변수
ENABLE_DRAWDOWN_CONTROL = os.getenv('ENABLE_DRAWDOWN_CONTROL', 'true').lower() == 'true'
USE_REALISTIC_COSTS = os.getenv('USE_REALISTIC_COSTS', 'true').lower() == 'true'


class StrategyAgent:
    """전략 에이전트"""
    
    def __init__(self, agent_id: str, strategy_params: Dict[str, Any], trading_costs: Optional[TradingCosts] = None):
        self.agent_id = agent_id
        self.strategy_params = strategy_params
        self.state = AgentState(
            balance=10000.0,
            position=None,
            trades=[],
            equity_curve=[10000.0],
            strategy_params=strategy_params
        )
        self.performance_history = []

        # 🔥 진입 조건 필터링을 위한 추적 변수
        self.last_trade_time = None  # 마지막 거래 시간 (쿨다운용)
        self.trade_count_in_window = 0  # 일정 기간 내 거래 횟수 (빈도 제한용)
        self.trade_window_start = None  # 거래 윈도우 시작 시간

        # 거래 비용 모델
        if USE_REALISTIC_COSTS:
            self.trading_costs = trading_costs if trading_costs else TradingCosts()
        else:
            self.trading_costs = None
        
        # Phase 5: RiskController 재사용 (매번 생성 방지)
        if ENABLE_DRAWDOWN_CONTROL:
            from rl_pipeline.simulation.risk_controller import RiskController
            self.risk_controller = RiskController()
        else:
            self.risk_controller = None
        
    def decide_action(self, market_state: MarketState) -> Action:
        """🚀 실제 전략 로직 기반 행동 결정"""
        try:
            # 실제 전략에서 사용하는 모든 지표들 활용
            strategy_params = self.strategy_params
            
            # 1. 기본 지표값들
            rsi = market_state.rsi
            macd = market_state.macd
            macd_signal = market_state.macd_signal
            mfi = getattr(market_state, 'mfi', 50.0)  # MFI (기본값 50)
            atr = getattr(market_state, 'atr', 0.02)   # ATR (기본값 2%)
            adx = getattr(market_state, 'adx', 25.0)  # ADX (기본값 25)
            volume_ratio = market_state.volume_ratio
            
            # 2. 전략 파라미터 추출 (실제 전략의 모든 파라미터 포함)
            rsi_min = strategy_params.get('rsi_min', 30)
            rsi_max = strategy_params.get('rsi_max', 70)
            volume_ratio_min = strategy_params.get('volume_ratio_min', 1.0)
            volume_ratio_max = strategy_params.get('volume_ratio_max', 3.0)
            macd_buy_threshold = strategy_params.get('macd_buy_threshold', 0.01)
            macd_sell_threshold = strategy_params.get('macd_sell_threshold', -0.01)
            mfi_min = strategy_params.get('mfi_min', 20)
            mfi_max = strategy_params.get('mfi_max', 80)
            atr_min = strategy_params.get('atr_min', 0.01)
            atr_max = strategy_params.get('atr_max', 0.05)
            adx_min = strategy_params.get('adx_min', 20)
            
            # 추가 전략 파라미터들
            # 🔥 ATR 기반 동적 Stop-Loss/Take-Profit
            stop_loss_atr_multiplier = strategy_params.get('stop_loss_atr_multiplier', 1.5)
            take_profit_atr_multiplier = strategy_params.get('take_profit_atr_multiplier', 3.0)

            # ATR 기반 동적 계산 (ATR이 높을수록 손절/익절 폭 증가)
            dynamic_stop_loss_pct = atr * stop_loss_atr_multiplier
            dynamic_take_profit_pct = atr * take_profit_atr_multiplier

            # 최소/최대 제한 (너무 좁거나 넓지 않게)
            stop_loss_pct = max(0.01, min(0.05, dynamic_stop_loss_pct))
            take_profit_pct = max(0.02, min(0.10, dynamic_take_profit_pct))

            position_size = strategy_params.get('position_size', 0.01)
            bb_period = strategy_params.get('bb_period', 20)
            bb_std = strategy_params.get('bb_std', 2.0)
            ma_period = strategy_params.get('ma_period', 20)
            
            # 3. 레짐 기반 전략 조정
            regime_stage = market_state.regime_stage
            regime_confidence = market_state.regime_confidence
            
            # 레짐별 파라미터 동적 조정
            if regime_stage >= 6:  # bullish, extreme_bullish
                rsi_min = max(15, rsi_min - 8)  # 더 공격적 매수
                rsi_max = min(85, rsi_max + 8)  # 더 관대한 매도
                volume_ratio_min = max(0.8, volume_ratio_min - 0.2)
                mfi_min = max(10, mfi_min - 10)
            elif regime_stage <= 2:  # bearish, extreme_bearish
                rsi_min = min(45, rsi_min + 8)  # 더 보수적 매수
                rsi_max = max(55, rsi_max - 8)  # 더 빠른 매도
                volume_ratio_min = min(1.5, volume_ratio_min + 0.3)
                mfi_min = min(30, mfi_min + 10)
            
            # 4. 복합 매수 조건 (실제 전략 로직)
            buy_conditions = []
            
            # RSI 조건 - 하락 돌파 시 매수 신호
            buy_conditions.append(rsi < rsi_max)  # rsi_max 이하일 때 매수 가능
            
            # MACD 조건 - 매수 신호 단순화
            buy_conditions.append(macd > macd_signal)
            
            # MFI 조건 (자금흐름) - 조건 완화
            buy_conditions.append(mfi < 70)
            
            # 거래량 조건 - 단순화
            buy_conditions.append(volume_ratio > 0.5)
            
            # 볼린저 밴드 조건 - 완화 (하단보다는 중간 이하)
            buy_conditions.append(market_state.price < market_state.bb_middle)
            
            # 레짐 조건 - 제외 (너무 제한적)
            # buy_conditions.append(regime_stage >= 4)  # 중립 이상에서만 매수
            
            # 5. 복합 매도 조건
            sell_conditions = []
            
            # RSI 조건
            sell_conditions.append(rsi > rsi_max)
            
            # MACD 조건
            sell_conditions.append(macd < macd_sell_threshold)
            sell_conditions.append(macd < macd_signal)  # MACD가 시그널 아래
            
            # MFI 조건
            sell_conditions.append(mfi > mfi_max)
            
            # 거래량 조건
            sell_conditions.append(volume_ratio > volume_ratio_min)
            
            # 볼린저 밴드 조건
            sell_conditions.append(market_state.price > market_state.bb_upper)
            
            # 6. 현재 포지션 확인 및 행동 결정
            if self.state.position is None:
                # 🔥 7개 진입 조건 필터 추가
                entry_filters_passed = []

                # Filter 1: Trend Strength (추세 강도) - ADX > 20
                trend_strength_min = strategy_params.get('trend_strength_min', 20)
                entry_filters_passed.append(adx >= trend_strength_min)

                # Filter 2: Volatility (변동성) - ATR가 적정 범위 내
                volatility_ok = atr_min <= atr <= atr_max
                entry_filters_passed.append(volatility_ok)

                # Filter 3: Volume (거래량) - 평균 이상
                volume_ok = volume_ratio >= volume_ratio_min
                entry_filters_passed.append(volume_ok)

                # Filter 4: Confirmation (확인) - 여러 지표가 동시에 신호
                buy_score = sum(buy_conditions)
                confirmation_threshold = strategy_params.get('confirmation_threshold', 3)
                confirmation_ok = buy_score >= confirmation_threshold
                entry_filters_passed.append(confirmation_ok)

                # Filter 5: Cooldown (쿨다운) - 마지막 거래 후 일정 시간 경과
                cooldown_minutes = strategy_params.get('cooldown_minutes', 60)
                if self.last_trade_time is None:
                    cooldown_ok = True
                else:
                    time_since_last_trade = (market_state.timestamp - self.last_trade_time).total_seconds() / 60
                    cooldown_ok = time_since_last_trade >= cooldown_minutes
                entry_filters_passed.append(cooldown_ok)

                # Filter 6: Frequency (빈도) - 일정 기간 내 거래 횟수 제한
                max_trades_per_day = strategy_params.get('max_trades_per_day', 10)
                if self.trade_window_start is None or \
                   (market_state.timestamp - self.trade_window_start).total_seconds() > 86400:  # 24시간
                    self.trade_window_start = market_state.timestamp
                    self.trade_count_in_window = 0
                frequency_ok = self.trade_count_in_window < max_trades_per_day
                entry_filters_passed.append(frequency_ok)

                # Filter 7: Signal Threshold (시그널 임계값) - buy_score가 임계값 이상
                signal_threshold = strategy_params.get('signal_threshold', 0.5)
                signal_strength = buy_score / len(buy_conditions) if len(buy_conditions) > 0 else 0
                signal_ok = signal_strength >= signal_threshold
                entry_filters_passed.append(signal_ok)

                # 🔥 진입 조건 완화: 7개 필터 중 4개 이상만 통과하면 OK (학습 데이터 증가)
                filters_passed_count = sum(entry_filters_passed)
                min_filters_required = strategy_params.get('min_filters_required', 4)  # 기본 4개
                enough_filters_passed = filters_passed_count >= min_filters_required

                # 매수 조건 확인
                required_buy_conditions = max(2, int(len(buy_conditions) * 0.2))

                # 🔥 진입 조건 완화: 4개 이상 필터 통과 + 기존 조건 만족 시 매수
                if buy_score >= required_buy_conditions and regime_confidence > 0.2 and enough_filters_passed:
                    # 거래 추적 정보 업데이트
                    self.last_trade_time = market_state.timestamp
                    self.trade_count_in_window += 1
                    return Action.BUY
            else:
                # 포지션이 있는 경우 - 스탑로스/테이크프로핏 확인
                position = self.state.position
                entry_price = position["entry_price"]
                current_price = market_state.price
                
                # 수익률 계산
                profit_pct = (current_price - entry_price) / entry_price
                
                # 스탑로스 확인
                if profit_pct <= -stop_loss_pct:
                    logger.debug(f"🛑 {self.agent_id} 스탑로스 실행: {profit_pct:.2%}")
                    return Action.SELL
                
                # 테이크프로핏 확인
                if profit_pct >= take_profit_pct:
                    logger.debug(f"💰 {self.agent_id} 테이크프로핏 실행: {profit_pct:.2%}")
                    return Action.SELL
                
                # 일반 매도 조건 확인 - 조건 완화
                required_sell_conditions = max(2, int(len(sell_conditions) * 0.3))  # 30% 이상 조건 충족 (완화)
                sell_score = sum(sell_conditions)
                
                # 🔥 조건 완화: regime_confidence 임계값을 0.4 → 0.2로 낮춤
                if sell_score >= required_sell_conditions and regime_confidence > 0.2:
                    return Action.SELL
            
            return Action.HOLD
            
        except Exception as e:
            logger.error(f"❌ 에이전트 {self.agent_id} 행동 결정 실패: {e}")
            return Action.HOLD
    
    def execute_action(self, action: Action, market_state: MarketState) -> Dict[str, Any]:
        """행동 실행"""
        try:
            trade_result = {"action": action.value, "timestamp": market_state.timestamp, "price": market_state.price}
            
            if action == Action.BUY and self.state.position is None:
                # 매수 실행
                base_position_size = self.state.balance * 0.95  # 잔고의 95% 사용
                
                # Phase 5: Drawdown 기반 포지션 크기 조정
                if ENABLE_DRAWDOWN_CONTROL and self.risk_controller and len(self.state.equity_curve) > 1:
                    max_drawdown = self.risk_controller.calculate_drawdown(self.state.equity_curve)
                    position_size = self.risk_controller.get_adjusted_position_size(
                        base_position_size,
                        max_drawdown
                    )
                else:
                    position_size = base_position_size
                
                # 거래 비용 계산
                if self.trading_costs:
                    # TradingCosts 모델 사용
                    quantity = position_size / market_state.price
                    entry_cost = self.trading_costs.calculate_entry_cost(
                        quantity, market_state.price, "TAKER"
                    )
                    net_position_size = position_size - entry_cost
                else:
                    # 기존 방식 (0.1% 수수료)
                    trading_fee = position_size * 0.001
                    net_position_size = position_size - trading_fee
                    entry_cost = trading_fee
                
                self.state.position = {
                    "position_type": "LONG",
                    "entry_price": market_state.price,
                    "entry_time": market_state.timestamp,
                    "size": net_position_size,  # 수수료 제외한 실제 투자 금액
                    "quantity": net_position_size / market_state.price,
                    "entry_cost": entry_cost
                }
                self.state.balance -= position_size  # 수수료 포함한 전체 금액 차감
                trade_result.update({
                    "type": "BUY",
                    "quantity": self.state.position["quantity"],
                    "value": position_size
                })
                
            elif action == Action.SELL and self.state.position is not None:
                # 매도 실행
                position = self.state.position
                exit_value = position["quantity"] * market_state.price
                
                # 거래 비용 계산
                if self.trading_costs:
                    # TradingCosts 모델 사용
                    exit_cost = self.trading_costs.calculate_exit_cost(
                        position["quantity"], market_state.price, "TAKER"
                    )
                    net_exit_value = exit_value - exit_cost
                    total_cost = position.get("entry_cost", 0) + exit_cost
                else:
                    # 기존 방식 (0.1% 수수료)
                    trading_fee = exit_value * 0.001
                    net_exit_value = exit_value - trading_fee
                    total_cost = trading_fee + position.get("entry_cost", trading_fee)
                
                pnl = net_exit_value - position["size"]
                
                self.state.balance += net_exit_value
                
                # 거래 기록
                trade_record = {
                    "entry_price": position["entry_price"],
                    "exit_price": market_state.price,
                    "entry_time": position["entry_time"],
                    "exit_time": market_state.timestamp,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "pnl_pct": pnl / position["size"] * 100,
                    "duration_minutes": (market_state.timestamp - position["entry_time"]).total_seconds() / 60
                }
                
                self.state.trades.append(trade_record)
                self.state.position = None
                
                trade_result.update({
                    "type": "SELL",
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "pnl_pct": trade_record["pnl_pct"]
                })
            
            # 자산 가치 업데이트
            current_value = self.state.balance
            if self.state.position is not None:
                current_value += self.state.position["quantity"] * market_state.price
            
            self.state.equity_curve.append(current_value)
            
            return trade_result
            
        except Exception as e:
            logger.error(f"❌ 에이전트 {self.agent_id} 행동 실행 실패: {e}")
            return {"action": action.value, "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성과 지표 계산"""
        try:
            if not self.state.trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl_per_trade": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0
                }
            
            trades = self.state.trades
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t["pnl"] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_pnl = sum(t["pnl"] for t in trades)
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
            
            # 최대 낙폭 계산
            equity_curve = self.state.equity_curve
            peak = equity_curve[0]
            max_drawdown = 0.0
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # 샤프 비율 계산 (간단화)
            if len(equity_curve) > 1:
                returns = np.diff(equity_curve) / equity_curve[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": avg_pnl_per_trade,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "final_balance": self.state.balance,
                "current_value": equity_curve[-1] if equity_curve else 10000.0
            }
            
        except Exception as e:
            logger.error(f"❌ 성과 지표 계산 실패: {e}")
            return {}

