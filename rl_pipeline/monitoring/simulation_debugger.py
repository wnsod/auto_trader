"""
Simulation 모듈 전용 디버그 로거
- Self-play 에피소드 추적
- 거래 행동 상세 분석
- 보상 분포 추적
- 에이전트 성능 비교
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .debug_logger import DebugLogger


class SimulationDebugger(DebugLogger):
    """Simulation 모듈 전용 디버거"""

    def __init__(self, session_id: str = None):
        super().__init__("simulation", session_id)

        # 시뮬레이션 통계
        self.simulation_stats = {
            "total_episodes": 0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "no_trade_episodes": 0
        }

    def log_selfplay_start(
        self,
        coin: str,
        interval: str,
        num_episodes: int,
        num_agents: int,
        candle_count: int,
        config: Dict[str, Any] = None
    ):
        """
        Self-play 시작 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            num_episodes: 에피소드 수
            num_agents: 에이전트 수
            candle_count: 캔들 데이터 개수
            config: 시뮬레이션 설정
        """
        self.log({
            "event": "selfplay_start",
            "coin": coin,
            "interval": interval,
            "num_episodes": num_episodes,
            "num_agents": num_agents,
            "candle_count": candle_count,
            "config": config or {},
            "message": f"🚀 Self-play 시작: {coin}-{interval}"
        })

    def log_episode_start(
        self,
        episode: int,
        total_episodes: int,
        regime: str,
        initial_balance: float,
        strategies: List[Dict[str, Any]]
    ):
        """
        에피소드 시작 로깅

        Args:
            episode: 에피소드 번호
            total_episodes: 전체 에피소드 수
            regime: 현재 레짐
            initial_balance: 초기 잔액
            strategies: 사용할 전략 리스트
        """
        self.simulation_stats["total_episodes"] += 1

        self.log({
            "event": "episode_start",
            "episode": episode,
            "total_episodes": total_episodes,
            "progress": f"{episode}/{total_episodes}",
            "regime": regime,
            "initial_balance": float(initial_balance),
            "num_strategies": len(strategies),
            "strategy_grades": [s.get("grade", "UNKNOWN") for s in strategies]
        })

    def log_agent_step(
        self,
        episode: int,
        step: int,
        agent_id: str,
        state: np.ndarray,
        action: int,
        action_probs: np.ndarray,
        reward: float,
        balance: float,
        position: Optional[Dict[str, Any]] = None,
        market_info: Optional[Dict[str, Any]] = None
    ):
        """
        에이전트 스텝 상세 로깅

        Args:
            episode: 에피소드 번호
            step: 스텝 번호
            agent_id: 에이전트 ID
            state: 상태 벡터
            action: 선택한 액션 (0=HOLD, 1=BUY, 2=SELL)
            action_probs: 액션 확률 분포
            reward: 받은 보상
            balance: 현재 잔액
            position: 포지션 정보 (옵션)
            market_info: 시장 정보 (옵션)
        """
        action_names = ["HOLD", "BUY", "SELL"]

        log_entry = {
            "event": "agent_step",
            "episode": episode,
            "step": step,
            "agent_id": agent_id,

            # 액션 정보
            "action": {
                "index": int(action),
                "name": action_names[action] if action < len(action_names) else f"action_{action}",
                "probabilities": {
                    action_names[i]: float(action_probs[i])
                    for i in range(min(len(action_names), len(action_probs)))
                },
                "confidence": float(action_probs[action]) if action < len(action_probs) else 0.0
            },

            # 상태 정보
            "state": {
                "dim": len(state),
                "values": state.tolist() if len(state) <= 20 else None,  # 20차원 이하만 전체 출력
                "mean": float(np.mean(state)),
                "std": float(np.std(state)),
                "min": float(np.min(state)),
                "max": float(np.max(state))
            },

            # 보상 & 잔액
            "reward": float(reward),
            "balance": float(balance)
        }

        # 포지션 정보
        if position:
            log_entry["position"] = {
                "has_position": position.get("has_position", False),
                "entry_price": float(position.get("entry_price", 0)),
                "quantity": float(position.get("quantity", 0)),
                "pnl": float(position.get("pnl", 0))
            }

        # 시장 정보
        if market_info:
            log_entry["market"] = {
                "price": float(market_info.get("price", 0)),
                "volume": float(market_info.get("volume", 0)),
                "rsi": float(market_info.get("rsi", 50)),
                "macd": float(market_info.get("macd", 0))
            }

        self.log(log_entry, level="DEBUG")  # 너무 많으므로 DEBUG 레벨

    def log_trade_execution(
        self,
        episode: int,
        step: int,
        agent_id: str,
        trade_type: str,
        price: float,
        quantity: float,
        fee: float,
        balance_before: float,
        balance_after: float,
        reason: str = None
    ):
        """
        거래 체결 로깅 (중요!)

        Args:
            episode: 에피소드 번호
            step: 스텝 번호
            agent_id: 에이전트 ID
            trade_type: 거래 유형 (BUY, SELL)
            price: 체결 가격
            quantity: 거래량
            fee: 수수료
            balance_before: 거래 전 잔액
            balance_after: 거래 후 잔액
            reason: 거래 이유 (옵션)
        """
        self.simulation_stats["total_trades"] += 1

        self.log({
            "event": "trade_execution",
            "episode": episode,
            "step": step,
            "agent_id": agent_id,
            "trade": {
                "type": trade_type,
                "price": float(price),
                "quantity": float(quantity),
                "fee": float(fee),
                "cost": float(price * quantity + fee)
            },
            "balance": {
                "before": float(balance_before),
                "after": float(balance_after),
                "change": float(balance_after - balance_before)
            },
            "reason": reason,
            "message": f"💰 {trade_type}: {quantity:.4f} @ {price:.2f}"
        })

    def log_episode_end(
        self,
        episode: int,
        agent_results: Dict[str, Dict[str, Any]],
        regime: str,
        total_steps: int
    ):
        """
        에피소드 종료 로깅 (핵심!)

        Args:
            episode: 에피소드 번호
            agent_results: 에이전트별 결과
                {
                    "agent_1": {
                        "total_trades": 3,
                        "win_rate": 0.66,
                        "total_pnl": 120.5,
                        "final_balance": 10120.5,
                        "sharpe_ratio": 0.5,
                        ...
                    }
                }
            regime: 레짐
            total_steps: 총 스텝 수
        """
        # 통계 업데이트
        for agent_id, result in agent_results.items():
            pnl = result.get("total_pnl", 0)
            self.simulation_stats["total_pnl"] += pnl

            trades = result.get("total_trades", 0)
            if trades == 0:
                self.simulation_stats["no_trade_episodes"] += 1
            else:
                win_rate = result.get("win_rate", 0)
                self.simulation_stats["winning_trades"] += int(trades * win_rate)
                self.simulation_stats["losing_trades"] += int(trades * (1 - win_rate))

        # 에이전트별 성과 비교
        pnls = [r.get("total_pnl", 0) for r in agent_results.values()]
        win_rates = [r.get("win_rate", 0) for r in agent_results.values()]
        sharpes = [r.get("sharpe_ratio", 0) for r in agent_results.values()]

        # 최고 성과 에이전트
        best_agent_id = max(agent_results.keys(), key=lambda k: agent_results[k].get("total_pnl", 0))
        best_agent_result = agent_results[best_agent_id]

        self.log({
            "event": "episode_end",
            "episode": episode,
            "regime": regime,
            "total_steps": total_steps,

            # 전체 통계
            "summary": {
                "num_agents": len(agent_results),
                "avg_pnl": float(np.mean(pnls)),
                "avg_win_rate": float(np.mean(win_rates)),
                "avg_sharpe": float(np.mean(sharpes)),
                "best_pnl": float(np.max(pnls)),
                "worst_pnl": float(np.min(pnls)),
                "pnl_std": float(np.std(pnls))
            },

            # 최고 에이전트
            "best_agent": {
                "agent_id": best_agent_id,
                "pnl": float(best_agent_result.get("total_pnl", 0)),
                "win_rate": float(best_agent_result.get("win_rate", 0)),
                "trades": int(best_agent_result.get("total_trades", 0)),
                "sharpe": float(best_agent_result.get("sharpe_ratio", 0))
            },

            # 에이전트별 상세 결과
            "agent_results": {
                agent_id: {
                    "total_trades": int(result.get("total_trades", 0)),
                    "win_rate": float(result.get("win_rate", 0)),
                    "total_pnl": float(result.get("total_pnl", 0)),
                    "avg_pnl_per_trade": float(result.get("avg_pnl_per_trade", 0)),
                    "max_drawdown": float(result.get("max_drawdown", 0)),
                    "sharpe_ratio": float(result.get("sharpe_ratio", 0)),
                    "final_balance": float(result.get("final_balance", 10000))
                }
                for agent_id, result in agent_results.items()
            }
        })

    def log_reward_distribution(self, episode: int, rewards: List[float]):
        """
        보상 분포 로깅 (중요!)

        Args:
            episode: 에피소드 번호
            rewards: 보상 리스트
        """
        if not rewards:
            return

        rewards = np.array(rewards)

        self.log({
            "event": "reward_distribution",
            "episode": episode,
            "count": len(rewards),
            "statistics": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "median": float(np.median(rewards)),
                "q25": float(np.percentile(rewards, 25)),
                "q75": float(np.percentile(rewards, 75))
            },
            "distribution": {
                "zero_ratio": float(np.mean(rewards == 0)),
                "positive_ratio": float(np.mean(rewards > 0)),
                "negative_ratio": float(np.mean(rewards < 0)),
                "large_positive": float(np.mean(rewards > 1.0)),
                "large_negative": float(np.mean(rewards < -1.0))
            },
            "warnings": {
                "too_sparse": np.mean(rewards == 0) > 0.9,  # 90% 이상이 0
                "low_variance": np.std(rewards) < 0.01,
                "has_outliers": np.max(np.abs(rewards)) > 10 * np.std(rewards)
            }
        })

    def log_action_sequence(self, episode: int, agent_id: str, actions: List[int], steps: List[int]):
        """
        액션 시퀀스 로깅 (패턴 분석용)

        Args:
            episode: 에피소드 번호
            agent_id: 에이전트 ID
            actions: 액션 리스트 [0, 0, 0, 1, 0, 2, ...]
            steps: 스텝 번호 리스트
        """
        action_names = ["HOLD", "BUY", "SELL"]
        action_sequence = [action_names[a] if a < len(action_names) else f"action_{a}" for a in actions]

        # 패턴 분석
        hold_streaks = []
        current_streak = 0
        for action in actions:
            if action == 0:  # HOLD
                current_streak += 1
            else:
                if current_streak > 0:
                    hold_streaks.append(current_streak)
                current_streak = 0

        self.log({
            "event": "action_sequence",
            "episode": episode,
            "agent_id": agent_id,
            "total_actions": len(actions),
            "sequence": action_sequence if len(actions) <= 100 else None,  # 100개 이하만 전체 출력
            "patterns": {
                "hold_ratio": float(np.mean(np.array(actions) == 0)),
                "buy_ratio": float(np.mean(np.array(actions) == 1)),
                "sell_ratio": float(np.mean(np.array(actions) == 2)),
                "avg_hold_streak": float(np.mean(hold_streaks)) if hold_streaks else 0,
                "max_hold_streak": int(np.max(hold_streaks)) if hold_streaks else 0,
                "action_switches": int(np.sum(np.diff(actions) != 0))
            }
        })

    def log_selfplay_end(
        self,
        coin: str,
        interval: str,
        total_episodes: int,
        summary: Dict[str, Any]
    ):
        """
        Self-play 종료 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            total_episodes: 총 에피소드 수
            summary: 전체 요약 통계
        """
        self.log({
            "event": "selfplay_end",
            "coin": coin,
            "interval": interval,
            "total_episodes": total_episodes,
            "summary": summary,
            "statistics": self.simulation_stats,
            "message": f"✅ Self-play 완료: {coin}-{interval}"
        })

        # 통계 저장
        self.stats.update(self.simulation_stats)
        self.save_stats()
