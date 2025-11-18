"""
Training 모듈 전용 디버그 로거
- PPO 학습 과정 상세 추적
- 액션 다양성 문제 감지
- 그래디언트 소실/폭발 감지
- 학습 정체 감지
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .debug_logger import DebugLogger


class TrainingDebugger(DebugLogger):
    """Training 모듈 전용 디버거"""

    def __init__(self, session_id: str = None):
        super().__init__("training", session_id)

        # 학습 추적용 통계
        self.training_stats = {
            "total_epochs": 0,
            "total_batches": 0,
            "action_diversity_warnings": 0,
            "gradient_issues": 0,
            "loss_improvements": 0,
            "best_loss": float('inf')
        }

    def log_training_start(self, config: Dict[str, Any]):
        """
        학습 시작 로깅

        Args:
            config: 학습 설정 (lr, epochs, batch_size 등)
        """
        self.log({
            "event": "training_start",
            "config": config,
            "message": "🚀 PPO 학습 시작"
        })

    def log_epoch_start(self, epoch: int, total_epochs: int, learning_rate: float):
        """
        Epoch 시작 로깅

        Args:
            epoch: 현재 epoch 번호
            total_epochs: 전체 epoch 수
            learning_rate: 현재 학습률
        """
        self.training_stats["total_epochs"] = epoch

        self.log({
            "event": "epoch_start",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "learning_rate": learning_rate,
            "progress": f"{epoch}/{total_epochs}"
        })

    def log_batch_training(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        entropy_loss: float,
        actions: List[int],
        action_probs: np.ndarray,
        entropy_coef: float,
        clip_ratio: float = None,
        kl_divergence: float = None
    ):
        """
        배치 학습 상세 로깅 (가장 중요!)

        Args:
            epoch: Epoch 번호
            batch_idx: 배치 인덱스
            total_batches: 전체 배치 수
            loss: 총 손실
            policy_loss: 정책 손실
            value_loss: 가치 함수 손실
            entropy_loss: 엔트로피 손실
            actions: 배치의 액션 리스트 [0, 0, 1, 2, ...]
            action_probs: 액션 확률 분포 (batch_size, num_actions)
            entropy_coef: 엔트로피 계수
            clip_ratio: PPO 클립 비율 (옵션)
            kl_divergence: KL divergence (옵션)
        """
        self.training_stats["total_batches"] += 1

        # 액션 분포 분석
        action_names = ["HOLD", "BUY", "SELL"]
        unique_actions = len(np.unique(actions))
        action_counts = np.bincount(actions, minlength=3)
        action_ratios = action_counts / len(actions)

        # 액션 확률 통계
        mean_action_probs = np.mean(action_probs, axis=0)
        max_action_prob = np.max(mean_action_probs)
        min_action_prob = np.min(mean_action_probs)

        # Entropy 계산
        probs = action_counts / len(actions)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(3)  # 3개 액션
        normalized_entropy = entropy / max_entropy

        # 액션 다양성 경고 감지
        action_diversity_warning = False
        if unique_actions == 1 or action_ratios.max() > 0.95:
            self.training_stats["action_diversity_warnings"] += 1
            action_diversity_warning = True

        # 상세 로그
        self.log({
            "event": "batch_training",
            "epoch": epoch,
            "batch": batch_idx,
            "total_batches": total_batches,
            "progress": f"{batch_idx}/{total_batches}",

            # 손실 정보
            "loss": {
                "total": float(loss),
                "policy": float(policy_loss),
                "value": float(value_loss),
                "entropy": float(entropy_loss)
            },

            # 액션 분포 (핵심!)
            "action_distribution": {
                action_names[i]: {
                    "count": int(action_counts[i]),
                    "ratio": float(action_ratios[i])
                }
                for i in range(len(action_names))
            },
            "action_stats": {
                "unique_actions": int(unique_actions),
                "total_actions": len(actions),
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "diversity_score": float(normalized_entropy),
                "dominant_action": action_names[np.argmax(action_ratios)],
                "dominant_ratio": float(action_ratios.max())
            },

            # 액션 확률 분석
            "action_probabilities": {
                "mean": {
                    action_names[i]: float(mean_action_probs[i])
                    for i in range(len(action_names))
                },
                "max_prob": float(max_action_prob),
                "min_prob": float(min_action_prob),
                "prob_spread": float(max_action_prob - min_action_prob)
            },

            # 하이퍼파라미터
            "hyperparameters": {
                "entropy_coef": float(entropy_coef),
                "clip_ratio": float(clip_ratio) if clip_ratio is not None else None,
                "kl_divergence": float(kl_divergence) if kl_divergence is not None else None
            },

            # 경고
            "warnings": {
                "action_diversity_low": action_diversity_warning,
                "dominant_action_warning": action_ratios.max() > 0.90,
                "entropy_too_low": normalized_entropy < 0.3
            }
        })

    def log_gradient_update(
        self,
        epoch: int,
        batch_idx: int,
        gradients: Dict[str, np.ndarray],
        learning_rate: float,
        grad_norm: float = None,
        clipped: bool = False
    ):
        """
        그래디언트 업데이트 로깅

        Args:
            epoch: Epoch 번호
            batch_idx: 배치 인덱스
            gradients: 그래디언트 딕셔너리
            learning_rate: 학습률
            grad_norm: 그래디언트 norm
            clipped: 그래디언트 클리핑 여부
        """
        # 그래디언트 통계 자동 계산
        self.log_gradient_stats(gradients)

        # 그래디언트 이슈 감지
        has_nan = any(np.any(np.isnan(g)) for g in gradients.values() if isinstance(g, np.ndarray))
        has_inf = any(np.any(np.isinf(g)) for g in gradients.values() if isinstance(g, np.ndarray))

        if has_nan or has_inf or (grad_norm and grad_norm > 1e3):
            self.training_stats["gradient_issues"] += 1

        self.log({
            "event": "gradient_update",
            "epoch": epoch,
            "batch": batch_idx,
            "learning_rate": float(learning_rate),
            "grad_norm": float(grad_norm) if grad_norm else None,
            "clipped": clipped,
            "warnings": {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "exploding": grad_norm > 1e3 if grad_norm else False,
                "vanishing": grad_norm < 1e-6 if grad_norm else False
            }
        })

    def log_epoch_end(
        self,
        epoch: int,
        avg_loss: float,
        best_loss: float,
        improved: bool,
        no_improvement_count: int,
        learning_rate: float
    ):
        """
        Epoch 종료 로깅

        Args:
            epoch: Epoch 번호
            avg_loss: 평균 손실
            best_loss: 최고 손실
            improved: 개선 여부
            no_improvement_count: 개선 없는 epoch 수
            learning_rate: 현재 학습률
        """
        if improved:
            self.training_stats["loss_improvements"] += 1
            self.training_stats["best_loss"] = best_loss

        self.log({
            "event": "epoch_end",
            "epoch": epoch,
            "avg_loss": float(avg_loss),
            "best_loss": float(best_loss),
            "improved": improved,
            "no_improvement_count": no_improvement_count,
            "learning_rate": float(learning_rate),
            "status": "✅ 개선" if improved else f"⚠️ 개선 없음 ({no_improvement_count}회)"
        })

    def log_early_stopping(self, epoch: int, reason: str, best_loss: float):
        """
        조기 종료 로깅

        Args:
            epoch: 종료된 Epoch
            reason: 종료 이유
            best_loss: 최고 손실
        """
        self.log({
            "event": "early_stopping",
            "epoch": epoch,
            "reason": reason,
            "best_loss": float(best_loss),
            "message": f"🛑 조기 종료: {reason}"
        }, level="WARNING")

    def log_learning_rate_adjustment(self, old_lr: float, new_lr: float, reason: str):
        """
        학습률 조정 로깅

        Args:
            old_lr: 이전 학습률
            new_lr: 새 학습률
            reason: 조정 이유
        """
        self.log({
            "event": "learning_rate_adjustment",
            "old_lr": float(old_lr),
            "new_lr": float(new_lr),
            "change_ratio": float(new_lr / old_lr) if old_lr > 0 else 0,
            "reason": reason,
            "message": f"📉 학습률 조정: {old_lr:.6f} → {new_lr:.6f}"
        })

    def log_training_data_stats(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray = None,
        advantages: np.ndarray = None,
        returns: np.ndarray = None
    ):
        """
        학습 데이터 통계 로깅

        Args:
            states: 상태 배열 (batch_size, state_dim)
            actions: 액션 배열 (batch_size,)
            rewards: 보상 배열 (batch_size,)
            values: 가치 추정 배열 (옵션)
            advantages: Advantage 배열 (옵션)
            returns: Return 배열 (옵션)
        """
        data_stats = {
            "event": "training_data_stats",
            "batch_size": len(states),
            "state_dim": states.shape[1] if len(states.shape) > 1 else 1,

            "states": {
                "shape": list(states.shape),
                "mean": float(np.mean(states)),
                "std": float(np.std(states)),
                "min": float(np.min(states)),
                "max": float(np.max(states)),
                "has_nan": bool(np.any(np.isnan(states))),
                "has_inf": bool(np.any(np.isinf(states)))
            },

            "actions": {
                "unique": len(np.unique(actions)),
                "distribution": np.bincount(actions.astype(int), minlength=3).tolist()
            },

            "rewards": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "zero_ratio": float(np.mean(rewards == 0)),
                "positive_ratio": float(np.mean(rewards > 0)),
                "negative_ratio": float(np.mean(rewards < 0))
            }
        }

        if values is not None:
            data_stats["values"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }

        if advantages is not None:
            data_stats["advantages"] = {
                "mean": float(np.mean(advantages)),
                "std": float(np.std(advantages)),
                "min": float(np.min(advantages)),
                "max": float(np.max(advantages))
            }

        if returns is not None:
            data_stats["returns"] = {
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "min": float(np.min(returns)),
                "max": float(np.max(returns))
            }

        self.log(data_stats)

    def log_training_end(self, total_epochs: int, best_loss: float, final_loss: float, converged: bool):
        """
        학습 종료 로깅

        Args:
            total_epochs: 총 학습 epoch 수
            best_loss: 최고 손실
            final_loss: 최종 손실
            converged: 수렴 여부
        """
        summary = {
            "event": "training_end",
            "total_epochs": total_epochs,
            "best_loss": float(best_loss),
            "final_loss": float(final_loss),
            "converged": converged,
            "statistics": self.training_stats,
            "message": "✅ 학습 완료" if converged else "⚠️ 학습 미수렴"
        }

        self.log(summary)

        # 통계 저장
        self.stats.update(self.training_stats)
        self.save_stats()
