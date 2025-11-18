"""
디버그 로깅 시스템 - 베이스 클래스
모든 모듈에서 사용할 수 있는 통합 디버그 로거
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class DebugLogger:
    """모듈별 디버그 로거 베이스 클래스"""

    def __init__(self, module_name: str, session_id: str = None):
        """
        Args:
            module_name: 모듈명 (training, simulation, routing 등)
            session_id: 세션 ID (None이면 자동 생성)
        """
        self.module_name = module_name
        self.session_id = session_id or self._create_session_id()
        self.session_dir = self._setup_session_dir()
        self.log_file = self.session_dir / f"{module_name}.jsonl"
        self.error_file = self.session_dir / f"{module_name}_errors.jsonl"
        self.stats_file = self.session_dir / f"{module_name}_stats.json"

        # 통계 수집용
        self.stats = {
            "total_logs": 0,
            "total_errors": 0,
            "start_time": datetime.now().isoformat(),
            "module": module_name
        }

    def _create_session_id(self) -> str:
        """세션 ID 생성: YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _setup_session_dir(self) -> Path:
        """세션 디렉토리 설정 및 latest 링크 생성"""
        base_dir = Path(__file__).parent.parent / "debug_logs"
        session_dir = base_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # latest 심볼릭 링크 업데이트 (Windows에서는 junction)
        latest_link = base_dir / "latest"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                if os.name == 'nt':  # Windows
                    os.system(f'rmdir "{latest_link}"')
                else:
                    latest_link.unlink()

            if os.name == 'nt':  # Windows junction
                os.system(f'mklink /J "{latest_link}" "{session_dir}"')
            else:  # Unix symlink
                latest_link.symlink_to(session_dir, target_is_directory=True)
        except Exception as e:
            print(f"⚠️ latest 링크 생성 실패 (무시): {e}")

        return session_dir

    def _serialize_value(self, value: Any) -> Any:
        """numpy, pandas 등 특수 타입을 JSON 직렬화 가능하게 변환"""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif hasattr(value, 'item'):  # numpy scalar
            return value.item()
        else:
            return value

    def log(self, data: Dict[str, Any], level: str = "INFO"):
        """
        디버그 데이터 로깅

        Args:
            data: 로깅할 데이터 (dict)
            level: 로그 레벨 (INFO, WARNING, ERROR, DEBUG)
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": self.module_name,
                "level": level,
                **self._serialize_value(data)
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            self.stats["total_logs"] += 1

        except Exception as e:
            self._log_internal_error(f"로그 작성 실패: {e}", data)

    def log_error(self, error_msg: str, context: Dict[str, Any] = None, exception: Exception = None):
        """
        에러 로깅 (별도 파일)

        Args:
            error_msg: 에러 메시지
            context: 에러 발생 시 컨텍스트 정보
            exception: 예외 객체 (있으면 traceback 포함)
        """
        try:
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": self.module_name,
                "error_message": error_msg,
                "context": self._serialize_value(context or {}),
            }

            if exception:
                error_entry["exception_type"] = type(exception).__name__
                error_entry["exception_str"] = str(exception)
                error_entry["traceback"] = traceback.format_exc()

            with open(self.error_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")

            self.stats["total_errors"] += 1

        except Exception as e:
            self._log_internal_error(f"에러 로그 작성 실패: {e}", error_entry)

    def log_distribution(self, name: str, values: List[float], bins: int = 10):
        """
        분포 정보 상세 로깅

        Args:
            name: 분포 이름
            values: 값 리스트
            bins: 히스토그램 bin 수
        """
        if not values:
            return

        values = np.array(values)

        # 히스토그램
        hist, bin_edges = np.histogram(values, bins=bins)

        self.log({
            "type": "distribution",
            "name": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "histogram": {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
        })

    def log_action_distribution(self, actions: List[int], action_names: List[str] = None):
        """
        액션 분포 상세 로깅

        Args:
            actions: 액션 인덱스 리스트 [0, 0, 1, 2, 0, ...]
            action_names: 액션 이름 리스트 (예: ["HOLD", "BUY", "SELL"])
        """
        if not actions:
            return

        actions = np.array(actions)
        unique_actions, counts = np.unique(actions, return_counts=True)
        total = len(actions)

        action_names = action_names or [f"action_{i}" for i in range(len(unique_actions))]

        distribution = {}
        for action_idx, count in zip(unique_actions, counts):
            action_name = action_names[int(action_idx)] if int(action_idx) < len(action_names) else f"action_{action_idx}"
            distribution[action_name] = {
                "count": int(count),
                "ratio": float(count / total)
            }

        # Entropy 계산
        probs = counts / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(unique_actions))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        self.log({
            "type": "action_distribution",
            "total_actions": total,
            "unique_actions": len(unique_actions),
            "distribution": distribution,
            "entropy": float(entropy),
            "max_entropy": float(max_entropy),
            "normalized_entropy": float(normalized_entropy),
            "diversity_score": float(normalized_entropy),  # 0~1, 높을수록 다양함
        })

    def log_model_weights(self, model_name: str, weights: Dict[str, np.ndarray], layer_prefix: str = ""):
        """
        모델 가중치 통계 로깅 (전체 가중치는 너무 크므로 통계만)

        Args:
            model_name: 모델 이름
            weights: 가중치 딕셔너리 {layer_name: weight_array}
            layer_prefix: 레이어 이름 프리픽스
        """
        weight_stats = {}

        for layer_name, weight_array in weights.items():
            if not isinstance(weight_array, np.ndarray):
                continue

            full_name = f"{layer_prefix}{layer_name}" if layer_prefix else layer_name
            weight_stats[full_name] = {
                "shape": list(weight_array.shape),
                "mean": float(np.mean(weight_array)),
                "std": float(np.std(weight_array)),
                "min": float(np.min(weight_array)),
                "max": float(np.max(weight_array)),
                "norm": float(np.linalg.norm(weight_array)),
                "sparsity": float(np.mean(np.abs(weight_array) < 1e-6))  # 거의 0인 비율
            }

        self.log({
            "type": "model_weights",
            "model_name": model_name,
            "layer_count": len(weight_stats),
            "weights": weight_stats
        })

    def log_gradient_stats(self, gradients: Dict[str, np.ndarray]):
        """
        그래디언트 통계 로깅 (그래디언트 소실/폭발 감지)

        Args:
            gradients: 그래디언트 딕셔너리 {param_name: gradient_array}
        """
        grad_stats = {}
        all_grads = []

        for param_name, grad_array in gradients.items():
            if not isinstance(grad_array, np.ndarray):
                continue

            grad_flat = grad_array.flatten()
            all_grads.extend(grad_flat)

            grad_stats[param_name] = {
                "shape": list(grad_array.shape),
                "mean": float(np.mean(grad_array)),
                "std": float(np.std(grad_array)),
                "min": float(np.min(grad_array)),
                "max": float(np.max(grad_array)),
                "norm": float(np.linalg.norm(grad_array)),
                "has_nan": bool(np.any(np.isnan(grad_array))),
                "has_inf": bool(np.any(np.isinf(grad_array)))
            }

        if all_grads:
            all_grads = np.array(all_grads)
            global_norm = float(np.linalg.norm(all_grads))

            # 그래디언트 소실/폭발 감지
            vanishing = global_norm < 1e-6
            exploding = global_norm > 1e3

            self.log({
                "type": "gradient_stats",
                "param_count": len(grad_stats),
                "gradients": grad_stats,
                "global_norm": global_norm,
                "global_mean": float(np.mean(all_grads)),
                "global_std": float(np.std(all_grads)),
                "vanishing": vanishing,
                "exploding": exploding,
                "has_nan": bool(np.any(np.isnan(all_grads))),
                "has_inf": bool(np.any(np.isinf(all_grads)))
            })

    def save_stats(self):
        """통계 정보 저장"""
        try:
            self.stats["end_time"] = datetime.now().isoformat()
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log_internal_error(f"통계 저장 실패: {e}", self.stats)

    def _log_internal_error(self, error_msg: str, data: Any):
        """내부 에러 로깅 (로거 자체 에러)"""
        try:
            error_log_file = self.session_dir / "logger_internal_errors.txt"
            with open(error_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {error_msg}\n")
                f.write(f"Data: {str(data)[:500]}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
                f.write("-" * 80 + "\n")
        except:
            pass  # 최후의 보루도 실패하면 포기

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료 시 통계 저장"""
        self.save_stats()
        return False
