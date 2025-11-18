"""
Evolution 모듈 전용 디버그 로거
- Self-play 진화 추적
- 전략 생성 및 선택
- 적응도 평가
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .debug_logger import DebugLogger


class EvolutionDebugger(DebugLogger):
    """Evolution 모듈 전용 디버거"""

    def __init__(self, session_id: str = None):
        super().__init__("evolution", session_id)

        # 진화 통계
        self.evolution_stats = {
            "total_generations": 0,
            "total_strategies_created": 0,
            "total_strategies_selected": 0
        }

    def log_evolution_start(
        self,
        coin: str,
        interval: str,
        initial_population: int,
        target_strategies: int,
        config: Dict[str, Any] = None
    ):
        """
        진화 시작 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            initial_population: 초기 개체수
            target_strategies: 목표 전략 수
            config: 진화 설정
        """
        self.log({
            "event": "evolution_start",
            "coin": coin,
            "interval": interval,
            "initial_population": initial_population,
            "target_strategies": target_strategies,
            "config": config or {},
            "message": f"🧬 진화 시작: {coin}-{interval}"
        })

    def log_strategy_generation(
        self,
        coin: str,
        interval: str,
        generation: int,
        num_strategies: int,
        generation_method: str,
        regime: str
    ):
        """
        전략 생성 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            generation: 세대 번호
            num_strategies: 생성된 전략 수
            generation_method: 생성 방법
            regime: 대상 레짐
        """
        self.evolution_stats["total_generations"] += 1
        self.evolution_stats["total_strategies_created"] += num_strategies

        self.log({
            "event": "strategy_generation",
            "coin": coin,
            "interval": interval,
            "generation": generation,
            "num_strategies": num_strategies,
            "generation_method": generation_method,
            "regime": regime,
            "message": f"🔨 {generation}세대: {num_strategies}개 전략 생성 ({generation_method})"
        })

    def log_strategy_evaluation(
        self,
        coin: str,
        interval: str,
        strategy_id: str,
        regime: str,
        fitness_score: float,
        performance_metrics: Dict[str, Any]
    ):
        """
        전략 평가 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategy_id: 전략 ID
            regime: 레짐
            fitness_score: 적응도 점수
            performance_metrics: 성능 지표
        """
        self.log({
            "event": "strategy_evaluation",
            "coin": coin,
            "interval": interval,
            "strategy_id": strategy_id,
            "regime": regime,
            "fitness_score": float(fitness_score),
            "performance": {
                "win_rate": float(performance_metrics.get("win_rate", 0)),
                "total_pnl": float(performance_metrics.get("total_pnl", 0)),
                "sharpe_ratio": float(performance_metrics.get("sharpe_ratio", 0)),
                "max_drawdown": float(performance_metrics.get("max_drawdown", 0)),
                "total_trades": int(performance_metrics.get("total_trades", 0))
            }
        }, level="DEBUG")

    def log_fitness_distribution(
        self,
        coin: str,
        interval: str,
        generation: int,
        fitness_scores: List[float]
    ):
        """
        적응도 분포 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            generation: 세대 번호
            fitness_scores: 적응도 점수 리스트
        """
        if not fitness_scores:
            return

        scores = np.array(fitness_scores)

        self.log({
            "event": "fitness_distribution",
            "coin": coin,
            "interval": interval,
            "generation": generation,
            "num_strategies": len(fitness_scores),
            "statistics": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75))
            },
            "quality": {
                "excellent": int(np.sum(scores > 0.8)),  # 80점 이상
                "good": int(np.sum((scores > 0.6) & (scores <= 0.8))),  # 60-80점
                "average": int(np.sum((scores > 0.4) & (scores <= 0.6))),  # 40-60점
                "poor": int(np.sum(scores <= 0.4))  # 40점 이하
            }
        })

    def log_selection(
        self,
        coin: str,
        interval: str,
        generation: int,
        selected_strategies: List[Dict[str, Any]],
        selection_method: str,
        selection_ratio: float
    ):
        """
        전략 선택 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            generation: 세대 번호
            selected_strategies: 선택된 전략 리스트
            selection_method: 선택 방법
            selection_ratio: 선택 비율
        """
        self.evolution_stats["total_strategies_selected"] += len(selected_strategies)

        # 선택된 전략 분석
        fitness_scores = [s.get("fitness_score", 0) for s in selected_strategies]
        grades = [s.get("grade", "UNKNOWN") for s in selected_strategies]
        grade_counts = {grade: grades.count(grade) for grade in set(grades)}

        self.log({
            "event": "selection",
            "coin": coin,
            "interval": interval,
            "generation": generation,
            "num_selected": len(selected_strategies),
            "selection_method": selection_method,
            "selection_ratio": float(selection_ratio),
            "selected_fitness": {
                "mean": float(np.mean(fitness_scores)) if fitness_scores else 0,
                "min": float(np.min(fitness_scores)) if fitness_scores else 0,
                "max": float(np.max(fitness_scores)) if fitness_scores else 0
            },
            "grade_distribution": grade_counts,
            "message": f"✅ {len(selected_strategies)}개 전략 선택"
        })

    def log_crossover(
        self,
        coin: str,
        interval: str,
        parent1_id: str,
        parent2_id: str,
        offspring_id: str,
        crossover_method: str
    ):
        """
        교차 연산 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            parent1_id: 부모1 ID
            parent2_id: 부모2 ID
            offspring_id: 자손 ID
            crossover_method: 교차 방법
        """
        self.log({
            "event": "crossover",
            "coin": coin,
            "interval": interval,
            "parent1_id": parent1_id,
            "parent2_id": parent2_id,
            "offspring_id": offspring_id,
            "crossover_method": crossover_method
        }, level="DEBUG")

    def log_mutation(
        self,
        coin: str,
        interval: str,
        strategy_id: str,
        mutation_type: str,
        mutation_rate: float,
        mutated_params: List[str]
    ):
        """
        돌연변이 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategy_id: 전략 ID
            mutation_type: 돌연변이 유형
            mutation_rate: 돌연변이 비율
            mutated_params: 변이된 파라미터
        """
        self.log({
            "event": "mutation",
            "coin": coin,
            "interval": interval,
            "strategy_id": strategy_id,
            "mutation_type": mutation_type,
            "mutation_rate": float(mutation_rate),
            "mutated_params": mutated_params,
            "num_mutations": len(mutated_params)
        }, level="DEBUG")

    def log_generation_summary(
        self,
        coin: str,
        interval: str,
        generation: int,
        population_size: int,
        avg_fitness: float,
        best_fitness: float,
        worst_fitness: float,
        improvement: float
    ):
        """
        세대 요약 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            generation: 세대 번호
            population_size: 개체수
            avg_fitness: 평균 적응도
            best_fitness: 최고 적응도
            worst_fitness: 최악 적응도
            improvement: 개선률
        """
        self.log({
            "event": "generation_summary",
            "coin": coin,
            "interval": interval,
            "generation": generation,
            "population_size": population_size,
            "fitness": {
                "avg": float(avg_fitness),
                "best": float(best_fitness),
                "worst": float(worst_fitness),
                "range": float(best_fitness - worst_fitness)
            },
            "improvement": float(improvement),
            "message": f"📊 {generation}세대: 평균 적응도 {avg_fitness:.3f} (개선: {improvement:+.1f}%)"
        })

    def log_convergence_check(
        self,
        coin: str,
        interval: str,
        generation: int,
        converged: bool,
        convergence_metric: float,
        threshold: float
    ):
        """
        수렴 체크 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            generation: 세대 번호
            converged: 수렴 여부
            convergence_metric: 수렴 지표
            threshold: 수렴 임계값
        """
        self.log({
            "event": "convergence_check",
            "coin": coin,
            "interval": interval,
            "generation": generation,
            "converged": converged,
            "convergence_metric": float(convergence_metric),
            "threshold": float(threshold),
            "message": "🎯 진화 수렴" if converged else f"🔄 진화 계속 (수렴도: {convergence_metric:.3f})"
        }, level="INFO" if converged else "DEBUG")

    def log_evolution_end(
        self,
        coin: str,
        interval: str,
        total_generations: int,
        final_population: int,
        best_strategies: List[Dict[str, Any]],
        converged: bool
    ):
        """
        진화 종료 로깅

        Args:
            coin: 코인 심볼
            interval: 인터벌
            total_generations: 총 세대 수
            final_population: 최종 개체수
            best_strategies: 최고 전략들
            converged: 수렴 여부
        """
        # 최고 전략 분석
        best_fitness = [s.get("fitness_score", 0) for s in best_strategies]

        self.log({
            "event": "evolution_end",
            "coin": coin,
            "interval": interval,
            "total_generations": total_generations,
            "final_population": final_population,
            "converged": converged,
            "best_strategies": {
                "count": len(best_strategies),
                "avg_fitness": float(np.mean(best_fitness)) if best_fitness else 0,
                "top_strategy_fitness": float(max(best_fitness)) if best_fitness else 0
            },
            "statistics": self.evolution_stats,
            "message": f"✅ 진화 완료: {total_generations}세대, {final_population}개 전략"
        })

        # 통계 저장
        self.stats.update(self.evolution_stats)
        self.save_stats()
