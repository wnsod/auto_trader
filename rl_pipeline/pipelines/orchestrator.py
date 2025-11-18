"""
통합 파이프라인 오케스트레이터
"""

import sys
import os
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# 새로운 파이프라인 구조 import
try:
    import rl_pipeline.core.env as core_env
    import rl_pipeline.core.errors as core_errors
    import rl_pipeline.strategy.manager as strategy_manager
    import rl_pipeline.simulation.selfplay as selfplay
    import rl_pipeline.routing.regime_router as regime_router
    import rl_pipeline.analysis.integrated_analyzer as integrated_analyzer
    import rl_pipeline.analysis.integrated_analysis_v1 as integrated_analysis_v1
    import rl_pipeline.db.schema as db_schema
    import rl_pipeline.db.connection_pool as db_pool
    import rl_pipeline.db.reads as db_reads
    import rl_pipeline.db.learning_results as learning_results

    config = core_env.config
    AZError = core_errors.AZError
    create_run_record = strategy_manager.create_run_record
    update_run_record = strategy_manager.update_run_record
    create_coin_strategies = strategy_manager.create_coin_strategies
    create_global_strategies = strategy_manager.create_global_strategies
    run_self_play_test = selfplay.run_self_play_test
    RegimeRouter = regime_router.RegimeRouter
    create_regime_routing_strategies = regime_router.create_regime_routing_strategies
    IntegratedAnalyzer = integrated_analyzer.IntegratedAnalyzer
    IntegratedAnalyzerV1 = integrated_analysis_v1.IntegratedAnalyzerV1
    analyze_coin_strategies = integrated_analyzer.analyze_coin_strategies
    analyze_global_strategies = integrated_analyzer.analyze_global_strategies
    ensure_indexes = db_schema.ensure_indexes
    setup_database_tables = db_schema.setup_database_tables
    create_coin_strategies_table = db_schema.create_coin_strategies_table
    get_optimized_db_connection = db_pool.get_optimized_db_connection
    save_selfplay_results = learning_results.save_selfplay_results
    save_regime_routing_results = learning_results.save_regime_routing_results
    save_integrated_analysis_results = learning_results.save_integrated_analysis_results

    NEW_PIPELINE_AVAILABLE = True
    # 🔥 중복 메시지 제거 (absolute_zero_system.py에서 이미 출력)

except ImportError as e:
    print(f"새로운 파이프라인 모듈 import 실패: {e}")
    config = None
    AZError = Exception
    NEW_PIPELINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# 환경변수
AZ_STRATEGY_POOL_SIZE = int(os.getenv('AZ_STRATEGY_POOL_SIZE', '15000'))
AZ_SELFPLAY_EPISODES = int(os.getenv('AZ_SELFPLAY_EPISODES', '200'))
AZ_SELFPLAY_AGENTS_PER_EPISODE = int(os.getenv('AZ_SELFPLAY_AGENTS_PER_EPISODE', '4'))  # 에피소드당 에이전트 수
PREDICTIVE_SELFPLAY_RATIO = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
PREDICTIVE_SELFPLAY_EPISODES = int(os.getenv('PREDICTIVE_SELFPLAY_EPISODES', '50'))  # 🔥 예측 Self-play 강화학습 에피소드 수 (50개 전략 × 50번 반복, 최대값)
PREDICTIVE_SELFPLAY_LEARNING_RATE = float(os.getenv('PREDICTIVE_SELFPLAY_LEARNING_RATE', '0.1'))  # 🔥 예측 정책 업데이트 학습률
PREDICTIVE_SELFPLAY_EARLY_STOP = os.getenv('PREDICTIVE_SELFPLAY_EARLY_STOP', 'true').lower() == 'true'  # 🔥 조기 종료 활성화
PREDICTIVE_SELFPLAY_EARLY_STOP_PATIENCE = int(os.getenv('PREDICTIVE_SELFPLAY_EARLY_STOP_PATIENCE', '15'))  # 🔥 개선: 조기 종료 인내심 (5 → 15)
PREDICTIVE_SELFPLAY_EARLY_STOP_ACCURACY = float(os.getenv('PREDICTIVE_SELFPLAY_EARLY_STOP_ACCURACY', '0.85'))  # 🔥 조기 종료 정확도 임계값
PREDICTIVE_SELFPLAY_MIN_IMPROVEMENT = float(os.getenv('PREDICTIVE_SELFPLAY_MIN_IMPROVEMENT', '0.01'))  # 🔥 최소 개선 임계값
PREDICTIVE_SELFPLAY_MIN_EPISODES = int(os.getenv('PREDICTIVE_SELFPLAY_MIN_EPISODES', '20'))  # 🔥 개선: 최소 에피소드 수 (10 → 20)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    coin: str
    interval: str
    strategies_created: int = 0
    selfplay_episodes: int = 0
    regime_detected: str = "neutral"
    routing_results: int = 0
    signal_score: float = 0.0
    signal_action: str = "HOLD"
    execution_time: float = 0.0
    status: str = "pending"
    created_at: str = ""
    selfplay_result: Optional[Dict[str, Any]] = None  # 🔥 self-play 결과 저장

def validate_selfplay_result(result: Dict, coin: str, interval: str) -> Dict[str, Any]:
    """예측 Self-play 결과 검증

    Args:
        result: Self-play 결과 dict
        coin: 코인 심볼
        interval: 인터벌

    Returns:
        Dict: 검증 결과 {'valid': bool, 'issues': List[str], 'warnings': List[str]}
    """
    issues = []
    warnings = []

    try:
        # 1. 필수 필드 확인
        required_fields = ['cycle_results', 'episodes', 'avg_accuracy', 'best_accuracy', 'strategy_count']
        for field in required_fields:
            if field not in result:
                issues.append(f"필수 필드 누락: {field}")

        # 2. 데이터 타입 확인
        if 'episodes' in result and not isinstance(result['episodes'], int):
            issues.append(f"episodes 타입 오류: {type(result['episodes'])}")

        if 'avg_accuracy' in result and not isinstance(result['avg_accuracy'], (int, float)):
            issues.append(f"avg_accuracy 타입 오류: {type(result['avg_accuracy'])}")

        if 'cycle_results' in result and not isinstance(result['cycle_results'], list):
            issues.append(f"cycle_results 타입 오류: {type(result['cycle_results'])}")

        # 3. 논리적 일관성 확인
        if 'episodes' in result and 'cycle_results' in result:
            if result['episodes'] != len(result['cycle_results']):
                warnings.append(f"에피소드 수 불일치: episodes={result['episodes']}, cycle_results 길이={len(result['cycle_results'])}")

        # 4. 정확도 범위 확인
        if 'avg_accuracy' in result:
            accuracy = result['avg_accuracy']
            if not (0 <= accuracy <= 1):
                issues.append(f"avg_accuracy 범위 오류: {accuracy} (0~1 범위 벗어남)")

            # 인터벌별 예상 정확도 범위
            expected_ranges = {
                '15m': (0.70, 1.00),
                '30m': (0.65, 1.00),
                '240m': (0.50, 0.85),
                '1d': (0.45, 0.80)
            }

            if interval in expected_ranges:
                min_acc, max_acc = expected_ranges[interval]
                if accuracy < min_acc * 0.8:  # 20% 마진
                    warnings.append(f"{interval} 정확도가 예상보다 낮음: {accuracy:.3f} (예상 범위: {min_acc:.2f}~{max_acc:.2f})")
                elif accuracy > max_acc * 1.1:
                    warnings.append(f"{interval} 정확도가 예상보다 높음: {accuracy:.3f} (과적합 가능성)")

        # 5. 전략 수 확인
        if 'strategy_count' in result:
            if result['strategy_count'] < 10:
                warnings.append(f"전략 수가 너무 적음: {result['strategy_count']}")
            elif result['strategy_count'] > 200:
                warnings.append(f"전략 수가 너무 많음: {result['strategy_count']}")

        # 6. 조기 종료 확인
        if 'episodes' in result:
            from rl_pipeline.pipelines.orchestrator import PREDICTIVE_SELFPLAY_EPISODES
            if result['episodes'] < 5:
                issues.append(f"에피소드 수가 너무 적음: {result['episodes']} (최소 5개 필요)")
            elif result['episodes'] < PREDICTIVE_SELFPLAY_EPISODES * 0.2:
                warnings.append(f"매우 이른 조기 종료: {result['episodes']}/{PREDICTIVE_SELFPLAY_EPISODES} 에피소드")

        # 7. cycle_results 상세 검증
        if 'cycle_results' in result and isinstance(result['cycle_results'], list):
            for idx, cycle in enumerate(result['cycle_results']):
                if not isinstance(cycle, dict):
                    issues.append(f"cycle_results[{idx}] 타입 오류: {type(cycle)}")
                    continue

                # 각 cycle의 필수 필드
                cycle_fields = ['episode', 'accuracy']
                for field in cycle_fields:
                    if field not in cycle:
                        issues.append(f"cycle_results[{idx}]에 필드 누락: {field}")

    except Exception as e:
        issues.append(f"검증 중 예외 발생: {str(e)}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'coin': coin,
        'interval': interval
    }


def validate_integrated_learning_data(
    coin: str,
    all_interval_selfplay: Dict[str, Dict],
    pipeline_results: List,
    min_intervals: int = 2,
    min_total_episodes: int = 10
) -> Dict[str, Any]:
    """
    통합 학습 데이터 검증

    Args:
        coin: 코인 심볼
        all_interval_selfplay: 인터벌별 Self-play 결과 {interval: result_dict}
        pipeline_results: 파이프라인 결과 리스트
        min_intervals: 최소 필요 인터벌 수
        min_total_episodes: 최소 총 에피소드 수

    Returns:
        검증 결과 딕셔너리
    """
    issues = []
    warnings = []
    stats = {}

    # 1. 인터벌 수 검증
    num_intervals = len(all_interval_selfplay)
    stats['num_intervals'] = num_intervals

    if num_intervals == 0:
        issues.append("Self-play 결과가 전혀 없음")
        return {
            'valid': False,
            'issues': issues,
            'warnings': warnings,
            'stats': stats
        }

    if num_intervals < min_intervals:
        warnings.append(f"인터벌 수 부족 ({num_intervals} < {min_intervals})")

    # 2. 각 인터벌별 데이터 검증
    interval_stats = {}
    total_episodes = 0
    total_accuracy_sum = 0
    accuracy_count = 0

    for interval, sp_result in all_interval_selfplay.items():
        interval_stat = {
            'interval': interval,
            'valid': True,
            'episodes': 0,
            'avg_accuracy': 0.0,
            'best_accuracy': 0.0,
            'issues': []
        }

        # 2.1 결과 타입 검증
        if not isinstance(sp_result, dict):
            interval_stat['valid'] = False
            interval_stat['issues'].append(f"결과가 dict 타입이 아님: {type(sp_result)}")
            interval_stats[interval] = interval_stat
            continue

        # 2.2 필수 필드 검증
        required_fields = ['cycle_results', 'episodes', 'avg_accuracy']
        missing_fields = [f for f in required_fields if f not in sp_result]
        if missing_fields:
            interval_stat['valid'] = False
            interval_stat['issues'].append(f"필수 필드 누락: {missing_fields}")

        # 2.3 에피소드 수 검증
        cycle_results = sp_result.get('cycle_results', [])
        episodes = sp_result.get('episodes', 0)

        if not isinstance(cycle_results, list):
            interval_stat['valid'] = False
            interval_stat['issues'].append("cycle_results가 list 타입이 아님")
        else:
            interval_stat['episodes'] = len(cycle_results)
            total_episodes += len(cycle_results)

            if len(cycle_results) != episodes:
                warnings.append(f"{interval}: cycle_results 길이({len(cycle_results)})와 episodes({episodes}) 불일치")

            # 인터벌별 최소 에피소드 검증
            interval_min_episodes = {
                '15m': 5,
                '30m': 5,
                '240m': 8,
                '1d': 10
            }
            min_eps = interval_min_episodes.get(interval, 5)
            if len(cycle_results) < min_eps:
                warnings.append(f"{interval}: 에피소드 수 부족 ({len(cycle_results)} < {min_eps})")

        # 2.4 정확도 검증
        avg_accuracy = sp_result.get('avg_accuracy', 0.0)
        best_accuracy = sp_result.get('best_accuracy', 0.0)

        interval_stat['avg_accuracy'] = avg_accuracy
        interval_stat['best_accuracy'] = best_accuracy

        if avg_accuracy > 0:
            total_accuracy_sum += avg_accuracy
            accuracy_count += 1

        # 정확도 범위 검증 (인터벌별 기대 범위)
        expected_ranges = {
            '15m': (0.60, 1.00),
            '30m': (0.55, 1.00),
            '240m': (0.40, 0.90),
            '1d': (0.35, 0.85)
        }

        if interval in expected_ranges:
            min_acc, max_acc = expected_ranges[interval]
            if avg_accuracy < min_acc:
                warnings.append(f"{interval}: 평균 정확도가 기대 범위보다 낮음 ({avg_accuracy:.2%} < {min_acc:.2%})")
            elif avg_accuracy > max_acc:
                warnings.append(f"{interval}: 평균 정확도가 비정상적으로 높음 ({avg_accuracy:.2%} > {max_acc:.2%})")

        if best_accuracy < avg_accuracy:
            warnings.append(f"{interval}: best_accuracy({best_accuracy:.2%}) < avg_accuracy({avg_accuracy:.2%})")

        # 2.5 cycle_results 내부 데이터 검증
        if isinstance(cycle_results, list) and len(cycle_results) > 0:
            for idx, cycle in enumerate(cycle_results):
                if not isinstance(cycle, dict):
                    warnings.append(f"{interval}: cycle_results[{idx}]가 dict가 아님")
                    continue

                if 'accuracy' not in cycle:
                    warnings.append(f"{interval}: cycle_results[{idx}]에 accuracy 없음")

                # 정확도 추세 검증 (마지막 5개 에피소드)
                if idx >= len(cycle_results) - 5:
                    cycle_acc = cycle.get('accuracy', 0)
                    if cycle_acc < 0.3:  # 너무 낮은 정확도
                        warnings.append(f"{interval}: 에피소드 {idx+1} 정확도 매우 낮음 ({cycle_acc:.2%})")

        interval_stats[interval] = interval_stat

    # 3. 총 에피소드 수 검증
    stats['total_episodes'] = total_episodes
    if total_episodes < min_total_episodes:
        issues.append(f"총 에피소드 수 부족 ({total_episodes} < {min_total_episodes})")

    # 4. 평균 정확도 검증
    if accuracy_count > 0:
        overall_avg_accuracy = total_accuracy_sum / accuracy_count
        stats['overall_avg_accuracy'] = overall_avg_accuracy

        if overall_avg_accuracy < 0.50:
            warnings.append(f"전체 평균 정확도가 낮음 ({overall_avg_accuracy:.2%})")
    else:
        stats['overall_avg_accuracy'] = 0.0
        warnings.append("정확도 데이터 없음")

    # 5. 인터벌 분포 검증
    stats['interval_distribution'] = interval_stats

    # 긴 인터벌(240m, 1d) 데이터 검증
    long_intervals = ['240m', '1d']
    has_long_interval = any(i in all_interval_selfplay for i in long_intervals)
    if not has_long_interval:
        warnings.append("장기 인터벌(240m, 1d) 데이터 없음 - 학습 품질 저하 가능")

    # 6. 파이프라인 결과와 일관성 검증
    pipeline_intervals = {r.interval for r in pipeline_results if r.interval}
    selfplay_intervals = set(all_interval_selfplay.keys())

    missing_in_selfplay = pipeline_intervals - selfplay_intervals
    if missing_in_selfplay:
        warnings.append(f"파이프라인은 완료되었으나 Self-play 결과 없는 인터벌: {missing_in_selfplay}")

    # 7. 데이터 품질 점수 계산
    quality_score = 0.0

    # 인터벌 수 점수 (최대 30점)
    quality_score += min(num_intervals / 4.0 * 30, 30)

    # 에피소드 수 점수 (최대 30점)
    quality_score += min(total_episodes / 50.0 * 30, 30)

    # 정확도 점수 (최대 40점)
    if accuracy_count > 0:
        # 50% 정확도를 0점, 80% 이상을 만점으로
        acc_score = max(0, (overall_avg_accuracy - 0.50) / 0.30 * 40)
        quality_score += min(acc_score, 40)

    stats['quality_score'] = round(quality_score, 2)

    # 8. 최종 검증 결과
    valid = len(issues) == 0

    return {
        'valid': valid,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats['quality_score']
    }


def validate_global_strategy_pool(
    pool: Dict[str, List[Dict]],
    coins: List[str],
    intervals: List[str],
    min_strategies_per_interval: int = 10
) -> Dict[str, Any]:
    """
    글로벌 전략 풀 검증 (1단계: 개별 전략 수집)

    Args:
        pool: 인터벌별 전략 풀 {interval: [strategies]}
        coins: 수집 대상 코인 목록
        intervals: 수집 대상 인터벌 목록
        min_strategies_per_interval: 인터벌당 최소 전략 수

    Returns:
        검증 결과 딕셔너리
    """
    issues = []
    warnings = []
    stats = {}

    # 1. 기본 검증
    if not pool:
        issues.append("수집된 전략 풀이 비어있음")
        return {
            'valid': False,
            'issues': issues,
            'warnings': warnings,
            'stats': {}
        }

    # 2. 인터벌별 전략 수 검증
    interval_stats = {}
    total_strategies = 0

    for interval in intervals:
        strategies = pool.get(interval, [])
        strategy_count = len(strategies)
        total_strategies += strategy_count

        interval_stat = {
            'interval': interval,
            'strategy_count': strategy_count,
            'valid': True,
            'issues': []
        }

        if strategy_count == 0:
            warnings.append(f"{interval}: 전략 없음")
            interval_stat['valid'] = False
            interval_stat['issues'].append("전략 없음")
        elif strategy_count < min_strategies_per_interval:
            warnings.append(f"{interval}: 전략 수 부족 ({strategy_count} < {min_strategies_per_interval})")

        # 전략 품질 검증 (샘플링)
        if strategies:
            # 첫 10개 전략 샘플링
            sample_size = min(10, len(strategies))
            for i, strategy in enumerate(strategies[:sample_size]):
                if not isinstance(strategy, dict):
                    interval_stat['issues'].append(f"전략 [{i}]가 dict 타입이 아님: {type(strategy)}")
                    continue

                # 필수 필드 확인
                required_fields = ['strategy_id', 'coin', 'interval']
                missing = [f for f in required_fields if f not in strategy]
                if missing:
                    interval_stat['issues'].append(f"전략 [{i}] 필수 필드 누락: {missing}")

        interval_stats[interval] = interval_stat

    stats['interval_distribution'] = interval_stats
    stats['total_strategies'] = total_strategies
    stats['intervals_covered'] = len([i for i in intervals if pool.get(i)])
    stats['intervals_expected'] = len(intervals)

    # 3. 전체 검증
    if stats['intervals_covered'] < len(intervals) / 2:
        warnings.append(f"인터벌 커버리지 낮음 ({stats['intervals_covered']}/{len(intervals)})")

    if total_strategies == 0:
        issues.append("전체 전략 수 0개")

    # 4. 품질 점수 계산
    quality_score = 0.0

    # 인터벌 커버리지 점수 (40점)
    coverage_ratio = stats['intervals_covered'] / max(1, len(intervals))
    quality_score += coverage_ratio * 40

    # 전략 수 점수 (60점) - 인터벌당 평균 50개 기준
    avg_strategies_per_interval = total_strategies / max(1, len(intervals))
    strategy_score = min(avg_strategies_per_interval / 50.0 * 60, 60)
    quality_score += strategy_score

    stats['quality_score'] = round(quality_score, 2)

    # 5. 최종 검증
    valid = len(issues) == 0

    return {
        'valid': valid,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats['quality_score']
    }


def validate_global_strategy_patterns(
    patterns: Dict[str, Any],
    min_patterns_per_interval: int = 3
) -> Dict[str, Any]:
    """
    글로벌 전략 패턴 검증 (3단계: 공통 패턴 추출)

    Args:
        patterns: 추출된 패턴 딕셔너리
        min_patterns_per_interval: 인터벌당 최소 패턴 수

    Returns:
        검증 결과 딕셔너리
    """
    issues = []
    warnings = []
    stats = {}

    # 1. 기본 검증
    if not patterns:
        issues.append("추출된 패턴 없음")
        return {
            'valid': False,
            'issues': issues,
            'warnings': warnings,
            'stats': {}
        }

    # 2. 패턴 구조 검증
    total_patterns = 0
    interval_pattern_stats = {}

    for interval, interval_patterns in patterns.items():
        if not isinstance(interval_patterns, (list, dict)):
            warnings.append(f"{interval}: 패턴 타입 오류 ({type(interval_patterns)})")
            continue

        pattern_count = len(interval_patterns) if isinstance(interval_patterns, (list, dict)) else 0
        total_patterns += pattern_count

        interval_pattern_stats[interval] = {
            'interval': interval,
            'pattern_count': pattern_count
        }

        if pattern_count < min_patterns_per_interval:
            warnings.append(f"{interval}: 패턴 수 부족 ({pattern_count} < {min_patterns_per_interval})")

    stats['interval_patterns'] = interval_pattern_stats
    stats['total_patterns'] = total_patterns
    stats['intervals_covered'] = len(patterns)

    # 3. 품질 점수
    quality_score = 0.0

    # 패턴 수 점수 (100점)
    if total_patterns > 0:
        quality_score = min(total_patterns / 20.0 * 100, 100)

    stats['quality_score'] = round(quality_score, 2)

    # 4. 최종 검증
    valid = len(issues) == 0 and total_patterns > 0

    return {
        'valid': valid,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats['quality_score']
    }


def validate_global_strategy_quality(
    final_strategies: Dict[str, List[Dict]],
    intervals: List[str],
    min_strategies_per_interval: int = 5
) -> Dict[str, Any]:
    """
    최종 글로벌 전략 품질 검증 (7단계: 저장 전)

    Args:
        final_strategies: 최종 글로벌 전략 {interval: [strategies]}
        intervals: 기대되는 인터벌 목록
        min_strategies_per_interval: 인터벌당 최소 전략 수

    Returns:
        검증 결과 딕셔너리
    """
    issues = []
    warnings = []
    stats = {}

    # 1. 기본 검증
    if not final_strategies:
        issues.append("최종 전략 없음")
        return {
            'valid': False,
            'issues': issues,
            'warnings': warnings,
            'stats': {}
        }

    # 2. 인터벌별 전략 검증
    interval_stats = {}
    total_strategies = 0

    for interval in intervals:
        strategies = final_strategies.get(interval, [])
        strategy_count = len(strategies)
        total_strategies += strategy_count

        interval_stat = {
            'interval': interval,
            'strategy_count': strategy_count,
            'valid': strategy_count >= min_strategies_per_interval
        }

        if strategy_count == 0:
            issues.append(f"{interval}: 최종 전략 없음")
            interval_stat['valid'] = False
        elif strategy_count < min_strategies_per_interval:
            warnings.append(f"{interval}: 최종 전략 수 부족 ({strategy_count} < {min_strategies_per_interval})")

        # 전략 구조 검증 (샘플링)
        if strategies:
            sample = strategies[0] if len(strategies) > 0 else None
            if sample and not isinstance(sample, dict):
                warnings.append(f"{interval}: 전략이 dict 타입이 아님")

        interval_stats[interval] = interval_stat

    stats['interval_distribution'] = interval_stats
    stats['total_strategies'] = total_strategies
    stats['intervals_covered'] = len([i for i in intervals if final_strategies.get(i)])
    stats['intervals_expected'] = len(intervals)

    # 3. 커버리지 검증
    coverage_ratio = stats['intervals_covered'] / max(1, len(intervals))
    if coverage_ratio < 0.5:
        warnings.append(f"인터벌 커버리지 낮음 ({stats['intervals_covered']}/{len(intervals)})")

    # 4. 품질 점수 계산
    quality_score = 0.0

    # 커버리지 점수 (50점)
    quality_score += coverage_ratio * 50

    # 전략 수 점수 (50점) - 인터벌당 평균 20개 기준
    avg_strategies = total_strategies / max(1, len(intervals))
    strategy_score = min(avg_strategies / 20.0 * 50, 50)
    quality_score += strategy_score

    stats['quality_score'] = round(quality_score, 2)
    stats['avg_strategies_per_interval'] = round(avg_strategies, 2)

    # 5. 최종 검증
    valid = len(issues) == 0 and total_strategies > 0

    return {
        'valid': valid,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats['quality_score']
    }


class IntegratedPipelineOrchestrator:
    """통합된 파이프라인 오케스트레이터"""
    
    # 🔧 클래스 변수: 모델 캐시 (메모리 효율성 - 동일 모델은 한 번만 로드)
    _neural_policy_cache: Dict[str, Dict[str, Any]] = {}
    _cache_key: Optional[str] = None
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id  # 디버그 세션 ID 저장
        if NEW_PIPELINE_AVAILABLE:
            self.strategy_manager = None
            self.regime_router = RegimeRouter(session_id=session_id)
            self.integrated_analyzer = IntegratedAnalyzer(session_id=session_id)
        else:
            self.strategy_manager = None
            self.regime_router = None
            self.integrated_analyzer = None

        # 🔥 self-play 결과 저장소 (인터벌별 - 통합 학습에 사용)
        self._current_selfplay_result: Dict[str, Dict[str, Any]] = {}

        logger.info("🚀 통합된 파이프라인 오케스트레이터 초기화 완료")
    
    def run_complete_pipeline(self, coin: str, interval: str, candle_data: pd.DataFrame) -> PipelineResult:
        """완전한 파이프라인 실행"""
        try:
            start_time = datetime.now()
            logger.info(f"🚀 {coin}-{interval} 통합 파이프라인 시작")
            
            # 1단계: 전략 생성
            logger.info("1️⃣ 전략 생성 단계 시작")
            strategies = self._create_strategies(coin, interval, candle_data)
            logger.info(f"✅ {len(strategies)}개 전략 생성 완료")

            # 🧬 1-1단계: 기존 전략 진화 (유전 알고리즘)
            evolved_genetic_strategies = self._evolve_existing_strategies(coin, interval, strategies)
            if evolved_genetic_strategies:
                strategies.extend(evolved_genetic_strategies)
                logger.info(f"🧬 {len(evolved_genetic_strategies)}개 진화 전략 추가 (총 {len(strategies)}개)")

            # 2단계: Self-play 진화 + 실제 캔들 데이터 전달 🔥
            logger.info("2️⃣ Self-play 진화 단계 시작")
            evolved_strategies = self._evolve_strategies_with_selfplay(coin, strategies, interval, candle_data)
            logger.info(f"✅ Self-play 진화 완료: {len(evolved_strategies)}개 전략")
            
            # 3단계: 통합분석 (레짐 라우팅 제거, 전략을 직접 전달)
            logger.info("3️⃣ 통합분석 단계 시작")
            analysis_result = self._perform_integrated_analysis(coin, interval, evolved_strategies, candle_data)
            
            # 🔥 analysis_result는 dict로 반환되므로 dict 방식으로 접근
            if isinstance(analysis_result, dict):
                signal_action = analysis_result.get('signal_action', 'HOLD')
                signal_score = analysis_result.get('signal_score', analysis_result.get('final_signal_score', 0.0))
                logger.info(f"✅ 통합분석 완료: {signal_action} (점수: {signal_score:.3f})")
            else:
                # 객체인 경우 (하위 호환성)
                signal_action = getattr(analysis_result, 'signal_action', 'HOLD')
                signal_score = getattr(analysis_result, 'final_signal_score', getattr(analysis_result, 'signal_score', 0.0))
                logger.info(f"✅ 통합분석 완료: {signal_action} (점수: {signal_score:.3f})")
            
            # 🔥 3-1단계: 전략 등급 동적 업데이트 (레짐 라우팅 제거)
            try:
                from rl_pipeline.analysis.strategy_grade_updater import StrategyGradeUpdater
                grade_updater = StrategyGradeUpdater()
                
                # 통합 분석 결과 기반 등급 업데이트만 수행
                analysis_grade_updates = grade_updater.update_grades_from_analysis_results(
                    coin, interval, analysis_result, evolved_strategies
                )
                
                # 통합 업데이트 적용
                if analysis_grade_updates:
                    updated_count = grade_updater.apply_grade_updates(coin, interval, analysis_grade_updates, update_db=True)
                    logger.info(f"🔥 [{coin}-{interval}] 전략 등급 업데이트 완료: {updated_count}개 전략")
                else:
                    logger.debug(f"📊 [{coin}-{interval}] 등급 업데이트 대상 없음")
                    
            except Exception as e:
                logger.warning(f"⚠️ 전략 등급 업데이트 실패: {e}")
            
            # 🔥 결과 저장: 통합 분석만 저장
            try:
                if analysis_result:
                    # regime 안전하게 추출
                    try:
                        regime = getattr(analysis_result, 'regime', 'neutral')
                    except:
                        regime = 'neutral'
                    learning_results.save_integrated_analysis_results(coin, interval, regime, analysis_result)
                else:
                    logger.warning(f"⚠️ 분석 결과가 없어 저장 건너뜀: {coin}-{interval}")
            except Exception as e:
                logger.warning(f"⚠️ 통합 분석 결과 저장 실패: {e}")
            
            # 🆕 시그널 계산용 요약 데이터 저장 (rl_strategies.db)
            try:
                from rl_pipeline.db.learning_results import (
                    save_strategy_summary_for_signals,
                    save_dna_summary_for_signals,
                    save_global_strategy_summary_for_signals,
                    save_analysis_summary_for_signals
                )
                
                logger.info(f"📊 {coin}-{interval} 시그널 계산용 요약 데이터 저장 시작...")
                
                # 전략 요약 저장
                save_strategy_summary_for_signals(coin, interval)
                
                # DNA 요약 저장
                save_dna_summary_for_signals(coin, interval)
                
                # 분석 요약 저장
                save_analysis_summary_for_signals(coin, interval)
                
                # 글로벌 전략 요약 저장 (인터벌별)
                save_global_strategy_summary_for_signals(interval)
                
                logger.info(f"✅ {coin}-{interval} 시그널 계산용 요약 데이터 저장 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ 시그널 계산용 요약 데이터 저장 실패: {e}")
            
            # 🔥 5단계: 실시간 시그널 저장 (거래 시스템 연동) - 선택적
            # ⚠️ absolute_zero_system은 trading_system.db와 무관해야 하므로 비활성화
            # 활성화하려면 ENABLE_TRADING_SYSTEM_INTEGRATION=true 환경변수 설정
            enable_trading_integration = os.getenv('ENABLE_TRADING_SYSTEM_INTEGRATION', 'false').lower() == 'true'
            if enable_trading_integration:
                try:
                    from rl_pipeline.db.realtime_signal_storage import save_realtime_signal_from_analysis
                    
                    logger.info("5️⃣ 실시간 시그널 저장 단계 시작")
                    signal_saved = save_realtime_signal_from_analysis(
                        coin, interval, analysis_result, candle_data
                    )
                    
                    if signal_saved:
                        logger.info(f"✅ [{coin}-{interval}] 실시간 시그널 저장 완료 (거래 시스템 연동)")
                    else:
                        logger.warning(f"⚠️ [{coin}-{interval}] 실시간 시그널 저장 실패")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 실시간 시그널 저장 실패: {e}")
            else:
                logger.debug(f"📊 {coin}-{interval}: 거래 시스템 연동 비활성화 (ENABLE_TRADING_SYSTEM_INTEGRATION=false)")
            
            # 결과 생성
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 🔥 analysis_result가 dict인지 객체인지 확인하여 안전하게 접근
            if analysis_result:
                if isinstance(analysis_result, dict):
                    regime_detected = analysis_result.get('regime', 'neutral')
                    signal_score = analysis_result.get('signal_score', analysis_result.get('final_signal_score', 0.0))
                    signal_action = analysis_result.get('signal_action', 'HOLD')
                else:
                    regime_detected = getattr(analysis_result, 'regime', 'neutral')
                    signal_score = getattr(analysis_result, 'final_signal_score', getattr(analysis_result, 'signal_score', 0.0))
                    signal_action = getattr(analysis_result, 'signal_action', 'HOLD')
            else:
                regime_detected = 'neutral'
                signal_score = 0.0
                signal_action = 'HOLD'
            
            result = PipelineResult(
                coin=coin,
                interval=interval,
                strategies_created=len(strategies),
                selfplay_episodes=len(evolved_strategies),
                regime_detected=regime_detected,
                routing_results=0,  # 레짐 라우팅 제거됨
                signal_score=signal_score,
                signal_action=signal_action,
                execution_time=execution_time,
                status="success",
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"🎉 {coin}-{interval} 파이프라인 완료: {execution_time:.2f}초")
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ {coin}-{interval} 파이프라인 실패: {e}")
            logger.debug(f"파이프라인 실패 상세 정보:\n{error_details}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 실패한 단계 정보 기록
            failed_step = getattr(e, 'failed_step', 'unknown')
            logger.warning(f"⚠️ {coin}-{interval} 파이프라인 실패 단계: {failed_step}, 실행시간: {execution_time:.2f}초")
            
            return PipelineResult(
                coin=coin,
                interval=interval,
                strategies_created=0,
                selfplay_episodes=0,
                regime_detected="unknown",
                routing_results=0,
                signal_score=0.0,
                signal_action="HOLD",
                execution_time=execution_time,
                status="failed",
                created_at=datetime.now().isoformat()
            )
    
    def _create_strategies(self, coin: str, interval: str, candle_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """1단계: 전략 생성 (코인별 전략만)"""
        try:
            if not NEW_PIPELINE_AVAILABLE:
                logger.warning("⚠️ 새로운 모듈들이 사용 불가능하여 기본 전략 생성")
                return self._create_default_strategies(coin, interval)
            
            # 코인별 전략 생성만 수행 (글로벌 전략은 모든 시간대 완료 후에 생성)
            # create_coin_strategies 내부에서 이미 데이터 부족 시 create_basic_strategy()로 폴백 처리됨
            strategies_count = create_coin_strategies(coin, [interval], {(coin, interval): candle_data})
            
            logger.info(f"📊 코인별 전략 생성 완료: {strategies_count}개")
            
            # 🔥 DB 커밋 후 체크포인트 수행 (다른 커넥션이 즉시 읽을 수 있도록)
            try:
                from rl_pipeline.db.connection_pool import get_optimized_db_connection
                with get_optimized_db_connection("strategies") as conn:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    conn.commit()
                    logger.info("🔍 WAL 체크포인트 완료 (다른 커넥션에서 즉시 조회 가능)")
            except Exception as e:
                logger.warning(f"⚠️ 체크포인트 실패: {e}")
            
            # DB에서 생성된 전략 조회 (방금 생성한 전략 포함) - 공통 함수 사용
            try:
                from rl_pipeline.db.reads import load_strategies_pool
                from datetime import datetime

                # 🆕 증분 학습: 학습 완료되지 않은 전략만 로드
                # training_history에 없는 전략 = 학습 필요한 전략
                logger.info(f"📊 {coin}-{interval}: 미학습 전략 로드 시작 (증분 학습 모드)")

                # 미학습 전략 로드 (LEFT JOIN으로 training_history 없는 것만)
                from rl_pipeline.db.connection_pool import get_optimized_db_connection

                db_strategies = []
                try:
                    with get_optimized_db_connection("strategies") as conn:
                        cursor = conn.cursor()

                        # training_history에 없는 전략만 조회
                        query = """
                            SELECT cs.*
                            FROM coin_strategies cs
                            LEFT JOIN strategy_training_history sth ON cs.id = sth.strategy_id
                            WHERE cs.coin = ? AND cs.interval = ?
                              AND sth.strategy_id IS NULL
                            ORDER BY cs.created_at DESC
                            LIMIT 100
                        """

                        cursor.execute(query, (coin, interval))
                        rows = cursor.fetchall()

                        # 컬럼명 가져오기
                        columns = [desc[0] for desc in cursor.description]

                        # 딕셔너리 리스트로 변환
                        for row in rows:
                            strategy_dict = dict(zip(columns, row))
                            db_strategies.append(strategy_dict)

                except Exception as e:
                    logger.error(f"❌ 미학습 전략 로드 실패: {e}")
                    # Fallback: 기존 방식
                    db_strategies = load_strategies_pool(
                        coin=coin,
                        interval=interval,
                        limit=100,
                        order_by="created_at DESC, id DESC",
                        include_unknown=True
                    )

                logger.info(f"✅ {coin}-{interval}: {len(db_strategies)}개 미학습 전략 로드 완료")
                
                # 🔍 디버깅: 실제 조회된 행 수 확인
                logger.info(f"🔍 DB 쿼리 결과: {len(db_strategies)}개 행 조회됨")
                
                # 전체 전략 수 확인
                try:
                    from rl_pipeline.db.connection_pool import get_optimized_db_connection
                    with get_optimized_db_connection("strategies") as conn:
                        cursor = conn.cursor()
                        count_query = "SELECT COUNT(*) FROM coin_strategies WHERE coin = ? AND interval = ?"
                        cursor.execute(count_query, (coin, interval))
                        total_count = cursor.fetchone()[0]
                        logger.info(f"🔍 DB 전체 전략 수: {total_count}개")
                except Exception:
                    pass
                
                # 🔍 디버깅: 첫 5개 전략의 ID 출력
                if db_strategies:
                    logger.info(f"🔍 로드된 전략 샘플 (최대 5개):")
                    for i, s in enumerate(db_strategies[:5]):
                        logger.info(f"  [{i+1}] ID: {s.get('id', 'N/A')}, created_at: {s.get('created_at', 'N/A')}")
                
                if db_strategies:
                    logger.info(f"✅ DB에서 {len(db_strategies)}개 전략 로드 완료 (방금 생성한 전략 포함)")
                    
                    # 🆕 방향성 필터링 (선택적 사용 - 환경변수로 제어)
                    # 전략 다양성을 위해 필터링을 선택적으로만 사용
                    enable_filtering = os.getenv('ENABLE_STRATEGY_DIRECTION_FILTERING', 'false').lower() == 'true'
                    
                    if enable_filtering:
                        filtered_strategies = self._filter_strategies_by_direction(
                            db_strategies, coin, interval, candle_data
                        )
                        
                        if filtered_strategies and len(filtered_strategies) >= len(db_strategies) * 0.5:
                            # 필터링 후 50% 이상 남으면 사용
                            logger.info(f"✅ 방향성 필터링 완료: {len(db_strategies)}개 → {len(filtered_strategies)}개 (방향성 있는 전략만)")
                            return filtered_strategies
                        else:
                            # 필터링 후 전략이 부족하면 원본 사용 (모든 전략 테스트)
                            logger.info(f"📊 방향성 필터링 결과 부족 ({len(filtered_strategies) if filtered_strategies else 0}개), 모든 전략 사용 (다양성 확보)")
                            return db_strategies
                    else:
                        # 필터링 비활성화: 모든 전략 사용 (다양성 확보)
                        logger.info(f"📊 방향성 필터링 비활성화, 모든 {len(db_strategies)}개 전략 사용 (다양성 확보)")
                        return db_strategies
                else:
                    logger.warning(f"⚠️ DB 조회 결과 0개, 기본 전략 사용")
                    return self._create_default_strategies(coin, interval)
            except Exception as e:
                logger.warning(f"⚠️ DB 전략 로드 실패: {e}")
                return self._create_default_strategies(coin, interval)
            
        except Exception as e:
            logger.error(f"❌ 전략 생성 실패: {e}")
            return self._create_default_strategies(coin, interval)
    
    def _run_predictive_selfplay(self, coin: str, interval: str, strategies: List[Dict[str, Any]], candle_data: pd.DataFrame):
        """🔥 예측 Self-play 강화학습 루프 실행 (전략 생성 직후)

        강화학습 루프를 통해 예측 정확도를 향상시킵니다:
        1. 예측 생성 (방향, 확신도, horizon_k)
        2. 실제 결과 확인 (TP/SL 도달 시점, 수익률)
        3. 보상 계산 (방향 정확도, horizon_k 정확도, 수익률)
        4. 정책 업데이트 (예측 정확도가 높은 전략의 확신도 증가, horizon_k 최적화)
        5. 반복 (여러 에피소드)

        학습 목표:
        - 방향 예측 정확도 향상 (predicted_dir)
        - 최적 캔들 시점 찾기 (horizon_k) - 몇 번째 캔들에서 가장 높은 수익률인지
        - 신뢰도 향상 (predicted_conf)

        Returns:
            Dict: Self-play 결과 (cycle_results, episodes, avg_accuracy 포함)
        """
        try:
            if not strategies or len(strategies) == 0:
                logger.warning("⚠️ 예측 Self-play: 전략이 없어 건너뜀")
                return None

            if candle_data is None or len(candle_data) < 20:
                logger.warning("⚠️ 예측 Self-play: 캔들 데이터가 부족하여 건너뜀")
                return None

            logger.info(f"🔥 예측 Self-play 강화학습 시작: {coin}-{interval} ({len(strategies)}개 전략, {PREDICTIVE_SELFPLAY_EPISODES}개 에피소드)")

            # 예측 Self-play 모듈이 있으면 사용
            try:
                from rl_pipeline.simulation import PREDICTIVE_SELFPLAY_AVAILABLE, run_predictive_self_play_test

                if PREDICTIVE_SELFPLAY_AVAILABLE and run_predictive_self_play_test:
                    # 예측 Self-play 실행
                    logger.info("📊 예측 Self-play 모듈 사용")
                    # 전략 파라미터 추출
                    from rl_pipeline.db.reads import extract_strategy_params
                    strategy_params_list = [extract_strategy_params(strategy) for strategy in strategies[:100]]  # 최대 100개

                    predictive_result = run_predictive_self_play_test(
                        strategies=strategy_params_list,
                        candle_data=candle_data,
                        coin=coin,
                        interval=interval
                    )

                    if predictive_result:
                        logger.info(f"✅ 예측 Self-play 완료: {predictive_result.get('episodes', 0)}개 에피소드")
                        return predictive_result
                    else:
                        logger.warning("⚠️ 예측 Self-play 결과 없음")
                        return None
                else:
                    # 🔥 강화학습 루프 실행 (예측 Self-play 모듈이 없을 때)
                    logger.info("📊 예측 Self-play 모듈 없음, 강화학습 루프 모드 사용")
                    result = self._run_predictive_rl_loop(coin, interval, strategies, candle_data)
                    return result

            except ImportError:
                # 🔥 강화학습 루프 실행 (예측 Self-play 모듈이 없을 때)
                logger.info("📊 예측 Self-play 모듈 없음, 강화학습 루프 모드 사용")
                result = self._run_predictive_rl_loop(coin, interval, strategies, candle_data)
                return result

        except Exception as e:
            logger.error(f"❌ 예측 Self-play 실행 실패: {e}")
            logger.exception(e)
            return None
    
    def _run_predictive_rl_loop(self, coin: str, interval: str, strategies: List[Dict[str, Any]], candle_data: pd.DataFrame):
        """🔥 예측 Self-play 강화학습 루프 실행

        반복 학습을 통해 예측 정확도를 향상시킵니다:
        1. 예측 생성 (학습된 정책 사용) - 모든 전략에 대해
        2. 실제 결과 확인 (TP/SL 도달 시점, 수익률)
        3. 보상 계산
        4. 정책 업데이트
        5. 반복 (최대 PREDICTIVE_SELFPLAY_EPISODES번)

        구조: 50개 전략 × 50개 에피소드 = 총 2500번의 예측 생성/학습
        각 에피소드마다 모든 전략에 대해 예측 생성 → 결과 확인 → 정책 업데이트

        Returns:
            Dict: Self-play 결과 (cycle_results, episodes, avg_accuracy 포함)
        """
        try:
            # 전략별 예측 정책 초기화 (확신도, horizon_k)
            strategy_policies = {}
            for strategy in strategies[:100]:  # 최대 100개 전략
                strategy_id = strategy.get('id', 'unknown')
                strategy_policies[strategy_id] = {
                    'predicted_conf': 0.5,  # 초기 확신도
                    'horizon_k': 10,  # 초기 horizon_k
                    'direction': None,  # 전략 방향 (buy/sell/neutral)
                    'accuracy_history': [],  # 정확도 이력
                    'reward_history': [],  # 보상 이력
                    'opposite_direction_count': 0,  # 🔥 반대 방향 발생 횟수
                    'total_predictions': 0,  # 🔥 총 예측 횟수
                    'direction_reassessed': False  # 🔥 방향 재평가 여부
                }

            # 🔥 인터벌별로 다른 조기 종료 조건 적용
            interval_config = {
                '15m': {'min_episodes': 20, 'patience': 15, 'accuracy_threshold': 0.75},  # 🔥 개선: min_episodes 10→20, patience 5→15, threshold 0.85→0.75
                '30m': {'min_episodes': 25, 'patience': 18, 'accuracy_threshold': 0.70},  # 🔥 개선: min_episodes 15→25, patience 6→18, threshold 0.80→0.70
                '240m': {'min_episodes': 30, 'patience': 20, 'accuracy_threshold': 0.65},  # 🔥 개선: min_episodes 20→30, patience 8→20, threshold 0.70→0.65
                '1d': {'min_episodes': 35, 'patience': 25, 'accuracy_threshold': 0.60}  # 🔥 개선: min_episodes 25→35, patience 10→25, threshold 0.65→0.60
            }
            config = interval_config.get(interval, {'min_episodes': PREDICTIVE_SELFPLAY_MIN_EPISODES, 'patience': PREDICTIVE_SELFPLAY_EARLY_STOP_PATIENCE, 'accuracy_threshold': PREDICTIVE_SELFPLAY_EARLY_STOP_ACCURACY})

            min_episodes = config['min_episodes']
            patience = config['patience']
            accuracy_threshold = config['accuracy_threshold']

            logger.info(f"📊 예측 Self-play 구조: {len(strategies)}개 전략 × {PREDICTIVE_SELFPLAY_EPISODES}개 에피소드 = 최대 {len(strategies) * PREDICTIVE_SELFPLAY_EPISODES}번 학습")
            logger.info(f"📊 {interval} 조기 종료 설정: 최소 에피소드 {min_episodes}개, patience {patience}회, 정확도 임계값 {accuracy_threshold:.2%}")

            # 🔥 조기 종료 설정 로깅 (디버그용)
            try:
                from rl_pipeline.monitoring.simulation_debugger import SimulationDebugger
                early_stop_debugger = SimulationDebugger(session_id=None)  # 전역 로거
                early_stop_debugger.log({
                    'event': 'early_stop_config',
                    'coin': coin,
                    'interval': interval,
                    'min_episodes': min_episodes,
                    'patience': patience,
                    'accuracy_threshold': accuracy_threshold,
                    'max_episodes': PREDICTIVE_SELFPLAY_EPISODES,
                    'min_improvement': PREDICTIVE_SELFPLAY_MIN_IMPROVEMENT
                })
            except Exception as debug_err:
                logger.debug(f"⚠️ 조기 종료 설정 로깅 실패: {debug_err}")

            # 조기 종료를 위한 변수
            best_accuracy = 0.0
            no_improvement_count = 0
            accuracy_history = []  # 전체 평균 정확도 이력
            cycle_results = []  # 🔥 에피소드별 결과 저장

            # 강화학습 루프
            for episode in range(PREDICTIVE_SELFPLAY_EPISODES):
                logger.debug(f"📊 예측 Self-play 에피소드 {episode + 1}/{PREDICTIVE_SELFPLAY_EPISODES} ({len(strategies)}개 전략)")

                # 1. 예측 생성 (학습된 정책 사용) - 모든 전략에 대해
                predictions = self._create_predictions_with_policy(
                    coin, interval, strategies, candle_data, strategy_policies, episode
                )

                if not predictions:
                    logger.warning(f"⚠️ 에피소드 {episode + 1}: 예측 생성 실패")
                    continue

                # 2. 실제 결과 확인 (TP/SL 도달 시점, 수익률)
                results = self._check_prediction_results(
                    coin, interval, predictions, candle_data
                )

                if not results:
                    logger.warning(f"⚠️ 에피소드 {episode + 1}: 결과 확인 실패")
                    continue

                # 3. 보상 계산 및 정책 업데이트
                self._update_prediction_policy(
                    coin, interval, results, strategy_policies
                )

                # 현재 에피소드의 평균 정확도 계산
                # 🔥 최근 5개 에피소드의 평균 정확도 사용 (더 안정적인 측정)
                current_accuracy = np.mean([
                    np.mean(p['accuracy_history'][-5:]) if len(p['accuracy_history']) >= 5 else (np.mean(p['accuracy_history']) if p['accuracy_history'] else 0.0)
                    for p in strategy_policies.values()
                ])
                accuracy_history.append(current_accuracy)

                # 🔥 에피소드 결과 저장 (학습 데이터 수집을 위해 results 키 추가)
                # 예측 self-play는 전략별 예측 결과를 results에 포함

                # 🆕 results를 episode_id로 매핑하여 빠르게 조회 (actual 값 포함)
                results_by_episode_id = {r['episode_id']: r for r in results}

                episode_results = {}
                for strategy_id, policy in strategy_policies.items():
                    if strategy_id in [p.get('strategy_id') for p in predictions]:
                        # 전략별 예측 결과 수집
                        strategy_predictions = [p for p in predictions if p.get('strategy_id') == strategy_id]
                        if strategy_predictions:
                            # 🔥 예측 방향을 trades 형식으로 변환 (학습 시스템이 액션을 추출할 수 있도록)
                            trades = []
                            for pred in strategy_predictions:
                                predicted_dir = pred.get('predicted_dir', 0)
                                # predicted_dir: 1=BUY, -1=SELL, 0=HOLD
                                if predicted_dir == 1:
                                    direction = 'BUY'
                                elif predicted_dir == -1:
                                    direction = 'SELL'
                                else:
                                    direction = 'HOLD'

                                # 🆕 episode_id로 actual 값 조회
                                episode_id = pred.get('episode_id')
                                actual_result = results_by_episode_id.get(episode_id, {})

                                trades.append({
                                    'direction': direction,
                                    'entry_price': round(pred.get('entry_price', 0.0), 8),  # 가격 소숫점 8자리
                                    'predicted_conf': round(pred.get('predicted_conf', 0.5), 2),  # 소숫점 2자리
                                    'horizon_k': int(pred.get('horizon_k', 10)),  # 정수
                                    'target_move_pct': round(pred.get('target_move_pct', 0.02), 4),  # 소숫점 4자리
                                    # 🆕 실제 결과 추가 (학습용 레이블)
                                    'actual_move_pct': round(actual_result.get('actual_move_pct', 0.0), 4),  # 소숫점 4자리
                                    'actual_horizon': int(actual_result.get('actual_horizon', pred.get('horizon_k', 10))),  # 정수
                                    'actual_dir': actual_result.get('actual_dir', 0),
                                    'reward': round(actual_result.get('reward', 0.0), 4)  # 소숫점 4자리
                                })
                            
                            # 🔥 전략 방향 정보 추가 (매수/매도 전략 구분을 위해)
                            strategy_direction = policy.get('direction', 'neutral')  # 'buy', 'sell', 'neutral'
                            
                            # 예측 결과를 성과 데이터 형식으로 변환
                            episode_results[strategy_id] = {
                                'total_pnl': 0.0,  # 예측 self-play에서는 직접 계산 불가
                                'win_rate': policy.get('accuracy_history', [0.0])[-1] if policy.get('accuracy_history') else 0.0,
                                'total_trades': len(strategy_predictions),
                                'trades': trades,  # 🔥 예측 방향을 trades 형식으로 변환
                                'accuracy': policy.get('accuracy_history', [0.0])[-1] if policy.get('accuracy_history') else 0.0,
                                'predicted_conf': policy.get('predicted_conf', 0.5),
                                'horizon_k': policy.get('horizon_k', 10),
                                'strategy_direction': strategy_direction  # 🔥 전략 방향 추가 (매수/매도 구분)
                            }

                cycle_results.append({
                    'episode': episode + 1,
                    'accuracy': current_accuracy,
                    'best_accuracy': best_accuracy,
                    'predictions': len(predictions),
                    'results': episode_results  # 🔥 학습 데이터 수집을 위해 추가
                })

                # 🔥 조기 종료 체크 (최소 에피소드 수 확인)
                if PREDICTIVE_SELFPLAY_EARLY_STOP and (episode + 1) >= min_episodes:
                    # 정확도 임계값 달성 체크 (최소 에피소드 수 이후에만)
                    if current_accuracy >= accuracy_threshold:
                        logger.info(f"🎯 조기 종료: 정확도 임계값 달성 ({current_accuracy:.3f} >= {accuracy_threshold:.3f})")
                        logger.info(f"✅ 예측 Self-play 강화학습 완료 (에피소드 {episode + 1}/{PREDICTIVE_SELFPLAY_EPISODES}): 평균 정확도 {current_accuracy:.3f}")
                        return {
                            'cycle_results': cycle_results,
                            'episodes': episode + 1,
                            'avg_accuracy': current_accuracy,
                            'best_accuracy': best_accuracy,
                            'strategy_count': len(strategies)
                        }

                    # 개선도 체크 (최소 에피소드 수 이후에만)
                    improvement = current_accuracy - best_accuracy if best_accuracy > 0 else current_accuracy

                    if improvement >= PREDICTIVE_SELFPLAY_MIN_IMPROVEMENT:
                        # 개선됨
                        best_accuracy = current_accuracy
                        no_improvement_count = 0
                    else:
                        # 개선 없음
                        no_improvement_count += 1
                        if no_improvement_count >= patience:
                            logger.info(f"🛑 조기 종료: {patience}회 연속 개선 없음 (최고 정확도: {best_accuracy:.3f}, 현재: {current_accuracy:.3f})")
                            logger.info(f"✅ 예측 Self-play 강화학습 완료 (에피소드 {episode + 1}/{PREDICTIVE_SELFPLAY_EPISODES}): 평균 정확도 {current_accuracy:.3f}")
                            return {
                                'cycle_results': cycle_results,
                                'episodes': episode + 1,
                                'avg_accuracy': current_accuracy,
                                'best_accuracy': best_accuracy,
                                'strategy_count': len(strategies)
                            }
                elif PREDICTIVE_SELFPLAY_EARLY_STOP and (episode + 1) < min_episodes:
                    # 최소 에피소드 수 미만: 개선 추적만 수행 (조기 종료 안 함)
                    improvement = current_accuracy - best_accuracy if best_accuracy > 0 else current_accuracy
                    if improvement >= PREDICTIVE_SELFPLAY_MIN_IMPROVEMENT:
                        best_accuracy = current_accuracy
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                # 중간 로깅 (10 에피소드마다)
                if (episode + 1) % 10 == 0:
                    logger.info(f"📈 에피소드 {episode + 1}/{PREDICTIVE_SELFPLAY_EPISODES}: 평균 정확도 {current_accuracy:.3f} (최고: {best_accuracy:.3f}, 개선 없음: {no_improvement_count}회)")

            # 최종 로깅 (모든 에피소드 완료)
            final_avg_accuracy = np.mean([
                np.mean(p['accuracy_history']) if p['accuracy_history'] else 0.0
                for p in strategy_policies.values()
            ])
            logger.info(f"✅ 예측 Self-play 강화학습 완료 (모든 에피소드 완료): 평균 정확도 {final_avg_accuracy:.3f} (최고: {best_accuracy:.3f})")

            # 🔥 결과 반환
            return {
                'cycle_results': cycle_results,
                'episodes': len(cycle_results),
                'avg_accuracy': final_avg_accuracy,
                'best_accuracy': best_accuracy,
                'strategy_count': len(strategies)
            }

        except Exception as e:
            logger.error(f"❌ 예측 Self-play 강화학습 루프 실패: {e}")
            logger.exception(e)
            return None
    
    def _create_predictions_with_policy(
        self, 
        coin: str, 
        interval: str, 
        strategies: List[Dict[str, Any]], 
        candle_data: pd.DataFrame,
        strategy_policies: Dict[str, Dict[str, Any]],
        episode: int
    ) -> List[Dict[str, Any]]:
        """🔥 학습된 정책을 사용하여 예측 생성"""
        try:
            from rl_pipeline.db.rl_writes import save_episode_prediction
            from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
            from datetime import datetime
            import uuid

            # 🔥 캔들 데이터 정렬 (timestamp 기준 오름차순)
            candle_data_sorted = candle_data.copy()
            if 'timestamp' in candle_data_sorted.columns:
                candle_data_sorted = candle_data_sorted.sort_values('timestamp', ascending=True).reset_index(drop=True)

            # 🔥 예측 self-play: 과거 진입점 사용 (미래 캔들 확보)
            # 전체 데이터의 70% 지점을 진입점으로 사용하여, 30%의 미래 데이터로 TP/SL 시뮬레이션
            total_candles = len(candle_data_sorted)
            entry_position = int(total_candles * 0.7)  # 70% 지점

            # 진입점 기준 이전 100개 캔들 사용 (지표 계산용)
            start_idx = max(0, entry_position - 100)
            recent_candles = candle_data_sorted.iloc[start_idx:entry_position].copy()

            if len(recent_candles) < 10:
                logger.warning("⚠️ 예측 생성: 캔들 데이터 부족")
                return []

            # 미래 캔들 확인 (시뮬레이션용)
            future_candles_available = total_candles - entry_position
            if future_candles_available < 10:
                logger.warning(f"⚠️ 예측 생성: 미래 캔들 부족 ({future_candles_available}개)")
                return []

            logger.info(f"📊 예측 생성 준비: 전체 {total_candles}개 캔들, 진입점 {entry_position}, 미래 {future_candles_available}개")

            # 전략 방향 분류를 위한 분석기 생성 (루프 밖에서 한 번만)
            analyzer = IntegratedAnalyzer()
            
            predictions = []

            # 🔥 전략마다 다른 캔들 위치 사용 (다양한 시장 상황 학습)
            num_strategies_to_process = min(100, len(strategies))

            for strategy_idx, strategy in enumerate(strategies[:100]):  # 최대 100개 전략 처리
                try:
                    strategy_id = strategy.get('id', f"strategy_{uuid.uuid4().hex[:8]}")

                    # 🔥 각 전략마다 다른 캔들 위치 선택
                    # recent_candles 구간 내에서 균등 분산 (최대 50개 캔들 범위)
                    max_lookback = min(50, len(recent_candles) - 20)  # 최소 20개는 남겨둠
                    candle_offset = strategy_idx % max_lookback
                    candle_idx = -1 - candle_offset  # -1, -2, -3, ..., -50

                    # 정책 가져오기 (없으면 초기화)
                    policy = strategy_policies.get(strategy_id, {
                        'predicted_conf': 0.5,
                        'horizon_k': 10,
                        'direction': None
                    })
                    
                    # 🔥 해당 전략의 캔들 위치에서 가격 및 지표 추출
                    current_price = float(recent_candles['close'].iloc[candle_idx])
                    current_rsi = float(recent_candles['rsi'].iloc[candle_idx]) if 'rsi' in recent_candles.columns else 50.0
                    current_macd = float(recent_candles['macd'].iloc[candle_idx]) if 'macd' in recent_candles.columns else 0.0
                    current_macd_signal = float(recent_candles['macd_signal'].iloc[candle_idx]) if 'macd_signal' in recent_candles.columns else 0.0
                    current_volume_ratio = float(recent_candles['volume_ratio'].iloc[candle_idx]) if 'volume_ratio' in recent_candles.columns else 1.0

                    # 전략 방향성 분류 (한 번만 수행)
                    if policy['direction'] is None:
                        strategy_direction = analyzer._classify_strategy_direction(strategy)
                        policy['direction'] = strategy_direction

                    # 🔥 전략 방향에 따라 예측 방향 결정 (일관성 유지)
                    # 비슷한 전략은 같은 방향으로 예측하되, 시장 상황에 따라 확신도만 조정
                    # 통합 분석에서 확신도 기반 가중 평균으로 최종 시그널 결정
                    if policy['direction'] == 'buy':
                        predicted_dir = 1  # 매수 전략은 항상 BUY 예측 (일관성)
                    elif policy['direction'] == 'sell':
                        predicted_dir = -1  # 매도 전략은 항상 SELL 예측 (일관성)
                    else:
                        predicted_dir = 0  # 중립 전략은 HOLD
                    
                    # 전략 파라미터 추출
                    rsi_min = strategy.get('rsi_min', 30.0)
                    rsi_max = strategy.get('rsi_max', 70.0)
                    macd_buy_threshold = strategy.get('macd_buy_threshold', 0.0)
                    macd_sell_threshold = strategy.get('macd_sell_threshold', 0.0)
                    volume_ratio_min = strategy.get('volume_ratio_min', 1.0)
                    
                    # 🔥 시장 상황과 전략 조건 비교하여 확신도만 조정
                    # 예측 방향은 전략 방향에 따라 고정, 확신도만 시장 상황에 따라 조정
                    market_alignment_score = 0.0  # 시장 상황과 전략의 일치도 (0.0 ~ 1.0)
                    
                    if policy['direction'] == 'buy':
                        # 매수 전략: 시장 상황이 매수 조건에 맞는지 확인
                        rsi_ok = current_rsi <= rsi_max  # RSI가 전략의 최대값 이하
                        macd_ok = current_macd > macd_buy_threshold  # MACD가 매수 임계값 이상
                        volume_ok = current_volume_ratio >= volume_ratio_min  # 거래량 충분
                        
                        # 시장 상황 점수 계산
                        if rsi_ok:
                            market_alignment_score += 0.4
                        if macd_ok:
                            market_alignment_score += 0.4
                        if volume_ok:
                            market_alignment_score += 0.2
                        
                        # 과매수 구간이면 확신도 크게 감소
                        if current_rsi > 70:
                            market_alignment_score *= 0.3  # 과매수 구간은 일치도 크게 감소
                            
                    elif policy['direction'] == 'sell':
                        # 매도 전략: 시장 상황이 매도 조건에 맞는지 확인
                        rsi_ok = current_rsi >= rsi_min  # RSI가 전략의 최소값 이상
                        macd_ok = current_macd < macd_sell_threshold  # MACD가 매도 임계값 이하
                        volume_ok = current_volume_ratio >= volume_ratio_min  # 거래량 충분
                        
                        # 시장 상황 점수 계산
                        if rsi_ok:
                            market_alignment_score += 0.4
                        if macd_ok:
                            market_alignment_score += 0.4
                        if volume_ok:
                            market_alignment_score += 0.2
                        
                        # 과매도 구간이면 확신도 크게 감소
                        if current_rsi < 30:
                            market_alignment_score *= 0.3  # 과매도 구간은 일치도 크게 감소
                    
                    elif policy['direction'] == 'neutral':
                        # 중립 전략: RSI가 중립 구간(30~70)일 때 높은 일치도
                        if 30 <= current_rsi <= 70:
                            market_alignment_score = 0.7
                        else:
                            market_alignment_score = 0.3
                    
                    # 🔥 학습된 확신도에 시장 상황 일치도를 반영
                    # 시장 상황이 전략 조건과 일치할수록 확신도 증가, 불일치하면 감소
                    # 통합 분석에서 확신도가 높은 전략의 예측이 더 큰 가중치를 받음
                    base_conf = max(0.1, min(1.0, policy['predicted_conf']))
                    predicted_conf = base_conf * (0.3 + 0.7 * market_alignment_score)  # 최소 30% 확신도 유지
                    predicted_conf = round(max(0.1, min(1.0, predicted_conf)), 2)  # 소숫점 2자리
                    
                    # 🔥 학습된 horizon_k 사용 (정책에서 가져옴)
                    horizon_k = max(1, int(policy['horizon_k']))

                    # 목표 변동률 설정
                    target_move_pct = round(0.02, 4)  # 목표 변동률 2% (소숫점 4자리)

                    # 🔥 진입 시점: 해당 전략의 캔들 위치 타임스탬프 사용
                    # 각 전략마다 다른 시점에서 예측 → 다양한 시장 상황 학습
                    if 'timestamp' in recent_candles.columns:
                        ts_value = recent_candles['timestamp'].iloc[candle_idx]
                        if isinstance(ts_value, pd.Timestamp):
                            ts_entry = int(ts_value.timestamp())  # pandas.Timestamp → Unix 타임스탬프
                        else:
                            ts_entry = int(ts_value)
                    else:
                        ts_entry = int(datetime.now().timestamp())

                    # 예측 저장
                    episode_id = f"pred_{coin}_{interval}_{strategy_id}_{episode}_{ts_entry}"
                    
                    save_episode_prediction(
                        episode_id=episode_id,
                        coin=coin,
                        interval=interval,
                        strategy_id=strategy_id,
                        state_key=f"{coin}_{interval}_{ts_entry}",
                        predicted_dir=predicted_dir,
                        predicted_conf=predicted_conf,
                        entry_price=current_price,
                        target_move_pct=target_move_pct,
                        horizon_k=horizon_k,
                        ts_entry=ts_entry
                    )
                    
                    predictions.append({
                        'episode_id': episode_id,
                        'strategy_id': strategy_id,
                        'predicted_dir': predicted_dir,
                        'predicted_conf': round(predicted_conf, 2),  # 소숫점 2자리
                        'horizon_k': int(horizon_k),  # 정수
                        'entry_price': round(current_price, 8),  # 가격 소숫점 8자리
                        'target_move_pct': round(target_move_pct, 4),  # 소숫점 4자리
                        'ts_entry': ts_entry
                    })
                    
                except Exception as e:
                    logger.debug(f"⚠️ 전략 {strategy.get('id', 'unknown')} 예측 생성 실패: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ 예측 생성 실패: {e}")
            return []
    
    def _check_prediction_results(
        self,
        coin: str,
        interval: str,
        predictions: List[Dict[str, Any]],
        candle_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """🔥 예측 결과 확인 (실제 TP/SL 도달 시점, 수익률 계산)
        
        실제 캔들 데이터에서 TP/SL 도달 시점을 찾고 수익률을 계산합니다.
        """
        try:
            from rl_pipeline.engine.reward_engine import RewardEngine
            from rl_pipeline.db.rl_writes import save_episode_summary
            
            reward_engine = RewardEngine()
            results = []
            
            # 인터벌에 따른 캔들 시간 계산 (초 단위)
            interval_seconds = {
                '15m': 15 * 60,
                '30m': 30 * 60,
                '240m': 240 * 60,
                '1d': 24 * 60 * 60
            }
            candle_seconds = interval_seconds.get(interval, 15 * 60)
            
            for pred in predictions:
                try:
                    episode_id = pred['episode_id']
                    strategy_id = pred['strategy_id']
                    predicted_dir = pred['predicted_dir']
                    predicted_conf = pred['predicted_conf']
                    horizon_k = pred['horizon_k']
                    entry_price = pred['entry_price']
                    target_move_pct = pred['target_move_pct']
                    ts_entry = pred['ts_entry']
                    
                    # 진입 시점의 캔들 인덱스 찾기
                    # 🔥 candle_data를 타임스탬프 기준으로 정렬하고 인덱스 리셋
                    candle_data_sorted = candle_data.copy()
                    if 'timestamp' in candle_data_sorted.columns:
                        # timestamp는 이미 datetime64[ns]로 변환되어 있음 (candle_loader.py에서 unit='s' 사용)
                        candle_data_sorted = candle_data_sorted.sort_values('timestamp').reset_index(drop=True)
                    
                    entry_idx = None
                    # 가장 가까운 타임스탬프 찾기
                    if 'timestamp' in candle_data_sorted.columns:
                        for idx in range(len(candle_data_sorted)):
                            row = candle_data_sorted.iloc[idx]
                            try:
                                candle_ts = int(pd.Timestamp(row['timestamp']).timestamp())
                                if abs(candle_ts - ts_entry) < candle_seconds * 2:  # 2배 여유 (인덱스 오차 고려)
                                    entry_idx = idx
                                    break
                            except Exception:
                                continue
                    
                    if entry_idx is None:
                        # 진입 시점을 찾을 수 없으면 스킵 (데이터 불일치)
                        logger.warning(f"⚠️ 진입 시점을 찾을 수 없음: {episode_id} (ts_entry={ts_entry})")
                        continue
                    
                    # TP/SL 계산 (전략에서 가져오거나 기본값 사용)
                    tp_pct = target_move_pct  # TP = 목표 변동률
                    sl_pct = -target_move_pct * 0.5  # SL = TP의 50% (기본값)
                    
                    # 실제 결과 확인 (horizon_k 범위 내에서)
                    actual_horizon = None
                    actual_move_pct = 0.0
                    first_event = 'expiry'
                    max_profit_pct = 0.0
                    max_profit_horizon = 0
                    
                    # horizon_k 범위 내에서 최대 수익률과 그 시점 찾기
                    # 🔥 candle_data_sorted 사용 (정렬된 데이터)
                    for k in range(1, min(horizon_k + 10, len(candle_data_sorted) - entry_idx)):
                        if entry_idx + k >= len(candle_data_sorted):
                            break
                        
                        current_candle = candle_data_sorted.iloc[entry_idx + k]
                        current_price = float(current_candle['close'])
                        
                        # 수익률 계산
                        if predicted_dir == 1:  # 상승 예측
                            move_pct = (current_price - entry_price) / entry_price
                        elif predicted_dir == -1:  # 하락 예측
                            move_pct = (entry_price - current_price) / entry_price
                        else:  # 중립
                            move_pct = abs(current_price - entry_price) / entry_price
                        
                        # 최대 수익률 추적
                        if move_pct > max_profit_pct:
                            max_profit_pct = move_pct
                            max_profit_horizon = k
                        
                        # TP/SL 도달 확인
                        if predicted_dir == 1:  # 상승 예측
                            if move_pct >= tp_pct:
                                first_event = 'TP'
                                actual_horizon = k
                                actual_move_pct = move_pct
                                break
                            elif move_pct <= sl_pct:
                                first_event = 'SL'
                                actual_horizon = k
                                actual_move_pct = move_pct
                                break
                        elif predicted_dir == -1:  # 하락 예측
                            if move_pct >= tp_pct:
                                first_event = 'TP'
                                actual_horizon = k
                                actual_move_pct = move_pct
                                break
                            elif move_pct <= sl_pct:
                                first_event = 'SL'
                                actual_horizon = k
                                actual_move_pct = move_pct
                                break
                    
                    # 만료 시 최대 수익률 사용
                    if first_event == 'expiry':
                        actual_horizon = horizon_k
                        actual_move_pct = max_profit_pct
                    
                    # 실제 방향 계산
                    actual_dir = 1 if actual_move_pct > 0.001 else (-1 if actual_move_pct < -0.001 else 0)
                    
                    # 보상 계산
                    reward_components = reward_engine.compute_reward(
                        predicted_dir=predicted_dir,
                        predicted_target=target_move_pct,
                        predicted_horizon=horizon_k,
                        actual_dir=actual_dir,
                        actual_move_pct=actual_move_pct,
                        actual_horizon=actual_horizon or horizon_k,
                        first_event=first_event,
                        interval=interval
                    )
                    
                    # 예측 정확도 플래그
                    acc_flag = reward_engine.compute_predictive_accuracy_flag(
                        first_event=first_event,
                        predicted_dir=predicted_dir,
                        actual_move_pct=actual_move_pct
                    )
                    
                    # 결과 저장
                    ts_exit = ts_entry + (actual_horizon or horizon_k) * candle_seconds
                    
                    save_episode_summary(
                        episode_id=episode_id,
                        ts_exit=ts_exit,
                        first_event=first_event,
                        t_hit=actual_horizon or horizon_k,
                        realized_ret_signed=actual_move_pct,
                        total_reward=reward_components.reward_total,
                        acc_flag=acc_flag,
                        coin=coin,
                        interval=interval,
                        strategy_id=strategy_id,
                        source_type='predictive'
                    )
                    
                    results.append({
                        'episode_id': episode_id,
                        'strategy_id': strategy_id,
                        'predicted_dir': predicted_dir,
                        'predicted_conf': round(predicted_conf, 2),  # 소숫점 2자리
                        'horizon_k': int(horizon_k),  # 정수
                        'actual_dir': actual_dir,
                        'actual_move_pct': round(actual_move_pct, 4),  # 소숫점 4자리
                        'actual_horizon': int(actual_horizon or horizon_k),  # 정수
                        'max_profit_pct': round(max_profit_pct, 4),  # 소숫점 4자리
                        'max_profit_horizon': int(max_profit_horizon),  # 정수
                        'first_event': first_event,
                        'reward': round(reward_components.reward_total, 4),  # 소숫점 4자리
                        'acc_flag': acc_flag
                    })
                    
                except Exception as e:
                    logger.debug(f"⚠️ 예측 결과 확인 실패: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 예측 결과 확인 실패: {e}")
            return []
    
    def _update_prediction_policy(
        self,
        coin: str,
        interval: str,
        results: List[Dict[str, Any]],
        strategy_policies: Dict[str, Dict[str, Any]]
    ):
        """🔥 예측 정책 업데이트 (강화학습)
        
        예측 정확도가 높은 전략의 확신도 증가, horizon_k 최적화
        """
        try:
            learning_rate = PREDICTIVE_SELFPLAY_LEARNING_RATE
            
            for result in results:
                strategy_id = result['strategy_id']
                
                if strategy_id not in strategy_policies:
                    continue
                
                policy = strategy_policies[strategy_id]
                
                # 정확도 및 보상 추출
                acc_flag = result['acc_flag']
                reward = result['reward']
                max_profit_pct = result['max_profit_pct']
                max_profit_horizon = result['max_profit_horizon']
                predicted_horizon = result['horizon_k']
                actual_horizon = result['actual_horizon']
                predicted_dir = result['predicted_dir']
                actual_dir = result['actual_dir']
                
                # 🔥 반대 방향 발생 추적
                policy['total_predictions'] += 1
                is_opposite_direction = False
                if predicted_dir == 1 and actual_dir == -1:  # 상승 예측했는데 하락
                    is_opposite_direction = True
                    policy['opposite_direction_count'] += 1
                elif predicted_dir == -1 and actual_dir == 1:  # 하락 예측했는데 상승
                    is_opposite_direction = True
                    policy['opposite_direction_count'] += 1
                
                # 정확도 이력 업데이트
                policy['accuracy_history'].append(acc_flag)
                policy['reward_history'].append(reward)
                
                # 최근 10개만 유지
                if len(policy['accuracy_history']) > 10:
                    policy['accuracy_history'] = policy['accuracy_history'][-10:]
                if len(policy['reward_history']) > 10:
                    policy['reward_history'] = policy['reward_history'][-10:]
                
                # 🔥 반대 방향 발생 빈도 계산 및 전략 방향 재평가
                if policy['total_predictions'] >= 5:  # 최소 5회 예측 후 재평가 가능
                    opposite_rate = policy['opposite_direction_count'] / policy['total_predictions']
                    
                    # 반대 방향 발생 빈도가 60% 이상이면 전략 방향 재평가
                    if opposite_rate >= 0.6 and not policy['direction_reassessed']:
                        # 🔥 전략 방향 재평가: 실제 성과 기반으로 재분류
                        original_direction = policy['direction']
                        new_direction = self._reassess_strategy_direction(
                            coin, interval, strategy_id, predicted_dir, actual_dir, 
                            policy['accuracy_history'], policy['reward_history']
                        )
                        
                        if new_direction != original_direction:
                            logger.warning(f"🔄 {coin}-{interval} 전략 {strategy_id} 방향 재평가: "
                                         f"{original_direction} → {new_direction} "
                                         f"(반대 방향 발생률: {opposite_rate:.1%})")
                            policy['direction'] = new_direction
                            policy['direction_reassessed'] = True
                            # 방향 재평가 후 카운터 리셋
                            policy['opposite_direction_count'] = 0
                            policy['total_predictions'] = 0
                
                # 🔥 확신도 업데이트 (정확도가 높을수록 증가)
                if acc_flag == 1:
                    # 예측 정확: 확신도 증가
                    policy['predicted_conf'] = round(min(1.0, policy['predicted_conf'] + learning_rate * 0.1), 2)  # 소숫점 2자리
                else:
                    # 예측 부정확: 확신도 감소
                    # 🔥 반대 방향이면 더 큰 페널티
                    penalty = learning_rate * (0.1 if is_opposite_direction else 0.05)
                    policy['predicted_conf'] = round(max(0.1, policy['predicted_conf'] - penalty), 2)  # 소숫점 2자리
                
                # 🔥 horizon_k 최적화 (최대 수익률이 발생한 시점으로 업데이트)
                if max_profit_horizon > 0 and max_profit_pct > 0.01:  # 최소 1% 수익률
                    # 최대 수익률 시점으로 horizon_k 업데이트 (지수 이동 평균)
                    policy['horizon_k'] = int(
                        policy['horizon_k'] * (1 - learning_rate) + 
                        max_profit_horizon * learning_rate
                    )
                    policy['horizon_k'] = max(1, min(50, policy['horizon_k']))  # 1~50 범위 제한
                
                # 보상 기반 추가 업데이트
                if reward > 0.5:
                    # 높은 보상: 확신도 추가 증가
                    policy['predicted_conf'] = round(min(1.0, policy['predicted_conf'] + learning_rate * 0.05), 2)  # 소숫점 2자리
                elif reward < -0.5:
                    # 낮은 보상: 확신도 추가 감소
                    policy['predicted_conf'] = round(max(0.1, policy['predicted_conf'] - learning_rate * 0.1), 2)  # 소숫점 2자리
            
        except Exception as e:
            logger.error(f"❌ 예측 정책 업데이트 실패: {e}")
    
    def _get_opposite_direction(self, direction: int, fallback_direction: int = 1) -> str:
        """방향의 반대 방향 반환 (neutral 금지)

        Args:
            direction: 원래 방향 (1=buy, -1=sell, 0=neutral)
            fallback_direction: direction이 0일 때 사용할 기본 방향

        Returns:
            'buy' 또는 'sell' (neutral 반환 안 함)
        """
        if direction == 1:
            return 'sell'
        elif direction == -1:
            return 'buy'
        else:
            # neutral이었다면 fallback 방향의 반대
            return 'sell' if fallback_direction == 1 else 'buy'

    def _reassess_strategy_direction(
        self,
        coin: str,
        interval: str,
        strategy_id: str,
        predicted_dir: int,
        actual_dir: int,
        accuracy_history: List[int],
        reward_history: List[float]
    ) -> str:
        """🔥 전략 방향 재평가 (실제 성과 기반)

        반대 방향이 자주 발생하면 실제 성과를 기반으로 방향을 재평가합니다.

        ⚠️ 핵심 철학: 재평가 후에도 반드시 예측을 계속해야 합니다 (neutral 금지).
        - 예측이 틀리든 맞든 → 계속 예측
        - 실제로 어떤 방향으로 얼마나 움직였는지 모두 저장
        - 그 패턴을 학습
        - 학습된 결과를 사용

        Args:
            coin: 코인 심볼
            interval: 인터벌
            strategy_id: 전략 ID
            predicted_dir: 예측 방향 (1/-1/0)
            actual_dir: 실제 방향 (1/-1/0)
            accuracy_history: 정확도 이력
            reward_history: 보상 이력

        Returns:
            재평가된 전략 방향 ('buy' 또는 'sell', neutral 절대 반환 안 함)
        """
        try:
            # 실제 성과 기반 방향 결정
            avg_accuracy = np.mean(accuracy_history) if accuracy_history else 0.0
            avg_reward = np.mean(reward_history) if reward_history else 0.0

            # 🔥 전략: 반대 방향이 자주 발생 → 실제 방향으로 변경
            # 예: BUY 예측이 자주 실패 → 실제로는 SELL 전략일 수 있음

            # 정확도나 보상이 낮으면 → 반대 방향으로 전환 (계속 예측)
            should_flip = (
                avg_accuracy < 0.4 or  # 정확도 40% 미만
                avg_reward < -0.1       # 보상 음수
            )

            if should_flip:
                # 반대 방향으로 변경하여 계속 예측
                return self._get_opposite_direction(predicted_dir, actual_dir)

            # 성과가 애매하면 → 일단 반대 방향으로 시도
            # (어차피 학습을 위해 계속 예측해야 함)
            return self._get_opposite_direction(predicted_dir, actual_dir)

        except Exception as e:
            logger.debug(f"⚠️ 전략 방향 재평가 실패: {e}")
            # 에러 시에도 neutral 금지 → 반대 방향으로
            return self._get_opposite_direction(predicted_dir, actual_dir)
    
    def _create_predictions_for_strategies(self, coin: str, interval: str, strategies: List[Dict[str, Any]], candle_data: pd.DataFrame):
        """🔥 전략에 대한 예측 생성 및 저장 (매수/매도 전략 구분)
        
        매수 전략: predicted_dir = +1 (상승 예측)
        매도 전략: predicted_dir = -1 (하락 예측)
        중립 전략: predicted_dir = 0 (중립 예측)
        
        ⚠️ 이 메서드는 하위 호환성을 위해 유지되지만, 강화학습 루프에서는 사용하지 않습니다.
        """
        try:
            from rl_pipeline.db.rl_writes import save_episode_prediction
            from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
            from datetime import datetime
            import uuid
            
            # 최근 캔들 데이터 사용 (예측 생성용)
            recent_candles = candle_data.tail(min(100, len(candle_data)))
            if len(recent_candles) < 10:
                logger.warning("⚠️ 예측 생성: 캔들 데이터 부족")
                return
            
            # 현재 가격 및 지표 계산
            current_price = float(recent_candles['close'].iloc[-1])
            
            # 전략 방향 분류를 위한 분석기 생성
            analyzer = IntegratedAnalyzer()
            
            # 전략별 예측 생성
            buy_predictions = 0
            sell_predictions = 0
            neutral_predictions = 0
            
            for strategy in strategies[:100]:  # 최대 100개 전략 처리
                try:
                    strategy_id = strategy.get('id', f"strategy_{uuid.uuid4().hex[:8]}")
                    
                    # 🔥 전략 방향성 분류 (매수/매도/중립)
                    strategy_direction = analyzer._classify_strategy_direction(strategy)
                    
                    # 🔥 전략 방향에 따라 예측 방향 결정
                    if strategy_direction == 'buy':
                        # 매수 전략 → 상승 예측 (+1)
                        predicted_dir = 1
                        buy_predictions += 1
                    elif strategy_direction == 'sell':
                        # 매도 전략 → 하락 예측 (-1)
                        predicted_dir = -1
                        sell_predictions += 1
                    else:
                        # 중립 전략 → 중립 예측 (0)
                        predicted_dir = 0
                        neutral_predictions += 1
                    
                    # 확신도 계산 (전략 파라미터 기반)
                    # RSI, MACD 등 지표를 기반으로 확신도 계산 가능
                    predicted_conf = 0.5  # 기본 확신도 (향후 개선 가능)
                    
                    # 목표 변동률 및 목표 캔들 수 설정
                    target_move_pct = 0.02  # 목표 변동률 2%
                    horizon_k = 10  # 목표 캔들 수 (인터벌에 따라 조정 가능)
                    
                    # 예측 저장
                    episode_id = f"pred_{coin}_{interval}_{strategy_id}_{int(datetime.now().timestamp())}"
                    ts_entry = int(datetime.now().timestamp())
                    
                    save_episode_prediction(
                        episode_id=episode_id,
                        coin=coin,
                        interval=interval,
                        strategy_id=strategy_id,
                        state_key=f"{coin}_{interval}_{ts_entry}",
                        predicted_dir=predicted_dir,
                        predicted_conf=predicted_conf,
                        entry_price=current_price,
                        target_move_pct=target_move_pct,
                        horizon_k=horizon_k,
                        ts_entry=ts_entry
                    )
                    
                except Exception as e:
                    logger.debug(f"⚠️ 전략 {strategy.get('id', 'unknown')} 예측 생성 실패: {e}")
                    continue
            
            logger.info(f"✅ 예측 생성 완료: 총 {buy_predictions + sell_predictions + neutral_predictions}개 (매수: {buy_predictions}, 매도: {sell_predictions}, 중립: {neutral_predictions})")
            
        except Exception as e:
            logger.error(f"❌ 예측 생성 실패: {e}")
    
    def _evolve_strategies_with_selfplay(self, coin: str, strategies: List[Dict[str, Any]], interval: str = None, candle_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """2단계: Self-play 진화 + 실제 캔들 데이터 사용 🔥"""
        try:
            if not strategies:
                logger.warning("⚠️ 진화할 전략이 없습니다")
                return []
            
            # 모든 전략을 Self-play에 사용 (100% 활용률)
            top_strategies = strategies  # 모든 전략 사용
            
            # 전략 파라미터 추출 - 공통 함수 사용
            from rl_pipeline.db.reads import extract_strategy_params
            strategy_params_list = [extract_strategy_params(strategy) for strategy in top_strategies]
            
            # Self-play 실행 - 전체 전략 풀에서 매 에피소드마다 샘플링
            from rl_pipeline.simulation.selfplay import run_self_play_test
            
            # DB에서 모든 전략 로드 (더 큰 풀 사용) - 공통 함수 사용
            all_strategies_pool = []
            try:
                from rl_pipeline.db.reads import load_strategies_pool, extract_strategy_params
                
                # interval 필터 추가하여 같은 interval 전략만 로드
                # 🔥 UNKNOWN 등급 전략도 포함 (include_unknown=True)
                db_strategies = load_strategies_pool(
                    coin=coin,
                    interval=interval,
                    limit=AZ_STRATEGY_POOL_SIZE,  # 0이면 제한 없음
                    order_by="id DESC",
                    include_unknown=True  # 🔥 UNKNOWN 등급 포함
                )
                
                if interval:
                    logger.info(f"📊 DB에서 {coin}-{interval} 전략 로드 중... (UNKNOWN 등급 포함, 최대 {AZ_STRATEGY_POOL_SIZE}개)")
                else:
                    logger.info(f"📊 DB에서 {coin} (모든 interval) 전략 로드 중... (UNKNOWN 등급 포함)")
                
                # 전략 파라미터 추출
                all_strategies_pool = [extract_strategy_params(s) for s in db_strategies]
                
                # 🔍 첫 3개만 상세 로그
                for i, params in enumerate(all_strategies_pool[:3]):
                    logger.info(f"  전략 {i+1}: RSI={params['rsi_min']:.1f}-{params['rsi_max']:.1f}, "
                               f"StopLoss={params['stop_loss_pct']:.3f}, TakeProfit={params['take_profit_pct']:.3f}")
                
                if interval:
                    logger.info(f"✅ DB에서 {len(all_strategies_pool)}개 {coin}-{interval} 전략 로드 완료")
                else:
                    logger.info(f"✅ DB에서 {len(all_strategies_pool)}개 {coin} 전략 로드 완료 (모든 interval)")
            except Exception as e:
                logger.warning(f"⚠️ 전체 전략 풀 로드 실패: {e}, 기본 전략만 사용")
                all_strategies_pool = strategy_params_list
            
            # 🔥 동적 에피소드 수 계산: 모든 전략이 self-play를 실행할 수 있도록
            agents_per_episode = AZ_SELFPLAY_AGENTS_PER_EPISODE
            total_strategies = len(all_strategies_pool) if all_strategies_pool else len(strategy_params_list)
            
            # 최소 에피소드 수: 모든 전략이 최소 1번씩 실행될 수 있도록
            # 중복 허용을 고려하여 여유있게 설정
            min_episodes_for_all = max(1, int(total_strategies / agents_per_episode * 1.2))  # 20% 여유
            # 기본 에피소드 수와 비교하여 더 큰 값 사용
            dynamic_episodes = max(AZ_SELFPLAY_EPISODES, min_episodes_for_all)
            
            if dynamic_episodes > AZ_SELFPLAY_EPISODES:
                logger.info(f"📈 에피소드 수 동적 조정: {AZ_SELFPLAY_EPISODES} → {dynamic_episodes}개 "
                           f"(전략 수: {total_strategies}개, 에피소드당 {agents_per_episode}개)")
            else:
                logger.info(f"📊 에피소드 수: {dynamic_episodes}개 (기본값 사용, 전략 수: {total_strategies}개)")
            
            # 🆕 하이브리드 모드 체크
            use_hybrid = os.getenv('USE_HYBRID', 'false').lower() == 'true'
            hybrid_config = None
            neural_policy = None
            
            if use_hybrid:
                try:
                    # 하이브리드 설정 로드
                    config_path = os.getenv('HYBRID_CONFIG_PATH', '/workspace/rl_pipeline/hybrid/config_hybrid.json')
                    if os.path.exists(config_path):
                        import json
                        with open(config_path, 'r') as f:
                            hybrid_config = json.load(f)
                        
                        # 모델 로드 (가장 최신 모델 또는 지정된 모델)
                        model_id = os.getenv('HYBRID_MODEL_ID', 'latest')
                        cache_key = None
                        ckpt_path = None
                        
                        if model_id == 'latest':
                            # 최신 모델 찾기
                            from rl_pipeline.db.connection_pool import get_strategy_db_pool
                            with get_strategy_db_pool().get_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute("""
                                    SELECT model_id, ckpt_path FROM policy_models 
                                    ORDER BY created_at DESC LIMIT 1
                                """)
                                result = cursor.fetchone()
                                if result:
                                    model_id = result[0]
                                    ckpt_path = result[1]
                                    cache_key = f"{model_id}_{ckpt_path}"
                                else:
                                    # 🔧 처음 실행 시 모델이 없는 것은 정상 (자동 학습 후 생성됨)
                                    logger.info("ℹ️ 학습된 모델이 아직 없습니다 (처음 실행 또는 자동 학습 대기 중)")
                                    logger.info("💡 자동 학습이 활성화되어 있으면 Self-play 후 모델이 생성되고, 다음 실행부터 하이브리드 모드가 자동 활성화됩니다.")
                                    logger.info("📊 현재는 규칙 기반 모드로 실행합니다 (정상 동작)")
                                    use_hybrid = False
                                    model_id = None
                        else:
                            # 지정된 모델 로드
                            checkpoint_dir = hybrid_config.get('paths', {}).get('checkpoints', '/workspace/rl_pipeline/artifacts/checkpoints')
                            ckpt_path = os.path.join(checkpoint_dir, f"{model_id}.ckpt")
                            cache_key = f"{model_id}_{ckpt_path}"
                            if not os.path.exists(ckpt_path):
                                logger.warning(f"⚠️ 모델 파일을 찾을 수 없습니다: {ckpt_path}")
                                use_hybrid = False
                                model_id = None
                        
                        # 🔧 모델 캐싱: 같은 모델은 한 번만 로드
                        if model_id and cache_key:
                            if cache_key == IntegratedPipelineOrchestrator._cache_key:
                                # 캐시된 모델 재사용
                                neural_policy = IntegratedPipelineOrchestrator._neural_policy_cache.get(cache_key)
                                if neural_policy:
                                    logger.debug(f"♻️ 하이브리드 모델 캐시 재사용: {model_id}")
                                else:
                                    # 캐시가 비어있으면 로드
                                    from rl_pipeline.hybrid.neural_policy_jax import load_ckpt
                                    neural_policy = load_ckpt(ckpt_path)
                                    IntegratedPipelineOrchestrator._neural_policy_cache[cache_key] = neural_policy
                                    IntegratedPipelineOrchestrator._cache_key = cache_key
                                    logger.info(f"✅ 하이브리드 모델 로드 및 캐시: {model_id}")
                            else:
                                # 새로운 모델 로드
                                from rl_pipeline.hybrid.neural_policy_jax import load_ckpt
                                # 🔧 기존 캐시 클리어 (메모리 절약)
                                if IntegratedPipelineOrchestrator._neural_policy_cache:
                                    logger.debug("🗑️ 기존 모델 캐시 클리어 (메모리 절약)")
                                    IntegratedPipelineOrchestrator._neural_policy_cache.clear()
                                neural_policy = load_ckpt(ckpt_path)
                                IntegratedPipelineOrchestrator._neural_policy_cache[cache_key] = neural_policy
                                IntegratedPipelineOrchestrator._cache_key = cache_key
                                logger.info(f"✅ 하이브리드 모델 로드 및 캐시: {model_id}")
                        else:
                            neural_policy = None
                        
                        if neural_policy:
                            hybrid_config['enable_neural'] = True
                        else:
                            hybrid_config['enable_neural'] = False
                            use_hybrid = False
                            
                except FileNotFoundError as e:
                    # 체크포인트 파일이 없는 경우 (새 모델 필요)
                    logger.debug(f"ℹ️ 체크포인트 파일 없음, 새 모델 사용: {e}")
                    use_hybrid = False
                except Exception as e:
                    # 🔧 체크포인트 로드 실패 시 더 명확한 메시지
                    error_msg = str(e)
                    if "unpack" in error_msg.lower() or "extra data" in error_msg.lower():
                        logger.warning(f"⚠️ 체크포인트 파일 손상 감지 (규칙 기반 모드로 폴백): {error_msg}")
                        logger.info(f"   💡 손상된 체크포인트는 무시하고 새 모델을 학습합니다")
                    else:
                        logger.warning(f"⚠️ 하이브리드 모드 설정 실패, 규칙 기반으로 폴백: {error_msg}")
                    use_hybrid = False
            
            # Self-play 실행 - 매 에피소드마다 다른 전략 샘플링 + 실제 캔들 데이터 🔥
            # 전략 풀 크기에 따라 동적으로 에이전트 수 조정
            # **중요**: 전략 풀 크기보다 작게 설정해야 다양성 확보 가능
            total_pool_size = len(all_strategies_pool) if all_strategies_pool else 0
            if total_pool_size > 8:
                agents_per_episode = 4  # 전략 풀이 충분하면 4개
            elif total_pool_size > 4:
                agents_per_episode = 3  # 전략 풀이 보통이면 3개
            elif total_pool_size > 0:
                agents_per_episode = min(2, total_pool_size)  # 전략 풀이 작으면 1-2개
            else:
                agents_per_episode = 3  # 전략 풀이 없으면 기본값
            
            logger.info(f"🎯 에이전트 설정: 전략 풀 {total_pool_size}개 중 매 에피소드 {agents_per_episode}개 사용 (다양성 확보)")
            
            # 🆕 적응형 예측 Self-play 비율 계산
            try:
                from rl_pipeline.pipelines.selfplay_adaptive import get_adaptive_predictive_ratio
                adaptive_ratio = get_adaptive_predictive_ratio(
                    coin=coin,
                    interval=interval,
                    base_ratio=PREDICTIVE_SELFPLAY_RATIO,
                    enable_auto=True
                )
            except Exception as e:
                logger.warning(f"⚠️ 적응형 비율 계산 실패, 기본값 사용: {e}")
                adaptive_ratio = PREDICTIVE_SELFPLAY_RATIO
            
            
            # Self-play 진화 실행 (기본 모드)
            logger.info("🚀 Self-play 진화 실행")
            selfplay_result = run_self_play_test(
                strategy_params_list,
                episodes=dynamic_episodes,
                all_strategy_pool=all_strategies_pool if all_strategies_pool else strategy_params_list,
                agents_per_episode=agents_per_episode,
                candle_data=candle_data,
                coin=coin,
                interval=interval,
                session_id=self.session_id  # 세션 ID 전달
            )
            
            
            
            # 전략 진화 적용
            evolved_strategies = self._apply_selfplay_evolution(
                strategies,
                selfplay_result,
                used_predictive=False,
                dual_mode=False
            )
            
            # 🔥 selfplay 결과 저장 (나중에 레짐 라우팅에서 사용)
            self._current_selfplay_result[interval] = selfplay_result
            
            return evolved_strategies
        except Exception as e:
            logger.error(f"❌ Self-play 진화 실패: {e}")
            logger.exception(e)
            return strategies  # 실패 시 원본 전략 반환

    
    def _analyze_strategy_synergy(self, coin: str, interval: str, strategies: List[Dict[str, Any]], candle_data: pd.DataFrame) -> Dict[str, Any]:
        """전략 간 시너지 분석"""
        try:
            if not strategies:
                return {'synergy_score': 0.5, 'synergy_patterns': {}}
            
            # 전략 파라미터 유사도 분석
            param_similarities = []
            for i, s1 in enumerate(strategies[:10]):
                for s2 in strategies[i+1:i+3]:
                    try:
                        # RSI 유사도
                        rsi_sim = 1.0 - abs(s1.get('rsi_min', 30) - s2.get('rsi_min', 30)) / 40.0
                        # Volume 유사도
                        vol_sim = 1.0 - abs(s1.get('volume_ratio_min', 1.0) - s2.get('volume_ratio_min', 1.0))
                        param_similarities.append((rsi_sim + vol_sim) / 2)
                    except:
                        pass
            
            synergy_score = sum(param_similarities) / len(param_similarities) if param_similarities else 0.5
            
            return {
                'synergy_score': float(synergy_score),
                'synergy_patterns': {
                    'strategy_count': len(strategies),
                    'avg_similarity': float(np.mean(param_similarities)) if param_similarities else 0.5
                }
            }
        except Exception as e:
            logger.error(f"Synergy 분석 실패: {e}")
            return {'synergy_score': 0.5}
    
    def _analyze_dual_selfplay_synergy(
        self, 
        coin: str, 
        interval: str, 
        evolved_strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        이중 Self-play 시너지 분석
        
        예측 실현 + 기존 Self-play 결과를 비교하여 상호 보완 관계 검증
        
        Args:
            coin: 코인 심볼
            interval: 인터벌
            evolved_strategies: 진화된 전략 리스트
        
        Returns:
            시너지 분석 결과
        """
        try:
            from rl_pipeline.db.connection_pool import get_optimized_db_connection
            
            with get_optimized_db_connection("strategies") as conn:
                cursor = conn.cursor()
                
                # 최근 예측 실현 결과 조회
                cursor.execute("""
                    SELECT 
                        AVG(CASE WHEN acc_flag = 1 THEN 1.0 ELSE 0.0 END) as avg_pred_accuracy,
                        AVG(realized_ret_signed) as avg_pred_return,
                        COUNT(*) as pred_count
                    FROM rl_episode_summary
                    WHERE coin = ? AND interval = ?
                      AND ts_exit >= datetime('now', '-7 days')
                """, (coin, interval))
                
                pred_result = cursor.fetchone()
                avg_pred_accuracy = pred_result[0] if pred_result[0] else 0.0
                avg_pred_return = pred_result[1] if pred_result[1] else 0.0
                pred_count = pred_result[2] if pred_result[2] else 0
                
                # 최근 기존 Self-play 결과 조회
                cursor.execute("""
                    SELECT 
                        AVG(win_rate) as avg_win_rate,
                        AVG(total_return) as avg_return,
                        COUNT(*) as trad_count
                    FROM simulation_results
                    WHERE coin = ? AND interval = ?
                      AND created_at >= datetime('now', '-7 days')
                """, (coin, interval))
                
                trad_result = cursor.fetchone()
                avg_win_rate = trad_result[0] if trad_result[0] else 0.0
                avg_return = trad_result[1] if trad_result[1] else 0.0
                trad_count = trad_result[2] if trad_result[2] else 0
                
                # 시너지 점수 계산
                # 예측 정확도가 높고 거래 성과도 좋으면 높은 시너지
                if pred_count > 0 and trad_count > 0:
                    synergy_score = (
                        (avg_pred_accuracy * 0.6) +  # 예측 정확도 60%
                        (min(avg_win_rate, 1.0) * 0.4)  # 승률 40%
                    )
                    
                    logger.info(f"💡 이중 Self-play 시너지 분석 ({coin}-{interval}):")
                    logger.info(f"   🎯 예측 실현: 정확도 {avg_pred_accuracy:.1%}, 수익 {avg_pred_return:+.2%} ({pred_count}건)")
                    logger.info(f"   📊 기존 Self-play: 승률 {avg_win_rate:.1%}, 수익 {avg_return:+.2%} ({trad_count}건)")
                    logger.info(f"   ⚡ 시너지 점수: {synergy_score:.2f} "
                              f"{'🔥 우수' if synergy_score > 0.7 else '✅ 양호' if synergy_score > 0.5 else '⚠️ 개선 필요'}")
                    
                    return {
                        'synergy_score': synergy_score,
                        'pred_accuracy': avg_pred_accuracy,
                        'pred_return': avg_pred_return,
                        'pred_count': pred_count,
                        'trad_win_rate': avg_win_rate,
                        'trad_return': avg_return,
                        'trad_count': trad_count
                    }
                
                return {'synergy_score': 0.5, 'insufficient_data': True}
                
        except Exception as e:
            logger.debug(f"⚠️ 이중 Self-play 시너지 분석 실패: {e}")
            return {'synergy_score': 0.5, 'error': str(e)}
    
    def _apply_selfplay_evolution(
        self, 
        strategies: List[Dict[str, Any]], 
        selfplay_result: Dict[str, Any],
        used_predictive: bool = False,
        dual_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Self-play 결과를 바탕으로 전략 진화 적용
        
        🔥 상호 보완 진화:
        - 예측 실현 Self-play: 예측 정확도 기반 진화 (방향/가격/시간 정확도)
        - 기존 Self-play: 거래 성과 기반 진화 (승률/수익률/샤프)
        - 동시 실행 모드: 두 결과 통합하여 등급 정확도 향상 🔥
        
        Args:
            strategies: 원본 전략 리스트
            selfplay_result: Self-play 결과
            used_predictive: 예측 실현 Self-play 사용 여부
            dual_mode: 동시 실행 모드 여부
        """
        try:
            evolved_strategies = []
            
            # Self-play 결과에서 학습된 패턴 추출
            summary = selfplay_result.get("summary", {})
            cycle_results = selfplay_result.get("cycle_results", [])
            
            # 🔥 동시 실행 모드: 두 결과 통합 처리 (등급 정확도 향상)
            if dual_mode and selfplay_result.get('dual_mode'):
                traditional_result = selfplay_result.get('traditional_result')
                predictive_result = selfplay_result.get('predictive_result')
                
                # 두 결과 모두 활용하여 종합 평가
                logger.info(f"🔥 동시 실행 모드: 두 방식 결과 통합 평가 중 (등급 정확도 향상 목표)")
                
                # 예측 정확도 데이터 추출
                pred_accuracy = 0.0
                pred_reward = 0.0
                pred_count = 0
                if predictive_result and predictive_result.get("status") in ["success", "failed"]:
                    episode_results = predictive_result.get("episode_results", [])
                    # 🔥 성공한 에피소드만 계산 (스킵 제외)
                    successful_episodes = [r for r in episode_results if r.get("status") == "success"]
                    if successful_episodes:
                        pred_count = len(successful_episodes)
                        # 🔥 result 구조 확인: result 안에 있거나 직접 있을 수 있음
                        acc_flags = []
                        rewards = []
                        for e in successful_episodes:
                            # result 안에서 먼저 찾고, 없으면 직접 찾기
                            result = e.get("result", {})
                            if result:
                                acc_flag = result.get("acc_flag")
                                total_reward = result.get("total_reward")
                            else:
                                acc_flag = e.get("acc_flag")
                                total_reward = e.get("total_reward")
                            
                            if acc_flag is not None:
                                acc_flags.append(acc_flag)
                            if total_reward is not None:
                                rewards.append(total_reward)
                        
                        if acc_flags:
                            pred_accuracy = sum(acc_flags) / len(acc_flags)
                        if rewards:
                            pred_reward = sum(rewards) / len(rewards)
                    else:
                        # 스킵된 에피소드 정보 로깅
                        skipped_count = len([r for r in episode_results if r.get("status") == "skipped"])
                        logger.debug(f"📊 예측 실현 에피소드: 성공 0개, 스킵 {skipped_count}개")
                
                # 거래 성과 데이터 추출
                trad_win_rate = 0.0
                trad_profit = 0.0
                trad_count = 0
                if traditional_result and traditional_result.get("status") == "success":
                    summary = traditional_result.get("summary", {}) or {}
                    
                    # 🔥 summary에서 직접 추출 (None 체크 강화)
                    trad_win_rate = summary.get("average_win_rate") if summary.get("average_win_rate") is not None else 0
                    trad_profit = summary.get("average_profit") if summary.get("average_profit") is not None else 0
                    trad_count = summary.get("total_trades") if summary.get("total_trades") is not None else 0
                    
                    # 🔥 summary에서 값이 없거나 0이면 cycle_results에서 계산 (개선)
                    if (trad_win_rate == 0 and trad_profit == 0) or (not summary):
                        cycle_results = traditional_result.get("cycle_results", [])
                        if cycle_results:
                            all_trades = []
                            all_profits = []
                            total_trades_from_cycles = 0
                            for cycle in cycle_results:
                                results = cycle.get("results", {})
                                for agent_id, agent_result in results.items():
                                    if agent_result:
                                        trades = agent_result.get("trades", [])
                                        trades_count = agent_result.get("total_trades", len(trades) if trades else 0)
                                        total_trades_from_cycles += trades_count
                                        if trades:
                                            all_trades.extend(trades)
                                        profit = agent_result.get("profit", 0)
                                        if profit != 0:
                                            all_profits.append(profit)
                            
                            if all_trades or total_trades_from_cycles > 0:
                                wins = sum(1 for t in all_trades if t.get("profit", 0) > 0) if all_trades else 0
                                trad_win_rate = wins / len(all_trades) if all_trades else 0
                                trad_profit = sum(all_profits) / len(all_profits) if all_profits else 0
                                trad_count = total_trades_from_cycles if total_trades_from_cycles > 0 else len(all_trades)
                                logger.debug(f"📊 cycle_results에서 계산: 승률 {trad_win_rate:.1%}, 수익 {trad_profit:.2f}, 거래 {trad_count}회")
                            elif total_trades_from_cycles == 0:
                                logger.warning(f"⚠️ cycle_results에서 거래가 발생하지 않음 (total_trades=0)")
                
                logger.info(f"📊 통합 평가:")
                logger.info(f"   🎯 예측 실현: 정확도 {pred_accuracy:.1%}, 보상 {pred_reward:.3f} ({pred_count}건 성공)")
                logger.info(f"   📊 기존 Self-play: 승률 {trad_win_rate:.1%}, 수익 {trad_profit:+.2f} ({trad_count}건 거래)")
                
                # 등급 정확도를 위해 두 데이터 모두 반영 (동시 실행 모드의 핵심!)
                if pred_accuracy > 0.6 and trad_win_rate > 0.5:
                    logger.info(f"✅ 우수한 통합 성과 → 등급 정확도 향상 기대")
                    # 동시 실행 모드에서는 예측 정확도와 거래 성과 모두 고려하여 등급 부여
                    # 이 정보는 이후 롤업 단계에서 활용됨
            
            # 🔥 예측 실현 Self-play 결과 처리 (단독 실행 시)
            elif used_predictive:
                episode_results = selfplay_result.get("episode_results", [])
                successful_episodes = [r for r in episode_results if r.get("status") == "success"]
                
                if successful_episodes:
                    # 예측 정확도 기반 진화
                    avg_accuracy = sum(e.get("acc_flag", 0) for e in successful_episodes) / len(successful_episodes)
                    avg_reward = sum(e.get("total_reward", 0) for e in successful_episodes) / len(successful_episodes)
                    
                    logger.info(f"📊 예측 실현 결과: 평균 정확도 {avg_accuracy:.1%}, 평균 보상 {avg_reward:.3f}")
                    
                    # 예측 성과가 좋은 전략은 예측 관련 파라미터 강화
                    if avg_accuracy > 0.7 and avg_reward > 0.5:
                        logger.info(f"✅ 예측 능력 우수 → 예측 관련 파라미터 강화")
            
            # 🔥 모든 에피소드에서 사용된 전략 파라미터 수집
            all_evolved_params = []
            for cycle in cycle_results:
                results = cycle.get("results", {})
                for agent_id, performance in results.items():
                    # 전략 파라미터가 있으면 수집
                    if 'strategy_params' in performance:
                        all_evolved_params.append(performance['strategy_params'])
            
            # 마지막 에피소드의 성과를 전략 성능 지표로 사용
            for i, strategy in enumerate(strategies):
                # 원본 전략 복사
                evolved_strategy = strategy.copy()
                
                # 🔥 마지막 에피소드에서 실제로 사용된 파라미터로 업데이트
                if cycle_results and len(cycle_results) > 0:
                    last_episode = cycle_results[-1]
                    last_results = last_episode.get("results", {})
                    if last_results and i < len(last_results):
                        # 마지막 에피소드의 각 에이전트 파라미터 사용
                        agent_performances = list(last_results.values())
                        if i < len(agent_performances) and 'strategy_params' in agent_performances[i]:
                            evolved_params = agent_performances[i]['strategy_params']
                            # 파라미터 업데이트
                            evolved_strategy['rsi_min'] = evolved_params.get('rsi_min', evolved_strategy.get('rsi_min', 30.0))
                            evolved_strategy['rsi_max'] = evolved_params.get('rsi_max', evolved_strategy.get('rsi_max', 70.0))
                            evolved_strategy['volume_ratio_min'] = evolved_params.get('volume_ratio_min', evolved_strategy.get('volume_ratio_min', 1.0))
                            evolved_strategy['volume_ratio_max'] = evolved_params.get('volume_ratio_max', evolved_strategy.get('volume_ratio_max', 2.0))
                            evolved_strategy['stop_loss_pct'] = evolved_params.get('stop_loss_pct', evolved_strategy.get('stop_loss_pct', 0.02))
                            evolved_strategy['take_profit_pct'] = evolved_params.get('take_profit_pct', evolved_strategy.get('take_profit_pct', 0.04))
                            evolved_strategy['macd_buy_threshold'] = evolved_params.get('macd_buy_threshold', evolved_strategy.get('macd_buy_threshold', 0.01))
                            evolved_strategy['macd_sell_threshold'] = evolved_params.get('macd_sell_threshold', evolved_strategy.get('macd_sell_threshold', -0.01))
                
                # Self-play 결과에서 실제 성능 지표 추출
                # 🔥 온라인 Self-play의 경우 summary에서 직접 가져오기 (전략별 누적 성과)
                if selfplay_result and selfplay_result.get("source") == "online_selfplay":
                    # 온라인 Self-play는 summary에 전략별 상세 성과가 있음
                    summary = selfplay_result.get("summary", {})
                    strategy_details = summary.get("strategy_details", {})
                    
                    # 현재 전략 ID로 성과 찾기
                    strategy_id = evolved_strategy.get('id')
                    if strategy_id and strategy_id in strategy_details:
                        strategy_perf = strategy_details[strategy_id]
                        evolved_strategy['profit'] = strategy_perf.get('avg_profit', 0.0) * strategy_perf.get('segment_count', 1)  # 총 수익
                        evolved_strategy['win_rate'] = 0.5 if strategy_perf.get('avg_pf', 0.0) > 1.0 else (0.3 if strategy_perf.get('avg_pf', 0.0) > 0.5 else 0.2)  # PF 기반 추정
                        evolved_strategy['trades_count'] = strategy_perf.get('total_trades', 0)
                        evolved_strategy['max_drawdown'] = strategy_perf.get('max_mdd', 0.0)
                        evolved_strategy['sharpe_ratio'] = strategy_perf.get('avg_sharpe', 0.0)
                        evolved_strategy['profit_factor'] = strategy_perf.get('avg_pf', 0.0)
                        logger.debug(f"📊 온라인 Self-play 성과 반영: {strategy_id}, 수익={evolved_strategy['profit']:.2f}, 거래={evolved_strategy['trades_count']}")
                    elif cycle_results:
                        # strategy_details에 없으면 cycle_results에서 계산
                        all_profits = []
                        all_win_rates = []
                        all_trades = []
                        all_drawdowns = []
                        all_sharpes = []
                        all_pfs = []
                        
                        # 모든 세그먼트의 누적 성과 사용
                        for cycle in cycle_results:
                            results = cycle.get("results", {})
                            for agent_id, agent_result in results.items():
                                if agent_id == strategy_id or not strategy_id:  # 전략 ID 매칭 또는 모든 전략
                                    if agent_result:
                                        all_profits.append(agent_result.get('total_pnl', 0.0))
                                        all_win_rates.append(agent_result.get('win_rate', 0.0))
                                        all_trades.append(agent_result.get('total_trades', 0))
                                        all_drawdowns.append(agent_result.get('max_drawdown', 0.0))
                                        all_sharpes.append(agent_result.get('sharpe_ratio', 0.0))
                                        all_pfs.append(agent_result.get('profit_factor', 0.0))
                        
                        if all_profits:
                            evolved_strategy['profit'] = sum(all_profits)  # 누적 수익
                            evolved_strategy['win_rate'] = sum(all_win_rates) / len(all_win_rates) if all_win_rates else 0.0
                            evolved_strategy['trades_count'] = sum(all_trades)
                            evolved_strategy['max_drawdown'] = max(all_drawdowns) if all_drawdowns else 0.0
                            evolved_strategy['sharpe_ratio'] = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0
                            evolved_strategy['profit_factor'] = sum(all_pfs) / len(all_pfs) if all_pfs else 0.0
                elif cycle_results:
                    # 전통 Self-play 또는 기타: 모든 에피소드의 누적 성과 사용
                    all_profits = []
                    all_win_rates = []
                    all_trades = []
                    all_drawdowns = []
                    all_sharpes = []
                    all_pfs = []
                    
                    for cycle in cycle_results:
                        results = cycle.get("results", {})
                        strategy_id_check = evolved_strategy.get('id')
                        for agent_id, agent_result in results.items():
                            if not strategy_id_check or agent_id == strategy_id_check:
                                if agent_result:
                                    all_profits.append(agent_result.get('total_pnl', 0.0))
                                    all_win_rates.append(agent_result.get('win_rate', 0.0))
                                    all_trades.append(agent_result.get('total_trades', 0))
                                    all_drawdowns.append(agent_result.get('max_drawdown', 0.0))
                                    all_sharpes.append(agent_result.get('sharpe_ratio', 0.0))
                                    all_pfs.append(agent_result.get('profit_factor', 0.0))
                    
                    if all_profits:
                        evolved_strategy['profit'] = sum(all_profits)  # 누적 수익
                        evolved_strategy['win_rate'] = sum(all_win_rates) / len(all_win_rates) if all_win_rates else 0.0
                        evolved_strategy['trades_count'] = sum(all_trades)
                        evolved_strategy['max_drawdown'] = max(all_drawdowns) if all_drawdowns else 0.0
                        evolved_strategy['sharpe_ratio'] = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0
                        evolved_strategy['profit_factor'] = sum(all_pfs) / len(all_pfs) if all_pfs else 0.0
                        
                        # 추가 성능 지표 계산
                        # Calmar Ratio
                        if evolved_strategy['max_drawdown'] > 0:
                            evolved_strategy['calmar_ratio'] = (evolved_strategy['profit'] / 10000.0) / evolved_strategy['max_drawdown']
                        else:
                            evolved_strategy['calmar_ratio'] = 0.0
                        
                        # Profit Factor (이미 위에서 설정된 경우 유지, 없으면 계산)
                        if 'profit_factor' not in evolved_strategy or evolved_strategy.get('profit_factor', 0.0) == 0.0:
                            evolved_strategy['profit_factor'] = evolved_strategy['win_rate'] / (1 - evolved_strategy['win_rate']) if evolved_strategy['win_rate'] < 1 else 10.0
                        
                        # Avg Profit Per Trade
                        if evolved_strategy['trades_count'] > 0:
                            evolved_strategy['avg_profit_per_trade'] = evolved_strategy['profit'] / evolved_strategy['trades_count']
                        else:
                            evolved_strategy['avg_profit_per_trade'] = 0.0
                        
                        # Complexity Score
                        param_count = sum([
                            1 if evolved_strategy.get('rsi_min') else 0,
                            1 if evolved_strategy.get('rsi_max') else 0,
                            1 if evolved_strategy.get('volume_ratio_min') else 0,
                            1 if evolved_strategy.get('volume_ratio_max') else 0,
                            1 if evolved_strategy.get('macd_buy_threshold') else 0,
                            1 if evolved_strategy.get('macd_sell_threshold') else 0,
                        ])
                        evolved_strategy['complexity_score'] = min(1.0, param_count / 6.0)
                        
                        # Score (종합 점수) - 절대 기준
                        profit = evolved_strategy.get('profit', 0)
                        win_rate = evolved_strategy.get('win_rate', 0)
                        sharpe = evolved_strategy.get('sharpe_ratio', 0)
                        max_dd = evolved_strategy.get('max_drawdown', 0)
                        profit_factor = evolved_strategy.get('profit_factor', 0)
                        trades_count = evolved_strategy.get('trades_count', 0)
                        
                        # 절대 기준으로 점수 계산
                        # profit을 퍼센트로 변환 (10000달러 기준)
                        profit_percent = (profit / 10000.0) * 100 if isinstance(profit, (int, float)) else 0.0
                        
                        # 1. 수익성 (35%): 실제 수익률 기준 (퍼센트)
                        if profit_percent > 10.0:  # 10% 이상
                            profit_score = 1.0
                        elif profit_percent > 5.0:  # 5% 이상
                            profit_score = 0.8
                        elif profit_percent > 2.0:  # 2% 이상
                            profit_score = 0.6
                        elif profit_percent > 0:  # 0% 이상
                            profit_score = 0.4
                        elif profit_percent > -2.0:  # -2% 이상
                            profit_score = 0.2
                        else:  # 손실
                            profit_score = 0.0
                        
                        # 2. 승률 (25%): 절대 기준
                        if win_rate > 0.65:
                            win_rate_score = 1.0
                        elif win_rate > 0.55:
                            win_rate_score = 0.8
                        elif win_rate > 0.50:
                            win_rate_score = 0.6
                        elif win_rate > 0.45:
                            win_rate_score = 0.4
                        elif win_rate > 0.40:
                            win_rate_score = 0.2
                        else:
                            win_rate_score = 0.0
                        
                        # 3. 샤프 비율 (20%)
                        if sharpe > 2.0:
                            sharpe_score = 1.0
                        elif sharpe > 1.5:
                            sharpe_score = 0.8
                        elif sharpe > 1.0:
                            sharpe_score = 0.6
                        elif sharpe > 0.5:
                            sharpe_score = 0.4
                        elif sharpe > 0:
                            sharpe_score = 0.2
                        else:
                            sharpe_score = 0.0
                        
                        # 4. 최대 낙폭 (10%)
                        max_dd_score = max(0, 1.0 - (max_dd / 0.2))  # 20% 이상 낙폭이면 0점
                        
                        # 5. 수익비 (10%) 🆕
                        if profit_factor > 3.0:  # 이익/손실 3배 이상
                            profit_factor_score = 1.0
                        elif profit_factor > 2.0:  # 이익/손실 2배 이상
                            profit_factor_score = 0.8
                        elif profit_factor > 1.5:  # 이익/손실 1.5배 이상
                            profit_factor_score = 0.6
                        elif profit_factor > 1.0:  # 이익/손실 1배 이상
                            profit_factor_score = 0.4
                        elif profit_factor > 0.7:  # 이익/손실 0.7배 이상
                            profit_factor_score = 0.2
                        else:  # 손실이 더 큼
                            profit_factor_score = 0.0
                        
                        evolved_strategy['score'] = (
                            profit_score * 0.30 +      # 30% (안정성 강화)
                            win_rate_score * 0.20 +   # 20% (PF가 더 중요)
                            sharpe_score * 0.25 +     # 25% (리스크 대비 효율 강화)
                            max_dd_score * 0.15 +     # 15% (장기 생존 핵심)
                            profit_factor_score * 0.10  # 10% (기초 품질)
                        )
                        
                        # 🔥 Quality Grade - 시뮬레이션 기반 등급 부여 제거
                        # 시뮬레이션은 파라미터 튜닝(진화)에만 사용하고,
                        # 등급은 레짐 라우팅(실제 백테스트) + 통합 분석 결과로만 결정
                        # 이유: 시뮬레이션 환경의 왜곡(특히 240m 100% 승률 등) 방지
                        evolved_strategy['quality_grade'] = 'UNKNOWN'  # 초기값, 라우팅/분석 후 업데이트됨

                        # 참고: 기존 시뮬레이션 기반 등급 계산 (현재 비활성화)
                        # from rl_pipeline.core.strategy_grading import StrategyGrading
                        # evolved_strategy['quality_grade'] = StrategyGrading.calculate_grade(
                        #     profit_percent=profit_percent,
                        #     win_rate=win_rate,
                        #     sharpe=sharpe,
                        #     max_dd=max_dd,
                        #     profit_factor=profit_factor,
                        #     is_initial_learning=False
                        # )
                
                # Self-play 결과에 따른 파라미터 조정
                if i < len(summary.get("top_performers", [])):
                    performer_data = summary["top_performers"][i]
                    
                    # 성과 기반 파라미터 조정
                    if performer_data.get("win_rate", 0) > 0.6:
                        # 높은 승률: 더 공격적으로
                        evolved_strategy['rsi_min'] = max(20, evolved_strategy.get('rsi_min', 30) - 5)
                        evolved_strategy['rsi_max'] = min(80, evolved_strategy.get('rsi_max', 70) + 5)
                    elif performer_data.get("win_rate", 0) < 0.4:
                        # 낮은 승률: 더 보수적으로
                        evolved_strategy['rsi_min'] = min(40, evolved_strategy.get('rsi_min', 30) + 5)
                        evolved_strategy['rsi_max'] = max(60, evolved_strategy.get('rsi_max', 70) - 5)
                
                # 🔥 상호 보완 진화 메타데이터 추가
                evolved_strategy['evolved'] = True
                evolved_strategy['evolution_source'] = 'predictive' if used_predictive else 'traditional'
                evolved_strategy['evolution_timestamp'] = datetime.now().isoformat()
                
                evolved_strategies.append(evolved_strategy)
            
            return evolved_strategies
            
        except Exception as e:
            logger.error(f"❌ Self-play 진화 적용 실패: {e}")
            return strategies

    def _evolve_existing_strategies(self, coin: str, interval: str, new_strategies: List[Dict]) -> List[Dict]:
        """
        기존 전략을 진화시켜 새로운 전략 생성 (유전 알고리즘)

        Args:
            coin: 코인
            interval: 인터벌
            new_strategies: 새로 생성된 전략 리스트

        Returns:
            진화된 전략 리스트
        """
        try:
            # 환경변수로 진화 활성화 여부 확인
            enable_evolution = os.getenv('ENABLE_STRATEGY_EVOLUTION', 'true').lower() == 'true'

            if not enable_evolution:
                logger.debug(f"⏭️ {coin}-{interval}: 전략 진화 비활성화")
                return []

            logger.info(f"🧬 {coin}-{interval}: 전략 진화 시작")

            # StrategyEvolver import
            from rl_pipeline.strategy.strategy_evolver import StrategyEvolver
            from rl_pipeline.db.connection_pool import get_strategy_db_pool

            # 기존 전략 조회 (DB에서)
            pool = get_strategy_db_pool()
            with pool.get_connection() as conn:
                cursor = conn.cursor()

                # 상위 등급 전략만 조회 (S, A, B)
                cursor.execute("""
                    SELECT
                        cs.id as strategy_id,
                        cs.coin,
                        cs.interval,
                        cs.params,
                        cs.regime,
                        sg.grade as quality_grade,
                        sr.avg_ret,
                        sr.win_rate,
                        sr.predictive_accuracy
                    FROM coin_strategies cs
                    LEFT JOIN strategy_grades sg ON cs.id = sg.strategy_id
                    LEFT JOIN rl_strategy_rollup sr ON cs.id = sr.strategy_id
                    WHERE cs.coin = ?
                      AND cs.interval = ?
                      AND sg.grade IN ('S', 'A', 'B')
                    ORDER BY sg.grade_score DESC
                    LIMIT 100
                """, (coin, interval))

                rows = cursor.fetchall()

                if not rows:
                    logger.debug(f"⏭️ {coin}-{interval}: 진화 가능한 상위 전략 없음")
                    return []

                # Dict로 변환
                import json
                existing_strategies = []
                for row in rows:
                    strategy_dict = {
                        'strategy_id': row[0],
                        'coin': row[1],
                        'interval': row[2],
                        'params': json.loads(row[3]) if row[3] else {},
                        'regime': row[4],
                        'quality_grade': row[5] or 'UNKNOWN',
                        'avg_ret': row[6] or 0.0,
                        'win_rate': row[7] or 0.0,
                        'predictive_accuracy': row[8] or 0.0
                    }
                    existing_strategies.append(strategy_dict)

                logger.info(f"📊 {coin}-{interval}: 진화 대상 전략 {len(existing_strategies)}개 발견")

            # StrategyEvolver 초기화
            evolver = StrategyEvolver()

            # 상위 전략 선별
            top_strategies = evolver.select_top_strategies(
                existing_strategies,
                top_percent=0.3,  # 상위 30%
                min_grade='B'     # B 등급 이상
            )

            if not top_strategies:
                logger.debug(f"⏭️ {coin}-{interval}: 선별된 상위 전략 없음")
                return []

            logger.info(f"✅ {coin}-{interval}: 상위 전략 {len(top_strategies)}개 선별")

            # 진화 실행 (교배 + 변이)
            # 최대 5개의 진화된 전략 생성
            max_evolved = min(5, len(top_strategies) // 2)
            evolved_strategies = []

            for i in range(max_evolved):
                try:
                    # 랜덤으로 두 부모 선택
                    import random
                    parent1 = random.choice(top_strategies)
                    parent2 = random.choice(top_strategies)

                    # 교배
                    child_params = evolver.crossover(parent1, parent2)

                    # 변이 (tuple 반환: (dict, str))
                    mutated_params, mutation_desc = evolver.mutate(child_params)

                    # 진화된 전략 생성 (Strategy 객체로 변환)
                    from rl_pipeline.core.types import Strategy
                    evolved_strategy = Strategy(
                        id=f"evolved_{coin}_{interval}_{i}_{datetime.now().timestamp()}",
                        coin=coin,
                        interval=interval,
                        **mutated_params
                    )

                    # 메타데이터 추가
                    evolved_strategy.parent_strategy_id = parent1.get('strategy_id')
                    evolved_strategy.similarity_classification = 'evolved'
                    evolved_strategy.similarity_score = 0.7

                    evolved_strategies.append(evolved_strategy)
                    logger.debug(f"🧬 진화 전략 #{i+1} 생성 (부모: {parent1.get('strategy_id')[:8]}...)")

                except Exception as e:
                    logger.warning(f"⚠️ 진화 전략 생성 실패: {e}")
                    continue

            if evolved_strategies:
                logger.info(f"✅ {coin}-{interval}: {len(evolved_strategies)}개 진화 전략 생성 완료")

            return evolved_strategies

        except Exception as e:
            logger.error(f"❌ {coin}-{interval}: 전략 진화 실패: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def run_partial_pipeline(self, coin: str, interval: str, candle_data: pd.DataFrame) -> PipelineResult:
        """1-2단계만 실행: 전략생성 → Self-play(옵션) → 통합분석"""
        try:
            start_time = datetime.now()
            
            # 1단계: 전략 생성
            logger.info("1️⃣ 전략 생성 단계 시작")
            strategies = self._create_strategies(coin, interval, candle_data)
            logger.info(f"✅ {len(strategies)}개 전략 생성 완료")

            # 🧬 1-1단계: 기존 전략 진화 (유전 알고리즘)
            evolved_genetic_strategies = self._evolve_existing_strategies(coin, interval, strategies)
            if evolved_genetic_strategies:
                strategies.extend(evolved_genetic_strategies)
                logger.info(f"🧬 {len(evolved_genetic_strategies)}개 진화 전략 추가 (총 {len(strategies)}개)")

            # 🔥 예측 Self-play 실행 (전략 생성 직후)
            predictive_selfplay_result = None
            enable_predictive_selfplay = os.getenv('ENABLE_PREDICTIVE_SELFPLAY', 'true').lower() == 'true'
            if enable_predictive_selfplay:
                logger.info("🔥 예측 Self-play 실행 (전략 생성 직후)")

                # 🔥 디버거 초기화
                simulation_debugger = None
                try:
                    from rl_pipeline.monitoring.simulation_debugger import SimulationDebugger
                    simulation_debugger = SimulationDebugger(session_id=self.session_id)
                except Exception as debug_err:
                    logger.debug(f"⚠️ SimulationDebugger 초기화 실패: {debug_err}")

                try:
                    # Self-play 시작 로깅
                    if simulation_debugger:
                        candle_count = len(candle_data) if candle_data is not None and not candle_data.empty else 0
                        simulation_debugger.log_selfplay_start(
                            coin=coin,
                            interval=interval,
                            num_episodes=PREDICTIVE_SELFPLAY_EPISODES,
                            num_agents=len(strategies[:100]),
                            candle_count=candle_count
                        )

                    predictive_selfplay_result = self._run_predictive_selfplay(coin, interval, strategies, candle_data)

                    if predictive_selfplay_result:
                        episodes = predictive_selfplay_result.get('episodes', 0)
                        avg_accuracy = predictive_selfplay_result.get('avg_accuracy', 0)
                        best_accuracy = predictive_selfplay_result.get('best_accuracy', 0)

                        logger.info(f"✅ 예측 Self-play 완료: {episodes}개 에피소드, 평균 정확도 {avg_accuracy:.3f}, 최고 정확도 {best_accuracy:.3f}")

                        # 🔥 Self-play 결과 검증
                        validation = validate_selfplay_result(predictive_selfplay_result, coin, interval)
                        if not validation['valid']:
                            logger.error(f"❌ Self-play 결과 검증 실패: {validation['issues']}")
                            if simulation_debugger:
                                simulation_debugger.log_error(
                                    error_msg="Self-play 결과 검증 실패",
                                    context={
                                        'coin': coin,
                                        'interval': interval,
                                        'issues': validation['issues'],
                                        'warnings': validation['warnings']
                                    }
                                )
                        else:
                            if validation['warnings']:
                                logger.warning(f"⚠️ Self-play 결과 경고: {validation['warnings']}")
                            logger.info(f"✅ Self-play 결과 검증 통과")

                            # 검증 결과 저장
                            if simulation_debugger:
                                simulation_debugger.log({
                                    'event': 'selfplay_validation',
                                    'coin': coin,
                                    'interval': interval,
                                    'valid': True,
                                    'warnings': validation['warnings']
                                })

                        # 🔥 Self-play 종료 로깅
                        if simulation_debugger:
                            simulation_debugger.log_selfplay_end(
                                coin=coin,
                                interval=interval,
                                total_episodes=episodes,
                                summary={
                                    'avg_accuracy': avg_accuracy,
                                    'best_accuracy': best_accuracy,
                                    'strategy_count': predictive_selfplay_result.get('strategy_count', 0),
                                    'type': 'predictive',
                                    'early_stopped': episodes < PREDICTIVE_SELFPLAY_EPISODES
                                }
                            )

                            # 🔥 에피소드별 정확도 저장
                            for cycle_result in predictive_selfplay_result.get('cycle_results', []):
                                simulation_debugger.log({
                                    'event': 'predictive_selfplay_episode',
                                    'coin': coin,
                                    'interval': interval,
                                    'episode': cycle_result.get('episode'),
                                    'accuracy': cycle_result.get('accuracy'),
                                    'best_accuracy': cycle_result.get('best_accuracy'),
                                    'predictions': cycle_result.get('predictions')
                                })
                    else:
                        logger.warning("⚠️ 예측 Self-play 결과 없음")

                        # 🔥 실패 로깅
                        if simulation_debugger:
                            simulation_debugger.log_error(
                                error_msg="예측 Self-play 결과 없음",
                                context={'coin': coin, 'interval': interval}
                            )
                except Exception as e:
                    logger.warning(f"⚠️ 예측 Self-play 실패 (계속 진행): {e}")

                    # 🔥 에러 로깅
                    if simulation_debugger:
                        simulation_debugger.log_error(
                            error_msg="예측 Self-play 실행 실패",
                            context={'coin': coin, 'interval': interval},
                            exception=e
                        )

            # 🔥 시뮬레이션 Self-play는 Paper Trading으로 대체되어 제거됨
            # Paper Trading이 실제 시장 데이터를 사용하여 더 정확한 성과 검증 가능
            evolved_strategies = strategies
            selfplay_result = predictive_selfplay_result

            logger.info("⏭️ 시뮬레이션 Self-play 건너뛰기 (Paper Trading으로 대체)")
            logger.info("   💡 예측 정확도는 예측 Self-play에서 수집됩니다")
            
            # 🔥 통합분석은 모든 인터벌의 전략 생성이 완료된 후에만 실행됨
            # (run_integrated_analysis_all_intervals에서 실행)
            logger.debug(f"📊 {coin}-{interval}: 전략 생성 완료, 통합분석은 모든 인터벌 완료 후 실행")

            # 🔥 Self-play 완료 후 롤업 및 등급 평가 추가 (자동화)
            try:
                logger.info(f"🔄 {coin}-{interval} 롤업 및 등급 평가 시작...")
                from rl_pipeline.engine.rollup_batch import run_full_rollup_and_grades
                
                rollup_result = run_full_rollup_and_grades(coin=coin, interval=interval)
                
                if rollup_result.get("success"):
                    graded_count = rollup_result.get('grades_updated', 0)
                    logger.info(f"✅ {coin}-{interval} 롤업 및 등급 평가 완료: {graded_count}개 전략")
                    
                    # 🔥 coin_strategies 테이블의 quality_grade도 동기화
                    try:
                        self._sync_strategy_grades_to_coin_strategies(coin, interval)
                    except Exception as sync_error:
                        logger.debug(f"⚠️ 등급 동기화 실패 (무시): {sync_error}")
                else:
                    logger.warning(f"⚠️ 롤업 실행 실패: {rollup_result.get('error', 'unknown')}")
            except Exception as e:
                logger.warning(f"⚠️ 롤업 및 등급 평가 실패: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 🔥 통합분석은 모든 인터벌 완료 후에만 실행되므로, 여기서는 기본값 사용
            regime_detected = 'neutral'
            signal_score = 0.0
            signal_action = 'HOLD'
            
            return PipelineResult(
                coin=coin,
                interval=interval,
                strategies_created=len(evolved_strategies),
                selfplay_episodes=len(selfplay_result.get('cycle_results', [])) if selfplay_result and isinstance(selfplay_result, dict) else 0,
                regime_detected=regime_detected,
                routing_results=0,  # 레짐 라우팅 제거됨
                signal_score=signal_score,
                signal_action=signal_action,
                execution_time=execution_time,
                status="partial_complete",
                created_at=datetime.now().isoformat(),
                selfplay_result=selfplay_result  # 🔥 self-play 결과 저장 (None일 수 있음)
            )
            
        except Exception as e:
            logger.error(f"❌ 부분 파이프라인 실행 실패: {e}")
            return PipelineResult(
                coin=coin,
                interval=interval,
                strategies_created=0,
                selfplay_episodes=0,
                regime_detected="unknown",
                routing_results=0,
                signal_score=0.0,
                signal_action="ERROR",
                execution_time=0.0,
                status="failed",
                created_at=datetime.now().isoformat()
            )
    
    def run_integrated_analysis_all_intervals(self, coin: str, pipeline_results: List[PipelineResult], all_candle_data: Dict[Tuple[str, str], pd.DataFrame] = None) -> PipelineResult:
        """전체 인터벌 통합분석 실행"""
        try:
            start_time = datetime.now()
            
            # 🔥 명확한 로그: 모든 인터벌 완료 후 통합 분석 시작
            intervals_completed = [r.interval for r in pipeline_results if r.interval]
            logger.info(f"📊 {coin}: 모든 인터벌 개별 처리 완료 ({len(intervals_completed)}개: {', '.join(intervals_completed)})")
            logger.info(f"🚀 {coin}: 전체 통합 분석 시작 (모든 인터벌 데이터 종합)")
            
            # 🔥 통합분석기 v1 초기화 (계층적 분석)
            logger.info(f"🚀 {coin}: 통합 분석 v1 실행 (계층적 구조: 장기=방향, 단기=타이밍)")
            analyzer_v1 = IntegratedAnalyzerV1()

            # 통합분석 실행 (v1: 단순히 coin만 전달, DB에서 자동 로드)
            try:
                # v1 분석 실행
                v1_result = analyzer_v1.analyze(coin)

                logger.info(f"✅ {coin}: v1 통합 분석 완료")
                logger.info(f"   방향: {v1_result['direction']}, 타이밍: {v1_result['timing']}, "
                          f"크기: {v1_result['size']:.3f}, 확신도: {v1_result['confidence']:.3f}, "
                          f"기간: {v1_result['horizon']}")

                # v1 결과를 v0 형식으로 매핑
                direction = v1_result['direction']
                timing = v1_result['timing']

                # signal_action 매핑
                if direction == 'NEUTRAL' or timing == 'WAIT':
                    signal_action = 'HOLD'
                elif direction == 'LONG' and timing == 'NOW':
                    signal_action = 'BUY'
                elif direction == 'SHORT' and timing == 'NOW':
                    signal_action = 'SELL'
                elif timing == 'EXIT':
                    # 청산 신호
                    if direction == 'LONG':
                        signal_action = 'SELL'  # 롱 청산
                    elif direction == 'SHORT':
                        signal_action = 'BUY'   # 숏 청산
                    else:
                        signal_action = 'HOLD'
                else:
                    signal_action = 'HOLD'

                # analysis_result 객체 생성 (v0 호환)
                analysis_result = type('obj', (object,), {
                    'signal_action': signal_action,
                    'final_signal_score': v1_result['size'],
                    'signal_confidence': v1_result['confidence'],
                    'direction': direction,
                    'timing': timing,
                    'horizon': v1_result['horizon'],
                    'v1_reason': v1_result['reason']
                })

            except Exception as analysis_error:
                logger.error(f"❌ 통합분석 v1 실행 실패: {analysis_error}")
                import traceback
                traceback.print_exc()
                analysis_result = type('obj', (object,), {
                    'signal_action': 'HOLD',
                    'final_signal_score': 0.5,
                    'signal_confidence': 0.0
                })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 분석 결과에서 값 추출
            signal_score = getattr(analysis_result, 'final_signal_score', 0.5)
            signal_action = getattr(analysis_result, 'signal_action', 'HOLD')
            
            # 🔥 실시간 시그널 저장 (전체 인터벌 통합 결과) - 선택적
            # ⚠️ absolute_zero_system은 trading_system.db와 무관해야 하므로 비활성화
            # 활성화하려면 ENABLE_TRADING_SYSTEM_INTEGRATION=true 환경변수 설정
            enable_trading_integration = os.getenv('ENABLE_TRADING_SYSTEM_INTEGRATION', 'false').lower() == 'true'
            if enable_trading_integration:
                try:
                    from rl_pipeline.db.realtime_signal_storage import save_realtime_signal_from_analysis
                    
                    # 가장 최신 캔들 데이터 선택
                    latest_candle_data = None
                    if all_candle_data:
                        latest_key = max(all_candle_data.keys(), 
                            key=lambda k: all_candle_data[k].index[-1] if hasattr(all_candle_data[k], 'index') and len(all_candle_data[k]) > 0 else 0,
                            default=None)
                        if latest_key:
                            latest_candle_data = all_candle_data[latest_key]
                    
                    save_realtime_signal_from_analysis(
                        coin, 'combined', analysis_result, latest_candle_data
                    )
                    logger.info(f"✅ [{coin}] 전체 인터벌 통합 실시간 시그널 저장 완료")
                except Exception as e:
                    logger.warning(f"⚠️ [{coin}] 실시간 시그널 저장 실패: {e}")
            else:
                logger.debug(f"📊 {coin}: 거래 시스템 연동 비활성화 (ENABLE_TRADING_SYSTEM_INTEGRATION=false)")
            
            # 🔥 통합 학습 실행 (모든 인터벌 self-play + 분석 결과 활용)
            trained_model_id = None
            try:
                from rl_pipeline.hybrid.auto_trainer import (
                    auto_train_from_integrated_analysis,
                    should_auto_train
                )
                
                # 🔥 명확한 로그: 통합 분석 완료 후 학습 시작
                logger.info(f"🚀 {coin}: 통합 분석 완료 → 통합 학습 시작 (모든 인터벌 self-play 데이터 활용)")
                
                # 모든 인터벌의 self-play 결과 수집 (PipelineResult에서)
                all_interval_selfplay = {}
                for result in pipeline_results:
                    if result.interval and result.selfplay_result:
                        all_interval_selfplay[result.interval] = result.selfplay_result
                    else:
                        # 🔥 디버깅: selfplay_result가 없는 경우 로그
                        if result.interval:
                            logger.info(f"📊 {coin}-{result.interval}: selfplay_result 없음 (통합 학습에서 제외)")

                logger.info(f"📊 {coin}: 통합 학습 데이터 수집 완료 - 인터벌: {list(all_interval_selfplay.keys())} ({len(all_interval_selfplay)}개)")

                # 🔥 통합 학습 데이터 검증
                validation_result = validate_integrated_learning_data(
                    coin=coin,
                    all_interval_selfplay=all_interval_selfplay,
                    pipeline_results=pipeline_results,
                    min_intervals=2,
                    min_total_episodes=10
                )

                # 🔥 검증 결과 로깅
                logger.info(f"📊 {coin}: 통합 학습 데이터 검증 완료")
                logger.info(f"   └─ 검증 통과: {validation_result['valid']}")
                logger.info(f"   └─ 데이터 품질 점수: {validation_result.get('quality_score', 0)}/100")
                logger.info(f"   └─ 인터벌 수: {validation_result['stats'].get('num_intervals', 0)}개")
                logger.info(f"   └─ 총 에피소드: {validation_result['stats'].get('total_episodes', 0)}개")
                logger.info(f"   └─ 평균 정확도: {validation_result['stats'].get('overall_avg_accuracy', 0):.2%}")

                if validation_result['issues']:
                    logger.error(f"❌ {coin}: 통합 학습 데이터 검증 실패:")
                    for issue in validation_result['issues']:
                        logger.error(f"   └─ {issue}")

                if validation_result['warnings']:
                    logger.warning(f"⚠️ {coin}: 통합 학습 데이터 경고:")
                    for warning in validation_result['warnings']:
                        logger.warning(f"   └─ {warning}")

                # 🔥 인터벌별 상세 통계 로깅
                interval_dist = validation_result['stats'].get('interval_distribution', {})
                if interval_dist:
                    logger.info(f"📊 {coin}: 인터벌별 Self-play 통계:")
                    for interval, stat in interval_dist.items():
                        logger.info(f"   └─ {interval}: {stat['episodes']}개 에피소드, 평균 정확도 {stat['avg_accuracy']:.2%}")
                        if stat.get('issues'):
                            for issue in stat['issues']:
                                logger.error(f"      └─ ❌ {issue}")

                # 🔥 디버그 시스템에 검증 결과 저장
                try:
                    from rl_pipeline.monitoring.simulation_debugger import SimulationDebugger
                    debugger = SimulationDebugger(session_id=self.session_id)
                    debugger.log({
                        'event': 'integrated_learning_validation',
                        'coin': coin,
                        'validation_result': {
                            'valid': validation_result['valid'],
                            'quality_score': validation_result.get('quality_score', 0),
                            'num_intervals': validation_result['stats'].get('num_intervals', 0),
                            'total_episodes': validation_result['stats'].get('total_episodes', 0),
                            'overall_avg_accuracy': validation_result['stats'].get('overall_avg_accuracy', 0),
                            'num_issues': len(validation_result['issues']),
                            'num_warnings': len(validation_result['warnings'])
                        },
                        'issues': validation_result['issues'],
                        'warnings': validation_result['warnings']
                    })
                except Exception as debug_error:
                    logger.debug(f"⚠️ 검증 결과 디버그 로깅 실패: {debug_error}")

                # 학습 조건 체크 (최소 에피소드 수)
                total_episodes = sum(
                    len(sp_result.get('cycle_results', []))
                    for sp_result in all_interval_selfplay.values()
                    if isinstance(sp_result, dict)
                )

                logger.info(f"📊 {coin}: 총 {total_episodes}개 에피소드 수집됨 (최소 필요: 10개)")

                # 🔥 검증 실패 시 학습 건너뛰기
                if not validation_result['valid']:
                    logger.error(f"❌ {coin}: 통합 학습 데이터 검증 실패로 학습 건너뜀")
                    logger.error(f"   └─ 검증 이슈: {validation_result['issues']}")
                elif validation_result.get('quality_score', 0) < 30:
                    logger.warning(f"⚠️ {coin}: 데이터 품질 점수 낮음 ({validation_result.get('quality_score', 0)}/100), 학습 건너뜀")
                elif all_interval_selfplay and total_episodes >= 10:
                    # ENABLE_AUTO_TRAINING 체크
                    auto_train_enabled = os.getenv('ENABLE_AUTO_TRAINING', 'false').lower() == 'true'
                    use_hybrid = os.getenv('USE_HYBRID', 'false').lower() == 'true'
                    
                    logger.info(f"📊 {coin}: 학습 조건 체크 - ENABLE_AUTO_TRAINING={auto_train_enabled}, USE_HYBRID={use_hybrid}")
                    
                    if auto_train_enabled and use_hybrid:
                        config_path = os.getenv('HYBRID_CONFIG_PATH', '/workspace/rl_pipeline/hybrid/config_hybrid.json')
                        
                        logger.info(f"🚀 {coin}: 통합 학습 시작 (인터벌: {list(all_interval_selfplay.keys())}, 총 {total_episodes}개 에피소드)")
                        
                        trained_model_id = auto_train_from_integrated_analysis(
                            coin=coin,
                            all_interval_selfplay=all_interval_selfplay,
                            analysis_result=analysis_result,  # 🔥 분석 결과 전달
                            config_path=config_path,
                            min_episodes=10
                        )
                        
                        if trained_model_id:
                            logger.info(f"✅ {coin}: 통합 학습 완료, 모델 ID: {trained_model_id}")
                        else:
                            logger.info(f"📊 {coin}: 통합 학습 데이터 부족 또는 학습 실패")
                    else:
                        if not auto_train_enabled:
                            logger.info(f"📊 {coin}: 자동 학습 비활성화 (ENABLE_AUTO_TRAINING=false)")
                        elif not use_hybrid:
                            logger.info(f"📊 {coin}: 자동 학습 비활성화 (USE_HYBRID=false)")
                else:
                    if not all_interval_selfplay:
                        logger.info(f"📊 {coin}: self-play 결과 없음, 통합 학습 건너뜀")
                    elif total_episodes < 10:
                        logger.info(f"📊 {coin}: 에피소드 수 부족 ({total_episodes} < 10), 통합 학습 건너뜀")
                        
            except ImportError as import_err:
                logger.warning(f"⚠️ 통합 학습 모듈 없음 (하이브리드 시스템 미설치): {import_err}")
                import traceback
                logger.debug(f"임포트 오류 상세:\n{traceback.format_exc()}")
            except Exception as e:
                logger.error(f"❌ 통합 학습 중 오류 (계속 진행): {e}")
                import traceback
                logger.debug(f"통합 학습 오류 상세:\n{traceback.format_exc()}")

            # 🔥 통합 분석 결과를 DB에 저장 (학습 완료 후 저장)
            try:
                # regime 추출 (v1에서는 regime 정보 사용 안 함)
                regime = 'neutral'

                # 🔥 명확한 로그: 학습 완료 후 통합 분석 결과 저장
                logger.info(f"💾 {coin}: 통합 학습 완료 → 통합 분석 결과 저장 시작")

                # centralized save 함수 사용 (rl_strategies.db에 올바른 스키마로 저장)
                learning_results.save_integrated_analysis_results(coin, "all_intervals", regime, analysis_result)
                logger.info(f"✅ {coin}-all_intervals 통합 분석 결과 저장 완료: {signal_action} (점수: {signal_score:.3f})")

                # 🔥 DB 커밋 완료 대기 (Paper Trading이 즉시 조회할 수 있도록)
                import time
                time.sleep(0.05)  # 50ms 대기

                # 🔥 개별 인터벌별로도 결과 저장 (Paper Trading이 개별 인터벌별 결과를 찾을 수 있도록)
                saved_intervals = []
                for result in pipeline_results:
                    if result.interval and result.interval != "all_intervals":
                        try:
                            learning_results.save_integrated_analysis_results(
                                coin, result.interval, regime, analysis_result
                            )
                            saved_intervals.append(result.interval)
                            logger.debug(f"✅ {coin}-{result.interval} 통합 분석 결과 저장 (개별 인터벌 복제)")
                        except Exception as e:
                            logger.debug(f"⚠️ {coin}-{result.interval} 통합 분석 결과 저장 실패 (무시): {e}")

                if saved_intervals:
                    logger.info(f"✅ {coin} 개별 인터벌 저장 완료: {', '.join(saved_intervals)}")

                # 🔥 모든 저장 완료 후 추가 대기
                time.sleep(0.05)  # 50ms 대기

            except Exception as save_err:
                logger.warning(f"⚠️ 통합 분석 결과 저장 실패: {save_err}")
                import traceback
                logger.debug(f"상세 에러:\n{traceback.format_exc()}")

            return PipelineResult(
                coin=coin,
                interval="all_intervals",
                strategies_created=sum(r.strategies_created for r in pipeline_results),
                selfplay_episodes=sum(r.selfplay_episodes for r in pipeline_results),
                regime_detected="multi_interval",
                routing_results=sum(r.routing_results for r in pipeline_results),
                signal_score=signal_score,
                signal_action=signal_action,
                execution_time=execution_time,
                status="success",
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ 전체 통합분석 실패: {e}")
            return PipelineResult(
                coin=coin,
                interval="all_intervals",
                strategies_created=0,
                selfplay_episodes=0,
                regime_detected="unknown",
                routing_results=0,
                signal_score=0.5,
                signal_action="HOLD",
                execution_time=0.0,
                status="failed",
                created_at=datetime.now().isoformat()
            )

    def _perform_integrated_analysis(self, coin: str, interval: str, strategies: List[Dict[str, Any]], 
                                   candle_data: pd.DataFrame) -> Any:
        """3단계: 통합분석 (🔥 다중 인터벌 분석 개선, 레짐 라우팅 제거)"""
        try:
            if not strategies:
                logger.warning("⚠️ 분석할 전략이 없습니다")
                return self._create_default_analysis_result(coin, interval)
            
            # 현재 레짐 감지 (regime_transition_prob 포함)
            current_regime, regime_confidence, regime_transition_prob = self.regime_router.detect_current_regime(coin, interval, candle_data)
            logger.info(f"📊 현재 레짐: {current_regime} (신뢰도: {regime_confidence:.2f}, 전환 확률: {regime_transition_prob:.2%})")
            
            logger.info(f"📊 분석 대상 전략: {len(strategies)}개")
            
            # 🔥 단일 인터벌 분석만 수행 (개별 인터벌 처리 시)
            # 다중 인터벌 분석은 run_integrated_analysis_all_intervals에서만 수행
            try:
                logger.info(f"📊 단일 인터벌 분석 실행: {coin}-{interval}")
                analysis_result = analyze_coin_strategies(coin, interval, current_regime, strategies, candle_data)
            except Exception as e:
                logger.warning(f"⚠️ 단일 인터벌 분석 실패: {e}")
                # 폴백: 기본 분석 결과 반환
                analysis_result = self._create_default_analysis_result(coin, interval)

            # 🔥 analysis_result는 dict로 반환되므로 dict 방식으로 접근
            if isinstance(analysis_result, dict):
                signal_action = analysis_result.get('signal_action', 'HOLD')
                signal_score = analysis_result.get('signal_score', analysis_result.get('final_signal_score', 0.0))
                logger.info(f"🔍 통합분석 완료: {signal_action} (점수: {signal_score:.3f})")
            else:
                # 객체인 경우 (하위 호환성)
                signal_action = getattr(analysis_result, 'signal_action', 'HOLD')
                signal_score = getattr(analysis_result, 'final_signal_score', getattr(analysis_result, 'signal_score', 0.0))
                logger.info(f"🔍 통합분석 완료: {signal_action} (점수: {signal_score:.3f})")

            # Dict로 변환 (validator 호환성)
            # 🔥 analysis_result가 이미 dict인지 확인
            if isinstance(analysis_result, dict):
                result_dict = analysis_result.copy()
                # signal_score가 없으면 final_signal_score에서 가져오기
                if 'signal_score' not in result_dict and 'final_signal_score' in result_dict:
                    result_dict['signal_score'] = result_dict.pop('final_signal_score')
                return result_dict
            else:
                # 객체인 경우 asdict 사용
                result_dict = asdict(analysis_result)
                result_dict['signal_score'] = result_dict.pop('final_signal_score')
                return result_dict
            
        except Exception as e:
            logger.error(f"❌ 통합분석 실패: {e}")
            return self._create_default_analysis_result(coin, interval)
    
    def _create_default_strategies(self, coin: str, interval: str) -> List[Dict[str, Any]]:
        """기본 전략 생성 - 24개 다양성 전략 + 시장 레짐별 전문 전략 + 기존 고등급 전략 참고"""
        try:
            import random
            
            # 🔍 기존 고등급 전략 로드하여 참고
            high_grade_base_strategies = []
            try:
                from rl_pipeline.db.reads import load_strategies_by_grade
                existing_strategies = load_strategies_by_grade(coin, interval, 'A', limit=10)  # A등급 상위 10개
                
                if existing_strategies and len(existing_strategies) >= 3:
                    logger.info(f"✅ 기존 고등급 전략 {len(existing_strategies)}개 로드하여 베이스로 사용")
                    
                    # 고등급 전략의 파라미터를 베이스로 사용
                    for strategy in existing_strategies[:5]:  # 상위 5개만 사용
                        if 'params' in strategy and isinstance(strategy['params'], dict):
                            base_params = {
                                'rsi_min': strategy['params'].get('rsi_min', 30),
                                'rsi_max': strategy['params'].get('rsi_max', 70),
                                'volume_ratio_min': strategy['params'].get('volume_ratio_min', 1.0),
                                'volume_ratio_max': strategy['params'].get('volume_ratio_max', 2.0),
                                'macd_buy_threshold': strategy['params'].get('macd_buy_threshold', 0.01),
                                'macd_sell_threshold': strategy['params'].get('macd_sell_threshold', -0.01),
                                'stop_loss_pct': strategy['params'].get('stop_loss_pct', 0.02),
                                'take_profit_pct': strategy['params'].get('take_profit_pct', 0.05),
                                'type': f'evolved_{strategy.get("quality_grade", "A")}'
                            }
                            high_grade_base_strategies.append(base_params)
                        elif 'rsi_min' in strategy:  # params가 dict가 아닌 경우
                            base_params = {
                                'rsi_min': strategy.get('rsi_min', 30),
                                'rsi_max': strategy.get('rsi_max', 70),
                                'volume_ratio_min': strategy.get('volume_ratio_min', 1.0),
                                'volume_ratio_max': strategy.get('volume_ratio_max', 2.0),
                                'macd_buy_threshold': strategy.get('macd_buy_threshold', 0.01),
                                'macd_sell_threshold': strategy.get('macd_sell_threshold', -0.01),
                                'stop_loss_pct': strategy.get('stop_loss_pct', 0.02),
                                'take_profit_pct': strategy.get('take_profit_pct', 0.05),
                                'type': f'evolved_{strategy.get("quality_grade", "A")}'
                            }
                            high_grade_base_strategies.append(base_params)
                    
                    logger.info(f"  ✅ 고등급 전략 베이스로 {len(high_grade_base_strategies)}개 준비")
                else:
                    logger.info(f"  ℹ️ 기존 고등급 전략 부족({len(existing_strategies) if existing_strategies else 0}개), 기본 템플릿 사용")
            except Exception as e:
                logger.debug(f"⚠️ 기존 전략 로드 실패 (무시): {e}")
            
            # 🎯 시장 레짐별 전략 템플릿 (전문성)
            regime_strategies = {
                'trend': [
                    # 상승 추세 추종 전략
                    {'rsi_min': 40, 'rsi_max': 75, 'volume_ratio_min': 1.2, 'volume_ratio_max': 2.5,
                     'macd_buy_threshold': 0.015, 'macd_sell_threshold': -0.008, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.06},
                    # 하락 추세 추종 전략
                    {'rsi_min': 25, 'rsi_max': 60, 'volume_ratio_min': 1.0, 'volume_ratio_max': 2.2,
                     'macd_buy_threshold': -0.01, 'macd_sell_threshold': 0.02, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.05},
                ],
                'range': [
                    # 박스권 돌파 전략
                    {'rsi_min': 20, 'rsi_max': 80, 'volume_ratio_min': 1.3, 'volume_ratio_max': 3.0,
                     'macd_buy_threshold': 0.02, 'macd_sell_threshold': -0.02, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.04},
                    # 박스권 내 거래 전략
                    {'rsi_min': 30, 'rsi_max': 70, 'volume_ratio_min': 0.8, 'volume_ratio_max': 1.5,
                     'macd_buy_threshold': 0.008, 'macd_sell_threshold': -0.008, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.07},
                ],
                'volatile': [
                    # 고변동성 대응 전략
                    {'rsi_min': 35, 'rsi_max': 65, 'volume_ratio_min': 1.5, 'volume_ratio_max': 4.0,
                     'macd_buy_threshold': 0.025, 'macd_sell_threshold': -0.025, 'stop_loss_pct': 0.04, 'take_profit_pct': 0.1},
                    # 안정적 변동성 대응
                    {'rsi_min': 38, 'rsi_max': 62, 'volume_ratio_min': 1.1, 'volume_ratio_max': 2.0,
                     'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.055},
                ],
                'neutral': [
                    # 보수적 균형 전략
                    {'rsi_min': 32, 'rsi_max': 68, 'volume_ratio_min': 1.0, 'volume_ratio_max': 2.0,
                     'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05},
                    # 공격적 균형 전략
                    {'rsi_min': 28, 'rsi_max': 72, 'volume_ratio_min': 1.2, 'volume_ratio_max': 2.3,
                     'macd_buy_threshold': 0.012, 'macd_sell_threshold': -0.012, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.06},
                ]
            }
            
            # 기본 전략 타입 (18개)
            basic_strategies = [
                # 보수적 전략들 (3개)
                {'rsi_min': 32, 'rsi_max': 68, 'volume_ratio_min': 1.0, 'volume_ratio_max': 2.0, 'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05, 'type': 'conservative'},
                {'rsi_min': 35, 'rsi_max': 65, 'volume_ratio_min': 0.9, 'volume_ratio_max': 1.9, 'macd_buy_threshold': 0.008, 'macd_sell_threshold': -0.008, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.045, 'type': 'conservative'},
                {'rsi_min': 30, 'rsi_max': 70, 'volume_ratio_min': 1.1, 'volume_ratio_max': 2.1, 'macd_buy_threshold': 0.012, 'macd_sell_threshold': -0.012, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.055, 'type': 'conservative'},
                
                # 공격적 전략들 (3개)
                {'rsi_min': 25, 'rsi_max': 75, 'volume_ratio_min': 1.2, 'volume_ratio_max': 2.5, 'macd_buy_threshold': 0.015, 'macd_sell_threshold': -0.015, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.07, 'type': 'aggressive'},
                {'rsi_min': 20, 'rsi_max': 80, 'volume_ratio_min': 1.3, 'volume_ratio_max': 2.8, 'macd_buy_threshold': 0.02, 'macd_sell_threshold': -0.02, 'stop_loss_pct': 0.035, 'take_profit_pct': 0.08, 'type': 'aggressive'},
                {'rsi_min': 28, 'rsi_max': 72, 'volume_ratio_min': 1.4, 'volume_ratio_max': 3.0, 'macd_buy_threshold': 0.018, 'macd_sell_threshold': -0.018, 'stop_loss_pct': 0.04, 'take_profit_pct': 0.09, 'type': 'aggressive'},
                
                # 균형 전략들 (3개)
                {'rsi_min': 35, 'rsi_max': 65, 'volume_ratio_min': 1.1, 'volume_ratio_max': 2.2, 'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01, 'stop_loss_pct': 0.025, 'take_profit_pct': 0.05, 'type': 'balanced'},
                {'rsi_min': 33, 'rsi_max': 67, 'volume_ratio_min': 1.05, 'volume_ratio_max': 2.1, 'macd_buy_threshold': 0.008, 'macd_sell_threshold': -0.008, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.048, 'type': 'balanced'},
                {'rsi_min': 36, 'rsi_max': 64, 'volume_ratio_min': 1.15, 'volume_ratio_max': 2.3, 'macd_buy_threshold': 0.012, 'macd_sell_threshold': -0.012, 'stop_loss_pct': 0.022, 'take_profit_pct': 0.052, 'type': 'balanced'},
                
                # 단기 전략들 (3개)
                {'rsi_min': 20, 'rsi_max': 80, 'volume_ratio_min': 1.5, 'volume_ratio_max': 3.0, 'macd_buy_threshold': 0.02, 'macd_sell_threshold': -0.02, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.08, 'type': 'short_term'},
                {'rsi_min': 22, 'rsi_max': 78, 'volume_ratio_min': 1.6, 'volume_ratio_max': 3.2, 'macd_buy_threshold': 0.022, 'macd_sell_threshold': -0.022, 'stop_loss_pct': 0.035, 'take_profit_pct': 0.085, 'type': 'short_term'},
                {'rsi_min': 18, 'rsi_max': 82, 'volume_ratio_min': 1.4, 'volume_ratio_max': 2.8, 'macd_buy_threshold': 0.018, 'macd_sell_threshold': -0.018, 'stop_loss_pct': 0.04, 'take_profit_pct': 0.09, 'type': 'short_term'},
                
                # 장기 전략들 (3개)
                {'rsi_min': 38, 'rsi_max': 62, 'volume_ratio_min': 0.8, 'volume_ratio_max': 1.8, 'macd_buy_threshold': 0.005, 'macd_sell_threshold': -0.005, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.04, 'type': 'long_term'},
                {'rsi_min': 40, 'rsi_max': 60, 'volume_ratio_min': 0.9, 'volume_ratio_max': 1.9, 'macd_buy_threshold': 0.006, 'macd_sell_threshold': -0.006, 'stop_loss_pct': 0.018, 'take_profit_pct': 0.042, 'type': 'long_term'},
                {'rsi_min': 42, 'rsi_max': 58, 'volume_ratio_min': 0.85, 'volume_ratio_max': 1.7, 'macd_buy_threshold': 0.004, 'macd_sell_threshold': -0.004, 'stop_loss_pct': 0.012, 'take_profit_pct': 0.038, 'type': 'long_term'},
                
                # 평균 회귀 전략들 (3개)
                {'rsi_min': 15, 'rsi_max': 85, 'volume_ratio_min': 0.9, 'volume_ratio_max': 1.5, 'macd_buy_threshold': -0.005, 'macd_sell_threshold': 0.005, 'stop_loss_pct': 0.04, 'take_profit_pct': 0.08, 'type': 'mean_reversion'},
                {'rsi_min': 12, 'rsi_max': 88, 'volume_ratio_min': 1.0, 'volume_ratio_max': 1.8, 'macd_buy_threshold': -0.008, 'macd_sell_threshold': 0.008, 'stop_loss_pct': 0.045, 'take_profit_pct': 0.085, 'type': 'mean_reversion'},
                {'rsi_min': 18, 'rsi_max': 82, 'volume_ratio_min': 0.85, 'volume_ratio_max': 1.6, 'macd_buy_threshold': -0.006, 'macd_sell_threshold': 0.006, 'stop_loss_pct': 0.035, 'take_profit_pct': 0.075, 'type': 'mean_reversion'},
            ]
            
            # 💰 시장 레짐별 전문 전략 추가 (6개)
            regime_pro_strategies = []
            for regime, strategies in regime_strategies.items():
                for strategy in strategies:
                    strategy['type'] = regime
                    regime_pro_strategies.append(strategy)
            
            # 모든 전략 합치기 (고등급 베이스 + 레짐별 + 기본)
            all_strategy_types = []
            
            # 1. 고등급 전략 베이스 (최우선)
            if high_grade_base_strategies:
                all_strategy_types.extend(high_grade_base_strategies)
                logger.info(f"  ✅ 고등급 베이스 전략 {len(high_grade_base_strategies)}개 추가")
            
            # 2. 레짐별 전문 전략
            all_strategy_types.extend(regime_pro_strategies)
            
            # 3. 기본 전략
            all_strategy_types.extend(basic_strategies)
            
            default_strategies = []
            for i, strategy_params in enumerate(all_strategy_types):
                # 고등급 전략 베이스인지 확인
                is_evolved = strategy_params.get('type', '').startswith('evolved_')
                
                if is_evolved:
                    # 고등급 전략 베이스: 최소한의 변동 (기존 성과 유지하되 미세 조정)
                    rsi_min = strategy_params['rsi_min'] + random.randint(-1, 1)
                    rsi_max = strategy_params['rsi_max'] + random.randint(-1, 1)
                    volume_ratio_min = max(0.3, strategy_params['volume_ratio_min'] + random.uniform(-0.05, 0.05))
                    volume_ratio_max = min(6.0, strategy_params['volume_ratio_max'] + random.uniform(-0.1, 0.1))
                    macd_buy = strategy_params['macd_buy_threshold'] + random.uniform(-0.001, 0.001)
                    macd_sell = strategy_params['macd_sell_threshold'] + random.uniform(-0.001, 0.001)
                    stop_loss = max(0.008, strategy_params['stop_loss_pct'] + random.uniform(-0.001, 0.001))
                    take_profit = max(0.015, strategy_params['take_profit_pct'] + random.uniform(-0.002, 0.002))
                else:
                    # 기본/레짐별 전략: 큰 변동 (다양성 확보)
                    rsi_min = max(5, strategy_params['rsi_min'] + random.randint(-3, 3))
                    rsi_max = min(95, strategy_params['rsi_max'] + random.randint(-3, 3))
                    volume_ratio_min = max(0.3, strategy_params['volume_ratio_min'] + random.uniform(-0.15, 0.15))
                    volume_ratio_max = min(6.0, strategy_params['volume_ratio_max'] + random.uniform(-0.2, 0.2))
                    macd_buy = strategy_params['macd_buy_threshold'] + random.uniform(-0.003, 0.003)
                    macd_sell = strategy_params['macd_sell_threshold'] + random.uniform(-0.003, 0.003)
                    stop_loss = max(0.008, strategy_params['stop_loss_pct'] + random.uniform(-0.003, 0.003))
                    take_profit = max(0.015, strategy_params['take_profit_pct'] + random.uniform(-0.008, 0.008))
                
                strategy = {
                    'strategy_id': f'{coin}_{interval}_default_{i+1:03d}',
                    'coin': coin,
                    'interval': interval,
                    'rsi_min': rsi_min,
                    'rsi_max': rsi_max,
                    'volume_ratio_min': volume_ratio_min,
                    'volume_ratio_max': volume_ratio_max,
                    'macd_buy_threshold': macd_buy,
                    'macd_sell_threshold': macd_sell,
                    'stop_loss_pct': stop_loss,
                    'take_profit_pct': take_profit,
                    'profit': random.uniform(-0.05, 0.1),
                    'win_rate': random.uniform(0.3, 0.7),
                    'trades_count': random.randint(5, 50),
                    'max_drawdown': random.uniform(0.05, 0.2),
                    'sharpe_ratio': random.uniform(0.5, 2.0),
                    'strategy_type': strategy_params.get('type', 'general'),
                    # 미사용 컬럼 활성화: 패턴 신뢰도/소스/강화 타입
                    'pattern_confidence': random.uniform(0.4, 0.8),
                    'pattern_source': 'evolved_base' if is_evolved else 'template',
                    'enhancement_type': 'selfplay_base' if is_evolved else 'standard'
                }
                default_strategies.append(strategy)
            
            logger.info(f"📊 기본 전략 생성 완료: {len(default_strategies)}개 (다양한 타입 + 레짐별 전문전략)")
            return default_strategies
            
        except Exception as e:
            logger.error(f"❌ 기본 전략 생성 실패: {e}")
            return []
    
    def _update_strategies_from_selfplay(self, coin: str, interval: str, selfplay_result: Dict[str, Any], evolved_strategies: List[Dict[str, Any]] = None):
        """Self-play 결과로 coin_strategies 테이블 성과 지표 및 등급 업데이트"""
        try:
            from rl_pipeline.db.writes import update_strategy_performance
            from rl_pipeline.db.connection_pool import get_optimized_db_connection
            
            # 🔥 evolved_strategies에서 등급 정보 매핑 생성 (전략 ID -> quality_grade)
            # 원본 strategies의 순서와 evolved_strategies의 순서가 일치한다고 가정
            quality_grade_map = {}
            strategy_index_map = {}  # evolved_strategies 인덱스 -> 원본 strategy_id
            all_strategy_ids = []  # 🔥 모든 전략 ID 리스트 (순서 보존)
            
            if evolved_strategies:
                # 원본 strategies는 selfplay 호출 전에 전달되었으므로, 
                # evolved_strategies의 순서로 매핑 가능 (순서가 보존됨)
                for idx, evolved_strategy in enumerate(evolved_strategies):
                    strategy_id = evolved_strategy.get('id')
                    quality_grade = evolved_strategy.get('quality_grade')
                    
                    if strategy_id:
                        all_strategy_ids.append(strategy_id)  # 순서 보존
                        
                        if quality_grade:
                            quality_grade_map[strategy_id] = quality_grade
                            strategy_index_map[idx] = strategy_id
                            
                            # _evolved 접미사 제거된 원본 ID도 매핑
                            if strategy_id.endswith('_evolved'):
                                original_id = strategy_id[:-8]  # '_evolved' 제거
                                quality_grade_map[original_id] = quality_grade
                        else:
                            strategy_index_map[idx] = strategy_id
            
            # Self-play 결과에서 성과 데이터 추출
            cycle_results = selfplay_result.get('cycle_results', [])
            if not cycle_results:
                # traditional_result나 predictive_result에서도 시도
                cycle_results = []
                if selfplay_result.get('traditional_result', {}).get('cycle_results'):
                    cycle_results.extend(selfplay_result['traditional_result']['cycle_results'])
                if selfplay_result.get('predictive_result', {}).get('cycle_results'):
                    cycle_results.extend(selfplay_result['predictive_result']['cycle_results'])
            
            # 🔥 온라인 Self-play 결과 처리 추가 (온라인 결과가 아직 변환되지 않은 경우)
            if not cycle_results:
                # online_result에서 segment_results 추출 및 변환
                try:
                    from rl_pipeline.hybrid.online_data_converter import (
                        extract_online_selfplay_result,
                        convert_online_segments_to_cycle_results
                    )
                    
                    online_segments = extract_online_selfplay_result(selfplay_result)
                    if online_segments:
                        summary = selfplay_result.get('summary', {})
                        # 온라인 결과가 online_result에 직접 있는 경우
                        if not online_segments and selfplay_result.get('online_result'):
                            online_result = selfplay_result.get('online_result', {})
                            online_segments = online_result.get('segment_results', [])
                            summary = online_result.get('summary', {})
                        
                        if online_segments:
                            cycle_results = convert_online_segments_to_cycle_results(online_segments, summary)
                            logger.debug(f"✅ {coin}-{interval}: 온라인 Self-play 결과 변환 완료 ({len(cycle_results)}개 cycle)")
                except ImportError:
                    logger.debug(f"⚠️ 온라인 데이터 변환 모듈 없음 (무시)")
                except Exception as e:
                    logger.debug(f"⚠️ 온라인 Self-play 결과 변환 실패: {e}")
            
            if not cycle_results:
                logger.debug(f"⚠️ {coin}-{interval}: Self-play 결과에 cycle_results 없음")
                return
            
            updated_count = 0
            skipped_count = 0
            
            # 🔥 첫 번째 cycle에서 agent_id -> strategy_id 매핑 생성
            agent_to_strategy_map = {}
            if cycle_results and len(cycle_results) > 0:
                first_cycle = cycle_results[0]
                first_results = first_cycle.get('results', {})
                agent_ids_sorted = sorted(first_results.keys())
                
                # 🔥 predictive_strategy_* 형태의 agent_id를 실제 전략 ID로 매핑
                for idx, agent_id in enumerate(agent_ids_sorted):
                    # 1. 인덱스 기반 매핑 (우선순위 1)
                    if idx < len(all_strategy_ids):
                        agent_to_strategy_map[agent_id] = all_strategy_ids[idx]
                    elif idx in strategy_index_map:
                        agent_to_strategy_map[agent_id] = strategy_index_map[idx]
                    # 2. agent_* 형태 처리
                    elif agent_id.startswith('agent_'):
                        try:
                            agent_idx = int(agent_id.split('_')[1]) - 1
                            if 0 <= agent_idx < len(all_strategy_ids):
                                agent_to_strategy_map[agent_id] = all_strategy_ids[agent_idx]
                            elif agent_idx in strategy_index_map:
                                agent_to_strategy_map[agent_id] = strategy_index_map[agent_idx]
                        except (ValueError, IndexError):
                            pass
                    # 3. predictive_strategy_* 형태 처리 (숫자 추출)
                    elif agent_id.startswith('predictive_strategy_'):
                        try:
                            # predictive_strategy_1 -> 0, predictive_strategy_3 -> 2 등
                            pred_idx = int(agent_id.split('_')[2]) - 1
                            if 0 <= pred_idx < len(all_strategy_ids):
                                agent_to_strategy_map[agent_id] = all_strategy_ids[pred_idx]
                                logger.debug(f"✅ {agent_id} → {all_strategy_ids[pred_idx]} 매핑 성공")
                        except (ValueError, IndexError):
                            pass
                    # 4. agent_id가 이미 strategy_id인 경우
                    if agent_id in quality_grade_map:
                        agent_to_strategy_map[agent_id] = agent_id
                
                logger.debug(f"🔍 {coin}-{interval}: agent_to_strategy 매핑 {len(agent_to_strategy_map)}개 생성")
            
            # 🔥 배치 업데이트로 변경: 모든 성과 데이터 수집 후 한 번에 업데이트
            batch_updates = []  # (strategy_id, performance_data) 튜플 리스트
            
            # 모든 cycle에서 성과 데이터 수집
            for cycle in cycle_results:
                results = cycle.get('results', {})
                
                for agent_id, performance in results.items():
                    try:
                        # 🔥 agent_id -> strategy_id 매핑 사용
                        strategy_id = agent_to_strategy_map.get(agent_id, agent_id)
                        
                        # 매핑이 없으면 agent_id를 그대로 사용 (agent_id가 strategy_id인 경우)
                        if strategy_id == agent_id and agent_id not in quality_grade_map:
                            # quality_grade_map에 없으면 agent_id를 그대로 사용
                            pass
                        
                        # 🔥 전략 ID가 'predictive_strategy_*' 형태면 건너뛰기 (실제 전략 ID가 아님)
                        if strategy_id and strategy_id.startswith('predictive_strategy_'):
                            logger.debug(f"⚠️ {agent_id} (매핑: {strategy_id})는 실제 전략 ID가 아니므로 건너뜀")
                            skipped_count += 1
                            continue
                        
                        # 성과 지표 추출 및 변환
                        # performance는 agent.get_performance_metrics()의 결과
                        total_pnl = performance.get('total_pnl', 0.0)
                        win_rate = performance.get('win_rate', 0.0)
                        trades_count = performance.get('total_trades', 0)
                        max_drawdown = performance.get('max_drawdown', 0.0)
                        sharpe_ratio = performance.get('sharpe_ratio', 0.0)
                        
                        # profit_factor 계산 (있으면 사용, 없으면 계산)
                        profit_factor = performance.get('profit_factor', 0.0)
                        if profit_factor == 0.0 and trades_count > 0:
                            # 간단한 profit_factor 계산: 총 수익 / 총 손실
                            total_profit = performance.get('total_profit', 0.0)
                            total_loss = performance.get('total_loss', 0.0)
                            if total_loss > 0:
                                profit_factor = abs(total_profit / total_loss)
                        
                        # avg_profit_per_trade 계산
                        avg_profit_per_trade = 0.0
                        if trades_count > 0:
                            avg_profit_per_trade = total_pnl / trades_count
                        
                        # profit을 퍼센트로 변환 (total_pnl이 이미 퍼센트일 수도 있음)
                        # 일반적으로 Self-play에서 total_pnl은 금액이므로 퍼센트로 변환
                        initial_capital = 10000.0  # Self-play 초기 자본
                        profit_pct = (total_pnl / initial_capital) * 100 if total_pnl != 0 else 0.0
                        
                        # 🔥 quality_grade 추가 (evolved_strategies에서 가져온 등급)
                        performance_data = {
                            'profit': profit_pct,  # 퍼센트로 저장
                            'win_rate': win_rate,
                            'trades_count': trades_count,
                            'max_drawdown': max_drawdown,
                            'sharpe_ratio': sharpe_ratio,
                            'profit_factor': profit_factor,
                            'avg_profit_per_trade': avg_profit_per_trade,
                        }
                        
                        # 🔥 quality_grade 추가 (evolved_strategies에서 계산된 등급)
                        if strategy_id in quality_grade_map:
                            performance_data['quality_grade'] = quality_grade_map[strategy_id]
                        elif strategy_id.endswith('_evolved') and strategy_id[:-8] in quality_grade_map:
                            # _evolved 접미사가 있는 경우 원본 ID로 매핑
                            original_id = strategy_id[:-8]
                            performance_data['quality_grade'] = quality_grade_map[original_id]
                        
                        batch_updates.append((strategy_id, performance_data))
                        
                    except Exception as e:
                        logger.warning(f"⚠️ 전략 {agent_id} 성과 데이터 수집 실패: {e}")
                        skipped_count += 1
                        continue
            
            # 🔥 배치 업데이트 실행 (한 번의 연결로 모든 전략 업데이트)
            if batch_updates:
                try:
                    import time
                    import sqlite3
                    
                    with get_optimized_db_connection("strategies") as conn:
                        cursor = conn.cursor()
                        
                        # updated_at 컬럼 존재 여부 확인 (한 번만)
                        cursor.execute("PRAGMA table_info(coin_strategies)")
                        columns = [col[1] for col in cursor.fetchall()]
                        has_updated_at = 'updated_at' in columns
                        
                        # 🔥 존재하는 전략 ID만 필터링 (배치 쿼리로 확인)
                        strategy_ids = [sid for sid, _ in batch_updates]
                        placeholders = ','.join(['?' for _ in strategy_ids])
                        
                        cursor.execute(f"""
                            SELECT id FROM coin_strategies 
                            WHERE id IN ({placeholders}) AND coin = ? AND interval = ?
                        """, strategy_ids + [coin, interval])
                        
                        existing_ids = {row[0] for row in cursor.fetchall()}
                        
                        # 존재하는 전략만 업데이트
                        valid_updates = [(sid, data) for sid, data in batch_updates if sid in existing_ids]
                        
                        if not valid_updates:
                            logger.warning(f"⚠️ {coin}-{interval}: 존재하는 전략이 없어 업데이트 건너뜀")
                            return
                        
                        # 🔥 배치 업데이트 실행 (재시도 로직 포함)
                        max_retries = 3
                        retry_delay = 1.0
                        
                        for attempt in range(max_retries):
                            try:
                                # 트랜잭션 시작
                                for strategy_id, performance_data in valid_updates:
                                    set_clauses = []
                                    values = []
                                    
                                    for key, value in performance_data.items():
                                        if key in ['profit', 'win_rate', 'max_drawdown', 'sharpe_ratio', 
                                                  'profit_factor', 'quality_grade', 'trades_count', 
                                                  'avg_profit_per_trade']:
                                            set_clauses.append(f"{key} = ?")
                                            values.append(value)
                                    
                                    if has_updated_at:
                                        set_clauses.append("updated_at = datetime('now')")
                                    
                                    values.extend([strategy_id, coin, interval])
                                    
                                    query = f"""
                                        UPDATE coin_strategies 
                                        SET {', '.join(set_clauses)} 
                                        WHERE id = ? AND coin = ? AND interval = ?
                                    """
                                    
                                    cursor.execute(query, tuple(values))
                                
                                # 모든 업데이트 성공 시 커밋
                                conn.commit()
                                updated_count = len(valid_updates)
                                logger.info(f"✅ {coin}-{interval}: {updated_count}개 전략 성과 배치 업데이트 완료")
                                break
                                
                            except sqlite3.OperationalError as db_locked_error:
                                if "database is locked" in str(db_locked_error) and attempt < max_retries - 1:
                                    wait_time = retry_delay * (attempt + 1)
                                    logger.warning(f"⚠️ {coin}-{interval} 배치 업데이트 DB 잠금, {wait_time:.1f}초 후 재시도 ({attempt+1}/{max_retries})")
                                    time.sleep(wait_time)
                                    conn.rollback()  # 롤백 후 재시도
                                    continue
                                else:
                                    logger.error(f"❌ {coin}-{interval} 배치 업데이트 최종 실패: {db_locked_error}")
                                    conn.rollback()
                                    raise
                                    
                except Exception as db_error:
                    logger.error(f"❌ {coin}-{interval}: 배치 업데이트 실패: {db_error}")
                    # 개별 업데이트로 폴백 (성능은 느리지만 동작)
                    for strategy_id, performance_data in batch_updates:
                        try:
                            fallback_data = {k: v for k, v in performance_data.items() if k != 'quality_grade'}
                            if fallback_data:
                                update_strategy_performance(strategy_id, fallback_data)
                                updated_count += 1
                        except Exception:
                            skipped_count += 1
            
            if updated_count > 0:
                logger.info(f"✅ {coin}-{interval}: {updated_count}개 전략 성과 업데이트 완료 (Self-play 결과 반영)")
            if skipped_count > 0:
                logger.warning(f"⚠️ {coin}-{interval}: {skipped_count}개 전략 성과 업데이트 건너뜀")
                
        except Exception as e:
            logger.error(f"❌ {coin}-{interval}: Self-play 결과로 전략 성과 업데이트 실패: {e}")
            raise
    
    def _sync_strategy_grades_to_coin_strategies(self, coin: str, interval: str):
        """strategy_grades 테이블의 등급을 coin_strategies.quality_grade에 동기화"""
        try:
            from rl_pipeline.db.connection_pool import get_optimized_db_connection
            
            with get_optimized_db_connection("strategies") as conn:
                cursor = conn.cursor()
                
                # 🔥 strategy_grades에서 등급 조회 (모든 등급 포함)
                cursor.execute("""
                    SELECT strategy_id, grade, predictive_accuracy
                    FROM strategy_grades
                    WHERE coin = ? AND interval = ?
                    ORDER BY strategy_id
                """, (coin, interval))
                
                grade_results = cursor.fetchall()
                
                if not grade_results:
                    logger.debug(f"⚠️ {coin}-{interval}: 동기화할 등급 데이터 없음")
                    return
                
                logger.debug(f"🔍 {coin}-{interval}: strategy_grades에서 {len(grade_results)}개 등급 데이터 발견")
                
                updated_count = 0
                skipped_count = 0
                not_found_count = 0
                
                # updated_at 컬럼 존재 여부 확인
                from rl_pipeline.core.utils import table_exists
                cursor.execute("PRAGMA table_info(coin_strategies)")
                columns = [col[1] for col in cursor.fetchall()]
                has_updated_at = 'updated_at' in columns
                
                # coin_strategies에 실제로 존재하는 전략 ID 확인 (디버깅용)
                cursor.execute("""
                    SELECT id FROM coin_strategies 
                    WHERE coin = ? AND interval = ?
                """, (coin, interval))
                existing_ids = {row[0] for row in cursor.fetchall()}
                logger.debug(f"🔍 {coin}-{interval}: coin_strategies에 존재하는 전략 수: {len(existing_ids)}")
                
                # 🔍 strategy_id 샘플 수집 (디버깅용)
                sample_not_found_ids = []
                
                for strategy_id, grade, predictive_accuracy in grade_results:
                    try:
                        # 전략이 coin_strategies에 존재하는지 확인
                        if strategy_id not in existing_ids:
                            not_found_count += 1
                            # 샘플 ID 수집 (최대 5개)
                            if len(sample_not_found_ids) < 5:
                                sample_not_found_ids.append(strategy_id)
                            skipped_count += 1
                            continue
                        
                        # coin_strategies 테이블 업데이트 (updated_at 컬럼이 있으면 포함)
                        if has_updated_at:
                            cursor.execute("""
                                UPDATE coin_strategies
                                SET quality_grade = ?, updated_at = datetime('now')
                                WHERE id = ? AND coin = ? AND interval = ?
                            """, (grade, strategy_id, coin, interval))
                        else:
                            cursor.execute("""
                                UPDATE coin_strategies
                                SET quality_grade = ?
                                WHERE id = ? AND coin = ? AND interval = ?
                            """, (grade, strategy_id, coin, interval))
                        
                        if cursor.rowcount > 0:
                            updated_count += 1
                            if updated_count <= 5:  # 처음 5개만 상세 로그
                                logger.debug(f"✅ {strategy_id} 등급 동기화: {grade} (정확도: {predictive_accuracy:.2%})")
                        else:
                            # 업데이트되지 않은 경우: 이미 같은 등급이거나 조건 불일치
                            skipped_count += 1
                            if skipped_count <= 3:
                                # 현재 등급 확인
                                cursor.execute("""
                                    SELECT quality_grade FROM coin_strategies 
                                    WHERE id = ? AND coin = ? AND interval = ?
                                """, (strategy_id, coin, interval))
                                current_grade = cursor.fetchone()
                                if current_grade:
                                    logger.debug(f"⚠️ {strategy_id} 등급 변경 없음: 현재={current_grade[0]}, 새={grade}")
                    except Exception as e:
                        logger.warning(f"⚠️ {strategy_id} 등급 동기화 실패: {e}")
                        continue
                
                conn.commit()
                
                if updated_count > 0:
                    logger.info(f"✅ {coin}-{interval} 등급 동기화 완료: {updated_count}개 전략 업데이트 "
                               f"(건너뜀: {skipped_count}개, coin_strategies에 없음: {not_found_count}개)")
                else:
                    if not_found_count > 0:
                        # 🔍 더 자세한 디버깅 정보 제공
                        sample_ids_str = ", ".join(sample_not_found_ids) if sample_not_found_ids else "없음"
                        
                        # 🔧 'unknown'으로 시작하는 ID는 시뮬레이션 self-play 결과로, coin_strategies에 없음이 정상
                        unknown_count = sum(1 for sid in sample_not_found_ids if isinstance(sid, str) and sid.startswith('unknown'))
                        if unknown_count > 0:
                            logger.debug(
                                f"⚠️ {coin}-{interval}: 등급 동기화 대상 없음 "
                                f"(strategy_grades: {len(grade_results)}개, coin_strategies에 없음: {not_found_count}개)\n"
                                f"   📋 누락된 strategy_id 샘플: {sample_ids_str}\n"
                                f"   💡 원인: 시뮬레이션 self-play 결과의 strategy_id (unknown_*)는 coin_strategies에 저장되지 않음 (정상 동작)"
                            )
                        else:
                            # 🔧 Self-play로 테스트된 모든 전략이 coin_strategies에 저장되지 않으므로 정상 동작
                            # 롤업은 rl_episode_summary의 모든 전략에 대해 계산하지만,
                            # coin_strategies에는 진화된 전략만 저장되므로 일부 strategy_id가 없을 수 있음
                            if not_found_count == len(grade_results):
                                # 모든 전략이 없는 경우: Self-play 전략들이 coin_strategies에 저장되지 않은 경우
                                logger.debug(
                                    f"ℹ️ {coin}-{interval}: 등급 동기화 대상 없음 "
                                    f"(strategy_grades: {len(grade_results)}개, coin_strategies에 없음: {not_found_count}개)\n"
                                    f"   📋 누락된 strategy_id 샘플: {sample_ids_str}\n"
                                    f"   💡 원인: Self-play로 테스트된 전략들이 coin_strategies에 저장되지 않음 (정상 동작)\n"
                                    f"   ℹ️ 롤업은 모든 테스트 전략에 대해 계산하지만, coin_strategies에는 진화된 전략만 저장됨"
                                )
                            else:
                                # 일부만 없는 경우: 경고 유지
                                logger.warning(
                                    f"⚠️ {coin}-{interval}: 등급 동기화 부분 실패 "
                                    f"(strategy_grades: {len(grade_results)}개, coin_strategies에 없음: {not_found_count}개)\n"
                                    f"   📋 누락된 strategy_id 샘플: {sample_ids_str}\n"
                                    f"   💡 원인: 일부 롤업 데이터의 strategy_id가 coin_strategies에 존재하지 않음 "
                                    f"(이전 실행 데이터 또는 Self-play 테스트 전략일 수 있음)"
                                )
                    else:
                        logger.debug(f"⚠️ {coin}-{interval}: 등급 동기화 대상 없음 "
                                     f"(모두 이미 동기화됨 또는 조건 불일치, skipped: {skipped_count}개)")
                
        except Exception as e:
            logger.error(f"❌ 등급 동기화 실패: {e}")
    
    def _filter_strategies_by_direction(
        self,
        strategies: List[Dict[str, Any]],
        coin: str,
        interval: str,
        candle_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        실제 캔들 데이터로 빠른 예측 테스트하여 방향성이 있는 전략만 필터링
        
        Args:
            strategies: 필터링할 전략 리스트
            coin: 코인 심볼
            interval: 인터벌
            candle_data: 실제 캔들 데이터
            
        Returns:
            방향성이 있는 전략만 필터링된 리스트
        """
        try:
            # 환경변수로 필터링 활성화 여부 제어 (기본값: false - 다양성 확보)
            enable_filtering = os.getenv('ENABLE_STRATEGY_DIRECTION_FILTERING', 'false').lower() == 'true'
            
            if not enable_filtering:
                logger.debug(f"📊 방향성 필터링 비활성화, 모든 전략 사용 (다양성 확보)")
                return strategies
            
            # 🆕 첫 생성 여부 확인 (기존 전략이 있으면 필터링, 없으면 완화)
            try:
                from rl_pipeline.db.connection_pool import get_optimized_db_connection
                with get_optimized_db_connection("strategies") as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM coin_strategies 
                        WHERE coin = ? AND interval = ?
                    """, (coin, interval))
                    existing_count = cursor.fetchone()[0]
                    
                    if existing_count == 0:
                        # 첫 생성 시 필터링 완화 (50% 이상 통과하면 사용)
                        logger.info(f"📊 {coin}-{interval} 첫 전략 생성, 필터링 완화 모드")
                        strict_mode = False
                    else:
                        # 기존 전략이 있으면 엄격한 필터링
                        logger.debug(f"📊 {coin}-{interval} 기존 전략 {existing_count}개 존재, 엄격한 필터링")
                        strict_mode = True
            except Exception as e:
                logger.debug(f"⚠️ 기존 전략 수 확인 실패: {e}, 엄격한 필터링 사용")
                strict_mode = True
            
            # 캔들 데이터 유효성 체크 (DataFrame 비교 안전하게)
            try:
                if candle_data is None:
                    logger.warning(f"⚠️ 캔들 데이터가 None이므로 필터링 건너뜀")
                    return strategies
                if isinstance(candle_data, pd.DataFrame):
                    if candle_data.empty or len(candle_data) < 10:
                        logger.warning(f"⚠️ 캔들 데이터 부족으로 필터링 건너뜀")
                        return strategies
                else:
                    logger.warning(f"⚠️ 캔들 데이터 타입이 DataFrame이 아님, 필터링 건너뜀")
                    return strategies
            except Exception as e:
                logger.warning(f"⚠️ 캔들 데이터 검증 실패: {e}, 필터링 건너뜀")
                return strategies

            # 🔥 수정: prediction_generator 선택적 import
            try:
                from rl_pipeline.engine.prediction_generator import PredictionGenerator
                prediction_generator = PredictionGenerator()
            except ImportError:
                logger.debug("⚠️ 하이브리드 모드 미사용 (PredictionGenerator 미구현), 방향성 필터링 건너뜀")
                return strategies
            
            # 캔들 데이터에서 샘플 추출 (최근 10개 캔들로 빠른 테스트)
            sample_candles = candle_data.tail(10).copy()
            
            filtered_strategies = []
            skipped_count = 0
            
            # 디버깅: 첫 몇 개 전략의 파라미터 확인
            if strategies:
                sample_params = self._extract_strategy_params_for_prediction(strategies[0])
                logger.debug(f"📊 필터링 테스트 샘플 파라미터: RSI={sample_params.get('rsi_min', 'N/A')}-{sample_params.get('rsi_max', 'N/A')}, "
                           f"MACD_buy={sample_params.get('macd_buy_threshold', 'N/A')}, MACD_sell={sample_params.get('macd_sell_threshold', 'N/A')}")
            
            for strategy in strategies:
                try:
                    # 전략 파라미터 추출
                    strategy_params = self._extract_strategy_params_for_prediction(strategy)
                    
                    # 샘플 캔들로 빠른 예측 테스트 (더 많은 샘플로 테스트)
                    has_direction = False
                    test_count = min(10, len(sample_candles))  # 최대 10개로 증가
                    
                    for idx in range(test_count):
                        candle = sample_candles.iloc[idx]
                        
                        # 시장 상태 추출 (안전한 방식으로 단일 값 추출)
                        def safe_get(candle, key, default):
                            """안전하게 단일 값 추출"""
                            try:
                                val = candle[key]
                                # Series나 DataFrame이면 첫 번째 값 추출
                                if isinstance(val, (pd.Series, pd.DataFrame)):
                                    val = val.iloc[0] if len(val) > 0 else default
                                # numpy array면 첫 번째 값
                                elif hasattr(val, 'item'):
                                    try:
                                        val = val.item()
                                    except (ValueError, AttributeError):
                                        pass
                                # NaN 체크 (pd.isna 대신 직접 체크)
                                if val is None:
                                    return default
                                try:
                                    val_float = float(val)
                                    # NaN, inf 체크
                                    import math
                                    if math.isnan(val_float) or math.isinf(val_float):
                                        return default
                                    return val_float
                                except (ValueError, TypeError):
                                    return default
                            except (KeyError, IndexError, AttributeError, TypeError, ValueError):
                                return default
                        
                        market_state = {
                            'rsi': safe_get(candle, 'rsi', 50.0),
                            'macd': safe_get(candle, 'macd', 0.0),
                            'macd_signal': safe_get(candle, 'macd_signal', 0.0),
                            'volume_ratio': safe_get(candle, 'volume_ratio', 1.0),
                            'mfi': safe_get(candle, 'mfi', 50.0),
                            'atr': safe_get(candle, 'atr', 0.02),
                            'adx': safe_get(candle, 'adx', 25.0),
                            'price': safe_get(candle, 'close', 0.0)
                        }
                        
                        # 예측 생성
                        prediction = prediction_generator.generate_prediction(
                            strategy=strategy_params,
                            market_state=market_state,
                            interval=interval,
                            entry_price=market_state['price'],
                            state_key=f"{coin}_{interval}_{idx}"
                        )
                        
                        # 방향성이 있으면 통과 (디버깅 로그 추가)
                        if prediction.predicted_dir != 0:
                            logger.debug(f"✅ 전략 {strategy.get('id', 'unknown')[:30]} dir={prediction.predicted_dir} 통과")
                            has_direction = True
                            break
                        # 디버깅: 첫 번째 전략의 예측 결과 로그
                        elif strategy == strategies[0] and idx == 0:
                            logger.debug(f"🔍 첫 전략 예측 테스트: dir={prediction.predicted_dir}, conf={prediction.predicted_conf:.3f}, "
                                       f"RSI={market_state['rsi']:.1f}, MACD={market_state['macd']:.6f}")
                    
                    if has_direction:
                        filtered_strategies.append(strategy)
                    else:
                        skipped_count += 1
                        # 디버깅: 첫 번째 전략만 상세 로그
                        if strategy == strategies[0]:
                            logger.debug(f"❌ 첫 전략 필터링 제외: dir=0만 나옴, 파라미터={strategy_params}")
                        
                except Exception as e:
                    # 예외 발생 시 전략 포함 (안전하게 처리)
                    logger.debug(f"⚠️ 전략 {strategy.get('id', 'unknown')} 필터링 테스트 실패: {e}, 포함")
                    filtered_strategies.append(strategy)
            
            if skipped_count > 0:
                logger.info(f"📊 방향성 필터링: {skipped_count}개 전략 제외 (dir=0만 나옴)")
            
            # 🆕 첫 생성 시 완화 모드: 필터링 후 전략이 적어도 일정 비율 이상 있으면 사용
            if not strict_mode:
                # 첫 생성 시: 30% 이상 통과했으면 사용, 아니면 원본 사용
                if len(filtered_strategies) >= len(strategies) * 0.3:
                    logger.info(f"📊 첫 생성 완화 모드: {len(filtered_strategies)}개 전략 통과 ({len(filtered_strategies)/len(strategies)*100:.1f}%), 사용")
                    return filtered_strategies
                else:
                    logger.info(f"📊 첫 생성 완화 모드: 필터링 후 {len(filtered_strategies)}개만 남음 (전체의 {len(filtered_strategies)/len(strategies)*100:.1f}%), 원본 전략 사용")
                    return strategies
            else:
                # 엄격한 모드: 필터링 후 전략이 없으면 원본 사용 (안전장치)
                if len(filtered_strategies) == 0:
                    logger.warning(f"⚠️ 방향성 필터링 후 전략이 없어서 원본 전략 사용")
                    return strategies
            
            return filtered_strategies
            
        except Exception as e:
            logger.warning(f"⚠️ 방향성 필터링 실패: {e}, 원본 전략 사용")
            return strategies
    
    def _extract_strategy_params_for_prediction(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """전략에서 예측 생성에 필요한 파라미터만 추출"""
        try:
            # strategy_conditions에서 파라미터 추출 (JSON 문자열이면 파싱)
            import json
            strategy_conditions = strategy.get('strategy_conditions', {})
            
            if isinstance(strategy_conditions, str):
                try:
                    strategy_conditions = json.loads(strategy_conditions) if strategy_conditions else {}
                except json.JSONDecodeError:
                    strategy_conditions = {}
            
            # 전략 파라미터 구성 (우선순위: strategy_conditions > strategy 직접 필드 > 기본값)
            # None 체크를 포함하여 실제 값이 있을 때만 사용
            def get_param(key, default):
                # strategy_conditions에서 먼저 찾기
                val = strategy_conditions.get(key) if strategy_conditions else None
                if val is not None:
                    return float(val)
                # strategy 직접 필드에서 찾기
                val = strategy.get(key)
                if val is not None:
                    return float(val)
                return default
            
            params = {
                'rsi_min': get_param('rsi_min', 30.0),
                'rsi_max': get_param('rsi_max', 70.0),
                'volume_ratio_min': get_param('volume_ratio_min', 1.0),
                'volume_ratio_max': get_param('volume_ratio_max', 2.0),
                'macd_buy_threshold': get_param('macd_buy_threshold', 0.01),
                'macd_sell_threshold': get_param('macd_sell_threshold', -0.01),
                'stop_loss_pct': get_param('stop_loss_pct', 0.02),
                'take_profit_pct': get_param('take_profit_pct', 0.05),
            }
            
            return params
            
        except Exception as e:
            logger.debug(f"⚠️ 전략 파라미터 추출 실패: {e}, 기본값 사용")
            return {
                'rsi_min': 30.0,
                'rsi_max': 70.0,
                'volume_ratio_min': 1.0,
                'volume_ratio_max': 2.0,
                'macd_buy_threshold': 0.01,
                'macd_sell_threshold': -0.01,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
            }
    
    def _create_default_analysis_result(self, coin: str, interval: str) -> Any:
        """기본 분석 결과 생성 - 더 현실적인 결과"""
        try:
            import random
            
            # 간단한 분석 결과 클래스 정의
            class SimpleAnalysisResult:
                def __init__(self, coin: str, interval: str):
                    self.coin = coin
                    self.interval = interval
                    self.regime = random.choice(['bullish', 'bearish', 'neutral', 'sideways'])
                    self.fractal_score = random.uniform(0.3, 0.8)
                    self.multi_timeframe_score = random.uniform(0.4, 0.9)
                    self.indicator_cross_score = random.uniform(0.2, 0.7)
                    self.ensemble_score = random.uniform(0.3, 0.8)
                    self.ensemble_confidence = random.uniform(0.5, 0.9)
                    
                    # 최종 시그널 점수 계산
                    self.final_signal_score = (self.fractal_score + self.multi_timeframe_score + 
                                             self.indicator_cross_score + self.ensemble_score) / 4
                    
                    # 시그널 액션 결정
                    if self.final_signal_score > 0.7:
                        self.signal_action = "BUY"
                    elif self.final_signal_score < 0.3:
                        self.signal_action = "SELL"
                    else:
                        self.signal_action = "HOLD"
                    
                    self.signal_confidence = self.ensemble_confidence
                    self.created_at = datetime.now().isoformat()
            
            result = SimpleAnalysisResult(coin, interval)
            logger.info(f"📊 기본 분석 결과 생성: {result.signal_action} (점수: {result.final_signal_score:.3f})")

            # Dict로 변환 (validator 호환성)
            return {
                'coin': result.coin,
                'interval': result.interval,
                'regime': result.regime,
                'fractal_score': result.fractal_score,
                'multi_timeframe_score': result.multi_timeframe_score,
                'indicator_cross_score': result.indicator_cross_score,
                'ensemble_score': result.ensemble_score,
                'ensemble_confidence': result.ensemble_confidence,
                'signal_score': result.final_signal_score,
                'signal_action': result.signal_action,
                'signal_confidence': result.signal_confidence,
                'created_at': result.created_at
            }
            
        except Exception as e:
            logger.error(f"❌ 기본 분석 결과 생성 실패: {e}")
            # 최소한의 결과라도 반환 (dict)
            return {
                'coin': coin,
                'interval': interval,
                'regime': 'neutral',
                'fractal_score': 0.5,
                'multi_timeframe_score': 0.5,
                'indicator_cross_score': 0.5,
                'ensemble_score': 0.5,
                'ensemble_confidence': 0.5,
                'signal_score': 0.5,
                'signal_action': 'HOLD',
                'signal_confidence': 0.5,
                'created_at': datetime.now().isoformat()
            }

# ============================================================================
# 통합된 Learning Results DB 관리
# ============================================================================

@contextmanager
def run_integrated_pipeline(coin: str, interval: str, candle_data: pd.DataFrame) -> PipelineResult:
    """통합된 파이프라인 실행"""
    orchestrator = IntegratedPipelineOrchestrator()
    return orchestrator.run_complete_pipeline(coin, interval, candle_data)
# 기본값은 모두 false로 설정되어 있어서 중요한 로그만 출력됨

