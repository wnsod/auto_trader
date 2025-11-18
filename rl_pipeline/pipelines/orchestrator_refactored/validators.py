"""
Orchestrator 검증 모듈
파이프라인 결과 검증 함수들
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def validate_selfplay_result(result: Dict, coin: str, interval: str) -> Dict[str, Any]:
    """
    예측 Self-play 결과 검증

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
            from ..orchestrator import PREDICTIVE_SELFPLAY_EPISODES
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
        min_intervals: 최소 인터벌 수
        min_total_episodes: 최소 총 에피소드 수

    Returns:
        검증 결과 딕셔너리
    """
    issues = []
    warnings = []
    stats = {}

    try:
        # 1. 최소 인터벌 수 확인
        num_intervals = len(all_interval_selfplay)
        stats['num_intervals'] = num_intervals

        if num_intervals < min_intervals:
            issues.append(f"인터벌 수 부족: {num_intervals}/{min_intervals}")
        elif num_intervals < 3:
            warnings.append(f"인터벌 수가 적음: {num_intervals} (권장: 3개 이상)")

        # 2. 총 에피소드 수 계산
        total_episodes = sum(
            sp.get('episodes', 0) for sp in all_interval_selfplay.values()
        )
        stats['total_episodes'] = total_episodes

        if total_episodes < min_total_episodes:
            issues.append(f"총 에피소드 수 부족: {total_episodes}/{min_total_episodes}")

        # 3. 평균 정확도 계산
        accuracies = [
            sp.get('avg_accuracy', 0) for sp in all_interval_selfplay.values()
            if 'avg_accuracy' in sp
        ]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            stats['avg_accuracy'] = avg_accuracy

            if avg_accuracy < 0.5:
                warnings.append(f"평균 정확도가 낮음: {avg_accuracy:.3f}")
            elif avg_accuracy > 0.95:
                warnings.append(f"평균 정확도가 매우 높음: {avg_accuracy:.3f} (과적합 가능성)")

        # 4. 인터벌별 상세 검증
        interval_stats = {}
        for interval, sp_result in all_interval_selfplay.items():
            interval_validation = validate_selfplay_result(sp_result, coin, interval)

            interval_stats[interval] = {
                'valid': interval_validation['valid'],
                'episodes': sp_result.get('episodes', 0),
                'accuracy': sp_result.get('avg_accuracy', 0),
                'issues_count': len(interval_validation['issues']),
                'warnings_count': len(interval_validation['warnings'])
            }

            # 인터벌별 이슈/경고 집계
            if interval_validation['issues']:
                for issue in interval_validation['issues']:
                    issues.append(f"[{interval}] {issue}")

            if interval_validation['warnings']:
                for warning in interval_validation['warnings']:
                    warnings.append(f"[{interval}] {warning}")

        stats['interval_stats'] = interval_stats

        # 5. 파이프라인 결과와 일치성 확인
        if pipeline_results:
            # 파이프라인 결과 개수와 Self-play 결과 개수 비교
            if len(pipeline_results) != num_intervals:
                warnings.append(f"파이프라인 결과 수({len(pipeline_results)})와 Self-play 결과 수({num_intervals}) 불일치")

        # 6. 품질 점수 계산
        quality_score = calculate_learning_data_quality_score(stats, issues, warnings)
        stats['quality_score'] = quality_score

        if quality_score < 50:
            issues.append(f"데이터 품질 점수가 너무 낮음: {quality_score}/100")
        elif quality_score < 70:
            warnings.append(f"데이터 품질 개선 필요: {quality_score}/100")

    except Exception as e:
        issues.append(f"통합 학습 데이터 검증 중 예외: {str(e)}")
        logger.error(f"통합 학습 데이터 검증 실패: {e}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats.get('quality_score', 0)
    }


def calculate_learning_data_quality_score(stats: Dict, issues: List, warnings: List) -> int:
    """
    학습 데이터 품질 점수 계산

    Args:
        stats: 통계 정보
        issues: 이슈 리스트
        warnings: 경고 리스트

    Returns:
        품질 점수 (0-100)
    """
    score = 100

    # 이슈당 10점 감점
    score -= len(issues) * 10

    # 경고당 3점 감점
    score -= len(warnings) * 3

    # 인터벌 수에 따른 점수 조정
    num_intervals = stats.get('num_intervals', 0)
    if num_intervals >= 4:
        score += 10
    elif num_intervals >= 3:
        score += 5
    elif num_intervals < 2:
        score -= 20

    # 총 에피소드 수에 따른 점수 조정
    total_episodes = stats.get('total_episodes', 0)
    if total_episodes >= 100:
        score += 10
    elif total_episodes >= 50:
        score += 5
    elif total_episodes < 20:
        score -= 15

    # 평균 정확도에 따른 점수 조정
    avg_accuracy = stats.get('avg_accuracy', 0)
    if 0.6 <= avg_accuracy <= 0.85:  # 이상적 범위
        score += 15
    elif 0.5 <= avg_accuracy < 0.6:
        score += 5
    elif avg_accuracy < 0.5:
        score -= 20
    elif avg_accuracy > 0.95:  # 과적합 가능성
        score -= 10

    return max(0, min(100, score))


def validate_global_strategy_pool(
    pool: Dict[str, List],
    coins: List[str],
    intervals: List[str],
    min_strategies_per_interval: int = 10
) -> Dict[str, Any]:
    """
    글로벌 전략 풀 검증

    Args:
        pool: 전략 풀
        coins: 코인 리스트
        intervals: 인터벌 리스트
        min_strategies_per_interval: 인터벌당 최소 전략 수

    Returns:
        검증 결과
    """
    issues = []
    warnings = []
    stats = {}

    try:
        # 총 전략 수
        total_strategies = sum(len(strategies) for strategies in pool.values())
        stats['total_strategies'] = total_strategies

        if total_strategies == 0:
            issues.append("전략 풀이 비어있음")
            return {
                'valid': False,
                'issues': issues,
                'warnings': warnings,
                'stats': stats,
                'quality_score': 0
            }

        # 인터벌별 전략 수 확인
        interval_coverage = {}
        for interval in intervals:
            count = len(pool.get(interval, []))
            interval_coverage[interval] = count

            if count < min_strategies_per_interval:
                issues.append(f"{interval} 인터벌 전략 부족: {count}/{min_strategies_per_interval}")
            elif count < min_strategies_per_interval * 2:
                warnings.append(f"{interval} 인터벌 전략 수가 적음: {count}")

        stats['interval_coverage'] = interval_coverage
        stats['intervals_covered'] = len([i for i, c in interval_coverage.items() if c > 0])
        stats['intervals_expected'] = len(intervals)

        # 코인 커버리지 확인 (전략에서 코인 정보 추출)
        coin_coverage = {}
        for interval, strategies in pool.items():
            for strategy in strategies:
                coin = strategy.get('coin', 'unknown')
                if coin not in coin_coverage:
                    coin_coverage[coin] = 0
                coin_coverage[coin] += 1

        stats['coin_coverage'] = coin_coverage
        stats['coins_covered'] = len(coin_coverage)

        # 품질 점수 계산
        quality_score = 100
        quality_score -= len(issues) * 15
        quality_score -= len(warnings) * 5

        # 커버리지 기반 점수 조정
        coverage_ratio = stats['intervals_covered'] / stats['intervals_expected'] if stats['intervals_expected'] > 0 else 0
        quality_score = int(quality_score * coverage_ratio)

        stats['quality_score'] = max(0, min(100, quality_score))

    except Exception as e:
        issues.append(f"전략 풀 검증 중 예외: {str(e)}")
        logger.error(f"전략 풀 검증 실패: {e}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats.get('quality_score', 0)
    }


def validate_global_strategy_patterns(
    patterns: Dict[str, Any],
    min_patterns_per_interval: int = 3
) -> Dict[str, Any]:
    """
    글로벌 전략 패턴 검증

    Args:
        patterns: 추출된 패턴
        min_patterns_per_interval: 인터벌당 최소 패턴 수

    Returns:
        검증 결과
    """
    issues = []
    warnings = []
    stats = {}

    try:
        # 총 패턴 수
        total_patterns = sum(
            len(p) if isinstance(p, list) else 1
            for p in patterns.values()
        )
        stats['total_patterns'] = total_patterns

        if total_patterns == 0:
            issues.append("패턴이 추출되지 않음")
            return {
                'valid': False,
                'issues': issues,
                'warnings': warnings,
                'stats': stats,
                'quality_score': 0
            }

        # 인터벌별 패턴 수
        interval_patterns = {}
        for key, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                interval_patterns[key] = len(pattern_list)
            else:
                interval_patterns[key] = 1

            if interval_patterns[key] < min_patterns_per_interval:
                warnings.append(f"{key} 패턴 수가 적음: {interval_patterns[key]}/{min_patterns_per_interval}")

        stats['interval_patterns'] = interval_patterns

        # 품질 점수
        quality_score = 100
        quality_score -= len(issues) * 20
        quality_score -= len(warnings) * 5

        # 패턴 다양성 고려
        if total_patterns >= 20:
            quality_score += 10
        elif total_patterns < 10:
            quality_score -= 15

        stats['quality_score'] = max(0, min(100, quality_score))

    except Exception as e:
        issues.append(f"패턴 검증 중 예외: {str(e)}")
        logger.error(f"패턴 검증 실패: {e}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats.get('quality_score', 0)
    }


def validate_global_strategy_quality(
    final_strategies: Dict[str, List],
    intervals: List[str],
    min_strategies_per_interval: int = 5
) -> Dict[str, Any]:
    """
    최종 글로벌 전략 품질 검증

    Args:
        final_strategies: 최종 전략들
        intervals: 인터벌 리스트
        min_strategies_per_interval: 인터벌당 최소 전략 수

    Returns:
        검증 결과
    """
    issues = []
    warnings = []
    stats = {}

    try:
        # 총 전략 수
        total_strategies = sum(len(s) for s in final_strategies.values())
        stats['total_strategies'] = total_strategies

        if total_strategies == 0:
            issues.append("최종 전략이 없음")
            return {
                'valid': False,
                'issues': issues,
                'warnings': warnings,
                'stats': stats,
                'quality_score': 0
            }

        # 인터벌별 전략 수
        interval_coverage = {}
        for interval in intervals:
            count = len(final_strategies.get(interval, []))
            interval_coverage[interval] = count

            if count < min_strategies_per_interval:
                issues.append(f"{interval} 전략 부족: {count}/{min_strategies_per_interval}")

        stats['interval_coverage'] = interval_coverage
        stats['avg_strategies_per_interval'] = total_strategies / len(intervals) if intervals else 0

        # 전략 품질 검증
        high_quality_count = 0
        low_quality_count = 0

        for strategies in final_strategies.values():
            for strategy in strategies:
                # 간단한 품질 체크
                if strategy.get('threshold', 0) > 0.7:
                    high_quality_count += 1
                elif strategy.get('threshold', 0) < 0.3:
                    low_quality_count += 1

        stats['high_quality_strategies'] = high_quality_count
        stats['low_quality_strategies'] = low_quality_count

        if low_quality_count > total_strategies * 0.3:
            warnings.append(f"낮은 품질 전략이 많음: {low_quality_count}/{total_strategies}")

        # 품질 점수
        quality_score = 100
        quality_score -= len(issues) * 20
        quality_score -= len(warnings) * 5

        # 전략 수와 품질 고려
        if total_strategies >= 50:
            quality_score += 10
        elif total_strategies < 20:
            quality_score -= 15

        if high_quality_count > total_strategies * 0.5:
            quality_score += 10

        stats['quality_score'] = max(0, min(100, quality_score))

    except Exception as e:
        issues.append(f"품질 검증 중 예외: {str(e)}")
        logger.error(f"품질 검증 실패: {e}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'quality_score': stats.get('quality_score', 0)
    }