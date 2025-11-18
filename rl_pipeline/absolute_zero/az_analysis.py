"""
Absolute Zero 시스템 - 분석 모듈
전략 분석, 평가, 스코어링 관련 함수들
"""

import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def calculate_global_analysis_data(all_coin_strategies: Dict[str, Any]) -> Dict[str, float]:
    """
    전체 코인 전략의 글로벌 분석 데이터 계산
    """
    try:
        # 프랙탈 점수 계산
        all_strategies = []
        for coin_data in all_coin_strategies.values():
            if 'strategies' in coin_data:
                all_strategies.extend(coin_data['strategies'])

        fractal_score = calculate_fractal_score(all_strategies)

        # 다중 시간대 일관성 계산
        coherence = calculate_multi_timeframe_coherence(all_coin_strategies)

        # 지표 간 교차 검증 계산
        cross_validation = calculate_indicator_cross_validation(all_strategies)

        return {
            'fractal_score': fractal_score,
            'multi_timeframe_coherence': coherence,
            'indicator_cross_validation': cross_validation,
            'total_strategies': len(all_strategies),
            'total_coins': len(all_coin_strategies)
        }

    except Exception as e:
        logger.error(f"글로벌 분석 데이터 계산 실패: {e}")
        return {
            'fractal_score': 0.0,
            'multi_timeframe_coherence': 0.0,
            'indicator_cross_validation': 0.0,
            'total_strategies': 0,
            'total_coins': 0
        }

def calculate_fractal_score(strategies: List[Dict]) -> float:
    """
    전략들의 프랙탈 점수 계산 (자기 유사성 패턴 분석)
    """
    try:
        if len(strategies) < 10:
            return 0.5  # 기본값

        # 전략들의 성과 지표를 배열로 변환
        performance_metrics = []
        for strategy in strategies:
            if all(k in strategy for k in ['win_rate', 'profit_factor', 'sharpe_ratio']):
                performance_metrics.append([
                    float(strategy['win_rate']),
                    float(strategy['profit_factor']),
                    float(strategy.get('sharpe_ratio', 0))
                ])

        if len(performance_metrics) < 5:
            return 0.5

        performance_array = np.array(performance_metrics)

        # 자기 유사성 계산: 다양한 스케일에서 패턴 비교
        scales = [2, 4, 8]
        similarities = []

        for scale in scales:
            if len(performance_array) >= scale * 2:
                # 스케일별로 데이터를 나누어 유사성 비교
                chunks = np.array_split(performance_array, scale)
                chunk_means = [np.mean(chunk, axis=0) for chunk in chunks]

                # 청크 간 상관관계 계산
                for i in range(len(chunk_means) - 1):
                    correlation = np.corrcoef(chunk_means[i], chunk_means[i+1])[0, 1]
                    if not np.isnan(correlation):
                        similarities.append(abs(correlation))

        if similarities:
            # 프랙탈 점수: 0~1 사이의 값 (높을수록 자기 유사성이 강함)
            fractal_score = np.mean(similarities)
            return float(np.clip(fractal_score, 0, 1))

        return 0.5

    except Exception as e:
        logger.debug(f"프랙탈 점수 계산 실패: {e}")
        return 0.5

def calculate_multi_timeframe_coherence(all_coin_strategies: Dict[str, Dict]) -> float:
    """
    다중 시간대 간의 일관성 점수 계산
    """
    try:
        coherence_scores = []

        for coin, coin_data in all_coin_strategies.items():
            if 'strategies' not in coin_data:
                continue

            strategies = coin_data['strategies']
            intervals = {}

            # 인터벌별로 전략 그룹화
            for strategy in strategies:
                interval = strategy.get('interval', 'unknown')
                if interval not in intervals:
                    intervals[interval] = []
                intervals[interval].append(strategy)

            if len(intervals) < 2:
                continue

            # 인터벌 간 일관성 계산
            interval_pairs = []
            interval_names = list(intervals.keys())

            for i in range(len(interval_names)):
                for j in range(i + 1, len(interval_names)):
                    interval1 = interval_names[i]
                    interval2 = interval_names[j]

                    # 두 인터벌 간의 평균 성과 비교
                    avg1 = np.mean([s.get('win_rate', 0) for s in intervals[interval1]])
                    avg2 = np.mean([s.get('win_rate', 0) for s in intervals[interval2]])

                    # 일관성 점수: 차이가 작을수록 높음
                    diff = abs(avg1 - avg2)
                    coherence = 1.0 - min(diff, 1.0)
                    interval_pairs.append(coherence)

            if interval_pairs:
                coherence_scores.append(np.mean(interval_pairs))

        if coherence_scores:
            return float(np.mean(coherence_scores))

        return 0.5

    except Exception as e:
        logger.debug(f"다중 시간대 일관성 계산 실패: {e}")
        return 0.5

def calculate_indicator_cross_validation(strategies: List[Dict]) -> float:
    """
    여러 지표 간의 교차 검증 점수 계산
    """
    try:
        if len(strategies) < 5:
            return 0.5

        indicator_performance = {
            'rsi': [],
            'macd': [],
            'bollinger': [],
            'volume': []
        }

        # 지표별 성과 수집
        for strategy in strategies:
            strategy_name = strategy.get('name', '').lower()
            win_rate = strategy.get('win_rate', 0)

            for indicator in indicator_performance.keys():
                if indicator in strategy_name:
                    indicator_performance[indicator].append(win_rate)

        # 최소 2개 이상의 지표에 데이터가 있어야 함
        active_indicators = {k: v for k, v in indicator_performance.items() if len(v) >= 2}

        if len(active_indicators) < 2:
            return 0.5

        # 지표 간 일관성 계산
        cross_validations = []
        indicator_names = list(active_indicators.keys())

        for i in range(len(indicator_names)):
            for j in range(i + 1, len(indicator_names)):
                ind1 = indicator_names[i]
                ind2 = indicator_names[j]

                avg1 = np.mean(active_indicators[ind1])
                avg2 = np.mean(active_indicators[ind2])

                # 교차 검증 점수: 성과가 비슷할수록 높음
                diff = abs(avg1 - avg2)
                validation_score = 1.0 - min(diff / 0.5, 1.0)  # 0.5를 최대 차이로 정규화
                cross_validations.append(validation_score)

        if cross_validations:
            return float(np.mean(cross_validations))

        return 0.5

    except Exception as e:
        logger.debug(f"지표 간 교차 검증 계산 실패: {e}")
        return 0.5

def validate_strategy_quality(strategy: Dict) -> bool:
    """
    개별 전략의 품질 검증
    """
    try:
        # 최소 요구사항 체크
        min_trades = strategy.get('trades', 0)
        win_rate = strategy.get('win_rate', 0)
        profit_factor = strategy.get('profit_factor', 0)

        # 최소 거래 수 확인
        if min_trades < 5:
            return False

        # 최소 승률 확인
        if win_rate < 0.3:  # 30% 이상
            return False

        # Profit Factor 확인
        if profit_factor < 0.8:  # 0.8 이상
            return False

        return True

    except Exception as e:
        logger.debug(f"전략 품질 검증 실패: {e}")
        return False

def analyze_strategy_distribution(strategies: List[Dict]) -> Dict[str, Any]:
    """
    전략 분포 분석
    """
    try:
        if not strategies:
            return {
                'total': 0,
                'by_interval': {},
                'by_performance': {},
                'quality_distribution': {}
            }

        # 인터벌별 분포
        by_interval = {}
        for strategy in strategies:
            interval = strategy.get('interval', 'unknown')
            by_interval[interval] = by_interval.get(interval, 0) + 1

        # 성과별 분포
        by_performance = {
            'excellent': 0,  # win_rate > 0.6
            'good': 0,       # win_rate > 0.5
            'average': 0,    # win_rate > 0.4
            'poor': 0        # win_rate <= 0.4
        }

        for strategy in strategies:
            win_rate = strategy.get('win_rate', 0)
            if win_rate > 0.6:
                by_performance['excellent'] += 1
            elif win_rate > 0.5:
                by_performance['good'] += 1
            elif win_rate > 0.4:
                by_performance['average'] += 1
            else:
                by_performance['poor'] += 1

        # 품질 분포
        quality_distribution = {
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0
        }

        for strategy in strategies:
            if validate_strategy_quality(strategy):
                profit_factor = strategy.get('profit_factor', 0)
                if profit_factor > 1.5:
                    quality_distribution['high_quality'] += 1
                elif profit_factor > 1.0:
                    quality_distribution['medium_quality'] += 1
                else:
                    quality_distribution['low_quality'] += 1

        return {
            'total': len(strategies),
            'by_interval': by_interval,
            'by_performance': by_performance,
            'quality_distribution': quality_distribution
        }

    except Exception as e:
        logger.error(f"전략 분포 분석 실패: {e}")
        return {
            'total': 0,
            'by_interval': {},
            'by_performance': {},
            'quality_distribution': {}
        }