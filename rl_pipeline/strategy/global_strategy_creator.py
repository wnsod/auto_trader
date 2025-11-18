"""
글로벌 전략 생성 모듈 - 개선된 버전
- 인터벌별 글로벌 전략 생성
- 등급/예측정확도/방향성/레짐 기반 선별
- 통합 인터벌 글로벌 전략 (등급 가중치)
"""

import logging
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# 등급 점수 매핑
GRADE_SCORES = {'S': 6, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1, 'UNKNOWN': 0}

# 등급 점수 (0.0~1.0)
GRADE_SCORES_NORMALIZED = {'S': 0.95, 'A': 0.85, 'B': 0.75, 'C': 0.65, 'D': 0.55, 'F': 0.45, 'UNKNOWN': 0.5}


def load_strategy_predictive_accuracy(strategy_id: str, coin: str, interval: str) -> float:
    """전략의 예측 정확도 로드"""
    try:
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # strategy_grades 테이블에서 우선 조회
            cursor.execute("""
                SELECT predictive_accuracy
                FROM strategy_grades
                WHERE strategy_id = ? AND coin = ? AND interval = ?
            """, (strategy_id, coin, interval))
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return float(result[0])
            
            # 없으면 rl_strategy_rollup에서 조회
            cursor.execute("""
                SELECT predictive_accuracy
                FROM rl_strategy_rollup
                WHERE strategy_id = ? AND coin = ? AND interval = ?
            """, (strategy_id, coin, interval))
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return float(result[0])
            
            return 0.5  # 기본값
            
    except Exception as e:
        logger.debug(f"⚠️ 예측 정확도 로드 실패 ({strategy_id}): {e}")
        return 0.5


def filter_strategies_for_global(
    strategies: List[Dict[str, Any]],
    coin: str,
    interval: str
) -> List[Dict[str, Any]]:
    """
    글로벌 전략 생성을 위한 전략 선별

    🔥 완전 완화 모드: 백테스트되지 않은 raw 전략도 포함
    - UNKNOWN 등급 허용
    - trades_count = 0도 허용 (아직 백테스트 안 한 전략)
    - 예측 정확도 요구사항 없음
    """
    try:
        # 🔥 필터링 완전 제거 - 모든 전략 포함
        # Self-play로 생성된 raw 전략도 글로벌 전략 생성에 사용

        all_strategies = []

        for strategy in strategies:
            grade = strategy.get('quality_grade') or strategy.get('grade', 'UNKNOWN')
            trades_count = strategy.get('trades_count') or strategy.get('total_trades', 0)
            profit = strategy.get('profit', 0.0)
            win_rate = strategy.get('win_rate', 0.5)

            # 🔥 우선순위 점수 계산 (모든 전략에 대해)
            grade_score = GRADE_SCORES_NORMALIZED.get(grade, 0.5)

            # 백테스트 안 된 전략은 기본 점수 0.5 부여
            if trades_count == 0:
                priority_score = 0.5
            else:
                # profit 정규화 (5% 이상이면 1.0)
                normalized_profit = min(profit / 0.05, 1.0) if profit >= 0 else max(0.0, 1.0 + profit / 0.02)

                priority_score = (
                    grade_score * 0.6 +          # 등급 60%
                    normalized_profit * 0.2 +    # 수익 20%
                    win_rate * 0.2               # 승률 20%
                )

            strategy['_priority_score'] = priority_score
            all_strategies.append(strategy)

        # 우선순위 점수 기준 정렬
        all_strategies.sort(key=lambda x: x.get('_priority_score', 0), reverse=True)

        # 🔥 상위 50%만 선별 (너무 많은 전략 사용 방지)
        top_count = max(10, int(len(all_strategies) * 0.5))  # 최소 10개
        selected_strategies = all_strategies[:top_count]

        logger.info(f"  ✅ [{coin}-{interval}] 필터링: {len(strategies)}개 → {len(selected_strategies)}개 선별 (상위 50%)")

        return selected_strategies

    except Exception as e:
        logger.error(f"❌ 전략 필터링 실패: {e}")
        return []


def cluster_similar_strategies(strategies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """유사한 전략을 클러스터링 (dna_hash 또는 파라미터 유사도 기반)"""
    try:
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for strategy in strategies:
            # DNA Hash 기반 클러스터링 (우선)
            dna_hash = strategy.get('dna_hash')
            if dna_hash:
                clusters[dna_hash].append(strategy)
                continue
            
            # 파라미터 기반 해시 생성
            params = strategy.get('params', {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    params = {}
            
            # 핵심 파라미터만 사용
            key_params = {
                'rsi_min': round(params.get('rsi_min', 30), 1),
                'rsi_max': round(params.get('rsi_max', 70), 1),
                'stop_loss_pct': round(params.get('stop_loss_pct', 0.02), 3),
                'take_profit_pct': round(params.get('take_profit_pct', 0.05), 3),
            }
            
            params_str = json.dumps(key_params, sort_keys=True)
            param_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]
            
            clusters[param_hash].append(strategy)
        
        # 클러스터별 검증 (최소 10개)
        valid_clusters = {}
        for cluster_id, cluster_strategies in clusters.items():
            if len(cluster_strategies) >= 10:
                valid_clusters[cluster_id] = cluster_strategies
            else:
                # 10개 미만: 유사한 클러스터와 병합 시도 (간단하게 무시)
                logger.debug(f"  ⚠️ 클러스터 {cluster_id}: {len(cluster_strategies)}개 (10개 미만, 제외)")
        
        logger.info(f"  ✅ 클러스터링: {len(clusters)}개 클러스터 → {len(valid_clusters)}개 유효 클러스터")
        
        return valid_clusters
        
    except Exception as e:
        logger.error(f"❌ 클러스터링 실패: {e}")
        return {}


def classify_strategy_direction_and_regime(strategy: Dict[str, Any]) -> Tuple[str, str]:
    """전략의 방향성과 레짐 추출"""
    try:
        # 방향성 추출
        direction = 'NEUTRAL'
        
        # 1. pattern_source 확인
        pattern_source = strategy.get('pattern_source', '')
        if pattern_source == 'direction_specialized':
            # strategy_conditions에서 direction 추출
            conditions = strategy.get('strategy_conditions', '{}')
            if isinstance(conditions, str):
                try:
                    conditions = json.loads(conditions)
                except:
                    conditions = {}
            
            direction = conditions.get('direction', 'NEUTRAL')
        
        # 2. params에서 추정
        if direction == 'NEUTRAL':
            params = strategy.get('params', {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    params = {}
            
            rsi_min = params.get('rsi_min', 30)
            rsi_max = params.get('rsi_max', 70)
            
            if rsi_min < 40:
                direction = 'BUY'
            elif rsi_max > 60:
                direction = 'SELL'
        
        # 레짐 추출
        regime = strategy.get('market_condition', 'neutral')
        if not regime or regime == '':
            regime = 'neutral'
        
        return direction, regime
        
    except Exception as e:
        logger.debug(f"⚠️ 방향성/레짐 추출 실패: {e}")
        return 'NEUTRAL', 'neutral'


def _classify_strategy_direction(strategy: Dict[str, Any]) -> str:
    """🔥 전략을 매수/매도 그룹으로 분류 (통합 분석기와 동일한 로직)"""
    try:
        # 1. 명시적 방향성 특화 전략 확인
        pattern_source = strategy.get('pattern_source', '')
        if pattern_source == 'direction_specialized':
            direction = strategy.get('direction', '')
            if direction == 'BUY':
                return 'buy'
            elif direction == 'SELL':
                return 'sell'
        
        # 2. 전략 파라미터 기반 분류
        rsi_min = strategy.get('rsi_min', 30.0)
        rsi_max = strategy.get('rsi_max', 70.0)
        
        # RSI 기준: 낮은 rsi_min (< 35) = 매수 전략, 높은 rsi_max (> 65) = 매도 전략
        if rsi_min < 35:
            buy_score = 1.0 - (rsi_min / 35.0)  # rsi_min이 낮을수록 매수 전략
        else:
            buy_score = 0.0
        
        if rsi_max > 65:
            sell_score = (rsi_max - 65.0) / 25.0  # rsi_max가 높을수록 매도 전략
        else:
            sell_score = 0.0
        
        # MACD 기준 추가
        macd_buy_threshold = strategy.get('macd_buy_threshold', 0.0)
        macd_sell_threshold = strategy.get('macd_sell_threshold', 0.0)
        
        if macd_buy_threshold > 0:
            buy_score += 0.3
        if macd_sell_threshold < 0:
            sell_score += 0.3
        
        # 3. 성과 데이터 기반 분류 (있는 경우)
        performance = strategy.get('performance_metrics', {})
        if isinstance(performance, str):
            import json
            performance = json.loads(performance) if performance else {}
        
        # 매수 거래 성공률이 높으면 매수 전략
        buy_win_rate = performance.get('buy_win_rate', 0.5)
        sell_win_rate = performance.get('sell_win_rate', 0.5)
        
        if buy_win_rate > sell_win_rate + 0.1:
            buy_score += 0.2
        elif sell_win_rate > buy_win_rate + 0.1:
            sell_score += 0.2
        
        # 4. 최종 분류
        if buy_score > sell_score + 0.2:
            return 'buy'
        elif sell_score > buy_score + 0.2:
            return 'sell'
        else:
            return 'neutral'
            
    except Exception as e:
        logger.debug(f"전략 방향 분류 실패 (무시): {e}")
        return 'neutral'

def create_global_strategy_for_interval(
    interval: str,
    interval_strategies: Dict[str, List[Dict[str, Any]]],
    strategy_type: str = 'performance_based'
) -> Optional[Dict[str, Any]]:
    """인터벌별 글로벌 전략 생성"""
    try:
        from rl_pipeline.strategy.analyzer import (
            _analyze_global_params_from_strategies,
            _analyze_common_strategy_patterns
        )
        
        # 필터링 및 클러스터링
        all_filtered_strategies = {}
        for coin, strategies in interval_strategies.items():
            filtered = filter_strategies_for_global(strategies, coin, interval)
            if filtered:
                all_filtered_strategies[coin] = filtered
        
        if not all_filtered_strategies:
            logger.warning(f"⚠️ [{interval}] 선별된 전략 없음")
            return None
        
        # 🔥 전략을 매수/매도 그룹으로 분리
        buy_strategies = {}  # {coin: [strategies]}
        sell_strategies = {}  # {coin: [strategies]}
        neutral_strategies = {}  # {coin: [strategies]}
        
        for coin, strategies in all_filtered_strategies.items():
            buy_list = []
            sell_list = []
            neutral_list = []
            
            for strategy in strategies:
                direction = _classify_strategy_direction(strategy)
                if direction == 'buy':
                    buy_list.append(strategy)
                elif direction == 'sell':
                    sell_list.append(strategy)
                else:
                    neutral_list.append(strategy)
            
            if buy_list:
                buy_strategies[coin] = buy_list
            if sell_list:
                sell_strategies[coin] = sell_list
            if neutral_list:
                neutral_strategies[coin] = neutral_list
        
        # 🔥 매수 그룹과 매도 그룹을 각각 종합하여 글로벌 전략 생성
        global_strategies = []
        
        # 1. 매수 그룹 글로벌 전략
        if buy_strategies:
            buy_global_params = _analyze_global_params_from_strategies(buy_strategies)
            buy_common_patterns = _analyze_common_strategy_patterns(buy_strategies)
            
            buy_strategy_id = f"GLOBAL_{interval}_BUY_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            buy_global_strategy = {
                'id': buy_strategy_id,
                'coin': 'GLOBAL',
                'interval': interval,
                'strategy_type': f'{strategy_type}_buy',
                'params': buy_global_params if strategy_type == 'performance_based' else buy_common_patterns,
                'name': f'Global {strategy_type.capitalize()} Strategy - BUY ({interval})',
                'description': f'매수 특화 글로벌 전략 ({interval}, {len(buy_strategies)}개 코인 종합)',
                'direction': 'BUY',  # 🔥 방향성 명시
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                '_num_coins': len(buy_strategies),
                '_num_strategies': sum(len(s) for s in buy_strategies.values())
            }
            global_strategies.append(buy_global_strategy)
            logger.info(f"✅ [{interval}] 글로벌 매수 전략 생성: {len(buy_strategies)}개 코인, {buy_global_strategy['_num_strategies']}개 전략")
        
        # 2. 매도 그룹 글로벌 전략
        if sell_strategies:
            sell_global_params = _analyze_global_params_from_strategies(sell_strategies)
            sell_common_patterns = _analyze_common_strategy_patterns(sell_strategies)
            
            sell_strategy_id = f"GLOBAL_{interval}_SELL_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sell_global_strategy = {
                'id': sell_strategy_id,
                'coin': 'GLOBAL',
                'interval': interval,
                'strategy_type': f'{strategy_type}_sell',
                'params': sell_global_params if strategy_type == 'performance_based' else sell_common_patterns,
                'name': f'Global {strategy_type.capitalize()} Strategy - SELL ({interval})',
                'description': f'매도 특화 글로벌 전략 ({interval}, {len(sell_strategies)}개 코인 종합)',
                'direction': 'SELL',  # 🔥 방향성 명시
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                '_num_coins': len(sell_strategies),
                '_num_strategies': sum(len(s) for s in sell_strategies.values())
            }
            global_strategies.append(sell_global_strategy)
            logger.info(f"✅ [{interval}] 글로벌 매도 전략 생성: {len(sell_strategies)}개 코인, {sell_global_strategy['_num_strategies']}개 전략")
        
        # 3. 중립 그룹도 포함 (하위 호환성)
        if neutral_strategies and not buy_strategies and not sell_strategies:
            # 매수/매도 그룹이 없으면 중립 그룹으로 글로벌 전략 생성
            global_params = _analyze_global_params_from_strategies(neutral_strategies)
            common_patterns = _analyze_common_strategy_patterns(neutral_strategies)
            
            strategy_id = f"GLOBAL_{interval}_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            global_strategy = {
                'id': strategy_id,
                'coin': 'GLOBAL',
                'interval': interval,
                'strategy_type': strategy_type,
                'params': global_params if strategy_type == 'performance_based' else common_patterns,
                'name': f'Global {strategy_type.capitalize()} Strategy ({interval})',
                'description': f'글로벌 전략 ({interval}, {len(neutral_strategies)}개 코인 종합)',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                '_num_coins': len(neutral_strategies),
                '_num_strategies': sum(len(s) for s in neutral_strategies.values())
            }
            global_strategies.append(global_strategy)
        
        # 🔥 여러 글로벌 전략이 생성된 경우 첫 번째 반환 (하위 호환성 유지)
        # TODO: 향후 여러 전략을 리스트로 반환하도록 개선 가능
        if global_strategies:
            return global_strategies[0]
        else:
            # 매수/매도 그룹이 모두 없으면 기존 방식으로 생성 (하위 호환성)
            global_params = _analyze_global_params_from_strategies(all_filtered_strategies)
            common_patterns = _analyze_common_strategy_patterns(all_filtered_strategies)
            
            # 등급 계산 (평균 등급)
            all_grades = []
            for coin, strategies in all_filtered_strategies.items():
                for s in strategies:
                    grade = s.get('quality_grade') or s.get('grade', 'C')
                    if grade in ['S', 'A', 'B', 'C', 'D', 'F']:
                        all_grades.append(grade)
            
            # 대표 등급 (가장 많은 등급)
            if all_grades:
                from collections import Counter
                grade_counter = Counter(all_grades)
                representative_grade = grade_counter.most_common(1)[0][0]
                
                # 평균 등급 점수
                avg_grade_score = sum(GRADE_SCORES.get(g, 3) for g in all_grades) / len(all_grades)
            else:
                representative_grade = 'B'
                avg_grade_score = 4.0
            
            # 글로벌 전략 생성
            strategy_id = f"GLOBAL_{interval}_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if strategy_type == 'performance_based':
                params = global_params
                name = f'Global Performance Strategy ({interval})'
                description = f'성능 기반 글로벌 전략 ({interval}, {len(all_filtered_strategies)}개 코인 종합)'
            else:
                params = common_patterns
                name = f'Global Pattern Strategy ({interval})'
                description = f'패턴 기반 글로벌 전략 ({interval}, {len(all_filtered_strategies)}개 코인 종합)'
            
            global_strategy = {
                'id': strategy_id,
                'coin': 'GLOBAL',
                'interval': interval,
                'strategy_type': strategy_type,
                'params': params,
                'name': name,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'quality_grade': representative_grade,
                '_avg_grade_score': avg_grade_score,
                '_num_coins': len(all_filtered_strategies),
                '_num_strategies': sum(len(s) for s in all_filtered_strategies.values())
            }
            
            logger.info(f"✅ [{interval}] 글로벌 전략 생성: {name} (등급: {representative_grade}, 전략 수: {global_strategy['_num_strategies']})")
            
            return global_strategy
        
    except Exception as e:
        logger.error(f"❌ [{interval}] 글로벌 전략 생성 실패: {e}")
        return None


def calculate_interval_grade_weights(
    interval_global_strategies: Dict[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """인터벌별 글로벌 전략의 등급 기반 가중치 계산"""
    try:
        interval_grades = {}
        interval_weights = {}
        
        for interval, strategies in interval_global_strategies.items():
            if not strategies:
                continue
            
            # 각 인터벌 글로벌 전략의 등급 추출
            grades = []
            for strategy in strategies:
                grade = strategy.get('quality_grade', 'C')
                if grade in ['S', 'A', 'B', 'C', 'D', 'F']:
                    grades.append(grade)
            
            if not grades:
                grade = 'C'
                grade_score = 3.0
            else:
                # 대표 등급 (가장 많은 등급)
                from collections import Counter
                grade_counter = Counter(grades)
                grade = grade_counter.most_common(1)[0][0]
                
                # 평균 등급 점수
                grade_score = sum(GRADE_SCORES.get(g, 3) for g in grades) / len(grades)
            
            interval_grades[interval] = {
                'grade': grade,
                'grade_score': grade_score,
                'grade_distribution': dict(Counter(grades)) if grades else {}
            }
        
        # 등급 점수 기반 가중치 계산
        total_grade_score = sum(g['grade_score'] for g in interval_grades.values())
        
        if total_grade_score > 0:
            for interval, grade_info in interval_grades.items():
                interval_weights[interval] = grade_info['grade_score'] / total_grade_score
        else:
            # 모두 0이면 균등 가중치
            for interval in interval_global_strategies.keys():
                interval_weights[interval] = 1.0 / len(interval_global_strategies) if interval_global_strategies else 0
        
        # 🔥 소수점 3자리로 포맷팅
        formatted_weights = {k: f"{v:.3f}" for k, v in interval_weights.items()}
        logger.info(f"📊 인터벌별 등급 가중치: {formatted_weights}")
        
        return interval_grades, interval_weights
        
    except Exception as e:
        logger.error(f"❌ 등급 가중치 계산 실패: {e}")
        return {}, {}


def create_global_strategy_all_intervals(
    interval_global_strategies: Dict[str, List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """통합 인터벌 글로벌 전략 생성 (등급 가중치 적용)"""
    try:
        if not interval_global_strategies:
            return None

        # 1. 각 인터벌별 글로벌 전략 점수 계산
        interval_scores = {}
        for interval, strategies in interval_global_strategies.items():
            if not strategies:
                continue

            # 각 전략의 점수 추출 (등급, 수익, 승률 종합)
            scores = []
            for strategy in strategies:
                grade = strategy.get('quality_grade', 'C')
                grade_score = GRADE_SCORES_NORMALIZED.get(grade, 0.5)

                profit = strategy.get('profit', 0.0)
                win_rate = strategy.get('win_rate', 0.5)

                # 전략 점수 계산
                normalized_profit = min(profit / 0.05, 1.0) if profit >= 0 else max(0.0, 1.0 + profit / 0.02)
                strategy_score = (
                    grade_score * 0.6 +
                    normalized_profit * 0.2 +
                    win_rate * 0.2
                )
                scores.append(strategy_score)

            interval_scores[interval] = sum(scores) / len(scores) if scores else 0.5

        # 2. 등급별 가중치 계산
        interval_grades, interval_weights = calculate_interval_grade_weights(interval_global_strategies)

        if not interval_weights:
            logger.warning("⚠️ 등급 가중치 계산 실패, 균등 가중치 사용")
            for interval in interval_scores.keys():
                interval_weights[interval] = 1.0 / len(interval_scores) if interval_scores else 0

        # 3. 등급 가중치 적용하여 최종 점수 계산
        final_score = sum(
            interval_scores[interval] * interval_weights.get(interval, 0)
            for interval in interval_scores.keys()
        )

        # 4. 종합 등급 결정
        all_grades = [g['grade'] for g in interval_grades.values()]
        if 'S' in all_grades:
            overall_grade = 'S'
        elif 'A' in all_grades:
            overall_grade = 'A'
        elif 'B' in all_grades:
            overall_grade = 'B'
        else:
            overall_grade = 'C'

        # 5. 통합 인터벌 글로벌 전략 생성
        all_intervals_strategy = {
            'id': f"GLOBAL_all_intervals_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'coin': 'GLOBAL',
            'interval': 'all_intervals',
            'strategy_type': 'multi_interval_grade_weighted',
            'params': {
                'interval_scores': interval_scores,
                'interval_grades': {iv: g['grade'] for iv, g in interval_grades.items()},
                'interval_weights': interval_weights,
                'final_score': final_score
            },
            'name': 'Global Multi-Interval Strategy (Grade-Weighted)',
            'description': f'등급 기반 가중치로 통합된 멀티 인터벌 글로벌 전략 ({len(interval_global_strategies)}개 인터벌)',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'quality_grade': overall_grade,
            'meta_analysis': {
                'interval_grades': interval_grades,
                'interval_scores': interval_scores,
                'grade_weighted_score': final_score
            }
        }

        logger.info(f"✅ 통합 인터벌 글로벌 전략 생성: {overall_grade}등급 (점수: {final_score:.3f})")

        return all_intervals_strategy

    except Exception as e:
        logger.error(f"❌ 통합 인터벌 글로벌 전략 생성 실패: {e}")
        return None


def create_regime_specific_global_strategy(
    interval: str,
    interval_strategies: Dict[str, List[Dict[str, Any]]],
    regime: str
) -> Optional[Dict[str, Any]]:
    """레짐별 글로벌 전략 생성"""
    try:
        from rl_pipeline.strategy.analyzer import _analyze_global_params_from_strategies

        # 해당 레짐의 전략만 필터링
        regime_strategies = {}
        for coin, strategies in interval_strategies.items():
            filtered = [s for s in strategies if s.get('market_condition') == regime]
            if filtered:
                regime_strategies[coin] = filtered

        if not regime_strategies:
            return None

        # 파라미터 분석
        global_params = _analyze_global_params_from_strategies(regime_strategies)

        # 평균 성능 계산
        all_profits = []
        all_win_rates = []
        all_grades = []

        for strategies in regime_strategies.values():
            for s in strategies:
                all_profits.append(s.get('profit', 0.0))
                all_win_rates.append(s.get('win_rate', 0.5))
                grade = s.get('quality_grade') or s.get('grade', 'C')
                if grade in ['S', 'A', 'B', 'C', 'D', 'F']:
                    all_grades.append(grade)

        avg_profit = sum(all_profits) / len(all_profits) if all_profits else 0.0
        avg_win_rate = sum(all_win_rates) / len(all_win_rates) if all_win_rates else 0.5

        # 대표 등급
        if all_grades:
            from collections import Counter
            representative_grade = Counter(all_grades).most_common(1)[0][0]
        else:
            representative_grade = 'B'

        strategy_id = f"GLOBAL_{interval}_{regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        regime_strategy = {
            'id': strategy_id,
            'coin': 'GLOBAL',
            'interval': interval,
            'strategy_type': f'regime_specific_{regime}',
            'params': global_params,
            'name': f'Global {regime.title()} Strategy ({interval})',
            'description': f'{regime} 레짐 특화 글로벌 전략 ({interval})',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'quality_grade': representative_grade,
            'market_condition': regime,
            'profit': avg_profit,
            'win_rate': avg_win_rate,
            '_num_strategies': sum(len(s) for s in regime_strategies.values())
        }

        logger.info(f"✅ [{interval}] {regime} 레짐 글로벌 전략 생성 (등급: {representative_grade}, 전략 수: {regime_strategy['_num_strategies']})")

        return regime_strategy

    except Exception as e:
        logger.debug(f"❌ [{interval}] {regime} 레짐 글로벌 전략 생성 실패: {e}")
        return None


def create_risk_profile_global_strategy(
    interval: str,
    interval_strategies: Dict[str, List[Dict[str, Any]]],
    risk_profile: str
) -> Optional[Dict[str, Any]]:
    """리스크 프로파일별 글로벌 전략 생성"""
    try:
        from rl_pipeline.strategy.analyzer import _analyze_global_params_from_strategies

        # 리스크 프로파일에 따라 전략 선별
        profile_strategies = {}

        for coin, strategies in interval_strategies.items():
            filtered = []
            for s in strategies:
                params = s.get('params', {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except:
                        params = {}

                stop_loss = params.get('stop_loss_pct', 0.02)
                take_profit = params.get('take_profit_pct', 0.05)
                risk_reward = take_profit / stop_loss if stop_loss > 0 else 2.5

                # 리스크 프로파일 필터링
                if risk_profile == 'conservative' and stop_loss <= 0.015 and risk_reward >= 3.0:
                    filtered.append(s)
                elif risk_profile == 'moderate' and 0.015 < stop_loss <= 0.025 and 2.0 <= risk_reward < 3.0:
                    filtered.append(s)
                elif risk_profile == 'aggressive' and stop_loss > 0.025:
                    filtered.append(s)

            if filtered:
                profile_strategies[coin] = filtered

        if not profile_strategies:
            return None

        # 파라미터 분석
        global_params = _analyze_global_params_from_strategies(profile_strategies)

        # 평균 성능 계산
        all_profits = []
        all_win_rates = []
        all_grades = []

        for strategies in profile_strategies.values():
            for s in strategies:
                all_profits.append(s.get('profit', 0.0))
                all_win_rates.append(s.get('win_rate', 0.5))
                grade = s.get('quality_grade') or s.get('grade', 'C')
                if grade in ['S', 'A', 'B', 'C', 'D', 'F']:
                    all_grades.append(grade)

        avg_profit = sum(all_profits) / len(all_profits) if all_profits else 0.0
        avg_win_rate = sum(all_win_rates) / len(all_win_rates) if all_win_rates else 0.5

        # 대표 등급
        if all_grades:
            from collections import Counter
            representative_grade = Counter(all_grades).most_common(1)[0][0]
        else:
            representative_grade = 'B'

        strategy_id = f"GLOBAL_{interval}_{risk_profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        risk_strategy = {
            'id': strategy_id,
            'coin': 'GLOBAL',
            'interval': interval,
            'strategy_type': f'risk_profile_{risk_profile}',
            'params': global_params,
            'name': f'Global {risk_profile.title()} Strategy ({interval})',
            'description': f'{risk_profile} 리스크 프로파일 글로벌 전략 ({interval})',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'quality_grade': representative_grade,
            'profit': avg_profit,
            'win_rate': avg_win_rate,
            '_num_strategies': sum(len(s) for s in profile_strategies.values())
        }

        logger.info(f"✅ [{interval}] {risk_profile} 리스크 프로파일 글로벌 전략 생성 (등급: {representative_grade})")

        return risk_strategy

    except Exception as e:
        logger.debug(f"❌ [{interval}] {risk_profile} 리스크 프로파일 글로벌 전략 생성 실패: {e}")
        return None


def create_enhanced_interval_strategies(
    interval: str,
    interval_strategies: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """인터벌별 다양한 타입의 글로벌 전략 생성 (확장: 12가지 타입)"""
    strategies = []

    # 🔥 기본 strategy_types (12가지로 확장)
    strategy_types = [
        'performance_based',
        'pattern_based',
        'risk_adjusted',
        'consistency_based',
        'sharpe_optimized',
        'drawdown_minimized',
        'volatility_adaptive',
        'momentum_based',
        'mean_reversion',        # 추가
        'trend_following',       # 추가
        'breakout_focused',      # 추가
        'scalping_optimized'     # 추가
    ]

    logger.info(f"🔥 [{interval}] 다양한 전략 타입 생성 시작 ({len(strategy_types)}가지)")

    for strategy_type in strategy_types:
        try:
            strategy = create_global_strategy_for_interval(interval, interval_strategies, strategy_type)
            if strategy:
                strategies.append(strategy)
        except Exception as e:
            logger.debug(f"  ⚠️ [{interval}] {strategy_type} 전략 생성 실패: {e}")

    logger.info(f"  ✅ [{interval}] 기본 전략: {len(strategies)}개 생성")

    # 레짐별 전략 (주요 레짐 3개만 생성하여 효율화)
    major_regimes = ['bullish', 'bearish', 'neutral']

    for regime in major_regimes:
        regime_strategy = create_regime_specific_global_strategy(interval, interval_strategies, regime)
        if regime_strategy:
            strategies.append(regime_strategy)

    logger.info(f"  ✅ [{interval}] 레짐별 전략: {len([s for s in strategies if 'regime_specific' in s.get('strategy_type', '')])}개 생성")

    # 리스크 프로파일별 전략 (3가지)
    risk_profiles = ['conservative', 'moderate', 'aggressive']

    for profile in risk_profiles:
        risk_strategy = create_risk_profile_global_strategy(interval, interval_strategies, profile)
        if risk_strategy:
            strategies.append(risk_strategy)

    logger.info(f"  ✅ [{interval}] 리스크 프로파일별 전략: {len([s for s in strategies if 'risk_profile' in s.get('strategy_type', '')])}개 생성")

    logger.info(f"🎉 [{interval}] 총 {len(strategies)}개 글로벌 전략 생성 완료")

    return strategies

