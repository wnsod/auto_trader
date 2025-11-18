"""
전략 진화 모듈 (Phase 3A)
유전 연산 기반 전략 진화 시스템

기능:
1. 상위 전략 선별 (S/A 등급 또는 상위 20%)
2. 교배(Crossover): 두 부모 전략의 파라미터 조합
3. 변이(Mutation): 민감 파라미터에 가중 적용
4. 다양성 계산 및 탐색 변이
5. 버전 관리 및 DB 저장
"""

import os
import logging
import numpy as np
import random
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from rl_pipeline.db.connection_pool import get_strategy_db_pool
from rl_pipeline.db.reads import fetch_all

logger = logging.getLogger(__name__)

# 환경변수
EVOLUTION_TOP_PERCENT = float(os.getenv('EVOLUTION_TOP_PERCENT', '0.2'))  # 상위 전략 선별 비율
EVOLUTION_MIN_GRADE = os.getenv('EVOLUTION_MIN_GRADE', 'B')  # 최소 등급
MUTATION_STRENGTH = float(os.getenv('MUTATION_STRENGTH', '0.1'))  # 변이 강도 (10%)
MUTATION_PROBABILITY = float(os.getenv('MUTATION_PROBABILITY', '0.3'))  # 변이 확률 (30%)
DIVERSITY_THRESHOLD = float(os.getenv('DIVERSITY_THRESHOLD', '0.3'))  # 최소 다양성 점수
EXPLORATION_MUTATION_STRENGTH = float(os.getenv('EXPLORATION_MUTATION_STRENGTH', '0.3'))  # 탐색 변이 강도


@dataclass
class EvolvedStrategy:
    """진화된 전략"""
    strategy_id: str
    parent_id: str
    version: int
    params: Dict[str, Any]
    mutation_desc: str
    segment_range: Dict[str, int]


class StrategyEvolver:
    """전략 진화 엔진"""
    
    def __init__(self):
        """초기화"""
        self.evolution_history: List[Dict[str, Any]] = []
        
        # 민감한 파라미터 목록 (변이 우선 적용)
        self.sensitive_params = [
            'rsi_min', 'rsi_max',
            'take_profit_pct', 'stop_loss_pct',
            'volume_ratio_min', 'volume_ratio_max',
            'macd_buy_threshold', 'macd_sell_threshold'
        ]
        
        logger.info("✅ Strategy Evolver 초기화 완료")
    
    def select_top_strategies(
        self,
        strategies: List[Dict[str, Any]],
        top_percent: float = EVOLUTION_TOP_PERCENT,
        min_grade: str = EVOLUTION_MIN_GRADE
    ) -> List[Dict[str, Any]]:
        """
        상위 전략 선별
        
        Args:
            strategies: 전략 리스트
            top_percent: 상위 비율 (0.0 ~ 1.0)
            min_grade: 최소 등급
        
        Returns:
            선별된 상위 전략 리스트
        """
        try:
            if not strategies:
                return []
            
            # 등급 우선순위
            grade_order = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5, 'UNKNOWN': 6}
            
            # 필터링: 최소 등급 이상
            filtered = []
            for strategy in strategies:
                grade = strategy.get('quality_grade', 'UNKNOWN')
                if grade == 'UNKNOWN' or grade is None:
                    grade = 'UNKNOWN'
                
                grade_rank = grade_order.get(grade, 999)
                min_grade_rank = grade_order.get(min_grade, 999)
                
                if grade_rank <= min_grade_rank:
                    filtered.append(strategy)
            
            if not filtered:
                logger.warning("⚠️ 최소 등급 이상의 전략이 없습니다")
                return []
            
            # 개선: Consistency Score 계산 및 반영
            for strategy in filtered:
                strategy_id = strategy.get('id')
                if strategy_id:
                    try:
                        # 세그먼트 결과에서 수익률 추출
                        segment_returns = self._fetch_segment_returns(strategy_id)
                        if segment_returns:
                            consistency = self.calculate_consistency_score(segment_returns)
                            strategy['consistency_score'] = consistency
                        else:
                            strategy['consistency_score'] = 0.5  # 기본값
                    except Exception as e:
                        logger.warning(f"⚠️ Consistency Score 계산 실패 ({strategy_id}): {e}")
                        strategy['consistency_score'] = 0.5
                else:
                    strategy['consistency_score'] = 0.5
            
            # 등급, Consistency Score, 성과 기준 정렬
            sorted_strategies = sorted(
                filtered,
                key=lambda s: (
                    grade_order.get(s.get('quality_grade', 'UNKNOWN'), 999),
                    -s.get('consistency_score', 0.5),  # 높을수록 좋음
                    -s.get('profit', 0.0),
                    -s.get('win_rate', 0.0),
                    -s.get('profit_factor', 0.0)
                )
            )
            
            # 상위 비율만 선별
            top_count = max(1, int(len(sorted_strategies) * top_percent))
            top_strategies = sorted_strategies[:top_count]
            
            logger.info(f"✅ 상위 전략 선별: {len(top_strategies)}/{len(strategies)} "
                       f"({top_percent*100:.1f}%, 최소 등급: {min_grade})")
            
            return top_strategies
            
        except Exception as e:
            logger.error(f"❌ 상위 전략 선별 실패: {e}")
            return []
    
    def _fetch_segment_returns(self, strategy_id: str) -> List[float]:
        """
        전략의 세그먼트 수익률 조회
        
        Args:
            strategy_id: 전략 ID
        
        Returns:
            세그먼트 수익률 리스트
        """
        try:
            with get_strategy_db_pool().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT profit FROM segment_scores
                    WHERE strategy_id = ?
                    ORDER BY created_at DESC
                    LIMIT 20
                """, (strategy_id,))
                
                results = cursor.fetchall()
                returns = [float(row[0]) for row in results if row[0] is not None]
                return returns
                
        except Exception as e:
            logger.debug(f"세그먼트 수익률 조회 실패 ({strategy_id}): {e}")
            return []
    
    def calculate_consistency_score(
        self,
        segment_returns: List[float],
        method: str = 'std_inverse'
    ) -> float:
        """
        Consistency Score 계산
        
        Args:
            segment_returns: 세그먼트별 수익률 리스트
            method: 계산 방법 ('std_inverse' 또는 'sharpe')
        
        Returns:
            Consistency Score (0.0 ~ 1.0)
        """
        try:
            if not segment_returns or len(segment_returns) < 2:
                return 0.5  # 기본값
            
            if method == 'std_inverse':
                # 방법 1: 표준편차 역수
                std_dev = np.std(segment_returns)
                consistency = 1 / (1 + std_dev)
                return float(np.clip(consistency, 0.0, 1.0))
            
            elif method == 'sharpe':
                # 방법 2: 샤프 비율 기반 (정규화 필요)
                mean_return = np.mean(segment_returns)
                std_dev = np.std(segment_returns)
                
                if std_dev == 0:
                    return 1.0
                
                sharpe = mean_return / std_dev
                # 정규화 (샤프 비율 -2 ~ +2를 0 ~ 1로 변환)
                normalized = (sharpe + 2) / 4
                return float(np.clip(normalized, 0.0, 1.0))
            
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Consistency Score 계산 실패: {e}")
            return 0.5
    
    def calculate_diversity_score(
        self,
        strategies: List[Dict[str, Any]]
    ) -> float:
        """
        전략 간 다양성 점수 계산
        
        Args:
            strategies: 전략 리스트
        
        Returns:
            다양성 점수 (0.0 ~ 1.0, 높을수록 다양함)
        """
        try:
            if len(strategies) < 2:
                return 1.0  # 전략이 1개면 최대 다양성
            
            # 파라미터 벡터 추출
            param_vectors = []
            for strategy in strategies:
                vector = []
                for param_name in self.sensitive_params:
                    value = strategy.get(param_name, 0.0)
                    # 정규화 (파라미터별 범위 고려)
                    if 'rsi' in param_name:
                        normalized = value / 100.0  # RSI는 0~100
                    elif 'pct' in param_name:
                        normalized = value / 0.1  # TP/SL은 보통 0.01~0.1
                    else:
                        normalized = value / 10.0  # 기본 스케일링
                    vector.append(normalized)
                
                param_vectors.append(vector)
            
            # 유클리드 거리 계산
            distances = []
            for i in range(len(param_vectors)):
                for j in range(i + 1, len(param_vectors)):
                    vec1 = np.array(param_vectors[i])
                    vec2 = np.array(param_vectors[j])
                    distance = np.linalg.norm(vec1 - vec2)
                    distances.append(distance)
            
            if not distances:
                return 0.0
            
            # 평균 거리를 다양성 점수로 사용 (정규화)
            avg_distance = np.mean(distances)
            # 평균 거리 0.5를 기준으로 정규화 (경험적 값)
            diversity = min(1.0, avg_distance / 0.5)
            
            return float(diversity)
            
        except Exception as e:
            logger.error(f"❌ 다양성 점수 계산 실패: {e}")
            return 0.5
    
    def crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        교배: 두 부모 전략의 파라미터 조합
        
        Args:
            parent1: 부모 전략 1
            parent2: 부모 전략 2
        
        Returns:
            자식 전략 파라미터
        """
        try:
            child_params = {}
            
            # 공통 파라미터 병합
            all_params = set(list(parent1.keys()) + list(parent2.keys()))
            
            for param_name in all_params:
                if param_name in ['id', 'coin', 'interval', 'parent_id', 'version']:
                    # 메타데이터는 제외
                    continue
                
                # 랜덤하게 부모 1 또는 부모 2에서 선택
                if random.random() < 0.5:
                    child_params[param_name] = parent1.get(param_name)
                else:
                    child_params[param_name] = parent2.get(param_name)
                
                # 없으면 기본값 유지
                if child_params[param_name] is None:
                    if param_name in parent1:
                        child_params[param_name] = parent1[param_name]
                    elif param_name in parent2:
                        child_params[param_name] = parent2[param_name]
            
            logger.debug(f"✅ 교배 완료: {len(child_params)}개 파라미터")
            return child_params
            
        except Exception as e:
            logger.error(f"❌ 교배 실패: {e}")
            return parent1.copy()  # 실패 시 부모 1 복사
    
    def crossover_with_weight(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        parent1_score: float = None,
        parent2_score: float = None
    ) -> Dict[str, Any]:
        """
        성과 기반 가중치 교배 (개선사항)
        
        Args:
            parent1: 부모 전략 1
            parent2: 부모 전략 2
            parent1_score: 부모 1 성과 점수 (None이면 자동 계산)
            parent2_score: 부모 2 성과 점수 (None이면 자동 계산)
        
        Returns:
            자식 전략 파라미터
        """
        try:
            # 성과 점수가 없으면 계산
            if parent1_score is None:
                parent1_score = self._calculate_parent_score(parent1)
            if parent2_score is None:
                parent2_score = self._calculate_parent_score(parent2)
            
            # 가중치 계산
            total_score = parent1_score + parent2_score
            if total_score > 0:
                p1_weight = parent1_score / total_score
            else:
                p1_weight = 0.5  # 동일 가중치
            
            child_params = {}
            all_params = set(list(parent1.keys()) + list(parent2.keys()))
            
            for param_name in all_params:
                if param_name in ['id', 'coin', 'interval', 'parent_id', 'version']:
                    continue
                
                val1 = parent1.get(param_name)
                val2 = parent2.get(param_name)
                
                if val1 is None:
                    child_params[param_name] = val2
                elif val2 is None:
                    child_params[param_name] = val1
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 수치 파라미터: 가중 평균
                    child_params[param_name] = p1_weight * val1 + (1 - p1_weight) * val2
                else:
                    # 비수치 파라미터: 가중치 기반 선택
                    child_params[param_name] = val1 if random.random() < p1_weight else val2
            
            logger.debug(f"✅ 가중 교배 완료 (p1_weight={p1_weight:.3f})")
            return child_params
            
        except Exception as e:
            logger.error(f"❌ 가중 교배 실패: {e}")
            return self.crossover(parent1, parent2)  # 실패 시 기본 교배
    
    def _calculate_parent_score(self, strategy: Dict[str, Any]) -> float:
        """
        전략 성과 점수 계산
        
        Args:
            strategy: 전략 딕셔너리
        
        Returns:
            성과 점수 (0.0 ~ 1.0)
        """
        try:
            # 등급 기반 점수
            grade_scores = {'S': 1.0, 'A': 0.8, 'B': 0.6, 'C': 0.4, 'D': 0.2, 'F': 0.0, 'UNKNOWN': 0.5}
            grade = strategy.get('quality_grade', 'UNKNOWN')
            grade_score = grade_scores.get(grade, 0.5)
            
            # Profit Factor 점수 (0~5 범위를 0~1로 정규화)
            pf = strategy.get('profit_factor', 0.0)
            pf_score = min(pf / 5.0, 1.0) if pf > 0 else 0.0
            
            # Win Rate 점수
            win_rate = strategy.get('win_rate', 0.0)
            win_rate_score = win_rate / 100.0 if win_rate > 0 else 0.0
            
            # 종합 점수 (가중 평균)
            total_score = grade_score * 0.4 + pf_score * 0.4 + win_rate_score * 0.2
            
            return float(np.clip(total_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"⚠️ 성과 점수 계산 실패: {e}")
            return 0.5  # 기본값
    
    def validate_strategy(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        전략 파라미터 유효성 검증 (강화 버전)
        
        Args:
            params: 전략 파라미터
        
        Returns:
            (유효성 여부, 실패 이유)
        """
        try:
            # 1. RSI 범위 검증 (0-100)
            rsi_min = params.get('rsi_min', 30.0)
            rsi_max = params.get('rsi_max', 70.0)
            if not (0.0 <= rsi_min <= 100.0 and 0.0 <= rsi_max <= 100.0):
                return False, f"RSI 범위 초과: min={rsi_min}, max={rsi_max}"
            if rsi_min >= rsi_max:
                return False, f"RSI min >= max: min={rsi_min}, max={rsi_max}"
            
            # 2. Stop Loss / Take Profit 검증
            stop_loss = params.get('stop_loss_pct', 0.15)
            take_profit = params.get('take_profit_pct', 1.50)

            # Stop Loss 범위: 0.05 ~ 0.30 (5% ~ 30%)
            if not (0.05 <= stop_loss <= 0.30):
                return False, f"Stop Loss 범위 초과 (0.05-0.30): {stop_loss}"

            # Take Profit 범위: 1.20 ~ 2.00 (120% ~ 200%)
            if not (1.20 <= take_profit <= 2.00):
                return False, f"Take Profit 범위 초과 (1.20-2.00): {take_profit}"
            
            if stop_loss >= take_profit:
                return False, f"SL >= TP: SL={stop_loss}, TP={take_profit}"
            
            # 3. Volume Ratio 검증
            vol_min = params.get('volume_ratio_min', 1.0)
            vol_max = params.get('volume_ratio_max', 2.0)
            if vol_min < 0 or vol_max < 0:
                return False, f"Volume Ratio 음수: min={vol_min}, max={vol_max}"
            if vol_min >= vol_max:
                return False, f"Volume Ratio min >= max: min={vol_min}, max={vol_max}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"❌ 전략 검증 실패: {e}")
            return False, str(e)
    
    def mutate(
        self,
        strategy: Dict[str, Any],
        strength: float = MUTATION_STRENGTH,
        probability: float = MUTATION_PROBABILITY
    ) -> Tuple[Dict[str, Any], str]:
        """
        변이: 전략 파라미터에 랜덤 변화 적용
        
        Args:
            strategy: 전략 파라미터
            strength: 변이 강도
            probability: 변이 확률
        
        Returns:
            (변형된 전략, 변이 설명)
        """
        try:
            mutated = strategy.copy()
            mutations = []
            
            # 민감한 파라미터에 우선 적용
            for param_name in self.sensitive_params:
                if param_name not in mutated:
                    continue
                
                if random.random() > probability:
                    continue  # 변이 확률 미충족
                
                old_value = mutated[param_name]
                
                # 파라미터별 변이
                if 'rsi' in param_name:
                    # RSI: ±10% 범위 내 변이
                    change = random.uniform(-strength * 10, strength * 10)
                    new_value = old_value + change
                    new_value = np.clip(new_value, 0.0, 100.0)
                elif 'pct' in param_name:
                    # TP/SL: ±10% 범위 내 변이
                    change = random.uniform(-strength * old_value, strength * old_value)
                    new_value = old_value + change
                    if 'take_profit' in param_name:
                        # Take Profit 범위: 1.20 ~ 2.00 (120% ~ 200%)
                        new_value = np.clip(new_value, 1.20, 2.00)
                    else:
                        # Stop Loss 범위: 0.05 ~ 0.30 (5% ~ 30%)
                        new_value = np.clip(new_value, 0.05, 0.30)
                elif 'volume_ratio' in param_name:
                    # Volume: ±20% 범위 내 변이
                    change = random.uniform(-strength * 2 * old_value, strength * 2 * old_value)
                    new_value = old_value + change
                    new_value = np.clip(new_value, 0.5, 10.0)
                else:
                    # 기타: ±10% 범위 내 변이
                    change = random.uniform(-strength * old_value, strength * old_value)
                    new_value = old_value + change
                
                if abs(new_value - old_value) > 1e-6:  # 의미있는 변화만 기록
                    mutated[param_name] = new_value
                    mutations.append(f"{param_name}: {old_value:.4f}→{new_value:.4f}")
            
            # 파라미터 클리핑 및 검증
            mutated = self._clip_and_fix_parameters(mutated)
            
            mutation_desc = ", ".join(mutations) if mutations else "no mutation"
            
            return mutated, mutation_desc
            
        except Exception as e:
            logger.error(f"❌ 변이 실패: {e}")
            return strategy.copy(), "mutation failed"
    
    def _clip_and_fix_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """파라미터 범위 클리핑 및 자동 수정"""
        fixed = params.copy()
        
        try:
            # RSI 클리핑
            rsi_min = fixed.get('rsi_min', 30.0)
            rsi_max = fixed.get('rsi_max', 70.0)
            fixed['rsi_min'] = max(0.0, min(100.0, rsi_min))
            fixed['rsi_max'] = max(0.0, min(100.0, rsi_max))
            
            # RSI min < max 보장
            if fixed['rsi_min'] >= fixed['rsi_max']:
                fixed['rsi_min'] = max(0.0, fixed['rsi_max'] - 5.0)
            
            # Stop Loss 클리핑 및 단위 변환 (0.05 ~ 0.30)
            stop_loss = fixed.get('stop_loss_pct', 0.15)
            # 🔥 단위 변환: 비정상적으로 큰 값 (> 1.0)이면 백분율로 변환
            if stop_loss > 1.0:
                stop_loss = stop_loss / 100.0  # 예: 18.33 → 0.1833
            fixed['stop_loss_pct'] = max(0.05, min(0.30, stop_loss))

            # Take Profit 클리핑 및 단위 변환 (1.20 ~ 2.00)
            take_profit = fixed.get('take_profit_pct', 1.50)
            # 🔥 단위 변환: 비정상적으로 큰 값 처리
            if take_profit > 100.0:
                take_profit = take_profit / 100.0  # 예: 150 → 1.50
            elif take_profit > 10.0:
                take_profit = take_profit / 10.0  # 예: 15.0 → 1.50
            elif take_profit < 1.0:
                take_profit = take_profit + 1.0  # 예: 0.50 → 1.50 (오래된 형식 변환)
            fixed['take_profit_pct'] = max(1.20, min(2.00, take_profit))
            
            # SL < TP 보장 (SL은 0.05~0.30, TP는 1.20~2.00이므로 정상적으로는 SL < TP)
            # 만약 SL >= TP인 경우, TP를 안전한 값으로 설정
            if fixed['stop_loss_pct'] >= fixed['take_profit_pct']:
                fixed['take_profit_pct'] = 1.50  # 기본값으로 리셋
                fixed['take_profit_pct'] = min(2.00, fixed['take_profit_pct'])
            
            # Volume Ratio 클리핑 (정상 범위: 0.3 ~ 10.0)
            vol_min = fixed.get('volume_ratio_min', 1.0)
            vol_max = fixed.get('volume_ratio_max', 2.0)
            fixed['volume_ratio_min'] = np.clip(vol_min, 0.3, 5.0)
            fixed['volume_ratio_max'] = np.clip(vol_max, 0.5, 10.0)
            
            # Volume min < max 보장
            if fixed['volume_ratio_min'] >= fixed['volume_ratio_max']:
                fixed['volume_ratio_max'] = fixed['volume_ratio_min'] + 0.5
            
            return fixed
            
        except Exception as e:
            logger.error(f"❌ 파라미터 수정 실패: {e}")
            return params
    
    def apply_exploration_mutation(
        self,
        strategies: List[Dict[str, Any]],
        threshold: float = DIVERSITY_THRESHOLD,
        strength: float = EXPLORATION_MUTATION_STRENGTH
    ) -> List[Dict[str, Any]]:
        """
        탐색 변이 적용 (다양성 부족 시)
        
        Args:
            strategies: 전략 리스트
            threshold: 다양성 임계값
            strength: 탐색 변이 강도
        
        Returns:
            변이된 전략 리스트
        """
        try:
            diversity = self.calculate_diversity_score(strategies)
            
            if diversity >= threshold:
                logger.debug(f"✅ 다양성 충분: {diversity:.3f} >= {threshold}")
                return strategies
            
            logger.warning(f"⚠️ 다양성 부족: {diversity:.3f} < {threshold}, 탐색 변이 적용")
            
            # 일부 전략에 큰 변이 강제 적용
            mutated_strategies = []
            mutation_count = max(1, int(len(strategies) * 0.3))  # 30%에 강제 변이
            
            for i, strategy in enumerate(strategies):
                if i < mutation_count:
                    # 큰 변이 적용
                    mutated, desc = self.mutate(strategy, strength=strength, probability=1.0)
                    mutated_strategies.append(mutated)
                    logger.debug(f"  탐색 변이 적용: {desc}")
                else:
                    mutated_strategies.append(strategy)
            
            return mutated_strategies
            
        except Exception as e:
            logger.error(f"❌ 탐색 변이 실패: {e}")
            return strategies
    
    def evolve_strategies(
        self,
        parent_strategies: List[Dict[str, Any]],
        n_children: int = 5,
        segment_range: Optional[Dict[str, int]] = None
    ) -> List[EvolvedStrategy]:
        """
        전략 진화 실행
        
        Args:
            parent_strategies: 부모 전략 리스트
            n_children: 생성할 자식 전략 수
            segment_range: 세그먼트 범위 (선택적)
        
        Returns:
            진화된 전략 리스트
        """
        try:
            if len(parent_strategies) < 2:
                logger.warning("⚠️ 진화를 위해서는 최소 2개 전략이 필요합니다")
                return []
            
            evolved_strategies = []
            
            # 다양성 체크 및 탐색 변이
            parent_strategies = self.apply_exploration_mutation(parent_strategies)
            
            for i in range(n_children):
                # 두 부모 랜덤 선택
                parent1, parent2 = random.sample(parent_strategies, 2)
                
                # 교배 (개선: 성과 기반 가중치 교배 사용)
                # 부모 성과 점수 계산
                parent1_score = self._calculate_parent_score(parent1)
                parent2_score = self._calculate_parent_score(parent2)
                
                # 가중 교배 사용 (성과 좋은 부모 우선)
                child_params = self.crossover_with_weight(parent1, parent2, parent1_score, parent2_score)
                
                # 변이
                mutated_params, mutation_desc = self.mutate(child_params)
                
                # 개선: 진화 품질 검증 및 자동 수정
                is_valid, reason = self.validate_strategy(mutated_params)
                if not is_valid:
                    logger.warning(f"⚠️ 진화된 전략 {i+1}번 유효성 검증 실패: {reason}, 자동 수정 시도")
                    # 파라미터 자동 수정
                    mutated_params = self._clip_and_fix_parameters(mutated_params)
                    
                    # 재검증
                    is_valid, reason = self.validate_strategy(mutated_params)
                    if not is_valid:
                        logger.warning(f"⚠️ 자동 수정 후에도 검증 실패: {reason}, 부모 1 사용")
                        mutated_params = parent1.copy()
                        mutation_desc = "validation_failed_use_parent"
                
                # 전략 ID 생성
                parent1_id = parent1.get('id', 'unknown')
                # version이 TEXT 타입이므로 안전하게 처리
                parent_version = parent1.get('version', 'v1')
                if isinstance(parent_version, str):
                    # "v1", "v2" 등의 문자열에서 숫자 추출
                    version_num = int(parent_version.replace('v', '')) if parent_version.replace('v', '').isdigit() else 1
                else:
                    version_num = int(parent_version) if parent_version else 1
                version_num += 1
                version = f"v{version_num}"
                child_id = f"{parent1_id}_v{version_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # 메타데이터 추가
                mutated_params['id'] = child_id
                mutated_params['parent_id'] = parent1_id
                mutated_params['version'] = version
                mutated_params['coin'] = parent1.get('coin', 'BTC')
                mutated_params['interval'] = parent1.get('interval', '15m')
                
                evolved = EvolvedStrategy(
                    strategy_id=child_id,
                    parent_id=parent1_id,
                    version=version,
                    params=mutated_params,
                    mutation_desc=mutation_desc,
                    segment_range=segment_range or {}
                )
                
                evolved_strategies.append(evolved)
                
                logger.debug(f"✅ 진화된 전략 생성: {child_id} (부모: {parent1_id})")
            
            logger.info(f"✅ {n_children}개 진화된 전략 생성 완료")
            return evolved_strategies
            
        except Exception as e:
            logger.error(f"❌ 전략 진화 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def save_evolved_strategies(
        self,
        evolved_strategies: List[EvolvedStrategy],
        coin: str,
        interval: str
    ) -> int:
        """
        진화된 전략을 DB에 저장
        
        Args:
            evolved_strategies: 진화된 전략 리스트
            coin: 코인 심볼
            interval: 인터벌
        
        Returns:
            저장된 전략 수
        """
        try:
            pool = get_strategy_db_pool()
            saved_count = 0
            
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                
                for evolved in evolved_strategies:
                    try:
                        # coin_strategies에 저장
                        params = evolved.params
                        
                        # 기본 정보
                        strategy_id = evolved.strategy_id
                        parent_id = evolved.parent_id
                        version = evolved.version
                        
                        # JSON 파라미터 생성
                        strategy_conditions = json.dumps({
                            k: v for k, v in params.items()
                            if k not in ['id', 'coin', 'interval', 'parent_id', 'version']
                        })
                        
                        # INSERT 또는 UPDATE
                        cursor.execute("""
                            INSERT OR REPLACE INTO coin_strategies (
                                id, coin, interval, parent_id, version,
                                strategy_type, strategy_conditions,
                                rsi_min, rsi_max, stop_loss_pct, take_profit_pct,
                                volume_ratio_min, volume_ratio_max,
                                macd_buy_threshold, macd_sell_threshold,
                                created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            strategy_id, coin, interval, parent_id, version,
                            'evolved', strategy_conditions,
                            params.get('rsi_min', 30.0),
                            params.get('rsi_max', 70.0),
                            params.get('stop_loss_pct', 0.02),
                            params.get('take_profit_pct', 0.04),
                            params.get('volume_ratio_min', 1.0),
                            params.get('volume_ratio_max', 2.0),
                            params.get('macd_buy_threshold', 0.01),
                            params.get('macd_sell_threshold', -0.01),
                            datetime.now().isoformat()
                        ))
                        
                        # strategy_lineage에 기록
                        segment_range_json = json.dumps(evolved.segment_range)
                        improvement_flag = 1  # 진화된 전략은 기본적으로 개선 플래그
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO strategy_lineage (
                                child_id, parent_id, mutation_desc, segment_range, improvement_flag
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (
                            strategy_id, parent_id, evolved.mutation_desc,
                            segment_range_json, improvement_flag
                        ))
                        
                        saved_count += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠️ 전략 저장 실패 ({evolved.strategy_id}): {e}")
                        continue
                
                conn.commit()
            
            logger.info(f"✅ {saved_count}개 진화된 전략 저장 완료")
            return saved_count
            
        except Exception as e:
            logger.error(f"❌ 진화된 전략 저장 실패: {e}")
            return 0


def evolve_strategies_from_segments(
    coin: str,
    interval: str,
    segment_results: List[Any],
    top_percent: float = EVOLUTION_TOP_PERCENT,
    n_children: int = 5
) -> List[EvolvedStrategy]:
    """
    세그먼트 결과로부터 전략 진화 실행 (편의 함수)
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        segment_results: 세그먼트 결과 리스트
        top_percent: 상위 전략 선별 비율
        n_children: 생성할 자식 전략 수
    
    Returns:
        진화된 전략 리스트
    """
    try:
        # DB에서 전략 조회
        pool = get_strategy_db_pool()
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 최근 세그먼트 결과 기반 전략 ID 추출
            strategy_ids = set()
            for segment_list in segment_results:
                if isinstance(segment_list, list):
                    for segment in segment_list:
                        if hasattr(segment, 'strategy_id'):
                            strategy_ids.add(segment.strategy_id)
            
            if not strategy_ids:
                logger.warning("⚠️ 세그먼트 결과에서 전략 ID를 찾을 수 없습니다")
                return []
            
            # 전략 조회
            placeholders = ','.join(['?' for _ in strategy_ids])
            cursor.execute(f"""
                SELECT * FROM coin_strategies
                WHERE id IN ({placeholders}) AND coin = ? AND interval = ?
            """, list(strategy_ids) + [coin, interval])
            
            rows = cursor.fetchall()
            
            # 컬럼명 추출
            columns = [desc[0] for desc in cursor.description]
            strategies = [dict(zip(columns, row)) for row in rows]
        
        if not strategies:
            logger.warning("⚠️ 진화할 전략이 없습니다")
            return []
        
        # 진화 실행
        evolver = StrategyEvolver()
        top_strategies = evolver.select_top_strategies(strategies, top_percent)
        
        if not top_strategies:
            logger.warning("⚠️ 상위 전략이 없습니다")
            return []
        
        evolved = evolver.evolve_strategies(top_strategies, n_children)
        
        # DB 저장
        saved = evolver.save_evolved_strategies(evolved, coin, interval)
        
        logger.info(f"✅ 전략 진화 완료: {len(evolved)}개 생성, {saved}개 저장")
        
        return evolved
        
    except Exception as e:
        logger.error(f"❌ 전략 진화 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

