"""
전략 등급 동적 업데이트 모듈
레짐 라우팅 및 통합 분석 결과를 반영하여 전략 등급을 업데이트

개선 사항:
- 예측 정확도 기반 상대평가 통합
- 코인-인터벌-레짐별 그룹 등급 업데이트
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class StrategyGradeUpdater:
    """전략 등급 동적 업데이트"""
    
    def __init__(self):
        logger.info("🚀 전략 등급 업데이터 초기화")
    
    def update_grades_from_routing_results(
        self,
        coin: str,
        interval: str,
        routing_results: List[Any],
        min_regime_performance: float = 0.6,
        grade_boost_threshold: float = 0.75
    ) -> Dict[str, Dict[str, Any]]:
        """🔥 레짐 라우팅 결과를 반영한 등급 업데이트
        
        Args:
            coin: 코인 심볼
            interval: 인터벌
            routing_results: 레짐 라우팅 결과 리스트
            min_regime_performance: 등급 상승을 위한 최소 성과 (기본: 0.6)
            grade_boost_threshold: 등급 상승을 위한 임계값 (기본: 0.75)
        
        Returns:
            {strategy_id: {'old_grade': ..., 'new_grade': ..., 'reason': ...}}
        """
        try:
            logger.info(f"📊 [{coin}-{interval}] 레짐 라우팅 결과 기반 등급 업데이트 시작")
            
            # 전략별 레짐 성과 집계
            strategy_regime_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            
            for result in routing_results:
                try:
                    strategy_id = result.routed_strategy.get('id') or result.routed_strategy.get('strategy_id')
                    if not strategy_id:
                        continue
                    
                    strategy_regime_performance[strategy_id].append({
                        'regime': result.regime,
                        'performance': result.regime_performance,
                        'routing_score': result.routing_score,
                        'confidence': result.routing_confidence
                    })
                except Exception as e:
                    logger.warning(f"⚠️ 라우팅 결과 파싱 실패: {e}", exc_info=True)
                    continue
            
            # 전략별 등급 업데이트 계산
            grade_updates: Dict[str, Dict[str, Any]] = {}
            
            for strategy_id, performances in strategy_regime_performance.items():
                try:
                    # 현재 등급 가져오기
                    current_grade = self._get_current_grade(strategy_id, coin, interval)
                    
                    # 평균 성과 계산
                    avg_performance = sum(p['performance'] for p in performances) / len(performances) if performances else 0.0
                    avg_routing_score = sum(p['routing_score'] for p in performances) / len(performances) if performances else 0.0
                    
                    # 레짐 적합도 계산 (다양한 레짐에서 좋은 성과를 보이는지)
                    regime_diversity = len(set(p['regime'] for p in performances))
                    regime_fitness = avg_performance * (1.0 + 0.1 * min(regime_diversity, 3))  # 최대 3개 레짐까지 보너스
                    
                    # 새 등급 계산
                    new_grade, reason = self._calculate_new_grade_from_routing(
                        current_grade, avg_performance, avg_routing_score, regime_fitness,
                        min_regime_performance, grade_boost_threshold
                    )
                    
                    if new_grade != current_grade:
                        grade_updates[strategy_id] = {
                            'old_grade': current_grade,
                            'new_grade': new_grade,
                            'reason': reason,
                            'avg_performance': avg_performance,
                            'regime_fitness': regime_fitness
                        }
                        logger.info(f"  📈 {strategy_id}: {current_grade} → {new_grade} ({reason})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {strategy_id} 등급 계산 실패: {e}")
                    continue
            
            logger.info(f"✅ [{coin}-{interval}] 등급 업데이트 완료: {len(grade_updates)}개 전략")
            return grade_updates
            
        except Exception as e:
            logger.error(f"❌ 등급 업데이트 실패: {e}", exc_info=True)
            return {}
    
    def update_grades_from_analysis_results(
        self,
        coin: str,
        interval: str,
        analysis_result: Any,
        strategies: List[Dict[str, Any]],
        min_cross_interval_score: float = 0.65
    ) -> Dict[str, Dict[str, Any]]:
        """🔥 통합 분석 결과를 반영한 등급 업데이트 (크로스 인터벌 성과)
        
        Args:
            coin: 코인 심볼
            interval: 인터벌
            analysis_result: 통합 분석 결과 (CoinSignalScore)
            strategies: 전략 리스트
            min_cross_interval_score: 크로스 인터벌 점수 임계값 (기본: 0.65)
        
        Returns:
            {strategy_id: {'old_grade': ..., 'new_grade': ..., 'reason': ...}}
        """
        try:
            logger.info(f"📊 [{coin}-{interval}] 통합 분석 결과 기반 등급 업데이트 시작")
            
            # 분석 결과에서 크로스 인터벌 성과 추출
            # (실제로는 context_analysis에서 가져와야 하지만, 현재 구조상 간소화)
            
            grade_updates: Dict[str, Dict[str, Any]] = {}
            
            # 최종 시그널 점수가 높으면 관련 전략들의 등급 상승 고려
            final_score = analysis_result.final_signal_score if hasattr(analysis_result, 'final_signal_score') else 0.5
            signal_confidence = analysis_result.signal_confidence if hasattr(analysis_result, 'signal_confidence') else 0.5
            
            # 고등급 전략들이 기여한 정도 계산
            high_grade_count = sum(1 for s in strategies if s.get('grade') in ['S', 'A'] or s.get('quality_grade') in ['S', 'A'])
            high_grade_ratio = high_grade_count / len(strategies) if strategies else 0.0
            
            # 시그널 점수와 신뢰도가 모두 높으면 관련 전략들 등급 상승
            if final_score >= 0.7 and signal_confidence >= 0.7 and high_grade_ratio >= 0.3:
                for strategy in strategies[:10]:  # 상위 10개만
                    strategy_id = strategy.get('id') or strategy.get('strategy_id')
                    if not strategy_id:
                        continue
                    
                    current_grade = strategy.get('grade') or strategy.get('quality_grade', 'C')
                    
                    # 고등급 전략이 크로스 인터벌에서 좋은 성과를 보이면 추가 상승
                    if current_grade in ['S', 'A']:
                        # 이미 고등급이므로 유지
                        continue
                    elif current_grade == 'B' and final_score >= 0.75:
                        new_grade = 'A'
                        grade_updates[strategy_id] = {
                            'old_grade': current_grade,
                            'new_grade': new_grade,
                            'reason': f'크로스 인터벌 우수 성과 (시그널 점수: {final_score:.2f})',
                            'final_score': final_score,
                            'signal_confidence': signal_confidence
                        }
                        logger.info(f"  📈 {strategy_id}: {current_grade} → {new_grade} (크로스 인터벌 우수 성과)")
            
            logger.info(f"✅ [{coin}-{interval}] 통합 분석 기반 등급 업데이트 완료: {len(grade_updates)}개 전략")
            return grade_updates
            
        except Exception as e:
            logger.error(f"❌ 통합 분석 기반 등급 업데이트 실패: {e}", exc_info=True)
            return {}
    
    def apply_grade_updates(
        self,
        coin: str,
        interval: str,
        grade_updates: Dict[str, Dict[str, Any]],
        update_db: bool = True
    ) -> int:
        """등급 업데이트를 데이터베이스에 적용"""
        if not grade_updates:
            return 0
        
        if not update_db:
            return len(grade_updates)
        
        try:
            from rl_pipeline.db.connection_pool import get_optimized_db_connection
            from rl_pipeline.core.env import config
            
            # DB 경로 확인 및 로깅
            db_path = config.get_strategy_db_path(coin)
            
            updated_count = 0
            with get_optimized_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                from rl_pipeline.core.utils import table_exists
                
                for strategy_id, update_info in grade_updates.items():
                    try:
                        new_grade = update_info['new_grade']
                        
                        # 1. strategies 테이블 업데이트 (ID만 사용)
                        if table_exists(cursor, "strategies"):
                            # strategies 테이블은 ID가 PK이므로 ID만으로 업데이트 가능
                            # 불필요한 symbol/interval 조건 제거하여 매칭 실패 방지
                            cursor.execute("""
                                UPDATE strategies
                                SET quality_grade = ?, updated_at = datetime('now')
                                WHERE id = ?
                            """, (new_grade, strategy_id))
                            
                            if cursor.rowcount > 0:
                                updated_count += 1
                                logger.debug(f"✅ strategies.{strategy_id} 등급 업데이트: {update_info['old_grade']} → {new_grade}")
                        
                        # 2. strategy_grades 테이블 업데이트
                        if table_exists(cursor, "strategy_grades"):
                            # 컬럼 확인
                            cursor.execute("PRAGMA table_info(strategy_grades)")
                            cols = [row[1] for row in cursor.fetchall()]
                            
                            # WHERE 절 구성
                            where_clause = "WHERE strategy_id = ?"
                            params = [new_grade, int(datetime.now().timestamp()), strategy_id]
                            
                            # symbol/coin 컬럼 처리
                            if 'symbol' in cols:
                                where_clause += " AND symbol = ?"
                                params.append(coin)
                            elif 'coin' in cols:
                                where_clause += " AND coin = ?"
                                params.append(coin)
                                
                            # interval 처리
                            if 'interval' in cols:
                                where_clause += " AND interval = ?"
                                params.append(interval)

                            cursor.execute(f"""
                                UPDATE strategy_grades
                                SET grade = ?, updated_at = ?
                                {where_clause}
                            """, tuple(params))
                            
                            if cursor.rowcount > 0 and updated_count == 0:
                                updated_count += 1
                                logger.debug(f"✅ strategy_grades.{strategy_id} 등급 업데이트 완료")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ {strategy_id} 등급 업데이트 개별 실패: {e}")
                        continue
                
                conn.commit()
            
            if updated_count > 0:
                logger.info(f"✅ [{coin}-{interval}] DB 등급 업데이트 완료: {updated_count}개 전략 (경로: {db_path})")
            else:
                logger.warning(f"⚠️ [{coin}-{interval}] DB 등급 업데이트 실패: 0개 업데이트 (대상 {len(grade_updates)}개, 경로: {db_path})")
                
            return updated_count
            
        except Exception as e:
            logger.error(f"❌ DB 등급 업데이트 전체 실패: {e}", exc_info=True)
            return 0
    
    def _get_current_grade(self, strategy_id: str, coin: str, interval: str) -> str:
        """현재 전략 등급 조회 (테이블 존재 확인 포함)"""
        try:
            from rl_pipeline.db.connection_pool import get_optimized_db_connection
            from rl_pipeline.core.utils import safe_query_one, table_exists
            from rl_pipeline.core.env import config
            
            # 🔥 코인별 DB 경로 명시적 사용
            db_path = config.get_strategy_db_path(coin)
            
            with get_optimized_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                # strategies 테이블에서 조회
                if table_exists(cursor, "strategies"):
                    # 컬럼 확인
                    cursor.execute("PRAGMA table_info(strategies)")
                    cols = [row[1] for row in cursor.fetchall()]
                    has_symbol = 'symbol' in cols
                    has_coin = 'coin' in cols
                    
                    query = "SELECT quality_grade FROM strategies WHERE id = ?"
                    params = [strategy_id]
                    
                    if has_symbol:
                        query += " AND symbol = ?"
                        params.append(coin)
                    elif has_coin:
                        query += " AND coin = ?"
                        params.append(coin)
                        
                    # interval 컬럼이 있다고 가정
                    if 'interval' in cols:
                        query += " AND interval = ?"
                        params.append(interval)
                        
                    result = safe_query_one(cursor, query, tuple(params), table_name="strategies")
                    if result and result[0]:
                        return result[0]
                
                # strategy_grades 테이블에서 조회 시도
                if table_exists(cursor, "strategy_grades"):
                    cursor.execute("PRAGMA table_info(strategy_grades)")
                    cols = [row[1] for row in cursor.fetchall()]
                    has_symbol = 'symbol' in cols
                    has_coin = 'coin' in cols
                    
                    query = "SELECT grade FROM strategy_grades WHERE strategy_id = ?"
                    params = [strategy_id]
                    
                    if has_symbol:
                        query += " AND symbol = ?"
                        params.append(coin)
                    elif has_coin:
                        query += " AND coin = ?"
                        params.append(coin)
                        
                    if 'interval' in cols:
                        query += " AND interval = ?"
                        params.append(interval)
                        
                    result = safe_query_one(cursor, query, tuple(params), table_name="strategy_grades")
                    if result and result[0]:
                        return result[0]
            
            return 'C'  # 기본값
            
        except Exception as e:
            logger.warning(f"⚠️ 등급 조회 실패 ({strategy_id}): {e}", exc_info=True)
            return 'C'
    
    def _calculate_new_grade_from_routing(
        self,
        current_grade: str,
        avg_performance: float,
        avg_routing_score: float,
        regime_fitness: float,
        min_performance: float,
        boost_threshold: float
    ) -> Tuple[str, str]:
        """라우팅 결과 기반 새 등급 계산"""
        try:
            grade_order = ['F', 'D', 'C', 'B', 'A', 'S']
            current_index = grade_order.index(current_grade) if current_grade in grade_order else 2

            # 성과 기반 등급 조정
            if avg_performance >= boost_threshold and regime_fitness >= 0.8:
                # 우수한 성과 → 등급 상승
                if current_index < len(grade_order) - 1:
                    new_index = min(current_index + 1, len(grade_order) - 1)
                    new_grade = grade_order[new_index]
                    reason = f'레짐 라우팅 우수 성과 (성과: {avg_performance:.2%}, 적합도: {regime_fitness:.2f})'
                    return new_grade, reason

            elif avg_performance < min_performance and avg_routing_score < 0.4:
                # 낮은 성과 → 등급 하락
                if current_index > 0:
                    new_index = max(current_index - 1, 0)
                    new_grade = grade_order[new_index]
                    reason = f'레짐 라우팅 낮은 성과 (성과: {avg_performance:.2%}, 라우팅 점수: {avg_routing_score:.2f})'
                    return new_grade, reason

            # 등급 유지
            return current_grade, '성과 기준 만족'

        except Exception as e:
            logger.warning(f"⚠️ 등급 계산 실패: {e}", exc_info=True)
            return current_grade, '계산 실패'

    def update_grades_with_relative_evaluation(
        self,
        coin: str,
        interval: str,
        regime: str,
        strategies: List[Dict[str, Any]],
        update_db: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        🆕 상대평가 기반 등급 업데이트 (예측 정확도 중심)

        Args:
            coin: 코인 심볼
            interval: 인터벌
            regime: 레짐
            strategies: 전략 리스트
            update_db: DB 업데이트 여부

        Returns:
            {strategy_id: {'old_grade': ..., 'new_grade': ..., 'composite_score': ...}}
        """
        try:
            from rl_pipeline.core.strategy_grading import RelativeGrading

            logger.info(f"📊 [{coin}-{interval}-{regime}] 상대평가 기반 등급 업데이트 시작")

            # 상대평가로 등급 계산
            strategy_scores = RelativeGrading.assign_grades_by_group(
                strategies, coin, interval, regime
            )

            if not strategy_scores:
                logger.warning(f"⚠️ [{coin}-{interval}-{regime}] 등급 계산 결과 없음")
                return {}

            # 등급 변경 내역 수집
            grade_updates = {}
            for score in strategy_scores:
                # 현재 등급 조회
                old_grade = self._get_current_grade(score.strategy_id, coin, interval)

                if score.grade != old_grade:
                    grade_updates[score.strategy_id] = {
                        'old_grade': old_grade,
                        'new_grade': score.grade,
                        'composite_score': score.composite_score,
                        'prediction_accuracy': score.prediction_accuracy,
                        'signal_precision': score.signal_precision,
                        'reason': f'상대평가 (종합점수: {score.composite_score:.3f})'
                    }
                    logger.info(
                        f"  📈 {score.strategy_id}: {old_grade} → {score.grade} "
                        f"(점수: {score.composite_score:.3f}, 예측: {score.prediction_accuracy:.2%})"
                    )

            # DB 업데이트
            if update_db and grade_updates:
                updated_count = self.apply_grade_updates(coin, interval, grade_updates, update_db=True)
                logger.info(f"✅ [{coin}-{interval}-{regime}] {updated_count}개 전략 등급 업데이트 완료")

            return grade_updates

        except Exception as e:
            logger.error(f"❌ 상대평가 등급 업데이트 실패: {e}", exc_info=True)
            return {}

    def batch_update_all_groups(
        self,
        all_strategies: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
        update_db: bool = True
    ) -> Dict[str, int]:
        """
        🆕 모든 코인-인터벌-레짐 그룹에 대해 일괄 등급 업데이트

        Args:
            all_strategies: {coin: {interval: {regime: [strategies]}}}
            update_db: DB 업데이트 여부

        Returns:
            {group_key: updated_count}
        """
        try:
            logger.info("🔄 전체 그룹 일괄 등급 업데이트 시작")

            update_results = {}
            total_updated = 0

            for coin, intervals in all_strategies.items():
                for interval, regimes in intervals.items():
                    for regime, strategies in regimes.items():
                        group_key = f"{coin}-{interval}-{regime}"

                        try:
                            grade_updates = self.update_grades_with_relative_evaluation(
                                coin, interval, regime, strategies, update_db
                            )

                            updated_count = len(grade_updates)
                            update_results[group_key] = updated_count
                            total_updated += updated_count

                        except Exception as e:
                            logger.warning(f"⚠️ {group_key} 업데이트 실패: {e}")
                            update_results[group_key] = 0
                            continue

            logger.info(f"✅ 전체 그룹 일괄 업데이트 완료: 총 {total_updated}개 전략 업데이트")
            return update_results

        except Exception as e:
            logger.error(f"❌ 일괄 업데이트 실패: {e}", exc_info=True)
            return {}
