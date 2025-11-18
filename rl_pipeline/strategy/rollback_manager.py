"""
롤백 관리 모듈 (Phase 5)
전략 성능 하락 시 롤백 처리

기능:
1. 성과 하락 감지
2. 부모 전략으로 롤백
3. 전략 동결 (진화 중단)
"""

import os
import logging
from typing import Dict, List, Any, Optional

from rl_pipeline.db.connection_pool import get_strategy_db_pool

logger = logging.getLogger(__name__)

# 환경변수
DEGRADATION_WINDOW = int(os.getenv('DEGRADATION_WINDOW', '5'))  # 최근 N 세그먼트 확인
DEGRADATION_THRESHOLD = float(os.getenv('DEGRADATION_THRESHOLD', '0.2'))  # 20% 이상 하락 감지
MIN_SEGMENTS_FOR_DETECTION = int(os.getenv('MIN_SEGMENTS_FOR_DETECTION', '3'))  # 최소 세그먼트 수


class StrategyRollbackManager:
    """전략 성능 하락 시 롤백 처리"""
    
    def __init__(self):
        """초기화"""
        self.pool = get_strategy_db_pool()
        logger.info("✅ Strategy Rollback Manager 초기화 완료")
    
    def detect_degradation(
        self,
        strategy_id: str,
        window: int = DEGRADATION_WINDOW
    ) -> bool:
        """
        최근 N 세그먼트 성과 하락 감지
        
        Args:
            strategy_id: 전략 ID
            window: 확인할 세그먼트 수
        
        Returns:
            하락 감지 여부
        """
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 최근 세그먼트 결과 조회
                cursor.execute("""
                    SELECT profit, pf, mdd, created_at
                    FROM segment_scores
                    WHERE strategy_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (strategy_id, window))
                
                segments = cursor.fetchall()
                
                if len(segments) < MIN_SEGMENTS_FOR_DETECTION:
                    logger.debug(f"⚠️ {strategy_id}: 세그먼트 수 부족 ({len(segments)} < {MIN_SEGMENTS_FOR_DETECTION})")
                    return False
                
                # 성과 추출
                profits = [row[0] for row in segments]
                profit_factors = [row[1] for row in segments]
                
                # 최근 절반과 이전 절반 비교
                mid = len(segments) // 2
                recent_profits = profits[:mid]
                previous_profits = profits[mid:]
                
                if not previous_profits or not recent_profits:
                    return False
                
                recent_avg = sum(recent_profits) / len(recent_profits)
                previous_avg = sum(previous_profits) / len(previous_profits)
                
                # 하락 비율 계산
                if previous_avg == 0:
                    degradation_ratio = 1.0 if recent_avg < 0 else 0.0
                else:
                    degradation_ratio = (previous_avg - recent_avg) / abs(previous_avg)
                
                is_degraded = degradation_ratio >= DEGRADATION_THRESHOLD
                
                if is_degraded:
                    logger.warning(f"⚠️ {strategy_id} 성과 하락 감지: "
                                 f"이전={previous_avg:.2f}, 최근={recent_avg:.2f}, "
                                 f"하락율={degradation_ratio:.2%}")
                
                return is_degraded
                
        except Exception as e:
            logger.error(f"❌ 성과 하락 감지 실패 ({strategy_id}): {e}")
            return False
    
    def rollback_to_parent(self, strategy_id: str) -> bool:
        """
        부모 전략으로 복구
        
        Args:
            strategy_id: 복구할 전략 ID
        
        Returns:
            복구 성공 여부
        """
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 부모 전략 조회
                cursor.execute("""
                    SELECT parent_id FROM strategy_lineage
                    WHERE child_id = ?
                """, (strategy_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"⚠️ {strategy_id}: 부모 전략 정보 없음")
                    return False
                
                parent_id = result[0]
                
                # 부모 전략 파라미터 조회
                cursor.execute("""
                    SELECT * FROM coin_strategies
                    WHERE id = ?
                """, (parent_id,))
                
                parent_row = cursor.fetchone()
                if not parent_row:
                    logger.error(f"❌ {strategy_id}: 부모 전략({parent_id})을 찾을 수 없음")
                    return False
                
                # 컬럼명 추출
                columns = [desc[0] for desc in cursor.description]
                parent_strategy = dict(zip(columns, parent_row))
                
                # 현재 전략을 부모 파라미터로 업데이트
                cursor.execute("""
                    UPDATE coin_strategies
                    SET rsi_min = ?,
                        rsi_max = ?,
                        stop_loss_pct = ?,
                        take_profit_pct = ?,
                        volume_ratio_min = ?,
                        volume_ratio_max = ?,
                        macd_buy_threshold = ?,
                        macd_sell_threshold = ?,
                        version = version - 1,
                        parent_id = ?
                    WHERE id = ?
                """, (
                    parent_strategy.get('rsi_min', 30.0),
                    parent_strategy.get('rsi_max', 70.0),
                    parent_strategy.get('stop_loss_pct', 0.02),
                    parent_strategy.get('take_profit_pct', 0.04),
                    parent_strategy.get('volume_ratio_min', 1.0),
                    parent_strategy.get('volume_ratio_max', 2.0),
                    parent_strategy.get('macd_buy_threshold', 0.01),
                    parent_strategy.get('macd_sell_threshold', -0.01),
                    parent_id,
                    strategy_id
                ))
                
                conn.commit()
                logger.info(f"✅ {strategy_id}: 부모 전략({parent_id})으로 롤백 완료")
                return True
                
        except Exception as e:
            logger.error(f"❌ 롤백 실패 ({strategy_id}): {e}")
            return False
    
    def freeze_strategy(self, strategy_id: str) -> bool:
        """
        더 이상 진화하지 않도록 동결
        
        Args:
            strategy_id: 동결할 전략 ID
        
        Returns:
            동결 성공 여부
        """
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # is_active 컬럼 확인 및 업데이트
                # (컬럼이 없을 수도 있으므로 안전하게 처리)
                try:
                    cursor.execute("""
                        UPDATE coin_strategies
                        SET is_active = 0
                        WHERE id = ?
                    """, (strategy_id,))
                    
                    if cursor.rowcount == 0:
                        # 컬럼이 없으면 quality_grade로 표시
                        cursor.execute("""
                            UPDATE coin_strategies
                            SET quality_grade = 'FROZEN'
                            WHERE id = ?
                        """, (strategy_id,))
                    
                    conn.commit()
                    logger.info(f"✅ {strategy_id}: 전략 동결 완료")
                    return True
                    
                except Exception as col_error:
                    # 컬럼이 없으면 quality_grade로 표시
                    cursor.execute("""
                        UPDATE coin_strategies
                        SET quality_grade = 'FROZEN'
                        WHERE id = ?
                    """, (strategy_id,))
                    conn.commit()
                    logger.info(f"✅ {strategy_id}: 전략 동결 완료 (quality_grade 사용)")
                    return True
                
        except Exception as e:
            logger.error(f"❌ 전략 동결 실패 ({strategy_id}): {e}")
            return False
    
    def check_and_rollback(self, strategy_id: str) -> bool:
        """
        성과 하락 감지 및 자동 롤백 (통합 함수)
        
        Args:
            strategy_id: 확인할 전략 ID
        
        Returns:
            롤백 실행 여부
        """
        try:
            if self.detect_degradation(strategy_id):
                logger.warning(f"⚠️ {strategy_id}: 성과 하락 감지, 롤백 실행")
                
                # 롤백 시도
                if self.rollback_to_parent(strategy_id):
                    return True
                else:
                    # 롤백 실패 시 동결
                    logger.warning(f"⚠️ {strategy_id}: 롤백 실패, 전략 동결")
                    return self.freeze_strategy(strategy_id)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 자동 롤백 실패 ({strategy_id}): {e}")
            return False

