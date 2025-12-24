"""
트랜잭션 관리 모듈 (Phase 5)
전략 진화 관련 DB 작업을 원자적으로 처리

기능:
1. 진화된 전략 저장 시 원자성 보장
2. 세그먼트 결과 저장 시 원자성 보장
3. 실패 시 자동 롤백
"""

import logging
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from rl_pipeline.db.connection_pool import get_strategy_db_pool
from rl_pipeline.core.errors import DBWriteError

logger = logging.getLogger(__name__)


class EvolutionTransactionManager:
    """전략 진화 관련 DB 작업을 원자적으로 처리"""
    
    def __init__(self):
        """초기화"""
        self.pool = get_strategy_db_pool()
        logger.info("✅ Evolution Transaction Manager 초기화 완료")
    
    @contextmanager
    def transaction(self):
        """트랜잭션 컨텍스트 매니저"""
        with self.pool.get_connection() as conn:
            try:
                yield conn
                conn.commit()
                logger.debug("✅ 트랜잭션 커밋 완료")
            except Exception as e:
                conn.rollback()
                logger.error(f"❌ 트랜잭션 롤백: {e}")
                raise DBWriteError(f"트랜잭션 실패: {e}") from e
    
    def save_evolved_strategy(
        self,
        strategy: Dict[str, Any],
        segment: Dict[str, Any],
        lineage: Dict[str, Any]
    ) -> bool:
        """
        진화된 전략을 원자적으로 저장
        
        Args:
            strategy: 전략 정보 (strategies에 저장)
            segment: 세그먼트 결과 (segment_scores에 저장)
            lineage: 계보 정보 (strategy_lineage에 저장)
        
        Returns:
            저장 성공 여부
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                
                # 1. strategies 업데이트/삽입
                self._update_coin_strategy(cursor, strategy)
                
                # 2. segment_scores 삽입
                self._insert_segment_score(cursor, segment)
                
                # 3. strategy_lineage 삽입
                self._insert_lineage(cursor, lineage)
                
                logger.info(f"✅ 진화된 전략 저장 완료: {strategy.get('id', 'unknown')}")
                return True
                
        except Exception as e:
            logger.error(f"❌ 진화된 전략 저장 실패: {e}")
            return False
    
    def save_segment_batch(
        self,
        segments: List[Dict[str, Any]]
    ) -> int:
        """
        여러 세그먼트 결과를 원자적으로 저장
        
        Args:
            segments: 세그먼트 결과 리스트
        
        Returns:
            저장된 세그먼트 수
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                saved_count = 0
                
                for segment in segments:
                    try:
                        self._insert_segment_score(cursor, segment)
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ 세그먼트 저장 실패 (건너뜀): {e}")
                        continue
                
                logger.info(f"✅ {saved_count}개 세그먼트 저장 완료")
                return saved_count
                
        except Exception as e:
            logger.error(f"❌ 세그먼트 배치 저장 실패: {e}")
            return 0
    
    def _update_coin_strategy(self, cursor, strategy: Dict[str, Any]):
        """strategies 테이블 업데이트/삽입"""
        try:
            import json
            from datetime import datetime
            
            strategy_id = strategy.get('id')
            if not strategy_id:
                raise ValueError("전략 ID가 필요합니다")
            
            # JSON 파라미터 생성
            strategy_conditions = json.dumps({
                k: v for k, v in strategy.items()
                if k not in ['id', 'coin', 'interval', 'parent_id', 'version', 'created_at']
            })
            
            # INSERT OR REPLACE
            cursor.execute("""
                INSERT OR REPLACE INTO strategies (
                    id, coin, interval, parent_id, version,
                    strategy_type, strategy_conditions,
                    regime,
                    rsi_min, rsi_max, stop_loss_pct, take_profit_pct,
                    volume_ratio_min, volume_ratio_max,
                    macd_buy_threshold, macd_sell_threshold,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id,
                strategy.get('coin', 'BTC'),
                strategy.get('interval', '15m'),
                strategy.get('parent_id'),
                strategy.get('version', 1),
                'evolved',
                strategy_conditions,
                strategy.get('regime', 'ranging'),  # 🔥 레짐 필드 추가
                strategy.get('rsi_min', 30.0),
                strategy.get('rsi_max', 70.0),
                strategy.get('stop_loss_pct', 0.02),
                strategy.get('take_profit_pct', 0.04),
                strategy.get('volume_ratio_min', 1.0),
                strategy.get('volume_ratio_max', 2.0),
                strategy.get('macd_buy_threshold', 0.01),
                strategy.get('macd_sell_threshold', -0.01),
                datetime.now().isoformat()
            ))
            
        except Exception as e:
            logger.error(f"❌ strategies 업데이트 실패: {e}")
            raise
    
    def _insert_segment_score(self, cursor, segment: Dict[str, Any]):
        """segment_scores 테이블 삽입"""
        try:
            cursor.execute("""
                INSERT INTO segment_scores (
                    strategy_id, market, interval,
                    start_idx, end_idx, start_timestamp, end_timestamp,
                    profit, pf, sharpe, mdd, trades_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment.get('strategy_id'),
                segment.get('market'),
                segment.get('interval'),
                segment.get('start_idx'),
                segment.get('end_idx'),
                segment.get('start_timestamp'),
                segment.get('end_timestamp'),
                segment.get('profit', 0.0),
                segment.get('pf', 0.0),
                segment.get('sharpe', 0.0),
                segment.get('mdd', 0.0),
                segment.get('trades_count', 0)
            ))
            
        except Exception as e:
            logger.error(f"❌ segment_scores 삽입 실패: {e}")
            raise
    
    def _insert_lineage(self, cursor, lineage: Dict[str, Any]):
        """strategy_lineage 테이블 삽입"""
        try:
            import json
            
            segment_range_json = json.dumps(lineage.get('segment_range', {}))
            
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_lineage (
                    child_id, parent_id, mutation_desc,
                    segment_range, improvement_flag
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                lineage.get('child_id'),
                lineage.get('parent_id'),
                lineage.get('mutation_desc'),
                segment_range_json,
                lineage.get('improvement_flag', 0)
            ))
            
        except Exception as e:
            logger.error(f"❌ strategy_lineage 삽입 실패: {e}")
            raise

