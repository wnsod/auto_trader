"""
리그 시스템 관리자 (LeagueManager)
전략의 승격(Promotion) 및 강등(Relegation) 로직 처리
Major(1군) vs Minor(2군) 시스템
"""

import logging
import sqlite3
from typing import List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class LeagueManager:
    """
    리그 시스템 관리 클래스
    
    구조:
    - Major League (1군): 실전 투입 가능한 검증된 정예 전략 (정원 제한 있음, 예: 50개)
    - Minor League (2군): 신규 생성 전략 및 1군에서 밀려난 전략 (육성군)
    
    동작:
    - 승격(Promotion): Minor 리그 1위 ~ N위 -> Major 리그로 승격
    - 강등(Relegation): Major 리그 최하위 N명 -> Minor 리그로 강등
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.major_capacity = 50  # 1군 정원
        self.promotion_count = 5  # 한 번에 승강되는 수
        
    def process_league_updates(self, coin: str, interval: str) -> Dict[str, int]:
        """리그 승강제 실행"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 1. 리그 데이터 초기화 확인 (league가 NULL인 경우 minor로 설정)
                cursor.execute("""
                    UPDATE strategies 
                    SET league = 'minor' 
                    WHERE league IS NULL AND symbol = ? AND interval = ?
                """, (coin, interval))
                
                # 2. Major 리그 현황 파악
                cursor.execute("""
                    SELECT COUNT(*) FROM strategies 
                    WHERE symbol = ? AND interval = ? AND league = 'major'
                """, (coin, interval))
                major_count = cursor.fetchone()[0]
                
                stats = {'promoted': 0, 'relegated': 0, 'major_count': major_count}
                
                # 3. Major 리그 정원 미달 시: Minor 최상위 전략 즉시 승격
                if major_count < self.major_capacity:
                    deficit = self.major_capacity - major_count
                    promoted = self._promote_strategies(cursor, coin, interval, limit=deficit)
                    stats['promoted'] += promoted
                    major_count += promoted
                
                # 4. 정기 승강제 (Major 꼴등 vs Minor 1등 교체)
                # 데이터가 충분할 때만 실행
                cursor.execute("""
                    SELECT COUNT(*) FROM strategies 
                    WHERE symbol = ? AND interval = ? AND league = 'minor'
                """, (coin, interval))
                minor_count = cursor.fetchone()[0]
                
                if major_count >= self.major_capacity and minor_count > 0:
                    # 강등 (Major 하위 N명)
                    relegated = self._relegate_strategies(cursor, coin, interval, limit=self.promotion_count)
                    stats['relegated'] += relegated
                    
                    # 승격 (Minor 상위 N명)
                    promoted = self._promote_strategies(cursor, coin, interval, limit=self.promotion_count)
                    stats['promoted'] += promoted
                
                conn.commit()
                
                if stats['promoted'] > 0 or stats['relegated'] > 0:
                    logger.info(f"🏆 {coin}-{interval} 리그 변동: 승격 {stats['promoted']}명, 강등 {stats['relegated']}명 (Major: {major_count}명)")
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ 리그 업데이트 실패 ({coin}-{interval}): {e}")
            return {'promoted': 0, 'relegated': 0}

    def _promote_strategies(self, cursor, coin: str, interval: str, limit: int) -> int:
        """Minor -> Major 승격 (성능 우수자)"""
        # 승격 기준: 종합 점수(score) 높은 순, 또는 승률/수익금 높은 순
        # score 컬럼이 없거나 0일 수 있으므로 복합 정렬 사용
        query = """
            SELECT id FROM strategies
            WHERE symbol = ? AND interval = ? AND league = 'minor'
              AND (lifecycle_status IS NULL OR lifecycle_status NOT IN ('DEAD', 'RETIRED'))
            ORDER BY 
                CASE quality_grade 
                    WHEN 'S' THEN 5 WHEN 'A' THEN 4 WHEN 'B' THEN 3 
                    WHEN 'C' THEN 2 WHEN 'D' THEN 1 ELSE 0 
                END DESC,
                win_rate DESC, 
                profit DESC
            LIMIT ?
        """
        cursor.execute(query, (coin, interval, limit))
        candidates = [row[0] for row in cursor.fetchall()]
        
        if not candidates:
            return 0
            
        placeholders = ','.join(['?'] * len(candidates))
        cursor.execute(f"""
            UPDATE strategies 
            SET league = 'major', updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
        """, candidates)
        
        return len(candidates)

    def _relegate_strategies(self, cursor, coin: str, interval: str, limit: int) -> int:
        """Major -> Minor 강등 (성능 저조자)"""
        # 강등 기준: 종합 점수 낮은 순
        query = """
            SELECT id FROM strategies
            WHERE symbol = ? AND interval = ? AND league = 'major'
            ORDER BY 
                win_rate ASC, 
                profit ASC,
                CASE quality_grade 
                    WHEN 'S' THEN 5 WHEN 'A' THEN 4 WHEN 'B' THEN 3 
                    WHEN 'C' THEN 2 WHEN 'D' THEN 1 ELSE 0 
                END ASC
            LIMIT ?
        """
        cursor.execute(query, (coin, interval, limit))
        candidates = [row[0] for row in cursor.fetchall()]
        
        if not candidates:
            return 0
            
        placeholders = ','.join(['?'] * len(candidates))
        cursor.execute(f"""
            UPDATE strategies 
            SET league = 'minor', updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
        """, candidates)
        
        return len(candidates)
