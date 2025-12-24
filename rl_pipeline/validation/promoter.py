"""
전략 승격/강등 관리자 (Strategy Promoter)
- 생애주기(Lifecycle) 상태 전이 관리
- QUARANTINE -> CANDIDATE -> ACTIVE -> RETIRED -> DEAD
- GPT.md의 '인사과' 역할 수행
"""

import logging
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from rl_pipeline.core.env import config
from rl_pipeline.db.connection_pool import get_optimized_db_connection

logger = logging.getLogger(__name__)

class StrategyPromoter:
    """전략 인사 관리 시스템"""
    
    def __init__(self, db_path: str = None):
        # 기본적으로 설정된 전략 DB 경로 사용
        self.db_path = db_path or config.STRATEGIES_DB
        
    def process_promotions(self, coin: str = None) -> Dict[str, int]:
        """승격/강등 심사 실행"""
        stats = {
            "promoted_to_candidate": 0,
            "promoted_to_active": 0,
            "demoted_to_retired": 0,
            "killed_to_dead": 0
        }
        
        try:
            # 코인별 DB 경로 확인 (엔진화 대응)
            target_db_path = config.get_strategy_db_path(coin) if coin else self.db_path
            
            with get_optimized_db_connection(target_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 1. QUARANTINE -> CANDIDATE 심사 (격리 해제)
                # 조건: 거래 30회 이상, 승률 50% 이상, MDD < 15%
                cursor.execute("""
                    SELECT id, symbol, interval, win_rate, trades_count, max_drawdown 
                    FROM strategies 
                    WHERE lifecycle_status = 'QUARANTINE'
                      AND trades_count >= 30
                      AND win_rate >= 0.5
                      AND max_drawdown > -0.15
                """)
                candidates = cursor.fetchall()
                
                for cand in candidates:
                    self._update_status(cursor, cand['id'], 'CANDIDATE', "격리 해제: 기본 요건 충족")
                    stats["promoted_to_candidate"] += 1
                    
                # 2. CANDIDATE -> ACTIVE 심사 (실전 투입)
                # 조건: 후보 중 상위 20% (수익 팩터, 샤프 지수 기준), 최근 승률 안정적
                # 여기서는 단순화하여 'CANDIDATE' 중 성과가 검증된 상위 그룹 승격
                cursor.execute("""
                    SELECT id, profit_factor, sharpe_ratio 
                    FROM strategies 
                    WHERE lifecycle_status = 'CANDIDATE'
                      AND profit_factor >= 1.2
                      AND sharpe_ratio >= 0.5
                    ORDER BY profit_factor DESC
                    LIMIT 10
                """)
                actives = cursor.fetchall()
                
                for active in actives:
                    self._update_status(cursor, active['id'], 'ACTIVE', "실전 승격: 우수 성과 달성")
                    stats["promoted_to_active"] += 1
                
                # 3. ACTIVE -> RETIRED 심사 (성능 저하로 은퇴)
                # 조건: 거래 50회 이상인데 수익 마이너스, 혹은 MDD > 20%
                cursor.execute("""
                    SELECT id, profit, max_drawdown 
                    FROM strategies 
                    WHERE lifecycle_status = 'ACTIVE'
                      AND (
                          (trades_count >= 50 AND profit < 0)
                          OR max_drawdown <= -0.20
                      )
                """)
                retirees = cursor.fetchall()
                
                for retiree in retirees:
                    self._update_status(cursor, retiree['id'], 'RETIRED', "성능 저하로 은퇴")
                    stats["demoted_to_retired"] += 1
                    
                # 4. ANY -> DEAD 심사 (즉결 처형: 물리 법칙 위반)
                # 조건: 파산 위험, MDD > 30% 등 심각한 손상
                cursor.execute("""
                    SELECT id 
                    FROM strategies 
                    WHERE lifecycle_status != 'DEAD'
                      AND max_drawdown <= -0.30
                """)
                dead_ones = cursor.fetchall()
                
                for dead in dead_ones:
                    self._update_status(cursor, dead['id'], 'DEAD', "물리 법칙 위반 (MDD 초과)")
                    stats["killed_to_dead"] += 1
                
                conn.commit()
                
            if sum(stats.values()) > 0:
                logger.info(f"⚖️ 인사 이동 완료 ({coin or 'ALL'}): {json.dumps(stats, ensure_ascii=False)}")
                
            return stats
            
        except Exception as e:
            logger.error(f"❌ 승격 심사 중 오류 발생: {e}")
            return stats

    def _update_status(self, cursor, strategy_id: str, new_status: str, reason: str):
        """상태 업데이트 헬퍼"""
        cursor.execute("""
            UPDATE strategies 
            SET lifecycle_status = ?, 
                description = description || ' | [' || ? || '] ' || ?
            WHERE id = ?
        """, (new_status, datetime.now().strftime('%Y-%m-%d'), reason, strategy_id))
        
        # 로그는 너무 많을 수 있으니 디버그 레벨로
        logger.debug(f"  └─ 전략 {strategy_id}: {new_status} ({reason})")

def run_promoter(coin: str = None):
    """프로모터 실행 헬퍼"""
    promoter = StrategyPromoter()
    promoter.process_promotions(coin)

