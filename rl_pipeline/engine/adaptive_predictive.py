"""
적응형 예측 비율 조정 모듈
학습 성과에 따라 예측형 Self-play 비율을 동적으로 조절
"""

import os
import logging
import sqlite3
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_adaptive_predictive_ratio(coin: str, interval: str) -> float:
    """
    학습 성과 기반 적응형 예측 비율 조회
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        
    Returns:
        적용할 예측 비율 (0.0 ~ 1.0)
    """
    # 기본값 설정
    default_ratio = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
    
    try:
        from rl_pipeline.core.env import config
        db_path = config.get_strategy_db_path(coin)
        
        if not os.path.exists(db_path):
            return default_ratio
            
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 최근 에피소드의 정확도 조회
            cursor.execute("""
                SELECT AVG(accuracy) 
                FROM rl_episode_summary 
                WHERE symbol = ? AND interval = ? 
                ORDER BY ts_exit DESC 
                LIMIT 20
            """, (coin, interval))
            
            result = cursor.fetchone()
            avg_accuracy = result[0] if result and result[0] is not None else 0.0
            
            # 정확도가 높으면 예측 비율 상향 (더 공격적으로)
            # 정확도가 낮으면 예측 비율 하향 (더 보수적으로)
            if avg_accuracy > 0.6:
                return min(0.8, default_ratio * 1.5)
            elif avg_accuracy < 0.4:
                return max(0.1, default_ratio * 0.8)
            else:
                return default_ratio
                
    except Exception as e:
        logger.debug(f"⚠️ 적응형 비율 조회 실패: {e}")
        return default_ratio

