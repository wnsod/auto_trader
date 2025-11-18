"""
리스크 제어 모듈 (Phase 5)
Drawdown 제어 및 포지션 크기 관리

기능:
1. Drawdown 감지
2. 포지션 크기 자동 축소
3. 일관성 점수 기반 필터링
"""

import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# 환경변수
DRAWDOWN_REDUCE_THRESHOLD = float(os.getenv('DRAWDOWN_REDUCE_THRESHOLD', '0.2'))  # 20%
POSITION_REDUCTION_FACTOR = float(os.getenv('POSITION_REDUCTION_FACTOR', '0.5'))  # 50% 축소
MIN_CONSISTENCY_SCORE = float(os.getenv('MIN_CONSISTENCY_SCORE', '0.3'))  # 최소 일관성 점수


class RiskController:
    """리스크 제어 엔진"""
    
    def __init__(self):
        """초기화"""
        logger.debug("✅ Risk Controller 초기화 완료")  # DEBUG 레벨로 변경 (과도한 로그 방지)
        self._last_logged_drawdown_range = None  # 마지막으로 로그 출력한 drawdown 범위
        self._log_interval = 0.05  # 5% 단위로만 로그 출력 (과도한 로그 방지)
    
    def calculate_drawdown(self, equity_curve: List[float]) -> float:
        """
        최대 낙폭 계산
        
        Args:
            equity_curve: 자산 곡선 리스트
        
        Returns:
            최대 낙폭 (0.0 ~ 1.0)
        """
        try:
            if not equity_curve or len(equity_curve) < 2:
                return 0.0
            
            max_equity = equity_curve[0]
            max_drawdown = 0.0
            
            for equity in equity_curve:
                if equity > max_equity:
                    max_equity = equity
                
                drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"❌ Drawdown 계산 실패: {e}")
            return 0.0
    
    def should_reduce_position(self, max_drawdown: float) -> bool:
        """
        포지션 축소 여부 판단
        
        Args:
            max_drawdown: 최대 낙폭
        
        Returns:
            축소 필요 여부
        """
        return max_drawdown >= DRAWDOWN_REDUCE_THRESHOLD
    
    def get_adjusted_position_size(
        self,
        base_size: float,
        max_drawdown: float
    ) -> float:
        """
        Drawdown 기반 조정된 포지션 크기
        
        Args:
            base_size: 기본 포지션 크기
            max_drawdown: 최대 낙폭
        
        Returns:
            조정된 포지션 크기
        """
        try:
            if max_drawdown < DRAWDOWN_REDUCE_THRESHOLD:
                return base_size
            
            # Drawdown이 임계값을 넘으면 축소
            reduction = POSITION_REDUCTION_FACTOR
            
            # Drawdown이 크면 더 많이 축소
            if max_drawdown > DRAWDOWN_REDUCE_THRESHOLD * 1.5:
                reduction = POSITION_REDUCTION_FACTOR * 0.5  # 더 많이 축소
            
            adjusted_size = base_size * (1 - reduction)
            
            # 로그 빈도 제한: drawdown 범위가 크게 변할 때만 로그 출력
            # 단, 개별 에이전트 로그는 DEBUG 레벨로만 출력 (과도한 로그 방지)
            current_range = int(max_drawdown / self._log_interval)
            if self._last_logged_drawdown_range is None or current_range != self._last_logged_drawdown_range:
                logger.debug(f"⚠️ 포지션 크기 축소: {base_size:.2f} → {adjusted_size:.2f} "
                             f"(drawdown={max_drawdown:.2%})")
                self._last_logged_drawdown_range = current_range
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"❌ 포지션 크기 조정 실패: {e}")
            return base_size
    
    def filter_by_consistency(
        self,
        strategies: List[Dict[str, Any]],
        segment_returns: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """
        일관성 점수 기반 전략 필터링
        
        Args:
            strategies: 전략 리스트
            segment_returns: 전략별 세그먼트 수익률 딕셔너리
        
        Returns:
            필터링된 전략 리스트
        """
        try:
            from rl_pipeline.strategy.strategy_evolver import StrategyEvolver
            
            evolver = StrategyEvolver()
            filtered = []
            
            for strategy in strategies:
                strategy_id = strategy.get('id')
                returns = segment_returns.get(strategy_id, [])
                
                if not returns:
                    # 세그먼트 데이터가 없으면 포함 (검증 안 됨)
                    filtered.append(strategy)
                    continue
                
                # Consistency Score 계산
                consistency = evolver.calculate_consistency_score(returns)
                
                if consistency >= MIN_CONSISTENCY_SCORE:
                    filtered.append(strategy)
                    logger.debug(f"✅ {strategy_id}: 일관성 통과 (score={consistency:.3f})")
                else:
                    logger.warning(f"⚠️ {strategy_id}: 일관성 부족 (score={consistency:.3f} < {MIN_CONSISTENCY_SCORE})")
            
            logger.info(f"✅ 일관성 필터링: {len(filtered)}/{len(strategies)} 전략 통과")
            return filtered
            
        except Exception as e:
            logger.error(f"❌ 일관성 필터링 실패: {e}")
            return strategies  # 실패 시 원본 반환

