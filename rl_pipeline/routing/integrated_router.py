"""
Integrated Router - Phase 4 통합
Regime Router + Signal Runtime Adapter 통합 라우팅 시스템
"""
import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.routing.regime_router import RegimeRouter, RegimeRoutingResult
from rl_pipeline.runtime import SignalRuntimeAdapter, SignalParameters, AdapterConfig

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class IntegratedRoutingResult(RegimeRoutingResult):
    """통합 라우팅 결과 (Phase 4 파라미터 포함)"""
    # Phase 4 파라미터
    signal_params: Optional[SignalParameters] = None

    # 추가 플래그
    has_valid_params: bool = False
    rejection_reason: Optional[str] = None

class IntegratedRouter:
    """통합 라우터 - Regime + Phase 4 Adapter"""

    def __init__(self,
                 adapter_config: Optional[AdapterConfig] = None,
                 use_signal_adapter: bool = True):
        """
        Args:
            adapter_config: Signal Adapter 설정
            use_signal_adapter: Signal Adapter 사용 여부
        """
        self.regime_router = RegimeRouter()
        self.signal_adapter = SignalRuntimeAdapter(adapter_config) if use_signal_adapter else None
        self.use_signal_adapter = use_signal_adapter

        logger.info(f"🚀 통합 라우터 초기화 (Signal Adapter: {'ON' if use_signal_adapter else 'OFF'})")

    def route_strategies_with_params(self,
                                    coin: str,
                                    interval: str,
                                    regime_tag: str,
                                    strategies: List[Dict[str, Any]],
                                    candle_data: pd.DataFrame,
                                    use_accumulated_data: bool = True
                                   ) -> List[IntegratedRoutingResult]:
        """
        전략 라우팅 + Phase 4 파라미터 생성

        Args:
            coin: 코인명
            interval: 인터벌
            regime_tag: 레짐 태그 (ranging/trending/volatile)
            strategies: 전략 리스트
            candle_data: 캔들 데이터
            use_accumulated_data: 누적 데이터 활용 여부

        Returns:
            통합 라우팅 결과 리스트
        """
        logger.info(f"\n🎯 통합 라우팅 시작: {coin} {interval} {regime_tag}")
        logger.info(f"   전략 수: {len(strategies)}개")

        # 1. Regime Router로 기본 라우팅
        routing_results = self.regime_router.route_strategies(
            coin=coin,
            interval=interval,
            strategies=strategies,
            candle_data=candle_data,
            use_accumulated_data=use_accumulated_data
        )

        logger.info(f"   ✅ Regime 라우팅 완료: {len(routing_results)}개 결과")

        # 2. Phase 4 Adapter로 파라미터 생성
        integrated_results = []

        for routing_result in routing_results:
            strategy_id = routing_result.routed_strategy.get('id') or \
                         routing_result.routed_strategy.get('strategy_id')

            if not strategy_id:
                logger.warning(f"   ⚠️ 전략 ID 없음, Phase 4 파라미터 생성 불가")
                integrated_result = IntegratedRoutingResult(
                    **routing_result.__dict__,
                    signal_params=None,
                    has_valid_params=False,
                    rejection_reason="No strategy ID"
                )
                integrated_results.append(integrated_result)
                continue

            # Phase 4 파라미터 생성
            signal_params = None
            rejection_reason = None

            if self.use_signal_adapter and self.signal_adapter:
                try:
                    signal_params = self.signal_adapter.get_signal_parameters(
                        coin=coin,
                        interval=interval,
                        regime_tag=regime_tag,
                        strategy_id=strategy_id
                    )

                    if not signal_params:
                        rejection_reason = "Filtered by Phase 4 (grade/sample/PF)"
                        logger.debug(f"   🚫 {strategy_id[:40]}... 필터링됨 (Phase 4)")
                    else:
                        logger.debug(f"   ✅ {strategy_id[:40]}... Phase 4 파라미터 생성")

                except Exception as e:
                    rejection_reason = f"Phase 4 error: {str(e)}"
                    logger.warning(f"   ⚠️ Phase 4 파라미터 생성 실패: {e}")

            # 통합 결과 생성
            integrated_result = IntegratedRoutingResult(
                **routing_result.__dict__,
                signal_params=signal_params,
                has_valid_params=(signal_params is not None),
                rejection_reason=rejection_reason
            )

            integrated_results.append(integrated_result)

        # 3. 결과 필터링 (Phase 4 통과한 것만)
        if self.use_signal_adapter:
            valid_results = [r for r in integrated_results if r.has_valid_params]
            logger.info(f"   ✅ Phase 4 필터링: {len(valid_results)}/{len(integrated_results)}개 통과")
            return valid_results
        else:
            return integrated_results

    def get_top_strategies(self,
                          coin: str,
                          interval: str,
                          regime_tag: str,
                          strategies: List[Dict[str, Any]],
                          candle_data: pd.DataFrame,
                          top_n: int = 3
                         ) -> List[IntegratedRoutingResult]:
        """
        상위 N개 전략 선택

        Args:
            coin: 코인명
            interval: 인터벌
            regime_tag: 레짐 태그
            strategies: 전략 리스트
            candle_data: 캔들 데이터
            top_n: 상위 N개

        Returns:
            상위 전략 리스트
        """
        results = self.route_strategies_with_params(
            coin=coin,
            interval=interval,
            regime_tag=regime_tag,
            strategies=strategies,
            candle_data=candle_data
        )

        if not results:
            logger.warning(f"   ⚠️ 유효한 전략 없음")
            return []

        # routing_score 기준으로 정렬
        sorted_results = sorted(results, key=lambda r: r.routing_score, reverse=True)
        top_strategies = sorted_results[:top_n]

        logger.info(f"\n🏆 상위 {len(top_strategies)}개 전략 선택:")
        for i, result in enumerate(top_strategies, 1):
            params = result.signal_params
            logger.info(f"   {i}. Score: {result.routing_score:.3f}")
            if params:
                logger.info(f"      TP: {params.tp*100:.2f}% | SL: {params.sl*100:.2f}% | "
                          f"Size: {params.size:.2f}x | Grade: {params.grade}")

        return top_strategies

def main():
    """테스트 함수"""
    logger.info("🚀 Integrated Router 테스트\n")

    # 테스트용 더미 데이터
    # 실제로는 DB에서 로드
    pass

if __name__ == "__main__":
    main()
