"""
Absolute Zero 시스템 - 글로벌 전략 모듈
글로벌 전략 생성 및 관리 기능
"""

import logging
from typing import Optional, List, Dict, Any

from .az_config import configure_logging

logger = logging.getLogger(__name__)

def generate_global_strategies_only(
    coin_filter: Optional[List[str]] = None,
    enable_training: bool = False
) -> Dict[str, Any]:
    """
    글로벌 전략만 독립적으로 생성

    Args:
        coin_filter: 특정 코인만 필터링 (None이면 모든 코인)
        enable_training: 글로벌 학습 실행 여부 (기본값: False)

    Returns:
        실행 결과 딕셔너리
    """
    try:
        configure_logging()
        logger.info("🌍 글로벌 전략 생성 실행 시작 (Synthesizer 방식)")

        # 검증 함수 import
        from rl_pipeline.pipelines.orchestrator import (
            validate_global_strategy_pool,
            validate_global_strategy_patterns,
            validate_global_strategy_quality
        )

        # GlobalStrategySynthesizer 사용
        from rl_pipeline.strategy.global_synthesizer import create_global_synthesizer
        from rl_pipeline.core.env import config
        from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
        from rl_pipeline.monitoring import SessionManager

        # 디버그 세션 생성
        session_manager = SessionManager()
        session_id = session_manager.create_session(
            coins=coin_filter or ["ALL"],
            intervals=["global"],
            config={"enable_training": enable_training}
        )

        # Synthesizer 초기화
        db_path = config.STRATEGIES_DB

        # DB에서 실제 사용 가능한 인터벌 조회 (하드코딩 제거)
        available_combinations = get_available_coins_and_intervals()
        intervals = sorted(list({itv for _, itv in available_combinations}))

        # 사용 가능한 인터벌이 없으면 기본값 사용
        if not intervals:
            intervals = config.UNIFIED_INTERVALS

        seed = 123  # 재현성을 위한 seed

        synthesizer = create_global_synthesizer(db_path, intervals, seed)

        # 코인 필터링
        if coin_filter:
            logger.info(f"📋 코인 필터: {coin_filter}")
            coins = coin_filter
        else:
            # 사용 가능한 모든 코인 가져오기
            coin_interval_combinations = get_available_coins_and_intervals()
            coins = list(set([c for c, _ in coin_interval_combinations]))
            logger.info(f"📊 발견된 코인: {len(coins)}개")

        # 7단계 Synthesizer 파이프라인 실행
        logger.info("📊 1단계: 개별 전략 수집...")
        # 필터 조건 완화: min_trades=1 (최소 1개 거래), max_dd=1.0 (100% 허용)
        pool = synthesizer.load_pool(coins=coins, min_trades=1, max_dd=1.0)

        # 1단계 검증: 전략 풀 검증
        pool_validation = validate_global_strategy_pool(
            pool=pool,
            coins=coins,
            intervals=intervals,
            min_strategies_per_interval=10
        )

        _log_validation_result("전략 풀", pool_validation)

        # 디버그 로그 저장
        _save_debug_log(session_id, 'global_strategy_pool_validation', pool_validation)

        if not pool:
            logger.warning("⚠️ 수집된 개별 전략 없음, 폴백만 생성")
            final = synthesizer.apply_fallbacks({})
            synthesizer.save(final)

            # 세션 종료
            session_manager.end_session(session_id, summary={
                'status': 'fallback_only',
                'strategies_generated': sum(len(s) for s in final.values())
            })

            return {"success": True, "count": sum(len(s) for s in final.values())}

        logger.info("📊 2단계: 전략 표준화...")
        std_pool = synthesizer.standardize(pool)

        logger.info("📊 3단계: 공통 패턴 추출...")
        patterns = synthesizer.extract_common_patterns(std_pool)

        # 3단계 검증: 패턴 검증
        pattern_validation = validate_global_strategy_patterns(
            patterns=patterns,
            min_patterns_per_interval=3
        )

        _log_validation_result("패턴 추출", pattern_validation)

        # 디버그 로그 저장
        _save_debug_log(session_id, 'global_strategy_pattern_validation', pattern_validation)

        logger.info("📊 4단계: 글로벌 전략 조립...")
        assembled = synthesizer.assemble_global_strategies(patterns)

        logger.info("📊 5단계: 빠른 샌니티백테스트...")
        tested = synthesizer.quick_sanity_backtest(assembled)

        logger.info("📊 6단계: 폴백 적용...")
        final = synthesizer.apply_fallbacks(tested)

        # 7단계 전: 최종 품질 검증
        final_validation = validate_global_strategy_quality(
            final_strategies=final,
            intervals=intervals,
            min_strategies_per_interval=5
        )

        _log_validation_result("최종 글로벌 전략 품질", final_validation)

        # 디버그 로그 저장
        _save_debug_log(session_id, 'global_strategy_quality_validation', final_validation)

        logger.info("📊 7단계: DB 저장...")
        synthesizer.save(final)

        # 최종 통계
        total_strategies = sum(len(strategies) for strategies in final.values())
        logger.info(f"✅ 글로벌 전략 생성 완료: 총 {total_strategies}개")

        for interval, strategies in final.items():
            logger.info(f"  - {interval}: {len(strategies)}개")

        # 글로벌 학습 실행 (옵션)
        if enable_training:
            logger.info("🎓 글로벌 학습 시작...")
            _run_global_training(final, session_id)

        # 세션 종료
        session_manager.end_session(session_id, summary={
            'status': 'completed',
            'strategies_generated': total_strategies,
            'intervals': list(final.keys()),
            'training_enabled': enable_training
        })

        return {
            "success": True,
            "count": total_strategies,
            "by_interval": {k: len(v) for k, v in final.items()}
        }

    except Exception as e:
        logger.error(f"❌ 글로벌 전략 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def _log_validation_result(name: str, validation: Dict[str, Any]):
    """검증 결과 로깅"""
    logger.info(f"📊 {name} 검증 완료")
    logger.info(f"   └─ 검증 통과: {validation['valid']}")
    logger.info(f"   └─ 품질 점수: {validation.get('quality_score', 0)}/100")

    if 'stats' in validation:
        stats = validation['stats']
        if 'total_strategies' in stats:
            logger.info(f"   └─ 총 전략 수: {stats['total_strategies']}개")
        if 'total_patterns' in stats:
            logger.info(f"   └─ 총 패턴 수: {stats['total_patterns']}개")
        if 'intervals_covered' in stats and 'intervals_expected' in stats:
            logger.info(f"   └─ 인터벌 커버리지: {stats['intervals_covered']}/{stats['intervals_expected']}")
        if 'avg_strategies_per_interval' in stats:
            logger.info(f"   └─ 인터벌당 평균: {stats['avg_strategies_per_interval']}개")

    if validation.get('issues'):
        logger.error(f"❌ {name} 검증 실패:")
        for issue in validation['issues']:
            logger.error(f"   └─ {issue}")

    if validation.get('warnings'):
        logger.warning(f"⚠️ {name} 경고:")
        for warning in validation['warnings']:
            logger.warning(f"   └─ {warning}")

def _save_debug_log(session_id: str, event_name: str, validation_result: Dict[str, Any]):
    """디버그 로그 저장"""
    try:
        from rl_pipeline.monitoring.simulation_debugger import SimulationDebugger
        debugger = SimulationDebugger(session_id=session_id)

        log_data = {
            'event': event_name,
            'validation_result': {
                'valid': validation_result['valid'],
                'quality_score': validation_result.get('quality_score', 0),
                'num_issues': len(validation_result.get('issues', [])),
                'num_warnings': len(validation_result.get('warnings', []))
            },
            'issues': validation_result.get('issues', []),
            'warnings': validation_result.get('warnings', [])
        }

        # stats가 있으면 포함
        if 'stats' in validation_result:
            log_data['validation_result'].update(validation_result['stats'])

        debugger.log(log_data)
    except Exception as debug_error:
        logger.debug(f"⚠️ 디버그 로그 저장 실패: {debug_error}")

def _run_global_training(final_strategies: Dict[str, List], session_id: str):
    """
    글로벌 전략에 대한 학습 실행

    Args:
        final_strategies: 최종 글로벌 전략들
        session_id: 세션 ID
    """
    try:
        from rl_pipeline.training.global_trainer import GlobalStrategyTrainer
        from rl_pipeline.core.env import config

        trainer = GlobalStrategyTrainer(
            strategies=final_strategies,
            session_id=session_id,
            config=config
        )

        # 학습 실행
        training_results = trainer.train()

        # 결과 로깅
        if training_results.get('success'):
            logger.info(f"✅ 글로벌 학습 완료")
            logger.info(f"   └─ 학습된 모델 수: {training_results.get('models_trained', 0)}")
            logger.info(f"   └─ 평균 성능: {training_results.get('avg_performance', 0):.2f}")
        else:
            logger.warning(f"⚠️ 글로벌 학습 부분 실패: {training_results.get('error', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ 글로벌 학습 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())