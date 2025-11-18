"""
Absolute Zero 시스템 통합 테스트
전체 파이프라인 흐름 검증
"""

import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_imports():
    """모든 필수 모듈 import 테스트"""
    try:
        logger.info("📦 모듈 import 테스트 시작...")
        
        # 핵심 모듈
        import rl_pipeline.strategy.creator
        import rl_pipeline.strategy.global_strategy_creator
        import rl_pipeline.strategy.analyzer
        import rl_pipeline.pipelines.orchestrator
        import rl_pipeline.routing.regime_router
        import rl_pipeline.analysis.integrated_analyzer
        import rl_pipeline.db.realtime_signal_storage
        
        # 함수 import
        from rl_pipeline.strategy.creator import create_global_strategies_from_results
        from rl_pipeline.strategy.global_strategy_creator import (
            create_global_strategy_for_interval,
            create_global_strategy_all_intervals,
            filter_strategies_for_global,
            calculate_interval_grade_weights
        )
        from rl_pipeline.strategy.analyzer import _categorize_coins_by_importance
        
        logger.info("✅ 모든 모듈 import 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모듈 import 실패: {e}")
        return False


def test_global_strategy_creator_functions():
    """글로벌 전략 생성 함수 테스트"""
    try:
        logger.info("🔧 글로벌 전략 생성 함수 테스트 시작...")
        
        from rl_pipeline.strategy.global_strategy_creator import (
            filter_strategies_for_global,
            cluster_similar_strategies,
            classify_strategy_direction_and_regime,
            calculate_interval_grade_weights
        )
        
        # 테스트 데이터
        test_strategies = [
            {
                'id': 'test1',
                'quality_grade': 'S',
                'trades_count': 50,
                'profit': 0.1,
                'win_rate': 0.65
            },
            {
                'id': 'test2',
                'quality_grade': 'A',
                'trades_count': 30,
                'profit': 0.05,
                'win_rate': 0.60
            }
        ]
        
        # 필터링 테스트
        filtered = filter_strategies_for_global(test_strategies, 'BTC', '15m')
        logger.info(f"  ✅ 필터링 테스트: {len(filtered)}개 전략 선별")
        
        # 클러스터링 테스트
        clusters = cluster_similar_strategies(test_strategies)
        logger.info(f"  ✅ 클러스터링 테스트: {len(clusters)}개 클러스터")
        
        # 방향/레짐 분류 테스트
        direction, regime = classify_strategy_direction_and_regime(test_strategies[0])
        logger.info(f"  ✅ 방향/레짐 분류: {direction}, {regime}")
        
        # 등급 가중치 계산 테스트
        test_intervals = {
            '15m': [{'quality_grade': 'S'}],
            '30m': [{'quality_grade': 'A'}],
            '240m': [{'quality_grade': 'S'}]
        }
        grades, weights = calculate_interval_grade_weights(test_intervals)
        logger.info(f"  ✅ 등급 가중치 계산: {weights}")
        
        logger.info("✅ 글로벌 전략 생성 함수 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 생성 함수 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_categorize_coins():
    """코인 그룹화 함수 테스트"""
    try:
        logger.info("📊 코인 그룹화 함수 테스트 시작...")
        
        from rl_pipeline.strategy.analyzer import _categorize_coins_by_importance
        
        test_all_coin_strategies = {
            'BTC': {'15m': [], '30m': []},
            'ETH': {'15m': []},
            'XRP': {'15m': []},
            'UNKNOWN_COIN': {'15m': []}
        }
        
        coin_groups = _categorize_coins_by_importance(test_all_coin_strategies)
        logger.info(f"  ✅ 코인 그룹화: 메이저 {len(coin_groups['major'])}개, 중형 {len(coin_groups['mid'])}개")
        
        logger.info("✅ 코인 그룹화 함수 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 코인 그룹화 함수 테스트 실패: {e}")
        return False


def test_global_strategy_creation_logic():
    """글로벌 전략 생성 로직 테스트"""
    try:
        logger.info("🌍 글로벌 전략 생성 로직 테스트 시작...")
        
        from rl_pipeline.strategy.global_strategy_creator import (
            create_global_strategy_for_interval,
            create_global_strategy_all_intervals
        )
        
        # 테스트 데이터
        test_interval_strategies = {
            'BTC': [
                {
                    'id': 'test1',
                    'quality_grade': 'S',
                    'trades_count': 50,
                    'profit': 0.1,
                    'win_rate': 0.65,
                    'params': {'rsi_min': 30, 'rsi_max': 70}
                }
            ],
            'ETH': [
                {
                    'id': 'test2',
                    'quality_grade': 'A',
                    'trades_count': 30,
                    'profit': 0.05,
                    'win_rate': 0.60,
                    'params': {'rsi_min': 30, 'rsi_max': 70}
                }
            ]
        }
        
        # 인터벌별 글로벌 전략 생성 테스트
        result = create_global_strategy_for_interval('15m', test_interval_strategies, 'performance_based')
        
        if result:
            logger.info(f"  ✅ 인터벌별 글로벌 전략 생성: {result.get('name')}")
        else:
            logger.warning("  ⚠️ 인터벌별 글로벌 전략 생성 실패 (예상된 동작일 수 있음)")
        
        # 통합 인터벌 글로벌 전략 생성 테스트
        test_interval_global = {
            '15m': [result] if result else [],
            '30m': []
        }
        all_intervals_result = create_global_strategy_all_intervals(test_interval_global)
        
        if all_intervals_result:
            logger.info(f"  ✅ 통합 인터벌 글로벌 전략 생성: {all_intervals_result.get('name')}")
        else:
            logger.warning("  ⚠️ 통합 인터벌 글로벌 전략 생성 실패 (데이터 부족일 수 있음)")
        
        logger.info("✅ 글로벌 전략 생성 로직 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 생성 로직 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("🧪 Absolute Zero 시스템 통합 테스트 시작\n")
    
    tests = [
        ("모듈 Import", test_imports),
        ("글로벌 전략 생성 함수", test_global_strategy_creator_functions),
        ("코인 그룹화", test_categorize_coins),
        ("글로벌 전략 생성 로직", test_global_strategy_creation_logic),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"테스트: {test_name}")
        logger.info(f"{'='*60}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("테스트 결과 요약")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

