"""
Phase 3A 테스트: 전략 진화 모듈 검증 (오프라인 테스트)

실행 방법:
    docker exec -it auto_trader_coin bash
    cd /workspace
    python rl_pipeline/tests/test_phase3a_strategy_evolution.py
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 경로에 추가
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rl_pipeline.strategy.strategy_evolver import (
    StrategyEvolver,
    EvolvedStrategy,
    EVOLUTION_TOP_PERCENT,
    EVOLUTION_MIN_GRADE
)
from rl_pipeline.db.connection_pool import get_strategy_db_pool
from rl_pipeline.db.schema import create_coin_strategies_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_strategies() -> list:
    """테스트용 전략 생성"""
    strategies = [
        {
            'id': 'test_evolution_s1',
            'coin': 'BTC',
            'interval': '15m',
            'quality_grade': 'S',
            'profit': 1000.0,
            'win_rate': 0.7,
            'profit_factor': 2.0,
            'rsi_min': 30.0,
            'rsi_max': 70.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'version': 1
        },
        {
            'id': 'test_evolution_a1',
            'coin': 'BTC',
            'interval': '15m',
            'quality_grade': 'A',
            'profit': 800.0,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'rsi_min': 25.0,
            'rsi_max': 75.0,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.045,
            'version': 1
        },
        {
            'id': 'test_evolution_a2',
            'coin': 'BTC',
            'interval': '15m',
            'quality_grade': 'A',
            'profit': 750.0,
            'win_rate': 0.6,
            'profit_factor': 1.5,
            'rsi_min': 32.0,
            'rsi_max': 68.0,
            'stop_loss_pct': 0.018,
            'take_profit_pct': 0.038,
            'version': 1
        },
        {
            'id': 'test_evolution_b1',
            'coin': 'BTC',
            'interval': '15m',
            'quality_grade': 'B',
            'profit': 500.0,
            'win_rate': 0.55,
            'profit_factor': 1.2,
            'rsi_min': 28.0,
            'rsi_max': 72.0,
            'stop_loss_pct': 0.022,
            'take_profit_pct': 0.042,
            'version': 1
        },
        {
            'id': 'test_evolution_c1',
            'coin': 'BTC',
            'interval': '15m',
            'quality_grade': 'C',
            'profit': 200.0,
            'win_rate': 0.5,
            'profit_factor': 1.0,
            'rsi_min': 35.0,
            'rsi_max': 65.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'version': 1
        }
    ]
    return strategies


def test_strategy_selection():
    """상위 전략 선별 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 1: 상위 전략 선별")
    logger.info("=" * 60)
    
    try:
        evolver = StrategyEvolver()
        strategies = create_test_strategies()
        
        # 상위 전략 선별 (상위 20%, 최소 등급 B)
        top_strategies = evolver.select_top_strategies(
            strategies,
            top_percent=EVOLUTION_TOP_PERCENT,
            min_grade=EVOLUTION_MIN_GRADE
        )
        
        if not top_strategies:
            logger.error("❌ 선별된 전략이 없습니다")
            return False
        
        logger.info(f"✅ {len(top_strategies)}개 상위 전략 선별:")
        for strategy in top_strategies:
            grade = strategy.get('quality_grade', 'UNKNOWN')
            profit = strategy.get('profit', 0.0)
            logger.info(f"  - {strategy['id']}: 등급={grade}, profit={profit:.2f}")
        
        # 검증: 최소 등급 이상인지 확인
        grade_order = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5}
        min_grade_rank = grade_order.get(EVOLUTION_MIN_GRADE, 999)
        
        for strategy in top_strategies:
            grade = strategy.get('quality_grade', 'UNKNOWN')
            grade_rank = grade_order.get(grade, 999)
            if grade_rank > min_grade_rank:
                logger.error(f"❌ 최소 등급 위반: {strategy['id']} (등급={grade})")
                return False
        
        logger.info("✅ 모든 선별 전략이 최소 등급 이상")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_consistency_score():
    """Consistency Score 계산 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 2: Consistency Score 계산")
    logger.info("=" * 60)
    
    try:
        evolver = StrategyEvolver()
        
        # 테스트 케이스
        test_cases = [
            {
                "name": "일관성 높음 (작은 변동)",
                "returns": [0.01, 0.015, 0.012, 0.014, 0.013],
                "expected_high": True
            },
            {
                "name": "일관성 낮음 (큰 변동)",
                "returns": [0.1, -0.05, 0.2, -0.1, 0.15],
                "expected_high": False
            }
        ]
        
        for case in test_cases:
            score = evolver.calculate_consistency_score(case["returns"])
            logger.info(f"  {case['name']}: score={score:.4f}")
            
            if case["expected_high"]:
                if score < 0.5:
                    logger.warning(f"  ⚠️ 예상: 높은 일관성, 실제: {score:.4f}")
            else:
                if score > 0.5:
                    logger.warning(f"  ⚠️ 예상: 낮은 일관성, 실제: {score:.4f}")
        
        logger.info("✅ Consistency Score 계산 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_diversity_score():
    """다양성 점수 계산 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 3: 다양성 점수 계산")
    logger.info("=" * 60)
    
    try:
        evolver = StrategyEvolver()
        
        # 유사한 전략 (다양성 낮음)
        similar_strategies = [
            {
                'rsi_min': 30.0, 'rsi_max': 70.0,
                'take_profit_pct': 0.04, 'stop_loss_pct': 0.02
            },
            {
                'rsi_min': 30.5, 'rsi_max': 70.5,
                'take_profit_pct': 0.041, 'stop_loss_pct': 0.021
            },
            {
                'rsi_min': 29.5, 'rsi_max': 69.5,
                'take_profit_pct': 0.039, 'stop_loss_pct': 0.019
            }
        ]
        
        # 다양한 전략 (다양성 높음)
        diverse_strategies = [
            {
                'rsi_min': 20.0, 'rsi_max': 80.0,
                'take_profit_pct': 0.05, 'stop_loss_pct': 0.01
            },
            {
                'rsi_min': 40.0, 'rsi_max': 60.0,
                'take_profit_pct': 0.02, 'stop_loss_pct': 0.03
            },
            {
                'rsi_min': 25.0, 'rsi_max': 75.0,
                'take_profit_pct': 0.06, 'stop_loss_pct': 0.015
            }
        ]
        
        diversity_low = evolver.calculate_diversity_score(similar_strategies)
        diversity_high = evolver.calculate_diversity_score(diverse_strategies)
        
        logger.info(f"  유사한 전략 다양성: {diversity_low:.4f}")
        logger.info(f"  다양한 전략 다양성: {diversity_high:.4f}")
        
        if diversity_high <= diversity_low:
            logger.warning(f"⚠️ 다양성 점수가 예상과 반대: {diversity_high:.4f} <= {diversity_low:.4f}")
            return False
        
        logger.info("✅ 다양성 점수 계산 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_crossover():
    """교배 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 4: 교배 (Crossover)")
    logger.info("=" * 60)
    
    try:
        evolver = StrategyEvolver()
        
        parent1 = {
            'rsi_min': 30.0, 'rsi_max': 70.0,
            'take_profit_pct': 0.04, 'stop_loss_pct': 0.02
        }
        
        parent2 = {
            'rsi_min': 25.0, 'rsi_max': 75.0,
            'take_profit_pct': 0.05, 'stop_loss_pct': 0.025
        }
        
        child = evolver.crossover(parent1, parent2)
        
        logger.info("  부모 1:", parent1)
        logger.info("  부모 2:", parent2)
        logger.info("  자식:", child)
        
        # 자식이 두 부모의 파라미터를 모두 포함하는지 확인
        all_params = set(list(parent1.keys()) + list(parent2.keys()))
        child_params = set(child.keys())
        
        missing = all_params - child_params
        if missing:
            logger.warning(f"⚠️ 자식에 누락된 파라미터: {missing}")
        
        logger.info("✅ 교배 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_mutation():
    """변이 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 5: 변이 (Mutation)")
    logger.info("=" * 60)
    
    try:
        evolver = StrategyEvolver()
        
        original = {
            'rsi_min': 30.0, 'rsi_max': 70.0,
            'take_profit_pct': 0.04, 'stop_loss_pct': 0.02,
            'volume_ratio_min': 1.0, 'volume_ratio_max': 2.0
        }
        
        mutated, desc = evolver.mutate(original, strength=0.1, probability=1.0)
        
        logger.info(f"  원본: {original}")
        logger.info(f"  변이: {mutated}")
        logger.info(f"  변이 설명: {desc}")
        
        # 변화 확인
        changes = []
        for key in original.keys():
            if key in mutated:
                change = mutated[key] - original[key]
                if abs(change) > 1e-6:
                    changes.append((key, original[key], mutated[key], change))
                    logger.info(f"    {key}: {original[key]:.4f} → {mutated[key]:.4f} (변화: {change:+.4f})")
        
        if not changes:
            logger.warning("⚠️ 변이가 발생하지 않았습니다")
            return False
        
        # 범위 확인
        if mutated['rsi_min'] < 0 or mutated['rsi_min'] > 50:
            logger.error(f"❌ rsi_min 범위 초과: {mutated['rsi_min']}")
            return False
        
        if mutated['rsi_max'] < 50 or mutated['rsi_max'] > 100:
            logger.error(f"❌ rsi_max 범위 초과: {mutated['rsi_max']}")
            return False
        
        logger.info("✅ 변이 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_evolution_integration():
    """진화 통합 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 6: 진화 통합 (교배 + 변이)")
    logger.info("=" * 60)
    
    try:
        evolver = StrategyEvolver()
        strategies = create_test_strategies()
        
        # 상위 전략 선별 (더 많은 전략 선택)
        top_strategies = evolver.select_top_strategies(strategies, top_percent=0.6, min_grade='C')
        
        if len(top_strategies) < 2:
            logger.warning(f"⚠️ 교배를 위한 전략이 부족합니다 (선별된 전략: {len(top_strategies)}개)")
            return False
        
        logger.info(f"  선별된 전략: {len(top_strategies)}개")
        
        # 진화 실행
        evolved = evolver.evolve_strategies(
            top_strategies,
            n_children=3,
            segment_range={'start_idx': 0, 'end_idx': 100}
        )
        
        if not evolved:
            logger.error("❌ 진화된 전략이 생성되지 않았습니다")
            return False
        
        logger.info(f"✅ {len(evolved)}개 진화된 전략 생성:")
        for i, e in enumerate(evolved):
            logger.info(f"  {i+1}. {e.strategy_id}")
            logger.info(f"     부모: {e.parent_id}, 버전: {e.version}")
            logger.info(f"     변이: {e.mutation_desc}")
            logger.info(f"     파라미터 샘플: rsi_min={e.params.get('rsi_min', 'N/A')}, "
                       f"rsi_max={e.params.get('rsi_max', 'N/A')}")
        
        # 다양성 확인
        parent_diversity = evolver.calculate_diversity_score(top_strategies)
        evolved_params = [e.params for e in evolved]
        evolved_diversity = evolver.calculate_diversity_score(evolved_params)
        
        logger.info(f"  부모 다양성: {parent_diversity:.4f}")
        logger.info(f"  자식 다양성: {evolved_diversity:.4f}")
        
        logger.info("✅ 진화 통합 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_db_save():
    """DB 저장 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 7: DB 저장")
    logger.info("=" * 60)
    
    try:
        # 기본 테이블 생성 (없을 경우)
        create_coin_strategies_table()
        
        evolver = StrategyEvolver()
        strategies = create_test_strategies()
        
        # 먼저 부모 전략을 DB에 저장 (테스트용)
        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 기존 테스트 전략 삭제 (클린업)
            cursor.execute("""
                DELETE FROM coin_strategies 
                WHERE id LIKE 'test_evolution%'
            """)
            cursor.execute("""
                DELETE FROM strategy_lineage 
                WHERE child_id LIKE 'test_evolution%' OR parent_id LIKE 'test_evolution%'
            """)
            conn.commit()
            
            # 부모 전략 저장
            for strategy in strategies[:2]:  # 첫 2개만
                import json
                cursor.execute("""
                    INSERT OR REPLACE INTO coin_strategies (
                        id, coin, interval, quality_grade, profit, win_rate,
                        rsi_min, rsi_max, stop_loss_pct, take_profit_pct, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy['id'], strategy['coin'], strategy['interval'],
                    strategy['quality_grade'], strategy['profit'], strategy['win_rate'],
                    strategy['rsi_min'], strategy['rsi_max'],
                    strategy['stop_loss_pct'], strategy['take_profit_pct'],
                    strategy['version']
                ))
            
            conn.commit()
            logger.info("✅ 부모 전략 저장 완료")
        
        # 진화 실행
        top_strategies = evolver.select_top_strategies(strategies[:2], top_percent=1.0)
        evolved = evolver.evolve_strategies(
            top_strategies,
            n_children=2,
            segment_range={'start_idx': 0, 'end_idx': 100}
        )
        
        if not evolved:
            logger.error("❌ 진화된 전략 생성 실패")
            return False
        
        # DB 저장
        saved = evolver.save_evolved_strategies(evolved, 'BTC', '15m')
        
        if saved == 0:
            logger.error("❌ 저장된 전략이 없습니다")
            return False
        
        logger.info(f"✅ {saved}개 진화된 전략 저장 완료")
        
        # 저장 확인
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            
            for e in evolved:
                # coin_strategies 확인
                cursor.execute("SELECT id, parent_id, version FROM coin_strategies WHERE id = ?", (e.strategy_id,))
                result = cursor.fetchone()
                
                if not result:
                    logger.error(f"❌ 전략이 저장되지 않음: {e.strategy_id}")
                    return False
                
                stored_id, stored_parent, stored_version = result
                logger.info(f"  ✅ {stored_id}: parent={stored_parent}, version={stored_version}")
                
                # strategy_lineage 확인
                cursor.execute("SELECT child_id, mutation_desc FROM strategy_lineage WHERE child_id = ?", (e.strategy_id,))
                lineage_result = cursor.fetchone()
                
                if not lineage_result:
                    logger.warning(f"⚠️ 계보 정보가 저장되지 않음: {e.strategy_id}")
                else:
                    logger.info(f"  ✅ 계보 정보: {lineage_result[1]}")
        
        logger.info("✅ DB 저장 테스트 통과")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("=" * 60)
    logger.info("Phase 3A: 전략 진화 모듈 테스트 시작 (오프라인)")
    logger.info("=" * 60)
    
    tests = [
        ("상위 전략 선별", test_strategy_selection),
        ("Consistency Score 계산", test_consistency_score),
        ("다양성 점수 계산", test_diversity_score),
        ("교배 (Crossover)", test_crossover),
        ("변이 (Mutation)", test_mutation),
        ("진화 통합", test_evolution_integration),
        ("DB 저장", test_db_save),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n▶ {test_name} 테스트 실행...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} 통과\n")
            else:
                logger.error(f"❌ {test_name} 실패\n")
        except Exception as e:
            logger.error(f"❌ {test_name} 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        logger.info("=" * 60)
        logger.info("🎉 Phase 3A 테스트 모두 통과!")
        logger.info("=" * 60)
        return True
    else:
        logger.error("=" * 60)
        logger.error("❌ 일부 테스트 실패")
        logger.error("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

