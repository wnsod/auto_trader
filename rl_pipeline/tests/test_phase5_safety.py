"""
Phase 5 테스트: 안전장치 모듈 검증

실행 방법:
    docker exec -it auto_trader_coin bash
    cd /workspace
    python rl_pipeline/tests/test_phase5_safety.py
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rl_pipeline.db.transaction_manager import EvolutionTransactionManager
from rl_pipeline.strategy.rollback_manager import StrategyRollbackManager
from rl_pipeline.monitoring.evolution_logger import EvolutionLogger
from rl_pipeline.simulation.risk_controller import RiskController
from rl_pipeline.simulation.overfitting_prevention import OverfittingPrevention
from rl_pipeline.db.schema import create_strategies_table

logging.basicConfig(level=logging.WARNING)  # 경고만 출력
logger = logging.getLogger(__name__)


def test_transaction_manager():
    """트랜잭션 관리자 테스트"""
    print("\n[테스트 1] 트랜잭션 관리자")
    try:
        manager = EvolutionTransactionManager()
        print("  ✅ EvolutionTransactionManager 생성 성공")
        
        # 트랜잭션 컨텍스트 테스트
        with manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result:
                print("  ✅ 트랜잭션 컨텍스트 작동 확인")
                return True
            else:
                print("  ❌ 트랜잭션 테스트 실패")
                return False
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        return False


def test_rollback_manager():
    """롤백 관리자 테스트"""
    print("\n[테스트 2] 롤백 관리자")
    try:
        create_strategies_table()
        
        manager = StrategyRollbackManager()
        print("  ✅ StrategyRollbackManager 생성 성공")
        
        # 성과 하락 감지 테스트 (실제 전략이 없으면 False 반환)
        result = manager.detect_degradation('nonexistent_strategy')
        print(f"  ✅ 성과 하락 감지 메서드 작동 (결과: {result})")
        
        return True
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        return False


def test_evolution_logger():
    """진화 로거 테스트"""
    print("\n[테스트 3] 진화 로거")
    try:
        logger_obj = EvolutionLogger()
        print("  ✅ EvolutionLogger 생성 성공")
        
        # 세그먼트 결과 로깅 테스트
        logger_obj.log_segment_result(
            'test_strategy',
            {'start_idx': 0, 'end_idx': 100},
            {'profit': 100.0, 'pf': 1.5}
        )
        print("  ✅ 세그먼트 결과 로깅 성공")
        
        # 변이 로깅 테스트
        logger_obj.log_mutation(
            'parent_id',
            'child_id',
            {'rsi_min': 30.0, 'rsi_max': 70.0}
        )
        print("  ✅ 변이 로깅 성공")
        
        # 리포트 생성 테스트
        report = logger_obj.generate_evolution_report(['test_strategy'])
        print(f"  ✅ 리포트 생성 성공 (전략 수: {len(report.get('strategies', {}))})")
        
        return True
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_controller():
    """리스크 제어 테스트"""
    print("\n[테스트 4] 리스크 제어")
    try:
        controller = RiskController()
        print("  ✅ RiskController 생성 성공")
        
        # Drawdown 계산 테스트
        equity_curve = [10000.0, 11000.0, 9000.0, 9500.0, 8000.0]
        mdd = controller.calculate_drawdown(equity_curve)
        print(f"  ✅ Drawdown 계산: {mdd:.2%}")
        
        if mdd > 0.1:  # 10% 이상
            print("  ✅ 높은 Drawdown 감지 확인")
        
        # 포지션 축소 판단 테스트
        should_reduce = controller.should_reduce_position(mdd)
        print(f"  ✅ 포지션 축소 필요: {should_reduce}")
        
        # 포지션 크기 조정 테스트
        adjusted = controller.get_adjusted_position_size(1000.0, mdd)
        print(f"  ✅ 조정된 포지션 크기: {adjusted:.2f}")
        
        return True
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overfitting_prevention():
    """과적합 방지 테스트"""
    print("\n[테스트 5] 과적합 방지")
    try:
        prevention = OverfittingPrevention()
        print("  ✅ OverfittingPrevention 생성 성공")
        
        # 데이터 분할 테스트
        train_end, val_end, test_end = prevention.split_data(1000)
        print(f"  ✅ 데이터 분할: train={train_end}, val={val_end}, test={test_end}")
        
        if train_end == 800 and val_end == 900:
            print("  ✅ 분할 비율 정확 (80%/10%/10%)")
        
        # 검증 성과 확인 테스트
        should_stop, is_improving = prevention.check_validation_performance(0.8)
        print(f"  ✅ 검증 성과 확인: should_stop={should_stop}, improving={is_improving}")
        
        # 연속 하락 테스트
        prevention.check_validation_performance(0.7)  # 하락
        prevention.check_validation_performance(0.6)  # 하락
        should_stop, _ = prevention.check_validation_performance(0.5)  # 하락
        
        if should_stop:
            print("  ✅ 조기 종료 조건 감지 성공")
        
        return True
    except Exception as e:
        print(f"  ❌ 예외: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Phase 5: 안전장치 모듈 테스트")
    print("=" * 60)
    
    tests = [
        ("트랜잭션 관리자", test_transaction_manager),
        ("롤백 관리자", test_rollback_manager),
        ("진화 로거", test_evolution_logger),
        ("리스크 제어", test_risk_controller),
        ("과적합 방지", test_overfitting_prevention),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} 예외: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")
    
    print(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("=" * 60)
        print("🎉 Phase 5 모든 테스트 통과!")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("❌ 일부 테스트 실패")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)

