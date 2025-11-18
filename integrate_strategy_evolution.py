"""
전략 진화 시스템을 orchestrator에 통합하는 패치
파이프라인 실행 시 자동으로 전략 진화 수행
"""
import sys
sys.path.insert(0, '/workspace')

import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_evolution_integration_code():
    """orchestrator.py에 추가할 전략 진화 통합 코드 생성"""

    integration_code = '''
    def _evolve_existing_strategies(self, coin: str, interval: str, new_strategies: List[Dict]) -> List[Dict]:
        """
        기존 전략을 진화시켜 새로운 전략 생성

        Args:
            coin: 코인
            interval: 인터벌
            new_strategies: 새로 생성된 전략 리스트

        Returns:
            진화된 전략 리스트
        """
        try:
            # 환경변수로 진화 활성화 여부 확인
            import os
            enable_evolution = os.getenv('ENABLE_STRATEGY_EVOLUTION', 'true').lower() == 'true'

            if not enable_evolution:
                logger.debug(f"⏭️ {coin}-{interval}: 전략 진화 비활성화")
                return []

            logger.info(f"🧬 {coin}-{interval}: 전략 진화 시작")

            # StrategyEvolver import
            from rl_pipeline.strategy.strategy_evolver import StrategyEvolver
            from rl_pipeline.db.connection_pool import get_strategy_db_pool

            # 기존 전략 조회 (DB에서)
            pool = get_strategy_db_pool()
            with pool.get_connection() as conn:
                cursor = conn.cursor()

                # 상위 등급 전략만 조회 (S, A, B)
                cursor.execute("""
                    SELECT
                        cs.id as strategy_id,
                        cs.coin,
                        cs.interval,
                        cs.params,
                        cs.regime,
                        sg.grade as quality_grade,
                        sr.avg_ret,
                        sr.win_rate,
                        sr.predictive_accuracy
                    FROM coin_strategies cs
                    LEFT JOIN strategy_grades sg ON cs.id = sg.strategy_id
                    LEFT JOIN rl_strategy_rollup sr ON cs.id = sr.strategy_id
                    WHERE cs.coin = ?
                      AND cs.interval = ?
                      AND sg.grade IN ('S', 'A', 'B')
                    ORDER BY sg.grade_score DESC
                    LIMIT 100
                """, (coin, interval))

                rows = cursor.fetchall()

                if not rows:
                    logger.debug(f"⏭️ {coin}-{interval}: 진화 가능한 상위 전략 없음")
                    return []

                # Dict로 변환
                import json
                existing_strategies = []
                for row in rows:
                    strategy_dict = {
                        'strategy_id': row[0],
                        'coin': row[1],
                        'interval': row[2],
                        'params': json.loads(row[3]) if row[3] else {},
                        'regime': row[4],
                        'quality_grade': row[5] or 'UNKNOWN',
                        'avg_ret': row[6] or 0.0,
                        'win_rate': row[7] or 0.0,
                        'predictive_accuracy': row[8] or 0.0
                    }
                    existing_strategies.append(strategy_dict)

                logger.info(f"📊 {coin}-{interval}: 진화 대상 전략 {len(existing_strategies)}개 발견")

            # StrategyEvolver 초기화
            evolver = StrategyEvolver()

            # 상위 전략 선별
            top_strategies = evolver.select_top_strategies(
                existing_strategies,
                top_percent=0.3,  # 상위 30%
                min_grade='B'     # B 등급 이상
            )

            if not top_strategies:
                logger.debug(f"⏭️ {coin}-{interval}: 선별된 상위 전략 없음")
                return []

            logger.info(f"✅ {coin}-{interval}: 상위 전략 {len(top_strategies)}개 선별")

            # 진화 실행 (교배 + 변이)
            # 최대 5개의 진화된 전략 생성
            max_evolved = min(5, len(top_strategies) // 2)
            evolved_strategies = []

            for i in range(max_evolved):
                try:
                    # 랜덤으로 두 부모 선택
                    import random
                    parent1 = random.choice(top_strategies)
                    parent2 = random.choice(top_strategies)

                    # 교배
                    child_params = evolver.crossover(parent1, parent2)

                    # 변이
                    mutated_params = evolver.mutate(child_params)

                    # 진화된 전략 생성
                    evolved_strategy = {
                        'params': mutated_params,
                        'coin': coin,
                        'interval': interval,
                        'regime': parent1.get('regime', 'neutral'),
                        'parent_strategy_id': parent1.get('strategy_id'),
                        'similarity_classification': 'evolved',
                        'similarity_score': 0.7  # 진화된 전략은 부모와 유사
                    }

                    evolved_strategies.append(evolved_strategy)
                    logger.debug(f"🧬 진화 전략 #{i+1} 생성 (부모: {parent1.get('strategy_id')[:8]}...)")

                except Exception as e:
                    logger.warning(f"⚠️ 진화 전략 생성 실패: {e}")
                    continue

            if evolved_strategies:
                logger.info(f"✅ {coin}-{interval}: {len(evolved_strategies)}개 진화 전략 생성 완료")

            return evolved_strategies

        except Exception as e:
            logger.error(f"❌ {coin}-{interval}: 전략 진화 실패: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    '''

    return integration_code


def print_integration_instructions():
    """통합 방법 안내"""

    instructions = """
================================================================================
전략 진화 시스템 통합 방법
================================================================================

1. orchestrator.py의 IntegratedPipelineOrchestrator 클래스에 메서드 추가:
   - 위치: 클래스 내부 어디든 (예: line 2700 근처)
   - 추가할 메서드: _evolve_existing_strategies()

2. run_complete_pipeline() 메서드 수정:
   - 위치: line 694 근처
   - 수정 전:
     ```python
     # 1단계: 전략 생성
     strategies = self._create_strategies(coin, interval, candle_data)
     logger.info(f"✅ {len(strategies)}개 전략 생성 완료")
     ```

   - 수정 후:
     ```python
     # 1단계: 전략 생성
     strategies = self._create_strategies(coin, interval, candle_data)
     logger.info(f"✅ {len(strategies)}개 전략 생성 완료")

     # 🧬 1-1단계: 기존 전략 진화
     evolved_strategies = self._evolve_existing_strategies(coin, interval, strategies)
     if evolved_strategies:
         strategies.extend(evolved_strategies)
         logger.info(f"🧬 {len(evolved_strategies)}개 진화 전략 추가 (총 {len(strategies)}개)")
     ```

3. run_partial_pipeline() 메서드도 동일하게 수정:
   - 위치: line 2723 근처
   - 동일한 코드 추가

4. 환경변수 설정 (선택):
   - ENABLE_STRATEGY_EVOLUTION=true (기본값)
   - EVOLUTION_TOP_PERCENT=0.3
   - MUTATION_STRENGTH=0.1

5. 검증:
   - 파이프라인 실행 시 "🧬 전략 진화 시작" 로그 확인
   - parent_strategy_id 필드에 부모 전략 ID 저장 확인

================================================================================
"""

    print(instructions)
    print("\n생성된 코드:")
    print(create_evolution_integration_code())


if __name__ == "__main__":
    print_integration_instructions()
