#!/usr/bin/env python
"""ADA RL 파이프라인 테스트 - 리팩토링 검증"""
import sys
sys.path.append('/workspace')

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/ada_pipeline_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """ADA 파이프라인 테스트 실행"""
    logger.info("=" * 80)
    logger.info("ADA RL Pipeline Test (Refactored Code Verification)")
    logger.info("=" * 80)

    try:
        # Orchestrator 생성
        orchestrator = IntegratedPipelineOrchestrator()

        # ADA 파이프라인 실행
        logger.info("Starting ADA pipeline with intervals: 15m, 30m, 240m, 1d")

        result = orchestrator.run_integrated_pipeline(
            coin='ADA',
            intervals=['15m', '30m', '240m', '1d']
        )

        logger.info("=" * 80)
        logger.info("Pipeline execution completed")
        logger.info(f"Result: {result}")
        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    result = main()
    if result:
        print("\n SUCCESS: Pipeline completed!")
    else:
        print("\n FAILURE: Pipeline failed!")
        sys.exit(1)
