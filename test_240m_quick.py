#!/usr/bin/env python
"""240m 인터벌 빠른 테스트 - 재평가 로직 검증"""
import sys
sys.path.append('/workspace')

from rl_pipeline.pipelines.orchestrator import IntegratedPipelineOrchestrator
from rl_pipeline.db.candle_data import CandleDataManager
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """240m 인터벌 빠른 테스트"""
    logger.info("=" * 80)
    logger.info("240m Interval Quick Test (Reassessment Logic Verification)")
    logger.info("=" * 80)

    try:
        # Candle Data 로드
        candle_manager = CandleDataManager()
        candle_data = candle_manager.get_candles('ADA', '240m')

        logger.info(f"Loaded {len(candle_data)} candles for ADA-240m")

        # Orchestrator 생성
        orchestrator = IntegratedPipelineOrchestrator()

        # 240m 파이프라인 실행
        logger.info("Running 240m pipeline...")

        result = orchestrator.run_complete_pipeline(
            coin='ADA',
            interval='240m',
            candle_data=candle_data
        )

        logger.info("=" * 80)
        logger.info("240m Pipeline Results:")
        logger.info("=" * 80)

        if result:
            logger.info(f"Status: {result.status}")
            logger.info(f"Signal: {result.signal}")

            if hasattr(result, 'selfplay_result') and result.selfplay_result:
                logger.info(f"Self-Play Accuracy: {result.selfplay_result.get('avg_accuracy', 'N/A')}")

        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    result = main()

    print("\n" + "=" * 80)
    if result and result.status == 'success':
        print("SUCCESS: 240m pipeline completed!")

        if hasattr(result, 'selfplay_result') and result.selfplay_result:
            accuracy = result.selfplay_result.get('avg_accuracy', 0.0)
            print(f"Average Accuracy: {accuracy:.2%}")

            if accuracy > 0:
                print(" PASS: Accuracy is NOT 0%!")
            else:
                print(" WARNING: Accuracy is still 0%")
    else:
        print("FAILURE: 240m pipeline failed!")
        sys.exit(1)
    print("=" * 80)
