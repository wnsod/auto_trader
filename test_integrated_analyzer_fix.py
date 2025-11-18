"""
수정된 integrated_analyzer 테스트
실제 점수 계산 확인
"""
import sys
sys.path.insert(0, '/workspace')

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integrated_analyzer():
    """통합 분석기 테스트"""
    try:
        from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer

        logger.info("=" * 80)
        logger.info("🧪 통합 분석기 수정 테스트")
        logger.info("=" * 80)

        # 테스트용 캔들 데이터 생성
        dates = pd.date_range(end=datetime.now(), periods=200, freq='15min')

        # 가격 데이터 (상승 추세)
        base_price = 100.0
        price_trend = np.linspace(0, 10, 200)  # 10% 상승
        price_noise = np.random.randn(200) * 0.5  # 노이즈
        close_prices = base_price + price_trend + price_noise

        candle_data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.rand(200) * 0.5,
            'high': close_prices + np.random.rand(200) * 0.5,
            'low': close_prices - np.random.rand(200) * 0.5,
            'close': close_prices,
            'volume': np.random.randint(1000000, 2000000, 200),
            'rsi': 50 + np.random.randn(200) * 10,  # RSI
            'macd': np.random.randn(200) * 0.01,  # MACD
            'macd_signal': np.random.randn(200) * 0.01,
            'bb_upper': close_prices + 2.0,
            'bb_lower': close_prices - 2.0,
            'bb_width': np.full(200, 0.04),
            'atr': np.full(200, 0.02),
            'volume_ratio': 1.0 + np.random.rand(200) * 0.5,
            'mfi': 50 + np.random.randn(200) * 10,
            'adx': 25 + np.random.rand(200) * 10,
        })

        # 더미 전략
        strategies = [
            {
                'id': 'test_strategy_1',
                'params': {
                    'rsi_min': 30,
                    'rsi_max': 70,
                    'volume_ratio_min': 1.0,
                    'volume_ratio_max': 2.0
                },
                'quality_grade': 'A'
            }
        ]

        # 분석기 초기화
        analyzer = IntegratedAnalyzer(session_id="test_session")

        logger.info(f"\n📊 분석 시작...")
        logger.info(f"   - 코인: BTC")
        logger.info(f"   - 인터벌: 15m")
        logger.info(f"   - 캔들 수: {len(candle_data)}")
        logger.info(f"   - 전략 수: {len(strategies)}")

        # 분석 실행
        result = analyzer.analyze_coin_strategies(
            coin="BTC",
            interval="15m",
            regime="trending",
            strategies=strategies,
            candle_data=candle_data
        )

        logger.info(f"\n✅ 분석 완료!")
        logger.info(f"\n📊 분석 결과:")
        logger.info(f"   - fractal_score: {result.fractal_score:.4f}")
        logger.info(f"   - multi_timeframe_score: {result.multi_timeframe_score:.4f}")
        logger.info(f"   - indicator_cross_score: {result.indicator_cross_score:.4f}")
        logger.info(f"   - ensemble_score: {result.ensemble_score:.4f}")
        logger.info(f"   - final_signal_score: {result.final_signal_score:.4f}")
        logger.info(f"   - signal_action: {result.signal_action}")
        logger.info(f"   - signal_confidence: {result.signal_confidence:.4f}")

        # 검증
        logger.info(f"\n🔍 검증:")

        # 0.5가 아닌 값이 있는지 확인
        non_default_scores = []
        if result.fractal_score != 0.5:
            non_default_scores.append('fractal_score')
        if result.multi_timeframe_score != 0.5:
            non_default_scores.append('multi_timeframe_score')
        if result.indicator_cross_score != 0.5:
            non_default_scores.append('indicator_cross_score')

        if non_default_scores:
            logger.info(f"   ✅ 실제 계산된 점수 발견: {', '.join(non_default_scores)}")
        else:
            logger.warning(f"   ⚠️ 모든 점수가 기본값(0.5)")

        # 점수가 합리적인 범위인지 확인
        scores = [result.fractal_score, result.multi_timeframe_score, result.indicator_cross_score]
        if all(0.0 <= s <= 1.0 for s in scores):
            logger.info(f"   ✅ 모든 점수가 유효 범위 (0.0 ~ 1.0)")
        else:
            logger.error(f"   ❌ 점수 범위 오류")

        logger.info("\n" + "=" * 80)
        logger.info("✅ 테스트 완료")
        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    result = test_integrated_analyzer()
    if result and (result.fractal_score != 0.5 or result.multi_timeframe_score != 0.5):
        logger.info("\n🎉 수정 성공! 실제 점수 계산이 작동합니다.")
    else:
        logger.error("\n❌ 수정 실패! 여전히 기본값이 반환됩니다.")
