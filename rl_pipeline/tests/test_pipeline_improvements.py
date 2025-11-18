"""
파이프라인 개선 사항 테스트 스크립트
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_candle_data(days: int = 30) -> pd.DataFrame:
    """샘플 캔들 데이터 생성"""
    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')
    
    # 간단한 랜덤 워크 데이터
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.01, len(dates))
    prices = base_price * (1 + returns).cumprod()
    
    # 고가/저가/종가 생성
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
    volume = np.random.uniform(1000000, 5000000, len(dates))
    
    df = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    # 기술 지표 추가
    df['rsi'] = 50 + np.random.normal(0, 15, len(df))
    df['macd'] = np.random.normal(0, 100, len(df))
    df['mfi'] = 50 + np.random.normal(0, 20, len(df))
    df['atr'] = prices * 0.01
    df['adx'] = 25 + np.random.normal(0, 10, len(df))
    df['bb_width'] = prices * 0.02
    df['volume_ratio'] = 1.0 + np.random.normal(0, 0.2, len(df))
    
    return df

def create_sample_strategies(count: int = 10) -> list:
    """샘플 전략 생성"""
    strategies = []
    
    for i in range(count):
        strategy = {
            'id': f'test_strategy_{i}',
            'strategy_id': f'test_strategy_{i}',
            'coin': 'BTCUSDT',
            'interval': '15m',
            'strategy_type': 'hybrid',
            'params': {
                'rsi_min': 30 + (i % 3) * 10,
                'rsi_max': 70 - (i % 3) * 5,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04 + (i % 2) * 0.01,
                'volume_ratio_min': 1.0,
                'volume_ratio_max': 2.0
            },
            'rsi_min': 30 + (i % 3) * 10,
            'rsi_max': 70 - (i % 3) * 5,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04 + (i % 2) * 0.01,
            'grade': ['S', 'A', 'B', 'C'][i % 4],
            'quality_grade': ['S', 'A', 'B', 'C'][i % 4],
            'profit': np.random.uniform(-0.05, 0.1),
            'win_rate': np.random.uniform(0.4, 0.7),
            'trades_count': np.random.randint(10, 100),
            'created_at': datetime.now().isoformat()
        }
        strategies.append(strategy)
    
    return strategies

def test_regime_detection():
    """레짐 감지 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 1: 인터벌별 레짐 감지")
    logger.info("=" * 60)
    
    try:
        from rl_pipeline.routing.regime_router import RegimeRouter
        
        router = RegimeRouter()
        candle_data = create_sample_candle_data(30)
        
        intervals = ['15m', '30m', '240m', '1d']
        interval_regimes = {}
        
        for interval in intervals:
            # 간단히 같은 데이터 사용 (실제로는 다른 인터벌 데이터여야 함)
            regime, confidence = router.detect_current_regime('BTCUSDT', interval, candle_data)
            interval_regimes[interval] = (regime, confidence)
            logger.info(f"✅ {interval}: 레짐={regime}, 신뢰도={confidence:.3f}")
        
        logger.info(f"✅ 레짐 감지 테스트 완료: {len(interval_regimes)}개 인터벌")
        return True
        
    except Exception as e:
        logger.error(f"❌ 레짐 감지 테스트 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_regime_alignment():
    """레짐 일치도 계산 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 2: 레짐 일치도 계산")
    logger.info("=" * 60)
    
    try:
        from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
        
        analyzer = IntegratedAnalyzer()
        
        # 시나리오 1: 모든 인터벌이 같은 레짐
        interval_regimes_1 = {
            '15m': ('bullish', 0.8),
            '30m': ('bullish', 0.75),
            '240m': ('bullish', 0.7),
            '1d': ('bullish', 0.85)
        }
        
        alignment_1, main_regime_1 = analyzer._calculate_regime_alignment(interval_regimes_1)
        logger.info(f"✅ 시나리오 1 (일치): 일치도={alignment_1:.3f}, 메인 레짐={main_regime_1}")
        
        # 시나리오 2: 인터벌이 다른 레짐
        interval_regimes_2 = {
            '15m': ('bullish', 0.8),
            '30m': ('bearish', 0.7),
            '240m': ('neutral', 0.6),
            '1d': ('bullish', 0.85)
        }
        
        alignment_2, main_regime_2 = analyzer._calculate_regime_alignment(interval_regimes_2)
        logger.info(f"✅ 시나리오 2 (불일치): 일치도={alignment_2:.3f}, 메인 레짐={main_regime_2}")
        
        logger.info(f"✅ 레짐 일치도 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 레짐 일치도 테스트 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_dynamic_weights():
    """동적 가중치 계산 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 3: 동적 가중치 계산")
    logger.info("=" * 60)
    
    try:
        from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
        
        analyzer = IntegratedAnalyzer()
        
        test_cases = [
            ('extreme_bullish', 'BTCUSDT', '15m'),
            ('bullish', 'BTCUSDT', '240m'),
            ('sideways_bullish', 'ETHUSDT', '30m'),
            ('neutral', 'BTCUSDT', '1d'),
        ]
        
        for regime, coin, interval in test_cases:
            weights = analyzer._calculate_dynamic_analysis_weights(regime, coin, interval)
            total = sum(weights.values())
            logger.info(f"✅ {regime}-{coin}-{interval}: {weights} (합계={total:.3f})")
        
        logger.info(f"✅ 동적 가중치 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 동적 가중치 테스트 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_backtest_cache():
    """백테스트 캐싱 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 4: 백테스트 캐싱")
    logger.info("=" * 60)
    
    try:
        from rl_pipeline.analysis.backtest_cache import get_backtest_cache
        
        cache = get_backtest_cache()
        
        strategy = create_sample_strategies(1)[0]
        candle_data = create_sample_candle_data(30)
        
        # 테스트 결과
        test_result = {
            'trades': 10,
            'profit': 0.05,
            'wins': 7,
            'win_rate': 0.7
        }
        
        # 캐시 저장
        cache.set(strategy, candle_data, test_result, 'bullish')
        logger.info("✅ 캐시 저장 완료")
        
        # 캐시 조회
        cached = cache.get(strategy, candle_data, 'bullish')
        if cached:
            logger.info(f"✅ 캐시 조회 성공: {cached}")
        else:
            logger.warning("⚠️ 캐시 조회 실패")
        
        # 통계
        stats = cache.get_stats()
        logger.info(f"✅ 캐시 통계: {stats}")
        
        logger.info(f"✅ 백테스트 캐싱 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 백테스트 캐싱 테스트 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_grade_updater():
    """전략 등급 업데이터 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 5: 전략 등급 업데이터")
    logger.info("=" * 60)
    
    try:
        from rl_pipeline.analysis.strategy_grade_updater import StrategyGradeUpdater
        
        updater = StrategyGradeUpdater()
        
        # 더미 라우팅 결과 생성
        from rl_pipeline.routing.regime_router import RegimeRoutingResult
        
        strategies = create_sample_strategies(5)
        routing_results = []
        
        for i, strategy in enumerate(strategies[:3]):
            result = RegimeRoutingResult(
                coin='BTCUSDT',
                interval='15m',
                regime='bullish',
                routed_strategy=strategy,
                routing_confidence=0.7 + i * 0.1,
                routing_score=0.6 + i * 0.1,
                regime_performance=0.65 + i * 0.1,
                regime_adaptation=0.7,
                created_at=datetime.now().isoformat()
            )
            routing_results.append(result)
        
        # 등급 업데이트 계산 (DB 업데이트는 하지 않음)
        grade_updates = updater.update_grades_from_routing_results(
            'BTCUSDT', '15m', routing_results
        )
        
        logger.info(f"✅ 등급 업데이트 계산 완료: {len(grade_updates)}개")
        for strategy_id, update_info in grade_updates.items():
            logger.info(f"  📈 {strategy_id}: {update_info['old_grade']} → {update_info['new_grade']} ({update_info['reason']})")
        
        logger.info(f"✅ 전략 등급 업데이터 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 전략 등급 업데이터 테스트 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_statistical_significance():
    """통계적 유의성 검증 테스트"""
    logger.info("=" * 60)
    logger.info("테스트 6: 통계적 유의성 검증")
    logger.info("=" * 60)
    
    try:
        from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
        
        analyzer = IntegratedAnalyzer()
        
        # 더미 맥락 분석 데이터
        context_analysis = {
            'cross_interval_performance': {
                'test_strategy_1': {
                    '15m': {'performance_score': 0.75, 'win_rate': 0.7, 'profit': 0.05},
                    '30m': {'performance_score': 0.72, 'win_rate': 0.68, 'profit': 0.04},
                },
                'test_strategy_2': {
                    '15m': {'performance_score': 0.68, 'win_rate': 0.65, 'profit': 0.03},
                    '30m': {'performance_score': 0.70, 'win_rate': 0.67, 'profit': 0.035},
                },
            }
        }
        
        strategies = create_sample_strategies(5)
        strategies[0]['grade'] = 'A'
        strategies[1]['grade'] = 'A'
        strategies[2]['grade'] = 'B'
        
        confidence = analyzer._calculate_context_based_confidence(
            '15m', context_analysis, strategies
        )
        
        logger.info(f"✅ 맥락 신뢰도 계산: {confidence:.3f}")
        logger.info(f"✅ 통계적 유의성 검증 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 통계적 유의성 검증 테스트 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def run_all_tests():
    """모든 테스트 실행"""
    logger.info("🚀 파이프라인 개선 사항 테스트 시작")
    logger.info("")
    
    tests = [
        ("레짐 감지", test_regime_detection),
        ("레짐 일치도", test_regime_alignment),
        ("동적 가중치", test_dynamic_weights),
        ("백테스트 캐싱", test_backtest_cache),
        ("전략 등급 업데이터", test_grade_updater),
        ("통계적 유의성", test_statistical_significance),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
            logger.info("")
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"총 {passed}/{total} 테스트 통과")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
