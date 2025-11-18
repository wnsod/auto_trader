"""
전략 생성 모듈
다양한 전략 생성 및 글로벌 전략 관리
"""

import logging
import os
import pandas as pd
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from rl_pipeline.core.types import Strategy, StrategyResult
from rl_pipeline.core.errors import StrategyError
from rl_pipeline.core.env import config
from rl_pipeline.core.utils import format_strategy_data
from rl_pipeline.data import load_candles, ensure_indicators
from rl_pipeline.strategy.param_space import sample_param_grid
from rl_pipeline.strategy.factory import make_strategy
from rl_pipeline.strategy.serializer import serialize_strategy
from rl_pipeline.db.writes import write_batch
from rl_pipeline.db.connection_pool import get_optimized_db_connection

# 레짐 기반 전략 관리
from rl_pipeline.core.regime_strategy_manager import get_target_regime_for_generation

# 증분 학습: 유사도 기반 전략 분류
from rl_pipeline.strategy.similarity import (
    classify_new_strategies_batch,
    calculate_smart_similarity,
    classify_strategy_by_similarity
)

# 개선 모듈 import (선택적)
try:
    from rl_pipeline.strategy.creator_enhancements import (
        filter_duplicate_strategies,
        create_grid_search_strategies,
        create_direction_specialized_strategies,
        create_enhanced_strategies_with_diversity,
        generate_strategy_hash
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    logger.warning("⚠️ 전략 생성 개선 모듈을 사용할 수 없습니다")

logger = logging.getLogger(__name__)

# 🚀 통합 분석 기반 지표 그룹 분류
INDICATOR_GROUPS = {
    'A': {  # 모멘텀/오실레이터
        'rsi': {'type': 'range', 'default_min': 30.0, 'default_max': 70.0},
        'macd': {'type': 'threshold', 'default_buy': 0.0, 'default_sell': 0.0},
        'macd_signal': {'type': 'threshold', 'default_buy': 0.0, 'default_sell': 0.0},
        'mfi': {'type': 'range', 'default_min': 20.0, 'default_max': 80.0}
    },
    'B': {  # 거래량/변동성
        'volume_ratio': {'type': 'range', 'default_min': 1.0, 'default_max': 2.0},
        'atr': {'type': 'range', 'default_min': 0.01, 'default_max': 0.05},
        'adx': {'type': 'threshold', 'default_min': 15.0}
    },
    'C': {  # 구조적/패턴
        'bb_position': {'type': 'range', 'default_min': 0.0, 'default_max': 1.0},
        'wave_phase': {'type': 'categorical', 'values': ['impulse', 'correction', 'consolidation', 'unknown']},
        'pattern_confidence': {'type': 'threshold', 'default_min': 0.5},
        'integrated_direction': {'type': 'categorical', 'values': ['bullish', 'bearish', 'neutral', 'strong_bullish', 'strong_bearish', 'mixed']}
    },
    'D': {  # 레짐/심리도
        'sentiment': {'type': 'range', 'default_min': -1.0, 'default_max': 1.0},
        'regime_confidence': {'type': 'threshold', 'default_min': 0.4}
    }
}

# 🚀 허용된 그룹 조합
ALLOWED_GROUP_COMBINATIONS = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('A', 'D'),
    ('B', 'D'),
    ('C', 'D')
]

# 🚀 레짐 기반 전략 타입 비율
REGIME_STRATEGY_TYPE_RATIOS = {
    'bullish': {'aggressive': 0.4, 'balanced': 0.5, 'conservative': 0.1},
    'bearish': {'aggressive': 0.1, 'balanced': 0.4, 'conservative': 0.5},
    'neutral': {'aggressive': 0.2, 'balanced': 0.6, 'conservative': 0.2},
    'volatile': {'aggressive': 0.5, 'balanced': 0.3, 'conservative': 0.2},
    'extreme_bullish': {'aggressive': 0.5, 'balanced': 0.4, 'conservative': 0.1},
    'extreme_bearish': {'aggressive': 0.1, 'balanced': 0.3, 'conservative': 0.6},
    'sideways_bullish': {'aggressive': 0.3, 'balanced': 0.5, 'conservative': 0.2},
    'sideways_bearish': {'aggressive': 0.2, 'balanced': 0.5, 'conservative': 0.3}
}

# 🚀 전략 타입별 최소 조건 수
STRATEGY_TYPE_MIN_CONDITIONS = {
    'aggressive': 2,  # 2개 조건 (각 그룹에서 1개씩)
    'balanced': (2, 3),  # 2~3개 조건 (유연하게)
    'conservative': 3  # 3개 조건 (각 그룹에서 최소 1개, 추가 1개)
}


def analyze_market(coin: str, interval: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    시장 분석 (Claude 제안 기반)
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        df: 캔들 데이터 DataFrame
    
    Returns:
        시장 분석 결과 딕셔너리 또는 None
    """
    try:
        # 🔥 인터벌별 최소 데이터 기준 (60일 기준으로 현실적으로 조정)
        # 60일 기준 최대 캔들 수: 15m=5760, 30m=2880, 240m=360, 1d=60
        # 실제 사용 인터벌: 15m, 30m, 240m, 1d만 존재
        min_data_by_interval = {
            '15m': 30,   # 약 7.5시간 (최소 30개)
            '30m': 20,   # 약 10시간 (최소 20개)
            '240m': 15,  # 약 2.5일 (최소 15개)
            '1d': 5,     # 약 5일 (최소 5개)
        }
        
        min_required = min_data_by_interval.get(interval, 30)  # 기본값: 30개
        
        if df.empty or len(df) < min_required:
            logger.warning(f"⚠️ {coin} {interval}: 시장 분석 데이터 부족 (최소 {min_required}개 필요, 현재 {len(df)}개)")
            return None
        
        import numpy as np
        
        # 최근 1000개 캔들 분석 (또는 전체 데이터)
        analysis_df = df.tail(min(1000, len(df)))
        
        # 가격 통계
        closes = analysis_df['close'].values
        price_mean = np.mean(closes)
        price_std = np.std(closes)
        price_trend = (closes[-1] - closes[0]) / closes[0] if len(closes) > 0 else 0.0
        
        # RSI 통계
        rsis = analysis_df['rsi'].dropna().values if 'rsi' in analysis_df.columns else []
        if len(rsis) > 0:
            rsi_mean = np.mean(rsis)
            rsi_std = np.std(rsis)
            rsi_25 = np.percentile(rsis, 25)
            rsi_75 = np.percentile(rsis, 75)
        else:
            rsi_mean = rsi_std = rsi_25 = rsi_75 = 50.0
        
        # 거래량 통계
        volumes = analysis_df['volume'].values if 'volume' in analysis_df.columns else []
        if len(volumes) > 0:
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)
        else:
            volume_mean = volume_std = 1.0

        # Volume Ratio 통계 (실제 전략 조건에 사용)
        volume_ratios = analysis_df['volume_ratio'].dropna().values if 'volume_ratio' in analysis_df.columns else []
        if len(volume_ratios) > 0:
            volume_ratio_mean = np.mean(volume_ratios)
            volume_ratio_std = np.std(volume_ratios)
            volume_ratio_25 = np.percentile(volume_ratios, 25)
            volume_ratio_75 = np.percentile(volume_ratios, 75)
        else:
            volume_ratio_mean = 1.0
            volume_ratio_std = 0.5
            volume_ratio_25 = 0.8
            volume_ratio_75 = 1.5

        # ATR 통계
        atrs = analysis_df['atr'].dropna().values if 'atr' in analysis_df.columns else []
        if len(atrs) > 0:
            atr_mean = np.mean(atrs)
            atr_std = np.std(atrs)
        else:
            atr_mean = atr_std = 0.02
        
        # 변동성 계산 (가격 변화율의 표준편차)
        if len(closes) > 1:
            price_changes = np.diff(closes) / closes[:-1]
            volatility = np.std(price_changes)
        else:
            volatility = 0.02
        
        # 시장 상황 판단
        if price_trend > 0.05:
            market_condition = 'bullish'
        elif price_trend < -0.05:
            market_condition = 'bearish'
        else:
            market_condition = 'neutral'
        
        analysis = {
            'price_mean': float(price_mean),
            'price_std': float(price_std),
            'price_trend': float(price_trend),  # 전체 추세 (%)

            'rsi_mean': float(rsi_mean),
            'rsi_std': float(rsi_std),
            'rsi_25': float(rsi_25),
            'rsi_75': float(rsi_75),

            'volume_mean': float(volume_mean),
            'volume_std': float(volume_std),

            # Volume Ratio 통계 추가 (전략 조건용)
            'volume_ratio_mean': float(volume_ratio_mean),
            'volume_ratio_std': float(volume_ratio_std),
            'volume_ratio_25': float(volume_ratio_25),
            'volume_ratio_75': float(volume_ratio_75),

            'atr_mean': float(atr_mean),
            'atr_std': float(atr_std),

            'volatility': float(volatility),  # 변동성 (%)

            'market_condition': market_condition
        }
        
        # 🔥 DEBUG 레벨로 변경 (전략 생성마다 호출되므로 중복 로그 방지)
        logger.debug(f"📊 {coin} {interval} 시장 분석:")
        logger.debug(f"  추세: {market_condition} ({price_trend*100:.2f}%)")
        logger.debug(f"  RSI: {rsi_mean:.1f} ± {rsi_std:.1f} (25%: {rsi_25:.1f}, 75%: {rsi_75:.1f})")
        logger.debug(f"  Volume Ratio: {volume_ratio_mean:.2f} ± {volume_ratio_std:.2f} (25%: {volume_ratio_25:.2f}, 75%: {volume_ratio_75:.2f})")
        logger.debug(f"  변동성: {volatility*100:.2f}%")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ {coin} {interval} 시장 분석 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_market_adaptive_strategies(
    coin: str,
    interval: str,
    df: pd.DataFrame,
    count: int = 50,
    regime: str = "ranging"
) -> List[Strategy]:
    """
    시장 분석 기반 전략 생성 (Claude 제안)

    Args:
        coin: 코인 심볼
        interval: 인터벌
        df: 캔들 데이터 DataFrame
        count: 생성할 전략 수
        regime: 타겟 레짐 (ranging, trending, volatile, 기본값: ranging)

    Returns:
        생성된 전략 리스트
    """
    try:
        from rl_pipeline.core.types import Strategy
        from rl_pipeline.strategy.strategy_evolver import StrategyEvolver
        
        # 시장 분석
        market = analyze_market(coin, interval, df)
        
        if not market:
            logger.warning(f"⚠️ {coin} {interval} 시장 데이터 없음, 기본 전략 생성")
            strategies = []
            for i in range(count):
                strategy = create_basic_strategy(coin, interval, regime=regime)
                strategies.append(strategy)
            return strategies
        
        strategies = []
        evolver = StrategyEvolver()
        
        # 1. 과매도 전략 (RSI < 30) - 20%
        oversold_count = count // 5
        for i in range(oversold_count):
            strategy_params = {
                # RSI: 시장 평균 - 1.5σ 근처
                'rsi_min': max(10.0, market['rsi_mean'] - 1.5 * market['rsi_std'] - 10.0),
                'rsi_max': max(20.0, market['rsi_mean'] - 1.0 * market['rsi_std']),

                # SL/TP: 변동성 기반
                'stop_loss_pct': max(0.01, min(0.05, market['volatility'] * 1.5)),
                'take_profit_pct': max(1.02, min(1.10, 1.0 + market['volatility'] * 3.0)),

                'volume_ratio_min': max(0.5, market.get('volume_ratio_25', 0.8)),
                'volume_ratio_max': min(5.0, market.get('volume_ratio_75', 1.5)),
            }

            # 파라미터 검증 및 수정
            strategy_params = evolver._clip_and_fix_parameters(strategy_params)
            
            # Strategy 객체 생성
            strategy = Strategy(
                id=f"{coin}_{interval}_oversold_{i}",
                params=strategy_params,
                version="v2.0",
                created_at=datetime.now(),
                coin=coin,
                interval=interval,
                strategy_type='oversold',
                rsi_min=strategy_params.get('rsi_min', 30.0),
                rsi_max=strategy_params.get('rsi_max', 70.0),
                volume_ratio_min=strategy_params.get('volume_ratio_min', 1.0),
                volume_ratio_max=strategy_params.get('volume_ratio_max', 2.0),
                stop_loss_pct=strategy_params.get('stop_loss_pct', 0.15),
                take_profit_pct=strategy_params.get('take_profit_pct', 1.50),
                regime=regime,
            )
            strategies.append(strategy)

        # 2. 과매수 전략 (RSI > 70) - 20%
        overbought_count = count // 5
        for i in range(overbought_count):
            strategy_params = {
                # RSI: 시장 평균 + 1.0σ 근처
                'rsi_min': min(70.0, market['rsi_mean'] + 1.0 * market['rsi_std']),
                'rsi_max': min(90.0, market['rsi_mean'] + 1.5 * market['rsi_std'] + 10.0),

                'stop_loss_pct': max(0.01, min(0.05, market['volatility'] * 1.5)),
                'take_profit_pct': max(1.02, min(1.10, 1.0 + market['volatility'] * 3.0)),

                'volume_ratio_min': max(0.5, market.get('volume_ratio_25', 0.8)),
                'volume_ratio_max': min(5.0, market.get('volume_ratio_75', 1.5)),
            }

            strategy_params = evolver._clip_and_fix_parameters(strategy_params)
            
            strategy = Strategy(
                id=f"{coin}_{interval}_overbought_{i}",
                params=strategy_params,
                version="v2.0",
                created_at=datetime.now(),
                coin=coin,
                interval=interval,
                strategy_type='overbought',
                rsi_min=strategy_params.get('rsi_min', 30.0),
                rsi_max=strategy_params.get('rsi_max', 70.0),
                volume_ratio_min=strategy_params.get('volume_ratio_min', 1.0),
                volume_ratio_max=strategy_params.get('volume_ratio_max', 2.0),
                stop_loss_pct=strategy_params.get('stop_loss_pct', 0.15),
                take_profit_pct=strategy_params.get('take_profit_pct', 1.50),
                regime=regime,
            )
            strategies.append(strategy)

        # 3. 평균 회귀 전략 - 20%
        mean_reversion_count = count // 5
        for i in range(mean_reversion_count):
            strategy_params = {
                # RSI: 시장 평균 ± 0.5σ
                'rsi_min': max(0.0, market['rsi_mean'] - 0.5 * market['rsi_std']),
                'rsi_max': min(100.0, market['rsi_mean'] + 0.5 * market['rsi_std']),

                'stop_loss_pct': max(0.01, min(0.03, market['volatility'] * 1.0)),
                'take_profit_pct': max(1.01, min(1.05, 1.0 + market['volatility'] * 2.0)),

                'volume_ratio_min': max(0.5, market.get('volume_ratio_25', 0.7) * 0.9),
                'volume_ratio_max': min(4.0, market.get('volume_ratio_75', 1.5) * 1.2),
            }
            
            strategy_params = evolver._clip_and_fix_parameters(strategy_params)
            
            strategy = Strategy(
                id=f"{coin}_{interval}_mean_reversion_{i}",
                params=strategy_params,
                version="v2.0",
                created_at=datetime.now(),
                coin=coin,
                interval=interval,
                strategy_type='mean_reversion',
                rsi_min=strategy_params.get('rsi_min', 30.0),
                rsi_max=strategy_params.get('rsi_max', 70.0),
                volume_ratio_min=strategy_params.get('volume_ratio_min', 1.0),
                volume_ratio_max=strategy_params.get('volume_ratio_max', 2.0),
                stop_loss_pct=strategy_params.get('stop_loss_pct', 0.15),
                take_profit_pct=strategy_params.get('take_profit_pct', 1.50),
                regime=regime,
            )
            strategies.append(strategy)

        # 4. 추세 추종 전략 (bullish 시장) - 20%
        trend_following_count = count // 5
        if market['market_condition'] == 'bullish':
            for i in range(trend_following_count):
                import random
                strategy_params = {
                    # RSI: 중간~높음
                    'rsi_min': 45.0 + random.uniform(-5, 5),
                    'rsi_max': 65.0 + random.uniform(-5, 5),

                    'stop_loss_pct': max(0.02, min(0.06, market['volatility'] * 2.0)),
                    'take_profit_pct': max(1.03, min(1.15, 1.0 + market['volatility'] * 4.0)),

                    'volume_ratio_min': max(0.8, market.get('volume_ratio_mean', 1.0) * 0.9),
                    'volume_ratio_max': min(5.0, market.get('volume_ratio_75', 1.5) * 1.5),
                }
                
                strategy_params = evolver._clip_and_fix_parameters(strategy_params)
                
                strategy = Strategy(
                    id=f"{coin}_{interval}_trend_following_{i}",
                    params=strategy_params,
                    version="v2.0",
                    created_at=datetime.now(),
                    coin=coin,
                    interval=interval,
                    strategy_type='trend_following',
                    rsi_min=strategy_params.get('rsi_min', 30.0),
                    rsi_max=strategy_params.get('rsi_max', 70.0),
                    volume_ratio_min=strategy_params.get('volume_ratio_min', 1.0),
                    volume_ratio_max=strategy_params.get('volume_ratio_max', 2.0),
                    stop_loss_pct=strategy_params.get('stop_loss_pct', 0.15),
                    take_profit_pct=strategy_params.get('take_profit_pct', 1.50),
                )
                strategies.append(strategy)
        else:
            # bullish가 아니면 랜덤 전략으로 대체
            for i in range(trend_following_count):
                strategy = create_guided_random_strategy(coin, interval, df, market['market_condition'], index=i, regime=regime)
                strategies.append(strategy)

        # 5. 랜덤 전략 (다양성 확보) - 나머지
        remaining = count - len(strategies)
        for i in range(remaining):
            strategy = create_guided_random_strategy(coin, interval, df, market['market_condition'], index=len(strategies) + i, regime=regime)
            strategies.append(strategy)
        
        logger.info(f"✅ {coin} {interval} 시장 분석 기반 전략 생성 완료: {len(strategies)}개")
        return strategies
        
    except Exception as e:
        logger.error(f"❌ {coin} {interval} 시장 분석 기반 전략 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def _save_strategies_to_db_lazy(strategies):
    """전략 저장 (circular import 방지를 위한 지연 import)"""
    from rl_pipeline.strategy.manager import save_strategies_to_db
    return save_strategies_to_db(strategies)


def create_intelligent_strategies_with_type(coin: str, interval: str, num_strategies: int, df: pd.DataFrame, strategy_type: str = "general") -> List[Strategy]:

    """🚀 타입별 지능형 전략 생성 (장기/단기 전반/단기 후반/단기만)"""

    try:

        strategies = []

        

        # 🆕 받은 데이터 검증 로그 추가

        logger.info(f"🔍 {coin} {interval} {strategy_type} 전략 생성 - 받은 데이터 검증:")

        logger.info(f"  - 데이터 개수: {len(df)}")

        logger.info(f"  - 컬럼 목록: {list(df.columns)}")

        

        # 필수 기술지표 확인

        required_indicators = ['rsi', 'volume_ratio', 'macd', 'macd_signal', 'mfi', 'atr', 'adx']

        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

        if missing_indicators:

            logger.warning(f"⚠️ {coin} {interval} 받은 데이터에 누락된 기술지표: {missing_indicators}")

        else:

            logger.info(f"✅ {coin} {interval} 받은 데이터에 모든 필수 기술지표 존재")

        

        # 1. 시장 상황 분석

        market_condition = classify_market_condition(df)

        logger.info(f"📊 {coin} {interval} {strategy_type} 시장 상황: {market_condition}")


        # 레짐 결정 (타겟 레짐 조회)
        regime = get_target_regime_for_generation(coin, interval)
        logger.info(f"📍 {coin} {interval} {strategy_type} 타겟 레짐: {regime}")

        # 2. 전략 타입별 특화 파라미터 조정

        if strategy_type == "long_term":

            # 장기 전략: 안정성 중심

            ai_ratio = 0.8  # AI 비율 높임

            risk_level = "low"

        elif strategy_type == "short_term_front":

            # 단기 전반: 초기 시장 상황 특화

            ai_ratio = 0.6

            risk_level = "medium"

        elif strategy_type == "short_term_back":

            # 단기 후반: 후기 시장 상황 특화

            ai_ratio = 0.6

            risk_level = "medium"

        elif strategy_type == "short_term_only":

            # 단기만: 전체 기간 단기 특화

            ai_ratio = 0.7

            risk_level = "medium"

        else:

            # 일반 전략

            ai_ratio = 0.5

            risk_level = "medium"

        

        # 3. 실제 데이터 기반 전략 생성

        if not df.empty and len(df) > 20:

            # 실제 지표값 계산

            avg_rsi = df['rsi'].mean()

            rsi_std = df['rsi'].std()

            avg_volume_ratio = df['volume_ratio'].mean()

            volume_std = df['volume_ratio'].std()

            

            logger.info(f"📈 {coin} {interval} {strategy_type} 실제 지표값:")

            logger.info(f"  - RSI: 평균={avg_rsi:.1f}, 표준편차={rsi_std:.1f}")

            logger.info(f"  - Volume: 평균={avg_volume_ratio:.2f}, 표준편차={volume_std:.2f}")

            

            # 동적 비율 계산

            intelligent_count = int(num_strategies * ai_ratio)

            random_count = num_strategies - intelligent_count

            

            logger.info(f"🎯 {coin} {interval} {strategy_type} 동적 비율: AI {intelligent_count}개 ({ai_ratio:.1%}), 랜덤 {random_count}개 ({1-ai_ratio:.1%})")

            

            # 🆕 시장 분석 기반 전략 생성 (Claude 제안) - 30% 할당
            market_adaptive_count = int(intelligent_count * 0.3)
            remaining_intelligent_count = intelligent_count - market_adaptive_count
            
            # 🆕 시장 분석 기반 전략 생성 (Claude 제안)
            if market_adaptive_count > 0:
                try:
                    market_strategies = create_market_adaptive_strategies(coin, interval, df, market_adaptive_count, regime=regime)
                    for strategy in market_strategies:
                        if strategy is not None:
                            strategy.risk_level = risk_level
                            strategy.strategy_type = strategy_type
                            strategies.append(strategy)
                    logger.info(f"✅ {coin} {interval} {strategy_type} 시장 분석 기반 전략 {len(market_strategies)}개 생성 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {coin} {interval} {strategy_type} 시장 분석 기반 전략 생성 실패, 기본 전략으로 대체: {e}")
                    # 실패 시 기본 전략으로 대체
                    for i in range(market_adaptive_count):
                        strategy_pattern = select_ai_strategy_pattern(market_condition, i, market_adaptive_count)
                        strategy = create_enhanced_market_adaptive_strategy(coin, interval, market_condition, strategy_pattern, df, index=i)
                        if strategy is not None:
                            strategy.risk_level = risk_level
                            strategy.strategy_type = strategy_type
                            strategies.append(strategy)
            
            # AI 전략 생성 (나머지)
            for i in range(remaining_intelligent_count):

                strategy_pattern = select_ai_strategy_pattern(market_condition, i, remaining_intelligent_count)

                strategy = create_enhanced_market_adaptive_strategy(coin, interval, market_condition, strategy_pattern, df, index=i)

                

                # 전략 타입별 특화 조정

                strategy.risk_level = risk_level

                strategy.strategy_type = strategy_type

                

                strategies.append(strategy)

            

            # 랜덤 전략 생성

            for i in range(random_count):

                strategy = create_guided_random_strategy(coin, interval, df, market_condition, index=intelligent_count + i, regime=regime)

                strategy.risk_level = risk_level

                strategy.strategy_type = strategy_type

                strategies.append(strategy)

        

        else:

            # 데이터 부족한 경우 기본 전략 생성

            logger.warning(f"⚠️ {coin} {interval} {strategy_type}: 데이터 부족, 기본 전략 생성")

            for i in range(num_strategies):

                strategy = create_basic_strategy(coin, interval)

                strategy.strategy_type = strategy_type

                strategies.append(strategy)

        

        logger.info(f"✅ {coin} {interval} {strategy_type} 전략 생성 완료: {len(strategies)}개")

        return strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} {strategy_type} 전략 생성 실패: {e}")

        return []



def create_intelligent_strategies(coin: str, interval: str, num_strategies: int, df: pd.DataFrame, regime: str = "ranging", suppress_detailed_logs: bool = False) -> List[Strategy]:
    """
    지능형 전략 생성

    Args:
        coin: 코인 심볼 (예: 'BTC')
        interval: 시간 간격 (예: '15m')
        num_strategies: 생성할 전략 수
        df: 캔들 데이터 DataFrame (필수 기술지표 포함)
        regime: 타겟 레짐 (ranging, trending, volatile, 기본값: ranging)
        suppress_detailed_logs: 상세 로그 억제 (추가 생성 시 True)

    Returns:
        생성된 전략 리스트

    Raises:
        ValueError: 파라미터가 유효하지 않을 때
    """
    try:
        # 공통 파라미터 검증 사용
        from rl_pipeline.strategy.common import StrategyCreationHelper
        StrategyCreationHelper.validate_params(coin, interval, num_strategies, df)

        strategies = []

        # 🆕 받은 데이터 검증 로그 추가 (상세 로그 억제 모드가 아닐 때만)
        if not suppress_detailed_logs:
            logger.info(f"🔍 {coin} {interval} 받은 데이터 검증:")
            logger.info(f"  - 데이터 개수: {len(df)}")
            logger.info(f"  - 컬럼 목록: {list(df.columns)}")

        # 필수 기술지표 확인
        required_indicators = ['rsi', 'volume_ratio', 'macd', 'macd_signal', 'mfi', 'atr', 'adx']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        if missing_indicators:
            logger.warning(f"⚠️ {coin} {interval} 받은 데이터에 누락된 기술지표: {missing_indicators}")
        elif not suppress_detailed_logs:
            logger.info(f"✅ {coin} {interval} 받은 데이터에 모든 필수 기술지표 존재")
        
        # 🚀 통합 분석 지표 확인 (선택적, 없어도 전략 생성 가능)
        integrated_indicators = ['bb_position', 'wave_phase', 'pattern_confidence', 'integrated_direction', 
                                'sentiment', 'regime_confidence', 'regime_label']
        available_integrated = [ind for ind in integrated_indicators if ind in df.columns]
        missing_integrated = [ind for ind in integrated_indicators if ind not in df.columns]
        if available_integrated and not suppress_detailed_logs:
            logger.info(f"✅ {coin} {interval} 통합 분석 지표 사용 가능: {len(available_integrated)}개 ({', '.join(available_integrated)})")
        if missing_integrated:
            logger.debug(f"💡 {coin} {interval} 통합 분석 지표 일부 없음: {len(missing_integrated)}개 ({', '.join(missing_integrated)}) - 기본 지표로 전략 생성")

        

        # 1. 시장 상황 분석

        market_condition = classify_market_condition(df)

        if not suppress_detailed_logs:
            logger.info(f"📊 {coin} {interval} 시장 상황: {market_condition}")

        

        # 2. 캔들 데이터에서 실제 기술지표 값들 추출 (강화된 버전)

        if not df.empty and len(df) > 20:  # 충분한 데이터가 있는 경우

            # 🚀 실제 RSI, MACD 등의 평균값 및 분포 계산

            has_real_data = (

                len(df) > 0 and 

                'rsi' in df.columns and 'volume_ratio' in df.columns and

                not df['rsi'].isna().all() and not df['volume_ratio'].isna().all() and

                df['rsi'].notna().sum() > 10 and df['volume_ratio'].notna().sum() > 10

            )

            

            if has_real_data:

                # 실제 데이터에서 지표값 계산

                avg_rsi = df['rsi'].mean()

                rsi_std = df['rsi'].std()

                rsi_min_actual = df['rsi'].min()

                rsi_max_actual = df['rsi'].max()

                

                avg_volume_ratio = df['volume_ratio'].mean()

                volume_std = df['volume_ratio'].std()

                

                avg_atr = df['atr'].mean() if 'atr' in df.columns else 0.02

                atr_std = df['atr'].std() if 'atr' in df.columns else 0.01

                

                avg_mfi = df['mfi'].mean() if 'mfi' in df.columns else 50

                avg_adx = df['adx'].mean() if 'adx' in df.columns else 25

                avg_macd = df['macd'].mean() if 'macd' in df.columns else 0.0

                avg_volatility = df['volatility'].mean() if 'volatility' in df.columns else 0.02

                

                if not suppress_detailed_logs:
                    logger.info(f"📈 {coin} {interval} 실제 지표값:")

                    logger.info(f"  - RSI: 평균={avg_rsi:.1f}, 표준편차={rsi_std:.1f}, 범위=[{rsi_min_actual:.1f}, {rsi_max_actual:.1f}]")

                    logger.info(f"  - Volume: 평균={avg_volume_ratio:.2f}, 표준편차={volume_std:.2f}")

                    logger.info(f"  - ATR: 평균={avg_atr:.4f}, 표준편차={atr_std:.4f}")

                    logger.info(f"  - MFI: 평균={avg_mfi:.1f}, ADX: 평균={avg_adx:.1f}")

                    logger.info(f"  - MACD: 평균={avg_macd:.6f}, Volatility: 평균={avg_volatility:.4f}")

            else:

                # 실제 데이터가 없으면 기본값 사용

                logger.warning(f"⚠️ {coin} {interval}: 실제 데이터 없음, 기본값 사용")

                

                # 기본값 설정

                avg_rsi = 50.0

                rsi_std = 15.0

                rsi_min_actual = 20.0

                rsi_max_actual = 80.0

                avg_volume_ratio = 1.0

                volume_std = 0.5

                avg_atr = 0.02

                atr_std = 0.01

                avg_mfi = 50.0

                avg_adx = 25.0

                avg_macd = 0.0

                avg_volatility = 0.02

            

            # 🆕 동적 비율 계산 (시장 상황, 데이터 품질, 성능 기반)

            ai_ratio = calculate_dynamic_ai_ratio(market_condition, df, coin, interval)

            intelligent_count = int(num_strategies * ai_ratio)

            random_count = num_strategies - intelligent_count

            

            # 🔥 상세 로그 억제 모드가 아닐 때만 동적 비율 로그 출력
            if not suppress_detailed_logs:
                logger.info(f"🎯 {coin} {interval} 동적 비율: AI {intelligent_count}개 ({ai_ratio:.1%}), 랜덤 {random_count}개 ({1-ai_ratio:.1%})")

            

            # 🆕 방향성 있는 구간 분석 (전략 생성 전 개선)
            directional_periods = _analyze_directional_periods(df)
            if not suppress_detailed_logs:
                logger.info(f"📊 {coin} {interval} 방향성 구간 분석: 상승 {directional_periods['bullish_count']}개, 하락 {directional_periods['bearish_count']}개, 중립 {directional_periods['neutral_count']}개")
            
            # 🆕 시장 분석 기반 전략 생성 (Claude 제안) - 30% 할당
            market_adaptive_count = int(intelligent_count * 0.3)
            remaining_intelligent_count = intelligent_count - market_adaptive_count
            
            # 🚀 통합 분석 기반 전략 생성 (새로운 시스템)
            # 기존 AI 전략 대신 통합 분석 기반 전략 생성
            integrated_strategies_count = remaining_intelligent_count
            
            logger.info(f"📊 {coin} {interval} 전략 비율: 시장 분석 기반 {market_adaptive_count}개, 통합 분석 기반 {integrated_strategies_count}개")
            logger.info(f"🚀 {coin} {interval} 통합 분석 전략 생성 시작 (그룹 조합 + OR 조건 시스템)")

            # 🆕 시장 분석 기반 전략 생성 (Claude 제안)
            if market_adaptive_count > 0:
                try:
                    market_strategies = create_market_adaptive_strategies(coin, interval, df, market_adaptive_count, regime=regime)
                    for strategy in market_strategies:
                        if strategy is not None:
                            strategies.append(strategy)
                    logger.info(f"✅ {coin} {interval} 시장 분석 기반 전략 {len(market_strategies)}개 생성 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {coin} {interval} 시장 분석 기반 전략 생성 실패, 기본 전략으로 대체: {e}")
                    # 실패 시 기본 전략으로 대체
                    for i in range(market_adaptive_count):
                        strategy_pattern = select_ai_strategy_pattern(market_condition, i, market_adaptive_count)
                        strategy = create_enhanced_market_adaptive_strategy(coin, interval, market_condition, strategy_pattern, df, index=i, regime=regime)
                        if strategy is not None:
                            strategies.append(strategy)
            
            # 🚀 통합 분석 기반 전략 생성 (새로운 시스템)
            # 🆕 중복 체크를 위한 해시 세트 (처음부터 중복 방지)
            seen_hashes = set()
            
            for i in range(integrated_strategies_count):
                # 중복 없이 전략 생성 (최대 시도 횟수 제한)
                max_attempts = 100
                strategy = None
                for attempt in range(max_attempts):
                    # 🚀 통합 분석 기반 전략 생성
                    strategy = create_integrated_analysis_strategy(
                        coin, interval, df, 
                        index=i*1000+attempt, 
                        regime=regime
                    )
                    
                    # 🚀 None 체크
                    if strategy is None:
                        continue
                    
                    # 🆕 중복 체크
                    if ENHANCEMENTS_AVAILABLE:
                        strategy_hash = generate_strategy_hash(strategy)
                        if not strategy_hash:
                            if attempt == 0:
                                logger.debug(f"🔍 {coin} {interval} 전략 {i}: 해시 생성 실패, 재시도 중...")
                            strategy = None
                            continue
                        elif strategy_hash not in seen_hashes:
                            seen_hashes.add(strategy_hash)
                            break  # 고유한 전략 생성 성공
                        else:
                            if attempt == 0 or attempt % 20 == 0:
                                logger.debug(f"🔍 {coin} {interval} 전략 {i}: 중복 감지 (시도 {attempt+1}/{max_attempts})")
                            strategy = None
                    else:
                        # ENHANCEMENTS_AVAILABLE이 없으면 해시 체크 없이 바로 추가
                        break
                
                if strategy is None:
                    logger.warning(f"⚠️ {coin} {interval} 통합 분석 전략 {i} 생성 실패 (최대 시도 횟수 초과), 건너뛰기")
                    continue
                
                strategies.append(strategy)

            

            # 🆕 지능화된 랜덤 전략 생성 (가이드된 랜덤) - 방향성 고려
            for i in range(random_count):
                # 중복 없이 랜덤 전략 생성 (최대 시도 횟수 제한)
                max_attempts = 50
                strategy = None
                for attempt in range(max_attempts):
                    # 방향성에 따라 랜덤 전략도 조정
                    if directional_periods['bullish_count'] > directional_periods['bearish_count']:
                        # 상승 구간이 많으면 매수 특화
                        strategy = create_guided_random_strategy(coin, interval, df, "bullish", index=intelligent_count*1000 + i*1000 + attempt, prefer_direction="buy", regime=regime)
                    elif directional_periods['bearish_count'] > directional_periods['bullish_count']:
                        # 하락 구간이 많으면 매도 특화
                        strategy = create_guided_random_strategy(coin, interval, df, "bearish", index=intelligent_count*1000 + i*1000 + attempt, prefer_direction="sell", regime=regime)
                    else:
                        # 균형이면 일반
                        strategy = create_guided_random_strategy(coin, interval, df, market_condition, index=intelligent_count*1000 + i*1000 + attempt, regime=regime)
                    
                    # 🚀 None 체크
                    if strategy is None:
                        continue
                    
                    # 🆕 중복 체크 (이미 생성된 전략과 비교) - 완화된 버전
                    if ENHANCEMENTS_AVAILABLE:
                        try:
                            strategy_hash = generate_strategy_hash(strategy)
                            if strategy_hash and strategy_hash not in seen_hashes:
                                seen_hashes.add(strategy_hash)
                                break  # 고유한 전략 생성 성공
                            elif not strategy_hash:
                                # 해시 생성 실패 시에도 전략 허용 (다양성 확보)
                                if attempt == 0 or attempt % 10 == 0:
                                    logger.debug(f"🔍 {coin} {interval} 랜덤 전략 {i}: 해시 생성 실패, 전략 허용")
                                break
                            else:
                                # 중복 감지: 최대 10번까지만 재시도, 그 이후에는 전략 허용
                                if attempt < 10:
                                    # 처음 10번은 재시도
                                    if attempt < 3:
                                        logger.debug(f"🔍 {coin} {interval} 랜덤 전략 {i+1}: 중복 감지 (시도 {attempt+1}), 재시도")
                                    strategy = None  # 중복이므로 다시 생성
                                else:
                                    # 10번 이상 중복이면 전략 허용 (너무 엄격한 중복 체크 방지)
                                    if attempt == 10:
                                        logger.debug(f"🔍 {coin} {interval} 랜덤 전략 {i+1}: 10번 중복 후 전략 허용 (약간의 차이 허용)")
                                    break
                        except Exception as e:
                            # 해시 생성 중 예외 발생 시 전략 허용 (안전 장치)
                            if attempt == 0:
                                logger.debug(f"🔍 {coin} {interval} 랜덤 전략 {i}: 해시 생성 예외 ({e}), 전략 허용")
                            break
                    else:
                        # ENHANCEMENTS_AVAILABLE이 없으면 해시 체크 없이 바로 추가
                        break
                
                if strategy is None:
                    logger.warning(f"⚠️ {coin} {interval} 랜덤 전략 {i} 생성 실패 (최대 시도 횟수 초과), 건너뛰기")
                    continue

                # 🚀 가이드된 랜덤 전략도 실제 데이터 기반으로 조정

                if strategy.rsi_condition:

                    strategy.rsi_condition['min'] = max(15, rsi_min_actual + (rsi_max_actual - rsi_min_actual) * 0.1)

                    strategy.rsi_condition['max'] = min(85, rsi_max_actual - (rsi_max_actual - rsi_min_actual) * 0.1)

                

                if strategy.volume_condition:

                    volume_min_actual = df['volume_ratio'].min() if 'volume_ratio' in df.columns else 0.5

                    volume_max_actual = df['volume_ratio'].max() if 'volume_ratio' in df.columns else 2.0

                    strategy.volume_condition['min_ratio'] = max(0.2, volume_min_actual * 1.1)

                    strategy.volume_condition['max_ratio'] = min(4.0, volume_max_actual * 0.9)

                

                if strategy.atr_condition:

                    atr_min_actual = df['atr'].min() if 'atr' in df.columns else 0.005

                    atr_max_actual = df['atr'].max() if 'atr' in df.columns else 0.08

                    strategy.atr_condition['min'] = max(0.002, atr_min_actual * 1.2)

                    strategy.atr_condition['max'] = min(0.12, atr_max_actual * 0.8)

                

                strategies.append(strategy)

        

        else:

            # 데이터가 부족한 경우 기본 전략 생성

            logger.warning(f"⚠️ {coin} {interval}: 데이터 부족, 기본 전략 생성")

            for i in range(num_strategies):

                strategy = create_basic_strategy(coin, interval, index=i)

                strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 지능형 전략 생성 실패: {e}")

        return []



def create_coin_strategies_dynamic(coin: str, intervals: List[str], all_candle_data: Dict[Tuple[str, str], pd.DataFrame]) -> int:

    """🆕 코인별 동적 분할 전략 생성 함수 - 15일 기준 자동 분할"""

    try:

        strategies_created = 0

        

        for interval in intervals:

            try:

                # 🚀 동적 기간 분할 계산

                from simulation.replay import calculate_dynamic_periods

                periods = calculate_dynamic_periods(coin, interval, all_candle_data)

                

                if not periods['has_data']:

                    logger.warning(f"⚠️ {coin} {interval}: 데이터가 없어 전략 생성 건너뜀")

                    continue

                

                logger.info(f"🔄 {coin} {interval} 동적 분할 전략 생성 시작...")

                

                # 🚀 장기 전략 생성 (15일 이상인 경우만)

                if periods['has_long_term']:

                    logger.info(f"📈 {coin} {interval} 장기 전략 생성: {periods['long_term_days']:.1f}일")

                    long_term_strategies = create_intelligent_strategies_with_type(

                        coin, interval, 

                        config.STRATEGIES_PER_COMBINATION,  # 600개 그대로

                        all_candle_data.get((coin, interval)),

                        "long_term"

                    )

                    if long_term_strategies:

                        saved_count = _save_strategies_to_db_lazy(long_term_strategies)

                        strategies_created += saved_count

                        logger.info(f"✅ {coin} {interval} 장기 전략 생성 완료: {saved_count}개")

                

                # 🚀 단기 전략 생성

                if periods['has_short_term']:

                    if periods['has_long_term']:

                        # 전반/후반 분할 전략 생성

                        logger.info(f"📊 {coin} {interval} 단기 전반 전략 생성: {periods['short_term_front_days']:.1f}일")

                        short_front_strategies = create_intelligent_strategies_with_type(

                            coin, interval,

                            config.STRATEGIES_PER_COMBINATION,  # 600개 그대로

                            all_candle_data.get((coin, interval)),

                            "short_term_front"

                        )

                        if short_front_strategies:

                            saved_count = _save_strategies_to_db_lazy(short_front_strategies)

                            strategies_created += saved_count

                            logger.info(f"✅ {coin} {interval} 단기 전반 전략 생성 완료: {saved_count}개")

                        

                        logger.info(f"📊 {coin} {interval} 단기 후반 전략 생성: {periods['short_term_back_days']:.1f}일")

                        short_back_strategies = create_intelligent_strategies_with_type(

                            coin, interval,

                            config.STRATEGIES_PER_COMBINATION,  # 600개 그대로

                            all_candle_data.get((coin, interval)),

                            "short_term_back"

                        )

                        if short_back_strategies:

                            saved_count = _save_strategies_to_db_lazy(short_back_strategies)

                            strategies_created += saved_count

                            logger.info(f"✅ {coin} {interval} 단기 후반 전략 생성 완료: {saved_count}개")

                    else:

                        # 단기만 전략 생성

                        logger.info(f"📊 {coin} {interval} 단기만 전략 생성: {periods['short_term_only_days']:.1f}일")

                        short_only_strategies = create_intelligent_strategies_with_type(

                            coin, interval,

                            config.STRATEGIES_PER_COMBINATION,  # 600개 그대로

                            all_candle_data.get((coin, interval)),

                            "short_term_only"

                        )

                        if short_only_strategies:

                            saved_count = _save_strategies_to_db_lazy(short_only_strategies)

                            strategies_created += saved_count

                            logger.info(f"✅ {coin} {interval} 단기만 전략 생성 완료: {saved_count}개")

                

                logger.info(f"✅ {coin} {interval}: 총 {strategies_created}개 전략 생성 완료")

                

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 전략 생성 실패: {e}")

                continue

        

        return strategies_created

        

    except Exception as e:

        logger.error(f"❌ {coin} 동적 분할 전략 생성 실패: {e}")

        return 0



def _load_trained_strategies(coin: str, interval: str) -> List[Dict[str, Any]]:
    """
    학습 완료된 전략 로드 (증분 학습용)

    Returns:
        학습 완료된 전략 리스트 (training_history와 조인)
    """
    try:
        from rl_pipeline.db.connection_pool import get_strategy_db_pool

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # strategy_training_history와 조인하여 학습 완료된 전략만 로드
            query = """
                SELECT cs.*
                FROM coin_strategies cs
                INNER JOIN strategy_training_history sth ON cs.id = sth.strategy_id
                WHERE cs.coin = ? AND cs.interval = ?
                ORDER BY sth.trained_at DESC
            """

            cursor.execute(query, (coin, interval))
            rows = cursor.fetchall()

            if not rows:
                return []

            # 컬럼명 가져오기
            columns = [desc[0] for desc in cursor.description]

            # 딕셔너리 리스트로 변환
            strategies = []
            for row in rows:
                strategy_dict = dict(zip(columns, row))
                strategies.append(strategy_dict)

            logger.info(f"✅ {coin} {interval}: 학습 완료된 전략 {len(strategies)}개 로드")
            return strategies

    except Exception as e:
        logger.warning(f"⚠️ {coin} {interval}: 학습 완료 전략 로드 실패: {e}")
        return []


def create_coin_strategies(coin: str, intervals: List[str], all_candle_data: Dict[Tuple[str, str], pd.DataFrame]) -> int:

    try:

        strategies_created = 0



        for interval in intervals:

            try:

                # 🚀 캐시된 데이터 사용

                df = all_candle_data.get((coin, interval))



                if df is None or df.empty:

                    logger.warning(f"⚠️ {coin} {interval}: 캐시된 데이터가 없어 전략 생성 건너뜀")

                    continue

                # 🆕 증분 학습: 기존 학습 완료 전략 로드
                existing_trained_strategies = _load_trained_strategies(coin, interval)
                logger.info(f"📚 {coin} {interval}: 기존 학습 완료 전략 {len(existing_trained_strategies)}개")

                # 🆕 레짐 타겟팅: 타겟 레짐 결정
                target_regime = get_target_regime_for_generation(coin, interval)
                logger.info(f"🎯 {coin} {interval}: 타겟 레짐 = {target_regime}")

                # 🚀 기존 함수 활용: 지능형 전략 생성 (create_intelligent_strategies 내부에서 데이터 검증 로그 출력)

                # 🆕 개선된 전략 생성 (방향성 확보)
                strategies = []
                
                # 환경변수로 개선 모드 제어
                use_enhanced_generation = os.getenv('USE_ENHANCED_STRATEGY_GENERATION', 'false').lower() == 'true'
                
                if use_enhanced_generation and ENHANCEMENTS_AVAILABLE:
                    logger.info(f"🚀 {coin} {interval}: 개선된 전략 생성 모드 사용")
                    
                    # 🆕 중복 체크를 위한 해시 세트 (처음부터 중복 방지)
                    seen_hashes = set()
                    
                    # 🆕 비율 가져오기 (환경변수 또는 기본값) - os는 이미 상단에서 import됨
                    intelligent_ratio = float(os.getenv('INTELLIGENT_STRATEGY_RATIO', '0.5'))
                    grid_ratio = float(os.getenv('GRID_SEARCH_STRATEGY_RATIO', '0.2'))
                    direction_ratio = float(os.getenv('DIRECTION_SPECIALIZED_RATIO', '0.3'))
                    
                    # 1. 기존 지능형 전략 (비율 기반) - 중복 체크 포함
                    intelligent_count = int(config.STRATEGIES_PER_COMBINATION * intelligent_ratio)
                    intelligent_strategies = create_intelligent_strategies(coin, interval, intelligent_count, df, regime=target_regime)
                    # 중복 제거하며 추가
                    added_count = 0
                    for s in intelligent_strategies:
                        if ENHANCEMENTS_AVAILABLE:
                            s_hash = generate_strategy_hash(s)
                            if s_hash and s_hash not in seen_hashes:
                                strategies.append(s)
                                seen_hashes.add(s_hash)
                                added_count += 1
                        else:
                            strategies.append(s)
                            added_count += 1
                    logger.info(f"✅ 지능형 전략: {added_count}개 ({intelligent_ratio:.0%}, 중복 제외)")
                    
                    # 2. 그리드 서치 전략 (비율 기반) - 중복 체크 포함
                    grid_count = int(config.STRATEGIES_PER_COMBINATION * grid_ratio)
                    # 그리드 서치 전략은 처음부터 중복 없이 생성되도록 수정 필요하지만, 일단 필터링으로 처리
                    grid_strategies_raw = create_grid_search_strategies(coin, interval, df, grid_count * 2, seed=42)  # 여유 있게 생성
                    grid_strategies = []
                    for s in grid_strategies_raw:
                        if ENHANCEMENTS_AVAILABLE:
                            s_hash = generate_strategy_hash(s)
                            if s_hash and s_hash not in seen_hashes:
                                grid_strategies.append(s)
                                seen_hashes.add(s_hash)
                                if len(grid_strategies) >= grid_count:
                                    break
                        else:
                            grid_strategies.append(s)
                            if len(grid_strategies) >= grid_count:
                                break
                    strategies.extend(grid_strategies)
                    logger.info(f"✅ 그리드 서치 전략: {len(grid_strategies)}개 ({grid_ratio:.0%})")
                    
                    # 3. 방향성별 특화 전략 (비율 기반, 각 방향 동일 분배) - 중복 체크 포함
                    direction_count = int(config.STRATEGIES_PER_COMBINATION * direction_ratio / 3)  # 각 방향에 동일 분배
                    direction_strategies_raw = create_direction_specialized_strategies(coin, interval, df, direction_count * 2)  # 여유 있게 생성
                    direction_strategies = {'BUY': [], 'SELL': [], 'HOLD': []}
                    for direction in ['BUY', 'SELL', 'HOLD']:
                        for s in direction_strategies_raw[direction]:
                            if ENHANCEMENTS_AVAILABLE:
                                s_hash = generate_strategy_hash(s)
                                if s_hash and s_hash not in seen_hashes:
                                    direction_strategies[direction].append(s)
                                    seen_hashes.add(s_hash)
                                    if len(direction_strategies[direction]) >= direction_count:
                                        break
                            else:
                                direction_strategies[direction].append(s)
                                if len(direction_strategies[direction]) >= direction_count:
                                    break
                    strategies.extend(direction_strategies['BUY'])
                    strategies.extend(direction_strategies['SELL'])
                    strategies.extend(direction_strategies['HOLD'])
                    logger.info(f"✅ 방향성별 특화 전략: {sum(len(v) for v in direction_strategies.values())}개")
                    
                    # 🆕 목표 개수 맞추기: 부족하면 추가 생성 (중복 체크 포함)
                    target_count = config.STRATEGIES_PER_COMBINATION
                    if len(strategies) < target_count:
                        shortage = target_count - len(strategies)
                        logger.info(f"🔧 목표 개수 부족: {len(strategies)}/{target_count}개, {shortage}개 추가 생성")
                        # 🔥 추가 생성 시에는 여유 있게 생성하되, 상세 로그는 억제
                        additional_strategies = create_intelligent_strategies(coin, interval, shortage * 3, df, regime=target_regime, suppress_detailed_logs=True)  # 여유 있게 생성, 상세 로그 억제
                        # 기존 전략과 중복 제거하며 추가
                        added_additional = 0
                        for s in additional_strategies:
                            if ENHANCEMENTS_AVAILABLE:
                                s_hash = generate_strategy_hash(s)
                                if s_hash and s_hash not in seen_hashes:
                                    strategies.append(s)
                                    seen_hashes.add(s_hash)
                                    added_additional += 1
                                    if len(strategies) >= target_count:
                                        break
                            else:
                                strategies.append(s)
                                added_additional += 1
                                if len(strategies) >= target_count:
                                    break
                        logger.info(f"✅ 목표 개수 맞춤: {len(strategies)}/{target_count}개 (추가 생성: {added_additional}개, 중복 없음)")
                    else:
                        logger.info(f"✅ 목표 개수 달성: {len(strategies)}/{target_count}개 (중복 없음)")
                else:
                    # 기존 방식 사용
                    strategies = create_intelligent_strategies(coin, interval, config.STRATEGIES_PER_COMBINATION, df, regime=target_regime)

                    # 중복 제거 (개선 모듈이 있으면)
                    if ENHANCEMENTS_AVAILABLE:
                        strategies = filter_duplicate_strategies(strategies)

                    # 🆕 목표 개수 맞추기: 중복 제거 후 부족하면 추가 생성
                    target_count = config.STRATEGIES_PER_COMBINATION
                    if len(strategies) < target_count:
                        shortage = target_count - len(strategies)
                        logger.info(f"🔧 목표 개수 부족: {len(strategies)}/{target_count}개, {shortage}개 추가 생성")
                        # 🔥 추가 생성 시에는 여유 있게 생성하되, 상세 로그는 억제
                        additional_strategies = create_intelligent_strategies(coin, interval, shortage * 2, df, regime=target_regime, suppress_detailed_logs=True)  # 여유 있게 생성, 상세 로그 억제
                        additional_strategies = filter_duplicate_strategies(additional_strategies)
                        # 기존 전략과 중복 제거
                        existing_hashes = {generate_strategy_hash(s) for s in strategies}
                        added_additional = 0
                        for s in additional_strategies:
                            s_hash = generate_strategy_hash(s)
                            if s_hash and s_hash not in existing_hashes and len(strategies) < target_count:
                                strategies.append(s)
                                existing_hashes.add(s_hash)
                                added_additional += 1
                        logger.info(f"✅ 목표 개수 맞춤: {len(strategies)}/{target_count}개 (추가 생성: {added_additional}개)")

                if strategies:

                    # 🆕 증분 학습: 유사도 기반 전략 분류
                    if existing_trained_strategies:
                        logger.info(f"🔍 {coin} {interval}: 유사도 검사 시작 ({len(strategies)}개 신규 전략 vs {len(existing_trained_strategies)}개 기존 전략)")

                        classified = classify_new_strategies_batch(
                            strategies,
                            existing_trained_strategies,
                            duplicate_threshold=0.9995,  # 🔥 조정: 더 엄격한 중복 판정
                            copy_threshold=0.995,  # 🔥 조정: 매우 유사한 전략
                            finetune_threshold=0.95,  # 🔥 조정: 어느 정도 유사한 전략
                            use_smart=True
                        )

                        # 통계 로깅
                        logger.info(f"📊 {coin} {interval} 유사도 분류 결과:")
                        logger.info(f"  - 중복(duplicate): {len(classified['duplicate'])}개 (저장 건너뜀)")
                        logger.info(f"  - 정책 복사(copy): {len(classified['copy'])}개 (부모 정책 복사, 3ep)")
                        logger.info(f"  - 미세 조정(finetune): {len(classified['finetune'])}개 (부모 기반, 7-12ep)")
                        logger.info(f"  - 신규(novel): {len(classified['novel'])}개 (전체 학습, 20ep)")

                        # 중복 제거: duplicate는 저장하지 않음
                        strategies = (
                            classified['copy'] +
                            classified['finetune'] +
                            classified['novel']
                        )

                        logger.info(f"✅ {coin} {interval}: 중복 제거 후 {len(strategies)}개 전략 저장 예정")
                    else:
                        # 첫 실행 또는 기존 학습 전략 없음 - 모두 novel로 처리
                        logger.info(f"ℹ️ {coin} {interval}: 기존 학습 전략 없음, 모든 전략을 신규로 처리")
                        for s in strategies:
                            # dict 형식인지 확인
                            if isinstance(s, dict):
                                s['similarity_classification'] = 'novel'
                                s['similarity_score'] = 0.0
                                s['parent_strategy_id'] = None
                            else:
                                # Strategy 객체인 경우 params와 객체 속성 모두에 저장
                                if not hasattr(s, 'params') or not isinstance(s.params, dict):
                                    logger.warning(f"⚠️ 전략 {getattr(s, 'id', 'unknown')}: params가 dict가 아님, 건너뜀")
                                    continue
                                # params에 저장
                                s.params['similarity_classification'] = 'novel'
                                s.params['similarity_score'] = 0.0
                                s.params['parent_strategy_id'] = None
                                # 객체 속성으로도 저장 (serialize_strategy가 getattr로 추출하기 위해)
                                s.similarity_classification = 'novel'
                                s.similarity_score = 0.0
                                s.parent_strategy_id = None

                    # 전략 저장

                    logger.info(f"🔍 {coin} {interval}: {len(strategies)}개 전략 생성됨, 저장 시작...")

                    saved_count = _save_strategies_to_db_lazy(strategies)

                    logger.info(f"🔍 {coin} {interval}: 실제 저장된 전략 수: {saved_count}")

                    strategies_created += saved_count

                    logger.info(f"✅ {coin} {interval}: {saved_count}개 전략 생성 및 저장 완료")

                else:

                    logger.warning(f"⚠️ {coin} {interval}: 전략 생성 실패")

                    

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 전략 생성 실패: {e}")

                continue

        

        return strategies_created

        

    except Exception as e:

        logger.error(f"❌ {coin} 전략 생성 실패: {e}")

        return 0



def _analyze_directional_periods(df: pd.DataFrame) -> Dict[str, int]:
    """
    캔들 데이터에서 방향성이 있는 구간 분석
    
    Returns:
        {
            'bullish_count': 상승 구간 수,
            'bearish_count': 하락 구간 수,
            'neutral_count': 중립 구간 수
        }
    """
    try:
        if df.empty or len(df) < 20:
            return {'bullish_count': 0, 'bearish_count': 0, 'neutral_count': 1}
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        # 윈도우 크기 (최근 20개씩 분석)
        window = 20
        step = 10  # 10개씩 건너뛰며 분석
        
        for start_idx in range(0, len(df) - window, step):
            window_data = df.iloc[start_idx:start_idx + window]
            
            # RSI 분석
            avg_rsi = window_data['rsi'].mean() if 'rsi' in window_data.columns else 50.0
            
            # MACD 분석
            avg_macd = window_data['macd'].mean() if 'macd' in window_data.columns else 0.0
            if 'macd_signal' in window_data.columns:
                avg_macd_signal = window_data['macd_signal'].mean()
                macd_bullish = avg_macd > avg_macd_signal and avg_macd > 0.005
                macd_bearish = avg_macd < avg_macd_signal and avg_macd < -0.005
            else:
                macd_bullish = avg_macd > 0.005
                macd_bearish = avg_macd < -0.005
            
            # 가격 추세 분석
            if len(window_data) >= 5:
                price_change = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
                price_bullish = price_change > 0.01  # 1% 이상 상승
                price_bearish = price_change < -0.01  # 1% 이상 하락
            else:
                price_bullish = False
                price_bearish = False
            
            # 방향성 판단 (더 완화된 기준)
            # 상승: RSI < 45 (과매도 복귀 가능) 또는 (MACD 상승) 또는 (가격 상승 + MACD 상승)
            # 하락: RSI > 55 (과매수 하락 가능) 또는 (MACD 하락) 또는 (가격 하락 + MACD 하락)
            bullish_signals = 0
            bearish_signals = 0
            
            if avg_rsi < 45:  # 과매도 구간
                bullish_signals += 1
            if avg_rsi > 55:  # 과매수 구간
                bearish_signals += 1
            if macd_bullish:
                bullish_signals += 1
            if macd_bearish:
                bearish_signals += 1
            if price_bullish:
                bullish_signals += 1
            if price_bearish:
                bearish_signals += 1
            
            # 최소 2개 신호가 있으면 방향성 있음
            if bullish_signals >= 2 and bullish_signals > bearish_signals:
                bullish_count += 1
            elif bearish_signals >= 2 and bearish_signals > bullish_signals:
                bearish_count += 1
            else:
                neutral_count += 1
        
        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count
        }
        
    except Exception as e:
        logger.warning(f"⚠️ 방향성 구간 분석 실패: {e}")
        return {'bullish_count': 0, 'bearish_count': 0, 'neutral_count': 1}

def classify_market_condition(df: pd.DataFrame) -> str:

    """🚀 새로운 통합 레짐 시스템 사용"""

    try:

        # 새로운 레짐 시스템에서 레짐 정보 가져오기

        if 'regime_label' in df.columns and not df.empty:

            latest_regime = df['regime_label'].iloc[-1]

            if pd.notna(latest_regime):

                return latest_regime

        

        # 폴백: 기본값

        return "neutral"

            

    except Exception as e:

        logger.error(f"❌ 레짐 분류 실패: {e}")

        return "neutral"


# 🚀 통합 분석 기반 전략 생성 헬퍼 함수들
def determine_strategy_type_by_regime(regime_label: str, market_condition: Dict[str, Any] = None) -> str:
    """
    레짐 및 시장 상황 기반 전략 타입 결정
    
    Args:
        regime_label: 레짐 라벨
        market_condition: 시장 상황 정보 (RSI, Volume Ratio 등)
    
    Returns:
        전략 타입: 'aggressive', 'balanced', 'conservative'
    """
    import random
    
    # 레짐별 기본 비율 가져오기
    regime = regime_label.lower() if regime_label else 'neutral'
    if regime not in REGIME_STRATEGY_TYPE_RATIOS:
        regime = 'neutral'
    
    base_ratios = REGIME_STRATEGY_TYPE_RATIOS[regime].copy()
    
    # 시장 상황 기반 추가 조정
    if market_condition:
        rsi = market_condition.get('rsi_mean', 50.0)
        volume_ratio = market_condition.get('volume_ratio_mean', 1.0)
        
        # RSI > 70 (과매수): 보수적 전략 비율 증가
        if rsi > 70:
            base_ratios['conservative'] = min(0.7, base_ratios['conservative'] + 0.2)
            base_ratios['aggressive'] = max(0.1, base_ratios['aggressive'] - 0.1)
        
        # RSI < 30 (과매도): 공격적 전략 비율 증가
        elif rsi < 30:
            base_ratios['aggressive'] = min(0.7, base_ratios['aggressive'] + 0.2)
            base_ratios['conservative'] = max(0.1, base_ratios['conservative'] - 0.1)
        
        # Volume Ratio > 2.0 (거래량 급증): 공격적 전략 비율 증가
        if volume_ratio > 2.0:
            base_ratios['aggressive'] = min(0.6, base_ratios['aggressive'] + 0.15)
            base_ratios['conservative'] = max(0.1, base_ratios['conservative'] - 0.1)
        
        # 비율 정규화
        total = sum(base_ratios.values())
        if total > 0:
            base_ratios = {k: v / total for k, v in base_ratios.items()}
    
    # 랜덤 선택 (비율 기반)
    rand = random.random()
    cumulative = 0.0
    for strategy_type, ratio in base_ratios.items():
        cumulative += ratio
        if rand <= cumulative:
            return strategy_type
    
    return 'balanced'  # 기본값


def select_indicator_group_combination() -> Tuple[str, str]:
    """
    허용된 그룹 조합 중 랜덤 선택
    
    Returns:
        (그룹1, 그룹2) 튜플
    """
    import random
    return random.choice(ALLOWED_GROUP_COMBINATIONS)


def select_indicators_from_group(group: str, num_indicators: int = None, df: pd.DataFrame = None) -> List[str]:
    """
    그룹에서 지표 선택 (OR 조건용)
    
    Args:
        group: 그룹 ID ('A', 'B', 'C', 'D')
        num_indicators: 선택할 지표 수 (None이면 1~3개 랜덤)
        df: 데이터프레임 (지표 존재 여부 확인용)
    
    Returns:
        선택된 지표 리스트
    """
    import random
    
    if group not in INDICATOR_GROUPS:
        return []
    
    available_indicators = list(INDICATOR_GROUPS[group].keys())
    
    # 데이터프레임에서 존재하는 지표만 필터링
    if df is not None:
        available_indicators = [ind for ind in available_indicators if ind in df.columns]
    
    if not available_indicators:
        return []
    
    # 선택할 지표 수 결정
    if num_indicators is None:
        num_indicators = random.randint(1, min(3, len(available_indicators)))
    else:
        num_indicators = min(num_indicators, len(available_indicators))
    
    # 랜덤 선택
    return random.sample(available_indicators, num_indicators)


def create_indicator_condition(indicator: str, group: str, df: pd.DataFrame, strategy_type: str = 'balanced', sample_seed: int = None) -> Dict[str, Any]:
    """
    지표별 조건 생성
    
    Args:
        indicator: 지표 이름
        group: 그룹 ID
        df: 데이터프레임
        strategy_type: 전략 타입
        sample_seed: 샘플링 시드 (전략마다 다른 샘플 선택용)
    
    Returns:
        조건 딕셔너리
    """
    if group not in INDICATOR_GROUPS or indicator not in INDICATOR_GROUPS[group]:
        return None
    
    indicator_config = INDICATOR_GROUPS[group][indicator]
    condition_type = indicator_config['type']
    
    condition = {
        'indicator': indicator,
        'type': condition_type,
        'group': group
    }
    
    # 실제 데이터 기반 파라미터 계산
    if indicator in df.columns and not df[indicator].isna().all():
        # 🔥 레짐별/시점별 다양성을 고려한 데이터 샘플링
        # 전체 데이터가 아닌 다양한 시점/레짐의 데이터를 샘플링하여 전략 다양성 확보
        indicator_data = df[indicator].dropna()
        if len(indicator_data) > 0:
            # 🔥 레짐별 필터링 시도 (regime_label 컬럼이 있으면)
            sampled_data = indicator_data
            if 'regime_label' in df.columns and len(df) > 100:
                # 레짐별로 데이터 분포를 고려하여 샘플링
                regime_data = df[['regime_label', indicator]].dropna()
                if len(regime_data) > 0:
                    # 각 레짐별로 최소 20개씩 샘플링 (다양성 확보)
                    regime_samples = []
                    for regime in regime_data['regime_label'].unique():
                        regime_indicator = regime_data[regime_data['regime_label'] == regime][indicator]
                        if len(regime_indicator) > 0:
                            # 레짐별로 랜덤 샘플링 (최대 50개)
                            # 🔥 시드에 지표 이름과 레짐, 전략 인덱스를 조합하여 전략마다 다른 샘플 선택
                            seed_value = hash(f"{indicator}_{regime}_{sample_seed}") % 10000 if sample_seed is not None else hash(f"{indicator}_{regime}") % 1000
                            sample_size = min(50, len(regime_indicator))
                            regime_samples.append(regime_indicator.sample(n=sample_size, random_state=seed_value))
                    
                    if regime_samples:
                        # 모든 레짐 샘플 합치기
                        sampled_data = pd.concat(regime_samples)
                    else:
                        # 레짐 샘플링 실패 시 전체 데이터에서 랜덤 샘플링
                        sample_size = min(200, len(indicator_data))
                        seed_value = hash(f"{indicator}_{sample_seed}") % 10000 if sample_seed is not None else hash(indicator) % 1000
                        sampled_data = indicator_data.sample(n=sample_size, random_state=seed_value)
                else:
                    # 레짐 데이터 없으면 전체에서 랜덤 샘플링
                    sample_size = min(200, len(indicator_data))
                    seed_value = hash(f"{indicator}_{sample_seed}") % 10000 if sample_seed is not None else hash(indicator) % 1000
                    sampled_data = indicator_data.sample(n=sample_size, random_state=seed_value)
            elif len(indicator_data) > 200:
                # 레짐 정보 없으면 전체에서 랜덤 샘플링 (다양성 확보)
                sample_size = min(200, len(indicator_data))
                seed_value = hash(f"{indicator}_{sample_seed}") % 10000 if sample_seed is not None else hash(indicator) % 1000
                sampled_data = indicator_data.sample(n=sample_size, random_state=seed_value)
            
            # 🔥 categorical 타입은 mean/std 계산 불가 (문자열 값)
            if condition_type == 'categorical':
                # 범주형 조건
                condition['values'] = indicator_config['values']
                # 가장 많이 나타나는 값 선택
                value_counts = sampled_data.value_counts()
                if len(value_counts) > 0:
                    condition['preferred'] = value_counts.index[0]
            else:
                # 🔥 numeric 타입만 mean/std 계산
                try:
                    # numeric으로 변환 시도
                    numeric_data = pd.to_numeric(sampled_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        mean_val = numeric_data.mean()
                        std_val = numeric_data.std()
                        
                        if condition_type == 'range':
                            # 범위 조건
                            if strategy_type == 'aggressive':
                                # 공격적: 더 넓은 범위
                                condition['min'] = max(indicator_config['default_min'], mean_val - std_val * 2)
                                condition['max'] = min(indicator_config['default_max'], mean_val + std_val * 2)
                            elif strategy_type == 'conservative':
                                # 보수적: 더 좁은 범위
                                condition['min'] = max(indicator_config['default_min'], mean_val - std_val * 0.5)
                                condition['max'] = min(indicator_config['default_max'], mean_val + std_val * 0.5)
                            else:
                                # 균형: 기본 범위
                                condition['min'] = max(indicator_config['default_min'], mean_val - std_val)
                                condition['max'] = min(indicator_config['default_max'], mean_val + std_val)
                        
                        elif condition_type == 'threshold':
                            # 임계값 조건
                            if 'default_min' in indicator_config:
                                condition['min'] = max(indicator_config['default_min'], mean_val - std_val * 0.5)
                            if 'default_buy' in indicator_config:
                                condition['buy'] = mean_val - std_val * 0.5
                            if 'default_sell' in indicator_config:
                                condition['sell'] = mean_val + std_val * 0.5
                    else:
                        # numeric 변환 실패 시 기본값 사용
                        raise ValueError("Numeric conversion failed")
                except (ValueError, TypeError) as e:
                    # numeric 변환 실패 시 기본값 사용
                    logger.debug(f"⚠️ {indicator} numeric 변환 실패, 기본값 사용: {e}")
                    if condition_type == 'range':
                        condition['min'] = indicator_config['default_min']
                        condition['max'] = indicator_config['default_max']
                    elif condition_type == 'threshold':
                        if 'default_min' in indicator_config:
                            condition['min'] = indicator_config['default_min']
                        if 'default_buy' in indicator_config:
                            condition['buy'] = indicator_config['default_buy']
                        if 'default_sell' in indicator_config:
                            condition['sell'] = indicator_config['default_sell']
        else:
            # 데이터가 없으면 기본값 사용
            if condition_type == 'range':
                condition['min'] = indicator_config['default_min']
                condition['max'] = indicator_config['default_max']
            elif condition_type == 'threshold':
                if 'default_min' in indicator_config:
                    condition['min'] = indicator_config['default_min']
                if 'default_buy' in indicator_config:
                    condition['buy'] = indicator_config['default_buy']
                if 'default_sell' in indicator_config:
                    condition['sell'] = indicator_config['default_sell']
            elif condition_type == 'categorical':
                condition['values'] = indicator_config['values']
    else:
        # 지표가 없으면 기본값 사용
        if condition_type == 'range':
            condition['min'] = indicator_config['default_min']
            condition['max'] = indicator_config['default_max']
        elif condition_type == 'threshold':
            if 'default_min' in indicator_config:
                condition['min'] = indicator_config['default_min']
            if 'default_buy' in indicator_config:
                condition['buy'] = indicator_config['default_buy']
            if 'default_sell' in indicator_config:
                condition['sell'] = indicator_config['default_sell']
        elif condition_type == 'categorical':
            condition['values'] = indicator_config['values']
    
    return condition


def create_integrated_analysis_strategy(
    coin: str,
    interval: str,
    df: pd.DataFrame,
    index: int = None,
    regime: str = "ranging"
) -> Optional[Strategy]:
    """
    🚀 통합 분석 기반 전략 생성 (그룹 조합 + OR 조건)
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        df: 캔들 데이터프레임 (모든 통합 분석 지표 포함)
        index: 전략 인덱스 (시드용)
        regime: 타겟 레짐
    
    Returns:
        생성된 Strategy 객체 또는 None
    """
    try:
        import random
        import numpy as np
        import uuid
        from rl_pipeline.core.types import Strategy
        
        # 시드 설정
        if index is not None:
            random.seed(index)
            np.random.seed(index)
        
        # 1. 레짐 확인
        regime_label = classify_market_condition(df)
        if pd.isna(regime_label) or not regime_label:
            regime_label = 'neutral'
        
        # 2. 시장 상황 분석
        market_analysis = analyze_market(coin, interval, df)
        
        # 3. 전략 타입 결정 (레짐 + 시장 상황 기반)
        strategy_type = determine_strategy_type_by_regime(regime_label, market_analysis)
        
        # 4. 그룹 조합 선택
        group1, group2 = select_indicator_group_combination()
        
        # 5. 각 그룹에서 지표 선택 (OR 조건용, 1~3개)
        group1_indicators = select_indicators_from_group(group1, num_indicators=None, df=df)
        group2_indicators = select_indicators_from_group(group2, num_indicators=None, df=df)
        
        if not group1_indicators or not group2_indicators:
            logger.warning(f"⚠️ {coin} {interval}: 그룹에서 지표 선택 실패")
            return None
        
        # 6. 각 지표의 조건 생성 (전략마다 다른 데이터 샘플 사용)
        group1_conditions = []
        for indicator in group1_indicators:
            # 🔥 전략 인덱스와 지표 이름을 조합하여 다양한 샘플 선택
            condition = create_indicator_condition(indicator, group1, df, strategy_type, sample_seed=index)
            if condition:
                group1_conditions.append(condition)
        
        group2_conditions = []
        for indicator in group2_indicators:
            # 🔥 전략 인덱스와 지표 이름을 조합하여 다양한 샘플 선택
            condition = create_indicator_condition(indicator, group2, df, strategy_type, sample_seed=index)
            if condition:
                group2_conditions.append(condition)
        
        if not group1_conditions or not group2_conditions:
            logger.warning(f"⚠️ {coin} {interval}: 조건 생성 실패")
            return None
        
        # 7. 전략 파라미터 설정 (기존 지표들도 포함)
        strategy_params = {}
        
        # RSI 파라미터 (그룹 A에 있으면 조건에서, 없으면 기본값)
        if 'rsi' in [c['indicator'] for c in group1_conditions + group2_conditions]:
            rsi_cond = next((c for c in group1_conditions + group2_conditions if c['indicator'] == 'rsi'), None)
            if rsi_cond:
                strategy_params['rsi_min'] = rsi_cond.get('min', 30.0)
                strategy_params['rsi_max'] = rsi_cond.get('max', 70.0)
        else:
            # 기본 RSI 값 (데이터 기반)
            if 'rsi' in df.columns:
                avg_rsi = df['rsi'].mean()
                rsi_std = df['rsi'].std()
                strategy_params['rsi_min'] = max(10, avg_rsi - rsi_std)
                strategy_params['rsi_max'] = min(90, avg_rsi + rsi_std)
            else:
                strategy_params['rsi_min'] = 30.0
                strategy_params['rsi_max'] = 70.0
        
        # Volume Ratio 파라미터
        if 'volume_ratio' in [c['indicator'] for c in group1_conditions + group2_conditions]:
            vol_cond = next((c for c in group1_conditions + group2_conditions if c['indicator'] == 'volume_ratio'), None)
            if vol_cond:
                strategy_params['volume_ratio_min'] = vol_cond.get('min', 1.0)
                strategy_params['volume_ratio_max'] = vol_cond.get('max', 2.0)
        else:
            if 'volume_ratio' in df.columns:
                avg_vol = df['volume_ratio'].mean()
                vol_std = df['volume_ratio'].std()
                strategy_params['volume_ratio_min'] = max(0.5, avg_vol - vol_std)
                strategy_params['volume_ratio_max'] = min(5.0, avg_vol + vol_std)
            else:
                strategy_params['volume_ratio_min'] = 1.0
                strategy_params['volume_ratio_max'] = 2.0
        
        # MACD 파라미터
        if 'macd' in [c['indicator'] for c in group1_conditions + group2_conditions]:
            macd_cond = next((c for c in group1_conditions + group2_conditions if c['indicator'] == 'macd'), None)
            if macd_cond:
                strategy_params['macd_buy_threshold'] = macd_cond.get('buy', 0.0)
                strategy_params['macd_sell_threshold'] = macd_cond.get('sell', 0.0)
        else:
            if 'macd' in df.columns:
                avg_macd = df['macd'].mean()
                macd_std = df['macd'].std()
                strategy_params['macd_buy_threshold'] = avg_macd - macd_std
                strategy_params['macd_sell_threshold'] = avg_macd + macd_std
            else:
                strategy_params['macd_buy_threshold'] = 0.0
                strategy_params['macd_sell_threshold'] = 0.0
        
        # MFI, ATR, ADX 등 기타 파라미터
        if 'mfi' in df.columns:
            avg_mfi = df['mfi'].mean()
            mfi_std = df['mfi'].std()
            strategy_params['mfi_min'] = max(10, avg_mfi - mfi_std)
            strategy_params['mfi_max'] = min(90, avg_mfi + mfi_std)
        else:
            strategy_params['mfi_min'] = 20.0
            strategy_params['mfi_max'] = 80.0
        
        if 'atr' in df.columns:
            avg_atr = df['atr'].mean()
            atr_std = df['atr'].std()
            strategy_params['atr_min'] = max(0.005, avg_atr - atr_std)
            strategy_params['atr_max'] = min(0.1, avg_atr + atr_std)
        else:
            strategy_params['atr_min'] = 0.01
            strategy_params['atr_max'] = 0.05
        
        if 'adx' in df.columns:
            avg_adx = df['adx'].mean()
            strategy_params['adx_min'] = max(15, avg_adx - 5)
        else:
            strategy_params['adx_min'] = 15.0
        
        # 손절/익절 설정
        strategy_params['stop_loss_pct'] = random.uniform(0.10, 0.20)
        strategy_params['take_profit_pct'] = random.uniform(1.40, 1.60)
        
        # 8. 메타데이터 생성
        metadata = {
            'indicator_groups': [group1, group2],
            'group_a_indicators': group1_indicators,
            'group_b_indicators': group2_indicators,
            'strategy_type': strategy_type,
            'min_conditions': STRATEGY_TYPE_MIN_CONDITIONS.get(strategy_type, 2),
            'regime_filter': regime_label,
            'regime_confidence_threshold': 0.4,
            'condition_logic': 'OR_AND',  # 그룹 내 OR, 그룹 간 AND
            'market_condition': market_analysis.get('trend', 'neutral') if market_analysis else 'neutral',
            'created_regime': regime_label,
            'conditions': {
                'group1': group1_conditions,
                'group2': group2_conditions
            }
        }
        
        # 9. Strategy 객체 생성
        strategy_id = f"{coin}_{interval}_{strategy_type}_{uuid.uuid4().hex[:8]}"
        
        strategy = Strategy(
            id=strategy_id,
            params=strategy_params,
            version="1.0",
            coin=coin,
            interval=interval,
            created_at=datetime.now(),
            strategy_type=strategy_type,
            regime=regime,
            rsi_min=strategy_params.get('rsi_min', 30.0),
            rsi_max=strategy_params.get('rsi_max', 70.0),
            volume_ratio_min=strategy_params.get('volume_ratio_min', 1.0),
            volume_ratio_max=strategy_params.get('volume_ratio_max', 2.0),
            macd_buy_threshold=strategy_params.get('macd_buy_threshold', 0.0),
            macd_sell_threshold=strategy_params.get('macd_sell_threshold', 0.0),
            mfi_min=strategy_params.get('mfi_min', 20.0),
            mfi_max=strategy_params.get('mfi_max', 80.0),
            atr_min=strategy_params.get('atr_min', 0.01),
            atr_max=strategy_params.get('atr_max', 0.05),
            adx_min=strategy_params.get('adx_min', 15.0),
            stop_loss_pct=strategy_params.get('stop_loss_pct', 0.15),
            take_profit_pct=strategy_params.get('take_profit_pct', 1.50),
            metadata=metadata  # 🚀 메타데이터 필드에 저장
        )
        
        # 메타데이터를 params에도 저장 (하위 호환성)
        strategy.params['metadata'] = metadata
        
        logger.info(f"✅ 통합 분석 전략 생성: {coin}-{interval}, 타입={strategy_type}, 레짐={regime_label}, "
                   f"그룹=({group1},{group2}), 그룹1지표={','.join(group1_indicators)}, 그룹2지표={','.join(group2_indicators)}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ 통합 분석 전략 생성 실패: {coin}-{interval} - {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_market_condition_by_interval(df: pd.DataFrame, interval: str) -> str:

    """🚀 새로운 통합 레짐 시스템 사용"""

    try:

        # 새로운 레짐 시스템에서 레짐 정보 가져오기

        if 'regime_label' in df.columns and not df.empty:

            latest_regime = df['regime_label'].iloc[-1]

            if pd.notna(latest_regime):

                return latest_regime

        

        # 폴백: 기본값

        return "neutral"

            

    except Exception as e:

        logger.error(f"❌ 레짐 분류 실패: {e}")

        return "neutral"



def calculate_market_condition_confidence(df: pd.DataFrame, interval: str) -> float:

    """시장 상황 신뢰도 계산"""

    try:

        if df.empty or len(df) < 20:

            return 0.5

        

        recent_df = df.tail(20)

        

        # RSI 일관성

        rsi_std = recent_df['rsi'].std() if 'rsi' in recent_df.columns else 15

        rsi_consistency = max(0, 1 - (rsi_std / 20))  # 표준편차가 낮을수록 높은 신뢰도

        

        # Volume 일관성

        volume_std = recent_df['volume_ratio'].std() if 'volume_ratio' in recent_df.columns else 1.0

        volume_consistency = max(0, 1 - (volume_std / 2.0))

        

        # MACD 신호 강도

        macd_strength = 0

        if 'macd' in recent_df.columns and 'macd_signal' in recent_df.columns:

            macd_diff = abs(recent_df['macd'] - recent_df['macd_signal'])

            macd_strength = min(1.0, macd_diff.mean() * 100)  # MACD 차이가 클수록 높은 신뢰도

        

        # 인터벌별 가중치

        interval_weights = {

            "1d": {"rsi": 0.6, "volume": 0.2, "macd": 0.2},

            "15m": {"rsi": 0.3, "volume": 0.4, "macd": 0.3},

            "30m": {"rsi": 0.3, "volume": 0.3, "macd": 0.4},

            "240m": {"rsi": 0.25, "volume": 0.25, "macd": 0.25, "adx": 0.25}

        }

        

        weights = interval_weights.get(interval, interval_weights["15m"])

        

        # ADX 강도 (240분봉만)

        adx_strength = 0

        if interval == "240m" and 'adx' in recent_df.columns:

            adx_strength = min(1.0, recent_df['adx'].mean() / 50)  # ADX 50을 최대값으로 정규화

        

        # 가중 평균 신뢰도 계산

        confidence = (

            weights.get('rsi', 0.3) * rsi_consistency +

            weights.get('volume', 0.3) * volume_consistency +

            weights.get('macd', 0.3) * macd_strength +

            weights.get('adx', 0) * adx_strength

        )

        

        return min(1.0, max(0.0, confidence))

        

    except Exception as e:

        logger.error(f"❌ {interval} 시장 상황 신뢰도 계산 실패: {e}")

        return 0.5



def calculate_dynamic_ai_ratio(market_condition: str, df: pd.DataFrame, coin: str, interval: str) -> float:

    """동적 AI 비율 계산"""

    try:

        # 기본 비율

        base_ratio = 0.6

        

        # 시장 상황별 조정

        if market_condition in ["overbought", "oversold"]:

            base_ratio = 0.8  # 극단적 상황에서는 AI 전략 비중 증가

        elif market_condition in ["bullish", "bearish"]:

            base_ratio = 0.7  # 추세 시장에서는 AI 전략 비중 증가

        elif market_condition == "low_volume":

            base_ratio = 0.4  # 저조한 시장에서는 랜덤 전략 비중 증가

        

        # 데이터 품질 기반 조정

        if not df.empty and len(df) > 50:

            # 충분한 데이터가 있으면 AI 비중 증가

            base_ratio = min(0.9, base_ratio + 0.1)

        

        return base_ratio

        

    except Exception as e:

        logger.error(f"❌ 동적 AI 비율 계산 실패: {e}")

        return 0.6



def select_ai_strategy_pattern(market_condition: str, index: int, total_count: int) -> str:

    """AI 전략 패턴 선택"""

    try:

        patterns = {

            "overbought": ["mean_reversion", "momentum_breakout", "volume_spike"],

            "oversold": ["momentum_reversal", "volume_confirmation", "trend_follow"],

            "bullish": ["trend_follow", "momentum_breakout", "volume_spike"],

            "bearish": ["mean_reversion", "momentum_reversal", "volume_confirmation"],

            "low_volume": ["range_trading", "mean_reversion", "volume_spike"],

            "neutral": ["range_trading", "mean_reversion", "trend_follow"]

        }

        

        available_patterns = patterns.get(market_condition, ["trend_follow", "mean_reversion"])

        return available_patterns[index % len(available_patterns)]

        

    except Exception as e:

        logger.error(f"❌ AI 전략 패턴 선택 실패: {e}")

        return "trend_follow"



def _calculate_macd_buy_threshold(df: pd.DataFrame, market_condition: str, pattern: str) -> float:

    """MACD 매수 임계값 계산"""

    try:

        if df.empty or 'macd' not in df.columns:

            return 0.0

        

        # 최근 MACD 값들의 통계 계산

        recent_macd = df['macd'].tail(20)

        macd_mean = recent_macd.mean()

        macd_std = recent_macd.std()

        

        # 시장 상황별 기본 임계값

        base_thresholds = {

            'bullish': 0.02,

            'bearish': -0.01,

            'neutral': 0.0,

            'volatile': 0.01

        }

        

        base_threshold = base_thresholds.get(market_condition, 0.0)

        

        # 패턴별 조정

        pattern_adjustments = {

            'momentum': 0.01,

            'reversal': -0.01,

            'trend': 0.005,

            'range': 0.0

        }

        

        pattern_adjustment = pattern_adjustments.get(pattern, 0.0)

        

        # 실제 MACD 값 기반 조정

        macd_adjustment = macd_mean * 0.1  # MACD 평균의 10% 반영

        

        final_threshold = base_threshold + pattern_adjustment + macd_adjustment

        

        # 안전한 범위로 제한

        return max(-0.1, min(0.1, final_threshold))

        

    except Exception as e:

        logger.error(f"❌ MACD 매수 임계값 계산 실패: {e}")

        return 0.0



def _calculate_macd_sell_threshold(df: pd.DataFrame, market_condition: str, pattern: str) -> float:

    """MACD 매도 임계값 계산"""

    try:

        if df.empty or 'macd' not in df.columns:

            return 0.0

        

        # 최근 MACD 값들의 통계 계산

        recent_macd = df['macd'].tail(20)

        macd_mean = recent_macd.mean()

        macd_std = recent_macd.std()

        

        # 시장 상황별 기본 임계값

        base_thresholds = {

            'bullish': -0.01,

            'bearish': 0.02,

            'neutral': 0.0,

            'volatile': -0.01

        }

        

        base_threshold = base_thresholds.get(market_condition, 0.0)

        

        # 패턴별 조정

        pattern_adjustments = {

            'momentum': -0.01,

            'reversal': 0.01,

            'trend': -0.005,

            'range': 0.0

        }

        

        pattern_adjustment = pattern_adjustments.get(pattern, 0.0)

        

        # 실제 MACD 값 기반 조정

        macd_adjustment = macd_mean * -0.1  # MACD 평균의 -10% 반영

        

        final_threshold = base_threshold + pattern_adjustment + macd_adjustment

        

        # 안전한 범위로 제한

        return max(-0.1, min(0.1, final_threshold))

        

    except Exception as e:

        logger.error(f"❌ MACD 매도 임계값 계산 실패: {e}")

        return 0.0



def create_enhanced_market_adaptive_strategy(
    coin: str,
    interval: str,
    market_condition: str,
    pattern: str,
    df: pd.DataFrame,
    index: int = None,
    force_buy_direction: bool = False,
    force_sell_direction: bool = False,
    regime: str = "ranging"
) -> Strategy:

    """시장 적응형 전략 생성 - 실제 데이터 기반 + 방향성 강제"""
    try:

        from rl_pipeline.core.types import Strategy

        

        # 실제 데이터 기반 파라미터 계산

        if not df.empty and len(df) > 20:

            # 실제 지표값 계산 (모든 지표 활용!)

            avg_rsi = df['rsi'].mean()

            rsi_std = df['rsi'].std()

            avg_volume_ratio = df['volume_ratio'].mean()

            volume_std = df['volume_ratio'].std()

            avg_atr = df['atr'].mean()

            atr_std = df['atr'].std()

            

            # MFI 계산 (사용 가능한 경우)

            avg_mfi = df['mfi'].mean() if 'mfi' in df.columns else 50.0

            mfi_std = df['mfi'].std() if 'mfi' in df.columns else 15.0

            

            # ADX 계산 (사용 가능한 경우)

            avg_adx = df['adx'].mean() if 'adx' in df.columns else 25.0

            adx_std = df['adx'].std() if 'adx' in df.columns else 10.0

            

            # MACD 계산 (사용 가능한 경우)

            avg_macd = df['macd'].mean() if 'macd' in df.columns else 0.0

            macd_std = df['macd'].std() if 'macd' in df.columns else 0.01

            

            # 🆕 방향성 강제 옵션 처리 (더 공격적으로 설정)
            if force_buy_direction:
                # 매수 특화: 낮은 RSI에서 매수, 높은 거래량으로 확인
                # RSI 범위를 명확히 낮게 설정 (과매도 구간 중심)
                rsi_min = max(10, min(30, avg_rsi - rsi_std * 1.5))  # 10-30 범위
                rsi_max = min(50, max(35, avg_rsi - rsi_std * 0.3))  # 35-50 범위로 제한
                volume_min = max(1.0, avg_volume_ratio * 0.8)  # 거래량 요구 완화
                volume_max = min(5.0, avg_volume_ratio + volume_std * 2)
                mfi_min = max(10, avg_mfi - mfi_std * 2)  # 더 낮은 MFI (과매도)
                mfi_max = min(70, avg_mfi + mfi_std)  # 높은 MFI 구간 제외
                adx_min = max(15, avg_adx - adx_std * 0.5)
                # MACD 매수 임계값을 더 낮게 설정 (약한 상승 신호도 포착)
                macd_buy_value = avg_macd - macd_std * 0.5  # 더 낮은 임계값
                macd_sell_value = avg_macd + macd_std * 3  # 매도는 매우 느슨하게
            elif force_sell_direction:
                # 매도 특화: 높은 RSI에서 매도, 하락 추세 확인
                # RSI 범위를 명확히 높게 설정 (과매수 구간 중심)
                rsi_min = max(50, min(60, avg_rsi + rsi_std * 0.3))  # 50-60 범위
                rsi_max = min(90, max(70, avg_rsi + rsi_std * 1.5))  # 70-90 범위
                volume_min = max(0.8, avg_volume_ratio * 0.9)
                volume_max = min(4.0, avg_volume_ratio + volume_std * 1.5)
                mfi_min = max(50, avg_mfi - mfi_std)  # 낮은 MFI 구간 제외
                mfi_max = min(90, avg_mfi + mfi_std * 2)  # 더 높은 MFI (과매수)
                adx_min = max(15, avg_adx - adx_std * 0.5)
                # MACD 매도 임계값을 더 높게 설정 (약한 하락 신호도 포착)
                macd_buy_value = avg_macd - macd_std * 3  # 매수는 매우 느슨하게
                macd_sell_value = avg_macd + macd_std * 0.5  # 더 높은 임계값
            else:
                # 일반 파라미터 (기존 로직)
                pass
            
            # 🔧 index를 시드로 사용하여 다양성 확보
            if index is not None:
                import random
                import numpy as np
                random.seed(index)  # index를 시드로 사용하여 재현 가능한 다양성
                np.random.seed(index)
            
            # 패턴별 실제 데이터 기반 파라미터 설정 (모든 지표 활용!)
            # 방향성 강제가 아닐 때만 패턴별 설정 적용

            if not (force_buy_direction or force_sell_direction):
                # 방향성 강제가 아닐 때만 패턴별 기본 설정 적용
                if pattern == "mean_reversion":
                    # 🔧 랜덤 오프셋 추가로 다양성 확보
                    rsi_offset = random.uniform(-rsi_std * 0.5, rsi_std * 0.5) if index is not None else 0
                    rsi_min = max(10, avg_rsi - rsi_std * 2 + rsi_offset)
                    rsi_max = min(90, avg_rsi + rsi_std * 2 + rsi_offset)
                    volume_offset = random.uniform(-volume_std * 0.3, volume_std * 0.3) if index is not None else 0
                    volume_min = max(0.5, avg_volume_ratio - volume_std + volume_offset)
                    volume_max = min(3.0, avg_volume_ratio + volume_std * 1.5 + volume_offset)
                    mfi_min = max(10, avg_mfi - mfi_std * 1.5)
                    mfi_max = min(90, avg_mfi + mfi_std * 1.5)
                    adx_min = max(15, avg_adx - adx_std * 0.5)
                    macd_offset = random.uniform(-macd_std * 0.5, macd_std * 0.5) if index is not None else 0
                    macd_buy_value = avg_macd - macd_std * 1.5 + macd_offset
                    macd_sell_value = avg_macd + macd_std * 1.5 + macd_offset
                elif pattern == "momentum_breakout":
                    rsi_offset = random.uniform(-rsi_std * 0.3, rsi_std * 0.3) if index is not None else 0
                    rsi_min = max(20, avg_rsi - rsi_std + rsi_offset)
                    rsi_max = min(95, avg_rsi + rsi_std * 2 + rsi_offset)
                    volume_offset = random.uniform(-volume_std * 0.2, volume_std * 0.4) if index is not None else 0
                    volume_min = max(1.0, avg_volume_ratio + volume_std * 0.5 + volume_offset)
                    volume_max = min(4.0, avg_volume_ratio + volume_std * 2 + volume_offset)
                    mfi_min = max(20, avg_mfi - mfi_std)
                    mfi_max = min(80, avg_mfi + mfi_std * 2)
                    adx_min = max(25, avg_adx + adx_std * 0.5)
                    macd_offset = random.uniform(-macd_std * 0.3, macd_std * 0.3) if index is not None else 0
                    macd_buy_value = avg_macd - macd_std * 0.5 + macd_offset
                    macd_sell_value = avg_macd + macd_std * 2 + macd_offset
                elif pattern == "trend_follow":
                    rsi_offset = random.uniform(-rsi_std * 0.4, rsi_std * 0.4) if index is not None else 0
                    rsi_min = max(15, avg_rsi - rsi_std * 1.5 + rsi_offset)
                    rsi_max = min(85, avg_rsi + rsi_std * 1.5 + rsi_offset)
                    volume_offset = random.uniform(-volume_std * 0.3, volume_std * 0.3) if index is not None else 0
                    volume_min = max(0.8, avg_volume_ratio - volume_std * 0.5 + volume_offset)
                    volume_max = min(3.0, avg_volume_ratio + volume_std * 1.5 + volume_offset)
                    mfi_min = max(15, avg_mfi - mfi_std * 1.2)
                    mfi_max = min(85, avg_mfi + mfi_std * 1.5)
                    adx_min = max(20, avg_adx)
                    macd_offset = random.uniform(-macd_std * 0.4, macd_std * 0.4) if index is not None else 0
                    macd_buy_value = avg_macd - macd_std + macd_offset
                    macd_sell_value = avg_macd + macd_std * 1.5 + macd_offset
                elif pattern == "volume_spike":
                    rsi_offset = random.uniform(-rsi_std * 0.3, rsi_std * 0.3) if index is not None else 0
                    rsi_min = max(10, avg_rsi - rsi_std * 1.5 + rsi_offset)
                    rsi_max = min(90, avg_rsi + rsi_std * 1.5 + rsi_offset)
                    volume_offset = random.uniform(-volume_std * 0.2, volume_std * 0.5) if index is not None else 0
                    volume_min = max(1.2, avg_volume_ratio + volume_std + volume_offset)
                    volume_max = min(5.0, avg_volume_ratio + volume_std * 3 + volume_offset)
                    mfi_min = max(10, avg_mfi - mfi_std * 2)
                    mfi_max = min(90, avg_mfi + mfi_std * 2)
                    adx_min = max(20, avg_adx - adx_std)
                    macd_offset = random.uniform(-macd_std * 0.5, macd_std * 0.5) if index is not None else 0
                    macd_buy_value = avg_macd - macd_std * 2 + macd_offset
                    macd_sell_value = avg_macd + macd_std * 2 + macd_offset
                else:  # range_trading
                    rsi_offset = random.uniform(-rsi_std * 0.2, rsi_std * 0.2) if index is not None else 0
                    rsi_min = max(20, avg_rsi - rsi_std + rsi_offset)
                    rsi_max = min(80, avg_rsi + rsi_std + rsi_offset)
                    volume_min = max(0.7, avg_volume_ratio - volume_std * 0.5)
                    volume_max = min(2.5, avg_volume_ratio + volume_std)
                    mfi_min = max(20, avg_mfi - mfi_std)
                    mfi_max = min(80, avg_mfi + mfi_std)
                    adx_min = max(15, avg_adx - adx_std)
                    macd_buy_value = avg_macd - macd_std
                    macd_sell_value = avg_macd + macd_std
            else:
                # 방향성 강제 시 패턴별 추가 조정만
                if pattern == "momentum_breakout":
                    if force_buy_direction:
                        volume_min = max(volume_min, 1.3)  # 모멘텀은 더 높은 거래량
                    elif force_sell_direction:
                        volume_min = max(volume_min, 1.1)

            

            # ATR 기반 손절/익절 설정 (다양하게!)
            # 🔧 index 기반 랜덤 배율로 다양성 확보
            atr_min = max(0.005, avg_atr - atr_std)
            atr_max = min(0.1, avg_atr + atr_std * 2)
            
            # stop_loss와 take_profit을 직접 설정 (15%, 50% 목표)
            # ATR 기반이 아닌 고정 범위 사용
            if index is not None:
                stop_loss_pct = random.uniform(0.10, 0.20)  # 10% ~ 20% (평균 15%)
                take_profit_pct = random.uniform(1.40, 1.60)  # 140% ~ 160% (40% ~ 60% 수익)
            else:
                stop_loss_pct = 0.15  # 15% 고정
                take_profit_pct = 1.50  # 150% (50% 수익)

            # 패턴별 미세 조정 (큰 차이는 없음)
            if pattern == "mean_reversion":
                # 평균 회귀: 약간 타이트하게
                if index is not None:
                    stop_loss_pct = random.uniform(0.10, 0.15)  # 10% ~ 15%
                    take_profit_pct = random.uniform(1.35, 1.50)  # 135% ~ 150% (35% ~ 50% 수익)
                else:
                    stop_loss_pct = 0.12
                    take_profit_pct = 1.40  # 140% (40% 수익)
            elif pattern == "momentum_breakout":
                # 모멘텀: 약간 여유있게
                if index is not None:
                    stop_loss_pct = random.uniform(0.15, 0.25)  # 15% ~ 25%
                    take_profit_pct = random.uniform(1.50, 1.70)  # 150% ~ 170% (50% ~ 70% 수익)
                else:
                    stop_loss_pct = 0.20
                    take_profit_pct = 1.60  # 160% (60% 수익)
            elif pattern == "trend_follow":
                # 추세 추종: 중간
                if index is not None:
                    stop_loss_pct = random.uniform(0.12, 0.18)  # 12% ~ 18%
                    take_profit_pct = random.uniform(1.45, 1.65)  # 145% ~ 165% (45% ~ 65% 수익)
                else:
                    stop_loss_pct = 0.15
                    take_profit_pct = 1.55  # 155% (55% 수익)

            

        else:

            # 데이터 부족 시 기본값 사용 (하지만 다양하게!)

            logger.warning(f"⚠️ {coin} {interval}: 데이터 부족, 기본 파라미터 사용")

            # 다양성을 위해 랜덤 적용
            # 🔧 index를 시드로 사용하여 다양성 확보
            import random
            import numpy as np
            
            if index is not None:
                random.seed(index)
                np.random.seed(index)

            offset = random.uniform(-5, 5)

            rsi_min = max(10, 20 + offset)  # 20으로 완화 (기존 30)

            rsi_max = min(90, 80 + offset * 1.5)  # 80으로 완화 (기존 70)

            volume_min = random.uniform(0.8, 1.2)

            volume_max = random.uniform(2.0, 3.0)

            stop_loss_pct = random.uniform(0.10, 0.20)  # 10% ~ 20% (평균 15%)

            take_profit_pct = random.uniform(1.40, 1.60)  # 140% ~ 160% (40% ~ 60% 수익)

        

        # 시장 상황별 미세 조정

        if market_condition == "overbought":

            rsi_min = max(5, rsi_min - 5)

            rsi_max = min(95, rsi_max - 5)

        elif market_condition == "oversold":

            rsi_min = max(5, rsi_min + 5)

            rsi_max = min(95, rsi_max + 5)

        elif market_condition == "bullish":

            rsi_min = max(10, rsi_min - 3)

            volume_min = max(0.8, volume_min * 1.1)

        elif market_condition == "bearish":

            rsi_max = min(90, rsi_max + 3)

            volume_min = max(0.8, volume_min * 1.1)

        

        # MFI, ADX가 정의되지 않은 경우 기본값 설정

        if 'mfi_min' not in locals():

            mfi_min = 20

            mfi_max = 80

        if 'adx_min' not in locals():

            adx_min = 20

        if 'macd_buy_value' not in locals():

            macd_buy_value = _calculate_macd_buy_threshold(df, market_condition, pattern)

        if 'macd_sell_value' not in locals():

            macd_sell_threshold = _calculate_macd_sell_threshold(df, market_condition, pattern)

        

        # 고유 ID 생성 (index 포함)

        unique_id_suffix = f"{int(time.time())}_{hash(pattern)}" if index is None else f"{int(time.time())}_{index}"

        strategy = Strategy(

            id=f"{coin}_{interval}_ai_{pattern}_{unique_id_suffix}",

            params={

                'mfi_min': mfi_min,

                'mfi_max': mfi_max,

                'adx_min': adx_min,

                'atr_min': atr_min if 'atr_min' in locals() else 0.01,

                'atr_max': atr_max if 'atr_max' in locals() else 0.05

            },

            version="v2.0",

            coin=coin,

            interval=interval,

            created_at=datetime.now(),

            strategy_type=f"ai_{pattern}",

            rsi_min=rsi_min,

            rsi_max=rsi_max,

            volume_ratio_min=volume_min,

            volume_ratio_max=volume_max,

            # 🆕 핵심 지표 min/max 필드 직접 할당
            mfi_min=mfi_min if 'mfi_min' in locals() else 20.0,
            mfi_max=mfi_max if 'mfi_max' in locals() else 80.0,
            atr_min=atr_min if 'atr_min' in locals() else 0.01,
            atr_max=atr_max if 'atr_max' in locals() else 0.05,
            adx_min=adx_min if 'adx_min' in locals() else 15.0,

            stop_loss_pct=stop_loss_pct,

            take_profit_pct=take_profit_pct,

            macd_buy_threshold=macd_buy_value,

            macd_sell_threshold=macd_sell_value,

            ma_period=20,

            bb_period=20,

            bb_std=2.0,

            rsi_condition={'min': rsi_min, 'max': rsi_max},

            volume_condition={'min': volume_min, 'max': volume_max},

            atr_condition={'min': atr_min if 'atr_min' in locals() else 0.01, 'max': atr_max if 'atr_max' in locals() else 0.05},

            # 🆕 레짐 정보
            regime=regime

        )



        return strategy

        

    except Exception as e:

        logger.error(f"❌ 시장 적응형 전략 생성 실패: {e}", exc_info=True)

        # 기본 전략 반환 (None 대신 기본 전략 반환)

        from rl_pipeline.core.types import Strategy

        return Strategy(

            id=f"{coin}_{interval}_ai_default_{int(time.time())}",

            params={},

            version="v2.0",

            coin=coin,

            interval=interval,

            created_at=datetime.now(),

            strategy_type="ai_default",

            rsi_min=30,

            rsi_max=70,

            volume_ratio_min=1.0,

            volume_ratio_max=2.0,

            # 🆕 핵심 지표 min/max 필드 직접 할당 (기본값)
            mfi_min=20.0,
            mfi_max=80.0,
            atr_min=0.01,
            atr_max=0.05,
            adx_min=15.0,

            stop_loss_pct=0.15,

            take_profit_pct=1.50,

            macd_buy_threshold=0.01,

            macd_sell_threshold=-0.01,

            ma_period=20,

            bb_period=20,

            bb_std=2.0,

            rsi_condition={'min': 30, 'max': 70},

            volume_condition={'min': 1.0, 'max': 2.0},

            atr_condition={'min': 0.01, 'max': 0.05}

        )



def create_guided_random_strategy(
    coin: str,
    interval: str,
    df: pd.DataFrame,
    market_condition: str,
    index: int = None,
    prefer_direction: str = None,
    regime: str = "ranging"
) -> Strategy:

    """가이드된 랜덤 전략 생성 - 실제 캔들 데이터 기반 + 방향성 선호 옵션"""
    try:

        from rl_pipeline.core.types import Strategy

        import random
        import numpy as np
        import time as time_module

        # 🔧 index를 시드로 사용하여 다양성 확보 (매번 다른 전략 생성)
        if index is not None:
            random.seed(index)
            np.random.seed(index)
        else:
            # index가 None이면 현재 시간 기반 시드 사용
            random.seed(int(time_module.time() * 1000) % 1000000)
            np.random.seed(int(time_module.time() * 1000) % 1000000)

        # 🔥 실제 캔들 데이터에서 지표 계산
        if not df.empty and len(df) > 20:
            # 실제 지표값 계산
            avg_rsi = df['rsi'].mean() if 'rsi' in df.columns and not df['rsi'].isna().all() else 50.0
            rsi_std = df['rsi'].std() if 'rsi' in df.columns and not df['rsi'].isna().all() else 15.0
            
            avg_volume_ratio = df['volume_ratio'].mean() if 'volume_ratio' in df.columns and not df['volume_ratio'].isna().all() else 1.0
            volume_std = df['volume_ratio'].std() if 'volume_ratio' in df.columns and not df['volume_ratio'].isna().all() else 0.5
            
            avg_atr = df['atr'].mean() if 'atr' in df.columns and not df['atr'].isna().all() else 0.02
            atr_std = df['atr'].std() if 'atr' in df.columns and not df['atr'].isna().all() else 0.01
            
            avg_mfi = df['mfi'].mean() if 'mfi' in df.columns and not df['mfi'].isna().all() else 50.0
            mfi_std = df['mfi'].std() if 'mfi' in df.columns and not df['mfi'].isna().all() else 15.0
            
            avg_adx = df['adx'].mean() if 'adx' in df.columns and not df['adx'].isna().all() else 25.0
            adx_std = df['adx'].std() if 'adx' in df.columns and not df['adx'].isna().all() else 10.0
        else:
            # 데이터 부족 시 기본값
            avg_rsi, rsi_std = 50.0, 15.0
            avg_volume_ratio, volume_std = 1.0, 0.5
            avg_atr, atr_std = 0.02, 0.01
            avg_mfi, mfi_std = 50.0, 15.0
            avg_adx, adx_std = 25.0, 10.0

        # 🆕 방향성 선호 옵션 처리 (실제 데이터 기반 + 랜덤 오프셋으로 다양성 확보)
        rsi_offset = random.uniform(-rsi_std * 0.3, rsi_std * 0.3)  # 랜덤 오프셋 추가
        volume_offset = random.uniform(-volume_std * 0.2, volume_std * 0.2)
        mfi_offset = random.uniform(-mfi_std * 0.2, mfi_std * 0.2)
        adx_offset = random.uniform(-adx_std * 0.1, adx_std * 0.1)
        atr_offset = random.uniform(-atr_std * 0.2, atr_std * 0.2)
        
        if prefer_direction == "buy":
            # 매수 선호: 낮은 RSI 구간 중심 (실제 데이터 기반 + 랜덤 오프셋)
            rsi_min = max(10, avg_rsi - rsi_std * 1.5 + rsi_offset)
            rsi_max = min(70, avg_rsi + rsi_std * 0.3 + rsi_offset)
            volume_min = max(1.0, avg_volume_ratio * 0.8 + volume_offset)
            volume_max = min(5.0, avg_volume_ratio + volume_std * 2 + volume_offset)
            mfi_min = max(10, avg_mfi - mfi_std * 2 + mfi_offset)
            mfi_max = min(70, avg_mfi + mfi_std + mfi_offset)
            adx_min = max(15, avg_adx - adx_std * 0.5 + adx_offset)
        elif prefer_direction == "sell":
            # 매도 선호: 높은 RSI 구간 중심 (실제 데이터 기반 + 랜덤 오프셋)
            rsi_min = max(50, avg_rsi + rsi_std * 0.3 + rsi_offset)
            rsi_max = min(90, avg_rsi + rsi_std * 1.5 + rsi_offset)
            volume_min = max(0.8, avg_volume_ratio * 0.9 + volume_offset)
            volume_max = min(4.0, avg_volume_ratio + volume_std * 1.5 + volume_offset)
            mfi_min = max(50, avg_mfi - mfi_std + mfi_offset)
            mfi_max = min(90, avg_mfi + mfi_std * 2 + mfi_offset)
            adx_min = max(15, avg_adx - adx_std * 0.5 + adx_offset)
        else:
            # 일반: 넓은 범위 (실제 데이터 기반 + 랜덤 오프셋)
            if market_condition == "overbought":
                rsi_min = max(20, avg_rsi - rsi_std * 0.5 + rsi_offset)
                rsi_max = min(80, avg_rsi + rsi_std * 1.2 + rsi_offset)
            elif market_condition == "oversold":
                rsi_min = max(10, avg_rsi - rsi_std * 1.5 + rsi_offset)
                rsi_max = min(75, avg_rsi + rsi_std * 0.5 + rsi_offset)
            else:
                rsi_min = max(10, avg_rsi - rsi_std * 1.5 + rsi_offset)
                rsi_max = min(90, avg_rsi + rsi_std * 1.5 + rsi_offset)
            
            volume_min = max(0.3, avg_volume_ratio - volume_std + volume_offset)
            volume_max = min(5.0, avg_volume_ratio + volume_std * 2 + volume_offset)
            mfi_min = max(10, avg_mfi - mfi_std * 1.5 + mfi_offset)
            mfi_max = min(90, avg_mfi + mfi_std * 1.5 + mfi_offset)
            adx_min = max(15, avg_adx - adx_std * 0.5 + adx_offset)
        
        # ATR 기반 손절/익절 (실제 데이터 + 랜덤 오프셋)
        atr_min = max(0.005, avg_atr - atr_std + atr_offset)
        atr_max = min(0.1, avg_atr + atr_std * 2 + atr_offset)
        # 손절/익절을 고정 범위로 설정 (15%, 50% 목표)
        stop_loss_pct = random.uniform(0.10, 0.20)  # 10% ~ 20% (평균 15%)
        take_profit_pct = random.uniform(1.40, 1.60)  # 140% ~ 160% (40% ~ 60% 수익)

        ma_period = random.choice([10, 15, 20, 25, 30, 40, 50])  # 더 다양한 기간

        bb_period = random.choice([10, 15, 20, 25, 30])  # 더 다양한 기간

        bb_std = random.uniform(1.0, 3.5)  # 더 넓은 표준편차

        

        # 고유 ID 생성 (index 포함)

        unique_id_suffix = f"{int(time.time())}_{hash(str(df.shape))}" if index is None else f"{int(time.time())}_{index}"

        strategy = Strategy(

            id=f"{coin}_{interval}_guided_random_{unique_id_suffix}",

            params={

                'mfi_min': mfi_min,

                'mfi_max': mfi_max,

                'adx_min': adx_min,

                'atr_min': atr_min if 'atr_min' in locals() else 0.01,

                'atr_max': atr_max if 'atr_max' in locals() else 0.05

            },

            version="v2.0",

            coin=coin,

            interval=interval,

            created_at=datetime.now(),

            strategy_type="guided_random",

            regime=regime,

            rsi_min=rsi_min,

            rsi_max=rsi_max,

            volume_ratio_min=volume_min,

            volume_ratio_max=volume_max,

            # 🆕 핵심 지표 min/max 필드 직접 할당 (실제 데이터 기반)
            mfi_min=mfi_min if 'mfi_min' in locals() else 20.0,
            mfi_max=mfi_max if 'mfi_max' in locals() else 80.0,
            atr_min=atr_min if 'atr_min' in locals() else 0.01,
            atr_max=atr_max if 'atr_max' in locals() else 0.05,
            adx_min=adx_min if 'adx_min' in locals() else 15.0,

            stop_loss_pct=stop_loss_pct,  # ✅ ATR 기반

            take_profit_pct=take_profit_pct,  # ✅ ATR 기반

            macd_buy_threshold=_calculate_macd_buy_threshold(df, market_condition, "random"),

            macd_sell_threshold=_calculate_macd_sell_threshold(df, market_condition, "random"),

            ma_period=ma_period,

            bb_period=bb_period,

            bb_std=bb_std,

            rsi_condition={'min': rsi_min, 'max': rsi_max},

            volume_condition={'min': volume_min, 'max': volume_max},

            atr_condition={'min': atr_min if 'atr_min' in locals() else 0.01, 'max': atr_max if 'atr_max' in locals() else 0.05}

        )

        

        return strategy

        

    except Exception as e:

        logger.error(f"❌ 가이드된 랜덤 전략 생성 실패: {e}")

        # 기본 전략 반환

        from rl_pipeline.core.types import Strategy

        return Strategy(

            id=f"{coin}_{interval}_random_default_{int(time.time())}",

            params={},

            version="v2.0",

            coin=coin,

            interval=interval,

            created_at=datetime.now(),

            strategy_type="random_default",

            rsi_min=30,

            rsi_max=70,

            volume_ratio_min=1.0,

            volume_ratio_max=2.0,

            ma_period=20,

            bb_period=20,

            bb_std=2.0,

            rsi_condition={'min': 30, 'max': 70},

            volume_condition={'min': 1.0, 'max': 2.0},

            atr_condition={'min': 0.01, 'max': 0.05}

        )



def create_basic_strategy(coin: str, interval: str, index: Optional[int] = None, regime: str = "ranging") -> Strategy:
    """
    기본 전략 생성 - 실제 데이터 기반

    Args:
        coin: 코인 심볼
        interval: 시간 간격
        index: 전략 인덱스 (선택적)
        regime: 타겟 레짐 (ranging, trending, volatile, 기본값: ranging)

    Returns:
        생성된 전략 객체
    """

    try:

        from rl_pipeline.core.types import Strategy

        

        # 실제 캔들 데이터 로드하여 기본값 계산

        try:

            df = load_candles(coin, interval, days=30)

            if not df.empty and len(df) > 20:

                # 실제 데이터 기반 기본값 계산

                avg_rsi = df['rsi'].mean()

                rsi_std = df['rsi'].std()

                avg_volume = df['volume_ratio'].mean()

                volume_std = df['volume_ratio'].std()

                

                # 안전한 범위로 제한

                rsi_min = max(20, min(40, avg_rsi - rsi_std))

                rsi_max = min(80, max(60, avg_rsi + rsi_std))

                volume_min = max(0.8, min(1.2, avg_volume - volume_std * 0.5))

                volume_max = min(2.5, max(1.8, avg_volume + volume_std * 0.5))

                

                logger.info(f"📊 {coin} {interval} 실제 데이터 기반 기본 전략: RSI({rsi_min:.1f}-{rsi_max:.1f}), Volume({volume_min:.2f}-{volume_max:.2f})")

            else:

                # 데이터 부족 시 보수적 기본값

                rsi_min, rsi_max = 30, 70

                volume_min, volume_max = 1.0, 2.0

                logger.warning(f"⚠️ {coin} {interval}: 데이터 부족, 보수적 기본값 사용")

        except Exception as e:

            logger.warning(f"⚠️ {coin} {interval}: 캔들 데이터 로드 실패, 기본값 사용: {e}")

            rsi_min, rsi_max = 30, 70

            volume_min, volume_max = 1.0, 2.0

        

        # 고유 ID 생성 (index 포함)

        unique_id_suffix = f"{int(time.time())}_{hash(str((coin, interval)))}" if index is None else f"{int(time.time())}_{index}"

        strategy = Strategy(

            id=f"{coin}_{interval}_basic_{unique_id_suffix}",

            params={},

            version="v2.0",

            coin=coin,

            interval=interval,

            created_at=datetime.now(),

            strategy_type="basic",

            rsi_min=rsi_min,

            rsi_max=rsi_max,

            volume_ratio_min=volume_min,

            volume_ratio_max=volume_max,

            macd_buy_threshold=_calculate_macd_buy_threshold(df, "neutral", "basic"),

            macd_sell_threshold=_calculate_macd_sell_threshold(df, "neutral", "basic"),

            ma_period=20,

            bb_period=20,

            bb_std=2.0,

            rsi_condition={'min': rsi_min, 'max': rsi_max},

            volume_condition={'min': volume_min, 'max': volume_max},

            atr_condition={'min': 0.01, 'max': 0.05},

            # 🆕 레짐 정보
            regime=regime

        )



        return strategy

        

    except Exception as e:

        logger.error(f"❌ 기본 전략 생성 실패: {e}")

        return None



def create_global_strategies(all_coin_data: Dict[str, Dict[str, pd.DataFrame]], 

                           global_analysis_results: Optional[Dict[str, Any]] = None) -> int:

    """

    글로벌 전략 생성 함수 - 모든 코인 데이터를 종합 분석

    

    Args:

        all_coin_data: 모든 코인의 캔들 데이터 {coin: {interval: DataFrame}}

        global_analysis_results: 글로벌 분석 결과

        

    Returns:

        생성된 전략 수

    """

    try:

        logger.info("🌍 글로벌 전략 생성 시작 (전체 코인 종합 분석)")

        

        # 🌍 전체 코인 데이터 분석

        if not all_coin_data:

            logger.warning("⚠️ 모든 코인 데이터 없음, 기본 글로벌 전략만 생성")

            return _create_basic_global_strategies()

        

        # 전체 코인 목록

        all_coins = list(all_coin_data.keys())

        logger.info(f"📊 글로벌 전략 대상 코인: {all_coins} ({len(all_coins)}개)")

        

        # 1. 코인 간 상관관계 분석

        correlation_params = _analyze_correlation_across_coins(all_coin_data)

        

        # 2. 전체 시장 트렌드 분석

        market_trend_params = _analyze_global_market_trend(all_coin_data)

        

        # 3. 전체 시장 레짐 분석

        regime_params = _analyze_global_regime(all_coin_data)

        

        # 글로벌 전략 생성

        global_strategies = []

        

        # 1. 시장 전체 트렌드 기반 전략

        market_trend_strategy = {

            'id': f"global_market_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

            'coin': 'GLOBAL',

            'interval': '240m',

            'strategy_type': 'market_trend',

            'params': market_trend_params,

            'name': 'Market Trend Strategy',

            'description': f'전체 시장 트렌드 기반 글로벌 전략 (코인 {len(all_coins)}개 분석)',

            'created_at': datetime.now().isoformat(),

            'updated_at': datetime.now().isoformat()

        }

        global_strategies.append(market_trend_strategy)

        

        # 2. 코인 간 상관관계 기반 전략

        correlation_strategy = {

            'id': f"global_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

            'coin': 'GLOBAL',

            'interval': '60m',

            'strategy_type': 'correlation',

            'params': correlation_params,

            'name': 'Correlation Strategy',

            'description': f'코인 간 상관관계 기반 글로벌 전략 (코인 {len(all_coins)}개 분석)',

            'created_at': datetime.now().isoformat(),

            'updated_at': datetime.now().isoformat()

        }

        global_strategies.append(correlation_strategy)

        

        # 3. 레짐 기반 글로벌 전략

        regime_strategy = {

            'id': f"global_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

            'coin': 'GLOBAL',

            'interval': '120m',

            'strategy_type': 'regime_based',

            'params': regime_params,

            'name': 'Regime Based Strategy',

            'description': f'시장 레짐 기반 글로벌 전략 (코인 {len(all_coins)}개 분석)',

            'created_at': datetime.now().isoformat(),

            'updated_at': datetime.now().isoformat()

        }

        global_strategies.append(regime_strategy)

        

        # 글로벌 전략을 데이터베이스에 저장

        if global_strategies:

            try:
                # 🔥 수정: global_strategies 테이블에 직접 저장
                import hashlib
                import json
                import sqlite3
                
                with sqlite3.connect(config.STRATEGIES_DB) as conn:
                    cursor = conn.cursor()
                    
                    # global_strategies 테이블 생성 (존재하지 않으면)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS global_strategies (
                            id TEXT PRIMARY KEY,
                            coin TEXT NOT NULL,
                            interval TEXT NOT NULL,
                            strategy_type TEXT NOT NULL,
                            params TEXT NOT NULL,
                            name TEXT,
                            description TEXT,
                            dna_hash TEXT,
                            source_type TEXT DEFAULT 'synthesized',
                            profit REAL DEFAULT 0.0,
                            profit_factor REAL DEFAULT 0.0,
                            win_rate REAL DEFAULT 0.5,
                            trades_count INTEGER DEFAULT 0,
                            quality_grade TEXT DEFAULT 'A',
                            market_condition TEXT DEFAULT 'neutral',
                            volatility_group TEXT DEFAULT 'MEDIUM',
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            meta TEXT
                        )
                    """)
                    
                    # 인덱스 생성
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_global_strategies_interval
                        ON global_strategies(interval)
                    """)
                    
                    conn.commit()
                    
                    saved_count = 0
                    for strategy in global_strategies:
                        try:
                            # dna_hash 생성
                            params_str = json.dumps(strategy.get('params', {}), sort_keys=True)
                            dna_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
                            
                            # quality_grade 추출
                            quality_grade = strategy.get('quality_grade', 'A')
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO global_strategies
                                (id, coin, interval, strategy_type, params, name, description,
                                 dna_hash, source_type, profit, profit_factor, win_rate, trades_count,
                                 quality_grade, market_condition, created_at, updated_at, meta)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                strategy.get('id'),
                                strategy.get('coin'),
                                strategy.get('interval'),
                                strategy.get('strategy_type'),
                                json.dumps(strategy.get('params', {})),
                                strategy.get('name'),
                                strategy.get('description'),
                                dna_hash,
                                'synthesized',
                                0.0,  # profit (초기값)
                                0.0,  # profit_factor (초기값)
                                0.5,  # win_rate (초기값)
                                0,    # trades_count (초기값)
                                quality_grade,
                                strategy.get('market_condition', 'neutral'),
                                strategy.get('created_at', datetime.now().isoformat()),
                                strategy.get('updated_at', datetime.now().isoformat()),
                                json.dumps(strategy.get('meta', {}))
                            ))
                            saved_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ 전략 저장 실패: {strategy.get('id')} - {e}")
                    
                    conn.commit()
                
                logger.info(f"✅ 글로벌 전략 생성 완료: {saved_count}개 저장 (global_strategies 테이블)")

                return saved_count

            except Exception as e:

                logger.error(f"❌ 전략 DB 저장 실패: {e}")

                logger.info(f"✅ 글로벌 전략 생성 완료: {len(global_strategies)}개 (저장 실패)")

                return len(global_strategies)

        else:

            logger.warning("⚠️ 생성된 글로벌 전략이 없습니다")

            return 0

            

    except Exception as e:

        logger.error(f"❌ 글로벌 전략 생성 실패: {e}")

        return 0



def create_global_strategies_from_results(all_coin_strategies: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> int:

    """

    글로벌 전략 생성 함수 - 구역 기반 최고 성능 전략 선정 방식 (🆕 구역 기반 시스템)



    Args:

        all_coin_strategies: 모든 코인의 self-play 결과 {coin: {interval: [strategy_list]}}



    Returns:

        생성된 전략 수

    """

    try:

        logger.info("🌍 글로벌 전략 생성 시작 (🆕 구역 기반 시스템)")



        if not all_coin_strategies:

            logger.warning("⚠️ self-play 결과 없음, 기본 글로벌 전략만 생성")

            return _create_basic_global_strategies()



        # 🆕 구역 기반 글로벌 전략 생성
        from rl_pipeline.strategy.zone_based_global_creator import (
            create_zone_based_global_strategies,
            save_global_strategies_to_db
        )

        logger.info("📊 구역 기반 글로벌 전략 생성 (regime × RSI × market × volatility)")

        # 구역 기반 글로벌 전략 생성
        global_strategies = create_zone_based_global_strategies(all_coin_strategies)

        if not global_strategies:
            logger.warning("⚠️ 구역 기반 글로벌 전략 생성 실패, 기본 글로벌 전략 생성")
            return _create_basic_global_strategies()

        logger.info(f"✅ 구역 기반 글로벌 전략 생성 완료: {len(global_strategies)}개")

        # 💾 DB 저장 (zone_based_global_creator의 save 함수 사용)
        saved_count = save_global_strategies_to_db(global_strategies)

        if saved_count > 0:
            logger.info(f"✅ 글로벌 전략 DB 저장 완료: {saved_count}개")
            return saved_count
        else:
            logger.warning("⚠️ 글로벌 전략 DB 저장 실패")
            return 0
            
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 생성 실패: {e}")
        return 0


def _create_basic_global_strategies() -> int:

    """기본 글로벌 전략 생성 (폴백)"""

    try:

        global_strategies = []

        

        for i, (stype, interval, params) in enumerate([

            ('market_trend', '240m', {'trend_threshold': 0.02, 'volume_threshold': 1.5, 'correlation_threshold': 0.7, 'risk_level': 'medium'}),

            ('correlation', '60m', {'correlation_window': 24, 'correlation_threshold': 0.8, 'diversification_factor': 0.3, 'rebalance_frequency': 4}),

            ('regime_based', '120m', {'regime_detection_window': 48, 'regime_confidence_threshold': 0.6, 'transition_sensitivity': 0.4, 'regime_weight_factor': 0.8})

        ]):

            global_strategies.append({

                'id': f"global_{stype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

                'coin': 'GLOBAL',

                'interval': interval,

                'strategy_type': stype,

                'params': params,

                'name': f'{stype} Strategy',

                'description': f'기본 {stype} 글로벌 전략 (전체 데이터 없음)',

                'created_at': datetime.now().isoformat(),

                'updated_at': datetime.now().isoformat()

            })

        

        from rl_pipeline.strategy.manager import StrategyManager

        manager = StrategyManager()

        saved_count = manager.save_strategies_to_db_dict(global_strategies)

        logger.info(f"✅ 기본 글로벌 전략 생성 완료: {saved_count}개")

        return saved_count

        

    except Exception as e:

        logger.error(f"❌ 기본 글로벌 전략 생성 실패: {e}")

        return 0


# ===================== 추가 분석 함수들 =====================
# 분석 함수들은 rl_pipeline.strategy.analyzer 모듈에 정의되어 있습니다.


