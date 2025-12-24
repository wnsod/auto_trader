"""
전략 분석 모듈
DNA/프랙탈 분석 및 전략 품질 분석
"""

import logging
import pandas as pd
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from rl_pipeline.core.types import Strategy
from rl_pipeline.core.errors import StrategyError
from rl_pipeline.core.env import config
from rl_pipeline.core.utils import format_strategy_data
from rl_pipeline.data import load_candles, ensure_indicators
from rl_pipeline.strategy.param_space import sample_param_grid
from rl_pipeline.strategy.factory import make_strategy
from rl_pipeline.strategy.serializer import serialize_strategy
from rl_pipeline.db.writes import write_batch
from rl_pipeline.db.connection_pool import get_optimized_db_connection

logger = logging.getLogger(__name__)

def extract_optimal_conditions_from_analysis(dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any]) -> Dict[str, Any]:

    """DNA/프랙탈 분석 결과에서 최적 조건 추출"""

    try:

        optimal_conditions = {

            'rsi_range': {'min': 10, 'max': 90},  # 학습 데이터 확보를 위해 넓은 범위

            'volume_ratio': {'min': 0.3, 'max': 5.0},  # 학습 데이터 확보를 위해 넓은 범위

            'profit_threshold': -0.02,  # -2% 손실까지 허용 (학습 데이터 확보)

            'win_rate_threshold': 0.25,  # 25% 승률 (학습 데이터 확보)

            'trades_threshold': 1,  # 최소 1회 거래 (학습 데이터 확보)

            'dna_quality_score': 0.3  # 더 낮은 기준

        }

        

        # DNA 분석 결과에서 최적 조건 추출

        if dna_analysis and 'optimal_conditions' in dna_analysis:

            dna_conditions = dna_analysis['optimal_conditions']

            if 'rsi_range' in dna_conditions:

                optimal_conditions['rsi_range'] = dna_conditions['rsi_range']

            if 'volume_ratio' in dna_conditions:

                optimal_conditions['volume_ratio'] = dna_conditions['volume_ratio']

            if 'dna_quality_score' in dna_analysis:

                optimal_conditions['dna_quality_score'] = dna_analysis['dna_quality_score']

        

        # 프랙탈 분석 결과에서 최적 조건 추출

        if fractal_analysis and 'optimal_conditions' in fractal_analysis:

            fractal_conditions = fractal_analysis['optimal_conditions']

            if 'rsi_min' in fractal_conditions and 'rsi_max' in fractal_conditions:

                optimal_conditions['rsi_range'] = {

                    'min': fractal_conditions['rsi_min'],

                    'max': fractal_conditions['rsi_max']

                }

            if 'volume_ratio_min' in fractal_conditions:

                optimal_conditions['volume_ratio']['min'] = fractal_conditions['volume_ratio_min']

        

        logger.debug(f"📊 최적 조건 추출 완료: {optimal_conditions}")

        return optimal_conditions

        

    except Exception as e:

        logger.error(f"❌ 최적 조건 추출 실패: {e}")

        return {

            'rsi_range': {'min': 30, 'max': 70},

            'volume_ratio': {'min': 1.0, 'max': 2.0},

            'profit_threshold': 0.0,

            'win_rate_threshold': 0.4,

            'trades_threshold': 3,

            'dna_quality_score': 0.5

        }



def extract_routing_patterns_from_analysis(dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:

    """DNA/프랙탈 분석 결과에서 라우팅 패턴 추출"""

    try:

        routing_patterns = []

        

        # DNA 분석에서 라우팅 패턴 추출

        if dna_analysis and 'routing_patterns' in dna_analysis:

            dna_patterns = dna_analysis['routing_patterns']

            for pattern in dna_patterns:

                routing_patterns.append({

                    'pattern_type': 'dna_based',

                    'market_condition': pattern.get('market_condition', 'neutral'),

                    'rsi_range': pattern.get('rsi_range', {'min': 30, 'max': 70}),

                    'volume_ratio': pattern.get('volume_ratio', {'min': 1.0, 'max': 2.0}),

                    'confidence': pattern.get('confidence', 0.5),

                    'source': 'dna_analysis'

                })

        

        # 프랙탈 분석에서 라우팅 패턴 추출

        if fractal_analysis and 'market_conditions' in fractal_analysis:

            fractal_conditions = fractal_analysis['market_conditions']

            for condition, params in fractal_conditions.items():

                routing_patterns.append({

                    'pattern_type': 'fractal_based',

                    'market_condition': condition,

                    'rsi_range': params.get('rsi_range', {'min': 30, 'max': 70}),

                    'volume_ratio': params.get('volume_ratio', {'min': 1.0, 'max': 2.0}),

                    'confidence': params.get('confidence', 0.5),

                    'source': 'fractal_analysis'

                })

        

        # 기본 패턴 추가 (분석 결과가 없는 경우)

        if not routing_patterns:

            routing_patterns = [

                {

                    'pattern_type': 'default',

                    'market_condition': 'neutral',

                    'rsi_range': {'min': 30, 'max': 70},

                    'volume_ratio': {'min': 1.0, 'max': 2.0},

                    'confidence': 0.5,

                    'source': 'default'

                }

            ]

        

        logger.debug(f"🎯 라우팅 패턴 추출 완료: {len(routing_patterns)}개 패턴")

        return routing_patterns

        

    except Exception as e:

        logger.error(f"❌ 라우팅 패턴 추출 실패: {e}")

        return [{

            'pattern_type': 'default',

            'market_condition': 'neutral',

            'rsi_range': {'min': 30, 'max': 70},

            'volume_ratio': {'min': 1.0, 'max': 2.0},

            'confidence': 0.5,

            'source': 'default'

        }]



def _analyze_global_params_from_strategies(all_strategies) -> Dict[str, Any]:

    """self-play 결과에서 전역 파라미터 분석 - 양수 수익/상위 전략만 사용
    
    Args:
        all_strategies: {coin: [strategies]} 또는 {coin: {interval: [strategies]}} 형태
    """

    try:

        positive_profit_strategies = []

        all_win_rates = []

        all_trade_counts = []

        

        # 🔥 두 가지 형태 지원: {coin: [strategies]} 또는 {coin: {interval: [strategies]}}
        for coin, coin_data in all_strategies.items():
            if isinstance(coin_data, list):
                # {coin: [strategies]} 형태
                strategies = coin_data
                for strategy in strategies:
                    profit = strategy.get('profit', 0)
                    win_rate = strategy.get('win_rate', 0)
                    grade = strategy.get('quality_grade', 'UNKNOWN')
                    
                    # 양수 수익 + 검증된 전략만 수집 (UNKNOWN 제외)
                    if profit is not None and profit > 0 and grade != 'UNKNOWN':
                        positive_profit_strategies.append({
                            'profit': float(profit),
                            'win_rate': float(win_rate),
                            'trades': int(strategy.get('total_trades', 0))
                        })
                    
                    if win_rate:
                        all_win_rates.append(float(win_rate))
                    if strategy.get('total_trades'):
                        all_trade_counts.append(int(strategy['total_trades']))
            elif isinstance(coin_data, dict):
                # {coin: {interval: [strategies]}} 형태 (하위 호환성)
                for interval, strategies in coin_data.items():
                    for strategy in strategies:
                        profit = strategy.get('profit', 0)
                        win_rate = strategy.get('win_rate', 0)
                        grade = strategy.get('quality_grade', 'UNKNOWN')
                        
                        # 양수 수익 + 검증된 전략만 수집 (UNKNOWN 제외)
                        if profit is not None and profit > 0 and grade != 'UNKNOWN':
                            positive_profit_strategies.append({
                                'profit': float(profit),
                                'win_rate': float(win_rate),
                                'trades': int(strategy.get('total_trades', 0))
                            })
                        
                        if win_rate:
                            all_win_rates.append(float(win_rate))
                        if strategy.get('total_trades'):
                            all_trade_counts.append(int(strategy['total_trades']))

        

        # 양수 수익 전략 분석

        if positive_profit_strategies:

            # 상위 50% 성과 전략만 사용

            sorted_strategies = sorted(positive_profit_strategies, key=lambda x: x['profit'], reverse=True)

            top_50_count = max(1, int(len(sorted_strategies) * 0.5))

            top_strategies = sorted_strategies[:top_50_count]

            

            avg_profit = sum(s['profit'] for s in top_strategies) / len(top_strategies)

            avg_win_rate = sum(s['win_rate'] for s in top_strategies) / len(top_strategies)

            avg_trades = sum(s['trades'] for s in top_strategies) / len(top_strategies)

            

            logger.info(f"  ✅ 양수 수익 전략 {len(positive_profit_strategies)}개 중 상위 50%인 {len(top_strategies)}개 분석")

        else:

            # 양수 수익 전략이 없는 경우: 전체 중 상위 30% 사용

            logger.warning("⚠️ 양수 수익 전략 없음, 전체 중 상위 30% 전략 사용")

            

            all_strategies = []

            # 🔥 두 가지 형태 지원: {coin: [strategies]} 또는 {coin: {interval: [strategies]}}
            for coin, coin_data in all_strategies.items():
                if isinstance(coin_data, list):
                    # {coin: [strategies]} 형태
                    strategies = coin_data
                    for strategy in strategies:
                        profit = strategy.get('profit', 0)
                        win_rate = strategy.get('win_rate', 0)
                        grade = strategy.get('quality_grade', 'UNKNOWN')
                        
                        # UNKNOWN 제외
                        if profit is not None and grade != 'UNKNOWN':
                            all_strategies.append({
                                'profit': float(profit),
                                'win_rate': float(win_rate),
                                'trades': int(strategy.get('total_trades', 0))
                            })
                elif isinstance(coin_data, dict):
                    # {coin: {interval: [strategies]}} 형태 (하위 호환성)
                    for interval, strategies in coin_data.items():
                        for strategy in strategies:
                            profit = strategy.get('profit', 0)
                            win_rate = strategy.get('win_rate', 0)
                            grade = strategy.get('quality_grade', 'UNKNOWN')
                            
                            # UNKNOWN 제외
                            if profit is not None and grade != 'UNKNOWN':
                                all_strategies.append({
                                    'profit': float(profit),
                                    'win_rate': float(win_rate),
                                    'trades': int(strategy.get('total_trades', 0))
                                })

            

            if all_strategies:

                sorted_all = sorted(all_strategies, key=lambda x: x['profit'], reverse=True)

                top_30_count = max(1, int(len(sorted_all) * 0.3))

                top_30 = sorted_all[:top_30_count]

                

                avg_profit = sum(s['profit'] for s in top_30) / len(top_30)

                avg_win_rate = sum(s['win_rate'] for s in top_30) / len(top_30)

                avg_trades = sum(s['trades'] for s in top_30) / len(top_30)

                

                logger.info(f"  ✅ 전체 전략 중 상위 30%인 {len(top_30)}개 분석")

            else:

                # 폴백: 기본값

                avg_profit = 0.02  # 2% 목표

                avg_win_rate = 0.5

                avg_trades = 100

        

        return {

            'target_profit': float(avg_profit * 1.1),  # 평균의 110%

            'min_win_rate': float(avg_win_rate * 0.9),  # 평균의 90%

            'max_trades': int(avg_trades * 1.2),

            'risk_factor': 0.02 if avg_win_rate > 0.55 else 0.03,

            'num_strategies_analyzed': len(positive_profit_strategies)

        }

    except Exception as e:

        logger.error(f"❌ 전역 파라미터 분석 실패: {e}")

        return {

            'target_profit': 0.05,

            'min_win_rate': 0.5,

            'max_trades': 100,

            'risk_factor': 0.02

        }



def _analyze_common_strategy_patterns(all_strategies: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

    """공통 전략 패턴 분석
    
    Args:
        all_strategies: {coin: [strategies]} 또는 {coin: {interval: [strategies]}} 형태
    """

    try:

        all_params = []

        # 🔥 두 가지 형태 지원: {coin: [strategies]} 또는 {coin: {interval: [strategies]}}
        for coin, coin_data in all_strategies.items():
            if isinstance(coin_data, list):
                # {coin: [strategies]} 형태
                strategies = coin_data
                for strategy in strategies:
                    # 🔥 전략이 dict인지 확인 (문자열이 아닌 경우만 처리)
                    if not isinstance(strategy, dict):
                        logger.debug(f"⚠️ 전략이 dict가 아님 (타입: {type(strategy)}), 건너뜀")
                        continue
                    params = strategy.get('params', {})
                    # params가 dict인지 확인
                    if isinstance(params, dict) and params:
                        all_params.append(params)
                    elif isinstance(params, str):
                        # params가 문자열인 경우 JSON 파싱 시도
                        try:
                            import json
                            params_dict = json.loads(params)
                            if isinstance(params_dict, dict) and params_dict:
                                all_params.append(params_dict)
                        except Exception:
                            logger.debug(f"⚠️ params JSON 파싱 실패: {params}")
                            continue
            elif isinstance(coin_data, dict):
                # {coin: {interval: [strategies]}} 형태 (하위 호환성)
                for interval, strategies in coin_data.items():
                    if not isinstance(strategies, list):
                        logger.debug(f"⚠️ {coin}-{interval}: strategies가 list가 아님 (타입: {type(strategies)}), 건너뜀")
                        continue
                    for strategy in strategies:
                        # 🔥 전략이 dict인지 확인 (문자열이 아닌 경우만 처리)
                        if not isinstance(strategy, dict):
                            logger.debug(f"⚠️ {coin}-{interval}: 전략이 dict가 아님 (타입: {type(strategy)}), 건너뜀")
                            continue
                        params = strategy.get('params', {})
                        # params가 dict인지 확인
                        if isinstance(params, dict) and params:
                            all_params.append(params)
                        elif isinstance(params, str):
                            # params가 문자열인 경우 JSON 파싱 시도
                            try:
                                import json
                                params_dict = json.loads(params)
                                if isinstance(params_dict, dict) and params_dict:
                                    all_params.append(params_dict)
                            except Exception:
                                logger.debug(f"⚠️ {coin}-{interval}: params JSON 파싱 실패: {params}")
                                continue

        

        if not all_params:

            return {

                'rsi_range': (30, 70),

                'macd_threshold': 0.01,

                'volume_ratio': 1.5,

                'stop_loss': 0.02,

                'take_profit': 0.05

            }

        

        # 공통 파라미터 추출

        rsi_mins = [p.get('rsi_min', 30) for p in all_params if 'rsi_min' in p]

        rsi_maxs = [p.get('rsi_max', 70) for p in all_params if 'rsi_max' in p]

        macd_thresholds = [p.get('macd_buy_threshold', 0.01) for p in all_params if 'macd_buy_threshold' in p]

        volume_ratios = [p.get('volume_ratio_min', 1.0) for p in all_params if 'volume_ratio_min' in p]

        

        return {

            'rsi_min': float(sum(rsi_mins) / len(rsi_mins)) if rsi_mins else 30.0,

            'rsi_max': float(sum(rsi_maxs) / len(rsi_maxs)) if rsi_maxs else 70.0,

            'macd_threshold': float(sum(macd_thresholds) / len(macd_thresholds)) if macd_thresholds else 0.01,

            'volume_ratio': float(sum(volume_ratios) / len(volume_ratios)) if volume_ratios else 1.5,

            'stop_loss': 0.02,

            'take_profit': 0.05,

            'num_patterns_analyzed': len(all_params)

        }

    except Exception as e:

        logger.error(f"❌ 공통 패턴 분석 실패: {e}")

        return {

            'rsi_range': (30, 70),

            'macd_threshold': 0.01,

            'volume_ratio': 1.5,

            'stop_loss': 0.02,

            'take_profit': 0.05

        }



def _analyze_correlation_across_coins(all_coin_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:

    """코인 간 상관관계 분석"""

    try:

        import numpy as np

        import pandas as pd

        

        if not all_coin_data:

            return {

                'correlation_window': 24,

                'correlation_threshold': 0.8,

                'diversification_factor': 0.3,

                'rebalance_frequency': 4

            }

        

        # 각 코인의 주요 인터벌에서 가격 데이터 추출

        price_data = {}

        for coin, intervals_data in all_coin_data.items():

            # 가장 길이가 긴 인터벌 선택

            best_interval = max(intervals_data.keys(), key=lambda x: len(intervals_data[x]) if intervals_data[x] is not None and not intervals_data[x].empty else 0)

            df = intervals_data[best_interval]

            

            if df is not None and not df.empty and 'close' in df.columns:

                # 최근 100개 캔들만 사용

                price_data[coin] = df['close'].tail(100).values

        

        if len(price_data) < 2:

            logger.warning("⚠️ 코인 간 상관관계 분석: 데이터 부족")

            return {

                'correlation_window': 24,

                'correlation_threshold': 0.8,

                'diversification_factor': 0.3,

                'rebalance_frequency': 4

            }

        

        # 상관관계 계산

        df_correlation = pd.DataFrame(price_data)

        correlation_matrix = df_correlation.corr()

        

        # 평균 상관관계

        # 대각선 제외하고 계산 (자기 자신 제외)

        mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)

        upper_triangle = correlation_matrix.values[mask]

        avg_correlation = np.mean(np.abs(upper_triangle))

        

        # 상관관계 기반 파라미터 설정

        if avg_correlation > 0.7:

            # 높은 상관관계 -> 적극적 다각화

            diversification = 0.5

            rebalance_freq = 2

        elif avg_correlation > 0.4:

            # 중간 상관관계 -> 보통 다각화

            diversification = 0.3

            rebalance_freq = 4

        else:

            # 낮은 상관관계 -> 보수적 다각화

            diversification = 0.2

            rebalance_freq = 6

        

        params = {

            'correlation_window': 24,

            'correlation_threshold': float(min(0.8, max(0.5, avg_correlation))),

            'diversification_factor': diversification,

            'rebalance_frequency': rebalance_freq,

            'avg_correlation': float(avg_correlation),

            'num_coins_analyzed': len(price_data)

        }

        

        logger.info(f"✅ 코인 간 상관관계 분석 완료: 평균 {avg_correlation:.3f} ({len(price_data)}개 코인)")

        return params

        

    except Exception as e:

        logger.error(f"❌ 코인 간 상관관계 분석 실패: {e}")

        return {

            'correlation_window': 24,

            'correlation_threshold': 0.8,

            'diversification_factor': 0.3,

            'rebalance_frequency': 4

        }



def _analyze_global_market_trend(all_coin_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:

    """전체 시장 트렌드 분석"""

    try:

        import numpy as np

        

        if not all_coin_data:

            return {

                'trend_threshold': 0.02,

                'volume_threshold': 1.5,

                'correlation_threshold': 0.7,

                'risk_level': 'medium'

            }

        

        # 모든 코인의 가격 변동률 계산

        all_returns = []

        all_volumes = []

        

        for coin, intervals_data in all_coin_data.items():

            for interval, df in intervals_data.items():

                if df is not None and not df.empty and 'close' in df.columns:

                    returns = df['close'].pct_change().dropna()

                    all_returns.extend(returns.tolist())

                    

                    if 'volume' in df.columns:

                        volume_data = df['volume'].tail(100).tolist()

                        all_volumes.extend(volume_data)

        

        if not all_returns:

            logger.warning("⚠️ 전체 시장 트렌드 분석: 데이터 부족")

            return {

                'trend_threshold': 0.02,

                'volume_threshold': 1.5,

                'correlation_threshold': 0.7,

                'risk_level': 'medium'

            }

        

        # 트렌드 강도 분석

        avg_return = np.mean(all_returns)

        std_return = np.std(all_returns)

        

        # 트렌드 강도에 따른 파라미터 설정

        if abs(avg_return) > 0.01:

            # 강한 트렌드

            trend_threshold = 0.03

            risk_level = 'high'

        elif abs(avg_return) > 0.005:

            # 보통 트렌드

            trend_threshold = 0.02

            risk_level = 'medium'

        else:

            # 약한 트렌드

            trend_threshold = 0.01

            risk_level = 'low'

        

        # 거래량 분석

        if all_volumes:

            avg_volume = np.mean(all_volumes)

            volume_threshold = 1.5 if avg_volume > 1e6 else 1.2

        else:

            volume_threshold = 1.5

        

        params = {

            'trend_threshold': float(trend_threshold),

            'volume_threshold': float(volume_threshold),

            'correlation_threshold': 0.7,

            'risk_level': risk_level,

            'avg_return': float(avg_return),

            'volatility': float(std_return),

            'num_coins_analyzed': len(all_coin_data)

        }

        

        logger.info(f"✅ 전체 시장 트렌드 분석 완료: 평균 수익률 {avg_return:.4f} ({len(all_coin_data)}개 코인)")

        return params

        

    except Exception as e:

        logger.error(f"❌ 전체 시장 트렌드 분석 실패: {e}")

        return {

            'trend_threshold': 0.02,

            'volume_threshold': 1.5,

            'correlation_threshold': 0.7,

            'risk_level': 'medium'

        }



def _analyze_global_regime(all_coin_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:

    """전체 시장 레짐 분석"""

    try:

        import numpy as np

        from rl_pipeline.routing.regime_router import RegimeRouter

        

        if not all_coin_data:

            return {

                'regime_detection_window': 48,

                'regime_confidence_threshold': 0.6,

                'transition_sensitivity': 0.4,

                'regime_weight_factor': 0.8

            }

        

        router = RegimeRouter()

        

        # 모든 코인의 레짐 분석

        all_regimes = []

        all_confidences = []

        

        for coin, intervals_data in all_coin_data.items():

            for interval, df in intervals_data.items():

                if df is not None and not df.empty:

                    try:

                        regime, confidence, regime_transition_prob = router.detect_current_regime(coin, interval, df)

                        all_regimes.append(regime)

                        all_confidences.append(confidence)

                    except Exception as e:

                        logger.debug(f"⚠️ {coin} {interval} 레짐 분석 실패: {e}")

        

        if not all_regimes:

            logger.warning("⚠️ 전체 시장 레짐 분석: 데이터 부족")

            return {

                'regime_detection_window': 48,

                'regime_confidence_threshold': 0.6,

                'transition_sensitivity': 0.4,

                'regime_weight_factor': 0.8

            }

        

        # 평균 레짐 신뢰도

        avg_confidence = np.mean(all_confidences) if all_confidences else 0.5

        

        # 레짐 분포 분석

        regime_counts = {}

        for regime in all_regimes:

            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        

        # 신뢰도 기반 파라미터 설정

        if avg_confidence > 0.7:

            confidence_threshold = 0.6

            weight_factor = 0.9

        elif avg_confidence > 0.5:

            confidence_threshold = 0.5

            weight_factor = 0.8

        else:

            confidence_threshold = 0.4

            weight_factor = 0.7

        

        # 레짐 전환 민감도

        # 다양한 레짐이 많으면 민감도 높음

        num_unique_regimes = len(regime_counts)

        if num_unique_regimes > 4:

            transition_sensitivity = 0.6

        elif num_unique_regimes > 2:

            transition_sensitivity = 0.4

        else:

            transition_sensitivity = 0.2

        

        params = {

            'regime_detection_window': 48,

            'regime_confidence_threshold': confidence_threshold,

            'transition_sensitivity': transition_sensitivity,

            'regime_weight_factor': weight_factor,

            'avg_confidence': float(avg_confidence),

            'regime_distribution': regime_counts,

            'num_coins_analyzed': len(all_coin_data)

        }

        

        logger.info(f"✅ 전체 시장 레짐 분석 완료: 평균 신뢰도 {avg_confidence:.3f} (레짐 {num_unique_regimes}종, {len(all_coin_data)}개 코인)")

        return params

        

    except Exception as e:

        logger.error(f"❌ 전체 시장 레짐 분석 실패: {e}")

        return {

            'regime_detection_window': 48,

            'regime_confidence_threshold': 0.6,

            'transition_sensitivity': 0.4,

            'regime_weight_factor': 0.8

        }



def _analyze_strategy_quality_distribution(all_strategies: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:

    """전략 품질 분포 분석 (메타 분석)"""

    try:

        grade_counts = {}

        total_strategies = 0

        

        for coin, intervals_data in all_strategies.items():

            for interval, strategies in intervals_data.items():

                for strategy in strategies:

                    grade = strategy.get('quality_grade', 'UNKNOWN')

                    grade_counts[grade] = grade_counts.get(grade, 0) + 1

                    total_strategies += 1

        

        return {

            'total_strategies': total_strategies,

            'grade_s': grade_counts.get('S', 0),

            'grade_a': grade_counts.get('A', 0),

            'grade_b': grade_counts.get('B', 0),

            'grade_c': grade_counts.get('C', 0),

            'grade_d': grade_counts.get('D', 0),

            'grade_f': grade_counts.get('F', 0),

            'quality_rate': (grade_counts.get('S', 0) + grade_counts.get('A', 0)) / total_strategies if total_strategies > 0 else 0

        }

    except Exception as e:

        logger.error(f"❌ 전략 품질 분포 분석 실패: {e}")

        return {}





def _analyze_regime_based_optimal_params(all_strategies) -> Dict[str, Any]:

    """레짐별 최적 파라미터 분석
    
    Args:
        all_strategies: {coin: [strategies]} 또는 {coin: {interval: [strategies]}} 형태
    """

    try:

        regime_performance = {}

        # 🔥 두 가지 형태 지원: {coin: [strategies]} 또는 {coin: {interval: [strategies]}}
        for coin, coin_data in all_strategies.items():
            if isinstance(coin_data, list):
                # {coin: [strategies]} 형태
                strategies = coin_data
                for strategy in strategies:
                    regime = strategy.get('regime', 'unknown')
                    quality = strategy.get('quality_grade', 'F')
                    
                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    
                    params = strategy.get('params', {})
                    # params가 문자열인 경우 JSON 파싱
                    if isinstance(params, str):
                        try:
                            import json
                            params = json.loads(params) if params else {}
                        except:
                            params = {}
                    
                    regime_performance[regime].append({
                        'profit': strategy.get('profit', 0),
                        'win_rate': strategy.get('win_rate', 0),
                        'quality': quality,
                        'params': params if isinstance(params, dict) else {}
                    })
            elif isinstance(coin_data, dict):
                # {coin: {interval: [strategies]}} 형태 (하위 호환성)
                for interval, strategies in coin_data.items():
                    for strategy in strategies:
                        regime = strategy.get('regime', 'unknown')
                        quality = strategy.get('quality_grade', 'F')
                        
                        if regime not in regime_performance:
                            regime_performance[regime] = []
                        
                        params = strategy.get('params', {})
                        # params가 문자열인 경우 JSON 파싱
                        if isinstance(params, str):
                            try:
                                import json
                                params = json.loads(params) if params else {}
                            except:
                                params = {}
                        
                        regime_performance[regime].append({
                            'profit': strategy.get('profit', 0),
                            'win_rate': strategy.get('win_rate', 0),
                            'quality': quality,
                            'params': params if isinstance(params, dict) else {}
                        })

        

        # 레짐별 최적 파라미터 추출

        optimal_params = {}

        for regime, strategies in regime_performance.items():

            # S/A등급 전략만 선택

            top_strategies = [s for s in strategies if s.get('quality', 'F') in ['S', 'A']]

            

            if top_strategies:

                # 평균 파라미터 계산
                rsi_mins = []
                rsi_maxs = []
                
                for s in top_strategies:
                    params = s.get('params', {})
                    # params가 문자열인 경우 JSON 파싱
                    if isinstance(params, str):
                        try:
                            import json
                            params = json.loads(params) if params else {}
                        except:
                            params = {}
                    
                    if isinstance(params, dict):
                        if 'rsi_min' in params:
                            rsi_mins.append(params['rsi_min'])
                        if 'rsi_max' in params:
                            rsi_maxs.append(params['rsi_max'])

                

                optimal_params[regime] = {

                    'rsi_min': sum(rsi_mins) / len(rsi_mins) if rsi_mins else 30,

                    'rsi_max': sum(rsi_maxs) / len(rsi_maxs) if rsi_maxs else 70,

                    'avg_profit': sum(s['profit'] for s in top_strategies) / len(top_strategies),

                    'avg_win_rate': sum(s['win_rate'] for s in top_strategies) / len(top_strategies),

                    'strategy_count': len(top_strategies)

                }

        

        return optimal_params

    except Exception as e:

        logger.error(f"❌ 레짐별 최적 파라미터 분석 실패: {e}")

        return {}





def _analyze_parameter_performance_correlation(all_strategies) -> Dict[str, Any]:

    """파라미터-성과 상관관계 분석
    
    Args:
        all_strategies: {coin: [strategies]} 또는 {coin: {interval: [strategies]}} 형태
    """

    try:

        param_data = {

            'rsi_min': [],

            'rsi_max': [],

            'macd_buy': [],

            'volume_ratio_min': []

        }

        # 🔥 두 가지 형태 지원: {coin: [strategies]} 또는 {coin: {interval: [strategies]}}
        for coin, coin_data in all_strategies.items():
            if isinstance(coin_data, list):
                # {coin: [strategies]} 형태
                strategies = coin_data
                for strategy in strategies:
                    params = strategy.get('params', {})
                    # params가 문자열인 경우 JSON 파싱
                    if isinstance(params, str):
                        try:
                            import json
                            params = json.loads(params) if params else {}
                        except:
                            params = {}
                    
                    performance = strategy.get('profit', 0)
                    
                    if isinstance(params, dict):
                        if 'rsi_min' in params:
                            param_data['rsi_min'].append((params['rsi_min'], performance))
                        if 'rsi_max' in params:
                            param_data['rsi_max'].append((params['rsi_max'], performance))
                        if 'macd_buy_threshold' in params:
                            param_data['macd_buy'].append((params['macd_buy_threshold'], performance))
                        if 'volume_ratio_min' in params:
                            param_data['volume_ratio_min'].append((params['volume_ratio_min'], performance))
            elif isinstance(coin_data, dict):
                # {coin: {interval: [strategies]}} 형태 (하위 호환성)
                for interval, strategies in coin_data.items():
                    for strategy in strategies:
                        params = strategy.get('params', {})
                        # params가 문자열인 경우 JSON 파싱
                        if isinstance(params, str):
                            try:
                                import json
                                params = json.loads(params) if params else {}
                            except:
                                params = {}
                        
                        performance = strategy.get('profit', 0)
                        
                        if isinstance(params, dict):
                            if 'rsi_min' in params:
                                param_data['rsi_min'].append((params['rsi_min'], performance))
                            if 'rsi_max' in params:
                                param_data['rsi_max'].append((params['rsi_max'], performance))
                            if 'macd_buy_threshold' in params:
                                param_data['macd_buy'].append((params['macd_buy_threshold'], performance))
                            if 'volume_ratio_min' in params:
                                param_data['volume_ratio_min'].append((params['volume_ratio_min'], performance))

        

        correlations = {}

        for param_name, data in param_data.items():

            if len(data) > 10:

                # 성과 상위 30%에 해당하는 파라미터 범위 찾기

                sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

                top_30 = sorted_data[:max(1, len(sorted_data) // 3)]

                

                values = [x[0] for x in top_30]

                correlations[param_name] = {

                    'optimal_min': min(values),

                    'optimal_max': max(values),

                    'optimal_avg': sum(values) / len(values),

                    'correlation_samples': len(data)

                }

        

        return correlations

    except Exception as e:

        logger.error(f"❌ 파라미터-성과 상관관계 분석 실패: {e}")

        return {}





def _analyze_coin_group_performance_difference(all_strategies) -> Dict[str, Any]:

    """코인 그룹별 성과 차이 분석
    
    Args:
        all_strategies: {coin: [strategies]} 또는 {coin: {interval: [strategies]}} 형태
    """

    try:

        major_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'TRX', 'AVAX', 

                      'MATIC', 'LTC', 'LINK', 'BCH', 'UNI', 'ATOM', 'ETC', 'XLM', 'ALGO', 'FIL']

        

        major_performances = []

        mid_performances = []

        # 🔥 두 가지 형태 지원: {coin: [strategies]} 또는 {coin: {interval: [strategies]}}
        for coin, coin_data in all_strategies.items():
            coin_profits = []
            
            if isinstance(coin_data, list):
                # {coin: [strategies]} 형태
                strategies = coin_data
                for strategy in strategies:
                    profit = strategy.get('profit', 0)
                    if profit:
                        coin_profits.append(profit)
            elif isinstance(coin_data, dict):
                # {coin: {interval: [strategies]}} 형태 (하위 호환성)
                for interval, strategies in coin_data.items():
                    for strategy in strategies:
                        profit = strategy.get('profit', 0)
                        if profit:
                            coin_profits.append(profit)
            
            avg_profit = sum(coin_profits) / len(coin_profits) if coin_profits else 0

            

            if coin in major_coins:

                major_performances.append(avg_profit)

            else:

                mid_performances.append(avg_profit)

        

        return {

            'major_avg': (sum(major_performances) / len(major_performances)) if major_performances else 0,

            'mid_avg': (sum(mid_performances) / len(mid_performances)) if mid_performances else 0,

            'major_count': len(major_performances),

            'mid_count': len(mid_performances)

        }

    except Exception as e:

        logger.error(f"❌ 코인 그룹별 성과 차이 분석 실패: {e}")

        return {}


def _categorize_coins_by_importance(all_strategies: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[str]]:
    """코인을 중요도별로 그룹화 (메이저/중형)"""
    try:
        major_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'TRX', 'AVAX', 
                      'MATIC', 'LTC', 'LINK', 'BCH', 'UNI', 'ATOM', 'ETC', 'XLM', 'ALGO', 'FIL']
        
        coin_groups = {
            'major': [],
            'mid': []
        }
        
        for coin in all_strategies.keys():
            if coin in major_coins:
                coin_groups['major'].append(coin)
            else:
                coin_groups['mid'].append(coin)
        
        logger.info(f"📊 코인 그룹화: 메이저 {len(coin_groups['major'])}개, 중형 {len(coin_groups['mid'])}개")
        
        return coin_groups
        
    except Exception as e:
        logger.error(f"❌ 코인 그룹화 실패: {e}")
        # 폴백: 모든 코인을 메이저로 처리
        return {
            'major': list(all_strategies.keys()),
            'mid': []
        }
