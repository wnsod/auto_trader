"""
전략 라우팅 모듈
시장 상황별 동적 라우팅 및 전략 선택
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

def run_dynamic_routing_by_market_condition(coin: str, intervals: List[str], 

                                          all_candle_data: Dict[Tuple[str, str], pd.DataFrame],

                                          dna_analysis: Dict[str, Any] = None,

                                          fractal_analysis: Dict[str, Any] = None) -> Dict[str, Any]:

    """시장 상황별 동적 라우팅 실행 - 실제 분석 결과 기반 복잡한 라우팅"""

    try:

        logger.info(f"🎯 {coin} 시장 상황별 동적 라우팅 시작")

        

        routing_results = {}

        total_routing_strategies = 0

        

        for interval in intervals:

            try:

                df = all_candle_data.get((coin, interval))

                if df is None or df.empty:

                    logger.warning(f"⚠️ {coin} {interval} 캔들 데이터 없음")

                    continue

                

                logger.info(f"🎯 {coin} {interval} 동적 라우팅 시작")

                

                # 1. 실제 시장 상황 분석

                market_condition = analyze_market_condition_from_actual_data(coin, interval, df)

                logger.info(f"📊 {coin} {interval} 시장 상황: {market_condition}")

                

                # 2. 실제 분석 결과 기반 라우팅 전략 생성

                routing_strategies = create_routing_strategies_from_actual_analysis(

                    coin, interval, market_condition, dna_analysis, fractal_analysis

                )

                

                if not routing_strategies:

                    logger.warning(f"⚠️ {coin} {interval} 라우팅 전략 생성 실패")

                    continue

                

                logger.info(f"🎯 {coin} {interval} 라우팅 전략 {len(routing_strategies)}개 생성됨")

                

                # 3. 실제 라우팅 실행

                routing_result = execute_routing_from_actual_strategies(

                    coin, interval, routing_strategies, df, market_condition

                )

                

                if routing_result:

                    routing_results[interval] = {

                        'market_condition': market_condition,

                        'routing_strategies': routing_strategies,

                        'routing_result': routing_result,

                        'success': routing_result.get('success', False),

                        'total_trades': routing_result.get('total_trades', 0),

                        'profit': routing_result.get('profit', 0.0),

                        'win_rate': routing_result.get('win_rate', 0.0)

                    }

                    total_routing_strategies += len(routing_strategies)

                    logger.info(f"✅ {coin} {interval} 라우팅 완료: {routing_result.get('total_trades', 0)}개 거래")

                else:

                    logger.warning(f"⚠️ {coin} {interval} 라우팅 실행 실패")

                

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 라우팅 실패: {e}")

                continue

        

        # 4. 라우팅 결과 통합 및 저장

        if routing_results:

            integrated_routing = integrate_routing_results_from_actual_data(coin, routing_results, dna_analysis, fractal_analysis)

            

            # 라우팅 결과 저장

            from rl_pipeline.db.writes import save_routing_by_market_condition

            save_routing_by_market_condition(coin, routing_results, integrated_routing)

            

            logger.info(f"✅ {coin} 시장 상황별 동적 라우팅 완료: {len(routing_results)}개 인터벌, {total_routing_strategies}개 전략")

            return integrated_routing

        else:

            logger.warning(f"⚠️ {coin} 동적 라우팅 실패: 모든 인터벌에서 실패")

            return create_default_routing_result(coin)

        

    except Exception as e:

        logger.error(f"❌ {coin} 시장 상황별 동적 라우팅 실패: {e}")

        return create_default_routing_result(coin)



def analyze_market_condition_from_actual_data(coin: str, interval: str, df: pd.DataFrame) -> str:

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

        logger.error(f"❌ {coin} 레짐 분석 실패: {e}")

        return "neutral"



def create_routing_strategies_from_actual_analysis(coin: str, interval: str, market_condition: str,

                                                  dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:

    """실제 분석 결과 기반 라우팅 전략 생성"""

    try:

        routing_strategies = []

        

        # 실제 DNA 분석 결과 활용

        market_dna = {}

        if dna_analysis and market_condition in dna_analysis:

            market_dna = dna_analysis[market_condition]

        

        # 실제 프랙탈 분석 결과 활용

        market_fractal = {}

        if fractal_analysis and market_condition in fractal_analysis:

            market_fractal = fractal_analysis[market_condition]

        

        # 시장 상황별 실제 분석 기반 전략 생성

        if market_condition == "overbought":

            routing_strategies.extend(create_reversal_strategies_from_analysis(coin, interval, market_dna, market_fractal))

        elif market_condition == "oversold":

            routing_strategies.extend(create_reversal_strategies_from_analysis(coin, interval, market_dna, market_fractal))

        elif market_condition == "bullish":

            routing_strategies.extend(create_trend_strategies_from_analysis(coin, interval, market_dna, market_fractal))

        elif market_condition == "bearish":

            routing_strategies.extend(create_trend_strategies_from_analysis(coin, interval, market_dna, market_fractal))

        elif market_condition == "neutral":

            routing_strategies.extend(create_range_strategies_from_analysis(coin, interval, market_dna, market_fractal))

        elif market_condition == "low_volume":

            routing_strategies.extend(create_conservative_strategies_from_analysis(coin, interval, market_dna, market_fractal))

        

        logger.info(f"🎯 {coin} {interval} {market_condition} 실제 분석 기반 라우팅 전략 {len(routing_strategies)}개 생성")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ 실제 분석 기반 라우팅 전략 생성 실패: {e}")

        return []



def create_reversal_strategies_from_analysis(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """실제 분석 결과 기반 반전 전략 생성"""

    try:

        strategies = []

        

        # DNA 분석 결과에서 RSI 패턴 추출

        rsi_patterns = market_dna.get('rsi_patterns', {})

        rsi_min = rsi_patterns.get('avg_min', 30)

        rsi_max = rsi_patterns.get('avg_max', 70)

        

        # 프랙탈 분석 결과에서 복잡도 추출

        complexity_score = market_fractal.get('complexity_score', 0.5)

        

        # 복잡도에 따른 전략 다양성 조정

        strategy_count = max(3, int(complexity_score * 10))

        

        for i in range(strategy_count):

            strategy = {

                'strategy_type': 'reversal',

                'coin': coin,

                'interval': interval,

                'rsi_min': max(20, rsi_min - 5 + i * 2),

                'rsi_max': min(80, rsi_max + 5 - i * 2),

                'volume_ratio_min': 1.5 + i * 0.1,

                'volume_ratio_max': 2.5 + i * 0.1,

                'macd_buy_threshold': -0.1 - i * 0.02,

                'macd_sell_threshold': 0.1 + i * 0.02,

                'atr_multiplier': 1.5 + i * 0.1,

                'adx_threshold': 25 + i * 2,

                'bb_period': 20,

                'bb_std_dev': 2.0 + i * 0.1,

                'strategy_conditions': f"reversal_strategy_{i+1}",

                'created_at': datetime.now().isoformat()

            }

            # 소숫점 정리 후 추가

            strategies.append(format_strategy_data(strategy))

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 반전 전략 생성 실패: {e}")

        return []



def create_trend_strategies_from_analysis(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """실제 분석 결과 기반 추세 전략 생성"""

    try:

        strategies = []

        

        # DNA 분석 결과에서 볼륨 패턴 추출

        volume_patterns = market_dna.get('volume_patterns', {})

        volume_min = volume_patterns.get('avg_min', 1.0)

        volume_max = volume_patterns.get('avg_max', 2.0)

        

        # 프랙탈 분석 결과에서 안정성 추출

        stability_score = market_fractal.get('stability_score', 0.5)

        

        # 안정성에 따른 전략 수 조정

        strategy_count = max(3, int(stability_score * 8))

        

        for i in range(strategy_count):

            strategy = {

                'strategy_type': 'trend_following',

                'coin': coin,

                'interval': interval,

                'rsi_min': 40 + i * 2,

                'rsi_max': 60 + i * 2,

                'volume_ratio_min': max(1.0, volume_min - 0.2 + i * 0.1),

                'volume_ratio_max': min(3.0, volume_max + 0.2 - i * 0.1),

                'macd_buy_threshold': 0.05 + i * 0.01,

                'macd_sell_threshold': -0.05 - i * 0.01,

                'atr_multiplier': 2.0 + i * 0.1,

                'adx_threshold': 30 + i * 3,

                'bb_period': 20,

                'bb_std_dev': 2.0,

                'strategy_conditions': f"trend_strategy_{i+1}",

                'created_at': datetime.now().isoformat()

            }

            strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 추세 전략 생성 실패: {e}")

        return []



def create_range_strategies_from_analysis(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """실제 분석 결과 기반 범위 거래 전략 생성"""

    try:

        strategies = []

        

        # 프랙탈 분석 결과에서 확장성 추출

        scalability_score = market_fractal.get('scalability_score', 0.5)

        

        # 확장성에 따른 전략 수 조정

        strategy_count = max(2, int(scalability_score * 6))

        

        for i in range(strategy_count):

            strategy = {

                'strategy_type': 'range_trading',

                'coin': coin,

                'interval': interval,

                'rsi_min': 35 + i * 3,

                'rsi_max': 65 - i * 3,

                'volume_ratio_min': 1.2 + i * 0.1,

                'volume_ratio_max': 1.8 - i * 0.1,

                'macd_buy_threshold': 0.0,

                'macd_sell_threshold': 0.0,

                'atr_multiplier': 1.0 + i * 0.2,

                'adx_threshold': 20 + i * 2,

                'bb_period': 20,

                'bb_std_dev': 1.5 + i * 0.1,

                'strategy_conditions': f"range_strategy_{i+1}",

                'created_at': datetime.now().isoformat()

            }

            strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 범위 거래 전략 생성 실패: {e}")

        return []



def create_conservative_strategies_from_analysis(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """실제 분석 결과 기반 보수적 전략 생성"""

    try:

        strategies = []

        

        # 프랙탈 분석 결과에서 안정성 추출

        stability_score = market_fractal.get('stability_score', 0.5)

        

        # 안정성에 따른 보수적 전략 수 조정

        strategy_count = max(2, int(stability_score * 4))

        

        for i in range(strategy_count):

            strategy = {

                'strategy_type': 'conservative',

                'coin': coin,

                'interval': interval,

                'rsi_min': 30 + i * 5,

                'rsi_max': 70 - i * 5,

                'volume_ratio_min': 1.0 + i * 0.2,

                'volume_ratio_max': 1.5 + i * 0.2,

                'macd_buy_threshold': 0.02 + i * 0.01,

                'macd_sell_threshold': -0.02 - i * 0.01,

                'atr_multiplier': 1.0 + i * 0.1,

                'adx_threshold': 25 + i * 2,

                'bb_period': 20,

                'bb_std_dev': 1.5 + i * 0.1,

                'strategy_conditions': f"conservative_strategy_{i+1}",

                'created_at': datetime.now().isoformat()

            }

            strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 보수적 전략 생성 실패: {e}")

        return []



def execute_routing_from_actual_strategies(coin: str, interval: str, routing_strategies: List[Dict[str, Any]], 

                                         df: pd.DataFrame, market_condition: str) -> Dict[str, Any]:

    """실제 전략으로 라우팅 실행"""

    try:

        if not routing_strategies or df.empty:

            return {}

        

        total_trades = 0

        total_profit = 0.0

        successful_trades = 0

        

        # 각 전략별로 간단한 백테스트 실행

        for strategy in routing_strategies:

            try:

                # 간단한 백테스트 로직

                trades, profit, wins, predictive_accuracy = execute_simple_backtest(strategy, df)

                total_trades += trades

                total_profit += profit

                successful_trades += wins

            except Exception as e:

                logger.warning(f"⚠️ 전략 백테스트 실패: {e}")

                continue

        

        # 라우팅 결과 계산

        win_rate = successful_trades / total_trades if total_trades > 0 else 0.0

        

        routing_result = {

            'coin': coin,

            'interval': interval,

            'market_condition': market_condition,

            'total_strategies': len(routing_strategies),

            'total_trades': total_trades,

            'successful_trades': successful_trades,

            'win_rate': win_rate,

            'total_profit': total_profit,

            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0.0,

            'success': total_trades > 0,

            'execution_timestamp': datetime.now().isoformat()

        }

        

        return routing_result

        

    except Exception as e:

        logger.error(f"❌ 라우팅 실행 실패: {e}")

        return {}



def execute_simple_backtest(strategy: Dict[str, Any], df: pd.DataFrame) -> Tuple[int, float, int, float]:
    """
    간단한 백테스트 실행 - 예측 정확도 계산 추가
    
    Returns:
        (trades, profit, wins, predictive_accuracy)
        - trades: 거래 횟수
        - profit: 총 수익률
        - wins: 승리 횟수
        - predictive_accuracy: 예측 정확도 (0.0 ~ 1.0)
    """
    try:
        # 🔥 데이터 검증
        if df.empty or len(df) < 50:
            return 0, 0.0, 0, 0.0

        # 🔥 필수 컬럼 체크
        if 'close' not in df.columns:
            logger.warning("⚠️ 백테스트: 'close' 컬럼 없음")
            return 0, 0.0, 0, 0.0

        trades = 0
        profit = 0.0
        wins = 0
        # 🔥 예측 정확도 계산을 위한 변수
        prediction_correct = 0  # 예측 맞춘 횟수
        prediction_total = 0    # 총 예측 횟수

        # 🔥 전략 파라미터 접근 수정 (getattr → get)
        rsi_min = strategy.get('rsi_min', 30.0)
        rsi_max = strategy.get('rsi_max', 70.0)

        # 🔥 파라미터 유효성 검증
        if not isinstance(rsi_min, (int, float)) or not isinstance(rsi_max, (int, float)):
            logger.warning(f"⚠️ 백테스트: 잘못된 RSI 파라미터 (min={rsi_min}, max={rsi_max})")
            rsi_min, rsi_max = 30.0, 70.0

        if rsi_min >= rsi_max:
            logger.warning(f"⚠️ 백테스트: rsi_min >= rsi_max ({rsi_min} >= {rsi_max}), 기본값 사용")
            rsi_min, rsi_max = 30.0, 70.0

        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        # 🔥 0으로 나누기 방지
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 매매 시뮬레이션
        position = None
        entry_price = 0.0
        entry_index = 0
        max_hold_periods = 20  # 🔥 최대 보유 기간 (20 캔들)

        for i in range(20, len(df)):  # RSI 계산을 위해 20부터 시작
            current_rsi = rsi.iloc[i]
            current_price = df['close'].iloc[i]

            # 🔥 NaN 체크
            if pd.isna(current_rsi) or pd.isna(current_price):
                continue

            # 🔥 가격 유효성 체크
            if current_price <= 0:
                continue

            if position is None:
                # 매수 신호 (상승 예측)
                if current_rsi < rsi_min:
                    position = 'long'
                    entry_price = current_price
                    entry_index = i
                    # 🔥 예측 정확도 계산: 매수 신호 = 상승 예측
                    # 매도 시점까지의 가격 변화로 예측 정확도 확인
                    # (매수 시점에서 즉시 확인하지 않고, 실제 거래 결과로 확인)
                    prediction_total += 1  # 예측 횟수 증가
            else:
                # 🔥 개선된 매도 조건: 3가지 경우
                should_exit = False

                # 1. 기본 매도 신호 (RSI > rsi_max)
                if current_rsi > rsi_max:
                    should_exit = True

                # 2. 손절 조건 (5% 이상 손실)
                elif (current_price - entry_price) / entry_price < -0.05:
                    should_exit = True

                # 3. 최대 보유 기간 초과 시 강제 청산
                elif (i - entry_index) >= max_hold_periods:
                    should_exit = True

                if should_exit:
                    trade_profit = (current_price - entry_price) / entry_price
                    profit += trade_profit
                    trades += 1
                    if trade_profit > 0:
                        wins += 1
                    
                    # 🔥 예측 정확도 계산: 매도 시점에서 예측 검증
                    # 매수 시점의 상승 예측이 맞았는지 확인
                    price_change = (current_price - entry_price) / entry_price
                    # 상승 예측이 맞았는지 (0.1% 이상 수익이면 예측 정확)
                    if price_change > 0.001:  # 0.1% 이상 수익 = 상승 예측 맞춤
                        prediction_correct += 1
                    # 손실이면 예측 틀림 (이미 prediction_total은 매수 시점에서 증가)
                    
                    position = None

        # 🔥 마지막 포지션이 열려있으면 강제 청산
        if position is not None and len(df) > 0:
            final_price = df['close'].iloc[-1]
            if not pd.isna(final_price) and final_price > 0:
                trade_profit = (final_price - entry_price) / entry_price
                profit += trade_profit
                trades += 1
                if trade_profit > 0:
                    wins += 1
                
                # 🔥 마지막 포지션의 예측 정확도 계산
                price_change = (final_price - entry_price) / entry_price
                if price_change > 0.001:  # 0.1% 이상 수익 = 상승 예측 맞춤
                    prediction_correct += 1
                # prediction_total은 이미 매수 시점에서 증가했음

        # 🔥 예측 정확도 계산
        predictive_accuracy = prediction_correct / prediction_total if prediction_total > 0 else 0.0
        
        return trades, profit, wins, predictive_accuracy

    except Exception as e:
        logger.error(f"❌ 간단한 백테스트 실패: {e}")
        import traceback
        logger.debug(f"백테스트 에러 상세:\n{traceback.format_exc()}")
        return 0, 0.0, 0, 0.0



def integrate_routing_results_from_actual_data(coin: str, routing_results: Dict[str, Any], 

                                             dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any]) -> Dict[str, Any]:

    """실제 데이터 기반 라우팅 결과 통합"""

    try:

        if not routing_results:

            return create_default_routing_result(coin)

        

        # 전체 성과 집계

        total_trades = sum(result.get('total_trades', 0) for result in routing_results.values())

        total_profit = sum(result.get('profit', 0.0) for result in routing_results.values())

        total_strategies = sum((result.get('routing_strategies', []) for result in routing_results.values()), [])

        

        # 평균 성과 계산

        avg_win_rate = sum(result.get('win_rate', 0.0) for result in routing_results.values()) / len(routing_results)

        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0.0

        

        # 시장 상황별 성과 분석

        market_performance = {}

        for interval, result in routing_results.items():

            market_condition = result.get('market_condition', 'unknown')

            if market_condition not in market_performance:

                market_performance[market_condition] = {

                    'intervals': [],

                    'total_trades': 0,

                    'total_profit': 0.0,

                    'avg_win_rate': 0.0

                }

            

            market_performance[market_condition]['intervals'].append(interval)

            market_performance[market_condition]['total_trades'] += result.get('total_trades', 0)

            market_performance[market_condition]['total_profit'] += result.get('profit', 0.0)

            market_performance[market_condition]['avg_win_rate'] += result.get('win_rate', 0.0)

        

        # 평균 계산

        for condition in market_performance:

            interval_count = len(market_performance[condition]['intervals'])

            market_performance[condition]['avg_win_rate'] /= interval_count

        

        # 통합 결과 생성

        integrated_result = {

            'coin': coin,

            'analysis_type': 'dynamic_routing',

            'timestamp': datetime.now().isoformat(),

            'total_intervals': len(routing_results),

            'total_strategies': len(total_strategies),

            'total_trades': total_trades,

            'total_profit': total_profit,

            'avg_win_rate': avg_win_rate,

            'avg_profit_per_trade': avg_profit_per_trade,

            'market_performance': market_performance,

            'routing_results': routing_results,

            'dna_analysis_summary': summarize_dna_analysis(dna_analysis),

            'fractal_analysis_summary': summarize_fractal_analysis(fractal_analysis),

            'routing_quality_score': calculate_routing_quality_score(routing_results),

            'success': total_trades > 0

        }

        

        return integrated_result

        

    except Exception as e:

        logger.error(f"❌ 라우팅 결과 통합 실패: {e}")

        return create_default_routing_result(coin)



def summarize_dna_analysis(dna_analysis: Dict[str, Any]) -> Dict[str, Any]:

    """DNA 분석 요약"""

    try:

        if not dna_analysis:

            return {}

        

        summary = {

            'analyzed_conditions': list(dna_analysis.keys()),

            'total_conditions': len(dna_analysis),

            'analysis_timestamp': datetime.now().isoformat()

        }

        

        # 각 조건별 품질 점수 집계

        quality_scores = []

        for condition, analysis in dna_analysis.items():

            if isinstance(analysis, dict) and 'dna_quality_score' in analysis:

                quality_scores.append(analysis['dna_quality_score'])

        

        if quality_scores:

            summary['avg_quality_score'] = sum(quality_scores) / len(quality_scores)

            summary['max_quality_score'] = max(quality_scores)

            summary['min_quality_score'] = min(quality_scores)

        

        return summary

        

    except Exception as e:

        logger.error(f"❌ DNA 분석 요약 실패: {e}")

        return {}



def summarize_fractal_analysis(fractal_analysis: Dict[str, Any]) -> Dict[str, Any]:

    """프랙탈 분석 요약"""

    try:

        if not fractal_analysis:

            return {}

        

        summary = {

            'analyzed_conditions': list(fractal_analysis.keys()),

            'total_conditions': len(fractal_analysis),

            'analysis_timestamp': datetime.now().isoformat()

        }

        

        # 각 조건별 품질 점수 집계

        quality_scores = []

        for condition, analysis in fractal_analysis.items():

            if isinstance(analysis, dict) and 'fractal_quality_score' in analysis:

                quality_scores.append(analysis['fractal_quality_score'])

        

        if quality_scores:

            summary['avg_quality_score'] = sum(quality_scores) / len(quality_scores)

            summary['max_quality_score'] = max(quality_scores)

            summary['min_quality_score'] = min(quality_scores)

        

        return summary

        

    except Exception as e:

        logger.error(f"❌ 프랙탈 분석 요약 실패: {e}")

        return {}



def calculate_routing_quality_score(routing_results: Dict[str, Any]) -> float:

    """라우팅 품질 점수 계산"""

    try:

        if not routing_results:

            return 0.0

        

        quality_factors = []

        

        # 거래 수 기반 점수

        total_trades = sum(result.get('total_trades', 0) for result in routing_results.values())

        trade_score = min(total_trades / 100, 1.0)

        quality_factors.append(trade_score)

        

        # 승률 기반 점수

        win_rates = [result.get('win_rate', 0.0) for result in routing_results.values()]

        if win_rates:

            avg_win_rate = sum(win_rates) / len(win_rates)

            win_rate_score = min(avg_win_rate, 1.0)

            quality_factors.append(win_rate_score)

        

        # 수익률 기반 점수

        profits = [result.get('profit', 0.0) for result in routing_results.values()]

        if profits:

            avg_profit = sum(profits) / len(profits)

            profit_score = min(max(avg_profit, 0.0) / 0.1, 1.0)  # 최대 10%로 정규화

            quality_factors.append(profit_score)

        

        # 인터벌 다양성 점수

        interval_diversity_score = min(len(routing_results) / 4, 1.0)

        quality_factors.append(interval_diversity_score)

        

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

        

    except Exception as e:

        logger.error(f"❌ 라우팅 품질 점수 계산 실패: {e}")

        return 0.0



def create_default_routing_result(coin: str) -> Dict[str, Any]:

    """기본 라우팅 결과 생성 (fallback)"""

    return {

        'coin': coin,

        'analysis_type': 'dynamic_routing',

        'timestamp': datetime.now().isoformat(),

        'total_intervals': 0,

        'total_strategies': 0,

        'total_trades': 0,

        'total_profit': 0.0,

        'avg_win_rate': 0.0,

        'avg_profit_per_trade': 0.0,

        'market_performance': {},

        'routing_results': {},

        'dna_analysis_summary': {},

        'fractal_analysis_summary': {},

        'routing_quality_score': 0.0,

        'success': False

    }



def create_routing_strategies_by_market(coin: str, interval: str, market_condition: str,

                                      dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:

    """시장 상황별 라우팅 전략 생성"""

    try:

        routing_strategies = []

        

        # 시장 상황별 DNA 패턴 로드

        market_dna = dna_analysis.get(market_condition, {}) if dna_analysis else {}

        

        # 시장 상황별 프랙탈 패턴 로드

        market_fractal = fractal_analysis.get(market_condition, {}) if fractal_analysis else {}

        

        # 시장 상황별 라우팅 전략 생성

        if market_condition == "overbought":

            # 과매수 시장: 반전 전략 중심

            routing_strategies.extend(create_reversal_routing_strategies(coin, interval, market_dna, market_fractal))

            

        elif market_condition == "oversold":

            # 과매도 시장: 반전 전략 중심

            routing_strategies.extend(create_reversal_routing_strategies(coin, interval, market_dna, market_fractal))

            

        elif market_condition == "bullish":

            # 상승 시장: 추세 추종 전략 중심

            routing_strategies.extend(create_trend_following_routing_strategies(coin, interval, market_dna, market_fractal))

            

        elif market_condition == "bearish":

            # 하락 시장: 추세 추종 전략 중심

            routing_strategies.extend(create_trend_following_routing_strategies(coin, interval, market_dna, market_fractal))

            

        elif market_condition == "neutral":

            # 중립 시장: 범위 거래 전략 중심

            routing_strategies.extend(create_range_trading_routing_strategies(coin, interval, market_dna, market_fractal))

            

        elif market_condition == "low_volume":

            # 저볼륨 시장: 보수적 전략 중심

            routing_strategies.extend(create_conservative_routing_strategies(coin, interval, market_dna, market_fractal))

        

        logger.info(f"🎯 {coin} {interval} {market_condition} 라우팅 전략 {len(routing_strategies)}개 생성")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ 라우팅 전략 생성 실패: {e}")

        return []



def create_reversal_routing_strategies(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """반전 전략 중심 라우팅 전략 생성"""

    try:

        strategies = []

        

        # DNA 패턴 기반 반전 전략

        if market_dna.get('rsi_patterns', {}).get('pattern_type') == 'high_rsi_narrow_range':

            strategy = {

                'strategy_id': f"{coin}_{interval}_reversal_dna_{int(time.time())}",

                'strategy_type': 'reversal_dna',

                'market_condition': 'overbought',

                'rsi_min': market_dna['rsi_patterns']['avg_min'],

                'rsi_max': market_dna['rsi_patterns']['avg_max'],

                'volume_ratio_min': 1.2,

                'volume_ratio_max': 2.5,

                'confidence': 0.8,

                'description': 'DNA 패턴 기반 반전 전략'

            }

            strategies.append(strategy)

        

        # 프랙탈 패턴 기반 반전 전략

        if market_fractal.get('wave_patterns', {}).get('dominant_pattern') == 'reversal_wave':

            strategy = {

                'strategy_id': f"{coin}_{interval}_reversal_fractal_{int(time.time())}",

                'strategy_type': 'reversal_fractal',

                'market_condition': 'overbought',

                'rsi_min': 70,

                'rsi_max': 85,

                'volume_ratio_min': 1.5,

                'volume_ratio_max': 3.0,

                'confidence': market_fractal.get('stability_score', 0.7),

                'description': '프랙탈 패턴 기반 반전 전략'

            }

            strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 반전 라우팅 전략 생성 실패: {e}")

        return []



def create_trend_following_routing_strategies(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """추세 추종 전략 중심 라우팅 전략 생성"""

    try:

        strategies = []

        

        # DNA 패턴 기반 추세 추종 전략

        if market_dna.get('rsi_patterns', {}).get('pattern_type') == 'medium_rsi_wide_range':

            strategy = {

                'strategy_id': f"{coin}_{interval}_trend_dna_{int(time.time())}",

                'strategy_type': 'trend_following_dna',

                'market_condition': 'bullish',

                'rsi_min': market_dna['rsi_patterns']['avg_min'],

                'rsi_max': market_dna['rsi_patterns']['avg_max'],

                'volume_ratio_min': 1.0,

                'volume_ratio_max': 2.0,

                'confidence': 0.7,

                'description': 'DNA 패턴 기반 추세 추종 전략'

            }

            strategies.append(strategy)

        

        # 프랙탈 패턴 기반 추세 추종 전략

        if market_fractal.get('wave_patterns', {}).get('dominant_pattern') == 'trend_wave':

            strategy = {

                'strategy_id': f"{coin}_{interval}_trend_fractal_{int(time.time())}",

                'strategy_type': 'trend_following_fractal',

                'market_condition': 'bullish',

                'rsi_min': 40,

                'rsi_max': 70,

                'volume_ratio_min': 1.2,

                'volume_ratio_max': 2.5,

                'confidence': market_fractal.get('stability_score', 0.6),

                'description': '프랙탈 패턴 기반 추세 추종 전략'

            }

            strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 추세 추종 라우팅 전략 생성 실패: {e}")

        return []



def create_range_trading_routing_strategies(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """범위 거래 전략 중심 라우팅 전략 생성"""

    try:

        strategies = []

        

        # DNA 패턴 기반 범위 거래 전략

        if market_dna.get('rsi_patterns', {}).get('pattern_type') == 'balanced_rsi_medium_range':

            strategy = {

                'strategy_id': f"{coin}_{interval}_range_dna_{int(time.time())}",

                'strategy_type': 'range_trading_dna',

                'market_condition': 'neutral',

                'rsi_min': market_dna['rsi_patterns']['avg_min'],

                'rsi_max': market_dna['rsi_patterns']['avg_max'],

                'volume_ratio_min': 0.8,

                'volume_ratio_max': 1.5,

                'confidence': 0.6,

                'description': 'DNA 패턴 기반 범위 거래 전략'

            }

            strategies.append(strategy)

        

        # 프랙탈 패턴 기반 범위 거래 전략

        if market_fractal.get('wave_patterns', {}).get('dominant_pattern') == 'sideways_wave':

            strategy = {

                'strategy_id': f"{coin}_{interval}_range_fractal_{int(time.time())}",

                'strategy_type': 'range_trading_fractal',

                'market_condition': 'neutral',

                'rsi_min': 35,

                'rsi_max': 65,

                'volume_ratio_min': 0.9,

                'volume_ratio_max': 1.8,

                'confidence': market_fractal.get('stability_score', 0.8),

                'description': '프랙탈 패턴 기반 범위 거래 전략'

            }

            strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 범위 거래 라우팅 전략 생성 실패: {e}")

        return []



def create_conservative_routing_strategies(coin: str, interval: str, market_dna: Dict[str, Any], market_fractal: Dict[str, Any]) -> List[Dict[str, Any]]:

    """보수적 전략 중심 라우팅 전략 생성"""

    try:

        strategies = []

        

        # 보수적 전략 생성

        strategy = {

            'strategy_id': f"{coin}_{interval}_conservative_{int(time.time())}",

            'strategy_type': 'conservative',

            'market_condition': 'low_volume',

            'rsi_min': 30,

            'rsi_max': 70,

            'volume_ratio_min': 0.5,

            'volume_ratio_max': 1.2,

            'confidence': 0.9,

            'description': '보수적 저볼륨 전략'

        }

        strategies.append(strategy)

        

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 보수적 라우팅 전략 생성 실패: {e}")

        return []



def execute_routing_by_market(coin: str, interval: str, market_condition: str, 

                            routing_strategies: List[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Any]:

    """시장 상황별 라우팅 실행"""

    try:

        if not routing_strategies:

            return {'success': False, 'message': '라우팅 전략이 없음'}

        

        # 시장 상황별 라우팅 실행 로직

        routing_result = {

            'success': True,

            'market_condition': market_condition,

            'strategies_count': len(routing_strategies),

            'execution_time': datetime.now().isoformat(),

            'routing_score': calculate_routing_score(routing_strategies, market_condition),

            'recommended_strategy': select_best_routing_strategy(routing_strategies, market_condition)

        }

        

        return routing_result

        

    except Exception as e:

        logger.error(f"❌ 라우팅 실행 실패: {e}")

        return {'success': False, 'error': str(e)}



def calculate_routing_score(routing_strategies: List[Dict[str, Any]], market_condition: str) -> float:

    """라우팅 점수 계산"""

    try:

        if not routing_strategies:

            return 0.0

        

        # 시장 상황별 가중치

        market_weights = {

            "overbought": 0.8,

            "oversold": 0.8,

            "bullish": 0.9,

            "bearish": 0.9,

            "neutral": 0.7,

            "low_volume": 0.6

        }

        

        weight = market_weights.get(market_condition, 0.7)

        

        # 평균 신뢰도 계산

        avg_confidence = sum(s.get('confidence', 0.5) for s in routing_strategies) / len(routing_strategies)

        

        return avg_confidence * weight

        

    except Exception as e:

        logger.error(f"❌ 라우팅 점수 계산 실패: {e}")

        return 0.0



def select_best_routing_strategy(routing_strategies: List[Dict[str, Any]], market_condition: str) -> Dict[str, Any]:

    """최적 라우팅 전략 선택"""

    try:

        if not routing_strategies:

            return {}

        

        # 신뢰도가 가장 높은 전략 선택

        best_strategy = max(routing_strategies, key=lambda s: s.get('confidence', 0.0))

        

        return {

            'strategy_id': best_strategy.get('strategy_id', ''),

            'strategy_type': best_strategy.get('strategy_type', ''),

            'confidence': best_strategy.get('confidence', 0.0),

            'market_condition': market_condition,

            'description': best_strategy.get('description', '')

        }

        

    except Exception as e:

        logger.error(f"❌ 최적 라우팅 전략 선택 실패: {e}")

        return {}



def create_integrated_routing_by_market(coin: str, routing_results: Dict[str, Any]) -> Dict[str, Any]:

    """통합 라우팅 결과 생성"""

    try:

        integrated_routing = {

            'coin': coin,

            'analysis_timestamp': datetime.now().isoformat(),

            'interval_results': routing_results,

            'overall_score': calculate_overall_routing_score(routing_results),

            'recommended_intervals': get_recommended_intervals(routing_results),

            'market_condition_summary': get_market_condition_summary(routing_results)

        }

        

        return integrated_routing

        

    except Exception as e:

        logger.error(f"❌ 통합 라우팅 결과 생성 실패: {e}")

        return {}



def calculate_overall_routing_score(routing_results: Dict[str, Any]) -> float:

    """전체 라우팅 점수 계산"""

    try:

        scores = []

        for interval, result in routing_results.items():

            if result.get('success', False):

                routing_score = result.get('routing_result', {}).get('routing_score', 0.0)

                scores.append(routing_score)

        

        return sum(scores) / len(scores) if scores else 0.0

        

    except Exception as e:

        logger.error(f"❌ 전체 라우팅 점수 계산 실패: {e}")

        return 0.0



def get_recommended_intervals(routing_results: Dict[str, Any]) -> List[str]:

    """추천 인터벌 목록 반환"""

    try:

        recommended = []

        for interval, result in routing_results.items():

            if result.get('success', False):

                routing_score = result.get('routing_result', {}).get('routing_score', 0.0)

                if routing_score > 0.7:  # 높은 점수 기준

                    recommended.append(interval)

        

        return sorted(recommended, key=lambda x: routing_results[x].get('routing_result', {}).get('routing_score', 0.0), reverse=True)

        

    except Exception as e:

        logger.error(f"❌ 추천 인터벌 목록 생성 실패: {e}")

        return []



def get_market_condition_summary(routing_results: Dict[str, Any]) -> Dict[str, Any]:

    """시장 상황 요약"""

    try:

        market_conditions = {}

        for interval, result in routing_results.items():

            market_condition = result.get('market_condition', 'unknown')

            if market_condition not in market_conditions:

                market_conditions[market_condition] = []

            market_conditions[market_condition].append(interval)

        

        return market_conditions

        

    except Exception as e:

        logger.error(f"❌ 시장 상황 요약 생성 실패: {e}")

        return {}



def run_coin_dynamic_routing(coin: str, intervals: List[str]) -> Dict[str, Any]:

    """기존 동적 라우팅 함수 (호환성 유지)"""

    try:

        logger.info(f"🎯 {coin} 기존 동적 라우팅 시작")

        

        # 통합 동적 라우팅 함수 호출

        return run_coin_dynamic_routing_integrated(coin, intervals, None, None, None)

        

    except Exception as e:

        logger.error(f"❌ {coin} 기존 동적 라우팅 실패: {e}")

        return {}



def run_coin_dynamic_routing_integrated(coin: str, intervals: List[str], 

                                      dna_analysis: Dict[str, Any] = None,

                                      fractal_analysis: Dict[str, Any] = None,

                                      all_candle_data: Dict[Tuple[str, str], pd.DataFrame] = None) -> bool:

    """🆕 통합 동적 라우팅 함수 - 모든 전략을 종합적으로 고려한 라우팅"""

    try:

        logger.info(f"🚀 {coin} 통합 동적 라우팅 시작 (DNA/프랙탈 분석 결과 활용)")

        

        # DNA/프랙탈 분석 결과에서 라우팅 패턴 추출

        routing_patterns = extract_routing_patterns_from_analysis(dna_analysis, fractal_analysis)

        logger.info(f"📊 {coin} 라우팅 패턴 추출 완료: {routing_patterns}")

        

        total_routing_strategies = 0

        

        for interval in intervals:

            try:

                # 🚀 동적 기간 분할 계산

                from simulation.replay import calculate_dynamic_periods

                periods = calculate_dynamic_periods(coin, interval, all_candle_data)

                

                if not periods['has_data']:

                    logger.warning(f"⚠️ {coin} {interval}: 데이터가 없어 라우팅 건너뜀")

                    continue

                

                logger.info(f"🚀 {coin} {interval} 통합 동적 라우팅 시작...")

                

                # 🚀 모든 전략 타입을 종합적으로 고려한 라우팅 전략 생성

                routing_strategies = create_integrated_routing_strategies(

                    coin, interval, periods, dna_analysis, fractal_analysis, routing_patterns

                )

                

                if routing_strategies:

                    # 라우팅 전략 저장

                    saved_count = save_dynamic_routing_strategies_to_db(routing_strategies)

                    total_routing_strategies += saved_count

                    logger.info(f"✅ {coin} {interval}: {saved_count}개 라우팅 전략 저장")

                else:

                    logger.warning(f"⚠️ {coin} {interval}: 라우팅 전략 생성 실패")

                

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 라우팅 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} 통합 동적 라우팅 완료: 총 {total_routing_strategies}개 라우팅 전략 생성")

        return True

        

    except Exception as e:

        logger.error(f"❌ {coin} 통합 동적 라우팅 실패: {e}")

        return False



def create_integrated_routing_strategies(coin: str, interval: str, periods: Dict[str, Any],

                                        dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any],

                                        routing_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:

    """통합 라우팅 전략 생성 - 모든 전략 타입을 종합적으로 고려"""

    try:

        logger.info(f"🎯 {coin} {interval} 통합 라우팅 전략 생성 시작")

        

        routing_strategies = []

        

        # 🚀 장기 전략 기반 라우팅 (15일 이상인 경우만)

        if periods['has_long_term']:

            logger.info(f"📈 {coin} {interval} 장기 기반 라우팅 전략 생성")

            long_term_routing = create_long_term_routing_strategies(

                coin, interval, periods, dna_analysis, fractal_analysis, routing_patterns

            )

            routing_strategies.extend(long_term_routing)

        

        # 🚀 단기 전략 기반 라우팅

        if periods['has_short_term']:

            if periods['has_long_term']:

                # 전반/후반 분할 라우팅

                logger.info(f"📊 {coin} {interval} 단기 전반 기반 라우팅 전략 생성")

                short_front_routing = create_short_term_front_routing_strategies(

                    coin, interval, periods, dna_analysis, fractal_analysis, routing_patterns

                )

                routing_strategies.extend(short_front_routing)

                

                logger.info(f"📊 {coin} {interval} 단기 후반 기반 라우팅 전략 생성")

                short_back_routing = create_short_term_back_routing_strategies(

                    coin, interval, periods, dna_analysis, fractal_analysis, routing_patterns

                )

                routing_strategies.extend(short_back_routing)

            else:

                # 단기만 라우팅

                logger.info(f"📊 {coin} {interval} 단기만 기반 라우팅 전략 생성")

                short_only_routing = create_short_term_only_routing_strategies(

                    coin, interval, periods, dna_analysis, fractal_analysis, routing_patterns

                )

                routing_strategies.extend(short_only_routing)

        

        # 🚀 하이브리드 라우팅 전략 생성 (장기+단기 조합)

        if periods['has_long_term'] and periods['has_short_term']:

            logger.info(f"🔄 {coin} {interval} 하이브리드 라우팅 전략 생성")

            hybrid_routing = create_hybrid_routing_strategies(

                coin, interval, periods, dna_analysis, fractal_analysis, routing_patterns

            )

            routing_strategies.extend(hybrid_routing)

        

        logger.info(f"✅ {coin} {interval} 통합 라우팅 전략 생성 완료: {len(routing_strategies)}개")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 통합 라우팅 전략 생성 실패: {e}")

        return []



def create_long_term_routing_strategies(coin: str, interval: str, periods: Dict[str, Any],

                                       dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any],

                                       routing_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:

    """장기 기반 라우팅 전략 생성 - 안정성 중심"""

    try:

        logger.info(f"📈 {coin} {interval} 장기 기반 라우팅 전략 생성 (안정성 중심)")

        

        # 장기 전략 조회 (A, B 등급)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                       rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                       quality_grade, complexity_score, score

                FROM strategies 

                WHERE symbol = ? AND interval = ?

                AND quality_grade IN ('A', 'B')

                ORDER BY profit DESC, win_rate DESC

                LIMIT 10

            """, (coin, interval))

            

            strategies = cursor.fetchall()

        

        routing_strategies = []

        

        for strategy in strategies:

            try:

                # 장기 라우팅 전략 생성 (안정성 중심)

                routing_strategy = {

                    'id': f"routing_long_{coin}_{interval}_{strategy[0]}",

                    'coin': coin,

                    'interval': interval,

                    'strategy_type': 'routing_long_term',

                    'base_strategy_id': strategy[0],

                    'routing_conditions': {

                        'market_condition': 'stable',

                        'volatility_threshold': 0.02,  # 낮은 변동성

                        'profit_threshold': 5.0,        # 높은 수익 기준

                        'risk_level': 'low',

                        'time_horizon': 'long_term'

                    },

                    'performance_metrics': {

                        'expected_profit': strategy[3] or 0,

                        'expected_trades': strategy[4] or 0,

                        'expected_win_rate': strategy[5] or 0,

                        'risk_score': 0.3  # 낮은 리스크

                    },

                    'created_at': datetime.now().isoformat(),

                    'updated_at': datetime.now().isoformat()

                }

                

                routing_strategies.append(routing_strategy)

                

            except Exception as e:

                logger.error(f"장기 라우팅 전략 {strategy[0]} 생성 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 장기 기반 라우팅 전략 생성 완료: {len(routing_strategies)}개")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 장기 기반 라우팅 전략 생성 실패: {e}")

        return []



def create_short_term_front_routing_strategies(coin: str, interval: str, periods: Dict[str, Any],

                                             dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any],

                                             routing_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:

    """단기 전반 기반 라우팅 전략 생성 - 민감성 중심"""

    try:

        logger.info(f"📊 {coin} {interval} 단기 전반 기반 라우팅 전략 생성 (민감성 중심)")

        

        # 단기 전반 전략 조회 (A, B 등급)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                       rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                       quality_grade, complexity_score, score

                FROM strategies 

                WHERE symbol = ? AND interval = ?

                AND quality_grade IN ('A', 'B')

                ORDER BY profit DESC, win_rate DESC

                LIMIT 8

            """, (coin, interval))

            

            strategies = cursor.fetchall()

        

        routing_strategies = []

        

        for strategy in strategies:

            try:

                # 단기 전반 라우팅 전략 생성 (민감성 중심)

                routing_strategy = {

                    'id': f"routing_short_front_{coin}_{interval}_{strategy[0]}",

                    'coin': coin,

                    'interval': interval,

                    'strategy_type': 'routing_short_term_front',

                    'base_strategy_id': strategy[0],

                    'routing_conditions': {

                        'market_condition': 'volatile',

                        'volatility_threshold': 0.05,  # 높은 변동성

                        'profit_threshold': 3.0,        # 중간 수익 기준

                        'risk_level': 'medium',

                        'time_horizon': 'short_term_front'

                    },

                    'performance_metrics': {

                        'expected_profit': strategy[3] or 0,

                        'expected_trades': strategy[4] or 0,

                        'expected_win_rate': strategy[5] or 0,

                        'risk_score': 0.6  # 중간 리스크

                    },

                    'created_at': datetime.now().isoformat(),

                    'updated_at': datetime.now().isoformat()

                }

                

                routing_strategies.append(routing_strategy)

                

            except Exception as e:

                logger.error(f"단기 전반 라우팅 전략 {strategy[0]} 생성 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 단기 전반 기반 라우팅 전략 생성 완료: {len(routing_strategies)}개")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 단기 전반 기반 라우팅 전략 생성 실패: {e}")

        return []



def create_short_term_back_routing_strategies(coin: str, interval: str, periods: Dict[str, Any],

                                            dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any],

                                            routing_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:

    """단기 후반 기반 라우팅 전략 생성 - 민감성 중심"""

    try:

        logger.info(f"📊 {coin} {interval} 단기 후반 기반 라우팅 전략 생성 (민감성 중심)")

        

        # 단기 후반 전략 조회 (A, B 등급)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                       rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                       quality_grade, complexity_score, score

                FROM strategies 

                WHERE symbol = ? AND interval = ?

                AND quality_grade IN ('A', 'B')

                ORDER BY profit DESC, win_rate DESC

                LIMIT 8

            """, (coin, interval))

            

            strategies = cursor.fetchall()

        

        routing_strategies = []

        

        for strategy in strategies:

            try:

                # 단기 후반 라우팅 전략 생성 (민감성 중심)

                routing_strategy = {

                    'id': f"routing_short_back_{coin}_{interval}_{strategy[0]}",

                    'coin': coin,

                    'interval': interval,

                    'strategy_type': 'routing_short_term_back',

                    'base_strategy_id': strategy[0],

                    'routing_conditions': {

                        'market_condition': 'volatile',

                        'volatility_threshold': 0.05,  # 높은 변동성

                        'profit_threshold': 3.0,        # 중간 수익 기준

                        'risk_level': 'medium',

                        'time_horizon': 'short_term_back'

                    },

                    'performance_metrics': {

                        'expected_profit': strategy[3] or 0,

                        'expected_trades': strategy[4] or 0,

                        'expected_win_rate': strategy[5] or 0,

                        'risk_score': 0.6  # 중간 리스크

                    },

                    'created_at': datetime.now().isoformat(),

                    'updated_at': datetime.now().isoformat()

                }

                

                routing_strategies.append(routing_strategy)

                

            except Exception as e:

                logger.error(f"단기 후반 라우팅 전략 {strategy[0]} 생성 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 단기 후반 기반 라우팅 전략 생성 완료: {len(routing_strategies)}개")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 단기 후반 기반 라우팅 전략 생성 실패: {e}")

        return []



def create_short_term_only_routing_strategies(coin: str, interval: str, periods: Dict[str, Any],

                                            dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any],

                                            routing_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:

    """단기만 기반 라우팅 전략 생성 - 민감성 중심"""

    try:

        logger.info(f"📊 {coin} {interval} 단기만 기반 라우팅 전략 생성 (민감성 중심)")

        

        # 단기만 전략 조회 (A, B 등급)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                       rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                       quality_grade, complexity_score, score

                FROM strategies 

                WHERE symbol = ? AND interval = ?

                AND quality_grade IN ('A', 'B')

                ORDER BY profit DESC, win_rate DESC

                LIMIT 10

            """, (coin, interval))

            

            strategies = cursor.fetchall()

        

        routing_strategies = []

        

        for strategy in strategies:

            try:

                # 단기만 라우팅 전략 생성 (민감성 중심)

                routing_strategy = {

                    'id': f"routing_short_only_{coin}_{interval}_{strategy[0]}",

                    'coin': coin,

                    'interval': interval,

                    'strategy_type': 'routing_short_term_only',

                    'base_strategy_id': strategy[0],

                    'routing_conditions': {

                        'market_condition': 'volatile',

                        'volatility_threshold': 0.05,  # 높은 변동성

                        'profit_threshold': 3.0,        # 중간 수익 기준

                        'risk_level': 'medium',

                        'time_horizon': 'short_term_only'

                    },

                    'performance_metrics': {

                        'expected_profit': strategy[3] or 0,

                        'expected_trades': strategy[4] or 0,

                        'expected_win_rate': strategy[5] or 0,

                        'risk_score': 0.6  # 중간 리스크

                    },

                    'created_at': datetime.now().isoformat(),

                    'updated_at': datetime.now().isoformat()

                }

                

                routing_strategies.append(routing_strategy)

                

            except Exception as e:

                logger.error(f"단기만 라우팅 전략 {strategy[0]} 생성 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 단기만 기반 라우팅 전략 생성 완료: {len(routing_strategies)}개")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 단기만 기반 라우팅 전략 생성 실패: {e}")

        return []



def create_hybrid_routing_strategies(coin: str, interval: str, periods: Dict[str, Any],

                                    dna_analysis: Dict[str, Any], fractal_analysis: Dict[str, Any],

                                    routing_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:

    """하이브리드 라우팅 전략 생성 - 장기+단기 조합"""

    try:

        logger.info(f"🔄 {coin} {interval} 하이브리드 라우팅 전략 생성 (장기+단기 조합)")

        

        # 장기 전략 조회 (A, B 등급)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                       rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                       quality_grade, complexity_score, score

                FROM strategies 

                WHERE symbol = ? AND interval = ?

                AND quality_grade IN ('A', 'B')

                ORDER BY profit DESC, win_rate DESC

                LIMIT 5

            """, (coin, interval))

            

            long_term_strategies = cursor.fetchall()

        

        # 단기 전략 조회 (A, B 등급)

        cursor.execute("""

            SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                   rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                   quality_grade, complexity_score, score

            FROM strategies 

            WHERE symbol = ? AND interval = ? AND strategy_type IN ('short_term_front', 'short_term_back')

            AND quality_grade IN ('A', 'B')

            ORDER BY profit DESC, win_rate DESC

            LIMIT 5

        """, (coin, interval))

        

        short_term_strategies = cursor.fetchall()

        

        routing_strategies = []

        

        # 하이브리드 라우팅 전략 생성 (장기+단기 조합)

        for long_strategy in long_term_strategies:

            for short_strategy in short_term_strategies:

                try:

                    # 하이브리드 라우팅 전략 생성

                    routing_strategy = {

                        'id': f"routing_hybrid_{coin}_{interval}_{long_strategy[0]}_{short_strategy[0]}",

                        'coin': coin,

                        'interval': interval,

                        'strategy_type': 'routing_hybrid',

                        'base_strategy_id': f"{long_strategy[0]}_{short_strategy[0]}",

                        'routing_conditions': {

                            'market_condition': 'adaptive',

                            'volatility_threshold': 0.03,  # 중간 변동성

                            'profit_threshold': 4.0,        # 중간 수익 기준

                            'risk_level': 'balanced',

                            'time_horizon': 'hybrid',

                            'long_term_weight': 0.6,       # 장기 60%

                            'short_term_weight': 0.4       # 단기 40%

                        },

                        'performance_metrics': {

                            'expected_profit': (long_strategy[3] or 0) * 0.6 + (short_strategy[3] or 0) * 0.4,

                            'expected_trades': (long_strategy[4] or 0) + (short_strategy[4] or 0),

                            'expected_win_rate': ((long_strategy[5] or 0) + (short_strategy[5] or 0)) / 2,

                            'risk_score': 0.45  # 균형 리스크

                        },

                        'created_at': datetime.now().isoformat(),

                        'updated_at': datetime.now().isoformat()

                    }

                    

                    routing_strategies.append(routing_strategy)

                    

                except Exception as e:

                    logger.error(f"하이브리드 라우팅 전략 생성 실패: {e}")

                    continue

        

        logger.info(f"✅ {coin} {interval} 하이브리드 라우팅 전략 생성 완료: {len(routing_strategies)}개")

        return routing_strategies

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 하이브리드 라우팅 전략 생성 실패: {e}")

        return []

    """🆕 코인별 동적 라우팅 함수 - DNA/프랙탈 분석 결과를 활용한 고도화된 동적 라우팅"""

    try:

        logger.info(f"🔄 {coin} 동적 라우팅 실행 (DNA/프랙탈 분석 결과 활용)")

        

        # DNA/프랙탈 분석 결과에서 라우팅 패턴 추출

        routing_patterns = extract_routing_patterns_from_analysis(dna_analysis, fractal_analysis)

        logger.info(f"🎯 {coin} 라우팅 패턴 추출 완료: {len(routing_patterns)}개 패턴")

        

        total_routing_strategies = 0

        

        for interval in intervals:

            try:

                logger.info(f"🔄 {coin} {interval} 동적 라우팅 전략 생성 시작...")

                

                # 1. 기존 고등급 전략들을 로드하여 패턴 분석

                existing_strategies = load_high_grade_strategies(coin, interval, num_strategies=5)

                logger.info(f"✅ {coin} {interval} 고등급 전략 {len(existing_strategies)}개 로드")

                

                if len(existing_strategies) < 3:

                    logger.warning(f"⚠️ {coin} {interval}: 고등급 전략이 부족하여 DNA/프랙탈 기반 전략 생성")

                    # DNA/프랙탈 분석 기반 전략 생성

                    strategies = create_dna_fractal_based_routing_strategies(

                        coin, interval, routing_patterns, num_strategies=5

                    )

                else:

                    # 기존 고등급 전략 + DNA/프랙탈 패턴을 기반으로 동적 라우팅 전략 생성

                    strategies = create_enhanced_dynamic_routing_strategies(

                        coin, interval, existing_strategies, routing_patterns, num_strategies=5

                    )

                

                if strategies:

                    # 2. 생성된 전략들을 DB에 저장

                    saved_count = save_dynamic_routing_strategies_to_db(strategies, coin, interval)

                    total_routing_strategies += saved_count

                    logger.info(f"✅ {coin} {interval}: {saved_count}개 동적 라우팅 전략 생성 및 저장")

                else:

                    logger.warning(f"⚠️ {coin} {interval}: 동적 라우팅 전략 생성 실패")

                    

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 동적 라우팅 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} 동적 라우팅 완료: 총 {total_routing_strategies}개 전략 생성")

        return True

        

    except Exception as e:

        logger.error(f"❌ {coin} 동적 라우팅 실패: {e}")

        return False



def create_dna_fractal_based_routing_strategies(coin: str, interval: str, 

                                               routing_patterns: List[Dict[str, Any]], 

                                               num_strategies: int = 5) -> List[Dict[str, Any]]:

    """DNA/프랙탈 분석 기반 라우팅 전략 생성"""

    try:

        strategies = []

        

        for i, pattern in enumerate(routing_patterns[:num_strategies]):

            strategy = {

                'id': f"{coin}_{interval}_dna_fractal_{pattern['market_condition']}_{int(time.time())}",

                'coin': coin,

                'interval': interval,

                'strategy_type': 'dna_fractal_routing',

                'rsi_min': pattern['rsi_range']['min'],

                'rsi_max': pattern['rsi_range']['max'],

                'volume_ratio_min': pattern['volume_ratio']['min'],

                'volume_ratio_max': pattern['volume_ratio']['max'],

                'ma_period': 20 + (i * 3),

                'bb_period': 20,

                'bb_std': 2.0 + (i * 0.1),

                'market_condition': pattern['market_condition'],

                'pattern_confidence': pattern['confidence'],

                'pattern_source': pattern['source'],

                'created_at': datetime.now().isoformat(),

                'is_active': 1

            }

            strategies.append(strategy)

        

        logger.info(f"✅ {coin} {interval}: {len(strategies)}개 DNA/프랙탈 기반 라우팅 전략 생성")

        return strategies

        

    except Exception as e:

        logger.error(f"❌ DNA/프랙탈 기반 라우팅 전략 생성 실패: {e}")

        return []



def create_enhanced_dynamic_routing_strategies(coin: str, interval: str, 

                                             existing_strategies: List[Strategy],

                                             routing_patterns: List[Dict[str, Any]], 

                                             num_strategies: int = 5) -> List[Dict[str, Any]]:

    """고등급 전략 + DNA/프랙탈 패턴 기반 향상된 동적 라우팅 전략 생성"""

    try:

        strategies = []

        

        if not existing_strategies:

            return create_dna_fractal_based_routing_strategies(coin, interval, routing_patterns, num_strategies)

        

        # 기존 전략들의 평균 패턴 계산

        avg_rsi_min = sum(s.rsi_min for s in existing_strategies if s.rsi_min) / len([s for s in existing_strategies if s.rsi_min])

        avg_rsi_max = sum(s.rsi_max for s in existing_strategies if s.rsi_max) / len([s for s in existing_strategies if s.rsi_max])

        avg_volume_min = sum(s.volume_ratio_min for s in existing_strategies if s.volume_ratio_min) / len([s for s in existing_strategies if s.volume_ratio_min])

        

        # DNA/프랙탈 패턴과 기존 전략 패턴을 결합

        for i, pattern in enumerate(routing_patterns[:num_strategies]):

            # 패턴 가중 평균 계산

            weight = pattern['confidence']

            combined_rsi_min = avg_rsi_min * (1 - weight) + pattern['rsi_range']['min'] * weight

            combined_rsi_max = avg_rsi_max * (1 - weight) + pattern['rsi_range']['max'] * weight

            combined_volume_min = avg_volume_min * (1 - weight) + pattern['volume_ratio']['min'] * weight

            

            strategy = {

                'id': f"{coin}_{interval}_enhanced_{pattern['market_condition']}_{int(time.time())}",

                'coin': coin,

                'interval': interval,

                'strategy_type': 'enhanced_routing',

                'rsi_min': max(10, min(90, combined_rsi_min)),

                'rsi_max': max(10, min(90, combined_rsi_max)),

                'volume_ratio_min': max(0.1, combined_volume_min),

                'volume_ratio_max': max(0.2, combined_volume_min + 1.0),

                'ma_period': 20 + (i * 2),

                'bb_period': 20,

                'bb_std': 2.0 + (i * 0.15),

                'market_condition': pattern['market_condition'],

                'pattern_confidence': pattern['confidence'],

                'pattern_source': pattern['source'],

                'enhancement_type': 'dna_fractal_integration',

                'created_at': datetime.now().isoformat(),

                'is_active': 1

            }

            strategies.append(strategy)

        

        logger.info(f"✅ {coin} {interval}: {len(strategies)}개 향상된 동적 라우팅 전략 생성")

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 향상된 동적 라우팅 전략 생성 실패: {e}")

        return []



def save_dynamic_routing_strategies_to_db(strategies: List[Dict[str, Any]], coin: str, interval: str) -> int:

    """동적 라우팅 전략들을 DB에 저장"""

    try:

        if not strategies:

            return 0

        

        saved_count = 0

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            for strategy in strategies:

                try:

                    # 동적 라우팅 전략 저장

                    cursor.execute("""

                        INSERT OR REPLACE INTO strategies (

                            id, coin, interval, strategy_type, strategy_conditions,

                            rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,

                            ma_period, bb_period, bb_std, market_condition,

                            pattern_confidence, pattern_source, enhancement_type,

                            created_at, is_active, quality_grade

                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                    """, (

                        strategy['id'],

                        strategy['coin'],

                        strategy['interval'],

                        strategy['strategy_type'],

                        json.dumps(strategy.get('strategy_conditions', {})),

                        strategy.get('rsi_min', 30),

                        strategy.get('rsi_max', 70),

                        strategy.get('volume_ratio_min', 1.0),

                        strategy.get('volume_ratio_max', 2.0),

                        strategy.get('ma_period', 20),

                        strategy.get('bb_period', 20),

                        strategy.get('bb_std', 2.0),

                        strategy.get('market_condition', 'neutral'),

                        strategy.get('pattern_confidence', 0.5),

                        strategy.get('pattern_source', 'unknown'),

                        strategy.get('enhancement_type', 'standard'),

                        strategy.get('created_at', datetime.now().isoformat()),

                        strategy.get('is_active', 1),

                        'B'  # 기본 등급

                    ))

                    

                    saved_count += 1

                    

                except Exception as e:

                    logger.error(f"❌ 동적 라우팅 전략 저장 실패: {strategy.get('id', 'unknown')} - {e}")

                    continue

            

            conn.commit()

        

        logger.info(f"✅ 동적 라우팅 전략 저장 완료: {saved_count}개")

        return saved_count

        

    except Exception as e:

        logger.error(f"❌ 동적 라우팅 전략 저장 실패: {e}")

        return 0



# =============================================================================

# 🤖 AI 학습용 데이터 수집 함수들

# =============================================================================



def run_dynamic_routing_with_iteration_control(

    coin: str, intervals: List[str], 

    dna_analysis: Dict[str, Any] = None,

    fractal_analysis: Dict[str, Any] = None,

    all_candle_data: Dict[Tuple[str, str], pd.DataFrame] = None

) -> Dict[str, Any]:

    """동적 반복 제어를 사용한 라우팅 실행"""

    try:

        logger.info(f"🔄 {coin} 동적 반복 제어 라우팅 시작")

        

        # 현재 라우팅 품질 점수 계산

        current_routing_quality = calculate_current_routing_quality(coin, intervals)

        

        # 이전 라우팅 품질 점수 조회 (있다면)

        previous_routing_quality = get_previous_routing_quality(coin)

        

        # 최적 반복 횟수 계산

        optimal_iterations = calculate_optimal_iterations(

            current_quality=current_routing_quality,

            previous_quality=previous_routing_quality,

            max_iterations=10,

            quality_threshold=0.85,  # 라우팅은 더 높은 기준

            improvement_threshold=0.03,  # 라우팅은 더 민감한 기준

            min_iterations=1

        )

        

        logger.info(f"🎯 {coin} 최적 라우팅 반복 횟수: {optimal_iterations}회")

        

        # 반복 실행

        total_results = {

            'total_trades': 0,

            'total_profit': 0.0,

            'avg_win_rate': 0.0,

            'routing_strategies_created': 0,

            'iterations_performed': 0,

            'quality_improvement': 0.0,

            'final_quality': current_routing_quality

        }

        

        previous_iteration_quality = current_routing_quality

        

        for iteration in range(optimal_iterations):

            try:

                logger.info(f"🔄 {coin} 라우팅 반복 {iteration + 1}/{optimal_iterations}")

                

                # 라우팅 실행 (기존 함수 호출)

                iteration_results = run_dynamic_routing_by_market_condition(

                    coin, intervals, dna_analysis, fractal_analysis, all_candle_data

                )

                

                # 결과 누적

                total_results['total_trades'] += iteration_results.get('total_trades', 0)

                total_results['total_profit'] += iteration_results.get('total_profit', 0.0)

                total_results['routing_strategies_created'] += iteration_results.get('routing_strategies_created', 0)

                total_results['iterations_performed'] += 1

                

                # 품질 개선도 계산

                current_iteration_quality = calculate_current_routing_quality(coin, intervals)

                quality_improvement = current_iteration_quality - previous_iteration_quality

                total_results['quality_improvement'] += quality_improvement

                total_results['final_quality'] = current_iteration_quality

                

                logger.info(f"📊 라우팅 반복 {iteration + 1} 완료 - 품질: {current_iteration_quality:.3f} (개선: {quality_improvement:+.3f})")

                

                # 조기 종료 조건 확인

                if current_iteration_quality >= 0.85 and quality_improvement < 0.01:

                    logger.info(f"🎯 라우팅 품질 목표 달성 및 개선도 미미 - 조기 종료")

                    break

                

                previous_iteration_quality = current_iteration_quality

                

            except Exception as e:

                logger.error(f"❌ {coin} 라우팅 반복 {iteration + 1} 실패: {e}")

                continue

        

        # 평균 승률 계산

        if total_results['total_trades'] > 0:

            total_results['avg_win_rate'] = total_results['total_profit'] / total_results['total_trades']

        

        logger.info(f"✅ {coin} 동적 반복 라우팅 완료: {total_results['iterations_performed']}회 반복, 최종 품질: {total_results['final_quality']:.3f}")

        return total_results

        

    except Exception as e:

        logger.error(f"❌ {coin} 동적 반복 라우팅 실패: {e}")

        return {'error': str(e)}



def calculate_current_routing_quality(coin: str, intervals: List[str]) -> float:

    """현재 라우팅 품질 점수 계산"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT AVG(sr.total_return) as avg_profit, AVG(sr.win_rate) as avg_win_rate,

                       AVG(sr.sharpe_ratio) as avg_sharpe, COUNT(*) as strategy_count

                FROM strategies cs

                LEFT JOIN simulation_results sr ON cs.id = sr.strategy_id

                WHERE cs.symbol = ? AND cs.interval IN ({})

                AND sr.total_trades > 0

                AND sr.total_return IS NOT NULL

            """.format(','.join(['?' for _ in intervals])), [coin] + intervals)

            

            result = cursor.fetchone()

            if result and result[3] > 0:  # strategy_count > 0

                avg_profit, avg_win_rate, avg_sharpe, strategy_count = result

                

                # 라우팅 품질 점수 계산 (0.0 ~ 1.0)

                profit_score = min(avg_profit / 0.15, 1.0) if avg_profit else 0  # 15% 수익률 = 1.0

                win_rate_score = min(avg_win_rate, 1.0) if avg_win_rate else 0

                sharpe_score = min(avg_sharpe / 2.5, 1.0) if avg_sharpe else 0  # 샤프 2.5 = 1.0

                

                quality_score = (profit_score * 0.5 + win_rate_score * 0.3 + sharpe_score * 0.2)

                return max(0.0, min(1.0, quality_score))

            

        return 0.0

        

    except Exception as e:

        logger.error(f"❌ {coin} 라우팅 품질 점수 계산 실패: {e}")

        return 0.0



def get_previous_routing_quality(coin: str) -> float:

    """이전 라우팅 품질 점수 조회"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT analysis_result FROM routing_quality_history 

                WHERE symbol = ?

                ORDER BY created_at DESC LIMIT 1

            """, (coin,))

            

            result = cursor.fetchone()

            if result and result[0]:

                import json

                data = json.loads(result[0])

                return data.get('quality_score', 0.0)

            

        return None

        

    except Exception as e:

        logger.error(f"❌ {coin} 이전 라우팅 품질 점수 조회 실패: {e}")

        return None


