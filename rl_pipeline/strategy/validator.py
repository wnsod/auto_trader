"""
전략 검증 모듈
전략 재검증 및 등급 관리
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

def revalidate_coin_strategies(coin: str, intervals: List[str]) -> Dict[str, Any]:

    """기존 재검증 함수 (호환성 유지)"""

    try:

        logger.info(f"🔄 {coin} 기존 재검증 시작")

        

        # 동적 재검증 함수 호출

        return revalidate_coin_strategies_dynamic(coin, intervals)

        

    except Exception as e:

        logger.error(f"❌ {coin} 기존 재검증 실패: {e}")

        return {}



def revalidate_coin_strategies_dynamic(coin: str, intervals: List[str], 

                                      dna_analysis: Dict[str, Any] = None,

                                      fractal_analysis: Dict[str, Any] = None,

                                      all_candle_data: Dict[Tuple[str, str], pd.DataFrame] = None) -> bool:

    """🆕 동적 분할 기반 재검증 함수 - 15일 기준으로 장기/단기별 재검증"""

    try:

        logger.info(f"🔄 {coin} 동적 분할 재검증 시작 (DNA/프랙탈 분석 결과 활용)")

        

        # DNA/프랙탈 분석 결과에서 최적 조건 추출

        optimal_conditions = extract_optimal_conditions_from_analysis(dna_analysis, fractal_analysis)

        logger.info(f"📊 {coin} 최적 조건 추출 완료: {optimal_conditions}")

        

        total_revalidated = 0

        total_passed = 0

        grade_updates = 0

        

        for interval in intervals:

            try:

                # 🚀 동적 기간 분할 계산

                from simulation.replay import calculate_dynamic_periods

                periods = calculate_dynamic_periods(coin, interval, all_candle_data)

                

                if not periods['has_data']:

                    logger.warning(f"⚠️ {coin} {interval}: 데이터가 없어 재검증 건너뜀")

                    continue

                

                logger.info(f"🔄 {coin} {interval} 동적 분할 재검증 시작...")

                

                # 🚀 장기 전략 재검증 (15일 이상인 경우만)

                if periods['has_long_term']:

                    logger.info(f"📈 {coin} {interval} 장기 전략 재검증: {periods['long_term_days']:.1f}일")

                    long_term_result = revalidate_long_term_strategies(

                        coin, interval, dna_analysis, fractal_analysis, optimal_conditions

                    )

                    total_revalidated += long_term_result['revalidated']

                    total_passed += long_term_result['passed']

                    grade_updates += long_term_result['grade_updates']

                

                # 🚀 단기 전략 재검증

                if periods['has_short_term']:

                    if periods['has_long_term']:

                        # 전반/후반 분할 재검증

                        logger.info(f"📊 {coin} {interval} 단기 전반 전략 재검증: {periods['short_term_front_days']:.1f}일")

                        short_front_result = revalidate_short_term_front_strategies(

                            coin, interval, dna_analysis, fractal_analysis, optimal_conditions

                        )

                        total_revalidated += short_front_result['revalidated']

                        total_passed += short_front_result['passed']

                        grade_updates += short_front_result['grade_updates']

                        

                        logger.info(f"📊 {coin} {interval} 단기 후반 전략 재검증: {periods['short_term_back_days']:.1f}일")

                        short_back_result = revalidate_short_term_back_strategies(

                            coin, interval, dna_analysis, fractal_analysis, optimal_conditions

                        )

                        total_revalidated += short_back_result['revalidated']

                        total_passed += short_back_result['passed']

                        grade_updates += short_back_result['grade_updates']

                    else:

                        # 단기만 재검증

                        logger.info(f"📊 {coin} {interval} 단기만 전략 재검증: {periods['short_term_only_days']:.1f}일")

                        short_only_result = revalidate_short_term_only_strategies(

                            coin, interval, dna_analysis, fractal_analysis, optimal_conditions

                        )

                        total_revalidated += short_only_result['revalidated']

                        total_passed += short_only_result['passed']

                        grade_updates += short_only_result['grade_updates']

                

                logger.info(f"✅ {coin} {interval}: 총 {total_revalidated}개 재검증, {total_passed}개 통과, {grade_updates}개 등급 변경")

                

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 재검증 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} 동적 분할 재검증 완료: 총 {total_revalidated}개 재검증, {total_passed}개 통과, {grade_updates}개 등급 변경")

        return True

        

    except Exception as e:

        logger.error(f"❌ {coin} 동적 분할 재검증 실패: {e}")

        return False



def revalidate_long_term_strategies(coin: str, interval: str, dna_analysis: Dict[str, Any], 

                                   fractal_analysis: Dict[str, Any], optimal_conditions: Dict[str, Any]) -> Dict[str, int]:

    """장기 전략 재검증 - 안정성 중심"""

    try:

        logger.info(f"📈 {coin} {interval} 장기 전략 재검증 시작 (안정성 중심)")

        

        # 장기 전략 조회 (실제 시뮬레이션 결과와 함께)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT cs.id, cs.strategy_conditions, cs.profit, cs.trades_count, cs.win_rate, 

                       cs.rsi_min, cs.rsi_max, cs.volume_ratio_min, cs.volume_ratio_max, 

                       cs.quality_grade, cs.complexity_score, cs.score,

                       sr.total_trades, sr.win_rate as sr_win_rate, sr.total_return, sr.max_drawdown, sr.sharpe_ratio

                FROM coin_strategies cs

                LEFT JOIN simulation_results sr ON cs.id = sr.strategy_id

                WHERE cs.coin = ? AND cs.interval = ?

                AND sr.total_trades > 0

                AND sr.total_return IS NOT NULL

                ORDER BY sr.total_return DESC

                LIMIT 200

            """, (coin, interval))

            

            strategies = cursor.fetchall()

            

        revalidated = 0

        passed = 0

        grade_updates = 0

        

        for strategy in strategies:

            try:

                revalidated += 1

                

                # 실제 시뮬레이션 결과 기반 장기 전략 재검증 기준 (안정성 중심)

                strategy_id = strategy[0]

                profit = strategy[14] or 0  # sr.total_return

                trades_count = strategy[12] or 0  # sr.total_trades

                win_rate = strategy[13] or 0  # sr.win_rate

                max_drawdown = strategy[15] or 0  # sr.max_drawdown

                sharpe_ratio = strategy[16] or 0  # sr.sharpe_ratio

                complexity_score = strategy[10] or 0

                current_grade = strategy[9] or 'C'

                

                # 장기 전략 기준: 학습 데이터 확보를 위해 완화된 기준

                is_stable = (max_drawdown < 0.25 and sharpe_ratio > 0.2) if max_drawdown and sharpe_ratio else True

                has_sufficient_trades = trades_count > 3  # 학습을 위해 낮춤

                has_good_performance = profit > 0.001 and win_rate > 0.35  # 학습을 위해 낮춤

                is_not_too_complex = complexity_score < 0.9  # 학습을 위해 완화

                

                if is_stable and has_sufficient_trades and has_good_performance and is_not_too_complex:

                    passed += 1

                    # 등급 상향 조정

                    new_grade = 'B' if current_grade == 'C' else 'A' if current_grade == 'B' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"장기 전략 {strategy_id} 등급 상향: {current_grade} → {new_grade}")

                else:

                    # 등급 하향 조정

                    new_grade = 'C' if current_grade == 'B' else 'D' if current_grade == 'C' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"장기 전략 {strategy_id} 등급 하향: {current_grade} → {new_grade}")

                        

            except Exception as e:

                logger.error(f"장기 전략 {strategy_id} 재검증 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 장기 전략 재검증 완료: {revalidated}개 재검증, {passed}개 통과, {grade_updates}개 등급 변경")

        return {'revalidated': revalidated, 'passed': passed, 'grade_updates': grade_updates}

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 장기 전략 재검증 실패: {e}")

        return {'revalidated': 0, 'passed': 0, 'grade_updates': 0}



def revalidate_short_term_front_strategies(coin: str, interval: str, dna_analysis: Dict[str, Any], 

                                         fractal_analysis: Dict[str, Any], optimal_conditions: Dict[str, Any]) -> Dict[str, int]:

    """단기 전반 전략 재검증 - 민감성 중심"""

    try:

        logger.info(f"📊 {coin} {interval} 단기 전반 전략 재검증 시작 (민감성 중심)")

        

        # 단기 전반 전략 조회 (실제 시뮬레이션 결과와 함께)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT cs.id, cs.strategy_conditions, cs.profit, cs.trades_count, cs.win_rate, 

                       cs.rsi_min, cs.rsi_max, cs.volume_ratio_min, cs.volume_ratio_max, 

                       cs.quality_grade, cs.complexity_score, cs.score,

                       sr.total_trades, sr.win_rate as sr_win_rate, sr.total_return, sr.max_drawdown, sr.sharpe_ratio

                FROM coin_strategies cs

                LEFT JOIN simulation_results sr ON cs.id = sr.strategy_id

                WHERE cs.coin = ? AND cs.interval = ?

                AND sr.total_trades > 0

                AND sr.total_return IS NOT NULL

                ORDER BY sr.total_return DESC

                LIMIT 100

            """, (coin, interval))

            

            strategies = cursor.fetchall()

            

        revalidated = 0

        passed = 0

        grade_updates = 0

        

        for strategy in strategies:

            try:

                revalidated += 1

                

                # 실제 시뮬레이션 결과 기반 단기 전략 재검증 기준 (민감성 중심)

                strategy_id = strategy[0]

                profit = strategy[14] or 0  # sr.total_return

                trades_count = strategy[12] or 0  # sr.total_trades

                win_rate = strategy[13] or 0  # sr.win_rate

                max_drawdown = strategy[15] or 0  # sr.max_drawdown

                sharpe_ratio = strategy[16] or 0  # sr.sharpe_ratio

                complexity_score = strategy[10] or 0

                current_grade = strategy[9] or 'C'

                

                # 단기 전략 기준: 학습 데이터 확보를 위해 완화된 기준

                has_high_sensitivity = complexity_score > 0.2  # 학습을 위해 낮춤

                has_sufficient_trades = trades_count > 2  # 학습을 위해 낮춤

                has_good_performance = profit > 0.001 and win_rate > 0.3  # 학습을 위해 낮춤

                is_responsive = max_drawdown < 0.3 if max_drawdown else True  # 학습을 위해 완화

                

                if has_high_sensitivity and has_sufficient_trades and has_good_performance and is_responsive:

                    passed += 1

                    # 등급 상향 조정

                    new_grade = 'B' if current_grade == 'C' else 'A' if current_grade == 'B' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"단기 전반 전략 {strategy_id} 등급 상향: {current_grade} → {new_grade}")

                else:

                    # 등급 하향 조정

                    new_grade = 'C' if current_grade == 'B' else 'D' if current_grade == 'C' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"단기 전반 전략 {strategy_id} 등급 하향: {current_grade} → {new_grade}")

                        

            except Exception as e:

                logger.error(f"단기 전반 전략 {strategy_id} 재검증 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 단기 전반 전략 재검증 완료: {revalidated}개 재검증, {passed}개 통과, {grade_updates}개 등급 변경")

        return {'revalidated': revalidated, 'passed': passed, 'grade_updates': grade_updates}

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 단기 전반 전략 재검증 실패: {e}")

        return {'revalidated': 0, 'passed': 0, 'grade_updates': 0}



def revalidate_short_term_back_strategies(coin: str, interval: str, dna_analysis: Dict[str, Any], 

                                        fractal_analysis: Dict[str, Any], optimal_conditions: Dict[str, Any]) -> Dict[str, int]:

    """단기 후반 전략 재검증 - 민감성 중심"""

    try:

        logger.info(f"📊 {coin} {interval} 단기 후반 전략 재검증 시작 (민감성 중심)")

        

        # 단기 후반 전략 조회 (실제 시뮬레이션 결과와 함께)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT cs.id, cs.strategy_conditions, cs.profit, cs.trades_count, cs.win_rate,

                       cs.rsi_min, cs.rsi_max, cs.volume_ratio_min, cs.volume_ratio_max,

                       cs.quality_grade, cs.complexity_score, cs.score,

                       sr.total_trades, sr.win_rate as sr_win_rate, sr.total_return, sr.max_drawdown, sr.sharpe_ratio

                FROM coin_strategies cs

                LEFT JOIN simulation_results sr ON cs.id = sr.strategy_id

                WHERE cs.coin = ? AND cs.interval = ?

                AND sr.total_trades > 0

                AND sr.total_return IS NOT NULL

                ORDER BY sr.total_return DESC

                LIMIT 30

            """, (coin, interval))

            

            strategies = cursor.fetchall()

            

        revalidated = 0

        passed = 0

        grade_updates = 0

        

        for strategy in strategies:

            try:

                revalidated += 1

                

                # 실제 시뮬레이션 결과 기반 단기 후반 전략 재검증 기준 (민감성 중심)

                strategy_id = strategy[0]

                profit = strategy[14] or 0  # sr.total_return

                trades_count = strategy[12] or 0  # sr.total_trades

                win_rate = strategy[13] or 0  # sr.win_rate

                max_drawdown = strategy[15] or 0  # sr.max_drawdown

                sharpe_ratio = strategy[16] or 0  # sr.sharpe_ratio

                complexity_score = strategy[10] or 0

                current_grade = strategy[9] or 'C'

                

                # 단기 후반 전략 기준: 학습 데이터 확보를 위해 완화된 기준

                has_high_sensitivity = complexity_score > 0.2  # 학습을 위해 낮춤

                has_sufficient_trades = trades_count > 2  # 학습을 위해 낮춤

                has_good_performance = profit > 0.001 and win_rate > 0.3  # 학습을 위해 낮춤

                

                if has_high_sensitivity and has_sufficient_trades and has_good_performance:

                    passed += 1

                    # 등급 상향 조정

                    new_grade = 'B' if current_grade == 'C' else 'A' if current_grade == 'B' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"단기 후반 전략 {strategy_id} 등급 상향: {current_grade} → {new_grade}")

                else:

                    # 등급 하향 조정

                    new_grade = 'C' if current_grade == 'B' else 'D' if current_grade == 'C' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"단기 후반 전략 {strategy_id} 등급 하향: {current_grade} → {new_grade}")

                        

            except Exception as e:

                logger.error(f"단기 후반 전략 {strategy[0]} 재검증 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 단기 후반 전략 재검증 완료: {revalidated}개 재검증, {passed}개 통과, {grade_updates}개 등급 변경")

        return {'revalidated': revalidated, 'passed': passed, 'grade_updates': grade_updates}

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 단기 후반 전략 재검증 실패: {e}")

        return {'revalidated': 0, 'passed': 0, 'grade_updates': 0}



def revalidate_short_term_only_strategies(coin: str, interval: str, dna_analysis: Dict[str, Any], 

                                       fractal_analysis: Dict[str, Any], optimal_conditions: Dict[str, Any]) -> Dict[str, int]:

    """단기만 전략 재검증 - 민감성 중심"""

    try:

        logger.info(f"📊 {coin} {interval} 단기만 전략 재검증 시작 (민감성 중심)")

        

        # 단기만 전략 조회 (실제 시뮬레이션 결과와 함께)

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT cs.id, cs.strategy_conditions, cs.profit, cs.trades_count, cs.win_rate,

                       cs.rsi_min, cs.rsi_max, cs.volume_ratio_min, cs.volume_ratio_max,

                       cs.quality_grade, cs.complexity_score, cs.score,

                       sr.total_trades, sr.win_rate as sr_win_rate, sr.total_return, sr.max_drawdown, sr.sharpe_ratio

                FROM coin_strategies cs

                LEFT JOIN simulation_results sr ON cs.id = sr.strategy_id

                WHERE cs.coin = ? AND cs.interval = ?

                AND sr.total_trades > 0

                AND sr.total_return IS NOT NULL

                ORDER BY sr.total_return DESC

                LIMIT 50

            """, (coin, interval))

            

            strategies = cursor.fetchall()

            

        revalidated = 0

        passed = 0

        grade_updates = 0

        

        for strategy in strategies:

            try:

                revalidated += 1

                

                # 실제 시뮬레이션 결과 기반 단기만 전략 재검증 기준 (민감성 중심)

                strategy_id = strategy[0]

                profit = strategy[14] or 0  # sr.total_return

                trades_count = strategy[12] or 0  # sr.total_trades

                win_rate = strategy[13] or 0  # sr.win_rate

                max_drawdown = strategy[15] or 0  # sr.max_drawdown

                sharpe_ratio = strategy[16] or 0  # sr.sharpe_ratio

                complexity_score = strategy[10] or 0

                current_grade = strategy[9] or 'C'

                

                # 단기만 전략 기준: 높은 민감성, 빠른 반응, 높은 거래 빈도

                has_high_sensitivity = complexity_score > 0.3  # 단기는 더 민감해야 함

                has_sufficient_trades = trades_count > 5  # 단기는 더 적은 거래로도 OK

                has_good_performance = profit > 0.005 and win_rate > 0.4  # 단기는 더 낮은 기준

                

                if has_high_sensitivity and has_sufficient_trades and has_good_performance:

                    passed += 1

                    # 등급 상향 조정

                    new_grade = 'B' if current_grade == 'C' else 'A' if current_grade == 'B' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"단기만 전략 {strategy_id} 등급 상향: {current_grade} → {new_grade}")

                else:

                    # 등급 하향 조정

                    new_grade = 'C' if current_grade == 'B' else 'D' if current_grade == 'C' else current_grade

                    if new_grade != current_grade:

                        update_strategy_grade(strategy_id, new_grade)

                        grade_updates += 1

                        logger.debug(f"단기만 전략 {strategy_id} 등급 하향: {current_grade} → {new_grade}")

                        

            except Exception as e:

                logger.error(f"단기만 전략 {strategy[0]} 재검증 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 단기만 전략 재검증 완료: {revalidated}개 재검증, {passed}개 통과, {grade_updates}개 등급 변경")

        return {'revalidated': revalidated, 'passed': passed, 'grade_updates': grade_updates}

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 단기만 전략 재검증 실패: {e}")

        return {'revalidated': 0, 'passed': 0, 'grade_updates': 0}

    """🆕 코인별 전략 재검증 함수 - DNA/프랙탈 분석 결과를 활용한 고도화된 재검증"""

    try:

        logger.info(f"🔄 {coin} 전략 재검증 시작 (DNA/프랙탈 분석 결과 활용)")

        

        # DNA/프랙탈 분석 결과에서 최적 조건 추출

        optimal_conditions = extract_optimal_conditions_from_analysis(dna_analysis, fractal_analysis)

        logger.info(f"📊 {coin} 최적 조건 추출 완료: {optimal_conditions}")

        

        total_revalidated = 0

        total_passed = 0

        grade_updates = 0

        

        # 🚀 해당 코인의 모든 기존 전략들에 대해 실제 재검증

        for interval in intervals:

            try:

                logger.info(f"🔄 {coin} {interval} 전략 재검증 시작...")

                

                # 1. 해당 코인-인터벌의 전략들을 조회

                with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

                    cursor = conn.cursor()

                    

                    # 해당 코인-인터벌의 전략 조회 (최근 100개)

                    try:

                        cursor.execute("""

                            SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                                   rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, 

                                   ma_period, bb_period, bb_std

                            FROM coin_strategies 

                            WHERE coin = ? AND interval = ?

                            ORDER BY created_at DESC

                            LIMIT 100

                        """, (coin, interval))

                    except sqlite3.OperationalError as e:

                        if any(col in str(e) for col in ["volume_ratio_min", "ma_period", "bb_period", "bb_std"]):

                            # 컬럼이 없는 경우 기본값으로 대체

                            cursor.execute("""

                                SELECT id, strategy_conditions, profit, trades_count, win_rate, 

                                       rsi_min, rsi_max, 1.0 as volume_ratio_min, 3.0 as volume_ratio_max, 

                                       20.0 as ma_period, 20.0 as bb_period, 2.0 as bb_std

                                FROM coin_strategies 

                                WHERE coin = ? AND interval = ?

                                ORDER BY created_at DESC

                                LIMIT 100

                            """, (coin, interval))

                        else:

                            raise e

                    

                    strategies = cursor.fetchall()

                    

                    if not strategies:

                        logger.warning(f"⚠️ {coin} {interval}: 재검증할 전략이 없음 - 분석할 데이터 부족 (기능적 실패 아님)")

                        logger.info(f"📊 {coin} {interval} 재검증 대상 전략 조회 결과: 0개 (DB에서 조회됨)")

                        continue

                    

                    logger.info(f"🔍 {coin} {interval}: {len(strategies)}개 전략 재검증 시작")

                    

                    # 2. 각 전략에 대해 재검증 수행

                    for strategy_row in strategies:

                        try:

                            id = strategy_row[0]

                            strategy_conditions = strategy_row[1]

                            profit = strategy_row[2] or 0

                            trades_count = strategy_row[3] or 0

                            win_rate = strategy_row[4] or 0

                            

                            # 기본값 설정

                            rsi_min = strategy_row[5] or 30

                            rsi_max = strategy_row[6] or 70

                            volume_ratio_min = strategy_row[7] or 1.0

                            volume_ratio_max = strategy_row[8] or 3.0

                            ma_period = strategy_row[9] or 20

                            bb_period = strategy_row[10] or 20

                            bb_std = strategy_row[11] or 2.0

                            

                            # 🚀 고도화된 재검증 로직

                            validation_result = perform_enhanced_strategy_validation(

                                id, profit, trades_count, win_rate,

                                rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,

                                ma_period, bb_period, bb_std, optimal_conditions

                            )

                            

                            if validation_result['passed']:

                                total_passed += 1

                                logger.debug(f"✅ {id}: 재검증 통과")

                            else:

                                logger.debug(f"❌ {id}: 재검증 실패 - {validation_result['reason']}")

                            

                            # 등급 업데이트

                            if validation_result['grade_updated']:

                                grade_updates += 1

                                update_strategy_grade(id, validation_result['new_grade'])

                            

                            total_revalidated += 1

                            

                        except Exception as e:

                            logger.error(f"❌ {id} 재검증 실패: {e}")

                            continue

                

                logger.info(f"✅ {coin} {interval}: {total_revalidated}개 전략 재검증 완료 (통과: {total_passed}, 등급업데이트: {grade_updates})")

                

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 재검증 실패: {e}")

                continue

        

        # 결과 요약

        success_rate = (total_passed / total_revalidated * 100) if total_revalidated > 0 else 0

        logger.info(f"🎉 {coin} 전략 재검증 완료: {total_revalidated}개 검증, {total_passed}개 통과 ({success_rate:.1f}%), {grade_updates}개 등급업데이트")

        

        return total_revalidated > 0

        

    except Exception as e:

        logger.error(f"❌ {coin} 전략 재검증 실패: {e}")

        return False



def perform_enhanced_strategy_validation(id: str, profit: float, trades_count: int, win_rate: float,

                                        rsi_min: float, rsi_max: float, volume_ratio_min: float, volume_ratio_max: float,

                                        ma_period: float, bb_period: float, bb_std: float, optimal_conditions: Dict[str, Any]) -> Dict[str, Any]:

    """고도화된 전략 검증 수행"""

    try:

        # 기본 검증 기준

        passed = True

        reason = ""

        grade_updated = False

        new_grade = "C"

        

        # 수익성 검증 (학습 데이터 확보를 위해 완화)

        if profit < optimal_conditions.get('profit_threshold', -0.01):  # 손실 허용 범위 확대

            passed = False

            reason += f"수익 부족 ({profit:.2f} < {optimal_conditions.get('profit_threshold', -0.01)}) "

        

        # 거래 횟수 검증 (학습 데이터 확보를 위해 완화)

        if trades_count < optimal_conditions.get('trades_threshold', 1):  # 최소 1회 거래

            passed = False

            reason += f"거래 횟수 부족 ({trades_count} < {optimal_conditions.get('trades_threshold', 1)}) "

        

        # 승률 검증 (학습 데이터 확보를 위해 완화)

        if win_rate < optimal_conditions.get('win_rate_threshold', 0.25):  # 25% 승률로 낮춤

            passed = False

            reason += f"승률 부족 ({win_rate:.2f} < {optimal_conditions.get('win_rate_threshold', 0.25)}) "

        

        # RSI 범위 검증

        rsi_range = optimal_conditions.get('rsi_range', {'min': 30, 'max': 70})

        if rsi_min < rsi_range['min'] or rsi_max > rsi_range['max']:

            passed = False

            reason += f"RSI 범위 초과 ({rsi_min}-{rsi_max} vs {rsi_range['min']}-{rsi_range['max']}) "

        

        # Volume 비율 검증

        volume_range = optimal_conditions.get('volume_ratio', {'min': 1.0, 'max': 2.0})

        if volume_ratio_min < volume_range['min'] or volume_ratio_max > volume_range['max']:

            passed = False

            reason += f"Volume 비율 범위 초과 ({volume_ratio_min}-{volume_ratio_max} vs {volume_range['min']}-{volume_range['max']}) "

        

        # 등급 결정

        if passed:

            if profit > 0.1 and win_rate > 0.6:

                new_grade = "A"

            elif profit > 0.05 and win_rate > 0.5:

                new_grade = "B"

            else:

                new_grade = "C"

            grade_updated = True

        

        return {

            'passed': passed,

            'reason': reason.strip(),

            'grade_updated': grade_updated,

            'new_grade': new_grade

        }

        

    except Exception as e:

        logger.error(f"❌ 전략 검증 실패: {e}")

        return {

            'passed': False,

            'reason': f"검증 오류: {e}",

            'grade_updated': False,

            'new_grade': "F"

        }



def update_strategy_grade(id: str, new_grade: str) -> bool:

    """전략 등급 업데이트"""

    try:

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                UPDATE coin_strategies 

                SET quality_grade = ?, updated_at = datetime('now')

                WHERE id = ?

            """, (new_grade, id))

            conn.commit()

            logger.debug(f"✅ 전략 등급 업데이트: {id} -> {new_grade}")

            return True

    except Exception as e:

        logger.error(f"❌ 전략 등급 업데이트 실패: {e}")

        return False



def load_high_grade_strategies(coin: str, interval: str, num_strategies: int = 5) -> List[Strategy]:

    """고등급 전략만 로드"""

    try:

        from rl_pipeline.core.types import Strategy

        

        strategies = []

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            # 고등급 전략만 조회 (A, A+, B+ 등급)

            try:

                cursor.execute("""

                    SELECT id, coin, interval, strategy_type, strategy_conditions,

                           rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,

                           ma_period, bb_period, bb_std, profit, trades_count, win_rate,

                           quality_grade

                    FROM coin_strategies 

                    WHERE coin = ? AND interval = ? AND is_active = 1 

                    AND quality_grade IN ('A', 'A+', 'B+')

                    ORDER BY profit DESC

                    LIMIT ?

                """, (coin, interval, num_strategies))

            except sqlite3.OperationalError as e:

                if any(col in str(e) for col in ["volume_ratio_min", "ma_period", "bb_period", "bb_std"]):

                    # 컬럼이 없는 경우 기본값으로 대체

                    cursor.execute("""

                        SELECT id, coin, interval, strategy_type, strategy_conditions,

                               rsi_min, rsi_max, 1.0 as volume_ratio_min, 3.0 as volume_ratio_max,

                               20.0 as ma_period, 20.0 as bb_period, 2.0 as bb_std, profit, trades_count, win_rate,

                               quality_grade

                        FROM coin_strategies 

                        WHERE coin = ? AND interval = ? AND is_active = 1 

                        AND quality_grade IN ('A', 'A+', 'B+')

                        ORDER BY profit DESC

                        LIMIT ?

                    """, (coin, interval, num_strategies))

                else:

                    raise e

            

            rows = cursor.fetchall()

            

            for row in rows:

                try:

                    id, coin_name, interval_name, strategy_type, strategy_conditions, \

                    rsi_min, rsi_max, volume_ratio_min, volume_ratio_max, ma_period, bb_period, bb_std, \

                    profit, trades_count, win_rate, quality_grade = row

                    

                    # Strategy 객체 생성

                    strategy = Strategy(

                        id=id,

                        params={},

                        version="v2.0",

                        coin=coin_name,

                        interval=interval_name,

                        created_at=datetime.now(),

                        strategy_type=strategy_type,

                        rsi_min=rsi_min,

                        rsi_max=rsi_max,

                        volume_ratio_min=volume_ratio_min,

                        volume_ratio_max=volume_ratio_max,

                        ma_period=ma_period,

                        bb_period=bb_period,

                        bb_std=bb_std

                    )

                    

                    strategies.append(strategy)

                    

                except Exception as e:

                    logger.error(f"❌ 전략 로드 실패: {e}")

                    continue

        

        logger.debug(f"✅ 고등급 전략 로드 완료: {len(strategies)}개")

        return strategies

        

    except Exception as e:

        logger.error(f"❌ 고등급 전략 로드 실패: {e}")

        return []



def calculate_current_strategy_quality(coin: str, interval: str) -> float:

    """현재 전략 품질 점수 계산"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT AVG(sr.total_return) as avg_profit, AVG(sr.win_rate) as avg_win_rate,

                       AVG(sr.sharpe_ratio) as avg_sharpe, COUNT(*) as strategy_count

                FROM coin_strategies cs

                LEFT JOIN simulation_results sr ON cs.id = sr.strategy_id

                WHERE cs.coin = ? AND cs.interval = ?

                AND sr.total_trades > 0

                AND sr.total_return IS NOT NULL

            """, (coin, interval))

            

            result = cursor.fetchone()

            if result and result[3] > 0:  # strategy_count > 0

                avg_profit, avg_win_rate, avg_sharpe, strategy_count = result

                

                # 품질 점수 계산 (0.0 ~ 1.0)

                profit_score = min(avg_profit / 0.1, 1.0) if avg_profit else 0  # 10% 수익률 = 1.0

                win_rate_score = min(avg_win_rate, 1.0) if avg_win_rate else 0

                sharpe_score = min(avg_sharpe / 2.0, 1.0) if avg_sharpe else 0  # 샤프 2.0 = 1.0

                

                quality_score = (profit_score * 0.4 + win_rate_score * 0.4 + sharpe_score * 0.2)

                return max(0.0, min(1.0, quality_score))

            

        return 0.0

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 품질 점수 계산 실패: {e}")

        return 0.0



def get_previous_strategy_quality(coin: str, interval: str) -> float:

    """이전 전략 품질 점수 조회"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT analysis_result FROM strategy_quality_history 

                WHERE coin = ? AND interval = ?

                ORDER BY created_at DESC LIMIT 1

            """, (coin, interval))

            

            result = cursor.fetchone()

            if result and result[0]:

                import json

                data = json.loads(result[0])

                return data.get('quality_score', 0.0)

            

        return None

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 이전 품질 점수 조회 실패: {e}")

        return None



def revalidate_with_dynamic_iteration(

    coin: str, interval: str, dna_analysis: Dict[str, Any], 

    fractal_analysis: Dict[str, Any], optimal_conditions: Dict[str, Any]

) -> Dict[str, Any]:

    """동적 반복 제어를 사용한 재검증"""

    try:

        logger.info(f"🔄 {coin} {interval} 동적 반복 제어 재검증 시작")

        

        # 현재 품질 점수 계산

        current_quality = calculate_current_strategy_quality(coin, interval)

        

        # 이전 품질 점수 조회 (있다면)

        previous_quality = get_previous_strategy_quality(coin, interval)

        

        # 최적 반복 횟수 계산

        optimal_iterations = calculate_optimal_iterations(

            current_quality=current_quality,

            previous_quality=previous_quality,

            max_iterations=10,

            quality_threshold=0.8,

            improvement_threshold=0.05,

            min_iterations=1

        )

        

        logger.info(f"🎯 {coin} {interval} 최적 반복 횟수: {optimal_iterations}회")

        

        # 반복 실행

        total_results = {

            'total_revalidated': 0,

            'total_passed': 0,

            'total_grade_updates': 0,

            'iterations_performed': 0,

            'quality_improvement': 0.0,

            'final_quality': current_quality

        }

        

        previous_iteration_quality = current_quality

        

        for iteration in range(optimal_iterations):

            try:

                logger.info(f"🔄 {coin} {interval} 재검증 반복 {iteration + 1}/{optimal_iterations}")

                

                # 재검증 실행 (기존 함수 호출)

                iteration_results = revalidate_coin_strategies_dynamic_single(

                    coin, interval, dna_analysis, fractal_analysis, optimal_conditions

                )

                

                # 결과 누적

                total_results['total_revalidated'] += iteration_results.get('total_revalidated', 0)

                total_results['total_passed'] += iteration_results.get('total_passed', 0)

                total_results['total_grade_updates'] += iteration_results.get('total_grade_updates', 0)

                total_results['iterations_performed'] += 1

                

                # 품질 개선도 계산

                current_iteration_quality = calculate_current_strategy_quality(coin, interval)

                quality_improvement = current_iteration_quality - previous_iteration_quality

                total_results['quality_improvement'] += quality_improvement

                total_results['final_quality'] = current_iteration_quality

                

                logger.info(f"📊 반복 {iteration + 1} 완료 - 품질: {current_iteration_quality:.3f} (개선: {quality_improvement:+.3f})")

                

                # 조기 종료 조건 확인

                if current_iteration_quality >= 0.8 and quality_improvement < 0.01:

                    logger.info(f"🎯 품질 목표 달성 및 개선도 미미 - 조기 종료")

                    break

                

                previous_iteration_quality = current_iteration_quality

                

            except Exception as e:

                logger.error(f"❌ {coin} {interval} 반복 {iteration + 1} 실패: {e}")

                continue

        

        logger.info(f"✅ {coin} {interval} 동적 반복 재검증 완료: {total_results['iterations_performed']}회 반복, 최종 품질: {total_results['final_quality']:.3f}")

        return total_results

        

    except Exception as e:

        logger.error(f"❌ {coin} {interval} 동적 반복 재검증 실패: {e}")

        return {'error': str(e)}


