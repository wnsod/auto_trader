"""
전략 생성 오케스트레이터
캔들/지표 호출 → 샘플링 → 검증 → 저장 요청의 전체 흐름 관리
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


# 분리된 모듈 imports
from rl_pipeline.strategy.router import (
    run_dynamic_routing_by_market_condition,
    run_coin_dynamic_routing,
    run_coin_dynamic_routing_integrated,
    save_dynamic_routing_strategies_to_db,
    run_dynamic_routing_with_iteration_control,
    calculate_current_routing_quality,
    get_previous_routing_quality,
)
from rl_pipeline.strategy.creator import (
    create_intelligent_strategies_with_type,
    create_intelligent_strategies,
    create_coin_strategies_dynamic,
    create_coin_strategies,
    classify_market_condition,
    create_enhanced_market_adaptive_strategy,
    create_guided_random_strategy,
    create_basic_strategy,
    create_global_strategies,
    create_global_strategies_from_results,
)
from rl_pipeline.strategy.validator import (
    revalidate_coin_strategies,
    revalidate_coin_strategies_dynamic,
    revalidate_with_dynamic_iteration,
    perform_enhanced_strategy_validation,
    update_strategy_grade,
    load_high_grade_strategies,
)
from rl_pipeline.strategy.analyzer import (
    extract_optimal_conditions_from_analysis,
    extract_routing_patterns_from_analysis,
)
from rl_pipeline.strategy.ai_collector import (
    collect_strategy_performance_for_ai,
    collect_strategy_comparison_for_ai,
    collect_learning_episode_for_ai,
    collect_learning_state_for_ai,
    collect_learning_action_for_ai,
    collect_learning_reward_for_ai,
    collect_model_training_data_for_ai,
)

class StrategyManager:
    """전략 생성 오케스트레이터"""
    
    def __init__(self):
        self.default_n_strategies = config.STRATEGIES_PER_COMBINATION
        self.default_sampling_method = "random"
    
    def create_default_strategies(self, coin: str, interval: str) -> List[Dict[str, Any]]:
        """기본 전략 생성 메서드
        
        Args:
            coin: 코인 심볼
            interval: 시간 간격
            
        Returns:
            기본 전략 리스트
        """
        try:
            logger.info(f"📊 기본 전략 생성: {coin} {interval}")
            
            # 기본 전략들 생성
            strategies = [
                {
                    'id': f"{coin}_{interval}_rsi_momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'coin': coin,
                    'interval': interval,
                    'strategy_type': 'rsi_momentum',
                    'params': {
                        'rsi_min': 30,
                        'rsi_max': 70,
                        'volume_ratio_min': 1.0,
                        'volume_ratio_max': 2.0
                    },
                    'name': f'RSI Momentum Strategy for {coin} {interval}',
                    'description': 'RSI 기반 모멘텀 전략'
                },
                {
                    'id': f"{coin}_{interval}_macd_crossover_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'coin': coin,
                    'interval': interval,
                    'strategy_type': 'macd_crossover',
                    'params': {
                        'macd_threshold': 0.001,
                        'volume_ratio_min': 0.8,
                        'volume_ratio_max': 2.5
                    },
                    'name': f'MACD Crossover Strategy for {coin} {interval}',
                    'description': 'MACD 크로스오버 전략'
                },
                {
                    'id': f"{coin}_{interval}_bb_mean_reversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'coin': coin,
                    'interval': interval,
                    'strategy_type': 'bb_mean_reversion',
                    'params': {
                        'bb_threshold': 0.02,
                        'volume_ratio_min': 1.2,
                        'volume_ratio_max': 1.8
                    },
                    'name': f'Bollinger Bands Mean Reversion for {coin} {interval}',
                    'description': '볼린저 밴드 평균 회귀 전략'
                },
                {
                    'id': f"{coin}_{interval}_volume_breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'coin': coin,
                    'interval': interval,
                    'strategy_type': 'volume_breakout',
                    'params': {
                        'volume_ratio_min': 2.0,
                        'volume_ratio_max': 5.0,
                        'rsi_min': 40,
                        'rsi_max': 60
                    },
                    'name': f'Volume Breakout Strategy for {coin} {interval}',
                    'description': '볼륨 브레이크아웃 전략'
                },
                {
                    'id': f"{coin}_{interval}_trend_following_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'coin': coin,
                    'interval': interval,
                    'strategy_type': 'trend_following',
                    'params': {
                        'macd_threshold': 0.002,
                        'rsi_min': 50,
                        'rsi_max': 80,
                        'volume_ratio_min': 1.5
                    },
                    'name': f'Trend Following Strategy for {coin} {interval}',
                    'description': '트렌드 추종 전략'
                }
            ]
            
            logger.info(f"✅ 기본 전략 생성 완료: {len(strategies)}개")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ 기본 전략 생성 실패: {e}")
            return []
    
    def save_strategies_to_db_dict(self, strategies: List[Dict[str, Any]]) -> int:
        """⚠️ Deprecated: write_batch()를 직접 사용하세요"""
        logger.warning("⚠️ save_strategies_to_db_dict()는 deprecated입니다. write_batch()를 직접 사용하세요.")
        return self._save_strategies_expanded(strategies)
    
    def _save_strategies_expanded(self, strategies: List[Dict[str, Any]]) -> int:
        """전략을 데이터베이스에 저장
        
        Args:
            strategies: 저장할 전략 리스트
            
        Returns:
            저장된 전략 수
        """
        try:
            if not strategies:
                logger.warning("⚠️ 저장할 전략이 없습니다")
                return 0
            
            logger.info(f"💾 전략 저장 시작: {len(strategies)}개")
            
            # 데이터베이스 연결 (절대 경로 사용)
            import sqlite3
            import os
            
            # 설정에서 경로 가져오기
            from rl_pipeline.core.env import config
            db_path = config.STRATEGIES_DB
            
            # 디렉터리가 없으면 생성
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            logger.info(f"📁 DB 경로: {db_path}")
            
            # 확장 스키마 사용을 위해 write_batch 사용
            from rl_pipeline.db.writes import write_batch
            from rl_pipeline.db.schema import create_coin_strategies_table
            
            # 테이블 생성 (없으면)
            create_coin_strategies_table()
            
            # dict를 확장 스키마로 변환
            expanded_strategies = []
            for strategy in strategies:
                try:
                    # params 추출
                    params = strategy.get('params', {})
                    if isinstance(params, str):
                        params = json.loads(params)
                    
                    # 확장 스키마로 변환
                    # MACD threshold: None이면 0.0으로 변환 (NULL 방지)
                    macd_buy = params.get('macd_buy_threshold')
                    macd_sell = params.get('macd_sell_threshold')
                    
                    expanded = {
                        'id': strategy.get('id', ''),
                        'coin': strategy.get('coin', ''),
                        'interval': strategy.get('interval', ''),
                        'strategy_type': strategy.get('strategy_type', 'hybrid'),
                        'strategy_conditions': json.dumps(params),
                        'name': strategy.get('name', ''),
                        'description': strategy.get('description', ''),
                        'created_at': strategy.get('created_at', datetime.now().isoformat()),
                        'rsi_min': params.get('rsi_min', 30.0),
                        'rsi_max': params.get('rsi_max', 70.0),
                        'volume_ratio_min': params.get('volume_ratio_min', 1.0),
                        'volume_ratio_max': params.get('volume_ratio_max', 2.0),
                        'macd_buy_threshold': macd_buy if macd_buy is not None else 0.0,
                        'macd_sell_threshold': macd_sell if macd_sell is not None else 0.0,
                        # 🆕 핵심 지표 min/max 값 저장
                        'mfi_min': params.get('mfi_min', 20.0),
                        'mfi_max': params.get('mfi_max', 80.0),
                        'atr_min': params.get('atr_min', 0.01),
                        'atr_max': params.get('atr_max', 0.05),
                        'adx_min': params.get('adx_min', 15.0),
                        'stop_loss_pct': params.get('stop_loss_pct', 0.02),
                        'take_profit_pct': params.get('take_profit_pct', 0.04),
                        'profit': params.get('profit', params.get('total_profit', 0.0)),
                        'win_rate': params.get('win_rate', 0.0),
                        'trades_count': params.get('trades_count', 0),
                        'max_drawdown': params.get('max_drawdown', 0.0),
                        'sharpe_ratio': params.get('sharpe_ratio', 0.0),
                        'calmar_ratio': params.get('calmar_ratio', 0.0),
                        'profit_factor': params.get('profit_factor', 0.0),
                        'avg_profit_per_trade': params.get('avg_profit_per_trade', 0.0),
                        'quality_grade': params.get('quality_grade') or strategy.get('quality_grade'),
                        'market_condition': params.get('market_condition', 'neutral'),
                        'score': params.get('score', 0.5),
                        'complexity_score': params.get('complexity_score', 0.6),
                        # 하이브리드 시스템 컬럼 (현재 미사용, 향후 확장용)
                        'hybrid_score': params.get('hybrid_score') or strategy.get('hybrid_score'),
                        'model_id': params.get('model_id') or strategy.get('model_id') or '',
                        # 활성화 상태 (현재는 모두 1, 향후 비활성화 로직 추가 시 활용)
                        'is_active': params.get('is_active', strategy.get('is_active', 1)),
                        # 🆕 증분 학습 메타데이터
                        'similarity_classification': params.get('similarity_classification') or strategy.get('similarity_classification'),
                        'similarity_score': params.get('similarity_score') or strategy.get('similarity_score'),
                        'parent_strategy_id': params.get('parent_strategy_id') or strategy.get('parent_strategy_id'),
                    }
                    expanded_strategies.append(expanded)
                    
                except Exception as e:
                    logger.error(f"⚠️ 전략 변환 실패: {strategy.get('id', 'unknown')} - {e}")
                    continue
            
            # write_batch로 일괄 저장
            saved_count = write_batch(expanded_strategies, 'coin_strategies', db_path=db_path)
            
            logger.info(f"✅ 전략 저장 완료: {saved_count}개")
            return saved_count
            
        except Exception as e:
            logger.error(f"❌ 전략 저장 실패: {e}")
            return 0

    def generate_strategies(self, coin: str, interval: str, n: int = None) -> List[Strategy]:
        """전략 생성 메인 함수 - 개선된 로직
        
        Args:
            coin: 코인 심볼
            interval: 시간 간격
            n: 생성할 전략 수
            
        Returns:
            생성된 전략 리스트
        """
        try:
            n = n or self.default_n_strategies
            logger.info(f"🚀 전략 생성 시작: {coin} {interval} ({n}개)")
            
            # 1. 캔들 데이터 로드
            logger.debug(f"📊 캔들 데이터 로드: {coin} {interval}")
            df = load_candles(coin, interval, days=60)
            
            if df.empty:
                logger.warning(f"⚠️ 캔들 데이터가 비어있음: {coin} {interval} - 전략 생성 불가 (기능적 실패 아님)")
                logger.info(f"📊 {coin} {interval} 캔들 데이터 로드 결과: 0개 행 (데이터 부족)")
                return []
            
            # 2. 지표 계산
            logger.debug(f"📈 지표 계산: {coin} {interval}")
            df = ensure_indicators(df)
            
            # 3. 다양한 전략 타입별 파라미터 샘플링
            strategies = []
            
            # 3.1 범위 거래 전략 (30%)
            range_count = int(n * 0.3)
            if range_count > 0:
                range_strategies = self._generate_range_trading_strategies(coin, interval, range_count)
                strategies.extend(range_strategies)
            
            # 3.2 평균 회귀 전략 (25%)
            mean_reversion_count = int(n * 0.25)
            if mean_reversion_count > 0:
                mr_strategies = self._generate_mean_reversion_strategies(coin, interval, mean_reversion_count)
                strategies.extend(mr_strategies)
            
            # 3.3 추세 추종 전략 (25%)
            trend_following_count = int(n * 0.25)
            if trend_following_count > 0:
                tf_strategies = self._generate_trend_following_strategies(coin, interval, trend_following_count)
                strategies.extend(tf_strategies)
            
            # 3.4 볼륨 스파이크 전략 (20%)
            volume_spike_count = n - len(strategies)
            if volume_spike_count > 0:
                vs_strategies = self._generate_volume_spike_strategies(coin, interval, volume_spike_count)
                strategies.extend(vs_strategies)
            
            logger.info(f"✅ 전략 생성 완료: {len(strategies)}개 생성됨")
            if len(strategies) == 0:
                logger.warning(f"⚠️ {coin} {interval}: 전략 생성 결과 0개 - 데이터 품질 문제 또는 생성 조건 미충족 (기능적 실패 아님)")
                logger.info(f"📊 {coin} {interval} 전략 생성 시도: {n}개 요청, {len(strategies)}개 생성됨")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ 전략 생성 실패: {e}")
            raise StrategyError(f"전략 생성 실패 ({coin} {interval}): {e}") from e
    
    def _generate_range_trading_strategies(self, coin: str, interval: str, n: int) -> List[Strategy]:
        """범위 거래 전략 생성 - 실제 캔들 데이터 기반"""
        try:
            import random
            strategies = []
            
            # 🔥 실제 캔들 데이터 로드 및 지표 계산
            df = load_candles(coin, interval, days=60)
            if not df.empty:
                df = ensure_indicators(df)
            
            # 실제 지표값 계산
            if not df.empty and len(df) > 20:
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
            
            for i in range(n):
                # 실제 데이터 기반으로 min/max 계산 (다양성을 위해 랜덤 오프셋 추가)
                rsi_offset = random.uniform(-rsi_std * 0.2, rsi_std * 0.2)
                params = {
                    'rsi_min': round(max(20, avg_rsi - rsi_std + rsi_offset), 1),
                    'rsi_max': round(min(80, avg_rsi + rsi_std + rsi_offset), 1),
                    'volume_ratio_min': round(max(0.5, avg_volume_ratio - volume_std * 0.5), 2),
                    'volume_ratio_max': round(min(3.0, avg_volume_ratio + volume_std), 2),
                    'mfi_min': round(max(20, avg_mfi - mfi_std), 1),
                    'mfi_max': round(min(80, avg_mfi + mfi_std), 1),
                    'atr_min': round(max(0.005, avg_atr - atr_std), 4),
                    'atr_max': round(min(0.1, avg_atr + atr_std * 2), 4),
                    'adx_min': round(max(15, avg_adx - adx_std), 1),
                    'macd_buy_threshold': (_calculate_macd_buy_threshold(df, "neutral", "range") if not df.empty else None) or 0.0,
                    'macd_sell_threshold': (_calculate_macd_sell_threshold(df, "neutral", "range") if not df.empty else None) or 0.0,
                    'stop_loss_pct': round(max(0.01, (avg_atr - atr_std) * 150), 3),  # ATR 기반
                    'take_profit_pct': round(min(0.08, (avg_atr + atr_std * 2) * 200), 2),  # ATR 기반
                    'position_size': 0.01,
                    'max_trades': 100,
                    'min_trades': 3,
                    'win_rate_threshold': 0.4,
                    'profit_threshold': 0.0,
                    'ma_period': 20,
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'pattern_confidence': 0.6,
                    'pattern_source': 'range_trading',
                    'enhancement_type': 'ai_range_trading'
                }
                
                strategy = make_strategy(params, coin, interval)
                strategies.append(strategy)
            
            logger.debug(f"✅ 범위 거래 전략 생성: {len(strategies)}개")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ 범위 거래 전략 생성 실패: {e}")
            return []
    
    def _generate_mean_reversion_strategies(self, coin: str, interval: str, n: int) -> List[Strategy]:
        """평균 회귀 전략 생성 - 실제 캔들 데이터 기반"""
        try:
            import random
            strategies = []
            
            # 🔥 실제 캔들 데이터 로드 및 지표 계산
            df = load_candles(coin, interval, days=60)
            if not df.empty:
                df = ensure_indicators(df)
            
            # 실제 지표값 계산
            if not df.empty and len(df) > 20:
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
            
            for i in range(n):
                # 실제 데이터 기반으로 min/max 계산 (평균 회귀: 넓은 범위)
                rsi_offset = random.uniform(-rsi_std * 0.5, rsi_std * 0.5)
                params = {
                    'rsi_min': round(max(15, avg_rsi - rsi_std * 2 + rsi_offset), 1),
                    'rsi_max': round(min(85, avg_rsi + rsi_std * 2 + rsi_offset), 1),
                    'volume_ratio_min': round(max(0.8, avg_volume_ratio - volume_std), 2),
                    'volume_ratio_max': round(min(3.5, avg_volume_ratio + volume_std * 1.5), 2),
                    'mfi_min': round(max(10, avg_mfi - mfi_std * 1.5), 1),
                    'mfi_max': round(min(90, avg_mfi + mfi_std * 1.5), 1),
                    'atr_min': round(max(0.005, avg_atr - atr_std), 4),
                    'atr_max': round(min(0.1, avg_atr + atr_std * 2), 4),
                    'adx_min': round(max(15, avg_adx - adx_std * 0.5), 1),
                    'macd_buy_threshold': (_calculate_macd_buy_threshold(df, "neutral", "reversal") if not df.empty else None) or 0.0,
                    'macd_sell_threshold': (_calculate_macd_sell_threshold(df, "neutral", "reversal") if not df.empty else None) or 0.0,
                    'stop_loss_pct': round(max(0.02, (avg_atr - atr_std) * 120), 3),  # ATR 기반 (좁은 손절)
                    'take_profit_pct': round(min(0.10, (avg_atr + atr_std * 2) * 150), 2),  # ATR 기반 (빠른 익절)
                    'position_size': 0.015,
                    'max_trades': 80,
                    'min_trades': 2,
                    'win_rate_threshold': 0.5,
                    'profit_threshold': 0.0,
                    'ma_period': 15,
                    'bb_period': 15,
                    'bb_std': 1.8,
                    'pattern_confidence': 0.7,
                    'pattern_source': 'mean_reversion',
                    'enhancement_type': 'ai_mean_reversion'
                }
                
                strategy = make_strategy(params, coin, interval)
                strategies.append(strategy)
            
            logger.debug(f"✅ 평균 회귀 전략 생성: {len(strategies)}개")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ 평균 회귀 전략 생성 실패: {e}")
            return []
    
    def _generate_trend_following_strategies(self, coin: str, interval: str, n: int) -> List[Strategy]:
        """추세 추종 전략 생성 - 실제 캔들 데이터 기반"""
        try:
            import random
            strategies = []
            
            # 🔥 실제 캔들 데이터 로드 및 지표 계산
            df = load_candles(coin, interval, days=60)
            if not df.empty:
                df = ensure_indicators(df)
            
            # 실제 지표값 계산
            if not df.empty and len(df) > 20:
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
            
            for i in range(n):
                # 실제 데이터 기반으로 min/max 계산 (추세 추종: 중간 범위, 높은 ADX)
                rsi_offset = random.uniform(-rsi_std * 0.4, rsi_std * 0.4)
                params = {
                    'rsi_min': round(max(35, avg_rsi - rsi_std * 1.5 + rsi_offset), 1),
                    'rsi_max': round(min(75, avg_rsi + rsi_std * 1.5 + rsi_offset), 1),
                    'volume_ratio_min': round(max(1.0, avg_volume_ratio - volume_std * 0.5), 2),
                    'volume_ratio_max': round(min(4.0, avg_volume_ratio + volume_std * 1.5), 2),
                    'mfi_min': round(max(15, avg_mfi - mfi_std * 1.2), 1),
                    'mfi_max': round(min(85, avg_mfi + mfi_std * 1.5), 1),
                    'atr_min': round(max(0.005, avg_atr - atr_std), 4),
                    'atr_max': round(min(0.1, avg_atr + atr_std * 2), 4),
                    'adx_min': round(max(20, avg_adx), 1),  # 추세 추종은 높은 ADX
                    'macd_buy_threshold': (_calculate_macd_buy_threshold(df, "neutral", "trend") if not df.empty else None) or 0.0,
                    'macd_sell_threshold': (_calculate_macd_sell_threshold(df, "neutral", "trend") if not df.empty else None) or 0.0,
                    'stop_loss_pct': round(max(0.015, (avg_atr - atr_std) * 180), 3),  # ATR 기반 (넓은 손절)
                    'take_profit_pct': round(min(0.12, (avg_atr + atr_std * 2) * 250), 2),  # ATR 기반 (큰 익절)
                    'position_size': 0.012,
                    'max_trades': 120,
                    'min_trades': 4,
                    'win_rate_threshold': 0.45,
                    'profit_threshold': 0.0,
                    'ma_period': 25,
                    'bb_period': 25,
                    'bb_std': 2.2,
                    'pattern_confidence': 0.65,
                    'pattern_source': 'trend_following',
                    'enhancement_type': 'ai_trend_follow'
                }
                
                strategy = make_strategy(params, coin, interval)
                strategies.append(strategy)
            
            logger.debug(f"✅ 추세 추종 전략 생성: {len(strategies)}개")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ 추세 추종 전략 생성 실패: {e}")
            return []
    
    def _generate_volume_spike_strategies(self, coin: str, interval: str, n: int) -> List[Strategy]:
        """볼륨 스파이크 전략 생성 - 실제 캔들 데이터 기반"""
        try:
            import random
            strategies = []
            
            # 🔥 실제 캔들 데이터 로드 및 지표 계산
            df = load_candles(coin, interval, days=60)
            if not df.empty:
                df = ensure_indicators(df)
            
            # 실제 지표값 계산
            if not df.empty and len(df) > 20:
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
            
            for i in range(n):
                # 실제 데이터 기반으로 min/max 계산 (볼륨 스파이크: 높은 거래량 중심)
                rsi_offset = random.uniform(-rsi_std * 0.3, rsi_std * 0.3)
                params = {
                    'rsi_min': round(max(20, avg_rsi - rsi_std * 1.5 + rsi_offset), 1),
                    'rsi_max': round(min(80, avg_rsi + rsi_std * 1.5 + rsi_offset), 1),
                    'volume_ratio_min': round(max(1.2, avg_volume_ratio + volume_std), 2),  # 높은 거래량
                    'volume_ratio_max': round(min(5.0, avg_volume_ratio + volume_std * 3), 2),
                    'mfi_min': round(max(10, avg_mfi - mfi_std * 2), 1),
                    'mfi_max': round(min(90, avg_mfi + mfi_std * 2), 1),
                    'atr_min': round(max(0.005, avg_atr - atr_std), 4),
                    'atr_max': round(min(0.1, avg_atr + atr_std * 2), 4),
                    'adx_min': round(max(20, avg_adx - adx_std), 1),
                    'macd_buy_threshold': (_calculate_macd_buy_threshold(df, "neutral", "volume") if not df.empty else None) or 0.0,
                    'macd_sell_threshold': (_calculate_macd_sell_threshold(df, "neutral", "volume") if not df.empty else None) or 0.0,
                    'stop_loss_pct': round(max(0.015, (avg_atr - atr_std) * 150), 3),  # ATR 기반
                    'take_profit_pct': round(min(0.08, (avg_atr + atr_std * 2) * 200), 2),  # ATR 기반
                    'position_size': 0.008,
                    'max_trades': 60,
                    'min_trades': 2,
                    'win_rate_threshold': 0.55,
                    'profit_threshold': 0.0,
                    'ma_period': 18,
                    'bb_period': 18,
                    'bb_std': 1.9,
                    'pattern_confidence': 0.75,
                    'pattern_source': 'volume_spike',
                    'enhancement_type': 'ai_volume_spike'
                }
                
                strategy = make_strategy(params, coin, interval)
                strategies.append(strategy)
            
            logger.debug(f"✅ 볼륨 스파이크 전략 생성: {len(strategies)}개")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ 볼륨 스파이크 전략 생성 실패: {e}")
            return []
    
    def generate_strategies_with_indicators(self, coin: str, interval: str, n: int = None) -> tuple[List[Strategy], Any]:
        """지표 데이터와 함께 전략 생성
        
        Args:
            coin: 코인 심볼
            interval: 시간 간격
            n: 생성할 전략 수
            
        Returns:
            (전략 리스트, 지표 데이터프레임) 튜플
        """
        try:
            n = n or self.default_n_strategies
            logger.info(f"🚀 전략 생성 (지표 포함): {coin} {interval} ({n}개)")
            
            # 1. 캔들 데이터 로드
            df = load_candles(coin, interval, days=60)
            
            if df.empty:
                logger.warning(f"⚠️ 캔들 데이터가 비어있음: {coin} {interval}")
                return [], df
            
            # 2. 지표 계산
            df = ensure_indicators(df)
            
            # 3. 전략 생성
            strategies = self.generate_strategies(coin, interval, n)
            
            logger.info(f"✅ 전략 생성 (지표 포함) 완료: {len(strategies)}개")
            return strategies, df
            
        except Exception as e:
            logger.error(f"❌ 전략 생성 (지표 포함) 실패: {e}")
            raise StrategyError(f"전략 생성 (지표 포함) 실패: {e}") from e
    
    def save_strategies_to_db(self, strategies: List[Strategy]) -> int:
        """전략들을 데이터베이스에 저장
        
        Args:
            strategies: 저장할 전략 리스트
            
        Returns:
            저장된 전략 수
        """
        try:
            if not strategies:
                logger.warning("⚠️ 저장할 전략이 없음")
                return 0
            
            # 전략들을 딕셔너리로 직렬화
            strategy_data = []
            for strategy in strategies:
                data = serialize_strategy(strategy)
                
                # strategy_conditions 생성
                # 🔥 CRITICAL FIX: TOP-LEVEL에서 먼저 확인, 없으면 params 확인 (파라미터 다양성 손실 버그 수정)
                strategy_conditions = {
                    'rsi_min': data.get('rsi_min') or data['params'].get('rsi_min', 30.0),
                    'rsi_max': data.get('rsi_max') or data['params'].get('rsi_max', 70.0),
                    'volume_ratio_min': data.get('volume_ratio_min') or data['params'].get('volume_ratio_min', 1.0),
                    'volume_ratio_max': data.get('volume_ratio_max') or data['params'].get('volume_ratio_max', 2.0),
                    'macd_buy_threshold': data.get('macd_buy_threshold') or data['params'].get('macd_buy_threshold', 0.01),
                    'macd_sell_threshold': data.get('macd_sell_threshold') or data['params'].get('macd_sell_threshold', -0.01),
                    'stop_loss_pct': data.get('stop_loss_pct') or data['params'].get('stop_loss_pct', 0.02),
                    'take_profit_pct': data.get('take_profit_pct') or data['params'].get('take_profit_pct', 0.04),
                }
                
                # 데이터베이스 저장용 필드 추가 (개선된 스키마에 맞춤)
                db_record = {
                    'id': data['id'],
                    'coin': data['coin'],
                    'interval': data['interval'],
                    'strategy_type': data.get('strategy_type', 'hybrid'),
                    'strategy_conditions': json.dumps(strategy_conditions),  # 실제 값 사용
                    # 🆕 레짐 필드 추가
                    'regime': data.get('regime', 'ranging'),
                    # 🔥 CRITICAL FIX: TOP-LEVEL에서 먼저 확인, 없으면 params 확인 (파라미터 다양성 손실 버그 수정)
                    'rsi_min': data.get('rsi_min') or data['params'].get('rsi_min', 30.0),
                    'rsi_max': data.get('rsi_max') or data['params'].get('rsi_max', 70.0),
                    'volume_ratio_min': data.get('volume_ratio_min') or data['params'].get('volume_ratio_min', 1.0),
                    'volume_ratio_max': data.get('volume_ratio_max') or data['params'].get('volume_ratio_max', 2.0),
                    'macd_buy_threshold': data.get('macd_buy_threshold') or data['params'].get('macd_buy_threshold') or getattr(strategy, 'macd_buy_threshold', None) or 0.0,
                    'macd_sell_threshold': data.get('macd_sell_threshold') or data['params'].get('macd_sell_threshold') or getattr(strategy, 'macd_sell_threshold', None) or 0.0,
                    # 🆕 핵심 지표 min/max 값 저장
                    # 🔥 CRITICAL FIX: TOP-LEVEL에서 먼저 확인 (파라미터 다양성 손실 버그 수정)
                    'mfi_min': (data.get('mfi_min') or 
                                data['params'].get('mfi_min') or 
                                getattr(strategy, 'mfi_min', None) or 20.0),
                    'mfi_max': (data.get('mfi_max') or 
                                data['params'].get('mfi_max') or 
                                getattr(strategy, 'mfi_max', None) or 80.0),
                    'atr_min': (data.get('atr_min') or 
                                data['params'].get('atr_min') or 
                                (getattr(strategy, 'atr_condition', {}).get('min') if hasattr(strategy, 'atr_condition') and strategy.atr_condition else None) or 0.01),
                    'atr_max': (data.get('atr_max') or 
                                data['params'].get('atr_max') or 
                                (getattr(strategy, 'atr_condition', {}).get('max') if hasattr(strategy, 'atr_condition') and strategy.atr_condition else None) or 0.05),
                    'adx_min': (data.get('adx_min') or 
                                data['params'].get('adx_min') or 
                                getattr(strategy, 'adx_min', None) or 15.0),
                    'stop_loss_pct': data.get('stop_loss_pct') or data['params'].get('stop_loss_pct', 0.02),
                    'take_profit_pct': data.get('take_profit_pct') or data['params'].get('take_profit_pct', 0.04),
                    'profit': 0.0,  # 시뮬레이션 후 업데이트
                    'win_rate': 0.0,  # 시뮬레이션 후 업데이트
                    'trades_count': 0,  # 시뮬레이션 후 업데이트
                    'created_at': data['created_at'],
                    'max_drawdown': 0.0,  # 시뮬레이션 후 업데이트
                    'sharpe_ratio': 0.0,  # 시뮬레이션 후 업데이트
                    'calmar_ratio': 0.0,  # 시뮬레이션 후 업데이트
                    'profit_factor': 0.0,  # 시뮬레이션 후 업데이트
                    'avg_profit_per_trade': 0.0,  # 시뮬레이션 후 업데이트
                    'quality_grade': 'UNKNOWN',  # 🔥 개선: 미검증 전략은 UNKNOWN
                    'complexity_score': 0.6,  # 기본 복잡도
                    'score': 0.5,  # 기본 점수
                    # 추가 필드들 (스키마에 맞춤)
                    # 🔥 CRITICAL FIX: TOP-LEVEL에서 먼저 확인 (파라미터 다양성 손실 버그 수정)
                    'ma_period': data.get('ma_period') or data['params'].get('ma_period', 20),
                    'bb_period': data.get('bb_period') or data['params'].get('bb_period', 20),
                    'bb_std': data.get('bb_std') or data['params'].get('bb_std', 2.0),
                    'market_condition': data.get('market_condition', 'neutral'),
                    'pattern_confidence': data.get('pattern_confidence', 0.5),
                    'pattern_source': data.get('pattern_source', 'unknown'),
                    'enhancement_type': data.get('enhancement_type', 'standard'),
                    'is_active': data.get('is_active', 1),
                    # 하이브리드 시스템 컬럼 (현재 미사용, 향후 확장용)
                    'hybrid_score': data.get('hybrid_score') or data['params'].get('hybrid_score'),
                    'model_id': data.get('model_id') or data['params'].get('model_id') or '',
                    # 🆕 증분 학습 메타데이터
                    'similarity_classification': (data.get('similarity_classification') or
                                                   data['params'].get('similarity_classification') or
                                                   getattr(strategy, 'similarity_classification', None)),
                    'similarity_score': (data.get('similarity_score') or
                                         data['params'].get('similarity_score') or
                                         getattr(strategy, 'similarity_score', None)),
                    'parent_strategy_id': (data.get('parent_strategy_id') or
                                           data['params'].get('parent_strategy_id') or
                                           getattr(strategy, 'parent_strategy_id', None)),
                    'params': json.dumps(data.get('params', {}))  # 전체 파라미터 저장
                }
                strategy_data.append(db_record)
            
            # 배치 저장
            logger.info(f"🔍 전략 저장 시작: {len(strategy_data)}개 전략 데이터 준비됨")
            saved_count = write_batch(strategy_data, 'coin_strategies')
            logger.info(f"🔍 write_batch 결과: {saved_count}개 저장됨")
            
            logger.info(f"✅ 전략 DB 저장 완료: {saved_count}개")
            return saved_count
            
        except Exception as e:
            logger.error(f"❌ 전략 DB 저장 실패: {e}")
            raise StrategyError(f"전략 DB 저장 실패: {e}") from e
    
    def generate_and_save_strategies(self, coin: str, interval: str, n: int = None) -> int:
        """전략 생성 및 저장 (통합 함수)
        
        Args:
            coin: 코인 심볼
            interval: 시간 간격
            n: 생성할 전략 수
            
        Returns:
            저장된 전략 수
        """
        try:
            # 전략 생성
            strategies = self.generate_strategies(coin, interval, n)
            
            # 데이터베이스 저장 (첫 번째 메서드 사용)
            # 전략을 딕셔너리로 변환
            strategy_dicts = []
            for strategy in strategies:
                if hasattr(strategy, '__dict__'):
                    strategy_dicts.append(strategy.__dict__)
                else:
                    strategy_dicts.append(strategy)
            
            saved_count = self.save_strategies_to_db_dict(strategy_dicts)
            
            logger.info(f"✅ 전략 생성 및 저장 완료: {saved_count}개")
            if saved_count == 0:
                logger.warning(f"⚠️ {coin} {interval}: 전략 생성 결과 0개 - 데이터 부족 또는 생성 조건 미충족 (기능적 실패 아님)")
                logger.info(f"📊 {coin} {interval} 전략 생성 시도: {len(strategies)}개 생성됨, {saved_count}개 저장됨")
            return saved_count
            
        except Exception as e:
            logger.error(f"❌ 전략 생성 및 저장 실패: {e}")
            raise StrategyError(f"전략 생성 및 저장 실패: {e}") from e
    
    def get_strategy_statistics(self, strategies: List[Strategy]) -> Dict[str, Any]:
        """전략 통계 정보 생성
        
        Args:
            strategies: 분석할 전략 리스트
            
        Returns:
            통계 정보 딕셔너리
        """
        try:
            if not strategies:
                return {
                    'total_count': 0,
                    'avg_complexity': 0.0,
                    'avg_confidence': 0.0,
                    'coin_distribution': {},
                    'interval_distribution': {}
                }
            
            # 기본 통계
            total_count = len(strategies)
            avg_complexity = sum(s.complexity_score for s in strategies) / total_count
            avg_confidence = sum(s.confidence for s in strategies) / total_count
            
            # 코인 분포
            coin_distribution = {}
            for strategy in strategies:
                coin_distribution[strategy.coin] = coin_distribution.get(strategy.coin, 0) + 1
            
            # 인터벌 분포
            interval_distribution = {}
            for strategy in strategies:
                interval_distribution[strategy.interval] = interval_distribution.get(strategy.interval, 0) + 1
            
            statistics = {
                'total_count': total_count,
                'avg_complexity': round(avg_complexity, 4),
                'avg_confidence': round(avg_confidence, 4),
                'coin_distribution': coin_distribution,
                'interval_distribution': interval_distribution
            }
            
            logger.debug(f"✅ 전략 통계 생성 완료: {total_count}개")
            return statistics
            
        except Exception as e:
            logger.error(f"❌ 전략 통계 생성 실패: {e}")
            raise StrategyError(f"전략 통계 생성 실패: {e}") from e

# 전역 인스턴스
_strategy_manager: Optional[StrategyManager] = None

def get_strategy_manager() -> StrategyManager:
    """전략 매니저 인스턴스 반환"""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
    return _strategy_manager

# 편의 함수들
def generate_strategies(coin: str, interval: str, n: int = None) -> List[Strategy]:
    """전략 생성 (편의 함수)"""
    manager = get_strategy_manager()
    return manager.generate_strategies(coin, interval, n)

def generate_strategies_with_indicators(coin: str, interval: str, n: int = None) -> tuple[List[Strategy], Any]:
    """지표 데이터와 함께 전략 생성 (편의 함수)"""
    manager = get_strategy_manager()
    return manager.generate_strategies_with_indicators(coin, interval, n)

def save_strategies_to_db(strategies: List[Strategy]) -> int:
    """전략들을 데이터베이스에 저장 (편의 함수)"""
    manager = get_strategy_manager()
    return manager.save_strategies_to_db(strategies)

def generate_and_save_strategies(coin: str, interval: str, n: int = None) -> int:
    """전략 생성 및 저장 (편의 함수)"""
    manager = get_strategy_manager()
    return manager.generate_and_save_strategies(coin, interval, n)

def get_strategy_statistics(strategies: List[Strategy]) -> Dict[str, Any]:
    """전략 통계 정보 생성 (편의 함수)"""
    manager = get_strategy_manager()
    return manager.get_strategy_statistics(strategies)

def create_run_record(run_id: str, notes: str = None, coin: str = None, interval: str = None) -> bool:
    """새로운 실행 기록 생성 - 중복 방지 (개선된 버전: coin, interval 포함)
    
    runs 테이블과 run_records 테이블 모두에 저장 (하위 호환성 유지)
    """
    try:
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        from rl_pipeline.core.env import config
        from datetime import datetime
        
        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:
            cursor = conn.cursor()
            
            # 🔥 1. runs 테이블에 저장 (기존 로직)
            cursor.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,))
            existing_runs = cursor.fetchone()[0]
            
            if existing_runs == 0:
                cursor.execute("""
                    INSERT INTO runs (run_id, coin, interval, start_time, notes, status)
                    VALUES (?, ?, ?, datetime('now'), ?, 'running')
                """, (run_id, coin, interval, notes))
            
            # 🔥 2. run_records 테이블에도 저장 (새로운 테이블)
            try:
                # run_records 테이블 존재 여부 확인
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='run_records'
                """)
                has_run_records = cursor.fetchone() is not None
                
                if has_run_records:
                    cursor.execute("SELECT COUNT(*) FROM run_records WHERE run_id = ?", (run_id,))
                    existing_records = cursor.fetchone()[0]
                    
                    if existing_records == 0:
                        now = datetime.now().isoformat()
                        cursor.execute("""
                            INSERT INTO run_records 
                            (run_id, status, message, coin, interval, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (run_id, 'running', notes, coin, interval, now, now))
                        logger.debug(f"✅ run_records 테이블에 저장 완료: {run_id}")
                    else:
                        logger.debug(f"⚠️ run_records에 이미 존재: {run_id}")
                else:
                    logger.debug(f"⚠️ run_records 테이블이 없음 (무시)")
            except Exception as e:
                logger.warning(f"⚠️ run_records 저장 실패 (무시): {e}")
            
            conn.commit()
            
            if existing_runs == 0:
                logger.info(f"✅ 실행 기록 생성 완료: {run_id} (coin={coin}, interval={interval})")
            else:
                logger.info(f"✅ 실행 기록 확인 완료 (이미 존재): {run_id}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ 실행 기록 생성 실패: {e}")
        return False

def update_run_record(run_id: str, status: str, message: str = "", 
                      strategies_count: int = None, successful_strategies: int = None, 
                      error_count: int = None) -> bool:
    """실행 기록 업데이트 - 개선된 버전 (통계 정보 포함)
    
    runs 테이블과 run_records 테이블 모두에 업데이트 (하위 호환성 유지)
    """
    try:
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        from rl_pipeline.core.env import config
        from datetime import datetime
        
        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:
            cursor = conn.cursor()
            
            # 업데이트할 필드들 동적 구성
            update_fields = ["status = ?", "notes = ?", "completed_at = datetime('now')"]
            update_values = [status, message]
            
            # 통계 정보 추가 (값이 제공된 경우만)
            if strategies_count is not None:
                update_fields.append("strategies_count = ?")
                update_values.append(strategies_count)
            if successful_strategies is not None:
                update_fields.append("successful_strategies = ?")
                update_values.append(successful_strategies)
            if error_count is not None:
                update_fields.append("error_count = ?")
                update_values.append(error_count)
            
            update_values.append(run_id)
            
            # 🔥 1. runs 테이블 업데이트 (기존 로직)
            try:
                query = f"UPDATE runs SET {', '.join(update_fields)} WHERE run_id = ?"
                cursor.execute(query, tuple(update_values))
            except Exception as e:
                # completed_at 컬럼이 없는 경우 제외하고 재시도
                update_fields_safe = [f for f in update_fields if 'completed_at' not in f]
                if update_fields_safe:
                    query = f"UPDATE runs SET {', '.join(update_fields_safe)} WHERE run_id = ?"
                    safe_values = [v for i, v in enumerate(update_values) if i < len(update_values) - 1]
                    safe_values.append(run_id)
                    cursor.execute(query, tuple(safe_values))
                else:
                    raise
            
            # 🔥 2. run_records 테이블도 업데이트
            try:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='run_records'
                """)
                has_run_records = cursor.fetchone() is not None
                
                if has_run_records:
                    run_records_fields = []
                    run_records_values = []
                    
                    if status:
                        run_records_fields.append("status = ?")
                        run_records_values.append(status)
                    
                    if message:
                        run_records_fields.append("message = ?")
                        run_records_values.append(message)
                    
                    if strategies_count is not None:
                        run_records_fields.append("strategies_count = ?")
                        run_records_values.append(strategies_count)
                    
                    if successful_strategies is not None:
                        run_records_fields.append("successful_strategies = ?")
                        run_records_values.append(successful_strategies)
                    
                    if error_count is not None:
                        run_records_fields.append("error_count = ?")
                        run_records_values.append(error_count)
                    
                    # updated_at 항상 업데이트
                    run_records_fields.append("updated_at = ?")
                    run_records_values.append(datetime.now().isoformat())
                    run_records_values.append(run_id)
                    
                    if run_records_fields:
                        query = f"UPDATE run_records SET {', '.join(run_records_fields)} WHERE run_id = ?"
                        cursor.execute(query, tuple(run_records_values))
                        logger.debug(f"✅ run_records 테이블 업데이트 완료: {run_id}")
            except Exception as e:
                logger.warning(f"⚠️ run_records 업데이트 실패 (무시): {e}")
            
            conn.commit()
            stats_info = ""
            if strategies_count is not None or successful_strategies is not None or error_count is not None:
                stats_info = f" (strategies={strategies_count}, successful={successful_strategies}, errors={error_count})"
            logger.info(f"✅ 실행 기록 업데이트 완료: {run_id} -> {status}{stats_info}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 실행 기록 업데이트 실패: {e}")
        return False

def create_missing_tables_if_needed():
    """누락된 테이블들 생성"""
    try:
        from rl_pipeline.db.schema import setup_database_tables
        setup_database_tables()
        logger.info("✅ 누락된 테이블들 생성 완료")
    except Exception as e:
        logger.error(f"❌ 테이블 생성 실패: {e}")

def calculate_optimal_iterations(
    current_quality: float,
    previous_quality: float = None,
    max_iterations: int = 10,
    quality_threshold: float = 0.8,
    improvement_threshold: float = 0.05,
    min_iterations: int = 1
) -> int:
    """
    품질 기준에 따른 최적 반복 횟수 계산
    
    Args:
        current_quality: 현재 품질 점수 (0.0 ~ 1.0)
        previous_quality: 이전 품질 점수 (0.0 ~ 1.0)
        max_iterations: 최대 반복 횟수
        quality_threshold: 품질 임계값 (이상이면 조기 종료)
        improvement_threshold: 개선 임계값 (이하이면 추가 반복)
        min_iterations: 최소 반복 횟수
    
    Returns:
        권장 반복 횟수
    """
    try:
        # 기본 반복 횟수
        recommended_iterations = min_iterations
        
        # 품질이 임계값 이상이면 조기 종료
        if current_quality >= quality_threshold:
            logger.info(f"🎯 품질 임계값 달성 ({current_quality:.3f} >= {quality_threshold:.3f}) - 조기 종료")
            return min_iterations
        
        # 이전 품질과 비교하여 개선도 계산
        if previous_quality is not None:
            improvement = current_quality - previous_quality
            
            # 개선도가 임계값 이하면 추가 반복 필요
            if improvement <= improvement_threshold:
                additional_iterations = min(3, max_iterations - min_iterations)
                recommended_iterations = min_iterations + additional_iterations
                logger.info(f"📈 개선도 부족 ({improvement:.3f} <= {improvement_threshold:.3f}) - 추가 반복: {additional_iterations}회")
            else:
                logger.info(f"✅ 충분한 개선도 ({improvement:.3f} > {improvement_threshold:.3f}) - 기본 반복")
        else:
            # 첫 실행이면 중간 수준으로 시작
            recommended_iterations = min(3, max_iterations)
            logger.info(f"🔄 첫 실행 - 중간 수준 반복: {recommended_iterations}회")
        
        # 최대 반복 횟수 제한
        recommended_iterations = min(recommended_iterations, max_iterations)
        
        logger.info(f"🎯 권장 반복 횟수: {recommended_iterations}회 (품질: {current_quality:.3f})")
        return recommended_iterations
        
    except Exception as e:
        logger.error(f"❌ 반복 횟수 계산 실패: {e}")
        return min_iterations

