"""
전략 생성 개선 모듈 - 방향성 확보를 위한 추가 기능
- 중복 검증
- 그리드 서치 기반 전략 생성
- 방향성별 특화 전략
- 파라미터 공간 체계적 커버리지
"""

import logging
import random
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from rl_pipeline.core.types import Strategy
from rl_pipeline.core.env import config

logger = logging.getLogger(__name__)


def generate_strategy_hash(strategy: Strategy) -> str:
    """전략의 핵심 파라미터로 해시 생성 (중복 검증용)"""
    try:
        # 🔧 None 값 처리 및 기본값 설정
        rsi_min = strategy.rsi_min if strategy.rsi_min is not None else 30.0
        rsi_max = strategy.rsi_max if strategy.rsi_max is not None else 70.0
        volume_ratio_min = strategy.volume_ratio_min if strategy.volume_ratio_min is not None else 1.0
        volume_ratio_max = strategy.volume_ratio_max if strategy.volume_ratio_max is not None else 2.0
        stop_loss_pct = strategy.stop_loss_pct if strategy.stop_loss_pct is not None else 0.02
        take_profit_pct = strategy.take_profit_pct if strategy.take_profit_pct is not None else 0.05
        macd_buy_threshold = strategy.macd_buy_threshold if strategy.macd_buy_threshold is not None else 0.01
        macd_sell_threshold = strategy.macd_sell_threshold if strategy.macd_sell_threshold is not None else -0.01
        
        # 핵심 파라미터만 포함 (적절한 반올림으로 실제 중복만 감지)
        # 🔧 반올림 정밀도 조정: 너무 높으면 중복이 많고, 너무 낮으면 실제 중복도 통과
        # 🆕 MFI, ATR, ADX 파라미터도 포함하여 중복 감지 정확도 향상
        mfi_min = getattr(strategy, 'mfi_min', None) or 20.0
        mfi_max = getattr(strategy, 'mfi_max', None) or 80.0
        atr_min = getattr(strategy, 'atr_min', None) or 0.01
        atr_max = getattr(strategy, 'atr_max', None) or 0.05
        adx_min = getattr(strategy, 'adx_min', None) or 15.0
        
        # 🔧 시스템에서 정의된 소수점 자리수 기준으로 반올림 (rl_pipeline/core/utils.py의 _format_decimal_precision 참고)
        # 기술지표: 4자리 (rsi, mfi, adx, atr, volume_ratio, macd)
        # 전략 파라미터: stop_loss/take_profit은 3자리, 나머지는 4자리
        key_params = {
            'rsi_min': round(float(rsi_min), 4),  # 4자리 (시스템 정의)
            'rsi_max': round(float(rsi_max), 4),  # 4자리 (시스템 정의)
            'volume_ratio_min': round(float(volume_ratio_min), 4),  # 4자리 (시스템 정의)
            'volume_ratio_max': round(float(volume_ratio_max), 4),  # 4자리 (시스템 정의)
            'stop_loss_pct': round(float(stop_loss_pct), 3),  # 3자리 (시스템 정의)
            'take_profit_pct': round(float(take_profit_pct), 3),  # 3자리 (시스템 정의)
            'macd_buy_threshold': round(float(macd_buy_threshold), 4),  # 4자리 (시스템 정의)
            'macd_sell_threshold': round(float(macd_sell_threshold), 4),  # 4자리 (시스템 정의)
            # 🆕 추가 지표 파라미터 (시스템 정의 정밀도 적용)
            'mfi_min': round(float(mfi_min), 4),  # 4자리 (시스템 정의)
            'mfi_max': round(float(mfi_max), 4),  # 4자리 (시스템 정의)
            'atr_min': round(float(atr_min), 4),  # 4자리 (시스템 정의)
            'atr_max': round(float(atr_max), 4),  # 4자리 (시스템 정의)
            'adx_min': round(float(adx_min), 4),  # 4자리 (시스템 정의)
        }
        
        # 정렬된 딕셔너리를 문자열로 변환
        params_str = json.dumps(key_params, sort_keys=True)
        hash_value = hashlib.md5(params_str.encode()).hexdigest()
        return hash_value
    except Exception as e:
        logger.warning(f"⚠️ 전략 해시 생성 실패: {e}, 전략 ID: {getattr(strategy, 'id', 'unknown')}")
        # 에러 발생 시에도 기본 해시 반환 (빈 문자열 대신)
        try:
            fallback_params = {
                'rsi_min': 30.0,
                'rsi_max': 70.0,
                'volume_ratio_min': 1.0,
                'volume_ratio_max': 2.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'macd_buy_threshold': 0.01,
                'macd_sell_threshold': -0.01,
            }
            fallback_str = json.dumps(fallback_params, sort_keys=True)
            return hashlib.md5(fallback_str.encode()).hexdigest() + "_error"
        except:
            return ""


def filter_duplicate_strategies(strategies: List[Strategy]) -> List[Strategy]:
    """중복 전략 필터링 (개선: 해시 + 파라미터 직접 비교)"""
    try:
        seen_hashes: Set[str] = set()
        unique_strategies = []
        duplicate_count = 0
        
        for strategy in strategies:
            strategy_hash = generate_strategy_hash(strategy)
            if strategy_hash and strategy_hash not in seen_hashes:
                seen_hashes.add(strategy_hash)
                unique_strategies.append(strategy)
            else:
                duplicate_count += 1
                if duplicate_count <= 5:  # 처음 5개만 상세 로그
                    logger.debug(f"🔍 중복 전략 제거: {strategy.id} (RSI={strategy.rsi_min:.1f}-{strategy.rsi_max:.1f}, "
                               f"SL={strategy.stop_loss_pct:.3f}, TP={strategy.take_profit_pct:.3f})")
        
        removed_count = len(strategies) - len(unique_strategies)
        if removed_count > 0:
            logger.info(f"✅ 중복 필터링: {len(strategies)}개 → {len(unique_strategies)}개 (제거: {removed_count}개)")
        else:
            logger.debug(f"🔍 중복 필터링: {len(strategies)}개 → {len(unique_strategies)}개 (중복 없음)")
        return unique_strategies
    except Exception as e:
        logger.error(f"❌ 중복 필터링 실패: {e}")
        return strategies


def create_grid_search_strategies(coin: str, interval: str, df: Any, 
                                   n_strategies: int, seed: int = None) -> List[Strategy]:
    """그리드 서치 기반 체계적 전략 생성 (캔들 데이터 기반)"""
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        strategies = []
        
        # 🆕 실제 캔들 데이터 분석 (지능형 전략과 동일한 방식)
        if not df.empty and len(df) > 20:
            has_real_data = (
                len(df) > 0 and 
                'rsi' in df.columns and 'volume_ratio' in df.columns and
                not df['rsi'].isna().all() and not df['volume_ratio'].isna().all() and
                df['rsi'].notna().sum() > 10 and df['volume_ratio'].notna().sum() > 10
            )
            
            if has_real_data:
                # 실제 데이터에서 지표값 계산
                rsi_min_actual = df['rsi'].min()
                rsi_max_actual = df['rsi'].max()
                rsi_mean = df['rsi'].mean()
                rsi_std = df['rsi'].std()
                
                volume_min_actual = df['volume_ratio'].min()
                volume_max_actual = df['volume_ratio'].max()
                volume_mean = df['volume_ratio'].mean()
                volume_std = df['volume_ratio'].std()
                
                # ATR 계산
                if 'atr' in df.columns:
                    atr_min_actual = df['atr'].min()
                    atr_max_actual = df['atr'].max()
                    atr_mean = df['atr'].mean()
                    atr_std = df['atr'].std()
                else:
                    atr_min_actual, atr_max_actual = 0.01, 0.05
                    atr_mean, atr_std = 0.02, 0.01
                
                # 데이터 기반 범위 확장 (±표준편차로 확장, 최소/최대값 보장)
                rsi_range_min = max(10, min(rsi_min_actual, rsi_mean - rsi_std * 2))
                rsi_range_max = min(90, max(rsi_max_actual, rsi_mean + rsi_std * 2))
                rsi_mid_low = (rsi_range_min + rsi_mean) / 2  # 낮은 구간
                rsi_mid_high = (rsi_mean + rsi_range_max) / 2  # 높은 구간
                
                volume_range_min = max(0.3, min(volume_min_actual, volume_mean - volume_std * 2))
                volume_range_max = min(5.0, max(volume_max_actual, volume_mean + volume_std * 2))
                volume_mid = (volume_range_min + volume_range_max) / 2
                
                logger.debug(f"📊 {coin} {interval} 그리드 서치 데이터 기반 범위: "
                           f"RSI=[{rsi_range_min:.1f}~{rsi_mid_low:.1f}, {rsi_mid_high:.1f}~{rsi_range_max:.1f}], "
                           f"Volume=[{volume_range_min:.2f}~{volume_mid:.2f}, {volume_mid:.2f}~{volume_range_max:.2f}]")
            else:
                # 데이터 부족 시 기본값 사용
                rsi_range_min, rsi_mid_low = 15, 30
                rsi_mid_high, rsi_range_max = 55, 85
                volume_range_min, volume_mid = 0.5, 1.5
                volume_range_max = 4.0
                atr_min_actual, atr_max_actual = 0.01, 0.05
        else:
            # 데이터 없으면 기본값 사용
            rsi_range_min, rsi_mid_low = 15, 30
            rsi_mid_high, rsi_range_max = 55, 85
            volume_range_min, volume_mid = 0.5, 1.5
            volume_range_max = 4.0
            atr_min_actual, atr_max_actual = 0.01, 0.05
        
        # 🆕 데이터 기반 파라미터 그리드 정의
        rsi_min_range = np.linspace(rsi_range_min, rsi_mid_low, 8)  # 낮은 구간
        rsi_max_range = np.linspace(rsi_mid_high, rsi_range_max, 8)  # 높은 구간
        volume_min_range = np.linspace(volume_range_min, volume_mid, 6)
        volume_max_range = np.linspace(volume_mid, volume_range_max, 6)
        # 손절/익절은 상대적으로 고정 (데이터와 무관하게 안전한 범위)
        stop_loss_range = np.linspace(0.01, 0.04, 5)  # 1%~4%
        take_profit_range = np.linspace(0.03, 0.08, 5)  # 3%~8%
        
        # 그리드 포인트 수 = 8*8*6*6*5*5 = 57,600 (너무 많음, 샘플링 필요)
        # Latin Hypercube Sampling으로 체계적이면서도 효율적으로 샘플링
        from scipy.stats import qmc
        
        # n_strategies개 샘플 생성
        sampler = qmc.LatinHypercube(d=6)  # 6차원 파라미터 공간
        samples = sampler.random(n=n_strategies)
        
        # 🆕 각 샘플을 데이터 기반 파라미터 범위로 매핑
        for i, sample in enumerate(samples):
            rsi_min = np.interp(sample[0], [0, 1], [rsi_range_min, rsi_mid_low])
            rsi_max = np.interp(sample[1], [0, 1], [rsi_mid_high, rsi_range_max])
            volume_min = np.interp(sample[2], [0, 1], [volume_range_min, volume_mid])
            volume_max = np.interp(sample[3], [0, 1], [volume_mid, volume_range_max])
            stop_loss = np.interp(sample[4], [0, 1], [0.01, 0.04])
            take_profit = np.interp(sample[5], [0, 1], [0.03, 0.08])
            
            # 유효성 검증
            if rsi_min >= rsi_max:
                continue
            if volume_min >= volume_max:
                continue
            if stop_loss >= take_profit:
                continue
            
            # 🆕 모든 지표 데이터 기반 계산
            if has_real_data and not df.empty:
                # MACD
                if 'macd' in df.columns:
                    # 🔥 평균(Mean) -> 중앙값(Median) 변경으로 이상치 영향 최소화
                    macd_mean = df['macd'].median()
                    macd_std = df['macd'].std()
                    macd_min_actual = df['macd'].min()
                    macd_max_actual = df['macd'].max()
                    macd_buy = macd_mean + macd_std * random.uniform(-1, 1)
                    macd_sell = macd_mean - macd_std * random.uniform(-1, 1)
                    # 실제 범위 내로 제한
                    macd_buy = max(macd_min_actual, min(macd_max_actual, macd_buy))
                    macd_sell = max(macd_min_actual, min(macd_max_actual, macd_sell))
                else:
                    macd_buy = random.uniform(0.005, 0.02)
                    macd_sell = random.uniform(-0.02, -0.005)
                
                # ATR
                atr_value = atr_mean + atr_std * random.uniform(-1, 1)
                atr_value = max(atr_min_actual, min(atr_max_actual, atr_value))
            else:
                # 데이터 없으면 기본값
                macd_buy = random.uniform(0.005, 0.02)
                macd_sell = random.uniform(-0.02, -0.005)
                atr_value = 0.02
            
            strategy = Strategy(
                id=f"{coin}_{interval}_grid_{i:04d}",
                params={
                    'rsi_min': rsi_min,
                    'rsi_max': rsi_max,
                    'volume_ratio_min': volume_min,
                    'volume_ratio_max': volume_max,
                    'stop_loss_pct': stop_loss,
                    'take_profit_pct': take_profit,
                    'macd_buy_threshold': macd_buy,
                    'macd_sell_threshold': macd_sell,
                },
                version="v2.0",
                coin=coin,
                interval=interval,
                created_at=datetime.now(),
                strategy_type="grid_search",
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                volume_ratio_min=volume_min,
                volume_ratio_max=volume_max,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                macd_buy_threshold=macd_buy,
                macd_sell_threshold=macd_sell,
                rsi_condition={'min': rsi_min, 'max': rsi_max},
                volume_condition={'min': volume_min, 'max': volume_max},
                atr_condition={'min': max(0.01, atr_min_actual if has_real_data else 0.01), 
                             'max': min(0.05, atr_max_actual if has_real_data else 0.05)},
                pattern_source='grid_search',
                enhancement_type='systematic'
            )
            strategies.append(strategy)
        
        logger.info(f"✅ 그리드 서치 전략 생성: {len(strategies)}개")
        return strategies
        
    except ImportError:
        # scipy가 없으면 간단한 그리드 생성
        logger.warning("⚠️ scipy 없음, 간단한 그리드 생성 사용")
        return create_simple_grid_strategies(coin, interval, df, n_strategies, seed)
    except Exception as e:
        logger.error(f"❌ 그리드 서치 전략 생성 실패: {e}")
        return []


def create_simple_grid_strategies(coin: str, interval: str, df: Any, 
                                  n_strategies: int, seed: int = None) -> List[Strategy]:
    """간단한 그리드 전략 생성 (scipy 없이) - 캔들 데이터 기반"""
    try:
        if seed is not None:
            random.seed(seed)
        
        strategies = []
        
        # 🆕 실제 캔들 데이터 분석 (그리드 서치와 동일한 방식)
        if not df.empty and len(df) > 20:
            has_real_data = (
                len(df) > 0 and 
                'rsi' in df.columns and 'volume_ratio' in df.columns and
                not df['rsi'].isna().all() and not df['volume_ratio'].isna().all() and
                df['rsi'].notna().sum() > 10 and df['volume_ratio'].notna().sum() > 10
            )
            
            if has_real_data:
                rsi_min_actual = df['rsi'].min()
                rsi_max_actual = df['rsi'].max()
                rsi_mean = df['rsi'].mean()
                rsi_std = df['rsi'].std()
                
                volume_min_actual = df['volume_ratio'].min()
                volume_max_actual = df['volume_ratio'].max()
                volume_mean = df['volume_ratio'].mean()
                volume_std = df['volume_ratio'].std()
                
                # 데이터 기반 범위 확장
                rsi_range_min = max(10, min(rsi_min_actual, rsi_mean - rsi_std * 2))
                rsi_range_max = min(90, max(rsi_max_actual, rsi_mean + rsi_std * 2))
                rsi_mid_low = (rsi_range_min + rsi_mean) / 2
                rsi_mid_high = (rsi_mean + rsi_range_max) / 2
                
                volume_range_min = max(0.3, min(volume_min_actual, volume_mean - volume_std * 2))
                volume_range_max = min(5.0, max(volume_max_actual, volume_mean + volume_std * 2))
                volume_mid = (volume_range_min + volume_range_max) / 2
                
                atr_min_actual = df['atr'].min() if 'atr' in df.columns else 0.01
                atr_max_actual = df['atr'].max() if 'atr' in df.columns else 0.05
            else:
                rsi_range_min, rsi_mid_low = 15, 30
                rsi_mid_high, rsi_range_max = 55, 85
                volume_range_min, volume_mid = 0.5, 1.5
                volume_range_max = 4.0
                atr_min_actual, atr_max_actual = 0.01, 0.05
        else:
            rsi_range_min, rsi_mid_low = 15, 30
            rsi_mid_high, rsi_range_max = 55, 85
            volume_range_min, volume_mid = 0.5, 1.5
            volume_range_max = 4.0
            atr_min_actual, atr_max_actual = 0.01, 0.05
            has_real_data = False
        
        # 🆕 데이터 기반 간단한 그리드: 각 파라미터를 균등하게 샘플링
        n_per_param = int(np.ceil(np.power(n_strategies, 1/6)))  # 6차원 공간
        
        rsi_min_range = np.linspace(rsi_range_min, rsi_mid_low, n_per_param)
        rsi_max_range = np.linspace(rsi_mid_high, rsi_range_max, n_per_param)
        volume_min_range = np.linspace(volume_range_min, volume_mid, n_per_param)
        volume_max_range = np.linspace(volume_mid, volume_range_max, n_per_param)
        stop_loss_range = np.linspace(0.01, 0.04, n_per_param)
        take_profit_range = np.linspace(0.03, 0.08, n_per_param)
        
        count = 0
        for rsi_min in rsi_min_range:
            for rsi_max in rsi_max_range:
                if rsi_min >= rsi_max:
                    continue
                for vol_min in volume_min_range:
                    for vol_max in volume_max_range:
                        if vol_min >= vol_max:
                            continue
                        for sl in stop_loss_range:
                            for tp in take_profit_range:
                                if sl >= tp:
                                    continue
                                if count >= n_strategies:
                                    break
                                
                                # 🆕 MACD 데이터 기반 계산
                                if has_real_data and 'macd' in df.columns:
                                    macd_mean = df['macd'].mean()
                                    macd_std = df['macd'].std()
                                    macd_min_actual = df['macd'].min()
                                    macd_max_actual = df['macd'].max()
                                    macd_buy = macd_mean + macd_std * random.uniform(-1, 1)
                                    macd_sell = macd_mean - macd_std * random.uniform(-1, 1)
                                    macd_buy = max(macd_min_actual, min(macd_max_actual, macd_buy))
                                    macd_sell = max(macd_min_actual, min(macd_max_actual, macd_sell))
                                else:
                                    macd_buy = random.uniform(0.005, 0.02)
                                    macd_sell = random.uniform(-0.02, -0.005)
                                
                                strategy = Strategy(
                                    id=f"{coin}_{interval}_grid_{count:04d}",
                                    params={
                                        'rsi_min': float(rsi_min),
                                        'rsi_max': float(rsi_max),
                                        'volume_ratio_min': float(vol_min),
                                        'volume_ratio_max': float(vol_max),
                                        'stop_loss_pct': float(sl),
                                        'take_profit_pct': float(tp),
                                        'macd_buy_threshold': macd_buy,
                                        'macd_sell_threshold': macd_sell,
                                    },
                                    version="v2.0",
                                    coin=coin,
                                    interval=interval,
                                    created_at=datetime.now(),
                                    strategy_type="simple_grid",
                                    rsi_min=float(rsi_min),
                                    rsi_max=float(rsi_max),
                                    volume_ratio_min=float(vol_min),
                                    volume_ratio_max=float(vol_max),
                                    stop_loss_pct=float(sl),
                                    take_profit_pct=float(tp),
                                    macd_buy_threshold=macd_buy,
                                    macd_sell_threshold=macd_sell,
                                    rsi_condition={'min': float(rsi_min), 'max': float(rsi_max)},
                                    volume_condition={'min': float(vol_min), 'max': float(vol_max)},
                                    atr_condition={'min': max(0.01, atr_min_actual if has_real_data else 0.01), 
                                                 'max': min(0.05, atr_max_actual if has_real_data else 0.05)},
                                    pattern_source='simple_grid',
                                    enhancement_type='systematic'
                                )
                                strategies.append(strategy)
                                count += 1
                            if count >= n_strategies:
                                break
                        if count >= n_strategies:
                            break
                    if count >= n_strategies:
                        break
                if count >= n_strategies:
                    break
            if count >= n_strategies:
                break
        
        logger.info(f"✅ 간단한 그리드 전략 생성: {len(strategies)}개")
        return strategies
        
    except Exception as e:
        logger.error(f"❌ 간단한 그리드 전략 생성 실패: {e}")
        return []


def create_direction_specialized_strategies(coin: str, interval: str, df: Any,
                                            n_per_direction: int = 100) -> Dict[str, List[Strategy]]:
    """방향성별 특화 전략 생성 (매수/매도/홀드) - 캔들 데이터 기반"""
    try:
        
        strategies_by_direction = {
            'BUY': [],
            'SELL': [],
            'HOLD': []
        }
        
        # 🆕 실제 캔들 데이터 분석 (지능형 전략과 동일한 방식)
        if not df.empty and len(df) > 20:
            has_real_data = (
                len(df) > 0 and 
                'rsi' in df.columns and 'volume_ratio' in df.columns and
                not df['rsi'].isna().all() and not df['volume_ratio'].isna().all() and
                df['rsi'].notna().sum() > 10 and df['volume_ratio'].notna().sum() > 10
            )
            
            if has_real_data:
                # 실제 데이터에서 지표값 계산
                rsi_min_actual = df['rsi'].min()
                rsi_max_actual = df['rsi'].max()
                rsi_mean = df['rsi'].mean()
                rsi_std = df['rsi'].std()
                
                volume_min_actual = df['volume_ratio'].min()
                volume_max_actual = df['volume_ratio'].max()
                volume_mean = df['volume_ratio'].mean()
                volume_std = df['volume_ratio'].std()
                
                # MACD 계산
                if 'macd' in df.columns:
                    macd_min_actual = df['macd'].min()
                    macd_max_actual = df['macd'].max()
                    macd_mean = df['macd'].mean()
                    macd_std = df['macd'].std()
                else:
                    macd_min_actual, macd_max_actual = -0.05, 0.05
                    macd_mean, macd_std = 0.0, 0.01
                
                # ATR 계산
                if 'atr' in df.columns:
                    atr_min_actual = df['atr'].min()
                    atr_max_actual = df['atr'].max()
                    atr_mean = df['atr'].mean()
                else:
                    atr_min_actual, atr_max_actual = 0.01, 0.05
                    atr_mean = 0.02
                
                logger.debug(f"📊 {coin} {interval} 방향성별 특화 데이터 기반 범위: "
                           f"RSI=[{rsi_min_actual:.1f}~{rsi_max_actual:.1f}], "
                           f"Volume=[{volume_min_actual:.2f}~{volume_max_actual:.2f}], "
                           f"MACD=[{macd_min_actual:.4f}~{macd_max_actual:.4f}]")
            else:
                # 데이터 부족 시 기본값
                rsi_min_actual, rsi_max_actual = 10, 90
                rsi_mean, rsi_std = 50, 15
                volume_min_actual, volume_max_actual = 0.3, 5.0
                volume_mean, volume_std = 1.0, 0.5
                macd_min_actual, macd_max_actual = -0.05, 0.05
                macd_mean, macd_std = 0.0, 0.01
                atr_min_actual, atr_max_actual = 0.01, 0.05
                atr_mean = 0.02
        else:
            # 데이터 없으면 기본값
            rsi_min_actual, rsi_max_actual = 10, 90
            rsi_mean, rsi_std = 50, 15
            volume_min_actual, volume_max_actual = 0.3, 5.0
            volume_mean, volume_std = 1.0, 0.5
            macd_min_actual, macd_max_actual = -0.05, 0.05
            macd_mean, macd_std = 0.0, 0.01
            atr_min_actual, atr_max_actual = 0.01, 0.05
            atr_mean = 0.02
            has_real_data = False
        
        # 1. 매수 특화 전략 (상승 추세 포착) - 성공 패턴 기반
        logger.info(f"📈 {coin} {interval} 매수 특화 전략 생성 (성공 패턴 기반)...")
        
        # 🆕 성공 패턴 추출: 저점에서 매수해서 성공한 케이스 찾기
        successful_buy_patterns = []
        if not df.empty and len(df) > 50:
            try:
                from trade.realtime_candles_calculate import calculate_pattern_pivot_points
                df_with_pivot = calculate_pattern_pivot_points(df.copy(), interval)
                
                # 🆕 인터벌에 따라 동적으로 경계 제외 범위 조정
                # pivot 계산에 필요한 최소값 (2개) + 여유분 (3개) = 5개
                # 미래 수익 확인에 필요한 10개는 유지하되, 전체 데이터의 10%를 넘지 않도록
                pivot_window_needed = 5  # pivot 계산에 필요한 앞쪽 여유분
                future_check_needed = 10  # 미래 수익 확인에 필요한 뒤쪽 개수
                max_exclude_ratio = 0.1  # 전체 데이터의 최대 10%만 제외
                
                total_needed = pivot_window_needed + future_check_needed
                max_exclude_count = int(len(df_with_pivot) * max_exclude_ratio)
                
                # 데이터가 충분하면 고정값 사용, 부족하면 비율로 조정
                if len(df_with_pivot) > total_needed * 2:
                    start_idx = pivot_window_needed
                    end_idx = len(df_with_pivot) - future_check_needed
                else:
                    # 데이터가 적으면 비율로 조정 (최소 3개는 앞쪽, 5개는 뒤쪽)
                    start_idx = max(3, int(len(df_with_pivot) * 0.05))
                    end_idx = len(df_with_pivot) - max(5, int(len(df_with_pivot) * 0.05))
                
                # 저점에서 매수해서 성공한 패턴 추출
                for i in range(start_idx, end_idx):
                    if df_with_pivot.iloc[i]['pivot_low'] == 1:
                        entry_price = df_with_pivot.iloc[i]['low']
                        entry_candle = df_with_pivot.iloc[i]
                        
                        # 이후 10개 캔들 중 최대 수익 확인
                        future_candles = df_with_pivot.iloc[i+1:i+11]
                        if len(future_candles) > 0:
                            max_price = future_candles['high'].max()
                            max_profit_pct = (max_price - entry_price) / entry_price if entry_price > 0 else 0
                            
                            # 2% 이상 수익 발생한 경우 성공 패턴으로 저장
                            if max_profit_pct >= 0.02:
                                pattern = {
                                    'rsi': entry_candle.get('rsi', 50.0),
                                    'macd': entry_candle.get('macd', 0.0),
                                    'macd_signal': entry_candle.get('macd_signal', 0.0),
                                    'volume_ratio': entry_candle.get('volume_ratio', 1.0),
                                    'mfi': entry_candle.get('mfi', 50.0),
                                    'atr': entry_candle.get('atr', 0.02),
                                    'profit_pct': max_profit_pct
                                }
                                successful_buy_patterns.append(pattern)
                
                if successful_buy_patterns:
                    logger.info(f"  ✅ {coin} {interval} 성공 매수 패턴 {len(successful_buy_patterns)}개 발견")
                else:
                    logger.debug(f"  ⚠️ {coin} {interval} 성공 매수 패턴 없음 (기본 범위 사용)")
            except Exception as e:
                logger.debug(f"  ⚠️ {coin} {interval} 성공 패턴 추출 실패: {e}")
        
        # 성공 패턴 기반 파라미터 생성 (Instance-based Imitation)
        # 통계적 평균(Mean)을 쓰지 않고, 성공했던 개별 케이스를 직접 모방하여 다양성 확보
        if successful_buy_patterns:
            logger.info(f"  🧬 {coin} {interval}: {len(successful_buy_patterns)}개의 성공 매수 패턴을 기반으로 정밀 전략 생성")
        
        for i in range(n_per_direction):
            # 🆕 성공 패턴 기반 파라미터 생성
            if successful_buy_patterns:
                # 1. 성공했던 케이스 중 하나를 무작위 선택 (Template)
                target_pattern = random.choice(successful_buy_patterns)
                
                # 2. 해당 케이스의 지표 값을 기준으로 좁은 탐색 범위 설정 (정밀 타격)
                # RSI: 타겟 값 주변 ±3~7 범위
                center_rsi = target_pattern.get('rsi', 50)
                rsi_span = random.uniform(3, 7)
                rsi_min = max(10, center_rsi - rsi_span)
                rsi_max = min(90, center_rsi + rsi_span)
                
                # Volume: 타겟 값 주변 ±15% 범위
                center_vol = target_pattern.get('volume_ratio', 1.0)
                vol_span_ratio = random.uniform(0.1, 0.2)
                volume_min = max(0.3, center_vol * (1 - vol_span_ratio))
                volume_max = min(5.0, center_vol * (1 + vol_span_ratio))
                
                # MACD: 타겟 값 주변 미세 조정
                center_macd = target_pattern.get('macd', 0.0)
                macd_span = 0.0005  # 매우 좁게
                macd_buy = center_macd + random.uniform(-macd_span, macd_span)
                macd_sell = 0.0 # 매수 전략에서 macd_sell_threshold는 청산용이거나 미사용
            else:
                # 성공 패턴이 없으면 기본 범위 사용 (기존 로직 유지)
                rsi_low_range = max(10, rsi_min_actual)
                rsi_low_range_max = min(rsi_mean - rsi_std, rsi_max_actual * 0.5)
                rsi_min = random.uniform(rsi_low_range, rsi_low_range_max)
                rsi_max = random.uniform(rsi_mean, min(rsi_mean + rsi_std * 1.5, rsi_max_actual))
                
                volume_high_min = max(volume_mean, volume_min_actual * 1.2)
                volume_min = random.uniform(volume_high_min, volume_max_actual * 0.8)
                volume_max = random.uniform(volume_min * 1.2, min(volume_max_actual, volume_mean + volume_std * 2))
                
                macd_buy_range_min = max(macd_min_actual, macd_mean - macd_std)
                macd_buy_range_max = min(macd_max_actual, macd_mean + macd_std * 2)
                macd_buy = random.uniform(macd_buy_range_min, macd_buy_range_max)
                macd_sell = random.uniform(macd_min_actual, min(macd_mean - macd_std, macd_max_actual))
            
            # 보수적 손절, 공격적 익절
            stop_loss = random.uniform(0.015, 0.025)
            take_profit = random.uniform(0.05, 0.1)
            
            # 🆕 성공 패턴 기반 전략 메타데이터
            strategy_metadata = {
                'success_pattern_based': len(successful_buy_patterns) > 0,
                'success_pattern_count': len(successful_buy_patterns),
                'entry_filter_type': 'low_point_detection'
            }
            
            strategy = Strategy(
                id=f"{coin}_{interval}_buy_specialized_{i:04d}",
                params={
                    'rsi_min': rsi_min,
                    'rsi_max': rsi_max,
                    'volume_ratio_min': volume_min,
                    'volume_ratio_max': volume_max,
                    'stop_loss_pct': stop_loss,
                    'take_profit_pct': take_profit,
                    'macd_buy_threshold': macd_buy,
                    'macd_sell_threshold': macd_sell,
                },
                version="v2.0",
                coin=coin,
                interval=interval,
                created_at=datetime.now(),
                strategy_type="buy_specialized",
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                volume_ratio_min=volume_min,
                volume_ratio_max=volume_max,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                macd_buy_threshold=macd_buy,
                macd_sell_threshold=macd_sell,
                rsi_condition={'min': rsi_min, 'max': rsi_max},
                volume_condition={'min': volume_min, 'max': volume_max},
                atr_condition={'min': max(0.01, atr_min_actual if has_real_data else 0.01), 
                             'max': min(0.05, atr_max_actual if has_real_data else 0.05)},
                pattern_source='direction_specialized',
                enhancement_type='buy_optimized',
                metadata=strategy_metadata
            )
            strategies_by_direction['BUY'].append(strategy)
        
        # 2. 매도 특화 전략 (하락 추세 포착) - 성공 패턴 기반
        logger.info(f"📉 {coin} {interval} 매도 특화 전략 생성 (성공 패턴 기반)...")
        
        # 🆕 성공 패턴 추출: 고점에서 매도해서 성공한 케이스 찾기
        successful_sell_patterns = []
        if not df.empty and len(df) > 50:
            try:
                from trade.realtime_candles_calculate import calculate_pattern_pivot_points
                df_with_pivot = calculate_pattern_pivot_points(df.copy(), interval)
                
                # 🆕 인터벌에 따라 동적으로 경계 제외 범위 조정
                # pivot 계산에 필요한 최소값 (2개) + 여유분 (3개) = 5개
                # 미래 수익 확인에 필요한 10개는 유지하되, 전체 데이터의 10%를 넘지 않도록
                pivot_window_needed = 5  # pivot 계산에 필요한 앞쪽 여유분
                future_check_needed = 10  # 미래 수익 확인에 필요한 뒤쪽 개수
                max_exclude_ratio = 0.1  # 전체 데이터의 최대 10%만 제외
                
                total_needed = pivot_window_needed + future_check_needed
                max_exclude_count = int(len(df_with_pivot) * max_exclude_ratio)
                
                # 데이터가 충분하면 고정값 사용, 부족하면 비율로 조정
                if len(df_with_pivot) > total_needed * 2:
                    start_idx = pivot_window_needed
                    end_idx = len(df_with_pivot) - future_check_needed
                else:
                    # 데이터가 적으면 비율로 조정 (최소 3개는 앞쪽, 5개는 뒤쪽)
                    start_idx = max(3, int(len(df_with_pivot) * 0.05))
                    end_idx = len(df_with_pivot) - max(5, int(len(df_with_pivot) * 0.05))
                
                # 고점에서 매도해서 성공한 패턴 추출
                for i in range(start_idx, end_idx):
                    if df_with_pivot.iloc[i]['pivot_high'] == 1:
                        entry_price = df_with_pivot.iloc[i]['high']
                        entry_candle = df_with_pivot.iloc[i]
                        
                        # 이후 10개 캔들 중 최대 손익 확인 (매도는 가격 하락이 수익)
                        future_candles = df_with_pivot.iloc[i+1:i+11]
                        if len(future_candles) > 0:
                            min_price = future_candles['low'].min()
                            max_profit_pct = (entry_price - min_price) / entry_price if entry_price > 0 else 0
                            
                            # 2% 이상 수익 발생한 경우 성공 패턴으로 저장
                            if max_profit_pct >= 0.02:
                                pattern = {
                                    'rsi': entry_candle.get('rsi', 50.0),
                                    'macd': entry_candle.get('macd', 0.0),
                                    'macd_signal': entry_candle.get('macd_signal', 0.0),
                                    'volume_ratio': entry_candle.get('volume_ratio', 1.0),
                                    'mfi': entry_candle.get('mfi', 50.0),
                                    'atr': entry_candle.get('atr', 0.02),
                                    'profit_pct': max_profit_pct
                                }
                                successful_sell_patterns.append(pattern)
                
                if successful_sell_patterns:
                    logger.info(f"  ✅ {coin} {interval} 성공 매도 패턴 {len(successful_sell_patterns)}개 발견")
                else:
                    logger.debug(f"  ⚠️ {coin} {interval} 성공 매도 패턴 없음 (기본 범위 사용)")
            except Exception as e:
                logger.debug(f"  ⚠️ {coin} {interval} 성공 패턴 추출 실패: {e}")
        
        # 성공 패턴 기반 파라미터 생성 (Instance-based Imitation) - 매도 전략
        if successful_sell_patterns:
            logger.info(f"  🧬 {coin} {interval}: {len(successful_sell_patterns)}개의 성공 매도 패턴을 기반으로 정밀 전략 생성")
        
        for i in range(n_per_direction):
            # 🆕 성공 패턴 기반 파라미터 생성
            if successful_sell_patterns:
                # 1. 성공했던 케이스 중 하나를 무작위 선택 (Template)
                target_pattern = random.choice(successful_sell_patterns)
                
                # 2. 해당 케이스의 지표 값을 기준으로 좁은 탐색 범위 설정
                # RSI: 타겟 값 주변 ±3~7 범위
                center_rsi = target_pattern.get('rsi', 50)
                rsi_span = random.uniform(3, 7)
                rsi_min = max(10, center_rsi - rsi_span)
                rsi_max = min(90, center_rsi + rsi_span)
                
                # Volume: 타겟 값 주변 ±15% 범위
                center_vol = target_pattern.get('volume_ratio', 1.0)
                vol_span_ratio = random.uniform(0.1, 0.2)
                volume_min = max(0.3, center_vol * (1 - vol_span_ratio))
                volume_max = min(5.0, center_vol * (1 + vol_span_ratio))
                
                # MACD: 타겟 값 주변 미세 조정
                center_macd = target_pattern.get('macd', 0.0)
                macd_span = 0.0005
                macd_sell = center_macd + random.uniform(-macd_span, macd_span)
                macd_buy = 0.0 # 매도 전략에서 macd_buy_threshold는 청산용이거나 미사용
            else:
                # 성공 패턴이 없으면 기본 범위 사용 (기존 로직 유지)
                rsi_high_range_min = max(rsi_mean + rsi_std, rsi_min_actual * 0.5)
                rsi_min = random.uniform(rsi_high_range_min, rsi_mean + rsi_std)
                rsi_max = random.uniform(max(rsi_mean + rsi_std * 1.5, rsi_max_actual * 0.8), min(90, rsi_max_actual))
                
                volume_surge_min = max(volume_mean + volume_std, volume_min_actual * 1.5)
                volume_min = random.uniform(volume_surge_min, volume_max_actual * 0.9)
                volume_max = random.uniform(volume_min * 1.2, min(volume_max_actual, volume_mean + volume_std * 3))
                
                macd_sell_range_min = max(macd_mean + macd_std, macd_min_actual)
                macd_sell_range_max = min(macd_max_actual, macd_mean + macd_std * 2)
                macd_sell = random.uniform(macd_sell_range_min, macd_sell_range_max)
                macd_buy = random.uniform(macd_min_actual, min(macd_mean + macd_std, macd_max_actual))
            
            # 공격적 손절, 보수적 익절
            stop_loss = random.uniform(0.02, 0.04)
            take_profit = random.uniform(0.03, 0.06)
            
            # 🆕 성공 패턴 기반 전략 메타데이터
            strategy_metadata = {
                'success_pattern_based': len(successful_sell_patterns) > 0,
                'success_pattern_count': len(successful_sell_patterns),
                'entry_filter_type': 'high_point_detection'
            }
            
            strategy = Strategy(
                id=f"{coin}_{interval}_sell_specialized_{i:04d}",
                params={
                    'rsi_min': rsi_min,
                    'rsi_max': rsi_max,
                    'volume_ratio_min': volume_min,
                    'volume_ratio_max': volume_max,
                    'stop_loss_pct': stop_loss,
                    'take_profit_pct': take_profit,
                    'macd_buy_threshold': macd_buy,
                    'macd_sell_threshold': macd_sell,
                },
                version="v2.0",
                coin=coin,
                interval=interval,
                created_at=datetime.now(),
                strategy_type="sell_specialized",
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                volume_ratio_min=volume_min,
                volume_ratio_max=volume_max,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                macd_buy_threshold=macd_buy,
                macd_sell_threshold=macd_sell,
                rsi_condition={'min': rsi_min, 'max': rsi_max},
                volume_condition={'min': volume_min, 'max': volume_max},
                atr_condition={'min': max(0.01, atr_min_actual if has_real_data else 0.01), 
                             'max': min(0.05, atr_max_actual if has_real_data else 0.05)},
                pattern_source='direction_specialized',
                enhancement_type='sell_optimized',
                metadata=strategy_metadata
            )
            strategies_by_direction['SELL'].append(strategy)
        
        # 3. 홀드 특화 전략 (생성하지 않음 - 관망은 매매 신호 부재의 결과여야 함)
        # logger.info(f"⚖️ {coin} {interval} 홀드 특화 전략 생성 건너뜀 (관망 전략 비활성화)")
        strategies_by_direction['HOLD'] = []
        
        total = sum(len(v) for v in strategies_by_direction.values())
        logger.info(f"✅ 방향성별 특화 전략 생성 완료: 총 {total}개 (BUY:{len(strategies_by_direction['BUY'])}, SELL:{len(strategies_by_direction['SELL'])})")
        return strategies_by_direction
        
    except Exception as e:
        logger.error(f"❌ 방향성별 특화 전략 생성 실패: {e}")
        return {'BUY': [], 'SELL': [], 'HOLD': []}


def create_enhanced_strategies_with_diversity(coin: str, interval: str, df: Any,
                                             total_count: int, seed: int = None) -> List[Strategy]:
    """다양성을 확보한 종합 전략 생성"""
    try:
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        all_strategies = []
        
        # 1. 그리드 서치 전략 (30%)
        grid_count = int(total_count * 0.3)
        grid_strategies = create_grid_search_strategies(coin, interval, df, grid_count, seed)
        all_strategies.extend(grid_strategies)
        logger.info(f"✅ 그리드 서치: {len(grid_strategies)}개")
        
        # 2. 방향성별 특화 전략 (40% - 각 방향성 20%)
        direction_count = int(total_count * 0.2)
        direction_strategies = create_direction_specialized_strategies(coin, interval, df, direction_count)
        all_strategies.extend(direction_strategies['BUY'])
        all_strategies.extend(direction_strategies['SELL'])
        # HOLD 전략은 추가하지 않음
        logger.info(f"✅ 방향성별 특화: {sum(len(v) for v in direction_strategies.values())}개")
        
        # 3. 기존 지능형 전략 (30%) - create_intelligent_strategies 호출은 별도로
        
        # 중복 제거
        unique_strategies = filter_duplicate_strategies(all_strategies)
        
        logger.info(f"✅ 종합 전략 생성 완료: {len(all_strategies)}개 생성 → {len(unique_strategies)}개 고유 전략")
        return unique_strategies
        
    except Exception as e:
        logger.error(f"❌ 종합 전략 생성 실패: {e}")
        return []

