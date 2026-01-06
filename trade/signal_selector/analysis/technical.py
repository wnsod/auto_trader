"""
technical 관련 Mixin 클래스
SignalSelector의 technical 기능을 담당합니다.
"""



# === 공통 import ===
import os
import sys
import logging
import traceback
import time
import json
import math
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

# signal_selector 내부 모듈
try:
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )
except ImportError:
    # 직접 실행 시 경로 추가
    _current = os.path.dirname(os.path.abspath(__file__))
    _signal_selector = os.path.dirname(_current)
    _trade = os.path.dirname(_signal_selector)
    sys.path.insert(0, _trade)
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, USE_GPU_ACCELERATION, AI_MODEL_AVAILABLE,
        SYNERGY_LEARNING_AVAILABLE, PERFORMANCE_CONFIG, CROSS_COIN_AVAILABLE,
        ENABLE_CROSS_COIN_LEARNING, workspace_dir
    )
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )

# 헬퍼 클래스 import (core에서만 필요)
try:
    from signal_selector.helpers import (
        ContextualBandit, RegimeChangeDetector, ExponentialDecayWeight,
        BayesianSmoothing, ActionSpecificScorer, ContextFeatureExtractor,
        OutlierGuardrail, EvolutionEngine, ContextMemory, RealTimeLearner,
        SignalTradeConnector
    )
except ImportError:
    pass  # 헬퍼가 필요없는 Mixin에서는 무시


class TechnicalAnalysisMixin:
    """
    TechnicalAnalysisMixin - technical 기능

    이 Mixin은 SignalSelector 클래스에서 상속받아 사용됩니다.
    """

    def _calculate_smart_indicators(self, candle: pd.Series, coin: str, interval: str, verbose: bool = True) -> Dict:
        """🚀 실제 캔들 DB의 풍부한 기술적 지표를 활용한 스마트 지표 계산
        
        Args:
            verbose: True면 로그 출력, False면 로그 생략 (중복 방지)
        """
        try:
            # 🚀 1. 실제 캔들 데이터에서 직접 지표 추출 (realtime_candles 파일들에서 계산된 값들)
            indicators = {}
            
            # 🚀 기본 OHLCV 데이터
            indicators['open'] = candle.get('open', 100.0)
            indicators['high'] = candle.get('high', 101.0)
            indicators['low'] = candle.get('low', 99.0)
            indicators['close'] = candle.get('close', 100.0)
            indicators['volume'] = candle.get('volume', 1000.0)
            
            # 🚀 오실레이터 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['rsi'] = candle.get('rsi', 50.0)
            indicators['mfi'] = candle.get('mfi', 50.0)
            
            # 🚀 트렌드 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['macd'] = candle.get('macd', 0.0)
            indicators['macd_signal'] = candle.get('macd_signal', 0.0)
            
            # 🚀 볼린저밴드 (realtime_candles_calculate.py에서 계산됨)
            indicators['bb_upper'] = candle.get('bb_upper', 1.05)
            indicators['bb_middle'] = candle.get('bb_middle', 1.0)
            indicators['bb_lower'] = candle.get('bb_lower', 0.95)
            
            # 🚀 변동성/추세 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['atr'] = candle.get('atr', 0.02)
            indicators['ma20'] = candle.get('ma20', 1.0)
            indicators['adx'] = candle.get('adx', 25.0)
            
            # 🚀 거래량 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['volume_ratio'] = candle.get('volume_ratio', 1.0)
            
            # 🚀 리스크 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['volatility'] = candle.get('volatility', 0.02)
            indicators['risk_score'] = candle.get('risk_score', 0.5)
            
            # 🚀 파동 분석 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['wave_phase'] = candle.get('wave_phase', 'unknown')
            indicators['confidence'] = candle.get('confidence', 0.5)
            indicators['zigzag_direction'] = candle.get('zigzag_direction', 0.0)
            indicators['zigzag_pivot_price'] = candle.get('zigzag_pivot_price', 100.0)
            indicators['wave_progress'] = candle.get('wave_progress', 0.5)
            
            # 🚀 패턴 분석 지표 (realtime_candles_calculate.py에서 계산됨)
            indicators['pattern_type'] = candle.get('pattern_type', 'none')
            indicators['pattern_confidence'] = candle.get('pattern_confidence', 0.0)
            
            # 🚀 통합 분석 지표 (realtime_candles_integrated.py에서 계산됨)
            indicators['volatility_level'] = candle.get('volatility_level', 'medium')
            indicators['risk_level'] = candle.get('risk_level', 'medium')
            indicators['integrated_direction'] = candle.get('integrated_direction', 'neutral')
            
            # 🚀 [New] 고급 기술적 분석 지표 (추정 로직 추가)
            # 데이터가 없을 경우 지표 기반으로 추정
            
            # 1. 엘리어트 파동 단계 추정 (향상됨)
            if 'elliott_wave' in candle and candle['elliott_wave'] != 'unknown':
                indicators['elliott_wave'] = candle['elliott_wave']
            else:
                indicators['elliott_wave'] = self._estimate_elliott_wave(
                    indicators['wave_phase'], indicators['macd'], indicators['macd_signal'], indicators['rsi']
                )
                
            # 2. 시장 구조 추정 (향상됨)
            if 'market_structure' in candle and candle['market_structure'] != 'unknown':
                indicators['market_structure'] = candle['market_structure']
            else:
                indicators['market_structure'] = self._analyze_market_structure(
                    indicators['close'], indicators['ma20'], 
                    indicators['bb_upper'], indicators['bb_lower'],
                    indicators['macd'], indicators['adx']
                )

            # 3. 다이버전스 분석 (신규 추가 - 실시간 계산)
            try:
                # 최근 캔들 데이터 로드 (history)
                history_df = self.get_recent_candles(coin, interval, limit=30)
                
                if not history_df.empty:
                    # 데이터프레임에 현재 캔들 정보가 최신인지 확인하고 아니면 추가/갱신 필요할 수 있음
                    # 여기서는 DB에서 가져온 최신 데이터를 사용하므로 신뢰
                    
                    # RSI 다이버전스
                    indicators['rsi_divergence'] = self.calculate_divergence(history_df, 'rsi', 'close')
                    
                    # MACD 다이버전스
                    indicators['macd_divergence'] = self.calculate_divergence(history_df, 'macd', 'close')
                    
                    # 거래량 다이버전스 (가격 상승 + 거래량 하락 등)
                    # indicators['volume_divergence'] = self.calculate_divergence(history_df, 'volume_ratio', 'close')
                else:
                     indicators['rsi_divergence'] = 'none'
                     indicators['macd_divergence'] = 'none'

            except Exception as div_err:
                print(f"⚠️ 실시간 다이버전스 계산 실패: {div_err}")
                indicators['rsi_divergence'] = 'none'
                indicators['macd_divergence'] = 'none'

            # 4. 패턴 타입이 unknown인 경우 추정 시도
            if indicators['pattern_type'] == 'unknown' or indicators['pattern_type'] == 'none':
                # 간단한 추세 패턴 추정
                if indicators['ma20'] < indicators['close']:
                    indicators['pattern_type'] = 'uptrend'
                elif indicators['ma20'] > indicators['close']:
                    indicators['pattern_type'] = 'downtrend'
            
            # 🚀 추가 계산된 지표들 (None 값 안전 처리)
            try:
                indicators['price_change'] = (indicators['close'] - indicators['open']) / indicators['open']
                
                # 🆕 [Fix] 모멘텀이 0.0인 경우 현재 캔들 등락률로 대체 (실시간성 확보)
                if indicators.get('price_momentum', 0.0) == 0.0:
                    indicators['price_momentum'] = indicators['price_change'] * 100  # 퍼센트 단위가 아니라 배율일 수 있으므로 확인 필요하지만 보통 그대로 씀
                    
                # 🆕 [Fix] ADX가 100.0(Max)이거나 0.0인 경우 보정 (데이터 부족/오류 방지)
                # ADX는 보통 0~100 사이지만, 100이 계속 나오는건 오류일 가능성 높음
                if indicators['adx'] >= 99.0 or indicators['adx'] <= 0.1:
                    # 변동성(ATR/Close) 기반으로 ADX 추정 (변동성이 크면 추세 강도도 높다고 가정)
                    # ATR(0.02) -> ADX(40) 정도 매핑
                    est_adx = 20.0 + (indicators['volatility'] * 1000)
                    indicators['adx'] = min(80.0, max(10.0, est_adx))
                    
            except (TypeError, ZeroDivisionError):
                indicators['price_change'] = 0.0
                indicators['price_momentum'] = 0.0
            
            try:
                indicators['high_low_ratio'] = (indicators['high'] - indicators['low']) / indicators['low']
            except (TypeError, ZeroDivisionError):
                indicators['high_low_ratio'] = 0.0
            
            try:
                indicators['close_to_bb_upper'] = (indicators['close'] - indicators['bb_upper']) / indicators['bb_upper']
            except (TypeError, ZeroDivisionError):
                indicators['close_to_bb_upper'] = 0.0
            
            try:
                indicators['close_to_bb_lower'] = (indicators['close'] - indicators['bb_lower']) / indicators['bb_lower']
            except (TypeError, ZeroDivisionError):
                indicators['close_to_bb_lower'] = 0.0
            
            try:
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            except TypeError:
                indicators['macd_histogram'] = 0.0
            
            # 🚀 실제 데이터 활용 로그 (verbose=True일 때만 출력하여 중복 방지)
            if verbose:
                rsi_log = indicators['rsi'] if indicators['rsi'] is not None else 50.0
                macd_log = indicators['macd'] if indicators['macd'] is not None else 0.0
                volume_log = indicators['volume_ratio'] if indicators['volume_ratio'] is not None else 1.0
                wave_log = indicators['wave_phase'] if indicators['wave_phase'] is not None else 'unknown'
                pattern_log = indicators['pattern_type'] if indicators['pattern_type'] is not None else 'none'
                direction_log = indicators['integrated_direction'] if indicators['integrated_direction'] is not None else 'neutral'
                
                print(f"📊 {coin}/{interval}: 실제 기술지표 활용 - RSI({rsi_log:.1f}), MACD({macd_log:.4f}), Volume({volume_log:.2f}x), Wave({wave_log}), Pattern({pattern_log}), Direction({direction_log})")
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ 스마트 지표 계산 실패: {e}")
            # 🚀 오류 시 기본 지표 반환
            return self._calculate_fast_indicators(candle)

    def _estimate_elliott_wave(self, wave_phase: str, macd: float, macd_signal: float, rsi: float = 50.0) -> str:
        """엘리어트 파동 단계 추정 (RSI 추가 활용)"""
        try:
            if wave_phase == 'impulse':
                if macd > macd_signal and macd > 0:
                    if rsi > 70:
                        return 'Wave 3 (Strong Impulse)'  # 과매수권 강력한 상승
                    return 'Wave 3 (Impulse)'
                elif macd > 0:
                    if rsi < 60 and macd < macd_signal:
                        return 'Wave 4 (Correction)' # 상승 중 조정
                    return 'Wave 5 (Ending Impulse)'
                else:
                    return 'Wave 1 (Starting Impulse)'
            elif wave_phase == 'correction':
                if macd < macd_signal and macd < 0:
                    if rsi < 30:
                        return 'Wave C (Strong Correction)' # 과매도권 강력한 하락
                    return 'Wave C (Correction)'
                elif macd < 0:
                     if rsi > 40 and macd > macd_signal:
                        return 'Wave B (Rebound)' # 하락 중 반등
                     return 'Wave A (Initial Correction)'
                else:
                    return 'Wave B (Rebound)'
            elif wave_phase == 'consolidation':
                return 'Sideways / Consolidation'
            else:
                # MACD & RSI 기반 정교한 추정
                if macd > 0 and macd > macd_signal:
                    if rsi > 60:
                        return 'Impulsive Move (Strong)'
                    return 'Impulsive Move'
                elif macd < 0 and macd < macd_signal:
                    if rsi < 40:
                        return 'Corrective Move (Strong)'
                    return 'Corrective Move'
                else:
                    return 'Unknown Phase'
        except Exception:
            return 'Unknown Phase'

    def _analyze_market_structure(self, close: float, ma20: float, bb_upper: float, bb_lower: float, macd: float, adx: float = 25.0) -> str:
        """시장 구조 분석 (ADX 활용하여 추세 강도 판단 추가)"""
        try:
            # 강한 추세 판단 (ADX > 30)
            is_strong_trend = adx > 30

            # 강한 상승 추세
            if close > bb_upper and macd > 0:
                return 'Strong Bullish Trend' if is_strong_trend else 'Bullish Overextended'
            # 상승 추세
            elif close > ma20 and macd > 0:
                return 'Bullish Structure'
            # 강한 하락 추세
            elif close < bb_lower and macd < 0:
                return 'Strong Bearish Trend' if is_strong_trend else 'Bearish Overextended'
            # 하락 추세
            elif close < ma20 and macd < 0:
                return 'Bearish Structure'
            # 박스권
            else:
                if adx < 20:
                    return 'Ranging (Weak Trend)'
                return 'Ranging / Consolidation'
        except Exception:
            return 'Unknown Structure'

    def _match_dna_pattern(self, current_dna: dict, pattern: dict) -> bool:
        """DNA 패턴 매칭 (크로스 코인 학습용)"""
        try:
            match_count = 0
            total_count = 0
            
            for key, value in pattern.items():
                if key in current_dna:
                    total_count += 1
                    if current_dna[key] == value:
                        match_count += 1
            
            if total_count == 0:
                return False
            
            match_ratio = match_count / total_count
            return match_ratio >= 0.7  # 70% 이상 매칭
            
        except Exception as e:
            print(f"⚠️ DNA 패턴 매칭 실패: {e}")
            return False

    def _calculate_fast_indicators(self, candle: pd.Series) -> Dict:
        """🚀 빠른 기술적 지표 계산 (핵심 지표만)"""
        try:
            indicators = {}
            
            # 🚀 1. 기본 가격 지표 (가장 빠름)
            close = candle.get('close', 0.0)
            open_price = candle.get('open', close)
            high = candle.get('high', close)
            low = candle.get('low', close)
            volume = candle.get('volume', 0.0)
            
            # 🚀 2. 간단한 RSI 계산 (14기간 대신 7기간 사용)
            rsi = self._calculate_fast_rsi(candle)
            indicators['rsi'] = rsi
            
            # 🚀 3. 간단한 MACD 계산
            macd = self._calculate_fast_macd(candle)
            indicators['macd'] = macd
            
            # 🚀 4. 거래량 비율 (간단한 계산)
            volume_ratio = self._calculate_fast_volume_ratio(candle)
            indicators['volume_ratio'] = volume_ratio
            
            # 🚀 5. 변동성 (간단한 계산)
            volatility = self._calculate_fast_volatility(candle)
            indicators['volatility'] = volatility
            
            # 🚀 6. 기본 패턴 정보
            indicators['wave_phase'] = 'unknown'  # 복잡한 계산 생략
            indicators['pattern_type'] = 'unknown'  # 복잡한 계산 생략
            indicators['structure_score'] = 0.5  # 기본값
            indicators['pattern_confidence'] = 0.5  # 기본값
            indicators['integrated_direction'] = 'neutral'  # 기본값
            indicators['integrated_strength'] = 0.5  # 기본값
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ 빠른 지표 계산 실패: {e}")
            return {'rsi': 50.0, 'macd': 0.0, 'volume_ratio': 1.0, 'volatility': 0.02}

    def _calculate_fast_rsi(self, candle: pd.Series) -> float:
        """🚀 빠른 RSI 계산 (7기간)"""
        try:
            # 캔들 데이터가 부족하면 기본값 반환
            if len(candle) < 7:
                return 50.0
            
            # 간단한 가격 변화 계산
            close = candle.get('close', 0.0)
            # 실제로는 더 복잡한 RSI 계산이 필요하지만, 여기서는 간단히 처리
            return 50.0  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 RSI 계산 실패: {e}")
            return 50.0

    def _calculate_fast_macd(self, candle: pd.Series) -> float:
        """🚀 빠른 MACD 계산"""
        try:
            # 간단한 MACD 계산 (실제로는 더 복잡한 계산 필요)
            close = candle.get('close', 0.0)
            return 0.0  # 기본값
            
        except Exception as e:
            print(f"⚠️ 빠른 MACD 계산 실패: {e}")
            return 0.0

    def _calculate_advanced_indicators_with_learning_engine(self, candle: pd.Series, coin: str, interval: str) -> Dict:
        """🚀 고급 지표 계산 (learning_engine.py 학습 결과 활용)"""
        try:
            advanced_indicators = {}
            
            # 🚀 1. 글로벌 학습 점수
            if self.global_learning_manager:
                global_score = self._get_global_learning_score(coin, interval, candle)
                advanced_indicators['global_learning_score'] = global_score
            
            # 🚀 2. 심볼별 튜닝 점수
            if self.symbol_finetuning_manager:
                symbol_score = self._get_symbol_tuning_score(coin, interval, candle)
                advanced_indicators['symbol_tuning_score'] = symbol_score
            
            # 🚀 3. 시너지 학습 점수
            if self.synergy_learner:
                synergy_score = self._get_synergy_learning_score(coin, interval, candle)
                advanced_indicators['synergy_learning_score'] = synergy_score
            
            return advanced_indicators
            
        except Exception as e:
            print(f"⚠️ 고급 지표 계산 실패: {e}")
            return {}

    def _calculate_representative_dna_pattern(self, patterns: list) -> dict:
        """
        여러 DNA 패턴에서 대표 패턴 계산 (최빈값 기반)

        Args:
            patterns: DNA 패턴 리스트

        Returns:
            대표 DNA 패턴 딕셔너리
        """
        from collections import Counter

        representative = {}

        # 각 지표별로 최빈값 계산
        for key in ['rsi_range', 'macd_range', 'volume_range', 'volatility_range',
                    'structure_range', 'wave_step', 'pattern_quality']:
            values = [p.get(key, 'unknown') for p in patterns if key in p]
            if values:
                # 최빈값 선택
                counter = Counter(values)
                representative[key] = counter.most_common(1)[0][0]
            else:
                representative[key] = 'unknown'

        # 인터벌은 첫 번째 패턴의 값 사용 (모두 동일해야 함)
        if patterns:
            representative['interval'] = patterns[0].get('interval', '15m')

        return representative

    def refresh_dna_patterns_if_needed(self, force: bool = False):
        """
        🧬 필요 시 DNA 패턴 자동 갱신

        Args:
            force: True면 시간 체크 없이 강제 갱신

        DNA 패턴을 1시간마다 자동 갱신하여 최신 학습 데이터 반영
        """
        current_time = time.time()
        update_interval = 3600  # 1시간 (초 단위)

        if force or (current_time - self.last_dna_update > update_interval):
            print(f"\n🔄 DNA 패턴 갱신 시작 (마지막 업데이트: {int((current_time - self.last_dna_update) / 60)}분 전)")
            self.load_dna_patterns_from_learning_data()
            print(f"✅ DNA 패턴 갱신 완료")

    def _get_synergy_pattern_bonus(self, coin: str, interval: str, candle: pd.Series) -> float:
        """시너지 패턴 기반 보너스 점수"""
        try:
            # 🆕 시너지 패턴 매니저에서 패턴 로드
            synergy_patterns = self._load_synergy_patterns()
            if not synergy_patterns:
                return 0.0
            
            # 현재 시장 조건에 맞는 시너지 패턴 찾기
            market_condition = self._detect_current_market_condition(coin, interval)
            synergy_bonus = 0.0
            
            if market_condition in synergy_patterns:
                pattern = synergy_patterns[market_condition]
                synergy_bonus = pattern.get('synergy_score', 0.0) * 0.1  # 10% 보너스
            
            return min(synergy_bonus, 0.15)  # 최대 0.15 보너스
            
        except Exception as e:
            logger.error(f"❌ 시너지 패턴 보너스 계산 실패: {e}")
            return 0.0
    
    def _analyze_dna_history_for_realtime(self, history_rows: List[tuple], coin: str = None) -> Dict[str, Any]:
        """DNA 히스토리 데이터를 실시간 분석에 활용"""
        try:
            features = {}
            
            # 필드별 데이터 그룹화
            field_data = {}
            for row in history_rows:
                if coin:
                    field, mean, std, q25, q50, q75, count, interval_focus, created_at = row
                else:
                    coin_name, field, mean, std, q25, q50, q75, count, interval_focus, created_at = row
                
                if field not in field_data:
                    field_data[field] = []
                
                field_data[field].append({
                    'mean': mean, 'std': std, 'q25': q25, 'q50': q50, 'q75': q75,
                    'count': count, 'interval_focus': interval_focus, 'created_at': created_at
                })
            
            # 실시간 분석에 유용한 특성들 추출
            for field, data_list in field_data.items():
                if len(data_list) >= 2:
                    # 시간순 정렬
                    data_list.sort(key=lambda x: x['created_at'])
                    
                    # 최신 vs 이전 비교 (실시간 변화 감지)
                    latest = data_list[-1]
                    previous = data_list[-2] if len(data_list) > 1 else latest
                    
                    # 변화율 계산
                    mean_change = (latest['mean'] - previous['mean']) / max(abs(previous['mean']), 1e-6)
                    
                    # 실시간 시그널에 활용할 특성들
                    features[f'{field}_momentum'] = mean_change  # 모멘텀 지표
                    features[f'{field}_stability'] = 1.0 - min(abs(mean_change), 1.0)  # 안정성 지표
                    features[f'{field}_current_level'] = latest['mean']  # 현재 수준
                    features[f'{field}_volatility'] = latest['std']  # 변동성
                    
                    # 분위수 정보 (실시간 신호 강도 판단용)
                    features[f'{field}_q25'] = latest['q25']
                    features[f'{field}_q75'] = latest['q75']
                    features[f'{field}_range'] = latest['q75'] - latest['q25']  # 범위
            
            # 전체적인 DNA 패턴 분석 (실시간 신호 품질 판단용)
            if len(history_rows) >= 3:
                features['dna_pattern_consistency'] = self._calculate_dna_pattern_consistency(field_data)
                features['dna_signal_strength'] = self._calculate_dna_signal_strength(field_data)
                features['dna_market_adaptation'] = self._calculate_dna_market_adaptation(field_data)
            
            return features
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ DNA 히스토리 실시간 분석 실패: {e}")
            return {}
    
    def _calculate_dna_pattern_consistency(self, field_data: Dict[str, List[Dict]]) -> float:
        """DNA 패턴 일관성 계산 (실시간 신호 신뢰도용)"""
        try:
            consistency_scores = []
            
            for field, data_list in field_data.items():
                if len(data_list) >= 3:
                    # 최근 3개 데이터 포인트의 일관성 계산
                    recent_data = data_list[-3:]
                    values = [d['mean'] for d in recent_data]
                    
                    if len(values) >= 2:
                        # 값들의 변화율 계산
                        changes = []
                        for i in range(1, len(values)):
                            change = abs(values[i] - values[i-1]) / max(abs(values[i-1]), 1e-6)
                            changes.append(change)
                        
                        if changes:
                            avg_change = sum(changes) / len(changes)
                            consistency = 1.0 - min(avg_change, 1.0)
                            consistency_scores.append(consistency)
            
            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            return 0.5
    
    def _calculate_dna_signal_strength(self, field_data: Dict[str, List[Dict]]) -> float:
        """DNA 신호 강도 계산 (실시간 신호 강도 판단용)"""
        try:
            strength_scores = []
            
            for field, data_list in field_data.items():
                if data_list:
                    latest = data_list[-1]
                    # 표준편차가 작을수록 강한 신호
                    signal_strength = 1.0 - min(latest['std'] / max(abs(latest['mean']), 1e-6), 1.0)
                    strength_scores.append(signal_strength)
            
            return sum(strength_scores) / len(strength_scores) if strength_scores else 0.5
            
        except Exception as e:
            return 0.5
    
    def _calculate_dna_market_adaptation(self, field_data: Dict[str, List[Dict]]) -> float:
        """DNA 시장 적응성 계산 (실시간 시장 적응도 판단용)"""
        try:
            adaptation_scores = []
            
            for field, data_list in field_data.items():
                if len(data_list) >= 2:
                    # 최근 데이터의 적응성 계산
                    recent_data = data_list[-2:]
                    adaptation = 1.0 - abs(recent_data[-1]['mean'] - recent_data[-2]['mean']) / max(abs(recent_data[-2]['mean']), 1e-6)
                    adaptation_scores.append(max(0.0, adaptation))
            
            return sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0.5
            
        except Exception as e:
            return 0.5
        """심화 통합 분석 결과 로드"""
        try:
            # 🆕 학습 엔진의 심화 분석 결과를 DB에서 로드
            db_path = STRATEGIES_DB_PATH
            if not os.path.exists(db_path):
                return None
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 여부 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_pipeline_results'")
                if not cursor.fetchone():
                    if self.debug_mode:
                        print("ℹ️ learning_pipeline_results 테이블이 존재하지 않습니다.")
                    return None
                
                cursor.execute("""
                    SELECT deep_analysis_result 
                    FROM learning_pipeline_results 
                    WHERE deep_analysis_result IS NOT NULL
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 심화 분석 결과 로드 실패: {e}")
            return None
    
    def get_dna_based_similar_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """DNA 기반 유사 코인 점수 계산 (240분 인터벌 우선 시스템 적용)"""
        try:
            # 🧬 현재 코인의 DNA 패턴 추출 (240분 우선 방식)
            current_dna = self._extract_current_dna_pattern_enhanced(coin, interval, candle)
            
            # 🧬 유사한 DNA를 가진 다른 코인들의 성과 점수 조회
            similar_scores = self._get_similar_dna_scores_enhanced(current_dna, coin, interval)
            
            if not similar_scores:
                # 🚀 유사한 DNA가 없으면 현재 코인의 기술적 지표 기반 점수 계산
                # 거래 이력이 쌓이면 자동으로 DNA 패턴 학습됨 (정상)
                return self._calculate_technical_based_score(candle, verbose=False)
            
            # 🧬 인터벌별 가중치 적용 (240분 우선) - 개선된 버전
            interval_weights = {
                '240m': 2.5,  # 240분: 가장 높은 가중치 (증가)
                '15m': 2.0,   # 15분: 두 번째 높은 가중치 (증가)
                '30m': 1.5,   # 30분: 보통 가중치 (증가)
                '1d': 1.2     # 1일: 기본 가중치 상향
            }
            
            # 🧬 가중 유사도 점수 계산 (개선된 버전)
            total_weight = 0.0
            weighted_sum = 0.0
            
            for similar_coin, similarity, score, similar_interval in similar_scores:
                # 인터벌 가중치 적용
                interval_weight = interval_weights.get(similar_interval, 1.0)
                
                # 🆕 유사도 보너스 (높은 유사도에 더 높은 가중치)
                similarity_bonus = 1.0
                if similarity > 0.8:
                    similarity_bonus = 1.5  # 매우 유사한 경우 50% 보너스
                elif similarity > 0.6:
                    similarity_bonus = 1.3  # 유사한 경우 30% 보너스
                elif similarity > 0.4:
                    similarity_bonus = 1.1  # 약간 유사한 경우 10% 보너스
                
                # 🆕 성과 점수 보너스 (높은 성과에 더 높은 가중치)
                performance_bonus = 1.0
                if score > 0.1:
                    performance_bonus = 1.4  # 높은 성과 40% 보너스
                elif score > 0.05:
                    performance_bonus = 1.2  # 중간 성과 20% 보너스
                
                combined_weight = similarity * interval_weight * similarity_bonus * performance_bonus * 0.6 + 0.4  # 최소 40% 가중치 보장
                
                weighted_sum += score * combined_weight
                total_weight += combined_weight
            
            if total_weight > 0:
                # 🧬 기본 DNA 점수 계산
                base_dna_score = weighted_sum / total_weight
                
                # 🆕 DNA 히스토리 데이터 로드 (실시간 분석용)
                dna_history_features = self._load_dna_analysis_results(coin)
                
                # 🆕 DNA 히스토리 기반 보정 (실시간 분석)
                history_bonus = 0.0
                if dna_history_features:
                    # 패턴 일관성 보너스
                    consistency = dna_history_features.get('dna_pattern_consistency', 0.5)
                    consistency_bonus = consistency * 0.1  # 최대 10% 보너스
                    
                    # 신호 강도 보너스
                    signal_strength = dna_history_features.get('dna_signal_strength', 0.5)
                    strength_bonus = signal_strength * 0.15  # 최대 15% 보너스
                    
                    # 시장 적응성 보너스
                    market_adaptation = dna_history_features.get('dna_market_adaptation', 0.5)
                    adaptation_bonus = market_adaptation * 0.1  # 최대 10% 보너스
                    
                    history_bonus = consistency_bonus + strength_bonus + adaptation_bonus
                    
                    if self.debug_mode:
                        print(f"🧬 {coin}/{interval}: DNA 히스토리 보너스 - 일관성({consistency:.3f}), 강도({signal_strength:.3f}), 적응({market_adaptation:.3f})")
                
                # 🆕 최종 점수 계산 (기본 DNA 점수 + 히스토리 보너스)
                final_score = base_dna_score + history_bonus
                
                # 🆕 최종 점수 보너스 (DNA 기반 점수 강화)
                if final_score > 0.05:
                    final_score *= 1.3  # 높은 DNA 점수에 30% 보너스
                elif final_score > 0.02:
                    final_score *= 1.2  # 중간 DNA 점수에 20% 보너스
                
                if self.debug_mode:
                    print(f"🧬 {coin}/{interval}: DNA 기반 점수 계산 성공 - 유사코인({len(similar_scores)}개), 기본점수({base_dna_score:.3f}), 히스토리보너스({history_bonus:.3f}), 최종점수({final_score:.3f})")
                
                return min(1.0, max(0.0, final_score))
            else:
                return self._calculate_technical_based_score(candle, verbose=False)
                
        except Exception as e:
            print(f"⚠️ DNA 기반 유사 점수 계산 오류 ({coin}/{interval}): {e}")
            return self._calculate_technical_based_score(candle)
    
    def _calculate_technical_based_score(self, candle: pd.Series, verbose: bool = True) -> float:
        """🚀 기술적 지표 기반 점수 계산 (DNA 대체용)"""
        try:
            # 🚀 실제 캔들 데이터에서 지표 추출 (None 값 안전 처리)
            rsi = candle.get('rsi', 50.0)
            macd = candle.get('macd', 0.0)
            volume_ratio = candle.get('volume_ratio', 1.0)
            volatility = candle.get('volatility', 0.02)
            wave_phase = candle.get('wave_phase', 'unknown')
            pattern_confidence = candle.get('pattern_confidence', 0.0)
            integrated_direction = candle.get('integrated_direction', 'neutral')
            
            # None 값 안전 처리
            if rsi is None:
                rsi = 50.0
            if macd is None:
                macd = 0.0
            if volume_ratio is None:
                volume_ratio = 1.0
            if volatility is None:
                volatility = 0.02
            if pattern_confidence is None:
                pattern_confidence = 0.0
            
            # 🚀 RSI 기반 점수 (0.0 ~ 1.0)
            if rsi < 20:  # 극도 과매도
                rsi_score = 0.9
            elif rsi < 30:  # 과매도
                rsi_score = 0.7
            elif rsi > 80:  # 극도 과매수
                rsi_score = 0.1
            elif rsi > 70:  # 과매수
                rsi_score = 0.3
            else:  # 중립
                rsi_score = 0.5
            
            # 🚀 MACD 기반 점수 (0.0 ~ 1.0)
            if macd > 0.01:  # 강한 상승 신호
                macd_score = 0.9
            elif macd > 0:  # 약한 상승 신호
                macd_score = 0.7
            elif macd > -0.01:  # 약한 하락 신호
                macd_score = 0.3
            else:  # 강한 하락 신호
                macd_score = 0.1
            
            # 🚀 거래량 기반 점수 (0.0 ~ 1.0)
            if volume_ratio > 2.0:  # 높은 거래량
                volume_score = 0.8
            elif volume_ratio > 1.0:  # 정상 거래량
                volume_score = 0.6
            else:  # 낮은 거래량
                volume_score = 0.4
            
            # 🚀 파동 단계 기반 점수
            wave_score = 0.5
            if wave_phase == 'impulse':
                wave_score = 0.8
            elif wave_phase == 'correction':
                wave_score = 0.3
            elif wave_phase == 'consolidation':
                wave_score = 0.6
            
            # 🚀 통합 방향성 기반 점수
            direction_score = 0.5
            if integrated_direction == 'strong_bullish':
                direction_score = 0.9
            elif integrated_direction == 'bullish':
                direction_score = 0.7
            elif integrated_direction == 'strong_bearish':
                direction_score = 0.1
            elif integrated_direction == 'bearish':
                direction_score = 0.3
            
            # 🚀 패턴 신뢰도 기반 점수
            pattern_score = 0.5 + (pattern_confidence * 0.5)  # 0.5 ~ 1.0
            
            # 🚀 최종 점수 계산 (가중 평균)
            final_score = (
                rsi_score * 0.25 +
                macd_score * 0.25 +
                volume_score * 0.15 +
                wave_score * 0.15 +
                direction_score * 0.15 +
                pattern_score * 0.05
            )
            
            if verbose:
                print(f"🔧 기술적 지표 기반 점수: RSI({rsi:.1f}→{rsi_score:.2f}), MACD({macd:.4f}→{macd_score:.2f}), Volume({volume_ratio:.2f}x→{volume_score:.2f}), 최종({final_score:.3f})")
            
            return np.clip(final_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ 기술적 지표 기반 점수 계산 실패: {e}")
            return 0.3  # 기본값
    
    def _extract_current_dna_pattern_enhanced(self, coin: str, interval: str, candle: pd.Series) -> dict:
        """현재 코인의 DNA 패턴 추출 (240분 우선 방식 적용)"""
        try:
            # 🧬 안전한 값 추출 (None 처리)
            rsi = safe_float(candle.get('rsi'), 50.0)
            macd = safe_float(candle.get('macd'), 0.0)
            volume_ratio = safe_float(candle.get('volume_ratio'), 1.0)
            volatility = safe_float(candle.get('volatility'), 0.0)
            structure_score = safe_float(candle.get('structure_score'), 0.5)
            wave_step = safe_float(candle.get('wave_step'), 0.0)
            pattern_quality = safe_float(candle.get('pattern_quality'), 0.5)
            timestamp = safe_float(candle.get('timestamp'), 0)
            
            # 🧬 핵심 지표들로 DNA 패턴 생성 (더 정교한 범주화)
            dna_pattern = {
                'rsi_range': self._categorize_rsi_enhanced(rsi),
                'macd_range': self._categorize_macd_enhanced(macd),
                'volume_range': self._categorize_volume_enhanced(volume_ratio),
                'volatility_range': self._categorize_volatility_enhanced(volatility),
                'structure_range': self._categorize_structure_enhanced(structure_score),
                'wave_step': self._categorize_wave_step(wave_step),
                'pattern_quality': self._categorize_pattern_quality(pattern_quality),
                'interval': interval,  # 인터벌 정보 추가
                'timestamp': timestamp
            }
            return dna_pattern
            
        except Exception as e:
            print(f"⚠️ DNA 패턴 추출 오류 ({coin}): {e}")
            return {}
    
    def _categorize_rsi_enhanced(self, rsi: float) -> str:
        """RSI 범주화 (더 정교한 분류)"""
        if rsi < 20:
            return 'extreme_oversold'
        elif rsi < 30:
            return 'oversold'
        elif rsi < 40:
            return 'low'
        elif rsi < 60:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        elif rsi < 80:
            return 'overbought'
        else:
            return 'extreme_overbought'
    
    def _categorize_macd_enhanced(self, macd: float) -> str:
        """MACD 범주화 (더 정교한 분류)"""
        if macd < -0.02:
            return 'extreme_bearish'
        elif macd < -0.01:
            return 'strong_bearish'
        elif macd < 0:
            return 'bearish'
        elif macd < 0.01:
            return 'bullish'
        elif macd < 0.02:
            return 'strong_bullish'
        else:
            return 'extreme_bullish'
    
    def _categorize_wave_step(self, wave_step: float) -> str:
        """웨이브 스텝 범주화"""
        # 안전한 값 처리
        wave_step = safe_float(wave_step, 0.0)
        
        if wave_step < 0.2:
            return 'early'
        elif wave_step < 0.5:
            return 'mid'
        elif wave_step < 0.8:
            return 'late'
        else:
            return 'action'
    
    def _categorize_pattern_quality(self, pattern_quality: float) -> str:
        """패턴 품질 범주화"""
        # 안전한 값 처리
        pattern_quality = safe_float(pattern_quality, 0.5)
        
        if pattern_quality < 0.3:
            return 'poor'
        elif pattern_quality < 0.6:
            return 'fair'
        elif pattern_quality < 0.8:
            return 'good'
        else:
            return 'excellent'
    
    # 🧬 기존 함수들 (호환성 유지)
    def _categorize_rsi(self, rsi: float) -> str:
        """RSI 범주화 (기존 호환성 유지)"""
        return self._categorize_rsi_enhanced(rsi)
    
    def _categorize_macd(self, macd: float) -> str:
        """MACD 범주화 (기존 호환성 유지)"""
        return self._categorize_macd_enhanced(macd)
    
    def _get_similar_dna_scores(self, current_dna: dict, exclude_coin: str) -> list:
        """유사한 DNA를 가진 코인들의 점수 조회 (기존 호환성 유지)"""
        # 기존 방식으로 호환성 유지
        try:
            similar_scores = []
            
            for strategy_key, strategy in self.coin_specific_strategies.items():
                if strategy_key.startswith(exclude_coin):
                    continue
                
                similarity = self._calculate_dna_similarity_enhanced(current_dna, strategy)
                
                if similarity > 0.25:
                    coin_name = strategy_key.split('_')[0]
                    interval = strategy_key.split('_')[1]
                    performance_score = self._calculate_performance_score_enhanced(strategy)
                    similar_scores.append((coin_name, similarity, performance_score))
            
            similar_scores.sort(key=lambda x: x[1], reverse=True)
            return similar_scores[:5]
            
        except Exception as e:
            print(f"⚠️ 유사 DNA 점수 조회 오류: {e}")
            return []
    
    def _extract_current_dna_pattern(self, coin: str, interval: str, candle: pd.Series) -> dict:
        """현재 코인의 DNA 패턴 추출 (기존 호환성 유지)"""
        return self._extract_current_dna_pattern_enhanced(coin, interval, candle)
    
    def _get_similar_dna_scores_enhanced(self, current_dna: dict, exclude_coin: str, current_interval: str) -> list:
        """유사한 DNA를 가진 코인들의 점수 조회 (240분 우선 시스템 적용)"""
        try:
            # print(f"🔍 {exclude_coin}/{current_interval}: DNA 유사도 검색 시작")
            # print(f"📊 사용 가능한 코인별 전략 수: {len(self.coin_specific_strategies)}")
            
            if not self.coin_specific_strategies:
                # print(f"❌ {exclude_coin}/{current_interval}: 코인별 전략이 로드되지 않음")
                return []
            
            similar_scores = []
            # available_keys = []  # 🆕 실제 사용 가능한 전략 키 수집 (자기 자신 제외)
            
            # 🧬 DNA 유사도 기반으로 유사한 코인들 찾기
            for strategy_key, strategy in self.coin_specific_strategies.items():
                # 🆕 자기 자신 제외 로직 개선 (정확한 매칭)
                coin_name = strategy_key.split('_')[0]
                if coin_name == exclude_coin:
                    continue  # 자기 자신 제외
                
                # 🆕 사용 가능한 전략 키 수집 (자기 자신 제외)
                # available_keys.append(strategy_key)
                
                # 🧬 DNA 유사도 계산 (향상된 방식)
                similarity = self._calculate_dna_similarity_enhanced(current_dna, strategy)
                
                # 🚨 유사도 임계값 적용 (더 유연하게)
                if similarity > 0.2:  # 30%에서 20%로 낮춤
                    interval = strategy_key.split('_')[1]
                    
                    # 🧬 해당 코인의 최근 성과 점수
                    performance_score = self._calculate_performance_score_enhanced(strategy)
                    
                    similar_scores.append((coin_name, similarity, performance_score, interval))
                    # print(f"✅ 유사 코인 발견: {coin_name}/{interval} (유사도: {similarity:.3f})")
            
            # 🆕 실제 사용 가능한 전략 키 출력 (점수 순으로 정렬하여 상위 5개)
            # if available_keys:
            #     sorted_available_keys = sorted(
            #         available_keys,
            #         key=lambda k: self.coin_specific_strategies[k].get('score', 0.0),
            #         reverse=True
            #     )[:5]
            #     print(f"📋 사용 가능한 전략 키 예시 ({exclude_coin}/{current_interval} 제외, 점수 상위 5개): {sorted_available_keys}")
            # else:
            #     print(f"📋 사용 가능한 전략 키: 없음 (자기 자신만 존재)")
            
            # print(f"📊 {exclude_coin}/{current_interval}: 총 {len(similar_scores)}개 유사 코인 발견")
            
            # 🚨 유사도 순으로 정렬
            similar_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 🚨 동적 필터링 (유사도에 따라 개수 조정)
            filtered_scores = []
            for score in similar_scores:
                # 🚨 유사도가 높을수록 더 많은 코인 선택
                if score[1] >= 0.8 and len(filtered_scores) < 8:  # 매우 유사한 경우
                    filtered_scores.append(score)
                elif score[1] >= 0.6 and len(filtered_scores) < 5:  # 유사한 경우
                    filtered_scores.append(score)
                elif score[1] >= 0.4 and len(filtered_scores) < 3:  # 약간 유사한 경우
                    filtered_scores.append(score)
                elif score[1] >= 0.3 and len(filtered_scores) < 2:  # 최소 유사한 경우
                    filtered_scores.append(score)
            
            # print(f"📊 {exclude_coin}/{current_interval}: 필터링 후 {len(filtered_scores)}개 유사 코인")
            return filtered_scores
            
        except Exception as e:
            print(f"⚠️ 유사 DNA 점수 조회 오류: {e}")
            return []
    
    def _calculate_dna_similarity_enhanced(self, current_dna: dict, strategy: dict) -> float:
        """DNA 유사도 계산 (향상된 방식)"""
        try:
            similarity_score = 0.0
            total_weight = 0.0
            
            # 🧬 각 지표별 가중치 설정
            weights = {
                'rsi_range': 0.25,
                'macd_range': 0.20,
                'volume_range': 0.15,
                'volatility_range': 0.15,
                'structure_range': 0.15,
                'wave_step': 0.05,
                'pattern_quality': 0.05
            }
            
            # 🚨 각 지표별 유사도 계산 (수정된 방식)
            for indicator, weight in weights.items():
                if indicator in current_dna and indicator in strategy:
                    current_value = current_dna[indicator]
                    strategy_value = strategy.get(indicator, 'unknown')
                    
                    # 🚨 정확한 매칭
                    if current_value == strategy_value:
                        similarity_score += weight
                    # 🚨 부분 매칭
                    elif self._is_similar_category(current_value, strategy_value):
                        similarity_score += weight * 0.5
                    
                    total_weight += weight
                else:
                    # 🚨 지표가 없어도 가중치는 추가 (정규화를 위해)
                    total_weight += weight
            
            # 🚨 인터벌 유사도 추가 (240분 우선)
            interval_weight = 0.1
            if 'interval' in current_dna and 'interval' in strategy:
                current_interval = current_dna['interval']
                strategy_interval = strategy['interval']
                
                if current_interval == strategy_interval:
                    similarity_score += 0.15
                elif (current_interval == '240m' and strategy_interval in ['15m', '30m']) or \
                     (strategy_interval == '240m' and current_interval in ['15m', '30m']):
                    similarity_score += 0.08
                
                total_weight += interval_weight
            
            # 🚨 정규화된 유사도 반환 (0.0 ~ 1.0 범위)
            normalized_similarity = similarity_score / total_weight if total_weight > 0 else 0.0
            return min(max(normalized_similarity, 0.0), 1.0)  # 0.0 ~ 1.0 범위로 제한
            
        except Exception as e:
            print(f"⚠️ DNA 유사도 계산 오류: {e}")
            return 0.0
    
    def _extract_signal_pattern(self, signal: SignalInfo) -> str:
        """🆕 시그널 패턴 추출 (None-Safe)"""
        try:
            # RSI 범주화 (안전한 값 처리)
            rsi = getattr(signal, 'rsi', 50.0)
            if rsi is None: rsi = 50.0
            rsi_level = self._discretize_rsi(float(rsi))
            
            # Direction 범주화
            direction = getattr(signal, 'integrated_direction', 'neutral')
            if not direction: direction = 'neutral'
            
            # BB Position 범주화
            bb_position = getattr(signal, 'bb_position', 'unknown')
            if not bb_position: bb_position = 'unknown'
            
            # Volume 범주화 (안전한 값 처리)
            vol = getattr(signal, 'volume_ratio', 1.0)
            if vol is None: vol = 1.0
            volume_level = self._discretize_volume(float(vol))
            
            # 패턴 조합
            pattern = f"{rsi_level}_{direction}_{bb_position}_{volume_level}"
            
            return pattern
            
        except Exception as e:
            print(f"⚠️ 시그널 패턴 추출 오류: {e}")
            return 'unknown_pattern'
    
    def _discretize_rsi(self, rsi: float) -> str:
        """RSI 값을 이산화 (None-Safe)"""
        if rsi is None: return 'neutral'
        try:
            rsi = float(rsi)
            if rsi < 30:
                return 'oversold'
            elif rsi < 45:
                return 'low'
            elif rsi < 55:
                return 'neutral'
            elif rsi < 70:
                return 'high'
            else:
                return 'overbought'
        except:
            return 'neutral'
    
    def extract_signal_pattern_from_state(self, state_key: str) -> str:
        """상태 키에서 시그널 패턴 추출 (Virtual Trading Learner와 동일한 방식)"""
        try:
            # 상태 키 예시: "BTC_5m_neutral_bullish_upper_low_early_neutral_low"
            parts = state_key.split('_')
            
            if len(parts) >= 6:
                # 핵심 시그널 패턴 추출
                # 예: "neutral_bullish_upper_low" 형태로 추출
                pattern_parts = parts[2:6]  # RSI, Direction, BB, Volume 부분
                return "_".join(pattern_parts)
            else:
                return "unknown_pattern"
                
        except Exception as e:
            print(f"⚠️ 상태 키에서 시그널 패턴 추출 오류: {e}")
            return "unknown_pattern"
    
    def _extract_signal_pattern_from_candle(self, candle: pd.Series, coin: str, interval: str) -> str:
        """캔들 데이터에서 시그널 패턴 추출 (피드백용)"""
        try:
            # RSI 범주화 (안전한 값 처리)
            rsi = safe_float(candle.get('rsi'), 50.0)
            if rsi < 30:
                rsi_cat = 'oversold'
            elif rsi > 70:
                rsi_cat = 'overbought'
            else:
                rsi_cat = 'neutral'
            
            # MACD 범주화 (안전한 값 처리)
            macd = safe_float(candle.get('macd'), 0.0)
            if macd > 0.001:
                macd_cat = 'bullish'
            elif macd < -0.001:
                macd_cat = 'bearish'
            else:
                macd_cat = 'neutral'
            
            # 거래량 범주화 (안전한 값 처리)
            volume_ratio = safe_float(candle.get('volume_ratio'), 1.0)
            if volume_ratio > 1.5:
                volume_cat = 'high'
            elif volume_ratio < 0.5:
                volume_cat = 'low'
            else:
                volume_cat = 'normal'
            
            # 변동성 범주화 (안전한 값 처리)
            volatility = safe_float(candle.get('volatility'), 0.02)
            if volatility > 0.05:
                vol_cat = 'high'
            elif volatility < 0.01:
                vol_cat = 'low'
            else:
                vol_cat = 'normal'
            
            # 파동 단계 (안전한 문자열 처리)
            wave_phase = safe_str(candle.get('wave_phase'), 'unknown')
            
            # 패턴 타입 (안전한 문자열 처리)
            pattern_type = safe_str(candle.get('pattern_type'), 'none')
            
            # 통합 방향 (안전한 문자열 처리)
            integrated_direction = safe_str(candle.get('integrated_direction'), 'neutral')
            
            return f"{rsi_cat}_{macd_cat}_{volume_cat}_{vol_cat}_{wave_phase}_{pattern_type}_{integrated_direction}"
            
        except Exception as e:
            print(f"⚠️ 캔들에서 시그널 패턴 추출 오류: {e}")
            return "unknown_pattern"
    
    def _evaluate_basic_technical_indicators(self, candle: pd.Series) -> float:
        """기본 기술적 지표 기반 평가 (폴백용)"""
        try:
            score = 0.0
            
            # RSI 기반 평가
            rsi = candle.get('rsi')
            if rsi is not None and not pd.isna(rsi):
                rsi = float(rsi)
                if rsi < 30:  # 과매도 - 매수 기회
                    score += 0.1
                elif rsi > 70:  # 과매수 - 매도 기회
                    score -= 0.1
            
            # MACD 기반 평가
            macd = candle.get('macd')
            if macd is not None and not pd.isna(macd):
                macd = float(macd)
                if macd > 0:  # 상승 신호
                    score += 0.05
                else:  # 하락 신호
                    score -= 0.05
            
            # 거래량 비율 기반 평가
            volume_ratio = candle.get('volume_ratio')
            if volume_ratio is not None and not pd.isna(volume_ratio):
                volume_ratio = float(volume_ratio)
                if volume_ratio > 1.5:  # 거래량 증가
                    score += 0.05
                elif volume_ratio < 0.8:  # 거래량 감소
                    score -= 0.05
            
            return score
            
        except Exception as e:
            print(f"⚠️ 기본 기술적 지표 평가 오류: {e}")
            return 0.0
    
    def _check_rsi_condition(self, current_rsi: float, rsi_condition: str) -> bool:
        """RSI 조건 확인"""
        try:
            if not rsi_condition:
                return False
            
            # JSON 형태의 조건 파싱
            import json
            condition = json.loads(rsi_condition) if isinstance(rsi_condition, str) else rsi_condition
            
            min_rsi = condition.get('min', 0)
            max_rsi = condition.get('max', 100)
            
            return min_rsi <= current_rsi <= max_rsi
            
        except Exception as e:
            print(f"⚠️ RSI 조건 확인 오류: {e}")
            return False
    
    def _check_macd_condition(self, current_macd: float, macd_condition: str) -> bool:
        """MACD 조건 확인"""
        try:
            if not macd_condition:
                return False
            
            import json
            condition = json.loads(macd_condition) if isinstance(macd_condition, str) else macd_condition
            
            signal_diff = condition.get('signal_diff', 0)
            
            # MACD가 신호선보다 높은지 확인
            return current_macd > signal_diff
            
        except Exception as e:
            print(f"⚠️ MACD 조건 확인 오류: {e}")
            return False
    
    def _check_wave_step_condition(self, current_wave_step: float, wave_step_condition: str) -> bool:
        """파동 단계 조건 확인"""
        try:
            if not wave_step_condition:
                return False
            
            import json
            condition = json.loads(wave_step_condition) if isinstance(wave_step_condition, str) else wave_step_condition
            
            min_step = condition.get('min', 0)
            max_step = condition.get('max', 100)
            
            return min_step <= current_wave_step <= max_step
            
        except Exception as e:
            print(f"⚠️ 파동 단계 조건 확인 오류: {e}")
            return False
    
    def _check_pattern_quality_condition(self, current_pattern_quality: float, pattern_quality_condition: str) -> bool:
        """패턴 품질 조건 확인"""
        try:
            if not pattern_quality_condition:
                return False
            
            import json
            condition = json.loads(pattern_quality_condition) if isinstance(pattern_quality_condition, str) else pattern_quality_condition
            
            min_quality = condition.get('min', 0)
            
            return current_pattern_quality >= min_quality
            
        except Exception as e:
            print(f"⚠️ 패턴 품질 조건 확인 오류: {e}")
            return False
    
    def _calculate_pattern_quality(self, rsi: float, macd: float, volume_ratio: float, structure_score: float, pattern_confidence: float) -> float:
        """패턴 품질을 다른 지표들을 기반으로 계산"""
        try:
            quality_factors = []
            
            # RSI 기반 품질 (30-70 범위가 좋음)
            if 30 <= rsi <= 70:
                quality_factors.append(0.8)
            elif 20 <= rsi <= 80:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.3)
            
            # MACD 기반 품질 (신호선과의 차이가 적당할 때)
            if abs(macd) < 0.01:
                quality_factors.append(0.7)
            elif abs(macd) < 0.05:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # 거래량 기반 품질 (적당한 거래량이 좋음)
            if 0.8 <= volume_ratio <= 2.0:
                quality_factors.append(0.8)
            elif 0.5 <= volume_ratio <= 3.0:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.4)
            
            # 구조점수 기반 품질
            quality_factors.append(structure_score)
            
            # 패턴 신뢰도 기반 품질
            quality_factors.append(pattern_confidence)
            
            # 평균 품질 계산
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            print(f"⚠️ 패턴 품질 계산 오류: {e}")
            return 0.5
    
    def _find_peaks_or_troughs(self, series: pd.Series, is_trough: bool = False, sensitivity: float = 0.0) -> List[Tuple[int, float]]:
        """
        시리즈에서 고점 또는 저점을 찾음 (민감도 조절 가능)
        
        Args:
            series: 판다스 시리즈 데이터
            is_trough: True면 저점, False면 고점 찾기
            sensitivity: 변화율 민감도 (노이즈 필터링용)
            
        Returns:
            (인덱스, 값) 튜플의 리스트
        """
        try:
            peaks = []
            values = series.values
            indices = series.index
            
            # 최소 3개 포인트 필요
            if len(values) < 3:
                return []
                
            for i in range(1, len(values) - 1):
                current_val = values[i]
                prev_val = values[i-1]
                next_val = values[i+1]
                
                if is_trough:
                    # 저점 조건: 이전값보다 작거나 같고, 다음값보다 작아야 함 (V자 형태)
                    # 민감도 적용: 주변 값보다 sensitivity 비율 이상 낮아야 유효한 저점으로 인정
                    if current_val <= prev_val and current_val < next_val:
                        if sensitivity > 0:
                            # 주변 값과의 차이가 민감도 이상이어야 함
                            if (prev_val - current_val) / (abs(current_val) + 1e-6) > sensitivity or \
                               (next_val - current_val) / (abs(current_val) + 1e-6) > sensitivity:
                                peaks.append((i, current_val))
                        else:
                            peaks.append((i, current_val))
                else:
                    # 고점 조건: 이전값보다 크거나 같고, 다음값보다 커야 함 (산 형태)
                    if current_val >= prev_val and current_val > next_val:
                        if sensitivity > 0:
                            # 주변 값과의 차이가 민감도 이상이어야 함
                            if (current_val - prev_val) / (abs(prev_val) + 1e-6) > sensitivity or \
                               (current_val - next_val) / (abs(next_val) + 1e-6) > sensitivity:
                                peaks.append((i, current_val))
                        else:
                            peaks.append((i, current_val))
                            
            return peaks
            
        except Exception as e:
            print(f"⚠️ 고점/저점 찾기 오류: {e}")
            return []

    def calculate_divergence(self, df: pd.DataFrame, indicator: str, price_col: str = 'close') -> str:
        """
        🚀 개선된 다이버전스 계산 (민감도 향상 및 트리플 다이버전스)
        """
        if len(df) < 12:
            return 'none'
        
        try:
            # 🚀 캐시 키 생성 (240m 인터벌 최적화)
            cache_key = f"divergence_{indicator}_{hash(str(df.tail(15)[['timestamp', price_col, indicator]].values.tobytes()))}"
            cached_result = self.get_cached_data(cache_key, max_age=300)  # 5분 캐시
            if cached_result is not None:
                return cached_result
            
            # 🚀 최근 20개 데이터만 사용 (민감도 향상 및 범위 확대)
            recent_df = df.tail(20).copy()
            recent_df = recent_df.dropna(subset=[indicator, price_col])
            
            if len(recent_df) < 6:
                return 'none'
            
            # 🚀 고점/저점 찾기 (민감도 조정)
            # RSI는 더 민감하게, MACD는 적당히
            indicator_sensitivity = 0.001 if indicator == 'rsi' else 0.002
            price_sensitivity = 0.001  # 가격은 더 민감하게
            
            peaks = self._find_peaks_or_troughs(recent_df[indicator], sensitivity=indicator_sensitivity)
            price_peaks = self._find_peaks_or_troughs(recent_df[price_col], sensitivity=price_sensitivity)
            troughs = self._find_peaks_or_troughs(recent_df[indicator], is_trough=True, sensitivity=indicator_sensitivity)
            price_troughs = self._find_peaks_or_troughs(recent_df[price_col], is_trough=True, sensitivity=price_sensitivity)
            
            result = 'none'

            # --- Bearish Divergence (하락 다이버전스) ---
            if len(peaks) >= 2 and len(price_peaks) >= 2:
                # 최근 2개 기준
                _, ind2 = peaks[-2] # 이전 고점
                _, ind1 = peaks[-1] # 최근 고점
                _, price2 = price_peaks[-2] # 이전 가격 고점
                _, price1 = price_peaks[-1] # 최근 가격 고점
                
                # 트리플 다이버전스 (3연속) 확인
                is_triple = False
                if len(peaks) >= 3 and len(price_peaks) >= 3:
                     _, ind3 = peaks[-3]
                     _, price3 = price_peaks[-3]
                     # 가격 고점 갱신 3연속 & 지표 고점 하락 3연속
                     if price1 > price2 > price3 and ind1 < ind2 < ind3:
                         is_triple = True

                # 가격 상승 & 지표 하락
                if price1 > price2 and ind1 < ind2:
                    result = 'bearish_triple' if is_triple else 'bearish'
                elif price1 > price2 * 1.0003 and ind1 < ind2 * 0.9997: # 약한 조건
                    result = 'weak_bearish'

            # --- Bullish Divergence (상승 다이버전스) ---
            if len(troughs) >= 2 and len(price_troughs) >= 2:
                 # 최근 2개 기준
                _, ind2 = troughs[-2] # 이전 저점
                _, ind1 = troughs[-1] # 최근 저점
                _, price2 = price_troughs[-2] # 이전 가격 저점
                _, price1 = price_troughs[-1] # 최근 가격 저점

                # 트리플 다이버전스 (3연속) 확인
                is_triple = False
                if len(troughs) >= 3 and len(price_troughs) >= 3:
                     _, ind3 = troughs[-3]
                     _, price3 = price_troughs[-3]
                     # 가격 저점 갱신 3연속 & 지표 저점 상승 3연속
                     if price1 < price2 < price3 and ind1 > ind2 > ind3:
                         is_triple = True

                # 가격 하락 & 지표 상승
                if price1 < price2 and ind1 > ind2:
                    result = 'bullish_triple' if is_triple else 'bullish'
                elif price1 < price2 * 0.9997 and ind1 > ind2 * 1.0003: # 약한 조건
                    result = 'weak_bullish'
            
            # 결과 캐시
            self.set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            print(f"⚠️ 다이버전스 계산 오류 ({indicator}): {e}")
            return 'none'

    def _calculate_simple_divergence(self, df: pd.DataFrame, indicator: str, price_col: str = 'close') -> str:
        """간단한 다이버전스 계산 (최소 데이터로도 가능)"""
        try:
            if len(df) < 3:
                return 'none'
            
            # 최근 5개 데이터 사용 (더 많은 데이터로 정확도 향상)
            recent_data = df.tail(5)
            
            if indicator not in recent_data.columns or price_col not in recent_data.columns:
                return 'none'
            
            # 최근 5개 값 추출
            indicator_values = recent_data[indicator].dropna().values
            price_values = recent_data[price_col].dropna().values
            
            if len(indicator_values) < 3 or len(price_values) < 3:
                return 'none'
            
            # 🚀 개선된 다이버전스 계산
            # 1. 최근 3개 포인트의 방향성 분석
            price_trend = (price_values[-1] - price_values[-3]) / (price_values[-3] + 1e-6)
            indicator_trend = (indicator_values[-1] - indicator_values[-3]) / (abs(indicator_values[-3]) + 1e-6)
            
            # 2. 중간 포인트와의 비교 (더 정확한 다이버전스 감지)
            price_mid = price_values[-2]
            indicator_mid = indicator_values[-2]
            
            price_early_trend = (price_mid - price_values[-3]) / (price_values[-3] + 1e-6)
            price_late_trend = (price_values[-1] - price_mid) / (price_mid + 1e-6)
            
            indicator_early_trend = (indicator_mid - indicator_values[-3]) / (abs(indicator_values[-3]) + 1e-6)
            indicator_late_trend = (indicator_values[-1] - indicator_mid) / (abs(indicator_mid) + 1e-6)
            
            # 🚀 다이버전스 판단 (더 민감한 조건)
            # Bearish divergence: 가격은 상승하지만 지표는 하락
            if (price_trend > 0.001 and indicator_trend < -0.001) or \
               (price_early_trend > 0.001 and price_late_trend > 0.001 and 
                indicator_early_trend < -0.001 and indicator_late_trend < -0.001):
                return 'bearish'
            
            # Bullish divergence: 가격은 하락하지만 지표는 상승
            elif (price_trend < -0.001 and indicator_trend > 0.001) or \
                 (price_early_trend < -0.001 and price_late_trend < -0.001 and 
                  indicator_early_trend > 0.001 and indicator_late_trend > 0.001):
                return 'bullish'
            
            # 🚀 약한 다이버전스 감지 (더 민감한 조건)
            elif abs(price_trend) > 0.0005 and abs(indicator_trend) > 0.0005:
                if price_trend > 0 and indicator_trend < 0:
                    return 'weak_bearish'
                elif price_trend < 0 and indicator_trend > 0:
                    return 'weak_bullish'
            
            return 'none'
                
        except Exception as e:
            return 'none'

    def _calculate_rsi_divergence(self, df: pd.DataFrame, price_col: str) -> str:
        """RSI 다이버전스 계산"""
        try:
            if 'rsi' not in df.columns or price_col not in df.columns:
                return 'none'
            
            # 🚀 최근 8개 데이터에서 고점/저점 찾기 (민감도 향상)
            rsi_values = df['rsi'].tail(8).values
            price_values = df[price_col].tail(8).values
            
            if len(rsi_values) < 4:
                return 'none'
            
            # RSI 고점/저점 찾기
            rsi_peaks = []
            rsi_troughs = []
            
            for i in range(1, len(rsi_values) - 1):
                if rsi_values[i] > rsi_values[i-1] and rsi_values[i] > rsi_values[i+1]:
                    rsi_peaks.append((i, rsi_values[i]))
                elif rsi_values[i] < rsi_values[i-1] and rsi_values[i] < rsi_values[i+1]:
                    rsi_troughs.append((i, rsi_values[i]))
            
            # 가격 고점/저점 찾기
            price_peaks = []
            price_troughs = []
            
            for i in range(1, len(price_values) - 1):
                if price_values[i] > price_values[i-1] and price_values[i] > price_values[i+1]:
                    price_peaks.append((i, price_values[i]))
                elif price_values[i] < price_values[i-1] and price_values[i] < price_values[i+1]:
                    price_troughs.append((i, price_values[i]))
            
            # 🚀 다이버전스 판단 (민감도 향상)
            # Bearish divergence: 가격은 상승하지만 RSI는 하락
            if len(rsi_peaks) >= 1 and len(price_peaks) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(rsi_peaks) >= 2 and len(price_peaks) >= 2:
                    if (price_peaks[-1][1] > price_peaks[-2][1] and 
                        rsi_peaks[-1][1] < rsi_peaks[-2][1]):
                        return 'bearish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(rsi_peaks) >= 2 and len(price_peaks) >= 2:
                    # 가격이 0.5% 이상 상승하고 RSI가 0.5% 이상 하락
                    if (price_peaks[-1][1] > price_peaks[-2][1] * 1.005 and 
                        rsi_peaks[-1][1] < rsi_peaks[-2][1] * 0.995):
                        return 'bearish'
            
            # Bullish divergence: 가격은 하락하지만 RSI는 상승
            if len(rsi_troughs) >= 1 and len(price_troughs) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(rsi_troughs) >= 2 and len(price_troughs) >= 2:
                    if (price_troughs[-1][1] < price_troughs[-2][1] and 
                        rsi_troughs[-1][1] > rsi_troughs[-2][1]):
                        return 'bullish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(rsi_troughs) >= 2 and len(price_troughs) >= 2:
                    # 가격이 0.5% 이상 하락하고 RSI가 0.5% 이상 상승
                    if (price_troughs[-1][1] < price_troughs[-2][1] * 0.995 and 
                        rsi_troughs[-1][1] > rsi_troughs[-2][1] * 1.005):
                        return 'bullish'
            
            return 'none'
            
        except Exception as e:
            print(f"⚠️ RSI 다이버전스 계산 오류: {e}")
            return 'none'

    def _calculate_macd_divergence(self, df: pd.DataFrame, price_col: str) -> str:
        """MACD 다이버전스 계산 (기존 로직 유지 - 호환성)"""
        try:
            if 'macd' not in df.columns or price_col not in df.columns:
                return 'none'
            
            # 🚀 최근 8개 데이터에서 고점/저점 찾기 (민감도 향상)
            macd_values = df['macd'].tail(8).values
            price_values = df[price_col].tail(8).values
            
            if len(macd_values) < 4:
                return 'none'
            
            # MACD 고점/저점 찾기
            macd_peaks = []
            macd_troughs = []
            
            for i in range(1, len(macd_values) - 1):
                if macd_values[i] > macd_values[i-1] and macd_values[i] > macd_values[i+1]:
                    macd_peaks.append((i, macd_values[i]))
                elif macd_values[i] < macd_values[i-1] and macd_values[i] < macd_values[i+1]:
                    macd_troughs.append((i, macd_values[i]))
            
            # 가격 고점/저점 찾기
            price_peaks = []
            price_troughs = []
            
            for i in range(1, len(price_values) - 1):
                if price_values[i] > price_values[i-1] and price_values[i] > price_values[i+1]:
                    price_peaks.append((i, price_values[i]))
                elif price_values[i] < price_values[i-1] and price_values[i] < price_values[i+1]:
                    price_troughs.append((i, price_values[i]))
            
            # 🚀 다이버전스 판단 (민감도 향상)
            # Bearish divergence: 가격은 상승하지만 MACD는 하락
            if len(macd_peaks) >= 1 and len(price_peaks) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(macd_peaks) >= 2 and len(price_peaks) >= 2:
                    if (price_peaks[-1][1] > price_peaks[-2][1] and 
                        macd_peaks[-1][1] < macd_peaks[-2][1]):
                        return 'bearish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(macd_peaks) >= 2 and len(price_peaks) >= 2:
                    # 가격이 0.5% 이상 상승하고 MACD가 0.5% 이상 하락
                    if (price_peaks[-1][1] > price_peaks[-2][1] * 1.005 and 
                        macd_peaks[-1][1] < macd_peaks[-2][1] * 0.995):
                        return 'bearish'
            
            # Bullish divergence: 가격은 하락하지만 MACD는 상승
            if len(macd_troughs) >= 1 and len(price_troughs) >= 1:
                # 강한 다이버전스 (기존 로직)
                if len(macd_troughs) >= 2 and len(price_troughs) >= 2:
                    if (price_troughs[-1][1] < price_troughs[-2][1] and 
                        macd_troughs[-1][1] > macd_troughs[-2][1]):
                        return 'bullish'
                
                # 🚀 약한 다이버전스 감지 (민감도 향상)
                if len(macd_troughs) >= 2 and len(price_troughs) >= 2:
                    # 가격이 0.5% 이상 하락하고 MACD가 0.5% 이상 상승
                    if (price_troughs[-1][1] < price_troughs[-2][1] * 0.995 and 
                        macd_troughs[-1][1] > macd_troughs[-2][1] * 1.005):
                        return 'bullish'
            
            return 'none'
            
        except Exception as e:
            print(f"⚠️ MACD 다이버전스 계산 오류: {e}")
            return 'none'

    def _get_default_synergy_patterns(self):
        """기본 시너지 패턴 반환 (fallback)"""
        return {
            'bullish_momentum': {
                'type': 'momentum',
                'market_condition': 'bull',
                'data': {'rsi_range': [30, 70], 'macd_positive': True, 'volume_increase': True},
                'confidence': 0.8,
                'success_rate': 0.75,
                'synergy_score': 0.6
            },
            'bearish_reversal': {
                'type': 'reversal',
                'market_condition': 'bear',
                'data': {'rsi_range': [70, 90], 'macd_negative': True, 'volume_spike': True},
                'confidence': 0.7,
                'success_rate': 0.65,
                'synergy_score': 0.455
            }
        }
    

