"""
StrategyScoreCalculator - 전략 점수 계산기 (레짐별 인덱싱 + 병렬 로딩 + 진행률 로그 추가 버전)
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
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

# signal_selector 내부 모듈
try:
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import (
        CANDLES_DB_PATH, STRATEGIES_DB_PATH, TRADING_SYSTEM_DB_PATH,
        DB_PATH, CACHE_SIZE, AI_MODEL_AVAILABLE
    )
    from signal_selector.utils import (
        safe_float, safe_str, get_optimized_db_connection
    )
except ImportError:
    _current = os.path.dirname(os.path.abspath(__file__))
    _signal_selector = os.path.dirname(_current)
    _trade = os.path.dirname(_signal_selector)
    sys.path.insert(0, _trade)
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.config import *
    from signal_selector.utils import *

class StrategyScoreCalculator:
    """전략 점수 계산을 담당하는 별도 클래스 (레짐별 주머니 인덱싱 적용)"""
    
    def __init__(self):
        # 🚀 [Optimization] 데이터 구조
        self.global_strategies = []
        self.global_strategies_indexed = defaultdict(list)
        
        # 🆕 세밀한 구간화 기반 글로벌 예측값 조회기
        self.global_prediction_lookup = None
        self.global_predictions_loaded = False
        
        self.coin_tuned_strategies = {}
        self.coin_tuned_strategies_indexed = defaultdict(lambda: defaultdict(list))
        
        self._index_lock = threading.Lock() # 병렬 로딩을 위한 락
        
        self.reliability_scores = {}
        self.global_strategies_loaded = False
        self.coin_strategies_loaded = False
        self.reliability_scores_loaded = False
        
        # 🆕 캐시 경로 설정
        from signal_selector.config import DATA_STORAGE_PATH
        self.cache_dir = os.path.join(DATA_STORAGE_PATH, 'cache', 'strategies')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.min_confidence = 0.2
        self.min_signal_score = 0.02
        
        # AI 모델 상태
        self.ai_model = None
        self.ai_model_loaded = False
        self.model_type = "none"
        
        # 초기화 실행
        self._init_db_tables()
        self.load_global_strategies()
        self.load_global_predictions()  # 🆕 세밀한 구간화 기반 예측값 로드
        self.load_coin_tuned_strategies()
        self.load_reliability_scores()
        
        if AI_MODEL_AVAILABLE:
            self._load_ai_model()

    def _init_db_tables(self):
        """데이터베이스 테이블 초기화 (엔진 모드 보호)"""
        # 🚀 [Fix] 엔진 모드에서는 테이블 생성을 건너뛰어 잠금 방지
        if os.environ.get('ENGINE_READ_ONLY') == 'true':
            return

        try:
            db_path = TRADING_SYSTEM_DB_PATH
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            from trade.core.database import get_db_connection
            with get_db_connection(db_path, read_only=False) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL DEFAULT 'combined',
                        signal_pattern TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_profit REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        score REAL,
                        feedback_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(coin, interval, signal_pattern, feedback_type)
                    )
                ''')
        except Exception:
            pass

    def _get_item_cache_path(self, item_name: str) -> str:
        return os.path.join(self.cache_dir, f"{item_name.lower()}.cache")

    def _is_cache_valid(self, cache_path: str, source_db_path: str) -> bool:
        if not os.path.exists(cache_path) or not os.path.exists(source_db_path):
            return False
        return os.path.getmtime(cache_path) >= os.path.getmtime(source_db_path)

    def _load_item_from_cache(self, item_name: str, source_db_path: str) -> Optional[Any]:
        cache_path = self._get_item_cache_path(item_name)
        if os.path.exists(cache_path) and self._is_cache_valid(cache_path, source_db_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except: pass
        return None

    def _save_item_to_cache(self, item_name: str, data: Any):
        try:
            cache_path = self._get_item_cache_path(item_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except: pass

    def load_global_strategies(self):
        """글로벌 전략 로드 및 레짐 인덱싱"""
        try:
            # 🆕 signal_selector.config의 finalize_path를 사용하여 /workspace 경로 변환
            from signal_selector.config import finalize_path as _finalize_path, workspace_dir
            
            db_path = None
            
            # 1. GLOBAL_STRATEGY_DB_PATH 환경변수 시도
            env_path = os.environ.get('GLOBAL_STRATEGY_DB_PATH')
            if env_path:
                db_path = _finalize_path(env_path)
                if db_path and os.path.isdir(db_path):
                    db_path = os.path.join(db_path, 'common_strategies.db')
            
            # 2. STRATEGY_DB_PATH 환경변수 시도
            if not db_path or not os.path.exists(db_path):
                base_dir = os.environ.get('STRATEGY_DB_PATH') or os.environ.get('STRATEGIES_DB_PATH')
                if base_dir:
                    base_dir = _finalize_path(base_dir)
                    if base_dir:
                        if os.path.isdir(base_dir):
                            db_path = os.path.join(base_dir, 'common_strategies.db')
                        elif os.path.exists(base_dir):
                            db_path = base_dir
            
            # 3. 기본 경로 폴백
            if not db_path or not os.path.exists(db_path):
                default_paths = [
                    os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage', 'learning_strategies', 'common_strategies.db'),
                    os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage', 'common_strategies.db'),
                    os.path.join(workspace_dir, 'data_storage', 'learning_strategies', 'common_strategies.db'),
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        db_path = path
                        break
            
            if not db_path or not os.path.exists(db_path):
                self.global_strategies_loaded = True
                return

            cached = self._load_item_from_cache('global_common', db_path)
            if cached is not None:
                self.global_strategies = cached
                self._rebuild_global_index()
                self.global_strategies_loaded = True
                print(f"🚀 [Speed] 글로벌 전략 캐시 로드 ({len(self.global_strategies)}개)")
                return

            # 🚀 [Fix] 직접 연결 대신 엔진 공용 DB 유틸리티 사용 (잠금 방지 핵심)
            from trade.core.database import get_db_connection
            with get_db_connection(db_path, read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_strategies'")
                if not cursor.fetchone():
                    self.global_strategies_loaded = True
                    return
                
                cursor.execute('SELECT interval, strategy_type, params, profit, win_rate, trades_count, regime FROM global_strategies')
                
                self.global_strategies = []
                for interval, st_type, params_json, profit, win_rate, trades, regime in cursor.fetchall():
                    try:
                        params = json.loads(params_json) if params_json else {}
                        self.global_strategies.append({
                            'interval': interval,
                            'regime': str(regime or params.get('regime', 'neutral')).lower(),
                            'strategy': {'type': st_type, 'params': params},
                            'metrics': {
                                'performance_score': (win_rate * 0.6 + (profit/100 if profit else 0) * 0.4),
                                'profit': profit, 'win_rate': win_rate, 'trades': trades
                            }
                        })
                    except: continue
            
            self._rebuild_global_index()
            self._save_item_to_cache('global_common', self.global_strategies)
            self.global_strategies_loaded = True
            print(f"✅ 글로벌 전략 {len(self.global_strategies)}개 인덱싱 완료")
        except Exception as e:
            # DB 없거나 연결 실패 시 조용히 기본값 사용 (정상 동작)
            self.global_strategies_loaded = True

    def _rebuild_global_index(self):
        self.global_strategies_indexed = defaultdict(list)
        for s in self.global_strategies:
            self.global_strategies_indexed[s['regime']].append(s)

    def load_global_predictions(self):
        """🆕 세밀한 구간화 기반 글로벌 예측값 로드"""
        try:
            from signal_selector.config import finalize_path as _finalize_path, workspace_dir
            
            db_path = None
            
            # 1. GLOBAL_STRATEGY_DB_PATH 환경변수 시도
            env_path = os.environ.get('GLOBAL_STRATEGY_DB_PATH')
            if env_path:
                db_path = _finalize_path(env_path)
                if db_path and os.path.isdir(db_path):
                    db_path = os.path.join(db_path, 'common_strategies.db')
            
            # 2. STRATEGY_DB_PATH 환경변수 시도
            if not db_path or not os.path.exists(db_path):
                base_dir = os.environ.get('STRATEGY_DB_PATH') or os.environ.get('STRATEGIES_DB_PATH')
                if base_dir:
                    base_dir = _finalize_path(base_dir)
                    if base_dir:
                        if os.path.isdir(base_dir):
                            db_path = os.path.join(base_dir, 'common_strategies.db')
                        elif os.path.exists(base_dir):
                            db_path = base_dir
            
            # 3. 기본 경로 폴백
            if not db_path or not os.path.exists(db_path):
                default_paths = [
                    os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage', 'learning_strategies', 'common_strategies.db'),
                    os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage', 'common_strategies.db'),
                    os.path.join(workspace_dir, 'data_storage', 'learning_strategies', 'common_strategies.db'),
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        db_path = path
                        break
            
            if not db_path or not os.path.exists(db_path):
                self.global_predictions_loaded = True
                return
            
            # GlobalPredictionLookup 초기화
            try:
                from rl_pipeline.strategy.binned_global_synthesizer import GlobalPredictionLookup
                
                # 🔧 디버그: 실제 경로 확인
                print(f"🔍 [DEBUG] GlobalPredictionLookup 경로: {db_path}")
                print(f"🔍 [DEBUG] 파일 존재: {os.path.exists(db_path)}, 디렉토리 여부: {os.path.isdir(db_path)}")
                
                self.global_prediction_lookup = GlobalPredictionLookup(db_path)
                count = self.global_prediction_lookup.load_cache()
                if count > 0:
                    print(f"✅ 구간화 기반 글로벌 예측값 {count}개 로드 완료")
                else:
                    print(f"⚠️ 글로벌 예측값 0개 (테이블 비어있거나 로드 실패)")
                self.global_predictions_loaded = True
            except ImportError:
                # rl_pipeline 모듈이 없는 환경 (트레이딩 전용)
                self.global_predictions_loaded = True
            
        except Exception as e:
            logger.debug(f"⚠️ 구간화 기반 글로벌 예측값 로드 실패: {e}")
            self.global_predictions_loaded = True

    def get_binned_global_prediction(self, interval: str, candle: pd.Series, strategy_params: Dict = None) -> Optional[Dict]:
        """
        🆕 세밀한 구간화 기반 글로벌 예측값 조회
        
        Args:
            interval: 인터벌
            candle: 캔들 데이터 (레짐, RSI 등 포함)
            strategy_params: 전략 파라미터 (선택적)
            
        Returns:
            예측값 딕셔너리 또는 None
        """
        if not self.global_prediction_lookup:
            return None
        
        try:
            regime = str(candle.get('regime') or candle.get('regime_label') or 'neutral').lower()
            quality_grade = 'B'  # 기본값
            
            # 캔들에서 시그널 파라미터 추출
            rsi = candle.get('rsi')
            mfi = candle.get('mfi')
            adx = candle.get('adx')
            volume_ratio = candle.get('volume_ratio') or candle.get('vol_ratio')
            atr = candle.get('atr') or candle.get('atr_pct')
            
            # 전략 파라미터에서 추가 정보
            if strategy_params:
                rsi_min = strategy_params.get('rsi_min', rsi)
                rsi_max = strategy_params.get('rsi_max', rsi)
                stop_loss_pct = strategy_params.get('stop_loss_pct')
                take_profit_pct = strategy_params.get('take_profit_pct')
                macd_buy = strategy_params.get('macd_buy_threshold')
                macd_sell = strategy_params.get('macd_sell_threshold')
            else:
                rsi_min = rsi
                rsi_max = rsi
                stop_loss_pct = None
                take_profit_pct = None
                macd_buy = None
                macd_sell = None
            
            prediction = self.global_prediction_lookup.lookup(
                interval=interval,
                regime=regime,
                quality_grade=quality_grade,
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                mfi_min=mfi,
                mfi_max=mfi,
                adx_min=adx,
                volume_ratio_min=volume_ratio,
                volume_ratio_max=volume_ratio,
                macd_buy_threshold=macd_buy,
                macd_sell_threshold=macd_sell,
                atr_min=atr,
                atr_max=atr,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                fallback_regime=True
            )
            
            return prediction
            
        except Exception as e:
            logger.debug(f"⚠️ 구간화 예측값 조회 실패: {e}")
            return None

    def load_coin_tuned_strategies(self):
        """코인별 전략 로드 및 레짐 인덱싱 (병렬 로딩 적용)"""
        try:
            db_path = STRATEGIES_DB_PATH
            if not os.path.isdir(db_path):
                self.coin_strategies_loaded = True
                return

            import glob
            db_files = glob.glob(os.path.join(db_path, '*_strategies.db'))
            target_coins_env = os.getenv('TARGET_COINS', 'ALL')
            target_coins = [] if target_coins_env.upper() == 'ALL' else [c.strip().lower() for c in target_coins_env.split(',') if c.strip()]
            
            # 필터링
            actual_files = []
            for f in db_files:
                if 'common_strategies.db' in f: continue
                coin_name = os.path.basename(f).replace('_strategies.db', '').lower()
                if target_coins and coin_name not in target_coins: continue
                actual_files.append((f, coin_name))

            total_files = len(actual_files)
            if total_files == 0:
                self.coin_strategies_loaded = True
                return

            print(f"📊 코인별 전략 로드 시작: 총 {total_files}개 DB 분석 중...")
            
            # 병렬 처리를 위한 함수
            def process_single_coin(file_info):
                f_path, c_name = file_info
                # 1. 캐시 확인
                cached = self._load_item_from_cache(f"coin_{c_name}", f_path)
                if cached is not None:
                    return c_name, cached, True

                # 2. DB 로드
                try:
                    with sqlite3.connect(f_path, timeout=5.0) as conn:
                        conn.execute("PRAGMA query_only = ON")
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")
                        if not cursor.fetchone(): return c_name, [], False
                        
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_label_stats'")
                        has_mfe = cursor.fetchone() is not None
                        
                        query = f'''
                            SELECT s.symbol, s.interval, s.profit, s.win_rate, s.quality_grade, s.strategy_type, s.strategy_conditions, 
                                   s.created_at, s.id, {'ls.rmax_p90, ls.rmin_p10, ls.n_signals' if has_mfe else 'NULL, NULL, NULL'}, s.league
                            FROM strategies s
                            {'LEFT JOIN strategy_label_stats ls ON s.id = ls.strategy_id AND s.symbol = ls.coin AND s.interval = ls.interval' if has_mfe else ''}
                            WHERE s.quality_grade != 'F' AND (s.league = 'major' OR s.quality_grade IN ('S', 'A', 'B'))
                            ORDER BY CASE s.league WHEN 'major' THEN 0 ELSE 1 END ASC,
                                     CASE s.quality_grade WHEN 'S' THEN 0 WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4 ELSE 5 END ASC,
                                     s.profit DESC
                            LIMIT 2000
                        '''
                        cursor.execute(query)
                        
                        coin_strategies = []
                        for row in cursor.fetchall():
                            try:
                                symbol, interval, profit, win_rate, grade, st_type, params_json, created_at, sid, rmax, rmin, n_sig, league = row
                                params = json.loads(params_json) if params_json else {}
                                coin_strategies.append({
                                    'strategy_id': sid, 'strategy_type': st_type, 'interval': interval,
                                    'tuned_parameters': params,
                                    'performance_metrics': {
                                        'avg_reward': profit, 'success_rate': win_rate, 'quality_grade': grade, 'league': league,
                                        'entry_score': (rmax - 1.5 * abs(rmin)) if rmax is not None else None,
                                        'risk_score': abs(rmin or 0), 'n_signals': n_sig
                                    }
                                })
                            except: continue
                        
                        if coin_strategies:
                            self._save_item_to_cache(f"coin_{c_name}", coin_strategies)
                        return c_name, coin_strategies, False
                except:
                    return c_name, [], False

            # ThreadPool로 병렬 실행
            loaded_count = 0
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_single_coin, f): f for f in actual_files}
                for future in as_completed(futures):
                    c_name, strategies, is_cached = future.result()
                    if strategies:
                        with self._index_lock:
                            self.coin_tuned_strategies[c_name] = strategies
                            for s in strategies:
                                params = s.get('tuned_parameters', {})
                                regime = str(params.get('market_regime') or params.get('regime') or 'neutral').lower()
                                self.coin_tuned_strategies_indexed[c_name][regime].append(s)
                    
                    loaded_count += 1
                    if loaded_count % 50 == 0 or loaded_count == total_files:
                        print(f"   ⏳ 전략 로딩 중... ({loaded_count}/{total_files})")

            self.coin_strategies_loaded = True
            total_strat = sum(len(s) for s in self.coin_tuned_strategies.values())
            print(f"✅ 코인별 전략 {total_strat}개 인덱싱 완료 (대상: {len(self.coin_tuned_strategies)}개 코인)")
        except Exception as e:
            logger.error(f"⚠️ 코인별 전략 로드 실패: {e}")

    def calculate_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """🧬 전략 점수 계산 (고속 인덱싱 버전)"""
        try:
            # 1. 글로벌 점수
            global_score = self._get_global_strategy_score(coin, interval, candle)
            
            # 2. 개별 코인 점수
            symbol_score = self._get_symbol_strategy_score(coin, interval, candle)
            
            # 3. 가중치 혼합 (기본 7:3)
            coin_weight, global_weight = 0.7, 0.3
            
            if symbol_score > 0 and coin.lower() in self.coin_tuned_strategies:
                base_score = (symbol_score * coin_weight) + (global_score * global_weight)
            else:
                base_score = global_score
            
            return max(0.0, min(1.0, base_score))
        except:
            return 0.5

    def _get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """글로벌 전략 점수 (🆕 구간화 예측값 우선 + 레짐 주머니 폴백)"""
        try:
            # 🆕 1. 세밀한 구간화 기반 예측값 우선 조회
            if self.global_prediction_lookup:
                binned_pred = self.get_binned_global_prediction(interval, candle)
                if binned_pred and binned_pred.get('confidence_score', 0) > 0.3:
                    # 예측값 기반 점수 계산
                    median_profit = binned_pred.get('median_profit', 0)
                    median_win_rate = binned_pred.get('median_win_rate', 0.5)
                    confidence = binned_pred.get('confidence_score', 0.5)
                    
                    # 점수 = win_rate * 0.5 + profit 기반 * 0.3 + 신뢰도 * 0.2
                    profit_score = max(0.0, min(1.0, (median_profit + 0.1) / 0.3)) if median_profit else 0.5
                    base_score = median_win_rate * 0.5 + profit_score * 0.3 + confidence * 0.2
                    
                    return max(0.0, min(1.0, base_score))
            
            # 2. 기존 방식: 레짐별 대표 전략 기반
            if not self.global_strategies_indexed: return 0.5
            current_regime = str(candle.get('regime') or candle.get('regime_label') or 'neutral').lower()
            
            target_regimes = self._get_adjacent_regimes(current_regime)
            candidates = []
            for r in target_regimes:
                candidates.extend(self.global_strategies_indexed.get(r, []))
            
            if not candidates: candidates = self.global_strategies_indexed.get('neutral', self.global_strategies[:10])

            best_match = None
            max_priority = -1
            for s in candidates:
                priority = 0
                if s['regime'] == current_regime: priority += 15
                elif s['regime'] == 'neutral': priority += 2
                if s['interval'] == interval: priority += 5
                
                if priority > max_priority:
                    max_priority = priority
                    best_match = s
                elif priority == max_priority and best_match:
                    if s['metrics']['performance_score'] > best_match['metrics']['performance_score']:
                        best_match = s

            if not best_match: return 0.5
            score = best_match['metrics'].get('performance_score', 0.5)
            
            params = best_match['strategy'].get('params', {})
            rsi_min, rsi_max = params.get('rsi_min'), params.get('rsi_max')
            if rsi_min is not None and rsi_max is not None:
                rsi = candle.get('rsi', 50.0)
                if not (rsi_min - 5 <= rsi <= rsi_max + 5): score *= 0.7
            
            return max(0.0, min(1.0, score))
        except: return 0.5

    def _get_symbol_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """코인별 전략 점수 (레짐 주머니 기반)"""
        try:
            coin_pockets = self.coin_tuned_strategies_indexed.get(coin.lower())
            if not coin_pockets: return 0.0
            
            current_regime = str(candle.get('regime') or candle.get('regime_label') or 'neutral').lower()
            target_regimes = self._get_adjacent_regimes(current_regime)
            candidates = []
            for r in target_regimes:
                candidates.extend(coin_pockets.get(r, []))
            
            if not candidates: return 0.0
            
            best_score = 0.0
            current_rsi = candle.get('rsi', 50.0)
            
            for strategy in candidates:
                if strategy.get('interval') != interval: continue
                
                metrics = strategy.get('performance_metrics', {})
                entry_score = metrics.get('entry_score')
                
                if entry_score is not None:
                    score = 0.3 + (max(0.0, min(1.0, (entry_score + 0.02) / 0.04)) * 0.5)
                    risk_score = metrics.get('risk_score', 0)
                    if risk_score > 0.03: score -= min(0.15, (risk_score - 0.03) * 3)
                else:
                    score = metrics.get('success_rate', 0.5)
                    avg_reward = metrics.get('avg_reward', 0.0)
                    if avg_reward > 0: score += min(0.2, avg_reward * 0.01)
                
                params = strategy.get('tuned_parameters', {})
                if params.get('rsi_min', 0) <= current_rsi <= params.get('rsi_max', 100): score += 0.1
                if str(params.get('market_regime') or params.get('regime')).lower() == current_regime: score += 0.15
                
                if score > best_score: best_score = score
            
            return max(0.0, min(1.0, best_score))
        except: return 0.0

    def _get_adjacent_regimes(self, regime: str) -> List[str]:
        regimes = ['extreme_bearish', 'bearish', 'sideways_bearish', 'neutral', 'sideways_bullish', 'bullish', 'extreme_bullish']
        try:
            idx = regimes.index(regime)
            return regimes[max(0, idx - 1):min(len(regimes), idx + 2)]
        except:
            return [regime, 'neutral']

    def load_reliability_scores(self):
        """신뢰도 점수 로드 (잠금 완벽 방지)"""
        try:
            db_path = TRADING_SYSTEM_DB_PATH
            if not os.path.exists(db_path):
                self.reliability_scores_loaded = True
                return
            
            from trade.core.database import get_db_connection
            # 🚀 [Fix] with 구문 + 읽기 전용 연결로 잠금 방지
            with get_db_connection(db_path, read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_feedback_scores'")
                if not cursor.fetchone():
                    self.reliability_scores_loaded = True
                    return
                cursor.execute('SELECT coin, interval, score, confidence FROM signal_feedback_scores')
                for coin, interval, score, confidence in cursor.fetchall():
                    key = f"{coin}_{interval}"
                    val = score if score is not None else (confidence or 0.0)
                    if key in self.reliability_scores:
                        self.reliability_scores[key] = (self.reliability_scores[key] + val) / 2.0
                    else:
                        self.reliability_scores[key] = val
            self.reliability_scores_loaded = True
        except: 
            self.reliability_scores_loaded = True

    def _get_reliability_bonus(self, coin: str, interval: str) -> float:
        key = f"{coin}_{interval}"
        if key in self.reliability_scores:
            return self.reliability_scores[key]
        return 1.0

    def _load_ai_model(self):
        self.ai_model_loaded = False

    def get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        return self._get_global_strategy_score(coin, interval, candle)
