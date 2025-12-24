"""
StrategyScoreCalculator - 전략 점수 계산기
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
    """전략 점수 계산을 담당하는 별도 클래스 (learning_engine.py 연동 강화)"""
    
    def __init__(self):
        self.global_strategies = {}  # 딕셔너리로 변경
        self.coin_tuned_strategies = {}
        self.reliability_scores = {}
        self.global_strategies_loaded = False
        self.coin_strategies_loaded = False
        self.reliability_scores_loaded = False
        
        # 🆕 학습 기반 임계값 관리
        self.use_learning_based_thresholds = True
        self.learning_feedback = None
        self.min_confidence = 0.5
        self.min_signal_score = 0.03
        
        # 🆕 AI 모델 초기화
        self.ai_model = None
        self.ai_model_loaded = False
        self.model_type = "none"
        self.current_coin = None
        self.feature_dim = 0
        
        # 데이터베이스 초기화
        self.create_signal_table()
        
        # 전략 데이터 로드
        self.load_global_strategies()
        self.load_coin_tuned_strategies()
        self.load_reliability_scores()
        
        # 🆕 AI 모델 로드
        if AI_MODEL_AVAILABLE:
            self._load_ai_model()
    
    def create_signal_table(self):
        """시그널 피드백 테이블 생성 (trading_system.db에 저장)"""
        try:
            # 🆕 절대 경로 사용 (TRADING_SYSTEM_DB_PATH 또는 fallback)
            try:
                from signal_selector.config import TRADING_SYSTEM_DB_PATH
                db_path = TRADING_SYSTEM_DB_PATH
            except ImportError:
                # fallback: 상대 경로를 절대 경로로 변환
                current_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_dir = os.path.dirname(current_dir)
                db_path = os.path.join(workspace_dir, 'data_storage', 'trading_system.db')
            
            # 🆕 디렉토리 존재 여부 확인 및 생성
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_feedback_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    feedback_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, interval, signal_type, feedback_type)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ 시그널 테이블 생성 실패: {e}")
    
    def load_global_strategies(self):
        """글로벌 학습 전략 로드 (learning_strategies.db의 global_strategies 테이블)"""
        try:
            # DB 경로 설정 (환경변수 우선 + 디렉토리 모드 지원)
            try:
                from signal_selector.config import STRATEGIES_DB_PATH
                db_path = STRATEGIES_DB_PATH
            except ImportError:
                # 폴백: DATA_STORAGE_PATH 사용
                data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                db_path = os.getenv('STRATEGY_DB_PATH', os.path.join(data_storage, 'learning_strategies'))
                
            # 🔧 디렉토리 모드 지원: 폴더인 경우 common_strategies.db 사용
            if os.path.isdir(db_path):
                db_path = os.path.join(db_path, 'common_strategies.db')
                
            if not os.path.exists(db_path):
                print(f"ℹ️ 글로벌 전략 DB 파일이 없습니다: {db_path} (정상 - 아직 학습 데이터 없음)")
                self.global_strategies_loaded = True
                return
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_strategies'")
            if not cursor.fetchone():
                print(f"⚠️ global_strategies 테이블이 존재하지 않음 - 기본 전략만 사용")
                self.global_strategies_loaded = True
                conn.close()
                return
            
            # 상위 글로벌 전략 로드 (변동성 그룹 전략 포함)
            cursor.execute('''
                SELECT strategy_type, params, profit, win_rate, trades_count, created_at
                FROM global_strategies
                WHERE profit IS NOT NULL
                ORDER BY 
                    CASE WHEN strategy_type LIKE 'volatility_group_%' THEN 0 ELSE 1 END, -- 변동성 그룹 전략 우선
                    profit DESC
                -- LIMIT 제거: 모든 학습된 전략 활용
            ''')
            
            strategies = cursor.fetchall()
            self.global_strategies = []
            
            for strategy_type, params_json, profit, win_rate, trades, created_at in strategies:
                try:
                    params = json.loads(params_json) if params_json else {}
                    
                    self.global_strategies.append({
                        'strategy': {
                            'type': strategy_type,
                            'params': params
                        },
                        'metrics': {
                            'performance_score': (win_rate * 0.4 + (profit/100 if profit else 0) * 0.6),
                            'profit': profit,
                            'win_rate': win_rate,
                            'trades': trades
                        },
                        'created_at': created_at
                    })
                except Exception:
                    continue
            
            conn.close()
            self.global_strategies_loaded = True
            print(f"✅ 글로벌 학습 전략 로드(Native DB): {len(self.global_strategies)}개")
            
        except Exception as e:
            print(f"⚠️ 글로벌 학습 전략 로드 실패: {e}")
            self.global_strategies_loaded = False
    
    def load_coin_tuned_strategies(self):
        """코인별 학습된 전략 로드 (MFE/MAE EntryScore 기반 완전 전환)"""
        try:
            # DB 경로 설정 (환경변수 우선 + 디렉토리 모드 지원)
            try:
                from signal_selector.config import STRATEGIES_DB_PATH
                db_path = STRATEGIES_DB_PATH
            except ImportError:
                # 폴백: DATA_STORAGE_PATH 사용
                data_storage = os.getenv('DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'data_storage'))
                db_path = os.getenv('STRATEGY_DB_PATH', os.path.join(data_storage, 'learning_strategies'))
                
            # 🔧 디렉토리 모드 지원: 폴더인 경우 내부의 모든 *_strategies.db 파일 로드
            if os.path.isdir(db_path):
                import glob
                db_files = glob.glob(os.path.join(db_path, '*_strategies.db'))
                
                if not db_files:
                    print(f"ℹ️ 코인 전략 DB 파일이 없습니다 (디렉토리 비어있음): {db_path}")
                    self.coin_strategies_loaded = True
                    return

                self.coin_tuned_strategies = {}
                
                for coin_db in db_files:
                    # common_strategies.db는 별도 로드하거나 제외
                    if 'common_strategies.db' in coin_db:
                        continue
                        
                    try:
                        conn = sqlite3.connect(coin_db)
                        cursor = conn.cursor()
                        
                        # 테이블 존재 확인 (strategies)
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")
                        if not cursor.fetchone():
                            conn.close()
                            continue
                        
                        # 🔥 MFE/MAE 통계 테이블 존재 확인 (방어 로직)
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_label_stats'")
                        has_mfe_stats = cursor.fetchone() is not None
                        
                        if has_mfe_stats:
                            # 🔥 MFE/MAE 완전 전환: EntryScore 기반 정렬 및 필터링
                            # EntryScore = rmax_p90 - 1.5 * abs(rmin_p10)
                            cursor.execute('''
                                SELECT 
                                    s.symbol, s.interval, s.profit, s.win_rate, s.quality_grade, 
                                    s.strategy_type, s.strategy_conditions, s.created_at, s.id,
                                    ls.rmax_p90, ls.rmin_p10, ls.n_signals,
                                    (COALESCE(ls.rmax_p90, 0) - 1.5 * ABS(COALESCE(ls.rmin_p10, 0))) as entry_score,
                                    ABS(COALESCE(ls.rmin_p10, 0)) as risk_score
                                FROM strategies s
                                LEFT JOIN strategy_label_stats ls 
                                    ON s.id = ls.strategy_id 
                                    AND s.symbol = ls.coin 
                                    AND s.interval = ls.interval
                                WHERE 
                                    -- MFE/MAE 전략: EntryScore >= 0 또는 통계 없음(NULL fallback)
                                    (ls.rmax_p90 IS NULL OR (ls.rmax_p90 - 1.5 * ABS(COALESCE(ls.rmin_p10, 0))) >= -0.01)
                                    AND COALESCE(s.profit, 0) >= 0
                                ORDER BY 
                                    -- 1순위: EntryScore가 있는 전략 우선
                                    CASE WHEN ls.rmax_p90 IS NOT NULL THEN 0 ELSE 1 END ASC,
                                    -- 2순위: EntryScore 내림차순
                                    (COALESCE(ls.rmax_p90, 0) - 1.5 * ABS(COALESCE(ls.rmin_p10, 0))) DESC,
                                    -- 3순위: 기존 profit (fallback)
                                    s.profit DESC
                                LIMIT 2000
                            ''')
                        else:
                            # 🔧 방어 로직: MFE/MAE 테이블 없으면 기존 방식으로 fallback
                            cursor.execute('''
                                SELECT 
                                    symbol, interval, profit, win_rate, quality_grade, 
                                    strategy_type, strategy_conditions, created_at, id,
                                    NULL as rmax_p90, NULL as rmin_p10, NULL as n_signals,
                                    NULL as entry_score, NULL as risk_score
                                FROM strategies
                                WHERE COALESCE(profit, 0) >= 0
                                ORDER BY 
                                    CASE COALESCE(quality_grade, 'F')
                                        WHEN 'S' THEN 0 WHEN 'A' THEN 1 WHEN 'B' THEN 2
                                        WHEN 'C' THEN 3 WHEN 'D' THEN 4 ELSE 5
                                    END ASC,
                                    profit DESC
                                LIMIT 2000
                            ''')
                        
                        strategies = cursor.fetchall()
                        
                        for row in strategies:
                            try:
                                coin, interval, profit, win_rate, grade, strategy_type, params_json, created_at, strategy_id, rmax_p90, rmin_p10, n_signals, entry_score, risk_score = row
                                params = json.loads(params_json) if params_json else {}
                                
                                if coin not in self.coin_tuned_strategies:
                                    self.coin_tuned_strategies[coin] = []
                                
                                self.coin_tuned_strategies[coin].append({
                                    'strategy_id': strategy_id,
                                    'strategy_type': strategy_type,
                                    'interval': interval,
                                    'tuned_parameters': params,
                                    'performance_metrics': {
                                        'avg_reward': profit,
                                        'success_rate': win_rate,
                                        'quality_grade': grade,
                                        # 🔥 MFE/MAE 지표 추가
                                        'entry_score': entry_score,
                                        'risk_score': risk_score,
                                        'rmax_p90': rmax_p90,
                                        'rmin_p10': rmin_p10,
                                        'n_signals': n_signals
                                    },
                                    'created_at': created_at
                                })
                            except Exception:
                                continue
                                
                        conn.close()
                    except Exception as e:
                        # print(f"⚠️ {os.path.basename(coin_db)} 로드 실패: {e}")
                        continue
                
                self.coin_strategies_loaded = True
                total_strategy_count = sum(len(s) for s in self.coin_tuned_strategies.values())
                # MFE/MAE 전략 개수 집계
                mfe_count = sum(1 for strategies in self.coin_tuned_strategies.values() 
                               for s in strategies if s['performance_metrics'].get('entry_score') is not None)
                print(f"✅ 코인별 전략 로드(MFE/MAE 전환): {len(self.coin_tuned_strategies)}개 코인, 총 {total_strategy_count}개 전략 (MFE/MAE: {mfe_count}개)")
                return
                
            # 기존 단일 파일 모드 (하위 호환성)
            if not os.path.exists(db_path):
                print(f"ℹ️ 코인 전략 DB 파일이 없습니다: {db_path} (정상 - 아직 학습 데이터 없음)")
                self.coin_strategies_loaded = True
                return
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 🚀 테이블 존재 여부 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_strategies'")
            if not cursor.fetchone():
                # print(f"⚠️ coin_strategies 테이블이 존재하지 않음 - 기본 전략만 사용")
                self.coin_strategies_loaded = True  # 로드 완료로 표시 (빈 상태)
                conn.close()
                return
            
            # 코인별 상위 전략 로드 (수익률 상위 5개씩)
            try:
                # 윈도우 함수 사용 (다중 전략 로드를 위해 제한 완화)
                cursor.execute('''
                    SELECT coin, interval, strategy_type, params, profit, win_rate, quality_grade, created_at
                        FROM coin_strategies
                        WHERE profit IS NOT NULL
                    ORDER BY profit DESC
                    LIMIT 2000
                ''')
            except sqlite3.OperationalError:
                # 윈도우 함수 미지원 시 단순 조회
                cursor.execute('''
                    SELECT coin, interval, strategy_type, params, profit, win_rate, quality_grade, created_at
                    FROM coin_strategies
                    WHERE profit IS NOT NULL
                    ORDER BY profit DESC
                    LIMIT 1000
                ''')
            
            strategies = cursor.fetchall()
            
            # 초기화
            self.coin_tuned_strategies = {}
            
            for coin, interval, strategy_type, params_json, profit, win_rate, grade, created_at in strategies:
                try:
                    params = json.loads(params_json) if params_json else {}
                    
                    if coin not in self.coin_tuned_strategies:
                        self.coin_tuned_strategies[coin] = []
                    
                    self.coin_tuned_strategies[coin].append({
                        'strategy_type': strategy_type,
                        'interval': interval,
                        'tuned_parameters': params,
                        'performance_metrics': {
                            'avg_reward': profit, # profit을 reward로 매핑
                            'success_rate': win_rate,
                            'quality_grade': grade
                        },
                        'created_at': created_at
                    })
                except Exception as e:
                    continue
            
            conn.close()
            self.coin_strategies_loaded = True
            count = sum(len(s) for s in self.coin_tuned_strategies.values())
            print(f"✅ 코인별 학습 전략 로드(Native DB): {len(self.coin_tuned_strategies)}개 코인, 총 {count}개 전략")
            
        except Exception as e:
            print(f"⚠️ 코인별 학습 전략 로드 실패: {e}")
            self.coin_strategies_loaded = False
    
    def load_reliability_scores(self):
        """신뢰도 점수 로드 (trading_system.db의 signal_feedback_scores 테이블)"""
        try:
            # 🆕 절대 경로 사용 (TRADING_SYSTEM_DB_PATH 또는 fallback)
            try:
                from signal_selector.config import TRADING_SYSTEM_DB_PATH
                db_path = TRADING_SYSTEM_DB_PATH
            except ImportError:
                # fallback: 상대 경로를 절대 경로로 변환
                current_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_dir = os.path.dirname(current_dir)
                db_path = os.path.join(workspace_dir, 'data_storage', 'trading_system.db')
            
            # 🆕 테이블 존재 여부 먼저 확인
            if not os.path.exists(db_path):
                print(f"ℹ️ 신뢰도 점수 DB 파일이 없습니다: {db_path} (정상 - 아직 데이터 없음)")
                self.reliability_scores_loaded = True  # 로드 완료로 표시 (빈 상태)
                return
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 🆕 테이블 존재 여부 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='signal_feedback_scores'
            """)
            if not cursor.fetchone():
                conn.close()
                print(f"ℹ️ signal_feedback_scores 테이블이 없습니다 (정상 - 아직 데이터 없음)")
                self.reliability_scores_loaded = True  # 로드 완료로 표시 (빈 상태)
                return
            
            # 🆕 테이블 스키마 확인 (필요한 컬럼이 있는지 체크)
            cursor.execute("PRAGMA table_info(signal_feedback_scores)")
            columns = [row[1] for row in cursor.fetchall()]
            required_columns = ['coin', 'interval', 'signal_type', 'score', 'feedback_type']
            
            if not all(col in columns for col in required_columns):
                # 필요한 컬럼이 없으면 조용히 무시 (다른 스키마로 생성된 테이블일 수 있음)
                conn.close()
                # print(f"ℹ️ signal_feedback_scores 테이블 스키마가 다릅니다 (정상 - 다른 형식의 테이블)")
                self.reliability_scores_loaded = True  # 로드 완료로 표시 (빈 상태)
                return
            
            cursor.execute('''
                SELECT coin, interval, signal_type, score, feedback_type
                FROM signal_feedback_scores
                ORDER BY created_at DESC
            ''')
            
            scores = cursor.fetchall()
            for coin, interval, signal_type, score, feedback_type in scores:
                key = f"{coin}_{interval}_{signal_type}_{feedback_type}"
                self.reliability_scores[key] = score
            
            conn.close()
            self.reliability_scores_loaded = True
            if len(self.reliability_scores) > 0:
                print(f"✅ 신뢰도 점수 로드 완료: {len(self.reliability_scores)}개")
            
        except Exception as e:
            # 🆕 "unable to open database file" 오류는 조용히 무시 (경로 문제일 수 있음)
            if "unable to open database file" in str(e).lower():
                print(f"ℹ️ 신뢰도 점수 DB 접근 불가 (정상 - 아직 데이터 없음): {e}")
                self.reliability_scores_loaded = True  # 로드 완료로 표시 (빈 상태)
            elif "no such column" in str(e).lower():
                # 스키마 불일치 오류는 조용히 무시
                # print(f"ℹ️ signal_feedback_scores 테이블 스키마가 다릅니다 (정상 - 다른 형식의 테이블)")
                self.reliability_scores_loaded = True  # 로드 완료로 표시 (빈 상태)
            else:
                print(f"⚠️ 신뢰도 점수 로드 실패: {e}")
                self.reliability_scores_loaded = False
    
    def _load_ai_model(self):
        """학습된 전략 기반 AI 모델 로드"""
        try:
            if not AI_MODEL_AVAILABLE:
                print("⚠️ AI 모델을 사용할 수 없습니다")
                return
            
            # 🆕 데이터베이스에서 학습된 전략 로드
            _load_learned_strategies_from_db()
            
            # 🆕 전략 기반 AI 모델 생성
            self.ai_model, self.model_type = _create_strategy_based_ai_model()
            self.feature_dim = 50
            self.ai_model_loaded = True
            print(f"✅ 학습된 전략 기반 AI 모델 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 학습된 전략 기반 AI 모델 로드 실패: {e}")
            self.ai_model_loaded = False
    
    def _load_learning_engines(self):
        """learning_engine.py의 학습 엔진들 로드"""
        try:
            if not AI_MODEL_AVAILABLE:
                return
            
            # 글로벌 학습 매니저 로드
            self.global_learning_manager = GlobalLearningManager()
            print("✅ 글로벌 학습 매니저 로드 완료")
            
            # 심볼별 튜닝 매니저 로드
            self.symbol_finetuning_manager = SymbolFinetuningManager()
            print("✅ 심볼별 튜닝 매니저 로드 완료")
            
            # 시너지 학습기 로드
            self.synergy_learner = ShortTermLongTermSynergyLearner()
            print("✅ 시너지 학습기 로드 완료")
            
            # 🆕 신뢰도 점수 계산기 로드
            self.reliability_calculator = ReliabilityScoreCalculator()
            print("✅ 신뢰도 점수 계산기 로드 완료")
            
            # 🆕 지속적 학습 관리자 로드
            self.continuous_learning_manager = ContinuousLearningManager()
            print("✅ 지속적 학습 관리자 로드 완료")
            
            # 🆕 라우팅 패턴 분석기 로드
            self.routing_pattern_analyzer = RoutingPatternAnalyzer()
            print("✅ 라우팅 패턴 분석기 로드 완료")
            
            # 🆕 상황별 학습 관리자 로드
            self.contextual_learning_manager = ContextualLearningManager()
            print("✅ 상황별 학습 관리자 로드 완료")
            
        except Exception as e:
            print(f"⚠️ 학습 엔진 로드 실패: {e}")
            self.global_learning_manager = None
            self.symbol_finetuning_manager = None
            self.synergy_learner = None
            self.reliability_calculator = None
            self.continuous_learning_manager = None
            self.routing_pattern_analyzer = None
            self.contextual_learning_manager = None
    
    def calculate_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """🧬 전략 점수 계산 (상호보완 방식: 개별 코인 + 글로벌 전략 비율 혼합)
        
        계산 방식:
        - 개별 코인 전략이 있으면: (개별 점수 × coin_weight) + (글로벌 점수 × global_weight)
        - 개별 코인 전략이 없으면: 글로벌 전략 100% fallback
        - 기본 비율: 개별 70%, 글로벌 30% (DB 동적 가중치 우선)
        
        Phase 기반 자동 진화 시스템도 지원:
        - Phase 1 (STATISTICAL): MFE/MAE 통계 기반
        - Phase 2 (PREDICTIVE): 예측 모델 기반
        - Phase 3 (TIMING_OPTIMIZED): RL Agent 기반
        """
        try:
            # 🧬 Auto-Evolution System 체크
            try:
                from rl_pipeline.evolution import get_auto_evolution, Phase
                
                evolution = get_auto_evolution()
                current_phase = evolution.phase_manager.get_phase(coin, interval)
                
                # Phase 2 또는 3인 경우 진화 시스템 사용
                if current_phase >= Phase.PREDICTIVE:
                    # 전략 정보 수집 (심볼별 전략에서)
                    strategy = {}
                    if self.coin_strategies_loaded and coin in self.coin_tuned_strategies:
                        strategies = self.coin_tuned_strategies.get(coin, {})
                        if interval in strategies:
                            strategy = strategies[interval]
                        elif strategies:
                            # interval이 없으면 첫 번째 전략 사용
                            strategy = list(strategies.values())[0] if isinstance(strategies, dict) else {}
                    
                    # 진화 시스템으로 시그널 계산
                    result = evolution.calculate_signal(
                        coin=coin,
                        interval=interval,
                        candle_data=candle.to_frame().T if hasattr(candle, 'to_frame') else candle,
                        strategy=strategy
                    )
                    
                    # 결과 점수 반환
                    if result and result.score > 0:
                        logger.debug(f"🧬 {coin}/{interval} Phase {current_phase.name} 점수: {result.score:.4f}")
                        return max(0.0, min(1.0, result.score))
                        
            except ImportError:
                # 진화 시스템 없으면 기본 로직 사용
                pass
            except Exception as evo_err:
                logger.debug(f"⚠️ 진화 시스템 호출 실패 (기본 로직 사용): {evo_err}")
            
            # 🔥 Phase 1 (STATISTICAL) 또는 폴백: 상호보완 비율 혼합 방식
            
            # 1. 글로벌 전략 점수 계산
            global_score = 0.5  # 기본값
            if self.global_strategies_loaded and self.global_strategies:
                global_score = self._get_global_strategy_score(coin, interval, candle)
            
            # 2. 개별 코인 전략 점수 계산
            symbol_score = None  # None = 개별 전략 없음
            has_coin_strategy = self.coin_strategies_loaded and coin in self.coin_tuned_strategies
            
            if has_coin_strategy:
                symbol_score = self._get_symbol_strategy_score(coin, interval, candle)
            
            # 3. 🔥 동적 가중치 로드 (DB 우선, 없으면 기본값)
            coin_weight, global_weight = self._get_coin_global_weights(coin)
            
            # 4. 🔥 상호보완 점수 계산
            if symbol_score is not None and has_coin_strategy:
                # 개별 코인 전략이 있으면 비율 혼합
                base_score = (symbol_score * coin_weight) + (global_score * global_weight)
                logger.debug(f"📊 {coin}/{interval} 상호보완: 개별({symbol_score:.3f}×{coin_weight:.2f}) + 글로벌({global_score:.3f}×{global_weight:.2f}) = {base_score:.3f}")
            else:
                # 개별 코인 전략이 없으면 글로벌 전략 100% fallback
                base_score = global_score
                logger.debug(f"📊 {coin}/{interval} 글로벌 fallback: {global_score:.3f}")
            
            # 5. 신뢰도 점수 적용
            if self.reliability_scores_loaded:
                reliability_bonus = self._get_reliability_bonus(coin, interval, candle)
                base_score *= reliability_bonus
            
            # 6. AI 모델 점수 적용 (보조적 역할)
            if self.ai_model_loaded:
                ai_score = self._get_ai_model_score(coin, interval, candle)
                # AI 점수는 보조적으로 20% 반영
                base_score = (base_score * 0.8) + (ai_score * 0.2)
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            print(f"⚠️ 전략 점수 계산 실패: {e}")
            return 0.5
    
    def _get_coin_global_weights(self, coin: str) -> Tuple[float, float]:
        """🔥 개별 코인 vs 글로벌 전략 동적 가중치 로드
        
        우선순위:
        1. DB의 coin_global_weights 테이블 (absolute_zero_system에서 계산된 값)
        2. 기본값: 개별 70%, 글로벌 30%
        
        Returns:
            (coin_weight, global_weight) 튜플
        """
        try:
            # DB에서 동적 가중치 로드
            from rl_pipeline.db.reads import get_coin_global_weights
            
            weights_data = get_coin_global_weights(coin)
            
            if weights_data and weights_data.get('coin_weight') is not None:
                coin_weight = weights_data['coin_weight']
                global_weight = weights_data['global_weight']
                
                # 유효성 검증
                if 0.0 <= coin_weight <= 1.0 and 0.0 <= global_weight <= 1.0:
                    return (coin_weight, global_weight)
                    
        except ImportError:
            # rl_pipeline 없으면 기본값 사용
            pass
        except Exception as e:
            logger.debug(f"⚠️ {coin} 동적 가중치 로드 실패: {e}")
        
        # 기본값: 개별 코인 70%, 글로벌 30%
        return (0.7, 0.3)
    
    def _get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """글로벌 전략 점수 계산 (변동성 그룹 매칭 지원)"""
        try:
            if not self.global_strategies:
                return 0.5
            
            # 🎯 변동성 그룹 매칭 시도
            target_strategy = None
            try:
                # 캔들 DB 경로 찾기
                candles_db_path = None
                try:
                    from signal_selector.config import CANDLES_DB_PATH
                    candles_db_path = CANDLES_DB_PATH
                except ImportError:
                    pass
                
                if candles_db_path:
                    # 변동성 프로파일 로드
                    try:
                        from rl_pipeline.utils.coin_volatility import get_volatility_profile
                        profile = get_volatility_profile(coin, candles_db_path)
                        vol_group = profile.get('volatility_group')
                        
                        if vol_group:
                            target_type = f'volatility_group_{vol_group}'
                            # 해당 타입의 전략 찾기
                            for s in self.global_strategies:
                                if s['strategy']['type'] == target_type:
                                    target_strategy = s
                                    # print(f"🎯 {coin}: 변동성 그룹({vol_group}) 글로벌 전략 매칭 성공")
                                    break
                    except ImportError:
                        pass
            except Exception as e:
                # print(f"⚠️ 변동성 매칭 중 오류: {e}")
                pass

            # 매칭된 전략이 없으면 가장 최근(성능 좋은) 전략 사용
            latest_strategy = target_strategy if target_strategy else self.global_strategies[0]
            
            strategy = latest_strategy['strategy']
            metrics = latest_strategy['metrics']
            
            # 전략 점수 계산
            score = 0.5
            if 'performance_score' in metrics:
                score = metrics['performance_score']
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5

    def get_global_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """Public wrapper for _get_global_strategy_score"""
        return self._get_global_strategy_score(coin, interval, candle)

    def _get_symbol_strategy_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """코인별 학습 전략 점수 계산 (MFE/MAE EntryScore 기반 완전 전환)"""
        try:
            if coin not in self.coin_tuned_strategies:
                return 0.5
            
            strategies = self.coin_tuned_strategies[coin]
            if not strategies:
                return 0.5
            
            # 현재 interval에 맞는 전략 필터링
            interval_strategies = [s for s in strategies if s.get('interval') == interval]
            
            # interval 매칭되는 게 없으면 전체 전략 사용 (fallback)
            target_strategies = interval_strategies if interval_strategies else strategies
            
            if not target_strategies:
                return 0.5
            
            # 🆕 현재 레짐 추정 (간소화된 로직)
            current_regime = 'neutral'
            current_rsi = candle.get('rsi', 50)
            if current_rsi > 60:
                current_regime = 'bullish'
            elif current_rsi < 40:
                current_regime = 'bearish'
            
            # 🔥 MFE/MAE 기반 최고 점수 계산
            best_score = 0.0
            best_entry_score = None
            best_risk_score = None
            
            for strategy in target_strategies:
                tuned_params = strategy.get('tuned_parameters', {})
                metrics = strategy.get('performance_metrics', {})
                
                # 🔥 MFE/MAE 기반 점수 계산 (우선)
                entry_score = metrics.get('entry_score')
                risk_score = metrics.get('risk_score')
                
                if entry_score is not None:
                    # MFE/MAE 전략: EntryScore를 0~1 범위로 정규화
                    # EntryScore 범위: 대략 -0.05 ~ 0.05 (5% 기준)
                    # 0.02 이상 = 1.0, -0.02 이하 = 0.0, 중간 = 선형 보간
                    normalized_entry = max(0.0, min(1.0, (entry_score + 0.02) / 0.04))
                    score = 0.3 + (normalized_entry * 0.5)  # 0.3 ~ 0.8 범위
                    
                    # 리스크 조정 (RiskScore가 높으면 페널티)
                    if risk_score is not None and risk_score > 0.03:
                        score -= min(0.15, (risk_score - 0.03) * 3)
                    
                    # n_signals 표본 수 보너스 (많을수록 신뢰도 높음)
                    n_signals = metrics.get('n_signals', 0)
                    if n_signals and n_signals >= 50:
                        score += 0.1
                    elif n_signals and n_signals >= 30:
                        score += 0.05
                else:
                    # 🔧 기존 방식 fallback (MFE/MAE 통계 없는 경우)
                    score = metrics.get('success_rate', 0.5)
                    avg_reward = metrics.get('avg_reward', 0.0)
                    if avg_reward and avg_reward > 0:
                        score += min(0.2, avg_reward * 0.01)
                    
                    # 등급 보너스 (기존)
                    grade = metrics.get('quality_grade', 'C')
                    if grade == 'S': score += 0.1
                    elif grade == 'A': score += 0.05
                
                # 전략 파라미터 일치 여부
                rsi_min = tuned_params.get('rsi_min')
                rsi_max = tuned_params.get('rsi_max')
                
                if rsi_min is not None and rsi_max is not None:
                    if rsi_min <= current_rsi <= rsi_max:
                        score += 0.1
                
                # 레짐 매칭 보너스
                strategy_regime = tuned_params.get('market_regime') or tuned_params.get('regime')
                if strategy_regime:
                    if strategy_regime == current_regime:
                        score += 0.15
                    elif strategy_regime == 'neutral':
                        score += 0.05
                
                if score > best_score:
                    best_score = score
                    best_entry_score = entry_score
                    best_risk_score = risk_score
            
            return max(0.0, min(1.0, best_score))
            
        except Exception as e:
            return 0.5
    
    def _get_reliability_bonus(self, coin: str, interval: str, candle: pd.Series) -> float:
        """신뢰도 보너스 계산"""
        try:
            # 신뢰도 점수 조회
            key = f"{coin}_{interval}_buy_positive"
            if key in self.reliability_scores:
                return self.reliability_scores[key]
            
            return 1.0
            
        except Exception as e:
            return 1.0
    
    def _get_ai_model_score(self, coin: str, interval: str, candle: pd.Series) -> float:
        """AI 모델 점수 계산"""
        try:
            if not self.ai_model_loaded:
                return 0.5
            
            # 특징 추출
            features = self._extract_features(candle)
            
            if self.model_type == "pytorch":
                # PyTorch 모델 추론
                try:
                    import torch
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        prediction = self.ai_model(features_tensor)
                        score = torch.sigmoid(prediction).item()
                except ImportError:
                    print("⚠️ PyTorch를 import할 수 없습니다. 기본 점수 사용")
                    score = 0.5
            elif self.model_type == "sklearn":
                # Scikit-learn 모델 추론
                score = self.ai_model.predict_proba([features])[0][1]
            else:
                return 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5
    
    def _extract_features(self, candle: pd.Series) -> List[float]:
        """특징 추출"""
        try:
            features = []
            
            # 기본 가격 특징
            features.append(candle['open'])
            features.append(candle['high'])
            features.append(candle['low'])
            features.append(candle['close'])
            features.append(candle['volume'])
            
            # 기술적 지표
            if 'rsi' in candle:
                features.append(candle['rsi'])
            else:
                features.append(50.0)
            
            if 'macd' in candle:
                features.append(candle['macd'])
            else:
                features.append(0.0)
            
            if 'bb_upper' in candle and 'bb_lower' in candle:
                bb_position = (candle['close'] - candle['bb_lower']) / (candle['bb_upper'] - candle['bb_lower'])
                features.append(bb_position)
            else:
                features.append(0.5)
            
            return features
            
        except Exception as e:
            return [0.0] * 8

