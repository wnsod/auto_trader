"""
실시간 시그널 생성기 - RL 시스템의 학습된 전략을 활용한 실시간 매매 시그널 생성

주요 기능:
1. RL Q-table 로드 및 시그널 생성
2. 인터벌별 시그널 통합
3. DB 저장
4. 🆕 AI 모델 기반 시그널 점수 계산

🆕 Absolute Zero System 개선사항 반영:
- 모든 고급 기술지표 활용 (다이버전스, 볼린저밴드 스퀴즈, 모멘텀, 트렌드 강도 등)
- 개선된 전략 평가 방식 (시장 적응성 평가 포함)
- 향상된 상태 표현 (더 정교한 상태 키 생성)
- 새로운 패턴 매칭 로직 (다이버전스, 스퀴즈, 강한 트렌드 등)
- 멀티인터벌 상태 추적 개선 (모든 고급 지표 포함)
- �� AI 모델 기반 전략 점수 예측

🚀 고성능 시스템 최적화:
- GPU 가속 (JAX 모델 추론)
- 고성능 캐시 시스템
- 크로스 코인 학습 컨텍스트 활용
- 병렬 처리 최적화
"""
import sys
import os

# 🆕 경로 설정 개선 - rl_pipeline 및 signal_selector 모듈을 찾을 수 있도록
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)  # trade/의 상위 디렉토리 (auto_trader 루트)

# trade 디렉토리를 sys.path에 추가 (signal_selector 모듈을 찾기 위해)
sys.path.insert(0, current_dir)

# rl_pipeline 디렉토리 경로 추가
rl_pipeline_path = os.path.join(workspace_dir, 'rl_pipeline')
if os.path.exists(rl_pipeline_path):
    sys.path.insert(0, rl_pipeline_path)
    sys.path.insert(0, workspace_dir)
    print(f"✅ rl_pipeline 경로 추가: {rl_pipeline_path}")
else:
    print(f"⚠️ rl_pipeline 디렉토리를 찾을 수 없음: {rl_pipeline_path}")
    # Docker 환경을 위한 fallback
    sys.path.insert(0, '/workspace/')
    sys.path.insert(0, '/workspace/rl_pipeline')
    sys.path.insert(0, '/workspace/trade')  # signal_selector 모듈을 찾기 위해

# 🔥 엔진화: rl_pipeline이 올바른 DB 경로를 사용하도록 환경 변수 사전 설정
# signal_selector 및 rl_pipeline 모듈 import 전에 설정해야 함 (중요!)
if not os.environ.get('STRATEGY_DB_PATH') and not os.environ.get('STRATEGIES_DB_PATH'):
    # 기본 경로 설정 (Directory Mode)
    default_strat_path = os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage', 'learning_strategies')
    # 폴더가 없으면 생성하지 않고, 존재할 때만 설정 (생성은 다른 곳에서)
    if os.path.isdir(default_strat_path):
        os.environ['STRATEGY_DB_PATH'] = default_strat_path
        print(f"🔧 전략 DB 경로 자동 설정: {default_strat_path}")

_strategies_dir = os.environ.get('STRATEGY_DB_PATH') or os.environ.get('STRATEGIES_DB_PATH')
if _strategies_dir and os.path.isdir(_strategies_dir):
    # 디렉토리 모드: common_strategies.db를 learning_results DB로 사용
    _common_db = os.path.join(_strategies_dir, 'common_strategies.db')
    os.environ['LEARNING_RESULTS_DB_PATH'] = _common_db
    os.environ['GLOBAL_STRATEGY_DB_PATH'] = _common_db
    
    # 🔥 [Fix] signal_selector 내부의 DB 연결 호환성을 위해 파일 경로로 설정
    # (Loader는 dirname으로 추론 가능하지만, Connector는 파일 경로가 필수)
    os.environ['STRATEGIES_DB_PATH'] = _common_db
    os.environ['STRATEGY_DB_PATH'] = _common_db 
    os.environ['RL_STRATEGIES_DB_PATH'] = _common_db 
    
    # 🔥 [Phase 진화] DATA_STORAGE_PATH 설정 (Phase 2/3 모델 경로용)
    # learning_strategies/ 의 상위 디렉토리가 data_storage/
    _data_storage_path = os.path.dirname(_strategies_dir)
    if not os.environ.get('DATA_STORAGE_PATH'):
        os.environ['DATA_STORAGE_PATH'] = _data_storage_path
    
    print(f"🔧 엔진화: 전략 폴더 모드 감지 (환경변수 재설정)")
    print(f"   📂 전략 폴더: {_strategies_dir}")
    print(f"   🌐 공용 DB: {_common_db}")
    print(f"   📁 데이터 저장소: {_data_storage_path}")

# 🔥 엔진화: 개별 코인 DB 경로 가져오기 함수 (rl_pipeline과 동일한 로직)
def get_coin_strategy_db_path(coin: str = None) -> str:
    """개별 코인의 전략 DB 경로 반환 (Directory Mode 지원)
    
    Args:
        coin: 코인 심볼 (예: 'BTC', 'ETH')
        
    Returns:
        DB 파일 경로 (예: /workspace/.../learning_strategies/btc_strategies.db)
    """
    base_path = os.environ.get('STRATEGY_DB_PATH') or os.environ.get('STRATEGIES_DB_PATH')
    
    if not base_path:
        # 환경 변수가 없으면 기본 경로 사용 (경로 유연화)
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _root_dir = os.path.dirname(_current_dir) # auto_trader/
        base_path = os.path.join(_root_dir, 'market', 'coin_market', 'data_storage', 'learning_strategies')
    
    # 1. 디렉토리 모드인지 확인 (확장자가 .db가 아니거나, 실제 디렉토리인 경우)
    is_directory = not base_path.endswith('.db') or os.path.isdir(base_path)
    
    if is_directory:
        if not coin:
            # 코인이 지정되지 않았는데 디렉토리 모드인 경우, 기본/공용 파일 반환
            return os.path.join(base_path, 'common_strategies.db')
        
        # 코인별 파일명 생성 (소문자 변환)
        return os.path.join(base_path, f"{coin.lower()}_strategies.db")
    
    # 2. 단일 파일 모드 (기존 호환성)
    return base_path

# 전역 함수로 노출 (signal_selector에서 사용 가능)
__all__ = ['get_coin_strategy_db_path']

# 🆕 signal_selector 모듈 import (리팩토링된 모듈 구조)
try:
    from signal_selector.config import (
        USE_GPU_ACCELERATION, JAX_PLATFORM_NAME, MAX_WORKERS, CACHE_SIZE,
        ENABLE_CROSS_COIN_LEARNING, CANDLES_DB_PATH, STRATEGIES_DB_PATH,
        TRADING_SYSTEM_DB_PATH, DB_PATH, PERFORMANCE_CONFIG,
        AI_MODEL_AVAILABLE, SYNERGY_LEARNING_AVAILABLE
    )
    from signal_selector.core.types import SignalInfo, SignalAction
    from signal_selector.utils import (
        safe_float, safe_str, TECHNICAL_INDICATORS_CONFIG,
        STATE_DISCRETIZATION_CONFIG, discretize_value, process_technical_indicators,
        get_optimized_db_connection, safe_db_write, safe_db_read,
        OptimizedCache, DatabasePool
    )
    from signal_selector.evaluators import (
        OffPolicyEvaluator, ConfidenceCalibrator, MetaCorrector
    )
    print("✅ signal_selector 모듈 로드 완료")
except ImportError as e:
    print(f"❌ signal_selector 모듈 import 필수: {e}")
    raise ImportError(f"signal_selector 모듈을 찾을 수 없습니다: {e}")

# 🆕 변동성 기반 시스템 import
try:
    from rl_pipeline.utils.coin_volatility import (
        get_volatility_profile,
        calculate_coin_volatility,
        classify_volatility_group
    )
    print("✅ 변동성 시스템 로드 완료")
    VOLATILITY_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 변동성 시스템 로드 실패: {e}")
    VOLATILITY_SYSTEM_AVAILABLE = False

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import traceback
import time
import os
import math
import logging
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# 🚀 고성능 시스템 설정 (새 모듈에서 import 실패 시에만 정의)

# 🆕 자체 데이터베이스 연결 시스템 (rl_pipeline 충돌 방지)
DB_POOL_AVAILABLE = True
CONFLICT_MANAGER_AVAILABLE = True
print("✅ 자체 데이터베이스 연결 시스템 사용")

# 🆕 자체 데이터베이스 함수들 구현 (새 모듈에서 import 실패 시에만 정의)

def get_strategy_db_pool():
    """전략 데이터베이스 풀 반환 (호환성)"""
    return None

def get_candle_db_pool():
    """캔들 데이터베이스 풀 반환 (호환성)"""
    return None

def get_conflict_manager():
    """충돌 관리자 반환 (호환성)"""
    return None

# 🆕 크로스 코인 학습 설정
CROSS_COIN_AVAILABLE = os.getenv('CROSS_COIN_AVAILABLE', 'false').lower() == 'true'

# 🆕 로거 설정
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 🚀 GPU 가속 설정
if USE_GPU_ACCELERATION:
    try:
        import jax
        # JAX 로거 레벨 조정 (TPU 백엔드 경고 숨김)
        import logging as std_logging
        jax_logger = std_logging.getLogger('jax._src.xla_bridge')
        jax_logger.setLevel(std_logging.ERROR)  # ERROR 이상의 로그만 표시
        
        # 환경 변수로 TPU 백엔드 시도 방지
        os.environ.setdefault('JAX_PLATFORM_NAME', JAX_PLATFORM_NAME)
        os.environ.setdefault('XLA_PLATFORM_NAME', JAX_PLATFORM_NAME)
        
        jax.config.update('jax_platform_name', JAX_PLATFORM_NAME)
        print(f"🚀 GPU 가속 활성화: {JAX_PLATFORM_NAME}")
    except ImportError:
        print("⚠️ JAX를 import할 수 없습니다. CPU 모드로 실행됩니다.")
        USE_GPU_ACCELERATION = False
        JAX_PLATFORM_NAME = 'cpu'
        jax = None

# 🆕 AI 모델 import (signal_selector.config에서 이미 처리됨)
# signal_selector.config에서 AI_MODEL_AVAILABLE과 SYNERGY_LEARNING_AVAILABLE을 이미 설정했으므로
# 여기서는 learning_engine 클래스들만 import 시도 (없어도 정상 작동)
try:
    from learning_engine import (
        PolicyTrainer, GlobalLearningManager, SymbolFinetuningManager, 
        ShortTermLongTermSynergyLearner, ReliabilityScoreCalculator,
        ContinuousLearningManager, RoutingPatternAnalyzer, 
        ContextualLearningManager, analyze_strategy_quality
    )
    # AI_MODEL_AVAILABLE은 signal_selector.config에서 이미 설정됨
    print("✅ learning_engine 고급 기능 로드 완료")
except ImportError:
    # learning_engine이 없어도 정상 작동 (learning_strategies.db에서 직접 로드)
    # 기본값 설정
    PolicyTrainer = None
    GlobalLearningManager = None
    SymbolFinetuningManager = None
    ShortTermLongTermSynergyLearner = None
    ReliabilityScoreCalculator = None
    ContinuousLearningManager = None
    RoutingPatternAnalyzer = None
    ContextualLearningManager = None
    analyze_strategy_quality = None
    # AI_MODEL_AVAILABLE과 SYNERGY_LEARNING_AVAILABLE은 signal_selector.config에서 이미 설정됨

# 🚀 크로스 코인 학습 컨텍스트 (활성화됨)
# absolute_zero_system의 전략 가중치 시스템 활용
CROSS_COIN_AVAILABLE = True
print("ℹ️ 크로스 코인 학습 컨텍스트가 활성화되었습니다. (글로벌 전략 + 개별 전략 통합)")

# 🆕 단기-장기 시너지 학습기 상태 확인
# SYNERGY_LEARNING_AVAILABLE은 signal_selector.config에서 이미 설정됨
if not SYNERGY_LEARNING_AVAILABLE:
    # 추가 정보만 출력 (경고는 config에서 이미 출력됨)
    pass

# 🆕 유틸리티 함수들 (새 모듈에서 import 실패 시에만 정의)

# 데이터베이스 경로 (Windows 환경 지원) - 새 모듈에서 import 실패 시에만 정의

# SignalAction과 SignalInfo는 새 모듈에서 import했으므로 중복 정의 제거

# 🆕 3단계: 심화 난이도 성능 업그레이드 시스템 (새 모듈에서 import 실패 시에만 정의)
# 🆕 2단계: 보통 난이도 성능 업그레이드 시스템

# ===================================================================
# 🆕 리팩토링: 모듈화된 클래스들 import
# ===================================================================
# SignalSelector - 핵심 시그널 선택 클래스
from signal_selector.core.selector import SignalSelector

# StrategyScoreCalculator - 전략 점수 계산기
from signal_selector.scoring import StrategyScoreCalculator

# Helper 클래스들 (필요시 사용)
from signal_selector.helpers import (
    ContextualBandit, RegimeChangeDetector, ExponentialDecayWeight,
    BayesianSmoothing, ActionSpecificScorer, ContextFeatureExtractor,
    OutlierGuardrail, EvolutionEngine, ContextMemory, RealTimeLearner,
    SignalTradeConnector
)

print("✅ 모듈화된 클래스들 import 완료")

def check_and_repair_db(db_path):
    """DB 무결성 검사 및 자동 복구 시도 (스키마 손상 대응)"""
    if not os.path.exists(db_path):
        return

    try:
        import sqlite3
        # 먼저 빠른 무결성 검사 시도 (스키마 로드)
        with sqlite3.connect(db_path) as conn:
            conn.execute("SELECT count(*) FROM sqlite_master")
            
    except sqlite3.DatabaseError as e:
        # 스키마 손상 에러 감지
        if "malformed database schema" in str(e) or "invalid rootpage" in str(e):
            print(f"⚠️ {os.path.basename(db_path)} 스키마 손상 감지. 자동 복구 시도 중...")
            try:
                # 독립적인 연결로 복구 시도
                with sqlite3.connect(db_path) as repair_conn:
                    repair_conn.execute("PRAGMA writable_schema = 1")
                    # 손상된 인덱스 제거 시도 (가장 흔한 원인인 idx_signals_ts)
                    repair_conn.execute("DELETE FROM sqlite_master WHERE type='index' AND name='idx_signals_ts'")
                    repair_conn.commit()
                    repair_conn.execute("PRAGMA writable_schema = 0")
                    # VACUUM은 시간이 걸릴 수 있으므로, 파일이 너무 크면 생략하거나 주의 필요
                    # 여기서는 안전하게 인덱스 제거만 수행
                    # repair_conn.execute("VACUUM") 
                print("✅ DB 인덱스 복구 완료 (손상된 인덱스 정의 제거)")
            except Exception as repair_err:
                print(f"❌ DB 복구 실패: {repair_err}. (파일 백업 후 삭제 권장)")
        else:
            print(f"⚠️ DB 연결 오류: {e}")

def main():
    """🚀 고성능 실시간 시그널 선택기 메인 실행 함수"""
    # 🆕 DB 경로 강제 보정 (표준 경로 준수)
    default_db_dir = os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage')
    try:
        os.makedirs(default_db_dir, exist_ok=True)
    except:
        pass
            
    if not os.environ.get('TRADING_SYSTEM_DB_PATH'):
        os.environ['TRADING_SYSTEM_DB_PATH'] = os.path.join(default_db_dir, 'trading_system.db')

    # 🆕 DB 무결성 사전 검사
    check_and_repair_db(os.environ['TRADING_SYSTEM_DB_PATH'])

    print("🚀 고성능 실시간 시그널 선택기 시작")
    print("🎯 목표: GPU 가속 + 크로스 코인 학습 통합 시그널 생성")
    print("🆕 고성능 캐시, 병렬 처리, 적응형 AI 모델 선택")
    print("=" * 60)
    
    # 🚀 고성능 시스템 설정 표시
    print("🚀 고성능 시스템 설정:")
    print(f"   - GPU 가속: {USE_GPU_ACCELERATION}")
    print(f"   - JAX 플랫폼: {JAX_PLATFORM_NAME}")
    print(f"   - 병렬 워커: {MAX_WORKERS}")
    print(f"   - 캐시 크기: {CACHE_SIZE:,}")
    print(f"   - 크로스 코인 학습: {ENABLE_CROSS_COIN_LEARNING}")
    print("=" * 60)
    
    try:
        # 시그널 선택기 초기화
        selector = SignalSelector()
        
        # 🚀 고성능 시스템 상태 확인
        print("\n🔍 고성능 시스템 상태 확인 중...")
        
        # 🚀 AI 모델 상태 확인
        if selector.ai_model_loaded:
            print("✅ AI 모델 로드 완료 - GPU 가속 AI 기반 시그널 점수 계산 활성화")
            print(f"   - 모델 타입: {selector.model_type}")
            print(f"   - GPU 가속: {USE_GPU_ACCELERATION}")
        else:
            # AI 모델이 없어도 정상 작동 (learning_strategies.db에서 직접 전략 로드)
            print("ℹ️ AI 모델 미사용 - 데이터베이스 기반 전략 로드 방식 사용 (정상)")
        
        # 🚀 크로스 코인 학습 상태 확인
        if selector.cross_coin_available:
            print("✅ 크로스 코인 학습 컨텍스트 로드 완료")
        else:
            # 크로스 코인 학습은 의도적으로 비활성화됨 (복잡한 의존성 문제로 인해 간소화)
            print("ℹ️ 크로스 코인 학습 컨텍스트 비활성화 (의도적 설정 - 정상)")
        
        # 🚀 캐시 시스템 상태 확인
        print(f"✅ 고성능 캐시 시스템: 최대 {selector.max_cache_size:,}개 항목")
        
        # 🆕 시스템 상태 확인
        print("\n🔍 데이터베이스 상태 확인 중...")
        
        # 데이터베이스 연결 확인
        try:
            with sqlite3.connect(CANDLES_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM candles")
                candle_count = cursor.fetchone()[0]
                print(f"  ✅ 캔들 데이터: {candle_count:,}개")
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM candles")
                coin_count = cursor.fetchone()[0]
                print(f"  ✅ 코인 수: {coin_count}개")
        except Exception as e:
            print(f"  ❌ 캔들 DB 연결 실패: {e}")
                
        try:
            # signals 테이블 존재 여부 확인 (TRADING_SYSTEM_DB_PATH 사용)
            with sqlite3.connect(TRADING_SYSTEM_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*) FROM signals")
                    signal_count = cursor.fetchone()[0]
                    print(f"  ✅ 기존 시그널: {signal_count:,}개")
                else:
                    print(f"  ℹ️ 시그널 테이블이 아직 생성되지 않았습니다 (TRADING_SYSTEM_DB_PATH)")
        except Exception as e:
            print(f"  ❌ 트레이딩 시스템 DB 연결 실패: {e}")
        
        print("✅ 시스템 상태 확인 완료")
        print("-" * 60)
        
        # �� 전체 코인 멀티인터벌 시그널 생성 (성능 최적화)
        print("\n🧪 전체 코인 멀티인터벌 시그널 생성 중...")
        
        # 1. 사용 가능한 코인 및 인터벌 자동 감지 (Dynamic Discovery)
        try:
            with sqlite3.connect(CANDLES_DB_PATH) as conn:
                # DB에 존재하는 모든 코인과 인터벌 조회
                rows = pd.read_sql("""
                    SELECT DISTINCT symbol as coin, interval
                    FROM candles 
                    ORDER BY symbol
                """, conn)
        except Exception as e:
            print(f"❌ 코인/인터벌 조회 실패: {e}")
            return
        
        if rows.empty:
            print("❌ 사용 가능한 코인/캔들 데이터가 없습니다")
            return

        # 2. 코인별 사용 가능한 인터벌 그룹핑
        coin_intervals_map = defaultdict(list)
        for _, row in rows.iterrows():
            coin_intervals_map[row['coin']].append(row['interval'])

        # 인터벌 정렬 헬퍼 함수 (분 단위 변환)
        def get_minutes(iv):
            iv = iv.lower()
            try:
                if iv.endswith('m'): return int(iv[:-1])
                if iv.endswith('h'): return int(iv[:-1]) * 60
                if iv.endswith('d'): return int(iv[:-1]) * 1440
                if iv.endswith('w'): return int(iv[:-1]) * 10080
            except: pass
            return 999999

        print(f"📊 총 {len(coin_intervals_map)}개 코인 감지됨")
        
        # 3. 코인별 멀티인터벌 시그널 생성
        combined_signals = []
        
        for coin, available_intervals in coin_intervals_map.items():
            # 인터벌을 시간 순서로 정렬 (단기 -> 장기)
            # 예: ['15m', '30m', '240m', '1d']
            sorted_intervals = sorted(available_intervals, key=get_minutes)
            
            # 너무 적은 인터벌은 건너뛰기 (최소 2개 이상 권장)
            if len(sorted_intervals) < 2:
                # print(f"⚠️ {coin}: 인터벌 부족 ({sorted_intervals}), 건너뜀")
                continue

            try:
                # 각 인터벌별 시그널 생성
                interval_signals = {}
                for interval in sorted_intervals:
                    signal = selector.generate_signal(coin, interval)
                    if signal:
                        interval_signals[interval] = signal
                
                # 멀티인터벌 시그널 결합 (DB 기반 동적 가중치 사용)
                if len(interval_signals) >= 2:  # 최소 2개 인터벌 이상 있어야 결합
                    combined_signal = selector.combine_multi_timeframe_signals(coin, interval_signals)
                    if combined_signal:
                        combined_signals.append(combined_signal)

                        # 🔥 통합 시그널 DB 저장
                        try:
                            selector.save_signal_to_db(combined_signal)
                        except Exception as save_err:
                            print(f"⚠️ {coin} 통합 시그널 DB 저장 실패: {save_err}")

                        # 🔥 코인 종합 점수 명확하게 출력
                        print(f"\n{'='*60}")
                        print(f"🎯 [{coin}] 최종 종합 시그널 (멀티인터벌 통합)")
                        print(f"{'='*60}")
                        
                        # 🔧 안전한 포맷팅 헬퍼
                        def _s(val, fmt=".4f"):
                            if val is None: return "N/A"
                            try: return f"{val:{fmt}}"
                            except: return str(val)

                        print(f"  📊 종합 점수: {_s(combined_signal.signal_score)}")
                        print(f"  📊 신뢰도: {_s(combined_signal.confidence)}")
                        # 🔧 액션은 시그널이 아닌 트레이더가 결정 (사용자 요청: 액션 노출 제거)
                        # print(f"  🎯 최종 액션: {combined_signal.action.value.upper()}")
                        print(f"  📈 사용된 인터벌: {len(interval_signals)}개 ({', '.join(interval_signals.keys())})")
                        
                        # 🆕 현재가 출력 포맷팅 함수 재사용
                        def _fmt_p(p):
                            if p is None: return "0"
                            if p < 1: return f"{p:.4f}"
                            if p < 100: return f"{p:.2f}"
                            return f"{int(p):,}"
                            
                        print(f"  💰 현재가: {_fmt_p(combined_signal.price)}원")
                        # 🆕 예상 목표가 출력 (달러 표시 제거 및 포맷팅 적용)
                        if hasattr(combined_signal, 'target_price') and combined_signal.target_price and combined_signal.target_price > 0 and combined_signal.price and combined_signal.price > 0:
                            # 🔧 목표가 유효성 검증 (현재가의 50%~200% 범위 내만 유효)
                            ratio = combined_signal.target_price / combined_signal.price
                            if 0.5 <= ratio <= 2.0:
                                expected_profit = (ratio - 1.0) * 100
                                
                                # 🕒 예상 소요 시간 추정 (인터벌 기반)
                                time_est_map = {
                                    '1m': "약 15분 내", '3m': "약 30분 내", '5m': "약 1시간 내",
                                    '15m': "약 4시간 내", '30m': "약 8시간 내", '60m': "약 12시간 내", '1h': "약 12시간 내",
                                    '240m': "약 24시간 내", '4h': "약 24시간 내", '1d': "약 3일 내", '1w': "약 1주 내"
                                }
                                est_time = time_est_map.get(combined_signal.interval, "단기~중기")
                                
                                # 🚨 [Safety] 목표가 상한 제한 (100% 이상 수익은 비현실적)
                                if expected_profit > 100.0:
                                    expected_profit = 100.0
                                    combined_signal.target_price = combined_signal.price * 2.0
                                    print(f"  🎯 예상 목표가 (보정됨): {_fmt_p(combined_signal.target_price)}원 (예상 수익: +100.00% [Max Cap], 도달 예상: {est_time})")
                                else:
                                    print(f"  🎯 예상 목표가: {_fmt_p(combined_signal.target_price)}원 (예상 수익: {expected_profit:+.2f}%, 도달 예상: {est_time})")
                        
                        print(f"  📊 RSI: {_s(combined_signal.rsi, '.2f')} | MFI: {_s(combined_signal.mfi, '.2f')} (자금흐름)")
                        print(f"  📊 MACD: {_s(combined_signal.macd, '.6f')} | ADX: {_s(combined_signal.adx, '.2f')} (추세강도)")
                        
                        # 🆕 변수 정의 복구 (volatility, vol_target_str)
                        volatility = combined_signal.volatility if combined_signal.volatility is not None else 0.0
                        price = combined_signal.price if combined_signal.price is not None else 0.0
                        volatility_amount = price * volatility
                        
                        direction_upper = combined_signal.integrated_direction.upper() if combined_signal.integrated_direction else 'NEUTRAL'
                        score = combined_signal.signal_score if combined_signal.signal_score is not None else 0.0
                        
                        vol_target_str = ""
                        if 'LONG' in direction_upper or 'BUY' in direction_upper or score > 0.6:
                            target = price + volatility_amount
                            vol_target_str = f"상방 목표 {_fmt_p(target)}원 (+{volatility*100:.2f}%)"
                        elif 'SHORT' in direction_upper or 'SELL' in direction_upper or score < 0.4:
                            target = price - volatility_amount
                            vol_target_str = f"하방 목표 {_fmt_p(target)}원 (-{volatility*100:.2f}%)"
                        else:
                            upper = price + volatility_amount
                            lower = price - volatility_amount
                            vol_target_str = f"변동 범위 {_fmt_p(lower)} ~ {_fmt_p(upper)}원 (±{volatility*100:.2f}%)"

                        # 🆕 변동성 및 밴드 정보 통합 출력
                        vol_info = f"변동성: {_s(volatility)}"
                        if hasattr(combined_signal, 'bb_width') and combined_signal.bb_width:
                             vol_info += f" | BB폭: {_s(combined_signal.bb_width, '.4f')}"
                        if hasattr(combined_signal, 'bb_squeeze') and combined_signal.bb_squeeze > 0.7:
                             vol_info += " (⚡Squeeze)"
                        print(f"  📊 {vol_info} -> {vol_target_str}")
                        
                        print(f"  📊 거래량 비율: {_s(combined_signal.volume_ratio, '.2f')}x | 모멘텀: {_s(combined_signal.price_momentum, '.2f')}")
                        print(f"  🌊 파동: {combined_signal.wave_phase} ({combined_signal.elliott_wave})")
                        print(f"  🏛️ 구조: {combined_signal.market_structure} | 패턴: {combined_signal.pattern_type}")
                        
                        # 다이버전스 발견 시 출력
                        divs = []
                        if combined_signal.rsi_divergence and combined_signal.rsi_divergence != 'none': divs.append(f"RSI {combined_signal.rsi_divergence}")
                        if combined_signal.macd_divergence and combined_signal.macd_divergence != 'none': divs.append(f"MACD {combined_signal.macd_divergence}")
                        if divs:
                            print(f"  ⚠️ 감지된 다이버전스: {', '.join(divs)}")

                        print(f"  🎯 통합 방향: {combined_signal.integrated_direction}")
                        print(f"{'='*60}\n")

                        print(f"✅ {coin}: 멀티인터벌 시그널 생성 성공 ({len(interval_signals)}개 인터벌)")
                    else:
                        print(f"⚠️ {coin}: 멀티인터벌 시그널 결합 실패")
                else:
                    print(f"⚠️ {coin}: 충분한 인터벌 데이터 없음 ({len(interval_signals)}개)")
                    
            except Exception as e:
                print(f"❌ {coin}: 시그널 생성 오류 - {e}")
            
            # 🔥 [리소스 최적화] 코인별 처리 후 DB 연결 즉시 해제 (파일 핸들 누수 방지)
            try:
                # 1. rl_pipeline의 연결 풀 정리 (코인별 DB 연결 해제)
                # sys.modules 확인으로 모듈 로드 여부 체크
                if 'rl_pipeline.db.connection_pool' in sys.modules:
                    from rl_pipeline.db.connection_pool import close_and_remove_strategy_pool
                    coin_db_path = get_coin_strategy_db_path(coin)
                    close_and_remove_strategy_pool(coin_db_path)
            except Exception:
                pass
        
        print(f"\n📊 멀티인터벌 시그널 생성 결과: {len(combined_signals)}/{len(coin_intervals_map)}개 코인")
        
        # 🆕 통계 카운터 수동 업데이트 (main 함수에서 생성된 시그널들)
        selector._signal_stats['total_signals_generated'] += len(combined_signals)
        
        # 🆕 시너지 학습 결과 활용 테스트 (불필요한 테스트 제거)
        # if selector.synergy_learning_available:
        #     print("\n🔄 시너지 학습 결과 활용 테스트...")
        #     selector._test_synergy_learning_integration()
        selector._signal_stats['successful_signals'] += len(combined_signals)
        
        # 🆕 상세한 통계 출력
        selector._log_signal_stats()
        
        print("\n✅ 실시간 시그널 선택기 테스트 완료")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
    finally:
        print("\n🎉 시스템 종료")

# ============================================================================
# 🆕 전략 점수 계산기 클래스는 이미 위에 정의됨 (중복 제거)
# ============================================================================

def save_dimension_info_to_db(coin: str, dimension_info: dict):
    """차원 정보를 데이터베이스에 저장 (개별 코인 DB 우선)"""
    try:
        import sqlite3
        
        # 🔥 엔진화: 개별 코인 DB 경로 사용
        db_path = get_coin_strategy_db_path(coin)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # dimension_info 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dimension_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT,
                    dimension_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 기존 데이터 삭제 (최신 차원 정보만 유지)
            cursor.execute("DELETE FROM dimension_info WHERE coin = ?", (coin,))
            
            # 새로운 차원 정보 저장
            cursor.execute("""
                INSERT INTO dimension_info (coin, dimension_data)
                VALUES (?, ?)
            """, (coin, json.dumps(dimension_info, ensure_ascii=False)))
            
            conn.commit()
            logger.info(f"✅ {coin} 차원 정보 데이터베이스 저장 완료")
            
    except Exception as e:
        logger.error(f"❌ {coin} 차원 정보 저장 실패: {e}")

def load_dimension_info_from_db(coin: str) -> dict:
    """데이터베이스에서 차원 정보 로드 (개별 코인 DB 우선)"""
    try:
        import sqlite3
        
        # 🔥 엔진화: 개별 코인 DB 경로 사용
        db_path = get_coin_strategy_db_path(coin)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT dimension_data FROM dimension_info WHERE coin = ? ORDER BY created_at DESC LIMIT 1", (coin,))
            row = cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            else:
                return {}
                
    except Exception as e:
        logger.error(f"❌ {coin} 차원 정보 로드 실패: {e}")
        return {}

def _load_learned_strategies_from_db():
    """데이터베이스에서 학습된 전략 로드 (글로벌/개별 분리 확인)"""
    try:
        # learning_strategies.db에서 coin_strategies 로드
        from signal_selector.config import STRATEGIES_DB_PATH
        
        # 🔥 [Fix] 파일 경로라면 디렉토리로 변환하여 처리 (호환성)
        target_path = STRATEGIES_DB_PATH
        if target_path.endswith('.db'):
            target_dir = os.path.dirname(target_path)
            if os.path.isdir(target_dir):
                target_path = target_dir
        
        # 🆕 디렉토리 모드 지원 (개별 코인 DB + 공용 DB)
        if os.path.isdir(target_path):
            print(f"📂 전략 저장소(폴더) 감지: {target_path}")
            
            # 1. 글로벌 전략 확인 (common_strategies.db)
            common_path = os.path.join(target_path, "common_strategies.db")
            if os.path.exists(common_path):
                try:
                    with sqlite3.connect(common_path) as conn:
                        cursor = conn.cursor()
                        # global_strategies 테이블 확인
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_strategies'")
                        if cursor.fetchone():
                            cursor.execute("SELECT COUNT(*) FROM global_strategies")
                            global_count = cursor.fetchone()[0]
                            print(f"  ✅ 글로벌 전략: {global_count:,}개 (common_strategies.db)")
                        else:
                            print("  ℹ️ 글로벌 전략 테이블 없음 (common_strategies.db)")
                except Exception as e:
                    print(f"  ⚠️ 글로벌 전략 DB 확인 실패: {e}")
            else:
                print("  ⚠️ 공용 DB(common_strategies.db)가 없습니다.")

            # 2. 개별 코인 전략 확인 (DB 파일 스캔)
            # 매번 전체 스캔은 비효율적이므로, 필요할 때 로드하도록 변경하거나 요약 정보만 출력
            db_files = [f for f in os.listdir(target_path) if f.endswith('_strategies.db')]
            
            if db_files:
                print(f"  ✅ 개별 코인 DB 파일: {len(db_files)}개 발견")
                # print(f"     - 파일 목록 (일부): {', '.join(db_files[:5])} ...") # 로그 간소화
                
                # 샘플 확인 (첫 번째 파일만)
                if len(db_files) > 0:
                    sample_db = os.path.join(target_path, db_files[0])
                    try:
                        with sqlite3.connect(sample_db) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")
                            if cursor.fetchone():
                                cursor.execute("SELECT COUNT(*) FROM strategies")
                                count = cursor.fetchone()[0]
                                print(f"     - 샘플 확인 ({db_files[0]}): 전략 {count:,}개 로드 가능")
                    except:
                        pass
            else:
                print("  ⚠️ 개별 코인 전략 DB 파일이 없습니다 (아직 학습된 코인 없음).")
            
            return

        # (이하 레거시 단일 파일 모드)
        rl_strategies_db = STRATEGIES_DB_PATH
        conn = sqlite3.connect(rl_strategies_db)
        cursor = conn.cursor()

        # coin_strategies 테이블에서 전략 로드
        try:
            cursor.execute("SELECT COUNT(*) FROM coin_strategies")
            coin_count = cursor.fetchone()[0]
            print(f"📊 코인별 전략 {coin_count:,}개 발견 (learning_strategies.db)")
        except sqlite3.OperationalError:
            print(f"⚠️ coin_strategies 테이블이 존재하지 않음")
            coin_count = 0

        # 글로벌 전략도 확인 (있으면)
        try:
            cursor.execute("SELECT COUNT(*) FROM global_strategies")
            global_count = cursor.fetchone()[0]
            print(f"📊 글로벌 전략 {global_count:,}개 발견")
        except:
            print(f"ℹ️ global_strategies 테이블 없음")

        conn.close()

    except Exception as e:
        print(f"⚠️ 학습된 전략 로드 실패: {e}")

def _create_strategy_based_ai_model():
    """학습된 전략 기반 AI 모델 생성"""
    try:
        feature_dim = 50  # 기본 차원
        ai_model = PolicyTrainer(feature_dim=feature_dim)
        model_type = "strategy_based"
        print(f"✅ 전략 기반 AI 모델 생성 완료 (차원: {feature_dim})")
        return ai_model, model_type
        
    except Exception as e:
        print(f"⚠️ 전략 기반 AI 모델 생성 실패: {e}")
        return _create_default_ai_model()

def _create_default_ai_model():
    """기본 AI 모델 생성"""
    try:
        feature_dim = 50
        ai_model = PolicyTrainer(feature_dim=feature_dim)
        model_type = "default"
        print(f"✅ 기본 AI 모델 생성 완료 (차원: {feature_dim})")
        return ai_model, model_type
        
    except Exception as e:
        print(f"⚠️ 기본 AI 모델 생성 실패: {e}")
        return None, "none"

if __name__ == "__main__":
    main()