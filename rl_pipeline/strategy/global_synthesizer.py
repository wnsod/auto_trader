"""
글로벌 전략 Synthesizer
개별 코인 전략들을 종합하여 글로벌 전략 생성
"""

import json
import hashlib
import logging
import math
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# 폴백 템플릿 상수
FALLBACK_TEMPLATES = {
    "trend_follow": {
        "rsi_min": 35.0,
        "rsi_max": 80.0,
        "atr_mult": 1.8,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.08,
        "volume_ratio_min": 1.2,
        "market_condition": "trending"
    },
    "mean_reversion": {
        "rsi_min": 25.0,
        "rsi_max": 60.0,
        "atr_mult": 1.2,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "volume_ratio_min": 0.8,
        "market_condition": "ranging"
    },
    "balanced": {
        "rsi_min": 30.0,
        "rsi_max": 70.0,
        "atr_mult": 1.5,
        "stop_loss_pct": 0.025,
        "take_profit_pct": 0.06,
        "volume_ratio_min": 1.0,
        "market_condition": "neutral"
    },
}

class GlobalStrategySynthesizer:
    """글로벌 전략 Synthesizer - 개별 코인 전략 종합"""
    
    def __init__(self, db_path: str, intervals: List[str], seed: int = 42):
        self.db_path = db_path
        self.intervals = intervals
        self.seed = seed
        
        # 재현성 보장을 위한 시드 설정
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"🚀 GlobalStrategySynthesizer 초기화 (seed={seed})")
    
    # ==================== 1단계: 수집 ====================
    def load_pool(
        self, 
        coins: Optional[List[str]] = None,
        min_trades: int = 30,
        max_dd: float = 0.6
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        개별 코인 전략 수집 (Directory Mode 지원)
        
        Args:
            coins: 특정 코인만 필터링 (None이면 전체)
            min_trades: 최소 거래 횟수
            max_dd: 최대 낙폭 임계값
            
        Returns:
            {interval: [strategy_dict, ...]} 형태의 딕셔너리
        """
        try:
            logger.info(f"📊 개별 전략 수집 시작 (min_trades={min_trades}, max_dd={max_dd})")
            
            pool = defaultdict(list)
            import os
            import glob
            
            # Directory Mode인지 확인 (디렉토리이거나 확장자가 없는 경우 디렉토리로 간주)
            is_directory_mode = os.path.isdir(self.db_path) or not self.db_path.endswith('.db')
            
            db_files = []
            
            if is_directory_mode:
                if not os.path.exists(self.db_path):
                    logger.warning(f"⚠️ 전략 DB 디렉토리가 존재하지 않습니다: {self.db_path}")
                    return {}
                    
                # 코인 필터가 있으면 해당 코인 파일만 찾기
                if coins:
                    for coin in coins:
                        # 대소문자 무시 매칭을 위해 glob 사용보다는 직접 구성 시도
                        # 하지만 파일시스템 대소문자 구분 여부에 따라 다를 수 있음
                        # 여기선 소문자 변환하여 시도
                        fpath = os.path.join(self.db_path, f"{coin.lower()}_strategies.db")
                        if os.path.exists(fpath):
                            db_files.append(fpath)
                else:
                    # 모든 *_strategies.db 파일 찾기
                    db_files = glob.glob(os.path.join(self.db_path, "*_strategies.db"))
            else:
                # Single File Mode
                if os.path.exists(self.db_path):
                    db_files = [self.db_path]
                else:
                    logger.warning(f"⚠️ 전략 DB 파일이 존재하지 않습니다: {self.db_path}")
                    return {}
            
            total_loaded = 0
            
            for db_file in db_files:
                try:
                    with sqlite3.connect(db_file) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()
                        
                        # 쿼리 실행
                        # 코인 필터는 파일 선택 단계에서 이미 적용되었거나(Directory Mode),
                        # Single File Mode에서는 쿼리로 적용해야 함
                        
                        where_clauses = ["trades_count >= ?", "max_drawdown <= ?"]
                        params = [min_trades, max_dd]
                        
                        if not is_directory_mode and coins:
                            placeholders = ','.join(['?' for _ in coins])
                            where_clauses.append(f"coin IN ({placeholders})")
                            params.extend(coins)
                        
                        query = f"""
                            SELECT * FROM strategies
                            WHERE {' AND '.join(where_clauses)}
                            ORDER BY 
                                CASE quality_grade
                                    WHEN 'S' THEN 0
                                    WHEN 'A' THEN 1
                                    WHEN 'B' THEN 2
                                    ELSE 3
                                END,
                                profit DESC,
                                win_rate DESC
                        """
                        
                        cursor.execute(query, params)
                        results = cursor.fetchall()
                        
                        for row in results:
                            strategy = dict(row)
                            interval = strategy.get('interval', '15m')
                            pool[interval].append(strategy)
                            total_loaded += 1
                            
                except Exception as db_err:
                    # 개별 DB 로드 실패는 로그만 남기고 계속 진행
                    # logger.debug(f"⚠️ DB 로드 실패 ({os.path.basename(db_file)}): {db_err}")
                    pass
            
            # 통계 출력
            for interval, strategies in pool.items():
                logger.info(f"  ✅ {interval}: {len(strategies)}개 전략")
            
            logger.info(f"✅ 개별 전략 수집 완료: {total_loaded}개 (총 {len(db_files)}개 파일 스캔)")
            return dict(pool)
                
        except Exception as e:
            logger.error(f"❌ 개별 전략 수집 실패: {e}")
            return {}
    
    # ==================== 2단계: 표준화 ====================
    def standardize(self, pool: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        전략 파라미터 표준화 (IQR 방식) - 최적화 버전
        """
        try:
            logger.info("📊 전략 파라미터 표준화 시작 (최적화 모드)")
            import numpy as np
            
            standardized_pool = {}
            # 키 파라미터 추출
            key_params = ['rsi_min', 'rsi_max', 'stop_loss_pct', 'take_profit_pct', 
                        'volume_ratio_min', 'volume_ratio_max']
            
            for interval, strategies in pool.items():
                if not strategies:
                    continue
                
                # 1. 각 파라미터별 통계량(IQR) 사전 계산 (인터벌별 1회)
                param_stats = {}
                for param in key_params:
                    # 유효한 숫자 값만 추출
                    values = [s.get(param) for s in strategies if s.get(param) is not None]
                    if not values:
                        continue
                    
                    v_arr = np.array(values)
                    q1 = np.percentile(v_arr, 25)
                    q3 = np.percentile(v_arr, 75)
                    iqr = q3 - q1 if q3 > q1 else 1.0
                    
                    param_stats[param] = {
                        'q1_q3_avg': (q1 + q3) / 2,
                        'iqr': iqr
                    }
                
                # 2. 사전 계산된 통계량으로 각 전략 변환 (O(N))
                standardized_strategies = []
                for strategy in strategies:
                    std_strategy = strategy.copy()
                    
                    # 표준화된 값 저장
                    std_strategy['_standardized'] = {}
                    
                    for param, stats in param_stats.items():
                        val = strategy.get(param)
                        if val is not None:
                            # IQR 방식 Z-Score
                            z_score = (val - stats['q1_q3_avg']) / stats['iqr']
                            std_strategy['_standardized'][param] = float(z_score)
                    
                    standardized_strategies.append(std_strategy)
                
                standardized_pool[interval] = standardized_strategies
                logger.info(f"  ✅ {interval}: {len(standardized_strategies)}개 표준화 완료")
            
            logger.info("✅ 표준화 완료")
            return standardized_pool
            
        except Exception as e:
            logger.error(f"❌ 표준화 실패: {e}")
            return pool
    
    # ==================== 3단계: 공통 패턴 추출 ====================
    def extract_common_patterns(
        self, 
        std_pool: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        7대 레짐별 공통 패턴 추출 (중간값 기반)
        """
        try:
            logger.info("📊 7대 레짐별 공통 패턴 추출 시작 (중간값 합성 방식)")
            
            import numpy as np
            patterns = {}
            
            # 공식 7대 레짐 정의
            OFFICIAL_REGIMES = [
                'extreme_bearish', 'bearish', 'sideways_bearish', 
                'neutral', 
                'sideways_bullish', 'bullish', 'extreme_bullish'
            ]
            
            for interval, strategies in std_pool.items():
                if not strategies:
                    continue
                
                # 공식 7대 레짐별로 그룹화
                regime_bins = defaultdict(list)
                for s in strategies:
                    # 7대 레짐 명칭 표준화 매핑
                    r = (s.get('regime') or s.get('market_condition') or 'neutral').lower()
                    
                    # 명칭 정규화 (3단계 레짐 등이 섞여있을 경우 대비)
                    if r in ['strong_bullish', 'uptrend']: r = 'extreme_bullish'
                    elif r in ['strong_bearish', 'downtrend']: r = 'extreme_bearish'
                    elif r in ['ranging', 'sideways']: r = 'neutral'
                    
                    # 공식 명칭에 포함되지 않으면 neutral로 수렴
                    if r not in OFFICIAL_REGIMES:
                        r = 'neutral'
                        
                    regime_bins[r].append(s)
                
                pattern_specs = []
                for regime in OFFICIAL_REGIMES:
                    bin_strategies = regime_bins.get(regime, [])
                    
                    if not bin_strategies:
                        # 해당 레짐에 데이터가 없으면 폴백 템플릿 사용 (지도의 빈 칸 채우기)
                        logger.debug(f"  ℹ️ {interval}-{regime}: 데이터 부족, 기본 템플릿 기반 생성")
                        template_map = {
                            'extreme_bullish': FALLBACK_TEMPLATES['trend_follow'],
                            'bullish': FALLBACK_TEMPLATES['trend_follow'],
                            'sideways_bullish': FALLBACK_TEMPLATES['balanced'],
                            'neutral': FALLBACK_TEMPLATES['balanced'],
                            'sideways_bearish': FALLBACK_TEMPLATES['balanced'],
                            'bearish': FALLBACK_TEMPLATES['mean_reversion'],
                            'extreme_bearish': FALLBACK_TEMPLATES['mean_reversion']
                        }
                        median_params = template_map.get(regime, FALLBACK_TEMPLATES['balanced']).copy()
                        support = 0.0
                        pf_avg = 1.0
                        tr_avg = 0.0
                        count = 0
                    else:
                        # 실제 데이터 기반 중간값 계산
                        median_params = self._calculate_median_params(bin_strategies)
                        support = len(bin_strategies) / len(strategies)
                        pf_avg = np.median([s.get('profit_factor', 1.0) for s in bin_strategies])
                        tr_avg = np.median([s.get('profit', 0.0) for s in bin_strategies])
                        count = len(bin_strategies)
                    
                    pattern_spec = {
                        'regime': regime,
                        'params': median_params,
                        'support': float(support),
                        'pf_avg': float(pf_avg),
                        'tr_avg': float(tr_avg),
                        'count': count
                    }
                    
                    pattern_specs.append(pattern_spec)
                
                patterns[interval] = pattern_specs
                logger.info(f"  ✅ {interval}: 7대 레짐 지도 완성 (데이터 기반: {len([p for p in pattern_specs if p['count'] > 0])}구역)")
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ 패턴 추출 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    # ==================== 4단계: 전역 전략화 ====================
    def assemble_global_strategies(
        self,
        patterns: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        패턴으로부터 글로벌 전략 생성 (레짐별 대표 전략)
        
        Args:
            patterns: {interval: [pattern_spec, ...]}
            
        Returns:
            {interval: [global_strategy_dict, ...]}
        """
        try:
            logger.info("📊 글로벌 전략 지도 조립 시작")
            
            global_strategies = {}
            
            for interval, pattern_specs in patterns.items():
                interval_strategies = []
                
                # 각 레짐별로 가장 보편적인(중간값) 전략 1개씩 생성
                for pattern in pattern_specs:
                    regime = pattern['regime']
                    params = pattern['params']
                    
                    # dna_hash 생성
                    dna_hash = self._make_dna_hash(params)
                    
                    # 글로벌 전략 생성
                    global_strategy = {
                        'id': f"GLOBAL_{interval}_{regime}_{dna_hash[:6]}",
                        'market_type': 'COIN',
                        'market': 'BITHUMB',
                        'symbol': 'GLOBAL', # symbol로 통일
                        'interval': interval,
                        'strategy_type': 'universal_median',
                        'params': params,
                        'name': f'Global {regime} Strategy',
                        'description': f'Synthesized from {pattern["count"]} {regime} strategies using median',
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'dna_hash': dna_hash,
                        'source_type': 'synthesized',
                        
                        # 성능 메타 (합성 데이터이므로 중간값 성과 기록)
                        'profit': pattern.get('tr_avg', 0.0),
                        'profit_factor': pattern.get('pf_avg', 1.0),
                        'win_rate': 0.5,
                        'trades_count': pattern.get('count', 0),
                        'quality_grade': 'S' if pattern.get('pf_avg', 0) > 1.5 else 'A',
                        'market_condition': regime,
                        'regime': regime,
                        
                        # 메타 정보
                        '_meta': {
                            'support': pattern.get('support', 0.0),
                            'pattern_count': pattern.get('count', 0),
                            'source': 'median_synthesis'
                        }
                    }
                    
                    interval_strategies.append(global_strategy)
                
                global_strategies[interval] = interval_strategies
                logger.info(f"  ✅ {interval}: {len(interval_strategies)}개 레짐별 글로벌 전략 생성")
            
            total = sum(len(s) for s in global_strategies.values())
            logger.info(f"✅ 글로벌 전략 지도 조립 완료: {total}개")
            
            return global_strategies
            
        except Exception as e:
            logger.error(f"❌ 글로벌 전략 조립 실패: {e}")
            return {}
    
    # ==================== 5단계: 빠른 샌티백테스트 ====================
    def quick_sanity_backtest(
        self,
        globals_by_interval: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        빠른 샌티백테스트
        
        Args:
            globals_by_interval: 글로벌 전략들
            
        Returns:
            검증된 전략들
        """
        try:
            logger.info("📊 빠른 샌티백테스트 시작")
            
            validated_strategies = {}
            
            for interval, strategies in globals_by_interval.items():
                validated = []
                
                for strategy in strategies:
                    # 간단한 검증: 파라미터 범위 체크
                    params = strategy.get('params', {})
                    
                    # 기본 검증 (RSI)
                    rsi_min = params.get('rsi_min', 30)
                    rsi_max = params.get('rsi_max', 70)
                    
                    if rsi_min >= rsi_max:
                        logger.warning(f"⚠️ {strategy.get('id')}: RSI 범위 오류 ({rsi_min} >= {rsi_max})")
                        continue
                    
                    # Stop Loss 검증 (20% 미만으로 완화하여 데이터 기반 중간값 수용)
                    sl_pct = params.get('stop_loss_pct', 0)
                    if sl_pct >= 1.0: # 1.0 이상이면 백분율(%)로 간주
                        sl_val = sl_pct / 100.0
                    else:
                        sl_val = sl_pct
                        
                    if not (0 < sl_val < 0.2): # 10% -> 20%로 완화
                        logger.warning(f"⚠️ {strategy.get('id')}: Stop Loss 범위 오류 (값: {sl_pct})")
                        continue
                    
                    # 검증 통과
                    validated.append(strategy)
                
                validated_strategies[interval] = validated
                logger.info(f"  ✅ {interval}: {len(validated)}개 검증 통과")
            
            logger.info("✅ 샌티백테스트 완료")
            return validated_strategies
            
        except Exception as e:
            logger.error(f"❌ 샌티백테스트 실패: {e}")
            return globals_by_interval
    
    # ==================== 6단계: 폴백 적용 ====================
    def apply_fallbacks(
        self,
        tested: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        폴백 전략 적용 (최소 2개 보장)
        
        Args:
            tested: 검증된 전략들
            
        Returns:
            폴백이 적용된 전략들
        """
        try:
            logger.info("📊 폴백 전략 적용 시작")
            
            final_strategies = {}
            
            for interval in self.intervals:
                strategies = tested.get(interval, [])
                
                # 각 템플릿으로 폴백 생성
                fallbacks = []
                for template_name, template_params in FALLBACK_TEMPLATES.items():
                    dna_hash = self._make_dna_hash(template_params)
                    
                    fallback = {
                        'id': f"GLOBAL_{interval}_FALLBACK_{template_name}_{dna_hash[:8]}",
                        'market_type': 'COIN',
                        'market': 'BITHUMB',
                        'symbol': 'GLOBAL', # symbol로 통일
                        'interval': interval,
                        'strategy_type': 'fallback',
                        'params': template_params,
                        'name': f'Fallback: {template_name}',
                        'description': f'Fallback strategy for {interval}',
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'dna_hash': dna_hash,
                        'source_type': 'fallback',
                        
                        # 성능 (폴백은 보수적)
                        'profit': 0.0,
                        'profit_factor': 1.0,
                        'win_rate': 0.5,
                        'trades_count': 0,
                        'quality_grade': 'C',
                        'market_condition': template_params.get('market_condition', 'neutral'),
                    }
                    fallbacks.append(fallback)
                
                # 기존 전략 + 폴백 통합
                all_strategies = strategies + fallbacks
                final_strategies[interval] = all_strategies
                
                logger.info(f"  ✅ {interval}: {len(strategies)}개 + {len(fallbacks)}개 폴백 = {len(all_strategies)}개")
            
            logger.info("✅ 폴백 적용 완료")
            return final_strategies
            
        except Exception as e:
            logger.error(f"❌ 폴백 적용 실패: {e}")
            return tested
    
    # ==================== 7단계: 저장 ====================
    def save(self, globals_by_interval: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        글로벌 전략 저장 (기존 데이터 초기화 후 저장)
        
        Args:
            globals_by_interval: {interval: [strategy_dict, ...]}
        """
        try:
            logger.info("💾 글로벌 전략 저장 시작 (기존 데이터 초기화)")

            import os
            import shutil
            import tempfile
            
            # Directory Mode 대응: 디렉토리면 common_strategies.db 파일로 경로 변경
            save_path = self.db_path
            if os.path.isdir(save_path) or not save_path.endswith('.db'):
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "common_strategies.db")
            else:
                # 🔥 .db 파일 경로일 때도 부모 디렉토리 존재 확인 및 생성
                parent_dir = os.path.dirname(save_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    logger.info(f"📁 출력 디렉토리 생성: {parent_dir}")
            
            # 🔥 Docker 볼륨 마운트 호환 - 임시 파일에 먼저 저장 후 복사
            temp_db_path = os.path.join(tempfile.gettempdir(), f"global_strategies_temp_{os.getpid()}.db")
            logger.info(f"📝 임시 DB 경로: {temp_db_path}")
            
            # 기존 파일이 있으면 임시 파일로 복사 (테이블 구조 유지)
            if os.path.exists(save_path):
                try:
                    shutil.copy(save_path, temp_db_path)
                    logger.info(f"📋 기존 DB를 임시 파일로 복사 완료")
                except Exception as copy_err:
                    logger.warning(f"⚠️ 기존 DB 복사 실패, 새로 생성: {copy_err}")
            
            # 🔥 Docker 볼륨 마운트 호환 - 임시 파일에 직접 연결
            with sqlite3.connect(temp_db_path, timeout=120, isolation_level=None) as conn:
                cursor = conn.cursor()
                
                # Docker 환경 호환을 위한 PRAGMA 설정
                cursor.execute("PRAGMA journal_mode=DELETE")  # WAL 대신 DELETE 모드 (Docker 호환)
                cursor.execute("PRAGMA mmap_size=0")  # mmap 비활성화 (Docker 볼륨 호환)
                cursor.execute("PRAGMA busy_timeout=120000")
                
                # 🔥 테이블 직접 생성 (연결 풀 우회)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS global_strategies (
                        id TEXT PRIMARY KEY,
                        market_type TEXT NOT NULL DEFAULT 'COIN',
                        market TEXT NOT NULL DEFAULT 'BITHUMB',
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        strategy_type TEXT,
                        params TEXT,
                        name TEXT,
                        description TEXT,
                        dna_hash TEXT,
                        source_type TEXT,
                        profit REAL DEFAULT 0.0,
                        profit_factor REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.5,
                        trades_count INTEGER DEFAULT 0,
                        quality_grade TEXT DEFAULT 'A',
                        market_condition TEXT DEFAULT 'neutral',
                        regime TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        meta TEXT
                    )
                """)
                logger.info("✅ global_strategies 테이블 준비 완료")
                
                # 🔥 [사용자 요청] 기존 글로벌 전략 모두 삭제 (새로운 지도로 대체)
                cursor.execute("DELETE FROM global_strategies")
                logger.info("  🗑️ 기존 글로벌 전략 삭제 완료")
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_strategies_interval ON global_strategies(interval)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_strategies_regime ON global_strategies(regime)")
                
                total_saved = 0
                for interval, strategies in globals_by_interval.items():
                    for strategy in strategies:
                        try:
                            # market_type, market, symbol 컬럼 대응
                            cursor.execute("""
                                INSERT OR REPLACE INTO global_strategies
                                (id, market_type, market, symbol, interval, strategy_type, params, name, description,
                                 dna_hash, source_type, profit, profit_factor, win_rate, trades_count,
                                 quality_grade, market_condition, regime, created_at, updated_at, meta)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                strategy.get('id'),
                                strategy.get('market_type', 'COIN'),
                                strategy.get('market', 'BITHUMB'),
                                strategy.get('symbol', 'GLOBAL'),
                                strategy.get('interval'),
                                strategy.get('strategy_type'),
                                json.dumps(strategy.get('params', {})),
                                strategy.get('name'),
                                strategy.get('description'),
                                strategy.get('dna_hash'),
                                strategy.get('source_type'),
                                strategy.get('profit', 0.0),
                                strategy.get('profit_factor', 0.0),
                                strategy.get('win_rate', 0.5),
                                strategy.get('trades_count', 0),
                                strategy.get('quality_grade', 'A'),
                                strategy.get('market_condition', 'neutral'),
                                strategy.get('regime', 'neutral'),
                                strategy.get('created_at'),
                                strategy.get('updated_at'),
                                json.dumps(strategy.get('_meta', {}))
                            ))
                            total_saved += 1
                        except Exception as e:
                            logger.warning(f"⚠️ 전략 저장 실패: {strategy.get('id')} - {e}")
                
                conn.commit()
                logger.info(f"✅ 임시 DB에 글로벌 전략 저장 완료: {total_saved}개")
            
            # 🔥 임시 파일을 원래 위치로 복사 (Docker 볼륨 마운트 우회)
            try:
                shutil.copy(temp_db_path, save_path)
                logger.info(f"✅ 최종 DB로 복사 완료: {save_path}")
            except Exception as copy_err:
                logger.error(f"❌ 최종 DB 복사 실패: {copy_err}")
                raise
            finally:
                # 임시 파일 삭제
                try:
                    os.remove(temp_db_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ 글로벌 전략 저장 실패: {e}")
            raise
    
    # ==================== 헬퍼 메서드 ====================
    @staticmethod
    def score_global(pf: float, tr: float, trades: int, grade: str) -> float:
        """글로벌 전략 스코어 계산"""
        grade_bonus = {'S': 0.15, 'A': 0.1, 'B': 0.05}.get(grade or '', 0.0)
        pf_norm = max(0.0, min(1.0, (pf - 1.0) / 2.0))
        tr_norm = max(0.0, min(1.0, tr / 2.0))
        return 0.5 * pf_norm + 0.3 * tr_norm + 0.2 * math.log(max(trades, 1), 10) + grade_bonus
    
    @staticmethod
    def _make_dna_hash(params: Dict[str, Any]) -> str:
        """파라미터 sequential기반 해시 생성"""
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
    
    def _calculate_median_params(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """중간값 파라미터 계산 (이상치에 강한 보편적 정답 도출)"""
        if not strategies:
            return FALLBACK_TEMPLATES['balanced'].copy()
        
        import numpy as np
        
        # 합성 대상 파라미터 리스트
        key_params = [
            'rsi_min', 'rsi_max', 'stop_loss_pct', 'take_profit_pct',
            'volume_ratio_min', 'volume_ratio_max', 'atr_min', 'atr_max',
            'macd_buy_threshold', 'macd_sell_threshold', 'mfi_min', 'mfi_max'
        ]
        
        median_params = {}
        for param in key_params:
            # None이 아닌 값들만 추출
            values = [s.get(param) for s in strategies if s.get(param) is not None]
            
            if values:
                # numpy.median을 사용하여 중간값 산출
                median_val = np.median(values)
                
                # 타입 변환 (JSON 저장을 위해 float로)
                if isinstance(median_val, (np.float32, np.float64)):
                    median_val = float(median_val)
                elif isinstance(median_val, (np.int32, np.int64)):
                    median_val = int(median_val)
                
                median_params[param] = median_val
            else:
                # 데이터가 없으면 'balanced' 템플릿에서 기본값 차용
                median_params[param] = FALLBACK_TEMPLATES['balanced'].get(param, 0.0)
        
        # 정성적 파라미터 (최빈값 사용)
        from collections import Counter
        conditions = [s.get('market_condition') for s in strategies if s.get('market_condition')]
        if conditions:
            median_params['market_condition'] = Counter(conditions).most_common(1)[0][0]
        else:
            median_params['market_condition'] = 'neutral'
            
        return median_params

# ==================== 팩토리 함수 ====================
def create_global_synthesizer(db_path: str, intervals: Optional[List[str]] = None, seed: int = 42) -> GlobalStrategySynthesizer:
    """GlobalStrategySynthesizer 인스턴스 생성"""
    if intervals is None:
        intervals = ['15m', '30m', '240m', '1d']
    
    return GlobalStrategySynthesizer(db_path, intervals, seed)

