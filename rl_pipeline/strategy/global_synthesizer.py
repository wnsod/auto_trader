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
        전략 파라미터 표준화 (IQR 방식)
        
        Args:
            pool: {interval: [strategy_dict, ...]}
            
        Returns:
            표준화된 pool
        """
        try:
            logger.info("📊 전략 파라미터 표준화 시작")
            
            standardized_pool = {}
            
            for interval, strategies in pool.items():
                if not strategies:
                    continue
                
                # 키 파라미터 추출
                key_params = ['rsi_min', 'rsi_max', 'stop_loss_pct', 'take_profit_pct', 
                            'volume_ratio_min', 'volume_ratio_max']
                
                standardized_strategies = []
                
                for strategy in strategies:
                    std_strategy = strategy.copy()
                    
                    # 표준화된 값 저장
                    std_strategy['_standardized'] = {}
                    
                    for param in key_params:
                        values = [s.get(param, 0) for s in strategies if s.get(param) is not None]
                        if not values:
                            continue
                        
                        # IQR 방식 표준화
                        q1 = sorted(values)[len(values) // 4]
                        q3 = sorted(values)[len(values) * 3 // 4]
                        iqr = q3 - q1 if q3 > q1 else 1.0
                        
                        value = strategy.get(param, 0)
                        if iqr > 0:
                            z_score = (value - (q1 + q3) / 2) / iqr
                            std_strategy['_standardized'][param] = z_score
                    
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
        공통 패턴 추출 (quantile bins)
        
        Args:
            std_pool: 표준화된 pool
            
        Returns:
            {interval: [pattern_spec, ...]} 형태
        """
        try:
            logger.info("📊 공통 패턴 추출 시작")
            
            patterns = {}
            
            for interval, strategies in std_pool.items():
                if not strategies:
                    continue
                
                # quantile bins으로 그룹화
                bins = self._create_quantile_bins(strategies)
                
                pattern_specs = []
                for bin_id, bin_strategies in bins.items():
                    if not bin_strategies:
                        continue
                    
                    # bin 내 평균 계산
                    avg_params = self._calculate_avg_params(bin_strategies)
                    
                    # 지원도 계산
                    support = len(bin_strategies) / len(strategies) if strategies else 0
                    
                    # 평균 성능
                    pf_avg = sum(s.get('profit_factor', 0) for s in bin_strategies) / len(bin_strategies)
                    tr_avg = sum(s.get('profit', 0) for s in bin_strategies) / len(bin_strategies)
                    
                    pattern_spec = {
                        'params': avg_params,
                        'support': support,
                        'pf_avg': pf_avg,
                        'tr_avg': tr_avg,
                        'count': len(bin_strategies)
                    }
                    
                    pattern_specs.append(pattern_spec)
                
                # support가 높은 순으로 정렬
                pattern_specs.sort(key=lambda x: x['support'], reverse=True)
                patterns[interval] = pattern_specs
                
                logger.info(f"  ✅ {interval}: {len(pattern_specs)}개 패턴 추출")
            
            logger.info("✅ 공통 패턴 추출 완료")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ 패턴 추출 실패: {e}")
            return {}
    
    # ==================== 4단계: 전역 전략화 ====================
    def assemble_global_strategies(
        self,
        patterns: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        패턴으로부터 글로벌 전략 생성
        
        Args:
            patterns: {interval: [pattern_spec, ...]}
            
        Returns:
            {interval: [global_strategy_dict, ...]}
        """
        try:
            logger.info("📊 글로벌 전략 조립 시작")
            
            global_strategies = {}
            
            for interval, pattern_specs in patterns.items():
                interval_strategies = []
                
                # 상위 3개 패턴만 사용
                for i, pattern in enumerate(pattern_specs[:3]):
                    params = pattern['params']
                    
                    # dna_hash 생성
                    dna_hash = self._make_dna_hash(params)
                    
                    # 글로벌 전략 생성
                    global_strategy = {
                        'id': f"GLOBAL_{interval}_{dna_hash[:8]}",
                        'coin': 'GLOBAL',
                        'interval': interval,
                        'strategy_type': 'meta_synthesized',
                        'params': params,
                        'name': f'Global Meta Strategy {i+1}',
                        'description': f'Synthesized from {len(pattern_specs)} patterns',
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'dna_hash': dna_hash,
                        
                        # 성능 메타
                        'profit': pattern.get('tr_avg', 0.0),
                        'profit_factor': pattern.get('pf_avg', 0.0),
                        'win_rate': 0.5,  # 기본값
                        'trades_count': 0,  # 기본값
                        'quality_grade': 'A',  # 글로벌 전략 기본 등급
                        'market_condition': params.get('market_condition', 'neutral'),
                        
                        # 메타 정보
                        '_meta': {
                            'support': pattern.get('support', 0.0),
                            'pattern_count': pattern.get('count', 0),
                            'source': 'synthesized'
                        }
                    }
                    
                    interval_strategies.append(global_strategy)
                
                global_strategies[interval] = interval_strategies
                logger.info(f"  ✅ {interval}: {len(interval_strategies)} contraStrategies created")
            
            total = sum(len(s) for s in global_strategies.values())
            logger.info(f"✅ 글로벌 전략 조립 완료: {total}개")
            
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
                    
                    # 기본 검증
                    rsi_min = params.get('rsi_min', 30)
                    rsi_max = params.get('rsi_max', 70)
                    
                    if rsi_min >= rsi_max:
                        logger.warning(f"⚠️ {strategy.get('id')}: RSI 범위 오류 ({'{rsi_min}'} >= {rsi_max})")
                        continue
                    
                    if not (0 < params.get('stop_loss_pct', 0) < 0.1):
                        logger.warning(f"⚠️ {strategy.get('id')}: Stop Loss 범위 오류")
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
                        'coin': 'GLOBAL',
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
        글로벌 전략 저장
        
        Args:
            globals_by_interval: {interval: [strategy_dict, ...]}
        """
        try:
            logger.info("💾 글로벌 전략 저장 시작")

            # 🔥 테이블 존재 보장 (엔진화 대응)
            try:
                from rl_pipeline.db.schema import create_global_strategies_table
                create_global_strategies_table()
            except Exception as e:
                logger.warning(f"⚠️ 테이블 생성 시도 중 오류 (무시 가능): {e}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # global_strategies 테이블 사용 (기존 스키마와 호환)
                # 테이블은 이미 db.schema.py에서 생성됨
                
                # 인덱스 생성 (기존 테이블에)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_global_strategies_interval
                    ON global_strategies(interval)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_global_strategies_type
                    ON global_strategies(strategy_type)
                """)
                
                conn.commit()
                
                total_saved = 0
                for interval, strategies in globals_by_interval.items():
                    for strategy in strategies:
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO global_strategies
                                (id, coin, interval, strategy_type, params, name, description,
                                 dna_hash, source_type, profit, profit_factor, win_rate, trades_count,
                                 quality_grade, market_condition, created_at, updated_at, meta)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                strategy.get('id'),
                                strategy.get('coin'),
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
                                strategy.get('quality_grade', 'C'),
                                strategy.get('market_condition', 'neutral'),
                                strategy.get('created_at'),
                                strategy.get('updated_at'),
                                json.dumps(strategy.get('_meta', {}))
                            ))
                            total_saved += 1
                        except Exception as e:
                            logger.warning(f"⚠️ 전략 저장 실패: {strategy.get('id')} - {e}")
                
                conn.commit()
                logger.info(f"✅ 글로벌 전략 저장 완료: {total_saved}개")
                
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
    
    def _create_quantile_bins(self, strategies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """quantile bins 생성"""
        if not strategies:
            return {}
        
        bins = defaultdict(list)
        
        for strategy in strategies:
            # 간단한 binning: RSI 범위 기준
            rsi_min = strategy.get('rsi_min', 30)
            
            if rsi_min < 30:
                bin_id = 'low_rsi'
            elif rsi_min < 50:
                bin_id = 'medium_rsi'
            else:
                bin_id = 'high_rsi'
            
            bins[bin_id].append(strategy)
        
        return dict(bins)
    
    def _calculate_avg_params(self, strategies: List[Dict[str, Any]]) -> Dict[str, float]:
        """평균 파라미터 계산"""
        if not strategies:
            return FALLBACK_TEMPLATES['balanced'].copy()
        
        key_params = ['rsi_min', 'rsi_max', 'stop_loss_pct', 'take_profit_pct',
                     'volume_ratio_min', 'volume_ratio_max']
        
        avg_params = {}
        for param in key_params:
            values = [s.get(param) for s in strategies if s.get(param) is not None]
            if values:
                avg_params[param] = sum(values) / len(values)
            else:
                # 기본값
                avg_params[param] = FALLBACK_TEMPLATES['balanced'].get(param, 0.0)
        
        return avg_params

# ==================== 팩토리 함수 ====================
def create_global_synthesizer(db_path: str, intervals: Optional[List[str]] = None, seed: int = 42) -> GlobalStrategySynthesizer:
    """GlobalStrategySynthesizer 인스턴스 생성"""
    if intervals is None:
        intervals = ['15m', '30m', '240m', '1d']
    
    return GlobalStrategySynthesizer(db_path, intervals, seed)

