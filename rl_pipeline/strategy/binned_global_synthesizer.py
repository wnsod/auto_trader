"""
세밀한 구간화 기반 글로벌 전략 Synthesizer
개별 코인 전략들의 시그널 조건별 예측값 중간값을 저장하여 모든 범위 커버리지 확보
"""

import json
import logging
import sqlite3
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==================== 구간화 설정 ====================
@dataclass
class BinConfig:
    """구간화 설정"""
    min_val: float
    max_val: float
    step: float
    
    def to_bin(self, value: float) -> Optional[int]:
        """값을 bin 인덱스로 변환"""
        if value is None:
            return None
        # 범위 클램핑
        clamped = max(self.min_val, min(self.max_val, value))
        return int((clamped - self.min_val) / self.step)
    
    def from_bin(self, bin_idx: int) -> float:
        """bin 인덱스를 대표값(중간)으로 변환"""
        return self.min_val + (bin_idx + 0.5) * self.step
    
    @property
    def num_bins(self) -> int:
        """총 bin 개수"""
        return int((self.max_val - self.min_val) / self.step)


# 세밀한 구간화 설정 정의
BIN_CONFIGS = {
    # RSI (1단위, 0-100)
    'rsi_min': BinConfig(0.0, 100.0, 1.0),
    'rsi_max': BinConfig(0.0, 100.0, 1.0),
    
    # MFI (1단위, 0-100)
    'mfi_min': BinConfig(0.0, 100.0, 1.0),
    'mfi_max': BinConfig(0.0, 100.0, 1.0),
    
    # ADX (2단위, 0-100)
    'adx_min': BinConfig(0.0, 100.0, 2.0),
    
    # Volume Ratio (0.1단위, 0.1-5.0)
    'volume_ratio_min': BinConfig(0.1, 5.0, 0.1),
    'volume_ratio_max': BinConfig(0.1, 10.0, 0.1),
    
    # MACD (0.0005단위, -0.02~0.02)
    'macd_buy_threshold': BinConfig(-0.02, 0.02, 0.0005),
    'macd_sell_threshold': BinConfig(-0.02, 0.02, 0.0005),
    
    # ATR Range (0.002단위, 0-0.15)
    'atr_min': BinConfig(0.0, 0.15, 0.002),
    'atr_max': BinConfig(0.0, 0.15, 0.002),
    
    # Stop Loss % (0.5%단위, 0-25%)
    'stop_loss_pct': BinConfig(0.0, 0.25, 0.005),
    
    # Take Profit % (1%단위, 0-60%)
    'take_profit_pct': BinConfig(0.0, 0.60, 0.01),
    
    # Bollinger Band Std (0.1단위, 1.0-4.0)
    'bb_std': BinConfig(1.0, 4.0, 0.1),
}


class BinnedGlobalStrategySynthesizer:
    """
    세밀한 구간화 기반 글로벌 전략 Synthesizer
    
    - 모든 시그널 조건을 세밀하게 bin 처리
    - 동일한 bin 조합에 속하는 전략들의 예측값 중간값 저장
    - 이를 통해 모든 범위에 대한 커버리지 확보
    """
    
    def __init__(self, source_db_path: str, output_db_path: str, intervals: List[str], seed: int = 42):
        """
        Args:
            source_db_path: 개별 코인 전략 DB 디렉토리 경로
            output_db_path: 글로벌 전략 저장 DB 경로
            intervals: 대상 인터벌 리스트
            seed: 랜덤 시드
        """
        self.source_db_path = source_db_path
        self.output_db_path = output_db_path
        self.intervals = intervals
        self.seed = seed
        
        # 재현성 보장
        np.random.seed(seed)
        
        logger.info(f"🚀 BinnedGlobalStrategySynthesizer 초기화")
        logger.info(f"  📂 소스: {source_db_path}")
        logger.info(f"  💾 출력: {output_db_path}")
        logger.info(f"  📊 인터벌: {intervals}")
    
    # ==================== 1단계: 전략 수집 ====================
    def load_all_strategies(
        self, 
        min_trades: int = 5,
        max_dd: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        모든 개별 코인 전략 수집
        
        Args:
            min_trades: 최소 거래 횟수
            max_dd: 최대 낙폭 임계값
            
        Returns:
            전략 딕셔너리 리스트
        """
        logger.info(f"📊 전략 수집 시작 (min_trades={min_trades}, max_dd={max_dd})")
        
        all_strategies = []
        
        # 디렉토리 모드 확인
        if os.path.isdir(self.source_db_path):
            import glob
            db_files = glob.glob(os.path.join(self.source_db_path, "*_strategies.db"))
        else:
            db_files = [self.source_db_path] if os.path.exists(self.source_db_path) else []
        
        logger.info(f"  🔍 {len(db_files)}개 DB 파일 발견")
        
        for db_file in db_files:
            try:
                with sqlite3.connect(db_file) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # strategies 테이블에서 전략 조회
                    query = """
                        SELECT * FROM strategies
                        WHERE trades_count >= ? AND max_drawdown <= ?
                    """
                    cursor.execute(query, (min_trades, max_dd))
                    
                    for row in cursor.fetchall():
                        strategy = dict(row)
                        
                        # params가 JSON 문자열인 경우 파싱
                        if 'params' in strategy and isinstance(strategy['params'], str):
                            try:
                                strategy['params'] = json.loads(strategy['params'])
                            except:
                                strategy['params'] = {}
                        
                        all_strategies.append(strategy)
                        
            except Exception as e:
                # logger.debug(f"  ⚠️ DB 로드 실패 ({os.path.basename(db_file)}): {e}")
                pass
        
        logger.info(f"✅ 전략 수집 완료: {len(all_strategies)}개")
        return all_strategies
    
    # ==================== 2단계: 구간화 ====================
    def _bin_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        전략을 구간화된 키로 변환
        
        Args:
            strategy: 전략 딕셔너리
            
        Returns:
            구간화된 전략 (bin 인덱스 포함)
        """
        # params에서 값 추출 (params가 딕셔너리인 경우와 플랫한 경우 모두 처리)
        params = strategy.get('params', {})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                params = {}
        
        # 플랫 구조 우선, 없으면 params에서 찾기
        def get_value(key: str) -> Optional[float]:
            val = strategy.get(key)
            if val is None and isinstance(params, dict):
                val = params.get(key)
            return val
        
        binned = {
            'interval': strategy.get('interval', '15m'),
            'regime': (strategy.get('regime') or strategy.get('market_condition') or 'neutral').lower(),
            'quality_grade': strategy.get('quality_grade', 'B'),
        }
        
        # 각 파라미터 구간화
        for param_name, bin_config in BIN_CONFIGS.items():
            value = get_value(param_name)
            binned[f'{param_name}_bin'] = bin_config.to_bin(value) if value is not None else None
        
        # 예측값 저장
        binned['profit'] = float(strategy.get('profit', 0.0) or 0.0)
        binned['win_rate'] = float(strategy.get('win_rate', 0.5) or 0.5)
        binned['profit_factor'] = float(strategy.get('profit_factor', 1.0) or 1.0)
        binned['sharpe_ratio'] = float(strategy.get('sharpe_ratio', 0.0) or 0.0)
        binned['max_drawdown'] = float(strategy.get('max_drawdown', 0.0) or 0.0)
        binned['trades_count'] = int(strategy.get('trades_count', 0) or 0)
        
        return binned
    
    def _make_bin_key(self, binned: Dict[str, Any]) -> Tuple:
        """구간화된 전략의 고유 키 생성"""
        return (
            binned['interval'],
            binned['regime'],
            binned['quality_grade'],
            binned.get('rsi_min_bin'),
            binned.get('rsi_max_bin'),
            binned.get('mfi_min_bin'),
            binned.get('mfi_max_bin'),
            binned.get('adx_min_bin'),
            binned.get('volume_ratio_min_bin'),
            binned.get('volume_ratio_max_bin'),
            binned.get('macd_buy_threshold_bin'),
            binned.get('macd_sell_threshold_bin'),
            binned.get('atr_min_bin'),
            binned.get('atr_max_bin'),
            binned.get('stop_loss_pct_bin'),
            binned.get('take_profit_pct_bin'),
            binned.get('bb_std_bin'),
        )
    
    # ==================== 3단계: 집계 (중간값 계산) ====================
    def aggregate_predictions(
        self, 
        strategies: List[Dict[str, Any]],
        min_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """
        동일한 bin 조합의 전략들의 예측값 중간값 계산
        
        Args:
            strategies: 전략 리스트
            min_samples: 최소 샘플 수 (이보다 적으면 제외)
            
        Returns:
            집계된 예측값 리스트
        """
        logger.info(f"📊 예측값 집계 시작 (min_samples={min_samples})")
        
        # 1. 모든 전략 구간화
        binned_strategies = [self._bin_strategy(s) for s in strategies]
        
        # 2. bin 키별로 그룹화
        bin_groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        for bs in binned_strategies:
            key = self._make_bin_key(bs)
            bin_groups[key].append(bs)
        
        logger.info(f"  🔢 고유 bin 조합: {len(bin_groups)}개")
        
        # 3. 각 그룹별 중간값 계산
        aggregated = []
        skipped_low_sample = 0
        
        for bin_key, group in bin_groups.items():
            if len(group) < min_samples:
                skipped_low_sample += 1
                continue
            
            # 예측값들 추출
            profits = [g['profit'] for g in group]
            win_rates = [g['win_rate'] for g in group]
            profit_factors = [g['profit_factor'] for g in group]
            sharpes = [g['sharpe_ratio'] for g in group]
            drawdowns = [g['max_drawdown'] for g in group]
            trades = [g['trades_count'] for g in group]
            
            # 중간값 계산
            sample = group[0]  # bin 키 정보용
            result = {
                'interval': sample['interval'],
                'regime': sample['regime'],
                'quality_grade': sample['quality_grade'],
                
                # bin 인덱스
                'rsi_min_bin': sample.get('rsi_min_bin'),
                'rsi_max_bin': sample.get('rsi_max_bin'),
                'mfi_min_bin': sample.get('mfi_min_bin'),
                'mfi_max_bin': sample.get('mfi_max_bin'),
                'adx_min_bin': sample.get('adx_min_bin'),
                'volume_ratio_min_bin': sample.get('volume_ratio_min_bin'),
                'volume_ratio_max_bin': sample.get('volume_ratio_max_bin'),
                'macd_buy_bin': sample.get('macd_buy_threshold_bin'),
                'macd_sell_bin': sample.get('macd_sell_threshold_bin'),
                'atr_min_bin': sample.get('atr_min_bin'),
                'atr_max_bin': sample.get('atr_max_bin'),
                'stop_loss_bin': sample.get('stop_loss_pct_bin'),
                'take_profit_bin': sample.get('take_profit_pct_bin'),
                'bb_std_bin': sample.get('bb_std_bin'),
                
                # 중간값 예측 결과
                'median_profit': float(np.median(profits)),
                'median_win_rate': float(np.median(win_rates)),
                'median_profit_factor': float(np.median(profit_factors)),
                'median_sharpe': float(np.median(sharpes)),
                'median_max_drawdown': float(np.median(drawdowns)),
                'median_trades': float(np.median(trades)),
                
                # 통계
                'sample_count': len(group),
                'std_profit': float(np.std(profits)) if len(profits) > 1 else 0.0,
                'confidence_score': self._calculate_confidence(len(group), profits),
                
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            
            aggregated.append(result)
        
        logger.info(f"  ⚠️ 샘플 부족으로 제외: {skipped_low_sample}개")
        logger.info(f"✅ 집계 완료: {len(aggregated)}개 예측 조합")
        
        return aggregated
    
    def _calculate_confidence(self, sample_count: int, profits: List[float]) -> float:
        """신뢰도 점수 계산 (샘플 수 + 분산 기반)"""
        if sample_count < 2:
            return 0.0
        
        # 샘플 수 기반 (로그 스케일)
        count_score = min(1.0, np.log(sample_count + 1) / np.log(100))
        
        # 분산 기반 (낮을수록 좋음)
        std = np.std(profits) if len(profits) > 1 else 1.0
        variance_score = max(0.0, 1.0 - std / 2.0)
        
        return 0.6 * count_score + 0.4 * variance_score
    
    # ==================== 4단계: 저장 ====================
    def save_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """
        예측값 저장
        
        Args:
            predictions: 집계된 예측값 리스트
            
        Returns:
            저장된 레코드 수
        """
        logger.info(f"💾 글로벌 전략 예측값 저장 시작")
        
        # 출력 DB 경로 처리
        output_path = self.output_db_path
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "common_strategies.db")
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with sqlite3.connect(output_path) as conn:
            cursor = conn.cursor()
            
            # 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_strategy_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interval TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    quality_grade TEXT,
                    
                    -- 구간화된 시그널 조건 (bin 인덱스)
                    rsi_min_bin INTEGER,
                    rsi_max_bin INTEGER,
                    mfi_min_bin INTEGER,
                    mfi_max_bin INTEGER,
                    adx_min_bin INTEGER,
                    volume_ratio_min_bin INTEGER,
                    volume_ratio_max_bin INTEGER,
                    macd_buy_bin INTEGER,
                    macd_sell_bin INTEGER,
                    atr_min_bin INTEGER,
                    atr_max_bin INTEGER,
                    stop_loss_bin INTEGER,
                    take_profit_bin INTEGER,
                    bb_std_bin INTEGER,
                    
                    -- 중간값 예측 결과
                    median_profit REAL,
                    median_win_rate REAL,
                    median_profit_factor REAL,
                    median_sharpe REAL,
                    median_max_drawdown REAL,
                    median_trades REAL,
                    
                    -- 통계
                    sample_count INTEGER,
                    std_profit REAL,
                    confidence_score REAL,
                    
                    -- 메타
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # 인덱스 생성 (자주 조회하는 컬럼에 대해)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gsp_interval ON global_strategy_predictions(interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gsp_regime ON global_strategy_predictions(regime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gsp_rsi ON global_strategy_predictions(rsi_min_bin, rsi_max_bin)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gsp_confidence ON global_strategy_predictions(confidence_score)")
            
            # 기존 데이터 삭제 (완전 교체)
            cursor.execute("DELETE FROM global_strategy_predictions")
            logger.info("  🗑️ 기존 예측 데이터 삭제 완료")
            
            # 배치 삽입
            insert_sql = """
                INSERT INTO global_strategy_predictions (
                    interval, regime, quality_grade,
                    rsi_min_bin, rsi_max_bin, mfi_min_bin, mfi_max_bin, adx_min_bin,
                    volume_ratio_min_bin, volume_ratio_max_bin,
                    macd_buy_bin, macd_sell_bin,
                    atr_min_bin, atr_max_bin,
                    stop_loss_bin, take_profit_bin, bb_std_bin,
                    median_profit, median_win_rate, median_profit_factor, median_sharpe, median_max_drawdown, median_trades,
                    sample_count, std_profit, confidence_score,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            saved_count = 0
            for pred in predictions:
                try:
                    cursor.execute(insert_sql, (
                        pred['interval'],
                        pred['regime'],
                        pred['quality_grade'],
                        pred.get('rsi_min_bin'),
                        pred.get('rsi_max_bin'),
                        pred.get('mfi_min_bin'),
                        pred.get('mfi_max_bin'),
                        pred.get('adx_min_bin'),
                        pred.get('volume_ratio_min_bin'),
                        pred.get('volume_ratio_max_bin'),
                        pred.get('macd_buy_bin'),
                        pred.get('macd_sell_bin'),
                        pred.get('atr_min_bin'),
                        pred.get('atr_max_bin'),
                        pred.get('stop_loss_bin'),
                        pred.get('take_profit_bin'),
                        pred.get('bb_std_bin'),
                        pred['median_profit'],
                        pred['median_win_rate'],
                        pred['median_profit_factor'],
                        pred['median_sharpe'],
                        pred['median_max_drawdown'],
                        pred['median_trades'],
                        pred['sample_count'],
                        pred['std_profit'],
                        pred['confidence_score'],
                        pred['created_at'],
                        pred['updated_at'],
                    ))
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ 예측 저장 실패: {e}")
            
            conn.commit()
            
        logger.info(f"✅ 글로벌 전략 예측값 저장 완료: {saved_count}개")
        return saved_count
    
    # ==================== 전체 파이프라인 ====================
    def run_synthesis(
        self, 
        min_trades: int = 5,
        max_dd: float = 0.8,
        min_samples: int = 2
    ) -> Dict[str, Any]:
        """
        전체 합성 파이프라인 실행
        
        Args:
            min_trades: 최소 거래 횟수
            max_dd: 최대 낙폭 임계값
            min_samples: 최소 샘플 수
            
        Returns:
            합성 결과 요약
        """
        logger.info("=" * 60)
        logger.info("🚀 세밀한 구간화 기반 글로벌 전략 합성 시작")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # 1. 전략 수집
        logger.info("📊 1단계: 전략 수집")
        strategies = self.load_all_strategies(min_trades=min_trades, max_dd=max_dd)
        
        if not strategies:
            logger.error("❌ 수집된 전략이 없습니다")
            return {'success': False, 'error': '전략 수집 실패'}
        
        # 2. 예측값 집계
        logger.info("📊 2단계: 예측값 집계 (구간화 + 중간값)")
        predictions = self.aggregate_predictions(strategies, min_samples=min_samples)
        
        if not predictions:
            logger.error("❌ 집계된 예측값이 없습니다")
            return {'success': False, 'error': '예측값 집계 실패'}
        
        # 3. 저장
        logger.info("📊 3단계: 글로벌 전략 예측값 저장")
        saved_count = self.save_predictions(predictions)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # 인터벌별 통계
        interval_stats = defaultdict(int)
        for pred in predictions:
            interval_stats[pred['interval']] += 1
        
        logger.info("=" * 60)
        logger.info("✨ 글로벌 전략 합성 완료")
        logger.info(f"  📊 입력 전략: {len(strategies)}개")
        logger.info(f"  📊 출력 예측: {saved_count}개")
        logger.info(f"  ⏱️ 소요 시간: {elapsed:.1f}초")
        for interval, count in sorted(interval_stats.items()):
            logger.info(f"    ● {interval}: {count}개")
        logger.info("=" * 60)
        
        return {
            'success': True,
            'input_strategies': len(strategies),
            'output_predictions': saved_count,
            'interval_stats': dict(interval_stats),
            'elapsed_seconds': elapsed,
        }


# ==================== 예측값 조회 클래스 ====================
class GlobalPredictionLookup:
    """글로벌 전략 예측값 조회 클래스"""
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: common_strategies.db 경로
        """
        self.db_path = db_path
        self._cache: Dict[Tuple, Dict[str, float]] = {}
        self._loaded = False
    
    def load_cache(self) -> int:
        """캐시 로드 (Docker 볼륨 마운트 호환성 포함)"""
        if not os.path.exists(self.db_path):
            logger.warning(f"⚠️ DB 파일 없음: {self.db_path}")
            return 0
        
        # 🆕 실제 사용할 DB 경로 (Docker 볼륨 마운트 문제 해결)
        effective_db_path = self._get_effective_db_path()
        if not effective_db_path:
            logger.warning(f"⚠️ DB 접근 불가: {self.db_path}")
            return 0
        
        try:
            with sqlite3.connect(effective_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM global_strategy_predictions")
                rows = cursor.fetchall()
                
                for row in rows:
                    key = (
                        row['interval'],
                        row['regime'],
                        row['quality_grade'],
                        row['rsi_min_bin'],
                        row['rsi_max_bin'],
                        row['mfi_min_bin'],
                        row['mfi_max_bin'],
                        row['adx_min_bin'],
                        row['volume_ratio_min_bin'],
                        row['volume_ratio_max_bin'],
                        row['macd_buy_bin'],
                        row['macd_sell_bin'],
                        row['atr_min_bin'],
                        row['atr_max_bin'],
                        row['stop_loss_bin'],
                        row['take_profit_bin'],
                        row['bb_std_bin'],
                    )
                    self._cache[key] = {
                        'median_profit': row['median_profit'],
                        'median_win_rate': row['median_win_rate'],
                        'median_profit_factor': row['median_profit_factor'],
                        'median_sharpe': row['median_sharpe'],
                        'median_max_drawdown': row['median_max_drawdown'],
                        'sample_count': row['sample_count'],
                        'confidence_score': row['confidence_score'],
                    }
                
                self._loaded = True
                logger.info(f"✅ 글로벌 예측 캐시 로드 완료: {len(self._cache)}개")
                return len(self._cache)
                
        except Exception as e:
            logger.warning(f"⚠️ 글로벌 예측 테이블 없음 또는 오류: {e}")
            return 0
    
    def _get_effective_db_path(self) -> Optional[str]:
        """
        Docker 볼륨 마운트 호환성을 위한 효과적인 DB 경로 반환
        
        Windows 호스트에서 Docker 볼륨 마운트된 큰 파일(>500MB)은 
        SQLite로 직접 열 때 불안정할 수 있음.
        이 경우 컨테이너 내부로 복사하여 사용.
        """
        import shutil
        
        # Docker 환경 감지
        is_docker = os.path.exists('/workspace')
        
        # 파일 크기 확인 (500MB 이상인 경우 큰 파일로 간주)
        file_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        is_large_file = file_size_mb > 500
        
        logger.debug(f"🔍 _get_effective_db_path: {self.db_path} (Docker: {is_docker}, Size: {file_size_mb:.0f}MB)")
        
        # Docker 환경 + 큰 파일인 경우 → 무조건 복사 사용
        if is_docker and is_large_file:
            try:
                # 캐시 디렉토리 생성
                cache_dir = '/tmp/global_strategy_cache'
                os.makedirs(cache_dir, exist_ok=True)
                
                # 파일명 추출
                filename = os.path.basename(self.db_path)
                cached_path = os.path.join(cache_dir, filename)
                
                # 파일 수정 시간 비교 (이미 복사된 경우 건너뛰기)
                need_copy = True
                if os.path.exists(cached_path):
                    src_mtime = os.path.getmtime(self.db_path)
                    dst_mtime = os.path.getmtime(cached_path)
                    if dst_mtime >= src_mtime:
                        need_copy = False
                        logger.info(f"📋 캐시된 DB 사용: {cached_path}")
                
                if need_copy:
                    logger.info(f"📋 DB 복사 중 ({file_size_mb:.0f}MB): {self.db_path} -> {cached_path}")
                    print(f"📋 큰 DB 파일 복사 중... ({file_size_mb:.0f}MB, 잠시 기다려주세요)")
                    shutil.copy2(self.db_path, cached_path)
                    logger.info(f"✅ DB 복사 완료")
                    print(f"✅ DB 복사 완료!")
                
                # 복사된 파일 열기 테스트
                test_conn = sqlite3.connect(cached_path, timeout=30)
                test_conn.execute("SELECT 1")
                test_conn.close()
                return cached_path
                
            except Exception as copy_err:
                logger.error(f"❌ DB 복사 실패: {copy_err}")
                # 복사 실패 시 직접 접근 시도 (폴백)
        
        # 직접 열기 시도 (작은 파일 또는 비-Docker 환경)
        try:
            test_conn = sqlite3.connect(self.db_path, timeout=10)
            test_conn.execute("SELECT 1")
            test_conn.close()
            return self.db_path
        except Exception as e:
            logger.warning(f"⚠️ DB 직접 접근 실패: {e}")
            return None
    
    def lookup(
        self,
        interval: str,
        regime: str,
        quality_grade: str,
        rsi_min: float = None,
        rsi_max: float = None,
        mfi_min: float = None,
        mfi_max: float = None,
        adx_min: float = None,
        volume_ratio_min: float = None,
        volume_ratio_max: float = None,
        macd_buy_threshold: float = None,
        macd_sell_threshold: float = None,
        atr_min: float = None,
        atr_max: float = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        bb_std: float = None,
        fallback_regime: bool = True,
    ) -> Optional[Dict[str, float]]:
        """
        시그널 조건에 해당하는 글로벌 예측값 조회
        
        Args:
            interval: 인터벌
            regime: 레짐
            quality_grade: 품질 등급
            rsi_min~bb_std: 시그널 파라미터들
            fallback_regime: 정확한 매칭 실패 시 레짐 무시하고 재검색
            
        Returns:
            예측값 딕셔너리 또는 None
        """
        if not self._loaded:
            self.load_cache()
        
        # bin 변환
        key = (
            interval,
            regime.lower(),
            quality_grade,
            BIN_CONFIGS['rsi_min'].to_bin(rsi_min) if rsi_min is not None else None,
            BIN_CONFIGS['rsi_max'].to_bin(rsi_max) if rsi_max is not None else None,
            BIN_CONFIGS['mfi_min'].to_bin(mfi_min) if mfi_min is not None else None,
            BIN_CONFIGS['mfi_max'].to_bin(mfi_max) if mfi_max is not None else None,
            BIN_CONFIGS['adx_min'].to_bin(adx_min) if adx_min is not None else None,
            BIN_CONFIGS['volume_ratio_min'].to_bin(volume_ratio_min) if volume_ratio_min is not None else None,
            BIN_CONFIGS['volume_ratio_max'].to_bin(volume_ratio_max) if volume_ratio_max is not None else None,
            BIN_CONFIGS['macd_buy_threshold'].to_bin(macd_buy_threshold) if macd_buy_threshold is not None else None,
            BIN_CONFIGS['macd_sell_threshold'].to_bin(macd_sell_threshold) if macd_sell_threshold is not None else None,
            BIN_CONFIGS['atr_min'].to_bin(atr_min) if atr_min is not None else None,
            BIN_CONFIGS['atr_max'].to_bin(atr_max) if atr_max is not None else None,
            BIN_CONFIGS['stop_loss_pct'].to_bin(stop_loss_pct) if stop_loss_pct is not None else None,
            BIN_CONFIGS['take_profit_pct'].to_bin(take_profit_pct) if take_profit_pct is not None else None,
            BIN_CONFIGS['bb_std'].to_bin(bb_std) if bb_std is not None else None,
        )
        
        # 정확한 매칭
        result = self._cache.get(key)
        if result:
            return result
        
        # Fallback: 유사 bin 검색 (RSI만 매칭 시도)
        if fallback_regime:
            # 레짐 무관하게 RSI bin만 매칭
            partial_matches = []
            for cache_key, cache_val in self._cache.items():
                # 인터벌, RSI bin만 매칭
                if (cache_key[0] == interval and 
                    cache_key[3] == key[3] and  # rsi_min_bin
                    cache_key[4] == key[4]):    # rsi_max_bin
                    partial_matches.append(cache_val)
            
            if partial_matches:
                # 가장 신뢰도 높은 것 반환
                return max(partial_matches, key=lambda x: x.get('confidence_score', 0))
        
        return None
    
    def get_prediction_for_strategy(
        self,
        strategy: Dict[str, Any],
        fallback_regime: bool = True
    ) -> Optional[Dict[str, float]]:
        """
        전략 딕셔너리로부터 글로벌 예측값 조회
        
        Args:
            strategy: 전략 딕셔너리
            fallback_regime: 레짐 폴백 사용
            
        Returns:
            예측값 딕셔너리
        """
        params = strategy.get('params', {})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                params = {}
        
        # 플랫 구조 우선
        def get_val(key: str):
            return strategy.get(key) or params.get(key)
        
        return self.lookup(
            interval=strategy.get('interval', '15m'),
            regime=(strategy.get('regime') or strategy.get('market_condition') or 'neutral'),
            quality_grade=strategy.get('quality_grade', 'B'),
            rsi_min=get_val('rsi_min'),
            rsi_max=get_val('rsi_max'),
            mfi_min=get_val('mfi_min'),
            mfi_max=get_val('mfi_max'),
            adx_min=get_val('adx_min'),
            volume_ratio_min=get_val('volume_ratio_min'),
            volume_ratio_max=get_val('volume_ratio_max'),
            macd_buy_threshold=get_val('macd_buy_threshold'),
            macd_sell_threshold=get_val('macd_sell_threshold'),
            atr_min=get_val('atr_min'),
            atr_max=get_val('atr_max'),
            stop_loss_pct=get_val('stop_loss_pct'),
            take_profit_pct=get_val('take_profit_pct'),
            bb_std=get_val('bb_std'),
            fallback_regime=fallback_regime,
        )


# ==================== 팩토리 함수 ====================
def create_binned_global_synthesizer(
    source_db_path: str,
    output_db_path: str = None,
    intervals: List[str] = None,
    seed: int = 42
) -> BinnedGlobalStrategySynthesizer:
    """BinnedGlobalStrategySynthesizer 인스턴스 생성"""
    if intervals is None:
        intervals = ['15m', '30m', '240m', '1d']
    
    if output_db_path is None:
        if os.path.isdir(source_db_path):
            output_db_path = os.path.join(source_db_path, "common_strategies.db")
        else:
            output_db_path = source_db_path
    
    return BinnedGlobalStrategySynthesizer(source_db_path, output_db_path, intervals, seed)


def create_global_prediction_lookup(db_path: str) -> GlobalPredictionLookup:
    """GlobalPredictionLookup 인스턴스 생성"""
    return GlobalPredictionLookup(db_path)
