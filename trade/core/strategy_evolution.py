"""
🧬 전략 진화 시스템 (Strategy Evolution System)

전략 × 레짐 조합별로 개별 진화하는 자동화 시스템

레벨 구조:
- Level 1: 기본 전략 (10가지 하드코딩 전략)
- Level 2: 전환 조합 학습 (A→B 전환 패턴 최적화)
- Level 3: AI 자동 조합 생성 (성공 패턴 분석 → 새 전략 생성)
- Level 4: 유전 알고리즘 진화 (파라미터 최적화 + 교배/돌연변이)

사용처:
- trade/strategy_signal_generator.py: 전략 선택 시 레벨 참조
- trade/virtual_trade_executor.py: 매매 시 진화 레벨 기록
- trade/virtual_trade_learner.py: 학습 후 레벨 업그레이드 체크
"""

import os
import sqlite3
import time
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import IntEnum


# ============================================================================
# 진화 레벨 정의
# ============================================================================
class EvolutionLevel(IntEnum):
    """전략 진화 레벨"""
    BASIC = 1           # 기본 전략 (하드코딩)
    TRANSITION = 2      # 전환 조합 학습
    AI_GENERATED = 3    # AI 자동 조합 생성
    GENETIC = 4         # 유전 알고리즘 진화


# ============================================================================
# 레벨 활성화 조건 정의
# ============================================================================
@dataclass
class LevelThresholds:
    """레벨 활성화 조건"""
    # Level 2 조건
    level2_min_trades: int = 50          # 최소 거래 횟수
    level2_min_confidence: float = 0.6   # 최소 신뢰도
    
    # Level 3 조건
    level3_min_switch_trades: int = 30   # 최소 전환 거래 횟수
    level3_min_switch_patterns: int = 5  # 최소 전환 패턴 수
    
    # Level 4 조건
    level4_min_ai_strategies: int = 20   # 최소 AI 생성 전략 수
    level4_min_ai_win_rate: float = 0.55 # 최소 AI 전략 승률


DEFAULT_THRESHOLDS = LevelThresholds()


# ============================================================================
# 진화 통계 데이터 구조
# ============================================================================
@dataclass
class EvolutionStats:
    """전략×레짐 조합의 진화 통계"""
    strategy: str
    regime: str
    level: int = 1
    total_trades: int = 0
    success_count: int = 0
    confidence: float = 0.0
    switch_trades: int = 0
    switch_patterns: int = 0
    ai_strategies: int = 0
    ai_win_rate: float = 0.0
    avg_profit: float = 0.0
    last_updated: int = 0
    
    @property
    def key(self) -> str:
        return f"{self.strategy}_{self.regime}"
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.success_count / self.total_trades


# ============================================================================
# AI 생성 전략 구조 (Level 3)
# ============================================================================
@dataclass
class AIGeneratedStrategy:
    """AI가 생성한 전략"""
    strategy_id: str                    # 고유 ID (예: "ai_gen_001")
    base_strategy: str                  # 기반 전략 (예: "trend")
    regime: str                         # 타겟 레짐 (예: "bullish")
    conditions: Dict[str, Any]          # 진입 조건
    exit_params: Dict[str, float]       # 청산 파라미터
    performance: Dict[str, float]       # 성과 통계
    created_at: int = 0
    trades_count: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    is_active: bool = True


# ============================================================================
# 유전자 구조 (Level 4)
# ============================================================================
@dataclass
class StrategyGene:
    """유전 알고리즘용 전략 유전자"""
    gene_id: str
    base_strategy: str
    regime: str
    
    # 청산 파라미터 (진화 대상)
    take_profit_pct: float = 10.0
    stop_loss_pct: float = 5.0
    max_holding_hours: int = 72
    trailing_trigger_pct: float = 5.0
    trailing_distance_pct: float = 2.0
    
    # 진입 조건 (진화 대상)
    min_signal_score: float = 0.1
    min_rsi: float = 20.0
    max_rsi: float = 80.0
    min_volume_ratio: float = 1.0
    
    # 성과 (적합도 계산용)
    fitness: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    sharpe_ratio: float = 0.0
    
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)


# ============================================================================
# 메인 진화 관리자 클래스
# ============================================================================
class StrategyEvolutionManager:
    """
    전략 진화 중앙 관리자
    
    각 전략×레짐 조합의 진화 상태를 관리하고,
    조건 충족 시 자동으로 다음 레벨로 진화시킵니다.
    """
    
    def __init__(self, db_path: str = None):
        """초기화"""
        self.db_path = db_path or os.environ.get('STRATEGY_DB_PATH', '')
        self.thresholds = DEFAULT_THRESHOLDS
        
        # 캐시 (성능 최적화)
        self._stats_cache: Dict[str, EvolutionStats] = {}
        self._cache_timestamp: int = 0
        self._cache_ttl: int = 300  # 5분 TTL
        
        # AI 생성 전략 캐시
        self._ai_strategies: Dict[str, AIGeneratedStrategy] = {}
        
        # 유전자 풀
        self._gene_pool: Dict[str, StrategyGene] = {}
        
        # 테이블 초기화
        self._init_tables()
    
    def _init_tables(self):
        """진화 관련 테이블 초기화"""
        if not self.db_path or not os.path.exists(os.path.dirname(self.db_path) or '.'):
            return
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # 진화 통계 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_evolution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        level INTEGER DEFAULT 1,
                        total_trades INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0.0,
                        switch_trades INTEGER DEFAULT 0,
                        switch_patterns INTEGER DEFAULT 0,
                        ai_strategies INTEGER DEFAULT 0,
                        ai_win_rate REAL DEFAULT 0.0,
                        avg_profit REAL DEFAULT 0.0,
                        last_updated INTEGER DEFAULT 0,
                        UNIQUE(strategy, regime)
                    )
                """)
                
                # AI 생성 전략 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ai_generated_strategies (
                        strategy_id TEXT PRIMARY KEY,
                        base_strategy TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        conditions TEXT,
                        exit_params TEXT,
                        performance TEXT,
                        created_at INTEGER,
                        trades_count INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        avg_profit REAL DEFAULT 0.0,
                        is_active INTEGER DEFAULT 1
                    )
                """)
                
                # 유전자 풀 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_genes (
                        gene_id TEXT PRIMARY KEY,
                        base_strategy TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        take_profit_pct REAL,
                        stop_loss_pct REAL,
                        max_holding_hours INTEGER,
                        trailing_trigger_pct REAL,
                        trailing_distance_pct REAL,
                        min_signal_score REAL,
                        min_rsi REAL,
                        max_rsi REAL,
                        min_volume_ratio REAL,
                        fitness REAL DEFAULT 0.0,
                        trades_count INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        avg_profit REAL DEFAULT 0.0,
                        sharpe_ratio REAL DEFAULT 0.0,
                        generation INTEGER DEFAULT 0,
                        parent_ids TEXT,
                        created_at INTEGER,
                        is_active INTEGER DEFAULT 1
                    )
                """)
                
                # 인덱스
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evolution_key ON strategy_evolution(strategy, regime)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_strat_regime ON ai_generated_strategies(base_strategy, regime)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_genes_regime ON strategy_genes(base_strategy, regime)")
                
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ 진화 테이블 초기화 오류: {e}")
    
    # ========================================================================
    # 레벨 조회 및 계산
    # ========================================================================
    def get_evolution_level(self, strategy: str, regime: str) -> int:
        """현재 진화 레벨 조회"""
        stats = self.get_evolution_stats(strategy, regime)
        return self._calculate_level(stats)
    
    def _calculate_level(self, stats: EvolutionStats) -> int:
        """통계 기반 레벨 계산"""
        t = self.thresholds
        
        # Level 4: 유전 진화 조건
        if (stats.ai_strategies >= t.level4_min_ai_strategies and
            stats.ai_win_rate >= t.level4_min_ai_win_rate):
            return EvolutionLevel.GENETIC
        
        # Level 3: AI 자동 생성 조건
        if (stats.switch_trades >= t.level3_min_switch_trades and
            stats.switch_patterns >= t.level3_min_switch_patterns):
            return EvolutionLevel.AI_GENERATED
        
        # Level 2: 전환 학습 조건
        if (stats.total_trades >= t.level2_min_trades and
            stats.confidence >= t.level2_min_confidence):
            return EvolutionLevel.TRANSITION
        
        # Level 1: 기본
        return EvolutionLevel.BASIC
    
    def get_evolution_stats(self, strategy: str, regime: str) -> EvolutionStats:
        """진화 통계 조회 (캐시 활용)"""
        key = f"{strategy}_{regime}"
        current_time = int(time.time())
        
        # 캐시 체크
        if (key in self._stats_cache and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._stats_cache[key]
        
        # DB 조회
        stats = self._load_stats_from_db(strategy, regime)
        
        # 캐시 저장
        self._stats_cache[key] = stats
        self._cache_timestamp = current_time
        
        return stats
    
    def _load_stats_from_db(self, strategy: str, regime: str) -> EvolutionStats:
        """DB에서 통계 로드"""
        stats = EvolutionStats(strategy=strategy, regime=regime)
        
        if not self.db_path or not os.path.exists(self.db_path):
            return stats
        
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute("""
                    SELECT level, total_trades, success_count, confidence,
                           switch_trades, switch_patterns, ai_strategies,
                           ai_win_rate, avg_profit, last_updated
                    FROM strategy_evolution
                    WHERE strategy = ? AND regime = ?
                """, (strategy, regime))
                
                row = cursor.fetchone()
                if row:
                    stats.level = row[0]
                    stats.total_trades = row[1]
                    stats.success_count = row[2]
                    stats.confidence = row[3]
                    stats.switch_trades = row[4]
                    stats.switch_patterns = row[5]
                    stats.ai_strategies = row[6]
                    stats.ai_win_rate = row[7]
                    stats.avg_profit = row[8]
                    stats.last_updated = row[9]
                    
        except Exception as e:
            print(f"⚠️ 진화 통계 로드 오류 ({strategy}_{regime}): {e}")
        
        return stats
    
    # ========================================================================
    # 통계 업데이트
    # ========================================================================
    def update_trade_result(self, strategy: str, regime: str, 
                           success: bool, profit_pct: float,
                           is_switch: bool = False, switch_from: str = None):
        """거래 결과로 통계 업데이트"""
        stats = self.get_evolution_stats(strategy, regime)
        
        # 기본 통계 업데이트
        stats.total_trades += 1
        if success:
            stats.success_count += 1
        
        # 신뢰도 계산 (샘플 수 기반)
        stats.confidence = min(1.0, stats.total_trades / 100.0)
        
        # 평균 수익 업데이트 (이동 평균)
        if stats.total_trades == 1:
            stats.avg_profit = profit_pct
        else:
            stats.avg_profit = (stats.avg_profit * 0.95) + (profit_pct * 0.05)
        
        # 전환 거래 통계
        if is_switch:
            stats.switch_trades += 1
            if switch_from:
                # 전환 패턴 수 카운트 (별도 로직 필요)
                pass
        
        stats.last_updated = int(time.time())
        
        # 레벨 재계산
        new_level = self._calculate_level(stats)
        old_level = stats.level
        stats.level = new_level
        
        # DB 저장
        self._save_stats_to_db(stats)
        
        # 캐시 업데이트
        self._stats_cache[stats.key] = stats
        
        # 레벨 업 시 알림
        if new_level > old_level:
            self._on_level_up(stats, old_level, new_level)
        
        return stats
    
    def _save_stats_to_db(self, stats: EvolutionStats):
        """DB에 통계 저장"""
        if not self.db_path:
            return
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO strategy_evolution
                    (strategy, regime, level, total_trades, success_count,
                     confidence, switch_trades, switch_patterns, ai_strategies,
                     ai_win_rate, avg_profit, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats.strategy, stats.regime, stats.level,
                    stats.total_trades, stats.success_count, stats.confidence,
                    stats.switch_trades, stats.switch_patterns, stats.ai_strategies,
                    stats.ai_win_rate, stats.avg_profit, stats.last_updated
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ 진화 통계 저장 오류: {e}")
    
    def _on_level_up(self, stats: EvolutionStats, old_level: int, new_level: int):
        """레벨 업 시 처리"""
        level_names = {
            1: "기본 전략",
            2: "전환 조합 학습",
            3: "AI 자동 생성",
            4: "유전 알고리즘 진화"
        }
        
        print(f"🎉 [{stats.strategy}_{stats.regime}] 레벨 업! "
              f"{level_names[old_level]} → {level_names[new_level]}")
        
        # Level 3 진입 시: AI 전략 생성 시작
        if new_level == EvolutionLevel.AI_GENERATED:
            self._trigger_ai_strategy_generation(stats)
        
        # Level 4 진입 시: 유전 알고리즘 초기화
        elif new_level == EvolutionLevel.GENETIC:
            self._trigger_genetic_evolution(stats)
    
    # ========================================================================
    # Level 3: AI 자동 전략 생성
    # ========================================================================
    def _trigger_ai_strategy_generation(self, stats: EvolutionStats):
        """AI 전략 생성 트리거"""
        print(f"🤖 [{stats.key}] AI 전략 생성 시작...")
        
        # 성공 패턴 분석 → 새 전략 생성
        # (실제 구현은 학습 데이터 기반)
        pass
    
    def generate_ai_strategy(self, base_strategy: str, regime: str,
                            success_patterns: List[Dict]) -> Optional[AIGeneratedStrategy]:
        """성공 패턴 기반 AI 전략 생성"""
        if not success_patterns:
            return None
        
        # 성공 패턴에서 공통 조건 추출
        common_conditions = self._extract_common_conditions(success_patterns)
        
        # 최적 청산 파라미터 계산
        exit_params = self._calculate_optimal_exit_params(success_patterns)
        
        strategy_id = f"ai_gen_{base_strategy}_{regime}_{int(time.time())}"
        
        ai_strategy = AIGeneratedStrategy(
            strategy_id=strategy_id,
            base_strategy=base_strategy,
            regime=regime,
            conditions=common_conditions,
            exit_params=exit_params,
            performance={},
            created_at=int(time.time())
        )
        
        # DB 저장
        self._save_ai_strategy(ai_strategy)
        
        # 캐시 저장
        self._ai_strategies[strategy_id] = ai_strategy
        
        # 통계 업데이트
        stats = self.get_evolution_stats(base_strategy, regime)
        stats.ai_strategies += 1
        self._save_stats_to_db(stats)
        
        print(f"✅ AI 전략 생성 완료: {strategy_id}")
        
        return ai_strategy
    
    def _extract_common_conditions(self, patterns: List[Dict]) -> Dict:
        """성공 패턴에서 공통 조건 추출"""
        if not patterns:
            return {}
        
        # 각 지표별 범위 수집
        rsi_values = [p.get('rsi', 50) for p in patterns if p.get('rsi')]
        volume_ratios = [p.get('volume_ratio', 1.0) for p in patterns if p.get('volume_ratio')]
        signal_scores = [p.get('signal_score', 0) for p in patterns if p.get('signal_score')]
        
        conditions = {}
        
        if rsi_values:
            conditions['rsi_range'] = (
                max(20, min(rsi_values) - 5),
                min(80, max(rsi_values) + 5)
            )
        
        if volume_ratios:
            conditions['min_volume_ratio'] = max(0.5, min(volume_ratios) * 0.8)
        
        if signal_scores:
            conditions['min_signal_score'] = max(0.05, min(signal_scores) * 0.9)
        
        return conditions
    
    def _calculate_optimal_exit_params(self, patterns: List[Dict]) -> Dict:
        """최적 청산 파라미터 계산"""
        if not patterns:
            return {'take_profit_pct': 10.0, 'stop_loss_pct': 5.0}
        
        profits = [p.get('profit_pct', 0) for p in patterns if p.get('profit_pct', 0) > 0]
        losses = [abs(p.get('profit_pct', 0)) for p in patterns if p.get('profit_pct', 0) < 0]
        
        # 75백분위 수익 = 목표 익절선
        if profits:
            take_profit = sorted(profits)[int(len(profits) * 0.75)] if profits else 10.0
        else:
            take_profit = 10.0
        
        # 75백분위 손실 = 손절선
        if losses:
            stop_loss = sorted(losses)[int(len(losses) * 0.75)] if losses else 5.0
        else:
            stop_loss = 5.0
        
        return {
            'take_profit_pct': round(take_profit, 2),
            'stop_loss_pct': round(stop_loss, 2),
            'max_holding_hours': 72
        }
    
    def _save_ai_strategy(self, strategy: AIGeneratedStrategy):
        """AI 전략 DB 저장"""
        if not self.db_path:
            return
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ai_generated_strategies
                    (strategy_id, base_strategy, regime, conditions, exit_params,
                     performance, created_at, trades_count, win_rate, avg_profit, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy.strategy_id, strategy.base_strategy, strategy.regime,
                    json.dumps(strategy.conditions), json.dumps(strategy.exit_params),
                    json.dumps(strategy.performance), strategy.created_at,
                    strategy.trades_count, strategy.win_rate, strategy.avg_profit,
                    1 if strategy.is_active else 0
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ AI 전략 저장 오류: {e}")
    
    # ========================================================================
    # Level 4: 유전 알고리즘 진화
    # ========================================================================
    def _trigger_genetic_evolution(self, stats: EvolutionStats):
        """유전 알고리즘 진화 트리거"""
        print(f"🧬 [{stats.key}] 유전 알고리즘 진화 시작...")
        
        # 초기 유전자 풀 생성
        self._initialize_gene_pool(stats.strategy, stats.regime)
    
    def _initialize_gene_pool(self, strategy: str, regime: str, population_size: int = 20):
        """초기 유전자 풀 생성"""
        from trade.core.strategies import STRATEGY_EXIT_RULES
        
        # 기본 전략 파라미터 가져오기
        base_rules = STRATEGY_EXIT_RULES.get(strategy)
        if not base_rules:
            return
        
        genes = []
        for i in range(population_size):
            # 기본값에서 ±20% 범위로 변이
            gene = StrategyGene(
                gene_id=f"gene_{strategy}_{regime}_{i}_{int(time.time())}",
                base_strategy=strategy,
                regime=regime,
                take_profit_pct=base_rules.take_profit_pct * random.uniform(0.8, 1.2),
                stop_loss_pct=base_rules.stop_loss_pct * random.uniform(0.8, 1.2),
                max_holding_hours=int(base_rules.max_holding_hours * random.uniform(0.8, 1.2)),
                trailing_trigger_pct=base_rules.trailing_trigger_pct * random.uniform(0.8, 1.2),
                trailing_distance_pct=base_rules.trailing_distance_pct * random.uniform(0.8, 1.2),
                min_signal_score=random.uniform(0.05, 0.2),
                min_rsi=random.uniform(15, 35),
                max_rsi=random.uniform(65, 85),
                min_volume_ratio=random.uniform(0.5, 1.5),
                generation=0
            )
            genes.append(gene)
            self._save_gene(gene)
        
        print(f"   🧬 초기 유전자 풀 {len(genes)}개 생성 완료")
    
    def evolve_generation(self, strategy: str, regime: str) -> List[StrategyGene]:
        """한 세대 진화 실행"""
        # 현재 유전자 풀 로드
        genes = self._load_gene_pool(strategy, regime)
        if len(genes) < 4:
            return genes
        
        # 적합도 기준 정렬
        genes.sort(key=lambda g: g.fitness, reverse=True)
        
        # 상위 50% 생존
        survivors = genes[:len(genes) // 2]
        
        # 교배로 자식 생성
        children = []
        while len(survivors) + len(children) < len(genes):
            parent1, parent2 = random.sample(survivors, 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            children.append(child)
            self._save_gene(child)
        
        new_generation = survivors + children
        
        # 세대 번호 업데이트
        max_gen = max(g.generation for g in genes)
        for gene in children:
            gene.generation = max_gen + 1
        
        print(f"   🧬 세대 {max_gen + 1} 진화 완료: 생존 {len(survivors)}개 + 자식 {len(children)}개")
        
        return new_generation
    
    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """두 유전자 교배"""
        child = StrategyGene(
            gene_id=f"gene_{parent1.base_strategy}_{parent1.regime}_{int(time.time())}_{random.randint(0, 999)}",
            base_strategy=parent1.base_strategy,
            regime=parent1.regime,
            # 각 파라미터를 부모 중 하나에서 랜덤 선택 또는 평균
            take_profit_pct=(parent1.take_profit_pct + parent2.take_profit_pct) / 2,
            stop_loss_pct=random.choice([parent1.stop_loss_pct, parent2.stop_loss_pct]),
            max_holding_hours=random.choice([parent1.max_holding_hours, parent2.max_holding_hours]),
            trailing_trigger_pct=(parent1.trailing_trigger_pct + parent2.trailing_trigger_pct) / 2,
            trailing_distance_pct=random.choice([parent1.trailing_distance_pct, parent2.trailing_distance_pct]),
            min_signal_score=(parent1.min_signal_score + parent2.min_signal_score) / 2,
            min_rsi=random.choice([parent1.min_rsi, parent2.min_rsi]),
            max_rsi=random.choice([parent1.max_rsi, parent2.max_rsi]),
            min_volume_ratio=(parent1.min_volume_ratio + parent2.min_volume_ratio) / 2,
            parent_ids=[parent1.gene_id, parent2.gene_id]
        )
        return child
    
    def _mutate(self, gene: StrategyGene, mutation_rate: float = 0.1) -> StrategyGene:
        """유전자 돌연변이"""
        if random.random() < mutation_rate:
            # 랜덤 파라미터 하나 변이
            param = random.choice([
                'take_profit_pct', 'stop_loss_pct', 'max_holding_hours',
                'trailing_trigger_pct', 'min_signal_score', 'min_rsi', 'max_rsi'
            ])
            
            current = getattr(gene, param)
            if isinstance(current, int):
                setattr(gene, param, int(current * random.uniform(0.7, 1.3)))
            else:
                setattr(gene, param, current * random.uniform(0.7, 1.3))
        
        return gene
    
    def _save_gene(self, gene: StrategyGene):
        """유전자 DB 저장"""
        if not self.db_path:
            return
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO strategy_genes
                    (gene_id, base_strategy, regime, take_profit_pct, stop_loss_pct,
                     max_holding_hours, trailing_trigger_pct, trailing_distance_pct,
                     min_signal_score, min_rsi, max_rsi, min_volume_ratio,
                     fitness, trades_count, win_rate, avg_profit, sharpe_ratio,
                     generation, parent_ids, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gene.gene_id, gene.base_strategy, gene.regime,
                    gene.take_profit_pct, gene.stop_loss_pct, gene.max_holding_hours,
                    gene.trailing_trigger_pct, gene.trailing_distance_pct,
                    gene.min_signal_score, gene.min_rsi, gene.max_rsi, gene.min_volume_ratio,
                    gene.fitness, gene.trades_count, gene.win_rate, gene.avg_profit,
                    gene.sharpe_ratio, gene.generation, json.dumps(gene.parent_ids),
                    int(time.time()), 1
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ 유전자 저장 오류: {e}")
    
    def _load_gene_pool(self, strategy: str, regime: str) -> List[StrategyGene]:
        """유전자 풀 로드"""
        genes = []
        
        if not self.db_path or not os.path.exists(self.db_path):
            return genes
        
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute("""
                    SELECT gene_id, take_profit_pct, stop_loss_pct, max_holding_hours,
                           trailing_trigger_pct, trailing_distance_pct, min_signal_score,
                           min_rsi, max_rsi, min_volume_ratio, fitness, trades_count,
                           win_rate, avg_profit, sharpe_ratio, generation, parent_ids
                    FROM strategy_genes
                    WHERE base_strategy = ? AND regime = ? AND is_active = 1
                    ORDER BY fitness DESC
                    LIMIT 50
                """, (strategy, regime))
                
                for row in cursor.fetchall():
                    gene = StrategyGene(
                        gene_id=row[0],
                        base_strategy=strategy,
                        regime=regime,
                        take_profit_pct=row[1],
                        stop_loss_pct=row[2],
                        max_holding_hours=row[3],
                        trailing_trigger_pct=row[4],
                        trailing_distance_pct=row[5],
                        min_signal_score=row[6],
                        min_rsi=row[7],
                        max_rsi=row[8],
                        min_volume_ratio=row[9],
                        fitness=row[10],
                        trades_count=row[11],
                        win_rate=row[12],
                        avg_profit=row[13],
                        sharpe_ratio=row[14],
                        generation=row[15],
                        parent_ids=json.loads(row[16]) if row[16] else []
                    )
                    genes.append(gene)
                    
        except Exception as e:
            print(f"⚠️ 유전자 풀 로드 오류: {e}")
        
        return genes
    
    # ========================================================================
    # 최적 전략 선택
    # ========================================================================
    def get_best_strategy_for_signal(self, signal_data: Dict, regime: str) -> Tuple[str, int, Dict]:
        """
        현재 시그널과 레짐에 맞는 최적 전략 반환
        
        Returns:
            (전략명, 진화레벨, 추가파라미터)
        """
        best_strategy = 'trend'
        best_level = 1
        best_params = {}
        best_score = 0.0
        
        from trade.core.strategies import StrategyType
        
        for strategy in StrategyType.all_types():
            level = self.get_evolution_level(strategy, regime)
            stats = self.get_evolution_stats(strategy, regime)
            
            # 기본 매칭 점수
            match_score = signal_data.get('strategy_scores', {}).get(strategy, {}).get('match', 0.5)
            
            # 레벨 보너스 (높은 레벨 = 더 검증됨)
            level_bonus = level * 0.1
            
            # 성과 보너스
            perf_bonus = stats.win_rate * 0.2 if stats.total_trades >= 10 else 0
            
            total_score = match_score + level_bonus + perf_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_strategy = strategy
                best_level = level
                
                # Level 3 이상: AI/유전자 파라미터 사용
                if level >= EvolutionLevel.AI_GENERATED:
                    best_params = self._get_evolved_params(strategy, regime, level)
        
        return best_strategy, best_level, best_params
    
    def _get_evolved_params(self, strategy: str, regime: str, level: int) -> Dict:
        """진화된 파라미터 조회"""
        params = {}
        
        if level == EvolutionLevel.GENETIC:
            # 최고 적합도 유전자 사용
            genes = self._load_gene_pool(strategy, regime)
            if genes:
                best_gene = genes[0]  # 이미 fitness 순 정렬됨
                params = {
                    'take_profit_pct': best_gene.take_profit_pct,
                    'stop_loss_pct': best_gene.stop_loss_pct,
                    'max_holding_hours': best_gene.max_holding_hours,
                    'trailing_trigger_pct': best_gene.trailing_trigger_pct,
                    'trailing_distance_pct': best_gene.trailing_distance_pct,
                    'min_signal_score': best_gene.min_signal_score,
                    'gene_id': best_gene.gene_id,
                    'generation': best_gene.generation
                }
        
        elif level == EvolutionLevel.AI_GENERATED:
            # AI 생성 전략 파라미터
            ai_strategies = self._load_ai_strategies(strategy, regime)
            if ai_strategies:
                best_ai = ai_strategies[0]  # 성과순 정렬 필요
                params = best_ai.exit_params.copy()
                params['ai_strategy_id'] = best_ai.strategy_id
                params['conditions'] = best_ai.conditions
        
        return params
    
    def _load_ai_strategies(self, strategy: str, regime: str) -> List[AIGeneratedStrategy]:
        """AI 생성 전략 로드"""
        strategies = []
        
        if not self.db_path or not os.path.exists(self.db_path):
            return strategies
        
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute("""
                    SELECT strategy_id, conditions, exit_params, performance,
                           created_at, trades_count, win_rate, avg_profit
                    FROM ai_generated_strategies
                    WHERE base_strategy = ? AND regime = ? AND is_active = 1
                    ORDER BY win_rate DESC
                    LIMIT 10
                """, (strategy, regime))
                
                for row in cursor.fetchall():
                    ai_strat = AIGeneratedStrategy(
                        strategy_id=row[0],
                        base_strategy=strategy,
                        regime=regime,
                        conditions=json.loads(row[1]) if row[1] else {},
                        exit_params=json.loads(row[2]) if row[2] else {},
                        performance=json.loads(row[3]) if row[3] else {},
                        created_at=row[4],
                        trades_count=row[5],
                        win_rate=row[6],
                        avg_profit=row[7]
                    )
                    strategies.append(ai_strat)
                    
        except Exception as e:
            print(f"⚠️ AI 전략 로드 오류: {e}")
        
        return strategies
    
    # ========================================================================
    # 진화 상태 요약
    # ========================================================================
    def get_evolution_summary(self) -> Dict[str, Any]:
        """전체 진화 상태 요약"""
        summary = {
            'total_combinations': 0,
            'by_level': {1: 0, 2: 0, 3: 0, 4: 0},
            'top_performers': [],
            'recent_level_ups': []
        }
        
        if not self.db_path or not os.path.exists(self.db_path):
            return summary
        
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # 레벨별 개수
                cursor = conn.execute("""
                    SELECT level, COUNT(*) FROM strategy_evolution
                    GROUP BY level
                """)
                for row in cursor.fetchall():
                    summary['by_level'][row[0]] = row[1]
                    summary['total_combinations'] += row[1]
                
                # 상위 성과자
                cursor = conn.execute("""
                    SELECT strategy, regime, level, avg_profit, total_trades
                    FROM strategy_evolution
                    WHERE total_trades >= 10
                    ORDER BY avg_profit DESC
                    LIMIT 5
                """)
                for row in cursor.fetchall():
                    summary['top_performers'].append({
                        'key': f"{row[0]}_{row[1]}",
                        'level': row[2],
                        'avg_profit': row[3],
                        'trades': row[4]
                    })
                    
        except Exception as e:
            print(f"⚠️ 진화 요약 조회 오류: {e}")
        
        return summary
    
    def print_evolution_status(self):
        """진화 상태 출력"""
        summary = self.get_evolution_summary()
        
        level_names = {
            1: "기본",
            2: "전환학습",
            3: "AI생성",
            4: "유전진화"
        }
        
        print("\n" + "=" * 60)
        print("🧬 전략 진화 시스템 상태")
        print("=" * 60)
        
        print(f"\n📊 전체 조합 수: {summary['total_combinations']}개")
        print("\n📈 레벨별 분포:")
        for level, count in summary['by_level'].items():
            bar = "█" * min(count, 20)
            print(f"   Level {level} ({level_names[level]}): {bar} {count}개")
        
        if summary['top_performers']:
            print("\n🏆 상위 성과 조합:")
            for perf in summary['top_performers']:
                print(f"   {perf['key']}: Level {perf['level']}, "
                      f"평균수익 {perf['avg_profit']:+.2f}%, {perf['trades']}거래")
        
        print("=" * 60 + "\n")


# ============================================================================
# 싱글톤 인스턴스
# ============================================================================
_evolution_manager: Optional[StrategyEvolutionManager] = None


def get_evolution_manager() -> StrategyEvolutionManager:
    """진화 관리자 싱글톤 인스턴스 반환"""
    global _evolution_manager
    if _evolution_manager is None:
        db_path = os.environ.get('STRATEGY_DB_PATH', '')
        _evolution_manager = StrategyEvolutionManager(db_path)
    return _evolution_manager


# ============================================================================
# 편의 함수들
# ============================================================================
def get_strategy_level(strategy: str, regime: str) -> int:
    """전략×레짐 조합의 현재 레벨 조회"""
    return get_evolution_manager().get_evolution_level(strategy, regime)


def update_evolution_stats(strategy: str, regime: str, success: bool, 
                          profit_pct: float, is_switch: bool = False,
                          switch_from: str = None) -> EvolutionStats:
    """거래 결과로 진화 통계 업데이트"""
    return get_evolution_manager().update_trade_result(
        strategy, regime, success, profit_pct, is_switch, switch_from
    )


def get_best_evolved_strategy(signal_data: Dict, regime: str) -> Tuple[str, int, Dict]:
    """현재 상황에 최적인 진화된 전략 반환"""
    return get_evolution_manager().get_best_strategy_for_signal(signal_data, regime)


def print_evolution_status():
    """진화 상태 출력"""
    get_evolution_manager().print_evolution_status()
