import os
import sys
import time
import sqlite3
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional

# 경로 설정 (trade.core.database에서 중앙화된 설정 로드)
try:
    from trade.core.database import TRADING_SYSTEM_DB_PATH, STRATEGY_DB_PATH, CANDLES_DB_PATH, get_db_connection
except ImportError:
    # 하위 호환성 및 대체 로직
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(current_dir)
    sys.path.insert(0, workspace_dir)
    sys.path.insert(0, current_dir)
    _DEFAULT_DB_DIR = os.path.join(workspace_dir, 'market', 'coin_market', 'data_storage')
    TRADING_SYSTEM_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'trading_system.db')
    STRATEGY_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'learning_strategies', 'common_strategies.db')
    CANDLES_DB_PATH = os.path.join(_DEFAULT_DB_DIR, 'trade_candles.db')
    def get_db_connection(db_path, read_only=True, **kwargs):
        timeout = kwargs.get('timeout', 60.0)
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        return conn

from trade.trade_executor import get_market_context
from trade.core.thompson import ThompsonSamplingLearner
from trade.core.thresholds import get_thresholds
from trade.core.learner.realtime import RealTimeLearner
from trade.core.learner.transfer import TransferLearner
from trade.core.learner.analyzer import PatternAnalyzer
from trade.core.learner.insight import MarketInsightMiner
from trade.core.learner.evaluator import PostTradeEvaluator
from trade.core.learner.evolution import EvolutionEngine

# 🆕 전략 시스템 임포트
try:
    from trade.core.strategies import (
        update_strategy_feedback, get_strategy_success_rate,
        get_market_strategy_preference, create_strategy_feedback_table,
        STRATEGY_EXIT_RULES, get_strategy_description,
        get_regime_adjustment, get_strategy_regime_compatibility  # 🆕 레짐 조정 함수
    )
    STRATEGY_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 전략 시스템 로드 실패: {e}")
    STRATEGY_SYSTEM_AVAILABLE = False

# 🧬 전략 진화 시스템 임포트
try:
    from trade.core.strategy_evolution import (
        get_evolution_manager, update_evolution_stats, get_strategy_level,
        EvolutionLevel, print_evolution_status
    )
    EVOLUTION_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 전략 진화 시스템 로드 실패: {e}")
    EVOLUTION_SYSTEM_AVAILABLE = False


# ============================================================================
# 🆕 손실 원인 분석 + 과잉 회피 방지 통합 시스템
# ============================================================================
class LossCause:
    """손실 원인 분류 (확정 손실)"""
    ENTRY_TIMING = "entry_timing"      # 진입 타이밍 실패 (매수 직후 하락)
    EXIT_TIMING = "exit_timing"        # 청산 타이밍 실패 (수익→손실 전환)
    STRATEGY_MISMATCH = "strategy_mismatch"  # 전략-레짐 부조화
    MARKET_SHOCK = "market_shock"      # 시장 급변 (예상치 못한 급락)
    OVERHOLD = "overhold"              # 보유 기간 초과
    UNKNOWN = "unknown"                # 원인 불명


class DrawdownAnalysis:
    """
    🆕 미실현 손실(Drawdown) 분석
    
    보유 중 큰 하락을 겪었지만 회복한 케이스 분석
    - 확정 손실 분석과 별도로 "내성" 학습에 활용
    """
    
    MIN_DRAWDOWN_PCT = 5.0  # 5% 이상 하락만 분석
    
    @staticmethod
    def analyze_drawdown(
        entry_price: float,
        exit_price: float,
        min_price_during_hold: float,
        final_profit_pct: float
    ) -> dict:
        """
        보유 중 최대 하락(MAE) 분석
        
        Returns:
            {
                'max_drawdown_pct': 보유 중 최대 하락률,
                'recovered': 회복 여부 (최종 손익 >= 0),
                'recovery_pct': 저점 대비 회복률,
                'analysis_type': 'deep_drawdown_recovered' | 'deep_drawdown_loss' | 'shallow_drawdown'
            }
        """
        if entry_price <= 0 or min_price_during_hold <= 0:
            return {}
        
        # MAE (Maximum Adverse Excursion) 계산
        max_drawdown_pct = ((entry_price - min_price_during_hold) / entry_price) * 100
        
        result = {
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'final_profit_pct': round(final_profit_pct, 2),
            'recovered': final_profit_pct >= 0,
            'recovery_pct': 0.0,
            'analysis_type': 'shallow_drawdown'
        }
        
        # 5% 이상 하락인 경우만 상세 분석
        if max_drawdown_pct >= DrawdownAnalysis.MIN_DRAWDOWN_PCT:
            # 저점 대비 회복률 계산
            if min_price_during_hold > 0:
                result['recovery_pct'] = round(
                    ((exit_price - min_price_during_hold) / min_price_during_hold) * 100, 2
                )
            
            if final_profit_pct >= 0:
                result['analysis_type'] = 'deep_drawdown_recovered'  # 🎉 버텨서 회복
            else:
                result['analysis_type'] = 'deep_drawdown_loss'  # 😢 못 버티고 손절
        
        return result
    
    @staticmethod
    def get_learning_weight_for_drawdown(analysis: dict) -> float:
        """
        미실현 손실 분석 결과에 따른 학습 가중치
        
        - 버텨서 회복한 케이스: 긍정적 학습 (인내심 강화)
        - 못 버텨서 손절한 케이스: 이미 확정 손실로 학습됨 (중복 방지)
        """
        analysis_type = analysis.get('analysis_type', 'shallow_drawdown')
        
        if analysis_type == 'deep_drawdown_recovered':
            # 🎉 버텨서 회복 → 인내심 학습 가중치
            max_dd = analysis.get('max_drawdown_pct', 0)
            # 더 깊은 하락에서 회복할수록 높은 가중치
            if max_dd >= 10:
                return 1.5  # 10%+ 하락에서 회복 → 높은 학습 가치
            elif max_dd >= 7:
                return 1.3
            else:
                return 1.1
        
        elif analysis_type == 'deep_drawdown_loss':
            # 확정 손실로 이미 학습되므로 중복 학습 방지
            return 0.0  # 학습하지 않음
        
        return 0.0  # shallow는 학습 대상 아님


class BalancedLearningGuard:
    """
    과잉 회피 방지 + 균형 학습 관리자
    
    손실 학습이 과도해져서 매수를 꺼려하는 현상을 방지
    """
    
    # 설정값
    MIN_BUY_PROBABILITY = 0.15         # 최소 매수 확률 (15% 이하로 내려가지 않음)
    MAX_LOSS_WEIGHT = 2.0              # 손실 학습 가중치 상한선
    TIME_DECAY_DAYS = 14               # 시간 감쇠 기준일 (14일 후 50% 감쇠)
    REGIME_CHANGE_DECAY = 0.7          # 레짐 변경 시 과거 학습 감쇠 (30% 감소)
    
    # 🆕 손실 분석 기준
    MIN_LOSS_PCT_FOR_ANALYSIS = 5.0    # 5% 이상 손실만 원인 분석 (잦은 분석/과잉 회피 방지)
    
    @staticmethod
    def apply_time_decay(weight: float, trade_timestamp: int) -> float:
        """
        시간 감쇠 적용: 오래된 손실일수록 영향력 감소
        
        - 당일: 100%
        - 1주일: ~75%
        - 2주일: ~50%
        - 1개월: ~25%
        """
        now = int(time.time())
        age_days = (now - trade_timestamp) / 86400  # 일 단위
        
        if age_days <= 0:
            return weight
        
        # 지수 감쇠: weight * e^(-age/decay_constant)
        decay_constant = BalancedLearningGuard.TIME_DECAY_DAYS
        decay_factor = pow(0.5, age_days / decay_constant)
        
        return weight * max(0.1, decay_factor)  # 최소 10%는 유지
    
    @staticmethod
    def cap_loss_weight(weight: float, is_loss: bool) -> float:
        """
        손실 학습 가중치 상한선 적용
        
        손실에 대한 과도한 페널티 방지
        """
        if is_loss:
            return min(weight, BalancedLearningGuard.MAX_LOSS_WEIGHT)
        return weight
    
    @staticmethod
    def ensure_minimum_probability(thompson_score: float, pattern: str = None) -> float:
        """
        최소 매수 확률 보장
        
        아무리 손실이 많아도 완전히 매수를 거부하지 않도록 함
        """
        return max(thompson_score, BalancedLearningGuard.MIN_BUY_PROBABILITY)
    
    @staticmethod
    def calculate_balanced_weight(
        base_weight: float,
        is_loss: bool,
        trade_timestamp: int,
        loss_cause: str = None,
        regime_changed: bool = False
    ) -> float:
        """
        균형 잡힌 학습 가중치 계산 (손실 분석 + 과잉 회피 방지 통합)
        
        Args:
            base_weight: 기본 가중치
            is_loss: 손실 여부
            trade_timestamp: 거래 시각
            loss_cause: 손실 원인 (LossCause)
            regime_changed: 레짐 변경 여부
            
        Returns:
            조정된 가중치
        """
        weight = base_weight
        
        # 1. 시간 감쇠 적용
        weight = BalancedLearningGuard.apply_time_decay(weight, trade_timestamp)
        
        # 2. 손실인 경우 원인별 가중치 조정
        if is_loss and loss_cause:
            if loss_cause == LossCause.MARKET_SHOCK:
                # 시장 급변은 예측 불가 → 가중치 낮춤 (과학습 방지)
                weight *= 0.5
            elif loss_cause == LossCause.STRATEGY_MISMATCH:
                # 전략-레짐 부조화는 중요한 학습 포인트
                weight *= 1.2
            elif loss_cause == LossCause.ENTRY_TIMING:
                # 진입 타이밍 실패는 보통 가중치
                weight *= 1.0
            elif loss_cause == LossCause.EXIT_TIMING:
                # 청산 타이밍 실패 (수익 → 손실) 중요도 높음
                weight *= 1.3
            elif loss_cause == LossCause.OVERHOLD:
                # 보유 기간 초과는 학습 필요
                weight *= 1.1
        
        # 3. 레짐 변경 시 과거 학습 영향력 감소
        if regime_changed:
            weight *= BalancedLearningGuard.REGIME_CHANGE_DECAY
        
        # 4. 손실 가중치 상한선 적용
        weight = BalancedLearningGuard.cap_loss_weight(weight, is_loss)
        
        return round(weight, 3)


def get_balanced_thompson_score(thompson_sampler, pattern: str) -> float:
    """
    Thompson 점수 조회 시 최소 매수 확률 보장
    
    손실 학습이 과도해도 완전히 매수를 거부하지 않도록 함
    """
    try:
        raw_score = thompson_sampler.get_success_probability(pattern)
        return BalancedLearningGuard.ensure_minimum_probability(raw_score, pattern)
    except Exception:
        return BalancedLearningGuard.MIN_BUY_PROBABILITY


def analyze_loss_cause(
    entry_price: float,
    exit_price: float,
    entry_timestamp: int,
    exit_timestamp: int,
    max_profit_pct: float,
    profit_loss_pct: float,
    strategy_type: str,
    market_regime: str,
    candle_data: pd.DataFrame = None
) -> tuple:
    """
    손실 원인 분석 (🆕 5% 이상 손실만 분석)
    
    Args:
        entry_price: 진입가
        exit_price: 청산가
        entry_timestamp: 진입 시각
        exit_timestamp: 청산 시각
        max_profit_pct: 보유 중 최대 수익률
        profit_loss_pct: 최종 손익률
        strategy_type: 전략 타입
        market_regime: 시장 레짐
        candle_data: 캔들 데이터 (옵션)
        
    Returns:
        (loss_cause: str, details: dict)
        - 손실이 아니거나 5% 미만이면 (None, {}) 반환
    """
    # 🆕 손실이 아니거나 기준 미만이면 분석하지 않음
    if profit_loss_pct >= 0:
        return None, {}  # 손실이 아님
    
    if abs(profit_loss_pct) < BalancedLearningGuard.MIN_LOSS_PCT_FOR_ANALYSIS:
        return None, {}  # 🆕 5% 미만 손실은 분석하지 않음 (과잉 회피 방지)
    
    details = {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit_loss_pct': profit_loss_pct,
        'max_profit_pct': max_profit_pct
    }
    
    holding_hours = (exit_timestamp - entry_timestamp) / 3600
    details['holding_hours'] = holding_hours
    
    # 1. 수익 → 손실 전환 (청산 타이밍 실패)
    if max_profit_pct and max_profit_pct > 1.0 and profit_loss_pct < 0:
        details['missed_profit'] = max_profit_pct - profit_loss_pct
        return LossCause.EXIT_TIMING, details
    
    # 2. 진입 직후 급락 (진입 타이밍 실패)
    # 캔들 데이터가 있으면 진입 후 1시간 내 최저가 확인
    if candle_data is not None and len(candle_data) > 0:
        try:
            early_candles = candle_data[
                (candle_data['timestamp'] >= entry_timestamp) & 
                (candle_data['timestamp'] <= entry_timestamp + 3600)
            ]
            if len(early_candles) > 0:
                min_low = early_candles['low'].min()
                early_drop = ((entry_price - min_low) / entry_price) * 100
                if early_drop > 2.0:  # 1시간 내 2% 이상 하락
                    details['early_drop_pct'] = early_drop
                    return LossCause.ENTRY_TIMING, details
        except Exception:
            pass
    
    # 3. 전략-레짐 부조화 체크
    if STRATEGY_SYSTEM_AVAILABLE and strategy_type and market_regime:
        try:
            compatibility, _ = get_strategy_regime_compatibility(strategy_type, market_regime)
            if compatibility < 0.6:  # 호환성 낮음
                details['compatibility'] = compatibility
                details['strategy'] = strategy_type
                details['regime'] = market_regime
                return LossCause.STRATEGY_MISMATCH, details
        except Exception:
            pass
    
    # 4. 급격한 손실 (시장 급변)
    if profit_loss_pct < -5.0 and holding_hours < 2:  # 2시간 내 5% 이상 손실
        details['rapid_loss'] = True
        return LossCause.MARKET_SHOCK, details
    
    # 5. 보유 기간 초과
    if holding_hours > 48:  # 48시간 이상 보유
        details['overhold_hours'] = holding_hours
        return LossCause.OVERHOLD, details
    
    # 6. 원인 불명
    return LossCause.UNKNOWN, details

class VirtualTradingLearner:
    """가상매매 결과와 시그널을 대조하여 시스템을 실시간으로 진화시키는 엔진"""
    
    def __init__(self):
        print("🚀 진화형 학습 엔진 초기화 중...")
        self.db_path = TRADING_SYSTEM_DB_PATH
        self.thompson_sampler = ThompsonSamplingLearner(db_path=STRATEGY_DB_PATH)
        self.realtime_learner = RealTimeLearner(self.thompson_sampler)
        self.transfer_learner = TransferLearner(STRATEGY_DB_PATH, self.db_path, self.thompson_sampler)
        self.pattern_analyzer = PatternAnalyzer()
        self.market_miner = MarketInsightMiner(self)
        self.evaluator = PostTradeEvaluator(STRATEGY_DB_PATH)
        self.evolution_engine = EvolutionEngine(STRATEGY_DB_PATH)
        self.processed_trade_ids = set()

    # Note: 시그널 예측 검증 (_finalize_forecast_accuracy)은 
    # strategy_signal_generator.py의 validate_signals_incremental()로 이전됨

    def _execute_real_time_learning(self):
        """가상매매 피드백 데이터를 기반으로 실시간 학습 실행 (쓰기 모드 안정성 강화)
        🚀 [성능] iterrows → to_dict('records') 최적화
        🆕 [균형 학습] 손실 원인 분석 + 과잉 회피 방지 통합
        """
        try:
            with get_db_connection(self.db_path, read_only=False) as conn:
                # 1. 미학습 피드백 로드 - 🚀 동적 컬럼 조회 (테이블 스키마에 따라 유연하게 처리)
                # 먼저 테이블에 존재하는 컬럼 확인
                cursor = conn.execute("PRAGMA table_info(virtual_trade_feedback)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                
                # 기본 필수 컬럼
                base_cols = ['id', 'coin', 'signal_pattern', 'profit_loss_pct', 'exit_price', 'entry_price', 'market_conditions']
                # 선택적 컬럼 (있으면 포함, 없으면 제외)
                optional_cols = ['strategy_type', 'holding_duration', 'entry_timestamp', 'exit_timestamp', 'max_profit_pct', 'max_loss_pct']
                
                select_cols = base_cols + [col for col in optional_cols if col in existing_cols]
                
                query = f"SELECT {', '.join(select_cols)} FROM virtual_trade_feedback WHERE is_learned = 0"
                feedback_df = pd.read_sql(query, conn)
                
                if feedback_df.empty:
                    return 0
                
                print(f"📖 {len(feedback_df)}건의 가상매매 피드백 학습 중...")
                
                # 🆕 손실 원인별 통계 수집
                loss_cause_stats = defaultdict(lambda: {'count': 0, 'total_loss': 0.0})
                # 🆕 미실현 손실(Drawdown) 통계 수집
                drawdown_stats = {
                    'deep_recovered': {'count': 0, 'avg_drawdown': 0.0, 'avg_recovery': 0.0},
                    'deep_loss': {'count': 0, 'avg_drawdown': 0.0}
                }
                
                # 🚀 [성능] iterrows 대신 to_dict('records') 사용 (2~5배 빠름)
                learned_ids = []  # 배치 업데이트용
                for row in feedback_df.to_dict('records'):
                    # 2. 톰슨 샘플링 지식 업데이트
                    pattern = row['signal_pattern']
                    profit_pct = row['profit_loss_pct']
                    success = profit_pct > 0
                    is_loss = profit_pct < 0
                    
                    # 🆕 [통합] 시그널 점수와 학습 데이터 통합
                    signal_score = row.get('signal_score', 0.0) or 0.0
                    
                    # 🆕 호가 정밀도 인식 (Tick-Aware Learning)
                    from trade.trade_manager import get_bithumb_tick_size
                    current_price = row.get('exit_price', 0) or row.get('entry_price', 0)
                    tick_size = get_bithumb_tick_size(current_price)
                    
                    # 기본 가중치
                    weight = 1.0
                    
                    if tick_size > 0 and current_price > 0:
                        price_diff = abs(profit_pct / 100 * current_price)
                        ticks_moved = price_diff / tick_size
                        if ticks_moved < 3.0:
                            weight *= 0.5
                    
                    # 시장 상황 파싱
                    market_cond = json.loads(row['market_conditions']) if row['market_conditions'] else {}
                    current_regime = market_cond.get('regime', 'neutral')
                    strategy_type = row.get('strategy_type', 'trend')
                    
                    # ═══════════════════════════════════════════════════════════
                    # 🆕 [손실 원인 분석 + 과잉 회피 방지] 통합 시스템
                    # ═══════════════════════════════════════════════════════════
                    loss_cause = None
                    loss_details = {}
                    
                    if is_loss:
                        # 손실 원인 분석
                        loss_cause, loss_details = analyze_loss_cause(
                            entry_price=row.get('entry_price', 0) or 0,
                            exit_price=row.get('exit_price', 0) or 0,
                            entry_timestamp=row.get('entry_timestamp', 0) or 0,
                            exit_timestamp=row.get('exit_timestamp', 0) or 0,
                            max_profit_pct=row.get('max_profit_pct', 0) or 0,
                            profit_loss_pct=profit_pct,
                            strategy_type=strategy_type,
                            market_regime=current_regime
                        )
                        
                        # 손실 원인별 통계 수집
                        if loss_cause:
                            loss_cause_stats[loss_cause]['count'] += 1
                            loss_cause_stats[loss_cause]['total_loss'] += abs(profit_pct)
                        
                        # 🆕 균형 학습 가중치 계산 (과잉 회피 방지)
                        entry_ts = row.get('entry_timestamp', 0) or int(time.time())
                        weight = BalancedLearningGuard.calculate_balanced_weight(
                            base_weight=weight,
                            is_loss=True,
                            trade_timestamp=entry_ts,
                            loss_cause=loss_cause,
                            regime_changed=False  # TODO: 레짐 변경 감지 연동
                        )
                        
                        # 🆕 손실 원인별 패턴 학습 (세분화된 학습)
                        if loss_cause and loss_cause != LossCause.UNKNOWN:
                            cause_pattern = f"{pattern}_loss_{loss_cause}"
                            self.thompson_sampler.update_distribution(
                                cause_pattern, False, profit_pct=profit_pct, weight=weight * 0.8
                            )
                    else:
                        # 성공인 경우도 시간 감쇠 적용
                        entry_ts = row.get('entry_timestamp', 0) or int(time.time())
                        weight = BalancedLearningGuard.apply_time_decay(weight, entry_ts)
                    
                    # ═══════════════════════════════════════════════════════════
                    # 🆕 [미실현 손실 분석] 보유 중 큰 하락 후 회복 케이스 학습
                    # ═══════════════════════════════════════════════════════════
                    entry_price = row.get('entry_price', 0) or 0
                    exit_price = row.get('exit_price', 0) or 0
                    max_loss_pct = row.get('max_loss_pct', None)  # 보유 중 최대 손실률 (있으면)
                    
                    if entry_price > 0 and exit_price > 0:
                        # max_loss_pct가 없으면 max_profit_pct로 추정 (간접 계산)
                        if max_loss_pct is None:
                            # 최대 손실은 보통 최대 수익의 반대 방향 변동이므로 추정
                            # (정확한 값은 캔들 분석 필요하지만, 여기선 간소화)
                            max_profit = row.get('max_profit_pct', 0) or 0
                            # 수익으로 끝났지만 중간에 하락이 있었을 가능성
                            if success and max_profit > profit_pct:
                                # 수익인데 최대 수익보다 낮게 끝남 → 중간에 하락 있었음
                                estimated_drawdown = max(0, -profit_pct + 2)  # 대략적 추정
                            else:
                                estimated_drawdown = abs(min(0, profit_pct))
                            min_price_during_hold = entry_price * (1 - estimated_drawdown / 100)
                        else:
                            min_price_during_hold = entry_price * (1 - abs(max_loss_pct) / 100)
                        
                        # Drawdown 분석
                        dd_analysis = DrawdownAnalysis.analyze_drawdown(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            min_price_during_hold=min_price_during_hold,
                            final_profit_pct=profit_pct
                        )
                        
                        # 🎉 버텨서 회복한 케이스 → 긍정적 학습
                        if dd_analysis.get('analysis_type') == 'deep_drawdown_recovered':
                            dd_weight = DrawdownAnalysis.get_learning_weight_for_drawdown(dd_analysis)
                            if dd_weight > 0:
                                # "인내심" 패턴 학습 (버텨서 회복하면 좋다)
                                patience_pattern = f"{pattern}_patience_recovered"
                                self.thompson_sampler.update_distribution(
                                    patience_pattern, True, profit_pct=profit_pct, weight=dd_weight
                                )
                                
                                # 통계 수집
                                drawdown_stats['deep_recovered']['count'] += 1
                                drawdown_stats['deep_recovered']['avg_drawdown'] += dd_analysis['max_drawdown_pct']
                                drawdown_stats['deep_recovered']['avg_recovery'] += dd_analysis['recovery_pct']
                        
                        # 😢 깊은 하락 후 손절 케이스 (확정 손실로 이미 학습되므로 통계만)
                        elif dd_analysis.get('analysis_type') == 'deep_drawdown_loss':
                            drawdown_stats['deep_loss']['count'] += 1
                            drawdown_stats['deep_loss']['avg_drawdown'] += dd_analysis['max_drawdown_pct']
                    
                    # 🆕 [통합] 시그널 점수 기반 가중치 조정
                    t = get_thresholds()
                    if abs(signal_score) > t.strong_buy:
                        if success:
                            weight *= 1.3
                        else:
                            weight *= 1.2  # 🆕 손실 시 가중치 축소 (1.5 → 1.2, 과잉 학습 방지)
                    elif abs(signal_score) > t.buy:
                        if success:
                            weight *= 1.1
                        else:
                            weight *= 1.0  # 🆕 손실 시 가중치 축소 (1.2 → 1.0)
                    
                    # 🆕 [레짐 기반 학습] 전략+레짐 호환성에 따라 가중치 조정
                    if STRATEGY_SYSTEM_AVAILABLE and strategy_type and strategy_type != 'None':
                        try:
                            compatibility, compat_desc = get_strategy_regime_compatibility(strategy_type, current_regime)
                            
                            if compatibility >= 1.2:  # 좋은 조합
                                if success:
                                    weight *= 1.3
                                else:
                                    weight *= 1.2  # 🆕 축소 (1.5 → 1.2)
                            elif compatibility <= 0.6:  # 나쁜 조합
                                if success:
                                    weight *= 1.4  # 예외 학습 중요
                                else:
                                    weight *= 0.5  # 🆕 예상된 실패 → 더 낮은 가중치 (0.7 → 0.5)
                        except Exception:
                            if current_regime == 'neutral':
                                weight *= 1.2 if success else 1.0
                    else:
                        if current_regime == 'neutral':
                            weight *= 1.2 if success else 1.0
                    
                    # 🆕 최종 가중치 상한선 적용 (과잉 학습 방지)
                    weight = BalancedLearningGuard.cap_loss_weight(weight, is_loss)
                    
                    # 🆕 [통합] 시그널 점수 + 레짐 정보를 패턴에 포함하여 학습
                    enhanced_pattern = f"{pattern}_sig{abs(signal_score):.2f}"
                    regime_pattern = f"{pattern}_{current_regime}"
                    
                    self.thompson_sampler.update_distribution(enhanced_pattern, success, profit_pct=profit_pct, weight=weight)
                    self.thompson_sampler.update_distribution(pattern, success, profit_pct=profit_pct, weight=weight * 0.8)
                    self.thompson_sampler.update_distribution(regime_pattern, success, profit_pct=profit_pct, weight=weight * 0.6)
                    
                    # 3. 실시간 학습기에 전달
                    self.realtime_learner.learn_from_trade(pattern, row['profit_loss_pct'])
                    
                    # 🆕 [전략 시스템] 전략별 + 레짐별 학습 피드백 저장
                    if STRATEGY_SYSTEM_AVAILABLE:
                        if strategy_type and strategy_type != 'None':
                            holding_hours = row.get('holding_duration', 0) / 3600.0
                            
                            try:
                                # 기본 전략 피드백
                                update_strategy_feedback(
                                    db_path=self.db_path,
                                    strategy_type=strategy_type,
                                    market_condition=current_regime,  # 🆕 레짐 정보 전달
                                    signal_pattern=pattern,
                                    success=success,
                                    profit_pct=row['profit_loss_pct'],
                                    holding_hours=holding_hours
                                )
                                
                                # 🆕 전략+레짐 조합 피드백 (더 세분화된 학습)
                                strategy_regime_key = f"{strategy_type}_{current_regime}"
                                update_strategy_feedback(
                                    db_path=self.db_path,
                                    strategy_type=strategy_regime_key,
                                    market_condition=current_regime,
                                    signal_pattern=pattern,
                                    success=success,
                                    profit_pct=row['profit_loss_pct'],
                                    holding_hours=holding_hours
                                )
                            except Exception as strat_err:
                                # 전략 피드백 저장 실패는 조용히 무시
                                pass
                    
                    # 4. 학습 완료 ID 수집 (🚀 배치 업데이트용)
                    learned_ids.append(row['id'])
                    self.processed_trade_ids.add(row['id'])
                
                # 🚀 [성능] 배치 UPDATE (개별 UPDATE 대신 한 번에 실행)
                if learned_ids:
                    placeholders = ','.join('?' * len(learned_ids))
                    conn.execute(f"UPDATE virtual_trade_feedback SET is_learned = 1 WHERE id IN ({placeholders})", learned_ids)
                
                conn.commit()
                
                # 🆕 [손실 원인 분석] 통계 출력 (5% 이상 손실만)
                if loss_cause_stats:
                    total_losses = sum(s['count'] for s in loss_cause_stats.values())
                    if total_losses > 0:
                        print(f"\n   📊 [손실 원인 분석] {total_losses}건 주요 손실(≥{BalancedLearningGuard.MIN_LOSS_PCT_FOR_ANALYSIS}%) 분석:")
                        cause_names = {
                            LossCause.ENTRY_TIMING: "진입 타이밍 ⏰",
                            LossCause.EXIT_TIMING: "청산 타이밍 📉",
                            LossCause.STRATEGY_MISMATCH: "전략-레짐 부조화 ⚖️",
                            LossCause.MARKET_SHOCK: "시장 급변 ⚡",
                            LossCause.OVERHOLD: "보유 기간 초과 ⏳",
                            LossCause.UNKNOWN: "원인 불명 ❓"
                        }
                        for cause, stats in sorted(loss_cause_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                            pct = (stats['count'] / total_losses) * 100
                            avg_loss = stats['total_loss'] / stats['count'] if stats['count'] > 0 else 0
                            cause_name = cause_names.get(cause, cause)
                            print(f"      - {cause_name}: {stats['count']}건 ({pct:.0f}%), 평균 -{avg_loss:.2f}%")
                        
                        # 🆕 가장 빈번한 손실 원인에 대한 권장사항
                        top_cause = max(loss_cause_stats.items(), key=lambda x: x[1]['count'])[0]
                        if top_cause == LossCause.ENTRY_TIMING:
                            print(f"      💡 권장: 진입 지연 또는 분할 매수 고려")
                        elif top_cause == LossCause.EXIT_TIMING:
                            print(f"      💡 권장: 트레일링 스탑 또는 부분 익절 활용")
                        elif top_cause == LossCause.STRATEGY_MISMATCH:
                            print(f"      💡 권장: 현재 레짐에 맞는 전략으로 전환 필요")
                        elif top_cause == LossCause.MARKET_SHOCK:
                            print(f"      💡 권장: 시장 급변은 예측 불가 - 과학습 주의 (가중치 낮춤)")
                        elif top_cause == LossCause.OVERHOLD:
                            print(f"      💡 권장: 보유 기간 목표 단축 또는 시간 기반 청산 규칙 추가")
                
                # 🆕 [미실현 손실 분석] Drawdown 통계 출력
                recovered_count = drawdown_stats['deep_recovered']['count']
                loss_count = drawdown_stats['deep_loss']['count']
                if recovered_count > 0 or loss_count > 0:
                    print(f"\n   📉 [미실현 손실 분석] 보유 중 {DrawdownAnalysis.MIN_DRAWDOWN_PCT}%+ 하락 케이스:")
                    
                    if recovered_count > 0:
                        avg_dd = drawdown_stats['deep_recovered']['avg_drawdown'] / recovered_count
                        avg_rec = drawdown_stats['deep_recovered']['avg_recovery'] / recovered_count
                        print(f"      🎉 버텨서 회복: {recovered_count}건 (평균 -{avg_dd:.1f}% → +{avg_rec:.1f}% 회복)")
                        print(f"         → '인내심' 패턴 긍정 학습 완료")
                    
                    if loss_count > 0:
                        avg_dd_loss = drawdown_stats['deep_loss']['avg_drawdown'] / loss_count
                        print(f"      😢 못 버티고 손절: {loss_count}건 (평균 -{avg_dd_loss:.1f}% 하락)")
                        print(f"         → 확정 손실로 이미 학습됨 (중복 학습 방지)")
                    
                    # 회복률 기반 권장사항
                    if recovered_count > 0 and loss_count > 0:
                        recovery_rate = recovered_count / (recovered_count + loss_count) * 100
                        if recovery_rate >= 60:
                            print(f"      💪 회복률 {recovery_rate:.0f}% - 인내심이 수익으로 이어지는 경향")
                        elif recovery_rate <= 30:
                            print(f"      ⚠️ 회복률 {recovery_rate:.0f}% - 손절 기준 재검토 필요")
                
                return len(feedback_df)
                
        except Exception as e:
            # 🔇 DB 접근 오류는 조용히 처리
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 실시간 학습 오류: {e}")
            return 0

    def _run_post_trade_evaluation(self) -> int:
        """🆕 매도 후 가격 추적 및 MFE/MAE 평가 실행 (시장 인사이트 복기 포함)"""
        completed_count = 0
        try:
            # 🚀 [Fix] 시스템 시간이 아닌 DB의 가장 최신 캔들 시간 기준
            from trade.core.database import CANDLES_DB_PATH
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as c_conn:
                max_ts_row = c_conn.execute("SELECT MAX(timestamp) FROM candles").fetchone()
                if not max_ts_row or not max_ts_row[0]:
                    print("⚠️ 캔들 데이터가 없어 복기 분석을 건너뜁니다.")
                    return 0
                
                latest_db_ts = max_ts_row[0]
                cutoff_ts = latest_db_ts - (24 * 3600)
                
                from datetime import datetime
                dt_str = datetime.fromtimestamp(latest_db_ts).strftime('%m-%d %H:%M')
                print(f"📊 [정밀 분석] 시장 인사이트 복기 시작... (데이터 시각: ~{dt_str})")
                
                # 최근 24시간 내 코인별 시가/종가 가져오기
                # 🚀 [Fix] 사용 가능한 interval 동적 확인 (1h가 없으면 240m 또는 1d 사용)
                # 먼저 사용 가능한 interval 확인
                available_intervals = pd.read_sql("""
                    SELECT DISTINCT interval FROM candles 
                    WHERE timestamp > ? 
                    ORDER BY 
                        CASE interval
                            WHEN '15m' THEN 1
                            WHEN '30m' THEN 2
                            WHEN '240m' THEN 3
                            WHEN '1d' THEN 4
                            ELSE 5
                        END
                """, c_conn, params=(cutoff_ts,))
                
                # 우선순위: 240m(4h) > 1d > 30m > 15m
                target_interval = None
                for preferred in ['240m', '1d', '30m', '15m']:
                    if preferred in available_intervals['interval'].values:
                        target_interval = preferred
                        break
                
                if target_interval is None:
                    # 사용 가능한 interval이 없으면 첫 번째 것 사용
                    if not available_intervals.empty:
                        target_interval = available_intervals['interval'].iloc[0]
                    else:
                        print("   ⚠️ 사용 가능한 캔들 interval이 없습니다.")
                        target_interval = '240m'  # 기본값
                
                # 🚀 [Fix] FIRST_VALUE/LAST_VALUE를 활용하여 24시간 변동폭 계산
                vol_df = pd.read_sql("""
                    SELECT DISTINCT symbol, 
                           FIRST_VALUE(close) OVER (PARTITION BY symbol ORDER BY timestamp ASC) as open_p,
                           LAST_VALUE(close) OVER (PARTITION BY symbol ORDER BY timestamp ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as close_p
                    FROM candles 
                    WHERE interval = ? AND timestamp > ?
                """, c_conn, params=(target_interval, cutoff_ts))
                
                if target_interval != '1h':
                    print(f"   ℹ️ 1h 캔들 없음 -> {target_interval} 캔들 사용 (24시간 변동폭 계산)")
                
                if not vol_df.empty:
                    vol_df['change'] = (vol_df['close_p'] - vol_df['open_p']) / vol_df['open_p'] * 100
                    big_movers = vol_df[vol_df['change'].abs() >= 5.0] # 5% 급등락 기준
                    
                    if not big_movers.empty:
                        print(f"   📈 최근 24시간 5% 이상 변동 코인 {len(big_movers)}개 감지 (복기 분석 중...)")
                        for _, row in big_movers.sort_values('change', ascending=False).head(5).iterrows():
                            print(f"      - {row['symbol']}: {row['change']:+.2f}% 변동")
                    else:
                        print("   ℹ️ 최근 24시간 내 ±5% 이상 변동한 코인 없음")

            # 기존 매도 품질 평가 로직 계속 진행 (trading_system.db 연결 필요)
            with get_db_connection(self.db_path, read_only=True) as conn:
                trades_df = pd.read_sql("""
                    SELECT coin, entry_price, exit_price, entry_timestamp, exit_timestamp,
                           profit_loss_pct, signal_pattern,
                           entry_strategy, exit_strategy, strategy_switch_count, switch_success
                    FROM virtual_trade_feedback 
                    WHERE exit_timestamp > ? AND is_learned = 1
                    ORDER BY exit_timestamp DESC
                    LIMIT 100
                """, conn, params=(cutoff_ts,))
            
            if trades_df.empty:
                return 0
            
            # 2. 각 거래를 evaluator에 추가 (아직 추적 중이 아닌 것만)
            for _, trade in trades_df.iterrows():
                trade_id = f"{trade['coin']}_{trade['entry_timestamp']}"
                if trade_id not in self.evaluator.tracked_trades:
                    self.evaluator.add_trade({
                        'coin': trade['coin'],
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['exit_price'],
                        'entry_timestamp': trade['entry_timestamp'],
                        'exit_timestamp': trade['exit_timestamp'],
                        'profit_loss_pct': trade['profit_loss_pct'],
                        'max_profit_pct': 0.0,  # 🆕 기본값 사용
                        'signal_pattern': trade.get('signal_pattern', 'unknown'),
                    })
            
            # 3. 현재 가격 조회
            current_prices = self._get_current_prices()
            
            # 4. 평가 실행
            if current_prices:
                completed_ids = self.evaluator.check_evaluations(current_prices)
                completed_count = len(completed_ids)
            
            # 5. 평가 결과를 Thompson Sampling에 반영
            feedbacks = self.evaluator.get_pending_feedback()
            for fb in feedbacks:
                pattern = fb.get('signal_pattern', 'unknown')
                adjustment = fb.get('adjustment_weight', 0.0)
                
                if fb.get('is_panic_sell'):
                    # 패닉 셀: 매도 기준을 더 높이도록 학습
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_sell_quality",
                        success=False,
                        profit_pct=-abs(fb.get('mfe', 0)),
                        weight=1.5
                    )
                elif fb.get('is_perfect_exit'):
                    # 완벽한 매도: 이 패턴의 신뢰도 상승
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_sell_quality",
                        success=True,
                        profit_pct=abs(fb.get('mae', 0)),
                        weight=1.5
                    )
            
            # 🆕 [전략 분리 학습] 진입/청산/전환 성공률 각각 학습
            self._learn_strategy_separated(trades_df)
            
            return completed_count
            
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 매도 품질 평가 실행 오류: {e}")
            return 0
    
    def _get_current_prices(self) -> Dict[str, float]:
        """현재 코인 가격 조회 (캔들 DB의 최신 데이터 기준)"""
        prices = {}
        try:
            from trade.core.database import CANDLES_DB_PATH
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
                # 🚀 [Fix] 전체 DB 기준 최신 타임스탬프 먼저 확보
                max_ts = conn.execute("SELECT MAX(timestamp) FROM candles").fetchone()[0]
                if not max_ts: return {}
                
                # 최신 타임스탬프에 해당하는 가격들만 조회
                df = pd.read_sql("""
                    SELECT symbol, close 
                    FROM candles 
                    WHERE timestamp = ?
                """, conn, params=(max_ts,))
                
                for _, row in df.iterrows():
                    prices[row['symbol']] = float(row['close'])
        except:
            pass
        return prices

    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 [전략 분리 학습] 진입/청산/전환 성공률 각각 학습
    # ═══════════════════════════════════════════════════════════════════════════
    def _learn_strategy_separated(self, trades_df: pd.DataFrame) -> int:
        """
        전략별 분리 학습:
        1. 진입 전략 정확도 (entry_strategy)
        2. 청산 전략 정확도 (exit_strategy)
        3. 전략 전환 성공률 (scalp_to_swing 등)
        4. 🆕 전략+레짐 조합별 학습
        🚀 [성능] iterrows → to_dict('records') 최적화
        """
        if trades_df.empty:
            return 0
        
        learned_count = 0
        regime_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'profit_sum': 0.0})
        
        try:
            from trade.core.strategies import update_strategy_feedback, create_strategy_feedback_table
            from trade.core.database import STRATEGY_DB_PATH
            
            # 🔧 테이블이 없을 수 있으므로 먼저 생성 (IF NOT EXISTS)
            try:
                create_strategy_feedback_table(STRATEGY_DB_PATH)
            except Exception:
                pass  # 이미 존재하면 무시
            
            # 🚀 [성능] iterrows 대신 to_dict('records') 사용
            for trade in trades_df.to_dict('records'):
                entry_strategy = trade.get('entry_strategy', 'trend')
                exit_strategy = trade.get('exit_strategy', entry_strategy)
                switch_count = trade.get('strategy_switch_count', 0) or 0
                switch_success = trade.get('switch_success', -1)
                profit_pct = trade.get('profit_loss_pct', 0.0) or 0.0
                pattern = trade.get('signal_pattern', 'unknown')
                
                # 성공 여부
                success = profit_pct > 0
                
                # 보유 시간 계산
                entry_ts = trade.get('entry_timestamp', 0) or 0
                exit_ts = trade.get('exit_timestamp', 0) or 0
                holding_hours = (exit_ts - entry_ts) / 3600.0 if exit_ts > entry_ts else 0
                
                # 🆕 시장 레짐 추출 (market_conditions에서)
                market_regime = 'neutral'
                market_cond_str = trade.get('market_conditions', '')
                if market_cond_str:
                    try:
                        market_cond = json.loads(market_cond_str) if isinstance(market_cond_str, str) else market_cond_str
                        market_regime = market_cond.get('regime', 'neutral')
                    except:
                        pass
                
                # 1️⃣ 진입 전략 학습
                if entry_strategy and entry_strategy != 'None':
                    update_strategy_feedback(
                        db_path=STRATEGY_DB_PATH,
                        strategy_type=entry_strategy,
                        market_condition=market_regime,  # 🆕 레짐 전달
                        signal_pattern=pattern,
                        success=success,
                        profit_pct=profit_pct,
                        holding_hours=holding_hours,
                        feedback_type='entry'
                    )
                    learned_count += 1
                    
                    # 🆕 전략+레짐 조합 통계 수집
                    strategy_regime_key = f"{entry_strategy}_{market_regime}"
                    regime_stats[strategy_regime_key]['total'] += 1
                    if success:
                        regime_stats[strategy_regime_key]['success'] += 1
                    regime_stats[strategy_regime_key]['profit_sum'] += profit_pct
                    
                    # 🆕 전략+레짐 조합 피드백도 저장
                    update_strategy_feedback(
                        db_path=STRATEGY_DB_PATH,
                        strategy_type=strategy_regime_key,
                        market_condition=market_regime,
                        signal_pattern=pattern,
                        success=success,
                        profit_pct=profit_pct,
                        holding_hours=holding_hours,
                        feedback_type='entry_regime'
                    )
                
                # 2️⃣ 청산 전략 학습 (전환된 경우만)
                if switch_count > 0 and exit_strategy != entry_strategy:
                    update_strategy_feedback(
                        db_path=STRATEGY_DB_PATH,
                        strategy_type=exit_strategy,
                        market_condition=market_regime,  # 🆕 레짐 전달
                        signal_pattern=pattern,
                        success=success,
                        profit_pct=profit_pct,
                        holding_hours=holding_hours,
                        feedback_type='exit'
                    )
                    learned_count += 1
                    
                    # 3️⃣ 전략 전환 성공률 학습 (레짐별)
                    switch_key = f"{entry_strategy}_to_{exit_strategy}"
                    switch_regime_key = f"{switch_key}_{market_regime}"  # 🆕 레짐별 전환 학습
                    
                    update_strategy_feedback(
                        db_path=STRATEGY_DB_PATH,
                        strategy_type=switch_key,
                        market_condition=market_regime,
                        signal_pattern=pattern,
                        success=(switch_success == 1) if switch_success >= 0 else success,
                        profit_pct=profit_pct,
                        holding_hours=holding_hours,
                        feedback_type='switch'
                    )
                    
                    # 🆕 레짐별 전환 학습
                    update_strategy_feedback(
                        db_path=STRATEGY_DB_PATH,
                        strategy_type=switch_regime_key,
                        market_condition=market_regime,
                        signal_pattern=pattern,
                        success=(switch_success == 1) if switch_success >= 0 else success,
                        profit_pct=profit_pct,
                        holding_hours=holding_hours,
                        feedback_type='switch_regime'
                    )
                    learned_count += 1
            
            if learned_count > 0:
                print(f"   📚 [전략 분리 학습] {learned_count}건 학습 완료 (진입/청산/전환 + 레짐별)")
                
                # 🆕 레짐별 성과 요약 출력
                if regime_stats:
                    print("   📊 [전략+레짐 조합 성과]")
                    sorted_stats = sorted(regime_stats.items(), 
                                         key=lambda x: x[1]['total'], reverse=True)[:5]
                    for key, stats in sorted_stats:
                        if stats['total'] >= 3:  # 최소 3건 이상만 출력
                            success_rate = stats['success'] / stats['total'] * 100
                            avg_profit = stats['profit_sum'] / stats['total']
                            print(f"      - {key}: 성공률 {success_rate:.0f}% ({stats['success']}/{stats['total']}), 평균수익 {avg_profit:+.2f}%")
                
                # 🧬 [진화 시스템] 진화 통계 업데이트
                if EVOLUTION_SYSTEM_AVAILABLE and regime_stats:
                    try:
                        evolution_updated = 0
                        for key, stats in regime_stats.items():
                            if stats['total'] < 2:
                                continue
                            
                            # key = "strategy_regime" 형태
                            parts = key.rsplit('_', 1)
                            if len(parts) == 2:
                                strategy, regime = parts[0], parts[1]
                            else:
                                strategy, regime = key, 'neutral'
                            
                            # 진화 통계에 각 거래 결과 반영
                            for _ in range(stats['total']):
                                success = stats['success'] > stats['total'] // 2
                                avg_profit = stats['profit_sum'] / stats['total']
                                
                                update_evolution_stats(
                                    strategy=strategy,
                                    regime=regime,
                                    success=success,
                                    profit_pct=avg_profit,
                                    is_switch=('_to_' in strategy),
                                    switch_from=strategy.split('_to_')[0] if '_to_' in strategy else None
                                )
                                evolution_updated += 1
                                break  # 배치 단위로 1회만 업데이트 (중복 방지)
                        
                        if evolution_updated > 0:
                            print(f"   🧬 [진화 시스템] {evolution_updated}개 전략×레짐 조합 진화 통계 업데이트")
                    except Exception as evo_err:
                        print(f"⚠️ 진화 통계 업데이트 오류: {evo_err}")
                
        except ImportError:
            pass  # 전략 모듈 없으면 무시
        except Exception as e:
            print(f"⚠️ 전략 분리 학습 오류: {e}")
        
        return learned_count

    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 [1] 진입 타이밍 최적화 학습
    # ═══════════════════════════════════════════════════════════════════════════
    def _learn_entry_timing_optimization(self) -> Dict[str, Any]:
        """매수 후 N분 동안 더 낮은 가격이 있었는지 분석하여 최적 진입 지연 시간 학습"""
        results = {'analyzed': 0, 'could_be_better': 0, 'avg_missed_pct': 0.0, 'optimal_delay_minutes': 0}
        
        try:
            from trade.core.database import CANDLES_DB_PATH
            
            # 최근 거래 내역 로드
            with get_db_connection(self.db_path, read_only=True) as conn:
                trades_df = pd.read_sql("""
                    SELECT coin, entry_price, entry_timestamp, signal_pattern, profit_loss_pct
                    FROM virtual_trade_feedback 
                    WHERE entry_timestamp > 0 AND entry_price > 0
                    ORDER BY entry_timestamp DESC
                    LIMIT 200
                """, conn)
            
            if trades_df.empty:
                return results
            
            delay_stats = defaultdict(lambda: {'better_count': 0, 'total': 0, 'saved_pct_sum': 0.0})
            missed_pcts = []
            
            # 🚀 [성능] 일괄 캔들 로드 + 메모리 필터링 (개별 쿼리 N회 → 1회)
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as c_conn:
                # 1. 필요한 코인 목록과 시간 범위 계산
                unique_coins = trades_df['coin'].unique().tolist()
                min_ts = int(trades_df['entry_timestamp'].min())
                max_ts = int(trades_df['entry_timestamp'].max()) + (120 * 60)  # 2시간 윈도우
                
                # 2. 모든 관련 캔들을 한 번에 로드
                placeholders = ','.join('?' * len(unique_coins))
                all_candles = pd.read_sql(f"""
                    SELECT symbol, timestamp, low, close
                    FROM candles 
                    WHERE symbol IN ({placeholders}) AND interval = '15m'
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY symbol, timestamp ASC
                """, c_conn, params=unique_coins + [min_ts, max_ts])
                
                # 3. 코인별로 인덱싱 (빠른 조회용)
                candle_cache = {coin: group.set_index('timestamp') 
                               for coin, group in all_candles.groupby('symbol', sort=False)}
                
                # 4. 각 거래 분석 (🚀 iterrows → to_dict)
                for trade in trades_df.to_dict('records'):
                    coin = trade['coin']
                    entry_ts = int(trade['entry_timestamp'])
                    entry_price = float(trade['entry_price'])
                    
                    if coin not in candle_cache:
                        continue
                    
                    # 메모리에서 필터링 (DB 쿼리 대신)
                    coin_candles = candle_cache[coin]
                    window_end = entry_ts + (120 * 60)
                    candles = coin_candles[(coin_candles.index >= entry_ts) & (coin_candles.index <= window_end)]
                    
                    if candles.empty:
                        continue
                    
                    results['analyzed'] += 1
                    
                    # 각 시간대별로 더 낮은 가격이 있었는지 확인
                    for delay_min in [15, 30, 45, 60, 90, 120]:
                        delay_ts = entry_ts + (delay_min * 60)
                        window = candles[candles.index <= delay_ts]
                        
                        if not window.empty:
                            min_low = window['low'].min()
                            if min_low < entry_price:
                                saved_pct = ((entry_price - min_low) / entry_price) * 100
                                delay_stats[delay_min]['better_count'] += 1
                                delay_stats[delay_min]['saved_pct_sum'] += saved_pct
                            delay_stats[delay_min]['total'] += 1
                    
                    # 전체 윈도우에서 최저가 확인
                    overall_min = candles['low'].min()
                    if overall_min < entry_price:
                        missed_pct = ((entry_price - overall_min) / entry_price) * 100
                        missed_pcts.append(missed_pct)
                        results['could_be_better'] += 1
            
            # 최적 지연 시간 계산
            if delay_stats:
                best_delay = 0
                best_score = 0
                
                for delay_min, stats in delay_stats.items():
                    if stats['total'] > 0:
                        hit_rate = stats['better_count'] / stats['total']
                        avg_saved = stats['saved_pct_sum'] / max(1, stats['better_count'])
                        # 점수 = 적중률 * 평균 절감률 (지연 시간에 대한 페널티 적용)
                        score = hit_rate * avg_saved * (1 - delay_min / 300)
                        
                        if score > best_score:
                            best_score = score
                            best_delay = delay_min
                
                results['optimal_delay_minutes'] = best_delay
            
            if missed_pcts:
                results['avg_missed_pct'] = sum(missed_pcts) / len(missed_pcts)
            
            # Thompson Sampling에 학습 결과 반영
            if results['analyzed'] > 10:
                improvement_rate = results['could_be_better'] / results['analyzed']
                self.thompson_sampler.update_distribution(
                    pattern="entry_timing_optimization",
                    success=improvement_rate < 0.3,  # 30% 미만이면 타이밍이 좋았다
                    profit_pct=results['avg_missed_pct'],
                    weight=1.0
                )
            
            return results
            
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 진입 타이밍 분석 오류: {e}")
            return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 [2] 보유 기간 최적화 학습
    # ═══════════════════════════════════════════════════════════════════════════
    def _learn_optimal_holding_period(self) -> Dict[str, Any]:
        """패턴별 최적 보유 기간 분석 및 학습"""
        results = {'patterns_analyzed': 0, 'recommendations': {}}
        
        try:
            from trade.core.database import CANDLES_DB_PATH
            
            with get_db_connection(self.db_path, read_only=True) as conn:
                # 🔧 max_profit_pct는 테이블에 없을 수 있으므로 제외 (캔들에서 직접 계산)
                trades_df = pd.read_sql("""
                    SELECT coin, entry_price, exit_price, entry_timestamp, exit_timestamp,
                           profit_loss_pct, signal_pattern
                    FROM virtual_trade_feedback 
                    WHERE entry_timestamp > 0 AND exit_timestamp > 0
                    ORDER BY exit_timestamp DESC
                    LIMIT 300
                """, conn)
            
            if trades_df.empty:
                return results
            
            # 패턴별 보유 기간과 수익률 분석
            pattern_stats = defaultdict(lambda: {
                'holding_periods': [],
                'profits': [],
                'max_profits': [],
                'optimal_periods': []
            })
            
            # 🚀 [성능] 일괄 캔들 로드 + 메모리 필터링 (개별 쿼리 N회 → 1회)
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as c_conn:
                # 1. 필요한 데이터 범위 계산
                unique_coins = trades_df['coin'].unique().tolist()
                min_ts = int(trades_df['entry_timestamp'].min())
                max_ts = int(trades_df['exit_timestamp'].max()) + 7200  # 매도 후 2시간 추적
                
                # 2. 모든 관련 캔들 일괄 로드
                placeholders = ','.join('?' * len(unique_coins))
                all_candles = pd.read_sql(f"""
                    SELECT symbol, timestamp, high, low, close
                    FROM candles 
                    WHERE symbol IN ({placeholders}) AND interval = '15m'
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY symbol, timestamp ASC
                """, c_conn, params=unique_coins + [min_ts, max_ts])
                
                # 3. 코인별 인덱싱
                candle_cache = {coin: group.set_index('timestamp')
                               for coin, group in all_candles.groupby('symbol', sort=False)}
                
                # 4. 거래 분석 (🚀 iterrows → to_dict)
                for trade in trades_df.to_dict('records'):
                    pattern = trade.get('signal_pattern', 'unknown')
                    if not pattern or pattern == 'unknown':
                        continue
                    
                    coin = trade['coin']
                    entry_ts = int(trade['entry_timestamp'])
                    exit_ts = int(trade['exit_timestamp'])
                    entry_price = float(trade['entry_price'])
                    actual_profit = float(trade.get('profit_loss_pct', 0) or 0)
                    
                    if coin not in candle_cache:
                        continue
                    
                    # 실제 보유 기간 (시간)
                    actual_holding_hours = (exit_ts - entry_ts) / 3600
                    
                    # 메모리에서 캔들 필터링
                    coin_candles = candle_cache[coin]
                    candles = coin_candles[(coin_candles.index >= entry_ts) & (coin_candles.index <= exit_ts + 7200)]
                    
                    if candles.empty or len(candles) < 2:
                        continue
                    
                    # 🚀 [성능] 벡터 연산으로 최적 매도 시점 찾기 (iterrows 제거)
                    profits_series = ((candles['high'] - entry_price) / entry_price) * 100
                    max_profit_idx = profits_series.idxmax()
                    max_profit_pct = profits_series.max()
                    max_profit_time = max_profit_idx  # 인덱스가 timestamp
                    
                    optimal_holding_hours = (max_profit_time - entry_ts) / 3600
                    
                    # 기본 패턴 (첫 단어만 사용)
                    base_pattern = pattern.split('_')[0] if '_' in pattern else pattern
                    
                    pattern_stats[base_pattern]['holding_periods'].append(actual_holding_hours)
                    pattern_stats[base_pattern]['profits'].append(actual_profit)
                    pattern_stats[base_pattern]['max_profits'].append(max_profit_pct)
                    pattern_stats[base_pattern]['optimal_periods'].append(optimal_holding_hours)
            
            # 패턴별 최적 보유 기간 계산
            recommendations = {}
            for pattern, stats in pattern_stats.items():
                if len(stats['holding_periods']) < 5:
                    continue
                
                results['patterns_analyzed'] += 1
                
                avg_actual = sum(stats['holding_periods']) / len(stats['holding_periods'])
                avg_optimal = sum(stats['optimal_periods']) / len(stats['optimal_periods'])
                avg_profit = sum(stats['profits']) / len(stats['profits'])
                avg_max_profit = sum(stats['max_profits']) / len(stats['max_profits'])
                
                # 최적 보유 기간 대비 실제 보유 기간 차이
                timing_gap = avg_actual - avg_optimal
                missed_profit = avg_max_profit - avg_profit
                
                recommendations[pattern] = {
                    'avg_holding_hours': round(avg_actual, 1),
                    'optimal_holding_hours': round(avg_optimal, 1),
                    'timing_gap_hours': round(timing_gap, 1),
                    'avg_profit_pct': round(avg_profit, 2),
                    'potential_profit_pct': round(avg_max_profit, 2),
                    'missed_profit_pct': round(missed_profit, 2),
                    'sample_count': len(stats['holding_periods'])
                }
                
                # Thompson Sampling에 학습
                # 너무 오래 들고 있었으면 (timing_gap > 2시간) 패턴 수정
                if timing_gap > 2:
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_holding_too_long",
                        success=False,
                        profit_pct=-missed_profit,
                        weight=1.2
                    )
                elif timing_gap < -1:  # 너무 일찍 팔았으면
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_holding_too_short",
                        success=False,
                        profit_pct=-missed_profit,
                        weight=1.2
                    )
                else:  # 적절한 타이밍
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_holding_optimal",
                        success=True,
                        profit_pct=avg_profit,
                        weight=1.0
                    )
            
            results['recommendations'] = recommendations
            return results
            
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 보유 기간 최적화 분석 오류: {e}")
            return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 [3] 손절/익절 임계값 동적 학습
    # ═══════════════════════════════════════════════════════════════════════════
    def _learn_dynamic_stop_take_profit(self) -> Dict[str, Any]:
        """패턴별 최적 손절/익절 라인 학습"""
        results = {'patterns_analyzed': 0, 'stop_loss_adjustments': {}, 'take_profit_adjustments': {}}
        
        try:
            from trade.core.database import CANDLES_DB_PATH
            
            with get_db_connection(self.db_path, read_only=True) as conn:
                trades_df = pd.read_sql("""
                    SELECT coin, entry_price, exit_price, entry_timestamp, exit_timestamp,
                           profit_loss_pct, signal_pattern
                    FROM virtual_trade_feedback 
                    WHERE entry_timestamp > 0 AND exit_timestamp > 0
                    ORDER BY exit_timestamp DESC
                    LIMIT 300
                """, conn)
            
            if trades_df.empty:
                return results
            
            # 패턴별 MFE(최대 유리 변동)/MAE(최대 불리 변동) 수집
            pattern_extremes = defaultdict(lambda: {
                'mfe_list': [],  # Maximum Favorable Excursion
                'mae_list': [],  # Maximum Adverse Excursion
                'final_profits': [],
                'stopped_out': 0,  # 손절로 끝난 횟수
                'took_profit': 0   # 익절로 끝난 횟수
            })
            
            # 🚀 [성능] 일괄 캔들 로드 (개별 쿼리 N회 → 1회)
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as c_conn:
                # 1. 필요한 데이터 범위 계산
                unique_coins = trades_df['coin'].unique().tolist()
                min_ts = int(trades_df['entry_timestamp'].min())
                max_ts = int(trades_df['exit_timestamp'].max())
                
                # 2. 일괄 로드
                placeholders = ','.join('?' * len(unique_coins))
                all_candles = pd.read_sql(f"""
                    SELECT symbol, timestamp, high, low
                    FROM candles 
                    WHERE symbol IN ({placeholders}) AND interval = '15m'
                    AND timestamp >= ? AND timestamp <= ?
                """, c_conn, params=unique_coins + [min_ts, max_ts])
                
                # 3. 코인별 인덱싱
                candle_cache = {coin: group.set_index('timestamp')
                               for coin, group in all_candles.groupby('symbol', sort=False)}
                
                # 4. 거래 분석 (🚀 iterrows → to_dict)
                for trade in trades_df.to_dict('records'):
                    pattern = trade.get('signal_pattern', 'unknown')
                    if not pattern or pattern == 'unknown':
                        continue
                    
                    coin = trade['coin']
                    entry_ts = int(trade['entry_timestamp'])
                    exit_ts = int(trade['exit_timestamp'])
                    entry_price = float(trade['entry_price'])
                    final_profit = float(trade.get('profit_loss_pct', 0) or 0)
                    
                    if coin not in candle_cache:
                        continue
                    
                    # 메모리에서 필터링
                    coin_candles = candle_cache[coin]
                    candles = coin_candles[(coin_candles.index >= entry_ts) & (coin_candles.index <= exit_ts)]
                    
                    if candles.empty:
                        continue
                    
                    # MFE/MAE 계산 (벡터 연산)
                    max_high = candles['high'].max()
                    min_low = candles['low'].min()
                    
                    mfe = ((max_high - entry_price) / entry_price) * 100  # 최대 수익
                    mae = ((entry_price - min_low) / entry_price) * 100   # 최대 손실 (양수로 표현)
                    
                    base_pattern = pattern.split('_')[0] if '_' in pattern else pattern
                    
                    pattern_extremes[base_pattern]['mfe_list'].append(mfe)
                    pattern_extremes[base_pattern]['mae_list'].append(mae)
                    pattern_extremes[base_pattern]['final_profits'].append(final_profit)
                    
                    if final_profit < -2:  # 2% 이상 손실
                        pattern_extremes[base_pattern]['stopped_out'] += 1
                    elif final_profit > 3:  # 3% 이상 이익
                        pattern_extremes[base_pattern]['took_profit'] += 1
            
            # 패턴별 최적 손절/익절 라인 계산
            stop_loss_adj = {}
            take_profit_adj = {}
            
            for pattern, extremes in pattern_extremes.items():
                if len(extremes['mfe_list']) < 5:
                    continue
                
                results['patterns_analyzed'] += 1
                
                # 통계 계산
                avg_mfe = sum(extremes['mfe_list']) / len(extremes['mfe_list'])
                avg_mae = sum(extremes['mae_list']) / len(extremes['mae_list'])
                avg_profit = sum(extremes['final_profits']) / len(extremes['final_profits'])
                
                # 75백분위수 MAE = 대부분의 거래가 이 범위 내에서 손실
                sorted_mae = sorted(extremes['mae_list'])
                mae_75pct = sorted_mae[int(len(sorted_mae) * 0.75)]
                
                # 50백분위수 MFE = 절반의 거래가 이 수익에 도달
                sorted_mfe = sorted(extremes['mfe_list'])
                mfe_50pct = sorted_mfe[int(len(sorted_mfe) * 0.5)]
                
                # 최적 손절선: 75백분위 MAE + 약간의 여유 (너무 타이트하면 손절이 잦음)
                optimal_stop_loss = -(mae_75pct + 0.5)
                
                # 최적 익절선: 50백분위 MFE (절반 이상이 도달하는 수익)
                optimal_take_profit = mfe_50pct * 0.9  # 90%만 목표 (확실한 익절)
                
                stop_loss_adj[pattern] = {
                    'current_default': -3.0,  # 현재 기본 손절선
                    'optimal': round(optimal_stop_loss, 2),
                    'avg_mae': round(avg_mae, 2),
                    'mae_75pct': round(mae_75pct, 2),
                    'stop_out_rate': extremes['stopped_out'] / len(extremes['mfe_list'])
                }
                
                take_profit_adj[pattern] = {
                    'current_default': 5.0,  # 현재 기본 익절선
                    'optimal': round(optimal_take_profit, 2),
                    'avg_mfe': round(avg_mfe, 2),
                    'mfe_50pct': round(mfe_50pct, 2),
                    'take_profit_rate': extremes['took_profit'] / len(extremes['mfe_list'])
                }
                
                # Thompson Sampling에 학습
                # 손절이 너무 잦은 패턴
                if extremes['stopped_out'] / len(extremes['mfe_list']) > 0.4:
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_stop_loss_too_tight",
                        success=False,
                        profit_pct=avg_profit,
                        weight=1.3
                    )
                
                # 익절을 잘 못하는 패턴 (MFE 대비 실현 수익이 낮음)
                if avg_mfe > 0 and avg_profit < avg_mfe * 0.3:
                    self.thompson_sampler.update_distribution(
                        pattern=f"{pattern}_take_profit_missed",
                        success=False,
                        profit_pct=avg_profit - avg_mfe,
                        weight=1.3
                    )
            
            results['stop_loss_adjustments'] = stop_loss_adj
            results['take_profit_adjustments'] = take_profit_adj
            
            # 글로벌 DB에 최적 임계값 저장
            global_db = os.environ.get('GLOBAL_STRATEGY_DB_PATH')
            if global_db and (stop_loss_adj or take_profit_adj):
                try:
                    with sqlite3.connect(global_db) as conn:
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS optimal_thresholds (
                                pattern TEXT PRIMARY KEY,
                                optimal_stop_loss REAL,
                                optimal_take_profit REAL,
                                avg_mae REAL,
                                avg_mfe REAL,
                                sample_count INTEGER,
                                last_updated INTEGER
                            )
                        """)
                        
                        for pattern in stop_loss_adj:
                            conn.execute("""
                                INSERT OR REPLACE INTO optimal_thresholds 
                                (pattern, optimal_stop_loss, optimal_take_profit, avg_mae, avg_mfe, sample_count, last_updated)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                pattern,
                                stop_loss_adj[pattern]['optimal'],
                                take_profit_adj.get(pattern, {}).get('optimal', 5.0),
                                stop_loss_adj[pattern]['avg_mae'],
                                take_profit_adj.get(pattern, {}).get('avg_mfe', 0),
                                len(pattern_extremes[pattern]['mfe_list']),
                                int(time.time())
                            ))
                        conn.commit()
                except Exception as db_err:
                    print(f"⚠️ 최적 임계값 저장 오류: {db_err}")
            
            return results
            
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 손절/익절 학습 오류: {e}")
            return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 [4] 연속 손실 패턴 분석
    # ═══════════════════════════════════════════════════════════════════════════
    def _learn_consecutive_loss_patterns(self) -> Dict[str, Any]:
        """연속 손실 시 공통점 분석 및 회피 학습"""
        results = {
            'total_streaks': 0,
            'max_streak': 0,
            'common_factors': {},
            'recommendations': []
        }
        
        try:
            with get_db_connection(self.db_path, read_only=True) as conn:
                # 🆕 최근 7일 데이터만 분석 (오래된 불완전한 데이터 제외)
                recent_cutoff = int(time.time()) - (7 * 24 * 3600)
                trades_df = pd.read_sql("""
                    SELECT coin, entry_timestamp, exit_timestamp, profit_loss_pct, 
                           signal_pattern, market_conditions
                    FROM virtual_trade_feedback 
                    WHERE exit_timestamp > ? AND market_conditions IS NOT NULL AND market_conditions != ''
                    ORDER BY exit_timestamp ASC
                    LIMIT 500
                """, conn, params=(recent_cutoff,))
            
            if trades_df.empty or len(trades_df) < 10:
                return results
            
            # 연속 손실 스트릭 찾기
            streaks = []
            current_streak = []
            
            for _, trade in trades_df.iterrows():
                profit = float(trade.get('profit_loss_pct', 0) or 0)
                
                if profit < 0:  # 손실
                    current_streak.append(trade)
                else:
                    if len(current_streak) >= 3:  # 3연패 이상만 분석
                        streaks.append(current_streak.copy())
                    current_streak = []
            
            # 마지막 스트릭 처리
            if len(current_streak) >= 3:
                streaks.append(current_streak)
            
            if not streaks:
                return results
            
            results['total_streaks'] = len(streaks)
            results['max_streak'] = max(len(s) for s in streaks)
            
            # 연속 손실 시 공통 요인 분석
            common_factors = {
                'patterns': defaultdict(int),
                'coins': defaultdict(int),
                'market_regimes': defaultdict(int),
                'time_of_day': defaultdict(int),
                'total_loss_pct': 0,
                'avg_loss_per_trade': 0
            }
            
            total_trades_in_streaks = 0
            
            for streak in streaks:
                for trade in streak:
                    total_trades_in_streaks += 1
                    
                    # 패턴 집계
                    pattern = trade.get('signal_pattern', 'unknown')
                    base_pattern = pattern.split('_')[0] if pattern and '_' in pattern else (pattern or 'unknown')
                    common_factors['patterns'][base_pattern] += 1
                    
                    # 코인 집계
                    common_factors['coins'][trade['coin']] += 1
                    
                    # 시장 상황 집계
                    market_cond = {}
                    if trade.get('market_conditions'):
                        try:
                            market_cond = json.loads(trade['market_conditions'])
                        except:
                            pass
                    regime = market_cond.get('regime', 'unknown')
                    common_factors['market_regimes'][regime] += 1
                    
                    # 시간대 집계
                    entry_ts = int(trade.get('entry_timestamp', 0))
                    if entry_ts > 0:
                        hour = datetime.fromtimestamp(entry_ts).hour
                        time_slot = f"{(hour // 4) * 4:02d}-{(hour // 4) * 4 + 4:02d}시"
                        common_factors['time_of_day'][time_slot] += 1
                    
                    # 손실 합계
                    common_factors['total_loss_pct'] += float(trade.get('profit_loss_pct', 0) or 0)
            
            if total_trades_in_streaks > 0:
                common_factors['avg_loss_per_trade'] = common_factors['total_loss_pct'] / total_trades_in_streaks
            
            # 가장 빈번한 요인 찾기
            recommendations = []
            
            # 위험한 패턴
            if common_factors['patterns']:
                worst_pattern = max(common_factors['patterns'].items(), key=lambda x: x[1])
                if worst_pattern[1] >= 3:
                    recommendations.append(f"⚠️ '{worst_pattern[0]}' 패턴에서 {worst_pattern[1]}회 연속 손실 발생 - 주의 필요")
                    self.thompson_sampler.update_distribution(
                        pattern=f"{worst_pattern[0]}_consecutive_loss",
                        success=False,
                        profit_pct=common_factors['avg_loss_per_trade'],
                        weight=2.0  # 높은 가중치
                    )
            
            # 위험한 시장 상황
            if common_factors['market_regimes']:
                worst_regime = max(common_factors['market_regimes'].items(), key=lambda x: x[1])
                if worst_regime[1] >= 3:
                    recommendations.append(f"⚠️ '{worst_regime[0]}' 시장에서 {worst_regime[1]}회 연속 손실 - 매매 자제 권장")
                    self.thompson_sampler.update_distribution(
                        pattern=f"regime_{worst_regime[0]}_danger",
                        success=False,
                        profit_pct=common_factors['avg_loss_per_trade'],
                        weight=1.5
                    )
                    
                    # 🆕 레짐+패턴 조합 위험 학습
                    if common_factors['patterns']:
                        worst_pattern = max(common_factors['patterns'].items(), key=lambda x: x[1])
                        if worst_pattern[1] >= 2:
                            danger_combo = f"{worst_pattern[0]}_{worst_regime[0]}"
                            recommendations.append(f"   ⛔ 특히 '{danger_combo}' 조합 주의 (레짐+패턴)")
                            self.thompson_sampler.update_distribution(
                                pattern=f"{danger_combo}_consecutive_loss",
                                success=False,
                                profit_pct=common_factors['avg_loss_per_trade'],
                                weight=2.0  # 높은 가중치로 강력 학습
                            )
            
            # 위험한 시간대
            if common_factors['time_of_day']:
                worst_time = max(common_factors['time_of_day'].items(), key=lambda x: x[1])
                if worst_time[1] >= 3:
                    recommendations.append(f"⚠️ {worst_time[0]} 시간대에 {worst_time[1]}회 연속 손실 - 해당 시간대 주의")
            
            # 3연패 이상 발생 시 휴식 권장
            if results['max_streak'] >= 5:
                recommendations.append(f"🛑 최대 {results['max_streak']}연패 기록 - 연속 손실 시 매매 일시 중단 권장")
            
            # Dict 변환 (defaultdict -> dict)
            results['common_factors'] = {
                'patterns': dict(common_factors['patterns']),
                'coins': dict(common_factors['coins']),
                'market_regimes': dict(common_factors['market_regimes']),
                'time_of_day': dict(common_factors['time_of_day']),
                'total_loss_pct': round(common_factors['total_loss_pct'], 2),
                'avg_loss_per_trade': round(common_factors['avg_loss_per_trade'], 2)
            }
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 연속 손실 패턴 분석 오류: {e}")
            return results

    def run_full_learning(self):
        """가상매매 결과 학습 및 자가진단 일괄 실행
        
        Note: 시그널 예측 검증은 strategy_signal_generator.py에서 처리
              (validate_signals_incremental 함수)
        
        🆕 균형 학습 시스템 (손실 분석 + 과잉 회피 방지):
        - 손실 원인별 분석: 진입/청산 타이밍, 전략-레짐 부조화, 시장 급변, 보유 초과
        - 과잉 회피 방지: 시간 감쇠, 가중치 상한선, 최소 매수 확률 보장
        """
        print("\n📖 가상매매 결과 학습 및 자가진단 시작...")
        print(f"   ⚖️ 균형 학습 활성화: 최소 매수확률 {BalancedLearningGuard.MIN_BUY_PROBABILITY*100:.0f}%, 손실 가중치 상한 {BalancedLearningGuard.MAX_LOSS_WEIGHT}x, 손실 분석 기준 {BalancedLearningGuard.MIN_LOSS_PCT_FOR_ANALYSIS}%↑")
        
        # 0. 시장 레짐 분석
        try:
            market_context = get_market_context()
            print(f"📊 시장 상태: [추세] {market_context.get('regime', 'neutral').upper()} | [확산] {market_context.get('breadth', 'neutral').upper()}")
        except Exception as e:
            print(f"⚠️ 시장 레짐 분석 오류: {e}")

        # 1. 실시간 학습 (가상매매 결과 기반)
        total_new = 0
        while True:
            new_count = self._execute_real_time_learning()
            if new_count == 0: break
            total_new += new_count
            
        # 2. 알파 가디언 자가 진단
        try:
            from trade.core.decision import get_ai_decision_engine
            guardian = get_ai_decision_engine(db_path=STRATEGY_DB_PATH)
            
            with get_db_connection(self.db_path) as conn:
                query = "SELECT * FROM virtual_trade_feedback ORDER BY exit_timestamp DESC LIMIT 100"
                feedback_history = pd.read_sql(query, conn).to_dict('records')
                
                if feedback_history:
                    quality = self.evolution_engine.evaluate_decision_quality(feedback_history, guardian)
                    new_bias = self.evolution_engine.update_meta_bias(quality, guardian)
                    guardian.save_meta_bias(new_bias)
                    
                    print(f"\n🛡️ [알파 가디언 자가진단 리포트]")
                    print(f"   📈 매수 성공률: {quality.get('buy_accuracy', 0):>6.1%} ({quality.get('profit_count', 0)}/{quality.get('buy_count', 0)}건)")
                    print(f"   ✨ 전역 성격 교정 완료: {new_bias.get('buy_threshold_offset', 0):+.2f}")
                else:
                    print(f"\n🛡️ [알파 가디언] 분석할 피드백 데이터가 아직 없습니다.")
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 알파 가디언 자가진단 오류: {e}")

        # 3. 시장 인사이트 복기 (놓친 매매/잘한 관망 학습)
        try:
            print("\n📊 [정밀 분석] 시장 인사이트 복기 시작...")
            self.market_miner.mine_insights()
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 시장 인사이트 분석 오류: {e}")

        # 4. 매도 품질 평가 (MFE/MAE 분석)
        try:
            print("\n📈 [매도 품질 평가] 매도 후 가격 추적 분석 중...")
            completed_evals = self._run_post_trade_evaluation()
            if completed_evals > 0:
                print(f"   ✅ {completed_evals}건의 매도 품질 평가 완료")
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 매도 품질 평가 오류: {e}")

        # 5. 전이 학습 (패턴 지식 공유)
        try:
            print("\n🔄 [전이 학습] 글로벌 패턴 지식 공유 시작...")
            self.transfer_learner.execute_transfer_learning()
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 전이 학습 오류: {e}")

        # ═══════════════════════════════════════════════════════════════════════
        # 🆕 신규 학습 모듈 (진입/보유/손익절/연속손실)
        # ═══════════════════════════════════════════════════════════════════════
        
        # 6. 진입 타이밍 최적화 학습
        try:
            print("\n⏱️ [진입 타이밍 최적화] 더 좋은 진입 기회 분석 중...")
            entry_results = self._learn_entry_timing_optimization()
            if entry_results['analyzed'] > 0:
                better_rate = (entry_results['could_be_better'] / entry_results['analyzed']) * 100
                print(f"   📊 분석 완료: {entry_results['analyzed']}건 중 {entry_results['could_be_better']}건({better_rate:.1f}%)은 더 좋은 가격 있었음")
                if entry_results['avg_missed_pct'] > 0:
                    print(f"   💡 평균 {entry_results['avg_missed_pct']:.2f}% 더 좋은 가격 존재")
                if entry_results['optimal_delay_minutes'] > 0:
                    print(f"   ⏰ 권장 진입 지연: {entry_results['optimal_delay_minutes']}분")
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 진입 타이밍 학습 오류: {e}")

        # 7. 보유 기간 최적화 학습
        try:
            print("\n⏳ [보유 기간 최적화] 패턴별 최적 보유 시간 분석 중...")
            holding_results = self._learn_optimal_holding_period()
            if holding_results['patterns_analyzed'] > 0:
                print(f"   📊 {holding_results['patterns_analyzed']}개 패턴 분석 완료")
                # 상위 3개 패턴만 출력
                for pattern, rec in list(holding_results['recommendations'].items())[:3]:
                    gap = rec['timing_gap_hours']
                    if abs(gap) > 1:
                        direction = "너무 오래" if gap > 0 else "너무 빨리"
                        print(f"   💡 '{pattern}': {direction} 보유 (실제 {rec['avg_holding_hours']:.1f}h vs 최적 {rec['optimal_holding_hours']:.1f}h)")
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 보유 기간 학습 오류: {e}")

        # 8. 손절/익절 임계값 동적 학습
        try:
            print("\n📉 [손절/익절 최적화] 패턴별 최적 임계값 분석 중...")
            threshold_results = self._learn_dynamic_stop_take_profit()
            if threshold_results['patterns_analyzed'] > 0:
                print(f"   📊 {threshold_results['patterns_analyzed']}개 패턴 분석 완료")
                # 주요 조정 필요 패턴 출력
                for pattern, adj in list(threshold_results['stop_loss_adjustments'].items())[:3]:
                    if abs(adj['optimal'] - adj['current_default']) > 1:
                        print(f"   🛑 '{pattern}' 손절: {adj['current_default']}% → {adj['optimal']}% 권장")
                for pattern, adj in list(threshold_results['take_profit_adjustments'].items())[:3]:
                    if abs(adj['optimal'] - adj['current_default']) > 1:
                        print(f"   ✅ '{pattern}' 익절: {adj['current_default']}% → {adj['optimal']}% 권장")
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 손절/익절 학습 오류: {e}")

        # 9. 연속 손실 패턴 분석
        try:
            print("\n🔴 [연속 손실 분석] 연패 패턴 및 회피 전략 분석 중...")
            streak_results = self._learn_consecutive_loss_patterns()
            if streak_results['total_streaks'] > 0:
                print(f"   📊 {streak_results['total_streaks']}회의 3연패 이상 발생 (최대 {streak_results['max_streak']}연패)")
                if streak_results['common_factors'].get('avg_loss_per_trade', 0) < 0:
                    print(f"   💸 연패 시 평균 손실: {streak_results['common_factors']['avg_loss_per_trade']:.2f}%/건")
                for rec in streak_results['recommendations']:
                    print(f"   {rec}")
            else:
                print("   ✅ 최근 3연패 이상 기록 없음 - 양호")
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"⚠️ 연속 손실 분석 오류: {e}")

        # 🧬 10. 전략 진화 상태 출력
        if EVOLUTION_SYSTEM_AVAILABLE:
            try:
                print("\n" + "=" * 60)
                print_evolution_status()
            except Exception as e:
                print(f"⚠️ 진화 상태 출력 오류: {e}")

        print(f"\n✅ 최종 완료: {total_new}건의 새로운 지식 습득 완료")

if __name__ == "__main__":
    learner = VirtualTradingLearner()
    learner.run_full_learning()
