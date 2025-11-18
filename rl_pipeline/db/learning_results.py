"""
Learning Results DB 관리 모듈
이제 rl_strategies.db로 통합됨
"""

import logging
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# DB 경로 - learning_results.db는 이제 rl_strategies.db로 통합됨
# config에서 LEARNING_RESULTS_DB_PATH = STRATEGIES_DB로 설정됨
from rl_pipeline.core.env import config
LEARNING_RESULTS_DB_PATH = config.LEARNING_RESULTS_DB_PATH


@contextmanager
def get_learning_db_connection(db_path: str = None):
    """learning_results.db 연결 관리"""
    if db_path is None:
        db_path = LEARNING_RESULTS_DB_PATH
    
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ rl_strategies.db (learning_results) 연결 실패: {e}")
        raise
    finally:
        if conn:
            conn.close()

def create_learning_results_tables(db_path: str = None) -> bool:
    """rl_strategies.db에 learning_results 테이블 생성 (통합됨)"""
    try:
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Self-play 진화 결과
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS selfplay_evolution_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    
                    -- 진화 전략 정보
                    initial_strategy TEXT NOT NULL,
                    evolved_strategy TEXT NOT NULL,
                    evolution_steps INTEGER DEFAULT 0,
                    
                    -- 진화 성과
                    initial_performance REAL DEFAULT 0.0,
                    evolved_performance REAL DEFAULT 0.0,
                    improvement_rate REAL DEFAULT 0.0,
                    
                    -- 진화 과정
                    evolution_history TEXT DEFAULT '[]',
                    adaptation_patterns TEXT DEFAULT '{}',
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. 레짐 기반 라우팅 결과
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regime_routing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    
                    -- 라우팅된 전략
                    routed_strategy TEXT NOT NULL,
                    routing_confidence REAL DEFAULT 0.0,
                    routing_score REAL DEFAULT 0.0,
                    
                    -- 레짐별 성능
                    regime_performance REAL DEFAULT 0.0,
                    regime_adaptation REAL DEFAULT 0.0,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 3. 통합분석 결과
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS integrated_analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    
                    -- 분석 결과
                    fractal_score REAL DEFAULT 0.0,
                    multi_timeframe_score REAL DEFAULT 0.0,
                    indicator_cross_score REAL DEFAULT 0.0,
                    
                    -- JAX 앙상블 결과
                    ensemble_score REAL DEFAULT 0.0,
                    ensemble_confidence REAL DEFAULT 0.0,
                    
                    -- 최종 시그널 점수
                    final_signal_score REAL DEFAULT 0.0,
                    signal_confidence REAL DEFAULT 0.0,
                    signal_action TEXT DEFAULT 'hold',
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 4. 실시간 학습 피드백
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS realtime_learning_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    signal_id TEXT NOT NULL,
                    
                    -- 시그널 정보
                    signal_score REAL DEFAULT 0.0,
                    signal_action TEXT NOT NULL,
                    signal_timestamp DATETIME NOT NULL,
                    
                    -- 실제 결과
                    actual_profit REAL DEFAULT 0.0,
                    actual_success BOOLEAN DEFAULT FALSE,
                    market_condition TEXT DEFAULT 'unknown',
                    
                    -- 학습 피드백
                    learning_adjustment REAL DEFAULT 0.0,
                    strategy_update TEXT DEFAULT '{}',
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 5. 글로벌 전략 결과
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_strategy_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- 글로벌 성능
                    overall_score REAL DEFAULT 0.0,
                    overall_confidence REAL DEFAULT 0.0,
                    policy_improvement REAL DEFAULT 0.0,
                    convergence_rate REAL DEFAULT 0.0,
                    
                    -- 상위 성능
                    top_performers TEXT DEFAULT '[]',
                    top_coins TEXT DEFAULT '[]',
                    top_intervals TEXT DEFAULT '[]',
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 6. 파이프라인 실행 로그
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    
                    -- 실행 통계
                    strategies_created INTEGER DEFAULT 0,
                    selfplay_episodes INTEGER DEFAULT 0,
                    regime_detected TEXT DEFAULT 'unknown',
                    routing_results INTEGER DEFAULT 0,
                    
                    -- 최종 결과
                    signal_score REAL DEFAULT 0.0,
                    signal_action TEXT DEFAULT 'HOLD',
                    execution_time REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'unknown',
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 🆕 7. 시그널 계산용 전략 요약 테이블 (realtime_signal_selector.py 전용)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_summary_for_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    
                    -- 최상위 전략 요약 정보
                    top_strategy_id TEXT,
                    top_strategy_params TEXT,  -- JSON: {rsi_min, rsi_max, volume_ratio_min, ...}
                    top_profit REAL DEFAULT 0.0,
                    top_win_rate REAL DEFAULT 0.0,
                    top_quality_grade TEXT,  -- S/A/B/C/D/F
                    
                    -- 평균 성능 지표
                    avg_profit REAL DEFAULT 0.0,
                    avg_win_rate REAL DEFAULT 0.0,
                    avg_sharpe_ratio REAL DEFAULT 0.0,
                    avg_calmar_ratio REAL DEFAULT 0.0,
                    avg_profit_factor REAL DEFAULT 0.0,
                    
                    -- 전략 통계
                    total_strategies INTEGER DEFAULT 0,
                    s_grade_count INTEGER DEFAULT 0,
                    a_grade_count INTEGER DEFAULT 0,
                    
                    -- 업데이트 시간
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(coin, interval)
                )
            """)
            
            # 🆕 8. 시그널 계산용 DNA 요약 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dna_summary_for_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT,
                    interval TEXT,
                    
                    -- DNA 요약 정보 (실시간 시그널 계산용)
                    profitability_score REAL DEFAULT 0.0,  -- win_rate mean
                    stability_score REAL DEFAULT 0.0,      -- trades_count 기반
                    scalability_score REAL DEFAULT 0.0,    -- complexity_score mean
                    dna_quality REAL DEFAULT 0.0,          -- total_strategies 기반
                    
                    -- DNA 패턴 요약
                    rsi_pattern TEXT,  -- 최빈 RSI 패턴
                    macd_pattern TEXT,  -- 최빈 MACD 패턴
                    volume_pattern TEXT,  -- 최빈 Volume 패턴
                    
                    -- DNA 히스토리 요약
                    dna_momentum REAL DEFAULT 0.0,  -- 최근 변화율
                    dna_stability REAL DEFAULT 0.0,  -- 안정성 점수
                    
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(coin, interval)
                )
            """)
            
            # 🆕 9. 시그널 계산용 글로벌 전략 요약 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_strategy_summary_for_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interval TEXT NOT NULL,
                    
                    -- 최상위 글로벌 전략 요약
                    top_global_strategy_id TEXT,
                    top_global_strategy_params TEXT,  -- JSON
                    top_global_score REAL DEFAULT 0.0,
                    
                    -- 평균 성능
                    avg_global_score REAL DEFAULT 0.0,
                    avg_global_confidence REAL DEFAULT 0.0,
                    
                    -- 통계
                    total_global_strategies INTEGER DEFAULT 0,
                    
                    -- 학습 품질 지표
                    learning_quality_score REAL DEFAULT 0.0,
                    reliability_score REAL DEFAULT 0.0,
                    
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(interval)
                )
            """)
            
            # 🆕 10. 시그널 계산용 프랙탈/시너지 요약 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_summary_for_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    
                    -- 프랙탈 분석 요약
                    fractal_score REAL DEFAULT 0.0,
                    fractal_pattern TEXT,  -- JSON
                    
                    -- 시너지 분석 요약
                    synergy_score REAL DEFAULT 0.0,
                    synergy_patterns TEXT,  -- JSON
                    
                    -- 최적 조건
                    optimal_rsi_min REAL DEFAULT 30.0,
                    optimal_rsi_max REAL DEFAULT 70.0,
                    optimal_volume_ratio REAL DEFAULT 1.0,
                    
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(coin, interval)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_selfplay_coin_interval ON selfplay_evolution_results(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_routing_coin_interval ON regime_routing_results(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_integrated_analysis_coin_interval ON integrated_analysis_results(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_realtime_feedback_coin_interval ON realtime_learning_feedback(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_logs_coin_interval ON pipeline_execution_logs(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_summary_coin_interval ON strategy_summary_for_signals(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dna_summary_coin_interval ON dna_summary_for_signals(coin, interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_strategy_summary_interval ON global_strategy_summary_for_signals(interval)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_summary_coin_interval ON analysis_summary_for_signals(coin, interval)")
            
            conn.commit()
            logger.info("✅ rl_strategies.db learning_results 테이블 생성 완료")
            return True
            
    except Exception as e:
        logger.error(f"❌ rl_strategies.db learning_results 테이블 생성 실패: {e}")
        return False

def save_selfplay_results(coin: str, interval: str, selfplay_result: Dict[str, Any], db_path: str = None) -> bool:
    """Self-play 결과를 rl_strategies.db에 저장"""
    try:
        import json
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        # 🔥 원본 summary 보존 (온라인 결과로 덮어써지지 않도록)
        original_summary = selfplay_result.get("summary", {})
        summary = original_summary.copy() if original_summary else {}
        cycle_results = selfplay_result.get("cycle_results", [])
        
        # 🔥 rl_strategies.db에 selfplay_results 테이블 생성 및 저장
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # selfplay_results 테이블이 없으면 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS selfplay_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    episodes INTEGER NOT NULL,
                    results TEXT NOT NULL,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(coin, interval, episodes, results)
                )
            """)
            
            saved_count = 0
            
            # 🔥 cycle_results가 없으면 traditional_result에서 가져오기 (dual mode 대응)
            if not cycle_results and selfplay_result.get('dual_mode'):
                traditional_result = selfplay_result.get('traditional_result')
                if traditional_result:
                    cycle_results = traditional_result.get("cycle_results", [])
            
            # 🔥 온라인 Self-play 결과 처리 추가 (온라인 결과가 아직 변환되지 않은 경우)
            online_summary = None  # 온라인 summary 별도 저장 (원본 summary 보존)
            if not cycle_results:
                try:
                    from rl_pipeline.hybrid.online_data_converter import (
                        extract_online_selfplay_result,
                        convert_online_segments_to_cycle_results
                    )
                    
                    online_segments = extract_online_selfplay_result(selfplay_result)
                    if online_segments:
                        cycle_results = convert_online_segments_to_cycle_results(online_segments, summary)
                        logger.debug(f"✅ 온라인 Self-play 결과 변환 완료 ({len(cycle_results)}개 cycle)")
                    # online_result에 직접 있는 경우
                    elif selfplay_result.get('online_result'):
                        online_result = selfplay_result.get('online_result', {})
                        online_segments = online_result.get('segment_results', [])
                        if online_segments:
                            online_summary = online_result.get('summary', {})  # 별도 저장
                            cycle_results = convert_online_segments_to_cycle_results(online_segments, online_summary)
                            logger.debug(f"✅ 온라인 Self-play 결과 변환 완료 (online_result에서) ({len(cycle_results)}개 cycle)")
                except ImportError:
                    logger.debug(f"⚠️ 온라인 데이터 변환 모듈 없음 (무시)")
                except Exception as e:
                    logger.debug(f"⚠️ 온라인 Self-play 결과 변환 실패: {e}")
            
            # 🔥 summary가 비어있거나 값이 0.0이면 cycle_results에서 직접 계산
            if cycle_results and (not summary or 
                summary.get("avg_win_rate", 0.0) == 0.0 and summary.get("avg_pnl", 0.0) == 0.0):
                try:
                    import numpy as np
                    all_performances = []
                    for result in cycle_results:
                        if "results" in result:
                            for agent_id, performance in result["results"].items():
                                all_performances.append(performance)
                    
                    if all_performances:
                        calculated_summary = {
                            "total_episodes": len(cycle_results),
                            "total_trades": sum(p.get("total_trades", 0) for p in all_performances),
                            "avg_win_rate": float(np.mean([p.get("win_rate", 0) for p in all_performances])),
                            "avg_pnl": float(np.mean([p.get("total_pnl", 0) for p in all_performances])),
                            "avg_sharpe_ratio": float(np.mean([p.get("sharpe_ratio", 0) for p in all_performances])),
                        }
                        # 기존 summary와 병합 (계산된 값 우선)
                        summary.update(calculated_summary)
                        logger.debug(f"✅ cycle_results에서 summary 계산 완료: win_rate={summary.get('avg_win_rate', 0):.2%}, pnl={summary.get('avg_pnl', 0):.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ cycle_results에서 summary 계산 실패: {e}")
            
            if not cycle_results:
                logger.warning(f"⚠️ Self-play 결과 저장: cycle_results가 없습니다. (dual_mode={selfplay_result.get('dual_mode', False)})")
                logger.info(f"✅ Self-play 결과 저장 완료 (rl_strategies.db): {coin}-{interval}, {saved_count}개")
                return False
            
            for cycle in cycle_results:
                episode = cycle.get("episode", 0)
                results = cycle.get("results", {})
                
                if not results:
                    continue
                
                for agent_id, performance in results.items():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO selfplay_results 
                            (coin, interval, episodes, results, summary, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            coin,
                            interval,
                            episode,
                            json.dumps({
                                "agent_id": agent_id,
                                "performance": performance
                            }),
                            json.dumps({
                                "total_episodes": len(cycle_results),
                                "episode": episode,
                                "total_trades": summary.get("total_trades", 0),  # 🔥 추가
                                "avg_win_rate": summary.get("avg_win_rate", 0.0),
                                "avg_pnl": summary.get("avg_pnl", 0.0),  # 🔥 수정 (avg_total_return → avg_pnl)
                                "avg_sharpe_ratio": summary.get("avg_sharpe_ratio", 0.0),  # 🔥 추가
                                "avg_total_return": summary.get("avg_total_return", summary.get("avg_pnl", 0.0)),  # 호환성 유지 (avg_pnl로 폴백)
                                "regime_performance": summary.get("regime_performance", {}),  # 🔥 추가 (선택)
                                "learning_progress": selfplay_result.get("learning_progress", {})
                            }),
                            datetime.now().isoformat()
                        ))
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Self-play 결과 일부 저장 실패: {e}")
                        continue
            
            conn.commit()
            logger.info(f"✅ Self-play 결과 저장 완료 (rl_strategies.db): {coin}-{interval}, {saved_count}개")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Self-play 결과 저장 실패: {e}")
        return False

def save_pipeline_execution_log(coin: str, interval: str, strategies_created: int,
                               selfplay_episodes: int, regime_detected: str,
                               routing_results: int, signal_score: float,
                               signal_action: str, execution_time: float,
                               status: str, db_path: str = None) -> bool:
    """파이프라인 실행 로그 저장"""
    try:
        # 음수 execution_time 방지
        if execution_time < 0:
            logger.warning(f"⚠️ 음수 execution_time 감지: {execution_time:.2f}초 → 0.0초로 변경 ({coin}-{interval})")
            execution_time = 0.0

        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO pipeline_execution_logs
                (coin, interval, strategies_created, selfplay_episodes, regime_detected,
                 routing_results, signal_score, signal_action, execution_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                coin, interval, strategies_created, selfplay_episodes, regime_detected,
                routing_results, signal_score, signal_action, execution_time, status
            ))
            
            conn.commit()
            logger.info(f"✅ 파이프라인 실행 로그 저장 완료: {coin}-{interval}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 파이프라인 실행 로그 저장 실패: {e}")
        return False

def save_regime_routing_results(coin: str, interval: str, routing_results: List[Any]) -> bool:
    """레짐 라우팅 결과를 rl_strategies.db에 저장"""
    try:
        from rl_pipeline.routing.regime_router import RegimeRoutingResult
        import json
        
        if not routing_results:
            logger.debug(f"레짐 라우팅 결과가 비어있어 저장 건너뜀: {coin}-{interval}")
            return True
        
        with get_learning_db_connection(LEARNING_RESULTS_DB_PATH) as conn:
            cursor = conn.cursor()
            
            saved_count = 0
            for result in routing_results:
                try:
                    # RegimeRoutingResult 객체인지 확인
                    if hasattr(result, 'routed_strategy'):
                        # 객체인 경우
                        routed_strategy_json = json.dumps(result.routed_strategy)
                        cursor.execute("""
                            INSERT INTO regime_routing_results 
                            (coin, interval, regime, routed_strategy, routing_confidence, 
                             routing_score, regime_performance, regime_adaptation, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            result.coin,
                            result.interval,
                            result.regime,
                            routed_strategy_json,
                            result.routing_confidence,
                            result.routing_score,
                            result.regime_performance,
                            result.regime_adaptation,
                            result.created_at
                        ))
                        saved_count += 1
                    elif isinstance(result, dict):
                        # 딕셔너리인 경우 (대체 처리)
                        routed_strategy_json = json.dumps(result.get('routed_strategy', result))
                        cursor.execute("""
                            INSERT INTO regime_routing_results 
                            (coin, interval, regime, routed_strategy, routing_confidence, 
                             routing_score, regime_performance, regime_adaptation, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            result.get('coin', coin),
                            result.get('interval', interval),
                            result.get('regime', 'neutral'),
                            routed_strategy_json,
                            result.get('routing_confidence', 0.5),
                            result.get('routing_score', 0.5),
                            result.get('regime_performance', 0.5),
                            result.get('regime_adaptation', 0.5),
                            result.get('created_at', datetime.now().isoformat())
                        ))
                        saved_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ 레짐 라우팅 결과 일부 저장 실패: {e}")
                    continue
            
            conn.commit()
            logger.info(f"✅ 레짐 라우팅 결과 저장 완료: {coin}-{interval}, {saved_count}개")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 레짐 라우팅 결과 저장 실패: {e}")
        return False

def save_regime_routing_to_rl_episodes(coin: str, interval: str, routing_results: List[Any]) -> bool:
    """
    🔥 레짐 라우팅 백테스트 결과를 rl_episodes 테이블에 저장
    Self-play 없이도 예측 정확도를 수집할 수 있도록 함
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        routing_results: 레짐 라우팅 결과 리스트
    
    Returns:
        성공 여부
    """
    try:
        from rl_pipeline.routing.regime_router import RegimeRoutingResult
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        import uuid
        import hashlib
        
        if not routing_results:
            logger.debug(f"레짐 라우팅 결과가 비어있어 rl_episodes 저장 건너뜀: {coin}-{interval}")
            return True
        
        saved_count = 0
        timestamp = int(datetime.now().timestamp())
        
        for result in routing_results:
            try:
                # RegimeRoutingResult 객체인지 확인
                if not hasattr(result, 'routed_strategy'):
                    continue
                
                strategy = result.routed_strategy
                strategy_id = strategy.get('id') or strategy.get('strategy_id') or 'unknown'
                predictive_accuracy = getattr(result, 'predictive_accuracy', 0.0)
                backtest_result = getattr(result, 'backtest_result', None)
                
                # 🔥 저장 조건 완화: 백테스트 결과가 없어도 기본 정보는 저장
                # (예측 정확도가 0이어도 시장 상태 정보는 유용)
                if not backtest_result:
                    # 백테스트 결과가 없으면 기본값으로 저장 (최소한의 데이터 수집)
                    backtest_result = {
                        'trades': 0,
                        'profit': 0.0,
                        'wins': 0,
                        'win_rate': 0.0,
                        'predictive_accuracy': 0.0,
                        'data_points': 0
                    }
                
                # 백테스트 결과에서 거래 정보 추출
                trades = backtest_result.get('trades', 0)
                
                # 🔥 거래가 0회여도 저장 (시장 상태 정보는 유용)
                # 예측 정확도가 0이어도 저장 (나중에 Paper Trading에서 업데이트 가능)
                
                # 🔥 각 거래를 에피소드로 저장
                # 간단한 방식: 백테스트 결과를 하나의 에피소드로 저장
                # episode_id 생성 (고유성 보장)
                episode_id = f"regime_routing_{coin}_{interval}_{strategy_id}_{timestamp}_{saved_count}"
                
                # 예측 방향 결정 (백테스트에서 매수 신호 = 상승 예측)
                predicted_dir = 1  # 상승 예측 (매수 신호)
                predicted_conf = min(predictive_accuracy, 1.0)  # 예측 정확도를 확신도로 사용
                
                # 전략 파라미터에서 목표 변동률 추정
                target_move_pct = strategy.get('take_profit', 0.05)  # 기본값 5%
                horizon_k = strategy.get('max_hold_periods', 20)  # 기본값 20 캔들
                
                # state_key 생성 (레짐 기반)
                regime = result.regime
                state_key = f"{regime}_{strategy_id}"
                
                # 진입 가격 추정 (백테스트 결과에서)
                entry_price = 1.0  # 정규화된 가격 (백테스트에서는 상대적)
                
                # 🔥 rl_episodes에 저장 (strategies DB 사용)
                try:
                    with get_optimized_db_connection("strategies") as strategies_conn:
                        cursor = strategies_conn.cursor()
                        
                        # rl_episodes 테이블에 저장
                        cursor.execute("""
                            INSERT OR REPLACE INTO rl_episodes (
                                episode_id, ts_entry, coin, interval, strategy_id, state_key,
                                predicted_dir, predicted_conf, entry_price, target_move_pct, horizon_k
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            episode_id, timestamp, coin, interval, strategy_id, state_key,
                            predicted_dir, predicted_conf, entry_price, target_move_pct, horizon_k
                        ))
                        
                        # rl_episode_summary 테이블에 저장
                        total_profit = backtest_result.get('profit', 0.0)
                        win_rate = backtest_result.get('win_rate', 0.0)
                        realized_ret_signed = total_profit / trades if trades > 0 else 0.0
                        acc_flag = 1 if predictive_accuracy >= 0.5 else 0
                        ts_exit = timestamp + (horizon_k * 900)  # 대략적인 종료 시간
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO rl_episode_summary (
                                episode_id, ts_exit, first_event, t_hit,
                                realized_ret_signed, total_reward, acc_flag,
                                coin, interval, strategy_id, source_type
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            episode_id, ts_exit, 'expiry', horizon_k,
                            realized_ret_signed, predictive_accuracy, acc_flag,
                            coin, interval, strategy_id, 'regime_routing'
                        ))
                        
                        strategies_conn.commit()
                        logger.debug(f"✅ rl_episodes 저장: {episode_id}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ rl_episodes 저장 실패 (무시): {e}")
                    continue
                
                saved_count += 1
                
            except Exception as e:
                logger.warning(f"⚠️ 레짐 라우팅 결과 rl_episodes 저장 실패: {e}")
                continue
        
        if saved_count > 0:
            logger.info(f"✅ 레짐 라우팅 결과 rl_episodes 저장 완료: {coin}-{interval}, {saved_count}개 에피소드")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 레짐 라우팅 결과 rl_episodes 저장 실패: {e}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return False

def save_integrated_analysis_results(coin: str, interval: str, regime: str, analysis_result: Any) -> bool:
    """통합 분석 결과를 rl_strategies.db에 저장 (integrated_analysis_results 테이블)

    완전한 스키마: id, coin, interval, regime, fractal_score, multi_timeframe_score,
                 indicator_cross_score, ensemble_score, ensemble_confidence,
                 final_signal_score, signal_confidence, signal_action, created_at
    """
    try:
        with get_learning_db_connection(LEARNING_RESULTS_DB_PATH) as conn:
            cursor = conn.cursor()

            # 안전하게 속성 접근
            try:
                result_coin = getattr(analysis_result, 'coin', coin)
                # interval은 파라미터 우선 사용 (개별 인터벌 저장 시 덮어쓰기 가능)
                result_interval = interval if interval else getattr(analysis_result, 'interval', 'all_intervals')
                result_regime = getattr(analysis_result, 'regime', regime if regime else 'neutral')

                # 분석 점수들
                fractal_score = getattr(analysis_result, 'fractal_score', 0.0)
                multi_timeframe_score = getattr(analysis_result, 'multi_timeframe_score', 0.0)
                indicator_cross_score = getattr(analysis_result, 'indicator_cross_score', 0.0)

                # 앙상블 점수
                ensemble_score = getattr(analysis_result, 'ensemble_score', 0.0)
                ensemble_confidence = getattr(analysis_result, 'ensemble_confidence', 0.0)

                # 최종 시그널
                final_signal_score = getattr(analysis_result, 'final_signal_score', 0.5)
                signal_confidence = getattr(analysis_result, 'signal_confidence', 0.5)
                signal_action = getattr(analysis_result, 'signal_action', 'HOLD')
                created_at = getattr(analysis_result, 'created_at', datetime.now().isoformat())
            except Exception as e:
                logger.warning(f"⚠️ 분석 결과 속성 접근 실패, 기본값 사용: {e}")
                result_coin = coin
                result_interval = interval
                result_regime = regime if regime else 'neutral'
                fractal_score = 0.0
                multi_timeframe_score = 0.0
                indicator_cross_score = 0.0
                ensemble_score = 0.0
                ensemble_confidence = 0.0
                final_signal_score = 0.5
                signal_confidence = 0.5
                signal_action = 'HOLD'
                created_at = datetime.now().isoformat()

            # 완전한 스키마에 맞춘 INSERT
            cursor.execute("""
                INSERT INTO integrated_analysis_results
                (coin, interval, regime, fractal_score, multi_timeframe_score,
                 indicator_cross_score, ensemble_score, ensemble_confidence,
                 final_signal_score, signal_confidence, signal_action, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_coin,
                result_interval,
                result_regime,
                fractal_score,
                multi_timeframe_score,
                indicator_cross_score,
                ensemble_score,
                ensemble_confidence,
                final_signal_score,
                signal_confidence,
                signal_action,
                created_at
            ))

            conn.commit()
            logger.info(f"✅ 통합 분석 결과 저장 완료: {coin}-{interval}")

        return True

    except Exception as e:
        logger.error(f"❌ 통합 분석 결과 저장 실패: {e}")
        return False

def load_integrated_analysis_results(coin: str, interval: str, db_path: str = None, limit: int = 1) -> Optional[Dict[str, Any]]:
    """통합 분석 결과를 rl_strategies.db에서 로드 (개별 코인 전략 분석)

    완전한 스키마: id, coin, interval, regime, fractal_score, multi_timeframe_score,
                 indicator_cross_score, ensemble_score, ensemble_confidence,
                 final_signal_score, signal_confidence, signal_action, created_at
    """
    try:
        db_path = db_path or LEARNING_RESULTS_DB_PATH

        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()

            # 최신 통합 분석 결과 조회 (완전한 스키마 반영)
            cursor.execute("""
                SELECT
                    coin, interval, regime, fractal_score, multi_timeframe_score,
                    indicator_cross_score, ensemble_score, ensemble_confidence,
                    final_signal_score, signal_confidence, signal_action, created_at
                FROM integrated_analysis_results
                WHERE coin = ? AND interval = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (coin, interval, limit))

            rows = cursor.fetchall()
            if not rows:
                return None

            # 가장 최신 결과 반환
            row = rows[0]
            result = {
                'coin': row[0],
                'interval': row[1],
                'regime': row[2],
                'fractal_score': row[3],
                'multi_timeframe_score': row[4],
                'indicator_cross_score': row[5],
                'ensemble_score': row[6],
                'ensemble_confidence': row[7],
                'final_signal_score': row[8],
                'signal_confidence': row[9],
                'signal_action': row[10],
                'created_at': row[11]
            }

            return result

    except Exception as e:
        logger.error(f"❌ 통합 분석 결과 로드 실패: {e}")
        return None

def save_strategy_summary_for_signals(coin: str, interval: str, db_path: str = None) -> bool:
    """rl_strategies.db의 coin_strategies를 요약하여 rl_strategies.db에 저장"""
    try:
        import json
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        
        # rl_strategies.db에서 전략 데이터 읽기
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 해당 코인/인터벌의 전략들 조회
            cursor.execute("""
                SELECT id, rsi_min, rsi_max, volume_ratio_min, volume_ratio_max,
                       macd_buy_threshold, macd_sell_threshold, profit, win_rate,
                       sharpe_ratio, calmar_ratio, profit_factor, quality_grade
                FROM coin_strategies
                WHERE coin = ? AND interval = ?
                ORDER BY profit DESC, win_rate DESC
                LIMIT 100
            """, (coin, interval))
            
            strategies = cursor.fetchall()
            
            if not strategies:
                logger.warning(f"⚠️ {coin}-{interval} 전략 데이터 없음")
                return False
            
            # 요약 정보 계산
            top_strategy = strategies[0]
            top_strategy_id = top_strategy[0]
            top_strategy_params = json.dumps({
                'rsi_min': top_strategy[1],
                'rsi_max': top_strategy[2],
                'volume_ratio_min': top_strategy[3],
                'volume_ratio_max': top_strategy[4],
                'macd_buy_threshold': top_strategy[5],
                'macd_sell_threshold': top_strategy[6]
            })
            top_profit = top_strategy[7] or 0.0
            top_win_rate = top_strategy[8] or 0.0
            top_quality_grade = top_strategy[12] or 'F'
            
            # 평균 계산
            profits = [s[7] or 0.0 for s in strategies]
            win_rates = [s[8] or 0.0 for s in strategies]
            sharpe_ratios = [s[9] or 0.0 for s in strategies]
            calmar_ratios = [s[10] or 0.0 for s in strategies]
            profit_factors = [s[11] or 1.0 for s in strategies]
            
            avg_profit = sum(profits) / len(profits) if profits else 0.0
            avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
            avg_sharpe_ratio = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0.0
            avg_calmar_ratio = sum(calmar_ratios) / len(calmar_ratios) if calmar_ratios else 0.0
            avg_profit_factor = sum(profit_factors) / len(profit_factors) if profit_factors else 1.0
            
            # 등급별 카운트
            s_grade_count = sum(1 for s in strategies if s[12] == 'S')
            a_grade_count = sum(1 for s in strategies if s[12] == 'A')
            
            # learning_results.db에 저장
            with get_learning_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO strategy_summary_for_signals
                    (coin, interval, top_strategy_id, top_strategy_params, top_profit, top_win_rate,
                     top_quality_grade, avg_profit, avg_win_rate, avg_sharpe_ratio, avg_calmar_ratio,
                     avg_profit_factor, total_strategies, s_grade_count, a_grade_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    coin, interval, top_strategy_id, top_strategy_params, top_profit, top_win_rate,
                    top_quality_grade, avg_profit, avg_win_rate, avg_sharpe_ratio, avg_calmar_ratio,
                    avg_profit_factor, len(strategies), s_grade_count, a_grade_count
                ))
                
                conn.commit()
                logger.info(f"✅ 전략 요약 저장 완료: {coin}-{interval} ({len(strategies)}개 전략)")
                return True
                
    except Exception as e:
        logger.error(f"❌ 전략 요약 저장 실패: {e}")
        return False

def _calculate_global_confidence(strategies: List[tuple]) -> float:
    """글로벌 전략 신뢰도 계산"""
    if not strategies:
        return 0.0
    
    try:
        # 등급별 가중치
        grade_weights = {
            'S': 1.0, 'A': 0.7, 'B': 0.4, 'C': 0.2, 'D': 0.1, 'F': 0.0,
            'UNKNOWN': 0.0
        }
        
        # 전략 데이터 구조: (strategy_id, score, ...)
        # strategies는 튜플 리스트로 가정
        grade_scores = []
        strategy_count = len(strategies)
        
        # 각 전략의 등급 정보 추출 (데이터 구조에 따라 조정 필요)
        for strategy in strategies:
            # strategy가 튜플인 경우, 등급 정보를 추출
            # 실제 구조에 맞게 조정 필요
            if isinstance(strategy, tuple) and len(strategy) > 0:
                # 등급 정보가 있는 경우 (예: strategy[3]에 등급이 있다고 가정)
                if len(strategy) > 3:
                    grade = strategy[3] if isinstance(strategy[3], str) else 'UNKNOWN'
                else:
                    grade = 'UNKNOWN'
            else:
                grade = 'UNKNOWN'
            
            grade_scores.append(grade_weights.get(grade, 0.0))
        
        avg_grade_score = sum(grade_scores) / len(grade_scores) if grade_scores else 0.0
        count_score = min(1.0, strategy_count / 100.0)
        
        # 등급 점수 70%, 전략 수 30% 가중치
        confidence = avg_grade_score * 0.7 + count_score * 0.3
        return round(confidence, 3)
        
    except Exception as e:
        logger.warning(f"⚠️ 글로벌 신뢰도 계산 실패, 기본값 사용: {e}")
        return 0.7

def _extract_rsi_pattern(dna_data: Dict) -> str:
    """실제 DNA 데이터에서 RSI 패턴 추출"""
    try:
        rsi_mean = dna_data.get('rsi_min', {}).get('mean', 50.0)
        
        if rsi_mean < 30:
            return "oversold_dominant"
        elif rsi_mean > 70:
            return "overbought_dominant"
        elif 40 <= rsi_mean <= 60:
            return "neutral_balanced"
        else:
            return "medium"
    except Exception as e:
        logger.debug(f"⚠️ RSI 패턴 추출 실패: {e}")
        return "medium"

def _extract_macd_pattern(dna_data: Dict) -> str:
    """실제 DNA 데이터에서 MACD 패턴 추출"""
    try:
        macd_buy = dna_data.get('macd_buy_threshold', {}).get('mean', 0.0)
        macd_sell = dna_data.get('macd_sell_threshold', {}).get('mean', 0.0)
        
        if macd_buy > 0.01 and macd_sell < -0.01:
            return "strong_trend_following"
        elif abs(macd_buy) < 0.005 and abs(macd_sell) < 0.005:
            return "neutral"
        else:
            return "moderate_trend"
    except Exception as e:
        logger.debug(f"⚠️ MACD 패턴 추출 실패: {e}")
        return "neutral"

def _extract_volume_pattern(dna_data: Dict) -> str:
    """실제 DNA 데이터에서 Volume 패턴 추출"""
    try:
        vol_min = dna_data.get('volume_ratio_min', {}).get('mean', 1.0)
        vol_max = dna_data.get('volume_ratio_max', {}).get('mean', 2.0)
        
        if vol_min > 1.5:
            return "high_volume_focus"
        elif vol_max < 1.2:
            return "low_volume_focus"
        else:
            return "normal"
    except Exception as e:
        logger.debug(f"⚠️ Volume 패턴 추출 실패: {e}")
        return "normal"

def save_dna_summary_for_signals(coin: str, interval: str = None, db_path: str = None) -> bool:
    """rl_strategies.db의 strategy_dna를 요약하여 learning_results.db에 저장"""
    try:
        import json
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        
        # rl_strategies.db에서 DNA 데이터 읽기
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # DNA 데이터 조회
            if coin:
                cursor.execute("""
                    SELECT dna_data FROM strategy_dna
                    WHERE coin = ? AND (interval = ? OR interval IS NULL)
                    ORDER BY created_at DESC LIMIT 1
                """, (coin, interval))
            else:
                cursor.execute("""
                    SELECT dna_data FROM strategy_dna
                    ORDER BY created_at DESC LIMIT 1
                """)
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"⚠️ DNA 데이터 없음: {coin}")
                return False
            
            dna_data = json.loads(row[0])
            
            # 요약 정보 계산
            profitability_score = dna_data.get('win_rate', {}).get('mean', 0.0)
            stability_score = min(1.0, dna_data.get('trades_count', {}).get('mean', 0) / 100.0)
            scalability_score = dna_data.get('complexity_score', {}).get('mean', 0.5)
            dna_quality = min(1.0, dna_data.get('analysis_info', {}).get('total_strategies', 0) / 1000.0)
            
            # 🔥 실제 DNA 데이터에서 패턴 추출
            rsi_pattern = _extract_rsi_pattern(dna_data)
            macd_pattern = _extract_macd_pattern(dna_data)
            volume_pattern = _extract_volume_pattern(dna_data)
            
            # learning_results.db에 저장
            with get_learning_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO dna_summary_for_signals
                    (coin, interval, profitability_score, stability_score, scalability_score,
                     dna_quality, rsi_pattern, macd_pattern, volume_pattern, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    coin, interval, profitability_score, stability_score, scalability_score,
                    dna_quality, rsi_pattern, macd_pattern, volume_pattern
                ))
                
                conn.commit()
                logger.info(f"✅ DNA 요약 저장 완료: {coin or '전체'}")
                return True
                
    except Exception as e:
        logger.error(f"❌ DNA 요약 저장 실패: {e}")
        return False

def save_global_strategy_summary_for_signals(interval: str, db_path: str = None) -> bool:
    """rl_strategies.db의 global_strategies를 요약하여 learning_results.db에 저장"""
    try:
        import json
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        
        # rl_strategies.db에서 글로벌 전략 읽기
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 글로벌 전략 조회
            cursor.execute("""
                SELECT strategy_id, performance_score, global_dna_pattern, 
                       global_fractal_score, global_synergy_score
                FROM global_strategies
                ORDER BY performance_score DESC
                LIMIT 50
            """)
            
            strategies = cursor.fetchall()
            
            if not strategies:
                logger.warning(f"⚠️ {interval} 글로벌 전략 데이터 없음")
                return False
            
            top_strategy = strategies[0]
            top_strategy_id = top_strategy[0]
            top_strategy_params = json.dumps({
                'dna_pattern': top_strategy[2],
                'fractal_score': top_strategy[3],
                'synergy_score': top_strategy[4]
            })
            top_global_score = top_strategy[1] or 0.0
            
            # 평균 계산
            scores = [s[1] or 0.0 for s in strategies]
            avg_global_score = sum(scores) / len(scores) if scores else 0.0
            # 🔥 실제 전략 등급 기반으로 글로벌 신뢰도 계산
            avg_global_confidence = _calculate_global_confidence(strategies)
            
            # learning_results.db에 저장
            with get_learning_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO global_strategy_summary_for_signals
                    (interval, top_global_strategy_id, top_global_strategy_params, top_global_score,
                     avg_global_score, avg_global_confidence, total_global_strategies, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    interval, top_strategy_id, top_strategy_params, top_global_score,
                    avg_global_score, avg_global_confidence, len(strategies)
                ))
                
                conn.commit()
                logger.info(f"✅ 글로벌 전략 요약 저장 완료: {interval} ({len(strategies)}개)")
                return True
                
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 요약 저장 실패: {e}")
        return False

def save_analysis_summary_for_signals(coin: str, interval: str, db_path: str = None) -> bool:
    """rl_strategies.db의 fractal_analysis/synergy_analysis를 요약하여 learning_results.db에 저장"""
    try:
        import json
        from rl_pipeline.db.connection_pool import get_optimized_db_connection
        
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        
        # rl_strategies.db에서 분석 데이터 읽기
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 프랙탈 분석
            cursor.execute("""
                SELECT fractal_score, pattern_distribution, optimal_rsi_min, optimal_rsi_max, optimal_volume_ratio
                FROM fractal_analysis
                WHERE coin = ? AND interval = ?
                ORDER BY created_at DESC LIMIT 1
            """, (coin, interval))
            
            fractal_row = cursor.fetchone()
            fractal_score = 0.0
            fractal_pattern = "{}"
            optimal_rsi_min = 30.0
            optimal_rsi_max = 70.0
            optimal_volume_ratio = 1.0
            
            if fractal_row:
                fractal_score = fractal_row[0] or 0.0
                fractal_pattern = fractal_row[1] or "{}"
                optimal_rsi_min = fractal_row[2] or 30.0
                optimal_rsi_max = fractal_row[3] or 70.0
                optimal_volume_ratio = fractal_row[4] or 1.0
            
            # 시너지 분석
            cursor.execute("""
                SELECT synergy_score, synergy_patterns
                FROM synergy_analysis
                WHERE coin = ? AND interval = ?
                ORDER BY created_at DESC LIMIT 1
            """, (coin, interval))
            
            synergy_row = cursor.fetchone()
            synergy_score = 0.0
            synergy_patterns = "{}"
            
            if synergy_row:
                synergy_score = synergy_row[0] or 0.0
                synergy_patterns = synergy_row[1] or "{}"
            
            # learning_results.db에 저장
            with get_learning_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_summary_for_signals
                    (coin, interval, fractal_score, fractal_pattern, synergy_score,
                     synergy_patterns, optimal_rsi_min, optimal_rsi_max, optimal_volume_ratio, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    coin, interval, fractal_score, fractal_pattern, synergy_score,
                    synergy_patterns, optimal_rsi_min, optimal_rsi_max, optimal_volume_ratio
                ))
                
                conn.commit()
                logger.info(f"✅ 분석 요약 저장 완료: {coin}-{interval}")
                return True
                
    except Exception as e:
        logger.error(f"❌ 분석 요약 저장 실패: {e}")
        return False

def save_global_strategy_results(
    overall_score: float,
    overall_confidence: float = 0.5,
    top_performers: List[Dict[str, Any]] = None,
    db_path: str = None
) -> bool:
    """글로벌 전략 결과를 learning_results.db에 저장
    
    Args:
        overall_score: 전체 성과 점수
        overall_confidence: 전체 신뢰도
        top_performers: 상위 성과자 리스트
        db_path: DB 경로 (기본값: LEARNING_RESULTS_DB_PATH)
    
    Returns:
        저장 성공 여부
    """
    try:
        import json
        
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        
        if top_performers is None:
            top_performers = []
        
        # 상위 성과자에서 코인/인터벌 추출
        # 🔥 타입 확인: 딕셔너리가 아니면 건너뛰기
        top_coins = []
        top_intervals = []
        for p in top_performers[:10]:
            if isinstance(p, dict):
                coin = p.get('coin', '')
                interval = p.get('interval', '')
                if coin:
                    top_coins.append(coin)
                if interval:
                    top_intervals.append(interval)
        
        top_coins = list(set(top_coins))
        top_intervals = list(set(top_intervals))
        
        # 이전 결과와 비교하여 policy_improvement 계산 (간단한 버전)
        policy_improvement = 0.0  # 추후 이전 데이터와 비교 로직 추가 가능
        
        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO global_strategy_results (
                    overall_score, overall_confidence, policy_improvement, convergence_rate,
                    top_performers, top_coins, top_intervals
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                overall_score,
                overall_confidence,
                policy_improvement,
                0.0,  # convergence_rate (계산 필요 시 추가)
                json.dumps(top_performers),
                json.dumps(top_coins),
                json.dumps(top_intervals)
            ))
            
            conn.commit()
            logger.info(f"✅ 글로벌 전략 결과 저장 완료: 점수 {overall_score:.3f}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 결과 저장 실패: {e}", exc_info=True)
        return False

def load_global_strategies_from_db(interval: str = None, db_path: str = None) -> List[Dict[str, Any]]:
    """글로벌 전략을 rl_strategies.db에서 로드"""
    try:
        import json
        from rl_pipeline.core.env import config
        
        db_path = db_path or config.STRATEGIES_DB
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # global_strategies 테이블 존재 확인
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='global_strategies'
            """)
            
            if not cursor.fetchone():
                logger.warning("⚠️ global_strategies 테이블이 없습니다")
                return []
            
            # 글로벌 전략 조회
            if interval:
                cursor.execute("""
                    SELECT id, coin, interval, strategy_type, params, name, description,
                           profit, profit_factor, win_rate, trades_count, quality_grade,
                           market_condition, created_at, updated_at, meta
                    FROM global_strategies
                    WHERE interval = ?
                    ORDER BY created_at DESC
                """, (interval,))
            else:
                cursor.execute("""
                    SELECT id, coin, interval, strategy_type, params, name, description,
                           profit, profit_factor, win_rate, trades_count, quality_grade,
                           market_condition, created_at, updated_at, meta
                    FROM global_strategies
                    ORDER BY created_at DESC
                """)
            
            strategies = []
            for row in cursor.fetchall():
                try:
                    strategy = {
                        'id': row[0],
                        'coin': row[1],
                        'interval': row[2],
                        'strategy_type': row[3],
                        'params': json.loads(row[4]) if row[4] else {},
                        'name': row[5],
                        'description': row[6],
                        'profit': row[7] or 0.0,
                        'profit_factor': row[8] or 0.0,
                        'win_rate': row[9] or 0.5,
                        'trades_count': row[10] or 0,
                        'quality_grade': row[11] or 'A',
                        'market_condition': row[12] or 'neutral',
                        'created_at': row[13],
                        'updated_at': row[14],
                        'meta': json.loads(row[15]) if row[15] else {}
                    }
                    strategies.append(strategy)
                except Exception as e:
                    logger.warning(f"⚠️ 글로벌 전략 파싱 실패: {e}")
                    continue
            
            return strategies
            
    except Exception as e:
        logger.error(f"❌ 글로벌 전략 로드 실패: {e}")
        return []

def get_pipeline_performance_summary(days: int = 7, db_path: str = None) -> Dict[str, Any]:
    """파이프라인 성능 요약"""
    try:
        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # 최근 N일간의 실행 로그 조회
            cursor.execute("""
                SELECT * FROM pipeline_execution_logs 
                WHERE created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            """.format(days))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {'error': 'No data found'}
            
            # 통계 계산
            total_runs = len(rows)
            successful_runs = len([r for r in rows if r['status'] == 'success'])
            failed_runs = total_runs - successful_runs
            
            avg_execution_time = sum(r['execution_time'] for r in rows) / total_runs
            avg_signal_score = sum(r['signal_score'] for r in rows if r['signal_score'] > 0) / successful_runs if successful_runs > 0 else 0
            
            # 액션별 분포
            action_distribution = {}
            for row in rows:
                action = row['signal_action']
                action_distribution[action] = action_distribution.get(action, 0) + 1
            
            # 레짐별 분포
            regime_distribution = {}
            for row in rows:
                regime = row['regime_detected']
                regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
            
            summary = {
                'period_days': days,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
                'avg_execution_time': avg_execution_time,
                'avg_signal_score': avg_signal_score,
                'action_distribution': action_distribution,
                'regime_distribution': regime_distribution,
                'created_at': datetime.now().isoformat()
            }
            
            return summary
            
    except Exception as e:
        logger.error(f"❌ 파이프라인 성능 요약 실패: {e}")
        return {'error': str(e)}

def save_realtime_feedback(
    coin: str,
    interval: str,
    signal_id: str,
    signal_score: float,
    signal_action: str,
    signal_timestamp: datetime,
    actual_profit: float = 0.0,
    actual_success: bool = False,
    market_condition: str = 'unknown',
    learning_adjustment: float = 0.0,
    strategy_update: Dict[str, Any] = None,
    db_path: str = None
) -> bool:
    """실시간 학습 피드백 저장 - 실제 매매 결과를 학습 루프로 피드백
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        signal_id: 시그널 고유 ID
        signal_score: 시그널 점수
        signal_action: 시그널 액션 (buy/sell/hold)
        signal_timestamp: 시그널 발생 시각
        actual_profit: 실제 수익률
        actual_success: 성공 여부
        market_condition: 시장 상태
        learning_adjustment: 학습 조정값
        strategy_update: 전략 업데이트 정보 (JSON)
        db_path: DB 경로
    
    Returns:
        저장 성공 여부
    """
    try:
        import json
        
        if db_path is None:
            db_path = LEARNING_RESULTS_DB_PATH
        
        if strategy_update is None:
            strategy_update = {}
        
        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO realtime_learning_feedback
                (coin, interval, signal_id, signal_score, signal_action, signal_timestamp,
                 actual_profit, actual_success, market_condition, learning_adjustment,
                 strategy_update, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                coin,
                interval,
                signal_id,
                signal_score,
                signal_action,
                signal_timestamp.isoformat() if isinstance(signal_timestamp, datetime) else signal_timestamp,
                actual_profit,
                1 if actual_success else 0,  # SQLite는 BOOLEAN을 INTEGER로 저장
                market_condition,
                learning_adjustment,
                json.dumps(strategy_update),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            logger.info(f"✅ 실시간 피드백 저장 완료: {coin}-{interval} (signal_id={signal_id}, profit={actual_profit:.2f}%)")
            return True
            
    except Exception as e:
        logger.error(f"❌ 실시간 피드백 저장 실패: {e}", exc_info=True)
        return False

def get_realtime_feedback_summary(
    coin: str = None,
    interval: str = None,
    days: int = 7,
    db_path: str = None
) -> Dict[str, Any]:
    """실시간 피드백 요약 통계 조회
    
    Args:
        coin: 코인 심볼 (None이면 전체)
        interval: 인터벌 (None이면 전체)
        days: 조회 기간 (일)
        db_path: DB 경로
    
    Returns:
        피드백 요약 통계
    """
    try:
        with get_learning_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # WHERE 조건 동적 구성
            conditions = ["signal_timestamp >= datetime('now', '-{} days')".format(days)]
            params = []
            
            if coin:
                conditions.append("coin = ?")
                params.append(coin)
            
            if interval:
                conditions.append("interval = ?")
                params.append(interval)
            
            where_clause = " AND ".join(conditions)
            
            # 통계 계산
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_feedbacks,
                    SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) as successful_signals,
                    AVG(actual_profit) as avg_profit,
                    AVG(signal_score) as avg_signal_score,
                    COUNT(DISTINCT coin) as distinct_coins,
                    COUNT(DISTINCT interval) as distinct_intervals
                FROM realtime_learning_feedback
                WHERE {where_clause}
            """, params)
            
            row = cursor.fetchone()
            
            if not row or row[0] == 0:
                return {
                    'total_feedbacks': 0,
                    'success_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_signal_score': 0.0
                }
            
            total = row[0]
            successful = row[1] or 0
            avg_profit = row[2] or 0.0
            avg_signal_score = row[3] or 0.0
            
            return {
                'total_feedbacks': total,
                'successful_signals': successful,
                'success_rate': successful / total if total > 0 else 0.0,
                'avg_profit': avg_profit,
                'avg_signal_score': avg_signal_score,
                'distinct_coins': row[4] or 0,
                'distinct_intervals': row[5] or 0,
                'period_days': days
            }
            
    except Exception as e:
        logger.error(f"❌ 피드백 요약 조회 실패: {e}")
        return {'error': str(e)}

# 편의 함수들
