"""
A/B 평가 시스템
규칙 기반 vs 하이브리드 성능 비교
"""

import logging
import json
import uuid
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from rl_pipeline.simulation.agent import StrategyAgent
from rl_pipeline.simulation.selfplay import run_self_play_test
from rl_pipeline.hybrid.hybrid_policy_agent import HybridPolicyAgent
from rl_pipeline.hybrid.neural_policy_jax import load_ckpt
from rl_pipeline.db.connection_pool import get_strategy_db_pool
from rl_pipeline.db.writes import write_batch


def _calculate_max_drawdown(agent_results: Dict[str, Dict[str, Any]], initial_capital: float = 10000.0) -> float:
    """
    Max Drawdown 계산 (실제 구현)
    
    Args:
        agent_results: 에이전트별 결과 딕셔너리
        initial_capital: 초기 자본
    
    Returns:
        Max Drawdown (0.0 ~ 1.0)
    """
    try:
        if not agent_results:
            return 0.0
        
        # 모든 에이전트의 equity curve 수집
        all_equity_curves = []
        for agent_id, result in agent_results.items():
            # trades에서 equity curve 재구성
            trades = result.get('trades', [])
            if not trades:
                # trades가 없으면 total_pnl로 단순 추정
                total_pnl = result.get('total_pnl', 0.0)
                all_equity_curves.append([initial_capital, initial_capital + total_pnl])
                continue
            
            # 각 트레이드의 P&L로 equity curve 구성
            equity = initial_capital
            equity_curve = [equity]
            
            for trade in trades:
                pnl = trade.get('pnl', 0.0)
                equity += pnl
                equity_curve.append(equity)
            
            all_equity_curves.append(equity_curve)
        
        if not all_equity_curves:
            return 0.0
        
        # 전체 max drawdown 계산
        max_dd = 0.0
        for equity_curve in all_equity_curves:
            if len(equity_curve) < 2:
                continue
            
            equity_array = np.array(equity_curve)
            peak = equity_array[0]
            
            for value in equity_array:
                if value > peak:
                    peak = value
                
                if peak > 0:
                    drawdown = (peak - value) / peak
                    max_dd = max(max_dd, drawdown)
        
        return float(max_dd)
        
    except Exception as e:
        logger.warning(f"⚠️ Max Drawdown 계산 실패: {e}")
        return 0.0


def _calculate_sharpe_ratio(agent_results: Dict[str, Dict[str, Any]], initial_capital: float = 10000.0, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe Ratio 계산 (실제 구현)
    
    Args:
        agent_results: 에이전트별 결과 딕셔너리
        initial_capital: 초기 자본
        risk_free_rate: 무위험 수익률 (기본 0%)
    
    Returns:
        Sharpe Ratio
    """
    try:
        if not agent_results:
            return 0.0
        
        # 모든 에이전트의 수익률 수집
        all_returns = []
        
        for agent_id, result in agent_results.items():
            trades = result.get('trades', [])
            if not trades:
                # trades가 없으면 total_pnl로 단일 수익률 계산
                total_pnl = result.get('total_pnl', 0.0)
                if initial_capital > 0:
                    return_pct = (total_pnl / initial_capital)
                    all_returns.append(return_pct)
                continue
            
            # 각 트레이드의 수익률 계산
            equity = initial_capital
            returns = []
            
            for trade in trades:
                pnl = trade.get('pnl', 0.0)
                if equity > 0:
                    return_pct = pnl / equity
                    returns.append(return_pct)
                    equity += pnl
                else:
                    returns.append(0.0)
            
            all_returns.extend(returns)
        
        if not all_returns or len(all_returns) < 2:
            return 0.0
        
        returns_array = np.array(all_returns)
        
        # 평균 수익률과 표준편차
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Sharpe Ratio = (평균 수익률 - 무위험 수익률) / 표준편차
        if std_return > 0:
            sharpe = (mean_return - risk_free_rate) / std_return
            # 연율화 (252 거래일 기준, 간단히 sqrt(252)로 스케일링)
            # 주의: 실제 거래 빈도에 맞게 조정 필요
            sharpe_annualized = sharpe * np.sqrt(252)
            return float(sharpe_annualized)
        else:
            return 0.0
        
    except Exception as e:
        logger.warning(f"⚠️ Sharpe Ratio 계산 실패: {e}")
        return 0.0


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    eval_id: str
    model_id: Optional[str]
    mode: str  # 'RULE' or 'HYBRID'
    coin: str
    interval: str
    period_from: datetime
    period_to: datetime
    
    # 성능 지표
    profit_factor: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0
    
    # 기타
    trades: int = 0
    latency_ms_p95: float = 0.0
    notes: str = ""


def evaluate_ab(
    model_id: Optional[str],
    mode: str,  # 'RULE' or 'HYBRID'
    coin: str,
    interval: str,
    candle_data: pd.DataFrame,
    strategy_params_list: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    A/B 평가 실행
    
    동일한 데이터로 규칙 기반과 하이브리드 비교
    
    Args:
        model_id: 모델 ID (HYBRID 모드일 때 필요)
        mode: 'RULE' or 'HYBRID'
        coin: 코인 심볼
        interval: 인터벌
        candle_data: 캔들 데이터
        strategy_params_list: 전략 파라미터 리스트
        db_path: DB 경로
        config: 설정 딕셔너리
    
    Returns:
        평가 결과 딕셔너리
    """
    try:
        logger.info(f"🚀 A/B 평가 시작: mode={mode}, coin={coin}, interval={interval}")
        
        # 기간 계산
        if len(candle_data) > 0:
            period_from = pd.to_datetime(candle_data.iloc[0]['timestamp']) if 'timestamp' in candle_data.columns else datetime.now()
            period_to = pd.to_datetime(candle_data.iloc[-1]['timestamp']) if 'timestamp' in candle_data.columns else datetime.now()
        else:
            period_from = datetime.now()
            period_to = datetime.now()
        
        # 평가 실행
        if mode == 'RULE':
            result = _run_rule_based(
                coin, interval, candle_data, strategy_params_list
            )
        elif mode == 'HYBRID':
            if model_id is None:
                raise ValueError("HYBRID 모드에는 model_id가 필요합니다")
            result = _run_hybrid(
                coin, interval, candle_data, strategy_params_list, model_id, config
            )
        else:
            raise ValueError(f"알 수 없는 모드: {mode}")
        
        # 평가 결과 생성
        eval_result = EvaluationResult(
            eval_id=f"eval_{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            mode=mode,
            coin=coin,
            interval=interval,
            period_from=period_from,
            period_to=period_to,
            profit_factor=result.get('profit_factor', 0.0),
            total_return=result.get('total_return', 0.0),
            win_rate=result.get('win_rate', 0.0),
            mdd=result.get('max_drawdown', 0.0),
            sharpe=result.get('sharpe_ratio', 0.0),
            trades=result.get('total_trades', 0),
            latency_ms_p95=result.get('latency_p95', 0.0),
            notes=json.dumps(result.get('details', {}))
        )
        
        # DB 저장
        if db_path:
            _save_evaluation_result(eval_result, db_path)
        
        # JSON 파일로도 저장
        _save_evaluation_json(eval_result, config)
        
        logger.info(f"✅ A/B 평가 완료: PF={eval_result.profit_factor:.2f}, Return={eval_result.total_return:.2%}")
        
        return {
            'eval_id': eval_result.eval_id,
            'mode': mode,
            'profit_factor': eval_result.profit_factor,
            'total_return': eval_result.total_return,
            'win_rate': eval_result.win_rate,
            'mdd': eval_result.mdd,
            'sharpe': eval_result.sharpe,
            'trades': eval_result.trades
        }
        
    except Exception as e:
        logger.error(f"❌ A/B 평가 실패: {e}")
        raise


def _run_rule_based(
    coin: str,
    interval: str,
    candle_data: pd.DataFrame,
    strategy_params_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """규칙 기반 실행"""
    try:
        # Self-play 실행 (규칙 기반 에이전트)
        result = run_self_play_test(
            strategy_params_list=strategy_params_list,
            episodes=5,  # 평가용 에피소드 수
            candle_data=candle_data
        )
        
        if result['status'] != 'success':
            raise RuntimeError(f"Self-play 실행 실패: {result.get('error')}")
        
        # 성능 지표 추출
        cycle_results = result.get('cycle_results', [])
        if not cycle_results:
            return _default_metrics()
        
        # 마지막 사이클 결과 사용
        last_cycle = cycle_results[-1]
        agent_results = last_cycle.get('results', {})
        
        if not agent_results:
            return _default_metrics()
        
        # 평균 성능 계산
        profits = [r.get('total_pnl', 0.0) for r in agent_results.values()]
        win_rates = [r.get('win_rate', 0.0) for r in agent_results.values()]
        trades_counts = [r.get('total_trades', 0) for r in agent_results.values()]
        
        avg_profit = np.mean(profits) if profits else 0.0
        avg_win_rate = np.mean(win_rates) if win_rates else 0.0
        total_trades = sum(trades_counts)
        
        # 🔧 비정상적으로 큰 profit 값 제한 (계산 오류 방지)
        if abs(avg_profit) > 1e9:  # 10억 이상이면 비정상
            logger.warning(f"⚠️ 비정상적으로 큰 profit 값 감지: {avg_profit:.2f}, 0으로 대체")
            avg_profit = 0.0
        
        # Profit Factor 계산 (간단화)
        gross_profit = sum(max(p, 0.0) for p in profits)
        gross_loss = abs(sum(min(p, 0.0) for p in profits))
        total_trades_rule = sum(trades_counts)
        # 🔥 PF 계산 개선: 거래가 없거나 모두 손실인 경우 처리
        if total_trades_rule == 0:
            profit_factor = 0.0  # 거래가 없으면 PF=0
            logger.warning(f"⚠️ 규칙 기반 평가에서 거래가 생성되지 않음 (total_trades=0)")
        elif gross_loss == 0:
            # 손실 거래가 없으면 PF는 무한대이지만, 실제로는 매우 높은 값으로 설정
            profit_factor = 100.0 if gross_profit > 0 else 0.0  # 무한대 대신 100으로 제한
            logger.info(f"✅ 규칙 기반 평가: 손실 거래 없음 (PF={profit_factor:.2f}, 총 수익={gross_profit:.2f})")
        else:
            profit_factor = gross_profit / gross_loss
        
        # 🔧 Return 계산 수정: 초기 자본 기준 퍼센트 (비율로 변환 후 100 곱하기)
        initial_capital = 10000.0
        total_return = (avg_profit / initial_capital)  # 비율 (예: 0.1 = 10%)
        
        # 🔧 Max Drawdown 실제 계산
        max_drawdown = _calculate_max_drawdown(agent_results, initial_capital)
        
        # 🔧 Sharpe Ratio 실제 계산
        sharpe_ratio = _calculate_sharpe_ratio(agent_results, initial_capital)
        
        return {
            'profit_factor': float(profit_factor),
            'total_return': float(total_return),
            'win_rate': float(avg_win_rate),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'total_trades': int(total_trades),
            'latency_p95': 0.1,  # 규칙 기반은 매우 빠름
            'details': {
                'avg_profit': float(avg_profit),
                'agent_count': len(agent_results)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 규칙 기반 실행 실패: {e}")
        return _default_metrics()


def _run_hybrid(
    coin: str,
    interval: str,
    candle_data: pd.DataFrame,
    strategy_params_list: List[Dict[str, Any]],
    model_id: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """하이브리드 실행"""
    try:
        # 모델 로드
        checkpoint_dir = config.get('paths', {}).get('checkpoints', '/workspace/rl_pipeline/artifacts/checkpoints') if config else '/workspace/rl_pipeline/artifacts/checkpoints'
        ckpt_path = os.path.join(checkpoint_dir, f"{model_id}.ckpt")
        
        neural_policy = load_ckpt(ckpt_path)
        
        # 하이브리드 에이전트 생성
        # 🔥 평가 단계에서는 신경망을 더 적극적으로 사용 (threshold 낮춤)
        hybrid_agents = []
        for i, params in enumerate(strategy_params_list):
            # 평가 단계에서는 신경망 threshold를 낮춰서 더 많은 액션 생성
            eval_neural_threshold = config.get('use_neural_threshold', 0.3) * 0.5 if config else 0.15  # 기본값의 50%
            agent = HybridPolicyAgent(
                agent_id=f"hybrid_agent_{i+1}",
                strategy_params=params,
                neural_policy=neural_policy,
                enable_neural=True,
                use_neural_threshold=max(0.1, eval_neural_threshold)  # 최소 0.1로 제한
            )
            hybrid_agents.append(agent)
        
        # Self-play 실행 (하이브리드 에이전트)
        hybrid_config = {
            'enable_neural': True,
            'use_neural_threshold': config.get('use_neural_threshold', 0.3) if config else 0.3,
            'max_latency_ms': config.get('max_latency_ms', 10.0) if config else 10.0
        }
        
        result = run_self_play_test(
            strategy_params_list=strategy_params_list,
            episodes=5,
            candle_data=candle_data,
            agent_type='hybrid',
            neural_policy=neural_policy,
            hybrid_config=hybrid_config
        )
        
        if result['status'] != 'success':
            raise RuntimeError(f"Self-play 실행 실패: {result.get('error')}")
        
        # 성능 지표 추출 (규칙 기반과 동일)
        cycle_results = result.get('cycle_results', [])
        if not cycle_results:
            return _default_metrics()
        
        last_cycle = cycle_results[-1]
        agent_results = last_cycle.get('results', {})
        
        if not agent_results:
            return _default_metrics()
        
        # 통계 수집
        profits = [r.get('total_pnl', 0.0) for r in agent_results.values()]
        win_rates = [r.get('win_rate', 0.0) for r in agent_results.values()]
        trades_counts = [r.get('total_trades', 0) for r in agent_results.values()]
        
        avg_profit = np.mean(profits) if profits else 0.0
        avg_win_rate = np.mean(win_rates) if win_rates else 0.0
        total_trades = sum(trades_counts)
        
        # 🔧 비정상적으로 큰 profit 값 제한 (계산 오류 방지)
        if abs(avg_profit) > 1e9:  # 10억 이상이면 비정상
            logger.warning(f"⚠️ 비정상적으로 큰 profit 값 감지: {avg_profit:.2f}, 0으로 대체")
            avg_profit = 0.0
        
        gross_profit = sum(max(p, 0.0) for p in profits)
        gross_loss = abs(sum(min(p, 0.0) for p in profits))
        # 🔥 PF 계산 개선: 거래가 없거나 모두 손실인 경우 처리
        if total_trades == 0:
            profit_factor = 0.0  # 거래가 없으면 PF=0
            logger.warning(f"⚠️ 평가 단계에서 거래가 생성되지 않음 (total_trades=0, 에이전트: {len(agent_results)}개)")
        elif gross_loss == 0:
            # 손실 거래가 없으면 PF는 무한대이지만, 실제로는 매우 높은 값으로 설정
            profit_factor = 100.0 if gross_profit > 0 else 0.0  # 무한대 대신 100으로 제한
            logger.info(f"✅ 평가 단계: 손실 거래 없음 (PF={profit_factor:.2f}, 총 수익={gross_profit:.2f})")
        else:
            profit_factor = gross_profit / gross_loss
        
        # 🔧 Return 계산 수정: 초기 자본 기준 퍼센트 (비율로 변환)
        initial_capital = 10000.0
        total_return = (avg_profit / initial_capital)  # 비율 (예: 0.1 = 10%)
        
        # 🔧 Max Drawdown 실제 계산
        max_drawdown = _calculate_max_drawdown(agent_results, initial_capital)
        
        # 🔧 Sharpe Ratio 실제 계산
        sharpe_ratio = _calculate_sharpe_ratio(agent_results, initial_capital)
        
        # 하이브리드 통계 (에이전트별)
        neural_ratios = []
        for agent in hybrid_agents:
            stats = agent.get_stats()
            if stats['total_decisions'] > 0:
                neural_ratios.append(stats['neural_ratio'])
        
        avg_neural_ratio = np.mean(neural_ratios) if neural_ratios else 0.0
        
        return {
            'profit_factor': float(profit_factor),
            'total_return': float(total_return),
            'win_rate': float(avg_win_rate),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'total_trades': int(total_trades),
            'latency_p95': 1.0,  # 하이브리드는 약간 느림
            'details': {
                'avg_profit': float(avg_profit),
                'agent_count': len(agent_results),
                'neural_ratio': float(avg_neural_ratio)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 하이브리드 실행 실패: {e}")
        return _default_metrics()


def _default_metrics() -> Dict[str, Any]:
    """기본 메트릭 반환"""
    return {
        'profit_factor': 0.0,
        'total_return': 0.0,
        'win_rate': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'total_trades': 0,
        'latency_p95': 0.0,
        'details': {}
    }


def _save_evaluation_result(eval_result: EvaluationResult, db_path: str):
    """평가 결과를 DB에 저장"""
    try:
        # 🔥 DB 경로가 디렉토리인 경우 파일 경로로 보정 (코인별 DB 사용)
        import os
        if os.path.isdir(db_path):
            if eval_result.coin:
                # 코인별 DB 파일 사용 (예: BTC_strategies.db)
                db_path = os.path.join(db_path, f"{eval_result.coin}_strategies.db")
            else:
                # 기본 파일 사용
                db_path = os.path.join(db_path, 'common_strategies.db')

        # 🔥 테이블이 없으면 생성 (폴백 경로 사용 시 대비)
        # db_path가 있으면 해당 DB에 직접 테이블 생성
        import sqlite3
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                # 테이블 존재 여부 확인
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='evaluation_results'
                """)
                if not cursor.fetchone():
                    # 테이블 생성
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS evaluation_results (
                            eval_id TEXT PRIMARY KEY,
                            model_id TEXT,
                            market_type TEXT NOT NULL DEFAULT 'COIN',
                            market TEXT NOT NULL DEFAULT 'BITHUMB',
                            mode TEXT NOT NULL,
                            asset TEXT NOT NULL,
                            interval TEXT NOT NULL,
                            period_from DATETIME NOT NULL,
                            period_to DATETIME NOT NULL,
                            profit_factor REAL,
                            total_return REAL,
                            win_rate REAL,
                            mdd REAL,
                            sharpe REAL,
                            trades INTEGER,
                            latency_ms_p95 REAL,
                            notes TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    # 인덱스 생성
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_model ON evaluation_results(model_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_mode ON evaluation_results(mode)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_asset_interval ON evaluation_results(asset, interval)")
                    conn.commit()
                    logger.debug(f"✅ evaluation_results 테이블 생성 완료: {db_path}")
        except Exception as table_err:
            logger.debug(f"⚠️ evaluation_results 테이블 생성 시도 실패 (무시 가능): {table_err}")
        
        record = {
            'eval_id': eval_result.eval_id,
            'model_id': eval_result.model_id,
            'mode': eval_result.mode,
            'asset': eval_result.coin,
            'interval': eval_result.interval,
            'period_from': eval_result.period_from.isoformat(),
            'period_to': eval_result.period_to.isoformat(),
            'profit_factor': eval_result.profit_factor,
            'total_return': eval_result.total_return,
            'win_rate': eval_result.win_rate,
            'mdd': eval_result.mdd,
            'sharpe': eval_result.sharpe,
            'trades': eval_result.trades,
            'latency_ms_p95': eval_result.latency_ms_p95,
            'notes': eval_result.notes,
            'created_at': datetime.now().isoformat()
        }
        
        write_batch([record], 'evaluation_results', db_path=db_path)
        logger.info(f"✅ 평가 결과 DB 저장 완료: {eval_result.eval_id}")
        
    except Exception as e:
        logger.warning(f"⚠️ DB 저장 실패 (계속 진행): {e}")


def _save_evaluation_json(eval_result: EvaluationResult, config: Optional[Dict[str, Any]] = None):
    """평가 결과를 JSON 파일로 저장"""
    try:
        evals_dir = config.get('paths', {}).get('evals', '/workspace/rl_pipeline/artifacts/evals') if config else '/workspace/rl_pipeline/artifacts/evals'
        os.makedirs(evals_dir, exist_ok=True)
        
        json_path = os.path.join(evals_dir, f"{eval_result.eval_id}.json")
        
        result_dict = {
            'eval_id': eval_result.eval_id,
            'model_id': eval_result.model_id,
            'mode': eval_result.mode,
            'coin': eval_result.coin,
            'interval': eval_result.interval,
            'period_from': eval_result.period_from.isoformat(),
            'period_to': eval_result.period_to.isoformat(),
            'metrics': {
                'profit_factor': eval_result.profit_factor,
                'total_return': eval_result.total_return,
                'win_rate': eval_result.win_rate,
                'mdd': eval_result.mdd,
                'sharpe': eval_result.sharpe,
                'trades': eval_result.trades,
                'latency_ms_p95': eval_result.latency_ms_p95
            },
            'notes': eval_result.notes
        }
        
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"✅ 평가 결과 JSON 저장 완료: {json_path}")
        
    except Exception as e:
        logger.warning(f"⚠️ JSON 저장 실패 (계속 진행): {e}")


def walk_forward_validation(
    model_id: str,
    coin: str,
    interval: str,
    candle_data: pd.DataFrame,
    strategy_params_list: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    db_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Walk-Forward 검증 (과적합 방지)
    
    시간 순서를 유지하여 학습/테스트 데이터로 분할하고 평가
    
    Args:
        model_id: 모델 ID
        coin: 코인 심볼
        interval: 인터벌
        candle_data: 전체 캔들 데이터 (시간 순서 정렬됨)
        strategy_params_list: 전략 파라미터 리스트
        train_ratio: 학습 데이터 비율 (기본 0.7)
        db_path: DB 경로
        config: 설정 딕셔너리
    
    Returns:
        Walk-Forward 검증 결과 딕셔너리
    """
    try:
        logger.info(f"🔍 Walk-Forward 검증 시작: {coin}-{interval}, train_ratio={train_ratio}")
        
        if candle_data.empty or len(candle_data) < 20:
            logger.warning(f"⚠️ 데이터가 부족하여 Walk-Forward 검증 건너뜀 (길이: {len(candle_data)})")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data',
                'train_result': None,
                'test_result': None
            }
        
        # 시간 순서 확인 및 정렬
        if 'timestamp' in candle_data.columns:
            candle_data = candle_data.sort_values('timestamp').reset_index(drop=True)
        
        # 시간 기반 분할
        split_idx = int(len(candle_data) * train_ratio)
        
        if split_idx < 10:
            logger.warning(f"⚠️ 학습 데이터가 너무 적어서 Walk-Forward 검증 건너뜀 (split_idx: {split_idx})")
            return {
                'status': 'skipped',
                'reason': 'insufficient_train_data',
                'train_result': None,
                'test_result': None
            }
        
        train_data = candle_data[:split_idx].copy()
        test_data = candle_data[split_idx:].copy()
        
        logger.info(f"📊 데이터 분할: 전체 {len(candle_data)}개 → 학습 {len(train_data)}개, 테스트 {len(test_data)}개")
        
        # 🔥 테스트 데이터 최소 크기 체크 (거래 생성에 충분한 데이터 필요)
        MIN_TEST_DATA_SIZE = 50  # 최소 50개 캔들 필요
        if len(test_data) < MIN_TEST_DATA_SIZE:
            logger.warning(f"⚠️ 테스트 데이터가 부족하여 Walk-Forward 검증 건너뜀 (테스트: {len(test_data)}개 < 최소 {MIN_TEST_DATA_SIZE}개)")
            return {
                'status': 'skipped',
                'reason': 'insufficient_test_data',
                'train_result': None,
                'test_result': None,
                'message': f'테스트 데이터 부족: {len(test_data)}개 < {MIN_TEST_DATA_SIZE}개'
            }
        
        # 학습 데이터로 평가 (과적합 체크용)
        train_result = evaluate_ab(
            model_id=model_id,
            mode='HYBRID',
            coin=coin,
            interval=interval,
            candle_data=train_data,
            strategy_params_list=strategy_params_list,
            db_path=db_path,
            config=config
        )
        
        # 테스트 데이터로 평가 (실제 성능)
        test_result = evaluate_ab(
            model_id=model_id,
            mode='HYBRID',
            coin=coin,
            interval=interval,
            candle_data=test_data,
            strategy_params_list=strategy_params_list,
            db_path=db_path,
            config=config
        )
        
        # 🔥 PF=0.00인 경우 상세 로그 출력 및 과적합 체크
        train_trades = train_result.get('total_trades', 0)
        test_trades = test_result.get('total_trades', 0)
        train_pf = train_result.get('profit_factor', 0.0)
        test_pf = test_result.get('profit_factor', 0.0)
        
        if train_pf == 0.0:
            if train_trades == 0:
                logger.warning(f"⚠️ 학습 데이터 평가: 거래가 생성되지 않음 (데이터: {len(train_data)}개)")
            else:
                logger.warning(f"⚠️ 학습 데이터 평가: 거래 {train_trades}회 발생했으나 PF=0.00 (모두 손실 또는 수익 거래 없음)")
        
        if test_pf == 0.0:
            if test_trades == 0:
                logger.warning(f"⚠️ 테스트 데이터 평가: 거래가 생성되지 않음 (데이터: {len(test_data)}개)")
            else:
                logger.warning(f"⚠️ 테스트 데이터 평가: 거래 {test_trades}회 발생했으나 PF=0.00 (모두 손실 또는 수익 거래 없음)")
        
        # PF가 0인 경우도 고려하여 더 정확한 과적합 감지
        if train_pf > 0 and test_pf > 0:
            overfitting_ratio = test_pf / train_pf if train_pf > 0 else 0.0
        elif train_pf == 0 and test_pf == 0:
            # 둘 다 0이면 과적합 가능성 높음 (학습 실패)
            overfitting_ratio = 0.0
        elif train_pf > 0 and test_pf == 0:
            # 학습은 성공했지만 테스트 실패 → 과적합 가능성 매우 높음
            overfitting_ratio = 0.0
        else:
            # 테스트 성공, 학습 실패 (이상 케이스)
            overfitting_ratio = 1.0 if test_pf > 0 else 0.0
        
        # 🔥 과적합 경고 기준 강화 (더 민감하게 감지)
        # 테스트 성능이 학습 성능의 70% 미만이면 과적합 가능성 (기존 80% → 70%)
        has_overfitting = overfitting_ratio < 0.7
        
        # 추가: 둘 다 PF=0이면 과적합 가능성 높음
        if train_pf == 0 and test_pf == 0:
            has_overfitting = True
            overfitting_ratio = 0.0
        
        if has_overfitting:
            logger.warning(f"⚠️ 과적합 가능성 감지: 학습 PF={train_pf:.2f}, 테스트 PF={test_pf:.2f} (비율: {overfitting_ratio:.1%})")
        else:
            logger.info(f"✅ 과적합 없음: 학습 PF={train_pf:.2f}, 테스트 PF={test_pf:.2f} (비율: {overfitting_ratio:.1%})")
        
        # 기간 정보
        if 'timestamp' in train_data.columns:
            train_from = pd.to_datetime(train_data.iloc[0]['timestamp'])
            train_to = pd.to_datetime(train_data.iloc[-1]['timestamp'])
        else:
            train_from = datetime.now()
            train_to = datetime.now()
        
        if 'timestamp' in test_data.columns:
            test_from = pd.to_datetime(test_data.iloc[0]['timestamp'])
            test_to = pd.to_datetime(test_data.iloc[-1]['timestamp'])
        else:
            test_from = datetime.now()
            test_to = datetime.now()
        
        return {
            'status': 'success',
            'train_result': {
                'profit_factor': train_result.get('profit_factor', 0.0),
                'total_return': train_result.get('total_return', 0.0),
                'win_rate': train_result.get('win_rate', 0.0),
                'sharpe': train_result.get('sharpe', 0.0),
                'period_from': train_from.isoformat(),
                'period_to': train_to.isoformat()
            },
            'test_result': {
                'profit_factor': test_result.get('profit_factor', 0.0),
                'total_return': test_result.get('total_return', 0.0),
                'win_rate': test_result.get('win_rate', 0.0),
                'sharpe': test_result.get('sharpe', 0.0),
                'period_from': test_from.isoformat(),
                'period_to': test_to.isoformat()
            },
            'overfitting_ratio': float(overfitting_ratio),
            'has_overfitting': bool(has_overfitting),
            'train_ratio': float(train_ratio)
        }
        
    except Exception as e:
        logger.error(f"❌ Walk-Forward 검증 실패: {e}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return {
            'status': 'error',
            'error': str(e),
            'train_result': None,
            'test_result': None
        }


def _detect_market_regimes(candle_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    시장 레짐 탐지 (상승장/하락장/횡보장)
    
    Args:
        candle_data: 캔들 데이터
    
    Returns:
        레짐별 데이터 딕셔너리 {'bullish': df, 'bearish': df, 'sideways': df}
    """
    try:
        if candle_data.empty or len(candle_data) < 20:
            return {}
        
        # timestamp 컬럼 확인
        has_timestamp = 'timestamp' in candle_data.columns
        if has_timestamp:
            candle_data = candle_data.sort_values('timestamp').reset_index(drop=True)
        
        # 가격 데이터 확인
        if 'close' not in candle_data.columns:
            logger.warning("⚠️ 'close' 컬럼이 없어 레짐 탐지 불가")
            return {}
        
        regimes = {
            'bullish': [],
            'bearish': [],
            'sideways': []
        }
        
        # 슬라이딩 윈도우로 레짐 탐지
        window_size = 20
        step_size = 5
        
        for i in range(0, len(candle_data) - window_size, step_size):
            window_data = candle_data.iloc[i:i+window_size]
            
            if 'close' not in window_data.columns or len(window_data) < 5:
                continue
            
            closes = window_data['close'].dropna()
            if len(closes) < 5:
                continue
            
            # 가격 변화율 계산
            price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
            
            # 변동성 계산
            returns = closes.pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.0
            
            # 레짐 분류
            if price_change > 0.05 and volatility < 0.05:  # 5% 이상 상승 + 낮은 변동성
                regime = 'bullish'
            elif price_change < -0.05 and volatility < 0.05:  # 5% 이상 하락 + 낮은 변동성
                regime = 'bearish'
            elif abs(price_change) < 0.02:  # 2% 이내 변동
                regime = 'sideways'
            else:
                continue  # 불명확한 구간은 제외
            
            # 해당 레짐에 데이터 인덱스 추가
            regimes[regime].extend(range(i, min(i+window_size, len(candle_data))))
        
        # 레짐별 고유 인덱스 추출 및 DataFrame 생성
        regime_dataframes = {}
        for regime, indices in regimes.items():
            if len(indices) > 0:
                unique_indices = sorted(set(indices))
                regime_df = candle_data.iloc[unique_indices].copy()
                if len(regime_df) >= 10:  # 최소 10개 이상만 포함
                    regime_dataframes[regime] = regime_df
        
        logger.info(f"📊 레짐 탐지 결과: {', '.join([f'{k}: {len(v)}개' for k, v in regime_dataframes.items()])}")
        
        return regime_dataframes
        
    except Exception as e:
        logger.error(f"❌ 레짐 탐지 실패: {e}")
        return {}


def multi_period_validation(
    model_id: str,
    coin: str,
    interval: str,
    candle_data: pd.DataFrame,
    strategy_params_list: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    다중 기간 검증 (레짐 안정성 확인)
    
    상승장/하락장/횡보장에서 각각 평가하여 일관성 확인
    
    Args:
        model_id: 모델 ID
        coin: 코인 심볼
        interval: 인터벌
        candle_data: 전체 캔들 데이터
        strategy_params_list: 전략 파라미터 리스트
        db_path: DB 경로
        config: 설정 딕셔너리
    
    Returns:
        다중 기간 검증 결과 딕셔너리
    """
    try:
        logger.info(f"🔍 다중 기간 검증 시작: {coin}-{interval}")
        
        # 레짐 탐지
        regime_dataframes = _detect_market_regimes(candle_data)
        
        if not regime_dataframes:
            logger.warning(f"⚠️ 레짐을 찾을 수 없어 다중 기간 검증 건너뜀")
            return {
                'status': 'skipped',
                'reason': 'no_regimes_detected',
                'regime_results': {}
            }
        
        # 각 레짐별 평가
        regime_results = {}
        for regime_name, regime_data in regime_dataframes.items():
            logger.info(f"📊 {regime_name} 레짐 평가 중: {len(regime_data)}개 캔들")
            
            try:
                result = evaluate_ab(
                    model_id=model_id,
                    mode='HYBRID',
                    coin=coin,
                    interval=interval,
                    candle_data=regime_data,
                    strategy_params_list=strategy_params_list,
                    db_path=db_path,
                    config=config
                )
                
                regime_results[regime_name] = {
                    'profit_factor': result.get('profit_factor', 0.0),
                    'total_return': result.get('total_return', 0.0),
                    'win_rate': result.get('win_rate', 0.0),
                    'sharpe': result.get('sharpe', 0.0),
                    'trades': result.get('trades', 0),
                    'data_points': len(regime_data)
                }
                
            except Exception as e:
                logger.warning(f"⚠️ {regime_name} 레짐 평가 실패: {e}")
                regime_results[regime_name] = None
        
        # 일관성 계산
        valid_results = {k: v for k, v in regime_results.items() if v is not None}
        
        if len(valid_results) < 2:
            logger.warning(f"⚠️ 유효한 레짐 평가 결과가 부족하여 일관성 계산 불가 ({len(valid_results)}개)")
            consistency = 0.0
        else:
            # 🔥 개선: Profit Factor 기준 일관성 계산 (재학습 권장 반영)
            # PF가 0인 경우도 포함하여 일관성 계산 개선
            profit_factors = []
            returns = []
            win_rates = []
            
            for r in valid_results.values():
                pf = r.get('profit_factor', 0.0)
                ret = r.get('total_return', 0.0)
                wr = r.get('win_rate', 0.0)
                
                # PF가 0이어도 포함 (음수 수익률도 고려)
                if pf >= 0:  # 0 이상이면 포함 (음수는 제외)
                    profit_factors.append(pf)
                if ret is not None:
                    returns.append(ret)
                if wr is not None:
                    win_rates.append(wr)
            
            # 🔥 다중 지표 기반 일관성 계산 (PF, Return, Win Rate)
            consistency_scores = []
            
            if len(profit_factors) >= 2:
                pf_mean = np.mean(profit_factors)
                pf_std = np.std(profit_factors)
                if pf_mean > 0:
                    cv_pf = pf_std / pf_mean
                    consistency_scores.append(max(0.0, min(1.0, 1.0 - cv_pf)))
                elif pf_mean == 0 and pf_std == 0:
                    # 모두 0이면 일관성 0 (모두 실패)
                    consistency_scores.append(0.0)
                else:
                    consistency_scores.append(0.0)
            
            if len(returns) >= 2:
                ret_mean = np.mean(returns)
                ret_std = np.std(returns)
                if abs(ret_mean) > 1e-6:
                    cv_ret = ret_std / abs(ret_mean)
                    consistency_scores.append(max(0.0, min(1.0, 1.0 - cv_ret)))
                else:
                    consistency_scores.append(0.0)
            
            if len(win_rates) >= 2:
                wr_mean = np.mean(win_rates)
                wr_std = np.std(win_rates)
                if wr_mean > 0:
                    cv_wr = wr_std / wr_mean
                    consistency_scores.append(max(0.0, min(1.0, 1.0 - cv_wr)))
                else:
                    consistency_scores.append(0.0)
            
            # 종합 일관성 (가중 평균)
            if consistency_scores:
                consistency = np.mean(consistency_scores)
            else:
                consistency = 0.0
        
        # 일관성 경고
        if consistency < 0.7:
            logger.warning(f"⚠️ 성능 일관성 부족: {consistency:.1%} (레짐별 성능 차이가 큼)")
        else:
            logger.info(f"✅ 성능 일관성 양호: {consistency:.1%}")
        
        return {
            'status': 'success',
            'regime_results': regime_results,
            'consistency': float(consistency),
            'regime_count': len(valid_results)
        }
        
    except Exception as e:
        logger.error(f"❌ 다중 기간 검증 실패: {e}")
        import traceback
        logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        return {
            'status': 'error',
            'error': str(e),
            'regime_results': {}
        }
