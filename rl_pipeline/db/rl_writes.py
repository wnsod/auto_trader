"""
예측형 강화학습 시스템 DB 저장 유틸리티
rl_episodes, rl_steps, rl_episode_summary 저장 함수
"""

import logging
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from rl_pipeline.db.connection_pool import get_optimized_db_connection
from rl_pipeline.core.errors import DBWriteError

logger = logging.getLogger(__name__)


def save_episode_prediction(
    episode_id: str,
    coin: str,
    interval: str,
    strategy_id: str,
    state_key: str,
    predicted_dir: int,
    predicted_conf: float,
    entry_price: float,
    target_move_pct: float,
    horizon_k: int,
    ts_entry: Optional[int] = None,
    db_connection=None
) -> bool:
    """
    에피소드 예측 저장 (rl_episodes 테이블)
    
    Args:
        episode_id: 에피소드 ID
        coin: 코인 심볼
        interval: 인터벌
        strategy_id: 전략 ID
        state_key: 시장 상태 키
        predicted_dir: 예측 방향 (-1/0/+1)
        predicted_conf: 예측 확신도 (0~1)
        entry_price: 진입 가격
        target_move_pct: 목표 변동률
        horizon_k: 목표 캔들 수
        ts_entry: 진입 타임스탬프 (None이면 현재 시간)
        db_connection: DB 연결 (None이면 새로 생성)
    
    Returns:
        성공 여부
    """
    try:
        if ts_entry is None:
            ts_entry = int(datetime.now().timestamp())
        
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _save_episode_prediction_impl(
                    episode_id, coin, interval, strategy_id, state_key,
                    predicted_dir, predicted_conf, entry_price,
                    target_move_pct, horizon_k, ts_entry, conn
                )
        else:
            return _save_episode_prediction_impl(
                episode_id, coin, interval, strategy_id, state_key,
                predicted_dir, predicted_conf, entry_price,
                target_move_pct, horizon_k, ts_entry, db_connection
            )
            
    except Exception as e:
        logger.error(f"❌ 에피소드 예측 저장 실패: {e}")
        raise DBWriteError(f"에피소드 예측 저장 실패: {e}") from e


def _save_episode_prediction_impl(
    episode_id: str,
    coin: str,
    interval: str,
    strategy_id: str,
    state_key: str,
    predicted_dir: int,
    predicted_conf: float,
    entry_price: float,
    target_move_pct: float,
    horizon_k: int,
    ts_entry: int,
    conn
) -> bool:
    """에피소드 예측 저장 구현"""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO rl_episodes (
                episode_id, ts_entry, coin, interval, strategy_id, state_key,
                predicted_dir, predicted_conf, entry_price, target_move_pct, horizon_k
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode_id, ts_entry, coin, interval, strategy_id, state_key,
            predicted_dir, predicted_conf, entry_price, target_move_pct, horizon_k
        ))
        
        conn.commit()
        logger.debug(f"✅ 에피소드 예측 저장: {episode_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 에피소드 예측 저장 실패: {e}")
        conn.rollback()
        return False


def save_episode_step(
    episode_id: str,
    ts: int,
    event: str,
    price: float,
    ret_raw: Optional[float] = None,
    ret_signed: Optional[float] = None,
    dd_pct_norm: Optional[float] = None,
    actual_move_pct: Optional[float] = None,
    prox: Optional[float] = None,
    dir_correct: Optional[int] = None,
    reward_dir: Optional[float] = None,
    reward_price: Optional[float] = None,
    reward_time: Optional[float] = None,
    reward_trade: Optional[float] = None,
    reward_calib: Optional[float] = None,
    reward_risk: Optional[float] = None,
    reward_total: Optional[float] = None,
    db_connection=None
) -> bool:
    """
    에피소드 스텝 저장 (rl_steps 테이블)
    
    Args:
        episode_id: 에피소드 ID
        ts: 타임스탬프
        event: 이벤트 ('TP', 'SL', 'expiry', 'hold', 'scalein', 'scaleout')
        price: 가격
        ret_raw: 원시 수익률
        ret_signed: 부호 포함 수익률
        dd_pct_norm: 정규화된 드로우다운
        actual_move_pct: 실제 변동률
        prox: 근접도
        dir_correct: 방향 정확도 (0/1)
        reward_*: 각 보상 컴포넌트
        reward_total: 총 보상
        db_connection: DB 연결
    
    Returns:
        성공 여부
    """
    try:
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _save_episode_step_impl(
                    episode_id, ts, event, price, ret_raw, ret_signed,
                    dd_pct_norm, actual_move_pct, prox, dir_correct,
                    reward_dir, reward_price, reward_time, reward_trade,
                    reward_calib, reward_risk, reward_total, conn
                )
        else:
            return _save_episode_step_impl(
                episode_id, ts, event, price, ret_raw, ret_signed,
                dd_pct_norm, actual_move_pct, prox, dir_correct,
                reward_dir, reward_price, reward_time, reward_trade,
                reward_calib, reward_risk, reward_total, db_connection
            )
            
    except Exception as e:
        logger.error(f"❌ 에피소드 스텝 저장 실패: {e}")
        raise DBWriteError(f"에피소드 스텝 저장 실패: {e}") from e


def _save_episode_step_impl(
    episode_id: str,
    ts: int,
    event: str,
    price: float,
    ret_raw: Optional[float],
    ret_signed: Optional[float],
    dd_pct_norm: Optional[float],
    actual_move_pct: Optional[float],
    prox: Optional[float],
    dir_correct: Optional[int],
    reward_dir: Optional[float],
    reward_price: Optional[float],
    reward_time: Optional[float],
    reward_trade: Optional[float],
    reward_calib: Optional[float],
    reward_risk: Optional[float],
    reward_total: Optional[float],
    conn
) -> bool:
    """에피소드 스텝 저장 구현"""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO rl_steps (
                episode_id, ts, event, price,
                ret_raw, ret_signed, dd_pct_norm, actual_move_pct, prox, dir_correct,
                reward_dir, reward_price, reward_time, reward_trade,
                reward_calib, reward_risk, reward_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode_id, ts, event, price,
            ret_raw, ret_signed, dd_pct_norm, actual_move_pct, prox, dir_correct,
            reward_dir, reward_price, reward_time, reward_trade,
            reward_calib, reward_risk, reward_total
        ))
        
        conn.commit()
        logger.debug(f"✅ 에피소드 스텝 저장: {episode_id}@{ts}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 에피소드 스텝 저장 실패: {e}")
        conn.rollback()
        return False


def save_episode_summary(
    episode_id: str,
    ts_exit: Optional[int],
    first_event: str,
    t_hit: Optional[int],
    realized_ret_signed: float,
    total_reward: float,
    acc_flag: int,
    coin: str,
    interval: str,
    strategy_id: str,
    source_type: str = 'predictive',
    db_connection=None
) -> bool:
    """
    에피소드 요약 저장 (rl_episode_summary 테이블)
    
    Args:
        episode_id: 에피소드 ID
        ts_exit: 종료 타임스탬프
        first_event: 첫 이벤트 ('TP', 'SL', 'expiry')
        t_hit: 목표 도달까지 걸린 캔들 수
        realized_ret_signed: 실제 수익률 (부호 포함)
        total_reward: 총 보상
        acc_flag: 예측 정확도 플래그 (0/1)
        coin: 코인 심볼
        interval: 인터벌
        strategy_id: 전략 ID
        source_type: 소스 타입 ('predictive' 또는 'simulation', 기본값: 'predictive')
        db_connection: DB 연결
    
    Returns:
        성공 여부
    """
    try:
        if ts_exit is None:
            ts_exit = int(datetime.now().timestamp())
        
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _save_episode_summary_impl(
                    episode_id, ts_exit, first_event, t_hit,
                    realized_ret_signed, total_reward, acc_flag,
                    coin, interval, strategy_id, source_type, conn
                )
        else:
            return _save_episode_summary_impl(
                episode_id, ts_exit, first_event, t_hit,
                realized_ret_signed, total_reward, acc_flag,
                coin, interval, strategy_id, source_type, db_connection
            )
            
    except Exception as e:
        logger.error(f"❌ 에피소드 요약 저장 실패: {e}")
        raise DBWriteError(f"에피소드 요약 저장 실패: {e}") from e


def _save_episode_summary_impl(
    episode_id: str,
    ts_exit: int,
    first_event: str,
    t_hit: Optional[int],
    realized_ret_signed: float,
    total_reward: float,
    acc_flag: int,
    coin: str,
    interval: str,
    strategy_id: str,
    source_type: str,
    conn
) -> bool:
    """에피소드 요약 저장 구현"""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO rl_episode_summary (
                episode_id, ts_exit, first_event, t_hit,
                realized_ret_signed, total_reward, acc_flag,
                coin, interval, strategy_id, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode_id, ts_exit, first_event, t_hit,
            realized_ret_signed, total_reward, acc_flag,
            coin, interval, strategy_id, source_type
        ))
        
        conn.commit()
        logger.debug(f"✅ 에피소드 요약 저장: {episode_id} (event={first_event}, acc={acc_flag})")
        return True
        
    except Exception as e:
        logger.error(f"❌ 에피소드 요약 저장 실패: {e}")
        conn.rollback()
        return False


def save_realtime_prediction(
    coin: str,
    interval: str,
    state_key: str,
    predicted_dir: int,
    predicted_conf: float,
    entry_price: float,
    target_move_pct: float,
    horizon_k: int,
    p_up: Optional[float] = None,
    e_ret: Optional[float] = None,
    prox_est: Optional[float] = None,
    regime: Optional[str] = None,
    source: Optional[str] = None,
    ts: Optional[int] = None,
    db_connection=None
) -> bool:
    """
    실시간 예측 저장 (realtime_predictions 테이블)
    
    대시보드에서 사용하는 실시간 예측 캐시
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        state_key: 시장 상태 키
        predicted_dir: 예측 방향
        predicted_conf: 예측 확신도
        entry_price: 진입 가격
        target_move_pct: 목표 변동률
        horizon_k: 목표 캔들 수
        p_up: 상승 확률
        e_ret: 기대 수익률
        prox_est: 예상 근접도
        regime: 레짐
        source: 예측 소스
        ts: 타임스탬프 (None이면 현재 시간)
        db_connection: DB 연결
    
    Returns:
        성공 여부
    """
    try:
        if ts is None:
            ts = int(datetime.now().timestamp())
        
        if db_connection is None:
            with get_optimized_db_connection("strategies") as conn:
                return _save_realtime_prediction_impl(
                    coin, interval, state_key, predicted_dir, predicted_conf,
                    entry_price, target_move_pct, horizon_k,
                    p_up, e_ret, prox_est, regime, source, ts, conn
                )
        else:
            return _save_realtime_prediction_impl(
                coin, interval, state_key, predicted_dir, predicted_conf,
                entry_price, target_move_pct, horizon_k,
                p_up, e_ret, prox_est, regime, source, ts, db_connection
            )
            
    except Exception as e:
        logger.error(f"❌ 실시간 예측 저장 실패: {e}")
        raise DBWriteError(f"실시간 예측 저장 실패: {e}") from e


def _save_realtime_prediction_impl(
    coin: str,
    interval: str,
    state_key: str,
    predicted_dir: int,
    predicted_conf: float,
    entry_price: float,
    target_move_pct: float,
    horizon_k: int,
    p_up: Optional[float],
    e_ret: Optional[float],
    prox_est: Optional[float],
    regime: Optional[str],
    source: Optional[str],
    ts: int,
    conn
) -> bool:
    """실시간 예측 저장 구현"""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO realtime_predictions (
                ts, coin, interval, state_key,
                predicted_dir, predicted_conf, entry_price, target_move_pct, horizon_k,
                p_up, e_ret, prox_est, regime, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, coin, interval, state_key,
            predicted_dir, predicted_conf, entry_price, target_move_pct, horizon_k,
            p_up, e_ret, prox_est, regime, source
        ))
        
        conn.commit()
        logger.debug(f"✅ 실시간 예측 저장: {coin}-{interval}@{ts}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 실시간 예측 저장 실패: {e}")
        conn.rollback()
        return False


if __name__ == "__main__":
    # 테스트
    print("RL DB 저장 유틸리티 테스트:")
    
    # 테스트: 에피소드 예측 저장
    episode_id = "test_ep_001"
    success = save_episode_prediction(
        episode_id=episode_id,
        coin=None,
        interval="15m",
        strategy_id="test_strategy_001",
        state_key="rsi_low_volume_high",
        predicted_dir=+1,
        predicted_conf=0.75,
        entry_price=45000.0,
        target_move_pct=0.015,
        horizon_k=8
    )
    print(f"에피소드 예측 저장: {success}")

