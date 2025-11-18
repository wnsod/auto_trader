"""
실시간 시그널 저장 모듈
통합 분석 결과를 거래 시스템이 사용할 수 있는 signals 테이블에 저장
"""

import logging
import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# trading_system.db 경로
TRADING_SYSTEM_DB_PATH = os.getenv('TRADING_SYSTEM_DB_PATH', 
    '/workspace/data_storage/trading_system.db')

@contextmanager
def get_trading_system_db_connection():
    """trading_system.db 연결 관리"""
    conn = None
    try:
        # ⚠️ absolute_zero_system은 trading_system.db를 사용하지 않음
        # 이 함수는 거래 시스템 연동이 활성화된 경우에만 호출되어야 함
        # 디렉토리 생성
        db_dir = os.path.dirname(TRADING_SYSTEM_DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.debug(f"📂 trading_system.db 디렉토리 생성: {db_dir}")
        
        conn = sqlite3.connect(TRADING_SYSTEM_DB_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ trading_system.db 연결 실패: {e}")
        raise
    finally:
        if conn:
            conn.close()

def ensure_signals_table():
    """signals 테이블 생성 (없으면 생성)"""
    try:
        with get_trading_system_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    signal_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    action TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    rsi REAL,
                    macd REAL,
                    wave_phase TEXT,
                    pattern_type TEXT,
                    risk_level TEXT,
                    volatility REAL,
                    volume_ratio REAL,
                    wave_progress REAL,
                    structure_score REAL,
                    pattern_confidence REAL,
                    integrated_direction TEXT,
                    integrated_strength REAL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, interval, timestamp)
                )
            """)
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals(coin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_combined ON signals(coin, interval) WHERE interval = "combined"')
            
            conn.commit()
            logger.debug("✅ signals 테이블 확인/생성 완료")
            
    except Exception as e:
        logger.error(f"❌ signals 테이블 생성 실패: {e}")
        raise

def save_realtime_signal_from_analysis(
    coin: str,
    interval: str,
    analysis_result: Any,
    candle_data: Optional[Any] = None
) -> bool:
    """🔥 통합 분석 결과를 실시간 시그널로 저장 (5단계)
    
    Args:
        coin: 코인 심볼
        interval: 인터벌 ('combined'로 저장)
        analysis_result: 통합 분석 결과 (CoinSignalScore)
        candle_data: 캔들 데이터 (가격 등 추출용, Optional)
    
    Returns:
        저장 성공 여부
    """
    try:
        # 테이블 확인
        ensure_signals_table()
        
        # 분석 결과에서 데이터 추출
        try:
            final_signal_score = getattr(analysis_result, 'final_signal_score', 0.5)
            signal_action = getattr(analysis_result, 'signal_action', 'HOLD')
            signal_confidence = getattr(analysis_result, 'signal_confidence', 0.5)
            regime = getattr(analysis_result, 'regime', 'neutral')
            
            # 점수를 -1.0 ~ 1.0 범위로 변환 (BUY: 양수, SELL: 음수)
            # signal_score가 0.0 ~ 1.0이면, BUY는 0.5 이상, SELL은 0.5 이하로 매핑
            if signal_action == 'BUY':
                signal_score = final_signal_score  # 0.5 ~ 1.0
            elif signal_action == 'SELL':
                signal_score = -(1.0 - final_signal_score)  # -1.0 ~ -0.5
            else:  # HOLD
                signal_score = final_signal_score - 0.5  # -0.5 ~ 0.5
            
            # action을 소문자로 변환 (signals 테이블 형식)
            action_map = {
                'BUY': 'buy',
                'SELL': 'sell',
                'HOLD': 'hold'
            }
            action = action_map.get(signal_action, 'hold')
            
            # 현재 가격 추출
            current_price = 0.0
            if candle_data is not None:
                try:
                    if hasattr(candle_data, 'iloc'):
                        # DataFrame인 경우
                        if len(candle_data) > 0 and 'close' in candle_data.columns:
                            current_price = float(candle_data['close'].iloc[-1])
                    elif isinstance(candle_data, dict):
                        current_price = float(candle_data.get('close', 0.0))
                except Exception as e:
                    logger.debug(f"가격 추출 실패: {e}")
            
            # 기본 지표 추출 (없으면 None)
            rsi = None
            macd = None
            if candle_data is not None:
                try:
                    if hasattr(candle_data, 'iloc'):
                        if 'rsi' in candle_data.columns and len(candle_data) > 0:
                            rsi = float(candle_data['rsi'].iloc[-1])
                        if 'macd' in candle_data.columns and len(candle_data) > 0:
                            macd = float(candle_data['macd'].iloc[-1])
                    elif isinstance(candle_data, dict):
                        rsi = candle_data.get('rsi')
                        macd = candle_data.get('macd')
                except Exception:
                    pass
            
            # risk_level 계산
            risk_level = 'low'
            if signal_confidence >= 0.8:
                risk_level = 'low'
            elif signal_confidence >= 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            # reason 생성
            reason = f"통합 분석 (레짐: {regime}, 신뢰도: {signal_confidence:.2f})"
            
        except Exception as e:
            logger.warning(f"⚠️ 분석 결과 파싱 실패: {e}")
            # 기본값 사용
            signal_score = 0.0
            action = 'hold'
            signal_confidence = 0.5
            current_price = 0.0
            rsi = None
            macd = None
            risk_level = 'medium'
            reason = '통합 분석 결과'
        
        # timestamp 생성 (현재 시간 Unix timestamp)
        timestamp = int(datetime.now().timestamp())
        
        # signals 테이블에 저장
        with get_trading_system_db_connection() as conn:
            cursor = conn.cursor()
            
            # UNIQUE 제약조건으로 인해 기존 레코드가 있으면 업데이트, 없으면 삽입
            cursor.execute("""
                INSERT OR REPLACE INTO signals (
                    timestamp, coin, interval, signal_score, confidence, action,
                    current_price, rsi, macd, risk_level, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                coin,
                'combined',  # interval은 항상 'combined'로 저장 (거래 시스템 형식)
                signal_score,
                signal_confidence,
                action,
                current_price,
                rsi,
                macd,
                risk_level,
                reason
            ))
            
            conn.commit()
            
            logger.info(f"✅ [{coin}] 실시간 시그널 저장 완료: {action} (점수: {signal_score:.3f}, 신뢰도: {signal_confidence:.3f})")
            return True
            
    except Exception as e:
        logger.error(f"❌ [{coin}] 실시간 시그널 저장 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def save_realtime_signal_batch(
    analysis_results: Dict[str, Any],
    candle_data_dict: Optional[Dict[str, Any]] = None
) -> int:
    """여러 코인의 분석 결과를 일괄 저장
    
    Args:
        analysis_results: {coin: analysis_result} 딕셔너리
        candle_data_dict: {coin: candle_data} 딕셔너리 (Optional)
    
    Returns:
        저장된 시그널 수
    """
    saved_count = 0
    for coin, analysis_result in analysis_results.items():
        candle_data = candle_data_dict.get(coin) if candle_data_dict else None
        if save_realtime_signal_from_analysis(coin, 'combined', analysis_result, candle_data):
            saved_count += 1
    
    logger.info(f"✅ 실시간 시그널 일괄 저장 완료: {saved_count}/{len(analysis_results)}개")
    return saved_count
