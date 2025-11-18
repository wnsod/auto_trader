"""
Paper Trading 자동 실행 모듈
- 전략 개발 완료 후 자동으로 Paper Trading 시작
- 주기적으로 실행 및 모니터링
"""

import os
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from .paper_trading import PaperTradingSystem, validate_for_live_trading

logger = logging.getLogger(__name__)

# 환경변수
ENABLE_AUTO_PAPER_TRADING = os.getenv('ENABLE_AUTO_PAPER_TRADING', 'true').lower() == 'true'
PAPER_TRADING_DURATION_DAYS = int(os.getenv('PAPER_TRADING_DURATION_DAYS', '30'))  # 기본 30일 (통계적 신뢰도와 피드백 속도 균형)

# rl_strategies.db 경로 사용 (별도 파일 생성하지 않음)
# 환경변수에서 직접 가져오거나, 없으면 기본 경로 사용
STRATEGIES_DB_PATH = os.getenv('STRATEGIES_DB_PATH')
if not STRATEGIES_DB_PATH:
    # 기본 경로 구성
    DATA_STORAGE_PATH = os.getenv('DATA_STORAGE_PATH', 'data_storage')
    STRATEGIES_DB_PATH = os.path.join(DATA_STORAGE_PATH, 'rl_strategies.db')


class AutoPaperTrading:
    """Paper Trading 자동 실행 시스템"""
    
    def __init__(self):
        # rl_strategies.db 사용 (별도 파일 생성하지 않음)
        self.db_path = STRATEGIES_DB_PATH
        self._ensure_db()
    
    def _ensure_db(self):
        """rl_strategies.db에 Paper Trading 테이블 생성"""
        try:
            # 디렉토리 생성 (필요한 경우)
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # rl_strategies.db에 테이블 추가
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_sessions (
                        session_id TEXT PRIMARY KEY,
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        initial_capital REAL NOT NULL,
                        current_capital REAL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_trades (
                        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        action TEXT NOT NULL,
                        price REAL NOT NULL,
                        size REAL NOT NULL,
                        profit REAL,
                        return_pct REAL,
                        timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES paper_trading_sessions(session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_performance (
                        session_id TEXT PRIMARY KEY,
                        total_return REAL,
                        total_trades INTEGER,
                        win_rate REAL,
                        avg_profit REAL,
                        avg_loss REAL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES paper_trading_sessions(session_id)
                    )
                """)
                
                # 인덱스 추가 (성능 최적화)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_paper_trading_sessions_coin_interval 
                    ON paper_trading_sessions(coin, interval)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_paper_trading_sessions_status 
                    ON paper_trading_sessions(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_paper_trading_trades_session_id 
                    ON paper_trading_trades(session_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_paper_trading_trades_timestamp 
                    ON paper_trading_trades(timestamp)
                """)
                
                conn.commit()
                logger.debug(f"✅ Paper Trading 테이블 생성 완료: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Paper Trading 테이블 생성 실패 ({self.db_path}): {e}")
    
    def start_paper_trading(
        self,
        coin: str,
        interval: str,
        initial_capital: float = 100000,
        duration_days: int = None
    ) -> str:
        """Paper Trading 세션 시작"""
        
        if duration_days is None:
            duration_days = PAPER_TRADING_DURATION_DAYS
        
        try:
            session_id = f"paper_{coin}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            end_time = start_time + timedelta(days=duration_days)
            
            # DB에 세션 저장
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO paper_trading_sessions
                    (session_id, coin, interval, start_time, end_time, initial_capital, current_capital, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, coin, interval, start_time.isoformat(), end_time.isoformat(),
                      initial_capital, initial_capital, 'running'))
                conn.commit()
            
            logger.info(f"🚀 Paper Trading 시작: {session_id} ({coin}-{interval}, {duration_days}일)")
            logger.info(f"   시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   종료 예정: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return session_id
        
        except Exception as e:
            logger.error(f"❌ Paper Trading 세션 시작 실패: {e}")
            return None
    
    def update_paper_trading(
        self,
        session_id: str,
        paper_trader: PaperTradingSystem
    ) -> bool:
        """Paper Trading 세션 업데이트"""
        
        try:
            performance = paper_trader.get_detailed_statistics()
            
            # DB 업데이트
            with sqlite3.connect(self.db_path) as conn:
                # 세션 업데이트
                conn.execute("""
                    UPDATE paper_trading_sessions
                    SET current_capital = ?
                    WHERE session_id = ?
                """, (paper_trader.capital, session_id))
                
                # 성과 업데이트
                conn.execute("""
                    INSERT OR REPLACE INTO paper_trading_performance
                    (session_id, total_return, total_trades, win_rate, avg_profit, avg_loss,
                     profit_factor, sharpe_ratio, max_drawdown, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    performance.get('total_return', 0),
                    performance.get('total_trades', 0),
                    performance.get('win_rate', 0),
                    performance.get('avg_profit', 0),
                    performance.get('avg_loss', 0),
                    performance.get('profit_factor', 0),
                    performance.get('sharpe_ratio', 0),
                    performance.get('max_drawdown', 0),
                    datetime.now().isoformat()
                ))
                
                # 거래 기록 저장
                new_trades = [t for t in paper_trader.trades if t.get('saved', False) is False]
                for trade in new_trades:
                    conn.execute("""
                        INSERT INTO paper_trading_trades
                        (session_id, coin, action, price, size, profit, return_pct, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        trade.get('coin', ''),
                        trade.get('type', ''),
                        trade.get('price', 0),
                        trade.get('size', 0),
                        trade.get('profit', 0),
                        trade.get('return_pct', 0),
                        trade.get('time', datetime.now()).isoformat() if isinstance(trade.get('time'), datetime) else datetime.now().isoformat()
                    ))
                    trade['saved'] = True
                
                conn.commit()
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Paper Trading 업데이트 실패: {e}")
            return False
    
    
    def get_active_sessions(self) -> List[Dict]:
        """활성 Paper Trading 세션 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.*, p.total_return, p.total_trades, p.win_rate
                    FROM paper_trading_sessions s
                    LEFT JOIN paper_trading_performance p ON s.session_id = p.session_id
                    WHERE s.status = 'running'
                    ORDER BY s.start_time DESC
                """)
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                sessions = []
                for row in rows:
                    session = dict(zip(columns, row))
                    sessions.append(session)
                
                return sessions
        
        except Exception as e:
            logger.error(f"❌ 활성 세션 조회 실패: {e}")
            return []
    
    def get_session_performance(self, session_id: str) -> Optional[Dict]:
        """세션 성과 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM paper_trading_performance
                    WHERE session_id = ?
                """, (session_id,))

                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))

                return None

        except Exception as e:
            logger.error(f"❌ 세션 성과 조회 실패: {e}")
            return None

    def cleanup_old_sessions(self, days_old: int = 14) -> int:
        """오래된 Paper Trading 세션 정리

        Args:
            days_old: 이 일수보다 오래된 running 세션을 종료

        Returns:
            정리된 세션 수
        """
        try:
            from datetime import datetime, timedelta

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 오래된 running 세션 조회
                cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
                cursor.execute("""
                    SELECT session_id, coin, interval
                    FROM paper_trading_sessions
                    WHERE status = 'running'
                    AND start_time < ?
                """, (cutoff_date,))

                old_sessions = cursor.fetchall()

                if not old_sessions:
                    logger.info(f"📊 {days_old}일 이상 된 활성 세션 없음")
                    return 0

                # 각 세션을 'expired'로 업데이트
                cleaned = 0
                for session_id, coin, interval in old_sessions:
                    try:
                        cursor.execute("""
                            UPDATE paper_trading_sessions
                            SET status = 'expired',
                                end_time = ?
                            WHERE session_id = ?
                        """, (datetime.now().isoformat(), session_id))

                        logger.info(f"✅ 오래된 세션 종료: {session_id} ({coin}-{interval})")
                        cleaned += 1

                    except Exception as e:
                        logger.error(f"❌ 세션 {session_id} 종료 실패: {e}")
                        continue

                conn.commit()
                logger.info(f"✅ 총 {cleaned}개의 오래된 세션 정리 완료")
                return cleaned

        except Exception as e:
            logger.error(f"❌ 오래된 세션 정리 실패: {e}")
            return 0


def create_strategy_signal_generator(coin: str, interval: str):
    """전략 기반 시그널 생성기 생성"""

    def generate_signal(coin: str, interval: str) -> Optional[Dict]:
        """시그널 생성"""
        try:
            # DB 연결 및 통합 분석 결과 조회
            import sqlite3
            from rl_pipeline.db.reads import fetch_integrated_analysis

            # 🔥 통합 DB에서 최신 통합 분석 결과 조회 (rl_strategies.db 사용)
            db_path = os.getenv('RL_STRATEGIES_DB_PATH', 'data_storage/rl_strategies.db')

            analysis = None
            if os.path.exists(db_path):
                # 🔥 DB 커밋 대기 및 재시도 로직 강화
                max_retries = 5  # 재시도 횟수 증가 (3 → 5)
                retry_delay = 0.2  # 재시도 지연 증가 (100ms → 200ms)
                
                for attempt in range(max_retries):
                    try:
                        with sqlite3.connect(db_path) as conn:
                            # 🔥 개별 인터벌로 먼저 조회
                            analysis = fetch_integrated_analysis(conn, coin, interval)
                            
                            # 🔥 개별 인터벌 결과가 없으면 all_intervals로 폴백 조회
                            if not analysis:
                                logger.debug(f"📊 {coin}-{interval} 개별 인터벌 결과 없음, all_intervals로 폴백 조회")
                                analysis = fetch_integrated_analysis(conn, coin, 'all_intervals')
                            
                            # 결과를 찾았으면 루프 종료
                            if analysis:
                                logger.info(f"✅ {coin}-{interval} 통합 분석 결과 조회 성공 (시도 {attempt + 1}/{max_retries})")
                                break
                    except Exception as db_err:
                        if attempt < max_retries - 1:
                            logger.debug(f"⚠️ {coin}-{interval} DB 조회 실패 (시도 {attempt + 1}/{max_retries}), 재시도 중...: {db_err}")
                            import time
                            time.sleep(retry_delay * (attempt + 1))  # 지수 백오프
                        else:
                            logger.warning(f"⚠️ {coin}-{interval} DB 조회 최종 실패: {db_err}")
                    
                    # 결과가 없으면 잠시 대기 후 재시도
                    if not analysis and attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay * (attempt + 1))
            else:
                logger.warning(f"⚠️ {coin}-{interval} 통합 분석 DB 파일 없음: {db_path}")

            if analysis:
                # integrated_analysis_results 테이블의 signal, score 사용
                signal_action = analysis.get('signal', 'HOLD')
                signal_score = analysis.get('score', 0.5)
                created_at = analysis.get('created_at')
                
                # 🔥 시그널 생성 로그 (HOLD 포함)
                logger.info(f"📊 {coin}-{interval} 시그널 생성: {signal_action} (점수: {signal_score:.3f}, 생성시간: {created_at})")
                
                return {
                    'action': signal_action,
                    'signal_score': signal_score,
                    'confidence': signal_score  # score를 confidence로 사용
                }

            # 🔥 통합 분석 결과가 없을 때 명확한 로그
            logger.warning(f"⚠️ {coin}-{interval} 통합 분석 결과 없음 (DB: {db_path}), 기본 HOLD 시그널 사용")
            
            # 폴백: 간단한 시그널 생성
            current_price = get_realtime_price(coin)
            if not current_price:
                logger.warning(f"⚠️ {coin}-{interval} 현재 가격 조회 실패, 시그널 생성 불가")
                return None

            return {
                'action': 'HOLD',
                'signal_score': 0.5,
                'confidence': 0.0
            }

        except Exception as e:
            logger.error(f"❌ 시그널 생성 실패 ({coin}-{interval}): {e}")
            return None

    return generate_signal


def get_realtime_price(coin: str) -> Optional[float]:
    """실시간 가격 조회"""
    try:
        intervals = ['15m', '30m', '240m', '1d']
        db_paths = [
            os.getenv('CANDLES_DB_PATH', 'data_storage/realtime_candles.db'),
            os.getenv('TRADING_SYSTEM_DB_PATH', 'data_storage/trading_system.db')
        ]
        
        for db_path in db_paths:
            try:
                with sqlite3.connect(db_path) as conn:
                    for interval in intervals:
                        query = """
                            SELECT close FROM candles 
                            WHERE coin = ? AND interval = ? 
                            ORDER BY timestamp DESC LIMIT 1
                        """
                        result = conn.execute(query, (coin, interval)).fetchone()
                        if result and result[0] and result[0] > 0:
                            return float(result[0])
            except Exception:
                continue
        
        return None
    
    except Exception as e:
        logger.error(f"❌ 실시간 가격 조회 실패 ({coin}): {e}")
        return None


def auto_start_paper_trading_after_pipeline(
    coin: str,
    intervals: List[str],
    duration_days: int = None
) -> Dict:
    """파이프라인 완료 후 자동으로 Paper Trading 시작 (백그라운드 실행)"""
    
    if not ENABLE_AUTO_PAPER_TRADING:
        logger.debug("📊 Paper Trading 자동 실행 비활성화")
        return {'status': 'disabled'}
    
    try:
        auto_paper = AutoPaperTrading()
        
        # 각 인터벌별로 Paper Trading 세션만 시작 (실제 실행은 별도 프로세스)
        results = []
        for interval in intervals:  # 🔥 모든 인터벌 처리
            try:
                logger.info(f"🚀 {coin}-{interval} Paper Trading 세션 생성")
                
                session_id = auto_paper.start_paper_trading(
                    coin=coin,
                    interval=interval,
                    duration_days=duration_days or PAPER_TRADING_DURATION_DAYS
                )
                
                if session_id:
                    results.append({
                        'coin': coin,
                        'interval': interval,
                        'session_id': session_id,
                        'status': 'created'
                    })
                    logger.info(f"✅ {coin}-{interval} Paper Trading 세션 생성 완료: {session_id}")
                else:
                    results.append({
                        'coin': coin,
                        'interval': interval,
                        'status': 'failed'
                    })
            
            except Exception as e:
                logger.error(f"❌ {coin}-{interval} Paper Trading 세션 생성 실패: {e}")
                continue
        
        return {
            'status': 'started',
            'results': results,
            'message': 'Paper Trading 세션이 생성되었습니다. 파이프라인 실행 시마다 모니터링이 실행됩니다.'
        }
    
    except Exception as e:
        logger.error(f"❌ Paper Trading 자동 시작 실패: {e}")
        return {'status': 'error', 'error': str(e)}


def run_paper_trading_monitor(coin: Optional[str] = None, session_limit: int = 10):
    """Paper Trading 모니터링 프로세스

    Args:
        coin: 특정 코인만 모니터링 (None이면 모든 활성 세션)
        session_limit: 한 번에 처리할 최대 세션 수 (과부하 방지)
    """

    try:
        auto_paper = AutoPaperTrading()

        # 활성 세션 조회
        active_sessions = auto_paper.get_active_sessions()

        if not active_sessions:
            logger.debug("📊 활성 Paper Trading 세션 없음")
            return

        # 특정 코인만 필터링
        if coin:
            active_sessions = [s for s in active_sessions if s.get('coin') == coin]
            if not active_sessions:
                logger.debug(f"📊 {coin}의 활성 Paper Trading 세션 없음")
                return

        # 🔥 중복 세션 제거 (coin-interval 조합으로 중복 제거, 최신 세션만 유지)
        seen_keys = {}
        for session in active_sessions:
            coin_interval_key = f"{session.get('coin')}-{session.get('interval')}"
            session_id = session.get('session_id', '')
            start_time = session.get('start_time', '')
            
            if coin_interval_key not in seen_keys:
                seen_keys[coin_interval_key] = session
            else:
                # 기존 세션과 비교하여 더 최신 세션 유지
                existing_start = seen_keys[coin_interval_key].get('start_time', '')
                if start_time > existing_start:
                    old_session_id = seen_keys[coin_interval_key].get('session_id', '')
                    seen_keys[coin_interval_key] = session
                    logger.debug(f"🔄 {coin_interval_key} 중복 세션 제거: {old_session_id} → {session_id}")
        
        active_sessions = list(seen_keys.values())
        
        # 세션 수 제한 (과부하 방지)
        if len(active_sessions) > session_limit:
            logger.warning(f"⚠️ 활성 세션이 너무 많음 ({len(active_sessions)}개). 최근 {session_limit}개만 처리")
            active_sessions = active_sessions[:session_limit]

        logger.info(f"📊 Paper Trading 모니터링: {len(active_sessions)}개 세션" +
                   (f" (코인: {coin})" if coin else ""))

        for session in active_sessions:
            try:
                coin = session['coin']
                interval = session['interval']
                session_id = session['session_id']
                
                # 시그널 생성기
                signal_generator = create_strategy_signal_generator(coin, interval)
                
                # Paper Trading 시스템 로드 또는 생성
                paper_trader = PaperTradingSystem(initial_capital=session['initial_capital'])
                
                # 🔥 기존 거래 및 포지션 복원
                with sqlite3.connect(auto_paper.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 1. 기존 거래 로드 (시간순 정렬)
                    cursor.execute("""
                        SELECT action, price, size, profit, return_pct, timestamp
                        FROM paper_trading_trades
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    """, (session_id,))
                    
                    db_trades = cursor.fetchall()
                    
                    # 2. 거래 복원: DB의 거래를 paper_trader에 복원
                    if db_trades:
                        logger.debug(f"📊 {coin}-{interval} 기존 거래 복원: {len(db_trades)}개")
                        
                        for trade_row in db_trades:
                            action, price, size, profit, return_pct, timestamp = trade_row
                            
                            # 거래 기록 복원
                            trade_dict = {
                                'type': action,
                                'coin': coin,
                                'price': price,
                                'size': size,
                                'time': datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.fromtimestamp(timestamp),
                                'saved': True  # 이미 DB에 저장된 거래
                            }
                            
                            if profit is not None:
                                trade_dict['profit'] = profit
                            if return_pct is not None:
                                trade_dict['return_pct'] = return_pct
                            
                            paper_trader.trades.append(trade_dict)
                            
                            # 포지션 복원: BUY 거래는 포지션으로, SELL 거래는 포지션 제거
                            if action == 'BUY':
                                # BUY 거래: 포지션 추가
                                if coin not in paper_trader.positions:
                                    paper_trader.positions[coin] = {
                                        'size': size,
                                        'entry_price': price,
                                        'entry_time': trade_dict['time']
                                    }
                                    # 자본 차감 (이미 차감된 것으로 간주)
                                    paper_trader.capital -= price * size
                            elif action == 'SELL':
                                # SELL 거래: 포지션 제거 및 자본 복원
                                if coin in paper_trader.positions:
                                    position = paper_trader.positions[coin]
                                    # 자본 복원
                                    paper_trader.capital += price * position['size']
                                    # 포지션 제거
                                    del paper_trader.positions[coin]
                        
                        logger.debug(f"✅ {coin}-{interval} 거래 복원 완료: 거래 {len(paper_trader.trades)}개, 포지션 {len(paper_trader.positions)}개, 자본 ${paper_trader.capital:.2f}")
                    else:
                        logger.debug(f"📊 {coin}-{interval} 기존 거래 없음 (새 세션)")
                
                # 실시간 시그널로 거래 실행
                signal = signal_generator(coin, interval)
                if signal:
                    action = signal.get('action', 'HOLD')
                    signal_score = signal.get('signal_score', 0.5)
                    current_price = paper_trader.get_realtime_price(coin)
                    
                    if current_price and action in ['BUY', 'SELL']:
                        # 🔥 거래 실행 결과 확인
                        trade_success = paper_trader.execute_paper_trade(action, coin, current_price)
                        if trade_success:
                            logger.info(f"✅ {coin}-{interval} Paper Trading 거래 성공: {action} @ ${current_price:.2f} (시그널 점수: {signal_score:.3f})")
                        else:
                            logger.warning(f"⚠️ {coin}-{interval} Paper Trading 거래 실패: {action} @ ${current_price:.2f} (자본 부족 또는 포지션 없음)")
                    elif not current_price:
                        logger.warning(f"⚠️ {coin}-{interval} Paper Trading: 현재 가격을 가져올 수 없음 (시그널: {action}, 점수: {signal_score:.3f})")
                    elif action == 'HOLD':
                        logger.info(f"ℹ️ {coin}-{interval} Paper Trading: 시그널 HOLD (거래 없음, 점수: {signal_score:.3f})")
                else:
                    logger.warning(f"⚠️ {coin}-{interval} Paper Trading: 시그널 생성 실패")
                
                # 성과 업데이트
                auto_paper.update_paper_trading(session_id, paper_trader)
                
                # 성과 확인 (BUY/SELL 거래 수를 별도로 표시)
                performance = auto_paper.get_session_performance(session_id)
                if performance:
                    total_return = performance.get('total_return', 0)
                    total_trades = performance.get('total_trades', 0)  # 완료된 거래 (SELL만 카운트)
                    
                    # 🔥 BUY 거래 수 확인 (포지션 수로 추정)
                    open_positions = len(paper_trader.positions)
                    total_buy_trades = len([t for t in paper_trader.trades if t.get('type') == 'BUY'])
                    total_sell_trades = len([t for t in paper_trader.trades if t.get('type') == 'SELL'])
                    
                    if total_trades == 0:
                        if total_buy_trades > 0:
                            logger.info(f"📊 {coin}-{interval} 성과: 수익률 {total_return:.2f}%, 완료 거래 {total_trades}회 (BUY {total_buy_trades}회 실행, 포지션 {open_positions}개 보유 중)")
                        else:
                            logger.info(f"📊 {coin}-{interval} 성과: 수익률 {total_return:.2f}%, 완료 거래 {total_trades}회 (세션 시작 직후, 아직 거래 없음)")
                    else:
                        logger.info(f"📊 {coin}-{interval} 성과: 수익률 {total_return:.2f}%, 완료 거래 {total_trades}회 (BUY {total_buy_trades}회, SELL {total_sell_trades}회, 포지션 {open_positions}개 보유 중)")
                else:
                    logger.debug(f"⚠️ {coin}-{interval} Paper Trading: 성과 데이터 없음 (세션 시작 직후)")
            
            except Exception as e:
                logger.error(f"❌ 세션 {session.get('session_id')} 모니터링 실패: {e}")
                continue
    
    except Exception as e:
        logger.error(f"❌ Paper Trading 모니터링 실패: {e}")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 명령줄 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        # 모니터링 모드: 한 번만 실행 (주기적 체크 제거)
        logger.info("📊 Paper Trading 모니터링 실행")
        run_paper_trading_monitor()
        logger.info("✅ Paper Trading 모니터링 완료")
    
    else:
        # 일반 모드: 한 번 실행
        auto_paper = AutoPaperTrading()
        sessions = auto_paper.get_active_sessions()
        logger.info(f"📊 활성 세션: {len(sessions)}개")
        
        if sessions:
            for session in sessions:
                logger.info(f"   - {session['coin']}-{session['interval']}: {session['session_id']}")
            
            # 모니터링 실행
            run_paper_trading_monitor()
        else:
            logger.info("📊 활성 세션이 없습니다")

