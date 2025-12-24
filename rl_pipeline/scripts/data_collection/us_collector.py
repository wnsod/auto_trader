"""
US Market Data Collector (us_collector.py)

기능:
1. S&P 500 티커 리스트 조회 (market_analyzer 활용)
2. yfinance 캔들 수집 (5m, 15m, 30m, 1d)
3. 증분 수집 + 재시도 + period 백업 요청 (KRX 수집기와 동일 패턴)
4. 데이터 정제 및 DB 저장 + 보관기간(Pruning)
"""

import os
import sys
import time
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# === 경로 설정 ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
US_MARKET_DIR = os.path.join(ROOT_DIR, "market", "us_market")

# rl_pipeline 등 공용 모듈 경로 추가
sys.path.append(ROOT_DIR)

# market_analyzer 임포트
try:
    from market.us_market.market_analyzer import fetch_sp500_tickers
except ImportError:
    sys.path.append(US_MARKET_DIR)
    from market_analyzer import fetch_sp500_tickers

# === 환경 변수 로드 ===
# run_learning.py / run_trading.py에서 env를 로드하지만, 직접 실행 시를 대비해 한 번 더 로드
ENV_PATH = os.path.join(US_MARKET_DIR, "config_learning.env")
if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.join(US_MARKET_DIR, "config_trading.env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# DB 경로 (실행 시 env에서 주입됨)
DB_PATH = os.environ.get("RL_DB_PATH", os.path.join(US_MARKET_DIR, "data_storage", "learning_candles.db"))

# 설정
INTERVALS = os.getenv("CANDLE_INTERVALS", "5m,15m,30m,1d").split(",")
DAYS_BACK_MAP = {
    "5m": int(os.getenv("DAYS_BACK_5M", 59)),
    "15m": int(os.getenv("DAYS_BACK_15M", 59)),
    "30m": int(os.getenv("DAYS_BACK_30M", 59)),
    "1d": int(os.getenv("DAYS_BACK_1D", 730)),
}
RETENTION_MAP = {
    "5m": int(os.getenv("RETENTION_DAYS_5M", 90)),
    "15m": int(os.getenv("RETENTION_DAYS_15M", 180)),
    "30m": int(os.getenv("RETENTION_DAYS_30M", 365)),
    "1d": int(os.getenv("RETENTION_DAYS_1D", 3650)),
}

# ======================
# DB 유틸
# ======================

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for interval in INTERVALS:
            table_name = f"candles_{interval}"
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def get_last_timestamp(symbol, interval):
    table_name = f"candles_{interval}"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol=?", (symbol,))
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    except Exception:
        return None


def save_to_db(interval, data_map):
    if not data_map:
        return
    table_name = f"candles_{interval}"
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        total = 0
        for symbol, df in data_map.items():
            if df is None or df.empty:
                continue
            records = []
            for ts, row in df.iterrows():
                try:
                    unix_ts = int(pd.to_datetime(ts).timestamp())
                except Exception:
                    continue
                records.append(
                    (
                        symbol,
                        unix_ts,
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    )
                )
            if records:
                cur.executemany(
                    f"""
                    INSERT OR REPLACE INTO {table_name}
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    records,
                )
                total += len(records)
        conn.commit()
        print(f"   💾 {interval}: {total}개 캔들 저장")


def prune_old_data():
    print("🧹 데이터 정리(Pruning) 시작...")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        now = int(time.time())
        for interval, retention_days in RETENTION_MAP.items():
            if interval not in INTERVALS:
                continue
            cutoff_ts = now - retention_days * 24 * 3600
            table_name = f"candles_{interval}"
            try:
                cur.execute(f"DELETE FROM {table_name} WHERE timestamp < ?", (cutoff_ts,))
                deleted = cur.rowcount
                if deleted > 0:
                    print(f"   🗑️ {interval}: {deleted}개 삭제 (< {retention_days}일)")
            except Exception as e:
                print(f"   ⚠️ {interval} 정리 오류: {e}")
        conn.commit()


# ======================
# yfinance 수집 (증분 + 재시도) - KRX 로직과 동일 패턴
# ======================

def fetch_data_yfinance(symbol, interval, days_back):
    MAX_RETRY = 3
    last_ts = get_last_timestamp(symbol, interval)
    end_dt = datetime.now(timezone.utc)  # UTC aware

    if last_ts:
        last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
        start_dt = last_dt + timedelta(days=1)
        if start_dt > end_dt:
            return pd.DataFrame()
        print(f"   🔄 증분 수집: {last_dt.strftime('%Y-%m-%d')} 이후")
    else:
        start_dt = end_dt - timedelta(days=days_back)

    str_start = start_dt.strftime("%Y-%m-%d")
    str_end = end_dt.strftime("%Y-%m-%d")

    # 분봉 기간 제한 (60일)
    if interval in ["5m", "15m", "30m"]:
        limit_dt = end_dt - timedelta(days=59)
        if start_dt < limit_dt:
            print(f"   ⚠️ {interval} 제한: {str_start} -> {limit_dt.strftime('%Y-%m-%d')} 로 조정")
            str_start = limit_dt.strftime("%Y-%m-%d")

    def _normalize(df):
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]

    for attempt in range(1, MAX_RETRY + 1):
        df = pd.DataFrame()
        try:
            # 기본 start/end 요청
            df = yf.download(symbol, start=str_start, end=str_end, interval=interval, progress=False, auto_adjust=True)
            # period 기반 백업
            if df.empty:
                period_days = min(days_back, 59 if interval in ["5m", "15m", "30m"] else days_back)
                period_str = f"{period_days}d"
                df = yf.download(symbol, period=period_str, interval=interval, progress=False, auto_adjust=True)

            df = _normalize(df)
            if not df.empty:
                return df

            print(f"   ⚠️ {symbol}/{interval} 재시도 {attempt}/{MAX_RETRY} (데이터 없음)")
            time.sleep(0.5 * attempt)
        except Exception as e:
            print(f"   ❌ yfinance 오류 {symbol}/{interval} 시도 {attempt}/{MAX_RETRY}: {e}")
            time.sleep(0.5 * attempt)

    return pd.DataFrame()


# ======================
# 메인
# ======================

def main():
    print(f"🚀 US Market Data Collector (DB: {DB_PATH})")
    init_db()

    target = os.getenv("TARGET_COINS", "ALL")
    if target == "ALL" or not target:
        tickers = fetch_sp500_tickers()
    else:
        tickers = [t.strip() for t in target.split(",") if t.strip()]

    print(f"📋 수집 대상: {len(tickers)}개 종목")

    total_saved = 0
    for interval in INTERVALS:
        print(f"\n⏰ {interval} 수집 중...")
        days_back = int(os.getenv(f"DAYS_BACK_{interval.upper()}", DAYS_BACK_MAP.get(interval, 59)))

        # 인터벌별 pruning
        try:
            retention_days = int(os.getenv(f"RETENTION_DAYS_{interval.upper()}", RETENTION_MAP.get(interval, 0)))
            cutoff_ts = int(time.time()) - retention_days * 24 * 3600
            table_name = f"candles_{interval}"
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute(f"DELETE FROM {table_name} WHERE timestamp < ?", (cutoff_ts,))
                deleted = cur.rowcount
                if deleted > 0:
                    print(f"   🧹 {interval}: {deleted}개 삭제 (< {retention_days}일)")
                conn.commit()
        except Exception as e:
            print(f"   ⚠️ {interval} pruning 오류: {e}")

        for idx, symbol in enumerate(tickers, start=1):
            print(f"   [{idx}/{len(tickers)}] {symbol}...", end="\r")
            df = fetch_data_yfinance(symbol, interval, days_back)
            if not df.empty:
                save_to_db(interval, {symbol: df})
                total_saved += len(df)
            time.sleep(0.2)

    print(f"\n✨ 수집 완료! 총 {total_saved}개 캔들 추가/갱신")


if __name__ == "__main__":
    main()

