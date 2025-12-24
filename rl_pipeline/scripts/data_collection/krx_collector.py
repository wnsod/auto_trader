import sys
import os
import sqlite3
import time
import pandas as pd
from datetime import datetime, timedelta
import subprocess

# ==========================================
# 📦 라이브러리 자동 설치 및 로드
# ==========================================
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import yfinance as yf
except ImportError:
    print("📦 yfinance 설치 중...")
    install_package("yfinance")
    import yfinance as yf

# yfinance 최적화 설정
try:
    yf.pdr_override()
except:
    pass

try:
    from pykrx import stock
except ImportError:
    print("📦 pykrx 설치 중...")
    install_package("pykrx")
    from pykrx import stock

# ==========================================
# 🇰🇷 KRX 주식 데이터 수집기 (Unified with yfinance)
# ==========================================
# 1. 설정 로드
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from rl_pipeline.core.env import config

DB_PATH = os.getenv('RL_DB_PATH', config.RL_DB)

# 2. 유틸리티 함수
def get_krx300_tickers():
    """KRX300 종목 리스트 (pykrx 사용)"""
    try:
        tickers = stock.get_index_portfolio_deposit_file("1028") 
        if not tickers:
            tickers = stock.get_market_ticker_list(market="KOSPI")[:200]
        return tickers
    except Exception:
        return []

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candles (
        symbol TEXT,
        interval TEXT,
        timestamp INTEGER,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (symbol, interval, timestamp)
    )
    """)
    conn.commit()
    conn.close()

def get_last_timestamp(symbol, interval):
    """DB에서 마지막 캔들의 타임스탬프 조회"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp) FROM candles 
            WHERE symbol = ? AND interval = ?
        """, (symbol, interval))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return result[0]
        return None
    except Exception:
        return None

def save_to_db(df, symbol, interval):
    if df.empty:
        return 0
    
    try:
        # 컬럼 이름 소문자 통일
        df.columns = [c.lower() for c in df.columns]
        
        # 필수 컬럼 확인
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return 0

        # 숫자형 변환 및 결측치 제거
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required)
        
        # 명시적 float 변환
        for col in required:
            df[col] = df[col].astype(float)
            
    except Exception as e:
        print(f"   ❌ 데이터 타입 변환 실패: {e}")
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    data_to_insert = []
    for idx, row in df.iterrows():
        # Timestamp 처리
        if isinstance(idx, datetime):
            ts = int(idx.timestamp())
        else:
            try:
                ts = int(pd.to_datetime(idx).timestamp())
            except:
                continue
                
        data_to_insert.append((
            symbol, interval, ts,
            row['open'], row['high'], row['low'], row['close'], row['volume']
        ))
    
    # 증분 수집이므로 INSERT OR IGNORE를 사용하여 기존 데이터 보존 (중복만 무시)
    # 하지만 수정 데이터 반영을 위해 REPLACE가 나을 수도 있음 -> 여기서는 안전하게 REPLACE 사용
    # (Yahoo 데이터가 수정될 수도 있으므로 덮어쓰기 허용)
    cursor.executemany('''
        INSERT OR REPLACE INTO candles (symbol, interval, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    
    conn.commit()
    conn.close()
    return len(data_to_insert)

def prune_old_data(interval, keep_days):
    """오래된 데이터 삭제 (Retention Policy 적용)"""
    if keep_days <= 0:
        return 0
        
    try:
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cutoff_ts = int(cutoff_date.timestamp())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 해당 인터벌의 오래된 데이터 일괄 삭제
        cursor.execute("DELETE FROM candles WHERE interval = ? AND timestamp < ?", (interval, cutoff_ts))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"   🧹 {interval}: {keep_days}일 이전 데이터 {deleted_count}개 삭제됨")
            
        return deleted_count
    except Exception as e:
        print(f"   ⚠️ 데이터 정리 중 오류: {e}")
        return 0

def fetch_data_yfinance(symbol, interval, days_back):
    """yfinance 증분 수집"""
    
    # 최대 재시도 횟수
    MAX_RETRY = 3
    
    # 1. DB에서 마지막 타임스탬프 확인
    last_ts = get_last_timestamp(symbol, interval)
    
    end_date = datetime.utcnow()  # UTC 기준으로 고정 (로컬 타임존 영향 최소화)
    
    if last_ts:
        # 마지막 데이터가 있으면 그 다음 날부터 수집
        last_date = datetime.fromtimestamp(last_ts)
        start_date = last_date + timedelta(days=1)
        
        # 만약 start_date가 미래라면 수집 불필요
        if start_date > end_date:
            return pd.DataFrame()
            
        print(f"   🔄 증분 수집: {last_date.strftime('%Y-%m-%d')} 이후 데이터 요청")
    else:
        # 데이터가 없으면 days_back 만큼 수집
        start_date = end_date - timedelta(days=days_back)
    
    str_start = start_date.strftime("%Y-%m-%d")
    str_end = end_date.strftime("%Y-%m-%d")
    
    # yfinance 인터벌 매핑
    yf_interval = interval
    
    # yfinance 분봉 기간 제한 정책 고려 (Start Date 조정)
    if interval in ['5m', '15m', '30m']:
        limit_date = end_date - timedelta(days=59)
        if start_date < limit_date:
            print(f"   ⚠️ {interval} 제한: {str_start} -> {limit_date.strftime('%Y-%m-%d')} 로 조정 (최대 60일)")
            str_start = limit_date.strftime("%Y-%m-%d")
            
    elif interval in ['60m', '1h']:
        limit_date = end_date - timedelta(days=729)
        if start_date < limit_date:
            print(f"   ⚠️ 60분봉 제한: {str_start} -> {limit_date.strftime('%Y-%m-%d')} 로 조정")
            str_start = limit_date.strftime("%Y-%m-%d")
    
    def _normalize(df):
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass
        df.columns = [c.lower() for c in df.columns]
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    # 시도: KS -> KQ, start/end 기반 → 실패 시 period 기반으로 재시도
    for attempt in range(1, MAX_RETRY + 1):
        df = pd.DataFrame()
        try:
            # 코스피(.KS) 우선 시도
            yf_symbol = f"{symbol}.KS"
            df = yf.download(yf_symbol, start=str_start, end=str_end, interval=yf_interval, progress=False, auto_adjust=True)
            
            # 데이터가 없으면 코스닥(.KQ) 시도
            if df.empty:
                yf_symbol = f"{symbol}.KQ"
                df = yf.download(yf_symbol, start=str_start, end=str_end, interval=yf_interval, progress=False, auto_adjust=True)
            
            # 여전히 없으면 period 기반으로 재시도 (yfinance가 자체 기간 제한 처리)
            if df.empty:
                period_days = min(days_back, 59 if interval in ['5m', '15m', '30m'] else days_back)
                period_str = f"{period_days}d"
                
                yf_symbol = f"{symbol}.KS"
                df = yf.download(yf_symbol, period=period_str, interval=yf_interval, progress=False, auto_adjust=True)
                
                if df.empty:
                    yf_symbol = f"{symbol}.KQ"
                    df = yf.download(yf_symbol, period=period_str, interval=yf_interval, progress=False, auto_adjust=True)
            
            if not df.empty:
                df = _normalize(df)
                if not df.empty:
                    return df
            
            print(f"   ⚠️ {symbol}/{interval} 재시도 {attempt}/{MAX_RETRY} (데이터 없음)")
            time.sleep(0.5 * attempt)  # 점진적 대기
            
        except Exception as e:
            print(f"   ❌ yfinance 수집 오류 ({symbol}/{interval}) 시도 {attempt}/{MAX_RETRY}: {e}")
            time.sleep(0.5 * attempt)
    
    return pd.DataFrame()

def prune_old_data(interval):
    """오래된 데이터 삭제 (Retention Policy 적용)"""
    try:
        # 설정된 보관 기간 가져오기 (기본값: 1년)
        retention_days = int(os.getenv(f'RETENTION_DAYS_{interval.upper()}', '365'))
        
        # 삭제 기준 타임스탬프 계산
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_ts = int(cutoff_date.timestamp())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 해당 인터벌의 오래된 데이터 삭제
        cursor.execute("DELETE FROM candles WHERE interval = ? AND timestamp < ?", (interval, cutoff_ts))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"   🧹 {interval} 정리: {retention_days}일 이전 데이터 {deleted_count}개 삭제됨")
            
    except Exception as e:
        print(f"   ⚠️ 데이터 정리 중 오류 ({interval}): {e}")

def main():
    print("🚀 [KRX Collector] Unified Mode (All yfinance) + Incremental")
    create_table()
    
    target_str = os.getenv('TARGET_COINS', 'ALL')
    if target_str == 'ALL' or not target_str:
        tickers = get_krx300_tickers()
    else:
        tickers = [t.strip() for t in target_str.split(',') if t.strip()]
    
    env_intervals = os.getenv('CANDLE_INTERVALS', '1d,1w')
    intervals = [i.strip() for i in env_intervals.split(',')]
    
    total_saved = 0
    
    for interval in intervals:
        print(f"\n⏰ {interval} 수집 중...")
        
        # 1. 데이터 정리 (먼저 정리해서 DB 가볍게 유지)
        prune_old_data(interval)
        
        days_back = int(os.getenv(f'DAYS_BACK_{interval.upper()}', '365'))
        
        count = 0
        for ticker in tickers:
            count += 1
            print(f"   [{count}/{len(tickers)}] {ticker}...", end='\r')
            
            df = fetch_data_yfinance(ticker, interval, days_back)
            saved = save_to_db(df, ticker, interval)
            total_saved += saved
            
            time.sleep(0.2)
            
    print(f"\n✨ 수집 완료! 총 {total_saved}개 캔들 추가/갱신됨.")

if __name__ == "__main__":
    main()
