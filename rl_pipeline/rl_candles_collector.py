import sys
sys.path.insert(0, '/workspace/')  # 절대 경로 추가

import sqlite3
import asyncio
import aiohttp
import pandas as pd
import os
import time
from datetime import datetime, timedelta, timezone
from threading import Thread
from queue import Queue
from collections import defaultdict

# 데이터베이스 경로 설정 (env에서 가져오기)
from rl_pipeline.core.env import config
DB_PATH = config.RL_DB  # CANDLES_DB_PATH
INTERVALS = ['15m', '30m', '240m', '1d']
KST = timezone(timedelta(hours=9))

# 🧪 테스트용 코인 목록 (None이면 전체 코인 수집, 리스트면 지정된 코인만 수집)
TEST_COINS = ['BTC', 'ETH', 'XRP', 'DOGE', 'SOL', 'ADA', 'DOT', 'LINK', 'AVAX', 'BNB']  # 메이저 코인 10개

# 🚀 최적화된 동시성 제어 설정
MAX_CONCURRENT_REQUESTS = 50  # 동시 요청 수 증가
REQUEST_TIMEOUT = 15  # 요청 타임아웃
RETRY_ATTEMPTS = 2  # 재시도 횟수
RETRY_DELAY = 0.5  # 재시도 간격
RATE_LIMIT_DELAY = 1.0  # 429 에러 시 대기 시간
MAX_RATE_LIMIT_RETRIES = 3  # 429 에러 최대 재시도 횟수

# 동시성 제어를 위한 세마포어
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
save_queue = asyncio.Queue(maxsize=3000)  # 큐 크기 증가

# 429 rate limit 로그 요약용
rate_limit_counter = defaultdict(int)
rate_limit_last_print = defaultdict(float)

# 🚀 실패한 요청 추적용 (간소화)
failed_requests = set()  # (coin, interval) 튜플로 저장
failed_requests_lock = asyncio.Lock()  # 스레드 안전을 위한 락

# 🚀 성능 모니터링
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_candles = 0
        self.last_progress_time = time.time()
        self.progress_interval = 10  # 10초마다 진행률 출력
        
    def log_request(self, success: bool, candle_count: int = 0):
        self.request_count += 1
        if success:
            self.success_count += 1
            self.total_candles += candle_count
        else:
            self.error_count += 1
        
        # 진행률 출력
        current_time = time.time()
        if current_time - self.last_progress_time >= self.progress_interval:
            self._print_progress()
            self.last_progress_time = current_time
    
    def _print_progress(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rps = self.request_count / elapsed
            cps = self.total_candles / elapsed
            success_rate = self.success_count / self.request_count if self.request_count > 0 else 0
            print(f"📈 진행률: {self.request_count:,} 요청, {self.total_candles:,} 캔들, "
                  f"{rps:.1f} req/s, {cps:.0f} 캔들/s, 성공률: {success_rate:.1%}")
    
    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            'elapsed_time': elapsed,
            'requests_per_second': self.request_count / elapsed if elapsed > 0 else 0,
            'success_rate': self.success_count / self.request_count if self.request_count > 0 else 0,
            'total_candles': self.total_candles,
            'candles_per_second': self.total_candles / elapsed if elapsed > 0 else 0
        }

# 전역 성능 모니터
performance_monitor = PerformanceMonitor()

# 🚀 Rate limiting을 위한 딜레이 관리
class RateLimiter:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 0.02  # 20ms 간격 (초당 50 요청) - 최적화됨
        self.rate_limit_until = 0
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        # Rate limit 대기
        if current_time < self.rate_limit_until:
            wait_time = self.rate_limit_until - current_time
            print(f"⏳ Rate limit 대기: {wait_time:.1f}초")
            await asyncio.sleep(wait_time)
            current_time = time.time()
        
        # 최소 간격 대기
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def set_rate_limit(self, retry_after=None):
        """429 에러 시 rate limit 설정"""
        if retry_after:
            self.rate_limit_until = time.time() + retry_after
        else:
            self.rate_limit_until = time.time() + RATE_LIMIT_DELAY

# 전역 rate limiter
rate_limiter = RateLimiter()

# 테이블 생성
def create_table():
    # 기존 DB 파일이 있다면 삭제하고 새로 생성
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 새로운 테이블 생성 (실제 사용되는 컬럼들만)
    cursor.execute("""
    CREATE TABLE candles (
        -- 🏷️ 기본 식별자 (3개)
        coin TEXT,
        interval TEXT,
        timestamp INTEGER,
        -- 💰 기본 OHLCV (4개)
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        -- 📉 핵심 오실레이터 (2개)
        rsi REAL,
        mfi REAL,
        -- 📊 핵심 트렌드 (2개)
        macd REAL,
        macd_signal REAL,
        -- 🌐 핵심 볼린저밴드 (5개)
        bb_upper REAL,
        bb_middle REAL,
        bb_lower REAL,
        bb_position REAL,
        bb_width REAL,
        -- 📈 핵심 추세/변동성 (3개)
        atr REAL,
        ma20 REAL,
        adx REAL,
        -- 📊 핵심 거래량 (1개)
        volume_ratio REAL,
        -- ⚠️ 핵심 리스크 (1개)
        risk_score REAL,
        -- 🧠 핵심 파동 (2개)
        wave_phase TEXT,
        confidence REAL,
        -- 🔄 핵심 파동 분석 (3개)
        zigzag_direction REAL,
        zigzag_pivot_price REAL,
        wave_progress REAL,
        -- 🎯 핵심 패턴 분석 (2개)
        pattern_type TEXT,
        pattern_confidence REAL,
        -- 🧠 핵심 통합 분석 (3개)
        volatility_level TEXT,
        risk_level TEXT,
        integrated_direction TEXT,
        -- 🚀 구조 점수 (1개)
        structure_score REAL,
        -- 🚀 심리도 분석 (2개)
        sentiment REAL,
        sentiment_label TEXT,
        PRIMARY KEY (coin, interval, timestamp)
    )
    """)
    
    # 🚀 인덱스 추가로 조회 성능 향상
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coin_interval ON candles(coin, interval)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON candles(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coin_timestamp ON candles(coin, timestamp)')
    
    conn.commit()
    conn.close()

# 🚀 최적화된 캔들 저장 워커 (비동기 버전) - 최근 100개만 유지
async def candle_saver_worker(save_queue):
    """🚀 최적화된 캔들 저장 워커 - 최근 100개만 유지"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    total_saved = 0
    batch_data = []
    batch_size = 1000  # 배치 크기 증가

    while True:
        try:
            item = await save_queue.get()
            if item is None:
                # 마지막 배치 처리
                if batch_data:
                    cursor.executemany('''
                        INSERT OR REPLACE INTO candles (
                            coin, interval, timestamp, open, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch_data)
                    conn.commit()
                    total_saved += len(batch_data)
                print(f"💾 총 저장된 캔들: {total_saved:,}개")
                break

            coin, interval, candles = item

            # 배치 데이터 추가
            for c in candles:
                # Unix timestamp로 변환
                timestamp = int(pd.to_datetime(c['timestamp'], format='%Y-%m-%dT%H:%M:%S').timestamp())
                
                batch_data.append((
                    coin, interval, timestamp,
                    c['open'], c['high'], c['low'], c['close'], c['volume']
                ))

            # 배치 크기에 도달하면 DB에 저장
            if len(batch_data) >= batch_size:
                cursor.executemany('''
                    INSERT OR REPLACE INTO candles (
                        coin, interval, timestamp, open, high, low, close, volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch_data)
                conn.commit()
                total_saved += len(batch_data)
                batch_data.clear()

        except Exception as e:
            print(f"⚠️ 저장 워커 오류: {e}")
            continue

    conn.close()
    
def build_url(interval):
    interval_map = {
        "15m": "https://api.bithumb.com/v1/candles/minutes/15",
        "30m": "https://api.bithumb.com/v1/candles/minutes/30",
        "240m": "https://api.bithumb.com/v1/candles/minutes/240",
        "1d": "https://api.bithumb.com/v1/candles/days"
    }
    return interval_map.get(interval)

async def get_all_coins(session: aiohttp.ClientSession):
    """🚀 최적화된 코인 목록 조회"""
    url = "https://api.bithumb.com/public/ticker/ALL_KRW"
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Rate limit 대기
            await rate_limiter.wait_if_needed()
            
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    coins = [coin for coin in data["data"] if coin != "date"]
                    print(f"✅ 코인 목록 조회 성공: {len(coins)}개")
                    return coins
                elif response.status == 429:
                    # Rate limit 처리
                    retry_after = None
                    try:
                        retry_after_header = response.headers.get('Retry-After')
                        if retry_after_header:
                            retry_after = int(retry_after_header)
                    except:
                        pass
                    
                    rate_limiter.set_rate_limit(retry_after)
                    print(f"⚠️ 코인 목록 조회 Rate limit (시도 {attempt + 1}/{RETRY_ATTEMPTS})")
                    
                    if attempt < RETRY_ATTEMPTS - 1:
                        backoff_delay = RETRY_DELAY * (2 ** attempt)
                        await asyncio.sleep(backoff_delay)
                        await rate_limiter.wait_if_needed()
                    continue
                else:
                    print(f"⚠️ 코인 목록 조회 실패 (HTTP {response.status})")
        except Exception as e:
            print(f"⚠️ 코인 목록 조회 오류 (시도 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                backoff_delay = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(backoff_delay)
    
    return []

async def fetch_candles(session: aiohttp.ClientSession, coin: str, interval: str, from_ts: int, to_ts: int):
    """🚀 최적화된 캔들 데이터 조회 (재시도 로직 포함) - 과거 수집 지원"""
    url = build_url(interval)
    if not url:
        return []

    to_dt = datetime.fromtimestamp(to_ts, tz=KST)
    params = {
        "market": f"KRW-{coin}",
        "count": 200,
        "to": to_dt.strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    key = f"{coin}/{interval}"

    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Rate limit 대기
            await rate_limiter.wait_if_needed()
            
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.get(url, params=params, timeout=timeout) as response:
                if response.status == 429:
                    rate_limit_counter[key] += 1
                    now = time.time()
                    if rate_limit_counter[key] % 10 == 1 or now - rate_limit_last_print[key] > 10:
                        rate_limit_last_print[key] = now
                    # Rate Limit 대응: 더 긴 대기 시간 적용
                    wait_time = min(3.0, rate_limit_counter[key] * 1.5)  # 최대 3초까지 증가
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    rate_limit_counter[key] = 0  # 정상 응답시 카운터 초기화
                
                if response.status == 200:
                    data = await response.json()
                    if not isinstance(data, list):
                        performance_monitor.log_request(False)
                        return []

                    candles = []
                    seen_timestamps = set()
                    
                    for candle in data:
                        try:
                            timestamp = candle.get('candle_date_time_kst')
                            if timestamp in seen_timestamps or not timestamp:
                                continue  # 중복 또는 잘못된 타임스탬프는 제외
                            seen_timestamps.add(timestamp)

                            candles.append({
                                'timestamp': timestamp,
                                'open': float(candle.get('opening_price', 0)),
                                'high': float(candle.get('high_price', 0)),
                                'low': float(candle.get('low_price', 0)),
                                'close': float(candle.get('trade_price', 0)),
                                'volume': round(float(candle.get('candle_acc_trade_price', 0)), 4)
                            })
                        except (ValueError, KeyError) as e:
                            continue

                    performance_monitor.log_request(True, len(candles))
                    return candles
                else:
                    pass
                    
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            pass
        
        if attempt < RETRY_ATTEMPTS - 1:
            # 지수 백오프
            backoff_delay = RETRY_DELAY * (2 ** attempt)
            await asyncio.sleep(backoff_delay)
    
    performance_monitor.log_request(False)
    
    # 실패한 요청 기록
    async with failed_requests_lock:
        failed_requests.add((coin, interval))
    
    return []

interval_minutes = {'15m': 15, '30m': 30, '240m': 240, '1d': 1440}

def split_into_chunks(from_ts, to_ts, gap_sec):
    chunks = []
    while to_ts > from_ts:
        end = to_ts
        start = max(from_ts, to_ts - gap_sec * 200)
        chunks.append((start, end))
        to_ts = start - gap_sec
    return chunks

# 🚀 최적화된 전체 데이터 수집
async def fetch_all(save_queue):
    """🚀 최적화된 전체 데이터 수집"""

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS * 3,
        limit_per_host=MAX_CONCURRENT_REQUESTS * 2,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={'User-Agent': 'Mozilla/5.0 (compatible; TradingBot/2.0)'}
    ) as session:
        coins = await get_all_coins(session)
        if not coins:
            return
        
        # 🧪 테스트용 코인 필터링
        if TEST_COINS is not None:
            # TEST_COINS에 지정된 코인만 필터링 (대소문자 구분 없이)
            coins_dict = {coin.upper(): coin for coin in coins}  # 원본 코인명 유지
            filtered_coins = []
            not_found = []
            
            for test_coin in TEST_COINS:
                test_coin_upper = test_coin.upper()
                if test_coin_upper in coins_dict:
                    filtered_coins.append(coins_dict[test_coin_upper])
                else:
                    not_found.append(test_coin)
            
            coins = filtered_coins
            if not_found:
                print(f"⚠️ 다음 코인을 찾을 수 없습니다: {', '.join(not_found)}")
            print(f"🧪 테스트 모드: 지정된 코인 {len(coins)}개만 수집 ({', '.join(coins)})")
        elif len(coins) > 10:
            # TEST_COINS가 None이고 코인이 많으면 처음 10개만 (기존 로직 유지)
            coins = coins[:10]
            print(f"🧪 테스트 모드: 코인 10개로 제한 ({', '.join(coins)})")


        # 과거 수집 기간은 인터벌별로 상이: 15m/30m/240m → 45일, 1d → 60일
        total_tasks = 0

        end_time_global = datetime.now(KST)
        for interval in INTERVALS:
            gap_sec = interval_minutes[interval] * 60
            print(f"\n⏰ {interval} 캔들 수집 중...")

            interval_tasks = []
            # 인터벌별 수집 기간 설정
            days_back = 60 if interval == '1d' else 45
            start_time = end_time_global - timedelta(days=days_back)
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time_global.timestamp()) - 300
            for coin in coins:
                chunks = split_into_chunks(start_ts, end_ts, gap_sec)
                
                for start, end in chunks:
                    task = limited_fetch(session, coin, interval, start, end, save_queue)
                    interval_tasks.append(task)

            total_tasks += len(interval_tasks)
            print(f"📋 {interval}: {len(interval_tasks):,}개 태스크 생성")

            # 🚀 더 큰 배치 단위로 실행 (메모리 효율성)
            batch_size = 200  # 배치 크기 증가
            completed_tasks = 0

            for i in range(0, len(interval_tasks), batch_size):
                batch = interval_tasks[i:i + batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)

                # 결과 처리
                for result in results:
                    if isinstance(result, Exception):
                        print(f"⚠️ 태스크 실행 오류: {result}")
                    elif result:
                        pass  # 성공한 경우 (이미 큐에 추가됨)

                completed_tasks += len(batch)
                if completed_tasks % (batch_size * 5) == 0:  # 5배치마다 진행률 출력
                    print(f"📈 {interval} 진행률: {completed_tasks:,}/{len(interval_tasks):,} ({completed_tasks/len(interval_tasks)*100:.1f}%)")


        # 실패한 요청들 처리 (파일 저장 제거)
        if failed_requests:
            logger.warning(f"⚠️ 실패한 요청들 ({len(failed_requests)}개): {failed_requests}")
        else:
            logger.info("✅ 모든 요청 성공!")

        stats = performance_monitor.get_stats()

        # 🚀 성능 평가
        if stats['requests_per_second'] > 50:
            pass
        elif stats['requests_per_second'] > 35:
            pass
        else:
            pass
            
async def limited_fetch(session: aiohttp.ClientSession, coin: str, interval: str, start_ts: int, end_ts: int, save_queue):
    """🚀 최적화된 제한된 페치 (세마포어 제어)"""
    async with semaphore:  # 세마포어로 동시 요청 수 제한
        candles = await fetch_candles(session, coin, interval, start_ts, end_ts)
        await asyncio.sleep(0.02)  # 최소 지연 시간 (50 동시 요청 대응)
        if candles:
            await save_queue.put((coin, interval, candles))
        return candles

# 메인 실행 함수 (비동기 버전)
async def main():
    
    # 테이블 생성 (기존 DB 파일 삭제 후 새로 생성)
    create_table()
    
    
    # 🚀 성능 예측
    estimated_coins = 400  # 예상 코인 수
    estimated_requests = estimated_coins * len(INTERVALS) * 8  # 4개 인터벌 × 8회 요청/코인
    estimated_time = estimated_requests / 50  # 초당 50 요청 가정

    # 🧪 이벤트 루프에 바인딩된 save_queue 생성
    save_queue = asyncio.Queue(maxsize=3000)

    # 🚀 저장 워커 시작
    worker = asyncio.create_task(candle_saver_worker(save_queue))

    try:
        # 🚀 데이터 수집 실행
        await fetch_all(save_queue)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass

    # 🚀 워커 종료 신호
    await save_queue.put(None)
    await worker

if __name__ == "__main__":
    asyncio.run(main())