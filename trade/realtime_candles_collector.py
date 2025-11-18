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

DB_PATH = '/workspace/data_storage/realtime_candles.db'
INTERVALS = ['15m', '30m', '240m', '1d']
REQUEST_COUNT = {'15m': 100, '30m': 100, '240m': 100, '1d': 100}

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

# 🚀 실패한 요청 추적용
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
        self.progress_interval = 5  # 5초마다 진행률 출력
        
    def log_request(self, success: bool, candle_count: int = 0):
        self.request_count += 1
        if success:
            self.success_count += 1
            self.total_candles += candle_count
        else:
            self.error_count += 1
        
        # 진행률 출력 (제거됨)
        current_time = time.time()
        if current_time - self.last_progress_time >= self.progress_interval:
            # self._print_progress()  # 제거됨
            self.last_progress_time = current_time
    
    def _print_progress(self):
        # 진행률 출력 제거됨
        pass
    
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
        self.min_interval = 0.02  # 20ms 간격 (초당 50 요청)
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
        # print(f"🗑️ 기존 DB 파일 삭제: {DB_PATH}")  # 제거됨
    
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
        -- ⚠️ 핵심 리스크 (2개)
        volatility REAL,
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
        -- 🚀 레짐 분석 (4개)
        regime_stage INTEGER,
        regime_label TEXT,
        regime_confidence REAL,
        regime_transition_prob REAL,
        PRIMARY KEY (coin, interval, timestamp)
    )
    """)
    
    # 🚀 인덱스 추가로 조회 성능 향상
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coin_interval ON candles(coin, interval)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON candles(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coin_timestamp ON candles(coin, timestamp)')
    
    conn.commit()
    conn.close()
    # print("✅ 새로운 candles 테이블 생성 완료")  # 제거됨



# 코인명 가져오기 (비동기 버전)
async def get_all_coins(session: aiohttp.ClientSession):
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
                    # print(f"✅ 코인 목록 조회 성공: {len(coins)}개")  # 제거됨
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
                    # print(f"⚠️ 코인 목록 조회 Rate limit (시도 {attempt + 1}/{RETRY_ATTEMPTS})")  # 제거됨
                    
                    if attempt < RETRY_ATTEMPTS - 1:
                        backoff_delay = RETRY_DELAY * (2 ** attempt)
                        await asyncio.sleep(backoff_delay)
                        await rate_limiter.wait_if_needed()
                    continue
                else:
                    # print(f"⚠️ 코인 목록 조회 실패 (HTTP {response.status})")  # 제거됨
                    pass
        except Exception as e:
            # print(f"⚠️ 코인 목록 조회 오류 (시도 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")  # 제거됨
            if attempt < RETRY_ATTEMPTS - 1:
                backoff_delay = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(backoff_delay)
    
    return []

# 캔들 가져오기 (비동기 버전)
def build_url(interval):
    interval_map = {
        
        "15m": "https://api.bithumb.com/v1/candles/minutes/15",
        "30m": "https://api.bithumb.com/v1/candles/minutes/30",
        "240m": "https://api.bithumb.com/v1/candles/minutes/240",
        "1d": "https://api.bithumb.com/v1/candles/days",
        "1w": "https://api.bithumb.com/v1/candles/weeks"
    }
    return interval_map.get(interval)

async def fetch_candles(session: aiohttp.ClientSession, coin: str, interval: str, count=200, oldest_timestamp=None):
    """🚀 최적화된 캔들 데이터 조회 (재시도 로직 포함) - 증분 업데이트 지원"""
    url = build_url(interval)
    if not url:
        return []

    params = {
        "market": f"KRW-{coin}",
        "count": count
    }
    
    # 🚀 증분 업데이트: 기존 데이터보다 새로운 데이터만 필터링
    if oldest_timestamp:
        # 기존 데이터의 가장 오래된 타임스탬프 이후의 데이터만 필요
        # print(f"🔄 {coin}/{interval}: 기존 데이터 이후의 새로운 데이터만 수집 (기존 최고: {datetime.fromtimestamp(oldest_timestamp)})")  # 제거됨
        pass

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
                        # print(f"⚠️ HTTP 429 (rate limit) for {key}, 최근 {rate_limit_counter[key]}회 반복 중...")  # 제거됨
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
                    # print(f"⚠️ 캔들 조회 실패 {coin}-{interval} (HTTP {response.status})")  # 제거됨
                    pass
                    
        except asyncio.TimeoutError:
            # print(f"⚠️ 캔들 조회 타임아웃 {coin}-{interval} (시도 {attempt + 1}/{RETRY_ATTEMPTS})")  # 제거됨
            pass
        except Exception as e:
            # print(f"⚠️ 캔들 조회 오류 {coin}-{interval} (시도 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")  # 제거됨
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
                # print(f"💾 총 저장된 캔들: {total_saved:,}개")  # 제거됨
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
            # print(f"⚠️ 저장 워커 오류: {e}")  # 제거됨
            continue

    conn.close()

# 🚀 인터벌별 캔들 정리 함수 (성능 최적화)
def cleanup_candles_by_interval():
    """🚀 인터벌별로 다른 정리 정책 적용"""
    # print("🧹 인터벌별 캔들 데이터 정리 시작...")  # 제거됨
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 삭제 전 전체 캔들 수 확인
        cursor.execute("SELECT COUNT(*) FROM candles")
        total_before = cursor.fetchone()[0]
        
        total_deleted = 0
        
        # 1. 단기 인터벌 (15m, 30m): 7일 이전 데이터 삭제
        short_intervals = ['15m', '30m']
        current_timestamp = int(datetime.now().timestamp())
        cutoff_timestamp = current_timestamp - (7 * 24 * 3600)  # 7일
        
        for interval in short_intervals:
            cursor.execute("""
                DELETE FROM candles 
                WHERE interval = ? AND timestamp < ?
            """, (interval, cutoff_timestamp))
            
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            
            if deleted_count > 0:
                # print(f"  🧹 {interval}: {deleted_count:,}개 삭제 (7일 이전)")  # 제거됨
                pass
        
        # 2. 장기 인터벌 (240m, 1d, 1w): 최근 100개만 유지
        long_intervals = ['240m', '1d', '1w']
        
        for interval in long_intervals:
            # 각 코인별로 최근 100개만 유지
            cursor.execute("SELECT DISTINCT coin FROM candles WHERE interval = ?", (interval,))
            coins = cursor.fetchall()
            
            for (coin,) in coins:
                cursor.execute("""
                    DELETE FROM candles 
                    WHERE coin = ? AND interval = ? 
                    AND timestamp NOT IN (
                        SELECT timestamp FROM candles 
                        WHERE coin = ? AND interval = ?
                        ORDER BY timestamp DESC 
                        LIMIT 100
                    )
                """, (coin, interval, coin, interval))
                
                deleted_count = cursor.rowcount
                total_deleted += deleted_count
                
                if deleted_count > 0:
                    # print(f"  🧹 {coin}/{interval}: {deleted_count:,}개 삭제 (최근 100개만 유지)")  # 제거됨
                    pass
        
        # 삭제 후 전체 캔들 수 확인
        cursor.execute("SELECT COUNT(*) FROM candles")
        total_after = cursor.fetchone()[0]
        
        conn.commit()
        
        # print(f"✅ 정리 완료: 총 {total_deleted:,}개 캔들 삭제")  # 제거됨
        # print(f"📊 DB 크기: {total_before:,}개 → {total_after:,}개 캔들")  # 제거됨
        
        # 인터벌별 캔들 수 통계
        cursor.execute("""
            SELECT interval, COUNT(*) as count 
            FROM candles 
            GROUP BY interval 
            ORDER BY count DESC
        """)
        
        interval_stats = cursor.fetchall()
        if interval_stats:
            # print(f"📈 인터벌별 캔들 수:")  # 제거됨
            # for interval, count in interval_stats:
            #     print(f"  ⏰ {interval}: {count:,}개")  # 제거됨
            pass
        
    except Exception as e:
        # print(f"⚠️ 캔들 정리 중 오류: {e}")  # 제거됨
        pass
    finally:
        conn.close()



# 🚀 최적화된 제한된 페치 (세마포어 제어)
async def limited_fetch(session: aiohttp.ClientSession, coin: str, interval: str, save_queue):
    """🚀 최적화된 제한된 페치 (세마포어 제어)"""
    async with semaphore:  # 세마포어로 동시 요청 수 제한
        candles = await fetch_candles(session, coin, interval, REQUEST_COUNT[interval])
        await asyncio.sleep(0.1)  # 지연 시간 추가 (Rate Limit 대응)
        if candles:
            await save_queue.put((coin, interval, candles))
        return candles

# 🚀 최적화된 전체 데이터 수집
async def fetch_all(save_queue):
    """🚀 최적화된 전체 데이터 수집"""
    # print("🚀 데이터 수집 시작")  # 제거됨

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
            # print("❌ 코인 목록 조회 실패")  # 제거됨
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

        # print(f"📊 수집 대상: {len(coins)}개 코인")  # 제거됨

        total_tasks = 0

        for interval in INTERVALS:
            # print(f"\n⏰ {interval} 캔들 수집 중...")  # 제거됨

            interval_tasks = []
            for coin in coins:
                task = limited_fetch(session, coin, interval, save_queue)
                interval_tasks.append(task)

            total_tasks += len(interval_tasks)
            # print(f"📋 {interval}: {len(interval_tasks):,}개 태스크 생성")  # 제거됨

            batch_size = 200  # 배치 크기 증가
            completed_tasks = 0

            for i in range(0, len(interval_tasks), batch_size):
                batch = interval_tasks[i:i + batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        print(f"⚠️ 태스크 실행 오류: {result}")

                completed_tasks += len(batch)
                if completed_tasks % (batch_size * 2) == 0:
                    # print(f"📈 {interval} 진행률: {completed_tasks:,}/{len(interval_tasks):,} ({completed_tasks/len(interval_tasks)*100:.1f}%)")  # 제거됨
                    pass

        # print(f"\n✅ 수집 완료 (총 {total_tasks:,}개 태스크)")  # 제거됨

        # 실패한 요청들 출력
        if failed_requests:
            # print(f"\n⚠️ 실패한 요청들 ({len(failed_requests)}개):")  # 제거됨
            # for coin, interval in sorted(failed_requests):
            #     print(f"  - {coin}-{interval}")  # 제거됨
            
            # 실패한 요청들을 파일로 저장
            failed_file = os.path.join(os.path.dirname(DB_PATH), 'failed_requests_realtime.txt')
            with open(failed_file, 'w', encoding='utf-8') as f:
                for coin, interval in sorted(failed_requests):
                    f.write(f"{coin},{interval}\n")
            # print(f"📁 실패한 요청 목록 저장: {failed_file}")  # 제거됨
        else:
            # print(f"\n✅ 모든 요청 성공!")  # 제거됨
            pass

        stats = performance_monitor.get_stats()
        # print(f"\n📊 성능 통계:")  # 제거됨
        # print(f"- 총 요청 수: {performance_monitor.request_count:,}")  # 제거됨
        # print(f"- 성공률: {stats['success_rate']:.1%}")  # 제거됨
        # print(f"- 초당 요청 수: {stats['requests_per_second']:.1f}")  # 제거됨
        # print(f"- 총 캔들 수: {stats['total_candles']:,}")  # 제거됨
        # print(f"- 초당 캔들 수: {stats['candles_per_second']:.1f}")  # 제거됨
        # print(f"- 소요 시간: {stats['elapsed_time']:.1f}초")  # 제거됨

        # 🚀 성능 평가
        if stats['requests_per_second'] > 20:
            # print(f"🚀 우수한 성능! 초당 {stats['requests_per_second']:.1f} 요청")  # 제거됨
            pass
        elif stats['requests_per_second'] > 15:
            # print(f"✅ 좋은 성능! 초당 {stats['requests_per_second']:.1f} 요청")  # 제거됨
            pass
        else:
            # print(f"⚠️ 개선 필요: 초당 {stats['requests_per_second']:.1f} 요청")  # 제거됨
            pass

# 메인 실행 함수 (비동기 버전)
async def main():
    # print("🚀 실시간 캔들 수집기 시작")  # 제거됨
    
    # 테이블 생성 (기존 DB 파일 삭제 후 새로 생성)
    create_table()
    
    # print(f"🚀 최적화 설정:")  # 제거됨
    # print(f"- 최대 동시 요청: {MAX_CONCURRENT_REQUESTS}")  # 제거됨
    # print(f"- 요청 타임아웃: {REQUEST_TIMEOUT}초")  # 제거됨
    # print(f"- 재시도 횟수: {RETRY_ATTEMPTS}")  # 제거됨
    # print(f"- 단기 인터벌(15m,30m): 7일 보관")  # 제거됨
    # print(f"- 장기 인터벌(240m,1d,1w): 최근 100개 보관")  # 제거됨
    
    # 🚀 성능 예측
    estimated_coins = 400  # 예상 코인 수
    estimated_requests = estimated_coins * len(INTERVALS)  # 6개 인터벌
    estimated_time = estimated_requests / 20  # 초당 20 요청 가정
    # print(f"\n📊 성능 예측:")  # 제거됨
    # print(f"- 예상 요청 수: {estimated_requests:,}개")  # 제거됨
    # print(f"- 예상 소요 시간: {estimated_time/60:.1f}분")  # 제거됨
    # print(f"- 목표 초당 요청: 20+ req/s")  # 제거됨
    # print(f"- Rate limit 방지: 40ms 간격, 최대 25 동시 요청")  # 제거됨
    # print(f"- 데이터 보관: 인터벌별 최적화 정책")  # 제거됨

    # 🧪 이벤트 루프에 바인딩된 save_queue 생성
    save_queue = asyncio.Queue(maxsize=3000)

    # 🚀 저장 워커 시작
    worker = asyncio.create_task(candle_saver_worker(save_queue))

    try:
        # 🚀 데이터 수집 실행
        await fetch_all(save_queue)
    except KeyboardInterrupt:
        # print("\n⚠️ 사용자에 의해 중단됨")  # 제거됨
        pass
    except Exception as e:
        # print(f"\n❌ 오류 발생: {e}")  # 제거됨
        pass

    # 🚀 워커 종료 신호
    await save_queue.put(None)
    await worker

    # 🧹 인터벌별 캔들 데이터 정리
    cleanup_candles_by_interval()

    # print("\n✅ 모든 데이터 수집 완료!")  # 제거됨

if __name__ == "__main__":
    asyncio.run(main())