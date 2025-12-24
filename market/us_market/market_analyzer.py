"""
US 주식 시장 분석 유틸리티 (market_analyzer.py)

기능:
1. S&P 500 종목 리스트 조회 (Wikipedia/Slickcharts 소스 또는 yfinance)
2. 펀더멘탈 데이터(PER, Market Cap, Sector 등) 조회 및 분석
3. 밸류에이션 평가 및 리스크 레벨 산출
4. 페니주(Penny Stock) 등 유의 종목 필터링

데이터 소스:
- yfinance (Yahoo Finance)
- Wikipedia (S&P 500 List)
"""

import os
import json
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ======================
# 경로 설정
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')
os.makedirs(DATA_DIR, exist_ok=True)

# 데이터 캐시 파일
SP500_TICKER_CACHE = os.path.join(DATA_DIR, 'sp500_tickers.json')
FUNDAMENTAL_CACHE_JSON = os.path.join(DATA_DIR, 'us_fundamentals.json')

CACHE_EXPIRE_HOURS = 24  # 하루 한 번 갱신

# ======================
# 1. S&P 500 티커 조회
# ======================

def fetch_sp500_tickers(use_cache=True) -> List[str]:
    """
    S&P 500 티커 리스트 조회 (Wikipedia -> GitHub dataset fallback)
    """
    if use_cache and os.path.exists(SP500_TICKER_CACHE):
        try:
            with open(SP500_TICKER_CACHE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 하루 지난 캐시는 갱신 시도
                if 'timestamp' in data:
                    last_update = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - last_update < timedelta(hours=CACHE_EXPIRE_HOURS):
                        return data['tickers']
        except:
            pass

    # 1) Wikipedia 시도
    print("📥 S&P 500 티커 리스트 다운로드 중 (Wikipedia)...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # BRK.B -> BRK-B
        with open(SP500_TICKER_CACHE, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 'tickers': tickers}, f)
        print(f"✅ S&P 500 티커 {len(tickers)}개 로드 완료")
        return tickers
    except Exception as e:
        print(f"❌ Wikipedia 실패: {e}")

    # 2) GitHub dataset fallback (datasets/s-and-p-500-companies)
    print("📥 GitHub dataset에서 S&P 500 티커 다운로드 시도...")
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        with open(SP500_TICKER_CACHE, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 'tickers': tickers}, f)
        print(f"✅ S&P 500 티커 {len(tickers)}개 로드 완료 (GitHub)")
        return tickers
    except Exception as e:
        print(f"❌ GitHub fallback 실패: {e}")

    # 3) 최종 비상 리스트
    print("⚠️ 비상용 하드코딩 리스트로 진행")
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'V', 'JNJ']

def get_all_us_symbols() -> List[str]:
    """전체 종목 코드 리스트 반환 (S&P 500)"""
    return fetch_sp500_tickers()

def get_korean_name(ticker: str) -> str:
    """미국 주식은 한글명이 없으므로 티커 반환 (필요 시 별도 매핑 가능)"""
    return ticker

# ======================
# 2. 펀더멘탈 데이터 (yfinance)
# ======================

def fetch_us_fundamentals(tickers: List[str] = None, force_refresh=False) -> Dict:
    """
    S&P 500 펀더멘탈 지표 조회 (yfinance Ticker 객체 활용)
    """
    # 1. 캐시 확인
    cached_data = {}
    if not force_refresh and os.path.exists(FUNDAMENTAL_CACHE_JSON):
        try:
            with open(FUNDAMENTAL_CACHE_JSON, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                if 'timestamp' in cached:
                    last_update = datetime.fromisoformat(cached['timestamp'])
                    if datetime.now() - last_update < timedelta(hours=CACHE_EXPIRE_HOURS):
                        cached_data = cached.get('data', {})
        except:
            pass

    if tickers is None:
        tickers = fetch_sp500_tickers()

    # 캐시에 없는 티커만 선별
    target_tickers = [t for t in tickers if t not in cached_data]
    
    if not target_tickers:
        return {t: cached_data[t] for t in tickers if t in cached_data}

    print(f"🌐 US 펀더멘탈 데이터 업데이트 시작 ({len(target_tickers)}개 종목)...")
    
    # yfinance는 대량 조회 시 Tickers 객체 사용이 빠름
    # 하지만 상세 info는 개별 접근이 필요할 수 있음
    # 여기서는 50개씩 끊어서 처리 권장
    
    new_data = {}
    chunk_size = 50
    
    for i in range(0, len(target_tickers), chunk_size):
        chunk = target_tickers[i:i+chunk_size]
        try:
            # 배치 로딩은 info를 한 번에 주지 않으므로, 루프 돌며 접근
            # 속도 개선을 위해 필요한 필드만 빠르게 가져오는 방법 고려
            # 여기서는 안정성을 위해 개별 Ticker 접근 (느리지만 확실)
            for t in chunk:
                try:
                    stock = yf.Ticker(t)
                    info = stock.info
                    
                    data = {
                        'symbol': t,
                        'name': info.get('shortName', t),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'per': info.get('trailingPE'),
                        'forward_per': info.get('forwardPE'),
                        'pbr': info.get('priceToBook'),
                        'eps': info.get('trailingEps'),
                        'div_yield': (info.get('dividendYield', 0) or 0) * 100, # % 단위 변환
                        'beta': info.get('beta'),
                        'current_price': info.get('currentPrice', 0),
                        'volume': info.get('averageVolume', 0)
                    }
                    new_data[t] = data
                except Exception as e:
                    print(f"⚠️ {t} info 조회 실패: {e}")
                    new_data[t] = {'symbol': t, 'error': str(e)}
            
            print(f"   ... {i + len(chunk)}/{len(target_tickers)} 완료")
            time.sleep(1) # API 부하 조절
            
        except Exception as e:
            print(f"❌ 배치 조회 실패: {e}")

    # 데이터 병합 및 저장
    merged_data = {**cached_data, **new_data}
    
    try:
        with open(FUNDAMENTAL_CACHE_JSON, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': merged_data
            }, f, indent=2)
    except Exception as e:
        print(f"❌ 캐시 저장 실패: {e}")

    return {t: merged_data[t] for t in tickers if t in merged_data}

def get_fundamental_data(ticker: str) -> Optional[Dict]:
    """특정 종목 펀더멘탈 조회"""
    # 단일 조회 시 전체 로드 방지를 위해 캐시 직접 읽기 권장
    # 여기서는 편의상 전체 로드 함수 호출
    data = fetch_us_fundamentals([ticker])
    return data.get(ticker)

# ======================
# 3. 평가 및 분석 로직
# ======================

def get_stock_tier(info: Dict) -> str:
    """시가총액 기준 티어 분류 (USD)"""
    cap = info.get('market_cap', 0)
    
    # 단위: USD
    if cap >= 200_000_000_000: # 200B (Mega Cap) - 애플, 마소 등
        return 'MEGA'
    elif cap >= 10_000_000_000: # 10B (Large Cap)
        return 'LARGE'
    elif cap >= 2_000_000_000: # 2B (Mid Cap)
        return 'MID'
    else:
        return 'SMALL'

def calculate_value_score(info: Dict) -> int:
    """가치투자 점수 계산 (0~100)"""
    score = 50
    
    per = info.get('per')
    pbr = info.get('pbr')
    div = info.get('div_yield', 0)
    beta = info.get('beta')
    
    # PER 평가 (미국장은 한국보다 PER가 높음)
    if per:
        if 0 < per < 15: score += 15
        elif 15 <= per < 25: score += 5
        elif per > 50: score -= 10
        
    # PBR 평가
    if pbr:
        if pbr < 3: score += 10
        elif pbr > 10: score -= 5
        
    # 배당 평가
    if div > 2.0: score += 10
    
    # 변동성(Beta) 평가
    if beta:
        if 0.8 < beta < 1.2: score += 5 # 시장 추종 안정적
        elif beta > 2.0: score -= 10 # 고변동성
        
    return max(0, min(100, score))

def evaluate_fundamental(info: Dict, warning_list: List[str] = None) -> Dict:
    """종목 종합 평가"""
    ticker = info.get('symbol')
    price = info.get('current_price', 0)
    cap = info.get('market_cap', 0)
    
    reasons = []
    passed = True
    
    # 1. 페니주 필터 (5달러 미만은 기관 수급 부족 가능성)
    if price < 5.0 and price > 0: 
        # S&P 500 편입 종목이라면 5달러 미만이어도 괜찮을 수 있으나 주의
        reasons.append(f"저가주 (${price})")
        # passed = False # S&P 500이라면 일단 통과시킬 수도 있음
        
    # 2. 시가총액 필터 (S&P 500이라도 너무 작아진 경우)
    if cap < 5_000_000_000: # 5B 미만
        # passed = False
        reasons.append(f"시총 감소 (${cap/1e9:.1f}B)")

    if warning_list and ticker in warning_list:
        passed = False
        reasons.append("유의 종목 지정")

    score = calculate_value_score(info)
    tier = get_stock_tier(info)
    
    risk = 'MEDIUM'
    if score < 40: risk = 'HIGH'
    if score > 70: risk = 'LOW'
    
    return {
        'pass': passed,
        'score': score,
        'weight': score / 50.0,
        'risk_level': risk,
        'tier': tier,
        'reasons': reasons if not passed else ['필터 통과']
    }

# ======================
# 4. 유의 종목 조회
# ======================

def get_market_warning_list_extended() -> List[str]:
    """
    유의 종목 리스트 (미장용)
    - 상장폐지 예정, 파산 신청 등 (수동 관리 또는 별도 소스 필요)
    - 여기서는 기본적인 페니주나 문제 종목 하드코딩 예시
    """
    # 예시: 파산 이슈가 있는 종목들
    warning_list = ['BBBYQ', 'SIVBQ'] 
    return warning_list

# ======================
# 5. 분석 실행기 (Main)
# ======================

def analyze_multiple_coins(tickers: List[str]) -> Dict:
    """다중 종목 분석"""
    print(f"\n📊 {len(tickers)}개 주식 펀더멘탈 분석 시작...")
    
    # 펀더멘탈 데이터 로드 (필요시 갱신)
    all_funds = fetch_us_fundamentals(tickers)
    warnings = get_market_warning_list_extended()
    
    results = {}
    for ticker in tickers:
        if ticker in all_funds:
            info = all_funds[ticker]
            if 'error' in info:
                evaluation = {'pass': False, 'score': 0, 'reasons': [f"데이터 오류: {info['error']}"]}
            else:
                evaluation = evaluate_fundamental(info, warnings)
                
            results[ticker] = {
                'fundamental': info,
                'evaluation': evaluation
            }
        else:
            results[ticker] = {
                'fundamental': None,
                'evaluation': {
                    'pass': False, 'score': 0, 'reasons': ['데이터 없음']
                }
            }
    return results

if __name__ == '__main__':
    print("🧪 US 마켓 분석기 테스트")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA', 'F'] # 애플, 테슬라, 엔비디아, 포드
    res = analyze_multiple_coins(test_tickers)
    
    for t, data in res.items():
        print(f"\n📌 {t}")
        fund = data['fundamental']
        if fund and 'current_price' in fund:
            print(f"  Price: ${fund['current_price']}, PER: {fund['per']}, PBR: {fund['pbr']}")
            print(f"  평가: {data['evaluation']}")
        else:
            print(f"  데이터 없음: {fund}")
