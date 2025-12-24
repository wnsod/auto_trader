"""
KRX 주식 시장 분석 유틸리티 (market_analyzer.py)

기능:
1. KRX 전 종목 리스트 및 한글명 조회 (pykrx)
2. 펀더멘탈 데이터(PER, PBR, DIV 등) 조회 및 분석
3. 밸류에이션 평가(저평가/고평가) 및 리스크 레벨 산출
4. 유의/관리 종목 필터링

데이터 소스:
- pykrx (Naver Finance 기반)
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import requests

# pykrx 임포트 시도 (없으면 안내)
try:
    from pykrx import stock
except ImportError:
    print("❌ pykrx 모듈이 설치되지 않았습니다. (pip install pykrx)")
    stock = None

# ======================
# 경로 설정
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')
os.makedirs(DATA_DIR, exist_ok=True)

# 데이터 캐시 파일
FUNDAMENTAL_CACHE_JSON = os.path.join(DATA_DIR, 'krx_fundamentals.json')
KRX_TICKER_CACHE_JSON = os.path.join(DATA_DIR, 'krx_tickers.json')

CACHE_EXPIRE_HOURS = 24  # 펀더멘탈 데이터는 하루 한 번 갱신

# ======================
# 1. 기본 정보 조회 (티커, 이름)
# ======================

def get_today_date_str():
    """오늘 날짜 (YYYYMMDD) 반환. 장 전이면 전일 기준일 수 있으나 pykrx가 알아서 최근 평일 처리함"""
    return datetime.now().strftime("%Y%m%d")

def fetch_all_tickers_info(use_cache=True) -> Dict[str, str]:
    """
    KOSPI + KOSDAQ 전 종목 티커 및 이름 조회
    Returns: {'005930': '삼성전자', ...}
    """
    if use_cache and os.path.exists(KRX_TICKER_CACHE_JSON):
        try:
            with open(KRX_TICKER_CACHE_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
            
    if stock is None:
        return {}
        
    print("📥 KRX 전 종목 티커 정보 다운로드 중...")
    try:
        today = get_today_date_str()
        market_map = {}
        
        # KOSPI
        kospi_tickers = stock.get_market_ticker_list(today, market="KOSPI")
        for ticker in kospi_tickers:
            name = stock.get_market_ticker_name(ticker)
            market_map[ticker] = name
            
        # KOSDAQ
        kosdaq_tickers = stock.get_market_ticker_list(today, market="KOSDAQ")
        for ticker in kosdaq_tickers:
            name = stock.get_market_ticker_name(ticker)
            market_map[ticker] = name
            
        # 캐시 저장
        with open(KRX_TICKER_CACHE_JSON, 'w', encoding='utf-8') as f:
            json.dump(market_map, f, ensure_ascii=False, indent=2)
            
        return market_map
        
    except Exception as e:
        print(f"❌ KRX 티커 조회 실패: {e}")
        return {}

def get_korean_name(ticker: str) -> str:
    """티커로 한글 종목명 조회"""
    ticker_map = fetch_all_tickers_info()
    name = ticker_map.get(ticker, ticker)
    if ticker in name: # 이미 이름이 티커면 그대로
        return name
    return f"{name}({ticker})"

def get_all_krw_symbols() -> List[str]:
    """전체 종목 코드 리스트 반환"""
    ticker_map = fetch_all_tickers_info()
    return list(ticker_map.keys())

# ======================
# 2. 펀더멘탈 데이터 (PER/PBR/DIV/BPS)
# ======================

def fetch_krx_fundamentals(force_refresh=False) -> Dict:
    """
    KRX 전 종목 펀더멘탈 지표 조회 (일괄)
    - PER, PBR, DIV, BPS, EPS 등
    """
    # 1. 캐시 확인
    if not force_refresh and os.path.exists(FUNDAMENTAL_CACHE_JSON):
        try:
            with open(FUNDAMENTAL_CACHE_JSON, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                if 'timestamp' in cached:
                    last_update = datetime.fromisoformat(cached['timestamp'])
                    if datetime.now() - last_update < timedelta(hours=CACHE_EXPIRE_HOURS):
                        return cached.get('data', {})
        except:
            pass

    if stock is None:
        return {}

    print(f"🌐 KRX 펀더멘탈 데이터 업데이트 시작 (pykrx)...")
    
    try:
        today = get_today_date_str()
        # KOSPI + KOSDAQ 전체 펀더멘탈 조회 (DataFrame 반환됨)
        # 컬럼: BPS, PER, PBR, EPS, DIV, DPS
        df_kospi = stock.get_market_fundamental_by_ticker(today, market="KOSPI")
        df_kosdaq = stock.get_market_fundamental_by_ticker(today, market="KOSDAQ")
        
        # 시가총액 정보도 가져오기 (Market Cap)
        df_cap_kospi = stock.get_market_cap_by_ticker(today, market="KOSPI")
        df_cap_kosdaq = stock.get_market_cap_by_ticker(today, market="KOSDAQ")
        
        # 병합 및 딕셔너리 변환
        all_data = {}
        
        for df, df_cap in [(df_kospi, df_cap_kospi), (df_kosdaq, df_cap_kosdaq)]:
            # 인덱스가 티커임
            for ticker in df.index:
                row = df.loc[ticker]
                cap_row = df_cap.loc[ticker] if ticker in df_cap.index else None
                
                # 데이터 정제 (0인 경우 None 처리 등은 선택)
                info = {
                    'per': float(row['PER']) if row['PER'] != 0 else None,
                    'pbr': float(row['PBR']) if row['PBR'] != 0 else None,
                    'div_yield': float(row['DIV']) if row['DIV'] != 0 else 0.0,
                    'eps': float(row['EPS']) if row['EPS'] != 0 else None,
                    'bps': float(row['BPS']) if row['BPS'] != 0 else None,
                    # 시가총액 등
                    'market_cap': int(cap_row['시가총액']) if cap_row is not None else 0,
                    'volume': int(cap_row['거래량']) if cap_row is not None else 0,
                    'close': int(cap_row['종가']) if cap_row is not None else 0,
                    'symbol': ticker
                }
                all_data[ticker] = info
                
        # 캐시 저장
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'data': all_data
        }
        with open(FUNDAMENTAL_CACHE_JSON, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ KRX 펀더멘탈 데이터 업데이트 완료 ({len(all_data)}개 종목)")
        return all_data

    except Exception as e:
        print(f"❌ KRX 펀더멘탈 조회 실패: {e}")
        return {}

def get_fundamental_data(ticker: str) -> Optional[Dict]:
    """특정 종목 펀더멘탈 조회 (캐시 기반)"""
    data = fetch_krx_fundamentals()
    return data.get(ticker)

# ======================
# 3. 평가 및 분석 로직
# ======================

def get_stock_tier(info: Dict) -> str:
    """시가총액 기준 티어 분류"""
    cap = info.get('market_cap', 0)
    
    # 단위: 원
    if cap >= 10_000_000_000_000: # 10조 이상
        return 'MEGA' # 삼성전자, 하이닉스 급
    elif cap >= 1_000_000_000_000: # 1조 이상
        return 'LARGE'
    elif cap >= 300_000_000_000: # 3000억 이상
        return 'MID'
    else:
        return 'SMALL'

def calculate_value_score(info: Dict) -> int:
    """
    가치투자 점수 계산 (0~100)
    - 저PER, 저PBR, 고배당일수록 높은 점수
    """
    score = 50 # 기본점수
    
    per = info.get('per')
    pbr = info.get('pbr')
    div = info.get('div_yield', 0)
    eps = info.get('eps', 0)
    
    # 1. 이익(PER) 평가
    if per:
        if 0 < per < 5: score += 20    # 초저평가
        elif 5 <= per < 10: score += 10 # 저평가
        elif 10 <= per < 20: score += 0 # 적정
        elif per >= 50: score -= 20    # 고평가/성장주
    else:
        if eps and eps < 0: score -= 30 # 적자 기업 페널티
        
    # 2. 자산(PBR) 평가
    if pbr:
        if 0 < pbr < 0.5: score += 20  # 청산가치 미만
        elif 0.5 <= pbr < 1.0: score += 10
        elif pbr > 5.0: score -= 10    # 고PBR
        
    # 3. 배당(DIV) 평가
    if div > 5.0: score += 15
    elif div > 3.0: score += 5
    
    return max(0, min(100, score))

def evaluate_fundamental(info: Dict, warning_list: List[str] = None) -> Dict:
    """
    종목 종합 평가
    """
    ticker = info.get('symbol')
    cap = info.get('market_cap', 0)
    volume = info.get('volume', 0)
    
    reasons = []
    passed = True
    
    # 1. 거래량 필터 (유동성)
    # 주가는 만원인데 거래량이 100주면 곤란함 -> 거래대금(대략)으로 체크 추천하지만 여기선 거래량
    if volume < 1000: # 거래량 극소
        passed = False
        reasons.append(f"거래량 부족 ({volume})")
        
    # 2. 시가총액 필터 (초소형주 제외)
    if cap < 30_000_000_000: # 300억 미만
        passed = False
        reasons.append(f"초소형주 ({cap//100000000}억)")
        
    # 3. 적자 지속 리스크 (PER가 없고 적자)
    if info.get('per') is None and info.get('eps', 0) < 0:
        # PBR이라도 아주 낮으면 자산주로 볼 수 있음
        if not (info.get('pbr') and info.get('pbr') < 0.5):
            passed = False
            reasons.append("적자 기업 (PER N/A)")
            
    # 4. 유의 종목 (외부 리스트)
    if warning_list and ticker in warning_list:
        passed = False
        reasons.append("관리/유의 종목 지정")

    score = calculate_value_score(info)
    tier = get_stock_tier(info)
    
    # 리스크 레벨 (간단 로직)
    risk = 'MEDIUM'
    if score < 30: risk = 'HIGH'
    if score > 70: risk = 'LOW'
    if cap < 100_000_000_000: risk = 'HIGH' # 1000억 미만은 변동성 큼
    
    return {
        'pass': passed,
        'score': score,
        'weight': score / 50.0, # 1.0 기준
        'risk_level': risk,
        'tier': tier,
        'reasons': reasons if not passed else ['필터 통과']
    }

# ======================
# 4. 유의 종목 조회 (KRX)
# ======================

def get_market_warning_list_extended() -> List[str]:
    """
    관리종목 + 거래정지 종목 리스트 조회
    """
    if stock is None: return []
    
    try:
        today = get_today_date_str()
        warning_list = []
        
        # 관리종목 (메서드명 호환성 체크)
        # pykrx 버전에 따라 메서드 이름이 다를 수 있음
        
        # 1. 관리종목 (Administrative Issue)
        if hasattr(stock, 'get_market_administrative_issue_ticker_list'):
            adm_kospi = stock.get_market_administrative_issue_ticker_list(today, "KOSPI")
            adm_kosdaq = stock.get_market_administrative_issue_ticker_list(today, "KOSDAQ")
            warning_list.extend(adm_kospi)
            warning_list.extend(adm_kosdaq)
        # 구버전 호환 (Manage Issue)
        elif hasattr(stock, 'get_market_manage_issue_ticker_list'):
             adm_kospi = stock.get_market_manage_issue_ticker_list(today, "KOSPI")
             adm_kosdaq = stock.get_market_manage_issue_ticker_list(today, "KOSDAQ")
             warning_list.extend(adm_kospi)
             warning_list.extend(adm_kosdaq)
        else:
            print("⚠️ pykrx에서 관리종목 조회 메서드를 찾을 수 없습니다.")

        # 2. 거래정지 종목 (Trading Halt) - 선택사항
        if hasattr(stock, 'get_market_trading_halt_ticker_list'):
             stop_kospi = stock.get_market_trading_halt_ticker_list(today, "KOSPI")
             stop_kosdaq = stock.get_market_trading_halt_ticker_list(today, "KOSDAQ")
             warning_list.extend(stop_kospi)
             warning_list.extend(stop_kosdaq)
        
        return list(set(warning_list))
    except Exception as e:
        print(f"⚠️ KRX 유의 종목 조회 실패: {e}")
        return []

# ======================
# 5. 분석 실행기 (Main)
# ======================

def analyze_multiple_coins(tickers: List[str]) -> Dict:
    """(호환성 유지용 이름) 다중 종목 분석"""
    print(f"\n📊 {len(tickers)}개 주식 펀더멘탈 분석 시작...")
    
    all_funds = fetch_krx_fundamentals()
    warnings = get_market_warning_list_extended()
    
    results = {}
    for ticker in tickers:
        if ticker in all_funds:
            info = all_funds[ticker]
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
    print("🧪 KRX 마켓 분석기 테스트")
    
    # 삼성전자(005930), 하이닉스(000660), 에코프로비엠(247540), 적자잡주 예시
    test_tickers = ['005930', '000660', '247540'] 
    
    res = analyze_multiple_coins(test_tickers)
    
    for t, data in res.items():
        name = get_korean_name(t)
        print(f"\n📌 {name}")
        fund = data['fundamental']
        if fund:
            print(f"  PER: {fund['per']}, PBR: {fund['pbr']}, 배당: {fund['div_yield']}%")
            print(f"  평가: {data['evaluation']}")
        else:
            print("  데이터 없음")
