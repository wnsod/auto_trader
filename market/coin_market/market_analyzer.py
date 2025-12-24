# market/coin_market/coin_market_analyzer.py
"""
코인 마켓 분석 유틸리티 (인스턴스 전용)

기능:
1. 한국어 이름 조회 (Bithumb API)
2. 펀더멘탈 데이터 조회 (CoinGecko API - 무료)
3. 펀더멘탈 평가 및 점수 계산
4. 리스크 레벨 계산
"""

import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# ======================
# 경로 설정 (현재 파일: market/coin_market/market_analyzer.py)
# ======================

# 현재 파일이 있는 디렉토리 (market/coin_market)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# 데이터를 저장할 디렉토리 (market/coin_market/data_storage)
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')

# 디렉토리가 없으면 생성
os.makedirs(DATA_DIR, exist_ok=True)

KOREAN_NAME_JSON = os.path.join(DATA_DIR, 'market_korean_name.json')
FUNDAMENTAL_CACHE_JSON = os.path.join(DATA_DIR, 'coin_fundamentals.json')
COIN_ID_MAP_JSON = os.path.join(DATA_DIR, 'coin_id_map.json')

# ======================
# CoinGecko API 설정
# ======================

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
CACHE_EXPIRE_MINUTES = 720  # 12시간 캐시 (업데이트 주기)

# 동적 로드를 위해 COIN_ID_MAP 초기화 로직 변경
# 1. 파일에서 로드 시도
# 2. 없거나 만료되면 API로 전체 로드
def load_coin_id_map():
    try:
        if os.path.exists(COIN_ID_MAP_JSON):
            with open(COIN_ID_MAP_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 하루 이상 지난 경우 갱신 권장 (여기서는 단순 로드만)
                return data
        return {}
    except Exception:
        return {}

COIN_ID_MAP = load_coin_id_map()


# ======================
# 1. 한국어 이름 관련 (기존)
# ======================

def fetch_market_korean_map():
    """Bithumb API에서 코인 한글명 다운로드"""
    url = "https://api.bithumb.com/v1/market/all?isDetails=false"
    headers = {"accept": "application/json"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 실패: {e}")
        return

    try:
        res_json = res.json()
        print("✅ 응답 타입:", type(res_json))
        print("✅ 응답 일부:", str(res_json)[:300])
    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        return

    # API 응답 구조 분석
    data_list = None

    if isinstance(res_json, dict):
        if isinstance(res_json.get("data"), list):
            data_list = res_json["data"]
        else:
            for key, value in res_json.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and "market" in value[0]:
                        data_list = value
                        break
    elif isinstance(res_json, list):
        data_list = res_json
    else:
        print("❌ 응답이 딕셔너리나 리스트가 아닙니다.")
        return

    if data_list is None:
        print("❌ 예상과 다른 응답 형식입니다.")
        print("✅ 전체 응답:", res_json)
        return

    market_map = {}
    for item in data_list:
        if isinstance(item, dict):
            market = item.get("market") or item.get("symbol") or item.get("code")
            korean_name = item.get("korean_name") or item.get("koreanName") or item.get("name_kr")

            if market and korean_name:
                if market.startswith("KRW-"):
                    market = market[4:]
                market_map[market] = korean_name

    if not market_map:
        print("❌ 유효한 마켓 데이터를 찾을 수 없습니다.")
        print("✅ 첫 번째 항목 예시:", data_list[0] if data_list else "데이터 없음")
        return

    try:
        os.makedirs(os.path.dirname(KOREAN_NAME_JSON), exist_ok=True)
    except OSError as e:
        if e.errno != 17: # File exists 에러는 무시 (이미 디렉토리가 있으면 OK)
            print(f"⚠️ 디렉토리 생성 경고 (무시): {e}")

    try:
        with open(KOREAN_NAME_JSON, "w", encoding="utf-8") as f:
            json.dump(market_map, f, ensure_ascii=False, indent=2)
        print(f"[✔] 코인 한글명 저장 완료: {KOREAN_NAME_JSON}")
        print(f"[✔] 저장된 코인 수: {len(market_map)}개")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")


def load_market_korean_map():
    """캐시된 한글명 로드 (24시간마다 자동 갱신)"""
    try:
        need_refresh = False
        
        if not os.path.exists(KOREAN_NAME_JSON):
            print("📥 코인 한글명 파일이 없습니다. API에서 다운로드 중...")
            need_refresh = True
        else:
            # 🆕 24시간 지나면 캐시 갱신
            file_mtime = datetime.fromtimestamp(os.path.getmtime(KOREAN_NAME_JSON))
            if datetime.now() - file_mtime > timedelta(hours=24):
                print("📥 코인 한글명 캐시 만료 (24시간 초과). 갱신 중...")
                need_refresh = True
        
        if need_refresh:
            fetch_market_korean_map()

        if os.path.exists(KOREAN_NAME_JSON):
            with open(KOREAN_NAME_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print("❌ 코인 한글명 파일을 생성할 수 없습니다.")
            return {}
    except Exception as e:
        print(f"❌ 코인 한글명 로드 실패: {e}")
        return {}


def get_market_warning_list() -> List[str]:
    """
    빗썸 API에서 유의 종목 리스트 조회
    Returns: ['BTC', 'ETH'] 등 심볼 리스트
    """
    url = "https://api.bithumb.com/v1/market/all?isDetails=true"
    headers = {"accept": "application/json"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        warning_list = []
        for item in data:
            market = item.get("market", "")
            if market.startswith("KRW-"):
                # CAUTION: 유의 종목
                if item.get("market_warning") == "CAUTION":
                    symbol = market.replace("KRW-", "")
                    warning_list.append(symbol)
                    
        return warning_list
    except Exception as e:
        print(f"⚠️ 유의 종목 조회 실패: {e}")
        return []


def get_market_warning_list_extended() -> List[str]:
    """
    빗썸 API에서 유의 종목 + 엽전주(0.005원 이하) 리스트 조회
    Returns: ['BTC', 'ETH'] 등 심볼 리스트
    """
    # 기존 v1 API (404 오류) 대신 public API 사용
    url = "https://api.bithumb.com/public/ticker/ALL_KRW"
    headers = {"accept": "application/json"}
    
    try:
        # 1. 기존 유의 종목 조회
        warning_list = get_market_warning_list()
        
        # 2. 엽전주 필터링 (현재가 0.005원 이하)
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        res_json = res.json()
        
        # 응답 구조: {'status': '0000', 'data': {'BTC': {...}, ...}}
        if res_json.get('status') != '0000':
            raise Exception(f"Bithumb API Error: {res_json.get('message')}")
            
        data = res_json.get('data', {})
        
        for symbol, info in data.items():
            if symbol == 'date': # 메타데이터 제외
                continue
                
            try:
                # closing_price가 문자열로 옴
                current_price = float(info.get("closing_price", 0))
                
                # 0.005원 이하 엽전주는 호가 갭 문제로 필터링
                # BTT(0.0006원), NFT(0.0005원) 등 포함
                if 0 < current_price <= 0.005:
                    if symbol not in warning_list:
                        warning_list.append(symbol)
                        # print(f"⚠️ 엽전주 필터링: {symbol} ({current_price}원)")
            except:
                pass
                    
        return warning_list
    except Exception as e:
        print(f"⚠️ 확장 유의 종목 조회 실패: {e}")
        return get_market_warning_list() # 실패 시 기본 유의 종목만 반환


def get_all_krw_symbols() -> List[str]:
    """
    빗썸 원화 마켓 전체 심볼 조회
    Returns: ['BTC', 'ETH', ...]
    """
    url = "https://api.bithumb.com/v1/market/all?isDetails=false"
    headers = {"accept": "application/json"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        symbols = []
        for item in data:
            market = item.get("market", "")
            if market.startswith("KRW-"):
                symbols.append(market.replace("KRW-", ""))
        return symbols
    except Exception as e:
        print(f"❌ 전체 심볼 조회 실패: {e}")
        return []


def get_korean_name(market_code):
    """코인 한글명 조회 (format: 한글명(심볼))"""
    try:
        kor_map = load_market_korean_map()
        korean_name = kor_map.get(market_code, market_code)
        
        # 이미 괄호가 포함되어 있다면 그대로 반환 (중복 방지)
        if "(" in str(korean_name) and ")" in str(korean_name):
            return korean_name
            
        # 심볼과 한글명이 같으면 그대로 반환하지 않고, 영어 이름이라도 (심볼) 붙여서 통일감 유지
        # 예: 비트코인 -> 비트코인(BTC)
        # 예: Bitcoin -> Bitcoin(BTC)
        return f"{korean_name}({market_code})"
    except Exception as e:
        print(f"❌ 한국어 이름 조회 실패 ({market_code}): {e}")
        return market_code


# ======================
# 2. 펀더멘탈 데이터 관련 (신규)
# ======================

def update_coin_id_map():
    """CoinGecko에서 전체 코인 리스트를 받아와 ID 매핑 갱신"""
    print("🌐 CoinGecko 전체 코인 리스트 업데이트 중...")
    url = f"{COINGECKO_BASE_URL}/coins/list"
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        coins = res.json()
        
        new_map = {}
        for coin in coins:
            symbol = coin['symbol'].upper()
            # 이미 있으면 시총 순위가 높은 걸 써야 하는데 여기선 정보가 없음
            # 대략적으로 id 길이가 짧은 것을 선호 (bitcoin vs bitcoin-pro)
            if symbol not in new_map:
                new_map[symbol] = coin['id']
            else:
                # 기존 ID와 비교해서 더 짧거나, 특정 키워드가 없는 것을 선호
                current_id = new_map[symbol]
                new_id = coin['id']
                if len(new_id) < len(current_id):
                    new_map[symbol] = new_id
        
        # 파일 저장
        os.makedirs(os.path.dirname(COIN_ID_MAP_JSON), exist_ok=True)
        with open(COIN_ID_MAP_JSON, "w", encoding="utf-8") as f:
            json.dump(new_map, f, indent=2)
            
        global COIN_ID_MAP
        COIN_ID_MAP = new_map
        print(f"✅ 코인 ID 매핑 업데이트 완료 ({len(new_map)}개)")
        
    except Exception as e:
        print(f"❌ 코인 ID 매핑 업데이트 실패: {e}")

def fetch_fundamentals_from_coingecko(coins: List[str] = None, force_refresh: bool = False) -> Dict:
    """
    CoinGecko API에서 펀더멘탈 데이터 조회 (일괄 업데이트 방식 권장)
    - coins가 None이면 상위 250개 코인 전체 조회
    """
    # 1. 캐시 확인
    cached = load_fundamentals_cache()
    if not force_refresh and cached and 'timestamp' in cached:
        try:
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cache_time < timedelta(minutes=CACHE_EXPIRE_MINUTES):
                # 캐시가 유효하고, 특정 코인만 요청했다면 필터링해서 반환
                if coins:
                    result = {}
                    for coin in coins:
                        if coin in cached.get('data', {}):
                            result[coin] = cached['data'][coin]
                    return result
                return cached.get('data', {})
        except Exception:
            pass

    # 2. 업데이트 필요 시 (API 호출)
    print(f"🌐 CoinGecko 펀더멘탈 데이터 전체 업데이트 시작 (12시간 주기)...")
    
    # ID 매핑이 비어있으면 먼저 업데이트
    if not COIN_ID_MAP:
        update_coin_id_map()
        
    # 상위 시총 250개 코인 조회 (페이지네이션)
    all_data = {}
    pages = [1, 2] # 상위 500개 조회
    
    for page in pages:
        url = f"{COINGECKO_BASE_URL}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 250,
            'page': page,
            'sparkline': False,
            'price_change_percentage': '24h,7d,30d'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data:
                symbol = item.get('symbol', '').upper()
                if symbol:
                    all_data[symbol] = {
                        'symbol': symbol,
                        'name': item.get('name'),
                        'current_price': item.get('current_price'),
                        'market_cap': item.get('market_cap'),
                        'market_cap_rank': item.get('market_cap_rank'),
                        'fully_diluted_valuation': item.get('fully_diluted_valuation'),
                        'total_volume': item.get('total_volume'),
                        'circulating_supply': item.get('circulating_supply'),
                        'total_supply': item.get('total_supply'),
                        'max_supply': item.get('max_supply'),
                        'ath': item.get('ath'),
                        'ath_change_percentage': item.get('ath_change_percentage'),
                        'ath_date': item.get('ath_date'),
                        'atl': item.get('atl'),
                        'atl_change_percentage': item.get('atl_change_percentage'),
                        'atl_date': item.get('atl_date'),
                        'price_change_percentage_24h': item.get('price_change_percentage_24h'),
                        'price_change_percentage_7d': item.get('price_change_percentage_7d_in_currency'),
                        'price_change_percentage_30d': item.get('price_change_percentage_30d_in_currency'),
                        'last_updated': item.get('last_updated')
                    }
            
            time.sleep(1) # API rate limit 고려
            
        except Exception as e:
            print(f"❌ 페이지 {page} 조회 실패: {e}")
            
    # 캐시 저장
    if all_data:
        save_fundamentals_cache(all_data)
        print(f"✅ 펀더멘탈 데이터 업데이트 완료 ({len(all_data)}개 코인)")
        
    # 요청된 코인 반환
    if coins:
        result = {}
        for coin in coins:
            if coin in all_data:
                result[coin] = all_data[coin]
        return result
        
    return all_data


def load_fundamentals_cache() -> Dict:
    """캐시된 펀더멘탈 데이터 로드"""
    try:
        if os.path.exists(FUNDAMENTAL_CACHE_JSON):
            with open(FUNDAMENTAL_CACHE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"❌ 펀더멘탈 캐시 로드 실패: {e}")
        return {}


def save_fundamentals_cache(data: Dict):
    """펀더멘탈 데이터 캐시 저장"""
    try:
        os.makedirs(os.path.dirname(FUNDAMENTAL_CACHE_JSON), exist_ok=True)

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        with open(FUNDAMENTAL_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"[✔] 펀더멘탈 캐시 저장: {len(data)}개 코인")
    except Exception as e:
        print(f"❌ 펀더멘탈 캐시 저장 실패: {e}")


def get_fundamental_data(coin: str, use_cache: bool = True) -> Optional[Dict]:
    """
    특정 코인의 펀더멘탈 데이터 조회 (로컬 캐시만 사용)
    - API 호출을 최소화하기 위해 파일 캐시에서만 조회
    - 캐시가 없거나 만료되었을 때만 전체 업데이트 수행
    """
    # 1. 메모리나 파일 캐시에서 먼저 조회
    cached = load_fundamentals_cache()
    if cached and 'data' in cached:
        data = cached['data']
        if coin in data:
            return data[coin]
            
    # 2. 없으면 전체 업데이트 한 번 시도 (use_cache가 False일 때만 강제)
    if not use_cache:
        result = fetch_fundamentals_from_coingecko([coin], force_refresh=True)
        return result.get(coin)
        
    return None


# ======================
# 3. 펀더멘탈 평가 관련 (신규)
# ======================

def get_coin_tier(market_cap_rank: int) -> str:
    """
    시총 순위로 코인 티어 분류

    Returns:
        'MEGA': Top 10 (BTC, ETH 등)
        'LARGE': 11-50
        'MID': 51-200
        'SMALL': 201+
    """
    if market_cap_rank <= 10:
        return 'MEGA'
    elif market_cap_rank <= 50:
        return 'LARGE'
    elif market_cap_rank <= 200:
        return 'MID'
    else:
        return 'SMALL'


def get_risk_level(coin_data: Dict) -> str:
    """
    리스크 레벨 계산

    Returns:
        'LOW': 안정적 (Top 10, 유동성 높음)
        'MEDIUM': 보통 (Top 50)
        'HIGH': 위험 (Top 200)
        'VERY_HIGH': 매우 위험 (200위 밖, 유동성 부족)
    """
    rank = coin_data.get('market_cap_rank', 999)
    market_cap = coin_data.get('market_cap', 0)
    volume = coin_data.get('total_volume', 0)

    # 유동성 비율
    volume_ratio = volume / market_cap if market_cap > 0 else 0

    # ATH 대비 하락률
    ath_change = coin_data.get('ath_change_percentage', 0)

    # 위험 점수 계산
    risk_score = 0

    if rank > 200:
        risk_score += 3
    elif rank > 100:
        risk_score += 2
    elif rank > 50:
        risk_score += 1

    if volume_ratio < 0.01:
        risk_score += 2
    elif volume_ratio < 0.05:
        risk_score += 1

    if ath_change < -95:
        risk_score += 3
    elif ath_change < -80:
        risk_score += 2
    elif ath_change < -50:
        risk_score += 1

    # 리스크 레벨 결정
    if risk_score >= 5:
        return 'VERY_HIGH'
    elif risk_score >= 3:
        return 'HIGH'
    elif risk_score >= 1:
        return 'MEDIUM'
    else:
        return 'LOW'


def calculate_fundamental_score(coin_data: Dict) -> int:
    """
    펀더멘탈 종합 점수 계산 (0-150)

    기준:
    - 100점: 기본 점수
    - +30점: 보너스 (Top 10, 강한 모멘텀 등)
    - -50점: 페널티 (유동성 부족, 과도한 하락 등)
    """
    score = 100
    rank = coin_data.get('market_cap_rank', 999)
    market_cap = coin_data.get('market_cap', 0)
    volume = coin_data.get('total_volume', 0)

    # 유동성 비율
    volume_ratio = volume / market_cap if market_cap > 0 else 0

    # 가격 변화율
    price_change_24h = coin_data.get('price_change_percentage_24h', 0)
    price_change_7d = coin_data.get('price_change_percentage_7d', 0)
    price_change_30d = coin_data.get('price_change_percentage_30d', 0)

    # ATH 변화율
    ath_change = coin_data.get('ath_change_percentage', 0)

    # 인플레이션 리스크
    fdv = coin_data.get('fully_diluted_valuation', 0)
    inflation_ratio = fdv / market_cap if market_cap > 0 else 1

    # 1. 시총 순위 점수
    if rank <= 10:
        score += 20  # Top 10 보너스
    elif rank <= 50:
        score += 10
    elif rank > 200:
        score -= 20  # 200위 밖 페널티

    # 2. 유동성 점수
    if volume_ratio > 0.2:
        score += 10  # 거래 활발
    elif volume_ratio > 0.1:
        score += 5
    elif volume_ratio < 0.01:
        score -= 20  # 유동성 부족

    # 3. 모멘텀 점수
    if price_change_7d > 10 and price_change_30d > 10:
        score += 10  # 강한 상승 추세
    elif price_change_7d > 5 and price_change_30d > 5:
        score += 5
    elif price_change_7d < -20 or price_change_30d < -30:
        score -= 10  # 강한 하락 추세

    # 4. ATH 위치 점수
    if ath_change > -10:
        score -= 10  # 고점권 (위험)
    elif ath_change > -30:
        score += 0  # 중간
    elif ath_change > -70:
        score += 5  # 저점권 (잠재력)
    elif ath_change < -95:
        score -= 20  # 극저점 (망한 코인 가능성)

    # 5. 인플레이션 리스크 점수
    if inflation_ratio > 5:
        score -= 15  # 높은 인플레이션 리스크
    elif inflation_ratio > 3:
        score -= 10
    elif inflation_ratio > 2:
        score -= 5

    # 점수 범위 제한 (0-150)
    return max(0, min(150, score))


def evaluate_fundamental(coin_data: Dict, warning_list: List[str] = None) -> Dict:
    """
    펀더멘탈 종합 평가 및 필터링

    Args:
        coin_data: CoinGecko에서 조회한 펀더멘탈 데이터
        warning_list: 거래유의 종목 리스트 (선택)

    Returns:
        {
            'pass': True/False,  # 실전 매매 허용 여부
            'score': 0-150,  # 펀더멘탈 점수
            'weight': 0.5-1.5,  # 신호 가중치 (score/100)
            'risk_level': 'LOW'/'MEDIUM'/'HIGH'/'VERY_HIGH',
            'tier': 'MEGA'/'LARGE'/'MID'/'SMALL',
            'reasons': ['...']  # 평가 사유
        }
    """
    rank = coin_data.get('market_cap_rank', 999)
    market_cap = coin_data.get('market_cap', 0)
    volume = coin_data.get('total_volume', 0)
    volume_ratio = volume / market_cap if market_cap > 0 else 0
    ath_change = coin_data.get('ath_change_percentage', 0)
    symbol = coin_data.get('symbol', '').upper()

    # 티어 및 리스크 계산
    tier = get_coin_tier(rank)
    risk_level = get_risk_level(coin_data)
    score = calculate_fundamental_score(coin_data)

    # 필터링 기준 (실전 매매 제외 조건)
    reasons = []
    passed = True

    # 🚨 0순위: 거래유의 종목 체크 (가장 중요)
    if warning_list and symbol in warning_list:
        passed = False
        reasons.append(f"🚨 거래유의 종목 지정 (상폐 위험 또는 엽전주)")

    # 필수 체크 (하나라도 실패하면 탈락)
    if rank > 200:
        passed = False
        reasons.append(f"시총 순위 {rank}위 (200위 밖)")

    if volume_ratio < 0.01:
        passed = False
        reasons.append(f"유동성 부족 (거래량/시총: {volume_ratio:.3f})")

    if ath_change < -95:
        passed = False
        reasons.append(f"ATH 대비 {ath_change:.1f}% 하락 (망한 코인 가능성)")

    if market_cap < 100_000_000:  # $100M
        passed = False
        reasons.append(f"시총 너무 낮음 (${market_cap:,.0f})")

    # 가중치 계산 (0.5-1.5)
    weight = score / 100
    weight = max(0.5, min(1.5, weight))

    return {
        'pass': passed,
        'score': score,
        'weight': weight,
        'risk_level': risk_level,
        'tier': tier,
        'reasons': reasons if not passed else ['모든 필터 통과']
    }


# ======================
# 4. 유틸리티 함수
# ======================

def get_max_position_by_tier(tier: str) -> float:
    """
    티어별 최대 포지션 크기 반환

    Returns:
        MEGA: 0.20 (20%)
        LARGE: 0.10 (10%)
        MID: 0.05 (5%)
        SMALL: 0.02 (2%)
    """
    tier_limits = {
        'MEGA': 0.20,
        'LARGE': 0.10,
        'MID': 0.05,
        'SMALL': 0.02
    }
    return tier_limits.get(tier, 0.02)


def analyze_multiple_coins(coins: List[str]) -> Dict:
    """
    여러 코인 펀더멘탈 일괄 분석

    Args:
        coins: 코인 심볼 리스트

    Returns:
        {
            'BTC': {
                'fundamental': {...},  # CoinGecko 데이터
                'evaluation': {...}    # 평가 결과
            }
        }
    """
    print(f"\n📊 {len(coins)}개 코인 펀더멘탈 분석 시작...")

    # 펀더멘탈 데이터 조회
    fundamentals = fetch_fundamentals_from_coingecko(coins)

    # 빗썸 유의 종목 조회 (API 호출 + 엽전주 필터링)
    warning_list = get_market_warning_list_extended()
    if warning_list:
        print(f"⚠️ 거래유의 종목(엽전주 포함) {len(warning_list)}개 식별됨: {warning_list[:10]}...")

    # 각 코인 평가
    results = {}
    for coin in coins:
        if coin in fundamentals:
            fund_data = fundamentals[coin]
            evaluation = evaluate_fundamental(fund_data, warning_list)

            results[coin] = {
                'fundamental': fund_data,
                'evaluation': evaluation
            }
        else:
            results[coin] = {
                'fundamental': None,
                'evaluation': {
                    'pass': False,
                    'score': 0,
                    'weight': 0.5,
                    'risk_level': 'UNKNOWN',
                    'tier': 'UNKNOWN',
                    'reasons': ['펀더멘탈 데이터 없음']
                }
            }

    return results


# ======================
# 5. 테스트/디버그
# ======================

if __name__ == '__main__':
    print("=" * 60)
    print("🧪 코인 마켓 분석 유틸리티 테스트")
    print("=" * 60)

    # 테스트할 코인 리스트
    test_coins = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'SHIB']

    print(f"\n1️⃣ 한국어 이름 조회 테스트")
    print("-" * 60)
    for coin in test_coins:
        korean_name = get_korean_name(coin)
        print(f"  {coin}: {korean_name}")

    print(f"\n2️⃣ 펀더멘탈 데이터 조회 테스트")
    print("-" * 60)
    results = analyze_multiple_coins(test_coins)

    for coin, data in results.items():
        print(f"\n📌 {coin} ({get_korean_name(coin)})")

        if data['fundamental']:
            fund = data['fundamental']
            eval_result = data['evaluation']

            print(f"  • 시총 순위: #{fund.get('market_cap_rank', 'N/A')}")
            print(f"  • 시가총액: ${fund.get('market_cap', 0):,.0f}")
            print(f"  • 24h 거래량: ${fund.get('total_volume', 0):,.0f}")
            print(f"  • 현재가: ${fund.get('current_price', 0):,.2f}")
            print(f"  • ATH 대비: {fund.get('ath_change_percentage', 0):.1f}%")
            print(f"  • 30일 변화: {fund.get('price_change_percentage_30d', 0):.1f}%")
            print(f"  ---")
            print(f"  • 티어: {eval_result['tier']}")
            print(f"  • 리스크: {eval_result['risk_level']}")
            print(f"  • 점수: {eval_result['score']}/100")
            print(f"  • 가중치: {eval_result['weight']:.2f}x")
            print(f"  • 실전 허용: {'✅ YES' if eval_result['pass'] else '❌ NO'}")
            print(f"  • 사유: {', '.join(eval_result['reasons'])}")
        else:
            print(f"  ❌ 펀더멘탈 데이터 없음")

    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
