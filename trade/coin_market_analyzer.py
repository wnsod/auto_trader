# trade/coin_market_analyzer.py
"""
코인 마켓 분석 유틸리티

기능:
1. 한국어 이름 조회 (Bithumb API)
2. 펀더멘탈 데이터 조회 (CoinGecko API - 무료)
3. 펀더멘탈 평가 및 점수 계산
4. 리스크 레벨 계산

사용처:
- trade/realtime_signal_selector.py (실전 매매 신호 필터링)
"""

import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# ======================
# 경로 설정
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KOREAN_NAME_JSON = os.path.abspath(os.path.join(BASE_DIR, '..', 'data_storage', 'market_korean_name.json'))
FUNDAMENTAL_CACHE_JSON = os.path.abspath(os.path.join(BASE_DIR, '..', 'data_storage', 'coin_fundamentals.json'))

# ======================
# CoinGecko API 설정
# ======================

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
CACHE_EXPIRE_MINUTES = 60  # 1시간 캐시 (무료 API rate limit 고려)

# 코인 심볼 매핑 (Bithumb/Upbit → CoinGecko ID)
COIN_ID_MAP = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'SOL': 'solana',
    'DOGE': 'dogecoin',
    'DOT': 'polkadot',
    'MATIC': 'polygon',
    'AVAX': 'avalanche-2',
    'LINK': 'chainlink',
    'UNI': 'uniswap',
    'ATOM': 'cosmos',
    'LTC': 'litecoin',
    'BCH': 'bitcoin-cash',
    'NEAR': 'near',
    'ALGO': 'algorand',
    'ICP': 'internet-computer',
    'FIL': 'filecoin',
    'APT': 'aptos',
    'ARB': 'arbitrum',
    'OP': 'optimism',
    'SHIB': 'shiba-inu',
    'TRX': 'tron',
    'BNB': 'binancecoin',
    'TON': 'the-open-network',
    'XLM': 'stellar',
    'HBAR': 'hedera-hashgraph',
    'VET': 'vechain',
    'ETC': 'ethereum-classic',
    'SAND': 'the-sandbox',
    'MANA': 'decentraland',
    'AXS': 'axie-infinity',
    'THETA': 'theta-token',
    'FTM': 'fantom',
    'EOS': 'eos',
    'AAVE': 'aave',
    'GRT': 'the-graph',
}


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

    os.makedirs(os.path.dirname(KOREAN_NAME_JSON), exist_ok=True)

    try:
        with open(KOREAN_NAME_JSON, "w", encoding="utf-8") as f:
            json.dump(market_map, f, ensure_ascii=False, indent=2)
        print(f"[✔] 코인 한글명 저장 완료: {KOREAN_NAME_JSON}")
        print(f"[✔] 저장된 코인 수: {len(market_map)}개")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")


def load_market_korean_map():
    """캐시된 한글명 로드"""
    try:
        if not os.path.exists(KOREAN_NAME_JSON):
            print("📥 코인 한글명 파일이 없습니다. API에서 다운로드 중...")
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


def get_korean_name(market_code):
    """코인 한글명 조회"""
    try:
        kor_map = load_market_korean_map()
        return kor_map.get(market_code, market_code)
    except Exception as e:
        print(f"❌ 한국어 이름 조회 실패 ({market_code}): {e}")
        return market_code


# ======================
# 2. 펀더멘탈 데이터 관련 (신규)
# ======================

def fetch_fundamentals_from_coingecko(coins: List[str], force_refresh: bool = False) -> Dict:
    """
    CoinGecko API에서 펀더멘탈 데이터 조회

    Args:
        coins: 코인 심볼 리스트 (예: ['BTC', 'ETH', 'SOL'])
        force_refresh: True면 캐시 무시하고 API 호출

    Returns:
        {
            'BTC': {
                'symbol': 'btc',
                'current_price': 50000,
                'market_cap': 950000000000,
                'market_cap_rank': 1,
                'total_volume': 25000000000,
                'circulating_supply': 19500000,
                'total_supply': 19500000,
                'max_supply': 21000000,
                'ath': 69000,
                'ath_change_percentage': -27.5,
                'ath_date': '2021-11-10',
                'atl': 67.81,
                'atl_change_percentage': 73700,
                'price_change_percentage_24h': 2.5,
                'price_change_percentage_7d': -1.2,
                'price_change_percentage_30d': 5.8,
                'last_updated': '2025-11-17T10:30:00Z'
            }
        }
    """
    # 캐시 확인
    if not force_refresh:
        cached = load_fundamentals_cache()
        if cached and 'timestamp' in cached:
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cache_time < timedelta(minutes=CACHE_EXPIRE_MINUTES):
                # 캐시 유효 - 요청된 코인들이 있으면 반환
                result = {}
                for coin in coins:
                    if coin in cached.get('data', {}):
                        result[coin] = cached['data'][coin]
                if result:
                    print(f"✅ 캐시에서 {len(result)}개 코인 펀더멘탈 로드")
                    return result

    # CoinGecko API 호출
    print(f"🌐 CoinGecko API 호출 중... (코인 {len(coins)}개)")

    # 코인 심볼 → CoinGecko ID 변환
    coin_ids = []
    for coin in coins:
        gecko_id = COIN_ID_MAP.get(coin)
        if gecko_id:
            coin_ids.append(gecko_id)
        else:
            print(f"⚠️ {coin}: CoinGecko ID 매핑 없음 (건너뜀)")

    if not coin_ids:
        print("❌ 조회 가능한 코인이 없습니다.")
        return {}

    # API 호출 (배치 처리)
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        'vs_currency': 'usd',
        'ids': ','.join(coin_ids),
        'order': 'market_cap_desc',
        'per_page': 250,
        'page': 1,
        'sparkline': False,
        'price_change_percentage': '24h,7d,30d'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # 심볼 기준으로 재매핑
        result = {}
        reverse_map = {v: k for k, v in COIN_ID_MAP.items()}

        for item in data:
            gecko_id = item.get('id')
            symbol = reverse_map.get(gecko_id)

            if symbol:
                result[symbol] = {
                    'symbol': item.get('symbol', '').upper(),
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

        # 캐시 저장
        save_fundamentals_cache(result)

        print(f"✅ CoinGecko에서 {len(result)}개 코인 펀더멘탈 조회 완료")
        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ CoinGecko API 요청 실패: {e}")
        return {}
    except Exception as e:
        print(f"❌ 펀더멘탈 데이터 파싱 실패: {e}")
        return {}


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
    특정 코인의 펀더멘탈 데이터 조회 (캐시 우선)

    Args:
        coin: 코인 심볼 (예: 'BTC')
        use_cache: 캐시 사용 여부

    Returns:
        펀더멘탈 데이터 딕셔너리 또는 None
    """
    result = fetch_fundamentals_from_coingecko([coin], force_refresh=not use_cache)
    return result.get(coin)


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


def evaluate_fundamental(coin_data: Dict) -> Dict:
    """
    펀더멘탈 종합 평가 및 필터링

    Args:
        coin_data: CoinGecko에서 조회한 펀더멘탈 데이터

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

    # 티어 및 리스크 계산
    tier = get_coin_tier(rank)
    risk_level = get_risk_level(coin_data)
    score = calculate_fundamental_score(coin_data)

    # 필터링 기준 (실전 매매 제외 조건)
    reasons = []
    passed = True

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

    # 각 코인 평가
    results = {}
    for coin in coins:
        if coin in fundamentals:
            fund_data = fundamentals[coin]
            evaluation = evaluate_fundamental(fund_data)

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
