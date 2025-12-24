import sys

sys.path.insert(0, '/workspace/')  # 절대 경로 추가

import requests
import sqlite3
import jwt
import uuid
import time
import os
import hashlib
import json
from urllib.parse import urlencode
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv('/workspace/.env')
except ImportError:
    print("⚠️ python-dotenv 모듈이 없습니다. 환경변수를 직접 설정하세요.")
    # 기본 환경변수 설정 (필요시 수정)
    os.environ.setdefault('API_KEY', '')
    os.environ.setdefault('API_SECRET', '')

from collections import OrderedDict
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# 🆕 한국어 코인명 조회
try:
    from market.coin_market.market_analyzer import get_korean_name
except ImportError:
    def get_korean_name(symbol):
        return symbol

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
API_URL = 'https://api.bithumb.com'
# DB_PATH 설정 (환경변수 우선, 없으면 trade_candles.db 사용)
# market/coin_market/data_storage 경로 찾기
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT, 'market', 'coin_market', 'data_storage')
DB_PATH = os.getenv('RL_DB_PATH', os.path.join(_DEFAULT_DB_DIR, 'trade_candles.db'))

SUPPORTED_COINS_CACHE = None
SUPPORTED_COINS_CACHE_TIMESTAMP = 0

# 🚀 [메모리 캐시] 지갑 정보 (DB 대체)
WALLET_CACHE = {}
WALLET_CACHE_TIMESTAMP = 0
WALLET_CACHE_TTL = 2.0  # 2초 캐싱

def create_holdings_table():
    """DB 테이블 생성 (더 이상 사용 안 함 - 호환성 유지)"""
    pass

def generate_bithumb_headers(endpoint, params=None):
    payload = {
        'access_key': API_KEY,
        'nonce': str(uuid.uuid4()),
        'timestamp': round(time.time() * 1000)
    }
    if params:
        query_string = urlencode(params).encode()
        query_hash = hashlib.sha512(query_string).hexdigest()
        payload.update({
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512'
        })
    jwt_token = jwt.encode(payload, API_SECRET)
    return {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json'
    }

def fetch_wallet_status(force_refresh=False):
    """🚀 빗썸 API로 지갑 정보 직접 조회 (메모리 캐싱)"""
    global WALLET_CACHE, WALLET_CACHE_TIMESTAMP
    
    now = time.time()
    if not force_refresh and (now - WALLET_CACHE_TIMESTAMP < WALLET_CACHE_TTL) and WALLET_CACHE:
        return WALLET_CACHE

    endpoint = '/v1/accounts'
    headers = generate_bithumb_headers(endpoint)
    
    try:
        response = requests.get(f'{API_URL}{endpoint}', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            new_cache = {}
            for coin_data in data:
                symbol = coin_data['currency']
                try:
                    quantity = float(coin_data['balance'])
                    avg_buy_price = float(coin_data.get('avg_buy_price', 0))
                    
                    if quantity > 0:
                        new_cache[symbol] = {
                            'quantity': quantity,
                            'avg_buy_price': avg_buy_price
                        }
                except (ValueError, TypeError):
                    continue
            
            WALLET_CACHE = new_cache
            WALLET_CACHE_TIMESTAMP = now
            return WALLET_CACHE
        else:
            print(f"❌ Wallet API 조회 실패: {response.text}")
            return WALLET_CACHE  # 실패 시 기존 캐시 반환
            
    except Exception as e:
        print(f"⚠️ Wallet 조회 중 오류: {e}")
        return WALLET_CACHE

def sync_wallet_to_db():
    """호환성 유지용 - 실제로는 메모리 캐시 갱신"""
    fetch_wallet_status(force_refresh=True)

def get_holding_coins():
    """보유 코인 목록 조회 (API 기반)"""
    wallet = fetch_wallet_status()
    # KRW, P 제외
    return [symbol for symbol in wallet.keys() if symbol not in ('KRW', 'P') and wallet[symbol]['quantity'] > 0]

def get_entry_price(symbol):
    """평균 매수가 조회 (API 기반)"""
    wallet = fetch_wallet_status()
    if symbol in wallet:
        return wallet[symbol]['avg_buy_price']
    return None

def get_coin_balance(symbol):
    """보유 수량 조회 (API 기반)"""
    wallet = fetch_wallet_status()
    if symbol in wallet:
        return wallet[symbol]['quantity']
    return 0.0

def get_latest_price(symbol, interval='15m'):
    """DB에서 최신 가격 조회 (시그널용) - 없을 경우 실시간 API 조회 시도"""
    # 1. 빗썸 API 실시간 가격 우선 시도 (가장 정확)
    realtime_price = get_realtime_ticker(symbol)
    if realtime_price:
        return realtime_price

    # 2. DB 조회 (Fallback)
    with sqlite3.connect(DB_PATH) as conn:
        try:
            row = conn.execute("""
                SELECT close FROM candles
                WHERE symbol=? AND interval=?
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol, interval)).fetchone()
        except:
            # symbol이 아니라 coin 컬럼일 수도 있음
            try:
                row = conn.execute("""
                    SELECT close FROM candles
                    WHERE coin=? AND interval=?
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol, interval)).fetchone()
            except:
                row = None
            
        if row and row[0] and row[0] > 0:
            return row[0]

    return None

def get_realtime_ticker(coin):
    """🚀 [초정밀] 빗썸 Public API로 실시간 현재가 직접 조회 (DB 지연 극복)"""
    try:
        # 심볼 정규화 (KRW- 제거)
        clean_coin = coin.replace('KRW-', '')
        url = f"https://api.bithumb.com/public/ticker/{clean_coin}_KRW"
        response = requests.get(url, timeout=1.5) # 짧은 타임아웃
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '0000':
                return float(data['data']['closing_price'])
    except Exception:
        pass # 조용히 실패 (DB 가격 사용하면 됨)
    return None

def get_latest_score(symbol, interval='240m'):
    with sqlite3.connect(DB_PATH) as conn:
        try:
            row = conn.execute("""
                SELECT signal_score FROM signal_summary
                WHERE symbol=?
                ORDER BY updated_at DESC LIMIT 1
            """, (symbol,)).fetchone()
        except:
            try:
                row = conn.execute("""
                    SELECT signal_score FROM signal_summary
                    WHERE coin=?
                    ORDER BY updated_at DESC LIMIT 1
                """, (symbol,)).fetchone()
            except:
                return None
    return row[0] if row else None

def get_holding_coins_scores(interval='240m'):
    symbols = get_holding_coins()
    return {symbol: get_latest_score(symbol, interval) or "점수 없음" for symbol in symbols}

def get_filtered_wallet_coins(min_balance_krw=10000, price_interval='15m', return_dict=False):
    """평가금액 기준 필터링된 보유 코인 목록 반환
    
    Args:
        min_balance_krw: 최소 평가금액 (원)
        price_interval: 가격 조회 인터벌
        return_dict: True면 코인별 상세 정보 딕셔너리 반환, False면 코인 심볼 리스트만 반환
    
    Returns:
        return_dict=False: ['BTC', 'ETH', ...] 형태의 리스트
        return_dict=True: {'BTC': {'entry_price': 100, 'current_price': 110, ...}, ...} 형태의 딕셔너리
    """
    wallet_coins = get_holding_coins()
    filtered_coins = []
    wallet_info = {}  # 🆕 코인별 상세 정보 저장

    for coin in wallet_coins:
        quantity = get_coin_balance(coin)
        # 실시간 가격 우선 사용
        latest_price = get_realtime_ticker(coin)
        if not latest_price:
             latest_price = get_latest_price(coin, price_interval)

        if not latest_price or latest_price <= 0:
            continue

        total_value = quantity * latest_price
        if total_value >= min_balance_krw:
            entry_price = get_entry_price(coin)
            if entry_price and entry_price > 0:
                profit_pct = ((latest_price - entry_price) / entry_price * 100)
            else:
                entry_price = 0
                profit_pct = 0
                
            filtered_coins.append(coin)
            
            # 🆕 상세 정보 저장 (STEP 2에서 재사용)
            wallet_info[coin] = {
                'entry_price': entry_price,
                'current_price': latest_price,
                'quantity': quantity,
                'total_value': total_value,
                'profit_pct': profit_pct
            }
            
            print(
                f"[지갑] {get_korean_name(coin)}: 매수가 {entry_price:.2f} | 현재가 {latest_price:.2f} | 수익률 {profit_pct:.2f}% | 평가금액 {total_value:.2f}원")

    # 🆕 return_dict 옵션에 따라 반환 형태 결정
    if return_dict:
        return wallet_info
    return filtered_coins

def get_total_wallet_krw():
    """총 추정 자산 (KRW + 코인 평가액)"""
    wallet = fetch_wallet_status()
    
    total_krw = 0
    if 'KRW' in wallet:
        total_krw += wallet['KRW']['quantity']
        
    for symbol, data in wallet.items():
        if symbol == 'KRW' or symbol == 'P':
            continue
            
        qty = data['quantity']
        if qty <= 0: continue
            
        price = get_realtime_ticker(symbol) or get_latest_price(symbol)
        if price:
            total_krw += price * qty
            
    return total_krw

def get_bithumb_supported_coins():
    global SUPPORTED_COINS_CACHE, SUPPORTED_COINS_CACHE_TIMESTAMP
    now = time.time()
    if SUPPORTED_COINS_CACHE and now - SUPPORTED_COINS_CACHE_TIMESTAMP < 86400:
        return SUPPORTED_COINS_CACHE
    try:
        res = requests.get('https://api.bithumb.com/public/ticker/ALL')
        data = res.json().get('data', {})
        SUPPORTED_COINS_CACHE = set([coin.upper() for coin in data if coin != 'date'])
        SUPPORTED_COINS_CACHE_TIMESTAMP = now
        return SUPPORTED_COINS_CACHE
    except:
        return set()

def get_order_detail(order_id):
    endpoint = f'/v1/order'
    params = {'uuid': order_id}
    headers = generate_bithumb_headers(endpoint, params)

    try:
        res = requests.get(f'{API_URL}{endpoint}', headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        print(f"[상세 주문 응답] {order_id}: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"🚨 주문 상세 정보 조회 실패 (RequestException): {order_id} | 오류: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"🚨 주문 상세 정보 조회 실패 (JSONDecodeError): {order_id} | 응답: {res.text}")
        return {}

def fetch_tick_size_from_bithumb(coin):
    try:
        url = f"https://api.bithumb.com/public/orderbook/KRW-{coin}"
        headers = {"accept": "application/json"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        data = res.json()

        if data.get('status') == '0000':
            bids = data['data']['bids']
            if len(bids) >= 2:
                p1 = float(bids[0]['price'])
                p2 = float(bids[1]['price'])
                tick_size = abs(p1 - p2)
                return tick_size
        return None  # 호가 정보가 없으면 None 반환
    except Exception as e:
        print(f"⚠️ tick size 조회 실패: {coin} → {e}")
        return None

def get_order_status(order_id, coin=None):
    result = get_order_detail(order_id)
    print("📦 상세 응답:", result)
    try:
        state = result.get('state', '')
        print(f"🔍 주문 상태: {state}")
        return state in ['completed', 'done']
    except Exception as e:
        print("❌ 주문 상태 확인 실패:", e)
        return False

# 주문 취소 함수 정의
def cancel_order(order_id):
    endpoint = '/v1/order'
    params = {'uuid': order_id}
    headers = generate_bithumb_headers(endpoint, params)

    try:
        response = requests.delete(f'{API_URL}{endpoint}', headers=headers, params=params)
        result = response.json()

        if response.status_code == 200:
            if result.get('status') == '0000':
                executed_quantity = float(result['data'].get('executed_volume', 0))
                remaining_volume = float(result['data'].get('remaining_volume', 0))
                return {
                    'success': True,
                    'executed_quantity': executed_quantity,
                    'remaining_volume': remaining_volume,
                    'message': "주문 취소 완료"
                }
            else:
                return {
                    'success': False,
                    'message': result.get('message', '취소 실패: status 오류'),
                    'executed_quantity': 0,
                    'remaining_volume': 0
                }
        else:
            return {
                'success': False,
                'message': result.get('message', f"HTTP 오류: {response.status_code}"),
                'executed_quantity': 0,
                'remaining_volume': 0
            }
    except Exception as e:
        return {
            'success': False,
            'executed_quantity': 0,
            'remaining_volume': 0,
            'message': f'Exception 발생: {str(e)}'
        }

def wait_for_balance_update(expected_krw_balance, timeout=60, interval=5):
    """잔고 반영 대기 (메모리 캐시 강제 갱신)"""
    waited = 0
    while waited < timeout:
        fetch_wallet_status(force_refresh=True) # 강제 갱신
        current_krw_balance = get_total_wallet_krw()
        if current_krw_balance >= expected_krw_balance:
            print(f"✅ KRW 잔고 반영 완료 ({current_krw_balance}원)")
            return True
        print(f"⏳ KRW 잔고 반영 대기 중... ({waited}s 경과)")
        sleep(interval)
        waited += interval
    print("❌ KRW 잔고 반영 실패, 시간 초과")
    return False

# 주문 가능 정보 조회 API
def get_order_chance(coin):
    endpoint = '/v1/orders/chance'
    params = {'market': f'KRW-{coin}'}
    headers = generate_bithumb_headers(endpoint, params)

    try:
        response = requests.get(f'{API_URL}{endpoint}', params=params, headers=headers)
        result = response.json()

        # ✅ 디버깅을 위한 응답 전체 출력
        # print(f"[디버깅] API 응답 전체 확인 → {result}")

        if response.status_code == 200:
            if 'bid_account' in result and 'ask_account' in result:
                ask_account = result['ask_account']
                bid_account = result['bid_account']

                available_sell_quantity = float(ask_account['balance'])
                available_buy_quantity = float(bid_account['balance'])

                return {
                    'sell_quantity': available_sell_quantity,
                    'buy_quantity': available_buy_quantity
                }
            elif 'status' in result and result['status'] == '0000':
                data = result['data']
                ask_account = data['ask_account']
                bid_account = data['bid_account']

                available_sell_quantity = float(ask_account['balance'])
                available_buy_quantity = float(bid_account['balance'])

                return {
                    'sell_quantity': available_sell_quantity,
                    'buy_quantity': available_buy_quantity
                }
            else:
                error_message = result.get('message', '알 수 없는 오류')
                print(f"❌ 주문 가능 정보 조회 실패: {error_message}")
                return None
        else:
            print(f"❌ API 응답 오류 (status_code: {response.status_code}) → {result}")
            return None

    except Exception as e:
        print(f"🚨 주문 가능 정보 조회 중 에러 발생: {e}")
        return None

def get_available_balance():
    """주문 가능 원화(KRW) 잔고 조회"""
    # 1. 지갑 캐시에서 조회 (빠름)
    wallet = fetch_wallet_status()
    if 'KRW' in wallet:
        return wallet['KRW']['quantity']
        
    # 2. 캐시에 없으면(또는 0이면) API 호출로 확인 (정확)
    chance = get_order_chance('BTC')
    if chance:
        return chance['buy_quantity']
    return 0.0

def calculate_order_units(coin, allocation_ratio, total_krw):
    current_price = get_realtime_ticker(coin) or get_latest_price(coin)
    if current_price is None:
        print(f"⚠️ {coin} 최신 가격 조회 실패")
        return 0

    tick_size = fetch_tick_size_from_bithumb(coin)
    if not tick_size:
        print(f"⚠️ {coin} tick size 조회 실패")
        return 0

    budget = total_krw * allocation_ratio
    units = budget / (current_price * 1.01)
    units = units - (units % tick_size)
    units = round(units, 8)

    total_order_amount = units * current_price * 1.01
    if total_order_amount < 5000:
        print(f"⚠️ {coin} 최소 주문 금액 미달 ({total_order_amount}원)")
        return 0

    return units

def execute_trades_parallel(trade_data_list, timeout_sec=60):
    """여러 주문을 병렬로 실행하고 결과를 반환"""
    if not trade_data_list:
        return []

    # 빗썸 API 제한 고려하여 동시 실행 스레드 수 조절 (안전하게 8개)
    max_workers = min(len(trade_data_list), 8)
    print(f"🚀 [병렬 실행] 총 {len(trade_data_list)}개의 주문 동시 실행 시작 (스레드: {max_workers}개)")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(execute_trade_with_timeout, trade, timeout_sec)
            for trade in trade_data_list
        ]
        results = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                print(f"⚠️ 병렬 실행 중 스레드 오류 발생: {e}")
                results.append(False)
        
    print(f"✅ [병렬 실행] {len(trade_data_list)}개 주문 처리 완료")
    return results

def execute_trade_with_timeout(trade_data, timeout_sec=60):
    coin = trade_data['coin']
    initial_units = float(trade_data.get('units', 0))
    is_market_order = trade_data.get('ord_type') == 'market'

    if trade_data['signal'] == -1:  # 매도
        initial_units = get_coin_balance(coin)
        if initial_units <= 0:
            print(f"⚠️ {coin} 보유 수량 없음, 매도 중단")
            return False
        trade_data['units'] = initial_units
        order_type_str = "시장가" if is_market_order else "지정가"
        print(f"🚀 {coin} {order_type_str} 매도 주문 시작: {initial_units}개")
    else:
        # 매수 (지정가 또는 시장가)
        if trade_data.get('ord_type') == 'price':
             amount = trade_data.get('price', 0)
             print(f"🚀 {coin} 시장가(금액) 매수 주문 시작: {amount:,.0f}원")
        elif trade_data.get('ord_type') == 'market':
             # 시장가 매수인데 수량 기준인 경우 (잘 없음)
             units = trade_data.get('units', 0)
             print(f"🚀 {coin} 시장가(수량) 매수 주문 시작: {units}개")
        else:
             print(f"🚀 {coin} 지정가 매수 주문 시작: {initial_units}개")

    order_id = execute_trade(trade_data)
    if not order_id:
        print(f"❌ 최초 주문 등록 실패: {coin}")
        return False

    # 🔧 시장가 주문: 짧은 대기 후 체결 확인 (보통 즉시 체결)
    if is_market_order:
        time.sleep(2)  # 시장가는 빠르게 체결되므로 2초만 대기
        order_detail = get_order_detail(order_id)
        if order_detail:
            state = order_detail.get('state', '').lower()
            remaining_volume = float(order_detail.get('remaining_volume', '0'))
            if state in ['done', 'completed'] or remaining_volume <= 0:
                print(f"✅ {coin} 시장가 주문 전량 체결 완료")
                return True
        # 시장가인데 체결 안 되면 잠시 더 대기
        for _ in range(5):
            time.sleep(1)
            order_detail = get_order_detail(order_id)
            if order_detail:
                state = order_detail.get('state', '').lower()
                remaining_volume = float(order_detail.get('remaining_volume', '0'))
                if state in ['done', 'completed'] or remaining_volume <= 0:
                    print(f"✅ {coin} 시장가 주문 전량 체결 완료")
                    return True
        print(f"⚠️ {coin} 시장가 주문이 7초 내 미체결 - 주문 상태 확인 필요")
        return False

    # 지정가 주문: 1분간 체결 대기
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        order_detail = get_order_detail(order_id)
        
        if not order_detail:
            print(f"⚠️ {coin} 주문 상세 정보 조회 실패")
            time.sleep(5)
            continue

        state = order_detail.get('state', '').lower()
        executed_volume = float(order_detail.get('executed_volume', '0'))
        remaining_volume = float(order_detail.get('remaining_volume', '0'))

        if state in ['done', 'completed'] or remaining_volume <= 0:
            print(f"✅ {coin} 지정가 주문 전량 체결 완료: {executed_volume}/{initial_units}")
            return True

        elapsed = time.time() - start_time
        print(f"⏳ {coin} 지정가 주문 체결 대기중... ({elapsed:.0f}s 경과) 상태: {state}, 체결량: {executed_volume}/{initial_units}")
        time.sleep(1)

    # 1분 후 지정가 주문 취소
    print(f"⏰ {coin} 1분 타임아웃 도달, 지정가 주문 취소 후 시장가 주문 시작")
    cancel_response = cancel_order(order_id)
    
    if not cancel_response or not cancel_response.get('success'):
        print(f"❌ {coin} 주문 취소 실패, 수동 확인 필요")
        return False
    
    executed_qty = float(cancel_response.get('executed_quantity', 0))
    remaining_volume = initial_units - executed_qty

    if remaining_volume <= 0:
        print(f"✅ {coin} 주문 취소 후 전량 체결 완료")
        return True

    print(f"🔄 {coin} 미체결 수량 시장가 주문 시작: {remaining_volume}개")

    # 시장가 주문으로 남은 수량 처리
    market_trade_data = {
        'coin': coin,
        'signal': trade_data['signal'],
        'ord_type': 'market'
    }
    
    if trade_data['signal'] == 1:  # 매수
        market_trade_data['price'] = remaining_volume * get_realtime_ticker(coin) * 1.01
    else:  # 매도
        market_trade_data['units'] = remaining_volume

    retry_order_id = execute_trade(market_trade_data)
    if not retry_order_id:
        print(f"❌ {coin} 시장가 주문 등록 실패")
        return False

    # 시장가 주문 체결 대기 (30초)
    retry_start = time.time()
    while time.time() - retry_start < 30:
        retry_detail = get_order_detail(retry_order_id)
        
        if not retry_detail:
            time.sleep(1)
            continue
            
        retry_state = retry_detail.get('state', '').lower()
        retry_executed = float(retry_detail.get('executed_volume', '0'))
        retry_remaining = float(retry_detail.get('remaining_volume', '0'))

        if retry_state in ['done', 'completed'] or retry_remaining <= 0:
            print(f"✅ {coin} 시장가 주문 전량 체결 완료")
            return True

        elapsed = time.time() - retry_start
        print(f"⏳ {coin} 시장가 주문 체결 대기중... ({elapsed:.0f}s 경과) 상태: {retry_state}")
        time.sleep(1)

    print(f"⏰ {coin} 시장가 주문 30초 타임아웃, 최종 취소")
    cancel_order(retry_order_id)
    print(f"❌ {coin} 시장가 주문마저 미체결, 수동 확인 필요")
    return False

def execute_trade(trade_data):
    coin = trade_data['coin'].upper()
    position_percentage = trade_data.get('position_percentage', None)
    ord_type = trade_data.get('ord_type', 'limit')
    trade_type = 'bid' if trade_data['signal'] == 1 else 'ask'

    SUPPORTED_COINS = get_bithumb_supported_coins()
    if coin not in SUPPORTED_COINS:
        print(f"⚠️ {coin} → 빗썸 미지원")
        return None

    tick_size = fetch_tick_size_from_bithumb(coin) or 1.0

    if ord_type == 'market' or ord_type == 'price':
        price = None
    else:
        realtime_price = get_realtime_ticker(coin)
        
        if 'price' in trade_data:
            target_price = float(trade_data['price'])
            if realtime_price:
                diff_pct = abs(realtime_price - target_price) / target_price * 100
                if diff_pct >= 0.3:
                    if trade_type == 'bid':
                        target_price = realtime_price * 1.001
                    else:
                        target_price = realtime_price * 0.999
            
            price = round(round(target_price / tick_size) * tick_size, 8)
        else:
            current_price = realtime_price if realtime_price else get_latest_price(coin)
            if not current_price:
                print(f"❌ {coin} 가격 정보 없음, 주문 실패")
                return None
                
            slippage_ticks = 10
            if trade_type == 'bid':
                price = current_price + (tick_size * slippage_ticks)
            else:
                price = current_price - (tick_size * slippage_ticks)
            
            price = round(round(price / tick_size) * tick_size, 8)

    if position_percentage:
        order_chance = get_order_chance(coin)
        krw_balance = float(order_chance['buy_quantity'])

        total_wallet_value = get_total_wallet_krw() + krw_balance
        budget = total_wallet_value * position_percentage

        if ord_type == 'market' and trade_type == 'bid':
            units = None
            total_order_amount = budget
        elif ord_type == 'market' and trade_type == 'ask':
            # 🔧 [버그 수정] 시장가 매도 시 현재가로 total_order_amount 계산
            units = get_coin_balance(coin)
            current_price_for_calc = get_realtime_ticker(coin) or get_latest_price(coin)
            total_order_amount = units * current_price_for_calc if current_price_for_calc else units * 1000
        else:
            if trade_type == 'ask':
                units = get_coin_balance(coin)
            else:
                units = budget / price
                units = round(units - (units % tick_size), 8)

            total_order_amount = units * price
    else:
        if ord_type == 'market' or ord_type == 'price':
            if trade_type == 'bid':
                total_order_amount = trade_data.get('price', 0)
                units = None
            else:
                units = trade_data.get('units', 0)
                total_order_amount = units * (get_realtime_ticker(coin) or get_latest_price(coin))
        else:
            units = trade_data.get('units', 0)
            units = round(units - (units % tick_size), 8)
            total_order_amount = units * (price if price else get_latest_price(coin))

    if total_order_amount < 5000:
        print(f"❌ 최소 주문 금액 미달: {total_order_amount}원")
        return None

    request_body = OrderedDict([
        ('market', f'KRW-{coin}'),
        ('side', trade_type),
        ('ord_type', ord_type)
    ])

    if ord_type == 'market' or ord_type == 'price':
        if trade_type == 'bid':
            request_body['price'] = str(total_order_amount)
        else:
            request_body['volume'] = str(units)
    else:
        request_body['volume'] = str(units)
        request_body['price'] = str(price)

    query_string = urlencode(request_body).encode()
    query_hash = hashlib.sha512(query_string).hexdigest()

    payload = {
        'access_key': API_KEY,
        'nonce': str(uuid.uuid4()),
        'timestamp': round(time.time() * 1000),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512'
    }

    jwt_token = jwt.encode(payload, API_SECRET)
    headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json'
    }

    response = requests.post(f'{API_URL}/v1/orders', headers=headers, data=json.dumps(request_body))

    try:
        result = response.json()
    except Exception as e:
        print("❌ 응답 파싱 오류:", e, response.text)
        return None

    if response.status_code in [200, 201]:
        if 'uuid' in result:
            order_id = result['uuid']
            print(f"✅ 주문 정상 등록됨: {coin} | order_id={order_id}")
            return order_id
        elif result.get('status') == '0000' and 'data' in result and 'order_id' in result['data']:
            order_id = result['data']['order_id']
            print(f"✅ 주문 정상 등록됨: {coin} | order_id={order_id}")
            return order_id
        else:
            error_msg = result.get('message', f"알 수 없는 오류 (status: {result.get('status', '없음')})")
            print(f"❌ 주문 실패: {error_msg}, 전체 응답: {result}")
            return None
    else:
        error_msg = result.get('message', f"HTTP 오류: 상태 코드 {response.status_code}")
        print(f"❌ HTTP 주문 실패: {error_msg}, 전체 응답: {result}")
        return None

if __name__ == "__main__":
    print("📊 보유 코인 현황:")
    wallet = fetch_wallet_status(force_refresh=True)
    for coin, data in wallet.items():
        if coin == 'KRW':
            print(f"- KRW 잔고: {data['quantity']:.0f}원")
        else:
            price = get_realtime_ticker(coin)
            if price:
                value = price * data['quantity']
                print(f"- {coin}: {data['quantity']}개 (평단 {data['avg_buy_price']:.0f}원) | 평가액 {value:.0f}원")
