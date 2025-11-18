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
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
API_URL = 'https://api.bithumb.com'
DB_PATH = '/workspace/data_storage/realtime_candles.db'

SUPPORTED_COINS_CACHE = None
SUPPORTED_COINS_CACHE_TIMESTAMP = 0


def create_holdings_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                coin TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_buy_price REAL DEFAULT 0
            );
        """)


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


def sync_wallet_to_db():
    endpoint = '/v1/accounts'
    headers = generate_bithumb_headers(endpoint)
    response = requests.get(f'{API_URL}{endpoint}', headers=headers)

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("🚨 API 응답이 JSON 형식이 아닙니다:", response.text)
        return

    if response.status_code == 200:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM holdings")
            for coin_data in data:
                coin = coin_data['currency']
                quantity = float(coin_data['balance'])
                try:
                    avg_buy_price = float(coin_data.get('avg_buy_price', 0))
                except:
                    avg_buy_price = 0.0

                conn.execute("""
                    INSERT INTO holdings (coin, quantity, avg_buy_price)
                    VALUES (?, ?, ?)
                    ON CONFLICT(coin) DO UPDATE SET
                        quantity=excluded.quantity,
                        avg_buy_price=excluded.avg_buy_price
                """, (coin, quantity, avg_buy_price))
        # print("✅ Wallet DB 동기화 완료") # 너무 빈번하게 출력되어 주석 처리
    else:
        print(f"❌ Wallet 정보 조회 실패: {response.json().get('message', '알 수 없는 오류')}")


def get_holding_coins():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT coin FROM holdings WHERE quantity > 0 AND coin NOT IN ('KRW', 'P')").fetchall()
    return [row[0] for row in rows]


def get_entry_price(coin):
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute("SELECT avg_buy_price FROM holdings WHERE coin=?", (coin,)).fetchone()
    return result[0] if result else None


def get_coin_balance(coin):
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute("SELECT quantity FROM holdings WHERE coin=?", (coin,)).fetchone()
    return result[0] if result else 0.0


def get_latest_price(coin, interval='15m'):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT close FROM candles
            WHERE coin=? AND interval=?
            ORDER BY timestamp DESC LIMIT 1
        """, (coin, interval)).fetchone()
        if row and row[0] and row[0] > 0:
            return row[0]

    for fallback_interval in ['15m', '30m', '240m', '1d']:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT close FROM candles
                WHERE coin=? AND interval=?
                ORDER BY timestamp DESC LIMIT 1
            """, (coin, fallback_interval)).fetchone()
            if row and row[0] and row[0] > 0:
                print(f"ℹ️ 가격 fallback → {coin} / {fallback_interval}")
                return row[0]
    return None


def get_latest_score(coin, interval='240m'):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT signal_score FROM signal_summary
            WHERE coin=?
            ORDER BY updated_at DESC LIMIT 1
        """, (coin,)).fetchone()
    return row[0] if row else None


def get_holding_coins_scores(interval='240m'):
    coins = get_holding_coins()
    return {coin: get_latest_score(coin, interval) or "점수 없음" for coin in coins}


def get_filtered_wallet_coins(min_balance_krw=10000, price_interval='15m'):
    wallet_coins = get_holding_coins()
    filtered_coins = []

    for coin in wallet_coins:
        quantity = get_coin_balance(coin)
        latest_price = get_latest_price(coin, price_interval)

        if not latest_price or latest_price <= 0:
            continue

        total_value = quantity * latest_price
        if total_value >= min_balance_krw:
            entry_price = get_entry_price(coin)
            profit_pct = ((latest_price - entry_price) / entry_price * 100) if entry_price and entry_price > 0 else 0
            filtered_coins.append(coin)
            print(
                f"[지갑] {coin}: 매수가 {entry_price:.2f} | 현재가 {latest_price:.2f} | 수익률 {profit_pct:.2f}% | 평가금액 {total_value:.2f}원")

    return filtered_coins


def get_total_wallet_krw():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT coin, quantity FROM holdings").fetchall()

    total_krw = 0
    for coin, qty in rows:
        if coin == 'KRW':
            total_krw += qty
            continue
        if coin == 'P' or qty <= 0:
            continue
        price = get_latest_price(coin)
        if price and qty:
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
                # API 응답 중 status가 '0000' 이 아닐 때
                return {
                    'success': False,
                    'message': result.get('message', '취소 실패: status 오류'),
                    'executed_quantity': 0,
                    'remaining_volume': 0
                }
        else:
            # status_code가 200이 아닐 때
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
    waited = 0
    while waited < timeout:
        sync_wallet_to_db()
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
        print(f"[디버깅] API 응답 전체 확인 → {result}")

        if response.status_code == 200:
            # 🔥 status가 없고 바로 데이터가 있는 경우 처리
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

def calculate_order_units(coin, allocation_ratio, total_krw):
    current_price = get_latest_price(coin)
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

# 개선된 주문 실행 및 상태 확인 로직 (1분 타임아웃)
def execute_trade_with_timeout(trade_data, timeout_sec=60):
    coin = trade_data['coin']
    initial_units = float(trade_data.get('units', 0))

    if trade_data['signal'] == -1:  # 매도
        initial_units = get_coin_balance(coin)
        if initial_units <= 0:
            print(f"⚠️ {coin} 보유 수량 없음, 매도 중단")
            return False
        trade_data['units'] = initial_units

    print(f"🚀 {coin} 지정가 주문 시작: {initial_units}개")
    order_id = execute_trade(trade_data)
    if not order_id:
        print(f"❌ 최초 주문 등록 실패: {coin}")
        return False

    # 1분간 지정가 주문 체결 대기
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
        time.sleep(5)

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
        # 매수 시에는 금액으로 주문
        market_trade_data['price'] = remaining_volume * get_latest_price(coin) * 1.01  # 1% 여유
    else:  # 매도
        # 매도 시에는 수량으로 주문
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
            time.sleep(3)
            continue
            
        retry_state = retry_detail.get('state', '').lower()
        retry_executed = float(retry_detail.get('executed_volume', '0'))
        retry_remaining = float(retry_detail.get('remaining_volume', '0'))

        if retry_state in ['done', 'completed'] or retry_remaining <= 0:
            print(f"✅ {coin} 시장가 주문 전량 체결 완료")
            return True

        elapsed = time.time() - retry_start
        print(f"⏳ {coin} 시장가 주문 체결 대기중... ({elapsed:.0f}s 경과) 상태: {retry_state}")
        time.sleep(3)

    # 시장가 주문도 30초 후 미체결 시 취소
    print(f"⏰ {coin} 시장가 주문 30초 타임아웃, 최종 취소")
    cancel_order(retry_order_id)
    print(f"❌ {coin} 시장가 주문마저 미체결, 수동 확인 필요")
    return False

# 주문 수량 부족 명확히 처리하는 개선된 로직
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

    if ord_type == 'market':
        price = None
    else:
        original_price = trade_data.get('price', get_latest_price(coin))
        price = round(round(original_price * (1.01 if trade_type == 'bid' else 0.99) / tick_size) * tick_size, 2)
        print(f"🔍 [디버깅] 계산된 가격: {price}, 원본 가격: {original_price}, tick_size: {tick_size}")

    if position_percentage:
        order_chance = get_order_chance(coin)
        krw_balance = float(order_chance['buy_quantity'])

        total_wallet_value = get_total_wallet_krw() + krw_balance
        budget = total_wallet_value * position_percentage

        if ord_type == 'market' and trade_type == 'bid':
            units = None
            total_order_amount = budget
            print(f"🔍 [디버깅] 시장가 매수 금액 (전체 자산 기준): {budget:.2f}원 (전체 자산: {total_wallet_value:.2f}원, 비율: {position_percentage:.2%})")
        else:
            if trade_type == 'ask':
                units = get_coin_balance(coin)
                print(f"🔍 [디버깅] 전량 매도 수량: {units}")
            else:
                units = budget / price
                units = round(units - (units % tick_size), 8)
                print(f"🔍 [디버깅] 계산된 수량: {units}, 예산: {budget:.2f}, 가격: {price}")

            total_order_amount = units * price
    else:
        # position_percentage가 없는 경우 (직접 units나 price 지정)
        if ord_type == 'market':
            if trade_type == 'bid':  # 매수
                # 시장가 매수는 금액으로 주문
                total_order_amount = trade_data.get('price', 0)
                units = None
                print(f"🔍 [디버깅] 시장가 매수 금액: {total_order_amount:.2f}원")
            else:  # 매도
                # 시장가 매도는 수량으로 주문
                units = trade_data.get('units', 0)
                total_order_amount = units * get_latest_price(coin)
                print(f"🔍 [디버깅] 시장가 매도 수량: {units}")
        else:
            # 지정가 주문
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

    if ord_type == 'market':
        if trade_type == 'bid':
            # 시장가 매수: 금액으로 주문
            request_body['price'] = str(total_order_amount)
            print(f"🔍 [디버깅] 시장가 매수 요청: 금액 {total_order_amount}원")
        else:
            # 시장가 매도: 수량으로 주문
            request_body['volume'] = str(units)
            print(f"🔍 [디버깅] 시장가 매도 요청: 수량 {units}")
    else:
        # 지정가 주문
        request_body['volume'] = str(units)
        request_body['price'] = str(price)
        print(f"🔍 [디버깅] 지정가 주문 요청: 수량 {units}, 가격 {price}")

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
    create_holdings_table()
    sync_wallet_to_db()
    print("📊 보유 코인 점수 현황:")
    for coin in get_holding_coins():
        score = get_latest_score(coin)
        print(f"- {coin}: {score}")