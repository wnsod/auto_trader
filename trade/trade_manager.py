import os
import sys

# 현재 스크립트의 디렉토리와 프로젝트 루트를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

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
                    # 🆕 available = balance - locked (매매 가능한 수량만 사용)
                    total_balance = float(coin_data['balance'])
                    locked_balance = float(coin_data.get('locked', 0))
                    quantity = total_balance - locked_balance
                    
                    avg_buy_price = float(coin_data.get('avg_buy_price', 0))
                    
                    if quantity > 0:
                        new_cache[symbol] = {
                            'quantity': quantity,
                            'total_balance': total_balance,
                            'locked_balance': locked_balance,
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
            
            # 🆕 [Fix] 빗썸 KRW 마켓 호가 단위(Tick Size) 규정에 맞춘 포맷팅
            def _fmt_p(p):
                if p is None or p <= 0: return "0"
                if p < 0.1: return f"{p:.4f}"
                if p < 1: return f"{p:.3f}"
                if p < 10: return f"{p:.2f}"
                if p < 100: return f"{p:.1f}"
                return f"{int(p):,}"

            print(
                f"[지갑] {get_korean_name(coin)}: 매수가 {_fmt_p(entry_price)} | 현재가 {_fmt_p(latest_price)} | 수익률 {profit_pct:+.2f}% | 평가금액 {total_value:,.0f}원")

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


def get_trade_history(hours: int = 24, state: str = 'done') -> list:
    """🆕 빗썸 API에서 거래내역 조회
    
    Args:
        hours: 조회 기간 (시간 단위, 기본 24시간)
        state: 주문 상태 (done=체결완료, wait=대기중, cancel=취소)
    
    Returns:
        거래내역 리스트 [{coin, side, price, volume, executed_volume, ...}, ...]
    """
    endpoint = '/v1/orders'
    params = {
        'state': state,
        'limit': 100,  # 최대 100개
        'order_by': 'desc'  # 최신순
    }
    headers = generate_bithumb_headers(endpoint, params)
    
    try:
        res = requests.get(f'{API_URL}{endpoint}', headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if not isinstance(data, list):
            print(f"⚠️ 거래내역 조회 실패: {data}")
            return []
        
        # 시간 필터링 (최근 N시간)
        cutoff_time = time.time() - (hours * 3600)
        filtered = []
        
        for order in data:
            # 체결 시간 파싱
            created_at = order.get('created_at', '')
            if created_at:
                try:
                    # ISO 8601 형식 파싱 (예: 2024-12-24T10:30:00+09:00)
                    from datetime import datetime as dt
                    order_time = dt.fromisoformat(created_at.replace('Z', '+00:00'))
                    order_timestamp = order_time.timestamp()
                    
                    if order_timestamp >= cutoff_time:
                        # 코인 심볼 추출 (KRW-BTC -> BTC)
                        market = order.get('market', '')
                        coin = market.replace('KRW-', '') if market.startswith('KRW-') else market
                        
                        # 🔧 가격 정보 추출 (시장가 주문 대응)
                        avg_price = float(order.get('avg_price', 0) or 0)
                        executed_volume = float(order.get('executed_volume', 0) or 0)
                        
                        # 🆕 시장가 주문의 경우 trades에서 체결 금액 계산
                        trades = order.get('trades', [])
                        total_funds = 0.0
                        
                        if trades:
                            # trades 배열에서 실제 체결 금액 합산
                            for trade in trades:
                                funds = float(trade.get('funds', 0) or 0)
                                total_funds += funds
                            
                            # avg_price 계산 (총 체결금액 / 총 체결수량)
                            if executed_volume > 0 and avg_price == 0:
                                avg_price = total_funds / executed_volume
                        
                        # 🆕 trades 정보가 없고 avg_price도 0이면 주문 상세 조회
                        if avg_price == 0 and executed_volume > 0:
                            order_uuid = order.get('uuid', '')
                            if order_uuid:
                                detail = get_order_detail_silent(order_uuid)
                                if detail:
                                    # 상세 조회에서 trades 가져오기
                                    detail_trades = detail.get('trades', [])
                                    if detail_trades:
                                        for trade in detail_trades:
                                            funds = float(trade.get('funds', 0) or 0)
                                            total_funds += funds
                                        if executed_volume > 0:
                                            avg_price = total_funds / executed_volume
                                    else:
                                        # trades도 없으면 executed_funds 사용
                                        executed_funds = float(detail.get('executed_funds', 0) or 0)
                                        if executed_funds > 0 and executed_volume > 0:
                                            avg_price = executed_funds / executed_volume
                                            total_funds = executed_funds
                        
                        # 체결 금액 계산
                        if total_funds == 0 and avg_price > 0 and executed_volume > 0:
                            total_funds = avg_price * executed_volume
                        
                        filtered.append({
                            'coin': coin,
                            'market': market,
                            'side': order.get('side'),  # bid(매수) / ask(매도)
                            'ord_type': order.get('ord_type'),
                            'price': float(order.get('price', 0) or 0),
                            'avg_price': avg_price,
                            'volume': float(order.get('volume', 0) or 0),
                            'executed_volume': executed_volume,
                            'executed_funds': total_funds,  # 🆕 체결 금액
                            'paid_fee': float(order.get('paid_fee', 0) or 0),
                            'state': order.get('state'),
                            'created_at': created_at,
                            'timestamp': order_timestamp
                        })
                except Exception as e:
                    continue
        
        return filtered
        
    except Exception as e:
        print(f"⚠️ 거래내역 조회 오류: {e}")
        return []


def get_order_detail_silent(order_id):
    """주문 상세 조회 (로그 없이)"""
    endpoint = f'/v1/order'
    params = {'uuid': order_id}
    headers = generate_bithumb_headers(endpoint, params)

    try:
        res = requests.get(f'{API_URL}{endpoint}', headers=headers, params=params, timeout=3)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None


def save_asset_snapshot():
    """🆕 현재 총 자산 스냅샷 저장 (24시간 수익률 계산용)"""
    try:
        total_krw = get_total_wallet_krw()
        timestamp = int(time.time())
        
        # DB에 저장
        snapshot_db = os.path.join(_DEFAULT_DB_DIR, 'asset_snapshots.db')
        with sqlite3.connect(snapshot_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    total_krw REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT INTO asset_snapshots (timestamp, total_krw)
                VALUES (?, ?)
            """, (timestamp, total_krw))
            
            # 7일 이상 된 데이터 정리
            week_ago = timestamp - (7 * 24 * 3600)
            conn.execute("DELETE FROM asset_snapshots WHERE timestamp < ?", (week_ago,))
            conn.commit()
            
    except Exception as e:
        print(f"⚠️ 자산 스냅샷 저장 오류: {e}")


def get_asset_snapshot(hours_ago: int = 24) -> float:
    """🆕 N시간 전 자산 스냅샷 조회"""
    try:
        target_time = int(time.time()) - (hours_ago * 3600)
        snapshot_db = os.path.join(_DEFAULT_DB_DIR, 'asset_snapshots.db')
        
        with sqlite3.connect(snapshot_db) as conn:
            # 가장 가까운 스냅샷 조회
            row = conn.execute("""
                SELECT total_krw, timestamp FROM asset_snapshots
                WHERE timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (target_time,)).fetchone()
            
            if row:
                return row[0]
        return 0.0
    except:
        return 0.0


def get_asset_performance_24h() -> dict:
    """🆕 24시간 자산 수익률 계산"""
    current_asset = get_total_wallet_krw()
    past_asset = get_asset_snapshot(hours_ago=24)
    
    if past_asset > 0:
        profit_amount = current_asset - past_asset
        profit_pct = (profit_amount / past_asset) * 100
    else:
        profit_amount = 0
        profit_pct = 0
    
    return {
        'current_asset': current_asset,
        'past_asset_24h': past_asset,
        'profit_amount': profit_amount,
        'profit_pct': profit_pct
    }


def print_trade_summary_24h():
    """🆕 24시간 거래내역 + 자산 수익률 출력"""
    print("\n" + "=" * 60)
    print("📊 최근 24시간 빗썸 거래내역")
    print("=" * 60)
    
    # 🆕 자산 수익률 (24시간 전 대비)
    perf = get_asset_performance_24h()
    if perf['past_asset_24h'] > 0:
        profit_sign = "+" if perf['profit_pct'] >= 0 else ""
        print(f"\n💰 24시간 자산 변화:")
        print(f"   24시간 전: {perf['past_asset_24h']:,.0f}원")
        print(f"   현재:      {perf['current_asset']:,.0f}원")
        print(f"   변화:      {profit_sign}{perf['profit_amount']:,.0f}원 ({profit_sign}{perf['profit_pct']:.2f}%)")
    else:
        print(f"\n💰 현재 총 자산: {perf['current_asset']:,.0f}원")
        print(f"   ℹ️ 24시간 전 스냅샷 없음 (첫 실행 또는 데이터 없음)")
    
    # 🆕 현재 스냅샷 저장 (다음 24시간 계산용)
    save_asset_snapshot()
    
    trades = get_trade_history(hours=24)
    
    if not trades:
        print("ℹ️ 최근 24시간 거래내역이 없습니다.")
        return {'total_buy': 0, 'total_sell': 0, 'net_profit': 0, 'trades': []}
    
    # 🆕 체결 금액 계산 헬퍼 함수
    def get_trade_amount(t):
        """체결 금액 조회 (executed_funds 우선, 없으면 avg_price * volume)"""
        funds = t.get('executed_funds', 0)
        if funds > 0:
            return funds
        return t.get('avg_price', 0) * t.get('executed_volume', 0)
    
    # 매수/매도 분류
    buys = [t for t in trades if t['side'] == 'bid']
    sells = [t for t in trades if t['side'] == 'ask']
    
    # 금액 계산 (executed_funds 사용)
    total_buy_amount = sum(get_trade_amount(t) for t in buys)
    total_sell_amount = sum(get_trade_amount(t) for t in sells)
    total_fee = sum(t['paid_fee'] for t in trades)
    
    # 요약 출력
    print(f"\n📈 매수: {len(buys)}건 | 총 {total_buy_amount:,.0f}원")
    print(f"📉 매도: {len(sells)}건 | 총 {total_sell_amount:,.0f}원")
    print(f"💸 수수료: {total_fee:,.0f}원")
    net_profit = total_sell_amount - total_buy_amount - total_fee
    profit_sign = "+" if net_profit >= 0 else ""
    print(f"💰 거래 순수익: {profit_sign}{net_profit:,.0f}원 (매도-매수-수수료)")
    
    # 코인별 거래내역
    print(f"\n📋 코인별 상세:")
    coin_stats = {}
    
    for t in trades:
        coin = t['coin']
        if coin not in coin_stats:
            coin_stats[coin] = {'buys': 0, 'sells': 0, 'buy_amount': 0, 'sell_amount': 0}
        
        amount = get_trade_amount(t)
        if t['side'] == 'bid':
            coin_stats[coin]['buys'] += 1
            coin_stats[coin]['buy_amount'] += amount
        else:
            coin_stats[coin]['sells'] += 1
            coin_stats[coin]['sell_amount'] += amount
    
    for coin, stats in sorted(coin_stats.items(), key=lambda x: x[1]['sell_amount'] + x[1]['buy_amount'], reverse=True):
        korean_name = get_korean_name(coin)
        net = stats['sell_amount'] - stats['buy_amount']
        net_str = f"+{net:,.0f}" if net >= 0 else f"{net:,.0f}"
        print(f"  {korean_name}: 매수 {stats['buys']}건({stats['buy_amount']:,.0f}원) | 매도 {stats['sells']}건({stats['sell_amount']:,.0f}원) | 순익 {net_str}원")
    
    # 최근 거래 5건
    print(f"\n📝 최근 거래 (최대 5건):")
    for t in trades[:5]:
        korean_name = get_korean_name(t['coin'])
        side_str = "🔵매수" if t['side'] == 'bid' else "🔴매도"
        amount = get_trade_amount(t)
        avg_price = t.get('avg_price', 0)
        
        # 시간 포맷팅
        from datetime import datetime as dt
        trade_time = dt.fromisoformat(t['created_at'].replace('Z', '+00:00'))
        time_str = trade_time.strftime('%m/%d %H:%M')
        
        # 🆕 평균가가 0이면 체결금액/수량으로 표시
        if avg_price > 0:
            print(f"  [{time_str}] {side_str} {korean_name}: {t['executed_volume']:.4f}개 @ {avg_price:,.0f}원 = {amount:,.0f}원")
        else:
            print(f"  [{time_str}] {side_str} {korean_name}: {t['executed_volume']:.4f}개 | 체결금액 {amount:,.0f}원")
    
    print("=" * 60)
    
    return {
        'total_buy': total_buy_amount,
        'total_sell': total_sell_amount,
        'total_fee': total_fee,
        'net_profit': net_profit,
        'buy_count': len(buys),
        'sell_count': len(sells),
        'trades': trades
    }

def fetch_tick_size_from_bithumb(coin):
    """실시간 오더북 기반 호가 단위 조회"""
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

def get_bithumb_tick_size(price: float) -> float:
    """🆕 가격대별 표준 호가 단위 계산 (API 호출 없이 즉시 계산)"""
    if price < 1: return 0.0001
    if price < 10: return 0.01
    if price < 100: return 0.1
    if price < 1000: return 1
    if price < 5000: return 5
    if price < 10000: return 10
    if price < 50000: return 50
    if price < 100000: return 100
    if price < 500000: return 500
    return 1000

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

    # 지정가 주문 또는 미체결 시장가(금액) 주문: 체결 대기
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
        
        # 🆕 시장가(금액) 매수의 경우 initial_units가 0일 수 있으므로 로그 보정
        log_initial = initial_units if initial_units > 0 else "시장가"

        if state in ['done', 'completed'] or (remaining_volume <= 0 and executed_volume > 0):
            print(f"✅ {coin} 주문 전량 체결 완료: {executed_volume}/{log_initial}")
            return True

        elapsed = time.time() - start_time
        print(f"⏳ {coin} 주문 체결 대기중... ({elapsed:.0f}s 경과) 상태: {state}, 체결량: {executed_volume}/{log_initial}")
        time.sleep(2)  # 너무 빈번한 조회 방지

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
            
            # 🔧 호가단위에 맞게 반올림 (정수로 나눈 후 곱하기)
            if tick_size > 0:
                price = round(int(target_price / tick_size) * tick_size, 8)
            else:
                price = round(target_price, 8)
        else:
            current_price = realtime_price if realtime_price else get_latest_price(coin)
            if not current_price or current_price <= 0:
                print(f"❌ {coin} 가격 정보 없음, 주문 실패")
                return None
            
            slippage_ticks = 10
            if trade_type == 'bid':
                price = current_price + (tick_size * slippage_ticks)
            else:
                price = current_price - (tick_size * slippage_ticks)
            
            # 🔧 호가단위에 맞게 반올림 (정수로 나눈 후 곱하기)
            if tick_size > 0:
                price = round(int(price / tick_size) * tick_size, 8)
            else:
                price = round(price, 8)
            
            # 🔧 가격이 0 이하가 되지 않도록 보정
            if price <= 0:
                price = current_price * 0.99  # 현재가의 99%로 설정
                if tick_size > 0:
                    price = round(int(price / tick_size) * tick_size, 8)
                else:
                    price = round(price, 8)

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
            # 🆕 [추가] 전량 매도 시 아주 작은 수량 제외 (Bithumb/Floating point 오차 방지)
            if units > 0:
                units = units * 0.9999  # 0.01% 여유
            
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
            if units <= 0:
                print(f"❌ {coin} 주문 수량이 0 이하입니다: {units}")
                return None
            
            # 🔧 [수정] 수량은 코인 가격에 따라 적절한 정밀도로 조정
            # Bithumb/Upbit V1 API에서는 코인마다 수량 정밀도가 다름
            # 일반적으로 가격이 매우 낮은 코인(예: PEPE)은 소수점 자리가 적음
            current_price_ref = get_realtime_ticker(coin) or get_latest_price(coin) or 1.0
            
            if current_price_ref < 1.0:
                # 1원 미만 코인 (PEPE 등): 소수점 4자리로 제한
                units = round(units, 4)
            elif current_price_ref < 100:
                # 100원 미만 코인: 소수점 6자리
                units = round(units, 6)
            else:
                # 그 외: 소수점 8자리 (최대 정밀도)
                units = round(units, 8)
            
            # 🔧 가격 확인 및 계산
            if price and price > 0:
                calc_price = price
            else:
                calc_price = get_realtime_ticker(coin) or get_latest_price(coin)
                if not calc_price or calc_price <= 0:
                    print(f"❌ {coin} 가격 정보 없음, 주문 실패")
                    return None
            
            total_order_amount = units * calc_price

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
