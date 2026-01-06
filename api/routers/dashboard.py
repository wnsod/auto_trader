from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel
import sqlite3
import os
import json
from datetime import datetime, time as dtime, timedelta, timezone
import logging

# 각 시장별 DB 경로 임포트 (Depends 제거용)
from api.database import (
    get_db_connection,
    COIN_DB_PATH, KR_DB_PATH, US_DB_PATH, 
    FOREX_DB_PATH, BOND_DB_PATH, COMMODITY_DB_PATH
)
from api.persona import persona_engine
from api.news_collector import news_collector
from llm_factory.store.sqlite_store import ConversationStore # 🆕 LLM Store 연동

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"]
)

# LLM Store 초기화
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LLM_DB_PATH = os.path.join(BASE_DIR, "llm_factory", "store", "conversation.db")
print(f"DEBUG: LLM_DB_PATH = {LLM_DB_PATH}")
llm_store = ConversationStore(db_path=LLM_DB_PATH)

# --- LLM 대화 응답 모델 ---
class LLMMessage(BaseModel):
    timestamp: str
    sender: str
    message: str
    mood: Optional[str] = "neutral"  # happy, sad, thinking, neutral

class LLMConversationsResponse(BaseModel):
    messages: List[LLMMessage]

# DB 경로 매핑
DB_PATHS = {
    "crypto": COIN_DB_PATH,
    "kr_stock": KR_DB_PATH,
    "us_stock": US_DB_PATH,
    "forex": FOREX_DB_PATH,
    "bond": BOND_DB_PATH,
    "commodity": COMMODITY_DB_PATH
}

# --- Response Models ---
# 한국어 이름 변환 로직
KOR_NAME_MAP = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KOR_NAME_FILE = os.path.join(BASE_DIR, "market", "coin_market", "data_storage", "market_korean_name.json")

def load_korean_names():
    global KOR_NAME_MAP
    try:
        if os.path.exists(KOR_NAME_FILE):
            with open(KOR_NAME_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    KOR_NAME_MAP = data
    except Exception as e:
        logger.error(f"Failed to load korean names: {e}")

def get_korean_name(symbol):
    if not KOR_NAME_MAP:
        load_korean_names()
    
    if symbol in KOR_NAME_MAP:
        return KOR_NAME_MAP[symbol]

    korean_map_mock = {
        'BTC': '비트코인', 'ETH': '이더리움', 'XRP': '리플', 'SOL': '솔라나',
        'DOGE': '도지코인', 'ADA': '에이다', 'AVAX': '아발란체', 'TRX': '트론',
        'DOT': '폴카닷', 'LINK': '체인링크', 'MATIC': '폴리곤', 'SHIB': '시바이누',
        'LTC': '라이트코인', 'BCH': '비트코인캐시', 'ATOM': '코스모스', 'XLM': '스텔라루멘',
        'ETC': '이더리움클래식', 'ALGO': '알고랜드', 'FIL': '파일코인', 'VET': '비체인',
        'MANA': '디센트럴랜드', 'SAND': '샌드박스', 'AXS': '엑시인피니티', 'THETA': '쎄타토큰',
        'EOS': '이오스', 'AAVE': '에이브', 'CAKE': '팬케이크스왑', 'XTZ': '테조스',
        'KLAY': '클레이튼', 'WEMIX': '위믹스', 'BORA': '보라'
    }
    
    name = korean_map_mock.get(symbol, symbol)
    if symbol == "ATOM": return "코스모스"
    if symbol == "DOT": return "폴카닷"
    return name

def check_krx_market_hours():
    """KRX 장 운영 시간 확인 (09:00 ~ 15:30, 평일)"""
    # KST = UTC+9
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    
    # 1. 주말 체크 (0=월, 6=일) -> 5=토, 6=일
    if now.weekday() >= 5:
        return "HOLIDAY"
        
    # 2. 시간 체크
    current_time = now.time()
    market_start = dtime(9, 0)
    market_end = dtime(15, 30)
    
    if market_start <= current_time <= market_end:
        return "OPEN"
    
    return "CLOSED"

class CharacterState(BaseModel):
    name: str
    image_key: str
    emotion: str
    dialogue: str
    is_active: bool
    thinking_log: Optional[str] = None
    market_regime: str = "Neutral"
    market_status: str = "OPEN"  # 🆕 추가: OPEN, CLOSED, HOLIDAY

class TradeLog(BaseModel):
    time: str
    type: str     
    category: str 
    message: str
    summary: str # 🆕 한 줄 요약 (스토리)
    kor_name: Optional[str] = None
    symbol: Optional[str] = None
    action_type: Optional[str] = None
    roi: Optional[str] = None
    entry_price: Optional[str] = None
    exit_price: Optional[str] = None
    profit_amt: Optional[str] = None
    holding_time: Optional[str] = None
    confidence_level: Optional[str] = None # 🆕 확신도 레벨 (High/Medium/Low)
    ai_score: Optional[float] = None
    fractal_score: Optional[float] = None
    mtf_score: Optional[float] = None
    cross_score: Optional[float] = None

class PositionItem(BaseModel):
    symbol: str
    kor_name: str
    roi: float
    entry_price: str
    current_price: str
    holding_time: str
    entry_time: str
    status: str
    target_price: Optional[str] = None 
    stop_loss_price: Optional[str] = None
    max_profit_pct: Optional[float] = None
    ai_score: Optional[float] = None
    ai_reason: Optional[str] = None
    fractal_score: Optional[float] = None
    mtf_score: Optional[float] = None
    cross_score: Optional[float] = None

class MarketStats(BaseModel):
    total_pnl: str
    win_rate: str
    active_count: str
    total_trades: str

class PositionsResponse(BaseModel):
    positions: List[PositionItem]
    history: List[PositionItem]
    stats: Optional[MarketStats]

class GlobalStatus(BaseModel):
    headline: str
    disclaimer: str
    total_assets: str
    market_mood: str
    market_regime: str = "Neutral"
    scanning_coins: str = ""
    news_headlines: List[str] = []
    guardian_opinion: Optional[str] = None # 🆕 알파 가디언의 한마디 (Raw 데이터)

# --- Helper Functions ---
def get_market_info(market_id: str):
    info = {
        "crypto": ("Crypto", "Crypto Cat", "cat_crypto"),
        "us_stock": ("US Stock", "Eagle Bot", "bot_eagle"),
        "kr_stock": ("Korea Stock", "Tiger Bot", "bot_tiger"),
        "forex": ("Forex", "Fox Bot", "bot_fox"),
        "bond": ("Bond", "Turtle Bot", "bot_turtle"),
        "commodity": ("Commodity", "Bear Bot", "bot_bear"),
    }
    return info.get(market_id, (market_id, "Unknown", "system"))

# --- Endpoints ---

@router.get("/latest-decisions")
def get_latest_guardian_decisions(limit: int = 10):
    """🆕 알파 가디언의 최신 가상매매 결정 및 분석 근거 조회 (LLM용)"""
    db_path = DB_PATHS.get("crypto")
    db = get_db_connection(db_path) if db_path else None
    results = []
    
    if db:
        try:
            # 🆕 신규 컬럼(ai_score, ai_reason) 존재 여부 확인 후 조회
            cursor = db.cursor()
            cursor.execute("PRAGMA table_info(virtual_trade_decisions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            query = "SELECT coin, decision, signal_score, timestamp, reason"
            if 'ai_score' in columns: query += ", ai_score"
            if 'ai_reason' in columns: query += ", ai_reason"
            query += " FROM virtual_trade_decisions ORDER BY timestamp DESC LIMIT ?"
            
            cur = db.execute(query, (limit,))
            for row in cur.fetchall():
                res = {
                    "coin": row['coin'],
                    "kor_name": get_korean_name(row['coin']),
                    "decision": row['decision'],
                    "timestamp": row['timestamp'],
                    "system_reason": row['reason']
                }
                if 'ai_score' in columns: res["ai_score"] = row['ai_score']
                if 'ai_reason' in columns: res["ai_reason"] = row['ai_reason']
                results.append(res)
        except Exception as e:
            logger.error(f"Error fetching latest decisions: {e}")
        finally:
            db.close()
    return results

@router.get("/llm-conversations", response_model=LLMConversationsResponse)
def get_llm_conversations(limit: int = 20):
    """🆕 LLM 에이전트들의 대화 로그 조회 (System Intelligence 패널용)"""
    messages = []
    
    try:
        # 최근 메시지 조회 (모든 에이전트)
        recent_msgs = llm_store.get_recent_messages(limit=limit)
        
        # 발신자별 이름 및 이모지 매핑
        sender_map = {
            "agent_coin": ("Crypto Cat", "thinking"),
            "agent_news": ("News Analyst", "neutral"),
            "agent_kr_stock": ("Tiger Cat", "neutral"),
            "agent_us_stock": ("Eagle Cat", "neutral"),
            "orchestrator": ("Orchestrator", "neutral"),
        }
        
        for msg in reversed(recent_msgs):  # 시간순 정렬
            try:
                sender_key = msg.get('sender', 'system')
                sender_info = sender_map.get(sender_key, (sender_key, "neutral"))
                
                # 메시지 내용 파싱
                content = json.loads(msg.get('content', '{}'))
                summary = content.get('summary', '')
                
                # summary가 없으면 다른 필드에서 추출
                if not summary:
                    if 'regime' in content:
                        summary = f"시장 레짐: {content['regime']}, 리스크: {content.get('risk_level', 'unknown')}"
                    elif 'impact_score' in content:
                        score = content['impact_score']
                        summary = f"뉴스 영향도: {score:+.1f} - {content.get('original_headline', '')[:30]}..."
                    else:
                        summary = str(content)[:50]
                
                # 감정/분위기 결정
                mood = sender_info[1]
                if 'risk_level' in content:
                    if content['risk_level'] == 'high':
                        mood = "sad"
                    elif content['risk_level'] == 'low':
                        mood = "happy"
                
                # 타임스탬프 포맷
                ts = msg.get('timestamp', '')
                if 'T' in ts:
                    ts = ts.split('T')[1][:5]  # HH:MM만 추출
                
                messages.append(LLMMessage(
                    timestamp=ts,
                    sender=sender_info[0],
                    message=summary,
                    mood=mood
                ))
            except Exception as e:
                logger.warning(f"Message parsing error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"LLM Conversations Error: {e}")
    
    return LLMConversationsResponse(messages=messages)

@router.get("/coin-names", response_model=Dict[str, str])
def get_coin_names():
    """전체 코인 한글명 매핑 데이터 반환"""
    if not KOR_NAME_MAP:
        load_korean_names()
    return KOR_NAME_MAP

@router.get("/global-status", response_model=GlobalStatus)
def get_global_status():
    """전역 상태 및 뉴스 (3초 주기) - LLM 결과 반영"""
    market_regime = "Neutral"
    scanning_coins = ""
    headline = "📢 [LIVE] AI Auto Trader System Status: Operational"
    
    # 1. LLM Store에서 최신 뉴스 분석 결과 조회
    try:
        recent_msgs = llm_store.get_recent_messages(sender="agent_news", limit=1)
        if recent_msgs:
            last_msg = recent_msgs[0]
            try:
                content = json.loads(last_msg['content'])
                # 뉴스 요약을 헤드라인으로 사용
                if content.get('summary'):
                    headline = f"📢 {content['summary']}"
                # 리스크 점수에 따라 분위기 조정 (예시)
                if content.get('impact_score', 0) < -0.5:
                    market_regime = "Risk-Off"
            except: pass
    except Exception as e:
        logger.error(f"Global Status Error: {e}")
    
    db_path = DB_PATHS.get("crypto")
    coin_db = get_db_connection(db_path) if db_path else None
    
    if coin_db:
        try:
            cur = coin_db.execute("SELECT key, value FROM system_status WHERE key IN ('market_regime', 'scanning_coins')")
            for row in cur.fetchall():
                if row['key'] == 'market_regime':
                    # LLM 판단이 없거나 중립적일 때만 DB 값 사용 (우선순위 로직은 나중에 조정)
                    if market_regime == "Neutral":
                        market_regime = row['value']
                elif row['key'] == 'scanning_coins':
                    scanning_coins = row['value']
        except Exception:
            pass
        finally:
            coin_db.close()

    return GlobalStatus(
        headline=headline,
        disclaimer="※ 본 방송은 AI 시뮬레이션 결과이며 실제 투자 권유가 아닙니다.",
        total_assets="$1,254,300", 
        market_mood="Greed",
        market_regime=market_regime,
        scanning_coins=scanning_coins,
        news_headlines=news_collector.get_latest_headlines()
    )

@router.get("/{market_id}/character", response_model=CharacterState)
def get_market_character(market_id: str):
    """캐릭터 상태, 감정, 대사 (2초 주기)"""
    db_path = DB_PATHS.get(market_id)
    db = get_db_connection(db_path) if db_path else None
    
    market_name, char_name, char_key = get_market_info(market_id)
    
    is_active = False
    emotion = "sleep"
    dialogue = "시스템 준비 중..."
    thinking_log = None
    
    # ... (기존 로직 동일) ...
    
    if db:
        try:
            # Thinking Log 조회
            try:
                cur = db.execute("SELECT value FROM system_status WHERE key='thinking_log'")
                row = cur.fetchone()
                if row: thinking_log = row['value']
            except: pass

            if not thinking_log:
                 try:
                    log_cur = db.execute("""
                        SELECT message FROM system_logs 
                        WHERE component IN ('Executor', 'RiskManager') 
                        ORDER BY id DESC LIMIT 1
                    """)
                    log_row = log_cur.fetchone()
                    if log_row: thinking_log = log_row['message']
                 except: pass

            # 포지션 요약 조회 (감정 판단용)
            positions = []
            cur = db.execute("SELECT coin, profit_loss_pct FROM virtual_positions")
            for row in cur.fetchall():
                 positions.append({"symbol": row['coin'], "roi": row['profit_loss_pct']})

            if positions or thinking_log:
                is_active = True
                
            # 페르소나 엔진 호출
            # Market Regime 조회
            market_regime = "Neutral"
            try:
                cur = db.execute("SELECT value FROM system_status WHERE key='market_regime'")
                row = cur.fetchone()
                if row: market_regime = row['value']
            except: pass

            # 🆕 LLM 에이전트 결과 우선 반영 (직접 DB 조회)
            try:
                # DB 경로 직접 지정 (BASE_DIR은 상단에 정의됨)
                llm_db_path = os.path.join(BASE_DIR, "llm_factory", "store", "conversation.db")
                
                agent_name = f"agent_{market_id}"
                if market_id == 'crypto': agent_name = 'agent_coin'
                elif market_id == 'kr_stock': agent_name = 'agent_kr_stock'
                elif market_id == 'us_stock': agent_name = 'agent_us_stock'
                
                if os.path.exists(llm_db_path):
                    with sqlite3.connect(llm_db_path) as llm_conn:
                        llm_conn.row_factory = sqlite3.Row
                        llm_cur = llm_conn.execute(
                            "SELECT content FROM messages WHERE sender = ? ORDER BY id DESC LIMIT 1", 
                            (agent_name,)
                        )
                        llm_row = llm_cur.fetchone()
                        
                        # DEBUG PRINT
                        print(f"DEBUG: Path={llm_db_path}, Agent={agent_name}, Row={llm_row}")
                        
                        if llm_row:
                            print(f"DEBUG: Content={llm_row['content']}")
                            content = json.loads(llm_row['content'])
                            if content.get('summary'):
                                dialogue = content['summary']
                                # 리스크/레짐에 따른 감정
                                if content.get('risk_level') == 'high': emotion = "sad"
                                elif content.get('regime', '').lower() in ['bull', 'risk-on']: emotion = "happy"
                                else: emotion = "neutral"
                                is_active = True
            except Exception as e:
                # logger.warning(f"LLM Direct Query Error: {e}")
                pass
        except Exception as e:
            # print(f"[{market_id}] Character Error: {e}")
            pass
        finally:
            db.close() # 명시적 종료
    
    # DB 연결 실패시에도 기본값 반환을 위해 market_regime 변수가 필요할 수 있으나,
    # 위 로직에서는 db가 있을 때만 market_regime을 갱신함.
    # db가 없으면 기본값 "Neutral" 사용.
    if not db:
        market_regime = "Neutral"
        
    # 🆕 마켓 상태 확인 (KRX 전용)
    market_status = "OPEN"
    if market_id == "kr_stock":
        market_status = check_krx_market_hours()

    return CharacterState(
        name=char_name,
        image_key=char_key,
        emotion=emotion,
        dialogue=dialogue,
        is_active=is_active,
        thinking_log=thinking_log,
        market_regime=market_regime,
        market_status=market_status
    )

@router.get("/{market_id}/logs", response_model=List[TradeLog])
def get_market_logs(market_id: str):
    """트레이드 로그 & 시스템 로그 (1초 주기)"""
    db_path = DB_PATHS.get(market_id)
    db = get_db_connection(db_path) if db_path else None
    logs = []
    
    if db:
        try:
            mixed_logs = []
            
            # 1. Trade History (Closed)
            try:
                # 🆕 ai_reason, ai_score, fractal_score, mtf_score, cross_score 조회 추가
                cur = db.execute("""
                    SELECT created_at, action, coin, profit_loss_pct, entry_price, exit_price, entry_timestamp, exit_timestamp, 
                           signal_pattern, entry_confidence, ai_reason, ai_score, fractal_score, mtf_score, cross_score
                    FROM virtual_trade_history 
                    ORDER BY created_at DESC LIMIT 10
                """)
                for row in cur.fetchall():
                    time_str = str(row['created_at'])
                    if 'T' in time_str: time_simple = time_str.split('T')[1][:5]
                    elif ' ' in time_str: time_simple = time_str.split(' ')[1][:5]
                    else: time_simple = time_str[:5]
                    
                    roi_val = row['profit_loss_pct']
                    raw_action = row['action']
                    
                    # 🆕 패턴/확신도 파싱 및 고급 스토리 생성
                    pattern = row.get('signal_pattern', 'unknown')
                    confidence = row.get('entry_confidence', 0.0)
                    ai_reason = row.get('ai_reason', '')
                    
                    # 확신도 레벨 변환
                    conf_level = "Low"
                    if confidence >= 0.8: conf_level = "High"
                    elif confidence >= 0.5: conf_level = "Medium"
                    
                    if raw_action and raw_action.startswith('buy'):
                        # [Trade Log] 매수 진입 - 스토리 생성
                        clean_msg = raw_action.replace('buy', '').replace('|', '').strip()
                        
                        # 요약 메시지 생성 (AI 분석 사유 최우선 활용)
                        summary = ai_reason if ai_reason else "AI 매수 시그널 포착"
                        
                        # ai_reason이 너무 길면 줄임
                        if len(summary) > 50:
                            summary = summary[:47] + "..."
                        
                        # 패턴 매핑 (고도화)
                        pattern_map = {
                            'RSI_OVERSOLD': "과매도 구간 진입, 기술적 반등 가능성 포착",
                            'GOLDEN_CROSS': "이동평균 골든크로스 발생, 상승 추세 전환",
                            'VOLATILITY_BREAKOUT': "변동성 돌파 감지, 강력한 상승 모멘텀",
                            'BOLLINGER_LOWER': "밴드 하단 지지 확인, 저가 매수 유효",
                            'DOUBLE_BOTTOM': "이중 바닥 패턴 완성, 추세 반전 신호",
                            'MACD_CROSS': "MACD 매수 시그널, 추세 강도 강화",
                            'VOLUME_SPIKE': "거래량 급증 동반한 상승 돌파",
                            'SUPPORT_BOUNCE': "주요 지지선 반등 확인",
                            'UNKNOWN': "복합 기술적 지표 긍정적 평가"
                        }
                        
                        if pattern and pattern != 'unknown':
                            summary = pattern_map.get(pattern, f"{pattern} 패턴 기반 매수 진입")
                        elif "Score" in clean_msg:
                            summary = "다중 보조지표 종합 점수 우수, 매수 진입"
                        
                        mixed_logs.append({
                            "sort_key": time_str,
                            "data": TradeLog(
                                time=time_simple,
                                type="trade",    
                                category="trade",
                                message=clean_msg,
                                summary=summary,
                                kor_name=get_korean_name(row['coin']),
                                symbol=row['coin'],
                                action_type="buy",
                                roi=None,
                                confidence_level=conf_level,
                                ai_score=row.get('ai_score', 0.0),
                                fractal_score=row.get('fractal_score', 0.5),
                                mtf_score=row.get('mtf_score', 0.5),
                                cross_score=row.get('cross_score', 0.5)
                            )
                        })
                    else:
                        # [History Log] 청산 완료
                        is_win = roi_val > 0 if roi_val is not None else False
                        action_type = "win" if is_win else "loss"
                        
                        # ... (holding_str 로직 동일)
                        holding_str = "-"
                        try:
                            if row['entry_timestamp'] and row['exit_timestamp']:
                                duration = row['exit_timestamp'] - row['entry_timestamp']
                                holding_str = f"{int(duration // 3600)}시간 {int((duration % 3600) // 60)}분"
                        except: pass

                        raw_act = row['action']
                        msg_body = "청산 완료" # 기본값
                        summary = "포지션 정리"
                        
                        # 청산 사유 고급화
                        if raw_act == 'stop_loss': 
                            msg_body = "손절 라인 이탈"
                            summary = "리스크 한계 도달, 원칙적 손절매 실행"
                        elif raw_act == 'take_profit': 
                            msg_body = "목표가 도달(익절)"
                            summary = "목표 수익률 달성, 차익 실현 완료"
                        elif raw_act == 'trailing_stop': 
                            msg_body = "트레일링 스탑"
                            summary = "추세 추종 중 반전 감지, 이익 보전 청산"
                        elif raw_act == 'sell': 
                            msg_body = "매도 시그널 발생"
                            summary = "하락 반전 시그널 포착, 전량 매도 대응"

                        mixed_logs.append({
                            "sort_key": time_str,
                            "data": TradeLog(
                                time=time_simple,
                                type="history",
                                category="history",
                                message=msg_body,
                                summary=summary,
                                kor_name=get_korean_name(row['coin']),
                                symbol=row['coin'],
                                action_type=action_type,
                                roi=f"{roi_val:.2f}%" if roi_val is not None else "0.00%",
                                entry_price=str(entry_pr) if 'entry_pr' in locals() else "", 
                                exit_price=str(exit_pr) if 'exit_pr' in locals() else "",
                                holding_time=holding_str,
                                confidence_level=conf_level,
                                ai_score=row.get('ai_score', 0.0),
                                fractal_score=row.get('fractal_score', 0.5),
                                mtf_score=row.get('mtf_score', 0.5),
                                cross_score=row.get('cross_score', 0.5)
                            )
                        })
            except: pass
            
            # 2. System Logs
            try:
                sys_cur = db.execute("""
                    SELECT created_at, component, message, level 
                    FROM system_logs 
                    WHERE component IN ('Strategy', 'Learner', 'RiskManager') 
                    ORDER BY created_at DESC LIMIT 60
                """)
                for row in sys_cur.fetchall():
                    msg = row['message']
                    # Strategy Score 필터링
                    if row['component'] == 'Strategy' and 'Score:' in msg:
                        try:
                            import re
                            score_match = re.search(r"Score:\s*([0-9\.]+)", msg)
                            if score_match and float(score_match.group(1)) < 0.7: continue
                        except: pass

                    time_str = str(row['created_at'])
                    if 'T' in time_str: time_simple = time_str.split('T')[1][:8]
                    elif ' ' in time_str: time_simple = time_str.split(' ')[1][:8]
                    else: time_simple = time_str[:8]

                    comp = row['component']
                    cat = "system"
                    if comp == 'Strategy': cat = "analysis"
                    elif comp == 'Learner': cat = "learning"
                    elif comp == 'RiskManager': cat = "risk"
                    
                    # 시스템 로그 요약 생성
                    summary = "시스템 이벤트"
                    if cat == "learning": summary = "AI 학습 수행"
                    elif cat == "risk": summary = "위험 감지"
                    elif cat == "analysis": summary = "시장 분석 중"

                    mixed_logs.append({
                        "sort_key": time_str,
                        "data": TradeLog(
                            time=time_simple[:5],
                            type="info",
                            category=cat,
                            message=msg,
                            summary=summary
                        )
                    })
            except: pass

            mixed_logs.sort(key=lambda x: x['sort_key'], reverse=True)
            logs = [item['data'] for item in mixed_logs]
            
        except Exception as e:
            # print(f"[{market_id}] Logs Error: {e}")
            pass
        finally:
            db.close()
            
    return logs

@router.get("/{market_id}/positions", response_model=PositionsResponse)
def get_market_positions(market_id: str):
    """보유 포지션 & 통계 & 히스토리 (5초 주기)"""
    db_path = DB_PATHS.get(market_id)
    db = get_db_connection(db_path) if db_path else None
    
    positions = []
    history = []
    stats = None
    
    if db:
        try:
            # Stats
            try:
                stats_cur = db.execute("SELECT total_profit_pct, win_rate, total_trades FROM virtual_performance_stats ORDER BY timestamp DESC LIMIT 1")
                stats_row = stats_cur.fetchone()
                if stats_row:
                    stats = MarketStats(
                        total_pnl=f"{stats_row['total_profit_pct']:+.2f}%",
                        win_rate=f"Win {stats_row['win_rate']:.1f}%",
                        active_count="Act 0", 
                        total_trades=f"Tot {stats_row['total_trades']}"
                    )
            except: pass

            # Positions
            # 🆕 target_price, stop_loss_price, max_profit_pct, fractal_score, mtf_score, cross_score 추가 조회
            cur = db.execute("""
                SELECT coin, profit_loss_pct, entry_price, current_price, entry_timestamp, holding_duration,
                       target_price, stop_loss_price, max_profit_pct, fractal_score, mtf_score, cross_score, ai_score, ai_reason
                FROM virtual_positions 
                ORDER BY profit_loss_pct DESC
            """)
            rows = cur.fetchall()
            from datetime import datetime
            for row in rows:
                entry_ts = row['entry_timestamp']
                holding_sec = row['holding_duration']
                
                # 🆕 상태 판단 (TP Near, SL Risk 등)
                status = "holding"
                roi = row['profit_loss_pct']
                tp = row.get('target_price', 0.0)
                sl = row.get('stop_loss_price', 0.0)
                curr = row['current_price']
                
                # 예시 로직: 목표가 95% 도달 시 TP Near
                if tp > 0 and curr >= tp * 0.99: status = "tp_near"
                elif sl > 0 and curr <= sl * 1.01: status = "sl_risk"
                
                positions.append(PositionItem(
                    symbol=row['coin'],
                    kor_name=get_korean_name(row['coin']),
                    roi=round(roi, 2),
                    entry_price=f"{row['entry_price']:.4f}",
                    current_price=f"{curr:.4f}",
                    entry_time=datetime.fromtimestamp(entry_ts).strftime("%H:%M"),
                    holding_time=f"{holding_sec // 3600}h {(holding_sec % 3600) // 60}m",
                    status=status,
                    target_price=str(tp) if tp else None,
                    stop_loss_price=str(sl) if sl else None,
                    max_profit_pct=row.get('max_profit_pct', 0.0),
                    ai_score=row.get('ai_score', 0.0),
                    ai_reason=row.get('ai_reason', ''),
                    fractal_score=row.get('fractal_score', 0.5),
                    mtf_score=row.get('mtf_score', 0.5),
                    cross_score=row.get('cross_score', 0.5)
                ))
            
            # Update Active Count
            if stats: stats.active_count = f"Act {len(positions)}"
            else: 
                stats = MarketStats(
                    total_pnl="-", win_rate="-", 
                    active_count=f"Act {len(positions)}", total_trades="-"
                )

            # History (24시간 이내 매매 내역)
            try:
                from datetime import datetime, timedelta
                # 24시간 전 타임스탬프 (초 단위)
                twenty_four_hours_ago_ts = int((datetime.now() - timedelta(hours=24)).timestamp())
                # 24시간 전 시간 문자열 (YYYY-MM-DD HH:MM:SS 형식)
                twenty_four_hours_ago_str = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
                
                # 먼저 테이블 구조 확인 (PRAGMA로 컬럼 확인)
                table_info = db.execute("PRAGMA table_info(virtual_trade_history)").fetchall()
                columns = [col[1] for col in table_info]
                has_exit_timestamp = 'exit_timestamp' in columns
                
                if has_exit_timestamp:
                    # exit_timestamp가 숫자(타임스탬프)인 경우
                    # 🆕 fractal_score, mtf_score, cross_score, ai_score, ai_reason 추가 조회
                    hist_cur = db.execute("""
                        SELECT coin, entry_price, exit_price, entry_time, exit_time, pnl, roi, holding_time, exit_timestamp,
                               fractal_score, mtf_score, cross_score, ai_score, ai_reason
                        FROM virtual_trade_history 
                        WHERE exit_timestamp >= ?
                        ORDER BY exit_timestamp DESC LIMIT 50
                    """, (twenty_four_hours_ago_ts,))
                else:
                    # exit_time이 문자열인 경우
                    # 🆕 fractal_score, mtf_score, cross_score, ai_score, ai_reason 추가 조회
                    hist_cur = db.execute("""
                        SELECT coin, entry_price, exit_price, entry_time, exit_time, pnl, roi, holding_time,
                               fractal_score, mtf_score, cross_score, ai_score, ai_reason
                        FROM virtual_trade_history 
                        WHERE exit_time >= ?
                        ORDER BY exit_time DESC LIMIT 50
                    """, (twenty_four_hours_ago_str,))
                
                for row in hist_cur.fetchall():
                    roi_val = row['roi']
                    entry_time_str = str(row['entry_time'])
                    exit_time_str = str(row['exit_time'])
                    
                    # 시간 포맷팅
                    if ' ' in entry_time_str:
                        entry_time_display = entry_time_str.split(' ')[1][:5]
                    elif 'T' in entry_time_str:
                        entry_time_display = entry_time_str.split('T')[1][:5]
                    else:
                        entry_time_display = entry_time_str[:5] if len(entry_time_str) >= 5 else entry_time_str
                    
                    history.append(PositionItem(
                        symbol=row['coin'],
                        kor_name=get_korean_name(row['coin']),
                        roi=round(roi_val, 2) if roi_val is not None else 0.0,
                        entry_price=f"{row['entry_price']:.4f}",
                        current_price=f"{row['exit_price']:.4f}",
                        entry_time=entry_time_display,
                        holding_time=row['holding_time'] if row.get('holding_time') else "-",
                        status="closed",
                        ai_score=row.get('ai_score', 0.0),
                        ai_reason=row.get('ai_reason', ''),
                        fractal_score=row.get('fractal_score', 0.5),
                        mtf_score=row.get('mtf_score', 0.5),
                        cross_score=row.get('cross_score', 0.5)
                    ))
            except Exception as e:
                # 디버깅용 에러 출력
                import traceback
                print(f"[{market_id}] History fetch error: {e}")
                print(f"[{market_id}] Traceback: {traceback.format_exc()}")
                pass
            
        except Exception as e:
            # print(f"[{market_id}] Positions Error: {e}")
            pass
        finally:
            db.close()
            
    return PositionsResponse(
        positions=positions,
        history=history,
        stats=stats
    )
