from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field

# --- 기본 프로토콜 정의 ---

class MarketSignal(BaseModel):
    """시장 분석 결과 (Core Protocol)"""
    market: Literal["coin", "kr_stock", "us_stock", "forex"]
    timestamp: str
    regime: Literal["bull", "bear", "neutral", "high_volatility"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"]
    summary: str
    key_factors: List[str] # 판단 근거 (예: "RSI 과매수", "거래량 급증")

class NewsAlert(BaseModel):
    """뉴스 분석 결과"""
    source: str = "news_collector"
    timestamp: str
    impact_score: float = Field(..., ge=-1.0, le=1.0) # -1(악재) ~ +1(호재)
    related_assets: List[str] # 예: ["BTC", "S&P500"]
    summary: str
    original_headline: str

class OrchestratorCommand(BaseModel):
    """오케스트레이터가 에이전트에게 내리는 지침"""
    target_agent: str
    action: Literal["analyze", "halt", "change_mode"]
    parameters: Optional[Dict] = None
    reason: str

# --- 대화용 래퍼 ---
class AgentMessage(BaseModel):
    sender: str
    receiver: str
    message_type: str # 'market_signal', 'news_alert', 'command'
    payload: Dict # 위 모델들의 dict 형태

