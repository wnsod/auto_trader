from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional
import os
import sys

# 프로젝트 루트 경로 추가 (trade 모듈 접근용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from trade.core.market import MarketAnalyzer
except ImportError:
    print("⚠️ MarketAnalyzer Import Failed. trade module not found.")
    MarketAnalyzer = None

router = APIRouter(
    prefix="/api/market",
    tags=["market"]
)

class MarketAnalysisResponse(BaseModel):
    score: float
    regime: str
    volatility: float
    raw_score: float
    details: Dict[str, float]  # sl, long, mid, short 점수

@router.get("/analysis", response_model=MarketAnalysisResponse)
def get_market_analysis():
    """4-Layer 시장 분석 결과 반환 (Short/Mid/Long/SuperLong)"""
    if not MarketAnalyzer:
        return MarketAnalysisResponse(
            score=0.5, regime="Module Error", volatility=0.0, raw_score=0.0, details={}
        )

    try:
        # MarketAnalyzer는 내부적으로 TRADING_SYSTEM_DB_PATH를 찾음
        analyzer = MarketAnalyzer()
        result = analyzer.analyze_market_regime()
        
        # details 키가 없을 경우 대비
        if 'details' not in result:
            result['details'] = {'sl': 0.0, 'long': 0.0, 'mid': 0.0, 'short': 0.0}
            
        return MarketAnalysisResponse(
            score=result.get('score', 0.5),
            regime=result.get('regime', 'Neutral'),
            volatility=result.get('volatility', 0.0),
            raw_score=result.get('raw_score', 0.0),
            details=result.get('details', {})
        )
    except Exception as e:
        print(f"Market Analysis API Error: {e}")
        return MarketAnalysisResponse(
            score=0.5, 
            regime="Analysis Error", 
            volatility=0.0, 
            raw_score=0.0,
            details={}
        )

