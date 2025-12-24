from datetime import datetime
from .base_agent import BaseAgent
from llm_factory.orchestrator.schemas import MarketSignal

class CoinAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="agent_coin")

    def process(self, market_data: dict) -> dict:
        """
        시장 데이터를 받아 LLM에게 분석을 요청하고 표준 포맷으로 반환
        """
        # 1. 프롬프트 구성 (실제 구현 시 더 정교하게)
        prompt = f"""
        Analyze the following crypto market data:
        Price: {market_data.get('price')}
        RSI: {market_data.get('rsi')}
        Volume: {market_data.get('volume')}
        
        Output format: JSON compatible with MarketSignal schema.
        """
        
        # 2. LLM 추론 (지금은 Mocking)
        # 실제로는 LLM이 생성한 JSON을 파싱해야 함
        mock_output = {
            "market": "coin",
            "timestamp": datetime.now().isoformat(),
            "regime": "neutral",
            "confidence": 0.75,
            "risk_level": "low",
            "summary": "가격 횡보 중이나 거래량이 소폭 상승하여 변동성 확대가 예상됨.",
            "key_factors": ["RSI 50 중립", "거래량 5% 증가"]
        }
        
        # 3. 데이터 검증 (Schema Validation)
        try:
            signal = MarketSignal(**mock_output)
            return signal.dict()
        except Exception as e:
            return {"error": str(e), "raw": mock_output}

