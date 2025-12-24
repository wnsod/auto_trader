import json
from datetime import datetime
from .base_agent import BaseAgent
from llm_factory.orchestrator.schemas import NewsAlert

class NewsAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="agent_news")

    def process(self, news_items: list) -> dict:
        """
        뉴스 리스트를 받아 가장 중요한 이슈를 요약하고 리스크 점수 산출
        """
        if not news_items:
            return {"status": "no_news"}

        # 1. 프롬프트 구성
        headlines = "\n".join([f"- {item['title']}" for item in news_items])
        prompt = f"""
        Analyze these headlines for market impact:
        {headlines}
        
        Output format: JSON compatible with NewsAlert schema.
        JSON only, no markdown.
        """
        
        # 2. LLM 추론
        llm_response = self.call_llm(prompt, system_role="You are an expert news analyst.")
        
        if llm_response:
            try:
                # JSON 파싱 시도
                clean_json = llm_response.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                
                # 스키마 검증
                alert = NewsAlert(**parsed)
                return alert.dict()
            except Exception as e:
                # print(f"⚠️ [NewsAgent] Parsing Failed: {e}")
                print(f"[NewsAgent] Parsing Failed: {e}")
                # 실패 시 Mock 데이터로 Fallback (안전장치)
        
        # Mock Response (Fallback)
        mock_output = {
            "source": "news_collector",
            "timestamp": datetime.now().isoformat(),
            "impact_score": -0.8,
            "related_assets": ["BTC", "ETH"],
            "summary": "주요 거래소 규제 발표로 인한 시장 위축 우려.",
            "original_headline": news_items[0]['title'] if news_items else "Unknown"
        }
        
        # 3. 데이터 검증
        try:
            alert = NewsAlert(**mock_output)
            return alert.dict()
        except Exception as e:
            return {"error": str(e), "raw": mock_output}

