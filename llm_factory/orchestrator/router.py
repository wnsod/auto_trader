from llm_factory.agents.coin_agent import CoinAgent
from llm_factory.agents.news_agent import NewsAgent
from llm_factory.store.sqlite_store import ConversationStore
from llm_factory.orchestrator.schemas import AgentMessage
import json

class Orchestrator:
    def __init__(self):
        self.store = ConversationStore()
        self.coin_agent = CoinAgent()
        self.news_agent = NewsAgent()
        # 나중에 여기에 각 에이전트를 등록하는 레지스트리 패턴 적용 가능

    def run_cycle(self, market_data: dict = None, news_data: list = None):
        """
        한 번의 분석 사이클 실행:
        1. 뉴스 분석 -> 2. (필요시) 시장 에이전트에 경고 -> 3. 시장 분석 -> 4. 종합 판단
        데이터가 None이면 각 에이전트가 스스로 수집하도록 함.
        """
        print("[Orchestrator] Starting Analysis Cycle...")
        
        # 1. 뉴스 분석 (데이터 없으면 내부 수집 로직 사용 - NewsAgent 구현에 따라 다름)
        # NewsAgent.process가 None을 받으면 최신 뉴스를 가져오도록 구현되어 있다고 가정하거나, 여기서 수집
        if news_data is None:
             # 실제 환경에서는 NewsCollector에서 가져와야 함.
             # 임시로 빈 리스트 넘기면 NewsAgent가 알아서 처리하거나 Mock 사용
             news_data = [] 

        news_result = self.news_agent.process(news_data)
        self.store.log_message(
            sender="agent_news", 
            receiver="orchestrator", 
            msg_type="news_alert", 
            content=news_result
        )
        
        # 2. 뉴스 영향도 체크 (라우팅 로직)
        impact = news_result.get('impact_score', 0)
        risk_alert = None
        
        if impact < -0.5:
            print("[Orchestrator] High Risk News Detected! Alerting Market Agents.")
            risk_alert = {
                "level": "high", 
                "source": "news", 
                "msg": f"Negative news impact: {news_result.get('summary')}"
            }
        
        # 3. 코인 시장 분석 (뉴스 리스크 반영)
        # market_data가 None이면 CoinAgent가 DB에서 직접 조회함
        coin_result = self.coin_agent.process(market_data)
        
        # (시뮬레이션) 만약 리스크가 감지되었다면 코인 에이전트의 결과를 덮어쓰거나 재요청한다고 가정
        if risk_alert:
            coin_result['risk_level'] = "high"
            if "News Risk Reflected" not in coin_result['summary']:
                coin_result['summary'] += " (News Risk Reflected)"
            
        self.store.log_message(
            sender="agent_coin",
            receiver="orchestrator",
            msg_type="market_signal",
            content=coin_result
        )

        print("Cycle Complete.")
        return {
            "global_status": "risk_on" if impact > -0.3 else "risk_off",
            "news_summary": news_result,
            "coin_summary": coin_result
        }

if __name__ == "__main__":
    # 테스트 실행
    orch = Orchestrator()
    
    # 더미 데이터
    dummy_market = {"price": 50000, "rsi": 45, "volume": 1000}
    dummy_news = [{"title": "SEC announces new crypto regulations", "content": "..."}]
    
    result = orch.run_cycle(dummy_market, dummy_news)
    print("\n--- Final Report ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

