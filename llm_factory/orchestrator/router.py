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

    def run_cycle(self, market_data: dict, news_data: list):
        """
        한 번의 분석 사이클 실행:
        1. 뉴스 분석 -> 2. (필요시) 시장 에이전트에 경고 -> 3. 시장 분석 -> 4. 종합 판단
        """
        # print("\U0001f504 [Orchestrator] Starting Analysis Cycle...")
        print("[Orchestrator] Starting Analysis Cycle...")
        
        # 1. 뉴스 분석
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
            # print("🚨 [Orchestrator] High Risk News Detected! Alerting Market Agents.")
            print("[Orchestrator] High Risk News Detected! Alerting Market Agents.")
            risk_alert = {
                "level": "high", 
                "source": "news", 
                "msg": f"Negative news impact: {news_result.get('summary')}"
            }
            # 실제로는 여기서 coin_agent에게 "보수적으로 봐라"는 프롬프트를 주입할 수 있음.
        
        # 3. 코인 시장 분석 (뉴스 리스크 반영)
        # 에이전트에게 리스크 정보를 컨텍스트로 전달하는 로직이 필요함 (여기선 mock data에 반영 안됨)
        coin_result = self.coin_agent.process(market_data)
        
        # (시뮬레이션) 만약 리스크가 감지되었다면 코인 에이전트의 결과를 덮어쓰거나 재요청한다고 가정
        if risk_alert:
            coin_result['risk_level'] = "high"
            coin_result['summary'] += " (News Risk Reflected)"
            
        self.store.log_message(
            sender="agent_coin",
            receiver="orchestrator",
            msg_type="market_signal",
            content=coin_result
        )

        # print("✅ Cycle Complete.")
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

