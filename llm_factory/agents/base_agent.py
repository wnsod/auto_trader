import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from llm_factory.utils.llm_client import get_openai_client

class BaseAgent(ABC):
    """모든 LLM 에이전트의 부모 클래스"""
    
    def __init__(self, agent_name: str, model_name: str = "gpt-4o-mini"):
        self.agent_name = agent_name
        self.model_name = model_name
        self.client = get_openai_client()

    @abstractmethod
    def process(self, input_data: Any) -> Dict:
        """입력 데이터를 받아 분석 결과(JSON)를 반환해야 함"""
        pass

    def call_llm(self, prompt: str, system_role: str = "You are a financial analyst.") -> str:
        """실제 LLM 호출"""
        if not self.client:
            return None # Mock 모드로 Fallback

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            # print(f"❌ [LLM Error] {e}")
            print(f"[LLM Error] {e}")
            return None

    def _mock_llm_inference(self, prompt: str, mock_response: Dict) -> Dict:
        """LLM 호출을 흉내내는 메서드 (API 연결 전 테스트용)"""
        # 실제로는 여기서 OpenAI API 등을 호출
        # print(f"🤖 [{self.agent_name}] Thinking...\nPrompt: {prompt[:50]}...")
        return mock_response

