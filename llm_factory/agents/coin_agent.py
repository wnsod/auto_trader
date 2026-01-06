import json
import sqlite3
import os
from datetime import datetime
from .base_agent import BaseAgent
from llm_factory.orchestrator.schemas import MarketSignal

# 코인 시장 DB 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COIN_TRADING_DB = os.path.join(PROJECT_ROOT, 'market', 'coin_market', 'data_storage', 'trading_system.db')

class CoinAgent(BaseAgent):
    """
    Virtual Trading Intelligence Engine (Alpha Guardian Voice)
    
    이 엔진은 가상매매 시스템(Alpha Guardian)의 원천 데이터와 실행 로그를 분석하여,
    투자 전략의 의도를 해석하고 대시보드에 필요한 '전략적 통찰'을 생성합니다.
    """
    def __init__(self):
        super().__init__(agent_name="agent_coin")
        self.db_path = COIN_TRADING_DB

    def _get_system_context(self, target_coin: str = None) -> dict:
        """가상매매 시스템의 전체 맥락(Context) 수집 엔진"""
        try:
            if not os.path.exists(self.db_path):
                print(f"[IntelligenceEngine] DB not found: {self.db_path}")
                return {}
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # 1. 포트폴리오 상태 엔진 (보유 현황 및 성과)
                positions = []
                pos_query = "SELECT * FROM virtual_positions ORDER BY profit_loss_pct DESC"
                if target_coin:
                    pos_query = "SELECT * FROM virtual_positions WHERE coin = ?"
                    cursor = conn.execute(pos_query, (target_coin,))
                else:
                    cursor = conn.execute(pos_query + " LIMIT 10")
                positions = [dict(row) for row in cursor.fetchall()]
                
                # 2. 알파 가디언 결정 엔진 로그 (AI의 속마음)
                guardian_thoughts = []
                guardian_query = """
                    SELECT * FROM virtual_trade_decisions 
                    WHERE ai_reason IS NOT NULL 
                    ORDER BY timestamp DESC LIMIT 5
                """
                cursor = conn.execute(guardian_query)
                guardian_thoughts = [dict(row) for row in cursor.fetchall()]
                
                # 🆕 2-1. 최근 거래 히스토리 (성공/실패 복기)
                recent_history = []
                history_query = """
                    SELECT * FROM virtual_trade_history 
                    ORDER BY exit_timestamp DESC LIMIT 3
                """
                cursor = conn.execute(history_query)
                recent_history = [dict(row) for row in cursor.fetchall()]
                
                # 3. 시장 환경 정보 (레짐, 변동성, 스캔 상태)
                status_dict = {}
                try:
                    cursor = conn.execute("SELECT key, value FROM system_status")
                    for row in cursor.fetchall():
                        status_dict[row['key']] = row['value']
                except: pass
                
                return {
                    "positions": positions,
                    "guardian_thoughts": guardian_thoughts,
                    "recent_history": recent_history,
                    "market_regime": status_dict.get('market_regime', 'Neutral'),
                    "scanning_count": len(status_dict.get('scanning_coins', '').split(',')) if status_dict.get('scanning_coins') else 0,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"[IntelligenceEngine] Context Collection Error: {e}")
            return {}

    def process(self, input_data: dict = None) -> dict:
        """
        [지능형 판단 루프]
        Raw 데이터를 전략적 문장으로 변환하는 핵심 엔진 로직
        """
        # 1. 엔진 가동을 위한 컨텍스트 로드
        # input_data가 dict 형식이면 'coin' 키를 확인, 아니면 None
        target_coin = None
        if isinstance(input_data, dict):
            target_coin = input_data.get('coin')
        
        context = self._get_system_context(target_coin)
        
        if not context:
            return {"error": "Failed to collect system context"}

        # 2. 알파 가디언 엔진의 의도 해석을 위한 프롬프트 구성
        prompt = self._build_strategic_prompt(context, target_coin)
        
        # 3. LLM을 통한 최종 전략 메시지 생성 (Alpha Guardian의 목소리)
        system_role = (
            "너는 섀도우 트레이딩 시스템의 매매 엔진인 '알파 가디언(Alpha Guardian)'의 수석 전략가야. "
            "단순히 지표를 읽는 게 아니라, 왜 그런 매매 결정을 내렸는지(Intent)를 투자자에게 '전문적이고 신뢰감 있게' 설명해야 해. "
            "가상매매 데이터임을 명시하면서도, 논리는 매우 정교해야 한다."
        )
        
        llm_response = self.call_llm(prompt, system_role=system_role)
        
        # 4. 파싱 및 반환
        return self._parse_engine_response(llm_response, context)

    def _build_strategic_prompt(self, ctx: dict, target_coin: str) -> str:
        """전략적 프롬프트 구성 엔진"""
        pos_summary = "\n".join([
            f"- {p['coin']}: ROI {p['profit_loss_pct']:+.2f}%, 상태: {p.get('trend_type', '횡보')} "
            f"(Fractal: {p.get('fractal_score', 0.5):.2f}, MTF: {p.get('mtf_score', 0.5):.2f}, Cross: {p.get('cross_score', 0.5):.2f})"
            for p in ctx['positions']
        ])
        thought_summary = "\n".join([
            f"- [{t['coin']}] {t['decision'].upper()}: {t['ai_reason']} (AI Score: {t['ai_score']:.2f}, "
            f"Fractal: {t.get('fractal_score', 0.5):.2f}, MTF: {t.get('mtf_score', 0.5):.2f}, Cross: {t.get('cross_score', 0.5):.2f})"
            for t in ctx['guardian_thoughts']
        ])
        
        # 🆕 최근 거래 복기 추가 (정밀 분석 점수 포함)
        history_summary = "\n".join([
            f"- {h['coin']}: ROI {h['profit_loss_pct']:+.2f}% ({h['action']}), 사유: {h.get('ai_reason', '기술적 청산')} "
            f"(Fractal: {h.get('fractal_score', 0.5):.2f}, MTF: {h.get('mtf_score', 0.5):.2f}, Cross: {h.get('cross_score', 0.5):.2f})"
            for h in ctx.get('recent_history', [])
        ])
        
        scope_desc = f"특정 종목({target_coin}) 분석" if target_coin else "전체 포트폴리오 전략"
        
        return f"""
        [Alpha Guardian System Context] - Scope: {scope_desc}
        - Current Regime: {ctx['market_regime']}
        - Active Positions:
        {pos_summary if pos_summary else "No active positions."}
        
        - Recent Trade Results (Self-Reflection):
        {history_summary if history_summary else "No recent trades completed."}
        
        - Alpha Guardian's Direct Thoughts (Real-time Analysis):
        {thought_summary if thought_summary else "No recent strategic updates."}
        
        [Task]
        위의 시스템 데이터를 바탕으로 현재의 운용 전략을 '알파 가디언'의 관점에서 요약해줘.
        1. 현재 시장 상황에 대한 엔진의 판단 (Regime 분석)
        2. 주요 포지션 유지 혹은 매도 사유 (Guardian의 판단 근거 활용)
        3. 향후 대응 계획
        
        모든 답변은 한국어로, 대시보드에 표시될 'summary' 필드에 집중해서 작성할 것.
        반드시 아래 JSON 스키마를 준수해. (Markdown 금지)
        
        {{
            "regime": "Bull/Bear/Neutral/High_Volatility",
            "confidence": 0.0 ~ 1.0,
            "risk_level": "High/Medium/Low",
            "summary": "알파 가디언의 전략적 요약 문장",
            "key_factors": ["핵심 판단 요소 1", "핵심 판단 요소 2"]
        }}
        """

    def _parse_engine_response(self, response: str, ctx: dict) -> dict:
        """LLM 응답 파싱 및 엔진 데이터 결합"""
        if not response:
            return self._get_fallback_response(ctx)
            
        try:
            clean_json = response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            parsed["market"] = "coin"
            parsed["timestamp"] = datetime.now().isoformat()
            
            return MarketSignal(**parsed).dict()
        except Exception as e:
            print(f"[IntelligenceEngine] Parsing Failed: {e}")
            return self._get_fallback_response(ctx)

    def _get_fallback_response(self, ctx: dict) -> dict:
        """엔진 Fallback 로직"""
        return {
            "market": "coin",
            "timestamp": datetime.now().isoformat(),
            "regime": ctx['market_regime'],
            "confidence": 0.5,
            "risk_level": "Medium",
            "summary": f"현재 {ctx['market_regime']} 시장 레짐 하에 알파 가디언 엔진이 안정적으로 가상매매를 수행 중입니다.",
            "key_factors": ["시스템 데이터 수집 완료", "레짐 분석 수행 중"]
        }
