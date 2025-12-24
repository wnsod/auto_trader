"""
트레이딩 코어 매니저 (Core Manager)
- AI, Risk, Market 등 핵심 모듈의 통합 관리 및 초기화
- 가상/실전 매매 Executor에서 공통으로 사용
"""
import os
import sys
import logging

# 🆕 공통 코어 모듈 임포트
from trade.core.ai import AIDecisionEngine
from trade.core.risk import RiskManager, OutlierGuardrail
from trade.core.tracker import ActionPerformanceTracker, ContextRecorder, LearningFeedback
from trade.core.market import MarketAnalyzer
# 🆕 통합 의사결정 시스템
from trade.core.judgement import JudgementSystem, DecisionType, JudgementResult

# 🆕 코인 마켓 분석기 (Optional)
try:
    import market.coin_market.market_analyzer as coin_analyzer
    COIN_MARKET_AVAILABLE = True
except ImportError:
    # 경로 문제로 임포트 실패 시 path 추가 후 재시도
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import market.coin_market.market_analyzer as coin_analyzer
        COIN_MARKET_AVAILABLE = True
    except ImportError:
        COIN_MARKET_AVAILABLE = False
        coin_analyzer = None

class CoreManager:
    """트레이딩 시스템의 핵심 모듈들을 관리하는 매니저"""
    
    def __init__(self):
        # 1. 핵심 엔진 초기화
        self.ai_engine = AIDecisionEngine()
        self.risk_manager = RiskManager()
        self.market_analyzer = MarketAnalyzer()
        
        # 2. 성과 추적 및 학습 시스템 초기화
        self.action_tracker = ActionPerformanceTracker()
        self.context_recorder = ContextRecorder()
        self.learning_feedback = LearningFeedback()
        self.outlier_guardrail = OutlierGuardrail()
        
        # 3. 마켓 어댑터 연결
        self.coin_analyzer = coin_analyzer if COIN_MARKET_AVAILABLE else None
        
        # 4. 통합 의사결정 시스템
        self.judgement_system = JudgementSystem()
        
        print("✅ Core Manager 초기화 완료 (AI, Risk, Market, Tracker, Judgement)")

    def evaluate_signal(self, 
                       signal_info, 
                       thompson_prob: float = 0.5, 
                       mode: str = 'real') -> JudgementResult:
        """
        통합 시그널 평가 (Judgement System)
        mode: 'real' (실전, 엄격함) or 'simulation' (가상, 관대함)
        """
        try:
            # 리스크 레벨 및 시장 상황 자동 분석
            # signal_info가 dict인 경우와 객체인 경우 모두 처리
            if isinstance(signal_info, dict):
                risk_level = signal_info.get('risk_level', 'medium')
            else:
                risk_level = getattr(signal_info, 'risk_level', 'medium')
                
            # Market Analyzer를 통해 시장 상황 조회 (여기서 직접 호출하여 일관성 유지)
            # 단, signal_info에 market_context가 포함되어 있다면 그걸 쓸 수도 있음
            # 여기서는 현재 시장 상황을 다시 조회 (최신성 보장)
            # 주의: signal_info가 특정 코인의 시그널이라면, 해당 코인 기준이 아니라 BTC 기준 시장 상황이 필요할 수 있음
            # MarketAnalyzer.get_market_context_from_signal() 활용
            
            market_context = {'trend': 'neutral', 'volatility': 0.02}
            try:
                # signal_info를 기반으로 시장 상황 추론 (보완 필요)
                market_context = self.market_analyzer.get_market_context_from_signal(signal_info)
            except Exception:
                pass

            # Judgement 평가
            result = self.judgement_system.evaluate(
                signal_info=signal_info,
                thompson_prob=thompson_prob,
                risk_level=risk_level,
                market_context=market_context
            )
            
            # 모드에 따른 최종 의사결정 보정 (Threshold 적용)
            # JudgementSystem.evaluate는 기본적으로 0.7/0.3 기준만 적용하므로,
            # 시뮬레이션 모드에서는 더 관대한 기준을 적용하여 Decision을 변경해줌
            
            if mode == 'simulation':
                # 가상 매매: 0.5 이상이면 PROMOTE (기존 HOLD -> PROMOTE)
                if result.decision == DecisionType.HOLD and result.score >= 0.5:
                    # 원본을 수정하지 않고 새로운 결과 객체 반환 (불변성 유지 권장)
                    # dataclass replace 사용 또는 속성 변경 (Python은 mutable)
                    result.decision = DecisionType.PROMOTE
                    result.reasons.append(f"가상 매매 기준 완화 (Score >= 0.5)")
            
            return result
            
        except Exception as e:
            print(f"⚠️ [Core] 시그널 평가 중 오류: {e}")
            # 오류 시 보수적으로 HOLD 반환
            from trade.core.judgement import JudgementComponents
            return JudgementResult(
                score=0.0,
                decision=DecisionType.HOLD,
                components=JudgementComponents(),
                reasons=[f"평가 오류: {str(e)}"]
            )

    def prefetch_market_data(self):
        """마켓 데이터 사전 로드 (딜레이 방지)"""
        if self.coin_analyzer:
            print("🔄 [Core] 펀더멘탈 데이터 사전 로드 중...")
            try:
                # 상위 500개 코인 데이터 일괄 업데이트
                self.coin_analyzer.fetch_fundamentals_from_coingecko(coins=None)
                print("✅ [Core] 펀더멘탈 데이터 로드 완료")
            except Exception as e:
                print(f"⚠️ [Core] 펀더멘탈 로드 실패: {e}")
    
    def get_fundamental_data(self, coin: str):
        """펀더멘탈 데이터 조회 (Safe Proxy)"""
        if self.coin_analyzer:
            try:
                return self.coin_analyzer.get_fundamental_data(coin)
            except Exception:
                return None
        return None

    def calculate_fundamental_score(self, fund_data: dict) -> float:
        """펀더멘탈 점수 계산 (Safe Proxy)"""
        if self.coin_analyzer and fund_data:
            try:
                return self.coin_analyzer.calculate_fundamental_score(fund_data)
            except Exception:
                return 0.0
        return 0.0

