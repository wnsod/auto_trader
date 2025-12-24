"""
AI 의사결정 엔진 (Core AI)
- 가상/실전 매매에서 공통으로 사용하는 지능형 의사결정 로직
- 패턴 인식, 시장 상황 분석, 리스크 평가를 종합하여 최종 판단
"""
import time
from typing import Dict, Optional
from trade.signal_selector.core.types import SignalInfo

class AIDecisionEngine:
    """AI 의사결정 엔진 - 지능형 거래 결정"""
    def __init__(self):
        self.decision_history = []
        
    def make_trading_decision(self, signal: SignalInfo, current_price: float, market_context: dict, coin_performance: Optional[dict] = None) -> str:
        """지능형 거래 결정 (코인 성과 반영)"""
        try:
            # 패턴 인식 기반 결정
            pattern_score = self._analyze_pattern(signal)
            
            # 시장 맥락 기반 결정
            context_score = self._analyze_market_context(market_context)
            
            # 리스크 평가
            risk_score = self._evaluate_risk(signal)
            
            # 코인 성과 보너스 계산
            coin_bonus = 0.0
            if coin_performance:
                coin_bonus = self._calculate_coin_bonus(coin_performance)
            
            # 최종 결정 (코인 보너스 반영)
            decision = self._make_final_decision(pattern_score, context_score, risk_score, signal, coin_bonus)
            
            # 결정 기록
            self.decision_history.append({
                'timestamp': time.time(),
                'signal': signal,
                'decision': decision,
                'scores': {
                    'pattern': pattern_score,
                    'context': context_score,
                    'risk': risk_score,
                    'coin_bonus': coin_bonus
                }
            })
            
            return decision
            
        except Exception as e:
            print(f"⚠️ AI 의사결정 오류: {e}")
            return 'hold'
            
    def _calculate_coin_bonus(self, coin_performance: dict) -> float:
        """코인 성과 기반 보너스 계산"""
        try:
            success_rate = coin_performance.get('success_rate', 0.5)
            avg_profit = coin_performance.get('avg_profit', 0.0)
            total_trades = coin_performance.get('total_trades', 0)
            
            if total_trades < 5:
                return 0.0
            
            # 성과 기반 보너스 (성공률과 수익률 고려)
            # avg_profit은 % 단위라고 가정
            performance_bonus = (success_rate - 0.5) * 0.2 + avg_profit * 0.05
            return max(-0.1, min(0.1, performance_bonus))
        except:
            return 0.0

    
    def _analyze_pattern(self, signal: SignalInfo) -> float:
        """패턴 분석"""
        try:
            # RSI 패턴 분석
            rsi_score = self._analyze_rsi_pattern(signal.rsi)
            
            # MACD 패턴 분석
            macd_score = self._analyze_macd_pattern(signal.macd)
            
            # 볼륨 패턴 분석
            volume_score = self._analyze_volume_pattern(signal.volume_ratio)
            
            # 종합 패턴 점수
            pattern_score = (rsi_score + macd_score + volume_score) / 3
            
            return pattern_score
            
        except Exception as e:
            print(f"⚠️ 패턴 분석 오류: {e}")
            return 0.5
    
    def _analyze_rsi_pattern(self, rsi: float) -> float:
        """RSI 패턴 분석"""
        if rsi < 30:
            return 0.8  # 과매도 - 매수 기회
        elif rsi < 45:
            return 0.6  # 낮은 RSI - 약간의 매수 기회
        elif rsi < 55:
            return 0.5  # 중립
        elif rsi < 70:
            return 0.4  # 높은 RSI - 약간의 매도 기회
        else:
            return 0.2  # 과매수 - 매도 기회
    
    def _analyze_macd_pattern(self, macd: float) -> float:
        """MACD 패턴 분석"""
        if macd > 0.1:
            return 0.8  # 강한 상승 모멘텀
        elif macd > 0:
            return 0.6  # 약한 상승 모멘텀
        elif macd > -0.1:
            return 0.4  # 약한 하락 모멘텀
        else:
            return 0.2  # 강한 하락 모멘텀
    
    def _analyze_volume_pattern(self, volume_ratio: float) -> float:
        """볼륨 패턴 분석"""
        if volume_ratio > 2.0:
            return 0.8  # 높은 거래량 - 강한 신호
        elif volume_ratio > 1.5:
            return 0.7  # 증가한 거래량
        elif volume_ratio > 0.8:
            return 0.5  # 정상 거래량
        else:
            return 0.3  # 낮은 거래량 - 약한 신호
    
    def _analyze_market_context(self, market_context: dict) -> float:
        """시장 맥락 분석"""
        try:
            trend = market_context.get('trend', 'neutral')
            volatility = market_context.get('volatility', 0.02)
            
            # 트렌드 기반 점수
            if trend == 'bullish':
                trend_score = 0.7
            elif trend == 'bearish':
                trend_score = 0.3
            else:
                trend_score = 0.5
            
            # 변동성 기반 점수 (적당한 변동성이 좋음)
            if 0.01 < volatility < 0.05:
                vol_score = 0.8
            elif volatility < 0.01:
                vol_score = 0.4  # 너무 낮은 변동성
            else:
                vol_score = 0.3  # 너무 높은 변동성
            
            return (trend_score + vol_score) / 2
            
        except Exception as e:
            print(f"⚠️ 시장 맥락 분석 오류: {e}")
            return 0.5
    
    def _evaluate_risk(self, signal: SignalInfo) -> float:
        """리스크 평가"""
        try:
            # 신호 신뢰도 기반 리스크
            confidence_risk = 1.0 - signal.confidence
            
            # 변동성 기반 리스크
            volatility_risk = min(signal.volatility * 10, 1.0)
            
            # 종합 리스크 점수 (낮을수록 좋음)
            risk_score = 1.0 - (confidence_risk + volatility_risk) / 2
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            print(f"⚠️ 리스크 평가 오류: {e}")
            return 0.5
    
    def _make_final_decision(self, pattern_score: float, context_score: float, risk_score: float, signal: SignalInfo, coin_bonus: float = 0.0) -> str:
        """최종 거래 결정"""
        try:
            # 가중 평균 점수 계산
            base_score = (pattern_score * 0.4 + context_score * 0.3 + risk_score * 0.3)
            
            # 코인 보너스 적용
            final_score = base_score + coin_bonus
            
            # 신호 점수와 결합
            combined_score = (final_score + signal.signal_score) / 2
            
            # 결정 임계값
            if combined_score > 0.7:
                return 'buy'
            elif combined_score < 0.3:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            print(f"⚠️ 최종 결정 오류: {e}")
            return 'hold'

