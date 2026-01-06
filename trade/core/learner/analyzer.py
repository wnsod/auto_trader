#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
패턴 분석 시스템 - 시그널 및 시장 상황을 학습용 패턴으로 정규화
"""

from typing import Dict, Any, Optional
from trade.core.models import SignalInfo

class PatternAnalyzer:
    """패턴 분석 및 정규화 도구"""
    
    def __init__(self):
        self.min_confidence = 0.4

    def extract_learning_pattern(self, signal: SignalInfo, market_context: Dict) -> str:
        """시그널과 시장 상황을 조합하여 고유 학습 패턴 생성"""
        try:
            # 1. 시그널 기반 핵심 상태
            rsi = getattr(signal, 'rsi', 50.0)
            vol = getattr(signal, 'volume_ratio', 1.0)
            rsi_state = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            vol_state = 'high_vol' if vol > 1.5 else 'low_vol' if vol < 0.5 else 'normal_vol'
            
            # 🆕 코인별 특성 분류 (Major vs. Alt)
            major_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
            coin_type = 'major' if any(m in signal.coin for m in major_coins) else 'alt'
            
            # 2. 시장 상황 (레짐) 반영
            regime = market_context.get('regime', 'neutral').lower()
            trend = market_context.get('trend', 'sideways').lower()
            
            # 3. 보조 지표 상태
            macd = getattr(signal, 'macd_divergence', 'none')
            bb = getattr(signal, 'bb_position', 'mid')
            
            # 🆕 조합: 코인종류_시장레짐_추세_RSI상태_거래량
            pattern = f"{coin_type}_{regime}_{trend}_{rsi_state}_{vol_state}"
            
            # 특이 패턴 추가 (다이버전스 등)
            if macd != 'none':
                pattern += f"_macd_{macd}"
            if bb in ['upper', 'lower']:
                pattern += f"_bb_{bb}"
                
            return pattern
        except Exception as e:
            print(f"⚠️ 패턴 추출 오류: {e}")
            return "unknown_basic_pattern"

    def analyze_pattern_efficiency(self, pattern_stats: Dict) -> Dict:
        """패턴의 학습 효율성 분석 (신뢰도, 기회비용 등)"""
        total = pattern_stats.get('total_samples', 0)
        alpha = pattern_stats.get('alpha', 1.0)
        beta = pattern_stats.get('beta', 1.0)
        
        success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        confidence = min(1.0, total / 20.0) # 20회 이상일 때 신뢰도 1.0
        
        return {
            'success_rate': success_rate,
            'confidence': confidence,
            'is_reliable': confidence > 0.7 and success_rate > 0.6,
            'needs_more_data': total < 5
        }

