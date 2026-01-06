#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Sequence Analyzer - 최근 5개 캔들 기반 정밀 분석 모듈
사용자 요청: 중장기(방향성), 단기(타이밍) 분석 강화
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

class SequenceAnalyzer:
    """최근 5개 캔들의 흐름(Sequence)을 분석하여 매매 의사결정을 보조"""
    
    @staticmethod
    def analyze_sequence(df: pd.DataFrame, interval: str) -> Dict:
        """
        최근 5개 캔들 데이터(df)를 분석
        df는 최신순(index 0이 가장 최신)으로 정렬되어 있어야 함
        """
        if df is None or len(df) < 5:
            return {'score_mod': 1.0, 'reason': '데이터 부족', 'passed': True}

        # 분석용 데이터 (과거 -> 현재 순으로 정렬)
        recent = df.head(5).iloc[::-1].reset_index(drop=True)
        
        # 1. 공통 지표 계산
        closes = recent['close'].values
        lows = recent['low'].values
        highs = recent['high'].values
        
        # 가격 기울기 (정규화)
        price_slope = np.polyfit(np.arange(5), closes / closes[0], 1)[0]
        
        # 2. 인터벌별 차별화 분석
        is_short_term = interval in ['15m', '30m']
        
        if is_short_term:
            # [단기/중단기] 타이밍(Timing) 분석: 모멘텀의 변화량 중시
            return SequenceAnalyzer._analyze_timing(recent, price_slope)
        else:
            # [중장기/장기] 방향성(Direction) 분석: 저점/고점의 경로 중시
            return SequenceAnalyzer._analyze_direction(recent, price_slope)

    @staticmethod
    def _analyze_timing(recent: pd.DataFrame, price_slope: float) -> Dict:
        """단기 타이밍 분석 (RSI, MACD 기울기 중시)"""
        reasons = []
        score_mod = 1.0
        
        # RSI 기울기 (있는 경우)
        if 'rsi' in recent.columns:
            rsi_values = recent['rsi'].values
            rsi_slope = np.polyfit(np.arange(5), rsi_values, 1)[0]
            if rsi_slope > 1.5: # RSI 급상승 중
                score_mod *= 1.2
                reasons.append(f"RSI 상승세({rsi_slope:.1f})")
            elif rsi_slope < -1.5: # RSI 급하락 중
                score_mod *= 0.8
                reasons.append(f"RSI 하락세({rsi_slope:.1f})")

        # 가격 모멘텀 가속도
        if price_slope > 0.005: # 강한 단기 상승
            score_mod *= 1.1
            reasons.append("단기 모멘텀 강함")
        
        return {
            'score_mod': score_mod,
            'reason': ", ".join(reasons) if reasons else "단기 흐름 중립",
            'passed': score_mod >= 0.9  # 너무 강한 하락세면 False 가능
        }

    @staticmethod
    def _analyze_direction(recent: pd.DataFrame, price_slope: float) -> Dict:
        """중장기 방향성 분석 (저점/고점 경로 중시)"""
        reasons = []
        score_mod = 1.0
        
        lows = recent['low'].values
        highs = recent['high'].values
        
        # 저점/고점 갱신 확인
        higher_lows = sum(1 for i in range(1, 5) if lows[i] > lows[i-1])
        higher_highs = sum(1 for i in range(1, 5) if highs[i] > highs[i-1])
        
        if higher_lows >= 3: # 저점이 높아지는 중
            score_mod *= 1.15
            reasons.append(f"저점 상승({higher_lows}/4)")
        elif higher_lows <= 1: # 저점이 낮아지는 중
            score_mod *= 0.85
            reasons.append(f"저점 하락 우려")

        if price_slope > 0.002: # 완만한 장기 우상향
            score_mod *= 1.1
            reasons.append("장기 방향성 상향")
        elif price_slope < -0.002:
            score_mod *= 0.9
            reasons.append("장기 방향성 하향")

        return {
            'score_mod': score_mod,
            'reason': ", ".join(reasons) if reasons else "장기 흐름 중립",
            'passed': score_mod >= 0.85
        }
