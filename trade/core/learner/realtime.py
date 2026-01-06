#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 학습기 - 실시간 위험 감지 및 즉각적 피드백 반영
"""

from typing import Dict, List, Optional
import time

class RealTimeLearner:
    """실시간 위험 학습 및 대응 시스템"""
    
    def __init__(self, thompson_sampler):
        self.sampler = thompson_sampler
        self.recent_events = []
        self.risk_threshold = -2.0  # -2% 손실 시 즉각 위험 학습

    def learn_from_trade(self, pattern: str, profit_pct: float):
        """완료된 거래로부터 즉시 학습"""
        success = profit_pct > 0
        self.sampler.update_distribution(
            pattern=pattern,
            success=success,
            profit_pct=profit_pct,
            weight=1.0
        )

    def learn_from_ongoing_drawdown(self, pattern: str, current_profit: float):
        """
        진행 중인 미실현 손실로부터 즉각적인 위험 학습
        (손절 전이라도 위험 패턴을 미리 인식하도록 함)
        """
        if current_profit <= self.risk_threshold:
            print(f"🚨 [실시간 위험 감지] {pattern} 패턴 {current_profit:.2f}% 손실 중... 즉시 위험 학습 반영")
            
            # 실패 가중치를 부여하여 Thompson 분포 업데이트 (weight 조절로 즉각 반응)
            # 아직 확정된 손실은 아니므로 weight=0.5 적용
            self.sampler.update_distribution(
                pattern=pattern,
                success=False,
                profit_pct=current_profit,
                weight=0.5
            )
            
            # 위험 패턴으로 별도 마킹 (추후 탐색 억제용)
            self._log_risk_event(pattern, current_profit)

    def _log_risk_event(self, pattern: str, drawdown: float):
        self.recent_events.append({
            'timestamp': time.time(),
            'pattern': pattern,
            'drawdown': drawdown,
            'type': 'risk_warning'
        })
        # 최근 100개만 유지
        if len(self.recent_events) > 100:
            self.recent_events.pop(0)

