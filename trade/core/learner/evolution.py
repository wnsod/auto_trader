#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시스템 진화 및 피드백 처리 엔진
학습 결과를 바탕으로 전략 매개변수를 조정하고 시스템 성능을 최적화
"""

import time
import json
from typing import Dict, List, Any

class FeedbackProcessor:
    """피드백 처리기 - 거래 결과 피드백 분석"""
    def __init__(self):
        self.feedback_queue = []
        self.processed_feedback = {}
        
    def process_feedback(self, trade_data: dict) -> dict:
        # 피드백 분석 로직
        return {}

class EvolutionEngine:
    """진화 엔진 - 성과 데이터를 기반으로 시스템 자가 진화 (🆕알파 가디언 자가반성 로직 포함)"""
    def __init__(self, strategy_db_path: str):
        self.strategy_db_path = strategy_db_path
        self.evolution_history = []
        self.performance_trends = {}
        
    def evaluate_decision_quality(self, trade_history: List[Dict], guardian) -> Dict:
        """
        🛡️ 알파 가디언의 사후 검증 시뮬레이션
        과거 시점에 현재의 알파 가디언이 있었다면 어땠을지를 시뮬레이션하여 평가
        
        🆕 시장 상황별로 성과를 분리하여 반환
        """
        if not trade_history:
            return {'buy_accuracy': 0.5, 'buy_count': 0, 'by_market': {}}
        
        # 🆕 시장 상황별 성과 추적
        market_performance = {}  # {market_type: {sim_success, sim_fail, missed_win, sell_success, sell_fail}}
        overall_sim_success = 0
        overall_sim_fail = 0
        overall_missed_win = 0
        overall_sell_success = 0  # 🆕 매도 성공
        overall_sell_fail = 0  # 🆕 매도 실패
            
        for trade in trade_history:
            # 1. 과거 데이터 복원
            try:
                market_context = json.loads(trade.get('market_conditions', '{}'))
            except:
                market_context = {'trend': 'neutral', 'volatility': 'medium'}
            
            # 🆕 시장 상황 분류
            market_type = guardian._classify_market_context(market_context) if hasattr(guardian, '_classify_market_context') else 'neutral'
            if market_type not in market_performance:
                market_performance[market_type] = {
                    'sim_success': 0,
                    'sim_fail': 0,
                    'missed_win': 0,
                    'sell_success': 0,  # 🆕 매도 성공
                    'sell_fail': 0  # 🆕 매도 실패
                }
                
            signal_data = {
                'coin': trade.get('coin'),
                'signal_score': trade.get('entry_signal_score', 0.0),
                'confidence': trade.get('entry_confidence', 0.5),
                'risk_level': trade.get('entry_risk_level', 'medium')
            }
            current_price = trade.get('entry_price', 0.0)
            coin_performance = {} # 간소화를 위해 일단 빈값
            
            # 2. 현재 알파 가디언에게 물어봄: "너라면 샀겠어?"
            sim_decision_result = guardian.make_trading_decision(
                signal_data, current_price, market_context, coin_performance
            )
            
            # 🔧 dict 반환값 처리
            if isinstance(sim_decision_result, dict):
                sim_decision = sim_decision_result.get('decision', 'hold')
            else:
                sim_decision = sim_decision_result
            
            actual_profit = trade.get('profit_loss_pct', 0.0)
            
            # 3. 결과 비교 (전체 + 시장 상황별)
            if sim_decision == 'buy' or sim_decision == 'BUY':
                if actual_profit > 0:
                    overall_sim_success += 1
                    market_performance[market_type]['sim_success'] += 1
                else:
                    overall_sim_fail += 1
                    market_performance[market_type]['sim_fail'] += 1
            else: # 'hold' (안 샀음)
                if actual_profit > 1.0: # 안 샀는데 1% 이상 올랐다면?
                    overall_missed_win += 1
                    market_performance[market_type]['missed_win'] += 1
            
            # 🆕 매도 판단 시뮬레이션 (보유 중인 포지션에서 매도 시점 평가)
            # 손실 거래에서 매도 판단이 적절했는지 평가
            if actual_profit < 0:  # 손실 거래만 평가
                exit_signal_score = trade.get('exit_signal_score', 0.0)  # 매도 시점의 시그널 점수
                exit_price = trade.get('exit_price', 0.0)
                entry_price = trade.get('entry_price', 0.0)
                
                # 매도 시점의 시장 상황 (진입 시점과 동일하게 가정, 실제로는 exit_timestamp 기반 조회 필요)
                exit_market_context = market_context
                
                # 매도 시점의 시그널 데이터
                exit_signal_data = {
                    'coin': trade.get('coin'),
                    'signal_score': exit_signal_score,
                    'confidence': trade.get('entry_confidence', 0.5),
                    'risk_level': trade.get('entry_risk_level', 'medium')
                }
                
                # 현재 알파 가디언에게 물어봄: "너라면 팔았겠어?"
                if exit_price > 0 and entry_price > 0:
                    sim_sell_decision_result = guardian.make_trading_decision(
                        exit_signal_data, exit_price, exit_market_context, {}
                    )
                    
                    if isinstance(sim_sell_decision_result, dict):
                        sim_sell_decision = sim_sell_decision_result.get('decision', 'hold')
                    else:
                        sim_sell_decision = sim_sell_decision_result
                    
                    # 매도 판단 평가:
                    # - 매도했는데 손실이 -10% 이상이면 실패 (너무 늦게 팔음)
                    # - 매도했는데 손실이 -10% 미만이면 성공 (적절히 팔음)
                    # - 매도 안 했는데 손실이 -10% 이상이면 실패 (팔았어야 함)
                    if sim_sell_decision == 'sell' or sim_sell_decision == 'SELL':
                        if actual_profit >= -10.0:  # 손절선(-10%) 이상 유지
                            overall_sell_success += 1
                            market_performance[market_type]['sell_success'] += 1
                        else:  # 손절선 이하로 떨어짐
                            overall_sell_fail += 1
                            market_performance[market_type]['sell_fail'] += 1
                    else:  # 매도 안 함
                        if actual_profit < -10.0:  # 손절선 이하로 떨어졌는데 안 팔음
                            overall_sell_fail += 1
                            market_performance[market_type]['sell_fail'] += 1
        
        total_sim_buys = overall_sim_success + overall_sim_fail
        accuracy = overall_sim_success / total_sim_buys if total_sim_buys > 0 else 0.5
        
        total_sell_decisions = overall_sell_success + overall_sell_fail
        sell_accuracy = overall_sell_success / total_sell_decisions if total_sell_decisions > 0 else 0.5  # 🆕 매도 정확도
        
        # 🆕 시장 상황별 정확도 계산
        by_market = {}
        for market_type, perf in market_performance.items():
            total = perf['sim_success'] + perf['sim_fail']
            sell_total = perf['sell_success'] + perf['sell_fail']
            if total > 0:
                by_market[market_type] = {
                    'buy_accuracy': perf['sim_success'] / total,
                    'buy_count': total,
                    'profit_count': perf['sim_success'],
                    'fail_count': perf['sim_fail'],
                    'missed_win_count': perf['missed_win'],
                    'sell_accuracy': perf['sell_success'] / sell_total if sell_total > 0 else 0.5,  # 🆕 매도 정확도
                    'sell_count': sell_total  # 🆕 매도 판단 횟수
                }
        
        return {
            'buy_accuracy': accuracy,
            'buy_count': total_sim_buys,
            'profit_count': overall_sim_success,
            'fail_count': overall_sim_fail,
            'missed_win_count': overall_missed_win,
            'sell_accuracy': sell_accuracy,  # 🆕 매도 정확도
            'sell_count': total_sell_decisions,  # 🆕 매도 판단 횟수
            'by_market': by_market  # 🆕 시장 상황별 성과
        }

    def update_meta_bias(self, quality: Dict, guardian=None) -> Dict:
        """
        시뮬레이션 결과를 바탕으로 알파 가디언의 성격 조정
        
        🆕 시장 상황별로 meta_bias 업데이트
        """
        # 🆕 시장 상황별 업데이트
        updated_markets = []
        by_market = quality.get('by_market', {})
        
        if by_market and guardian and hasattr(guardian, 'save_meta_bias_by_market'):
            # 시장 상황별로 개별 업데이트
            for market_type, market_quality in by_market.items():
                # 해당 시장 상황의 기존 바이어스 조회
                market_context = {'regime': market_type}  # 간단한 컨텍스트 생성
                current_bias = guardian.get_market_specific_bias(market_context)
                
                new_bias = {
                    'buy_threshold_offset': current_bias.get('buy_threshold_offset', -0.05),
                    'sell_threshold_offset': current_bias.get('sell_threshold_offset', 0.0),
                    'risk_weight_multiplier': current_bias.get('risk_weight_multiplier', 1.0)
                }
                
                # 시장 상황별 성과 기반 조정
                buy_accuracy = market_quality.get('buy_accuracy', 0.5)
                buy_count = market_quality.get('buy_count', 0)
                sell_accuracy = market_quality.get('sell_accuracy', 0.5)  # 🆕 매도 정확도
                sell_count = market_quality.get('sell_count', 0)  # 🆕 매도 판단 횟수
                
                # 최소 거래 횟수 이상일 때만 업데이트 (신뢰도 확보)
                if buy_count >= 3:
                    if buy_accuracy < 0.4:
                        new_bias['buy_threshold_offset'] = 0.10
                        new_bias['risk_weight_multiplier'] = 1.3
                    elif buy_accuracy < 0.5:
                        new_bias['buy_threshold_offset'] = 0.05
                        new_bias['risk_weight_multiplier'] = 1.1
                    elif buy_accuracy > 0.6:
                        # 성공률이 높으면 더 공격적으로
                        new_bias['buy_threshold_offset'] = max(-0.10, new_bias['buy_threshold_offset'] - 0.02)
                        new_bias['risk_weight_multiplier'] = max(0.8, new_bias['risk_weight_multiplier'] - 0.1)
                    
                    # 놓친 수익이 많으면 기준 완화
                    if market_quality.get('missed_win_count', 0) > 5:
                        new_bias['buy_threshold_offset'] -= 0.03
                
                # 🆕 매도 성과 기반 조정 (최소 판단 횟수 이상일 때만)
                if sell_count >= 3:
                    if sell_accuracy < 0.4:
                        # 매도 판단이 부정확하면 매도 임계값을 더 엄격하게 (더 일찍 팔도록)
                        new_bias['sell_threshold_offset'] = min(0.05, new_bias['sell_threshold_offset'] + 0.02)
                    elif sell_accuracy < 0.5:
                        # 매도 판단이 약간 부정확하면 약간 엄격하게
                        new_bias['sell_threshold_offset'] = min(0.03, new_bias['sell_threshold_offset'] + 0.01)
                    elif sell_accuracy > 0.7:
                        # 매도 판단이 정확하면 매도 임계값을 완화 (더 오래 보유)
                        new_bias['sell_threshold_offset'] = max(-0.03, new_bias['sell_threshold_offset'] - 0.01)
                
                # 시장 상황별 바이어스 저장
                guardian.save_meta_bias_by_market(market_type, new_bias)
                updated_markets.append(market_type)
        
        # 전역 meta_bias 업데이트 (하위 호환성 유지)
        if guardian and hasattr(guardian, 'meta_bias'):
            new_bias = {
                'buy_threshold_offset': guardian.meta_bias.get('buy_threshold_offset', -0.05),
                'sell_threshold_offset': guardian.meta_bias.get('sell_threshold_offset', 0.0),
                'risk_weight_multiplier': guardian.meta_bias.get('risk_weight_multiplier', 1.0)
            }
        else:
            new_bias = {
                'buy_threshold_offset': -0.05,
                'sell_threshold_offset': 0.0,
                'risk_weight_multiplier': 1.0
            }
        
        # 전역 성과 기반 조정
        buy_accuracy = quality.get('buy_accuracy', 0.5)
        sell_accuracy = quality.get('sell_accuracy', 0.5)  # 🆕 매도 정확도
        sell_count = quality.get('sell_count', 0)  # 🆕 매도 판단 횟수
        
        if buy_accuracy < 0.4:
            new_bias['buy_threshold_offset'] = 0.10
            new_bias['risk_weight_multiplier'] = 1.3
        elif buy_accuracy < 0.5:
            new_bias['buy_threshold_offset'] = 0.05
            new_bias['risk_weight_multiplier'] = 1.1
            
        if quality.get('missed_win_count', 0) > 10:
            new_bias['buy_threshold_offset'] -= 0.03
        
        # 🆕 전역 매도 성과 기반 조정
        if sell_count >= 3:
            if sell_accuracy < 0.4:
                new_bias['sell_threshold_offset'] = min(0.05, new_bias['sell_threshold_offset'] + 0.02)
            elif sell_accuracy < 0.5:
                new_bias['sell_threshold_offset'] = min(0.03, new_bias['sell_threshold_offset'] + 0.01)
            elif sell_accuracy > 0.7:
                new_bias['sell_threshold_offset'] = max(-0.03, new_bias['sell_threshold_offset'] - 0.01)
        
        # 🆕 업데이트된 시장 상황 정보 포함
        if updated_markets:
            new_bias['_updated_markets'] = updated_markets
        
        return new_bias

