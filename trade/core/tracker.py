"""
성과 추적 및 피드백 엔진 (Core Tracker)
- 가상/실전 매매에서 공통으로 사용하는 성과 기록 로직
- 거래 결과, 컨텍스트, 패턴 학습 성과 기록
"""
import time
from typing import Dict, Optional

class ActionPerformanceTracker:
    """액션별 성과 추적기"""
    def __init__(self):
        self.action_performance = {
            'buy': {'trades': 0, 'wins': 0, 'total_profit': 0.0},
            'sell': {'trades': 0, 'wins': 0, 'total_profit': 0.0},
            'hold': {'trades': 0, 'wins': 0, 'total_profit': 0.0}
        }
    
    def record_action_result(self, action: str, profit: float, success: bool):
        """액션 결과 기록"""
        if action in self.action_performance:
            self.action_performance[action]['trades'] += 1
            self.action_performance[action]['total_profit'] += profit
            if success:
                self.action_performance[action]['wins'] += 1
    
    def get_action_performance(self, action: str) -> dict:
        """액션별 성과 반환"""
        if action not in self.action_performance:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        perf = self.action_performance[action]
        if perf['trades'] == 0:
            return {'success_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}
        
        return {
            'success_rate': perf['wins'] / perf['trades'],
            'avg_profit': perf['total_profit'] / perf['trades'],
            'total_trades': perf['trades']
        }

class ContextRecorder:
    """컨텍스트 기록기 - 거래 당시의 상황 기록"""
    def __init__(self):
        self.trade_contexts = {}
    
    def record_trade_context(self, trade_id: str, context: dict):
        """거래 컨텍스트 기록"""
        self.trade_contexts[trade_id] = {
            'timestamp': time.time(),
            'context': context
        }
    
    def get_trade_context(self, trade_id: str) -> dict:
        """거래 컨텍스트 조회"""
        return self.trade_contexts.get(trade_id, {})

class LearningFeedback:
    """학습 피드백 시스템 - 거래 결과 및 패턴 학습"""
    def __init__(self):
        self.trade_feedback = {}
        self.pattern_performance = {}
        self.coin_performance = {} # 코인별 성과 추가
        
    def record_trade_result(self, coin: str, trade_result: dict):
        """거래 결과 기록"""
        try:
            # 거래 결과 저장
            timestamp = trade_result.get('entry_timestamp') or trade_result.get('timestamp') or int(time.time())
            trade_id = f"{coin}_{timestamp}"
            self.trade_feedback[trade_id] = trade_result
            
            # 패턴 성과 업데이트
            signal_pattern = trade_result.get('signal_pattern', 'unknown')
            profit = trade_result.get('profit_loss_pct', 0.0)
            
            self._update_pattern_stats(signal_pattern, profit)
            self._update_coin_stats(coin, profit)
            
        except Exception as e:
            print(f"⚠️ 거래 결과 기록 오류: {e}")

    def _update_pattern_stats(self, pattern: str, profit: float):
        if pattern not in self.pattern_performance:
            self.pattern_performance[pattern] = {
                'total_trades': 0,
                'successful_trades': 0,
                'total_profit': 0.0
            }
        
        stats = self.pattern_performance[pattern]
        stats['total_trades'] += 1
        stats['total_profit'] += profit
        if profit > 0:
            stats['successful_trades'] += 1

    def _update_coin_stats(self, coin: str, profit: float):
        if coin not in self.coin_performance:
            self.coin_performance[coin] = {
                'total_trades': 0,
                'successful_trades': 0,
                'total_profit': 0.0
            }
        
        stats = self.coin_performance[coin]
        stats['total_trades'] += 1
        stats['total_profit'] += profit
        if profit > 0:
            stats['successful_trades'] += 1

    def get_coin_learning_data(self, coin: str) -> dict:
        """코인별 학습 데이터 반환"""
        if coin not in self.coin_performance:
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0}
            
        stats = self.coin_performance[coin]
        if stats['total_trades'] == 0:
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0}
            
        return {
            'success_rate': stats['successful_trades'] / stats['total_trades'],
            'avg_profit': stats['total_profit'] / stats['total_trades'],
            'total_trades': stats['total_trades']
        }

