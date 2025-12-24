"""
Contextual Bandit 클래스
"""
import os
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
# signal_selector imports
try:
    from signal_selector.core.types import SignalInfo, SignalAction
except ImportError:
    import sys
    _current = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_current)
    sys.path.insert(0, _parent)
    from core.types import SignalInfo, SignalAction


class ContextualBandit:
    """컨텍스추얼 밴딧 시스템 (UCB/Thompson Sampling)"""
    def __init__(self, exploration_factor: float = 1.0):
        self.exploration_factor = exploration_factor
        self.action_counts = {}
        self.action_rewards = {}
        self.total_trials = 0
        
    def select_action(self, context: str, available_actions: List[str]) -> str:
        """UCB 기반 액션 선택"""
        try:
            if not available_actions:
                return 'hold'
            
            # 초기화
            for action in available_actions:
                if action not in self.action_counts:
                    self.action_counts[action] = 0
                    self.action_rewards[action] = 0.0
            
            # UCB 점수 계산
            ucb_scores = {}
            for action in available_actions:
                if self.action_counts[action] == 0:
                    ucb_scores[action] = float('inf')  # 탐색 우선
                else:
                    avg_reward = self.action_rewards[action] / self.action_counts[action]
                    exploration_bonus = self.exploration_factor * math.sqrt(
                        math.log(self.total_trials) / self.action_counts[action]
                    )
                    ucb_scores[action] = avg_reward + exploration_bonus
            
            # 최고 UCB 점수 액션 선택
            best_action = max(ucb_scores.items(), key=lambda x: x[1])[0]
            return best_action
            
        except Exception as e:
            print(f"⚠️ 컨텍스추얼 밴딧 액션 선택 오류: {e}")
            return 'hold'
    
    def update_reward(self, action: str, reward: float):
        """액션 보상 업데이트"""
        try:
            if action not in self.action_counts:
                self.action_counts[action] = 0
                self.action_rewards[action] = 0.0
            
            self.action_counts[action] += 1
            self.action_rewards[action] += reward
            self.total_trials += 1
            
        except Exception as e:
            print(f"⚠️ 컨텍스추얼 밴딧 보상 업데이트 오류: {e}")



