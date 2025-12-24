"""
시그널-거래 연결 클래스
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


class SignalTradeConnector:
    """시그널-매매 연결 시스템"""
    def __init__(self):
        self.connections = {}
        self.pending_signals = {}
        
    def connect_signal_to_trade(self, signal: SignalInfo, trade_result: dict):
        """시그널과 매매 결과 연결"""
        try:
            connection_id = f"{signal.coin}_{signal.timestamp}"
            self.connections[connection_id] = {
                'signal': signal,
                'trade_result': trade_result,
                'connected_at': time.time()
            }
            print(f"🔗 시그널-매매 연결: {signal.coin} 연결 완료")
        except Exception as e:
            print(f"⚠️ 시그널-매매 연결 오류: {e}")



