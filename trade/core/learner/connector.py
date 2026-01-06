#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시그널-매매 연결기 - 시그널 생성 원인과 최종 결과를 정확히 매칭
"""

import time
from typing import Dict, Optional
from trade.core.models import SignalInfo

class SignalTradeConnector:
    """시그널-매매 인과관계 연결 도구"""
    
    def __init__(self):
        self.connections = {}  # connection_id: {signal, trade_result}

    def connect_signal_to_trade(self, signal: SignalInfo, trade_record: Dict):
        """시그널과 거래 내역을 연결하여 추적 가능하게 함"""
        try:
            connection_id = f"{signal.coin}_{signal.timestamp}"
            self.connections[connection_id] = {
                'signal': signal,
                'trade_record': trade_record,
                'connected_at': time.time()
            }
            # print(f"🔗 시그널-매매 연결 완료: {signal.coin}")
        except Exception as e:
            print(f"⚠️ 시그널-매매 연결 오류: {e}")

    def get_connection_info(self, coin: str, timestamp: int) -> Optional[Dict]:
        connection_id = f"{coin}_{timestamp}"
        return self.connections.get(connection_id)

    def cleanup_old_connections(self, max_age_hours: int = 48):
        """오래된 연결 데이터 정리"""
        current_time = time.time()
        cutoff = current_time - (max_age_hours * 3600)
        
        expired_ids = [
            conn_id for conn_id, data in self.connections.items()
            if data['connected_at'] < cutoff
        ]
        
        for conn_id in expired_ids:
            del self.connections[conn_id]

