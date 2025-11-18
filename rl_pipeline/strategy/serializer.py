"""
Strategy Serializer 모듈 - 전략 객체 직렬화를 위한 함수들
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

def serialize_strategy(strategy) -> Dict[str, Any]:
    """
    전략 객체를 딕셔너리로 직렬화하는 함수
    
    Args:
        strategy: 직렬화할 전략 객체
        
    Returns:
        직렬화된 전략 딕셔너리
    """
    logger.debug(f"📊 전략 직렬화: {getattr(strategy, 'name', 'Unknown')}")
    
    try:
        # dataclass인 경우 asdict 사용
        if hasattr(strategy, '__dataclass_fields__'):
            data = asdict(strategy)
        else:
            # 일반 객체인 경우 속성을 직접 추출
            base_params = getattr(strategy, 'params', {})
            
            # params에 추가해야 할 개별 파라미터들 추출
            individual_params = {
                'rsi_min': getattr(strategy, 'rsi_min', None),
                'rsi_max': getattr(strategy, 'rsi_max', None),
                'volume_ratio_min': getattr(strategy, 'volume_ratio_min', None),
                'volume_ratio_max': getattr(strategy, 'volume_ratio_max', None),
                'macd_buy_threshold': getattr(strategy, 'macd_buy_threshold', None),
                'macd_sell_threshold': getattr(strategy, 'macd_sell_threshold', None),
                'stop_loss_pct': getattr(strategy, 'stop_loss_pct', None),
                'take_profit_pct': getattr(strategy, 'take_profit_pct', None),
                'ma_period': getattr(strategy, 'ma_period', None),
                'bb_period': getattr(strategy, 'bb_period', None),
                'bb_std': getattr(strategy, 'bb_std', None),
                # 🆕 증분 학습 메타데이터
                'similarity_classification': getattr(strategy, 'similarity_classification', None),
                'similarity_score': getattr(strategy, 'similarity_score', None),
                'parent_strategy_id': getattr(strategy, 'parent_strategy_id', None),
            }
            
            # None이 아닌 값만 params에 추가
            for key, value in individual_params.items():
                if value is not None:
                    base_params[key] = value
            
            data = {
                'id': getattr(strategy, 'id', None),
                'coin': getattr(strategy, 'coin', ''),
                'interval': getattr(strategy, 'interval', ''),
                'strategy_type': getattr(strategy, 'strategy_type', ''),
                'params': base_params,
                'name': getattr(strategy, 'name', ''),
                'description': getattr(strategy, 'description', ''),
                'created_at': getattr(strategy, 'created_at', None),
                'updated_at': getattr(strategy, 'updated_at', None),
                # 개별 파라미터들도 top-level에 추가 (DB 저장을 위해)
                'rsi_min': getattr(strategy, 'rsi_min', None),
                'rsi_max': getattr(strategy, 'rsi_max', None),
                'volume_ratio_min': getattr(strategy, 'volume_ratio_min', None),
                'volume_ratio_max': getattr(strategy, 'volume_ratio_max', None),
                'macd_buy_threshold': getattr(strategy, 'macd_buy_threshold', None),
                'macd_sell_threshold': getattr(strategy, 'macd_sell_threshold', None),
                'stop_loss_pct': getattr(strategy, 'stop_loss_pct', None),
                'take_profit_pct': getattr(strategy, 'take_profit_pct', None),
                'ma_period': getattr(strategy, 'ma_period', None),
                'bb_period': getattr(strategy, 'bb_period', None),
                'bb_std': getattr(strategy, 'bb_std', None),
                # 🆕 증분 학습 메타데이터
                'similarity_classification': getattr(strategy, 'similarity_classification', None),
                'similarity_score': getattr(strategy, 'similarity_score', None),
                'parent_strategy_id': getattr(strategy, 'parent_strategy_id', None),
            }
        
        # None 값 제거
        data = {k: v for k, v in data.items() if v is not None}
        
        return data
        
    except Exception as e:
        logger.error(f"❌ 전략 직렬화 실패: {e}")
        return {
            'id': None,
            'coin': '',
            'interval': '',
            'strategy_type': '',
            'params': {},
            'name': '',
            'description': '',
        }

def deserialize_strategy(data: Dict[str, Any]) -> Optional[object]:
    """
    딕셔너리를 전략 객체로 역직렬화하는 함수 (더미 구현)
    
    Args:
        data: 역직렬화할 딕셔너리
        
    Returns:
        역직렬화된 전략 객체 또는 None
    """
    logger.debug(f"📊 전략 역직렬화: {data.get('name', 'Unknown')}")
    
    try:
        # 더미 구현 - 실제로는 전략 클래스로 변환
        from .factory import Strategy
        
        return Strategy(
            coin=data.get('coin', ''),
            interval=data.get('interval', ''),
            strategy_type=data.get('strategy_type', ''),
            params=data.get('params', {}),
            name=data.get('name', ''),
            description=data.get('description', ''),
        )
        
    except Exception as e:
        logger.error(f"❌ 전략 역직렬화 실패: {e}")
        return None

