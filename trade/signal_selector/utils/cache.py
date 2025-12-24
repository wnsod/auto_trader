"""
캐시 시스템 모듈 - 최적화된 LRU 캐시
"""

import time
from typing import Any, Optional, Dict
from collections import OrderedDict
import threading


class OptimizedCache:
    """
    🚀 최적화된 LRU 캐시 시스템
    
    Thread-safe LRU 캐시로 최근 사용된 항목을 유지하고 오래된 항목을 자동으로 제거합니다.
    """
    def __init__(self, max_size: int = 10000):
        self.cache = OrderedDict()
        self.timestamps = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, max_age: int = 300) -> Optional[Any]:
        """
        캐시에서 값 조회
        
        Args:
            key: 캐시 키
            max_age: 최대 유효 시간 (초)
        
        Returns:
            캐시된 값 또는 None
        """
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < max_age:
                    # LRU 업데이트
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return self.cache[key]
                else:
                    # 만료된 캐시 제거
                    del self.cache[key]
                    del self.timestamps[key]
            self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
        """
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def __len__(self) -> int:
        """
        캐시 크기 반환 (len() 함수 지원)

        Returns:
            캐시에 저장된 항목 수
        """
        with self.lock:
            return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 조회

        Returns:
            캐시 통계 딕셔너리 (hits, misses, hit_rate, size)
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }

    def __contains__(self, key: str) -> bool:
        """캐시에 키가 있는지 확인 (in 연산자 지원)"""
        with self.lock:
            return key in self.cache

    def __delitem__(self, key: str) -> None:
        """캐시에서 항목 삭제 (del 연산자 지원)"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]

    def items(self):
        """캐시 항목 반환 (타임스탬프 포함)"""
        with self.lock:
            return [(k, (v, self.timestamps.get(k, 0))) for k, v in self.cache.items()]

    def clear(self) -> None:
        """캐시 전체 삭제"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

