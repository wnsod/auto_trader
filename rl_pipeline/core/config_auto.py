"""
시스템 리소스를 감지하여 최적 설정 자동 생성
아키텍처 개선: 자동 설정 감지 시스템
"""

import os
from typing import Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AutoConfigDetector:
    """시스템 리소스 자동 감지"""
    
    @staticmethod
    def detect_cpu() -> Dict[str, int]:
        """CPU 정보 감지"""
        cpu_count = os.cpu_count() or 4
        return {
            'cores': cpu_count,
            'recommended_workers': min(cpu_count, 16),
            'recommended_batch_size': 50 if cpu_count < 4 else 100
        }
    
    @staticmethod
    def detect_memory() -> Dict[str, Any]:
        """메모리 정보 감지"""
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                total_gb = mem.total / (1024**3)
                available_gb = mem.available / (1024**3)
                
                # 메모리 크기에 따른 권장 설정
                if total_gb < 4:
                    batch_size = 25
                    max_workers = 2
                elif total_gb < 8:
                    batch_size = 50
                    max_workers = 4
                else:
                    batch_size = 100
                    max_workers = min(8, os.cpu_count() or 4)
                
                return {
                    'total_gb': total_gb,
                    'available_gb': available_gb,
                    'recommended_batch_size': batch_size,
                    'recommended_max_workers': max_workers
                }
            except Exception:
                pass
        
        # psutil 없거나 실패 시 기본값
        return {
            'total_gb': 8.0,
            'available_gb': 4.0,
            'recommended_batch_size': 50,
            'recommended_max_workers': 4
        }
    
    @staticmethod
    def detect_gpu() -> Dict[str, Any]:
        """GPU 감지 (JAX 사용 가능 여부)"""
        try:
            import jax
            devices = jax.devices()
            has_gpu = any('gpu' in str(d).lower() for d in devices)
            return {
                'available': has_gpu,
                'device_count': len(devices),
                'recommend_hybrid': has_gpu
            }
        except ImportError:
            return {
                'available': False,
                'device_count': 0,
                'recommend_hybrid': False
            }
    
    @staticmethod
    def generate_optimal_config() -> Dict[str, Any]:
        """최적 설정 자동 생성"""
        cpu_info = AutoConfigDetector.detect_cpu()
        mem_info = AutoConfigDetector.detect_memory()
        gpu_info = AutoConfigDetector.detect_gpu()
        
        return {
            'MAX_WORKERS': min(
                cpu_info['recommended_workers'],
                mem_info['recommended_max_workers']
            ),
            'BATCH_SIZE': mem_info['recommended_batch_size'],
            'ENABLE_HYBRID': gpu_info['recommend_hybrid'],
            'ENABLE_AUTO_TRAINING': gpu_info['available'],
            'detected_cpu_cores': cpu_info['cores'],
            'detected_memory_gb': mem_info['total_gb'],
            'detected_gpu': gpu_info['available']
        }

