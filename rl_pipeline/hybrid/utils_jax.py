"""
JAX 유틸리티 함수
RNG, device, checkpoint 관련 유틸
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# JAX 가용성 확인
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


def get_rng_key(seed: Optional[int] = None) -> 'jax.random.PRNGKey':
    """
    JAX 랜덤 키 생성
    
    Args:
        seed: 시드 값 (None이면 현재 시간 사용)
    
    Returns:
        JAX PRNGKey
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX가 설치되지 않았습니다.")
    
    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**31)
    
    return jax.random.PRNGKey(seed)


def get_device_info() -> dict:
    """
    JAX 디바이스 정보 조회
    
    Returns:
        디바이스 정보 딕셔너리
    """
    if not JAX_AVAILABLE:
        return {'available': False, 'devices': []}
    
    try:
        devices = jax.devices()
        device_info = {
            'available': True,
            'devices': [str(d) for d in devices],
            'default_device': str(devices[0]),
            'device_count': len(devices)
        }
        
        # GPU 여부 확인
        device_info['has_gpu'] = any('gpu' in str(d).lower() for d in devices)
        device_info['has_tpu'] = any('tpu' in str(d).lower() for d in devices)
        
        return device_info
        
    except Exception as e:
        logger.warning(f"⚠️ 디바이스 정보 조회 실패: {e}")
        return {'available': False, 'error': str(e)}


def ensure_checkpoint_dir(path: str) -> str:
    """
    체크포인트 디렉토리 확인 및 생성
    
    Args:
        path: 체크포인트 경로
    
    Returns:
        정규화된 경로
    """
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    return path


def save_checkpoint_metadata(model_id: str, metadata: dict, checkpoint_dir: str):
    """
    체크포인트 메타데이터 저장
    
    Args:
        model_id: 모델 ID
        metadata: 메타데이터 딕셔너리
        checkpoint_dir: 체크포인트 디렉토리
    """
    import json
    
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        metadata_path = os.path.join(checkpoint_dir, f"{model_id}.meta.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"✅ 메타데이터 저장 완료: {metadata_path}")
        
    except Exception as e:
        logger.warning(f"⚠️ 메타데이터 저장 실패: {e}")

