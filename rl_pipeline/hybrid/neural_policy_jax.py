"""
JAX 기반 신경망 정책 네트워크
PPO용 정책 및 가치 네트워크
"""

import logging
import os
import pickle
import struct
import warnings
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# TensorFlow Protobuf 경고 숨김 (JAX 로드 시 발생하는 경고, 기능 영향 없음)
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Sharding info not provided.*', category=UserWarning)

# JAX 가용성 확인 및 CUDA 백엔드 초기화
try:
    import logging as std_logging
    
    # 🔥 JAX 로드 전에 환경 변수 설정 (CUDA 우선, 실패 시 CPU 자동 전환)
    # JAX_PLATFORMS을 빈 문자열로 설정하면 자동으로 사용 가능한 백엔드를 선택
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = ''  # 자동 백엔드 선택 (CUDA가 있으면 CUDA, 없으면 CPU)
    
    # JAX TPU 백엔드 경고 로거 레벨 조정 (경고 숨김)
    jax_logger = std_logging.getLogger('jax._src.xla_bridge')
    jax_logger.setLevel(std_logging.CRITICAL)  # TPU 관련 메시지 숨김
    
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import checkpoints
    
    # 🔥 CUDA 백엔드 사용 가능 여부 확인
    try:
        # JAX 플랫폼 초기화 시도 (조용히)
        devices = jax.devices()
        available_backends = jax.devices()[0].platform if devices else 'unknown'
        
        # CUDA 사용 가능 여부 확인
        cuda_available = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        
        if cuda_available:
            logger.info(f"✅ JAX CUDA 백엔드 사용 가능: {devices}")
            # CUDA 사용 강제 (가능한 경우)
            try:
                jax.config.update('jax_platform_name', 'cuda')
                logger.info("✅ JAX CUDA 플랫폼 설정 완료")
            except Exception as config_err:
                logger.debug(f"⚠️ JAX CUDA 플랫폼 강제 설정 실패, 자동 선택 사용: {config_err}")
        else:
            logger.info(f"💻 JAX CPU 백엔드 사용: {devices} (CUDA 사용 불가)")
            # CPU로 명시적 설정
            try:
                jax.config.update('jax_platform_name', 'cpu')
            except:
                pass
    except Exception as device_check_err:
        logger.debug(f"💻 JAX 디바이스 확인 실패, CPU 모드로 진행: {device_check_err}")
        try:
            jax.config.update('jax_platform_name', 'cpu')
        except:
            pass
    
    # 🔧 Orbax 체크포인트 로거 레벨 조정 (과도한 INFO 메시지 숨김)
    orbax_loggers = [
        'orbax.checkpoint',
        'orbax.checkpoint.checkpoint_handler',
        'orbax.checkpoint.checkpoints',
        'jax.checkpoint',  # JAX checkpoint 모듈
        'flax.training.checkpoints'  # Flax checkpoints
    ]
    for logger_name in orbax_loggers:
        orbax_logger = std_logging.getLogger(logger_name)
        orbax_logger.setLevel(std_logging.WARNING)  # WARNING 이상만 표시 (INFO 숨김)
    
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    nn = None
    logger.warning("⚠️ JAX/Flax가 설치되지 않았습니다. pip install jax[cuda12] 또는 jax[cuda11] (RTX 5090용)")
except Exception as e:
    # JAX 초기화 중 다른 에러 발생 시 CPU 모드로 폴백
    logger.warning(f"⚠️ JAX 초기화 중 에러 발생: {e}, CPU 모드로 진행")
    try:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from flax.training import checkpoints
        jax.config.update('jax_platform_name', 'cpu')
        JAX_AVAILABLE = True
        logger.info("💻 JAX CPU 모드로 초기화 완료")
    except:
        JAX_AVAILABLE = False
        jax = None
        jnp = None
        nn = None


if JAX_AVAILABLE:
    class PolicyNetwork(nn.Module):
        """정책 네트워크 (PPO용)"""
        
        hidden_dim: int = 128
        action_dim: int = 3  # HOLD, BUY, SELL
        
        @nn.compact
        def __call__(self, x):
            """
            순전파

            Args:
                x: 상태 벡터 (batch_size, obs_dim)

            Returns:
                action_logits: (batch_size, action_dim) - 방향 예측 (UP/DOWN/NEUTRAL)
                value: (batch_size, 1) - 상태 가치
                price_change: (batch_size, 1) - 변동률 예측 (%)
                horizon: (batch_size, 1) - 타이밍 예측 (캔들 수)
            """
            # 🔥 학습 성능 개선: 초기화 개선 (Xavier/Glorot 초기화)
            # kernel_init: Xavier uniform 초기화 (더 안정적인 학습)
            kernel_init = nn.initializers.xavier_uniform()
            bias_init = nn.initializers.zeros_init()

            # 공유 레이어
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=kernel_init,
                bias_init=bias_init
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=kernel_init,
                bias_init=bias_init
            )(x)
            x = nn.relu(x)

            # 분기: 4개의 헤드
            # 🔥 Action head: 방향 분류 (UP/DOWN/NEUTRAL)
            action_kernel_init = nn.initializers.xavier_uniform()
            action_logits = nn.Dense(
                self.action_dim,
                name='action_head',
                kernel_init=action_kernel_init,
                bias_init=bias_init
            )(x)

            # Value head: 상태 가치
            value = nn.Dense(
                1,
                name='value_head',
                kernel_init=kernel_init,
                bias_init=bias_init
            )(x)

            # 🆕 Price change head: 변동률 예측 (회귀)
            # 범위: -10% ~ +10% 정도 예상
            price_change = nn.Dense(
                1,
                name='price_change_head',
                kernel_init=kernel_init,
                bias_init=bias_init
            )(x)
            # tanh로 범위 제한 후 스케일링: -0.1 ~ +0.1 (±10%)
            price_change = jnp.tanh(price_change) * 0.1

            # 🆕 Horizon head: 타이밍 예측 (회귀)
            # 범위: 1 ~ 20 캔들 정도 예상
            horizon = nn.Dense(
                1,
                name='horizon_head',
                kernel_init=kernel_init,
                bias_init=bias_init
            )(x)
            # sigmoid로 0~1 범위로 만든 후 1~20으로 스케일링
            horizon = nn.sigmoid(horizon) * 19 + 1

            # 🆕 Analysis head: 분석 점수 예측 (회귀)
            # 범위: 0 ~ 100 (분석 점수는 0~100 사이)
            analysis_score = nn.Dense(
                1,
                name='analysis_head',
                kernel_init=kernel_init,
                bias_init=bias_init
            )(x)
            # sigmoid로 0~1 범위로 만든 후 100으로 스케일링
            analysis_score = nn.sigmoid(analysis_score) * 100.0

            return action_logits, value, price_change, horizon, analysis_score
else:
    # JAX 없을 때 폴백 클래스
    class PolicyNetwork:
        """폴백 정책 네트워크 (JAX 미설치 시 - 규칙 기반으로 폴백)"""
        pass


def init_model(
    rng_key,
    obs_dim: int = 25,  # 🔥 확장 지표 포함 기본값 (20 → 25)
    action_dim: int = 3,
    hidden_dim: int = 128
) -> Dict:
    """
    모델 초기화
    
    Args:
        rng_key: JAX 랜덤 키
        obs_dim: 관측 차원 (기본 25: 확장 지표 포함)
        action_dim: 액션 차원 (기본 3: HOLD/BUY/SELL)
        hidden_dim: 은닉층 차원 (기본 128)
    
    Returns:
        {
            'params': Flax params,
            'model_def': PolicyNetwork,
            'obs_dim': int,
            'action_dim': int,
            'hidden_dim': int
        }
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX가 설치되지 않았습니다. pip install jax flax")
    
    model = PolicyNetwork(hidden_dim=hidden_dim, action_dim=action_dim)
    
    # 모델 초기화용 샘플 입력 생성
    sample_input = jnp.ones((1, obs_dim))
    params = model.init(rng_key, sample_input)
    
    logger.info(f"✅ 정책 네트워크 초기화 완료: obs_dim={obs_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
    
    return {
        'params': params,
        'model_def': model,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim
    }


def apply(
    params: Dict,
    state_vec: np.ndarray,
    rng_key,
    deterministic: bool = False
) -> Dict:
    """
    순전파: state_vector → action_logits, value
    
    Args:
        params: 모델 파라미터 딕셔너리
        state_vec: 상태 벡터 (obs_dim,) 또는 (batch_size, obs_dim)
        rng_key: JAX 랜덤 키
        deterministic: True면 최대 확률 액션, False면 샘플링
    
    Returns:
        {
            'action_logits': np.ndarray,  # (action_dim,)
            'value': float,
            'action_probs': np.ndarray,  # (action_dim,)
            'action': int,  # 0: HOLD, 1: BUY, 2: SELL
            'action_name': str,
            'confidence': float  # 최대 확률
        }
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX가 설치되지 않았습니다.")
    
    model = params['model_def']
    
    # 입력 형태 확인 및 변환
    state_vec = np.asarray(state_vec, dtype=np.float32)
    if state_vec.ndim == 1:
        state_vec = state_vec.reshape(1, -1)
    
    state_vec_jax = jnp.array(state_vec)
    
    # 순전파
    action_logits, value, price_change, horizon, analysis_score = model.apply(params['params'], state_vec_jax)

    # 🔥 액션 샘플링 개선: Temperature 기반 탐험 강화
    # deterministic=False일 때 temperature를 적용하여 탐험 증가
    temperature = 1.5 if not deterministic else 1.0  # 탐험 모드에서는 1.5배 온도 적용

    # Temperature-scaled logits (높은 온도 = 더 균등한 분포 = 더 많은 탐험)
    scaled_logits = action_logits[0] / temperature

    # Softmax로 확률 계산
    action_probs = jax.nn.softmax(scaled_logits)

    # 액션 결정
    if deterministic:
        # 최대 확률 액션 선택
        action_idx = int(jnp.argmax(action_probs))
    else:
        # Temperature-scaled 샘플링 (더 많은 탐험)
        action_idx = int(jax.random.categorical(rng_key, scaled_logits))

    # Action 이름 매핑
    action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    action_name = action_map.get(action_idx, 'HOLD')

    return {
        'action_logits': np.array(action_logits[0]),
        'value': round(float(value[0, 0]), 4),  # 소숫점 4자리
        'action_probs': np.array(action_probs),
        'action': int(action_idx),
        'action_name': action_name,
        'confidence': round(float(jnp.max(action_probs)), 2),  # 소숫점 2자리
        'price_change_pct': round(float(price_change[0, 0]), 4),  # 🆕 변동률 예측 (소숫점 4자리)
        'horizon_k': int(jnp.round(horizon[0, 0])),  # 🆕 타이밍 예측 (정수)
        'predicted_analysis_score': round(float(analysis_score[0, 0]), 2)  # 🆕 분석 점수 예측 (소숫점 2자리)
    }


def save_ckpt(params: Dict, path: str) -> None:
    """
    체크포인트 저장
    
    Args:
        params: 모델 파라미터 딕셔너리
        path: 저장 경로
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX가 설치되지 않았습니다.")
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Flax 체크포인트 형식으로 저장
        checkpoint_dir = os.path.dirname(path)
        checkpoint_name = os.path.basename(path).replace('.ckpt', '')
        
        # 🔧 Orbax 로거 레벨 임시 조정 (저장 중 상세 메시지 숨김)
        import logging as std_logging
        orbax_loggers = [
            'orbax.checkpoint',
            'jax.checkpoint',
            'flax.training.checkpoints'
        ]
        original_levels = {}
        for logger_name in orbax_loggers:
            orbax_logger = std_logging.getLogger(logger_name)
            original_levels[logger_name] = orbax_logger.level
            orbax_logger.setLevel(std_logging.WARNING)
        
        try:
            # Flax checkpoints.save() 사용
            checkpoints.save_checkpoint(
                checkpoint_dir,
                target=params['params'],
                step=0,
                prefix=checkpoint_name,
                keep=1
            )
        finally:
            # 원래 로거 레벨 복원
            for logger_name, original_level in original_levels.items():
                std_logging.getLogger(logger_name).setLevel(original_level)
        
        # 추가 메타데이터 저장
        metadata_path = path + '.meta'
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'obs_dim': params['obs_dim'],
                'action_dim': params['action_dim'],
                'hidden_dim': params['hidden_dim'],
                'model_class': 'PolicyNetwork'
            }, f)
        
        logger.info(f"✅ 체크포인트 저장 완료: {path}")
        
    except Exception as e:
        logger.error(f"❌ 체크포인트 저장 실패: {e}")
        raise


def load_ckpt(path: str, rng_key=None) -> Dict:
    """
    체크포인트 로드
    
    Args:
        path: 체크포인트 경로
        rng_key: JAX 랜덤 키 (필요 시)
    
    Returns:
        모델 파라미터 딕셔너리
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX가 설치되지 않았습니다.")
    
    # 체크포인트 파라미터와 메타데이터를 분리하여 안전하게 로드
    restored_params = None
    metadata = {
        'obs_dim': 15,
        'action_dim': 3,
        'hidden_dim': 128
    }
    
    try:
        checkpoint_dir = os.path.dirname(path)
        checkpoint_name = os.path.basename(path).replace('.ckpt', '')
        
        # 🔧 Orbax 형식 체크포인트 디렉토리 확인
        # Orbax는 `{prefix}0` 디렉토리 형식으로 저장됨 (예: ppo_20251031_154621_0c18d72b0)
        orbax_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name + '0')
        
        # 실제 체크포인트 디렉토리 찾기
        actual_checkpoint_path = None
        if os.path.exists(orbax_checkpoint_path) and os.path.isdir(orbax_checkpoint_path):
            actual_checkpoint_path = orbax_checkpoint_path
            logger.debug(f"✅ Orbax 체크포인트 디렉토리 발견: {actual_checkpoint_path}")
        else:
            # prefix로 검색 시도 (디렉토리 목록에서 찾기)
            logger.debug(f"🔍 체크포인트 디렉토리 검색: {checkpoint_dir}, prefix: {checkpoint_name}")
            try:
                for item in os.listdir(checkpoint_dir):
                    item_path = os.path.join(checkpoint_dir, item)
                    # {prefix}0 형식의 디렉토리 찾기 (.meta 파일 제외)
                    if (item.startswith(checkpoint_name) and 
                        item.endswith('0') and 
                        os.path.isdir(item_path) and
                        '.meta' not in item):
                        actual_checkpoint_path = item_path
                        logger.debug(f"✅ 체크포인트 발견: {actual_checkpoint_path}")
                        break
            except OSError as list_err:
                logger.warning(f"⚠️ 체크포인트 디렉토리 목록 조회 실패: {list_err}")
        
        if actual_checkpoint_path is None:
            raise FileNotFoundError(f"체크포인트 디렉토리를 찾을 수 없습니다: {checkpoint_dir}, prefix: {checkpoint_name}, 시도한 경로: {orbax_checkpoint_path}")
        
        # 🔧 Orbax 로거 레벨 임시 조정 (로드 중 상세 메시지 숨김)
        import logging as std_logging
        orbax_loggers = [
            'orbax.checkpoint',
            'jax.checkpoint',
            'flax.training.checkpoints'
        ]
        original_levels = {}
        for logger_name in orbax_loggers:
            orbax_logger = std_logging.getLogger(logger_name)
            original_levels[logger_name] = orbax_logger.level
            orbax_logger.setLevel(std_logging.WARNING)
        
        try:
            # 🔥 개선: Orbax 체크포인트 디렉토리가 확인되면 prefix 방식을 건너뛰고 바로 직접 로드
            # Legacy checkpoint 형식 오류 방지 (unpack(b) received extra data)
            if actual_checkpoint_path and os.path.isdir(actual_checkpoint_path):
                # Orbax 체크포인트 디렉토리를 직접 로드 (prefix 방식으로 인한 legacy 형식 시도 방지)
                logger.info(f"Restoring orbax checkpoint from {actual_checkpoint_path}")
                try:
                    restored_params = checkpoints.restore_checkpoint(
                        actual_checkpoint_path,
                        target=None
                    )
                except (struct.error, pickle.UnpicklingError, EOFError, ValueError) as restore_err:
                    # 체크포인트 파일 손상 또는 형식 불일치 시 다른 방법 시도
                    error_msg = str(restore_err)
                    if "unpack" in error_msg.lower() or "extra data" in error_msg.lower():
                        logger.warning(f"⚠️ 체크포인트 파일 손상 감지 (다른 방법 시도): {error_msg}")
                    else:
                        logger.debug(f"⚠️ Orbax 체크포인트 직접 로드 실패 (다른 방법 시도): {restore_err}")
                    restored_params = None
            else:
                # Orbax 체크포인트 디렉토리를 찾지 못한 경우에만 prefix 방식 시도
                logger.debug(f"🔍 Orbax 체크포인트 디렉토리 없음, prefix 방식 시도: {checkpoint_name}")
                try:
                    restored_params = checkpoints.restore_checkpoint(
                        checkpoint_dir,
                        target=None,
                        prefix=checkpoint_name,
                        step=None  # None이면 최신 체크포인트
                    )
                except (struct.error, pickle.UnpicklingError, EOFError, ValueError) as restore_err:
                    # 체크포인트 파일 손상 또는 형식 불일치 시 다른 방법 시도
                    error_msg = str(restore_err)
                    if "unpack" in error_msg.lower() or "extra data" in error_msg.lower():
                        logger.warning(f"⚠️ 체크포인트 파일 손상 감지 (다른 방법 시도): {error_msg}")
                    else:
                        logger.debug(f"⚠️ prefix 방식 실패 (다른 방법 시도): {restore_err}")
                    restored_params = None
            
            if restored_params is None:
                # 체크포인트를 찾지 못한 경우, 디렉토리 검색 재시도
                logger.debug(f"⚠️ 체크포인트 로드 실패, 디렉토리 검색 재시도: {checkpoint_dir}")
                # 디렉토리 내부의 체크포인트 파일 찾기
                if actual_checkpoint_path and os.path.isdir(actual_checkpoint_path):
                    # 체크포인트 디렉토리 내부에서 최신 체크포인트 찾기
                    checkpoint_files = []
                    try:
                        for item in os.listdir(actual_checkpoint_path):
                            if item.startswith('checkpoint_') or item == 'checkpoint':
                                item_path = os.path.join(actual_checkpoint_path, item)
                                if os.path.isfile(item_path) or os.path.isdir(item_path):
                                    checkpoint_files.append(item_path)
                    except OSError as list_err:
                        logger.debug(f"⚠️ 체크포인트 디렉토리 목록 조회 실패: {list_err}")
                    
                    if checkpoint_files:
                        # 최신 파일 선택
                        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                        logger.debug(f"🔍 체크포인트 파일 발견: {latest_checkpoint}")
                        # Flax는 디렉토리 단위로 로드하므로, 디렉토리 경로 사용
                        try:
                            restored_params = checkpoints.restore_checkpoint(
                                actual_checkpoint_path,
                                target=None
                            )
                        except (struct.error, pickle.UnpicklingError, EOFError, ValueError) as restore_err2:
                            logger.warning(f"⚠️ 체크포인트 디렉토리 로드 실패 (파일 손상 가능): {restore_err2}")
                            restored_params = None
                    else:
                        # 체크포인트 디렉토리 자체를 로드
                        try:
                            restored_params = checkpoints.restore_checkpoint(
                                actual_checkpoint_path,
                                target=None
                            )
                        except (struct.error, pickle.UnpicklingError, EOFError, ValueError) as restore_err3:
                            logger.warning(f"⚠️ 체크포인트 직접 로드 실패 (파일 손상 가능): {restore_err3}")
                            restored_params = None
            
            if restored_params is not None:
                logger.info(f"✅ 체크포인트 로드 성공: {actual_checkpoint_path or checkpoint_dir}")
            else:
                raise FileNotFoundError(f"체크포인트를 찾을 수 없거나 손상됨: {actual_checkpoint_path or checkpoint_dir}")
        finally:
            # 원래 로거 레벨 복원
            for logger_name, original_level in original_levels.items():
                std_logging.getLogger(logger_name).setLevel(original_level)
        
        if restored_params is None:
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")
        
    except FileNotFoundError:
        # 체크포인트 파일이 없는 경우는 명확한 에러
        raise
    except (struct.error, pickle.UnpicklingError, EOFError, ValueError) as unpickle_err:
        # 체크포인트 파일 손상 또는 형식 불일치
        error_msg = str(unpickle_err)
        logger.error(f"❌ 체크포인트 파라미터 로드 실패 (파일 손상 또는 형식 불일치): {error_msg}")
        logger.warning(f"⚠️ 체크포인트 로드 실패로 새 모델을 초기화합니다")
        raise
    except Exception as e:
        # 체크포인트 파라미터 로드 실패 (기타 에러)
        error_msg = str(e)
        logger.error(f"❌ 체크포인트 파라미터 로드 실패: {error_msg}")
        raise
    
    # 메타데이터 로드 (에러 발생 시 기본값 사용, 전체 로드 실패하지 않음)
    metadata_path = path + '.meta'
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                # 🔧 pickle 로드 시 에러 처리 개선 (unpack 에러 방지)
                try:
                    loaded_metadata = pickle.load(f)
                except (pickle.UnpicklingError, EOFError, ValueError, struct.error) as unpickle_err:
                    # pickle 로드 실패 시 파일이 손상되었거나 형식이 다를 수 있음
                    # struct.error는 "unpack(b) received extra data" 포함
                    logger.debug(f"⚠️ 메타데이터 pickle 로드 실패 (기본값 사용): {unpickle_err}")
                    loaded_metadata = None
                
                if loaded_metadata is not None:
                    # 검증: 딕셔너리이고 필수 키가 있는지 확인
                    if isinstance(loaded_metadata, dict):
                        if 'obs_dim' in loaded_metadata and 'action_dim' in loaded_metadata and 'hidden_dim' in loaded_metadata:
                            metadata = loaded_metadata
                            logger.debug(f"✅ 메타데이터 로드 성공: {metadata_path}")
                        else:
                            logger.warning(f"⚠️ 메타데이터 형식 불일치, 기본값 사용: {metadata_path}")
                    else:
                        logger.warning(f"⚠️ 메타데이터가 딕셔너리가 아님, 기본값 사용: {metadata_path}")
                else:
                    logger.warning(f"⚠️ 메타데이터 로드 결과 없음, 기본값 사용: {metadata_path}")
        except Exception as meta_err:
            # 메타데이터 로드 실패해도 기본값으로 계속 진행 (치명적이지 않음)
            error_msg = str(meta_err)
            if "unpack" in error_msg.lower() or "extra data" in error_msg.lower():
                logger.warning(f"⚠️ 체크포인트 메타데이터 문제 (기본값으로 계속 진행): {error_msg}")
            else:
                logger.warning(f"⚠️ 메타데이터 로드 실패 (기본값 사용): {meta_err}")
    else:
        logger.debug(f"ℹ️ 메타데이터 파일 없음, 기본값 사용: {metadata_path}")
    
    # 모델 재구성
    try:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        model = PolicyNetwork(
            hidden_dim=metadata['hidden_dim'],
            action_dim=metadata['action_dim']
        )
        
        logger.info(f"✅ 체크포인트 로드 완료: {path}")
        
        return {
            'params': restored_params,
            'model_def': model,
            'obs_dim': metadata['obs_dim'],
            'action_dim': metadata['action_dim'],
            'hidden_dim': metadata['hidden_dim']
        }
    except Exception as e:
        # 모델 재구성 실패
        logger.error(f"❌ 모델 재구성 실패: {e}")
        import traceback
        logger.debug(f"모델 재구성 상세 에러:\n{traceback.format_exc()}")
        raise

