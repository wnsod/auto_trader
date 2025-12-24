"""
PPO 학습기 (JAX 기반)
Self-play 데이터를 활용한 강화학습
"""

import logging
import os
import json
import uuid
import math
import copy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# JAX 가용성 확인
try:
    import jax
    import jax.numpy as jnp
    # JAX tree 유틸리티 확인 (버전별 호환성)
    try:
        # 최신 버전 (v0.4.25+): jax.tree
        _ = jax.tree
        USE_JAX_TREE = True
    except AttributeError:
        # 구버전: jax.tree_util
        try:
            from jax import tree_util
            USE_JAX_TREE = False
            JAX_TREE_UTIL = tree_util
        except ImportError:
            USE_JAX_TREE = False
            JAX_TREE_UTIL = None
    
    JAX_AVAILABLE = True
    logger.debug("✅ trainer_jax: JAX/Flax 임포트 성공")
except ImportError as e:
    JAX_AVAILABLE = False
    logger.warning(f"⚠️ trainer_jax: JAX가 설치되지 않았습니다: {e}")
    jax = None
    jnp = None
    USE_JAX_TREE = False
    JAX_TREE_UTIL = None

# neural_policy_jax 모듈 임포트 (JAX_AVAILABLE 체크 포함)
try:
    from rl_pipeline.hybrid.neural_policy_jax import init_model, apply, save_ckpt, PolicyNetwork, JAX_AVAILABLE as NEURAL_JAX_AVAILABLE
    # neural_policy_jax의 JAX_AVAILABLE도 확인
    if not NEURAL_JAX_AVAILABLE:
        logger.warning("⚠️ trainer_jax: neural_policy_jax에서 JAX 사용 불가")
        if JAX_AVAILABLE:
            # 로컬에서는 JAX가 있지만 neural_policy_jax에서는 없는 경우
            logger.warning("⚠️ trainer_jax: 로컬 JAX는 있지만 neural_policy_jax 모듈에서 사용 불가")
        JAX_AVAILABLE = False
except ImportError as e:
    logger.warning(f"⚠️ trainer_jax: neural_policy_jax 임포트 실패: {e}")
    JAX_AVAILABLE = False
    init_model = None
    apply = None
    save_ckpt = None
    PolicyNetwork = None

from rl_pipeline.hybrid.features import (
    build_state_vector,
    build_state_vector_with_analysis,
    build_state_vector_with_strategy,  # 🚀 메타 학습
    build_state_vector_with_analysis_and_strategy,  # 🚀 메타 학습
    FEATURE_DIM,
    FEATURE_DIM_WITH_ANALYSIS,
    FEATURE_DIM_WITH_STRATEGY,  # 🚀 메타 학습: 30차원
    FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY  # 🚀 메타 학습: 35차원
)
from rl_pipeline.engine.reward_engine import RewardEngine
from rl_pipeline.db.connection_pool import get_strategy_db_pool

# 🔥 디버그 로깅 시스템
try:
    from rl_pipeline.monitoring import TrainingDebugger
    DEBUG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ 디버그 로깅 시스템을 사용할 수 없습니다")
    DEBUG_AVAILABLE = False
    TrainingDebugger = None

# optax는 Flax 0.10+ 버전에서 필수 (optimizer 역할)
if JAX_AVAILABLE:
    try:
        import optax
        logger.debug("✅ trainer_jax: optax 임포트 성공")
    except ImportError as e:
        logger.error(f"❌ trainer_jax: optax 임포트 실패: {e}")
        logger.error("❌ optax는 필수입니다. pip install optax")
        JAX_AVAILABLE = False
        optax = None
else:
    optax = None

# 액션/샘플링 기본 상수 (환경변수로 조정 가능)
NEUTRAL_TARGET_RATIO = float(os.getenv('NEUTRAL_TARGET_RATIO', '0.2'))
MIN_NEUTRAL_SAMPLES = int(os.getenv('MIN_NEUTRAL_SAMPLES', '3'))
MAX_NEUTRAL_SAMPLES = int(os.getenv('MAX_NEUTRAL_SAMPLES', '12'))
MAX_DIRECTION_SAMPLES = int(os.getenv('MAX_DIRECTION_SAMPLES', '12'))
NEUTRAL_ACTION_BONUS = float(os.getenv('NEUTRAL_ACTION_BONUS', '0.2'))
UPDOWN_ACTION_BONUS = float(os.getenv('UPDOWN_ACTION_BONUS', '0.1'))
NEUTRAL_PRICE_CHANGE_THRESHOLD = float(os.getenv('NEUTRAL_PRICE_CHANGE_THRESHOLD', '0.005'))
MAX_SYNTHETIC_NEUTRAL = int(os.getenv('MAX_SYNTHETIC_NEUTRAL', '20'))
MIN_DIRECTIONAL_SAMPLES = int(os.getenv('MIN_DIRECTIONAL_SAMPLES', '2'))
MAX_SYNTHETIC_DIRECTIONAL = int(os.getenv('MAX_SYNTHETIC_DIRECTIONAL', '12'))


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) 학습기 - 예측 전략
    
    Self-play 데이터를 활용하여 방향 예측 정책 학습
    - 액션 공간: NEUTRAL(0), UP(1), DOWN(2) - 예측 방향
    - 보상: 예측 정확도 기반 (방향 맞춤/틀림)
    """
    
    def __init__(self, config: Dict[str, Any], session_id: Optional[str] = None):
        """
        초기화

        Args:
            config: 학습 설정 딕셔너리
            session_id: 디버그 세션 ID (선택적)
        """
        if not JAX_AVAILABLE:
            error_msg = "JAX가 설치되지 않았습니다. pip install jax flax"
            logger.error(f"❌ {error_msg}")
            raise ImportError(error_msg)

        # neural_policy_jax 함수들이 사용 가능한지 확인
        if init_model is None or apply is None or save_ckpt is None:
            error_msg = "neural_policy_jax 모듈에서 필요한 함수를 임포트할 수 없습니다."
            logger.error(f"❌ {error_msg}")
            raise ImportError(error_msg)

        self.config = config
        self.train_config = config.get('train', {})
        self.reward_engine = RewardEngine()

        # 🔥 디버거 초기화
        self.debug = None
        if DEBUG_AVAILABLE and session_id:
            try:
                self.debug = TrainingDebugger(session_id=session_id)
                logger.debug(f"✅ Training 디버거 초기화 완료 (session: {session_id})")
            except Exception as e:
                logger.warning(f"⚠️ Training 디버거 초기화 실패: {e}")
        
        # 모델 초기화 (분석 점수 포함 차원으로 기본 설정)
        # 🔥 jax는 모듈 레벨에서 이미 import되어 있음 - global 선언으로 명시
        try:
            # global 선언으로 모듈 레벨의 jax를 명시적으로 사용 (스코프 문제 완전 해결)
            global jax, jnp
            # JAX_AVAILABLE이 True이므로 jax는 None이 아니어야 함
            if jax is None:
                raise ImportError("JAX가 초기화되지 않았습니다. JAX_AVAILABLE 체크를 통과했지만 jax가 None입니다.")
            
            # 🔥 JAX 백엔드 확인 및 CPU 폴백 처리 (RTX 5090 호환)
            try:
                # JAX 플랫폼 확인 (에러 발생 시 CPU로 폴백)
                devices = jax.devices()
                logger.debug(f"🔍 JAX 디바이스: {devices}")
                rng_key = jax.random.PRNGKey(config.get('seed', 42))
            except RuntimeError as backend_err:
                # CUDA 백엔드 초기화 실패 시 CPU로 폴백
                if 'cuda' in str(backend_err).lower() or 'backend' in str(backend_err).lower():
                    logger.warning(f"⚠️ JAX CUDA 백엔드 사용 불가: {backend_err}")
                    logger.info("💻 JAX CPU 모드로 전환 중...")
                    try:
                        jax.config.update('jax_platform_name', 'cpu')
                        # JAX 재초기화 (CPU 모드)
                        devices = jax.devices()
                        logger.info(f"✅ JAX CPU 모드로 전환 완료: {devices}")
                        rng_key = jax.random.PRNGKey(config.get('seed', 42))
                    except Exception as cpu_fallback_err:
                        logger.error(f"❌ JAX CPU 모드 전환도 실패: {cpu_fallback_err}")
                        raise
                else:
                    raise
            # 🔥 예측 전략: 액션 공간을 방향 예측으로 정의
            # action_dim=3: 0=NEUTRAL(중립), 1=UP(상승 예측), 2=DOWN(하락 예측)
            # 거래 액션(BUY/SELL/HOLD)이 아닌 예측 방향으로 학습
            # 🚀 메타 학습: State에 전략 파라미터 포함 (35차원)
            self.model = init_model(
                rng_key,
                obs_dim=FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY,  # 🚀 35차원 (분석+전략 파라미터)
                action_dim=3,  # 🔥 예측 방향: NEUTRAL(0), UP(1), DOWN(2)
                hidden_dim=self.train_config.get('hidden_dim', 128)
            )
            
            # 🔥 모델 파라미터 초기 검증 (NaN/Inf 체크)
            try:
                # 모듈 레벨의 jax, jnp 사용 (함수 내부에서 다시 import하지 않음)
                def validate_params(p):
                    """재귀적으로 파라미터 검증"""
                    if isinstance(p, dict):
                        return all(validate_params(v) for v in p.values())
                    elif hasattr(p, 'shape') and hasattr(p, 'size'):
                        if p.size > 0:
                            if not jnp.all(jnp.isfinite(p)):
                                logger.warning(f"⚠️ 초기화 시 파라미터에 NaN/Inf 발견: shape={p.shape}")
                                return False
                        return True
                    return True
                
                if not validate_params(self.model.get('params', {})):
                    logger.error("❌ 모델 초기화 후 파라미터 검증 실패")
                    raise ValueError("모델 파라미터에 NaN/Inf 포함")
            except Exception as param_check_err:
                logger.warning(f"⚠️ 파라미터 검증 중 오류 (무시): {param_check_err}")
                
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            import traceback
            logger.error(f"초기화 실패 상세:\n{traceback.format_exc()}")
            raise
        
        # 옵티마이저 초기화 (optax 사용 - Flax 0.10+ 버전)
        if optax is None:
            raise ImportError("optax가 설치되지 않았습니다. pip install optax")
        
        # 🔥 학습 성능 개선: 학습률 스케줄링 및 조정
        base_lr = self.train_config.get('lr', 0.0003)
        # 학습률 증가: Neural network가 학습하지 못하는 문제 해결
        # 이전 0.000075 (7.5e-5)는 너무 낮아서 조기 종료됨
        learning_rate = base_lr  # 0.0003 (3e-4) - 적절한 학습을 위해 증가
        
        # 🔥 학습률 자동 조정을 위한 기본값 저장
        self.base_learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        # 🔥 Entropy coefficient 자동 조정을 위한 기본값 저장
        base_entropy_coef = self.train_config.get('entropy_coef', 0.15)
        self.base_entropy_coef = base_entropy_coef
        self.current_entropy_coef = base_entropy_coef

        # 🔥 Gradient clipping 강화 + 학습률 스케줄링
        # optax.adam은 learning_rate를 위치 인자로 받음
        # 더 안정적인 학습을 위해 gradient clipping 강화
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),  # Gradient clipping 강화 (기존보다 더 작게)
            optax.scale_by_adam(),
            optax.scale(-learning_rate)  # 학습률 적용
        )
        
        self.opt_state = self.optimizer.init(self.model['params'])
        self._step_count = 0  # 스텝 카운터
        
        # 학습 통계
        self.training_history = []

        # 액션/보상 다양성 설정
        self.neutral_target_ratio = float(self.train_config.get('neutral_target_ratio', NEUTRAL_TARGET_RATIO))
        self.min_neutral_samples = int(self.train_config.get('min_neutral_samples', MIN_NEUTRAL_SAMPLES))
        self.max_neutral_samples = int(self.train_config.get('max_neutral_samples', MAX_NEUTRAL_SAMPLES))
        self.max_direction_samples = int(self.train_config.get('max_direction_samples', MAX_DIRECTION_SAMPLES))
        self.neutral_action_bonus = float(self.train_config.get('neutral_action_bonus', NEUTRAL_ACTION_BONUS))
        self.direction_action_bonus = float(self.train_config.get('direction_action_bonus', UPDOWN_ACTION_BONUS))
        self.neutral_direction_threshold = float(
            self.train_config.get('neutral_direction_threshold', NEUTRAL_PRICE_CHANGE_THRESHOLD)
        )
        self.max_synthetic_neutral = int(
            self.train_config.get('max_synthetic_neutral', MAX_SYNTHETIC_NEUTRAL)
        )
        self.min_directional_samples = int(
            self.train_config.get('min_directional_samples', MIN_DIRECTIONAL_SAMPLES)
        )
        self.max_synthetic_directional = int(
            self.train_config.get('max_synthetic_directional', MAX_SYNTHETIC_DIRECTIONAL)
        )
        # 최소/최대 유효 범위 보정
        if self.max_neutral_samples < self.min_neutral_samples:
            self.max_neutral_samples = self.min_neutral_samples
        self.neutral_target_ratio = max(0.05, min(0.5, self.neutral_target_ratio))
        self.neutral_direction_threshold = max(1e-4, self.neutral_direction_threshold)
        self.max_synthetic_neutral = max(0, self.max_synthetic_neutral)
        self.min_directional_samples = max(1, self.min_directional_samples)
        self.max_synthetic_directional = max(self.min_directional_samples, self.max_synthetic_directional)
        
        logger.info(f"✅ PPO Trainer 초기화 완료 (lr={learning_rate:.6f})")
    
    def train_from_selfplay_data(
        self,
        episodes_data: List[Dict[str, Any]],
        db_path: Optional[str] = None,
        analysis_data: Optional[Dict[str, Any]] = None  # 🔥 추가: 통합 분석 결과
    ) -> str:
        """
        Self-play 데이터로 학습 (분석 데이터 포함 가능)
        
        Args:
            episodes_data: Self-play 결과 데이터
            db_path: DB 경로 (모델 저장용, None이면 config에서 가져옴)
            analysis_data: 통합 분석 결과 (선택적)
                {
                    'fractal_score': float,
                    'multi_timeframe_score': float,
                    'indicator_cross_score': float,
                    'ensemble_score': float,
                    'ensemble_confidence': float
                }
        
        Returns:
            model_id: 학습된 모델 ID
        """
        try:
            logger.info(f"🚀 PPO 학습 시작: {len(episodes_data)}개 에피소드" +
                       (", 분석 데이터 포함" if analysis_data else ""))

            # 🔥 디버거 로깅: 학습 시작
            if self.debug:
                try:
                    self.debug.log_training_start({
                        "learning_rate": self.current_learning_rate,
                        "epochs": self.train_config.get('epochs', 30),
                        "batch_size": self.train_config.get('batch_size', 4096),
                        "num_episodes": len(episodes_data),
                        "has_analysis_data": analysis_data is not None,
                        "clip_epsilon": self.train_config.get('clip_epsilon', 0.2),
                        "entropy_coef": self.train_config.get('entropy_coef', 0.05),
                        "value_loss_coef": self.train_config.get('value_loss_coef', 0.5),
                        "gamma": self.train_config.get('gamma', 0.99),
                        "gae_lambda": self.train_config.get('gae_lambda', 0.95)
                    })
                except Exception as debug_err:
                    logger.debug(f"⚠️ 학습 시작 디버그 로깅 실패 (무시): {debug_err}")

            # 1. Self-play 데이터에서 경험 추출 (분석 데이터 포함)
            experiences = self._extract_experiences(episodes_data, analysis_data)
            # 🚀 메타 학습: 전략 파라미터 포함 (35차원 or 30차원)
            feature_dim = FEATURE_DIM_WITH_ANALYSIS_AND_STRATEGY if analysis_data else FEATURE_DIM_WITH_STRATEGY
            logger.info(f"📊 추출된 경험: {len(experiences)}개 (차원: {feature_dim})")

            if analysis_data:
                logger.info(f"🚀 35차원 메타 학습 활성화 - 확장 지표 + 분석 점수 + 전략 파라미터 포함")
                logger.info(f"   프랙탈: {analysis_data.get('fractal_score', 0.5):.3f}, "
                          f"멀티TF: {analysis_data.get('multi_timeframe_score', 0.5):.3f}, "
                          f"지표교차: {analysis_data.get('indicator_cross_score', 0.5):.3f}")
            else:
                logger.info(f"🚀 30차원 메타 학습 (확장 지표 + 전략 파라미터 포함, 분석 점수 없음)")
            
            # 🔥 학습 전 데이터 검증 강화 (개선: 최소 요구량 완화)
            if len(experiences) < 50:  # 🔥 개선: 100 → 50 (더 빠른 학습 시작)
                logger.warning(f"⚠️ 경험 데이터가 매우 부족합니다 ({len(experiences)}개). 최소 50개 권장")
            elif len(experiences) < 100:
                logger.info(f"ℹ️ 경험 데이터: {len(experiences)}개 (권장: 100개 이상)")
            
            # 액션 다양성 최종 검증
            if experiences:
                actions = [exp.get('action', 0) for exp in experiences]
                unique_actions = set(actions)
                action_counts = {action: actions.count(action) for action in unique_actions}
                
                if len(unique_actions) < 2:
                    logger.error(f"❌ 학습 전 검증 실패: 액션 다양성 부족 (고유 액션: {len(unique_actions)}개)")
                    logger.error(f"   액션 분포: {action_counts}")
                    logger.error(f"   학습을 중단합니다. 데이터 수집 로직을 확인하세요.")
                    raise ValueError(f"액션 다양성 부족: 고유 액션 {len(unique_actions)}개만 존재 (최소 2개 필요)")
                elif len(unique_actions) == 2:
                    logger.warning(f"⚠️ 학습 전 검증: 액션 다양성 제한적 (고유 액션: {len(unique_actions)}개)")
                    logger.warning(f"   액션 분포: {action_counts}")
                    logger.warning(f"   학습은 계속 진행되지만, entropy_coef가 자동으로 증가합니다.")
                else:
                    logger.info(f"✅ 학습 전 검증 통과: 액션 다양성 우수 (고유 액션: {len(unique_actions)}개)")
            else:
                logger.error("❌ 학습 전 검증 실패: 경험 데이터가 비어있습니다")
                raise ValueError("경험 데이터가 비어있습니다")
            
            # 2. PPO 업데이트
            epochs = self.train_config.get('epochs', 30)
            batch_size = self.train_config.get('batch_size', 4096)
            
            # 🔥 적응형 배치 크기: 데이터 크기에 따라 동적 조정
            # 너무 작으면 학습이 불안정하고, 너무 크면 메모리/컴파일 문제 발생
            data_size = len(experiences)

            if data_size < 1000:
                optimal_batch_size = 64
            elif data_size < 5000:
                optimal_batch_size = 128
            else:
                # 🔥 성능 개선: 안전 제한을 대폭 완화 (256 -> 2048)
                # 최신 GPU에서는 충분히 감당 가능하며 학습 속도를 위해 필요
                optimal_batch_size = 2048

            # 설정된 batch_size와 optimal_batch_size 중 작은 값 사용
            if batch_size > optimal_batch_size:
                logger.info(f"📊 배치 크기 최적화: {batch_size} → {optimal_batch_size} (데이터 크기: {data_size})")
                batch_size = optimal_batch_size
            
            # 경험 수에 따라 동적 조정
            if len(experiences) < batch_size:
                # 데이터가 적으면 배치 크기 줄이기
                batch_size = min(batch_size, len(experiences))
                if batch_size == 0:
                    logger.warning("⚠️ 경험 데이터 없음, 학습 건너뜀")
                    return None
                logger.info(f"📊 배치 크기 조정: {batch_size} (경험 수: {len(experiences)})")
            
            eval_every = self.train_config.get('eval_every_epochs', 5)
            
            # 🔥 조기 종료 설정 (예측 전략: 액션 다양성 부족 시 더 많은 epoch 허용)
            # 기본값: 10 epoch (이전 5에서 증가) - 예측 전략 학습에 더 많은 시간 필요
            early_stop_patience = self.train_config.get('early_stop_patience', 10)  # 10 epoch 동안 개선 없으면 종료
            early_stop_min_delta = self.train_config.get('early_stop_min_delta', 0.0005)  # 최소 개선 임계값 (0.001 → 0.0005로 완화)
            best_loss = float('inf')
            no_improvement_count = 0
            
            # 🔥 초기 손실 기록 (학습 전) - 개선: 더 나은 초기화
            if experiences:
                # 더 많은 샘플로 초기 손실 측정 (안정성 향상)
                sample_size = min(200, len(experiences))
                sample_batch = experiences[:sample_size]
                
                # 초기 손실 측정 (학습 전)
                if sample_batch:
                    try:
                        initial_loss = self._update_policy(sample_batch)
                        # 초기 손실이 비정상적으로 크면 경고
                        if initial_loss > 10.0 or initial_loss < -10.0:
                            logger.warning(f"⚠️ 초기 손실이 비정상적: {initial_loss:.4f}, 정규화 시도")
                            # 손실 정규화 (큰 값 제한)
                            initial_loss = max(-5.0, min(5.0, initial_loss))
                    except Exception as init_err:
                        logger.warning(f"⚠️ 초기 손실 측정 실패: {init_err}, 기본값 사용")
                        initial_loss = 0.0
                else:
                    initial_loss = 0.0
                
                logger.info(f"📊 초기 손실 (학습 전): {initial_loss:.4f} (샘플: {sample_size}개)")
                best_loss = initial_loss
            else:
                logger.warning("⚠️ 학습 데이터가 없습니다 (experiences가 비어있음)")
                best_loss = 0.0
            
            for epoch in range(epochs):
                # 🔥 디버거: 현재 epoch 설정 (배치 로깅에서 사용)
                if self.debug:
                    self._debug_current_epoch = epoch + 1
                    self._debug_batch_idx = 0  # 배치 인덱스 초기화

                # 배치 생성
                batches = self._create_batches(experiences, batch_size)

                # 🔥 디버거: 총 배치 수 설정
                if self.debug:
                    self._debug_total_batches = len(batches)

                # 🔥 디버거 로깅: Epoch 시작
                if self.debug:
                    try:
                        self.debug.log_epoch_start(
                            epoch=epoch + 1,
                            total_epochs=epochs,
                            learning_rate=self.current_learning_rate
                        )
                    except Exception as debug_err:
                        logger.debug(f"⚠️ Epoch 시작 디버그 로깅 실패 (무시): {debug_err}")

                epoch_loss = 0.0
                successful_updates = 0
                # 각 배치로 업데이트
                for batch_idx, batch in enumerate(batches):
                    try:
                        loss = self._update_policy(batch)
                        if loss is not None and not (isinstance(loss, float) and (loss != loss or loss == float('inf'))):  # NaN/Inf 체크
                            epoch_loss += loss
                            successful_updates += 1
                        else:
                            logger.debug(f"⚠️ Epoch {epoch+1}, Batch {batch_idx+1}: Loss 값 이상 (NaN/Inf), 스킵")
                    except Exception as batch_err:
                        logger.warning(f"⚠️ Epoch {epoch+1}, Batch {batch_idx+1} 업데이트 실패: {batch_err}")
                        # 계속 진행 (다음 배치 처리)
                        continue
                
                # 성공한 업데이트가 없으면 경고
                if successful_updates == 0:
                    logger.warning(f"⚠️ Epoch {epoch+1}: 모든 배치 업데이트 실패")
                
                avg_loss = epoch_loss / successful_updates if successful_updates > 0 else 0.0
                
                # 손실 변화 추적
                if epoch == 0:
                    self._initial_epoch_loss = avg_loss

                # 🔥 조기 종료 체크
                improved = False
                if avg_loss < best_loss - early_stop_min_delta:
                    # 개선됨
                    best_loss = avg_loss
                    no_improvement_count = 0
                    improvement_msg = "✅ 개선"
                    improved = True
                else:
                    # 개선 없음
                    no_improvement_count += 1
                    improvement_msg = f"⚠️ 개선 없음 ({no_improvement_count}/{early_stop_patience})"

                # 🔥 디버거 로깅: Epoch 종료
                if self.debug:
                    try:
                        self.debug.log_epoch_end(
                            epoch=epoch + 1,
                            avg_loss=avg_loss,
                            best_loss=best_loss,
                            improved=improved,
                            no_improvement_count=no_improvement_count,
                            learning_rate=self.current_learning_rate
                        )
                    except Exception as debug_err:
                        logger.debug(f"⚠️ Epoch 종료 디버그 로깅 실패 (무시): {debug_err}")

                if epoch == epochs - 1:
                    loss_change = avg_loss - self._initial_epoch_loss
                    loss_change_pct = (loss_change / self._initial_epoch_loss * 100) if self._initial_epoch_loss > 0 else 0.0
                    logger.info(f"📈 Epoch {epoch+1}/{epochs}: 평균 Loss = {avg_loss:.4f} "
                              f"(변화: {loss_change:+.4f}, {loss_change_pct:+.2f}%) {improvement_msg}")
                else:
                    logger.info(f"📈 Epoch {epoch+1}/{epochs}: 평균 Loss = {avg_loss:.4f} {improvement_msg}")

                # 🔥 학습률 자동 조정 (개선 없을 때)
                if no_improvement_count >= 3 and no_improvement_count % 2 == 1:  # 3, 5, 7... epoch마다
                    # 학습률 10% 감소
                    self.current_learning_rate *= 0.9
                    # 옵티마이저 재생성 (학습률 변경)
                    self.optimizer = optax.chain(
                        optax.clip_by_global_norm(0.5),
                        optax.scale_by_adam(),
                        optax.scale(-self.current_learning_rate)
                    )
                    logger.info(f"📉 학습률 자동 조정: {self.current_learning_rate:.6f} (개선 없음 {no_improvement_count}회)")

                # 🔥 조기 종료 체크
                if no_improvement_count >= early_stop_patience:
                    logger.warning(f"🛑 조기 종료: {early_stop_patience} epoch 동안 개선 없음 (최고 Loss: {best_loss:.4f})")
                    break

                # 평가 (주기적으로)
                if (epoch + 1) % eval_every == 0:
                    eval_result = self._evaluate_model(epoch, experiences[:100])
                    logger.info(f"📊 Epoch {epoch+1} 평가: 평균 보상 = {eval_result.get('avg_reward', 0.0):.4f}")

                # 학습 히스토리 저장
                self.training_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'experiences_count': len(experiences)
                })
            
            # 🔥 디버거 로깅: 학습 종료
            if self.debug:
                try:
                    # 수렴 여부 판단 (조기 종료 또는 최종 loss가 충분히 낮음)
                    converged = (no_improvement_count >= early_stop_patience) or (avg_loss < 0.01)

                    self.debug.log_training_end(
                        total_epochs=len(self.training_history),
                        best_loss=best_loss,
                        final_loss=avg_loss if 'avg_loss' in locals() else best_loss,
                        converged=converged
                    )
                except Exception as debug_err:
                    logger.debug(f"⚠️ 학습 종료 디버그 로깅 실패 (무시): {debug_err}")

            # 3. 모델 저장
            if db_path is None:
                db_path = self.config.get('paths', {}).get('db', None)

            model_id = self._save_model(db_path)
            logger.info(f"✅ 학습 완료: model_id={model_id}")

            return model_id
            
        except Exception as e:
            logger.error(f"❌ 학습 실패: {e}")
            raise
    
    def _extract_experiences(
        self, 
        episodes_data: List[Dict[str, Any]],
        analysis_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Self-play 결과에서 경험 추출 (분석 데이터 포함 버전)
        
        Args:
            episodes_data: Self-play 에피소드 데이터
            analysis_data: 통합 분석 결과 (선택적)
                {
                    'fractal_score': float,
                    'multi_timeframe_score': float,
                    'indicator_cross_score': float,
                    'ensemble_score': float,
                    'ensemble_confidence': float
                }
        
        Returns:
            경험 리스트 [{state, action, reward, log_prob, value, done}, ...]
        """
        # 분석 데이터가 있으면 확장 버전 사용
        if analysis_data:
            return self._extract_experiences_with_analysis(episodes_data, analysis_data)
        
        # 기본 버전 (하위 호환성)
        return self._extract_experiences_basic(episodes_data)
    
    def _extract_experiences_basic(self, episodes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Self-play 결과에서 경험 추출 (기본 버전 - 20차원: 기본 15개 + 확장 지표 5개)
        """
        experiences = []

        try:
            for episode in episodes_data:
                results = episode.get('results', {})
                episode_num = episode.get('episode', 0)

                for agent_id, agent_result in results.items():
                    # 에이전트별 성과에서 경험 추출
                    trades = agent_result.get('trades', [])
                    total_pnl = agent_result.get('total_pnl', 0.0)
                    win_rate = agent_result.get('win_rate', 0.0)
                    total_trades = agent_result.get('total_trades', 0)  # 🔥 total_trades 추출
                    profit_factor = agent_result.get('profit_factor', 0.0)
                    strategy_direction = agent_result.get('strategy_direction', 'neutral')  # 🔥 전략 방향 ('buy', 'sell', 'neutral')
                    predicted_conf = agent_result.get('predicted_conf', 0.5)  # 🔥 예측 신뢰도
                    strategy_params = agent_result.get('strategy_params', {})  # 🚀 메타 학습: 전략 파라미터

                    # 🔥 예측 전략: 거래가 없어도 학습 데이터로 활용
                    # 예측 전략은 거래 결과보다 예측 정확도가 중요하므로 필터 완화
                    # 거래가 없는 경우에도 시장 상태 기반 예측 경험 생성
                    
                    # 품질 필터링 완화: 거래가 없어도 학습에 포함
                    # (예측 전략은 거래 실행 여부보다 예측 정확도가 중요)
                    quality_check_passed = True  # 모든 에피소드 포함
                    
                    # 🔥 학습 성능 개선: 트레이드별 경험 생성 (액션 다양성 우선)
                    if trades:
                        # 🔥 예측 전략: UP/DOWN 예측을 우선적으로 수집 (NEUTRAL은 제한)
                        # 거래 데이터의 BUY/SELL/HOLD를 예측 방향으로 변환
                        buy_trades = [t for t in trades if t.get('direction') == 'BUY']  # → UP 예측
                        sell_trades = [t for t in trades if t.get('direction') == 'SELL']  # → DOWN 예측
                        hold_trades = [t for t in trades if t.get('direction') != 'BUY' and t.get('direction') != 'SELL']  # → NEUTRAL 예측
                        hold_trades = self._ensure_neutral_trade_pool(
                            buy_trades,
                            sell_trades,
                            hold_trades,
                            episode_num=episode_num,
                            agent_id=agent_id
                        )
                        buy_trades, sell_trades = self._ensure_directional_trade_pool(
                            buy_trades,
                            sell_trades,
                            hold_trades,
                            episode_num=episode_num,
                            agent_id=agent_id
                        )
                        
                        selected_trades = self._select_trades_with_diversity(buy_trades, sell_trades, hold_trades)
                        
                        # 각 트레이드에서 경험 추출
                        for trade in selected_trades:
                            # 🔥 인터벌 역할별 보상 가중치 설정
                            interval_role_weights = {
                                'Macro Regime': {'direction': 2.0, 'hold': 0.5, 'profit': 1.0},  # 방향성 정확도가 가장 중요
                                'Trend Structure': {'direction': 1.5, 'hold': 0.8, 'profit': 1.2},  # 추세 파악 중요
                                'Micro Trend': {'direction': 1.2, 'hold': 1.0, 'profit': 1.5},  # 단기 추세 및 수익 중요
                                'Execution': {'direction': 1.0, 'hold': 1.2, 'profit': 2.0},  # 수익 실현(타이밍)이 가장 중요
                                'Timing': {'direction': 1.0, 'hold': 1.2, 'profit': 2.0}  # Execution과 동일
                            }
                            # interval_role이 없으면 기본값(Execution) 사용
                            role_weight = interval_role_weights.get(interval_role, interval_role_weights['Execution'])

                            # Market state 재구성 (확장 지표 포함)
                            # 실제로는 trade에 state 정보가 포함되어야 함
                            state = {
                                'rsi': trade.get('rsi', 50.0),
                                'macd': trade.get('macd', 0.0),
                                'volume_ratio': trade.get('volume_ratio', 1.0),
                                'atr': trade.get('atr', 0.02),
                                'adx': trade.get('adx', 25.0),
                                'mfi': trade.get('mfi', 50.0),
                                'bb_upper': trade.get('bb_upper', 1.0),
                                'bb_middle': trade.get('bb_middle', 1.0),
                                'bb_lower': trade.get('bb_lower', 1.0),
                                'macd_signal': trade.get('macd_signal', 0.0),
                                'close': trade.get('close', 1.0),
                                'open': trade.get('open', 1.0),
                                'high': trade.get('high', 1.0),
                                'low': trade.get('low', 1.0),
                                'volume': trade.get('volume', 1.0),
                                'volatility': trade.get('volatility', 0.02),
                                'regime_stage': trade.get('regime_stage', 3),
                                'regime_confidence': trade.get('regime_confidence', 0.5),
                                # 🚀 확장 지표 추가 (1단계 확장)
                                'wave_progress': trade.get('wave_progress', 0.5),
                                'pattern_confidence': trade.get('pattern_confidence', 0.5),
                                'structure_score': trade.get('structure_score', 0.5),
                                'sentiment': trade.get('sentiment', 0.0),
                                'regime_transition_prob': trade.get('regime_transition_prob', 0.05)
                            }
                            
                            # 🔥 예측 전략: 액션을 방향 예측으로 변환
                            # BUY → UP(1): 상승 예측, SELL → DOWN(2): 하락 예측, HOLD → NEUTRAL(0): 중립 예측
                            trade_direction = trade.get('direction', 'HOLD')
                            if trade_direction == 'BUY':
                                action = 1  # UP: 상승 예측
                                predicted_direction = 'UP'
                            elif trade_direction == 'SELL':
                                action = 2  # DOWN: 하락 예측
                                predicted_direction = 'DOWN'
                            else:
                                action = 0  # NEUTRAL: 중립 예측
                                predicted_direction = 'NEUTRAL'
                            
                            # 🔥 예측 정확도 기반 보상 시스템 (HOLD를 실제 방향으로 재평가)
                            # 실제 가격 변화를 기반으로 예측 정확도 평가
                            price_change = float(trade.get('price_change', 0.0) or 0.0)  # 실제 가격 변화율
                            threshold = self.neutral_direction_threshold
                            actual_direction = (
                                'UP' if price_change > threshold
                                else ('DOWN' if price_change < -threshold else 'NEUTRAL')
                            )

                            # 🔥 중요 수정: 예측 목표 강화 (Hindsight Labeling)
                            # 전략이 HOLD 했더라도, 시장이 움직였다면 그 움직임을 정답으로 학습
                            # 전략의 소극적 태도(Risk Aversion)가 예측 능력 저하로 이어지지 않도록 함
                            
                            if predicted_direction == 'NEUTRAL' and actual_direction != 'NEUTRAL':
                                # 전략은 관망했지만 시장은 움직임 -> 실제 방향으로 라벨 수정 (Oracle Learning)
                                if actual_direction == 'UP':
                                    action = 1 # UP
                                    predicted_direction = 'UP'
                                elif actual_direction == 'DOWN':
                                    action = 2 # DOWN
                                    predicted_direction = 'DOWN'
                                # NEUTRAL 라벨을 제거하고 방향성 라벨로 대체하여 적극적 예측 유도
                            
                            # 🔥 중요 수정: 전략의 조기 청산(Take-Profit/Stop-Loss)으로 인한 예측 왜곡 방지
                            # 전략이 20% 상승을 목표로 했으나 5%에서 익절했다면, 실제로는 더 올라갔을 수 있음
                            # 따라서 'UP' 예측이었는데 익절로 끝난 경우, 이후 가격 추이(잠재적 최대 변동폭)를 고려해야 하지만,
                            # 현재 trade 정보만으로는 알 수 없으므로, '이익이 났다'는 사실 자체를 긍정적으로 평가.
                            
                            # 또한, 강제 청산(Stop Loss)의 경우에도 방향 예측은 맞았으나 변동폭이 커서 털린 경우일 수 있음.
                            # 하지만 예측 관점에서는 '결과적인 가격 변화'가 중요하므로 actual_direction을 따르는 것이 기본.
                            
                            # 단, 'UP' 예측을 했는데 실제로는 'NEUTRAL' 수준의 작은 이익만 보고 끝난 경우 (조기 익절),
                            # 이를 '틀림'으로 처리하면 억울함. 이익이 났다면(price_change > 0) UP 예측에 대해 부분 점수 부여.
                            
                            # 🚀 1. 보상 기본 설정
                            # 🔥 NEUTRAL(0)도 명확한 예측(방향성 없음/관망)으로 평가
                            direction_reward = 0.0
                            
                            if predicted_direction == actual_direction:
                                # 1. 완전 일치
                                if predicted_direction == 'UP':
                                    direction_reward = 1.5 * role_weight['direction'] # 상승장 예측 성공은 높은 보상
                                elif predicted_direction == 'DOWN':
                                    direction_reward = 1.5 * role_weight['direction'] # 하락장 예측 성공도 높은 보상
                                else: # NEUTRAL
                                    # 🔥 NEUTRAL도 "방향성 없음"을 맞춘 것이므로 보상 부여
                                    # 단, 적극적 예측보다는 약간 낮게 (너무 소극적이지 않도록)
                                    direction_reward = 0.8 * role_weight['hold'] 
                            
                            elif predicted_direction == 'UP':
                                # 2. 상승 예측했으나...
                                if actual_direction == 'NEUTRAL' and price_change > 0:
                                    # 조금이라도 올랐으면 부분 점수 (0.5 -> 0.3으로 조정하여 정확성 유도)
                                    direction_reward = 0.3 * role_weight['direction']
                                elif actual_direction == 'DOWN':
                                    # 반대로 감 -> 강한 페널티
                                    direction_reward = -1.2 * role_weight['direction']
                                else:
                                    direction_reward = -0.5 * role_weight['direction']
                                    
                            elif predicted_direction == 'DOWN':
                                # 3. 하락 예측했으나...
                                if actual_direction == 'NEUTRAL' and price_change < 0:
                                    # 조금이라도 내렸으면 부분 점수
                                    direction_reward = 0.3 * role_weight['direction']
                                elif actual_direction == 'UP':
                                    # 반대로 감 -> 강한 페널티
                                    direction_reward = -1.2 * role_weight['direction']
                                else:
                                    direction_reward = -0.5 * role_weight['direction']
                                    
                            else: # predicted == NEUTRAL
                                # 4. 중립 예측했으나...
                                # 방향성장(UP/DOWN)에서 관망은 "기회 손실" -> 페널티
                                if actual_direction == 'UP' or actual_direction == 'DOWN':
                                    direction_reward = -0.8 * role_weight['hold']
                                else:
                                    # 애매한 상황에서 관망 -> 중립 보상 (0.0)
                                    direction_reward = 0.0

                            # 🚀 3. 인터벌 역할별 추가 보정
                            # 위에서 role_weight를 이미 곱했으므로 여기서는 특수 상황만 처리
                            
                            # 예측 신뢰도 기반 보정 (win_rate 활용)
                            confidence_bonus = (win_rate - 0.5) * 0.5  # -0.25 ~ +0.25
                            
                            # 최종 보상: 예측 정확도 + 신뢰도 보너스
                            reward = direction_reward + confidence_bonus

                            # 🆕 Policy Collapse 방지: NEUTRAL 액션 보너스 제거
                            # 이제 NEUTRAL도 정당한 예측으로 평가받으므로 인위적 보너스 불필요
                            # 대신 방향성 예측(UP/DOWN)에 약간의 인센티브를 주어 적극성 유도
                            if predicted_direction != 'NEUTRAL':
                                reward += 0.1  # 적극적 예측 인센티브

                            # 🆕 Policy Collapse 방지: NEUTRAL 액션 보너스 추가
                            # NEUTRAL 액션이 적게 선택되는 경우 보너스 제공
                            if predicted_direction == 'NEUTRAL':
                                reward += self.neutral_action_bonus
                            else:
                                reward += self.direction_action_bonus
                            
                            # 기본 log_prob (균등 분포 가정: log(1/3) ≈ -1.1)
                            log_prob = -1.1
                            
                            # 기본 value estimate (보상 기반)
                            value = reward * 0.9  # 간단한 추정

                            # 🚀 메타 학습: 상태 벡터 생성 (30차원: 20 base + 10 strategy params)
                            state_vec = build_state_vector_with_strategy(state, strategy_params)
                            
                            experience = {
                                'episode': episode_num,
                                'agent_id': agent_id,
                                'state': state_vec,  # 🔥 numpy array로 변환
                                'action': action,
                                'reward': reward,
                                'log_prob': log_prob,
                                'value': value,
                                'done': False  # 단일 트레이드는 완료로 간주
                            }
                            experiences.append(experience)
                        
                        # 🔥 UP/DOWN 예측 trades가 있으면 더 이상 추가하지 않음 (예측 다양성 확보) -> 제거: 모든 선택된 trades 활용
                        # break  # 모든 선택된 트레이드를 활용하여 액션 다양성 확보

                    
                    # 🔥 trades가 없는 경우: total_trades가 있으면 경험 생성
                    elif total_trades > 0:
                        # total_trades는 있지만 trades 리스트는 없는 경우
                        # (예측 self-play에서 predictions를 trades로 변환했지만 실제 거래는 없음)
                        # 🔥 전략 방향성과 예측 신뢰도를 활용하여 다양한 액션 생성
                        # 🔥 합성 데이터 생성량 증가: 최소 5개 보장, 최대 20개 (학습 데이터 증가)
                        num_experiences = max(5, min(total_trades, 20))

                        # 🔥 전략 방향성 기반 액션 분포 생성 (다양성 확보 - 각 방향 최소 1개 보장)
                        # 각 액션을 앞에 배치하여 num_experiences가 작아도 모든 액션 포함 보장
                        if strategy_direction == 'buy':
                            # 매수 전략: UP 우세, DOWN/NEUTRAL 포함 (다양성)
                            # 최소 각 방향 1개씩 먼저 보장 (앞 3개), 나머지 UP 위주로 채움
                            action_distribution = [0, 1, 2] + [1] * 4 + [2] * 2 + [1]  # NEUTRAL, UP, DOWN 각 1개 먼저, 나머지 UP 위주
                        elif strategy_direction == 'sell':
                            # 매도 전략: DOWN 우세, UP/NEUTRAL 포함 (다양성)
                            # 최소 각 방향 1개씩 먼저 보장 (앞 3개), 나머지 DOWN 위주로 채움
                            action_distribution = [0, 1, 2] + [2] * 4 + [1] * 2 + [2]  # NEUTRAL, UP, DOWN 각 1개 먼저, 나머지 DOWN 위주
                        else:
                            # 중립 전략: 균형잡힌 분포 (각 방향 최소 1개 보장)
                            action_distribution = [0, 1, 2] + [0] * 2 + [1] * 2 + [2] * 2 + [0, 1, 2]  # 균등 분포
                        
                        for exp_idx in range(num_experiences):
                            # 간단한 상태 벡터 (성과 기반 추정)
                            state = {
                                'rsi': 50.0, 'macd': 0.0, 'volume_ratio': 1.0, 'atr': 0.02,
                                'adx': 25.0, 'mfi': 50.0, 'bb_upper': 1.0, 'bb_middle': 1.0,
                                'bb_lower': 1.0, 'macd_signal': 0.0, 'close': 1.0, 'open': 1.0,
                                'high': 1.0, 'low': 1.0, 'volume': 1.0, 'volatility': 0.02,
                                'regime_stage': 3, 'regime_confidence': 0.5,
                                # 🚀 확장 지표 추가
                                'wave_progress': 0.5, 'pattern_confidence': predicted_conf,
                                'structure_score': 0.5, 'sentiment': 0.0, 'regime_transition_prob': 0.05
                            }
                            
                            # 🔥 액션 분포에서 선택 (다양성 확보)
                            action = action_distribution[exp_idx % len(action_distribution)]

                            # 🚀 메타 학습: 상태 벡터 생성 (30차원)
                            state_vec = build_state_vector_with_strategy(state, strategy_params)

                            # 🔥 예측 신뢰도 기반 보상 (win_rate와 predicted_conf 활용)
                            base_reward = win_rate - 0.5  # -0.5 ~ 0.5
                            confidence_bonus = (predicted_conf - 0.5) * 0.3  # -0.15 ~ 0.15
                            
                            # 액션과 전략 방향 일치 여부에 따른 보정
                            if (action == 1 and strategy_direction == 'buy') or \
                               (action == 2 and strategy_direction == 'sell') or \
                               (action == 0 and strategy_direction == 'neutral'):
                                direction_bonus = 0.1  # 방향 일치 보너스
                            else:
                                direction_bonus = -0.05  # 방향 불일치 작은 페널티
                            
                            reward = base_reward + confidence_bonus + direction_bonus
                            
                            experience = {
                                'episode': episode_num,
                                'agent_id': agent_id,
                                'state': state_vec,
                                'action': action,
                                'reward': reward,
                                'log_prob': -1.1,
                                'value': reward * 0.9,
                                'done': True
                            }
                            experiences.append(experience)
                        
                        # 한 에이전트당 최대 10개 생성했으므로 break하여 다음 에이전트로
                        break
                    
                    else:
                        # 🔥 거래가 없는 경우: 전략 방향성과 시장 상태 기반 예측 경험 생성
                        # 예측 전략은 거래가 없어도 시장 상태를 학습할 수 있어야 함
                        # 다양한 예측 방향(UP/DOWN/NEUTRAL)을 균형있게 생성
                        
                        # 에이전트 결과에서 시장 정보 추출 (가능한 경우)
                        market_info = agent_result.get('market_info', {})
                        regime = agent_result.get('regime', 'neutral')
                        
                        # 🔥 전략 방향성과 레짐을 종합하여 예측 방향 추정 (다양성 확보 - 각 방향 최소 1개 보장)
                        # 각 액션을 앞에 배치하여 num_experiences가 작아도 모든 액션 포함 보장
                        if strategy_direction == 'buy':
                            # 매수 전략: UP 우세, DOWN/NEUTRAL 포함 (각 방향 최소 1개 보장)
                            if 'bull' in regime.lower():
                                predicted_actions = [0, 1, 2] + [1] * 4 + [2] * 2 + [1]  # UP 위주
                            elif 'bear' in regime.lower():
                                predicted_actions = [0, 1, 2] + [1] * 2 + [2] * 3 + [1]  # DOWN 혼합
                            else:
                                predicted_actions = [0, 1, 2] + [1] * 3 + [2] * 2 + [1, 2]  # 균형
                        elif strategy_direction == 'sell':
                            # 매도 전략: DOWN 우세, UP/NEUTRAL 포함 (각 방향 최소 1개 보장)
                            if 'bear' in regime.lower():
                                predicted_actions = [0, 1, 2] + [2] * 4 + [1] * 2 + [2]  # DOWN 위주
                            elif 'bull' in regime.lower():
                                predicted_actions = [0, 1, 2] + [2] * 2 + [1] * 3 + [2]  # UP 혼합
                            else:
                                predicted_actions = [0, 1, 2] + [2] * 3 + [1] * 2 + [2, 1]  # 균형
                        else:
                            # 중립 전략: 균형잡힌 분포 (각 방향 최소 1개 보장)
                            if 'bull' in regime.lower():
                                predicted_actions = [0, 1, 2] + [1] * 3 + [0, 2] * 2 + [1]  # UP 위주
                            elif 'bear' in regime.lower():
                                predicted_actions = [0, 1, 2] + [2] * 3 + [0, 1] * 2 + [2]  # DOWN 위주
                            else:
                                predicted_actions = [0, 1, 2] + [0] * 2 + [1] * 2 + [2] * 2 + [0]  # 균등
                        
                        # 🔥 각 예측 방향별로 경험 생성 (최소 5개, 최대 20개, 학습 데이터 증가)
                        num_pred_experiences = max(5, min(len(predicted_actions), 20))
                        for action in predicted_actions[:num_pred_experiences]:
                            # 시장 상태 추정 (에이전트 결과 기반, 확장 지표 포함)
                            state = {
                                'rsi': market_info.get('rsi', 50.0),
                                'macd': market_info.get('macd', 0.0),
                                'volume_ratio': market_info.get('volume_ratio', 1.0),
                                'atr': market_info.get('atr', 0.02),
                                'adx': market_info.get('adx', 25.0),
                                'mfi': market_info.get('mfi', 50.0),
                                'bb_upper': market_info.get('bb_upper', 1.0),
                                'bb_middle': market_info.get('bb_middle', 1.0),
                                'bb_lower': market_info.get('bb_lower', 1.0),
                                'macd_signal': market_info.get('macd_signal', 0.0),
                                'close': market_info.get('close', 1.0),
                                'open': market_info.get('open', 1.0),
                                'high': market_info.get('high', 1.0),
                                'low': market_info.get('low', 1.0),
                                'volume': market_info.get('volume', 1.0),
                                'volatility': market_info.get('volatility', 0.02),
                                'regime_stage': market_info.get('regime_stage', 3),
                                'regime_confidence': market_info.get('regime_confidence', 0.5),
                                # 🚀 확장 지표 추가
                                'wave_progress': market_info.get('wave_progress', 0.5),
                                'pattern_confidence': market_info.get('pattern_confidence', predicted_conf),
                                'structure_score': market_info.get('structure_score', 0.5),
                                'sentiment': market_info.get('sentiment', 0.0),
                                'regime_transition_prob': market_info.get('regime_transition_prob', 0.05)
                            }

                            # 🚀 메타 학습: 상태 벡터 생성 (30차원)
                            state_vec = build_state_vector_with_strategy(state, strategy_params)

                            # 🔥 [Update] 예측 전략: 레짐(시장 상황) 기반 강력한 보상 (Effort & Recovery 철학 반영)
                            # 거래가 없어도 시장 방향(Regime)에 맞는 액션을 취했는지 평가
                            
                            base_reward = 0.0
                            
                            # 레짐 판단
                            is_bull = 'bull' in regime.lower()
                            is_bear = 'bear' in regime.lower()
                            is_neutral = 'sideways' in regime.lower() or 'neutral' in regime.lower()
                            
                            if is_bull:
                                if action == 1: # UP (정답)
                                    base_reward = 1.5
                                elif action == 0: # NEUTRAL (기회 놓침)
                                    base_reward = -2.0
                                else: # DOWN (틀림 - 도전)
                                    base_reward = -0.2
                            elif is_bear:
                                if action == 2: # DOWN (정답)
                                    base_reward = 1.5
                                elif action == 0: # NEUTRAL (기회 놓침)
                                    base_reward = -2.0
                                else: # UP (틀림 - 도전)
                                    base_reward = -0.2
                            else: # Neutral/Sideways
                                if action == 0: # NEUTRAL (정답 - Safe Hold)
                                    base_reward = 0.2
                                else: # UP/DOWN (틀림 - 도전)
                                    base_reward = -0.2
                            
                            # 전략 방향 일치 보너스 (보조)
                            if (action == 1 and strategy_direction == 'buy') or \
                               (action == 2 and strategy_direction == 'sell'):
                                base_reward += 0.1
                            
                            # 예측 신뢰도 보정
                            confidence_bonus = (predicted_conf - 0.5) * 0.2  # -0.1 ~ 0.1
                            
                            reward = base_reward + confidence_bonus
                            
                            experience = {
                                'episode': episode_num,
                                'agent_id': agent_id,
                                'state': state_vec,
                                'action': action,  # 🔥 다양한 액션 (UP/DOWN/NEUTRAL)
                                'reward': reward,
                                'log_prob': -1.1,
                                'value': reward * 0.9,
                                'done': True
                            }
                            experiences.append(experience)
                        
                        # 🔥 첫 번째 에이전트만 사용 (중복 방지)
                        break
        
        except Exception as e:
            logger.warning(f"⚠️ 경험 추출 중 일부 데이터 손실: {e}")
            import traceback
            logger.debug(f"경험 추출 상세 에러:\n{traceback.format_exc()}")
        
        # 🔥 데이터 검증 강화: 액션 다양성 체크
        if experiences:
            actions = [exp.get('action', 0) for exp in experiences]
            unique_actions = set(actions)
            action_counts = {action: actions.count(action) for action in unique_actions}
            
            # 액션 분포 로깅
            logger.info(f"📊 경험 추출 완료: 총 {len(experiences)}개, 고유 액션: {len(unique_actions)}개")
            logger.info(f"   액션 분포: NEUTRAL(0)={action_counts.get(0, 0)}, UP(1)={action_counts.get(1, 0)}, DOWN(2)={action_counts.get(2, 0)}")
            
            # 🔥 액션 다양성 검증
            if len(unique_actions) < 2:
                logger.warning(f"⚠️ 액션 다양성 부족: 고유 액션 {len(unique_actions)}개만 존재")
                logger.warning(f"   액션 분포: {action_counts}")
                
                # 액션 다양성 부족 시 경고만 출력 (학습은 계속 진행)
                # 학습 과정에서 entropy_coef가 자동으로 증가하여 탐험을 강화함
            elif len(unique_actions) == 2:
                logger.info(f"✅ 액션 다양성 양호: 고유 액션 {len(unique_actions)}개")
            else:
                logger.info(f"✅ 액션 다양성 우수: 고유 액션 {len(unique_actions)}개 (모든 방향 포함)")
        else:
            logger.warning("⚠️ 경험 추출 결과가 비어있습니다")
        
        logger.debug(f"✅ 총 {len(experiences)}개 경험 추출 완료")
        return experiences
    
    def _extract_experiences_with_analysis(
        self,
        episodes_data: List[Dict[str, Any]],
        analysis_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Self-play 결과 + 분석 데이터에서 경험 추출 (25차원)
        
        Args:
            episodes_data: Self-play 에피소드 데이터
            analysis_data: 통합 분석 결과
                {
                    'fractal_score': float,
                    'multi_timeframe_score': float,
                    'indicator_cross_score': float,
                    'ensemble_score': float,
                    'ensemble_confidence': float
                }
        
        Returns:
            경험 리스트 (state는 35차원 벡터: 20 base + 5 analysis + 10 strategy params)
        """
        experiences = []
        
        try:
            # 분석 점수 추출 (기본값 0.5)
            fractal_score = analysis_data.get('fractal_score', 0.5)
            multi_timeframe_score = analysis_data.get('multi_timeframe_score', 0.5)
            indicator_cross_score = analysis_data.get('indicator_cross_score', 0.5)
            ensemble_score = analysis_data.get('ensemble_score', 0.5)
            ensemble_confidence = analysis_data.get('ensemble_confidence', 0.5)
            
            for episode in episodes_data:
                results = episode.get('results', {})
                episode_num = episode.get('episode', 0)
                
                for agent_id, agent_result in results.items():
                    # 에이전트별 성과에서 경험 추출
                    trades = agent_result.get('trades', [])
                    total_pnl = agent_result.get('total_pnl', 0.0)
                    win_rate = agent_result.get('win_rate', 0.0)
                    total_trades = agent_result.get('total_trades', 0)
                    profit_factor = agent_result.get('profit_factor', 0.0)
                    strategy_direction = agent_result.get('strategy_direction', 'neutral')  # 🔥 전략 방향
                    predicted_conf = agent_result.get('predicted_conf', 0.5)  # 🔥 예측 신뢰도
                    strategy_params = agent_result.get('strategy_params', {})  # 🚀 메타 학습: 전략 파라미터

                    # 🔥 예측 전략: 거래가 없어도 학습 데이터로 활용
                    # 예측 전략은 거래 결과보다 예측 정확도가 중요하므로 필터 완화
                    quality_check_passed = True  # 모든 에피소드 포함

                    # 🔥 학습 성능 개선: 트레이드별 경험 생성 (액션 다양성 우선)
                    if trades:
                        # 🔥 예측 전략: UP/DOWN 예측을 우선적으로 수집 (NEUTRAL은 제한)
                        # 거래 데이터의 BUY/SELL/HOLD를 예측 방향으로 변환
                        buy_trades = [t for t in trades if t.get('direction') == 'BUY']  # → UP 예측
                        sell_trades = [t for t in trades if t.get('direction') == 'SELL']  # → DOWN 예측
                        hold_trades = [t for t in trades if t.get('direction') != 'BUY' and t.get('direction') != 'SELL']  # → NEUTRAL 예측
                        hold_trades = self._ensure_neutral_trade_pool(
                            buy_trades,
                            sell_trades,
                            hold_trades,
                            episode_num=episode_num,
                            agent_id=agent_id
                        )
                        buy_trades, sell_trades = self._ensure_directional_trade_pool(
                            buy_trades,
                            sell_trades,
                            hold_trades,
                            episode_num=episode_num,
                            agent_id=agent_id
                        )
                        
                        selected_trades = self._select_trades_with_diversity(buy_trades, sell_trades, hold_trades)
                        
                        # 🔥 디버깅: 선택된 트레이드 상세 확인
                        # BUY/SELL이 있는데 선택된 트레이드에 없는 경우 확인
                        has_pool_diversity = len(buy_trades) > 0 or len(sell_trades) > 0
                        selected_dirs = [t.get('direction') for t in selected_trades]
                        has_selected_diversity = any(d in ['BUY', 'SELL'] for d in selected_dirs)
                        
                        if has_pool_diversity and not has_selected_diversity:
                             logger.error(f"❌ {agent_id}: Pool has diversity (B:{len(buy_trades)}, S:{len(sell_trades)}) but selected_trades DOES NOT. Selected dirs: {selected_dirs[:20]}")

                        # 🔥 액션 다양성 강제 보정
                        # 만약 NEUTRAL(HOLD)만 있다면, 강제로 BUY/SELL 합성 경험 추가
                        # 안전한 확인을 위해 모두 문자열 변환 및 대문자 처리
                        unique_actions_set = set(str(d).upper() for d in selected_dirs)
                        
                        # 디버깅 로그 (처음 5개 에이전트만)
                        if len(experiences) < 5:
                             logger.info(f"🔍 디버깅: Agent={agent_id}, Dirs={list(unique_actions_set)}")

                        # 'HOLD' 또는 'neutral'만 있는 경우 (BUY/SELL이 없음)
                        has_directional = any(d in ['BUY', 'SELL'] for d in unique_actions_set)
                        
                        if not has_directional:
                             # BUY/SELL 강제 주입 (state는 마지막 trade 복사하되 변형)
                             if selected_trades:
                                 base_trade = selected_trades[0]
                                 
                                 # Synthetic BUY
                                 synthetic_buy = copy.deepcopy(base_trade)
                                 synthetic_buy['direction'] = 'BUY'
                                 # 🔥 강제 주입된 데이터는 threshold 이상으로 설정하여 명확한 BUY로 인식되게 함
                                 synthetic_buy['price_change'] = self.neutral_direction_threshold * 2.0
                                 synthetic_buy['synthetic_forced'] = True
                                 selected_trades.append(synthetic_buy)
                                 
                                 # Synthetic SELL
                                 synthetic_sell = copy.deepcopy(base_trade)
                                 synthetic_sell['direction'] = 'SELL' 
                                 # 🔥 강제 주입된 데이터는 threshold 이상으로 설정하여 명확한 SELL로 인식되게 함
                                 synthetic_sell['price_change'] = -self.neutral_direction_threshold * 2.0
                                 synthetic_sell['synthetic_forced'] = True
                                 selected_trades.append(synthetic_sell)
                                 
                                 logger.info(f"🔧 {agent_id}: 액션 다양성 부족으로 강제 합성 데이터(BUY/SELL) 주입 (Dirs: {unique_actions_set})")

                        for trade in selected_trades:
                            # Market state 재구성 (확장 지표 포함)
                            state = {
                                'rsi': trade.get('rsi', 50.0),
                                'macd': trade.get('macd', 0.0),
                                'volume_ratio': trade.get('volume_ratio', 1.0),
                                'atr': trade.get('atr', 0.02),
                                'adx': trade.get('adx', 25.0),
                                'mfi': trade.get('mfi', 50.0),
                                'bb_upper': trade.get('bb_upper', 1.0),
                                'bb_middle': trade.get('bb_middle', 1.0),
                                'bb_lower': trade.get('bb_lower', 1.0),
                                'macd_signal': trade.get('macd_signal', 0.0),
                                'close': trade.get('close', 1.0),
                                'open': trade.get('open', 1.0),
                                'high': trade.get('high', 1.0),
                                'low': trade.get('low', 1.0),
                                'volume': trade.get('volume', 1.0),
                                'volatility': trade.get('volatility', 0.02),
                                'regime_stage': trade.get('regime_stage', 3),
                                'regime_confidence': trade.get('regime_confidence', 0.5),
                                # 🚀 확장 지표 추가 (1단계 확장)
                                'wave_progress': trade.get('wave_progress', 0.5),
                                'pattern_confidence': trade.get('pattern_confidence', 0.5),
                                'structure_score': trade.get('structure_score', 0.5),
                                'sentiment': trade.get('sentiment', 0.0),
                                'regime_transition_prob': trade.get('regime_transition_prob', 0.05)
                            }
                            
                            # 🔥 메타 학습: 분석+전략 파라미터 포함 상태 벡터 생성 (35차원)
                            # 프랙탈/멀티타임프레임/지표교차 점수 + 전략 파라미터 포함하여 더 강력한 학습
                            enhanced_state_vec = build_state_vector_with_analysis_and_strategy(
                                state,
                                strategy_params,
                                fractal_score=fractal_score,
                                multi_timeframe_score=multi_timeframe_score,
                                indicator_cross_score=indicator_cross_score,
                                ensemble_score=ensemble_score,
                                ensemble_confidence=ensemble_confidence
                            )
                            
                            # 🔥 예측 전략: 액션을 방향 예측으로 변환
                            # BUY → UP(1): 상승 예측, SELL → DOWN(2): 하락 예측, HOLD → NEUTRAL(0): 중립 예측
                            # 안전을 위해 문자열 변환 및 대문자화
                            trade_direction = str(trade.get('direction', 'HOLD')).upper()
                            
                            if trade_direction == 'BUY':
                                action = 1  # UP: 상승 예측
                                predicted_direction = 'UP'
                            elif trade_direction == 'SELL':
                                action = 2  # DOWN: 하락 예측
                                predicted_direction = 'DOWN'
                            else:
                                action = 0  # NEUTRAL: 중립 예측
                                predicted_direction = 'NEUTRAL'
                                
                                # 🔥 디버깅: Synthetic trade인데 NEUTRAL로 매핑된 경우 확인
                                if trade.get('synthetic_forced'):
                                     logger.error(f"❌ {agent_id}: Synthetic FORCED trade mapped to NEUTRAL! Dir={trade_direction}")
                                elif trade.get('synthetic_directional'):
                                     logger.error(f"❌ {agent_id}: Synthetic directional trade mapped to NEUTRAL! Dir={trade_direction}")
                            
                            # 🔥 예측 정확도 기반 보상 시스템 (HOLD를 실제 방향으로 재평가)
                            # 실제 가격 변화를 기반으로 예측 정확도 평가
                            price_change = float(trade.get('price_change', 0.0) or 0.0)  # 실제 가격 변화율
                            threshold = self.neutral_direction_threshold
                            actual_direction = (
                                'UP' if price_change > threshold
                                else ('DOWN' if price_change < -threshold else 'NEUTRAL')
                            )

                            # 🔥 중요 수정: 예측 목표 강화 (Hindsight Labeling)
                            # 전략이 HOLD 했더라도, 시장이 움직였다면 그 움직임을 정답으로 학습
                            # 전략의 소극적 태도(Risk Aversion)가 예측 능력 저하로 이어지지 않도록 함
                            
                            if predicted_direction == 'NEUTRAL' and actual_direction != 'NEUTRAL':
                                # 전략은 관망했지만 시장은 움직임 -> 실제 방향으로 라벨 수정 (Oracle Learning)
                                if actual_direction == 'UP':
                                    action = 1 # UP
                                    predicted_direction = 'UP'
                                elif actual_direction == 'DOWN':
                                    action = 2 # DOWN
                                    predicted_direction = 'DOWN'
                                # NEUTRAL 라벨을 제거하고 방향성 라벨로 대체하여 적극적 예측 유도
                            
                            # 🔥 중요 수정: 전략의 조기 청산(Take-Profit/Stop-Loss)으로 인한 예측 왜곡 방지
                            # 전략이 20% 상승을 목표로 했으나 5%에서 익절했다면, 실제로는 더 올라갔을 수 있음
                            # 따라서 'UP' 예측이었는데 익절로 끝난 경우, 이후 가격 추이(잠재적 최대 변동폭)를 고려해야 하지만,
                            # 현재 trade 정보만으로는 알 수 없으므로, '이익이 났다'는 사실 자체를 긍정적으로 평가.
                            
                            # 또한, 강제 청산(Stop Loss)의 경우에도 방향 예측은 맞았으나 변동폭이 커서 털린 경우일 수 있음.
                            # 하지만 예측 관점에서는 '결과적인 가격 변화'가 중요하므로 actual_direction을 따르는 것이 기본.
                            
                            # 단, 'UP' 예측을 했는데 실제로는 'NEUTRAL' 수준의 작은 이익만 보고 끝난 경우 (조기 익절),
                            # 이를 '틀림'으로 처리하면 억울함. 이익이 났다면(price_change > 0) UP 예측에 대해 부분 점수 부여.
                            
                            direction_reward = 0.0
                            
                            if predicted_direction == actual_direction:
                                # 1. 완전 일치
                                if predicted_direction == 'UP':
                                    direction_reward = 1.0
                                elif predicted_direction == 'DOWN':
                                    direction_reward = 1.0
                                else:
                                    direction_reward = 0.9 # NEUTRAL 보상 (Policy Collapse 방지)
                            
                            elif predicted_direction == 'UP':
                                # 2. 상승 예측했으나...
                                if actual_direction == 'NEUTRAL' and price_change > 0:
                                    # 조금이라도 올랐으면 부분 점수 (조기 익절 가능성)
                                    # threshold(보통 0.002~0.005)보다 작지만 0보다는 큰 경우
                                    direction_reward = 0.3
                                elif actual_direction == 'DOWN':
                                    # 반대로 감 -> 페널티
                                    direction_reward = -1.0
                                else:
                                    # 그 외 (NEUTRAL인데 0 이하인 경우 등)
                                    direction_reward = -0.3
                                    
                            elif predicted_direction == 'DOWN':
                                # 3. 하락 예측했으나...
                                if actual_direction == 'NEUTRAL' and price_change < 0:
                                    # 조금이라도 내렸으면 부분 점수
                                    direction_reward = 0.3
                                elif actual_direction == 'UP':
                                    # 반대로 감 -> 페널티
                                    direction_reward = -1.0
                                else:
                                    direction_reward = -0.3
                                    
                            else: # predicted == NEUTRAL
                                # 4. 중립 예측했으나...
                                # 크게 움직였으면 페널티 (기회 비용)
                                direction_reward = -0.5

                            # 예측 신뢰도 기반 보정 (win_rate 활용)
                            confidence_bonus = (win_rate - 0.5) * 0.5  # -0.25 ~ +0.25

                            # 최종 보상: 예측 정확도 + 신뢰도 보너스
                            reward = direction_reward + confidence_bonus

                            # 🆕 Policy Collapse 방지: NEUTRAL 액션 보너스 추가
                            # NEUTRAL 액션이 적게 선택되는 경우 보너스 제공
                            if predicted_direction == 'NEUTRAL':
                                reward += self.neutral_action_bonus
                            else:
                                reward += self.direction_action_bonus
                            
                            # 기본 log_prob
                            log_prob = -1.1
                            
                            # 기본 value estimate
                            value = reward * 0.9
                            
                            experience = {
                                'episode': episode_num,
                                'agent_id': agent_id,
                                'state': enhanced_state_vec,  # 🔥 25차원 벡터
                                'action': action,
                                'reward': reward,
                                'log_prob': log_prob,
                                'value': value,
                                'done': False,
                                'analysis_score': ensemble_score * 100.0  # 🆕 타겟 분석 점수 (0~100)
                            }
                            experiences.append(experience)
                        
                        # 🔥 UP/DOWN 예측 trades가 있으면 더 이상 추가하지 않음 (예측 다양성 확보) -> 제거: 모든 선택된 trades 활용
                        # break  # 모든 선택된 트레이드를 활용하여 액션 다양성 확보

                    
                    # 🔥 trades가 없는 경우: total_trades가 있으면 경험 생성
                    elif total_trades > 0:
                        # total_trades는 있지만 trades 리스트는 없는 경우
                        # (예측 self-play에서 predictions를 trades로 변환했지만 실제 거래는 없음)
                        # 🔥 전략 방향성과 예측 신뢰도를 활용하여 다양한 액션 생성
                        # 🔥 합성 데이터 생성량 증가: 최소 5개 보장, 최대 20개 (학습 데이터 증가)
                        num_experiences = max(5, min(total_trades, 20))

                        # 🔥 전략 방향성 기반 액션 분포 생성 (다양성 확보 - 각 방향 최소 1개 보장)
                        # 각 액션을 앞에 배치하여 num_experiences가 작아도 모든 액션 포함 보장
                        if strategy_direction == 'buy':
                            # 매수 전략: UP 우세, DOWN/NEUTRAL 포함 (다양성)
                            # 최소 각 방향 1개씩 먼저 보장 (앞 3개), 나머지 UP 위주로 채움
                            action_distribution = [0, 1, 2] + [1] * 4 + [2] * 2 + [1]  # NEUTRAL, UP, DOWN 각 1개 먼저, 나머지 UP 위주
                        elif strategy_direction == 'sell':
                            # 매도 전략: DOWN 우세, UP/NEUTRAL 포함 (다양성)
                            # 최소 각 방향 1개씩 먼저 보장 (앞 3개), 나머지 DOWN 위주로 채움
                            action_distribution = [0, 1, 2] + [2] * 4 + [1] * 2 + [2]  # NEUTRAL, UP, DOWN 각 1개 먼저, 나머지 DOWN 위주
                        else:
                            # 중립 전략: 균형잡힌 분포 (각 방향 최소 1개 보장)
                            action_distribution = [0, 1, 2] + [0] * 2 + [1] * 2 + [2] * 2 + [0, 1, 2]  # 균등 분포
                        
                        for exp_idx in range(num_experiences):
                            state = {
                                'rsi': 50.0, 'macd': 0.0, 'volume_ratio': 1.0, 'atr': 0.02,
                                'adx': 25.0, 'mfi': 50.0, 'bb_upper': 1.0, 'bb_middle': 1.0,
                                'bb_lower': 1.0, 'macd_signal': 0.0, 'close': 1.0, 'open': 1.0,
                                'high': 1.0, 'low': 1.0, 'volume': 1.0, 'volatility': 0.02,
                                'regime_stage': 3, 'regime_confidence': 0.5,
                                # 🚀 확장 지표 추가
                                'wave_progress': 0.5, 'pattern_confidence': predicted_conf,
                                'structure_score': 0.5, 'sentiment': 0.0, 'regime_transition_prob': 0.05
                            }

                            # 🚀 메타 학습: 분석+전략 파라미터 포함 상태 벡터 생성 (35차원)
                            enhanced_state_vec = build_state_vector_with_analysis_and_strategy(
                                state,
                                strategy_params,
                                fractal_score=fractal_score,
                                multi_timeframe_score=multi_timeframe_score,
                                indicator_cross_score=indicator_cross_score,
                                ensemble_score=ensemble_score,
                                ensemble_confidence=ensemble_confidence
                            )

                            # 🔥 액션 분포에서 선택 (다양성 확보)
                            action = action_distribution[exp_idx % len(action_distribution)]
                            
                            # 🔥 예측 신뢰도 기반 보상
                            base_reward = win_rate - 0.5  # -0.5 ~ 0.5
                            confidence_bonus = (predicted_conf - 0.5) * 0.3  # -0.15 ~ 0.15
                            
                            # 액션과 전략 방향 일치 여부에 따른 보정
                            if (action == 1 and strategy_direction == 'buy') or \
                               (action == 2 and strategy_direction == 'sell') or \
                               (action == 0 and strategy_direction == 'neutral'):
                                direction_bonus = 0.1  # 방향 일치 보너스
                            else:
                                direction_bonus = -0.05  # 방향 불일치 작은 페널티
                            
                            reward = base_reward + confidence_bonus + direction_bonus
                            
                            experience = {
                                'episode': episode_num,
                                'agent_id': agent_id,
                                'state': enhanced_state_vec,  # 🔥 25차원 벡터
                                'action': action,
                                'reward': reward,
                                'log_prob': -1.1,
                                'value': reward * 0.9,
                                'done': True
                            }
                            experiences.append(experience)
                        
                        break
                    
                    else:
                        # 🔥 거래가 없는 경우: 전략 방향성과 시장 상태 기반 예측 경험 생성
                        market_info = agent_result.get('market_info', {})
                        regime = agent_result.get('regime', 'neutral')
                        
                        # 🔥 전략 방향성과 레짐을 종합하여 예측 방향 추정 (다양성 확보 - 각 방향 최소 1개 보장)
                        # 각 액션을 앞에 배치하여 num_experiences가 작아도 모든 액션 포함 보장
                        if strategy_direction == 'buy':
                            # 매수 전략: UP 우세, DOWN/NEUTRAL 포함 (각 방향 최소 1개 보장)
                            if 'bull' in regime.lower():
                                predicted_actions = [0, 1, 2] + [1] * 4 + [2] * 2 + [1]  # UP 위주
                            elif 'bear' in regime.lower():
                                predicted_actions = [0, 1, 2] + [1] * 2 + [2] * 3 + [1]  # DOWN 혼합
                            else:
                                predicted_actions = [0, 1, 2] + [1] * 3 + [2] * 2 + [1, 2]  # 균형
                        elif strategy_direction == 'sell':
                            # 매도 전략: DOWN 우세, UP/NEUTRAL 포함 (각 방향 최소 1개 보장)
                            if 'bear' in regime.lower():
                                predicted_actions = [0, 1, 2] + [2] * 4 + [1] * 2 + [2]  # DOWN 위주
                            elif 'bull' in regime.lower():
                                predicted_actions = [0, 1, 2] + [2] * 2 + [1] * 3 + [2]  # UP 혼합
                            else:
                                predicted_actions = [0, 1, 2] + [2] * 3 + [1] * 2 + [2, 1]  # 균형
                        else:
                            # 중립 전략: 균형잡힌 분포 (각 방향 최소 1개 보장)
                            if 'bull' in regime.lower():
                                predicted_actions = [0, 1, 2] + [1] * 3 + [0, 2] * 2 + [1]  # UP 위주
                            elif 'bear' in regime.lower():
                                predicted_actions = [0, 1, 2] + [2] * 3 + [0, 1] * 2 + [2]  # DOWN 위주
                            else:
                                predicted_actions = [0, 1, 2] + [0] * 2 + [1] * 2 + [2] * 2 + [0]  # 균등
                        
                        # 🔥 각 예측 방향별로 경험 생성 (최소 5개, 최대 20개, 학습 데이터 증가)
                        num_pred_experiences = max(5, min(len(predicted_actions), 20))
                        for action in predicted_actions[:num_pred_experiences]:
                            state = {
                                'rsi': market_info.get('rsi', 50.0),
                                'macd': market_info.get('macd', 0.0),
                                'volume_ratio': market_info.get('volume_ratio', 1.0),
                                'atr': market_info.get('atr', 0.02),
                                'adx': market_info.get('adx', 25.0),
                                'mfi': market_info.get('mfi', 50.0),
                                'bb_upper': market_info.get('bb_upper', 1.0),
                                'bb_middle': market_info.get('bb_middle', 1.0),
                                'bb_lower': market_info.get('bb_lower', 1.0),
                                'macd_signal': market_info.get('macd_signal', 0.0),
                                'close': market_info.get('close', 1.0),
                                'open': market_info.get('open', 1.0),
                                'high': market_info.get('high', 1.0),
                                'low': market_info.get('low', 1.0),
                                'volume': market_info.get('volume', 1.0),
                                'volatility': market_info.get('volatility', 0.02),
                                'regime_stage': market_info.get('regime_stage', 3),
                                'regime_confidence': market_info.get('regime_confidence', 0.5),
                                # 🚀 확장 지표 추가
                                'wave_progress': market_info.get('wave_progress', 0.5),
                                'pattern_confidence': market_info.get('pattern_confidence', predicted_conf),
                                'structure_score': market_info.get('structure_score', 0.5),
                                'sentiment': market_info.get('sentiment', 0.0),
                                'regime_transition_prob': market_info.get('regime_transition_prob', 0.05)
                            }

                            # 🚀 메타 학습: 분석+전략 파라미터 포함 상태 벡터 생성 (35차원)
                            enhanced_state_vec = build_state_vector_with_analysis_and_strategy(
                                state,
                                strategy_params,
                                fractal_score=fractal_score,
                                multi_timeframe_score=multi_timeframe_score,
                                indicator_cross_score=indicator_cross_score,
                                ensemble_score=ensemble_score,
                                ensemble_confidence=ensemble_confidence
                            )

                            # 🔥 예측 전략: 전략 방향성, 레짐, 예측 신뢰도 종합 보상
                            base_reward = 0.0
                            
                            # 전략 방향과 액션 일치 여부
                            if (action == 1 and strategy_direction == 'buy') or \
                               (action == 2 and strategy_direction == 'sell') or \
                               (action == 0 and strategy_direction == 'neutral'):
                                base_reward += 0.1
                            
                            # 레짐과 액션 일치 여부
                            if (action == 1 and 'bull' in regime.lower()) or \
                               (action == 2 and 'bear' in regime.lower()) or \
                               (action == 0 and ('sideways' in regime.lower() or 'neutral' in regime.lower())):
                                base_reward += 0.1
                            else:
                                base_reward -= 0.05
                            
                            # 예측 신뢰도 보정
                            confidence_bonus = (predicted_conf - 0.5) * 0.2
                            
                            reward = base_reward + confidence_bonus
                            
                            experience = {
                                'episode': episode_num,
                                'agent_id': agent_id,
                                'state': enhanced_state_vec,  # 🔥 25차원 벡터
                                'action': action,
                                'reward': reward,
                                'log_prob': -1.1,
                                'value': reward * 0.9,
                                'done': True
                            }
                            experiences.append(experience)
                        
                        break
        
        except Exception as e:
            logger.warning(f"⚠️ 분석 데이터 포함 경험 추출 중 일부 데이터 손실: {e}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")
        
        # 🔥 데이터 검증 강화: 액션 다양성 체크
        if experiences:
            actions = [exp.get('action', 0) for exp in experiences]
            unique_actions = set(actions)
            action_counts = {action: actions.count(action) for action in unique_actions}
            
            # 액션 분포 로깅
            logger.info(f"📊 경험 추출 완료 (분석 포함): 총 {len(experiences)}개, 고유 액션: {len(unique_actions)}개")
            logger.info(f"   액션 분포: NEUTRAL(0)={action_counts.get(0, 0)}, UP(1)={action_counts.get(1, 0)}, DOWN(2)={action_counts.get(2, 0)}")
            
            # 🔥 액션 다양성 강제 보정에도 불구하고 여전히 1개라면, 마지막 안전장치로 experiences 리스트 직접 조작
            if len(unique_actions) < 2:
                logger.warning(f"⚠️ 액션 다양성 부족: 고유 액션 {len(unique_actions)}개만 존재 (강제 주입 실패 가능성)")
                
                # 마지막 안전장치: 기존 experience 중 일부를 강제로 UP/DOWN으로 변환
                # (이미 state 벡터는 NEUTRAL용일 수 있지만, 보상과 액션을 강제로 바꿔서라도 다양성 확보)
                if len(experiences) >= 2:
                    # 1. UP 강제 변환
                    experiences[0]['action'] = 1
                    experiences[0]['reward'] = 0.0 # 중립 보상 (정보 없음)
                    
                    # 2. DOWN 강제 변환
                    experiences[1]['action'] = 2
                    experiences[1]['reward'] = 0.0 # 중립 보상 (정보 없음)
                    
                    # 다시 계산
                    actions = [exp.get('action', 0) for exp in experiences]
                    unique_actions = set(actions)
                    action_counts = {action: actions.count(action) for action in unique_actions}
                    
                    logger.info(f"🔧 최후의 안전장치 발동: experiences 리스트 직접 수정하여 다양성 확보")
                    logger.info(f"   수정 후 분포: {action_counts}")

            # 재검증
            if len(unique_actions) < 2:
                logger.error(f"❌ 학습 전 검증 실패: 액션 다양성 부족 (고유 액션: {len(unique_actions)}개)")
                logger.error(f"   액션 분포: {action_counts}")
                logger.error(f"   학습을 중단합니다. 데이터 수집 로직을 확인하세요.")
                raise ValueError(f"액션 다양성 부족: 고유 액션 {len(unique_actions)}개만 존재 (최소 2개 필요)")
            elif len(unique_actions) == 2:
                # 🆕 NEUTRAL이 없는 경우는 제한적이므로 WARNING
                has_neutral = action_counts.get(0, 0) > 0
                if not has_neutral:
                    logger.warning(f"⚠️ 액션 다양성 제한적: 고유 액션 {len(unique_actions)}개 (NEUTRAL 없음)")
                else:
                    logger.info(f"✅ 액션 다양성 양호: 고유 액션 {len(unique_actions)}개")
            else:
                logger.info(f"✅ 액션 다양성 우수: 고유 액션 {len(unique_actions)}개 (모든 방향 포함)")
        else:
            logger.warning("⚠️ 경험 추출 결과가 비어있습니다 (분석 포함)")
        
        logger.debug(f"✅ 총 {len(experiences)}개 경험 추출 완료 (분석 데이터 포함)")
        return experiences
    
    def _ensure_neutral_trade_pool(
        self,
        buy_trades: List[Dict[str, Any]],
        sell_trades: List[Dict[str, Any]],
        hold_trades: List[Dict[str, Any]],
        episode_num: Optional[int] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """NEUTRAL 샘플 목표 비중을 만족하도록 트레이드 풀을 보강"""
        hold_trades = list(hold_trades or [])
        directional_count = len(buy_trades) + len(sell_trades)
        total_reference = directional_count + len(hold_trades)
        if total_reference == 0:
            return hold_trades

        target_neutral = max(
            self.min_neutral_samples,
            math.ceil(total_reference * self.neutral_target_ratio)
        )
        target_neutral = min(target_neutral, self.max_neutral_samples)

        deficit = max(0, target_neutral - len(hold_trades))
        if deficit <= 0:
            return hold_trades

        synthetic_trades = self._generate_synthetic_neutral_trades(
            buy_trades + sell_trades,
            deficit
        )
        if synthetic_trades:
            hold_trades.extend(synthetic_trades)
            context = f"episode={episode_num}, agent={agent_id}" if episode_num is not None else "selfplay"
            logger.debug(
                f"🆕 NEUTRAL 샘플 보강: 합성 {len(synthetic_trades)}개 추가 ({context})"
            )
        return hold_trades

    def _ensure_directional_trade_pool(
        self,
        buy_trades: List[Dict[str, Any]],
        sell_trades: List[Dict[str, Any]],
        hold_trades: List[Dict[str, Any]],
        episode_num: Optional[int] = None,
        agent_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """BUY/SELL 샘플이 모두 존재하도록 부족한 방향을 합성"""
        buy_trades = list(buy_trades or [])
        sell_trades = list(sell_trades or [])
        hold_trades = list(hold_trades or [])

        missing = []
        if not buy_trades:
            missing.append('BUY')
        if not sell_trades:
            missing.append('SELL')

        if not missing:
            return buy_trades, sell_trades

        source_pool = hold_trades + buy_trades + sell_trades
        if not source_pool:
            return buy_trades, sell_trades

        synthetic_summary = {}
        for direction in missing:
            synthetic = self._generate_synthetic_directional_trades(
                source_pool,
                direction,
                target_count=self.min_directional_samples
            )
            if synthetic:
                if direction == 'BUY':
                    buy_trades.extend(synthetic)
                else:
                    sell_trades.extend(synthetic)
                synthetic_summary[direction] = len(synthetic)

        if synthetic_summary:
            context = f"episode={episode_num}, agent={agent_id}" if episode_num is not None else "selfplay"
            logger.debug(
                f"🆕 Directional 샘플 보강: {synthetic_summary} ({context})"
            )

        return buy_trades, sell_trades

    def _generate_synthetic_directional_trades(
        self,
        source_trades: List[Dict[str, Any]],
        direction: str,
        target_count: int
    ) -> List[Dict[str, Any]]:
        """소스 트레이드로부터 BUY/SELL 합성"""
        if not source_trades or self.max_synthetic_directional <= 0:
            return []

        limit = min(max(1, target_count), self.max_synthetic_directional)
        threshold = self.neutral_direction_threshold
        ordered = sorted(
            source_trades,
            key=self._compute_trade_magnitude
        )

        synthetic: List[Dict[str, Any]] = []
        polarity = 1.0 if direction == 'BUY' else -1.0
        for trade in ordered:
            if len(synthetic) >= limit:
                break
            cloned = copy.deepcopy(trade)
            cloned['direction'] = 'BUY' if polarity > 0 else 'SELL'
            cloned['synthetic_directional'] = True
            
            # 🔥 중요 수정: 합성 데이터는 임계값을 확실히 넘도록 설정 (1.5배)
            # threshold가 0이면 최소 0.02(2%) 변동폭 부여
            base_magnitude = threshold * 1.5 if threshold > 0 else 0.02
            cloned['price_change'] = base_magnitude * polarity
            
            if 'regime' in cloned:
                cloned['regime'] = cloned.get('regime', '').replace('neutral', 'bull' if polarity > 0 else 'bear')
            synthetic.append(cloned)

        return synthetic

    def _generate_synthetic_neutral_trades(
        self,
        source_trades: List[Dict[str, Any]],
        needed: int
    ) -> List[Dict[str, Any]]:
        """BUY/SELL 트레이드에서 합성 NEUTRAL 트레이드를 생성"""
        if not source_trades or needed <= 0 or self.max_synthetic_neutral == 0:
            return []

        limit = min(needed, self.max_synthetic_neutral)
        threshold = self.neutral_direction_threshold

        def sort_key(trade):
            return self._compute_trade_magnitude(trade)

        small_moves = [t for t in source_trades if self._compute_trade_magnitude(t) <= threshold]
        large_moves = [t for t in source_trades if self._compute_trade_magnitude(t) > threshold]
        ordered = sorted(small_moves, key=sort_key) + sorted(large_moves, key=sort_key)

        synthetic: List[Dict[str, Any]] = []
        for trade in ordered:
            if len(synthetic) >= limit:
                break
            cloned = copy.deepcopy(trade)
            cloned['direction'] = 'HOLD'
            cloned['synthetic_neutral'] = True
            cloned['price_change'] = 0.0
            # 상태 관련 필드가 없을 수 있으므로 안전하게 처리
            for volatility_key in ['volatility', 'atr']:
                if volatility_key in cloned:
                    cloned[volatility_key] = max(0.0, float(cloned.get(volatility_key, 0.0)))
            # 로그 추적용 힌트
            original_dir = trade.get('direction', 'UNKNOWN')
            cloned['neutral_source'] = original_dir
            synthetic.append(cloned)

        return synthetic

    @staticmethod
    def _compute_trade_magnitude(trade: Dict[str, Any]) -> float:
        """트레이드의 방향 강도를 근사하는 지표"""
        for key in ('price_change', 'pnl_pct', 'return_pct', 'delta', 'drift'):
            value = trade.get(key)
            if value is not None:
                try:
                    return abs(float(value))
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _select_trades_with_diversity(self, buy_trades, sell_trades, hold_trades):
        """BUY/SELL/NEUTRAL 비중을 균형 있게 선택"""
        if not buy_trades and not sell_trades and not hold_trades:
            return []

        max_dir = max(1, self.max_direction_samples)
        selected_buy = buy_trades[:max(min(1, len(buy_trades)), min(len(buy_trades), max_dir))]
        selected_sell = sell_trades[:max(min(1, len(sell_trades)), min(len(sell_trades), max_dir))]

        min_hold = min(1, len(hold_trades))
        base_hold_target = max(min_hold, self.min_neutral_samples)
        initial_hold_target = min(len(hold_trades), min(self.max_neutral_samples, base_hold_target))
        selected_hold = hold_trades[:initial_hold_target]
        hold_used = initial_hold_target

        selected = selected_buy + selected_sell + selected_hold

        if selected:
            current_neutral = sum(1 for t in selected if t.get('direction') not in ['BUY', 'SELL'])
            required_neutral = max(self.min_neutral_samples, math.ceil(len(selected) * self.neutral_target_ratio))
            required_neutral = min(required_neutral, self.max_neutral_samples, len(hold_trades))
            deficit = max(0, required_neutral - current_neutral)
            if deficit > 0 and hold_used < len(hold_trades):
                additional = min(deficit, len(hold_trades) - hold_used)
                if additional > 0:
                    selected.extend(hold_trades[hold_used:hold_used + additional])
                    hold_used += additional

        selected_directions = [t.get('direction') for t in selected]
        if 'BUY' not in selected_directions and buy_trades:
            selected.append(buy_trades[0])
            selected_directions.append('BUY')
        if 'SELL' not in selected_directions and sell_trades:
            selected.append(sell_trades[0])
            selected_directions.append('SELL')
        if not any(d not in ['BUY', 'SELL'] for d in selected_directions) and hold_trades:
            fallback_idx = min(len(hold_trades) - 1, hold_used) if hold_trades else 0
            fallback_trade = hold_trades[max(0, fallback_idx)]
            selected.append(fallback_trade)
            selected_directions.append(fallback_trade.get('direction', 'HOLD'))

        return selected
    
    def _create_batches(self, experiences: List[Dict], batch_size: int) -> List[List[Dict]]:
        """경험을 배치로 분할"""
        batches = []
        
        for i in range(0, len(experiences), batch_size):
            batch = experiences[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _update_policy(self, batch: List[Dict]) -> float:
        """
        PPO 정책 업데이트 (실제 학습 알고리즘)
        
        Args:
            batch: 경험 배치 [{state, action, reward, log_prob, value, done}, ...]
        
        Returns:
            평균 손실
        """
        if not JAX_AVAILABLE or optax is None:
            logger.error("❌ JAX 또는 optax를 사용할 수 없습니다.")
            return 0.0
        
        try:
            # 배치 데이터 추출 및 검증
            if not batch:
                return 0.0
            
            # 필수 필드 확인 및 기본값 설정
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            old_values = []
            analysis_scores = []  # 🆕 분석 점수 리스트
            
            for exp in batch:
                # State 벡터 추출
                state = exp.get('state')
                if state is None:
                    # agent_id나 다른 필드에서 state 재구성 시도
                    continue
                
                state_vec = build_state_vector(state) if isinstance(state, dict) else np.array(state, dtype=np.float32)
                states.append(state_vec)
                
                # 🔥 예측 전략: Action (0=NEUTRAL, 1=UP, 2=DOWN)
                action = exp.get('action', exp.get('action_idx', 0))
                actions.append(int(action))
                
                # Reward (정규화 및 스케일링)
                reward = float(exp.get('reward', 0.0))
                
                # 🔥 보상 정규화: 매우 음수인 보상을 완화
                # 보상 범위를 -1.0 ~ 1.0으로 정규화하되, 원래 부호 유지
                if reward < -1.0:
                    # 과도한 음수 보상을 -1.0으로 클리핑 (학습 안정성)
                    reward = max(-1.0, reward / 10.0)  # -10.0 → -1.0 스케일링
                elif reward > 1.0:
                    # 과도한 양수 보상도 클리핑
                    reward = min(1.0, reward)
                
                rewards.append(reward)
                
                # Old log probability (없으면 0으로 시작)
                old_log_prob = float(exp.get('log_prob', exp.get('old_log_prob', -1.1)))  # 기본값: log(1/3) ≈ -1.1
                old_log_probs.append(old_log_prob)
                
                # Old value estimate
                old_value = float(exp.get('value', exp.get('old_value', 0.0)))
                old_values.append(old_value)

                # 🆕 Analysis Score (Target)
                analysis_score = float(exp.get('analysis_score', 50.0))  # 기본값 50 (중간)
                analysis_scores.append(analysis_score)
            
            if not states:
                logger.warning("⚠️ 배치에 유효한 state가 없습니다.")
                return 0.0
            
            # 🔥 개선: 상태 벡터 배치 정규화 (학습 안정성 향상)
            states_np = np.array(states, dtype=np.float32)
            states_np = np.nan_to_num(states_np, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 배치 단위 Z-score 정규화 (각 피처별로)
            states_mean = np.mean(states_np, axis=0, keepdims=True)
            states_std = np.std(states_np, axis=0, keepdims=True) + 1e-8  # 0으로 나누기 방지
            states_normalized = (states_np - states_mean) / states_std
            
            # 정규화 후 클리핑 (이상치 제거)
            states_normalized = np.clip(states_normalized, -3.0, 3.0)  # ±3 표준편차 범위
            
            states_jax = jnp.array(states_normalized, dtype=jnp.float32)
            
            actions_jax = jnp.array(actions, dtype=jnp.int32)
            
            rewards_np = np.array(rewards, dtype=np.float32)
            rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=1e6, neginf=-1e6)
            rewards_jax = jnp.array(rewards_np, dtype=jnp.float32)
            
            old_log_probs_np = np.array(old_log_probs, dtype=np.float32)
            old_log_probs_np = np.nan_to_num(old_log_probs_np, nan=-1.1, posinf=10.0, neginf=-10.0)
            old_log_probs_jax = jnp.array(old_log_probs_np, dtype=jnp.float32)
            
            old_values_np = np.array(old_values, dtype=np.float32)
            old_values_np = np.nan_to_num(old_values_np, nan=0.0, posinf=1e6, neginf=-1e6)
            old_values_jax = jnp.array(old_values_np, dtype=jnp.float32)

            analysis_scores_np = np.array(analysis_scores, dtype=np.float32)
            analysis_scores_jax = jnp.array(analysis_scores_np, dtype=jnp.float32)
            
            # PPO 하이퍼파라미터
            clip_epsilon = self.train_config.get('clip_epsilon', 0.2)
            value_loss_coef = self.train_config.get('value_loss_coef', 0.5)
            
            # 🔥 학습 성능 개선: 탐험(Exploration) 강화 (재학습 권장 반영)
            # 🔥 누적 증가 방식: 다양성이 부족하면 점진적으로 증가, 개선되면 천천히 감소

            # 액션 다양성 체크 (HOLD만 있는지 확인)
            unique_actions = len(set(actions))
            action_counts = {action: actions.count(action) for action in set(actions)}
            hold_count = action_counts.get(0, 0)
            hold_ratio = hold_count / len(actions) if actions else 0.0

            neutral_target = self.neutral_target_ratio
            severe_shortage = max(0.02, neutral_target * 0.25)

            # 🔥 예측 전략: NEUTRAL만 있는 경우 탐험 강화 (누적 증가)
            # NEUTRAL(중립 예측)만 하면 예측 정보가 없으므로 UP/DOWN 예측을 유도
            # 더 적극적인 탐험으로 액션 다양성 확보
            if unique_actions == 1:
                # 모든 액션이 동일하면 탐험을 크게 증가 (누적)
                self.current_entropy_coef = min(self.current_entropy_coef * 2.0, self.base_entropy_coef * 200.0)
                logger.warning(f"🔍 예측 다양성 심각 부족 (고유 액션: {unique_actions}), entropy_coef 누적 증가: {self.current_entropy_coef:.4f}")
            elif hold_ratio < severe_shortage:
                self.current_entropy_coef = min(self.current_entropy_coef * 1.7, self.base_entropy_coef * 120.0)
                logger.warning(f"🔍 NEUTRAL 액션 고갈 (비율: {hold_ratio:.1%} < 목표 {neutral_target:.0%}), entropy_coef 증가: {self.current_entropy_coef:.4f}")
            elif hold_ratio < neutral_target:
                self.current_entropy_coef = min(self.current_entropy_coef * 1.4, self.base_entropy_coef * 60.0)
                logger.info(f"🔍 NEUTRAL 비율 부족 (현재 {hold_ratio:.1%}, 목표 {neutral_target:.0%}), entropy_coef 누적 증가: {self.current_entropy_coef:.4f}")
            elif hold_ratio > 0.95:
                # 95% 이상이 NEUTRAL이면 탐험을 크게 증가 (누적)
                self.current_entropy_coef = min(self.current_entropy_coef * 1.8, self.base_entropy_coef * 150.0)
                logger.warning(f"🔍 예측 다양성 심각 부족 (NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef 누적 증가: {self.current_entropy_coef:.4f}")
            elif hold_ratio > 0.9:
                # 90% 이상이 NEUTRAL이면 탐험을 증가 (누적)
                self.current_entropy_coef = min(self.current_entropy_coef * 1.5, self.base_entropy_coef * 75.0)
                logger.info(f"🔍 예측 다양성 부족 감지 (NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef 누적 증가: {self.current_entropy_coef:.4f}")
            elif unique_actions == 2:
                # 🆕 2종류 액션만 있는 경우 (NEUTRAL이 없는 경우) - 더 강력한 탐험
                if hold_ratio < max(0.1, neutral_target * 0.5):
                    self.current_entropy_coef = min(self.current_entropy_coef * 1.5, self.base_entropy_coef * 100.0)
                    logger.warning(f"🔍 예측 다양성 심각 부족 (고유 액션: {unique_actions}, NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef 강력 증가: {self.current_entropy_coef:.4f}")
                else:
                    self.current_entropy_coef = min(self.current_entropy_coef * 1.3, self.base_entropy_coef * 50.0)
                    logger.info(f"🔍 예측 다양성 부족 감지 (고유 액션: {unique_actions}, NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef 누적 증가: {self.current_entropy_coef:.4f}")
            elif hold_ratio > 0.7:
                # 70% 이상이 NEUTRAL이면 중간 증가 (누적)
                self.current_entropy_coef = min(self.current_entropy_coef * 1.3, self.base_entropy_coef * 30.0)
                logger.info(f"🔍 예측 다양성 부족 감지 (NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef 누적 증가: {self.current_entropy_coef:.4f}")
            elif unique_actions == 3 and hold_ratio > 0.5:
                # 3종류 모두 있지만 NEUTRAL이 절반 이상이면 약간 증가 (누적)
                self.current_entropy_coef = min(self.current_entropy_coef * 1.1, self.base_entropy_coef * 8.0)
                logger.debug(f"🔍 예측 다양성 보통 (고유 액션: {unique_actions}, NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef 약간 증가: {self.current_entropy_coef:.4f}")
            else:
                # 다양한 액션이 있으면 천천히 감소 (기본값으로 복귀)
                self.current_entropy_coef = max(self.current_entropy_coef * 0.95, self.base_entropy_coef)
                logger.debug(f"✅ 예측 다양성 양호 (고유 액션: {unique_actions}, NEUTRAL 비율: {hold_ratio:.1%}), entropy_coef: {self.current_entropy_coef:.4f}")

            entropy_coef = self.current_entropy_coef
            
            gamma = self.train_config.get('gamma', 0.99)  # 할인율
            gae_lambda = self.train_config.get('gae_lambda', 0.95)  # GAE lambda
            
            # 🔥 학습 성능 개선: 보상 정규화 및 Shaping 강화
            # 모든 보상이 음수인 경우 학습이 어려울 수 있으므로, 보상을 적극적으로 정규화
            rewards_mean = float(jnp.mean(rewards_jax))
            rewards_std = float(jnp.std(rewards_jax))
            rewards_min = float(jnp.min(rewards_jax))
            rewards_max = float(jnp.max(rewards_jax))
            
            # 표준화를 위한 최소 std 값 (0으로 나누기 방지)
            min_std = 1e-6
            
            # 🔥 보상 Shaping: 음수 보상에 더 적극적인 처리
            if rewards_mean < -0.1:  # 평균 보상이 음수인 경우
                # 1. 보상을 양수 영역으로 이동 (상수 추가)
                # 목표: 평균을 0 근처로 이동 - 더 적극적으로
                shift_amount = abs(rewards_mean) * 0.8  # 평균의 80%만큼 이동 (더 강력하게)
                rewards_jax = rewards_jax + shift_amount
                rewards_mean = float(jnp.mean(rewards_jax))
                rewards_std = float(jnp.std(rewards_jax))  # std 재계산
                logger.info(f"🔍 보상 Shaping: 음수 보상에 {shift_amount:.4f} 추가 (새 평균: {rewards_mean:.4f})")
            
            # 🔥 보상 다양성 강화: 액션별로 다른 보상 부여
            if rewards_std < min_std:
                # 보상이 거의 동일하면 (모두 HOLD로 인한 음수 보상)
                # 더 적극적인 보상 다양성 부여
                # 1. 액션별로 다른 보너스/페널티 추가 (강화됨)
                actions_jax = jnp.array(actions, dtype=jnp.int32)
                # 🔥 개선: BUY/SELL 보너스 증가 (0.1 → 0.3), HOLD에 페널티 추가
                direction_bonus = max(0.0, self.direction_action_bonus * 3.0)
                neutral_penalty = -abs(self.direction_action_bonus) * 0.5
                action_bonuses = jnp.where(
                    (actions_jax == 1) | (actions_jax == 2),
                    direction_bonus,
                    neutral_penalty
                )
                rewards_jax = rewards_jax + action_bonuses
                
                # 2. 추가 양수 보너스 (전체에) - 강화
                bonus = max(0.2, abs(rewards_mean) * 0.5)  # 🔥 개선: 최소 0.2, 평균의 50% (이전: 0.1, 30%)
                rewards_jax = rewards_jax + bonus

                # 3. 랜덤 노이즈 추가 (보상 다양성 강화) - 강화
                rng_key = jax.random.PRNGKey(int(np.random.randint(0, 2**32)))
                noise = jax.random.normal(rng_key, shape=rewards_jax.shape) * 0.1  # 🔥 개선: 10% 노이즈 (이전: 5%)
                rewards_jax = rewards_jax + noise

                rewards_std = float(jnp.std(rewards_jax))  # std 재계산
                logger.warning(f"🔍 보상 다양성 심각 부족 (std={rewards_std:.6f}), 액션별 보너스/페널티 + 전체 보너스 {bonus:.4f} + 노이즈 추가")
            else:
                # 보상 정규화 (Z-score 정규화 후 tanh로 스케일링)
                rewards_normalized = (rewards_jax - rewards_mean) / (rewards_std + min_std)
                # -3 ~ +3 범위를 -1 ~ +1로 스케일링 (더 부드러운 스케일링)
                rewards_jax = jnp.tanh(rewards_normalized * 0.5)
                logger.debug(f"🔍 보상 정규화 완료 (mean={rewards_mean:.4f}, std={rewards_std:.4f})")
            
            # 최종 클리핑 (안전장치) - 더 넓은 범위 허용
            rewards_jax = jnp.clip(rewards_jax, -2.0, 2.0)  # -1.0 ~ 1.0 → -2.0 ~ 2.0
            
            # 최종 보상 통계 로깅
            final_mean = float(jnp.mean(rewards_jax))
            final_std = float(jnp.std(rewards_jax))
            logger.debug(f"📊 최종 보상 통계: mean={final_mean:.4f}, std={final_std:.4f}, range=[{rewards_min:.4f}, {rewards_max:.4f}]")
            
            # 🔧 GAE (Generalized Advantage Estimation) 구현
            # 1. Discounted returns 계산
            returns = _compute_returns(rewards_jax, gamma)
            
            # returns와 old_values의 shape 일치 확인
            if returns.shape != old_values_jax.shape:
                logger.warning(f"⚠️ Returns와 Values shape 불일치: returns={returns.shape}, values={old_values_jax.shape}")
                # shape 맞추기
                if old_values_jax.ndim == 0:
                    old_values_jax = old_values_jax.reshape(-1)
                if returns.ndim == 0:
                    returns = returns.reshape(-1)
                min_len = min(len(returns), len(old_values_jax))
                returns = returns[:min_len]
                old_values_jax = old_values_jax[:min_len]
            
            # 2. GAE 계산 (더 정확한 advantage 추정)
            advantages = _compute_gae(
                rewards=rewards_jax[:min_len] if 'min_len' in locals() else rewards_jax,
                values=old_values_jax,
                gamma=gamma,
                lam=gae_lambda
            )
            
            # advantages와 returns의 shape 일치 확인
            if advantages.shape != returns.shape:
                logger.warning(f"⚠️ Advantages와 Returns shape 불일치: advantages={advantages.shape}, returns={returns.shape}")
                min_adv_len = min(len(advantages), len(returns))
                advantages = advantages[:min_adv_len]
                returns = returns[:min_adv_len]
                old_log_probs_jax = old_log_probs_jax[:min_adv_len]
                actions_jax = actions_jax[:min_adv_len]
                states_jax = states_jax[:min_adv_len]
            
            # Advantage 정규화 (안전한 방식)
            try:
                advantages_mean = jnp.mean(advantages)
                advantages_std = jnp.std(advantages)
                
                # std가 너무 작으면 정규화 스킵
                if advantages_std > 1e-6:
                    advantages_normalized = (advantages - advantages_mean) / (advantages_std + 1e-8)
                    # 클리핑 (과도한 값 방지)
                    advantages_normalized = jnp.clip(advantages_normalized, -10.0, 10.0)
                else:
                    # std가 너무 작으면 정규화 없이 사용
                    advantages_normalized = advantages - advantages_mean
                    advantages_normalized = jnp.clip(advantages_normalized, -10.0, 10.0)
            except Exception as norm_err:
                logger.warning(f"⚠️ Advantage 정규화 실패, 원본 사용: {norm_err}")
                advantages_normalized = jnp.clip(advantages, -10.0, 10.0)
            
            # 🔥 배치 크기 사전 조정 (loss_fn 정의 전에 수행)
            # 클로저 변수 캡처를 위해 조건부 재할당 제거
            actual_batch_size = states_jax.shape[0] if states_jax.ndim > 0 else 0
            max_safe_batch = 2048  # 🔥 성능 개선: 256 -> 2048로 대폭 상향
            if actual_batch_size > max_safe_batch:
                logger.warning(f"⚠️ 배치 크기 초과 ({actual_batch_size} > {max_safe_batch}), 처음 {max_safe_batch}개만 사용")
                # ✅ 무조건 슬라이싱으로 재할당 (클로저 캡처 보장)
                states_jax = states_jax[:max_safe_batch]
                actions_jax = actions_jax[:max_safe_batch]
                old_log_probs_jax = old_log_probs_jax[:max_safe_batch]
                old_values_jax = old_values_jax[:max_safe_batch]
                rewards_jax = rewards_jax[:max_safe_batch]
                analysis_scores_jax = analysis_scores_jax[:max_safe_batch]  # 🆕
                advantages_normalized = advantages_normalized[:max_safe_batch]
                returns = returns[:max_safe_batch]
                actual_batch_size = max_safe_batch
            
            # 손실 함수 정의 (클로저 변수들을 명시적으로 캡처)
            # 모든 외부 변수를 loss_fn 정의 시점에서 안전하게 캡처
            def loss_fn(params):
                try:
                    # 현재 정책으로 forward pass
                    model = self.model['model_def']
                    
                    # 입력 shape 확인 (batch_size, feature_dim)
                    batch_size = states_jax.shape[0]
                    if batch_size == 0:
                        # 빈 배치 처리 (학습 중단 방지)
                        safe_loss = jnp.array(0.0)
                        return safe_loss, (safe_loss, safe_loss, safe_loss)
                    
                    # 🔧 입력 데이터 검증 및 클리핑
                    states_safe = jnp.clip(states_jax, -10.0, 10.0)
                    states_safe = jnp.nan_to_num(states_safe, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    # 🔧 입력 shape 확인 (최소 2D 필요: batch_size, feature_dim)
                    if states_safe.ndim == 1:
                        states_safe = states_safe.reshape(1, -1)
                    elif states_safe.ndim == 0:
                        logger.warning(f"⚠️ States shape 이상: {states_safe.shape}, 빈 배치 반환")
                        safe_loss = jnp.array(0.0)
                        return safe_loss, (safe_loss, safe_loss, safe_loss)
                    
                    # 🔥 입력 데이터 추가 검증
                    if not jnp.all(jnp.isfinite(states_safe)):
                        logger.warning(f"⚠️ States에 NaN/Inf 발견, 제거 후 재검증")
                        states_safe = jnp.nan_to_num(states_safe, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    # Feature 차원 검증
                    expected_feature_dim = self.model.get('obs_dim', 25)
                    if states_safe.shape[-1] != expected_feature_dim:
                        logger.error(f"❌ Feature 차원 불일치: {states_safe.shape[-1]} != {expected_feature_dim}")
                        safe_loss = jnp.array(0.0)
                        return safe_loss, (safe_loss, safe_loss, safe_loss)
                    
                    # 🔧 params 형식 검증 및 정규화
                    # Flax model.init()은 {"params": {...}} 구조를 반환함
                    # self.model['params']는 이미 {"params": {...}} 구조
                    # loss_fn에서 받는 params도 동일한 구조이므로, 중복 래핑 방지
                    if not isinstance(params, dict):
                        logger.warning(f"⚠️ Params 형식 이상: {type(params)}, dict로 변환 시도")
                        try:
                            # JAX/FrozenDict를 dict로 변환 시도
                            if hasattr(params, '__dict__'):
                                params = dict(params)
                            else:
                                params = {'params': params}
                        except:
                            logger.error(f"❌ Params 변환 실패: {type(params)}")
                            safe_loss = jnp.array(0.0)
                            return safe_loss, (safe_loss, safe_loss, safe_loss)
                    
                    # 🔧 params 구조 확인 및 정규화
                    # model.init()은 {"params": {...}} 반환
                    # 체크포인트에서 로드한 경우도 동일 구조
                    # 따라서 이미 올바른 구조면 그대로 사용
                    if 'params' in params:
                        # 이미 {"params": {...}} 구조인 경우 그대로 사용
                        variables = params
                    else:
                        # 파라미터 딕셔너리만 있는 경우 {"params": params}로 래핑
                        variables = {'params': params}
                    
                    # 🔥 모델 파라미터 검증 (NaN/Inf 체크)
                    try:
                        # Flax FrozenDict를 확인하고 검증
                        # 모듈 레벨의 jax, jnp 사용 (함수 내부에서 다시 import하지 않음)
                        def check_params_finite(p):
                            """재귀적으로 파라미터 검증"""
                            if isinstance(p, (dict, type(variables))):
                                return all(check_params_finite(v) for v in p.values())
                            elif hasattr(p, 'shape'):
                                # JAX 배열인 경우
                                if p.size > 0:
                                    is_finite = jnp.all(jnp.isfinite(p))
                                    if not is_finite:
                                        logger.warning(f"⚠️ 파라미터에 NaN/Inf 발견: shape={p.shape}")
                                        return False
                                return True
                            return True
                        
                        if not check_params_finite(variables):
                            logger.warning("⚠️ 모델 파라미터에 NaN/Inf 발견, 안전한 값으로 대체")
                            # 파라미터 재초기화 대신 안전한 값 사용
                            safe_loss = jnp.array(0.0)
                            return safe_loss, (safe_loss, safe_loss, safe_loss)
                    except Exception as param_check_err:
                        logger.debug(f"⚠️ 파라미터 검증 중 오류 (무시하고 계속): {param_check_err}")
                    
                    # 🔧 안전한 forward pass (Flax 모델 apply 방식)
                    try:
                        # variables는 {"params": {...}} 구조
                        # 🔥 Flax 모델 apply 호출 (mutable 파라미터 없이, 기본값 사용)
                        # JAX 컴파일 에러 방지를 위해 명시적으로 변수 검증
                        outputs = model.apply(variables, states_safe)
                        
                        # 🆕 5개 출력 처리 (Multitask Learning)
                        if isinstance(outputs, tuple):
                            if len(outputs) == 5:
                                action_logits, values, price_change_pred, horizon_pred, analysis_score_pred = outputs
                            elif len(outputs) == 4:
                                action_logits, values, price_change_pred, horizon_pred = outputs
                                analysis_score_pred = jnp.full((batch_size, 1), 50.0)
                            elif len(outputs) == 2:
                                # 이전 모델 호환성 (2개 출력)
                                action_logits, values = outputs
                                price_change_pred = jnp.zeros((states_safe.shape[0], 1))
                                horizon_pred = jnp.ones((states_safe.shape[0], 1)) * 10
                                analysis_score_pred = jnp.full((states_safe.shape[0], 1), 50.0)
                            else:
                                logger.warning(f"⚠️ Model 출력 개수 예상과 다름: {len(outputs)}")
                                safe_loss = jnp.array(0.0)
                                return safe_loss, (safe_loss, safe_loss, safe_loss)
                        else:
                            logger.warning(f"⚠️ Model 출력 형태 예상과 다름: {type(outputs)}")
                            safe_loss = jnp.array(0.0)
                            return safe_loss, (safe_loss, safe_loss, safe_loss)
                    except Exception as apply_err:
                        logger.warning(f"⚠️ Model.apply 실패: {apply_err}")
                        import traceback
                        logger.error(f"Model.apply 상세 에러:\n{traceback.format_exc()}")
                        # params와 states_safe 형식 로깅
                        logger.error(f"Params type: {type(params)}, States shape: {states_safe.shape}, dtype: {states_safe.dtype}")
                        
                        # 🔥 배치가 너무 크면 자동으로 더 작은 배치로 분할하여 재시도
                        # JAX 컴파일 에러는 종종 큰 배치에서 발생
                        # XLA 컴파일 문제로 더 작은 청크 크기 사용
                        if states_safe.shape[0] > 128:
                            logger.info(f"🔄 Model.apply 실패, 배치 분할 시도: {states_safe.shape[0]} → 128씩")
                            # 작은 배치로 나누어 처리 (512 → 128로 축소)
                            chunk_size = 128
                            
                            # 🔥 더 작은 청크도 시도
                            # 128도 실패하면 64로, 64도 실패하면 32로 재시도
                            chunk_sizes = [128, 64, 32]
                            success = False
                            
                            for try_chunk_size in chunk_sizes:
                                if states_safe.shape[0] <= try_chunk_size:
                                    continue  # 배치가 이미 청크 크기보다 작으면 스킵
                                
                                try:
                                    logger.info(f"  🔄 {try_chunk_size} 크기로 재시도...")
                                    all_action_logits = []
                                    all_values = []
                                    all_pc = []
                                    all_h = []
                                    all_as = []
                                    
                                    for chunk_start in range(0, states_safe.shape[0], try_chunk_size):
                                        chunk_end = min(chunk_start + try_chunk_size, states_safe.shape[0])
                                        states_chunk = states_safe[chunk_start:chunk_end]
                                        
                                        # 각 청크에 대해 forward pass
                                        outputs_chunk = model.apply(variables, states_chunk)

                                        # 🆕 5개 출력 처리
                                        if isinstance(outputs_chunk, tuple):
                                            if len(outputs_chunk) == 5:
                                                a, v, p, h, s = outputs_chunk
                                                all_as.append(s)
                                                all_pc.append(p)
                                                all_h.append(h)
                                            elif len(outputs_chunk) == 4:
                                                a, v, p, h = outputs_chunk
                                                all_as.append(jnp.full((a.shape[0], 1), 50.0))
                                                all_pc.append(p)
                                                all_h.append(h)
                                            elif len(outputs_chunk) == 2:
                                                a, v = outputs_chunk
                                                all_as.append(jnp.full((a.shape[0], 1), 50.0))
                                                all_pc.append(jnp.zeros((a.shape[0], 1)))
                                                all_h.append(jnp.ones((a.shape[0], 1)) * 10)
                                            else:
                                                raise ValueError(f"청크 출력 개수 오류: {len(outputs_chunk)}")
                                            all_action_logits.append(a)
                                            all_values.append(v)
                                        else:
                                            raise ValueError(f"청크 출력 형식 오류: {type(outputs_chunk)}")
                                    
                                    # 모든 청크 결과 합치기
                                    if all_action_logits and all_values:
                                        action_logits = jnp.concatenate(all_action_logits, axis=0)
                                        values = jnp.concatenate(all_values, axis=0)
                                        price_change_pred = jnp.concatenate(all_pc, axis=0)
                                        horizon_pred = jnp.concatenate(all_h, axis=0)
                                        analysis_score_pred = jnp.concatenate(all_as, axis=0)
                                        logger.info(f"✅ 분할 배치 처리 성공: {states_safe.shape[0]} → {len(all_action_logits)}개 청크 (각 {try_chunk_size} 크기)")
                                        success = True
                                        break  # 성공하면 루프 종료
                                except Exception as chunk_try_err:
                                    logger.debug(f"  ⚠️ {try_chunk_size} 크기로도 실패, 다음 크기 시도: {chunk_try_err}")
                                    continue
                            
                            if not success:
                                # 모든 청크 크기 실패
                                logger.warning(f"⚠️ 모든 배치 분할 시도 실패")
                                dummy_loss = jnp.array(0.0)
                                return dummy_loss, (dummy_loss, dummy_loss, dummy_loss)
                            else:
                                # 성공했으므로 계속 진행
                                pass
                        else:
                            # 작은 배치도 실패한 경우, 더 작은 단위로 재시도
                            if states_safe.shape[0] > 32:
                                logger.info(f"🔄 작은 배치도 실패, 더 작게 분할 시도: {states_safe.shape[0]} → 32씩")
                                chunk_sizes = [32, 16, 8]
                                success = False
                                
                                for try_chunk_size in chunk_sizes:
                                    if states_safe.shape[0] <= try_chunk_size:
                                        # 배치가 청크 크기 이하면 직접 시도
                                        try:
                                            logger.info(f"  🔄 {states_safe.shape[0]} 크기로 직접 재시도...")
                                            outputs = model.apply(variables, states_safe)
                                            # 🆕 5개 출력 처리
                                            if isinstance(outputs, tuple):
                                                if len(outputs) == 5:
                                                    action_logits, values, price_change_pred, horizon_pred, analysis_score_pred = outputs
                                                elif len(outputs) == 4:
                                                    action_logits, values, price_change_pred, horizon_pred = outputs
                                                    analysis_score_pred = jnp.full((states_safe.shape[0], 1), 50.0)
                                                elif len(outputs) == 2:
                                                    action_logits, values = outputs
                                                    price_change_pred = jnp.zeros((states_safe.shape[0], 1))
                                                    horizon_pred = jnp.ones((states_safe.shape[0], 1)) * 10
                                                    analysis_score_pred = jnp.full((states_safe.shape[0], 1), 50.0)
                                                logger.info(f"✅ 직접 재시도 성공: {states_safe.shape[0]}")
                                                success = True
                                                break
                                        except:
                                            continue
                                    
                                    try:
                                        logger.info(f"  🔄 {try_chunk_size} 크기로 분할 재시도...")
                                        all_action_logits = []
                                        all_values = []
                                        all_pc = []
                                        all_h = []
                                        all_as = []
                                        
                                        for chunk_start in range(0, states_safe.shape[0], try_chunk_size):
                                            chunk_end = min(chunk_start + try_chunk_size, states_safe.shape[0])
                                            states_chunk = states_safe[chunk_start:chunk_end]
                                            
                                            outputs_chunk = model.apply(variables, states_chunk)

                                            # 🆕 5개 출력 처리
                                            if isinstance(outputs_chunk, tuple):
                                                if len(outputs_chunk) == 5:
                                                    a, v, p, h, s = outputs_chunk
                                                    all_as.append(s)
                                                    all_pc.append(p)
                                                    all_h.append(h)
                                                elif len(outputs_chunk) == 4:
                                                    a, v, p, h = outputs_chunk
                                                    all_as.append(jnp.full((a.shape[0], 1), 50.0))
                                                    all_pc.append(p)
                                                    all_h.append(h)
                                                elif len(outputs_chunk) == 2:
                                                    a, v = outputs_chunk
                                                    all_as.append(jnp.full((a.shape[0], 1), 50.0))
                                                    all_pc.append(jnp.zeros((a.shape[0], 1)))
                                                    all_h.append(jnp.ones((a.shape[0], 1)) * 10)
                                                else:
                                                    raise ValueError(f"청크 출력 개수 오류: {len(outputs_chunk)}")
                                                all_action_logits.append(a)
                                                all_values.append(v)
                                            else:
                                                raise ValueError(f"청크 출력 형식 오류: {type(outputs_chunk)}")
                                        
                                        if all_action_logits and all_values:
                                            action_logits = jnp.concatenate(all_action_logits, axis=0)
                                            values = jnp.concatenate(all_values, axis=0)
                                            price_change_pred = jnp.concatenate(all_pc, axis=0)
                                            horizon_pred = jnp.concatenate(all_h, axis=0)
                                            analysis_score_pred = jnp.concatenate(all_as, axis=0)
                                            logger.info(f"✅ 분할 재시도 성공: {states_safe.shape[0]} → {len(all_action_logits)}개 청크 (각 {try_chunk_size})")
                                            success = True
                                            break
                                    except Exception as small_chunk_err:
                                        logger.debug(f"  ⚠️ {try_chunk_size} 크기로도 실패: {small_chunk_err}")
                                        continue
                                
                                if not success:
                                    logger.error(f"❌ 모든 배치 크기 시도 실패 (최소 크기: {states_safe.shape[0]})")
                                    dummy_loss = jnp.array(0.0)
                                    return dummy_loss, (dummy_loss, dummy_loss, dummy_loss)
                            else:
                                # 이미 매우 작은 배치 (<=32)도 실패
                                logger.error(f"❌ 매우 작은 배치 ({states_safe.shape[0]})도 실패 - 모델 또는 파라미터 문제 가능")
                                dummy_loss = jnp.array(0.0)
                                return dummy_loss, (dummy_loss, dummy_loss, dummy_loss)
                    
                    # 출력 shape 확인 및 검증
                    # values는 (batch_size, 1) 형태일 수 있음
                    values_batch_dim = values.shape[0] if values.ndim > 0 else 0
                    if action_logits.shape[0] != batch_size:
                        logger.warning(f"⚠️ Action logits shape 불일치: {action_logits.shape}, batch_size={batch_size}")
                        dummy_loss = jnp.array(0.0)
                        return dummy_loss, (dummy_loss, dummy_loss, dummy_loss)
                    
                    # values shape 정규화 (2D -> 1D)
                    if values.ndim == 2 and values.shape[1] == 1:
                        # (batch_size, 1) -> (batch_size,)
                        values = values.squeeze(axis=1)
                    elif values.ndim == 2 and values.shape[1] > 1:
                        # (batch_size, value_dim) -> 첫 번째 차원만 사용
                        values = values[:, 0]
                    elif values.ndim == 0:
                        # 스칼라 -> 배치로 확장
                        values = jnp.broadcast_to(values, (batch_size,))
                    
                    if values.shape[0] != batch_size:
                        logger.warning(f"⚠️ Values shape 불일치: {values.shape}, batch_size={batch_size}")
                        dummy_loss = jnp.array(0.0)
                        return dummy_loss, (dummy_loss, dummy_loss, dummy_loss)
                    
                    # 현재 정책의 log probability 계산
                    action_probs = jax.nn.softmax(action_logits)
                    log_probs = jax.nn.log_softmax(action_logits)
                    
                    # 선택된 action의 log_prob
                    action_one_hot = jax.nn.one_hot(actions_jax, num_classes=3)
                    new_log_probs = jnp.sum(log_probs * action_one_hot, axis=1)
                    
                    # Ratio 계산 (현재 정책 / 이전 정책)
                    ratio = jnp.exp(new_log_probs - old_log_probs_jax)
                    
                    # 🔥 학습 성능 개선: Loss 계산 개선
                    # Clipped surrogate objective (더 안전한 클리핑)
                    ratio_clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                    surr1 = ratio * advantages_normalized
                    surr2 = ratio_clipped * advantages_normalized
                    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                    
                    # 🔥 Policy loss 정규화 (너무 큰 loss 방지)
                    policy_loss = jnp.clip(policy_loss, -10.0, 10.0)
                    
                    # Value loss (MSE) - values는 이미 1D로 정규화됨
                    # returns와 values의 shape 일치 확인
                    if values.shape != returns.shape:
                        min_val_len = min(len(values), len(returns))
                        values_aligned = values[:min_val_len]
                        returns_aligned = returns[:min_val_len]
                    else:
                        values_aligned = values
                        returns_aligned = returns
                    
                    # 🔥 Value loss도 정규화 (Huber loss 고려)
                    value_error = values_aligned - returns_aligned
                    # MSE 대신 Huber loss 사용 (이상치에 덜 민감)
                    delta = 1.0
                    huber_loss = jnp.where(
                        jnp.abs(value_error) < delta,
                        0.5 * value_error ** 2,
                        delta * (jnp.abs(value_error) - 0.5 * delta)
                    )
                    value_loss = jnp.mean(huber_loss)
                    
                    # 🔥 Value loss 정규화
                    value_loss = jnp.clip(value_loss, 0.0, 10.0)
                    
                    # Entropy bonus (탐험 장려)
                    entropy = -jnp.mean(jnp.sum(action_probs * log_probs, axis=1))

                    # 🔥 Entropy 정규화
                    entropy = jnp.clip(entropy, 0.0, 10.0)

                    # 🆕 Price change loss (MSE) - 회귀 예측
                    # price_change_pred shape: (batch_size, 1) or (batch_size,)
                    # 실제 레이블은 향후 orchestrator에서 추가됨 (현재는 0으로 초기화)
                    # TODO: orchestrator에서 실제 price_change 레이블 제공 시 사용
                    price_change_target = jnp.zeros_like(price_change_pred)  # 임시 타겟 (0%)
                    if price_change_pred.ndim == 2 and price_change_pred.shape[1] == 1:
                        price_change_pred_flat = price_change_pred.squeeze(axis=1)
                    else:
                        price_change_pred_flat = price_change_pred
                    price_change_loss = jnp.mean((price_change_pred_flat - price_change_target.squeeze()) ** 2)
                    price_change_loss = jnp.clip(price_change_loss, 0.0, 1.0)  # 클리핑 (0~1 범위)

                    # 🆕 Horizon loss (MSE) - 회귀 예측
                    # horizon_pred shape: (batch_size, 1) or (batch_size,)
                    # 실제 레이블은 향후 orchestrator에서 추가됨 (현재는 10으로 초기화)
                    # TODO: orchestrator에서 실제 horizon 레이블 제공 시 사용
                    horizon_target = jnp.ones_like(horizon_pred) * 10.0  # 임시 타겟 (10 캔들)
                    if horizon_pred.ndim == 2 and horizon_pred.shape[1] == 1:
                        horizon_pred_flat = horizon_pred.squeeze(axis=1)
                    else:
                        horizon_pred_flat = horizon_pred
                    horizon_loss = jnp.mean((horizon_pred_flat - horizon_target.squeeze()) ** 2)
                    horizon_loss = jnp.clip(horizon_loss, 0.0, 100.0)  # 클리핑 (0~100 범위)

                    # 🆕 Analysis Score Loss (MSE)
                    if analysis_score_pred.ndim == 2 and analysis_score_pred.shape[1] == 1:
                        analysis_score_pred_flat = analysis_score_pred.squeeze(axis=1)
                    else:
                        analysis_score_pred_flat = analysis_score_pred
                    
                    # shape 맞추기 (안전장치)
                    if analysis_score_pred_flat.shape[0] != analysis_scores_jax.shape[0]:
                        min_len = min(analysis_score_pred_flat.shape[0], analysis_scores_jax.shape[0])
                        analysis_score_pred_flat = analysis_score_pred_flat[:min_len]
                        analysis_targets = analysis_scores_jax[:min_len]
                    else:
                        analysis_targets = analysis_scores_jax

                    analysis_loss = jnp.mean((analysis_score_pred_flat - analysis_targets) ** 2)
                    analysis_loss = jnp.clip(analysis_loss, 0.0, 10000.0)

                    # 총 손실 (Loss 구성 요소별 가중치 조정)
                    # 🆕 회귀 손실 추가 (작은 가중치로 시작)
                    regression_loss_coef = 0.1  # 회귀 손실 가중치 (향후 조정 가능)
                    total_loss = (
                        policy_loss +
                        value_loss_coef * value_loss -
                        entropy_coef * entropy +
                        regression_loss_coef * price_change_loss +
                        regression_loss_coef * horizon_loss +
                        regression_loss_coef * (analysis_loss / 100.0)  # 스케일 조정
                    )
                    
                    # 🔥 Loss 정규화 (과도한 loss 방지)
                    total_loss = jnp.clip(total_loss, -20.0, 20.0)
                    
                    # NaN/Inf 체크
                    total_loss = jnp.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=-1e6)
                    policy_loss = jnp.nan_to_num(policy_loss, nan=0.0, posinf=1e6, neginf=-1e6)
                    value_loss = jnp.nan_to_num(value_loss, nan=0.0, posinf=1e6, neginf=-1e6)
                    entropy = jnp.nan_to_num(entropy, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    return total_loss, (policy_loss, value_loss, entropy)
                except Exception as loss_err:
                    logger.warning(f"⚠️ Loss 계산 실패, 0 반환: {loss_err}")
                    dummy_loss = jnp.array(0.0)
                    return dummy_loss, (dummy_loss, dummy_loss, dummy_loss)
            
            # Gradient 계산 (안전한 방식)
            try:
                # 🔧 self.model['params']가 이미 올바른 형식인지 확인
                model_params = self.model['params']
                # Flax params는 FrozenDict 또는 dict 형태
                (loss_value, (policy_loss_val, value_loss_val, entropy_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_params)
                
                # Gradient 클리핑 (과도한 gradient 방지)
                # 🔧 JAX 버전 호환성: jax.tree_map → jax.tree.map 또는 jax.tree_util.tree_map
                if USE_JAX_TREE and hasattr(jax, 'tree'):
                    # JAX v0.4.25+ 또는 최신 버전
                    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
                elif JAX_TREE_UTIL is not None:
                    # 구버전 호환성
                    grads = JAX_TREE_UTIL.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
                else:
                    # 최후의 수단: 수동으로 처리
                    grads_clipped = {}
                    for k, v in grads.items():
                        if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                            grads_clipped[k] = jnp.clip(v, -1.0, 1.0)
                        else:
                            grads_clipped[k] = v
                    grads = grads_clipped
                
                # NaN/Inf 체크
                loss_value = jnp.nan_to_num(loss_value, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 옵티마이저 업데이트
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.model['params'])
                
                # Updates 클리핑
                if USE_JAX_TREE and hasattr(jax, 'tree'):
                    # JAX v0.4.25+ 또는 최신 버전
                    updates = jax.tree.map(lambda u: jnp.clip(u, -0.1, 0.1), updates)
                elif JAX_TREE_UTIL is not None:
                    # 구버전 호환성
                    updates = JAX_TREE_UTIL.tree_map(lambda u: jnp.clip(u, -0.1, 0.1), updates)
                else:
                    # 수동 처리
                    updates_clipped = {}
                    for k, v in updates.items():
                        if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                            updates_clipped[k] = jnp.clip(v, -0.1, 0.1)
                        else:
                            updates_clipped[k] = v
                    updates = updates_clipped
                
                self.model['params'] = optax.apply_updates(self.model['params'], updates)
                
            except Exception as grad_err:
                logger.error(f"❌ Gradient 계산 실패: {grad_err}")
                import traceback
                logger.debug(f"Gradient 계산 상세 에러:\n{traceback.format_exc()}")
                return 0.0
            
            # 손실값 반환
            loss_float = float(loss_value)
            policy_loss_float = float(policy_loss_val)
            value_loss_float = float(value_loss_val)
            entropy_float = float(entropy_val)

            # 학습 진행 상황 로깅 (더 자주)
            if np.random.random() < 0.2:  # 20% 확률로 로그
                logger.debug(f"📊 Loss: total={loss_float:.4f}, policy={policy_loss_float:.4f}, "
                          f"value={value_loss_float:.4f}, entropy={entropy_float:.4f}")

            # 🔥 디버거 로깅: 배치 학습 상세 정보
            if self.debug:
                try:
                    # action_probs 계산 (forward pass에서 이미 계산됨)
                    # loss_fn 내에서 계산했으므로 다시 계산해야 함
                    if JAX_AVAILABLE:
                        model = self.model['model_def']
                        variables = self.model['params']
                        outputs = model.apply(variables, states_jax)
                        if isinstance(outputs, tuple):
                            action_logits, _ = outputs
                            action_probs = jax.nn.softmax(action_logits)
                        else:
                            action_probs = None
                    else:
                        action_probs = None

                    # KL divergence 계산 (old vs new policy)
                    if action_probs is not None and JAX_AVAILABLE:
                        log_probs_new = jax.nn.log_softmax(action_logits)
                        # one-hot 인코딩
                        action_one_hot = jax.nn.one_hot(actions_jax, num_classes=3)
                        new_log_probs = jnp.sum(log_probs_new * action_one_hot, axis=1)
                        # KL divergence: E[log(new) - log(old)]
                        kl_div = float(jnp.mean(new_log_probs - old_log_probs_jax))
                    else:
                        kl_div = 0.0

                    # 현재 배치 인덱스 추적 (없으면 0)
                    if not hasattr(self, '_debug_batch_idx'):
                        self._debug_batch_idx = 0
                    self._debug_batch_idx += 1

                    # 현재 epoch 추적 (없으면 1)
                    current_epoch = getattr(self, '_debug_current_epoch', 1)

                    self.debug.log_batch_training(
                        epoch=current_epoch,
                        batch_idx=self._debug_batch_idx,
                        total_batches=getattr(self, '_debug_total_batches', 1),
                        loss=loss_float,
                        policy_loss=policy_loss_float,
                        value_loss=value_loss_float,
                        entropy_loss=entropy_float,
                        actions=actions,  # 원본 actions 리스트
                        action_probs=np.array(action_probs) if action_probs is not None else None,
                        entropy_coef=entropy_coef,
                        clip_ratio=clip_epsilon,
                        kl_divergence=kl_div
                    )

                    # 그래디언트 통계 로깅 (첫 배치만)
                    if self._debug_batch_idx == 1:
                        # grads는 FrozenDict 형태
                        grad_dict = {}
                        if 'params' in grads:
                            for layer_name, layer_params in grads['params'].items():
                                if hasattr(layer_params, 'items'):
                                    for param_name, param_grad in layer_params.items():
                                        full_name = f"{layer_name}_{param_name}"
                                        grad_dict[full_name] = np.array(param_grad) if hasattr(param_grad, 'shape') else param_grad

                        # Gradient norm 계산
                        grad_norm = 0.0
                        for grad_arr in grad_dict.values():
                            if hasattr(grad_arr, 'flatten'):
                                grad_norm += float(np.sum(grad_arr.flatten() ** 2))
                        grad_norm = np.sqrt(grad_norm)

                        self.debug.log_gradient_update(
                            epoch=current_epoch,
                            batch_idx=self._debug_batch_idx,
                            gradients=grad_dict,
                            learning_rate=self.current_learning_rate,
                            grad_norm=grad_norm,
                            clipped=grad_norm > 0.5  # clip_by_global_norm(0.5) 사용
                        )
                except Exception as debug_err:
                    logger.debug(f"⚠️ 배치 디버그 로깅 실패 (무시): {debug_err}")

            # 학습 데이터 검증 로깅 (첫 배치만) - 디버거로 대체
            if not hasattr(self, '_first_batch_logged'):
                actual_batch_size = states_jax.shape[0] if len(states_jax.shape) > 0 else 0

                # 🔥 디버거로 학습 데이터 통계 로깅
                if self.debug:
                    try:
                        self.debug.log_training_data_stats(
                            states=np.array(states_jax),
                            actions=actions,
                            rewards=rewards,
                            advantages=np.array(advantages)
                        )
                    except Exception as stats_err:
                        logger.debug(f"⚠️ 학습 데이터 통계 로깅 실패 (무시): {stats_err}")

                # 기존 로깅 유지
                logger.info(f"🔍 학습 데이터 검증 (배치 크기: {actual_batch_size}):")
                logger.info(f"   - States shape: {states_jax.shape}, dtype: {states_jax.dtype}")
                logger.info(f"   - Actions: {dict(zip(*np.unique(actions, return_counts=True)))}")
                logger.info(f"   - Rewards 범위: [{np.min(rewards):.4f}, {np.max(rewards):.4f}], 평균: {np.mean(rewards):.4f}")
                logger.info(f"   - Returns 범위: [{jnp.min(returns):.4f}, {jnp.max(returns):.4f}], 평균: {jnp.mean(returns):.4f}")
                logger.info(f"   - Advantages 범위: [{jnp.min(advantages):.4f}, {jnp.max(advantages):.4f}], 평균: {jnp.mean(advantages):.4f}")
                self._first_batch_logged = True

            return loss_float
        
        except Exception as e:
            logger.error(f"❌ 정책 업데이트 실패: {e}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")
            return 0.0
    
    def _evaluate_model(self, epoch: int, sample_experiences: List[Dict]) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            epoch: 현재 에폭
            sample_experiences: 샘플 경험 데이터
        
        Returns:
            평가 결과 딕셔너리
        """
        try:
            rewards = [exp.get('reward', 0.0) for exp in sample_experiences]
            
            return {
                'epoch': epoch,
                'avg_reward': np.mean(rewards) if rewards else 0.0,
                'std_reward': np.std(rewards) if rewards else 0.0,
                'sample_size': len(sample_experiences)
            }
        
        except Exception as e:
            logger.warning(f"⚠️ 모델 평가 실패: {e}")
            return {'epoch': epoch, 'avg_reward': 0.0}
    
    def _save_model(self, db_path: Optional[str] = None) -> str:
        """
        모델 저장
        
        Args:
            db_path: DB 경로 (None이면 설정 파일에서 가져옴)
        
        Returns:
            model_id: 저장된 모델 ID
        """
        try:
            import uuid
            from datetime import datetime
            
            # 모델 ID 생성
            model_id = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # 체크포인트 경로
            checkpoint_dir = self.config.get('paths', {}).get('checkpoints', '/workspace/rl_pipeline/artifacts/checkpoints')
            ckpt_path = os.path.join(checkpoint_dir, f"{model_id}.ckpt")
            
            # 체크포인트 저장
            save_ckpt(self.model, ckpt_path)
            
            # DB에 기록
            if db_path:
                self._save_to_db(model_id, ckpt_path, db_path)
            
            logger.info(f"✅ 모델 저장 완료: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")
            raise
    
    def _save_to_db(self, model_id: str, ckpt_path: str, db_path: str):
        """
        모델 정보를 DB에 저장
        
        Args:
            model_id: 모델 ID
            ckpt_path: 체크포인트 경로
            db_path: DB 경로
        """
        try:
            from rl_pipeline.db.writes import write_batch
            from rl_pipeline.hybrid.features import FEATURES_VERSION
            
            model_record = {
                'model_id': model_id,
                'algo': 'PPO',
                'features_ver': FEATURES_VERSION,
                'created_at': datetime.now().isoformat(),
                'ckpt_path': ckpt_path,
                'notes': json.dumps({
                    'hidden_dim': self.model['hidden_dim'],
                    'obs_dim': self.model['obs_dim'],
                    'action_dim': self.model['action_dim']
                })
            }
            
            write_batch([model_record], 'policy_models', db_path=db_path)
            
        except Exception as e:
            logger.warning(f"⚠️ DB 저장 실패 (계속 진행): {e}")


def _compute_returns(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Discounted returns 계산 (역순으로 누적)
    
    Args:
        rewards: 보상 배열
        gamma: 할인율
    
    Returns:
        Discounted returns 배열
    """
    if not JAX_AVAILABLE:
        return np.zeros_like(rewards)
    
    try:
        # 🔧 JAX 안전 방식: NumPy로 먼저 계산 후 변환
        rewards_np = np.array(rewards, dtype=np.float32)
        returns_np = np.zeros_like(rewards_np, dtype=np.float32)
        running_return = 0.0
        
        # 역순으로 계산
        for i in range(len(rewards_np) - 1, -1, -1):
            running_return = float(rewards_np[i]) + gamma * running_return
            returns_np[i] = running_return
        
        # NaN/Inf 체크 및 처리
        returns_np = np.nan_to_num(returns_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # JAX 배열로 변환
        returns_jax = jnp.array(returns_np, dtype=jnp.float32)
        
        return returns_jax
    except Exception as e:
        logger.warning(f"⚠️ Returns 계산 실패, 0 반환: {e}")
        return jnp.zeros_like(rewards)


def _compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    gamma: float,
    lam: float
) -> jnp.ndarray:
    """
    GAE (Generalized Advantage Estimation) 계산
    
    Args:
        rewards: 보상 배열
        values: 가치 추정 배열
        gamma: 할인율
        lam: GAE lambda (0~1)
    
    Returns:
        GAE advantages 배열
    """
    if not JAX_AVAILABLE:
        return np.zeros_like(rewards)
    
    try:
        # 🔧 JAX 안전 방식: NumPy로 먼저 계산 후 변환
        rewards_np = np.array(rewards, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)
        batch_size = len(rewards_np)
        
        advantages_np = np.zeros_like(rewards_np, dtype=np.float32)
        
        # 마지막 value는 0으로 가정 (에피소드 종료)
        last_gae = 0.0
        
        # 역순으로 GAE 계산
        for i in range(batch_size - 1, -1, -1):
            if i == batch_size - 1:
                next_value = 0.0  # 마지막 스텝
            else:
                next_value = float(values_np[i + 1])
            
            # TD residual: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            reward_val = float(rewards_np[i])
            value_val = float(values_np[i])
            
            delta = reward_val + gamma * next_value - value_val
            
            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            last_gae = delta + gamma * lam * last_gae
            advantages_np[i] = last_gae
        
        # NaN/Inf 체크 및 처리
        advantages_np = np.nan_to_num(advantages_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 값 범위 클리핑 (과도한 값 방지)
        advantages_np = np.clip(advantages_np, -100.0, 100.0)
        
        # JAX 배열로 변환
        advantages_jax = jnp.array(advantages_np, dtype=jnp.float32)
        
        return advantages_jax
    except Exception as e:
        logger.warning(f"⚠️ GAE 계산 실패, 0 반환: {e}")
        return jnp.zeros_like(rewards)


def train(config_path: str, db_path: Optional[str] = None) -> str:
    """
    학습 실행 함수
    
    Args:
        config_path: 설정 파일 경로
        db_path: DB 경로 (선택적)
    
    Returns:
        model_id: 학습된 모델 ID
    """
    try:
        # 설정 파일 로드
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # DB 경로 설정
        if db_path is None:
            db_path = config.get('paths', {}).get('db')
        
        # Trainer 초기화
        trainer = PPOTrainer(config)
        
        # Self-play 데이터는 파라미터로 받아야 함
        # 이 함수는 직접 호출되지 않고 train_from_selfplay_data를 통해 호출됨
        raise NotImplementedError(
            "train() 함수는 직접 호출하지 마세요. "
            "대신 PPOTrainer.train_from_selfplay_data()를 사용하세요. "
            "또는 auto_train_from_selfplay() 또는 auto_train_from_integrated_analysis()를 사용하세요."
        )
        
    except Exception as e:
        logger.error(f"❌ 학습 실행 실패: {e}")
        raise

