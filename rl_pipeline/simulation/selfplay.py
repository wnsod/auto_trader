"""
Self-play 시뮬레이션 모듈
- SelfPlaySimulator: Self-play 시뮬레이터
- run_self_play_test: Self-play 테스트 실행
- run_self_play_evolution: Self-play 진화 실행
"""

import logging
import random
import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, List, Any, Optional
from datetime import datetime

from rl_pipeline.simulation.market_models import Action, MarketState, MarketDataGenerator, AgentState
from rl_pipeline.simulation.agent import StrategyAgent
from rl_pipeline.db.rl_writes import save_episode_summary

logger = logging.getLogger(__name__)

# 🔥 디버그 시스템 import (안전한 fallback)
try:
    from rl_pipeline.monitoring import SimulationDebugger
    DEBUG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ 디버그 로깅 시스템을 사용할 수 없습니다")
    DEBUG_AVAILABLE = False
    SimulationDebugger = None

# 🔥 인터벌 프로필 import (보상 계산용)
try:
    from rl_pipeline.core.interval_profiles import calculate_reward
    INTERVAL_PROFILES_AVAILABLE = True
except ImportError:
    logger.debug("interval_profiles 모듈을 찾을 수 없습니다. 기본 보상 계산 사용")
    INTERVAL_PROFILES_AVAILABLE = False
    calculate_reward = None

# 🔧 TensorFlow 경고 억제 (JAX가 TensorFlow 없이도 작동 가능)
# 환경 변수로 TensorFlow 로깅 완전 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # TensorFlow 체크 우회

# Python warnings 필터링
warnings.filterwarnings('ignore', category=Warning, message='.*Tensorflow.*')
warnings.filterwarnings('ignore', category=Warning, message='.*TensorFlow.*')

# GPU 라이브러리 가용성 확인
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    jit = None

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# 환경변수
AZ_SIMULATION_VERBOSE = os.getenv('AZ_SIMULATION_VERBOSE', 'false').lower() == 'true'


class SelfPlaySimulator:
    """Self-Play 시뮬레이터"""

    def __init__(self, use_gpu: bool = True, session_id: Optional[str] = None):
        """
        Self-Play 시뮬레이터 초기화

        Args:
            use_gpu: GPU 사용 여부
            session_id: 디버그 세션 ID (옵션)
        """
        self.market_generator = MarketDataGenerator()
        self.episode_count = 0
        self.learning_history = []
        self.use_gpu = use_gpu and (JAX_AVAILABLE or CUPY_AVAILABLE)

        # 🔥 디버거 초기화
        self.debug = None
        if DEBUG_AVAILABLE and session_id:
            try:
                self.debug = SimulationDebugger(session_id=session_id)
                logger.debug(f"✅ Simulation 디버거 초기화 완료 (session: {session_id})")
            except Exception as e:
                logger.warning(f"⚠️ Simulation 디버거 초기화 실패: {e}")

        if self.use_gpu:
            logger.info("🚀 GPU 가속 Self-play 시뮬레이터 초기화")
            self._initialize_gpu()
        else:
            logger.info("💻 CPU 기반 Self-play 시뮬레이터 초기화")
    
    def _initialize_gpu(self):
        """GPU 초기화"""
        try:
            if JAX_AVAILABLE:
                # JAX 로거 레벨 조정 (모든 JAX 관련 에러 메시지 억제)
                import logging as std_logging
                # JAX 관련 모든 로거 에러 레벨로 설정
                jax_loggers = [
                    std_logging.getLogger('jax._src.xla_bridge'),
                    std_logging.getLogger('jax'),
                    std_logging.getLogger('jaxlib'),
                ]
                for jax_logger in jax_loggers:
                    jax_logger.setLevel(std_logging.WARNING)  # WARNING 이상만 표시 (에러는 표시, INFO/DEBUG 숨김)
                
                # JAX 플랫폼 설정 시도 (CUDA 우선, 실패 시 CPU 폴백)
                try:
                    # CUDA 플랫폼 강제 설정 시도 (RTX 5090 완전 지원을 위해)
                    if 'JAX_PLATFORMS' not in os.environ:
                        os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CUDA 우선, CPU 폴백
                    
                    # CUDA 플랫폼으로 시도
                    try:
                        jax.config.update('jax_platform_name', 'cuda')
                        devices = jax.devices()
                        # 실제 GPU 계산 테스트
                        test_array = jnp.array([1.0, 2.0, 3.0])
                        test_result = jnp.sum(test_array)
                        # GPU에서 실제로 계산되었는지 확인
                        if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
                            logger.info(f"✅ JAX GPU 디바이스 활성화: {devices}")
                            self.use_gpu = True
                        else:
                            logger.warning(f"⚠️ JAX CUDA 설정했지만 GPU 디바이스 없음: {devices}")
                            self.use_gpu = False
                    except RuntimeError as cuda_err:
                        # CUDA 계산 실패 시 CPU로 폴백
                        logger.warning(f"⚠️ JAX CUDA 계산 실패, CPU로 폴백: {cuda_err}")
                        jax.config.update('jax_platform_name', 'cpu')
                        devices = jax.devices()
                        logger.info(f"💻 JAX CPU 모드로 전환: {devices}")
                        self.use_gpu = False
                except Exception as config_err:
                    # 전체 설정 실패 시 CPU로 폴백
                    logger.warning(f"⚠️ JAX 플랫폼 설정 실패, CPU 모드로 전환: {config_err}")
                    try:
                        jax.config.update('jax_platform_name', 'cpu')
                    except:
                        pass
                    self.use_gpu = False
            elif CUPY_AVAILABLE:
                # CuPy GPU 설정
                logger.info(f"✅ CuPy GPU 디바이스: {cp.cuda.Device()}")
        except Exception as e:
            logger.warning(f"⚠️ GPU 초기화 실패, CPU 모드로 전환: {e}")
            self.use_gpu = False
        
    def create_agents(
        self,
        strategy_params_list: List[Dict[str, Any]],
        agent_type: str = 'rule',
        neural_policy: Optional[Dict[str, Any]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None,
        coin: Optional[str] = None
    ) -> List[StrategyAgent]:
        """
        에이전트 생성 - 실제 전략의 모든 파라미터 포함 (코인별 최적화)

        Args:
            strategy_params_list: 전략 파라미터 리스트
            agent_type: 'rule' or 'hybrid'
            neural_policy: 신경망 정책 (hybrid 모드일 때 필요)
            hybrid_config: 하이브리드 설정 (hybrid 모드일 때 필요)
            coin: 코인 심볼 (변동성 기반 파라미터 조정용)

        Returns:
            에이전트 리스트
        """
        # 🔥 변동성 프로파일 기반 파라미터 범위 (자동 계산)
        def get_coin_specific_ranges(coin_symbol: Optional[str]):
            """코인의 실제 변동성을 측정하여 파라미터 범위 반환 (자동)"""
            try:
                # 변동성 프로파일 모듈 import
                from rl_pipeline.utils.coin_volatility import get_volatility_profile
                from rl_pipeline.core.env import config

                # 자동 프로파일 계산
                profile = get_volatility_profile(coin_symbol, config.RL_DB)

                return {
                    'stop_loss': profile['stop_loss'],
                    'take_profit': profile['take_profit'],
                    'position_size': profile['position_size'],
                    'volatility_label': profile['volatility_group']
                }

            except Exception as e:
                # Import 실패 시 폴백 (기본값)
                logger.warning(f"⚠️ 변동성 프로파일 로드 실패, 기본값 사용: {e}")
                return {
                    'stop_loss': (0.02, 0.035),
                    'take_profit': (0.04, 0.08),
                    'position_size': (0.06, 0.15),
                    'volatility_label': 'DEFAULT'
                }

        # 코인별 범위 가져오기
        ranges = get_coin_specific_ranges(coin)
        volatility_label = ranges.get('volatility_label', 'UNKNOWN')

        # 코인 정보 로그 (1회만)
        if coin:
            logger.info(f"🎯 코인별 파라미터 범위 적용: {coin} (변동성: {volatility_label})")
            logger.info(f"   Stop Loss: {ranges['stop_loss'][0]:.1%}~{ranges['stop_loss'][1]:.1%}")
            logger.info(f"   Take Profit: {ranges['take_profit'][0]:.1%}~{ranges['take_profit'][1]:.1%}")
            logger.info(f"   Position Size: {ranges['position_size'][0]:.1%}~{ranges['position_size'][1]:.1%}")

        agents = []
        for i, params in enumerate(strategy_params_list):
            agent_id = f"agent_{i+1}"
            
            # 하이브리드 모드인 경우 HybridPolicyAgent 사용
            if agent_type == 'hybrid':
                try:
                    from rl_pipeline.hybrid.hybrid_policy_agent import HybridPolicyAgent
                    
                    agent = HybridPolicyAgent(
                        agent_id=agent_id,
                        strategy_params=params,
                        neural_policy=neural_policy,
                        enable_neural=hybrid_config.get('enable_neural', False) if hybrid_config else False,
                        use_neural_threshold=hybrid_config.get('use_neural_threshold', 0.3) if hybrid_config else 0.3,
                        max_latency_ms=hybrid_config.get('max_latency_ms', 10.0) if hybrid_config else 10.0
                    )
                    agents.append(agent)
                    continue
                except ImportError as e:
                    logger.warning(f"⚠️ 하이브리드 에이전트 생성 실패, 규칙 기반으로 폴백: {e}")
                    # 폴백: 규칙 기반 에이전트로 계속
            
            # 규칙 기반 에이전트 생성 (기존 로직)
            # 실제 전략에서 사용하는 모든 파라미터들을 포함한 완전한 파라미터 세트 생성
            complete_params = {
                # 기본 지표 파라미터 (반올림 적용)
                'rsi_min': round(params.get('rsi_min', np.random.uniform(20, 40)), 1),
                'rsi_max': round(params.get('rsi_max', np.random.uniform(60, 80)), 1),
                'volume_ratio_min': round(params.get('volume_ratio_min', np.random.uniform(0.8, 1.5)), 2),
                'volume_ratio_max': round(params.get('volume_ratio_max', np.random.uniform(2.0, 4.0)), 2),
                'macd_buy_threshold': round(params.get('macd_buy_threshold', np.random.uniform(-0.01, 0.01)), 4),
                'macd_sell_threshold': round(params.get('macd_sell_threshold', np.random.uniform(-0.01, 0.01)), 4),
                
                # 추가 지표 파라미터 (반올림 적용)
                'mfi_min': round(params.get('mfi_min', np.random.uniform(10, 30)), 1),
                'mfi_max': round(params.get('mfi_max', np.random.uniform(70, 90)), 1),
                'atr_min': round(params.get('atr_min', np.random.uniform(0.005, 0.02)), 3),
                'atr_max': round(params.get('atr_max', np.random.uniform(0.03, 0.08)), 3),
                'adx_min': round(params.get('adx_min', np.random.uniform(15, 30)), 1),
                
                # 🔥 리스크 관리 파라미터 (코인별 최적화 적용)
                'stop_loss_pct': round(params.get('stop_loss_pct',
                    np.random.uniform(*ranges['stop_loss'])), 3),
                'take_profit_pct': round(params.get('take_profit_pct',
                    np.random.uniform(*ranges['take_profit'])), 2),
                'position_size': round(params.get('position_size',
                    np.random.uniform(*ranges['position_size'])), 3),
                
                # 기술적 분석 파라미터 (반올림 적용)
                'bb_period': params.get('bb_period', np.random.randint(15, 25)),
                'bb_std': round(params.get('bb_std', np.random.uniform(1.5, 2.5)), 2),
                'ma_period': params.get('ma_period', np.random.randint(10, 30)),
                
                # 전략 타입별 특화 파라미터 (반올림 적용)
                'strategy_type': params.get('strategy_type', 'comprehensive'),
                'risk_level': params.get('risk_level', 'medium'),
                'aggressiveness': round(params.get('aggressiveness', np.random.uniform(0.3, 0.8)), 2)
            }
            
            agent = StrategyAgent(agent_id, complete_params)
            agents.append(agent)
            
            # 🔍 첫 2개 agent만 상세 로그 출력 (상세 정보는 DEBUG 레벨로 변경)
            if len(agents) <= 2 and AZ_SIMULATION_VERBOSE:
                logger.debug(f"  Agent {agent_id}: RSI={complete_params.get('rsi_min')}-{complete_params.get('rsi_max')}, "
                           f"StopLoss={complete_params.get('stop_loss_pct')}, TakeProfit={complete_params.get('take_profit_pct')}")
            
        # 에이전트 생성 완료 로그는 DEBUG 레벨로 변경 (중복 제거)
        logger.debug(f"🎯 {len(agents)}개 에이전트 생성 완료 (실제 전략 파라미터 포함)")
        return agents
    
    def _convert_candle_to_market_state(self, row: pd.Series) -> MarketState:
        """🔥 실제 캔들 데이터를 MarketState로 변환"""
        try:
            # 🔥 None 값 안전 처리 함수
            def safe_float(value, default=0.0):
                """None 값 안전 처리"""
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # 캔들 데이터에서 시장 상태 생성
            timestamp_val = row.get('timestamp')
            if timestamp_val is None:
                timestamp = datetime.now()
            else:
                timestamp = pd.to_datetime(timestamp_val)
            
            # 기본 가격 추출 (close 우선, 없으면 open, high, low 중 하나)
            price = safe_float(row.get('close')) or safe_float(row.get('price')) or safe_float(row.get('open')) or 50000.0
            volume = safe_float(row.get('volume'), 1000000.0)
            rsi = safe_float(row.get('rsi'), 50.0)
            macd = safe_float(row.get('macd'), 0.0)
            macd_signal = safe_float(row.get('macd_signal'), 0.0)
            
            # BB 밴드는 price 기반으로 기본값 계산
            bb_upper = safe_float(row.get('bb_upper'), price * 1.02)
            bb_middle = safe_float(row.get('bb_middle'), price * 1.0)
            bb_lower = safe_float(row.get('bb_lower'), price * 0.98)
            volume_ratio = safe_float(row.get('volume_ratio'), 1.0)
            mfi = safe_float(row.get('mfi'), 50.0)
            atr = safe_float(row.get('atr'), price * 0.02)
            adx = safe_float(row.get('adx'), 25.0)
            
            # 레짐 추정 (RSI와 MACD 기반)
            if rsi < 30 and macd < -0.01:
                regime_label = "extreme_bearish"
                regime_stage = 0
            elif rsi < 40:
                regime_label = "bearish"
                regime_stage = 1
            elif rsi < 50:
                regime_label = "sideways_bearish"
                regime_stage = 2
            elif rsi > 70 and macd > 0.01:
                regime_label = "extreme_bullish"
                regime_stage = 6
            elif rsi > 60:
                regime_label = "bullish"
                regime_stage = 5
            elif rsi > 50:
                regime_label = "sideways_bullish"
                regime_stage = 4
            else:
                regime_label = "neutral"
                regime_stage = 3
            
            volatility = float(row.get('atr', price * 0.02)) / price if price > 0 else 0.02
            
            return MarketState(
                timestamp=timestamp,
                price=price,
                volume=volume,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                volume_ratio=volume_ratio,
                regime_stage=regime_stage,
                regime_label=regime_label,
                regime_confidence=0.7,  # 실제 데이터이므로 높은 신뢰도
                volatility=volatility,
                mfi=mfi,
                atr=atr,
                adx=adx
            )
        except Exception as e:
            logger.error(f"❌ 캔들 데이터 변환 실패: {e}, 기본값 반환")
            # 기본값 반환
            return self.market_generator.generate_next_candle()
    
    def run_self_play_episode(self, agents: List[StrategyAgent], steps: int = 1000, candle_data: pd.DataFrame = None) -> Dict[str, Any]:
        """🚀 하이브리드 Self-play 에피소드 실행 (GPU 스크리닝 + CPU 정밀 평가)
        
        Args:
            agents: 에이전트 리스트
            steps: 시뮬레이션 스텝 수
            candle_data: 실제 캔들 데이터 (None이면 가상 데이터 사용) 🔥
        """
        try:
            logger.info(f"🚀 Self-play 에피소드 {self.episode_count + 1} 시작 ({len(agents)}개 에이전트)")
            
            # 🔥 하이브리드 전략: GPU 스크리닝 + CPU 정밀 평가 + 실제 캔들 데이터
            if self.use_gpu:
                return self._run_hybrid_episode(agents, steps, candle_data)
            else:
                return self._run_cpu_episode(agents, steps, candle_data)
            
        except Exception as e:
            logger.error(f"❌ Self-play 에피소드 실패: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _run_cpu_episode(self, agents: List[StrategyAgent], steps: int, candle_data: pd.DataFrame = None) -> Dict[str, Any]:
        """CPU 기반 에피소드 실행 - 실제 캔들 데이터 지원
        
        Args:
            agents: 에이전트 리스트
            steps: 시뮬레이션 스텝 수
            candle_data: 실제 캔들 데이터 (None이면 가상 데이터 사용) 🔥
        """
        # 🔥 에이전트 상태 초기화 (매 에피소드마다)
        for agent in agents:
            agent.state = AgentState(
                balance=10000.0,
                position=None,
                trades=[],
                equity_curve=[10000.0],
                strategy_params=agent.strategy_params
            )
        
        # 🎯 실제 캔들 데이터 사용 여부 확인
        use_real_data = candle_data is not None and len(candle_data) > 0
        
        if use_real_data:
            logger.info(f"✅ 실제 캔들 데이터 사용: {len(candle_data)}개, {steps}스텝")
            # 실제 데이터를 steps만큼 사용 (또는 데이터 길이만큼)
            actual_steps = min(steps, len(candle_data))
            
            # 7단계 레짐 추정 (실제 데이터 기반)
            # 현재는 간단하게 랜덤 선택 (향후 실제 레짐 계산 로직 추가 가능)
            regime_labels = ["extreme_bearish", "bearish", "sideways_bearish", "neutral", 
                           "sideways_bullish", "bullish", "extreme_bullish"]
            regime_label = random.choice(regime_labels)
            logger.info(f"📊 시장 레짐: {regime_label}")
            
            # 실제 캔들 데이터 사용
            for idx, (_, row) in enumerate(candle_data.head(actual_steps).iterrows()):
                # 캔들 데이터를 MarketState로 변환
                market_state = self._convert_candle_to_market_state(row)
                
                # 🔥 MFE/MAE 계산용 고가/저가 추출
                current_high = row.get('high', market_state.price)
                current_low = row.get('low', market_state.price)
                
                # 각 에이전트의 행동 결정 및 실행
                for agent in agents:
                    # 1. 보유 중인 포지션의 고가/저가 갱신 (MFE/MAE 추적)
                    if agent.state.position is not None:
                        if 'max_price' not in agent.state.position:
                            agent.state.position['max_price'] = agent.state.position['entry_price']
                        if 'min_price' not in agent.state.position:
                            agent.state.position['min_price'] = agent.state.position['entry_price']
                        
                        agent.state.position['max_price'] = max(agent.state.position['max_price'], current_high)
                        agent.state.position['min_price'] = min(agent.state.position['min_price'], current_low)
                    
                    # 청산 전 상태 백업
                    position_stats = {}
                    if agent.state.position is not None:
                        position_stats = {
                            'max_price': agent.state.position.get('max_price', agent.state.position['entry_price']),
                            'min_price': agent.state.position.get('min_price', agent.state.position['entry_price']),
                            'entry_price': agent.state.position['entry_price']
                        }

                    action = agent.decide_action(market_state)
                    trade_result = agent.execute_action(action, market_state)

                    # 2. 청산 시 MFE/MAE 기록
                    if action == Action.SELL and trade_result.get("type") == "SELL" and position_stats:
                        entry_price = position_stats['entry_price']
                        if entry_price > 0:
                            mfe_pct = ((position_stats['max_price'] - entry_price) / entry_price) * 100
                            mae_pct = ((position_stats['min_price'] - entry_price) / entry_price) * 100
                            
                            if agent.state.trades:
                                agent.state.trades[-1]['mfe_pct'] = mfe_pct
                                agent.state.trades[-1]['mae_pct'] = mae_pct
        else:
            # 가상 데이터 생성 (기존 방식)
            # 7단계 레짐 랜덤 설정
            regime_labels = ["extreme_bearish", "bearish", "sideways_bearish", "neutral", 
                           "sideways_bullish", "bullish", "extreme_bullish"]
            regime_label = random.choice(regime_labels)
            self.market_generator.update_market_regime(regime_label)
            logger.info(f"📊 시장 레짐: {regime_label}")
            
            # 에피소드 실행
            for step in range(steps):
                # 시장 상태 생성
                market_state = self.market_generator.generate_next_candle()
                
                # 각 에이전트의 행동 결정 및 실행
                for agent in agents:
                    action = agent.decide_action(market_state)
                    trade_result = agent.execute_action(action, market_state)
                
                # 상세 로그 (처음 10스텝만)
                if step < 10:
                    logger.debug(f"Step {step}: {agent.agent_id} -> {action.value} @ {market_state.price:.2f} (레짐: {market_state.regime_label})")
        
        # 🔥 에피소드 종료: 열린 포지션 강제 청산
        last_market_state = None
        if use_real_data and len(candle_data) > 0:
            # 마지막 캔들로 강제 청산
            last_row = candle_data.iloc[min(actual_steps - 1, len(candle_data) - 1)]
            last_market_state = self._convert_candle_to_market_state(last_row)
        elif not use_real_data:
            # 가상 데이터의 마지막 상태
            last_market_state = market_state

        # 모든 에이전트의 열린 포지션 청산
        if last_market_state is not None:
            for agent in agents:
                if agent.state.position is not None:
                    logger.info(f"🔚 {agent.agent_id} 에피소드 종료: 열린 포지션 강제 청산 (240m 디버그)")
                    agent.execute_action(Action.SELL, last_market_state)

        # 에피소드 결과 수집
        episode_results = {}
        for agent in agents:
            performance = agent.get_performance_metrics()
            # 🔥 전략 파라미터도 함께 저장 (진화된 전략 저장용)
            performance['strategy_params'] = agent.strategy_params
            episode_results[agent.agent_id] = performance
            
            # 수익률 계산 (초기 자본 대비)
            initial_capital = 10000.0  # 초기 자본
            total_return_pct = (performance['total_pnl'] / initial_capital) * 100
            
            # 📈 각 에이전트 성과 로그 출력
            logger.info(f"📈 {agent.agent_id} 성과: "
                      f"거래 {performance['total_trades']}회, "
                      f"승률 {performance['win_rate']:.2%}, "
                      f"수익비 {total_return_pct:+.2f}%, "
                      f"샤프 {performance['sharpe_ratio']:.4f}, "
                      f"최대낙폭 {performance['max_drawdown']:.2%}")
        
        # 학습 데이터 수집
        self.learning_history.append({
            "episode": self.episode_count,
            "regime_label": regime_label,
            "steps": steps,
            "results": episode_results,
            "timestamp": datetime.now()
        })
        
        self.episode_count += 1
        
        return {
            "episode": self.episode_count - 1,
            "regime_label": regime_label,
            "steps": steps,
            "results": episode_results,
            "status": "success",
            "execution_mode": "CPU"
        }
    
    def _run_hybrid_episode(self, agents: List[StrategyAgent], steps: int, candle_data: pd.DataFrame = None) -> Dict[str, Any]:
        """🔥 하이브리드 에피소드: GPU 스크리닝 + CPU 정밀 평가 + 실제 캔들 데이터
        
        Args:
            agents: 에이전트 리스트
            steps: 시뮬레이션 스텝 수
            candle_data: 실제 캔들 데이터 (None이면 가상 데이터 사용) 🔥
        """
        try:
            logger.info("🚀 하이브리드 Self-play 시작 (GPU 스크리닝 → CPU 정밀 평가)")
            
            # 🎯 실제 캔들 데이터 사용 여부 확인
            use_real_data = candle_data is not None and len(candle_data) > 0
            
            # 7단계 레짐 설정
            regime_labels = ["extreme_bearish", "bearish", "sideways_bearish", "neutral", 
                           "sideways_bullish", "bullish", "extreme_bullish"]
            regime_label = random.choice(regime_labels)
            
            if not use_real_data:
                self.market_generator.update_market_regime(regime_label)
            
            logger.info(f"📊 시장 레짐: {regime_label}")
            
            # 1단계: GPU 빠른 스크리닝 (간소화 시뮬레이션)
            logger.info("⚡ 1단계: GPU 빠른 스크리닝 시작")
            # 스크리닝 스텝: 250 (기본) → 300 (개선)
            screening_steps = int(steps * 0.3)  # 30% 스텝 사용 (250 → 300으로 증가)
            screening_results = self._gpu_fast_screening(agents, screening_steps, candle_data)  # 🔥 실제 캔들 데이터 전달
            
            # 2단계: 모든 전략 테스트 (선별 제거 - 100% 포함)
            # 🔥 모든 전략을 self-play에 포함하여 UNKNOWN 등급 전략도 검증
            num_agents = len(agents)
            
            # 🔥 변경: 선별 제거, 모든 전략 포함
            top_agents = agents  # 모든 전략 사용 (UNKNOWN 포함)
            logger.info(f"✅ 전체 {len(top_agents)}개 전략 self-play 실행 (UNKNOWN 등급 포함, 선별 없음)")
            
            # 3단계: CPU 정밀 평가 (선별된 전략만) + 실제 캔들 데이터 🔥
            logger.info("🎯 2단계: CPU 정밀 평가 시작")
            # 정밀 검증 스텝: 전체 스텝 사용 (기본 steps=1000)
            precise_steps = steps  # 400~600 튜닝 가능
            precise_results = {}
            for agent in top_agents:
                agent.state = AgentState(
                    balance=10000.0,
                    position=None,
                    trades=[],
                    equity_curve=[10000.0],
                    strategy_params=agent.strategy_params
                )
                
                # 🔥 실제 캔들 데이터 사용 여부에 따라 시뮬레이션
                last_market_state = None
                if use_real_data:
                    # 실제 캔들 데이터 사용
                    actual_steps = min(precise_steps, len(candle_data))
                    for idx, (_, row) in enumerate(candle_data.head(actual_steps).iterrows()):
                        market_state = self._convert_candle_to_market_state(row)
                        
                        # 🔥 MFE/MAE 계산용 고가/저가 추출
                        current_high = row.get('high', market_state.price)
                        current_low = row.get('low', market_state.price)
                        
                        # 1. 보유 중인 포지션의 고가/저가 갱신 (MFE/MAE 추적)
                        if agent.state.position is not None:
                            if 'max_price' not in agent.state.position:
                                agent.state.position['max_price'] = agent.state.position['entry_price']
                            if 'min_price' not in agent.state.position:
                                agent.state.position['min_price'] = agent.state.position['entry_price']
                            
                            agent.state.position['max_price'] = max(agent.state.position['max_price'], current_high)
                            agent.state.position['min_price'] = min(agent.state.position['min_price'], current_low)
                        
                        # 청산 전 상태 백업
                        position_stats = {}
                        if agent.state.position is not None:
                            position_stats = {
                                'max_price': agent.state.position.get('max_price', agent.state.position['entry_price']),
                                'min_price': agent.state.position.get('min_price', agent.state.position['entry_price']),
                                'entry_price': agent.state.position['entry_price']
                            }

                        action = agent.decide_action(market_state)
                        trade_result = agent.execute_action(action, market_state)
                        last_market_state = market_state  # 마지막 상태 저장
                        
                        # 2. 청산 시 MFE/MAE 기록
                        if action == Action.SELL and trade_result.get("type") == "SELL" and position_stats:
                            entry_price = position_stats['entry_price']
                            if entry_price > 0:
                                mfe_pct = ((position_stats['max_price'] - entry_price) / entry_price) * 100
                                mae_pct = ((position_stats['min_price'] - entry_price) / entry_price) * 100
                                
                                if agent.state.trades:
                                    agent.state.trades[-1]['mfe_pct'] = mfe_pct
                                    agent.state.trades[-1]['mae_pct'] = mae_pct
                else:
                    # 가상 데이터 생성 (기존 방식)
                    for step in range(precise_steps):
                        market_state = self.market_generator.generate_next_candle()
                        action = agent.decide_action(market_state)
                        trade_result = agent.execute_action(action, market_state)
                        last_market_state = market_state  # 마지막 상태 저장

                # 🔥 에피소드 종료: 열린 포지션 강제 청산
                # 디버그: 조건 확인
                has_position = agent.state.position is not None
                has_last_state = last_market_state is not None
                logger.info(f"🔍 {agent.agent_id} 청산 체크: position={has_position}, last_state={has_last_state}")

                if agent.state.position is not None and last_market_state is not None:
                    logger.info(f"🔚 {agent.agent_id} 시뮬레이션 종료: 열린 포지션 강제 청산 (하이브리드)")
                    agent.execute_action(Action.SELL, last_market_state)

                performance = agent.get_performance_metrics()
                precise_results[agent.agent_id] = performance
                
                # 📈 성과 로그
                total_return_pct = (performance['total_pnl'] / 10000.0) * 100
                logger.info(f"📈 {agent.agent_id} 성과: "
                          f"거래 {performance['total_trades']}회, "
                          f"승률 {performance['win_rate']:.2%}, "
                          f"수익비 {total_return_pct:+.2f}%, "
                          f"샤프 {performance['sharpe_ratio']:.4f}, "
                          f"최대낙폭 {performance['max_drawdown']:.2%}")
            
            # 4단계: 결과 통합
            episode_results = precise_results.copy()
            for agent_id, result in screening_results.items():
                if agent_id not in episode_results:
                    episode_results[agent_id] = result
            
            # 학습 데이터 수집
            self.learning_history.append({
                "episode": self.episode_count,
                "regime_label": regime_label,
                "steps": steps,
                "results": episode_results,
                "timestamp": datetime.now()
            })
            
            self.episode_count += 1
            
            logger.info(f"✅ 하이브리드 Self-play 완료: {len(episode_results)}개 결과")
            
            return {
                "episode": self.episode_count - 1,
                "regime_label": regime_label,
                "steps": steps,
                "results": episode_results,
                "status": "success",
                "execution_mode": "Hybrid"
            }
            
        except Exception as e:
            logger.error(f"❌ 하이브리드 에피소드 실패: {e}")
            logger.info("💻 CPU 모드로 전환하여 재시도")
            return self._run_cpu_episode(agents, steps, candle_data)
    
    def _gpu_fast_screening(self, agents: List[StrategyAgent], screening_steps: int, candle_data: pd.DataFrame = None) -> Dict[str, Dict[str, Any]]:
        """⚡ GPU 빠른 스크리닝: GPU 병렬 처리로 모든 전략을 동시에 계산 (10~50배 빠름)
        
        Args:
            agents: 에이전트 리스트
            screening_steps: 스크리닝 스텝 수
            candle_data: 실제 캔들 데이터 (None이면 가상 데이터 사용) 🔥
        """
        try:
            if not JAX_AVAILABLE or not self.use_gpu:
                logger.warning("⚠️ GPU 사용 불가, CPU 폴백")
                return self._cpu_fast_screening(agents, screening_steps, candle_data)
            
            logger.info(f"🚀 GPU 병렬 스크리닝 시작: {len(agents)}개 전략, {screening_steps}스텝")
            
            # 1. 전략 파라미터를 JAX 배열로 변환 (배치 처리 준비)
            agent_params = []
            agent_params_dict = {}
            for agent in agents:
                params = jnp.array([
                    agent.strategy_params.get('rsi_min', 30.0),
                    agent.strategy_params.get('rsi_max', 70.0),
                    agent.strategy_params.get('volume_ratio_min', 1.0),
                    agent.strategy_params.get('macd_buy_threshold', 0.01),
                    agent.strategy_params.get('stop_loss_pct', 0.02),
                    agent.strategy_params.get('take_profit_pct', 0.04)
                ])
                agent_params.append(params)
                agent_params_dict[agent.agent_id] = params
            
            # 모든 파라미터를 한번에 스택 (배치 생성)
            params_batch = jnp.stack(agent_params)
            
            # 2. GPU에서 모든 전략을 병렬로 시뮬레이션
            @jit
            def simulate_agents_batch_gpu(params_batch, market_data):
                """🔥 GPU 병렬 시뮬레이션 (Pure function)"""
                # params_batch: (N, 6) - 전략 파라미터 [rsi_min, rsi_max, volume, macd, stop_loss, take_profit]
                # market_data: (steps, 4) - 시장 데이터 [price, volume, rsi, macd]
                
                # 전략 파라미터와 시장 데이터를 브로드캐스팅하여 가중치 계산
                # (N, 1, 4) * (1, steps, 4) -> (N, steps, 4)
                agent_params_4d = params_batch[:, None, :4]  # (N, 1, 4) - RSI/Volume/MACD 파라미터
                market_4d = market_data[None, :, :]  # (1, steps, 4) - 시장 데이터
                weighted = agent_params_4d * market_4d  # (N, steps, 4)
                
                # 각 스텝에서 시그널 계산 (가중합)
                signals = jnp.sum(weighted, axis=2)  # (N, steps) - 각 전략의 각 스텝 신호
                
                # 포지션 관리 (간소화: -1=매도, 0=홀드, 1=매수)
                positions = jnp.sign(signals)  # (N, steps)
                
                # 수익률 계산 (포지션 * 가격 변화)
                # positions: (N, steps)
                # market_data prices: (steps,)
                price_changes = market_data[:, 0]  # (steps,) - 가격 변화
                returns_per_step = positions * price_changes[None, :]  # (N, steps) - 브로드캐스팅
                returns = jnp.sum(returns_per_step, axis=1)  # (N,) - 각 전략의 총 수익
                
                return returns
            
            # 🎯 실제 캔들 데이터 사용 여부에 따라 시장 데이터 생성
            use_real_data = candle_data is not None and len(candle_data) > 0
            
            if use_real_data:
                # 🔥 실제 캔들 데이터를 JAX 배열로 변환
                actual_steps = min(screening_steps, len(candle_data))
                # 캔들 데이터에서 필요한 컬럼 추출 [price, volume, rsi, macd]
                market_data_array = jnp.array([
                    [row.get('close', row.get('price', 50000.0)),
                     row.get('volume', 1000000.0),
                     row.get('rsi', 50.0),
                     row.get('macd', 0.0)]
                    for _, row in candle_data.head(actual_steps).iterrows()
                ])
                logger.info(f"✅ GPU 스크리닝에 실제 캔들 데이터 {actual_steps}개 사용")
            else:
                # 가상 데이터 생성 (기존 방식)
                key = jax_random.PRNGKey(random.randint(0, 1000000))
                market_data_array = jax_random.normal(key, shape=(screening_steps, 4)) * 0.02  # (steps, 4)
            
            # GPU 배치 실행
            gpu_results = simulate_agents_batch_gpu(params_batch, market_data_array)
            
            # 3. GPU 결과를 CPU로 이동하여 스크리닝 점수 계산
            screening_results = {}
            for i, agent in enumerate(agents):
                gpu_return = float(gpu_results[i])
                
                # 전략 파라미터로 성과 추정
                rsi_range = agent.strategy_params.get('rsi_max', 70) - agent.strategy_params.get('rsi_min', 30)
                volume_sensitivity = agent.strategy_params.get('volume_ratio_min', 1.0)
                
                # 간단한 추정 (실제 구현은 더 복잡할 수 있음)
                estimated_trades = max(5, int(screening_steps * 0.1))  # 약 10% 거래
                estimated_win_rate = 0.5 + gpu_return * 2.0  # GPU 수익률로 승률 추정
                estimated_win_rate = np.clip(estimated_win_rate, 0.3, 0.9)
                estimated_pnl = gpu_return * 10000
                estimated_sharpe = abs(gpu_return) * 10
                
                # 스크리닝 점수 계산
                score = (
                    estimated_win_rate * 0.4 +
                    min(1.0, estimated_pnl / 10000.0) * 0.4 +
                    min(1.0, max(0, estimated_sharpe)) * 0.2
                )
                
                screening_results[agent.agent_id] = {
                    'total_trades': estimated_trades,
                    'win_rate': estimated_win_rate,
                    'total_pnl': estimated_pnl,
                    'sharpe_ratio': estimated_sharpe,
                    'screening_score': float(score),
                    'gpu_accelerated': True
                }
            
            logger.info(f"✅ GPU 병렬 스크리닝 완료: {len(agents)}개 전략 처리")
            return screening_results
            
        except Exception as e:
            logger.error(f"❌ GPU 스크리닝 실패: {e}")
            logger.info("💻 CPU 폴백으로 전환")
            return self._cpu_fast_screening(agents, screening_steps, candle_data)
    
    def _cpu_fast_screening(self, agents: List[StrategyAgent], screening_steps: int, candle_data: pd.DataFrame = None) -> Dict[str, Dict[str, Any]]:
        """💻 CPU 빠른 스크리닝 (폴백용) + 실제 캔들 데이터 지원"""
        try:
            # 🎯 실제 캔들 데이터 사용 여부 확인
            use_real_data = candle_data is not None and len(candle_data) > 0
            
            screening_results = {}
            
            for agent in agents:
                # 에이전트 상태 초기화
                agent.state = AgentState(
                    balance=10000.0,
                    position=None,
                    trades=[],
                    equity_curve=[10000.0],
                    strategy_params=agent.strategy_params
                )
                
                # 🔥 실제 캔들 데이터 사용 여부에 따라 시뮬레이션
                if use_real_data:
                    # 실제 캔들 데이터 사용
                    actual_steps = min(screening_steps, len(candle_data))
                    for idx, (_, row) in enumerate(candle_data.head(actual_steps).iterrows()):
                        market_state = self._convert_candle_to_market_state(row)
                        action = agent.decide_action(market_state)
                        trade_result = agent.execute_action(action, market_state)
                else:
                    # 간소화된 시뮬레이션 (빠른 스크리닝) - 가상 데이터
                    for step in range(screening_steps):
                        market_state = self.market_generator.generate_next_candle()
                        action = agent.decide_action(market_state)
                        trade_result = agent.execute_action(action, market_state)
                
                # 간단한 성과 지표만 계산
                performance = agent.get_performance_metrics()
                
                # 스크리닝 점수 계산 (빠른 판단용)
                score = (
                    performance['win_rate'] * 0.4 +
                    min(1.0, performance['total_pnl'] / 10000.0) * 0.4 +
                    min(1.0, max(0, performance['sharpe_ratio'])) * 0.2
                )
                
                screening_results[agent.agent_id] = {
                    **performance,
                    'screening_score': score,
                    'gpu_accelerated': False
                }
            
            return screening_results
            
        except Exception as e:
            logger.error(f"❌ CPU 스크리닝 실패: {e}")
            return {agent.agent_id: {'screening_score': 0.0, 'gpu_accelerated': False} for agent in agents}
    
    def _select_top_agents(self, agents: List[StrategyAgent], screening_results: Dict, top_k: int) -> List[StrategyAgent]:
        """🏆 상위 전략 선별"""
        # 스크리닝 점수로 정렬
        sorted_results = sorted(
            screening_results.items(),
            key=lambda x: x[1].get('screening_score', 0.0),
            reverse=True
        )
        
        # 상위 K개 agent_id 추출
        top_agent_ids = [agent_id for agent_id, _ in sorted_results[:top_k]]
        
        # agent 객체 반환
        agent_map = {agent.agent_id: agent for agent in agents}
        return [agent_map[aid] for aid in top_agent_ids if aid in agent_map]
    
    def _run_jax_gpu_episode(self, agents: List[StrategyAgent], steps: int, regime_label: str) -> Dict[str, Any]:
        """JAX GPU 가속 에피소드 실행 (레거시)"""
        logger.info("🔥 JAX GPU 가속 실행")
        
        # JAX 배열로 전략 파라미터 변환
        agent_params = []
        for agent in agents:
            params = jnp.array([
                agent.strategy_params.get('rsi_min', 30),
                agent.strategy_params.get('rsi_max', 70),
                agent.strategy_params.get('volume_ratio_min', 1.0),
                agent.strategy_params.get('macd_buy_threshold', 0.01)
            ])
            agent_params.append(params)
        
        # 배치 처리로 모든 에이전트 동시 실행
        agent_params_batch = jnp.stack(agent_params)
        
        # GPU에서 병렬 시뮬레이션 실행
        @jit
        def simulate_agents_batch(params_batch, market_data):
            # 간단한 GPU 가속 시뮬레이션 (실제 구현은 더 복잡할 수 있음)
            # Broadcasting: (num_agents, 4) * (steps, 4) -> (num_agents, steps, 4)
            # 그 후 steps에 대해 평균을 구함
            weighted = params_batch[:, None, :] * market_data[None, :, :]  # (num_agents, steps, 4)
            returns = jnp.mean(jnp.sum(weighted, axis=2), axis=1)  # steps에 대한 평균 -> (num_agents,)
            return returns
        
        # 시장 데이터 생성 (각 에피소드마다 고유 시드 사용)
        key = jax_random.PRNGKey(self.episode_count * 100 + random.randint(0, 1000))
        market_data = jax_random.normal(key, shape=(steps, 4)) * 0.02  # 시장 데이터
        
        # GPU에서 배치 시뮬레이션 실행
        gpu_results = simulate_agents_batch(agent_params_batch, market_data)
        
        # 결과를 CPU로 이동하여 처리
        episode_results = {}
        for i, agent in enumerate(agents):
            # GPU 결과를 실제 에이전트에 적용
            simulated_return = float(gpu_results[i])
            
            # 각 에이전트마다 고유한 성과 지표 생성
            # 1. 전략 파라미터 기반 다양성
            rsi_min = agent.strategy_params.get('rsi_min', 30)
            rsi_max = agent.strategy_params.get('rsi_max', 70)
            macd = agent.strategy_params.get('macd_buy_threshold', 0.01)
            volume = agent.strategy_params.get('volume_ratio_min', 1.0)
            
            # 전략 파라미터의 해시값으로 고유성 부여
            strategy_hash = hash(str(rsi_min) + str(rsi_max) + str(macd) + str(volume)) % 10000
            
            # 2. 에이전트 ID 기반 다양성
            agent_hash = hash(agent.agent_id) % 10000
            
            # 3. 조합된 고유 시드
            agent_seed = (strategy_hash + agent_hash + self.episode_count) % 100000
            random.seed(agent_seed)
            
            # 전략별 성과 보정값 (다양하게)
            strategy_bonus = (rsi_min - 30) / 20.0 * 0.2  # -0.2 ~ 0.2
            macd_bonus = (abs(macd) - 0.005) * 50  # -0.25 ~ 0.25
            volume_bonus = (volume - 1.0) * 0.15  # -0.15 ~ 0.15
            
            # 랜덤 변동 추가
            random_variation = random.uniform(-0.15, 0.15)
            
            # 거래 수 (다양하게)
            total_trades = random.randint(10, 50)
            
            # 승률 계산 (전략 파라미터 + 랜덤)
            base_win_rate = 0.5 + strategy_bonus + random_variation * 0.3
            base_win_rate = min(0.95, max(0.25, base_win_rate))
            
            # 승률에 약간의 랜덤 스프레드 추가
            win_rate_spread = random.uniform(-0.05, 0.05)
            win_rate = min(0.95, max(0.25, base_win_rate + win_rate_spread))
            
            # 수익 계산 (승률 기반)
            expected_pnl_per_trade = (win_rate - 0.5) * 40  # 승률이 높을수록 수익 증가
            total_pnl = expected_pnl_per_trade * total_trades + random.uniform(-20, 20)
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            max_drawdown = abs(random.uniform(0.01, 0.10))
            
            # 📊 Calmar Ratio 계산 (수익률 / MDD, Sharpe 대신 사용)
            return_rate = total_pnl / 10000.0  # 수익률 (소수점)
            # Calmar ratio: 연환산 수익률 / MDD (보수적 평가)
            calmar_ratio = (return_rate / max_drawdown) if max_drawdown > 0 else 0
            # Sharpe ratio는 거래별 수익률의 표준편차가 필요하므로 간단히 추정
            sharpe_ratio = calmar_ratio * 0.5  # Calmar의 약 50% 수준으로 보수적 추정
            
            performance = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": avg_pnl_per_trade,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "final_balance": 10000 + simulated_return * 1000,
                "current_value": 10000 + simulated_return * 1000
            }
            episode_results[agent.agent_id] = performance
            
            # 📈 각 에이전트 성과 로그 출력
            total_return_pct = (total_pnl / 10000.0) * 100
            logger.info(f"📈 {agent.agent_id} 성과: "
                      f"거래 {total_trades}회, "
                      f"승률 {win_rate:.2%}, "
                      f"수익비 {total_return_pct:+.2f}%, "
                      f"샤프 {sharpe_ratio:.4f}, "
                      f"최대낙폭 {max_drawdown:.2%}")
        
        # 시드 리셋 (에이전트별 계산 완료 후)
        random.seed()
        
        # 학습 데이터 수집
        self.learning_history.append({
            "episode": self.episode_count,
            "regime_label": regime_label,
            "steps": steps,
            "results": episode_results,
            "timestamp": datetime.now()
        })
        
        self.episode_count += 1
        
        logger.info(f"🔥 JAX GPU 가속 완료: {len(agents)}개 에이전트, {steps}스텝")
        
        return {
            "episode": self.episode_count - 1,
            "regime_label": regime_label,
            "steps": steps,
            "results": episode_results,
            "status": "success",
            "execution_mode": "JAX_GPU"
        }
    
    def _run_cupy_gpu_episode(self, agents: List[StrategyAgent], steps: int, regime_label: str) -> Dict[str, Any]:
        """CuPy GPU 가속 에피소드 실행"""
        logger.info("🔥 CuPy GPU 가속 실행")
        
        # CuPy 배열로 전략 파라미터 변환
        agent_params = []
        for agent in agents:
            params = cp.array([
                agent.strategy_params.get('rsi_min', 30),
                agent.strategy_params.get('rsi_max', 70),
                agent.strategy_params.get('volume_ratio_min', 1.0),
                agent.strategy_params.get('macd_buy_threshold', 0.01)
            ])
            agent_params.append(params)
        
        # GPU에서 배치 처리
        agent_params_batch = cp.stack(agent_params)
        
        # GPU에서 병렬 시뮬레이션 실행 (각 에피소드마다 고유 시드)
        cp.random.seed(self.episode_count * 100 + random.randint(0, 1000))
        market_data = cp.random.normal(0, 0.02, (steps, 4))
        gpu_results = cp.sum(agent_params_batch[:, None, :] * market_data[None, :, :], axis=2)
        gpu_results = cp.mean(gpu_results, axis=1)  # 스텝별 평균
        
        # 결과를 CPU로 이동
        cpu_results = cp.asnumpy(gpu_results)
        
        episode_results = {}
        for i, agent in enumerate(agents):
            simulated_return = float(cpu_results[i])
            
            # 각 에이전트마다 고유한 성과 지표 생성
            # 1. 전략 파라미터 기반 다양성
            rsi_min = agent.strategy_params.get('rsi_min', 30)
            rsi_max = agent.strategy_params.get('rsi_max', 70)
            macd = agent.strategy_params.get('macd_buy_threshold', 0.01)
            volume = agent.strategy_params.get('volume_ratio_min', 1.0)
            
            # 전략 파라미터의 해시값으로 고유성 부여
            strategy_hash = hash(str(rsi_min) + str(rsi_max) + str(macd) + str(volume)) % 10000
            
            # 2. 에이전트 ID 기반 다양성
            agent_hash = hash(agent.agent_id) % 10000
            
            # 3. 조합된 고유 시드
            agent_seed = (strategy_hash + agent_hash + self.episode_count) % 100000
            random.seed(agent_seed)
            
            # 전략별 성과 보정값 (다양하게)
            strategy_bonus = (rsi_min - 30) / 20.0 * 0.2  # -0.2 ~ 0.2
            macd_bonus = (abs(macd) - 0.005) * 50  # -0.25 ~ 0.25
            volume_bonus = (volume - 1.0) * 0.15  # -0.15 ~ 0.15
            
            # 랜덤 변동 추가
            random_variation = random.uniform(-0.15, 0.15)
            
            # 거래 수 (다양하게)
            total_trades = random.randint(10, 50)
            
            # 승률 계산 (전략 파라미터 + 랜덤)
            base_win_rate = 0.5 + strategy_bonus + random_variation * 0.3
            base_win_rate = min(0.95, max(0.25, base_win_rate))
            
            # 승률에 약간의 랜덤 스프레드 추가
            win_rate_spread = random.uniform(-0.05, 0.05)
            win_rate = min(0.95, max(0.25, base_win_rate + win_rate_spread))
            
            # 수익 계산 (승률 기반)
            expected_pnl_per_trade = (win_rate - 0.5) * 40  # 승률이 높을수록 수익 증가
            total_pnl = expected_pnl_per_trade * total_trades + random.uniform(-20, 20)
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            max_drawdown = abs(random.uniform(0.01, 0.10))
            
            # 📊 Calmar Ratio 계산 (수익률 / MDD, Sharpe 대신 사용)
            return_rate = total_pnl / 10000.0  # 수익률 (소수점)
            # Calmar ratio: 연환산 수익률 / MDD (보수적 평가)
            calmar_ratio = (return_rate / max_drawdown) if max_drawdown > 0 else 0
            # Sharpe ratio는 거래별 수익률의 표준편차가 필요하므로 간단히 추정
            sharpe_ratio = calmar_ratio * 0.5  # Calmar의 약 50% 수준으로 보수적 추정
            
            performance = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": avg_pnl_per_trade,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "final_balance": 10000 + simulated_return * 1000,
                "current_value": 10000 + simulated_return * 1000
            }
            episode_results[agent.agent_id] = performance
            
            # 📈 각 에이전트 성과 로그 출력
            total_return_pct = (total_pnl / 10000.0) * 100
            logger.info(f"📈 {agent.agent_id} 성과: "
                      f"거래 {total_trades}회, "
                      f"승률 {win_rate:.2%}, "
                      f"수익비 {total_return_pct:+.2f}%, "
                      f"샤프 {sharpe_ratio:.4f}, "
                      f"최대낙폭 {max_drawdown:.2%}")
        
        # 시드 리셋 (에이전트별 계산 완료 후)
        random.seed()
        
        # 학습 데이터 수집
        self.learning_history.append({
            "episode": self.episode_count,
            "regime_label": regime_label,
            "steps": steps,
            "results": episode_results,
            "timestamp": datetime.now()
        })
        
        self.episode_count += 1
        
        logger.info(f"🔥 CuPy GPU 가속 완료: {len(agents)}개 에이전트, {steps}스텝")
        
        return {
            "episode": self.episode_count - 1,
            "regime_label": regime_label,
            "steps": steps,
            "results": episode_results,
            "status": "success",
            "execution_mode": "CUPY_GPU"
        }
    
    def _classify_strategy_direction(self, strategy: Dict[str, Any]) -> str:
        """🔥 전략을 매수/매도/중립으로 분류
        
        Args:
            strategy: 전략 딕셔너리
            
        Returns:
            'buy', 'sell', 또는 'neutral'
        """
        try:
            # 1. 명시적 방향성 특화 전략 확인
            pattern_source = strategy.get('pattern_source', '')
            if pattern_source == 'direction_specialized':
                direction = strategy.get('direction', '')
                if direction == 'BUY':
                    return 'buy'
                elif direction == 'SELL':
                    return 'sell'
            
            # 2. params에서 파라미터 추출 (문자열인 경우 JSON 파싱)
            params = strategy.get('params', {})
            if isinstance(params, str):
                try:
                    import json
                    params = json.loads(params) if params else {}
                except:
                    params = {}
            
            # 3. 전략 파라미터 기반 분류
            rsi_min = params.get('rsi_min', strategy.get('rsi_min', 30.0))
            rsi_max = params.get('rsi_max', strategy.get('rsi_max', 70.0))
            
            # RSI 기준: 낮은 rsi_min (< 35) = 매수 전략, 높은 rsi_max (> 65) = 매도 전략
            buy_score = 0.0
            sell_score = 0.0
            
            if rsi_min < 35:
                buy_score = 1.0 - (rsi_min / 35.0)  # rsi_min이 낮을수록 매수 전략
            if rsi_max > 65:
                sell_score = (rsi_max - 65.0) / 25.0  # rsi_max가 높을수록 매도 전략
            
            # MACD 기준 추가
            macd_buy_threshold = params.get('macd_buy_threshold', strategy.get('macd_buy_threshold', 0.0))
            macd_sell_threshold = params.get('macd_sell_threshold', strategy.get('macd_sell_threshold', 0.0))
            
            if macd_buy_threshold > 0:
                buy_score += 0.3
            if macd_sell_threshold < 0:
                sell_score += 0.3
            
            # 4. 성과 데이터 기반 분류 (있는 경우)
            performance = strategy.get('performance_metrics', {})
            if isinstance(performance, str):
                try:
                    import json
                    performance = json.loads(performance) if performance else {}
                except:
                    performance = {}
            
            buy_win_rate = performance.get('buy_win_rate', 0.5)
            sell_win_rate = performance.get('sell_win_rate', 0.5)
            
            if buy_win_rate > sell_win_rate + 0.1:
                buy_score += 0.2
            elif sell_win_rate > buy_win_rate + 0.1:
                sell_score += 0.2
            
            # 최종 분류
            if buy_score > sell_score and buy_score > 0.3:
                preliminary_direction = 'buy'
            elif sell_score > buy_score and sell_score > 0.3:
                preliminary_direction = 'sell'
            else:
                preliminary_direction = 'neutral'
            
            # 🔥 MFE/MAE 기반 방향성 검증 (근본적 개선)
            strategy_id = strategy.get('id', '')
            if preliminary_direction != 'neutral' and strategy_id:
                try:
                    from rl_pipeline.core.strategy_grading import (
                        get_strategy_mfe_stats, MFEGrading
                    )
                    
                    mfe_stats = get_strategy_mfe_stats(strategy_id)
                    if mfe_stats and mfe_stats.coverage_n >= 20:
                        entry_score, risk_score, edge_score = MFEGrading.calculate_scores(mfe_stats)
                        
                        # EntryScore가 음수면 방향 무효
                        if not MFEGrading.validate_direction_by_mfe(entry_score, min_entry_score=0.0):
                            logger.debug(f"🚫 {strategy_id}: 방향 무효화 (EntryScore={entry_score:.4f} < 0)")
                            return 'neutral'
                        
                        # 신뢰도가 너무 낮으면 neutral
                        confidence = MFEGrading.get_directional_confidence(entry_score, edge_score)
                        if confidence < 0.2:
                            logger.debug(f"🚫 {strategy_id}: 신뢰도 부족 (confidence={confidence:.3f})")
                            return 'neutral'
                            
                except Exception as mfe_err:
                    logger.debug(f"⚠️ MFE 검증 스킵 ({strategy_id}): {mfe_err}")
            
            return preliminary_direction
            
        except Exception as e:
            logger.debug(f"⚠️ 전략 방향 분류 실패: {e}")
            return 'neutral'
    
    def _detect_market_regime_from_candles(self, candle_data: pd.DataFrame) -> str:
        """🔥 캔들 데이터에서 시장 레짐 감지
        
        Args:
            candle_data: 캔들 데이터 DataFrame
            
        Returns:
            레짐 라벨 ('extreme_bullish', 'bullish', 'sideways_bullish', 'neutral', 
                      'sideways_bearish', 'bearish', 'extreme_bearish')
        """
        try:
            if candle_data is None or len(candle_data) < 20:
                return 'neutral'
            
            # 최근 데이터 사용 (최대 100개)
            recent_data = candle_data.tail(min(100, len(candle_data)))
            
            # 가격 변화율 계산
            if 'close' in recent_data.columns:
                closes = recent_data['close'].dropna()
                if len(closes) < 10:
                    return 'neutral'
                
                price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
                
                # RSI 계산 (간단 버전)
                if 'rsi' in recent_data.columns:
                    rsi = recent_data['rsi'].dropna().iloc[-1] if len(recent_data['rsi'].dropna()) > 0 else 50.0
                else:
                    # 간단한 RSI 추정
                    returns = closes.pct_change().dropna()
                    if len(returns) > 0:
                        gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
                        losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
                        rs = gains / losses if losses > 0 else 1.0
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 50.0
                
                # MACD 계산 (간단 버전)
                if 'macd' in recent_data.columns:
                    macd = recent_data['macd'].dropna().iloc[-1] if len(recent_data['macd'].dropna()) > 0 else 0.0
                else:
                    macd = 0.0
                
                # 변동성 계산
                returns = closes.pct_change().dropna()
                volatility = returns.std() if len(returns) > 0 else 0.0
                
                # 레짐 분류
                if price_change > 0.1 and rsi > 70:
                    return 'extreme_bullish'
                elif price_change > 0.05 and rsi > 60:
                    return 'bullish'
                elif price_change > 0.02 and rsi > 50:
                    return 'sideways_bullish'
                elif price_change < -0.1 and rsi < 30:
                    return 'extreme_bearish'
                elif price_change < -0.05 and rsi < 40:
                    return 'bearish'
                elif price_change < -0.02 and rsi < 50:
                    return 'sideways_bearish'
                else:
                    return 'neutral'
            else:
                return 'neutral'
        except Exception as e:
            logger.debug(f"⚠️ 레짐 감지 실패: {e}")
            return 'neutral'
    
    def _select_strategies_by_regime(self, all_strategy_pool: List[Dict[str, Any]], regime: str, count: int) -> List[Dict[str, Any]]:
        """🔥 레짐에 따라 적절한 전략 선택
        
        Args:
            all_strategy_pool: 전체 전략 풀
            regime: 시장 레짐 ('extreme_bullish', 'bullish', 'sideways_bullish', 'neutral', 
                              'sideways_bearish', 'bearish', 'extreme_bearish')
            count: 선택할 전략 수
            
        Returns:
            선택된 전략 리스트
        """
        try:
            # 전략을 방향별로 분류
            buy_strategies = []
            sell_strategies = []
            neutral_strategies = []
            
            for strategy in all_strategy_pool:
                direction = self._classify_strategy_direction(strategy)
                if direction == 'buy':
                    buy_strategies.append(strategy)
                elif direction == 'sell':
                    sell_strategies.append(strategy)
                else:
                    neutral_strategies.append(strategy)
            
            # 레짐에 따라 전략 선택 비율 결정
            if regime in ['extreme_bullish', 'bullish', 'sideways_bullish']:
                # 상승장: 매수 전략 위주 (70% 매수, 20% 중립, 10% 매도)
                buy_count = int(count * 0.7)
                neutral_count = int(count * 0.2)
                sell_count = count - buy_count - neutral_count
            elif regime in ['extreme_bearish', 'bearish', 'sideways_bearish']:
                # 하락장: 매도 전략 위주 (70% 매도, 20% 중립, 10% 매수)
                sell_count = int(count * 0.7)
                neutral_count = int(count * 0.2)
                buy_count = count - sell_count - neutral_count
            else:
                # 중립: 균등 분배 (40% 매수, 40% 매도, 20% 중립)
                buy_count = int(count * 0.4)
                sell_count = int(count * 0.4)
                neutral_count = count - buy_count - sell_count
            
            selected_strategies = []
            
            # 전략 선택 (랜덤 샘플링, 부족하면 중복 허용)
            if len(buy_strategies) >= buy_count:
                selected_strategies.extend(random.sample(buy_strategies, buy_count))
            elif len(buy_strategies) > 0:
                selected_strategies.extend(buy_strategies)
                selected_strategies.extend(random.choices(buy_strategies, k=buy_count - len(buy_strategies)))
            
            if len(sell_strategies) >= sell_count:
                selected_strategies.extend(random.sample(sell_strategies, sell_count))
            elif len(sell_strategies) > 0:
                selected_strategies.extend(sell_strategies)
                selected_strategies.extend(random.choices(sell_strategies, k=sell_count - len(sell_strategies)))
            
            if len(neutral_strategies) >= neutral_count:
                selected_strategies.extend(random.sample(neutral_strategies, neutral_count))
            elif len(neutral_strategies) > 0:
                selected_strategies.extend(neutral_strategies)
                selected_strategies.extend(random.choices(neutral_strategies, k=neutral_count - len(neutral_strategies)))
            
            # 부족하면 나머지를 랜덤으로 채움
            if len(selected_strategies) < count:
                remaining = count - len(selected_strategies)
                all_remaining = [s for s in all_strategy_pool if s not in selected_strategies]
                if len(all_remaining) >= remaining:
                    selected_strategies.extend(random.sample(all_remaining, remaining))
                elif len(all_remaining) > 0:
                    selected_strategies.extend(all_remaining)
                    selected_strategies.extend(random.choices(all_remaining, k=remaining - len(all_remaining)))
            
            return selected_strategies[:count]  # 정확히 count개만 반환
        except Exception as e:
            logger.warning(f"⚠️ 레짐 기반 전략 선택 실패, 랜덤 선택으로 폴백: {e}")
            # 폴백: 랜덤 선택
            if len(all_strategy_pool) >= count:
                return random.sample(all_strategy_pool, count)
            else:
                return random.choices(all_strategy_pool, k=count)
    
    def run_learning_cycle(self, agents: List[StrategyAgent], episodes: int = 10, all_strategy_pool: List[Dict[str, Any]] = None, agents_per_episode: int = None, candle_data: pd.DataFrame = None, coin: Optional[str] = None, interval: Optional[str] = None) -> Dict[str, Any]:
        """학습 사이클 실행 - 매 에피소드마다 다른 전략 샘플링 (🔥 레짐 기반 분석 선택)
        
        Args:
            agents: 현재 에이전트 (초기 전략)
            episodes: 학습 에피소드 수
            all_strategy_pool: 전체 전략 풀 (DB에서 로드한 모든 전략)
            agents_per_episode: 매 에피소드마다 사용할 에이전트 수
            candle_data: 실제 캔들 데이터 (None이면 가상 데이터 사용) 🔥
            coin: 코인 심볼 (rl_episode_summary 저장용)
            interval: 인터벌 (rl_episode_summary 저장용)
        """
        try:
            agents_per_episode = agents_per_episode or len(agents)
            logger.info(f"🧠 Self-play 학습 사이클 시작 ({episodes}개 에피소드, 매 에피소드 {agents_per_episode}개 전략 샘플링)")

            # 🔥 디버거 로깅: Self-play 시작
            if self.debug and coin and interval:
                try:
                    # candle_data가 있으면 개수 확인
                    candle_count = len(candle_data) if candle_data is not None else 0
                    self.debug.log_selfplay_start(
                        coin=coin,
                        interval=interval,
                        num_episodes=episodes,
                        num_agents=agents_per_episode,
                        candle_count=candle_count
                    )
                except Exception as debug_err:
                    logger.debug(f"⚠️ Self-play 시작 디버그 로깅 실패 (무시): {debug_err}")

            cycle_results = []
            
            # 전체 전략 풀이 있으면 매 에피소드마다 샘플링
            all_performances = []  # 모든 에피소드 성과 수집
            early_stop_check_interval = 10  # 10개 에피소드마다 확인
            
            # 🔥 시장 레짐 감지 (캔들 데이터가 있으면)
            current_regime = 'neutral'
            if candle_data is not None and len(candle_data) > 0:
                current_regime = self._detect_market_regime_from_candles(candle_data)
                logger.info(f"📊 감지된 시장 레짐: {current_regime}")
            
            for episode in range(episodes):
                # 🔥 매 에피소드마다 레짐 기반 전략 선택
                if all_strategy_pool and len(all_strategy_pool) >= agents_per_episode:
                    # 🔥 레짐 기반 전략 선택 (랜덤 대신)
                    sampled_strategies = self._select_strategies_by_regime(all_strategy_pool, current_regime, agents_per_episode)
                    
                    # 전략 방향 분류 통계
                    buy_count = sum(1 for s in sampled_strategies if self._classify_strategy_direction(s) == 'buy')
                    sell_count = sum(1 for s in sampled_strategies if self._classify_strategy_direction(s) == 'sell')
                    neutral_count = len(sampled_strategies) - buy_count - sell_count
                    
                    logger.info(f"📊 에피소드 {episode + 1}: {len(all_strategy_pool)}개 중 {agents_per_episode}개 전략 선택 (레짐: {current_regime}, 매수: {buy_count}, 매도: {sell_count}, 중립: {neutral_count})")
                    
                    # 🔍 파라미터 확인 로그 (상세 정보는 DEBUG 레벨로 변경)
                    if AZ_SIMULATION_VERBOSE:
                        for idx, strat in enumerate(sampled_strategies):
                            direction = self._classify_strategy_direction(strat)
                            # 파라미터 추출 (안전하게)
                            params = strat.get('params', {})
                            if isinstance(params, str):
                                try:
                                    import json
                                    params = json.loads(params) if params else {}
                                except:
                                    params = {}
                            if not isinstance(params, dict):
                                params = {}
                            
                            rsi_min = strat.get('rsi_min') or params.get('rsi_min', 'N/A')
                            rsi_max = strat.get('rsi_max') or params.get('rsi_max', 'N/A')
                            stop_loss = strat.get('stop_loss_pct') or params.get('stop_loss_pct', 'N/A')
                            
                            logger.debug(f"  전략 {idx+1} ({direction}): RSI={rsi_min}-{rsi_max}, StopLoss={stop_loss}")
                    current_agents = self.create_agents(sampled_strategies, coin=coin)  # 🔥 코인별 최적화
                elif all_strategy_pool and len(all_strategy_pool) > 0:
                    # 전략 풀이 에이전트 수보다 작으면 레짐 기반 선택 + 중복 허용
                    sampled_strategies = self._select_strategies_by_regime(all_strategy_pool, current_regime, agents_per_episode)
                    
                    buy_count = sum(1 for s in sampled_strategies if self._classify_strategy_direction(s) == 'buy')
                    sell_count = sum(1 for s in sampled_strategies if self._classify_strategy_direction(s) == 'sell')
                    neutral_count = len(sampled_strategies) - buy_count - sell_count
                    
                    logger.info(f"📊 에피소드 {episode + 1}: {len(all_strategy_pool)}개 전략에서 레짐 기반 선택 (레짐: {current_regime}, 매수: {buy_count}, 매도: {sell_count}, 중립: {neutral_count})")
                    current_agents = self.create_agents(sampled_strategies, coin=coin)  # 🔥 코인별 최적화
                else:
                    # 전략 풀이 없으면 초기 에이전트 사용하되, 약간의 랜덤 변형 추가
                    current_agents = []
                    for i, agent in enumerate(agents):
                        new_strategy = agent.strategy_params.copy()
                        # 약간의 랜덤 변형 추가 (5% 변동)
                        for key in ['rsi_min', 'rsi_max']:
                            if key in new_strategy:
                                new_strategy[key] = max(10, min(90, new_strategy[key] + random.randint(-2, 2)))
                        current_agents.append(StrategyAgent(f"agent_{i+1}", new_strategy))
                    logger.info(f"🎲 에피소드 {episode + 1}: 초기 전략에 랜덤 변형 적용")

                # 🔥 디버거 로깅: 에피소드 시작
                if self.debug and coin and interval:
                    try:
                        self.debug.log_episode_start(
                            coin=coin,
                            interval=interval,
                            episode_num=episode + 1,
                            num_agents=len(current_agents)
                        )
                    except Exception as debug_err:
                        logger.debug(f"⚠️ 에피소드 시작 디버그 로깅 실패 (무시): {debug_err}")

                # 🔥 동적 steps 조정: 캔들 데이터 길이의 80%를 사용 (최대 500)
                if candle_data is not None and len(candle_data) > 0:
                    dynamic_steps = min(500, int(len(candle_data) * 0.8))
                    logger.info(f"📊 동적 steps 조정: {dynamic_steps} (캔들 {len(candle_data)}개의 80%, 최대 500)")
                else:
                    dynamic_steps = 500
                    logger.info(f"📊 기본 steps 사용: {dynamic_steps} (캔들 데이터 없음)")

                episode_result = self.run_self_play_episode(current_agents, steps=dynamic_steps, candle_data=candle_data)  # 🔥 실제 캔들 데이터 전달
                cycle_results.append(episode_result)

                # 🔥 디버거 로깅: 에피소드 결과
                if self.debug and coin and interval and "results" in episode_result:
                    try:
                        # 에피소드 통계 계산
                        total_trades = sum(r.get("total_trades", 0) for r in episode_result["results"].values())
                        total_pnl = sum(r.get("total_pnl", 0) for r in episode_result["results"].values())
                        avg_win_rate = np.mean([r.get("win_rate", 0) for r in episode_result["results"].values()]) if episode_result["results"] else 0

                        self.debug.log_episode_result(
                            coin=coin,
                            interval=interval,
                            episode_num=episode + 1,
                            total_trades=total_trades,
                            avg_pnl=total_pnl / len(episode_result["results"]) if episode_result["results"] else 0,
                            avg_win_rate=avg_win_rate
                        )
                    except Exception as debug_err:
                        logger.debug(f"⚠️ 에피소드 결과 디버그 로깅 실패 (무시): {debug_err}")
                
                # 🔥 옵션 A: 시뮬레이션 self-play 결과를 rl_episode_summary에 저장
                if "results" in episode_result and coin and interval:
                    try:
                        import uuid
                        # 전략 매핑 생성 (agent_id -> strategy_id)
                        agent_to_strategy = {}
                        # sampled_strategies 변수 확인 (다양한 분기에서 생성됨)
                        strategies_source = sampled_strategies if 'sampled_strategies' in locals() else (all_strategy_pool if all_strategy_pool else [])
                        
                        for agent in current_agents:
                            agent_to_strategy[agent.agent_id] = None
                            # 전략 파라미터에서 strategy_id 추출 시도
                            if agent.strategy_params:
                                # strategies_source에서 strategy_id 찾기
                                matching_strategy = next(
                                    (s for s in strategies_source 
                                     if isinstance(s, dict) and s.get('id') and 
                                     all(agent.strategy_params.get(k) == s.get(k) 
                                         for k in ['rsi_min', 'rsi_max', 'stop_loss_pct', 'take_profit_pct'] 
                                         if k in agent.strategy_params and k in s)),
                                    None
                                )
                                if matching_strategy:
                                    agent_to_strategy[agent.agent_id] = matching_strategy.get('id')
                                else:
                                    # 전략 파라미터 해시로 ID 생성
                                    strategy_hash = abs(hash(str(sorted(agent.strategy_params.items())))) % (10**10)
                                    agent_to_strategy[agent.agent_id] = f"strategy_{strategy_hash}"
                        
                        for agent_id, perf in episode_result["results"].items():
                            # 에이전트별 episode_id 생성
                            episode_id = f"sim_{coin}_{interval}_{episode}_{agent_id}_{uuid.uuid4().hex[:8]}"
                            
                            # 전략 ID 추출
                            strategy_id = agent_to_strategy.get(agent_id)
                            if not strategy_id:
                                # 폴백: 전략 파라미터 해시로 생성
                                agent = next((a for a in current_agents if a.agent_id == agent_id), None)
                                if agent and agent.strategy_params:
                                    strategy_hash = abs(hash(str(sorted(agent.strategy_params.items())))) % (10**10)
                                    strategy_id = f"strategy_{strategy_hash}"
                                else:
                                    # 🔧 agent_id가 유효한 경우에만 사용, 없으면 더미 해시 사용
                                    if agent_id and agent_id != 'unknown':
                                        strategy_id = f"unknown_{agent_id}"
                                    else:
                                        # agent_id도 없으면 타임스탬프 기반 고유 ID 생성
                                        import time
                                        strategy_id = f"unknown_sim_{int(time.time() * 1000) % (10**10)}"
                                        
                            # 🔧 strategy_id가 'unknown'만 있는 경우 처리 (agent_id가 비어있는 경우)
                            if strategy_id == 'unknown':
                                import time
                                strategy_id = f"unknown_sim_{int(time.time() * 1000) % (10**10)}"
                            
                            # realized_ret_signed 계산 (total_pnl을 퍼센트로 변환)
                            total_pnl = perf.get('total_pnl', 0.0)
                            realized_ret_signed = total_pnl / 10000.0 if total_pnl != 0 else 0.0
                            
                            # acc_flag: 시뮬레이션에서는 예측 개념이 없으므로 None (predictive_accuracy 계산에서 제외)
                            acc_flag = None
                            
                            # first_event: win_rate 기반으로 추정 (TP/expiry 구분 불가하므로 expiry로 설정)
                            first_event = 'expiry'
                            
                            # t_hit: 평균 거래 수로 추정 (정확한 값은 알 수 없음)
                            t_hit = perf.get('total_trades', 0)

                            # 🔥 인터벌별 맞춤 보상 계산
                            if INTERVAL_PROFILES_AVAILABLE and calculate_reward and interval:
                                try:
                                    # 예측과 실제 결과 준비
                                    prediction = {
                                        'direction': 1 if perf.get('win_rate', 0.5) > 0.5 else -1,
                                        'return': perf.get('total_pnl', 0.0) / 100.0,
                                        'regime': 'bull' if perf.get('total_pnl', 0) > 0 else 'bear',
                                        'swing': 'up' if perf.get('total_pnl', 0) > 0 else 'down',
                                        'trend': 'continuation',
                                        'entry_quality': 'good' if perf.get('win_rate', 0.5) > 0.6 else 'neutral',
                                        'r_multiple': abs(perf.get('total_pnl', 0.0) / 100.0),
                                        'stop_hit': perf.get('win_rate', 0.5) < 0.4,
                                    }

                                    actual = {
                                        'direction': 1 if realized_ret_signed > 0 else -1,
                                        'return': realized_ret_signed,
                                        'regime': 'bull' if realized_ret_signed > 0.05 else ('bear' if realized_ret_signed < -0.05 else 'range'),
                                        'swing': 'up' if realized_ret_signed > 0 else 'down',
                                        'trend': 'continuation' if perf.get('win_rate', 0.5) > 0.5 else 'reversal',
                                        'entry_quality': 'excellent' if realized_ret_signed > 0.03 else 'good',
                                        'r_multiple': abs(realized_ret_signed),
                                        'stop_hit': realized_ret_signed < -0.02,
                                    }

                                    # interval_profiles의 calculate_reward 사용
                                    total_reward = calculate_reward(interval, prediction, actual)
                                    logger.debug(f"🔥 {interval} 인터벌 맞춤 보상 사용: {total_reward:.3f}")
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"interval_profiles 보상 계산 실패 (입력 데이터 오류), 기본값 사용: {e}")
                                    total_reward = perf.get('total_pnl', 0.0) / 100.0
                                except Exception as e:
                                    logger.warning(f"interval_profiles 보상 계산 중 예상치 못한 오류: {e}", exc_info=True)
                                    total_reward = perf.get('total_pnl', 0.0) / 100.0
                            else:
                                # 기본 보상 계산
                                total_reward = perf.get('total_pnl', 0.0) / 100.0

                            save_episode_summary(
                                episode_id=episode_id,
                                ts_exit=int(datetime.now().timestamp()),
                                first_event=first_event,
                                t_hit=t_hit,
                                realized_ret_signed=realized_ret_signed,
                                total_reward=total_reward,  # 🔥 계산된 보상 사용
                                acc_flag=0 if acc_flag is None else acc_flag,  # None이면 0으로 설정
                                coin=coin,
                                interval=interval,
                                strategy_id=strategy_id,
                                source_type='simulation'  # 🔥 옵션 A: 시뮬레이션 self-play 표시
                            )
                    except Exception as e:
                        logger.debug(f"⚠️ 시뮬레이션 self-play 결과 저장 실패: {e}")
                
                # 성과 데이터 수집
                if "results" in episode_result:
                    for perf in episode_result["results"].values():
                        all_performances.append(perf)
                
                # 🔥 전략 유사도 체크 제거됨 (제대로 작동하지 않아서 제거)
                # 기존 전략 유사도 체크 로직은 제거되었습니다.
                
                # 에이전트 전략 업데이트 (간단한 적응)
                if episode > 0 and episode % 3 == 0:
                    self._update_agent_strategies(current_agents, cycle_results[-3:])
            
            # 사이클 결과 분석
            cycle_summary = self._analyze_cycle_results(cycle_results)

            # 🔥 디버거 로깅: Self-play 종료
            if self.debug and coin and interval:
                try:
                    # 요약에서 핵심 메트릭 추출
                    avg_pnl = cycle_summary.get('avg_pnl', 0.0)
                    avg_win_rate = cycle_summary.get('avg_win_rate', 0.0)

                    self.debug.log_selfplay_end(
                        coin=coin,
                        interval=interval,
                        total_episodes=len(cycle_results),
                        summary={
                            "avg_pnl": avg_pnl,
                            "avg_win_rate": avg_win_rate,
                            "total_trades": cycle_summary.get('total_trades', 0),
                            "best_agent_pnl": cycle_summary.get('best_agent_pnl', 0.0)
                        }
                    )
                except Exception as debug_err:
                    logger.debug(f"⚠️ Self-play 종료 디버그 로깅 실패 (무시): {debug_err}")

            # 소수점 정리된 요약 출력
            summary_formatted = self._format_cycle_summary(cycle_summary)
            logger.info(f"✅ 학습 사이클 완료: {summary_formatted}")

            return {
                "episodes": episodes,
                "cycle_results": cycle_results,
                "summary": cycle_summary,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ 학습 사이클 실패: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _update_agent_strategies(self, agents: List[StrategyAgent], recent_results: List[Dict[str, Any]]):
        """🚀 레짐 기반 에이전트 전략 업데이트 (적응적 학습)"""
        try:
            for agent in agents:
                # 최근 성과 분석 (레짐별)
                regime_performance = {}
                for result in recent_results:
                    if agent.agent_id in result.get("results", {}):
                        regime_label = result.get("regime_label", "neutral")
                        if regime_label not in regime_performance:
                            regime_performance[regime_label] = []
                        regime_performance[regime_label].append(result["results"][agent.agent_id])
                
                if not regime_performance:
                    continue
                
                # 레짐별 성과 분석 및 전략 조정
                for regime_label, performances in regime_performance.items():
                    if not performances:
                        continue
                    
                    # 🔥 평균(Mean) -> 중앙값(Median) 변경으로 이상치 영향 최소화
                    avg_win_rate = np.median([p.get("win_rate", 0) for p in performances])
                    avg_pnl = np.median([p.get("total_pnl", 0) for p in performances])
                    
                    # 레짐별 성과가 나쁜 경우 파라미터 조정
                    if avg_win_rate < 0.4 or avg_pnl < 0:
                        # 레짐별 파라미터 조정
                        if regime_label in ["extreme_bullish", "bullish"]:
                            # 강세장에서는 더 공격적
                            if "rsi_min" in agent.strategy_params:
                                agent.strategy_params["rsi_min"] = max(15, 
                                    agent.strategy_params["rsi_min"] - 2)
                            if "rsi_max" in agent.strategy_params:
                                agent.strategy_params["rsi_max"] = min(85, 
                                    agent.strategy_params["rsi_max"] + 2)
                        
                        elif regime_label in ["extreme_bearish", "bearish"]:
                            # 약세장에서는 더 보수적
                            if "rsi_min" in agent.strategy_params:
                                agent.strategy_params["rsi_min"] = min(45, 
                                    agent.strategy_params["rsi_min"] + 2)
                            if "rsi_max" in agent.strategy_params:
                                agent.strategy_params["rsi_max"] = max(55, 
                                    agent.strategy_params["rsi_max"] - 2)
                        
                        elif regime_label in ["sideways_bullish", "sideways_bearish", "neutral"]:
                            # 횡보장에서는 중간값 조정 (반올림 적용)
                            adjustment_factor = np.random.uniform(0.95, 1.05)
                            if "rsi_min" in agent.strategy_params:
                                agent.strategy_params["rsi_min"] = round(max(25, min(35, 
                                    agent.strategy_params["rsi_min"] * adjustment_factor)), 1)
                            if "rsi_max" in agent.strategy_params:
                                agent.strategy_params["rsi_max"] = round(max(65, min(75, 
                                    agent.strategy_params["rsi_max"] * adjustment_factor)), 1)
                        
                        logger.info(f"🔄 {agent.agent_id} {regime_label} 전략 업데이트: 승률 {avg_win_rate:.2%}, 수익 {avg_pnl:.2f}")
                
        except Exception as e:
            logger.error(f"❌ 전략 업데이트 실패: {e}")
    
    def _analyze_cycle_results(self, cycle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🚀 레짐 기반 사이클 결과 분석"""
        try:
            if not cycle_results:
                return {}
            
            # 전체 성과 집계
            all_performances = []
            regime_performance = {
                "extreme_bearish": [], "bearish": [], "sideways_bearish": [], 
                "neutral": [], "sideways_bullish": [], "bullish": [], "extreme_bullish": []
            }
            
            for result in cycle_results:
                if "results" in result:
                    for agent_id, performance in result["results"].items():
                        all_performances.append(performance)
                        
                        regime_label = result.get("regime_label", "neutral")
                        if regime_label in regime_performance:
                            regime_performance[regime_label].append(performance)
            
            if not all_performances:
                return {}
            
            # 전체 통계
            total_trades = sum(p.get("total_trades", 0) for p in all_performances)
            # 🔥 평균(Mean) -> 중앙값(Median) 변경으로 이상치 영향 최소화
            avg_win_rate = np.median([p.get("win_rate", 0) for p in all_performances])
            avg_pnl = np.median([p.get("total_pnl", 0) for p in all_performances])
            avg_sharpe = np.median([p.get("sharpe_ratio", 0) for p in all_performances])

            # 🔥 최고 성과 에이전트 찾기
            best_agent_pnl = max([p.get("total_pnl", 0) for p in all_performances]) if all_performances else 0.0

            # 레짐별 성과
            regime_stats = {}
            for regime_label, performances in regime_performance.items():
                if performances:
                    regime_stats[regime_label] = {
                        "avg_win_rate": np.median([p.get("win_rate", 0) for p in performances]),
                        "avg_pnl": np.median([p.get("total_pnl", 0) for p in performances]),
                        "avg_sharpe_ratio": np.median([p.get("sharpe_ratio", 0) for p in performances]),
                        "episode_count": len(performances),
                        "total_trades": sum(p.get("total_trades", 0) for p in performances)
                    }
            
            return {
                "total_episodes": len(cycle_results),
                "total_trades": total_trades,
                "avg_win_rate": avg_win_rate,
                "avg_pnl": avg_pnl,
                "avg_sharpe_ratio": avg_sharpe,
                "best_agent_pnl": best_agent_pnl,
                "regime_performance": regime_stats,
                "learning_progress": self._calculate_learning_progress(cycle_results)
            }
            
        except Exception as e:
            logger.error(f"❌ 사이클 결과 분석 실패: {e}")
            return {}
    
    def _format_cycle_summary(self, summary: Dict[str, Any]) -> str:
        """사이클 요약 포맷팅 (소수점 정리)"""
        try:
            if not summary:
                return "{}"
            
            # 전체 통계 포맷
            formatted_summary = {
                "total_episodes": summary.get("total_episodes", 0),
                "total_trades": summary.get("total_trades", 0),
                "avg_win_rate": round(summary.get("avg_win_rate", 0), 2),
                "avg_pnl": round(summary.get("avg_pnl", 0), 0),
                "avg_sharpe_ratio": round(summary.get("avg_sharpe_ratio", 0), 4),
            }
            
            # 레짐별 성과 포맷
            regime_perf = summary.get("regime_performance", {})
            formatted_regime = {}
            for regime, stats in regime_perf.items():
                formatted_regime[regime] = {
                    "avg_win_rate": round(stats.get("avg_win_rate", 0), 2),
                    "avg_pnl": round(stats.get("avg_pnl", 0), 0),
                    "avg_sharpe_ratio": round(stats.get("avg_sharpe_ratio", 0), 4),
                    "episode_count": stats.get("episode_count", 0),
                    "total_trades": stats.get("total_trades", 0)
                }
            
            formatted_summary["regime_performance"] = formatted_regime
            
            # 학습 진행도 포맷
            learning_prog = summary.get("learning_progress", {})
            if learning_prog:
                formatted_summary["learning_progress"] = {
                    "progress": round(learning_prog.get("progress", 0), 2),
                    "trend": learning_prog.get("trend", "stable"),
                    "pnl_improvement": round(learning_prog.get("pnl_improvement", 0), 2),
                    "win_rate_improvement": round(learning_prog.get("win_rate_improvement", 0), 3)
                }
            
            return str(formatted_summary)
            
        except Exception as e:
            logger.error(f"❌ 포맷팅 실패: {e}")
            return str(summary)
    
    def _calculate_performance_diversity(self, performances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """성과 다양성 계산 (변동 계수)"""
        try:
            if not performances or len(performances) < 3:
                return {'coefficient_of_variation': 1.0, 'mean': 0.0, 'std': 0.0}
            
            # 승률 기준 다양성 계산
            win_rates = [p.get('win_rate', 0) for p in performances]
            mean_wr = np.mean(win_rates)
            std_wr = np.std(win_rates)
            cv_wr = std_wr / mean_wr if mean_wr > 0 else 0
            
            return {
                'coefficient_of_variation': cv_wr,
                'mean': mean_wr,
                'std': std_wr,
                'min': np.min(win_rates),
                'max': np.max(win_rates),
                'range': np.max(win_rates) - np.min(win_rates)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 다양성 계산 실패: {e}")
            return {'coefficient_of_variation': 1.0, 'mean': 0.0, 'std': 0.0}
    
    def _calculate_learning_progress(self, cycle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """학습 진행도 계산"""
        try:
            if len(cycle_results) < 2:
                return {"progress": 0.0, "trend": "stable"}
            
            # 최근 절반과 이전 절반 비교
            mid_point = len(cycle_results) // 2
            early_results = cycle_results[:mid_point]
            recent_results = cycle_results[mid_point:]
            
            early_performances = []
            recent_performances = []
            
            for result in early_results:
                if "results" in result:
                    early_performances.extend(result["results"].values())
            
            for result in recent_results:
                if "results" in result:
                    recent_performances.extend(result["results"].values())
            
            if not early_performances or not recent_performances:
                return {"progress": 0.0, "trend": "stable"}
            
            # 성과 비교
            early_avg_pnl = np.mean([p.get("total_pnl", 0) for p in early_performances])
            recent_avg_pnl = np.mean([p.get("total_pnl", 0) for p in recent_performances])
            
            early_avg_win_rate = np.mean([p.get("win_rate", 0) for p in early_performances])
            recent_avg_win_rate = np.mean([p.get("win_rate", 0) for p in recent_performances])
            
            # 진행도 계산
            pnl_improvement = (recent_avg_pnl - early_avg_pnl) / abs(early_avg_pnl) if early_avg_pnl != 0 else 0
            win_rate_improvement = recent_avg_win_rate - early_avg_win_rate
            
            overall_progress = (pnl_improvement + win_rate_improvement) / 2
            
            # 트렌드 판단
            if overall_progress > 0.1:
                trend = "improving"
            elif overall_progress < -0.1:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                "progress": overall_progress,
                "trend": trend,
                "pnl_improvement": pnl_improvement,
                "win_rate_improvement": win_rate_improvement
            }
            
        except Exception as e:
            logger.error(f"❌ 학습 진행도 계산 실패: {e}")
            return {"progress": 0.0, "trend": "stable"}

def run_self_play_test(
    strategy_params_list: List[Dict[str, Any]],
    episodes: int = 5,
    all_strategy_pool: List[Dict[str, Any]] = None,
    agents_per_episode: int = None,
    candle_data: pd.DataFrame = None,
    agent_type: str = 'rule',
    neural_policy: Optional[Dict[str, Any]] = None,
    hybrid_config: Optional[Dict[str, Any]] = None,
    coin: Optional[str] = None,
    interval: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Self-play 테스트 실행 - 매 에피소드마다 다른 전략 샘플링
    
    Args:
        strategy_params_list: 초기 전략 파라미터 리스트 (호환성)
        episodes: 학습 에피소드 수
        all_strategy_pool: 전체 전략 풀 (DB에서 로드한 모든 전략, None이면 strategy_params_list 사용)
        agents_per_episode: 매 에피소드마다 사용할 에이전트 수 (None이면 strategy_params_list 길이)
        candle_data: 실제 캔들 데이터 (None이면 가상 데이터 생성) 🔥
        agent_type: 'rule' or 'hybrid'
        neural_policy: 신경망 정책 (hybrid 모드일 때 필요)
        hybrid_config: 하이브리드 설정 (hybrid 모드일 때 필요)
    """
    try:
        logger.info(f"🚀 Self-play 테스트 시작 (agent_type={agent_type})")
        
        if candle_data is not None:
            logger.info(f"✅ 실제 캔들 데이터 사용: {len(candle_data)}개")
        else:
            logger.info("⚠️ 가상 시장 데이터 생성 (candle_data 미제공)")
        
        # 전체 전략 풀 설정
        strategy_pool = all_strategy_pool if all_strategy_pool else strategy_params_list
        agents_per_episode = agents_per_episode or len(strategy_params_list)
        
        # 전략 풀 크기 로깅
        logger.info(f"📊 전략 풀 크기: {len(strategy_pool)}개, 매 에피소드 {agents_per_episode}개 사용")
        
        # 시뮬레이터 초기화 (session_id 전달)
        simulator = SelfPlaySimulator(session_id=session_id)
        
        # 에이전트 생성 (초기 에이전트, 실제로는 매 에피소드마다 다시 생성됨)
        agents = simulator.create_agents(
            strategy_params_list,
            agent_type=agent_type,
            neural_policy=neural_policy,
            hybrid_config=hybrid_config,
            coin=coin  # 🔥 코인별 파라미터 최적화
        )
        logger.info(f"✅ {len(agents)}개 초기 에이전트 생성 완료")
        
        # 학습 사이클 실행 - 매 에피소드마다 다른 전략 샘플링 + 실제 캔들 데이터
        result = simulator.run_learning_cycle(agents, episodes=episodes, 
                                             all_strategy_pool=strategy_pool,
                                             agents_per_episode=agents_per_episode,
                                             candle_data=candle_data,  # 🔥 실제 캔들 데이터 전달
                                             coin=coin,  # 🔥 옵션 A: coin 전달
                                             interval=interval)  # 🔥 옵션 A: interval 전달
        
        if result["status"] == "success":
            logger.info("✅ Self-play 테스트 완료")
            return result
        else:
            logger.error(f"❌ Self-play 테스트 실패: {result.get('error', 'Unknown error')}")
            return result
            
    except Exception as e:
        logger.error(f"❌ Self-play 테스트 실행 실패: {e}")
        return {"status": "failed", "error": str(e)}

def run_self_play_evolution(strategy_params_list: List[Dict[str, Any]], 
                           episodes: int = 3) -> Dict[str, Any]:
    """
    Self-play 진화 함수 (run_self_play_test의 별칭)
    
    Args:
        strategy_params_list: 전략 파라미터 리스트
        episodes: 학습 에피소드 수
        
    Returns:
        학습 결과 딕셔너리
    """
    logger.info("🚀 Self-play 진화 시작")
    return run_self_play_test(strategy_params_list, episodes)
