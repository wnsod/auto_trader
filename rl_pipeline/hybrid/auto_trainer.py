"""
자동 학습 및 검증 모듈
Self-play 결과를 수집하여 자동으로 학습 및 평가
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from rl_pipeline.hybrid.trainer_jax import PPOTrainer
from rl_pipeline.hybrid.evaluator import (
    evaluate_ab,
    walk_forward_validation,
    multi_period_validation
)
from rl_pipeline.hybrid.validation_checker import (
    evaluate_validation_results,
    should_retrain,
    get_retrain_suggestions
)
from rl_pipeline.db.connection_pool import get_strategy_db_pool
from rl_pipeline.db.connection_pool import get_optimized_db_connection

# 증분 학습
from rl_pipeline.hybrid.incremental_trainer import (
    save_training_history,
    copy_parent_policy,
    train_strategies_incremental
)


def _create_adjusted_config(
    config_path: Optional[str],
    suggestions: Dict[str, Any],
    previous_attempts: int
) -> Optional[str]:
    """
    재학습 제안에 따라 하이퍼파라미터를 조정한 임시 설정 파일 생성
    
    Args:
        config_path: 원본 설정 파일 경로
        suggestions: 재학습 제안 딕셔너리
        previous_attempts: 재시도 횟수
    
    Returns:
        조정된 설정 파일 경로 (실패 시 None)
    """
    try:
        # 원본 설정 로드
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # 기본 설정
            config = {
                'train': {
                    'epochs': 30,
                    'batch_size': 4096,
                    'lr': 0.0003,
                    'hidden_dim': 128
                },
                'paths': {
                    'checkpoints': '/workspace/rl_pipeline/artifacts/checkpoints',
                    'db': '/workspace/data_storage/rl_strategies.db'
                }
            }
        
        # 하이퍼파라미터 조정
        train_config = config.setdefault('train', {})
        
        # 🔥 거래 0회 문제 해결: 학습률 및 탐험 증가
        if suggestions.get('adjust_learning_rate'):
            # 재시도 횟수에 따라 학습률 조정 (더 많은 재시도 = 더 작은 학습률)
            base_lr = train_config.get('lr', 0.0003)
            if previous_attempts == 1:
                # 첫 재시도: 학습률 약간 증가 (탐험 증가)
                adjusted_lr = base_lr * 1.5  # 0.0003 → 0.00045
            elif previous_attempts == 2:
                # 두 번째 재시도: 학습률 더 증가
                adjusted_lr = base_lr * 2.0  # 0.0003 → 0.0006
            else:
                # 세 번째 이상: 학습률 감소 (안정성 우선)
                adjusted_lr = base_lr * 0.5  # 0.0003 → 0.00015
            
            train_config['lr'] = adjusted_lr
            logger.info(f"🔧 학습률 조정: {base_lr:.6f} → {adjusted_lr:.6f} (재시도: {previous_attempts}회)")
        
        # 🔥 하이퍼파라미터 조정: 에포크 수 증가 (더 많은 학습)
        if suggestions.get('adjust_hyperparameters'):
            base_epochs = train_config.get('epochs', 30)
            # 재시도 횟수에 따라 에포크 수 증가
            adjusted_epochs = base_epochs + (previous_attempts * 10)  # 재시도마다 10 에포크 추가
            train_config['epochs'] = adjusted_epochs
            logger.info(f"🔧 에포크 수 조정: {base_epochs} → {adjusted_epochs} (재시도: {previous_attempts}회)")
        
        # 임시 설정 파일 저장
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_config_path = os.path.join(
            temp_dir,
            f"hybrid_config_adjusted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return temp_config_path
        
    except Exception as e:
        logger.warning(f"⚠️ 하이퍼파라미터 조정 실패 (원본 설정 사용): {e}")
        return None


def collect_selfplay_data_for_training(
    coin: str,
    interval: str,
    selfplay_result: Dict[str, Any],
    min_episodes: int = 10
) -> List[Dict[str, Any]]:
    """
    Self-play 결과에서 학습 데이터 수집
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        selfplay_result: Self-play 결과 딕셔너리
        min_episodes: 최소 에피소드 수
    
    Returns:
        학습용 에피소드 데이터 리스트
    """
    try:
        episodes_data = []
        
        # cycle_results에서 데이터 추출
        cycle_results = selfplay_result.get("cycle_results", [])
        
        # 🔥 개선: 예측 실현 Self-play 및 온라인 Self-play 결과도 변환하여 포함
        if not cycle_results or len(cycle_results) == 0:
            # 1순위: 온라인 Self-play 결과 확인
            try:
                from rl_pipeline.hybrid.online_data_converter import (
                    extract_online_selfplay_result,
                    convert_online_segments_to_cycle_results
                )
                
                online_segments = extract_online_selfplay_result(selfplay_result)
                if online_segments:
                    logger.info(f"📊 {coin}-{interval}: 온라인 Self-play 세그먼트 {len(online_segments)}개 발견, 변환 중...")
                    converted_cycles = convert_online_segments_to_cycle_results(online_segments)
                    if converted_cycles:
                        cycle_results = converted_cycles
                        logger.info(f"✅ {coin}-{interval}: 온라인 Self-play 결과 변환 완료 ({len(converted_cycles)}개 cycle)")
            except ImportError:
                logger.debug(f"⚠️ 온라인 데이터 변환 모듈 없음 (무시)")
            except Exception as e:
                logger.warning(f"⚠️ 온라인 Self-play 결과 변환 실패: {e}")
            
            # 2순위: 예측 실현 에피소드 확인 (온라인 결과가 없을 때)
            if not cycle_results:
                try:
                    from rl_pipeline.hybrid.predictive_data_converter import (
                        extract_predictive_episodes_from_selfplay_result,
                        convert_predictive_episodes_to_cycle_results
                    )
                    
                    predictive_episodes = extract_predictive_episodes_from_selfplay_result(selfplay_result)
                    if predictive_episodes:
                        logger.info(f"📊 {coin}-{interval}: 예측 실현 에피소드 {len(predictive_episodes)}개 발견, 변환 중...")
                        converted_cycles = convert_predictive_episodes_to_cycle_results(predictive_episodes)
                        if converted_cycles:
                            cycle_results = converted_cycles
                            logger.info(f"✅ {coin}-{interval}: 예측 실현 결과 변환 완료 ({len(converted_cycles)}개 cycle)")
                except ImportError:
                    logger.debug(f"⚠️ 예측 데이터 변환 모듈 없음 (무시)")
                except Exception as e:
                    logger.warning(f"⚠️ 예측 실현 결과 변환 실패: {e}")
        
        if not cycle_results:
            logger.warning(f"⚠️ {coin}-{interval}: cycle_results가 없어 학습 데이터 수집 불가")
            logger.warning(f"   selfplay_result 타입: {type(selfplay_result)}")
            logger.warning(f"   selfplay_result keys: {list(selfplay_result.keys()) if isinstance(selfplay_result, dict) else 'N/A'}")
            # 🔥 추가 디버깅: status 확인
            if isinstance(selfplay_result, dict):
                logger.warning(f"   status: {selfplay_result.get('status', 'N/A')}")
                logger.warning(f"   episodes: {selfplay_result.get('episodes', 'N/A')}")
            return []
        
        logger.debug(f"📊 {coin}-{interval}: cycle_results {len(cycle_results)}개 발견")
        
        # 🔥 학습 성능 개선: 모든 액션(BUY/SELL/HOLD) 포함
        # HOLD도 중요한 액션이므로 배제하지 않음
        actions_check = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        skipped_no_results = 0
        skipped_no_trades = 0  # 액션과 trades 모두 없는 경우만 제외
        
        # 🔥 디버깅: 첫 번째 cycle 상세 확인
        if cycle_results:
            first_cycle = cycle_results[0]
            logger.warning(f"  🔍 {coin}-{interval}: 첫 번째 cycle 디버깅:")
            logger.warning(f"     episode: {first_cycle.get('episode', 'N/A')}")
            logger.warning(f"     results 타입: {type(first_cycle.get('results', {}))}")
            results_raw = first_cycle.get('results', {})
            logger.warning(f"     results 값 자체: {results_raw}")
            logger.warning(f"     results 키 개수: {len(results_raw) if isinstance(results_raw, dict) else 'N/A'}")
            if isinstance(results_raw, dict) and results_raw:
                first_agent_id = list(results_raw.keys())[0]
                first_agent_result = results_raw[first_agent_id]
                logger.warning(f"     첫 번째 agent_id: {first_agent_id}")
                logger.warning(f"     첫 번째 agent 결과 타입: {type(first_agent_result)}")
                logger.warning(f"     첫 번째 agent 결과 keys: {list(first_agent_result.keys()) if isinstance(first_agent_result, dict) else 'N/A'}")
                if isinstance(first_agent_result, dict):
                    logger.warning(f"     total_pnl: {first_agent_result.get('total_pnl', 'N/A')}")
                    logger.warning(f"     win_rate: {first_agent_result.get('win_rate', 'N/A')}")
                    logger.warning(f"     trades 타입: {type(first_agent_result.get('trades', []))}")
                    logger.warning(f"     trades 개수: {len(first_agent_result.get('trades', []))}")
            elif not results_raw:
                logger.warning(f"     ⚠️ results가 비어있거나 None입니다!")
        
        for cycle in cycle_results:
            episode_num = cycle.get("episode", 0)
            results = cycle.get("results", {})
            
            if not results:
                skipped_no_results += 1
                # 🔥 디버깅: 왜 results가 비어있는지 확인
                if skipped_no_results == 1:  # 첫 번째만 로그
                    logger.warning(f"  🔍 {coin}-{interval}: episode {episode_num}의 results가 비어있음")
                    logger.warning(f"     cycle keys: {list(cycle.keys())}")
                    logger.warning(f"     results 값: {results}")
                continue
            
            # 에피소드 내 액션 다양성 체크
            episode_actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            has_any_trades = False
            has_performance_data = False  # 🔥 성과 데이터 존재 여부 확인
            
            for agent_id, agent_result in results.items():
                # 🔥 성과 데이터 확인 (total_pnl, win_rate 등이 있으면 포함)
                if isinstance(agent_result, dict):
                    if 'total_pnl' in agent_result or 'win_rate' in agent_result or 'total_return' in agent_result:
                        has_performance_data = True
                
                # 🔥 전략 방향 확인 (매수/매도 전략 구분)
                strategy_direction = agent_result.get('strategy_direction', 'neutral') if isinstance(agent_result, dict) else 'neutral'
                
                trades = agent_result.get('trades', [])
                if trades:
                    has_any_trades = True
                    for trade in trades:
                        direction = trade.get('direction', 'HOLD')
                        episode_actions[direction] = episode_actions.get(direction, 0) + 1
                        actions_check[direction] = actions_check.get(direction, 0) + 1
                        
                        # 🔥 전략 방향과 예측 방향 일치 여부 확인 (매수 전략은 BUY만, 매도 전략은 SELL만)
                        if strategy_direction == 'buy' and direction != 'BUY':
                            logger.debug(f"  ⚠️ {coin}-{interval}: 매수 전략 {agent_id}가 {direction} 예측 생성 (BUY 예상)")
                        elif strategy_direction == 'sell' and direction != 'SELL':
                            logger.debug(f"  ⚠️ {coin}-{interval}: 매도 전략 {agent_id}가 {direction} 예측 생성 (SELL 예상)")
            
            # 🔥 HOLD도 중요한 액션이므로 기본적으로 모든 에피소드 포함
            # 다만 BUY/SELL이 있는 에피소드를 우선시하고, HOLD만 있는 에피소드도 포함
            total_episode_actions = sum(episode_actions.values())
            has_buy_sell = (episode_actions.get('BUY', 0) > 0 or 
                          episode_actions.get('SELL', 0) > 0)
            only_hold = (total_episode_actions > 0 and 
                        episode_actions.get('BUY', 0) == 0 and 
                        episode_actions.get('SELL', 0) == 0)
            
            # 🔥 필터링 로직: 모든 에피소드 기본 포함, HOLD만 있는 것도 포함
            # BUY/SELL이 있으면 우선 포함, HOLD만 있어도 포함 (HOLD는 유효한 액션)
            # 다만, 에피소드에 액션 데이터도 없고 trades도 없는 경우만 제외
            # 🔥 개선: results가 비어있지 않으면 포함 (성과 데이터가 있으면 학습 가능)
            should_include = True  # 기본적으로 모두 포함
            
            # results가 비어있지 않으면 포함 (성과 데이터가 있으면 학습 가능)
            # total_episode_actions == 0 and not has_any_trades and not has_performance_data인 경우만 제외
            if total_episode_actions == 0 and not has_any_trades and not has_performance_data:
                # 🔥 디버깅: 왜 제외되는지 확인
                if skipped_no_trades == 0:  # 첫 번째만 로그
                    logger.warning(f"  🔍 {coin}-{interval}: episode {episode_num} 제외 - 액션 없음, trades 없음, 성과 데이터 없음")
                    logger.warning(f"     results 키 개수: {len(results)}")
                    if results:
                        first_agent_id = list(results.keys())[0]
                        first_agent_result = results[first_agent_id]
                        logger.warning(f"     첫 번째 agent_id: {first_agent_id}")
                        logger.warning(f"     첫 번째 agent keys: {list(first_agent_result.keys()) if isinstance(first_agent_result, dict) else 'N/A'}")
                        logger.warning(f"     total_pnl: {first_agent_result.get('total_pnl', 'N/A') if isinstance(first_agent_result, dict) else 'N/A'}")
                skipped_no_trades += 1
                should_include = False
            elif has_performance_data and not has_any_trades:
                # 🔥 성과 데이터는 있지만 trades는 없는 경우 포함 (성과 기반 학습)
                logger.debug(f"  📊 {coin}-{interval}: episode {episode_num} 포함 - 성과 데이터 있음 (trades 없음)")
            
            if should_include:
                # 🔥 전략 방향별 분류 (매수/매도 전략 구분)
                buy_strategies = {}
                sell_strategies = {}
                neutral_strategies = {}
                
                for agent_id, agent_result in results.items():
                    strategy_direction = agent_result.get('strategy_direction', 'neutral') if isinstance(agent_result, dict) else 'neutral'
                    if strategy_direction == 'buy':
                        buy_strategies[agent_id] = agent_result
                    elif strategy_direction == 'sell':
                        sell_strategies[agent_id] = agent_result
                    else:
                        neutral_strategies[agent_id] = agent_result
                
                episode_data = {
                    'episode': episode_num,
                    'coin': coin,
                    'interval': interval,
                    'results': results,
                    'timestamp': datetime.now().isoformat(),
                    'action_counts': episode_actions,  # 액션 분포 저장
                    'strategy_directions': {  # 🔥 전략 방향별 분류 추가
                        'buy': buy_strategies,
                        'sell': sell_strategies,
                        'neutral': neutral_strategies
                    }
                }
                episodes_data.append(episode_data)
                
                # 🔥 HOLD만 있는 에피소드에 대한 정보 로깅 (경고 아님)
                if only_hold:
                    logger.debug(f"  📊 {coin}-{interval}: 에피소드 {episode_num}은 HOLD만 포함 (정상적인 액션)")
        
        # 상세 로깅
        total_actions = sum(actions_check.values())
        if total_actions > 0:
            action_dist = {k: v/total_actions for k, v in actions_check.items()}
            logger.info(f"📊 {coin}-{interval}: 액션 분포 - BUY: {action_dist.get('BUY', 0):.1%}, "
                       f"SELL: {action_dist.get('SELL', 0):.1%}, HOLD: {action_dist.get('HOLD', 0):.1%}")
            
            # 🔥 HOLD 비율이 높아도 정상적일 수 있음 (시장 상황에 따라)
            # 경고 대신 정보 로깅으로 변경
            if action_dist.get('HOLD', 0) > 0.9:
                logger.info(f"📊 {coin}-{interval}: HOLD 비율이 높음 ({action_dist.get('HOLD', 0):.1%}) - "
                          f"시장 상황에 따라 정상적인 액션 선택일 수 있음")
            elif action_dist.get('HOLD', 0) > 0.7:
                logger.debug(f"📊 {coin}-{interval}: HOLD 비율이 다소 높음 ({action_dist.get('HOLD', 0):.1%})")
        else:
            # 액션 데이터가 없지만 에피소드는 포함됨 (성과 기반 학습)
            logger.debug(f"📊 {coin}-{interval}: 액션 데이터 없음 (성과 기반 학습 데이터로 포함)")
        
        # 수집 상세 정보 로깅
        logger.info(f"✅ {coin}-{interval}: {len(episodes_data)}개 에피소드 데이터 수집 완료")
        if skipped_no_results > 0 or skipped_no_trades > 0:
            logger.debug(f"   📊 제외된 에피소드: results 없음={skipped_no_results}, "
                        f"액션/trades 모두 없음={skipped_no_trades}")
        
        # 🔥 min_episodes 체크는 함수 끝에서 (수집 후)
        if min_episodes > 0 and len(episodes_data) < min_episodes:
            logger.info(f"📊 {coin}-{interval}: 수집된 에피소드 수 부족 ({len(episodes_data)} < {min_episodes}), 빈 리스트 반환")
            return []
        
        return episodes_data
        
    except Exception as e:
        logger.error(f"❌ 학습 데이터 수집 실패: {e}")
        return []


def auto_train_from_selfplay(
    coin: str,
    interval: str,
    selfplay_result: Dict[str, Any],
    config_path: Optional[str] = None,
    min_episodes: int = 10,
    previous_attempts: int = 0  # 🔥 재시도 횟수 추적
) -> Optional[str]:
    """
    Self-play 결과로 자동 학습
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        selfplay_result: Self-play 결과
        config_path: 설정 파일 경로 (None이면 기본값)
        min_episodes: 최소 에피소드 수
    
    Returns:
        학습된 모델 ID (실패 시 None)
    """
    try:
        # JAX 가용성 확인 (더 자세한 체크)
        try:
            import jax
            import jax.numpy as jnp
            from flax import linen as nn
            logger.debug("✅ JAX/Flax 임포트 성공")
        except ImportError as import_err:
            logger.warning(f"⚠️ JAX가 설치되지 않아 학습을 건너뜁니다: {import_err}")
            return None
        
        # neural_policy_jax 모듈의 JAX_AVAILABLE도 확인
        try:
            from rl_pipeline.hybrid.neural_policy_jax import JAX_AVAILABLE as NEURAL_JAX_AVAILABLE
            if not NEURAL_JAX_AVAILABLE:
                logger.warning("⚠️ neural_policy_jax 모듈에서 JAX 사용 불가, 학습 건너뜁니다")
                return None
        except ImportError as import_err:
            logger.warning(f"⚠️ neural_policy_jax 모듈 임포트 실패: {import_err}")
            return None
        
        # 학습 데이터 수집
        episodes_data = collect_selfplay_data_for_training(
            coin, interval, selfplay_result, min_episodes
        )
        
        if not episodes_data:
            logger.info(f"📊 {coin}-{interval}: 학습 데이터 없음, 학습 건너뜀")
            return None
        
        # 설정 파일 로드
        if config_path is None:
            config_path = os.getenv(
                'HYBRID_CONFIG_PATH',
                '/workspace/rl_pipeline/hybrid/config_hybrid.json'
            )
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ 설정 파일 없음: {config_path}, 기본 설정 사용")
            config = {
                'train': {
                    'epochs': 20,
                    'batch_size': 2048,
                    'lr': 0.0003,
                    'hidden_dim': 128
                },
                'paths': {
                    'checkpoints': '/workspace/rl_pipeline/artifacts/checkpoints',
                    'db': '/workspace/data_storage/rl_strategies.db'
                }
            }
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Trainer 초기화 및 학습
        logger.info(f"🚀 {coin}-{interval}: 신경망 학습 시작 ({len(episodes_data)}개 에피소드)")
        
        try:
            trainer = PPOTrainer(config)
        except ImportError as e:
            logger.error(f"❌ PPOTrainer 초기화 실패 (JAX 관련): {e}")
            return None
        except Exception as e:
            logger.error(f"❌ PPOTrainer 초기화 실패: {e}")
            import traceback
            logger.debug(f"초기화 실패 상세:\n{traceback.format_exc()}")
            return None

        try:
            # 🆕 증분 학습: episodes_data에서 전략 정보 추출
            from rl_pipeline.db.reads import load_strategies_pool

            # 최근 생성된 전략들 로드 (similarity_classification 메타데이터 포함)
            strategies = load_strategies_pool(
                coin=coin,
                interval=interval,
                limit=100,  # 최근 100개만
                order_by="created_at DESC",
                include_unknown=True
            )

            # 🔥 증분 학습 적용 여부 확인
            has_incremental_metadata = any(
                s.get('similarity_classification') in ['copy', 'finetune', 'novel']
                for s in strategies
            )

            if has_incremental_metadata and len(strategies) > 0:
                logger.info(f"🔄 {coin}-{interval}: 증분 학습 모드 활성화")
                model_id = train_strategies_incremental(
                    strategies=strategies,
                    episodes_data=episodes_data,
                    trainer=trainer,
                    db_path=config.get('paths', {}).get('db'),
                    analysis_data=None
                )
            else:
                logger.info(f"📊 {coin}-{interval}: 일반 학습 모드 (메타데이터 없음)")
                model_id = trainer.train_from_selfplay_data(
                    episodes_data,
                    db_path=config.get('paths', {}).get('db')
                )
            
            if model_id:
                logger.info(f"✅ {coin}-{interval}: 신경망 학습 완료: {model_id}")
                
                # 🔥 자동 평가 실행 (환경변수 체크)
                if should_auto_evaluate(model_id):
                    try:
                        # 평가에 필요한 데이터 준비
                        from rl_pipeline.data.candles_loader import load_candles
                        
                        # 캔들 데이터 로드 (최근 30일)
                        candle_data = load_candles(coin, interval, days=30)
                        
                        if candle_data is not None and len(candle_data) > 0:
                            # 전략 파라미터 리스트 가져오기 (DB에서 또는 selfplay_result에서)
                            strategy_params_list = []
                            if selfplay_result and 'cycle_results' in selfplay_result:
                                # cycle_results에서 전략 파라미터 추출
                                for cycle in selfplay_result['cycle_results']:
                                    results = cycle.get('results', {})
                                    for agent_id, perf in results.items():
                                        if 'strategy_params' in perf:
                                            strategy_params_list.append(perf['strategy_params'])
                            
                            # 중복 제거
                            if strategy_params_list:
                                seen = set()
                                unique_params = []
                                for params in strategy_params_list:
                                    params_str = str(sorted(params.items()))
                                    if params_str not in seen:
                                        seen.add(params_str)
                                        unique_params.append(params)
                                strategy_params_list = unique_params[:10]  # 최대 10개
                            
                            # 기본 전략 파라미터가 없으면 간단한 기본값 사용
                            if not strategy_params_list:
                                strategy_params_list = [{
                                    'rsi_min': 30.0, 'rsi_max': 70.0,
                                    'volume_ratio_min': 1.0, 'volume_ratio_max': 2.0,
                                    'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01,
                                    'stop_loss_pct': 0.02, 'take_profit_pct': 0.05
                                }]
                            
                            # 자동 평가 실행
                            eval_result = auto_evaluate_model(
                                model_id=model_id,
                                coin=coin,
                                interval=interval,
                                candle_data=candle_data,
                                strategy_params_list=strategy_params_list,
                                config=config
                            )
                            
                            if eval_result:
                                logger.info(f"✅ {coin}-{interval}: 자동 평가 완료")
                                
                                # 🔥 검증 결과 평가 및 재학습 판단
                                passed, reason, details = evaluate_validation_results(eval_result)
                                
                                if passed:
                                    logger.info(f"✅ {coin}-{interval}: 검증 합격 - 모델 사용 가능")
                                    if details.get('warnings'):
                                        logger.info(f"   ⚠️ 경고 {len(details['warnings'])}개: {', '.join(details['warnings'][:2])}")
                                else:
                                    logger.warning(f"⚠️ {coin}-{interval}: 검증 불합격 - {reason}")
                                    
                                    # 재학습 필요 여부 확인 (재시도 횟수 추적)
                                    needs_retrain, retrain_reason = should_retrain(eval_result, previous_attempts=previous_attempts)
                                    
                                    if needs_retrain:
                                        logger.warning(f"🔄 {coin}-{interval}: 재학습 권장 - {retrain_reason} (재시도: {previous_attempts}회)")
                                        
                                        # 재학습 제안 가져오기
                                        suggestions = get_retrain_suggestions(eval_result)
                                        logger.info(f"💡 재학습 제안: {suggestions.get('reason', '')}")
                                        
                                        # 🔥 자동 재학습 여부 확인 (환경변수)
                                        auto_retrain_enabled = os.getenv('ENABLE_AUTO_RETRAIN', 'false').lower() == 'true'
                                        
                                        if auto_retrain_enabled:
                                            logger.info(f"🔄 {coin}-{interval}: 자동 재학습 시작 (환경변수 활성화됨, 재시도: {previous_attempts + 1}회)")
                                            # 재학습 실행 (재귀 호출, 재시도 횟수 증가)
                                            retrain_model_id = auto_train_from_selfplay(
                                                coin=coin,
                                                interval=interval,
                                                selfplay_result=selfplay_result,
                                                config_path=config_path,
                                                min_episodes=min_episodes,
                                                previous_attempts=previous_attempts + 1  # 🔥 재시도 횟수 증가
                                            )
                                            if retrain_model_id:
                                                logger.info(f"✅ {coin}-{interval}: 재학습 완료: {retrain_model_id}")
                                                return retrain_model_id
                                            else:
                                                logger.warning(f"⚠️ {coin}-{interval}: 재학습 실패")
                                        else:
                                            logger.info(f"💡 자동 재학습 비활성화 (ENABLE_AUTO_RETRAIN=false), 수동 재학습 권장")
                                    else:
                                        logger.info(f"📊 {coin}-{interval}: 재학습 불필요 또는 최대 시도 횟수 초과")
                            else:
                                logger.warning(f"⚠️ {coin}-{interval}: 자동 평가 실패 (계속 진행)")
                        else:
                            logger.warning(f"⚠️ {coin}-{interval}: 평가용 캔들 데이터 없음, 평가 건너뜀")
                    except Exception as eval_err:
                        logger.warning(f"⚠️ {coin}-{interval}: 자동 평가 중 오류 (계속 진행): {eval_err}")
            else:
                logger.warning(f"⚠️ {coin}-{interval}: 학습 완료했지만 모델 ID가 없음")
            
            return model_id
        except Exception as train_err:
            logger.error(f"❌ 학습 실행 중 오류: {train_err}")
            import traceback
            logger.debug(f"학습 오류 상세:\n{traceback.format_exc()}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 자동 학습 실패: {e}")
        import traceback
        logger.debug(f"자동 학습 실패 상세:\n{traceback.format_exc()}")
        return None


def auto_train_from_integrated_analysis(
    coin: str,
    all_interval_selfplay: Dict[str, Dict[str, Any]],  # {interval: selfplay_result}
    analysis_result: Any,  # CoinSignalScore 또는 dict
    config_path: Optional[str] = None,
    min_episodes: int = 10,
    previous_attempts: int = 0  # 🔥 재시도 횟수 추적
) -> Optional[str]:
    """
    통합 분석 단계에서 자동 학습 (모든 인터벌 self-play + 분석 결과 활용)
    
    Args:
        coin: 코인 심볼
        all_interval_selfplay: 모든 인터벌의 self-play 결과 딕셔너리
            {
                '15m': {...selfplay_result...},
                '30m': {...selfplay_result...},
                ...
            }
        analysis_result: 통합 분석 결과 (CoinSignalScore 또는 dict)
        config_path: 설정 파일 경로 (None이면 기본값)
        min_episodes: 최소 에피소드 수 (인터벌별)
    
    Returns:
        학습된 모델 ID (실패 시 None)
    """
    try:
        logger.info(f"🚀 {coin}: 통합 학습 시작 (인터벌 수: {len(all_interval_selfplay)})")
        logger.info(f"📊 학습 대상 인터벌: {list(all_interval_selfplay.keys())}")
        
        # JAX 가용성 확인
        try:
            import jax
            import jax.numpy as jnp
            from flax import linen as nn
            logger.debug("✅ JAX/Flax 임포트 성공")
        except ImportError as import_err:
            logger.warning(f"⚠️ JAX가 설치되지 않아 학습을 건너뜁니다: {import_err}")
            return None
        
        # neural_policy_jax 모듈 확인
        try:
            from rl_pipeline.hybrid.neural_policy_jax import JAX_AVAILABLE as NEURAL_JAX_AVAILABLE
            if not NEURAL_JAX_AVAILABLE:
                logger.warning("⚠️ neural_policy_jax 모듈에서 JAX 사용 불가, 학습 건너뜁니다")
                return None
        except ImportError as import_err:
            logger.warning(f"⚠️ neural_policy_jax 모듈 임포트 실패: {import_err}")
            return None
        
        # 통합 분석 결과에서 분석 점수 추출
        if hasattr(analysis_result, 'fractal_score'):
            # CoinSignalScore 객체인 경우
            analysis_data = {
                'fractal_score': analysis_result.fractal_score,
                'multi_timeframe_score': analysis_result.multi_timeframe_score,
                'indicator_cross_score': analysis_result.indicator_cross_score,
                'ensemble_score': analysis_result.ensemble_score,
                'ensemble_confidence': analysis_result.ensemble_confidence
            }
            logger.debug(f"✅ 분석 결과: CoinSignalScore 객체에서 추출")
        elif isinstance(analysis_result, dict):
            # dict인 경우
            analysis_data = {
                'fractal_score': analysis_result.get('fractal_score', 0.5),
                'multi_timeframe_score': analysis_result.get('multi_timeframe_score', 0.5),
                'indicator_cross_score': analysis_result.get('indicator_cross_score', 0.5),
                'ensemble_score': analysis_result.get('ensemble_score', 0.5),
                'ensemble_confidence': analysis_result.get('ensemble_confidence', 0.5)
            }
            logger.debug(f"✅ 분석 결과: dict에서 추출")
        elif hasattr(analysis_result, 'signal_score') or hasattr(analysis_result, 'signal_action'):
            # PipelineResult 객체인 경우 (분석 점수는 없지만 signal_score는 있음)
            # DB에서 최신 통합 분석 결과 조회 시도
            logger.debug(f"ℹ️ 분석 결과: PipelineResult 객체 감지 (타입: {type(analysis_result).__name__}), DB에서 최신 분석 결과 조회 시도")
            try:
                from rl_pipeline.db.reads import fetch_integrated_analysis
                from rl_pipeline.db.connection_pool import get_strategy_db_pool
                
                pool = get_strategy_db_pool()
                with pool.get_connection() as conn:
                    # 최신 통합 분석 결과 조회
                    latest_analysis = fetch_integrated_analysis(conn, coin, 'all_intervals')
                    if latest_analysis and isinstance(latest_analysis, dict):
                        # fetch_integrated_analysis 반환 형식: multi_tf_score (multi_timeframe_score 아님)
                        analysis_data = {
                            'fractal_score': latest_analysis.get('fractal_score', 0.5),
                            'multi_timeframe_score': latest_analysis.get('multi_tf_score', latest_analysis.get('multi_timeframe_score', 0.5)),
                            'indicator_cross_score': latest_analysis.get('indicator_cross_score', 0.5),
                            'ensemble_score': latest_analysis.get('score', latest_analysis.get('ensemble_score', 0.5)),
                            'ensemble_confidence': latest_analysis.get('confidence', latest_analysis.get('signal_confidence', 0.5))
                        }
                        logger.info(f"✅ 분석 결과: DB에서 최신 통합 분석 결과 조회 성공 (프랙탈={analysis_data['fractal_score']:.3f}, 멀티TF={analysis_data['multi_timeframe_score']:.3f}, 앙상블={analysis_data['ensemble_score']:.3f})")
                    else:
                        # DB 조회 실패 시 기본값 사용
                        analysis_data = {
                            'fractal_score': 0.5,
                            'multi_timeframe_score': 0.5,
                            'indicator_cross_score': 0.5,
                            'ensemble_score': 0.5,
                            'ensemble_confidence': 0.5
                        }
                        logger.warning(f"⚠️ 분석 결과: PipelineResult 객체이지만 DB에서 분석 점수를 찾을 수 없음 (coin={coin}, interval=all_intervals), 기본값 사용")
            except Exception as db_err:
                # DB 조회 실패 시 기본값 사용
                analysis_data = {
                    'fractal_score': 0.5,
                    'multi_timeframe_score': 0.5,
                    'indicator_cross_score': 0.5,
                    'ensemble_score': 0.5,
                    'ensemble_confidence': 0.5
                }
                logger.warning(f"⚠️ 분석 결과: PipelineResult 객체이지만 DB 조회 실패 ({type(db_err).__name__}: {str(db_err)[:100]}), 기본값 사용")
        else:
            # 알 수 없는 형식
            result_type = type(analysis_result).__name__
            result_str = str(analysis_result)[:200] if analysis_result is not None else "None"
            logger.warning(f"⚠️ 분석 결과 형식이 예상과 다름 (타입: {result_type}), 기본값 사용")
            logger.debug(f"   분석 결과 내용: {result_str}")
            analysis_data = {
                'fractal_score': 0.5,
                'multi_timeframe_score': 0.5,
                'indicator_cross_score': 0.5,
                'ensemble_score': 0.5,
                'ensemble_confidence': 0.5
            }
        
        logger.info(f"📊 분석 점수: 프랙탈={analysis_data['fractal_score']:.3f}, "
                   f"멀티TF={analysis_data['multi_timeframe_score']:.3f}, "
                   f"지표교차={analysis_data['indicator_cross_score']:.3f}")
        
        # 모든 인터벌의 학습 데이터 수집 및 결합
        all_episodes_data = []
        total_episodes = 0
        
        for interval, selfplay_result in all_interval_selfplay.items():
            if not selfplay_result:
                logger.debug(f"  ⚠️ {coin}-{interval}: selfplay_result가 비어있음, 스킵")
                continue
            
            # 🔥 디버깅: selfplay_result 구조 확인
            logger.info(f"  📊 {coin}-{interval}: selfplay_result 타입={type(selfplay_result)}, keys={list(selfplay_result.keys()) if isinstance(selfplay_result, dict) else 'N/A'}")
            
            # 🔥 cycle_results 확인 (디버깅)
            if isinstance(selfplay_result, dict):
                cycle_results = selfplay_result.get("cycle_results", [])
                logger.info(f"  📊 {coin}-{interval}: cycle_results 존재={cycle_results is not None}, 길이={len(cycle_results) if cycle_results else 0}")
                if cycle_results and len(cycle_results) > 0:
                    first_cycle = cycle_results[0] if cycle_results else {}
                    logger.info(f"  📊 {coin}-{interval}: 첫 번째 cycle 타입={type(first_cycle)}, keys={list(first_cycle.keys()) if isinstance(first_cycle, dict) else 'N/A'}")
            
            episodes_data = collect_selfplay_data_for_training(
                coin, interval, selfplay_result, min_episodes=0  # 최소 에피소드 체크는 전체에서
            )
            
            if episodes_data:
                all_episodes_data.extend(episodes_data)
                total_episodes += len(episodes_data)
                logger.info(f"  ✅ {interval}: {len(episodes_data)}개 에피소드 추가")
            else:
                logger.warning(f"  ⚠️ {coin}-{interval}: 에피소드 데이터 수집 실패 (0개)")
                # 🔥 디버깅: 왜 실패했는지 확인
                if isinstance(selfplay_result, dict):
                    cycle_results = selfplay_result.get("cycle_results", [])
                    logger.warning(f"    cycle_results: {len(cycle_results)}개")
                    if cycle_results:
                        first_cycle = cycle_results[0] if cycle_results else {}
                        logger.warning(f"    첫 번째 cycle keys: {list(first_cycle.keys()) if isinstance(first_cycle, dict) else 'N/A'}")
        
        # 최소 에피소드 수 체크 (전체 기준)
        if total_episodes < min_episodes:
            logger.info(f"📊 {coin}: 총 에피소드 수 부족 ({total_episodes} < {min_episodes}), 학습 건너뜀")
            return None
        
        # 🔥 개선: 중복 에피소드 체크 및 제거 (완화된 기준)
        try:
            seen_episodes = set()
            unique_episodes = []
            duplicate_count = 0
            
            for episode in all_episodes_data:
                # 🔥 완화된 고유성 판단: (coin, interval, episode_num, timestamp)
                # states/actions 해시는 너무 엄격하여 거의 모든 에피소드가 중복으로 판단됨
                # episode_num과 timestamp로 충분히 구분 가능
                episode_num = episode.get('episode', 0)
                timestamp = episode.get('timestamp', '')
                interval_key = episode.get('interval', interval)
                
                # 🔥 results에서 추출한 정보도 포함 (더 정확한 구분)
                results = episode.get('results', {})
                first_agent_id = list(results.keys())[0] if results else ''
                first_result = results.get(first_agent_id, {}) if first_agent_id else {}
                total_pnl = first_result.get('total_pnl', 0.0)
                total_trades = first_result.get('total_trades', 0)
                
                # 🔥 완화된 키: episode_num + interval + total_trades + timestamp (처음 10자만)
                # 같은 에피소드 번호라도 다른 interval이나 성과면 다른 에피소드로 간주
                episode_key = (
                    coin,
                    interval_key,
                    episode_num,
                    total_trades,  # 거래 수로 구분
                    round(total_pnl, 2),  # 수익을 소수점 2자리로 반올림하여 구분
                    timestamp[:10] if timestamp else ''  # 타임스탬프 처음 10자
                )
                
                if episode_key not in seen_episodes:
                    seen_episodes.add(episode_key)
                    unique_episodes.append(episode)
                else:
                    duplicate_count += 1
            
            if duplicate_count > 0:
                logger.info(f"📊 {coin}: 중복 에피소드 {duplicate_count}개 제거, {len(unique_episodes)}개 고유 에피소드 사용")
                all_episodes_data = unique_episodes
                total_episodes = len(unique_episodes)
        except Exception as e:
            logger.warning(f"⚠️ 중복 체크 실패 (무시): {e}")
        
        # 🔥 개선: 학습 빈도 제한 체크 (최근 N시간 내 학습했으면 스킵)
        try:
            from rl_pipeline.db.connection_pool import get_optimized_db_connection
            min_training_interval_hours = int(os.getenv('MIN_TRAINING_INTERVAL_HOURS', '6'))  # 기본 6시간
            
            with get_optimized_db_connection("strategies") as conn:
                cursor = conn.cursor()
                # 최근 학습 기록 조회
                cursor.execute("""
                    SELECT MAX(created_at) as last_training
                    FROM hybrid_models
                    WHERE coin = ? AND status = 'completed'
                """, (coin,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    from datetime import datetime, timedelta
                    last_training_str = result[0]
                    if isinstance(last_training_str, str):
                        last_training = datetime.fromisoformat(last_training_str.replace('Z', '+00:00'))
                    else:
                        last_training = result[0]
                    
                    time_since_last = datetime.now() - (last_training.replace(tzinfo=None) if last_training.tzinfo else last_training)
                    
                    if time_since_last.total_seconds() < min_training_interval_hours * 3600:
                        hours_remaining = (min_training_interval_hours * 3600 - time_since_last.total_seconds()) / 3600
                        logger.info(f"📊 {coin}: 최근 학습 후 {time_since_last.total_seconds()/3600:.1f}시간 경과, "
                                  f"최소 간격({min_training_interval_hours}시간) 미달로 학습 건너뜀 "
                                  f"(남은 시간: {hours_remaining:.1f}시간)")
                        return None
        except Exception as e:
            logger.debug(f"⚠️ 학습 빈도 체크 실패 (무시하고 계속): {e}")
        
        logger.info(f"✅ {coin}: 총 {total_episodes}개 고유 에피소드 수집 완료")
        
        # 설정 파일 로드
        if config_path is None:
            config_path = os.getenv(
                'HYBRID_CONFIG_PATH',
                '/workspace/rl_pipeline/hybrid/config_hybrid.json'
            )
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ 설정 파일 없음: {config_path}, 기본 설정 사용")
            config = {
                'train': {
                    'epochs': 30,
                    'batch_size': 4096,
                    'lr': 0.0003,
                    'hidden_dim': 128
                },
                'paths': {
                    'checkpoints': '/workspace/rl_pipeline/artifacts/checkpoints',
                    'db': '/workspace/data_storage/rl_strategies.db'
                }
            }
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Trainer 초기화 및 학습
        logger.info(f"🚀 {coin}: 통합 신경망 학습 시작")
        logger.info(f"   📊 에피소드: {total_episodes}개")
        logger.info(f"   🔥 분석 데이터 포함: 25차원 상태 벡터 사용 (확장 지표 포함)")
        logger.info(f"   📈 분석 점수: 프랙탈={analysis_data['fractal_score']:.3f}, "
                   f"멀티TF={analysis_data['multi_timeframe_score']:.3f}, "
                   f"지표교차={analysis_data['indicator_cross_score']:.3f}")
        
        try:
            trainer = PPOTrainer(config)
        except ImportError as e:
            logger.error(f"❌ PPOTrainer 초기화 실패 (JAX 관련): {e}")
            return None
        except Exception as e:
            logger.error(f"❌ PPOTrainer 초기화 실패: {e}")
            import traceback
            logger.debug(f"초기화 실패 상세:\n{traceback.format_exc()}")
            return None

        try:
            # 🆕 증분 학습: 통합 학습에도 적용
            from rl_pipeline.db.reads import load_strategies_pool

            # 모든 인터벌의 최근 전략 로드
            all_strategies = []
            for interval in all_interval_selfplay.keys():
                strategies = load_strategies_pool(
                    coin=coin,
                    interval=interval,
                    limit=50,  # 인터벌당 50개씩
                    order_by="created_at DESC",
                    include_unknown=True
                )
                all_strategies.extend(strategies)

            # 증분 학습 적용 여부 확인
            has_incremental_metadata = any(
                s.get('similarity_classification') in ['copy', 'finetune', 'novel']
                for s in all_strategies
            )

            if has_incremental_metadata and len(all_strategies) > 0:
                logger.info(f"🔄 {coin}: 통합 학습 - 증분 학습 모드 활성화 ({len(all_strategies)}개 전략)")
                model_id = train_strategies_incremental(
                    strategies=all_strategies,
                    episodes_data=all_episodes_data,
                    trainer=trainer,
                    db_path=config.get('paths', {}).get('db'),
                    analysis_data=analysis_data
                )
            else:
                logger.info(f"📊 {coin}: 통합 학습 - 일반 학습 모드")
                model_id = trainer.train_from_selfplay_data(
                    all_episodes_data,
                    db_path=config.get('paths', {}).get('db'),
                    analysis_data=analysis_data  # 🔥 분석 데이터 전달
                )
            
            if model_id:
                logger.info(f"✅ {coin}: 통합 신경망 학습 완료: {model_id}")
                
                # 🔥 자동 평가 실행 (환경변수 체크)
                if should_auto_evaluate(model_id):
                    try:
                        # 평가는 첫 번째 인터벌로 실행 (대표성)
                        first_interval = list(all_interval_selfplay.keys())[0] if all_interval_selfplay else None
                        if first_interval:
                            from rl_pipeline.data.candles_loader import load_candles
                            
                            # 캔들 데이터 로드
                            candle_data = load_candles(coin, first_interval, days=30)
                            
                            if candle_data is not None and len(candle_data) > 0:
                                # 전략 파라미터 추출
                                strategy_params_list = []
                                first_selfplay = all_interval_selfplay.get(first_interval, {})
                                if first_selfplay and 'cycle_results' in first_selfplay:
                                    for cycle in first_selfplay['cycle_results']:
                                        results = cycle.get('results', {})
                                        for agent_id, perf in results.items():
                                            if 'strategy_params' in perf:
                                                strategy_params_list.append(perf['strategy_params'])
                                
                                # 중복 제거
                                if strategy_params_list:
                                    seen = set()
                                    unique_params = []
                                    for params in strategy_params_list:
                                        params_str = str(sorted(params.items()))
                                        if params_str not in seen:
                                            seen.add(params_str)
                                            unique_params.append(params)
                                    strategy_params_list = unique_params[:10]
                                
                                if not strategy_params_list:
                                    strategy_params_list = [{
                                        'rsi_min': 30.0, 'rsi_max': 70.0,
                                        'volume_ratio_min': 1.0, 'volume_ratio_max': 2.0,
                                        'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01,
                                        'stop_loss_pct': 0.02, 'take_profit_pct': 0.05
                                    }]
                                
                                # 자동 평가 실행
                                eval_result = auto_evaluate_model(
                                    model_id=model_id,
                                    coin=coin,
                                    interval=first_interval,
                                    candle_data=candle_data,
                                    strategy_params_list=strategy_params_list,
                                    config=config
                                )
                                
                                if eval_result:
                                    logger.info(f"✅ {coin}-{first_interval}: 통합 학습 자동 평가 완료")
                                    
                                    # 🔥 검증 결과 평가 및 재학습 판단
                                    passed, reason, details = evaluate_validation_results(eval_result)
                                    
                                    if passed:
                                        logger.info(f"✅ {coin}-{first_interval}: 검증 합격 - 모델 사용 가능")
                                        if details.get('warnings'):
                                            logger.info(f"   ⚠️ 경고 {len(details['warnings'])}개: {', '.join(details['warnings'][:2])}")
                                    else:
                                        logger.warning(f"⚠️ {coin}-{first_interval}: 검증 불합격 - {reason}")
                                        
                                        # 재학습 필요 여부 확인 (재시도 횟수 추적)
                                        needs_retrain, retrain_reason = should_retrain(eval_result, previous_attempts=previous_attempts)
                                        
                                        if needs_retrain:
                                            logger.warning(f"🔄 {coin}-{first_interval}: 재학습 권장 - {retrain_reason} (재시도: {previous_attempts}회)")
                                            
                                            # 재학습 제안 가져오기
                                            suggestions = get_retrain_suggestions(eval_result)
                                            logger.info(f"💡 재학습 제안: {suggestions.get('reason', '')}")
                                            
                                            # 🔥 자동 재학습 여부 확인 (환경변수)
                                            auto_retrain_enabled = os.getenv('ENABLE_AUTO_RETRAIN', 'false').lower() == 'true'
                                            
                                            if auto_retrain_enabled:
                                                logger.info(f"🔄 {coin}-{first_interval}: 자동 재학습 시작 (환경변수 활성화됨, 재시도: {previous_attempts + 1}회)")
                                                
                                                # 🔥 재학습 시 하이퍼파라미터 조정 적용
                                                adjusted_config_path = None
                                                if suggestions.get('adjust_learning_rate') or suggestions.get('adjust_entropy_coef'):
                                                    # 하이퍼파라미터 조정이 필요한 경우 임시 설정 파일 생성
                                                    adjusted_config_path = _create_adjusted_config(
                                                        config_path=config_path,
                                                        suggestions=suggestions,
                                                        previous_attempts=previous_attempts + 1
                                                    )
                                                    logger.info(f"🔧 하이퍼파라미터 조정 적용: {adjusted_config_path}")
                                                
                                                # 재학습 실행 (재시도 횟수 증가, 조정된 설정 사용)
                                                retrain_model_id = auto_train_from_integrated_analysis(
                                                    coin=coin,
                                                    all_interval_selfplay=all_interval_selfplay,
                                                    analysis_result=analysis_result,
                                                    config_path=adjusted_config_path or config_path,  # 🔥 조정된 설정 사용
                                                    min_episodes=min_episodes,
                                                    previous_attempts=previous_attempts + 1  # 🔥 재시도 횟수 증가
                                                )
                                                if retrain_model_id:
                                                    logger.info(f"✅ {coin}-{first_interval}: 재학습 완료: {retrain_model_id}")
                                                    return retrain_model_id
                                                else:
                                                    logger.warning(f"⚠️ {coin}-{first_interval}: 재학습 실패")
                                            else:
                                                logger.info(f"💡 자동 재학습 비활성화 (ENABLE_AUTO_RETRAIN=false), 수동 재학습 권장")
                                        else:
                                            logger.info(f"📊 {coin}-{first_interval}: 재학습 불필요 또는 최대 시도 횟수 초과")
                                else:
                                    logger.warning(f"⚠️ {coin}-{first_interval}: 통합 학습 자동 평가 실패 (계속 진행)")
                    except Exception as eval_err:
                        logger.warning(f"⚠️ {coin}: 통합 학습 자동 평가 중 오류 (계속 진행): {eval_err}")
            else:
                logger.warning(f"⚠️ {coin}: 학습 완료했지만 모델 ID가 없음")
            
            return model_id
        except Exception as train_err:
            logger.error(f"❌ 학습 실행 중 오류: {train_err}")
            import traceback
            logger.error(f"학습 오류 상세:\n{traceback.format_exc()}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 통합 자동 학습 실패: {e}")
        import traceback
        logger.error(f"통합 자동 학습 실패 상세:\n{traceback.format_exc()}")
        return None


def auto_train_from_global_strategies(
    all_coin_selfplay: Dict[str, Dict[str, Dict[str, Any]]],  # {coin: {interval: selfplay_result}}
    all_coin_analysis: Dict[str, Any],  # {coin: analysis_result} 또는 글로벌 분석 결과
    config_path: Optional[str] = None,
    min_episodes: int = 20,  # 글로벌 학습은 더 많은 데이터 필요
    previous_attempts: int = 0,  # 🔥 재시도 횟수 추적
    session_id: Optional[str] = None  # 디버그 세션 ID
) -> Optional[str]:
    """
    글로벌 전략 생성 단계에서 자동 학습 (모든 코인 self-play + 글로벌 분석 결과 활용)
    
    Args:
        all_coin_selfplay: 모든 코인의 self-play 결과
            {
                'BTC': {
                    '15m': {...selfplay_result...},
                    '30m': {...selfplay_result...},
                    ...
                },
                'ETH': {...},
                ...
            }
        all_coin_analysis: 모든 코인의 통합 분석 결과 또는 글로벌 분석 결과
            - 옵션 1: {coin: analysis_result}
            - 옵션 2: 글로벌 통합 분석 결과 (단일 dict)
        config_path: 설정 파일 경로 (None이면 기본값)
        min_episodes: 최소 에피소드 수 (모든 코인 합산)
    
    Returns:
        학습된 모델 ID (실패 시 None)
    """
    try:
        logger.info(f"🌍 글로벌 학습 시작 (코인 수: {len(all_coin_selfplay)})")
        
        # JAX 가용성 확인
        try:
            import jax
            import jax.numpy as jnp
            from flax import linen as nn
            logger.debug("✅ JAX/Flax 임포트 성공")
        except ImportError as import_err:
            logger.warning(f"⚠️ JAX가 설치되지 않아 학습을 건너뜁니다: {import_err}")
            return None
        
        # neural_policy_jax 모듈 확인
        try:
            from rl_pipeline.hybrid.neural_policy_jax import JAX_AVAILABLE as NEURAL_JAX_AVAILABLE
            if not NEURAL_JAX_AVAILABLE:
                logger.warning("⚠️ neural_policy_jax 모듈에서 JAX 사용 불가, 학습 건너뜁니다")
                return None
        except ImportError as import_err:
            logger.warning(f"⚠️ neural_policy_jax 모듈 임포트 실패: {import_err}")
            return None
        
        # 글로벌 분석 결과에서 분석 점수 추출
        if isinstance(all_coin_analysis, dict):
            # 코인별 분석 결과인지 글로벌 분석 결과인지 판단
            if 'fractal_score' in all_coin_analysis or 'overall_score' in all_coin_analysis:
                # 글로벌 분석 결과 (단일 dict)
                if 'fractal_score' in all_coin_analysis:
                    # 개별 코인 분석 결과와 동일한 형식
                    analysis_data = {
                        'fractal_score': all_coin_analysis.get('fractal_score', 0.5),
                        'multi_timeframe_score': all_coin_analysis.get('multi_timeframe_score', 0.5),
                        'indicator_cross_score': all_coin_analysis.get('indicator_cross_score', 0.5),
                        'ensemble_score': all_coin_analysis.get('ensemble_score', 0.5),
                        'ensemble_confidence': all_coin_analysis.get('ensemble_confidence', 0.5)
                    }
                else:
                    # GlobalSignalScore 형식 (overall_score 포함)
                    analysis_data = {
                        'fractal_score': all_coin_analysis.get('overall_score', 0.5),
                        'multi_timeframe_score': all_coin_analysis.get('overall_score', 0.5),
                        'indicator_cross_score': all_coin_analysis.get('overall_score', 0.5),
                        'ensemble_score': all_coin_analysis.get('overall_score', 0.5),
                        'ensemble_confidence': all_coin_analysis.get('overall_confidence', 0.5)
                    }
            else:
                # 코인별 분석 결과가 딕셔너리로 들어온 경우 (평균 계산)
                all_fractal_scores = []
                all_multi_tf_scores = []
                all_indicator_scores = []
                all_ensemble_scores = []
                all_confidence_scores = []
                
                for coin, analysis_result in all_coin_analysis.items():
                    if hasattr(analysis_result, 'fractal_score'):
                        all_fractal_scores.append(analysis_result.fractal_score)
                        all_multi_tf_scores.append(analysis_result.multi_timeframe_score)
                        all_indicator_scores.append(analysis_result.indicator_cross_score)
                        all_ensemble_scores.append(analysis_result.ensemble_score)
                        all_confidence_scores.append(analysis_result.ensemble_confidence)
                    elif isinstance(analysis_result, dict):
                        all_fractal_scores.append(analysis_result.get('fractal_score', 0.5))
                        all_multi_tf_scores.append(analysis_result.get('multi_timeframe_score', 0.5))
                        all_indicator_scores.append(analysis_result.get('indicator_cross_score', 0.5))
                        all_ensemble_scores.append(analysis_result.get('ensemble_score', 0.5))
                        all_confidence_scores.append(analysis_result.get('ensemble_confidence', 0.5))
                
                # 평균 계산
                analysis_data = {
                    'fractal_score': sum(all_fractal_scores) / len(all_fractal_scores) if all_fractal_scores else 0.5,
                    'multi_timeframe_score': sum(all_multi_tf_scores) / len(all_multi_tf_scores) if all_multi_tf_scores else 0.5,
                    'indicator_cross_score': sum(all_indicator_scores) / len(all_indicator_scores) if all_indicator_scores else 0.5,
                    'ensemble_score': sum(all_ensemble_scores) / len(all_ensemble_scores) if all_ensemble_scores else 0.5,
                    'ensemble_confidence': sum(all_confidence_scores) / len(all_confidence_scores) if all_confidence_scores else 0.5
                }
        else:
            logger.warning(f"⚠️ 글로벌 분석 결과 형식이 예상과 다름, 기본값 사용")
            analysis_data = {
                'fractal_score': 0.5,
                'multi_timeframe_score': 0.5,
                'indicator_cross_score': 0.5,
                'ensemble_score': 0.5,
                'ensemble_confidence': 0.5
            }
        
        logger.info(f"📊 글로벌 분석 점수: 프랙탈={analysis_data['fractal_score']:.3f}, "
                   f"멀티TF={analysis_data['multi_timeframe_score']:.3f}, "
                   f"지표교차={analysis_data['indicator_cross_score']:.3f}")
        
        # 모든 코인-인터벌의 학습 데이터 수집 및 결합
        all_episodes_data = []
        total_episodes = 0
        coins_processed = 0
        intervals_processed = 0
        
        for coin, coin_selfplay in all_coin_selfplay.items():
            if not coin_selfplay:
                continue
            
            for interval, selfplay_result in coin_selfplay.items():
                if not selfplay_result:
                    continue
                
                episodes_data = collect_selfplay_data_for_training(
                    coin, interval, selfplay_result, min_episodes=0
                )
                
                if episodes_data:
                    all_episodes_data.extend(episodes_data)
                    total_episodes += len(episodes_data)
                    intervals_processed += 1
                    logger.debug(f"  ✅ {coin}-{interval}: {len(episodes_data)}개 에피소드 추가")
            
            if coin_selfplay:
                coins_processed += 1
        
        # 최소 에피소드 수 체크 (전체 기준)
        if total_episodes < min_episodes:
            logger.info(f"📊 글로벌: 총 에피소드 수 부족 ({total_episodes} < {min_episodes}), 학습 건너뜀")
            return None
        
        logger.info(f"✅ 글로벌: {coins_processed}개 코인, {intervals_processed}개 인터벌, 총 {total_episodes}개 에피소드 수집 완료")
        
        # 설정 파일 로드
        if config_path is None:
            config_path = os.getenv(
                'HYBRID_CONFIG_PATH',
                '/workspace/rl_pipeline/hybrid/config_hybrid.json'
            )
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ 설정 파일 없음: {config_path}, 기본 설정 사용")
            config = {
                'train': {
                    'epochs': 50,  # 글로벌 학습은 더 많은 에포크
                    'batch_size': 8192,  # 더 큰 배치 크기
                    'lr': 0.0003,
                    'hidden_dim': 128
                },
                'paths': {
                    'checkpoints': '/workspace/rl_pipeline/artifacts/checkpoints',
                    'db': '/workspace/data_storage/rl_strategies.db'
                }
            }
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # 글로벌 학습용 설정 조정
            if 'train' in config:
                config['train']['epochs'] = config['train'].get('epochs', 30) * 2  # 글로벌은 2배
                config['train']['batch_size'] = config['train'].get('batch_size', 2048) * 2  # 2배
        
        # Trainer 초기화 및 학습
        logger.info(f"🌍 글로벌 신경망 학습 시작 ({total_episodes}개 에피소드, {coins_processed}개 코인, 글로벌 분석 데이터 포함)")

        try:
            trainer = PPOTrainer(config, session_id=session_id)
        except ImportError as e:
            logger.error(f"❌ PPOTrainer 초기화 실패 (JAX 관련): {e}")
            return None
        except Exception as e:
            logger.error(f"❌ PPOTrainer 초기화 실패: {e}")
            import traceback
            logger.debug(f"초기화 실패 상세:\n{traceback.format_exc()}")
            return None
        
        try:
            model_id = trainer.train_from_selfplay_data(
                all_episodes_data,
                db_path=config.get('paths', {}).get('db'),
                analysis_data=analysis_data  # 🔥 글로벌 분석 데이터 전달
            )
            
            if model_id:
                logger.info(f"🌍 글로벌 신경망 학습 완료: {model_id}")
                
                # 🔥 자동 평가 실행 (환경변수 체크)
                if should_auto_evaluate(model_id):
                    try:
                        # 평가는 첫 번째 코인-인터벌 조합으로 실행
                        first_coin = list(all_coin_selfplay.keys())[0] if all_coin_selfplay else None
                        if first_coin:
                            first_coin_data = all_coin_selfplay[first_coin]
                            first_interval = list(first_coin_data.keys())[0] if first_coin_data else None
                            
                            if first_interval:
                                from rl_pipeline.data.candles_loader import load_candles
                                
                                # 캔들 데이터 로드
                                candle_data = load_candles(first_coin, first_interval, days=30)
                                
                                if candle_data is not None and len(candle_data) > 0:
                                    # 전략 파라미터 추출
                                    strategy_params_list = []
                                    first_selfplay = first_coin_data.get(first_interval, {})
                                    if first_selfplay and 'cycle_results' in first_selfplay:
                                        for cycle in first_selfplay['cycle_results']:
                                            results = cycle.get('results', {})
                                            for agent_id, perf in results.items():
                                                if 'strategy_params' in perf:
                                                    strategy_params_list.append(perf['strategy_params'])
                                    
                                    # 중복 제거
                                    if strategy_params_list:
                                        seen = set()
                                        unique_params = []
                                        for params in strategy_params_list:
                                            params_str = str(sorted(params.items()))
                                            if params_str not in seen:
                                                seen.add(params_str)
                                                unique_params.append(params)
                                        strategy_params_list = unique_params[:10]
                                    
                                    if not strategy_params_list:
                                        strategy_params_list = [{
                                            'rsi_min': 30.0, 'rsi_max': 70.0,
                                            'volume_ratio_min': 1.0, 'volume_ratio_max': 2.0,
                                            'macd_buy_threshold': 0.01, 'macd_sell_threshold': -0.01,
                                            'stop_loss_pct': 0.02, 'take_profit_pct': 0.05
                                        }]
                                    
                                    # 자동 평가 실행
                                    eval_result = auto_evaluate_model(
                                        model_id=model_id,
                                        coin=first_coin,
                                        interval=first_interval,
                                        candle_data=candle_data,
                                        strategy_params_list=strategy_params_list,
                                        config=config
                                    )
                                    
                                    if eval_result:
                                        logger.info(f"✅ {first_coin}-{first_interval}: 글로벌 학습 자동 평가 완료")
                                        
                                        # 🔥 검증 결과 평가 및 재학습 판단
                                        passed, reason, details = evaluate_validation_results(eval_result)
                                        
                                        if passed:
                                            logger.info(f"✅ {first_coin}-{first_interval}: 검증 합격 - 모델 사용 가능")
                                            if details.get('warnings'):
                                                logger.info(f"   ⚠️ 경고 {len(details['warnings'])}개: {', '.join(details['warnings'][:2])}")
                                        else:
                                            logger.warning(f"⚠️ {first_coin}-{first_interval}: 검증 불합격 - {reason}")
                                            
                                            # 재학습 필요 여부 확인 (재시도 횟수 추적)
                                            needs_retrain, retrain_reason = should_retrain(eval_result, previous_attempts=previous_attempts)
                                            
                                            if needs_retrain:
                                                logger.warning(f"🔄 {first_coin}-{first_interval}: 재학습 권장 - {retrain_reason} (재시도: {previous_attempts}회)")
                                                
                                                # 재학습 제안 가져오기
                                                suggestions = get_retrain_suggestions(eval_result)
                                                logger.info(f"💡 재학습 제안: {suggestions.get('reason', '')}")
                                                
                                                # 🔥 자동 재학습 여부 확인 (환경변수)
                                                auto_retrain_enabled = os.getenv('ENABLE_AUTO_RETRAIN', 'false').lower() == 'true'
                                                
                                                if auto_retrain_enabled:
                                                    logger.info(f"🔄 {first_coin}-{first_interval}: 자동 재학습 시작 (환경변수 활성화됨, 재시도: {previous_attempts + 1}회)")
                                                    # 재학습 실행 (재시도 횟수 증가)
                                                    retrain_model_id = auto_train_from_global_strategies(
                                                        all_coin_selfplay=all_coin_selfplay,
                                                        all_coin_analysis=all_coin_analysis,
                                                        config_path=config_path,
                                                        min_episodes=min_episodes,
                                                        previous_attempts=previous_attempts + 1  # 🔥 재시도 횟수 증가
                                                    )
                                                    if retrain_model_id:
                                                        logger.info(f"✅ {first_coin}-{first_interval}: 재학습 완료: {retrain_model_id}")
                                                        return retrain_model_id
                                                    else:
                                                        logger.warning(f"⚠️ {first_coin}-{first_interval}: 재학습 실패")
                                                else:
                                                    logger.info(f"💡 자동 재학습 비활성화 (ENABLE_AUTO_RETRAIN=false), 수동 재학습 권장")
                                            else:
                                                logger.info(f"📊 {first_coin}-{first_interval}: 재학습 불필요 또는 최대 시도 횟수 초과")
                                    else:
                                        logger.warning(f"⚠️ {first_coin}-{first_interval}: 글로벌 학습 자동 평가 실패 (계속 진행)")
                    except Exception as eval_err:
                        logger.warning(f"⚠️ 글로벌: 학습 자동 평가 중 오류 (계속 진행): {eval_err}")
            else:
                logger.warning(f"⚠️ 글로벌: 학습 완료했지만 모델 ID가 없음")
            
            return model_id
        except Exception as train_err:
            logger.error(f"❌ 글로벌 학습 실행 중 오류: {train_err}")
            import traceback
            logger.debug(f"글로벌 학습 오류 상세:\n{traceback.format_exc()}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 글로벌 자동 학습 실패: {e}")
        import traceback
        logger.debug(f"글로벌 자동 학습 실패 상세:\n{traceback.format_exc()}")
        return None


def auto_evaluate_model(
    model_id: str,
    coin: str,
    interval: str,
    candle_data: Any,  # pd.DataFrame
    strategy_params_list: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    학습된 모델 자동 평가 (A/B 테스트)
    
    Args:
        model_id: 모델 ID
        coin: 코인 심볼
        interval: 인터벌
        candle_data: 캔들 데이터
        strategy_params_list: 전략 파라미터 리스트
        config: 설정 딕셔너리
    
    Returns:
        평가 결과 딕셔너리 (실패 시 None)
    """
    try:
        logger.info(f"🔍 {coin}-{interval}: 모델 평가 시작: {model_id}")
        
        # 설정 로드
        if config is None:
            config_path = os.getenv(
                'HYBRID_CONFIG_PATH',
                '/workspace/rl_pipeline/hybrid/config_hybrid.json'
            )
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
        
        db_path = config.get('paths', {}).get('db', '/workspace/data_storage/rl_strategies.db')
        
        # A/B 평가 실행
        result = evaluate_ab(
            model_id=model_id,
            mode='HYBRID',
            coin=coin,
            interval=interval,
            candle_data=candle_data,
            strategy_params_list=strategy_params_list,
            db_path=db_path,
            config=config
        )
        
        # 규칙 기반도 평가 (비교용)
        rule_result = None
        try:
            rule_result = evaluate_ab(
                model_id=None,
                mode='RULE',
                coin=coin,
                interval=interval,
                candle_data=candle_data,
                strategy_params_list=strategy_params_list,
                db_path=db_path,
                config=config
            )
            
            # 비교 결과 계산
            improvement = {
                'profit_factor_improvement': (
                    result['profit_factor'] - rule_result['profit_factor']
                ) / rule_result['profit_factor'] if rule_result['profit_factor'] > 0 else 0.0,
                'return_improvement': (
                    result['total_return'] - rule_result['total_return']
                ) / abs(rule_result['total_return']) if rule_result['total_return'] != 0 else 0.0,
                'win_rate_improvement': result['win_rate'] - rule_result['win_rate']
            }
            
            logger.info(f"✅ {coin}-{interval}: A/B 평가 완료")
            logger.info(f"   📊 Profit Factor: {rule_result['profit_factor']:.2f} → {result['profit_factor']:.2f} ({improvement['profit_factor_improvement']:+.1%})")
            logger.info(f"   📊 Return: {rule_result['total_return']:.2%} → {result['total_return']:.2%} ({improvement['return_improvement']:+.1%})")
            logger.info(f"   📊 Win Rate: {rule_result['win_rate']:.2%} → {result['win_rate']:.2%} ({improvement['win_rate_improvement']:+.1%})")
            
            result['comparison'] = {
                'rule': rule_result,
                'hybrid': result,
                'improvement': improvement
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 규칙 기반 평가 실패 (계속 진행): {e}")
        
        # 🔥 Walk-Forward 검증 실행
        walk_forward_result = None
        try:
            logger.info(f"🔍 Walk-Forward 검증 시작...")
            walk_forward_result = walk_forward_validation(
                model_id=model_id,
                coin=coin,
                interval=interval,
                candle_data=candle_data,
                strategy_params_list=strategy_params_list,
                train_ratio=0.7,
                db_path=db_path,
                config=config
            )
            
            if walk_forward_result.get('status') == 'success':
                logger.info(f"✅ Walk-Forward 검증 완료")
                if walk_forward_result.get('has_overfitting'):
                    logger.warning(f"   ⚠️ 과적합 가능성 감지")
                else:
                    logger.info(f"   ✅ 과적합 없음 확인")
            else:
                logger.info(f"   📊 Walk-Forward 검증 건너뜀: {walk_forward_result.get('reason', 'unknown')}")
                
        except Exception as e:
            logger.warning(f"⚠️ Walk-Forward 검증 실패 (계속 진행): {e}")
        
        # 🔥 다중 기간 검증 실행
        multi_period_result = None
        try:
            logger.info(f"🔍 다중 기간 검증 시작...")
            multi_period_result = multi_period_validation(
                model_id=model_id,
                coin=coin,
                interval=interval,
                candle_data=candle_data,
                strategy_params_list=strategy_params_list,
                db_path=db_path,
                config=config
            )
            
            if multi_period_result.get('status') == 'success':
                logger.info(f"✅ 다중 기간 검증 완료")
                consistency = multi_period_result.get('consistency', 0.0)
                regime_count = multi_period_result.get('regime_count', 0)
                logger.info(f"   📊 일관성: {consistency:.1%}, 레짐 수: {regime_count}개")
            else:
                logger.info(f"   📊 다중 기간 검증 건너뜀: {multi_period_result.get('reason', 'unknown')}")
                
        except Exception as e:
            logger.warning(f"⚠️ 다중 기간 검증 실패 (계속 진행): {e}")
        
        # 🔥 통합 결과 반환
        result['walk_forward'] = walk_forward_result
        result['multi_period'] = multi_period_result
        
        logger.info(f"✅ {coin}-{interval}: 전체 평가 완료 (A/B + Walk-Forward + 다중 기간)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 모델 평가 실패: {e}")
        import traceback
        logger.debug(f"모델 평가 실패 상세:\n{traceback.format_exc()}")
        return None


def should_auto_train(coin: str, interval: str, selfplay_result: Dict[str, Any], min_episodes: int = 10) -> bool:
    """
    자동 학습 조건 체크
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        selfplay_result: Self-play 결과
        min_episodes: 최소 에피소드 수
    
    Returns:
        자동 학습 여부
    """
    try:
        # 환경변수 체크
        auto_train_enabled = os.getenv('ENABLE_AUTO_TRAINING', 'false').lower() == 'true'
        use_hybrid = os.getenv('USE_HYBRID', 'false').lower() == 'true'
        
        if not auto_train_enabled:
            return False
        if not use_hybrid:
            return False
        
        # Self-play 결과 체크
        if not selfplay_result:
            return False
        
        status = selfplay_result.get('status')
        if status != 'success':
            return False
        
        # 에피소드 수 체크
        cycle_results = selfplay_result.get('cycle_results', [])
        if len(cycle_results) < min_episodes:
            return False
        
        # JAX 가용성 체크
        try:
            import jax
            from rl_pipeline.hybrid.neural_policy_jax import JAX_AVAILABLE
            if not JAX_AVAILABLE:
                return False
        except ImportError:
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ 자동 학습 조건 체크 실패: {e}")
        return False


def should_auto_evaluate(model_id: str) -> bool:
    """
    자동 평가 조건 체크
    
    Args:
        model_id: 모델 ID
    
    Returns:
        자동 평가 여부
    """
    try:
        # 환경변수 체크
        auto_eval_enabled = os.getenv('ENABLE_AUTO_EVALUATION', 'true').lower() == 'true'
        use_hybrid = os.getenv('USE_HYBRID', 'false').lower() == 'true'
        
        if not auto_eval_enabled:
            return False
        if not use_hybrid:
            return False
        
        # 모델 ID 체크
        if not model_id:
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ 자동 평가 조건 체크 실패: {e}")
        return False
