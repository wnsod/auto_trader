"""
학습 데이터 검증 및 분석 도구

Self-play에서 수집된 데이터가 올바르게 학습에 사용되는지 확인
"""

import logging
import json
from typing import Dict, List, Any, Optional
import numpy as np
from rl_pipeline.hybrid.trainer_jax import PPOTrainer
from rl_pipeline.hybrid.auto_trainer import collect_selfplay_data_for_training

logger = logging.getLogger(__name__)


def verify_training_data(
    coin: str,
    interval: str,
    selfplay_result: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    학습 데이터 검증
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        selfplay_result: Self-play 결과
        verbose: 상세 출력 여부
    
    Returns:
        검증 결과 딕셔너리
    """
    results = {
        'coin': coin,
        'interval': interval,
        'status': 'unknown',
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # 1. Self-play 데이터 수집
        episodes_data = collect_selfplay_data_for_training(
            coin, interval, selfplay_result, min_episodes=1
        )
        
        if not episodes_data:
            results['status'] = 'failed'
            results['issues'].append('에피소드 데이터가 없습니다')
            return results
        
        results['stats']['episodes_count'] = len(episodes_data)
        
        # 2. Trainer로 경험 추출 시뮬레이션
        # 임시 Trainer 생성 (실제 학습 없이 검증만)
        dummy_config = {
            'train': {
                'epochs': 1,
                'batch_size': 1024,
                'lr': 0.0003,
                'hidden_dim': 128
            }
        }
        
        try:
            trainer = PPOTrainer(dummy_config)
            experiences = trainer._extract_experiences(episodes_data)
        except Exception as e:
            results['status'] = 'failed'
            results['issues'].append(f'경험 추출 실패: {e}')
            return results
        
        if not experiences:
            results['status'] = 'failed'
            results['issues'].append('추출된 경험이 없습니다')
            return results
        
        results['stats']['experiences_count'] = len(experiences)
        
        # 3. 경험 데이터 품질 검증
        quality_issues = _verify_experience_quality(experiences, verbose)
        results['warnings'].extend(quality_issues['warnings'])
        results['issues'].extend(quality_issues['errors'])
        
        # 4. 통계 정보
        results['stats'].update(_calculate_experience_stats(experiences))
        
        # 5. 학습 가능 여부 판단
        if len(experiences) < 100:
            results['warnings'].append(f'경험 데이터가 부족합니다 ({len(experiences)}개 < 100개 권장)')
        
        if quality_issues['errors']:
            results['status'] = 'failed'
        elif quality_issues['warnings']:
            results['status'] = 'warning'
        else:
            results['status'] = 'ok'
        
        # 6. 상세 출력
        if verbose:
            _print_verification_report(results)
        
    except Exception as e:
        results['status'] = 'error'
        results['issues'].append(f'검증 중 오류: {e}')
        logger.error(f"❌ 검증 실패: {e}")
    
    return results


def _verify_experience_quality(experiences: List[Dict[str, Any]], verbose: bool) -> Dict[str, List[str]]:
    """경험 데이터 품질 검증"""
    warnings = []
    errors = []
    
    state_count = 0
    action_count = 0
    reward_count = 0
    valid_count = 0
    
    state_features = {}
    action_dist = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
    reward_stats = []
    
    for i, exp in enumerate(experiences):
        # 필수 필드 확인
        if 'state' not in exp:
            errors.append(f'경험 #{i}: state 필드 없음')
            continue
        state_count += 1
        
        if 'action' not in exp:
            errors.append(f'경험 #{i}: action 필드 없음')
            continue
        action_count += 1
        
        if 'reward' not in exp:
            errors.append(f'경험 #{i}: reward 필드 없음')
            continue
        reward_count += 1
        
        # State 검증
        state = exp['state']
        if isinstance(state, dict):
            # State 딕셔너리 필드 확인
            required_fields = ['rsi', 'macd', 'volume_ratio', 'price', 'close']
            missing_fields = [f for f in required_fields if f not in state]
            if missing_fields:
                warnings.append(f'경험 #{i}: State 필드 부족: {missing_fields}')
            
            # State 통계 수집
            for key, value in state.items():
                if key not in state_features:
                    state_features[key] = []
                if isinstance(value, (int, float)):
                    state_features[key].append(value)
        
        # Action 검증
        action = exp.get('action')
        if action in action_dist:
            action_dist[action] += 1
        elif action is None:
            errors.append(f'경험 #{i}: action이 None')
        else:
            warnings.append(f'경험 #{i}: 유효하지 않은 action 값: {action}')
        
        # Reward 검증
        reward = exp.get('reward', 0.0)
        if isinstance(reward, (int, float)):
            reward_stats.append(reward)
            if abs(reward) > 1000:
                warnings.append(f'경험 #{i}: 과도한 reward 값: {reward}')
            if np.isnan(reward) or np.isinf(reward):
                errors.append(f'경험 #{i}: NaN/Inf reward: {reward}')
        else:
            errors.append(f'경험 #{i}: reward 타입 오류: {type(reward)}')
        
        # Log prob, value 검증
        if 'log_prob' not in exp:
            warnings.append(f'경험 #{i}: log_prob 없음 (기본값 사용됨)')
        if 'value' not in exp:
            warnings.append(f'경험 #{i}: value 없음 (기본값 사용됨)')
        
        valid_count += 1
    
    # 통계 기반 경고
    if len(experiences) > 0:
        if action_dist[0] == 0 and action_dist[1] == 0 and action_dist[2] == 0:
            errors.append('모든 action이 유효하지 않음')
        elif max(action_dist.values()) / len(experiences) > 0.9:
            warnings.append(f'Action 분포가 불균형함: {action_dist}')
        
        if reward_stats:
            reward_mean = np.mean(reward_stats)
            reward_std = np.std(reward_stats)
            if abs(reward_mean) > 10:
                warnings.append(f'Reward 평균이 과도함: {reward_mean:.4f}')
            if reward_std > 100:
                warnings.append(f'Reward 표준편차가 과도함: {reward_std:.4f}')
    
    if verbose and valid_count > 0:
        logger.info(f"✅ 유효한 경험: {valid_count}/{len(experiences)}")
        logger.info(f"   Action 분포: HOLD={action_dist[0]}, BUY={action_dist[1]}, SELL={action_dist[2]}")
        if reward_stats:
            logger.info(f"   Reward 통계: mean={np.mean(reward_stats):.4f}, std={np.std(reward_stats):.4f}")
    
    return {'warnings': warnings, 'errors': errors}


def _calculate_experience_stats(experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """경험 데이터 통계 계산"""
    stats = {
        'total_experiences': len(experiences),
        'unique_episodes': len(set(exp.get('episode', 0) for exp in experiences)),
        'actions': {},
        'rewards': {},
        'states': {}
    }
    
    if not experiences:
        return stats
    
    # Action 분포
    actions = [exp.get('action', 0) for exp in experiences]
    stats['actions'] = {
        'HOLD': actions.count(0),
        'BUY': actions.count(1),
        'SELL': actions.count(2),
        'invalid': len([a for a in actions if a not in [0, 1, 2]])
    }
    
    # Reward 통계
    rewards = [float(exp.get('reward', 0.0)) for exp in experiences]
    if rewards:
        rewards_arr = np.array(rewards)
        stats['rewards'] = {
            'mean': float(np.mean(rewards_arr)),
            'std': float(np.std(rewards_arr)),
            'min': float(np.min(rewards_arr)),
            'max': float(np.max(rewards_arr)),
            'positive_count': int(np.sum(rewards_arr > 0)),
            'negative_count': int(np.sum(rewards_arr < 0))
        }
    
    # State 필드 통계
    if experiences and 'state' in experiences[0]:
        first_state = experiences[0]['state']
        if isinstance(first_state, dict):
            stats['states'] = {
                'type': 'dict',
                'fields': list(first_state.keys()),
                'field_count': len(first_state)
            }
        else:
            stats['states'] = {
                'type': type(first_state).__name__
            }
    
    return stats


def _print_verification_report(results: Dict[str, Any]):
    """검증 결과 출력"""
    print("\n" + "="*60)
    print(f"📊 학습 데이터 검증 결과: {results['coin']}-{results['interval']}")
    print("="*60)
    
    print(f"\n✅ 상태: {results['status'].upper()}")
    
    print(f"\n📈 통계:")
    for key, value in results['stats'].items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    if results['warnings']:
        print(f"\n⚠️  경고 ({len(results['warnings'])}개):")
        for warning in results['warnings'][:10]:  # 최대 10개만
            print(f"   - {warning}")
        if len(results['warnings']) > 10:
            print(f"   ... 외 {len(results['warnings']) - 10}개")
    
    if results['issues']:
        print(f"\n❌ 문제 ({len(results['issues'])}개):")
        for issue in results['issues'][:10]:  # 최대 10개만
            print(f"   - {issue}")
        if len(results['issues']) > 10:
            print(f"   ... 외 {len(results['issues']) - 10}개")
    
    print("\n" + "="*60 + "\n")


def test_training_data_extraction(
    coin: str = '0G',
    interval: str = '15m'
) -> None:
    """
    학습 데이터 추출 테스트 (실제 Self-play 결과 없이)
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
    """
    logger.info(f"🧪 학습 데이터 추출 테스트 시작: {coin}-{interval}")
    
    # 더미 Self-play 결과 생성
    dummy_result = {
        "cycle_results": [
            {
                "episode": i,
                "regime_label": "neutral",
                "results": {
                    f"agent_{j}": {
                        "total_trades": 10,
                        "win_rate": 0.5,
                        "total_pnl": 100.0 * (j + 1),
                        "trades": [
                            {
                                "direction": "BUY" if j % 2 == 0 else "SELL",
                                "pnl": 10.0 * (j + 1),
                                "rsi": 50.0 + j,
                                "macd": 0.01 * j,
                                "volume_ratio": 1.0 + j * 0.1,
                                "atr": 0.02,
                                "adx": 25.0,
                                "mfi": 50.0,
                                "close": 50000.0,
                                "bb_upper": 51000.0,
                                "bb_middle": 50000.0,
                                "bb_lower": 49000.0
                            }
                            for _ in range(5)
                        ]
                    }
                    for j in range(4)
                }
            }
            for i in range(20)
        ]
    }
    
    # 검증 실행
    results = verify_training_data(coin, interval, dummy_result, verbose=True)
    
    # 결과 반환
    return results


if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    test_training_data_extraction()

