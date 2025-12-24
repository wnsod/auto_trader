"""
증분 학습 (Incremental Learning) 트레이너

유사도 기반 전략 분류에 따라 차등적인 학습 전략 적용:
- duplicate: 학습 건너뜀 (이미 creator에서 제거됨)
- copy: 부모 정책 복사 (3 에피소드)
- finetune: 부모 기반 미세 조정 (7-12 에피소드)
- novel: 전체 학습 (20 에피소드)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def save_training_history(
    strategy_id: str,
    training_episodes: int,
    avg_accuracy: float,
    parent_strategy_id: Optional[str] = None,
    similarity_score: float = 0.0,
    training_source: str = 'trained',
    policy_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    전략 학습 이력을 DB에 저장

    Args:
        strategy_id: 전략 ID
        training_episodes: 학습에 사용된 에피소드 수
        avg_accuracy: 평균 정확도
        parent_strategy_id: 부모 전략 ID (있으면)
        similarity_score: 유사도 점수
        training_source: 'trained', 'copied', 'finetuned'
        policy_data: 정책 데이터 (복사된 경우)

    Returns:
        성공 여부
    """
    try:
        import time
        import random
        from rl_pipeline.db.connection_pool import get_strategy_db_pool

        # 최대 재시도 횟수 설정
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                pool = get_strategy_db_pool()
                with pool.get_connection() as conn:
                    cursor = conn.cursor()

                    # 기존 이력이 있으면 업데이트, 없으면 삽입
                    cursor.execute("""
                        INSERT OR REPLACE INTO strategy_training_history (
                            strategy_id,
                            trained_at,
                            training_episodes,
                            avg_accuracy,
                            parent_strategy_id,
                            similarity_score,
                            training_source,
                            policy_data,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        strategy_id,
                        datetime.now().isoformat(),
                        training_episodes,
                        avg_accuracy,
                        parent_strategy_id,
                        similarity_score,
                        training_source,
                        json.dumps(policy_data) if policy_data else None
                    ))

                    conn.commit()
                    logger.info(f"✅ 학습 이력 저장: {strategy_id} ({training_source}, {training_episodes}ep, acc={avg_accuracy:.3f})")
                    return True
            
            except Exception as e:
                is_locked = "database is locked" in str(e) or "disk I/O error" in str(e) or "malformed" in str(e)
                if is_locked and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                    logger.warning(f"⚠️ 학습 이력 저장 일시적 실패 ({attempt+1}/{max_retries}), {wait_time:.2f}초 후 재시도: {strategy_id} - {e}")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ 학습 이력 저장 실패 (최종): {strategy_id} - {e}")
                    raise e

    except Exception as e:
        logger.error(f"❌ 학습 이력 저장 실패: {strategy_id} - {e}")
        return False


def _has_parent_policy_data(parent_strategy_id: str) -> bool:
    """
    부모 전략에 정책 데이터가 있는지 사전 검증
    
    Args:
        parent_strategy_id: 부모 전략 ID
        
    Returns:
        policy_data 존재 여부
    """
    try:
        from rl_pipeline.db.connection_pool import get_strategy_db_pool

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 부모 전략의 policy_data 존재 여부만 확인
            cursor.execute("""
                SELECT policy_data
                FROM strategy_training_history
                WHERE strategy_id = ? AND policy_data IS NOT NULL
                ORDER BY trained_at DESC
                LIMIT 1
            """, (parent_strategy_id,))

            row = cursor.fetchone()
            return row is not None and row[0] is not None

    except Exception as e:
        logger.debug(f"부모 정책 데이터 검증 실패: {parent_strategy_id} - {e}")
        return False


def load_parent_policy(parent_strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    부모 전략의 학습된 정책 로드

    Args:
        parent_strategy_id: 부모 전략 ID

    Returns:
        정책 데이터 (dict) 또는 None
    """
    try:
        from rl_pipeline.db.connection_pool import get_strategy_db_pool

        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # 부모 전략의 정책 데이터 조회
            cursor.execute("""
                SELECT policy_data
                FROM strategy_training_history
                WHERE strategy_id = ? AND policy_data IS NOT NULL
                ORDER BY trained_at DESC
                LIMIT 1
            """, (parent_strategy_id,))

            row = cursor.fetchone()
            if row and row[0]:
                policy_data = json.loads(row[0])
                logger.debug(f"✅ 부모 정책 로드: {parent_strategy_id}")
                return policy_data
            else:
                logger.debug(f"ℹ️ 부모 정책 데이터 없음: {parent_strategy_id} (정상: novel로 처리)")
                return None

    except Exception as e:
        logger.error(f"❌ 부모 정책 로드 실패: {parent_strategy_id} - {e}")
        return None


def copy_parent_policy(strategy: Dict[str, Any]) -> bool:
    """
    부모 전략의 정책을 현재 전략으로 복사

    Args:
        strategy: 전략 (similarity_classification='copy')

    Returns:
        성공 여부 (실패 시 strategy를 novel로 재분류)
    """
    try:
        strategy_id = strategy.get('id')
        if not strategy_id:
            logger.error(f"❌ 전략 ID 없음, 정책 복사 불가")
            return False

        parent_id = strategy.get('parent_strategy_id')
        similarity_score = strategy.get('similarity_score', 0.0)

        if not parent_id:
            logger.debug(f"ℹ️ {strategy_id}: 부모 ID 없음, novel로 재분류")
            # copy → novel로 재분류
            strategy['similarity_classification'] = 'novel'
            strategy['parent_strategy_id'] = None
            return False

        # 🔥 사전 검증: 부모 전략의 policy_data 존재 여부 확인
        if not _has_parent_policy_data(parent_id):
            logger.debug(f"ℹ️ {strategy_id}: 부모 정책 데이터 없음 (부모: {parent_id}), novel로 재분류")
            # copy → novel로 재분류
            strategy['similarity_classification'] = 'novel'
            strategy['parent_strategy_id'] = None
            return False

        # 부모 정책 로드
        parent_policy = load_parent_policy(parent_id)

        if not parent_policy:
            logger.debug(f"ℹ️ {strategy_id}: 부모 정책 로드 실패, novel로 재분류")
            # copy → novel로 재분류
            strategy['similarity_classification'] = 'novel'
            strategy['parent_strategy_id'] = None
            return False

        # 학습 이력 저장 (복사)
        save_training_history(
            strategy_id=strategy_id,
            training_episodes=3,  # 복사는 3 에피소드로 기록
            avg_accuracy=0.95,  # 부모와 거의 동일하다고 가정
            parent_strategy_id=parent_id,
            similarity_score=similarity_score,
            training_source='copied',
            policy_data=parent_policy
        )

        logger.info(f"✅ {strategy_id}: 부모 정책 복사 완료 (부모: {parent_id}, 유사도: {similarity_score:.3f})")
        return True

    except Exception as e:
        logger.error(f"❌ 정책 복사 실패: {strategy.get('id')} - {e}")
        # 예외 발생 시에도 novel로 재분류
        strategy['similarity_classification'] = 'novel'
        strategy['parent_strategy_id'] = None
        return False


def train_strategies_incremental(
    strategies: List[Dict[str, Any]],
    episodes_data: List[Dict[str, Any]],
    trainer,
    db_path: Optional[str] = None,
    analysis_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    증분 학습: 유사도 분류에 따라 차등 학습

    Args:
        strategies: 전략 리스트 (similarity_classification 메타데이터 포함)
        episodes_data: Self-play 에피소드 데이터
        trainer: PPOTrainer 인스턴스
        db_path: DB 경로
        analysis_data: 통합 분석 데이터

    Returns:
        모델 ID (성공 시) 또는 None
    """
    try:
        # 전략 분류별 카운트
        copy_count = sum(1 for s in strategies if s.get('similarity_classification') == 'copy')
        finetune_count = sum(1 for s in strategies if s.get('similarity_classification') == 'finetune')
        novel_count = sum(1 for s in strategies if s.get('similarity_classification') == 'novel')

        logger.info(f"🔄 증분 학습 시작:")
        logger.info(f"  - 정책 복사(copy): {copy_count}개")
        logger.info(f"  - 미세 조정(finetune): {finetune_count}개")
        logger.info(f"  - 신규 학습(novel): {novel_count}개")

        # 1. 정책 복사 전략 처리
        copied_strategies = []
        reclassified_to_novel = []
        
        for strategy in strategies:
            if strategy.get('similarity_classification') == 'copy':
                if copy_parent_policy(strategy):
                    copied_strategies.append(strategy)
                else:
                    # copy 실패 시 novel로 재분류되었는지 확인
                    if strategy.get('similarity_classification') == 'novel':
                        reclassified_to_novel.append(strategy)

        if copied_strategies:
            logger.info(f"✅ 정책 복사 완료: {len(copied_strategies)}개")
        
        if reclassified_to_novel:
            logger.debug(f"ℹ️ 부모 정책 없음으로 novel로 재분류: {len(reclassified_to_novel)}개")
            # 재분류된 전략을 novel 리스트에 추가
            novel_count += len(reclassified_to_novel)

        # 2. 미세 조정 전략 처리 (episode 데이터를 줄여서 학습)
        finetune_strategies = [s for s in strategies if s.get('similarity_classification') == 'finetune']

        if finetune_strategies:
            logger.info(f"🔥 미세 조정 학습 시작: {finetune_strategies}개 전략")

            # 에피소드 데이터를 finetune 전략에 맞게 필터링
            # TODO: 실제로는 전략별로 데이터를 분리해야 하지만, 일단 모든 데이터 사용
            # 대신 epochs를 줄여서 학습 시간 단축

            # epochs를 기본의 40%로 줄임 (30 → 12)
            original_epochs = trainer.train_config.get('epochs', 30)
            trainer.train_config['epochs'] = int(original_epochs * 0.4)

            try:
                model_id = trainer.train_from_selfplay_data(
                    episodes_data,
                    db_path=db_path,
                    analysis_data=analysis_data
                )

                if model_id:
                    # 학습 이력 저장
                    for strategy in finetune_strategies:
                        save_training_history(
                            strategy_id=strategy.get('id'),
                            training_episodes=int(original_epochs * 0.4),
                            avg_accuracy=0.85,  # 임시값, 실제로는 학습 결과에서 가져와야 함
                            parent_strategy_id=strategy.get('parent_strategy_id'),
                            similarity_score=strategy.get('similarity_score', 0.0),
                            training_source='finetuned'
                        )

                    logger.info(f"✅ 미세 조정 완료: {len(finetune_strategies)}개 전략")

            finally:
                # epochs 복원
                trainer.train_config['epochs'] = original_epochs

        # 3. 신규 전략 전체 학습 (재분류된 전략 포함)
        novel_strategies = [s for s in strategies if s.get('similarity_classification') == 'novel']

        if novel_strategies:
            if reclassified_to_novel:
                logger.info(f"🔥 신규 전략 전체 학습 시작: {len(novel_strategies)}개 전략 (재분류 포함: {len(reclassified_to_novel)}개)")
            else:
                logger.info(f"🔥 신규 전략 전체 학습 시작: {len(novel_strategies)}개 전략")

            model_id = trainer.train_from_selfplay_data(
                episodes_data,
                db_path=db_path,
                analysis_data=analysis_data
            )

            if model_id:
                # 학습 이력 저장
                for strategy in novel_strategies:
                    save_training_history(
                        strategy_id=strategy.get('id'),
                        training_episodes=trainer.train_config.get('epochs', 30),
                        avg_accuracy=0.75,  # 임시값, 실제로는 학습 결과에서 가져와야 함
                        parent_strategy_id=None,
                        similarity_score=0.0,
                        training_source='trained'
                    )

                logger.info(f"✅ 신규 전략 학습 완료: {len(novel_strategies)}개")
                return model_id

        logger.info(f"✅ 증분 학습 완료: copy={len(copied_strategies)}, finetune={len(finetune_strategies)}, novel={len(novel_strategies)}")
        return None

    except Exception as e:
        logger.error(f"❌ 증분 학습 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
