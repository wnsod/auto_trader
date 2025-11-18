"""
AI 학습 데이터 수집 모듈
전략 성능 데이터 수집 및 학습 지원
"""

import logging
import pandas as pd
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from rl_pipeline.core.types import Strategy
from rl_pipeline.core.errors import StrategyError
from rl_pipeline.core.env import config
from rl_pipeline.core.utils import format_strategy_data
from rl_pipeline.data import load_candles, ensure_indicators
from rl_pipeline.strategy.param_space import sample_param_grid
from rl_pipeline.strategy.factory import make_strategy
from rl_pipeline.strategy.serializer import serialize_strategy
from rl_pipeline.db.writes import write_batch
from rl_pipeline.db.connection_pool import get_optimized_db_connection

logger = logging.getLogger(__name__)

def collect_strategy_performance_for_ai(id: str, coin: str, interval: str, 

                                       performance_data: Dict[str, Any]) -> bool:

    """AI 학습을 위한 전략 성능 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        from datetime import datetime

        

        performance_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO strategy_performance_history (

                    performance_id, id, coin, interval, execution_time,

                    profit_loss, win_rate, max_drawdown, sharpe_ratio, total_trades,

                    market_condition, performance_score, created_at

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                performance_id, id, coin, interval, datetime.now(),

                performance_data.get('profit_loss', 0.0),

                performance_data.get('win_rate', 0.0),

                performance_data.get('max_drawdown', 0.0),

                performance_data.get('sharpe_ratio', 0.0),

                performance_data.get('total_trades', 0),

                performance_data.get('market_condition', 'unknown'),

                performance_data.get('performance_score', 0.0),

                datetime.now()

            ))

            

            conn.commit()

            logger.debug(f"✅ AI 학습용 전략 성능 데이터 수집 완료: {id}")

            return True

            

    except Exception as e:

        logger.error(f"❌ AI 학습용 전략 성능 데이터 수집 실패: {e}")

        return False



def collect_strategy_comparison_for_ai(strategy_a_id: str, strategy_b_id: str, 

                                     coin: str, interval: str, 

                                     comparison_data: Dict[str, Any]) -> bool:

    """AI 학습을 위한 전략 비교 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        from datetime import datetime

        

        comparison_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO strategy_comparison_matrix (

                    comparison_id, strategy_a_id, strategy_b_id, coin, interval,

                    comparison_metric, strategy_a_score, strategy_b_score,

                    winner_id, confidence_level, comparison_date, created_at

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                comparison_id, strategy_a_id, strategy_b_id, coin, interval,

                comparison_data.get('comparison_metric', 'profit_loss'),

                comparison_data.get('strategy_a_score', 0.0),

                comparison_data.get('strategy_b_score', 0.0),

                comparison_data.get('winner_id', strategy_a_id),

                comparison_data.get('confidence_level', 0.5),

                datetime.now(),

                datetime.now()

            ))

            

            conn.commit()

            logger.debug(f"✅ AI 학습용 전략 비교 데이터 수집 완료: {strategy_a_id} vs {strategy_b_id}")

            return True

            

    except Exception as e:

        logger.error(f"❌ AI 학습용 전략 비교 데이터 수집 실패: {e}")

        return False



def collect_learning_episode_for_ai(coin: str, interval: str, 

                                   episode_data: Dict[str, Any]) -> str:

    """AI 학습을 위한 에피소드 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        from datetime import datetime

        

        episode_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO learning_episodes (

                    episode_id, coin, interval, start_time, end_time,

                    total_reward, episode_length, success_rate,

                    market_condition, volatility_level, created_at

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                episode_id, coin, interval, 

                episode_data.get('start_time', datetime.now()),

                episode_data.get('end_time'),

                episode_data.get('total_reward', 0.0),

                episode_data.get('episode_length', 0),

                episode_data.get('success_rate', 0.0),

                episode_data.get('market_condition', 'unknown'),

                episode_data.get('volatility_level', 'medium'),

                datetime.now()

            ))

            

            conn.commit()

            logger.debug(f"✅ AI 학습용 에피소드 데이터 수집 완료: {episode_id}")

            return episode_id

            

    except Exception as e:

        logger.error(f"❌ AI 학습용 에피소드 데이터 수집 실패: {e}")

        return ""



def collect_learning_state_for_ai(episode_id: str, step_number: int,

                                 state_data: Dict[str, Any]) -> str:

    """AI 학습을 위한 상태 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        import json

        from datetime import datetime

        

        state_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO learning_states (

                    state_id, episode_id, step_number, market_data,

                    technical_indicators, market_condition, volatility_level,

                    state_vector, created_at

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                state_id, episode_id, step_number,

                json.dumps(state_data.get('market_data', {})),

                json.dumps(state_data.get('technical_indicators', {})),

                state_data.get('market_condition', 'unknown'),

                state_data.get('volatility_level', 'medium'),

                json.dumps(state_data.get('state_vector', [])),

                datetime.now()

            ))

            

            conn.commit()

            logger.debug(f"✅ AI 학습용 상태 데이터 수집 완료: {state_id}")

            return state_id

            

    except Exception as e:

        logger.error(f"❌ AI 학습용 상태 데이터 수집 실패: {e}")

        return ""



def collect_learning_action_for_ai(state_id: str, action_type: str,

                                   action_params: Dict[str, Any],

                                   confidence_score: float = 0.0) -> str:

    """AI 학습을 위한 행동 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        import json

        from datetime import datetime

        

        action_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO learning_actions (

                    action_id, state_id, action_type, action_params,

                    confidence_score, created_at

                ) VALUES (?, ?, ?, ?, ?, ?)

            """, (

                action_id, state_id, action_type,

                json.dumps(action_params),

                confidence_score,

                datetime.now()

            ))

            

            conn.commit()

            logger.debug(f"✅ AI 학습용 행동 데이터 수집 완료: {action_id}")

            return action_id

            

    except Exception as e:

        logger.error(f"❌ AI 학습용 행동 데이터 수집 실패: {e}")

        return ""



def collect_learning_reward_for_ai(action_id: str, reward_value: float,

                                  reward_type: str, reward_details: Dict[str, Any] = None) -> bool:

    """AI 학습을 위한 보상 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        import json

        from datetime import datetime

        

        reward_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO learning_rewards (

                    reward_id, action_id, reward_value, reward_type,

                    reward_details, created_at

                ) VALUES (?, ?, ?, ?, ?, ?)

            """, (

                reward_id, action_id, reward_value, reward_type,

                json.dumps(reward_details or {}),

                datetime.now()

            ))

            

            conn.commit()

            logger.debug(f"✅ AI 학습용 보상 데이터 수집 완료: {reward_id}")

            return True

            

    except Exception as e:

        logger.error(f"❌ AI 학습용 보상 데이터 수집 실패: {e}")

        return False



def collect_model_training_data_for_ai(model_type: str, coin: str, interval: str,

                                       training_data: Dict[str, Any]) -> str:

    """AI 학습을 위한 모델 학습 데이터 수집"""

    try:

        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        from rl_pipeline.core.env import config

        import uuid

        import json

        from datetime import datetime

        

        training_id = str(uuid.uuid4())

        

        with get_optimized_db_connection(config.STRATEGIES_DB) as conn:

            cursor = conn.cursor()

            

            cursor.execute("""

                INSERT INTO model_training_data (

                    training_id, model_type, coin, interval, training_data,

                    validation_data, test_data, feature_columns, target_column,

                    data_quality_score, training_date, created_at

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                training_id, model_type, coin, interval,

                json.dumps(training_data.get('training_data', {})),

                json.dumps(training_data.get('validation_data', {})),

                json.dumps(training_data.get('test_data', {})),

                json.dumps(training_data.get('feature_columns', [])),

                training_data.get('target_column', 'profit_loss'),

                training_data.get('data_quality_score', 0.5),

                datetime.now(),

                datetime.now()

            ))

            

            conn.commit()

        logger.debug(f"✅ AI 학습용 모델 학습 데이터 수집 완료: {training_id}")

        return training_id

        

    except Exception as e:

        logger.error(f"❌ AI 학습용 모델 학습 데이터 수집 실패: {e}")

        return ""



# ===== 동적 반복 제어 함수들 =====


