
import os
import sys
import logging
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rl_pipeline.core.env import config
from rl_pipeline.db.writes import write_batch
from rl_pipeline.db.connection_pool import get_optimized_db_connection

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_db_path_and_write():
    coin = 'BTC'
    
    # 1. DB 경로 확인
    db_path = config.get_strategy_db_path(coin)
    logger.info(f"🔍 [Check 1] config.get_strategy_db_path('{coin}'): {db_path}")
    
    # 2. 디렉토리/파일 상태 확인
    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            logger.error(f"❌ [Check 2] DB Path is a directory! {db_path}")
        else:
            logger.info(f"✅ [Check 2] DB Path exists and is a file.")
    else:
        logger.info(f"ℹ️ [Check 2] DB Path does not exist yet (will be created).")
        
    # 3. 테이블 생성 확인 (db/schema.py 사용)
    try:
        from rl_pipeline.db.schema import create_strategies_table
        create_strategies_table(db_path)
        logger.info(f"✅ [Check 3] Table creation successful.")
    except Exception as e:
        logger.error(f"❌ [Check 3] Table creation failed: {e}")
        return

    # 4. 더미 데이터 저장 시도
    dummy_strategy = {
        'id': 'test_strategy_001',
        'coin': coin,
        'interval': '15m',
        'strategy_type': 'test',
        'params': {'rsi_min': 30, 'rsi_max': 70},  # JSON 변환 필요 없음 (write_batch 내부 처리에 따라 다름, manager.py에서는 json.dumps 함)
        'created_at': datetime.now().isoformat(),
        'quality_grade': 'A',
        'is_active': 1
    }
    
    # manager.py의 로직을 흉내내어 확장 스키마로 변환
    import json
    expanded = {
        'id': dummy_strategy['id'],
        'coin': dummy_strategy['coin'],
        # 'symbol': dummy_strategy['coin'], # symbol 제거
        'interval': dummy_strategy['interval'],
        'strategy_type': dummy_strategy['strategy_type'],
        'strategy_conditions': json.dumps(dummy_strategy['params']),
        'description': 'Test Strategy',
        'created_at': dummy_strategy['created_at'],
        # 필수 필드들
        'rsi_min': 30.0,
        'rsi_max': 70.0,
        'volume_ratio_min': 1.0,
        'volume_ratio_max': 2.0,
        'macd_buy_threshold': 0.0,
        'macd_sell_threshold': 0.0,
        'mfi_min': 20.0,
        'mfi_max': 80.0,
        'atr_min': 0.01,
        'atr_max': 0.05,
        'adx_min': 15.0,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'profit': 0.0,
        'win_rate': 0.0,
        'trades_count': 0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'calmar_ratio': 0.0,
        'profit_factor': 0.0,
        'avg_profit_per_trade': 0.0,
        'quality_grade': 'A',
        'market_condition': 'neutral',
        'score': 0.5,
        'complexity_score': 0.6,
        'is_active': 1
    }
    
    try:
        logger.info(f"ℹ️ [Check 4] Attempting to write batch to {db_path}...")
        # manager.py와 동일하게 호출
        count = write_batch([expanded], 'strategies', db_path=db_path)
        logger.info(f"✅ [Check 4] write_batch returned: {count}")
    except Exception as e:
        logger.error(f"❌ [Check 4] write_batch failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 5. 저장된 데이터 확인
    try:
        with get_optimized_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM strategies WHERE id='test_strategy_001'")
            result = cursor.fetchone()
            if result and result[0] == 1:
                logger.info(f"✅ [Check 5] Data verification successful. Record found.")
            else:
                logger.error(f"❌ [Check 5] Data verification failed. Record NOT found. Count: {result}")
    except Exception as e:
        logger.error(f"❌ [Check 5] Verification query failed: {e}")

if __name__ == "__main__":
    # 환경변수 설정 (run_learning.py 흉내)
    # 현재 작업 디렉토리 기반으로 경로 설정
    current_dir = os.getcwd()
    data_storage_path = os.path.join(current_dir, 'market', 'coin_market', 'data_storage')
    strategies_dir = os.path.join(data_storage_path, 'learning_strategies')
    
    os.environ['DATA_STORAGE_PATH'] = data_storage_path
    os.environ['STRATEGY_DB_PATH'] = strategies_dir # 디렉토리로 설정
    os.environ['STRATEGIES_DB_PATH'] = strategies_dir
    
    # 디렉토리 생성
    if not os.path.exists(strategies_dir):
        os.makedirs(strategies_dir)
        
    logger.info(f"ℹ️ Environment setup:")
    logger.info(f"  DATA_STORAGE_PATH: {data_storage_path}")
    logger.info(f"  STRATEGY_DB_PATH: {strategies_dir}")
    
    verify_db_path_and_write()

