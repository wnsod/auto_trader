import sys
sys.path.insert(0, /workspace)
from rl_pipeline.db.connection_pool import get_optimized_db_connection

with get_optimized_db_connection(strategies) as conn:
    cursor = conn.cursor()
    cursor.execute("
        SELECT id, similarity_classification, similarity_score, parent_strategy_id
        FROM coin_strategies
        WHERE coin=ADA AND interval=15m
        ORDER BY created_at DESC
        LIMIT 10
    ")
    rows = cursor.fetchall()
    print(Most
