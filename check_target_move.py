#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT target_move_pct, entry_price
    FROM rl_episodes
    WHERE episode_id LIKE 'pred_%'
    LIMIT 5
""")

rows = cursor.fetchall()
print("Raw target_move_pct values:")
for target_pct, entry_price in rows:
    print(f"  target_move_pct = {target_pct}")
    print(f"  entry_price = {entry_price}")
    if target_pct > 0:
        tp_amount = entry_price * target_pct / 100
        print(f"  If {target_pct}% of {entry_price} = {tp_amount:.2f} KRW")
        print(f"  TP price would be: {entry_price + tp_amount:.2f} (for buy)")
    print()

conn.close()
