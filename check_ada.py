#!/usr/bin/env python
import sys
sys.path.append('/workspace')
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

cursor.execute("SELECT interval, COUNT(*) FROM rl_strategy_rollup WHERE coin='ADA' GROUP BY interval")
print("ADA 인터벌:", cursor.fetchall())
conn.close()
