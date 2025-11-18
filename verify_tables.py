#!/usr/bin/env python
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# coin_strategies 테이블 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_strategies'")
if cursor.fetchone():
    print("✅ coin_strategies 테이블 존재")

    # LINK 전략 확인
    cursor.execute("SELECT COUNT(*) FROM coin_strategies WHERE coin='LINK'")
    link_count = cursor.fetchone()[0]
    print(f"   LINK 전략: {link_count}개")

    # 전체 전략 수
    cursor.execute("SELECT coin, COUNT(*) FROM coin_strategies GROUP BY coin")
    for coin, count in cursor.fetchall():
        print(f"   {coin}: {count}개")
else:
    print("❌ coin_strategies 테이블 없음")

conn.close()
