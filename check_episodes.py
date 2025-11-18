#!/usr/bin/env python
import sqlite3
from datetime import datetime

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 테이블 목록
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cursor.fetchall()]
print("📋 테이블 목록:", ', '.join(tables))
print()

# 각 테이블에서 최근 에피소드 확인 (pred_ 프리픽스)
for table in tables:
    if 'episode' in table.lower() or 'summary' in table.lower():
        print(f"🔍 {table} 테이블:")
        try:
            # pred_로 시작하는 에피소드 개수
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE id LIKE 'pred_%'")
            count = cursor.fetchone()[0]
            print(f"  pred_ 에피소드 개수: {count}")

            if count > 0:
                # 최근 5개 확인
                cursor.execute(f"""
                    SELECT id, ts_entry, entry_price, first_event
                    FROM {table}
                    WHERE id LIKE 'pred_%'
                    ORDER BY ts_entry DESC
                    LIMIT 5
                """)
                episodes = cursor.fetchall()
                print(f"  최근 5개 에피소드:")
                for ep_id, ts_entry, entry_price, first_event in episodes:
                    if ts_entry and ts_entry > 0:
                        entry_time = datetime.fromtimestamp(ts_entry).strftime('%Y-%m-%d %H:%M:%S')
                        days_ago = (datetime.now().timestamp() - ts_entry) / 86400
                        print(f"    {ep_id[:50]}...")
                        print(f"      진입: {entry_time} ({days_ago:.1f}일 전)")
                        print(f"      가격: {entry_price:,.0f}, 이벤트: {first_event}")
                    else:
                        print(f"    {ep_id[:50]}: ts_entry={ts_entry} (오류!)")
        except Exception as e:
            print(f"  ⚠️ 오류: {e}")
        print()

conn.close()
