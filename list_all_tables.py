#!/usr/bin/env python3
"""DB 테이블 목록 및 스키마 확인"""

import sqlite3
from pathlib import Path

db_path = "/workspace/data_storage/rl_strategies.db"
if not Path(db_path).exists():
    db_path = "./data_storage/rl_strategies.db"

print(f"🔍 DB: {db_path}\n")

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # 모든 테이블 목록
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"📊 전체 테이블 개수: {len(tables)}\n")
    print("=" * 80)

    for table_name in tables:
        print(f"\n🗂️  {table_name}")

        # 테이블 스키마
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        # 데이터 개수
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        print(f"   📊 행 개수: {count}")
        print(f"   📋 컬럼 ({len(columns)}개):")

        for col in columns:
            col_id, col_name, col_type, not_null, default, pk = col
            flags = []
            if pk:
                flags.append("PK")
            if not_null:
                flags.append("NOT NULL")
            if default:
                flags.append(f"DEFAULT {default}")

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"      {col_id}. {col_name} ({col_type}){flag_str}")

    print("\n" + "=" * 80)
    print("✅ 테이블 목록 출력 완료")
