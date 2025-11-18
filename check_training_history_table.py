#!/usr/bin/env python
"""Check if strategy_training_history table exists"""
import sqlite3

try:
    conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_training_history'")
    result = cursor.fetchone()

    if result:
        print("✅ strategy_training_history table exists")

        # Get table schema
        cursor.execute("PRAGMA table_info(strategy_training_history)")
        columns = cursor.fetchall()
        print("\nTable schema:")
        for col in columns:
            print(f"  {col[1]} {col[2]}")
    else:
        print("❌ strategy_training_history table NOT found")
        print("Creating table now...")

        from rl_pipeline.db.schema import setup_database_tables
        if setup_database_tables():
            print("✅ Database tables created successfully")
        else:
            print("❌ Failed to create database tables")

    conn.close()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
