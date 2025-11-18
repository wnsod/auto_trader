"""Check rl_strategies.db schema"""
import sqlite3

db_path = '/workspace/data_storage/rl_strategies.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print("📊 Tables in rl_strategies.db:")
for table in sorted(tables):
    print(f"  - {table}")

print("\n" + "="*80 + "\n")

# Check integrated_analysis_results
print("🔍 integrated_analysis_results schema:")
cursor.execute("PRAGMA table_info(integrated_analysis_results)")
columns = cursor.fetchall()
if columns:
    for col in columns:
        print(f"  {col[1]:30s} {col[2]:15s} {'NOT NULL' if col[3] else ''}")
else:
    print("  ❌ Table does not exist")

print("\n" + "="*80 + "\n")

# Check global_strategies
print("🔍 global_strategies schema:")
cursor.execute("PRAGMA table_info(global_strategies)")
columns = cursor.fetchall()
if columns:
    for col in columns:
        print(f"  {col[1]:30s} {col[2]:15s} {'NOT NULL' if col[3] else ''}")
else:
    print("  ❌ Table does not exist")

conn.close()
