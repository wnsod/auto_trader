#!/usr/bin/env python
"""Verify similarity metadata is saved in database"""
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# Check recent strategies
cursor.execute("""
    SELECT id, similarity_classification, similarity_score, parent_strategy_id
    FROM coin_strategies
    WHERE coin='ADA' AND interval='15m'
    ORDER BY created_at DESC
    LIMIT 5
""")

print("Most recent 5 ADA-15m strategies:")
print("-" * 80)
for row in cursor.fetchall():
    print(f"ID: {row[0][:40]}")
    print(f"  Classification: {row[1]}")
    print(f"  Score: {row[2]}")
    print(f"  Parent: {row[3]}")
    print()

# Check counts by classification
cursor.execute("""
    SELECT similarity_classification, COUNT(*)
    FROM coin_strategies
    WHERE coin='ADA' AND interval='15m'
    GROUP BY similarity_classification
""")

print("\nClassification counts:")
print("-" * 80)
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
print("\n✅ Database verification complete!")
