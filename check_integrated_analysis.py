import sqlite3

db_path = '/workspace/data_storage/learning_results.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print(f"Checking database: {db_path}\n")

# Check if table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='integrated_analysis_results'")
table_exists = cursor.fetchone()
print(f"Table 'integrated_analysis_results' exists: {table_exists is not None}\n")

if table_exists:
    # Check total rows
    cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results")
    total_count = cursor.fetchone()[0]
    print(f"Total rows in integrated_analysis_results: {total_count}\n")

    # Check LINK rows
    cursor.execute("SELECT COUNT(*) FROM integrated_analysis_results WHERE coin = 'LINK'")
    link_count = cursor.fetchone()[0]
    print(f"Rows for LINK: {link_count}\n")

    # Get all coins
    cursor.execute("SELECT DISTINCT coin FROM integrated_analysis_results")
    coins = [row[0] for row in cursor.fetchall()]
    print(f"Coins in DB: {coins}\n")

    # Get LINK rows if any
    if link_count > 0:
        cursor.execute("""
            SELECT coin, interval, signal_action, final_signal_score, created_at
            FROM integrated_analysis_results
            WHERE coin = 'LINK'
            ORDER BY created_at DESC
            LIMIT 10
        """)
        print("LINK integrated analysis results:")
        for row in cursor.fetchall():
            print(f"  {row}")

    # Get latest rows for any coin
    cursor.execute("""
        SELECT coin, interval, signal_action, final_signal_score, created_at
        FROM integrated_analysis_results
        ORDER BY created_at DESC
        LIMIT 10
    """)
    print("\nLatest 10 rows (all coins):")
    for row in cursor.fetchall():
        print(f"  {row}")

conn.close()
