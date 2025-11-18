"""integrated_analysis_results의 신호 분석"""
import sqlite3

conn = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor = conn.cursor()

# 신호 분포
print("\n📊 신호 분포:")
cursor.execute('SELECT signal_action, COUNT(*) as cnt FROM integrated_analysis_results GROUP BY signal_action')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}개')

# 최근 BUY 신호
print("\n📈 최근 BUY 신호 10개:")
cursor.execute('''
    SELECT coin, interval, signal_action, final_signal_score, created_at
    FROM integrated_analysis_results
    WHERE signal_action = "BUY"
    ORDER BY created_at DESC
    LIMIT 10
''')
for row in cursor.fetchall():
    print(f'  {row[0]}-{row[1]}: {row[2]} (점수: {row[3]:.3f}, 시각: {row[4]})')

# SELL 신호 확인
print("\n📉 SELL 신호:")
cursor.execute('SELECT COUNT(*) FROM integrated_analysis_results WHERE signal_action = "SELL"')
sell_count = cursor.fetchone()[0]
print(f'  총 {sell_count}개')

if sell_count > 0:
    cursor.execute('''
        SELECT coin, interval, signal_action, final_signal_score, created_at
        FROM integrated_analysis_results
        WHERE signal_action = "SELL"
        ORDER BY created_at DESC
        LIMIT 10
    ''')
    for row in cursor.fetchall():
        print(f'  {row[0]}-{row[1]}: {row[2]} (점수: {row[3]:.3f}, 시각: {row[4]})')
else:
    print('  ⚠️ SELL 신호가 하나도 없습니다!')

conn.close()
