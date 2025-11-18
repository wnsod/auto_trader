#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전략 데이터 확인 스크립트"""

import sqlite3
import pandas as pd
from collections import Counter

# DB 경로
rl_strategies_db = "data_storage/rl_strategies.db"

try:
    conn = sqlite3.connect(rl_strategies_db)
    
    # 1. 전체 전략 수 확인
    total_count = pd.read_sql("SELECT COUNT(*) as count FROM coin_strategies", conn)
    print(f"[전략] 전체 전략 수: {total_count.iloc[0]['count']:,}개")
    
    # 2. 조건별 전략 수 확인
    condition_count = pd.read_sql("""
        SELECT COUNT(*) as count 
        FROM coin_strategies
        WHERE trades_count >= 1 AND profit > 0
    """, conn)
    print(f"[전략] 조건 만족 전략 수 (trades_count >= 1 AND profit > 0): {condition_count.iloc[0]['count']:,}개")
    
    # 3. 코인별 전략 수 확인
    coin_counts = pd.read_sql("""
        SELECT coin, COUNT(*) as count
        FROM coin_strategies
        WHERE trades_count >= 1 AND profit > 0
        GROUP BY coin
        ORDER BY count DESC
    """, conn)
    print(f"\n[전략] 코인별 전략 수:")
    for _, row in coin_counts.iterrows():
        print(f"  {row['coin']}: {row['count']}개")
    
    # 4. 인터벌별 전략 수 확인
    interval_counts = pd.read_sql("""
        SELECT interval, COUNT(*) as count
        FROM coin_strategies
        WHERE trades_count >= 1 AND profit > 0
        GROUP BY interval
        ORDER BY count DESC
    """, conn)
    print(f"\n[전략] 인터벌별 전략 수:")
    for _, row in interval_counts.iterrows():
        print(f"  {row['interval']}: {row['count']}개")
    
    # 5. 코인+인터벌 조합별 전략 수 확인 (전략 키)
    strategy_keys = pd.read_sql("""
        SELECT coin, interval, COUNT(*) as count
        FROM coin_strategies
        WHERE trades_count >= 1 AND profit > 0
        GROUP BY coin, interval
        ORDER BY count DESC
        LIMIT 20
    """, conn)
    print(f"\n[전략] 전략 키별 전략 수 (상위 20개):")
    for _, row in strategy_keys.iterrows():
        key = f"{row['coin']}_{row['interval']}"
        print(f"  {key}: {row['count']}개")
    
    # 6. 실제 로드되는 전략 키 확인 (로드 로직과 동일한 쿼리)
    loaded_strategies = pd.read_sql("""
        SELECT coin as symbol, interval, profit, win_rate, trades_count, id as strategy_id,
               COALESCE(score, profit * win_rate, 0.5) as score
        FROM coin_strategies
        WHERE trades_count >= 1
        AND profit > 0
        ORDER BY COALESCE(score, profit * win_rate, 0.0) DESC
        LIMIT 1000
    """, conn)
    
    print(f"\n[전략] 실제 로드되는 전략 수: {len(loaded_strategies)}개")
    
    # 7. 전략 키별로 그룹화 (같은 키가 여러 개 있을 수 있음)
    strategy_key_dict = {}
    for _, row in loaded_strategies.iterrows():
        key = f"{row['symbol']}_{row['interval']}"
        if key not in strategy_key_dict:
            strategy_key_dict[key] = []
        strategy_key_dict[key].append({
            'strategy_id': row['strategy_id'],
            'profit': row['profit'],
            'win_rate': row['win_rate'],
            'score': row['score']
        })
    
    print(f"\n[전략] 고유한 전략 키 수: {len(strategy_key_dict)}개")
    print(f"\n[전략] 전략 키 목록 (처음 20개):")
    for i, (key, strategies) in enumerate(list(strategy_key_dict.items())[:20]):
        print(f"  {i+1}. {key}: {len(strategies)}개 전략 (최고 점수: {max(s['score'] for s in strategies):.4f})")
    
    # 8. SOL 전략 확인
    sol_strategies = pd.read_sql("""
        SELECT coin, interval, profit, win_rate, trades_count, score
        FROM coin_strategies
        WHERE coin = 'SOL' AND trades_count >= 1 AND profit > 0
        ORDER BY COALESCE(score, profit * win_rate, 0.0) DESC
        LIMIT 10
    """, conn)
    
    print(f"\n[SOL] SOL 전략 확인:")
    if len(sol_strategies) > 0:
        print(f"  SOL 전략 {len(sol_strategies)}개 발견:")
        for _, row in sol_strategies.iterrows():
            print(f"    {row['coin']}_{row['interval']}: profit={row['profit']:.4f}, win_rate={row['win_rate']:.2f}, trades={row['trades_count']}")
    else:
        print(f"  [경고] SOL 전략이 없습니다 (조건: trades_count >= 1 AND profit > 0)")
        
        # 조건 완화해서 확인
        sol_all = pd.read_sql("""
            SELECT coin, interval, profit, win_rate, trades_count
            FROM coin_strategies
            WHERE coin = 'SOL'
            ORDER BY id DESC
            LIMIT 10
        """, conn)
        if len(sol_all) > 0:
            print(f"  [정보] SOL 전략 전체 {len(sol_all)}개 (조건 없이):")
            for _, row in sol_all.iterrows():
                print(f"    {row['coin']}_{row['interval']}: profit={row['profit']:.4f}, win_rate={row['win_rate']:.2f}, trades={row['trades_count']}")
        else:
            print(f"  [오류] SOL 전략이 전혀 없습니다")
    
    conn.close()
    
except Exception as e:
    print(f"[오류] 오류 발생: {e}")
    import traceback
    traceback.print_exc()

