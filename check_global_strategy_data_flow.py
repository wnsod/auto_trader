#!/usr/bin/env python
"""글로벌 전략 생성 데이터 흐름 확인"""
import sys
sys.path.append('/workspace')

import sqlite3

print("=" * 70)
print("글로벌 전략 생성 데이터 흐름 분석")
print("=" * 70)
print()

# 1. rl_strategies.db 테이블 확인
print("1️⃣  rl_strategies.db 테이블:")
print("-" * 70)
conn1 = sqlite3.connect('/workspace/data_storage/rl_strategies.db')
cursor1 = conn1.cursor()

cursor1.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables1 = [r[0] for r in cursor1.fetchall()]

for table in tables1:
    cursor1.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor1.fetchone()[0]
    if count > 0:
        print(f"   ✅ {table:<40} {count:>8}개")
    else:
        print(f"   ⚠️  {table:<40} {count:>8}개")

conn1.close()
print()

# 2. learning_results.db 테이블 확인
print("2️⃣  learning_results.db 테이블:")
print("-" * 70)
conn2 = sqlite3.connect('/workspace/data_storage/learning_results.db')
cursor2 = conn2.cursor()

cursor2.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables2 = [r[0] for r in cursor2.fetchall()]

for table in tables2:
    cursor2.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor2.fetchone()[0]
    if count > 0:
        print(f"   ✅ {table:<40} {count:>8}개")
    else:
        print(f"   ⚠️  {table:<40} {count:>8}개")

conn2.close()
print()

# 3. 글로벌 전략 생성 프로세스
print("=" * 70)
print("3️⃣  글로벌 전략 생성 프로세스 (코드 분석)")
print("=" * 70)
print()

print("📍 위치: absolute_zero_system.py:1007-1103")
print()
print("📊 입력 데이터:")
print("   - DB: rl_strategies.db")
print("   - 테이블: coin_strategies")
print("   - 내용: Self-play 후 롤업된 전략 (profit, win_rate, quality_grade 포함)")
print()
print("🔧 처리:")
print("   1. 모든 코인의 coin_strategies 로드")
print("   2. Zone별 그룹화 (regime × RSI × market × volatility)")
print("   3. 각 Zone에서 최고 전략 선택")
print("   4. global_strategies 테이블에 저장")
print()
print("📍 출력 데이터:")
print("   - DB: rl_strategies.db")
print("   - 테이블: global_strategies")
print()

# 4. 통합분석 vs 글로벌 전략
print("=" * 70)
print("4️⃣  통합분석 vs 글로벌 전략 차이")
print("=" * 70)
print()

print("┌─────────────────────────────────────────────────────────────────┐")
print("│ 통합분석 (Integrated Analysis)                                  │")
print("├─────────────────────────────────────────────────────────────────┤")
print("│ • 위치: learning_results.db → integrated_analysis_results       │")
print("│ • 입력: 개별 코인의 4개 인터벌 롤업 데이터                      │")
print("│ • 출력: BUY/SELL/HOLD 시그널                                    │")
print("│ • 용도: Paper Trading 시그널 생성                               │")
print("│ • 범위: 개별 코인별 (BTC, ETH, SOL 각각)                        │")
print("└─────────────────────────────────────────────────────────────────┘")
print()

print("┌─────────────────────────────────────────────────────────────────┐")
print("│ 글로벌 전략 (Global Strategies)                                 │")
print("├─────────────────────────────────────────────────────────────────┤")
print("│ • 위치: rl_strategies.db → global_strategies                    │")
print("│ • 입력: 모든 코인의 coin_strategies (전략 파라미터)             │")
print("│ • 출력: Zone별 최고 전략 (180개 Zone)                           │")
print("│ • 용도: 다중 코인 포트폴리오 전략                               │")
print("│ • 범위: 모든 코인 통합 (BTC+ETH+SOL+...)                        │")
print("└─────────────────────────────────────────────────────────────────┘")
print()

# 5. 데이터 흐름 요약
print("=" * 70)
print("5️⃣  전체 데이터 흐름")
print("=" * 70)
print()

print("개별 코인 (예: BTC):")
print("  1. 전략 생성 → coin_strategies (rl_strategies.db)")
print("  2. Self-play → 성과 업데이트 (profit, win_rate)")
print("  3. 롤업/등급 → rl_strategy_rollup (rl_strategies.db)")
print("  4. 통합분석 → integrated_analysis_results (learning_results.db)")
print("     └─> BUY/SELL/HOLD 시그널")
print("  5. Paper Trading → paper_trading_sessions (rl_strategies.db)")
print("     └─> 통합분석 시그널 사용")
print()

print("모든 코인 완료 후:")
print("  6. 글로벌 전략 생성:")
print("     - 입력: coin_strategies (모든 코인)")
print("     - 출력: global_strategies (rl_strategies.db)")
print("     - ❌ 통합분석 결과 사용 안 함!")
print()

print("=" * 70)
print("✅ 결론:")
print("=" * 70)
print("글로벌 전략은 coin_strategies (Self-play 후 전략)만 사용")
print("통합분석 결과는 사용하지 않음!")
print()
