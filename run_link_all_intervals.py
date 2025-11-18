#!/usr/bin/env python
"""
LINK 코인 전체 인터벌 처리
15m, 30m, 240m, 1d → 통합 분석까지
"""
import sys
import os

# 환경 변수 설정
os.environ['AZ_COINS'] = 'LINK'
os.environ['AZ_INTERVALS'] = '15m,30m,240m,1d'
os.environ['AZ_MODE'] = 'train'
os.environ['AZ_CANDLE_DAYS'] = '60'

# 모든 단계 실행
os.environ['AZ_SKIP_CANDLE_FETCH'] = 'false'
os.environ['AZ_SKIP_STRATEGY_CREATION'] = 'false'
os.environ['AZ_SKIP_PREDICTIVE_RL'] = 'false'
os.environ['AZ_SKIP_ROLLUP'] = 'false'
os.environ['AZ_SKIP_GRADING'] = 'false'

sys.path.append('/workspace')

from rl_pipeline.absolute_zero_system import main

print("=" * 70)
print("LINK 전체 인터벌 처리 + 통합 분석")
print("=" * 70)
print()
print("인터벌: 15m, 30m, 240m, 1d")
print("단계: 전략 생성 → 예측 self-play → 롤업 → 등급 → 통합 분석")
print()

main()
