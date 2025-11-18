#!/usr/bin/env python
"""
absolute_zero_system을 LINK 1개 코인으로 실행
"""
import sys
import os

# 환경 변수 설정
os.environ['AZ_COINS'] = 'LINK'
os.environ['AZ_INTERVALS'] = '15m'
os.environ['AZ_MODE'] = 'train'
os.environ['AZ_CANDLE_DAYS'] = '60'

# 전략 생성만 수행
os.environ['AZ_SKIP_CANDLE_FETCH'] = 'false'
os.environ['AZ_SKIP_STRATEGY_CREATION'] = 'false'
os.environ['AZ_SKIP_PREDICTIVE_RL'] = 'true'  # 일단 스킵
os.environ['AZ_SKIP_ROLLUP'] = 'true'  # 일단 스킵
os.environ['AZ_SKIP_GRADING'] = 'true'  # 일단 스킵

sys.path.append('/workspace')

from rl_pipeline.absolute_zero_system import main

print("=" * 70)
print("Absolute Zero System: LINK-15m 전략 생성")
print("=" * 70)
print()

main()
