"""
240m 인터벌만 간단 테스트
"""
import sys
sys.path.append('/workspace')

from rl_pipeline.absolute_zero_system import main

# absolute_zero_system의 main을 호출하되,
# BTC 240m만 처리하도록 코인 리스트를 수정
import os
os.environ['TEST_SINGLE_COIN'] = 'BTC'
os.environ['TEST_SINGLE_INTERVAL'] = '240m'

print("🚀 BTC 240m 테스트 시작...")
print("강제 청산 로그를 확인하세요...")

# Note: absolute_zero_system.py를 수정하지 않고는 단일 코인/인터벌만 실행하기 어렵습니다.
# 대신 전체 시스템을 실행하되 로그를 모니터링합니다.
print("\n⚠️ 전체 시스템 실행 중... 240m 로그만 필터링하려면:")
print("docker exec auto_trader_coin bash -c \"tail -f /workspace/rl_pipeline/az_test_v2.log | grep '240m\\|청산'\"")
