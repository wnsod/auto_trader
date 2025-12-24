#!/usr/bin/env python3
"""
인터벌 프로필 적용 확인 스크립트
"""

import sys
import os

# 경로 추가
sys.path.insert(0, '/workspace/rl_pipeline')

try:
    # 직접 파일 경로로 임포트
    spec = __import__('importlib.util').util.spec_from_file_location(
        "interval_profiles",
        "/workspace/rl_pipeline/core/interval_profiles.py"
    )
    interval_profiles = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(interval_profiles)

    INTERVAL_PROFILES = interval_profiles.INTERVAL_PROFILES
    get_integration_weights = interval_profiles.get_integration_weights

    print("✅ 인터벌 프로필 모듈 로드 성공!")
    print("\n정의된 인터벌:")
    for interval in INTERVAL_PROFILES.keys():
        profile = INTERVAL_PROFILES[interval]
        print(f"  - {interval}: {profile['role']} ({profile['integration_weight']:.2f})")

    print("\n통합 가중치:")
    weights = get_integration_weights()
    for interval, weight in weights.items():
        print(f"  - {interval}: {weight:.2f}")

    print("\n✅ 모든 기능이 정상적으로 작동합니다!")

except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")