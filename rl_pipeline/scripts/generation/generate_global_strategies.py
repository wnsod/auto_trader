"""
글로벌 전략만 독립적으로 생성하는 스크립트
self-play가 완료된 결과를 기반으로 글로벌 전략 생성

개선 사항(리팩토링):
- 타입 힌트/리턴 코드 명확화, 함수 분리
- 코인 소싱: 인자 우선 → DB 검출 → 환경변수(DEFAULT_COIN) 폴백
- 에러 메시지/종료 코드 일관화
- 🔥 글로벌 학습 기능 추가 (--train 옵션)
"""

import sys
import os
import logging
import argparse
from typing import List, Optional

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from absolute_zero_system import generate_global_strategies_only

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='글로벌 전략만 생성 (self-play 결과 기반)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 모든 코인의 글로벌 전략 생성
  python generate_global_strategies.py
  
  # 특정 코인의 글로벌 전략만 생성
  python generate_global_strategies.py BTC ETH SOL
  
  # 디버그 모드
  python generate_global_strategies.py --debug BTC
  
  # 글로벌 전략 생성 후 학습 실행
  python generate_global_strategies.py --train
  
  # 특정 코인 + 학습
  python generate_global_strategies.py --train BTC ETH
        """
    )
    
    parser.add_argument('coins', nargs='*', default=None,
                        help='특정 코인만 필터링 (예: BTC ETH SOL)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 로그 활성화')
    parser.add_argument('--train', action='store_true',
                        help='글로벌 전략 생성 후 자동 학습 실행 (ENABLE_AUTO_TRAINING=true 필요)')
    
    return parser.parse_args(argv)


def _setup_logging(debug: bool) -> logging.Logger:
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def _resolve_coin_filter(cli_coins: Optional[List[str]]) -> Optional[List[str]]:
    """코인 필터를 결정: CLI > DB 검출 > 환경 폴백(None이면 전체 처리)"""
    if cli_coins:
        return cli_coins
    try:
        # DB에서 사용 가능한 코인 목록 검출
        from rl_pipeline.data.candle_loader import get_available_coins_and_intervals
        available = get_available_coins_and_intervals()
        coins = sorted(list({c for c, _ in available}))
        return coins or None
    except Exception:
        # 폴백: 환경변수 DEFAULT_COIN, 없으면 전체(None)
        default_coin = os.getenv('DEFAULT_COIN')
        return [default_coin] if default_coin else None


def main(argv: Optional[List[str]] = None) -> int:
    """메인 함수
    Returns: 종료 코드(0 성공, 1 실패)
    """
    args = _parse_args(argv)
    logger = _setup_logging(args.debug)

    # 코인 필터 해석: CLI > DB > 환경 폴백(None은 전체 처리 의미)
    coin_filter = _resolve_coin_filter(args.coins)
    if coin_filter:
        logger.info(f"📊 선택된 코인(해결됨): {coin_filter}")
    else:
        logger.info("📊 선택된 코인 없음 → 전체 코인 대상으로 처리")

    # 글로벌 전략 생성
    if args.train:
        logger.info("🚀 글로벌 전략 생성 및 학습 시작...")
        logger.info("   (--train 옵션 활성화, ENABLE_AUTO_TRAINING=true 필요)")
    else:
        logger.info("🚀 글로벌 전략 생성 시작...")
    
    result = generate_global_strategies_only(
        coin_filter=coin_filter,
        enable_training=args.train
    )

    if result.get("success"):
        count = result.get("count", 0)
        logger.info(f"✅ 글로벌 전략 생성 완료: {count}개")
        
        # 학습 결과 출력
        if args.train and "trained_model_id" in result:
            logger.info(f"✅ 글로벌 학습 완료: 모델 ID = {result['trained_model_id']}")
        elif args.train:
            logger.info("⚠️ 글로벌 학습은 실행되었지만 모델이 생성되지 않았습니다")
            logger.info("   (ENABLE_AUTO_TRAINING=true 및 USE_HYBRID=true 설정 확인 필요)")
        
        return 0
    else:
        reason = result.get("reason", result.get("error", "unknown"))
        logger.error(f"❌ 글로벌 전략 생성 실패: {reason}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
