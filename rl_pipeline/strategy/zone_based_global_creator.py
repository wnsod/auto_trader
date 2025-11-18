"""
구역 기반 글로벌 전략 생성 모듈

모든 구역(Zone)을 커버하는 글로벌 전략 생성:
- 4차원 구역 정의: regime × RSI × market_condition × volatility
- 각 구역에서 최고 성능 전략 선정
- 총 180개 구역 (3 × 5 × 3 × 4)
- 변동성은 기존 coin_volatility.py의 4그룹 시스템 활용 (LOW/MEDIUM/HIGH/VERY_HIGH)
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


def classify_rsi_zone(rsi_min: float, rsi_max: float) -> str:
    """
    RSI 범위를 5개 구역으로 분류

    Args:
        rsi_min: RSI 최소값
        rsi_max: RSI 최대값

    Returns:
        구역 이름 (oversold/low/neutral/high/overbought)
    """
    # RSI 중간값 기준으로 분류
    rsi_mid = (rsi_min + rsi_max) / 2

    if rsi_mid <= 30:
        return 'oversold'  # 0-30: 과매도
    elif rsi_mid <= 45:
        return 'low'       # 30-45: 낮음
    elif rsi_mid <= 55:
        return 'neutral'   # 45-55: 중립
    elif rsi_mid <= 70:
        return 'high'      # 55-70: 높음
    else:
        return 'overbought'  # 70-100: 과매수


def classify_regime(strategy: Dict[str, Any]) -> str:
    """
    전략의 레짐 분류

    Args:
        strategy: 전략 dict

    Returns:
        레짐 (ranging/trending/volatile)
    """
    # params에서 regime 추출
    params = strategy.get('params', {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except:
            params = {}

    # regime이 직접 명시된 경우
    regime = params.get('regime') or strategy.get('regime')
    if regime:
        return regime

    # regime이 없으면 strategy_type에서 추론
    strategy_type = strategy.get('strategy_type', '')

    if 'trend' in strategy_type.lower():
        return 'trending'
    elif 'volatile' in strategy_type.lower() or 'breakout' in strategy_type.lower():
        return 'volatile'
    else:
        return 'ranging'


def classify_market_condition(strategy: Dict[str, Any]) -> str:
    """
    시장 상황 분류

    Args:
        strategy: 전략 dict

    Returns:
        시장 상황 (bearish/neutral/bullish)
    """
    market_condition = strategy.get('market_condition')
    if market_condition:
        return market_condition

    # params에서 추출
    params = strategy.get('params', {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except:
            params = {}

    market_condition = params.get('market_condition')
    if market_condition:
        return market_condition

    # 기본값
    return 'neutral'


def classify_volatility_level(strategy: Dict[str, Any]) -> str:
    """
    변동성 수준 분류 (기존 coin_volatility.py 시스템 활용)

    Args:
        strategy: 전략 dict

    Returns:
        변동성 그룹 (LOW/MEDIUM/HIGH/VERY_HIGH)
    """
    # params에서 ATR 추출
    params = strategy.get('params', {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except:
            params = {}

    atr_min = params.get('atr_min', 0)
    atr_max = params.get('atr_max', 0)
    atr_mid = (atr_min + atr_max) / 2 if (atr_min or atr_max) else 0

    # 기존 coin_volatility.py의 4그룹 시스템 활용
    # LOW: 0.0 ~ 0.005 (메이저 코인: BTC 등)
    # MEDIUM: 0.005 ~ 0.007 (메이저 알트: ETH, BNB 등)
    # HIGH: 0.007 ~ 0.009 (알트코인: ADA, SOL, AVAX 등)
    # VERY_HIGH: 0.009 ~ 1.0 (고변동성: DOGE, SHIB 등)

    if atr_mid < 0.005:
        return 'LOW'
    elif atr_mid < 0.007:
        return 'MEDIUM'
    elif atr_mid < 0.009:
        return 'HIGH'
    else:
        return 'VERY_HIGH'


def get_zone_key(strategy: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    전략의 구역 키 생성

    Args:
        strategy: 전략 dict

    Returns:
        (regime, rsi_zone, market_condition, volatility_level)
    """
    # params 추출
    params = strategy.get('params', {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except:
            params = {}

    # RSI 범위
    rsi_min = params.get('rsi_min', 30)
    rsi_max = params.get('rsi_max', 70)
    rsi_zone = classify_rsi_zone(rsi_min, rsi_max)

    # 레짐
    regime = classify_regime(strategy)

    # 시장 상황
    market_condition = classify_market_condition(strategy)

    # 변동성
    volatility_level = classify_volatility_level(strategy)

    return (regime, rsi_zone, market_condition, volatility_level)


def calculate_strategy_score(strategy: Dict[str, Any]) -> float:
    """
    전략의 종합 점수 계산

    Args:
        strategy: 전략 dict

    Returns:
        종합 점수 (0.0 ~ 1.0)
    """
    # 성과 지표 추출
    profit = strategy.get('profit', 0) or 0
    win_rate = strategy.get('win_rate', 0) or 0
    sharpe_ratio = strategy.get('sharpe_ratio', 0) or 0
    max_drawdown = abs(strategy.get('max_drawdown', 0) or 0)

    # 종합 점수 계산 (가중 평균)
    score = (
        profit * 0.4 +           # 수익률 40%
        win_rate * 0.3 +         # 승률 30%
        sharpe_ratio * 0.2 +     # 샤프 비율 20%
        (1 - max_drawdown) * 0.1 # 손실 제한 10%
    )

    # 0~1 범위로 정규화
    return max(0.0, min(1.0, score))


def group_strategies_by_zone(
    all_coin_strategies: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    """
    모든 코인 전략을 구역별로 그룹화

    Args:
        all_coin_strategies: {coin: {interval: [strategies]}}

    Returns:
        {zone_key: [strategies]}
    """
    zones = defaultdict(list)

    total_strategies = 0

    for coin, interval_strategies in all_coin_strategies.items():
        for interval, strategies in interval_strategies.items():
            for strategy in strategies:
                try:
                    # 구역 키 생성
                    zone_key = get_zone_key(strategy)

                    # 전략에 메타데이터 추가
                    strategy['_source_coin'] = coin
                    strategy['_source_interval'] = interval
                    strategy['_zone_key'] = '-'.join(zone_key)

                    # 구역에 추가
                    zones[zone_key].append(strategy)
                    total_strategies += 1

                except Exception as e:
                    logger.debug(f"전략 분류 실패 ({coin}): {e}")
                    continue

    logger.info(f"📊 전략 구역 분류 완료: {total_strategies}개 전략 → {len(zones)}개 구역")

    return zones


def select_best_strategy_per_zone(
    zones: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]
) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    """
    각 구역에서 최고 성능 전략 선정

    Args:
        zones: {zone_key: [strategies]}

    Returns:
        {zone_key: best_strategy}
    """
    best_strategies = {}

    for zone_key, strategies in zones.items():
        if not strategies:
            continue

        # 각 전략의 점수 계산
        scored_strategies = [
            (calculate_strategy_score(s), s) for s in strategies
        ]

        # 점수 기준 정렬
        scored_strategies.sort(reverse=True, key=lambda x: x[0])

        # 최고 전략 선정
        best_score, best_strategy = scored_strategies[0]

        best_strategies[zone_key] = best_strategy

        logger.debug(
            f"구역 {'-'.join(zone_key)}: "
            f"{len(strategies)}개 중 최고 선정 "
            f"(점수: {best_score:.3f}, 출처: {best_strategy.get('_source_coin')})"
        )

    logger.info(f"✅ 구역별 최고 전략 선정 완료: {len(best_strategies)}개 구역")

    return best_strategies


def create_global_strategy_from_best(
    zone_key: Tuple[str, str, str, str],
    best_strategy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    최고 전략을 기반으로 글로벌 전략 생성

    Args:
        zone_key: (regime, rsi_zone, market_condition, volatility_level)
        best_strategy: 최고 성능 전략

    Returns:
        글로벌 전략 dict
    """
    regime, rsi_zone, market_condition, volatility_level = zone_key
    zone_str = '-'.join(zone_key)

    # 글로벌 전략 ID 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    global_id = f"GLOBAL_ZONE_{zone_str}_{timestamp}"

    # 원본 전략 복사
    global_strategy = {
        'id': global_id,
        'coin': 'GLOBAL',
        'interval': best_strategy.get('_source_interval', '240m'),
        'strategy_type': f'zone_based_{regime}',
        'params': {
            # 기존 params에 파라미터 값 추가 (coin_strategies는 컬럼에 저장됨)
            **(best_strategy.get('params', {})),
            'rsi_min': best_strategy.get('rsi_min'),
            'rsi_max': best_strategy.get('rsi_max'),
            'volume_ratio_min': best_strategy.get('volume_ratio_min'),
            'volume_ratio_max': best_strategy.get('volume_ratio_max'),
            'macd_buy_threshold': best_strategy.get('macd_buy_threshold'),
            'macd_sell_threshold': best_strategy.get('macd_sell_threshold'),
            'mfi_min': best_strategy.get('mfi_min'),
            'mfi_max': best_strategy.get('mfi_max'),
            'atr_min': best_strategy.get('atr_min'),
            'atr_max': best_strategy.get('atr_max'),
            'adx_min': best_strategy.get('adx_min'),
            'stop_loss_pct': best_strategy.get('stop_loss_pct'),
            'take_profit_pct': best_strategy.get('take_profit_pct'),
        },
        'name': f'Global Zone Strategy ({zone_str})',
        'description': (
            f'구역 기반 글로벌 전략: {zone_str} | '
            f'출처: {best_strategy.get("_source_coin")} | '
            f'성과: profit={best_strategy.get("profit", 0):.2%}, '
            f'win_rate={best_strategy.get("win_rate", 0):.2%}'
        ),
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),

        # Note: rsi_min 등 파라미터 필드는 global_strategies 테이블 컬럼에 없음
        # 대신 params JSON 필드에 포함됨 (위에서 추가됨)

        # 성과 지표 복사
        'profit': best_strategy.get('profit', 0),
        'win_rate': best_strategy.get('win_rate', 0),
        'sharpe_ratio': best_strategy.get('sharpe_ratio', 0),
        'max_drawdown': best_strategy.get('max_drawdown', 0),
        'profit_factor': best_strategy.get('profit_factor', 0),
        'trades_count': best_strategy.get('trades_count', 0),
        'quality_grade': best_strategy.get('quality_grade', 'A'),

        # 구역 메타데이터
        'zone_key': zone_str,
        'regime': regime,
        'rsi_zone': rsi_zone,
        'market_condition': market_condition,
        'volatility_level': volatility_level,

        # 출처 정보
        'source_coin': best_strategy.get('_source_coin'),
        'source_strategy_id': best_strategy.get('id'),
        'source_type': 'zone_based',

        # 증분 학습 메타데이터 (일단 novel로 설정)
        'similarity_classification': 'novel',
        'similarity_score': 0.0,
        'parent_strategy_id': None
    }

    return global_strategy


def create_zone_based_global_strategies(
    all_coin_strategies: Dict[str, Dict[str, List[Dict[str, Any]]]],
    enable_similarity_check: bool = True
) -> List[Dict[str, Any]]:
    """
    구역 기반 글로벌 전략 생성 (메인 함수)

    Args:
        all_coin_strategies: {coin: {interval: [strategies]}}
        enable_similarity_check: 유사도 검사 활성화 여부

    Returns:
        글로벌 전략 리스트
    """
    logger.info("🌍 구역 기반 글로벌 전략 생성 시작")

    try:
        # 1. 전략을 구역별로 그룹화
        zones = group_strategies_by_zone(all_coin_strategies)

        if not zones:
            logger.warning("⚠️ 분류된 구역 없음")
            return []

        # 구역 분포 로깅
        zone_distribution = defaultdict(int)
        for zone_key in zones.keys():
            regime, rsi_zone, market_condition, volatility = zone_key
            zone_distribution[regime] += 1

        logger.info(f"📊 구역 분포:")
        for regime, count in sorted(zone_distribution.items()):
            logger.info(f"  - {regime}: {count}개 구역")

        # 2. 각 구역에서 최고 전략 선정
        best_strategies = select_best_strategy_per_zone(zones)

        if not best_strategies:
            logger.warning("⚠️ 선정된 최고 전략 없음")
            return []

        # 3. 기존 글로벌 전략 로드 (유사도 검사용)
        existing_global_strategies = []
        if enable_similarity_check:
            try:
                from rl_pipeline.db.connection_pool import get_optimized_db_connection

                with get_optimized_db_connection("strategies") as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT * FROM global_strategies
                        WHERE zone_key IS NOT NULL
                    """)

                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]

                    for row in rows:
                        strategy = dict(zip(columns, row))

                        # params JSON 파싱
                        if 'params' in strategy and isinstance(strategy['params'], str):
                            try:
                                strategy['params'] = json.loads(strategy['params'])
                            except:
                                pass

                        existing_global_strategies.append(strategy)

                logger.info(f"📊 기존 글로벌 전략 로드: {len(existing_global_strategies)}개 (유사도 검사용)")

            except Exception as e:
                logger.warning(f"⚠️ 기존 글로벌 전략 로드 실패: {e}")
                enable_similarity_check = False

        # 4. 글로벌 전략 생성 (유사도 검사 포함)
        global_strategies = []

        for zone_key, best_strategy in best_strategies.items():
            try:
                global_strategy = create_global_strategy_from_best(zone_key, best_strategy)

                # 유사도 검사
                if enable_similarity_check and existing_global_strategies:
                    from rl_pipeline.strategy.similarity import classify_strategy_by_similarity

                    classification, similarity_score, parent_id = classify_strategy_by_similarity(
                        global_strategy,
                        existing_global_strategies,
                        use_smart=False  # 글로벌 전략은 simple similarity 사용
                    )

                    # 유사도 정보 업데이트
                    global_strategy['similarity_classification'] = classification
                    global_strategy['similarity_score'] = similarity_score
                    global_strategy['parent_strategy_id'] = parent_id

                    logger.debug(
                        f"  유사도 검사: {zone_key} → {classification} "
                        f"(score: {similarity_score:.3f})"
                    )

                    # duplicate는 건너뜀 (중복 방지)
                    if classification == 'duplicate':
                        logger.info(f"  ⚠️ 중복 전략 건너뜀: {'-'.join(zone_key)}")
                        continue

                global_strategies.append(global_strategy)

            except Exception as e:
                logger.error(f"❌ 글로벌 전략 생성 실패 ({'-'.join(zone_key)}): {e}")
                continue

        logger.info(f"✅ 구역 기반 글로벌 전략 생성 완료: {len(global_strategies)}개")

        # 통계 출력
        regime_counts = defaultdict(int)
        rsi_counts = defaultdict(int)
        similarity_counts = defaultdict(int)

        for strategy in global_strategies:
            regime_counts[strategy['regime']] += 1
            rsi_counts[strategy['rsi_zone']] += 1
            classification = strategy.get('similarity_classification', 'novel')
            similarity_counts[classification] += 1

        logger.info(f"📈 생성된 글로벌 전략 분포:")
        logger.info(f"  레짐별: {dict(regime_counts)}")
        logger.info(f"  RSI별: {dict(rsi_counts)}")

        if enable_similarity_check:
            logger.info(f"  유사도별: {dict(similarity_counts)}")

        return global_strategies

    except Exception as e:
        logger.error(f"❌ 구역 기반 글로벌 전략 생성 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


def save_global_strategies_to_db(
    global_strategies: List[Dict[str, Any]]
) -> int:
    """
    글로벌 전략을 데이터베이스에 저장

    Args:
        global_strategies: 글로벌 전략 리스트

    Returns:
        저장된 개수
    """
    if not global_strategies:
        return 0

    try:
        from rl_pipeline.db.writes import write_batch
        from rl_pipeline.core.env import config

        # params를 JSON 문자열로 변환
        for strategy in global_strategies:
            params = strategy.get('params', {})
            if isinstance(params, dict):
                strategy['params'] = json.dumps(params)

        # DB에 저장
        saved_count = write_batch(
            global_strategies,
            'global_strategies',
            db_path=config.STRATEGIES_DB
        )

        logger.info(f"✅ 글로벌 전략 DB 저장 완료: {saved_count}개")
        return saved_count

    except Exception as e:
        logger.error(f"❌ 글로벌 전략 DB 저장 실패: {e}")
        return 0


def get_global_strategy_for_situation(
    regime: str,
    rsi_zone: str,
    market_condition: str,
    volatility_level: str,
    interval: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    특정 상황(구역)에 맞는 글로벌 전략 조회

    Args:
        regime: 레짐 (ranging/trending/volatile)
        rsi_zone: RSI 구역 (oversold/low/neutral/high/overbought)
        market_condition: 시장 상황 (bearish/neutral/bullish)
        volatility_level: 변동성 수준 (LOW/MEDIUM/HIGH/VERY_HIGH)
        interval: 인터벌 (선택사항, None이면 모든 인터벌)

    Returns:
        글로벌 전략 dict 또는 None
    """
    try:
        from rl_pipeline.db.connection_pool import get_optimized_db_connection

        # 구역 키 생성
        zone_key = f"{regime}-{rsi_zone}-{market_condition}-{volatility_level}"

        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()

            # 해당 구역 전략 조회
            if interval:
                query = """
                    SELECT * FROM global_strategies
                    WHERE zone_key = ? AND interval = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cursor.execute(query, (zone_key, interval))
            else:
                query = """
                    SELECT * FROM global_strategies
                    WHERE zone_key = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cursor.execute(query, (zone_key,))

            row = cursor.fetchone()

            if row:
                # 컬럼명 가져오기
                columns = [desc[0] for desc in cursor.description]
                strategy = dict(zip(columns, row))

                # params JSON 파싱
                if 'params' in strategy and isinstance(strategy['params'], str):
                    try:
                        strategy['params'] = json.loads(strategy['params'])
                    except:
                        pass

                logger.debug(f"✅ 글로벌 전략 조회 성공: {zone_key}")
                return strategy

            # 해당 구역에 전략이 없으면 None 반환
            logger.debug(f"⚠️ 글로벌 전략 없음: {zone_key}")
            return None

    except Exception as e:
        logger.error(f"❌ 글로벌 전략 조회 실패: {e}")
        return None


def get_global_strategy_by_zone_with_fallback(
    regime: str,
    rsi_zone: str,
    market_condition: str,
    volatility_level: str,
    interval: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    구역 기반 글로벌 전략 조회 (fallback 포함)

    특정 구역에 전략이 없으면 인접 구역에서 찾음

    Args:
        regime: 레짐
        rsi_zone: RSI 구역
        market_condition: 시장 상황
        volatility_level: 변동성 수준
        interval: 인터벌

    Returns:
        글로벌 전략 dict 또는 None
    """
    # 1차 시도: 정확한 구역 매칭
    strategy = get_global_strategy_for_situation(
        regime, rsi_zone, market_condition, volatility_level, interval
    )

    if strategy:
        return strategy

    # 2차 시도: RSI neutral로 fallback
    if rsi_zone != 'neutral':
        logger.debug(f"🔄 Fallback: RSI {rsi_zone} → neutral")
        strategy = get_global_strategy_for_situation(
            regime, 'neutral', market_condition, volatility_level, interval
        )
        if strategy:
            return strategy

    # 3차 시도: market_condition neutral로 fallback
    if market_condition != 'neutral':
        logger.debug(f"🔄 Fallback: market {market_condition} → neutral")
        strategy = get_global_strategy_for_situation(
            regime, rsi_zone, 'neutral', volatility_level, interval
        )
        if strategy:
            return strategy

    # 4차 시도: 같은 변동성 그룹, 모든 RSI/market neutral
    logger.debug(f"🔄 Fallback: All neutral")
    strategy = get_global_strategy_for_situation(
        regime, 'neutral', 'neutral', volatility_level, interval
    )
    if strategy:
        return strategy

    # 모두 실패하면 None
    logger.warning(f"⚠️ 글로벌 전략 fallback 모두 실패: {regime}-{volatility_level}")
    return None
