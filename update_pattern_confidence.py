"""
기존 전략들의 pattern_confidence 계산 및 업데이트
"""
import sys
sys.path.insert(0, '/workspace')

import sqlite3
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = '/workspace/data_storage/rl_strategies.db'

def calculate_pattern_confidence(params: Dict[str, Any]) -> float:
    """
    전략의 패턴 신뢰도 계산

    고려 요소:
    1. 파라미터 범위의 합리성 (좁을수록 신뢰도 높음)
    2. 전략 복잡도 (조건 수가 적절하면 신뢰도 높음)
    3. 기본값에서 벗어난 정도 (커스터마이징 정도)
    """
    confidence = 0.5  # 기본값

    try:
        # 1. RSI 범위 (좁을수록 좋음, 하지만 너무 좁으면 안됨)
        rsi_min = params.get('rsi_min', 30)
        rsi_max = params.get('rsi_max', 70)
        rsi_range = rsi_max - rsi_min

        if 20 <= rsi_range <= 40:  # 적절한 범위
            confidence += 0.15
        elif 10 <= rsi_range < 20:  # 조금 좁음
            confidence += 0.10
        elif 40 < rsi_range <= 60:  # 조금 넓음
            confidence += 0.05
        else:  # 너무 좁거나 너무 넓음
            confidence -= 0.05

        # 2. Volume ratio 범위
        vol_min = params.get('volume_ratio_min', 1.0)
        vol_max = params.get('volume_ratio_max', 2.0)
        vol_range = vol_max - vol_min

        if 0.8 <= vol_range <= 2.0:  # 적절한 범위
            confidence += 0.10
        elif 0.5 <= vol_range < 0.8 or 2.0 < vol_range <= 3.0:
            confidence += 0.05
        else:
            confidence -= 0.05

        # 3. MACD 임계값 (절댓값이 적절하면 좋음)
        macd_buy = abs(params.get('macd_buy_threshold', 0.01))
        macd_sell = abs(params.get('macd_sell_threshold', -0.01))

        if 0.005 <= macd_buy <= 0.02 and 0.005 <= macd_sell <= 0.02:
            confidence += 0.10
        elif 0.002 <= macd_buy <= 0.03 and 0.002 <= macd_sell <= 0.03:
            confidence += 0.05
        else:
            confidence -= 0.05

        # 4. Stop loss / Take profit 비율 (리스크 리워드 비율)
        stop_loss = params.get('stop_loss_pct', 0.02)
        take_profit = params.get('take_profit_pct', 0.05)

        if stop_loss > 0 and take_profit > 0:
            risk_reward = take_profit / stop_loss
            if 1.5 <= risk_reward <= 3.0:  # 적절한 리스크 리워드 비율
                confidence += 0.15
            elif 1.0 <= risk_reward < 1.5 or 3.0 < risk_reward <= 4.0:
                confidence += 0.08
            else:
                confidence -= 0.05

        # 5. 기본값과의 차이 (커스터마이징 정도)
        customization_score = 0
        if rsi_min != 30:
            customization_score += 1
        if rsi_max != 70:
            customization_score += 1
        if vol_min != 1.0:
            customization_score += 1
        if vol_max != 2.0:
            customization_score += 1

        if customization_score >= 3:  # 3개 이상 커스터마이징
            confidence += 0.10
        elif customization_score >= 2:
            confidence += 0.05

        # 0.0 ~ 1.0 범위로 클리핑
        confidence = max(0.0, min(1.0, confidence))

    except Exception as e:
        logger.warning(f"패턴 신뢰도 계산 실패: {e}")
        confidence = 0.5

    return confidence


def update_all_strategies():
    """모든 전략의 pattern_confidence 업데이트"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        logger.info("🔄 전략 pattern_confidence 업데이트 시작...")

        # 모든 전략 조회
        cursor.execute("""
            SELECT id, params
            FROM coin_strategies
        """)
        strategies = cursor.fetchall()

        logger.info(f"📊 처리할 전략: {len(strategies)}개")

        updated_count = 0
        for strategy_id, params_json in strategies:
            try:
                # params JSON 파싱
                if params_json:
                    params = json.loads(params_json) if isinstance(params_json, str) else params_json
                else:
                    params = {}

                # pattern_confidence 계산
                confidence = calculate_pattern_confidence(params)

                # 업데이트
                cursor.execute("""
                    UPDATE coin_strategies
                    SET pattern_confidence = ?
                    WHERE id = ?
                """, (confidence, strategy_id))

                updated_count += 1

                if updated_count % 1000 == 0:
                    logger.info(f"⏳ 진행 중: {updated_count}/{len(strategies)}")
                    conn.commit()

            except Exception as e:
                logger.error(f"❌ 전략 {strategy_id} 업데이트 실패: {e}")
                continue

        conn.commit()

        logger.info(f"✅ pattern_confidence 업데이트 완료: {updated_count}개 전략")

        # 결과 확인
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                AVG(pattern_confidence) as avg_confidence,
                MIN(pattern_confidence) as min_confidence,
                MAX(pattern_confidence) as max_confidence,
                SUM(CASE WHEN pattern_confidence != 0.5 THEN 1 ELSE 0 END) as non_default
            FROM coin_strategies
        """)

        result = cursor.fetchone()
        logger.info(f"\n📊 업데이트 결과:")
        logger.info(f"   - 총 전략 수: {result[0]}")
        logger.info(f"   - 평균 신뢰도: {result[1]:.4f}")
        logger.info(f"   - 최소 신뢰도: {result[2]:.4f}")
        logger.info(f"   - 최대 신뢰도: {result[3]:.4f}")
        logger.info(f"   - 기본값(0.5) 아닌 전략: {result[4]}개 ({result[4]/result[0]*100:.1f}%)")

        conn.close()

        return updated_count

    except Exception as e:
        logger.error(f"❌ pattern_confidence 업데이트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


if __name__ == "__main__":
    updated = update_all_strategies()
    logger.info(f"\n✅ 총 {updated}개 전략 pattern_confidence 업데이트 완료")
