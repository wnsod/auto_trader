"""
Self-play 적응형 비율 계산 모듈
전략 성숙도에 따라 PREDICTIVE_SELFPLAY_RATIO 자동 조정
"""

import logging
from typing import Dict, Any, Optional
from rl_pipeline.db.connection_pool import get_optimized_db_connection

logger = logging.getLogger(__name__)


def calculate_strategy_maturity(coin: str, interval: str) -> Dict[str, Any]:
    """
    전략 성숙도 평가
    
    평가 기준:
    1. 전략 수: 초기(<100), 안정화(100~500), 성숙(500+)
    2. 품질 분포: 고등급(S/A) 전략 비율
    3. 예측 정확도: 예측 성공률 (예측 실현 Self-play 결과)
    4. 안정성: 최근 성과 변동성
    
    Returns:
        {
            'stage': 'initial' | 'stabilized' | 'mature',
            'strategy_count': int,
            'quality_rate': float,  # S/A 등급 비율
            'prediction_accuracy': float,  # 예측 정확도 (0.0~1.0)
            'maturity_score': float,  # 종합 성숙도 점수 (0.0~1.0)
            'recommended_ratio': float  # 추천 PREDICTIVE_SELFPLAY_RATIO
        }
    """
    try:
        with get_optimized_db_connection("strategies") as conn:
            cursor = conn.cursor()
            
            # 1. 전략 수 조회
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM coin_strategies
                WHERE coin = ? AND interval = ?
            """, (coin, interval))
            strategy_count = cursor.fetchone()[0]
            
            # 2. 품질 분포 조회
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN quality_grade IN ('S', 'A') THEN 1 ELSE 0 END) as high_grade,
                    AVG(win_rate) as avg_win_rate,
                    AVG(profit) as avg_profit
                FROM coin_strategies
                WHERE coin = ? AND interval = ?
            """, (coin, interval))
            
            quality_result = cursor.fetchone()
            if quality_result and quality_result[0] > 0:
                total, high_grade, avg_win_rate, avg_profit = quality_result
                quality_rate = high_grade / total if total > 0 else 0.0
            else:
                quality_rate = 0.0
                avg_win_rate = 0.0
                avg_profit = 0.0
            
            # 3. 예측 정확도 조회 (rl_episode_summary 테이블에서)
            prediction_accuracy = 0.5  # 기본값
            try:
                cursor.execute("""
                    SELECT AVG(acc_flag) as avg_accuracy
                    FROM rl_episode_summary
                    WHERE coin = ? AND interval = ?
                    AND ts_exit >= datetime('now', '-7 days')
                """, (coin, interval))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    prediction_accuracy = float(result[0])
            except Exception as e:
                logger.debug(f"⚠️ 예측 정확도 조회 실패(기본값 사용): {e}")
            
            # 4. 성숙도 단계 판단
            stage = "initial"
            if strategy_count >= 500:
                stage = "mature"
            elif strategy_count >= 100:
                stage = "stabilized"
            
            # 5. 종합 성숙도 점수 계산 (0.0 ~ 1.0)
            # - 전략 수: 30% (500개 이상 = 1.0)
            count_score = min(1.0, strategy_count / 500.0)
            
            # - 품질 비율: 30% (S/A 등급 30% 이상 = 1.0)
            quality_score = min(1.0, quality_rate / 0.3) if quality_rate > 0 else 0.0
            
            # - 예측 정확도: 25% (70% 이상 = 1.0)
            accuracy_score = min(1.0, prediction_accuracy / 0.7) if prediction_accuracy > 0.5 else 0.0
            
            # - 평균 승률: 15% (50% 이상 = 1.0)
            win_rate_score = min(1.0, (avg_win_rate or 0.0) / 0.5) if avg_win_rate else 0.0
            
            maturity_score = (
                count_score * 0.30 +
                quality_score * 0.30 +
                accuracy_score * 0.25 +
                win_rate_score * 0.15
            )
            
            # 6. 추천 비율 계산
            # 초기 단계: 0.2 (20%) - 예측 능력 기초 학습
            # 안정화 단계: 0.5 (50%) - 균형 학습
            # 성숙 단계: 0.8-1.0 (80-100%) - 예측 정확도 최적화
            
            if stage == "initial":
                recommended_ratio = 0.2
            elif stage == "stabilized":
                # 성숙도 점수에 따라 0.3 ~ 0.7 사이 조정
                recommended_ratio = 0.3 + (maturity_score * 0.4)
            else:  # mature
                # 예측 정확도에 따라 0.8 ~ 1.0 사이 조정
                if prediction_accuracy >= 0.7:
                    recommended_ratio = 1.0  # 정확도 높으면 100%
                else:
                    recommended_ratio = 0.8 + (prediction_accuracy * 0.2)  # 0.8 ~ 1.0
            
            logger.info(f"📊 {coin}-{interval} 성숙도 평가: "
                       f"단계={stage}, 전략={strategy_count}개, "
                       f"품질={quality_rate:.1%}, 예측정확도={prediction_accuracy:.1%}, "
                       f"점수={maturity_score:.2f}, 추천비율={recommended_ratio:.1%}")
            
            return {
                'stage': stage,
                'strategy_count': strategy_count,
                'quality_rate': quality_rate,
                'prediction_accuracy': prediction_accuracy,
                'maturity_score': maturity_score,
                'recommended_ratio': recommended_ratio,
                'avg_win_rate': avg_win_rate or 0.0,
                'avg_profit': avg_profit or 0.0
            }
            
    except Exception as e:
        logger.error(f"❌ {coin}-{interval} 성숙도 평가 실패: {e}")
        # 기본값 반환
        return {
            'stage': 'initial',
            'strategy_count': 0,
            'quality_rate': 0.0,
            'prediction_accuracy': 0.5,
            'maturity_score': 0.0,
            'recommended_ratio': 0.2,
            'avg_win_rate': 0.0,
            'avg_profit': 0.0
        }


def get_adaptive_predictive_ratio(
    coin: str,
    interval: str,
    base_ratio: Optional[float] = None,
    enable_auto: bool = True
) -> float:
    """
    적응형 예측 Self-play 비율 계산
    
    Args:
        coin: 코인 심볼
        interval: 인터벌
        base_ratio: 기본 비율 (None이면 환경변수 사용)
        enable_auto: 자동 조정 활성화 (False면 base_ratio만 사용)
    
    Returns:
        최종 PREDICTIVE_SELFPLAY_RATIO (0.0 ~ 1.0)
    """
    try:
        import os
        
        # 기본값: 환경변수 또는 기본값 0.2
        if base_ratio is None:
            base_ratio = float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))
        
        # 자동 조정 비활성화 시 기본값 반환
        enable_auto_env = os.getenv('ENABLE_AUTO_PREDICTIVE_RATIO', 'true').lower() == 'true'
        if not enable_auto or not enable_auto_env:
            logger.debug(f"📊 {coin}-{interval} 자동 비율 조정 비활성화, 기본값 사용: {base_ratio:.1%}")
            return base_ratio
        
        # 성숙도 평가
        maturity = calculate_strategy_maturity(coin, interval)
        recommended_ratio = maturity['recommended_ratio']
        
        # 기본값과 추천값 중 더 큰 값 사용 (점진적 증가 보장)
        final_ratio = max(base_ratio, recommended_ratio)
        
        # 최대 1.0으로 제한
        final_ratio = min(1.0, final_ratio)
        
        if final_ratio != base_ratio:
            logger.info(f"🔄 {coin}-{interval} 자동 비율 조정: {base_ratio:.1%} → {final_ratio:.1%} "
                       f"(단계: {maturity['stage']}, 성숙도: {maturity['maturity_score']:.2f})")
        else:
            logger.debug(f"📊 {coin}-{interval} 비율 유지: {final_ratio:.1%} "
                        f"(단계: {maturity['stage']})")
        
        return final_ratio
        
    except Exception as e:
        logger.warning(f"⚠️ {coin}-{interval} 적응형 비율 계산 실패, 기본값 사용: {e}")
        import os
        return float(os.getenv('PREDICTIVE_SELFPLAY_RATIO', '0.2'))

