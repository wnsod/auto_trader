"""
새로운 등급 시스템 테스트 스크립트

예측 정확도 기반 상대평가 시스템 테스트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from rl_pipeline.core.strategy_grading import RelativeGrading, PredictionMetrics, StrategyScore
from rl_pipeline.analysis.strategy_grade_updater import StrategyGradeUpdater

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_prediction_metrics():
    """예측 정확도 계산 테스트"""
    logger.info("\n" + "="*80)
    logger.info("1️⃣ 예측 정확도 계산 테스트")
    logger.info("="*80)

    test_cases = [
        {"name": "고승률 고수익팩터", "win_rate": 0.65, "profit_factor": 2.5, "trades": 50},
        {"name": "중승률 중수익팩터", "win_rate": 0.45, "profit_factor": 1.5, "trades": 30},
        {"name": "저승률 저수익팩터", "win_rate": 0.35, "profit_factor": 0.9, "trades": 20},
        {"name": "거래횟수 부족", "win_rate": 0.55, "profit_factor": 2.0, "trades": 5},
    ]

    for case in test_cases:
        prediction_acc = PredictionMetrics.calculate_prediction_accuracy(
            case["win_rate"], case["profit_factor"], case["trades"]
        )
        signal_prec = PredictionMetrics.calculate_signal_precision(
            5.0, case["win_rate"], case["trades"]
        )

        logger.info(f"\n📊 {case['name']}:")
        logger.info(f"   승률: {case['win_rate']:.2%}, 수익팩터: {case['profit_factor']:.2f}, 거래: {case['trades']}회")
        logger.info(f"   → 예측 정확도: {prediction_acc:.2%}")
        logger.info(f"   → 신호 정밀도: {signal_prec:.2%}")


def test_composite_score():
    """종합 점수 계산 테스트"""
    logger.info("\n" + "="*80)
    logger.info("2️⃣ 종합 점수 계산 테스트")
    logger.info("="*80)

    test_strategies = [
        {
            "name": "공격적 전략",
            "profit": 15.0,
            "win_rate": 0.45,
            "sharpe": 1.3,
            "max_dd": 0.25,
            "profit_factor": 3.0,
            "trades": 100
        },
        {
            "name": "안정적 전략",
            "profit": 5.0,
            "win_rate": 0.55,
            "sharpe": 1.8,
            "max_dd": 0.08,
            "profit_factor": 2.2,
            "trades": 80
        },
        {
            "name": "평범한 전략",
            "profit": 2.0,
            "win_rate": 0.38,
            "sharpe": 0.5,
            "max_dd": 0.30,
            "profit_factor": 1.3,
            "trades": 50
        },
        {
            "name": "손실 전략",
            "profit": -3.0,
            "win_rate": 0.30,
            "sharpe": -0.2,
            "max_dd": 0.45,
            "profit_factor": 0.7,
            "trades": 40
        }
    ]

    for strategy in test_strategies:
        composite_score = RelativeGrading.calculate_composite_score(
            profit_percent=strategy["profit"],
            win_rate=strategy["win_rate"],
            sharpe=strategy["sharpe"],
            max_dd=strategy["max_dd"],
            profit_factor=strategy["profit_factor"],
            trades_count=strategy["trades"]
        )

        logger.info(f"\n📊 {strategy['name']}:")
        logger.info(f"   수익: {strategy['profit']:+.1f}%, 승률: {strategy['win_rate']:.2%}")
        logger.info(f"   Sharpe: {strategy['sharpe']:.2f}, 낙폭: {strategy['max_dd']:.2%}")
        logger.info(f"   → 종합 점수: {composite_score:.3f}")


def test_relative_grading():
    """상대평가 등급 부여 테스트"""
    logger.info("\n" + "="*80)
    logger.info("3️⃣ 상대평가 등급 부여 테스트")
    logger.info("="*80)

    # 샘플 전략 20개 생성
    import random
    random.seed(42)

    strategies = []
    for i in range(20):
        # 다양한 성과 분포 생성
        win_rate = random.uniform(0.25, 0.70)
        profit_pct = random.uniform(-5, 20)

        strategies.append({
            'id': f'strategy_{i+1:02d}',
            'profit': profit_pct * 100,  # 달러 단위 (10000 = 100%)
            'win_rate': win_rate,
            'sharpe': random.uniform(-0.5, 2.5),
            'max_dd': random.uniform(0.05, 0.50),
            'profit_factor': random.uniform(0.5, 3.5),
            'trades': random.randint(20, 150)
        })

    # 상대평가 실행
    coin = "BTC"
    interval = "15m"
    regime = "trending"

    logger.info(f"\n📊 테스트 그룹: {coin}-{interval}-{regime} (전략 {len(strategies)}개)")

    scored_strategies = RelativeGrading.assign_grades_by_group(
        strategies, coin, interval, regime
    )

    # 등급별 분포 출력
    grade_counts = {}
    for score in scored_strategies:
        grade_counts[score.grade] = grade_counts.get(score.grade, 0) + 1

    logger.info("\n📊 등급 분포:")
    for grade in ['S', 'A', 'B', 'C', 'D', 'F']:
        count = grade_counts.get(grade, 0)
        percentage = (count / len(scored_strategies)) * 100 if scored_strategies else 0
        logger.info(f"   {grade}등급: {count}개 ({percentage:.1f}%)")

    # 상위 5개 전략 출력
    logger.info("\n🏆 상위 5개 전략:")
    for i, score in enumerate(scored_strategies[:5], 1):
        logger.info(
            f"   {i}. {score.strategy_id} [{score.grade}등급] "
            f"종합: {score.composite_score:.3f}, "
            f"예측: {score.prediction_accuracy:.2%}, "
            f"수익: {score.profit_percent:+.1f}%"
        )

    # 하위 3개 전략 출력
    logger.info("\n📉 하위 3개 전략:")
    for i, score in enumerate(scored_strategies[-3:], len(scored_strategies)-2):
        logger.info(
            f"   {i}. {score.strategy_id} [{score.grade}등급] "
            f"종합: {score.composite_score:.3f}, "
            f"예측: {score.prediction_accuracy:.2%}, "
            f"수익: {score.profit_percent:+.1f}%"
        )


def test_grade_weights():
    """가중치 영향 테스트"""
    logger.info("\n" + "="*80)
    logger.info("4️⃣ 가중치 영향 분석 테스트")
    logger.info("="*80)

    weights = RelativeGrading.WEIGHTS

    logger.info("\n⚖️ 현재 가중치 설정:")
    for metric, weight in weights.items():
        logger.info(f"   {metric}: {weight:.2%}")

    logger.info("\n📊 각 지표별 영향력 분석:")

    # 기본 전략
    base_strategy = {
        "profit": 5.0,
        "win_rate": 0.45,
        "sharpe": 1.0,
        "max_dd": 0.20,
        "profit_factor": 1.5,
        "trades": 50
    }

    base_score = RelativeGrading.calculate_composite_score(**base_strategy)
    logger.info(f"\n기준 전략 점수: {base_score:.3f}")

    # 각 지표를 20% 향상시켰을 때 영향
    improvements = {
        "prediction_accuracy": ("예측 정확도", "win_rate", 0.45 * 1.2),
        "profit": ("수익률", "profit", 5.0 * 1.2),
        "sharpe": ("Sharpe 비율", "sharpe", 1.0 * 1.2),
        "max_dd": ("최대 낙폭", "max_dd", 0.20 * 0.8),  # 낮을수록 좋으므로 -20%
    }

    logger.info("\n각 지표를 20% 개선했을 때 점수 변화:")
    for key, (name, param, new_value) in improvements.items():
        test_strategy = base_strategy.copy()
        test_strategy[param] = new_value

        new_score = RelativeGrading.calculate_composite_score(**test_strategy)
        delta = new_score - base_score

        logger.info(
            f"   {name} 개선: {base_score:.3f} → {new_score:.3f} "
            f"(변화: {delta:+.3f}, {(delta/base_score)*100:+.1f}%)"
        )


def main():
    """메인 테스트 함수"""
    try:
        logger.info("\n" + "🚀"*40)
        logger.info("새로운 등급 시스템 테스트 시작")
        logger.info("🚀"*40)

        # 테스트 실행
        test_prediction_metrics()
        test_composite_score()
        test_relative_grading()
        test_grade_weights()

        logger.info("\n" + "✅"*40)
        logger.info("모든 테스트 완료!")
        logger.info("✅"*40 + "\n")

        logger.info("\n📝 다음 단계:")
        logger.info("   1. 실제 전략 데이터로 등급 재계산")
        logger.info("   2. absolute_zero_system.py에 통합")
        logger.info("   3. Self-play 결과와 비교 검증")

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
