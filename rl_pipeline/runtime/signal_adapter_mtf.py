"""
Signal Runtime Adapter MTF Extension
MTF 컨텍스트를 활용한 신호 파라미터 보정
"""
import sys
import os
import logging
from typing import Optional, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.runtime.signal_adapter import SignalRuntimeAdapter, SignalParameters
from rl_pipeline.db.connection_pool import get_strategy_db_pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalRuntimeAdapterMTF(SignalRuntimeAdapter):
    """MTF 확장 신호 어댑터"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("🚀 MTF 확장 어댑터 초기화")

    def get_signal_parameters_with_mtf(self,
                                      coin: str,
                                      interval: str,
                                      regime_tag: str,
                                      strategy_id: str,
                                      base_ts: int,
                                      htf_intervals: Optional[List[str]] = None,
                                      use_mtf_gating: bool = True,
                                      coherence_threshold: float = 0.2) -> Optional[SignalParameters]:
        """
        MTF 컨텍스트를 활용한 신호 파라미터 조회 (확장 버전)

        Args:
            coin: 코인명
            interval: 인터벌
            regime_tag: 레짐 태그
            strategy_id: 전략 ID
            base_ts: Base 신호 시각
            htf_intervals: HTF 인터벌 리스트 (None이면 자동 선택)
            use_mtf_gating: MTF 게이팅 사용 여부
            coherence_threshold: Coherence 임계값

        Returns:
            SignalParameters 또는 None (기준 미달)
        """
        # 1. 기본 파라미터 조회
        base_params = self.get_signal_parameters(coin, interval, regime_tag, strategy_id)

        if not base_params:
            return None

        # 2. HTF 인터벌 자동 선택
        if htf_intervals is None:
            htf_intervals = self._select_htf_intervals(interval)

        # HTF 인터벌이 없으면 기본 파라미터 반환
        if not htf_intervals:
            logger.debug(f"HTF 인터벌 없음, 기본 파라미터 사용")
            return base_params

        # 3. MTF 컨텍스트 조회
        mtf_contexts = self._get_mtf_contexts(coin, interval, strategy_id, base_ts, htf_intervals)

        if not mtf_contexts:
            # MTF 컨텍스트 없으면 기본 파라미터 반환
            logger.debug(f"MTF 컨텍스트 없음, 기본 파라미터 사용")
            return base_params

        # 4. MTF 게이팅 (필터링)
        if use_mtf_gating:
            avg_coherence = sum(ctx['coherence'] for ctx in mtf_contexts) / len(mtf_contexts)
            if avg_coherence < coherence_threshold:
                logger.debug(f"MTF 게이팅 실패: coherence={avg_coherence:.3f} < {coherence_threshold}")
                return None

        # 5. MTF 보정 적용
        adjusted_params = self._apply_mtf_adjustments(base_params, mtf_contexts)

        logger.debug(f"MTF 보정 완료: coherence={avg_coherence:.3f}")

        return adjusted_params

    def _select_htf_intervals(self, base_interval: str) -> List[str]:
        """Base 인터벌에 맞는 HTF 인터벌 자동 선택"""
        if base_interval in ['15m', '30m']:
            return ['240m', '1d']
        elif base_interval == '240m':
            return ['1d']
        else:
            return []

    def _get_mtf_contexts(self,
                         coin: str,
                         base_interval: str,
                         strategy_id: str,
                         base_ts: int,
                         htf_intervals: List[str]) -> List[Dict]:
        """MTF 컨텍스트 조회"""
        pool = get_strategy_db_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            contexts = []
            for htf_interval in htf_intervals:
                cursor.execute("""
                    SELECT htf_regime, htf_trend_state, htf_vol_bucket,
                           align_sign, scale_ratio, coherence
                    FROM mtf_signal_context
                    WHERE base_ts = ?
                      AND base_interval = ?
                      AND base_strategy_id = ?
                      AND htf_interval = ?
                """, (base_ts, base_interval, strategy_id, htf_interval))

                row = cursor.fetchone()
                if row:
                    contexts.append({
                        'htf_interval': htf_interval,
                        'htf_regime': row[0],
                        'htf_trend_state': row[1],
                        'htf_vol_bucket': row[2],
                        'align_sign': row[3],
                        'scale_ratio': row[4],
                        'coherence': row[5]
                    })

            return contexts

    def _apply_mtf_adjustments(self,
                               base_params: SignalParameters,
                               mtf_contexts: List[Dict]) -> SignalParameters:
        """MTF 보정 적용"""
        # 평균 컨텍스트 계산
        n_contexts = len(mtf_contexts)
        avg_trend_up = sum(1 for ctx in mtf_contexts if ctx['htf_trend_state'] == 'up') / n_contexts
        avg_vol_bucket = sum(ctx['htf_vol_bucket'] for ctx in mtf_contexts) / n_contexts
        avg_scale_ratio = sum(ctx['scale_ratio'] for ctx in mtf_contexts) / n_contexts
        avg_coherence = sum(ctx['coherence'] for ctx in mtf_contexts) / n_contexts

        # 1. TP 보정 (HTF 트렌드가 상승이면 TP 상향)
        tp_adjustment = 1.0 + (0.15 * avg_trend_up)  # 최대 15% 상향
        adjusted_tp = base_params.tp * tp_adjustment

        # 2. SL 보정 (HTF 변동성이 높으면 SL 확대)
        sl_adjustment = 1.0 + (0.05 * (avg_vol_bucket / 4.0))  # 최대 5% 확대
        adjusted_sl = base_params.sl * sl_adjustment

        # 3. Hold 보정 (Scale ratio 기반)
        hold_adjustment = max(0.8, min(1.2, avg_scale_ratio))
        adjusted_hold = int(base_params.target_hold * hold_adjustment)
        adjusted_hold = max(2, min(100, adjusted_hold))  # 2~100 범위

        # 4. Size 보정 (Coherence 기반)
        size_adjustment = 0.8 + (0.4 * avg_coherence)  # coherence 0.0→0.8x, 1.0→1.2x
        adjusted_size = base_params.size * size_adjustment

        # 새로운 파라미터 생성
        return SignalParameters(
            tp=round(adjusted_tp, 4),
            sl=round(adjusted_sl, 4),
            target_hold=adjusted_hold,
            size=round(adjusted_size, 2),
            grade=base_params.grade,
            confidence=round(base_params.confidence * avg_coherence, 3),  # Coherence로 신뢰도 보정
            pf=base_params.pf,
            win_rate=base_params.win_rate,
            n_signals=base_params.n_signals,
            rmax_mean=base_params.rmax_mean,
            rmin_mean=base_params.rmin_mean
        )


def main():
    """테스트 함수"""
    logger.info("🚀 MTF 확장 어댑터 테스트\n")

    adapter = SignalRuntimeAdapterMTF()

    # 1. MTF 컨텍스트가 있는 신호 조회
    pool = get_strategy_db_pool()
    with pool.get_connection() as conn:
        cursor = conn.cursor()

        # MTF 컨텍스트가 있는 신호 샘플
        cursor.execute("""
            SELECT DISTINCT m.base_ts, m.base_interval, m.base_strategy_id, m.base_regime
            FROM mtf_signal_context m
            JOIN strategy_label_stats s
              ON m.base_strategy_id = s.strategy_id
             AND m.base_interval = s.interval
             AND m.base_regime = s.regime_tag
            JOIN strategy_grades g
              ON s.strategy_id = g.strategy_id
             AND s.interval = g.interval
             AND s.regime_tag = g.regime_tag
            WHERE g.grade IN ('S', 'A', 'B')
            ORDER BY m.coherence DESC
            LIMIT 5
        """)

        test_signals = cursor.fetchall()

    if not test_signals:
        logger.warning("⚠️ MTF 컨텍스트가 있는 신호가 없습니다. MTF 컨텍스트를 먼저 생성하세요.")
        return

    logger.info(f"✅ {len(test_signals)}개 신호로 테스트\n")

    # 2. 각 신호별 비교 테스트 (기본 vs MTF)
    for base_ts, base_interval, strategy_id, regime_tag in test_signals:
        # 코인명 추출 (strategy_id에서)
        coin = strategy_id.split('_')[0]

        logger.info(f"📊 신호: {coin} {base_interval} {regime_tag} (ts={base_ts})")
        logger.info(f"   전략: {strategy_id[:50]}...")

        # 기본 파라미터
        base_params = adapter.get_signal_parameters(coin, base_interval, regime_tag, strategy_id)

        # MTF 파라미터
        mtf_params = adapter.get_signal_parameters_with_mtf(
            coin, base_interval, regime_tag, strategy_id, base_ts
        )

        if base_params and mtf_params:
            logger.info(f"\n   기본 파라미터:")
            logger.info(f"      TP: {base_params.tp*100:.2f}% | SL: {base_params.sl*100:.2f}%")
            logger.info(f"      Hold: {base_params.target_hold} | Size: {base_params.size:.2f}x")
            logger.info(f"      Confidence: {base_params.confidence:.1%}")

            logger.info(f"\n   MTF 보정 파라미터:")
            logger.info(f"      TP: {mtf_params.tp*100:.2f}% (+{(mtf_params.tp/base_params.tp-1)*100:.1f}%)")
            logger.info(f"      SL: {mtf_params.sl*100:.2f}% ({(mtf_params.sl/base_params.sl-1)*100:+.1f}%)")
            logger.info(f"      Hold: {mtf_params.target_hold} ({mtf_params.target_hold-base_params.target_hold:+d})")
            logger.info(f"      Size: {mtf_params.size:.2f}x ({(mtf_params.size/base_params.size-1)*100:+.1f}%)")
            logger.info(f"      Confidence: {mtf_params.confidence:.1%}")
        elif not mtf_params:
            logger.warning(f"   ⚠️ MTF 게이팅 실패")
        else:
            logger.warning(f"   ⚠️ 기본 파라미터 생성 실패")

        logger.info("")

    logger.info("🎉 테스트 완료!")


if __name__ == "__main__":
    main()
