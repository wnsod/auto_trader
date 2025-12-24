"""
MTF (Multi-Timeframe) 컨텍스트 빌더
Base 신호에 대한 HTF 컨텍스트 생성 및 coherence 계산
"""
import sys
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rl_pipeline.db.connection_pool import get_strategy_db_pool, get_candle_db_pool

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 인터벌별 분 단위 매핑
INTERVAL_MINUTES = {
    '15m': 15,
    '30m': 30,
    '240m': 240,
    '1d': 1440
}

@dataclass
class MTFContext:
    """MTF 컨텍스트 데이터"""
    base_ts: int
    base_interval: str
    base_strategy_id: str
    base_regime: str

    htf_interval: str
    htf_regime: str
    htf_trend_state: str
    htf_vol_bucket: int

    align_sign: int
    scale_ratio: float
    coherence: float


class MTFContextBuilder:
    """MTF 컨텍스트 빌더"""

    def __init__(self,
                 coherence_w_align: float = 0.6,
                 scale_clip_min: float = 0.1,
                 scale_clip_max: float = 50.0):
        """
        Args:
            coherence_w_align: 정렬 가중치 (0~1)
            scale_clip_min: scale_ratio 최소값
            scale_clip_max: scale_ratio 최대값
        """
        self.coherence_w_align = coherence_w_align
        self.scale_clip_min = scale_clip_min
        self.scale_clip_max = scale_clip_max

        logger.info(f"🚀 MTF 컨텍스트 빌더 초기화")
        logger.info(f"   coherence_w_align: {coherence_w_align}")
        logger.info(f"   scale_clip: [{scale_clip_min}, {scale_clip_max}]")

    @staticmethod
    def map_regime_label_to_tag(regime_label: str) -> str:
        """
        HTF 캔들의 regime_label을 우리 시스템의 regime_tag로 매핑

        Args:
            regime_label: HTF 캔들의 레짐 레이블 (예: sideways_bearish, extreme_bullish)

        Returns:
            regime_tag: ranging/trending/volatile
        """
        regime_label_lower = regime_label.lower()

        # Trending 매핑
        if any(keyword in regime_label_lower for keyword in ['bullish', 'bearish', 'trend']):
            return 'trending'

        # Volatile 매핑
        if any(keyword in regime_label_lower for keyword in ['extreme', 'volatile']):
            return 'volatile'

        # Ranging 매핑 (기본값)
        return 'ranging'

    def find_htf_candle_for_ts(self,
                               base_ts: int,
                               base_interval: str,
                               htf_interval: str,
                               coin: str) -> Optional[Tuple]:
        """
        Base 신호 시각에 해당하는 HTF 캔들 찾기 (벽시계 정렬)

        Args:
            base_ts: Base 신호 시각 (epoch)
            base_interval: Base 인터벌
            htf_interval: HTF 인터벌
            coin: 코인명

        Returns:
            HTF 캔들 레코드 또는 None
        """
        try:
            candle_pool = get_candle_db_pool()
            with candle_pool.get_connection() as conn:
                cursor = conn.cursor()

                # HTF 캔들 구간 계산
                htf_minutes = INTERVAL_MINUTES.get(htf_interval, 15)
                htf_duration = htf_minutes * 60  # 초 단위

                # base_ts가 포함되는 HTF 캔들 찾기
                # HTF 캔들의 시작 시각 <= base_ts < HTF 캔들의 종료 시각
                cursor.execute("""
                    SELECT timestamp, regime_label, rsi, macd, macd_signal, bb_upper, bb_lower, bb_middle,
                           atr, adx, mfi, volume_ratio
                    FROM candles
                    WHERE symbol = ?
                      AND interval = ?
                      AND timestamp <= ?
                      AND timestamp + ? > ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (coin, htf_interval, base_ts, htf_duration, base_ts))

                result = cursor.fetchone()
                return result

        except Exception as e:
            logger.error(f"❌ HTF 캔들 조회 실패: {e}")
            return None

    def calculate_htf_trend_state(self, candle_data: Tuple) -> str:
        """
        HTF 트렌드 상태 계산

        Args:
            candle_data: 캔들 데이터 (ts, regime, rsi, macd, macd_signal, ...)

        Returns:
            'up', 'down', 'flat'
        """
        try:
            # candle_data = (ts, regime, rsi, macd, macd_signal, bb_upper, bb_lower, bb_middle, ...)
            macd = candle_data[3] if len(candle_data) > 3 else 0
            macd_signal = candle_data[4] if len(candle_data) > 4 else 0

            # MACD 기반 트렌드 판단
            if macd > macd_signal and macd > 0:
                return 'up'
            elif macd < macd_signal and macd < 0:
                return 'down'
            else:
                return 'flat'

        except Exception as e:
            logger.error(f"❌ HTF 트렌드 계산 실패: {e}")
            return 'flat'

    def calculate_htf_vol_bucket(self, candle_data: Tuple) -> int:
        """
        HTF 변동성 버킷 계산 (0~4)

        Args:
            candle_data: 캔들 데이터

        Returns:
            0~4 (0: 매우 낮음, 4: 매우 높음)
        """
        try:
            # candle_data = (ts, regime, rsi, macd, macd_signal, bb_upper, bb_lower, bb_middle, atr, ...)
            bb_upper = candle_data[5] if len(candle_data) > 5 else 1.02
            bb_lower = candle_data[6] if len(candle_data) > 6 else 0.98
            bb_middle = candle_data[7] if len(candle_data) > 7 else 1.0

            if bb_middle == 0:
                return 2  # 중간값

            # BB 폭 계산
            bb_width = (bb_upper - bb_lower) / bb_middle

            # 폭에 따라 버킷 할당 (경험적 분위수)
            if bb_width < 0.02:
                return 0
            elif bb_width < 0.04:
                return 1
            elif bb_width < 0.06:
                return 2
            elif bb_width < 0.08:
                return 3
            else:
                return 4

        except Exception as e:
            logger.error(f"❌ HTF 변동성 버킷 계산 실패: {e}")
            return 2

    def calculate_coherence(self,
                           base_rmax_mean: float,
                           htf_rmax_mean: float,
                           base_kmax_mean: float,
                           htf_kmax_mean: float,
                           base_interval: str,
                           htf_interval: str) -> Tuple[int, float, float]:
        """
        Coherence (정합도) 계산

        Args:
            base_rmax_mean: Base r_max 평균
            htf_rmax_mean: HTF r_max 평균
            base_kmax_mean: Base k_max 평균
            htf_kmax_mean: HTF k_max 평균
            base_interval: Base 인터벌
            htf_interval: HTF 인터벌

        Returns:
            (align_sign, scale_ratio, coherence)
        """
        try:
            # 1. 방향 일치 (align_sign)
            base_sign = 1 if base_rmax_mean > 0 else -1 if base_rmax_mean < 0 else 0
            htf_sign = 1 if htf_rmax_mean > 0 else -1 if htf_rmax_mean < 0 else 0
            align_sign = 1 if base_sign == htf_sign else 0

            # 2. 시간 스케일 비율 (scale_ratio)
            base_minutes = INTERVAL_MINUTES.get(base_interval, 15)
            htf_minutes = INTERVAL_MINUTES.get(htf_interval, 240)

            # 안전 장치: 0 나누기 방지
            if base_kmax_mean == 0 or base_minutes == 0:
                scale_ratio = 1.0
            else:
                # scale_ratio = (kmax_htf * htf_minutes) / (kmax_base * base_minutes)
                numerator = htf_kmax_mean * htf_minutes
                denominator = base_kmax_mean * base_minutes
                if denominator == 0:
                    scale_ratio = 1.0
                else:
                    scale_ratio = numerator / denominator

                # 클리핑
                scale_ratio = max(self.scale_clip_min, min(self.scale_clip_max, scale_ratio))

            # 3. Coherence 점수
            # coherence = w * align_sign + (1-w) * exp(-|log(scale_ratio)|)
            w = self.coherence_w_align

            # 안전 장치: log(0) 방지
            if scale_ratio <= 0:
                scale_factor = 0.0
            else:
                log_scale = abs(np.log(scale_ratio))
                scale_factor = np.exp(-log_scale)

            coherence = w * align_sign + (1 - w) * scale_factor
            coherence = float(np.clip(coherence, 0.0, 1.0))

            return align_sign, float(scale_ratio), coherence

        except Exception as e:
            logger.error(f"❌ Coherence 계산 실패: {e}")
            return 0, 1.0, 0.5

    def build_context_for_signal(self,
                                 coin: str,
                                 base_ts: int,
                                 base_interval: str,
                                 base_strategy_id: str,
                                 base_regime: str,
                                 htf_intervals: List[str] = ['240m', '1d']) -> List[MTFContext]:
        """
        단일 신호에 대한 MTF 컨텍스트 생성

        Args:
            coin: 코인명
            base_ts: Base 신호 시각
            base_interval: Base 인터벌
            base_strategy_id: 전략 ID
            base_regime: Base 레짐
            htf_intervals: HTF 인터벌 리스트

        Returns:
            MTFContext 리스트
        """
        contexts = []

        try:
            strategy_pool = get_strategy_db_pool()
            with strategy_pool.get_connection() as conn:
                cursor = conn.cursor()

                # Base 라벨 통계 조회
                cursor.execute("""
                    SELECT rmax_mean, kmax_mean
                    FROM strategy_label_stats
                    WHERE symbol = ? AND interval = ? AND regime_tag = ? AND strategy_id = ?
                """, (coin, base_interval, base_regime, base_strategy_id))

                base_stats = cursor.fetchone()

                if not base_stats:
                    logger.warning(f"   ⚠️ Base 통계 없음: {coin} {base_interval} {base_regime}")
                    return contexts

                base_rmax_mean, base_kmax_mean = base_stats

            # 각 HTF 인터벌에 대해 컨텍스트 생성
            for htf_interval in htf_intervals:
                # HTF 캔들 찾기
                htf_candle = self.find_htf_candle_for_ts(base_ts, base_interval, htf_interval, coin)

                if not htf_candle:
                    logger.debug(f"   ⚠️ HTF 캔들 없음: {htf_interval}")
                    continue

                # HTF 레짐 (regime_label → regime_tag 매핑)
                htf_regime_label = htf_candle[1] if len(htf_candle) > 1 else 'neutral'
                htf_regime = self.map_regime_label_to_tag(htf_regime_label)

                # HTF 트렌드 상태
                htf_trend_state = self.calculate_htf_trend_state(htf_candle)

                # HTF 변동성 버킷
                htf_vol_bucket = self.calculate_htf_vol_bucket(htf_candle)

                # HTF 라벨 통계 조회 (레짐별 평균 사용)
                # 같은 strategy_id가 없어도 HTF 레짐의 평균 통계로 coherence 계산 가능
                with strategy_pool.get_connection() as conn:
                    cursor = conn.cursor()

                    # 레짐별 평균 통계 조회
                    cursor.execute("""
                        SELECT AVG(rmax_mean) AS rmax_mean, AVG(kmax_mean) AS kmax_mean
                        FROM strategy_label_stats
                        WHERE symbol = ? AND interval = ? AND regime_tag = ?
                    """, (coin, htf_interval, htf_regime))

                    htf_stats = cursor.fetchone()

                if not htf_stats or htf_stats[0] is None:
                    # HTF 통계가 없으면 coherence 계산 불가
                    logger.debug(f"   ⚠️ HTF 통계 없음: {htf_interval} {htf_regime}")
                    continue

                htf_rmax_mean, htf_kmax_mean = htf_stats

                # Coherence 계산
                align_sign, scale_ratio, coherence = self.calculate_coherence(
                    base_rmax_mean, htf_rmax_mean,
                    base_kmax_mean, htf_kmax_mean,
                    base_interval, htf_interval
                )

                # MTFContext 생성
                context = MTFContext(
                    base_ts=base_ts,
                    base_interval=base_interval,
                    base_strategy_id=base_strategy_id,
                    base_regime=base_regime,
                    htf_interval=htf_interval,
                    htf_regime=htf_regime,
                    htf_trend_state=htf_trend_state,
                    htf_vol_bucket=htf_vol_bucket,
                    align_sign=align_sign,
                    scale_ratio=scale_ratio,
                    coherence=coherence
                )

                contexts.append(context)

                logger.debug(f"   ✅ {htf_interval}: coherence={coherence:.3f} "
                          f"align={align_sign} scale={scale_ratio:.2f}")

            return contexts

        except Exception as e:
            logger.error(f"❌ MTF 컨텍스트 생성 실패: {e}")
            return contexts

    def save_contexts(self, contexts: List[MTFContext]) -> int:
        """
        MTF 컨텍스트를 DB에 저장

        Args:
            contexts: MTFContext 리스트

        Returns:
            저장된 개수
        """
        if not contexts:
            return 0

        try:
            strategy_pool = get_strategy_db_pool()
            with strategy_pool.get_connection() as conn:
                cursor = conn.cursor()

                saved_count = 0
                for ctx in contexts:
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO mtf_signal_context
                            (base_ts, base_interval, base_strategy_id, base_regime,
                             htf_interval, htf_regime, htf_trend_state, htf_vol_bucket,
                             align_sign, scale_ratio, coherence, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ctx.base_ts, ctx.base_interval, ctx.base_strategy_id, ctx.base_regime,
                            ctx.htf_interval, ctx.htf_regime, ctx.htf_trend_state, ctx.htf_vol_bucket,
                            ctx.align_sign, ctx.scale_ratio, ctx.coherence, int(time.time())
                        ))
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"   ⚠️ 컨텍스트 저장 실패: {e}")

                conn.commit()

                return saved_count

        except Exception as e:
            logger.error(f"❌ 컨텍스트 저장 실패: {e}")
            return 0

    def build_and_save_for_coin_interval(self,
                                        coin: str,
                                        base_interval: str,
                                        htf_intervals: List[str] = ['240m', '1d'],
                                        limit: Optional[int] = None) -> Dict[str, int]:
        """
        특정 코인·인터벌의 모든 신호에 대해 MTF 컨텍스트 생성 및 저장

        Args:
            coin: 코인명
            base_interval: Base 인터벌
            htf_intervals: HTF 인터벌 리스트
            limit: 최대 처리 개수 (None이면 전체)

        Returns:
            {'processed': N, 'saved': M}
        """
        logger.info(f"\n🔧 MTF 컨텍스트 빌드 시작: {coin} {base_interval}")
        logger.info(f"   HTF 인터벌: {htf_intervals}")

        try:
            strategy_pool = get_strategy_db_pool()
            with strategy_pool.get_connection() as conn:
                cursor = conn.cursor()

                # 라벨링된 신호 조회 (ts, regime_tag, strategy_id 기준)
                query = """
                    SELECT DISTINCT ts, regime_tag, strategy_id
                    FROM strategy_signal_labels
                    WHERE symbol = ? AND interval = ?
                    ORDER BY ts
                """

                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query, (coin, base_interval))
                signals = cursor.fetchall()

            logger.info(f"   신호 수: {len(signals)}개")

            processed_count = 0
            saved_count = 0

            for ts, regime_tag, strategy_id in signals:
                # MTF 컨텍스트 생성
                contexts = self.build_context_for_signal(
                    coin=coin,
                    base_ts=ts,
                    base_interval=base_interval,
                    base_strategy_id=strategy_id,
                    base_regime=regime_tag,
                    htf_intervals=htf_intervals
                )

                # 저장
                if contexts:
                    saved = self.save_contexts(contexts)
                    saved_count += saved

                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"   진행: {processed_count}/{len(signals)} "
                              f"(저장: {saved_count}개)")

            logger.info(f"✅ 완료: {processed_count}개 처리, {saved_count}개 저장\n")

            return {'processed': processed_count, 'saved': saved_count}

        except Exception as e:
            logger.error(f"❌ MTF 컨텍스트 빌드 실패: {e}")
            return {'processed': 0, 'saved': 0}

    def update_mtf_stats(self) -> bool:
        """
        mtf_stats_by_pair 테이블 갱신

        Returns:
            성공 여부
        """
        logger.info("\n🔧 MTF 통계 갱신 시작...")

        try:
            strategy_pool = get_strategy_db_pool()
            with strategy_pool.get_connection() as conn:
                cursor = conn.cursor()

                # base_interval, htf_interval, regime_combo별로 집계
                cursor.execute("""
                    SELECT base_interval, htf_interval,
                           base_regime || '-' || htf_regime AS regime_combo,
                           AVG(align_sign) AS align_rate_mean,
                           AVG(scale_ratio) AS scale_ratio_mean,
                           AVG(coherence) AS coherence_mean,
                           COUNT(*) AS n_pairs
                    FROM mtf_signal_context
                    GROUP BY base_interval, htf_interval, regime_combo
                """)

                stats = cursor.fetchall()

                logger.info(f"   통계 레코드: {len(stats)}개")

                for stat in stats:
                    base_interval, htf_interval, regime_combo, align_rate, scale_ratio, coherence, n_pairs = stat

                    cursor.execute("""
                        INSERT OR REPLACE INTO mtf_stats_by_pair
                        (base_interval, htf_interval, regime_combo,
                         align_rate_mean, scale_ratio_mean, coherence_mean, n_pairs, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        base_interval, htf_interval, regime_combo,
                        align_rate, scale_ratio, coherence, n_pairs, int(time.time())
                    ))

                conn.commit()

                logger.info(f"✅ MTF 통계 갱신 완료\n")
                return True

        except Exception as e:
            logger.error(f"❌ MTF 통계 갱신 실패: {e}")
            return False


def main():
    """테스트 함수"""
    logger.info("🚀 MTF 컨텍스트 빌더 테스트\n")

    # 빌더 생성
    builder = MTFContextBuilder()

    # BTC 15m 테스트 (최대 50개 신호)
    result = builder.build_and_save_for_coin_interval(
        coin='BTC',
        base_interval='15m',
        htf_intervals=['240m', '1d'],
        limit=50
    )

    logger.info(f"📊 테스트 결과: {result}")

    # 통계 갱신
    builder.update_mtf_stats()

    logger.info("🎉 테스트 완료!")


if __name__ == "__main__":
    main()
