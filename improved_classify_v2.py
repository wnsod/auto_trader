"""전략 방향 분류 함수 v2 - strategy_type 우선 활용"""

def _classify_strategy_direction(self, strategy: Dict[str, Any]) -> str:
    """🔥 전략을 매수/매도 그룹으로 분류 (strategy_type 우선 버전)

    Args:
        strategy: 전략 딕셔너리

    Returns:
        'buy', 'sell', 또는 'neutral'
    """
    try:
        # ⭐ 1. strategy_type 우선 확인 (가장 정확한 정보)
        strategy_type = strategy.get('strategy_type', '').lower()

        if strategy_type:
            # oversold = 과매도 = 매수 기회
            if 'oversold' in strategy_type or strategy_type == 'buy':
                return 'buy'

            # overbought = 과매수 = 매도 기회
            elif 'overbought' in strategy_type or strategy_type == 'sell':
                return 'sell'

            # mean_reversion = 평균 회귀 -> RSI 기반 판단
            elif 'mean_reversion' in strategy_type or 'reversion' in strategy_type:
                rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
                # 평균 회귀는 극단에서 반대 방향
                if rsi_midpoint < 40:
                    return 'buy'  # 낮은 RSI에서 반등 기대
                elif rsi_midpoint > 60:
                    return 'sell'  # 높은 RSI에서 하락 기대
                else:
                    return 'neutral'

            # trend_following = 추세 추종 -> MACD/ADX 기반 판단
            elif 'trend' in strategy_type:
                macd_buy = strategy.get('macd_buy_threshold', 0.0)
                macd_sell = strategy.get('macd_sell_threshold', 0.0)

                # MACD 차이로 추세 방향 판단
                if macd_buy > macd_sell + 0.01:
                    return 'buy'  # 상승 추세 추종
                elif macd_sell < macd_buy - 0.01:
                    return 'sell'  # 하락 추세 추종
                else:
                    # RSI로 2차 판단
                    rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
                    if rsi_midpoint < 48:
                        return 'buy'
                    elif rsi_midpoint > 52:
                        return 'sell'
                    else:
                        return 'neutral'

            # hybrid나 기타 타입은 다음 단계로
            # (여기서는 패스)

        # 2. 전략 ID/이름 기반 분류 (strategy_type 없을 때)
        buy_score = 0.0
        sell_score = 0.0

        strategy_id = strategy.get('id', '')
        if 'oversold' in strategy_id.lower():
            buy_score += 0.8
        elif 'overbought' in strategy_id.lower():
            sell_score += 0.8
        elif 'buy' in strategy_id.lower():
            buy_score += 0.5
        elif 'sell' in strategy_id.lower():
            sell_score += 0.5

        # 3. 명시적 방향성 특화 전략 확인
        pattern_source = strategy.get('pattern_source', '')
        if pattern_source == 'direction_specialized':
            direction = strategy.get('direction', '')
            if direction == 'BUY':
                buy_score += 1.0
            elif direction == 'SELL':
                sell_score += 1.0

        # 4. RSI 기반 분류 (중앙값과 범위 활용)
        rsi_min = strategy.get('rsi_min', 30.0)
        rsi_max = strategy.get('rsi_max', 70.0)
        rsi_midpoint = (rsi_min + rsi_max) / 2.0
        rsi_range = rsi_max - rsi_min

        if rsi_midpoint < 50:
            buy_score += (50 - rsi_midpoint) / 50.0
        elif rsi_midpoint > 50:
            sell_score += (rsi_midpoint - 50) / 50.0

        # RSI 범위 특화
        if rsi_range < 30:
            specialization_bonus = (30 - rsi_range) / 30.0 * 0.3
            if rsi_midpoint < 50:
                buy_score += specialization_bonus
            else:
                sell_score += specialization_bonus

        # 극단적 RSI
        if rsi_min < 30:
            buy_score += (30 - rsi_min) / 30.0 * 0.5
        if rsi_max > 70:
            sell_score += (rsi_max - 70) / 30.0 * 0.5

        # 5. MACD 기준
        macd_buy_threshold = strategy.get('macd_buy_threshold', 0.0)
        macd_sell_threshold = strategy.get('macd_sell_threshold', 0.0)

        if macd_buy_threshold > 0:
            buy_score += min(macd_buy_threshold * 10, 0.5)
        if macd_sell_threshold < 0:
            sell_score += min(abs(macd_sell_threshold) * 10, 0.5)

        macd_diff = macd_buy_threshold - macd_sell_threshold
        if macd_diff > 0.02:
            buy_score += 0.2
        elif macd_diff < -0.02:
            sell_score += 0.2

        # 6. 볼륨 기준
        volume_ratio_min = strategy.get('volume_ratio_min', 1.0)
        if volume_ratio_min > 1.5:
            if rsi_midpoint < 50:
                buy_score += (volume_ratio_min - 1.0) * 0.2
            else:
                sell_score += (volume_ratio_min - 1.0) * 0.2

        # 7. MFI
        mfi_min = strategy.get('mfi_min', 20.0)
        mfi_max = strategy.get('mfi_max', 80.0)
        mfi_midpoint = (mfi_min + mfi_max) / 2.0

        if mfi_midpoint < 50:
            buy_score += (50 - mfi_midpoint) / 100.0
        elif mfi_midpoint > 50:
            sell_score += (mfi_midpoint - 50) / 100.0

        # 8. 최종 분류 (임계값 0.05)
        score_diff = abs(buy_score - sell_score)

        if buy_score > sell_score and score_diff > 0.05:
            return 'buy'
        elif sell_score > buy_score and score_diff > 0.05:
            return 'sell'
        else:
            # RSI 중앙값으로 최종 결정
            if rsi_midpoint < 48:
                return 'buy'
            elif rsi_midpoint > 52:
                return 'sell'
            else:
                return 'neutral'

    except Exception as e:
        logger.debug(f"전략 방향 분류 실패 (무시): {e}")
        # 에러 시 기본 분류
        try:
            rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
            if rsi_midpoint < 48:
                return 'buy'
            elif rsi_midpoint > 52:
                return 'sell'
        except:
            pass
        return 'neutral'
