#!/usr/bin/env python
"""전략 방향 분류 함수 교체 스크립트"""
import re

# 새로운 함수 정의
NEW_FUNCTION = '''    def _classify_strategy_direction(self, strategy: Dict[str, Any]) -> str:
        """🔥 전략을 매수/매도 그룹으로 분류 (개선 버전)

        Args:
            strategy: 전략 딕셔너리

        Returns:
            'buy', 'sell', 또는 'neutral'
        """
        try:
            buy_score = 0.0
            sell_score = 0.0

            # 1. 전략 ID/이름 기반 분류 (가장 명확한 신호)
            strategy_id = strategy.get('id', '')
            if 'oversold' in strategy_id.lower():
                buy_score += 0.8  # oversold = 과매도 = 매수 기회
            elif 'overbought' in strategy_id.lower():
                sell_score += 0.8  # overbought = 과매수 = 매도 기회
            elif 'buy' in strategy_id.lower():
                buy_score += 0.5
            elif 'sell' in strategy_id.lower():
                sell_score += 0.5

            # 2. 명시적 방향성 특화 전략 확인
            pattern_source = strategy.get('pattern_source', '')
            if pattern_source == 'direction_specialized':
                direction = strategy.get('direction', '')
                if direction == 'BUY':
                    buy_score += 1.0
                elif direction == 'SELL':
                    sell_score += 1.0

            # 3. RSI 기반 분류 (개선: 중앙값과 범위 폭 활용)
            rsi_min = strategy.get('rsi_min', 30.0)
            rsi_max = strategy.get('rsi_max', 70.0)

            # RSI 중앙값 계산
            rsi_midpoint = (rsi_min + rsi_max) / 2.0
            rsi_range = rsi_max - rsi_min

            # RSI 중앙값이 50보다 낮으면 매수 전략, 높으면 매도 전략
            if rsi_midpoint < 50:
                # 중앙값이 낮을수록 매수 성향 강함
                buy_score += (50 - rsi_midpoint) / 50.0  # 0.0 ~ 1.0
            elif rsi_midpoint > 50:
                # 중앙값이 높을수록 매도 성향 강함
                sell_score += (rsi_midpoint - 50) / 50.0  # 0.0 ~ 1.0

            # RSI 범위가 좁을수록 특화된 전략 (가중치 증가)
            if rsi_range < 30:
                specialization_bonus = (30 - rsi_range) / 30.0 * 0.3
                if rsi_midpoint < 50:
                    buy_score += specialization_bonus
                else:
                    sell_score += specialization_bonus

            # 극단적인 RSI 범위 (추가 점수)
            if rsi_min < 30:  # 과매도 영역 포함
                buy_score += (30 - rsi_min) / 30.0 * 0.5
            if rsi_max > 70:  # 과매수 영역 포함
                sell_score += (rsi_max - 70) / 30.0 * 0.5

            # 4. MACD 기준 (개선: 임계값 차이도 고려)
            macd_buy_threshold = strategy.get('macd_buy_threshold', 0.0)
            macd_sell_threshold = strategy.get('macd_sell_threshold', 0.0)

            # MACD 매수 임계값이 양수면 매수 성향
            if macd_buy_threshold > 0:
                buy_score += min(macd_buy_threshold * 10, 0.5)  # 최대 0.5

            # MACD 매도 임계값이 음수면 매도 성향
            if macd_sell_threshold < 0:
                sell_score += min(abs(macd_sell_threshold) * 10, 0.5)  # 최대 0.5

            # MACD 차이 (buy - sell)가 크면 추세 추종 성향
            macd_diff = macd_buy_threshold - macd_sell_threshold
            if macd_diff > 0.02:
                buy_score += 0.2
            elif macd_diff < -0.02:
                sell_score += 0.2

            # 5. 볼륨 기준
            volume_ratio_min = strategy.get('volume_ratio_min', 1.0)

            # 높은 볼륨 요구 = 돌파/추세 전략 = 방향성 강함
            if volume_ratio_min > 1.5:
                # RSI 중앙값에 따라 방향 결정
                if rsi_midpoint < 50:
                    buy_score += (volume_ratio_min - 1.0) * 0.2
                else:
                    sell_score += (volume_ratio_min - 1.0) * 0.2

            # 6. 스탑로스/이익실현 비율
            stop_loss_pct = strategy.get('stop_loss_pct', 0.02)
            take_profit_pct = strategy.get('take_profit_pct', 0.04)

            risk_reward_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 2.0

            # 높은 리스크-보상 비율 = 공격적 전략
            if risk_reward_ratio > 2.5:
                # 공격적 전략은 RSI 중앙값 방향으로 강화
                if rsi_midpoint < 50:
                    buy_score += 0.2
                else:
                    sell_score += 0.2

            # 7. MFI (Money Flow Index) - RSI와 유사하지만 거래량 고려
            mfi_min = strategy.get('mfi_min', 20.0)
            mfi_max = strategy.get('mfi_max', 80.0)

            mfi_midpoint = (mfi_min + mfi_max) / 2.0

            if mfi_midpoint < 50:
                buy_score += (50 - mfi_midpoint) / 100.0  # 0.0 ~ 0.5
            elif mfi_midpoint > 50:
                sell_score += (mfi_midpoint - 50) / 100.0  # 0.0 ~ 0.5

            # 8. 성과 데이터 기반 분류 (있는 경우)
            performance = strategy.get('performance_metrics', {})
            if isinstance(performance, str):
                import json
                performance = json.loads(performance) if performance else {}

            buy_win_rate = performance.get('buy_win_rate', 0.5)
            sell_win_rate = performance.get('sell_win_rate', 0.5)

            # 승률 차이가 크면 그 방향으로 분류
            if buy_win_rate > sell_win_rate + 0.1:
                buy_score += (buy_win_rate - sell_win_rate) * 0.5
            elif sell_win_rate > buy_win_rate + 0.1:
                sell_score += (sell_win_rate - buy_win_rate) * 0.5

            # 9. ADX (추세 강도) - 높을수록 추세 전략
            adx_min = strategy.get('adx_min', 15.0)

            if adx_min > 25:  # 강한 추세 필요
                # RSI 방향으로 추가 점수
                if rsi_midpoint < 50:
                    buy_score += (adx_min - 25) / 50.0 * 0.3
                else:
                    sell_score += (adx_min - 25) / 50.0 * 0.3

            # 10. 최종 분류 (임계값 완화: 0.2 → 0.05)
            # 약간의 차이만 있어도 분류되도록 변경
            score_diff = abs(buy_score - sell_score)

            if buy_score > sell_score and score_diff > 0.05:
                return 'buy'
            elif sell_score > buy_score and score_diff > 0.05:
                return 'sell'
            else:
                # 점수가 비슷하면 RSI 중앙값으로 최종 결정
                if rsi_midpoint < 48:  # 48 이하면 매수
                    return 'buy'
                elif rsi_midpoint > 52:  # 52 이상이면 매도
                    return 'sell'
                else:
                    return 'neutral'  # 정말 중립적인 경우만

        except Exception as e:
            logger.debug(f"전략 방향 분류 실패 (무시): {e}")
            # 에러 시 기본 분류 시도
            try:
                rsi_midpoint = (strategy.get('rsi_min', 30.0) + strategy.get('rsi_max', 70.0)) / 2.0
                if rsi_midpoint < 48:
                    return 'buy'
                elif rsi_midpoint > 52:
                    return 'sell'
            except:
                pass
            return 'neutral'

'''

print("=" * 80)
print("전략 방향 분류 함수 교체")
print("=" * 80)
print()

# 파일 읽기
file_path = '/workspace/rl_pipeline/analysis/integrated_analyzer.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"✅ 파일 읽기 완료: {len(content)} bytes")

# 함수 찾기 및 교체
pattern = r'(    def _classify_strategy_direction\(self.*?)(    def _calculate_interval_strategy_score)'
match = re.search(pattern, content, re.DOTALL)

if match:
    print(f"✅ _classify_strategy_direction 함수 찾음")

    # 교체
    new_content = content[:match.start(1)] + NEW_FUNCTION + match.group(2) + content[match.end(2):]

    # 파일 쓰기
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ 함수 교체 완료")
    print(f"   이전 함수 길이: {len(match.group(1))} bytes")
    print(f"   새 함수 길이: {len(NEW_FUNCTION)} bytes")
    print()
else:
    print("❌ 함수를 찾을 수 없음")
    exit(1)

print("=" * 80)
print("✅ 교체 완료!")
print("=" * 80)
