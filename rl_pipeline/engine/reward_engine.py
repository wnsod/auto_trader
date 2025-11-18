"""
보상 엔진 (Reward Engine)
예측 정확도 기반 보상 계산 시스템
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from rl_pipeline.engine.interval_profile import (
    get_interval_profile,
    get_risk_gamma,
    get_sigma_min
)

logger = logging.getLogger(__name__)

# 동시 히트 정책 상수
TIE_RULE = "SL_FIRST"  # TP/SL 동시 발생 시 SL 우선 (보수적 정책)

# 기본 보상 가중치
DEFAULT_WEIGHTS = {
    'dir': 0.35,      # 방향 정확도 (가장 중요)
    'price': 0.25,    # 목표 달성 (근접도)
    'time': 0.15,     # 시간 정확도
    'trade': 0.15,    # 거래 성과
    'calib': 0.10     # 캘리브레이션
}

# 시간 감쇠 람다 (기본값)
DEFAULT_LAMBDA = 0.7


@dataclass
class RewardComponents:
    """보상 구성 요소"""
    reward_dir: float = 0.0      # 방향 정확도 보상
    reward_price: float = 0.0    # 목표 달성 보상
    reward_time: float = 0.0     # 시간 정확도 보상
    reward_trade: float = 0.0    # 거래 성과 보상
    reward_calib: float = 0.0    # 캘리브레이션 보상
    reward_risk: float = 0.0     # 리스크 페널티
    reward_total: float = 0.0    # 총 보상


class RewardEngine:
    """보상 엔진 - 예측 정확도 기반 보상 계산"""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        lambda_time: float = DEFAULT_LAMBDA
    ):
        """
        초기화
        
        Args:
            weights: 보상 가중치 (None이면 기본값 사용)
            lambda_time: 시간 감쇠 람다 값
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.lambda_time = lambda_time
        
        logger.info(f"✅ Reward Engine 초기화 완료 (lambda={lambda_time})")
    
    def compute_reward(
        self,
        predicted_dir: int,
        predicted_target: float,
        predicted_horizon: int,
        actual_dir: int,
        actual_move_pct: float,
        actual_horizon: int,
        first_event: str,
        dd_pct_norm: float = 0.0,
        interval: str = "15m",
        sigma: Optional[float] = None,
        atr_pct: Optional[float] = None,
        tier_reward_weight: float = 1.0  # 🔥 3단계 보상 가중치 (기본값 1.0)
    ) -> RewardComponents:
        """
        보상 계산
        
        Args:
            predicted_dir: 예측 방향 (+1/-1/0)
            predicted_target: 예측 목표 변동률
            predicted_horizon: 예측 목표 캔들 수
            actual_dir: 실제 방향 (+1/-1/0)
            actual_move_pct: 실제 변동률
            actual_horizon: 실제 도달 캔들 수
            first_event: 첫 이벤트 ('TP', 'SL', 'expiry')
            dd_pct_norm: 정규화된 드로우다운
            interval: 인터벌
            sigma: 근접도 계산용 시그마 (None이면 interval_profile에서 조회)
            atr_pct: ATR 비율 (None이면 sigma만 사용)
        
        Returns:
            RewardComponents 객체
        """
        try:
            # 1. 방향 정확도 보상
            R_dir = self._compute_direction_reward(
                predicted_dir, actual_dir, predicted_target, actual_move_pct
            )
            
            # 2. 목표 달성 보상 (근접도)
            if sigma is None:
                sigma = self._get_sigma(interval, atr_pct)
            R_price = self._compute_price_reward(
                predicted_target, actual_move_pct, sigma
            )
            
            # 3. 시간 정확도 보상
            R_time = self._compute_time_reward(
                predicted_horizon, actual_horizon
            )
            
            # 4. 거래 성과 보상
            R_trade = self._compute_trade_reward(
                first_event, actual_move_pct, predicted_dir
            )
            
            # 5. 캘리브레이션 보상
            R_calib = self._compute_calibration_reward(
                predicted_dir, actual_dir, actual_move_pct
            )
            
            # 6. 리스크 페널티
            risk_gamma = get_risk_gamma(interval)
            R_risk = -risk_gamma * dd_pct_norm
            
            # 7. 총 보상 계산
            reward_total = (
                self.weights['dir'] * R_dir +
                self.weights['price'] * R_price +
                self.weights['time'] * R_time +
                self.weights['trade'] * R_trade +
                self.weights['calib'] * R_calib +
                R_risk
            )
            
            # 🔥 3단계 보상 가중치 적용
            reward_total = reward_total * tier_reward_weight
            
            # 🔥 추가 개선: Profit Factor 기반 보상 보너스 (재학습 권장 반영)
            # 실제 거래 성과가 좋을 때 추가 보상 부여
            if first_event == 'TP':
                # TP 도달 시 PF 기반 보너스 (PF > 1.0이면 보너스, < 1.0이면 페널티)
                # 실제 거래에서 PF가 높을수록 더 많은 보상
                pf_bonus = min(0.3, actual_move_pct * 10) if actual_move_pct > 0 else 0.0
                reward_total += pf_bonus
            elif first_event == 'SL':
                # SL 도달 시 PF 기반 페널티 (더 큰 손실)
                pf_penalty = max(-0.3, actual_move_pct * 10) if actual_move_pct < 0 else 0.0
                reward_total += pf_penalty
            
            return RewardComponents(
                reward_dir=R_dir,
                reward_price=R_price,
                reward_time=R_time,
                reward_trade=R_trade,
                reward_calib=R_calib,
                reward_risk=R_risk,
                reward_total=reward_total
            )
            
        except Exception as e:
            logger.error(f"❌ 보상 계산 실패: {e}")
            # 에러 시 기본값 반환
            return RewardComponents()
    
    def _compute_direction_reward(
        self,
        predicted_dir: int,
        actual_dir: int,
        predicted_target: float,
        actual_move_pct: float
    ) -> float:
        """
        방향 정확도 보상
        
        Returns:
            0.0 ~ 1.0 (정확할수록 높음)
        """
        # 방향 일치 체크
        if predicted_dir == +1:  # 상승 예측
            if actual_move_pct > 0:
                return 1.0
            else:
                return 0.0
        elif predicted_dir == -1:  # 하락 예측
            if actual_move_pct < 0:
                return 1.0
            else:
                return 0.0
        else:  # predicted_dir == 0 (횡보 예측)
            # 횡보 예측은 작은 변동률일 때 보상
            threshold = abs(predicted_target) * 0.5
            if abs(actual_move_pct) < threshold:
                return 0.5  # 부분 보상
            else:
                return 0.0
    
    def _compute_price_reward(
        self,
        predicted_target: float,
        actual_move_pct: float,
        sigma: float
    ) -> float:
        """
        목표 달성 보상 (근접도)
        
        가우시안 근접도: exp(-(error/σ)²)
        
        Args:
            predicted_target: 예측 목표 변동률
            actual_move_pct: 실제 변동률
            sigma: 근접도 계산용 시그마
        
        Returns:
            0.0 ~ 1.0 (가까울수록 높음)
        """
        error = abs(actual_move_pct - predicted_target)
        
        # 가우시안 근접도
        try:
            prox = np.exp(-(error / sigma) ** 2)
            return float(prox)
        except (OverflowError, ZeroDivisionError):
            # 에러 처리
            if error == 0:
                return 1.0
            else:
                return 0.0
    
    def _compute_time_reward(
        self,
        predicted_horizon: int,
        actual_horizon: int
    ) -> float:
        """
        시간 정확도 보상
        
        time_bonus = exp(-λ × time_error)
        time_error = |t_hit - horizon_k| / horizon_k
        
        Args:
            predicted_horizon: 예측 목표 캔들 수
            actual_horizon: 실제 도달 캔들 수
        
        Returns:
            0.0 ~ 1.0 (정확할수록 높음)
        """
        if predicted_horizon <= 0:
            return 0.0
        
        time_error = abs(actual_horizon - predicted_horizon) / predicted_horizon
        
        try:
            time_bonus = np.exp(-self.lambda_time * time_error)
            return float(time_bonus)
        except (OverflowError, ZeroDivisionError):
            if actual_horizon == predicted_horizon:
                return 1.0
            else:
                return 0.0
    
    def _compute_trade_reward(
        self,
        first_event: str,
        actual_move_pct: float,
        predicted_dir: int
    ) -> float:
        """
        거래 성과 보상
        
        Args:
            first_event: 첫 이벤트 ('TP', 'SL', 'expiry')
            actual_move_pct: 실제 변동률
            predicted_dir: 예측 방향
        
        Returns:
            0.0 ~ 1.0 (반대 방향 시 음수 가능)
        """
        if first_event == 'TP':
            # TP 도달 = 성공
            return 1.0
        elif first_event == 'SL':
            # SL 도달 = 실패 (동시 히트 시 SL 우선: TIE_RULE)
            return 0.0
        elif first_event == 'expiry':
            # 🔥 만료 시 개선: 가격 변화가 0%여도 학습 가능하도록
            # 실제 변동률이 매우 작은 경우(0.001% 이하)도 처리
            min_move_threshold = 0.00001  # 0.001% 미만은 0으로 간주
            
            if predicted_dir == +1 and actual_move_pct > min_move_threshold:
                # 상승 예측, 실제 상승 (부분 성공) - 변동률 크기에 따라 보상 차등
                move_magnitude = min(abs(actual_move_pct) * 100, 1.0)  # 1%까지 정규화
                return 0.3 + (move_magnitude * 0.2)  # 0.3 ~ 0.5 (변동률이 클수록 높은 보상)
            elif predicted_dir == -1 and actual_move_pct < -min_move_threshold:
                # 하락 예측, 실제 하락 (부분 성공)
                move_magnitude = min(abs(actual_move_pct) * 100, 1.0)
                return 0.3 + (move_magnitude * 0.2)  # 0.3 ~ 0.5
            elif predicted_dir == +1 and actual_move_pct < -min_move_threshold:
                # 🔥 상승 예측했는데 실제 하락 (명확한 실패, 페널티)
                opposite_move = abs(actual_move_pct)
                penalty = -min(opposite_move * 20, 0.5)  # 반대 방향 페널티
                return penalty
            elif predicted_dir == -1 and actual_move_pct > min_move_threshold:
                # 🔥 하락 예측했는데 실제 상승 (명확한 실패, 페널티)
                opposite_move = abs(actual_move_pct)
                penalty = -min(opposite_move * 20, 0.5)
                return penalty
            elif abs(actual_move_pct) <= min_move_threshold:
                # 🔥 횡보 (가격 변화 거의 없음) - 방향성 예측에 부분 점수 부여
                # 변동성이 낮은 시장에서도 학습 가능하도록 최소 보상 제공
                if predicted_dir != 0:  # 방향 예측이 있었으면
                    # 방향이 맞다면(시장이 움직이지 않았지만 예측 방향이 맞았다면) 작은 보상
                    # 실제로는 움직이지 않았으므로 매우 작은 보상 (0.1)
                    return 0.1
                else:
                    # 중립 예측이었고 실제로 횡보였으면 중간 보상
                    return 0.25
            else:
                # 기타 (매우 작은 변동)
                return 0.05  # 최소 보상 (학습 가능하도록)
        else:
            # 기타 이벤트
            return 0.0
    
    def _compute_calibration_reward(
        self,
        predicted_dir: int,
        actual_dir: int,
        actual_move_pct: float
    ) -> float:
        """
        캘리브레이션 보상
        
        예측 확신도와 실제 결과의 일치도
        
        Args:
            predicted_dir: 예측 방향
            actual_dir: 실제 방향
            actual_move_pct: 실제 변동률
        
        Returns:
            0.0 ~ 1.0
        """
        # 방향 일치도
        dir_match = 1.0 if predicted_dir == actual_dir else 0.0
        
        # 변동률 크기 고려 (큰 변동일수록 높은 보상)
        magnitude = min(abs(actual_move_pct) * 10, 1.0)  # 0.1% = 1.0
        
        return (dir_match + magnitude) / 2.0
    
    def _get_sigma(self, interval: str, atr_pct: Optional[float]) -> float:
        """
        근접도 계산용 시그마 조회
        
        sigma_min과 ATR% 중 큰 값 사용
        """
        sigma_min = get_sigma_min(interval)
        
        if atr_pct is not None:
            return max(sigma_min, atr_pct)
        else:
            return sigma_min
    
    def compute_predictive_accuracy_flag(
        self,
        first_event: str,
        predicted_dir: int,
        actual_move_pct: float
    ) -> int:
        """
        예측 정확도 플래그 계산
        
        Returns:
            1 (정확) / 0 (부정확)
        """
        if first_event == 'TP':
            # TP 도달 = 예측 성공 (상승/하락/중립 모두)
            return 1
        elif first_event == 'SL':
            # SL 도달 = 예측 실패
            return 0
        elif first_event == 'expiry':
            # 🔥 만료 시 개선: 가격 변화가 0%여도 학습 가능하도록
            min_move_threshold = 0.00001  # 0.001% 미만은 0으로 간주
            
            if predicted_dir == +1:
                # 상승 예측: 실제 변동률이 양수면 성공
                if actual_move_pct > min_move_threshold:
                    return 1  # 상승 성공
                elif actual_move_pct < -min_move_threshold:
                    return 0  # 하락 실패
                else:
                    # 횡보 (거의 변화 없음) - 방향성 예측에 부분 점수
                    # 변동성이 낮은 시장에서도 학습 가능하도록
                    return 0  # 0% 변화는 정확도 0 (하지만 보상은 0.1 제공)
            elif predicted_dir == -1:
                # 하락 예측: 실제 변동률이 음수면 성공
                if actual_move_pct < -min_move_threshold:
                    return 1  # 하락 성공
                elif actual_move_pct > min_move_threshold:
                    return 0  # 상승 실패
                else:
                    # 횡보
                    return 0  # 0% 변화는 정확도 0 (하지만 보상은 0.1 제공)
            elif predicted_dir == 0:
                # 🔥 중립 예측: 작은 변동 범위 내에 있으면 성공
                # 중립 범위: ±0.5% 이내 (일반적인 노이즈 범위)
                neutral_threshold = 0.005  # 0.5%
                return 1 if abs(actual_move_pct) <= neutral_threshold else 0
            else:
                return 0
        else:
            return 0


# 편의 함수
def compute_reward(
    predicted_dir: int,
    predicted_target: float,
    predicted_horizon: int,
    actual_dir: int,
    actual_move_pct: float,
    actual_horizon: int,
    first_event: str,
    interval: str = "15m",
    **kwargs
) -> RewardComponents:
    """보상 계산 편의 함수"""
    engine = RewardEngine()
    return engine.compute_reward(
        predicted_dir=predicted_dir,
        predicted_target=predicted_target,
        predicted_horizon=predicted_horizon,
        actual_dir=actual_dir,
        actual_move_pct=actual_move_pct,
        actual_horizon=actual_horizon,
        first_event=first_event,
        interval=interval,
        **kwargs
    )


if __name__ == "__main__":
    # 테스트
    engine = RewardEngine()
    
    # 테스트 케이스 1: TP 도달 (완전 성공)
    reward1 = engine.compute_reward(
        predicted_dir=+1,
        predicted_target=0.015,
        predicted_horizon=8,
        actual_dir=+1,
        actual_move_pct=0.015,
        actual_horizon=5,
        first_event='TP',
        interval='15m'
    )
    print(f"테스트 1 (TP 도달): reward_total={reward1.reward_total:.3f}")
    
    # 테스트 케이스 2: SL 도달 (실패)
    reward2 = engine.compute_reward(
        predicted_dir=+1,
        predicted_target=0.015,
        predicted_horizon=8,
        actual_dir=-1,
        actual_move_pct=-0.02,
        actual_horizon=3,
        first_event='SL',
        interval='15m'
    )
    print(f"테스트 2 (SL 도달): reward_total={reward2.reward_total:.3f}")

