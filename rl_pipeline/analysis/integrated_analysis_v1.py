#!/usr/bin/env python
"""
통합 분석 v1 - 재설계된 계층적 구조

설계 원칙:
1. Layer 1 (방향): 1d × 0.6 + 240m × 0.4 → LONG/SHORT/NEUTRAL
2. Layer 2 (타이밍): 30m × 0.6 + 15m × 0.4 → NOW/WAIT/EXIT
3. Layer 3 (리스크): confidence → size
4. Grade 필터링: D/F 등급 제외, S=1.0, A=0.8, B=0.5, C=0.3 가중치
5. 시간 감쇠: 최근 데이터에 더 높은 가중치 (half_life=14일)
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math

# Grade 가중치 맵
GRADE_WEIGHTS = {
    'S': 1.0,
    'A': 0.8,
    'B': 0.5,
    'C': 0.3,
    'D': 0.0,  # 필터링됨
    'F': 0.0   # 필터링됨
}

# Layer 1 가중치 (방향 결정)
DIRECTION_WEIGHTS = {
    '1d': 0.6,
    '240m': 0.4
}

# Layer 2 가중치 (타이밍 결정)
TIMING_WEIGHTS = {
    '30m': 0.6,
    '15m': 0.4
}

# 시간 감쇠 파라미터
TIME_DECAY_HALF_LIFE_DAYS = 14


class IntegratedAnalyzerV1:
    """
    재설계된 통합 분석기

    출력 형식:
    {
        'direction': 'LONG' | 'SHORT' | 'NEUTRAL',
        'timing': 'NOW' | 'WAIT' | 'EXIT',
        'size': 0.0 ~ 1.0,
        'confidence': 0.0 ~ 1.0,
        'horizon': '15m' | '30m' | '240m' | '1d',
        'reason': {...}
    }
    """

    def __init__(self, db_path: str = '/workspace/data_storage/rl_strategies.db'):
        self.db_path = db_path

    def analyze(self, coin: str) -> Dict:
        """
        전체 통합 분석 실행

        Args:
            coin: 코인 심볼 (예: 'LINK')

        Returns:
            통합 분석 결과 딕셔너리
        """
        # 각 인터벌별 전략 데이터 로드
        interval_data = self._load_interval_data(coin)

        if not interval_data:
            return self._neutral_signal("데이터 없음")

        # Layer 1: 방향 결정
        direction, direction_strength, direction_reason = self._determine_direction(interval_data)

        # Layer 2: 타이밍 결정
        timing, timing_confidence, timing_reason = self._determine_timing(interval_data)

        # Layer 3: 리스크/크기 결정
        confidence = self._calculate_confidence(direction_strength, timing_confidence, interval_data)
        size = self._calculate_position_size(confidence, direction_strength)
        horizon = self._determine_horizon(direction, timing, interval_data)

        # 종합 이유
        reason = {
            'direction': direction_reason,
            'timing': timing_reason,
            'interval_scores': {k: v['weighted_score'] for k, v in interval_data.items() if v},
            'divergence': self._detect_divergence(interval_data)
        }

        return {
            'direction': direction,
            'timing': timing,
            'size': round(size, 3),
            'confidence': round(confidence, 3),
            'horizon': horizon,
            'reason': reason
        }

    def _load_interval_data(self, coin: str) -> Dict[str, Dict]:
        """
        각 인터벌별 전략 데이터 로드 및 전처리

        Returns:
            {
                '15m': {'strategies': [...], 'weighted_score': 0.0},
                '30m': {...},
                '240m': {...},
                '1d': {...}
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        intervals = ['15m', '30m', '240m', '1d']
        result = {}

        for interval in intervals:
            # 전략 및 등급 데이터 조인
            cursor.execute("""
                SELECT
                    sg.strategy_id,
                    sg.grade,
                    sg.predictive_accuracy,
                    rsr.avg_ret,
                    rsr.win_rate,
                    rsr.avg_sharpe_ratio,
                    rsr.avg_dd,
                    rsr.avg_reward,
                    rsr.avg_profit_factor,
                    rsr.last_updated
                FROM strategy_grades sg
                JOIN rl_strategy_rollup rsr ON sg.strategy_id = rsr.strategy_id
                WHERE sg.coin = ? AND sg.interval = ?
            """, (coin, interval))

            rows = cursor.fetchall()

            if not rows:
                result[interval] = None
                continue

            # 전략 리스트 생성
            strategies = []
            for row in rows:
                (sid, grade, pred_acc, avg_ret, win_rate, sharpe, avg_dd,
                 avg_reward, profit_factor, last_updated) = row

                strategies.append({
                    'strategy_id': sid,
                    'grade': grade,
                    'predictive_accuracy': pred_acc,
                    'avg_ret': avg_ret,
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe if sharpe else 0.0,
                    'avg_dd': avg_dd if avg_dd else 0.0,
                    'avg_reward': avg_reward if avg_reward else 0.0,
                    'profit_factor': profit_factor if profit_factor else 0.0,
                    'last_updated': last_updated
                })

            # 등급 필터링 및 가중치 계산
            filtered_strategies = self._filter_and_weight_strategies(strategies)

            # 인터벌 종합 점수 계산
            weighted_score = self._calculate_interval_score(filtered_strategies)

            result[interval] = {
                'strategies': filtered_strategies,
                'weighted_score': weighted_score,
                'total_count': len(strategies),
                'filtered_count': len(filtered_strategies)
            }

        conn.close()
        return result

    def _filter_and_weight_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """
        전략 필터링 및 가중치 적용

        1. D/F 등급 제외
        2. Grade 가중치 적용 (S=1.0, A=0.8, B=0.5, C=0.3)
        3. 시간 감쇠 적용 (half_life=14일)
        """
        filtered = []
        now = datetime.now()

        for s in strategies:
            grade = s['grade']

            # D/F 등급 필터링
            if grade in ['D', 'F']:
                continue

            # Grade 가중치
            grade_weight = GRADE_WEIGHTS.get(grade, 0.0)
            if grade_weight == 0.0:
                continue

            # 시간 감쇠 가중치
            if s['last_updated']:
                try:
                    last_updated = datetime.fromisoformat(s['last_updated'])
                    days_ago = (now - last_updated).days
                    time_weight = math.exp(-days_ago * math.log(2) / TIME_DECAY_HALF_LIFE_DAYS)
                except:
                    time_weight = 1.0
            else:
                time_weight = 1.0

            # 종합 가중치
            total_weight = grade_weight * time_weight

            s['grade_weight'] = grade_weight
            s['time_weight'] = time_weight
            s['total_weight'] = total_weight

            filtered.append(s)

        return filtered

    def _calculate_interval_score(self, strategies: List[Dict]) -> float:
        """
        인터벌 종합 점수 계산

        가중 평균: Σ(avg_ret × total_weight) / Σ(total_weight)
        """
        if not strategies:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for s in strategies:
            avg_ret = s['avg_ret'] if s['avg_ret'] else 0.0
            weight = s['total_weight']

            weighted_sum += avg_ret * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return weighted_sum / weight_sum

    # ==================== Layer 1: 방향 결정 ====================

    def _determine_direction(self, interval_data: Dict[str, Dict]) -> Tuple[str, float, Dict]:
        """
        Layer 1: 방향 결정

        1d × 0.6 + 240m × 0.4 → LONG/SHORT/NEUTRAL

        Returns:
            (direction, strength, reason)
            - direction: 'LONG' | 'SHORT' | 'NEUTRAL'
            - strength: 0.0 ~ 1.0 (방향 확신도)
            - reason: {'1d_score': ..., '240m_score': ..., 'weighted_score': ...}
        """
        score_1d = 0.0
        score_240m = 0.0

        # 1d 점수
        if interval_data.get('1d') and interval_data['1d']:
            score_1d = interval_data['1d']['weighted_score']

        # 240m 점수
        if interval_data.get('240m') and interval_data['240m']:
            score_240m = interval_data['240m']['weighted_score']

        # 가중 평균
        weighted_score = (score_1d * DIRECTION_WEIGHTS['1d'] +
                         score_240m * DIRECTION_WEIGHTS['240m'])

        # 방향 결정 임계값
        if weighted_score > 0.02:  # +2% 이상
            direction = 'LONG'
            strength = min(abs(weighted_score) / 0.10, 1.0)  # 10%면 strength=1.0
        elif weighted_score < -0.02:  # -2% 이하
            direction = 'SHORT'
            strength = min(abs(weighted_score) / 0.10, 1.0)
        else:
            direction = 'NEUTRAL'
            strength = 0.0

        reason = {
            '1d_score': round(score_1d, 4),
            '240m_score': round(score_240m, 4),
            'weighted_score': round(weighted_score, 4),
            'threshold': 0.02
        }

        return direction, strength, reason

    # ==================== Layer 2: 타이밍 결정 ====================

    def _determine_timing(self, interval_data: Dict[str, Dict]) -> Tuple[str, float, Dict]:
        """
        Layer 2: 타이밍 결정

        30m × 0.6 + 15m × 0.4 → NOW/WAIT/EXIT

        Returns:
            (timing, confidence, reason)
            - timing: 'NOW' | 'WAIT' | 'EXIT'
            - confidence: 0.0 ~ 1.0
            - reason: {...}
        """
        score_30m = 0.0
        score_15m = 0.0

        # 30m 점수
        if interval_data.get('30m') and interval_data['30m']:
            score_30m = interval_data['30m']['weighted_score']

        # 15m 점수
        if interval_data.get('15m') and interval_data['15m']:
            score_15m = interval_data['15m']['weighted_score']

        # 가중 평균
        weighted_score = (score_30m * TIMING_WEIGHTS['30m'] +
                         score_15m * TIMING_WEIGHTS['15m'])

        # 타이밍 결정
        if weighted_score > 0.005:  # +0.5% 이상
            timing = 'NOW'
            confidence = min(abs(weighted_score) / 0.02, 1.0)  # 2%면 confidence=1.0
        elif weighted_score < -0.005:  # -0.5% 이하
            timing = 'EXIT'
            confidence = min(abs(weighted_score) / 0.02, 1.0)
        else:
            timing = 'WAIT'
            confidence = 0.5  # 중립

        reason = {
            '30m_score': round(score_30m, 4),
            '15m_score': round(score_15m, 4),
            'weighted_score': round(weighted_score, 4),
            'threshold': 0.005
        }

        return timing, confidence, reason

    # ==================== Layer 3: 리스크/크기 결정 ====================

    def _calculate_confidence(self, direction_strength: float, timing_confidence: float,
                             interval_data: Dict[str, Dict]) -> float:
        """
        종합 확신도 계산

        방향 확신도와 타이밍 확신도의 조합
        """
        # 인터벌 일치도 체크
        convergence = self._check_convergence(interval_data)

        # 종합 확신도
        confidence = (direction_strength * 0.5 +
                     timing_confidence * 0.3 +
                     convergence * 0.2)

        return min(confidence, 1.0)

    def _calculate_position_size(self, confidence: float, direction_strength: float) -> float:
        """
        포지션 크기 계산

        confidence × direction_strength → size (0.0 ~ 1.0)
        """
        # 기본 크기
        base_size = confidence * direction_strength

        # 최소/최대 제한
        if base_size < 0.1:
            return 0.0  # 너무 작으면 거래 안 함

        return min(base_size, 1.0)

    def _determine_horizon(self, direction: str, timing: str,
                          interval_data: Dict[str, Dict]) -> str:
        """
        거래 기간 결정

        방향과 타이밍 조합에 따라 최적 인터벌 선택
        """
        # LONG/SHORT 방향이면서 NOW 타이밍
        if direction in ['LONG', 'SHORT'] and timing == 'NOW':
            # 가장 강한 신호를 보이는 인터벌 선택
            best_interval = '15m'
            best_score = 0.0

            for interval in ['15m', '30m', '240m', '1d']:
                if interval_data.get(interval) and interval_data[interval]:
                    score = abs(interval_data[interval]['weighted_score'])
                    if score > best_score:
                        best_score = score
                        best_interval = interval

            return best_interval

        # WAIT or NEUTRAL
        return '15m'  # 기본값

    # ==================== 유틸리티 ====================

    def _check_convergence(self, interval_data: Dict[str, Dict]) -> float:
        """
        인터벌 수렴도 체크

        모든 인터벌이 같은 방향이면 1.0, 분산되어 있으면 0.0
        """
        scores = []
        for interval in ['15m', '30m', '240m', '1d']:
            if interval_data.get(interval) and interval_data[interval]:
                scores.append(interval_data[interval]['weighted_score'])

        if len(scores) < 2:
            return 0.5

        # 모두 양수 또는 모두 음수면 수렴
        all_positive = all(s > 0 for s in scores)
        all_negative = all(s < 0 for s in scores)

        if all_positive or all_negative:
            return 1.0

        # 분산 계산
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # 분산이 작을수록 수렴도 높음
        convergence = 1.0 / (1.0 + variance * 100)

        return convergence

    def _detect_divergence(self, interval_data: Dict[str, Dict]) -> Dict:
        """
        인터벌 발산 감지

        장기/단기 인터벌 간 방향 불일치
        """
        long_term_score = 0.0
        short_term_score = 0.0

        # 장기 (1d, 240m)
        if interval_data.get('1d') and interval_data['1d']:
            long_term_score += interval_data['1d']['weighted_score'] * 0.6
        if interval_data.get('240m') and interval_data['240m']:
            long_term_score += interval_data['240m']['weighted_score'] * 0.4

        # 단기 (30m, 15m)
        if interval_data.get('30m') and interval_data['30m']:
            short_term_score += interval_data['30m']['weighted_score'] * 0.6
        if interval_data.get('15m') and interval_data['15m']:
            short_term_score += interval_data['15m']['weighted_score'] * 0.4

        # 방향 불일치 감지
        divergent = (long_term_score > 0 and short_term_score < 0) or \
                   (long_term_score < 0 and short_term_score > 0)

        return {
            'is_divergent': divergent,
            'long_term_score': round(long_term_score, 4),
            'short_term_score': round(short_term_score, 4)
        }

    def _neutral_signal(self, reason: str) -> Dict:
        """중립 신호 반환"""
        return {
            'direction': 'NEUTRAL',
            'timing': 'WAIT',
            'size': 0.0,
            'confidence': 0.0,
            'horizon': '15m',
            'reason': {'error': reason}
        }


# ==================== 편의 함수 ====================

def analyze_coin(coin: str, db_path: str = '/workspace/data_storage/rl_strategies.db') -> Dict:
    """
    코인 통합 분석 실행

    Args:
        coin: 코인 심볼
        db_path: DB 경로

    Returns:
        통합 분석 결과
    """
    analyzer = IntegratedAnalyzerV1(db_path)
    return analyzer.analyze(coin)


if __name__ == '__main__':
    # 테스트
    result = analyze_coin('LINK')

    print("=" * 70)
    print("통합 분석 v1 결과")
    print("=" * 70)
    print(f"방향:     {result['direction']}")
    print(f"타이밍:   {result['timing']}")
    print(f"크기:     {result['size']:.3f}")
    print(f"확신도:   {result['confidence']:.3f}")
    print(f"기간:     {result['horizon']}")
    print()
    print("이유:")
    import json
    print(json.dumps(result['reason'], indent=2, ensure_ascii=False))
