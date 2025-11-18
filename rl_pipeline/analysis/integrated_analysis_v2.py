#!/usr/bin/env python
"""
통합 분석 v2 - 학습 가능 파라미터 (⚠️ 폐기됨 - 과적합 확인)

⚠️⚠️⚠️ 중요: 이 파일은 실험 실패로 폐기되었습니다 ⚠️⚠️⚠️

============================================================================
폐기 이유: Walk-Forward Analysis에서 과적합 확인
============================================================================

날짜: 2025-11-16
결과: v2 파라미터 학습이 v1보다 나쁜 성능을 보임

**Walk-Forward Analysis 결과:**
┌──────────────┬──────────┬──────────┬────────────┐
│ 데이터셋     │ v1 점수  │ v2 점수  │ Test/Train │
├──────────────┼──────────┼──────────┼────────────┤
│ Train (70%)  │ 0.560    │ 0.580    │ -          │
│ Test (30%)   │ 0.505    │ 0.503    │ -          │
│ 안정성       │ 90.2%    │ 86.7%    │ v2 과적합! │
└──────────────┴──────────┴──────────┴────────────┘

**핵심 문제:**
1. 과적합: Test/Train 비율 86.7% (기준 90% 미만)
2. 파라미터 불안정: 데이터에 따라 최적값이 극단적으로 변함
   - 전체 데이터: 1d=95%, 15m=97.7%
   - Train 데이터: 240m=99.75%, 30m=99.75% (정반대!)
3. Test 성능: v1(0.505) > v2(0.503) → v1이 실전에서 더 좋음

**실패 원인 분석:**
1. 파라미터 공간이 너무 넓음 (14차원)
2. 데이터 크기 부족 (LINK 586개 전략만)
3. 극단값 방지 제약 부족 (99.4% 포지션 같은 극단적 결과)
4. 금융 시장의 비정상성(non-stationary) → 과거 최적이 미래 최적 아님

**교훈:**
✅ 단순한 휴리스틱(v1)이 복잡한 학습(v2)보다 나을 수 있다
✅ 도메인 지식 기반 고정 파라미터가 더 로버스트하다
✅ 데이터가 적을 때는 학습 기반 접근법 위험하다
✅ Train 성능만 보지 말고 반드시 Test 검증 필요

**대안:**
→ v1(고정 파라미터) 사용 권장
→ 필요시 v1 파라미터 수동 미세 조정
→ 학습 재시도하려면: 더 많은 데이터 수집 + 강한 제약 추가

**관련 문서:**
- INTEGRATED_ANALYSIS_V2_FINAL_REPORT.md (전체 실험 결과)
- walk_forward_analysis_result.json (검증 데이터)

============================================================================

원래 의도: v1과 동일한 로직이지만, 모든 파라미터를 외부에서 주입 가능
학습 알고리즘 (Bayesian Optimization)이 파라미터를 최적화

⚠️ 이 코드는 참고용으로만 보관됨. 실제 사용하지 말 것!
"""

import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
import json


class V2Parameters:
    """
    v2 학습 가능 파라미터

    Raw parameters (제약 없는 공간):
        - direction_logits: [2] → softmax → [1d, 240m] weights
        - timing_logits: [2] → softmax → [30m, 15m] weights
        - grade_logits: [3] → softmax → [A, B, C] weights (S=1.0 고정)
        - log_half_life: scalar → exp → days
        - log_direction_threshold: scalar → exp → threshold
        - log_timing_threshold: scalar → exp → threshold
        - logit_min_position: scalar → sigmoid → min_position
        - confidence_logits: [3] → softmax → [direction, timing, convergence] weights

    Total: 14 raw parameters
    """

    def __init__(self, raw_params: Optional[List[float]] = None):
        """
        Args:
            raw_params: 14개 raw 파라미터 리스트
                        None이면 v1 기본값으로 초기화
        """
        if raw_params is None:
            # v1 기본값 (역변환)
            # direction_weights: [0.6, 0.4] → softmax^-1
            self.direction_logits = [np.log(1.5), 0.0]  # → [0.6, 0.4]

            # timing_weights: [0.6, 0.4] → softmax^-1
            self.timing_logits = [np.log(1.5), 0.0]  # → [0.6, 0.4]

            # grade_weights: [0.8, 0.5, 0.3] normalized → [0.5, 0.3125, 0.1875]
            # softmax^-1([0.5, 0.3125, 0.1875])
            self.grade_logits = [np.log(2.667), np.log(1.667), 0.0]  # → [0.5, 0.3125, 0.1875]

            # time decay
            self.log_half_life = np.log(14)  # → 14

            # thresholds
            self.log_direction_threshold = np.log(0.02)  # → 0.02
            self.log_timing_threshold = np.log(0.005)  # → 0.005

            # min position
            self.logit_min_position = np.log(0.1 / (1 - 0.1))  # → 0.1

            # confidence_weights: [0.5, 0.3, 0.2] → softmax^-1
            self.confidence_logits = [np.log(2.5), np.log(1.5), 0.0]  # → [0.5, 0.3, 0.2]
        else:
            # Raw 파라미터로부터 초기화
            assert len(raw_params) == 14, f"Expected 14 params, got {len(raw_params)}"
            self.direction_logits = raw_params[0:2]
            self.timing_logits = raw_params[2:4]
            self.grade_logits = raw_params[4:7]
            self.log_half_life = raw_params[7]
            self.log_direction_threshold = raw_params[8]
            self.log_timing_threshold = raw_params[9]
            self.logit_min_position = raw_params[10]
            self.confidence_logits = raw_params[11:14]

    def to_raw(self) -> List[float]:
        """Raw 파라미터 리스트로 변환"""
        return (
            self.direction_logits +
            self.timing_logits +
            self.grade_logits +
            [self.log_half_life] +
            [self.log_direction_threshold] +
            [self.log_timing_threshold] +
            [self.logit_min_position] +
            self.confidence_logits
        )

    def transform(self) -> Dict:
        """
        Raw parameters → Actual parameters

        모든 제약 조건이 자동으로 만족됨:
        - Softmax: 합 = 1.0
        - Exp: > 0
        - Sigmoid: 0 < x < 1
        """
        direction_weights = self._softmax(self.direction_logits)
        timing_weights = self._softmax(self.timing_logits)
        grade_weights_normalized = self._softmax(self.grade_logits)
        confidence_weights = self._softmax(self.confidence_logits)

        # Grade weights: softmax 후 v1 비율에 맞게 scale
        # v1: [0.8, 0.5, 0.3], normalized: [0.5, 0.3125, 0.1875]
        # scale factor = 0.8 / 0.5 = 1.6
        grade_scale = 1.6
        grade_A = grade_weights_normalized[0] * grade_scale
        grade_B = grade_weights_normalized[1] * grade_scale
        grade_C = grade_weights_normalized[2] * grade_scale

        return {
            'DIRECTION_WEIGHTS': {
                '1d': direction_weights[0],
                '240m': direction_weights[1]
            },
            'TIMING_WEIGHTS': {
                '30m': timing_weights[0],
                '15m': timing_weights[1]
            },
            'GRADE_WEIGHTS': {
                'S': 1.0,
                'A': grade_A,
                'B': grade_B,
                'C': grade_C,
                'D': 0.0,
                'F': 0.0
            },
            'TIME_DECAY_HALF_LIFE_DAYS': np.exp(self.log_half_life),
            'DIRECTION_THRESHOLD': np.exp(self.log_direction_threshold),
            'TIMING_THRESHOLD': np.exp(self.log_timing_threshold),
            'MIN_POSITION_SIZE': self._sigmoid(self.logit_min_position),
            'CONFIDENCE_WEIGHTS': {
                'direction': confidence_weights[0],
                'timing': confidence_weights[1],
                'convergence': confidence_weights[2]
            }
        }

    @staticmethod
    def _softmax(logits: List[float]) -> np.ndarray:
        """Softmax transformation"""
        logits = np.array(logits)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / exp_logits.sum()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid transformation"""
        return 1.0 / (1.0 + np.exp(-x))

    def __repr__(self):
        params = self.transform()
        return f"V2Parameters(\n" + \
               f"  direction_weights={params['DIRECTION_WEIGHTS']},\n" + \
               f"  timing_weights={params['TIMING_WEIGHTS']},\n" + \
               f"  grade_weights={{S:{params['GRADE_WEIGHTS']['S']}, A:{params['GRADE_WEIGHTS']['A']:.3f}, B:{params['GRADE_WEIGHTS']['B']:.3f}, C:{params['GRADE_WEIGHTS']['C']:.3f}}},\n" + \
               f"  half_life={params['TIME_DECAY_HALF_LIFE_DAYS']:.1f},\n" + \
               f"  direction_threshold={params['DIRECTION_THRESHOLD']:.4f},\n" + \
               f"  timing_threshold={params['TIMING_THRESHOLD']:.4f},\n" + \
               f"  min_position={params['MIN_POSITION_SIZE']:.3f},\n" + \
               f"  confidence_weights={params['CONFIDENCE_WEIGHTS']}\n" + \
               ")"


class IntegratedAnalyzerV2:
    """
    통합 분석 v2 - 파라미터 주입 가능

    v1과 동일한 로직이지만, V2Parameters를 받아서 사용
    """

    def __init__(self, params: Optional[V2Parameters] = None,
                 db_path: str = '/workspace/data_storage/rl_strategies.db'):
        """
        Args:
            params: V2Parameters 객체 (None이면 v1 기본값)
            db_path: DB 경로
        """
        self.db_path = db_path

        # 파라미터 설정
        if params is None:
            params = V2Parameters()  # v1 기본값
        self.params = params.transform()

    def analyze(self, coin: str) -> Dict:
        """
        전체 통합 분석 실행 (v1과 동일한 인터페이스)

        Args:
            coin: 코인 심볼

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
        """각 인터벌별 전략 데이터 로드 (v1과 동일)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        intervals = ['15m', '30m', '240m', '1d']
        result = {}

        for interval in intervals:
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

            filtered_strategies = self._filter_and_weight_strategies(strategies)
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
        """전략 필터링 및 가중치 적용 (v2 파라미터 사용)"""
        filtered = []
        now = datetime.now()

        for s in strategies:
            grade = s['grade']

            # D/F 등급 필터링
            if grade in ['D', 'F']:
                continue

            # Grade 가중치 (v2 파라미터)
            grade_weight = self.params['GRADE_WEIGHTS'].get(grade, 0.0)
            if grade_weight == 0.0:
                continue

            # 시간 감쇠 가중치 (v2 파라미터)
            if s['last_updated']:
                try:
                    last_updated = datetime.fromisoformat(s['last_updated'])
                    days_ago = (now - last_updated).days
                    half_life = self.params['TIME_DECAY_HALF_LIFE_DAYS']
                    time_weight = math.exp(-days_ago * math.log(2) / half_life)
                except:
                    time_weight = 1.0
            else:
                time_weight = 1.0

            total_weight = grade_weight * time_weight

            s['grade_weight'] = grade_weight
            s['time_weight'] = time_weight
            s['total_weight'] = total_weight

            filtered.append(s)

        return filtered

    def _calculate_interval_score(self, strategies: List[Dict]) -> float:
        """인터벌 종합 점수 계산 (v1과 동일)"""
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

    def _determine_direction(self, interval_data: Dict[str, Dict]) -> Tuple[str, float, Dict]:
        """Layer 1: 방향 결정 (v2 파라미터 사용)"""
        score_1d = 0.0
        score_240m = 0.0

        if interval_data.get('1d') and interval_data['1d']:
            score_1d = interval_data['1d']['weighted_score']

        if interval_data.get('240m') and interval_data['240m']:
            score_240m = interval_data['240m']['weighted_score']

        # v2 파라미터 사용
        weights = self.params['DIRECTION_WEIGHTS']
        weighted_score = (score_1d * weights['1d'] +
                         score_240m * weights['240m'])

        # v2 임계값 사용
        threshold = self.params['DIRECTION_THRESHOLD']

        if weighted_score > threshold:
            direction = 'LONG'
            strength = min(abs(weighted_score) / 0.10, 1.0)
        elif weighted_score < -threshold:
            direction = 'SHORT'
            strength = min(abs(weighted_score) / 0.10, 1.0)
        else:
            direction = 'NEUTRAL'
            strength = 0.0

        reason = {
            '1d_score': round(score_1d, 4),
            '240m_score': round(score_240m, 4),
            'weighted_score': round(weighted_score, 4),
            'threshold': threshold
        }

        return direction, strength, reason

    def _determine_timing(self, interval_data: Dict[str, Dict]) -> Tuple[str, float, Dict]:
        """Layer 2: 타이밍 결정 (v2 파라미터 사용)"""
        score_30m = 0.0
        score_15m = 0.0

        if interval_data.get('30m') and interval_data['30m']:
            score_30m = interval_data['30m']['weighted_score']

        if interval_data.get('15m') and interval_data['15m']:
            score_15m = interval_data['15m']['weighted_score']

        # v2 파라미터 사용
        weights = self.params['TIMING_WEIGHTS']
        weighted_score = (score_30m * weights['30m'] +
                         score_15m * weights['15m'])

        # v2 임계값 사용
        threshold = self.params['TIMING_THRESHOLD']

        if weighted_score > threshold:
            timing = 'NOW'
            confidence = min(abs(weighted_score) / 0.02, 1.0)
        elif weighted_score < -threshold:
            timing = 'EXIT'
            confidence = min(abs(weighted_score) / 0.02, 1.0)
        else:
            timing = 'WAIT'
            confidence = 0.5

        reason = {
            '30m_score': round(score_30m, 4),
            '15m_score': round(score_15m, 4),
            'weighted_score': round(weighted_score, 4),
            'threshold': threshold
        }

        return timing, confidence, reason

    def _calculate_confidence(self, direction_strength: float, timing_confidence: float,
                             interval_data: Dict[str, Dict]) -> float:
        """종합 확신도 계산 (v2 파라미터 사용)"""
        convergence = self._check_convergence(interval_data)

        # v2 파라미터 사용
        weights = self.params['CONFIDENCE_WEIGHTS']
        confidence = (direction_strength * weights['direction'] +
                     timing_confidence * weights['timing'] +
                     convergence * weights['convergence'])

        return min(confidence, 1.0)

    def _calculate_position_size(self, confidence: float, direction_strength: float) -> float:
        """포지션 크기 계산 (v2 파라미터 사용)"""
        base_size = confidence * direction_strength

        # v2 최소 포지션 크기 사용
        min_size = self.params['MIN_POSITION_SIZE']

        if base_size < min_size:
            return 0.0

        return min(base_size, 1.0)

    def _determine_horizon(self, direction: str, timing: str,
                          interval_data: Dict[str, Dict]) -> str:
        """거래 기간 결정 (v1과 동일)"""
        if direction in ['LONG', 'SHORT'] and timing == 'NOW':
            best_interval = '15m'
            best_score = 0.0

            for interval in ['15m', '30m', '240m', '1d']:
                if interval_data.get(interval) and interval_data[interval]:
                    score = abs(interval_data[interval]['weighted_score'])
                    if score > best_score:
                        best_score = score
                        best_interval = interval

            return best_interval

        return '15m'

    def _check_convergence(self, interval_data: Dict[str, Dict]) -> float:
        """인터벌 수렴도 체크 (v1과 동일)"""
        scores = []
        for interval in ['15m', '30m', '240m', '1d']:
            if interval_data.get(interval) and interval_data[interval]:
                scores.append(interval_data[interval]['weighted_score'])

        if len(scores) < 2:
            return 0.5

        all_positive = all(s > 0 for s in scores)
        all_negative = all(s < 0 for s in scores)

        if all_positive or all_negative:
            return 1.0

        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        convergence = 1.0 / (1.0 + variance * 100)

        return convergence

    def _detect_divergence(self, interval_data: Dict[str, Dict]) -> Dict:
        """인터벌 발산 감지 (v2 파라미터 사용)"""
        long_term_score = 0.0
        short_term_score = 0.0

        # v2 파라미터 사용
        direction_weights = self.params['DIRECTION_WEIGHTS']
        timing_weights = self.params['TIMING_WEIGHTS']

        if interval_data.get('1d') and interval_data['1d']:
            long_term_score += interval_data['1d']['weighted_score'] * direction_weights['1d']
        if interval_data.get('240m') and interval_data['240m']:
            long_term_score += interval_data['240m']['weighted_score'] * direction_weights['240m']

        if interval_data.get('30m') and interval_data['30m']:
            short_term_score += interval_data['30m']['weighted_score'] * timing_weights['30m']
        if interval_data.get('15m') and interval_data['15m']:
            short_term_score += interval_data['15m']['weighted_score'] * timing_weights['15m']

        divergent = (long_term_score > 0 and short_term_score < 0) or \
                   (long_term_score < 0 and short_term_score > 0)

        return {
            'is_divergent': bool(divergent),  # JSON serializable
            'long_term_score': round(float(long_term_score), 4),
            'short_term_score': round(float(short_term_score), 4)
        }

    def _neutral_signal(self, reason: str) -> Dict:
        """중립 신호 반환 (v1과 동일)"""
        return {
            'direction': 'NEUTRAL',
            'timing': 'WAIT',
            'size': 0.0,
            'confidence': 0.0,
            'horizon': '15m',
            'reason': {'error': reason}
        }


# ==================== 편의 함수 ====================

def analyze_coin_v2(coin: str, params: Optional[V2Parameters] = None,
                   db_path: str = '/workspace/data_storage/rl_strategies.db') -> Dict:
    """
    코인 통합 분석 v2 실행

    Args:
        coin: 코인 심볼
        params: V2Parameters (None이면 v1 기본값)
        db_path: DB 경로

    Returns:
        통합 분석 결과
    """
    analyzer = IntegratedAnalyzerV2(params, db_path)
    return analyzer.analyze(coin)


if __name__ == '__main__':
    # 테스트: v1 기본값으로 실행
    print("=" * 70)
    print("v2 테스트 (v1 기본 파라미터)")
    print("=" * 70)
    print()

    # v1 기본 파라미터 생성
    params_v1 = V2Parameters()
    print("파라미터:")
    print(params_v1)
    print()

    # 분석 실행
    result = analyze_coin_v2('LINK', params_v1)

    print("결과:")
    print(f"  방향:     {result['direction']}")
    print(f"  타이밍:   {result['timing']}")
    print(f"  크기:     {result['size']:.3f}")
    print(f"  확신도:   {result['confidence']:.3f}")
    print(f"  기간:     {result['horizon']}")
    print()
    print("이유:")
    print(json.dumps(result['reason'], indent=2, ensure_ascii=False))
