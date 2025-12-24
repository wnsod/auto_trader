#!/usr/bin/env python
"""
통합 분석 모듈 (Integrated Analyzer)

설계 원칙:
1. Layer 1 (방향): 1d × 35% + 240m × 30% → LONG/SHORT/NEUTRAL (interval_profiles 가중치 사용)
2. Layer 2 (타이밍): 30m × 20% + 15m × 15% → NOW/WAIT/EXIT (interval_profiles 가중치 사용)
3. Layer 3 (리스크): confidence → size
4. Grade 필터링: D/F 등급 제외, S=1.0, A=0.8, B=0.5, C=0.3 가중치
5. 시간 감쇠: 최근 데이터에 더 높은 가중치 (half_life=14일)

참고: 이 모듈은 이전의 integrated_analysis_v1.py를 대체하며 단일 진실 공급원(SSOT) 역할을 합니다.
v2(학습 가능 파라미터)는 과적합 이슈로 폐기되었습니다.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

# 🔥 interval_profiles import
try:
    from rl_pipeline.core.interval_profiles import (
        get_integration_weights,
        get_interval_role,
        INTERVAL_PROFILES
    )
    INTERVAL_PROFILES_AVAILABLE = True
except ImportError:
    logger.warning("interval_profiles 모듈을 찾을 수 없습니다. 기본 가중치 사용")
    INTERVAL_PROFILES_AVAILABLE = False
    get_integration_weights = None
    get_interval_role = None
    INTERVAL_PROFILES = None

# Grade 가중치 맵
GRADE_WEIGHTS = {
    'S': 1.0,
    'A': 0.8,
    'B': 0.5,
    'C': 0.3,
    'D': 0.0,  # 필터링됨
    'F': 0.0   # 필터링됨
}

# 🔥 기본 가중치 (interval_profiles 없을 때 폴백)
DEFAULT_WEIGHTS = {
    '1d': 0.35,
    '240m': 0.30,
    '30m': 0.20,
    '15m': 0.15
}

# 시간 감쇠 파라미터
TIME_DECAY_HALF_LIFE_DAYS = 14


class IntegratedAnalyzer:
    """
    통합 분석기 (구 IntegratedAnalyzerV1)
    
    출력 형식:
    {
        'direction': 'LONG' | 'SHORT' | 'NEUTRAL',
        'timing': 'NOW' | 'WAIT' | 'EXIT',
        'size': 0.0 ~ 1.0,
        'confidence': 0.0 ~ 1.0,
        'direction_strength': 0.0 ~ 1.0, # 방향성 강도
        'timing_confidence': 0.0 ~ 1.0,  # 타이밍 확신도
        'horizon': '15m' | '30m' | '240m' | '1d',
        'reason': {...}
    }
    """

    def __init__(self, db_path: str = None, session_id: Optional[str] = None):
        import os
        if db_path is None:
            # 환경변수 우선
            db_path = os.getenv('STRATEGY_DB_PATH') or os.getenv('STRATEGIES_DB_PATH')
            
            # 환경변수도 없으면 DATA_STORAGE_PATH 기반 추론
            if not db_path:
                data_storage = os.getenv('DATA_STORAGE_PATH')
                if data_storage:
                    db_path = os.path.join(data_storage, 'learning_strategies.db')
                else:
                    # 최후의 수단: 에러 발생 (하드코딩 제거됨)
                    raise ValueError("❌ STRATEGY_DB_PATH 또는 DATA_STORAGE_PATH 환경변수가 필요합니다.")
            
        self.db_path = db_path
        self.session_id = session_id

        # 🔥 interval_profiles 가중치 로드
        self.interval_weights = self._load_interval_weights()

        # 🔥 방향 레이어 가중치 (1d + 240m)
        if '1d' in self.interval_weights and '240m' in self.interval_weights:
            total_dir = self.interval_weights['1d'] + self.interval_weights['240m']
            if total_dir > 0:
                self.direction_weights = {
                    '1d': self.interval_weights['1d'] / total_dir,
                    '240m': self.interval_weights['240m'] / total_dir
                }
            else:
                self.direction_weights = {'1d': 0.5, '240m': 0.5}
        else:
             # Fallback if keys missing (though default weights have them)
             self.direction_weights = {'1d': 0.6, '240m': 0.4}

        # 🔥 타이밍 레이어 가중치 (30m + 15m)
        if '30m' in self.interval_weights and '15m' in self.interval_weights:
            total_time = self.interval_weights['30m'] + self.interval_weights['15m']
            if total_time > 0:
                self.timing_weights = {
                    '30m': self.interval_weights['30m'] / total_time,
                    '15m': self.interval_weights['15m'] / total_time
                }
            else:
                self.timing_weights = {'30m': 0.6, '15m': 0.4}
        else:
             self.timing_weights = {'30m': 0.6, '15m': 0.4}

        logger.info(f"🎯 통합 분석기 초기화 완료")
        
        # 가중치 소수점 포맷팅 (보기 좋게)
        fmt_dir_weights = {k: round(v, 3) for k, v in self.direction_weights.items()}
        fmt_time_weights = {k: round(v, 3) for k, v in self.timing_weights.items()}
        
        logger.info(f"   방향 가중치 (1d/240m): {fmt_dir_weights}")
        logger.info(f"   타이밍 가중치 (30m/15m): {fmt_time_weights}")

    def _load_interval_weights(self) -> Dict[str, float]:
        """interval_profiles에서 가중치 로드"""
        if INTERVAL_PROFILES_AVAILABLE and get_integration_weights:
            try:
                weights = get_integration_weights()
                if weights:
                    logger.info("✅ interval_profiles 가중치 사용")
                    
                    # 역할 정보 로깅
                    if get_interval_role:
                        for interval, weight in weights.items():
                            try:
                                role = get_interval_role(interval)
                                logger.info(f"   {interval}: {weight:.3f} ({role})")
                            except:
                                pass
                                
                    return weights
            except Exception as e:
                logger.warning(f"interval_profiles 가중치 로드 실패, 기본값 사용: {e}")
        
        logger.info("📊 기본 가중치 사용 (interval_profiles 없음)")
        return DEFAULT_WEIGHTS

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

        # 🔥 인터벌별 역할 정보 추가
        if INTERVAL_PROFILES_AVAILABLE:
            for interval in interval_data:
                if interval_data[interval] and get_interval_role:
                    try:
                        role = get_interval_role(interval)
                        interval_data[interval]['role'] = role
                        interval_data[interval]['weight'] = self.interval_weights.get(interval, 0)
                        
                        # 인터벌 프로필에서 목표 정보 가져오기
                        if INTERVAL_PROFILES and interval in INTERVAL_PROFILES:
                            profile = INTERVAL_PROFILES[interval]
                            interval_data[interval]['profile'] = {
                                'focus': profile.get('focus', ''),
                                'profit_threshold': profile.get('labeling', {}).get('profit_threshold', 0),
                                'target_horizon': profile.get('labeling', {}).get('target_horizon', 0)
                            }
                    except:
                        pass

        # Layer 1: 방향 결정 (1d + 240m)
        direction, direction_strength, direction_reason = self._determine_direction(interval_data)
        
        # Layer 2: 타이밍 결정 (30m + 15m)
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
            'interval_roles': {k: v.get('role', '') for k, v in interval_data.items() if v},
            'interval_weights': self.interval_weights,
            'divergence': self._detect_divergence(interval_data)
        }
        
        return {
            'direction': direction,
            'timing': timing,
            'size': round(size, 3),
            'confidence': round(confidence, 3),
            'direction_strength': round(direction_strength, 3), # 🔥 추가: 방향성 강도
            'timing_confidence': round(timing_confidence, 3),   # 🔥 추가: 타이밍 확신도
            'horizon': horizon,
            'reason': reason,
            'interval_profiles_used': INTERVAL_PROFILES_AVAILABLE
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
            # 🔥 스키마 변경: coin → symbol
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
                WHERE sg.symbol = ? AND sg.interval = ?
            """, (coin, interval))
            
            rows = cursor.fetchall()
            
            if not rows:
                result[interval] = None
                continue
                
            # 전략 리스트 생성
            strategies = []
            for row in rows:
                strategy = {
                    'strategy_id': row[0],
                    'grade': row[1],
                    'predictive_accuracy': row[2] or 0,
                    'avg_ret': row[3] or 0,
                    'win_rate': row[4] or 0,
                    'avg_sharpe_ratio': row[5] or 0,
                    'avg_dd': row[6] or 0,
                    'avg_reward': row[7] or 0,
                    'avg_profit_factor': row[8] or 0,
                    'last_updated': row[9]
                }
                strategies.append(strategy)
            
            # Grade 기반 필터링 (D, F 제외)
            filtered = [s for s in strategies if s['grade'] not in ['D', 'F']]
            
            if not filtered:
                result[interval] = None
                continue
                
            # 가중 점수 계산
            weighted_score = self._calculate_weighted_score(filtered)
            
            result[interval] = {
                'strategies': filtered,
                'weighted_score': weighted_score,
                'total_count': len(strategies),
                'filtered_count': len(filtered)
            }
            
        conn.close()
        return result

    def _determine_direction(self, interval_data: Dict) -> Tuple[str, float, Dict]:
        """
        Layer 1: 방향 결정 (1d와 240m 데이터 기반)
        """
        direction_scores = {}
        
        # 1d 데이터
        if interval_data.get('1d'):
            score = interval_data['1d']['weighted_score']
            direction_scores['1d'] = score * self.direction_weights['1d']
        else:
            direction_scores['1d'] = 0.5 * self.direction_weights['1d']
            
        # 240m 데이터
        if interval_data.get('240m'):
            score = interval_data['240m']['weighted_score']
            direction_scores['240m'] = score * self.direction_weights['240m']
        else:
            direction_scores['240m'] = 0.5 * self.direction_weights['240m']
            
        # 종합 점수
        total_score = sum(direction_scores.values())
        
        # 방향 결정
        if total_score > 0.6:
            direction = 'LONG'
        elif total_score < 0.4:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
            
        # 강도 계산 (0.5에서 얼마나 떨어져 있는가)
        strength = abs(total_score - 0.5) * 2
        
        reason = {
            'scores': direction_scores,
            'total': total_score,
            'weights': self.direction_weights,
            'interval_roles': {
                '1d': (interval_data.get('1d') or {}).get('role', 'Macro Regime'),
                '240m': (interval_data.get('240m') or {}).get('role', 'Trend Structure')
            }
        }
        
        return direction, strength, reason

    def _determine_timing(self, interval_data: Dict) -> Tuple[str, float, Dict]:
        """
        Layer 2: 타이밍 결정 (30m과 15m 데이터 기반)
        """
        timing_scores = {}
        
        # 30m 데이터
        if interval_data.get('30m'):
            score = interval_data['30m']['weighted_score']
            timing_scores['30m'] = score * self.timing_weights['30m']
        else:
            timing_scores['30m'] = 0.5 * self.timing_weights['30m']
            
        # 15m 데이터
        if interval_data.get('15m'):
            score = interval_data['15m']['weighted_score']
            timing_scores['15m'] = score * self.timing_weights['15m']
        else:
            timing_scores['15m'] = 0.5 * self.timing_weights['15m']
            
        # 종합 점수
        total_score = sum(timing_scores.values())
        
        # 타이밍 결정
        if total_score > 0.65:
            timing = 'NOW'
        elif total_score < 0.35:
            timing = 'EXIT'
        else:
            timing = 'WAIT'
            
        # 신뢰도 계산
        confidence = abs(total_score - 0.5) * 2
        
        reason = {
            'scores': timing_scores,
            'total': total_score,
            'weights': self.timing_weights,
            'interval_roles': {
                '30m': (interval_data.get('30m') or {}).get('role', 'Micro Trend'),
                '15m': (interval_data.get('15m') or {}).get('role', 'Execution')
            }
        }
        
        return timing, confidence, reason

    def _calculate_weighted_score(self, strategies: List[Dict]) -> float:
        """전략 리스트의 가중 점수 계산"""
        if not strategies:
            return 0.5
            
        total_weight = 0
        weighted_sum = 0
        
        for strategy in strategies:
            # Grade 가중치
            grade_weight = GRADE_WEIGHTS.get(strategy['grade'], 0)
            if grade_weight == 0:
                continue
                
            # 시간 감쇠 계산
            if strategy['last_updated']:
                last_updated = datetime.fromisoformat(strategy['last_updated'])
                days_old = (datetime.now() - last_updated).days
                time_weight = math.exp(-days_old / TIME_DECAY_HALF_LIFE_DAYS * math.log(2))
            else:
                time_weight = 0.5
                
            # 종합 가중치
            weight = grade_weight * time_weight
            
            # 점수 계산 (여러 지표 종합)
            # 🔥 수정된 등급 산정 로직 반영 (방향성/타이밍 중심)
            # Sharpe Ratio가 비정상적으로 클 경우 캡핑 (예: 10.0)
            sharpe = strategy['avg_sharpe_ratio']
            if sharpe > 10.0: sharpe = 10.0
            elif sharpe < -10.0: sharpe = -10.0
            
            score = (
                strategy['predictive_accuracy'] * 0.50 + # 방향성 (가장 중요)
                strategy['win_rate'] * 0.30 +            # 타이밍 (중요)
                (sharpe / 2 if sharpe > 0 else 0.5) * 0.10 +
                strategy['avg_reward'] * 0.10
            )
            
            weighted_sum += score * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.5
            
        return weighted_sum / total_weight

    def _calculate_confidence(self, direction_strength: float, timing_confidence: float, 
                             interval_data: Dict) -> float:
        """종합 신뢰도 계산"""
        # 기본 신뢰도
        base_confidence = (direction_strength + timing_confidence) / 2
        
        # 데이터 가용성 보너스
        available_intervals = sum(1 for v in interval_data.values() if v)
        availability_bonus = available_intervals / 4 * 0.1
        
        # 등급 분포 보너스 (S, A 등급이 많을수록)
        high_grade_ratio = 0
        total_strategies = 0
        for interval_info in interval_data.values():
            if interval_info and interval_info['strategies']:
                high_grade = sum(1 for s in interval_info['strategies'] 
                               if s['grade'] in ['S', 'A'])
                high_grade_ratio += high_grade
                total_strategies += len(interval_info['strategies'])
        
        if total_strategies > 0:
            grade_bonus = (high_grade_ratio / total_strategies) * 0.1
        else:
            grade_bonus = 0
            
        return min(1.0, base_confidence + availability_bonus + grade_bonus)

    def _calculate_position_size(self, confidence: float, direction_strength: float) -> float:
        """포지션 크기 계산"""
        # 기본 크기 = 신뢰도
        base_size = confidence
        
        # 방향 강도에 따른 조정
        size = base_size * (0.5 + direction_strength * 0.5)
        
        # 최소/최대 제한
        return max(0.1, min(1.0, size))

    def _determine_horizon(self, direction: str, timing: str, interval_data: Dict) -> str:
        """투자 시간대 결정"""
        if direction == 'NEUTRAL':
            return '15m'  # 중립일 때는 단기
            
        if timing == 'NOW':
            # 즉시 진입일 때는 단기 모니터링
            return '15m' if interval_data.get('15m') else '30m'
        elif timing == 'WAIT':
            # 대기일 때는 중기 모니터링
            return '30m' if interval_data.get('30m') else '240m'
        else:  # EXIT
            # 청산일 때는 즉시
            return '15m'

    def _detect_divergence(self, interval_data: Dict) -> Dict:
        """인터벌 간 다이버전스 감지"""
        scores = {}
        for interval, data in interval_data.items():
            if data:
                scores[interval] = data['weighted_score']
        
        if len(scores) < 2:
            return {'detected': False, 'message': '데이터 부족'}
            
        # 장기와 단기의 차이
        long_term = scores.get('1d', 0.5)
        short_term = scores.get('15m', 0.5)
        divergence = abs(long_term - short_term)
        
        if divergence > 0.3:
            return {
                'detected': True,
                'strength': divergence,
                'message': f"장기({long_term:.2f})와 단기({short_term:.2f}) 신호 불일치"
            }
            
        return {'detected': False, 'strength': divergence, 'message': '신호 일치'}

    def _neutral_signal(self, reason: str) -> Dict:
        """중립 신호 반환"""
        return {
            'direction': 'NEUTRAL',
            'timing': 'WAIT',
            'size': 0,
            'confidence': 0,
            'horizon': '240m',
            'reason': {'message': reason},
            'direction_strength': 0.5,  # 🔥 기본값 (중립)
            'timing_confidence': 0.5,   # 🔥 기본값 (중립)
            'interval_profiles_used': INTERVAL_PROFILES_AVAILABLE
        }

