"""
Thompson Sampling 공통 모듈

가상매매와 실전매매에서 공통으로 사용하는 Thompson Sampling 관련 유틸리티
"""
import os
import sys

# 경로 설정
_current_dir = os.path.dirname(os.path.abspath(__file__))
_trade_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_trade_dir)

if _trade_dir not in sys.path:
    sys.path.append(_trade_dir)
if _project_root not in sys.path:
    sys.path.append(_project_root)

from typing import Tuple, Optional, Any
from dataclasses import dataclass

# ThompsonSamplingLearner 임포트
try:
    from trade.virtual_trade_learner import ThompsonSamplingLearner
    THOMPSON_AVAILABLE = True
except ImportError:
    ThompsonSamplingLearner = None
    THOMPSON_AVAILABLE = False
    print("⚠️ ThompsonSamplingLearner 로드 실패")


@dataclass
class ThompsonScore:
    """Thompson Sampling 결과"""
    score: float           # 샘플링된 승률 (0.0 ~ 1.0)
    total_samples: int     # 총 샘플 수
    pattern: str           # 패턴 문자열
    is_new_pattern: bool   # 신규 패턴 여부


class ThompsonScoreCalculator:
    """Thompson Sampling 점수 계산기 (싱글톤)"""
    
    _instance = None
    _sampler = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._init_sampler()
    
    def _init_sampler(self):
        """Thompson Sampler 초기화"""
        if not THOMPSON_AVAILABLE:
            self._sampler = None
            return
        
        try:
            # DB 경로 설정
            _DEFAULT_DB_DIR = os.path.join(_project_root, 'market', 'coin_market', 'data_storage')
            _env_strategy_base = os.getenv('STRATEGY_DB_PATH')
            _default_strategy_base = os.path.join(_DEFAULT_DB_DIR, 'learning_strategies')
            
            if _env_strategy_base and (_env_strategy_base.startswith('/workspace') or _env_strategy_base.startswith('\\workspace')):
                if os.name == 'nt':
                    strategy_base = _default_strategy_base
                else:
                    strategy_base = _env_strategy_base
            else:
                strategy_base = _env_strategy_base or _default_strategy_base
            
            if os.path.isdir(strategy_base) or not strategy_base.endswith('.db'):
                strategy_db_path = os.path.join(strategy_base, 'common_strategies.db')
            else:
                strategy_db_path = strategy_base
            
            self._sampler = ThompsonSamplingLearner(db_path=strategy_db_path)
            print(f"✅ Thompson Sampler 초기화 완료 (DB: {strategy_db_path})")
            
        except Exception as e:
            print(f"⚠️ Thompson Sampler 초기화 오류: {e}")
            self._sampler = None
    
    @property
    def sampler(self):
        """Thompson Sampler 인스턴스 반환"""
        return self._sampler
    
    def get_score(self, signal: Any) -> ThompsonScore:
        """시그널에서 Thompson 점수 조회
        
        Args:
            signal: SignalInfo 객체 또는 패턴 추출 가능한 객체
            
        Returns:
            ThompsonScore: 점수 정보
        """
        try:
            if self._sampler is None:
                return ThompsonScore(score=0.0, total_samples=0, pattern='unknown', is_new_pattern=True)
            
            # 패턴 추출
            pattern = self.extract_pattern(signal)
            
            # Thompson Sampling에서 확률 샘플링
            # 🔧 sample_success_rate는 (float, str)을 반환 - 문자열은 신뢰도 메시지
            result = self._sampler.sample_success_rate(pattern)
            
            # 결과 파싱
            if isinstance(result, tuple):
                sampled_rate = result[0]
                confidence_msg = result[1] if len(result) > 1 else ""
            else:
                sampled_rate = float(result)
                confidence_msg = ""
            
            # 신규 패턴 여부 (신뢰도 메시지에서 추출 또는 기본값 사용)
            is_new_pattern = "새 패턴" in confidence_msg or "데이터 부족" in confidence_msg
            
            # 총 샘플 수 추출 시도 (메시지에서)
            total_samples = 0
            if "회" in confidence_msg:
                try:
                    import re
                    match = re.search(r'(\d+)회', confidence_msg)
                    if match:
                        total_samples = int(match.group(1))
                except:
                    pass
            
            return ThompsonScore(
                score=sampled_rate,
                total_samples=total_samples,
                pattern=pattern,
                is_new_pattern=is_new_pattern
            )
            
        except Exception as e:
            print(f"⚠️ Thompson 점수 조회 오류: {e}")
            return ThompsonScore(score=0.0, total_samples=0, pattern='unknown', is_new_pattern=True)
    
    def get_score_from_pattern(self, pattern: str) -> ThompsonScore:
        """패턴 문자열에서 직접 Thompson 점수 조회
        
        Args:
            pattern: 패턴 문자열
            
        Returns:
            ThompsonScore: 점수 정보
        """
        try:
            if self._sampler is None:
                return ThompsonScore(score=0.0, total_samples=0, pattern=pattern, is_new_pattern=True)
            
            # Thompson Sampling에서 확률 샘플링
            # 🔧 sample_success_rate는 (float, str)을 반환
            result = self._sampler.sample_success_rate(pattern)
            
            if isinstance(result, tuple):
                sampled_rate = result[0]
                confidence_msg = result[1] if len(result) > 1 else ""
            else:
                sampled_rate = float(result)
                confidence_msg = ""
            
            # 신규 패턴 여부
            is_new_pattern = "새 패턴" in confidence_msg or "데이터 부족" in confidence_msg
            
            # 총 샘플 수 추출
            total_samples = 0
            if "회" in confidence_msg:
                try:
                    import re
                    match = re.search(r'(\d+)회', confidence_msg)
                    if match:
                        total_samples = int(match.group(1))
                except:
                    pass
            
            return ThompsonScore(
                score=sampled_rate,
                total_samples=total_samples,
                pattern=pattern,
                is_new_pattern=is_new_pattern
            )
            
        except Exception as e:
            print(f"⚠️ Thompson 점수 조회 오류: {e}")
            return ThompsonScore(score=0.0, total_samples=0, pattern=pattern, is_new_pattern=True)
    
    def extract_pattern(self, signal: Any) -> str:
        """시그널에서 패턴 추출
        
        Args:
            signal: SignalInfo 객체 또는 유사 객체
            
        Returns:
            str: 패턴 문자열
        """
        try:
            # SignalInfo 객체에서 속성 추출
            action = getattr(signal, 'action', None)
            if hasattr(action, 'value'):
                action = action.value
            
            rsi_raw = getattr(signal, 'rsi', 50.0)
            volume_ratio_raw = getattr(signal, 'volume_ratio', 1.0)
            wave_phase = getattr(signal, 'wave_phase', 'unknown')
            pattern_type = getattr(signal, 'pattern_type', 'none')
            
            # 🔧 타입 안전 변환 (문자열/None 처리)
            try:
                rsi = float(rsi_raw) if rsi_raw is not None else 50.0
            except (ValueError, TypeError):
                rsi = 50.0
            
            try:
                volume_ratio = float(volume_ratio_raw) if volume_ratio_raw is not None else 1.0
            except (ValueError, TypeError):
                volume_ratio = 1.0
            
            # RSI 구간 분류
            if rsi < 30:
                rsi_zone = 'oversold'
            elif rsi > 70:
                rsi_zone = 'overbought'
            else:
                rsi_zone = 'neutral'
            
            # 볼륨 구간 분류
            if volume_ratio > 2.0:
                vol_zone = 'high_vol'
            elif volume_ratio < 0.5:
                vol_zone = 'low_vol'
            else:
                vol_zone = 'normal_vol'
            
            # 패턴 생성
            pattern = f"{action}_{rsi_zone}_{vol_zone}_{wave_phase}_{pattern_type}"
            return pattern
            
        except Exception as e:
            print(f"⚠️ 패턴 추출 오류: {e}")
            return 'unknown_pattern'
    
    def should_execute(self, signal: Any, signal_score: float = 0.0) -> Tuple[bool, float, str]:
        """매매 실행 여부 결정
        
        Args:
            signal: SignalInfo 객체
            signal_score: 시그널 점수 (-1.0 ~ 1.0)
            
        Returns:
            Tuple[bool, float, str]: (실행 여부, 최종 점수, 사유)
        """
        try:
            if self._sampler is None:
                return True, signal_score, "Thompson Sampler 없음 - 기본 승인"
            
            # Thompson 점수 조회
            thompson_result = self.get_score(signal)
            
            # 🔧 시그널 점수 정규화 (-1~+1 → 0~1)
            normalized_signal_score = (signal_score + 1.0) / 2.0
            
            # 가중치 (탐색 단계에서는 시그널 점수 비중 증가)
            if thompson_result.is_new_pattern:
                signal_weight = 0.7
                thompson_weight = 0.2
                exploration_bonus = 0.15
                threshold = 0.30
            else:
                signal_weight = 0.6
                thompson_weight = 0.3
                exploration_bonus = 0.05
                threshold = 0.40
            
            profit_weight = 1.0 - signal_weight - thompson_weight
            profit_bonus = 0.5  # 기본 보너스
            
            # 최종 점수 계산
            final_score = (
                normalized_signal_score * signal_weight +
                thompson_result.score * thompson_weight +
                profit_bonus * profit_weight +
                exploration_bonus
            )
            
            # 액션별 임계값 비교
            action = getattr(signal, 'action', None)
            if hasattr(action, 'value'):
                action = action.value
            
            if action == 'buy':
                should_execute = final_score >= threshold
            elif action == 'sell':
                should_execute = final_score >= (threshold - 0.1)  # 매도는 더 관대하게
            else:
                should_execute = True  # hold 등은 항상 허용
            
            reason = f"Thompson: {thompson_result.score:.2f} (샘플 {thompson_result.total_samples}개)"
            if thompson_result.is_new_pattern:
                reason += " [신규패턴]"
            
            return should_execute, final_score, reason
            
        except Exception as e:
            print(f"⚠️ Thompson 실행 결정 오류: {e}")
            return True, signal_score, f"오류로 기본 승인: {e}"


# 싱글톤 인스턴스
_calculator = None

def get_thompson_calculator() -> ThompsonScoreCalculator:
    """Thompson 점수 계산기 싱글톤 인스턴스 반환"""
    global _calculator
    if _calculator is None:
        _calculator = ThompsonScoreCalculator()
    return _calculator


def get_thompson_score(signal: Any) -> float:
    """시그널에서 Thompson 점수 조회 (간편 함수)
    
    Args:
        signal: SignalInfo 객체
        
    Returns:
        float: Thompson 점수 (0.0 ~ 1.0)
    """
    calculator = get_thompson_calculator()
    result = calculator.get_score(signal)
    return result.score


def get_thompson_score_from_pattern(pattern: str) -> float:
    """패턴에서 Thompson 점수 조회 (간편 함수)
    
    Args:
        pattern: 패턴 문자열
        
    Returns:
        float: Thompson 점수 (0.0 ~ 1.0)
    """
    calculator = get_thompson_calculator()
    result = calculator.get_score_from_pattern(pattern)
    return result.score


def should_execute_trade(signal: Any, signal_score: float = 0.0) -> Tuple[bool, float, str]:
    """매매 실행 여부 결정 (간편 함수)
    
    Args:
        signal: SignalInfo 객체
        signal_score: 시그널 점수 (-1.0 ~ 1.0)
        
    Returns:
        Tuple[bool, float, str]: (실행 여부, 최종 점수, 사유)
    """
    calculator = get_thompson_calculator()
    return calculator.should_execute(signal, signal_score)


def extract_signal_pattern(signal: Any) -> str:
    """시그널에서 패턴 추출 (간편 함수)
    
    Args:
        signal: SignalInfo 객체
        
    Returns:
        str: 패턴 문자열
    """
    calculator = get_thompson_calculator()
    return calculator.extract_pattern(signal)

