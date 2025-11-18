"""
검증 결과 평가 및 자동 재학습 로직
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def evaluate_validation_results(
    eval_result: Dict[str, Any],
    strict_mode: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    검증 결과를 평가하여 합격/불합격 판단
    
    Args:
        eval_result: auto_evaluate_model()의 결과
        strict_mode: 엄격한 모드 (기본 False, True면 더 엄격한 기준)
    
    Returns:
        (합격 여부, 이유, 상세 정보) 튜플
    """
    if not eval_result:
        return False, "평가 결과 없음", {}
    
    issues = []
    warnings = []
    score = 0.0
    
    # 1. A/B 평가 체크
    ab_result = eval_result.get('comparison', {})
    if ab_result:
        hybrid_result = ab_result.get('hybrid', {})
        improvement = ab_result.get('improvement', {})
        
        pf = hybrid_result.get('profit_factor', 0.0)
        return_pct = hybrid_result.get('total_return', 0.0)
        win_rate = hybrid_result.get('win_rate', 0.0)
        mdd = hybrid_result.get('mdd', 0.0)
        trades = hybrid_result.get('trades', 0)
        
        # Profit Factor 체크
        if pf < 1.0:
            issues.append(f"Profit Factor가 1.0 미만 ({pf:.2f})")
            score -= 2.0
        elif pf < 1.2:
            warnings.append(f"Profit Factor가 낮음 ({pf:.2f}, 권장: >1.2)")
            score -= 0.5
        else:
            score += 1.0
        
        # Return 체크
        if return_pct < 0:
            issues.append(f"음수 수익률 ({return_pct:.2%})")
            score -= 2.0
        elif return_pct < 0.05:
            warnings.append(f"수익률이 낮음 ({return_pct:.2%}, 권장: >5%)")
            score -= 0.5
        else:
            score += 1.0
        
        # Win Rate 체크
        if win_rate < 0.4:
            issues.append(f"승률이 너무 낮음 ({win_rate:.1%})")
            score -= 1.0
        elif win_rate < 0.5:
            warnings.append(f"승률이 낮음 ({win_rate:.1%}, 권장: >50%)")
            score -= 0.3
        else:
            score += 0.5
        
        # Max Drawdown 체크
        if mdd > 0.3:
            issues.append(f"Max Drawdown이 너무 큼 ({mdd:.1%})")
            score -= 1.5
        elif mdd > 0.2:
            warnings.append(f"Max Drawdown이 큼 ({mdd:.1%}, 권장: <20%)")
            score -= 0.5
        
        # Trades 체크 (거래 0회는 Critical issue)
        if trades == 0:
            issues.append(f"거래 수가 0개 (모델이 액션을 생성하지 못함)")
            score -= 5.0  # 🔥 Critical issue: 거래 0회는 매우 심각한 문제
        elif trades < 5:
            issues.append(f"거래 수가 부족함 ({trades}개)")
            score -= 1.0
        elif trades < 10:
            warnings.append(f"거래 수가 적음 ({trades}개, 권장: >=10)")
            score -= 0.3
        
        # 규칙 대비 개선도 체크
        pf_improvement = improvement.get('profit_factor_improvement', 0.0)
        if pf_improvement < -0.2:  # 20% 이상 악화
            issues.append(f"규칙 대비 성능 악화 ({pf_improvement:.1%})")
            score -= 2.0
        elif pf_improvement < 0:
            warnings.append(f"규칙 대비 성능 개선 없음 ({pf_improvement:.1%})")
            score -= 0.5
    
    # 2. Walk-Forward 검증 체크
    wf_result = eval_result.get('walk_forward')
    if wf_result and wf_result.get('status') == 'success':
        has_overfitting = wf_result.get('has_overfitting', False)
        overfitting_ratio = wf_result.get('overfitting_ratio', 1.0)
        
        if has_overfitting:
            issues.append(f"과적합 가능성 (비율: {overfitting_ratio:.1%})")
            score -= 1.5
        else:
            score += 1.0
    elif wf_result:
        warnings.append(f"Walk-Forward 검증 건너뜀: {wf_result.get('reason', 'unknown')}")
    
    # 3. 다중 기간 검증 체크
    mp_result = eval_result.get('multi_period')
    if mp_result and mp_result.get('status') == 'success':
        consistency = mp_result.get('consistency', 0.0)
        regime_count = mp_result.get('regime_count', 0)
        
        if consistency < 0.5:
            issues.append(f"레짐별 성능 일관성 부족 ({consistency:.1%})")
            score -= 1.0
        elif consistency < 0.7:
            warnings.append(f"레짐별 성능 일관성 낮음 ({consistency:.1%}, 권장: >70%)")
            score -= 0.5
        else:
            score += 0.5
        
        if regime_count < 2:
            warnings.append(f"평가된 레짐 수가 적음 ({regime_count}개, 권장: >=2)")
            score -= 0.3
    elif mp_result:
        warnings.append(f"다중 기간 검증 건너뜀: {mp_result.get('reason', 'unknown')}")
    
    # 최종 판단
    if strict_mode:
        # 엄격 모드: issues가 하나라도 있으면 불합격
        passed = len(issues) == 0
    else:
        # 일반 모드: critical issues만 체크 (거래 0회 추가)
        critical_issues = [
            i for i in issues 
            if any(keyword in i for keyword in [
                'Profit Factor가 1.0 미만', 
                '음수 수익률', 
                '20% 이상 악화', 
                '과적합 가능성',
                '거래 수가 0개'  # 🔥 거래 0회는 Critical issue
            ])
        ]
        passed = len(critical_issues) == 0 and score >= -1.0
    
    # 이유 생성
    if passed:
        reason = "검증 합격"
        if warnings:
            reason += f" (경고: {len(warnings)}개)"
    else:
        reason = f"검증 불합격: {', '.join(issues[:3])}"  # 최대 3개만 표시
    
    details = {
        'score': score,
        'issues': issues,
        'warnings': warnings,
        'critical_count': len([i for i in issues if any(keyword in i for keyword in [
            'Profit Factor', '음수', '과적합', '거래 수가 0개'  # 🔥 거래 0회 포함
        ])])
    }
    
    return passed, reason, details


def should_retrain(
    eval_result: Dict[str, Any],
    previous_attempts: int = 0,
    max_attempts: int = 3
) -> Tuple[bool, str]:
    """
    재학습이 필요한지 판단
    
    Args:
        eval_result: 평가 결과
        previous_attempts: 이전 재시도 횟수
        max_attempts: 최대 재시도 횟수
    
    Returns:
        (재학습 필요 여부, 이유) 튜플
    """
    # 검증 결과 평가
    passed, reason, details = evaluate_validation_results(eval_result)
    
    if passed:
        return False, "검증 합격, 재학습 불필요"
    
    # 불합격 시 재학습 필요
    critical_count = details.get('critical_count', 0)
    issues = details.get('issues', [])
    
    # 🔥 거래 0회는 최대 시도 횟수와 무관하게 무조건 재학습
    has_zero_trades = any('거래 수가 0개' in issue for issue in issues)
    
    if has_zero_trades:
        # 거래 0회는 최대 시도 횟수 무시하고 재학습 (모델이 액션을 생성하지 못하는 심각한 문제)
        return True, f"거래 0회 감지 (모델이 액션을 생성하지 못함) - 최대 시도 횟수 무시하고 재학습"
    
    # 최대 재시도 횟수 체크 (거래 0회가 아닌 경우만)
    if previous_attempts >= max_attempts:
        return False, f"최대 재시도 횟수 초과 ({max_attempts}회)"
    
    if critical_count > 0:
        # Critical issue가 있으면 무조건 재학습
        return True, f"Critical issue {critical_count}개 발견: {reason}"
    else:
        # Warning만 있으면 선택적 재학습
        return True, f"성능 개선 필요: {reason}"


def get_retrain_suggestions(
    eval_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    재학습 시 개선 제안 생성
    
    Args:
        eval_result: 평가 결과
    
    Returns:
        개선 제안 딕셔너리
    """
    suggestions = {
        'adjust_hyperparameters': False,
        'adjust_learning_rate': False,
        'adjust_entropy_coef': False,
        'adjust_batch_size': False,
        'collect_more_data': False,
        'reason': ''
    }
    
    reasons = []
    
    # Walk-Forward 결과 분석
    wf_result = eval_result.get('walk_forward')
    if wf_result and wf_result.get('has_overfitting'):
        suggestions['adjust_learning_rate'] = True
        suggestions['adjust_entropy_coef'] = True
        reasons.append("과적합 감지 → 학습률 감소 및 탐험 증가 권장")
    
    # A/B 평가 결과 분석
    ab_result = eval_result.get('comparison', {})
    if ab_result:
        hybrid_result = ab_result.get('hybrid', {})
        pf = hybrid_result.get('profit_factor', 0.0)
        trades = hybrid_result.get('trades', 0)
        
        # 🔥 거래 0회는 가장 심각한 문제
        if trades == 0:
            suggestions['adjust_learning_rate'] = True
            suggestions['adjust_entropy_coef'] = True
            suggestions['adjust_hyperparameters'] = True
            suggestions['collect_more_data'] = True
            reasons.append("거래 0회 → 모델이 액션을 생성하지 못함: 학습률 조정, 탐험 증가(entropy_coef), 하이퍼파라미터 조정 필요")
        elif pf < 1.0:
            suggestions['adjust_hyperparameters'] = True
            suggestions['collect_more_data'] = True
            reasons.append("Profit Factor < 1.0 → 하이퍼파라미터 조정 및 더 많은 학습 데이터 필요")
    
    # 다중 기간 검증 결과 분석
    mp_result = eval_result.get('multi_period')
    if mp_result:
        consistency = mp_result.get('consistency', 0.0)
        
        if consistency < 0.5:
            suggestions['collect_more_data'] = True
            reasons.append("레짐 일관성 부족 → 다양한 시장 상황의 학습 데이터 필요")
    
    suggestions['reason'] = '; '.join(reasons) if reasons else '일반적인 재학습 권장'
    
    return suggestions

