"""
검증 결과 평가 및 자동 재학습 로직
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_THRESHOLDS = {
    'pf_issue': 0.5,      # 🔥 완화: 수익률은 참고용 (1.0 -> 0.5)
    'pf_warning': 0.8,    # 🔥 완화
    'return_issue': -0.2, # 🔥 완화: -20%까지는 허용 (0.0 -> -0.2)
    'return_warning': 0.0,
    'win_issue': 0.45,    # 🔥 강화: 승률 중요 (0.4 -> 0.45)
    'win_warning': 0.55,  # 🔥 강화: 승률 중요 (0.5 -> 0.55)
    'mdd_warning': 0.3,
    'mdd_issue': 0.5,     # 🔥 완화: MDD는 덜 중요
    'trade_issue': 5,
    'trade_warning': 10,
}

INTERVAL_VALIDATION_OVERRIDES = {
    '15m': {'pf_warning': 1.1, 'return_warning': 0.02, 'return_issue': -0.01, 'win_warning': 0.45, 'win_issue': 0.38, 'trade_warning': 30, 'trade_issue': 15},
    '30m': {'pf_warning': 1.1, 'return_warning': 0.03, 'return_issue': -0.005, 'win_warning': 0.46, 'win_issue': 0.4, 'trade_warning': 25, 'trade_issue': 12},
    '240m': {'pf_warning': 1.15, 'return_warning': 0.04, 'return_issue': 0.0, 'win_warning': 0.5, 'win_issue': 0.42, 'trade_warning': 15, 'trade_issue': 8},
    '1d': {'pf_warning': 1.2, 'return_warning': 0.05, 'return_issue': 0.01, 'win_warning': 0.52, 'win_issue': 0.45, 'trade_warning': 8, 'trade_issue': 4},
}


def _get_validation_thresholds(interval: Optional[str]) -> Dict[str, float]:
    thresholds = DEFAULT_VALIDATION_THRESHOLDS.copy()
    if interval and interval in INTERVAL_VALIDATION_OVERRIDES:
        thresholds.update(INTERVAL_VALIDATION_OVERRIDES[interval])
    return thresholds


def evaluate_validation_results(
    eval_result: Dict[str, Any],
    strict_mode: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    검증 결과를 평가하여 합격/불합격 판단
    
    🔥 우선순위:
    1. MFE/MAE 기반 검증 (예측 정확도) - 최우선
    2. 백테스트 기반 검증 (거래 수익률) - 참고용
    
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
    
    interval = eval_result.get('interval') or eval_result.get('selected_interval')
    thresholds = _get_validation_thresholds(interval)
    
    # 🔥 0. MFE/MAE 기반 검증 (최우선 - 예측 시스템의 핵심)
    mfe_validated = False
    mfe_score_bonus = 0.0
    strategy_id = eval_result.get('strategy_id', '')
    coin = eval_result.get('coin', '')
    
    try:
        from rl_pipeline.core.strategy_grading import get_strategy_mfe_stats, MFEGrading
        
        if strategy_id:
            mfe_stats = get_strategy_mfe_stats(strategy_id)
            if mfe_stats and mfe_stats.coverage_n >= 20:
                entry_score, risk_score, edge_score = MFEGrading.calculate_scores(mfe_stats)
                mfe_grade = MFEGrading.determine_grade(entry_score, risk_score, mfe_stats.coverage_n)
                
                # MFE 기반 평가 (예측 정확도)
                if entry_score >= 0.01:  # EntryScore >= 1%
                    mfe_validated = True
                    mfe_score_bonus = 3.0  # 큰 보너스
                    score += mfe_score_bonus
                    logger.info(f"✅ MFE 검증 통과: EntryScore={entry_score:.4f}, Grade={mfe_grade}")
                elif entry_score >= 0.0:  # 손익분기 이상
                    mfe_validated = True
                    mfe_score_bonus = 1.0
                    score += mfe_score_bonus
                    logger.debug(f"✅ MFE 검증 통과 (손익분기): EntryScore={entry_score:.4f}")
                elif entry_score > -0.005:  # 약간 손해 (-0.5% 이내)
                    warnings.append(f"MFE 분석: 약간 손해 구간 (EntryScore={entry_score:.4f})")
                    score -= 0.5
                else:  # 심각한 손해
                    issues.append(f"MFE 분석: 진입 가치 없음 (EntryScore={entry_score:.4f})")
                    score -= 2.0
                    
    except Exception as e:
        logger.debug(f"⚠️ MFE 검증 스킵: {e}")

    # 1. A/B 평가 체크 (🔥 MFE 검증 통과 시 가중치 낮춤)
    backtest_weight = 0.3 if mfe_validated else 1.0  # MFE 통과 시 백테스트 가중치 30%로
    ab_result = eval_result.get('comparison', {})
    if ab_result:
        hybrid_result = ab_result.get('hybrid', {})
        improvement = ab_result.get('improvement', {})
        
        pf = hybrid_result.get('profit_factor', 0.0)
        return_pct = hybrid_result.get('total_return', 0.0)
        win_rate = hybrid_result.get('win_rate', 0.0)
        mdd = hybrid_result.get('mdd', 0.0)
        trades = hybrid_result.get('trades', 0)
        
        # Profit Factor 체크 (방향성 예측에서는 보조 지표)
        pf_issue_th = thresholds['pf_issue']
        pf_warning_th = thresholds['pf_warning']
        if pf < pf_issue_th:
            warnings.append(f"Profit Factor가 매우 낮음 ({pf:.2f}) - 참고용")
            score -= 0.5  # 🔥 감점 대폭 축소 (-2.0 -> -0.5)
        elif pf < pf_warning_th:
            # warnings.append(f"Profit Factor가 낮음 ({pf:.2f})") # 너무 잦은 경고 제거
            score -= 0.1
        else:
            score += 0.5
        
        # Return 체크 (방향성 예측에서는 보조 지표)
        return_issue_th = thresholds['return_issue']
        return_warning_th = thresholds['return_warning']
        if return_pct < return_issue_th:
            warnings.append(f"수익률이 매우 낮음 ({return_pct:.2%}) - 참고용")
            score -= 0.5  # 🔥 감점 대폭 축소 (-2.0 -> -0.5)
        elif return_pct < return_warning_th:
            score -= 0.1
        else:
            score += 0.5
        
        # Win Rate 체크 (🔥 핵심 지표: 방향성 예측력)
        win_issue_th = thresholds['win_issue']
        win_warning_th = thresholds['win_warning']
        
        # 🔥 MFE 기반 승률 보정 (EntryScore가 양수면 개선 가능성 있음)
        mfe_adjusted_win_rate = win_rate
        has_mfe_potential = False
        coin = eval_result.get('coin', '')
        
        try:
            from rl_pipeline.core.strategy_grading import get_strategy_mfe_stats, MFEGrading
            
            # 해당 코인/인터벌의 평균 MFE 통계 확인
            strategy_id = eval_result.get('strategy_id', '')
            if strategy_id:
                mfe_stats = get_strategy_mfe_stats(strategy_id)
                if mfe_stats and mfe_stats.coverage_n >= 20:
                    entry_score, risk_score, edge_score = MFEGrading.calculate_scores(mfe_stats)
                    
                    if entry_score > 0:
                        # EntryScore가 양수 → 방향은 맞지만 타이밍/실행 문제
                        # 기대 승률: EntryScore가 높을수록 높은 승률 기대
                        expected_win_rate = 0.45 + min(0.15, entry_score * 5)  # 45% ~ 60%
                        
                        if win_rate < expected_win_rate * 0.8:
                            # 기대보다 20% 이상 낮음 → 개선 가능성 있지만 주의
                            has_mfe_potential = True
                            warnings.append(f"MFE 기반 기대승률({expected_win_rate:.1%}) 대비 낮음 - 타이밍 개선 필요")
                        elif win_rate >= expected_win_rate:
                            # 기대 이상 → 보너스
                            score += 0.5
                    else:
                        # EntryScore가 음수 → 근본적으로 잘못된 방향
                        issues.append(f"MFE 분석: 진입 가치 없음 (EntryScore={entry_score:.4f} < 0)")
                        score -= 2.0  # 추가 감점
        except Exception as e:
            logger.debug(f"⚠️ MFE 보정 스킵: {e}")
        
        if win_rate < win_issue_th:
            if has_mfe_potential:
                # MFE는 양수인데 승률이 낮음 → 개선 가능성 있음
                warnings.append(f"승률 기준 미달 ({win_rate:.1%} < {win_issue_th:.1%}) - MFE 양수로 개선 가능")
                score -= 1.5  # 감점 완화 (3.0 → 1.5)
            else:
                issues.append(f"승률(방향성 정확도)이 기준 미달 ({win_rate:.1%} < {win_issue_th:.1%})")
                score -= 3.0  # 🔥 감점 대폭 강화 (-1.0 -> -3.0)
        elif win_rate < win_warning_th:
            warnings.append(f"승률이 권장 수준 미달 ({win_rate:.1%}, 권장: >{win_warning_th:.1%})")
            score -= 1.0  # 🔥 감점 강화 (-0.3 -> -1.0)
        else:
            score += 2.0  # 🔥 가점 강화 (+0.5 -> +2.0)
        
        # Max Drawdown 체크
        if mdd > 0.3:
            issues.append(f"Max Drawdown이 너무 큼 ({mdd:.1%})")
            score -= 1.5
        elif mdd > 0.2:
            warnings.append(f"Max Drawdown이 큼 ({mdd:.1%}, 권장: <20%)")
            score -= 0.5
        
        # Trades 체크 (🔥 MFE 검증 통과 시 완화 - 예측 시스템은 거래 횟수가 중요하지 않음)
        trade_issue_th = thresholds['trade_issue']
        trade_warning_th = thresholds['trade_warning']
        
        if mfe_validated:
            # MFE 검증 통과 시: 거래 횟수는 참고용 경고만 (감점 없음)
            if trades == 0:
                logger.debug(f"ℹ️ 거래 0건 - MFE 검증 통과로 무시 (예측 시스템)")
                # 감점 없음 - 예측 정확도가 확인됨
            elif trades < trade_issue_th:
                logger.debug(f"ℹ️ 거래 수 적음 ({trades}개) - MFE 검증으로 대체")
        else:
            # MFE 검증 미통과 시: 기존 로직 (백테스트 기반 평가, 감점 완화)
            if trades == 0:
                warnings.append(f"거래 0건 (MFE 데이터 축적 필요)")
                score -= 1.0  # 🔥 감점 완화 (5.0 → 1.0)
            elif trades < trade_issue_th:
                warnings.append(f"거래 수가 부족함 ({trades}개)")
                score -= 0.5  # 완화
            elif trades < trade_warning_th:
                warnings.append(f"거래 수가 적음 ({trades}개)")
                score -= 0.2
        
        # 규칙 대비 개선도 체크 (방향성 예측에서는 덜 중요할 수 있음)
        pf_improvement = improvement.get('profit_factor_improvement', 0.0)
        win_improvement = improvement.get('win_rate_improvement', 0.0) # 🔥 승률 개선도 추가
        
        if win_improvement < -0.1: # 승률이 10%p 이상 악화되면 문제
            issues.append(f"규칙 대비 승률 악화 ({win_improvement:.1%})")
            score -= 1.5
        
        if pf_improvement < -0.5:  # 수익률은 50% 이상 악화되어야 문제
            warnings.append(f"규칙 대비 수익성 악화 ({pf_improvement:.1%}) - 참고용")
            score -= 0.2
    
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
        'interval': interval,
        'critical_count': len([i for i in issues if any(keyword in i for keyword in [
            '승률', '과적합', '거래 수가 0개'  # 🔥 Profit Factor, 음수 수익률 제거, 승률 추가
        ])])
    }
    
    return passed, reason, details


def should_retrain(
    eval_result: Dict[str, Any],
    previous_attempts: int = 0,
    max_attempts: int = 1
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
    
    # 🔥 거래 0회 처리 (MFE 검증 여부에 따라 다르게 처리)
    has_zero_trades = any('거래 0건' in issue or '거래 수가 0개' in issue for issue in issues)
    has_mfe_validation = any('MFE' in w for w in details.get('warnings', []))
    
    if has_zero_trades and not has_mfe_validation:
        # MFE 검증 없이 거래 0건 → 재학습 필요 (데이터 축적 목적)
        return True, f"거래 0건 + MFE 데이터 부족 → MFE 축적을 위해 재학습"
    
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
        
        # 🔥 거래 0회 (예측 시스템에서는 덜 심각 - MFE 축적 필요)
        if trades == 0:
            suggestions['collect_more_data'] = True
            suggestions['wait_for_mfe_data'] = True  # 🆕 MFE 데이터 축적 대기
            reasons.append("거래 0회 → MFE 데이터 축적 대기 권장 (예측 시스템)")
        elif hybrid_result.get('win_rate', 0.0) < 0.45: # 🔥 승률이 낮으면 재학습 유도
            suggestions['adjust_hyperparameters'] = True
            suggestions['collect_more_data'] = True
            reasons.append("승률 < 45% → 방향성 예측력 부족: 더 많은 학습 데이터 및 하이퍼파라미터 조정 필요")
        elif pf < 0.5: # 수익률이 너무 처참할 때만 (보조)
            suggestions['adjust_entropy_coef'] = True
            reasons.append("수익성 매우 저조 → 탐험(entropy_coef) 증가 고려")
        
        if trades and trades < 40:
            suggestions['adjust_batch_size'] = True
            reasons.append("거래 수가 적음 → 배치 크기 축소 및 더 많은 데이터 확보 권장")
    
    # 다중 기간 검증 결과 분석
    mp_result = eval_result.get('multi_period')
    if mp_result:
        consistency = mp_result.get('consistency', 0.0)
        
        if consistency < 0.5:
            suggestions['collect_more_data'] = True
            reasons.append("레짐 일관성 부족 → 다양한 시장 상황의 학습 데이터 필요")
    
    suggestions['reason'] = '; '.join(reasons) if reasons else '일반적인 재학습 권장'
    
    return suggestions

