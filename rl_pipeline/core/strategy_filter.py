"""
전략 필터링 모듈 - 물리 법칙 및 생존 조건 기반
개선 사항: MFE/MAE 기반 Gate Score 필터링 추가
"""

import sqlite3
import argparse
from typing import List, Dict, Tuple, Optional, Any
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


def calculate_risk_of_ruin(win_rate: float, profit_ratio: float, loss_ratio: float) -> float:
    """파산 확률(Risk of Ruin) 계산"""
    if win_rate <= 0 or profit_ratio <= 0 or loss_ratio <= 0:
        return 1.0
    
    b = profit_ratio / loss_ratio
    kelly_f = (win_rate * (b + 1) - 1) / b
    
    if kelly_f <= 0:
        return 1.0
    
    expectancy = (win_rate * profit_ratio) - ((1 - win_rate) * loss_ratio)
    
    if expectancy <= 0:
        return 1.0
        
    return 0.0


def apply_mfe_filter(
    db_path: str = "data_storage/learning_strategies.db",
    min_entry_score: float = -0.005  # 최소 -0.5% (약간의 불리함까지 허용)
) -> int:
    """MFE/MAE 기반 필터링 (GPT.md 2번 항목)
    
    전략의 기대 수익(Upside)보다 기대 손실(Downside)이 너무 큰 경우 제거
    (승률이 높아도 손익비가 극도로 나쁜 '물리는 전략' 제거)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. strategy_label_stats 테이블과 조인하여 EntryScore 계산
        # EntryScore = rmax_p90 - 1.5 * abs(rmin_p10)
        
        cursor.execute("""
            SELECT s.id, ls.rmax_p90, ls.rmin_p10, ls.n_signals
            FROM strategies s
            JOIN strategy_label_stats ls 
                ON s.id = ls.strategy_id 
                AND s.symbol = ls.coin 
                AND s.interval = ls.interval
            WHERE ls.n_signals >= 20
        """)
        
        rows = cursor.fetchall()
        
        ids_to_remove = []
        
        for row in rows:
            sid, rmax_p90, rmin_p10, n = row
            
            # EntryScore 계산 (k=1.5)
            # rmin_p10은 음수이므로 절대값 취함
            entry_score = rmax_p90 - (1.5 * abs(rmin_p10))
            
            if entry_score < min_entry_score:
                ids_to_remove.append(sid)
                
        if ids_to_remove:
            logger.info(f"⚠️ {len(ids_to_remove)}개 전략이 MFE/MAE 필터링(EntryScore < {min_entry_score})에 걸렸습니다.")
            # 실제 제거는 잠시 보류 (로깅만)
            # placeholder = ','.join('?' * len(ids_to_remove))
            # cursor.execute(f"DELETE FROM strategies WHERE id IN ({placeholder})", ids_to_remove)
            # conn.commit()
            
        logger.info(f"⚖️ MFE/MAE 필터링 완료: {len(ids_to_remove)}개 부적격 전략 발견 (삭제 보류)")
        return len(ids_to_remove)
        
    except Exception as e:
        logger.warning(f"⚠️ MFE 필터링 실패 (테이블 없을 수 있음): {e}")
        return 0
    finally:
        conn.close()


def update_league_rankings(
    db_path: str = "data_storage/learning_strategies.db",
    top_n_per_group: int = 100,  # 코인×인터벌×레짐별 상위 N개만 major
    min_entry_score: float = 0.0  # major 승격 최소 조건
) -> dict:
    """
    🔥 MFE/MAE 기반 리그 승강제 업데이트
    
    - EntryScore 기준으로 전략 순위 매김
    - 코인 × 인터벌 × 레짐별로 그룹화하여 상위 N개 → major 리그
    - 나머지 → minor 리그 (데이터 유지, 트레이딩 제외)
    - 삭제 없음! 모든 전략 데이터 보존
    
    Returns:
        {'promoted': N, 'demoted': M, 'total_major': K}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    results = {'promoted': 0, 'demoted': 0, 'total_major': 0}
    
    try:
        # 1. 모든 전략의 EntryScore 계산 및 league_score 업데이트
        # 🔥 regime 컬럼 추가!
        cursor.execute("""
            SELECT s.id, s.symbol, s.interval, s.regime, s.league,
                   ls.rmax_p90, ls.rmin_p10, ls.n_signals
            FROM strategies s
            LEFT JOIN strategy_label_stats ls 
                ON s.id = ls.strategy_id 
                AND s.symbol = ls.coin 
                AND s.interval = ls.interval
        """)
        
        rows = cursor.fetchall()
        
        # 전략별 EntryScore 계산
        strategy_scores = []
        for row in rows:
            # 🔥 regime 컬럼 추가됨
            sid, symbol, interval, regime, current_league, rmax_p90, rmin_p10, n_signals = row
            
            # MFE/MAE 데이터가 없으면 기본 점수 부여 (평가 보류)
            if rmax_p90 is None or rmin_p10 is None or (n_signals or 0) < 20:
                entry_score = -999.0  # 평가 불가 (minor 유지)
            else:
                entry_score = rmax_p90 - (1.5 * abs(rmin_p10))
            
            # 🔥 regime이 None이면 'neutral'로 기본값 설정
            regime = regime or 'neutral'
            
            strategy_scores.append({
                'id': sid,
                'symbol': symbol,
                'interval': interval,
                'regime': regime,  # 🔥 regime 추가!
                'current_league': current_league,
                'entry_score': entry_score,
                'n_signals': n_signals or 0
            })
        
        # 2. league_score 일괄 업데이트
        update_data = [(s['entry_score'], s['id']) for s in strategy_scores if s['entry_score'] > -999]
        if update_data:
            cursor.executemany("UPDATE strategies SET league_score = ? WHERE id = ?", update_data)
        
        # 3. 🔥 코인×인터벌×레짐별 그룹화 및 순위 결정
        from collections import defaultdict
        groups = defaultdict(list)
        
        for s in strategy_scores:
            # 🔥 regime 추가됨
            key = (s['symbol'], s['interval'], s['regime'])
            groups[key].append(s)
        
        promoted_ids = []
        demoted_ids = []
        
        # 🔥 코인×인터벌×레짐별 순회
        for (symbol, interval, regime), strategies in groups.items():
            # EntryScore 기준 내림차순 정렬
            strategies.sort(key=lambda x: x['entry_score'], reverse=True)
            
            for rank, s in enumerate(strategies):
                # 상위 N개 + 최소 점수 충족 → major
                should_be_major = (rank < top_n_per_group) and (s['entry_score'] >= min_entry_score)
                
                if should_be_major and s['current_league'] != 'major':
                    promoted_ids.append(s['id'])
                elif not should_be_major and s['current_league'] == 'major':
                    demoted_ids.append(s['id'])
        
        # 4. 리그 업데이트 (승격/강등)
        if promoted_ids:
            placeholder = ','.join('?' * len(promoted_ids))
            cursor.execute(f"UPDATE strategies SET league = 'major' WHERE id IN ({placeholder})", promoted_ids)
            results['promoted'] = len(promoted_ids)
            logger.info(f"🏆 {len(promoted_ids)}개 전략 major 리그 승격")
        
        if demoted_ids:
            placeholder = ','.join('?' * len(demoted_ids))
            cursor.execute(f"UPDATE strategies SET league = 'minor' WHERE id IN ({placeholder})", demoted_ids)
            results['demoted'] = len(demoted_ids)
            logger.info(f"📉 {len(demoted_ids)}개 전략 minor 리그 강등")
        
        conn.commit()
        
        # 5. 최종 major 리그 수 확인
        cursor.execute("SELECT COUNT(*) FROM strategies WHERE league = 'major'")
        results['total_major'] = cursor.fetchone()[0]
        
        logger.info(f"⚖️ 리그 업데이트 완료: major {results['total_major']}개 (승격 {results['promoted']}, 강등 {results['demoted']})")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 리그 업데이트 실패: {e}")
        conn.rollback()
        return results
    finally:
        conn.close()


def perform_stress_test(
    db_path: str = "data_storage/learning_strategies.db",
    n_simulations: int = 1000,
    n_trades: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """몬테카를로 스트레스 테스트 수행"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    results = {
        "tested_count": 0,
        "failed_count": 0,
        "high_risk_strategies": []
    }
    
    try:
        cursor.execute("""
            SELECT id, symbol as coin, interval, win_rate, profit, trades_count, profit_factor
            FROM strategies
            WHERE win_rate IS NOT NULL 
              AND trades_count >= 10
        """)
        strategies = cursor.fetchall()
        results["tested_count"] = len(strategies)
        
        logger.info(f"🔬 스트레스 테스트 시작: {len(strategies)}개 전략 (Simulations={n_simulations}, Trades={n_trades})")
        
        risk_threshold = -20.0
        failed_ids = []
        
        for strat in strategies:
            sid = strat['id']
            win_rate = strat['win_rate'] or 0.0
            
            if win_rate <= 0: continue
                
            pf = strat['profit_factor'] if strat['profit_factor'] else 1.2
            if pf <= 0: continue

            if win_rate >= 1.0 or win_rate <= 0.0: continue
                
            payoff_ratio = pf * (1.0 - win_rate) / win_rate
            
            final_equity_curves = []
            
            for _ in range(n_simulations):
                outcomes = np.random.choice([1, 0], size=n_trades, p=[win_rate, 1-win_rate])
                returns = np.where(outcomes == 1, payoff_ratio, -1.0)
                cumulative = np.cumsum(returns)
                final_equity_curves.append(cumulative[-1])
            
            var_95 = np.percentile(final_equity_curves, (1 - confidence_level) * 100)
            
            if var_95 < risk_threshold:
                failed_ids.append(sid)
                results["high_risk_strategies"].append(sid)
                
        results["failed_count"] = len(failed_ids)
        
        if failed_ids:
            logger.warning(f"⚠️ {len(failed_ids)}개 전략이 스트레스 테스트(VaR)를 통과하지 못했으나, 삭제하지 않고 유지합니다.")
        
        logger.info(f"📉 스트레스 테스트 완료: {len(failed_ids)}개 전략이 고위험군으로 분류됨 (VaR 95% < {risk_threshold})")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 스트레스 테스트 실패: {e}")
        return results
    finally:
        conn.close()


def apply_physics_laws_filter(
    db_path: str = "data_storage/learning_strategies.db",
    max_mdd_pct: float = 0.20,
    min_trades: int = 5,
    min_profit_factor: float = 0.5,
    strict_mode: bool = True
) -> int:
    """물리 법칙 기반 생존 필터링"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    removed_count = 0
    reasons = {"mdd_violation": 0, "ruin_risk": 0, "insufficient_trades": 0, "low_pf": 0}
    
    try:
        cursor.execute("SELECT id, symbol as coin, interval, max_drawdown, win_rate, trades_count, profit_factor, profit FROM strategies")
        strategies = cursor.fetchall()
        
        ids_to_remove = []
        
        for strat in strategies:
            sid = strat['id']
            mdd = strat['max_drawdown'] if strat['max_drawdown'] is not None else 1.0
            win_rate = strat['win_rate'] if strat['win_rate'] is not None else 0.0
            trades = strat['trades_count'] if strat['trades_count'] is not None else 0
            pf = strat['profit_factor'] if strat['profit_factor'] is not None else 0.0
            
            mdd_val = abs(mdd)
            if mdd_val > max_mdd_pct:
                ids_to_remove.append(sid)
                reasons["mdd_violation"] += 1
                continue
                
            if trades < min_trades:
                if strict_mode:
                    ids_to_remove.append(sid)
                    reasons["insufficient_trades"] += 1
                    continue
            
            if pf < min_profit_factor:
                ids_to_remove.append(sid)
                reasons["low_pf"] += 1
                continue
                
            if win_rate > 0 and win_rate < 1:
                payoff_ratio = pf * (1 - win_rate) / win_rate
                if payoff_ratio > 0:
                    kelly = win_rate - (1 - win_rate) / payoff_ratio
                    if kelly <= 0:
                        ids_to_remove.append(sid)
                        reasons["ruin_risk"] += 1
                        continue
            elif win_rate == 0:
                 ids_to_remove.append(sid)
                 reasons["ruin_risk"] += 1
                 continue
        
        if ids_to_remove:
            logger.info(f"⚠️ {len(ids_to_remove)}개 전략이 물리 법칙을 위반했으나, 사용자 요청으로 삭제하지 않습니다.")
            
        logger.info(f"⚖️ 물리 법칙 필터링 완료: 총 {len(ids_to_remove)}개 위반 (삭제 안함)")
        logger.info(f"   └─ MDD 초과({max_mdd_pct*100}%): {reasons['mdd_violation']}개")
        logger.info(f"   └─ 통계 부족(<{min_trades}회): {reasons['insufficient_trades']}개")
        logger.info(f"   └─ 손익비 미달(<{min_profit_factor}): {reasons['low_pf']}개")
        logger.info(f"   └─ 파산 위험(Kelly<=0): {reasons['ruin_risk']}개")
        
        # 🔥 추가: MFE 필터링 연동
        apply_mfe_filter(db_path)
        
        return removed_count
        
    except Exception as e:
        logger.error(f"❌ 물리 법칙 필터링 실패: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def remove_low_grade_strategies(db_path: str = "data_storage/learning_strategies.db") -> int:
    """낮은 등급 또는 등급이 없는 전략 제거"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, symbol as coin, interval, quality_grade
            FROM strategies
            WHERE quality_grade = 'F' 
               OR quality_grade IS NULL 
               OR quality_grade = 'UNKNOWN'
        """)
        low_grade_strategies = cursor.fetchall()
        
        if not low_grade_strategies:
            logger.info("제거할 낮은 등급 전략이 없습니다.")
            return 0
            
        low_grade_ids = [row[0] for row in low_grade_strategies]
        
        if low_grade_ids:
            logger.info(f"⚠️ {len(low_grade_ids)}개 낮은 등급 전략이 발견되었으나, 사용자 요청으로 삭제하지 않습니다.")
        
        logger.info(f"✅ {len(low_grade_ids)}개 낮은 등급/등급 없음 전략 발견 (삭제 안함)")
        return 0
    
    except Exception as e:
        logger.error(f"❌ 낮은 등급 전략 제거 실패: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def keep_top_strategies(
    db_path: str = "data_storage/learning_strategies.db",
    top_percent: float = 0.1
) -> Tuple[int, int]:
    """각 코인/인터벌/레짐별 상위 전략만 유지 (정원 관리)"""
    from rl_pipeline.core.env import config
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_kept = 0
    total_removed = 0
    
    limit_per_combo = config.STRATEGIES_PER_COMBINATION
    logger.info(f"🧹 전략 정원 관리 시작 (제한: 조합당 {limit_per_combo}개)")
    
    try:
        cursor.execute("SELECT DISTINCT symbol as coin, interval, regime FROM strategies")
        combinations = cursor.fetchall()
        
        if not combinations:
            logger.warning("전략이 없습니다.")
            return 0, 0
        
        for coin, interval, regime in combinations:
            regime_clause = "IS NULL" if regime is None else "= ?"
            params = (coin, interval) if regime is None else (coin, interval, regime)
            
            cursor.execute(f"""
                SELECT COUNT(*) FROM strategies
                WHERE symbol = ? AND interval = ? AND regime {regime_clause}
            """, params)
            total_count = cursor.fetchone()[0]
            
            if total_count <= limit_per_combo:
                total_kept += total_count
                continue
                
            keep_count = limit_per_combo
            
            # 상위 전략 우선순위: Profit Factor > Sharpe > Win Rate
            cursor.execute(f"""
                SELECT id FROM strategies
                WHERE symbol = ? AND interval = ? AND regime {regime_clause}
                ORDER BY
                    COALESCE(profit_factor, 0) DESC,
                    COALESCE(sharpe_ratio, 0) DESC,
                    COALESCE(win_rate, 0) DESC
                LIMIT ?
            """, (*params, keep_count))
            
            keep_ids = [row[0] for row in cursor.fetchall()]
            
            if not keep_ids: continue
            
            chunk_size = 900
            placeholders = ','.join(['?' for _ in keep_ids])
            delete_params = list(params) + keep_ids
            
            cursor.execute(f"""
                DELETE FROM strategies
                WHERE id IN (
                    SELECT id FROM strategies
                    WHERE symbol = ? AND interval = ? AND regime {regime_clause}
                    AND id NOT IN ({placeholders})
                )
            """, delete_params)
            
            removed = cursor.rowcount
            total_kept += len(keep_ids)
            total_removed += removed
            
            regime_str = regime if regime else "Common"
            logger.info(f"  {coin}-{interval} [{regime_str}]: 정원 초과({total_count}/{limit_per_combo}) -> {removed}개 하위 전략 제거")
        
        conn.commit()
        
        logger.info(f"\n✅ 정원 관리 완료: 총 {total_kept}개 유지, {total_removed}개 제거")
        return total_kept, total_removed
    
    except Exception as e:
        logger.error(f"❌ 상위 전략 선별 실패: {e}")
        conn.rollback()
        return 0, 0
    finally:
        conn.close()


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description='전략 필터링')
    parser.add_argument('--action', choices=['remove_f_grade', 'filter_performance', 'keep_top', 'physics_filter', 'stress_test', 'mfe_filter'],
                       required=True, help='실행할 액션')
    parser.add_argument('--db_path', default='data_storage/learning_strategies.db',
                       help='데이터베이스 경로')
    # ... (args 생략) ...
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    if args.action == 'remove_f_grade':
        remove_low_grade_strategies(args.db_path)
    elif args.action == 'keep_top':
        keep_top_strategies(args.db_path)
    elif args.action == 'physics_filter':
        apply_physics_laws_filter(args.db_path)
    elif args.action == 'stress_test':
        perform_stress_test(args.db_path)
    elif args.action == 'mfe_filter':
        apply_mfe_filter(args.db_path)

if __name__ == "__main__":
    main()
