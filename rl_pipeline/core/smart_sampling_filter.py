"""
스마트 샘플링: 2단계 학습 시스템
Phase 1: 빠른 스크리닝 (현재 설정 유지)
Phase 2: 생존자만 심화 학습
"""
import sqlite3
import logging

logger = logging.getLogger(__name__)


def apply_smart_sampling_filter(
    db_path: str,
    phase1_min_trades: int = 15,
    phase2_min_trades: int = 30,
    max_mdd_pct: float = 0.99
):
    """스마트 샘플링 필터: 2단계 학습"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, coin, interval, trades_count, win_rate, profit, max_drawdown, profit_factor
            FROM strategies
            WHERE trades_count IS NOT NULL
        """)
        all_strategies = cursor.fetchall()
        
        phase1_survivors = []
        phase2_needed = []
        phase2_ready = []
        to_remove = []
        
        for row in all_strategies:
            sid, coin, interval, trades, win_rate, profit, mdd, pf = row
            
            if mdd and mdd < -max_mdd_pct:
                to_remove.append(sid)
                continue
                
            if win_rate == 0:
                # 사용자 요청: 초기 단계에서 승률 0이라도 제거하지 않음 (데이터 수집 우선)
                # to_remove.append(sid)
                phase1_survivors.append((sid, coin, interval, trades))
                phase2_needed.append((sid, coin, interval, trades))
                continue
            
            if trades >= phase2_min_trades:
                phase2_ready.append((sid, coin, interval, trades))
            elif trades >= phase1_min_trades:
                phase1_survivors.append((sid, coin, interval, trades))
                phase2_needed.append((sid, coin, interval, trades))
            else:
                # 사용자 요청: 물리 법칙/샘플링 필터링에 의한 삭제 방지
                # 거래 횟수 부족(<15회)이라도 삭제하지 않고 계속 데이터 수집하도록 Phase 2 필요 그룹에 포함
                phase1_survivors.append((sid, coin, interval, trades))
                phase2_needed.append((sid, coin, interval, trades))
                # to_remove.append(sid)
        
        if to_remove:
            # 사용자 요청: 삭제 로직 완전 비활성화 (데이터 보존)
            logger.info(f"⚠️ {len(to_remove)}개 전략이 삭제 대상이나, 사용자 요청으로 삭제하지 않습니다.")
            
            # chunk_size = 900
            # for i in range(0, len(to_remove), chunk_size):
            #     chunk = to_remove[i:i+chunk_size]
            #     placeholders = ",".join(["?" for _ in chunk])
                
            #     try:
            #         cursor.execute(f"DELETE FROM rl_episode_summary WHERE strategy_id IN ({placeholders})", chunk)
            #         cursor.execute(f"DELETE FROM rl_episodes WHERE strategy_id IN ({placeholders})", chunk)
            #     except sqlite3.OperationalError:
            #         pass
                
            #     cursor.execute(f"DELETE FROM strategies WHERE id IN ({placeholders})", chunk)
            
            # conn.commit()
        
        logger.info(f"🎯 스마트 샘플링 필터링 결과:")
        logger.info(f"   ✅ Phase 2 완료 (≥{phase2_min_trades}회): {len(phase2_ready)}개 전략")
        logger.info(f"   🔄 Phase 2 필요 ({phase1_min_trades}~{phase2_min_trades-1}회): {len(phase2_needed)}개 전략")
        logger.info(f"   ❌ 제거 대상 (<{phase1_min_trades}회): {len(to_remove)}개 전략 (유지됨)")
        
        return {
            "phase1_survivors": len(phase1_survivors),
            "phase2_needed": phase2_needed,
            "phase2_ready": len(phase2_ready),
            "removed": len(to_remove)
        }
        
    except Exception as e:
        logger.error(f"❌ 스마트 샘플링 필터링 실패: {e}")
        conn.rollback()
        return {"phase1_survivors": 0, "phase2_needed": [], "phase2_ready": 0, "removed": 0}
    finally:
        conn.close()
