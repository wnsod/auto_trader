#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시장 인사이트 마이너 (정밀 버전) - 캔들 데이터 기반 놓친 기회/잘한 관망 학습
🆕 방향 B: 단기(15m)에서 트리거 발견 → 모든 인터벌(15m, 30m, 240m, 1d) 동시 학습
"""

import os
import sqlite3
import pandas as pd
import time
from typing import Dict, Set, List, Optional
from collections import defaultdict
from trade.core.database import get_db_connection, TRADING_SYSTEM_DB_PATH
from trade.core.sequence_analyzer import SequenceAnalyzer

# 분석 대상 인터벌 (단기 → 장기 순서)
ANALYSIS_INTERVALS = ['15m', '30m', '240m', '1d']
TRIGGER_INTERVAL = '15m'  # 트리거 감지용 (가장 민감한 인터벌)

# 헬퍼 함수
def safe_float(value, default: float = 0.0) -> float:
    """안전한 float 변환"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

class MarketInsightMiner:
    """시장 인사이트 마이너 - 캔들 데이터로 놓친 기회/잘한 관망 학습
    
    🆕 방향 B 로직:
    1. 15m 캔들에서 ±5% 이상 변동 트리거 감지
    2. 트리거 시점 T에 각 인터벌(15m, 30m, 240m, 1d)의 시그널 조회
    3. 각 인터벌이 해당 방향을 맞췄는지 평가
    4. 맞춘 인터벌은 신뢰도 UP, 틀린 인터벌은 신뢰도 DOWN
    """
    
    def __init__(self, learner):
        self.learner = learner
        self.db_path = TRADING_SYSTEM_DB_PATH
        self.min_rise_threshold = 5.0   # 5% 이상 상승 (놓친 기회)
        self.min_drop_threshold = -5.0  # 5% 이상 하락 (잘한 관망)
        self.lookback_hours = 24        # 🆕 최근 24시간 캔들 분석 (6시간→24시간 확장)
        self.processed_insights = set()  # 이미 학습한 인사이트 추적
        
        # 🆕 인터벌별 적중률 통계
        self.interval_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    def _create_pattern(self, coin: str, max_profit_pct: float, max_loss_pct: float, start_price: float = 0, max_high: float = 0, min_low: float = 0) -> str:
        """패턴 생성 - 단순히 수익률뿐만 아니라 캔들의 움직임 특성까지 반영"""
        # 1. 가격 변동폭 수준 (Volatility)
        profit_level = "high_rise" if max_profit_pct >= 10.0 else "medium_rise" if max_profit_pct >= 5.0 else "low_rise"
        loss_level = "high_drop" if max_loss_pct <= -10.0 else "medium_drop" if max_loss_pct <= -5.0 else "low_drop"
        
        # 2. 되돌림 수준 (Retracement) 분석
        retracement_level = "stable"
        if max_high > start_price and max_profit_pct > 3.0:
            current_retracement = ((max_high - min_low) / (max_high - start_price)) if (max_high - start_price) > 0 else 0
            if current_retracement > 0.7: retracement_level = "deep_pullback"
            elif current_retracement > 0.3: retracement_level = "healthy_pullback"
            
        return f"{coin}_{profit_level}_{loss_level}_{retracement_level}"

    def mine_insights(self):
        """🆕 방향 B: 15m에서 트리거 발견 → 모든 인터벌 시그널 동시 학습"""
        try:
            # 🚀 [Fix] PC 시각이 아닌 DB 최신 캔들 시각 기준
            try:
                from trade.core.database import get_latest_candle_timestamp
                current_time = get_latest_candle_timestamp()
            except:
                current_time = int(time.time())
                
            lookback_seconds = self.lookback_hours * 3600
            start_time = current_time - lookback_seconds
            
            # 🔧 캔들 DB 경로 설정 (다중 폴백)
            candles_db_path = os.environ.get('CANDLES_DB_PATH') or os.environ.get('RL_DB_PATH')
            if not candles_db_path:
                try:
                    from signal_selector.config import CANDLES_DB_PATH as CONFIG_CANDLES_PATH
                    candles_db_path = CONFIG_CANDLES_PATH
                except ImportError:
                    try:
                        from trade.core.database import CANDLES_DB_PATH as CORE_CANDLES_PATH
                        candles_db_path = CORE_CANDLES_PATH
                    except ImportError:
                        # 🆕 최후의 폴백: 기본 경로 시도
                        default_path = os.path.join(os.path.dirname(self.db_path), 'trade_candles.db')
                        if os.path.exists(default_path):
                            candles_db_path = default_path
            
            if not candles_db_path or not os.path.exists(candles_db_path):
                print(f"   ⚠️ 캔들 DB를 찾을 수 없습니다. (경로: {candles_db_path})")
                print(f"   💡 환경변수 CANDLES_DB_PATH 또는 RL_DB_PATH를 설정해주세요.")
                return
            
            # 1. 🎯 15m 캔들에서만 트리거 감지 (가장 민감)
            with get_db_connection(candles_db_path, read_only=True) as conn:
                query = """
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM candles 
                    WHERE timestamp >= ? AND interval = ?
                    ORDER BY symbol, timestamp ASC
                """
                trigger_candles = pd.read_sql(query, conn, params=(start_time, TRIGGER_INTERVAL))
            
            if trigger_candles.empty:
                print(f"   ℹ️ 최근 {self.lookback_hours}시간 {TRIGGER_INTERVAL} 캔들 데이터가 없습니다.")
                return

            # 보유/거래 중인 코인 제외
            held_coins = self._get_held_or_traded_coins(start_time)
            analyzed_coins = set(trigger_candles['symbol'].unique()) - held_coins
            print(f"   📊 분석 대상: {len(analyzed_coins)}개 코인 (보유/거래 중 {len(held_coins)}개 제외)")
            
            opportunity_count = 0
            avoidance_count = 0
            interval_results = defaultdict(lambda: {'correct': 0, 'total': 0})

            for coin, group in trigger_candles.groupby('symbol'):
                if coin in held_coins: continue
                
                group = group.reset_index(drop=True)
                if len(group) < 5: continue
                
                base_price = group.iloc[0]['close']
                trigger_ts = None
                is_bullish_move = False
                final_profit = 0.0

                # 2. 🎯 트리거 지점 포착 (±5% 변동 시작점)
                for i in range(1, len(group)):
                    change = ((group.iloc[i]['high'] - base_price) / base_price) * 100
                    drop = ((group.iloc[i]['low'] - base_price) / base_price) * 100
                    
                    if change >= 5.0:  # 상승 트리거
                        trigger_ts = group.iloc[i]['timestamp']
                        max_after = group.iloc[i:]['high'].max()
                        final_profit = ((max_after - base_price) / base_price) * 100
                        is_bullish_move = True
                        break
                    elif drop <= -5.0:  # 하락 트리거
                        trigger_ts = group.iloc[i]['timestamp']
                        min_after = group.iloc[i:]['low'].min()
                        final_profit = ((min_after - base_price) / base_price) * 100
                        is_bullish_move = False
                        break
                
                if not trigger_ts or abs(final_profit) < 5.0:
                    continue
                
                # 3. 🆕 트리거 시점에 모든 인터벌 시그널 조회 및 평가
                all_interval_signals = self._load_all_interval_signals(coin, trigger_ts)
                
                if not all_interval_signals:
                    continue
                
                # 4. 🆕 각 인터벌이 방향을 맞췄는지 평가
                correct_intervals = []
                wrong_intervals = []
                
                for interval, signal in all_interval_signals.items():
                    direction = str(signal.get('integrated_direction', 'neutral')).upper()
                    score = safe_float(signal.get('signal_score', 0.0))
                    
                    # 🆕 [5-Candle Sequence Analysis] 트리거 시점의 흐름 분석 추가
                    seq_bonus = 1.0
                    seq_reason = ""
                    try:
                        # 트리거 시점 기준 최근 5개 캔들 로드
                        recent_candles = self._get_recent_candles_at_ts(coin, interval, trigger_ts)
                        if recent_candles is not None and len(recent_candles) >= 5:
                            analysis = SequenceAnalyzer.analyze_sequence(recent_candles, interval)
                            seq_bonus = analysis['score_mod']
                            seq_reason = analysis['reason']
                            
                            # 흐름 분석과 실제 방향이 일치하는지 확인 (학습 가중치용)
                            flow_matched = (is_bullish_move and seq_bonus > 1.05) or (not is_bullish_move and seq_bonus < 0.95)
                            if flow_matched:
                                # 5캔들 흐름이 정답 방향을 가리키고 있었다면, 시그널이 못맞춘 것에 대한 페널티 강화
                                if (is_bullish_move and score < 0.1) or (not is_bullish_move and score > -0.1):
                                    seq_reason += " (흐름은 맞았으나 시그널이 놓침)"
                    except Exception as seq_err:
                        print(f"      ⚠️ {interval} 흐름 분석 오류: {seq_err}")

                    # 시그널이 실제 움직임과 일치하는지 판단
                    predicted_bullish = any(x in direction for x in ['BULL', 'LONG', 'BUY', 'STRONG BULL'])
                    predicted_bearish = any(x in direction for x in ['BEAR', 'SHORT', 'SELL', 'STRONG BEAR'])
                    
                    is_correct = False
                    if is_bullish_move and predicted_bullish and score > 0.1:
                        is_correct = True
                    elif not is_bullish_move and predicted_bearish and score < -0.1:
                        is_correct = True
                    elif not is_bullish_move and not predicted_bullish and score < 0:  # 관망 잘함
                        is_correct = True
                    
                    interval_results[interval]['total'] += 1
                    if is_correct:
                        interval_results[interval]['correct'] += 1
                        correct_intervals.append(interval)
                    else:
                        wrong_intervals.append(interval)
                    
                    # 5. 🆕 Thompson Sampling 업데이트 (인터벌별 패턴)
                    pattern = f"{coin}_{interval}_{direction}"
                    # 🆕 흐름 분석 결과(seq_bonus)를 학습 가중치에 반영
                    # 흐름이 명확(seq_bonus != 1.0)할수록 더 강하게 학습
                    weight = 1.5 if is_correct else 0.8
                    if seq_bonus > 1.1 or seq_bonus < 0.9:
                        weight *= 1.2 # 더 확신 있는 학습
                        
                    self.learner.thompson_sampler.update_distribution(
                        pattern=pattern,
                        success=is_correct,
                        profit_pct=final_profit if is_correct else -abs(final_profit),
                        weight=weight
                    )
                    
                    if seq_reason:
                        print(f"      - {interval}: {seq_reason}")
                
                # 6. 🆕 인터벌 가중치 업데이트 (DB 저장)
                self._update_interval_weights(coin, correct_intervals, wrong_intervals)
                
                # 로그 출력
                move_type = "폭등" if is_bullish_move else "폭락"
                correct_str = ', '.join(correct_intervals) if correct_intervals else '없음'
                wrong_str = ', '.join(wrong_intervals) if wrong_intervals else '없음'
                
                if is_bullish_move:
                    print(f"   🧠 [{coin}] {move_type} +{final_profit:.1f}% | ✅맞춤: {correct_str} | ❌틀림: {wrong_str}")
                    opportunity_count += 1
                else:
                    print(f"   🧠 [{coin}] {move_type} {final_profit:.1f}% | ✅회피: {correct_str} | ❌예측실패: {wrong_str}")
                    avoidance_count += 1

            # 7. 🆕 인터벌별 성적표 출력
            if interval_results:
                print(f"\n   📊 [인터벌별 예측 성적표]")
                print(f"   ┌──────────┬───────────┬───────────┐")
                print(f"   │ 인터벌   │  적중률   │  샘플 수  │")
                print(f"   ├──────────┼───────────┼───────────┤")
                for interval in ANALYSIS_INTERVALS:
                    if interval in interval_results:
                        stats = interval_results[interval]
                        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        print(f"   │ {interval:<8} │ {accuracy:>8.1f}% │ {stats['total']:>8}건 │")
                print(f"   └──────────┴───────────┴───────────┘")
            
            # 결과 요약
            if opportunity_count > 0 or avoidance_count > 0:
                print(f"\n   📈 [놓친 기회 학습] {opportunity_count}건의 폭등 패턴 (전 인터벌 학습)")
                print(f"   📉 [관망 잘함 학습] {avoidance_count}건의 폭락 회피 (전 인터벌 학습)")
                print(f"   ✅ 총 {opportunity_count + avoidance_count}건 × {len(ANALYSIS_INTERVALS)}개 인터벌 = {(opportunity_count + avoidance_count) * len(ANALYSIS_INTERVALS)}건 지식 습득")
            else:
                print(f"   ℹ️ 최근 {self.lookback_hours}시간 내 ±5% 이상 변동한 미보유 코인 없음")

        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"   ⚠️ 정밀 인사이트 분석 오류: {e}")
                import traceback
                traceback.print_exc()

    def _load_all_interval_signals(self, coin: str, timestamp: int) -> Dict[str, Dict]:
        """🆕 특정 시점에 모든 인터벌의 시그널을 로드 (방향 B 핵심)"""
        signals = {}
        try:
            with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=True) as conn:
                for interval in ANALYSIS_INTERVALS:
                    # 🆕 시그널 검색 범위 확장: timestamp 이전 1시간 이내의 시그널 허용
                    time_window = 3600  # 1시간
                    query = """
                        SELECT * FROM signals 
                        WHERE coin = ? AND interval = ? AND timestamp <= ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT 1
                    """
                    df = pd.read_sql(query, conn, params=(coin, interval, timestamp, timestamp - time_window))
                    
                    if not df.empty:
                        row = df.iloc[0]
                        signals[interval] = {
                            'signal_score': safe_float(row.get('signal_score', 0.0)),
                            'confidence': safe_float(row.get('confidence', 0.5)),
                            'integrated_direction': row.get('integrated_direction', 'neutral'),
                            'rsi': safe_float(row.get('rsi', 50.0)),
                            'macd': safe_float(row.get('macd', 0.0)),
                            'volume_ratio': safe_float(row.get('volume_ratio', 1.0)),
                            'pattern_type': row.get('pattern_type', 'none'),
                            'timestamp': int(row.get('timestamp', 0))
                        }
        except Exception as e:
            if "unable to open" not in str(e).lower() and "locked" not in str(e).lower():
                print(f"   ⚠️ 시그널 로드 오류 ({coin}): {e}")
        return signals

    def _update_interval_weights(self, coin: str, correct_intervals: List[str], wrong_intervals: List[str]):
        """🆕 코인별 인터벌 가중치 업데이트 (정확한 인터벌은 가중치 UP)"""
        try:
            with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=False) as conn:
                # 테이블 생성 (없으면)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS coin_interval_weights (
                        coin TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        weight REAL DEFAULT 1.0,
                        correct_count INTEGER DEFAULT 0,
                        total_count INTEGER DEFAULT 0,
                        last_updated INTEGER,
                        PRIMARY KEY (coin, interval)
                    )
                """)
                
                current_ts = int(time.time())
                
                # 정확한 인터벌: 가중치 증가
                for interval in correct_intervals:
                    conn.execute("""
                        INSERT INTO coin_interval_weights (coin, interval, weight, correct_count, total_count, last_updated)
                        VALUES (?, ?, 1.05, 1, 1, ?)
                        ON CONFLICT(coin, interval) DO UPDATE SET
                            weight = MIN(2.0, weight * 1.02),
                            correct_count = correct_count + 1,
                            total_count = total_count + 1,
                            last_updated = ?
                    """, (coin, interval, current_ts, current_ts))
                
                # 틀린 인터벌: 가중치 감소
                for interval in wrong_intervals:
                    conn.execute("""
                        INSERT INTO coin_interval_weights (coin, interval, weight, correct_count, total_count, last_updated)
                        VALUES (?, ?, 0.95, 0, 1, ?)
                        ON CONFLICT(coin, interval) DO UPDATE SET
                            weight = MAX(0.5, weight * 0.98),
                            total_count = total_count + 1,
                            last_updated = ?
                    """, (coin, interval, current_ts, current_ts))
                
                conn.commit()
        except:
            pass  # 가중치 업데이트 실패는 무시

    def _load_historical_signal(self, coin: str, timestamp: int):
        """특정 시점의 정밀 시그널 정보를 DB에서 복원 (레거시 호환)"""
        try:
            from trade.core.models import SignalInfo, SignalAction
            with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=True) as conn:
                query = """
                    SELECT * FROM signals 
                    WHERE coin = ? AND timestamp <= ? 
                    ORDER BY timestamp DESC LIMIT 1
                """
                df = pd.read_sql(query, conn, params=(coin, timestamp))
                
                if not df.empty:
                    row = df.iloc[0]
                    sig = SignalInfo(
                        coin=row['coin'],
                        interval=row['interval'],
                        action=SignalAction.BUY,
                        signal_score=safe_float(row['signal_score']),
                        confidence=safe_float(row['confidence']),
                        reason=row.get('reason', ''),
                        timestamp=int(row['timestamp'])
                    )
                    sig.rsi = safe_float(row.get('rsi', 50.0))
                    sig.volume_ratio = safe_float(row.get('volume_ratio', 1.0))
                    sig.macd = safe_float(row.get('macd', 0.0))
                    sig.pattern_type = row.get('pattern_type', 'none')
                    sig.integrated_direction = row.get('integrated_direction', 'neutral')
                    return sig
        except:
            pass
        return None

    def _get_held_or_traded_coins(self, since_timestamp: int) -> Set[str]:
        """🔧 [수정] 현재 보유 중인 코인만 반환 (최근 거래 코인은 제외하지 않음)
        
        기존: virtual_trade_history에서 최근 거래한 코인도 모두 제외 → 분석 대상 과도하게 축소
        수정: virtual_positions에서 현재 보유 중인 코인만 제외
        """
        held_coins = set()
        try:
            with get_db_connection(TRADING_SYSTEM_DB_PATH) as conn:
                cursor = conn.cursor()
                # 🔧 현재 보유 중인 코인만 조회 (virtual_trade_history 제외)
                cursor.execute("SELECT DISTINCT coin FROM virtual_positions")
                for row in cursor.fetchall():
                    held_coins.add(row[0])
        except:
            pass
        return held_coins

    def get_learned_interval_weights(self, coin: str) -> Dict[str, float]:
        """🆕 학습된 코인별 인터벌 가중치 조회 (signal_selector에서 활용)
        
        Returns:
            {'15m': 1.15, '30m': 0.95, '240m': 1.05, '1d': 1.00}
        """
        weights = {iv: 1.0 for iv in ANALYSIS_INTERVALS}  # 기본값
        try:
            with get_db_connection(TRADING_SYSTEM_DB_PATH, read_only=True) as conn:
                query = """
                    SELECT interval, weight FROM coin_interval_weights
                    WHERE coin = ? AND total_count >= 3
                """
                df = pd.read_sql(query, conn, params=(coin,))
                for _, row in df.iterrows():
                    weights[row['interval']] = row['weight']
        except:
            pass
        return weights

    def _get_recent_candles_at_ts(self, coin: str, interval: str, timestamp: int, count: int = 5) -> Optional[pd.DataFrame]:
        """🆕 특정 시점 기준 최근 N개의 캔들 데이터 조회 (Sequence 분석용)"""
        try:
            from trade.core.database import CANDLES_DB_PATH
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume, rsi
                    FROM candles 
                    WHERE symbol = ? AND interval = ? AND timestamp <= ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(coin, interval, timestamp, count))
                return df if not df.empty else None
        except Exception as e:
            # symbol -> coin 마이그레이션 대응
            try:
                from trade.core.database import CANDLES_DB_PATH
                with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
                    query = """
                        SELECT timestamp, open, high, low, close, volume, rsi
                        FROM candles 
                        WHERE coin = ? AND interval = ? AND timestamp <= ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """
                    df = pd.read_sql(query, conn, params=(coin, interval, timestamp, count))
                    return df if not df.empty else None
            except:
                return None

    def _cleanup_old_insights(self):
        current_time = int(time.time())
        cutoff_id = (current_time - (24 * 3600)) // 3600
        # 🆕 [Fix] '_v2' 접미사 대응: 끝에서 두 번째 요소를 숫자로 추출
        new_processed = set()
        for i in self.processed_insights:
            parts = i.split('_')
            try:
                # {coin}_{ts}_v2 형식에서 ts 추출
                ts_part = parts[-2] if parts[-1] == 'v2' else parts[-1]
                if int(ts_part) >= cutoff_id:
                    new_processed.add(i)
            except (ValueError, IndexError):
                continue
        self.processed_insights = new_processed
