#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 통합 의사결정 엔진
가상 매매와 실전 매매에서 동일한 AI 판단 로직을 제공합니다.
"""

import os
import time
import sqlite3
import json
from typing import Dict, Any, Optional

class TradingAIDecisionEngine:
    """🛡️ 알파 가디언 (통합 의사결정 엔진)"""
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.decision_history = []
        self.coin_decision_patterns = {}
        self.market_adaptations = {}
        # 🆕 알파 가디언의 '자가 반성' 바이어스 (초기값: 학습을 위해 더 공격적으로 설정)
        self.meta_bias = {
            'buy_threshold_offset': -0.05,  # 초기값: 더 공격적으로 매수 (학습 데이터 확보)
            'sell_threshold_offset': 0.0,
            'risk_weight_multiplier': 1.0,
            'confidence_threshold': 0.15    # 🆕 신뢰도 문턱값도 성격(Bias)으로 관리
        }
        # 🆕 시장 상황별 meta_bias (시장 상황별 학습된 성격)
        self.meta_bias_by_market = {}  # {market_type: {buy_threshold_offset: ..., ...}}
        if self.db_path:
            self._create_tables()
            self._load_meta_bias()
            self._load_meta_bias_by_market()
        
    def _create_tables(self):
        """바이어스 저장용 테이블 생성 (안정성 강화)"""
        # 🚀 [Fix] DB 경로가 없으면 스킵
        if not self.db_path:
            return
            
        try:
            from trade.core.database import get_db_connection
            with get_db_connection(self.db_path, read_only=False) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS guardian_bias (
                        key TEXT PRIMARY KEY,
                        value REAL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # 🆕 시장 상황별 바이어스 저장용 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS guardian_bias_by_market (
                        market_type TEXT PRIMARY KEY,
                        bias_json TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception: pass

    def _load_meta_bias(self):
        """DB에서 알파 가디언의 최신 '성격' 로드 (안정성 강화)"""
        # 🚀 [Fix] DB 경로가 없거나 파일이 없으면 조용히 스킵
        if not self.db_path or not os.path.exists(self.db_path):
            return
            
        try:
            from trade.core.database import get_db_connection
            with get_db_connection(self.db_path, read_only=True, timeout=30.0) as conn:
                cursor = conn.cursor()
                # 🚀 [Fix] 테이블 존재 여부 먼저 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='guardian_bias'")
                if not cursor.fetchone():
                    return
                    
                cursor.execute("SELECT key, value FROM guardian_bias")
                for key, value in cursor.fetchall():
                    if key in self.meta_bias:
                        self.meta_bias[key] = value
                
                # 🆕 소수점 정리 후 출력
                b = self.meta_bias
                print(f"🛡️ [알파 가디언] 전역 성격 로드 완료")
                print(f"   └ 🛒 매수성향: {b.get('buy_threshold_offset', 0):+.2f} | 💰 매도성향: {b.get('sell_threshold_offset', 0):+.2f} | ⚠️ 리스크감도: {b.get('risk_weight_multiplier', 1.0):.2f}x")
        except Exception: pass

    def save_meta_bias(self, new_bias: dict):
        """새로운 성격을 DB에 저장 (안정성 강화)"""
        try:
            # 🆕 메타데이터 필드 필터링 (SQLite에 저장 가능한 필드만)
            valid_bias_keys = ['buy_threshold_offset', 'sell_threshold_offset', 'risk_weight_multiplier', 'confidence_threshold']
            filtered_bias = {k: v for k, v in new_bias.items() if k in valid_bias_keys}
            
            self.meta_bias.update(filtered_bias)
            from trade.core.database import get_db_connection
            with get_db_connection(self.db_path, timeout=60.0) as conn:
                for key, value in self.meta_bias.items():
                    # 🆕 타입 검증: 숫자만 저장 (리스트 등 제외)
                    if isinstance(value, (int, float)):
                        conn.execute("""
                            INSERT OR REPLACE INTO guardian_bias (key, value, updated_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        """, (key, value))
                conn.commit()
        except Exception:
            # 🔇 엔진 모드: 저장 실패 조용히 처리 (다음 턴에 재시도)
            pass
    
    def _classify_market_context(self, market_context: dict) -> str:
        """
        시장 상황을 분류하여 학습 키 생성
        
        Returns:
            market_type: 'extreme_bearish', 'bearish', 'sideways_bearish', 
                       'neutral', 'sideways_bullish', 'bullish', 'extreme_bullish'
        """
        try:
            # 1. regime 우선 확인 (가장 정확한 분류)
            regime = market_context.get('regime', '').lower()
            if regime:
                # 7개 레짐 체계 지원
                valid_regimes = [
                    'extreme_bearish', 'bearish', 'sideways_bearish',
                    'neutral', 'sideways_bullish', 'bullish', 'extreme_bullish'
                ]
                if regime in valid_regimes:
                    return regime
            
            # 2. trend + volatility 조합으로 분류 (fallback)
            trend = market_context.get('trend', 'neutral').lower()
            volatility = market_context.get('volatility', 'medium')
            score = market_context.get('score', 0.5)
            
            # score 기반 분류 (0.0 ~ 1.0)
            if score >= 0.85:
                return 'extreme_bullish'
            elif score >= 0.70:
                return 'bullish'
            elif score >= 0.55:
                return 'sideways_bullish'
            elif score >= 0.45:
                return 'neutral'
            elif score >= 0.30:
                return 'sideways_bearish'
            elif score >= 0.15:
                return 'bearish'
            else:
                return 'extreme_bearish'
                
        except Exception as e:
            print(f"⚠️ 시장 상황 분류 오류: {e}")
            return 'neutral'
    
    def _load_meta_bias_by_market(self):
        """DB에서 시장 상황별 '성격' 로드 (안정성 강화)"""
        # 🚀 [Fix] DB 경로가 없거나 파일이 없으면 조용히 스킵
        if not self.db_path or not os.path.exists(self.db_path):
            return
            
        try:
            from trade.core.database import get_db_connection
            with get_db_connection(self.db_path, read_only=True, timeout=30.0) as conn:
                cursor = conn.cursor()
                # 🚀 [Fix] 테이블 존재 여부 먼저 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='guardian_bias_by_market'")
                if not cursor.fetchone():
                    return
                    
                cursor.execute("SELECT market_type, bias_json FROM guardian_bias_by_market")
                for market_type, bias_json in cursor.fetchall():
                    try:
                        bias_dict = json.loads(bias_json)
                        self.meta_bias_by_market[market_type] = bias_dict
                    except Exception:
                        pass  # 파싱 오류 무시
                if self.meta_bias_by_market:
                    print(f"🛡️ [알파 가디언] 시장 상황별 특수 성격 로드 완료 ({len(self.meta_bias_by_market)}개 레짐)")
        except Exception:
            # 🚀 [Fix] 모든 DB 접근 오류 조용히 처리 (선택적 데이터이므로 연산 계속)
            pass
    
    def save_meta_bias_by_market(self, market_type: str, new_bias: dict):
        """시장 상황별 새로운 성격을 DB에 저장 (안정성 강화)"""
        try:
            # 🆕 메타데이터 필드 필터링 (SQLite에 저장 가능한 필드만)
            valid_bias_keys = ['buy_threshold_offset', 'sell_threshold_offset', 'risk_weight_multiplier', 'confidence_threshold']
            filtered_bias = {k: v for k, v in new_bias.items() if k in valid_bias_keys}
            
            # 기존 바이어스와 병합 (없으면 기본값 사용)
            default_bias = {
                'buy_threshold_offset': -0.05,
                'sell_threshold_offset': 0.0,
                'risk_weight_multiplier': 1.0,
                'confidence_threshold': 0.15
            }
            
            if market_type in self.meta_bias_by_market:
                default_bias.update(self.meta_bias_by_market[market_type])
            
            default_bias.update(filtered_bias)
            self.meta_bias_by_market[market_type] = default_bias
            
            from trade.core.database import get_db_connection
            with get_db_connection(self.db_path, timeout=60.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO guardian_bias_by_market 
                    (market_type, bias_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (market_type, json.dumps(default_bias)))
                conn.commit()
        except Exception:
            # 🔇 엔진 모드: 저장 실패 조용히 처리 (다음 턴에 재시도)
            pass
    
    def get_market_specific_bias(self, market_context: dict) -> dict:
        """
        현재 시장 상황에 맞는 meta_bias 반환
        
        Returns:
            시장 상황별 meta_bias (없으면 전역 meta_bias 반환)
        """
        try:
            market_type = self._classify_market_context(market_context)
            if market_type in self.meta_bias_by_market:
                return self.meta_bias_by_market[market_type]
            else:
                # 시장 상황별 바이어스가 없으면 전역 meta_bias 반환
                return self.meta_bias.copy()
        except Exception as e:
            print(f"⚠️ 시장 상황별 바이어스 조회 오류: {e}")
            return self.meta_bias.copy()

    def make_trading_decision(self, signal_data: dict, current_price: float, 
                            market_context: dict, coin_performance: dict) -> Dict[str, Any]:
        """🛡️ 알파 가디언의 통합 의사결정 (시그널 + 성과 + 시장 상황 + 리스크 + 🆕자가 반성)"""
        try:
            # 🆕 signal_data가 객체인 경우 dict로 변환하거나 getattr 사용
            def get_val(data, key, default=None):
                if isinstance(data, dict):
                    return data.get(key, default)
                return getattr(data, key, default)

            # 기본 시그널 분석
            signal_score = get_val(signal_data, 'signal_score', 0.0)
            confidence = get_val(signal_data, 'confidence', 0.0)
            
            # 🆕 시장 상황별 meta_bias 조회 (학습된 시장 상황별 성격)
            market_specific_bias = self.get_market_specific_bias(market_context)
            market_type = self._classify_market_context(market_context)
            
            # 1. 코인별 성과 기반 조정
            coin_bonus = self._calculate_coin_performance_bonus(coin_performance)
            
            # 2. 시장 컨텍스트 기반 조정 (🆕 하드코딩 제거 - 시장 상황별 meta_bias로 대체)
            # 시장 상황별 meta_bias가 학습되면 하드코딩된 보너스는 불필요
            # 초기 학습 단계에서는 작은 가중치로 유지 (점진적 제거)
            market_bonus = self._calculate_market_context_bonus(market_context) * 0.3  # 70% 감소
            
            # 3. 리스크 조정 (🆕시장 상황별 학습된 민감도 반영)
            # 🚀 [Refactor] 리스크 계산 시 시장 컨텍스트 전달 (Regime-Adaptive Risk)
            risk_adjustment = self._calculate_risk_adjustment(signal_data, current_price, market_context) * market_specific_bias['risk_weight_multiplier']
            
            # 4. 최종 의사결정 점수 계산
            # 🆕 시장 상황별 학습된 바이어스는 임계값에 반영되므로, market_bonus는 점진적으로 제거
            final_score = signal_score + coin_bonus + market_bonus - risk_adjustment
            
            # 5. 액션 결정 (🆕시장 상황별 학습된 바이어스 적용)
            # 🆕 학습을 위해 임계값 완화: buy_threshold를 낮추고, confidence 조건도 완화
            buy_threshold = 0.25 + market_specific_bias['buy_threshold_offset']  # 시장 상황별 학습된 오프셋
            sell_threshold = -0.25 + market_specific_bias['sell_threshold_offset']  # 시장 상황별 학습된 오프셋
            
            # 🆕 [시스템화] 하드코딩 제거: 시장 상황별/자가학습된 신뢰도 문턱값 적용
            min_confidence = market_specific_bias.get('confidence_threshold', 0.15)
            
            # 결정 및 근거 생성
            coin_regime = get_val(signal_data, 'wave_phase', 'unknown')
            coin_direction = get_val(signal_data, 'integrated_direction', 'neutral')
            
            # 🆕 가독성을 위해 영어 용어를 한국어로 변환
            regime_map = {
                'consolidation': '박스권/횡보',
                'impulse': '강한추세',
                'correction': '조정/반등',
                'expansion': '확산/변동',
                'reversal': '반전',
                'unknown': '정보부족'
            }
            coin_regime_kr = regime_map.get(coin_regime.lower(), coin_regime)
            
            if final_score > buy_threshold and confidence > min_confidence:
                decision = 'buy'
                reason = f"시그널 {signal_score:.3f} + 성과 {coin_bonus:.3f} + 시장 {market_bonus:.3f} - 리스크 {risk_adjustment:.3f} = {final_score:.3f} (임계값 {buy_threshold:.3f} 초과, 코인: {coin_regime_kr}/{coin_direction}, 시장: {market_type})"
            elif final_score < sell_threshold and confidence > min_confidence:
                decision = 'sell'
                reason = f"시그널 {signal_score:.3f} + 성과 {coin_bonus:.3f} + 시장 {market_bonus:.3f} - 리스크 {risk_adjustment:.3f} = {final_score:.3f} (임계값 {sell_threshold:.3f} 미만, 코인: {coin_regime_kr}/{coin_direction}, 시장: {market_type})"
            else:
                decision = 'hold'
                reason = f"시그널 {signal_score:.3f} + 성과 {coin_bonus:.3f} + 시장 {market_bonus:.3f} - 리스크 {risk_adjustment:.3f} = {final_score:.3f} (임계값 범위 내, 코인: {coin_regime_kr}/{coin_direction}, 시장: {market_type})"
            
            return {
                'decision': decision,
                'final_score': final_score,
                'reason': reason
            }
                
        except Exception as e:
            print(f"⚠️ AI 의사결정 오류: {e}")
            return {
                'decision': 'hold',
                'final_score': 0.0,
                'reason': f'알파 가디언 분석 오류: {str(e)}'
            }

    def _calculate_coin_performance_bonus(self, coin_performance: dict) -> float:
        """🆕 수익비(Profit Factor) 기반 성과 보너스 계산 (승률 보조)"""
        try:
            # 1️⃣ [우선] 전달받은 coin_performance에서 전체 성과 확인
            success_rate = coin_performance.get('success_rate', 0.5)
            avg_profit = coin_performance.get('avg_profit', 0.0)
            total_trades = coin_performance.get('total_trades', 0)
            profit_factor = coin_performance.get('profit_factor', 1.0) # 기본값 1.0
            
            # 2️⃣ [대안] 코인별 데이터가 없으면 전체 가상매매 성과 조회
            if total_trades == 0:
                global_stats = self._get_global_trading_stats()
                success_rate = global_stats.get('success_rate', 0.5)
                avg_profit = global_stats.get('avg_profit', 0.0)
                total_trades = global_stats.get('total_trades', 0)
                profit_factor = global_stats.get('profit_factor', 1.0)
            
            # 3️⃣ 여전히 데이터가 없으면 중립 반환
            if total_trades == 0:
                return 0.0
            
            # 4️⃣ 수익비(Profit Factor) 기반 보너스 계산 (범위: -0.05 ~ +0.05)
            bonus = 0.0
            
            # 수익비 평가 (가중치 높음)
            if profit_factor >= 2.0:
                bonus += 0.04
            elif profit_factor >= 1.5:
                bonus += 0.02
            elif profit_factor >= 1.2:
                bonus += 0.01
            elif profit_factor < 0.8:
                bonus -= 0.02
            elif profit_factor < 0.5:
                bonus -= 0.04
                
            # 승률 보조 평가 (가중치 낮음 - 추세추종 전략 보완)
            if success_rate >= 0.6:
                bonus += 0.01
            elif success_rate < 0.3: # 승률이 너무 낮으면 페널티
                bonus -= 0.01
            
            # 신뢰도 가중치 (거래 횟수가 적으면 영향력 감소)
            confidence_weight = min(1.0, total_trades / 10.0)
            
            return max(-0.05, min(0.05, bonus * confidence_weight))
            
        except Exception as e:
            return 0.0
    
    def _get_global_trading_stats(self) -> dict:
        """🆕 전체 가상매매 성과 통계 조회 (수익비 포함)"""
        try:
            if not self.db_path or not os.path.exists(self.db_path):
                # print(f"⚠️ [알파 가디언] DB 경로 없음: {self.db_path}")
                return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0, 'profit_factor': 1.0}
            
            from trade.core.database import get_db_connection, TRADING_SYSTEM_DB_PATH
            
            # 🚀 [Fix] self.db_path가 아니라 TRADING_SYSTEM_DB_PATH 사용 (정합성)
            target_db = TRADING_SYSTEM_DB_PATH if TRADING_SYSTEM_DB_PATH else self.db_path
            
            with get_db_connection(target_db, read_only=True) as conn:
                cursor = conn.cursor()
                
                # virtual_trade_history 테이블 존재 여부 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='virtual_trade_history'")
                if not cursor.fetchone():
                    # print("⚠️ [알파 가디언] virtual_trade_history 테이블 없음")
                    return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0, 'profit_factor': 1.0}

                # virtual_trade_history 테이블에서 최근 7일 성과 조회
                # 🚀 [Fix] 날짜 필터 제거 (데이터가 적으면 전체 기간 조회)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN profit_pct > 0 THEN 1 ELSE 0 END) as wins,
                        AVG(profit_pct) as avg_profit,
                        SUM(CASE WHEN profit_pct > 0 THEN profit_pct ELSE 0 END) as gross_profit,
                        ABS(SUM(CASE WHEN profit_pct < 0 THEN profit_pct ELSE 0 END)) as gross_loss
                    FROM virtual_trade_history
                    -- WHERE exit_timestamp > strftime('%s', 'now') - 604800  -- 최근 7일 제한 해제 (데이터 확보 우선)
                """)
                row = cursor.fetchone()
                
                if row and row[0] > 0:
                    total = row[0]
                    wins = row[1] or 0
                    avg_profit = row[2] or 0.0
                    gross_profit = row[3] or 0.0
                    gross_loss = row[4] or 0.0
                    
                    # 수익비 계산 (손실 0이면 무한대 대신 3.0 상한)
                    profit_factor = 3.0 if gross_loss == 0 else (gross_profit / gross_loss)
                    
                    # print(f"📊 [알파 가디언] 전체 성과 로드: {total}건, 승률 {wins/total:.2f}, PF {profit_factor:.2f}")
                    return {
                        'success_rate': wins / total,
                        'avg_profit': avg_profit,
                        'total_trades': total,
                        'profit_factor': profit_factor
                    }
            
            # print("⚠️ [알파 가디언] 거래 이력 없음")
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0, 'profit_factor': 1.0}
        except Exception as e:
            # print(f"⚠️ [알파 가디언] 성과 조회 오류: {e}")
            return {'success_rate': 0.5, 'avg_profit': 0.0, 'total_trades': 0, 'profit_factor': 1.0}
    
    def _calculate_market_context_bonus(self, market_context: dict) -> float:
        """🆕 7개 레짐 기반 시장 보너스 계산 (거래량 상위 40% 코인 추세 반영)"""
        try:
            # 1️⃣ 전달받은 레짐 정보 확인
            regime = market_context.get('regime', 'neutral').lower()
            
            # 2️⃣ 레짐 정보가 없으면 거래량 상위 40% 코인 기반으로 직접 계산
            if regime in ['neutral', 'unknown', '']:
                calculated_regime = self._calculate_market_regime_from_top_coins()
                if calculated_regime:
                    regime = calculated_regime
            
            # 3️⃣ 7개 레짐에 대한 차등 보너스 (작은 점수 차이)
            # 범위: -0.03 ~ +0.03 (기존 -0.05 ~ +0.05에서 축소)
            regime_bonus_map = {
                'extreme_bullish': 0.03,    # 극강세
                'strong_bullish': 0.02,     # 강세
                'bullish': 0.01,            # 상승
                'sideways_bullish': 0.005,  # 약간 상승
                'neutral': 0.0,             # 중립
                'sideways_bearish': -0.005, # 약간 하락
                'bearish': -0.01,           # 하락
                'strong_bearish': -0.02,    # 약세
                'extreme_bearish': -0.03    # 극약세
            }
            
            bonus = regime_bonus_map.get(regime, 0.0)
            
            return bonus
        except Exception:
            return 0.0
    
    def _calculate_market_regime_from_top_coins(self) -> str:
        """🆕 거래량 상위 40% 코인의 추세를 기반으로 시장 레짐 계산"""
        try:
            from trade.core.database import get_db_connection, CANDLES_DB_PATH
            
            if not CANDLES_DB_PATH or not os.path.exists(CANDLES_DB_PATH):
                return 'neutral'
            
            with get_db_connection(CANDLES_DB_PATH, read_only=True) as conn:
                # 거래량 상위 40% 코인의 최근 레짐 분포 조회
                cursor = conn.cursor()
                cursor.execute("""
                    WITH ranked AS (
                        SELECT symbol, regime_label, volume,
                               PERCENT_RANK() OVER (ORDER BY volume DESC) as pct_rank
                        FROM candles
                        WHERE interval = '15m'
                          AND timestamp > strftime('%s', 'now') - 3600
                        GROUP BY symbol
                        HAVING timestamp = MAX(timestamp)
                    )
                    SELECT regime_label, COUNT(*) as cnt
                    FROM ranked
                    WHERE pct_rank <= 0.4
                    GROUP BY regime_label
                    ORDER BY cnt DESC
                """)
                
                rows = cursor.fetchall()
                if not rows:
                    return 'neutral'
                
                # 가장 많은 레짐 반환
                dominant_regime = rows[0][0] if rows[0][0] else 'neutral'
                return dominant_regime.lower()
                
        except Exception:
            return 'neutral'
    
    def _calculate_risk_adjustment(self, signal_data: dict, current_price: float, market_context: dict = None) -> float:
        """🆕 정밀 리스크 조정 (RSI 과열, 변동성, 급등 여부 반영 + 레짐 적응형)"""
        try:
            # 데이터 추출 헬퍼
            def get_val(data, key, default=None):
                if isinstance(data, dict):
                    return data.get(key, default)
                return getattr(data, key, default)

            # 지표 추출
            rsi = float(get_val(signal_data, 'rsi', 50.0) or 50.0)
            volatility = float(get_val(signal_data, 'volatility', 0.0) or 0.0)
            price_momentum = float(get_val(signal_data, 'price_momentum', 0.0) or 0.0)
            wave_phase = get_val(signal_data, 'wave_phase', 'unknown').lower()
            
            # 🆕 [Regime-Adaptive Risk] 7단계 레짐별 리스크 민감도 정의
            market_regime = market_context.get('regime', 'neutral').lower() if market_context else 'neutral'
            
            # 레짐별 RSI/급등 페널티 맵 (RSI>80, 급등>20%)
            regime_risk_map = {
                'extreme_bullish':  {'rsi': 0.00, 'pump': 0.01}, # 불장: 과열 용인
                'strong_bullish':   {'rsi': 0.01, 'pump': 0.02},
                'bullish':          {'rsi': 0.02, 'pump': 0.03},
                'sideways_bullish': {'rsi': 0.03, 'pump': 0.05}, # 여기서부터 급등 주의
                'neutral':          {'rsi': 0.05, 'pump': 0.05}, # 횡보장: 급등은 곧 하락
                'sideways_bearish': {'rsi': 0.05, 'pump': 0.05},
                'bearish':          {'rsi': 0.06, 'pump': 0.08},
                'strong_bearish':   {'rsi': 0.07, 'pump': 0.09},
                'extreme_bearish':  {'rsi': 0.08, 'pump': 0.10}  # 하락장: 급등은 설거지
            }
            
            # 현재 레짐에 맞는 리스크 설정 가져오기 (기본값: neutral)
            risk_params = regime_risk_map.get(market_regime, regime_risk_map['neutral'])
            
            # 해당 코인의 개별 국면이 Impulse(강한추세)라면 한 단계 더 완화
            if wave_phase in ['impulse', 'expansion']:
                risk_params['rsi'] = max(0.0, risk_params['rsi'] - 0.01)
                risk_params['pump'] = max(0.0, risk_params['pump'] - 0.01)

            adjustment = 0.0
            
            # 1️⃣ RSI 과열/과매도 (역추세 리스크 관리)
            if rsi >= 80:
                adjustment += risk_params['rsi']  # 레짐별 차등 적용
            elif rsi <= 30:
                adjustment -= 0.02  # 과매도 구간 (기회)
            
            # 2️⃣ 변동성 폭주 (예측 불가능성)
            if volatility >= 10.0:
                adjustment += 0.03
            
            # 3️⃣ 이미 폭등한 코인 (뒷북 방지)
            if price_momentum >= 20.0:
                adjustment += risk_params['pump']  # 레짐별 차등 적용
            
            return adjustment
            
        except Exception:
            return 0.0

# 글로벌 싱글톤 인스턴스 관리
_ai_engine = None

def get_ai_decision_engine(db_path: str = None):
    global _ai_engine
    if _ai_engine is None:
        if db_path is None:
            # 🆕 trade.core.database에서 중앙화된 경로 로드 (정합성 유지)
            try:
                from trade.core.database import STRATEGY_DB_PATH
                db_path = STRATEGY_DB_PATH
            except ImportError:
                import os
                _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                db_path = os.path.join(_BASE_DIR, 'market', 'coin_market', 'data_storage', 'learning_strategies', 'common_strategies.db')
        
        _ai_engine = TradingAIDecisionEngine(db_path=db_path)
    return _ai_engine
