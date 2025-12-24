"""
시장 분석 엔진 (Core Market)
- 가상/실전 매매에서 공통으로 사용하는 시장 분석 로직
- 트렌드, 변동성, 거래량 분석 등
- 3-Layer (Short/Mid/Long) 동적 시장 레짐 분석
"""
import time
import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional

class MarketAnalyzer:
    """시장 분석기 - 시장 상황 실시간 분석 (Centralized)"""
    def __init__(self, db_path: str = None, candle_db_path: str = None):
        self.market_conditions = {}
        # DB 경로 설정 (환경변수 우선)
        self.db_path = db_path or os.getenv('TRADING_DB_PATH')
        if not self.db_path:
             # 기본값: trade/../market/coin_market/data_storage/trading_system.db
             base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             self.db_path = os.path.join(base_dir, 'market', 'coin_market', 'data_storage', 'trading_system.db')
        
        # 🆕 캔들 DB 경로 (거래량 조회용)
        self.candle_db_path = candle_db_path or os.getenv('RL_DB_PATH')
        if not self.candle_db_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.candle_db_path = os.path.join(base_dir, 'market', 'coin_market', 'data_storage', 'trade_candles.db')
        
        # 🆕 상위 코인 캐시 (5분 유효)
        self._top_coins_cache = {'coins': [], 'ts': 0, 'total': 0}
        
        # 🆕 시장 분석 대상 비율 (40% = 상위 40% 코인만 분석)
        self.market_analysis_ratio = 0.40
    
    def _get_top_volume_coins(self, ratio: float = None) -> list:
        """🆕 거래량 상위 N% 코인 조회 (5분 캐시, 비율 기반)
        
        Args:
            ratio: 상위 비율 (기본값: self.market_analysis_ratio = 0.40)
                   예) 0.40 = 상위 40%
        
        Returns:
            거래량 상위 N% 코인 심볼 리스트
        """
        try:
            if ratio is None:
                ratio = self.market_analysis_ratio
            
            current_time = time.time()
            # 캐시 유효 시 반환
            if self._top_coins_cache['coins'] and (current_time - self._top_coins_cache['ts'] < 300):
                return self._top_coins_cache['coins']
            
            with sqlite3.connect(self.candle_db_path) as conn:
                # 1. 전체 코인 수 조회
                total_query = """
                    SELECT COUNT(DISTINCT symbol) as cnt FROM candles
                    WHERE interval='1d' AND timestamp=(SELECT MAX(timestamp) FROM candles WHERE interval='1d')
                """
                total_df = pd.read_sql(total_query, conn)
                total_coins = total_df['cnt'].iloc[0] if not total_df.empty else 0
                
                # 2. 상위 N% 계산 (최소 50개, 최대 500개)
                target_count = int(total_coins * ratio)
                target_count = max(50, min(target_count, 500))
                
                # 3. 거래량 상위 코인 조회
                query = """
                    SELECT symbol FROM candles
                    WHERE interval='1d' AND timestamp=(SELECT MAX(timestamp) FROM candles WHERE interval='1d')
                    ORDER BY volume DESC
                    LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(target_count,))
                coins = df['symbol'].tolist() if not df.empty else []
                
                # 캐시 갱신
                self._top_coins_cache = {'coins': coins, 'ts': current_time, 'total': total_coins}
                
                # 로그 (5분에 한번)
                if current_time % 300 < 60:
                    print(f"📊 시장 분석 대상: 전체 {total_coins}개 중 상위 {len(coins)}개 ({ratio*100:.0f}%)")
                
                return coins
        except Exception as e:
            print(f"⚠️ 거래량 상위 코인 조회 오류: {e}")
            return []
        
    def analyze_market_regime(self) -> Dict:
        """전체 시장 상황 정밀 분석 (🆕 거래량 상위 40% 코인 기준, 동적 인터벌 4-Layer 분석)"""
        try:
            # 🆕 거래량 상위 40% 코인만 분석 (비율 기반, 유동적)
            top_coins = self._get_top_volume_coins()
            
            with sqlite3.connect(self.db_path) as conn:
                current_time = int(datetime.now().timestamp())
                
                # 1. 현재 DB에 존재하는 모든 인터벌 조회
                try:
                    intervals_df = pd.read_sql("SELECT DISTINCT interval FROM signals", conn)
                except Exception:
                    # 테이블이 없거나 오류 시 기본값
                    return {'score': 0.5, 'regime': 'Neutral', 'volatility': 0.0}

                if intervals_df.empty:
                    return {'score': 0.5, 'regime': 'Neutral', 'volatility': 0.0}
                
                avail_intervals = intervals_df['interval'].tolist()
                
                # 2. 인터벌 시간 순 정렬 (분 단위 변환)
                def get_minutes(iv):
                    iv = str(iv).lower()
                    if iv == 'combined': return 0 # combined는 특수 취급
                    try:
                        if iv.endswith('m'): return int(iv[:-1])
                        if iv.endswith('h'): return int(iv[:-1]) * 60
                        if iv.endswith('d'): return int(iv[:-1]) * 1440
                        if iv.endswith('w'): return int(iv[:-1]) * 10080
                    except: pass
                    return 999999
                
                # combined 제외하고 시간순 정렬
                sorted_intervals = sorted([i for i in avail_intervals if i != 'combined'], key=get_minutes)
                
                if not sorted_intervals:
                    return {'score': 0.5, 'regime': 'Neutral', 'volatility': 0.0}
                
                # 3. 동적 그룹핑 (Short / Mid / Long / Super Long) - 4 Layer
                n = len(sorted_intervals)
                if n == 1:
                    short_ivs = mid_ivs = long_ivs = super_long_ivs = sorted_intervals
                elif n == 2:
                    short_ivs = [sorted_intervals[0]]
                    mid_ivs = [sorted_intervals[0]]
                    long_ivs = [sorted_intervals[1]]
                    super_long_ivs = [sorted_intervals[1]]
                elif n == 3:
                    short_ivs = [sorted_intervals[0]]
                    mid_ivs = [sorted_intervals[1]]
                    long_ivs = [sorted_intervals[2]]
                    super_long_ivs = [sorted_intervals[2]]
                else:
                    # 4등분 (가용 인터벌이 많을 경우 적절히 배분)
                    # 예: 15m, 30m, 60m, 240m, 1d -> 
                    # S: 15m, 30m
                    # M: 60m
                    # L: 240m
                    # SL: 1d
                    
                    # 간단하게 인덱스 기반 분할
                    p1 = max(1, n // 4)
                    p2 = max(2, 2 * n // 4)
                    p3 = max(3, 3 * n // 4)
                    
                    short_ivs = sorted_intervals[:p1]
                    mid_ivs = sorted_intervals[p1:p2]
                    long_ivs = sorted_intervals[p2:p3]
                    super_long_ivs = sorted_intervals[p3:]
                
                # 4. 데이터 조회 (🆕 거래량 상위 40% 코인만, 최대 24시간)
                interval_placeholders = ', '.join(['?'] * (len(sorted_intervals) + 1))
                
                # 🆕 상위 코인 필터 추가
                if top_coins:
                    coin_placeholders = ', '.join(['?'] * len(top_coins))
                    coin_filter = f"AND coin IN ({coin_placeholders})"
                    params = [current_time - 86400] + sorted_intervals + ['combined'] + top_coins
                else:
                    # 상위 코인 조회 실패 시 전체 분석 (fallback)
                    coin_filter = ""
                    params = [current_time - 86400] + sorted_intervals + ['combined']
                
                df = pd.read_sql(f"""
                    SELECT coin, interval, signal_score, volatility, timestamp
                    FROM signals 
                    WHERE timestamp > ? 
                    AND interval IN ({interval_placeholders})
                    {coin_filter}
                    ORDER BY timestamp DESC
                """, conn, params=params)
                
                if df.empty:
                    return {'score': 0.5, 'regime': 'Neutral', 'volatility': 0.0}
                
                # 🆕 분석 대상 코인 수 로그 (디버깅용, 가끔만)
                if current_time % 300 < 60:  # 5분마다 한번씩
                    unique_coins = df['coin'].nunique()
                    print(f"📊 시장 분석 대상: 거래량 상위 {unique_coins}개 코인")

                # 5. 각 그룹별 점수 계산
                # Short: 최근 2시간 (7200초) - 초단기 반응
                short_mask = (df['interval'].isin(short_ivs)) & (df['timestamp'] > current_time - 7200)
                df_short = df[short_mask].groupby('coin').first()
                
                # Mid: 최근 6시간 (21600초) - 반나절 흐름
                mid_mask = (df['interval'].isin(mid_ivs)) & (df['timestamp'] > current_time - 21600)
                df_mid = df[mid_mask].groupby('coin').first()

                # Long: 최근 12시간 (43200초) - 반나절~하루
                long_mask = (df['interval'].isin(long_ivs)) & (df['timestamp'] > current_time - 43200)
                df_long = df[long_mask].groupby('coin').first()

                # Super Long: 최근 24시간 (86400초) - 하루 전체
                sl_mask = (df['interval'].isin(super_long_ivs))
                df_sl = df[sl_mask].groupby('coin').first()

                # 점수 계산 (계층적 폴백)
                sl_score = df_sl['signal_score'].mean() if not df_sl.empty else 0.0
                long_score = df_long['signal_score'].mean() if not df_long.empty else sl_score
                mid_score = df_mid['signal_score'].mean() if not df_mid.empty else long_score
                short_score = df_short['signal_score'].mean() if not df_short.empty else mid_score
                
                avg_volatility = df['volatility'].mean() if not df.empty else 0.02

                # 6. [4-Layer 가중 평균] (Short 50% / Mid 30% / Long 15% / S.Long 5%)
                final_score = (short_score * 0.50) + (mid_score * 0.30) + (long_score * 0.15) + (sl_score * 0.05)
                
                # 레짐 분류
                market_regime = "Neutral"
                if final_score >= 0.5: market_regime = "Extreme Bullish"
                elif final_score >= 0.2: market_regime = "Bullish"
                elif final_score >= 0.05: market_regime = "Sideways Bullish"
                elif final_score > -0.05: market_regime = "Neutral"
                elif final_score > -0.2: market_regime = "Sideways Bearish"
                elif final_score > -0.5: market_regime = "Bearish"
                else: market_regime = "Extreme Bearish"
                
                if avg_volatility > 0.05:
                    market_regime += " (High Vol)"

                normalized_score = (final_score + 1) / 2
                
                return {
                    'score': normalized_score,
                    'regime': market_regime,
                    'volatility': avg_volatility,
                    'raw_score': final_score,
                    'details': {'sl': sl_score, 'long': long_score, 'mid': mid_score, 'short': short_score}
                }
                
        except Exception as e:
            print(f"⚠️ 시장 상황 분석 오류 (Core): {e}")
            return {'score': 0.5, 'regime': 'Neutral', 'volatility': 0.0}

    def analyze_market_condition(self, coin: str, interval: str) -> dict:
        """시장 상황 분석 (기본 구현)"""
        try:
            # 기본 시장 상황
            market_condition = {
                'trend': 'neutral',
                'volatility': 0.02,
                'volume_trend': 'normal',
                'momentum': 'neutral',
                'timestamp': int(time.time())
            }
            
            # 코인별 시장 상황 업데이트
            key = f"{coin}_{interval}"
            self.market_conditions[key] = market_condition
            
            return market_condition
            
        except Exception as e:
            print(f"⚠️ 시장 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02, 'timestamp': int(time.time())}
    
    def get_market_context_from_signal(self, btc_signal) -> Dict:
        """BTC 시그널 기반 전체 시장 컨텍스트 분석"""
        try:
            if not btc_signal:
                return {'trend': 'neutral', 'volatility': 0.02}
                
            signal_score = btc_signal.signal_score
            
            if signal_score > 0.3:
                trend = 'bullish'
            elif signal_score < -0.3:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            volatility = getattr(btc_signal, 'volatility', 0.02)
            
            return {
                'trend': trend,
                'volatility': volatility
            }
            
        except Exception as e:
            print(f"⚠️ 시장 컨텍스트 분석 오류: {e}")
            return {'trend': 'neutral', 'volatility': 0.02}

