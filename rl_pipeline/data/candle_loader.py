"""
캔들 데이터 로더 모듈
"""

import logging
import sqlite3
import pandas as pd
import os
from typing import Dict, List, Tuple, Any

from rl_pipeline.core.regime_classifier import classify_regime_from_old

logger = logging.getLogger(__name__)

# DB 경로
CANDLES_DB_PATH = os.getenv('CANDLES_DB_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'data_storage', 'rl_candles.db'))

# 환경변수
AZ_CANDLE_DAYS = int(os.getenv('AZ_CANDLE_DAYS', '60'))  # 기본 60일 (신생 코인은 가용 데이터만큼 사용)
AZ_ALLOW_FALLBACK = os.getenv('AZ_ALLOW_FALLBACK', 'false').lower() == 'true'
AZ_FALLBACK_PAIRS = os.getenv('AZ_FALLBACK_PAIRS', '')


def get_available_coins_and_intervals() -> List[tuple]:
    """rl_candles.db에서 사용 가능한 코인과 인터벌 조합을 가져옵니다"""
    try:
        conn = sqlite3.connect(CANDLES_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT coin, interval 
            FROM candles 
            ORDER BY coin, interval
        """)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"❌ 코인/인터벌 조합 조회 실패: {e}")
        # 운영 기본: 빈 리스트 반환(즉시 종료 유도). 필요 시 환경변수 기반 제한적 폴백 허용
        if AZ_ALLOW_FALLBACK and AZ_FALLBACK_PAIRS:
            try:
                pairs = []
                for token in AZ_FALLBACK_PAIRS.split(';'):
                    token = token.strip()
                    if not token:
                        continue
                    coin, itv = token.split(':', 1)
                    pairs.append((coin.strip(), itv.strip()))
                if pairs:
                    logger.warning(f"⚠️ AZ_ALLOW_FALLBACK=true - 환경변수 폴백 사용: {pairs}")
                    return pairs
            except Exception:
                logger.warning("⚠️ AZ_FALLBACK_PAIRS 파싱 실패 - 폴백 미사용")
        return []

def load_candle_data_for_coin(coin: str, intervals: List[str]) -> Dict[tuple, Any]:
    """특정 코인의 모든 인터벌에 대한 캔들 데이터를 로드합니다.
    환경변수 AZ_CANDLE_DAYS로 히스토리 일수를 조절합니다(기본 60일).
    신생 코인의 경우 가용 데이터만큼 사용합니다 (최소 7일).
    """
    try:
        all_candle_data = {}

        conn = sqlite3.connect(CANDLES_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for interval in intervals:
                try:
                    # 캔들 데이터 조회 (존재하는 컬럼만) - 인터벌별 일수 기반 limit 계산
                    days = AZ_CANDLE_DAYS if AZ_CANDLE_DAYS > 0 else 60
                    if interval == '15m':
                        limit = 96 * days  # 15분 = 하루에 96개, 60일 = 5760개
                    elif interval == '30m':
                        limit = 48 * days  # 30분 = 하루에 48개, 60일 = 2880개
                    elif interval == '240m' or interval == '4h':
                        limit = 6 * days  # 240분(4h) = 하루에 6개, 60일 = 360개
                    elif interval == '1d':
                        limit = days  # 1일 = 하루에 1개, 60일 = 60개
                    elif interval.endswith('h'):
                        # 시간 단위 인터벌 (예: 1h, 2h)
                        try:
                            hours = int(interval[:-1])
                            limit = (24 // hours) * days
                        except:
                            limit = 10000
                    elif interval.endswith('m'):
                        # 분 단위 인터벌 (예: 5m, 60m)
                        try:
                            minutes = int(interval[:-1])
                            limit = (1440 // minutes) * days  # 하루 1440분
                        except:
                            limit = 10000
                    else:
                        # 알 수 없는 인터벌은 보수적으로 넉넉히 로드
                        limit = 10000
                    
                    # 🚀 모든 통합 분석 지표 포함 (SELECT * 사용)
                    cursor.execute("""
                        SELECT * FROM candles
                        WHERE coin = ? AND interval = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (coin, interval, limit))
                    
                    rows = cursor.fetchall()
                    if rows:
                        import pandas as pd
                        # 🚀 모든 컬럼 자동 감지 (동적 컬럼 목록)
                        if rows:
                            # 첫 번째 행에서 컬럼 이름 가져오기
                            column_names = [description[0] for description in cursor.description]
                            df = pd.DataFrame(rows, columns=column_names)
                        else:
                            df = pd.DataFrame()
                        # 🔥 Unix 타임스탬프를 datetime으로 변환 (unit='s'로 초 단위 명시)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

                        # 7단계 레짐을 3단계로 매핑 (ranging, trending, volatile)
                        df['regime'] = df['regime_label'].apply(
                            lambda x: classify_regime_from_old(x) if pd.notna(x) else 'ranging'
                        )

                        all_candle_data[(coin, interval)] = df

                        # 신생 코인 체크 (최소 7일 데이터 필요)
                        min_candles_needed = {
                            '15m': 672,  # 7일 * 96개
                            '30m': 336,  # 7일 * 48개
                            '240m': 42,  # 7일 * 6개
                            '4h': 42,    # 7일 * 6개
                            '1d': 7      # 7일 * 1개
                        }
                        min_required = min_candles_needed.get(interval, 100)

                        if len(df) < min_required:
                            logger.warning(f"⚠️ {coin} {interval}: 신생 코인 감지 ({len(df)}개 캔들, 최소 {min_required}개 권장)")
                            if len(df) < min_required // 2:
                                logger.error(f"❌ {coin} {interval}: 데이터 부족 ({len(df)}개 < 최소 {min_required//2}개)")
                                # 데이터가 너무 적으면 제거
                                del all_candle_data[(coin, interval)]
                                continue
                        else:
                            expected_candles = limit
                            if len(df) < expected_candles * 0.8:  # 기대값의 80% 미만
                                logger.info(f"📊 {coin} {interval}: 가용 데이터 사용 ({len(df)}개/{expected_candles}개 목표)")
                            else:
                                logger.info(f"✅ {coin} {interval}: {len(df)}개 캔들 데이터 로드 완료")
                    else:
                        logger.warning(f"⚠️ {coin} {interval}: 캔들 데이터 없음")
                        
                except Exception as e:
                    logger.error(f"❌ {coin} {interval} 캔들 데이터 로드 실패: {e}")
                    continue
        
        conn.close()
        return all_candle_data
        
    except Exception as e:
        logger.error(f"❌ {coin} 캔들 데이터 로드 실패: {e}")
        return {}

# Self-play 분석 함수는 새로운 파이프라인에서 처리되므로 제거

