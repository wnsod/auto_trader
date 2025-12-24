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
def _get_default_candles_db_path():
    """캔들 DB 경로를 스마트하게 탐색합니다."""
    # 0. RL_DB_PATH 환경변수 최우선 (run_learning.py 등에서 설정)
    if os.getenv('RL_DB_PATH'):
        return os.getenv('RL_DB_PATH')
        
    # 1. 환경변수 직접 지정
    if os.getenv('CANDLES_DB_PATH'):
        return os.getenv('CANDLES_DB_PATH')
        
    # 2. DATA_STORAGE_PATH 환경변수 확인 (Docker 경로 하드코딩보다 우선)
    if os.getenv('DATA_STORAGE_PATH'):
        ds_path = os.path.join(os.getenv('DATA_STORAGE_PATH'), 'rl_candles.db')
        if os.path.exists(ds_path):
            return ds_path
            
    # 3. Docker 표준 경로 확인 (하위 호환성 유지하되 우선순위 낮춤)
    docker_path = '/workspace/data_storage/rl_candles.db'
    if os.path.exists(docker_path):
        return docker_path
            
    # 4. 프로젝트 구조 기반 탐색 (현재 파일: rl_pipeline/data/candle_loader.py)
    # 목표: data_storage/rl_candles.db (프로젝트 루트 아래)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # ..(data) -> ..(rl_pipeline) -> ..(root)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    
    # 프로젝트 루트 아래 data_storage 확인
    root_ds_path = os.path.join(project_root, 'data_storage', 'rl_candles.db')
    if os.path.exists(root_ds_path):
        return root_ds_path
        
    # 5. 기존 레거시 경로 (rl_pipeline/data_storage) - 호환성
    legacy_path = os.path.join(current_dir, '..', 'data_storage', 'rl_candles.db')
    if os.path.exists(legacy_path):
        return os.path.abspath(legacy_path)
        
    # 파일이 어디에도 없으면 Docker 표준 경로 반환 (기본값)
    return docker_path

CANDLES_DB_PATH = _get_default_candles_db_path()

# 환경변수
AZ_CANDLE_DAYS = int(os.getenv('AZ_CANDLE_DAYS', '60'))  # 기본 60일 (신생 코인은 가용 데이터만큼 사용)
AZ_ALLOW_FALLBACK = os.getenv('AZ_ALLOW_FALLBACK', 'false').lower() == 'true'
AZ_FALLBACK_PAIRS = os.getenv('AZ_FALLBACK_PAIRS', '')


def get_available_coins_and_intervals() -> List[tuple]:
    """rl_candles.db에서 사용 가능한 코인과 인터벌 조합을 가져옵니다"""
    try:
        db_path = os.path.abspath(CANDLES_DB_PATH)
        if not os.path.exists(db_path):
            logger.warning(f"⚠️ 캔들 DB 파일이 없습니다: {db_path}")
        
        # 읽기 전용 모드로 연결 시도 (파일이 없으면 에러 발생 가능성 있음)
        # uri=True를 사용하면 file: 경로 사용 가능
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError:
            # 파일이 없거나 열 수 없는 경우 일반 모드로 재시도 (생성될 수 있음 - 하지만 여기선 조회만)
            conn = sqlite3.connect(db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(candles)")
        columns = [info[1] for info in cursor.fetchall()]
        has_symbol = 'symbol' in columns

        if has_symbol:
            # symbol 컬럼이 있으면 symbol을 사용하여 조회
            cursor.execute("""
                SELECT DISTINCT symbol as coin, interval 
                FROM candles 
                ORDER BY symbol, interval
            """)
        else:
            # symbol 컬럼이 없으면 coin 컬럼 사용
            cursor.execute("""
                SELECT DISTINCT coin, interval 
                FROM candles 
                ORDER BY coin, interval
            """)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        db_path = os.path.abspath(CANDLES_DB_PATH)
        logger.error(f"❌ 코인/인터벌 조합 조회 실패: {e}")
        logger.error(f"   - DB 경로: {db_path}")
        logger.error(f"   - 존재 여부: {os.path.exists(db_path)}")
        if os.path.exists(db_path):
            logger.error(f"   - 파일 크기: {os.path.getsize(db_path)} bytes")
            logger.error(f"   - 읽기 권한: {os.access(db_path, os.R_OK)}")
            
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
                    # 🆕 컬럼 호환성: symbol 컬럼이 있으면 coin 대신 사용 (COALESCE 또는 컬럼 확인)
                    try:
                        # symbol, coin 컬럼 존재 여부 확인
                        cursor.execute("PRAGMA table_info(candles)")
                        columns = [info[1] for info in cursor.fetchall()]
                        has_symbol = 'symbol' in columns
                        has_coin = 'coin' in columns
                        
                        if has_symbol and has_coin:
                            # 둘 다 있으면 둘 다 확인
                            cursor.execute("""
                                SELECT * FROM candles
                                WHERE (symbol = ? OR coin = ?) AND interval = ?
                                ORDER BY timestamp DESC
                                LIMIT ?
                            """, (coin, coin, interval, limit))
                        elif has_symbol:
                            # symbol만 있으면 symbol만 확인
                            cursor.execute("""
                                SELECT * FROM candles
                                WHERE symbol = ? AND interval = ?
                                ORDER BY timestamp DESC
                                LIMIT ?
                            """, (coin, interval, limit))
                        elif has_coin:
                            # coin만 있으면 coin만 확인
                            cursor.execute("""
                                SELECT * FROM candles
                                WHERE symbol = ? AND interval = ?
                                ORDER BY timestamp DESC
                                LIMIT ?
                            """, (coin, interval, limit))
                        else:
                            raise ValueError("❌ 테이블에 'symbol' 또는 'coin' 컬럼이 없습니다.")
                    except Exception as query_err:
                        logger.error(f"❌ 쿼리 실행 실패: {query_err}")
                        continue
                    
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

                        # 신생 코인 체크 (최소 7일 -> 최소 데이터 기준 대폭 완화 for KRX)
                        # 주식 시장은 장 운영시간이 짧아 데이터 개수가 적으므로 기준을 낮춤
                        min_candles_needed = {
                            '15m': 80,   # 최소 하루치
                            '30m': 40,   # 최소 2~3일치
                            '240m': 10,  # 최소 2일치
                            '4h': 10,
                            '60m': 20,   # 1시간봉 추가
                            '1d': 5,     # 1주일(5거래일)
                            '1w': 2,     # 2주
                            '1mo': 2,    # 2달
                            '1M': 2      # 2달 (별칭)
                        }
                        min_required = min_candles_needed.get(interval, 20)

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

