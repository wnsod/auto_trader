"""
🔬 수동 통합 분석 스크립트
IntegratedAnalyzer를 직접 호출하여 analysis_ratios 테이블을 1회성으로 채웁니다.

Usage:
    python market/coin_market/manual_analysis_ratios.py
    
Docker:
    docker exec -it <container_id> python /workspace/market/coin_market/manual_analysis_ratios.py
"""

import os
import sys
import glob
import json
import sqlite3
import traceback
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Optional

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))

DATA_DIR = os.path.join(ROOT_DIR, "market", "coin_market", "data_storage")
STRATEGIES_DIR = os.path.join(DATA_DIR, "learning_strategies")
CANDLE_DB = os.path.join(DATA_DIR, "learning_strategies.db")

# 2. 실행 환경 변수 설정
os.environ['PYTHONPATH'] = ROOT_DIR
os.environ['RL_DB_PATH'] = CANDLE_DB
os.environ['STRATEGY_DB_PATH'] = STRATEGIES_DIR
os.environ['AZ_INTERVALS'] = "15m,30m,240m,1d"

# 3. 엔진 모듈 경로 추가
sys.path.append(ROOT_DIR)

INTERVALS = ["15m", "30m", "240m", "1d"]

# 4. IntegratedAnalyzer 로드
try:
    from rl_pipeline.analysis.integrated_analyzer import IntegratedAnalyzer
    print("✅ IntegratedAnalyzer 로드 완료")
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ IntegratedAnalyzer 로드 실패: {e}")
    print("   → 전략 DB 기반 간접 추정 모드로 전환")
    ANALYZER_AVAILABLE = False


def get_coin_list_from_dbs() -> List[str]:
    """전략 DB 파일들에서 코인 리스트 추출"""
    coins = []
    db_files = glob.glob(os.path.join(STRATEGIES_DIR, "*_strategies.db"))
    
    for db_path in db_files:
        filename = os.path.basename(db_path)
        if filename in ["common_strategies.db", "learning_strategies.db", "trade_candles.db", "learning_candles.db"]:
            continue
        
        # BTC_strategies.db -> BTC
        coin = filename.replace("_strategies.db", "")
        if coin:
            coins.append(coin)
    
    return sorted(list(set(coins)))


def load_candle_data(coin: str, interval: str) -> Optional[pd.DataFrame]:
    """캔들 데이터 로드"""
    try:
        with sqlite3.connect(CANDLE_DB) as conn:
            query = """
                SELECT * FROM candles 
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp DESC
                LIMIT 500
            """
            df = pd.read_sql(query, conn, params=(coin, interval))
            if not df.empty:
                return df.sort_values('timestamp').reset_index(drop=True)
        return None
    except Exception as e:
        return None


def calculate_with_integrated_analyzer(coin: str, regime: str = 'neutral') -> Dict[str, Any]:
    """
    IntegratedAnalyzer를 직접 호출하여 분석 비율 계산
    (absolute_zero_system.py와 동일한 방식)
    """
    try:
        analyzer = IntegratedAnalyzer(session_id=None)
        
        # 1. 프렉탈 비율 계산 (IntegratedAnalyzer 메서드 직접 호출)
        fractal_ratios = analyzer._get_coin_optimal_fractal_intervals(coin, regime)
        
        # 2. 멀티 타임프레임 비율 계산
        multi_timeframe_ratios = analyzer._get_coin_optimal_multi_timeframe_ratios(coin, regime)
        
        # 3. 지표 교차 비율 계산
        indicator_cross_ratios = analyzer._get_coin_optimal_indicator_cross_ratios(coin, regime)
        
        # 4. 최적 분석 모듈 선택
        # 캔들 데이터 로드 시도
        candle_data = None
        for interval in INTERVALS:
            candle_data = load_candle_data(coin, interval)
            if candle_data is not None and len(candle_data) >= 20:
                break
        
        if candle_data is not None:
            optimal_modules = analyzer._select_optimal_analysis_modules(coin, INTERVALS[0], regime, candle_data)
        else:
            optimal_modules = {"fractal": 0.5, "multi_timeframe": 0.5, "indicator_cross": 0.5}
        
        # 5. 인터벌 가중치 계산 (전략 성과 기반)
        interval_weights = calculate_interval_weights_from_strategies(coin)
        
        # 6. 성과 점수 계산
        performance_score = calculate_performance_score_from_strategies(coin)
        
        return {
            'fractal_ratios': fractal_ratios,
            'multi_timeframe_ratios': multi_timeframe_ratios,
            'indicator_cross_ratios': indicator_cross_ratios,
            'optimal_modules': optimal_modules,
            'interval_weights': interval_weights,
            'performance_score': performance_score,
            'accuracy_score': 0.5
        }
        
    except Exception as e:
        print(f"  ⚠️ IntegratedAnalyzer 분석 실패: {e}")
        return None


def calculate_interval_weights_from_strategies(coin: str) -> Dict[str, float]:
    """전략 DB에서 인터벌별 성과 기반 가중치 계산"""
    db_path = os.path.join(STRATEGIES_DIR, f"{coin}_strategies.db")
    
    try:
        interval_weights = {}
        interval_scores = {}
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            for interval in INTERVALS:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as count,
                        AVG(profit) as avg_profit,
                        AVG(win_rate) as avg_winrate,
                        AVG(profit_factor) as avg_pf
                    FROM strategies
                    WHERE interval = ? AND trades_count >= 3 AND max_drawdown <= 0.8
                """, (interval,))
                
                row = cursor.fetchone()
                if row and row['count'] > 0:
                    avg_profit = row['avg_profit'] or 0
                    avg_winrate = row['avg_winrate'] or 0.5
                    avg_pf = row['avg_pf'] or 1.0
                    
                    score = (
                        avg_profit * 0.4 +
                        (avg_winrate - 0.5) * 2.0 * 0.3 +
                        min(avg_pf - 1.0, 2.0) * 0.15
                    )
                    interval_scores[interval] = max(0.1, score + 0.5)
                else:
                    interval_scores[interval] = 0.25
        
        total_score = sum(interval_scores.values())
        if total_score > 0:
            interval_weights = {iv: round(score / total_score, 4) for iv, score in interval_scores.items()}
        else:
            interval_weights = {iv: 1.0 / len(INTERVALS) for iv in INTERVALS}
        
        return interval_weights
        
    except Exception as e:
        return {iv: 1.0 / len(INTERVALS) for iv in INTERVALS}


def calculate_performance_score_from_strategies(coin: str) -> float:
    """전략 DB에서 성과 점수 계산"""
    db_path = os.path.join(STRATEGIES_DIR, f"{coin}_strategies.db")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    AVG(profit) as avg_profit,
                    AVG(win_rate) as avg_winrate,
                    AVG(profit_factor) as avg_pf,
                    COUNT(*) as count
                FROM strategies
                WHERE trades_count >= 3
            """)
            
            row = cursor.fetchone()
            if row and row[3] > 0:
                avg_profit = row[0] or 0
                avg_winrate = row[1] or 0.5
                avg_pf = row[2] or 1.0
                
                score = (
                    min(1.0, (avg_profit + 0.3) / 0.6) * 0.4 +
                    avg_winrate * 0.35 +
                    min(1.0, avg_pf / 3.0) * 0.25
                )
                return round(max(0.0, min(1.0, score)), 3)
        
        return 0.5
        
    except Exception as e:
        return 0.5


def save_analysis_ratios_direct(coin: str, ratios_data: Dict[str, Any]) -> bool:
    """analysis_ratios 테이블에 직접 저장"""
    db_path = os.path.join(STRATEGIES_DIR, f"{coin}_strategies.db")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_ratios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_type TEXT DEFAULT 'coin',
                    market TEXT DEFAULT 'binance',
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    analysis_type TEXT DEFAULT 'neutral',
                    fractal_ratios TEXT,
                    multi_timeframe_ratios TEXT,
                    indicator_cross_ratios TEXT,
                    symbol_specific_ratios TEXT,
                    volatility_ratios TEXT,
                    volume_ratios TEXT,
                    optimal_modules TEXT,
                    interval_weights TEXT,
                    performance_score REAL DEFAULT 0.0,
                    accuracy_score REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 유니크 인덱스 생성
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_ratios_unique 
                ON analysis_ratios(symbol, interval, analysis_type)
            """)
            
            # JSON 직렬화
            json_fields = ['fractal_ratios', 'multi_timeframe_ratios', 'indicator_cross_ratios',
                          'symbol_specific_ratios', 'volatility_ratios', 'volume_ratios', 
                          'optimal_modules', 'interval_weights']
            
            serialized_data = {}
            for field in json_fields:
                val = ratios_data.get(field, {})
                if isinstance(val, dict):
                    serialized_data[field] = json.dumps(val)
                else:
                    serialized_data[field] = val if val else '{}'
            
            interval = ratios_data.get('interval', 'all')
            analysis_type = ratios_data.get('regime', 'neutral')
            
            # 기존 레코드 확인
            cursor.execute("""
                SELECT id FROM analysis_ratios 
                WHERE symbol = ? AND interval = ? AND analysis_type = ?
            """, (coin, interval, analysis_type))
            
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("""
                    UPDATE analysis_ratios SET
                        fractal_ratios = ?,
                        multi_timeframe_ratios = ?,
                        indicator_cross_ratios = ?,
                        symbol_specific_ratios = ?,
                        volatility_ratios = ?,
                        volume_ratios = ?,
                        optimal_modules = ?,
                        interval_weights = ?,
                        performance_score = ?,
                        accuracy_score = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND interval = ? AND analysis_type = ?
                """, (
                    serialized_data.get('fractal_ratios', '{}'),
                    serialized_data.get('multi_timeframe_ratios', '{}'),
                    serialized_data.get('indicator_cross_ratios', '{}'),
                    serialized_data.get('symbol_specific_ratios', '{}'),
                    serialized_data.get('volatility_ratios', '{}'),
                    serialized_data.get('volume_ratios', '{}'),
                    serialized_data.get('optimal_modules', '{}'),
                    serialized_data.get('interval_weights', '{}'),
                    ratios_data.get('performance_score', 0.0),
                    ratios_data.get('accuracy_score', 0.0),
                    coin, interval, analysis_type
                ))
            else:
                cursor.execute("""
                    INSERT INTO analysis_ratios 
                    (market_type, market, symbol, interval, analysis_type,
                     fractal_ratios, multi_timeframe_ratios, indicator_cross_ratios,
                     symbol_specific_ratios, volatility_ratios, volume_ratios,
                     optimal_modules, interval_weights, performance_score, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'coin', 'binance', coin, interval, analysis_type,
                    serialized_data.get('fractal_ratios', '{}'),
                    serialized_data.get('multi_timeframe_ratios', '{}'),
                    serialized_data.get('indicator_cross_ratios', '{}'),
                    serialized_data.get('symbol_specific_ratios', '{}'),
                    serialized_data.get('volatility_ratios', '{}'),
                    serialized_data.get('volume_ratios', '{}'),
                    serialized_data.get('optimal_modules', '{}'),
                    serialized_data.get('interval_weights', '{}'),
                    ratios_data.get('performance_score', 0.0),
                    ratios_data.get('accuracy_score', 0.0)
                ))
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"  ❌ 저장 실패: {e}")
        return False


def process_single_coin(coin: str) -> bool:
    """단일 코인 분석 및 저장 (IntegratedAnalyzer 직접 사용)"""
    db_path = os.path.join(STRATEGIES_DIR, f"{coin}_strategies.db")
    
    if not os.path.exists(db_path):
        print(f"  ⚠️ {coin}: DB 파일 없음")
        return False
    
    try:
        if ANALYZER_AVAILABLE:
            # ✅ IntegratedAnalyzer 직접 호출 (absolute_zero_system.py와 동일)
            result = calculate_with_integrated_analyzer(coin, 'neutral')
            
            if result:
                ratios_data = {
                    'interval': 'all',
                    'regime': 'neutral',
                    'fractal_ratios': result['fractal_ratios'],
                    'multi_timeframe_ratios': result['multi_timeframe_ratios'],
                    'indicator_cross_ratios': result['indicator_cross_ratios'],
                    'optimal_modules': result['optimal_modules'],
                    'interval_weights': result['interval_weights'],
                    'performance_score': result['performance_score'],
                    'accuracy_score': result['accuracy_score'],
                    'coin_specific_ratios': {},
                    'volatility_ratios': {},
                }
            else:
                return False
        else:
            # 폴백: 전략 DB 기반 간접 추정
            return process_single_coin_fallback(coin)
        
        # 저장
        success = save_analysis_ratios_direct(coin, ratios_data)
        
        if success:
            # 결과 미리보기
            iw = result['interval_weights']
            icr = result['indicator_cross_ratios']
            weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in iw.items()])
            indicator_str = ", ".join([f"{k}:{v:.2f}" for k, v in icr.items()])
            print(f"  ✅ {coin}: IntegratedAnalyzer 분석 완료")
            print(f"      인터벌: {weights_str}")
            print(f"      지표: {indicator_str}")
        
        return success
        
    except Exception as e:
        print(f"  ❌ {coin}: 처리 실패 - {e}")
        traceback.print_exc()
        return False


def process_single_coin_fallback(coin: str) -> bool:
    """폴백: 전략 DB 기반 간접 추정"""
    db_path = os.path.join(STRATEGIES_DIR, f"{coin}_strategies.db")
    
    try:
        # 인터벌 가중치
        interval_weights = calculate_interval_weights_from_strategies(coin)
        
        # 성과 점수
        performance_score = calculate_performance_score_from_strategies(coin)
        
        # 기본 비율 (폴백)
        fractal_ratios = {iv: 0.5 for iv in INTERVALS}
        mtf_ratios = {iv: 0.5 for iv in INTERVALS}
        indicator_cross_ratios = {"rsi": 0.5, "macd": 0.5, "mfi": 0.5, "atr": 0.5, "adx": 0.5, "bb": 0.5}
        optimal_modules = {"fractal": 0.5, "multi_timeframe": 0.5, "indicator_cross": 0.5}
        
        ratios_data = {
            'interval': 'all',
            'regime': 'neutral',
            'fractal_ratios': fractal_ratios,
            'multi_timeframe_ratios': mtf_ratios,
            'indicator_cross_ratios': indicator_cross_ratios,
            'optimal_modules': optimal_modules,
            'interval_weights': interval_weights,
            'performance_score': performance_score,
            'accuracy_score': 0.5,
            'coin_specific_ratios': {},
            'volatility_ratios': {},
        }
        
        success = save_analysis_ratios_direct(coin, ratios_data)
        
        if success:
            weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in interval_weights.items()])
            print(f"  ✅ {coin}: 폴백 모드 저장 완료 (가중치: {weights_str})")
        
        return success
        
    except Exception as e:
        print(f"  ❌ {coin}: 폴백 처리 실패 - {e}")
        return False


def run_manual_analysis():
    """전체 코인 통합 분석 실행"""
    print("=" * 70)
    print("🔬 Absolute Zero 시스템 - 통합 분석 비율 생성 (수동 실행)")
    print(f"📍 대상 디렉토리: {STRATEGIES_DIR}")
    print(f"📍 캔들 DB: {CANDLE_DB}")
    print(f"📍 IntegratedAnalyzer: {'✅ 사용 가능' if ANALYZER_AVAILABLE else '❌ 사용 불가 (폴백 모드)'}")
    print("-" * 70)
    
    # 1. 코인 리스트 조회
    coins = get_coin_list_from_dbs()
    print(f"📊 발견된 코인: {len(coins)}개")
    
    if not coins:
        print("❌ 처리할 코인이 없습니다.")
        return
    
    # 2. 각 코인 처리
    print("\n🔄 코인별 분석 비율 계산 시작...")
    print("   (IntegratedAnalyzer 직접 호출 - absolute_zero_system.py와 동일)")
    success_count = 0
    fail_count = 0
    
    for i, coin in enumerate(coins, 1):
        print(f"\n[{i}/{len(coins)}] {coin} 처리 중...")
        
        if process_single_coin(coin):
            success_count += 1
        else:
            fail_count += 1
    
    # 3. 결과 요약
    print("\n" + "=" * 70)
    print("✨ 통합 분석 비율 생성 완료!")
    print(f"   ✅ 성공: {success_count}개 코인")
    print(f"   ❌ 실패: {fail_count}개 코인")
    print("=" * 70)
    
    # 4. 검증 - 첫 번째 코인의 결과 확인
    if coins:
        sample_coin = coins[0]
        sample_db = os.path.join(STRATEGIES_DIR, f"{sample_coin}_strategies.db")
        
        try:
            with sqlite3.connect(sample_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM analysis_ratios WHERE symbol = ?", (sample_coin,))
                count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT interval_weights, fractal_ratios, indicator_cross_ratios, performance_score
                    FROM analysis_ratios 
                    WHERE symbol = ? AND interval = 'all'
                """, (sample_coin,))
                row = cursor.fetchone()
                
                print(f"\n📋 샘플 검증 ({sample_coin}):")
                print(f"   레코드 수: {count}개")
                if row:
                    print(f"   interval_weights: {row[0]}")
                    print(f"   fractal_ratios: {row[1]}")
                    print(f"   indicator_cross_ratios: {row[2]}")
                    print(f"   performance_score: {row[3]}")
        except Exception as e:
            print(f"\n⚠️ 검증 실패: {e}")


if __name__ == "__main__":
    run_manual_analysis()
