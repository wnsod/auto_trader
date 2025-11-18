#!/usr/bin/env python3
"""10개 코인에서 8개 코인으로 변경된 매매 분석"""
import re
from collections import defaultdict

LOG_FILE = r'C:\auto_trader\trade\test_realtime_log.txt'

print('=' * 80)
print('매매 내역 및 포지션 변화 분석')
print('=' * 80)

try:
    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 최근 1000줄만 분석
    recent_lines = lines[-1000:]

    # 타임스탬프별로 포지션 수 추적
    position_counts = []
    positions_detail = []
    trades = []

    current_timestamp = None
    current_positions = []

    for i, line in enumerate(recent_lines):
        # 타임스탬프 추출
        timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
        if timestamp_match:
            current_timestamp = timestamp_match.group(1)

        # 포지션 정보 (코인명: ... 형태)
        if '보유' in line or '매수가' in line:
            coin_match = re.search(r'([A-Z]{3,})\s*[:：]', line)
            if coin_match and current_timestamp:
                coin = coin_match.group(1)
                if coin not in current_positions:
                    current_positions.append(coin)

        # 포지션 섹션의 끝을 감지 (다음 섹션 시작 또는 종료)
        if ('=' * 20 in line or '시그널' in line or '수익' in line) and current_positions:
            if current_timestamp:
                positions_detail.append({
                    'timestamp': current_timestamp,
                    'count': len(current_positions),
                    'coins': current_positions.copy()
                })
            current_positions = []

        # 매도/매수 완료
        if '매도 완료' in line or '매수 완료' in line:
            trades.append({
                'timestamp': current_timestamp or 'Unknown',
                'line': line.strip()
            })

    # 포지션 개수 변화 추적
    print('\n[INFO] 포지션 개수 변화:')
    print('-' * 80)

    if positions_detail:
        prev_count = None
        for pos in positions_detail[-20:]:  # 최근 20개
            if prev_count and pos['count'] != prev_count:
                print(f"\n[ALERT] 변화 감지: {prev_count}개 -> {pos['count']}개")
                print(f"   시간: {pos['timestamp']}")
                print(f"   코인: {', '.join(pos['coins'])}")
            else:
                print(f"   {pos['timestamp']}: {pos['count']}개 코인 보유")
            prev_count = pos['count']
    else:
        print('   포지션 정보를 찾을 수 없습니다')

    # 매매 기록 확인
    print('\n\n[INFO] 최근 매매 기록:')
    print('-' * 80)

    if trades:
        print(f'\n총 {len(trades)}개의 거래 발견:\n')
        for trade in trades[-10:]:  # 최근 10개
            print(f"   {trade['timestamp']}: {trade['line']}")
    else:
        print('   [ALERT] 매매 기록이 없습니다!')

    # 마지막 포지션 상태
    print('\n\n[INFO] 최종 포지션 상태:')
    print('-' * 80)

    if positions_detail:
        last_pos = positions_detail[-1]
        print(f'\n시간: {last_pos["timestamp"]}')
        print(f'보유 코인 수: {last_pos["count"]}개')
        print(f'보유 코인: {", ".join(last_pos["coins"])}')

    # 가설: 10개 -> 8개로 줄어든 원인 분석
    print('\n\n[INFO] 분석 결과:')
    print('-' * 80)

    if not trades:
        print('\n[ALERT] 문제 발견: 포지션은 줄어들었지만 매매 기록이 없습니다!')
        print('\n가능한 원인:')
        print('   1. 매도가 실행되었지만 로그에 기록되지 않음')
        print('   2. 가상매매 시스템의 손절/익절이 조용히 실행됨')
        print('   3. 로그 기록 누락 버그')
        print('   4. 다른 프로세스가 포지션을 정리함')
    else:
        print('\n[OK] 매매 기록이 있습니다. 위 내역을 확인하세요.')

    print('\n' + '=' * 80)

except FileNotFoundError:
    print(f'[ERROR] 파일을 찾을 수 없습니다: {LOG_FILE}')
except Exception as e:
    print(f'[ERROR] 오류 발생: {e}')
    import traceback
    traceback.print_exc()
