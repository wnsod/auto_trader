"""
성능 개선 사항 테스트 및 검증 스크립트
"""
import re
import time
import sys
from collections import Counter
from datetime import datetime

# Windows 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def analyze_log_file(log_path):
    """로그 파일 분석"""
    print("=" * 80)
    print(f"로그 파일 분석: {log_path}")
    print(f"분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 1. 거래 발생 횟수 확인
        print("\n1️⃣ 거래 발생 분석")
        print("-" * 80)

        # "거래 X회" 패턴 찾기
        trade_patterns = re.findall(r'거래\s+(\d+)회', content)
        if trade_patterns:
            trade_counts = [int(t) for t in trade_patterns]
            non_zero_trades = [t for t in trade_counts if t > 0]

            print(f"  총 거래 기록: {len(trade_counts)}개")
            print(f"  거래 발생 케이스: {len(non_zero_trades)}개")
            if non_zero_trades:
                print(f"  평균 거래 횟수: {sum(non_zero_trades) / len(non_zero_trades):.1f}회")
                print(f"  최대 거래 횟수: {max(non_zero_trades)}회")
                print(f"  거래 발생률: {len(non_zero_trades) / len(trade_counts) * 100:.1f}%")

                if len(non_zero_trades) / len(trade_counts) >= 0.3:
                    print("  ✅ 성공: 거래 발생률 30% 이상")
                else:
                    print(f"  ⚠️ 부족: 거래 발생률 {len(non_zero_trades) / len(trade_counts) * 100:.1f}% (목표: 30%)")
            else:
                print("  ❌ 실패: 거래가 전혀 발생하지 않음")
        else:
            print("  ⚠️ 거래 기록을 찾을 수 없음")

        # 2. 액션 다양성 분석
        print("\n2️⃣ 액션 다양성 분석")
        print("-" * 80)

        # "액션 분포" 패턴 찾기
        action_dist_patterns = re.findall(r'액션 분포:.*?NEUTRAL\(0\)=(\d+),.*?UP\(1\)=(\d+),.*?DOWN\(2\)=(\d+)', content)
        if action_dist_patterns:
            total_neutral = sum(int(p[0]) for p in action_dist_patterns)
            total_up = sum(int(p[1]) for p in action_dist_patterns)
            total_down = sum(int(p[2]) for p in action_dist_patterns)
            total_actions = total_neutral + total_up + total_down

            print(f"  NEUTRAL (HOLD): {total_neutral}회 ({total_neutral/total_actions*100:.1f}%)")
            print(f"  UP (BUY): {total_up}회 ({total_up/total_actions*100:.1f}%)")
            print(f"  DOWN (SELL): {total_down}회 ({total_down/total_actions*100:.1f}%)")

            # 3가지 액션 모두 사용?
            if total_neutral > 0 and total_up > 0 and total_down > 0:
                print("  ✅ 성공: 3가지 액션 모두 사용")
            elif total_down > 0:
                print("  ⚠️ 개선됨: SELL 액션 발생 (이전에는 0)")
            else:
                print("  ❌ 실패: SELL 액션 없음")

            # SELL 비율 확인
            if total_down / total_actions >= 0.1:
                print(f"  ✅ 성공: SELL 비율 {total_down/total_actions*100:.1f}% (목표: 10% 이상)")
            elif total_down > 0:
                print(f"  ⚠️ 부족: SELL 비율 {total_down/total_actions*100:.1f}% (목표: 10% 이상)")
        else:
            print("  ⚠️ 액션 분포 기록을 찾을 수 없음")

        # 3. 보상 다양성 분석
        print("\n3️⃣ 보상 다양성 분석")
        print("-" * 80)

        # "std=X.XXX" 패턴 찾기
        std_patterns = re.findall(r'std=([0-9.]+)', content)
        if std_patterns:
            std_values = [float(s) for s in std_patterns]
            avg_std = sum(std_values) / len(std_values)

            print(f"  평균 표준편차: {avg_std:.6f}")
            print(f"  최대 표준편차: {max(std_values):.6f}")
            print(f"  최소 표준편차: {min(std_values):.6f}")

            if avg_std >= 0.15:
                print(f"  ✅ 성공: 평균 std {avg_std:.6f} (목표: 0.15 이상)")
            elif avg_std >= 0.1:
                print(f"  ⚠️ 개선됨: 평균 std {avg_std:.6f} (이전: ~0.04)")
            else:
                print(f"  ❌ 부족: 평균 std {avg_std:.6f} (목표: 0.15 이상)")
        else:
            print("  ⚠️ 표준편차 기록을 찾을 수 없음")

        # 4. Self-play 에피소드 수 확인
        print("\n4️⃣ Self-play 에피소드 분석")
        print("-" * 80)

        # "에피소드 X/Y" 패턴 찾기
        episode_patterns = re.findall(r'에피소드\s+(\d+)/(\d+)', content)
        if episode_patterns:
            max_episode = max(int(e[0]) for e in episode_patterns)
            print(f"  최대 실행 에피소드: {max_episode}회")

            if max_episode >= 20:
                print(f"  ✅ 성공: {max_episode}회 (목표: 20회 이상)")
            elif max_episode >= 10:
                print(f"  ⚠️ 개선됨: {max_episode}회 (이전: ~5회)")
            else:
                print(f"  ❌ 부족: {max_episode}회 (목표: 20회 이상)")
        else:
            print("  ⚠️ 에피소드 기록을 찾을 수 없음")

        # 5. 경고 메시지 카운트
        print("\n5️⃣ 경고 메시지 분석")
        print("-" * 80)

        warning_count = content.count('WARNING')
        total_lines = content.count('\n')

        print(f"  총 경고 수: {warning_count}개")
        print(f"  총 라인 수: {total_lines}개")
        print(f"  경고 비율: {warning_count/total_lines*100:.2f}%")

        if warning_count / total_lines < 0.1:
            print("  ✅ 성공: 경고 비율 10% 미만")
        elif warning_count / total_lines < 0.2:
            print("  ⚠️ 양호: 경고 비율 20% 미만")
        else:
            print(f"  ❌ 높음: 경고 비율 {warning_count/total_lines*100:.2f}%")

        # 특정 경고 카운트
        diversity_warnings = content.count('예측 다양성 심각 부족')
        reward_warnings = content.count('보상 다양성 심각 부족')

        print(f"\n  주요 경고:")
        print(f"    - 예측 다양성 부족: {diversity_warnings}회")
        print(f"    - 보상 다양성 부족: {reward_warnings}회")

        # 6. 종합 평가
        print("\n" + "=" * 80)
        print("종합 평가")
        print("=" * 80)

        success_criteria = []

        # 거래 발생
        if trade_patterns and non_zero_trades:
            if len(non_zero_trades) / len(trade_counts) >= 0.3:
                success_criteria.append(True)
            else:
                success_criteria.append(False)

        # 액션 다양성
        if action_dist_patterns:
            if total_down > 0:
                success_criteria.append(True)
            else:
                success_criteria.append(False)

        # 보상 std
        if std_patterns:
            if avg_std >= 0.1:
                success_criteria.append(True)
            else:
                success_criteria.append(False)

        # 에피소드 수
        if episode_patterns:
            if max_episode >= 10:
                success_criteria.append(True)
            else:
                success_criteria.append(False)

        success_count = sum(success_criteria)
        total_criteria = len(success_criteria)

        print(f"\n달성률: {success_count}/{total_criteria} ({success_count/total_criteria*100:.0f}%)")

        if success_count >= total_criteria * 0.75:
            print("\n🎉 성공: 개선 목표 75% 이상 달성!")
        elif success_count >= total_criteria * 0.5:
            print("\n⚠️ 부분 성공: 개선 목표 50% 이상 달성")
        else:
            print("\n❌ 실패: 개선 목표 50% 미만 달성")

        return success_criteria

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == '__main__':
    import sys

    # 로컬 파일 분석
    log_file = 'C:\\auto_trader\\rl_pipeline\\test_rl_log.txt'

    print("\n\n" + "=" * 80)
    print("성능 개선 사항 테스트 시작")
    print("=" * 80)

    result = analyze_log_file(log_file)

    print("\n\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)

    # 도커 컨테이너 내부에서 실행할 경우
    if len(sys.argv) > 1 and sys.argv[1] == '--docker':
        docker_log = '/workspace/rl_pipeline/test_rl_log.txt'
        analyze_log_file(docker_log)
