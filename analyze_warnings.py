"""로그 파일의 WARNING 메시지 분석"""
from collections import Counter
import re
import sys

# 출력 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

log_file = 'C:\\auto_trader\\rl_pipeline\\test_rl_log.txt'

warnings = []
warning_types = Counter()

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'WARNING' in line:
            # WARNING 뒤의 메시지 추출
            match = re.search(r'WARNING - (.+)$', line)
            if match:
                warning_msg = match.group(1)
                warnings.append((line.strip()[:100], warning_msg))

                # 경고 유형 분류 (숫자나 특정 값 제거하고 패턴화)
                pattern = re.sub(r'\d+\.\d+', 'X.X', warning_msg)  # 소수점 숫자
                pattern = re.sub(r'\d+', 'N', pattern)  # 정수
                pattern = re.sub(r'\(.*?\)', '', pattern)  # 괄호 내용 제거
                pattern = pattern[:150]  # 너무 길면 자르기
                warning_types[pattern] += 1

print("=" * 80)
print("WARNING 메시지 분석 결과")
print("=" * 80)
print(f"\n총 WARNING 수: {len(warnings)}개\n")

print("\n상위 경고 유형 (발생 빈도순):")
print("-" * 80)
for i, (pattern, count) in enumerate(warning_types.most_common(15), 1):
    print(f"{i}. [{count}회] {pattern}")

print("\n\n" + "=" * 80)
print("샘플 경고 메시지 (처음 30개)")
print("=" * 80)
for i, (full_line, msg) in enumerate(warnings[:30], 1):
    print(f"\n{i}. {msg[:200]}")
