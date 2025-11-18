#!/usr/bin/env python
"""통합 학습 정확도 문제 분석"""
import json

file_path = '/workspace/rl_pipeline/debug_logs/20251116_103025_ADA_4intervals/simulation.jsonl'

print("=" * 80)
print("통합 학습 정확도 문제 분석")
print("=" * 80)
print()

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 인터벌별 selfplay_end 결과 수집
interval_results = {}

for line in lines:
    try:
        data = json.loads(line)

        if data.get('event') == 'selfplay_end':
            interval = data.get('interval', 'unknown')
            coin = data.get('coin', 'unknown')
            summary = data.get('summary', {})

            key = f"{coin}-{interval}"
            interval_results[key] = {
                'total_episodes': data.get('total_episodes', 0),
                'avg_accuracy': summary.get('avg_accuracy', 0),
                'best_accuracy': summary.get('best_accuracy', 0),
                'type': summary.get('type', 'unknown'),
                'early_stopped': summary.get('early_stopped', False),
                'strategy_count': summary.get('strategy_count', 0)
            }
    except:
        continue

print("📊 인터벌별 Self-Play 결과:")
print("-" * 80)

for key in sorted(interval_results.keys()):
    result = interval_results[key]
    print(f"\n{key}:")
    print(f"  - 타입: {result['type']}")
    print(f"  - 총 에피소드: {result['total_episodes']}개")
    print(f"  - 평균 정확도: {result['avg_accuracy']:.4f} ({result['avg_accuracy']*100:.2f}%)")
    print(f"  - 최고 정확도: {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
    print(f"  - 전략 수: {result['strategy_count']}개")
    print(f"  - 조기 종료: {'✅ YES' if result['early_stopped'] else '❌ NO'}")

# 개별 에피소드 정확도 분포 확인
print("\n" + "=" * 80)
print("📈 인터벌별 에피소드 정확도 분포:")
print("-" * 80)

interval_episodes = {}

for line in lines:
    try:
        data = json.loads(line)

        if data.get('event') == 'predictive_selfplay_episode':
            interval = data.get('interval', 'unknown')
            accuracy = data.get('accuracy', 0)

            if interval not in interval_episodes:
                interval_episodes[interval] = []

            interval_episodes[interval].append(accuracy)
    except:
        continue

for interval in sorted(interval_episodes.keys()):
    accuracies = interval_episodes[interval]
    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        unique_count = len(set(accuracies))

        print(f"\n{interval}:")
        print(f"  - 에피소드 수: {len(accuracies)}개")
        print(f"  - 평균: {avg:.4f} ({avg*100:.2f}%)")
        print(f"  - 범위: {min_acc:.4f} ~ {max_acc:.4f}")
        print(f"  - 고유값 개수: {unique_count}개")
        print(f"  - 처음 5개: {accuracies[:5]}")
        print(f"  - 마지막 5개: {accuracies[-5:]}")

print("\n" + "=" * 80)
print("✅ 분석 완료!")
print("=" * 80)
