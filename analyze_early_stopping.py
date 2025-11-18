#!/usr/bin/env python
"""조기 종료 상세 분석"""
import json

file_path = '/workspace/rl_pipeline/debug_logs/20251116_103025_ADA_4intervals/simulation.jsonl'

print("=" * 80)
print("조기 종료 상세 분석")
print("=" * 80)
print()

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 인터벌별 에피소드 진행 분석
interval_episodes = {}

for line in lines:
    try:
        data = json.loads(line)

        if data.get('event') == 'predictive_selfplay_episode':
            interval = data.get('interval', 'unknown')
            episode = data.get('episode', 0)
            accuracy = data.get('accuracy', 0)

            if interval not in interval_episodes:
                interval_episodes[interval] = []

            interval_episodes[interval].append({
                'episode': episode,
                'accuracy': accuracy
            })
    except:
        continue

# 조기 종료 설정 확인
for line in lines:
    try:
        data = json.loads(line)
        if data.get('event') == 'early_stop_config':
            print("📊 조기 종료 설정:")
            print(f"  - 인터벌: {data.get('interval')}")
            print(f"  - 최소 에피소드: {data.get('min_episodes')}개")
            print(f"  - Patience: {data.get('patience')}회")
            print(f"  - 정확도 임계값: {data.get('accuracy_threshold', 0)*100:.0f}%")
            print(f"  - 최대 에피소드: {data.get('max_episodes')}개")
            print()
    except:
        continue

print("=" * 80)
print("인터벌별 학습 진행 분석:")
print("=" * 80)

for interval in sorted(interval_episodes.keys()):
    episodes = interval_episodes[interval]
    print(f"\n{interval}:")
    print(f"  - 총 에피소드: {len(episodes)}개")

    if len(episodes) > 0:
        # 정확도 변화 추이
        accuracies = [ep['accuracy'] for ep in episodes]

        # 첫 10개와 마지막 10개 비교
        first_10 = accuracies[:10]
        last_10 = accuracies[-10:]

        first_avg = sum(first_10) / len(first_10) if first_10 else 0
        last_avg = sum(last_10) / len(last_10) if last_10 else 0
        improvement = last_avg - first_avg

        print(f"  - 처음 10개 평균: {first_avg:.4f} ({first_avg*100:.2f}%)")
        print(f"  - 마지막 10개 평균: {last_avg:.4f} ({last_avg*100:.2f}%)")
        print(f"  - 개선폭: {improvement:.4f} ({improvement*100:+.2f}%p)")

        # 에피소드별 상세 (처음 5개, 중간 5개, 마지막 5개)
        mid_point = len(episodes) // 2

        print(f"\n  📈 에피소드 진행 상황:")
        first_5 = [f"{ep['accuracy']:.3f}" for ep in episodes[:5]]
        last_5 = [f"{ep['accuracy']:.3f}" for ep in episodes[-5:]]
        print(f"     처음 5개: {first_5}")
        if len(episodes) > 10:
            mid_5 = [f"{ep['accuracy']:.3f}" for ep in episodes[mid_point-2:mid_point+3]]
            print(f"     중간 5개: {mid_5}")
        print(f"     마지막 5개: {last_5}")

        # 개선이 있었는지 확인
        if improvement > 0.01:
            print(f"  ✅ 학습 진행 중 (개선폭 {improvement*100:.2f}%p)")
        elif improvement > 0:
            print(f"  🟡 약간 개선 (개선폭 {improvement*100:.2f}%p)")
        else:
            print(f"  ❌ 개선 없음 (변화 {improvement*100:.2f}%p)")

print("\n" + "=" * 80)
print("✅ 분석 완료!")
print("=" * 80)
