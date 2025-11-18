#!/usr/bin/env python
"""ADA 파이프라인 결과 검증 스크립트"""
import json
import sys

session_dir = '/workspace/rl_pipeline/debug_logs/20251116_114038_ADA_4intervals/'

print("=" * 80)
print("ADA RL Pipeline Results Verification (After Refactoring)")
print("=" * 80)
print()

# simulation.jsonl 파싱
interval_results = {}

try:
    with open(f'{session_dir}/simulation.jsonl', 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                if 'selfplay_result' in data and data['selfplay_result']:
                    interval = data.get('interval', 'unknown')
                    selfplay = data['selfplay_result']

                    avg_accuracy = selfplay.get('avg_accuracy', 0.0)
                    best_accuracy = selfplay.get('best_accuracy', 0.0)
                    episodes = selfplay.get('episodes', 0)

                    if interval not in interval_results:
                        interval_results[interval] = {
                            'avg_accuracy': avg_accuracy,
                            'best_accuracy': best_accuracy,
                            'episodes': episodes
                        }
            except json.JSONDecodeError:
                continue

    # 결과 출력
    print("Interval Self-Play Results:")
    print("-" * 80)
    print()

    for interval in ['15m', '30m', '240m', '1d']:
        if interval in interval_results:
            result = interval_results[interval]
            avg_acc = result['avg_accuracy'] * 100
            best_acc = result['best_accuracy'] * 100
            episodes = result['episodes']

            status = ""
            if interval == '240m':
                if avg_acc > 0:
                    status = "PASS (Was 0%!)"
                else:
                    status = "STILL 0%"
            else:
                status = "OK"

            print(f"{interval:>6s}: {avg_acc:6.2f}% avg, {best_acc:6.2f}% best ({episodes} episodes) - {status}")
        else:
            print(f"{interval:>6s}: No data found")

    print()
    print("=" * 80)

    # 240m 검증
    if '240m' in interval_results:
        avg_240m = interval_results['240m']['avg_accuracy'] * 100
        print()
        print("240m Verification Result:")
        print("-" * 80)
        print(f"Average Accuracy: {avg_240m:.2f}%")

        if avg_240m > 0:
            print(" SUCCESS: 240m accuracy is NOT 0%!")
            print(" Refactoring worked - predictions are continuing despite reassessment!")
        else:
            print(" WARNING: 240m accuracy is still 0%")
            print(" Need further investigation")
    else:
        print()
        print("ERROR: No 240m data found in simulation results")

    print("=" * 80)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
