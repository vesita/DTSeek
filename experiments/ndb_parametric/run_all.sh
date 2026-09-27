#!/usr/bin/env bash
# 全部参数化臂 × 2 seed。每跑前记一次 nvidia-smi 快照（GPU 占用要写进报告）。
set -u
cd "$(dirname "$0")/../.." || exit 1
R=experiments/ndb_parametric
mkdir -p "$R/results" "$R/logs"
: > "$R/logs/gpu_snapshots.txt"
for arm in slots16 slots32 diffwrite fastweight fastweight_cls; do
  for seed in 42 43; do
    {
      echo "=== $arm seed=$seed ==="
      nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader
    } >> "$R/logs/gpu_snapshots.txt"
    echo ">>> $arm seed=$seed"
    uv run python "$R/ab_parametric.py" --arm "$arm" --seed "$seed" \
      --json-out "$R/results/${arm}_seed${seed}.json" \
      > "$R/logs/${arm}_seed${seed}.log" 2>&1
    echo "    exit=$? $(grep -o '"repeat_mention_acc": [0-9.]*' "$R/results/${arm}_seed${seed}.json" | head -1)"
  done
done
echo "ALL DONE"
