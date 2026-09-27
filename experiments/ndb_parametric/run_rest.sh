#!/usr/bin/env bash
# 补齐剩余 3 次运行；被外部 SIGTERM 打断时自动重试（最多 4 次/配置）。
# 用 `python -u`（经 uv run）保证日志实时落盘，被杀时也能看出跑到哪一步。
set -u
cd "$(dirname "$0")/../.." || exit 1
R=experiments/ndb_parametric
for spec in "diffwrite 42" "diffwrite 43" "fastweight_cls 43"; do
  set -- $spec; arm=$1; seed=$2
  for attempt in 1 2 3 4; do
    [ -s "$R/results/${arm}_seed${seed}.json" ] && break
    {
      echo "=== $arm seed=$seed attempt=$attempt ==="
      nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader
    } >> "$R/logs/gpu_snapshots.txt"
    echo ">>> $arm seed=$seed attempt=$attempt"
    uv run python -u "$R/ab_parametric.py" --arm "$arm" --seed "$seed" \
      --json-out "$R/results/${arm}_seed${seed}.json" \
      >> "$R/logs/${arm}_seed${seed}.log" 2>&1
    echo "    exit=$?"
  done
  echo "    result: $(grep -o '"repeat_mention_acc": [0-9.]*' "$R/results/${arm}_seed${seed}.json" 2>/dev/null | head -1)"
done
echo "RETRY DONE"
