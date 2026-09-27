#!/usr/bin/env bash
# semantic-NDB 的完整 AB 序列。先跑两条**校验臂**（必须复现既有日志），再跑语义臂。
# 用途：`bash experiments/ndb_semantic/run_ab.sh 2>&1 | tee experiments/ndb_semantic/ab_run.log`
set -u
cd "$(dirname "$0")/../.."
E=experiments/ndb_semantic
PY="uv run python"

echo "########## 1. 校验臂 base seed=42（应复现 0.3773）##########"
timeout 1800 $PY $E/ab_semantic.py --arm base --seed 42 --tag validate_base_seed42

echo "########## 2. 校验臂 literal seed=42（应复现 0.8408）##########"
timeout 1800 $PY $E/ab_semantic.py --arm literal --seed 42 --tag validate_literal_seed42

echo "########## 3. 码本 seed=43 ##########"
timeout 1800 $PY $E/make_codebook.py --seed 43

echo "########## 4. semantic seed=42 ##########"
timeout 1800 $PY $E/ab_semantic.py --arm semantic --seed 42 \
  --codebook $E/codebook_seed42.pt --tag semantic_seed42

echo "########## 5. semantic seed=43 ##########"
timeout 1800 $PY $E/ab_semantic.py --arm semantic --seed 43 \
  --codebook $E/codebook_seed43.pt --tag semantic_seed43

echo "########## DONE ##########"
