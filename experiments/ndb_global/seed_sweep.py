"""多 seed 稳健性：idiom 训练切分里到底缺了几个探针成语？

train_task_card.py 用 random.Random(seed).shuffle + 前 10% 当验证，所以「训练语料」
随 seed 变化。这里扫 7 个 seed，看有多少探针成语从未在训练切分里被标为正例。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.builtin.idiom.dataset import build_idiom_dataset  # noqa: E402
from dtseek.tasks.builtin.idiom.lexicon import IDIOMS  # noqa: E402

SEEDS = [0, 1, 7, 42, 123, 2024, 31337]

data = build_idiom_dataset(target_samples=14000)
print(f"数据集 {len(data)} 条，词表 {len(IDIOMS)} 个成语")
total_missing = 0
for seed in SEEDS:
    d = list(data)
    random.Random(seed).shuffle(d)
    n_val = max(200, len(d) // 10)
    train = d[n_val:]
    cov = {sp["word"] for it in train for sp in it.get("spans", [])}
    miss = sorted(IDIOMS - cov)
    total_missing += len(miss)
    print(f"  seed={seed:6d} 训练切分 {len(train)} 条 -> 覆盖 {len(cov)}/{len(IDIOMS)}"
          f"  缺 {len(miss)} ({len(miss)/len(IDIOMS)*100:.2f}%)  {miss}")
print(f"7 个 seed 合计缺失 {total_missing} 次，平均 {total_missing/len(SEEDS):.2f} 个/seed")
