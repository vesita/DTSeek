#!/usr/bin/env python
"""全局键→id 表的**贝叶斯上界**：即使一张表在训练集上完美统计，它能到多少？

`lngtab` 只可能学 `P(id | 提及起始 n-gram)` 的边缘分布。本脚本不做梯度，直接在训练集上
计数得到每把键的后验众数，再在验证集上评 —— 这是**任何**确定性全局表（含无穷容量的
完美全局表）在该键空间上能拿到的上界。因此它给出的数字不依赖优化器、学习率、容量。

键与 `MentionNDB`/`lngtab` 逐字同源：提及起始处的第 1 个字（unigram）与
第 1、2 个字（bigram）。分开报 unigram / bigram / 回退三级，以及按类别拆分。

用法：uv run python experiments/ndb_ngram/oracle_global.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def kind_of(spans):
    out, seen = [], []
    for m in sorted(spans, key=lambda x: x["start"]):
        same = any(e["label"] == m["label"] for e in seen)
        if not same:
            k = "first"
        else:
            sw_same = any(e["word"] == m["word"] and e["label"] == m["label"] for e in seen)
            sw_oth = any(e["word"] == m["word"] and e["label"] != m["label"] for e in seen)
            k = "literal_same_id" if sw_same else ("literal_other_id" if sw_oth else "alias")
        out.append(k)
        seen.append({"label": m["label"], "word": m["word"]})
    return out


def keys_of(text, start):
    """与 MentionNDB 的 (1,2) 阶同源：起始字 / 起始双字。"""
    u = text[start:start + 1]
    b = text[start:start + 2]
    return u, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=9000)
    args = ap.parse_args()

    from dtseek.tasks.plugin import resolve_tasks

    card = resolve_tasks(["person"])["person"]
    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]

    # ── 在训练集上计数（四种表：unigram / bigram / 回退 to unigram / 回退 to 全局）──
    cnt_u: dict[str, Counter] = defaultdict(Counter)
    cnt_b: dict[str, Counter] = defaultdict(Counter)
    glob = Counter()
    for sample in train:
        text = sample["text"]
        for m in sample["spans"]:
            lab = m["label"]
            if lab <= 0:
                continue
            u, b = keys_of(text, m["start"])
            cnt_u[u][lab] += 1
            cnt_b[b][lab] += 1
            glob[lab] += 1
    glob_mode = glob.most_common(1)[0][0]

    def pred(table, key, backoff):
        c = table.get(key)
        return c.most_common(1)[0][0] if c else backoff

    stats = {name: Counter() for name in ("unigram", "bigram", "bigram->unigram")}
    hit = {name: Counter() for name in stats}
    seen_b = seen_u = 0
    tot = 0
    for sample in val:
        text = sample["text"]
        spans = sorted(sample["spans"], key=lambda x: x["start"])
        kinds = kind_of(spans)
        for m, k in zip(spans, kinds):
            if k == "first":
                continue
            u, b = keys_of(text, m["start"])
            if b in cnt_b:
                seen_b += 1
            if u in cnt_u:
                seen_u += 1
            tot += 1
            preds = {
                "unigram": pred(cnt_u, u, glob_mode),
                "bigram": pred(cnt_b, b, glob_mode),
                "bigram->unigram": pred(cnt_b, b, pred(cnt_u, u, glob_mode)),
            }
            for name, p in preds.items():
                stats[name][k] += 1
                if p == m["label"]:
                    hit[name][k] += 1

    print(f"=== 全局键→id 表的贝叶斯上界 | seed={args.seed} "
          f"train={len(train)} val={len(val)} 重复提及={tot} ===")
    print(f"键在训练集出现率：unigram {seen_u/max(1,tot):.4f}  bigram {seen_b/max(1,tot):.4f}")
    out = {"seed": args.seed, "n_repeat": tot,
           "key_coverage": {"unigram": seen_u / max(1, tot), "bigram": seen_b / max(1, tot)},
           "tables": {}}
    for name in stats:
        totn = sum(stats[name].values())
        toth = sum(hit[name].values())
        row = {"repeat_acc": toth / max(1, totn)}
        print(f"\n  [{name}] repeat 合计 = {toth}/{totn} = {row['repeat_acc']:.4f}  "
              f"（对照：non-param NDB ≈ 0.84，base ≈ 0.377）")
        for k in ("literal_same_id", "literal_other_id", "alias"):
            n, o = stats[name][k], hit[name][k]
            row[k] = {"n": n, "hit": o, "acc": o / max(1, n)}
            print(f"      {k:>16}: {o:4d}/{n:4d} = {o/max(1,n):.4f}")
        out["tables"][name] = row
    print("\nORACLE " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
