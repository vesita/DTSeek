#!/usr/bin/env python
"""汇总 AB：三臂 × seed 的主/次指标 + 代价 + 别名类拆解，并做噪声判据。

数据来源：
  * base / literal 参考：/tmp/ab_ndb2/person_{base,ndb}_seed{42,43}.log 里最后的 AB_METRICS 行
  * semantic：本目录 ab_run.log 里 tag=semantic_seed* 的 AB_METRICS 行
  * 校验臂：ab_run.log 里 tag=validate_* 的行，用来证明本目录的复刻脚本与参考逐位一致
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ABLOG = OUT / "ab_run.log"
REF = Path("/tmp/ab_ndb2")
KEYS = ("repeat_mention_acc", "first_mention_acc", "id_acc", "cluster_f1", "bg_fp",
        "exact_match", "span_hit")


def metrics_of(log: Path) -> dict | None:
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("AB_METRICS "):
            return json.loads(line[len("AB_METRICS "):])
    return None


def main():
    rows: dict[tuple[str, int], dict] = {}
    for seed in (42, 43):
        for name, arm in (("base", "base"), ("ndb", "literal")):
            r = metrics_of(REF / f"person_{name}_seed{seed}.log")
            if r:
                rows[(arm, seed)] = r
    if ABLOG.exists():
        for line in ABLOG.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("AB_METRICS "):
                continue
            r = json.loads(line[len("AB_METRICS "):])
            tag = r.get("tag", "")
            if tag.startswith("semantic_"):
                rows[("semantic", r["seed"])] = r
            elif tag.startswith("validate_"):
                rows[(f"CHECK-{r['arm']}", r["seed"])] = r

    print("=" * 108)
    print("AB 原始指标（reference 来自 /tmp/ab_ndb2，semantic 来自本目录 ab_run.log）")
    print("=" * 108)
    hdr = f"{'arm':>14} {'seed':>4} " + " ".join(f"{k[:14]:>15}" for k in KEYS)
    print(hdr)
    for (arm, seed), r in sorted(rows.items()):
        m = r["metrics"]
        print(f"{arm:>14} {seed:>4} " + " ".join(f"{m.get(k, float('nan')):>15.4f}" for k in KEYS))

    print("\n" + "=" * 108)
    print("配对 Δ vs base（同 seed）；semantic 只有 42/43 两 seed，与 base 同 seed 相减")
    print("=" * 108)
    for arm in ("literal", "semantic"):
        deltas = {k: [] for k in KEYS}
        seeds_used = []
        for seed in (42, 43):
            if (arm, seed) in rows and ("base", seed) in rows:
                seeds_used.append(seed)
                for k in KEYS:
                    deltas[k].append(rows[(arm, seed)]["metrics"].get(k, float("nan"))
                                   - rows[("base", seed)]["metrics"].get(k, float("nan")))
        if not seeds_used:
            continue
        print(f"\n{arm} (seeds {seeds_used})")
        for k in KEYS:
            d = deltas[k]
            mean = statistics.fmean(d)
            rng = (max(d) - min(d)) if len(d) > 1 else float("nan")
            print(f"  {k:>22}: Δ = " + "  ".join(f"{x:+.4f}" for x in d)
                  + f"   mean={mean:+.4f}" + (f"  range={rng:.4f}" if len(d) > 1 else ""))

    print("\n" + "=" * 108)
    print("别名类拆解（by_kind，口径与 evaluate_task 一致：未发射的提及不进分母）")
    print("=" * 108)
    for (arm, seed), r in sorted(rows.items()):
        bk = r.get("by_kind")
        if not bk:
            continue
        s = "  ".join(f"{k}={v['hit']}/{v['n']}={v['acc']:.4f}"
                      for k, v in bk.items() if v["n"])
        print(f"{arm:>14} seed={seed}: {s}")

    print("\n" + "=" * 108)
    print("代价")
    print("=" * 108)
    for (arm, seed), r in sorted(rows.items()):
        print(f"{arm:>14} seed={seed}: 步时={r['sec_per_step']*1000:7.1f}ms  "
              f"峰值显存={r['peak_mem_mb']:6.0f}MB  "
              f"头部参数={r['n_head_params']:,}  门控参数={r['n_ndb_params']}  "
              f"冻结统计={r.get('n_frozen_stats',0):,}  表={r.get('ndb_table_gb',0):.5f}GB  "
              f"train={r['train_sec']:.1f}s")

    print("\n" + "=" * 108)
    print("复刻校验（本目录脚本 vs 既有日志，必须逐位一致）")
    print("=" * 108)
    for (arm, seed), r in sorted(rows.items()):
        if not arm.startswith("CHECK-"):
            continue
        real = arm.replace("CHECK-", "")
        ref = rows.get((real, seed))
        if not ref:
            print(f"{arm} seed={seed}: 无参考")
            continue
        same = all(abs(r["metrics"].get(k, 0) - ref["metrics"].get(k, 0)) < 1e-12 for k in KEYS)
        print(f"{arm} seed={seed}: repeat_mention_acc={r['metrics']['repeat_mention_acc']:.16f} "
              f"vs 参考 {ref['metrics']['repeat_mention_acc']:.16f} => "
              f"{'逐位一致' if same else '★不一致★'}")


if __name__ == "__main__":
    sys.exit(main())
