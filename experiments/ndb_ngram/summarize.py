#!/usr/bin/env python
"""汇总实验日志：每臂×每seed 指标、按类别拆分、配对 Δ（vs literal）、代价、梯度/旁路。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LOGDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ngram_ab")
MAIN = "repeat_mention_acc"
COLS = ("repeat_mention_acc", "first_mention_acc", "id_acc", "cluster_f1", "bg_fp")


def load():
    recs = {}
    for p in sorted(LOGDIR.glob("*.log")):
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("AB_METRICS "):
                r = json.loads(line[len("AB_METRICS "):])
                r["_log"] = str(p)
                recs[r["tag"]] = r
    return recs


def get(recs, arm, seed, tag=None):
    if tag and tag in recs:
        return recs[tag]
    for r in recs.values():
        if r["arm"] == arm and r["seed"] == seed:
            return r
    return None


def main():
    recs = load()
    primary = [("base", 42), ("base", 43),
               ("literal", 42), ("literal", 43),
               ("lngtab", 42), ("lngtab", 43),
               ("ngrammer", 42), ("ngrammer", 43),
               ("nplm", 42), ("nplm", 43)]
    print("=" * 134)
    print("每臂 × 每 seed：主指标 / 首次提及 / 按类别拆分 / 代价   （literal 用 NDB 门控 lr=3e-4；参数化臂 lr=1e-3）")
    print("=" * 134)
    print(f"{'arm':<9}{'seed':>5}{'repeat':>9}{'first':>8}{'id_acc':>8}{'f1':>8}{'bg_fp':>8}"
          f"{'alias':>8}{'lit_same':>9}{'lit_other':>10}{'n_mem':>9}{'s/step':>8}{'peakMB':>8}")
    for a, s in primary:
        r = get(recs, a, s)
        if not r:
            continue
        m, bk = r["metrics"], r.get("by_kind", {})
        print(f"{a:<9}{s:>5}{m[MAIN]:>9.4f}{m['first_mention_acc']:>8.4f}"
              f"{m['id_acc']:>8.4f}{m['cluster_f1']:>8.4f}{m['bg_fp']:>8.4f}"
              f"{(bk.get('alias',{}).get('acc',float('nan'))):>8.4f}"
              f"{(bk.get('literal_same_id',{}).get('acc',float('nan'))):>9.4f}"
              f"{(bk.get('literal_other_id',{}).get('acc',float('nan'))):>10.4f}"
              f"{r.get('n_mem_params',0):>9,}{r['sec_per_step']:>8.4f}{r['peak_mem_mb']:>8.0f}")

    print("\n" + "=" * 134)
    print("配对 Δ（参数化臂 − literal，同 seed）；负 = 不如非参数 NDB")
    print("=" * 134)
    for a, s in primary:
        if a in ("base", "literal"):
            continue
        r, L = get(recs, a, s), get(recs, "literal", s)
        if not r or not L:
            continue
        dm = {k: r["metrics"][k] - L["metrics"][k] for k in COLS}
        bkr, bkL = r.get("by_kind", {}), L.get("by_kind", {})
        da = (bkr.get("alias", {}).get("acc", float("nan"))
              - bkL.get("alias", {}).get("acc", float("nan")))
        dl = (bkr.get("literal_same_id", {}).get("acc", float("nan"))
              - bkL.get("literal_same_id", {}).get("acc", float("nan")))
        print(f"{a:<9}{s:>5}  " + "  ".join(f"{k[:6]}={dm[k]:+.4f}" for k in COLS)
              + f"  alias={da:+.4f}  lit_same={dl:+.4f}")

    print("\n" + "=" * 134)
    print("梯度与旁路消融（bypass = 把该臂输出置零后重评；freeze = 参数不进优化器）")
    print("=" * 134)
    for tag, r in sorted(recs.items(), key=lambda kv: kv[0]):
        g = r.get("grad_summary", {})
        b = r.get("bypass_metrics")
        bstr = (f"bypass repeat={b[MAIN]:.4f} (Δ={b[MAIN]-r['metrics'][MAIN]:+.4f})"
                if b else "bypass=n/a")
        gp = r.get("grad_per_param", {})
        nz = {k: round(v, 5) for k, v in gp.items() if v is not None and v > 0}
        print(f"{tag:<24} arm={r['arm']:<8} seed={r['seed']} freeze={int(r.get('freeze_mem',0))} "
              f"grad_mean={g.get('grad_norm_mean', float('nan')):.4f} "
              f"all_nonzero={g.get('grad_all_nonzero')} | {bstr}")
        print(f"{'':24} 每参数 |grad|>0：{nz}")


if __name__ == "__main__":
    main()
