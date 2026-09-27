#!/usr/bin/env python
"""汇总：参考臂（复用 /tmp/ab_ndb2 日志）+ 参数化臂（results/*.json）→ 一张表 + results.json。

拆分口径（alias / literal_same_id）对参考臂取自 `experiments/ndb_semantic/by_kind_ref.jsonl`
（同一测量台，逐位复核过），参数化臂取自本目录 results/*.json。

用法：uv run python experiments/ndb_parametric/consolidate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REF = Path("/tmp/ab_ndb2")
COLUMNS = ("repeat_mention_acc", "first_mention_acc", "id_acc", "cluster_f1", "bg_fp")
ARMS = ("base", "literal", "slots16", "slots32", "diffwrite", "fastweight", "fastweight_cls")


def ref_arm(prefix: str) -> dict[int, dict]:
    out = {}
    for seed in (42, 43):
        line = [l for l in (REF / f"person_{prefix}_seed{seed}.log").read_text().splitlines()
                if l.startswith("AB_METRICS ")]
        if not line:
            continue
        rec = json.loads(line[-1][len("AB_METRICS "):])
        out[seed] = {"metrics": rec["metrics"], "n_mem_params": rec.get("n_ndb_params", 0),
                     "n_head_params": rec.get("n_head_params"),
                     "sec_per_step": rec.get("sec_per_step"),
                     "peak_mem_mb": rec.get("peak_mem_mb"),
                     "mem_stats": rec.get("ndb_stats", {}), "by_kind": None,
                     "grad_info": {}, "bypass_metrics": None, "source": str(REF)}
    # 拆分口径来自同一测量台
    bk = HERE.parent / "ndb_semantic" / "by_kind_ref.jsonl"
    if bk.exists():
        for line in bk.read_text().splitlines():
            r = json.loads(line)
            if r["tag"] == f"person_{prefix}_seed{r['seed']}" and r["seed"] in out:
                out[r["seed"]]["by_kind"] = r["by_kind"]
    return out


def main():
    table = {}
    for arm in ("base", "literal"):
        table[arm] = ref_arm("base" if arm == "base" else "ndb")
    for arm in ARMS:
        if arm in table:
            continue
        table[arm] = {}
        for seed in (42, 43):
            p = HERE / "results" / f"{arm}_seed{seed}.json"
            if p.exists():
                table[arm][seed] = json.loads(p.read_text())

    lines, summary = [], {"arms": {}, "gpu_note": None}
    def fmt(v, n=4):
        return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

    lines.append("臂 × seed 全部指标（person 卡；8 epoch × 150 step；batch 64；num_layers 2）")
    lines.append("")
    hdr = (f"{'arm':16s} {'seed':>4s} " + " ".join(f"{c[:12]:>12s}" for c in COLUMNS)
           + f" {'alias':>8s} {'lit_same':>9s} {'params':>8s} {'s/step':>8s} {'peakMB':>8s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for arm in ARMS:
        for seed in (42, 43):
            r = table.get(arm, {}).get(seed)
            if not r:
                lines.append(f"{arm:16s} {seed:>4d}  (缺)")
                continue
            m = r["metrics"]; b = r.get("by_kind") or {}
            lines.append(
                f"{arm:16s} {seed:>4d} "
                + " ".join(f"{fmt(m.get(c)):>12s}" for c in COLUMNS)
                + f" {fmt((b.get('alias') or {}).get('acc')):>8s}"
                + f" {fmt((b.get('literal_same_id') or {}).get('acc')):>9s}"
                + f" {r.get('n_mem_params', 0):>8d}"
                + f" {fmt(r.get('sec_per_step'), 4):>8s}"
                + f" {fmt(r.get('peak_mem_mb'), 1):>8s}")
        lines.append("")

    # 配对 Δ（同 seed，臂 − base / 臂 − literal）
    lines.append("配对 Δ（同 seed 逐对相减，再给均值）")
    lines.append("")
    lines.append(f"{'arm':16s} {'metric':24s} {'seed42':>10s} {'seed43':>10s} {'mean':>10s}")
    lines.append("-" * 76)
    summary["deltas"] = {}
    for arm in ARMS:
        if arm in ("base", "literal"):
            continue
        for ref in ("base", "literal"):
            for c in ("repeat_mention_acc", "first_mention_acc", "id_acc", "cluster_f1", "bg_fp"):
                ds = []
                for seed in (42, 43):
                    a, b = table.get(arm, {}).get(seed), table.get(ref, {}).get(seed)
                    if a and b:
                        ds.append(a["metrics"][c] - b["metrics"][c])
                if not ds:
                    continue
                summary["deltas"].setdefault(arm, {})[f"vs_{ref}_{c}"] = ds
                if c == "repeat_mention_acc":
                    lines.append(f"{arm:16s} {('Δ '+ref+' '+c)[:24]:24s} "
                                 + " ".join(f"{d:>10.4f}" for d in ds)
                                 + f" {sum(ds)/len(ds):>10.4f}")
    lines.append("")

    # 成本 / 梯度 / 旁路
    lines.append("成本、梯度非空、旁路消融")
    lines.append("")
    for arm in ARMS:
        for seed in (42, 43):
            r = table.get(arm, {}).get(seed)
            if not r:
                continue
            g = r.get("grad_info") or {}
            bp = r.get("bypass_metrics")
            tag = arm if arm in ("base", "literal") else arm
            if g:
                lines.append(f"{tag:16s} seed{seed}  grad_mean/step={g.get('grad_norm_mean_per_step'):.6g} "
                             f"非零步={g.get('n_steps_with_nonzero_grad')}/{g.get('n_grad_steps')} "
                             f"None={g.get('n_param_grad_none')} per_param={json.dumps(g.get('per_param_last_grad_norm'))}")
            if bp:
                lines.append(f"{'':16s} seed{seed}  旁路 repeat={bp['repeat_mention_acc']:.4f} "
                             f"(开={r['metrics']['repeat_mention_acc']:.4f}, "
                             f"Δ={r['metrics']['repeat_mention_acc']-bp['repeat_mention_acc']:+.4f}) "
                             f"first={bp['first_mention_acc']:.4f} (开={r['metrics']['first_mention_acc']:.4f})")
    lines.append("")

    snaps = HERE / "logs" / "gpu_snapshots.txt"
    if snaps.exists():
        lines.append("GPU 占用快照（每次训练前；本机 8151MiB，同机另有他人在跑）")
        lines.append(snaps.read_text().strip())
        summary["gpu_note"] = snaps.read_text().strip()

    out = "\n".join(lines)
    (HERE / "summary_table.txt").write_text(out)
    summary["arms"] = {a: {str(s): {"metrics": r["metrics"], "by_kind": r.get("by_kind"),
                                    "n_mem_params": r.get("n_mem_params"),
                                    "sec_per_step": r.get("sec_per_step"),
                                    "peak_mem_mb": r.get("peak_mem_mb"),
                                    "grad_info": r.get("grad_info"),
                                    "bypass_metrics": r.get("bypass_metrics"),
                                    "mem_stats": r.get("mem_stats")}
                            for s, r in table[a].items()} for a in ARMS}
    (HERE / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
