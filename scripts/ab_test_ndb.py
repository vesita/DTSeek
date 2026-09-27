#!/usr/bin/env python
"""Mention-NDB 在 person 卡上的**受控** AB 对照（可重跑）。

两条臂除 `--ndb` 外一切相同：同一冻结基座、同一数据、同一 seed、同一 epochs /
steps-per-epoch / num-layers / batch-size。每个 (seed, arm) 是一个独立进程，
从 `training/train_task_card.py` 的 `AB_METRICS` 行取机器可读指标。

    uv run python scripts/ab_test_ndb.py --seeds 42,43 --epochs 8 --steps-per-epoch 150

判据：主 = `repeat_mention_acc` 的 Δ；次 = `cluster_f1` / `exact_match` / `first_mention_acc`。
脚本同时报**配对 Δ（同 seed 相减）+ 均值 + 极差**，以及参数量 / 单步耗时 / 峰值显存 /
表显存四项代价 —— 差异落进噪声就别写"趋势向好"。
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAIN = "repeat_mention_acc"
SECONDARY = ("cluster_f1", "exact_match", "first_mention_acc", "id_acc", "span_hit", "bg_fp")
ARMS = ("base", "ndb")


def run_one(arm: str, seed: int, args) -> dict:
    out = Path(args.out_dir) / f"{args.card}_{arm}_seed{seed}.pt"
    cmd = [sys.executable, str(ROOT / "training" / "train_task_card.py"),
           "--card", args.card, "--base", args.base, "--out", str(out),
           "--epochs", str(args.epochs), "--steps-per-epoch", str(args.steps_per_epoch),
           "--batch-size", str(args.batch_size), "--num-layers", str(args.num_layers),
           "--seed", str(seed)]
    if arm == "ndb":
        cmd += ["--ndb", "--ndb-read", args.ndb_read,
                "--ndb-levels", args.ndb_levels, "--ndb-slots", args.ndb_slots]
    log = Path(args.out_dir) / f"{args.card}_{arm}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(cmd[1:])}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"[fail] {arm} seed={seed} 退出码 {proc.returncode}，日志 {log}")
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(1)
    for line in proc.stdout.splitlines():
        if line.startswith("AB_METRICS "):
            rec = json.loads(line[len("AB_METRICS "):])
            rec["log"] = str(log)
            return rec
    raise SystemExit(f"{arm} seed={seed} 没有输出 AB_METRICS；日志 {log}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mention-NDB 的受控 AB 对照")
    ap.add_argument("--card", default="person")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seeds", default="42,43", help="逗号分隔；至少 2 个才有极差")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--ndb-read", choices=("true", "pred"), default="pred",
                    help="训练时的读注意力来源：pred=指针自己的预测（部署口径，默认）；"
                         "true=教师强制真值起点。默认曾经是 true，会在训练/评估之间留一条"
                         "口径缝（训练查金标起点、推理只能查预测起点）。")
    ap.add_argument("--ndb-levels", default="1,2")
    ap.add_argument("--ndb-slots", default="8192,4096")
    ap.add_argument("--out-dir", default="/tmp/ab_ndb")
    ap.add_argument("--json-out", default=None, help="把汇总写成 JSON")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print("⚠ 只给了一个 seed：只能给点估计，给不出极差（噪声判据不成立）")

    runs: dict[tuple[str, int], dict] = {}
    for seed in seeds:
        for arm in ARMS:
            runs[(arm, seed)] = run_one(arm, seed, args)

    keys = [MAIN] + list(SECONDARY)
    print(f"\n================ 原始指标（--ndb-read {args.ndb_read}）================")
    for (arm, seed), rec in runs.items():
        m = rec["metrics"]
        print(f"{arm:>4} seed={seed}  " + "  ".join(f"{k}={m.get(k, float('nan')):.4f}" for k in keys))

    print("\n================ 配对 Δ（ndb − base，同 seed）================")
    summary: dict = {"seeds": seeds, "card": args.card, "ndb_read": args.ndb_read, "per_metric": {}}
    for k in keys:
        deltas = [runs[("ndb", s)]["metrics"].get(k, float("nan"))
                  - runs[("base", s)]["metrics"].get(k, float("nan")) for s in seeds]
        mean = statistics.fmean(deltas)
        rng = max(deltas) - min(deltas)
        print(f"{k:>20}: Δ = " + "  ".join(f"{d:+.4f}" for d in deltas)
              + f"   mean={mean:+.4f}  range={rng:.4f}")
        summary["per_metric"][k] = {"deltas": deltas, "mean": mean, "range": rng}

    print("\n================ 代价（实测）================")
    for arm in ARMS:
        recs = [runs[(arm, s)] for s in seeds]
        head = statistics.fmean(r["n_head_params"] for r in recs)
        ndb_p = statistics.fmean(r["n_ndb_params"] for r in recs)
        sps = statistics.fmean(r["sec_per_step"] for r in recs)
        mem = statistics.fmean(r["peak_mem_mb"] for r in recs)
        tbl = max((r.get("ndb_table_gb", 0.0) for r in recs), default=0.0)
        print(f"{arm:>4}: 头部参数={head:,.0f}  NDB门控参数={ndb_p:,.0f}  "
              f"单步={sps*1000:.1f}ms  峰值显存={mem:.0f}MB  表={tbl:.3f}GB")
    summary["cost"] = {
        arm: {
            "n_head_params": statistics.fmean(r["n_head_params"] for r in [runs[(arm, s)] for s in seeds]),
            "n_ndb_params": statistics.fmean(r["n_ndb_params"] for r in [runs[(arm, s)] for s in seeds]),
            "sec_per_step": statistics.fmean(r["sec_per_step"] for r in [runs[(arm, s)] for s in seeds]),
            "peak_mem_mb": statistics.fmean(r["peak_mem_mb"] for r in [runs[(arm, s)] for s in seeds]),
            "table_gb": max((r.get("ndb_table_gb", 0.0)
                             for r in [runs[(arm, s)] for s in seeds]), default=0.0),
        } for arm in ARMS
    }
    summary["ndb_stats"] = {str(s): runs[("ndb", s)]["ndb_stats"] for s in seeds}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n汇总已写 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
