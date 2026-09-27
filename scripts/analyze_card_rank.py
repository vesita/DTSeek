"""任务卡的参数有效秩分析 —— 用来看某张卡"该有多大"。

插件化之后每张卡是独立产物，容量就成了独立旋钮：秩利用率低 = 这张卡被喂了用不上的容量，
可以缩层/缩宽换体积与速度，甚至换更好的泛化。

指标（沿用 dev-notes/03 的口径，便于纵向比较）：
  - 有效秩 erank(W) = exp(-Σ p_i ln p_i)，p_i = σ_i / Σσ          （Roy & Vetterli 奇异值熵法）
  - 满秩比 util = erank / min(M, N)                                （< 30% 视为秩坍缩）
  - 可压缩比 M·N / (erank·(M+N))：低秩分解需要这么多参数，比值越大越该压

用法：
    uv run python scripts/analyze_card_rank.py --card checkpoints/cards/pronoun.pt
    uv run python scripts/analyze_card_rank.py --card checkpoints/cards/*.pt
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.artifacts import read_card  # noqa: E402


def effective_rank(w: torch.Tensor) -> float:
    if w.dim() != 2:
        return float("nan")
    s = torch.linalg.svdvals(w.float())
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def analyze(path: str) -> dict:
    ck = read_card(path)
    sd = ck["decoder"]
    total = sum(v.numel() for v in sd.values())
    rows = []
    for name, w in sd.items():
        if w.dim() != 2:
            continue
        m, n = w.shape
        er = effective_rank(w)
        util = er / min(m, n)
        compress = (m * n) / max(1e-9, er * (m + n))
        rows.append({"name": name, "shape": (m, n), "params": m * n,
                     "erank": er, "util": util, "compress": compress})
    rows.sort(key=lambda r: r["util"])
    return {"task": ck["task"], "total": total, "rows": rows,
            "classes": len(ck["spec"]["classes"]), "path": path}


def main() -> int:
    ap = argparse.ArgumentParser(description="任务卡有效秩分析")
    ap.add_argument("--card", nargs="+", required=True)
    args = ap.parse_args()

    for path in args.card:
        r = analyze(path)
        print("=" * 78)
        print(f"【{r['task']}】{path}   {r['classes']} 类，解码器参数 {r['total']:,}")
        print("=" * 78)
        print(f"  {'模块':34s} {'形状':>12s} {'参数':>9s} {'有效秩':>8s} {'满秩比':>7s} {'可压缩':>7s}")
        for row in r["rows"]:
            flag = "  ⚠坍缩" if row["util"] < 0.30 else ("  ·偏低" if row["util"] < 0.50 else "")
            print(f"  {row['name']:34s} {str(row['shape']):>12s} {row['params']:>9,} "
                  f"{row['erank']:>8.2f} {row['util']*100:>6.1f}% {row['compress']:>6.2f}x{flag}")
        low = [r for r in r["rows"] if r["util"] < 0.5]
        print(f"\n  满秩比 < 50% 的模块：{len(low)}/{len(r['rows'])}，占参数 "
              f"{sum(x['params'] for x in low):,} / {sum(x['params'] for x in r['rows']):,}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
