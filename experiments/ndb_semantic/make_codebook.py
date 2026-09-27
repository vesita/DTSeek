#!/usr/bin/env python
"""构建 semantic-NDB 的**冻结**语义键：度量 W + 两级码本。

纪律：
  * 键的构造只看得见 **train split**（与该 seed 的训练完全相同的那一份），
    绝不碰 val —— 否则就是 val 泄漏，指标不可信。
  * W 与码本都是 no_grad 数据统计量（和字面 NDB 的 n-gram 哈希同地位），
    训练时只有 write_gate / read_gate / level_weight 三个门控是参数。

产出 `codebook_seed{seed}.pt`：W [D,D] + 两级码本中心 [K_l, D]（已单位化）。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ndb_semantic"))
OUT = Path(__file__).resolve().parent

from oracle_bound import encode_hidden, feat_of  # noqa: E402
from oracle_vq_learned import make_pairs, train_W  # noqa: E402


def kmeans(P, K, seed, iters=40):
    g = torch.Generator().manual_seed(seed)
    c = P[torch.randperm(P.shape[0], generator=g)[:K]].clone()
    for _ in range(iters):
        a = (P @ c.T).argmax(-1)
        nc, cnt = torch.zeros_like(c), torch.zeros(K)
        nc.index_add_(0, a, P)
        cnt.index_add_(0, a, torch.ones(P.shape[0]))
        e = cnt == 0
        nc[e] = c[e]
        c = nc / nc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", default="8,4", help="两级码本大小（对应字面臂的 levels 1,2）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from dtseek.tasks.plugin import resolve_tasks
    card = resolve_tasks(["person"])["person"]
    data = card.build_dataset(9000)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    print(f"[split] seed={args.seed} val={len(val)} train={len(train)}（与训练逐行同构）")

    def mentions(ds):
        out = []
        for si, d in enumerate(ds):
            for m in sorted(d["spans"], key=lambda x: x["start"]):
                out.append({"s": si, "label": m["label"], "word": m["word"],
                            "start": m["start"], "end": m["end"]})
        return out

    m_tr = mentions(train)
    ht = encode_hidden([d["text"] for d in train if d["spans"]], "checkpoints/base_encoder.pt", device)
    pos_idx = [i for i, d in enumerate(train) if d["spans"]]
    remap = {o: n for n, o in enumerate(pos_idx)}
    m_tr = [dict(m, s=remap[m["s"]]) for m in m_tr]
    ft = {}
    for m in m_tr:
        v = feat_of(ht[m["s"]], m, "mean").float()
        ft[(m["s"], m["start"])] = v / v.norm().clamp_min(1e-9)
    A, B, y = make_pairs(ft, m_tr)
    W, auc_tr = train_W(A, B, y, device)
    print(f"[W] train 配对 {A.shape[0]}，训练 AUC={auc_tr:.4f}")

    P = torch.stack([(v.to(device) @ W).cpu() for v in ft.values()])
    P = P / P.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ks = [int(x) for x in args.ks.split(",")]
    cents = [kmeans(P, k, 7 + i) for i, k in enumerate(ks)]
    # 码本容量体检：每级各码号里的样本数分布（空码会让门控永远关着）
    for k, c in zip(ks, cents):
        cnt = torch.bincount((P @ c.T).argmax(-1), minlength=k)
        print(f"[codebook] K={k} 各码号点数 min/median/max = "
              f"{int(cnt.min())}/{int(cnt.median())}/{int(cnt.max())}")
    out = Path(args.out) if args.out else OUT / f"codebook_seed{args.seed}.pt"
    torch.save({"W": W.cpu(), "centroids": [c.cpu() for c in cents],
                "ks": ks, "train_auc": auc_tr, "split_seed": args.seed}, out)
    print(f"[out] {out}  (W {W.numel()} 个冻结数 + 码本 {sum(ks)}×128)")


if __name__ == "__main__":
    main()
