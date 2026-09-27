#!/usr/bin/env python
"""离散键的码本选型：随机码本(LSH 式) vs k-means，以及多码本并集。

k-means 扫描（oracle_vq_learned.py）显示绝对召回在 K=8 最好（0.319）但覆盖率已经很低。
k-means 会把容量浪费在「全局密度」上；NDB 要的是**地址一致**，不是聚类质量。
所以这里测另一族码本：固定随机码本（球面 LSH）。
  collide(v_i, v_j) <=> argmax_k <Wv_i, c_k> == argmax_k <Wv_j, c_k>
等价于「在 K 个随机方向上取最大者」——K 越大，碰撞所需夹角越小（越纯、覆盖越低）。
再测多码本并集（模拟字面 NDB 的多级 (1,2) 混合）。

产出：哪一档的**别名绝对召回**最高，以及它对应的覆盖率/纯度。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ndb_semantic"))
OUT = Path(__file__).resolve().parent

from oracle_bound import split_like_train  # noqa: E402
from oracle_metric import build_feats, nn_recall  # noqa: E402
from oracle_vq_learned import get_hidden, make_pairs, train_W  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get_hidden 会在缓存缺失时自动重建（hidden_cache.pt 是派生缓存，不入交付）
    hv, ht, m_val, m_tr, _ = get_hidden(device)
    fv = build_feats(hv, m_val, "mean")
    ft = build_feats(ht, m_tr, "mean")
    A, B, y = make_pairs(ft, m_tr)
    W, tr_auc = train_W(A, B, y, device)
    print(f"W: train AUC={tr_auc:.4f}")

    def proj(d):
        out = {}
        for k, v in d.items():
            p = (v.to(device) @ W).cpu()
            out[k] = p / p.norm().clamp_min(1e-9)
        return out
    pv = proj(fv)
    keys = list(pv.keys())
    P = torch.stack(list(pv.values()))
    alias = [m for m in m_val if m["kind"] == "alias"]
    n_alias = len(alias)
    prior = 0.3731

    def eval_codes(codes_of, name):
        """codes_of: list of dicts key -> code（多级）。取并集口径：任一级同码即覆盖。"""
        cov = ok = 0
        for m in alias:
            ear = [e for e in m_val if e["s"] == m["s"] and e["start"] < m["start"]]
            votes = Counter()
            for e in ear:
                for cd in codes_of:
                    if cd[(e["s"], e["start"])] == cd[(m["s"], m["start"])]:
                        votes[e["label"]] += 1
            if votes:
                cov += 1
                if votes.most_common(1)[0][0] == m["label"]:
                    ok += 1
        # 码内纯度（同一样本内、至少一级同码的配对里，同人物占比）
        good = tot = 0
        bys = defaultdict(list)
        for m in m_val:
            bys[m["s"]].append(m)
        for si, ms in bys.items():
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    if any(cd[(si, ms[i]["start"])] == cd[(si, ms[j]["start"])] for cd in codes_of):
                        tot += 1
                        good += ms[i]["label"] == ms[j]["label"]
        row = {"coverage": cov / n_alias, "top1_given_cov": ok / max(1, cov),
               "recall": ok / n_alias, "in_code_purity": good / max(1, tot),
               "n_in_code_pairs": tot}
        print(f"  {name:>26}: 覆盖={row['coverage']:.4f} top1|覆盖={row['top1_given_cov']:.4f} "
              f"绝对召回={row['recall']:.4f} 纯度={row['in_code_purity']:.4f} (配对={tot})")
        return row

    print(f"\n===== 随机码本（球面 LSH）=====  prior={prior:.4f}")
    res = {}
    D = P.shape[1]
    for K in (16, 64, 256, 1024, 4096, 8192):
        g = torch.Generator().manual_seed(11 + K)
        C = F.normalize(torch.randn(K, D, generator=g), dim=-1)
        cd = {k: int((pv[k] @ C.T).argmax()) for k in keys}
        res[f"rand{K}"] = eval_codes([cd], f"随机码本 K={K}")

    print("\n===== 多码本并集（模拟多级）=====")
    for K, L in ((64, 2), (64, 4), (256, 2), (256, 4), (1024, 4)):
        cds = []
        for l in range(L):
            g = torch.Generator().manual_seed(1000 + 97 * K + l)
            C = F.normalize(torch.randn(K, D, generator=g), dim=-1)
            cds.append({k: int((pv[k] @ C.T).argmax()) for k in keys})
        res[f"rand{K}x{L}"] = eval_codes(cds, f"{L}×随机码本 K={K}")

    print("\n===== k-means 对照（同 W 空间）=====")
    for K in (8, 16, 32, 64):
        g = torch.Generator().manual_seed(7)
        cent = P[torch.randperm(P.shape[0], generator=g)[:K]].clone()
        for _ in range(40):
            asg = (P @ cent.T).argmax(-1)
            nc, cnt = torch.zeros_like(cent), torch.zeros(K)
            nc.index_add_(0, asg, P)
            cnt.index_add_(0, asg, torch.ones(P.shape[0]))
            e = cnt == 0
            nc[e] = cent[e]
            cent = nc / nc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cd = {k: int((pv[k] @ cent.T).argmax()) for k in keys}
        res[f"kmeans{K}"] = eval_codes([cd], f"k-means K={K}")

    # 参照：连续 1-NN（非离散）
    o, _ = nn_recall(pv, m_val, alias, same_doc=True)
    print(f"\n  参照：连续 1-NN 绝对召回 = {o:.4f}（离散键达不到这个数）")
    res["continuous_nn1"] = o
    (OUT / "oracle_codebook.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"写出 {OUT/'oracle_codebook.json'}")


if __name__ == "__main__":
    main()
