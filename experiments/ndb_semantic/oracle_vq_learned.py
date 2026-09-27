#!/usr/bin/env python
"""钉死「学出来的度量 + 离散码本」这个上界，并做强反证对照。

oracle_metric.py 发现：
  * 原始余弦：别名类 1-NN 召回 0.275（低于 prior 0.373）—— 冻结特征里**取不出来**
  * 在 train 上学一个线性度量 W 后：别名类 1-NN 召回 0.774（远超 prior）
所以「冻结基座隐状态 → 语义键」这个假设的成败，全押在「W 是不是真的学到了共指，
而不是抓到了数据集/模板的旁门」上。本文件用三个对照把它钉死：

  1. **打乱标签对照**：训练 W 时把同一样本内的 label 随机置换（保结构、毁共指），
     再测 val 别名 1-NN 召回。若仍高 => W 抓的是文档/模板旁门，不是共指。
  2. **同字面异人对照**：只看「表面字面相同但真值不同人」的配对（如两个不同男性都叫「他」），
     看 W 把它们分开的能力（AUC）。
  3. **W 空间里的离散码本 K 扫描**：这才是语义键的忠实模拟 ——
     同码号 = 键相同；别名覆盖率 / 码内纯度 / 绝对召回。

只有 (1) 打乱后掉回 prior、(3) 在高覆盖下有正召回，才值得去训 semantic-NDB。
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
OUT = Path(__file__).resolve().parent

from oracle_bound import encode_hidden, feat_of  # noqa: E402
from oracle_metric import mention_list, build_feats, nn_recall  # noqa: E402
from oracle_separability import auc  # noqa: E402
from oracle_bound import split_like_train  # noqa: E402

CACHE = OUT / "hidden_cache.pt"


def get_hidden(device):
    if CACHE.exists():
        blob = torch.load(CACHE, weights_only=False)
        print(f"[hidden] 复用缓存 {CACHE}")
        return blob["hv"], blob["ht"], blob["m_val"], blob["m_tr"], blob["pos_idx"]
    val, train = split_like_train(42)
    m_val, m_tr = mention_list(val), mention_list(train)
    hv = encode_hidden([d["text"] for d in val], "checkpoints/base_encoder.pt", device)
    pos_idx = [i for i, d in enumerate(train) if d["spans"]]
    ht = encode_hidden([train[i]["text"] for i in pos_idx], "checkpoints/base_encoder.pt", device)
    remap = {old: new for new, old in enumerate(pos_idx)}
    m_tr = [dict(m, s=remap[m["s"]]) for m in m_tr]
    torch.save({"hv": hv, "ht": ht, "m_val": m_val, "m_tr": m_tr, "pos_idx": pos_idx}, CACHE)
    return hv, ht, m_val, m_tr, pos_idx


def make_pairs(feats, mentions):
    A, B, y = [], [], []
    bys = defaultdict(list)
    for m in mentions:
        bys[m["s"]].append(m)
    for si, ms in bys.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                A.append(feats[(si, ms[i]["start"])])
                B.append(feats[(si, ms[j]["start"])])
                y.append(1.0 if ms[i]["label"] == ms[j]["label"] else 0.0)
    return torch.stack(A), torch.stack(B), torch.tensor(y)


def train_W(A, B, y, device, iters=3000, seed=5, shuffle_labels=False, perm=None):
    D = A.shape[1]
    g = torch.Generator().manual_seed(seed)
    W = (torch.eye(D) + 0.01 * torch.randn(D, D, generator=g)).to(device).requires_grad_(True)
    s = torch.nn.Parameter(torch.tensor(4.0, device=device))
    opt = torch.optim.Adam([W, s], lr=1e-3)
    A, B, y = A.to(device), B.to(device), y.to(device)
    if shuffle_labels:
        y = perm.to(device)
    for _ in range(iters):
        opt.zero_grad()
        sim = F.cosine_similarity(A @ W, B @ W, dim=-1)
        F.binary_cross_entropy_with_logits(s * sim, y).backward()
        opt.step()
    with torch.no_grad():
        sim = F.cosine_similarity(A @ W, B @ W, dim=-1).cpu()
        tr_auc = auc(sim, y.cpu().bool())
    return W.detach(), tr_auc


def project(feats, W, device):
    out = {}
    for k, v in feats.items():
        p = (v.to(device) @ W).cpu()
        out[k] = p / p.norm().clamp_min(1e-9)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hv, ht, m_val, m_tr, _ = get_hidden(device)
    print(f"val 提及 {len(m_val)}  train 提及 {len(m_tr)}")

    fv = build_feats(hv, m_val, "mean")
    ft = build_feats(ht, m_tr, "mean")
    A, B, y = make_pairs(ft, m_tr)
    print(f"train 配对 {A.shape[0]}（同人物 {int(y.sum())}）")

    report = {}
    alias = [m for m in m_val if m["kind"] == "alias"]
    prior = 0.3731

    # ---------- 1. 真 W vs 打乱标签 W ----------
    W, tr_auc = train_W(A, B, y, device)
    g = torch.Generator().manual_seed(2024)
    perm = y[torch.randperm(y.shape[0], generator=g)]
    Wp, tr_auc_p = train_W(A, B, y, device, shuffle_labels=True, perm=perm)
    print(f"\n===== 1. 打乱标签对照 =====")
    print(f"  真 W   : train AUC = {tr_auc:.4f}")
    print(f"  打乱 W : train AUC = {tr_auc_p:.4f}")

    # ---------- 2. 配对可分性：真 W vs 打乱 W（val 上分组，无泄漏） ----------
    Av, Bv, yv = make_pairs(fv, m_val)
    _sw = []
    for si in sorted({m["s"] for m in m_val}):
        ms = [m for m in m_val if m["s"] == si]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                _sw.append(1.0 if ms[i]["word"] == ms[j]["word"] else 0.0)
    same_word = torch.tensor(_sw)
    assert same_word.shape[0] == yv.shape[0], (same_word.shape, yv.shape)
    def pair_auc(Wm):
        with torch.no_grad():
            sim = F.cosine_similarity(Av.to(device) @ Wm, Bv.to(device) @ Wm, dim=-1).cpu()
            simr = F.cosine_similarity(Av, Bv, dim=-1)
        dw = same_word < 0.5
        ss = same_word > 0.5
        return {"auc_all": auc(sim, yv.bool()),
                "auc_diffword": auc(sim[dw], yv.bool()[dw]),
                "auc_sameword": auc(sim[ss], yv.bool()[ss]),
                "raw_cos_auc_all": auc(simr, yv.bool())}
    pa, pap = pair_auc(W), pair_auc(Wp)
    print(f"\n===== 2. val 配对可分性（分组无泄漏）=====")
    for name, d in (("真 W", pa), ("打乱 W", pap)):
        print(f"  {name}: AUC(全部)={d['auc_all']:.4f}  AUC(表面不同配对)={d['auc_diffword']:.4f}  "
              f"AUC(表面相同配对)={d['auc_sameword']:.4f}")
    print(f"  原始余弦 AUC(全部) = {pa['raw_cos_auc_all']:.4f}")
    report["pair_auc"] = {"real_W": pa, "shuffled_W": pap}

    fvW, fvWp = project(fv, W, device), project(fv, Wp, device)
    print(f"\n===== 别名类 1-NN 召回（val）=====")
    print(f"  prior = {prior:.4f}")
    for name, f in (("原始余弦", fv), ("真 W", fvW), ("打乱 W", fvWp)):
        o, n = nn_recall(f, m_val, alias, same_doc=True)
        print(f"  {name:>8}: {o:.4f}  (n={n})")
        report[f"alias_nn1_{name}"] = o

    # ---------- 3. W 空间里的离散码本 K 扫描 ----------
    print(f"\n===== 3. 离散码本 K 扫描（W 空间，别名类）=====")
    pts = torch.stack(list(fvW.values()))
    keys = list(fvW.keys())
    pos_of = {}
    for m in m_val:
        pos_of.setdefault(m["s"], []).append(m)
    print(f"{'K':>6} {'别名覆盖':>9} {'码内纯度':>9} {'覆盖内top1':>10} {'绝对召回':>9}")
    sweep = {}
    for K in (8, 16, 32, 64, 128, 256, 512):
        K = min(K, pts.shape[0])
        g = torch.Generator().manual_seed(7)
        cent = pts[torch.randperm(pts.shape[0], generator=g)[:K]].clone()
        for _ in range(30):
            asg = (pts @ cent.T).argmax(-1)
            nc, cnt = torch.zeros_like(cent), torch.zeros(K)
            nc.index_add_(0, asg, pts)
            cnt.index_add_(0, asg, torch.ones(pts.shape[0]))
            e = cnt == 0
            nc[e] = cent[e]
            cent = nc / nc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        code = {k: int((fvW[k] @ cent.T).argmax()) for k in keys}
        # 码内纯度：同一码号里、同一样本内的配对，同人物占比
        by_code = defaultdict(list)
        for m in m_val:
            by_code[code[(m["s"], m["start"])]].append(m)
        good = tot = 0
        for c, ms in by_code.items():
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    if ms[i]["s"] != ms[j]["s"]:
                        continue
                    tot += 1
                    good += ms[i]["label"] == ms[j]["label"]
        cov = ok = 0
        for m in alias:
            c = code[(m["s"], m["start"])]
            ear = [e for e in m_val if e["s"] == m["s"] and e["start"] < m["start"]
                   and code[(e["s"], e["start"])] == c]
            if ear:
                cov += 1
                if Counter(e["label"] for e in ear).most_common(1)[0][0] == m["label"]:
                    ok += 1
        row = {"K": K, "alias_coverage": cov / max(1, len(alias)),
               "in_code_purity": good / max(1, tot), "n_in_code_pairs": tot,
               "alias_top1_given_cov": ok / max(1, cov),
               "alias_recall": ok / max(1, len(alias))}
        sweep[str(K)] = row
        print(f"{K:>6} {row['alias_coverage']:>9.4f} {row['in_code_purity']:>9.4f} "
              f"{row['alias_top1_given_cov']:>10.4f} {row['alias_recall']:>9.4f}"
              f"   (码内同样本配对={tot}, 同码同人物先验=0.2534)")
    report["vq_sweep_W"] = sweep

    # 词面拆解：别名类里，学过度量能召回的是哪些词
    ok_by_word = Counter()
    tot_by_word = Counter()
    for m in alias:
        ear = [e for e in m_val if e["s"] == m["s"] and e["start"] < m["start"]]
        if not ear:
            continue
        q = fvW[(m["s"], m["start"])]
        M = torch.stack([fvW[(e["s"], e["start"])] for e in ear])
        best = ear[int((M @ q).argmax())]
        tot_by_word[m["word"]] += 1
        if best["label"] == m["label"]:
            ok_by_word[m["word"]] += 1
    print("\n  别名类各词面的 1-NN 召回（真 W）:")
    for w, t in tot_by_word.most_common(8):
        print(f"    {w!r:>6}: {ok_by_word[w]:4d}/{t:4d} = {ok_by_word[w]/t:.3f}")
    report["alias_by_word"] = {w: {"n": t, "hit": ok_by_word[w]} for w, t in tot_by_word.items()}

    (OUT / "oracle_vq_learned.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\n写出 {OUT/'oracle_vq_learned.json'}")


if __name__ == "__main__":
    main()
