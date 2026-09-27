#!/usr/bin/env python
"""在「冻结基座 + 离散码本」这个假设下，把上界算到**最宽松**的一档。

前面的测量留下了两个必须解释的张力：
  * 原始余弦：同人物配对 AUC = 0.453~0.481（**低于 0.5**，方向反了）
  * 但训练一个配对分类器（分组 CV）得到 AUC = 0.770、acc 只比多数类高 +0.016

后者说「冻结特征里也许有可学的东西」。所以本文件做三件事把它钉死：

  A. 正对照 —— 冻结特征到底编码了什么？
     跨文档的字面 1-NN：拿一条 val 提及，在**其他文档**里找特征最近邻，
     看是不是同一个词。若这个很高 => 特征主要编码「字面/上下文」，不是「同一人物」。
     同时报「同一样本内」的字面 1-NN 作对照。

  B. 探针细节 —— 0.770 的 AUC 是真的身份信号，还是在利用合成模板？
     报 balanced accuracy、在「表面不同的配对」上的 AUC，以及
     「把同一样本的两条提及配对的模板相似度」这一捷径的可利用性。

  C. **最宽松上界** —— 用 train split 学一个度量（在冻结特征上线性投影 W），
     再用它做别名类的 1-NN 检索。这是「冻结基座 + 学出来的码本」能达到的最好情况。
     如果连这个都超不过 prior，那这个假设就不成立（不是「还没调好」）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent

from oracle_bound import encode_hidden, feat_of, repeat_mentions, split_like_train  # noqa: E402
from oracle_separability import auc  # noqa: E402


def mention_list(data):
    out = []
    for si, d in enumerate(data):
        seen = []
        for m in sorted(d["spans"], key=lambda x: x["start"]):
            same = any(e["label"] == m["label"] for e in seen)
            kind = "first"
            if same:
                sw_same = any(e["word"] == m["word"] and e["label"] == m["label"] for e in seen)
                sw_oth = any(e["word"] == m["word"] and e["label"] != m["label"] for e in seen)
                kind = ("literal_same_id" if sw_same else
                        "literal_other_id" if sw_oth else "alias")
            out.append({"s": si, "label": m["label"], "word": m["word"],
                        "start": m["start"], "end": m["end"], "kind": kind})
            seen.append({"label": m["label"], "word": m["word"]})
    return out


def build_feats(hidden, mentions, mode="mean"):
    f = {}
    for m in mentions:
        v = feat_of(hidden[m["s"]], m, mode).float()
        f[(m["s"], m["start"])] = v / v.norm().clamp_min(1e-9)
    return f


def nn_recall(feats, mentions, subset, same_doc=True, exclude_self_doc=False):
    ok = n = 0
    for m in subset:
        pool = [e for e in mentions if e is not m]
        if same_doc:
            pool = [e for e in pool if e["s"] == m["s"] and e["start"] < m["start"]]
        else:
            pool = [e for e in pool if (e["s"] != m["s"]) if exclude_self_doc]
        if not pool:
            continue
        q = feats[(m["s"], m["start"])]
        M = torch.stack([feats[(e["s"], e["start"])] for e in pool])
        j = int((M @ q).argmax())
        # 跨文档口径：看是不是同一个词
        ok += (pool[j]["word"] == m["word"]) if not same_doc else (pool[j]["label"] == m["label"])
        n += 1
    return ok / max(1, n), n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val, train = split_like_train(42)
    print(f"val={len(val)} train={len(train)}")
    m_val = mention_list(val)
    m_tr = mention_list(train)
    texts_val = [d["text"] for d in val]
    texts_tr = [d["text"] for d in train if d["spans"]]     # 只编码正例（省时）

    hv = encode_hidden(texts_val, "checkpoints/base_encoder.pt", device)
    ht = encode_hidden(texts_tr, "checkpoints/base_encoder.pt", device)
    # train 侧样本号要重映射（跳过了背景样本）
    pos_idx = [i for i, d in enumerate(train) if d["spans"]]
    remap = {old: new for new, old in enumerate(pos_idx)}
    m_tr = [dict(m, s=remap[m["s"]]) for m in m_tr]

    report = {}
    for mode in ("start", "mean", "start2"):
        fv = build_feats(hv, m_val, mode)
        # ---------- A. 正对照：特征编码字面 ----------
        cross, n_cross = nn_recall(fv, m_val, m_val, same_doc=False, exclude_self_doc=True)
        withins, n_within = nn_recall(fv, m_val, m_val, same_doc=True)
        print(f"\n===== A. 正对照 [{mode}] =====")
        print(f"  跨文档 1-NN 命中『同一个词』 = {cross:.4f}  (n={n_cross})")
        print(f"  同文档 1-NN 命中『同一个人』 = {withins:.4f}  (n={n_within})")
        report[f"{mode}_surface_cross_doc"] = cross
        report[f"{mode}_identity_within_doc"] = withins

        # ---------- B. 探针细节 ----------
        if mode == "mean":
            X, y, grp, dw = [], [], [], []
            for si, d in enumerate(val):
                ms = [m for m in m_val if m["s"] == si]
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        a, b = fv[(si, ms[i]["start"])], fv[(si, ms[j]["start"])]
                        X.append(torch.cat([a * b, (a - b).abs(), a + b]))
                        y.append(1.0 if ms[i]["label"] == ms[j]["label"] else 0.0)
                        dw.append(1.0 if ms[i]["word"] != ms[j]["word"] else 0.0)
                        grp.append(si)
            X, y = torch.stack(X), torch.tensor(y)
            grp, dw = torch.tensor(grp), torch.tensor(dw)
            folds, res = 5, {"acc": [], "auc": [], "auc_dw": [], "bal": [], "base": []}
            for f in range(folds):
                te, tr = (grp % folds) == f, (grp % folds) != f
                w = torch.zeros(X.shape[1], requires_grad=True)
                b0 = torch.zeros(1, requires_grad=True)
                opt = torch.optim.Adam([w, b0], lr=0.05)
                for _ in range(500):
                    opt.zero_grad()
                    F.binary_cross_entropy_with_logits(X[tr] @ w + b0, y[tr]).backward()
                    opt.step()
                lo = (X[te] @ w + b0).detach()
                pred = lo > 0
                res["acc"].append(float((pred == y[te].bool()).float().mean()))
                res["auc"].append(auc(lo, y[te].bool()))
                sel = dw[te].bool()
                res["auc_dw"].append(auc(lo[sel], y[te].bool()[sel]))
                tp = float((pred & y[te].bool()).sum()); fn = float((~pred & y[te].bool()).sum())
                tn = float((~pred & ~y[te].bool()).sum()); fp = float((pred & ~y[te].bool()).sum())
                res["bal"].append(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)))
                res["base"].append(float(max(y[te].mean(), 1 - y[te].mean())))
            m = {k: sum(v) / folds for k, v in res.items()}
            print(f"\n===== B. 配对探针细节 [{mode}] =====")
            print(f"  acc={m['acc']:.4f} (多数类 {m['base']:.4f})  balanced_acc={m['bal']:.4f}")
            print(f"  AUC(全部)={m['auc']:.4f}   AUC(仅表面不同配对)={m['auc_dw']:.4f}")
            report["probe_detail"] = m

            # ---------- C. 最宽松：在 train 上学度量 W，val 上做别名 1-NN ----------
            ft = build_feats(ht, m_tr, mode)
            pairs_tr, ytr = [], []
            for si in {m["s"] for m in m_tr}:
                ms = [m for m in m_tr if m["s"] == si]
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        pairs_tr.append((ft[(si, ms[i]["start"])], ft[(si, ms[j]["start"])]))
                        ytr.append(1.0 if ms[i]["label"] == ms[j]["label"] else 0.0)
            ytr = torch.tensor(ytr)
            print(f"\n===== C. 学一个度量 W（train 配对 {len(pairs_tr)}）=====")
            D = pairs_tr[0][0].shape[0]
            g = torch.Generator().manual_seed(5)
            W = (torch.eye(D) + 0.01 * torch.randn(D, D, generator=g)).to(device).requires_grad_(True)
            s = torch.nn.Parameter(torch.tensor(4.0, device=device))
            opt = torch.optim.Adam([W, s], lr=1e-3)
            A = torch.stack([p[0] for p in pairs_tr]).to(device)
            Bm = torch.stack([p[1] for p in pairs_tr]).to(device)
            ytr = ytr.to(device)
            for it in range(3000):
                opt.zero_grad()
                sim = F.cosine_similarity(A @ W, Bm @ W, dim=-1)
                loss = F.binary_cross_entropy_with_logits(s * sim, ytr)
                loss.backward()
                opt.step()
            with torch.no_grad():
                sim = F.cosine_similarity(A @ W, Bm @ W, dim=-1).cpu()
                ytr = ytr.cpu()
                print(f"  train 上 AUC = {auc(sim, ytr.bool()):.4f}  loss={loss.item():.4f}")

            fvW = {k: (v.to(device) @ W.detach()).cpu() for k, v in fv.items()}
            for k in fvW:
                fvW[k] = fvW[k] / fvW[k].norm().clamp_min(1e-9)
            alias = [m for m in m_val if m["kind"] == "alias"]
            okw, nw = nn_recall(fvW, m_val, alias, same_doc=True)
            oko, no = nn_recall(fv, m_val, alias, same_doc=True)
            prior = 0.3731
            print(f"  别名类 1-NN 召回：原始特征={oko:.4f}  学过度量={okw:.4f}  prior={prior:.4f}")
            report["learned_metric"] = {"alias_nn1_raw": oko, "alias_nn1_learned": okw,
                                        "alias_n": nw, "prior_alias": prior}
            # 学过度量下，按 kind 全报一遍
            for name in ("literal_same_id", "literal_other_id"):
                sub = [m for m in m_val if m["kind"] == name]
                o, n = nn_recall(fvW, m_val, sub, same_doc=True)
                print(f"  [{name:>16} n={n}] 学过度量 1-NN 召回 = {o:.4f}")
                report["learned_metric"][f"{name}_nn1_learned"] = o

    (OUT / "oracle_metric.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\n写出 {OUT/'oracle_metric.json'}")


if __name__ == "__main__":
    main()
