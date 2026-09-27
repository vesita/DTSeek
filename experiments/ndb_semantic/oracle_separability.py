#!/usr/bin/env python
"""语义键上界的**决定性测量**：冻结基座隐状态里到底有没有「同一人物」的可分离信号。

第一版 oracle 的两个测量缺陷（已修正到本文件）：
  1. 「随机正交阵对照」是**空对照** —— 正交变换逐位保持余弦相似度，所以召回率必然
     一模一样（实测 0.2500 vs 0.2500）。本文件改用**内容无关的随机特征**做对照。
  2. 只看 1-NN 召回不足以判「码本能不能用」，因为没有正对照验对齐。
     本文件加两个正对照：
       * literal_same_id 子集（同字面同 id）的 1-NN 召回 —— 特征图若对齐正确必然很高
       * 同/异人物配对的余弦相似度 AUC（不做任何学习，纯冻结特征）
     以及一个**最宽松**的探针：在冻结特征上训练一个配对分类器（分组交叉验证），
     回答「就算给一个学出来的度量，冻结特征够不够支撑语义键」。

判据（写死在输出里）：
  * 1-NN 召回 <= 随机挑一个更早提及的 prior  => 语义键连瞎猜都不如，不值得做
  * 同/异配对 AUC <= 0.5（或探针 vs 多数类基线无增益）=> 冻结特征里没有该信号
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

from oracle_bound import encode_hidden, feat_of, split_like_train  # noqa: E402


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC（labels: True=正类）。"""
    pos = scores[labels]
    neg = scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # rank-based，处理并列
    allv = torch.cat([pos, neg])
    order = torch.argsort(allv)
    ranks = torch.empty_like(allv, dtype=torch.float)
    ranks[order] = torch.arange(1, allv.numel() + 1, dtype=torch.float)
    # 并列取平均秩
    uniq, inv = torch.unique(allv, return_inverse=True)
    for u_i, u in enumerate(uniq):
        m = allv == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[:pos.numel()].sum()
    return float((r_pos - pos.numel() * (pos.numel() + 1) / 2) /
                 (pos.numel() * neg.numel()))


def main():
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val, _ = split_like_train(seed)
    texts = [d["text"] for d in val]
    hidden = encode_hidden(texts, "checkpoints/base_encoder.pt", device)
    print(f"编码 {len(val)} 条 val 文本，D={hidden[0].shape[-1]}")

    # ---- 按 evaluate_task 口径标注每个提及 ----
    mentions = []   # (sample, label, word, start, end, kind)
    for si, d in enumerate(val):
        seen = []
        for m in sorted(d["spans"], key=lambda x: x["start"]):
            same = any(e["label"] == m["label"] for e in seen)
            kind = "first"
            if same:
                sw_same = any(e["word"] == m["word"] and e["label"] == m["label"] for e in seen)
                sw_oth = any(e["word"] == m["word"] and e["label"] != m["label"] for e in seen)
                kind = ("literal_same_id" if sw_same else
                        "literal_other_id" if sw_oth else "alias")
            mentions.append({"s": si, "label": m["label"], "word": m["word"],
                             "start": m["start"], "end": m["end"], "kind": kind})
            seen.append({"label": m["label"], "word": m["word"]})
    print("提及构成:", Counter(m["kind"] for m in mentions))

    report = {"n_val": len(val), "mention_kinds": dict(Counter(m["kind"] for m in mentions)),
              "features": {}}

    for mode in ("start", "mean", "start2"):
        feats = {}
        for m in mentions:
            feats[(m["s"], m["start"])] = feat_of(hidden[m["s"]], m, mode).float()
        for k in feats:
            feats[k] = feats[k] / feats[k].norm().clamp_min(1e-9)

        # ---- 配对：同人物 vs 异人物 ----
        pairs, same = [], []
        for si, d in enumerate(val):
            ms = [m for m in mentions if m["s"] == si]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a = feats[(si, ms[i]["start"])]
                    b = feats[(si, ms[j]["start"])]
                    pairs.append((a, b, ms[i]["word"], ms[j]["word"]))
                    same.append(ms[i]["label"] == ms[j]["label"])
        same_t = torch.tensor(same)
        cosv = torch.tensor([float(torch.dot(a, b)) for a, b, _, _ in pairs])
        # 只统计**表面不同**的配对（这才是别名键要解决的）
        diffword = torch.tensor([wa != wb for _, _, wa, wb in pairs])
        res = {"n_pairs": len(pairs), "n_same": int(same_t.sum()),
               "cos_same": float(cosv[same_t].mean()),
               "cos_diff": float(cosv[~same_t].mean()),
               "auc_all_pairs": auc(cosv, same_t),
               "n_pairs_diffword": int(diffword.sum()),
               "auc_diffword_pairs": auc(cosv[diffword], same_t[diffword])}
        print(f"\n===== 特征 [{mode}] 配对可分性 =====")
        print(f"  配对 {len(pairs)}（同人物 {int(same_t.sum())}）"
              f"  余弦 same={res['cos_same']:+.4f} diff={res['cos_diff']:+.4f}")
        print(f"  AUC(全部配对) = {res['auc_all_pairs']:.4f}"
              f"   AUC(仅表面不同配对, n={int(diffword.sum())}) = {res['auc_diffword_pairs']:.4f}")

        same_word = torch.tensor([wa == wb for _, _, wa, wb in pairs])
        res["cos_same_word"] = float(cosv[same_word].mean())
        res["cos_diff_word"] = float(cosv[~same_word].mean())
        res["auc_same_word_task"] = auc(cosv, same_word)
        print(f"  [正对照] 余弦 同字面={res['cos_same_word']:+.4f} "
              f"异字面={res['cos_diff_word']:+.4f}  AUC(同字面判别)={res['auc_same_word_task']:.4f}")

        # ---- 1-NN 召回，分子集 + 正对照 ----
        def nn1(subset):
            ok = 0
            for m in subset:
                ear = [e for e in mentions
                       if e["s"] == m["s"] and e["start"] < m["start"]]
                if not ear:
                    continue
                sims = [float(torch.dot(feats[(m["s"], m["start"])],
                                        feats[(m["s"], e["start"])])) for e in ear]
                best = ear[int(torch.tensor(sims).argmax())]
                ok += (best["label"] == m["label"])
            return ok, len(subset)

        def prior(subset):
            tot = []
            for m in subset:
                ear = [e for e in mentions
                       if e["s"] == m["s"] and e["start"] < m["start"]]
                if not ear:
                    continue
                tot.append(sum(1 for e in ear if e["label"] == m["label"]) / len(ear))
            return sum(tot) / max(1, len(tot))

        # 内容无关随机特征对照
        g = torch.Generator().manual_seed(99)
        rnd = {k: torch.randn(feats[k].shape[0], generator=g) for k in feats}
        for k in rnd:
            rnd[k] = rnd[k] / rnd[k].norm().clamp_min(1e-9)

        def nn1_rand(subset):
            ok = 0
            for m in subset:
                ear = [e for e in mentions
                       if e["s"] == m["s"] and e["start"] < m["start"]]
                if not ear:
                    continue
                sims = [float(torch.dot(rnd[(m["s"], m["start"])], rnd[(m["s"], e["start"])]))
                        for e in ear]
                best = ear[int(torch.tensor(sims).argmax())]
                ok += (best["label"] == m["label"])
            return ok, len(subset)

        sub_res = {}
        for name in ("alias", "literal_same_id", "literal_other_id"):
            sub = [m for m in mentions if m["kind"] == name]
            ok, n = nn1(sub)
            rok, _ = nn1_rand(sub)
            pr = prior(sub)
            sub_res[name] = {"n": n, "nn1_recall": ok / max(1, n),
                             "random_feat_nn1": rok / max(1, n), "prior": pr}
            print(f"  [{name:>16} n={n:4d}] nn1={ok/max(1,n):.4f}  "
                  f"随机特征={rok/max(1,n):.4f}  prior={pr:.4f}")
        res["by_kind"] = sub_res
        report["features"][mode] = res

        # ---- 最宽松：在冻结特征上**训练**一个配对分类器（分组 CV） ----
        if mode == "mean":
            X, y, grp = [], [], []
            for si, d in enumerate(val):
                ms = [m for m in mentions if m["s"] == si]
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        a = feats[(si, ms[i]["start"])]
                        b = feats[(si, ms[j]["start"])]
                        X.append(torch.cat([a * b, (a - b).abs(), a + b]))
                        y.append(1.0 if ms[i]["label"] == ms[j]["label"] else 0.0)
                        grp.append(si)
            X = torch.stack(X)
            y = torch.tensor(y)
            grp = torch.tensor(grp)
            print(f"\n===== 训练探针（配对分类器，冻结特征 [{mode}]）X={tuple(X.shape)} =====")
            folds = 5
            accs, aucs, base_accs = [], [], []
            for f in range(folds):
                te = (grp % folds) == f
                tr = ~te
                w = torch.zeros(X.shape[1], requires_grad=True)
                b0 = torch.zeros(1, requires_grad=True)
                opt = torch.optim.Adam([w, b0], lr=0.05)
                Xtr, ytr = X[tr], y[tr]
                for _ in range(400):
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(Xtr @ w + b0, ytr)
                    loss.backward()
                    opt.step()
                logit = X[te] @ w + b0
                pred = (logit > 0).float()
                accs.append(float((pred == y[te]).float().mean()))
                aucs.append(auc(logit.detach(), y[te].bool()))
                base_accs.append(float(max(y[te].mean(), 1 - y[te].mean())))
            probe = {"probe_acc": sum(accs) / folds, "probe_auc": sum(aucs) / folds,
                     "majority_baseline_acc": sum(base_accs) / folds}
            print(f"  探针 acc = {probe['probe_acc']:.4f}   AUC = {probe['probe_auc']:.4f}"
                  f"   多数类基线 acc = {probe['majority_baseline_acc']:.4f}")
            print(f"  增益 = acc - 基线 = {probe['probe_acc'] - probe['majority_baseline_acc']:+.4f}")
            report["probe"] = probe

    # ══════════════════════════════════════════════════════════════════════
    # 码本 K 扫描：有没有任何一个 K 让「同码 → 同人物」成立？
    # ══════════════════════════════════════════════════════════════════════
    mode = "mean"
    feats = {}
    for m in mentions:
        feats[(m["s"], m["start"])] = feat_of(hidden[m["s"]], m, mode).float()
    for k in feats:
        feats[k] = feats[k] / feats[k].norm().clamp_min(1e-9)
    pts = torch.stack(list(feats.values()))
    keys = list(feats.keys())
    print("\n===== 码本 K 扫描（VQ 式语义键的忠实模拟，特征=mean）=====")
    print(f"{'K':>6} {'码内同人物比':>12} {'别名覆盖率':>10} {'别名覆盖内top1':>14} {'别名绝对召回':>12}")
    sweep = {}
    for K in (4, 8, 16, 32, 64, 128, 256):
        K = min(K, pts.shape[0])
        g = torch.Generator().manual_seed(7)
        cent = pts[torch.randperm(pts.shape[0], generator=g)[:K]].clone()
        for _ in range(30):
            assign = (pts @ cent.T).argmax(-1)
            newc = torch.zeros_like(cent)
            cnt = torch.zeros(K)
            newc.index_add_(0, assign, pts)
            cnt.index_add_(0, assign, torch.ones(pts.shape[0]))
            empty = cnt == 0
            newc[empty] = cent[empty]
            cent = newc / newc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        code = {k: int((feats[k] @ cent.T).argmax()) for k in keys}
        # 全局码内纯度：同一码号的提及里，同人物的配对占比 vs 全局先验
        by_code = defaultdict(list)
        for m in mentions:
            by_code[code[(m["s"], m["start"])]].append(m)
        good = tot = 0
        for c, ms in by_code.items():
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    same_s = ms[i]["s"] == ms[j]["s"]
                    same_l = ms[i]["label"] == ms[j]["label"]
                    # 跨样本的同 label 无意义（label 是样本内分配），只算同一样本
                    if not same_s:
                        continue
                    tot += 1
                    good += same_l
        base_rate = (sum(1 for m in mentions for n2 in mentions
                         if m["s"] == n2["s"] and m["start"] < n2["start"]
                         and m["label"] == n2["label"]) /
                     max(1, sum(1 for m in mentions for n2 in mentions
                                if m["s"] == n2["s"] and m["start"] < n2["start"])))
        alias = [m for m in mentions if m["kind"] == "alias"]
        cov = ok = 0
        for m in alias:
            c = code[(m["s"], m["start"])]
            ear = [e for e in mentions if e["s"] == m["s"] and e["start"] < m["start"]
                   and code[(e["s"], e["start"])] == c]
            if ear:
                cov += 1
                votes = Counter(e["label"] for e in ear)
                if votes.most_common(1)[0][0] == m["label"]:
                    ok += 1
        row = {"K": K, "same_sample_in_code_purity": good / max(1, tot),
               "same_sample_pairs_in_code": tot, "global_prior": base_rate,
               "alias_coverage": cov / max(1, len(alias)),
               "alias_top1_given_cov": ok / max(1, cov),
               "alias_recall": ok / max(1, len(alias))}
        sweep[str(K)] = row
        print(f"{K:>6} {row['same_sample_in_code_purity']:>12.4f} "
              f"{row['alias_coverage']:>10.4f} {row['alias_top1_given_cov']:>14.4f} "
              f"{row['alias_recall']:>12.4f}   (同码同人物先验={base_rate:.4f}, 码内同样本配对={tot})")
    report["codebook_sweep"] = sweep

    (OUT / "oracle_separability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT/'oracle_separability.json'}")


if __name__ == "__main__":
    main()
