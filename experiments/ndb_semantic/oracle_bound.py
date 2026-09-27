#!/usr/bin/env python
"""语义键 Mention-NDB 的**数据侧 oracle 上界**计算（不训练任何东西）。

回答三个问题（对应父任务的三问）：

  Q1 别名类重复提及占重复提及的比例是多少？
     口径与 `runtime.evaluate_task` 的 `repeat_mention_acc` 完全一致：
     在 val 集上按位置顺序遍历每个样本的真值 spans，label 此前出现过 => 重复提及。
     再按「字面是否完全重复」把它二分。

  Q2 用冻结基座在该提及处的隐状态做最近邻：对每个别名重复提及，它此前出现的提及里，
     语义最近的那个是否就是同一个真值 id？召回率多少？
     同时给出**忠实于 NDB 机制**的 VQ 版本：k-means 码本 → 同码号取多数 id。

  Q3 由此得到语义键能覆盖的上界（字面已覆盖 ∪ 语义可召回）。

对照（防测量 bug）：
  * `prior_recall`：不看语义、随机挑一个更早提及的期望命中率 = 同 id 更早提及占比。
    语义 1-NN 必须显著高于它，否则「语义键」等价于瞎猜。
  * 随机投影特征对照：把隐状态乘一个固定随机正交阵，召回率应≈prior。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════
# 数据：与 training/train_task_card.py 逐行同构的构建 + 切分
# ══════════════════════════════════════════════════════════════════════════
def split_like_train(seed: int, samples: int = 9000):
    """完全复刻 train_task_card.py 的数据构造与切分顺序。"""
    from dtseek.tasks.plugin import resolve_tasks

    card = resolve_tasks(["person"])["person"]
    data = card.build_dataset(samples)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    return val, train


def repeat_mentions(data: list[dict]) -> list[dict]:
    """按 evaluate_task 的口径抽出所有「重复提及」，并做字面分解。

    返回每条记录：
      sample / idx / label / word / start / end / earlier(更早提及的记录) / kind
    kind:
      literal_same_id  该字面（整词）此前出现过且**同 id** -> 现有字面键直接可覆盖
      literal_other_id 该字面此前出现过但**只有不同 id** -> 字面键会撞车（覆盖但答错）
      alias            该字面此前从未出现过 -> 别名类（现有键完全检索不到）
    """
    recs: list[dict] = []
    for si, d in enumerate(data):
        spans = sorted(d["spans"], key=lambda x: x["start"])
        seen: list[dict] = []
        for m in spans:
            lab = m["label"]
            same_word_same = [e for e in seen if e["word"] == m["word"] and e["label"] == lab]
            same_word_other = [e for e in seen if e["word"] == m["word"] and e["label"] != lab]
            is_repeat = any(e["label"] == lab for e in seen)
            if is_repeat:
                if same_word_same:
                    kind = "literal_same_id"
                elif same_word_other:
                    kind = "literal_other_id"
                else:
                    kind = "alias"
                recs.append({
                    "sample": si, "text": d["text"], "label": lab, "word": m["word"],
                    "start": m["start"], "end": m["end"], "kind": kind,
                    "earlier": [dict(e) for e in seen],
                })
            seen.append({"label": lab, "word": m["word"], "start": m["start"], "end": m["end"]})
    return recs


# ══════════════════════════════════════════════════════════════════════════
# 语义特征：冻结基座在该提及处的隐状态
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_hidden(texts: list[str], base: str, device, batch: int = 32,
                  tok_limit: int = 128) -> list[torch.Tensor]:
    """返回每条文本的 [L, D] 隐状态（L = 真实字符数，去掉 padding）。"""
    from dtseek.tasks.artifacts import load_base_encoder
    from nano_char_tokenizer import NanoCharTokenizer

    enc, _ = load_base_encoder(base, device)
    enc.eval()
    tok = NanoCharTokenizer()
    out: list[torch.Tensor] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids, am = [], []
        for t in chunk:
            e = tok.encode(t, max_length=tok_limit, padding=True)
            ids.append(e["input_ids"])
            am.append(e["attention_mask"])
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        am_t = torch.tensor(am, dtype=torch.bool, device=device)
        h = enc(ids_t, attention_mask=am_t)          # [B, L, D]
        for b, t in enumerate(chunk):
            out.append(h[b, :len(t)].float().cpu())
    del enc
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return out


def feat_of(h: torch.Tensor, m: dict, mode: str) -> torch.Tensor:
    s = min(int(m["start"]), h.shape[0] - 1)
    e = min(max(int(m["end"]) - 1, s), h.shape[0] - 1)
    if mode == "start":
        v = h[s]
    elif mode == "mean":
        v = h[s:e + 1].mean(0)
    elif mode == "start2":
        v = torch.cat([h[s], h[min(s + 1, h.shape[0] - 1)]])
    else:
        raise ValueError(mode)
    return v


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════
def compute(seed: int, base: str, device, feat_modes, vq_ks, cache: Path | None):
    if cache and cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[data] 复用缓存 {cache}")
    else:
        val, train = split_like_train(seed)
        blob = {"val": val, "train": train}
        if cache:
            cache.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
            print(f"[data] 写出 {cache}")
    val = blob["val"]

    all_reps = repeat_mentions(val)
    n_rep = len(all_reps)
    kinds = Counter(r["kind"] for r in all_reps)
    print(f"\n===== Q1 字面分解（seed={seed}, val 样本 {len(val)}）=====")
    for k, v in kinds.most_common():
        print(f"  {k:>18}: {v:5d}  {v/max(1,n_rep)*100:5.1f}%")
    print(f"  {'总重复提及':>18}: {n_rep:5d}")
    alias = [r for r in all_reps if r["kind"] == "alias"]
    print(f"  别名类（字面此前从未出现）占重复提及 = {len(alias)/max(1,n_rep)*100:.1f}%")

    # 词面构成：别名类到底是什么词
    print("  别名类的词面 top10:", Counter(r["word"] for r in alias).most_common(10))
    print("  字面同 id 类的词面 top10:",
          Counter(r["word"] for r in all_reps if r["kind"] == "literal_same_id").most_common(10))

    # ---- prior 对照：随机挑一个更早提及的期望命中率 ----
    def prior(r):
        e = r["earlier"]
        return sum(1 for x in e if x["label"] == r["label"]) / max(1, len(e))

    priors = [prior(r) for r in all_reps]
    prior_alias = [prior(r) for r in alias]
    print(f"\n  prior（随机更早提及就是同 id）(全体) = {sum(priors)/max(1,len(priors)):.4f}")
    print(f"  prior（别名子集）              = {sum(prior_alias)/max(1,len(prior_alias)):.4f}")

    # ---- 语义特征 ----
    texts = [d["text"] for d in val]
    hidden = encode_hidden(texts, base, device)
    hidden_by_sample = {i: h for i, h in enumerate(hidden)}

    dim = hidden[0].shape[-1]
    g = torch.Generator().manual_seed(1234)
    rand_proj = torch.randn(dim, dim, generator=g)
    rand_proj = torch.linalg.qr(rand_proj).Q          # 固定随机正交阵（保余弦对照）

    results = {"seed": seed, "n_val": len(val), "n_repeat": n_rep,
               "kinds": dict(kinds),
               "alias_ratio": len(alias) / max(1, n_rep),
               "prior_all": sum(priors) / max(1, len(priors)),
               "prior_alias": sum(prior_alias) / max(1, len(prior_alias)),
               "features": {}}

    for mode in feat_modes:
        # 预取**所有**提及（含首现）的特征 —— 更早提及里有首现，缺了就 KeyError
        feats = {}          # (sample, start) -> tensor
        all_mentions = 0
        for si, d in enumerate(val):
            for m in d["spans"]:
                feats[(si, m["start"])] = feat_of(hidden_by_sample[si], m, mode)
                all_mentions += 1
        assert len(feats) == all_mentions, "键冲突：同一样本两个提及起止位置相同"
        print(f"\n[feat:{mode}] 覆盖 {all_mentions} 个提及（含首现）")

        def nn_hit(r, use_rand=False, k=1):
            if not r["earlier"]:
                return False
            q = feats[(r["sample"], r["start"])]
            q = q / q.norm().clamp_min(1e-9)
            cand = []
            for e in r["earlier"]:
                v = feats[(r["sample"], e["start"])]
                cand.append((e, v))
            M = torch.stack([v for _, v in cand])
            M = M / M.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            if use_rand:
                q = rand_proj @ q
                M = M @ rand_proj.T
            sim = M @ q
            order = torch.argsort(sim, descending=True)[:k]
            labs = [cand[int(i)][0]["label"] for i in order]
            if k == 1:
                return labs[0] == r["label"]
            return r["label"] in labs

        # 每个子集分别报（alias 是关键）
        for subset_name, subset in (("alias", alias), ("all_repeat", all_reps)):
            hit1 = sum(nn_hit(r) for r in subset)
            hit3 = sum(nn_hit(r, k=3) for r in subset)
            hit_rand = sum(nn_hit(r, use_rand=True) for r in subset)
            pr = sum(prior(r) for r in subset) / max(1, len(subset))
            key = f"{mode}|{subset_name}"
            results["features"][key] = {
                "n": len(subset), "nn1_recall": hit1 / max(1, len(subset)),
                "nn3_recall": hit3 / max(1, len(subset)),
                "random_proj_nn1": hit_rand / max(1, len(subset)),
                "prior": pr,
            }
            print(f"\n===== Q2 语义 1-NN [{mode}] 子集={subset_name} (n={len(subset)}) =====")
            print(f"  nn1 召回 = {hit1/max(1,len(subset)):.4f}   nn3 召回 = {hit3/max(1,len(subset)):.4f}")
            print(f"  随机正交对照 = {hit_rand/max(1,len(subset)):.4f}   prior = {pr:.4f}")

        # ---- VQ 码本模拟（忠实于 NDB 机制）----
        if mode == feat_modes[0]:
            # 码本在**全部提及**（含首现）上拟合，覆盖整个提及分布
            all_pts = torch.stack(list(feats.values()))
            all_pts = all_pts / all_pts.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            print(f"[VQ] 码本拟合点数 = {all_pts.shape[0]}")
            results["vq"] = {}
            for K in vq_ks:
                K = min(K, all_pts.shape[0])
                # 简单 k-means（球面 + 有限迭代，够用作上界估计）
                gg = torch.Generator().manual_seed(7)
                cent = all_pts[torch.randperm(all_pts.shape[0], generator=gg)[:K]].clone()
                assign = None
                for _ in range(25):
                    sim = all_pts @ cent.T
                    assign = sim.argmax(-1)
                    newc = torch.zeros_like(cent)
                    cnt = torch.zeros(K)
                    newc.index_add_(0, assign, all_pts)
                    cnt.index_add_(0, assign, torch.ones(all_pts.shape[0]))
                    empty = cnt == 0
                    newc[empty] = cent[empty]
                    cent = newc / newc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                code = {}
                for i, r in enumerate(all_reps):
                    code[(r["sample"], r["start"])] = int((feats[(r["sample"], r["start"])] /
                                                           feats[(r["sample"], r["start"])].norm().clamp_min(1e-9)
                                                           @ cent.T).argmax())
                for subset_name, subset in (("alias", alias), ("all_repeat", all_reps)):
                    ok = 0
                    cov = 0
                    for r in subset:
                        c = code[(r["sample"], r["start"])]
                        same = [e for e in r["earlier"]
                                if code.get((r["sample"], e["start"])) == c]
                        if same:
                            cov += 1
                            votes = Counter(e["label"] for e in same)
                            if votes.most_common(1)[0][0] == r["label"]:
                                ok += 1
                        else:
                            # 空码：门控关闭，退回分类头 —— 这里按「不可召回」计
                            pass
                    results["vq"][f"K{K}|{subset_name}"] = {
                        "n": len(subset), "coverage": cov / max(1, len(subset)),
                        "top1_given_cov": ok / max(1, cov),
                        "recall": ok / max(1, len(subset)),
                    }
                    print(f"\n===== Q2 VQ 码本 [{mode}] K={K} 子集={subset_name} =====")
                    print(f"  覆盖(码内至少一个更早提及) = {cov/max(1,len(subset)):.4f}"
                          f"  覆盖内 top1 = {ok/max(1,cov):.4f}  绝对召回 = {ok/max(1,len(subset)):.4f}")

    # ---- Q3 上界 ----
    best_alias = max((v["nn1_recall"] for k, v in results["features"].items()
                      if k.endswith("|alias")), default=0.0)
    lit_ok = kinds.get("literal_same_id", 0)
    # 字面臂上界：字面同 id 的重复提及可被现有键覆盖（其余不是它的职责）
    upper_literal = lit_ok / max(1, n_rep)
    # 语义臂上界：字面可覆盖的 + 别名类里语义能召回的
    upper_sem = (lit_ok + best_alias * len(alias)) / max(1, n_rep)
    results["upper_bound"] = {
        "literal_only_recall_upper": upper_literal,
        "semantic_union_recall_upper": upper_sem,
        "best_alias_nn1": best_alias,
        "note": "上界 = 数据侧可检索比例，不含门控/训练损失；实际 acc 只会更低",
    }
    print("\n===== Q3 上界 =====")
    print(f"  现字面键可覆盖（literal_same_id） = {upper_literal:.4f}")
    print(f"  语义键并集上界 = {upper_sem:.4f}（别名类最佳 nn1 = {best_alias:.4f}）")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--feat-modes", default="start,mean")
    ap.add_argument("--vq-ks", default="64,256,1024")
    ap.add_argument("--json-out", default=str(OUT / "oracle_bound.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = OUT / f"person_val_seed{args.seed}.json"
    res = compute(args.seed, args.base, device,
                  args.feat_modes.split(","), [int(k) for k in args.vq_ks.split(",")],
                  cache)
    Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\n写出 {args.json_out}")


if __name__ == "__main__":
    main()
