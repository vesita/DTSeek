#!/usr/bin/env python
"""把已有 checkpoint 的 `repeat_mention_acc` **按提及类别拆开**，量出语义键的真实天花板。

`repeat_mention_acc` 是把「字面重复」和「别名类（字面首次出现）」混在一个分母里的。
语义键只可能改善别名那一栏，所以必须先知道：
  * 现字面 NDB 在**别名类**上已经拿到多少？(headroom = 1 − 这个数)
  * 语义码本在别名类上的绝对召回上界是多少？(oracle_vq_learned.json)

只有 headroom 明显大于码本能覆盖的部分，训 semantic-NDB 才有意义。

用法：
  uv run python experiments/ndb_semantic/eval_by_kind.py \
      --ckpt /tmp/ab_ndb2/person_ndb_seed42.pt --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ndb_semantic"))
OUT = Path(__file__).resolve().parent


def kind_of(truth_spans):
    """给每个真值提及标注类别的列表（与 oracle_bound.repeat_mentions 同口径）。"""
    out, seen = [], []
    for m in sorted(truth_spans, key=lambda x: x["start"]):
        same = any(e["label"] == m["label"] for e in seen)
        if not same:
            k = "first"
        else:
            sw_same = any(e["word"] == m["word"] and e["label"] == m["label"] for e in seen)
            sw_oth = any(e["word"] == m["word"] and e["label"] != m["label"] for e in seen)
            k = "literal_same_id" if sw_same else ("literal_other_id" if sw_oth else "alias")
        out.append(k)
        seen.append({"label": m["label"], "word": m["word"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=9000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card
    from dtseek.tasks.plugin import resolve_tasks
    from dtseek.tasks.runtime import GenericTaskDataset, _rollout, _truth_of
    import mention_ndb_snapshot as mndb_mod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    card = resolve_tasks(["person"])["person"]
    spec = card.spec
    ck = read_card(args.ckpt)
    dec, spec = build_card_decoder(ck, device)
    tokenizer = NanoCharTokenizer()
    doc_encoder, base_ck = load_base_encoder(args.base, device)

    ndb = None
    if ck.get("extra", {}).get("ndb"):
        kw = ck["extra"]["ndb"]["kwargs"]
        ndb = mndb_mod.MentionNDB(**kw).to(device)
        ndb.load_state_dict({k: v.to(device) for k, v in ck["extra"]["ndb"]["state_dict"].items()})
        ndb.eval()
        print(f"[ndb] 从卡里恢复：levels={ndb.levels} slots={ndb.slots} read_true={ndb.read_true}")

    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val = data[:n_val]
    loader = DataLoader(GenericTaskDataset(val, tokenizer, spec), batch_size=args.batch_size,
                        shuffle=False)

    stats = Counter()
    hit = Counter()
    bs = args.batch_size
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mem = doc_encoder(inp, attention_mask=mask)
            B = inp.shape[0]
            if ndb is not None:
                ndb.reset(B, device)
            preds = _rollout(dec, mem, mask, B, spec, ndb=ndb, input_ids=inp)
            for b in range(B):
                truth = _truth_of(batch, b, spec)
                if not truth:
                    continue
                # shuffle=False + drop_last=False => 样本号可精确还原；
                # annotate_all=True 保证 val[i]["spans"] 未被 max_steps 截断
                src_spans = sorted(val[bi * bs + b]["spans"], key=lambda x: x["start"])
                assert len(src_spans) == len(truth), (
                    f"样本 {bi*bs+b} 真值数 {len(src_spans)} != truth {len(truth)}")
                for (lab, s0, e0), sp in zip(truth, src_spans):
                    assert sp["label"] == lab and sp["start"] == s0, "真值对齐失败"
                kinds = kind_of(src_spans)
                got = preds[b]
                pred_at = {(s0, e0): lab for lab, s0, e0 in got}
                for (lab, s0, e0), k in zip(truth, kinds):
                    if k == "first":
                        continue
                    # 与 evaluate_task 逐字同口径：**没发射出这个 span 的提及整个跳过**，
                    # 不进分母。少了这一步分母会大一倍多，指标对不上日志。
                    if (s0, e0) not in pred_at:
                        continue
                    stats[k] += 1
                    if pred_at[(s0, e0)] == lab:
                        hit[k] += 1

    print(f"\n===== {args.tag or args.ckpt} seed={args.seed} 按提及类别拆开 =====")
    tot_n = tot_ok = 0
    for k in ("literal_same_id", "literal_other_id", "alias"):
        n, o = stats[k], hit[k]
        tot_n += n
        tot_ok += o
        print(f"  {k:>16}: {o:4d}/{n:4d} = {o/max(1,n):.4f}")
    print(f"  {'repeat 合计':>16}: {tot_ok:4d}/{tot_n:4d} = {tot_ok/max(1,tot_n):.4f}")
    res = {"ckpt": args.ckpt, "seed": args.seed, "tag": args.tag,
           "by_kind": {k: {"n": stats[k], "hit": hit[k],
                           "acc": hit[k] / max(1, stats[k])} for k in stats},
           "repeat_acc": tot_ok / max(1, tot_n)}
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
