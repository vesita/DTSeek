#!/usr/bin/env python
"""语义键 Mention-NDB 的受控 AB：base / literal-NDB / semantic-NDB 三臂。

**本脚本逐行复刻 `training/train_task_card.py` 的初始化与训练顺序**（含 RNG 消耗顺序），
唯一的差别是键的构造。为了证明复刻成立，脚本自带校验：
  `--arm base --seed 42` 必须复现日志里的 0.3773，
  `--arm literal --seed 42` 必须复现 0.8408。
复现不了就不许拿 semantic 的数字下结论。

用法：
  uv run python experiments/ndb_semantic/ab_semantic.py --arm literal --seed 42
  uv run python experiments/ndb_semantic/ab_semantic.py --arm semantic --seed 42 \
      --codebook experiments/ndb_semantic/codebook_seed42.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ndb_semantic"))
OUT = Path(__file__).resolve().parent


def kind_of(spans):
    out, seen = [], []
    for m in sorted(spans, key=lambda x: x["start"]):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("base", "literal", "semantic"), required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=9000)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ndb-lr", type=float, default=3e-4)
    ap.add_argument("--codebook", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
    from dtseek.tasks.artifacts import load_base_encoder
    from dtseek.tasks.plugin import resolve_tasks
    from dtseek.tasks.runtime import (GenericTaskDataset, evaluate_task, task_loss,
                                      _rollout, _truth_of)
    import mention_ndb_snapshot as lit
    import semantic_ndb as sem

    # ── 与 train_task_card.py 完全相同的初始化顺序（RNG 消耗必须一致）──
    card = resolve_tasks(["person"])["person"]
    spec = card.spec
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NanoCharTokenizer()
    doc_encoder, base_ck = load_base_encoder(args.base, device)
    hidden_dim = base_ck["hidden_dim"]
    decoder = RobustARSliceDecoder(hidden_dim=hidden_dim, num_classes=spec.num_classes,
                                   num_heads=4, num_layers=args.num_layers).to(device)
    n_head = sum(p.numel() for p in decoder.parameters())

    ndb = None
    n_ndb = n_frozen = 0
    if args.arm == "literal":
        ndb = lit.MentionNDB(hidden_dim=hidden_dim, num_classes=spec.num_classes,
                             vocab_size=8192, levels=(1, 2), slots=[8192, 4096],
                             max_table_gb=0.25, read_true=True).to(device)
    elif args.arm == "semantic":
        if not args.codebook:
            raise SystemExit("--arm semantic 需要 --codebook")
        cb = torch.load(args.codebook, map_location="cpu", weights_only=False)
        ndb = sem.SemanticNDB(hidden_dim=hidden_dim, num_classes=spec.num_classes,
                              W=cb["W"], centroids=cb["centroids"],
                              max_table_gb=0.25, read_true=True).to(device)
        sem.attach_memory_hook(ndb, doc_encoder)
        n_frozen = cb["W"].numel() + sum(c.numel() for c in cb["centroids"])
        print(f"  semantic 码本：K={[c.shape[0] for c in cb['centroids']]} "
              f"train_AUC={cb['train_auc']:.4f} 冻结数={n_frozen:,}")
    if ndb is not None:
        n_ndb = sum(p.numel() for p in ndb.parameters())
        ndb.reset(args.batch_size, device)
        print(f"  NDB：{ndb.extra_repr()} | 门控参数 {n_ndb:,}")

    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr_head, weight_decay=1e-4)
    ndb_opt = (torch.optim.AdamW(ndb.parameters(), lr=args.ndb_lr, weight_decay=0.0)
               if ndb is not None else None)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))

    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    it = iter(train_loader)
    n_steps = 0
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                mem = doc_encoder(inp, attention_mask=mask)
            loss = task_loss(decoder, mem, mask, batch, spec, device, ndb=ndb)
            optimizer.zero_grad()
            if ndb_opt is not None:
                ndb_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            if ndb_opt is not None:
                ndb_opt.step()
            scheduler.step()
            running += loss.item()
            n_steps += 1
        if epoch % 4 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss {running/args.steps_per_epoch:.3f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0

    decoder.eval()
    metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=ndb)
    print(f"  验证：{json.dumps({k: round(v,4) for k,v in metrics.items() if isinstance(v,float)}, ensure_ascii=False)}")

    # ── 按提及类别拆开（语义键只可能动 alias 那一栏）──
    # ★ evaluate_task 结尾会把 encoder/decoder 切回 .train()，而 decoder 默认 dropout=0.1：
    #   不切回 eval 就带着 dropout 再 rollout 一遍，指标会比 evaluate_task 差一截（实测过）。
    doc_encoder.eval()
    decoder.eval()
    stats, hit = Counter(), Counter()
    bs = args.batch_size
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mem = doc_encoder(inp, attention_mask=mask)
            B = inp.shape[0]
            if ndb is not None:
                ndb.reset(B, device)
            preds = _rollout(decoder, mem, mask, B, spec, ndb=ndb, input_ids=inp)
            for b in range(B):
                truth = _truth_of(batch, b, spec)
                if not truth:
                    continue
                src = sorted(val[bi * bs + b]["spans"], key=lambda x: x["start"])
                assert len(src) == len(truth)
                kinds = kind_of(src)
                pred_at = {(s0, e0): lab for lab, s0, e0 in preds[b]}
                for (lab, s0, e0), k in zip(truth, kinds):
                    if k == "first" or (s0, e0) not in pred_at:
                        continue          # 与 evaluate_task 同口径：未发射的跳过
                    stats[k] += 1
                    if pred_at[(s0, e0)] == lab:
                        hit[k] += 1
    by_kind = {k: {"n": stats[k], "hit": hit[k], "acc": hit[k] / max(1, stats[k])}
               for k in ("literal_same_id", "literal_other_id", "alias")}
    print("  按类别：" + "  ".join(f"{k}={v['hit']}/{v['n']}({v['acc']:.4f})"
                                   for k, v in by_kind.items()))
    if ndb is not None:
        print(f"  NDB 统计：{json.dumps(ndb.stats(), ensure_ascii=False)}")

    rec = {"arm": args.arm, "seed": args.seed, "tag": args.tag or f"{args.arm}_seed{args.seed}",
           "metrics": metrics, "by_kind": by_kind,
           "n_head_params": n_head, "n_ndb_params": n_ndb, "n_frozen_stats": n_frozen,
           "train_sec": train_sec, "n_steps": n_steps, "sec_per_step": train_sec / max(1, n_steps),
           "peak_mem_mb": peak_mb,
           "ndb_table_gb": (ndb.table_gb(args.batch_size) if ndb is not None else 0.0),
           "ndb_stats": (ndb.stats() if ndb is not None else {})}
    print("AB_METRICS " + json.dumps(rec, ensure_ascii=False))
    return rec


if __name__ == "__main__":
    main()
