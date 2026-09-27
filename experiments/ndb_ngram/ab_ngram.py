#!/usr/bin/env python
"""可学习 n-gram 结构 vs 非参数 NDB 的受控 AB。

**逐行复刻 `training/train_task_card.py` 的初始化与训练顺序**（含 RNG 消耗顺序），
唯一的差别是记忆/特征结构。自带测量台校验：

    --arm base    --seed 42  → repeat_mention_acc 必须 = 0.3772683858643744
    --arm literal --seed 42  → repeat_mention_acc 必须 = 0.8407517309594461

参数化臂（lngtab / ngrammer / nplm）额外做 RNG 对齐：先用一个同参的 `MentionNDB`
占位对象消耗掉与 literal 臂**完全相同**的 torch RNG，再建真身并把 RNG 状态恢复到
占位之后 —— 于是下游的数据 shuffle 顺序与 literal 臂逐位相同，配对 Δ 里不掺
「数据顺序不同」这个噪声。

用法：
    uv run python experiments/ndb_ngram/ab_ngram.py --arm lngtab --seed 42
    uv run python experiments/ndb_ngram/ab_ngram.py --arm lngtab --seed 42 --freeze-mem
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
sys.path.insert(0, str(ROOT / "experiments" / "ndb_ngram"))

NDB_KW = dict(vocab_size=8192, levels=(1, 2), slots=[8192, 4096],
              max_table_gb=0.25, read_true=True)


def kind_of(spans):
    """与 ndb_semantic/eval_by_kind.py 同口径的提及类别拆分。"""
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


class GradTracker:
    """设备端累计记忆模块的梯度范数，**每步不做 .item() 同步**（不污染 sec_per_step）。"""

    def __init__(self, module):
        self.module = module
        self.sq = None       # Σ ||g||²
        self.mx = None       # max ||g||
        self.n_nz = None     # 梯度非零的步数
        self.n = 0

    def step(self):
        if self.module is None:
            return
        s = None
        for p in self.module.parameters():
            if p.grad is None:
                continue
            v = p.grad.detach().float().pow(2).sum()
            s = v if s is None else s + v
        if s is None:
            return
        self.n += 1
        nrm = s.sqrt()
        self.sq = nrm if self.sq is None else self.sq + nrm
        self.mx = nrm if self.mx is None else torch.maximum(self.mx, nrm)
        nz = (nrm > 0).float()
        self.n_nz = nz if self.n_nz is None else self.n_nz + nz

    def summary(self) -> dict:
        if self.sq is None:
            return {"grad_steps": 0}
        return {"grad_steps": self.n,
                "grad_norm_mean": float(self.sq) / max(1, self.n),
                "grad_norm_max": float(self.mx),
                "grad_nonzero_steps": int(self.n_nz),
                "grad_all_nonzero": bool(self.n_nz.item() == self.n)}


def grad_of(module) -> dict:
    """训练结束后逐个参数报 |grad|（用于断言「表确实拿到了非零梯度」）。"""
    out = {}
    if module is None:
        return out
    for n, p in module.named_parameters():
        out[n] = (None if p.grad is None else float(p.grad.detach().float().norm()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("base", "literal", "lngtab", "ngrammer", "nplm"),
                    required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-mem", type=float, default=1e-3,
                    help="参数化记忆的学习率（NDB 门控参考值是 3e-4）")
    ap.add_argument("--samples", type=int, default=9000)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-mem", action="store_true",
                    help="旁路消融：把记忆参数从优化器移除（仍前向、仍记梯度）")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
    from dtseek.decoder.mention_ndb import MentionNDB
    from dtseek.tasks.artifacts import load_base_encoder
    from dtseek.tasks.plugin import resolve_tasks
    from dtseek.tasks.runtime import (GenericTaskDataset, evaluate_task, task_loss,
                                      _rollout, _truth_of)
    from ngram_memories import build_memory

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

    mem = None
    n_mem = 0
    rng_aligned = False
    if args.arm == "literal":
        mem = MentionNDB(hidden_dim=hidden_dim, num_classes=spec.num_classes,
                         **NDB_KW).to(device)
    elif args.arm in ("lngtab", "ngrammer", "nplm"):
        # RNG 对齐：占位对象消耗掉与 literal 臂相同的随机数，再建真身并恢复状态
        _dummy = MentionNDB(hidden_dim=hidden_dim, num_classes=spec.num_classes, **NDB_KW)
        rng_aligned = True
        rng_after_literal = torch.get_rng_state()
        del _dummy
        mem = build_memory(args.arm, hidden_dim, spec.num_classes, vocab_size=8192).to(device)
        torch.set_rng_state(rng_after_literal)
    if mem is not None:
        n_mem = sum(p.numel() for p in mem.parameters())
        mem.reset(args.batch_size, device)
        print(f"  mem={type(mem).__name__} | 参数 {n_mem:,} | rng_aligned={rng_aligned}")

    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr_head, weight_decay=1e-4)
    mem_params = [p for p in mem.parameters()] if mem is not None else []
    mem_opt = None
    if mem is not None and not args.freeze_mem:
        mem_opt = torch.optim.AdamW(mem_params, lr=args.lr_mem, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))

    tracker = GradTracker(mem)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
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
                memf = doc_encoder(inp, attention_mask=mask)
            loss = task_loss(decoder, memf, mask, batch, spec, device, ndb=mem)
            optimizer.zero_grad()
            if mem_opt is not None:
                mem_opt.zero_grad(set_to_none=True)
            loss.backward()
            tracker.step()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            if mem_opt is not None:
                mem_opt.step()
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
    metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=mem)
    print(f"  验证：{json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, ensure_ascii=False)}")

    # ── 旁路消融：输出置零（bypass=True → read 原样返回 log_softmax(cls_logits)）──
    bypass_metrics = None
    if mem is not None and hasattr(mem, "bypass"):
        decoder.eval()
        mem.bypass = True
        bypass_metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=mem)
        mem.bypass = False
        print(f"  旁路消融：repeat={bypass_metrics['repeat_mention_acc']:.4f} "
              f"first={bypass_metrics['first_mention_acc']:.4f}")

    # ── 按提及类别拆开（evaluate_task 会切回 .train()，必须手动切回 eval）──
    doc_encoder.eval()
    decoder.eval()
    stats, hit = Counter(), Counter()
    bs = args.batch_size
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            memf = doc_encoder(inp, attention_mask=mask)
            B = inp.shape[0]
            if mem is not None:
                mem.reset(B, device)
            preds = _rollout(decoder, memf, mask, B, spec, ndb=mem, input_ids=inp)
            for b in range(B):
                truth = _truth_of(batch, b, spec)
                if not truth:
                    continue
                src = sorted(val[bi * bs + b]["spans"], key=lambda x: x["start"])
                assert len(src) == len(truth), f"样本 {bi*bs+b} 真值对齐失败"
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

    grad_summary = tracker.summary()
    grad_per_param = grad_of(mem)
    print(f"  grad：{json.dumps(grad_summary, ensure_ascii=False)}")
    if mem is not None:
        print(f"  mem 统计：{json.dumps(mem.stats(), ensure_ascii=False)}")

    rec = {"arm": args.arm, "seed": args.seed, "tag": args.tag or f"{args.arm}_seed{args.seed}",
           "freeze_mem": bool(args.freeze_mem), "rng_aligned": rng_aligned,
           "lr_mem": args.lr_mem,
           "metrics": metrics, "bypass_metrics": bypass_metrics, "by_kind": by_kind,
           "n_head_params": n_head, "n_mem_params": n_mem,
           "grad_summary": grad_summary, "grad_per_param": grad_per_param,
           "train_sec": train_sec, "n_steps": n_steps, "sec_per_step": train_sec / max(1, n_steps),
           "peak_mem_mb": peak_mb,
           "mem_stats": (mem.stats() if mem is not None else {})}
    print("AB_METRICS " + json.dumps(rec, ensure_ascii=False))
    return rec


if __name__ == "__main__":
    main()
