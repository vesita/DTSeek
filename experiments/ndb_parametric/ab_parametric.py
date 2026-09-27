#!/usr/bin/env python
"""参数化 / 可微记忆 vs 非参数 NDB 的受控 AB（person 卡）。

**逐行复刻 `training/train_task_card.py` 与 `experiments/ndb_semantic/ab_semantic.py`
的初始化与训练顺序**（含全局 RNG 消耗顺序），唯一差别是记忆结构。自带测量台校验：

    --arm base    --seed 42  → repeat_mention_acc 必须 == 0.3772683858643744
    --arm literal --seed 42  → repeat_mention_acc 必须 == 0.8407517309594461

不同就说明 harness 不等价，必须先修 harness。

臂：
  base        无记忆（复用既有日志，本脚本只作校验）
  literal     现有 MentionNDB（键 = 字面 n-gram），复用既有日志
  slots16/32  静态可学习矩阵 M ∈ R^{S×C}
  diffwrite   可微情节写（每文档 M，非参数）
  fastweight  外积记忆（键值都是学习投影）
  fastweight_cls 外积记忆（值 = onehot(cls)，参数化臂的最强候选）

用法：
  uv run python experiments/ndb_parametric/ab_parametric.py --arm base --seed 42
  uv run python experiments/ndb_parametric/ab_parametric.py --arm slots32 --seed 42 \
      --json-out experiments/ndb_parametric/results/slots32_seed42.json
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
sys.path.insert(0, str(ROOT / "experiments" / "ndb_parametric"))
OUT = Path(__file__).resolve().parent

ARMS = ("base", "literal", "slots16", "slots32", "diffwrite",
        "fastweight", "fastweight_cls")
MEM_ARMS = ("slots16", "slots32", "diffwrite", "fastweight", "fastweight_cls")


def kind_of(spans):
    """与 `experiments/ndb_semantic/ab_semantic.py` 完全相同的拆分口径。"""
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


def build_mem(arm, hidden_dim, num_classes, seed, device):
    if arm == "literal":
        from dtseek.decoder.mention_ndb import MentionNDB
        m = MentionNDB(hidden_dim=hidden_dim, num_classes=num_classes, vocab_size=8192,
                       levels=(1, 2), slots=[8192, 4096], max_table_gb=0.25, read_true=True)
        return m.to(device)
    import memories as mem
    if arm == "slots16":
        m = mem.SlotsMemory(hidden_dim, num_classes, n_slots=16, seed=seed)
    elif arm == "slots32":
        m = mem.SlotsMemory(hidden_dim, num_classes, n_slots=32, seed=seed)
    elif arm == "diffwrite":
        m = mem.DiffWriteMemory(hidden_dim, num_classes, seed=seed)
    elif arm == "fastweight":
        m = mem.FastWeightMemory(hidden_dim, num_classes, seed=seed, key_dim=32,
                                 value_from_cls=False)
    elif arm == "fastweight_cls":
        m = mem.FastWeightMemory(hidden_dim, num_classes, seed=seed, key_dim=32,
                                 value_from_cls=True)
    else:
        raise SystemExit(f"未知臂 {arm}")
    return m.to(device)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=9000)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ndb-lr", type=float, default=3e-4)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--log", default=None, help="把 stdout 也写入这个文件")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
    from dtseek.tasks.artifacts import load_base_encoder
    from dtseek.tasks.plugin import resolve_tasks
    from dtseek.tasks.runtime import (GenericTaskDataset, evaluate_task, task_loss,
                                      _rollout, _truth_of)

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
    if args.arm in MEM_ARMS or args.arm == "literal":
        mem = build_mem(args.arm, hidden_dim, spec.num_classes, args.seed, device)
        n_mem = sum(p.numel() for p in mem.parameters())
        mem.reset(args.batch_size, device)
        print(f"  记忆臂 {args.arm}：{mem.extra_repr()} | 参数 {n_mem:,}")

    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr_head, weight_decay=1e-4)
    mem_opt = (torch.optim.AdamW(mem.parameters(), lr=args.ndb_lr, weight_decay=0.0)
               if mem is not None and n_mem > 0 else None)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))

    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    it = iter(train_loader)
    n_steps = 0
    # 数据顺序指纹：证明与 base 臂共享同一条全局 RNG 轨迹（同样的洗牌）
    first_batch_fp = None
    # 记忆梯度诊断：设备上累加，只在 epoch 边界同步一次（避免每步 .item() 污染计时）
    grad_ss = torch.zeros((), device=device)
    grad_nz = torch.zeros((), device=device)      # 有非零梯度的步数
    grad_none = 0
    per_param_last: dict[str, float] = {}
    n_grad_steps = 0
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            if first_batch_fp is None:
                first_batch_fp = float(batch["input_ids"].to(torch.float64).sum().item())
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                dm = doc_encoder(inp, attention_mask=mask)
            loss = task_loss(decoder, dm, mask, batch, spec, device, ndb=mem)
            optimizer.zero_grad()
            if mem_opt is not None:
                mem_opt.zero_grad(set_to_none=True)
            loss.backward()
            if mem is not None and n_mem > 0:
                step_has = torch.zeros((), device=device)
                for p in mem.parameters():
                    if p.grad is None:
                        grad_none += 1
                        continue
                    g = p.grad.detach()
                    grad_ss = grad_ss + g.pow(2).sum()
                    step_has = torch.maximum(step_has, (g.abs().sum() > 0).to(step_has.dtype))
                grad_nz = grad_nz + step_has
                n_grad_steps += 1
            # ★ 只裁剪 decoder：与 train_task_card.py / ab_semantic.py 逐字一致。
            #   给记忆参数也裁剪会改变 NDB 的更新（实测把 0.8408 变成 0.8473），
            #   那就不是在复现参考臂了。
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

    grad_info: dict = {}
    if mem is not None and n_mem > 0:
        for nname, p in mem.named_parameters():
            per_param_last[nname] = (float(p.grad.detach().norm().item())
                                     if p.grad is not None else -1.0)
        ss = float(grad_ss.item())
        grad_info = {
            "grad_norm_mean_per_step": (ss / max(1, n_grad_steps)) ** 0.5,
            "grad_norm_total_sq": ss,
            "n_grad_steps": n_grad_steps,
            "n_steps_with_nonzero_grad": int(grad_nz.item()),
            "n_param_grad_none": grad_none,
            "per_param_last_grad_norm": per_param_last,
            "all_grads_nonzero": bool(grad_nz.item() == n_grad_steps and grad_none == 0),
        }
        print(f"  梯度：mean/step={grad_info['grad_norm_mean_per_step']:.6g} "
              f"非零步={grad_info['n_steps_with_nonzero_grad']}/{n_grad_steps} "
              f"None 次数={grad_none} | per-param={json.dumps(per_param_last)}")

    decoder.eval()
    if mem is not None:
        mem.eval()
    metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=mem)
    print(f"  验证：{json.dumps({k: round(v,4) for k,v in metrics.items() if isinstance(v,float)}, ensure_ascii=False)}")

    # ── 旁路消融：同一份训练好的权重，把记忆输出置零，再评估一次（配对）──
    bypass_metrics = None
    if mem is not None and hasattr(mem, "bypass"):
        mem.bypass = True
        doc_encoder.eval(); decoder.eval()
        bypass_metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=mem)
        mem.bypass = False
        print(f"  旁路：{json.dumps({k: round(v,4) for k,v in bypass_metrics.items() if isinstance(v,float)}, ensure_ascii=False)}")

    # ── 按提及类别拆开（别名类 = 共指；字面同 id = 字面检索）──
    doc_encoder.eval(); decoder.eval()
    stats, hit = Counter(), Counter()
    bs = args.batch_size
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            dm = doc_encoder(inp, attention_mask=mask)
            B = inp.shape[0]
            if mem is not None:
                mem.reset(B, device)
            preds = _rollout(decoder, dm, mask, B, spec, ndb=mem, input_ids=inp)
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
                        continue
                    stats[k] += 1
                    if pred_at[(s0, e0)] == lab:
                        hit[k] += 1
    by_kind = {k: {"n": stats[k], "hit": hit[k], "acc": hit[k] / max(1, stats[k])}
               for k in ("literal_same_id", "literal_other_id", "alias")}
    print("  按类别：" + "  ".join(f"{k}={v['hit']}/{v['n']}({v['acc']:.4f})"
                                   for k, v in by_kind.items()))
    if mem is not None:
        print(f"  记忆统计：{json.dumps(mem.stats(), ensure_ascii=False)}")

    rec = {"arm": args.arm, "seed": args.seed, "tag": args.tag or f"{args.arm}_seed{args.seed}",
           "metrics": metrics, "by_kind": by_kind, "bypass_metrics": bypass_metrics,
           "n_head_params": n_head, "n_mem_params": n_mem,
           "train_sec": train_sec, "n_steps": n_steps,
           "sec_per_step": train_sec / max(1, n_steps),
           "peak_mem_mb": peak_mb, "grad_info": grad_info,
           "mem_stats": (mem.stats() if mem is not None else {}),
           "first_batch_fingerprint": first_batch_fp}
    print("AB_METRICS " + json.dumps(rec, ensure_ascii=False))
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


if __name__ == "__main__":
    main()
