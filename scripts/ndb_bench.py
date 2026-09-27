#!/usr/bin/env python
"""Mention-NDB 的**同构**逐步计时探针（优化前 / 优化后对照用）。

为什么要单独写一个：`ab_test_ndb.py` 报的 `sec_per_step` 是**端到端**口径，
它把基座、解码器、NDB、优化器全混在一起，只给一个总数 —— 无法回答
「NDB 自己占多少 / 其中前向多少、反向多少 / 优化后还剩多少」。
本脚本用**与 `training/train_task_card.py` 逐行同构**的训练循环（同样的 seed、
同样的数据集与 DataLoader、同样的 `task_loss`、同样的 backward / clip / step / scheduler
顺序），量四件事：

    step : base 臂 / ndb 臂的整体步时（只在测区两端同步，取稳态的末 N 次）
    iso  : NDB 自身前向 —— reset + 16×(read_attention + read + write)，
           张量形状与调用顺序都与真实循环一致，只是不接解码器
    bwd  : 「读 + 凸混合 + cross_entropy」相对「纯 CE」的 forward / backward 增量
    check: 两个实现（--impl 与 --impl-ref）在同一条输入上的输出是否**逐位相同**

对照用法（把优化前的文件先留一份）：

    cp src/dtseek/decoder/mention_ndb.py /tmp/ndb_before.py
    uv run python scripts/ndb_bench.py --mode all --impl /tmp/ndb_before.py --tag before
    uv run python scripts/ndb_bench.py --mode all --tag after      # 优化后（包内实现）

判据纪律：`step` 模式取**稳态末 N 次**均值（前几次含 CUDA 预热），
base 与 ndb 用同一 seed 构造，因而消费同一条 batch 序列，Δ 才可比。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, task_loss  # noqa: E402

SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 14000,
           "person": 9000, "idiom": 14000, "ownership": 8000}


def gpu_state() -> dict:
    """当前卡上的 compute 进程 —— 计时必须在**没有别人占卡**时做。

    这台机器上常有并行的 DSH 会话在跑别的 NDB 实验，实测同一条 baseline 的
    `full` 口径能在 16ms 与 181ms 之间跳 —— 所以每次计时都把当时的占用记进产物，
    并在有「像训练进程的别人」时拒绝出数（`--allow-busy` 可强行覆盖，但产物会标注）。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout
    except Exception as e:                                   # pragma: no cover - 环境相关
        return {"available": False, "error": str(e), "apps": []}
    me = {os.getpid(), os.getppid()}
    apps, foreign = [], []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, name, mem = int(parts[0]), parts[1], parts[2]
        apps.append({"pid": pid, "name": name, "mem": mem})
        if pid not in me and "python" in name.lower():
            foreign.append({"pid": pid, "name": name, "mem": mem})
    return {"available": True, "apps": apps, "foreign_ml": foreign}


def load_ndb_class(impl: str | None):
    """`impl=None` → 包内 `dtseek.decoder.mention_ndb`；否则从给定 .py 文件加载。"""
    if impl is None:
        from dtseek.decoder.mention_ndb import MentionNDB  # noqa: E402
        return MentionNDB
    spec = importlib.util.spec_from_file_location("ndb_impl_external", impl)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法从 {impl} 加载 MentionNDB")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MentionNDB


def ndb_kwargs(args, hidden_dim: int, num_classes: int) -> dict:
    return dict(hidden_dim=hidden_dim, num_classes=num_classes, vocab_size=args.ndb_vocab,
                levels=tuple(int(x) for x in args.ndb_levels.split(",") if x.strip()),
                slots=[int(x) for x in args.ndb_slots.split(",") if x.strip()],
                max_table_gb=args.ndb_max_table_gb, read_true=args.ndb_read == "true")


def build_setup(args, ndb_cls):
    """复现 train_task_card.main 的初始化顺序（seed / 数据 / 模型 / 优化器）。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    card = resolve_tasks([args.card])[args.card]
    spec = card.spec
    tokenizer = NanoCharTokenizer()
    doc_encoder, base_ck = load_base_encoder(args.base, device)
    hidden_dim = base_ck["hidden_dim"]
    decoder = RobustARSliceDecoder(
        hidden_dim=hidden_dim, num_classes=spec.num_classes, num_heads=4,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
    ).to(device)

    samples = args.samples or SAMPLES.get(args.card, 8000)
    data = card.build_dataset(samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    train_loader = DataLoader(GenericTaskDataset(data[n_val:], tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    ndb = None
    if ndb_cls is not None:
        ndb = ndb_cls(**ndb_kwargs(args, hidden_dim, spec.num_classes)).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr_head, weight_decay=1e-4)
    ndb_opt = (torch.optim.AdamW(ndb.parameters(), lr=args.ndb_lr, weight_decay=0.0)
               if ndb is not None else None)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200)
    return dict(device=device, spec=spec, doc_encoder=doc_encoder, decoder=decoder,
                ndb=ndb, optimizer=optimizer, ndb_opt=ndb_opt, scheduler=scheduler,
                train_loader=train_loader, hidden_dim=hidden_dim)


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _summarize(name: str, per_iter: list[float], steps_per_iter: int = 1) -> dict:
    """报 均值 / 中位 / **最小** —— 别人占卡只会让某次变慢，取最小是真实下界。"""
    srt = sorted(per_iter)
    p10 = srt[max(0, len(srt) // 10)]
    out = {"mean_ms": statistics.fmean(per_iter) * 1e3, "median_ms": statistics.median(per_iter) * 1e3,
           "min_ms": min(per_iter) * 1e3, "p10_ms": p10 * 1e3, "n": len(per_iter)}
    triple = "" if steps_per_iter == 1 else f"  (最小 {out['min_ms'] / steps_per_iter:.3f} ms/三元组)"
    print(f"[{name}] 均值={out['mean_ms']:.2f}  中位={out['median_ms']:.2f}  "
          f"**最小={out['min_ms']:.2f}**  p10={out['p10_ms']:.2f} ms/次"
          f"{triple}  n={len(per_iter)}")
    return out


def make_step_fn(args, ndb_cls):
    """建一个 setup + 与 train_task_card 逐行同构的 `one_step()`，返回 (setup, one_step)。"""
    st = build_setup(args, ndb_cls)
    device = st["device"]
    doc_encoder, decoder = st["doc_encoder"], st["decoder"]
    ndb, ndb_opt, optimizer, scheduler = st["ndb"], st["ndb_opt"], st["optimizer"], st["scheduler"]
    spec = st["spec"]
    state = {"it": iter(st["train_loader"])}
    decoder.train()

    def one_step():
        try:
            batch = next(state["it"])
        except StopIteration:
            state["it"] = iter(st["train_loader"])
            batch = next(state["it"])
        batch = _move(batch, device)
        inp, mask = batch["input_ids"], batch["attention_mask"]
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

    return st, one_step


def run_step_arm(args, ndb_cls, label: str) -> dict:
    """整体训练步（wall clock），与 train_task_card 的循环逐行同构。"""
    st, one_step = make_step_fn(args, ndb_cls)
    device = st["device"]
    for _ in range(args.warmup):
        one_step()
    _sync(device)
    per_iter = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_step()
        per_iter.append(time.perf_counter() - t0)
    _sync(device)
    return _summarize(f"step:{label}", per_iter[-args.steady:])


def run_interleaved(args, label: str) -> dict:
    """**配对交替**口径：base 与 ndb 各走一步、交替 args.iters 轮。

    为什么需要它：这台机器常有别的会话占卡，别人只会让某一次变慢。两个臂在
    **同一轮里紧挨着跑**，于是 (a) 每轮的配对差 `ndb−base` 消掉当时的干扰，
    (b) 各自取**最小**给出无干扰下界。单臂顺序跑（run_step_arm）在占卡时会把
    基座臂和 ndb 臂落在不同的负载窗口里，Δ 就没法比。
    """
    st_base, base_step = make_step_fn(args, None)
    st_ndb, ndb_step = make_step_fn(args, load_ndb_class(args.impl))
    device = st_base["device"]
    for _ in range(args.warmup):
        base_step()
        ndb_step()
    _sync(device)
    pairs_base, pairs_ndb = [], []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        base_step()
        t1 = time.perf_counter()
        ndb_step()
        t2 = time.perf_counter()
        pairs_base.append(t1 - t0)
        pairs_ndb.append(t2 - t1)
    _sync(device)
    # 配对差取**最小**与**中位**：前者是无干扰下界，后者对个别尖峰不敏感
    diffs = sorted(b - a for a, b in zip(pairs_base, pairs_ndb))
    out = {
        "base": _summarize(f"interleave:{label}/base", pairs_base),
        "ndb": _summarize(f"interleave:{label}/ndb", pairs_ndb),
        "diff_min_ms": diffs[0] * 1e3,
        "diff_median_ms": statistics.median(diffs) * 1e3,
        "diff_mean_ms": statistics.fmean(diffs) * 1e3,
    }
    bmin = out["base"]["min_ms"]
    out["diff_min_pct"] = out["diff_min_ms"] / bmin * 100.0
    out["diff_median_pct"] = out["diff_median_ms"] / bmin * 100.0
    # 注：`diff_min` 是「配对差的最小值」，噪声下可以被一个恰好很快的 ndb 轮次拉到
    # 很负，**不是**可用估计；只留中位/均值（都在同一轮里配对，抵消当时的负载）。
    print(f"[interleave:{label}] 配对 Δ(ndb−base)：中位={out['diff_median_ms']:+.2f}ms "
          f"({out['diff_median_pct']:.1f}%)  均值={out['diff_mean_ms']:+.2f}ms "
          f"  [min-of-diff={out['diff_min_ms']:+.2f}ms 仅留痕]")
    return out


def run_ablate(args, label: str) -> dict:
    """把 NDB 的步时增量拆成「读」「写」「谁都不开」三块，**在真实 task_loss 循环里**。

    做法是把 `ndb.read` 换成恒等、把 `ndb.write` 换成空操作（不碰 runtime.py）：
      none       = 读写都关（≈ 不接 NDB 的等价物，但模块/优化器还在）
      read_only  = 只读
      write_only = 只写
      full       = 读写都开
    于是 读的代价 = full − write_only，写的代价 = full − read_only。
    """
    out: dict = {}
    for name, do_read, do_write in (("none", False, False), ("read_only", True, False),
                                    ("write_only", False, True), ("full", True, True)):
        st, one_step = make_step_fn(args, load_ndb_class(args.impl))
        ndb, device = st["ndb"], st["device"]
        if not do_read:
            ndb.read = lambda cls_logits, *a, **k: cls_logits      # noqa: ARG005
        if not do_write:
            ndb.write = lambda *a, **k: None                       # noqa: ARG005
        for _ in range(args.warmup):
            one_step()
        _sync(device)
        per = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            one_step()
            per.append(time.perf_counter() - t0)
        _sync(device)
        out[name] = _summarize(f"ablate:{label}/{name}", per[-args.steady:])
    base = out["none"]["min_ms"]
    for k, v in out.items():
        v["delta_vs_none_ms"] = v["min_ms"] - base
        v["delta_vs_none_pct"] = v["delta_vs_none_ms"] / base * 100.0
    out["read_cost_ms"] = out["full"]["min_ms"] - out["write_only"]["min_ms"]
    out["write_cost_ms"] = out["full"]["min_ms"] - out["read_only"]["min_ms"]
    print(f"[ablate:{label}] 相对 none：read_only={out['read_only']['delta_vs_none_ms']:+.2f}ms "
          f"write_only={out['write_only']['delta_vs_none_ms']:+.2f}ms "
          f"full={out['full']['delta_vs_none_ms']:+.2f}ms ({out['full']['delta_vs_none_pct']:.1f}%)")
    print(f"[ablate:{label}] 归因：读={out['read_cost_ms']:+.2f}ms  写={out['write_cost_ms']:+.2f}ms")
    return out


@torch.no_grad()
def run_iso(args, ndb_cls, label: str) -> dict:
    """NDB 自身前向：reset + 16×(read_attention + read + write)，形状与真实循环一致。"""
    st = build_setup(args, ndb_cls)
    device = st["device"]
    ndb, spec = st["ndb"], st["spec"]
    if ndb is None:
        raise SystemExit("iso 模式需要 NDB")
    batch = _move(next(iter(st["train_loader"])), device)
    inp, mask = batch["input_ids"], batch["attention_mask"]
    B, L = inp.shape
    h = torch.randn(B, st["hidden_dim"], device=device)
    cls = torch.randn(B, spec.num_classes, device=device)
    start = torch.randn(B, L, device=device)

    def one_step():
        ndb.reset(B, device)
        for s in range(spec.max_steps):
            attn = ndb.read_attention(start, mask, batch["starts"][:, s])
            ndb.read(cls, h, inp, attn)
            with ndb.write_enabled():
                ndb.write(h, inp, batch["starts"][:, s], batch["labels"][:, s],
                          batch["step_mask"][:, s])

    for _ in range(args.warmup):
        one_step()
    _sync(device)
    per_iter = [(_time(lambda: one_step())) for _ in range(args.iters)]
    _sync(device)
    return _summarize(f"iso:{label}", per_iter[-args.steady:], steps_per_iter=spec.max_steps)


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_bwd(args, ndb_cls, label: str) -> tuple[float, float, float, float]:
    """「读 + 凸混合 + CE」相对「纯 CE」的 forward / backward 增量。

    两个变体在**同一批真实张量**上跑（同一 h / cls_logits / attn），
    用 retain_graph 让同一张解码器图可被重复反向；差值是配对的。
    """
    st = build_setup(args, ndb_cls)
    device = st["device"]
    ndb, spec, decoder = st["ndb"], st["spec"], st["decoder"]
    st["doc_encoder"].eval()
    decoder.eval()
    batch = _move(next(iter(st["train_loader"])), device)
    inp, mask = batch["input_ids"], batch["attention_mask"]
    B = inp.shape[0]
    with torch.no_grad():
        mem = st["doc_encoder"](inp, attention_mask=mask)
        q = decoder.bos_query.expand(B, 1, -1)
        out = decoder.forward_step(q, mem, doc_mask=mask)
        attn = ndb.read_attention(out["start_logits"], mask, batch["starts"][:, 0])
        h = out["last_hidden"].squeeze(1)
        t_labels = batch["labels"][:, 0]
    cls_logits = out["cls_logits"].detach().requires_grad_(True)
    h = h.detach().requires_grad_(True)

    # 先种一张非空表：空表会把门控归零，混合项变成空操作（测出来就是假的）
    with torch.no_grad():
        ndb.reset(B, device)
        with ndb.write_enabled():
            for s in range(spec.max_steps):
                ndb.write(h, inp, batch["starts"][:, s], batch["labels"][:, s],
                          batch["step_mask"][:, s])

    def measure(mixed: bool):
        fwd, bwd = [], []
        for _ in range(args.iters):
            _sync(device)
            t0 = time.perf_counter()
            logits = ndb.read(cls_logits, h, inp, attn) if mixed else cls_logits
            loss = F.cross_entropy(logits, t_labels)
            _sync(device)
            t1 = time.perf_counter()
            loss.backward(retain_graph=True)
            _sync(device)
            t2 = time.perf_counter()
            cls_logits.grad = None
            h.grad = None
            ndb.zero_grad(set_to_none=True)
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
        return (statistics.fmean(fwd[-args.steady:]) * 1e3,
                statistics.fmean(bwd[-args.steady:]) * 1e3)

    for _ in range(args.warmup):
        measure(True)
    bf, bb = measure(False)
    nf, nb = measure(True)
    print(f"[bwd:{label}] 纯 CE           : fwd={bf:.2f}ms  bwd={bb:.2f}ms")
    print(f"[bwd:{label}] read+mix+CE     : fwd={nf:.2f}ms  bwd={nb:.2f}ms")
    print(f"[bwd:{label}] Δ(混叠项)       : fwd={nf - bf:+.2f}ms  bwd={nb - bb:+.2f}ms")
    return nf - bf, nb - bb, nf, nb


def run_check(args, label: str) -> bool:
    """两个实现（--impl 与 --impl-ref）在同一条输入序列上的输出逐位比较。"""
    if not args.impl:
        raise SystemExit("check 模式需要 --impl（对照实现）")
    cls_a = load_ndb_class(args.impl)
    cls_b = load_ndb_class(args.impl_ref)
    st = build_setup(args, cls_a)
    b_ndb = cls_b(**ndb_kwargs(args, st["hidden_dim"], st["spec"].num_classes)).to(st["device"])
    b_ndb.load_state_dict(st["ndb"].state_dict())     # 同一份门控权重，逐位比较才有意义
    device = st["device"]
    batch = _move(next(iter(st["train_loader"])), device)
    inp, mask = batch["input_ids"], batch["attention_mask"]
    B, L = inp.shape
    torch.manual_seed(1234)
    h = torch.randn(B, st["hidden_dim"], device=device)
    cls = torch.randn(B, st["spec"].num_classes, device=device)
    start = torch.randn(B, L, device=device)

    a, b = st["ndb"], b_ndb
    ok = True
    for tag, true_starts in (("read_true", batch["starts"][:, 0]), ("read_pred", None)):
        a.reset(B, device)
        b.reset(B, device)
        aa, bb = a.read_attention(start, mask, true_starts), b.read_attention(start, mask, true_starts)
        if not torch.equal(aa, bb):
            print(f"[check] read_attention({tag}) 不逐位相同  max|Δ|={(aa - bb).abs().max():.3e}")
            ok = False
        la, lb = a.read(cls, h, inp, aa), b.read(cls, h, inp, bb)
        if not torch.equal(la, lb):
            print(f"[check] read({tag}) 不逐位相同  max|Δ|={(la - lb).abs().max():.3e}")
            ok = False
    a.reset(B, device)
    b.reset(B, device)
    with a.write_enabled(), b.write_enabled():
        for s in range(st["spec"].max_steps):
            a.write(h, inp, batch["starts"][:, s], batch["labels"][:, s], batch["step_mask"][:, s])
            b.write(h, inp, batch["starts"][:, s], batch["labels"][:, s], batch["step_mask"][:, s])
    for li in range(len(a.slots)):
        if not torch.equal(a._counts[li], b._counts[li]):
            print(f"[check] _counts[{li}] 不逐位相同  max|Δ|={(a._counts[li] - b._counts[li]).abs().max():.3e}")
            ok = False
        if not torch.equal(a._totals[li], b._totals[li]):
            print(f"[check] _totals[{li}] 不逐位相同")
            ok = False
    print(f"[check:{label}] {'逐位相同 ✓' if ok else '存在差异 ✗'}"
          f"（impl={args.impl} vs ref={args.impl_ref or 'package'}）")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mention-NDB 同构逐步计时探针")
    ap.add_argument("--mode", choices=("step", "interleave", "ablate", "iso", "bwd", "check", "all"), default="all")
    ap.add_argument("--impl", default=None, help="NDB 实现文件（默认用包内 dtseek.decoder.mention_ndb）")
    ap.add_argument("--impl-ref", default=None, help="check 模式的参照实现（默认用包内实现）")
    ap.add_argument("--tag", default="", help="输出标签（before / after 等）")
    ap.add_argument("--card", default="person")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=None)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--ndb-lr", type=float, default=3e-4)
    ap.add_argument("--ndb-vocab", type=int, default=8192)
    ap.add_argument("--ndb-levels", default="1,2")
    ap.add_argument("--ndb-slots", default="8192,4096")
    ap.add_argument("--ndb-max-table-gb", type=float, default=0.25)
    ap.add_argument("--ndb-read", choices=("true", "pred"), default="true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--steady", type=int, default=20, help="取末 N 次的均值")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--allow-busy", action="store_true",
                    help="卡上有别的 python compute 进程时也照跑（产物会标注，数字不可信）")
    args = ap.parse_args(argv)
    label = args.tag or ("pkg" if args.impl is None else Path(args.impl).stem)

    gpu = gpu_state()
    print(f"[gpu] {json.dumps(gpu, ensure_ascii=False)}")
    if gpu.get("foreign_ml") and args.device is None and torch.cuda.is_available() and not args.allow_busy:
        raise SystemExit(
            "[gpu] 卡上有别的 python compute 进程，计时口径不可信："
            f"{gpu['foreign_ml']}。等它结束再跑，或用 --allow-busy 强行覆盖。")

    result: dict = {"tag": label, "impl": args.impl, "seed": args.seed,
                    "batch_size": args.batch_size, "card": args.card, "gpu": gpu}
    if args.mode in ("interleave", "all"):
        result["interleave"] = run_interleaved(args, label)
    if args.mode in ("step", "all"):
        base = run_step_arm(args, None, label + "/base")
        ndb = run_step_arm(args, load_ndb_class(args.impl), label + "/ndb")
        result["step_base"], result["step_ndb"] = base, ndb
        result["step_base_ms"], result["step_ndb_ms"] = base["mean_ms"], ndb["mean_ms"]
        d = ndb["mean_ms"] - base["mean_ms"]
        dmin = ndb["min_ms"] - base["min_ms"]
        result.update(step_delta_ms=d, step_delta_pct=d / base["mean_ms"] * 100.0,
                      step_delta_min_ms=dmin, step_delta_min_pct=dmin / base["min_ms"] * 100.0)
        print(f"[step:{label}] NDB 增量（均值口径）= {d:.2f} ms/步 = {result['step_delta_pct']:.1f}%")
        print(f"[step:{label}] NDB 增量（最小口径）= {dmin:.2f} ms/步 = {result['step_delta_min_pct']:.1f}%")
    if args.mode in ("ablate", "all"):
        result["ablate"] = run_ablate(args, label)
    if args.mode in ("iso", "all"):
        result["ndb_iso"] = run_iso(args, load_ndb_class(args.impl), label)
        result["ndb_iso_ms"] = result["ndb_iso"]["mean_ms"]
    if args.mode in ("bwd", "all"):
        df, db, nf, nb = run_bwd(args, load_ndb_class(args.impl), label)
        result.update(mix_fwd_delta_ms=df, mix_bwd_delta_ms=db,
                      mix_fwd_ms=nf, mix_bwd_ms=nb)
    if args.mode in ("check", "all") and args.impl:
        result["check_bitwise_equal"] = run_check(args, label)
    print("NDB_BENCH " + json.dumps(result, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
