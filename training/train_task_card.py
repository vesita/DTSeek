"""在**冻结的基座**上单独训练一张任务卡。

插件化的核心入口：加/换一个任务只需要跑这个脚本（分钟级），不用重训基座。
基座任务无关、训一次就不再变；每张卡是独立小产物，运行时 attach 进来即可。

    uv run python training/train_task_card.py --card pronoun \
        --base checkpoints/base_encoder.pt --epochs 12 --steps-per-epoch 150

缩容消融（看这张卡到底需要多大）：
    ... --card pronoun --num-layers 1 --dim-feedforward 256 --out /tmp/pronoun_small.pt

提及检索记忆（Mention-NDB，`dtseek.decoder.mention_ndb`）——**默认关闭**：
    ... --card person --epochs 8 --steps-per-epoch 150 --ndb --out /tmp/person_ndb.pt

不加 `--ndb` 时，训练/验证路径与加这个开关之前**逐位一致**（`runtime.task_loss` /
`evaluate_task` 的 `ndb` 参数默认 None，只是多一次 `is None` 判断）。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.tasks.artifacts import (  # noqa: E402
    load_base_encoder, save_card,
)
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402


def _build_ndb(args, hidden_dim: int, num_classes: int):
    """按参数建 Mention-NDB；返回 (ndb, ndb_kwargs)。"""
    from dtseek.decoder.mention_ndb import MentionNDB  # noqa: E402

    levels = tuple(int(x) for x in str(args.ndb_levels).split(",") if x.strip())
    slots = [int(x) for x in str(args.ndb_slots).split(",") if x.strip()]
    kwargs = {
        "hidden_dim": hidden_dim, "num_classes": num_classes,
        "vocab_size": args.ndb_vocab, "levels": levels, "slots": slots,
        "max_table_gb": args.ndb_max_table_gb,
        "read_true": args.ndb_read == "true",
    }
    return MentionNDB(**kwargs), kwargs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="在冻结基座上单独训练一张任务卡")
    ap.add_argument("--card", required=True, help="任务卡名（取自注册表）")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--out", default=None, help="默认 checkpoints/cards/<card>.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=None, help="默认用该卡的 DEFAULT_TASK_SAMPLES")
    ap.add_argument("--num-layers", type=int, default=2, help="缩容旋钮")
    ap.add_argument("--dim-feedforward", type=int, default=None, help="默认 hidden_dim*4")
    ap.add_argument("--seed", type=int, default=42)
    # ---- Mention-NDB（默认全关）----
    ap.add_argument("--ndb", action="store_true",
                    help="给身份槽任务接提及检索记忆（默认关闭）")
    ap.add_argument("--ndb-levels", default="1,2", help="n-gram 阶数，逗号分隔")
    ap.add_argument("--ndb-slots", default="8192,4096", help="每级槽位数，逗号分隔")
    ap.add_argument("--ndb-vocab", type=int, default=8192, help="键的取值域（字表大小）")
    ap.add_argument("--ndb-lr", type=float, default=3e-4, help="读写门控学习率（独立 AdamW）")
    ap.add_argument("--ndb-max-table-gb", type=float, default=0.25, help="表显存硬上限")
    ap.add_argument("--ndb-read", choices=("true", "pred"), default="true",
                    help="训练时的读注意力：true=教师强制真值起点，pred=用指针自己的预测")
    args = ap.parse_args(argv)

    card = resolve_tasks([args.card])[args.card]
    spec = card.spec
    out = Path(args.out) if args.out else ROOT / "checkpoints" / "cards" / f"{args.card}.pt"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NanoCharTokenizer()

    doc_encoder, base_ck = load_base_encoder(args.base, device)   # 已冻结、已 eval
    hidden_dim = base_ck["hidden_dim"]
    print(f"卡片 {args.card}（{spec.label}）| 基座 {args.base}（{hidden_dim} 维，已冻结）| 设备 {device}")
    print(f"  头规格：layers={args.num_layers}, ff={args.dim_feedforward or hidden_dim * 4}")

    decoder = RobustARSliceDecoder(
        hidden_dim=hidden_dim, num_classes=spec.num_classes, num_heads=4,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
    ).to(device)
    n_head = sum(p.numel() for p in decoder.parameters())
    print(f"  解码器参数 {n_head:,}")

    ndb = None
    ndb_kwargs: dict = {}
    n_ndb = 0
    if args.ndb:
        if not spec.identity_labels:
            print(f"  ⚠ 任务 {args.card} 不是身份槽任务，NDB 的检索目标没有意义，已拒绝启用")
            return 2
        ndb, ndb_kwargs = _build_ndb(args, hidden_dim, spec.num_classes)
        ndb = ndb.to(device)
        n_ndb = sum(p.numel() for p in ndb.parameters())
        # 用一张最大 batch 的空表试算表显存：超限在这里就炸，不跑到一半 OOM
        ndb.reset(args.batch_size, device)
        print(f"  NDB 开启：levels={ndb.levels} slots={ndb.slots} read={args.ndb_read} | "
              f"门控参数 {n_ndb:,} | 表(batch={args.batch_size}) {ndb.table_gb():.3f}GB "
              f"（上限 {args.ndb_max_table_gb}GB）")

    samples = args.samples or {"pronoun": 6000, "sentiment": 32000, "relation": 14000,
                               "person": 9000, "idiom": 14000, "ownership": 8000}.get(args.card, 8000)
    data = card.build_dataset(samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr_head, weight_decay=1e-4)
    # NDB 门控用**独立**优化器（与 nanoSeek 一致：门控 lr 固定，不随头部的余弦退火衰减）
    ndb_opt = (torch.optim.AdamW(ndb.parameters(), lr=args.ndb_lr, weight_decay=0.0)
               if ndb is not None else None)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))

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
    show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print(f"  验证：{json.dumps(show, ensure_ascii=False)}")

    ndb_state = ndb.stats() if ndb is not None else {}
    if ndb is not None:
        print(f"  NDB 统计：{json.dumps(ndb_state, ensure_ascii=False)}")

    extra = ({"ndb": {"kwargs": ndb_kwargs,
                      "state_dict": {k: v.cpu() for k, v in ndb.state_dict().items()},
                      "stats": ndb_state}} if ndb is not None else {})
    save_card(out, decoder, task=args.card, spec=spec, hidden_dim=hidden_dim,
              decoder_kwargs={"num_heads": 4, "num_layers": args.num_layers,
                              "dim_feedforward": args.dim_feedforward},
              base_format=base_ck["format"],
              train_args={"epochs": args.epochs, "lr_head": args.lr_head, "samples": samples,
                          "base": str(args.base), "metrics": metrics},
              extra=extra)
    print(f"  任务卡已存 {out}（{out.stat().st_size/1024:.0f} KB，参数 {n_head:,}）")

    # 机器可读汇总：AB 脚本按行抓这行，不解析人类可读输出
    print("AB_METRICS " + json.dumps({
        "card": args.card, "seed": args.seed, "ndb": bool(args.ndb),
        "ndb_read": args.ndb_read if args.ndb else None,
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "num_layers": args.num_layers, "batch_size": args.batch_size,
        "metrics": metrics, "n_head_params": n_head, "n_ndb_params": n_ndb,
        "train_sec": train_sec, "n_steps": n_steps,
        "sec_per_step": train_sec / max(1, n_steps),
        "peak_mem_mb": peak_mb, "ndb_stats": ndb_state,
        # 表按**训练 batch** 计（ndb.stats() 里的 table_gb 是最后一个 batch 的，会偏小）
        "ndb_table_gb": ndb.table_gb(args.batch_size) if ndb is not None else 0.0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

