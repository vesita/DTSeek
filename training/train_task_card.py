"""在**冻结的基座**上单独训练一张任务卡。

插件化的核心入口：加/换一个任务只需要跑这个脚本（分钟级），不用重训基座。
基座任务无关、训一次就不再变；每张卡是独立小产物，运行时 attach 进来即可。

    uv run python training/train_task_card.py --card pronoun \
        --base checkpoints/base_encoder.pt --epochs 12 --steps-per-epoch 150

缩容消融（看这张卡到底需要多大）：
    ... --card pronoun --num-layers 1 --dim-feedforward 256 --out /tmp/pronoun_small.pt
"""
import argparse
import json
import random
import sys
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))

    it = iter(train_loader)
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
            loss = task_loss(decoder, mem, mask, batch, spec, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        if epoch % 4 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss {running/args.steps_per_epoch:.3f}")

    decoder.eval()
    metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec)
    show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print(f"  验证：{json.dumps(show, ensure_ascii=False)}")

    save_card(out, decoder, task=args.card, spec=spec, hidden_dim=hidden_dim,
              decoder_kwargs={"num_heads": 4, "num_layers": args.num_layers,
                              "dim_feedforward": args.dim_feedforward},
              base_format=base_ck["format"],
              train_args={"epochs": args.epochs, "lr_head": args.lr_head, "samples": samples,
                          "base": str(args.base), "metrics": metrics})
    print(f"  任务卡已存 {out}（{out.stat().st_size/1024:.0f} KB，参数 {n_head:,}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
