"""多任务交替联合训练引擎 v3 —— **任务卡注册表驱动**。

训练脚本不再认识任何具体任务：任务从 `dtseek.tasks.plugin` 的注册表里来，
样本量、类别数、发射步数、损失权重全部由各任务卡的 `TaskSpec` 决定。

    加一个任务 = 在 src/dtseek/tasks/ 下加一个模块（本文件一行都不用改）

训练范式：
- 共享同一个 NanoDocEncoder 基座（RMSNorm + RoPE + QK-Norm + SwiGLU + FlashAttn），
  放开梯度，随各任务交替反向传播持续进化；
- 每个任务各持一张独立的轻量 Decoder 任务卡，彼此不共享参数。

为什么必须逐任务验证（而不是只看 loss）：
只用 loss 观察时，模型会把"喜欢"圈成"颜"、对中性疑问句误报，loss 一路下降却
完全没暴露 —— 因为 loss 只衡量"平均拟合"，不衡量"位置是否落在词上"。
所以每个任务都测可判对错的指标，见 `dtseek.tasks.runtime.evaluate_task`。
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
from dtseek.tasks.plugin import all_tasks, resolve_tasks
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss

#: 历史默认样本量：情绪词典 199 词，样本量必须与词典规模成比例，否则长尾欠训
DEFAULT_TASK_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "ownership": 8000}

ENCODER_KWARGS = {
    "num_layers": 3,
    "num_heads": 4,
    "max_len": 128,
    "rope_theta": 10000.0,
    "use_qk_norm": True,
    "swiglu_scale": 8 / 3,
}
DECODER_KWARGS = {"num_heads": 4, "num_layers": 2}
HIDDEN_DIM = 128


def train_multitask(num_epochs: int = 16, batch_size: int = 64,
                    lr_base: float = 3e-4, lr_head: float = 1e-3,
                    samples_per_task: int = 6000,
                    task_samples: dict | None = None,
                    tasks: list[str] | None = None,
                    steps_per_epoch: int | None = None,
                    freeze_base: bool = False,
                    init_from: str | None = None,
                    seed: int = 42,
                    ckpt_path: str = "checkpoints/multitask_v2_dtseek.pt",
                    metrics_path: str = "checkpoints/multitask_v2_metrics.json") -> dict:
    """多任务交替联合训练。

    Args:
        task_samples: 逐任务样本量覆盖，例如 {"sentiment": 24000}。未列出的任务用
            `DEFAULT_TASK_SAMPLES`，仍未有则用 `samples_per_task`。
        tasks: 只训这几张任务卡（名字取自注册表）；None = 全部已注册任务。
        steps_per_epoch: 每个 epoch 的步数。None = 取最短的那个任务的批数。
            注意默认值会被**最小的**任务卡拖住：加一张小数据集的任务卡，所有任务
            都跟着少训。要训够就显式给一个值（迭代器耗尽会自动重开）。
        freeze_base: 冻结共享基座，只训各任务头。基座一旦冻结，各任务头**彼此完全独立**
            （唯一的耦合通道就是基座），互相干扰随之消失；同时基座转 eval 关掉 dropout。
        init_from: 从某个 ckpt 热启动：加载 doc_encoder 与同名任务头。冻结基座时几乎总要
            配它 —— 从随机基座冻结等于让任务头去读噪声。
        seed: 全局随机种子。加了它训练才可复现 —— 否则重构前后无法比对。
    """
    cards = resolve_tasks(tasks)
    if not cards:
        raise ValueError("没有任何任务卡可训；检查 dtseek/tasks/ 是否有模块注册")

    resolved = {}
    for name in cards:
        if task_samples and name in task_samples:
            resolved[name] = task_samples[name]
        else:
            resolved[name] = DEFAULT_TASK_SAMPLES.get(name, samples_per_task)

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"多任务联合调优引擎 v3 | 设备: {device} | seed: {seed}")
    print(f"  任务卡：{'、'.join(f'{n}({cards[n].spec.label})' for n in cards)}")

    tokenizer = NanoCharTokenizer()

    # 1. 逐任务数据集（全部来自各任务卡自己的 build_dataset）
    print("[1/4] 构建任务数据集 ...")
    raw = {name: card.build_dataset(resolved[name]) for name, card in cards.items()}

    _eval_sets = {}
    for name, data in raw.items():
        random.Random(seed).shuffle(data)
        n_val = max(200, len(data) // 10)
        _eval_sets[name] = data[:n_val]
        data[:] = data[n_val:]

    loaders = {
        name: DataLoader(GenericTaskDataset(data, tokenizer, cards[name].spec),
                         batch_size=batch_size, shuffle=True, drop_last=True)
        for name, data in raw.items()
    }
    val_loaders = {
        name: DataLoader(GenericTaskDataset(_eval_sets[name], tokenizer, cards[name].spec),
                         batch_size=batch_size, shuffle=False)
        for name in raw
    }

    # 2. 共享高效基座
    print("[2/4] 构建 NanoDocEncoder 高效基座 ...")
    doc_encoder = NanoDocEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
        **ENCODER_KWARGS,
    ).to(device)

    # 3. 逐任务独立任务卡（类别数由 spec 决定，不再写死 4）
    decoders = {
        name: RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                   num_classes=cards[name].spec.num_classes,
                                   **DECODER_KWARGS).to(device)
        for name in cards
    }

    if init_from:
        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        missing, unexpected = doc_encoder.load_state_dict(ck["doc_encoder"], strict=False)
        resumed = []
        for name, dec in decoders.items():
            if name in ck.get("decoders", {}):
                dec.load_state_dict(ck["decoders"][name])
                resumed.append(name)
        print(f"  热启动自 {init_from}：基座已载入（missing={len(missing)}, unexpected={len(unexpected)}）"
              f"，续训任务头 {resumed}")

    head_params = [p for d in decoders.values() for p in d.parameters()]
    if freeze_base:
        for p in doc_encoder.parameters():
            p.requires_grad_(False)
        doc_encoder.eval()          # 冻结后关掉 dropout，前向变成确定性函数
        print(f"  ❄️ 基座已冻结（{sum(p.numel() for p in doc_encoder.parameters()):,} 参数），"
              f"只训 {sum(p.numel() for p in head_params):,} 个头参数")
        groups = [{"params": head_params, "lr": lr_head}]
    else:
        groups = [{"params": doc_encoder.parameters(), "lr": lr_base},
                  {"params": head_params, "lr": lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    # 余弦退火：基座与任务头同步衰减
    total_steps = num_epochs * min(len(x) for x in loaders.values())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # 4. 交替训练
    print("[3/4] 开始多任务交替联合训练 ...")
    iters = {name: iter(x) for name, x in loaders.items()}
    if steps_per_epoch is None:
        steps_per_epoch = min(len(x) for x in loaders.values())
    print(f"  每 epoch {steps_per_epoch} 步 × {num_epochs} epoch")

    for epoch in range(1, num_epochs + 1):
        if not freeze_base:
            doc_encoder.train()
        for d in decoders.values():
            d.train()

        running = {name: 0.0 for name in raw}
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for name in raw:
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(loaders[name])
                    batch = next(iters[name])

                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                if freeze_base:
                    with torch.no_grad():
                        doc_memory = doc_encoder(inp, attention_mask=mask)
                else:
                    doc_memory = doc_encoder(inp, attention_mask=mask)  # 基座共享梯度
                l = task_loss(decoders[name], doc_memory, mask, batch, cards[name].spec, device)
                batch_loss = batch_loss + l
                running[name] += l.item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head_params if freeze_base else list(doc_encoder.parameters()) + head_params, 1.0)
            optimizer.step()
            scheduler.step()

        msg = "  ".join(f"{n}={running[n]/steps_per_epoch:.3f}" for n in raw)
        print(f"Epoch {epoch:2d}/{num_epochs} | loss: {msg}")

    # 5. 逐任务可判对错验证
    print("[4/4] 逐任务验证（可判对错指标）...")
    report = {}
    for name in raw:
        metrics = evaluate_task(doc_encoder, decoders[name], val_loaders[name],
                                device, cards[name].spec)
        if freeze_base:
            doc_encoder.eval()          # evaluate_task 末尾会 train()，冻结时不该恢复
        report[name] = metrics
        extra = ""
        if "pair_exact" in metrics:
            extra = (f"  对完整命中率={metrics['pair_exact']*100:5.1f}%"
                     f"  配对顺序={metrics['pair_order']*100:5.1f}%"
                     f"  奇数切片={metrics['pair_odd']*100:5.1f}%")
        print(f"  [{name:9s}] 首切片类别准确率={metrics['cls_acc']*100:5.1f}%  "
              f"区间完全命中率={metrics['span_hit']*100:5.1f}%  "
              f"背景句误报率={metrics['bg_fp']*100:5.1f}%"
              f"{extra}  (n_cls={metrics['n_cls']}, n_bg={metrics['n_bg']})")

    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "hidden_dim": HIDDEN_DIM,
        "encoder": "NanoDocEncoder",
        "encoder_kwargs": ENCODER_KWARGS,
        "decoder_kwargs": DECODER_KWARGS,
        "task_order": list(cards),
        "task_specs": {name: card.spec.to_snapshot() for name, card in cards.items()},
        "train_args": {"num_epochs": num_epochs, "batch_size": batch_size, "seed": seed,
                       "lr_base": lr_base, "lr_head": lr_head, "task_samples": resolved,
                       "freeze_base": freeze_base, "init_from": init_from},
    }, ckpt_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n多任务联合模型已保存至 {ckpt_path} ✅")
    print(f"验证指标已落盘 {metrics_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DTSeek 多任务交替联合训练")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help=f"只训这几张任务卡；留空 = 全部。可选：{sorted(all_tasks())}")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="每 epoch 步数；留空 = 取最短任务的批数（加小任务会拖低总步数）")
    parser.add_argument("--samples-per-task", type=int, default=6000)
    parser.add_argument("--task-samples", action="append", default=[], metavar="NAME=N",
                        help="逐任务样本量覆盖，可重复。例：--task-samples sentiment=32000")
    parser.add_argument("--freeze-base", action="store_true",
                        help="冻结共享基座只训任务头；配合 --init-from 使用")
    parser.add_argument("--init-from", default=None, help="从该 ckpt 热启动基座与同名任务头")
    parser.add_argument("--lr-head", type=float, default=None, help="任务头学习率（默认 1e-3）")
    parser.add_argument("--ckpt", default="checkpoints/multitask_v2_dtseek.pt")
    parser.add_argument("--metrics", default="checkpoints/multitask_v2_metrics.json")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.task_samples:
        name, _, value = item.partition("=")
        if not value.isdigit():
            parser.error(f"--task-samples 需要 NAME=N 形式，收到 {item!r}")
        overrides[name] = int(value)

    train_multitask(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        freeze_base=args.freeze_base,
        init_from=args.init_from,
        **({"lr_head": args.lr_head} if args.lr_head else {}),
        samples_per_task=args.samples_per_task,
        task_samples=overrides or None,
        tasks=args.tasks,
        ckpt_path=args.ckpt,
        metrics_path=args.metrics,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
