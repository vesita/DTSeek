"""多任务交替联合训练引擎 v2（nanoSeek 高效基座 + 可判对错的逐任务验证）

训练范式：
- 共享同一个 NanoDocEncoder 基座（RMSNorm + RoPE + QK-Norm + SwiGLU + FlashAttn），
  放开梯度，随三任务交替反向传播持续进化；
- 三个任务各持独立轻量 Decoder 任务卡（Task Cartridge）：
    1. pronoun   : 人称代词切片 (无代词/第一/第二/第三人称)
    2. sentiment : 对话情绪切片 (中性/积极/愤怒/悲伤)
    3. ownership : 发言归属人切片 (无归属/用户/助手/第三方)
- 每个训练步交替消费三任务的 batch，基座同时吸收三路梯度。

为什么必须逐任务验证（而不是只看 loss）：
上一版情绪任务只用 loss 观察，结果把"喜欢"圈成"颜"、对中性疑问句误报，
loss 一路下降却完全没暴露——因为 loss 只衡量"平均拟合"，不衡量"位置是否落在词上"
和"中性句是否被误触发"。这里为每个任务测三个可判对错的指标：
  1. cls_acc      : 首切片类别正确率（分类对不对）
  2. span_hit     : 首切片起止区间与真值完全一致的比例（指针落点准不准）
  3. bg_fp        : 背景句被误报出切片的比例（中性句会不会乱开火，越低越好）
"""
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.nano_doc_encoder import NanoDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder
from dtseek.rich_ar_dataset import build_rich_ar_dataset
from dtseek.sentiment_dataset import build_sentiment_dataset
from dtseek.ownership_dataset import build_ownership_dataset


TASK_SPECS = {
    "pronoun": ["无代词", "第一人称", "第二人称", "第三人称"],
    "sentiment": ["中性", "积极", "愤怒", "悲伤"],
    "ownership": ["无归属", "用户", "助手", "第三方"],
}


class GenericTaskDataset(Dataset):
    """把 {text, spans:[{label,start,end}]} 编码成固定 max_steps 的自回归监督张量。"""

    def __init__(self, data, tokenizer, max_len=64, max_steps=4):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_steps = max_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode(item["text"], max_length=self.max_len, padding=True)
        spans = sorted(item["spans"], key=lambda x: x["start"])[:self.max_steps]
        L = len(item["text"])

        labels = [0] * self.max_steps
        starts = [0] * self.max_steps
        ends = [0] * self.max_steps
        norm_starts = [0.0] * self.max_steps
        norm_ends = [0.0] * self.max_steps
        actions = [0] * self.max_steps
        step_mask = [0.0] * self.max_steps

        if len(spans) == 0:
            # 背景句：首步直接预测类别 0 + <eos>，之后所有步不计损失
            step_mask[0] = 1.0
        else:
            for i, s in enumerate(spans):
                step_mask[i] = 1.0
                labels[i] = s["label"]
                s_idx = min(self.max_len - 1, s["start"])
                e_idx = min(self.max_len - 1, max(s["start"], s["end"] - 1))
                starts[i] = s_idx
                ends[i] = e_idx
                norm_starts[i] = s_idx / max(1, L)
                norm_ends[i] = e_idx / max(1, L)
                actions[i] = 0 if (i == len(spans) - 1) else 1  # 0=<eos>, 1=<cont>

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "starts": torch.tensor(starts, dtype=torch.long),
            "ends": torch.tensor(ends, dtype=torch.long),
            "norm_starts": torch.tensor(norm_starts, dtype=torch.float),
            "norm_ends": torch.tensor(norm_ends, dtype=torch.float),
            "actions": torch.tensor(actions, dtype=torch.long),
            "step_mask": torch.tensor(step_mask, dtype=torch.float),
            "is_bg": torch.tensor(1.0 if len(spans) == 0 else 0.0, dtype=torch.float),
        }


def task_loss(decoder, doc_memory, mask, batch, max_steps, device):
    """单任务的自回归多步损失（教师强制）。

    分类损失对背景类（id=0）加权：背景句在自回归范式下只在第 0 步贡献 **一个**
    监督信号，而有切片的句子贡献 N 个信号 —— 天然被双重稀释。轻微上调背景类权重
    可抑制"永远开火"的退化解（v1 的中性句 100% 误报）。
    """
    t_labels = batch["labels"].to(device)
    t_starts = batch["starts"].to(device)
    t_ends = batch["ends"].to(device)
    t_nstarts = batch["norm_starts"].to(device)
    t_nends = batch["norm_ends"].to(device)
    t_actions = batch["actions"].to(device)
    step_mask = batch["step_mask"].to(device)
    B = doc_memory.shape[0]

    # 背景类权重 1.3，其余 1.0
    cls_w = torch.tensor([1.3, 1.0, 1.0, 1.0], device=device)
    # <eos>(0) 与 <cont>(1) 均衡：稍偏 <cont>，避免过早停机
    act_w = torch.tensor([0.8, 1.2], device=device)

    q_seq = decoder.bos_query.expand(B, 1, -1)
    loss = torch.tensor(0.0, device=device)

    for s in range(max_steps):
        step_out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
        m = step_mask[:, s]
        if m.sum() > 0:
            l_cls = (F.cross_entropy(step_out["cls_logits"], t_labels[:, s],
                                     weight=cls_w, reduction="none") * m).sum() / m.sum()
            l_s = (F.cross_entropy(step_out["start_logits"], t_starts[:, s], reduction="none") * m).sum() / m.sum()
            l_e = (F.cross_entropy(step_out["end_logits"], t_ends[:, s], reduction="none") * m).sum() / m.sum()
            l_act = (F.cross_entropy(step_out["action_logits"], t_actions[:, s],
                                     weight=act_w, reduction="none") * m).sum() / m.sum()
            loss = loss + (l_cls + 1.5 * l_s + 1.5 * l_e + l_act)

        # 教师强制：用真值切片状态驱动下一步（训练稳定、并行度高）
        next_q = decoder.get_step_input(
            prev_hidden=step_out["last_hidden"],
            prev_cls=t_labels[:, s:s + 1],
            prev_start=t_nstarts[:, s:s + 1, None],
            prev_end=t_nends[:, s:s + 1, None],
        )
        q_seq = torch.cat([q_seq, next_q], dim=1)

    return loss


@torch.no_grad()
def evaluate_task(doc_encoder, decoder, loader, device, max_steps, max_len=64):
    """逐任务三指标验证：cls_acc / span_hit / bg_fp。"""
    doc_encoder.eval()
    decoder.eval()

    cls_ok = cls_tot = 0
    span_ok = span_tot = 0
    bg_fired = bg_tot = 0

    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        doc_memory = doc_encoder(inp, attention_mask=mask)
        B = inp.shape[0]

        # 首步：只判"第一个切片"——这是位置漂移与误报最直接的观测量
        q0 = decoder.bos_query.expand(B, 1, -1)
        out = decoder.forward_step(q0, doc_memory, doc_mask=mask)

        pred_cls = out["cls_logits"].argmax(-1)          # [B]
        pred_s = out["start_logits"].argmax(-1)          # [B]
        pred_e = out["end_logits"].argmax(-1)            # [B]

        t_labels = batch["labels"][:, 0].to(device)
        t_s = batch["starts"][:, 0].to(device)
        t_e = batch["ends"][:, 0].to(device)
        is_bg = batch["is_bg"].to(device)

        # 1. 分类正确率（只看有切片的样本，背景句单列进 bg_fp）
        real = (t_labels > 0)
        cls_ok += ((pred_cls == t_labels) & real).sum().item()
        cls_tot += real.sum().item()

        # 2. 区间完全命中率（起止都对才算）
        span_ok += (((pred_s == t_s) & (pred_e == t_e)) & real).sum().item()
        span_tot += real.sum().item()

        # 3. 背景句误开火率（预测类别非 0 即为误报）
        bg_fired += ((pred_cls > 0) & (is_bg > 0.5)).sum().item()
        bg_tot += (is_bg > 0.5).sum().item()

    doc_encoder.train()
    decoder.train()
    return {
        "cls_acc": cls_ok / max(1, cls_tot),
        "span_hit": span_ok / max(1, span_tot),
        "bg_fp": bg_fired / max(1, bg_tot),
        "n_cls": cls_tot,
        "n_bg": bg_tot,
    }


def train_multitask(num_epochs: int = 16, batch_size: int = 64,
                    lr_base: float = 3e-4, lr_head: float = 1e-3,
                    samples_per_task: int = 6000,
                    task_samples: dict = None):
    """多任务交替联合训练。

    Args:
        task_samples: 逐任务样本量覆盖，例如 {"sentiment": 24000}。
            情绪任务词典有 186 词、每类约 62 词，若每类只有 1500 条
            （= 每词仅 8 个样本）模型根本学不全，实测表现为个别词类别判错。
            样本量应与词典规模成比例。
    """
    if task_samples is None:
        task_samples = {"pronoun": samples_per_task,
                        "sentiment": samples_per_task,
                        "ownership": samples_per_task}
    else:
        for k in ("pronoun", "sentiment", "ownership"):
            task_samples.setdefault(k, samples_per_task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"多任务联合调优引擎 v2 | 设备: {device}")

    tokenizer = NanoCharTokenizer()

    # 1. 三任务数据集
    print("[1/4] 构建三任务数据集 ...")
    raw = {
        "pronoun": build_rich_ar_dataset(target_samples=task_samples["pronoun"]),
        "sentiment": build_sentiment_dataset(target_samples=task_samples["sentiment"]),
        "ownership": build_ownership_dataset(target_samples=task_samples["ownership"]),
    }

    max_steps = 4
    _eval_sets = {}
    for name, data in raw.items():
        random.Random(42).shuffle(data)
        n_val = max(200, len(data) // 10)
        _eval_sets[name] = data[:n_val]
        data[:] = data[n_val:]

    loaders = {
        name: DataLoader(GenericTaskDataset(data, tokenizer, max_steps=max_steps),
                         batch_size=batch_size, shuffle=True, drop_last=True)
        for name, data in raw.items()
    }
    val_loaders = {
        name: DataLoader(GenericTaskDataset(_eval_sets[name], tokenizer, max_steps=max_steps),
                         batch_size=batch_size, shuffle=False)
        for name in raw
    }

    # 2. 共享高效基座
    print("[2/4] 构建 NanoDocEncoder 高效基座 ...")
    hidden_dim = 128
    doc_encoder = NanoDocEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=4,
        max_len=128,
        dropout=0.1,
        rope_theta=10000.0,
        use_qk_norm=True,
        swiglu_scale=8 / 3,
    ).to(device)

    # 3. 三张独立任务卡
    decoders = {
        name: RobustARSliceDecoder(hidden_dim=hidden_dim, num_classes=4,
                                   num_heads=4, num_layers=2).to(device)
        for name in raw
    }

    head_params = [p for d in decoders.values() for p in d.parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": doc_encoder.parameters(), "lr": lr_base},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )
    # 余弦退火：基座与任务头同步衰减
    total_steps = num_epochs * min(len(l) for l in loaders.values())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # 4. 交替训练
    print("[3/4] 开始多任务交替联合训练 ...")
    iters = {name: iter(l) for name, l in loaders.items()}
    steps_per_epoch = min(len(l) for l in loaders.values())

    for epoch in range(1, num_epochs + 1):
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
                doc_memory = doc_encoder(inp, attention_mask=mask)  # 基座共享梯度
                l = task_loss(decoders[name], doc_memory, mask, batch, max_steps, device)
                batch_loss = batch_loss + l
                running[name] += l.item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(doc_encoder.parameters()) + head_params, 1.0)
            optimizer.step()
            scheduler.step()

        msg = "  ".join(f"{n}={running[n]/steps_per_epoch:.3f}" for n in raw)
        print(f"Epoch {epoch:2d}/{num_epochs} | loss: {msg}")

    # 5. 逐任务可判对错验证
    print("[4/4] 逐任务验证（可判对错指标）...")
    report = {}
    for name in raw:
        metrics = evaluate_task(doc_encoder, decoders[name], val_loaders[name],
                                device, max_steps)
        report[name] = metrics
        print(f"  [{name:9s}] 首切片类别准确率={metrics['cls_acc']*100:5.1f}%  "
              f"区间完全命中率={metrics['span_hit']*100:5.1f}%  "
              f"背景句误报率={metrics['bg_fp']*100:5.1f}%  "
              f"(n_cls={metrics['n_cls']}, n_bg={metrics['n_bg']})")

    ckpt_path = "checkpoints/multitask_v2_dtseek.pt"
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "hidden_dim": hidden_dim,
        "encoder": "NanoDocEncoder",
        "tasks": TASK_SPECS,
    }, ckpt_path)

    with open("checkpoints/multitask_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n多任务联合模型已保存至 {ckpt_path} ✅")
    print(f"验证指标已落盘 checkpoints/multitask_v2_metrics.json")


if __name__ == "__main__":
    # 情绪任务样本量按词典规模放大：199 词 ⇒ 每类 8000 条，配合数据集内部的
    # "每词下限 60 / 上限 120" 机制，保证词典里每个词都有足够且均衡的监督。
    train_multitask(
        num_epochs=16,
        task_samples={"pronoun": 6000, "sentiment": 32000, "ownership": 8000},
    )
