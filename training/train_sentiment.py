"""方案 A 落地：冻结通用 Doc Encoder 主干，仅微调 Decoder 任务卡的情绪切片训练脚本。

核心验证点：
1. 通用编码器 (Doc Encoder) 完全冻结 (requires_grad = False)，参数 100% 保持不动；
2. 仅训练轻量 Decoder 任务卡 (~0.6M 参数)；
3. 验证能否直接学会新任务体系（0:中性，1:积极，2:愤怒，3:悲伤）以及对应情绪关键词的切片定位。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.sentiment_dataset import build_sentiment_dataset
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64, max_steps=3):
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
            step_mask[0] = 1.0  # 中性句：首步直接预测类别 0 与 <eos>
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
                actions[i] = 0 if (i == len(spans) - 1) else 1

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
            "raw_text": item["text"],
        }


def train_sentiment_adapter(num_epochs: int = 8, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = NanoCharTokenizer()
    all_data = build_sentiment_dataset(target_samples=8000)
    val_size = int(len(all_data) * 0.1)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"训练样本数: {len(train_data)}, 验证样本数: {len(val_data)}")

    max_steps = 3
    train_loader = DataLoader(SentimentDataset(train_data, tokenizer, max_steps=max_steps), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SentimentDataset(val_data, tokenizer, max_steps=max_steps), batch_size=batch_size, shuffle=False)

    hidden_dim = 128

    # 1. 加载并完全冻结通用 Doc Encoder 主干 (方案 A 核心)
    doc_encoder = SimpleDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, num_layers=3, num_heads=4).to(device)
    base_ckpt = "checkpoints/robust_ar_dtseek.pt"
    if os.path.exists(base_ckpt):
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        doc_encoder.load_state_dict(ckpt["doc_encoder"])
        print(f"已加载已预训练的主干权重: {base_ckpt}")

    # ★ 彻底冻结主干参数，不计算任何反向梯度！
    for param in doc_encoder.parameters():
        param.requires_grad = False
    doc_encoder.eval()

    trainable_params_doc = sum(p.numel() for p in doc_encoder.parameters() if p.requires_grad)
    print(f"Doc Encoder 可训练参数量: {trainable_params_doc} (已完全冻结 ✅)")

    # 2. 独立构建并训练轻量情绪解码任务卡 (Decoder Task Adapter)
    decoder = RobustARSliceDecoder(hidden_dim=hidden_dim, num_classes=4, num_heads=4, num_layers=2).to(device)
    trainable_params_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder 情绪任务卡可训练参数量: {trainable_params_dec:,d} ({trainable_params_dec/1e6:.2f}M)")

    # 仅将 Decoder 参数送入优化器
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        decoder.train()
        total_loss, total_steps = 0.0, 0

        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            t_labels = batch["labels"].to(device)
            t_starts = batch["starts"].to(device)
            t_ends = batch["ends"].to(device)
            t_nstarts = batch["norm_starts"].to(device)
            t_nends = batch["norm_ends"].to(device)
            t_actions = batch["actions"].to(device)
            step_mask = batch["step_mask"].to(device)
            B = inp.shape[0]

            optimizer.zero_grad()

            # 前向抽取特征图（无梯度，极快）
            with torch.no_grad():
                doc_memory = doc_encoder(inp, attention_mask=mask)

            loss = torch.tensor(0.0, device=device)
            q_seq = decoder.bos_query.expand(B, 1, -1)

            for s in range(max_steps):
                step_out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
                m = step_mask[:, s]

                if m.sum() > 0:
                    l_cls = (F.cross_entropy(step_out["cls_logits"], t_labels[:, s], reduction="none") * m).sum() / m.sum()
                    l_s = (F.cross_entropy(step_out["start_logits"], t_starts[:, s], reduction="none") * m).sum() / m.sum()
                    l_e = (F.cross_entropy(step_out["end_logits"], t_ends[:, s], reduction="none") * m).sum() / m.sum()
                    l_act = (F.cross_entropy(step_out["action_logits"], t_actions[:, s], reduction="none") * m).sum() / m.sum()

                    loss = loss + (l_cls + 1.5 * l_s + 1.5 * l_e + l_act)

                next_q = decoder.get_step_input(
                    prev_hidden=step_out["last_hidden"],
                    prev_cls=t_labels[:, s:s+1],
                    prev_start=t_nstarts[:, s:s+1, None],
                    prev_end=t_nends[:, s:s+1, None],
                )
                q_seq = torch.cat([q_seq, next_q], dim=1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_steps += B

        # Validation
        decoder.eval()
        v_loss, v_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                t_labels = batch["labels"].to(device)
                t_starts = batch["starts"].to(device)
                t_ends = batch["ends"].to(device)
                t_actions = batch["actions"].to(device)
                step_mask = batch["step_mask"].to(device)
                B = inp.shape[0]

                doc_memory = doc_encoder(inp, attention_mask=mask)
                q_seq = decoder.bos_query.expand(B, 1, -1)
                b_loss = torch.tensor(0.0, device=device)

                for s in range(max_steps):
                    step_out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
                    m = step_mask[:, s]
                    if m.sum() > 0:
                        l_cls = (F.cross_entropy(step_out["cls_logits"], t_labels[:, s], reduction="none") * m).sum() / m.sum()
                        l_s = (F.cross_entropy(step_out["start_logits"], t_starts[:, s], reduction="none") * m).sum() / m.sum()
                        l_e = (F.cross_entropy(step_out["end_logits"], t_ends[:, s], reduction="none") * m).sum() / m.sum()
                        l_act = (F.cross_entropy(step_out["action_logits"], t_actions[:, s], reduction="none") * m).sum() / m.sum()
                        b_loss = b_loss + (l_cls + 1.5 * l_s + 1.5 * l_e + l_act)

                    next_q = decoder.get_step_input(
                        prev_hidden=step_out["last_hidden"],
                        prev_cls=t_labels[:, s:s+1],
                        prev_start=batch["norm_starts"].to(device)[:, s:s+1, None],
                        prev_end=batch["norm_ends"].to(device)[:, s:s+1, None],
                    )
                    q_seq = torch.cat([q_seq, next_q], dim=1)

                v_loss += b_loss.item() * B
                v_steps += B

        print(f"Epoch {epoch:2d}/{num_epochs} - 训练 Loss: {total_loss/total_steps:.4f} - 验证 Loss: {v_loss/v_steps:.4f}")

    ckpt_path = "checkpoints/sentiment_adapter_dtseek.pt"
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "hidden_dim": hidden_dim,
        "task_name": "sentiment",
        "classes": ["中性", "积极", "愤怒", "悲伤"],
    }, ckpt_path)
    print(f"\n情绪任务卡适配权重成功保存至 {ckpt_path} ✅")


if __name__ == "__main__":
    train_sentiment_adapter()
