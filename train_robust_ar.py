"""Fast & Accurate Robust AR Slice Training on focused multi-pronoun dataset."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.rich_ar_dataset import build_rich_ar_dataset
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.robust_ar_model import RobustARSliceDecoder


class RobustARDataset(Dataset):
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
            step_mask[0] = 1.0  # 1 step predicting <eos> and class=0
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


def train_fast_robust(num_epochs: int = 6, batch_size: int = 128, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = NanoCharTokenizer()
    all_data = build_rich_ar_dataset(target_samples=5000)
    val_size = int(len(all_data) * 0.1)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"Train samples: {len(train_data)}, Val: {len(val_data)}")

    max_steps = 4
    train_loader = DataLoader(RobustARDataset(train_data, tokenizer, max_steps=max_steps), batch_size=batch_size, shuffle=True)

    hidden_dim = 128
    doc_encoder = SimpleDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, num_layers=3, num_heads=4).to(device)
    decoder = RobustARSliceDecoder(hidden_dim=hidden_dim, num_classes=4, num_heads=4, num_layers=2).to(device)

    optimizer = torch.optim.AdamW(
        list(doc_encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, num_epochs + 1):
        doc_encoder.train()
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
            doc_memory = doc_encoder(inp, attention_mask=mask)

            loss = torch.tensor(0.0, device=device)
            q_seq = decoder.bos_query.expand(B, 1, -1)  # [B, 1, D]

            for s in range(max_steps):
                step_out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
                m = step_mask[:, s]

                if m.sum() > 0:
                    l_cls = (F.cross_entropy(step_out["cls_logits"], t_labels[:, s], reduction="none") * m).sum() / m.sum()
                    l_s = (F.cross_entropy(step_out["start_logits"], t_starts[:, s], reduction="none") * m).sum() / m.sum()
                    l_e = (F.cross_entropy(step_out["end_logits"], t_ends[:, s], reduction="none") * m).sum() / m.sum()
                    action_weight = torch.tensor([0.7, 1.3], device=device)
                    l_act = (F.cross_entropy(step_out["action_logits"], t_actions[:, s], weight=action_weight, reduction="none") * m).sum() / m.sum()

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

        print(f"Epoch {epoch:2d}/{num_epochs} - Train Loss: {total_loss/total_steps:.4f}")

    ckpt_path = "checkpoints/robust_ar_dtseek.pt"
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "hidden_dim": hidden_dim,
        "max_steps": max_steps,
    }, ckpt_path)
    print(f"\nRobust AR Slice Model successfully saved to {ckpt_path} ✅")


if __name__ == "__main__":
    train_fast_robust()
