"""Training script for Autoregressive Sequential Slice Emission (<slice>...<cont>/<eos>)."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.multispan_dataset import build_multispan_dataset
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.ar_slice_model import AutoregressiveSliceDecoder


class ARDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode(item["text"], max_length=self.max_len, padding=True)
        spans = item["spans"]

        # Sort spans by text position
        spans = sorted(spans, key=lambda x: x["start"])

        slice_steps = []
        for i, s in enumerate(spans):
            is_last = (i == len(spans) - 1)
            # action: 1 for <cont>, 0 for <eos>
            action = 0 if is_last else 1
            slice_steps.append({
                "label": s["label"],
                "start": min(self.max_len - 1, s["start"]),
                "end": min(self.max_len - 1, s["end"] - 1),  # pointer to last token of span
                "action": action,
            })

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "slice_steps": slice_steps,
            "raw_text": item["text"],
        }


def collate_ar(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "slice_steps": [b["slice_steps"] for b in batch],
    }


def train_ar_slices(num_epochs: int = 10, batch_size: int = 64, lr: float = 8e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = NanoCharTokenizer()
    all_data = build_multispan_dataset(target_samples=10000)
    val_size = int(len(all_data) * 0.1)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"Train samples: {len(train_data)}, Val: {len(val_data)}")

    train_loader = DataLoader(ARDataset(train_data, tokenizer), batch_size=batch_size, shuffle=True, collate_fn=collate_ar)
    val_loader = DataLoader(ARDataset(val_data, tokenizer), batch_size=batch_size, shuffle=False, collate_fn=collate_ar)

    hidden_dim = 128
    doc_encoder = SimpleDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, num_layers=3, num_heads=4).to(device)
    decoder = AutoregressiveSliceDecoder(hidden_dim=hidden_dim, num_classes=4, num_heads=4, num_layers=2).to(device)

    optimizer = torch.optim.AdamW(
        list(doc_encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, num_epochs + 1):
        doc_encoder.train()
        decoder.train()
        total_loss = 0.0
        total_steps = 0

        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            batch_slices = batch["slice_steps"]
            B = inp.shape[0]

            optimizer.zero_grad()
            doc_memory = doc_encoder(inp, attention_mask=mask)

            loss = torch.tensor(0.0, device=device)
            batch_loss_count = 0

            # Teacher forcing across dynamic slice steps
            for b in range(B):
                slices = batch_slices[b]
                q_hist = decoder.bos_query.clone()  # [1, 1, D]

                if len(slices) == 0:
                    # Negative sentence: directly predict <eos> (action=0) and label=0
                    step_out = decoder.forward_step(q_hist, doc_memory[b:b+1], doc_mask=mask[b:b+1])
                    l_cls = F.cross_entropy(step_out["cls_logits"], torch.tensor([0], device=device))
                    l_act = F.cross_entropy(step_out["action_logits"], torch.tensor([0], device=device))
                    loss = loss + (l_cls + l_act)
                    batch_loss_count += 1
                else:
                    for s in slices:
                        step_out = decoder.forward_step(q_hist, doc_memory[b:b+1], doc_mask=mask[b:b+1])
                        l_cls = F.cross_entropy(step_out["cls_logits"], torch.tensor([s["label"]], device=device))
                        l_s = F.cross_entropy(step_out["start_logits"], torch.tensor([s["start"]], device=device))
                        l_e = F.cross_entropy(step_out["end_logits"], torch.tensor([s["end"]], device=device))
                        l_act = F.cross_entropy(step_out["action_logits"], torch.tensor([s["action"]], device=device))

                        loss = loss + (l_cls + 1.5 * l_s + 1.5 * l_e + l_act)
                        batch_loss_count += 1

                        # Append step representation back into query history
                        q_hist = torch.cat([q_hist, step_out["last_hidden"]], dim=1)

            loss = loss / max(1, batch_loss_count)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_steps += B

        print(f"Epoch {epoch:2d}/{num_epochs} - Loss: {total_loss/total_steps:.4f}")

    ckpt_path = "checkpoints/ar_slice_dtseek.pt"
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "hidden_dim": hidden_dim,
    }, ckpt_path)
    print(f"\nAutoregressive Slice Model saved to {ckpt_path} ✅")


if __name__ == "__main__":
    train_ar_slices()
