"""Training script for Dynamic Variable Multi-Span Classification Model (DETR paradigm)."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.multispan_dataset import build_multispan_dataset
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.slot_detector import SpanSlotDecoder
from dtseek.slot_loss import HungarianMatcher, SetCriterion


class MultiSpanDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode(item["text"], max_length=self.max_len, padding=True)
        L = len(item["text"])

        labels, spans = [], []
        for s in item["spans"]:
            labels.append(s["label"])
            spans.append([s["center"], s["width"]])

        if len(labels) == 0:
            target_labels = torch.empty(0, dtype=torch.int64)
            target_spans = torch.empty((0, 2), dtype=torch.float)
        else:
            target_labels = torch.tensor(labels, dtype=torch.int64)
            target_spans = torch.tensor(spans, dtype=torch.float)

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "target_labels": target_labels,
            "target_spans": target_spans,
            "raw_text": item["text"],
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    targets = []
    for b in batch:
        targets.append({
            "labels": b["target_labels"],
            "spans": b["target_spans"],
            "raw_text": b["raw_text"],
        })
    return {"input_ids": input_ids, "attention_mask": attention_mask, "targets": targets}


def train_multispan_model(num_epochs: int = 12, batch_size: int = 64, lr: float = 8e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = NanoCharTokenizer()
    all_data = build_multispan_dataset(target_samples=16000)
    val_size = int(len(all_data) * 0.1)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    train_loader = DataLoader(MultiSpanDataset(train_data, tokenizer), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MultiSpanDataset(val_data, tokenizer), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    hidden_dim = 128
    doc_encoder = SimpleDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, num_layers=3, num_heads=4).to(device)
    # Configure 8 Detection Slots (K=8 parallel slots per sentence)
    slot_decoder = SpanSlotDecoder(hidden_dim=hidden_dim, num_heads=4, num_slots=8, num_classes=4, num_layers=2).to(device)

    matcher = HungarianMatcher(cost_class=1.0, cost_span=3.0)
    criterion = SetCriterion(matcher=matcher, num_classes=4, eos_coef=0.25).to(device)

    optimizer = torch.optim.AdamW(
        list(doc_encoder.parameters()) + list(slot_decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, num_epochs + 1):
        doc_encoder.train()
        slot_decoder.train()
        total_loss, total_ce, total_span = 0.0, 0.0, 0.0
        total_matched = 0
        total_batches = 0

        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["targets"]]
            B = inp.shape[0]

            optimizer.zero_grad()
            doc_mem = doc_encoder(inp, attention_mask=mask)
            out = slot_decoder(doc_mem, doc_mask=mask)

            loss_dict = criterion(
                pred_logits=out["logits"],
                pred_spans=out["spans"],
                pred_bounds=out["span_bounds"],
                pred_conf=out["confidence"],
                targets=targets,
            )

            loss_dict["loss"].backward()
            optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_ce += loss_dict["loss_ce"].item()
            total_span += loss_dict["loss_span"].item()
            total_matched += loss_dict["num_matched"]
            total_batches += 1

        # Validation
        doc_encoder.eval()
        slot_decoder.eval()
        v_loss, v_ce, v_span, v_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["targets"]]
                doc_mem = doc_encoder(inp, attention_mask=mask)
                out = slot_decoder(doc_mem, doc_mask=mask)

                v_loss_dict = criterion(
                    pred_logits=out["logits"],
                    pred_spans=out["spans"],
                    pred_bounds=out["span_bounds"],
                    pred_conf=out["confidence"],
                    targets=targets,
                )
                v_loss += v_loss_dict["loss"].item()
                v_ce += v_loss_dict["loss_ce"].item()
                v_span += v_loss_dict["loss_span"].item()
                v_batches += 1

        print(f"Epoch {epoch:2d}/{num_epochs} - Train Loss: {total_loss/total_batches:.3f} (CE: {total_ce/total_batches:.3f}, Span: {total_span/total_batches:.3f}) - Val Loss: {v_loss/v_batches:.3f}")

    ckpt_path = "checkpoints/slot_multispan_dtseek.pt"
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "slot_decoder": slot_decoder.state_dict(),
        "hidden_dim": hidden_dim,
        "num_slots": 8,
    }, ckpt_path)
    print(f"\nMulti-Span Detection Model successfully saved to {ckpt_path} ✅")


if __name__ == "__main__":
    train_multispan_model()
