"""Production-grade training script with Orthogonal Query Projector & Full Evaluation Suite."""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dtseek.tokenizer import NanoCharTokenizer
from dtseek.real_corpus import build_real_pronoun_dataset
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.model import DTSeekConfig, DTSeekModel
from dtseek.query_projector import TextGuidedQueryProjector
from dtseek.evaluation import evaluate_benchmark, print_eval_report


class PronounDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode(item["text"], max_length=self.max_len, padding=True)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def train_pronoun_model(num_epochs: int = 8, batch_size: int = 64, lr: float = 5e-4, samples_per_class: int = 4000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = NanoCharTokenizer()
    # 4000 per class -> 16,000 real sentences
    all_data = build_real_pronoun_dataset(target_per_class=samples_per_class)
    val_size = int(len(all_data) * 0.1)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_loader = DataLoader(PronounDataset(train_data, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PronounDataset(val_data, tokenizer), batch_size=batch_size, shuffle=False)

    hidden_dim = 128
    doc_encoder = SimpleDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, num_layers=3, num_heads=4).to(device)
    config = DTSeekConfig(hidden_dim=hidden_dim, num_heads=4, num_decoder_layers=2, use_background_class=False)
    dtseek = DTSeekModel(config).to(device)

    # Orthogonal Query Projector (explicit 4 queries: 0:null, 1:1st, 2:2nd, 3:3rd)
    query_proj = TextGuidedQueryProjector(hidden_dim=hidden_dim, num_classes=4, use_background_class=False).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": doc_encoder.parameters(), "lr": lr},
            {"params": dtseek.parameters(), "lr": lr},
            {"params": query_proj.parameters(), "lr": lr * 2.0},
        ],
        weight_decay=1e-4,
    )

    for epoch in range(1, num_epochs + 1):
        doc_encoder.train()
        dtseek.train()
        query_proj.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            B = input_ids.shape[0]

            optimizer.zero_grad()
            doc_memory = doc_encoder(input_ids, attention_mask=mask)
            queries, q_mask = query_proj(batch_size=B)

            # Direct forward through decoder layers
            q = queries
            for layer in dtseek.decoder_layers:
                q = layer(q, doc_memory, doc_mask=mask, query_mask=q_mask)
            q = dtseek.final_norm(q)
            logits = dtseek.cat_scorer(q).squeeze(-1)  # [B, 4]

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += B

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation step
        doc_encoder.eval()
        dtseek.eval()
        query_proj.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                B = input_ids.shape[0]

                doc_memory = doc_encoder(input_ids, attention_mask=mask)
                queries, q_mask = query_proj(batch_size=B)
                q = queries
                for layer in dtseek.decoder_layers:
                    q = layer(q, doc_memory, doc_mask=mask, query_mask=q_mask)
                q = dtseek.final_norm(q)
                logits = dtseek.cat_scorer(q).squeeze(-1)

                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += B

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:2d}/{num_epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc*100:5.1f}% - Val Acc: {val_acc*100:5.1f}%")

    # Save Checkpoint
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "pronoun_dtseek.pt")
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "dtseek": dtseek.state_dict(),
        "query_proj": query_proj.state_dict(),
        "config": config,
    }, ckpt_path)
    print(f"\nModel saved to {ckpt_path}")

    # Run Benchmark Evaluation Suite
    def query_generator(dev):
        q, _ = query_proj(batch_size=1)
        return q

    # Adapter wrapper for evaluation
    class WrapperModel(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, queries, doc_mem, doc_mask=None):
            q = queries
            for layer in self.core.decoder_layers:
                q = layer(q, doc_mem, doc_mask=doc_mask)
            q = self.core.final_norm(q)
            logits = self.core.cat_scorer(q).squeeze(-1)
            probs = F.softmax(logits, dim=-1)
            # Simple margin-based confidence
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            return {"logits": logits, "probs": probs, "confidence": margin}

    eval_data = evaluate_benchmark(
        doc_encoder=doc_encoder,
        dtseek=WrapperModel(dtseek),
        query_generator=query_generator,
        tokenizer=tokenizer,
        device=device,
        val_loader=val_loader,
    )
    print_eval_report(eval_data)


if __name__ == "__main__":
    train_pronoun_model()
