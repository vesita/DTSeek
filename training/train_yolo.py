"""Training Script for YOLO-style [Localization Span + Category + Confidence] Model."""
import torch
from torch.utils.data import DataLoader, Dataset

from nano_char_tokenizer import NanoCharTokenizer
from dtseek.detection_dataset import build_detection_dataset
from dtseek.doc_encoder import SimpleDocEncoder
from dtseek.model import DTSeekConfig, DTSeekModel
from dtseek.query_projector import TextGuidedQueryProjector
from dtseek.detection_loss import YOLODetectionLoss


class DetectionDataset(Dataset):
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
        start_norm = item["start"] / max(1, L)
        end_norm = item["end"] / max(1, L)

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "spans": torch.tensor([item["center"], item["width"]], dtype=torch.float),
            "bounds": torch.tensor([start_norm, end_norm], dtype=torch.float),
            "raw_text": item["text"],
            "raw_start": item["start"],
            "raw_end": item["end"],
        }


def train_yolo_detection(num_epochs: int = 10, batch_size: int = 64, lr: float = 6e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = NanoCharTokenizer()
    all_data = build_detection_dataset(target_per_class=3500)
    val_size = int(len(all_data) * 0.1)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    train_loader = DataLoader(DetectionDataset(train_data, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DetectionDataset(val_data, tokenizer), batch_size=batch_size, shuffle=False)

    hidden_dim = 128
    doc_encoder = SimpleDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, num_layers=3, num_heads=4).to(device)
    config = DTSeekConfig(hidden_dim=hidden_dim, num_heads=4, num_decoder_layers=2, use_background_class=False)
    dtseek = DTSeekModel(config).to(device)

    # 4 queries: 0:null background, 1:1st, 2:2nd, 3:3rd
    query_proj = TextGuidedQueryProjector(hidden_dim=hidden_dim, num_classes=4, use_background_class=False).to(device)
    criterion = YOLODetectionLoss(lambda_cls=1.0, lambda_box=3.0, lambda_conf=0.5)

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
        total_loss, total_cls, total_box = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            t_spans = batch["spans"].to(device)
            t_bounds = batch["bounds"].to(device)
            B = inp.shape[0]

            optimizer.zero_grad()
            doc_mem = doc_encoder(inp, attention_mask=mask)
            queries, q_mask = query_proj(batch_size=B)

            out = dtseek(queries, doc_mem, doc_mask=mask, query_mask=q_mask)

            losses = criterion(
                pred_logits=out["logits"],
                pred_spans=out["spans"],
                pred_bounds=out["span_bounds"],
                pred_conf=out["confidence"],
                target_labels=labels,
                target_spans=t_spans,
                target_bounds=t_bounds,
            )

            losses["loss"].backward()
            optimizer.step()

            total_loss += losses["loss"].item() * B
            total_cls += losses["loss_cls"].item() * B
            total_box += losses["loss_box"].item() * B
            preds = torch.argmax(out["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += B

        train_acc = correct / total

        # Validation
        doc_encoder.eval()
        dtseek.eval()
        query_proj.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                t_spans = batch["spans"].to(device)
                t_bounds = batch["bounds"].to(device)
                B = inp.shape[0]

                doc_mem = doc_encoder(inp, attention_mask=mask)
                queries, q_mask = query_proj(batch_size=B)
                out = dtseek(queries, doc_mem, doc_mask=mask, query_mask=q_mask)

                v_losses = criterion(
                    pred_logits=out["logits"],
                    pred_spans=out["spans"],
                    pred_bounds=out["span_bounds"],
                    pred_conf=out["confidence"],
                    target_labels=labels,
                    target_spans=t_spans,
                    target_bounds=t_bounds,
                )
                v_loss += v_losses["loss"].item() * B
                preds = torch.argmax(out["logits"], dim=-1)
                v_correct += (preds == labels).sum().item()
                v_total += B

        val_acc = v_correct / v_total
        print(f"Epoch {epoch:2d}/{num_epochs} - Loss: {total_loss/total:.3f} (Cls: {total_cls/total:.3f}, Box: {total_box/total:.3f}) - Cls Acc: {train_acc*100:.1f}% - Val Acc: {val_acc*100:.1f}%")

    ckpt_path = "checkpoints/yolo_dtseek.pt"
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "dtseek": dtseek.state_dict(),
        "query_proj": query_proj.state_dict(),
        "config": config,
    }, ckpt_path)
    print(f"\nYOLO-style Decision & Detection Model saved to {ckpt_path} ✅")


if __name__ == "__main__":
    train_yolo_detection()
