"""Dynamic Slot-based DETR architecture for Variable-Count Span Slice Classification.

Unlike fixed-anchor YOLO, this model uses K learnable Detection Slots (e.g. K=8).
Each slot independently predicts:
1. Category: [0: background / no-slice, 1: 1st person, 2: 2nd person, 3: 3rd person, ...]
2. Span: (center, width) -> [start, end]
3. Confidence: objectness probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanSlotDecoder(nn.Module):
    """DETR-style Decoder for Variable Span Detection Slots."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_slots: int = 8, num_classes: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Learnable detection query slots [1, K, D]
        self.slots = nn.Parameter(torch.empty(1, num_slots, hidden_dim))
        nn.init.orthogonal_(self.slots)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output Heads per slot
        # 1. Classification (including 0: background/no-object)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        # 2. 1D Span Bounding (center, width) normalized to [0, 1]
        self.span_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
        # 3. Objectness confidence
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        doc_memory: torch.Tensor,
        doc_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            doc_memory: [B, L_doc, D]
            doc_mask: [B, L_doc] (True = valid, False = pad)
        Returns:
            dict:
                logits: [B, num_slots, num_classes]
                probs: [B, num_slots, num_classes]
                spans: [B, num_slots, 2] (center, width)
                span_bounds: [B, num_slots, 2] (start, end)
                confidence: [B, num_slots]
        """
        B = doc_memory.shape[0]
        queries = self.slots.expand(B, -1, -1)
        doc_pad_mask = ~doc_mask.bool() if doc_mask is not None else None

        # Cross attention: Queries attend to Doc Memory with Slot Self-Attention
        h = self.decoder(tgt=queries, memory=doc_memory, memory_key_padding_mask=doc_pad_mask)
        h = self.norm(h)

        logits = self.cls_head(h)
        probs = F.softmax(logits, dim=-1)

        spans = self.span_head(h)  # [B, K, 2] (center, width)
        center, width = spans[..., 0], spans[..., 1]
        start = (center - width / 2.0).clamp(min=0.0, max=1.0)
        end = (center + width / 2.0).clamp(min=0.0, max=1.0)
        span_bounds = torch.stack([start, end], dim=-1)

        conf = self.conf_head(h).squeeze(-1)

        return {
            "logits": logits,
            "probs": probs,
            "spans": spans,
            "span_bounds": span_bounds,
            "confidence": conf,
        }
