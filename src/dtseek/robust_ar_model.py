"""Production-grade Autoregressive Slice Emission Model with Causal Masking & Slice Embedding Feedback.

Enhancements:
1. Target-level Causal Mask: Query tokens cannot peek into future slice slots!
2. Explicit Slice Projection: Each emitted slice projects (class_embedding + position_embedding)
   into the next step query input, so the model unambiguously knows WHICH slice it just completed!
3. Robust Dual Pointer Head with Cosine scaling.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustARSliceDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 4, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # BOS Query Token representing "Emit first slice"
        self.bos_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.05)

        # Category embedding for previous step feedback
        self.cls_embedding = nn.Embedding(num_classes, hidden_dim)
        # Position embedding for start and end token pointers
        self.pos_proj = nn.Linear(2, hidden_dim)

        # Slice Feedback Projector: mixes [previous_hidden, class_emb, pos_emb] -> next step input query
        self.feedback_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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

        # Output Heads
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.start_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.end_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 2)  # 0: <eos>, 1: <cont>

    def get_step_input(
        self,
        prev_hidden: torch.Tensor,   # [B, 1, D]
        prev_cls: torch.Tensor,      # [B, 1]
        prev_start: torch.Tensor,    # [B, 1] (normalized in [0, 1])
        prev_end: torch.Tensor,      # [B, 1] (normalized in [0, 1])
    ) -> torch.Tensor:
        """Constructs conditioned query for next step based on emitted slice."""
        c_emb = self.cls_embedding(prev_cls)  # [B, 1, D]
        pos_vec = torch.cat([prev_start, prev_end], dim=-1)  # [B, 1, 2]
        p_emb = self.pos_proj(pos_vec)        # [B, 1, D]

        fused = torch.cat([prev_hidden, c_emb, p_emb], dim=-1)  # [B, 1, 3D]
        return self.feedback_proj(fused)  # [B, 1, D]

    def forward_step(
        self,
        query_sequence: torch.Tensor,    # [B, step_len, D]
        doc_memory: torch.Tensor,        # [B, L, D]
        doc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, step_len, D = query_sequence.shape
        L = doc_memory.shape[1]
        doc_pad_mask = ~doc_mask.bool() if doc_mask is not None else None

        # Standard Causal Mask for autoregressive queries
        causal_mask = nn.Transformer.generate_square_subsequent_mask(step_len, device=query_sequence.device)

        h = self.decoder(
            tgt=query_sequence,
            memory=doc_memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=doc_pad_mask,
        )
        h = self.norm(h)
        last_h = h[:, -1]  # [B, D]

        cls_logits = self.cls_head(last_h)

        # Dual Pointer Network
        s_query = self.start_ptr(last_h).unsqueeze(1)  # [B, 1, D]
        start_logits = torch.bmm(s_query, doc_memory.transpose(1, 2)).squeeze(1) / (D ** 0.5)

        e_query = self.end_ptr(last_h).unsqueeze(1)
        end_logits = torch.bmm(e_query, doc_memory.transpose(1, 2)).squeeze(1) / (D ** 0.5)

        if doc_mask is not None:
            start_logits = start_logits.masked_fill(~doc_mask, -1e4)
            end_logits = end_logits.masked_fill(~doc_mask, -1e4)

        action_logits = self.action_head(last_h)

        return {
            "cls_logits": cls_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "action_logits": action_logits,
            "last_hidden": last_h.unsqueeze(1),
        }
