"""Autoregressive Slice Emission Model (Pointer / Anchor Sequence Paradigm).

Architecture:
1. Doc Encoder encodes raw text -> doc_memory [1, L, D]
2. Auto-regressive Slice Decoder:
   Step 0: Feed <bos_slice>, cross-attend doc_memory -> Output Slot:
       - Category (1..C)
       - Span: (start_idx, end_idx) using dual-pointer over L
       - Next Action: <cont> (continue to next slice) vs <eos> (stop, no more slices)
   If <cont>, feed previously predicted slice embedding back into decoder -> predict next slice!
   If <eos> or max_slices reached -> Finish!
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveSliceDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 4, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # 0: null, 1: 1st, 2: 2nd, 3: 3rd

        # Action: 0: <eos> (stop emission), 1: <cont> (continue emitting)
        self.bos_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.05)

        # Slice embedding projection: projects previously predicted (class_emb + start_pos + end_pos) into decoder
        self.slice_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
        # 1. Classification
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        # 2. Dual Pointer Network: directly predicts start and end logits over L text tokens
        self.start_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.end_ptr = nn.Linear(hidden_dim, hidden_dim)
        # 3. Action Head: <eos>(0) vs <cont>(1)
        self.action_head = nn.Linear(hidden_dim, 2)

    def forward_step(
        self,
        query_history: torch.Tensor,     # [B, step_len, D]
        doc_memory: torch.Tensor,        # [B, L, D]
        doc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Runs single autoregressive step.
        Returns:
            cls_logits: [B, num_classes]
            start_logits: [B, L]
            end_logits: [B, L]
            action_logits: [B, 2] (0: <eos>, 1: <cont>)
            last_hidden: [B, 1, D]
        """
        B, L, D = doc_memory.shape
        doc_pad_mask = ~doc_mask.bool() if doc_mask is not None else None

        h = self.decoder(tgt=query_history, memory=doc_memory, memory_key_padding_mask=doc_pad_mask)
        h = self.norm(h)
        last_h = h[:, -1]  # [B, D]

        # 1. Category Logits
        cls_logits = self.cls_head(last_h)  # [B, C]

        # 2. Dual Pointer Network over doc_memory (Dot-product attention over text positions)
        # start_logits = (last_h * W_s) . doc_memory^T -> [B, L]
        s_query = self.start_ptr(last_h).unsqueeze(1)  # [B, 1, D]
        start_logits = torch.bmm(s_query, doc_memory.transpose(1, 2)).squeeze(1) / (D ** 0.5)

        e_query = self.end_ptr(last_h).unsqueeze(1)    # [B, 1, D]
        end_logits = torch.bmm(e_query, doc_memory.transpose(1, 2)).squeeze(1) / (D ** 0.5)

        if doc_mask is not None:
            start_logits = start_logits.masked_fill(~doc_mask, -1e4)
            end_logits = end_logits.masked_fill(~doc_mask, -1e4)

        # 3. Action Logits: <eos>(0) vs <cont>(1)
        action_logits = self.action_head(last_h)  # [B, 2]

        return {
            "cls_logits": cls_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "action_logits": action_logits,
            "last_hidden": last_h.unsqueeze(1),
        }
