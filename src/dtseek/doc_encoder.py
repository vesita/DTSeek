"""Embedding-based Doc Encoder for DTSeek when training from scratch with nanoSeek vocabulary."""
import math
import torch
import torch.nn as nn


class SimpleDocEncoder(nn.Module):
    """A lightweight Transformer encoder (8192 vocab -> hidden_dim) for DTSeek proof-of-concept."""

    def __init__(self, vocab_size: int = 8192, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.embedding(input_ids) + self.pos_emb(pos)
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.norm(h)
