"""Core architecture of DTSeek.

Combines:
1. Input Doc Encoder (Single forward pass for long text)
2. Task Query Projector with constant Background class (Q_null)
3. Cross-Attention Decoder (DETR style: Self-Attn for mutual exclusion + Cross-Attn over Doc Memory)
4. Category Scorer & Confidence / Act Heads
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DTSeekConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 2
    dropout: float = 0.1
    max_doc_len: int = 4096
    # Task Query parameters
    use_background_class: bool = True  # Q_null for OOD / unclassified detection
    temperature_init: float = 1.0


class CategoryQueryProjector(nn.Module):
    """Maps candidate class descriptions or pre-embedded labels into D-dimensional Queries.
    
    Includes a learned background token (Q_null) similar to DETR's no-object token.
    """
    def __init__(self, hidden_dim: int, use_background_class: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_background_class = use_background_class
        if use_background_class:
            self.null_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        else:
            self.null_query = None

    def forward(self, class_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            class_embeddings: [Batch, NumClasses, HiddenDim]
        Returns:
            queries: [Batch, NumClasses (+1 if null), HiddenDim]
            query_mask: [Batch, NumClasses (+1 if null)] boolean mask (True = valid)
        """
        B, C, D = class_embeddings.shape
        if self.use_background_class:
            null_expanded = self.null_query.expand(B, 1, D)
            # Null query is always appended at index 0 as background class
            queries = torch.cat([null_expanded, class_embeddings], dim=1)
            mask = torch.ones(B, C + 1, dtype=torch.bool, device=class_embeddings.device)
        else:
            queries = class_embeddings
            mask = torch.ones(B, C, dtype=torch.bool, device=class_embeddings.device)
        return queries, mask


class DETRDecoderLayer(nn.Module):
    """DETR-style Decoder layer:
    1. Self-Attention between queries (models category competition / mutual exclusion).
    2. Cross-Attention from queries to document memory.
    3. FFN (SwiGLU or standard GELU FFN).
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        doc_memory: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, C, D]
            doc_memory: [B, L_doc, D]
            doc_mask: [B, L_doc] (True = valid token, False = padding)
            query_mask: [B, C] (True = valid query, False = padding)
        """
        # 1. Self Attention across queries
        key_padding_mask = ~query_mask if query_mask is not None else None
        q_norm = self.norm1(queries)
        q2, _ = self.self_attn(q_norm, q_norm, q_norm, key_padding_mask=key_padding_mask)
        queries = queries + q2

        # 2. Cross Attention: Queries attend to Doc Memory
        doc_padding_mask = ~doc_mask if doc_mask is not None else None
        q_norm = self.norm2(queries)
        q_cross, _ = self.cross_attn(q_norm, doc_memory, doc_memory, key_padding_mask=doc_padding_mask)
        queries = queries + q_cross

        # 3. FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class DTSeekModel(nn.Module):
    """DTSeek End-to-End Decision Model."""

    def __init__(self, config: DTSeekConfig, doc_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.doc_encoder = doc_encoder
        self.hidden_dim = config.hidden_dim

        # 1. Query Projector
        self.query_projector = CategoryQueryProjector(config.hidden_dim, config.use_background_class)

        # 2. Cross-Decoder Stack
        self.decoder_layers = nn.ModuleList([
            DETRDecoderLayer(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # 3. Output Heads
        # Category Scorer: output 1 logit per candidate query
        self.cat_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # Confidence / Act Head: predict overall calibration score & action readiness
        # Takes pooled document representation + top-2 margin & entropy
        self.act_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Learnable temperature for calibration
        self.register_buffer("temperature", torch.tensor(config.temperature_init))

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Runs Doc Encoder to extract Memory Cache [B, L_doc, D]."""
        if self.doc_encoder is not None:
            output = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state
            return output
        raise NotImplementedError("Doc encoder not provided; pass doc_memory directly.")

    def forward(
        self,
        class_embeddings: torch.Tensor,
        doc_memory: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            class_embeddings: [Batch, NumClasses, HiddenDim]
            doc_memory: [Batch, L_doc, HiddenDim]
            doc_mask: [Batch, L_doc] (True = valid, False = pad)
        Returns:
            Dict containing:
                logits: [Batch, NumClasses (+1 if background)]
                probs: [Batch, NumClasses (+1 if background)]
                confidence: [Batch, 1]
                is_background: [Batch] bool tensor
        """
        B, C, D = class_embeddings.shape
        queries, q_mask = self.query_projector(class_embeddings)

        # Run DETR-style Cross-Decoder
        for layer in self.decoder_layers:
            queries = layer(queries, doc_memory, doc_mask=doc_mask, query_mask=q_mask)
        queries = self.final_norm(queries)

        # 1. Category Score (Logits per Query)
        logits = self.cat_scorer(queries).squeeze(-1)  # [B, TotalQueries]

        # Scaled by calibration temperature
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        probs = F.softmax(scaled_logits, dim=-1)

        # 2. Extract confidence features (Top1, Margin, Entropy, K)
        p = probs.detach()
        num_classes = p.shape[-1]
        top1 = p.topk(1, dim=-1).values
        if num_classes >= 2:
            top2 = p.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = top1[:, 0]

        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1) / max(1.0, float(torch.log(torch.tensor(max(2, num_classes)))))
        k_feat = torch.full((B,), num_classes / 255.0, device=p.device)
        feats = torch.stack([top1.squeeze(-1), margin, ent, k_feat], dim=-1)

        # Pooled doc representation (mean pooling over valid tokens)
        if doc_mask is not None:
            mask_expanded = doc_mask.unsqueeze(-1).float()
            doc_pooled = (doc_memory * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
        else:
            doc_pooled = doc_memory.mean(dim=1)

        confidence = self.act_head(torch.cat([doc_pooled.detach(), feats], dim=-1))

        # Check background class (at index 0 if enabled)
        best_idx = torch.argmax(probs, dim=-1)
        is_background = (best_idx == 0) if self.config.use_background_class else torch.zeros(B, dtype=torch.bool, device=p.device)

        return {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
            "best_index": best_idx,
            "is_background": is_background,
        }
