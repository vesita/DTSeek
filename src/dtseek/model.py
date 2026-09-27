"""Core architecture of DTSeek with YOLO/DETR-style Detection Head.

Outputs:
1. Category Classification: Logits per candidate class Query + Background Class.
2. Span Localization (1D Bounding Box): Normalized (center, width) -> (start_token, end_token) in document.
3. Objectness / Confidence: Probability that the detected trigger exists and is valid.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DTSeekConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_decoder_layers: int = 2
    dropout: float = 0.1
    max_doc_len: int = 1024
    use_background_class: bool = True
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
        B, C, D = class_embeddings.shape
        if self.use_background_class:
            null_expanded = self.null_query.expand(B, 1, D)
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
    3. FFN.
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
    """DTSeek End-to-End Decision & Detection Model (YOLO/DETR paradigm)."""

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

        # 3. Output Heads (YOLO style: Class + 1D Box Span + Objectness)
        # 3.1 Category Scorer: output classification logit per query
        self.cat_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # 3.2 Span Localization Head (1D Box regression): (center, width) in [0, 1]
        self.span_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2),
            nn.Sigmoid(),  # Output normalized (center, width)
        )

        # 3.3 Objectness / Confidence Head: predict detection reliability
        self.act_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.register_buffer("temperature", torch.tensor(config.temperature_init))

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        query_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                logits: [Batch, TotalQueries]
                probs: [Batch, TotalQueries]
                spans: [Batch, TotalQueries, 2] -> normalized (center, width)
                span_bounds: [Batch, TotalQueries, 2] -> normalized (start, end)
                confidence: [Batch, 1]
                best_index: [Batch]
                is_background: [Batch]
        """
        B, C, D = class_embeddings.shape
        if self.query_projector.use_background_class:
            queries, q_mask = self.query_projector(class_embeddings)
        else:
            queries = class_embeddings
            q_mask = query_mask if query_mask is not None else torch.ones(B, C, dtype=torch.bool, device=queries.device)

        # Run DETR-style Cross-Decoder
        for layer in self.decoder_layers:
            queries = layer(queries, doc_memory, doc_mask=doc_mask, query_mask=q_mask)
        queries = self.final_norm(queries)

        # 1. Category Classification Logits & Probs
        logits = self.cat_scorer(queries).squeeze(-1)  # [B, TotalQueries]
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        probs = F.softmax(scaled_logits, dim=-1)

        # 2. YOLO-style 1D Span Localization (center, width) -> (start, end)
        spans = self.span_head(queries)  # [B, TotalQueries, 2] (center, width)
        center = spans[..., 0]
        width = spans[..., 1]
        start = (center - width / 2.0).clamp(min=0.0, max=1.0)
        end = (center + width / 2.0).clamp(min=0.0, max=1.0)
        span_bounds = torch.stack([start, end], dim=-1)

        # 3. Objectness & Confidence Estimation
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

        if doc_mask is not None:
            mask_expanded = doc_mask.unsqueeze(-1).float()
            doc_pooled = (doc_memory * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
        else:
            doc_pooled = doc_memory.mean(dim=1)

        confidence = self.act_head(torch.cat([doc_pooled.detach(), feats], dim=-1))

        best_idx = torch.argmax(probs, dim=-1)
        is_background = (best_idx == 0) if self.config.use_background_class else torch.zeros(B, dtype=torch.bool, device=p.device)

        return {
            "logits": logits,
            "probs": probs,
            "spans": spans,
            "span_bounds": span_bounds,
            "confidence": confidence,
            "best_index": best_idx,
            "is_background": is_background,
        }
