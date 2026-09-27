"""DTSeek 动态切片决策模型核心架构定义。

包含模块：
1. Doc Encoder 文本特征提取器（对长文本单次前向编码）
2. 候选类别 Query 投影层（注入常驻学习式背景类 Q_null，防止强制归类）
3. DETR 风格 Cross-Decoder 解码交互层（类别自注意力建模互斥性 + 跨模态文本特征检索）
4. 双头输出层（分类打分头 + 1D 边界切片回归头 + 置信度判决头）
"""
from dataclasses import dataclass

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
    """候选类别 Query 投射器：将类别描述或预设 ID 映射为 D 维查询向量。
    
    自动引入常驻学习式背景类（Q_null），类似于目标检测中的 No-Object 槽位。
    """
    def __init__(self, hidden_dim: int, use_background_class: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_background_class = use_background_class
        if use_background_class:
            self.null_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        else:
            self.null_query = None

    def forward(self, class_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    """DETR 风格交叉注意力解码层：
    1. Query 之间自注意力（Self-Attention）：学习类别间的竞争与互斥关系；
    2. Query 查询文本记忆（Cross-Attention）：主动检索输入文本中的证据；
    3. FFN 前馈网络。
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
        doc_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Query 之间的自注意力计算
        key_padding_mask = ~query_mask if query_mask is not None else None
        q_norm = self.norm1(queries)
        q2, _ = self.self_attn(q_norm, q_norm, q_norm, key_padding_mask=key_padding_mask)
        queries = queries + q2

        # 2. 跨模态注意力：Query 检索文本特征图
        doc_padding_mask = ~doc_mask if doc_mask is not None else None
        q_norm = self.norm2(queries)
        q_cross, _ = self.cross_attn(q_norm, doc_memory, doc_memory, key_padding_mask=doc_padding_mask)
        queries = queries + q_cross

        # 3. 前馈网络
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class DTSeekModel(nn.Module):
    """DTSeek 端到端决策与语句切片检测模型。"""

    def __init__(self, config: DTSeekConfig, doc_encoder: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.doc_encoder = doc_encoder
        self.hidden_dim = config.hidden_dim

        # 1. 类别 Query 投射器
        self.query_projector = CategoryQueryProjector(config.hidden_dim, config.use_background_class)

        # 2. 交叉解码器堆叠
        self.decoder_layers = nn.ModuleList([
            DETRDecoderLayer(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # 3. 输出头（类别打分 + 1D 切片回归 + 置信度判决）
        # 3.1 分类打分头
        self.cat_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # 3.2 语句切片回归头：输出归一化的 (center, width)
        self.span_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2),
            nn.Sigmoid(),
        )

        # 3.3 置信度头：综合评估决策的可靠性与置信度
        self.act_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.register_buffer("temperature", torch.tensor(config.temperature_init))

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """运行文档编码器提取长文本记忆特征图。"""
        if self.doc_encoder is not None:
            output = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state
            return output
        raise NotImplementedError("未配置文档编码器，请直接传递 doc_memory。")

    def forward(
        self,
        class_embeddings: torch.Tensor,
        doc_memory: torch.Tensor,
        doc_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, C, D = class_embeddings.shape
        if self.query_projector.use_background_class:
            queries, q_mask = self.query_projector(class_embeddings)
        else:
            queries = class_embeddings
            q_mask = query_mask if query_mask is not None else torch.ones(B, C, dtype=torch.bool, device=queries.device)

        # 逐层运行 DETR 交叉解码
        for layer in self.decoder_layers:
            queries = layer(queries, doc_memory, doc_mask=doc_mask, query_mask=q_mask)
        queries = self.final_norm(queries)

        # 1. 预测类别 Logits 与归一化概率
        logits = self.cat_scorer(queries).squeeze(-1)
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        probs = F.softmax(scaled_logits, dim=-1)

        # 2. 预测语句切片归一化区间 (center, width) -> (start, end)
        spans = self.span_head(queries)
        center = spans[..., 0]
        width = spans[..., 1]
        start = (center - width / 2.0).clamp(min=0.0, max=1.0)
        end = (center + width / 2.0).clamp(min=0.0, max=1.0)
        span_bounds = torch.stack([start, end], dim=-1)

        # 3. 统计特征提取与置信度估计
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
