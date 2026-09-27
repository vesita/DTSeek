"""具备因果掩码与切片显式反馈机制的稳健自回归切片发射解码器。

核心机制：
1. 目标端因果注意力掩码（Causal Mask）：确保 Query 序列只能回顾已发出的历史切片，严禁窥探未来未生成的切片；
2. 显式切片特征反馈（Explicit Feedback Projection）：每一步将上一轮预测的类别嵌入与位置向量投影回填，让模型明确感知“已扫描过哪些区域”；
3. 双指针网络（Dual Pointer Network）：通过点积注意力直接索引文档文本的起止字符索引。
"""
import torch
import torch.nn as nn


class RobustARSliceDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 4, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 起始 Query Token（代表“开始发射第 1 个切片”）
        self.bos_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.05)

        # 类别嵌入层（将上一轮预测类别映射为特征）
        self.cls_embedding = nn.Embedding(num_classes, hidden_dim)
        # 位置投射层（将上一轮起止位置归一化值映射为特征）
        self.pos_proj = nn.Linear(2, hidden_dim)

        # 切片反馈投影器：将 [上一轮隐状态, 类别嵌入, 位置向量] 融合映射为下一轮的驱动 Query
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

        # 输出头
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.start_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.end_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 2)  # 0: <eos>(终止), 1: <cont>(继续发射)

    def get_step_input(
        self,
        prev_hidden: torch.Tensor,   # [B, 1, D]
        prev_cls: torch.Tensor,      # [B, 1]
        prev_start: torch.Tensor,    # [B, 1] (归一化起止位置 [0, 1])
        prev_end: torch.Tensor,      # [B, 1]
    ) -> torch.Tensor:
        """基于上一轮发射的切片构建下一轮的条件驱动 Query。"""
        c_emb = self.cls_embedding(prev_cls)  # [B, 1, D]
        pos_vec = torch.cat([prev_start, prev_end], dim=-1)  # [B, 1, 2]
        p_emb = self.pos_proj(pos_vec)        # [B, 1, D]

        fused = torch.cat([prev_hidden, c_emb, p_emb], dim=-1)  # [B, 1, 3D]
        return self.feedback_proj(fused)  # [B, 1, D]

    def forward_step(
        self,
        query_sequence: torch.Tensor,    # [B, step_len, D]
        doc_memory: torch.Tensor,        # [B, L, D]
        doc_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, step_len, D = query_sequence.shape
        L = doc_memory.shape[1]
        doc_pad_mask = ~doc_mask.bool() if doc_mask is not None else None

        # 生成因果注意力掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(step_len, device=query_sequence.device)

        h = self.decoder(
            tgt=query_sequence,
            memory=doc_memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=doc_pad_mask,
        )
        h = self.norm(h)
        last_h = h[:, -1]  # 取当前步的最新隐状态 [B, D]

        cls_logits = self.cls_head(last_h)

        # 双指针点积注意力计算
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
