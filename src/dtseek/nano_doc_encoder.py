"""NanoDocEncoder：移植 nanoSeek 实测验证核心资产的高效中文文本编码基座。

与旧版 SimpleDocEncoder 的差异（每一项都来自 nanoSeek 的实测结论）：

1. RMSNorm 替代 LayerNorm
   - 去掉跨通道均值计算（LayerNorm 需要两次 reduce），GPU 上减少显存搬运；
   - 无 bias 参数，同规模下更省参数；
   - 依据：nanoSeek/model/utils.py#RMSNorm（LLaMA / DeepSeek 标准）。

2. RoPE 旋转位置编码替代可学习绝对位置编码
   - 绝对位置编码无法外推，短句训完遇到长句位置直接发散；
   - RoPE 在 q·k 点积里自动带出「相对位置」项 (m-n)，天然支持长度外推；
   - 依据：nanoSeek/model/utils.py#precompute_rope_freqs / apply_rotary_pos_emb。

3. QK-Norm（注意力查询键归一化）
   - 对 q/k 做 L2 归一化到单位球面 + 每头可学习温控标量；
   - 从源头压制点积方差爆炸（Logits 尖刺 → Softmax 熵塌缩 → 注意力退化）；
   - 依据：nanoSeek/model/attention.py 的 use_qk_norm 分支。

4. SwiGLU 门控前馈替代 GELU MLP
   - 输出 = SiLU(x·W1) ⊙ (x·W2)，带可学习门控，同参数量下表达容量更高；
   - 取 hidden = 8/3 · d 时与 GELU MLP 参数量持平；
   - 依据：nanoSeek/model/mlp.py#SwiGLU。

5. Flash Attention（F.scaled_dot_product_attention）
   - 显存由 O(L²) 降为线性，长文本吞吐显著提升；
   - 依据：nanoSeek/model/attention.py 的 flash 分支。

与 nanoSeek 的唯一结构差异：本编码器是 **双向（非因果）** 的。
决策/切片检测任务需要整句的双向上下文（"他"的指代要看后半句），
而 nanoSeek 是自回归语言模型，必须因果掩码。其余组件完全一致。

接口与 SimpleDocEncoder 完全一致（drop-in 替换）：
    encoder = NanoDocEncoder(vocab_size=8192, hidden_dim=128, num_layers=3, num_heads=4)
    doc_memory = encoder(input_ids, attention_mask)   # [B, L, D]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础算子（逐字移植自 nanoSeek/model/utils.py，保持数值行为一致）
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm：只按均方根缩放，不减均值、无 bias。

    LayerNorm: (x - mean) / std * w + b   —— 需要两次跨通道 reduce
    RMSNorm  : x / sqrt(mean(x²) + eps) * w —— 一次 reduce，更快
    """

    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt 比 pow(-0.5) 数值更稳
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 旋转：后半段取负搬到前半段。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的 cos/sin 表，形状 (seq_len, head_dim)。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (seq_len, head_dim)，与 rotate_half 配对
    return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """把 RoPE 作用到 q 与 k 上（两者都加，点积才带相对位置）。

    q, k: (B, n_head, T, head_dim)；cos, sin: (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈：SiLU(x·W1) ⊙ (x·W2) → W3。

    hidden = hidden_scale · hidden_dim，默认 8/3（与 GELU MLP 参数量持平）。
    """

    def __init__(self, hidden_dim: int, hidden_scale: float = 8 / 3, dropout: float = 0.1,
                 clamp: float = 0.0):
        super().__init__()
        hidden = int(hidden_scale * hidden_dim)
        # 对齐到 64 的倍数，保证 Tensor Core / 向量化 kernel 高效
        hidden = max(64, (hidden + 63) // 64 * 64)
        self.clamp = clamp
        self.c_fc = nn.Linear(hidden_dim, hidden, bias=False)    # 值分支
        self.c_fc2 = nn.Linear(hidden_dim, hidden, bias=False)   # 门控分支
        self.c_proj = nn.Linear(hidden, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.c_fc(x)) * self.c_fc2(x)
        if self.clamp > 0:
            # V4 稳定性技巧：在 c_proj 之前钳制门控乘积，从源头压制异常值。
            h = h.clamp(-self.clamp, self.clamp)
        return self.dropout(self.c_proj(h))


# ---------------------------------------------------------------------------
# 高效双向注意力块
# ---------------------------------------------------------------------------

class NanoAttention(nn.Module):
    """双向多头注意力：RoPE + QK-Norm + Flash-SDPA。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 rope_theta: float = 10000.0, use_qk_norm: bool = True):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout_p = dropout

        # QK-Norm：单位球面归一化 + 每头可学习温控标量（初始 sqrt(head_dim)）
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.qk_scale = nn.Parameter(torch.full((num_heads,), self.head_dim ** 0.5))

        self.rope_theta = rope_theta
        # SDPA 内部缩放固定为 1/sqrt(head_dim)；QK-Norm 的可学习 scale 在外部调整有效温度
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            cos, sin: [T, head_dim]
            attn_mask: [B, 1, 1, T] bool，True = 参与注意力（key 维度）
        """
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                    # 各 [B, T, H, hd]
        q = q.transpose(1, 2)                          # [B, H, T, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 1. RoPE 注入相对位置
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 2. QK-Norm：L2 归一到单位球面，再乘每头可学习尺度
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.qk_scale.view(1, -1, 1, 1)
            k = F.normalize(k, dim=-1) * self.qk_scale.view(1, -1, 1, 1)

        # 3. Flash Attention（双向：is_causal=False）
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.attn_scale,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class NanoTransformerBlock(nn.Module):
    """Pre-RMSNorm 残差块：RMSNorm → Attention → 残差 → RMSNorm → SwiGLU → 残差。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 rope_theta: float = 10000.0, use_qk_norm: bool = True,
                 swiglu_scale: float = 8 / 3, swiglu_clamp: float = 0.0):
        super().__init__()
        self.ln_1 = RMSNorm(hidden_dim)
        self.attn = NanoAttention(hidden_dim, num_heads, dropout, rope_theta, use_qk_norm)
        self.ln_2 = RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, swiglu_scale, dropout, swiglu_clamp)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(self.ln_1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# 编码器主体
# ---------------------------------------------------------------------------

class NanoDocEncoder(nn.Module):
    """高效中文文本编码基座（双向，drop-in 替换 SimpleDocEncoder）。

    产出：Doc Memory 特征图 [B, L, D]，供下游各任务卡（Decoder）交叉查询。
    """

    def __init__(self, vocab_size: int = 8192, hidden_dim: int = 128, num_layers: int = 3,
                 num_heads: int = 4, max_len: int = 512, dropout: float = 0.1,
                 rope_theta: float = 10000.0, use_qk_norm: bool = True,
                 swiglu_scale: float = 8 / 3, swiglu_clamp: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.blocks = nn.ModuleList([
            NanoTransformerBlock(hidden_dim, num_heads, dropout, rope_theta,
                                 use_qk_norm, swiglu_scale, swiglu_clamp)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)

        # RoPE cos/sin 表注册为 buffer：随 .to(device) 移动、进 state_dict
        head_dim = hidden_dim // num_heads
        cos, sin = precompute_rope_freqs(head_dim, max_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            input_ids: [B, L] 字符 token id
            attention_mask: [B, L]，1 = 有效，0 = padding（可选）
        Returns:
            doc_memory: [B, L, D]
        """
        B, L = input_ids.shape
        if L > self.max_len:
            raise ValueError(f"序列长度 {L} 超过 max_len {self.max_len}；"
                             f"长文本请先用 segmenter 分句。")

        h = self.embedding(input_ids)

        # RoPE 表按当前实际长度切片
        cos = self.rope_cos[:L]
        sin = self.rope_sin[:L]

        # SDPA 布尔掩码：[B, 1, 1, L]，True = 该 key 位置参与注意力
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(torch.bool).view(B, 1, 1, L)

        for block in self.blocks:
            h = block(h, cos, sin, attn_mask)

        return self.norm(h)
