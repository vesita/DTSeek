"""可学习的 n-gram 结构 —— 与 `MentionNDB` **同一套键**，唯一变量是「表里存的是不是参数」。

问题：把 n-gram 做成参与梯度的网络结构，能否追上/超过非参数的 Mention-NDB？
核心对照臂是 `LearnableNGramTable`（lngtab）：键的构造与 `MentionNDB` **逐字相同**
（提及起始处 1~2 字字面 n-gram 的多项式哈希取模），但槽里存的不是 no_grad 的
按样本计数，而是一张**全局 `nn.Embedding(slots, num_classes)`**，被梯度更新。
于是「非参数写入 / 文档局部情节」这一个变量被单独隔离出来。

三个臂（都通过与 `MentionNDB` 相同的 `ndb=` 接口混入类别 logits）：

    arm        键 = 提及起始处的…                     值                    梯度
    ────────   ───────────────────────────────────   ───────────────────   ────
    lngtab     1~2 字前缀哈希（与 NDB 逐字相同）        全局可学习分布        表参与
    ngrammer   局部窗口（p-1,p,p+1）1~2 阶哈希嵌入      求和投影→类偏置       层参与
    nplm       提及前 4 字的字符嵌入拼接                前馈→类分布           层参与

三者都**没有写入**（`write()` 是 no-op）：全局参数无法承载「本文档内这个字面刚被指成谁」
这种随文档重置的状态 —— 这正是要检验的推论。

注入点说明：测量台（`src/dtseek/tasks/runtime.py` 的 `task_loss` / `evaluate_task`）只提供
`ndb.read(cls_logits, h, input_ids, attn)` 一个挂载点，`src/` 本次只读。所以 ngrammer 的
「局部 n-gram 特征层」也落在**类别 logits 混合**上，而不是解码器输入特征上。这一点在
README 里明写，不当作与原设想等价。
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LearnableNGramTable", "NGrammerLayer", "FeedForwardNPLM", "build_memory"]

_HASH_MULT = 1_000_003


def _hash_prefix(input_ids: torch.Tensor, pos: torch.Tensor, n: int) -> torch.Tensor:
    """与 `MentionNDB._keys` 逐字相同的「向前取 n 字」多项式滚动哈希。"""
    last = input_ids.shape[1] - 1
    key = input_ids.gather(1, pos.clamp(0, last)).to(torch.int64)
    for j in range(1, n):
        pj = (pos + j).clamp(0, last)
        key = key * _HASH_MULT + input_ids.gather(1, pj).to(torch.int64)
    return key


# ======================================================================
# 公共骨架：读注意力 / 门控混合 / 与 NDB 相同的对外接口
# ======================================================================
class _MemoryBase(nn.Module):
    """`MentionNDB` 的接口子集，供 `task_loss` / `evaluate_task` 无差别挂载。

    - `reset(B, device)`    : 只作废文档级键缓存（没有表要清）
    - `read_attention(...)` : 与 NDB **完全相同的 one-hot、detach 语义**
    - `write_enabled()`     : no-op 上下文（全局参数没有「写」这一步）
    - `write(...)`          : no-op，返回 None
    - `read(...)`           : 门控凸混合，返回混合后的 log 概率
    - `bypass`              : True 时 `read()` 原样返回 `log_softmax(cls_logits)`（旁路消融）
    """

    n_stats = 3          # [p_ng 最大概率, 归一化熵, 常数 1]
    n_levels = 0

    def __init__(self, hidden_dim: int, num_classes: int, n_levels: int,
                 extra_gate_in: int = 0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.n_levels = int(n_levels)
        self.bypass = False
        self._batch = 0
        self._flat: torch.Tensor | None = None
        self._flat_src: torch.Tensor | None = None
        self._last_attn: torch.Tensor | None = None
        self._last_pos: torch.Tensor | None = None
        self._last_gate_t: torch.Tensor | None = None
        # 读门控结构与 MentionNDB 同形（hidden + n_stats + 1 额外输入），bias 同样 -2.0
        self.read_gate = nn.Linear(self.hidden_dim + self.n_stats + int(extra_gate_in), 1)
        nn.init.constant_(self.read_gate.bias, -2.0)
        # 只有真正按级混合的臂（lngtab）才持有 level_weight，避免出现恒无梯度的死参数
        self.level_weight = (nn.Parameter(torch.zeros(n_levels)) if n_levels > 1 else None)

    # ---- 生命周期 ----
    def reset(self, batch_size: int, device) -> None:
        self._batch = int(batch_size)
        self._flat = None
        self._flat_src = None
        self._last_attn = self._last_pos = None

    def reset_stats(self) -> None:
        self._last_gate_t = None

    @contextlib.contextmanager
    def write_enabled(self):
        yield self

    @property
    def write_active(self) -> bool:
        return False

    def write(self, h, input_ids, starts, labels, valid) -> None:
        return None

    def table_gb(self, batch_size: int | None = None) -> float:
        return 0.0

    def extra_repr(self) -> str:
        return f"mem={type(self).__name__} classes={self.num_classes} params={self.n_params()}"

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- 读注意力：与 MentionNDB.read_attention 同语义 ----
    def read_attention(self, start_logits, mask, true_starts=None, hard: bool = True):
        B, L = start_logits.shape
        with torch.no_grad():
            if true_starts is not None:
                pos = true_starts.clamp(0, L - 1)
                a = F.one_hot(pos, L).to(start_logits.dtype)
            elif hard:
                pos = start_logits.argmax(-1)
                a = F.one_hot(pos, L).to(start_logits.dtype)
            else:
                pos = None
                a = F.softmax(start_logits.float(), dim=-1).to(start_logits.dtype)
            a = a * mask.to(a.dtype)
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
            self._last_attn, self._last_pos = a, pos
        return a

    def _pos_of(self, attn: torch.Tensor) -> torch.Tensor:
        """本次 `read()` 的位置：认得自己的 one-hot 就用缓存位置，否则回退 argmax。"""
        if attn is self._last_attn and self._last_pos is not None:
            return self._last_pos
        return attn.argmax(-1)

    def _is_onehot(self, attn: torch.Tensor) -> bool:
        return attn is self._last_attn and self._last_pos is not None

    # ---- 门控凸混合（与 NDB 同式） ----
    def _mix(self, cls_logits, h, p_ng):
        p_ng = p_ng.clamp_min(1e-9)
        p_ng = p_ng / p_ng.sum(-1, keepdim=True)
        maxp = p_ng.max(-1).values
        ent = -(p_ng * torch.log(p_ng.clamp_min(1e-9))).sum(-1) / math.log(self.num_classes)
        feat = torch.stack([maxp, ent, torch.ones_like(maxp)], dim=-1)
        g = torch.sigmoid(self.read_gate(torch.cat([h, feat], dim=-1)))
        p_model = F.softmax(cls_logits.float(), dim=-1)
        p_new = (1 - g) * p_model + g * p_ng
        p_new = p_new.clamp_min(1e-9)
        p_new = p_new / p_new.sum(-1, keepdim=True)
        with torch.no_grad():
            self._last_gate_t = g.mean()
        return torch.log(p_new).to(cls_logits.dtype)

    def read(self, cls_logits, h, input_ids, attn, targets=None):
        if self.bypass:
            return torch.log_softmax(cls_logits.float(), dim=-1).to(cls_logits.dtype)
        p_ng = self._p_ng(input_ids, attn)
        return self._mix(cls_logits, h, p_ng)

    def stats(self) -> dict:
        return {"mem": type(self).__name__, "n_params": self.n_params(),
                "last_gate": (float(self._last_gate_t) if self._last_gate_t is not None
                              else float("nan"))}


# ======================================================================
# 臂 1：可学习 n-gram 表（最关键的一条）
# ======================================================================
class LearnableNGramTable(_MemoryBase):
    """与 `MentionNDB` 键逐字相同，但槽里存**全局可学习分布**而非按样本的 no_grad 计数。

    每一级一张 `nn.Embedding(slots_li, num_classes)`（跨样本共享、被梯度更新）。
    读：按 start 指针（one-hot）取槽 → 取分布 → 按 `level_weight` 混合。
    「同一字面在 A 文档是人物1、B 文档是人物3」这种文档局部赋值，全局表在原理上
    只能学 id 的边缘分布 —— 这正是本臂要证伪/证实的推论。
    """

    def __init__(self, hidden_dim, num_classes, vocab_size=8192,
                 levels=(1, 2), slots=(8192, 4096)):
        super().__init__(hidden_dim, num_classes, n_levels=len(levels))
        self.levels = tuple(int(x) for x in levels)
        self.slots = [int(s) for s in (slots if not isinstance(slots, int)
                                       else [slots] * len(self.levels))]
        self.emb = nn.ModuleList([nn.Embedding(s, num_classes) for s in self.slots])
        for e in self.emb:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)

    def _ensure_flat(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, L, nlev] 每级的槽位（与 NDB 同构造，不含级间偏移 —— 各级各有一张表）。"""
        if self._flat is not None and self._flat_src is input_ids:
            return self._flat
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        cols = [torch.remainder(_hash_prefix(input_ids, pos, n), self.slots[li])
                for li, n in enumerate(self.levels)]
        self._flat = torch.stack(cols, dim=-1)
        self._flat_src = input_ids
        return self._flat

    def _level_probs(self, input_ids, idx):
        """idx [B, nlev] → 每级分布列表 + 门控特征。"""
        probs = []
        for li, e in enumerate(self.emb):
            logits = e(idx[:, li])                     # [B, C]
            probs.append(F.softmax(logits.float(), dim=-1))
        return probs

    def _p_ng(self, input_ids, attn):
        flat = self._ensure_flat(input_ids)
        B, L = input_ids.shape
        alpha = torch.softmax(self.level_weight, dim=0)
        if self._is_onehot(attn):
            pos = self._pos_of(attn)
            idx = flat.gather(1, pos.view(B, 1, 1).expand(B, 1, self.n_levels)).squeeze(1)
            probs = self._level_probs(input_ids, idx)
            p = alpha[0] * probs[0]
            for li in range(1, self.n_levels):
                p = p + alpha[li] * probs[li]
            return p
        # 软读回退：逐位置取分布再加权（数值上等价于 one-hot 路径的极限）
        p = torch.zeros(B, self.num_classes, device=input_ids.device)
        for li, e in enumerate(self.emb):
            logits = e(flat[:, :, li])                 # [B, L, C]
            pl = F.softmax(logits.float(), dim=-1)
            p = p + alpha[li] * (attn.unsqueeze(-1) * pl).sum(1)
        return p


# ======================================================================
# 臂 2：N-grammer 式局部 n-gram 特征层
# ======================================================================
class NGrammerLayer(_MemoryBase):
    """局部窗口多阶 n-gram 的 hashed embedding：求和 → 投影 → 类别 logits。

    窗口取提及起点附近的 `p-1, p, p+1`（含向后一个字，与 NDB 的「只向前取前缀」不同，
    这是本臂与 lngtab 的键差异，README 里明写）。阶数 1~2，两个阶各用一段哈希区间，
    共享一张 `nn.Embedding(hash_buckets, dim)`；每阶在窗口内取均值，两阶拼接后线性投影。
    """

    def __init__(self, hidden_dim, num_classes, embed_dim: int = 32,
                 slots=(8192, 4096), window: int = 1):
        super().__init__(hidden_dim, num_classes, n_levels=1)
        self.embed_dim = int(embed_dim)
        self.slots = [int(s) for s in slots]
        self.orders = (1, 2)
        self.window = int(window)
        self.hash_dim = sum(self.slots)
        self.emb = nn.Embedding(self.hash_dim, embed_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.proj = nn.Linear(embed_dim * len(self.orders), num_classes)
        self.offsets = [0, self.slots[0]]

    def _dist(self, input_ids, pos):
        """每阶在窗口内取 hashed embedding 的均值，两阶拼接后投影成类别分布。"""
        parts = []
        for oi, n in enumerate(self.orders):
            embs = []
            for d in range(-self.window, self.window + 1):
                p = (pos + d).clamp(0, input_ids.shape[1] - 1)
                slot = torch.remainder(_hash_prefix(input_ids, p, n),
                                       self.slots[oi]) + self.offsets[oi]
                embs.append(self.emb(slot))                       # [B, P, D]
            parts.append(torch.stack(embs, dim=0).mean(dim=0))    # 窗口内均值
        h_feat = torch.cat(parts, dim=-1)                         # [B, P, 2D]
        logits = self.proj(h_feat)
        return F.softmax(logits.float(), dim=-1)

    def _p_ng(self, input_ids, attn):
        B, L = input_ids.shape
        if self._is_onehot(attn):
            pos = self._pos_of(attn).view(B, 1)
            return self._dist(input_ids, pos).squeeze(1)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pl = self._dist(input_ids, pos)                        # [B, L, C]
        return (attn.unsqueeze(-1) * pl).sum(1)


# ======================================================================
# 臂 3：前馈神经 n-gram（NPLM）
# ======================================================================
class FeedForwardNPLM(_MemoryBase):
    """提及起点前 `ctx` 个字符的嵌入拼接 → 前馈 → 类别分布（Bengio 式 NPLM）。

    注意：只用**前文**（`p-ctx+1 .. p`），不含后文；与 NDB 的「向前取 n-gram」在取向上
    相反 —— NDB 取的是提及本身的首字/双字，这里取的是提及之前的上下文。
    """

    def __init__(self, hidden_dim, num_classes, vocab_size: int = 8192,
                 ctx: int = 4, embed_dim: int = 32, ff: int = 128):
        super().__init__(hidden_dim, num_classes, n_levels=1)
        self.ctx = int(ctx)
        self.vocab_size = int(vocab_size)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.fc1 = nn.Linear(ctx * embed_dim, ff)
        self.fc2 = nn.Linear(ff, num_classes)

    def _dist(self, input_ids, pos):
        B, P = pos.shape
        feats = []
        for j in range(self.ctx):
            pj = (pos - (self.ctx - 1 - j)).clamp(0, input_ids.shape[1] - 1)
            feats.append(self.emb(input_ids.gather(1, pj)))
        x = torch.cat(feats, dim=-1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x).float(), dim=-1)

    def _p_ng(self, input_ids, attn):
        B, L = input_ids.shape
        if self._is_onehot(attn):
            pos = self._pos_of(attn).view(B, 1)
            return self._dist(input_ids, pos).squeeze(1)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pl = self._dist(input_ids, pos)
        return (attn.unsqueeze(-1) * pl).sum(1)


def build_memory(arm: str, hidden_dim: int, num_classes: int, vocab_size: int = 8192):
    if arm == "lngtab":
        return LearnableNGramTable(hidden_dim, num_classes, vocab_size=vocab_size,
                                   levels=(1, 2), slots=(8192, 4096))
    if arm == "ngrammer":
        return NGrammerLayer(hidden_dim, num_classes)
    if arm == "nplm":
        return FeedForwardNPLM(hidden_dim, num_classes, vocab_size=vocab_size)
    raise ValueError(f"未知参数化臂 {arm!r}")
