"""参数化 / 可微记忆三臂 —— 与 `MentionNDB` **逐接口同构**，无需改 `src/`。

`runtime.task_loss` / `evaluate_task` / `_rollout` 只以鸭子类型调用记忆模块：
`reset` / `read_attention` / `read` / `write_enabled` / `write` / `parameters`。
所以只要实现这五个方法，就能替换 NDB，且训练循环一字不改。

三臂（外加一个 NDB 的「值进梯度」变体用于归因）：

| 臂 | 状态 | 键 | 值 | 梯度到 |
|---|---|---|---|---|
| `slots{S}` | 静态 `M ∈ R^{S×C}`（`nn.Parameter`） | 无（全局） | 无（本轮不写） | `M` + 注意力投影 |
| `diffwrite` | 每文档零初始化 `M ∈ R^{B×C}`（**非参数**） | **无键** | `σ(W_w h)·onehot(cls)` | `W_w`、读门控 |
| `fastweight` | 每文档零初始化 `M ∈ R^{B×C×dk}` | `W_k h`（学习投影） | `W_v h`（学习投影） | `W_k/W_q/W_v` |
| `fastweight_cls` | 同上 | `W_k h` | `onehot(cls) ⊗` 写权重（**值带 id**） | `W_k/W_q` |

`fastweight` 严格按任务书实现（键值都来自隐状态的线性投影）；但它**结构上无法存 id**
（值里没有类别信息），所以额外给一个唯一改动了「值换成 onehot(cls)」的 `fastweight_cls`，
作为参数化记忆的**最强候选**——若连它都输，结论才不是稻草人。

## 三个共同的硬约束

1. **参数初始化不消耗全局 RNG**：全部用局部 `torch.Generator(seed)` 初始化。
   这样本实验所有参数化臂与 `base` 臂**共享同一条全局 RNG 轨迹**（同样的
   DataLoader 洗牌顺序、同样的 dropout 抽样），配对性远强于与 `literal` 臂的对比
   （`literal` 构造 264 个门控参数时消耗了全局 RNG，与 base 的数据顺序本就不同）。
2. **评估期不写真值**：`write()` 只在 `write_enabled()` 里生效，而 `runtime` 只在
   训练循环和 `_rollout`（模型自己发射的切片）里打开它。
3. **读门控必须能被旁路**：`bypass=True` 时 `read()` 原样返回 `cls_logits`，
   用于「输出置零」消融。
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG1P_100 = math.log1p(100.0)


class _ParamMem(nn.Module):
    """公共骨架：局部 RNG 初始化、门控、旁路、诊断量的惰性物化。"""

    #: 子类覆盖：本模块的「状态」是否由 write() 逐文档累积（决定 covered 语义）
    episodic = False

    def __init__(self, hidden_dim: int, num_classes: int, seed: int = 0,
                 read_gate_bias: float = -2.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        # 局部生成器：**不动全局 RNG**（见模块 docstring 约束 1）
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed) * 2654435761 % (2**31))
        self._batch = 0
        self._write_on = False
        self.bypass = False
        self._read_gate_bias = read_gate_bias
        # 诊断（设备标量，stats() 里才物化）
        self._last_gate_t: torch.Tensor | None = None
        self._last_covered_t: torch.Tensor | None = None
        self._n_written_t: torch.Tensor | None = None

    # ---- 局部 RNG 下的参数工厂（不碰全局 RNG）----
    def _param(self, *shape, scale: float = 0.02, bias: bool = False) -> nn.Parameter:
        t = torch.empty(*shape, dtype=torch.float32)
        if bias:
            t.zero_()
        else:
            t.normal_(0.0, scale, generator=self._gen)
        return nn.Parameter(t)

    def _gate(self, in_dim: int, bias: float) -> nn.Linear:
        # ★ `nn.Linear.__init__` 会调 `reset_parameters()`，它消耗**全局** RNG。
        #   不把它围起来，参数化臂就会和 base 臂错开洗牌顺序（实测 base 指纹
        #   2226919 → 2366693）。构造期间存取全局 RNG 状态，等于这次构造不消耗全局 RNG。
        st = torch.get_rng_state()
        try:
            lin = nn.Linear(in_dim, 1)
        finally:
            torch.set_rng_state(st)
        with torch.no_grad():
            lin.weight.normal_(0.0, 0.02, generator=self._gen)
            lin.bias.fill_(bias)
        return lin

    # ---- 生命周期 ----
    def reset(self, batch_size: int, device) -> None:
        self._batch = int(batch_size)

    def reset_stats(self) -> None:
        self._last_gate_t = self._last_covered_t = self._n_written_t = None

    @contextlib.contextmanager
    def write_enabled(self):
        prev, self._write_on = self._write_on, True
        try:
            yield self
        finally:
            self._write_on = prev

    @property
    def write_active(self) -> bool:
        return self._write_on

    def read_attention(self, start_logits, mask, true_starts=None, hard: bool = True):
        """本实验三臂都不按位置检索（读由隐状态驱动），返回零权重占位。"""
        return torch.zeros_like(start_logits)

    # ---- 混入口径：与 NDB 一字不差 ----
    def _mix(self, cls_logits, h, p_mem, covered_feat, extra_feat=None):
        """`p = (1-g)·softmax(cls) + g·p_mem`，再取 log（与 `MentionNDB.read` 同式）。"""
        if self.bypass:
            return cls_logits
        dev = cls_logits.device
        B = cls_logits.shape[0]
        p_mem = p_mem / p_mem.sum(-1, keepdim=True).clamp_min(1e-9)
        covered = (covered_feat > 0).to(cls_logits.dtype).unsqueeze(-1)
        feats = [h, covered_feat.unsqueeze(-1)]
        if extra_feat is not None:
            feats.insert(1, extra_feat.unsqueeze(-1))
        g = torch.sigmoid(self.read_gate(torch.cat(feats, dim=-1)))
        g = g * covered
        p_model = F.softmax(cls_logits.float(), dim=-1)
        p_new = (1 - g) * p_model + g * p_mem.to(p_model.dtype)
        p_new = p_new.clamp_min(1e-9)
        p_new = p_new / p_new.sum(-1, keepdim=True)
        with torch.no_grad():
            self._last_gate_t = g.mean()
            self._last_covered_t = covered.mean()
        return torch.log(p_new).to(cls_logits.dtype)

    def stats(self) -> dict:
        f = lambda t: float("nan") if t is None else float(t.item())
        return {
            "batch": self._batch,
            "last_gate": f(self._last_gate_t),
            "last_covered": f(self._last_covered_t),
            "n_written": 0 if self._n_written_t is None else int(self._n_written_t.item()),
            "read_gate_bias": float(self.read_gate.bias.detach()),
            "bypass": self.bypass,
        }

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_dim} classes={self.num_classes}"


# ======================================================================
# 臂 1：slots —— 静态可学习记忆矩阵 M ∈ R^{S×C}
# ======================================================================
class SlotsMemory(_ParamMem):
    """`M ∈ R^{S×C}` 是 `nn.Parameter`；读 = 隐状态对 M 做注意力，混入类别 logits。

    这是「矩阵进梯度」的最小号形态：等价于一个**隐藏层宽 S、权重直接当类别 logits**
    的 2 层 MLP。它**没有键、也没有情节状态**：同一篇文档里第 1 次和第 5 次提及看到的
    是同一张 M，区别只来自隐状态 h。
    """

    def __init__(self, hidden_dim: int, num_classes: int, n_slots: int = 32, seed: int = 0):
        super().__init__(hidden_dim, num_classes, seed=seed)
        self.n_slots = int(n_slots)
        self.M = self._param(self.n_slots, num_classes, scale=0.02)          # 记忆矩阵
        self.W_k = self._param(self.n_slots, hidden_dim, scale=0.02)         # 注意力键
        self.read_gate = self._gate(hidden_dim + 1, -2.0)

    def write(self, h, input_ids, starts, labels, valid) -> None:
        return None                                                          # 静态：不写

    def read(self, cls_logits, h, input_ids, attn, targets=None):
        if self.bypass:
            return cls_logits
        a = F.softmax((h @ self.W_k.t()) / math.sqrt(self.hidden_dim), dim=-1)   # [B,S]
        p_mem = F.softmax(a @ self.M, dim=-1)                                    # [B,C]
        covered = torch.ones(h.shape[0], device=h.device)                        # 静态总有内容
        return self._mix(cls_logits, h, p_mem, covered)

    def extra_repr(self) -> str:
        return f"slots={self.n_slots} M={tuple(self.M.shape)} params=" \
               f"{sum(p.numel() for p in self.parameters())}"


# ======================================================================
# 臂 2：diffwrite —— 可微情节写（每文档零初始化 M，非参数）
# ======================================================================
class DiffWriteMemory(_ParamMem):
    """`M ∈ R^{B×C}` 逐文档零初始化（**非参数**），写 `M += σ(W_w h)·onehot(cls)`，读 = 归一化 M。

    NDB 的可微对应物，但**没有键**：M 只累积「本文档已发射过哪些类别」，是一个
    文档级类别直方图，与「哪个字面指向哪个 id」无关。梯度经 `σ(W_w h)` 回传到 `W_w`。
    """

    episodic = True

    def __init__(self, hidden_dim: int, num_classes: int, seed: int = 0):
        super().__init__(hidden_dim, num_classes, seed=seed)
        self.write_gate = self._gate(hidden_dim, 1.0)      # σ(1)≈0.73，同 NDB
        # _mix 的门控输入是 cat([h, log1p(total), covered]) = D+2
        self.read_gate = self._gate(hidden_dim + 2, -2.0)
        self.M: torch.Tensor | None = None

    def reset(self, batch_size: int, device) -> None:
        super().reset(batch_size, device)
        self.M = torch.zeros(batch_size, self.num_classes, device=device)   # 可微，不是 buffer

    def write(self, h, input_ids, starts, labels, valid) -> None:
        if not self._write_on or self.bypass:
            return None
        w = torch.sigmoid(self.write_gate(h)).squeeze(-1) * valid.float()
        ok = ((valid > 0.5) & (labels > 0)).to(w.dtype)
        oh = F.one_hot(labels.clamp(0, self.num_classes - 1), self.num_classes).to(w.dtype)
        self.M = self.M + (w * ok).unsqueeze(-1) * oh                       # 可微：梯度到 write_gate
        if self._n_written_t is None:
            self._n_written_t = torch.zeros((), dtype=torch.long, device=h.device)
        self._n_written_t = self._n_written_t + ok.sum()
        return None

    def read(self, cls_logits, h, input_ids, attn, targets=None):
        if self.bypass:
            return cls_logits
        s = self.M.sum(-1)                                                   # [B]
        feat = torch.log1p(s) / _LOG1P_100
        return self._mix(cls_logits, h, self.M, s, extra_feat=feat)

    def extra_repr(self) -> str:
        return f"episodic M[batch={self._batch},{self.num_classes}] params=" \
               f"{sum(p.numel() for p in self.parameters())}"


# ======================================================================
# 臂 3：fastweight —— 外积记忆 M = Σ_j v_j k_j^T
# ======================================================================
class FastWeightMemory(_ParamMem):
    """`M ∈ R^{B×C×dk}` 逐文档累积外积；`k = W_k h`、`v = W_v h`、读 `M q`（`q = W_q h`）。

    `value_from_cls=False`（默认）：严格按任务书 —— 键值都来自学习到的线性投影。
    **结构上无法存身份**（值里没有类别信息），只能存「隐状态的某种关联」。
    `value_from_cls=True`：唯一改动是把值换成 `onehot(cls)`，使记忆真的能存 id
    （写权重仍由 `σ(W_w h)` 给出，梯度到 `W_k/W_q/W_w`）。这是参数化记忆的最强候选。
    """

    episodic = True

    def __init__(self, hidden_dim: int, num_classes: int, seed: int = 0,
                 key_dim: int = 32, value_from_cls: bool = False):
        super().__init__(hidden_dim, num_classes, seed=seed)
        self.key_dim = int(key_dim)
        self.value_from_cls = bool(value_from_cls)
        self.W_k = self._param(self.key_dim, hidden_dim, scale=0.02)
        self.W_q = self._param(self.key_dim, hidden_dim, scale=0.02)
        self.W_v = self._param(num_classes, hidden_dim, scale=0.02)   # 仅 value_from_cls=False 用
        self.write_gate = self._gate(hidden_dim, 1.0)
        self.read_gate = self._gate(hidden_dim + 2, -2.0)
        self.M: torch.Tensor | None = None

    def reset(self, batch_size: int, device) -> None:
        super().reset(batch_size, device)
        self.M = torch.zeros(batch_size, self.num_classes, self.key_dim, device=device)

    def write(self, h, input_ids, starts, labels, valid) -> None:
        if not self._write_on or self.bypass:
            return None
        w = (torch.sigmoid(self.write_gate(h)).squeeze(-1) * valid.float())
        k = h @ self.W_k.t()                                                  # [B,dk]
        if self.value_from_cls:
            ok = ((valid > 0.5) & (labels > 0)).to(w.dtype)
            v = F.one_hot(labels.clamp(0, self.num_classes - 1),
                          self.num_classes).to(w.dtype)                       # [B,C]
            wv = w * ok
            if self._n_written_t is None:
                self._n_written_t = torch.zeros((), dtype=torch.long, device=h.device)
            self._n_written_t = self._n_written_t + ok.sum()
        else:
            v = h @ self.W_v.t()                                              # [B,C]
            wv = w
        self.M = self.M + wv.unsqueeze(-1).unsqueeze(-1) * (v.unsqueeze(-1) * k.unsqueeze(1))
        return None

    def read(self, cls_logits, h, input_ids, attn, targets=None):
        if self.bypass:
            return cls_logits
        q = h @ self.W_q.t()                                                  # [B,dk]
        z = (self.M * q.unsqueeze(1)).sum(-1)                                 # [B,C]
        p_mem = F.softmax(z, dim=-1)
        mass = self.M.flatten(1).norm(dim=-1)                                 # [B]
        feat = torch.log1p(mass) / _LOG1P_100
        covered = mass
        return self._mix(cls_logits, h, p_mem, covered, extra_feat=feat)

    def extra_repr(self) -> str:
        return (f"dk={self.key_dim} value_from_cls={self.value_from_cls} "
                f"params={sum(p.numel() for p in self.parameters())}")
